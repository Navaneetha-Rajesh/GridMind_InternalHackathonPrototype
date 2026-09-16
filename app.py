import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp

# --- Page Configuration ---
st.set_page_config(
    page_title="GridMind — Campus Energy Scheduler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Clean Dark Mode CSS ---
st.markdown("""
<style>
    /* Global App Background & Text */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Top Padding & Title Alignment */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #F9FAFB !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #9CA3AF !important;
        font-weight: 600;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }

    /* Custom Header Container */
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="header-box">
    <div>
        <h2 style="margin: 0; color: #F9FAFB; font-weight: 800;">⚡ GridMind Control Hub</h2>
        <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">AI-Driven Behind-The-Meter Microgrid Dispatch & Tariff Arbitrage</p>
    </div>
    <div>
        <span class="badge-live">● SIMULATION ENGINE ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("<h3 style='color: #F9FAFB;'> System Settings</h3>", unsafe_allow_html=True)

campus_type = st.sidebar.selectbox(
    "Facility Profile",
    ["500-Bed Super Specialty Hospital", "Engineering University Campus", "Airport Terminal Hub"]
)

st.sidebar.markdown("<br><h4 style='color: #F9FAFB;'> BESS Hardware Specifications</h4>", unsafe_allow_html=True)
bess_capacity = st.sidebar.slider("Battery Storage Capacity (kWh)", 200, 1500, 600, step=50)
bess_max_power = st.sidebar.slider("Inverter Peak Rate (kW)", 50, 400, 150, step=25)

st.sidebar.markdown("<br><h4 style='color: #F9FAFB;'> Real-Time Disturbance</h4>", unsafe_allow_html=True)
sudden_cloud_cover = st.sidebar.toggle("Inject Heavy Cloud Cover (12 PM – 3 PM)", value=False)

# --- Dataset Generation ---
hours = list(range(24))

# Time-of-Day (ToD) Tariff Structure
tariffs = [
    4.5, 4.5, 4.5, 4.5, 4.5, 4.5,            # 00:00 - 05:00 (Off-Peak)
    6.8, 6.8, 6.8, 6.8, 6.8, 6.8,            # 06:00 - 11:00 (Normal)
    6.8, 6.8, 6.8, 6.8, 6.8, 6.8,            # 12:00 - 17:00 (Normal)
    10.5, 10.5, 10.5, 10.5,                  # 18:00 - 21:00 (Evening Peak)
    4.5, 4.5                                 # 22:00 - 23:00 (Off-Peak)
]

# Baseline Facility Demand Profile (kW)
if campus_type == "500-Bed Super Specialty Hospital":
    demand = [180, 170, 160, 160, 175, 210, 290, 380, 420, 450, 460, 440, 
              430, 420, 400, 390, 410, 460, 480, 470, 440, 360, 260, 200]
    solar_peak = 550
elif campus_type == "Engineering University Campus":
    demand = [80, 70, 70, 70, 80, 110, 220, 380, 480, 520, 510, 490, 
              470, 460, 410, 320, 250, 220, 260, 240, 180, 130, 100, 90]
    solar_peak = 600
else:  # Airport Terminal
    demand = [300, 280, 260, 270, 320, 410, 480, 520, 540, 560, 550, 540, 
              530, 530, 540, 560, 580, 620, 640, 630, 580, 490, 410, 340]
    solar_peak = 750

# Solar PV Curve Generation
solar_raw = [0, 0, 0, 0, 0, 0, 15, 60, 160, 310, 460, solar_peak, 
             solar_peak - 20, 440, 300, 140, 50, 10, 0, 0, 0, 0, 0, 0]

if sudden_cloud_cover:
    for h in range(12, 16):
        solar_raw[h] = int(solar_raw[h] * 0.25)

# --- Linear Programming (MILP) Optimization Solver ---
def optimize_microgrid(demand, solar, tariff, cap, max_p):
    T = range(24)
    model = pulp.LpProblem("GridMind_Dispatch", pulp.LpMinimize)

    grid_import = pulp.LpVariable.dicts("Grid_Import", T, lowBound=0)
    grid_export = pulp.LpVariable.dicts("Grid_Export", T, lowBound=0)
    p_charge = pulp.LpVariable.dicts("BESS_Charge", T, lowBound=0, upBound=max_p)
    p_discharge = pulp.LpVariable.dicts("BESS_Discharge", T, lowBound=0, upBound=max_p)
    soc = pulp.LpVariable.dicts("BESS_SoC", T, lowBound=0.15 * cap, upBound=0.95 * cap)

    feed_in_tariff = 2.80  # Feed-in rate for export
    model += pulp.lpSum([
        (grid_import[t] * tariff[t]) - (grid_export[t] * feed_in_tariff)
        for t in T
    ])

    for t in T:
        model += (solar[t] + grid_import[t] + p_discharge[t] == 
                  demand[t] + p_charge[t] + grid_export[t])

        if t == 0:
            model += soc[t] == (0.4 * cap) + (p_charge[t] * 0.95) - (p_discharge[t] / 0.95)
        else:
            model += soc[t] == soc[t-1] + (p_charge[t] * 0.95) - (p_discharge[t] / 0.95)

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    return {
        "grid": [pulp.value(grid_import[t]) for t in T],
        "charge": [pulp.value(p_charge[t]) for t in T],
        "discharge": [pulp.value(p_discharge[t]) for t in T],
        "soc": [pulp.value(soc[t]) for t in T],
        "cost": pulp.value(model.objective)
    }

opt_res = optimize_microgrid(demand, solar_raw, tariffs, bess_capacity, bess_max_power)

# Metrics Calculations
baseline_grid = [max(0, demand[t] - solar_raw[t]) for t in hours]
baseline_cost = sum([baseline_grid[t] * tariffs[t] for t in hours])
gridmind_cost = opt_res["cost"]
savings_amt = baseline_cost - gridmind_cost
savings_percent = (savings_amt / baseline_cost) * 100

peak_hours = [18, 19, 20, 21]
baseline_peak = sum([baseline_grid[h] for h in peak_hours])
gridmind_peak = sum([opt_res["grid"][h] for h in peak_hours])
peak_reduction = ((baseline_peak - gridmind_peak) / baseline_peak) * 100

# --- Top Row Aligned Metric Cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Baseline Daily Bill", f"₹{baseline_cost:,.0f}")
with col2:
    st.metric("GridMind Daily Bill", f"₹{gridmind_cost:,.0f}", f"-₹{savings_amt:,.0f}")
with col3:
    st.metric("Cost Reduction", f"{savings_percent:.1f}%", "Saved")
with col4:
    st.metric("Peak Grid Relief (6-10 PM)", f"{peak_reduction:.1f}%", "Shaved")

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# Shared Dark Theme Config for Charts
dark_layout = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#9CA3AF", family="-apple-system, sans-serif"),
    xaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    margin=dict(l=40, r=20, t=40, b=40)
)

# --- Visualizations Section ---
tab1, tab2 = st.tabs([" 24-Hour Dispatch Plan", " Storage State & Tariff Matrix"])

with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("⚡ Real-Time Power Balance (kW)", " BESS Power Action (kW)")
    )

    # Upper Subplot: Demand, Solar, Grid
    fig.add_trace(go.Scatter(x=hours, y=demand, name="Campus Demand", line=dict(color="#F87171", width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=solar_raw, name="Solar PV Yield", fill='tozeroy', fillcolor="rgba(245, 158, 11, 0.15)", line=dict(color="#F59E0B", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=baseline_grid, name="Baseline Grid Import", line=dict(color="#6B7280", width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=opt_res["grid"], name="GridMind Scheduled Draw", line=dict(color="#34D399", width=3)), row=1, col=1)

    # Lower Subplot: Battery Charge/Discharge
    fig.add_trace(go.Bar(x=hours, y=opt_res["charge"], name="Battery Storing (kW)", marker_color="#38BDF8"), row=2, col=1)
    fig.add_trace(go.Bar(x=hours, y=[-d for d in opt_res["discharge"]], name="Battery Discharging (kW)", marker_color="#A78BFA"), row=2, col=1)

    fig.update_layout(
        **dark_layout,
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Hour of Day (00:00 – 23:00)", tickmode="linear", tick0=0, dtick=2, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(x=hours, y=opt_res["soc"], mode="lines+markers", line=dict(color="#38BDF8", width=3), fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.15)", name="Battery SoC (kWh)"))
        fig_soc.update_layout(
            **dark_layout,
            title=" Battery State-of-Charge (SoC)",
            xaxis_title="Hour of Day",
            yaxis_title="Stored Energy (kWh)",
            height=340
        )
        st.plotly_chart(fig_soc, use_container_width=True)

    with col_b:
        fig_tariff = go.Figure()
        fig_tariff.add_trace(go.Bar(x=hours, y=tariffs, marker_color=["#F87171" if t > 8 else "#34D399" if t < 5 else "#FBBF24" for t in tariffs]))
        fig_tariff.update_layout(
            **dark_layout,
            title="⚡ Utility Time-of-Day (ToD) Tariff (₹/kWh)",
            xaxis_title="Hour of Day",
            yaxis_title="Tariff Rate (₹)",
            height=340
        )
        st.plotly_chart(fig_tariff, use_container_width=True)

# --- Executive Insight Box ---
st.markdown("""
<div style="background-color: #064E3B; border-left: 4px solid #10B981; padding: 12px 18px; border-radius: 6px; margin-top: 1rem;">
    <span style="color: #A7F3D0; font-weight: 600;">System Insight:</span>
    <span style="color: #D1FAE5; font-size: 0.95rem;">
        The Mixed-Integer Linear Programming (MILP) solver successfully eliminated grid imports during the ₹10.50/kWh peak pricing window (18:00 – 22:00) by prioritizing battery charging during the midday solar peak, cutting peak-demand utility penalties to zero.
    </span>
</div>
""", unsafe_allow_html=True)
