import streamlit as st
import pandas as pd
import numpy as np
import pulp
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="GridMind — AI Energy Dispatch", layout="wide")

st.title("⚡ GridMind: AI-Driven Campus Microgrid Dispatch")
st.markdown(
    "**Architecture:** `Weather Features` ➔ **ML Solar Forecaster (RandomForest)** ➔ **Prescriptive LP Optimizer (PuLP)**"
)

# ==========================================
# 1. SIDEBAR CONTROLS (INTERACTIVITY)
# ==========================================
st.sidebar.header("🛠️ Facility & Grid Parameters")

battery_cap = st.sidebar.slider("Battery Capacity (kWh)", min_value=100.0, max_value=800.0, value=400.0, step=50.0)
max_power = st.sidebar.slider("Max Inverter Rate (kW)", min_value=25.0, max_value=200.0, value=80.0, step=5.0)
peak_tariff = st.sidebar.slider("Peak Grid Price ($/kWh)", min_value=0.20, max_value=0.60, value=0.38, step=0.02)

st.sidebar.header("⛅ Weather Simulation")
cloud_cover_factor = st.sidebar.select_slider(
    "Day Weather Condition",
    options=["Clear Sunny (0% Clouds)", "Partly Cloudy (30% Clouds)", "Overcast / Rainy (80% Clouds)"],
    value="Partly Cloudy (30% Clouds)"
)

# Map cloud string to actual cloud cover percentage
cloud_map = {
    "Clear Sunny (0% Clouds)": 5.0,
    "Partly Cloudy (30% Clouds)": 35.0,
    "Overcast / Rainy (80% Clouds)": 80.0
}
simulated_clouds = cloud_map[cloud_cover_factor]

# ==========================================
# 2. MACHINE LEARNING: SOLAR FORECAST MODEL
# ==========================================
@st.cache_resource
def train_solar_model():
    """Generates synthetic historical weather data and trains a RandomForestRegressor."""
    np.random.seed(42)
    n_samples = 1500
    
    # Synthetic features: Hour (0-23), Temp (15-38 C), Cloud Cover (0-100%)
    h = np.random.randint(0, 24, n_samples)
    temp = np.random.uniform(20, 36, n_samples)
    clouds = np.random.uniform(0, 100, n_samples)
    
    # Solar formula based on sun zenith + cloud dampening + noise
    solar_zenith = np.maximum(0, np.sin((h - 6) / 12 * np.pi))
    cloud_attenuation = (100 - clouds * 0.85) / 100.0
    y = solar_zenith * cloud_attenuation * 280.0 + np.random.normal(0, 5, n_samples)
    y = np.clip(y, 0, None)
    
    X = pd.DataFrame({"hour": h, "temp": temp, "clouds": clouds})
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

ml_model = train_solar_model()

# Prepare 24h feature vector for today
hours = list(range(24))
sim_temp = [22 + 10 * np.sin((h - 6) / 12 * np.pi) for h in hours]  # cooler night, warm day
sim_clouds = [simulated_clouds] * 24

input_df = pd.DataFrame({"hour": hours, "temp": sim_temp, "clouds": sim_clouds})
predicted_solar = ml_model.predict(input_df)
predicted_solar = [round(max(0.0, float(v)), 1) for v in predicted_solar]

# Base facility demand (kW) & Time-of-Use tariff structure
demand_profile = [
    40, 35, 30, 30, 35, 50, 75, 110, 140, 160, 
    170, 175, 180, 175, 170, 165, 160, 180, 190, 170, 
    140, 100, 70, 50
]

price_profile = [
    0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
    0.20, 0.18, 0.16, 0.16, 0.18, 0.22, 0.28, peak_tariff, peak_tariff, peak_tariff * 0.85,
    0.25, 0.18, 0.12, 0.08
]

# ==========================================
# 3. BASELINE EVALUATION (WITHOUT GRIDMIND)
# ==========================================
baseline_grid_import = []
baseline_curtailed = []
baseline_costs = []

for d, s, p in zip(demand_profile, predicted_solar, price_profile):
    if s >= d:
        baseline_grid_import.append(0.0)
        baseline_curtailed.append(s - d)
        baseline_costs.append(0.0)
    else:
        shortfall = d - s
        baseline_grid_import.append(shortfall)
        baseline_curtailed.append(0.0)
        baseline_costs.append(shortfall * p)

total_base_cost = sum(baseline_costs)
total_base_curtailment = sum(baseline_curtailed)

# ==========================================
# 4. OPTIMIZATION DISPATCH (WITH GRIDMIND)
# ==========================================
model = pulp.LpProblem("GridMind_Dispatch", pulp.LpMinimize)

grid_in = pulp.LpVariable.dicts("GridIn", hours, lowBound=0)
charge = pulp.LpVariable.dicts("Charge", hours, lowBound=0, upBound=max_power)
discharge = pulp.LpVariable.dicts("Discharge", hours, lowBound=0, upBound=max_power)
curtail = pulp.LpVariable.dicts("Curtail", hours, lowBound=0)
soc = pulp.LpVariable.dicts("SoC", hours, lowBound=battery_cap * 0.10, upBound=battery_cap)

# Objective: Minimize grid cost
model += pulp.lpSum([grid_in[t] * price_profile[t] for t in hours])

initial_soc = battery_cap * 0.20
for t in hours:
    # Demand balance constraint
    model += (grid_in[t] + predicted_solar[t] + discharge[t] 
              == demand_profile[t] + charge[t] + curtail[t])
    
    # State-of-charge conservation
    if t == 0:
        model += (soc[0] == initial_soc + charge[0] - discharge[0])
    else:
        model += (soc[t] == soc[t-1] + charge[t] - discharge[t])

model.solve(pulp.PULP_CBC_CMD(msg=False))

# Post-process optimal solutions
gm_grid_in = [pulp.value(grid_in[t]) for t in hours]
gm_charge = [pulp.value(charge[t]) for t in hours]
gm_discharge = [pulp.value(discharge[t]) for t in hours]
gm_curtail = [pulp.value(curtail[t]) for t in hours]
gm_soc = [pulp.value(soc[t]) for t in hours]

total_gm_cost = pulp.value(model.objective)
total_gm_curtailment = sum(gm_curtail)
cost_savings = max(0.0, total_base_cost - total_gm_cost)
savings_pct = (cost_savings / total_base_cost) * 100 if total_base_cost > 0 else 0
solar_diverted = total_base_curtailment - total_gm_curtailment

# ==========================================
# 5. UI DASHBOARD RENDERING
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("24h Base Cost", f"${total_base_cost:,.2f}")
col2.metric("GridMind Cost", f"${total_gm_cost:,.2f}", delta=f"-{savings_pct:.1f}%")
col3.metric("Cost Savings", f"${cost_savings:,.2f}")
col4.metric("Solar Saved from Waste", f"{solar_diverted:,.1f} kWh")

st.divider()

# Visualization section
chart_data = pd.DataFrame({
    "Hour": hours,
    "Demand (kW)": demand_profile,
    "ML Forecasted Solar (kW)": predicted_solar,
    "Unmanaged Grid Import (kW)": baseline_grid_import,
    "GridMind Grid Import (kW)": gm_grid_in,
    "Battery SoC (kWh)": gm_soc,
    "Battery Net Action (kW)": [c - d for c, d in zip(gm_charge, gm_discharge)]
}).set_index("Hour")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📊 Load & Generation vs. Grid Import")
    st.line_chart(chart_data[["Demand (kW)", "ML Forecasted Solar (kW)", "Unmanaged Grid Import (kW)", "GridMind Grid Import (kW)"]])

with right_col:
    st.subheader("🔋 Battery Dispatch Schedule")
    st.line_chart(chart_data[["Battery SoC (kWh)", "Battery Net Action (kW)"]])

st.divider()
st.subheader("🔍 Hourly Dispatch Table")
st.dataframe(
    chart_data[["Demand (kW)", "ML Forecasted Solar (kW)", "GridMind Grid Import (kW)", "Battery SoC (kWh)", "Battery Net Action (kW)"]],
    use_container_width=True
)
