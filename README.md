# GridMind — Smart AI Microgrid Energy Scheduler

> **Team:** NECKSA
> **Theme:** Clean & Green Technology / Smart Automation
> **Hackathon:** Smart India Hackathon (SIH) 2026
> 
> 

---

## Overview

**GridMind** is an intelligent behind-the-meter energy dispatch platform designed for large commercial and industrial facilities (such as hospitals, university campuses, and airport hubs).

Solar-equipped facilities often face a costly duck-curve mismatch: they generate excess solar power at noon when it cannot be fully utilized, yet face steep Time-of-Day (ToD) tariff surcharges and maximum demand penalties when drawing grid power during evening peak hours.

GridMind uses 24-hour predictive lookaheads paired with Mixed-Integer Linear Programming (MILP) to automatically orchestrate battery energy storage systems (BESS), shiftable equipment, and EV charging fleets. It stores free daytime solar locally and discharges it during expensive utility peak hours, cutting electricity bills by 15% to 30% while keeping critical baseloads protected.

---

## Key Features

* **24-Hour Horizon Dispatch:** Evaluates daily solar yield, campus demand profiles, and utility tariff blocks to produce a 24-hour cost-optimal schedule.
* **Peak-Tariff Arbitrage:** Eliminates expensive 6:00 PM – 10:00 PM evening peak grid imports by discharging clean power stored on-site at midday.
* **Maximum Demand (kVA) Protection:** Automatically caps peak grid draw to avoid costly utility contract demand penalty surcharges.
* **Shielded Critical Loads:** Safeguards critical non-interruptible loads (e.g., hospital ICUs, servers) while intelligently adjusting elastic equipment (e.g., chillers, water pumps, EV chargers).
* **Dynamic Disturbance Adaptation:** Recalculates dispatch variables in real time to handle unexpected cloud cover, monsoon rain, or load spikes.
* **Interactive Operations Dashboard:** Built with a dark-mode UI that features real-time power balance curves, battery state-of-charge (SoC) tracking, and cost savings analytics.

---

## System Architecture

```
[Smart Meters / PV Inverters / BMS]
                 │
                 ▼ (MQTT / Modbus Telemetry)
       [Data Ingestion Layer]
                 │
                 ▼ (Historical Weather & Load Data)
     [XGBoost / LightGBM Forecasting]
                 │
                 ▼ (24h PV & Demand Profiles)
    [MILP Optimization Engine (PuLP / CBC)]
                 │
                 ▼ (Optimal Dispatch Setpoints)
    [FastAPI Execution & Web Dashboard (Streamlit)]

```

---

## Tech Stack

| Layer | Technologies |
| --- | --- |
| **Optimization Engine** | Python, PuLP, Coin-OR CBC Linear Programming Solver |
| **Data Analytics** | Pandas, NumPy |
| **Frontend & Telemetry** | Streamlit, Plotly Graph Objects (`make_subplots`) |
| **Protocols & IoT (Target)** | MQTT, Modbus TCP, TimescaleDB |
| **Forecasting Framework** | XGBoost, LightGBM (trained on NREL NSRDB & DOE building load profiles)

 |

---

## Mathematical Formulation

GridMind solves a constrained optimization problem across 24 hourly steps ($t = 0 \dots 23$):

### Objective Function

$$\min \sum_{t=0}^{23} \Big( P_{\text{grid, import}}(t) \cdot C_{\text{tariff}}(t) - P_{\text{grid, export}}(t) \cdot C_{\text{feed-in}} \Big)$$

### Constraints

1. **Power Balance Equality:**

$$P_{\text{solar}}(t) + P_{\text{grid, import}}(t) + P_{\text{discharge}}(t) = P_{\text{demand}}(t) + P_{\text{charge}}(t) + P_{\text{grid, export}}(t)$$


2. **Battery SoC Dynamics (with round-trip efficiency):**

$$\text{SoC}(t) = \text{SoC}(t-1) + \big(P_{\text{charge}}(t) \cdot \eta_{\text{ch}}\big) - \left(\frac{P_{\text{discharge}}(t)}{\eta_{\text{dis}}}\right)$$


3. **Operational Thresholds:**

$$0.15 \cdot \text{Capacity} \le \text{SoC}(t) \le 0.95 \cdot \text{Capacity}$$


$$0 \le P_{\text{charge}}(t) \le P_{\text{max, inverter}}, \quad 0 \le P_{\text{discharge}}(t) \le P_{\text{max, inverter}}$$



---

## Installation & Setup

### Prerequisites

* Python 3.9+
* Recommended: Clean virtual environment

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/gridmind.git
cd gridmind

```

### 2. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install streamlit pulp plotly pandas numpy

```

### 4. Run the Prototype

```bash
streamlit run app.py

```

*The interactive dashboard will launch at `http://localhost:8501`.*

---

## Prototype Controls & Testing Guide

* **Facility Profiles:** Switch between **500-Bed Super Specialty Hospital**, **Engineering University Campus**, and **Airport Terminal Hub** to test against different baseline load profiles.
* **BESS Hardware Sizing:** Adjust battery storage capacity ($kWh$) and inverter limits ($kW$) via sidebar sliders to see real-time ROI and savings adjustments.
* **Disturbance Simulation:** Check the **"Inject Heavy Cloud Cover"** toggle to simulate monsoon rain between 12:00 PM and 3:00 PM; verify that the solver re-optimizes battery reserves to safeguard evening operations.

---

## Project Roadmap

* [x] **Phase 1: Mathematical Model & Solver Validation** (Baseline formulation using PuLP/CBC against standard Time-of-Day tariffs)
* [x] **Phase 2: Interactive Operational Simulator** (Streamlit dark-mode dashboard with real-time disturbance testing)
* [ ] **Phase 3: Hardware-in-the-Loop Simulation** (Virtual Modbus TCP gateway connecting to simulated inverters and battery BMS)
* [ ] **Phase 4: Campus Pilot Testing** (Live deployment on institutional microgrid nodes with shiftable EV fleet charging)

---

## Team NECKSA

* **Project:** GridMind Microgrid Dispatch Engine
* **Submission Track:** Software
