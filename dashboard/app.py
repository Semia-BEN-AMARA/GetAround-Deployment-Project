import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="GetAround - Operational Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 GetAround: Late Checkout & Operational Impact Analysis")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
delay_path = os.path.join(BASE_DIR, "src", "get_around_delay_analysis.xlsx")
pricing_path = os.path.join(BASE_DIR, "src", "get_around_pricing_project.csv")

if not os.path.exists(delay_path):
    delay_path = "dashboard/src/get_around_delay_analysis.xlsx"
    pricing_path = "dashboard/src/get_around_pricing_project.csv"

@st.cache_data
def load_data():
    df_delay = pd.read_excel(delay_path, sheet_name="rentals_data")
    try:
        df_pricing = pd.read_csv(pricing_path)
        avg_price = df_pricing["rental_price_per_day"].mean()
    except:
        avg_price = 121.22  # Fallback baseline
    return df_delay, avg_price

df, AVG_DAILY_PRICE = load_data()

# ==========================================
# 2. SIMULATION UTILITY FUNCTION (Must be defined BEFORE usage)
# ==========================================
def simulate_impact(data_scoped, threshold_mins):
    total_scope_count = len(data_scoped)
    sub_consecutive = data_scoped[data_scoped['time_delta_with_previous_rental_in_minutes'].notna()].copy()
    
    sub_consecutive['is_prob'] = sub_consecutive.apply(
        lambda r: 1 if r['delay_at_checkout_in_minutes'] > r['time_delta_with_previous_rental_in_minutes'] else 0, axis=1
    )
    scope_problems_baseline = sub_consecutive['is_prob'].sum()
    
    affected = data_scoped[data_scoped['time_delta_with_previous_rental_in_minutes'] < threshold_mins]
    num_affected = len(affected)
    pct_affected = num_affected / total_scope_count if total_scope_count > 0 else 0
    
    est_loss = num_affected * AVG_DAILY_PRICE
    
    solved = data_scoped[
        (data_scoped['time_delta_with_previous_rental_in_minutes'] < threshold_mins) & 
        (data_scoped['delay_at_checkout_in_minutes'] > data_scoped['time_delta_with_previous_rental_in_minutes'])
    ]
    
    return num_affected, pct_affected * 100, int(est_loss), len(solved), scope_problems_baseline

# ==========================================
# 3. GLOBAL CONTROLS & SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🛠️ Global Dashboard Controls")

selected_scope = st.sidebar.selectbox(
    "1. Operational Fleet Scope:", 
    options=["all", "connect", "mobile"],
    index=0
)

selected_status = st.sidebar.selectbox(
    "2. Checkout Analysis Status Filter:",
    options=["All", "Only Completed (ended)", "Only Canceled (canceled)"],
    index=1
)

# Apply global scope filtering
if selected_scope == "connect":
    df_scoped = df[df['checkin_type'] == 'connect'].copy()
elif selected_scope == "mobile":
    df_scoped = df[df['checkin_type'] == 'mobile'].copy()
else:
    df_scoped = df.copy()

# Apply status filtering specifically for delay insights breakdown
if selected_status == "Only Completed (ended)":
    df_status_filtered = df_scoped[df_scoped['state'] == 'ended'].copy()
elif selected_status == "Only Canceled (canceled)":
    df_status_filtered = df_scoped[df_scoped['state'] == 'canceled'].copy()
else:
    df_status_filtered = df_scoped.copy()

# Macro variable calculations
total_rentals_count = len(df_scoped)
canceled_rentals_count = len(df_scoped[df_scoped['state'] == 'canceled'])
ended_rentals = df_scoped[df_scoped['state'] == 'ended']

consecutive_rentals = df_scoped[df_scoped['time_delta_with_previous_rental_in_minutes'].notna()].copy()
total_consecutive = len(consecutive_rentals)

# Baseline physical schedule overlaps
consecutive_rentals['is_problematic'] = consecutive_rentals.apply(
    lambda row: 1 if row['delay_at_checkout_in_minutes'] > row['time_delta_with_previous_rental_in_minutes'] else 0,
    axis=1
)
base_problematic_total = consecutive_rentals['is_problematic'].sum()

# ==========================================
# 4. MACRO FLEET ACTIVITY OVERVIEW
# ==========================================
st.markdown(f"## 📊 1. Macro Fleet Activity Overview (Scope: {selected_scope.upper()})")
macro_col1, macro_col2, macro_col3, macro_col4 = st.columns(4)

with macro_col1:
    st.metric(label="Total Logged Rentals", value=f"{total_rentals_count:,}")
with macro_col2:
    st.metric(label="Completed Rentals ('ended')", value=f"{len(ended_rentals):,}")
with macro_col3:
    st.metric(label="Global Cancellations", value=f"{canceled_rentals_count:,}", delta=f"{(canceled_rentals_count/total_rentals_count):.1%} Rate" if total_rentals_count > 0 else "0%", delta_color="inverse")
with macro_col4:
    st.metric(label="Consecutive Bookings (Risk Zone)", value=f"{total_consecutive:,}")

st.divider()

# ==========================================
# 5. FOCUS ON DELAY & CUSTOM SEUIL FILTER
# ==========================================
st.markdown(f"## 🔍 2. Focus: Checkout Delays and Customer Friction ({selected_status})")

critical_threshold_input = st.slider(
    "Define custom limit for a 'Critical Delay' (minutes):", 
    min_value=15, max_value=120, value=60, step=15
)

# Calculations matching the dynamically selected status filter
total_status_volume = len(df_status_filtered)
late_in_filtered_fleet = len(df_status_filtered[df_status_filtered['delay_at_checkout_in_minutes'] > 0])
fleet_late_rate = (late_in_filtered_fleet / total_status_volume) if total_status_volume > 0 else 0.0

consecutive_status_filtered = consecutive_rentals[consecutive_rentals['state'] == 'ended'] if selected_status == "Only Completed (ended)" else consecutive_rentals
total_consecutive_status = len(consecutive_status_filtered)
late_in_consecutive = len(consecutive_status_filtered[consecutive_status_filtered['delay_at_checkout_in_minutes'] > 0])
consecutive_late_rate = (late_in_consecutive / total_consecutive_status) if total_consecutive_status > 0 else 0.0

critical_late_count = len(df_status_filtered[df_status_filtered['delay_at_checkout_in_minutes'] > critical_threshold_input])
pct_critical = (critical_late_count / total_status_volume) if total_status_volume > 0 else 0.0

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    st.metric(label="Late Checkout Rate (Consecutive Chains)", value=f"{consecutive_late_rate:.1%}")
    st.caption("Risk Zone behavior (rentals with a back-to-back booking scheduled).")

with col_f2:
    st.metric(label="Global Late Rate (Entire Fleet Volume)", value=f"{fleet_late_rate:.1%}")
    st.caption("Overall late checkouts rate across all single or isolated rentals.")

with col_f3:
    st.metric(label="Total Problematic Cases", value=base_problematic_total)
    st.caption("Real scheduling overlaps recorded in historical logs.")

with col_f4:
    st.metric(label="Critical Delays (> {} mins)".format(critical_threshold_input), value=f"{pct_critical:.1%}")

st.info("""
ℹ *CRITERIA DEFINITION:* A **Total Problematic Case** represents a hard schedule collision. 
        It occurs precisely when a primary driver's checkout delay exceeds the scheduled buffer window, physically trapping the vehicle and disrupting the subsequent check-in.
""")

st.divider()

# ==========================================
# 6. INTERACTIVE POLICY SIMULATION
# ==========================================
st.markdown("## ⚙ 3. Policy Simulation: Threshold vs Revenue Loss")

selected_threshold = st.slider("Select Mandatory Minimum Time Window Feature (minutes):", min_value=0, max_value=240, value=120, step=30)

# Run simulation safely
blocked, pct_revenue, money_lost, solved_cases, scope_baseline = simulate_impact(df_scoped, selected_threshold)

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    st.metric(label="Blocked Rentals", value=blocked)
with sim_col2:
    st.metric(label="Potential Owner Revenue Lost (€)", value=f"- {money_lost:,} €", delta=f"{pct_revenue:.2f}% impacted", delta_color="inverse")
with sim_col3:
    st.metric(label="Problematic Cases (Solved / Total Baseline)", value=f"{solved_cases} / {scope_baseline}")

st.error("""
🚨 **ATTENTION:** A potential raise of revenue loss could occur if the blocked reservations are longer than 1 day!
         
**REQUIRED ACTION:** Please verify fleet utilization immediately and contact late drivers to protect high-impact booking chains.
""")

# ==========================================
# 7. VISUAL CHART (PLOTLY)
# ==========================================
st.markdown("### 📈 Visualizing the Strategic Dilemma")

curve_data = []
for t in range(0, 241, 30):
    b, p, m, s, sb = simulate_impact(df_scoped, t)
    curve_data.append({"Threshold": t, "Potential Owner Revenue Lost (€)": m, "Solved Problems": s})

df_curve = pd.DataFrame(curve_data)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_curve["Threshold"], y=df_curve["Potential Owner Revenue Lost (€)"],
    name="Potential Owner Revenue Lost (€)", line=dict(color='crimson', width=3)
))
fig.add_trace(go.Scatter(
    x=df_curve["Threshold"], y=df_curve["Solved Problems"],
    name="Solved Problems Count", line=dict(color='royalblue', width=3),
    yaxis="y2"
))

fig.update_layout(
    title=f"Optimization Curve for Scope: {selected_scope.upper()}",
    xaxis=dict(title=dict(text="Threshold Minimum Duration (Minutes)")),
    yaxis=dict(
        title=dict(text="Potential Owner Revenue Lost (€)", font=dict(color="crimson")),
        tickfont=dict(color="crimson")
    ),
    yaxis2=dict(
        title=dict(text="Customer Issues Solved", font=dict(color="royalblue")),
        tickfont=dict(color="royalblue"),
        overlaying="y",
        side="right"
    ),
    legend=dict(x=0.01, y=0.99),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)