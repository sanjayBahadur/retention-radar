"""Retention simulator page - Interactive Markov scenario modeling and ROI planning."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from retention_radar.markov.simulate import simulate_retention_scenario

st.set_page_config(page_title="Retention Simulator | Retention Radar", page_icon="🎛️", layout="wide")

st.title("🎛️ Interactive Retention Scenario Simulator")
st.markdown(
    "Simulate policy interventions on transition dynamics to forecast financial and retention impacts."
)

base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"


# Load base transition matrix and parameters
@st.cache_data
def load_base_simulator_data():
    p_path = base_dir / "markov_P.parquet"
    if p_path.exists():
        df_p = pd.read_parquet(p_path)
        pivot = df_p.pivot(index="from_state", columns="to_state", values="p").fillna(0.0)
        states = list(pivot.index)
        P = pivot.values
    else:
        states = ["New", "Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
        P = np.eye(6)

    # State rewards (gross margin per period)
    rewards = np.array([45.0, 160.0, 70.0, 0.0, 0.0, 0.0])
    # Baseline distribution
    init_dist = np.array([0.02, 0.20, 0.60, 0.10, 0.05, 0.03])
    init_dist = init_dist / init_dist.sum()

    return states, P, rewards, init_dist


states, P, rewards, init_dist = load_base_simulator_data()

# Sidebar: Intervention Controls
st.sidebar.header("🕹️ Intervention Parameters")

at_risk_lift = st.sidebar.slider(
    "Shift At-Risk -> Regular (+%):",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Targeted winback promotion moving At-Risk shoppers back to Regular buying frequency.",
)

dormant_lift = st.sidebar.slider(
    "Shift Dormant -> Regular (+%):",
    min_value=0.0,
    max_value=0.10,
    value=0.02,
    step=0.01,
    help="Reactivation offer for Dormant customers.",
)

offer_cost = st.sidebar.slider(
    "Cost per Targeted Offer ($):",
    min_value=0.50,
    max_value=10.00,
    value=2.00,
    step=0.50,
)

sim_horizon = st.sidebar.slider(
    "Planning Horizon (Periods):",
    min_value=4,
    max_value=26,
    value=13,
    step=1,
    help="Number of 28-day periods to project forward (~1 year = 13 periods).",
)

population = st.sidebar.selectbox(
    "Customer Base Scale:",
    [10_000, 50_000, 100_000, 500_000],
    index=2,
)

# Run simulation
intervention = {}
if at_risk_lift > 0:
    intervention["At-Risk"] = {"Regular": at_risk_lift}
if dormant_lift > 0:
    if "Dormant" not in intervention:
        intervention["Dormant"] = {}
    intervention["Dormant"]["Regular"] = dormant_lift

sim_res = simulate_retention_scenario(
    P=P,
    state_names=states,
    state_rewards=rewards,
    initial_distribution=init_dist,
    intervention=intervention if intervention else None,
    horizon_periods=sim_horizon,
    population=population,
)

# Financial calculation
targeted_share = 0.10 if at_risk_lift > 0 else 0.0
total_targeted_customers = population * targeted_share
total_campaign_cost = total_targeted_customers * offer_cost
delta_gross_margin = sim_res["delta_margin_dollars"]
net_profit = delta_gross_margin - total_campaign_cost
roi_ratio = (delta_gross_margin / total_campaign_cost) if total_campaign_cost > 0 else 0.0

# KPI Display Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Incremental Gross Margin",
        f"${delta_gross_margin:,.0f}",
        delta=f"+{delta_gross_margin / 1000:,.1f}k Margin",
    )

with kpi2:
    st.metric(
        "Net Incremental Profit",
        f"${net_profit:,.0f}",
        delta=f"{roi_ratio:.2f}x ROI",
        delta_color="normal" if net_profit >= 0 else "inverse",
    )

with kpi3:
    st.metric(
        "Lapsed Customers Prevented",
        f"{abs(sim_res['delta_lapsed_households']):,.0f}",
        delta="Retained Shoppers",
    )

with kpi4:
    st.metric(
        "Total Campaign Cost",
        f"${total_campaign_cost:,.0f}",
        delta=f"${offer_cost:.2f} / offer",
    )

st.write("")

# Visual comparison: Trajectory Projection
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Projected State Composition: Baseline vs Intervention")
    periods_arr = list(range(sim_horizon + 1))

    # Cumulative gross margin comparison
    cum_base = np.cumsum(
        [
            sim_res["baseline_trajectory"][p].dot(rewards) * population
            for p in range(sim_horizon + 1)
        ]
    )
    cum_intv = np.cumsum(
        [
            sim_res["counterfactual_trajectory"][p].dot(rewards) * population
            for p in range(sim_horizon + 1)
        ]
    )

    df_cum = pd.DataFrame(
        {
            "Period": periods_arr,
            "Baseline Cumulative Margin ($)": cum_base,
            "Intervention Cumulative Margin ($)": cum_intv,
        }
    )

    fig_cum = px.line(
        df_cum,
        x="Period",
        y=["Baseline Cumulative Margin ($)", "Intervention Cumulative Margin ($)"],
        labels={"value": "Cumulative Gross Margin ($)", "variable": "Policy"},
        title=f"Cumulative Portfolio Margin Over {sim_horizon} Periods",
        color_discrete_sequence=["#94A3B8", "#10B981"],
    )
    fig_cum.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_cum, use_container_width=True)

with col_right:
    st.subheader("Precomputed Scenario Frontier")
    scenarios_path = base_dir / "scenarios.parquet"
    if scenarios_path.exists():
        df_scenarios = pd.read_parquet(scenarios_path)
        fig_front = px.scatter(
            df_scenarios,
            x="at_risk_to_regular_shift",
            y="delta_margin_dollars",
            color="dormant_to_regular_shift",
            labels={
                "at_risk_to_regular_shift": "At-Risk Shift Lift",
                "delta_margin_dollars": "Delta Margin ($)",
                "dormant_to_regular_shift": "Dormant Lift",
            },
            title="Policy Response Surface Grid",
        )
        fig_front.update_layout(height=400)
        st.plotly_chart(fig_front, use_container_width=True)
    else:
        st.info("Precomputed scenarios grid not found.")
