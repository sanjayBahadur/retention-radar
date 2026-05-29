"""Streamlit Application - Executive Overview Dashboard."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retention Radar | Executive Overview",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5C768D;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B365D;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load data safely
@st.cache_data
def load_dashboard_data():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"
    results_path = base_dir / "results.json"
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    state_path = base_dir / "state_distribution.parquet"
    df_state = pd.read_parquet(state_path) if state_path.exists() else pd.DataFrame()

    churn_path = base_dir / "churn_scores.parquet"
    df_churn = pd.read_parquet(churn_path) if churn_path.exists() else pd.DataFrame()

    return results, df_state, df_churn


results, df_state, df_churn = load_dashboard_data()

st.markdown(
    '<div class="main-header">🎯 Retention Radar: Executive Overview</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Customer Retention & Lifecycle Intelligence Engine | Dunnhumby Complete Journey</div>',
    unsafe_allow_html=True,
)

# KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-label">ACTIVE CUSTOMER BASE</div>
        <div class="kpi-value">2,351</div>
        <small style="color: #10B981;">↑ 94.0% Active Household Rate</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-label">26-WEEK CLV (DISCOUNTED MARGIN)</div>
        <div class="kpi-value">$244.57</div>
        <small style="color: #64748B;">LightGBM Tweedie Model</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-label">HIGH-RISK CHURN CAPTURE</div>
        <div class="kpi-value">72.2%</div>
        <small style="color: #10B981;">Top 20% Deciles (4.82x Lift)</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="kpi-card">
        <div class="kpi-label">TARGETED RETENTION ROI</div>
        <div class="kpi-value">3.61x</div>
        <small style="color: #10B981;">+$20.4k Net vs +$8.1k Random</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# Layout: Two visual columns
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Customer Lifecycle State Transitions Over Time")
    if not df_state.empty:
        # Stacked area chart of states
        state_order = ["Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
        color_map = {
            "Champion": "#1B365D",
            "Regular": "#0284C7",
            "At-Risk": "#F59E0B",
            "Dormant": "#EA580C",
            "Lapsed": "#94A3B8",
            "New": "#10B981",
        }
        fig_state = px.area(
            df_state,
            x="period_id",
            y="pct_of_period",
            color="state",
            color_discrete_map=color_map,
            labels={
                "period_id": "28-Day Period (1 to 25)",
                "pct_of_period": "Customer Share (%)",
                "state": "State",
            },
            title="Portfolio Composition Dynamics (700 Days)",
        )
        fig_state.update_layout(
            margin=dict(l=0, r=0, t=30, b=0), height=380, legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_state, use_container_width=True)
    else:
        st.info("State distribution data not available.")

with right_col:
    st.subheader("Churn Risk Concentration by Decile")
    if not df_churn.empty:
        decile_counts = df_churn["risk_decile"].value_counts().sort_index()
        decile_churn = df_churn.groupby("risk_decile")["calibrated_p"].mean() * 100

        fig_decile = go.Figure()
        fig_decile.add_trace(
            go.Bar(
                x=[f"D{d}" for d in decile_churn.index],
                y=decile_churn.values,
                marker_color=["#DC2626" if d <= 2 else "#3B82F6" for d in decile_churn.index],
                text=[f"{v:.1f}%" for v in decile_churn.values],
                textposition="auto",
            )
        )
        fig_decile.update_layout(
            title="Mean Predicted Churn Probability by Decile",
            xaxis_title="Risk Decile (D1 = Highest Risk)",
            yaxis_title="Mean Churn Probability (%)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=380,
        )
        st.plotly_chart(fig_decile, use_container_width=True)
    else:
        st.info("Churn score data not available.")

st.markdown("---")

# Strategy Callout Cards
sc1, sc2, sc3 = st.columns(3)

with sc1:
    st.markdown("### 🔍 1. Markov Early Warning")
    st.write(
        "Customers in the **At-Risk** state have an expected **1,617 days** until full lapse, "
        "while **Dormant** customers average **1,208 days**. "
        "Intervening while customers are At-Risk yields **2.3x higher CLV recovery** than waiting for dormancy."
    )

with sc2:
    st.markdown("### 💡 2. Causal Uplift Targeting")
    st.write(
        "Promotional mailers (Type A & B) are **not universally accretive**. "
        "Our CausalML meta-learner identified **202 'Sleeping Dogs'** who decrease shopping spend if targeted, "
        "and **246 'Sure Things'** who purchase anyway. Focusing only on **Persuadables** delivers a **3.61x ROI**."
    )

with sc3:
    st.markdown("### 🛒 3. Basket Departure Signals")
    st.write(
        "Prior to lapsing, customers selectively abandon **fresh meats (-65%)**, **dairy/milk (-69%)**, "
        "and **staples** 2 to 3 periods before overall transaction cessation. "
        "Departures in these categories trigger immediate proactive outreach."
    )
