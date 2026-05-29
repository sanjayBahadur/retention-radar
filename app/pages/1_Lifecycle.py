"""Lifecycle page - Markov transition matrix, absorbing fundamentals, and survival curves."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Markov Lifecycle Dynamics | Retention Radar", page_icon="🔄", layout="wide"
)

st.title("🔄 Customer Lifecycle Dynamics & Markov Modeling")
st.markdown("Mathematical modeling of customer health progression across 28-day periods.")

base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"

# 1. Transition Matrix
st.subheader("1. Empirical 1-Step Transition Probability Matrix P")
trans_path = base_dir / "transition_matrix.parquet"
p_ci_path = base_dir / "markov_P.parquet"

if p_ci_path.exists():
    df_p = pd.read_parquet(p_ci_path)
    # Pivot to square matrix
    pivot_p = df_p.pivot(index="from_state", columns="to_state", values="p").fillna(0.0)

    fig_heat = px.imshow(
        pivot_p,
        labels=dict(x="Destination State (t+1)", y="Source State (t)", color="Probability"),
        x=pivot_p.columns,
        y=pivot_p.index,
        color_continuous_scale="Blues",
        text_auto=".3f",
        title="Markov State Transition Matrix (Smoothed MLE with 95% Bootstrap CIs)",
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Transition matrix artifact not found.")

st.markdown("---")

# 2. Expected Days to Lapse & Fundamental Matrix
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Expected Days to Full Lapse")
    metrics_path = base_dir / "markov_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            mm = json.load(f)
            lapse_dict = mm.get("expected_days_to_lapse", {})
            if lapse_dict:
                df_lapse = (
                    pd.DataFrame(lapse_dict).T.reset_index().rename(columns={"index": "state"})
                )
                fig_lapse = go.Figure()
                fig_lapse.add_trace(
                    go.Bar(
                        x=df_lapse["state"],
                        y=df_lapse["mean_days"],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=df_lapse["ci_upper_95"] - df_lapse["mean_days"],
                            arrayminus=df_lapse["mean_days"] - df_lapse["ci_lower_95"],
                        ),
                        marker_color=["#10B981", "#1B365D", "#0284C7", "#F59E0B", "#EA580C"],
                        text=[f"{v:.0f}d" for v in df_lapse["mean_days"]],
                        textposition="auto",
                    )
                )
                fig_lapse.update_layout(
                    xaxis_title="Current Customer State",
                    yaxis_title="Expected Days until Lapsed",
                    title="Absorbing Chain Fundamental Matrix Days-to-Absorption",
                    height=380,
                )
                st.plotly_chart(fig_lapse, use_container_width=True)
    else:
        st.info("Markov metrics not found.")

with col2:
    st.subheader("3. Markov Reward CLV by State")
    if metrics_path.exists():
        with open(metrics_path) as f:
            mm = json.load(f)
            clv_dict = mm.get("markov_reward_clv", {})
            if clv_dict:
                df_rclv = pd.DataFrame(list(clv_dict.items()), columns=["state", "reward_clv"])
                fig_rclv = px.bar(
                    df_rclv,
                    x="state",
                    y="reward_clv",
                    color="state",
                    color_discrete_sequence=[
                        "#10B981",
                        "#1B365D",
                        "#0284C7",
                        "#F59E0B",
                        "#EA580C",
                        "#94A3B8",
                    ],
                    text_auto="$.2f",
                    title="Infinite-Horizon Discounted Expected Margin ($)",
                    labels={"state": "Starting State", "reward_clv": "Reward CLV ($)"},
                )
                fig_rclv.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_rclv, use_container_width=True)

st.markdown("---")

# 3. Kaplan-Meier Survival Curves
st.subheader("4. Empirical Kaplan-Meier Survival Curves")
surv_path = base_dir / "survival_curves.parquet"
if surv_path.exists():
    df_surv = pd.read_parquet(surv_path)
    fig_surv = px.line(
        df_surv,
        x="period",
        y="survival_probability",
        color="starting_state",
        labels={
            "period": "Duration (28-day periods)",
            "survival_probability": "Retention Probability (Not Lapsed)",
        },
        title="Time-to-Lapsed Survival Curves Stratified by Baseline State",
    )
    fig_surv.update_layout(height=400)
    st.plotly_chart(fig_surv, use_container_width=True)
else:
    st.info("Survival curves artifact not found.")
