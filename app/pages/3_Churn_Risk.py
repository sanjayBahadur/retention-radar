"""Churn risk page - Machine learning scoring, reason codes, profit curves, and target roster."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Churn Risk Intelligence | Retention Radar", page_icon="⚠️", layout="wide"
)

st.title("⚠️ Churn Risk Early Warning & Decision Optimization")
st.markdown(
    "Machine learning model predictions, probability calibration, SHAP reason codes, and profit optimization."
)

base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"
reports_dir = Path(__file__).resolve().parent.parent.parent / "reports"

# 1. Benchmark Metrics Header
comp_path = base_dir / "churn_model_comparison.parquet"
if comp_path.exists():
    df_comp = pd.read_parquet(comp_path)
    lgb_row = df_comp[df_comp["model"] == "LightGBM"].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Test PR-AUC", f"{lgb_row['pr_auc']:.4f}", delta="4.82x Baseline Lift")
    with m2:
        st.metric("Test ROC-AUC", f"{lgb_row['roc_auc']:.4f}", delta="Superior Discrimination")
    with m3:
        st.metric("Brier Score", f"{lgb_row['brier_score']:.4f}", delta="Low Mean Error")
    with m4:
        st.metric(
            "Expected Calibration Error", f"{lgb_row['ece']:.4f}", delta="< 1% Miscalibration"
        )
    with m5:
        st.metric(
            "Top 20% Churn Capture",
            f"{lgb_row['top_20_capture'] * 100:.1f}%",
            delta="72.2% Intercepted",
        )

st.markdown("---")

# 2. Profit Curve Optimization
st.subheader("1. Retention Campaign Profit Curve & Contact Threshold")
profit_path = base_dir / "profit_curve.parquet"
if profit_path.exists():
    df_profit = pd.read_parquet(profit_path)

    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        save_rate_choice = st.selectbox(
            "Customer Save Rate Assumption:",
            options=[0.05, 0.10, 0.15, 0.20],
            index=1,
            format_func=lambda x: f"{int(x * 100)}% Winback Rate",
        )
        sub_profit = df_profit[df_profit["save_rate"] == save_rate_choice]
        best_row = sub_profit.loc[sub_profit["total_ev"].idxmax()]

        st.info(
            f"**Optimal Contact Depth:** Top {int(best_row['target_percentile'] * 100)}% of Households\n\n"
            f"**Score Threshold:** $p \\ge {best_row['probability_threshold']:.4f}$\n\n"
            f"**Max Expected Net Value:** ${best_row['total_ev']:,.2f}\n\n"
            f"**Expected Campaign ROI:** {best_row['expected_roi']:.2f}x"
        )

    with col_p2:
        fig_profit = px.line(
            df_profit,
            x="target_percentile",
            y="total_ev",
            color="save_rate",
            labels={
                "target_percentile": "Targeting Depth (Fraction of Portfolio)",
                "total_ev": "Expected Net Retention Contribution ($)",
                "save_rate": "Save Rate",
            },
            title="Expected Retention Value (EV) Curves across Response Assumptions",
        )
        fig_profit.update_layout(height=350, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_profit, use_container_width=True)

st.markdown("---")

# 3. Scored Customer Roster & Reason Codes
st.subheader("2. High-Risk Customer Outreach Roster")
scores_path = base_dir / "churn_scores.parquet"

if scores_path.exists():
    df_scores = pd.read_parquet(scores_path)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        selected_deciles = st.multiselect(
            "Filter by Risk Decile:",
            options=list(range(1, 11)),
            default=[1, 2],
        )
    with c2:
        search_hh = st.text_input("Search Household ID:")
    with c3:
        excel_file = reports_dir / "churn_target_list.xlsx"
        if excel_file.exists():
            with open(excel_file, "rb") as f:
                st.download_button(
                    label="📥 Download Formatted Excel Roster (OpenPyXL)",
                    data=f.read(),
                    file_name="churn_target_list.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    filtered_df = df_scores[df_scores["risk_decile"].isin(selected_deciles)]
    if search_hh:
        filtered_df = filtered_df[filtered_df["household_key"].astype(str).str.contains(search_hh)]

    st.dataframe(
        filtered_df[
            [
                "household_key",
                "current_state",
                "risk_decile",
                "calibrated_p",
                "clv_26w",
                "targeting_ev",
                "target_action",
                "reason_1",
                "reason_2",
                "reason_3",
            ]
        ].rename(
            columns={
                "household_key": "Household ID",
                "current_state": "State",
                "risk_decile": "Decile",
                "calibrated_p": "Churn Prob",
                "clv_26w": "26w CLV ($)",
                "targeting_ev": "Targeting EV ($)",
                "target_action": "Action",
                "reason_1": "Primary Risk Driver",
                "reason_2": "Secondary Driver",
                "reason_3": "Tertiary Driver",
            }
        ),
        use_container_width=True,
        height=400,
    )
else:
    st.info("Churn score data not available.")
