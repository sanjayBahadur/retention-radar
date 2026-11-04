"""Departure prediction, evidence, and an inspectable household roster."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "Churn model",
    "The ranking estimates the chance of no shopping trips in the next 56 days.",
)
comparison = read_output("churn_model_comparison.parquet")
selected = comparison.loc[comparison["model"] == "LightGBM"].iloc[0]
a, b, c = st.columns(3)
a.metric("Departures captured in highest-risk 20%", f"{selected.top_20_capture:.1%}")
b.metric("PR-AUC on future test periods", f"{selected.pr_auc:.3f}")
c.metric(
    "95% household-bootstrap interval",
    f"{selected.pr_auc_ci_lower:.3f}–{selected.pr_auc_ci_upper:.3f}",
)
st.caption(
    "Capture is the share of departures in the highest-scored fifth at each cutoff. Average precision summarizes ranking quality."
)
with st.expander("Model comparison"):
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.write(
        "Test cutoffs 20–23 follow training cutoffs 6–13 and validation cutoffs 16–17. Confidence intervals resample whole households."
    )
    st.markdown(
        "[Read the evaluation and error analysis](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/evaluation.md)"
    )
scores = read_output("churn_scores.parquet")
st.subheader("Risk groups")
means = scores.groupby("risk_decile", as_index=False)["calibrated_p"].mean()
figure = px.bar(
    means,
    x="risk_decile",
    y="calibrated_p",
    color_discrete_sequence=[GREEN],
    labels={
        "risk_decile": "Risk group (1 = highest-risk tenth)",
        "calibrated_p": "Mean estimated departure probability",
    },
)
figure.update_yaxes(tickformat=".0%")
chart(figure)
st.subheader("Household scores")
st.caption(
    "Household IDs from the public dataset, with the main factors contributing to each score."
)
deciles = st.multiselect("Risk groups", list(range(1, 11)), default=[1, 2])
query = st.text_input("Household ID contains")
filtered = scores.loc[scores["risk_decile"].isin(deciles)].sort_values(
    "calibrated_p", ascending=False
)
if query:
    filtered = filtered.loc[filtered["household_key"].astype(str).str.contains(query, regex=False)]
columns = ["household_key", "current_state", "calibrated_p", "clv_26w", "reason_1", "reason_2"]
st.dataframe(
    filtered[columns].rename(
        columns={
            "household_key": "Household ID",
            "current_state": "State",
            "calibrated_p": "Departure probability",
            "clv_26w": "Estimated 26-week margin ($)",
            "reason_1": "Model reason 1",
            "reason_2": "Model reason 2",
        }
    ),
    hide_index=True,
    width="stretch",
)
st.download_button(
    "Download the filtered list (.csv)",
    filtered.to_csv(index=False),
    file_name="retention-radar-households.csv",
    mime="text/csv",
)
