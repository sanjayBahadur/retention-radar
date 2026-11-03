"""Explain lifecycle states before showing transition estimates."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import STATE_COLORS, STATES, chart, page, read_output

page(
    "How shopping activity changes",
    "A lifecycle state summarizes recent shopping activity. Each step below represents 28 days.",
)
with st.expander("What do the six states mean?", expanded=True):
    st.markdown("""
- **New:** first observed purchase after the warm-up window.
- **Champion:** shopped this period, with trailing spending in the highest quarter of active households.
- **Regular:** shopped this period and is neither New nor Champion.
- **At-Risk:** missed one period after shopping in the previous period.
- **Dormant:** missed two or three consecutive periods.
- **Lapsed:** missed four or more consecutive periods.
""")
distribution = read_output("state_distribution.parquet")
st.subheader("The mix of household states over time")
figure = px.area(
    distribution,
    x="period_id",
    y="household_count",
    color="state",
    category_orders={"state": STATES},
    color_discrete_map=STATE_COLORS,
    labels={"period_id": "28-day period", "household_count": "Households", "state": "State"},
)
chart(figure)
st.subheader("Where do households move next?")
transitions = read_output("markov_P.parquet")
matrix = transitions.pivot(
    index="source_state", columns="target_state", values="transition_probability"
).reindex(index=STATES, columns=STATES)
figure = px.imshow(
    matrix,
    text_auto=".0%",
    color_continuous_scale=["#f5f3ec", "#234e3b"],
    labels={"x": "Next period", "y": "Current period", "color": "Probability"},
    zmin=0,
    zmax=1,
)
figure.update_layout(height=460)
chart(figure)
st.caption(
    "Read across a row. Each cell estimates the chance of moving to that state in one period. This empirical matrix allows return visits from Lapsed. A separate long-run calculation treats Lapsed as absorbing."
)
with st.expander("Technical detail: long-range estimates and observed survival"):
    metrics = read_output("markov_metrics.json")
    st.write(
        "Expected time to lapse projects current transition patterns far beyond the observation window. It depends on stable behavior and the absorbing-state simplification."
    )
    st.write(
        f"Backtest distribution error (lower is better): Markov "
        f"{metrics['backtest_avg_markov_tvd']:.4f}; unchanged-distribution baseline "
        f"{metrics['backtest_avg_persistence_tvd']:.4f}."
    )
    st.caption(
        "Assumption checks also find evidence against a time-stable, first-order chain. "
        "Use lifecycle groups to describe shopping activity; treat long-range projections cautiously."
    )
    st.json(metrics["expected_days_to_lapse"])
    survival = read_output("survival_curves.parquet")
    figure = px.line(
        survival,
        x="timeline_days",
        y="survival_probability",
        color="baseline_state",
        color_discrete_map=STATE_COLORS,
        labels={
            "timeline_days": "Days since baseline",
            "survival_probability": "Share not yet lapsed",
            "baseline_state": "Starting state",
        },
    )
    figure.update_yaxes(tickformat=".0%")
    chart(figure)
