"""Overview of the retail analysis and its main findings."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

results = read_output("results.json")
categories = read_output("early_warning_categories.parquet")
capture = results["churn"]["top_20pct_capture_rate"]
forecasts = read_output("clv_comparison.parquet").set_index("method")
page(
    "Finding shoppers at risk of leaving",
    "I used grocery purchase histories to rank households by the chance of making no shopping trips in the next eight weeks.",
)
st.caption("dunnhumby Complete Journey · 2,500 households · 25 complete 28-day periods")
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Where spending changed")
    top = categories.sort_values("net_drop_effect").head(6).copy()
    top["difference_pp"] = top["net_drop_effect"] * 100
    figure = px.bar(
        top,
        x="difference_pp",
        y="commodity_desc",
        orientation="h",
        labels={
            "difference_pp": "Change in share, relative to active households (percentage points)",
            "commodity_desc": "",
        },
        color_discrete_sequence=[GREEN],
    )
    figure.update_layout(yaxis=dict(autorange="reversed"), height=350)
    chart(figure)
    st.caption("Earlier and later three-period windows around first dormancy.")
with right:
    st.metric("Departures in the highest-risk fifth", f"{capture:.0%}")
    st.write(
        "At each of four test cutoffs, I selected the highest-risk 20%. That group contained three quarters of subsequent departures."
    )
    st.subheader("A simple forecast held up best")
    st.write(
        f"Recent spending gave the lowest average error for 26-week sales: "
        f"${forecasts.loc['Recent 26-week spend', 'mae_dollars']:,.0f} per household, "
        f"compared with ${forecasts.loc['BG/NBD + Gamma-Gamma', 'mae_dollars']:,.0f} for BG/NBD "
        f"and ${forecasts.loc['LightGBM (Tweedie)', 'mae_dollars']:,.0f} for LightGBM."
    )
st.divider()
for path, label in [
    ("pages/3_Churn_Risk.py", "Churn model and household scores"),
    ("pages/5_Baskets.py", "Shopping categories and basket rules"),
    ("pages/2_Retention_Simulator.py", "Offer costs and response"),
    ("pages/4_Campaigns.py", "Historical campaigns"),
    ("pages/1_Lifecycle.py", "Customer lifecycle"),
    ("pages/6_Methodology.py", "Data and methods"),
]:
    st.page_link(path, label=label)
