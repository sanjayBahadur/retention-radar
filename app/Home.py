"""A guided entry to the retail case study."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "When shoppers start drifting away",
    "Can shopping patterns help a retailer decide who to contact—and what to test before spending money on retention?",
)
results = read_output("results.json")
categories = read_output("early_warning_categories.parquet")
capture = results["churn"]["top_20pct_capture_rate"]

st.caption(
    "Public grocery data · 2,500 households · approximately 2.6 million transaction line items · 28-day lifecycle periods"
)
st.divider()
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("1. Start with the shopping changes")
    st.write(
        "Category spending shares change around dormancy. These are clues to investigate; they do not explain why someone leaves."
    )
    top = categories.sort_values("net_drop_effect").head(6).copy()
    top["difference_pp"] = top["net_drop_effect"] * 100
    figure = px.bar(
        top,
        x="difference_pp",
        y="commodity_desc",
        orientation="h",
        labels={
            "difference_pp": "Difference in share change (percentage points)",
            "commodity_desc": "",
        },
        color_discrete_sequence=[GREEN],
    )
    figure.update_layout(yaxis=dict(autorange="reversed"), height=350)
    chart(figure)
    st.caption(
        "Dormant group minus active comparison group. The later window includes dormancy; this is not a validated advance warning."
    )
with right:
    st.subheader("2. Understand the risk")
    st.metric("Departures captured in the highest-risk 20%", f"{capture:.1%}")
    st.write(
        "A departure means no shopping trips in the next two periods (56 days). The model concentrates observed departures into a smaller outreach list."
    )
    st.write(
        "Identifying a likely departure does not tell us whether an offer will persuade that person to stay."
    )
    st.subheader("3. Choose a test")
    st.write(
        "Randomly assign eligible households to an offer or a holdout. Compare outcomes and contact costs before expanding the policy."
    )

st.divider()
st.subheader("Explore one question at a time")
for path, label in [
    ("pages/5_Baskets.py", "Shopping patterns — what changes in the basket?"),
    ("pages/3_Churn_Risk.py", "Departure risk — how well can we prioritize outreach?"),
    (
        "pages/2_Retention_Simulator.py",
        "Retention scenario — which assumptions change the estimate?",
    ),
    ("pages/4_Campaigns.py", "Campaigns — what needs a controlled test?"),
    ("pages/1_Lifecycle.py", "Lifecycle — how does shopping activity change over time?"),
    ("pages/6_Methodology.py", "Methods — definitions, evaluation, and limitations"),
]:
    st.page_link(path, label=label)
