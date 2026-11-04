"""A transparent, household-level retention sensitivity analysis."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "Offer costs and response",
    "Adjust contact depth, save rate and cost to estimate a campaign’s 26-week margin.",
)
scores = read_output("churn_scores.parquet").sort_values(["calibrated_p", "raw_p"], ascending=False)
left, right = st.columns([1, 2], gap="large")
with left:
    depth = st.slider("Share of households contacted (%)", 0, 100, 20)
    save_rate = st.slider("Assumed share of departures prevented (%)", 0, 30, 10)
    cost = st.number_input(
        "Cost per contacted household ($)", min_value=0.0, max_value=100.0, value=2.0, step=0.5
    )
    st.caption("One contact per household, with a 26-week value horizon.")
count = int(len(scores) * depth / 100)
targets = scores.head(count)
gross = float(
    (targets["calibrated_p"] * (save_rate / 100) * targets["clv_26w"].clip(lower=0)).sum()
)
total_cost = count * cost
net = gross - total_cost
with right:
    st.metric("Estimated net 26-week margin", f"${net:,.0f}")
    a, b = st.columns(2)
    a.metric("Households contacted once", f"{count:,}")
    b.metric("Contact cost", f"${total_cost:,.0f}")
    st.write(
        "For each selected household: estimated departure probability × assumed save rate × estimated 26-week margin, minus contact cost."
    )

rows = []
for rate in range(0, 31):
    value = (
        float((targets["calibrated_p"] * (rate / 100) * targets["clv_26w"].clip(lower=0)).sum())
        - total_cost
    )
    rows.append({"Assumed save rate (%)": rate, "Estimated net margin ($)": value})
st.subheader("Margin at different save rates")
figure = px.line(
    rows, x="Assumed save rate (%)", y="Estimated net margin ($)", color_discrete_sequence=[GREEN]
)
figure.add_hline(y=0, line_dash="dot", line_color="#865715", annotation_text="Break-even")
chart(figure)
with st.expander("Calculation inputs"):
    st.write(
        "The save rate is a planning assumption. The calculation values each retained household "
        "at its estimated 26-week margin and subtracts one contact cost."
    )
