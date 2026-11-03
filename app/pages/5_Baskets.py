"""Inspect spending-share changes and basket associations."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "What changes in the shopping basket?",
    "Compare category spending shares around first dormancy, then inspect products that tend to appear together.",
)
categories = read_output("early_warning_categories.parquet")
top = categories.sort_values("net_drop_effect").head(12).copy()
top["difference_pp"] = top["net_drop_effect"] * 100
st.subheader("Category share changes around dormancy")
figure = px.bar(
    top,
    x="difference_pp",
    y="commodity_desc",
    orientation="h",
    color_discrete_sequence=[GREEN],
    labels={
        "difference_pp": "Difference in share change (percentage points)",
        "commodity_desc": "Category",
    },
)
figure.update_layout(yaxis=dict(autorange="reversed"), height=480)
chart(figure)
st.write(
    "A negative bar means spending share fell more in the dormant group than in the active comparison group. A 3-point share change is different from a 3% drop in spending."
)
st.caption(
    "Earlier and later three-period windows are compared; the later window includes dormancy itself. This does not establish a prospective warning interval. Competitor purchases and household motivations are unobserved."
)
with st.expander("Inspect the category calculations"):
    st.dataframe(categories, hide_index=True, width="stretch")
st.subheader("Which products tend to appear together?")
st.write(
    "Basket associations can suggest merchandising questions. They do not show that a placement or promotion will increase sales or prevent departures."
)
rules = read_output("basket_rules.parquet")
a, b = st.columns(2)
with a:
    lift = st.slider(
        "Minimum lift",
        1.0,
        5.0,
        1.5,
        step=0.1,
        help="Lift above 1 means the items co-occur more often than expected from their separate frequencies.",
    )
with b:
    query = st.text_input("Category contains", placeholder="For example, milk")
filtered = rules.loc[rules["lift"] >= lift]
if query:
    mask = filtered["antecedents_str"].str.contains(
        query, case=False, regex=False, na=False
    ) | filtered["consequents_str"].str.contains(query, case=False, regex=False, na=False)
    filtered = filtered.loc[mask]
st.caption(
    f"{len(filtered):,} matching rules; showing the 50 highest-lift matches. Support is the share of all baskets containing the combination. Confidence is the share containing the second item among baskets with the first."
)
st.dataframe(
    filtered.sort_values("lift", ascending=False).head(50), hide_index=True, width="stretch"
)
