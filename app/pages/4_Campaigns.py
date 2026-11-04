"""Historical campaign response and offer-study planning."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "Historical campaigns",
    "Estimated changes in daily spending following an offer, using household-held-out evaluation.",
)
scores = read_output("uplift_scores.parquet")
metadata = read_output("uplift_metadata.json")
evaluation = read_output("uplift_eval.parquet")
a, b = st.columns(2)
a.metric("Households scored", f"{scores.household_key.nunique():,}")
b.metric("Selected learner", metadata["selected_learner"])
st.subheader("Estimated offer response")
figure = px.histogram(
    scores,
    x="uplift_tau",
    nbins=40,
    color_discrete_sequence=[GREEN],
    labels={"uplift_tau": "Estimated change in daily spending ($)"},
)
figure.add_vline(x=0, line_dash="dot", line_color="#865715")
chart(figure)
with st.expander("Model comparison and evaluation"):
    st.dataframe(evaluation, hide_index=True, width="stretch")
    st.write(
        "All three folds hold out entire households. The input is historical campaign targeting; "
        "each household’s latest evaluated campaign supplies its roster score."
    )
    st.markdown(
        "[Campaign model card](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/model_cards/uplift.md)"
    )
with st.expander("Estimated campaign margin"):
    st.caption(
        "One offer per unique household; 35-day spending scenario with configured margin and contact cost."
    )
    st.dataframe(read_output("uplift_roi.parquet"), hide_index=True, width="stretch")
st.subheader("Offer study")
plan = read_output("ab_test_design.json")
st.write(
    f"The planning calculation calls for {plan['total_sample_size']:,} households to detect "
    f"a ${plan['minimum_detectable_effect_spend_day']:.2f} daily spending difference, "
    f"with {plan['available_active_households']:,} households in the current eligible group."
)
st.write(
    "I would use an initial pilot to refine the outcome variance, then size a larger offer-versus-control study."
)
with st.expander("Sample-size inputs"):
    st.json(plan)
