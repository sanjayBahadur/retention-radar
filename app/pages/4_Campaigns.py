"""Explain observational uplift without presenting it as a proven policy."""

import plotly.express as px
import streamlit as st

from retention_radar.dashboard import GREEN, chart, page, read_output

page(
    "Would an offer change behavior?",
    "Uplift asks how an outcome might differ with an offer versus without one. Historical targeting makes that harder to estimate than departure risk.",
)
st.info(
    "Campaign exposure was not randomized in this retail dataset. Model estimates can reflect selection bias. They do not justify permanent contact suppression or claims of realized profit."
)
evaluation = read_output("uplift_eval.parquet")
st.subheader("Compare the observational models")
st.dataframe(evaluation, hide_index=True, width="stretch")
st.caption(
    "Qini and AUUC assess uplift ranking under the evaluation assumptions. Read the model card for the split strategy, observation counts, and remaining confounding."
)
st.markdown(
    "[Uplift model card](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/model_cards/uplift.md)"
)
scores = read_output("uplift_scores.parquet")
a, b = st.columns(2)
a.metric("Rows in the latest-campaign scored list", f"{len(scores):,}")
b.metric("Unique households in that list", f"{scores['household_key'].nunique():,}")
st.subheader("Distribution of estimated offer effects")
figure = px.histogram(
    scores,
    x="uplift_tau",
    nbins=40,
    color_discrete_sequence=[GREEN],
    labels={"uplift_tau": "Estimated change in daily spending ($)"},
)
figure.add_vline(x=0, line_dash="dot", line_color="#865715")
chart(figure)
st.caption(
    "Negative estimates suggest a hypothesis for a controlled test. They do not prove that an offer annoys a household or causes it to leave."
)
with st.expander("Explore modeled campaign economics"):
    roi = read_output("uplift_roi.parquet")
    st.dataframe(roi, hide_index=True, width="stretch")
    st.write(
        "These values combine estimated response and assumed costs. They are not observed incremental profits. Check the model card for whether rows represent distinct households or campaign observations."
    )
st.subheader("The next useful step: a randomized test")
st.markdown("""
1. Define the eligible household population and choose one offer.
2. Randomly assign each household to the offer or a holdout; keep it in one arm.
3. Preselect the outcome, follow-up window, minimum detectable effect, and sample size.
4. Compare net outcomes and uncertainty before rolling out or withholding contact.
""")
with st.expander("Sample-size planning assumptions"):
    st.json(read_output("ab_test_design.json"))
