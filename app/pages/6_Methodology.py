"""An approachable explanation of methods, definitions, and limitations."""

import streamlit as st

from retention_radar.dashboard import page

page(
    "How to read the analysis",
    "A short guide to the data, the methods, and what the results can support.",
)
st.subheader("From transactions to a decision")
st.markdown("""
**Transaction line items → SQL/dbt tables → household shopping histories → model evaluation → a proposed experiment.**

The public dunnhumby Complete Journey dataset contains purchases at one grocery chain. Python and DuckDB prepare the data; dbt builds baskets and household-period states. Saved outputs feed both this app and the public case study.
""")
with st.expander("Definitions", expanded=True):
    st.markdown("""
- **Period:** 28 days.
- **Churn / departure:** no trips in the next two periods (56 days). This is a prediction target, not proof of a permanent loss.
- **Dormancy:** two or three consecutive missed periods. It differs from the forward-looking prediction target.
- **CLV:** estimated future customer margin over a stated horizon; the household score uses 26 weeks.
- **Uplift:** the change an offer might cause relative to no offer.
- **Calibration:** agreement between estimated probabilities and observed outcome frequencies.
- **Percentage point:** an absolute difference between percentages. A share falling from 10% to 7% falls 3 percentage points, or 30% relative to its original value.
""")
with st.expander("What makes evaluation credible?"):
    st.write(
        "Keep future outcomes out of training features and use disjoint evaluation windows. Compare against simple baselines. Report uncertainty, sample sizes, and the unit of observation. Repeated records from one household are not independent households."
    )
    st.markdown(
        "[Read evaluation and error analysis](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/evaluation.md)"
    )
with st.expander("Limits worth keeping in view", expanded=True):
    st.markdown("""
- One chain cannot reveal competitor switching or explain why a household leaves.
- Category comparisons include dormancy; a useful advance-warning system requires a separate prospective test.
- Historical campaign selection can confound uplift estimates, even after adjustment.
- Predicted future margin and assumed save rates make campaign profit a scenario, not a realized result.
- Long-range Markov estimates assume stable transitions and simplify Lapsed as absorbing.
- Small demographic samples limit fairness comparisons. Omitting demographics does not remove every proxy or establish fairness.
""")
with st.expander("Code, data, and further detail"):
    st.markdown("""
- [Project README and reproduction steps](https://github.com/sanjayBahadur/retention-radar)
- [Data dictionary](https://github.com/sanjayBahadur/retention-radar/blob/main/docs/data_dictionary.md)
- [Technical report](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/technical_report.md)
- [Churn model card](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/model_cards/churn.md)
- [Customer-value model card](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/model_cards/clv.md)
- [Uplift model card](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/model_cards/uplift.md)
""")
