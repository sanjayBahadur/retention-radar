"""Definitions and links to the analytical work."""

import streamlit as st

from retention_radar.dashboard import page

page(
    "Data and methods",
    "dunnhumby Complete Journey: product-line transactions, baskets and household purchase histories.",
)
st.write(
    "DuckDB and dbt build the shopping tables. Python produces the forecasts and evaluations. "
    "The app and public report use the same saved analysis outputs."
)
st.subheader("Definitions")
st.markdown("""
- **Period:** 28 days.
- **Departure / churn:** no visits in the next two periods, or 56 days.
- **Dormant:** two or three consecutive missed periods.
- **Customer value:** forecast spending converted into estimated 26-week margin.
- **Uplift:** estimated spending difference between an offer and no offer.
- **Calibration:** how well predicted probabilities match observed frequencies.
- **Percentage point:** a share falling from 10% to 7% falls three points.
""")
with st.expander("Evaluation windows"):
    st.write(
        "Churn training uses cutoffs 6–13, validation 16–17 and testing 20–23. "
        "Household-clustered intervals account for repeated observations. Customer-value "
        "training outcomes end on day 518; test outcomes cover days 519–700."
    )
with st.expander("Study setup"):
    st.write(
        "Category comparisons use retrospective spending windows around dormancy. "
        "Campaign estimates use historical targeting and household-grouped folds. "
        "Financial scenarios combine forecast margin, contact costs and an assumed response rate. "
        "The technical report contains the full assumptions and diagnostics."
    )
st.subheader("Further reading")
st.markdown("""
- [Code and setup](https://github.com/sanjayBahadur/retention-radar)
- [SQL queries](https://github.com/sanjayBahadur/retention-radar/tree/main/dbt/analyses)
- [Data dictionary](https://github.com/sanjayBahadur/retention-radar/blob/main/docs/data_dictionary.md)
- [Technical report](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/technical_report.md)
- [Error analysis](https://github.com/sanjayBahadur/retention-radar/blob/main/reports/evaluation.md)
- [Model cards](https://github.com/sanjayBahadur/retention-radar/tree/main/reports/model_cards)
""")
