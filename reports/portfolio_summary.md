# Retention Radar: project description

## Short description

Retention Radar explores how grocery shopping patterns can help prioritize limited retention outreach. It combines SQL and dbt data models, churn prediction, customer-value forecasts, and a guided dashboard, while keeping predicted risk separate from evidence that an offer changes behavior.

## Resume version

Built a retail retention analysis using DuckDB, dbt, Python, and Streamlit on purchase histories for 2,500 households. Compared churn models on later time periods, evaluated 26-week value forecasts, and translated the results into an accessible dashboard and a proposed randomized outreach test.

## Interview walkthrough

1. **Question:** Who should a retailer contact when its outreach budget is limited?
2. **Data:** Product-line transactions become baskets and household-period histories. Historical campaigns use a separate household-campaign grain.
3. **Evaluation:** Compare risk models on later periods and value forecasts on nonoverlapping future outcome windows. Inspect uncertainty and errors, rather than choosing by one headline metric.
4. **Decision:** Explore contact depth and costs with explicit assumptions. Estimated returns are scenarios.
5. **Next evidence:** A randomized household-level offer test would measure incremental contribution and contact guardrails.

Current findings and metrics are in the [case study](https://sanjaybahadur.github.io/retention-radar/) and [generated results](../data/outputs/results.json). There is no claim of realized campaign savings, confirmed competitor switching, or deployment to a real retailer.
