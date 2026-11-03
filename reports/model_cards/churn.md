# Churn model card

## Purpose

Rank eligible households by the probability of making no purchases in the next two 28-day periods. A high score is a reason to investigate or include a household in an experiment; it does not demonstrate that an offer will help.

## Model and data

The main model is a LightGBM classifier with isotonic probability calibration. Features summarize activity available at each cutoff, including recency, visits, spend, basket characteristics, and campaign history. The project also compares logistic regression, a recency baseline, and a training-only state baseline.

The evaluation unit is a household-cutoff observation. A household can appear at several cutoffs. Current scores use one record per household at the scoring cutoff.

## Evaluation

Training cutoffs: 6–13. Validation cutoffs: 16–17. Buffer: 18–19. Test cutoffs: 20–23. Tuning and calibration use validation observations; reported test outcomes come from later periods.

Detailed metrics are in [`churn_model_comparison.parquet`](../../data/outputs/churn_model_comparison.parquet), with headline values in [`results.json`](../../data/outputs/results.json).

- **Average precision / PR-AUC:** Ranking quality for the departure outcome.
- **Brier score and calibration error:** Agreement between predicted probabilities and observed outcomes.
- **Top-20% capture:** Fraction of observed departures contained in the highest-scored fifth of evaluation observations.
- **Household-clustered bootstrap intervals:** Uncertainty allowing for repeated observations of each household.

Close model scores should be interpreted with uncertainty. Individual-model confidence intervals alone do not establish a statistically significant difference or equivalence.

## Explanation and decision use

SHAP reason codes describe which inputs contributed to a prediction. They do not identify causes or prove that changing a feature changes the outcome. Contact-list economics combine predicted risk with a cutoff-correct 26-week value forecast, assumed save rate, and contact cost. They are planning scenarios.

## Limitations

- One retailer's historical population; no evidence of transfer to another chain.
- Inactivity at this retailer is not proof of a customer leaving permanently.
- Future purchasing changes, seasonality, and distribution shift can reduce performance.
- Demographic coverage is incomplete and subgroup departure counts can be small. Dropping demographic features does not establish fairness or eliminate proxies.
- A batch score is not a live operational service.

Monitor errors, calibration, and contact outcomes by time and relevant groups. Use randomized outreach before claiming incremental savings. Do not use the model for punitive actions, pricing discrimination, or decisions outside this retail analysis.
