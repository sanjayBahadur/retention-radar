# Retention Radar: technical notes

This document explains the analytical choices behind the [visual case study](https://sanjaybahadur.github.io/retention-radar/). Current metrics are generated into [results.json](../data/outputs/results.json); the corresponding Parquet files retain the detailed evaluations.

## 1. Question and units

The central decision is how to prioritize limited retention outreach. Three different quantities support that decision:

| Quantity | Meaning | Evidence type |
| --- | --- | --- |
| Departure risk | Probability of no purchases in the next two 28-day periods | Prediction evaluated on later observations |
| Customer value | Forecast sales over 26 weeks, converted to discounted margin using explicit assumptions | Forecast with a separate temporal evaluation |
| Campaign uplift | Estimated difference between outcomes with and without contact | Observational estimate requiring causal assumptions |

Do not interpret predicted risk as treatment responsiveness, or modeled margin as realized profit.

A transaction row is a **product line**, not a shopping trip. Basket facts aggregate product lines. Lifecycle facts contain one row per household and 28-day period. Churn evaluation contains repeated household-cutoff observations. Historical campaign analysis contains household-campaign observations; its contact roster uses unique households. Counts and joins must preserve those grains.

## 2. Warehouse and lifecycle

The warehouse uses DuckDB and dbt:

- Staging models cast and align transactions, households, products, campaigns, coupons, redemptions, and promotions.
- Intermediate models aggregate baskets, activity, and coupon transactions.
- Marts expose `dim_household`, `dim_product`, `fct_baskets`, `fct_campaign_exposure`, and `fct_household_period_state`.

A period lasts 28 days. The first three periods provide initial history. Lifecycle states follow [states.yaml](../configs/states.yaml):

| State | Definition |
| --- | --- |
| New | First observed purchase in the current period, after warm-up |
| Champion | Purchased this period; trailing three-period spend is in the current period’s upper quartile |
| Regular | Purchased this period and is neither New nor Champion |
| At-Risk | One consecutive period without purchases |
| Dormant | Two or three consecutive periods without purchases |
| Lapsed | Four or more consecutive periods without purchases |

These are operational labels. They do not describe an individual's intent or permanent customer status.

## 3. Churn evaluation

The target is no purchases in either of the next two periods among eligible, non-Lapsed households. Features summarize information available at the cutoff: recency, visit counts, spending trends, basket variety, and campaign history.

Training uses cutoffs 6–13; validation uses 16–17; training labels end by period 15 and validation labels end by period 19; cutoffs 18–19 separate validation from test; test uses 20–23. LightGBM is tuned on validation data and calibrated with isotonic regression. Baselines include logistic regression, a recency baseline, and a training-only state baseline.

Average precision (reported as PR-AUC) measures ranking performance for an uncommon outcome. Brier score and expected calibration error describe probability accuracy. Top-20% capture measures the fraction of observed departures falling in the highest-scored fifth of evaluation observations. It does not mean those departures were prevented.

Bootstrap intervals resample households to preserve repeated measurements. Overlapping individual-model intervals do not establish equivalence; a paired comparison is needed for a formal claim about the difference. Model choice should also consider contact-list quality, calibration, interpretability, and maintenance.

### Error analysis

The [generated evaluation review](evaluation.md) reports current error slices. Inspect false alarms and missed departures at a contact depth fixed before inspecting test outcomes. Compare performance across cutoffs, tenure bands, and visit-frequency groups. Always show observation counts, unique-household counts, and departure counts. Small subgroup differences are uncertain. Demographic missingness is substantial, so demographic audits cannot establish broad fairness, and dropping sensitive inputs cannot remove all proxies.

## 4. Customer value

The customer-value benchmark uses genuine 26-week targets (182 days). The earlier origin is day 336, with labels from days 337–518; the later evaluation origin is day 518, with labels from days 519–700. The later outcome window is not used to fit the evaluated model. Forecast errors are reported in **sales dollars**, separately from conversion to discounted margin.

Purchase-frequency and monetary quantities use the same purchase occasion. Candidate models include a recent-spend baseline, BG/NBD with Gamma-Gamma, and a LightGBM Tweedie regressor. See the [CLV model card](model_cards/clv.md) for selection and current output fields. Scoring a historical churn cutoff must use a value forecast based on information available at that cutoff.

This is a limited historical evaluation on the same retailer's population. It does not establish transfer to another retailer or future economic conditions.

## 5. Outreach economics

For household `i`, the illustrative retention value is:

```text
expected net value = predicted departure probability
                   × assumed save rate
                   × forecast 26-week margin
                   − contact cost
```

Margin rate, discount rate, contact cost, and save rate are explicit assumptions in [project.yaml](../configs/project.yaml). A positive estimate is conditional on those assumptions. It is not an observed treatment effect, a guarantee, or a claim of money earned. Scaling a scenario to a larger population does not add validation.

## 6. Historical campaign response

The campaign analysis uses pre-campaign features, recorded contact, and subsequent spend. Learners estimate conditional outcome differences. Household-grouped cross-validation keeps the same household out of its model's training fold; evaluation covers all held-out folds.

Historical campaign assignments were not randomized. Conditional exchangeability, sufficient overlap, and limited interference are assumptions, not facts proven by a fitted model. Propensity trimming can improve overlap but cannot remove unmeasured confounding. Randomized Hillstrom and X5 benchmarks test implementation behavior in other settings; they do not establish a causal effect for Dunnhumby's campaigns.

Campaign observation counts differ from unique household counts. Repeated campaign observations are retained for evaluation, while a deduplicated household roster supports a single-contact scenario. Estimated response segments are exploratory descriptions. A negative point estimate does not justify permanent suppression or establish that contact harms a household.

## 7. Category changes, lifecycle models, and attribution

Category analysis compares early and late spending shares around dormancy with an active comparison group. Report the difference in share changes in **percentage points**. Relative dollar-spend drops answer a different question. The windows include the approach to dormancy; they are not an independently validated real-time lead-time detector. No competitor purchases or customer motives are observed.

The empirical lifecycle transition matrix allows returns from Lapsed. Absorbing-chain calculations temporarily treat Lapsed as terminal to derive expected time-to-lapse and discounted state values. Their long-run extrapolations rely on stable transitions and can extend beyond the observed record. Infinite-horizon state values are separate from the 26-week forecast.

Campaign attribution assigns credit under first-touch, last-touch, linear, and Markov removal rules. Removing a node in an attribution graph is a model operation, not a randomized removal of marketing. Attribution credit is not incremental causal impact.

## 8. Rebuilding and checks

Install Python 3.12 dependencies as described in the [README](../README.md). The full analysis requires the source datasets in the configured raw-data directory:

```bash
make data
make dbt
.venv/bin/python -m retention_radar.pipeline --skip-ingest --skip-dbt
.venv/bin/python -m retention_radar.reporting.numbers
.venv/bin/python -m retention_radar.reporting.generate_html_report
.venv/bin/python -m retention_radar.bi.export
make lint
make test
```

Tests cover analytical functions, schemas, and application behavior. Warehouse-dependent tests require local source data. Passing checks establishes the behavior tested; it does not prove that causal assumptions or financial scenarios hold in practice. Dataset provenance and redistribution considerations are described in the [data use note](../docs/data_governance.md).
