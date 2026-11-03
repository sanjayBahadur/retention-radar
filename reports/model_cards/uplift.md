# Campaign response: model notes

## Question

Which households appear more responsive to a promotion, and which idea should a retailer test next?

This analysis compares historical spending changes among contacted and uncontacted households. Campaigns were not randomized, so the result is an exploratory ranking. It does not prove that an offer caused a change in spending.

## Data and evaluation

- **Observation:** one household in one historical campaign. A household can appear in several campaigns.
- **Inputs:** shopping activity during the preceding 56 days, campaign type, demographic-record availability, and the number of overlapping campaigns.
- **Outcome:** spending per day during the campaign minus spending per day in the preceding 56 days.
- **Models:** S-, T-, X- and doubly robust learners with LightGBM base regressors.
- **Evaluation:** all three household-grouped folds. Each household is scored by models trained on other households. The propensity model is also fitted only on the training households in that fold.
- **Overlap:** keep campaign observations with estimated contact probabilities from 0.05 to 0.95. This changes the population the findings describe.

`data/outputs/uplift_eval.parquet` contains the pooled out-of-fold comparisons. Qini measures how well a score ranks differences in observed responses. On this observational dataset, Qini and top-segment treated/control contrasts can still reflect selection bias. The model with the highest pooled Qini supplies the exploratory ranking; there is no additional untouched test set after that selection.

Household grouping prevents sharing a household across training and evaluation. It does not simulate a future campaign or eliminate unmeasured confounding. A separate time-based campaign evaluation would be needed before making a deployment decision.

## Households, score bands and planning scenarios

`uplift_observations.parquet` preserves every evaluated campaign-household observation. `uplift_scores.parquet` keeps each household once, using its latest evaluated campaign context. Those contexts may come from different dates: this is a retrospective planning roster, not a live campaign list.

The four conventional segment labels are shorthand for model score bands:

| Label | Rule | Interpretation |
|---|---|---|
| Persuadables | Predicted change above $0.15/day | Positive model score |
| Sleeping Dogs | Predicted change below −$0.15/day | Negative model score |
| Sure Things | Score between those thresholds, higher prior spending | Near-zero score, higher baseline activity |
| Lost Causes | Score between those thresholds, lower prior spending | Near-zero score, lower baseline activity |

The labels are not observed customer traits. Negative scores do not establish irritation or harm and do not justify permanent suppression.

`uplift_roi.parquet` models one offer per unique household at each contact depth. Assumptions are a **35-day scenario**, the configured gross-margin percentage, and the configured offer cost. Net profit is predicted extra spend × days × margin, minus contact costs; ROI is net profit divided by contact costs. Random-contact profit uses the overall mean predicted effect at the same contact count. These are model-based scenarios, not measured campaign returns or independent policy evaluations.

## Experiment proposal

`ab_test_design.json` describes an eight-week household-randomized offer-versus-control experiment within the top 30% of the frozen ranking. Its $0.50/day minimum detectable effect is an explicit planning assumption. Sample size uses historical household outcome variance, not variation in predicted treatment effects. The output reports whether the available pool is large enough; a pilot should refine the variance estimate.

Freeze the eligible population and analysis before launch, randomize each household once, record other promotions and actual costs, and analyze everyone by their original assignment. Check net margin, complaints and opt-outs before considering rollout. This tests an offer within the selected pool; it does not by itself prove the ranking beats random targeting.

## Supporting benchmarks

Hillstrom and X5 RetailHero provide randomized-data benchmarks for the modeling workflow. Their findings and the synthetic confounding stress test belong to those datasets. They do not validate causal assumptions for historical Dunnhumby campaigns. Propensity balance can reveal observed differences; it cannot verify the absence of unmeasured confounding.

## Reproduce

After building the warehouse and CLV outputs:

```bash
.venv/bin/python -m retention_radar.uplift.dunnhumby
.venv/bin/python -m retention_radar.uplift.power
```

See `uplift_metadata.json` for counts, assumptions and limitations from the current run.
