# Explore the data in Tableau or Power BI

The project includes CSV exports and connection guidance. It does not include a finished Tableau workbook or Power BI report.

Generate exports from the analysis outputs:

```bash
.venv/bin/python -m retention_radar.bi.export
```

Files are written to `data/outputs/bi/`. The exporter preserves source column names.

## Tables and safe relationships

| File | Grain | Suggested use |
| --- | --- | --- |
| `bi_churn_scores.csv` | Household at a scoring cutoff | Risk, forecast margin, explanation, and contact-list exploration |
| `bi_customer_segments.csv` | Unique household contact roster | Exploratory response estimates; inspect campaign context before joining |
| `bi_lifecycle_states.csv` | Period–state | Activity composition over time |
| `bi_campaign_attribution.csv` | Campaign type | Credit allocation under different attribution rules |
| `bi_basket_rules.csv` | One of the first 100 exported rules | Filterable support, confidence, and lift |

Import CSVs with household IDs as whole numbers or text, probabilities as decimal numbers, and money fields as currency. Keep aggregate lifecycle, attribution, and basket-rule tables as separate data sources. If relating household tables, verify uniqueness first and use a single filter direction; do not physically join aggregate tables to a customer roster or sum duplicated customer values.

## Suggested report

1. **Shopping changes:** Lifecycle composition and category-share changes with definitions beside the chart.
2. **Contact priorities:** Risk versus forecast margin, filters for risk decile, and a household table with reason codes.
3. **What to test:** Observational campaign response estimates and transparent financial assumptions.

A count of selected households is not churn capture. Capture requires labeled future outcomes from the evaluation dataset. Current score exports do not contain those outcomes.

## Example measures

These examples use actual columns from the churn export. They illustrate analysis, not a measured campaign return.

Power BI:

```dax
Scored Households = DISTINCTCOUNT('bi_churn_scores'[household_key])

Forecast Margin at Risk =
SUMX(
    'bi_churn_scores',
    'bi_churn_scores'[calibrated_p] * 'bi_churn_scores'[clv_26w]
)

Scenario Net Value = SUM('bi_churn_scores'[targeting_ev])
```

`Forecast Margin at Risk` is a descriptive product of risk and forecast value. `Scenario Net Value` uses the assumptions embedded when scoring; changing a report filter does not change the assumed save rate or contact cost.

Tableau:

```text
// Forecast margin at risk
[calibrated_p] * [clv_26w]

// Risk band for exploring the current roster
IF [risk_decile] <= 2 THEN "Highest-scored fifth"
ELSE "Remaining households"
END
```

For a custom scenario, define visible parameters for save rate and contact cost:

```text
[calibrated_p] * [Assumed Save Rate] * [clv_26w] - [Contact Cost]
```

Use the configured margin and discount assumptions in the underlying value forecast. Do not add margin a second time to `clv_26w`, which is already a margin estimate.

## Interpretation

Category-share differences use percentage points. Attribution shares describe a credit-allocation model. Estimated campaign response is observational. Negative estimates do not establish harm or justify automatic suppression. See the [technical notes](../reports/technical_report.md) for evaluation and the [data use note](../docs/data_governance.md) before sharing row-level exports.
