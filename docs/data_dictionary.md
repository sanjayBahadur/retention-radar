# Data dictionary

This guide describes the fields needed to follow the analysis. A **grain** states what one row represents. Keep it explicit when counting, joining, or summing tables.

## Warehouse tables

| Table | Grain | Important fields |
| --- | --- | --- |
| `raw.transaction_data` | Product line in a basket | `household_key`, `basket_id`, `product_id`, `day`, `sales_value`, `quantity` |
| `raw.product` | Product | `product_id`, `department`, `commodity_desc`, `brand` |
| `raw.hh_demographic` | Household with demographic information | `household_key`, age, income, and household-size bands |
| `raw.campaign_table` | Household contacted in a campaign | `household_key`, `campaign` |
| `raw.campaign_desc` | Campaign | `campaign`, `start_day`, `end_day`, `description` |
| `raw.coupon` | Campaign–coupon–product association | `campaign`, `coupon_upc`, `product_id` |
| `raw.coupon_redempt` | Recorded coupon redemption | `household_key`, `campaign`, `coupon_upc`, `day` |
| `raw.causal_data` | Product–store–week promotion record | `product_id`, `store_id`, `week_no`, `display`, `mailer` |
| `marts.fct_baskets` | Basket | Household, day, spend, items, departments, discounts |
| `marts.fct_household_period_state` | Household–period | Activity, trailing spend, missed periods, state |
| `marts.fct_campaign_exposure` | Household–campaign exposure | Campaign timing and recorded exposure |

`household_key` is a pseudonymous source identifier. `day` is a relative day number, not a calendar date. `sales_value` is recorded product-line sales in dollars. Row counts of transactions must not be called shopping trips; aggregate by basket first.

`retail_disc` records retail discounts, `coupon_disc` manufacturer coupon discounts, and `coupon_match_disc` retailer coupon matching. Discounts are recorded as negative values; keep these signs when reconstructing pre-discount sales.

## Lifecycle

Each period is 28 days. Definitions are maintained in [states.yaml](../configs/states.yaml).

- **New:** First observed purchase this period, after the first three warm-up periods.
- **Champion:** Active this period and in the current period’s upper quartile of trailing three-period spend.
- **Regular:** Active, neither New nor Champion.
- **At-Risk:** One consecutive period without a purchase.
- **Dormant:** Two or three consecutive periods without a purchase.
- **Lapsed:** Four or more consecutive periods without a purchase. Returns remain possible in the observed data.

## Main analysis outputs

| File | Grain and interpretation |
| --- | --- |
| `state_distribution.parquet` | Period–state; `household_count` is a count and `pct_of_period` is a population share |
| `markov_P.parquet` | Source–target state; `transition_probability` is a one-period transition probability |
| `churn_features.parquet` | Household–cutoff; inputs available at the cutoff and a subsequent inactivity label |
| `churn_model_comparison.parquet` | Model; later-period evaluation and uncertainty |
| `churn_scores.parquet` | Household at the scoring cutoff; risk and illustrative outreach value |
| `clv_comparison.parquet` | Forecast method; evaluation error in sales dollars over 26 weeks |
| `clv.parquet` | Household; `clv_26w` is forecast discounted margin dollars, using stated assumptions |
| `uplift_eval.parquet` | Learner; held-out observational campaign evaluation |
| `uplift_observations.parquet` | Household–campaign; all retained out-of-fold campaign observations |
| `uplift_scores.parquet` | Unique household contact roster; latest evaluated campaign context, retrospective and observational |
| `early_warning_categories.parquet` | Commodity; early-to-late share changes and comparison-group difference |
| `basket_rules.parquet` | Item-set rule; support, confidence, and association lift |
| `results.json` | Generated summary; values for the dashboard and reports |

## Reading scores and financial fields

- `calibrated_p`: Estimated probability of no purchases in the next two periods, between 0 and 1.
- `risk_decile`: Ranked tenth of the scored population; 1 is highest risk.
- `targeting_ev`: Illustrative net value: risk × assumed save rate × forecast margin − contact cost.
- `reason_1`, `reason_2`, `reason_3`: Prediction explanations, not causal explanations.
- `net_drop_effect`: Difference between the groups' category-share changes. Multiply by 100 and label **percentage points**, not percent spend lost.
- `uplift_tau`: Estimated change in spend per day under contact versus no contact; causal interpretation requires assumptions that the historical data cannot verify.

The [technical notes](../reports/technical_report.md) explain evaluation windows and limitations. Read actual Parquet columns when developing integrations; export files preserve their source column names.
