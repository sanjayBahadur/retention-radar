# Retention Radar: Business Intelligence Starter Package (Power BI & Tableau)

This directory provides the production data model, connection recipes, complete data dictionary, and 10 ready-to-paste calculated fields (DAX and Tableau formulas) for deploying **Retention Radar** into enterprise analytics platforms (Power BI, Tableau, Looker, Metabase).

---

## 1. Data Architecture & Export Formats

All datasets are pre-processed, schema-validated via Pandera, and exported to `data/outputs/bi/` as UTF-8 CSVs (and corresponding Parquet files in `data/outputs/`):

| File Name | Grain / Primary Key | Record Count | Description |
| :--- | :--- | :--- | :--- |
| `bi_churn_scores.csv` | `household_key` | 2,351 | Active household churn risk scores, deciles, calibrated probabilities, expected margin loss, and top-3 SHAP reason codes |
| `bi_customer_segments.csv` | `household_key`, `campaign` | 9,444 | Causal uplift CATE $\tau$, targeting decile, and uplift segment (`Persuadables`, `Sure Things`, `Sleeping Dogs`, `Lost Causes`) |
| `bi_lifecycle_states.csv` | `period_id`, `state` | 175 | Longitudinal Markov state distributions, transition shares, and cohort movement |
| `bi_campaign_attribution.csv` | `campaign_type` | 3 | Algorithmic Markov removal effect shares vs first-touch, last-touch, and linear heuristics |
| `bi_basket_rules.csv` | `antecedents_str`, `consequents_str` | 100 | Top merchandise association rules (support $\ge 1\%$, lift $> 1.20$) for cross-selling and churn prevention |

---

## 2. Platform Connection Instructions

### Connecting in Power BI Desktop
1. Open **Power BI Desktop** $\rightarrow$ Click **Get Data** $\rightarrow$ Select **Text/CSV** or **Folder**.
2. Browse to `<project_root>/data/outputs/bi/`.
3. Load the 5 CSV tables:
   - Ensure `household_key` is formatted as **Whole Number** or **Text**.
   - Ensure probability and margin columns (`calibrated_p`, `clv_26w`, `expected_margin_loss`, `uplift_tau`) are formatted as **Decimal Number** / **Currency ($)**.
4. Go to the **Model View** to define relationships:
   - Create a **1-to-Many relationship** from `bi_churn_scores` (`household_key`) to `bi_customer_segments` (`household_key`).
   - Cross-filter direction: **Both** or **Single (bi_churn_scores filters bi_customer_segments)**.

### Connecting in Tableau Desktop
1. Open **Tableau Desktop** $\rightarrow$ Under *Connect* $\rightarrow$ Select **Text file**.
2. Select `bi_churn_scores.csv` as the primary logical table.
3. Add a Relationship (noodle) to `bi_customer_segments.csv` joining on `household_key = household_key`.
4. Connect independently or as separate datasources to `bi_lifecycle_states.csv`, `bi_campaign_attribution.csv`, and `bi_basket_rules.csv`.
5. Extract the data or use Live connection for real-time batch file refreshes.

---

## 3. Data Dictionary

### `bi_churn_scores.csv`
- `household_key` *(Integer)*: Unique anonymized Dunnhumby household identifier.
- `state` *(String)*: Current Markov lifecycle health state (`Champion`, `Regular`, `At-Risk`, `New`).
- `calibrated_p` *(Float $\in [0, 1]$)*: Isotonically calibrated probability of churning over the next 28-day cycle.
- `risk_decile` *(Integer $\in [1, 10]$)*: 1 = Highest 10% churn probability; 10 = Lowest 10% churn probability.
- `clv_26w` *(Float)*: Projected 26-week customer margin lifetime value ($).
- `expected_margin_loss` *(Float)*: `calibrated_p * clv_26w` — dollar margin at immediate risk.
- `primary_driver` *(String)*: Top SHAP feature contributing to elevated churn risk.
- `secondary_driver` *(String)*: Second highest SHAP feature contribution.
- `tertiary_driver` *(String)*: Third highest SHAP feature contribution.

### `bi_customer_segments.csv`
- `household_key` *(Integer)*: Household identifier.
- `campaign` *(Integer)*: Direct marketing campaign ID.
- `treated` *(Integer $\in \{0, 1\}$)*: 1 if household was contacted in campaign; 0 if uncontacted control.
- `propensity_score` *(Float $\in [0.05, 0.95]$)*: Estimated conditional probability of receiving treatment.
- `uplift_tau` *(Float)*: Individual conditional average treatment effect ($\hat{\tau}(X)$ in incremental spend per day).
- `segment` *(String)*: Uplift classification:
  - `Persuadables`: $\hat{\tau} > 0.05$ (target with retention coupon).
  - `Sure Things`: Baseline spend high, $\hat{\tau} \approx 0$ (do not discount).
  - `Sleeping Dogs`: $\hat{\tau} < -0.05$ (negative response; suppress from communications).
  - `Lost Causes`: Baseline spend low, $\hat{\tau} \le 0$ (do not waste budget).

---

## 4. Calculated Fields Reference

### 5 Power BI DAX Measures

1. **Total Churn Value at Risk ($):**
   ```dax
   Total_Churn_Value_At_Risk = 
   SUMX('bi_churn_scores', 'bi_churn_scores'[calibrated_p] * 'bi_churn_scores'[clv_26w])
   ```

2. **Top Decile Capture Rate (%):**
   ```dax
   Top_Decile_Churn_Capture = 
   DIVIDE(
       CALCULATE(COUNTROWS('bi_churn_scores'), 'bi_churn_scores'[risk_decile] = 1),
       CALCULATE(COUNTROWS('bi_churn_scores'), ALL('bi_churn_scores')),
       0
   )
   ```

3. **Incremental Net Uplift Margin ($) at 10% Winback:**
   ```dax
   Net_Uplift_Profit = 
   SUMX(
       FILTER('bi_customer_segments', 'bi_customer_segments'[segment] = "Persuadables"),
       ('bi_customer_segments'[uplift_tau] * 28 * 0.28) - 1.50
   )
   ```

4. **Targeting ROI Ratio:**
   ```dax
   Targeting_ROI = 
   DIVIDE(
       [Net_Uplift_Profit],
       CALCULATE(COUNTROWS('bi_customer_segments'), 'bi_customer_segments'[segment] = "Persuadables") * 1.50,
       0
   )
   ```

5. **Sleeping Dogs Headcount Alert:**
   ```dax
   Sleeping_Dogs_Count = 
   CALCULATE(
       DISTINCTCOUNT('bi_customer_segments'[household_key]),
       'bi_customer_segments'[segment] = "Sleeping Dogs"
   )
   ```

---

### 5 Tableau Calculated Fields

1. **Urgency Classification:**
   ```sql
   // Name: [Outreach Urgency Level]
   IF [calibrated_p] >= 0.25 AND [clv_26w] >= 500 THEN "Priority Tier 1 (High P, High CLV)"
   ELSEIF [calibrated_p] >= 0.25 THEN "Priority Tier 2 (High P, Modest CLV)"
   ELSEIF [clv_26w] >= 500 THEN "Priority Tier 3 (VIP Watchlist)"
   ELSE "Low Risk / Self-Sustaining"
   END
   ```

2. **Campaign Net Value Creation ($):**
   ```sql
   // Name: [Net Campaign Profit]
   IF [segment] = "Persuadables" THEN ([uplift_tau] * 28.0 * 0.28) - 1.50
   ELSEIF [segment] = "Sleeping Dogs" THEN ([uplift_tau] * 28.0 * 0.28) - 1.50 // Destruction
   ELSE -1.50 // Pure cost waste on Sure Things and Lost Causes
   END
   ```

3. **Sleeping Dog Warning Flag:**
   ```sql
   // Name: [Is Sleeping Dog Suppression]
   IF [segment] = "Sleeping Dogs" THEN "🚨 SUPPRESS FROM OUTREACH"
   ELSE "ELIGIBLE"
   END
   ```

4. **Estimated Expected Value Saved:**
   ```sql
   // Name: [Expected Margin Saved @ 10% Winback]
   [expected_margin_loss] * 0.10
   ```

5. **Early Warning Lift Ratio:**
   ```sql
   // Name: [Association Lift Level]
   IF [lift] >= 2.0 THEN "Strong Complement (Lift >= 2.0)"
   ELSEIF [lift] >= 1.5 THEN "Moderate Complement (Lift 1.5-2.0)"
   ELSE "Baseline Association"
   END
   ```

---

## 5. Recommended BI Dashboard Structure

We recommend configuring a 3-tab layout matching the core operational surfaces:

1. **Executive Churn Command Center:**
   - Top KPI cards: Total Value at Risk, Top Decile Churners, 26-Week Mean CLV, Active Households.
   - Scatter plot: `clv_26w` (Y-axis) vs `calibrated_p` (X-axis) colored by `state` to isolate high-value at-risk accounts.
   - Target list grid with sortable columns and multi-select filters on `primary_driver` and `risk_decile`.

2. **Causal Marketing & Uplift Allocator:**
   - Donut chart: Distribution of households across the 4 uplift segments (`Persuadables`, `Sure Things`, `Sleeping Dogs`, `Lost Causes`).
   - Bar chart: Channel Attribution share comparison (Markov Removal Effect vs First/Last/Linear Touch).
   - Net profit curve by targeting percentile to guide marketing spend thresholds.

3. **Merchandise Affinity & Category Early Warning:**
   - Horizontal bar chart of top early-warning precursor categories (chicken/poultry, beef, diapers, infant formula).
   - Filterable table of top association rules with Antecedent $\rightarrow$ Consequent relationships.
