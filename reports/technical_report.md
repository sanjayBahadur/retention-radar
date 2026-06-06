# Retention Radar: Customer Lifecycle Analytics, Churn Prediction, and Causal Uplift Optimization

**Author:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)  
**Published:** June 2026  
**Repository:** `github.com/sanjayBahadur/retention-radar`  
**License:** MIT License  

---

## 1. Abstract

Customer defection in grocery retail is subtle, non-abrupt, and multi-phased. Traditional retention systems rely on lagging rule-based heuristics that detect churn only after customer relationships have severed, followed by mass promotional blasts that destroy retail margins on non-marginal shoppers while provoking churn among promotional-averse accounts.

This paper presents **Retention Radar**, an open-source customer retention and causal optimization platform developed on 2.6 million grocery transactions across 2,500 households over 102 weeks. The architecture couples discrete-time Markov chains with supervised machine learning, causal uplift modeling, and merchandise affinity networks:
1. **Early-Warning Basket Contraction (Observed Association):** Empirical basket analysis indicates that household defection is preceded by an 8-week precursor phase wherein households strip high-involvement meats (fresh poultry $-76.2\%$, beef $-64.0\%$) and infant/pet consumables (diapers $-66.1\%$, pet food $-74.0\%$) from their baskets while continuing baseline purchases of milk and bread.
2. **Predictive Churn Discrimination:** An Optuna-tuned LightGBM classifier with isotonic probability calibration achieves a Test PR-AUC of **0.3370** ($95\%$ household-clustered CI: $[0.2997, 0.3802]$), performing comparably to regularized logistic regression (**0.3199** $[0.2822, 0.3616]$) with overlapping confidence intervals, while providing calibration improvements (ECE **0.0091 vs 0.0104**) and top-decile lift gains (**4.64x vs 4.51x**). In the top 20% outreach bracket, LightGBM captures **73.88% of all churners**.
3. **Demographic Fairness & Post-Ablation Audit:** An equity audit revealed proxy bias caused by $68\%$ missingness in retail loyalty cards. Ablating demographic features in favor of pure transaction behavior maintained strong predictive performance (Test PR-AUC **0.3627**) and increased minority recall, though subgroup sample sizes are small (8–18 churners per demographic cell).
4. **Causal Uplift & Intervention Policy:** Benchmarked against randomized trials (Hillstrom $N=42,693$ and X5 RetailHero $N=60,000$), an observational S-Learner suggests a **3.61x ROI** ($+\$20,435$ net profit vs. $\$8,173$ random baseline) for targeting the top 30% uplift segment. We identify 202 accounts (2.1%) with negative point-estimate treatment effects ("Sleeping Dogs"), though we caution that observational Qini ($0.67$) indicates residual confounding relative to randomized benchmarks ($0.054$).
5. **Production Governance:** The pipeline enforces zero temporal leakage via rolling-origin cross-validation, validates outputs against 16 Pandera runtime schemas, achieves **93% test coverage** across 65 unit and integration tests, and monitors feature stability via Population Stability Index (PSI) drift tracking.

---

## 2. Business Problem & Economic Framing

Grocery retail operates on slim operating margins ($20\% - 30\%$). In this environment, acquiring a new customer costs 5 to 7 times more than retaining an existing customer. However, enterprise retention programs typically encounter three structural failures:
- **Lagging Indicator Trap:** Defining churn as "zero transactions in 30 days" identifies defection only after customer habits have migrated to a competing grocer.
- **Margin Erosion from Sure Things:** Sending blanket promotional discounts rewards loyal shoppers who would have purchased full-price items regardless.
- **Attrition Acceleration from Sleeping Dogs:** Contacting promotional-averse customers triggers notification fatigue, unsubscriptions, and defection.

We formalize retention intervention as an **Expected Value ($EV$) Optimization Problem**:
$$\max_{\pi} \sum_{i=1}^N \pi_i \cdot \left( p_i \cdot \delta \cdot \text{CLV}_i - c \right)$$
where:
- $\pi_i \in \{0, 1\}$ is the binary outreach targeting decision for household $i$.
- $p_i = P(\text{Churn}_i = 1 \mid X_i)$ is the calibrated forward churn probability.
- $\delta \in [0, 1]$ is the empirical winback save rate (set to $\delta = 0.10$ `# assumption`).
- $\text{CLV}_i$ is the predicted 26-week customer margin lifetime value ($).
- $c$ is the unit communication and coupon fulfillment cost ($c = \$2.00$ `# assumption`).

Targeting is financially justified if and only if:
$$p_i \ge \frac{c}{\delta \cdot \text{CLV}_i}$$

---

## 3. Data Warehouse Architecture & Leakage Prevention

### 3.1 Datasets
- **Dunnhumby "The Complete Journey":** 2,500 frequent-shopper households over 102 consecutive weeks (700 days, 2,595,732 transaction line items, 272,110 baskets, 92,339 products, 30 direct marketing campaigns).
- **Kevin Hillstrom Email Marketing Benchmark:** 64,000 customers in a 3-arm randomized controlled trial (Men's E-Mail, Women's E-Mail, No E-Mail Control) used for causal algorithm verification.
- **X5 RetailHero Benchmark:** 60,000 randomized retail customer records utilized for confounding stress-testing.

### 3.2 Warehouse Modeling via dbt & DuckDB
The warehouse is modeled in DuckDB using dbt-core with automated schema and referential integrity tests:
- **Staging (`raw` $\rightarrow$ `stg`):** Type casting, surrogate hashing, and date alignment (`stg_transactions`, `stg_demographics`, `stg_products`, `stg_campaigns`).
- **Intermediate (`int`):** Aggregation of basket totals, discount depths, and department shares (`int_basket_summaries`, `int_customer_weekly_metrics`).
- **Marts (`dim`, `fct`):** Dimension tables (`dim_household`, `dim_product`) and dimensional facts (`fct_baskets`, `fct_campaign_exposure`, `fct_household_period_state`).

### 3.3 Strict Temporal Rolling-Origin Partitions
To eliminate temporal data leakage, transaction history is partitioned into 25 discrete 28-day periods (700 days):
- **Warm-Up (Periods 1–3, Days 1–84):** History accumulation to eliminate left-censoring.
- **Training Origin Cutoffs (Periods 6–15, Days 168–420):** 23,468 household-cutoff observation pairs.
- **Validation Cutoffs (Periods 16–17, Days 421–476):** 4,677 observation pairs for Optuna tuning and isotonic calibration.
- **Buffer Zone (Periods 18–19, Days 477–532):** Two 28-day periods left unallocated to guarantee zero label overlap between validation and testing.
- **Out-of-Time Test Cutoffs (Periods 20–23, Days 533–644):** 9,397 observation pairs evaluated strictly out-of-time.
- **Holdout CLV Horizon (Periods 20–25, Days 533–700):** 26 weeks of observed forward gross margin.

```
Timeline:
[P1-P3: Warm-up] -> [P4-P15: Train Cutoffs 6-15] -> [P16-P17: Val] -> [P18-P19: Buffer] -> [P20-P23: Test Cutoffs] -> [P24-P25]
```

A dedicated automated regression test verifies that artificially appending future transaction records to the warehouse results in exactly zero modification to pre-cutoff historical feature matrices.

---

## 4. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted across all 2,500 households and 272,110 multi-item baskets:
- **Total Gross Revenue:** $\$7,921,184.70$ with an average basket spend of $\$29.11$ and mean spend per household of $\$3,168.47$.
- **Purchase Rhythm:** Median visit cadence of 4.35 trips per 28-day period with an exponential tail ($P90 = 8.2$ trips).
- **Revenue Inequality:** Lorenz curve analysis yielded a **Gini coefficient of 0.5034**, with the top $20\%$ of households generating **53.1% of total revenue** ($\$4,206,149$).
- **Baseline Churn Rate:** Across active households, the empirical 2-period churn rate averages **4.53%** per period (rising to $7.58\%$ across out-of-time test cutoff observations).
- **Demographic Missingness:** Only $32.04\%$ of households ($N = 801$) possess registered demographic profiles; $67.96\%$ of accounts shop anonymously without profile cards.

---

## 5. Mathematical Foundations of Markov Chains

### 5.1 State Definitions
Household health is modeled over 28-day periods across 6 discrete states:
1. **New:** First observed transaction occurs in the current period.
2. **Champion:** Active in the current period with period spend $\ge \$652.68$ ($75^{\text{th}}$ percentile threshold fit strictly on training periods 4–19).
3. **Regular:** Active in the current period with spend $< \$652.68$.
4. **At-Risk:** 0 transactions in the current period, but active in the immediately preceding period $t-1$.
5. **Dormant:** 0 transactions in periods $t$ and $t-1$, but active in period $t-2$.
6. **Lapsed:** 0 transactions for $\ge 3$ consecutive periods ($\ge 84$ days). Treated as an absorbing terminal state for theoretical fundamentals.

### 5.2 Transition Probability Matrix & Canonical Decomposition
The empirical transition matrix $P \in \mathbb{R}^{6 \times 6}$ is estimated via Maximum Likelihood Estimation with structural Laplace smoothing ($\alpha = 0.5$). Grouping transient states $T = \{\text{New, Champion, Regular, At-Risk, Dormant}\}$ and absorbing state $A = \{\text{Lapsed}\}$ yields the canonical form:
$$P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}$$
where $Q \in \mathbb{R}^{5 \times 5}$ governs transitions among transient states, and $R \in \mathbb{R}^{5 \times 1}$ governs absorption into Lapsed.

### 5.3 Fundamental Matrix & Expected Days to Lapse
The **Fundamental Matrix** $N \in \mathbb{R}^{5 \times 5}$ is computed as:
$$N = (I - Q)^{-1} = \sum_{k=0}^{\infty} Q^k$$
Each entry $N_{ij}$ denotes the expected number of 28-day periods a household starting in state $i$ will spend in state $j$ before absorption. Expected days to lapse is $T_{\text{days}} = 28 \cdot N \mathbf{1}$.

95% Confidence Intervals derived from 1,000 household block bootstrap resamples:
- **Champion:** Mean = **2,104.8 days** ($95\%$ CI: $[1,949.6, 2,288.1]$ days $\approx 5.77$ years).
- **Regular:** Mean = **1,904.8 days** ($95\%$ CI: $[1,747.7, 2,088.8]$ days $\approx 5.22$ years).
- **New:** Mean = **1,887.5 days** ($95\%$ CI: $[1,732.4, 2,072.8]$ days $\approx 5.17$ years).
- **At-Risk:** Mean = **1,617.1 days** ($95\%$ CI: $[1,468.5, 1,795.1]$ days $\approx 4.43$ years).
- **Dormant:** Mean = **1,208.3 days** ($95\%$ CI: $[1,076.9, 1,367.2]$ days $\approx 3.31$ years).

*Methodological Caveat:* These 3- to 5-year projections represent an analytical infinite-horizon steady-state extrapolation from a 2-year observed corpus. Furthermore, in empirical retail behavior, the absorbing state is an approximation: **20.3% of households who reach 3 missed periods ("Lapsed") do eventually return** in subsequent observation windows.

### 5.4 Markov Reward Process & Lifetime Value
Assigning empirical period gross margin rewards $r = (r_{\text{New}}, r_{\text{Champion}}, r_{\text{Regular}}, r_{\text{At-Risk}}, r_{\text{Dormant}})^T$ and applying the continuous cost of capital discount factor $\gamma = (1 + 0.10)^{-28/365} = 0.9927$ (`# assumption`), the infinite-horizon discounted state value vector $V$ satisfies the Bellman equation:
$$V = r + \gamma Q V \implies V = (I - \gamma Q)^{-1} r$$

Empirical State CLV Valuations:
- **Champion:** **$2,406.85**
- **Regular:** **$1,784.72**
- **New:** **$1,777.49**
- **At-Risk:** **$1,455.93**
- **Dormant:** **$1,077.89**
- **Lapsed:** **$0.00**

---

## 6. Algorithmic Multi-Touch Attribution

Traditional rule-based attribution introduces substantial distortion into marketing budget allocation:
- **First-Touch:** Attributes $100\%$ of credit to the first recorded campaign exposure.
- **Last-Touch:** Attributes $100\%$ of credit to the final campaign exposure preceding conversion.
- **Linear Touch:** Distributes credit equally across all touchpoints in the path.

### 6.1 Markov Removal Effect Formulation
We model multi-channel campaign exposure as a directed graph where nodes represent marketing touches (Campaign Types A, B, and C) and transitions reflect sequential customer interactions terminating in Conversion or Non-Conversion.  
The **Removal Effect** $R_k$ of channel $k$ measures the relative decrease in overall graph conversion probability when channel node $k$ is removed:
$$R_k = \frac{P(\text{Conversion}) - P(\text{Conversion} \setminus \{k\})}{P(\text{Conversion})}$$
Normalized channel credit is computed as $\alpha_k = \frac{R_k}{\sum_{j} R_j}$.

### 6.2 Empirical Channel Credit Comparison

| Campaign Type | Primary Channel Mechanics | Markov Removal Share | First-Touch Share | Last-Touch Share | Linear Share | Coupon Redemption Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Type A** | High-involvement mass circulars / mailers | **43.33%** | 66.00% | 33.54% | 55.60% | 43.10% |
| **Type B** | Targeted direct-mail loyalty coupons | **41.12%** | 31.99% | 58.89% | 37.16% | 41.40% |
| **Type C** | Small-scale targeted promotional reminders | **15.55%** | 2.01% | 7.57% | 7.24% | 15.50% |

**Key Finding:** First-touch severely over-credits Type A mass mailers (+22.7 percentage points) while under-crediting mid-funnel nurture touchpoints. Conversely, last-touch over-credits Type B direct coupons (+17.8 percentage points). The algorithmic Markov removal effect reveals that Type A ($43.3\%$) and Type B ($41.1\%$) are near-equal structural partners in driving conversion.

---

## 7. Customer Lifetime Value (CLV) Benchmarking

Three distinct CLV modeling architectures were benchmarked on an out-of-time 26-week holdout cohort (Periods 20–25 / Weeks 77–102, $N = 2,500$ households, realized mean 26-week gross margin = $\$244.57$):

| Architecture | Model Family | Holdout Spearman Rank Corr | Holdout MAE ($ Margin) | Holdout RMSE ($ Margin) | Decision Utility |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **LightGBM (Tweedie)** | Gradient Boosted Trees ($p=1.5$) | **0.9162** | **$228.71** | **$349.64** | **Optimal rank discrimination & monetary precision** |
| Markov Reward Process | Absorbing Chain $(I - \gamma Q)^{-1} r$ | 0.6960 | $2,407.81 | $3,142.05 | Measures steady-state infinite-horizon value |
| BG/NBD + Gamma-Gamma | Probabilistic BTYD | 0.6799 | $784.89 | $1,124.30 | Struggles with zero-inflated forward periods |

The LightGBM Tweedie regressor ($p = 1.5$) was selected as champion. By minimizing compound Poisson-Gamma deviance, it naturally accommodates zero-spend forward periods without arbitrary two-stage hurdle splits.

---

## 8. Predictive Churn Modeling & Calibration

### 8.1 Problem Formulation & Partitions
Churn is defined as a binary indicator:
$$Y_{i, t} = \mathbb{I}(\text{Spend}_{i, [t+1, t+2]} == 0)$$
reflecting complete inactivity across the forward 8 weeks. Models were trained across historical cutoffs 6–15 ($N = 23,468$), tuned on validation cutoffs 16–17 ($N = 4,677$), and evaluated strictly out-of-time on test cutoffs 20–23 ($N = 9,397$, base churn rate $7.58\%$).

### 8.2 Out-of-Time Benchmark Results (Household-Clustered CIs)

Confidence intervals are calculated using **household-clustered bootstrap resampling** (1,000 resamples clustered by `household_key`) to properly account for repeated measurements of the 2,500 households across the 4 test cutoffs:

| Model Architecture | Test PR-AUC | 95% Clustered CI | Test ROC-AUC | Brier Score | ECE (10-bin) | Top 10% Lift | Top 10% Capture | Top 20% Capture |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned + Calibrated)** | **0.3370** | **[0.2997, 0.3802]** | **0.8769** | **0.0580** | **0.0091** | **4.64x** | **46.35%** | **73.88%** |
| Scaled Logistic Regression | 0.3199 | [0.2822, 0.3616] | 0.8725 | 0.0585 | 0.0104 | 4.51x | 45.08% | 72.33% |
| Cox Proportional Hazards Baseline | 0.3278 | [0.2915, 0.3715] | 0.8671 | 0.0621 | 0.0472 | 4.55x | 45.51% | 69.94% |
| Markov Lifecycle Baseline | 0.2421 | [0.2108, 0.2764] | 0.7816 | 0.0606 | 0.0102 | 4.40x | 43.96% | 60.81% |

### 8.3 Scientific Interpretation of Results
1. **Model Parity with Logistic Regression:** LightGBM achieves a Test PR-AUC of **0.3370** vs. Logistic Regression's **0.3199**. Because their 95% clustered confidence intervals overlap substantially ($[0.300, 0.380]$ vs. $[0.282, 0.362]$), LightGBM should be viewed as performing **comparably to regularized logistic regression**. Its primary operational advantages lie in probability calibration (ECE **0.0091 vs 0.0104**) and top-decile lift (**4.64x vs 4.51x**).
2. **Competitive Survival and Markov Baselines:** Proper formulation of the Cox Proportional Hazards baseline (aligning duration to cumulative customer tenure) achieves ROC-AUC **0.8671** and PR-AUC **0.3278**, confirming that hazard models provide strong baseline ordering. Similarly, the 2-step Markov transition baseline achieves ROC-AUC **0.7816**, reflecting meaningful state-dependent lapse dynamics.
3. **Targeting Efficiency:** In the top 20% outreach bracket, LightGBM captures **73.88% of all churners**, identifying 5,600 at-risk households per 100,000 customer base.

---

## 9. Model Explainability & SHAP Reason Codes

Global feature importance was computed via SHAP TreeExplainer across the out-of-time test cohort (`reports/figures/churn_shap_beeswarm.png`):
- `recency_days`: Days elapsed since last transaction is the single strongest predictor. Households with recency $> 35$ days exhibit exponentially increasing churn odds.
- `spend_3p`: Trailing 12-week margin spend. Low spend sharply increases churn vulnerability.
- `trips_1p`: Transactions in the immediate past 28-day period. A collapse in visit frequency provides a rapid warning signal.
- `ols_spend_slope_6p`: Trailing spend trajectory slope over 6 periods. A steep negative trajectory flags active margin withdrawal.

Every household in `reports/churn_target_list.xlsx` is assigned its top-3 personalized SHAP reason codes for operational winback tailoring.

---

## 10. Demographic Fairness Audit & Post-Ablation Analysis

### 10.1 Baseline Fairness Audit (Top 20% Cutoff)
Evaluating baseline model predictions across demographic slices in the test partition ($N = 9,397$ total; $712$ churners):

| Dimension | Group | Sample Size | Churn Count | Churn Rate | Recall | FPR | Recall Gap | Disparity Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | All Households | 9,397 | 712 | 7.58% | 74.02% | 15.58% | 0.00% | False |
| **Demographic Status** | Known Demographics | 3,204 | 39 | 1.22% | 28.21% | 0.13% | 45.81% | **True** |
| **Demographic Status** | Unknown (No Profile) | 6,193 | 673 | 10.87% | 76.67% | 24.44% | 2.65% | False |
| **Income** | <$35K | 848 | 14 | 1.65% | 14.29% | 0.00% | 59.73% | **True** |
| **Income** | $35K-$75K | 1,456 | 10 | 0.69% | 30.00% | 0.28% | 44.02% | **True** |
| **Income** | $75K+ | 900 | 15 | 1.67% | 40.00% | 0.00% | 34.02% | **True** |
| **Age** | 19-34 | 752 | 8 | 1.06% | 0.00% | 0.27% | 74.02% | **True** |
| **Age** | 35-54 | 1,928 | 13 | 0.67% | 38.46% | 0.10% | 35.56% | **True** |
| **Age** | 55+ | 524 | 18 | 3.44% | 33.33% | 0.00% | 40.69% | **True** |

Because loyalty members who fill out demographic cards are disproportionately loyal (churn rate 1.22% vs. 10.87% for anonymous shoppers), inclusion of demographic features caused demographic indicators to act as proxies for low overall volume, artificially depressing recall on cardholders.

### 10.2 Post-Ablation Model (Without Demographics)
Retraining exclusively on behavioral transaction features:

| Dimension | Group | Sample Size | Churn Count | Churn Rate | Recall | FPR | Recall Gap | Disparity Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | All Households | 9,397 | 712 | 7.58% | 73.88% | 15.59% | 0.00% | False |
| **Demographic Status** | Known Demographics | 3,204 | 39 | 1.22% | 46.15% | 0.63% | 27.73% | **True** |
| **Demographic Status** | Unknown (No Profile) | 6,193 | 673 | 10.87% | 75.48% | 24.17% | 1.60% | False |
| **Income** | <$35K | 848 | 14 | 1.65% | 42.86% | 0.48% | 31.02% | **True** |
| **Income** | $35K-$75K | 1,456 | 10 | 0.69% | 30.00% | 0.62% | 43.88% | **True** |
| **Income** | $75K+ | 900 | 15 | 1.67% | 60.00% | 0.79% | 13.88% | **True** |
| **Age** | 19-34 | 752 | 8 | 1.06% | 12.50% | 0.40% | 61.38% | **True** |
| **Age** | 35-54 | 1,928 | 13 | 0.67% | 69.23% | 0.73% | 4.65% | **False** |
| **Age** | 55+ | 524 | 18 | 3.44% | 44.44% | 0.59% | 29.44% | **True** |

**Analytical Reality:** Removing demographics improved Known Demographics recall from **28.21% to 46.15%** (and age 35–54 recall from **38.46% to 69.23%**) while preserving overall PR-AUC (**0.3627**). However, because demographic subgroups contain only **8 to 18 churners each** (total 39 churners across 3,204 known demographic households), point estimates of subgroup recall are noisy, and claims of "eliminating proxy bias" must be tempered by these sample size realities.

---

## 11. Causal Uplift Modeling & Decision Governance

### 11.1 Benchmark Validation on Randomized RCTs
- **Hillstrom Email Marketing ($N = 42,693$):** CausalML T-Learner recovered an empirical visit rate difference of **+4.54%**, falling precisely within the 95% bootstrap CI ($[+3.86\%, +5.18\%]$).
- **X5 RetailHero ($N = 60,000$):** In an unconfounded randomized trial, our causal suite recovered the ground truth ATE of **+0.0377** (Qini = 0.054).
- **Confounding Stress Test:** Injected synthetic selection bias into the X5 cohort:
  - *Naive Difference in Means:* Estimate = $+0.0454$ (**Bias = +0.0078**; overstates effect by $+20.7\%$).
  - *Inverse Propensity Weighting (IPW):* Estimate = $+0.0362$ (Bias = $-0.0015$).
  - *Augmented IPW (AIPW / Doubly Robust):* Estimate = **+0.0376** (**Bias = -0.0001**; virtually zero bias).

### 11.2 Dunnhumby Observational Uplift Modeling
Across 21 eligible retention campaigns (28,334 trimmed household-campaign observations after common support filtering $0.05 \le e(X) \le 0.95$):
- *S-Learner (LightGBM):* **Qini = 0.6706**, **AUUC = 0.9820**, Top 30% CATE = **+$0.4758/day**.
- *T-Learner:* Qini = 0.2526, AUUC = 0.7210.
- *DR-Learner:* Qini = 0.2060, AUUC = 0.6955.
- *X-Learner:* Qini = 0.0754, AUUC = 0.5565.

### 11.3 Customer Uplift Segmentation & Observational Limitations

Applying heuristic thresholds ($\pm \$0.15/\text{day}$) to CATE point estimates yields:
- **Persuadables:** 8,860 accounts (93.8%), mean $\hat{\tau} = +\$1.20$/day.
- **Sure Things:** 246 accounts (2.6%), mean $\hat{\tau} = +\$0.00$/day.
- **Sleeping Dogs:** 202 accounts (2.1%), mean $\hat{\tau} = -\$0.60$/day.
- **Lost Causes:** 137 accounts (1.5%), mean $\hat{\tau} = +\$0.02$/day.

**Critical Methodological Limitations:**
1. **Residual Confounding:** The observational Qini of **0.6706** is an order of magnitude higher than the randomized benchmark (X5 Qini **0.054**). This discrepancy strongly indicates residual unobserved confounding in campaign targeting.
2. **Heuristic Segments:** The 93.8% "Persuadable" clustering and 2.1% "Sleeping Dog" cohort arise from heuristic $\pm 0.15$ point-estimate cutoffs rather than individual statistical significance.
3. **Targeting ROI:** While the top 30% uplift policy generates **$20,435 in net profit** compared to $\$8,173$ under random outreach (a 3.61x efficiency ratio), this policy must be validated via an RCT before operational rollout.

---

## 12. Market Basket Analysis & Early-Warning Categories

### 12.1 Association Rule Mining
Using FP-Growth on 12,849 multi-item baskets (`min_support = 0.01, min_lift = 1.20`), we mined **27,312 association rules** (`reports/figures/basket_rules_network.png`):
- Top Complementary Pairs: `(Dry Noodles/Pasta, Soft Drinks) -> (Baked Bread/Buns/Rolls, Pasta Sauce)` with $\text{Lift} = 11.85$.
- Network topology reveals that Fluid Milk, Baked Bread, and Soft Drinks act as central hubs connecting distinct department clusters.

### 12.2 Precursor Category Abandonment (Observed Association)
Evaluating commodity spend share changes during the 6 periods preceding dormancy ($[t^*-5, t^*-3]$ vs $[t^*-2, t^*]$) against matched active controls:

| Rank | Merchandise Commodity | Lapsing Households | Pre-Lapse Spend Change | Matched Control Change | Net Drop Effect |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Diapers & Disposables** | 43 | -66.1% | +0.4% | **-3.61%** |
| **2** | **Chicken / Poultry** | 88 | -76.2% | -1.2% | **-3.33%** |
| **3** | **Dog Foods** | 51 | -74.0% | +0.2% | **-2.92%** |
| **4** | **Beef** | 343 | -64.0% | -0.8% | **-2.69%** |
| **5** | **Infant Formula** | 26 | -66.4% | +0.1% | **-2.12%** |

*Interpretation:* The steep decline in fresh proteins and infant consumables provides an empirical early-warning indicator. While observational data cannot confirm whether households switched to competitor grocers or altered consumption habits, these basket shifts provide category merchandising teams with an operational lead time of 6 to 8 weeks.

---

## 13. Production Engineering, Monitoring & Governance

### 13.1 Schema Contracts & Data Quality
16 strict `DataFrameSchema` definitions validate all Parquet datasets:
- Enforce strict typing, non-nullability on primary keys, and logical bounds ($p \in [0, 1]$, CIs ordered, lift $\ge 1.0$).

### 13.2 Automated Drift Monitoring via Population Stability Index (PSI)
Population Stability Index is monitored across 11 features:
- **10 of 11 features are STABLE** ($\text{PSI} < 0.10$).
- Scheduled retraining cadence set to every 3 periods (84 days) or immediately if churn score $\text{PSI} \ge 0.20$.

### 13.3 Unified Pipeline Runner & Test Suite
The full pipeline executes via `python -m retention_radar.pipeline`:
- **65 unit and integration tests** pass in **29.35 seconds**.
- Test coverage: **93%** across the entire codebase (`ingest.py` 77%, `pipeline.py` 90%, `churn/train.py` 85%, `features/build_features.py` 71%).

---

## 14. Consumption Layer: Dashboards & BI Package

1. **Streamlit Executive Command Center:** 6 interactive operational pages (`app/Home.py`, `1_Lifecycle.py`, `2_Retention_Simulator.py`, `3_Churn_Risk.py`, `4_Campaigns.py`, `5_Baskets.py`, `6_Methodology.py`).
2. **Business Intelligence Package (`bi/`):** Clean CSV exports in `data/outputs/bi/`, star schema documentation, and 10 production DAX and Tableau calculated fields.
3. **Interactive HTML Report (`docs/index.html`):** Standalone zero-dependency analytical briefing.

---

## 15. Limitations & What I Would Do Differently

Senior practitioners recognize that no analytical architecture is without caveats. In a production enterprise deployment, the following areas require refinement:

1. **Unobserved Competitor Switching:** The Dunnhumby corpus contains transactions from a single retail chain. When a household ceases buying fresh chicken or diapers, we observe category attrition within our stores, but cannot observe whether they shifted spend to a discount grocer, a specialty butcher, or simply altered dietary habits. Integrating third-party panel data (e.g. Numerator, Circana) is essential to confirm true competitor switching.
2. **Observational Uplift Confounding:** The S-Learner Qini of 0.67 on observational data is implausibly high compared to randomized benchmarks (X5 Qini 0.054). In commercial practice, marketing teams target promotions to responsive accounts, creating selection bias that even propensity trimming and doubly robust estimators cannot fully remove. Uplift models should never be deployed for negative suppression without continuous randomized holdout experiments.
3. **Markov Absorbing State Violation:** Modeling "Lapsed" as an absorbing state is a mathematical simplification. In reality, 20.3% of lapsed households return in later cycles. Transitioning from an absorbing Markov chain to a semi-Markov or non-homogeneous recurrent process would better capture winback behavior.
4. **Demographic Sample Sizes:** While removing demographic features eliminates proxy bias and increases recall on loyalty cardholders, the demographic subsets contained only 8 to 18 churners each. Rigorous fairness audits require stratified sampling or larger cohorts to attain adequate statistical power.
5. **Dynamic Sequential Optimization:** Currently, churn scoring and uplift targeting operate as single-period cross-sectional interventions. Real-world customer management involves multi-period coupon sequences, which are better modeled via Constrained Markov Decision Processes (CMDPs) or contextual bandits.

---

## 16. References

1. Dunnhumby. (2013). *The Complete Journey: Household-Level Transaction and Promotion Corpus*.
2. Hillstrom, K. (2008). *The MineThatData E-Mail Analytics and Customer Lifetime Value Challenge*.
3. X5 Retail Group. (2019). *RetailHero Uplift Modeling Competition Benchmark*.
4. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model. *Marketing Science*, 24(2), 275-284.
5. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 4765-4774.
6. Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning. *Proceedings of the National Academy of Sciences (PNAS)*, 116(10), 4156-4165.
7. Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules. *VLDB*, 487-499.
