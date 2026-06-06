# Model Card: Causal Uplift & Retention Experiment Design

## 1. Model Details
- **Model Name:** Retention Radar Causal Uplift Estimator (CausalML Meta-Learner Suite)
- **Model Version:** 1.0.0
- **Model Type:** Causal Machine Learning (S-Learner, T-Learner, X-Learner, Doubly Robust DR-Learner) with LightGBM base estimators
- **Developed by:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)
- **Release Date:** May 2026
- **License:** MIT

## 2. Intended Use & Executive Objective
- **Primary Objective:** Estimate heterogeneous treatment effects (Conditional Average Treatment Effect, $\text{CATE} = \tau(X) = \mathbb{E}[Y(1) - Y(0) | X]$) to identify which retail grocery households genuinely increase purchase velocity when contacted with a promotional offer, versus households who would purchase regardless (*Sure Things*) or decrease engagement (*Sleeping Dogs*).
- **Target Application:** Direct-to-consumer personalized promotion targeting, campaign spend optimization, and A/B test validation.

## 3. Data & Multi-Dataset Validation Framework

Because Dunnhumby campaign exposure is observational rather than randomized (promotional mailers were historically targeted based on past purchase behavior), this analysis follows a three-part causal validation strategy:

1. **Hillstrom Benchmark:** Validation on a classic randomized email marketing trial ($N = 42,693$). Verifies that the meta-learner pipeline recovers the empirical randomized difference in visit rate ($+4.52\%$, 95% CI: $[+3.86\%, +5.18\%]$) without estimation bias ($\text{Estimated ATE} = +4.54\%$).
2. **Part A — Randomized Ground Truth on X5 RetailHero ($N = 60,000$):**
   - Validated randomized trial with treatment share = 49.98% ($|SMD| < 0.023$ across all customer pre-communication attributes).
   - Ground truth empirical uplift: **+3.77 percentage points**.
   - Evaluated S-, T-, X-, and DR-learners. DR-Learner achieved top performance (Qini = 0.0541, AUUC = 0.5505, Top-10% Uplift = +4.70%).
3. **Part C — Confounding Stress Test:**
   - Synthetically introduced severe tenure-based selection bias into X5 to evaluate debiasing power under unobserved confounding.
   - **Naive Difference in Means:** Estimated +4.54% (**Bias = +0.78 percentage points**).
   - **Inverse Propensity Weighting (IPW):** Estimated +3.62% (**Bias = -0.15 percentage points**).
   - **Augmented IPW (AIPW / Doubly Robust):** Estimated +3.76% (**Bias = -0.01 percentage points**), recovering the true ground truth effect.
4. **Part B — Observational Uplift on Dunnhumby Complete Journey ($N = 46,396$ units across 21 campaigns):**
   - Units: 21 eligible campaigns with $\ge 100$ treated and $\ge 100$ control households in the 56-day pre-window.
   - Propensity modeling with overlap trimming $[0.05, 0.95]$: Retained 28,334 units.
   - **IPW Balance:** Reduced covariate Standardized Mean Differences from as high as 0.3903 down to strictly $< 0.064$ (pre-spend SMD: 0.0005, pre-trips SMD: 0.0003), satisfying the balance criterion ($|SMD| < 0.10$).

## 4. Benchmark Performance & Learner Comparison

Evaluated on Dunnhumby holdout test households via 3-fold GroupKFold by household (preventing cross-campaign household data leakage):

| Meta-Learner | Qini Score | AUUC | Uplift at Top 10% ($/day) | Uplift at Top 30% ($/day) | Mean CATE ($/day) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S-Learner (LightGBM)** | **0.6706** | **0.9820** | **+$0.9266** | **+$0.4758** | **+$0.5583** |
| T-Learner (LightGBM) | 0.2526 | 0.7210 | +$0.4003 | +$0.3154 | +$0.6544 |
| DR-Learner (Doubly Robust) | 0.2060 | 0.6955 | -$0.0708 | +$0.2361 | +$0.6828 |
| X-Learner (LightGBM) | 0.0754 | 0.5565 | +$0.3134 | +$0.1145 | +$0.6739 |

## 5. Customer Uplift Segmentation

Households are partitioned into four behavioral archetypes based on treatment effect $\tau_i$ and baseline non-promoted purchase velocity:

| Segment | Household Count | Share | Mean Uplift ($\Delta$ Spend/Day) | Avg 26-Week CLV | Strategic Retention Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Persuadables** | 8,860 | 93.8% | +$0.6068/day | $332.82 | **Target:** Highly responsive; generates substantial positive incremental margin. |
| **Sure Things** | 246 | 2.6% | +$0.0227/day | $562.54 | **Suppress:** Loyal, high-spend households ($14.73/day baseline); avoid subsidizing organic demand. |
| **Sleeping Dogs** | 202 | 2.1% | -$0.5984/day | $584.98 | **Do Not Contact:** High-value customers alienated by irrelevant coupon communication. |
| **Lost Causes** | 137 | 1.5% | +$0.0874/day | $249.59 | **Deprioritize:** Low baseline activity ($2.46/day); promotional cost exceeds incremental yield. |

## 6. Targeting ROI Curve

Across targeting depths from 10% to 100% of the customer base ($\text{offer\_cost} = \$2.00$, $\text{gross\_margin} = 25\%$):
- **Top 10% Policy:** Delivers **5.52x ROI** ($10,428 net profit vs $2,723 for random selection).
- **Top 30% Policy:** Yields **3.61x ROI** ($20,435 net profit vs $8,173 for random selection), capturing 60% of total possible incremental gross profit.
- **Blanket (100%) Targeting:** Net ROI falls to 1.44x due to coupon dilution and negative responses among *Sleeping Dogs*.

## 7. Confirming A/B Test Design

A power analysis was conducted via `statsmodels` to design a randomized controlled experiment validating the top-30% targeting policy against business-as-usual:
- **Primary Metric:** Incremental spend per day ($\Delta \text{spend/day}$).
- **Parameters:** $\alpha = 0.05$, Statistical Power ($1 - \beta$) = 0.80, two-sided independent $t$-test.
- **Minimum Detectable Effect (MDE):** +$0.4758 spend/day ($\text{Cohen's } d = 0.3656$).
- **Sample Size Required:** **119 households per arm** (238 total households).
- **Recommended Duration:** **8 weeks** (two full 28-day retail purchase cycles) to observe steady-state repeat frequency across active store patrons.

## 8. Causal Assumptions & Governance
1. **Conditional Unconfoundedness ($Y(0), Y(1) \perp T \mid X$):** Given pre-window shopping metrics (frequency, spend, discount share, coupon usage, concurrent campaigns), treatment assignment is conditionally independent of potential outcomes.
2. **Positivity / Common Support ($0.05 \le e(X) \le 0.95$):** All evaluated households possess a non-zero probability of receiving treatment or control.
3. **SUTVA:** No household spillover interference; grocery consumption within one household does not constrain adjacent household shopping choices.
