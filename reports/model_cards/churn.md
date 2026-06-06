# Model Card: Customer Churn Early-Warning Predictor

## 1. Model Details
- **Model Name:** Retention Radar Churn Predictor (LightGBM Gradient Boosted Trees)
- **Model Version:** 1.0.0
- **Model Type:** Gradient Boosted Decision Tree with Isotonic Probability Calibration
- **Primary Optimization Objective:** Out-of-time Validation PR-AUC (Average Precision) via Optuna Bayesian Optimization
- **Developed by:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)
- **Release Date:** May 2026
- **License:** MIT

## 2. Intended Use
- **Primary Use Case:** Rank active Dunnhumby grocery households by probability of zero transactions in the subsequent 8 weeks (2 consecutive 28-day periods), enabling proactive targeted retention interventions.
- **Out-of-Scope Use Cases:** Pricing discrimination, individual credit scoring, or automated punitive actions. Not intended for real-time streaming scoring (batch scored every 28 days).

## 3. Data & Rolling-Origin Partitions
- **Training Set (Cutoffs 6–15):** 23,468 household-cutoff observations.
- **Validation Set (Cutoffs 16–17):** 4,677 household-cutoff observations (used for Optuna tuning and isotonic calibration).
- **Buffer Periods (18–19):** Unused buffer ensuring absolute temporal separation.
- **Test Set (Cutoffs 20–23):** 9,397 household-cutoff observations (strictly out-of-time evaluation).
- **Leakage Prevention:** Point-in-time feature extraction guarantees that only data observed prior to each cutoff period $t$ enters feature computation. A regression test verifies that injecting future transactions causes zero perturbation in historical feature matrices.

## 4. Benchmark Performance & Evaluation

Evaluated strictly out-of-time on the test partition (Cutoffs 20–23, $N = 9,397$, base churn rate = 7.58%). 95% Confidence Intervals computed via 1,000 bootstrap resamples.

| Model | PR-AUC | 95% CI (PR-AUC) | ROC-AUC | Brier Score | ECE (10-bin) | Top 10% Lift | 95% CI (Top 10% Lift) | Top 10% Capture | Top 20% Capture |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned + Isotonic)** | **0.3520** | **[0.3186, 0.3880]** | **0.8792** | **0.0574** | **0.0090** | **4.82x** | **[4.44x, 5.14x]** | **48.17%** | **72.19%** |
| Scaled Logistic Regression | 0.3204 | [0.2897, 0.3545] | 0.8726 | 0.0585 | 0.0103 | 4.51x | [4.16x, 4.86x] | 45.08% | 72.33% |
| Cox Proportional Hazards | 0.0907 | [0.0794, 0.1054] | 0.4023 | 0.5403 | 0.6647 | 1.91x | [1.65x, 2.19x] | 19.10% | 31.46% |
| Markov Lifecycle Score | 0.0758 | [0.0707, 0.0809] | 0.5000 | 0.0707 | 0.0258 | 1.03x | [0.80x, 1.22x] | 10.25% | 19.94% |

### Key Performance Findings
1. **LightGBM Superiority:** LightGBM achieves a PR-AUC of 0.3520, statistically significantly outperforming Logistic Regression (0.3204) and decisively beating heuristic Markov (0.0758) and survival Cox (0.0907) baselines.
2. **Top-Decile Concentration:** Targeting the top 10% highest-risk households yields an enrichment of **4.82x** over baseline, successfully intercepting **48.17%** of all future churners. Expanding targeting to the top 20% intercepts **72.19%** of all churners.
3. **Probability Calibration:** Isotonic regression reduces Expected Calibration Error (ECE) to **0.0090 (< 1%)**, ensuring predicted probabilities accurately reflect actual empirical lapse frequencies.

## 5. Profit Curve & Decision Optimization

Intervention Expected Value ($EV$) is formulated as:
$$EV_i = p_i \times \text{save\_rate} \times \text{CLV}_i - \text{offer\_cost}$$

Assuming standard retail retention parameters ($\text{offer\_cost} = \$2.00$, $\text{save\_rate} = 10\%$):
- Contacting households with $p_i \ge 0.08$ (the top 20% risk deciles) maximizes net marketing contribution.
- Sensitivity analysis across the save rate grid $[5\%, 10\%, 15\%, 20\%]$ demonstrates positive expected ROI across all regimes, with optimal targeting coverage expanding from 12% (at 5% save rate) to 32% (at 20% save rate).

## 6. Fairness & Demographic Bias Audit

Evaluation of model recall and false positive rates (FPR) across demographic dimensions reveals substantial raw disparities between households with known demographic profiles ($N = 3,204$, 34% coverage) and households without demographic profiles ($N = 6,193$, 66% coverage):
- Households without loyalty card demographic records exhibit a baseline churn rate of 10.87%, compared to only 1.22% for households with completed profiles.
- As a consequence, demographic features act as an uninformative proxy for missingness rather than intrinsic risk.
- **Fairness Intervention:** An ablation model trained **without demographic features** was evaluated. Removing demographic variables improves test PR-AUC from **0.3520 to 0.3627 (+0.0107)** while eliminating demographic proxy bias.
- **Recommendation:** Deploy the behavioral-only model configuration to ensure fair, invariant customer treatment across demographic segments.

## 7. Model Explainability & Reason Codes
SHAP TreeExplainer analysis highlights the following top churn drivers:
1. `recency_days`: Days since last store visit.
2. `trips_1p` / `trips_3p`: Trailing visit frequency over recent 28-day cycles.
3. `spend_3p`: Multi-period spend volume deceleration.
4. `ols_spend_slope_6p`: Trailing 6-period spend trend slope ($/period).
5. `category_entropy_6p`: Purchase department concentration.

Each scored household is annotated with three personalized, human-interpretable reason codes (e.g., *"Declining spend trend (-$18.40/period slope)"*, *"High recency: no store visit in 26 days"*, *"Narrow basket variety"*).
