# Model Card: Customer Lifetime Value (CLV) Predictor

## 1. Model Details
- **Model Name:** Retention Radar Customer Lifetime Value Engine
- **Model Version:** 1.0.0
- **Model Architectures Benchmarked:**
  1. **LightGBM Tweedie Regressor:** Gradient boosted decision trees optimizing a compound Poisson-Gamma deviance loss ($1 < p < 2$).
  2. **BG/NBD + Gamma-Gamma:** Classical probabilistic Buy-Till-You-Die model (Beta-Geometric / Negative Binomial Distribution for transaction frequency, Gamma-Gamma for basket monetary spend).
  3. **Markov Reward Process:** Fundamental matrix reward formulation ($V = (I - \gamma Q)^{-1} r$).
- **Selected Champion Model:** LightGBM Tweedie Regressor
- **Prediction Target:** 26-week forward household gross margin contribution ($) evaluated strictly out-of-time on holdout periods 20–25 (weeks 77–102).
- **Developed by:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)
- **Release Date:** May 2026
- **License:** MIT

---

## 2. Intended Use
- **Primary Use Case:** Predict forward 6-month gross margin dollars for each household to rank customer value, calculate individual expected value saved ($EV_i = p_i \times \text{save\_rate} \times \text{CLV}_i - \text{offer\_cost}$), and allocate retention campaign budgets efficiently.
- **Out-of-Scope Use Cases:** Real-time credit limits, dynamic personalized grocery pricing, or exclusionary marketing practices.

---

## 3. Data & Rolling-Origin Evaluation Partitions

- **Calibration / Training Window (Weeks 1–76 / Periods 4–19):** Historical purchase frequency, inter-purchase times, monetary value, category penetration, and discount sensitivity computed up to cutoff period 19.
- **Holdout Evaluation Window (Weeks 77–102 / Periods 20–25):** 26 consecutive weeks of out-of-time realized household gross margin dollars.
- **Leakage Prevention:** Features are extracted strictly on transactions occurring $\le$ day 532 (Period 19 end). No future transaction observations pollute the training feature matrix.

---

## 4. Benchmark Performance & Evaluation

Evaluated on the out-of-time 26-week holdout cohort ($N = 2,500$ households, mean realized 26-week gross margin = $244.57):

| Model Architecture | Spearman Rank Correlation | MAE ($ Margin) | RMSE ($ Margin) | Method Strengths & Limitations |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM (Tweedie)** | **0.9162** | **$228.71** | **$349.64** | **Superior rank ordering and monetary precision; captures non-linear category trends and promo elasticities** |
| Markov Reward Process | 0.6960 | $2,407.81 | $3,142.05 | Excellent long-run steady-state valuation; overestimates short-term 26-week horizon due to infinite-horizon formulation |
| BG/NBD + Gamma-Gamma | 0.6799 | $784.89 | $1,124.30 | Strong probabilistic foundation; assumes independent frequency and monetary processes, struggling with zero-inflated periods |

### Key Performance Findings
1. **Decisive Rank Correlation:** LightGBM Tweedie achieves an exceptional Spearman rank correlation of **0.9162**, substantially outperforming BG/NBD (0.6799) and Markov Reward (0.6960). In retention decisioning, accurate rank ordering is critical for prioritization.
2. **Handling Zero Inflation:** The Tweedie deviance parameter ($p = 1.5$) naturally accounts for customers with zero transactions in the forward window without requiring artificial two-stage hurdle architectures.
3. **Monetary Error:** LightGBM Tweedie achieves an out-of-time MAE of **$228.71**, providing realistic household-level dollar projections.

---

## 5. Feature Importance & Drivers of Lifetime Value

Tree-based feature importance indicates the primary drivers of forward customer margin:
1. `spend_3p`: Total margin spend in the trailing 12 weeks (most significant signal of ongoing household spending power).
2. `trips_3p`: Purchase visit frequency over the trailing 12 weeks.
3. `recency_days`: Days elapsed since the last transaction (strongest negative indicator; recency > 35 days severely depresses expected CLV).
4. `fresh_share_3p`: Proportion of basket spend dedicated to fresh meats, produce, and dairy (high fresh share strongly correlates with long-term retention).
5. `coupon_redempt_rate`: Moderate coupon engagement indicates price responsiveness and store loyalty without eroding baseline margins.

---

## 6. Integration into Retention Operations

The predicted 26-week CLV ($\text{clv\_26w}$) is directly integrated into the Retention Radar operational stack:
- **Profit Curve Weighting:** Multiplied by calibrated churn risk ($p_i$) and expected winback rate ($\delta$) to calculate net expected value creation.
- **Roster Prioritization:** High-CLV accounts ($\text{clv\_26w} \ge \$500$) are elevated to VIP Outreach tiers even if their baseline churn risk is moderate ($p_i \approx 0.15$), protecting the highest-margin customer relationships.
- **Drift Monitoring:** Monitored quarterly via Population Stability Index (PSI) to detect macro-economic spending shifts or inflation-induced basket drift.
