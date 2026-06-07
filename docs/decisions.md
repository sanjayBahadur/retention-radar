# Decisions

Every judgment call, with the alternative rejected and why. Agents add entries as they make decisions; the human records checkpoint approvals here.

## Decisions made during planning (2026-11-01)

| # | Decision | Alternative rejected | Why |
| --- | --- | --- | --- |
| 1 | Use X5 RetailHero (randomized) as causal ground truth for uplift; Hillstrom for tests | Uplift on Dunnhumby alone | Dunnhumby targeting is behavioral (Type A coupons are personalized), so it can't prove causal effects alone |
| 2 | Derive coupon use from transaction discount fields | `coupon_redempt` as the main source | The redemption table covers far fewer households |
| 3 | Periods 1–3 are warm-up only | Treat day 1 as everyone's start | Most households were customers before the data starts (left-censoring) |
| 4 | Markov churn score = P(Dormant or Lapsed at t+2) | P(Lapsed within 2 periods) | Matches the ML label exactly, so the comparison is fair |
| 5 | Buffer periods 18–19 between validation and test | Contiguous split | Two-period labels would otherwise overlap the test window |
| 6 | CausalML for learners and metrics; no scikit-uplift | scikit-uplift | Its last release is stale against current scikit-learn |
| 7 | dbt-core 1.12 (`<2`) | dbt 2.0 | Mature DuckDB adapter; dbt 2.0 rejects some 1.x YAML habits |
| 8 | Makefile + `pipeline.py`; Dagster, MLflow, Docker, Quarto optional | Full orchestration stack by default | Target analyst postings don't name them; they add setup risk |
| 9 | Power BI or Tableau dashboard is required | Streamlit only | Power BI appears in Publix, Target and Kroger 84.51° postings |
| 10 | Python 3.12 | 3.13 | PyMC-Marketing needs 3.12+; CausalML wheels stop at 3.12 |

## Open inputs from the human

- [x] Operating system → Linux environment; `bi_tool: tableau` with full Power BI CSV exports and cross-platform instructions provided
- [x] Gross margin assumption: 0.25 (25% retail standard grocery margin placeholder)
- [x] Dunnhumby dataset loaded via local archive / Kaggle mirror
- [ ] Databricks Free Edition account (Phase 13 scale track stretch only)

## Decisions made during the build

| Date | Phase | Decision | Alternative rejected | Why |
| :--- | :---: | :--- | :--- | :--- |
| 2026-04-10 | 1 | Seeded 60-household sample for fixture | 200-household unconstrained sample | Restricts committed test fixture to 9.05 MB (under 10 MB limit) while retaining all table relationships |
| 2026-04-17 | 2 | Approved lifecycle state definitions and Champion spend threshold | Dynamic per-period spend cutoffs | P75 cutoff ($652.68) fit strictly on train periods 4–19 prevents temporal data leakage; max state share is 62.31% (well within 70% limit) |
| 2026-04-17 | 2 | Permitted empirical non-absorbing transitions from Lapsed state | Enforcing strict permanent absorption | Empirical return rate from Lapsed is 20.29%; true grocery customers return after extended absences |
| 2026-05-12 | 7 | Selected LightGBM Tweedie regressor as primary 26-week CLV scoring model | BG/NBD + Gamma-Gamma | LightGBM Tweedie achieved superior holdout Spearman rank correlation (0.9162 vs 0.6046 for BG/NBD and 0.6960 for Markov) and lower MAE ($228.71 vs $811.67) |
| 2026-05-19 | 6 | Approved LightGBM churn model, top-20% contact threshold ($p \ge 0.08$), and behavioral-only fairness configuration | Logistic regression and inclusion of noisy demographics | LightGBM achieves superior PR-AUC (0.3520 [0.3186, 0.3880]), 4.82x top-decile lift (capturing 48.2% of churners), 72.2% top-20% capture, and ECE of 0.0090; removing demographics boosts PR-AUC to 0.3627 and eliminates disparity from 68% missingness in loyalty profiles |
| 2026-05-26 | 8 | Selected top-30% uplift targeting policy and S-Learner / DR-Learner causal suite | Blanket targeting or naive response modeling | Top-30% policy yields 3.61x ROI ($20,435 net profit vs $8,173 random) while suppressing 202 identified Sleeping Dogs; IPW reduces covariate SMDs to < 0.064, and AIPW eliminates selection bias (+0.0376 vs +0.0377 ground truth) in confounding stress test |
| 2026-05-29 | 9 | Filtered non-merchandise departments and identified fresh/staple categories as early-warning churn signals | Raw unconstrained department association mining | Eliminates administrative kiosk/coupon distortions; empirical spend analysis shows lapsing households sharply drop center-of-plate fresh meats (Beef/Chicken) and baby/pet essentials (Diapers/Formula/Dog Food) 2–3 periods before overall dormancy |
| 2026-06-02 | 10 | Established automated PSI drift monitoring with quarterly retraining cadence | Ad-hoc or purely time-based retraining without drift metrics | Enforces retraining trigger when churn score PSI > 0.20 or when $\ge 2$ core behavioral features exhibit significant drift (PSI > 0.20); scheduled retraining cadence set to every 3 periods (84 days) |
| 2026-06-05 | 11 | Approved 6-page interactive Streamlit command center and dual Tableau / Power BI starter package | Streamlit only without external BI exports | Enables both direct exploratory executive dashboarding and enterprise deployment into Tableau / Power BI with 10 production DAX and Tableau calculated measures |
| 2026-06-07 | 12 | Final Human Checkpoint: Approved comprehensive technical report, executive summary, and interview pack | Summary notes without academic-grade technical specification | Fully documents mathematical formulations, causal assumptions, model cards, and quantified ROI ($+\$432\text{k}$ net margin per 100k households) grounded strictly in verified `results.json` |
