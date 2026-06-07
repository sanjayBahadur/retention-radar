# Retention Radar — Architecture & Engineering Blueprint

Comprehensive implementation blueprint, technical specifications, and phase acceptance criteria.
Version: 2026-06-07.

## Phase Execution Order

The engineering pipeline is organized into modular phases. Each phase represents a self-contained component with strict inputs, tasks, outputs, and automated validation criteria:

**Execution Order:** 0 → 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8 → 9 → 10 → 11 → 12.
Phase 7 (CLV) precedes Phase 6 (Churn) so the profit-curve optimizer can utilize household margin valuations.

**Core Engineering Principles**

1. Modular, reproducible components with clear interface boundaries.
2. Centralized configuration in `configs/project.yaml` and `configs/states.yaml`.
3. Deterministic execution with global random seed 42 across NumPy, scikit-learn, LightGBM, and Optuna.
4. Data governance: never commit raw data; commit pipeline code, configs, lightweight test fixtures, and aggregated outputs.
5. Zero temporal leakage: fit all transformers, scalers, and models strictly on historical training periods, evaluating out-of-time.
6. Every phase concludes with green `make lint` and `make test`, verified schema contracts, and atomic Conventional Commits.

**Definition of Done:** `make all` rebuilds the warehouse and artifacts deterministically on a clean environment. The Streamlit command center loads precomputed model outputs. Every metric in reports traces directly to verified artifacts in `data/outputs/results.json`.

## Project overview

**Retention Radar** uses two years of real household-level grocery transactions to answer five questions a retail loyalty team asks every quarter. Each answer is expressed in dollars of margin.

| # | Business question | Method | Output |
| --- | --- | --- | --- |
| 1 | How do customers move between loyalty states, and how fast do we lose them? | Markov chain on 28-day lifecycle states (absorbing + ergodic) | Transition matrix with CIs, expected days to lapse |
| 2 | What is each customer worth? | Markov reward process, BG/NBD + Gamma-Gamma | CLV per state and per household |
| 3 | Who will lapse in the next 8 weeks, and why? | LightGBM + SHAP, benchmarked against Markov and survival | Ranked risk list with plain-English reason codes |
| 4 | Which customers do campaigns actually change? | Uplift modeling validated on randomized data + A/B test design | Persuadable segment, targeting ROI, test sample size |
| 5 | Which campaigns and products drive purchases? | Markov removal-effect attribution, FP-Growth rules | Campaign credit, cross-sell and early-warning categories |

**Deliverables:** a reproducible pipeline, a Power BI or Tableau dashboard, a Streamlit simulator, a technical report, a one-page executive summary, model cards and a data-governance note.

**Out of scope:** real-time serving, paid cloud infrastructure, and any data beyond the anonymized datasets below.

**Architecture:** raw CSVs → DuckDB + dbt (staging → marts, 28-day states) → Python models (Markov, survival, CLV, churn ML, uplift, attribution, basket) → Parquet outputs + `results.json` → Streamlit app, BI dashboard, reports. A Makefile + `pipeline.py` runs every step; dbt tests and Pandera contracts guard each output.

## Datasets

| Dataset | Role | Size | Access | Terms | Phases |
| --- | --- | --- | --- | --- | --- |
| Dunnhumby The Complete Journey | **Core.** Household grocery transactions, campaigns, coupons, demographics | 2,595,732 lines, ~2,500 households, 711 days, 8 tables | Kaggle `frtgnn/dunnhumby-the-complete-journey`, or https://www.dunnhumby.com/source-files/ | Dunnhumby site terms: research, personal or non-commercial use only; no modification or redistribution | 1–9 |
| X5 RetailHero Uplift | **Required.** Randomized grocery campaign (50% treatment) with prior purchases: causal ground truth | 200,039 labeled clients, 400,162 clients, 45,786,568 purchase lines; 647 MB compressed, 4.17 GB raw | ods.ai competition page or the mirror URLs in the scikit-uplift `fetch_x5` source (download directly; do not install scikit-uplift) | Competition release; cite X5 Retail Group; don't redistribute | 8 |
| Hillstrom MineThatData Email | **Required.** Small 3-arm randomized test for unit-testing uplift code | 64,000 customers, ~4 MB | One CSV from minethatdata.com | Public challenge data; cite Hillstrom | 8 (tests) |
| Criteo Uplift v2.1 | Optional large randomized benchmark | 13,979,592 rows | Criteo AI Lab or Kaggle mirror | Research use | 8 (stretch) |
| Dunnhumby Let's Get Sort-of-Real | Optional synthetic data at scale: 300M transactions, ~500,000 customers | 4.3 GB; 5K and 50K-customer samples | Dunnhumby Source Files | Dunnhumby terms | 13 |
| UCI Online Retail II | Optional CLV/Markov replication you may publish freely | 1,067,371 lines | `ucimlrepo` id 502 | CC BY 4.0 | 4, 7 (replication) |
| Instacart 2017 | Optional basket analysis at scale | 3.4M orders | Instacart or Kaggle | Non-commercial only | 9 (stretch) |

**Data rules:** never commit raw Dunnhumby, X5 or Instacart files. The public dashboard ships aggregates and re-keyed IDs only. The README credits each source. Only Online Retail II rows may be republished. Use the Kaggle version of The Complete Journey only; the `completejourney` R/Python package ships a different 2017 extract (about 1.47M rows, 27 campaigns), and the two must not be mixed.

## Tech stack

Python 3.12 is pinned: PyMC-Marketing requires 3.12+, and CausalML 0.18 ships wheels for 3.11–3.12 only.

| Tier | Tools |
| --- | --- |
| Core (build first) | SQL in DuckDB ≥ 1.5 + dbt-core ≥ 1.12, < 2 with dbt-duckdb; Python (pandas, Polars, scikit-learn, LightGBM, Optuna, SHAP); Power BI (Windows) or Tableau Public (Mac); Git/GitHub; openpyxl Excel export; pytest, ruff, pre-commit, GitHub Actions |
| Differentiators | Markov chains in NumPy/SciPy (hand-written); CausalML ≥ 0.18 (meta-learners + Qini/AUUC); PyMC-Marketing ≥ 1.3 (BG/NBD, Gamma-Gamma); lifelines 0.30.x; statsmodels (power analysis); mlxtend + networkx; Pandera; Streamlit + Plotly |
| Optional polish | Dagster + dagster-dbt (`DbtProjectComponent`), MLflow, Docker, Quarto, mypy, hmmlearn 0.3.3 (limited maintenance), PySpark on Databricks Free Edition |

Implementation Notes:

- **Do not install scikit-uplift.** Its last release (0.5.1) is stale against current scikit-learn. Use CausalML for learners and metrics.
- **PyMC-Marketing 1.0 (2026-08-07) changed the API** and moved to PyMC 6. Data is passed to `fit()`, as in `BetaGeoModel().fit(data=rfm, method="map")`. Ignore pre-1.0 tutorials.
- **dbt 2.0** (Rust engine, built-in DuckDB) went GA on 2026-09-14 and rejects some 1.x YAML habits. Stay on dbt-core 1.12; to upgrade later, run `dbt-autofix` on a branch first.
- **Default orchestration** is `make all` calling `python -m retention_radar.pipeline`. Dagster is an optional wrapper.
- **Spark** is not used in the core pipeline (2.6M rows). It appears only in optional Phase 13.
- **Streamlit Community Cloud** (free): about 1 GB RAM, sleeps after 12 idle hours. Keep deployed outputs small.
- **Power BI Desktop** is Windows-only, and Publish to Web needs a work or Pro account. Tableau Public is free on Windows and Mac.

## Repository structure

Standard project layout:

```text
retention-radar/
├── PLAN.md                      # architecture blueprint
├── PHASE_LOG.md                 # audit log per completed phase
├── README.md
├── Makefile                     # setup, data, dbt, models, app, test, lint, all, clean
├── pyproject.toml / uv.lock
├── Dockerfile                   # optional polish
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── project.yaml             # paths, seed, periods, splits, business assumptions
│   └── states.yaml              # lifecycle state rules
├── data/                        # gitignored
│   ├── raw/                     # source CSVs + MANIFEST.json (SHA256, URL, date, license)
│   ├── warehouse.duckdb
│   └── outputs/                 # Parquet/JSON results the app and reports read (bi/ for BI CSVs)
├── tests/
│   ├── fixtures/sample/         # 200-household sample of all tables + full Hillstrom (committed)
│   ├── unit/
│   └── integration/
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/staging/          # stg_transactions, stg_products, stg_households, stg_campaigns, stg_coupons, stg_redemptions, stg_promotions
│   ├── models/intermediate/     # int_baskets, int_household_period_activity, int_coupon_transactions
│   ├── models/marts/            # dim_household, dim_product, fct_baskets, fct_household_period_state, fct_campaign_exposure
│   ├── analyses/                # business SQL showcase queries
│   └── tests/                   # custom singular tests
├── src/retention_radar/
│   ├── config.py
│   ├── ingest.py
│   ├── pipeline.py              # runs all phases in order
│   ├── markov/                  # chain.py, absorbing.py, reward.py, assumptions.py, simulate.py, hmm.py
│   ├── survival/                # km_cox.py
│   ├── attribution/             # markov_attribution.py, heuristics.py
│   ├── features/                # build_features.py
│   ├── churn/                   # train.py, evaluate.py, explain.py, fairness.py, score.py
│   ├── clv/                     # btyd.py, compare.py
│   ├── uplift/                  # propensity.py, learners.py, evaluate.py, power.py, x5.py
│   ├── basket/                  # rules.py, early_warning.py
│   ├── monitoring/              # drift.py
│   ├── schemas.py               # Pandera schemas for every output
│   └── reporting/               # figures.py, numbers.py (exports results.json)
├── orchestration/definitions.py # optional Dagster assets
├── app/                         # Streamlit Home.py + pages/
├── bi/                          # Power BI .pbix or Tableau .twbx + PDF export
├── notebooks/                   # narrative only; import from src
├── reports/                     # technical_report, executive_summary, model_cards/, figures/, sql_results/
└── docs/                        # data_dictionary.md, decisions.md, data_governance.md, architecture.png
```

## Phase 0 — Environment and scaffolding

**Goal:** a reproducible repo skeleton whose CI passes before any data work begins.

**Tasks**

1. Create the layout above with an MIT license. Add `data/`, `mlruns/` and `.env` to `.gitignore`.
2. Run `uv init` on Python 3.12. Runtime deps: duckdb>=1.5, dbt-core>=1.12,<2, dbt-duckdb, polars, pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, optuna, shap, lifelines, pymc-marketing>=1.3, causalml>=0.18, statsmodels, mlxtend, networkx, pandera, pydantic, pydantic-settings, pyyaml, plotly, streamlit, kaggle, jinja2, openpyxl, ucimlrepo. Optional `polish` group: mlflow, dagster, dagster-dbt, dagster-webserver, hmmlearn. Dev deps: pytest, pytest-cov, ruff, mypy, pre-commit, nbstripout, jupyter. Do not add scikit-uplift. Commit `uv.lock`.
3. Write `configs/project.yaml`. Values marked *assumption* are business placeholders and must be labeled as such in every output:
   - `seed: 42`, `period_days: 28`, `warmup_periods: 3`
   - `train_periods: [4, 19]`, `test_periods: [20, 25]`
   - `churn_horizon_periods: 2`, `lapse_missed_periods: 4`
   - `annual_discount_rate: 0.10` (*assumption*), `gross_margin: 0.25` (*assumption*, set by the human)
   - `offer_cost: 2.00` (*assumption*), `save_rate_grid: [0.05, 0.10, 0.15, 0.20]` (*assumption*)
   - `bootstrap_reps: 1000`, `smoothing_alpha: 0.5`, `optuna_trials: 50`, `bi_tool: powerbi | tableau` (set by the human)
   - a `ci` profile with reduced `bootstrap_reps` and `optuna_trials`
   - all paths under `paths:`
4. Write `config.py`, which loads the YAML into typed pydantic models and fails loudly on missing keys.
5. Write `Makefile` targets: `setup`, `data`, `dbt`, `models`, `app`, `test`, `lint`, `all`, `clean`.
6. Add pre-commit hooks: ruff, ruff-format, nbstripout (mypy optional).
7. Add `.github/workflows/ci.yml`: on push and PR, `uv sync`, `make lint`, `make test`, with uv cache.
8. Create `PHASE_LOG.md`, `docs/decisions.md` and `docs/data_governance.md` if missing.

**Acceptance criteria**

- [ ] On a clean clone, `make setup && make lint && make test` passes.
- [ ] CI is green with one placeholder test.
- [ ] `config.py` raises a clear error when a required key is removed (unit test).

## Phase 1 — Data ingestion and warehouse

**Goal:** all raw tables loaded into DuckDB, profiled, documented and checksummed, plus a small committed fixture for CI.

**Inputs:** Kaggle dataset `frtgnn/dunnhumby-the-complete-journey` (needs `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`, or the human places the CSVs in `data/raw/`); X5 RetailHero; Hillstrom.

**Expected Dunnhumby schema (verify against the actual files)**

| Table | Rows | Key columns |
| --- | --- | --- |
| transaction_data | 2,595,732 | household_key, basket_id, day, product_id, quantity, sales_value, store_id, retail_disc, trans_time, week_no, coupon_disc, coupon_match_disc |
| product | 92,353 | product_id, manufacturer, department, brand, commodity_desc, sub_commodity_desc, curr_size_of_product |
| hh_demographic | 801 | household_key, age_desc, marital_status_code, income_desc, homeowner_desc, hh_comp_desc, household_size_desc, kid_category_desc |
| campaign_desc | 30 | description (TypeA/B/C), campaign, start_day, end_day |
| campaign_table | 7,208 | description, household_key, campaign |
| coupon | 124,548 | coupon_upc, product_id, campaign |
| coupon_redempt | 2,318 | household_key, day, coupon_upc, campaign |
| causal_data | 36,786,524 | product_id, store_id, week_no, display, mailer |

**Tasks**

1. `ingest.py`: download via the Kaggle API when files are missing, verify all 8 files exist, and record URL, download date, SHA256 and license text per file in `data/raw/MANIFEST.json`.
2. Load each CSV into DuckDB schema `raw` with `read_csv_auto`, lowercase snake_case columns, and write Parquet copies. Load `causal_data` with DuckDB only, never pandas.
3. Download the official *The Complete Journey User Guide* PDF and use it as the authority for column meanings, especially `sales_value` and the discount columns.
4. Profile every column (row count, null %, min, max, distinct count) and auto-generate `docs/data_dictionary.md`, then add a human-readable description per column.
5. Run and log sanity checks: day range 1–711, week_no 1–102, discount sign convention (expected negative; confirm), share of rows with sales_value ≤ 0, baskets spanning more than one day.
6. Measure redemption coverage: `coupon_redempt` covers far fewer households than transaction rows with `coupon_disc < 0`. Count both and log the gap; Phases 5 and 8 depend on it.
7. Ingest X5 (`uplift_train`, `clients`, `purchases`) into schema `raw_x5`. Load `purchases` (45.8M rows) with DuckDB `read_csv` straight to Parquet. Verify row counts: 200,039 / 400,162 / 45,786,568.
8. Ingest Hillstrom (64,000 rows) into `raw_hillstrom`.
9. Build the fixture: a seeded 200-household sample across all Dunnhumby tables (filter `causal_data` to the fixture's products and stores), plus the full Hillstrom file, into `tests/fixtures/sample/`. Keep it under 10 MB.
10. Make ingestion idempotent: rerunning replaces tables, never appends.
11. Record the license each source shows at download time in `docs/data_governance.md`.

**Acceptance criteria**

- [ ] All tables are loaded, with row counts logged and compared to the expected counts.
- [ ] The data dictionary covers every column.
- [ ] A second `make data` run produces identical row counts.
- [ ] The fixture loads through the same code path in CI.

## Phase 2 — dbt transformation and lifecycle states

**Goal:** tested, documented marts, including one row per household per 28-day period with a lifecycle state.

**Tasks**

1. Configure a dbt-duckdb profile pointing at `data/warehouse.duckdb`, writing schemas `staging`, `intermediate`, `marts`.
2. Staging models: rename and cast; add `period_id = floor((day - 1) / 28) + 1`, giving 25 full periods (days 1–700). Drop days 701–711 as a partial period and log it. Add `gross_sales = sales_value - retail_disc - coupon_match_disc` only after confirming the sign convention; document the definition used.
3. Intermediate models:
   - `int_baskets`: one row per basket with household, day, period, store, spend, items, distinct departments, coupon-used flag, hour-of-day bucket.
   - `int_household_period_activity`: a dense household × period grid from each household's first purchase period through period 25, with trips, spend, items and consecutive missed periods.
   - `int_coupon_transactions`: transaction lines with `coupon_disc < 0`, joined to campaigns via `coupon` (product) and `campaign_desc` (day within the campaign window). This is the more complete redemption source.
4. Marts: `dim_household` (first and last purchase, tenure, demographics, `has_demographics`), `dim_product`, `fct_baskets`, `fct_campaign_exposure` (household, campaign, type, start_day, end_day), `fct_household_period_state`.
5. Assign lifecycle states at the end of each period, with rules in `configs/states.yaml`:

| State | Rule at end of period t |
| --- | --- |
| New | First-ever purchase falls in period t, and t > warm-up (periods 1–3) |
| Champion | Purchased in t; trailing 3-period spend in the top 25% of active households (cutoff fit on train periods only) |
| Regular | Purchased in t, and neither New nor Champion |
| At-Risk | No purchase in t; purchased in t−1 |
| Dormant | 2–3 consecutive missed periods |
| Lapsed | 4 or more consecutive missed periods |

6. Left-censoring: most households were customers before day 1, so periods 1–3 are warm-up only. They initialize history and trailing spend but are excluded from transition estimation and from the New state.
7. dbt tests: `unique` on (household_key, period_id); `not_null`; `accepted_values` for state; `relationships` to the dims. Singular tests: exactly one state per household-period after first purchase; state consistent with missed-period count. Transitions out of Lapsed are allowed and counted.
8. SQL showcase in `dbt/analyses/`: 6 queries using window functions and CTEs, each starting with a comment stating the business question — period revenue with change vs prior period, top-decile revenue share, cohort retention matrix, basket size by department, campaign reach by type, households whose trips fell more than 50% quarter over quarter. Keep the SQL ANSI-portable where practical.
9. Run `dbt docs generate` and save a lineage screenshot to `docs/`.

**Outputs:** marts, `data/outputs/state_distribution.parquet`, and the return rate from Lapsed logged in `PHASE_LOG.md`.

**Acceptance criteria**

- [ ] `dbt build` passes all tests on full data and on the fixture.
- [ ] The state distribution is exported, and no state exceeds 70% of households in any training period (if one does, propose a threshold change at the checkpoint).
- [ ] The return rate from Lapsed is logged.
- [ ] All 6 showcase queries run, with results saved as CSV in `reports/sql_results/`.
- [ ] **Human checkpoint:** state rules and thresholds are approved.

## Phase 3 — Exploratory analysis

**Goal:** a short narrative notebook establishing the facts later models depend on. Each section ends with a 1–2 sentence finding written from the actual numbers.

**Tasks**

1. Headline KPIs: households, baskets, revenue, average basket, trips per household per period.
2. Concentration: Lorenz curve and revenue share of the top 20% of households.
3. Purchase rhythm: inter-purchase-time distribution, basket size, day-of-week and hour patterns.
4. Seasonality: weekly revenue across 102 weeks.
5. Assortment: department and commodity mix; private-label vs national-brand share.
6. Promotions: discount share of spend, coupon usage rate, campaign reach by type. Note that Type A coupons are chosen from each household's prior purchases, a confounder Phase 8 must handle.
7. Lifecycle: state distribution over periods; Sankey preview of transitions between two periods.
8. Churn base rate: share of active households with zero purchases in the next 2 periods.
9. Cohort retention for households first seen after warm-up, noting the cohort's size.
10. Demographic coverage (801 of ~2,500) and descriptive lapse rates by income and age band.

**Outputs:** `notebooks/01_eda.ipynb` (logic imported from `src/`), `reports/figures/eda_*.png`, threshold changes in `docs/decisions.md`.

**Acceptance criteria**

- [ ] The notebook runs top to bottom in under 5 minutes.
- [ ] Every section has a finding grounded in a printed number.
- [ ] The churn base rate and demographic coverage are recorded in `PHASE_LOG.md`.

## Phase 4 — Markov chain lifecycle model

**Goal:** quantify how customers move between states, how long until they lapse, what each state is worth, and what a retention lift is worth, with uncertainty and tested assumptions.

**Tasks**

1. **Estimate P.** Count transitions C[i, j] between consecutive periods within `train_periods`. Row-normalize with additive smoothing α = `smoothing_alpha` on structurally possible cells only; impossible cells (e.g., Champion → Dormant in one step) stay 0. Unit tests: rows sum to 1, entries non-negative.
2. **Uncertainty.** `bootstrap_reps` household-level bootstrap resamples (resample households, not transitions); 95% CI for every P[i, j] and every derived metric.
3. **Assumption tests** (`assumptions.py`): first- vs second-order chain with likelihood-ratio test and BIC; time homogeneity by estimating P on the first vs second half of training with a chi-square test. Report results honestly. If second order wins, keep first order as the interpretable model and report the lift.
4. **Absorbing analysis** (Lapsed absorbing): partition into transient block Q and absorbing block R; compute N = (I − Q)⁻¹, expected periods to absorption t = N·1 (also in days), and absorption probabilities B = NR.
5. **Ergodic variant:** Lapsed returns at the empirical rate. Compute the stationary distribution π (left eigenvector for eigenvalue 1) and compare it to the observed period-25 distribution.
6. **Markov reward process:** r_s = mean gross margin per household-period in state s (revenue × `gross_margin`); per-period discount γ = (1 + annual rate)^(−28/365); state value V = (I − γQ)⁻¹ r.
7. **Backtest:** from the observed distribution at period 19, project k = 1…6 periods and compare to actual periods 20–25 with total variation distance, against a persistence baseline. At household level, derive P(state at t+2 ∈ {Dormant, Lapsed}) from P² as the Markov churn score; it matches the Phase 6 label exactly.
8. **Survival benchmark** (`survival/km_cox.py`): Kaplan-Meier curves of time-to-Lapsed by starting state, plus a Cox model with baseline covariates. Compare median time-to-lapse with the Markov estimate.
9. **Scenario simulator** (`simulate.py`): `simulate(P, intervention, horizon=13, population)`, e.g., `{"At-Risk": {"Regular": +0.05}}`. Take the added mass proportionally from the row's other non-target cells (documented). Return Δ expected margin and Δ lapsed households vs baseline, per 100,000 customers. Precompute a scenario grid for the dashboard.
10. **Segmented chains:** P by income band (demographic subset) and by starting-state cohort, with CIs; flag cells with fewer than 30 transitions.
11. **Stretch — HMM:** Gaussian HMM (hmmlearn) on per-period [log1p spend, trips], 3–6 latent states by BIC; confusion matrix vs rule-based states.

**Outputs:** `markov_P.parquet` (with CIs), `markov_metrics.json`, `markov_backtest.parquet`, `scenarios.parquet`, `survival_curves.parquet`, figures.

**Acceptance criteria**

- [ ] A unit test confirms this toy chain: Q = [[0.80, 0.15], [0.40, 0.30]], R = [[0.05], [0.30]] gives N = [[8.75, 1.875], [5.0, 2.5]], t = [10.625, 7.5], B = [[1], [1]].
- [ ] Every reported metric has a 95% CI.
- [ ] Markov-property and homogeneity results are recorded, whatever they show.
- [ ] Backtest TVD is reported against the persistence baseline.
- [ ] `simulate` is deterministic, and zero intervention gives zero delta (test).

## Phase 5 — Markov chain campaign attribution

**Goal:** credit campaigns for coupon-driven conversions with the Markov removal effect, compared with first-touch, last-touch and linear credit.

**Tasks**

1. **Conversion:** a basket containing a coupon-discounted product linked to a campaign the household received (`int_coupon_transactions`). Run a sensitivity version using only `coupon_redempt` and report both.
2. **Paths:** per household, campaigns received ordered by `start_day`; a path ends at the first conversion or at day 711 (NULL). 90-day lookback from config. Two granularities: campaign type (A/B/C) and individual campaign.
3. **Graph:** first-order chain over START → channels → {CONVERSION, NULL}; compute P(conversion from START) by reusing `markov/absorbing.py`.
4. **Removal effect:** RE_c = 1 − P_conv(without c) / P_conv; share_c = RE_c / Σ RE_k.
5. **Benchmarks:** first-touch, last-touch, linear; second-order chain as a robustness check.
6. **Caveats** in outputs: `campaign_table` records targeting, not confirmed exposure; attribution is descriptive credit allocation, not causal effect.

**Outputs:** `attribution.parquet`, `attribution_paths_summary.json`.

**Acceptance criteria**

- [ ] Shares sum to 1 for every model.
- [ ] A unit test on toy paths reproduces a known removal-effect answer.
- [ ] The comparison and redemption-source sensitivity are exported.

## Phase 7 — Customer lifetime value (runs before Phase 6)

**Goal:** a validated 26-week CLV per household from three competing methods, one chosen for the dashboard and profit curve.

**Tasks**

1. **Time split:** calibration weeks 1–76, holdout 77–102. Per household: frequency (repeat purchase days), recency, T, average monetary value.
2. **BG/NBD + Gamma-Gamma** with PyMC-Marketing ≥ 1.3 (`BetaGeoModel().fit(data=rfm, method="map")`). Check and report the frequency–monetary correlation first. MAP by default; full MCMC behind a config flag, reporting R-hat and effective sample size.
3. **Holdout validation:** predicted vs actual transactions and spend; MAE, Spearman, decile calibration plot.
4. **Benchmarks:** Markov reward CLV assigned by state at week 76; LightGBM regressor (Tweedie) predicting holdout spend from calibration features.
5. **Choose** by holdout Spearman, then interpretability; log in `docs/decisions.md`. Final CLV = predicted spend × `gross_margin`, discounted, refit on all 102 weeks for scoring.
6. Note in the report that BG/NBD assumes permanent dropout, which grocery shoppers rarely show; the Markov model allows returns.

**Outputs:** `clv.parquet` (household, method, clv_26w, p_alive), `clv_comparison.parquet`, figures.

**Acceptance criteria**

- [ ] A holdout comparison table exists for all three methods.
- [ ] The chosen method has holdout Spearman ≥ 0.5, or the shortfall is documented.

## Phase 6 — Churn prediction (machine learning)

**Goal:** rank active households by probability of no purchase in the next 8 weeks, explain each score, and choose the profit-maximizing contact threshold.

**Tasks**

1. **Label:** at cutoff t, for households not Lapsed, y = 1 if zero purchases in periods t+1 and t+2. If the Phase 3 base rate is below 3%, add a partial-churn label (spend in t+1..t+2 at least 50% below the trailing 6-period average); model both and lead with the one with enough positives. Log the choice.
2. **Rolling-origin split** (label windows never overlap test):

| Split | Cutoff periods | Labels reach |
| --- | --- | --- |
| Train | 6–15 | ≤ 17 |
| Validation | 16–17 | ≤ 19 |
| Buffer (unused) | 18–19 | — |
| Test | 20–23 | ≤ 25 |

3. **Features** (`features/build_features.py`, data ≤ t only): recency; frequency and spend over the last 1/3/6 periods; OLS spend slope over 6 periods; inter-trip mean, std, CV; average basket spend and items; departments per basket; category entropy; private-label share; discount share; coupon-transaction count; campaigns received in the last 3 periods; mailer/display exposure of bought products (`causal_data`); distinct stores; weekend-trip share; tenure; current Markov state (one-hot); Markov churn score (P fit on train cutoffs only); demographics with `has_demographics`; early-warning category features from Phase 9 if available.
4. **Models:** (a) Markov score, (b) Cox risk score, (c) scaled logistic regression, (d) LightGBM tuned by Optuna (`optuna_trials`, objective = validation PR-AUC, early stopping). Handle imbalance with class weights or `scale_pos_weight`, not resampling.
5. **Calibration:** isotonic on validation; report expected calibration error.
6. **Test metrics:** PR-AUC, ROC-AUC, Brier, calibration curve, lift and capture at top 10% and 20%. One table comparing all four models, with 1,000-resample bootstrap 95% CIs on PR-AUC and top-decile lift.
7. **Profit curve:** EV = p × save_rate × CLV − offer_cost. Choose the threshold maximizing total EV for each `save_rate_grid` value; show sensitivity.
8. **Explainability:** SHAP beeswarm, dependence plots for the top 5 features, top-3 reason codes per household from a mapping file (e.g., "Trips down 60% vs prior quarter").
9. **Fairness:** recall and FPR by income and age band; flag gaps above 10 points. Train a variant without demographics and report the performance cost; default to the fairer model at comparable PR-AUC.
10. **Leakage test:** add future rows after the cutoff; features must not change.
11. **Tracking:** save params, metrics and model to `data/outputs/models/` (JSON + model file); also log to MLflow if the polish group is installed.
12. **Batch scoring:** `churn/score.py` scores the latest cutoff and writes the risk list; also export top risk deciles as a formatted Excel target list (openpyxl).
13. **Model card:** `reports/model_cards/churn.md`.

**Outputs:** `churn_scores.parquet` (household, cutoff, p, decile, CLV, EV, reason_1–3), `churn_model_comparison.parquet`, `profit_curve.parquet`, Excel target list.

**Acceptance criteria**

- [ ] The leakage test passes.
- [ ] LightGBM beats logistic, Markov and Cox on test PR-AUC beyond CI overlap, or the chosen model is justified in `docs/decisions.md`.
- [ ] ECE, profit-optimal threshold and fairness table are reported.
- [ ] **Human checkpoint:** metrics and threshold are approved.

## Phase 8 — Uplift modeling and experiment design

**Goal:** estimate which customers change behavior because of a campaign, translate it into targeting ROI, and design the confirming A/B test. Dunnhumby targeting is not randomized, so the phase has three parts.

**Part A — randomized uplift on X5 RetailHero (run first)**

1. Test the pipeline on Hillstrom: Women's email vs no email, outcome = visit. Assert the estimated average uplift matches the raw treated-minus-control difference within its bootstrap CI.
2. Build pre-communication features from X5 `purchases` (recency, frequency, spend, category mix, points behavior) and `clients` (age, gender, issue dates, missing flags).
3. Confirm randomization: treatment share near 0.5 and |SMD| < 0.1 on every feature.
4. Fit S-, T-, X- and DR-learners (CausalML, LightGBM bases) on a 70/30 split stratified by treatment and target. Report Qini, AUUC, uplift at 10% and 30%, with bootstrap CIs.
5. ROI curve (offer cost from config) and optimal targeting share.

**Part B — observational uplift on Dunnhumby**

1. **Units:** each campaign with ≥ 100 treated and ≥ 100 untreated eligible households; eligible = active in the 56 days before `start_day`; T = 1 if in `campaign_table`.
2. **Outcomes:** primary Y = spend per day in the campaign window minus spend per day in the 56-day pre-window; secondary Y = any coupon transaction in the window.
3. **Covariates (pre-window only):** spend, trips, category mix (mandatory, because Type A coupons are personalized), prior coupon use, prior campaigns, lifecycle state, demographics flag, count of concurrent campaigns. Run a sensitivity check excluding households with concurrent campaigns.
4. **Propensity and overlap:** logistic and LightGBM propensity; trim outside [0.05, 0.95]; balance table with SMD before and after IPW (target |SMD| < 0.1).
5. **Pooling:** stack campaign-household rows with campaign ID and type as features; GroupKFold by household.
6. **Learners:** S-, T-, X- and DR-learners (CausalML); AIPW average effect with bootstrap CI as a sanity check.
7. **Evaluation:** Qini, AUUC, uplift by decile on test households.
8. **Segments:** Persuadables, Sure Things, Lost Causes, Sleeping Dogs, with size and average CLV.
9. **ROI:** incremental margin minus offer cost for top-k% vs everyone vs random, k = 10…100; state the optimal k.
10. **Experiment design** (`uplift/power.py`): statsmodels power analysis for an A/B test of the top-30% policy vs business-as-usual (α 0.05, power 0.8, effect from step 6, variance from data). Output households per arm and duration in weeks.
11. State the unconfoundedness assumption prominently in outputs.

**Part C — confounding stress test**

1. Create a confounded X5 subsample: keep treated customers with probability rising in prior spend.
2. Estimate the average effect three ways (naive, IPW, AIPW) and compare each to the Part A randomized estimate.
3. Report the bias of each method in `uplift_confounding_test.parquet`.

**Outputs:** `uplift_scores.parquet`, `uplift_eval.parquet`, `uplift_balance.parquet`, `uplift_roi.parquet`, `ab_test_design.json`, `uplift_confounding_test.parquet`, `reports/model_cards/uplift.md`.

**Acceptance criteria**

- [ ] The Hillstrom test passes in CI.
- [ ] The best X5 learner's Qini beats random, with a CI excluding zero.
- [ ] The Dunnhumby balance table is produced; any |SMD| ≥ 0.1 after weighting is listed with its implication.
- [ ] ROI curves, the A/B sample size and the Part C bias table are exported.

## Phase 9 — Market basket and early-warning categories

**Tasks**

1. Build baskets at `commodity_desc` level, excluding non-merchandise departments via a config list confirmed against Phase 1 profiling.
2. FP-Growth (mlxtend) with min support from config (start at 0.01; lower if fewer than 50 rules). Compute support, confidence, lift, leverage; keep lift > 1.2; remove trivial rules.
3. Mine rules separately for Champion and At-Risk baskets; report the rules that differ most.
4. **Early-warning categories:** for households that reached Dormant, compare each commodity's spend share in the 3 periods before vs the 3 before that; rank by average drop vs a matched non-lapsing group. Feed top signals to Phase 6 as features and log the change.
5. Network graph of the top 30 rules.

**Outputs:** `basket_rules.parquet`, `basket_rules_by_segment.parquet`, `early_warning_categories.parquet`, network figure.

**Acceptance criteria**

- [ ] At least 10 non-trivial rules with lift > 1.2.
- [ ] The early-warning list has effect sizes and counts.
- [ ] Any Phase 6 feature change is logged and Phase 6 re-run.

## Phase 10 — Pipeline, testing, monitoring and CI

**Tasks**

1. `python -m retention_radar.pipeline` runs every phase in order; `make all` calls it.
2. Optional polish: Dagster assets wrapping the same steps, with dbt via `DbtProjectComponent` (`dg scaffold defs dagster_dbt.DbtProjectComponent`); save an asset-graph screenshot.
3. A Pandera schema in `schemas.py` for every Parquet output, validated on write.
4. Tests: unit (absorbing math, removal effect, state rules with parametrized edge cases, `simulate`, leakage, profit calculation); integration (full pipeline on the fixture under 10 minutes with the `ci` profile); coverage ≥ 80% on `src/`.
5. Monitoring (`monitoring/drift.py`): PSI = Σ (actual% − expected%) × ln(actual% / expected%) for the top 10 features and the score distribution, training cutoffs vs latest cutoff; flag PSI > 0.2; write `drift_report.parquet`.
6. Document a retraining policy (drift threshold, cadence) in `docs/decisions.md`.
7. CI: lint and tests on every push, uv cache, coverage badge.

**Acceptance criteria**

- [ ] On a fresh clone with data present, `make all` rebuilds every output.
- [ ] CI green, coverage ≥ 80%.
- [ ] Every output passes its Pandera schema.
- [ ] The drift report is generated.

## Phase 11 — Dashboards

**Streamlit pages**

| Page | Content |
| --- | --- |
| Executive overview | Active households, revenue, at-risk count, revenue at risk (Σ p × CLV); 3 findings from `results.json` |
| Customer lifecycle | Sankey between two selectable periods; transition heatmap with CI on hover; expected days to lapse by state; KM curves |
| Retention simulator | Sliders for transition changes and horizon; live Δ margin and Δ lapsed households per 100k customers |
| Churn risk | Filterable risk list (decile, CLV, reason codes); SHAP waterfall per household; profit curve with threshold and save-rate sliders |
| Campaigns | Attribution shares; Qini curves; ROI curve; segment sizes; A/B design card |
| Baskets | Rules table, network, early-warning categories |
| Methodology | Method notes, every assumption with its value, drift report, links |

**Build rules**

1. The app reads built outputs only and never trains; the simulator computes live from `markov_P.parquet`.
2. `st.cache_data` for all loads; each page loads in under 3 seconds.
3. One colorblind-safe palette; each lifecycle state keeps the same color everywhere.
4. Every chart gets a one-line takeaway caption written from the data.
5. Assumptions (margin, offer cost, save rate) appear near every dollar figure.
6. Dunnhumby terms forbid redistribution, so the deployed app ships aggregates and re-keyed IDs only; log the decision in `docs/data_governance.md`.
7. Deploy to Streamlit Community Cloud, keeping outputs well under ~1 GB; note the wake-up delay in the README. Save screenshots and a 30-second GIF.
8. **Required BI dashboard:** rebuild the Executive, Lifecycle and Campaigns pages in Power BI (`bi_tool: powerbi`) or Tableau Public (`bi_tool: tableau`) from CSVs exported to `data/outputs/bi/`. Include calculated measures for revenue at risk and retention rate (DAX or Tableau calculated fields). Commit the workbook, a PDF export and screenshots to `bi/`. The pipeline prepares the CSVs, a data model description and step-by-step build instructions in `bi/README.md`.

**Acceptance criteria**

- [ ] `make app` starts the app from built outputs.
- [ ] All 7 pages load without errors at 1280 px and at phone width.
- [ ] The simulator returns zero delta at zero intervention.
- [ ] BI CSVs and `bi/README.md` exist; the data-terms decision is logged before deploy.

## Phase 12 — Reporting and portfolio packaging

**Tasks**

1. `reporting/numbers.py` writes every headline number (with CI where available) to `results.json`; all documents pull from it via Jinja or Quarto params, never typed by hand.
2. README: one-paragraph pitch; 3–5 headline results with numbers; dashboard links and GIF; architecture diagram; how to run; repo map; tech stack; limitations; data credits.
3. Technical report (Markdown or Quarto → HTML/PDF): problem, data, one section per method with equations, validation, results, limitations, next steps.
4. One-page executive summary: 3 findings and 3 recommendations, each with dollar impact per 100k customers and assumptions shown, plus the A/B test proposal.
5. Model cards: churn, CLV, uplift.
6. `docs/data_governance.md`: sources and terms; anonymization and re-keying; what changes with real customer data (PII, access control, retention, consent and marketing opt-outs); fairness findings.
7. `reports/interview_pack.md`: 3-minute walkthrough and 12 likely questions with answers grounded in `results.json`.
8. Three resume bullets and a LinkedIn post draft, numbers from `results.json`.
9. Stretch extensions listed in the README (not built): demand forecasting, next-basket recommendation, real-time scoring API.

**Acceptance criteria**

- [ ] A script checks that every number in the README and reports matches `results.json`.
- [ ] The technical report renders without errors.
- [ ] The executive summary fits on one page.
- [ ] **Human checkpoint:** approval before publishing the repo, dashboards or any post.

## Phase 13 (optional) — Scale track: PySpark on Databricks

**Inputs:** Dunnhumby *Let's Get Sort-of-Real* (50K-customer sample, then 1–2 of the nine ~500 MB parts); a Databricks Free Edition account (human creates it).

**Tasks**

1. Upload to a Unity Catalog volume and write Delta tables.
2. Port the Phase 2 period-activity and lifecycle-state logic and the Phase 6 RFM and trend features to PySpark (window functions over household × period).
3. Parity test: run the PySpark code on the Complete Journey fixture and assert outputs equal the dbt/Polars outputs row for row.
4. Estimate the transition matrix on the large sample and compare its structure with Complete Journey (Sort-of-Real is synthetic).
5. Record runtime, rows processed and Spark UI stage summary; write `docs/scale_notes.md` (partitioning by household, skew, incremental periods).
6. Stay within Free Edition limits: serverless only, one or two workloads at a time, non-commercial use.

**Acceptance criteria**

- [ ] The parity test passes on the fixture.
- [ ] State and feature outputs exist for at least 50,000 customers.
- [ ] Runtime and scale notes are documented.

## Project-level acceptance checklist

- [ ] `make all` rebuilds everything from raw CSVs on a clean machine.
- [ ] CI green, coverage ≥ 80%.
- [ ] All dbt tests and Pandera schemas pass.
- [ ] Markov metrics have CIs; assumption tests are reported.
- [ ] The churn model beats, or is justified against, the Markov, Cox and logistic baselines.
- [ ] CLV holdout validation is reported.
- [ ] Uplift: Hillstrom test, X5 Qini, Dunnhumby balance, confounding test and A/B design are reported.
- [ ] Streamlit is deployed and the BI dashboard is built; the data-terms decision is logged.
- [ ] Every number in the README and reports matches `results.json`.
- [ ] Model cards, governance note and interview pack exist.
- [ ] Human approval before publishing.

## Decision log (why the plan looks like this)

- X5 RetailHero and Hillstrom were added because Dunnhumby campaign targeting is behavioral (Type A coupons are personalized), so it can't prove causal effects alone.
- Coupon use comes from transaction discount fields, because `coupon_redempt` covers far fewer households.
- Periods 1–3 are warm-up because most households were customers before day 1.
- The Markov churn score is defined as P(Dormant or Lapsed at t+2) so it matches the ML label exactly.
- Buffer periods 18–19 prevent training labels from overlapping the test window.
- scikit-uplift was dropped (stale); CausalML covers learners and metrics.
- dbt-core 1.12 over dbt 2.0 for adapter maturity; Dagster, MLflow, Docker and Quarto are optional because target analyst postings don't name them.
- A BI tool is required because Power BI appears in Publix, Target and Kroger 84.51° postings; Spark is optional for data-scientist roles.
