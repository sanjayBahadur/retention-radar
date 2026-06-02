"""Methodology & Audit page - System architecture, causal DAG, model cards, and dataset credits."""

import streamlit as st

st.set_page_config(
    page_title="Methodology & Governance | Retention Radar", page_icon="📐", layout="wide"
)

st.title("📐 Methodology, Causal Governance & Architecture")
st.markdown(
    "Rigorous architectural design, causal identification strategies, model governance cards, and attribution."
)

# 1. System Architecture & Data Flow
st.subheader("1. End-to-End System Architecture")

st.markdown("""
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RAW DATA INGESTION                                   │
│  Dunnhumby 'The Complete Journey'  │  Hillstrom Email Marketing  │  X5 RetailHero       │
│  (2,500 HH, 2.6M rows, 102 weeks)  │  (64,000 customers)         │  (60,000 sample)     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              WAREHOUSE & DBT MODELING                                  │
│  DuckDB Warehouse (`data/retention_radar.duckdb`)                                      │
│  ├── Staging: stg_transactions, stg_demographics, stg_products, stg_campaigns          │
│  ├── Intermediate: int_basket_summaries, int_customer_weekly_metrics                   │
│  └── Marts: dim_household, dim_product, fct_baskets, fct_household_period_state       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
┌──────────────────────────────┐ ┌─────────────────────────┐ ┌───────────────────────────┐
│       MARKOV LIFECYCLE       │ │     CHURN PREDICTION    │ │      CAUSAL UPLIFT        │
│ • State Transition P Matrix  │ │ • Rolling Origin (No LK)│ │ • S/T/X/DR Learners       │
│ • Fundamental Matrix N       │ │ • Optuna LightGBM       │ │ • Propensity Trim [5%,95%]│
│ • Expected Days to Lapse     │ │ • Isotonic Calibration  │ │ • IPW/AIPW Confounding    │
│ • Reward CLV Engine          │ │ • Top Decile Lift 4.82x │ │ • Sleeping Dog Detection  │
└──────────────┬───────────────┘ └────────────┬────────────┘ └─────────────┬─────────────┘
               │                              │                            │
               └──────────────────────────────┼────────────────────────────┘
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          DECISIONING & CONSUMPTION LAYER                               │
│  ├── Executive Streamlit Dashboard (6 multi-page interactive operational views)        │
│  ├── High-Risk Retention Outreach Target Roster (`reports/churn_target_list.xlsx`)     │
│  ├── Business Intelligence Ready Marts (`data/outputs/bi/*.csv` for Tableau & Power BI)│
│  └── Automated Drift Monitoring & Pandera Schema Validation                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
""")

st.markdown("---")

# 2. Causal Identification Strategy
st.subheader("2. Causal Identification Strategy & Identification Assumptions")

col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.markdown("#### Causal Directed Acyclic Graph (DAG)")
    st.markdown("""
    ```
          ┌──────────────────────────────────────────────┐
          │ Pre-Treatment Covariates X                   │
          │ (Tenure, Historical Spend, Basket Rhythms,   │
          │  Coupon Sensitivity, Loyalty Engagement)     │
          └──────────────────────┬───────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       ┌────────────────────┐          ┌────────────────────┐
       │ Treatment T        │          │ Outcome Y          │
       │ (Campaign Offer /  ├─────────►│ (Incremental Spend │
       │  Retention Coupon) │          │  or Winback Margin)│
       └────────────────────┘          └────────────────────┘
                  ▲                               │
                  │        Unobserved U           │
                  └───────────────────────────────┘
                     (Tested via AIPW & Trimming)
    ```
    """)

with col_d2:
    st.markdown("#### Causal Verification Checklist")
    st.markdown(r"""
    - **1. Conditional Exchangeability (Unconfoundedness):**
      $(Y(1), Y(0)) \perp T \mid X$. All key purchase history, basket cadence, and redemption behaviors are observed in the 28-day pre-window.
    - **2. Positivity (Common Support):**
      $0.05 \le e(X) \le 0.95$. Propensity score distribution is trimmed to eliminate non-overlapping extreme tails, verified across all treatment groups.
    - **3. Stable Unit Treatment Value Assumption (SUTVA):**
      No interference across households. One customer receiving an email coupon does not alter the grocery purchasing utility of a separate household.
    - **4. Confounding Stress Testing:**
      Validated against randomized benchmarks (X5 RetailHero). AIPW recovered the ground truth ATE (+0.0377) with virtually zero bias (-0.0001) where naive estimation inflated effect by +20.7%.
    """)

st.markdown("---")

# 3. Model Governance & Production Cards
st.subheader("3. Production Model Cards & Audit Artifacts")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    with st.container(border=True):
        st.markdown("### ⚠️ Churn Early Warning")
        st.markdown("""
        - **Algorithm:** LightGBM Classifier (Optuna Tuned)
        - **Calibration:** Isotonic (ECE = 0.0090)
        - **Performance:** PR-AUC 0.3520 (4.82x baseline lift)
        - **Top 20% Capture:** 72.19% of all churners
        - **Fairness:** Demographic-blind behavioral model eliminates demographic proxy bias (+0.0107 PR-AUC)
        - **Artifact:** `reports/model_cards/churn.md`
        """)

with col_m2:
    with st.container(border=True):
        st.markdown("### 🎯 Causal Uplift")
        st.markdown("""
        - **Algorithm:** S-Learner with CausalML LightGBM
        - **Metrics:** Qini = 0.6706, AUUC = 0.9820
        - **Top 30% CATE:** +$0.4758 incremental margin/day
        - **Sleeping Dogs:** 202 customers (-$0.60/day) suppressed
        - **ROI:** 3.61x over random outreach ($20,435 net profit)
        - **Artifact:** `reports/model_cards/uplift.md`
        """)

with col_m3:
    with st.container(border=True):
        st.markdown("### 💰 Customer Lifetime Value")
        st.markdown("""
        - **Algorithm:** LightGBM Tweedie Regression Benchmark
        - **Baselines:** BG/NBD + Gamma-Gamma & Markov Chain
        - **Spearman Rank Corr:** 0.9162 (vs Markov 0.6960, BG/NBD 0.6046)
        - **MAE:** $228.71 per household (26-week horizon)
        - **Target Roster:** Integrated into profit optimization
        - **Artifact:** `reports/model_cards/clv.md`
        """)

st.markdown("---")

# 4. Pipeline Monitoring & Data Integrity Guardrails
st.subheader("4. Production Pipeline Integrity & Monitoring")

st.markdown("""
- **Data Leakage Prohibition:** Strictly temporal rolling-origin cross-validation (train: cutoffs 6–15, val: 16–17, buffer: 18–19, test: 20–23). Zero future information leakage.
- **Population Stability Index (PSI) Drift Monitoring:** Automated baseline drift tracking. All 11 primary predictive features evaluated on every batch score run (Status: All STABLE, PSI < 0.10).
- **Pandera Schema Validation:** 16 runtime schemas enforce types, nullability, primary key uniqueness, and probability bounds $[0, 1]$ before outputs are delivered to downstream applications.
""")

st.markdown("---")

# 5. Dataset Credits & Citations
st.subheader("5. Open Dataset Credits & Citations")

st.markdown("""
1. **Dunnhumby — "The Complete Journey":**
   Household-level grocery transaction dataset covering 2,500 frequent shoppers over 102 consecutive weeks. Includes basket-level line items, product taxonomy, and direct campaign exposure records.
2. **Kevin Hillstrom — "Email Marketing & Customer Lifetime Value":**
   Randomized controlled trial benchmark of 64,000 customers testing men's vs women's marketing treatments against an uncontacted control arm.
3. **X5 RetailHero — "Randomized Uplift Benchmark":**
   Large-scale randomized retail uplift dataset containing 400,000+ customer records used for validating CATE estimation algorithms under confounding stress tests.
""")
