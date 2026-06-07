# Executive Summary: Retention Radar Customer Intelligence

**Author:** Sanjay Bahadur  
**Audience:** Chief Marketing Officer, VP of Customer Analytics, VP of Merchandising  
**Scope:** Customer Retention, Lifecycle Migration, and Causal Uplift Optimization  
**Date:** June 2026  

---

### Executive Overview

Customer defection in grocery retail is gradual and multi-stage: households rarely stop shopping overnight. Using the 2,500-household, 102-week Dunnhumby transaction corpus, this project built **Retention Radar**—an end-to-end customer intelligence system combining absorbing Markov chains, machine learning churn prediction, causal uplift modeling, and basket affinity tracking.

---

### Core Findings & Analytical Realities

1. **Early-Warning Basket Contraction (Observed Association):**  
   Lapsing households exhibit significant basket shifts **6 to 8 weeks** prior to dormancy. Mean spend drops by **-76.2%** in fresh poultry/chicken, **-64.0%** in beef, and **-66.1%** in diapers and infant essentials relative to active cohorts. While observational data cannot confirm competitor store switching, these shifts provide an operational lead-time to intervene before complete transaction cessation.
2. **Predictive Churn Modeling & Demographic Parity:**  
   Our Optuna-tuned LightGBM model achieves a Test PR-AUC of **0.3370** (95% household-clustered CI [0.300–0.380]), performing comparably to regularized logistic regression (**0.3199** [0.282–0.362]) with overlapping confidence intervals, while delivering superior calibration (**ECE 0.0091 vs 0.0104**) and top-decile lift (**4.64x vs 4.51x**). In the top 20% outreach bracket, LightGBM captures **73.88%** of all churners. An ablation audit removing demographic features showed negligible performance loss (PR-AUC 0.3627), though subgroup audits remain constrained by small churn sample sizes (8–18 churners per demographic subgroup).
3. **Causal Uplift & Intervention Economics:**  
   Applying an S-Learner on observational campaign data suggests a **3.61x ROI** for targeting the top 30% uplift segment ($20,435 net profit vs. $8,173 random baseline). However, the high observational Qini (0.67) relative to randomized benchmarks (X5 RetailHero Qini 0.054) indicates residual confounding. While 202 households (2.1%) were categorized as potential negative-responders ("Sleeping Dogs") based on point estimates ($\hat{\tau} < -0.15$), unvalidated observational estimates should not dictate hard suppression without randomized field trials.

---

### Quantified Financial Impact (Grounded in `results.json`)

All figures below are mathematically derived from verified test set metrics and explicit business assumptions:

| Initiative | Derivation & Inputs | Net Impact per 26-Wk Cycle | Annualized Net Impact |
| :--- | :--- | :---: | :---: |
| **High-Risk Churn Outreach** *(Target Top 20% predicted churners)* | $100\text{k HH} \times 7.58\% \text{ churn} \times 73.88\% \text{ capture} \times 10\% \text{ save rate} \times \$244.57 \text{ CLV} - (20\text{k contacted} \times \$2.00)$ | **+$96,962** | **+$193,923** |
| **Uplift-Optimized Targeting** *(Top 30% causal uplift vs. random)* | Model policy net profit ($\$20,435$) minus random allocation net profit ($\$8,173$) on Dunnhumby coupon campaign | **+$12,262** | **+$24,524** |

*Key Business Assumptions:* 25.0% gross retail margin; $2.00 offer delivery cost per household; 10.0% winback save rate; $244.57 mean 26-week customer lifetime value (gross margin); 10.0% annual cost of capital.

---

### Recommended Validation: Randomized Controlled Trial (RCT)

Before rolling out automated suppression or large-scale interventions, we propose an **8-week randomized controlled trial**:
- **Design:** 1:1 balanced trial comparing the **Top-30% Uplift Policy** against **Business-As-Usual Control**.
- **Power Calculation:** **231 households per arm (462 total)** required to detect an incremental spend effect of **+$0.48/day** ($\alpha = 0.05$, $80\%$ statistical power, baseline $\sigma = \$1.81$/day).
- **Primary Metric:** Incremental margin spend per household, evaluated with Benjamini-Hochberg FDR control.
