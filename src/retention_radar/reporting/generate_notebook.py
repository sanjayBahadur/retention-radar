"""Generate a pre-rendered Jupyter walkthrough notebook for GitHub viewing."""

from pathlib import Path

import nbformat as nbf

from retention_radar.config import get_project_root


def build_walkthrough_notebook(dest_path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.14"},
    }

    cells = []

    # Title & Header
    cells.append(
        nbf.v4.new_markdown_cell("""# Retention Radar: Executive Walkthrough & Technical Demonstration
**Author:** Sanjay Bahadur (`mallasanjay2099@gmail.com`)
**Scope:** Customer Lifecycle Analytics, Causal Uplift (CATE), and Precursor Churn Signals
**Data:** 2.6M Dunnhumby Transactions across 2,500 Households over 102 Weeks

---

## Executive Overview
Traditional retail retention relies on blunt heuristics ("has not visited in 30 days") followed by blanket discount blasts. This notebook walks through **Retention Radar**—an algorithmic decision engine that combines:
1. **Absorbing Markov Chains** to map customer health trajectories and infinite-horizon value.
2. **Optuna-Tuned LightGBM** for out-of-time churn early-warning with strict leakage prevention.
3. **Causal Uplift Modeling (CATE)** to isolate the **202 'Sleeping Dogs'** whose churn is accelerated by marketing contact.
4. **Precursor Category Departure** proving households strip butcher meats and baby consumables 6–8 weeks before complete defection.""")
    )

    # Imports & Setup
    cells.append(
        nbf.v4.new_code_cell("""import json
from pathlib import Path
import pandas as pd
import numpy as np

# Load precomputed verified metrics
outputs_dir = Path("../data/outputs")
with open(outputs_dir / "results.json") as f:
    results = json.load(f)

print("Retention Radar Outputs Loaded Successfully.")
print(f"Project: {results['project']} | Updated: {results['last_updated']}")""")
    )

    # Section 1: EDA & Lorenz Curve
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 1. Exploratory Data Analysis & Revenue Inequality

In retail grocery, customer value is extremely skewed. Lorenz curve analysis reveals significant revenue concentration:
- **Gini Coefficient:** `0.5034`
- **Top 20% Revenue Share:** `53.1%` of all grocery sales ($4.2M of $7.9M total revenue).
- **Baseline Churn Rate:** `4.53%` per 28-day period (7.58% across test cutoffs).""")
    )

    cells.append(
        nbf.v4.new_code_cell("""eda_summary = {
    "Total Households": 2500,
    "Total Baskets": 272110,
    "Total Revenue": "$7,921,184.70",
    "Gini Coefficient": 0.5034,
    "Top 20% Revenue Share": "53.1%",
    "Demographic Missingness": "67.96% (Anonymous Loyalty Shoppers)"
}
pd.DataFrame(list(eda_summary.items()), columns=["Metric", "Value"])""")
    )

    # Section 2: Absorbing Markov Chains
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 2. Customer Lifecycle Dynamics (Absorbing Markov Chains)

We model customer migration across 6 states: 5 transient (*New, Champion, Regular, At-Risk, Dormant*) and 1 absorbing (*Lapsed*, defined as $\\ge 3$ consecutive periods without shopping).

Using the fundamental matrix $N = (I - Q)^{-1}$, we calculate the **Expected Days to Lapse**:
- **Champions:** Expected **2,105 days** ($95\\%$ CI: $[1,950, 2,288]$ days) $\approx 5.8$ years.
- **At-Risk:** Expected **1,617 days** ($95\\%$ CI: $[1,469, 1,795]$ days) $\approx 4.4$ years before complete defection, providing ample runway for intervention.

Using Bellman discounted rewards $V = (I - \\gamma Q)^{-1} r$ at a 10% annual discount rate:
- **Champion CLV:** **$2,406.85**
- **At-Risk CLV:** **$1,455.93**""")
    )

    cells.append(
        nbf.v4.new_code_cell("""df_lapse = pd.DataFrame(results["markov"]["expected_days_to_lapse"]).T
df_lapse["Reward_CLV"] = [results["markov"]["markov_reward_clv"][s] for s in df_lapse.index]
df_lapse[["mean_days", "ci_lower_95", "ci_upper_95", "Reward_CLV"]]""")
    )

    # Section 3: Churn Prediction & Calibration
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 3. Calibrated Machine Learning Churn Early Warning

Models were trained across historical cutoffs 6–15, tuned with Optuna on validation cutoffs 16–17, and evaluated strictly out-of-time on test cutoffs 20–23 ($N = 9,397$, base churn rate 7.58%):
- **LightGBM PR-AUC:** **0.3520** [0.3186, 0.3880] (a **4.82x lift** over baseline).
- **Top 10% Churn Capture:** **48.17%** of all future churners intercepted in the top decile.
- **Top 20% Churn Capture:** **72.19%** of all future churners intercepted.
- **Expected Calibration Error (ECE):** **0.0090 (< 1%)** via Isotonic Regression.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""df_churn_comp = pd.read_parquet(outputs_dir / "churn_model_comparison.parquet")
df_churn_comp[["model", "pr_auc", "roc_auc", "brier_score", "ece", "top_10_lift", "top_20_capture"]]""")
    )

    # Section 4: Fairness & Demographic Ablation
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 4. Algorithmic Fairness & The Demographic Missingness Paradox

In the loyalty program, 68% of households have no demographic profiles, while the 32% with profiles are self-selected loyal shoppers with an empirical churn rate of only 1.22% (vs 10.87% for unknown shoppers).
- Tree splits initially exploited the demographic missingness flag as a shortcut for churn risk.
- **Ablation Solution:** Stripping demographic features completely and training exclusively on transaction behavior improved Test PR-AUC to **0.3627 (+0.0107)** while eliminating demographic proxy disparities.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""fairness_ablation = {
    "Specification": ["Baseline Model (With Demographics)", "Ablated Model (Pure Behavioral)"],
    "Test PR-AUC": [0.3520, 0.3627],
    "Demographic Proxy Disparity": ["Present (Missingness Flag Exploited)", "Completely Eliminated (0 Disparity)"],
    "Production Decision": ["Rejected", "Selected for Production Scoring"]
}
pd.DataFrame(fairness_ablation)""")
    )

    # Section 5: Causal Uplift & Sleeping Dogs
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 5. Causal Uplift Modeling & The "Sleeping Dogs" Guardrail

Uplift modeling estimates the Conditional Average Treatment Effect (CATE): $\tau(X) = E[Y(1) - Y(0) \\mid X]$.
- Validated against randomized controlled trials (Hillstrom $N = 42,693$ and X5 RetailHero $N = 60,000$).
- In confounding stress testing, Augmented IPW (AIPW) eliminated selection bias (-0.0001 bias) where naive estimation inflated effect by +20.7%.
- Segmented Dunnhumby customers into 4 quadrants:
  - **Persuadables:** 8,860 households (+$1.20/day CATE). Target with retention offers.
  - **Sure Things:** 246 households. Will purchase regardless; suppress to save budget.
  - **Sleeping Dogs:** **202 households (2.1%) with negative uplift (-$0.60/day)**. Marketing contact causes annoyance and triggers churn! Hard-suppressed from communications.
  - **Lost Causes:** 137 households. Unresponsive; do not waste budget.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""df_roi = pd.read_parquet(outputs_dir / "uplift_roi.parquet")
df_roi[["targeting_percentile", "households_targeted", "mean_uplift_spend_per_day", "incremental_margin", "net_profit", "roi_ratio", "random_net_profit"]]""")
    )

    # Section 6: Precursor Category Departure
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 6. Precursor Category Abandonment (The 8-Week Departure Runway)

Evaluating category spend share drops during the 6 periods prior to dormancy against matched active controls:
- **Diapers & Baby Care:** Net drop = **-66.1%**
- **Fresh Poultry / Chicken:** Net drop = **-76.2%**
- **Dog Foods / Pet Care:** Net drop = **-74.0%**
- **Fresh Beef:** Net drop = **-64.0%**

**Core Takeaway:** Households do not lapse abruptly. They switch primary grocers for butcher meats and family consumables 6 to 8 weeks before overall transactions drop to zero.""")
    )

    cells.append(
        nbf.v4.new_code_cell("""df_ew = pd.read_parquet(outputs_dir / "early_warning_categories.parquet")
df_ew[["rank", "commodity_desc", "lapsing_hh_count", "lapsing_share_change", "net_drop_effect"]].head(10)""")
    )

    # Section 7: Conclusion & Links
    cells.append(
        nbf.v4.new_markdown_cell("""---
## 7. Operational Deliverables & Next Steps

All production outputs are pre-packaged for enterprise consumption:
- **Interactive Executive Dashboard:** `docs/index.html` (viewable on GitHub Pages)
- **High-Risk Target Roster:** `reports/churn_target_list.xlsx` (471 scored accounts with SHAP reason codes)
- **Tableau & Power BI Package:** `data/outputs/bi/` with complete documentation in `bi/README.md`
- **Model Cards:** `reports/model_cards/churn.md`, `uplift.md`, and `clv.md`""")
    )

    nb.cells = cells

    with open(dest_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    return dest_path


if __name__ == "__main__":
    root = get_project_root()
    dest = root / "notebooks" / "02_executive_walkthrough.ipynb"
    build_walkthrough_notebook(dest)
    print(f"Generated walkthrough notebook: {dest}")
