"""A/B test power analysis and sample size determination for uplift policy validation."""

import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

from retention_radar.config import get_project_root, load_project_config


def design_retention_ab_test(
    alpha: float = 0.05,
    power: float = 0.80,
    active_households: int = 2400,
    variance_spend_day: float = 2.85,
    expected_uplift_spend_day: float = 0.4758,
) -> Dict[str, Any]:
    """Design confirming A/B experiment comparing top-30% uplift policy against business-as-usual."""
    # Standard deviation of outcome (delta spend per day)
    sigma = np.sqrt(variance_spend_day)

    # Standardized effect size (Cohen's d)
    effect_size = expected_uplift_spend_day / sigma

    power_analysis = TTestIndPower()
    # Solve for sample size per arm
    sample_size_per_arm = power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )

    n_per_arm = int(np.ceil(sample_size_per_arm))
    total_required = n_per_arm * 2

    # Calculate test duration in weeks given active pool
    # If active household pool is 2,400, sample is reached in parallel across cohorts
    # Duration in weeks needed to observe steady-state purchasing cycles (minimum 4 to 8 weeks)
    recommended_duration_weeks = 8

    design = {
        "experiment_name": "Retention Radar Top-30% Uplift Policy vs BAU",
        "primary_metric": "incremental_spend_per_day",
        "statistical_test": "Two-sample independent t-test (two-sided)",
        "significance_level_alpha": alpha,
        "statistical_power": power,
        "baseline_standard_deviation": round(float(sigma), 4),
        "minimum_detectable_effect_spend_day": round(float(expected_uplift_spend_day), 4),
        "cohens_d": round(float(effect_size), 4),
        "required_sample_size_per_arm": n_per_arm,
        "total_sample_size": total_required,
        "available_active_households": active_households,
        "recommended_duration_weeks": recommended_duration_weeks,
        "sampling_ratio": "1:1 (50% treatment policy / 50% business-as-usual)",
        "causal_assumptions": [
            "Unconfoundedness (Conditional Independence): Given pre-window covariates, treatment assignment is independent of potential outcomes.",
            "Stable Unit Treatment Value Assumption (SUTVA): No interference between households; grocery consumption in one household does not affect neighbors.",
            "Positivity (Common Support): All eligible households have a strictly positive probability of receiving the retention offer (0.05 <= e(X) <= 0.95).",
        ],
    }

    return design


def main() -> None:
    """Run power analysis and write ab_test_design.json."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load actual variance and effect from uplift_eval and uplift_scores if available
    eval_path = out_dir / "uplift_eval.parquet"
    expected_uplift = 0.4758
    if eval_path.exists():
        eval_df = pd.read_parquet(eval_path)
        if "uplift_at_30pct" in eval_df.columns:
            expected_uplift = float(eval_df.iloc[0]["uplift_at_30pct"])

    scores_path = out_dir / "uplift_scores.parquet"
    var_spend = 2.85
    if scores_path.exists():
        scores_df = pd.read_parquet(scores_path)
        if "uplift_tau" in scores_df.columns:
            var_spend = float(np.var(scores_df["uplift_tau"]) + 1.5)

    print("Running A/B experiment power analysis...")
    design = design_retention_ab_test(
        alpha=0.05,
        power=0.80,
        active_households=2400,
        variance_spend_day=var_spend,
        expected_uplift_spend_day=expected_uplift,
    )

    out_file = out_dir / "ab_test_design.json"
    with open(out_file, "w") as f:
        json.dump(design, f, indent=2)

    print(f"\nA/B Test Design saved to {out_file}:")
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
