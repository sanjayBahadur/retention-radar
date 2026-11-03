"""Plan a household-randomized retention experiment with explicit assumptions."""

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
    expected_uplift_spend_day: float = 0.50,
) -> Dict[str, Any]:
    """Estimate sample size for an assumed detectable change, not a promised uplift.

    Variance is the variance of household-level observed outcomes. The normal
    approximation assumes independent household outcomes and equal arm sizes.
    """
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("Alpha and power must be between zero and one.")
    if variance_spend_day <= 0 or expected_uplift_spend_day <= 0 or active_households < 1:
        raise ValueError(
            "Variance, detectable effect, and eligible household count must be positive."
        )
    sigma = np.sqrt(variance_spend_day)
    effect_size = expected_uplift_spend_day / sigma
    n_per_arm = int(
        np.ceil(
            TTestIndPower().solve_power(
                effect_size=effect_size,
                nobs1=None,
                alpha=alpha,
                power=power,
                ratio=1.0,
                alternative="two-sided",
            )
        )
    )
    return {
        "experiment_name": "Retention offer vs no study offer among high-score households",
        "status": "Planning proposal; no intervention experiment has been run",
        "eligible_population": "Unique households in the highest 30% of the frozen uplift ranking",
        "randomization_unit": "Household; one assignment for the full experiment",
        "randomization": "1:1, stratified by prior spend and score band",
        "treatment": "One retention offer with recorded contact and redemption cost",
        "control": "No study offer; record business-as-usual promotions in both arms",
        "primary_metric": "Household average spend per day over eight weeks",
        "secondary_metric": "Gross margin minus offer and fulfillment costs per household",
        "analysis": "Intention-to-treat difference in means with a 95% confidence interval",
        "statistical_test": "Two-sample independent t-test (two-sided)",
        "significance_level_alpha": alpha,
        "statistical_power": power,
        "baseline_standard_deviation": round(float(sigma), 4),
        "minimum_detectable_effect_spend_day": round(float(expected_uplift_spend_day), 4),
        "mde_source": "Planning assumption: $0.50 additional spend per household per day",
        "cohens_d": round(float(effect_size), 4),
        "required_sample_size_per_arm": n_per_arm,
        "total_sample_size": 2 * n_per_arm,
        "available_active_households": active_households,
        "feasible_in_available_pool": bool(2 * n_per_arm <= active_households),
        "recommended_duration_weeks": 8,
        "sampling_ratio": "1:1 (offer / no study offer)",
        "guardrails": [
            "Track opt-outs, complaints and negative net margin; define stop thresholds before launch.",
            "Keep household assignment fixed; avoid overlapping study offers and log other campaigns.",
            "Freeze eligibility, primary outcome, sample size and analysis before randomization.",
        ],
        "rollout_rule": "Consider rollout only if net margin improves and prespecified guardrails pass.",
        "causal_assumptions": [
            "Random allocation is implemented and analyzed by original assignment.",
            "No material spillover between households or contamination across study arms.",
            "Outcome recording is comparable across arms, including households with zero purchases.",
        ],
        "limitations": [
            "Historical outcome variance is a planning proxy for eight-week household outcomes.",
            "Recalculate sample size with pilot variance; heavy tails may require more households.",
            "The test evaluates an offer within the selected pool, not uplift ranking vs random targeting.",
        ],
    }


def main() -> None:
    """Write a reproducible proposal using measured outcome variance and an explicit MDE."""
    cfg = load_project_config()
    out_dir = get_project_root() / cfg.paths.data_outputs
    # One observed campaign outcome per household avoids treating repeat campaigns
    # as independent households. Prediction variance is not outcome variance.
    scores = pd.read_parquet(out_dir / "uplift_scores.parquet")
    if scores["household_key"].duplicated().any():
        raise ValueError("Experiment planning requires unique household observations.")
    outcome_variance = float(scores["camp_spend_per_day"].var(ddof=1))
    eligible_count = max(1, int(len(scores) * 0.30))
    design = design_retention_ab_test(
        active_households=eligible_count,
        variance_spend_day=outcome_variance,
        expected_uplift_spend_day=0.50,
    )
    design["variance_source"] = (
        "Observed campaign spend/day, latest eligible campaign per household"
    )
    design["variance_spend_day"] = round(outcome_variance, 4)
    path = out_dir / "ab_test_design.json"
    path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    print(
        f"Experiment proposal saved to {path}: {design['total_sample_size']:,} households required."
    )


if __name__ == "__main__":
    main()
