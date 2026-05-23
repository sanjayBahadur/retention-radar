"""Pandera schemas for strict runtime and pipeline output table validation."""

from pathlib import Path
from typing import Dict

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from retention_radar.config import get_project_root, load_project_config

# 1. State Distribution Schema
state_distribution_schema = DataFrameSchema(
    columns={
        "period_id": Column(pa.Int, Check.greater_than_or_equal_to(1)),
        "state": Column(pa.String),
        "household_count": Column(pa.Int, Check.greater_than_or_equal_to(0)),
        "pct_of_period": Column(pa.Float, Check.in_range(0.0, 100.0)),
    },
    strict=False,
)

# 2. Transition Matrix Schema
transition_matrix_schema = DataFrameSchema(
    columns={
        "Champion": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "Regular": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "At-Risk": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "Dormant": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "Lapsed": Column(pa.Float, Check.in_range(0.0, 1.0)),
    },
    strict=False,
)

# 3. Campaign Attribution Schema
attribution_schema = DataFrameSchema(
    columns={
        "campaign_type": Column(pa.String),
        "first_touch_share": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "last_touch_share": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "linear_share": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "markov_removal_effect_share": Column(pa.Float, Check.in_range(0.0, 1.0)),
    },
    strict=False,
)

# 4. CLV Schema
clv_schema = DataFrameSchema(
    columns={
        "household_key": Column(pa.Int),
        "method": Column(pa.String),
        "clv_26w": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "p_alive": Column(pa.Float, Check.in_range(0.0, 1.0)),
    },
    strict=False,
)

# 5. CLV Comparison Schema
clv_comparison_schema = DataFrameSchema(
    columns={
        "method": Column(pa.String),
        "spearman_rank_corr": Column(pa.Float, Check.in_range(-1.0, 1.0)),
        "mae_dollars": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "rmse_dollars": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
    },
    strict=False,
)

# 6. Churn Features Schema
churn_features_schema = DataFrameSchema(
    columns={
        "household_key": Column(pa.Int, coerce=True),
        "cutoff_period": Column(pa.Int, coerce=True),
        "current_state": Column(pa.String),
        "recency_days": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "spend_3p": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "churn_label": Column(pa.Int, Check.isin([0, 1]), coerce=True),
        "split": Column(pa.String, Check.isin(["train", "val", "test"])),
    },
    strict=False,
    coerce=True,
)

# 7. Churn Model Comparison Schema
churn_model_comparison_schema = DataFrameSchema(
    columns={
        "model": Column(pa.String),
        "pr_auc": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "roc_auc": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "brier_score": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "ece": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "top_10_lift": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
    },
    strict=False,
)

# 8. Churn Scores Schema
churn_scores_schema = DataFrameSchema(
    columns={
        "household_key": Column(pa.Int, coerce=True),
        "cutoff_period": Column(pa.Int, coerce=True),
        "current_state": Column(pa.String),
        "calibrated_p": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "risk_decile": Column(pa.Int, Check.in_range(1, 10), coerce=True),
        "clv_26w": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "targeting_ev": Column(pa.Float),
        "target_action": Column(pa.String),
        "reason_1": Column(pa.String),
        "reason_2": Column(pa.String),
        "reason_3": Column(pa.String),
    },
    strict=False,
    coerce=True,
)

# 9. Profit Curve Schema
profit_curve_schema = DataFrameSchema(
    columns={
        "save_rate": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "target_percentile": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "households_targeted": Column(pa.Int, Check.greater_than_or_equal_to(0)),
        "total_ev": Column(pa.Float),
        "total_cost": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "expected_roi": Column(pa.Float),
    },
    strict=False,
)

# 10. Uplift Balance Schema
uplift_balance_schema = DataFrameSchema(
    columns={
        "covariate": Column(pa.String),
        "raw_smd": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "ipw_smd": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "balanced_after_ipw": Column(pa.Bool),
    },
    strict=False,
)

# 11. Uplift Confounding Test Schema
uplift_confounding_schema = DataFrameSchema(
    columns={
        "method": Column(pa.String),
        "estimate": Column(pa.Float),
        "bias": Column(pa.Float),
    },
    strict=False,
)

# 12. Uplift Evaluation Schema
uplift_eval_schema = DataFrameSchema(
    columns={
        "learner": Column(pa.String),
        "qini": Column(pa.Float),
        "auuc": Column(pa.Float),
        "uplift_at_10pct": Column(pa.Float),
        "uplift_at_30pct": Column(pa.Float),
        "mean_cate": Column(pa.Float),
    },
    strict=False,
)

# 13. Uplift ROI Schema
uplift_roi_schema = DataFrameSchema(
    columns={
        "targeting_percentile": Column(pa.Int, Check.in_range(1, 100)),
        "households_targeted": Column(pa.Int, Check.greater_than(0)),
        "mean_uplift_spend_per_day": Column(pa.Float),
        "incremental_margin": Column(pa.Float),
        "offer_cost": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "net_profit": Column(pa.Float),
        "roi_ratio": Column(pa.Float),
    },
    strict=False,
)

# 14. Uplift Scores Schema
uplift_scores_schema = DataFrameSchema(
    columns={
        "household_key": Column(pa.Int, coerce=True),
        "campaign": Column(pa.Int, coerce=True),
        "treated": Column(pa.Int, Check.isin([0, 1]), coerce=True),
        "propensity_score": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "uplift_tau": Column(pa.Float),
        "segment": Column(
            pa.String, Check.isin(["Persuadables", "Sure Things", "Sleeping Dogs", "Lost Causes"])
        ),
    },
    strict=False,
    coerce=True,
)

# 15. Basket Rules Schema
basket_rules_schema = DataFrameSchema(
    columns={
        "antecedents_str": Column(pa.String),
        "consequents_str": Column(pa.String),
        "support": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "confidence": Column(pa.Float, Check.in_range(0.0, 1.0)),
        "lift": Column(pa.Float, Check.greater_than_or_equal_to(1.20)),
    },
    strict=False,
)

# 16. Early Warning Categories Schema
early_warning_schema = DataFrameSchema(
    columns={
        "commodity_desc": Column(pa.String),
        "lapsing_hh_count": Column(pa.Int, Check.greater_than(0)),
        "net_drop_effect": Column(pa.Float),
        "rank": Column(pa.Int, Check.greater_than(0)),
    },
    strict=False,
)

# 17. Drift Report Schema
drift_report_schema = DataFrameSchema(
    columns={
        "feature_name": Column(pa.String),
        "baseline_mean": Column(pa.Float),
        "current_mean": Column(pa.Float),
        "psi": Column(pa.Float, Check.greater_than_or_equal_to(0.0)),
        "drift_status": Column(pa.String),
        "alert_flag": Column(pa.Bool),
    },
    strict=False,
)

SCHEMA_REGISTRY = {
    "state_distribution.parquet": state_distribution_schema,
    "transition_matrix.parquet": transition_matrix_schema,
    "attribution.parquet": attribution_schema,
    "clv.parquet": clv_schema,
    "clv_comparison.parquet": clv_comparison_schema,
    "churn_features.parquet": churn_features_schema,
    "churn_model_comparison.parquet": churn_model_comparison_schema,
    "churn_scores.parquet": churn_scores_schema,
    "profit_curve.parquet": profit_curve_schema,
    "uplift_balance.parquet": uplift_balance_schema,
    "uplift_confounding_test.parquet": uplift_confounding_schema,
    "uplift_eval.parquet": uplift_eval_schema,
    "uplift_roi.parquet": uplift_roi_schema,
    "uplift_scores.parquet": uplift_scores_schema,
    "basket_rules.parquet": basket_rules_schema,
    "early_warning_categories.parquet": early_warning_schema,
    "drift_report.parquet": drift_report_schema,
}


def validate_all_parquet_outputs(out_dir: Path) -> Dict[str, bool]:
    """Validate all generated Parquet files against their respective Pandera schemas."""
    validation_results = {}
    for filename, schema in SCHEMA_REGISTRY.items():
        file_path = out_dir / filename
        if not file_path.exists():
            continue
        try:
            df = pd.read_parquet(file_path)
            schema.validate(df)
            validation_results[filename] = True
        except Exception as e:
            print(f"Validation failed for {filename}: {e}")
            validation_results[filename] = False

    return validation_results


def main() -> None:
    """Validate all parquet files in data/outputs/."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs

    print("Validating all Parquet artifacts against Pandera schemas...")
    results = validate_all_parquet_outputs(out_dir)
    for fn, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {fn}: {status}")

    all_passed = all(results.values())
    assert all_passed, "One or more Parquet files failed Pandera schema validation!"
    print(f"\nAll {len(results)} Parquet artifacts strictly conform to Pandera schemas.")


if __name__ == "__main__":
    main()
