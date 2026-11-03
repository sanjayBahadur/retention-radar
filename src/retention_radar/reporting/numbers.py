"""Build public findings from saved artifacts, with explicit scenario assumptions."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from retention_radar.config import get_project_root, load_project_config

REQUIRED_ARTIFACTS = (
    "models/churn_lightgbm_metrics.json",
    "clv_at_churn_cutoff.parquet",
    "markov_metrics.json",
    "attribution.parquet",
    "clv_comparison.parquet",
    "clv_summary.json",
    "churn_model_comparison.parquet",
    "churn_features.parquet",
    "churn_scores.parquet",
    "profit_curve.parquet",
    "uplift_eval.parquet",
    "uplift_scores.parquet",
    "uplift_roi.parquet",
    "uplift_metadata.json",
    "ab_test_design.json",
    "basket_rules.parquet",
    "early_warning_categories.parquet",
    "drift_report.parquet",
)


def compile_results_json(out_dir: Path) -> dict:
    """Fail on missing inputs; derive findings rather than silently using example numbers."""
    missing = [name for name in REQUIRED_ARTIFACTS if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))
    cfg = load_project_config()

    def table(name):
        return pd.read_parquet(out_dir / f"{name}.parquet")

    def document(name):
        return json.loads((out_dir / f"{name}.json").read_text())

    comparison = table("churn_model_comparison")
    churn = comparison.set_index("model").loc["LightGBM"]
    scores = table("churn_scores")
    if scores.household_key.duplicated().any():
        raise ValueError("Churn roster must contain one row per household.")
    features = table("churn_features")
    test = features[features["split"] == "test"]
    clv_mean = float(scores.clv_26w.mean())
    capture = float(churn.top_20_capture)
    base_rate = float(test.churn_label.mean())
    # Illustrative population, not a claim of measured intervention effects.
    population, contact_share, save_rate = 100_000, 0.20, 0.10
    contacts = int(population * contact_share)
    captured = population * base_rate * capture
    gross = captured * save_rate * clv_mean
    cost = contacts * cfg.offer_cost
    frontier = table("profit_curve")
    at_default = frontier[frontier.save_rate == save_rate]
    best = at_default.loc[at_default.total_ev.idxmax()]

    uplift_meta = document("uplift_metadata")
    uplift_eval = table("uplift_eval").set_index("learner").loc[uplift_meta["selected_learner"]]
    uplift_scores = table("uplift_scores")
    if uplift_scores.household_key.duplicated().any():
        raise ValueError("Uplift roster must contain one row per household.")
    policy = table("uplift_roi").set_index("targeting_percentile").loc[30]
    segments = uplift_scores.segment.value_counts()
    clv_comp = table("clv_comparison").sort_values("spearman_rank_corr", ascending=False)
    clv_best = clv_comp.set_index("method").loc[document("clv_summary")["chosen_method"]]
    attribution = table("attribution")
    shares = dict(zip(attribution.campaign_type, attribution.markov_removal_effect_share))
    markov = document("markov_metrics")
    drift = table("drift_report")
    rules = table("basket_rules")
    warning = table("early_warning_categories")
    results = {
        "project": "Retention Radar: deciding where retention outreach deserves a test",
        "author": "Sanjay Bahadur",
        "last_updated": datetime.now(timezone.utc).date().isoformat(),
        "evidence": {
            "churn": "out-of-time prediction",
            "market_basket": "observed association",
            "uplift": "observational estimate",
            "financial_impact": "modeled scenario",
        },
        "markov": {key: markov[key] for key in ("expected_days_to_lapse", "markov_reward_clv")},
        "attribution": {
            "removal_effect_shares": shares,
            "top_contributor": max(shares, key=shares.get),
        },
        "clv": {
            "selected_method": document("clv_summary")["chosen_method"],
            "roster_method": "LightGBM fixed before the historical scoring cutoff",
            "holdout_spearman_rank_corr": float(clv_best.spearman_rank_corr),
            "holdout_mae_dollars": float(clv_best.mae_dollars),
            "holdout_rmse_dollars": float(clv_best.rmse_dollars),
            "error_units": "sales dollars over 26 weeks",
            "mean_26w_clv_margin_dollars": round(clv_mean, 2),
            "validation": document("clv_summary"),
        },
        "churn": {
            "best_model": "LightGBM (selected for the demonstration; compare with logistic regression)",
            "test_pr_auc": float(churn.pr_auc),
            "test_pr_auc_ci_95": [float(churn.pr_auc_ci_lower), float(churn.pr_auc_ci_upper)],
            "test_roc_auc": float(churn.roc_auc),
            "test_brier_score": float(churn.brier_score),
            "expected_calibration_error_ece": float(churn.ece),
            "top_10pct_lift": float(churn.top_10_lift),
            "top_10pct_capture_rate": float(churn.top_10_capture),
            "top_20pct_capture_rate": capture,
            "optimal_targeting_threshold_p": float(best.probability_threshold),
            "test_observations": len(test),
            "test_households": int(test.household_key.nunique()),
            "base_churn_rate": base_rate,
            "scored_households": len(scores),
            "roster_cutoff_period": int(scores.cutoff_period.max()),
        },
        "financial_impact": {
            "interpretation": "Illustrative scenario, not realized profit. Mean CLV approximates value of captured households; response is assumed.",
            "scale_households": population,
            "base_churn_rate": base_rate,
            "expected_churners": population * base_rate,
            "top_20pct_contacted_households": contacts,
            "contact_cost_per_household_dollars": cfg.offer_cost,
            "total_campaign_cost_dollars": cost,
            "top_20pct_churn_capture_rate": capture,
            "captured_churners": captured,
            "intervention_save_rate": save_rate,
            "saved_households": captured * save_rate,
            "mean_26w_clv_margin_dollars": clv_mean,
            "gross_margin_retained_dollars": round(gross, 2),
            "net_margin_per_26w_cycle_dollars": round(gross - cost, 2),
        },
        "uplift": {
            "dunnhumby_best_learner": uplift_meta["selected_learner"],
            "dunnhumby_qini": float(uplift_eval.qini),
            "dunnhumby_auuc": float(uplift_eval.auuc),
            "uplift_at_30pct_spend_day": float(uplift_eval.uplift_at_30pct),
            "top_30pct_policy_roi": float(policy.roi_ratio),
            "top_30pct_net_profit": float(policy.net_profit),
            "random_targeting_net_profit": float(policy.random_net_profit),
            "segments": {
                "persuadables_count": int(segments.get("Persuadables", 0)),
                "sure_things_count": int(segments.get("Sure Things", 0)),
                "sleeping_dogs_count": int(segments.get("Sleeping Dogs", 0)),
                "lost_causes_count": int(segments.get("Lost Causes", 0)),
            },
            "metadata": uplift_meta,
            "ab_test": document("ab_test_design"),
        },
        "market_basket": {
            "total_rules_lift_above_1_2": len(rules),
            "max_lift": float(rules.lift.max()),
            "top_early_warning_categories": warning.commodity_desc.head(5).tolist(),
            "effect_unit": "percentage-point change in category spend share relative to controls",
        },
        "monitoring": {
            "metrics_monitored": len(drift),
            "stable_metrics_count": int((drift.drift_status == "Stable").sum()),
            "retraining_alert_flag": bool(drift.alert_flag.any()),
        },
        "artifact_sha256": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in REQUIRED_ARTIFACTS
        },
    }
    # Benchmark files are optional and have their own scope; absence is not a pass.
    results["benchmarks"] = {
        name: document(name)
        if (out_dir / f"{name}.json").is_file()
        else {"status": "not available"}
        for name in ("hillstrom_benchmark", "x5_benchmark")
    }
    return results


def main() -> None:
    root = get_project_root()
    out_dir = root / load_project_config().paths.data_outputs
    results = compile_results_json(out_dir)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Compiled results.json from", len(REQUIRED_ARTIFACTS), "required artifacts.")


if __name__ == "__main__":
    main()
