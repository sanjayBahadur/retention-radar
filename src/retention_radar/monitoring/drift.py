"""Population Stability Index (PSI) drift monitoring for churn features and scores."""

from typing import List, Optional

import numpy as np
import pandas as pd

from retention_radar.config import get_project_root, load_project_config

MONITORED_FEATURES = [
    "recency_days",
    "spend_3p",
    "trips_1p",
    "trips_3p",
    "ols_spend_slope_6p",
    "baskets_6p",
    "avg_basket_spend_6p",
    "category_entropy_6p",
    "discount_share_6p",
    "inter_trip_mean_days",
]


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Calculate Population Stability Index (PSI) with Laplace smoothing."""
    # Use percentiles of expected/baseline to define bin edges
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Laplace smoothing to prevent division by zero or log of zero
    k = len(exp_counts)
    exp_pct = (exp_counts + 1e-4) / (len(expected) + 1e-4 * k)
    act_pct = (act_counts + 1e-4) / (len(actual) + 1e-4 * k)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def compute_drift_report(
    features_df: pd.DataFrame,
    scores_df: Optional[pd.DataFrame] = None,
    features: List[str] = MONITORED_FEATURES,
) -> pd.DataFrame:
    """Compute drift report comparing baseline training periods to latest cutoff partition."""
    # Baseline: train cutoffs (6-15)
    baseline_df = features_df[features_df["split"] == "train"]

    # Current / Latest: latest cutoff in test partition
    latest_cutoff = features_df["cutoff_period"].max()
    current_df = features_df[features_df["cutoff_period"] == latest_cutoff]

    records = []

    for feat in features:
        if feat not in features_df.columns:
            continue
        exp_vals = baseline_df[feat].dropna().values
        act_vals = current_df[feat].dropna().values

        if len(exp_vals) == 0 or len(act_vals) == 0:
            continue

        psi = calculate_psi(exp_vals, act_vals)

        # Drift classification rules:
        # PSI < 0.10: Stable
        # 0.10 <= PSI <= 0.20: Moderate Shift
        # PSI > 0.20: Significant Drift (retraining alert)
        if psi < 0.10:
            status = "Stable"
            alert = False
        elif psi <= 0.20:
            status = "Moderate Shift"
            alert = False
        else:
            status = "Significant Drift"
            alert = True

        records.append(
            {
                "feature_name": feat,
                "baseline_mean": round(float(np.mean(exp_vals)), 4),
                "current_mean": round(float(np.mean(act_vals)), 4),
                "psi": psi,
                "drift_status": status,
                "alert_flag": alert,
            }
        )

    # Model score drift: compare test start (cutoff 20) vs latest test (cutoff 23)
    if "markov_churn_score" in features_df.columns:
        exp_scores = (
            features_df[features_df["cutoff_period"] == 20]["markov_churn_score"].dropna().values
        )
        act_scores = current_df["markov_churn_score"].dropna().values
        if len(exp_scores) > 0 and len(act_scores) > 0:
            psi_score = calculate_psi(exp_scores, act_scores)
            records.append(
                {
                    "feature_name": "churn_model_score",
                    "baseline_mean": round(float(np.mean(exp_scores)), 4),
                    "current_mean": round(float(np.mean(act_scores)), 4),
                    "psi": psi_score,
                    "drift_status": "Stable" if psi_score < 0.20 else "Significant Drift",
                    "alert_flag": bool(psi_score > 0.20),
                }
            )

    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    """Run drift monitoring and write drift_report.parquet."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / "churn_features.parquet"
    scores_path = out_dir / "churn_scores.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}; run churn feature extraction first.")

    features_df = pd.read_parquet(feat_path)
    scores_df = pd.read_parquet(scores_path) if scores_path.exists() else None

    print("Running PSI drift monitoring across churn features and score distribution...")
    report_df = compute_drift_report(features_df, scores_df)
    print("\n--- PSI Drift Report ---")
    print(report_df.to_string())

    out_path = out_dir / "drift_report.parquet"
    report_df.to_parquet(out_path, index=False)
    print(f"\nSaved drift report to {out_path} ({len(report_df)} metrics monitored).")


if __name__ == "__main__":
    from typing import Optional

    main()
