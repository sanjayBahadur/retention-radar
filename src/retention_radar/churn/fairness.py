"""Demographic fairness audit and bias analysis for churn prediction."""

import json
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from retention_radar.churn.train import prepare_feature_matrices
from retention_radar.config import get_project_root, load_project_config


def map_income_group(income_desc: str) -> str:
    """Map raw Dunnhumby income descriptions to standard bands."""
    if pd.isna(income_desc) or income_desc == "Unknown":
        return "Unknown"
    if any(k in income_desc for k in ["Under 15K", "15-24K", "25-34K"]):
        return "<$35K"
    if any(k in income_desc for k in ["35-49K", "50-74K"]):
        return "$35K-$75K"
    if any(
        k in income_desc
        for k in ["75-99K", "100-124K", "125-149K", "150-174K", "175-199K", "200-249K", "250K+"]
    ):
        return "$75K+"
    return "Unknown"


def map_age_group(age_desc: str) -> str:
    """Map raw Dunnhumby age descriptions to standard bands."""
    if pd.isna(age_desc) or age_desc == "Unknown":
        return "Unknown"
    if "19-24" in age_desc or "25-34" in age_desc:
        return "19-34"
    if "35-44" in age_desc or "45-54" in age_desc:
        return "35-54"
    if "55-64" in age_desc or "65+" in age_desc:
        return "55+"
    return "Unknown"


def audit_demographic_fairness(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    top_pct: float = 0.20,
) -> pd.DataFrame:
    """Audit recall and FPR across income and age demographic bands at a given targeting cutoff."""
    n = len(test_df)
    k = max(1, int(n * top_pct))
    threshold = np.sort(y_prob)[::-1][k]

    y_pred_binary = (y_prob >= threshold).astype(int)
    y_true = test_df["churn_label"].values

    df_eval = test_df.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred_binary
    df_eval["income_band"] = df_eval["income_desc"].apply(map_income_group)
    df_eval["age_band"] = df_eval["age_desc"].apply(map_age_group)

    records = []

    def compute_group_metrics(
        slice_name: str, group_label: str, sub: pd.DataFrame
    ) -> Dict[str, Any]:
        n_group = len(sub)
        n_pos = int(sub["y_true"].sum())
        n_neg = n_group - n_pos

        if n_pos == 0:
            recall = 0.0
        else:
            tp = int(((sub["y_pred"] == 1) & (sub["y_true"] == 1)).sum())
            recall = tp / n_pos

        if n_neg == 0:
            fpr = 0.0
        else:
            fp = int(((sub["y_pred"] == 1) & (sub["y_true"] == 0)).sum())
            fpr = fp / n_neg

        return {
            "dimension": slice_name,
            "group": group_label,
            "sample_size": n_group,
            "churn_count": n_pos,
            "churn_rate": round(float(n_pos / max(1, n_group)), 4),
            "recall": round(float(recall), 4),
            "fpr": round(float(fpr), 4),
        }

    # Overall baseline
    records.append(compute_group_metrics("Overall", "All Households", df_eval))

    # Income groups
    for group, sub in df_eval.groupby("income_band"):
        records.append(compute_group_metrics("Income", group, sub))

    # Age groups
    for group, sub in df_eval.groupby("age_band"):
        records.append(compute_group_metrics("Age", group, sub))

    # Demographic coverage
    for group, sub in df_eval.groupby("has_demographics"):
        label = "Known Demographics" if group == 1 else "Unknown (No Demographics)"
        records.append(compute_group_metrics("Demographic Status", label, sub))

    res_df = pd.DataFrame(records)

    # Flag disparities > 10 percentage points from overall recall or FPR
    overall_recall = res_df.loc[res_df["dimension"] == "Overall", "recall"].values[0]
    overall_fpr = res_df.loc[res_df["dimension"] == "Overall", "fpr"].values[0]

    res_df["recall_gap"] = (res_df["recall"] - overall_recall).abs().round(4)
    res_df["fpr_gap"] = (res_df["fpr"] - overall_fpr).abs().round(4)
    res_df["disparity_flag"] = (res_df["recall_gap"] > 0.10) | (res_df["fpr_gap"] > 0.10)

    return res_df


def train_model_without_demographics(
    splits: Dict[str, pd.DataFrame],
    best_params: Dict[str, Any],
    seed: int = 42,
) -> Tuple[lgb.LGBMClassifier, float, float]:
    """Train LightGBM without demographic features to assess fairness vs performance trade-off."""
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train_raw, y_train, _ = prepare_feature_matrices(train_df, include_demographics=False)
    X_val_raw, y_val, _ = prepare_feature_matrices(val_df, include_demographics=False)
    X_test_raw, y_test, _ = prepare_feature_matrices(test_df, include_demographics=False)

    all_cols = sorted(
        list(set(X_train_raw.columns) | set(X_val_raw.columns) | set(X_test_raw.columns))
    )
    X_train = X_train_raw.reindex(columns=all_cols, fill_value=0.0)
    X_val_raw.reindex(columns=all_cols, fill_value=0.0)
    X_test = X_test_raw.reindex(columns=all_cols, fill_value=0.0)

    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X_train, y_train)

    test_preds = clf.predict_proba(X_test)[:, 1]
    pr_auc_no_demo = float(average_precision_score(y_test, test_preds))
    roc_auc_no_demo = float(
        _lgb_model_roc_auc := average_precision_score(y_test, test_preds)
    )  # placeholder

    from sklearn.metrics import roc_auc_score

    roc_auc_no_demo = float(roc_auc_score(y_test, test_preds))

    return clf, round(pr_auc_no_demo, 4), round(roc_auc_no_demo, 4)


def main() -> None:
    """Run demographic fairness audit and export results."""
    project_root = get_project_root()
    cfg = load_project_config()

    feat_path = project_root / cfg.paths.data_outputs / "churn_features.parquet"
    df = pd.read_parquet(feat_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Load model and metrics
    models_dir = project_root / cfg.paths.models_dir
    with open(models_dir / "churn_lightgbm_metrics.json", "r") as f:
        meta = json.load(f)

    # Re-predict with best model

    from retention_radar.churn.train import prepare_feature_matrices

    booster = lgb.Booster(model_file=str(models_dir / "churn_lightgbm.txt"))
    X_test_raw, _, _ = prepare_feature_matrices(test_df)
    X_test = X_test_raw.reindex(columns=meta["feature_names"], fill_value=0.0)
    raw_preds = booster.predict(X_test)

    fairness_df = audit_demographic_fairness(test_df, raw_preds, top_pct=0.20)
    print("\n--- Demographic Fairness Audit: Baseline Model (Top 20% Targeting Cutoff) ---")
    print(fairness_df.to_string())

    out_dir = project_root / cfg.paths.data_outputs
    fairness_path = out_dir / "churn_fairness_audit.parquet"
    fairness_df.to_parquet(fairness_path, index=False)
    print(f"\nFairness audit saved to {fairness_path}")

    # Post-Ablation Fairness Audit (Model trained without demographic features)
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    clf_no_demo, pr_no_demo, roc_no_demo = train_model_without_demographics(
        splits, meta["best_params"]
    )
    X_test_no_demo_raw, _, _ = prepare_feature_matrices(test_df, include_demographics=False)
    X_test_no_demo = X_test_no_demo_raw.reindex(columns=clf_no_demo.feature_name_, fill_value=0.0)
    preds_no_demo = clf_no_demo.predict_proba(X_test_no_demo)[:, 1]

    post_fairness_df = audit_demographic_fairness(test_df, preds_no_demo, top_pct=0.20)
    print("\n--- Demographic Fairness Audit: Post-Ablation Model (Without Demographics) ---")
    print(post_fairness_df.to_string())

    post_fairness_path = out_dir / "churn_fairness_post_ablation.parquet"
    post_fairness_df.to_parquet(post_fairness_path, index=False)
    print(f"Post-ablation fairness audit saved to {post_fairness_path}")


if __name__ == "__main__":
    import json
    from typing import Any

    main()
