"""SHAP explainability and human-readable reason codes for churn prediction."""

from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_explanations(
    lgb_model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    figures_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    """Compute SHAP values, plot beeswarm and top feature dependence plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)

    # In binary classification, shap_values might be a list [class 0, class 1] or a single array
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # 1. Beeswarm summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test, max_display=15, show=False)
    plt.title("SHAP Feature Importance (Churn Prediction)", fontsize=13, pad=15)
    plt.tight_layout()
    beeswarm_path = figures_dir / "churn_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()

    # 2. Top 5 feature dependence plots
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    top_5_idx = np.argsort(-mean_abs_shap)[:5]
    top_5_features = [X_test.columns[i] for i in top_5_idx]

    for feat in top_5_features:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat, sv, X_test, show=False)
        plt.title(f"SHAP Dependence: {feat}", fontsize=12)
        plt.tight_layout()
        dep_path = figures_dir / f"churn_shap_dep_{feat}.png"
        plt.savefig(dep_path, dpi=150)
        plt.close()

    return sv, top_5_features


def assign_reason_codes(
    df: pd.DataFrame,
    X: pd.DataFrame,
    shap_matrix: np.ndarray,
) -> pd.DataFrame:
    """Generate top-3 human-readable churn reason codes for each household based on SHAP push."""
    reason_1 = []
    reason_2 = []
    reason_3 = []

    feature_names = list(X.columns)

    for idx in range(len(df)):
        row_feat = X.iloc[idx]
        row_shap = shap_matrix[idx]

        # Rank features by positive SHAP contribution (pushing toward churn)
        ranked_indices = np.argsort(-row_shap)

        candidate_reasons: List[str] = []
        for feat_idx in ranked_indices:
            feat_name = feature_names[feat_idx]
            val = row_feat[feat_name]

            if feat_name == "recency_days" and val > 14:
                candidate_reasons.append(f"High recency: no store visit in {int(val)} days")
            elif feat_name == "ols_spend_slope_6p" and val < 0:
                candidate_reasons.append(f"Declining spend trend (${abs(val):.2f}/period slope)")
            elif feat_name == "consecutive_missed_periods" and val >= 1:
                candidate_reasons.append(f"Missed {int(val)} consecutive purchase periods")
            elif feat_name == "trips_1p" and val == 0:
                candidate_reasons.append("Zero visits in the most recent 28-day period")
            elif feat_name == "category_entropy_6p" and val < 1.2:
                candidate_reasons.append("Narrow basket variety / single-department concentration")
            elif feat_name == "coupon_trips_6p" and val == 0:
                candidate_reasons.append("Zero coupon engagement over the last 6 periods")
            elif feat_name == "inter_trip_mean_days" and val > 21:
                candidate_reasons.append(f"Long inter-trip intervals ({val:.1f} days average)")
            elif feat_name == "avg_basket_spend_6p" and val < 25:
                candidate_reasons.append(f"Low average basket spend (${val:.2f})")
            elif feat_name == "discount_share_6p" and val < 0.05:
                candidate_reasons.append("Low promotion responsiveness (< 5% discount share)")
            elif feat_name == "weekend_trip_share_6p" and val == 0:
                candidate_reasons.append("Zero weekend shopping activity")

            if len(candidate_reasons) >= 3:
                break

        # Defaults if fewer than 3 reasons identified
        defaults = [
            "Below-average shopping frequency",
            "Trailing spend below household baseline",
            "Reduced seasonal basket size",
        ]
        while len(candidate_reasons) < 3:
            for d in defaults:
                if d not in candidate_reasons:
                    candidate_reasons.append(d)
                if len(candidate_reasons) == 3:
                    break

        reason_1.append(candidate_reasons[0])
        reason_2.append(candidate_reasons[1])
        reason_3.append(candidate_reasons[2])

    result_df = df.copy()
    result_df["reason_1"] = reason_1
    result_df["reason_2"] = reason_2
    result_df["reason_3"] = reason_3
    return result_df
