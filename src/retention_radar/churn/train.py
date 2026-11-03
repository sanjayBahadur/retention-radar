"""Model training and evaluation for customer churn prediction."""

import json
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from retention_radar.config import get_project_root, load_project_config

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

NUMERIC_FEATURES = [
    "consecutive_missed_periods",
    "tenure_periods",
    "recency_days",
    "trips_1p",
    "spend_1p",
    "trips_3p",
    "spend_3p",
    "trips_6p",
    "spend_6p",
    "baskets_6p",
    "avg_basket_spend_6p",
    "avg_basket_items_6p",
    "avg_dept_per_basket_6p",
    "weekend_trip_share_6p",
    "distinct_stores_6p",
    "discount_share_6p",
    "coupon_trips_6p",
    "inter_trip_mean_days",
    "inter_trip_std_days",
    "inter_trip_cv",
    "private_label_share_6p",
    "category_entropy_6p",
    "campaigns_last_3p",
    "ols_spend_slope_6p",
    "markov_churn_score",
    "has_demographics",
]

CATEGORICAL_FEATURES = [
    "current_state",
    "age_desc",
    "income_desc",
    "household_size_desc",
]


def prepare_feature_matrices(
    df: pd.DataFrame,
    include_demographics: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare encoded feature matrix and target series."""
    num_cols = list(NUMERIC_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)

    if not include_demographics:
        num_cols = [c for c in num_cols if c != "has_demographics"]
        cat_cols = [
            c for c in cat_cols if c not in ["age_desc", "income_desc", "household_size_desc"]
        ]

    X_num = df[num_cols].fillna(0.0).copy()
    X_cat = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["churn_label"].astype(int)
    feature_names = list(X.columns)
    return X, y, feature_names


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with uniform binning."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (
            (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if i < n_bins - 1
            else (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        )
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_lift_and_capture(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    percentile: float = 0.10,
    period_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Evaluate a contact budget separately at each cutoff, then pool counts.

    Without period_ids this retains the single-ranking behavior used by callers
    evaluating one cohort. Stable sorting gives ties the same treatment as the
    error review, without exceeding the contact budget.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    periods = np.zeros(len(y_true)) if period_ids is None else np.asarray(period_ids)
    if len(periods) != len(y_true):
        raise ValueError("Each prediction needs a matching period identifier.")
    selected = []
    for period in np.unique(periods):
        indices = np.flatnonzero(periods == period)
        top_k = max(1, int(len(indices) * percentile))
        ranked = np.argsort(-y_prob[indices], kind="stable")
        selected.extend(indices[ranked[:top_k]])
    top_y = y_true[selected]

    base_rate = np.mean(y_true)
    top_rate = np.mean(top_y)
    lift = top_rate / (base_rate + 1e-9)
    capture = np.sum(top_y) / (np.sum(y_true) + 1e-9)

    return round(float(lift), 3), round(float(capture), 4)


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    period_ids: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals (household-clustered when cluster_ids is provided)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if period_ids is not None:
        period_ids = np.asarray(period_ids)
        if len(period_ids) != n:
            raise ValueError("Each prediction needs a matching period identifier.")
    pr_aucs = []
    lifts = []

    if cluster_ids is not None:
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
        for _ in range(n_bootstrap):
            sampled_c = rng.choice(unique_clusters, size=n_clusters, replace=True)
            idx = np.concatenate([cluster_to_idx[c] for c in sampled_c])
            sample_y = y_true[idx]
            sample_p = y_prob[idx]
            if np.sum(sample_y) == 0:
                continue
            pr_aucs.append(average_precision_score(sample_y, sample_p))
            lift, _ = compute_lift_and_capture(
                sample_y,
                sample_p,
                percentile=0.10,
                period_ids=period_ids[idx] if period_ids is not None else None,
            )
            lifts.append(lift)
    else:
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            sample_y = y_true[idx]
            sample_p = y_prob[idx]
            if np.sum(sample_y) == 0:
                continue
            pr_aucs.append(average_precision_score(sample_y, sample_p))
            lift, _ = compute_lift_and_capture(
                sample_y,
                sample_p,
                percentile=0.10,
                period_ids=period_ids[idx] if period_ids is not None else None,
            )
            lifts.append(lift)

    return {
        "pr_auc_ci": (
            round(float(np.percentile(pr_aucs, 2.5)), 4),
            round(float(np.percentile(pr_aucs, 97.5)), 4),
        ),
        "top_10_lift_ci": (
            round(float(np.percentile(lifts, 2.5)), 3),
            round(float(np.percentile(lifts, 97.5)), 3),
        ),
    }


def fit_optuna_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    seed: int = 42,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Tune LightGBM hyperparameters with Optuna to maximize validation PR-AUC."""
    pos_weight = float((len(y_train) - np.sum(y_train)) / np.sum(y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pos_weight),
            "random_state": seed,
            "verbosity": -1,
            "n_jobs": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        val_preds = clf.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, val_preds))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({"random_state": seed, "verbosity": -1, "n_jobs": -1})
    best_clf = lgb.LGBMClassifier(**best_params)
    best_clf.fit(X_train, y_train)

    return best_clf, best_params


def train_and_evaluate_all_models(
    splits: Dict[str, pd.DataFrame],
    n_trials: int = 30,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, lgb.LGBMClassifier, IsotonicRegression]:
    """Train all four churn models, calibrate on validation, and evaluate on test set."""
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Align feature columns across splits
    X_train_raw, y_train, feat_cols = prepare_feature_matrices(train_df)
    X_val_raw, y_val, _ = prepare_feature_matrices(val_df)
    X_test_raw, y_test, _ = prepare_feature_matrices(test_df)

    # Reindex to ensure identical feature columns
    all_cols = sorted(X_train_raw.columns)
    X_train = X_train_raw.reindex(columns=all_cols, fill_value=0.0)
    X_val = X_val_raw.reindex(columns=all_cols, fill_value=0.0)
    X_test = X_test_raw.reindex(columns=all_cols, fill_value=0.0)

    y_test_arr = y_test.values

    predictions: Dict[str, np.ndarray] = {}
    calibrated_predictions: Dict[str, np.ndarray] = {}

    # Simple baselines learned from training data only.
    state_rates = train_df.assign(label=y_train.values).groupby("current_state")["label"].mean()
    state_scores = test_df["current_state"].map(state_rates).fillna(y_train.mean()).to_numpy()
    predictions["State Baseline"] = state_scores
    calibrated_predictions["State Baseline"] = state_scores

    # An explicit recency rule is a useful business baseline, without a survival
    # model fitted to repeated rows with incorrectly defined event durations.
    recency_train = train_df["recency_days"].to_numpy()
    recency_scores = np.searchsorted(np.sort(recency_train), test_df["recency_days"]) / len(
        train_df
    )
    predictions["Recency Baseline"] = recency_scores
    iso_recency = IsotonicRegression(out_of_bounds="clip")
    iso_recency.fit(train_df["recency_days"], y_train)
    calibrated_predictions["Recency Baseline"] = iso_recency.predict(test_df["recency_days"])

    # 3. Scaled Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(X_train_scaled, y_train)
    predictions["Logistic Regression"] = lr.predict_proba(X_test_scaled)[:, 1]

    # Isotonic calibration for Logistic Regression
    lr_val_preds = lr.predict_proba(X_val_scaled)[:, 1]
    iso_lr = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_lr.fit(lr_val_preds, y_val.values)
    calibrated_predictions["Logistic Regression"] = iso_lr.transform(
        predictions["Logistic Regression"]
    )

    # 4. Optuna-tuned LightGBM
    lgb_model, best_lgb_params = fit_optuna_lightgbm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=n_trials,
        seed=seed,
    )
    raw_lgb_val = lgb_model.predict_proba(X_val)[:, 1]
    raw_lgb_test = lgb_model.predict_proba(X_test)[:, 1]
    predictions["LightGBM"] = raw_lgb_test

    # Isotonic calibration for LightGBM fit on validation
    iso_lgb = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_lgb.fit(raw_lgb_val, y_val.values)
    calibrated_lgb_test = iso_lgb.transform(raw_lgb_test)
    calibrated_predictions["LightGBM"] = calibrated_lgb_test

    # Evaluate all models on test set
    comparison_rows = []
    for model_name, p_raw in predictions.items():
        p_cal = calibrated_predictions[model_name]
        pr_auc = round(float(average_precision_score(y_test_arr, p_raw)), 4)
        roc_auc = round(float(roc_auc_score(y_test_arr, p_raw)), 4)
        brier = round(float(brier_score_loss(y_test_arr, p_cal)), 4)
        ece = compute_ece(y_test_arr, p_cal)

        period_ids = test_df["cutoff_period"].to_numpy()
        lift_10, cap_10 = compute_lift_and_capture(
            y_test_arr, p_raw, percentile=0.10, period_ids=period_ids
        )
        lift_20, cap_20 = compute_lift_and_capture(
            y_test_arr, p_raw, percentile=0.20, period_ids=period_ids
        )

        # Household-clustered bootstrap CIs
        cluster_ids = (
            test_df["household_key"].values if "household_key" in test_df.columns else None
        )
        cis = bootstrap_metric_cis(
            y_test_arr,
            p_raw,
            cluster_ids=cluster_ids,
            n_bootstrap=n_bootstrap,
            seed=seed,
            period_ids=period_ids,
        )

        comparison_rows.append(
            {
                "model": model_name,
                "pr_auc": pr_auc,
                "pr_auc_ci_lower": cis["pr_auc_ci"][0],
                "pr_auc_ci_upper": cis["pr_auc_ci"][1],
                "roc_auc": roc_auc,
                "brier_score": brier,
                "ece": ece,
                "top_10_lift": lift_10,
                "top_10_lift_ci_lower": cis["top_10_lift_ci"][0],
                "top_10_lift_ci_upper": cis["top_10_lift_ci"][1],
                "top_10_capture": cap_10,
                "top_20_lift": lift_20,
                "top_20_capture": cap_20,
            }
        )

    comp_df = (
        pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    )

    artifacts = {
        "best_lgb_params": best_lgb_params,
        "feature_names": all_cols,
        "test_labels": y_test_arr,
        "test_calibrated_preds": calibrated_lgb_test,
        "test_raw_preds": raw_lgb_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    return artifacts, comp_df, lgb_model, iso_lgb


def main() -> None:
    """Run model training, validation, and benchmarking."""
    project_root = get_project_root()
    cfg = load_project_config()

    feat_path = project_root / cfg.paths.data_outputs / "churn_features.parquet"
    if not feat_path.exists():
        from retention_radar.features.build_features import main as build_feat_main

        build_feat_main()

    df = pd.read_parquet(feat_path)
    splits = {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }

    n_trials = cfg.optuna_trials if hasattr(cfg, "optuna_trials") else 30
    n_bootstrap = cfg.bootstrap_reps if hasattr(cfg, "bootstrap_reps") else 1000

    print(
        f"Training churn models with {n_trials} Optuna trials and {n_bootstrap} bootstrap reps..."
    )
    artifacts, comp_df, lgb_model, iso_lgb = train_and_evaluate_all_models(
        splits=splits,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        seed=cfg.seed,
    )

    out_dir = project_root / cfg.paths.data_outputs
    models_dir = project_root / cfg.paths.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model comparison parquet
    comp_path = out_dir / "churn_model_comparison.parquet"
    comp_df.to_parquet(comp_path, index=False)
    print("\nModel Comparison Table:")
    print(comp_df.to_string())

    predictions = splits["test"][
        ["household_key", "cutoff_period", "current_state", "tenure_periods", "churn_label"]
    ].copy()
    predictions["raw_p"] = artifacts["test_raw_preds"]
    predictions["calibrated_p"] = artifacts["test_calibrated_preds"]
    predictions.to_parquet(out_dir / "churn_test_predictions.parquet", index=False)

    # Save LightGBM model and metrics
    lgb_model.booster_.save_model(str(models_dir / "churn_lightgbm.txt"))
    metrics_summary = {
        "best_params": artifacts["best_lgb_params"],
        "feature_names": artifacts["feature_names"],
        "test_metrics": comp_df[comp_df["model"] == "LightGBM"].iloc[0].to_dict(),
    }
    with open(models_dir / "churn_lightgbm_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nArtifacts saved to {comp_path} and {models_dir}")


if __name__ == "__main__":
    main()
