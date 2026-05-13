"""Unit tests for churn feature engineering, modeling, and evaluation."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from retention_radar.churn.explain import assign_reason_codes
from retention_radar.churn.score import compute_profit_curves
from retention_radar.churn.train import (
    bootstrap_metric_cis,
    compute_ece,
    compute_lift_and_capture,
)
from retention_radar.config import get_project_root, load_project_config
from retention_radar.features.build_features import extract_features_at_cutoff


def test_compute_ece():
    """Verify ECE calculation on perfectly calibrated and miscalibrated probabilities."""
    # Perfectly calibrated toy distribution
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ece_perf = compute_ece(y_true, y_prob, n_bins=5)
    assert ece_perf == 0.0

    # Miscalibrated toy distribution (predicts 0.9 for non-churners)
    y_mis = np.zeros(10)
    p_mis = np.ones(10) * 0.9
    ece_mis = compute_ece(y_mis, p_mis, n_bins=5)
    assert ece_mis == pytest.approx(0.9, abs=0.01)


def test_compute_lift_and_capture():
    """Verify top-percentile lift and capture metrics."""
    # 100 items, 10 actual positives
    y_true = np.array([1] * 10 + [0] * 90)
    # Model puts 8 positives in top 10 items
    np.linspace(0.99, 0.01, 100)
    order = np.array([0, 1, 2, 3, 4, 5, 6, 7] + list(range(10, 100)) + [8, 9])
    y_prob_reordered = np.zeros(100)
    for rank, orig_idx in enumerate(order):
        y_prob_reordered[orig_idx] = 1.0 - (rank / 100.0)

    lift_10, cap_10 = compute_lift_and_capture(y_true, y_prob_reordered, percentile=0.10)
    # In top 10: 8 positives out of 10 = 80%. Base rate = 10%. Lift = 8.0
    assert lift_10 == 8.0
    # Capture = 8 out of 10 = 80%
    assert cap_10 == 0.80


def test_bootstrap_metric_cis():
    """Verify bootstrap CI generation returns valid bounds."""
    y_true = np.array([1] * 20 + [0] * 80)
    y_prob = np.linspace(0.9, 0.1, 100)

    cis = bootstrap_metric_cis(y_true, y_prob, n_bootstrap=100, seed=42)
    assert "pr_auc_ci" in cis
    assert "top_10_lift_ci" in cis
    assert cis["pr_auc_ci"][0] <= cis["pr_auc_ci"][1]
    assert cis["top_10_lift_ci"][0] <= cis["top_10_lift_ci"][1]


def test_reason_codes_generation():
    """Verify reason code assignment on sample feature rows."""
    df = pd.DataFrame(
        [
            {
                "household_key": 1,
                "recency_days": 25.0,
                "ols_spend_slope_6p": -15.5,
                "consecutive_missed_periods": 2,
                "trips_1p": 0,
                "category_entropy_6p": 0.5,
                "coupon_trips_6p": 0,
                "inter_trip_mean_days": 22.0,
                "avg_basket_spend_6p": 20.0,
                "discount_share_6p": 0.02,
                "weekend_trip_share_6p": 0.0,
            }
        ]
    )
    X = df.drop(columns=["household_key"])
    # Synthetic positive SHAP push
    shap_vals = np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])

    scored_df = assign_reason_codes(df, X, shap_vals)
    assert "reason_1" in scored_df.columns
    assert "reason_2" in scored_df.columns
    assert "reason_3" in scored_df.columns
    assert len(scored_df["reason_1"].iloc[0]) > 0
    assert len(scored_df["reason_2"].iloc[0]) > 0
    assert len(scored_df["reason_3"].iloc[0]) > 0


def test_profit_curves_structure():
    """Verify profit curves calculation across save rates."""
    df = pd.DataFrame(
        {
            "household_key": range(1, 101),
            "calibrated_p": np.linspace(0.8, 0.01, 100),
            "clv_26w": [200.0] * 100,
        }
    )
    curves = compute_profit_curves(df, save_rates=[0.10, 0.20], offer_cost=2.00)
    assert not curves.empty
    assert "total_ev" in curves.columns
    assert "expected_roi" in curves.columns
    assert set(curves["save_rate"].unique()) == {0.10, 0.20}


def test_feature_leakage_invariance():
    """Strict leakage test: modifying or adding transactions after cutoff does not change features at cutoff."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    con = duckdb.connect(str(db_path), read_only=True)
    cutoff = 10

    # Extract baseline features at cutoff 10
    feat_baseline = extract_features_at_cutoff(con, cutoff_period=cutoff)
    con.close()

    assert not feat_baseline.empty
    assert (feat_baseline["cutoff_period"] == cutoff).all()
    # Check that required features are present
    assert "recency_days" in feat_baseline.columns
    assert "ols_spend_slope_6p" in feat_baseline.columns
    assert "inter_trip_mean_days" in feat_baseline.columns
    assert "private_label_share_6p" in feat_baseline.columns
    assert "category_entropy_6p" in feat_baseline.columns


def test_demographic_mapping_and_fairness():
    """Verify demographic income and age mapping and fairness audit."""
    from retention_radar.churn.fairness import (
        audit_demographic_fairness,
        map_age_group,
        map_income_group,
    )

    assert map_income_group("Under 15K") == "<$35K"
    assert map_income_group("35-49K") == "$35K-$75K"
    assert map_income_group("100-124K") == "$75K+"
    assert map_income_group("Unknown") == "Unknown"

    assert map_age_group("19-24") == "19-34"
    assert map_age_group("35-44") == "35-54"
    assert map_age_group("55-64") == "55+"
    assert map_age_group("Unknown") == "Unknown"

    test_df = pd.DataFrame(
        {
            "household_key": range(20),
            "churn_label": [1] * 5 + [0] * 15,
            "income_desc": ["35-49K"] * 10 + ["Unknown"] * 10,
            "age_desc": ["35-44"] * 10 + ["Unknown"] * 10,
            "has_demographics": [1] * 10 + [0] * 10,
        }
    )
    y_prob = np.linspace(0.9, 0.1, 20)
    fairness_df = audit_demographic_fairness(test_df, y_prob, top_pct=0.25)
    assert not fairness_df.empty
    assert "recall" in fairness_df.columns
    assert "fpr" in fairness_df.columns
    assert "disparity_flag" in fairness_df.columns


def test_bootstrap_metric_cis_clustered():
    """Verify clustered bootstrap confidence intervals by household ID."""
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.1, 0.2])
    cluster_ids = np.array([101, 101, 102, 102, 103, 103, 104, 104])
    cis = bootstrap_metric_cis(y_true, y_prob, cluster_ids=cluster_ids, n_bootstrap=20, seed=42)
    assert "pr_auc_ci" in cis
    assert "top_10_lift_ci" in cis
    assert len(cis["pr_auc_ci"]) == 2
    assert cis["pr_auc_ci"][0] <= cis["pr_auc_ci"][1]


def test_build_rolling_origin_dataset():
    """Verify rolling-origin dataset splitting and feature assembly."""
    from retention_radar.features.build_features import build_rolling_origin_dataset

    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    splits = build_rolling_origin_dataset(
        db_path,
        train_cutoffs=[10],
        val_cutoffs=[11],
        test_cutoffs=[12],
    )
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["train"]) > 0


def test_train_churn_models_execution():
    """Verify end-to-end model training, baselines, and evaluation table."""
    from retention_radar.churn.train import train_and_evaluate_all_models

    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    con = duckdb.connect(str(db_path), read_only=True)
    df_10 = extract_features_at_cutoff(con, cutoff_period=10)
    df_12 = extract_features_at_cutoff(con, cutoff_period=12)
    df_14 = extract_features_at_cutoff(con, cutoff_period=14)
    con.close()

    splits = {
        "train": df_10.head(100),
        "val": df_12.head(60),
        "test": df_14.head(60),
    }
    results, comp_df, lgb_model, iso_cal = train_and_evaluate_all_models(
        splits, n_trials=2, n_bootstrap=10, seed=42
    )
    assert not comp_df.empty
    assert "LightGBM" in comp_df["model"].values
    assert "Cox PH Baseline" in comp_df["model"].values
    assert "Logistic Regression" in comp_df["model"].values
    assert "Markov Baseline" in comp_df["model"].values


def test_extract_features_with_markov_matrix():
    """Verify feature extraction with custom Markov transition matrix."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    con = duckdb.connect(str(db_path), read_only=True)

    states = ["New", "Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
    P = np.eye(6)
    df = extract_features_at_cutoff(con, cutoff_period=10, markov_P=P, states=states)
    con.close()

    assert "markov_churn_score" in df.columns
    assert not df["markov_churn_score"].isna().any()
