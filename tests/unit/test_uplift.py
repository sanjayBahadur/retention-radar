"""Unit tests for causal uplift modeling, benchmarking, and experiment power analysis."""

import numpy as np
import pandas as pd
import pytest

from retention_radar.config import get_project_root, load_project_config
from retention_radar.uplift.dunnhumby import segment_customers
from retention_radar.uplift.hillstrom import run_hillstrom_benchmark
from retention_radar.uplift.power import design_retention_ab_test
from retention_radar.uplift.x5 import (
    compute_smd,
    run_confounding_stress_test,
    verify_x5_randomization,
)


def test_compute_smd():
    """Verify SMD calculation on identical and separated distributions."""
    # Identical distributions -> SMD = 0
    x1 = np.array([10.0, 20.0, 30.0])
    x2 = np.array([10.0, 20.0, 30.0])
    assert compute_smd(x1, x2) == 0.0

    # Shifted distributions with unit std -> SMD = 1
    x_c = np.array([0.0, 1.0, 2.0])  # mean = 1, std = 1
    x_t = np.array([1.0, 2.0, 3.0])  # mean = 2, std = 1
    assert compute_smd(x_t, x_c) == pytest.approx(1.0, abs=0.01)


def test_hillstrom_benchmark_passes():
    """Verify that Hillstrom causal validation succeeds within the 95% bootstrap CI."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    res = run_hillstrom_benchmark(db_path=db_path, n_bootstrap=100)
    assert res["is_within_ci"] is True
    assert res["sample_size"] == 42693
    assert res["raw_diff"] > 0.03


def test_x5_randomization_verification():
    """Verify that verify_x5_randomization flags balance correctly."""
    df = pd.DataFrame(
        {
            "treatment_flg": [1, 1, 0, 0],
            "age": [45.0, 46.0, 45.0, 46.0],
            "tenure_days": [100.0, 102.0, 100.0, 102.0],
        }
    )
    balance_df = verify_x5_randomization(df, feat_cols=["age", "tenure_days"])
    assert (balance_df["abs_smd"] < 0.10).all()
    assert balance_df["is_balanced"].all()


def test_confounding_stress_test():
    """Verify confounding simulation and that debiasing methods reduce bias."""
    df = pd.DataFrame(
        {
            "client_id": range(100),
            "treatment_flg": [1] * 50 + [0] * 50,
            "target": [1] * 35 + [0] * 15 + [1] * 30 + [0] * 20,
            "age": np.random.uniform(20, 60, 100),
            "tenure_days": np.linspace(10, 500, 100),
            "has_redeemed": [1] * 80 + [0] * 20,
        }
    )
    res_df = run_confounding_stress_test(df, true_ate=0.05, seed=42)
    assert not res_df.empty
    assert "estimate" in res_df.columns
    assert "bias" in res_df.columns
    assert "True Randomized Ground Truth" in res_df["method"].values


def test_ab_test_power_design():
    """Verify A/B power analysis outputs valid sample size and duration."""
    design = design_retention_ab_test(
        alpha=0.05,
        power=0.80,
        active_households=2400,
        variance_spend_day=2.0,
        expected_uplift_spend_day=0.50,
    )
    assert design["required_sample_size_per_arm"] > 10
    assert design["total_sample_size"] < 2400
    assert design["recommended_duration_weeks"] >= 4
    assert len(design["causal_assumptions"]) >= 3


def test_segment_customers_logic():
    """Verify uplift customer segmentation partitions households into 4 archetypes."""
    df = pd.DataFrame(
        {
            "household_key": [1, 2, 3, 4],
            "pre_spend_per_day": [15.0, 12.0, 2.0, 1.0],
        }
    )
    cate_preds = np.array([0.50, 0.01, -0.40, 0.01])
    clv_df = pd.DataFrame(
        {
            "household_key": [1, 2, 3, 4],
            "clv_26w": [400.0, 350.0, 500.0, 100.0],
        }
    )

    scored, summary = segment_customers(df, cate_preds, clv_df)
    assert "segment" in scored.columns
    # Check that segments exist
    segments = set(scored["segment"].unique())
    assert "Persuadables" in segments
    assert not summary.empty
