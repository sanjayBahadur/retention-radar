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


@pytest.mark.requires_warehouse
def test_hillstrom_benchmark_uses_holdout():
    """Report a held-out sanity comparison without requiring accidental CI agreement."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    res = run_hillstrom_benchmark(db_path=db_path, n_bootstrap=100)
    assert isinstance(res["is_within_ci"], bool)
    assert res["evaluation_sample_size"] < res["sample_size"]
    assert -1 <= res["causalml_t_learner_ate"] <= 1
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
    """Check age-selection diagnostics without promising reduced reference deviation."""
    df = pd.DataFrame(
        {
            "client_id": range(100),
            "treatment_flg": [1] * 50 + [0] * 50,
            "target": [1] * 35 + [0] * 15 + [1] * 30 + [0] * 20,
            "age": np.random.default_rng(42).uniform(20, 60, 100),
            "tenure_days": np.linspace(10, 500, 100),
            "has_redeemed": [1] * 80 + [0] * 20,
        }
    )
    res_df = run_confounding_stress_test(df, reference_ate=0.05, seed=42)
    assert not res_df.empty
    assert "estimate" in res_df.columns
    assert "reference_deviation" in res_df.columns
    assert "Randomized sample reference" in res_df["method"].values


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


def test_roi_rejects_repeated_households():
    """A campaign history cannot silently count the same contact multiple times."""
    from retention_radar.uplift.dunnhumby import compute_targeting_roi_curve

    with pytest.raises(ValueError, match="one row per household"):
        compute_targeting_roi_curve(pd.DataFrame({"household_key": [1, 1]}), np.array([1.0, 2.0]))


def test_roi_costs_and_margin_use_unique_households():
    from retention_radar.uplift.dunnhumby import compute_targeting_roi_curve

    rows = pd.DataFrame({"household_key": range(10)})
    result = compute_targeting_roi_curve(rows, np.ones(10), offer_cost=2, gross_margin=0.25)
    all_contacts = result.iloc[-1]
    assert all_contacts["households_targeted"] == 10
    assert all_contacts["offer_cost"] == 20
    assert all_contacts["net_profit"] == pytest.approx(10 * 35 * 0.25 - 20)
    assert all_contacts["random_net_profit"] == all_contacts["net_profit"]


def test_latest_context_deduplicates_before_segmenting():
    from retention_radar.uplift.dunnhumby import latest_household_scores

    observations = pd.DataFrame(
        {
            "household_key": [1, 1, 2],
            "campaign": [7, 5, 7],
            "start_day": [200, 100, 200],
        }
    )
    scores = latest_household_scores(observations)
    assert scores["household_key"].is_unique
    assert scores.loc[scores["household_key"] == 1, "campaign"].item() == 7


def test_every_fold_predicts_only_unseen_households(monkeypatch):
    """Regression test for evaluating only the first group fold or sharing households."""
    import retention_radar.uplift.dunnhumby as module

    class HouseholdCheckingLearner:
        def __init__(self, learner):
            pass

        def fit(self, X, treatment, y, **kwargs):
            self.train_households = set(X[:, 0])

        def predict(self, X, **kwargs):
            assert not self.train_households.intersection(X[:, 0])
            return X[:, 0] * 0.01

    for name in ("BaseSLearner", "BaseTLearner", "BaseXLearner", "BaseDRLearner"):
        monkeypatch.setattr(module, name, HouseholdCheckingLearner)
    households = np.repeat(np.arange(30), 2)
    treatment = np.tile([0, 1], 30)
    data = pd.DataFrame(
        {
            "household_key": households,
            "household_feature": households.astype(float),
            "treated": treatment,
            "delta_spend_per_day": treatment + households * 0.02,
        }
    )
    evaluation, overlap, all_scores = module.evaluate_grouped_learners(
        data,
        ["household_feature"],
        n_splits=3,
    )
    assert len(overlap) == len(data)
    assert set(all_scores["evaluation_fold"]) == {0, 1, 2}
    assert all_scores.groupby("household_key")["evaluation_fold"].nunique().eq(1).all()
    assert evaluation["fold_count"].eq(3).all()


def test_experiment_marks_infeasible_pool():
    design = design_retention_ab_test(active_households=10, variance_spend_day=50.0)
    assert design["feasible_in_available_pool"] is False
    assert design["minimum_detectable_effect_spend_day"] == 0.5
    assert design["randomization_unit"].startswith("Household")


@pytest.mark.requires_warehouse
def test_x5_features_exclude_unverified_post_treatment_dates():
    from retention_radar.uplift.x5 import load_x5_preprocessed

    cfg = load_project_config()
    rows, features = load_x5_preprocessed(
        get_project_root() / cfg.paths.warehouse_db, sample_size=200
    )
    assert len(rows) == 200
    assert rows["client_id"].is_unique
    assert set(features) <= {"age", "age_is_missing", "gender_M", "gender_U"}
    assert not {"has_redeemed", "days_to_redeem", "tenure_days"}.intersection(features)
