"""Unit tests for Phase 4 Markov chain lifecycle models, absorbing math, and simulator."""

import json

import numpy as np
import pandas as pd

from retention_radar.config import get_project_root
from retention_radar.markov.absorbing import compute_absorbing_metrics
from retention_radar.markov.simulate import simulate_retention_scenario


def test_toy_absorbing_chain():
    """Verify exact toy chain absorbing metrics specified in PLAN.md acceptance criteria:

    Q = [[0.80, 0.15], [0.40, 0.30]]
    R = [[0.05], [0.30]]
    yields:
    N = [[8.75, 1.875], [5.0, 2.5]]
    t = [10.625, 7.5]
    B = [[1.0], [1.0]]
    """
    Q = np.array([[0.80, 0.15], [0.40, 0.30]], dtype=float)
    R = np.array([[0.05], [0.30]], dtype=float)

    res = compute_absorbing_metrics(Q, R, period_days=28)

    expected_N = np.array([[8.75, 1.875], [5.0, 2.5]], dtype=float)
    expected_t = np.array([10.625, 7.5], dtype=float)
    expected_B = np.array([[1.0], [1.0]], dtype=float)

    np.testing.assert_allclose(res["N"], expected_N, atol=1e-5)
    np.testing.assert_allclose(res["t_periods"], expected_t, atol=1e-5)
    np.testing.assert_allclose(res["B"], expected_B, atol=1e-5)


def test_simulate_deterministic_and_zero_delta():
    """Verify simulator is deterministic and zero intervention yields exactly zero delta."""
    states = ["Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
    P = np.array(
        [
            [0.70, 0.20, 0.10, 0.00, 0.00],
            [0.15, 0.65, 0.20, 0.00, 0.00],
            [0.05, 0.35, 0.00, 0.60, 0.00],
            [0.02, 0.18, 0.00, 0.40, 0.40],
            [0.00, 0.10, 0.00, 0.00, 0.90],
        ],
        dtype=float,
    )
    rewards = np.array([150.0, 60.0, 0.0, 0.0, 0.0], dtype=float)
    init_dist = np.array([0.20, 0.50, 0.15, 0.10, 0.05], dtype=float)

    # 1. Zero intervention
    res_zero = simulate_retention_scenario(
        P=P,
        state_names=states,
        state_rewards=rewards,
        initial_distribution=init_dist,
        intervention=None,
        horizon_periods=13,
        population=100_000,
    )
    assert res_zero["delta_margin_dollars"] == 0.0
    assert res_zero["delta_lapsed_households"] == 0.0

    # 2. Determinism check
    intervention = {"At-Risk": {"Regular": 0.05}}
    run_1 = simulate_retention_scenario(
        P=P,
        state_names=states,
        state_rewards=rewards,
        initial_distribution=init_dist,
        intervention=intervention,
        horizon_periods=13,
        population=100_000,
    )
    run_2 = simulate_retention_scenario(
        P=P,
        state_names=states,
        state_rewards=rewards,
        initial_distribution=init_dist,
        intervention=intervention,
        horizon_periods=13,
        population=100_000,
    )
    assert run_1["delta_margin_dollars"] == run_2["delta_margin_dollars"]
    assert run_1["delta_lapsed_households"] == run_2["delta_lapsed_households"]
    assert run_1["delta_margin_dollars"] > 0.0
    assert run_1["delta_lapsed_households"] < 0.0  # fewer households lapsed


def test_markov_outputs_exist_and_valid():
    """Verify Phase 4 output files exist and satisfy schema properties."""
    root = get_project_root()
    out_dir = root / "data" / "outputs"

    # 1. markov_P.parquet
    p_path = out_dir / "markov_P.parquet"
    assert p_path.exists()
    df_p = pd.read_parquet(p_path)
    assert len(df_p) == 36  # 6x6 matrix
    assert "ci_lower_95" in df_p.columns
    assert "ci_upper_95" in df_p.columns

    # 2. markov_metrics.json
    m_path = out_dir / "markov_metrics.json"
    assert m_path.exists()
    with open(m_path) as f:
        metrics = json.load(f)
    assert "expected_days_to_lapse" in metrics
    assert "markov_reward_clv" in metrics
    assert "assumption_tests" in metrics

    # 3. scenarios.parquet
    s_path = out_dir / "scenarios.parquet"
    assert s_path.exists()
    df_s = pd.read_parquet(s_path)
    assert len(df_s) > 0

    # 4. survival_curves.parquet
    surv_path = out_dir / "survival_curves.parquet"
    assert surv_path.exists()
    df_surv = pd.read_parquet(surv_path)
    assert "survival_probability" in df_surv.columns


def test_markov_estimation_and_stationary():
    """Verify empirical transition matrix estimation and stationary distribution calculation."""
    from retention_radar.markov.chain import (
        compute_stationary_distribution,
        estimate_transition_matrix,
    )

    states = ["Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
    df = pd.DataFrame(
        {
            "household_key": [1, 1, 2, 2],
            "state": ["Champion", "Regular", "Regular", "At-Risk"],
            "next_state": ["Regular", "Regular", "At-Risk", "Dormant"],
        }
    )
    P, C = estimate_transition_matrix(df, states, smoothing_alpha=0.5)
    assert P.shape == (5, 5)
    assert C.shape == (5, 5)
    # Check row sums of P
    row_sums = P.sum(axis=1)
    for s in row_sums:
        if s > 0:
            assert abs(s - 1.0) < 1e-4

    pi = compute_stationary_distribution(P)
    assert len(pi) == 5
    assert abs(sum(pi) - 1.0) < 1e-4


def test_markov_assumption_tests():
    """Verify 1st vs 2nd order and time homogeneity tests."""
    from retention_radar.markov.assumptions import test_first_vs_second_order, test_time_homogeneity

    states = ["Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
    df = pd.DataFrame(
        {
            "household_key": [1, 1, 1, 2, 2, 2],
            "period_id": [4, 5, 6, 4, 5, 6],
            "state": ["Champion", "Regular", "At-Risk", "Regular", "Regular", "Champion"],
            "prior_state": ["Champion", "Champion", "Regular", "Regular", "Regular", "Regular"],
        }
    )

    res_order = test_first_vs_second_order(df, states)
    assert "lr_statistic" in res_order
    assert "log_likelihood_1st_order" in res_order

    res_homo = test_time_homogeneity(df, states, split_period=5)
    assert "chi2_statistic" in res_homo
