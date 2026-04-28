"""Unit tests for Phase 7 Customer Lifetime Value estimation and validation."""

import numpy as np
import pandas as pd

from retention_radar.clv.btyd import (
    fit_bgnbd_mle,
    fit_gamma_gamma,
    predict_bgnbd_purchases,
    predict_monetary_value,
    predict_p_alive,
)
from retention_radar.clv.compare import extract_rfm_split
from retention_radar.config import get_project_root, load_project_config


def test_clv_outputs_exist_and_satisfy_criteria():
    """Verify clv.parquet and clv_comparison.parquet exist with Spearman >= 0.5."""
    root = get_project_root()
    out_dir = root / "data" / "outputs"

    comp_path = out_dir / "clv_comparison.parquet"
    assert comp_path.exists()
    comp_df = pd.read_parquet(comp_path)

    assert len(comp_df) == 3
    methods = comp_df["method"].tolist()
    assert "LightGBM (Tweedie)" in methods
    assert "BG/NBD + Gamma-Gamma" in methods
    assert "Markov Reward Process" in methods

    # Acceptance criterion: chosen method has holdout Spearman >= 0.5
    top_spearman = comp_df.iloc[0]["spearman_rank_corr"]
    assert top_spearman >= 0.50, f"Top Spearman {top_spearman} is below 0.50 threshold"

    # Verify clv.parquet
    clv_path = out_dir / "clv.parquet"
    assert clv_path.exists()
    clv_df = pd.read_parquet(clv_path)
    assert len(clv_df) == 2500
    assert "clv_26w" in clv_df.columns
    assert "p_alive" in clv_df.columns
    assert clv_df["clv_26w"].min() >= 0.0
    assert clv_df["p_alive"].between(0.0, 1.0).all()


def test_bgnbd_fitting_and_prediction():
    """Verify BG/NBD MLE optimization and transaction predictions."""
    freq = np.array([5, 10, 1, 0, 15], dtype=float)
    rec = np.array([50, 70, 10, 0, 90], dtype=float)
    T = np.array([100, 100, 100, 100, 100], dtype=float)

    params = fit_bgnbd_mle(freq, rec, T)
    assert len(params) == 4
    assert all(p > 0 for p in params)

    p_alive = predict_p_alive(freq, rec, T, params)
    assert len(p_alive) == 5
    assert (p_alive >= 0.0).all() and (p_alive <= 1.0).all()

    expected_tx = predict_bgnbd_purchases(freq, rec, T, params, future_t=26.0)
    assert len(expected_tx) == 5
    assert (expected_tx >= 0.0).all()


def test_gamma_gamma_fitting_and_prediction():
    """Verify Gamma-Gamma monetary value estimation and Bayesian shrinkage."""
    freq = np.array([5, 12, 3, 20, 8], dtype=float)
    monetary = np.array([35.0, 50.0, 20.0, 80.0, 45.0], dtype=float)

    params = fit_gamma_gamma(freq, monetary)
    assert len(params) == 3
    assert all(p > 0 for p in params)

    pred_m = predict_monetary_value(freq, monetary, params)
    assert len(pred_m) == 5
    assert (pred_m > 0).all()


def test_extract_rfm_split():
    """Verify extract_rfm_split pulls valid calibration and holdout data."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db

    df, corr, p_val = extract_rfm_split(db_path, cal_end_day=532, holdout_end_day=700)
    assert not df.empty
    assert "frequency" in df.columns
    assert "actual_holdout_spend" in df.columns
    assert len(df) > 1000
