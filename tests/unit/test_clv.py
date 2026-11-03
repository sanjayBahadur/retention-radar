"""Unit tests for Phase 7 Customer Lifetime Value estimation and validation."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from retention_radar.clv.btyd import (
    fit_bgnbd_mle,
    fit_gamma_gamma,
    predict_bgnbd_purchases,
    predict_monetary_value,
    predict_p_alive,
)
from retention_radar.clv.compare import FEATURES, extract_rfm_split, forecast_metrics
from retention_radar.config import get_project_root, load_project_config


def test_clv_outputs_exist_and_satisfy_criteria():
    """Verify forecast artifacts, selection, and temporal boundaries agree."""
    root = get_project_root()
    out_dir = root / "data" / "outputs"

    comp_path = out_dir / "clv_comparison.parquet"
    assert comp_path.exists()
    comp_df = pd.read_parquet(comp_path)

    assert len(comp_df) == 3
    methods = comp_df["method"].tolist()
    assert "LightGBM (Tweedie)" in methods
    assert "BG/NBD + Gamma-Gamma" in methods
    assert "Recent 26-week spend" in methods

    # The selected method minimizes dollar error; there is no arbitrary score gate.
    assert comp_df["mae_dollars"].is_monotonic_increasing
    summary = json.loads((out_dir / "clv_summary.json").read_text())
    assert summary["chosen_method"] == comp_df.iloc[0]["method"]
    assert summary["train_outcome_end_day"] <= summary["test_feature_cutoff_day"]
    assert summary["test_outcome_end_day"] - summary["test_feature_cutoff_day"] == 182

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


@pytest.mark.requires_warehouse
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


def test_rfm_aggregates_shopping_days_and_keeps_future_out_of_features(tmp_path):
    """Item lines become visit totals; future labels cannot change features."""
    path = tmp_path / "visits.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA raw")
        con.execute(
            "CREATE TABLE raw.transaction_data (household_key INT, day INT, basket_id INT, sales_value DOUBLE)"
        )
        con.executemany(
            "INSERT INTO raw.transaction_data VALUES (?, ?, ?, ?)",
            [
                (1, 200, 1, 10),
                (1, 200, 1, 20),
                (1, 201, 2, 40),
                (1, 201, 3, 50),
                (1, 337, 4, 500),
                (1, 518, 5, 600),
                (1, 519, 6, 700),
            ],
        )
    before, _, _ = extract_rfm_split(path, 336, 518)
    assert before.iloc[0]["frequency"] == 1
    assert before.iloc[0]["monetary_value"] == 90  # repeat day, not average item
    assert before.iloc[0]["cal_spend"] == 120
    assert before.iloc[0]["actual_holdout_spend"] == 1100
    with duckdb.connect(str(path)) as con:
        con.execute("UPDATE raw.transaction_data SET sales_value = 1000 WHERE day > 336")
    after, _, _ = extract_rfm_split(path, 336, 518)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert after.iloc[0]["actual_holdout_spend"] == 2000
    later, _, _ = extract_rfm_split(path, 518, 700)
    assert later.iloc[0]["actual_holdout_spend"] == 1000  # excludes day 518


def test_forecast_errors_are_sales_dollars():
    metrics = forecast_metrics(np.array([100.0, 200.0]), np.array([80.0, 160.0]))
    assert metrics["mae_dollars"] == 30
    assert metrics["rmse_dollars"] == 31.62


def test_bgnbd_forecast_respects_horizon_and_inactivity():
    """Conditional forecasts allow dropout rather than scaling visits linearly."""
    params = (1.0, 10.0, 2.0, 3.0)
    freq = np.array([4.0, 4.0])
    last = np.array([3.0, 20.0])
    age = np.array([26.0, 26.0])
    assert (predict_bgnbd_purchases(freq, last, age, params, 0) == 0).all()
    short = predict_bgnbd_purchases(freq, last, age, params, 13)
    long = predict_bgnbd_purchases(freq, last, age, params, 26)
    assert (long >= short).all()
    assert (long < 2 * short).all()
    assert long[0] < long[1]
