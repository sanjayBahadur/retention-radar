"""Unit tests for PSI drift monitoring."""

import numpy as np
import pandas as pd

from retention_radar.monitoring.drift import calculate_psi, compute_drift_report


def test_calculate_psi_stable():
    """Verify PSI between identical distributions is near 0."""
    rng = np.random.default_rng(42)
    x1 = rng.normal(10, 2, 1000)
    x2 = rng.normal(10, 2, 1000)
    psi = calculate_psi(x1, x2)
    assert psi < 0.10


def test_calculate_psi_drift():
    """Verify PSI flags substantial distribution shifts."""
    rng = np.random.default_rng(42)
    x1 = rng.normal(10, 2, 1000)
    x2 = rng.normal(15, 2, 1000)
    psi = calculate_psi(x1, x2)
    assert psi > 0.20


def test_compute_drift_report():
    """Verify compute_drift_report generates report with status and alert flag."""
    df = pd.DataFrame(
        {
            "household_key": range(200),
            "cutoff_period": [10] * 100 + [23] * 100,
            "split": ["train"] * 100 + ["test"] * 100,
            "recency_days": np.linspace(1, 30, 200),
            "spend_3p": np.linspace(100, 1000, 200),
            "trips_1p": [2] * 200,
            "markov_churn_score": [0.05] * 200,
        }
    )
    report = compute_drift_report(df, features=["recency_days", "spend_3p"])
    assert not report.empty
    assert "feature_name" in report.columns
    assert "psi" in report.columns
    assert "drift_status" in report.columns
    assert "alert_flag" in report.columns
