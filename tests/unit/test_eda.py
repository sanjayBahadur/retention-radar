"""Unit tests for Exploratory Data Analysis metrics and outputs."""

import duckdb

from retention_radar.config import get_project_root, load_project_config
from retention_radar.eda.analysis import (
    compute_churn_base_rate,
    compute_concentration,
    compute_demographic_coverage,
    compute_headline_kpis,
)


def test_headline_kpis():
    """Verify headline KPIs match expected warehouse metrics."""
    root = get_project_root()
    cfg = load_project_config()
    con = duckdb.connect(str(root / cfg.paths.warehouse_db), read_only=True)
    kpis = compute_headline_kpis(con)
    con.close()

    assert kpis["total_households"] == 2500
    assert kpis["total_baskets"] == 272110
    assert kpis["total_revenue"] > 7900000.0
    assert kpis["avg_basket_spend"] > 25.0
    assert kpis["trips_per_hh_per_period"] > 4.0


def test_concentration_metrics():
    """Verify Lorenz concentration and Gini coefficient."""
    root = get_project_root()
    cfg = load_project_config()
    con = duckdb.connect(str(root / cfg.paths.warehouse_db), read_only=True)
    conc = compute_concentration(con)
    con.close()

    assert 50.0 < conc["top_20pct_revenue_share"] < 60.0
    assert 0.45 < conc["gini_coefficient"] < 0.55


def test_churn_base_rate():
    """Verify binary churn rate across training cutoffs."""
    root = get_project_root()
    cfg = load_project_config()
    con = duckdb.connect(str(root / cfg.paths.warehouse_db), read_only=True)
    churn = compute_churn_base_rate(con)
    con.close()

    assert churn["total_active_instances"] > 10000
    assert 3.0 < churn["churn_base_rate_pct"] < 6.0


def test_demographic_coverage():
    """Verify demographic coverage calculation."""
    root = get_project_root()
    cfg = load_project_config()
    con = duckdb.connect(str(root / cfg.paths.warehouse_db), read_only=True)
    demo = compute_demographic_coverage(con)
    con.close()

    assert demo["total_households"] == 2500
    assert demo["demographic_households"] == 801
    assert abs(demo["coverage_pct"] - 32.04) < 0.1


def test_eda_figures_exist():
    """Verify all 5 generated EDA figures exist in reports/figures/."""
    root = get_project_root()
    fig_dir = root / "reports" / "figures"
    expected_figures = [
        "eda_cohort_retention.png",
        "eda_lorenz_curve.png",
        "eda_purchase_rhythm.png",
        "eda_state_transitions.png",
        "eda_weekly_seasonality.png",
    ]
    for fn in expected_figures:
        p = fig_dir / fn
        assert p.exists(), f"Missing figure: {fn}"
        assert p.stat().st_size > 1000
