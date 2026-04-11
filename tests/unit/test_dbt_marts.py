"""Unit and schema tests for dbt marts, state distribution, and SQL showcase queries."""

import duckdb
import pandas as pd

from retention_radar.config import get_project_root, load_project_config


def test_marts_row_counts_and_keys():
    """Verify mart tables have expected dimensions and unique primary keys."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    con = duckdb.connect(str(db_path), read_only=True)

    hh_count = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT household_key) FROM marts.dim_household;"
    ).fetchone()
    assert hh_count[0] == hh_count[1]
    assert hh_count[0] == 2500

    prod_count = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT product_id) FROM marts.dim_product;"
    ).fetchone()
    assert prod_count[0] == prod_count[1]
    assert prod_count[0] == 92353

    # Check state table unique constraint on household_key + period_id
    state_unique = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT household_key || '-' || period_id)
        FROM marts.fct_household_period_state;
    """).fetchone()[0]
    assert state_unique == 0

    con.close()


def test_state_distribution_acceptance_criteria():
    """Verify state distribution export exists and no state exceeds 70% in training periods."""
    root = get_project_root()
    dist_path = root / "data" / "outputs" / "state_distribution.parquet"
    assert dist_path.exists(), "state_distribution.parquet must be exported"

    df = pd.read_parquet(dist_path)
    train_df = df[df["period_id"].between(4, 19)]
    max_pct = train_df["pct_of_period"].max()
    assert max_pct < 70.0, f"Max state percentage {max_pct}% exceeds 70% threshold"


def test_sql_showcase_results_exist():
    """Verify all 6 showcase query result CSVs exist and contain data."""
    root = get_project_root()
    results_dir = root / "reports" / "sql_results"
    expected_files = [
        "01_period_revenue_change.csv",
        "02_top_decile_revenue_share.csv",
        "03_cohort_retention_matrix.csv",
        "04_basket_size_by_department.csv",
        "05_campaign_reach_by_type.csv",
        "06_quarter_over_quarter_trip_drops.csv",
    ]
    for fn in expected_files:
        p = results_dir / fn
        assert p.exists(), f"Missing showcase query result: {fn}"
        assert p.stat().st_size > 0
