"""Early-warning category departure signals identifying precursor category abandonment before lapse."""

import duckdb
import pandas as pd

from retention_radar.config import get_project_root, load_project_config

EXCLUDED_DEPARTMENTS = [
    "MISC. TRANS.",
    "MISC SALES TRAN",
    "COUP/STR & MFG",
    "KIOSK-GAS",
    "CNTRL/STORE SUP",
    "CHARITABLE CONT",
    "PHOTO",
    "PROD-WHS SALES",
    "PHARMACY SUPPLY",
    "",
]


def compute_early_warning_categories(
    con: duckdb.DuckDBPyConnection,
    min_early_spend: float = 500.0,
) -> pd.DataFrame:
    """Identify commodities with the largest drop in spend share prior to dormancy vs non-lapsing controls."""
    excl_list = ", ".join(f"'{d}'" for d in EXCLUDED_DEPARTMENTS)

    query = f"""
    WITH dormant_households AS (
        SELECT household_key, MIN(period_id) AS dormant_period
        FROM marts.fct_household_period_state
        WHERE state = 'Dormant' AND period_id BETWEEN 8 AND 20
        GROUP BY household_key
    ),
    active_controls AS (
        SELECT household_key, 14 AS pseudo_dormant_period
        FROM marts.fct_household_period_state
        WHERE period_id BETWEEN 8 AND 20 AND state IN ('Champion', 'Regular')
        GROUP BY household_key
        HAVING COUNT(DISTINCT period_id) >= 12
    ),
    -- Lapsing commodity spend
    lapsing_spend AS (
        SELECT
            d.household_key,
            p.commodity_desc,
            SUM(CASE WHEN t.period_id BETWEEN d.dormant_period - 5 AND d.dormant_period - 3 THEN t.sales_value ELSE 0 END) AS early_spend,
            SUM(CASE WHEN t.period_id BETWEEN d.dormant_period - 2 AND d.dormant_period THEN t.sales_value ELSE 0 END) AS late_spend
        FROM dormant_households d
        JOIN staging.stg_transactions t ON d.household_key = t.household_key
        JOIN staging.stg_products p ON t.product_id = p.product_id
        WHERE t.period_id BETWEEN d.dormant_period - 5 AND d.dormant_period
          AND p.department NOT IN ({excl_list})
          AND p.commodity_desc IS NOT NULL
        GROUP BY d.household_key, p.commodity_desc
    ),
    lapsing_totals AS (
        SELECT
            household_key,
            SUM(early_spend) AS total_early,
            SUM(late_spend) AS total_late
        FROM lapsing_spend
        GROUP BY household_key
    ),
    lapsing_shares AS (
        SELECT
            s.commodity_desc,
            COUNT(DISTINCT s.household_key) AS lapsing_hh_count,
            SUM(s.early_spend) AS early_spend_sum,
            SUM(s.late_spend) AS late_spend_sum,
            AVG((s.late_spend / NULLIF(t.total_late, 0)) - (s.early_spend / NULLIF(t.total_early, 0))) AS lapsing_mean_share_change
        FROM lapsing_spend s
        JOIN lapsing_totals t ON s.household_key = t.household_key
        WHERE t.total_early > 0 AND t.total_late > 0
        GROUP BY s.commodity_desc
        HAVING SUM(s.early_spend) >= {min_early_spend}
    ),
    -- Control commodity spend
    control_spend AS (
        SELECT
            c.household_key,
            p.commodity_desc,
            SUM(CASE WHEN t.period_id BETWEEN c.pseudo_dormant_period - 5 AND c.pseudo_dormant_period - 3 THEN t.sales_value ELSE 0 END) AS early_spend,
            SUM(CASE WHEN t.period_id BETWEEN c.pseudo_dormant_period - 2 AND c.pseudo_dormant_period THEN t.sales_value ELSE 0 END) AS late_spend
        FROM active_controls c
        JOIN staging.stg_transactions t ON c.household_key = t.household_key
        JOIN staging.stg_products p ON t.product_id = p.product_id
        WHERE t.period_id BETWEEN c.pseudo_dormant_period - 5 AND c.pseudo_dormant_period
          AND p.department NOT IN ({excl_list})
          AND p.commodity_desc IS NOT NULL
        GROUP BY c.household_key, p.commodity_desc
    ),
    control_totals AS (
        SELECT
            household_key,
            SUM(early_spend) AS total_early,
            SUM(late_spend) AS total_late
        FROM control_spend
        GROUP BY household_key
    ),
    control_shares AS (
        SELECT
            s.commodity_desc,
            AVG((s.late_spend / NULLIF(t.total_late, 0)) - (s.early_spend / NULLIF(t.total_early, 0))) AS control_mean_share_change
        FROM control_spend s
        JOIN control_totals t ON s.household_key = t.household_key
        WHERE t.total_early > 0 AND t.total_late > 0
        GROUP BY s.commodity_desc
    )
    SELECT
        l.commodity_desc,
        l.lapsing_hh_count,
        ROUND(l.early_spend_sum, 2) AS lapsing_early_spend,
        ROUND(l.late_spend_sum, 2) AS lapsing_late_spend,
        ROUND(l.lapsing_mean_share_change, 5) AS lapsing_share_change,
        ROUND(COALESCE(c.control_mean_share_change, 0.0), 5) AS control_share_change,
        ROUND(l.lapsing_mean_share_change - COALESCE(c.control_mean_share_change, 0.0), 5) AS net_drop_effect
    FROM lapsing_shares l
    LEFT JOIN control_shares c ON l.commodity_desc = c.commodity_desc
    ORDER BY net_drop_effect ASC;
    """
    df = con.execute(query).df()
    df["rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    """Run early-warning category analysis and save outputs."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)
    print("Computing early-warning category signals...")
    df = compute_early_warning_categories(con, min_early_spend=500.0)
    con.close()

    print("\nTop 15 Early-Warning Categories (Largest Pre-Lapse Share Drops):")
    print(df.head(15).to_string())

    out_path = out_dir / "early_warning_categories.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved early-warning categories to {out_path} ({len(df)} categories analyzed).")


if __name__ == "__main__":
    main()
