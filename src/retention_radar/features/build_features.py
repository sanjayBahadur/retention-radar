"""Point-in-time feature engineering for rolling-origin churn prediction."""

from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from retention_radar.config import get_project_root, load_project_config


def extract_features_at_cutoff(
    con: duckdb.DuckDBPyConnection,
    cutoff_period: int,
    markov_P: Optional[np.ndarray] = None,
    states: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract strictly leakage-free features using only data observed up to cutoff_period.

    Target y = 1 if zero purchases in periods cutoff+1 and cutoff+2 (for non-Lapsed households).
    """
    query = f"""
        WITH eligible AS (
            SELECT
                household_key,
                state,
                consecutive_missed_periods,
                first_period_id,
                trailing_3period_spend
            FROM marts.fct_household_period_state
            WHERE period_id = {cutoff_period}
              AND state != 'Lapsed'
        ),

        -- Historical activity up to cutoff
        past_activity AS (
            SELECT
                household_key,
                period_id,
                trips,
                spend,
                items,
                coupon_used
            FROM marts.fct_household_period_state
            WHERE period_id <= {cutoff_period}
        ),

        -- Multi-period rolling metrics (1p, 3p, 6p)
        rolling_metrics AS (
            SELECT
                household_key,
                SUM(CASE WHEN period_id = {cutoff_period} THEN trips ELSE 0 END) AS trips_1p,
                SUM(CASE WHEN period_id = {cutoff_period} THEN spend ELSE 0.0 END) AS spend_1p,
                SUM(CASE WHEN period_id >= {cutoff_period} - 2 THEN trips ELSE 0 END) AS trips_3p,
                SUM(CASE WHEN period_id >= {cutoff_period} - 2 THEN spend ELSE 0.0 END) AS spend_3p,
                SUM(CASE WHEN period_id >= {cutoff_period} - 5 THEN trips ELSE 0 END) AS trips_6p,
                SUM(CASE WHEN period_id >= {cutoff_period} - 5 THEN spend ELSE 0.0 END) AS spend_6p
            FROM past_activity
            WHERE period_id >= {cutoff_period} - 5
            GROUP BY household_key
        ),

        -- Basket level metrics over last 6 periods
        basket_metrics AS (
            SELECT
                b.household_key,
                COUNT(b.basket_id) AS baskets_6p,
                ROUND(AVG(b.spend), 2) AS avg_basket_spend_6p,
                ROUND(AVG(b.items), 2) AS avg_basket_items_6p,
                ROUND(AVG(b.distinct_departments), 2) AS avg_dept_per_basket_6p,
                ROUND(SUM(CASE WHEN b.day % 7 IN (0, 6) THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(b.basket_id), 0), 4) AS weekend_trip_share_6p,
                COUNT(DISTINCT b.store_id) AS distinct_stores_6p,
                ROUND(SUM(b.total_retail_disc + b.total_coupon_disc) / NULLIF(SUM(b.gross_spend), 0), 4) AS discount_share_6p,
                SUM(b.coupon_used) AS coupon_trips_6p
            FROM marts.fct_baskets b
            WHERE b.period_id BETWEEN {cutoff_period} - 5 AND {cutoff_period}
            GROUP BY b.household_key
        ),

        -- Inter-trip spacing over last 6 periods
        inter_trips AS (
            SELECT
                household_key,
                ROUND(AVG(gap_days), 2) AS inter_trip_mean_days,
                ROUND(COALESCE(STDDEV_SAMP(gap_days), 0.0), 2) AS inter_trip_std_days
            FROM (
                SELECT
                    household_key,
                    day - LAG(day) OVER (PARTITION BY household_key ORDER BY day) AS gap_days
                FROM marts.fct_baskets
                WHERE period_id BETWEEN {cutoff_period} - 5 AND {cutoff_period}
            ) sub
            WHERE gap_days IS NOT NULL
            GROUP BY household_key
        ),

        -- Product level features over last 6 periods: private label and entropy
        tx_stats AS (
            WITH dept_spend AS (
                SELECT
                    t.household_key,
                    p.department,
                    SUM(t.sales_value) AS dept_val,
                    SUM(CASE WHEN p.is_private_label = 1 THEN t.sales_value ELSE 0 END) AS priv_val
                FROM staging.stg_transactions t
                JOIN staging.stg_products p ON t.product_id = p.product_id
                WHERE t.period_id BETWEEN {cutoff_period} - 5 AND {cutoff_period}
                GROUP BY t.household_key, p.department
            ),
            hh_totals AS (
                SELECT
                    household_key,
                    SUM(dept_val) AS total_val,
                    SUM(priv_val) AS total_priv_val
                FROM dept_spend
                GROUP BY household_key
            )
            SELECT
                d.household_key,
                ROUND(h.total_priv_val / NULLIF(h.total_val, 0), 4) AS private_label_share_6p,
                -ROUND(SUM((d.dept_val / h.total_val) * LN(NULLIF(d.dept_val / h.total_val, 0))), 4) AS category_entropy_6p
            FROM dept_spend d
            JOIN hh_totals h ON d.household_key = h.household_key
            WHERE d.dept_val > 0 AND h.total_val > 0
            GROUP BY d.household_key, h.total_val, h.total_priv_val
        ),

        -- Recency in days up to cutoff
        recency_calc AS (
            SELECT
                household_key,
                ({cutoff_period} * 28 - MAX(day)) AS recency_days
            FROM marts.fct_baskets
            WHERE period_id <= {cutoff_period}
            GROUP BY household_key
        ),

        -- Campaign exposure over last 3 periods
        campaign_metrics AS (
            SELECT
                household_key,
                COUNT(DISTINCT campaign) AS campaigns_last_3p
            FROM marts.fct_campaign_exposure
            WHERE start_period BETWEEN {cutoff_period} - 2 AND {cutoff_period}
            GROUP BY household_key
        ),

        -- Future activity to derive target label y
        future_trips AS (
            SELECT
                household_key,
                COALESCE(SUM(trips), 0) AS next_2p_trips
            FROM marts.fct_household_period_state
            WHERE period_id IN ({cutoff_period} + 1, {cutoff_period} + 2)
            GROUP BY household_key
        )

        SELECT
            e.household_key,
            {cutoff_period} AS cutoff_period,
            e.state AS current_state,
            e.consecutive_missed_periods,
            ({cutoff_period} - e.first_period_id + 1) AS tenure_periods,
            COALESCE(rc.recency_days, ({cutoff_period} - e.first_period_id + 1) * 28) AS recency_days,
            COALESCE(r.trips_1p, 0) AS trips_1p,
            COALESCE(r.spend_1p, 0.0) AS spend_1p,
            COALESCE(r.trips_3p, 0) AS trips_3p,
            COALESCE(r.spend_3p, 0.0) AS spend_3p,
            COALESCE(r.trips_6p, 0) AS trips_6p,
            COALESCE(r.spend_6p, 0.0) AS spend_6p,
            COALESCE(b.baskets_6p, 0) AS baskets_6p,
            COALESCE(b.avg_basket_spend_6p, 0.0) AS avg_basket_spend_6p,
            COALESCE(b.avg_basket_items_6p, 0.0) AS avg_basket_items_6p,
            COALESCE(b.avg_dept_per_basket_6p, 0.0) AS avg_dept_per_basket_6p,
            COALESCE(b.weekend_trip_share_6p, 0.0) AS weekend_trip_share_6p,
            COALESCE(b.distinct_stores_6p, 0) AS distinct_stores_6p,
            COALESCE(b.discount_share_6p, 0.0) AS discount_share_6p,
            COALESCE(b.coupon_trips_6p, 0) AS coupon_trips_6p,
            COALESCE(it.inter_trip_mean_days, 28.0) AS inter_trip_mean_days,
            COALESCE(it.inter_trip_std_days, 0.0) AS inter_trip_std_days,
            COALESCE(ts.private_label_share_6p, 0.0) AS private_label_share_6p,
            COALESCE(ts.category_entropy_6p, 0.0) AS category_entropy_6p,
            COALESCE(c.campaigns_last_3p, 0) AS campaigns_last_3p,
            d.has_demographics,
            COALESCE(d.age_desc, 'Unknown') AS age_desc,
            COALESCE(d.income_desc, 'Unknown') AS income_desc,
            COALESCE(d.household_size_desc, 'Unknown') AS household_size_desc,
            CASE WHEN COALESCE(f.next_2p_trips, 0) = 0 THEN 1 ELSE 0 END AS churn_label
        FROM eligible e
        LEFT JOIN rolling_metrics r ON e.household_key = r.household_key
        LEFT JOIN basket_metrics b ON e.household_key = b.household_key
        LEFT JOIN inter_trips it ON e.household_key = it.household_key
        LEFT JOIN tx_stats ts ON e.household_key = ts.household_key
        LEFT JOIN recency_calc rc ON e.household_key = rc.household_key
        LEFT JOIN campaign_metrics c ON e.household_key = c.household_key
        LEFT JOIN marts.dim_household d ON e.household_key = d.household_key
        LEFT JOIN future_trips f ON e.household_key = f.household_key
        ORDER BY e.household_key;
    """

    df = con.execute(query).df()

    # Inter-trip coefficient of variation (CV)
    df["inter_trip_cv"] = (df["inter_trip_std_days"] / (df["inter_trip_mean_days"] + 1e-5)).round(4)

    # Compute OLS spend slope over 6 periods per household
    spend_trends = con.execute(f"""
        SELECT household_key, period_id, spend
        FROM marts.fct_household_period_state
        WHERE period_id BETWEEN {cutoff_period} - 5 AND {cutoff_period};
    """).df()

    slopes: Dict[int, float] = {}
    x = np.arange(6)
    x_mean = 2.5
    x_var = np.sum((x - x_mean) ** 2)

    for hh, grp in spend_trends.groupby("household_key"):
        p_map = dict(zip(grp["period_id"], grp["spend"]))
        y = np.array([p_map.get(p, 0.0) for p in range(cutoff_period - 5, cutoff_period + 1)])
        slope = np.sum((x - x_mean) * (y - np.mean(y))) / x_var
        slopes[hh] = round(float(slope), 2)

    df["ols_spend_slope_6p"] = df["household_key"].map(slopes).fillna(0.0)

    # Markov churn score: P(state at t+2 in {Dormant, Lapsed}) from P^2
    default_state_churn = {
        "Champion": 0.0050,
        "Regular": 0.0612,
        "New": 0.0945,
        "At-Risk": 0.2800,
        "Dormant": 0.4234,
        "Lapsed": 0.6352,
    }
    if markov_P is not None and states is not None:
        P2 = markov_P @ markov_P
        dorm_idx = states.index("Dormant") if "Dormant" in states else -1
        lapse_idx = states.index("Lapsed") if "Lapsed" in states else -1
        score_map = {}
        for s_idx, s in enumerate(states):
            score = 0.0
            if dorm_idx >= 0:
                score += P2[s_idx, dorm_idx]
            if lapse_idx >= 0:
                score += P2[s_idx, lapse_idx]
            score_map[s] = round(float(score), 4)
        df["markov_churn_score"] = df["current_state"].map(score_map).fillna(0.075)
    else:
        df["markov_churn_score"] = df["current_state"].map(default_state_churn).fillna(0.075)

    return df


def build_rolling_origin_dataset(
    db_path: Path,
    train_cutoffs: List[int] = list(range(6, 14)),
    val_cutoffs: List[int] = [16, 17],
    test_cutoffs: List[int] = [20, 21, 22, 23],
    markov_P: Optional[np.ndarray] = None,
    states: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build rolling-origin datasets for train, validation, and test splits."""
    con = duckdb.connect(str(db_path), read_only=True)

    def extract_split(cutoffs: List[int]) -> pd.DataFrame:
        frames = [
            extract_features_at_cutoff(con, cutoff, markov_P=markov_P, states=states)
            for cutoff in cutoffs
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    train_df = extract_split(train_cutoffs)
    val_df = extract_split(val_cutoffs)
    test_df = extract_split(test_cutoffs)

    con.close()
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def main() -> None:
    """Build and save features dataset."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    # A full-history transition matrix would leak future information into early cutoffs.
    # Use the fixed state heuristic; fit/evaluate statistical models on separate splits.
    splits = build_rolling_origin_dataset(db_path=db_path)

    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.concat(
        [
            splits["train"].assign(split="train"),
            splits["val"].assign(split="val"),
            splits["test"].assign(split="test"),
        ],
        ignore_index=True,
    )

    parquet_path = out_dir / "churn_features.parquet"
    full_df.to_parquet(parquet_path, index=False)
    print(f"Saved churn features to {parquet_path} ({len(full_df)} rows).")


if __name__ == "__main__":
    main()
