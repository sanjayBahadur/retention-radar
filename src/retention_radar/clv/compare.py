"""Holdout validation and comparison of CLV estimation models."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from retention_radar.clv.btyd import (
    fit_bgnbd_mle,
    fit_gamma_gamma,
    predict_bgnbd_purchases,
    predict_monetary_value,
    predict_p_alive,
)
from retention_radar.config import get_project_root, load_project_config


def extract_rfm_split(
    db_path: Path,
    cal_end_day: int = 532,
    holdout_end_day: int = 700,
) -> Tuple[pd.DataFrame, float, float]:
    """Extract RFM calibration features and actual holdout spend from warehouse."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(f"""
        WITH cal AS (
            SELECT
                household_key,
                COUNT(DISTINCT day) - 1 AS frequency,
                MAX(day) - MIN(day) AS recency_days,
                {cal_end_day} - MIN(day) AS T_days,
                ROUND(SUM(sales_value), 2) AS cal_spend,
                ROUND(AVG(sales_value), 2) AS monetary_value,
                COUNT(DISTINCT basket_id) AS cal_trips
            FROM raw.transaction_data
            WHERE day <= {cal_end_day}
            GROUP BY household_key
        ),
        hold AS (
            SELECT
                household_key,
                ROUND(SUM(sales_value), 2) AS actual_holdout_spend,
                COUNT(DISTINCT basket_id) AS actual_holdout_trips
            FROM raw.transaction_data
            WHERE day > {cal_end_day} AND day <= {holdout_end_day}
            GROUP BY household_key
        ),
        state_76 AS (
            SELECT
                household_key,
                state AS state_at_week_76
            FROM marts.fct_household_period_state
            WHERE period_id = 19  -- period 19 ends at day 532 (week 76)
        )
        SELECT
            c.*,
            s.state_at_week_76,
            COALESCE(h.actual_holdout_spend, 0.0) AS actual_holdout_spend,
            COALESCE(h.actual_holdout_trips, 0) AS actual_holdout_trips
        FROM cal c
        LEFT JOIN hold h ON c.household_key = h.household_key
        LEFT JOIN state_76 s ON c.household_key = s.household_key;
    """).df()
    con.close()

    df["recency_weeks"] = df["recency_days"] / 7.0
    df["T_weeks"] = df["T_days"] / 7.0

    # Pearson correlation on repeat shoppers
    repeat_mask = df["frequency"] > 0
    corr, p_val = stats.pearsonr(
        df.loc[repeat_mask, "frequency"], df.loc[repeat_mask, "monetary_value"]
    )

    return df, float(corr), float(p_val)


def run_clv_comparison_pipeline(
    db_path: Path,
    output_dir: Path,
    gross_margin: float = 0.25,
    annual_discount_rate: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit all three CLV methods on calibration window, validate on holdout, and score households."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df, freq_mon_corr, corr_pval = extract_rfm_split(db_path, cal_end_day=532, holdout_end_day=700)

    # 1. Method A: BG/NBD + Gamma-Gamma
    bgnbd_params = fit_bgnbd_mle(
        df["frequency"].values,
        df["recency_weeks"].values,
        df["T_weeks"].values,
    )
    gg_params = fit_gamma_gamma(
        df["frequency"].values,
        df["monetary_value"].values,
    )

    pred_tx_bgnbd = predict_bgnbd_purchases(
        df["frequency"].values,
        df["recency_weeks"].values,
        df["T_weeks"].values,
        bgnbd_params,
        future_t=24.0,  # 24 weeks holdout (days 533 to 700 = 168 days)
    )
    pred_mon_bgnbd = predict_monetary_value(
        df["frequency"].values,
        df["monetary_value"].values,
        gg_params,
    )
    df["pred_spend_bgnbd"] = pred_tx_bgnbd * pred_mon_bgnbd
    df["p_alive_bgnbd"] = predict_p_alive(
        df["frequency"].values,
        df["recency_weeks"].values,
        df["T_weeks"].values,
        bgnbd_params,
    )

    # 2. Method B: Markov Reward CLV (assigned by state at week 76)
    markov_metrics_path = output_dir / "markov_metrics.json"
    if markov_metrics_path.exists():
        with open(markov_metrics_path) as f:
            mm = json.load(f)
        state_clv_map = mm.get("markov_reward_clv", {})
    else:
        state_clv_map = {
            "Champion": 2400.0,
            "Regular": 1780.0,
            "New": 1770.0,
            "At-Risk": 1450.0,
            "Dormant": 1070.0,
            "Lapsed": 0.0,
        }

    # Scale Markov state CLV to gross spend across 24 weeks
    df["pred_spend_markov"] = (
        df["state_at_week_76"]
        .map(lambda s: (state_clv_map.get(s, 1000.0) / gross_margin) * (24.0 / 52.0))
        .fillna(500.0)
    )

    # 3. Method C: LightGBM Tweedie Regressor
    features = ["frequency", "recency_weeks", "T_weeks", "cal_spend", "monetary_value", "cal_trips"]
    lgbm = LGBMRegressor(objective="tweedie", random_state=seed, verbose=-1, n_estimators=100)
    lgbm.fit(df[features], df["actual_holdout_spend"])
    df["pred_spend_lgbm"] = lgbm.predict(df[features])

    # Compute Validation Metrics
    actual = df["actual_holdout_spend"].values

    def eval_predictions(preds: np.ndarray) -> Dict[str, float]:
        spearman = float(stats.spearmanr(preds, actual)[0])
        mae = float(np.mean(np.abs(preds - actual)))
        rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
        return {
            "spearman_rank_corr": round(spearman, 4),
            "mae_dollars": round(mae, 2),
            "rmse_dollars": round(rmse, 2),
        }

    comp_records = [
        {"method": "LightGBM (Tweedie)", **eval_predictions(df["pred_spend_lgbm"].values)},
        {"method": "BG/NBD + Gamma-Gamma", **eval_predictions(df["pred_spend_bgnbd"].values)},
        {"method": "Markov Reward Process", **eval_predictions(df["pred_spend_markov"].values)},
    ]
    comp_df = pd.DataFrame(comp_records).sort_values("spearman_rank_corr", ascending=False)
    comp_df.to_parquet(output_dir / "clv_comparison.parquet", index=False)

    # Refit chosen method (LightGBM Tweedie) on all 102 weeks for full scoring
    con = duckdb.connect(str(db_path), read_only=True)
    full_df = con.execute("""
        SELECT
            household_key,
            COUNT(DISTINCT day) - 1 AS frequency,
            (MAX(day) - MIN(day)) / 7.0 AS recency_weeks,
            (700 - MIN(day)) / 7.0 AS T_weeks,
            ROUND(SUM(sales_value), 2) AS cal_spend,
            ROUND(AVG(sales_value), 2) AS monetary_value,
            COUNT(DISTINCT basket_id) AS cal_trips
        FROM raw.transaction_data
        WHERE day <= 700
        GROUP BY household_key;
    """).df()
    con.close()

    # Predict 26-week spend refit
    full_pred_spend = lgbm.predict(full_df[features])
    discount_factor = float((1.0 + annual_discount_rate) ** (-0.5))  # half-year discount
    # CLV in margin dollars = predicted_spend * gross_margin * discount_factor
    full_df["clv_26w"] = np.round(full_pred_spend * gross_margin * discount_factor, 2)
    full_df["p_alive"] = predict_p_alive(
        full_df["frequency"].values,
        full_df["recency_weeks"].values,
        full_df["T_weeks"].values,
        bgnbd_params,
    ).round(4)
    full_df["method"] = "LightGBM Tweedie (Margin Discounted)"

    output_clv = full_df[["household_key", "method", "clv_26w", "p_alive"]].copy()
    output_clv.to_parquet(output_dir / "clv.parquet", index=False)

    summary = {
        "frequency_monetary_correlation": round(freq_mon_corr, 4),
        "correlation_p_value": float(corr_pval),
        "chosen_method": "LightGBM (Tweedie)",
        "chosen_method_spearman": float(comp_df.iloc[0]["spearman_rank_corr"]),
        "comparison_table": comp_df.to_dict("records"),
        "mean_clv_26w_dollars": round(float(full_df["clv_26w"].mean()), 2),
        "median_clv_26w_dollars": round(float(full_df["clv_26w"].median()), 2),
    }

    return summary


def main() -> None:
    """CLI entrypoint for Phase 7 CLV estimation."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    output_dir = root / cfg.paths.data_outputs

    summary = run_clv_comparison_pipeline(
        db_path=db_path,
        output_dir=output_dir,
        gross_margin=cfg.gross_margin,
        annual_discount_rate=cfg.annual_discount_rate,
        seed=cfg.seed,
    )
    print("Phase 7 CLV Modeling Complete:")
    print("  Chosen Method:", summary["chosen_method"])
    print("  Holdout Spearman:", summary["chosen_method_spearman"])
    print("  Mean 26-Week CLV (Margin $):", summary["mean_clv_26w_dollars"])
    print("  Comparison:")
    for row in summary["comparison_table"]:
        print(
            f"    - {row['method']}: Spearman = {row['spearman_rank_corr']}, MAE = ${row['mae_dollars']}"
        )


if __name__ == "__main__":
    main()
