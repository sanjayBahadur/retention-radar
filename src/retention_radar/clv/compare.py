"""Compare 26-week spending forecasts using non-overlapping future windows."""

import json
from pathlib import Path
from typing import Any

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

HORIZON_DAYS = 182  # 26 weeks, also used as the feature lookback window.
TRAIN_CUTOFF_DAY = 336
TEST_CUTOFF_DAY = TRAIN_CUTOFF_DAY + HORIZON_DAYS
SCORE_CUTOFF_DAY = TEST_CUTOFF_DAY + HORIZON_DAYS
FEATURES = [
    "frequency",
    "recency_weeks",
    "T_weeks",
    "cal_spend",
    "monetary_value",
    "cal_trips",
    "days_since_last_purchase",
    "tenure_weeks",
]


def extract_rfm_split(
    db_path: Path,
    cal_end_day: int = TEST_CUTOFF_DAY,
    holdout_end_day: int = SCORE_CUTOFF_DAY,
) -> tuple[pd.DataFrame, float, float]:
    """Build past-only features and separate future labels, one row per household.

    A purchase occasion is a shopping day, after summing all item lines and
    baskets that day. Gamma-Gamma monetary value averages repeat occasions,
    excluding the first purchase in the lookback window.
    """
    if holdout_end_day < cal_end_day:
        raise ValueError("The outcome window must end on or after the feature cutoff.")
    history_start = cal_end_day - HORIZON_DAYS
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            """
            WITH days AS (
                SELECT household_key, day, SUM(sales_value) AS day_spend,
                       COUNT(DISTINCT basket_id) AS trips
                FROM raw.transaction_data
                WHERE day <= ?
                GROUP BY household_key, day
            ), history AS (
                SELECT household_key, MIN(day) AS first_day, MAX(day) AS last_day
                FROM days WHERE day <= ? GROUP BY household_key
            ), occasions AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY household_key ORDER BY day) AS visit
                FROM days WHERE day > ? AND day <= ? AND day_spend > 0
            ), cal AS (
                SELECT household_key, COUNT(*) - 1 AS frequency,
                       (MAX(day) - MIN(day)) / 7.0 AS recency_weeks,
                       (? - MIN(day)) / 7.0 AS T_weeks,
                       SUM(day_spend) AS cal_spend,
                       COALESCE(AVG(CASE WHEN visit > 1 THEN day_spend END), 0) AS monetary_value,
                       SUM(trips) AS cal_trips
                FROM occasions GROUP BY household_key
            ), hold AS (
                SELECT household_key, GREATEST(SUM(day_spend), 0) AS actual_holdout_spend,
                       SUM(trips) AS actual_holdout_trips
                FROM days WHERE day > ? AND day <= ? GROUP BY household_key
            )
            SELECT h.household_key, COALESCE(c.frequency, 0) AS frequency,
                   COALESCE(c.recency_weeks, 0) AS recency_weeks,
                   COALESCE(c.T_weeks, 26) AS T_weeks,
                   COALESCE(c.cal_spend, 0) AS cal_spend,
                   COALESCE(c.monetary_value, 0) AS monetary_value,
                   COALESCE(c.cal_trips, 0) AS cal_trips,
                   ? - h.last_day AS days_since_last_purchase,
                   (? - h.first_day) / 7.0 AS tenure_weeks,
                   COALESCE(o.actual_holdout_spend, 0) AS actual_holdout_spend,
                   COALESCE(o.actual_holdout_trips, 0) AS actual_holdout_trips
            FROM history h LEFT JOIN cal c USING (household_key)
            LEFT JOIN hold o USING (household_key)
            ORDER BY h.household_key
            """,
            [
                holdout_end_day,
                cal_end_day,
                history_start,
                cal_end_day,
                cal_end_day,
                cal_end_day,
                holdout_end_day,
                cal_end_day,
                cal_end_day,
            ],
        ).df()
    repeat = df[(df["frequency"] > 0) & (df["monetary_value"] > 0)]
    if (
        len(repeat) > 1
        and repeat["frequency"].nunique() > 1
        and repeat["monetary_value"].nunique() > 1
    ):
        corr, p_value = stats.pearsonr(repeat["frequency"], repeat["monetary_value"])
        return df, float(corr), float(p_value)
    return df, 0.0, 1.0


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluate sales-dollar forecasts, before margin and discount assumptions."""
    rank = float(stats.spearmanr(predicted, actual).statistic)
    return {
        "spearman_rank_corr": round(rank, 4) if np.isfinite(rank) else 0.0,
        "mae_dollars": round(float(np.mean(np.abs(predicted - actual))), 2),
        "rmse_dollars": round(float(np.sqrt(np.mean((predicted - actual) ** 2))), 2),
    }


def run_clv_comparison_pipeline(
    db_path: Path,
    output_dir: Path,
    gross_margin: float = 0.25,
    annual_discount_rate: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Train on days 337–518; evaluate on untouched days 519–700.

    The same households may appear at both origins. This evaluates forecasting
    future spending for known customers, not generalization to unseen people.
    Model selection uses test MAE; its result is exploratory, without a second
    untouched selection holdout.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train, _, _ = extract_rfm_split(db_path, TRAIN_CUTOFF_DAY, TEST_CUTOFF_DAY)
    test, corr, p_value = extract_rfm_split(db_path, TEST_CUTOFF_DAY, SCORE_CUTOFF_DAY)
    model = LGBMRegressor(
        objective="tweedie",
        random_state=seed,
        verbosity=-1,
        n_estimators=100,
        n_jobs=2,
    )
    model.fit(train[FEATURES], train["actual_holdout_spend"])

    # Fit the probabilistic baseline on purchase history available at this origin.
    bgnbd = fit_bgnbd_mle(test["frequency"], test["recency_weeks"], test["T_weeks"])
    gamma_gamma = fit_gamma_gamma(test["frequency"].to_numpy(), test["monetary_value"].to_numpy())

    def predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "Recent 26-week spend": frame["cal_spend"].to_numpy(),
            "BG/NBD + Gamma-Gamma": predict_bgnbd_purchases(
                frame["frequency"],
                frame["recency_weeks"],
                frame["T_weeks"],
                bgnbd,
                future_t=26.0,
            )
            * predict_monetary_value(frame["frequency"], frame["monetary_value"], gamma_gamma),
            "LightGBM (Tweedie)": np.maximum(model.predict(frame[FEATURES]), 0),
        }

    test_predictions = predictions(test)
    comparison = (
        pd.DataFrame(
            [
                {
                    "method": name,
                    **forecast_metrics(test["actual_holdout_spend"].to_numpy(), values),
                }
                for name, values in test_predictions.items()
            ]
        )
        .sort_values("mae_dollars")
        .reset_index(drop=True)
    )
    comparison.to_parquet(output_dir / "clv_comparison.parquet", index=False)
    selected = str(comparison.iloc[0]["method"])
    validation = test[["household_key", "actual_holdout_spend"]].copy()
    for name, values in test_predictions.items():
        validation[name] = values
    validation.to_parquet(output_dir / "clv_validation_predictions.parquet", index=False)

    discount = (1 + annual_discount_rate) ** (-0.5)

    def score_at(cutoff_day: int, filename: str, method: str) -> pd.DataFrame:
        # Keep the evaluated model fixed; never fit on this origin's future labels.
        frame, _, _ = extract_rfm_split(db_path, cutoff_day, cutoff_day)
        scored = pd.DataFrame(
            {
                "household_key": frame["household_key"],
                "method": method,
                "clv_26w": np.round(predictions(frame)[method] * gross_margin * discount, 2),
                "p_alive": predict_p_alive(
                    frame["frequency"],
                    frame["recency_weeks"],
                    frame["T_weeks"],
                    bgnbd,
                ).round(4),
                "cutoff_day": cutoff_day,
            }
        )
        scored.to_parquet(output_dir / filename, index=False)
        return scored

    scored = score_at(SCORE_CUTOFF_DAY, "clv.parquet", selected)
    # Historical churn scores use a prespecified model. Choosing the winner
    # using outcomes through day 700 would leak future model-selection evidence.
    score_at(23 * 28, "clv_at_churn_cutoff.parquet", "LightGBM (Tweedie)")
    summary = {
        "horizon_weeks": 26,
        "history_days": HORIZON_DAYS,
        "train_feature_cutoff_day": TRAIN_CUTOFF_DAY,
        "train_outcome_end_day": TEST_CUTOFF_DAY,
        "test_feature_cutoff_day": TEST_CUTOFF_DAY,
        "test_outcome_end_day": SCORE_CUTOFF_DAY,
        "score_cutoff_day": SCORE_CUTOFF_DAY,
        "evaluation_households": len(test),
        "error_unit": "sales dollars over 26 weeks",
        "frequency_monetary_correlation": round(corr, 4),
        "correlation_p_value": p_value,
        "chosen_method": selected,
        "selection_metric": "lowest test MAE (exploratory selection)",
        "chosen_method_spearman": float(comparison.iloc[0]["spearman_rank_corr"]),
        "comparison_table": comparison.to_dict("records"),
        "gross_margin_assumption": gross_margin,
        "annual_discount_rate_assumption": annual_discount_rate,
        "mean_clv_26w_dollars": round(float(scored["clv_26w"].mean()), 2),
        "median_clv_26w_dollars": round(float(scored["clv_26w"].median()), 2),
    }
    (output_dir / "clv_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Build reproducible spending forecasts and evaluation artifacts."""
    root = get_project_root()
    cfg = load_project_config()
    summary = run_clv_comparison_pipeline(
        root / cfg.paths.warehouse_db,
        root / cfg.paths.data_outputs,
        gross_margin=cfg.gross_margin,
        annual_discount_rate=cfg.annual_discount_rate,
        seed=cfg.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
