"""Kaplan-Meier survival curves and Cox proportional hazards regression."""

from pathlib import Path
from typing import Any, Dict

import duckdb
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter


def fit_survival_models(
    db_path: Path,
    output_parquet: Path,
) -> Dict[str, Any]:
    """Fit Kaplan-Meier survival curves by starting state and Cox regression for time-to-lapsed."""
    con = duckdb.connect(str(db_path), read_only=True)

    # Extract time-to-lapsed from period 4 (start of training)
    df = con.execute("""
        WITH baseline AS (
            SELECT
                household_key,
                state AS baseline_state,
                trips AS baseline_trips,
                spend AS baseline_spend
            FROM marts.fct_household_period_state
            WHERE period_id = 4
        ),
        lapsed_events AS (
            SELECT
                household_key,
                MIN(period_id) AS lapsed_period
            FROM marts.fct_household_period_state
            WHERE period_id >= 4 AND state = 'Lapsed'
            GROUP BY household_key
        )
        SELECT
            b.household_key,
            b.baseline_state,
            b.baseline_trips,
            b.baseline_spend,
            d.has_demographics,
            COALESCE(l.lapsed_period - 4, 21) AS duration_periods,
            CASE WHEN l.lapsed_period IS NOT NULL THEN 1 ELSE 0 END AS event_lapsed
        FROM baseline b
        LEFT JOIN lapsed_events l ON b.household_key = l.household_key
        LEFT JOIN marts.dim_household d ON b.household_key = d.household_key;
    """).df()
    con.close()

    # 1. Fit Kaplan-Meier by baseline state
    kmf = KaplanMeierFitter()
    curve_records = []
    median_survival: Dict[str, float | None] = {}

    for state in ["Champion", "Regular", "At-Risk", "Dormant"]:
        sub = df[df["baseline_state"] == state]
        if len(sub) == 0:
            continue
        kmf.fit(durations=sub["duration_periods"], event_observed=sub["event_lapsed"], label=state)
        median = float(kmf.median_survival_time_)
        median_survival[state] = median if np.isfinite(median) else None
        for t, s in zip(kmf.timeline, kmf.survival_function_[state]):
            curve_records.append(
                {
                    "baseline_state": state,
                    "timeline_periods": float(t),
                    "timeline_days": float(t * 28),
                    "survival_probability": float(s),
                }
            )

    curves_df = pd.DataFrame(curve_records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    curves_df.to_parquet(output_parquet, index=False)

    # 2. Fit Cox Proportional Hazards Model
    cox_data = df[
        ["baseline_trips", "baseline_spend", "has_demographics", "duration_periods", "event_lapsed"]
    ].copy()
    cox_data["log_spend"] = np.log1p(cox_data["baseline_spend"])
    cox_data = cox_data.drop(columns=["baseline_spend"])

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="duration_periods", event_col="event_lapsed")
    cph_summary = cph.summary.to_dict()

    return {
        "median_survival_periods": median_survival,
        "concordance_scope": "Descriptive fit on observed histories; not held-out evaluation",
        "concordance_index": round(float(cph.concordance_index_), 4),
        "cph_summary": cph_summary,
    }
