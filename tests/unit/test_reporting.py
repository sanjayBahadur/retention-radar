"""Regression checks for evidence-backed findings and error accounting."""

import json

import pandas as pd
import pytest

from retention_radar.config import get_project_root
from retention_radar.reporting.evaluation import summarize_errors
from retention_radar.reporting.numbers import compile_results_json


def test_missing_artifacts_cannot_publish_success(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing report inputs"):
        compile_results_json(tmp_path)
    assert not (tmp_path / "results.json").exists()


def test_public_findings_match_actual_model_artifacts():
    out = get_project_root() / "data" / "outputs"
    results = compile_results_json(out)
    evaluation = pd.read_parquet(out / "uplift_eval.parquet").set_index("learner")
    selected = results["uplift"]["dunnhumby_best_learner"]
    assert results["uplift"]["dunnhumby_qini"] == evaluation.loc[selected, "qini"]
    roster = pd.read_parquet(out / "uplift_scores.parquet")
    assert sum(results["uplift"]["segments"].values()) == roster.household_key.nunique()
    assert (
        results["clv"]["selected_method"]
        == json.loads((out / "clv_summary.json").read_text())["chosen_method"]
    )
    assert results["evidence"]["financial_impact"] == "modeled scenario"
    saved = json.loads((out / "results.json").read_text())
    assert saved["artifact_sha256"] == results["artifact_sha256"], (
        "Regenerate stale public findings."
    )
    errors = pd.read_parquet(out / "churn_error_analysis.parquet")
    overall = errors.loc[errors.dimension == "Overall"].iloc[0]
    assert overall.capture_rate == pytest.approx(
        results["churn"]["top_20pct_capture_rate"], abs=0.0001
    )


def test_error_counts_reconcile_at_each_cutoff():
    predictions = pd.DataFrame(
        {
            "household_key": list(range(10)) * 2,
            "cutoff_period": [20] * 10 + [21] * 10,
            "current_state": ["Regular"] * 20,
            "tenure_periods": [10] * 20,
            "churn_label": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 2,
            "raw_p": list(range(10, 0, -1)) * 2,
        }
    )
    summary = summarize_errors(predictions)
    overall = summary[summary.dimension == "Overall"].iloc[0]
    assert overall.contacts == 4
    assert overall.captured_departures == 2
    assert overall.missed_departures == 2
    assert overall.false_alarms == 2
    assert overall.households == 10


def test_zero_value_households_do_not_gain_invented_value():
    from retention_radar.churn.score import compute_profit_curves

    scores = pd.DataFrame({"calibrated_p": [0.9, 0.8], "clv_26w": [0.0, 0.0]})
    frontier = compute_profit_curves(scores, save_rates=[0.1], offer_cost=2.0)
    assert (frontier.total_ev == -frontier.total_cost).all()
    best = frontier.loc[frontier.total_ev.idxmax()]
    assert best.households_targeted == 0
    assert best.total_cost == 0
