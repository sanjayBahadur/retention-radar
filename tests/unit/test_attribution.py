"""Unit tests for Phase 5 Markov campaign attribution and removal effects."""

import json

import pandas as pd

from retention_radar.attribution.markov_attribution import compute_removal_effects
from retention_radar.config import get_project_root


def test_toy_paths_removal_effects():
    """Verify removal effects on a known toy graph:

    Journeys:
      - A -> B -> CONVERSION (wt = 10)
      - A -> NULL (wt = 10)
      - B -> CONVERSION (wt = 10)
      - B -> NULL (wt = 10)

    Theoretical calculation:
      Base P(CONV) = 0.50
      Without A: P(CONV) = 0.3333 -> RE_A = 1/3
      Without B: P(CONV) = 0.0000 -> RE_B = 1.0
      Relative Shares: A = 0.25 (25%), B = 0.75 (75%)
    """
    channels = ["A", "B"]
    paths = [
        {"path": ["A", "B"], "converted": True, "weight": 10.0},
        {"path": ["A"], "converted": False, "weight": 10.0},
        {"path": ["B"], "converted": True, "weight": 10.0},
        {"path": ["B"], "converted": False, "weight": 10.0},
    ]

    shares = compute_removal_effects(paths, channels)
    assert abs(shares["A"] - 0.25) < 0.01
    assert abs(shares["B"] - 0.75) < 0.01
    assert abs(sum(shares.values()) - 1.0) < 1e-4


def test_attribution_outputs_exist_and_sum_to_one():
    """Verify attribution.parquet exists, columns sum to 1.0, and sensitivity is logged."""
    root = get_project_root()
    p_path = root / "data" / "outputs" / "attribution.parquet"
    assert p_path.exists()
    df = pd.read_parquet(p_path)

    for col in [
        "markov_removal_effect_share",
        "first_touch_share",
        "last_touch_share",
        "linear_share",
        "markov_redempt_sensitivity_share",
    ]:
        col_sum = df[col].sum()
        assert abs(col_sum - 1.0) < 0.01, f"{col} sum is {col_sum}, expected 1.0"

    summary_path = root / "data" / "outputs" / "attribution_paths_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert "caveat" in summary
    assert "models" in summary


def test_heuristic_attribution_models():
    """Verify first-touch, last-touch, and linear attribution models on multi-touch paths."""
    from retention_radar.attribution.heuristics import compute_heuristic_attribution

    channels = ["TYPEA", "TYPEB", "TYPEC"]
    paths = [
        {"path": ["TYPEA", "TYPEB"], "converted": True, "weight": 1.0},
        {"path": ["TYPEA", "TYPEB", "TYPEC"], "converted": True, "weight": 1.0},
        {"path": ["TYPEC"], "converted": False, "weight": 1.0},
    ]

    res = compute_heuristic_attribution(paths, channels)
    assert "first_touch" in res
    assert "last_touch" in res
    assert "linear" in res

    # In first-touch: both converted paths start with TYPEA -> TYPEA = 1.0
    assert res["first_touch"]["TYPEA"] == 1.0
    # In last-touch: one ends with TYPEB, one with TYPEC -> 0.5 each
    assert res["last_touch"]["TYPEB"] == 0.5
    assert res["last_touch"]["TYPEC"] == 0.5
