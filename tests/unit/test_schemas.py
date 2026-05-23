"""Unit tests for Pandera schemas validation."""

import pandas as pd

from retention_radar.config import get_project_root, load_project_config
from retention_radar.schemas import (
    attribution_schema,
    basket_rules_schema,
    clv_schema,
    state_distribution_schema,
    validate_all_parquet_outputs,
)


def test_state_distribution_schema_valid():
    """Verify state distribution schema validates properly formatted dataframe."""
    df = pd.DataFrame(
        {
            "period_id": [1, 2],
            "state": ["Champion", "Regular"],
            "household_count": [100, 200],
            "pct_of_period": [33.3, 66.7],
        }
    )
    validated = state_distribution_schema.validate(df)
    assert not validated.empty


def test_clv_schema_valid():
    """Verify CLV schema validates non-negative CLV and bounded alive probability."""
    df = pd.DataFrame(
        {
            "household_key": [1, 2],
            "method": ["LightGBM Tweedie", "LightGBM Tweedie"],
            "clv_26w": [250.0, 180.5],
            "p_alive": [0.95, 0.88],
        }
    )
    validated = clv_schema.validate(df)
    assert not validated.empty


def test_attribution_schema_valid():
    """Verify attribution schema validates share values in [0, 1]."""
    df = pd.DataFrame(
        {
            "campaign_type": ["TYPEA", "TYPEB"],
            "first_touch_share": [0.5, 0.5],
            "last_touch_share": [0.6, 0.4],
            "linear_share": [0.55, 0.45],
            "markov_removal_effect_share": [0.48, 0.52],
        }
    )
    validated = attribution_schema.validate(df)
    assert not validated.empty


def test_basket_rules_schema_valid():
    """Verify basket rules schema enforces lift >= 1.20."""
    df = pd.DataFrame(
        {
            "antecedents_str": ["A, B"],
            "consequents_str": ["C"],
            "support": [0.05],
            "confidence": [0.60],
            "lift": [2.50],
        }
    )
    validated = basket_rules_schema.validate(df)
    assert not validated.empty


def test_validate_all_parquet_outputs():
    """Verify validate_all_parquet_outputs successfully validates all parquet artifacts on disk."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs

    results = validate_all_parquet_outputs(out_dir)
    assert len(results) >= 10
    assert all(results.values())
