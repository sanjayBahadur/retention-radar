"""Unit tests for configuration validation."""

import pytest
from pydantic import ValidationError

from retention_radar.config import (
    ProjectConfig,
    load_project_config,
    load_states_config,
)


def test_load_project_config():
    """Verify project configuration loads default values properly."""
    cfg = load_project_config()
    assert cfg.seed == 42
    assert cfg.period_days == 28
    assert cfg.warmup_periods == 3
    assert cfg.train_periods == [4, 19]
    assert cfg.test_periods == [20, 25]
    assert cfg.gross_margin == 0.25
    assert cfg.annual_discount_rate == 0.10
    assert cfg.offer_cost == 2.00
    assert cfg.save_rate_grid == [0.05, 0.10, 0.15, 0.20]
    assert cfg.bootstrap_reps == 1000
    assert cfg.smoothing_alpha == 0.5
    assert cfg.optuna_trials == 50


def test_load_ci_profile():
    """Verify CI profile reduces bootstrap and Optuna trials for rapid execution."""
    cfg = load_project_config(profile="ci")
    assert cfg.bootstrap_reps == 50
    assert cfg.optuna_trials == 5


def test_load_states_config():
    """Verify lifecycle states configuration and ordering."""
    states_cfg = load_states_config()
    assert len(states_cfg.states) == 6
    assert states_cfg.state_names == ["New", "Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
    assert states_cfg.champion_top_pct == 0.25
    assert states_cfg.lapsed_threshold_missed_periods == 4


def test_config_missing_required_key_fails(tmp_path):
    """Verify ProjectConfig raises ValidationError when required keys are omitted."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("seed: 42\nperiod_days: 28\n")
    with pytest.raises(ValidationError):
        load_project_config(invalid_yaml)


def test_config_invalid_period_bounds(tmp_path):
    """Verify ProjectConfig raises ValueError if start period is after end period."""
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {
                "seed": 42,
                "period_days": 28,
                "warmup_periods": 3,
                "train_periods": [19, 4],  # invalid range
                "test_periods": [20, 25],
                "churn_horizon_periods": 2,
                "lapse_missed_periods": 4,
                "annual_discount_rate": 0.10,
                "gross_margin": 0.25,
                "offer_cost": 2.00,
                "save_rate_grid": [0.05],
                "bootstrap_reps": 100,
                "smoothing_alpha": 0.5,
                "optuna_trials": 10,
                "bi_tool": "tableau",
                "profiles": {"default": {"bootstrap_reps": 100, "optuna_trials": 10}},
                "paths": {
                    "data_raw": "data/raw",
                    "warehouse_db": "data/warehouse.duckdb",
                    "data_outputs": "data/outputs",
                    "bi_exports": "data/outputs/bi",
                    "models_dir": "data/outputs/models",
                    "reports_dir": "reports",
                    "figures_dir": "reports/figures",
                    "sql_results": "reports/sql_results",
                    "sample_fixture": "tests/fixtures/sample",
                },
            }
        )
