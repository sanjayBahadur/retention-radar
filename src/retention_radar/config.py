"""Configuration loading and validation for Retention Radar."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_raw: Path
    warehouse_db: Path
    data_outputs: Path
    bi_exports: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    sql_results: Path
    sample_fixture: Path


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap_reps: int
    optuna_trials: int


class LifecycleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    transient: bool = True
    absorbing: bool = False
    order: int


class StatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    states: List[LifecycleState]
    champion_top_pct: float = Field(..., ge=0.0, le=1.0)
    trailing_spend_periods: int = Field(..., ge=1)
    lapsed_threshold_missed_periods: int = Field(..., ge=1)
    dormant_min_missed_periods: int = Field(..., ge=1)
    dormant_max_missed_periods: int = Field(..., ge=1)

    @property
    def state_names(self) -> List[str]:
        return [s.name for s in sorted(self.states, key=lambda s: s.order)]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(...)
    period_days: int = Field(...)
    warmup_periods: int = Field(...)
    train_periods: List[int] = Field(..., min_length=2, max_length=2)
    test_periods: List[int] = Field(..., min_length=2, max_length=2)
    churn_horizon_periods: int = Field(...)
    lapse_missed_periods: int = Field(...)

    # Business assumptions (must be labeled as assumptions in outputs)
    annual_discount_rate: float = Field(..., ge=0.0, le=1.0)
    gross_margin: float = Field(..., ge=0.0, le=1.0)
    offer_cost: float = Field(..., ge=0.0)
    save_rate_grid: List[float] = Field(..., min_length=1)

    # Statistical parameters
    bootstrap_reps: int = Field(..., ge=1)
    smoothing_alpha: float = Field(..., ge=0.0)
    optuna_trials: int = Field(..., ge=1)
    bi_tool: str = "tableau"

    profiles: Dict[str, ProfileConfig]
    paths: PathsConfig

    @field_validator("train_periods", "test_periods")
    @classmethod
    def validate_period_range(cls, v: List[int]) -> List[int]:
        if v[0] > v[1]:
            raise ValueError(f"Start period {v[0]} cannot be greater than end period {v[1]}")
        return v


def get_project_root() -> Path:
    """Find the project root containing configs/."""
    # Look upward from current file
    curr = Path(__file__).resolve().parent
    for p in [curr, curr.parent, curr.parent.parent]:
        if (p / "configs" / "project.yaml").is_file():
            return p
    # Fallback to working directory
    cwd = Path.cwd()
    if (cwd / "configs" / "project.yaml").is_file():
        return cwd
    return curr.parent.parent


def load_project_config(
    config_path: Optional[Path] = None,
    profile: Optional[str] = None,
) -> ProjectConfig:
    """Load and validate configs/project.yaml into a typed ProjectConfig model."""
    root = get_project_root()
    path = config_path or (root / "configs" / "project.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Project configuration file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    config = ProjectConfig.model_validate(data)

    # Apply profile overrides if requested
    if profile:
        if profile not in config.profiles:
            raise KeyError(f"Profile '{profile}' not defined in {path}")
        prof = config.profiles[profile]
        config.bootstrap_reps = prof.bootstrap_reps
        config.optuna_trials = prof.optuna_trials

    return config


def load_states_config(config_path: Optional[Path] = None) -> StatesConfig:
    """Load and validate configs/states.yaml into a typed StatesConfig model."""
    root = get_project_root()
    path = config_path or (root / "configs" / "states.yaml")
    if not path.exists():
        raise FileNotFoundError(f"States configuration file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    return StatesConfig.model_validate(data)
