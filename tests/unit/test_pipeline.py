"""Unit tests for pipeline orchestration."""

import pytest

from retention_radar.pipeline import run_step


def test_run_step_success():
    """Verify run_step logs execution and returns elapsed duration."""
    executed = []

    def mock_step():
        executed.append(True)

    elapsed = run_step("Test Step", mock_step)
    assert elapsed >= 0.0
    assert len(executed) == 1


def test_run_step_failure():
    """Verify run_step propagates exceptions upon failure."""

    def failing_step():
        raise ValueError("Simulated failure in pipeline step")

    with pytest.raises(ValueError, match="Simulated failure"):
        run_step("Failing Step", failing_step)


def test_run_pipeline_orchestration(monkeypatch):
    """Verify run_pipeline executes and coordinates all modeling phases."""
    import retention_radar.pipeline as pl

    recorded_steps = []

    def mock_run_step(step_name, fn):
        recorded_steps.append(step_name)
        return 0.05

    monkeypatch.setattr(pl, "run_step", mock_run_step)

    # Test with skip flags
    pl.run_pipeline(skip_ingest=True, skip_dbt=True)
    assert len(recorded_steps) >= 13
    assert any("Exploratory" in s for s in recorded_steps)
    assert any("Markov" in s for s in recorded_steps)
    assert any("CLV" in s for s in recorded_steps)
    assert any("Churn" in s for s in recorded_steps)
    assert any("Uplift" in s for s in recorded_steps)


def test_pipeline_cli_main(monkeypatch):
    """Verify CLI entrypoint parses flags and invokes run_pipeline."""
    import sys
    from unittest.mock import MagicMock

    import retention_radar.pipeline as pl

    mock_run = MagicMock()
    monkeypatch.setattr(pl, "run_pipeline", mock_run)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--skip-ingest", "--skip-dbt"])

    pl.main()
    mock_run.assert_called_once_with(skip_ingest=True, skip_dbt=True)
