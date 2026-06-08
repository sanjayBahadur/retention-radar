"""Pytest configuration and custom hooks for Retention Radar test suite."""

import pytest

from retention_radar.config import get_project_root, load_project_config


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_warehouse: mark test as requiring local data/warehouse.duckdb"
    )


def pytest_runtest_setup(item):
    """Automatically skip tests marked with requires_warehouse if warehouse.duckdb is missing."""
    if item.get_closest_marker("requires_warehouse"):
        root = get_project_root()
        cfg = load_project_config()
        if not (root / cfg.paths.warehouse_db).exists():
            pytest.skip("Full data/warehouse.duckdb not present in clone / CI environment")
