"""Smoke-test every delivered page against the included analysis artifacts."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("page", sorted((ROOT / "app").rglob("*.py")))
def test_dashboard_page_runs(page):
    if page.name in {"ui.py", "__init__.py"}:
        return
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception, [error.message for error in app.exception]
