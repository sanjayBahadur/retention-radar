"""Unit tests for market basket analysis and early-warning category signals."""

import duckdb
import pytest

from retention_radar.basket.early_warning import compute_early_warning_categories
from retention_radar.basket.rules import mine_association_rules
from retention_radar.config import get_project_root, load_project_config


def test_mine_association_rules_synthetic():
    """Verify FP-Growth association rule mining on synthetic multi-item baskets."""
    # Synthetic baskets where A and B always appear together (high lift)
    baskets = [
        ["MILK", "BREAD", "BUTTER"],
        ["MILK", "BREAD"],
        ["MILK", "BREAD", "EGGS"],
        ["MILK", "BREAD"],
        ["BEER", "CHIPS"],
        ["BEER", "CHIPS"],
        ["BEER", "CHIPS"],
        ["SODA"],
        ["APPLES", "ORANGES"],
        ["MILK", "BREAD", "CHEESE"],
    ]

    rules = mine_association_rules(baskets, min_support=0.20, min_lift=1.20)
    assert not rules.empty
    assert "antecedents_str" in rules.columns
    assert "consequents_str" in rules.columns
    assert "lift" in rules.columns
    assert (rules["lift"] >= 1.20).all()
    assert (rules["confidence"] > 0).all()


def test_mine_association_rules_empty():
    """Verify graceful handling of insufficient baskets."""
    rules = mine_association_rules([], min_support=0.01)
    assert rules.empty


@pytest.mark.requires_warehouse
def test_early_warning_categories():
    """Verify early-warning category analysis produces valid ranked outputs on warehouse."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    con = duckdb.connect(str(db_path), read_only=True)
    df = compute_early_warning_categories(con, min_early_spend=500.0)
    con.close()

    assert not df.empty
    assert len(df) >= 10
    required_cols = [
        "commodity_desc",
        "lapsing_hh_count",
        "net_drop_effect",
        "rank",
    ]
    for c in required_cols:
        assert c in df.columns

    # Verify that top ranked category exhibits negative drop effect
    assert df.iloc[0]["net_drop_effect"] < 0
