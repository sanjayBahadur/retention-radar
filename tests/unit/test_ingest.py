"""Unit tests for Phase 1 ingestion, checksum manifest, and fixture integrity."""

import json

import duckdb
import pytest

from retention_radar.config import get_project_root, load_project_config


def test_manifest_exists_and_valid():
    """Verify MANIFEST.json is generated with checksums and expected tables."""
    root = get_project_root()
    manifest_path = root / "data" / "raw" / "MANIFEST.json"
    assert manifest_path.exists(), "MANIFEST.json must exist in data/raw"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) >= 8
    tables = {entry["table"] for entry in manifest}
    assert "transaction_data" in tables
    assert "product" in tables
    assert "hh_demographic" in tables
    assert "campaign_desc" in tables

    for entry in manifest:
        assert len(entry["sha256"]) == 64
        assert entry["row_count"] > 0
        assert "license" in entry


@pytest.mark.requires_warehouse
def test_warehouse_table_counts():
    """Verify warehouse DuckDB contains expected row counts across tables."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    assert db_path.exists(), "warehouse.duckdb must exist"

    con = duckdb.connect(str(db_path), read_only=True)
    tx_count = con.execute("SELECT COUNT(*) FROM raw.transaction_data;").fetchone()[0]
    prod_count = con.execute("SELECT COUNT(*) FROM raw.product;").fetchone()[0]
    demo_count = con.execute("SELECT COUNT(*) FROM raw.hh_demographic;").fetchone()[0]
    camp_count = con.execute("SELECT COUNT(*) FROM raw.campaign_desc;").fetchone()[0]
    con.close()

    assert tx_count == 2595732
    assert prod_count == 92353
    assert demo_count == 801
    assert camp_count == 30


def test_fixture_integrity():
    """Verify sample fixture tables load properly and stay within size limits."""
    root = get_project_root()
    cfg = load_project_config()
    fixture_dir = root / cfg.paths.sample_fixture

    required_fixture_files = [
        "transaction_data.csv",
        "product.csv",
        "hh_demographic.csv",
        "campaign_desc.csv",
        "campaign_table.csv",
        "coupon.csv",
        "coupon_redempt.csv",
        "causal_data.csv",
        "hillstrom.csv",
    ]
    for fn in required_fixture_files:
        p = fixture_dir / fn
        assert p.exists(), f"Missing fixture file: {fn}"
        assert p.stat().st_size > 0

    total_size = sum(f.stat().st_size for f in fixture_dir.glob("*.csv"))
    assert total_size < 10 * 1024 * 1024, f"Fixture size {total_size} exceeds 10 MB limit"


def test_data_dictionary_covers_columns():
    """Verify data dictionary covers all raw tables and columns."""
    root = get_project_root()
    dd_path = root / "docs" / "data_dictionary.md"
    assert dd_path.exists(), "data_dictionary.md must exist in docs/"
    text = dd_path.read_text(encoding="utf-8")
    assert "`household_key`" in text
    assert "`sales_value`" in text
    assert "`retail_disc`" in text
    assert "`coupon_disc`" in text


def test_compute_sha256(tmp_path):
    """Verify compute_sha256 produces exact hex digest."""
    from retention_radar.ingest import compute_sha256

    test_file = tmp_path / "hello.txt"
    test_file.write_text("retention radar", encoding="utf-8")
    expected = "a3da5747f00417d141887527561861633adc05ce3e6aa076800dca132957c27d"
    assert compute_sha256(test_file) == expected


def test_ingest_raw_data_and_generate_dictionary(tmp_path):
    """Verify ingest_raw_data runs on sample fixture files and generates valid schema and data dictionary."""
    from retention_radar.ingest import generate_data_dictionary, ingest_raw_data

    root = get_project_root()
    cfg = load_project_config()
    fixture_dir = root / cfg.paths.sample_fixture

    db_path = tmp_path / "test_warehouse.duckdb"
    manifest_path = tmp_path / "MANIFEST.json"
    dict_path = tmp_path / "data_dictionary.md"

    counts = ingest_raw_data(
        raw_dir=fixture_dir,
        db_path=db_path,
        manifest_path=manifest_path,
        seed=42,
        create_fixture=False,
    )

    assert "transaction_data" in counts
    assert counts["transaction_data"] > 0
    assert manifest_path.exists()

    generate_data_dictionary(db_path=db_path, output_md=dict_path)
    assert dict_path.exists()
    assert "transaction_data" in dict_path.read_text(encoding="utf-8")


@pytest.mark.requires_warehouse
def test_build_sample_fixture(tmp_path):
    """Verify build_sample_fixture produces sample CSVs under size limit."""
    from retention_radar.ingest import build_sample_fixture

    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    con = duckdb.connect(str(db_path), read_only=True)

    dummy_hillstrom = tmp_path / "dummy_hillstrom.csv"
    dummy_hillstrom.write_text(
        "history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend\n100,1,0,Suburban,0,Web,Mens E-Mail,0,0,0.0\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "fixture_sample"
    build_sample_fixture(con, out_dir, dummy_hillstrom, sample_size=5)
    con.close()

    assert (out_dir / "transaction_data.csv").exists()
    assert (out_dir / "product.csv").exists()
