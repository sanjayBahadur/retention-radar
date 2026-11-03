"""Data ingestion, schema loading into DuckDB, profiling, and fixture sampling."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import duckdb

from retention_radar.config import get_project_root, load_project_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def ingest_raw_data(
    raw_dir: Path,
    db_path: Path,
    manifest_path: Path,
    seed: int = 42,
    create_fixture: bool = True,
    fixture_dir: Path = None,
) -> Dict[str, Any]:
    """Ingest raw CSV tables into DuckDB and create profiling documents."""
    logger.info(f"Connecting to DuckDB at {db_path}...")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw_x5;")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw_hillstrom;")

    manifest_entries: List[Dict[str, Any]] = []

    # 1. Ingest Core Dunnhumby tables
    dunnhumby_tables = [
        "campaign_desc",
        "campaign_table",
        "coupon",
        "coupon_redempt",
        "hh_demographic",
        "product",
        "transaction_data",
        "causal_data",
    ]

    dunnhumby_counts: Dict[str, int] = {}
    for table in dunnhumby_tables:
        csv_path = raw_dir / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required raw file: {csv_path}")

        logger.info(f"Ingesting Dunnhumby table '{table}' from {csv_path}...")
        sha256 = compute_sha256(csv_path)

        # Idempotent table creation in schema raw with clean lowercase snake_case columns
        con.execute(f"DROP TABLE IF EXISTS raw.{table};")
        desc_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{csv_path}', header=true)"
        ).fetchall()
        select_expr = ", ".join([f'"{c[0]}" AS {c[0].lower()}' for c in desc_cols])
        con.execute(f"""
            CREATE TABLE raw.{table} AS
            SELECT {select_expr} FROM read_csv_auto('{csv_path}', header=true);
        """)
        row_count = con.execute(f"SELECT COUNT(*) FROM raw.{table};").fetchone()[0]
        dunnhumby_counts[table] = row_count

        manifest_entries.append(
            {
                "source": "Dunnhumby The Complete Journey",
                "table": table,
                "filename": csv_path.name,
                "sha256": sha256,
                "row_count": row_count,
                "url": "https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey",
                "license": "DbCL-1.0 / Dunnhumby non-commercial research terms",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # 2. Ingest Hillstrom benchmark dataset
    hillstrom_csv = raw_dir / "hillstrom.csv"
    if hillstrom_csv.exists():
        logger.info(f"Ingesting Hillstrom benchmark dataset from {hillstrom_csv}...")
        con.execute("DROP TABLE IF EXISTS raw_hillstrom.customers;")
        desc_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{hillstrom_csv}', header=true)"
        ).fetchall()
        select_expr = ", ".join([f'"{c[0]}" AS {c[0].lower()}' for c in desc_cols])
        con.execute(f"""
            CREATE TABLE raw_hillstrom.customers AS
            SELECT {select_expr} FROM read_csv_auto('{hillstrom_csv}', header=true);
        """)
        hillstrom_count = con.execute("SELECT COUNT(*) FROM raw_hillstrom.customers;").fetchone()[0]
        manifest_entries.append(
            {
                "source": "Hillstrom MineThatData",
                "table": "customers",
                "filename": hillstrom_csv.name,
                "sha256": compute_sha256(hillstrom_csv),
                "row_count": hillstrom_count,
                "url": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
                "license": "Public challenge dataset",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # 3. Ingest X5 RetailHero benchmark data
    x5_dir = raw_dir / "x5"
    if (x5_dir / "clients.csv.gz").exists():
        logger.info("Ingesting X5 RetailHero clients...")
        con.execute("DROP TABLE IF EXISTS raw_x5.clients;")
        desc_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{x5_dir}/clients.csv.gz', header=true)"
        ).fetchall()
        select_expr = ", ".join([f'"{c[0]}" AS {c[0].lower()}' for c in desc_cols])
        con.execute(f"""
            CREATE TABLE raw_x5.clients AS
            SELECT {select_expr} FROM read_csv('{x5_dir}/clients.csv.gz', header=true);
        """)
        x5_clients_cnt = con.execute("SELECT COUNT(*) FROM raw_x5.clients;").fetchone()[0]
        manifest_entries.append(
            {
                "source": "X5 RetailHero",
                "table": "clients",
                "filename": "clients.csv.gz",
                "sha256": compute_sha256(x5_dir / "clients.csv.gz"),
                "row_count": x5_clients_cnt,
                "url": "https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz",
                "license": "X5 RetailHero Competition Terms",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if (x5_dir / "uplift_train.csv.gz").exists():
        logger.info("Ingesting X5 RetailHero uplift_train...")
        con.execute("DROP TABLE IF EXISTS raw_x5.uplift_train;")
        desc_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{x5_dir}/uplift_train.csv.gz', header=true)"
        ).fetchall()
        select_expr = ", ".join([f'"{c[0]}" AS {c[0].lower()}' for c in desc_cols])
        con.execute(f"""
            CREATE TABLE raw_x5.uplift_train AS
            SELECT {select_expr} FROM read_csv('{x5_dir}/uplift_train.csv.gz', header=true);
        """)
        x5_train_cnt = con.execute("SELECT COUNT(*) FROM raw_x5.uplift_train;").fetchone()[0]
        manifest_entries.append(
            {
                "source": "X5 RetailHero",
                "table": "uplift_train",
                "filename": "uplift_train.csv.gz",
                "sha256": compute_sha256(x5_dir / "uplift_train.csv.gz"),
                "row_count": x5_train_cnt,
                "url": "https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz",
                "license": "X5 RetailHero Competition Terms",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Write MANIFEST.json
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    # Build 200-household test fixture
    if create_fixture and fixture_dir:
        build_sample_fixture(con, fixture_dir, hillstrom_csv, seed=seed)

    con.close()
    return dunnhumby_counts


def build_sample_fixture(
    con: duckdb.DuckDBPyConnection,
    fixture_dir: Path,
    hillstrom_csv: Path,
    seed: int = 42,
    sample_size: int = 60,
) -> None:
    """Extract a representative sample across all tables for fast unit testing in CI (<10MB)."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating fixture in {fixture_dir} (sample_size={sample_size}, seed={seed})...")

    # Select representative households deterministically
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fixture_households AS
        SELECT household_key FROM (
            SELECT DISTINCT household_key
            FROM raw.transaction_data
            ORDER BY hash(household_key + {seed})
            LIMIT {sample_size}
        );
    """)

    # Filter transactions
    con.execute(f"""
        COPY (
            SELECT t.*
            FROM raw.transaction_data t
            INNER JOIN fixture_households h ON t.household_key = h.household_key
            ORDER BY t.day, t.household_key, t.basket_id
        ) TO '{fixture_dir}/transaction_data.csv' (HEADER, DELIMITER ',');
    """)

    # Products bought by fixture households
    con.execute(f"""
        COPY (
            SELECT DISTINCT p.*
            FROM raw.product p
            WHERE p.product_id IN (
                SELECT DISTINCT t.product_id
                FROM raw.transaction_data t
                INNER JOIN fixture_households h ON t.household_key = h.household_key
            )
            ORDER BY p.product_id
        ) TO '{fixture_dir}/product.csv' (HEADER, DELIMITER ',');
    """)

    # Demographic data for fixture households
    con.execute(f"""
        COPY (
            SELECT d.*
            FROM raw.hh_demographic d
            INNER JOIN fixture_households h ON d.household_key = h.household_key
            ORDER BY d.household_key
        ) TO '{fixture_dir}/hh_demographic.csv' (HEADER, DELIMITER ',');
    """)

    # Campaigns received by fixture households
    con.execute(f"""
        COPY (
            SELECT c.*
            FROM raw.campaign_table c
            INNER JOIN fixture_households h ON c.household_key = h.household_key
            ORDER BY c.household_key, c.campaign
        ) TO '{fixture_dir}/campaign_table.csv' (HEADER, DELIMITER ',');
    """)

    # Campaign descriptions
    con.execute(f"""
        COPY (
            SELECT * FROM raw.campaign_desc ORDER BY campaign
        ) TO '{fixture_dir}/campaign_desc.csv' (HEADER, DELIMITER ',');
    """)

    # Coupons relevant to fixture products/campaigns
    con.execute(f"""
        COPY (
            SELECT DISTINCT cp.*
            FROM raw.coupon cp
            WHERE cp.product_id IN (
                SELECT DISTINCT t.product_id
                FROM raw.transaction_data t
                INNER JOIN fixture_households h ON t.household_key = h.household_key
            )
            ORDER BY cp.coupon_upc
        ) TO '{fixture_dir}/coupon.csv' (HEADER, DELIMITER ',');
    """)

    # Redemptions by fixture households
    con.execute(f"""
        COPY (
            SELECT r.*
            FROM raw.coupon_redempt r
            INNER JOIN fixture_households h ON r.household_key = h.household_key
            ORDER BY r.household_key, r.day
        ) TO '{fixture_dir}/coupon_redempt.csv' (HEADER, DELIMITER ',');
    """)

    # Causal data filtered to fixture products & stores & active weeks
    con.execute(f"""
        COPY (
            SELECT c.*
            FROM raw.causal_data c
            INNER JOIN (
                SELECT DISTINCT product_id, store_id, week_no
                FROM raw.transaction_data
                WHERE household_key IN (SELECT household_key FROM fixture_households)
            ) t ON c.product_id = t.product_id AND c.store_id = t.store_id AND c.week_no = t.week_no
            ORDER BY c.store_id, c.week_no, c.product_id
        ) TO '{fixture_dir}/causal_data.csv' (HEADER, DELIMITER ',');
    """)

    # Copy full Hillstrom benchmark into fixture dir
    if hillstrom_csv.exists():
        con.execute(f"""
            COPY (SELECT * FROM raw_hillstrom.customers)
            TO '{fixture_dir}/hillstrom.csv' (HEADER, DELIMITER ',');
        """)

    logger.info("Sample fixture generated successfully.")


def generate_data_dictionary(db_path: Path, output_md: Path) -> None:
    """Generate a comprehensive data dictionary profiling every column across raw tables."""
    con = duckdb.connect(str(db_path), read_only=True)
    tables = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'raw'
        ORDER BY table_name;
    """).fetchall()

    lines = [
        "# Retention Radar Data Dictionary",
        "",
        "Profiling summary across all raw ingested retail datasets.",
        "",
    ]

    descriptions: Dict[str, Dict[str, str]] = {
        "transaction_data": {
            "household_key": "Uniquely identifies each frequent-shopper household",
            "basket_id": "Uniquely identifies each shopping trip / transaction occasion",
            "day": "Day number of the transaction (day 1 to 711 across 102 weeks)",
            "product_id": "Uniquely identifies each purchased retail product UPC",
            "quantity": "Number of units purchased in this transaction line",
            "sales_value": "Amount of dollars paid by customer after all discounts applied",
            "store_id": "Uniquely identifies the physical grocery retail store location",
            "retail_disc": "Loyalty card / price reduction discount applied (negative float)",
            "trans_time": "Time of day of the purchase (HHMM format)",
            "week_no": "Week number of the transaction (week 1 to 102)",
            "coupon_disc": "Manufacturer coupon discount applied (negative float)",
            "coupon_match_disc": "Retailer match on manufacturer coupon discount (negative float)",
        },
        "product": {
            "product_id": "Uniquely identifies each product item",
            "manufacturer": "Numeric ID representing the manufacturing brand owner",
            "department": "High-level retail category department (e.g. GROCERY, PASTRIES)",
            "brand": "Indicator whether brand is Private Label or National",
            "commodity_desc": "Commodity category group (e.g. MILK, SOFT DRINKS, BREAD)",
            "sub_commodity_desc": "Granular category classification",
            "curr_size_of_product": "Package size string (e.g. 16 OZ, 1 GAL)",
        },
        "hh_demographic": {
            "household_key": "Household identifier linking to transactions",
            "age_desc": "Estimated age band of primary shopper (e.g. 35-44, 45-54)",
            "marital_status_code": "Marital status code (A = Married, B = Single, U = Unknown)",
            "income_desc": "Annual household income bracket (e.g. 35-49K, 50-74K)",
            "homeowner_desc": "Home ownership status (Homeowner, Renter, Probable)",
            "hh_comp_desc": "Household composition classification",
            "household_size_desc": "Number of individuals residing in household (1, 2, 3, 4, 5+)",
            "kid_category_desc": "Presence and number of children in household",
        },
        "campaign_desc": {
            "description": "Campaign strategy classification (Type A, Type B, Type C)",
            "campaign": "Numeric identifier for the direct mail campaign",
            "start_day": "Calendar day number when the campaign mailings began",
            "end_day": "Calendar day number when the campaign mailings concluded",
        },
        "campaign_table": {
            "description": "Campaign strategy type",
            "household_key": "Household targeted for campaign communication",
            "campaign": "Numeric identifier for the campaign",
        },
        "coupon": {
            "coupon_upc": "Standard universal product code representing the coupon",
            "product_id": "Product ID for which coupon is valid",
            "campaign": "Campaign under which coupon was distributed",
        },
        "coupon_redempt": {
            "household_key": "Household that redeemed the coupon",
            "day": "Day number when the coupon was redeemed",
            "coupon_upc": "UPC code of the redeemed coupon",
            "campaign": "Campaign under which redeemed coupon was issued",
        },
        "causal_data": {
            "product_id": "Product identifier",
            "store_id": "Store location identifier",
            "week_no": "Week number of promotional activity",
            "display": "In-store display location placement flag (0, 1, 2, etc.)",
            "mailer": "Circular / weekly mailer feature page placement flag (0, A, B, etc.)",
        },
    }

    for (tbl,) in tables:
        total_rows = con.execute(f"SELECT COUNT(*) FROM raw.{tbl};").fetchone()[0]
        cols_info = con.execute(f"DESCRIBE raw.{tbl};").fetchall()

        lines.append(f"### `raw.{tbl}` ({total_rows:,} rows)")
        lines.append("")
        lines.append("| Column | Type | Distinct | Null % | Min | Max | Description |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")

        for col_name, col_type, null_ok, *rest in cols_info:
            stats = con.execute(f"""
                SELECT
                    COUNT(DISTINCT {col_name}),
                    ROUND(SUM(CASE WHEN {col_name} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2),
                    MIN(CAST({col_name} AS VARCHAR)),
                    MAX(CAST({col_name} AS VARCHAR))
                FROM raw.{tbl};
            """).fetchone()

            distinct_cnt, null_pct, min_v, max_v = stats
            min_str = (str(min_v)[:20] + "...") if min_v and len(str(min_v)) > 20 else str(min_v)
            max_str = (str(max_v)[:20] + "...") if max_v and len(str(max_v)) > 20 else str(max_v)
            desc = descriptions.get(tbl, {}).get(col_name, "Documented field")

            lines.append(
                f"| `{col_name}` | {col_type} | {distinct_cnt:,} | {null_pct}% | {min_str} | {max_str} | {desc} |"
            )
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Data dictionary generated at {output_md}")
    con.close()


def main() -> None:
    """CLI entrypoint for data ingestion."""
    root = get_project_root()
    cfg = load_project_config()

    raw_dir = root / cfg.paths.data_raw
    db_path = root / cfg.paths.warehouse_db
    manifest_path = raw_dir / "MANIFEST.json"
    fixture_dir = root / cfg.paths.sample_fixture
    data_dict_md = root / "docs" / "raw_schema_profile.md"

    counts = ingest_raw_data(
        raw_dir=raw_dir,
        db_path=db_path,
        manifest_path=manifest_path,
        seed=cfg.seed,
        create_fixture=True,
        fixture_dir=fixture_dir,
    )

    generate_data_dictionary(db_path=db_path, output_md=data_dict_md)
    logger.info("Phase 1 ingestion complete. Summary:")
    for tbl, count in counts.items():
        logger.info(f"  - {tbl}: {count:,} rows")


if __name__ == "__main__":
    main()
