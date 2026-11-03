"""Exercise raw data → dbt → features → models without credentials or saved outputs."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd
import yaml
from generate_synthetic_fixture import generate_synthetic_fixtures

from retention_radar.churn.train import train_and_evaluate_all_models
from retention_radar.config import get_project_root
from retention_radar.features.build_features import (
    build_rolling_origin_dataset,
    extract_features_at_cutoff,
)
from retention_radar.ingest import ingest_raw_data


def verify_fixture():
    root = get_project_root()
    with TemporaryDirectory(prefix="retention-fixture-") as directory:
        work = Path(directory)
        raw, warehouse = work / "raw", work / "fixture.duckdb"
        generate_synthetic_fixtures(raw)
        ingest_raw_data(raw, warehouse, work / "manifest.json", create_fixture=False)
        profile = {
            "retention_radar": {
                "target": "dev",
                "outputs": {"dev": {"type": "duckdb", "path": str(warehouse), "threads": 2}},
            }
        }
        (work / "profiles.yml").write_text(yaml.safe_dump(profile))

        def build():
            command = [
                str(Path(sys.executable).parent / "dbt"),
                "build",
                "--project-dir",
                str(root / "dbt"),
                "--profiles-dir",
                str(work),
                "--target-path",
                str(work / "target"),
                "--log-path",
                str(work / "logs"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)

        build()
        with duckdb.connect(str(warehouse)) as connection:
            before = extract_features_at_cutoff(connection, 10)
            # Change the future source data, then rebuild every derived table.
            connection.execute("""
                INSERT INTO raw.transaction_data
                SELECT * REPLACE (day + 350 AS day, basket_id + 1000000 AS basket_id)
                FROM raw.transaction_data WHERE day BETWEEN 1 AND 100
            """)
        build()
        with duckdb.connect(str(warehouse), read_only=True) as connection:
            after = extract_features_at_cutoff(connection, 10)
        columns = before.columns.drop("churn_label")
        pd.testing.assert_frame_equal(
            before.sort_values("household_key")[columns].reset_index(drop=True),
            after.sort_values("household_key")[columns].reset_index(drop=True),
        )
        splits = build_rolling_origin_dataset(warehouse)
        artifacts, comparison, _, _ = train_and_evaluate_all_models(
            splits, n_trials=2, n_bootstrap=20
        )
        assert np.isfinite(comparison["pr_auc"]).all()
        assert len(artifacts["test_labels"]) == len(splits["test"])
        assert max(splits["train"].cutoff_period) + 2 < min(splits["val"].cutoff_period)
        assert max(splits["val"].cutoff_period) + 2 < min(splits["test"].cutoff_period)
        print("Fixture passed: raw → dbt (including constraints) → features → model evaluation.")
        print("Future transaction changes left historical features unchanged after a full rebuild.")
        print(comparison[["model", "pr_auc", "top_20_capture"]].to_string(index=False))


if __name__ == "__main__":
    verify_fixture()
