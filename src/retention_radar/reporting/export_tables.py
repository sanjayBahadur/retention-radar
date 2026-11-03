"""Export lifecycle shares and dbt SQL examples from the current warehouse."""

import subprocess
import sys
from pathlib import Path

import duckdb

from retention_radar.config import get_project_root, load_project_config


def main():
    root = get_project_root()
    cfg = load_project_config()
    dbt = root / "dbt"
    subprocess.run(
        [
            str(Path(sys.executable).parent / "dbt"),
            "compile",
            "--project-dir",
            str(dbt),
            "--profiles-dir",
            str(dbt),
            "--select",
            "path:analyses",
        ],
        cwd=dbt,
        check=True,
    )
    out = root / cfg.paths.data_outputs
    queries = root / cfg.paths.sql_results
    out.mkdir(parents=True, exist_ok=True)
    queries.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(root / cfg.paths.warehouse_db), read_only=True) as connection:
        distribution = connection.execute("""
            SELECT period_id, state, COUNT(*) AS household_count,
                   COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY period_id) AS pct_of_period
            FROM marts.fct_household_period_state
            GROUP BY period_id, state ORDER BY period_id, state
        """).df()
        distribution.to_parquet(out / "state_distribution.parquet", index=False)
        for source in sorted((dbt / "analyses").glob("*.sql")):
            compiled = dbt / "target" / "compiled" / "retention_radar" / "analyses" / source.name
            connection.execute(compiled.read_text()).df().to_csv(
                queries / f"{source.stem}.csv", index=False
            )
    print("Exported current lifecycle shares and SQL analysis results.")


if __name__ == "__main__":
    main()
