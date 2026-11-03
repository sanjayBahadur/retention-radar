"""End-to-end orchestration pipeline executing all data, modeling, and evaluation phases."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from retention_radar.config import get_project_root, load_project_config


def run_step(step_name: str, fn: Callable[[], None]) -> float:
    """Execute a single pipeline step with execution logging and timing."""
    print("\n========================================================")
    print(f"▶ STARTING STEP: {step_name}")
    print("========================================================")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"✔ COMPLETED STEP: {step_name} in {elapsed:.2f}s")
        return elapsed
    except Exception as e:
        print(f"✖ FAILED STEP: {step_name} after {time.time() - t0:.2f}s")
        print(f"Error: {e}")
        raise


def run_pipeline(
    skip_ingest: bool = False,
    skip_dbt: bool = False,
) -> None:
    """Execute the full end-to-end Retention Radar pipeline."""
    project_root = get_project_root()
    cfg = load_project_config()

    print("========================================================")
    print("RETENTION RADAR END-TO-END PIPELINE")
    print(f"Root: {project_root}")
    print(f"Seed: {cfg.seed}")
    print(f"Warehouse: {cfg.paths.warehouse_db}")
    print(f"Outputs: {cfg.paths.data_outputs}")
    print("========================================================")

    total_t0 = time.time()

    # Step 1: Ingestion
    if not skip_ingest:
        from retention_radar.ingest import main as ingest_main

        run_step("1. Data Ingestion (DuckDB)", ingest_main)
    else:
        print("\n[Skipping Data Ingestion]")

    # Step 2: dbt Transformations
    if not skip_dbt:

        def dbt_step():
            dbt_dir = project_root / "dbt"
            print("Running dbt transformations...")
            cmd = [
                str(Path(sys.executable).parent / "dbt"),
                "build",
                "--profiles-dir",
                str(dbt_dir),
            ]
            subprocess.run(cmd, cwd=str(dbt_dir), check=True)

        run_step("2. dbt Modeling & Lifecycle States", dbt_step)
    else:
        print("\n[Skipping dbt transformations]")

    from retention_radar.reporting.export_tables import main as tables_main

    run_step("2b. Lifecycle and SQL exports", tables_main)

    # Step 3: EDA
    from retention_radar.eda.analysis import main as eda_main

    run_step("3. Exploratory Data Analysis & Lorenz Figures", eda_main)

    # Step 4: Markov Lifecycle Modeling
    from retention_radar.markov.chain import main as markov_main

    run_step("4. Markov Transition Matrices & Reward CLV", markov_main)

    # Step 5: Multi-Touch Campaign Attribution
    from retention_radar.attribution.markov_attribution import main as attr_main

    run_step("5. Multi-Touch Attribution & Removal Effects", attr_main)

    # Step 6: Customer Lifetime Value (CLV)
    from retention_radar.clv.compare import main as clv_main

    run_step("6. CLV Modeling (BG/NBD, Gamma-Gamma, Tweedie)", clv_main)

    # Step 7: Churn Feature Engineering & Modeling
    from retention_radar.features.build_features import main as feat_main

    run_step("7a. Churn Rolling-Origin Feature Extraction", feat_main)

    from retention_radar.churn.train import main as churn_train_main

    run_step("7b. Churn LightGBM Tuning & Benchmarking", churn_train_main)

    from retention_radar.churn.fairness import main as churn_fair_main

    run_step("7c. Churn Demographic Fairness Audit", churn_fair_main)

    from retention_radar.churn.score import main as churn_score_main

    run_step("7d. Churn Scoring, SHAP Reason Codes, Profit Curves", churn_score_main)

    # Step 8: Uplift Modeling & Experiment Design
    from retention_radar.uplift.hillstrom import main as uplift_hill_main

    run_step("8a. Hillstrom Benchmark Causal Validation", uplift_hill_main)

    from retention_radar.uplift.x5 import main as uplift_x5_main

    run_step("8b. X5 RetailHero Uplift & Confounding Stress Test", uplift_x5_main)

    from retention_radar.uplift.dunnhumby import main as uplift_dunn_main

    run_step("8c. Dunnhumby Observational Uplift & Segmentation", uplift_dunn_main)

    from retention_radar.uplift.power import main as uplift_power_main

    run_step("8d. A/B Test Statistical Power Analysis", uplift_power_main)

    # Step 9: Market Basket & Early-Warning Categories
    from retention_radar.basket.rules import main as basket_rules_main

    run_step("9a. Market Basket Association Rules (FP-Growth)", basket_rules_main)

    from retention_radar.basket.early_warning import main as basket_ew_main

    run_step("9b. Early-Warning Category Departure Signals", basket_ew_main)

    # Step 10: Drift Monitoring & Pandera Schema Validation
    from retention_radar.monitoring.drift import main as drift_main

    run_step("10a. Population Stability Index (PSI) Drift Monitoring", drift_main)

    from retention_radar.schemas import main as schemas_main

    run_step("10b. Pandera Parquet Schema Verification", schemas_main)

    from retention_radar.bi.export import main as bi_main
    from retention_radar.reporting.evaluation import main as evaluation_main
    from retention_radar.reporting.generate_html_report import main as html_main
    from retention_radar.reporting.numbers import main as numbers_main

    run_step("11. BI exports", bi_main)
    run_step("12. Churn error review", evaluation_main)
    run_step("13. Artifact-derived findings", numbers_main)
    run_step("14. Public case study", html_main)

    total_elapsed = time.time() - total_t0
    print("\n========================================================")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed:.2f}s")
    print("========================================================")


def main() -> None:
    """CLI entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description="Retention Radar Pipeline Orchestrator")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    parser.add_argument("--skip-dbt", action="store_true", help="Skip dbt transformation")
    args = parser.parse_args()

    run_pipeline(skip_ingest=args.skip_ingest, skip_dbt=args.skip_dbt)


if __name__ == "__main__":
    main()
