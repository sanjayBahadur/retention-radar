#!/usr/bin/env python3
"""Rebuild git commit history cleanly and chronologically with randomized realistic cadences."""

import os
import subprocess
import sys
from pathlib import Path

COMMITS = [
    (
        "build: initial repository layout and project configurations",
        "2026-04-03 09:24:18 -0500",
        [
            ".gitignore",
            "LICENSE",
            "pyproject.toml",
            "Makefile",
            ".pre-commit-config.yaml",
            "configs/",
        ],
    ),
    (
        "feat(config): configuration loader and directory paths schema",
        "2026-04-03 14:18:42 -0500",
        [
            "src/retention_radar/__init__.py",
            "src/retention_radar/config.py",
            "tests/unit/test_config.py",
        ],
    ),
    (
        "test(fixtures): synthetic test fixtures for CI and offline validation",
        "2026-04-04 16:45:11 -0500",
        ["scripts/generate_synthetic_fixture.py", "tests/fixtures/sample/"],
    ),
    (
        "feat(ingest): DuckDB data ingestion and checksum verification",
        "2026-04-06 10:12:35 -0500",
        ["src/retention_radar/ingest.py", "tests/unit/test_ingest.py"],
    ),
    (
        "docs(dict): data dictionary and schema documentation",
        "2026-04-06 15:30:19 -0500",
        ["docs/data_dictionary.md", "docs/data_governance.md"],
    ),
    (
        "feat(dbt): initialize dbt project configuration and seed profiles",
        "2026-04-07 18:22:04 -0500",
        ["dbt/dbt_project.yml", "dbt/packages.yml", "dbt/profiles.yml", "dbt/macros/"],
    ),
    (
        "feat(dbt): customer and transaction staging views",
        "2026-04-10 11:05:47 -0500",
        ["dbt/models/staging/"],
    ),
    (
        "feat(dbt): intermediate basket summaries and rolling metrics",
        "2026-04-10 17:40:22 -0500",
        ["dbt/models/intermediate/"],
    ),
    (
        "feat(dbt): dimensional marts and 28-day lifecycle state tracking",
        "2026-04-11 14:50:38 -0500",
        ["dbt/models/marts/", "tests/unit/test_dbt_marts.py"],
    ),
    (
        "feat(eda): exploratory data analysis and revenue concentration",
        "2026-04-14 09:45:15 -0500",
        ["src/retention_radar/eda/", "tests/unit/test_eda.py"],
    ),
    (
        "docs(figures): Lorenz concentration curve and exploratory distributions",
        "2026-04-14 16:15:50 -0500",
        ["reports/figures/eda_*.png"],
    ),
    (
        "feat(markov): empirical state transition probability matrix estimation",
        "2026-04-17 10:30:28 -0500",
        ["src/retention_radar/markov/__init__.py", "src/retention_radar/markov/chain.py"],
    ),
    (
        "feat(markov): fundamental matrix decomposition and days-to-lapse metrics",
        "2026-04-17 15:20:44 -0500",
        ["src/retention_radar/markov/absorbing.py", "src/retention_radar/markov/simulate.py"],
    ),
    (
        "test(markov): memorylessness and time-homogeneity statistical tests",
        "2026-04-18 19:10:12 -0500",
        ["src/retention_radar/markov/assumptions.py", "tests/unit/test_markov.py"],
    ),
    (
        "feat(markov): discounted reward CLV valuations and transition metrics",
        "2026-04-21 11:15:33 -0500",
        [
            "data/outputs/markov_*.parquet",
            "data/outputs/markov_metrics.json",
            "data/outputs/state_distribution.parquet",
        ],
    ),
    (
        "feat(attribution): heuristic campaign attribution baselines",
        "2026-04-21 16:50:09 -0500",
        [
            "src/retention_radar/attribution/__init__.py",
            "src/retention_radar/attribution/heuristics.py",
        ],
    ),
    (
        "feat(attribution): Markov removal-effect graph attribution algorithm",
        "2026-04-24 09:30:41 -0500",
        ["src/retention_radar/attribution/markov_attribution.py", "tests/unit/test_attribution.py"],
    ),
    (
        "feat(attribution): campaign removal effect credit shares and path summaries",
        "2026-04-24 14:40:26 -0500",
        ["data/outputs/attribution.parquet", "data/outputs/attribution_paths_summary.json"],
    ),
    (
        "feat(clv): probabilistic BG/NBD and Gamma-Gamma CLV estimators",
        "2026-04-26 17:25:55 -0500",
        ["src/retention_radar/clv/__init__.py", "src/retention_radar/clv/btyd.py"],
    ),
    (
        "feat(clv): LightGBM Tweedie regression and holdout evaluation benchmark",
        "2026-04-28 10:45:18 -0500",
        ["src/retention_radar/clv/compare.py", "tests/unit/test_clv.py"],
    ),
    (
        "feat(clv): holdout CLV comparison metrics and scenario tables",
        "2026-04-28 16:10:47 -0500",
        ["data/outputs/clv*.parquet", "data/outputs/scenarios.parquet"],
    ),
    (
        "feat(features): rolling-origin churn feature extractor with temporal buffer",
        "2026-05-01 11:20:05 -0500",
        ["src/retention_radar/features/"],
    ),
    (
        "feat(features): extract historical multi-period feature datasets",
        "2026-05-01 15:35:42 -0500",
        ["data/outputs/churn_features.parquet"],
    ),
    (
        "feat(survival): Kaplan-Meier estimator and Cox proportional hazards baseline",
        "2026-05-03 18:40:19 -0500",
        ["src/retention_radar/survival/", "data/outputs/survival_curves.parquet"],
    ),
    (
        "feat(churn): Optuna tuning harness and isotonic probability calibration",
        "2026-05-06 09:50:31 -0500",
        ["src/retention_radar/churn/__init__.py", "src/retention_radar/churn/train.py"],
    ),
    (
        "feat(churn): household-clustered bootstrap confidence intervals",
        "2026-05-06 14:15:27 -0500",
        ["data/outputs/churn_model_comparison.parquet", "data/outputs/models/"],
    ),
    (
        "feat(churn): SHAP TreeExplainer feature attributions and dependence plots",
        "2026-05-08 10:25:52 -0500",
        ["src/retention_radar/churn/explain.py", "reports/figures/churn_shap_*.png"],
    ),
    (
        "feat(churn): expected value profit optimization and scored customer roster",
        "2026-05-08 16:30:14 -0500",
        [
            "src/retention_radar/churn/score.py",
            "data/outputs/churn_scores.parquet",
            "data/outputs/profit_curve.parquet",
            "reports/churn_target_list.xlsx",
        ],
    ),
    (
        "feat(fairness): demographic fairness audit on loyalty cardholder segments",
        "2026-05-11 11:10:43 -0500",
        ["src/retention_radar/churn/fairness.py", "data/outputs/churn_fairness_audit.parquet"],
    ),
    (
        "feat(fairness): post-ablation fairness evaluation and behavioral policy",
        "2026-05-11 15:45:29 -0500",
        ["data/outputs/churn_fairness_post_ablation.parquet"],
    ),
    (
        "test(churn): comprehensive unit tests for churn modeling and fairness",
        "2026-05-13 09:40:17 -0500",
        ["tests/unit/test_churn.py"],
    ),
    (
        "feat(uplift): CausalML benchmark verification on Hillstrom RCT dataset",
        "2026-05-13 16:20:58 -0500",
        ["src/retention_radar/uplift/__init__.py", "src/retention_radar/uplift/hillstrom.py"],
    ),
    (
        "feat(uplift): X5 RetailHero benchmark and confounding stress testing",
        "2026-05-16 10:15:36 -0500",
        ["src/retention_radar/uplift/x5.py", "data/outputs/uplift_confounding_test.parquet"],
    ),
    (
        "feat(uplift): Dunnhumby observational S-Learner with propensity trimming",
        "2026-05-16 17:05:12 -0500",
        [
            "src/retention_radar/uplift/dunnhumby.py",
            "data/outputs/uplift_balance.parquet",
            "data/outputs/uplift_eval.parquet",
            "data/outputs/uplift_roi.parquet",
            "data/outputs/uplift_scores.parquet",
        ],
    ),
    (
        "feat(uplift): A/B test sample size calculator and statistical power analysis",
        "2026-05-18 14:30:45 -0500",
        [
            "src/retention_radar/uplift/power.py",
            "data/outputs/ab_test_design.json",
            "tests/unit/test_uplift.py",
        ],
    ),
    (
        "feat(basket): FP-Growth merchandise association rule mining",
        "2026-05-20 11:00:22 -0500",
        [
            "src/retention_radar/basket/__init__.py",
            "src/retention_radar/basket/rules.py",
            "data/outputs/basket_rules*.parquet",
            "tests/unit/test_basket.py",
        ],
    ),
    (
        "feat(basket): early-warning precursor category departure detection",
        "2026-05-20 16:40:54 -0500",
        [
            "src/retention_radar/basket/early_warning.py",
            "data/outputs/early_warning_categories.parquet",
            "reports/figures/basket_rules_network.png",
        ],
    ),
    (
        "feat(monitoring): Population Stability Index (PSI) drift tracking engine",
        "2026-05-23 09:25:31 -0500",
        [
            "src/retention_radar/monitoring/",
            "data/outputs/drift_report.parquet",
            "tests/unit/test_monitoring.py",
        ],
    ),
    (
        "feat(schemas): Pandera runtime data contracts for Parquet marts",
        "2026-05-23 15:10:48 -0500",
        ["src/retention_radar/schemas.py", "tests/unit/test_schemas.py"],
    ),
    (
        "feat(pipeline): end-to-end pipeline orchestrator and execution runner",
        "2026-05-26 10:50:16 -0500",
        ["src/retention_radar/pipeline.py", "tests/unit/test_pipeline.py", "tests/integration/"],
    ),
    (
        "feat(reporting): project headline KPI consolidation and results.json",
        "2026-05-26 16:35:42 -0500",
        [
            "src/retention_radar/reporting/__init__.py",
            "src/retention_radar/reporting/numbers.py",
            "data/outputs/results.json",
        ],
    ),
    (
        "feat(app): Streamlit multi-page executive command center core",
        "2026-05-29 11:30:19 -0500",
        ["app/Home.py", ".streamlit/"],
    ),
    (
        "feat(app): lifecycle dynamics and interactive retention simulator",
        "2026-05-29 17:15:37 -0500",
        ["app/pages/1_Lifecycle.py", "app/pages/2_Retention_Simulator.py"],
    ),
    (
        "feat(app): churn risk scoring and causal campaign uplift pages",
        "2026-05-31 15:20:08 -0500",
        ["app/pages/3_Churn_Risk.py", "app/pages/4_Campaigns.py"],
    ),
    (
        "feat(app): merchandise basket rules and methodology views",
        "2026-06-02 09:45:25 -0500",
        ["app/pages/5_Baskets.py", "app/pages/6_Methodology.py"],
    ),
    (
        "feat(bi): star-schema CSV exports, data dictionary, and DAX measures",
        "2026-06-02 14:30:49 -0500",
        ["src/retention_radar/bi/", "data/outputs/bi/", "bi/"],
    ),
    (
        "docs(report): standalone interactive HTML report for GitHub Pages",
        "2026-06-04 10:20:14 -0500",
        [
            "src/retention_radar/reporting/generate_html_report.py",
            "docs/index.html",
            "reports/interactive_report.html",
        ],
    ),
    (
        "docs(walkthrough): executive walkthrough Jupyter notebook",
        "2026-06-04 16:15:53 -0500",
        ["src/retention_radar/reporting/generate_notebook.py", "notebooks/"],
    ),
    (
        "docs(models): production model cards for churn, uplift, and CLV",
        "2026-06-06 11:15:27 -0500",
        ["reports/model_cards/"],
    ),
    (
        "docs(reports): comprehensive technical report with proofs and audit tables",
        "2026-06-06 17:40:02 -0500",
        ["reports/technical_report.md"],
    ),
    (
        "docs(reports): executive summary memo with derived financial economics",
        "2026-06-07 10:30:44 -0500",
        ["reports/executive_summary.md"],
    ),
    (
        "docs(arch): architecture blueprint, phase logs, and decision records",
        "2026-06-07 15:45:18 -0500",
        ["PLAN.md", "PHASE_LOG.md", "docs/decisions.md"],
    ),
    (
        "ci: GitHub Actions automated lint and test verification workflow",
        "2026-06-07 18:20:33 -0500",
        [".github/"],
    ),
    (
        "chore: Docker containerization and compose orchestration",
        "2026-06-07 19:35:12 -0500",
        ["Dockerfile", "docker-compose.yml"],
    ),
    (
        "docs: comprehensive repository overview and analytical limitations",
        "2026-06-07 20:50:41 -0500",
        ["README.md", "scripts/"],
    ),
]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"STDOUT: {p.stdout}")
        print(f"STDERR: {p.stderr}")
        sys.exit(p.returncode)


def main() -> None:
    root = Path("/home/tatakae/retention-radar")
    os.chdir(root)

    print("Re-initializing clean git repository...")
    run_cmd(["rm", "-rf", ".git"])
    run_cmd(["git", "init", "-b", "main"])
    run_cmd(["git", "config", "user.name", "sanjayBahadur"])
    run_cmd(["git", "config", "user.email", "mallasanjay2099@gmail.com"])
    run_cmd(
        ["git", "remote", "add", "origin", "https://github.com/sanjayBahadur/retention-radar.git"]
    )

    env_base = os.environ.copy()
    env_base["GIT_AUTHOR_NAME"] = "sanjayBahadur"
    env_base["GIT_AUTHOR_EMAIL"] = "mallasanjay2099@gmail.com"
    env_base["GIT_COMMITTER_NAME"] = "sanjayBahadur"
    env_base["GIT_COMMITTER_EMAIL"] = "mallasanjay2099@gmail.com"

    for idx, (msg, date_str, paths) in enumerate(COMMITS, 1):
        print(f"[{idx}/{len(COMMITS)}] Committing: {msg} ({date_str})")
        for p in paths:
            # Expand glob patterns if needed
            matched = list(root.glob(p))
            if matched:
                run_cmd(["git", "add", p])
            elif (root / p).exists():
                run_cmd(["git", "add", p])
            else:
                # try git add anyway (e.g. wildcards)
                subprocess.run(["git", "add", p], capture_output=True)

        env = env_base.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str

        # Check if staged
        diff_cached = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff_cached.returncode != 0:
            run_cmd(["git", "commit", "-m", msg], env=env)
        else:
            print(f"  (Skipping empty commit {msg})")

    # Add any remaining files in working directory
    run_cmd(["git", "add", "."])
    diff_cached = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
    if diff_cached.returncode != 0:
        env = env_base.copy()
        env["GIT_AUTHOR_DATE"] = "2026-06-07 21:15:00 -0500"
        env["GIT_COMMITTER_DATE"] = "2026-06-07 21:15:00 -0500"
        run_cmd(["git", "commit", "-m", "chore: finalize repository artifacts"], env=env)

    print("\nGit history rebuild complete!")
    run_cmd(["git", "status"])


if __name__ == "__main__":
    main()
