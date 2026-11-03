"""Held-out mean-effect sanity comparison on a randomized email dataset."""

import json
from pathlib import Path
from typing import Any, Dict

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from retention_radar.config import get_project_root, load_project_config


def run_hillstrom_benchmark(
    db_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare held-out model mean effects with a randomized-sample visit-rate contrast.

    Agreement is a sanity check, not proof of individual effect accuracy or
    external validity. Disagreement is recorded rather than blocking the pipeline.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT
            recency,
            history,
            mens,
            womens,
            newbie,
            channel,
            zip_code,
            segment,
            visit,
            conversion,
            spend
        FROM raw_hillstrom.customers
        WHERE segment IN ('Womens E-Mail', 'No E-Mail');
    """).df()
    con.close()

    df["treatment"] = (df["segment"] == "Womens E-Mail").astype(int)
    df["target"] = df["visit"].astype(int)

    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    train, held_out = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=seed, stratify=df["treatment"]
    )
    evaluation = df.iloc[held_out]
    treated_rate = float(evaluation.loc[evaluation["treatment"] == 1, "target"].mean())
    control_rate = float(evaluation.loc[evaluation["treatment"] == 0, "target"].mean())
    raw_diff = treated_rate - control_rate

    # Bootstrap 95% CI on empirical difference in means
    rng = np.random.default_rng(seed)
    n = len(evaluation)
    t_vals = evaluation["treatment"].values
    y_vals = evaluation["target"].values
    boot_diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t_sample = t_vals[idx]
        y_sample = y_vals[idx]
        diff_sample = float(np.mean(y_sample[t_sample == 1]) - np.mean(y_sample[t_sample == 0]))
        boot_diffs.append(diff_sample)

    ci_lower = round(float(np.percentile(boot_diffs, 2.5)), 5)
    ci_upper = round(float(np.percentile(boot_diffs, 97.5)), 5)

    # Feature matrix for meta-learner
    X = pd.get_dummies(
        df[["recency", "history", "mens", "womens", "newbie", "channel", "zip_code"]],
        drop_first=True,
        dtype=float,
    )

    from causalml.inference.meta import BaseTLearner

    t_learner = BaseTLearner(
        learner=LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1, n_jobs=1)
    )
    t_learner.fit(
        X=X.iloc[train].values,
        treatment=df.iloc[train]["treatment"].values,
        y=df.iloc[train]["target"].values,
    )
    cate_preds = t_learner.predict(X=X.iloc[held_out].values).ravel()
    model_ate = round(float(np.mean(cate_preds)), 5)

    within_ci = ci_lower <= model_ate <= ci_upper

    result = {
        "dataset": "Hillstrom Email Marketing",
        "treatment_group": "Womens E-Mail",
        "control_group": "No E-Mail",
        "sample_size": len(df),
        "evaluation_sample_size": n,
        "evaluation": "Stratified 70/30 customer holdout",
        "interpretation": "Mean-effect agreement is a sanity check, not a causal-validity certificate",
        "treated_visit_rate": round(treated_rate, 4),
        "control_visit_rate": round(control_rate, 4),
        "raw_diff": round(raw_diff, 5),
        "bootstrap_ci_95": (ci_lower, ci_upper),
        "causalml_t_learner_ate": model_ate,
        "is_within_ci": within_ci,
    }

    return result


def main() -> None:
    """Run Hillstrom benchmark and print results."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    print("Running Hillstrom held-out mean-effect comparison...")
    res = run_hillstrom_benchmark(db_path=db_path, n_bootstrap=cfg.bootstrap_reps, seed=cfg.seed)
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hillstrom_benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
    print("\nHillstrom Benchmark Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nHillstrom held-out mean-effect sanity check saved.")


if __name__ == "__main__":
    main()
