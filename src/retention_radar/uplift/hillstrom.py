"""Hillstrom benchmark test for causal uplift validation."""

from pathlib import Path
from typing import Any, Dict

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from retention_radar.config import get_project_root, load_project_config


def run_hillstrom_benchmark(
    db_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validate that CausalML estimated ATE matches empirical difference within 95% bootstrap CI."""
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

    treated_rate = float(df.loc[df["treatment"] == 1, "target"].mean())
    control_rate = float(df.loc[df["treatment"] == 0, "target"].mean())
    raw_diff = treated_rate - control_rate

    # Bootstrap 95% CI on empirical difference in means
    rng = np.random.default_rng(seed)
    n = len(df)
    t_vals = df["treatment"].values
    y_vals = df["target"].values
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

    t_learner = BaseTLearner(learner=LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1))
    t_learner.fit(X=X.values, treatment=t_vals, y=y_vals)
    cate_preds = t_learner.predict(X=X.values).ravel()
    model_ate = round(float(np.mean(cate_preds)), 5)

    within_ci = ci_lower <= model_ate <= ci_upper

    result = {
        "dataset": "Hillstrom Email Marketing",
        "treatment_group": "Womens E-Mail",
        "control_group": "No E-Mail",
        "sample_size": n,
        "treated_visit_rate": round(treated_rate, 4),
        "control_visit_rate": round(control_rate, 4),
        "raw_diff": round(raw_diff, 5),
        "bootstrap_ci_95": (ci_lower, ci_upper),
        "causalml_t_learner_ate": model_ate,
        "is_within_ci": within_ci,
    }

    if not within_ci:
        raise AssertionError(
            f"CausalML ATE {model_ate} is outside the empirical bootstrap 95% CI ({ci_lower}, {ci_upper})"
        )

    return result


def main() -> None:
    """Run Hillstrom benchmark and print results."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db

    print("Running Hillstrom benchmark causal validation...")
    res = run_hillstrom_benchmark(db_path=db_path)
    print("\nHillstrom Benchmark Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nHillstrom causal validation PASSED.")


if __name__ == "__main__":
    main()
