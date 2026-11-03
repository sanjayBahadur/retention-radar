"""Randomized uplift modeling and confounding stress test on X5 RetailHero."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseDRLearner, BaseSLearner, BaseTLearner, BaseXLearner
from causalml.metrics import auuc_score, qini_score
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from retention_radar.config import get_project_root, load_project_config


def compute_smd(x_treated: np.ndarray, x_control: np.ndarray) -> float:
    """Compute Standardized Mean Difference (SMD) between treated and control groups."""
    mean_t = np.mean(x_treated)
    mean_c = np.mean(x_control)
    var_t = np.var(x_treated, ddof=1)
    var_c = np.var(x_control, ddof=1)
    pooled_std = np.sqrt((var_t + var_c) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float(np.abs(mean_t - mean_c) / pooled_std)


def load_x5_preprocessed(
    db_path: Path,
    sample_size: int = 60000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess X5 clients and uplift dataset."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT
            u.client_id,
            u.treatment_flg,
            u.target,
            COALESCE(c.age, 46.0) AS age,
            CASE WHEN c.age IS NULL THEN 1 ELSE 0 END AS age_is_missing,
            COALESCE(c.gender, 'U') AS gender
        FROM raw_x5.uplift_train u
        LEFT JOIN raw_x5.clients c ON u.client_id = c.client_id
        ORDER BY u.client_id;
    """).df()
    con.close()

    # Redemption and issue-date features are excluded: their timing relative
    # to treatment is not established by this extract.
    df["age"] = df["age"].clip(lower=18, upper=95)

    df = pd.get_dummies(df, columns=["gender"], drop_first=True, dtype=float)
    feat_cols = ["age", "age_is_missing"] + [c for c in df if c.startswith("gender_")]

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df, feat_cols


def verify_x5_randomization(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Describe observed baseline balance; balance alone cannot prove random assignment."""
    treated = df[df["treatment_flg"] == 1]
    control = df[df["treatment_flg"] == 0]

    records = []

    for col in feat_cols:
        smd = compute_smd(treated[col].values, control[col].values)
        records.append(
            {
                "feature": col,
                "treated_mean": round(float(np.mean(treated[col])), 4),
                "control_mean": round(float(np.mean(control[col])), 4),
                "abs_smd": round(smd, 4),
                "is_balanced": smd < 0.10,
            }
        )

    balance_df = pd.DataFrame(records)
    return balance_df


def fit_and_evaluate_learners(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Fit S-, T-, X-, and DR-learners and compute Qini, AUUC, and uplift metrics."""
    X_train = train_df[feat_cols].values
    t_train = train_df["treatment_flg"].values
    y_train = train_df["target"].values

    X_test = test_df[feat_cols].values
    t_test = test_df["treatment_flg"].values
    y_test = test_df["target"].values

    np.random.seed(seed)  # CausalML DR-Learner uses NumPy for internal splits.
    predictions: Dict[str, np.ndarray] = {}

    # 1. S-Learner
    s_learner = BaseSLearner(
        learner=LGBMRegressor(n_estimators=80, random_state=seed, verbose=-1, n_jobs=1)
    )
    s_learner.fit(X=X_train, treatment=t_train, y=y_train)
    predictions["S-Learner"] = s_learner.predict(X=X_test).ravel()

    # 2. T-Learner
    t_learner = BaseTLearner(
        learner=LGBMRegressor(n_estimators=80, random_state=seed, verbose=-1, n_jobs=1)
    )
    t_learner.fit(X=X_train, treatment=t_train, y=y_train)
    predictions["T-Learner"] = t_learner.predict(X=X_test).ravel()

    # 3. X-Learner
    p_propensity = np.full(len(X_train), 0.5)
    p_test_propensity = np.full(len(X_test), 0.5)
    x_learner = BaseXLearner(
        learner=LGBMRegressor(n_estimators=80, random_state=seed, verbose=-1, n_jobs=1)
    )
    x_learner.fit(X=X_train, treatment=t_train, y=y_train, p=p_propensity)
    predictions["X-Learner"] = x_learner.predict(X=X_test, p=p_test_propensity).ravel()

    # 4. DR-Learner (Doubly Robust)
    dr_learner = BaseDRLearner(
        learner=LGBMRegressor(n_estimators=80, random_state=seed, verbose=-1, n_jobs=1)
    )
    dr_learner.fit(X=X_train, treatment=t_train, y=y_train, p=p_propensity)
    predictions["DR-Learner"] = dr_learner.predict(X=X_test, p=p_test_propensity).ravel()

    # Metrics evaluation on test set
    n_test = len(test_df)
    results = []

    for name, cate in predictions.items():
        eval_df = pd.DataFrame({"y": y_test, "w": t_test, "pred": cate})
        qini = float(qini_score(eval_df)["pred"])
        auuc = float(auuc_score(eval_df)["pred"])

        # Uplift at top 10% and top 30%
        order = np.argsort(-cate)
        top_10_k = int(n_test * 0.10)
        top_30_k = int(n_test * 0.30)

        def get_top_uplift(k: int) -> float:
            idx = order[:k]
            sub_y = y_test[idx]
            sub_t = t_test[idx]
            t_rate = np.mean(sub_y[sub_t == 1]) if np.sum(sub_t == 1) > 0 else 0.0
            c_rate = np.mean(sub_y[sub_t == 0]) if np.sum(sub_t == 0) > 0 else 0.0
            return float(t_rate - c_rate)

        uplift_10 = get_top_uplift(top_10_k)
        uplift_30 = get_top_uplift(top_30_k)

        # Bootstrap Qini 95% CI
        rng = np.random.default_rng(seed)
        boot_qinis = []
        for _ in range(min(n_bootstrap, 200)):
            b_idx = rng.choice(n_test, size=n_test, replace=True)
            b_eval = pd.DataFrame({"y": y_test[b_idx], "w": t_test[b_idx], "pred": cate[b_idx]})
            try:
                boot_qinis.append(float(qini_score(b_eval)["pred"]))
            except Exception:
                pass

        if not boot_qinis:
            raise ValueError("No valid Qini bootstrap samples; uncertainty cannot be reported.")
        q_lower = round(float(np.percentile(boot_qinis, 2.5)), 4)
        q_upper = round(float(np.percentile(boot_qinis, 97.5)), 4)

        results.append(
            {
                "learner": name,
                "qini": round(qini, 4),
                "qini_ci_lower": q_lower,
                "qini_ci_upper": q_upper,
                "auuc": round(auuc, 4),
                "uplift_at_10pct": round(uplift_10, 4),
                "uplift_at_30pct": round(uplift_30, 4),
                "ate_mean": round(float(np.mean(cate)), 4),
            }
        )

    eval_table = pd.DataFrame(results).sort_values("qini", ascending=False).reset_index(drop=True)
    return eval_table, predictions


def run_confounding_stress_test(
    df: pd.DataFrame,
    reference_ate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare estimates after age-based selection to a finite randomized-sample reference.

    The reference is uncertain and selection changes the sample. Deviations from
    it are not population bias estimates, and this does not test hidden confounding.
    """
    rng = np.random.default_rng(seed)
    df_c = df.copy()

    # Only observed age drives synthetic selection, and age is available to estimators.
    age_norm = (df_c["age"] - df_c["age"].min()) / (df_c["age"].max() - df_c["age"].min() + 1e-5)
    keep_prob = np.where(df_c["treatment_flg"] == 1, 0.2 + 0.8 * age_norm, 0.5)
    keep_mask = rng.uniform(0, 1, size=len(df_c)) < keep_prob
    confounded_df = df_c[keep_mask].copy().reset_index(drop=True)

    y = confounded_df["target"].values
    t = confounded_df["treatment_flg"].values
    X = confounded_df[["age"]].values

    # 1. Naive difference in means
    treated_mean = np.mean(y[t == 1])
    control_mean = np.mean(y[t == 0])
    naive_ate = float(treated_mean - control_mean)

    # Cross-fit nuisance models, so each row is evaluated with models fitted
    # on other clients. The selection exercise uses only measured age.
    e = np.zeros(len(y))
    mu_1 = np.zeros(len(y))
    mu_0 = np.zeros(len(y))
    for train, test in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, t):
        propensity = LogisticRegression(max_iter=1000, random_state=seed)
        propensity.fit(X[train], t[train])
        e[test] = np.clip(propensity.predict_proba(X[test])[:, 1], 0.05, 0.95)
        for arm, predictions in ((1, mu_1), (0, mu_0)):
            arm_train = train[t[train] == arm]
            model = LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1, n_jobs=1)
            model.fit(X[arm_train], y[arm_train])
            predictions[test] = model.predict(X[test])

    weights = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    ipw_treated = np.sum(y[t == 1] * weights[t == 1]) / np.sum(weights[t == 1])
    ipw_control = np.sum(y[t == 0] * weights[t == 0]) / np.sum(weights[t == 0])
    ipw_ate = float(ipw_treated - ipw_control)

    aipw_scores = (mu_1 - mu_0) + (t * (y - mu_1) / e) - ((1.0 - t) * (y - mu_0) / (1.0 - e))
    aipw_ate = float(np.mean(aipw_scores))

    methods = [
        {
            "method": "Randomized sample reference",
            "estimate": round(reference_ate, 4),
            "bias": 0.0000,
        },
        {
            "method": "Naive Difference in Means",
            "estimate": round(naive_ate, 4),
            "bias": round(naive_ate - reference_ate, 4),
        },
        {
            "method": "Inverse Propensity Weighting (IPW)",
            "estimate": round(ipw_ate, 4),
            "bias": round(ipw_ate - reference_ate, 4),
        },
        {
            "method": "Augmented IPW (AIPW / Doubly Robust)",
            "estimate": round(aipw_ate, 4),
            "bias": round(aipw_ate - reference_ate, 4),
        },
    ]

    result = pd.DataFrame(methods)
    # Keep the original artifact column for compatibility; it is deviation from
    # an uncertain sample reference, not known population estimation bias.
    result["reference_deviation"] = result["bias"]
    return result


def main() -> None:
    """Run X5 randomized uplift modeling and confounding stress test."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading X5 RetailHero dataset...")
    df, feat_cols = load_x5_preprocessed(db_path=db_path, sample_size=60000, seed=cfg.seed)

    print("\nChecking X5 baseline covariate balance...")
    balance_df = verify_x5_randomization(df, feat_cols)
    print(balance_df.to_string())
    balance_df.to_parquet(out_dir / "x5_balance.parquet", index=False)

    # Finite-sample randomized difference; not known population ground truth
    reference_ate = float(
        df.loc[df["treatment_flg"] == 1, "target"].mean()
        - df.loc[df["treatment_flg"] == 0, "target"].mean()
    )
    print(f"\nX5 randomized-sample visit-rate difference: {reference_ate:+.4f}")

    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=cfg.seed, stratify=df[["treatment_flg", "target"]]
    )

    print("\nFitting CausalML S-, T-, X-, and DR-learners...")
    n_boot = min(cfg.bootstrap_reps, 200)
    eval_table, _ = fit_and_evaluate_learners(
        train_df=train_df,
        test_df=test_df,
        feat_cols=feat_cols,
        n_bootstrap=n_boot,
        seed=cfg.seed,
    )
    print("\nX5 Uplift Learner Comparison Table:")
    print(eval_table.to_string())

    eval_table.to_parquet(out_dir / "x5_uplift_eval.parquet", index=False)
    benchmark = {
        "dataset": "X5 RetailHero",
        "sample_size": len(df),
        "evaluation_sample_size": len(test_df),
        "reference_visit_rate_difference": round(reference_ate, 6),
        "treatment_share": round(float(df["treatment_flg"].mean()), 4),
        "features": feat_cols,
        "learner_comparison": eval_table.to_dict("records"),
        "balance": balance_df.to_dict("records"),
        "observed_balance_below_0_10": bool(balance_df["is_balanced"].all()),
        "evaluation": "Stratified 70/30 client holdout",
        "limits": [
            "Randomized-sample difference is an estimate, not known population ground truth.",
            "Redemption and issue-date features excluded because pre-treatment timing is unverified.",
            "Age-selection exercise uses measured confounding; it cannot verify hidden confounding.",
            "Bootstrap intervals condition on fitted models and do not include model-selection uncertainty.",
        ],
    }
    (out_dir / "x5_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    print("\nRunning age-selection sensitivity exercise...")
    confounding_df = run_confounding_stress_test(df, reference_ate=reference_ate, seed=cfg.seed)
    print("\nConfounding Stress Test Results:")
    print(confounding_df.to_string())

    confound_path = out_dir / "uplift_confounding_test.parquet"
    confounding_df.to_parquet(confound_path, index=False)
    print(f"\nSaved confounding stress test to {confound_path}")


if __name__ == "__main__":
    main()
