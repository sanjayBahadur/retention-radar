"""Core Markov chain lifecycle modeling, bootstrap inference, and reward process."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from retention_radar.config import get_project_root, load_project_config, load_states_config
from retention_radar.markov.absorbing import compute_absorbing_metrics, partition_transition_matrix
from retention_radar.markov.assumptions import test_first_vs_second_order, test_time_homogeneity
from retention_radar.markov.simulate import precompute_scenario_grid
from retention_radar.survival.km_cox import fit_survival_models


def estimate_transition_matrix(
    df: pd.DataFrame,
    states: List[str],
    smoothing_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate empirical transition matrix P with additive smoothing on structural cells only."""
    K = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((K, K), dtype=float)

    for _, row in df.iterrows():
        src = row["state"]
        dst = row["next_state"]
        if pd.notna(src) and pd.notna(dst) and src in state_to_idx and dst in state_to_idx:
            C[state_to_idx[src], state_to_idx[dst]] += 1.0

    # Structural mask: cells that have observed transitions or are structurally permitted
    structural_mask = C > 0

    # Additive smoothing on structural cells only
    C_smoothed = np.copy(C)
    C_smoothed[structural_mask] += smoothing_alpha

    row_sums = C_smoothed.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(row_sums > 0, C_smoothed / row_sums, 0.0)

    return P, C


def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute ergodic stationary distribution pi where pi * P = pi and sum(pi) = 1."""
    K = P.shape[0]
    # Solve (P.T - I) pi = 0 subject to sum(pi) = 1
    A = P.T - np.eye(K)
    A[-1] = np.ones(K)
    b = np.zeros(K)
    b[-1] = 1.0
    try:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.maximum(0.0, pi)
        return pi / pi.sum()
    except Exception:
        # Fallback to power iteration
        pi = np.ones(K) / K
        for _ in range(200):
            pi = pi @ P
        return pi / pi.sum()


def compute_markov_rewards(
    df: pd.DataFrame,
    states: List[str],
    gross_margin: float = 0.25,
) -> np.ndarray:
    """Compute mean gross margin reward r_s per household-period for each state."""
    rewards = np.zeros(len(states), dtype=float)
    for i, s in enumerate(states):
        sub = df[df["state"] == s]
        if len(sub) > 0:
            mean_spend = float(sub["spend"].mean())
            rewards[i] = mean_spend * gross_margin
    return rewards


def run_household_bootstrap(
    df: pd.DataFrame,
    states: List[str],
    state_rewards: np.ndarray,
    n_reps: int = 1000,
    smoothing_alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform household-level block bootstrap to construct 95% CIs for transitions and metrics."""
    rng = np.random.default_rng(seed)
    households = df["household_key"].unique()
    n_hh = len(households)
    K = len(states)

    P_samples = np.zeros((n_reps, K, K), dtype=float)
    t_days_samples = []

    # Map each household to its transitions tensor for ultrafast bootstrap
    state_to_idx = {s: i for i, s in enumerate(states)}
    valid = df[df["state"].isin(state_to_idx) & df["next_state"].isin(state_to_idx)].copy()
    src_idx = valid["state"].map(state_to_idx).to_numpy()
    dst_idx = valid["next_state"].map(state_to_idx).to_numpy()
    hh_idx_map = {hh: idx for idx, hh in enumerate(households)}
    h_idx = valid["household_key"].map(hh_idx_map).to_numpy()

    hh_counts = np.zeros((n_hh, K, K), dtype=float)
    np.add.at(hh_counts, (h_idx, src_idx, dst_idx), 1.0)

    transient_states = [s for s in states if s != "Lapsed"]
    t_idx = [states.index(s) for s in transient_states]
    lapsed_idx = [states.index("Lapsed")]

    for rep in range(n_reps):
        sample_indices = rng.integers(0, n_hh, size=n_hh)
        weights = np.bincount(sample_indices, minlength=n_hh)
        C_rep = np.tensordot(weights, hh_counts, axes=(0, 0))

        structural_mask = C_rep > 0
        C_smoothed = np.copy(C_rep)
        C_smoothed[structural_mask] += smoothing_alpha
        row_sums = C_smoothed.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            P_rep = np.where(row_sums > 0, C_smoothed / row_sums, 0.0)
        P_samples[rep] = P_rep

        # Absorbing metrics on transient block
        Q_rep = P_rep[np.ix_(t_idx, t_idx)]
        R_rep = P_rep[np.ix_(t_idx, lapsed_idx)]
        try:
            abs_res = compute_absorbing_metrics(Q_rep, R_rep, period_days=28)
            t_days_samples.append(abs_res["t_days"])
        except Exception:
            pass

    P_lower = np.percentile(P_samples, 2.5, axis=0)
    P_upper = np.percentile(P_samples, 97.5, axis=0)

    t_days_arr = (
        np.array(t_days_samples) if t_days_samples else np.zeros((1, len(transient_states)))
    )
    t_lower = np.percentile(t_days_arr, 2.5, axis=0)
    t_upper = np.percentile(t_days_arr, 97.5, axis=0)

    return {
        "P_lower": P_lower,
        "P_upper": P_upper,
        "t_days_lower": {s: float(t_lower[i]) for i, s in enumerate(transient_states)},
        "t_days_upper": {s: float(t_upper[i]) for i, s in enumerate(transient_states)},
    }


def backtest_markov_projections(
    con: duckdb.DuckDBPyConnection,
    P: np.ndarray,
    states: List[str],
    base_period: int = 19,
    test_periods: List[int] = None,
) -> pd.DataFrame:
    """Project state distributions from period 19 through periods 20–25 and compare with persistence baseline."""
    if test_periods is None:
        test_periods = [20, 21, 22, 23, 24, 25]

    K = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Period 19 observed distribution
    p19_df = con.execute(f"""
        SELECT state, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pct
        FROM marts.fct_household_period_state
        WHERE period_id = {base_period}
        GROUP BY state;
    """).df()

    p19_dist = np.zeros(K, dtype=float)
    for _, r in p19_df.iterrows():
        p19_dist[state_to_idx[r["state"]]] = float(r["pct"])

    records = []
    proj_dist = np.copy(p19_dist)

    for k, p_id in enumerate(test_periods, start=1):
        proj_dist = proj_dist @ P

        actual_df = con.execute(f"""
            SELECT state, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pct
            FROM marts.fct_household_period_state
            WHERE period_id = {p_id}
            GROUP BY state;
        """).df()
        actual_dist = np.zeros(K, dtype=float)
        for _, r in actual_df.iterrows():
            actual_dist[state_to_idx[r["state"]]] = float(r["pct"])

        # Total Variation Distance (TVD) = 0.5 * sum(|p - q|)
        markov_tvd = float(0.5 * np.sum(np.abs(proj_dist - actual_dist)))
        persistence_tvd = float(0.5 * np.sum(np.abs(p19_dist - actual_dist)))

        records.append(
            {
                "k_steps": k,
                "period_id": p_id,
                "markov_tvd": round(markov_tvd, 4),
                "persistence_tvd": round(persistence_tvd, 4),
                "markov_beats_baseline": bool(markov_tvd < persistence_tvd),
            }
        )

    return pd.DataFrame(records)


def run_markov_pipeline(
    db_path: Path,
    output_dir: Path,
    bootstrap_reps: int = 1000,
    gross_margin: float = 0.25,
    annual_discount_rate: float = 0.10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete Phase 4 lifecycle modeling workflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    states_cfg = load_states_config()
    states = (
        states_cfg.state_names
    )  # ['New', 'Champion', 'Regular', 'At-Risk', 'Dormant', 'Lapsed']

    con = duckdb.connect(str(db_path), read_only=True)
    train_df = con.execute("""
        SELECT household_key, period_id, state, prior_state,
               CASE WHEN period_id < 19 THEN next_state END AS next_state, spend, trips
        FROM marts.fct_household_period_state
        WHERE period_id BETWEEN 4 AND 19;
    """).df()

    # 1. Estimate Transition Matrix
    P, C = estimate_transition_matrix(train_df, states, smoothing_alpha=0.5)

    # 2. Absorbing Chain Analysis (Lapsed as absorbing)
    Q, R, transient_states, absorbing_states = partition_transition_matrix(P, states, ["Lapsed"])
    # Enforce strict absorbing row for Lapsed in absorbing variant
    P_absorbing = np.copy(P)
    lapsed_idx = states.index("Lapsed")
    P_absorbing[lapsed_idx, :] = 0.0
    P_absorbing[lapsed_idx, lapsed_idx] = 1.0

    abs_metrics = compute_absorbing_metrics(Q, R, period_days=28)

    # 3. Ergodic Analysis (Empirical return from Lapsed)
    pi_ergodic = compute_stationary_distribution(P)

    # Period 25 distribution for comparison
    p25_df = con.execute("""
        SELECT state, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pct
        FROM marts.fct_household_period_state
        WHERE period_id = 25
        GROUP BY state;
    """).df()
    state_to_idx = {s: i for i, s in enumerate(states)}
    obs_p25 = np.zeros(len(states))
    for _, r in p25_df.iterrows():
        obs_p25[state_to_idx[r["state"]]] = float(r["pct"])

    # 4. Markov Reward Process: V = (I - gamma * Q)^(-1) * r
    r = compute_markov_rewards(train_df, states, gross_margin=gross_margin)
    gamma = float((1.0 + annual_discount_rate) ** (-28.0 / 365.0))
    t_idx = [states.index(s) for s in transient_states]
    r_transient = r[t_idx]
    I_minus_gamma_Q = np.eye(len(transient_states)) - gamma * Q
    V_transient = np.linalg.inv(I_minus_gamma_Q) @ r_transient

    state_clv = {s: round(float(V_transient[i]), 2) for i, s in enumerate(transient_states)}
    state_clv["Lapsed"] = 0.0

    # 5. Bootstrap Uncertainty CIs
    ci_results = run_household_bootstrap(
        train_df, states, r, n_reps=bootstrap_reps, smoothing_alpha=0.5, seed=seed
    )

    # Export markov_P.parquet with CIs
    records_P = []
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            records_P.append(
                {
                    "source_state": src,
                    "target_state": dst,
                    "transition_probability": round(float(P[i, j]), 4),
                    "ci_lower_95": round(float(ci_results["P_lower"][i, j]), 4),
                    "ci_upper_95": round(float(ci_results["P_upper"][i, j]), 4),
                    "transition_count": int(C[i, j]),
                }
            )
    df_P = pd.DataFrame(records_P)
    df_P.to_parquet(output_dir / "markov_P.parquet", index=False)

    # 6. Assumption Tests
    order_test = test_first_vs_second_order(train_df, states)
    homog_test = test_time_homogeneity(train_df, states, split_period=11)

    # 7. Backtest vs Persistence Baseline
    backtest_df = backtest_markov_projections(con, P, states, base_period=19)
    backtest_df.to_parquet(output_dir / "markov_backtest.parquet", index=False)

    # 8. Scenario Simulator Grid
    precompute_scenario_grid(
        P=P,
        state_names=states,
        state_rewards=r,
        initial_dist=pi_ergodic,
        output_parquet=output_dir / "scenarios.parquet",
    )

    # 9. Survival Analysis Benchmark
    surv_results = fit_survival_models(
        db_path=db_path,
        output_parquet=output_dir / "survival_curves.parquet",
    )

    con.close()

    # Compile Summary Metrics JSON
    metrics_summary = {
        "transient_states": transient_states,
        "absorbing_states": absorbing_states,
        "discount_factor_gamma": round(gamma, 4),
        "expected_days_to_lapse": {
            s: {
                "mean_days": round(float(abs_metrics["t_days"][i]), 1),
                "ci_lower_95": round(ci_results["t_days_lower"][s], 1),
                "ci_upper_95": round(ci_results["t_days_upper"][s], 1),
            }
            for i, s in enumerate(transient_states)
        },
        "markov_reward_clv": state_clv,
        "stationary_distribution": {
            s: round(float(pi_ergodic[i]), 4) for i, s in enumerate(states)
        },
        "period_25_observed_distribution": {
            s: round(float(obs_p25[i]), 4) for i, s in enumerate(states)
        },
        "assumption_tests": {
            "order_test": order_test,
            "homogeneity_test": homog_test,
        },
        "survival_benchmark": surv_results,
        "backtest_avg_markov_tvd": round(float(backtest_df["markov_tvd"].mean()), 4),
        "backtest_avg_persistence_tvd": round(float(backtest_df["persistence_tvd"].mean()), 4),
    }

    with open(output_dir / "markov_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, allow_nan=False)

    return metrics_summary


def main() -> None:
    """CLI entrypoint for Phase 4 Markov modeling."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    output_dir = root / cfg.paths.data_outputs

    metrics = run_markov_pipeline(
        db_path=db_path,
        output_dir=output_dir,
        bootstrap_reps=cfg.bootstrap_reps,
        gross_margin=cfg.gross_margin,
        annual_discount_rate=cfg.annual_discount_rate,
        seed=cfg.seed,
    )
    print("Phase 4 Markov Modeling Complete:")
    print("  Expected Days to Lapse (At-Risk):", metrics["expected_days_to_lapse"]["At-Risk"])
    print("  State CLV Values:", metrics["markov_reward_clv"])
    print(
        "  Markov vs Persistence TVD:",
        metrics["backtest_avg_markov_tvd"],
        "vs",
        metrics["backtest_avg_persistence_tvd"],
    )
    print("  Assumption Tests:", metrics["assumption_tests"])


if __name__ == "__main__":
    main()
