"""Deterministic scenario simulation engine for Markov chain retention interventions."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def apply_intervention(
    P: np.ndarray,
    state_names: List[str],
    intervention: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Apply targeted probability shifts to transition matrix P.

    Added probability mass to target cells is subtracted proportionally from
    all other structurally non-zero, non-target cells in the same row.
    """
    P_mod = np.copy(P).astype(float)
    state_to_idx = {s: i for i, s in enumerate(state_names)}

    for src_state, targets in intervention.items():
        if src_state not in state_to_idx:
            continue
        row_idx = state_to_idx[src_state]

        for tgt_state, delta in targets.items():
            if tgt_state not in state_to_idx or abs(delta) < 1e-9:
                continue
            col_idx = state_to_idx[tgt_state]

            # Current mass and available mass in other non-target cells
            other_indices = [
                j for j in range(len(state_names)) if j != col_idx and P_mod[row_idx, j] > 0
            ]
            other_mass = sum(P_mod[row_idx, j] for j in other_indices)

            if other_mass <= 0:
                continue

            # Ensure we don't subtract more than available
            actual_delta = (
                min(delta, other_mass) if delta > 0 else max(delta, -P_mod[row_idx, col_idx])
            )
            P_mod[row_idx, col_idx] += actual_delta

            # Subtract proportionally from other non-target cells
            for j in other_indices:
                proportion = P_mod[row_idx, j] / other_mass
                P_mod[row_idx, j] -= actual_delta * proportion

        # Re-normalize row to exactly 1.0
        row_sum = P_mod[row_idx].sum()
        if row_sum > 0:
            P_mod[row_idx] = P_mod[row_idx] / row_sum

    return P_mod


def simulate_retention_scenario(
    P: np.ndarray,
    state_names: List[str],
    state_rewards: np.ndarray,
    initial_distribution: Optional[np.ndarray] = None,
    intervention: Optional[Dict[str, Dict[str, float]]] = None,
    horizon_periods: int = 13,
    population: int = 100_000,
    lapsed_state_name: str = "Lapsed",
) -> Dict[str, float]:
    """Simulate Markov trajectory under baseline vs intervention over a given horizon.

    Returns deterministic metrics per population (default 100,000 customers).
    """
    K = len(state_names)
    if initial_distribution is None:
        init_dist = np.ones(K) / K
    else:
        init_dist = np.asarray(initial_distribution, dtype=float)
        init_dist = init_dist / init_dist.sum()

    rewards = np.asarray(state_rewards, dtype=float)
    lapsed_idx = state_names.index(lapsed_state_name) if lapsed_state_name in state_names else -1

    # 1. Baseline projection
    curr_base = np.copy(init_dist)
    total_margin_base = 0.0

    for _ in range(horizon_periods):
        total_margin_base += float(np.dot(curr_base, rewards))
        curr_base = curr_base @ P

    lapsed_base = float(curr_base[lapsed_idx] * population) if lapsed_idx >= 0 else 0.0

    # 2. Intervened projection
    if not intervention:
        return {
            "baseline_margin_per_hh": round(total_margin_base, 2),
            "intervened_margin_per_hh": round(total_margin_base, 2),
            "delta_margin_per_hh": 0.0,
            "delta_margin_dollars": 0.0,
            "delta_lapsed_households": 0.0,
            "baseline_lapsed_households": round(lapsed_base, 1),
            "intervened_lapsed_households": round(lapsed_base, 1),
        }

    P_int = apply_intervention(P, state_names, intervention)
    curr_int = np.copy(init_dist)
    total_margin_int = 0.0

    for _ in range(horizon_periods):
        total_margin_int += float(np.dot(curr_int, rewards))
        curr_int = curr_int @ P_int

    lapsed_int = float(curr_int[lapsed_idx] * population) if lapsed_idx >= 0 else 0.0

    delta_margin_per_hh = total_margin_int - total_margin_base
    delta_margin_dollars = delta_margin_per_hh * population
    delta_lapsed = lapsed_int - lapsed_base

    return {
        "baseline_margin_per_hh": round(total_margin_base, 2),
        "intervened_margin_per_hh": round(total_margin_int, 2),
        "delta_margin_per_hh": round(delta_margin_per_hh, 2),
        "delta_margin_dollars": round(delta_margin_dollars, 2),
        "delta_lapsed_households": round(delta_lapsed, 1),
        "baseline_lapsed_households": round(lapsed_base, 1),
        "intervened_lapsed_households": round(lapsed_int, 1),
    }


def precompute_scenario_grid(
    P: np.ndarray,
    state_names: List[str],
    state_rewards: np.ndarray,
    initial_dist: np.ndarray,
    output_parquet: Path,
) -> pd.DataFrame:
    """Precompute a standard grid of retention lift scenarios for dashboard caching."""
    records = []
    # Test varying lift from At-Risk -> Regular (+1% to +10%)
    for lift in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        # Test varying lift from Dormant -> Regular (+1% to +5%)
        for d_lift in [0.0, 0.02, 0.05]:
            intervention = {}
            if lift > 0:
                intervention["At-Risk"] = {"Regular": lift}
            if d_lift > 0:
                intervention["Dormant"] = {"Regular": d_lift}

            res = simulate_retention_scenario(
                P=P,
                state_names=state_names,
                state_rewards=state_rewards,
                initial_distribution=initial_dist,
                intervention=intervention if intervention else None,
                horizon_periods=13,
                population=100_000,
            )
            records.append(
                {
                    "at_risk_to_regular_lift": lift,
                    "dormant_to_regular_lift": d_lift,
                    "delta_margin_dollars": res["delta_margin_dollars"],
                    "delta_lapsed_households": res["delta_lapsed_households"],
                    "baseline_margin_per_hh": res["baseline_margin_per_hh"],
                    "intervened_margin_per_hh": res["intervened_margin_per_hh"],
                }
            )

    df = pd.DataFrame(records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df
