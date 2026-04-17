"""Absorbing Markov chain mathematical operations."""

from typing import Dict, Tuple

import numpy as np


def compute_absorbing_metrics(
    Q: np.ndarray,
    R: np.ndarray,
    period_days: int = 28,
) -> Dict[str, np.ndarray]:
    """Compute fundamental matrix N = (I - Q)^(-1), absorption times t, and absorption probabilities B = NR.

    Args:
        Q: (k, k) transient-to-transient transition matrix
        R: (k, r) transient-to-absorbing transition matrix
        period_days: Length of a single period in days

    Returns:
        Dict containing:
            - N: (k, k) fundamental matrix
            - t_periods: (k,) expected periods until absorption
            - t_days: (k,) expected days until absorption
            - B: (k, r) probability of absorption into each absorbing state
    """
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    k = Q.shape[0]

    I_k = np.eye(k, dtype=float)
    I_minus_Q = I_k - Q

    # Fundamental matrix N = (I - Q)^(-1)
    N = np.linalg.inv(I_minus_Q)

    # Expected periods to absorption t = N * 1
    t_periods = N.sum(axis=1)
    t_days = t_periods * period_days

    # Absorption probabilities B = N * R
    B = N @ R

    return {
        "N": N,
        "t_periods": t_periods,
        "t_days": t_days,
        "B": B,
    }


def partition_transition_matrix(
    P: np.ndarray,
    state_names: list[str],
    absorbing_states: list[str],
) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Partition standard transition matrix P into transient block Q and absorbing block R."""
    transient_states = [s for s in state_names if s not in absorbing_states]

    t_idx = [state_names.index(s) for s in transient_states]
    a_idx = [state_names.index(s) for s in absorbing_states]

    Q = P[np.ix_(t_idx, t_idx)]
    R = P[np.ix_(t_idx, a_idx)]

    return Q, R, transient_states, absorbing_states
