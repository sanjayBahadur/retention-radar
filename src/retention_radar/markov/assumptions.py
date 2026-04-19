"""Statistical assumption tests for Markov chains: order verification and time homogeneity."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def test_first_vs_second_order(
    df: pd.DataFrame,
    states: List[str],
) -> Dict[str, Any]:
    """Test 1st vs 2nd-order Markov chain using likelihood-ratio test and BIC."""
    # Ensure ordered by household and period
    df = df.sort_values(["household_key", "period_id"])
    s_curr = df["state"].values
    s_prev = df["prior_state"].values

    # Compute prior-prior state
    df["prior_2_state"] = df.groupby("household_key")["prior_state"].shift(1)
    s_prev2 = df["prior_2_state"].values

    valid_mask = pd.notna(s_curr) & pd.notna(s_prev) & pd.notna(s_prev2)
    s0 = s_prev2[valid_mask]
    s1 = s_prev[valid_mask]
    s2 = s_curr[valid_mask]
    N = len(s2)

    # 1st-order counts: C1[i, j]
    state_to_idx = {s: i for i, s in enumerate(states)}
    K = len(states)

    C1 = np.zeros((K, K), dtype=float)
    for a, b in zip(s1, s2):
        if a in state_to_idx and b in state_to_idx:
            C1[state_to_idx[a], state_to_idx[b]] += 1.0

    row_sums1 = C1.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P1 = np.where(row_sums1 > 0, C1 / row_sums1, 0.0)
        log_P1 = np.where(P1 > 0, np.log(P1), 0.0)
    ll_1 = float(np.sum(C1 * log_P1))

    # 2nd-order counts: C2[(i, j), k]
    C2 = np.zeros((K, K, K), dtype=float)
    for a, b, c in zip(s0, s1, s2):
        if a in state_to_idx and b in state_to_idx and c in state_to_idx:
            C2[state_to_idx[a], state_to_idx[b], state_to_idx[c]] += 1.0

    pair_sums = C2.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P2 = np.where(pair_sums > 0, C2 / pair_sums, 0.0)
        log_P2 = np.where(P2 > 0, np.log(P2), 0.0)
    ll_2 = float(np.sum(C2 * log_P2))

    # Likelihood ratio test statistic: 2 * (LL2 - LL1)
    lr_stat = max(0.0, 2.0 * (ll_2 - ll_1))
    # Degrees of freedom: non-zero rows in C2 * (K - 1) - non-zero rows in C1 * (K - 1)
    df_count = max(1, (np.sum(pair_sums.squeeze() > 0) - np.sum(row_sums1.squeeze() > 0)) * (K - 1))
    p_value = float(stats.chi2.sf(lr_stat, df=df_count))

    # BICs
    k1 = np.sum(row_sums1 > 0) * (K - 1)
    k2 = np.sum(pair_sums > 0) * (K - 1)
    bic_1 = float(-2.0 * ll_1 + k1 * np.log(N))
    bic_2 = float(-2.0 * ll_2 + k2 * np.log(N))

    return {
        "log_likelihood_1st_order": round(ll_1, 2),
        "log_likelihood_2nd_order": round(ll_2, 2),
        "lr_statistic": round(lr_stat, 2),
        "degrees_of_freedom": int(df_count),
        "p_value": float(p_value),
        "bic_1st_order": round(bic_1, 2),
        "bic_2nd_order": round(bic_2, 2),
        "preferred_model": "1st_order" if bic_1 < bic_2 else "2nd_order",
    }


def test_time_homogeneity(
    df: pd.DataFrame,
    states: List[str],
    split_period: int = 11,
) -> Dict[str, Any]:
    """Test time homogeneity by comparing transitions in first half vs second half of training."""
    state_to_idx = {s: i for i, s in enumerate(states)}
    K = len(states)

    df_h1 = df[df["period_id"] <= split_period]
    df_h2 = df[df["period_id"] > split_period]

    def count_transitions(sub_df: pd.DataFrame) -> np.ndarray:
        C = np.zeros((K, K), dtype=float)
        s_curr = sub_df["state"].values
        s_prev = sub_df["prior_state"].values
        for a, b in zip(s_prev, s_curr):
            if pd.notna(a) and pd.notna(b) and a in state_to_idx and b in state_to_idx:
                C[state_to_idx[a], state_to_idx[b]] += 1.0
        return C

    C1 = count_transitions(df_h1)
    C2 = count_transitions(df_h2)
    C_pooled = C1 + C2

    row_pooled = C_pooled.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_pooled = np.where(row_pooled > 0, C_pooled / row_pooled, 0.0)

    # Anderson-Goodman chi-square test for time homogeneity
    chi2_stat = 0.0
    valid_cells = 0
    for i in range(K):
        for j in range(K):
            if P_pooled[i, j] > 0:
                n1_i = C1[i, :].sum()
                n2_i = C2[i, :].sum()
                if n1_i > 0 and C1[i, j] > 0:
                    p1_ij = C1[i, j] / n1_i
                    chi2_stat += n1_i * ((p1_ij - P_pooled[i, j]) ** 2) / P_pooled[i, j]
                if n2_i > 0 and C2[i, j] > 0:
                    p2_ij = C2[i, j] / n2_i
                    chi2_stat += n2_i * ((p2_ij - P_pooled[i, j]) ** 2) / P_pooled[i, j]
                valid_cells += 1

    dof = max(1, valid_cells - K)
    p_val = float(stats.chi2.sf(chi2_stat, df=dof))

    return {
        "chi2_statistic": round(float(chi2_stat), 2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_val),
        "is_homogeneous_005": bool(p_val >= 0.05),
    }
