"""Markov chain campaign attribution using absorbing removal effects."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from retention_radar.attribution.heuristics import compute_heuristic_attribution
from retention_radar.config import get_project_root, load_project_config


def compute_removal_effects(
    paths: List[Dict[str, Any]],
    channels: List[str],
) -> Dict[str, float]:
    """Compute Markov removal effect attribution shares across customer touchpoint journeys."""
    states = ["START"] + channels + ["CONVERSION", "NULL"]
    state_to_idx = {s: i for i, s in enumerate(states)}
    K = len(states)

    # Transition count matrix
    C = np.zeros((K, K), dtype=float)
    for p in paths:
        route = ["START"] + p["path"] + (["CONVERSION"] if p["converted"] else ["NULL"])
        weight = p.get("weight", 1.0)
        for i in range(len(route) - 1):
            s1, s2 = route[i], route[i + 1]
            if s1 in state_to_idx and s2 in state_to_idx:
                C[state_to_idx[s1], state_to_idx[s2]] += weight

    # Row normalize to transition matrix P
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(row_sums > 0, C / row_sums, 0.0)
    # Absorbing terminal states: CONVERSION and NULL
    conv_idx = state_to_idx["CONVERSION"]
    null_idx = state_to_idx["NULL"]
    P[conv_idx, :] = 0.0
    P[conv_idx, conv_idx] = 1.0
    P[null_idx, :] = 0.0
    P[null_idx, null_idx] = 1.0

    def calc_p_conv(mat: np.ndarray) -> float:
        # Transient states: START + channels
        t_states = ["START"] + channels
        t_idx = [state_to_idx[s] for s in t_states]
        Q = mat[np.ix_(t_idx, t_idx)]
        R = mat[np.ix_(t_idx, [conv_idx, null_idx])]
        I_mat = np.eye(len(t_states))
        try:
            N = np.linalg.inv(I_mat - Q)
            B = N @ R
            # B[0, 0] is probability from START (index 0) to CONVERSION (index 0 of R)
            return float(B[0, 0])
        except Exception:
            return 0.0

    base_p_conv = calc_p_conv(P)
    if base_p_conv <= 0:
        return {c: 1.0 / len(channels) for c in channels}

    removal_effects: Dict[str, float] = {}
    for ch in channels:
        P_sub = np.copy(P)
        ch_idx = state_to_idx[ch]
        # Removing channel: route any transition into ch to NULL instead
        for row in range(K):
            if P_sub[row, ch_idx] > 0:
                P_sub[row, null_idx] += P_sub[row, ch_idx]
                P_sub[row, ch_idx] = 0.0

        p_conv_sub = calc_p_conv(P_sub)
        re = max(0.0, 1.0 - (p_conv_sub / base_p_conv))
        removal_effects[ch] = re

    total_re = sum(removal_effects.values())
    if total_re == 0:
        return {c: 1.0 / len(channels) for c in channels}

    return {c: round(v / total_re, 4) for c, v in removal_effects.items()}


def extract_customer_journeys(
    db_path: Path,
    use_only_coupon_redempt: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract touchpoint journeys from warehouse exposures and redemptions."""
    con = duckdb.connect(str(db_path), read_only=True)

    # Campaign targeting exposures
    exposures_df = con.execute("""
        SELECT
            household_key,
            campaign_type,
            start_day
        FROM marts.fct_campaign_exposure
        ORDER BY household_key, start_day;
    """).df()

    # Conversions: either all coupon transactions or only coupon_redempt
    if use_only_coupon_redempt:
        conv_df = con.execute("""
            SELECT DISTINCT household_key, day
            FROM staging.stg_redemptions;
        """).df()
    else:
        conv_df = con.execute("""
            SELECT DISTINCT household_key, day
            FROM intermediate.int_coupon_transactions;
        """).df()

    con.close()

    conv_days_by_hh = {}
    for _, r in conv_df.iterrows():
        hh = r["household_key"]
        conv_days_by_hh.setdefault(hh, []).append(r["day"])

    channels = sorted(exposures_df["campaign_type"].unique().tolist())

    paths = []
    for hh, group in exposures_df.groupby("household_key"):
        seq = group["campaign_type"].tolist()
        last_day = group["start_day"].max()
        hh_conv_days = conv_days_by_hh.get(hh, [])

        # Check if conversion occurred during or after campaign exposure (within 90 days lookback)
        converted = any(
            c_day >= group["start_day"].min() and c_day <= last_day + 90 for c_day in hh_conv_days
        )

        paths.append(
            {
                "household_key": hh,
                "path": seq,
                "converted": converted,
                "weight": 1.0,
            }
        )

    return paths, channels


def run_attribution_pipeline(
    db_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute campaign attribution comparing Markov removal effects with heuristics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary Source: int_coupon_transactions
    paths_primary, channels = extract_customer_journeys(db_path, use_only_coupon_redempt=False)
    markov_shares = compute_removal_effects(paths_primary, channels)
    heuristic_shares = compute_heuristic_attribution(paths_primary, channels)

    # 2. Sensitivity Source: coupon_redempt only
    paths_sens, _ = extract_customer_journeys(db_path, use_only_coupon_redempt=True)
    markov_sens = compute_removal_effects(paths_sens, channels)

    # Format output DataFrame
    records = []
    for ch in channels:
        records.append(
            {
                "campaign_type": ch,
                "markov_removal_effect_share": markov_shares[ch],
                "first_touch_share": heuristic_shares["first_touch"][ch],
                "last_touch_share": heuristic_shares["last_touch"][ch],
                "linear_share": heuristic_shares["linear"][ch],
                "markov_redempt_sensitivity_share": markov_sens[ch],
            }
        )

    df_attr = pd.DataFrame(records)
    # Ensure shares sum exactly to 1.0
    for col in [
        "markov_removal_effect_share",
        "first_touch_share",
        "last_touch_share",
        "linear_share",
        "markov_redempt_sensitivity_share",
    ]:
        s = df_attr[col].sum()
        if s > 0:
            df_attr[col] = (df_attr[col] / s).round(4)

    df_attr.to_parquet(output_dir / "attribution.parquet", index=False)

    summary = {
        "channels": channels,
        "total_journeys_evaluated": len(paths_primary),
        "primary_conversions_count": sum(1 for p in paths_primary if p["converted"]),
        "sensitivity_conversions_count": sum(1 for p in paths_sens if p["converted"]),
        "models": {
            "markov_removal_effects": markov_shares,
            "first_touch": heuristic_shares["first_touch"],
            "last_touch": heuristic_shares["last_touch"],
            "linear": heuristic_shares["linear"],
            "sensitivity_coupon_redempt": markov_sens,
        },
        "caveat": "Targeting exposures reflect distributed mailers, not confirmed physical opening; attribution represents descriptive budget allocation rather than counterfactual uplift.",
    }

    with open(output_dir / "attribution_paths_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """CLI entrypoint for Phase 5 attribution modeling."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    output_dir = root / cfg.paths.data_outputs

    summary = run_attribution_pipeline(db_path=db_path, output_dir=output_dir)
    print("Phase 5 Attribution Modeling Complete:")
    print("  Channels:", summary["channels"])
    print("  Models Comparison:")
    for m_name, shares in summary["models"].items():
        print(f"    - {m_name}: {shares}")


if __name__ == "__main__":
    main()
