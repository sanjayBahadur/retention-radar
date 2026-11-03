"""Observational uplift modeling, propensity weighting, and customer segmentation on Dunnhumby."""

import json
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd
from causalml.inference.meta import BaseDRLearner, BaseSLearner, BaseTLearner, BaseXLearner
from causalml.metrics import auuc_score, qini_score
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from retention_radar.config import get_project_root, load_project_config


def extract_dunnhumby_campaign_units(db_path: Path) -> pd.DataFrame:
    """Extract campaign-household units, pre-window covariates, and campaign window outcomes."""
    con = duckdb.connect(str(db_path), read_only=True)
    # Sum currency as decimal: parallel floating-point sums can change rounded
    # covariates at half-cent boundaries and make model runs differ.
    query = """
    WITH valid_campaigns AS (
        SELECT campaign, campaign_type, start_day, end_day, (end_day - start_day + 1) AS camp_duration
        FROM staging.stg_campaigns
        WHERE campaign IN (5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 29, 30)
    ),
    concurrent_counts AS (
        SELECT c1.campaign, COUNT(c2.campaign) AS concurrent_campaigns
        FROM staging.stg_campaigns c1
        LEFT JOIN staging.stg_campaigns c2 ON c1.campaign != c2.campaign
          AND NOT (c1.end_day < c2.start_day OR c1.start_day > c2.end_day)
        GROUP BY c1.campaign
    ),
    eligible_units AS (
        SELECT
            c.campaign,
            c.campaign_type,
            c.start_day,
            c.end_day,
            c.camp_duration,
            b.household_key,
            CASE WHEN t.household_key IS NOT NULL THEN 1 ELSE 0 END AS treated
        FROM valid_campaigns c
        JOIN marts.fct_baskets b ON b.day BETWEEN c.start_day - 56 AND c.start_day - 1
        LEFT JOIN marts.fct_campaign_exposure t ON c.campaign = t.campaign AND b.household_key = t.household_key
        GROUP BY c.campaign, c.campaign_type, c.start_day, c.end_day, c.camp_duration, b.household_key, t.household_key
    ),
    pre_activity AS (
        SELECT
            u.campaign,
            u.household_key,
            COUNT(b.basket_id) AS pre_trips,
            ROUND(SUM(CAST(b.spend AS DECIMAL(18, 2))) / 56.0, 4) AS pre_spend_per_day,
            ROUND(SUM(CAST(b.total_retail_disc AS DECIMAL(18, 2)) + CAST(b.total_coupon_disc AS DECIMAL(18, 2))) / NULLIF(SUM(CAST(b.gross_spend AS DECIMAL(18, 2))), 0), 4) AS pre_discount_share,
            SUM(b.coupon_used) AS pre_coupon_trips
        FROM eligible_units u
        JOIN marts.fct_baskets b ON u.household_key = b.household_key AND b.day BETWEEN u.start_day - 56 AND u.start_day - 1
        GROUP BY u.campaign, u.household_key
    ),
    camp_activity AS (
        SELECT
            u.campaign,
            u.household_key,
            COUNT(b.basket_id) AS camp_trips,
            ROUND(SUM(CAST(b.spend AS DECIMAL(18, 2))) / u.camp_duration, 4) AS camp_spend_per_day,
            MAX(b.coupon_used) AS camp_coupon_used
        FROM eligible_units u
        JOIN marts.fct_baskets b ON u.household_key = b.household_key AND b.day BETWEEN u.start_day AND u.end_day
        GROUP BY u.campaign, u.household_key, u.camp_duration
    )
    SELECT
        u.campaign,
        u.campaign_type,
        u.start_day,
        u.camp_duration,
        u.household_key,
        u.treated,
        COALESCE(p.pre_trips, 0) AS pre_trips,
        COALESCE(p.pre_spend_per_day, 0.0) AS pre_spend_per_day,
        COALESCE(p.pre_discount_share, 0.0) AS pre_discount_share,
        COALESCE(p.pre_coupon_trips, 0) AS pre_coupon_trips,
        COALESCE(ca.camp_trips, 0) AS camp_trips,
        COALESCE(ca.camp_spend_per_day, 0.0) AS camp_spend_per_day,
        COALESCE(ca.camp_coupon_used, 0) AS camp_coupon_used,
        ROUND(COALESCE(ca.camp_spend_per_day, 0.0) - COALESCE(p.pre_spend_per_day, 0.0), 4) AS delta_spend_per_day,
        d.has_demographics,
        COALESCE(cc.concurrent_campaigns, 0) AS concurrent_campaigns
    FROM eligible_units u
    LEFT JOIN pre_activity p ON u.campaign = p.campaign AND u.household_key = p.household_key
    LEFT JOIN camp_activity ca ON u.campaign = ca.campaign AND u.household_key = ca.household_key
    LEFT JOIN marts.dim_household d ON u.household_key = d.household_key
    LEFT JOIN concurrent_counts cc ON u.campaign = cc.campaign;
    """
    df = con.execute(query).df()
    con.close()
    return df.sort_values(["household_key", "campaign"]).reset_index(drop=True)


def compute_balance_table(
    df: pd.DataFrame,
    covariates: List[str],
    propensity_scores: np.ndarray,
) -> pd.DataFrame:
    """Compute Standardized Mean Differences (SMD) before and after Inverse Propensity Weighting (IPW)."""
    t = df["treated"].values
    e = np.clip(propensity_scores, 0.05, 0.95)
    weights = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))

    records = []
    for col in covariates:
        x = df[col].values
        x_t = x[t == 1]
        x_c = x[t == 0]

        mean_t = np.mean(x_t)
        mean_c = np.mean(x_c)
        pooled_std = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2.0)
        unweighted_smd = np.abs(mean_t - mean_c) / (pooled_std + 1e-9)

        # Weighted means
        w_t = weights[t == 1]
        w_c = weights[t == 0]
        weighted_mean_t = np.sum(x_t * w_t) / np.sum(w_t)
        weighted_mean_c = np.sum(x_c * w_c) / np.sum(w_c)
        weighted_smd = np.abs(weighted_mean_t - weighted_mean_c) / (pooled_std + 1e-9)

        records.append(
            {
                "covariate": col,
                "raw_treated_mean": round(float(mean_t), 4),
                "raw_control_mean": round(float(mean_c), 4),
                "raw_smd": round(float(unweighted_smd), 4),
                "ipw_treated_mean": round(float(weighted_mean_t), 4),
                "ipw_control_mean": round(float(weighted_mean_c), 4),
                "ipw_smd": round(float(weighted_smd), 4),
                "balanced_after_ipw": bool(weighted_smd < 0.10),
            }
        )

    return pd.DataFrame(records)


def compute_targeting_roi_curve(
    test_df: pd.DataFrame,
    cate_preds: np.ndarray,
    offer_cost: float = 2.00,
    gross_margin: float = 0.25,
    scenario_days: int = 35,
) -> pd.DataFrame:
    """Model one offer per unique household; predicted effects are unvalidated assumptions."""
    if test_df.empty or len(test_df) != len(cate_preds) or not np.isfinite(cate_preds).all():
        raise ValueError("Targeting scenarios require finite predictions for a nonempty roster.")
    if test_df["household_key"].duplicated().any():
        raise ValueError("Targeting scenarios require one row per household.")
    if scenario_days <= 0 or offer_cost < 0 or not 0 <= gross_margin <= 1:
        raise ValueError("Invalid scenario duration, offer cost, or margin.")
    df_eval = test_df.copy()
    df_eval["cate"] = cate_preds
    df_sorted = df_eval.sort_values("cate", ascending=False).reset_index(drop=True)
    n = len(df_sorted)

    records = []
    percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    # Blanket baseline (everyone targeted)
    overall_mean_cate = float(df_sorted["cate"].mean())

    for pct in percentiles:
        k = max(1, int(n * pct))
        sub = df_sorted.iloc[:k]
        targeted_mean_cate = float(sub["cate"].mean())

        incremental_spend = targeted_mean_cate * scenario_days * k
        incremental_margin = incremental_spend * gross_margin
        total_offer_cost = offer_cost * k
        net_profit = incremental_margin - total_offer_cost
        roi = net_profit / total_offer_cost if total_offer_cost > 0 else 0.0

        # Random targeting comparison
        random_spend = overall_mean_cate * scenario_days * k
        random_margin = random_spend * gross_margin
        random_net = random_margin - total_offer_cost

        records.append(
            {
                "targeting_percentile": int(pct * 100),
                "scenario_days": scenario_days,
                "households_targeted": k,
                "mean_uplift_spend_per_day": round(targeted_mean_cate, 4),
                "incremental_margin": round(incremental_margin, 2),
                "offer_cost": round(total_offer_cost, 2),
                "net_profit": round(net_profit, 2),
                "roi_ratio": round(roi, 3),
                "random_net_profit": round(random_net, 2),
            }
        )

    return pd.DataFrame(records)


def segment_customers(
    df: pd.DataFrame,
    cate_preds: np.ndarray,
    clv_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign exploratory score bands, not proven individual treatment-response types."""
    if df["household_key"].duplicated().any():
        raise ValueError("Customer segments require one row per household.")
    df_seg = df.copy()
    df_seg["uplift_tau"] = cate_preds

    # Join CLV
    if "clv_26w" in clv_df.columns:
        df_seg = df_seg.merge(
            clv_df[["household_key", "clv_26w"]].drop_duplicates("household_key"),
            on="household_key",
            how="left",
        )
    else:
        df_seg["clv_26w"] = np.nan

    # Segmentation rules:
    # Persuadables: positive uplift (tau > 0.15)
    # Sleeping Dogs: negative uplift (tau < -0.15)
    # Sure Things: tau near 0, but high baseline spend (pre_spend_per_day >= median)
    # Lost Causes: tau near 0, and low baseline spend (pre_spend_per_day < median)
    median_spend = df_seg["pre_spend_per_day"].median()

    conditions = [
        (df_seg["uplift_tau"] > 0.15),
        (df_seg["uplift_tau"] < -0.15),
        (df_seg["pre_spend_per_day"] >= median_spend),
    ]
    choices = [
        "Persuadables",
        "Sleeping Dogs",
        "Sure Things",
    ]
    df_seg["segment"] = np.select(conditions, choices, default="Lost Causes")

    # Segment summary
    summary_records = []
    total_hh = len(df_seg)
    for seg_name, grp in df_seg.groupby("segment"):
        summary_records.append(
            {
                "segment": seg_name,
                "household_count": len(grp),
                "segment_share": round(float(len(grp) / total_hh), 4),
                "mean_uplift_spend_per_day": round(float(grp["uplift_tau"].mean()), 4),
                "avg_26w_clv": round(float(grp["clv_26w"].mean()), 2),
                "pre_spend_per_day": round(float(grp["pre_spend_per_day"].mean()), 2),
            }
        )

    seg_summary_df = (
        pd.DataFrame(summary_records)
        .sort_values("household_count", ascending=False)
        .reset_index(drop=True)
    )
    return df_seg, seg_summary_df


def evaluate_grouped_learners(
    df: pd.DataFrame,
    feature_columns: List[str],
    seed: int = 42,
    n_splits: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every household in a held-out fold, including train-only propensity fits.

    Qini and treated/control contrasts remain descriptive on observational data.
    Household grouping avoids sharing the same household across train and test;
    it does not remove campaign confounding or establish causal identification.
    """
    learner_classes = {
        "S-Learner": BaseSLearner,
        "T-Learner": BaseTLearner,
        "X-Learner": BaseXLearner,
        "DR-Learner": BaseDRLearner,
    }
    scored = df.copy().reset_index(drop=True)
    scored[feature_columns] = scored[feature_columns].fillna(0.0)
    scored["propensity_score"] = np.nan
    scored["evaluation_fold"] = -1
    for name in learner_classes:
        scored[name] = np.nan

    groups = scored["household_key"]
    folds = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(folds.split(scored, groups=groups)):
        train = scored.iloc[train_idx]
        test = scored.iloc[test_idx]
        propensity = LogisticRegression(max_iter=2000, random_state=seed)
        propensity.fit(train[feature_columns], train["treated"])
        p_train = propensity.predict_proba(train[feature_columns])[:, 1]
        p_test = propensity.predict_proba(test[feature_columns])[:, 1]
        train_overlap = (p_train >= 0.05) & (p_train <= 0.95)
        test_overlap = (p_test >= 0.05) & (p_test <= 0.95)
        training = train.loc[train_overlap]
        held_out = test.loc[test_overlap]
        scored.loc[test_idx, "propensity_score"] = p_test
        scored.loc[test_idx, "evaluation_fold"] = fold
        if training["treated"].nunique() != 2 or held_out.empty:
            raise ValueError("Each fold needs treated and control training data and test overlap.")

        for name, learner_class in learner_classes.items():
            # DR-Learner also uses NumPy for internal sample splits.
            np.random.seed(seed + fold)
            model = learner_class(
                learner=LGBMRegressor(n_estimators=60, random_state=seed, verbose=-1, n_jobs=1)
            )
            fit_args = {
                "X": training[feature_columns].to_numpy(),
                "treatment": training["treated"].to_numpy(),
                "y": training["delta_spend_per_day"].to_numpy(),
            }
            predict_args = {"X": held_out[feature_columns].to_numpy()}
            if name in ("X-Learner", "DR-Learner"):
                fit_args["p"] = p_train[train_overlap]
                predict_args["p"] = p_test[test_overlap]
            model.fit(**fit_args)
            scored.loc[held_out.index, name] = model.predict(**predict_args).ravel()
        print(f"Fold {fold + 1}/{n_splits}: scored {len(held_out):,} held-out observations.")

    overlap = scored.dropna(subset=list(learner_classes)).copy()
    rows = []
    for name in learner_classes:
        cate = overlap[name].to_numpy()
        metric_data = pd.DataFrame(
            {"y": overlap["delta_spend_per_day"], "w": overlap["treated"], "pred": cate}
        )
        # CausalML creates random reference rankings; seed them for repeatable scores.
        np.random.seed(seed)
        qini = float(qini_score(metric_data)["pred"])
        np.random.seed(seed)
        auuc = float(auuc_score(metric_data)["pred"])
        ranked = overlap.sort_values(name, ascending=False)
        contrasts = {}
        for fraction in (0.10, 0.30):
            top = ranked.head(max(1, int(len(ranked) * fraction)))
            treated = top.loc[top["treated"] == 1, "delta_spend_per_day"]
            control = top.loc[top["treated"] == 0, "delta_spend_per_day"]
            contrasts[f"uplift_at_{int(fraction * 100)}pct"] = float(
                treated.mean() - control.mean()
            )
        rows.append(
            {
                "learner": name,
                "qini": round(qini, 4),
                "auuc": round(auuc, 4),
                **{key: round(value, 4) for key, value in contrasts.items()},
                "mean_cate": round(float(cate.mean()), 4),
                "fold_count": n_splits,
                "observation_count": len(overlap),
                "unique_households": overlap["household_key"].nunique(),
            }
        )
    evaluation = pd.DataFrame(rows).sort_values("qini", ascending=False).reset_index(drop=True)
    return evaluation, overlap, scored


def latest_household_scores(observations: pd.DataFrame) -> pd.DataFrame:
    """Use the latest evaluated campaign context once per household for a planning roster."""
    return (
        observations.sort_values(["start_day", "campaign"])
        .drop_duplicates("household_key", keep="last")
        .sort_values("household_key")
        .reset_index(drop=True)
    )


def main() -> None:
    """Build descriptive uplift comparisons and clearly labeled planning scenarios."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = extract_dunnhumby_campaign_units(project_root / cfg.paths.warehouse_db)
    covariates = [
        "pre_trips",
        "pre_spend_per_day",
        "pre_discount_share",
        "pre_coupon_trips",
        "has_demographics",
        "concurrent_campaigns",
    ]
    encoded = pd.get_dummies(raw_df, columns=["campaign_type"], drop_first=True, dtype=float)
    features = covariates + [c for c in encoded if c.startswith("campaign_type_")]
    evaluation, observations, all_scored = evaluate_grouped_learners(encoded, features, cfg.seed)
    selected_learner = str(evaluation.iloc[0]["learner"])
    observations["uplift_tau"] = observations[selected_learner]
    # Keep the original campaign label alongside the encoded covariates.
    observations["campaign_type"] = raw_df.loc[observations.index, "campaign_type"]
    observations.to_parquet(out_dir / "uplift_observations.parquet", index=False)
    evaluation.to_parquet(out_dir / "uplift_eval.parquet", index=False)
    balance = compute_balance_table(
        observations, covariates, observations["propensity_score"].values
    )
    balance.to_parquet(out_dir / "uplift_balance.parquet", index=False)

    households = latest_household_scores(observations)
    clv = pd.read_parquet(out_dir / "clv.parquet")
    scores, segments = segment_customers(households, households["uplift_tau"].values, clv)
    scores.to_parquet(out_dir / "uplift_scores.parquet", index=False)
    segments.to_parquet(out_dir / "uplift_segments.parquet", index=False)
    roi = compute_targeting_roi_curve(
        households, households["uplift_tau"].values, cfg.offer_cost, cfg.gross_margin
    )
    roi.to_parquet(out_dir / "uplift_roi.parquet", index=False)
    metadata = {
        "selected_learner": selected_learner,
        "selection_method": "Highest pooled household-held-out Qini; exploratory model comparison",
        "evaluation_folds": 3,
        "eligible_observations": len(all_scored),
        "overlap_observations": len(observations),
        "unique_households": len(households),
        "observation_unit": "One household in one historical campaign",
        "roster_unit": "One household in its latest evaluated campaign context",
        "scenario_days": 35,
        "offer_cost_dollars": cfg.offer_cost,
        "gross_margin_assumption": cfg.gross_margin,
        "limitations": [
            "Historical campaign assignment was not randomized; unmeasured confounding remains.",
            "Household-held-out evaluation is not a future-campaign validation.",
            "Model selection and reported ranking use the same cross-validation; no final untouched test.",
            "Qini and top-segment treated/control contrasts are descriptive, not causal estimates.",
            "Net profit uses predicted effects, a 35-day window and assumed margin; no realized savings.",
            "Score bands do not establish that an individual benefits from or is harmed by contact.",
            "The latest historical contexts differ by household; this is a retrospective planning roster.",
        ],
    }
    (out_dir / "uplift_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(evaluation.to_string(index=False))
    print(segments.to_string(index=False))
    print(f"Saved {len(observations):,} observations and {len(households):,} unique households.")


if __name__ == "__main__":
    main()
