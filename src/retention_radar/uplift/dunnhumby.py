"""Observational uplift modeling, propensity weighting, and customer segmentation on Dunnhumby."""

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
            ROUND(SUM(b.spend) / 56.0, 4) AS pre_spend_per_day,
            ROUND(SUM(b.total_retail_disc + b.total_coupon_disc) / NULLIF(SUM(b.gross_spend), 0), 4) AS pre_discount_share,
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
            ROUND(SUM(b.spend) / u.camp_duration, 4) AS camp_spend_per_day,
            MAX(b.coupon_used) AS camp_coupon_used
        FROM eligible_units u
        JOIN marts.fct_baskets b ON u.household_key = b.household_key AND b.day BETWEEN u.start_day AND u.end_day
        GROUP BY u.campaign, u.household_key, u.camp_duration
    )
    SELECT
        u.campaign,
        u.campaign_type,
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
    return df


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
) -> pd.DataFrame:
    """Compute incremental margin, cost, and net ROI across targeting depth (10% to 100%)."""
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

        # Assuming average campaign window duration ~ 35 days
        camp_days = 35.0
        incremental_spend = targeted_mean_cate * camp_days * k
        incremental_margin = incremental_spend * gross_margin
        total_offer_cost = offer_cost * k
        net_profit = incremental_margin - total_offer_cost
        roi = net_profit / total_offer_cost if total_offer_cost > 0 else 0.0

        # Random targeting comparison
        random_spend = overall_mean_cate * camp_days * k
        random_margin = random_spend * gross_margin
        random_net = random_margin - total_offer_cost

        records.append(
            {
                "targeting_percentile": int(pct * 100),
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
    """Segment customers into Persuadables, Sure Things, Lost Causes, Sleeping Dogs."""
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
        df_seg["clv_26w"] = 250.0

    df_seg["clv_26w"] = df_seg["clv_26w"].fillna(250.0)

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


def main() -> None:
    """Run Dunnhumby observational uplift modeling, propensity balance, and evaluation."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db
    out_dir = project_root / cfg.paths.data_outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting Dunnhumby campaign-household observational units...")
    raw_df = extract_dunnhumby_campaign_units(db_path=db_path)
    print(
        f"Extracted {len(raw_df)} eligible units across {raw_df['household_key'].nunique()} households."
    )

    covariates = [
        "pre_trips",
        "pre_spend_per_day",
        "pre_discount_share",
        "pre_coupon_trips",
        "has_demographics",
        "concurrent_campaigns",
    ]

    # One-hot encode campaign type
    df = pd.get_dummies(raw_df, columns=["campaign_type"], drop_first=True, dtype=float)
    feat_cols = covariates + [c for c in df.columns if c.startswith("campaign_type_")]

    # 1. Propensity modeling
    print("\nFitting propensity score model (Logistic Regression)...")
    X_prop = df[feat_cols].fillna(0.0).values
    t_prop = df["treated"].values

    prop_clf = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    prop_clf.fit(X_prop, t_prop)
    propensities = prop_clf.predict_proba(X_prop)[:, 1]
    df["propensity_score"] = propensities

    # Overlap trimming: [0.05, 0.95]
    overlap_mask = (df["propensity_score"] >= 0.05) & (df["propensity_score"] <= 0.95)
    trimmed_df = df[overlap_mask].copy().reset_index(drop=True)
    print(f"Trimmed extreme propensity units: retained {len(trimmed_df)} / {len(df)} units.")

    # 2. Covariate Balance Table before and after IPW
    print("\nComputing covariate balance table (SMD before vs after IPW)...")
    balance_df = compute_balance_table(
        df=trimmed_df,
        covariates=covariates,
        propensity_scores=trimmed_df["propensity_score"].values,
    )
    print(balance_df.to_string())
    balance_path = out_dir / "uplift_balance.parquet"
    balance_df.to_parquet(balance_path, index=False)

    # 3. Fit meta-learners with GroupKFold by household
    print("\nEvaluating meta-learners with GroupKFold by household...")
    gkf = GroupKFold(n_splits=3)
    train_idx, test_idx = next(gkf.split(trimmed_df, groups=trimmed_df["household_key"]))

    train_data = trimmed_df.iloc[train_idx].copy().reset_index(drop=True)
    test_data = trimmed_df.iloc[test_idx].copy().reset_index(drop=True)

    X_tr = train_data[feat_cols].values
    t_tr = train_data["treated"].values
    y_tr = train_data["delta_spend_per_day"].values
    p_tr = train_data["propensity_score"].values

    X_te = test_data[feat_cols].values
    t_te = test_data["treated"].values
    y_te = test_data["delta_spend_per_day"].values
    p_te = test_data["propensity_score"].values

    learners = {
        "S-Learner": BaseSLearner(
            learner=LGBMRegressor(n_estimators=60, random_state=cfg.seed, verbose=-1)
        ),
        "T-Learner": BaseTLearner(
            learner=LGBMRegressor(n_estimators=60, random_state=cfg.seed, verbose=-1)
        ),
        "X-Learner": BaseXLearner(
            learner=LGBMRegressor(n_estimators=60, random_state=cfg.seed, verbose=-1)
        ),
        "DR-Learner": BaseDRLearner(
            learner=LGBMRegressor(n_estimators=60, random_state=cfg.seed, verbose=-1)
        ),
    }

    eval_rows = []
    best_cate = None
    best_qini = -999.0

    for name, model in learners.items():
        if name in ["X-Learner", "DR-Learner"]:
            model.fit(X=X_tr, treatment=t_tr, y=y_tr, p=p_tr)
            cate = model.predict(X=X_te, p=p_te).ravel()
        else:
            model.fit(X=X_tr, treatment=t_tr, y=y_tr)
            cate = model.predict(X=X_te).ravel()

        eval_df = pd.DataFrame({"y": y_te, "w": t_te, "pred": cate})
        qini = float(qini_score(eval_df)["pred"])
        auuc = float(auuc_score(eval_df)["pred"])

        order = np.argsort(-cate)
        top_10 = order[: int(len(test_data) * 0.10)]
        top_30 = order[: int(len(test_data) * 0.30)]
        uplift_10 = float(
            y_te[top_10][t_te[top_10] == 1].mean() - y_te[top_10][t_te[top_10] == 0].mean()
        )
        uplift_30 = float(
            y_te[top_30][t_te[top_30] == 1].mean() - y_te[top_30][t_te[top_30] == 0].mean()
        )

        eval_rows.append(
            {
                "learner": name,
                "qini": round(qini, 4),
                "auuc": round(auuc, 4),
                "uplift_at_10pct": round(uplift_10, 4),
                "uplift_at_30pct": round(uplift_30, 4),
                "mean_cate": round(float(np.mean(cate)), 4),
            }
        )

        if qini > best_qini:
            best_qini = qini
            best_cate = cate

    eval_table = pd.DataFrame(eval_rows).sort_values("qini", ascending=False).reset_index(drop=True)
    print("\nDunnhumby Uplift Evaluation:")
    print(eval_table.to_string())
    eval_table.to_parquet(out_dir / "uplift_eval.parquet", index=False)

    # 4. Targeting ROI Curve
    print("\nComputing targeting ROI curve...")
    offer_cost = cfg.offer_cost if hasattr(cfg, "offer_cost") else 2.00
    gross_margin = cfg.gross_margin if hasattr(cfg, "gross_margin") else 0.25
    roi_df = compute_targeting_roi_curve(
        test_df=test_data,
        cate_preds=best_cate,
        offer_cost=offer_cost,
        gross_margin=gross_margin,
    )
    print(roi_df.to_string())
    roi_df.to_parquet(out_dir / "uplift_roi.parquet", index=False)

    # 5. Customer Segmentation
    print("\nSegmenting customers into Persuadables, Sure Things, Lost Causes, Sleeping Dogs...")
    clv_path = out_dir / "clv.parquet"
    clv_df = pd.read_parquet(clv_path) if clv_path.exists() else pd.DataFrame()
    scored_test, seg_summary = segment_customers(test_data, best_cate, clv_df)
    print(seg_summary.to_string())

    # Save full scores
    scored_cols = [
        "household_key",
        "campaign",
        "campaign_type",
        "treated",
        "propensity_score",
        "uplift_tau",
        "segment",
        "clv_26w",
    ]
    save_cols = [c for c in scored_cols if c in scored_test.columns]
    scored_test[save_cols].to_parquet(out_dir / "uplift_scores.parquet", index=False)
    print(f"\nArtifacts exported to {out_dir}")


if __name__ == "__main__":
    main()
