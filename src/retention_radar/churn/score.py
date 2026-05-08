"""Batch scoring, profit curve optimization, and Excel target list export."""

import json
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.isotonic import IsotonicRegression

from retention_radar.churn.explain import assign_reason_codes, compute_shap_explanations
from retention_radar.churn.train import prepare_feature_matrices
from retention_radar.config import get_project_root, load_project_config


def compute_profit_curves(
    df: pd.DataFrame,
    save_rates: List[float] = [0.05, 0.10, 0.15, 0.20],
    offer_cost: float = 2.00,
) -> pd.DataFrame:
    """Compute expected value and targeting profit curves across save rates and thresholds."""
    rows = []
    # Rank households by calibrated churn probability descending
    df_sorted = df.sort_values("calibrated_p", ascending=False).reset_index(drop=True)
    n = len(df_sorted)

    percentiles = np.linspace(0.01, 1.0, 100)

    for save_rate in save_rates:
        # Individual expected value of intervention
        # EV = p * save_rate * CLV - offer_cost
        clv_margin = np.maximum(df_sorted["clv_26w"].fillna(df_sorted["clv_26w"].median()), 10.0)
        p = df_sorted["calibrated_p"]
        indiv_ev = p * save_rate * clv_margin - offer_cost

        cum_ev = np.cumsum(indiv_ev)
        cum_cost = np.arange(1, n + 1) * offer_cost
        cum_gross = cum_ev + cum_cost

        for pct in percentiles:
            k = max(1, int(n * pct))
            total_ev = cum_ev.iloc[k - 1]
            total_cost = cum_cost[k - 1]
            total_gross = cum_gross.iloc[k - 1]
            roi = total_gross / total_cost if total_cost > 0 else 0.0

            rows.append(
                {
                    "save_rate": save_rate,
                    "target_percentile": round(pct, 2),
                    "households_targeted": k,
                    "probability_threshold": round(float(df_sorted["calibrated_p"].iloc[k - 1]), 4),
                    "total_ev": round(float(total_ev), 2),
                    "total_cost": round(float(total_cost), 2),
                    "expected_roi": round(float(roi), 3),
                }
            )

    return pd.DataFrame(rows)


def export_excel_target_list(
    df_scores: pd.DataFrame,
    output_path: Path,
    top_deciles: List[int] = [1, 2],
) -> None:
    """Export formatted Excel target list for marketing retention team using openpyxl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_df = df_scores[df_scores["risk_decile"].isin(top_deciles)].copy()
    target_df = target_df.sort_values(["risk_decile", "calibrated_p"], ascending=[True, False])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "High Risk Retention Targets"

    # Header styling
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="E0E0E0"),
        right=Side(style="thin", color="E0E0E0"),
        top=Side(style="thin", color="E0E0E0"),
        bottom=Side(style="thin", color="E0E0E0"),
    )

    high_risk_fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")

    columns = [
        ("household_key", "Household ID", center_align, "@"),
        ("current_state", "Lifecycle State", center_align, "@"),
        ("risk_decile", "Risk Decile", center_align, "0"),
        ("calibrated_p", "Churn Probability", right_align, "0.0%"),
        ("clv_26w", "26-Week CLV ($)", right_align, "$#,##0.00"),
        ("targeting_ev", "Targeting EV ($)", right_align, "$#,##0.00"),
        ("target_action", "Recommended Action", center_align, "@"),
        ("reason_1", "Primary Risk Driver", left_align, "@"),
        ("reason_2", "Secondary Driver", left_align, "@"),
        ("reason_3", "Tertiary Driver", left_align, "@"),
    ]

    # Write headers
    for col_idx, (col_id, col_label, align, num_fmt) in enumerate(columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_label)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    # Write rows
    for row_idx, (_, row) in enumerate(target_df.iterrows(), 2):
        is_decile_1 = row["risk_decile"] == 1
        for col_idx, (col_id, col_label, align, num_fmt) in enumerate(columns, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row.get(col_id, ""))
            cell.font = data_font
            cell.alignment = align
            cell.number_format = num_fmt
            cell.border = thin_border
            if is_decile_1:
                cell.fill = high_risk_fill

    # Auto-fit column widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(str(output_path))
    print(f"Exported Excel target list to {output_path} ({len(target_df)} households).")


def main() -> None:
    """Run batch scoring, profit curve computation, explainability, and report generation."""
    project_root = get_project_root()
    cfg = load_project_config()

    feat_path = project_root / cfg.paths.data_outputs / "churn_features.parquet"
    if not feat_path.exists():
        from retention_radar.features.build_features import main as build_feat_main

        build_feat_main()

    df = pd.read_parquet(feat_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    models_dir = project_root / cfg.paths.models_dir
    with open(models_dir / "churn_lightgbm_metrics.json", "r") as f:
        meta = json.load(f)

    # Load model and fit isotonic calibration on validation set
    booster = lgb.Booster(model_file=str(models_dir / "churn_lightgbm.txt"))

    X_val_raw, y_val, _ = prepare_feature_matrices(val_df)
    X_val = X_val_raw.reindex(columns=meta["feature_names"], fill_value=0.0)
    val_preds = booster.predict(X_val)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_preds, y_val.values)

    # Score latest test period (cutoff 23 is the latest test cutoff)
    latest_cutoff = test_df["cutoff_period"].max()
    score_df = test_df[test_df["cutoff_period"] == latest_cutoff].copy().reset_index(drop=True)

    X_score_raw, _, _ = prepare_feature_matrices(score_df)
    X_score = X_score_raw.reindex(columns=meta["feature_names"], fill_value=0.0)

    raw_scores = booster.predict(X_score)
    calibrated_scores = iso.transform(raw_scores)

    score_df["raw_p"] = raw_scores.round(4)
    score_df["calibrated_p"] = calibrated_scores.round(4)

    # Risk decile (1 = top 10% highest churn probability)
    ranks = score_df["calibrated_p"].rank(method="first", ascending=False)
    score_df["risk_decile"] = pd.qcut(ranks, q=10, labels=list(range(1, 11))).astype(int)

    # Merge CLV estimates from data/outputs/clv.parquet
    clv_path = project_root / cfg.paths.data_outputs / "clv.parquet"
    if clv_path.exists():
        clv_df = pd.read_parquet(clv_path)
        score_df = score_df.merge(
            clv_df[["household_key", "clv_26w"]],
            on="household_key",
            how="left",
        )
    else:
        score_df["clv_26w"] = 250.0

    score_df["clv_26w"] = score_df["clv_26w"].fillna(score_df["clv_26w"].median()).round(2)

    # Compute EV at default assumption: save_rate = 0.10, offer_cost = 2.00
    offer_cost = cfg.offer_cost if hasattr(cfg, "offer_cost") else 2.00
    default_save_rate = 0.10
    score_df["targeting_ev"] = (
        score_df["calibrated_p"] * default_save_rate * score_df["clv_26w"] - offer_cost
    ).round(2)
    score_df["target_action"] = np.where(
        score_df["targeting_ev"] > 0, "High-Priority Offer", "Standard Marketing"
    )

    # Compute SHAP values and assign human readable reason codes
    figures_dir = project_root / cfg.paths.figures_dir
    print("Computing SHAP values and generating explainability plots...")
    shap_vals, top_feats = compute_shap_explanations(booster, X_score, figures_dir)
    score_df = assign_reason_codes(score_df, X_score, shap_vals)

    # Output churn_scores.parquet
    out_dir = project_root / cfg.paths.data_outputs
    scores_out_path = out_dir / "churn_scores.parquet"
    final_cols = [
        "household_key",
        "cutoff_period",
        "current_state",
        "calibrated_p",
        "risk_decile",
        "clv_26w",
        "targeting_ev",
        "target_action",
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    score_df[final_cols].to_parquet(scores_out_path, index=False)
    print(f"Saved churn scores to {scores_out_path} ({len(score_df)} active households scored).")

    # Compute and save profit curves
    print("Computing profit curves across save rate grid...")
    save_rates = cfg.save_rate_grid if hasattr(cfg, "save_rate_grid") else [0.05, 0.10, 0.15, 0.20]
    profit_df = compute_profit_curves(score_df, save_rates=save_rates, offer_cost=offer_cost)
    profit_path = out_dir / "profit_curve.parquet"
    profit_df.to_parquet(profit_path, index=False)
    print(f"Saved profit curves to {profit_path}.")

    # Export formatted Excel target list
    excel_path = project_root / cfg.paths.reports_dir / "churn_target_list.xlsx"
    export_excel_target_list(score_df, excel_path, top_deciles=[1, 2])


if __name__ == "__main__":
    main()
