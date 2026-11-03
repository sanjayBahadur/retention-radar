"""Readable error review of the held-out churn predictions."""

import numpy as np
import pandas as pd

from retention_radar.config import get_project_root, load_project_config


def summarize_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Review a 20% contact budget independently at each historical cutoff."""
    data = predictions.copy()
    data["selected"] = False
    for _, period in data.groupby("cutoff_period"):
        chosen = period.sort_values("raw_p", ascending=False, kind="stable").head(
            max(1, int(len(period) * 0.20))
        )
        data.loc[chosen.index, "selected"] = True
    data["tenure_group"] = np.where(
        data.tenure_periods <= 6, "Up to 6 periods", "More than 6 periods"
    )
    rows = []
    groups = [("Overall", "All observations", data)]
    for column in ("cutoff_period", "current_state", "tenure_group"):
        groups.extend((column, str(label), group) for label, group in data.groupby(column))
    for dimension, group, frame in groups:
        churn = frame.churn_label.astype(bool)
        selected = frame.selected
        true_positive = int((churn & selected).sum())
        actual = int(churn.sum())
        contacts = int(selected.sum())
        rows.append(
            {
                "dimension": dimension,
                "group": group,
                "observations": len(frame),
                "households": int(frame.household_key.nunique()),
                "actual_departures": actual,
                "contacts": contacts,
                "captured_departures": true_positive,
                "missed_departures": actual - true_positive,
                "false_alarms": contacts - true_positive,
                "capture_rate": true_positive / actual if actual else None,
                "contact_precision": true_positive / contacts if contacts else None,
            }
        )
    return pd.DataFrame(rows)


def main():
    root = get_project_root()
    cfg = load_project_config()
    out = root / cfg.paths.data_outputs
    review = summarize_errors(pd.read_parquet(out / "churn_test_predictions.parquet"))
    review.to_parquet(out / "churn_error_analysis.parquet", index=False)
    lines = [
        "# Where the churn ranking helps—and where it misses",
        "",
        "This review ranks households separately at each test cutoff and selects the highest-risk 20%. "
        "Selection uses the raw model ranking; calibrated probabilities are used for financial scenarios. "
        "A departure means no purchases in the next two 28-day periods.",
        "",
        "A false alarm is a contacted household that continued shopping. A missed departure is a household "
        "that stopped visiting but fell outside the contact budget. Neither proves that an offer would help.",
        "",
        "| Group | Observations | Departures | Contacted | Captured | Missed | False alarms | Capture |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in review.itertuples():
        capture = f"{row.capture_rate:.1%}" if pd.notna(row.capture_rate) else "—"
        lines.append(
            f"| {row.dimension}: {row.group} | {row.observations:,} | {row.actual_departures:,} | "
            f"{row.contacts:,} | {row.captured_departures:,} | {row.missed_departures:,} | {row.false_alarms:,} | {capture} |"
        )
    lines += [
        "",
        "## How to interpret this",
        "",
        "Households can appear at several cutoffs. These are descriptive counts, not independent samples "
        "or proof that a subgroup difference is meaningful. Small groups need particular care. "
        "The overall model comparison reports household-clustered bootstrap intervals; this table does not "
        "attach independent-row confidence intervals to repeated household observations.",
        "",
        "Before deployment, investigate missed departures and false alarms with an analyst, review changes "
        "across time, and run a household-randomized offer experiment. "
        "These historical errors were used for review, not further model tuning.",
        "",
    ]
    (root / cfg.paths.reports_dir / "evaluation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
