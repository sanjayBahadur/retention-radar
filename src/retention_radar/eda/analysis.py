"""Core logic for exploratory data analysis, KPI calculations, and chart exports."""

import json
from pathlib import Path
from typing import Any, Dict

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from retention_radar.config import get_project_root, load_project_config


def compute_headline_kpis(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Compute headline retail KPIs across all 25 periods (days 1 to 700)."""
    res = con.execute("""
        SELECT
            COUNT(DISTINCT household_key) AS total_households,
            COUNT(DISTINCT basket_id) AS total_baskets,
            ROUND(SUM(spend), 2) AS total_revenue,
            ROUND(AVG(spend), 2) AS avg_basket_spend,
            ROUND(SUM(spend) / COUNT(DISTINCT household_key), 2) AS spend_per_household,
            ROUND(COUNT(DISTINCT basket_id) * 1.0 / (COUNT(DISTINCT household_key) * 25), 2) AS trips_per_hh_per_period
        FROM marts.fct_baskets;
    """).fetchone()

    return {
        "total_households": int(res[0]),
        "total_baskets": int(res[1]),
        "total_revenue": float(res[2]),
        "avg_basket_spend": float(res[3]),
        "spend_per_household": float(res[4]),
        "trips_per_hh_per_period": float(res[5]),
    }


def compute_concentration(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Compute Lorenz concentration and top 20% household spend share."""
    df = con.execute("""
        SELECT household_key, lifetime_spend
        FROM marts.dim_household
        ORDER BY lifetime_spend ASC;
    """).df()

    spend = df["lifetime_spend"].values
    total_spend = spend.sum()
    cum_spend = np.cumsum(spend) / total_spend
    cum_pop = np.arange(1, len(spend) + 1) / len(spend)

    # Top 20% revenue share
    top_20_share = float(spend[int(len(spend) * 0.8) :].sum() * 100.0 / total_spend)

    # Gini coefficient
    gini = float(1.0 - 2.0 * np.trapezoid(cum_spend, cum_pop))

    return {
        "cum_pop": cum_pop,
        "cum_spend": cum_spend,
        "top_20pct_revenue_share": round(top_20_share, 2),
        "gini_coefficient": round(gini, 4),
    }


def compute_churn_base_rate(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Compute binary churn base rate: share of active households with zero purchases in next 2 periods."""
    res = con.execute("""
        WITH active_households AS (
            SELECT period_id, household_key
            FROM marts.fct_household_period_state
            WHERE period_id BETWEEN 6 AND 13
              AND trips > 0
        ),
        future_activity AS (
            SELECT
                a.period_id, a.household_key,
                COALESCE(SUM(f.trips), 0) AS next_2_periods_trips
            FROM active_households a
            LEFT JOIN marts.fct_household_period_state f
                ON a.household_key = f.household_key
                AND f.period_id IN (a.period_id + 1, a.period_id + 2)
            GROUP BY a.period_id, a.household_key
        )
        SELECT
            COUNT(*) AS total_active_instances,
            SUM(CASE WHEN next_2_periods_trips = 0 THEN 1 ELSE 0 END) AS churn_instances,
            ROUND(SUM(CASE WHEN next_2_periods_trips = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_base_rate_pct
        FROM future_activity;
    """).fetchone()

    return {
        "total_active_instances": int(res[0]),
        "churn_instances": int(res[1]),
        "churn_base_rate_pct": float(res[2]),
    }


def compute_demographic_coverage(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Compute demographic coverage and lapse rates across income/age bands."""
    res = con.execute("""
        SELECT
            COUNT(*) AS total_households,
            SUM(has_demographics) AS demographic_households,
            ROUND(SUM(has_demographics) * 100.0 / COUNT(*), 2) AS coverage_pct
        FROM marts.dim_household;
    """).fetchone()

    return {
        "total_households": int(res[0]),
        "demographic_households": int(res[1]),
        "coverage_pct": float(res[2]),
    }


def generate_all_eda_figures(db_path: Path, figures_dir: Path) -> Dict[str, Any]:
    """Generate and save publication figures for exploratory data analysis."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)

    # 1. Lorenz Concentration Curve
    conc = compute_concentration(con)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(
        conc["cum_pop"] * 100,
        conc["cum_spend"] * 100,
        label=f"Observed Spend (Gini = {conc['gini_coefficient']})",
        color="#1f77b4",
        lw=2,
    )
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Perfect Equality")
    ax.axvline(
        80,
        color="firebrick",
        linestyle=":",
        label=f"Top 20% HHs generate {conc['top_20pct_revenue_share']}% revenue",
    )
    ax.set_title("Household Spend Concentration (Lorenz Curve)", fontsize=11, fontweight="bold")
    ax.set_xlabel("% of Households (Sorted by Spend)")
    ax.set_ylabel("Cumulative % of Revenue")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_lorenz_curve.png")
    plt.close(fig)

    # 2. Purchase Rhythm (Inter-purchase Days & Hour of Day)
    rhythm_df = con.execute("""
        WITH trips AS (
            SELECT DISTINCT household_key, day
            FROM marts.fct_baskets
            ORDER BY household_key, day
        ),
        diffs AS (
            SELECT
                day - LAG(day) OVER (PARTITION BY household_key ORDER BY day) AS intertrip_days
            FROM trips
        )
        SELECT intertrip_days FROM diffs WHERE intertrip_days IS NOT NULL AND intertrip_days <= 60;
    """).df()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.hist(rhythm_df["intertrip_days"], bins=40, color="#2ca02c", edgecolor="white", alpha=0.85)
    median_days = float(rhythm_df["intertrip_days"].median())
    ax.axvline(
        median_days,
        color="darkred",
        linestyle="--",
        lw=2,
        label=f"Median Inter-trip Gap = {median_days:.0f} days",
    )
    ax.set_title("Customer Inter-Purchase Time Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Days Between Consecutive Shopping Occasions")
    ax.set_ylabel("Occasion Count")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_purchase_rhythm.png")
    plt.close(fig)

    # 3. Weekly Revenue Across 102 Weeks
    weekly_rev = con.execute("""
        SELECT week_no, ROUND(SUM(sales_value), 2) AS weekly_revenue
        FROM staging.stg_transactions
        GROUP BY week_no
        ORDER BY week_no;
    """).df()

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(weekly_rev["week_no"], weekly_rev["weekly_revenue"] / 1000, color="#9467bd", lw=2)
    ax.set_title("Weekly Store Revenue (102 Weeks)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Week Number")
    ax.set_ylabel("Revenue ($ Thousands)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_weekly_seasonality.png")
    plt.close(fig)

    # 4. Lifecycle State Trajectory Over 25 Periods
    state_df = con.execute("""
        SELECT period_id, state, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY period_id) AS pct
        FROM marts.fct_household_period_state
        GROUP BY period_id, state
        ORDER BY period_id, state;
    """).df()
    state_pivot = state_df.pivot(index="period_id", columns="state", values="pct").fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = {
        "Champion": "#2ca02c",
        "Regular": "#1f77b4",
        "New": "#17becf",
        "At-Risk": "#ff7f0e",
        "Dormant": "#d62728",
        "Lapsed": "#7f7f7f",
    }
    ordered_cols = [
        c
        for c in ["Champion", "Regular", "New", "At-Risk", "Dormant", "Lapsed"]
        if c in state_pivot.columns
    ]
    state_pivot[ordered_cols].plot(
        kind="area", stacked=True, ax=ax, color=[colors[c] for c in ordered_cols], alpha=0.85
    )
    ax.set_title(
        "Household Lifecycle State Composition Across 28-Day Periods",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("28-Day Period ID (1 to 25)")
    ax.set_ylabel("% of Households")
    ax.set_ylim(0, 100)
    ax.legend(title="Lifecycle State", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_state_transitions.png")
    plt.close(fig)

    # 5. Cohort Retention Curves
    cohort_df = con.execute("""
        WITH cohorts AS (
            SELECT household_key, first_period_id AS cohort
            FROM marts.fct_household_period_state
            WHERE period_id = first_period_id AND first_period_id BETWEEN 4 AND 10
        ),
        activity AS (
            SELECT
                c.cohort,
                s.period_id - c.cohort AS elapsed_periods,
                COUNT(DISTINCT CASE WHEN s.trips > 0 THEN s.household_key END) * 100.0 / COUNT(DISTINCT c.household_key) AS retention_pct
            FROM cohorts c
            INNER JOIN marts.fct_household_period_state s ON c.household_key = s.household_key
            WHERE s.period_id >= c.cohort
            GROUP BY c.cohort, s.period_id - c.cohort
        )
        SELECT * FROM activity WHERE elapsed_periods <= 12 ORDER BY cohort, elapsed_periods;
    """).df()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for c_id, grp in cohort_df.groupby("cohort"):
        ax.plot(
            grp["elapsed_periods"], grp["retention_pct"], marker="o", label=f"Cohort Period {c_id}"
        )
    ax.set_title(
        "Post-Warmup Customer Acquisition Cohort Retention", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("28-Day Periods Since Acquisition")
    ax.set_ylabel("% Active (Trips > 0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_cohort_retention.png")
    plt.close(fig)

    summary = {
        "kpis": compute_headline_kpis(con),
        "concentration": conc,
        "churn": compute_churn_base_rate(con),
        "demographics": compute_demographic_coverage(con),
    }
    con.close()
    return summary


def main() -> None:
    """Run full EDA pipeline and output summary metrics."""
    root = get_project_root()
    cfg = load_project_config()
    db_path = root / cfg.paths.warehouse_db
    figures_dir = root / cfg.paths.figures_dir

    results = generate_all_eda_figures(db_path=db_path, figures_dir=figures_dir)
    serializable = {
        **results,
        "concentration": {
            key: value
            for key, value in results["concentration"].items()
            if key not in {"cum_pop", "cum_spend"}
        },
    }
    out = root / cfg.paths.data_outputs
    out.mkdir(parents=True, exist_ok=True)
    (out / "eda_summary.json").write_text(json.dumps(serializable, indent=2) + "\n")
    print("EDA Generation Complete:")
    print("  Headline KPIs:", results["kpis"])
    print("  Top 20% Revenue Share:", results["concentration"]["top_20pct_revenue_share"], "%")
    print("  Gini Coefficient:", results["concentration"]["gini_coefficient"])
    print("  Churn Base Rate (Active HHs):", results["churn"]["churn_base_rate_pct"], "%")
    print("  Demographic Coverage:", results["demographics"]["coverage_pct"], "%")


if __name__ == "__main__":
    main()
