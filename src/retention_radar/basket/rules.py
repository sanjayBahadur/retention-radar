"""Market basket association rule mining using FP-Growth and network visualization."""

from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from retention_radar.config import get_project_root, load_project_config

EXCLUDED_DEPARTMENTS = [
    "MISC. TRANS.",
    "MISC SALES TRAN",
    "COUP/STR & MFG",
    "KIOSK-GAS",
    "CNTRL/STORE SUP",
    "CHARITABLE CONT",
    "PHOTO",
    "PROD-WHS SALES",
    "PHARMACY SUPPLY",
    "",
]


def extract_baskets(
    con: duckdb.DuckDBPyConnection,
    state_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """Extract merchandise transactions grouped into baskets at commodity level."""
    state_clause = f"AND s.state = '{state_filter}'" if state_filter else ""
    limit_clause = f"LIMIT {limit}" if limit else ""
    excl_list = ", ".join(f"'{d}'" for d in EXCLUDED_DEPARTMENTS)

    query = f"""
        SELECT
            t.basket_id,
            t.household_key,
            p.commodity_desc
        FROM staging.stg_transactions t
        JOIN staging.stg_products p ON t.product_id = p.product_id
        JOIN marts.fct_household_period_state s ON t.household_key = s.household_key AND t.period_id = s.period_id
        WHERE p.department NOT IN ({excl_list})
          AND p.commodity_desc IS NOT NULL
          AND t.period_id BETWEEN 4 AND 19
          {state_clause}
        {limit_clause};
    """
    df = con.execute(query).df()
    baskets_grouped = (
        df.groupby("basket_id")["commodity_desc"].apply(lambda s: list(set(s))).tolist()
    )
    # Filter baskets with at least 2 distinct commodities
    valid_baskets = [b for b in baskets_grouped if len(b) >= 2]
    return valid_baskets, df


def mine_association_rules(
    baskets: List[List[str]],
    min_support: float = 0.01,
    min_lift: float = 1.20,
) -> pd.DataFrame:
    """Mine non-trivial association rules using FP-Growth."""
    if len(baskets) < 10:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)

    # Start with min_support, lower if too few frequent itemsets
    frequent = fpgrowth(df_te, min_support=min_support, use_colnames=True)
    if len(frequent) < 30 and min_support > 0.005:
        min_support = 0.005
        frequent = fpgrowth(df_te, min_support=min_support, use_colnames=True)

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)
    if rules.empty:
        return pd.DataFrame()

    # Format antecedents and consequents as readable strings
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

    # Select and round metrics
    cols = [
        "antecedents_str",
        "consequents_str",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    res_df = rules[cols].copy().sort_values("lift", ascending=False).reset_index(drop=True)
    for num_col in ["support", "confidence", "lift", "leverage", "conviction"]:
        res_df[num_col] = res_df[num_col].round(4)

    return res_df


def plot_basket_rules_network(
    rules_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Plot directed network graph of top association rules using NetworkX and Matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_rules = rules_df.head(top_n)

    G = nx.DiGraph()

    for _, row in sub_rules.iterrows():
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        # Limit label length for neat rendering
        ant_short = ant if len(ant) <= 24 else ant[:21] + "..."
        con_short = con if len(con) <= 24 else con[:21] + "..."
        lift = float(row["lift"])
        G.add_edge(ant_short, con_short, weight=lift)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # Edge widths and colors based on lift
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    norm_weights = [
        (w - min(weights)) / (max(weights) - min(weights) + 1e-5) * 3 + 1 for w in weights
    ]

    nx.draw_networkx_nodes(
        G, pos, node_size=1800, node_color="#E3F2FD", edgecolors="#1B365D", linewidths=1.5
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=norm_weights,
        edge_color=weights,
        edge_cmap=plt.cm.viridis,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", font_weight="bold")

    plt.title("Market Basket Association Rules Network (Top 30 Rules by Lift)", fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved association rules network graph to {output_path}")


def main() -> None:
    """Run market basket association rule mining for overall and segment-specific baskets."""
    project_root = get_project_root()
    cfg = load_project_config()
    db_path = project_root / cfg.paths.warehouse_db
    out_dir = project_root / cfg.paths.data_outputs
    figures_dir = project_root / cfg.paths.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)

    print("Extracting merchandise baskets across training periods...")
    all_baskets, _ = extract_baskets(con, limit=150000)
    print(f"Extracted {len(all_baskets)} multi-item baskets.")

    print("\nMining global association rules via FP-Growth...")
    rules_df = mine_association_rules(all_baskets, min_support=0.01, min_lift=1.20)
    print(f"Mined {len(rules_df)} rules with lift > 1.20.")
    print(rules_df.head(10).to_string())

    rules_path = out_dir / "basket_rules.parquet"
    rules_df.to_parquet(rules_path, index=False)

    # Segment specific rules: Champion vs At-Risk
    print("\nMining segment-specific baskets: Champion vs At-Risk...")
    champ_baskets, _ = extract_baskets(con, state_filter="Champion", limit=80000)
    at_risk_baskets, _ = extract_baskets(con, state_filter="At-Risk", limit=80000)
    con.close()

    champ_rules = mine_association_rules(champ_baskets, min_support=0.015, min_lift=1.20)
    at_risk_rules = mine_association_rules(at_risk_baskets, min_support=0.015, min_lift=1.20)

    # Compare segment rules
    champ_rules["segment"] = "Champion"
    at_risk_rules["segment"] = "At-Risk"
    segment_rules = pd.concat([champ_rules, at_risk_rules], ignore_index=True)

    seg_path = out_dir / "basket_rules_by_segment.parquet"
    segment_rules.to_parquet(seg_path, index=False)
    print(f"Saved segment rules to {seg_path} ({len(segment_rules)} rows).")

    # Plot network graph of top 30 global rules
    network_path = figures_dir / "basket_rules_network.png"
    plot_basket_rules_network(rules_df, network_path, top_n=30)


if __name__ == "__main__":
    main()
