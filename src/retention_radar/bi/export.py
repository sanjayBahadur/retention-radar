"""Export flattened BI datasets for Power BI and Tableau visualization."""

from pathlib import Path
from typing import Dict

import pandas as pd

from retention_radar.config import get_project_root, load_project_config


def export_bi_datasets(out_dir: Path, bi_dir: Path) -> Dict[str, Path]:
    """Export clean, well-structured CSV tables for external BI tools."""
    bi_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    # 1. Lifecycle States
    state_path = out_dir / "state_distribution.parquet"
    if state_path.exists():
        df_state = pd.read_parquet(state_path)
        dest = bi_dir / "bi_lifecycle_states.csv"
        df_state.to_csv(dest, index=False)
        exported_files["bi_lifecycle_states.csv"] = dest

    # 2. Churn Risk Scores & Outreach Targets
    churn_path = out_dir / "churn_scores.parquet"
    if churn_path.exists():
        df_churn = pd.read_parquet(churn_path)
        dest = bi_dir / "bi_churn_scores.csv"
        df_churn.to_csv(dest, index=False)
        exported_files["bi_churn_scores.csv"] = dest

    # 3. Campaign Attribution Shares
    attr_path = out_dir / "attribution.parquet"
    if attr_path.exists():
        df_attr = pd.read_parquet(attr_path)
        dest = bi_dir / "bi_campaign_attribution.csv"
        df_attr.to_csv(dest, index=False)
        exported_files["bi_campaign_attribution.csv"] = dest

    # 4. Market Basket Association Rules
    basket_path = out_dir / "basket_rules.parquet"
    if basket_path.exists():
        df_basket = pd.read_parquet(basket_path).head(100)  # Top 100 for clean BI consumption
        dest = bi_dir / "bi_basket_rules.csv"
        df_basket.to_csv(dest, index=False)
        exported_files["bi_basket_rules.csv"] = dest

    # 5. Customer Uplift Segments
    uplift_path = out_dir / "uplift_scores.parquet"
    if uplift_path.exists():
        df_uplift = pd.read_parquet(uplift_path)
        dest = bi_dir / "bi_customer_segments.csv"
        df_uplift.to_csv(dest, index=False)
        exported_files["bi_customer_segments.csv"] = dest

    return exported_files


def main() -> None:
    """Run BI dataset export."""
    project_root = get_project_root()
    cfg = load_project_config()
    out_dir = project_root / cfg.paths.data_outputs
    bi_dir = project_root / cfg.paths.bi_exports

    print("Exporting BI datasets for Power BI / Tableau...")
    files = export_bi_datasets(out_dir=out_dir, bi_dir=bi_dir)
    for name, p in files.items():
        print(f"  {name} -> {p}")
    print(f"\nSuccessfully exported {len(files)} BI datasets to {bi_dir}.")


if __name__ == "__main__":
    main()
