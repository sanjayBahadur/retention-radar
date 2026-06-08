"""Integration test running end-to-end pipeline components under CI profile."""

import pytest

from retention_radar.attribution.markov_attribution import main as attr_main
from retention_radar.basket.early_warning import main as basket_ew_main
from retention_radar.basket.rules import main as basket_rules_main
from retention_radar.churn.fairness import main as churn_fairness_main
from retention_radar.churn.score import main as churn_score_main
from retention_radar.clv.compare import main as clv_main
from retention_radar.eda.analysis import main as eda_main
from retention_radar.markov.chain import main as markov_main
from retention_radar.monitoring.drift import main as drift_main
from retention_radar.schemas import main as schemas_main
from retention_radar.uplift.dunnhumby import main as uplift_dunnhumby_main
from retention_radar.uplift.hillstrom import main as uplift_hillstrom_main
from retention_radar.uplift.power import main as uplift_power_main
from retention_radar.uplift.x5 import main as uplift_x5_main


@pytest.mark.integration
def test_pipeline_e2e_components(tmp_path, monkeypatch):
    """Verify end-to-end pipeline components execute cleanly on warehouse database."""
    import shutil

    from retention_radar.config import get_project_root, load_project_config

    root = get_project_root()
    real_cfg = load_project_config()

    # Create temporary outputs and reports directories pre-seeded with current outputs
    tmp_outputs = tmp_path / "outputs"
    shutil.copytree(root / real_cfg.paths.data_outputs, tmp_outputs)
    tmp_reports = tmp_path / "reports"
    tmp_reports.mkdir()
    tmp_figures = tmp_reports / "figures"
    shutil.copytree(root / real_cfg.paths.figures_dir, tmp_figures)

    def mock_load_project_config(*args, **kwargs):
        cfg = real_cfg.model_copy(deep=True)
        cfg.paths.data_outputs = tmp_outputs
        cfg.paths.reports_dir = tmp_reports
        cfg.paths.figures_dir = tmp_figures
        return cfg

    # Patch load_project_config across all modules
    monkeypatch.setattr("retention_radar.config.load_project_config", mock_load_project_config)
    for mod in [
        "retention_radar.eda.analysis",
        "retention_radar.markov.chain",
        "retention_radar.attribution.markov_attribution",
        "retention_radar.clv.compare",
        "retention_radar.churn.fairness",
        "retention_radar.churn.score",
        "retention_radar.uplift.hillstrom",
        "retention_radar.uplift.x5",
        "retention_radar.uplift.dunnhumby",
        "retention_radar.uplift.power",
        "retention_radar.basket.rules",
        "retention_radar.basket.early_warning",
        "retention_radar.monitoring.drift",
        "retention_radar.schemas",
    ]:
        try:
            monkeypatch.setattr(f"{mod}.load_project_config", mock_load_project_config)
        except AttributeError:
            pass

    # 1. EDA
    eda_main()

    # 2. Markov Modeling
    markov_main()

    # 3. Attribution
    attr_main()

    # 4. CLV Benchmarking
    clv_main()

    # 5. Churn Fairness & Scoring
    churn_fairness_main()
    churn_score_main()

    # 6. Uplift Pipeline
    uplift_hillstrom_main()
    uplift_x5_main()
    uplift_dunnhumby_main()
    uplift_power_main()

    # 7. Market Basket & Early Warning
    basket_rules_main()
    basket_ew_main()

    # 8. Monitoring & Validation
    drift_main()
    schemas_main()
