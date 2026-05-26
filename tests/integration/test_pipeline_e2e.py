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
def test_pipeline_e2e_components():
    """Verify end-to-end pipeline components execute cleanly on warehouse database."""
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
