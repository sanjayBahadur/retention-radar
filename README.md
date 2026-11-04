# Retention Radar

I analyzed purchase histories for 2,500 grocery households to explore where a retailer could focus a limited retention budget. Three findings shaped the project:

- **The highest-risk fifth contained 75% of later departures.** Selecting 20% of households at each test cutoff identified 534 of 712 departures across the evaluation periods. A departure means eight weeks without purchases.
- **Recent spending was a strong customer-value baseline.** Using the previous 26 weeks of spend produced a lower average forecast error than the more complex forecasting models in the later 26-week comparison.
- **Category shares changed around dormancy.** Diapers and poultry showed declines of 3.61 and 3.33 percentage points relative to the active comparison group.

**[Explore the dashboard](https://sanjaybahadur.github.io/retention-radar/)** · [Findings and proposed test](reports/executive_summary.md) · [Technical notes](reports/technical_report.md)

The project uses Dunnhumby's *The Complete Journey*, with SQL and dbt models in DuckDB, Python analysis, and a Streamlit application. The dashboard follows the findings from shopping behavior to risk ranking and an outreach cost scenario. Historical campaign analysis adds an exploratory view of response to offers.

The cost scenario uses forecast margin, contact cost, and an assumed save rate. [Generated results](data/outputs/results.json), [evaluation details](reports/evaluation.md), and [model notes](reports/model_cards/) provide the numbers and methods behind the charts.

## Run locally

The linked dashboard opens directly in a browser. To run the Streamlit application:

```bash
git clone https://github.com/sanjayBahadur/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make app
```

Open `http://localhost:8501`. Included outputs are enough to explore the application. Alternatively, run `docker compose up --build`.

## Development

```bash
make lint
make test
make fixture  # build and check a small synthetic dataset without credentials
make reports  # rebuild reports and the dashboard from existing outputs
```

With the original raw datasets available, `make all` rebuilds the analysis. Financial assumptions are in [project.yaml](configs/project.yaml), and lifecycle definitions are in [states.yaml](configs/states.yaml). The [technical notes](reports/technical_report.md) explain evaluation windows, assumptions, and the full build sequence.

## Code and analysis

| Resource | Contents |
| --- | --- |
| [SQL analysis](dbt/analyses/) | Cohorts, revenue concentration, baskets, and campaign reach |
| [dbt models](dbt/models/) | Staging, basket aggregation, and household lifecycle tables |
| [Dashboard source](app/Home.py) | Guided overview and interactive exploration |
| [Evaluation review](reports/evaluation.md) | Missed departures, false alarms, and results across periods |
| [Model cards](reports/model_cards/) | Churn, customer value, and historical campaign response |
| [BI export guide](bi/README.md) | Tableau and Power BI connection examples |
| [Data dictionary](docs/data_dictionary.md) | Table grains, fields, and lifecycle definitions |

Built by [Sanjay Bahadur](https://github.com/sanjayBahadur). Code: [MIT License](LICENSE). Dataset terms: [data use note](docs/data_governance.md).
