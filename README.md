# Retention Radar

**Who should a retailer contact when its retention budget is limited?**

Retention Radar explores shopping patterns across 2,500 grocery households in Dunnhumby's *The Complete Journey*. It brings SQL analysis, customer-risk prediction, and a simple decision dashboard together to examine that question.

**[Explore the visual case study](https://sanjaybahadur.github.io/retention-radar/)** · [Short findings](reports/executive_summary.md) · [Technical notes](reports/technical_report.md)

## What the project does

1. **Understand shopping changes.** Compare household activity and category spending over time.
2. **Rank departure risk.** Estimate which households will make no purchases in the next eight weeks, comparing a tree model with simpler baselines.
3. **Explore an outreach decision.** Combine predicted risk, estimated customer value, and explicit cost assumptions. Examine historical campaign response as a separate, observational analysis.
4. **Design a test.** Propose a randomized experiment before treating estimated savings as real business impact.

A risk prediction answers **“Who might stop visiting?”** It does not answer **“Whose behavior will an offer change?”** That distinction guides the analysis.

## What to take away

- Ranking households by purchase history captures roughly three quarters of later departures within a 20% contact budget at each test cutoff. The comparison includes simple baselines and household-clustered uncertainty estimates.
- Recent spending beat the more complex customer-value forecasts on mean absolute error in this temporal comparison. More complexity did not automatically improve the result.
- Category spending changes are useful clues to investigate. They do not establish competitor switching or prove that a category-specific offer would prevent departure.
- Estimated outreach returns depend on forecast customer value, contact costs, and an assumed save rate. They are scenarios, not money earned by a deployed campaign.
- Historical campaigns were not randomly assigned. Their estimated treatment effects require a controlled test before use in an automated contact or suppression policy.

The dashboard and [generated results](data/outputs/results.json) contain the current numbers. Keeping those figures in generated outputs avoids maintaining conflicting copies throughout the documentation.

## Try it

The visual case study works in a browser without installing Python. To explore the Streamlit application locally:

```bash
git clone https://github.com/sanjayBahadur/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make app
```

Open `http://localhost:8501`. The application reads the included analysis outputs; the original transaction data and warehouse are needed only to rebuild the full analysis.

Docker is another local option:

```bash
docker compose up --build
```

## Explore the work

| Resource | What it explains |
| --- | --- |
| [Dashboard source](app/Home.py) | Guided overview and interactive exploration |
| [SQL analysis](dbt/analyses/) | Cohorts, revenue concentration, basket size, and campaign reach |
| [dbt models](dbt/models/) | Staging, basket aggregation, and household lifecycle tables |
| [Technical notes](reports/technical_report.md) | Evaluation design, units, assumptions, and limitations |
| [Evaluation review](reports/evaluation.md) | Error rates, sample counts, and changes across time and groups |
| [Model cards](reports/model_cards/) | Intended use and evaluation of churn, customer value, and uplift models |
| [BI export guide](bi/README.md) | Table grains and examples for Tableau or Power BI |
| [Data dictionary](docs/data_dictionary.md) | Important fields and lifecycle definitions |

## Development and rebuilding

```bash
make lint
make test
make fixture  # synthetic raw data → dbt → features → models; no credentials needed
```

The full analysis uses Python 3.12, DuckDB, dbt, pandas, scikit-learn, LightGBM, and Streamlit. Financial assumptions live in [project.yaml](configs/project.yaml); lifecycle definitions live in [states.yaml](configs/states.yaml). Use `make reports` to rebuild the case study, findings, error review and walkthrough notebook from existing artifacts. With the original raw data available, `make all` rebuilds the full analysis.

See the [technical notes](reports/technical_report.md) for the build sequence and validation limits.

## Boundaries

This is a portfolio analysis, not a deployed retention service. It studies one retailer's historical data. It cannot observe purchases elsewhere, establish an individual customer's motives, or demonstrate realized campaign savings. Demographic audit groups are small, and removing demographic inputs does not establish fairness.

Code is available under the [MIT License](LICENSE). Dataset terms are separate; see the [data use note](docs/data_governance.md).

Built by [Sanjay Bahadur](https://github.com/sanjayBahadur).
