.PHONY: setup data dbt models app test lint all reports fixture clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

setup:
	uv sync --extra dev

app:
	.venv/bin/streamlit run app/Home.py

data:
	$(PYTHON) -m retention_radar.ingest

dbt:
	cd dbt && ../.venv/bin/dbt build --profiles-dir .

models:
	$(PYTHON) -m retention_radar.pipeline --skip-ingest --skip-dbt

all:
	$(PYTHON) -m retention_radar.pipeline
	$(PYTHON) -m retention_radar.reporting.generate_notebook

reports:
	$(PYTHON) -m retention_radar.reporting.evaluation
	$(PYTHON) -m retention_radar.reporting.numbers
	$(PYTHON) -m retention_radar.reporting.generate_html_report
	$(PYTHON) -m retention_radar.reporting.generate_notebook

fixture:
	$(PYTHON) scripts/verify_fixture.py

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .
	$(RUFF) format --check .

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
