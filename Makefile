.PHONY: setup data dbt models app test lint all clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
STREAMLIT := .venv/bin/streamlit

setup:
	uv sync --all-extras
	uv pip install -e .

data:
	$(PYTHON) -m retention_radar.ingest

dbt:
	cd dbt && dbt deps && dbt build --profiles-dir .

models:
	$(PYTHON) -m retention_radar.markov.chain
	$(PYTHON) -m retention_radar.attribution.markov_attribution
	$(PYTHON) -m retention_radar.clv.compare
	$(PYTHON) -m retention_radar.features.build_features
	$(PYTHON) -m retention_radar.churn.train
	$(PYTHON) -m retention_radar.churn.fairness
	$(PYTHON) -m retention_radar.churn.score
	$(PYTHON) -m retention_radar.uplift.hillstrom
	$(PYTHON) -m retention_radar.uplift.x5
	$(PYTHON) -m retention_radar.uplift.dunnhumby
	$(PYTHON) -m retention_radar.uplift.power
	$(PYTHON) -m retention_radar.basket.rules
	$(PYTHON) -m retention_radar.basket.early_warning

app:
	$(STREAMLIT) run app/Home.py

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .
	$(RUFF) format --check .

all: data dbt models
	$(PYTHON) -m retention_radar.pipeline
	$(PYTHON) -m retention_radar.reporting.numbers

clean:
	rm -rf data/warehouse.duckdb*
	rm -rf data/outputs/*
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
