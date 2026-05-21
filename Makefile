.PHONY: install test lint audit validate-data build-cohort cohort discovery run-discovery estimation run-estimation robustness run-robustness build-assets assets health-check all clean

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall src scripts

audit:
	$(PYTHON) scripts/run_data_validation.py

validate-data:
	$(PYTHON) scripts/run_data_validation.py

build-cohort:
	$(PYTHON) scripts/build_cohort.py

cohort: build-cohort

discovery:
	$(PYTHON) scripts/run_discovery.py

run-discovery: discovery

estimation:
	$(PYTHON) scripts/run_estimation.py

run-estimation: estimation

robustness:
	$(PYTHON) scripts/run_robustness.py

run-robustness: robustness

assets:
	$(PYTHON) scripts/build_report_assets.py

build-assets: assets

health-check:
	$(PYTHON) scripts/check_repo_health.py

all:
	$(PYTHON) scripts/run_pipeline.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
