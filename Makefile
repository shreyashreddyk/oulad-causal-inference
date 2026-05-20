.PHONY: install test lint audit cohort discovery estimation robustness assets clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall src scripts

audit:
	$(PYTHON) scripts/run_data_validation.py

cohort:
	$(PYTHON) scripts/build_cohort.py

discovery:
	$(PYTHON) scripts/run_discovery.py

estimation:
	$(PYTHON) scripts/run_estimation.py

robustness:
	$(PYTHON) scripts/run_robustness.py

assets:
	$(PYTHON) scripts/build_report_assets.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
