# OULAD Causal Inference Project

This repository contains the reproducible code pipeline for the DSC 245 project:

**Causal Discovery and Inference for Early Online Engagement and Course Completion in OULAD**

The project estimates the observational effect of high early online engagement on course success in the Open University Learning Analytics Dataset. Core logic lives in `src/oulad_causal/`; scripts are deterministic pipeline entry points; notebooks are only for review and interpretation of saved outputs.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the test suite:

```bash
.venv/bin/python -m pytest
```

The Makefile automatically uses `.venv/bin/python` when it exists.

## Raw Data

Raw OULAD data are local-only and intentionally gitignored. Put the official archive here:

```text
data/raw/anonymisedData.zip
```

Alternatively, extract the seven standard OULAD CSV files directly under `data/raw/`:

- `courses.csv`
- `assessments.csv`
- `vle.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentAssessment.csv`
- `studentVle.csv`

You can also point the pipeline at another raw data directory:

```bash
OULAD_RAW_DATA_DIR=/path/to/raw make validate-data
```

or pass `--raw-source` to the individual scripts.

## Run The Pipeline

Run stages one at a time:

```bash
make validate-data
make build-cohort
make run-discovery
make run-estimation
make run-robustness
make build-assets
make health-check
```

Run everything in order:

```bash
make all
```

`make all` first checks whether the raw OULAD files are available. If they are missing, it stops with placement instructions rather than producing partial or fabricated outputs.

## Outputs

Pipeline artifacts are written under:

- `data/metadata/`: raw-data validation summaries.
- `data/processed/`: cohort, DAG, discovery, estimation, and machine-readable metadata.
- `reports/figures/`: generated figures.
- `reports/tables/`: generated report tables.
- `reports/drafts/`: generated report-writing aids.
- `docs/`: generated summaries, identification notes, decisions, and the reproducibility runbook.

Use `make health-check` after any full run to verify that the expected artifact inventory exists.

## Reproduction Source Of Truth

See `docs/reproducibility_runbook.md` for the final run order, expected artifact inventory, raw-data placement rules, and manual steps still required from the human team.
