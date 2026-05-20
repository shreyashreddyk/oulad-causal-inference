# OULAD Causal Inference Project

This repository is a Python package-centered project for the DSC 245 course project:

**Causal Discovery and Inference for Early Online Engagement and Course Completion in OULAD**

The goal is to estimate the causal effect of high early online engagement on course success in the Open University Learning Analytics Dataset (OULAD), with a limited but meaningful causal discovery component used to interrogate and stress-test a domain-informed DAG.

This repository currently contains the reproducible project skeleton only. It does not assume that raw OULAD files are present.

## Intended Pipeline

1. **Data audit**
   - Inspect raw OULAD tables after they are obtained.
   - Verify schemas, missingness, outcome labels, date ranges, and module-presentation structure.
   - Expected artifacts: audit notes, metadata summaries, validation logs.

2. **Cohort construction**
   - Build a student-module-presentation analysis cohort.
   - Define eligible observations and preserve exclusion counts.
   - Expected artifacts: `data/interim/` cohort extracts and `data/metadata/` cohort documentation.

3. **Feature construction**
   - Construct pre-treatment covariates, registration timing features, module-presentation identifiers, and early VLE engagement measures.
   - Define the primary treatment as high early engagement during the first 14 days, normalized within module-presentation.
   - Prepare robustness variants for 7-day and 21-day windows.
   - Expected artifacts: cleaned feature matrices and feature dictionaries.

4. **DAG specification**
   - Encode a domain-informed DAG separating confounders, treatment, mediators, outcomes, and ambiguous variables.
   - Identify the primary adjustment set before estimation.
   - Expected artifacts: DAG source file, DAG figure, adjustment-set notes.

5. **Causal discovery review**
   - Run limited discovery methods on a reduced variable set, such as PC, FCI, and/or GES.
   - Compare discovered graphs to the hand-built DAG without treating discovery output as definitive truth.
   - Expected artifacts: discovery graph files, stability summaries, comparison notes.

6. **Effect estimation**
   - Estimate the effect of high early engagement on course success using regression adjustment, IPTW, and doubly robust/AIPW-style estimators where appropriate.
   - Report overlap, balance, and model diagnostics.
   - Expected artifacts: treatment effect tables, diagnostic plots, model summaries.

7. **Robustness and subgroup analysis**
   - Repeat core estimates across engagement windows, module presentations, and pre-specified subgroups such as prior education or previous attempts.
   - Expected artifacts: robustness tables, subgroup figures, sensitivity notes.

8. **Report and presentation assets**
   - Generate final report-ready figures and presentation-ready tables directly from saved outputs.
   - Expected artifacts: files under `reports/figures/` and `reports/tables/`.

## Reproducibility

Install the package and dependencies in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run checks:

```bash
make test
```

Raw data should be stored locally under `data/raw/` when available. That directory is intentionally gitignored.

