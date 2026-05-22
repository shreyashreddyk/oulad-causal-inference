# Causal Discovery and Inference for Early Online Engagement in OULAD

This repository is a graduate causal inference project for DSC 245. It studies whether high early online engagement is associated with higher course-completion success in the Open University Learning Analytics Dataset, using a reproducible Python pipeline rather than notebook state.

The project is deliberately framed as observational causal inference, not prediction. The central question is:

> Among comparable students in the same broad course context, how would the probability of course success differ under high versus lower early online engagement?

The analysis combines a hand-built DAG, bounded causal discovery, propensity-based estimation, doubly robust estimation, overlap and balance diagnostics, robustness checks, and report-ready artifacts.

## Project Snapshot

| Item | Current output |
| --- | ---: |
| Raw student-module records | 32,593 |
| Analytic cohort records | 28,128 |
| Early-unregistration exclusions | 4,465 |
| Module presentations | 22 |
| Primary treated records | 14,134 |
| Primary control records | 13,994 |
| Overall success rate | 0.547 |
| Success rate, lower engagement | 0.419 |
| Success rate, high engagement | 0.674 |

The primary analysis estimates the average treatment effect risk difference for high versus lower engagement during the first 14 days of a module presentation. High engagement is defined within module-presentation, so a student is compared against peers facing the same course and presentation context.

## Why This Is a Causal Project

A simple predictive model could learn that active students tend to pass. That is not the goal here. This project asks an intervention-style question and makes the required assumptions explicit:

- The treatment is high cumulative VLE activity during days 0 through 13.
- The outcome is course success, coded as Pass or Distinction versus Fail or Withdrawn.
- Adjustment is limited to pre-treatment student background, registration timing, planned study load, module-presentation context, and scheduled early assessment load.
- Later assessment submissions, later scores, banked assessment status, and later VLE activity are treated as post-treatment variables or mediators and are excluded from the primary adjustment set.
- Unmeasured motivation, available study time, outside support, and competing obligations remain important validity threats.

The result is not claimed as randomized evidence. It is an observational estimate under documented exchangeability, positivity, consistency, and timing assumptions.

## Identification Strategy

The hand-built DAG separates:

- Baseline confounders: demographics, region, education, deprivation band, disability, prior attempts, studied credits, registration timing, and module context.
- Scheduled course context: early assessment count, type, and weight.
- Treatment: high 14-day early engagement.
- Mediating processes: later assessment behavior and later participation.
- Outcome: final course success.

The primary adjustment set comes from `oulad_causal.dag.recommended_baseline_adjustment_set()`. It is intentionally conservative about timing: variables that could be affected by early engagement are not adjusted for in the main estimate.

Key artifacts:

- `data/processed/primary_dag.yaml`
- `reports/figures/primary_dag.png`
- `docs/identification_plan.md`

## Main Results

The preferred estimator is AIPW, reported alongside regression adjustment and stabilized IPTW. All three successful estimators point to a similar positive risk difference.

| Estimator | Risk difference | 95% CI | Notes |
| --- | ---: | ---: | --- |
| Regression adjustment | 0.234926 | [0.234157, 0.235694] | Separate logistic outcome regressions |
| Stabilized IPTW | 0.233985 | [0.216530, 0.251439] | Stabilized weights, no default truncation |
| AIPW | 0.235260 | [0.224048, 0.246471] | Preferred doubly robust estimate |

Interpretation: under the documented observational assumptions, high early engagement is associated with an estimated increase of about 23.5 percentage points in course-success probability. This should be read as assumption-dependent evidence, not as proof that increasing clicks alone would mechanically cause the same improvement.

The raw outcome contrast is directionally consistent with the adjusted estimates: 67.4 percent success among high-engagement records versus 41.9 percent among lower-engagement records. The adjusted models refine this comparison using the DAG-based baseline covariates.

## Diagnostics

The project does not hide diagnostic warnings. The overlap flag is raised because the treatment model includes a small number of extreme propensity scores:

| Diagnostic | Value |
| --- | ---: |
| Propensity score minimum | 0.000240 |
| Propensity score maximum | 0.853557 |
| Scores clipped for finite arithmetic | 5 |
| Common-support outside share | 0.000498 |
| Effective sample size after weighting | 26,694.30 |
| Maximum weighted absolute SMD | 0.022340 |

This is a useful diagnostic pattern: overlap is not perfect, but the common-support violation is tiny and post-weighting balance is strong. Nearest-neighbor matching is skipped rather than forced because the overlap gate fails.

Key artifacts:

- `data/processed/effect_estimates_main.csv`
- `data/processed/balance_table_main.csv`
- `data/processed/estimation_run_metadata.json`
- `reports/figures/overlap_plot.png`
- `reports/figures/love_plot_main.png`

## Robustness Checks

The robustness stage varies both the engagement window and the threshold used to define high engagement. Across the 7-day, 14-day, and 21-day windows crossed with median, top-tertile, and top-quartile thresholds, the AIPW risk differences range from 0.194479 to 0.236357.

| Window | Threshold | AIPW risk difference |
| ---: | --- | ---: |
| 7 days | Median | 0.203618 |
| 7 days | Top tertile | 0.194479 |
| 7 days | Top quartile | 0.196320 |
| 14 days | Median | 0.235260 |
| 14 days | Top tertile | 0.218716 |
| 14 days | Top quartile | 0.218022 |
| 21 days | Median | 0.236357 |
| 21 days | Top tertile | 0.224888 |
| 21 days | Top quartile | 0.219158 |

The project also includes module-presentation checks, subgroup checks by education, prior attempts, and disability, placebo-style checks on pre-treatment quantities, and an illustrative additive sensitivity grid. These are diagnostic and interpretive aids, not additional proof of causality.

Key artifacts:

- `data/processed/robustness_estimates_long.csv`
- `reports/tables/robustness_summary.csv`
- `reports/tables/robustness_environment_summary.csv`
- `reports/tables/robustness_subgroup_placebo_sensitivity_summary.csv`
- `reports/figures/robustness_window_threshold_heatmap.png`

## Causal Discovery

Causal discovery is used as exploratory support for the DAG, not as a replacement for identification assumptions. The workflow runs PC, FCI, and GES on a reduced 12-variable discretized matrix. This keeps the discovery task course-aligned and interpretable while avoiding a large mixed-type graph that would be difficult to defend.

The refreshed run completed successfully:

| Method | Status | Edges |
| --- | --- | ---: |
| PC | Success | 27 |
| FCI | Success | 25 |
| GES | Success | 19 |

Discovery output is compared against the hand-built DAG skeleton and repeated-subsample stability checks. Stable agreements are useful supporting evidence, while missing or unstable edges are treated as limitations rather than contradictions that automatically rewrite the causal story.

Key artifacts:

- `data/processed/discovery_edges.csv`
- `data/processed/discovery_stability_edges.csv`
- `data/processed/discovery_hand_dag_comparison.csv`
- `docs/discovery_summary.md`
- `reports/figures/discovery_comparison.png`

## Reproducible Pipeline

Core logic lives in `src/oulad_causal/`. Scripts are deterministic entry points. Notebooks are review surfaces for saved artifacts.

Pipeline stages:

1. Raw OULAD validation.
2. Analytic cohort construction.
3. DAG artifact generation.
4. Bounded causal discovery.
5. Primary effect estimation.
6. Robustness, subgroup, placebo, and sensitivity checks.
7. Final report assets and repository health check.

Every stage writes artifacts to disk. The treatment threshold cutoffs are saved explicitly in `data/processed/treatment_threshold_cutoffs.csv`, so the binary engagement definitions can be audited by module-presentation, window, and threshold.

## Repository Layout

```text
src/oulad_causal/        Reusable cohort, DAG, discovery, estimation, robustness, and visualization logic
scripts/                 Deterministic command-line pipeline stages
notebooks/               Executed review notebooks over saved artifacts
data/metadata/           Raw-data validation outputs
data/processed/          Generated analytic and model artifacts
reports/figures/         Generated figures
reports/tables/          Generated report tables
reports/drafts/          Generated interpretation aids
docs/                    Identification, data dictionary, decisions, and generated summaries
tests/                   Unit and pipeline tests
```

Generated data and figures are local artifacts and are gitignored unless explicitly whitelisted. The raw OULAD archive is also local-only.

## How To Reproduce

Use Python 3.10 or newer. Place the official OULAD archive at:

```text
data/raw/anonymisedData.zip
```

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the full pipeline:

```bash
make all
```

Run verification:

```bash
make health-check
make test
make lint
```

Populate notebooks:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_dag_and_discovery_review.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_effect_estimation_review.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_presentation_figures.ipynb
```

The presentation notebook may create slide-friendly copies under `reports/figures/slides/`. That folder is separate from the main figure folder and remains ignored by git.

## Limitations and Next Steps

- The analysis remains observational and cannot remove unmeasured motivation or time-availability confounding.
- VLE clicks measure platform activity quantity, not necessarily learning quality.
- The primary estimate is a total-effect estimand; it intentionally does not adjust for later mediators.
- Overlap is flagged and should be discussed in the final report.
- Future work should add stronger formal sensitivity analysis, consider trimmed-weight specifications, and report uncertainty for robustness-grid estimates.

The value of the project is the complete causal workflow: explicit assumptions, reproducible feature construction, discovery as a diagnostic companion, doubly robust estimation, honest diagnostics, and report-ready outputs.
