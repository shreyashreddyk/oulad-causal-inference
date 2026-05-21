"""Build final report-ready figures, tables, and presentation notes from saved artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/oulad_causal_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/oulad_causal_xdg_cache")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TABLES_DIR
from oulad_causal.viz import (
    cohort_flow_report_table,
    ensure_report_dirs,
    main_effect_report_table,
    robustness_report_table,
    write_discovery_comparison_figure,
    write_subgroup_summary_figure,
    write_treatment_prevalence_figure,
)


COHORT_FLOW_SOURCE = PROCESSED_DATA_DIR / "cohort_flow_table.csv"
COHORT_SOURCE = PROCESSED_DATA_DIR / "oulad_analytic_cohort.parquet"
DISCOVERY_COMPARISON_SOURCE = PROCESSED_DATA_DIR / "discovery_hand_dag_comparison.csv"
DISCOVERY_STABILITY_SOURCE = PROCESSED_DATA_DIR / "discovery_stability_edges.csv"
EFFECT_ESTIMATES_SOURCE = PROCESSED_DATA_DIR / "effect_estimates_main.csv"
ROBUSTNESS_WINDOW_SOURCE = TABLES_DIR / "robustness_window_threshold_summary.csv"
ROBUSTNESS_SUBGROUP_SOURCE = TABLES_DIR / "robustness_subgroup_placebo_sensitivity_summary.csv"
PRIMARY_DAG_FIGURE = FIGURES_DIR / "primary_dag.png"
OVERLAP_FIGURE = FIGURES_DIR / "overlap_plot.png"

FINAL_ARTIFACTS = {
    "dag_figure": FIGURES_DIR / "primary_dag.png",
    "cohort_flow_table": TABLES_DIR / "cohort_flow.csv",
    "treatment_prevalence_figure": FIGURES_DIR / "treatment_prevalence.png",
    "discovery_comparison_figure": FIGURES_DIR / "discovery_comparison.png",
    "overlap_plot": FIGURES_DIR / "overlap_plot.png",
    "main_effect_estimates_table": TABLES_DIR / "main_effect_estimates.csv",
    "robustness_summary_table": TABLES_DIR / "robustness_summary.csv",
    "subgroup_summary_figure": FIGURES_DIR / "subgroup_summary.png",
    "results_walkthrough": REPORTS_DIR / "drafts" / "results_walkthrough.md",
    "presentation_asset_plan": DOCS_DIR / "presentation_asset_plan.md",
}


def main() -> None:
    """Build deterministic report assets from existing saved outputs."""

    ensure_report_dirs()
    _require_inputs(
        [
            COHORT_FLOW_SOURCE,
            COHORT_SOURCE,
            DISCOVERY_COMPARISON_SOURCE,
            DISCOVERY_STABILITY_SOURCE,
            EFFECT_ESTIMATES_SOURCE,
            ROBUSTNESS_WINDOW_SOURCE,
            ROBUSTNESS_SUBGROUP_SOURCE,
            PRIMARY_DAG_FIGURE,
            OVERLAP_FIGURE,
        ]
    )

    cohort_flow = pd.read_csv(COHORT_FLOW_SOURCE)
    cohort = pd.read_parquet(COHORT_SOURCE)
    discovery_comparison = pd.read_csv(DISCOVERY_COMPARISON_SOURCE)
    discovery_stability = pd.read_csv(DISCOVERY_STABILITY_SOURCE)
    effect_estimates = pd.read_csv(EFFECT_ESTIMATES_SOURCE)
    robustness_window = pd.read_csv(ROBUSTNESS_WINDOW_SOURCE)
    robustness_subgroup = pd.read_csv(ROBUSTNESS_SUBGROUP_SOURCE)

    paths: dict[str, Path] = {}
    paths["dag_figure"] = PRIMARY_DAG_FIGURE
    paths["overlap_plot"] = OVERLAP_FIGURE
    cohort_flow_report_table(cohort_flow, FINAL_ARTIFACTS["cohort_flow_table"])
    paths["cohort_flow_table"] = FINAL_ARTIFACTS["cohort_flow_table"]
    write_treatment_prevalence_figure(cohort, FINAL_ARTIFACTS["treatment_prevalence_figure"])
    paths["treatment_prevalence_figure"] = FINAL_ARTIFACTS["treatment_prevalence_figure"]
    write_discovery_comparison_figure(
        discovery_comparison,
        discovery_stability,
        FINAL_ARTIFACTS["discovery_comparison_figure"],
    )
    paths["discovery_comparison_figure"] = FINAL_ARTIFACTS["discovery_comparison_figure"]
    main_effect_report_table(effect_estimates, FINAL_ARTIFACTS["main_effect_estimates_table"])
    paths["main_effect_estimates_table"] = FINAL_ARTIFACTS["main_effect_estimates_table"]
    robustness_report_table(robustness_window, FINAL_ARTIFACTS["robustness_summary_table"])
    paths["robustness_summary_table"] = FINAL_ARTIFACTS["robustness_summary_table"]
    write_subgroup_summary_figure(robustness_subgroup, FINAL_ARTIFACTS["subgroup_summary_figure"])
    paths["subgroup_summary_figure"] = FINAL_ARTIFACTS["subgroup_summary_figure"]

    _write_results_walkthrough(FINAL_ARTIFACTS["results_walkthrough"])
    paths["results_walkthrough"] = FINAL_ARTIFACTS["results_walkthrough"]
    _write_presentation_asset_plan(FINAL_ARTIFACTS["presentation_asset_plan"])
    paths["presentation_asset_plan"] = FINAL_ARTIFACTS["presentation_asset_plan"]

    print("Wrote final report assets:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def _require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing saved input artifacts. Run upstream stages first:\n{formatted}")


def _write_results_walkthrough(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Results Walkthrough Draft

This draft lists the final report and presentation assets generated from saved pipeline outputs. Keep interpretation cautious and replace placeholders only after human review.

## DAG figure

- Artifact: `reports/figures/primary_dag.png`
- Sources: `reports/figures/primary_dag.png`, `data/processed/primary_dag.yaml`
- Shows: the domain-informed DAG separating baseline covariates, treatment, mediating post-treatment processes, and course success.
- Interpretation placeholder: Explain why the primary adjustment set uses pre-treatment and scheduled course-context variables only.

## Cohort flow table

- Artifact: `reports/tables/cohort_flow.csv`
- Source: `data/processed/cohort_flow_table.csv`
- Shows: the row counts retained after required joins and the primary treatment-eligibility exclusion.
- Interpretation placeholder: Summarize cohort construction and note any exclusions relevant to external validity.

## Treatment prevalence figure

- Artifact: `reports/figures/treatment_prevalence.png`
- Sources: `data/processed/oulad_analytic_cohort.parquet`
- Shows: the share of records classified as high engagement under the median, top-tertile, and top-quartile thresholds for the first 14 days.
- Interpretation placeholder: Explain the threshold definitions without implying treatment assignment was randomized.

## Discovery comparison figure

- Artifact: `reports/figures/discovery_comparison.png`
- Sources: `data/processed/discovery_hand_dag_comparison.csv`, `data/processed/discovery_stability_edges.csv`
- Shows: how many discovered skeleton edges overlap with the hand-built DAG and summary counts for unrecovered hand-DAG edges and stable repeated-subsample edges.
- Interpretation placeholder: Describe discovery as exploratory support, not a replacement for the identification plan.

## Overlap plot

- Artifact: `reports/figures/overlap_plot.png`
- Source: `reports/figures/overlap_plot.png`
- Shows: estimated propensity-score distributions for high versus lower early engagement groups.
- Interpretation placeholder: Discuss overlap diagnostics and the flagged limitations before interpreting the estimates.

## Main effect estimates table

- Artifact: `reports/tables/main_effect_estimates.csv`
- Source: `data/processed/effect_estimates_main.csv`
- Shows: regression adjustment, stabilized IPTW, preferred AIPW, and matching status for the primary risk-difference estimand.
- Interpretation placeholder: State the preferred estimate and describe it as observational under the documented assumptions.

## Robustness summary table

- Artifact: `reports/tables/robustness_summary.csv`
- Source: `reports/tables/robustness_window_threshold_summary.csv`
- Shows: AIPW estimates across early-engagement windows and treatment thresholds.
- Interpretation placeholder: Identify patterns across definitions while avoiding inflated robustness claims.

## Subgroup summary figure

- Artifact: `reports/figures/subgroup_summary.png`
- Source: `reports/tables/robustness_subgroup_placebo_sensitivity_summary.csv`
- Shows: successful subgroup estimates for pre-specified subgroup variables that passed adequacy gates.
- Interpretation placeholder: Treat subgroup differences as descriptive robustness checks, not definitive heterogeneity.
""",
        encoding="utf-8",
    )


def _write_presentation_asset_plan(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Presentation Asset Checklist

- [ ] Problem and estimand slide: `reports/figures/primary_dag.png`
- [ ] Cohort construction slide: `reports/tables/cohort_flow.csv`
- [ ] Treatment definition slide: `reports/figures/treatment_prevalence.png`
- [ ] Discovery review slide: `reports/figures/discovery_comparison.png`
- [ ] Overlap and diagnostics slide: `reports/figures/overlap_plot.png`
- [ ] Main results slide: `reports/tables/main_effect_estimates.csv`
- [ ] Robustness slide: `reports/tables/robustness_summary.csv`
- [ ] Subgroup checks slide: `reports/figures/subgroup_summary.png`

Notes:

- Keep titles neutral and report-friendly.
- Use `reports/drafts/results_walkthrough.md` for interpretation placeholders.
- Do not add claims that are not supported by saved pipeline outputs.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
