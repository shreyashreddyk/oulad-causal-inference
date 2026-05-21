"""Build final report-ready figures, tables, and presentation notes from saved artifacts."""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/oulad_causal_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/oulad_causal_xdg_cache")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TABLES_DIR
from oulad_causal.logging_utils import add_log_level_argument, configure_logging
from oulad_causal.viz import (
    cohort_flow_report_table,
    ensure_report_dirs,
    main_effect_report_table,
    robustness_report_table,
    write_discovery_comparison_figure,
    write_subgroup_summary_figure,
    write_treatment_prevalence_figure,
)


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build deterministic report assets from existing saved outputs."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    source_paths = _source_paths(args)
    final_artifacts = _final_artifacts(args)
    ensure_report_dirs(
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )
    _require_inputs(
        [
            source_paths["cohort_flow"],
            source_paths["cohort"],
            source_paths["discovery_comparison"],
            source_paths["discovery_stability"],
            source_paths["effect_estimates"],
            source_paths["robustness_window"],
            source_paths["robustness_subgroup"],
            source_paths["primary_dag_figure"],
            source_paths["overlap_figure"],
        ]
    )

    cohort_flow = pd.read_csv(source_paths["cohort_flow"])
    cohort = pd.read_parquet(source_paths["cohort"])
    discovery_comparison = pd.read_csv(source_paths["discovery_comparison"])
    discovery_stability = pd.read_csv(source_paths["discovery_stability"])
    effect_estimates = pd.read_csv(source_paths["effect_estimates"])
    robustness_window = pd.read_csv(source_paths["robustness_window"])
    robustness_subgroup = pd.read_csv(source_paths["robustness_subgroup"])

    paths: dict[str, Path] = {}
    paths["dag_figure"] = source_paths["primary_dag_figure"]
    paths["overlap_plot"] = source_paths["overlap_figure"]
    cohort_flow_report_table(cohort_flow, final_artifacts["cohort_flow_table"])
    paths["cohort_flow_table"] = final_artifacts["cohort_flow_table"]
    write_treatment_prevalence_figure(cohort, final_artifacts["treatment_prevalence_figure"])
    paths["treatment_prevalence_figure"] = final_artifacts["treatment_prevalence_figure"]
    write_discovery_comparison_figure(
        discovery_comparison,
        discovery_stability,
        final_artifacts["discovery_comparison_figure"],
    )
    paths["discovery_comparison_figure"] = final_artifacts["discovery_comparison_figure"]
    main_effect_report_table(effect_estimates, final_artifacts["main_effect_estimates_table"])
    paths["main_effect_estimates_table"] = final_artifacts["main_effect_estimates_table"]
    robustness_report_table(robustness_window, final_artifacts["robustness_summary_table"])
    paths["robustness_summary_table"] = final_artifacts["robustness_summary_table"]
    write_subgroup_summary_figure(robustness_subgroup, final_artifacts["subgroup_summary_figure"])
    paths["subgroup_summary_figure"] = final_artifacts["subgroup_summary_figure"]

    _write_results_walkthrough(final_artifacts["results_walkthrough"])
    paths["results_walkthrough"] = final_artifacts["results_walkthrough"]
    _write_presentation_asset_plan(final_artifacts["presentation_asset_plan"])
    paths["presentation_asset_plan"] = final_artifacts["presentation_asset_plan"]

    LOGGER.info("Wrote final report assets:")
    for name, path in paths.items():
        LOGGER.info("- %s: %s", name, path)
    return 0


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "cohort_flow": args.processed_dir / "cohort_flow_table.csv",
        "cohort": args.processed_dir / "oulad_analytic_cohort.parquet",
        "discovery_comparison": args.processed_dir / "discovery_hand_dag_comparison.csv",
        "discovery_stability": args.processed_dir / "discovery_stability_edges.csv",
        "effect_estimates": args.processed_dir / "effect_estimates_main.csv",
        "robustness_window": args.tables_dir / "robustness_window_threshold_summary.csv",
        "robustness_subgroup": args.tables_dir / "robustness_subgroup_placebo_sensitivity_summary.csv",
        "primary_dag_figure": args.figures_dir / "primary_dag.png",
        "overlap_figure": args.figures_dir / "overlap_plot.png",
    }


def _final_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "cohort_flow_table": args.tables_dir / "cohort_flow.csv",
        "treatment_prevalence_figure": args.figures_dir / "treatment_prevalence.png",
        "discovery_comparison_figure": args.figures_dir / "discovery_comparison.png",
        "main_effect_estimates_table": args.tables_dir / "main_effect_estimates.csv",
        "robustness_summary_table": args.tables_dir / "robustness_summary.csv",
        "subgroup_summary_figure": args.figures_dir / "subgroup_summary.png",
        "results_walkthrough": args.reports_dir / "drafts" / "results_walkthrough.md",
        "presentation_asset_plan": args.docs_dir / "presentation_asset_plan.md",
    }


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
    raise SystemExit(main())
