"""Build the cleaned OULAD analysis cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/oulad_causal_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/oulad_causal_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.cohort import CohortConfig, build_analytic_cohort
from oulad_causal.config import FIGURES_DIR, PROCESSED_DATA_DIR, ProjectPaths
from oulad_causal.features import treatment_column_name
from oulad_causal.io import load_oulad_tables


def main() -> None:
    """Build the analytic cohort and save reproducible artifacts."""

    args = parse_args()
    paths = ProjectPaths.from_overrides(raw_data_dir=args.raw_data_dir)
    raw_source = Path(args.raw_source).expanduser().resolve() if args.raw_source else None
    processed_dir = _resolve_output_dir(args.processed_dir, PROCESSED_DATA_DIR)
    figures_dir = _resolve_output_dir(args.figures_dir, FIGURES_DIR)

    tables = load_oulad_tables(raw_source=raw_source, raw_data_dir=paths.raw_data_dir)
    config = CohortConfig(primary_window=args.primary_window)
    result = build_analytic_cohort(tables, config=config)

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = processed_dir / "oulad_analytic_cohort.parquet"
    flow_path = processed_dir / "cohort_flow_table.csv"
    summary_path = processed_dir / "cohort_summary.json"

    result.cohort.to_parquet(cohort_path, index=False)
    result.flow_table.to_csv(flow_path, index=False)
    summary = {
        **result.summary,
        "artifact_paths": {
            "analytic_cohort": str(cohort_path),
            "cohort_flow_table": str(flow_path),
            "cohort_summary": str(summary_path),
            "cohort_flow_plot": str(figures_dir / "cohort_flow.png"),
            "treatment_prevalence_plot": str(figures_dir / "treatment_prevalence.png"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    write_cohort_flow_plot(result.flow_table, figures_dir / "cohort_flow.png")
    write_treatment_prevalence_plot(
        result.cohort,
        window_days=config.primary_window,
        figures_path=figures_dir / "treatment_prevalence.png",
    )

    print(f"Wrote analytic cohort to {cohort_path}")
    print(f"Wrote cohort flow table to {flow_path}")
    print(f"Wrote cohort summary to {summary_path}")
    print(f"Wrote figures to {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source", help="Explicit OULAD archive or extracted CSV directory.")
    parser.add_argument("--raw-data-dir", help="Raw data directory. Defaults to data/raw or OULAD_RAW_DATA_DIR.")
    parser.add_argument("--processed-dir", help="Directory for processed cohort outputs.")
    parser.add_argument("--figures-dir", help="Directory for cohort diagnostic figures.")
    parser.add_argument("--primary-window", type=int, default=14, help="Primary treatment window in days.")
    return parser.parse_args()


def write_cohort_flow_plot(flow_table: pd.DataFrame, figures_path: Path) -> None:
    """Write a simple cohort row-count plot."""

    ax = flow_table.plot.barh(x="stage", y="row_count", legend=False, figsize=(8, 3.5))
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.set_title("Analytic Cohort Flow")
    plt.tight_layout()
    ax.figure.savefig(figures_path, dpi=160)
    plt.close(ax.figure)


def write_treatment_prevalence_plot(
    cohort: pd.DataFrame,
    *,
    window_days: int,
    figures_path: Path,
) -> None:
    """Write treatment prevalence for the primary early-engagement window."""

    rows = []
    for threshold_name in ["median", "top_tertile", "top_quartile"]:
        column = treatment_column_name(window_days, threshold_name)
        values = cohort[column].dropna()
        rows.append(
            {
                "threshold": threshold_name,
                "prevalence": float(values.mean()) if not values.empty else 0.0,
            }
        )
    plot_data = pd.DataFrame(rows)
    ax = plot_data.plot.bar(x="threshold", y="prevalence", legend=False, figsize=(6, 3.5))
    ax.set_xlabel("")
    ax.set_ylabel("Prevalence")
    ax.set_ylim(0, 1)
    ax.set_title(f"High Engagement Prevalence, First {window_days} Days")
    plt.xticks(rotation=0)
    plt.tight_layout()
    ax.figure.savefig(figures_path, dpi=160)
    plt.close(ax.figure)


def _resolve_output_dir(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


if __name__ == "__main__":
    main()
