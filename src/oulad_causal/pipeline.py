"""End-to-end pipeline inventory and health-check helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from oulad_causal.config import PROJECT_ROOT, RAW_DATA_DIR


PIPELINE_STAGES: tuple[str, ...] = (
    "validation",
    "cohort",
    "discovery",
    "estimation",
    "robustness",
    "final_assets",
)

MAKE_TARGET_BY_STAGE: dict[str, str] = {
    "validation": "make validate-data",
    "cohort": "make build-cohort",
    "discovery": "make run-discovery",
    "estimation": "make run-estimation",
    "robustness": "make run-robustness",
    "final_assets": "make build-assets",
}

EXPECTED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "validation": (
        "data/metadata/raw_file_inventory.csv",
        "data/metadata/table_shapes.csv",
        "data/metadata/schema_validation.csv",
        "data/metadata/duplicate_key_summary.csv",
        "data/metadata/missingness_summary.csv",
        "data/metadata/date_range_summary.csv",
        "data/metadata/category_frequency_summary.csv",
        "data/metadata/data_validation_summary.json",
    ),
    "cohort": (
        "data/processed/oulad_analytic_cohort.parquet",
        "data/processed/cohort_flow_table.csv",
        "data/processed/cohort_summary.json",
        "data/processed/treatment_threshold_cutoffs.csv",
        "data/processed/primary_dag.yaml",
        "data/processed/dag_variable_availability.csv",
        "reports/figures/cohort_flow.png",
        "reports/figures/treatment_prevalence.png",
        "reports/figures/primary_dag.png",
    ),
    "discovery": (
        "data/processed/discovery_analysis_data.parquet",
        "data/processed/discovery_preprocessing_map.json",
        "data/processed/discovery_edges.csv",
        "data/processed/discovery_stability_edges.csv",
        "data/processed/discovery_hand_dag_comparison.csv",
        "data/processed/discovery_run_metadata.json",
        "docs/discovery_summary.md",
    ),
    "estimation": (
        "data/processed/effect_estimates_main.csv",
        "data/processed/balance_table_main.csv",
        "data/processed/estimation_run_metadata.json",
        "reports/figures/overlap_plot.png",
        "reports/figures/love_plot_main.png",
        "docs/estimation_summary.md",
    ),
    "robustness": (
        "data/processed/robustness_estimates_long.csv",
        "data/processed/robustness_run_metadata.json",
        "reports/tables/robustness_window_threshold_summary.csv",
        "reports/tables/robustness_environment_summary.csv",
        "reports/tables/robustness_subgroup_placebo_sensitivity_summary.csv",
        "reports/figures/robustness_window_threshold_heatmap.png",
        "reports/figures/robustness_module_presentation_estimates.png",
        "reports/figures/robustness_subgroup_estimates.png",
        "reports/figures/robustness_placebo_sensitivity.png",
        "docs/robustness_summary.md",
    ),
    "final_assets": (
        "reports/tables/cohort_flow.csv",
        "reports/tables/main_effect_estimates.csv",
        "reports/tables/robustness_summary.csv",
        "reports/figures/discovery_comparison.png",
        "reports/figures/subgroup_summary.png",
        "reports/drafts/results_walkthrough.md",
        "docs/presentation_asset_plan.md",
        "docs/reproducibility_runbook.md",
    ),
}


@dataclass(frozen=True)
class HealthCheckResult:
    """Grouped health-check status for expected pipeline artifacts."""

    missing_by_stage: dict[str, tuple[Path, ...]]

    @property
    def passed(self) -> bool:
        return not any(self.missing_by_stage.values())


def expected_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    stages: Iterable[str] | None = None,
) -> dict[str, tuple[Path, ...]]:
    """Return expected artifact paths grouped by pipeline stage."""

    selected = tuple(stages) if stages is not None else PIPELINE_STAGES
    _validate_stages(selected)
    root = Path(project_root)
    return {
        stage: tuple(root / relative for relative in EXPECTED_ARTIFACTS[stage])
        for stage in selected
    }


def check_expected_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    stages: Iterable[str] | None = None,
) -> HealthCheckResult:
    """Check whether expected pipeline artifacts exist."""

    grouped = expected_artifacts(project_root=project_root, stages=stages)
    return HealthCheckResult(
        missing_by_stage={
            stage: tuple(path for path in paths if not path.exists())
            for stage, paths in grouped.items()
        }
    )


def format_health_check(result: HealthCheckResult, *, project_root: Path = PROJECT_ROOT) -> str:
    """Format a human-readable health-check report."""

    if result.passed:
        return "Repository health check passed: all expected pipeline artifacts exist."

    root = Path(project_root).resolve()
    lines = ["Repository health check failed. Missing expected artifacts:"]
    for stage in PIPELINE_STAGES:
        missing = result.missing_by_stage.get(stage, ())
        if not missing:
            continue
        lines.append(f"\n[{stage}] Run `{MAKE_TARGET_BY_STAGE[stage]}`")
        for path in missing:
            lines.append(f"- {_relative_to(path, root)}")
    return "\n".join(lines)


def raw_data_available(
    *,
    raw_source: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
) -> bool:
    """Return True when all standard OULAD raw files can be located."""

    from oulad_causal.io import locate_oulad_tables

    return all(location.found for location in locate_oulad_tables(raw_source, raw_data_dir=raw_data_dir))


def raw_data_missing_message(
    *,
    raw_source: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
) -> str:
    """Return a concise placement message for missing raw OULAD data."""

    from oulad_causal.io import locate_oulad_tables, resolve_raw_source

    source = resolve_raw_source(raw_source, raw_data_dir=raw_data_dir)
    raw_dir = Path(raw_data_dir or RAW_DATA_DIR).expanduser().resolve()
    missing = [
        location.filename
        for location in locate_oulad_tables(raw_source, raw_data_dir=raw_data_dir)
        if not location.found
    ]
    missing_text = ", ".join(missing) if missing else "none"
    return (
        "Raw OULAD data are missing or incomplete.\n"
        f"Looked at: {source}\n"
        f"Missing files: {missing_text}\n"
        "Place the official OULAD archive at "
        f"{raw_dir / 'anonymisedData.zip'} or extract the seven CSV files under {raw_dir}.\n"
        "You can also pass --raw-source or set OULAD_RAW_DATA_DIR."
    )


def stage_names(value: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize and validate stage names from CLI input."""

    if value is None:
        return PIPELINE_STAGES
    stages = tuple(value)
    _validate_stages(stages)
    return stages


def _validate_stages(stages: Iterable[str]) -> None:
    unknown = [stage for stage in stages if stage not in EXPECTED_ARTIFACTS]
    if unknown:
        valid = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"Unknown pipeline stage(s): {unknown}. Expected one of: {valid}.")


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
