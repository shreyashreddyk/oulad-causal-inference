"""Cohort construction logic.

Use this module to build the student-module-presentation analysis cohort and to
record inclusion and exclusion counts as reproducible artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import pandas as pd

from oulad_causal.features import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WINDOWS,
    KEY_COLUMNS,
    PRESENTATION_COLUMNS,
    TreatmentDefinition,
    add_baseline_covariates,
    add_outcome_features,
    add_treatment_thresholds,
    add_within_presentation_z_scores,
    aggregate_early_vle_clicks,
    build_early_assessment_load_features,
    early_click_column,
    early_click_z_column,
    treatment_available_column,
)


@dataclass(frozen=True)
class CohortConfig:
    """Configuration for deterministic analytic cohort construction."""

    windows: tuple[int, ...] = DEFAULT_WINDOWS
    primary_window: int = 14
    thresholds: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    exclude_early_unregistration: bool = True
    raw_unregistration_column: str = "date_unregistration"


@dataclass
class CohortBuildResult:
    """Analytic cohort plus reproducibility metadata."""

    cohort: pd.DataFrame
    flow_table: pd.DataFrame
    treatment_definitions: list[TreatmentDefinition]
    threshold_cutoffs: pd.DataFrame
    summary: dict[str, object]


def build_analytic_cohort(
    tables: Mapping[str, pd.DataFrame],
    *,
    config: CohortConfig | None = None,
) -> CohortBuildResult:
    """Build the student x module-presentation analytic cohort from OULAD tables."""

    config = config or CohortConfig()
    _require_tables(tables, ["studentInfo", "studentRegistration", "courses", "studentVle", "assessments"])

    student_info = tables["studentInfo"].copy()
    student_registration = tables["studentRegistration"].copy()
    courses = tables["courses"].copy()
    student_vle = tables["studentVle"].copy()
    assessments = tables["assessments"].copy()

    _require_unique(student_info, KEY_COLUMNS, "studentInfo")
    _require_unique(student_registration, KEY_COLUMNS, "studentRegistration")
    _require_unique(courses, PRESENTATION_COLUMNS, "courses")

    flow_rows: list[dict[str, object]] = []
    start_count = len(student_info)
    flow_rows.append(_flow_row("student_info_records", start_count, 0, "studentInfo records loaded."))

    cohort = student_info.merge(
        student_registration,
        on=KEY_COLUMNS,
        how="left",
        indicator="registration_merge_status",
    )
    missing_registration = int(cohort["registration_merge_status"].ne("both").sum())
    cohort = cohort.merge(
        courses,
        on=PRESENTATION_COLUMNS,
        how="left",
        indicator="course_merge_status",
    )
    missing_course = int(cohort["course_merge_status"].ne("both").sum())
    missing_required = cohort["registration_merge_status"].ne("both") | cohort["course_merge_status"].ne("both")
    if missing_required.any():
        cohort = cohort.loc[~missing_required].copy()
    flow_rows.append(
        _flow_row(
            "after_required_table_joins",
            len(cohort),
            int(missing_required.sum()),
            f"Dropped records without registration or course metadata; missing_registration={missing_registration}, missing_course={missing_course}.",
        )
    )

    cohort["date_registration"] = pd.to_numeric(cohort["date_registration"], errors="coerce")
    cohort["date_unregistration"] = pd.to_numeric(cohort["date_unregistration"], errors="coerce")
    cohort["module_presentation_length"] = pd.to_numeric(
        cohort["module_presentation_length"], errors="coerce"
    )

    if config.exclude_early_unregistration:
        early_unregistration = cohort["date_unregistration"].notna() & (
            cohort["date_unregistration"] < config.primary_window
        )
        cohort = cohort.loc[~early_unregistration].copy()
        excluded = int(early_unregistration.sum())
        description = (
            f"Dropped records with date_unregistration < {config.primary_window}; "
            "missing unregistration dates are retained."
        )
    else:
        excluded = 0
        description = "Early-unregistration exclusion disabled by CohortConfig."
    flow_rows.append(
        _flow_row("after_primary_treatment_eligibility", len(cohort), excluded, description)
    )

    cohort = add_outcome_features(cohort)
    cohort = add_baseline_covariates(cohort)

    clicks = aggregate_early_vle_clicks(student_vle, windows=config.windows)
    cohort = cohort.merge(clicks, on=KEY_COLUMNS, how="left")
    for window in config.windows:
        cohort[early_click_column(window)] = cohort[early_click_column(window)].fillna(0)
        cohort[treatment_available_column(window)] = _treatment_available(cohort, window)

    assessment_load = build_early_assessment_load_features(assessments, windows=config.windows)
    cohort = cohort.merge(assessment_load, on=PRESENTATION_COLUMNS, how="left")
    assessment_columns = [column for column in assessment_load.columns if column not in PRESENTATION_COLUMNS]
    for column in assessment_columns:
        cohort[column] = cohort[column].fillna(0)

    cohort = add_within_presentation_z_scores(
        cohort,
        [early_click_column(window) for window in config.windows],
    )
    cohort, treatment_definitions, threshold_cutoffs = add_treatment_thresholds(
        cohort,
        windows=config.windows,
        thresholds=config.thresholds,
    )

    cohort = _order_columns(cohort)
    flow_table = pd.DataFrame(flow_rows)
    summary = build_cohort_summary(
        cohort=cohort,
        flow_table=flow_table,
        treatment_definitions=treatment_definitions,
        threshold_cutoffs=threshold_cutoffs,
        config=config,
    )

    return CohortBuildResult(
        cohort=cohort,
        flow_table=flow_table,
        treatment_definitions=treatment_definitions,
        threshold_cutoffs=threshold_cutoffs,
        summary=summary,
    )


def build_cohort_summary(
    *,
    cohort: pd.DataFrame,
    flow_table: pd.DataFrame,
    treatment_definitions: list[TreatmentDefinition],
    threshold_cutoffs: pd.DataFrame,
    config: CohortConfig,
) -> dict[str, object]:
    """Build machine-readable cohort metadata."""

    baseline_columns = sorted(column for column in cohort.columns if column.startswith("baseline_"))
    assessment_columns = sorted(column for column in cohort.columns if column.startswith("early_assessment_"))
    click_columns = [early_click_column(window) for window in config.windows]
    click_z_columns = [early_click_z_column(window) for window in config.windows]
    outcome_columns = ["outcome_success", "outcome_withdrawn"]
    treatment_columns = [definition.column for definition in treatment_definitions]
    availability_columns = [treatment_available_column(window) for window in config.windows]

    treatment_prevalence = {}
    for column in treatment_columns:
        values = cohort[column].dropna()
        treatment_prevalence[column] = {
            "nonmissing_count": int(values.shape[0]),
            "treated_count": int(values.sum()) if not values.empty else 0,
            "prevalence": float(values.mean()) if not values.empty else None,
        }

    return {
        "cohort_size": int(len(cohort)),
        "unit_of_analysis": "student x module_presentation record",
        "primary_window_days": config.primary_window,
        "window_definition": "Window N includes dates 0 through N-1.",
        "exclusion_counts": flow_table[["stage", "excluded_count"]].to_dict("records"),
        "feature_columns": {
            "baseline_covariates": baseline_columns,
            "early_assessment_load": assessment_columns,
            "early_click_counts": click_columns,
            "early_click_z_scores": click_z_columns,
            "treatment_availability": availability_columns,
            "outcomes": outcome_columns,
        },
        "treatment_definitions_available": [asdict(definition) for definition in treatment_definitions],
        "treatment_prevalence": treatment_prevalence,
        "threshold_cutoff_rows": int(len(threshold_cutoffs)),
        "post_treatment_variables_excluded_from_baseline": [
            "studentAssessment.date_submitted",
            "studentAssessment.score",
            "studentAssessment.is_banked",
            "studentVle activity after each treatment window",
        ],
    }


def _require_tables(tables: Mapping[str, pd.DataFrame], names: list[str]) -> None:
    missing = [name for name in names if name not in tables]
    if missing:
        raise KeyError(f"Missing required OULAD tables: {missing}")


def _require_unique(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    duplicate_rows = int(frame.duplicated(subset=columns, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"{table_name} has {duplicate_rows} duplicate rows for key {columns}.")


def _treatment_available(cohort: pd.DataFrame, window: int) -> pd.Series:
    unregistration = pd.to_numeric(cohort["date_unregistration"], errors="coerce")
    return (unregistration.isna() | (unregistration >= window)).astype("bool")


def _flow_row(stage: str, row_count: int, excluded_count: int, description: str) -> dict[str, object]:
    return {
        "stage": stage,
        "row_count": int(row_count),
        "excluded_count": int(excluded_count),
        "description": description,
    }


def _order_columns(cohort: pd.DataFrame) -> pd.DataFrame:
    leading = [
        "code_module",
        "code_presentation",
        "id_student",
        "final_result",
        "outcome_success",
        "outcome_withdrawn",
    ]
    ordered = [column for column in leading if column in cohort.columns]
    ordered.extend(column for column in cohort.columns if column not in ordered)
    return cohort.loc[:, ordered]
