"""Feature construction for treatment, outcomes, and covariates.

Primary feature logic belongs here, including early engagement windows,
module-presentation normalization, and pre-treatment covariate construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
PRESENTATION_COLUMNS = ["code_module", "code_presentation"]
DEFAULT_WINDOWS = (7, 14, 21)
DEFAULT_THRESHOLDS = {
    "median": 0.50,
    "top_tertile": 2 / 3,
    "top_quartile": 0.75,
}


@dataclass(frozen=True)
class TreatmentDefinition:
    """Metadata for one binary early-engagement treatment column."""

    column: str
    window_days: int
    threshold_name: str
    quantile: float
    score_column: str
    availability_column: str
    comparison: str = ">= cutoff"


def aggregate_early_vle_clicks(
    student_vle: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Aggregate cumulative VLE clicks for each early post-start window.

    A window of ``N`` days includes relative dates ``0`` through ``N - 1``.
    Negative pre-start activity is intentionally excluded from treatment.
    """

    required = set(KEY_COLUMNS + ["date", "sum_click"])
    missing = required.difference(student_vle.columns)
    if missing:
        raise KeyError(f"studentVle is missing required columns: {sorted(missing)}")

    base = student_vle[KEY_COLUMNS + ["date", "sum_click"]].copy()
    base["date"] = pd.to_numeric(base["date"], errors="coerce")
    base["sum_click"] = pd.to_numeric(base["sum_click"], errors="coerce").fillna(0)
    keys = base[KEY_COLUMNS].drop_duplicates()

    for window in windows:
        in_window = base[(base["date"] >= 0) & (base["date"] < window)]
        clicks = (
            in_window.groupby(KEY_COLUMNS, as_index=False)["sum_click"]
            .sum()
            .rename(columns={"sum_click": early_click_column(window)})
        )
        keys = keys.merge(clicks, on=KEY_COLUMNS, how="left")
        keys[early_click_column(window)] = keys[early_click_column(window)].fillna(0)

    return keys


def build_early_assessment_load_features(
    assessments: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Summarize scheduled assessment load available by each early window.

    Uses only the assessment schedule and weights, not student submissions,
    scores, or banked assessment outcomes.
    """

    required = set(PRESENTATION_COLUMNS + ["id_assessment", "assessment_type", "date", "weight"])
    missing = required.difference(assessments.columns)
    if missing:
        raise KeyError(f"assessments is missing required columns: {sorted(missing)}")

    base = assessments[PRESENTATION_COLUMNS + ["id_assessment", "assessment_type", "date", "weight"]].copy()
    base["date"] = pd.to_numeric(base["date"], errors="coerce")
    base["weight"] = pd.to_numeric(base["weight"], errors="coerce").fillna(0)
    presentations = assessments[PRESENTATION_COLUMNS].drop_duplicates()

    for window in windows:
        in_window = base[base["date"].notna() & (base["date"] >= 0) & (base["date"] < window)]
        grouped = in_window.groupby(PRESENTATION_COLUMNS)
        load = grouped.agg(
            **{
                assessment_count_column(window): ("id_assessment", "nunique"),
                assessment_weight_column(window): ("weight", "sum"),
            }
        ).reset_index()
        for assessment_type in ("CMA", "TMA", "Exam"):
            type_counts = (
                in_window[in_window["assessment_type"] == assessment_type]
                .groupby(PRESENTATION_COLUMNS)["id_assessment"]
                .nunique()
                .reset_index(name=assessment_type_count_column(window, assessment_type))
            )
            load = load.merge(type_counts, on=PRESENTATION_COLUMNS, how="left")

        presentations = presentations.merge(load, on=PRESENTATION_COLUMNS, how="left")
        created = [
            assessment_count_column(window),
            assessment_weight_column(window),
            *[assessment_type_count_column(window, value) for value in ("CMA", "TMA", "Exam")],
        ]
        for column in created:
            presentations[column] = presentations[column].fillna(0)

    return presentations


def add_outcome_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add course success and withdrawal outcome indicators from ``final_result``."""

    if "final_result" not in frame.columns:
        raise KeyError("final_result is required to construct outcomes.")

    result = frame.copy()
    success_labels = {"Pass", "Distinction"}
    failure_labels = {"Fail", "Withdrawn"}
    known = success_labels | failure_labels
    result["outcome_success"] = np.where(
        result["final_result"].isin(success_labels),
        1,
        np.where(result["final_result"].isin(failure_labels), 0, pd.NA),
    )
    result["outcome_withdrawn"] = np.where(
        result["final_result"].eq("Withdrawn"),
        1,
        np.where(result["final_result"].isin(known), 0, pd.NA),
    )
    result["outcome_success"] = result["outcome_success"].astype("Int64")
    result["outcome_withdrawn"] = result["outcome_withdrawn"].astype("Int64")
    return result


def add_baseline_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned baseline covariates from student, registration, and course fields."""

    required = set(
        PRESENTATION_COLUMNS
        + [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "date_registration",
            "module_presentation_length",
        ]
    )
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"cohort frame is missing baseline columns: {sorted(missing)}")

    result = frame.copy()
    for column in ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]:
        result[f"baseline_{column}"] = result[column].fillna("<MISSING>").replace("", "<MISSING>")

    result["baseline_num_of_prev_attempts"] = pd.to_numeric(result["num_of_prev_attempts"], errors="coerce")
    result["baseline_studied_credits"] = pd.to_numeric(result["studied_credits"], errors="coerce")
    result["baseline_date_registration"] = pd.to_numeric(result["date_registration"], errors="coerce")
    result["baseline_missing_date_registration"] = result["baseline_date_registration"].isna().astype("int64")
    result["baseline_registered_before_start"] = (result["baseline_date_registration"] < 0).astype("Int64")
    result["baseline_module_presentation_length"] = pd.to_numeric(
        result["module_presentation_length"], errors="coerce"
    )
    result["baseline_module_presentation"] = (
        result["code_module"].astype(str) + "_" + result["code_presentation"].astype(str)
    )
    return result


def add_within_presentation_z_scores(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    *,
    group_columns: Iterable[str] = PRESENTATION_COLUMNS,
) -> pd.DataFrame:
    """Add within-presentation z-scores for numeric feature columns."""

    result = frame.copy()
    groups = list(group_columns)
    for column in value_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        means = values.groupby([result[group] for group in groups]).transform("mean")
        stds = values.groupby([result[group] for group in groups]).transform(lambda value: value.std(ddof=0))
        z = (values - means) / stds.replace(0, np.nan)
        result[z_score_column(column)] = z.fillna(0)
    return result


def add_treatment_thresholds(
    frame: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    group_columns: Iterable[str] = PRESENTATION_COLUMNS,
) -> tuple[pd.DataFrame, list[TreatmentDefinition], pd.DataFrame]:
    """Add quantile-threshold binary treatments for each early engagement window."""

    result = frame.copy()
    groups = list(group_columns)
    definitions: list[TreatmentDefinition] = []
    cutoff_rows: list[dict[str, object]] = []

    for window in windows:
        score_column = early_click_z_column(window)
        availability_column = treatment_available_column(window)
        if score_column not in result.columns:
            raise KeyError(f"Missing score column {score_column!r}.")
        if availability_column not in result.columns:
            raise KeyError(f"Missing availability column {availability_column!r}.")

        available = result[availability_column].fillna(False).astype(bool)
        for threshold_name, quantile in thresholds.items():
            treatment_column = treatment_column_name(window, threshold_name)
            result[treatment_column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
            cutoff_column = f"_{treatment_column}_cutoff"
            result[cutoff_column] = pd.NA

            for group_values, group_index in result[available].groupby(groups).groups.items():
                scores = pd.to_numeric(result.loc[group_index, score_column], errors="coerce")
                cutoff = scores.quantile(quantile)
                result.loc[group_index, cutoff_column] = cutoff
                result.loc[group_index, treatment_column] = (scores >= cutoff).astype("int64").values
                cutoff_rows.append(
                    {
                        "code_module": _group_value(group_values, 0),
                        "code_presentation": _group_value(group_values, 1),
                        "window_days": window,
                        "threshold_name": threshold_name,
                        "quantile": quantile,
                        "cutoff": float(cutoff) if pd.notna(cutoff) else None,
                        "eligible_count": int(scores.notna().sum()),
                    }
                )

            result = result.drop(columns=[cutoff_column])
            definitions.append(
                TreatmentDefinition(
                    column=treatment_column,
                    window_days=window,
                    threshold_name=threshold_name,
                    quantile=float(quantile),
                    score_column=score_column,
                    availability_column=availability_column,
                )
            )

    return result, definitions, pd.DataFrame(cutoff_rows)


def early_click_column(window: int) -> str:
    return f"early_clicks_{window}d"


def early_click_z_column(window: int) -> str:
    return z_score_column(early_click_column(window))


def treatment_available_column(window: int) -> str:
    return f"treatment_available_{window}d"


def treatment_column_name(window: int, threshold_name: str) -> str:
    return f"treatment_high_engagement_{window}d_{threshold_name}"


def assessment_count_column(window: int) -> str:
    return f"early_assessment_count_{window}d"


def assessment_weight_column(window: int) -> str:
    return f"early_assessment_weight_{window}d"


def assessment_type_count_column(window: int, assessment_type: str) -> str:
    return f"early_assessment_{assessment_type.lower()}_count_{window}d"


def z_score_column(column: str) -> str:
    return f"{column}_z"


def _group_value(group_values: object, position: int) -> object:
    if isinstance(group_values, tuple):
        return group_values[position]
    return group_values if position == 0 else None
