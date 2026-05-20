"""Tests for cohort construction helpers."""

import pandas as pd

from oulad_causal import cohort


def test_cohort_module_imports() -> None:
    assert cohort.__doc__


def test_build_analytic_cohort_excludes_early_unregistration_and_retains_missing() -> None:
    result = cohort.build_analytic_cohort(synthetic_tables(), config=cohort.CohortConfig(windows=(7, 14, 21)))

    built = result.cohort

    assert set(built["id_student"]) == {1, 3, 4}
    assert built[["code_module", "code_presentation", "id_student"]].duplicated().sum() == 0
    assert result.flow_table.loc[
        result.flow_table["stage"] == "after_primary_treatment_eligibility", "excluded_count"
    ].iloc[0] == 1
    retained_missing = built[built["id_student"] == 3].iloc[0]
    assert pd.isna(retained_missing["date_unregistration"])
    assert retained_missing["baseline_missing_date_registration"] == 1


def test_build_analytic_cohort_marks_window_specific_treatment_availability() -> None:
    result = cohort.build_analytic_cohort(synthetic_tables(), config=cohort.CohortConfig(windows=(7, 14, 21)))
    built = result.cohort.set_index("id_student")

    assert built.loc[4, "treatment_available_14d"]
    assert not built.loc[4, "treatment_available_21d"]
    assert pd.isna(built.loc[4, "treatment_high_engagement_21d_median"])
    assert built.loc[3, "early_clicks_14d"] == 0


def test_build_analytic_cohort_adds_outcomes_and_metadata() -> None:
    result = cohort.build_analytic_cohort(synthetic_tables(), config=cohort.CohortConfig(windows=(14,)))
    built = result.cohort.set_index("id_student")

    assert built.loc[1, "outcome_success"] == 1
    assert built.loc[3, "outcome_withdrawn"] == 1
    assert result.summary["cohort_size"] == 3
    assert "treatment_high_engagement_14d_median" in result.summary["treatment_prevalence"]


def synthetic_tables() -> dict[str, pd.DataFrame]:
    return {
        "courses": pd.DataFrame(
            [
                {"code_module": "AAA", "code_presentation": "2013J", "module_presentation_length": 268},
            ]
        ),
        "assessments": pd.DataFrame(
            [
                {
                    "code_module": "AAA",
                    "code_presentation": "2013J",
                    "id_assessment": 1,
                    "assessment_type": "TMA",
                    "date": 13,
                    "weight": 10,
                }
            ]
        ),
        "studentInfo": pd.DataFrame(
            [
                _student_info(1, "Pass"),
                _student_info(2, "Fail"),
                _student_info(3, "Withdrawn"),
                _student_info(4, "Distinction"),
            ]
        ),
        "studentRegistration": pd.DataFrame(
            [
                _registration(1, -20, pd.NA),
                _registration(2, -10, 10),
                _registration(3, pd.NA, pd.NA),
                _registration(4, -5, 18),
            ]
        ),
        "studentVle": pd.DataFrame(
            [
                _vle(1, 0, 5),
                _vle(1, 13, 5),
                _vle(2, 0, 99),
                _vle(4, 13, 20),
                _vle(4, 20, 40),
            ]
        ),
    }


def _student_info(id_student: int, final_result: str) -> dict[str, object]:
    return {
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": id_student,
        "gender": "F",
        "region": "Scotland",
        "highest_education": "HE Qualification",
        "imd_band": pd.NA,
        "age_band": "35-55",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
        "final_result": final_result,
    }


def _registration(id_student: int, date_registration: object, date_unregistration: object) -> dict[str, object]:
    return {
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": id_student,
        "date_registration": date_registration,
        "date_unregistration": date_unregistration,
    }


def _vle(id_student: int, date: int, sum_click: int) -> dict[str, object]:
    return {
        "code_module": "AAA",
        "code_presentation": "2013J",
        "id_student": id_student,
        "id_site": 10,
        "date": date,
        "sum_click": sum_click,
    }
