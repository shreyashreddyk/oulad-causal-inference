"""Tests for feature construction helpers."""

import pandas as pd

from oulad_causal import features


def test_features_module_imports() -> None:
    assert features.__doc__


def test_early_vle_aggregation_uses_zero_to_window_minus_one() -> None:
    student_vle = pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": -1, "sum_click": 100},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": 0, "sum_click": 2},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": 6, "sum_click": 3},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": 7, "sum_click": 5},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": 13, "sum_click": 7},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date": 14, "sum_click": 11},
        ]
    )

    aggregated = features.aggregate_early_vle_clicks(student_vle, windows=(7, 14))

    row = aggregated.iloc[0]
    assert row["early_clicks_7d"] == 5
    assert row["early_clicks_14d"] == 17


def test_early_assessment_load_uses_scheduled_due_dates_only() -> None:
    assessments = pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_assessment": 1,
                "assessment_type": "TMA",
                "date": 13,
                "weight": 10,
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_assessment": 2,
                "assessment_type": "CMA",
                "date": 14,
                "weight": 5,
            },
        ]
    )

    load = features.build_early_assessment_load_features(assessments, windows=(14,))

    row = load.iloc[0]
    assert row["early_assessment_count_14d"] == 1
    assert row["early_assessment_weight_14d"] == 10
    assert row["early_assessment_tma_count_14d"] == 1
    assert row["early_assessment_cma_count_14d"] == 0


def test_within_presentation_z_score_normalization_and_zero_variance() -> None:
    frame = pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "early_clicks_14d": 0},
            {"code_module": "AAA", "code_presentation": "2013J", "early_clicks_14d": 2},
            {"code_module": "BBB", "code_presentation": "2013J", "early_clicks_14d": 5},
            {"code_module": "BBB", "code_presentation": "2013J", "early_clicks_14d": 5},
        ]
    )

    normalized = features.add_within_presentation_z_scores(frame, ["early_clicks_14d"])

    aaa = normalized[normalized["code_module"] == "AAA"]["early_clicks_14d_z"].tolist()
    bbb = normalized[normalized["code_module"] == "BBB"]["early_clicks_14d_z"].tolist()
    assert aaa == [-1.0, 1.0]
    assert bbb == [0.0, 0.0]


def test_threshold_construction_uses_greater_than_or_equal_cutoff() -> None:
    frame = pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 1,
                "early_clicks_14d_z": 0.0,
                "treatment_available_14d": True,
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 2,
                "early_clicks_14d_z": 1.0,
                "treatment_available_14d": True,
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 3,
                "early_clicks_14d_z": 1.0,
                "treatment_available_14d": True,
            },
        ]
    )

    treated, definitions, cutoffs = features.add_treatment_thresholds(frame, windows=(14,))

    assert {definition.column for definition in definitions} == {
        "treatment_high_engagement_14d_median",
        "treatment_high_engagement_14d_top_tertile",
        "treatment_high_engagement_14d_top_quartile",
    }
    assert treated["treatment_high_engagement_14d_median"].tolist() == [0, 1, 1]
    median_cutoff = cutoffs[cutoffs["threshold_name"] == "median"].iloc[0]
    assert median_cutoff["cutoff"] == 1.0
