"""Tests for robustness checks and artifact generation."""

import numpy as np
import pandas as pd

from oulad_causal import robustness


def test_window_adjustment_columns_replace_scheduled_assessment_window() -> None:
    columns = (
        "baseline_gender",
        "early_assessment_count_14d",
        "early_assessment_weight_14d",
        "early_assessment_cma_count_14d",
    )

    adjusted = robustness.window_adjustment_columns(21, columns)

    assert adjusted == (
        "baseline_gender",
        "early_assessment_count_21d",
        "early_assessment_weight_21d",
        "early_assessment_cma_count_21d",
    )


def test_treatment_spec_names_treatment_and_availability_columns() -> None:
    spec = robustness.treatment_spec(7, "top_quartile")

    assert spec["treatment_column"] == "treatment_high_engagement_7d_top_quartile"
    assert spec["availability_column"] == "treatment_available_7d"


def test_gate_reason_documents_small_or_degenerate_strata() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0],
            "outcome": [1, 1, 0, 0],
        }
    )

    reason = robustness.gate_reason(
        frame,
        treatment_column="treatment",
        outcome_column="outcome",
        min_n=4,
        min_treated=1,
        min_control=2,
    )

    assert reason == "control_count=1 below minimum 2"


def test_subgroup_and_placebo_adjustment_columns_remove_target_family() -> None:
    base = (
        "baseline_highest_education",
        "baseline_num_of_prev_attempts",
        "baseline_disability",
        "baseline_date_registration",
        "baseline_missing_date_registration",
        "baseline_registered_before_start",
        "baseline_gender",
    )

    subgroup = robustness.subgroup_adjustment_columns("prior_attempts", base)
    placebo = robustness.placebo_adjustment_columns("placebo_registered_before_start", base)

    assert "baseline_num_of_prev_attempts" not in subgroup
    assert "baseline_gender" in subgroup
    assert "baseline_date_registration" not in placebo
    assert "baseline_missing_date_registration" not in placebo
    assert "baseline_registered_before_start" not in placebo
    assert "baseline_gender" in placebo


def test_synthetic_robustness_pipeline_writes_expected_artifacts(tmp_path) -> None:
    frame = _synthetic_robustness_frame()
    config = robustness.RobustnessConfig(
        cohort_path=tmp_path / "cohort.parquet",
        processed_dir=tmp_path / "processed",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        docs_dir=tmp_path / "docs",
        windows=(7, 14),
        thresholds=("median",),
        module_min_n=40,
        module_min_treated=10,
        module_min_control=10,
        subgroup_min_n=20,
        subgroup_min_treated=5,
        subgroup_min_control=5,
    )

    result = robustness.run_robustness_checks(frame, config=config)
    paths = robustness.write_robustness_artifacts(result, config=config)

    assert not result.window_threshold_summary.empty
    assert set(result.window_threshold_summary["window_days"]) == {7, 14}
    assert "aipw" in set(result.estimates_long["estimator"])
    assert result.metadata["grid"]["pooled_scenarios"] == 2
    assert result.metadata["scenario_counts"]["placebo_success"] >= 1
    for path in paths.values():
        assert path.exists()


def _synthetic_robustness_frame() -> pd.DataFrame:
    n = 120
    idx = np.arange(n)
    modules = np.where(idx < 60, "AAA_2013J", "BBB_2014J")
    prior_attempts = idx % 3
    disability = np.where(idx % 4 == 0, "Y", "N")
    registered = (idx % 5 != 0).astype(int)
    treatment_14 = ((idx + (modules == "AAA_2013J").astype(int)) % 2).astype(int)
    treatment_7 = ((idx + prior_attempts) % 2).astype(int)
    outcome = ((treatment_14 + (idx % 4 == 0).astype(int) + (idx % 7 == 0).astype(int)) > 0).astype(int)
    outcome[::11] = 0

    frame = pd.DataFrame(
        {
            "code_module": np.where(modules == "AAA_2013J", "AAA", "BBB"),
            "code_presentation": np.where(modules == "AAA_2013J", "2013J", "2014J"),
            "id_student": idx,
            "outcome_success": outcome,
            "baseline_gender": np.where(idx % 2 == 0, "F", "M"),
            "baseline_region": np.where(idx % 3 == 0, "North", "South"),
            "baseline_age_band": np.where(idx % 3 == 0, "0-35", "35-55"),
            "baseline_highest_education": np.where(idx % 2 == 0, "A Level or Equivalent", "HE Qualification"),
            "baseline_imd_band": np.where(idx % 4 == 0, "0-10%", "70-80%"),
            "baseline_disability": disability,
            "baseline_num_of_prev_attempts": prior_attempts,
            "baseline_studied_credits": 60 + (idx % 3) * 30,
            "baseline_date_registration": -30 + (idx % 10),
            "baseline_missing_date_registration": 0,
            "baseline_registered_before_start": registered,
            "baseline_module_presentation": modules,
            "baseline_module_presentation_length": np.where(modules == "AAA_2013J", 240, 260),
            "treatment_available_7d": True,
            "treatment_available_14d": True,
            "treatment_available_21d": True,
            "treatment_high_engagement_7d_median": treatment_7,
            "treatment_high_engagement_14d_median": treatment_14,
            "treatment_high_engagement_21d_median": treatment_14,
        }
    )
    for window in (7, 14, 21):
        frame[f"early_assessment_count_{window}d"] = idx % 2
        frame[f"early_assessment_weight_{window}d"] = (idx % 5) * 5
        frame[f"early_assessment_cma_count_{window}d"] = idx % 2
        frame[f"early_assessment_tma_count_{window}d"] = (idx + 1) % 2
        frame[f"early_assessment_exam_count_{window}d"] = 0
        frame[f"treatment_high_engagement_{window}d_top_tertile"] = frame[
            f"treatment_high_engagement_{window}d_median"
        ]
        frame[f"treatment_high_engagement_{window}d_top_quartile"] = frame[
            f"treatment_high_engagement_{window}d_median"
        ]
    return frame
