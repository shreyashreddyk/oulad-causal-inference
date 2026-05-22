"""Tests for treatment effect estimation helpers."""

import numpy as np
import pandas as pd
import pytest

from oulad_causal import config, estimation


def test_estimation_module_imports() -> None:
    assert estimation.__doc__


def test_project_root_contains_pyproject() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_estimator_interface_returns_main_estimators() -> None:
    frame = _synthetic_frame()
    cfg = estimation.EstimationConfig(
        adjustment_columns=("x_num", "x_cat"),
        poor_overlap_propensity_lower=0.0,
        poor_overlap_propensity_upper=1.0,
        poor_overlap_outside_common_support_share=1.0,
        poor_overlap_max_abs_smd=99.0,
    )

    result = estimation.estimate_effects(frame, config=cfg)

    estimators = set(result.effect_estimates["estimator"])
    assert {"regression_adjustment", "stabilized_iptw", "aipw"}.issubset(estimators)
    assert result.effect_estimates.loc[result.effect_estimates["estimator"] == "aipw", "preferred"].item()
    assert result.metadata["estimand"]["adjustment_columns"] == ["x_num", "x_cat"]
    assert result.metadata["model_settings"]["seed"] == 245
    assert result.propensity_scores.between(0, 1).all()
    assert result.stabilized_weights.gt(0).all()
    assert {"std_error", "ci_lower", "ci_upper", "uncertainty_method"}.issubset(result.effect_estimates.columns)
    aipw = result.effect_estimates.loc[result.effect_estimates["estimator"] == "aipw"].iloc[0]
    assert aipw["std_error"] >= 0
    assert aipw["ci_lower"] <= aipw["estimate"] <= aipw["ci_upper"]


def test_propensity_scores_are_clipped_for_computation_only() -> None:
    raw = np.array([0.0, 0.5, 1.0])

    clipped = estimation.clip_propensity_scores(raw, epsilon=0.01)

    assert clipped.tolist() == [0.01, 0.5, 0.99]


def test_standardized_mean_differences_match_hand_checkable_values() -> None:
    covariates = pd.DataFrame({"x": [0.0, 2.0, 2.0, 4.0]})
    treatment = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 3.0, 1.0])

    unweighted = estimation.standardized_mean_differences(covariates, treatment=treatment)
    weighted = estimation.standardized_mean_differences(covariates, treatment=treatment, weights=weights)

    assert unweighted["x"] == pytest.approx(2.0)
    assert weighted["x"] == pytest.approx(1.0 / np.sqrt(0.75))


def test_balance_table_contains_before_and_after_weighting_columns() -> None:
    covariates = pd.DataFrame({"x": [0.0, 2.0, 2.0, 4.0], "z": [1.0, 1.0, 0.0, 0.0]})
    treatment = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 3.0, 3.0, 1.0])

    table = estimation.standardized_mean_difference_table(covariates, treatment=treatment, weights=weights)

    assert {
        "variable",
        "smd_unweighted",
        "smd_weighted",
        "abs_smd_unweighted",
        "abs_smd_weighted",
        "improved_after_weighting",
    }.issubset(table.columns)
    assert set(table["variable"]) == {"x", "z"}


def test_missing_required_columns_raise_clear_error() -> None:
    frame = _synthetic_frame().drop(columns=["x_cat"])
    cfg = estimation.EstimationConfig(adjustment_columns=("x_num", "x_cat"))

    with pytest.raises(KeyError, match="Missing required estimation columns"):
        estimation.validate_estimation_columns(frame, config=cfg)


def test_optional_matching_skips_when_diagnostics_fail() -> None:
    row = estimation.nearest_neighbor_matching_estimate(
        treatment=np.array([0, 0, 1, 1]),
        outcome=np.array([0, 1, 1, 1]),
        propensity=np.array([0.2, 0.3, 0.7, 0.8]),
        poor_overlap=True,
        config=estimation.EstimationConfig(adjustment_columns=("x_num",)),
    )

    assert row["estimator"] == "nearest_neighbor_matching"
    assert row["status"] == "skipped"
    assert "poor overlap" in row["notes"]


def _synthetic_frame() -> pd.DataFrame:
    n = 80
    x_num = np.tile(np.arange(8), 10).astype(float)
    x_cat = np.where(np.arange(n) % 3 == 0, "A", "B")
    treatment = ((x_num + (x_cat == "A").astype(float) + np.arange(n) % 2) > 4).astype(int)
    outcome = ((0.2 * x_num + 0.7 * treatment + (x_cat == "A").astype(float)) > 1.4).astype(int)
    outcome[0] = 0
    outcome[-1] = 1
    return pd.DataFrame(
        {
            "treatment_high_engagement_14d_median": treatment,
            "outcome_success": outcome,
            "treatment_available_14d": True,
            "x_num": x_num,
            "x_cat": x_cat,
        }
    )
