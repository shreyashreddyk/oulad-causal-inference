"""Tests for reduced causal discovery helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_causal import discovery


def test_select_discovery_variables_is_bounded_and_excludes_post_treatment() -> None:
    variables = discovery.select_discovery_variables()

    assert 8 <= len(variables) <= 12
    assert discovery.PRIMARY_TREATMENT_COLUMN in variables
    assert discovery.PRIMARY_OUTCOME_COLUMN in variables
    assert "early_clicks_14d_z" not in variables
    assert "score" not in variables
    assert "date_submitted" not in variables
    assert "is_banked" not in variables


def test_preprocess_discovery_data_returns_finite_discrete_matrix() -> None:
    cohort = _fixture_cohort()

    encoded, metadata = discovery.preprocess_discovery_data(cohort)

    assert list(encoded.columns) == list(discovery.DISCOVERY_VARIABLES)
    assert encoded.shape == (4, len(discovery.DISCOVERY_VARIABLES))
    assert all(str(dtype).startswith("int") for dtype in encoded.dtypes)
    assert np.isfinite(encoded.to_numpy(dtype=float)).all()
    assert metadata["columns"]["baseline_age_band"]["preprocessing"] == "explicit_ordinal"
    assert metadata["columns"]["baseline_studied_credits"]["preprocessing"] == "quantile_bins"
    assert "A Level or Equivalent" in metadata["columns"]["baseline_highest_education"]["mapping"]


def test_compare_with_hand_built_dag_separates_skeleton_and_orientation() -> None:
    edges = pd.DataFrame(
        [
            {
                "method": "pc",
                "source": "baseline_gender",
                "target": discovery.PRIMARY_TREATMENT_COLUMN,
                "endpoint_source": "TAIL",
                "endpoint_target": "ARROW",
                "edge_type": "baseline_gender --> treatment_high_engagement_14d_median",
                "directed_source": "baseline_gender",
                "directed_target": discovery.PRIMARY_TREATMENT_COLUMN,
                "is_directed": True,
                "skeleton_key": discovery._skeleton_key("baseline_gender", discovery.PRIMARY_TREATMENT_COLUMN),
            },
            {
                "method": "ges",
                "source": discovery.PRIMARY_TREATMENT_COLUMN,
                "target": "baseline_gender",
                "endpoint_source": "TAIL",
                "endpoint_target": "ARROW",
                "edge_type": "treatment_high_engagement_14d_median --> baseline_gender",
                "directed_source": discovery.PRIMARY_TREATMENT_COLUMN,
                "directed_target": "baseline_gender",
                "is_directed": True,
                "skeleton_key": discovery._skeleton_key("baseline_gender", discovery.PRIMARY_TREATMENT_COLUMN),
            },
        ]
    )

    comparison = discovery.compare_with_hand_built_dag(edges)
    discovered = comparison[comparison["method"].isin(["pc", "ges"])].reset_index(drop=True)

    assert discovered.loc[0, "in_hand_skeleton"]
    assert discovered.loc[0, "in_hand_directed"]
    assert discovered.loc[1, "in_hand_skeleton"]
    assert not discovered.loc[1, "in_hand_directed"]


def test_write_discovery_summary_includes_required_sections(tmp_path: Path) -> None:
    path = tmp_path / "discovery_summary.md"
    preprocessing = {"variables": list(discovery.DISCOVERY_VARIABLES)}
    metadata = {
        "row_count": 4,
        "alpha": 0.01,
        "methods": {"pc": {"status": "success", "seconds": 0.1, "edge_count": 1}},
        "artifacts": {"combined_edges": "data/processed/discovery_edges.csv"},
    }
    comparison = pd.DataFrame(
        [
            {
                "method": "pc",
                "source": "baseline_gender",
                "target": discovery.PRIMARY_TREATMENT_COLUMN,
                "skeleton_key": discovery._skeleton_key("baseline_gender", discovery.PRIMARY_TREATMENT_COLUMN),
                "in_hand_skeleton": True,
                "in_hand_directed": True,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "method": "pc",
                "var_a": "baseline_gender",
                "var_b": discovery.PRIMARY_TREATMENT_COLUMN,
                "edge_frequency": 0.8,
                "directed_a_to_b_frequency": 0.8,
                "directed_b_to_a_frequency": 0.0,
            }
        ]
    )

    discovery.write_discovery_summary(
        summary_path=path,
        preprocessing_metadata=preprocessing,
        run_metadata=metadata,
        comparison=comparison,
        stability=stability,
    )

    text = path.read_text(encoding="utf-8")
    assert "## What discovery supports" in text
    assert "## What discovery does not establish" in text


def test_fci_failure_handling_does_not_block_other_methods(tmp_path: Path, monkeypatch) -> None:
    cohort_path = tmp_path / "cohort.parquet"
    _fixture_cohort().to_parquet(cohort_path, index=False)

    class FakeGraph:
        def __init__(self) -> None:
            self.graph = np.zeros((len(discovery.DISCOVERY_VARIABLES), len(discovery.DISCOVERY_VARIABLES)), dtype=int)

        def get_graph_edges(self) -> list[object]:
            return []

    def fake_run_method(method: str, *_args, **_kwargs) -> FakeGraph:
        if method == "fci":
            raise RuntimeError("FCI failed in fixture")
        return FakeGraph()

    monkeypatch.setattr(discovery, "_run_method", fake_run_method)

    paths = discovery.run_discovery_pipeline(
        discovery.DiscoveryConfig(
            cohort_path=cohort_path,
            processed_dir=tmp_path / "processed",
            figures_dir=tmp_path / "figures",
            docs_dir=tmp_path / "docs",
            stability_reps=2,
            stability_sample_size=2,
        )
    )

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["methods"]["fci"]["status"] == "failed"
    assert metadata["methods"]["pc"]["status"] == "success"
    assert metadata["methods"]["ges"]["status"] == "success"
    assert paths["summary"].exists()


def _fixture_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "baseline_gender": ["F", "M", "F", "M"],
            "baseline_age_band": ["0-35", "35-55", "55<=", "35-55"],
            "baseline_highest_education": [
                "A Level or Equivalent",
                "HE Qualification",
                "Lower Than A Level",
                "Post Graduate Qualification",
            ],
            "baseline_imd_band": ["0-10%", "30-40%", "70-80%", "90-100%"],
            "baseline_disability": ["N", "N", "Y", "N"],
            "baseline_num_of_prev_attempts": [0, 1, 0, 2],
            "baseline_studied_credits": [30, 60, 120, 240],
            "baseline_registered_before_start": [1, 1, 0, 1],
            "baseline_module_presentation_length": [240, 240, 268, 268],
            "early_assessment_weight_14d": [0.0, 0.0, 10.0, 10.0],
            discovery.PRIMARY_TREATMENT_COLUMN: [0, 1, 0, 1],
            discovery.PRIMARY_OUTCOME_COLUMN: [0, 1, 0, 1],
        }
    )
