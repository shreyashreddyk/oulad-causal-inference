"""Tests for final report asset export helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from oulad_causal import viz


def test_write_report_table_selects_renames_and_formats(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "name": ["a", "b"],
            "estimate": [0.1234567, None],
            "ignored": [1, 2],
        }
    )

    exported = viz.write_report_table(
        frame,
        tmp_path / "table.csv",
        columns=("name", "estimate"),
        rename={"name": "Name", "estimate": "Estimate"},
        numeric_formats={"estimate": "{:.3f}"},
    )

    assert list(exported.columns) == ["Name", "Estimate"]
    assert exported.loc[0, "Estimate"] == "0.123"
    assert exported.loc[1, "Estimate"] == ""
    round_tripped = pd.read_csv(tmp_path / "table.csv", keep_default_na=False)
    assert round_tripped.to_dict(orient="records") == exported.to_dict(orient="records")


def test_figure_helpers_write_expected_pngs(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        {
            "treatment_high_engagement_14d_median": [0, 1, 1, 0],
            "treatment_high_engagement_14d_top_tertile": [0, 1, 0, 0],
            "treatment_high_engagement_14d_top_quartile": [0, 1, 0, 0],
        }
    )
    comparison = pd.DataFrame(
        {
            "method": ["pc", "pc", "fci", "ges", "hand_dag_missing_from_discovery"],
            "in_hand_skeleton": [True, False, True, False, True],
        }
    )
    stability = pd.DataFrame({"edge_frequency": [0.80, 0.50, 0.90]})
    subgroup = pd.DataFrame(
        {
            "section": ["subgroup", "subgroup", "placebo"],
            "scenario_status": ["success", "success", "success"],
            "subgroup_variable": ["prior_attempts", "disability", None],
            "subgroup_level": ["0", "N", None],
            "estimate": [0.2, 0.1, 0.0],
        }
    )

    outputs = [
        viz.write_treatment_prevalence_figure(cohort, tmp_path / "prevalence.png"),
        viz.write_discovery_comparison_figure(comparison, stability, tmp_path / "discovery.png"),
        viz.write_subgroup_summary_figure(subgroup, tmp_path / "subgroup.png"),
    ]

    for output in outputs:
        assert output.exists()
        assert output.read_bytes().startswith(b"\x89PNG")


def test_build_report_assets_reports_missing_inputs(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_report_assets.py"
    spec = importlib.util.spec_from_file_location("build_report_assets", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(FileNotFoundError, match="Missing saved input artifacts"):
        module._require_inputs([tmp_path / "missing.csv"])
