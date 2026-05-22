"""Tests for script CLI entry points."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import (
    build_cohort,
    build_report_assets,
    run_data_validation,
    run_discovery,
    run_estimation,
    run_robustness,
)


def test_scripts_accept_standard_log_level_argument() -> None:
    modules = [
        run_data_validation,
        build_cohort,
        run_discovery,
        run_estimation,
        run_robustness,
        build_report_assets,
    ]

    for module in modules:
        args = module.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


def test_run_data_validation_main_returns_zero_on_non_strict_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result = SimpleNamespace(
        critical_failures=["missing"],
        passed=False,
        summaries={
            "table_shapes": pd.DataFrame(),
            "schema_validation": pd.DataFrame(),
            "duplicate_key_summary": pd.DataFrame(),
            "date_range_summary": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(run_data_validation, "locate_oulad_tables", lambda **_kwargs: [])
    monkeypatch.setattr(run_data_validation, "load_oulad_tables", lambda **_kwargs: {})
    monkeypatch.setattr(run_data_validation, "validate_oulad_raw_data", lambda *_args: result)
    monkeypatch.setattr(run_data_validation, "write_validation_artifacts", lambda *_args: None)
    monkeypatch.setattr(run_data_validation, "refresh_data_dictionary", lambda *_args: None)
    monkeypatch.setattr(run_data_validation, "refresh_decisions_log", lambda *_args: None)

    code = run_data_validation.main(
        [
            "--non-strict",
            "--metadata-dir",
            str(tmp_path / "metadata"),
            "--docs-dir",
            str(tmp_path / "docs"),
            "--log-level",
            "ERROR",
        ]
    )

    assert code == 0


def test_build_cohort_main_returns_zero_with_mocked_pipeline(monkeypatch, tmp_path: Path) -> None:
    fake_result = SimpleNamespace(
        cohort=pd.DataFrame({"id_student": [1]}),
        flow_table=pd.DataFrame({"stage": ["loaded"], "row_count": [1], "excluded_count": [0]}),
        threshold_cutoffs=pd.DataFrame(
            {
                "code_module": ["AAA"],
                "code_presentation": ["2013J"],
                "window_days": [14],
                "threshold_name": ["median"],
                "quantile": [0.5],
                "cutoff": [0.0],
                "eligible_count": [1],
            }
        ),
        summary={"cohort_size": 1},
    )
    monkeypatch.setattr(build_cohort, "load_oulad_tables", lambda **_kwargs: {})
    monkeypatch.setattr(build_cohort, "build_analytic_cohort", lambda *_args, **_kwargs: fake_result)
    monkeypatch.setattr(build_cohort, "write_cohort_flow_plot", lambda *_args: None)
    monkeypatch.setattr(build_cohort, "write_treatment_prevalence_plot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(build_cohort, "write_dag_artifacts", lambda **_kwargs: {})

    code = build_cohort.main(
        [
            "--processed-dir",
            str(tmp_path / "processed"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--log-level",
            "ERROR",
        ]
    )

    assert code == 0
    assert (tmp_path / "processed" / "treatment_threshold_cutoffs.csv").exists()


def test_stage_script_mains_return_zero_with_mocked_work(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_discovery, "run_discovery_pipeline", lambda _config: {"summary": tmp_path / "x"})
    monkeypatch.setattr(run_estimation, "run_estimation_pipeline", lambda _config: object())
    monkeypatch.setattr(run_estimation, "write_estimation_artifacts", lambda *_args, **_kwargs: {"main": tmp_path / "x"})
    monkeypatch.setattr(run_robustness, "run_robustness_pipeline", lambda _config: object())
    monkeypatch.setattr(run_robustness, "write_robustness_artifacts", lambda *_args, **_kwargs: {"main": tmp_path / "x"})

    assert run_discovery.main(["--cohort-path", str(tmp_path / "cohort.parquet"), "--log-level", "ERROR"]) == 0
    assert run_estimation.main(["--cohort-path", str(tmp_path / "cohort.parquet"), "--log-level", "ERROR"]) == 0
    assert run_robustness.main(["--cohort-path", str(tmp_path / "cohort.parquet"), "--log-level", "ERROR"]) == 0


def test_build_report_assets_main_returns_zero_with_mocked_exports(monkeypatch, tmp_path: Path) -> None:
    def fake_write(_df, output_path, *args, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("artifact\n")
        return Path(output_path)

    def fake_write_three(_df, _other, output_path, *args, **kwargs):
        return fake_write(_df, output_path, *args, **kwargs)

    monkeypatch.setattr(build_report_assets, "_require_inputs", lambda _paths: None)
    monkeypatch.setattr(build_report_assets.pd, "read_csv", lambda _path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(build_report_assets.pd, "read_parquet", lambda _path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(build_report_assets, "cohort_flow_report_table", fake_write)
    monkeypatch.setattr(build_report_assets, "write_treatment_prevalence_figure", fake_write)
    monkeypatch.setattr(build_report_assets, "write_discovery_comparison_figure", fake_write_three)
    monkeypatch.setattr(build_report_assets, "main_effect_report_table", fake_write)
    monkeypatch.setattr(build_report_assets, "robustness_report_table", fake_write)
    monkeypatch.setattr(build_report_assets, "write_subgroup_summary_figure", fake_write)

    code = build_report_assets.main(
        [
            "--processed-dir",
            str(tmp_path / "processed"),
            "--tables-dir",
            str(tmp_path / "tables"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--docs-dir",
            str(tmp_path / "docs"),
            "--log-level",
            "ERROR",
        ]
    )

    assert code == 0
