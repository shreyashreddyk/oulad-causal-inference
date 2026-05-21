"""Tests for reproducibility pipeline helpers and script CLIs."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from oulad_causal import pipeline
from scripts import check_repo_health, run_pipeline


def test_expected_artifact_health_check_passes_when_inventory_exists(tmp_path: Path) -> None:
    for relative_paths in pipeline.EXPECTED_ARTIFACTS.values():
        for relative in relative_paths:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n")

    result = pipeline.check_expected_artifacts(project_root=tmp_path)

    assert result.passed
    assert "passed" in pipeline.format_health_check(result, project_root=tmp_path)


def test_expected_artifact_health_check_groups_missing_files_by_stage(tmp_path: Path) -> None:
    result = pipeline.check_expected_artifacts(project_root=tmp_path, stages=("estimation",))
    message = pipeline.format_health_check(result, project_root=tmp_path)

    assert not result.passed
    assert "[estimation] Run `make run-estimation`" in message
    assert "data/processed/effect_estimates_main.csv" in message


def test_raw_data_missing_message_points_to_archive_location(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    assert not pipeline.raw_data_available(raw_data_dir=raw_dir)
    message = pipeline.raw_data_missing_message(raw_data_dir=raw_dir)
    assert "anonymisedData.zip" in message
    assert "courses.csv" in message


def test_check_repo_health_main_returns_nonzero_for_missing_artifacts(tmp_path: Path) -> None:
    code = check_repo_health.main(["--project-root", str(tmp_path), "--stage", "validation", "--log-level", "ERROR"])

    assert code == 1


def test_run_pipeline_fails_gracefully_when_raw_data_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run_pipeline.main(["--raw-data-dir", str(tmp_path / "raw"), "--log-level", "ERROR"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Raw OULAD data are missing or incomplete" in captured.err
    assert "Traceback" not in captured.err


def test_run_pipeline_invokes_stages_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(run_pipeline, "raw_data_available", lambda **_kwargs: True)

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    code = run_pipeline.main(["--raw-data-dir", str(tmp_path), "--log-level", "ERROR"])

    assert code == 0
    assert [Path(call[1]).name for call in calls] == [
        "run_data_validation.py",
        "build_cohort.py",
        "run_discovery.py",
        "run_estimation.py",
        "run_robustness.py",
        "build_report_assets.py",
        "check_repo_health.py",
    ]


def test_makefile_exposes_required_reproducibility_targets() -> None:
    text = Path("Makefile").read_text()

    for target in (
        "validate-data:",
        "build-cohort:",
        "run-discovery:",
        "run-estimation:",
        "run-robustness:",
        "build-assets:",
        "health-check:",
        "all:",
    ):
        assert target in text
    assert "scripts/run_pipeline.py" in text
