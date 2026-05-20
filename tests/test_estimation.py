"""Tests for treatment effect estimation helpers."""

from oulad_causal import config, estimation


def test_estimation_module_imports() -> None:
    assert estimation.__doc__


def test_project_root_contains_pyproject() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
