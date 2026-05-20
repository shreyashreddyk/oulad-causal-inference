"""Tests for data validation helpers."""

from pathlib import Path

import pandas as pd

from scripts.run_data_validation import refresh_data_dictionary, refresh_decisions_log
from test_io import write_fixture_csvs
from oulad_causal.io import load_oulad_tables, locate_oulad_tables
from oulad_causal.validation import validate_oulad_raw_data, write_validation_artifacts


def test_validation_module_imports() -> None:
    assert validate_oulad_raw_data


def test_validation_passes_on_complete_synthetic_fixture(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)

    result = validate_fixture(tmp_path)

    assert result.passed
    assert result.summaries["schema_validation"]["is_valid"].all()


def test_validation_reports_missing_files(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path, skip={"vle"})

    result = validate_fixture(tmp_path)

    assert not result.passed
    assert any("Missing required raw file: vle.csv" in failure for failure in result.critical_failures)


def test_validation_reports_missing_expected_columns(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)
    student_info_path = tmp_path / "studentInfo.csv"
    student_info = pd.read_csv(student_info_path)
    student_info = student_info.drop(columns=["final_result"])
    student_info.to_csv(student_info_path, index=False)

    result = validate_fixture(tmp_path)

    schema = result.summaries["schema_validation"]
    student_info_schema = schema.loc[schema["table_name"] == "studentInfo"].iloc[0]
    assert student_info_schema["missing_columns"] == "final_result"
    assert not result.passed


def test_validation_reports_duplicate_keys_for_unique_tables(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)
    courses_path = tmp_path / "courses.csv"
    courses = pd.read_csv(courses_path)
    pd.concat([courses, courses.iloc[[0]]], ignore_index=True).to_csv(courses_path, index=False)

    result = validate_fixture(tmp_path)

    duplicates = result.summaries["duplicate_key_summary"]
    courses_duplicates = duplicates.loc[duplicates["table_name"] == "courses"].iloc[0]
    assert courses_duplicates["duplicate_rows"] == 2
    assert courses_duplicates["status"] == "failed"
    assert not result.passed


def test_student_vle_repeated_rows_are_not_duplicate_key_failures(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)

    result = validate_fixture(tmp_path)

    duplicates = result.summaries["duplicate_key_summary"]
    student_vle_duplicates = duplicates.loc[duplicates["table_name"] == "studentVle"].iloc[0]
    assert student_vle_duplicates["status"] == "not_applicable"
    assert not any("studentVle has" in failure for failure in result.critical_failures)


def test_missingness_category_and_date_summaries(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)

    result = validate_fixture(tmp_path)

    missingness = result.summaries["missingness_summary"]
    imd_missingness = missingness[
        (missingness["table_name"] == "studentInfo") & (missingness["column"] == "imd_band")
    ].iloc[0]
    assert imd_missingness["missing_count"] == 1

    categories = result.summaries["category_frequency_summary"]
    pass_frequency = categories[
        (categories["table_name"] == "studentInfo")
        & (categories["column"] == "final_result")
        & (categories["category"] == "Pass")
    ].iloc[0]
    assert pass_frequency["count"] == 1

    dates = result.summaries["date_range_summary"]
    registration_dates = dates[
        (dates["table_name"] == "studentRegistration") & (dates["column"] == "date_registration")
    ].iloc[0]
    assert registration_dates["min"] == -20.0
    assert registration_dates["max"] == -5.0


def test_validation_artifacts_and_docs_are_written(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    metadata_dir = tmp_path / "metadata"
    docs_dir = tmp_path / "docs"
    raw_dir.mkdir()
    docs_dir.mkdir()
    (docs_dir / "data_dictionary.md").write_text("# Data Dictionary\n\nManual note.\n")
    (docs_dir / "decisions_log.md").write_text("# Decisions Log\n\nManual note.\n")
    write_fixture_csvs(raw_dir)
    result = validate_fixture(raw_dir)

    write_validation_artifacts(result, metadata_dir)
    refresh_data_dictionary(docs_dir / "data_dictionary.md", result)
    refresh_decisions_log(docs_dir / "decisions_log.md", result, raw_dir, raw_dir)

    assert (metadata_dir / "raw_file_inventory.csv").exists()
    assert (metadata_dir / "data_validation_summary.json").exists()
    assert "Generated Raw OULAD Data Audit" in (docs_dir / "data_dictionary.md").read_text()
    assert "Generated Data Validation Decision" in (docs_dir / "decisions_log.md").read_text()
    assert "Manual note." in (docs_dir / "decisions_log.md").read_text()


def validate_fixture(raw_source: Path):
    locations = locate_oulad_tables(raw_source=raw_source)
    tables = load_oulad_tables(raw_source=raw_source)
    return validate_oulad_raw_data(tables, locations)
