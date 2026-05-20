"""Tests for input/output helpers."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from oulad_causal.io import OULAD_TABLE_SPECS, load_oulad_table, load_oulad_tables, locate_oulad_tables


def test_io_module_imports() -> None:
    assert OULAD_TABLE_SPECS


def test_load_oulad_tables_from_extracted_directory(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path)

    tables = load_oulad_tables(raw_source=tmp_path)
    locations = locate_oulad_tables(raw_source=tmp_path)

    assert set(tables) == set(OULAD_TABLE_SPECS)
    assert all(location.found for location in locations)
    assert list(tables["studentInfo"].columns) == list(OULAD_TABLE_SPECS["studentInfo"].columns)
    assert tables["courses"].loc[0, "code_module"] == "AAA"


def test_load_oulad_tables_from_zip_archive(tmp_path: Path) -> None:
    extracted = tmp_path / "csvs"
    extracted.mkdir()
    write_fixture_csvs(extracted)
    archive = tmp_path / "anonymisedData.zip"
    with ZipFile(archive, "w") as zip_file:
        for spec in OULAD_TABLE_SPECS.values():
            zip_file.write(extracted / spec.filename, arcname=spec.filename)

    tables = load_oulad_tables(raw_source=archive)
    locations = locate_oulad_tables(raw_source=archive)

    assert all(location.source_type == "zip" for location in locations)
    assert tables["vle"].shape[0] == 2
    assert load_oulad_table("courses", raw_source=archive).shape == (1, 3)


def test_locate_oulad_tables_reports_missing_files(tmp_path: Path) -> None:
    write_fixture_csvs(tmp_path, skip={"vle"})

    locations = locate_oulad_tables(raw_source=tmp_path)
    missing = {location.table_name for location in locations if not location.found}

    assert missing == {"vle"}


def write_fixture_csvs(base: Path, skip: set[str] | None = None) -> None:
    skip = skip or set()
    rows = fixture_rows()
    for table_name, records in rows.items():
        if table_name in skip:
            continue
        pd.DataFrame(records, columns=OULAD_TABLE_SPECS[table_name].columns).to_csv(
            base / OULAD_TABLE_SPECS[table_name].filename,
            index=False,
        )


def fixture_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "courses": [
            {"code_module": "AAA", "code_presentation": "2013J", "module_presentation_length": 268},
        ],
        "assessments": [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_assessment": 1,
                "assessment_type": "TMA",
                "date": 19,
                "weight": 10,
            }
        ],
        "vle": [
            {
                "id_site": 10,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "activity_type": "resource",
                "week_from": "",
                "week_to": "",
            },
            {
                "id_site": 11,
                "code_module": "AAA",
                "code_presentation": "2013J",
                "activity_type": "forumng",
                "week_from": "",
                "week_to": "",
            },
        ],
        "studentInfo": [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 100,
                "gender": "F",
                "region": "Scotland",
                "highest_education": "HE Qualification",
                "imd_band": "90-100%",
                "age_band": "35-55",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Pass",
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 101,
                "gender": "M",
                "region": "London Region",
                "highest_education": "A Level or Equivalent",
                "imd_band": "",
                "age_band": "0-35",
                "num_of_prev_attempts": 1,
                "studied_credits": 120,
                "disability": "Y",
                "final_result": "Fail",
            },
        ],
        "studentRegistration": [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 100,
                "date_registration": -20,
                "date_unregistration": "",
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 101,
                "date_registration": -5,
                "date_unregistration": 40,
            },
        ],
        "studentAssessment": [
            {"id_assessment": 1, "id_student": 100, "date_submitted": 18, "is_banked": 0, "score": 85},
            {"id_assessment": 1, "id_student": 101, "date_submitted": 20, "is_banked": 0, "score": 55},
        ],
        "studentVle": [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 100,
                "id_site": 10,
                "date": 0,
                "sum_click": 3,
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 100,
                "id_site": 10,
                "date": 0,
                "sum_click": 2,
            },
        ],
    }
