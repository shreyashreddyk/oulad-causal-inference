"""Input/output utilities for project artifacts.

Functions here should read and write tabular data, metadata, and model outputs
using explicit paths. Avoid silently inventing raw data locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import pandas as pd

from oulad_causal.config import RAW_DATA_DIR


@dataclass(frozen=True)
class OuladTableSpec:
    """Schema metadata for one standard OULAD raw CSV table."""

    name: str
    filename: str
    columns: tuple[str, ...]
    unique_key: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()


OULAD_TABLE_SPECS: dict[str, OuladTableSpec] = {
    "courses": OuladTableSpec(
        name="courses",
        filename="courses.csv",
        columns=("code_module", "code_presentation", "module_presentation_length"),
        unique_key=("code_module", "code_presentation"),
    ),
    "assessments": OuladTableSpec(
        name="assessments",
        filename="assessments.csv",
        columns=(
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ),
        unique_key=("id_assessment",),
        date_columns=("date",),
        categorical_columns=("assessment_type",),
    ),
    "vle": OuladTableSpec(
        name="vle",
        filename="vle.csv",
        columns=("id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"),
        unique_key=("id_site",),
        categorical_columns=("activity_type",),
    ),
    "studentInfo": OuladTableSpec(
        name="studentInfo",
        filename="studentInfo.csv",
        columns=(
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ),
        unique_key=("code_module", "code_presentation", "id_student"),
        categorical_columns=(
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "disability",
            "final_result",
        ),
    ),
    "studentRegistration": OuladTableSpec(
        name="studentRegistration",
        filename="studentRegistration.csv",
        columns=("code_module", "code_presentation", "id_student", "date_registration", "date_unregistration"),
        unique_key=("code_module", "code_presentation", "id_student"),
        date_columns=("date_registration", "date_unregistration"),
    ),
    "studentAssessment": OuladTableSpec(
        name="studentAssessment",
        filename="studentAssessment.csv",
        columns=("id_assessment", "id_student", "date_submitted", "is_banked", "score"),
        unique_key=("id_assessment", "id_student"),
        date_columns=("date_submitted",),
    ),
    "studentVle": OuladTableSpec(
        name="studentVle",
        filename="studentVle.csv",
        columns=("code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"),
        date_columns=("date",),
    ),
}


@dataclass(frozen=True)
class RawTableLocation:
    """Where a raw OULAD table was found."""

    table_name: str
    filename: str
    found: bool
    source_type: str
    source_path: Path | None
    archive_member: str | None = None


def resolve_raw_source(raw_source: str | Path | None = None, *, raw_data_dir: str | Path | None = None) -> Path:
    """Resolve a raw OULAD source path.

    Resolution order is explicit source, then ``anonymisedData.zip`` in the raw
    directory, then the raw directory itself for extracted CSV files.
    """

    if raw_source is not None:
        source = Path(raw_source).expanduser().resolve()
        return source

    raw_dir = Path(raw_data_dir or RAW_DATA_DIR).expanduser().resolve()
    archive = raw_dir / "anonymisedData.zip"
    if archive.exists():
        return archive
    return raw_dir


def locate_oulad_tables(raw_source: str | Path | None = None, *, raw_data_dir: str | Path | None = None) -> list[RawTableLocation]:
    """Return source locations for every expected OULAD table."""

    source = resolve_raw_source(raw_source, raw_data_dir=raw_data_dir)
    if source.is_file() and is_zipfile(source):
        return _locate_tables_in_zip(source)
    return _locate_tables_in_directory(source)


def load_oulad_table(
    table_name: str,
    raw_source: str | Path | None = None,
    *,
    raw_data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load one standard OULAD table as a pandas DataFrame."""

    if table_name not in OULAD_TABLE_SPECS:
        expected = ", ".join(OULAD_TABLE_SPECS)
        raise KeyError(f"Unknown OULAD table {table_name!r}. Expected one of: {expected}.")

    source = resolve_raw_source(raw_source, raw_data_dir=raw_data_dir)
    spec = OULAD_TABLE_SPECS[table_name]
    if source.is_file() and is_zipfile(source):
        with ZipFile(source) as archive:
            member = _find_archive_member(archive, spec.filename)
            if member is None:
                raise FileNotFoundError(f"Missing {spec.filename} in archive {source}.")
            with archive.open(member) as handle:
                return pd.read_csv(handle)

    path = source / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {spec.filename} under raw source {source}.")
    return pd.read_csv(path)


def load_oulad_tables(
    raw_source: str | Path | None = None,
    *,
    raw_data_dir: str | Path | None = None,
    include_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load all available standard OULAD tables keyed by table name."""

    locations = locate_oulad_tables(raw_source, raw_data_dir=raw_data_dir)
    tables: dict[str, pd.DataFrame] = {}
    for location in locations:
        if not location.found:
            if include_missing:
                tables[location.table_name] = pd.DataFrame()
            continue
        tables[location.table_name] = load_oulad_table(
            location.table_name,
            raw_source=location.source_path,
            raw_data_dir=raw_data_dir,
        )
    return tables


def _locate_tables_in_zip(source: Path) -> list[RawTableLocation]:
    with ZipFile(source) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
    locations = []
    for spec in OULAD_TABLE_SPECS.values():
        member = members.get(spec.filename)
        locations.append(
            RawTableLocation(
                table_name=spec.name,
                filename=spec.filename,
                found=member is not None,
                source_type="zip",
                source_path=source,
                archive_member=member,
            )
        )
    return locations


def _locate_tables_in_directory(source: Path) -> list[RawTableLocation]:
    locations = []
    for spec in OULAD_TABLE_SPECS.values():
        path = source / spec.filename
        locations.append(
            RawTableLocation(
                table_name=spec.name,
                filename=spec.filename,
                found=path.exists(),
                source_type="directory",
                source_path=source,
            )
        )
    return locations


def _find_archive_member(archive: ZipFile, filename: str) -> str | None:
    for member in archive.namelist():
        if Path(member).name == filename:
            return member
    return None
