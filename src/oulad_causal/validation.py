"""Data validation routines.

This module will contain schema checks, missingness checks, label audits, and
cohort integrity checks used before downstream causal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from oulad_causal.io import OULAD_TABLE_SPECS, RawTableLocation


DATE_WARNING_MIN = -400
DATE_WARNING_MAX = 700


@dataclass
class ValidationResult:
    """Collection of validation summary tables and status messages."""

    summaries: dict[str, pd.DataFrame]
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.critical_failures


def validate_oulad_raw_data(
    tables: Mapping[str, pd.DataFrame],
    locations: list[RawTableLocation],
) -> ValidationResult:
    """Validate available raw OULAD tables and return audit summaries."""

    critical_failures: list[str] = []
    warnings: list[str] = []

    file_inventory = summarize_file_presence(locations)
    missing_files = file_inventory.loc[~file_inventory["found"], "filename"].tolist()
    for filename in missing_files:
        critical_failures.append(f"Missing required raw file: {filename}")

    table_shapes = summarize_table_shapes(tables)
    schema_validation = summarize_schema(tables)
    for row in schema_validation.itertuples(index=False):
        if row.missing_columns:
            critical_failures.append(
                f"{row.table_name} is missing expected columns: {row.missing_columns}"
            )
        if row.extra_columns:
            warnings.append(f"{row.table_name} has unexpected extra columns: {row.extra_columns}")
        if not row.column_order_matches and not row.missing_columns and not row.extra_columns:
            warnings.append(f"{row.table_name} columns are present but not in the standard OULAD order.")

    duplicate_key_summary = summarize_duplicate_keys(tables)
    for row in duplicate_key_summary.itertuples(index=False):
        if row.checked and row.duplicate_rows > 0:
            critical_failures.append(
                f"{row.table_name} has {row.duplicate_rows} duplicate rows for key {row.key_columns}"
            )

    missingness_summary = summarize_missingness(tables)
    date_range_summary = summarize_date_ranges(tables)
    for row in date_range_summary.itertuples(index=False):
        if row.warning:
            warnings.append(f"{row.table_name}.{row.column}: {row.warning}")

    category_frequency_summary = summarize_category_frequencies(tables)

    return ValidationResult(
        summaries={
            "raw_file_inventory": file_inventory,
            "table_shapes": table_shapes,
            "schema_validation": schema_validation,
            "duplicate_key_summary": duplicate_key_summary,
            "missingness_summary": missingness_summary,
            "date_range_summary": date_range_summary,
            "category_frequency_summary": category_frequency_summary,
        },
        critical_failures=critical_failures,
        warnings=warnings,
    )


def summarize_file_presence(locations: list[RawTableLocation]) -> pd.DataFrame:
    """Summarize whether each expected raw OULAD file is present."""

    rows = [
        {
            "table_name": location.table_name,
            "filename": location.filename,
            "found": bool(location.found),
            "source_type": location.source_type,
            "source_path": str(location.source_path) if location.source_path else "",
            "archive_member": location.archive_member or "",
            "status": "found" if location.found else "missing",
        }
        for location in locations
    ]
    return pd.DataFrame(rows)


def summarize_table_shapes(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize row and column counts for loaded tables."""

    rows = []
    for table_name, spec in OULAD_TABLE_SPECS.items():
        table = tables.get(table_name)
        rows.append(
            {
                "table_name": table_name,
                "loaded": table is not None,
                "row_count": int(len(table)) if table is not None else 0,
                "column_count": int(len(table.columns)) if table is not None else 0,
                "expected_column_count": len(spec.columns),
            }
        )
    return pd.DataFrame(rows)


def summarize_schema(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare actual raw table columns with exact standard OULAD columns."""

    rows = []
    for table_name, spec in OULAD_TABLE_SPECS.items():
        table = tables.get(table_name)
        actual = list(table.columns) if table is not None else []
        expected = list(spec.columns)
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        rows.append(
            {
                "table_name": table_name,
                "loaded": table is not None,
                "expected_columns": ", ".join(expected),
                "actual_columns": ", ".join(actual),
                "missing_columns": ", ".join(missing),
                "extra_columns": ", ".join(extra),
                "column_order_matches": actual == expected,
                "is_valid": table is not None and not missing and not extra and actual == expected,
            }
        )
    return pd.DataFrame(rows)


def summarize_duplicate_keys(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize duplicate key rows for tables with declared unique keys."""

    rows = []
    for table_name, spec in OULAD_TABLE_SPECS.items():
        table = tables.get(table_name)
        key_columns = list(spec.unique_key)
        checked = table is not None and bool(key_columns) and all(column in table.columns for column in key_columns)
        duplicate_rows = 0
        duplicate_groups = 0
        missing_key_columns: list[str] = []

        if table is None:
            status = "not_loaded"
        elif not key_columns:
            status = "not_applicable"
        else:
            missing_key_columns = [column for column in key_columns if column not in table.columns]
            if missing_key_columns:
                status = "skipped_missing_key_columns"
            else:
                duplicate_mask = table.duplicated(subset=key_columns, keep=False)
                duplicate_rows = int(duplicate_mask.sum())
                if duplicate_rows:
                    duplicate_groups = int(table.loc[duplicate_mask, key_columns].drop_duplicates().shape[0])
                status = "failed" if duplicate_rows else "passed"

        rows.append(
            {
                "table_name": table_name,
                "key_columns": ", ".join(key_columns),
                "checked": checked,
                "duplicate_rows": duplicate_rows,
                "duplicate_groups": duplicate_groups,
                "missing_key_columns": ", ".join(missing_key_columns),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize missing values for every loaded table column."""

    rows = []
    for table_name in OULAD_TABLE_SPECS:
        table = tables.get(table_name)
        if table is None:
            continue
        row_count = len(table)
        for column in table.columns:
            missing_count = int(table[column].isna().sum())
            rows.append(
                {
                    "table_name": table_name,
                    "column": column,
                    "dtype": str(table[column].dtype),
                    "row_count": row_count,
                    "missing_count": missing_count,
                    "nonmissing_count": row_count - missing_count,
                    "missing_fraction": missing_count / row_count if row_count else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_date_ranges(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize relative-day date columns and flag coarse sanity issues."""

    rows = []
    courses = tables.get("courses")
    max_course_length = _max_numeric(courses["module_presentation_length"]) if courses is not None and "module_presentation_length" in courses else None

    for table_name, spec in OULAD_TABLE_SPECS.items():
        table = tables.get(table_name)
        if table is None:
            continue
        for column in spec.date_columns:
            if column not in table.columns:
                continue
            values = pd.to_numeric(table[column], errors="coerce")
            raw_nonmissing = table[column].notna()
            nonnumeric_count = int(raw_nonmissing.sum() - values.notna().sum())
            missing_count = int(table[column].isna().sum())
            minimum = _none_if_nan(values.min())
            maximum = _none_if_nan(values.max())
            below_warning_min = int((values < DATE_WARNING_MIN).sum())
            above_warning_max = int((values > DATE_WARNING_MAX).sum())
            after_known_course_max = (
                int((values > max_course_length).sum()) if max_course_length is not None else 0
            )
            warning = _date_warning(
                nonnumeric_count=nonnumeric_count,
                below_warning_min=below_warning_min,
                above_warning_max=above_warning_max,
                after_known_course_max=after_known_course_max,
            )
            rows.append(
                {
                    "table_name": table_name,
                    "column": column,
                    "row_count": len(table),
                    "missing_count": missing_count,
                    "nonnumeric_count": nonnumeric_count,
                    "min": minimum,
                    "max": maximum,
                    "below_warning_min_count": below_warning_min,
                    "above_warning_max_count": above_warning_max,
                    "after_max_module_length_count": after_known_course_max,
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)


def summarize_category_frequencies(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize category counts for important raw categorical columns."""

    rows = []
    for table_name, spec in OULAD_TABLE_SPECS.items():
        table = tables.get(table_name)
        if table is None:
            continue
        row_count = len(table)
        for column in spec.categorical_columns:
            if column not in table.columns:
                continue
            counts = table[column].fillna("<MISSING>").value_counts(dropna=False).sort_index()
            for value, count in counts.items():
                rows.append(
                    {
                        "table_name": table_name,
                        "column": column,
                        "category": str(value),
                        "count": int(count),
                        "fraction": int(count) / row_count if row_count else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def write_validation_artifacts(result: ValidationResult, metadata_dir: str | Path) -> None:
    """Write validation summary CSV and JSON artifacts to disk."""

    output_dir = Path(metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, summary in result.summaries.items():
        summary.to_csv(output_dir / f"{name}.csv", index=False)

    status = {
        "passed": result.passed,
        "critical_failure_count": len(result.critical_failures),
        "warning_count": len(result.warnings),
        "critical_failures": result.critical_failures,
        "warnings": result.warnings,
    }
    (output_dir / "data_validation_summary.json").write_text(json.dumps(status, indent=2) + "\n")


def _max_numeric(series: pd.Series) -> float | None:
    value = pd.to_numeric(series, errors="coerce").max()
    return _none_if_nan(value)


def _none_if_nan(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _date_warning(
    *,
    nonnumeric_count: int,
    below_warning_min: int,
    above_warning_max: int,
    after_known_course_max: int,
) -> str:
    warnings = []
    if nonnumeric_count:
        warnings.append(f"{nonnumeric_count} nonnumeric values")
    if below_warning_min:
        warnings.append(f"{below_warning_min} values below {DATE_WARNING_MIN}")
    if above_warning_max:
        warnings.append(f"{above_warning_max} values above {DATE_WARNING_MAX}")
    if after_known_course_max:
        warnings.append(f"{after_known_course_max} values after the maximum module length")
    return "; ".join(warnings)
