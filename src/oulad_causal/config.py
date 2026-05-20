"""Project configuration helpers.

This module should hold path constants and lightweight configuration loading.
It must not hard-code raw OULAD file paths beyond repository-relative defaults.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class ProjectPaths:
    """Repository paths used by deterministic pipeline scripts."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    metadata_dir: Path = METADATA_DIR
    docs_dir: Path = DOCS_DIR

    @classmethod
    def from_overrides(
        cls,
        *,
        raw_data_dir: str | Path | None = None,
        metadata_dir: str | Path | None = None,
        docs_dir: str | Path | None = None,
    ) -> "ProjectPaths":
        """Build path config from CLI-style overrides and environment defaults."""

        raw_override = raw_data_dir or os.environ.get("OULAD_RAW_DATA_DIR")
        return cls(
            raw_data_dir=_resolve_path(raw_override, default=RAW_DATA_DIR),
            metadata_dir=_resolve_path(metadata_dir, default=METADATA_DIR),
            docs_dir=_resolve_path(docs_dir, default=DOCS_DIR),
        )


def _resolve_path(value: str | Path | None, *, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
