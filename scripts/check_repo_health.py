"""Verify that expected end-to-end pipeline artifacts exist."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.config import PROJECT_ROOT as DEFAULT_PROJECT_ROOT
from oulad_causal.logging_utils import add_log_level_argument, configure_logging
from oulad_causal.pipeline import (
    PIPELINE_STAGES,
    check_expected_artifacts,
    format_health_check,
    stage_names,
)


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--stage",
        action="append",
        choices=PIPELINE_STAGES,
        help="Restrict the check to one stage. May be provided multiple times.",
    )
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the repository health check."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    stages = stage_names(args.stage)
    result = check_expected_artifacts(project_root=args.project_root, stages=stages)
    message = format_health_check(result, project_root=args.project_root)
    if result.passed:
        LOGGER.info(message)
        return 0
    LOGGER.error(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
