"""Run the full reproducible OULAD causal pipeline in order."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.logging_utils import add_log_level_argument, configure_logging
from oulad_causal.pipeline import raw_data_available, raw_data_missing_message


LOGGER = logging.getLogger(__name__)

PIPELINE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("validate-data", ("scripts/run_data_validation.py",)),
    ("build-cohort", ("scripts/build_cohort.py",)),
    ("run-discovery", ("scripts/run_discovery.py",)),
    ("run-estimation", ("scripts/run_estimation.py",)),
    ("run-robustness", ("scripts/run_robustness.py",)),
    ("build-assets", ("scripts/build_report_assets.py",)),
    ("health-check", ("scripts/check_repo_health.py",)),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source", help="Explicit OULAD archive or extracted CSV directory.")
    parser.add_argument("--raw-data-dir", help="Raw data directory. Defaults to data/raw or OULAD_RAW_DATA_DIR.")
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Run the pipeline stages but skip the final artifact health check.",
    )
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full pipeline with clear missing-data and stage-failure messages."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    if not raw_data_available(raw_source=args.raw_source, raw_data_dir=args.raw_data_dir):
        LOGGER.error(raw_data_missing_message(raw_source=args.raw_source, raw_data_dir=args.raw_data_dir))
        return 2

    for target, command in PIPELINE_COMMANDS:
        if target == "health-check" and args.skip_health_check:
            continue
        full_command = _command_with_common_options(command, args)
        LOGGER.info("Running %s", target)
        completed = subprocess.run(full_command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            LOGGER.error("Pipeline stopped at `%s` with exit code %s.", target, completed.returncode)
            LOGGER.error("After fixing the issue, rerun `%s` or `make all`.", target)
            return completed.returncode

    LOGGER.info("Full pipeline completed successfully.")
    return 0


def _command_with_common_options(command: tuple[str, ...], args: argparse.Namespace) -> list[str]:
    full = [sys.executable, *command, "--log-level", args.log_level]
    script = command[0]
    if script in {"scripts/run_data_validation.py", "scripts/build_cohort.py"}:
        if args.raw_source:
            full.extend(["--raw-source", args.raw_source])
        if args.raw_data_dir:
            full.extend(["--raw-data-dir", args.raw_data_dir])
    return full


if __name__ == "__main__":
    raise SystemExit(main())
