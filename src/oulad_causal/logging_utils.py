"""Shared logging helpers for command-line pipeline scripts."""

from __future__ import annotations

import argparse
import logging


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def add_log_level_argument(parser: argparse.ArgumentParser, *, default: str = "INFO") -> None:
    """Add the standard log-level option to a CLI parser."""

    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=default,
        help="Logging verbosity. Defaults to INFO.",
    )


def configure_logging(level: str = "INFO") -> None:
    """Configure process logging with a compact pipeline-friendly format."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
