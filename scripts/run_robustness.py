"""Run robustness checks for the OULAD early-engagement estimand."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/oulad_causal_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/oulad_causal_xdg_cache")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR
from oulad_causal.dag import ANALYTIC_COHORT_PATH
from oulad_causal.logging_utils import add_log_level_argument, configure_logging
from oulad_causal.robustness import RobustnessConfig, run_robustness_pipeline, write_robustness_artifacts


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-path", type=Path, default=ANALYTIC_COHORT_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--seed", type=int, default=245)
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic robustness pipeline."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    config = RobustnessConfig(
        cohort_path=args.cohort_path,
        processed_dir=args.processed_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        docs_dir=args.docs_dir,
        seed=args.seed,
    )
    result = run_robustness_pipeline(config)
    paths = write_robustness_artifacts(result, config=config)

    LOGGER.info("Wrote robustness artifacts:")
    for name, path in paths.items():
        LOGGER.info("- %s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
