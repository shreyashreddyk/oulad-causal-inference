"""Run reduced causal discovery analyses."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from oulad_causal.dag import ANALYTIC_COHORT_PATH
from oulad_causal.discovery import DiscoveryConfig, run_discovery_pipeline
from oulad_causal.logging_utils import add_log_level_argument, configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-path", type=Path, default=ANALYTIC_COHORT_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--seed", type=int, default=245)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--stability-reps", type=int, default=20)
    parser.add_argument("--stability-sample-size", type=int, default=3000)
    parser.add_argument("--skip-fci", action="store_true")
    add_log_level_argument(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic discovery pipeline."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    paths = run_discovery_pipeline(
        DiscoveryConfig(
            cohort_path=args.cohort_path,
            processed_dir=args.processed_dir,
            figures_dir=args.figures_dir,
            docs_dir=args.docs_dir,
            alpha=args.alpha,
            seed=args.seed,
            stability_reps=args.stability_reps,
            stability_sample_size=args.stability_sample_size,
            skip_fci=args.skip_fci,
        )
    )
    LOGGER.info("Wrote discovery artifacts:")
    for name, path in paths.items():
        LOGGER.info("- %s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
