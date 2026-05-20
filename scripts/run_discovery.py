"""Run reduced causal discovery analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

from oulad_causal.config import DOCS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from oulad_causal.dag import ANALYTIC_COHORT_PATH
from oulad_causal.discovery import DiscoveryConfig, run_discovery_pipeline


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    """Run the deterministic discovery pipeline."""

    args = parse_args()
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
    print("Wrote discovery artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

