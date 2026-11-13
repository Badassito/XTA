"""Package existing scalar TTA runs as a stock-Slicer confidence review."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XTA.config import _confidence_hundredths
from XTA.sweep_publication import publish_sweep_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Example: --root STUDY --branch 0.50 run_A --branch 0.80 run_B. "
        "Move the common study directory as a unit to preserve source references for edit transfer."
    ))
    parser.add_argument("--root", type=Path, required=True,
                        help="Existing common study directory containing every supplied scalar run")
    parser.add_argument("--branch", action="append", nargs=2, metavar=("CUTOFF", "RUN_DIR"), required=True,
                        help="Decimal cutoff and scalar output directory, relative to --root or absolute; repeat per cutoff")
    parser.add_argument("--name", default="review", help="Fresh review folder under --root (default: review)")
    parser.add_argument("--memory-mib", type=float, default=128, help="Grouped writer numerical slab budget")
    parser.add_argument("--temp", type=Path,
                        help="Fast temporary directory for private mask/coverage spools (default: system temp)")
    parser.add_argument("--workers", type=int,
                        help="Cutoffs packaged concurrently, each with its own --memory-mib budget "
                             "(default: one per cutoff, at most 8 and at most the CPU count)")
    args = parser.parse_args(argv)
    study = args.root.expanduser().resolve()
    if not study.is_dir():
        parser.error("--root must be an existing study directory")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    try:
        branches = [(_confidence_hundredths(c, "branch cutoff"), study / Path(p).expanduser())
                    for c, p in args.branch]
        # With a low-quality tier, native stays reference-only: no native payload is read.
        output = publish_sweep_review(study, branches, sweep_id=uuid.uuid4().hex,
                                      source_mode="independent_scalar", review_name=args.name,
                                      memory_mib=args.memory_mib, temp_dir=args.temp,
                                      workers=args.workers)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
