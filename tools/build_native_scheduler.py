"""Prepare the required native TTA scheduler for a source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="External build cache (default: XTA_NATIVE_SCHEDULER_CACHE or user cache).")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if (root / "setup.py").is_file():
        sys.path.insert(0, str(root))
    from XTA.native_scheduler_build import NativeSchedulerBuildError, build_native_scheduler

    try:
        print(build_native_scheduler(cache_dir=args.cache_dir))
    except NativeSchedulerBuildError as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
