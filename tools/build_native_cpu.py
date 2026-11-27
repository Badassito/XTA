"""Build the ahead-of-time native CPU extension for a source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir", type=Path,
        help="Outside-checkout build directory (default: XTA_NATIVE_CPU_ARTIFACTS or user cache)",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if (root / "setup.py").is_file():
        sys.path.insert(0, str(root))
    from XTA.native_cpu_build import NativeCpuBuildError, build_native_cpu

    try:
        print(build_native_cpu(artifacts_dir=args.artifacts_dir))
    except NativeCpuBuildError as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
