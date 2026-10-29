#!/usr/bin/env python3
"""Retired tilted-Azimuthal forward-scatter benchmark.

Use tools/benchmark_native_pull_projection.py to probe the maintained CPU
reference/compiled native destination pull implementation. Its CLI differs;
run it with --help. That tool does not qualify CUDA performance.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_known_args(argv)
    parser.exit(2, 'This benchmark is retired: its forward-scatter backend cannot cover '
                   'native destination cells. Use tools/benchmark_native_pull_projection.py '
                   '--help for the maintained CPU projection probe. CUDA performance '
                   'requires separate qualification.\n')


if __name__ == '__main__':
    main()
