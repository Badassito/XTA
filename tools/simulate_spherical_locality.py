"""Check Spherical range ownership through the native scheduler on one host.

Each Spherical parent submits the retained 1,212-frame, 22-lease shape. Spawned
host actors echo the exact assigned ranges back through validated native receipts.
The tool exercises parent-worker residency and credits without loading CUDA,
constructing Spherical plans, or estimating avoided cache builds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.profile_tta_dispatch import TaskSpec, run_replay


_FRAMES_PER_PARENT = 1212
_FRAMES_PER_TASK = 57


def simulate(*, parents=120, workers=4, view_limit=4, max_seconds=180.0):
    if not 1 <= parents <= 120:
        raise ValueError('parents must be from 1 through 120')
    specs = []
    for parent_id in range(parents):
        view_name = f'spherical_replay_{parent_id:03d}__tta_a0'
        for start in range(0, _FRAMES_PER_PARENT, _FRAMES_PER_TASK):
            count = min(_FRAMES_PER_TASK, _FRAMES_PER_PARENT - start)
            specs.append(TaskSpec(len(specs), view_name, 'spherical', start, count))
    replay = run_replay(specs, workers=workers, ticket_capacity=view_limit,
                        max_seconds=max_seconds, resident_group=True)
    if replay['max_workers_per_parent'] != 1:
        raise RuntimeError('Native Spherical parent was not resident on one actor')
    return {
        'schema': 'xta.spherical-native-ownership/1',
        'scope': __doc__.strip(),
        'verified_frames': parents * _FRAMES_PER_PARENT,
        'frames_per_parent': _FRAMES_PER_PARENT,
        'requested_parent_ticket_limit': view_limit,
        'native_replay': replay,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parents', type=int, default=120)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--view-limit', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=180.)
    args = parser.parse_args()
    if not 1 <= args.parents <= 120 or args.workers < 1 or args.view_limit < 1:
        parser.error('Use 1–120 parents and positive worker and view limits')
    result = simulate(parents=args.parents, workers=args.workers,
                      view_limit=args.view_limit, max_seconds=args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
