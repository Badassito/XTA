"""External policy controls. Importing this module never imports Torch or a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from .augmentation_policy import (
    AugmentationDefinition, assert_augmentation_definition_unchanged,
    inspect_augmentation_definition,
)


def add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    """The shared PTA/TTA policy selector and total-copy ratio."""
    parser.add_argument('--augmentation', default=None, metavar='POLICY.py',
                        help='External Python augmentation policy (TTA requires a GPU policy).')
    parser.add_argument('--augmentation_ratio', default=1.0, type=float, metavar='N',
                        help='Total copies including one unaugmented original; TTA requires an integer N >= 1.')


def add_tta_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    add_augmentation_arguments(parser)
    parser.add_argument('--augmentation_granularity', choices=('slice', 'slab', 'lease', 'view'),
                        default='slice', help='Policy re-roll scope within each view/angle/tile trajectory.')
    parser.add_argument('--augmentation_slab_slices', type=int, default=32,
                        help='Logical slices per seed group in slab mode; independent of scheduling.')
    parser.add_argument('--augmentation_cache_mib', type=int, default=512,
                        help='Per-worker LRU budget for reusable spatial maps (0 disables caching).')
    parser.add_argument('--augmentation_seed', type=int, default=0,
                        help='Reproducible external-policy run seed, independent of GPU index.')
    parser.add_argument('--augmentation_coverage', choices=('packed', 'none'), default='packed',
                        help='Persist bit-packed inverse-validity sidecars in the output directory, or retain recipes only.')


@dataclass(frozen=True)
class TtaAugmentationSettings:
    ratio: int = 1
    path: str = ''
    content_sha256: str = ''
    granularity: str = 'slice'
    slab_slices: int = 32
    seed: int = 0
    coverage: str = 'packed'
    cache_mib: int = 512

    @property
    def enabled(self) -> bool:
        return self.ratio > 1

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self), 'schema': 'xta.tta.external-augmentation/1',
            'base_pass': 0, 'augmented_passes': list(range(1, self.ratio)),
            'interpolate_augmented': False,
            'support_coordinates': 'unaugmented model raster, before output-to-processing affine',
            'support_meaning': 'locally invertible, converged, in-frame inverse; NOT source-volume acquisition coverage',
            'seed_scheme': 'sha256-canonical-json-v1',
        }

    def assert_unchanged(self) -> None:
        if self.path:
            from pathlib import Path
            assert_augmentation_definition_unchanged(AugmentationDefinition(
                Path(self.path), self.content_sha256, 'build_gpu_augmentation',
            ))


def resolve_tta_augmentation(args: Any, *, gpu_devices: Any, cpu_enabled: bool) -> TtaAugmentationSettings:
    ratio = float(getattr(args, 'augmentation_ratio', 1.0))
    if not math.isfinite(ratio) or ratio < 1 or not ratio.is_integer():
        raise ValueError('--augmentation_ratio in TTA must be a finite integer >= 1 (one base + N-1 augmented passes)')
    path = getattr(args, 'augmentation', None)
    if ratio > 1 and not path:
        raise ValueError('--augmentation_ratio > 1 requires --augmentation')
    slab = int(getattr(args, 'augmentation_slab_slices', 32))
    if int(getattr(args, 'augmentation_cache_mib', 512)) < 0:
        raise ValueError('--augmentation_cache_mib must be >= 0')
    if slab < 1:
        raise ValueError('--augmentation_slab_slices must be >= 1')
    definition = inspect_augmentation_definition(str(path)) if path else None
    if definition and definition.export_name != 'build_gpu_augmentation':
        raise ValueError('TTA external augmentation is GPU-only; select a GPU_*.py policy exporting build_gpu_augmentation')
    if ratio > 1 and (not gpu_devices or cpu_enabled):
        raise ValueError('TTA external augmentation requires GPU-only --device; CPU and hybrid prediction are not supported for this feature')
    return TtaAugmentationSettings(
        ratio=int(ratio), path=str(definition.path) if definition else '',
        content_sha256=definition.content_sha256 if definition else '',
        granularity=str(getattr(args, 'augmentation_granularity', 'slice')),
        slab_slices=slab, seed=int(getattr(args, 'augmentation_seed', 0)),
        coverage=str(getattr(args, 'augmentation_coverage', 'packed')),
        cache_mib=int(getattr(args, 'augmentation_cache_mib', 512)),
    )


def policy_seed(settings: TtaAugmentationSettings, *, trajectory: str,
                pass_index: int, slice_index: int, lease_start: int) -> int:
    if pass_index < 1 or pass_index >= settings.ratio:
        raise ValueError('policy seed requested for a non-augmented or nonexistent pass')
    if settings.granularity == 'slice':
        group = int(slice_index)
    elif settings.granularity == 'slab':
        group = int(slice_index) // settings.slab_slices
    elif settings.granularity == 'lease':
        group = int(lease_start)
    elif settings.granularity == 'view':
        group = 0
    else:
        raise ValueError(f'unsupported augmentation granularity: {settings.granularity}')
    value = ['xta-tta-policy-v1', settings.content_sha256, settings.seed,
             str(trajectory), int(pass_index), settings.granularity, group]
    digest = hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)
