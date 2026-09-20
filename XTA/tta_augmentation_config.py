"""External policy controls. Importing this module never imports Torch or a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .augmentation_policy import (
    AugmentationDefinition, assert_augmentation_definition_unchanged,
    inspect_augmentation_definition, resolve_augmentation_definitions,
)


def add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    """The shared PTA/TTA policy selector and total-copy ratio."""
    parser.add_argument('--augmentation', default=None, nargs='+', action='extend',
                        metavar='BACKEND:POLICY.py',
                        help='External policies: cpu:POLICY.py gpu:POLICY.py. TTA requires an entry for each active inference backend.')
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
    cpu_path: str = ''
    cpu_sha256: str = ''
    cpu_export_name: str = 'build_augmentation'
    gpu_path: str = ''
    gpu_sha256: str = ''
    export_name: str = 'build_gpu_augmentation'
    backend: str = ''

    @property
    def enabled(self) -> bool:
        return self.ratio > 1

    def for_backend(self, backend: str) -> 'TtaAugmentationSettings':
        backend = str(backend).lower()
        if backend not in {'cpu', 'gpu'}:
            raise ValueError(f'unsupported augmentation backend: {backend}')
        path = self.cpu_path if backend == 'cpu' else self.gpu_path
        digest = self.cpu_sha256 if backend == 'cpu' else self.gpu_sha256
        export = self.cpu_export_name if backend == 'cpu' else 'build_gpu_augmentation'
        if not path and not (self.cpu_path or self.gpu_path):
            path, digest = self.path, self.content_sha256
            export = self.export_name
        if self.enabled and not path:
            raise ValueError(f'--augmentation requires a {backend}: policy for {backend.upper()} inference')
        return replace(self, path=path, content_sha256=digest, export_name=export, backend=backend)

    def definitions(self) -> dict[str, AugmentationDefinition]:
        from pathlib import Path
        definitions = {}
        if self.cpu_path:
            definitions['cpu'] = AugmentationDefinition(Path(self.cpu_path), self.cpu_sha256, self.cpu_export_name)
        if self.gpu_path:
            definitions['gpu'] = AugmentationDefinition(Path(self.gpu_path), self.gpu_sha256, 'build_gpu_augmentation')
        if not definitions and self.path:
            key = self.backend or ('gpu' if self.export_name == 'build_gpu_augmentation' else 'cpu')
            definitions[key] = AugmentationDefinition(Path(self.path), self.content_sha256, self.export_name)
        return definitions

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self), 'schema': 'xta.tta.external-augmentation/2',
            'policies': {backend: {'path': str(definition.path), 'sha256': definition.content_sha256,
                                    'export_name': definition.export_name}
                         for backend, definition in self.definitions().items()},
            'base_pass': 0, 'augmented_passes': list(range(1, self.ratio)),
            'interpolate_augmented': False,
            'support_coordinates': 'unaugmented model raster, before output-to-processing affine',
            'support_meaning': 'locally invertible, converged, in-frame inverse; NOT source-volume acquisition coverage',
            'seed_scheme': 'sha256-canonical-json-v1',
        }

    def assert_unchanged(self) -> None:
        for definition in self.definitions().values():
            assert_augmentation_definition_unchanged(definition)


def resolve_tta_augmentation(args: Any, *, gpu_devices: Any, cpu_enabled: bool) -> TtaAugmentationSettings:
    ratio = float(getattr(args, 'augmentation_ratio', 1.0))
    if not math.isfinite(ratio) or ratio < 1 or not ratio.is_integer():
        raise ValueError('--augmentation_ratio in TTA must be a finite integer >= 1 (one base + N-1 augmented passes)')
    entries = getattr(args, 'augmentation', None)
    if ratio > 1 and not entries:
        raise ValueError('--augmentation_ratio > 1 requires --augmentation')
    slab = int(getattr(args, 'augmentation_slab_slices', 32))
    if int(getattr(args, 'augmentation_cache_mib', 512)) < 0:
        raise ValueError('--augmentation_cache_mib must be >= 0')
    if slab < 1:
        raise ValueError('--augmentation_slab_slices must be >= 1')
    definitions = resolve_augmentation_definitions(entries)
    if ratio > 1:
        if not gpu_devices and not cpu_enabled:
            raise ValueError('TTA external augmentation requires an active CPU or GPU inference backend')
        for backend, active in (('cpu', cpu_enabled), ('gpu', bool(gpu_devices))):
            if active and backend not in definitions:
                raise ValueError(f'--augmentation requires a {backend}: policy for {backend.upper()} inference')
    cpu = definitions.get('cpu')
    gpu = definitions.get('gpu')
    selected = gpu if gpu is not None else cpu
    digest = selected.content_sha256 if selected else ''
    path = str(selected.path) if selected else ''
    if cpu is not None and gpu is not None:
        digest = hashlib.sha256(json.dumps({k: d.content_sha256 for k, d in sorted(definitions.items())},
                                          sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        path = ''
    return TtaAugmentationSettings(
        ratio=int(ratio), path=path, content_sha256=digest,
        granularity=str(getattr(args, 'augmentation_granularity', 'slice')),
        slab_slices=slab, seed=int(getattr(args, 'augmentation_seed', 0)),
        coverage=str(getattr(args, 'augmentation_coverage', 'packed')),
        cache_mib=int(getattr(args, 'augmentation_cache_mib', 512)),
        cpu_path=str(cpu.path) if cpu else '', cpu_sha256=cpu.content_sha256 if cpu else '',
        cpu_export_name=cpu.export_name if cpu else 'build_augmentation',
        gpu_path=str(gpu.path) if gpu else '', gpu_sha256=gpu.content_sha256 if gpu else '',
        export_name=selected.export_name if selected else 'build_gpu_augmentation',
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
