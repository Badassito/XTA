"""Dependency-light external reconciliation policy discovery and settings."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class ReconciliationSettings:
    path: str = ''
    sha256: str = ''
    memory_mib: int = 4096

    @property
    def enabled(self):
        return bool(self.path)

    def assert_unchanged(self):
        if self.path and hashlib.sha256(Path(self.path).read_bytes()).hexdigest() != self.sha256:
            raise RuntimeError('Reconciliation policy changed during execution')


def add_reconciliation_arguments(parser):
    parser.add_argument('--reconciliation', default=None, metavar='POLICY.py',
                        help='External policy for reconciling retained source-space layer evidence before global postprocessing.')
    parser.add_argument('--reconciliation_memory_mib', type=int, default=4096, metavar='MIB',
                        help='Working-memory budget for reconciliation voting and component statistics.')


def resolve_reconciliation(args) -> ReconciliationSettings:
    memory = int(getattr(args, 'reconciliation_memory_mib', 4096))
    if memory <= 0:
        raise ValueError('--reconciliation_memory_mib must be positive')
    raw = getattr(args, 'reconciliation', None)
    if not raw:
        return ReconciliationSettings(memory_mib=memory)
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != '.py':
        raise ValueError('--reconciliation requires a Python policy file')
    content = path.read_bytes()
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f'Invalid reconciliation policy syntax: {exc}') from exc
    exports = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'build_reconciliation']
    if len(exports) != 1:
        raise ValueError('Reconciliation policy must define build_reconciliation()')
    return ReconciliationSettings(str(path), hashlib.sha256(content).hexdigest(), memory)


def load_reconciliation_policy(settings: ReconciliationSettings) -> dict[str, Any]:
    settings.assert_unchanged()
    source = Path(settings.path).read_bytes()
    if hashlib.sha256(source).hexdigest() != settings.sha256:
        raise RuntimeError('Reconciliation policy changed while loading')
    spec = importlib.util.spec_from_file_location('_xta_reconciliation_' + settings.sha256[:24], settings.path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    parent = str(Path(settings.path).parent)
    sys.path.insert(0, parent)
    try:
        exec(compile(source, settings.path, 'exec'), module.__dict__)
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:
            pass
    result = module.build_reconciliation()
    settings.assert_unchanged()
    return validate_policy(result)


def validate_policy(value) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError('build_reconciliation() must return a policy dictionary')
    defaults = dict(name='weighted', mode='weighted', grouping='sections', threshold=1.5,
                    min_sources=2, min_prediction_sources=1, island_weighting=False,
                    island_weight_min=.75, island_weight_max=1.25,
                    angular_tolerance_deg=1., anchor_confidence=.8, anchor_bonus=0.,
                    provenance_weights={'prediction': 1., 'bridge': .35, 'mixed': .5},
                    decide=None)
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown reconciliation policy fields: {sorted(unknown)}')
    result = {**defaults, **value}
    if result['mode'] not in {'union', 'weighted', 'confidence'}:
        raise ValueError('Reconciliation mode must be union, weighted, or confidence')
    if result['grouping'] not in {'sections', 'views'}:
        raise ValueError('Reconciliation grouping must be sections or views')
    for name in ('threshold', 'island_weight_min', 'island_weight_max',
                 'angular_tolerance_deg', 'anchor_confidence', 'anchor_bonus'):
        number = float(result[name])
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        result[name] = number
    if not .01 <= result['angular_tolerance_deg'] <= 15:
        raise ValueError('angular_tolerance_deg must be in [.01,15]')
    if not 0 < result['island_weight_min'] <= result['island_weight_max'] <= 2:
        raise ValueError('Island weights require 0 < minimum <= maximum <= 2')
    if not 0 <= result['anchor_confidence'] <= 1:
        raise ValueError('anchor_confidence must be in [0,1]')
    for name in ('min_sources', 'min_prediction_sources'):
        number = result[name]
        if isinstance(number, bool) or int(number) != number or int(number) < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
        result[name] = int(number)
    weights = dict(result['provenance_weights'])
    if set(weights) != {'prediction', 'bridge', 'mixed'}:
        raise ValueError('provenance_weights must define prediction, bridge, and mixed')
    for name, weight in weights.items():
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError('Provenance weights must be finite and nonnegative')
        weights[name] = float(weight)
    result['provenance_weights'] = weights
    if result['decide'] is not None and not callable(result['decide']):
        raise ValueError('decide must be callable or None')
    return result
