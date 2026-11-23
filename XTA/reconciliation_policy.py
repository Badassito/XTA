"""Dependency-light external reconciliation policy discovery and settings."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SWEEP_RECONCILIATION_POLICY_STAGE = 'XTA_YOLO_SWEEP_RECONCILIATION_POLICY_STAGE'


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
    validated = validate_policy(result)
    validated['proposal_policy_sha256'] = settings.sha256
    return validated


def validate_policy(value) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError('build_reconciliation() must return a policy dictionary')
    defaults = dict(name='weighted', mode='weighted', grouping='sections', threshold=1.5,
                    min_sources=2, min_prediction_sources=1, island_weighting=False,
                    island_weight_min=.75, island_weight_max=1.25,
                    angular_tolerance_deg=1., anchor_confidence=.8, anchor_bonus=0.,
                    provenance_weights={'prediction': 1., 'bridge': .35, 'mixed': .5,
                                        'extrapolation': .35},
                    decide=None, select_proposals=None, proposal_api_version=1,
                    sam_bridge_policy=None, proposal_policy_sha256='')
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
    if set(weights) == {'prediction', 'bridge', 'mixed'}:
        # Historical policies inherit their explicit interpolation-addition
        # weight; a policy may separately opt out with extrapolation=0.
        weights['extrapolation'] = weights['bridge']
    if set(weights) != {'prediction', 'bridge', 'mixed', 'extrapolation'}:
        raise ValueError('provenance_weights must define prediction, bridge, mixed, and extrapolation')
    for name, weight in weights.items():
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError('Provenance weights must be finite and nonnegative')
        weights[name] = float(weight)
    result['provenance_weights'] = weights
    if result['decide'] is not None and not callable(result['decide']):
        raise ValueError('decide must be callable or None')
    if result['select_proposals'] is not None and not callable(result['select_proposals']):
        raise ValueError('select_proposals must be callable or None')
    if result['proposal_api_version'] != 1:
        raise ValueError('proposal_api_version must be 1')
    if result['sam_bridge_policy'] is not None:
        from .sam_policy import resolve_sam_bridge_policy
        resolve_sam_bridge_policy(result)
    if not isinstance(result['proposal_policy_sha256'], str):
        raise ValueError('proposal_policy_sha256 must be a string')
    return result


def _policy_record_sha256(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(',', ':')).encode('ascii')).hexdigest()


def stage_sweep_reconciliation_report(report, directory):
    """Stage the resolved policy by content and return the cutoff-specific report."""
    from .json_publication import write_json_atomic

    result = dict(report)
    record = {key: result.pop(key) for key in ('policy', 'policy_path', 'policy_sha256')}
    digest = _policy_record_sha256(record)
    path = Path(directory) / f'{digest}.json'
    if not path.is_file():
        write_json_atomic(path, record, sort_keys=True)
    result['policy_record_sha256'] = digest
    return result


def _sweep_reconciliation_policy_digest(receipt):
    report = receipt.get('reconciliation')
    source_digest = (receipt.get('confidence_sweep') or {}).get('reconciliation_policy_sha256')
    if report is None and source_digest is None:
        return None
    digest = report.get('policy_record_sha256') if isinstance(report, dict) else None
    if (not isinstance(digest, str) or re.fullmatch('[a-f0-9]{64}', digest) is None
            or not isinstance(source_digest, str) or re.fullmatch('[a-f0-9]{64}', source_digest) is None
            or report.get('schema') != 'xta.reconciliation/1'
            or {'policy', 'policy_path', 'policy_sha256'}.intersection(report)):
        raise ValueError('Cutoff receipt has an invalid reconciliation report or policy reference')
    return digest


def _verified_sweep_reconciliation_policy(receipt, digest, record):
    source_digest = receipt['confidence_sweep']['reconciliation_policy_sha256']
    if (not isinstance(record, dict) or set(record) != {'policy', 'policy_path', 'policy_sha256'}
            or not isinstance(record['policy'], dict) or not isinstance(record['policy_path'], str)
            or record['policy_sha256'] != source_digest or _policy_record_sha256(record) != digest):
        raise ValueError('Resolved reconciliation policy differs from the cutoff receipt')
    return record


def accept_sweep_reconciliation_policy(receipt, policies, directory):
    """Read a privately staged policy on its first accepted cutoff."""
    digest = _sweep_reconciliation_policy_digest(receipt)
    if digest is None:
        return None
    if digest in policies:
        record = policies[digest]
    else:
        path = Path(directory) / f'{digest}.json'
        if path.is_symlink() or not path.is_file():
            raise ValueError('Cutoff receipt has no privately staged reconciliation policy')
        record = json.loads(path.read_bytes())
    return digest, _verified_sweep_reconciliation_policy(receipt, digest, record)


def load_reconciliation_report(run_manifest: str | Path) -> dict[str, Any] | None:
    """Read a standalone, legacy sweep, or accepted embedded cutoff report."""
    path = Path(run_manifest).resolve()
    if path.is_dir():
        path /= 'manifest.json'
    data = path.read_bytes() if path.is_file() else b'{}'
    receipt = json.loads(data)
    if not isinstance(receipt, dict):
        raise ValueError('Expected a JSON run manifest')
    if receipt.get('schema') == 'xta.reconciliation/1':
        return receipt
    if 'reconciliation' not in receipt:
        legacy = path.parent / 'reconciliation' / 'manifest.json'
        return json.loads(legacy.read_bytes()) if legacy.is_file() else None
    digest = _sweep_reconciliation_policy_digest(receipt)
    if digest is None:
        return None
    ownership = receipt['confidence_sweep']
    cutoff = ownership.get('cutoff_hundredths')
    if (type(cutoff) is not int or not 0 <= cutoff <= 100 or path.name != 'manifest.json'
            or path.parent.name != f'conf_{cutoff:03d}' or path.parent.parent.name != 'confidence_cutoffs'):
        raise ValueError('Reconciliation cutoff receipt is outside its sweep layout')
    family = json.loads((path.parents[2] / 'manifest.json').read_bytes())
    if not isinstance(family, dict) or family.get('schema') != 'xta.confidence_sweep/1':
        raise ValueError('Expected a JSON confidence sweep family manifest')
    relative = f'confidence_cutoffs/conf_{cutoff:03d}/manifest.json'
    status = receipt.get('status')
    matching = [branch for branch in family.get('branches', ()) if isinstance(branch, dict)
                and type(branch.get('cutoff_hundredths')) is int and branch['cutoff_hundredths'] == cutoff]
    if (not ownership.get('sweep_id')
            or family.get('sweep_id') != ownership['sweep_id'] or status not in ('complete', 'incomplete')
            or len(matching) != 1 or matching[0].get('status') != status
            or matching[0].get('manifest') != relative
            or matching[0].get('manifest_sha256') != hashlib.sha256(data).hexdigest()):
        raise ValueError('Reconciliation cutoff receipt differs from its family manifest')
    policies = family.get('reconciliation_policies')
    record = _verified_sweep_reconciliation_policy(
        receipt, digest, policies.get(digest) if isinstance(policies, dict) else None)
    return {**receipt['reconciliation'], **record}
