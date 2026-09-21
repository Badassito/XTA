"""Bounded reconciliation over immutable, independently editable layer evidence.

This numerical core uses TYX arrays and has no inference, output, or Slicer
dependencies. Readers and writers are supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .reconciliation_policy import validate_policy


@dataclass
class EvidenceLayer:
    layer_id: str
    shape_tyx: tuple[int, int, int]
    metadata: Mapping[str, Any]
    read_slab: Callable[[int, int], np.ndarray]
    confidence_reader: Callable | None = None


def evidence_role(metadata):
    kind = str(metadata.get('mask_kind', '')).lower()
    if kind == 'bridge':
        return 'bridge'
    if kind == 'yolo':
        return 'prediction'
    if kind == 'union':
        return 'mixed'
    raise ValueError(f'Unsupported additive evidence kind: {kind!r}')


def additive_layers(layers):
    result, ids = [], set()
    for layer in layers:
        metadata = layer.metadata
        if metadata.get('layer_role', 'additive_component') != 'additive_component':
            continue
        if metadata.get('recomposition_op', 'union') != 'union':
            continue
        if metadata.get('source') == 'global':
            continue
        if layer.layer_id in ids:
            raise ValueError(f'Duplicate reconciliation layer identity: {layer.layer_id}')
        ids.add(layer.layer_id)
        evidence_role(metadata)
        result.append(layer)
    return tuple(sorted(result, key=lambda item: item.layer_id))


def _island_weights(layers, layer_groups, descriptors, policy, memory_mib, progress):
    from .reconciliation_components import component_statistics
    records, cohorts, members = {}, {}, {}
    for layer, group in zip(layers, layer_groups):
        members.setdefault((group, evidence_role(layer.metadata)), []).append(layer)
    group_records = {}
    for index, (key, entries) in enumerate(sorted(members.items())):
        group, role = key
        def read_union(z0, z1):
            union = np.zeros((z1 - z0, *entries[0].shape_tyx[1:]), dtype=bool)
            for entry in entries:
                union |= _binary_mask(entry, z0, z1, union.shape)
            return union
        # Reserve caller-owned input/union buffers in addition to the labeler's budget.
        stats = component_statistics(read_union, entries[0].shape_tyx, memory_mib=float(memory_mib) * .8)
        cohort = (str(descriptors[group]['kind']), role)
        if stats['foreground_voxels'] > 0:
            cohorts.setdefault(cohort, []).append(key)
        group_records[key] = {**stats, 'weight': 1., 'cohort': '/'.join(cohort),
                              'evidence_group': descriptors[group]['group_key'],
                              'statistics_scope': 'union_of_group_members_of_this_role'}
        if progress:
            progress('components', index + 1, len(members))
    for ids in cohorts.values():
        values = np.log1p([group_records[key]['largest_component_voxels'] for key in ids])
        median = float(np.median(values))
        scale = float(np.percentile(values, 75) - np.percentile(values, 25))
        for key, value in zip(ids, values):
            weight = 1. if scale <= 1e-12 else 1. + .25 * (float(value) - median) / scale
            group_records[key]['weight'] = float(np.clip(weight, policy['island_weight_min'], policy['island_weight_max']))
    for layer, group in zip(layers, layer_groups):
        records[layer.layer_id] = dict(group_records[(group, evidence_role(layer.metadata))])
    return {key: record['weight'] for key, record in records.items()}, records


def _view_descriptor(layer):
    metadata = layer.metadata
    name = str(metadata.get('physical_view_name') or metadata.get('view_name') or layer.layer_id)
    return {'group_key': 'view:' + name, 'kind': 'view'}


def _binary_mask(layer, z0, z1, shape):
    values = np.asarray(layer.read_slab(z0, z1))
    if values.shape != shape or values.dtype not in (np.dtype('uint8'), np.dtype('bool')):
        raise ValueError(f'Reconciliation readers must return uint8/bool binary slabs: {layer.layer_id}')
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f'Reconciliation masks must be binary 0/1: {layer.layer_id}')
    return values != 0


def voting_memory_plan(shape_tyx, group_count, memory_mib, *, union_only=False):
    """Plan the same bounded vote workspace before inference or array allocation."""
    shape = tuple(int(v) for v in shape_tyx)
    if len(shape) != 3 or min(shape) <= 0 or int(group_count) < 0:
        raise ValueError('Voting memory plan needs a positive TYX shape and nonnegative group count')
    if not math.isfinite(float(memory_mib)) or memory_mib <= 0:
        raise ValueError('Voting memory budget must be positive and finite')
    budget = int(float(memory_mib) * 1024**2)
    bytes_per_voxel = 8 if union_only else 6 * int(group_count) + 160
    plane_bytes = shape[1] * shape[2] * bytes_per_voxel
    if plane_bytes > budget:
        raise ValueError(f'Reconciliation needs at least {math.ceil(plane_bytes / 1024**2)} MiB for one XY slab with {group_count} groups')
    depth = min(shape[0], 16, max(1, budget // plane_bytes))
    return dict(slab_depth=depth, minimum_working_bytes=plane_bytes,
                planned_working_bytes=depth * plane_bytes, bytes_per_voxel=bytes_per_voxel)


def _union_only(layers, shape, policy, write_slab, memory_mib, progress):
    plan = voting_memory_plan(shape, int(bool(layers)), memory_mib, union_only=True)
    depth = plan['slab_depth']
    records = {layer.layer_id: dict(group=0, role=evidence_role(layer.metadata), island_weight=1.,
                                    foreground_voxels=0) for layer in layers}
    total = 0
    for z0 in range(0, shape[0], depth):
        z1 = min(shape[0], z0 + depth)
        candidate = np.zeros((z1 - z0, *shape[1:]), bool)
        for layer in layers:
            mask = _binary_mask(layer, z0, z1, candidate.shape)
            records[layer.layer_id]['foreground_voxels'] += int(np.count_nonzero(mask))
            candidate |= mask
        total += int(np.count_nonzero(candidate))
        write_slab(z0, z1, candidate.astype(np.uint8))
        if progress:
            progress('voting', z1, shape[0])
    return dict(schema='xta.reconciliation/1', policy={k: v for k, v in policy.items() if k != 'decide'},
        shape_tyx=list(shape), memory_mib=memory_mib, slab_depth=depth,
        planned_working_bytes=plan['planned_working_bytes'], layer_count=len(layers), group_count=int(bool(layers)),
        groups=['union'] if layers else [], layers=records, components={},
        counts=dict(candidate_voxels=total, retained_voxels=total, rejected_voxels=0,
                    duplicate_group_support_removed=0, confidence_known_voxels=0,
                    confidence_unknown_prediction_voxels=0, anchored_voxels=0),
        evidence_interpretation='exact union of additive candidates')


def reconcile(layers: Sequence[EvidenceLayer], *, shape_tyx, policy, write_slab,
              memory_mib=4096, geometry_context=None, progress=None) -> dict:
    """Write one candidate-subset result without mutating any input layer.

    Scores are positive support, not calibrated voxel probabilities. Absent or
    unobserved data never becomes an invented negative vote. File multiplicity
    within an evidence group contributes its maximum, not a sum.
    """
    policy = validate_policy(policy)
    shape = tuple(int(v) for v in shape_tyx)
    if len(shape) != 3 or min(shape) <= 0 or not math.isfinite(float(memory_mib)) or memory_mib <= 0:
        raise ValueError('Reconciliation requires a positive TYX shape and memory budget')
    layers = additive_layers(layers)
    if any(tuple(layer.shape_tyx) != shape for layer in layers):
        raise ValueError('Every reconciliation layer must use the same reference grid')
    if len(layers) > 65535:
        raise ValueError('Too many reconciliation layers for exact support counters')
    if policy['mode'] == 'union' and policy['decide'] is None:
        return _union_only(layers, shape, policy, write_slab, memory_mib, progress)
    if policy['mode'] == 'confidence':
        missing = [layer.layer_id for layer in layers
                   if evidence_role(layer.metadata) == 'prediction' and layer.confidence_reader is None
                   and not layer.metadata.get('empty_segment', False)]
        if missing:
            raise ValueError(f'Confidence evidence is unavailable for {len(missing)} prediction layers; first: {missing[0]}')
    if policy['grouping'] == 'sections':
        from .reconciliation_geometry import section_descriptor, section_codes
        descriptors = [section_descriptor(layer.metadata, geometry_context=geometry_context) for layer in layers]
    else:
        descriptors = [_view_descriptor(layer) for layer in layers]
        section_codes = None
    groups, group_descriptors = {}, []
    layer_groups = []
    for descriptor in descriptors:
        key = descriptor['group_key']
        if key not in groups:
            groups[key] = len(groups)
            group_descriptors.append(descriptor)
        layer_groups.append(groups[key])
    group_count = len(groups)
    # Six bytes per group hold score, direct-source support, and anchor support;
    # reserve additional temporaries, dynamic orientation codes and input planes.
    plan = voting_memory_plan(shape, group_count, memory_mib)
    slab_depth = plan['slab_depth']
    weights, component_records = ({layer.layer_id: 1. for layer in layers}, {})
    if policy['island_weighting']:
        weights, component_records = _island_weights(layers, layer_groups, group_descriptors,
                                                      policy, memory_mib, progress)
    counts = dict(candidate_voxels=0, retained_voxels=0, rejected_voxels=0,
                  duplicate_group_support_removed=0, confidence_known_voxels=0,
                  confidence_unknown_prediction_voxels=0, anchored_voxels=0)
    source_records = {layer.layer_id: dict(group=groups[descriptors[i]['group_key']],
        role=evidence_role(layer.metadata), island_weight=weights[layer.layer_id], foreground_voxels=0)
        for i, layer in enumerate(layers)}
    for z0 in range(0, shape[0], slab_depth):
        z1 = min(shape[0], z0 + slab_depth)
        block_shape = (z1 - z0, *shape[1:])
        scores = np.zeros((group_count, *block_shape), np.float32)
        direct = np.zeros((group_count, *block_shape), bool)
        anchors = np.zeros((group_count, *block_shape), bool)
        candidate = np.zeros(block_shape, bool)
        for layer, group in zip(layers, layer_groups):
            mask = _binary_mask(layer, z0, z1, block_shape)
            candidate |= mask
            source_records[layer.layer_id]['foreground_voxels'] += int(np.count_nonzero(mask))
            role = evidence_role(layer.metadata)
            value = mask.astype(np.float32)
            known = mask
            if policy['mode'] == 'confidence' and role == 'prediction':
                if layer.confidence_reader is None:
                    if np.any(mask):
                        raise ValueError(f'Nonempty prediction layer has no confidence evidence: {layer.layer_id}')
                    continue
                raw, observed = layer.confidence_reader(z0, z1)
                raw, observed = np.asarray(raw), np.asarray(observed)
                if raw.shape != block_shape or observed.shape != block_shape or raw.dtype != np.uint8 or observed.dtype != np.bool_:
                    raise ValueError(f'Confidence reader must return uint8 scores and boolean known support: {layer.layer_id}')
                known = mask & observed & (raw > 0)
                value = np.where(known, raw.astype(np.float32) / 255., 0.)
                counts['confidence_known_voxels'] += int(np.count_nonzero(known))
                counts['confidence_unknown_prediction_voxels'] += int(np.count_nonzero(mask & ~known))
                anchors[group] |= known & (value >= policy['anchor_confidence'])
            contribution = value * float(policy['provenance_weights'][role] * weights[layer.layer_id])
            np.maximum(scores[group], contribution, out=scores[group])
            if role == 'prediction':
                direct[group] |= known & (contribution > 0)
        if policy['grouping'] == 'sections' and policy['mode'] != 'union':
            codes = [section_codes(descriptor, z0, z1, shape, geometry_context,
                                   angular_tolerance_deg=policy['angular_tolerance_deg'])
                     for descriptor in group_descriptors]
            for current in range(group_count):
                for previous in range(current):
                    if np.isscalar(codes[current]) and np.isscalar(codes[previous]):
                        if codes[current] != codes[previous]:
                            continue
                    same = np.equal(codes[current], codes[previous]) & (scores[current] > 0)
                    if not np.any(same):
                        continue
                    counts['duplicate_group_support_removed'] += int(np.count_nonzero(same))
                    np.maximum(scores[previous], np.where(same, scores[current], 0.), out=scores[previous])
                    direct[previous] |= same & direct[current]
                    anchors[previous] |= same & anchors[current]
                    scores[current][same] = 0.
                    direct[current][same] = False
                    anchors[current][same] = False
        support = np.count_nonzero(scores > 0, axis=0).astype(np.uint16)
        direct_support = np.count_nonzero(direct, axis=0).astype(np.uint16)
        total = scores.sum(axis=0, dtype=np.float32)
        anchored = anchors.any(axis=0) & (direct_support >= 2)
        effective = total + np.where(anchored, policy['anchor_bonus'], 0.)
        counts['anchored_voxels'] += int(np.count_nonzero(anchored))
        block = dict(candidate=candidate, score=effective, support=support,
                     prediction_support=direct_support, anchored=anchored, z0=z0, z1=z1)
        if policy['decide'] is not None:
            for value in (candidate, effective, support, direct_support, anchored):
                value.flags.writeable = False
            keep = np.asarray(policy['decide'](block))
            if keep.dtype != np.bool_ or keep.shape != block_shape:
                raise ValueError('Custom reconciliation decide() must return a boolean array with the block shape')
        elif policy['mode'] == 'union':
            keep = candidate
        else:
            keep = (candidate & (effective >= policy['threshold']) & (support >= policy['min_sources'])
                    & (direct_support >= policy['min_prediction_sources']))
        if np.any(keep & ~candidate):
            raise ValueError('Reconciliation cannot introduce foreground outside candidate evidence')
        retained = int(np.count_nonzero(keep))
        candidates = int(np.count_nonzero(candidate))
        counts['retained_voxels'] += retained
        counts['candidate_voxels'] += candidates
        counts['rejected_voxels'] += candidates - retained
        write_slab(z0, z1, keep.astype(np.uint8))
        if progress:
            progress('voting', z1, shape[0])
    return dict(schema='xta.reconciliation/1', policy={k: v for k, v in policy.items() if k != 'decide'},
        shape_tyx=list(shape), memory_mib=memory_mib, slab_depth=slab_depth,
        planned_working_bytes=plan['planned_working_bytes'], layer_count=len(layers), group_count=group_count,
        groups=list(groups), counts=counts, layers=source_records, components=component_records,
        evidence_interpretation='positive support; geometry groups cap correlated observations; scores are not calibrated voxel probabilities')
