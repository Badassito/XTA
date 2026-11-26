"""Bounded reconciliation over immutable, independently editable layer evidence.

This numerical core uses TYX arrays and has no inference, output, or Slicer
dependencies. Readers and writers are supplied by the caller.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .reconciliation_policy import validate_policy

_VOTE_TILE = 128


@dataclass
class EvidenceLayer:
    layer_id: str
    shape_tyx: tuple[int, int, int]
    metadata: Mapping[str, Any]
    read_slab: Callable[[int, int], np.ndarray]
    confidence_reader: Callable | None = None
    read_mask_crop: Callable | None = None
    mask_crop_bounds: tuple[int, int, int, int, int, int] | None = None


def evidence_role(metadata):
    kind = str(metadata.get('mask_kind', '')).lower()
    if kind == 'bridge':
        return 'bridge'
    if kind == 'extrapolation':
        return 'extrapolation'
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


def _component_group_bounds(entries):
    bounds = [entry.mask_crop_bounds for entry in entries]
    if any(bound is None or entry.read_mask_crop is None for entry, bound in zip(entries, bounds)):
        return None
    active = [bound for bound in bounds if bound[0] < bound[1]]
    if not active:
        return (0, 1, 0, 1, 0, 1)
    return tuple(min(bound[i] for bound in active) if i % 2 == 0
                 else max(bound[i] for bound in active) for i in range(6))


def _component_group_stats(entries, bounds, memory_mib):
    from .reconciliation_components import component_statistics
    shape = entries[0].shape_tyx
    if bounds is None:
        def read_union(z0, z1):
            union = np.zeros((z1 - z0, *shape[1:]), dtype=bool)
            for entry in entries:
                union |= _binary_mask(entry, z0, z1, union.shape)
            return union

        return component_statistics(read_union, shape, memory_mib=memory_mib)

    start, stop, y0, y1, x0, x1 = bounds
    cropped_shape = (stop-start, y1-y0, x1-x0)

    def read_cropped_union(a, b):
        first_z, stop_z = start+a, start+b
        union = np.zeros((b-a, *cropped_shape[1:]), dtype=bool)
        for entry in entries:
            first, last = entry.mask_crop_bounds[:2]
            for z in range(max(first, first_z), min(last, stop_z)):
                crop = entry.read_mask_crop(z)
                if crop is None:
                    continue
                top, left, bottom, right, values = crop
                if not (y0 <= top < bottom <= y1 and x0 <= left < right <= x1):
                    raise ValueError(f'Reconciliation mask crop exceeds planned bounds: {entry.layer_id}')
                binary = _checked_binary(values, (bottom-top, right-left), entry.layer_id)
                union[z-first_z, top-y0:bottom-y0, left-x0:right-x0] |= binary
        return union

    stats = component_statistics(read_cropped_union, cropped_shape, memory_mib=memory_mib)
    return {**stats, 'shape_tyx': shape, 'component_working_bounds_tyx': bounds}


def _island_weights(layers, layer_groups, descriptors, policy, memory_mib, progress, *, workers=1):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .reconciliation_components import component_lane_count
    records, cohorts, members = {}, {}, {}
    for layer, group in zip(layers, layer_groups):
        members.setdefault((group, evidence_role(layer.metadata)), []).append(layer)
    plans = [(key, entries, _component_group_bounds(entries)) for key, entries in sorted(members.items())]
    shapes = [entries[0].shape_tyx if bounds is None else
              tuple(bounds[i+1]-bounds[i] for i in (0, 2, 4)) for _, entries, bounds in plans]
    # A managed source owns independent immutable readers for each group.
    # Generic callbacks keep their caller-thread reading order.
    budget = float(memory_mib) * .8
    lanes = component_lane_count(shapes, budget, workers) if plans and all(row[2] is not None for row in plans) else 1

    def measure(plan):
        key, entries, bounds = plan
        return key, _component_group_stats(entries, bounds, budget / lanes)

    def collect(results):
        measured = {}
        for key, stats in results:
            measured[key] = stats
            if progress:
                progress('components', len(measured), len(plans))
        return measured

    if lanes == 1:
        measured = collect(map(measure, plans))
    else:
        # Start the larger unions before dispatching smaller ones.
        order = sorted(range(len(plans)), key=lambda index: math.prod(shapes[index]), reverse=True)
        with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix='island-components') as pool:
            pending = [pool.submit(measure, plans[index]) for index in order]
            measured = collect(future.result() for future in as_completed(pending))

    group_records = {}
    for key, stats in sorted(measured.items()):
        group, role = key
        cohort = (str(descriptors[group]['kind']), role)
        if stats['foreground_voxels'] > 0:
            cohorts.setdefault(cohort, []).append(key)
        group_records[key] = {**stats, 'weight': 1., 'cohort': '/'.join(cohort),
                              'evidence_group': descriptors[group]['group_key'],
                              'statistics_scope': 'union_of_group_members_of_this_role'}
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


def _checked_binary(values, shape, layer_id):
    values = np.asarray(values)
    if values.shape != shape or values.dtype not in (np.dtype('uint8'), np.dtype('bool')):
        raise ValueError(f'Reconciliation readers must return uint8/bool binary slabs: {layer_id}')
    if values.dtype == np.uint8 and values.max(initial=0) > 1:
        raise ValueError(f'Reconciliation masks must be binary 0/1: {layer_id}')
    return values != 0


def _binary_mask(layer, z0, z1, shape):
    return _checked_binary(layer.read_slab(z0, z1), shape, layer.layer_id)


def _mark_vote_tiles(tiles, z, y0, y1, x0, x1):
    tiles[z, y0//_VOTE_TILE:(y1-1)//_VOTE_TILE+1, x0//_VOTE_TILE:(x1-1)//_VOTE_TILE+1] = True


def _mark_mask_tiles(tiles, selection, mask, count):
    z, y, x = selection
    region = tiles[z, y.start//_VOTE_TILE:(y.stop-1)//_VOTE_TILE+1,
                   x.start//_VOTE_TILE:(x.stop-1)//_VOTE_TILE+1]
    if region.size == 1 or count == mask.size:
        region.fill(True)
    elif not region.all():
        # Crops start at arbitrary offsets in the full-grid tile lattice.
        row_starts = [0, *range(_VOTE_TILE - y.start % _VOTE_TILE, mask.shape[0], _VOTE_TILE)]
        col_starts = [0, *range(_VOTE_TILE - x.start % _VOTE_TILE, mask.shape[1], _VOTE_TILE)]
        occupied_cols = np.logical_or.reduceat(mask, col_starts, axis=1)
        region |= np.logical_or.reduceat(occupied_cols, row_starts, axis=0)


def _evidence_regions(layer, z0, z1, block_shape, needs_confidence, *, allow_sparse, score_tiles=None):
    """Read only the recorded mask crops for a source-aligned packed layer."""
    reader = layer.confidence_reader
    iter_scores = getattr(reader, 'iter_crops', None)
    if (not allow_sparse or layer.read_mask_crop is None
            or (needs_confidence and reader is not None and not callable(iter_scores))):
        mask = _binary_mask(layer, z0, z1, block_shape)
        raw = observed = None
        if needs_confidence and reader is not None:
            raw, observed = map(np.asarray, reader(z0, z1))
            if (raw.shape != block_shape or observed.shape != block_shape
                    or raw.dtype != np.uint8 or observed.dtype != np.bool_):
                raise ValueError(f'Confidence reader must return uint8 scores and boolean known support: {layer.layer_id}')
        yield (slice(None), slice(None), slice(None)), mask, raw, observed
        return
    for local_z, z in enumerate(range(z0, z1)):
        record = layer.read_mask_crop(z)
        if record is None:
            continue
        y0, x0, y1, x1, values = record
        if not (0 <= y0 < y1 <= block_shape[1] and 0 <= x0 < x1 <= block_shape[2]):
            raise ValueError(f'Reconciliation mask crop is outside the source grid: {layer.layer_id}')
        mask = _checked_binary(values, (y1-y0, x1-x0), layer.layer_id)
        raw = observed = None
        if needs_confidence and reader is not None:
            raw = np.zeros(mask.shape, np.uint8)
            for sy0, sy1, sx0, sx1, scores in iter_scores(z):
                top, bottom = max(sy0, y0), min(sy1, y1)
                left, right = max(sx0, x0), min(sx1, x1)
                if top < bottom and left < right:
                    target = raw[top-y0:bottom-y0, left-x0:right-x0]
                    np.maximum(target, scores[top-sy0:bottom-sy0, left-sx0:right-sx0], out=target)
                    if score_tiles is not None:
                        _mark_vote_tiles(score_tiles, local_z, top, bottom, left, right)
            observed = raw > 0
        yield (local_z, slice(y0, y1), slice(x0, x1)), mask, raw, observed


def _collision_pairs(descriptors, cached_codes):
    pairs = []
    for current, descriptor in enumerate(descriptors):
        for previous, earlier in enumerate(descriptors[:current]):
            kinds = {descriptor['kind'], earlier['kind']}
            # Azimuthal sections share planar code space. Sphere, cylinder and
            # unknown code ranges are disjoint from the plane codes.
            if 'azimuthal' in kinds and not kinds <= {'plane', 'azimuthal'}:
                continue
            current_code, previous_code = cached_codes.get(current), cached_codes.get(previous)
            if np.isscalar(current_code) and np.isscalar(previous_code) and current_code != previous_code:
                continue
            pairs.append((current, previous))
    return pairs


def _merge_section_votes(scores, direct, extrapolated, codes, pairs, group_bounds):
    removed = 0
    codes = [code if np.isscalar(code) else np.broadcast_to(code, scores.shape[1:]) for code in codes]
    for current, previous in pairs:
        left, right = group_bounds[current], group_bounds[previous]
        if left is None or right is None:
            continue
        section = tuple(slice(max(left[axis], right[axis]), min(left[axis+1], right[axis+1]))
                        for axis in (0, 2, 4))
        if any(region.start >= region.stop for region in section):
            continue
        first_code = codes[current] if np.isscalar(codes[current]) else codes[current][section]
        second_code = codes[previous] if np.isscalar(codes[previous]) else codes[previous][section]
        incoming, retained = scores[current][section], scores[previous][section]
        same = np.equal(first_code, second_code) & (incoming > 0) & (retained > 0)
        duplicates = int(np.count_nonzero(same))
        if not duplicates:
            continue
        removed += duplicates
        np.maximum(retained, incoming, out=retained, where=same)
        incoming[same] = 0.
        direct[previous][section] |= same & direct[current][section]
        direct[current][section][same] = False
        if extrapolated is not None:
            extrapolated[previous][section] |= same & extrapolated[current][section]
            extrapolated[current][section][same] = False
    return removed


def _collect_layer_votes(entries, z0, z1, candidate, anchored, scores, direct, extrapolated,
                         confidence_mode, anchor_confidence, compare_sections, vote_tiles=None):
    """Collect one lane's evidence with exclusive ownership of its score groups."""
    group_bounds, foreground = {}, {}
    known_total = unknown_total = 0
    max_finite_score = np.finfo(np.float32).max
    for layer, group, role, weight in entries:
        needs_confidence = confidence_mode and role == 'prediction'
        allow_sparse = weight <= max_finite_score
        tiles = None if vote_tiles is None or weight == 0 else vote_tiles[group]
        score_tiles = tiles if needs_confidence else None
        for selection, mask, raw, observed in _evidence_regions(
                layer, z0, z1, candidate.shape, needs_confidence,
                allow_sparse=allow_sparse, score_tiles=score_tiles):
            foreground_count = int(np.count_nonzero(mask))
            if not foreground_count and allow_sparse:
                continue
            candidate[selection] |= mask
            foreground[layer.layer_id] = foreground.get(layer.layer_id, 0) + foreground_count
            known = mask
            if needs_confidence:
                if raw is None:
                    if foreground_count:
                        raise ValueError(f'Nonempty prediction layer has no confidence evidence: {layer.layer_id}')
                    continue
                known = mask & observed & (raw > 0)
                value = np.where(known, raw.astype(np.float32) / 255., np.float32(0.))
                known_count = int(np.count_nonzero(known))
                known_total += known_count
                unknown_total += foreground_count - known_count
                anchored[selection] |= known & (value >= anchor_confidence)
            else:
                value = mask.astype(np.float32)
            contribution = value * weight
            target = scores[group][selection]
            np.maximum(target, contribution, out=target)
            if role == 'prediction':
                direct[group][selection] |= known & (contribution > 0)
            elif role == 'extrapolation' and extrapolated is not None:
                extrapolated[group][selection] |= mask & (contribution > 0)
            if tiles is not None and not needs_confidence:
                _mark_mask_tiles(tiles, selection, mask, foreground_count)
            if compare_sections:
                z, y, x = selection
                bounds = ((z, z+1, y.start, y.stop, x.start, x.stop) if isinstance(z, int)
                          else (0, candidate.shape[0], 0, candidate.shape[1], 0, candidate.shape[2]))
                previous = group_bounds.get(group)
                group_bounds[group] = bounds if previous is None else (
                    min(previous[0], bounds[0]), max(previous[1], bounds[1]),
                    min(previous[2], bounds[2]), max(previous[3], bounds[3]),
                    min(previous[4], bounds[4]), max(previous[5], bounds[5]),
                )
    return candidate, anchored, group_bounds, foreground, known_total, unknown_total


def _voting_tile_grid(entries, shape, group_count, confidence_mode):
    """Bound support with score-block or mask-crop coordinates already being read."""
    if group_count < 2 or min(shape[1:]) < _VOTE_TILE:
        return None
    finite_weight_limit = float(np.finfo(np.float32).max) / (1 + group_count)
    for layer, _, role, weight in entries:
        reader = layer.confidence_reader
        if (layer.read_mask_crop is None or weight > finite_weight_limit
                or math.copysign(1.0, weight) < 0
                or (confidence_mode and role == 'prediction' and reader is not None
                    and not callable(getattr(reader, 'iter_crops', None)))):
            return None
    return tuple(math.ceil(axis / _VOTE_TILE) for axis in shape[1:])


def _reduce_vote_groups(scores, direct, extrapolated, tiles):
    """Reduce group votes over covered regions, retaining full callback arrays."""
    if tiles is None or np.count_nonzero(tiles) * 2 >= tiles.size:
        return ((scores > 0).sum(axis=0, dtype=np.uint16),
                direct.sum(axis=0, dtype=np.uint16),
                scores.sum(axis=0, dtype=np.float32),
                None if extrapolated is None else extrapolated.sum(axis=0, dtype=np.uint16))
    shape = scores.shape[1:]
    support = np.zeros(shape, dtype=np.uint16)
    direct_support = np.zeros(shape, dtype=np.uint16)
    total = np.zeros(shape, dtype=np.float32)
    extrapolation_support = None if extrapolated is None else np.zeros(shape, dtype=np.uint16)
    # Each row has an even number of changes when both endpoints are treated
    # as empty. Pair runs in C order so a voxel sees the original group order.
    changes = np.diff(tiles, axis=-1, prepend=False, append=False)
    groups, planes, rows, columns = np.nonzero(changes)
    for group, z, y, start, stop in zip(groups[::2], planes[::2], rows[::2],
                                       columns[::2], columns[1::2], strict=True):
        region = (z, slice(y*_VOTE_TILE, (y+1)*_VOTE_TILE),
                  slice(start*_VOTE_TILE, stop*_VOTE_TILE))
        selected = (group, *region)
        values = scores[selected]
        np.add(support[region], values > 0, out=support[region])
        np.add(direct_support[region], direct[selected], out=direct_support[region])
        # The original C-order group stack reduces a strided first axis.
        # Do not reduce a one-voxel crop with NumPy's pairwise summation.
        np.add(total[region], values, out=total[region])
        if extrapolated is not None:
            np.add(extrapolation_support[region], extrapolated[selected], out=extrapolation_support[region])
    return support, direct_support, total, extrapolation_support


def _voting_lanes(entries, parallel_layer_costs, shape, plan, memory_mib, workers):
    """Admit concurrent groups without shrinking the callback's original slabs."""
    if workers <= 1 or not parallel_layer_costs:
        return entries, (), 0
    by_group = {}
    for entry in entries:
        by_group.setdefault(entry[1], []).append(entry)
    costs = {
        group: sum(parallel_layer_costs[member[0].layer_id] for member in members)
        for group, members in by_group.items()
        if all(member[0].layer_id in parallel_layer_costs and member[3] <= np.finfo(np.float32).max
               for member in members)
    }
    plane = math.prod(shape[1:])
    # Every lane owns two boolean slabs and one set of decoded crop, score and
    # numeric temporaries. The decoder also batches bounded compressed blocks.
    lane_bytes = 2 * plan['slab_depth'] * plane + 32 * plane + 20 * 1024**2
    spare_bytes = int(float(memory_mib) * 1024**2) - plan['planned_working_bytes']
    lanes = min(16, int(workers), len(costs), spare_bytes // lane_bytes)
    if lanes < 2:
        return entries, (), 0

    pending = [([], 0) for _ in range(lanes)]
    # Keep one owner per group, assigning the largest indexed crop workloads
    # first. A section with many small views can cost less than one wide layer.
    for group in sorted(costs, key=lambda group: (-costs[group], group)):
        index = min(range(lanes), key=lambda lane: pending[lane][1])
        members, load = pending[index]
        members.extend(by_group[group])
        pending[index] = members, load + costs[group]
    serial = [entry for entry in entries if entry[1] not in costs]
    return serial, tuple(members for members, _ in pending), lane_bytes


def voting_memory_plan(shape_tyx, group_count, memory_mib, *, union_only=False,
                       include_extrapolation=False):
    """Plan the same bounded vote workspace before inference or array allocation."""
    shape = tuple(int(v) for v in shape_tyx)
    if len(shape) != 3 or min(shape) <= 0 or int(group_count) < 0:
        raise ValueError('Voting memory plan needs a positive TYX shape and nonnegative group count')
    if not math.isfinite(float(memory_mib)) or memory_mib <= 0:
        raise ValueError('Voting memory budget must be positive and finite')
    budget = int(float(memory_mib) * 1024**2)
    bytes_per_voxel = 8 if union_only else (6 + int(include_extrapolation)) * int(group_count) + 160
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
    return dict(schema='xta.reconciliation/1', policy={k: v for k, v in policy.items() if k not in {'decide', 'select_proposals'}},
        shape_tyx=list(shape), memory_mib=memory_mib, slab_depth=depth,
        planned_working_bytes=plan['planned_working_bytes'], layer_count=len(layers), group_count=int(bool(layers)),
        groups=['union'] if layers else [], layers=records, components={},
        counts=dict(candidate_voxels=total, retained_voxels=total, rejected_voxels=0,
                    duplicate_group_support_removed=0, confidence_known_voxels=0,
                    confidence_unknown_prediction_voxels=0, anchored_voxels=0),
        evidence_interpretation='exact union of additive candidates')


def reconcile(layers: Sequence[EvidenceLayer], *, shape_tyx, policy, write_slab,
              memory_mib=4096, geometry_context=None, progress=None,
              managed_workers=1, parallel_layer_costs=None, phase_span=None) -> dict:
    """Write one candidate-subset result without mutating any input layer.

    Scores are positive support, not calibrated voxel probabilities. Absent or
    unobserved data never becomes an invented negative vote. File multiplicity
    within an evidence group contributes its maximum, not a sum.

    Managed callers may provide layer costs from independent immutable readers.
    A zero cost certifies that the entire mask's slice index is empty.
    The optional phase span measures coordinator wall time outside the report.
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
    # Preserve the slab boundaries supplied to custom decision callbacks.
    has_extrapolation = any(evidence_role(layer.metadata) == 'extrapolation' for layer in layers)
    plan = voting_memory_plan(shape, group_count, memory_mib, include_extrapolation=has_extrapolation)
    slab_depth = plan['slab_depth']
    weights, component_records = ({layer.layer_id: 1. for layer in layers}, {})
    if policy['island_weighting']:
        weights, component_records = _island_weights(layers, layer_groups, group_descriptors,
            policy, memory_mib, progress, workers=managed_workers)
    counts = dict(candidate_voxels=0, retained_voxels=0, rejected_voxels=0,
                  duplicate_group_support_removed=0, confidence_known_voxels=0,
                  confidence_unknown_prediction_voxels=0, anchored_voxels=0)
    source_records = {layer.layer_id: dict(group=groups[descriptors[i]['group_key']],
        role=evidence_role(layer.metadata), island_weight=weights[layer.layer_id], foreground_voxels=0)
        for i, layer in enumerate(layers)}
    compare_sections = section_codes is not None and policy['mode'] != 'union'
    angular_tolerance = policy['angular_tolerance_deg']
    measure = phase_span if phase_span is not None else lambda name: nullcontext()
    cached_codes = {}
    if compare_sections:
        # Section codes with no Z coordinate broadcast from one plane to each slab.
        with measure('tta.reconciliation.voting.section_collision'):
            cached_codes = {
                i: section_codes(descriptor, 0, 1, shape, geometry_context,
                                 angular_tolerance_deg=angular_tolerance)
                for i, descriptor in enumerate(group_descriptors)
                if descriptor['kind'] != 'azimuthal' or 2 not in descriptor['axes_xyz'][:2]
            }
    pairs = _collision_pairs(group_descriptors, cached_codes) if compare_sections else ()
    decide = policy['decide']
    anchor_bonus = policy['anchor_bonus']
    threshold = policy['threshold']
    entries = []
    parallel_layer_costs = parallel_layer_costs or {}
    for layer, group in zip(layers, layer_groups, strict=True):
        role = evidence_role(layer.metadata)
        weight = float(policy['provenance_weights'][role] * weights[layer.layer_id])
        if parallel_layer_costs.get(layer.layer_id) == 0 and weight <= np.finfo(np.float32).max:
            continue
        entries.append((layer, group, role, weight))
    serial_entries, lanes, lane_bytes = _voting_lanes(
        entries, parallel_layer_costs, shape, plan, memory_mib, managed_workers)
    confidence_mode = policy['mode'] == 'confidence'
    anchor_confidence = policy['anchor_confidence']
    tile_grid = _voting_tile_grid(entries, shape, group_count, confidence_mode)
    score_plane = np.empty(shape[1:], np.float64) if anchor_bonus and decide is None else None
    pool = (ThreadPoolExecutor(max_workers=len(lanes), thread_name_prefix='reconciliation-vote')
            if lanes else nullcontext())
    with pool as executor:
        for z0 in range(0, shape[0], slab_depth):
            z1 = min(shape[0], z0 + slab_depth)
            block_shape = (z1 - z0, *shape[1:])
            scores = np.zeros((group_count, *block_shape), np.float32)
            direct = np.zeros((group_count, *block_shape), bool)
            anchored = np.zeros(block_shape, bool)
            extrapolated = np.zeros((group_count, *block_shape), bool) if has_extrapolation and decide is not None else None
            candidate = np.zeros(block_shape, bool)
            group_bounds = [None] * group_count
            vote_tiles = (None if tile_grid is None else
                          np.zeros((group_count, block_shape[0], *tile_grid), bool))

            def collect(batch, local_candidate, local_anchored):
                return _collect_layer_votes(batch, z0, z1, local_candidate, local_anchored,
                    scores, direct, extrapolated, confidence_mode, anchor_confidence, bool(pairs), vote_tiles)

            def merge(result):
                local_candidate, local_anchored, bounds, foreground, known, unknown = result
                if local_candidate is not candidate:
                    np.logical_or(candidate, local_candidate, out=candidate)
                    np.logical_or(anchored, local_anchored, out=anchored)
                for group, extent in bounds.items():
                    group_bounds[group] = extent
                for layer_id, voxels in foreground.items():
                    source_records[layer_id]['foreground_voxels'] += voxels
                counts['confidence_known_voxels'] += known
                counts['confidence_unknown_prediction_voxels'] += unknown

            with measure('tta.reconciliation.voting.collection'):
                futures = [executor.submit(collect, lane, np.zeros(block_shape, bool),
                                           np.zeros(block_shape, bool)) for lane in lanes]
                merge(collect(serial_entries, candidate, anchored))
                for future in futures:
                    merge(future.result())
                if futures:
                    del future
                del futures
            if compare_sections:
                with measure('tta.reconciliation.voting.section_collision'):
                    codes = [cached_codes[i] if i in cached_codes else
                             section_codes(descriptor, z0, z1, shape, geometry_context,
                                           angular_tolerance_deg=angular_tolerance)
                             for i, descriptor in enumerate(group_descriptors)]
                    counts['duplicate_group_support_removed'] += _merge_section_votes(
                        scores, direct, extrapolated, codes, pairs, group_bounds)
            # The layer-count check bounds each exact support count to uint16.
            with measure('tta.reconciliation.voting.terminal_reduction'):
                # Collision merging only writes to existing positive support.
                support, direct_support, total, extrapolation_support = _reduce_vote_groups(
                    scores, direct, extrapolated, vote_tiles)
            # Merging duplicate groups preserves the union of their confidence anchors.
            anchored &= direct_support >= 2
            counts['anchored_voxels'] += int(np.count_nonzero(anchored))
            if decide is not None:
                if extrapolation_support is None:
                    extrapolation_support = np.broadcast_to(np.uint16(0), block_shape)
                effective = total.astype(np.float64)
                np.add(effective, anchor_bonus, out=effective, where=anchored)
                block = dict(candidate=candidate, score=effective, support=support,
                             prediction_support=direct_support, extrapolation_support=extrapolation_support,
                             anchored=anchored, z0=z0, z1=z1)
                for value in (candidate, effective, support, direct_support, extrapolation_support, anchored):
                    value.flags.writeable = False
                with measure('tta.reconciliation.voting.callback'):
                    keep = np.asarray(decide(block))
                if keep.dtype != np.bool_ or keep.shape != block_shape:
                    raise ValueError('Custom reconciliation decide() must return a boolean array with the block shape')
            elif policy['mode'] == 'union':
                keep = candidate
            else:
                # Compare at policy precision without upcasting the entire score slab.
                if score_plane is None:
                    meets_threshold = np.greater_equal(
                        total, threshold, signature=(np.float64, np.float64, np.bool_),
                    )
                else:
                    meets_threshold = np.empty(block_shape, bool)
                    for values, anchor, matches in zip(total, anchored, meets_threshold):
                        np.copyto(score_plane, values)
                        np.add(score_plane, anchor_bonus, out=score_plane, where=anchor)
                        np.greater_equal(score_plane, threshold, out=matches)
                keep = (candidate & meets_threshold & (support >= policy['min_sources'])
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
    parallel_bytes = lane_bytes * len(lanes)
    return dict(schema='xta.reconciliation/1', policy={k: v for k, v in policy.items() if k not in {'decide', 'select_proposals'}},
        shape_tyx=list(shape), memory_mib=memory_mib, slab_depth=slab_depth,
        planned_working_bytes=plan['planned_working_bytes'] + parallel_bytes,
        layer_count=len(layers), group_count=group_count,
        groups=list(groups), counts=counts, layers=source_records, components=component_records,
        **({'execution': dict(concurrent_vote_workers=len(lanes),
                             parallel_vote_working_bytes=parallel_bytes)} if lanes else {}),
        evidence_interpretation='positive support; geometry groups cap correlated observations; scores are not calibrated voxel probabilities')
