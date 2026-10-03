"""Experiment-only outer-context contracts; never a production/default change.

Inputs are complete swept-corrected plans with original detector observations.
Existing rasterized contracts are embedded by integer offsets, never rebuilt
using the new image origin. A2 is fixed-raw remeasurement with changed declared
acceptance geometry, not a fresh pipeline-equivalent policy replay.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import operator
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy import ndimage as ndi

from XTA.sam_crop_tiling import TILE_MAX, STRIDE, HALO, MAX_TILES_PER_RUN, axis_windows

SCHEMA = 'xta.sam_outer_crop_geometry/1'
_STACKS = ('write_masks', 'known_foreground_masks', 'unrelated_masks')
_MAPS = ('branch_evaluation_masks', 'edge_write_masks', 'branch_permitted_masks', 'edge_contract_masks')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _plain(value):
    if hasattr(value, 'items'):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value):
    if hasattr(value, 'items'):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(array):
    return np.frombuffer(np.ascontiguousarray(array, dtype=bool).tobytes(), dtype=bool).reshape(array.shape)


def _bbox(mask):
    rows = np.flatnonzero(np.any(mask, axis=1))
    if not rows.size:
        return None
    columns = np.flatnonzero(np.any(mask, axis=0))
    return (int(rows[0]), int(columns[0]), int(rows[-1]) + 1, int(columns[-1]) + 1)


def world_mask_hash(mask, crop, frame):
    """Hash a tight global support box and pixels without a full-canvas stack."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('World-mask proof requires a two-dimensional Boolean mask')
    bounds = _bbox(mask)
    if bounds is None:
        return _hash(dict(frame=int(frame), bbox=None, support=''))
    y0, x0, y1, x1 = bounds
    world = [y0 + crop[0], x0 + crop[1], y1 + crop[0], x1 + crop[1]]
    pixels = np.packbits(np.ascontiguousarray(mask[y0:y1, x0:x1]).reshape(-1), bitorder='little').tobytes()
    return _hash(dict(frame=int(frame), bbox=world, shape=[y1-y0, x1-x0], support=hashlib.sha256(pixels).hexdigest()))


def group_world_hashes(group, observations, *, source_frame_offset=0):
    result = {}
    crop = tuple(group.context_bbox_yx)
    for name in (*_STACKS, 'acceptance_masks'):
        value = np.asarray(getattr(group, name))
        for index, frame in enumerate(group.frame_indices):
            result[f'{name}:{frame}'] = world_mask_hash(value[index], crop, int(frame)+source_frame_offset)
    for name in _MAPS:
        for key, value in sorted(getattr(group, name).items()):
            value = np.asarray(value)
            if value.ndim == 2:
                frame = observations[key].frame_index
                result[f'{name}:{key}:{frame}'] = world_mask_hash(value, crop, int(frame)+source_frame_offset)
            elif value.ndim == 3:
                for index, frame in enumerate(group.frame_indices):
                    result[f'{name}:{key}:{frame}'] = world_mask_hash(value[index], crop, int(frame)+source_frame_offset)
            else:
                raise ValueError('Unexpected contract mask dimensionality')
    return result


def _embed(value, old, new):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.ndim not in (2, 3) or value.shape[-2:] != (old[2]-old[0], old[3]-old[1]):
        raise ValueError('Contract shape differs from its original context')
    if not (new[0] <= old[0] < old[2] <= new[2] and new[1] <= old[1] < old[3] <= new[3]):
        raise ValueError('Expanded context must contain every original contract pixel')
    if old == new:
        return value
    pads = ((old[0]-new[0], new[2]-old[2]), (old[1]-new[1], new[3]-old[3]))
    return _readonly(np.pad(value, ((0, 0), *pads) if value.ndim == 3 else pads))


def _domain_bbox(group):
    # Negative/unrelated detections do not define the family's context scale.
    support = np.zeros(group.acceptance_masks.shape[-2:], bool)
    for value in (group.acceptance_masks, group.write_masks, group.known_foreground_masks):
        for plane in value:
            support |= plane
    for name in _MAPS:
        for value in getattr(group, name).values():
            if value.ndim == 2:
                support |= value
            else:
                for plane in value:
                    support |= plane
    local = _bbox(support)
    if local is None:
        raise ValueError('A planned family has no declared support')
    crop = group.context_bbox_yx
    return tuple(v + crop[i % 2] for i, v in enumerate(local))


def _ellipse(radii):
    ry, rx = radii
    yy, xx = np.indices((2*ry+1, 2*rx+1))
    yy, xx = yy-ry, xx-rx
    if ry == 0:
        return yy == 0
    if rx == 0:
        return xx == 0
    return (yy/ry)**2 + (xx/rx)**2 <= 1.0 + 1e-12


def _expand_acceptance(stack, radii, crop, canvas_yx):
    ry, rx = radii
    if not (ry or rx):
        return stack, []
    result = np.empty_like(stack)
    source_clipped = set()
    footprint = _ellipse(radii)
    for frame, plane in enumerate(stack):
        expanded = ndi.binary_dilation(np.pad(plane, ((ry, ry), (rx, rx))), structure=footprint)
        # Check intended support outside the artificial C rectangle. Source-edge
        # cropping is attributable, and never waives the quality containment test.
        interior = np.zeros_like(expanded)
        interior[ry:ry+plane.shape[0], rx:rx+plane.shape[1]] = True
        yy, xx = np.nonzero(expanded & ~interior)
        yy, xx = yy-ry+crop[0], xx-rx+crop[1]
        if np.any((yy >= 0) & (yy < canvas_yx[0]) & (xx >= 0) & (xx < canvas_yx[1])):
            raise ValueError('Acceptance expansion would be clipped by artificial image context')
        for name, outside in (('top', yy < 0), ('bottom', yy >= canvas_yx[0]),
                              ('left', xx < 0), ('right', xx >= canvas_yx[1])):
            if np.any(outside):
                source_clipped.add(name)
        result[frame] = expanded[ry:ry+plane.shape[0], rx:rx+plane.shape[1]]
    return _readonly(result), sorted(source_clipped)


def build_outer_crop_variant(base_plan, variant, shape_tyx, *, native_scale_yx=(1., 1.),
                            source_frame_offset=0, memory_mib=512,
                            tile_size=TILE_MAX, tile_stride=STRIDE,
                            full_width_group_ids=()):
    """Return an attributable experimental plan and JSON-serializable proof.

    B1 is an identity recipe for the corrected planner. C2/C3 take B1. A2 must
    take C2 and reuses its exact raw/image contracts for offline remeasurement.
    Cfull requires explicit original/current development group IDs.
    """
    variant = str(variant)
    if variant not in {'B1', 'C2', 'C3', 'A2', 'Cfull'}:
        raise ValueError('Outer-crop variant must be B1, C2, C3, A2 or Cfull')
    raw_shape = tuple(shape_tyx)
    if len(raw_shape) != 3 or any(isinstance(v, (bool, np.bool_)) for v in raw_shape):
        raise ValueError('shape_tyx must contain positive integer canvas dimensions')
    try:
        shape = tuple(operator.index(v) for v in raw_shape)
    except TypeError as error:
        raise ValueError('shape_tyx must contain positive integer canvas dimensions') from error
    if any(v <= 0 for v in shape):
        raise ValueError('shape_tyx must contain positive integer canvas dimensions')
    if isinstance(source_frame_offset, (bool, np.bool_)):
        raise ValueError('source_frame_offset must be a nonnegative integer')
    source_frame_offset = operator.index(source_frame_offset)
    if source_frame_offset < 0:
        raise ValueError('source_frame_offset must be a nonnegative integer')
    scale = tuple(float(v) for v in native_scale_yx)
    if len(scale) != 2 or any(not math.isfinite(v) or v <= 0 for v in scale):
        raise ValueError('native_scale_yx must contain two positive finite scales')
    if (tile_size, tile_stride) != (TILE_MAX, STRIDE):
        raise ValueError('Experiments must preserve production 1008/128/752 tiling')
    if isinstance(memory_mib, bool) or not isinstance(memory_mib, int) or memory_mib != 512:
        raise ValueError('Frozen outer-crop experiments require equal declared 512 MiB caps')
    selected_full = set(map(str, full_width_group_ids))
    if variant == 'Cfull' and len(selected_full) != 1:
        raise ValueError('Cfull requires one explicitly selected oversized development group')
    observations = {item.observation_id: item for item in base_plan.observations}
    group_map, groups, records, total = {}, [], [], 0
    refused_ids = set()

    def refuse(group, contract, recipe, reasons, origin):
        """Retain attribution without allocating ungenerated mask contracts."""
        identifier = 'outer_refused_' + _hash(dict(base=group.group_id, variant=variant,
            crop=recipe['context_bbox_yx'], reasons=sorted(reasons)))[:24]
        recipe.update(group_id=identifier, status='refused', refusal_origin=origin,
                      refusal_reasons=sorted(set(reasons)), retained_contract_bytes=0,
                      world_contracts_preserved={}, world_hashes_before={}, world_hashes_after={})
        contract.update(context_bbox_yx=recipe['context_bbox_yx'],
                        unclipped_context_bbox_yx=recipe['requested_context_bbox_yx'],
                        canvas_clamped_sides=recipe['source_clipped_sides'],
                        outer_crop_experiment=recipe)
        empty = _readonly(np.empty((0,), dtype=bool))
        masks = {name: empty for name in (*_STACKS, 'acceptance_masks')}
        masks.update({name: MappingProxyType({}) for name in _MAPS})
        groups.append(replace(group, group_id=identifier,
            context_bbox_yx=tuple(recipe['context_bbox_yx']), crop_contract=_freeze(contract),
            status='unresolved', reasons=tuple(sorted(set((*group.reasons, *reasons)))), **masks))
        records.append(_plain(recipe))
        group_map[group.group_id] = identifier
        refused_ids.add(group.group_id)

    for group in base_plan.groups:
        old = tuple(map(int, group.context_bbox_yx))
        if not (0 <= old[0] < old[2] <= shape[1] and 0 <= old[1] < old[3] <= shape[2]):
            raise ValueError('Original crop lies outside its declared canvas')
        contract = _plain(getattr(group, 'crop_contract', {}))
        prior = contract.get('outer_crop_experiment', {})
        original_group = str(prior.get('original_group_id', group.group_id))
        lineage = str(prior.get('original_seed_lineage_key') or _hash(dict(
            observations=sorted(group.observation_ids), endpoints=sorted(group.endpoint_ids),
            edges=sorted((edge.source_id, edge.target_id, edge.candidate_index) for edge in group.edges))))
        if group.status != 'planned':
            recipe = dict(schema=SCHEMA, variant=variant, original_group_id=original_group,
                base_group_id=str(group.group_id), original_seed_lineage_key=lineage,
                baseline_group_status=group.status,
                legacy_raster_origin_yx=contract.get('legacy_raster_origin_yx'),
                base_context_bbox_yx=list(old), requested_context_bbox_yx=list(old),
                context_bbox_yx=list(old), source_clipped_sides=[],
                quality_waiver=False, fresh_pipeline_equivalent=False,
                raw_reuse_parent_group_id=None)
            refuse(group, contract, recipe, group.reasons or ('baseline_family_unresolved',), 'baseline')
            continue
        if 'legacy_raster_origin_yx' not in contract:
            raise ValueError('Corrected B1 must retain its literal-v25 raster reference')
        if variant == 'A2' and prior.get('variant') != 'C2':
            raise ValueError('A2 requires the identical C2 image/raw geometry')
        if variant in {'C2', 'C3', 'Cfull'} and prior.get('variant') not in (None, 'B1'):
            raise ValueError('Context variants require the corrected B1 baseline')
        domain = _domain_bbox(group)
        span = (domain[2]-domain[0], domain[3]-domain[1])
        long_axis = max(range(2), key=lambda i: (span[i]*scale[i], i))
        requested, guard = list(old), None
        if variant == 'B1':
            requested = list(contract.get('unclipped_context_bbox_yx', old))
        elif variant == 'A2':
            requested = list(prior.get('requested_context_bbox_yx', old))
        if variant in {'C2', 'C3'}:
            target = 14 * (2 if variant == 'C2' else 3)
            guard = math.ceil(target * span[long_axis] / (1008 - 2*target))
            requested[long_axis] = min(old[long_axis], domain[long_axis]-guard)
            requested[long_axis+2] = max(old[long_axis+2], domain[long_axis+2]+guard)
        if variant == 'Cfull' and (group.group_id in selected_full or original_group in selected_full):
            if long_axis != 1:
                raise ValueError('Cfull development diagnostic requires an oversized X family')
            requested[1], requested[3] = 0, shape[2]
        new = (max(0, requested[0]), max(0, requested[1]),
               min(shape[1], requested[2]), min(shape[2], requested[3]))
        clipped = [name for i, name in enumerate(('top', 'left', 'bottom', 'right')) if requested[i] != new[i]]
        pixels = (new[2]-new[0])*(new[3]-new[1])
        endpoints = len(group.endpoint_ids)
        charge = pixels * (2*len(group.frame_indices)*(5+2*len(group.edges)) + 4*endpoints + 16)
        topology = pixels * len(group.frame_indices) * 16
        old_tiles = [len(axis_windows(old[i], old[i+2])) for i in range(2)]
        new_tiles = [len(axis_windows(new[i], new[i+2])) for i in range(2)]
        refusal_reasons = []
        if pixels > 4_194_304:
            refusal_reasons.append('outer_context_crop_pixel_limit')
        if max(charge, topology) > 512 * 1024**2:
            refusal_reasons.append('outer_context_group_memory_limit')
        if total + charge > 512 * 1024**2:
            refusal_reasons.append('outer_context_total_memory_limit')
        if new_tiles[0] != old_tiles[0] or new_tiles[1] > old_tiles[1]+1 or math.prod(new_tiles) > MAX_TILES_PER_RUN:
            refusal_reasons.append('outer_context_tile_growth_limit')
        if refusal_reasons:
            recipe = dict(schema=SCHEMA, variant=variant, original_group_id=original_group,
                base_group_id=str(group.group_id), original_seed_lineage_key=lineage,
                legacy_raster_origin_yx=contract['legacy_raster_origin_yx'],
                domain_bbox_yx=list(domain), base_context_bbox_yx=list(old),
                requested_context_bbox_yx=requested, context_bbox_yx=list(new),
                source_clipped_sides=clipped, charged_contract_bytes=charge,
                topology_workspace_bytes=topology, requested_total_at_decision_bytes=total+charge,
                tiles_before_yx=old_tiles, tiles_after_yx=new_tiles,
                quality_waiver=False, fresh_pipeline_equivalent=False,
                raw_reuse_parent_group_id=None)
            refuse(group, contract, recipe, refusal_reasons, 'variant')
            continue
        updates = {name: _embed(getattr(group, name), old, new) for name in (*_STACKS, 'acceptance_masks')}
        for name in _MAPS:
            updates[name] = MappingProxyType({key: _embed(value, old, new)
                                             for key, value in getattr(group, name).items()})
        radii, a_clipped = [0, 0], []
        if variant == 'A2':
            radii = [max(0, math.ceil(7*(new[i+2]-new[i])/1008)-8) for i in range(2)]
            try:
                updates['acceptance_masks'], a_clipped = _expand_acceptance(
                    updates['acceptance_masks'], radii, new, shape[1:])
            except ValueError as error:
                if 'clipped by artificial image context' not in str(error):
                    raise
                recipe = dict(schema=SCHEMA, variant=variant, original_group_id=original_group,
                    base_group_id=str(group.group_id), original_seed_lineage_key=lineage,
                    legacy_raster_origin_yx=contract['legacy_raster_origin_yx'],
                    base_context_bbox_yx=list(old), requested_context_bbox_yx=requested,
                    context_bbox_yx=list(new), source_clipped_sides=clipped,
                    acceptance_extra_radii_yx=radii, quality_waiver=False,
                    fresh_pipeline_equivalent=False, raw_reuse_parent_group_id=str(group.group_id))
                refuse(group, contract, recipe, ('acceptance_artificial_context_clipping',), 'variant')
                continue
        before = group_world_hashes(group, observations, source_frame_offset=source_frame_offset)
        group_id = group.group_id if variant == 'B1' else 'outer_group_' + _hash(
            dict(lineage=lineage, variant=variant, crop=new, radii=radii))[:24]
        recipe = dict(schema=SCHEMA, variant=variant, status='planned', refusal_reasons=[],
            original_group_id=original_group,
            base_group_id=str(group.group_id), group_id=str(group_id), original_seed_lineage_key=lineage,
            legacy_raster_origin_yx=contract['legacy_raster_origin_yx'], domain_bbox_yx=list(domain),
            base_context_bbox_yx=list(old), requested_context_bbox_yx=requested, context_bbox_yx=list(new),
            embedding_offset_yx=[old[0]-new[0], old[1]-new[1]], long_axis='yx'[long_axis],
            native_scale_yx=list(scale), sampler_side=1008, vit_patch=14, target_model_guard_px=None if guard is None else (28 if variant=='C2' else 42),
            requested_long_guard_px=guard, acceptance_extra_radii_yx=radii,
            acceptance_scale_basis='family_C2_model_grid_heuristic_shared_by_whole_and_tiled',
            acceptance_is_child_tile_uncertainty_measurement=False,
            source_clipped_sides=clipped, acceptance_source_clipped_sides=a_clipped,
            quality_waiver=False, fresh_pipeline_equivalent=False,
            charged_contract_bytes=charge, topology_workspace_bytes=topology,
            retained_contract_bytes=charge,
            tiles_before_yx=old_tiles, tiles_after_yx=new_tiles,
            tiling=dict(tile_max=TILE_MAX, halo=HALO, stride=STRIDE),
            raw_reuse_parent_group_id=str(group.group_id) if variant == 'A2' else None)
        baseline_contract = contract.get('baseline_crop_contract')
        if baseline_contract is None:
            baseline_contract = dict(contract)
            baseline_contract.pop('outer_crop_experiment', None)
        contract.update(context_bbox_yx=list(new), unclipped_context_bbox_yx=requested,
                        canvas_clamped_sides=clipped, canvas_shape_yx=list(shape[1:]),
                        crop_pixels=pixels, charged_contract_bytes=charge,
                        baseline_crop_contract=baseline_contract,
                        scalar_padding_fields_role='baseline_floor_only; anisotropic context is declared in outer_crop_experiment',
                        outer_crop_experiment=recipe)
        new_group = replace(group, group_id=group_id, context_bbox_yx=new,
                            crop_contract=MappingProxyType(contract), **updates)
        after = group_world_hashes(new_group, observations, source_frame_offset=source_frame_offset)
        preserved = {key: before[key] == after[key] for key in before
                     if variant != 'A2' or not key.startswith('acceptance_masks:')}
        if not all(preserved.values()):
            raise AssertionError('World-coordinate contract pixels changed during rebasing')
        recipe['world_hashes_before'], recipe['world_hashes_after'] = before, after
        recipe['world_contracts_preserved'] = preserved
        new_group = replace(new_group, crop_contract=_freeze(contract))
        group_map[group.group_id] = group_id
        groups.append(new_group)
        records.append(_plain(recipe))
        total += charge
    if variant == 'Cfull' and not any(record['base_group_id'] in selected_full or record['original_group_id'] in selected_full for record in records):
        raise ValueError('Explicit Cfull group does not occur in the baseline plan')
    runs, run_maps, removed_runs = [], [], []
    for run in base_plan.runs:
        if run.group_id in refused_ids:
            removed_runs.append(dict(base_run_id=run.run_id, base_group_id=run.group_id,
                                     status='refused_no_model_job'))
            continue
        new_id = run.run_id if variant == 'B1' else 'outer_run_' + _hash(dict(
            base_run_id=run.run_id, group_id=group_map[run.group_id], variant=variant))[:24]
        runs.append(replace(run, run_id=new_id, group_id=group_map[run.group_id]))
        run_maps.append(dict(run_id=new_id, base_run_id=run.run_id,
            raw_reuse_parent_run_id=run.run_id if variant == 'A2' else None,
            seed_ids=list(run.seed_ids), held_out_ids=list(run.held_out_ids), expected_frames=list(run.expected_frames)))
    proof = dict(schema=SCHEMA, variant=variant, source_frame_offset=int(source_frame_offset),
                 coordinate_space='working_canvas', shape_tyx=list(shape), groups=records, runs=run_maps,
                 recipe_parameters=dict(native_scale_yx=list(scale), sampler_side=1008, vit_patch=14,
                     context_tokens={'C2': 2, 'C3': 3}, acceptance_fpn_cells=2,
                     acceptance_fpn_step_px=3.5, legacy_acceptance_over_write_px=8,
                     full_width_group_ids=sorted(selected_full)),
                 caps=dict(planner_group_bytes=512*1024**2, planner_total_bytes=512*1024**2,
                           policy_topology_bytes=512*1024**2, max_crop_pixels=4_194_304),
                 total_charged_contract_bytes=total, fresh_pipeline_equivalent=False,
                 original_family_count=len(base_plan.groups),
                 planned_family_count=len(groups)-len(refused_ids), refused_family_count=len(refused_ids),
                 cohort_complete=not refused_ids and base_plan.status != 'unresolved',
                 removed_runs=removed_runs, baseline_plan_status=base_plan.status,
                 baseline_plan_reasons=list(base_plan.reasons),
                 interpretation='Changed declared A diagnostic with exact C2 raw reuse' if variant=='A2' else 'New image-context generation attempt')
    proof['recipe_sha256'] = _hash(proof)
    fingerprint = _hash(dict(base=base_plan.planning_fingerprint, recipe_sha256=proof['recipe_sha256']))
    status = ('partial' if runs else 'unresolved') if refused_ids else base_plan.status
    reasons = tuple(sorted(set((*base_plan.reasons, 'outer_crop_experiment_refused_groups')))) if refused_ids else base_plan.reasons
    plan = replace(base_plan, groups=tuple(groups), runs=tuple(runs), planning_fingerprint=fingerprint,
                   status=status, reasons=reasons)
    return plan, proof


def write_geometry_proof(path, proof):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    return path


__all__ = ['build_outer_crop_variant', 'group_world_hashes', 'world_mask_hash', 'write_geometry_proof']
