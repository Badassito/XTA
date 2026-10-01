"""Model-free export of selected SAM proposals as view-native binary NRRDs.

These files are fixed-proposal scope replays. They do not re-run the upstream
tile gate, regenerate missing hypotheses, or claim source-grid projection.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from .reconciliation_io import ReferenceGeometry, write_seg_nrrd
from .sam_evidence import SamEvidenceBundle
from .sam_policy import resolve_sam_bridge_policy, select_sam_proposals
from .sam_mask_reader import effective_candidate_mask


def _selected_plane(bundle, run_ids, frame, shape_yx, selection=None):
    output = np.zeros(shape_yx, dtype=np.uint8)
    for run_id in run_ids:
        run = bundle.runs[run_id]
        if str(frame) not in run['candidate_mask_keys']:
            continue
        group = bundle.groups[run['group_id']]
        y0, x0, y1, x1 = map(int, group['context_bbox_yx'])
        if not (0 <= y0 < y1 <= shape_yx[0] and 0 <= x0 < x1 <= shape_yx[1]):
            raise ValueError('SAM replay crop lies outside its declared native canvas')
        mask = effective_candidate_mask(bundle, run_id, frame, selection)
        if mask.shape != (y1 - y0, x1 - x0):
            raise ValueError('SAM replay mask differs from its declared crop')
        output[y0:y1, x0:x1] |= mask
    return output


def replay_sam_directional_nrrds(bundle, output, *, policy=None,
                                upstream_fingerprints=None,
                                frozen_evidence=False, memory_mib=256):
    """Select immutable proposals and atomically publish two slots per pass.

    Spatial axes are the recorded view-native X, Y and increasing frame index.
    The manifest retains the original transform and dependency snapshot. A
    nonidentity canvas needs separate projection to compare source-grid files.
    No detector, Torch, or SAM runtime is imported by this module.
    """
    budget = float(memory_mib)
    if isinstance(memory_mib, bool) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('SAM replay memory_mib must be finite and positive')
    budget_bytes = int(budget * 1024**2)
    reader_cache_bytes = min(32 * 1024**2, max(0, budget_bytes // 8))
    if not isinstance(bundle, SamEvidenceBundle):
        bundle = SamEvidenceBundle.open(bundle, max_mask_bytes=max(1, budget_bytes // 4))
    shape = tuple(bundle.scope.get('shape_tyx', ()))
    if (len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)):
        raise ValueError('Directional SAM replay requires its recorded shape_tyx')
    shape = tuple(map(int, shape))
    largest_crop = max((math.prod(record['shape']) for record in bundle.records.values()), default=0)
    if 4 * math.prod(shape[1:]) + 32 * largest_crop + reader_cache_bytes > budget_bytes:
        raise ValueError('SAM replay budget must fit native planes and mask measurements')
    # The operational replay budget must cover proposal selection too. Reject
    # oversized topology before selection without changing quality policy/hash.
    source_policy = policy or {}
    if 'kind' in source_policy and 'mode' not in source_policy:
        source_policy = {'sam_bridge_policy': source_policy}
    policy_group_bytes = int(resolve_sam_bridge_policy(source_policy)['max_group_bytes'])
    for group in bundle.groups.values():
        y0, x0, y1, x1 = map(int, group['context_bbox_yx'])
        if not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
            raise ValueError('SAM replay crop lies outside its declared native canvas')
        frames = tuple(group['frame_indices'])
        if any(isinstance(frame, bool) or not isinstance(frame, int)
               or not 0 <= frame < shape[0] for frame in frames):
            raise ValueError('SAM replay frame lies outside its declared native canvas')
        topology_bytes = len(frames) * (y1 - y0) * (x1 - x0) * 16
        if topology_bytes > min(budget_bytes, policy_group_bytes):
            raise ValueError('SAM replay budget cannot fit bounded group topology')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('SAM replay destination must be fresh')
    selection = select_sam_proposals(bundle, policy=policy,
        upstream_fingerprints=upstream_fingerprints, frozen_evidence=frozen_evidence,
        reader_cache_bytes=reader_cache_bytes)
    spacing = tuple(float(v) for v in bundle.scope.get('spacing_zyx', (1., 1., 1.)))
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError('Invalid recorded SAM native spacing')
    # Generic right-handed view coordinates avoid inventing anatomical/source
    # orientation for processing canvases. The source transform stays explicit.
    geometry = ReferenceGeometry(shape, space='3D-right-handed', directions_xyz=(
        (spacing[2], 0., 0.), (0., spacing[1], 0.), (0., 0., spacing[0])))
    passes = sorted({int(run['pass_index']) for run in bundle.runs.values()})
    if not passes:
        passes = [int(bundle.scope.get('pass_index', 1))]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ('.' + output.name + '.replay-' + uuid.uuid4().hex)
    staging.mkdir()
    layers = []
    try:
        with bundle.reader(max_cache_bytes=reader_cache_bytes) as reader:
            selection_view = {**selection, 'mask_filter': reader.filter_snapshot(selection)}
            for pass_index in passes:
                for direction in ('forward', 'backward'):
                    run_ids = tuple(run_id for run_id in selection['selected_run_ids']
                        if int(reader.runs[run_id]['pass_index']) == pass_index
                        and reader.runs[run_id]['direction'] == direction)
                    name = f'sam_bridge_pass{pass_index:02d}_{direction}.seg.nrrd'
                    def read_slab(start, stop):
                        return np.stack([_selected_plane(reader, run_ids, frame, shape[1:], selection_view)
                                         for frame in range(start, stop)])
                    artifact = write_seg_nrrd(staging / name, shape_tyx=shape,
                        read_slab=read_slab, geometry=geometry, memory_mib=budget,
                        chunk_slices=1, segment_name=f'Selected SAM {direction} pass {pass_index}')
                    artifact['path'] = name
                    layers.append(dict(artifact, interpolation_backend='sam',
                        interpolation_direction=direction, pass_index=pass_index,
                        proposal_selection_status='policy_selected', sam_run_ids=list(run_ids),
                        interpolation_policy_identity=selection['policy_hash']))
        manifest = dict(schema='xta.sam_directional_replay/1', complete=True,
            coordinate_space='view_native', source_grid_projected=False,
            direction_semantics='increasing/decreasing view-native frame index',
            shape_tyx=list(shape), spacing_zyx=list(spacing),
            evidence_fingerprint=bundle.evidence_fingerprint,
            scope=json.loads(json.dumps(dict(bundle.scope), default=_json_value)),
            selection=selection, layers=layers,
            mask_reader=dict(reader.stats),
            limitation='Fixed proposals only; changed upstream tile support requires regeneration.')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True,
            allow_nan=False), encoding='utf-8')
        if SamEvidenceBundle.open(bundle.directory).evidence_fingerprint != bundle.evidence_fingerprint:
            raise RuntimeError('SAM evidence changed during directional replay')
        if output.exists():
            raise FileExistsError('SAM replay destination appeared during export')
        os.replace(staging, output)
    except BaseException as error:
        # Only this invocation's UUID-owned directory is eligible for cleanup.
        if staging.exists():
            shutil.rmtree(staging)
        if isinstance(error, ValueError) and 'SAM evidence' in str(error) and 'changed' in str(error):
            raise RuntimeError('SAM evidence changed during directional replay') from error
        raise
    return manifest


def _json_value(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f'Unsupported retained SAM metadata type: {type(value).__name__}')


__all__ = ['replay_sam_directional_nrrds']
