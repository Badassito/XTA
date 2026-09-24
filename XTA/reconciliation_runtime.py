"""TTA adapter for the frontend-independent reconciliation engine."""
from __future__ import annotations

from dataclasses import fields
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .reconciliation import EvidenceLayer, evidence_role, reconcile
from .reconciliation_policy import load_reconciliation_policy, validate_policy


def union_nrrd_exports_can_overlap(layer_sink, assembled_union, *, policy, source_shape_tyx):
    """Allow union postprocessing beside completed, independent component stores.

    Called only after scheduler producers have joined. The final sink wait still
    precedes output-manifest publication and scratch cleanup. Unknown backings,
    live arrays and ordinary raw maps retain the earlier export barrier.
    """
    from .interpolation import CTILE_INDEX_DTYPE, MASK_STORE_FORMATS

    policy = validate_policy(policy)
    if (policy['mode'] != 'union' or policy['decide'] is not None
            or not isinstance(assembled_union, np.ndarray)
            or tuple(assembled_union.shape) != tuple(source_shape_tyx)
            or assembled_union.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_))):
        return False
    snapshot = getattr(layer_sink, 'pending_layer_refs', None)
    if not callable(snapshot):
        return False

    # np.asarray(memmap) and its views retain the memmap in their base chain.
    # An unidentifiable external buffer cannot prove file-backed non-aliasing.
    union_files = set()
    owner = assembled_union
    seen = set()
    while isinstance(owner, np.ndarray) and id(owner) not in seen:
        seen.add(id(owner))
        if isinstance(owner, np.memmap):
            try:
                identity = os.stat(owner.filename)
            except (OSError, TypeError, ValueError):
                return False
            if not identity.st_ino:
                return False
            union_files.add((identity.st_dev, identity.st_ino))
            break
        if owner.base is None and owner.flags.owndata:
            break
        owner = owner.base
    else:
        return False

    for ref in snapshot():
        if (getattr(ref, 'live_array', None) is not None
                or getattr(ref, 'storage_format', '') not in MASK_STORE_FORMATS):
            return False
        try:
            root = Path(ref.path)
            metadata = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
            shape = tuple(int(value) for value in metadata['shape'])
            paths = [root / name for name in ('meta.json', 'index.bin', 'chunks.bin')]
            identities = [path.stat() for path in paths]
            # Raw-bbox writers publish meta.json after closing the payload and
            # writing the final index. Their immutable files remain owned by the
            # run until the final sink join; check that this is a complete store.
            if (metadata['format'] != ref.storage_format or shape != tuple(ref.shape)
                    or len(shape) != 3 or min(shape) <= 0
                    or metadata['index_record_bytes'] != CTILE_INDEX_DTYPE.itemsize
                    or identities[1].st_size != shape[0] * CTILE_INDEX_DTYPE.itemsize
                    or metadata['stats']['raw_payload_bytes'] != identities[2].st_size
                    or any(not item.st_ino or (item.st_dev, item.st_ino) in union_files
                           for item in identities)):
                return False
        except (OSError, ValueError, TypeError, KeyError):
            return False
    return True


def preflight_reconciliation(views, *, source_shape_tyx, processing_shape_tyx, settings, policy):
    """Check a conservative planned-view workspace before model inference begins."""
    from .reconciliation import voting_memory_plan
    from .reconciliation_geometry import section_descriptor
    context = dict(views_by_name={str(view.name): view for view in views},
                   source_shape_tyx=tuple(source_shape_tyx), processing_shape_tyx=tuple(processing_shape_tyx))
    groups = set()
    union_only = policy['mode'] == 'union' and policy['decide'] is None
    if union_only:
        return {**_union_reuse_count_plan(source_shape_tyx, settings.memory_mib),
                'planned_group_count': 0, 'memory_mib': settings.memory_mib}
    if not union_only:
        for view in views:
            metadata = dict(view_name=view.name, physical_view_name=view.physical_view_name or view.name,
                            view_family=view.family)
            key = (section_descriptor(metadata, geometry_context=context)['group_key']
                   if policy['grouping'] == 'sections' else metadata['physical_view_name'])
            groups.add(key)
    plan = voting_memory_plan(source_shape_tyx, len(groups), settings.memory_mib, union_only=union_only)
    if policy['island_weighting'] and not union_only:
        from .reconciliation_components import _plan_slabs
        _plan_slabs(source_shape_tyx, float(settings.memory_mib) * .8)
    return {**plan, 'planned_group_count': len(groups), 'memory_mib': settings.memory_mib}


class RuntimeLayer:
    """Read an immutable component backing with the existing output mapping."""
    def __init__(self, ref, output_shape):
        from .outputs import _open_nrrd_layer_ref, _nrrd_layer_ref_is_raw_bbox_store
        from .interpolation import RawBBoxMaskStore
        self.ref, self.shape_tyx = ref, tuple(map(int, output_shape))
        self.source = (RawBBoxMaskStore.open(ref.path, mmap_payload=True)
                       if _nrrd_layer_ref_is_raw_bbox_store(ref) else _open_nrrd_layer_ref(ref))

    def read_slab(self, z0, z1):
        from .outputs import _read_layer_slice_in_output_shape
        if not 0 <= z0 <= z1 <= self.shape_tyx[0]:
            raise IndexError('Reconciliation read is outside the reference grid')
        return np.stack([_read_layer_slice_in_output_shape(self.source, self.shape_tyx, z)
                         for z in range(z0, z1)]).astype(np.uint8, copy=False)

    def close(self):
        from .outputs import _close_nrrd_layer_source
        if self.source is not None:
            _close_nrrd_layer_source(self.source)
            self.source = None


def _union_reuse_count_plan(shape_tyx, memory_mib):
    """Bound count windows while borrowing the pipeline's already-owned union."""
    shape = tuple(int(value) for value in shape_tyx)
    if len(shape) != 3 or min(shape) <= 0 or not math.isfinite(float(memory_mib)) or memory_mib <= 0:
        raise ValueError('Union reuse requires a positive TYX shape and memory budget')
    budget = int(float(memory_mib) * 1024**2)
    if budget < shape[2]:
        raise ValueError(f'Union counting needs at least {shape[2]} bytes for one row')
    rows = min(shape[1], max(1, budget // shape[2]))
    return dict(strategy='reuse_assembled_union', slab_depth=1, count_window_rows=rows,
                minimum_working_bytes=shape[2], planned_working_bytes=rows * shape[2],
                bytes_per_voxel=1, new_source_volume_bytes=0)


def _reconciliation_refs(refs, shape, *, needs_confidence):
    """Resolve layer identity and metadata without opening component payloads."""
    from .confidence_evidence import lookup_confidence_evidence
    from .outputs import _nrrd_layer_zero_skip_window
    result, seen = [], {}
    for ref in refs:
        if (str(getattr(ref, 'layer_role', 'additive_component')) != 'additive_component'
                or str(getattr(ref, 'recomposition_op', 'union')) != 'union'
                or str(getattr(ref, 'source', '')) == 'global'):
            continue
        identity = f'{ref.model_name}/{ref.key}'
        signature = (str(Path(ref.path).resolve()), tuple(ref.shape), ref.source, ref.mask_kind)
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError(f'Conflicting reconciliation layer identity: {identity}')
            continue
        seen[identity] = signature
        metadata = {field.name: getattr(ref, field.name) for field in fields(ref)
                    if field.name not in {'live_array', 'path'}}
        metadata.update(layer_key=ref.key, empty_segment=_nrrd_layer_zero_skip_window(ref, shape) == (0, 0))
        evidence_role(metadata)
        confidence = lookup_confidence_evidence(ref)
        if confidence is not None:
            if needs_confidence and tuple(confidence.shape) != shape:
                raise ValueError(f'Confidence grid does not match layer {identity}')
            metadata['confidence_evidence'] = str(confidence.path)
            metadata['confidence_coordinate_space'] = getattr(confidence, 'coordinate_space', 'source')
            metadata['confidence_storage_shape_tyx'] = list(getattr(confidence, 'storage_shape', confidence.shape))
        result.append((identity, ref, metadata, confidence))
    return sorted(result, key=lambda item: item[0])


def _publish_reconciliation_report(report, records, settings, output_dir):
    """Publish the selected policy and truthful metadata for either execution path."""
    settings.assert_unchanged()
    report.update(policy_path=settings.path, policy_sha256=settings.sha256,
                  stage='source_grid_before_global_postprocessing',
                  confidence_evidence='reconciliation_evidence/manifest.json',
                  source_layers_preserved=True)
    for identity, _ref, metadata, _confidence in records:
        report['layers'][identity]['metadata'] = {
            key: metadata.get(key) for key in ('model_name', 'layer_key', 'view_name',
                'physical_view_name', 'view_family', 'source', 'mask_kind', 'tile_config_id',
                'tile_acceptance', 'stage', 'confidence_evidence', 'confidence_coordinate_space',
                'confidence_storage_shape_tyx')}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / 'policy.py'
    snapshot.write_bytes(Path(settings.path).read_bytes())
    settings.assert_unchanged()
    pending = output_dir / 'manifest.json.partial'
    try:
        pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        pending.replace(output_dir / 'manifest.json')
    finally:
        pending.unlink(missing_ok=True)


def _reuse_assembled_union(assembled_union, records, *, shape, settings, policy, output_dir):
    """Borrow the final additive union, counting it once without rebuilding it."""
    if (not isinstance(assembled_union, np.ndarray) or tuple(assembled_union.shape) != shape
            or assembled_union.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_))):
        raise ValueError('Assembled reconciliation union must be a uint8/bool array on the source grid')
    plan = _union_reuse_count_plan(shape, settings.memory_mib)
    total, windows = 0, 0
    # Counting reads borrowed views only. Windows are bounded even when a whole
    # native XY plane exceeds the requested budget; no dense output is allocated.
    for z in range(shape[0]):
        for y in range(0, shape[1], plan['count_window_rows']):
            window = assembled_union[z, y:y + plan['count_window_rows']]
            if assembled_union.dtype == np.uint8 and int(window.max(initial=0)) > 1:
                raise ValueError('Assembled reconciliation union must contain binary 0/1 values')
            total += int(np.count_nonzero(window))
            windows += 1
    layers = {identity: dict(group=0, role=evidence_role(metadata), island_weight=1.,
        foreground_voxels=0 if metadata['empty_segment'] else None,
        foreground_count_source='empty_segment_metadata' if metadata['empty_segment'] else 'not_rescanned')
        for identity, _ref, metadata, _confidence in records}
    report = dict(schema='xta.reconciliation/1', policy={key: value for key, value in policy.items() if key != 'decide'},
        shape_tyx=list(shape), memory_mib=settings.memory_mib,
        slab_depth=plan['slab_depth'], planned_working_bytes=plan['planned_working_bytes'],
        layer_count=len(records), group_count=int(bool(records)), groups=['union'] if records else [],
        layers=layers, components={},
        counts=dict(candidate_voxels=total, retained_voxels=total, rejected_voxels=0,
                    duplicate_group_support_removed=None, confidence_known_voxels=None,
                    confidence_unknown_prediction_voxels=None, anchored_voxels=None),
        execution=dict(strategy='reuse_assembled_union', output_ownership='borrowed',
                       component_payload_reads=0, confidence_payload_reads=0, new_source_volume_bytes=0,
                       count_source='bounded_assembled_union_count', count_windows=windows,
                       logical_union_count_bytes=math.prod(shape) * assembled_union.dtype.itemsize,
                       binary_validation='maximum_per_count_window' if assembled_union.dtype == np.uint8 else 'boolean_dtype',
                       count_window_rows=plan['count_window_rows']),
        unmeasured_counts=['layer_foreground_voxels_without_empty_metadata',
                           'confidence_known_voxels', 'confidence_unknown_prediction_voxels',
                           'duplicate_group_support_removed', 'anchored_voxels'],
        evidence_interpretation='reuse of the pipeline-assembled additive union before global postprocessing')
    _publish_reconciliation_report(report, records, settings, output_dir)
    return assembled_union, report


def reconcile_tta_layers(refs, *, views, source_shape_tyx, processing_shape_tyx,
                         settings, output_dir, workspace, policy=None, assembled_union=None):
    """Produce one source-grid result; component backings remain unmodified."""
    policy = load_reconciliation_policy(settings) if policy is None else validate_policy(policy)
    settings.assert_unchanged()
    shape = tuple(map(int, source_shape_tyx))
    needs_confidence = policy['mode'] == 'confidence'
    records = _reconciliation_refs(refs, shape, needs_confidence=needs_confidence)
    if policy['mode'] == 'union' and policy['decide'] is None and assembled_union is not None:
        return _reuse_assembled_union(assembled_union, records, shape=shape, settings=settings,
                                     policy=policy, output_dir=output_dir)
    output_dir, workspace = Path(output_dir), Path(workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / 'reconciled_union.u8.dat'
    if path.exists():
        raise FileExistsError(f'Reconciliation output already exists: {path}')
    context = dict(views_by_name={str(view.name): view for view in views},
                   source_shape_tyx=shape, processing_shape_tyx=tuple(map(int, processing_shape_tyx)))
    owners, confidence_owners, layers = [], [], []
    output = None
    try:
        for identity, ref, metadata, confidence in records:
            owner = RuntimeLayer(ref, shape)
            owners.append(owner)
            confidence_reader = confidence.reader() if needs_confidence and confidence is not None else None
            if confidence_reader is not None:
                confidence_owners.append(confidence_reader)
            layers.append(EvidenceLayer(identity, shape, metadata, owner.read_slab,
                                        confidence_reader))
        output = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
        last = [0.]
        def progress(stage, current, total):
            now = time.monotonic()
            if now - last[0] >= 5 or current == total:
                print(f'Reconciliation {stage}: {current}/{total}', flush=True)
                last[0] = now
        def write(z0, z1, value):
            output[z0:z1] = value
        report = reconcile(layers, shape_tyx=shape, policy=policy, write_slab=write,
                           memory_mib=settings.memory_mib, geometry_context=context, progress=progress)
        settings.assert_unchanged()
        output.flush()
        _publish_reconciliation_report(report, records, settings, output_dir)
        return output, report
    except BaseException:
        if output is not None:
            output._mmap.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        try:
            with ExitStack() as cleanup:
                for owner in owners:
                    cleanup.callback(owner.close)
                for owner in confidence_owners:
                    close = getattr(owner, 'close', None)
                    if callable(close):
                        cleanup.callback(close)
        except BaseException:
            if output is not None:
                output._mmap.close()
            path.unlink(missing_ok=True)
            raise
