"""TTA adapter for the frontend-independent reconciliation engine."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np

from .reconciliation import EvidenceLayer, reconcile
from .reconciliation_policy import load_reconciliation_policy


def preflight_reconciliation(views, *, source_shape_tyx, processing_shape_tyx, settings, policy):
    """Check a conservative planned-view workspace before model inference begins."""
    from .reconciliation import voting_memory_plan
    from .reconciliation_geometry import section_descriptor
    context = dict(views_by_name={str(view.name): view for view in views},
                   source_shape_tyx=tuple(source_shape_tyx), processing_shape_tyx=tuple(processing_shape_tyx))
    groups = set()
    union_only = policy['mode'] == 'union' and policy['decide'] is None
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


def reconcile_tta_layers(refs, *, views, source_shape_tyx, processing_shape_tyx,
                         settings, output_dir, workspace, policy=None):
    """Produce one source-grid result; component backings remain unmodified."""
    from .confidence_evidence import lookup_confidence_evidence
    from .outputs import _nrrd_layer_zero_skip_window
    policy = policy or load_reconciliation_policy(settings)
    shape = tuple(map(int, source_shape_tyx))
    output_dir, workspace = Path(output_dir), Path(workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / 'reconciled_union.u8.dat'
    if path.exists():
        raise FileExistsError(f'Reconciliation output already exists: {path}')
    context = dict(views_by_name={str(view.name): view for view in views},
                   source_shape_tyx=shape, processing_shape_tyx=tuple(map(int, processing_shape_tyx)))
    owners, layers, seen = [], [], {}
    output = None
    try:
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
            owner = RuntimeLayer(ref, shape)
            owners.append(owner)
            metadata = {field.name: getattr(ref, field.name) for field in fields(ref)
                        if field.name not in {'live_array', 'path'}}
            metadata.update(layer_key=ref.key, empty_segment=_nrrd_layer_zero_skip_window(ref, shape) == (0, 0))
            confidence = lookup_confidence_evidence(ref)
            if confidence is not None and tuple(confidence.shape) != shape:
                raise ValueError(f'Confidence grid does not match layer {identity}')
            if confidence is not None:
                metadata['confidence_evidence'] = str(confidence.path)
            layers.append(EvidenceLayer(identity, shape, metadata, owner.read_slab,
                                        confidence.reader() if confidence is not None else None))
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
        report.update(policy_path=settings.path, policy_sha256=settings.sha256,
                      stage='source_grid_before_global_postprocessing',
                      confidence_evidence='reconciliation_evidence/manifest.json',
                      source_layers_preserved=True)
        for layer in layers:
            report['layers'][layer.layer_id]['metadata'] = {
                key: layer.metadata.get(key) for key in ('model_name', 'layer_key', 'view_name',
                    'physical_view_name', 'view_family', 'source', 'mask_kind', 'tile_config_id',
                    'tile_acceptance', 'stage', 'confidence_evidence')}
        snapshot = output_dir / 'policy.py'
        snapshot.write_bytes(Path(settings.path).read_bytes())
        settings.assert_unchanged()
        pending = output_dir / 'manifest.json.partial'
        pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        pending.replace(output_dir / 'manifest.json')
        return output, report
    except BaseException:
        if output is not None:
            output._mmap.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        for owner in owners:
            owner.close()
