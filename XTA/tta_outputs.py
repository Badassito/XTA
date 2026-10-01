"""Single-owner teardown handoff for settled TTA output artifacts.

The global assembly, output scheduling, and manifest construction phases still live in the
pipeline because they actively reshape and cross-reference view/tile registries.  This module
establishes the prerequisite for their later extraction: once those phases settle, every live
array, registry, model, and source-volume handle is transferred by identity to one fail-closed
owner and retired in the exact pre-refactor order before complete-manifest publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def measure_bridge_output_survival(
    refs: Sequence[object],
    final_output: np.ndarray,
    *,
    memory_mib: float = 256,
    stage: str = 'final_output_after_global_postprocessing',
) -> list[dict[str, object]]:
    """Count selected SAM support surviving a downstream source-grid transaction.

    Pixel retention is measured separately from connection topology.  Counts do
    not certify that an endpoint remains connected, even if every addition was
    retained: a final policy may also have removed its original attachments.
    """
    from .outputs import _nrrd_layer_zero_skip_window
    from .reconciliation_runtime import RuntimeLayer

    refs = [ref for ref in refs if (getattr(ref, 'mask_kind', '') == 'bridge'
            and getattr(ref, 'interpolation_backend', '') == 'sam'
            and getattr(ref, 'layer_role', 'additive_component') == 'additive_component'
            and getattr(ref, 'recomposition_op', 'union') == 'union')]
    if not refs:
        return []
    final = np.asarray(final_output)
    if final.ndim != 3 or final.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise ValueError('Bridge survival requires a binary TYX final output')
    if not math.isfinite(float(memory_mib)) or float(memory_mib) <= 0:
        raise ValueError('Bridge survival memory_mib must be positive and finite')
    # The current source-grid readers return complete planes. Reject an
    # insufficient budget explicitly rather than allocate an unbounded slab.
    if final.shape[1] * final.shape[2] * 4 > int(float(memory_mib) * 1024**2):
        raise ValueError('Bridge survival memory budget must fit four source-grid planes')
    receipts = []
    seen = set()
    for ref in refs:
        if (getattr(ref, 'mask_kind', '') != 'bridge'
                or getattr(ref, 'interpolation_backend', '') != 'sam'
                or getattr(ref, 'layer_role', 'additive_component') != 'additive_component'
                or getattr(ref, 'recomposition_op', 'union') != 'union'):
            continue
        identity = (getattr(ref, 'model_name', ''), getattr(ref, 'key', ''))
        if identity in seen:
            continue
        seen.add(identity)
        if getattr(ref, 'proposal_selection_status', '') != 'policy_selected':
            raise ValueError('Bridge survival cannot audit an unselected SAM candidate')
        window = _nrrd_layer_zero_skip_window(ref, tuple(final.shape))
        start, stop = window if window is not None else (0, int(final.shape[0]))
        measured_start = time.perf_counter()
        selected, retained = 0, 0
        owner = None
        try:
            if stop > start:
                owner = RuntimeLayer(ref, final.shape)
                for z in range(start, stop):
                    mask = owner.read_slab(z, z + 1)
                    output = final[z:z + 1]
                    if np.any(output > 1):
                        raise ValueError('Bridge survival final output must be binary 0/1')
                    selected += int(np.count_nonzero(mask))
                    retained += int(np.count_nonzero(mask.astype(bool) & output.astype(bool)))
        finally:
            if owner is not None:
                owner.close()
        receipts.append(dict(
            model_name=identity[0], layer_key=identity[1],
            interpolation_backend='sam', interpolation_direction=getattr(ref, 'interpolation_direction', ''),
            stage=stage, proposal_selection_status='policy_selected',
            selected_bridge_connection_status=getattr(ref, 'selected_bridge_connection_status', ''),
            selected_voxels=selected, retained_voxels=retained,
            removed_voxels=selected - retained,
            component_payload_reads=int(stop > start), source_grid_planes_read=max(0, stop - start),
            support_measurement_seconds=time.perf_counter() - measured_start,
            support_survival='later_filtered' if retained < selected else 'support_preserved',
            connection_survival='not_assessed',
            connection_survival_reason='Voxel retention does not verify endpoint attachment or local connectivity.',
            sam_run_ids=list(getattr(ref, 'sam_run_ids', ())),
            interpolation_policy_identity=getattr(ref, 'interpolation_policy_identity', ''),
            gate_support_identity=getattr(ref, 'gate_support_identity', ''),
        ))
    topology_started = time.perf_counter()
    connections = measure_sam_final_connections(refs, final, memory_mib=memory_mib, stage=stage)
    topology_seconds = time.perf_counter() - topology_started
    for receipt in receipts:
        key = (str(receipt['model_name']), str(receipt['layer_key']))
        topology = connections.get(key)
        if topology is not None:
            receipt['group_connections'] = topology['groups']
            receipt['connection_survival'] = topology['status']
            receipt['connection_survival_reason'] = topology['reason']
            receipt['batch_connection_measurement_seconds'] = topology_seconds
    return receipts


def measure_sam_final_connections(
    refs: Sequence[object], final_output: np.ndarray, *, memory_mib: float = 256,
    stage: str = 'final_output_after_global_postprocessing',
) -> dict[tuple[str, str], dict[str, object]]:
    """Check local SAM repairs against the final source grid and fixed attachments.

    Only surviving selected additions and surviving original endpoint silhouettes
    can supply a path. Other final foreground, including a remote existing route,
    cannot certify the intended local repair. The initial supported mapping is
    native Transverse with an identity source transform.
    """
    from .sam_evidence import SamEvidenceBundle, load_sam_online_selection
    from .sam_mask_reader import effective_candidate_mask
    from scipy import ndimage

    final = np.asarray(final_output)
    if final.ndim != 3 or final.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise ValueError('Final SAM connection audit requires a binary TYX output')
    if not math.isfinite(float(memory_mib)) or float(memory_mib) <= 0:
        raise ValueError('Final SAM connection audit memory_mib must be positive and finite')
    budget = int(float(memory_mib) * 1024**2)
    scopes = {}
    results = {}
    for ref in refs:
        if getattr(ref, 'interpolation_backend', '') != 'sam' or getattr(ref, 'mask_kind', '') != 'bridge':
            continue
        if getattr(ref, 'proposal_selection_status', '') != 'policy_selected':
            raise ValueError('Final SAM connection audit requires completed proposal selection')
        identity = (str(getattr(ref, 'model_name', '')), str(getattr(ref, 'key', '')))
        path = str(getattr(ref, 'proposal_evidence_path', ''))
        runs = tuple(map(str, getattr(ref, 'sam_run_ids', ())))
        transform = getattr(ref, 'native_transform', {}) or {}
        native_shape = tuple(transform.get('native_shape_tyx', transform.get('shape_tyx', ())))
        source_shape = tuple(transform.get('source_shape_tyx', ()))
        view = str(transform.get('view_name', transform.get('view', getattr(ref, 'physical_view_name', '')))).lower()
        identity_transform = (transform.get('kind', 'identity') == 'identity'
                              and native_shape == tuple(final.shape) == source_shape
                              and view == 'transverse' and float(transform.get('angle_deg', 0)) == 0)
        if not path or not runs or not identity_transform:
            reason = ('No selected contributors in this directional slot.' if not runs else
                      'Retained proposal evidence is unavailable.' if not path else
                      'Final connection audit requires a verified identity Transverse source transform.')
            results[identity] = dict(status='not_assessed', reason=reason, groups=[])
            continue
        key = (str(Path(path).resolve()), str(getattr(ref, 'interpolation_policy_identity', '')),
               int(getattr(ref, 'interpolation_connectivity', 6)))
        scope = scopes.setdefault(key, dict(refs=[], runs=set()))
        scope['refs'].append((identity, ref))
        scope['runs'].update(runs)
    for (path, policy_identity, connectivity), scope in scopes.items():
        if connectivity not in (6, 18, 26):
            raise ValueError('SAM final topology connectivity must be 6, 18 or 26')
        bundle_path = Path(path)
        if bundle_path.is_file() and bundle_path.name == 'manifest.json':
            bundle_path = bundle_path.parent
        bundle = SamEvidenceBundle.open(bundle_path, max_mask_bytes=max(1, budget // 4))
        if not bundle.manifest.get('complete'):
            raise ValueError('Final SAM connection audit cannot certify incomplete proposal evidence')
        if scope['runs'] - set(bundle.runs):
            raise ValueError('Final SAM connection audit references missing proposal contributors')
        selection = load_sam_online_selection(bundle, policy_hash=policy_identity,
                                               selected_run_ids=scope['runs'])
        reader_cache_bytes = min(32 * 1024**2, max(0, budget // 8))
        with bundle.reader(max_cache_bytes=reader_cache_bytes) as reader:
            selection_view = {**(selection or {}), 'mask_filter': reader.filter_snapshot(selection)}
            group_runs = {}
            for run_id in sorted(scope['runs']):
                run = bundle.runs[run_id]
                if not run.get('complete'):
                    raise ValueError('Final SAM connection audit references an incomplete selected run')
                group_runs.setdefault(str(run['group_id']), []).append(run_id)
            group_receipts = {}
            structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
            for group_id, selected in sorted(group_runs.items()):
                group = bundle.groups[group_id]
                frames = tuple(map(int, group['frame_indices']))
                y0, x0, y1, x1 = map(int, group['context_bbox_yx'])
                shape = (len(frames), y1 - y0, x1 - x0)
                if (not frames or frames != tuple(range(frames[0], frames[-1] + 1))
                        or not 0 <= frames[0] <= frames[-1] < final.shape[0]
                        or not 0 <= y0 < y1 <= final.shape[1] or not 0 <= x0 < x1 <= final.shape[2]):
                    raise ValueError('Final SAM connection crop is outside the declared source geometry')
                if math.prod(shape) * 16 > budget - reader_cache_bytes:
                    group_receipts[group_id] = dict(group_id=group_id, stage=stage, status='not_assessed',
                        reason='Final connection topology exceeds the declared memory budget.', edges=[])
                    continue
                surviving = np.zeros(shape, bool)
                frame_indices = {frame: index for index, frame in enumerate(frames)}
                for run_id in selected:
                    for frame in bundle.runs[run_id]['observed_frames']:
                        frame = int(frame)
                        surviving[frame_indices[frame]] |= (effective_candidate_mask(reader, run_id, frame, selection_view)
                            & final[frame, y0:y1, x0:x1].astype(bool))
                endpoints = {str(value['observation_id']): value for value in group['endpoints']}
                edges = []
                for edge in group.get('edges', ()):
                    source, target = endpoints[str(edge['source_id'])], endpoints[str(edge['target_id'])]
                    source_frame, target_frame = int(source['frame_index']), int(target['frame_index'])
                    lo, hi = sorted((source_frame, target_frame))
                    z0, z1 = frame_indices[lo], frame_indices[hi] + 1
                    local = surviving[z0:z1].copy()
                    edge_additions = local.copy()
                    for frame in range(lo, hi + 1):
                        contract_key = f"edge_contract:{edge['edge_id']}:{frame}"
                        known_key = f'known_foreground:{frame}'
                        if contract_key in group['mask_keys']:
                            contract = reader.group_mask(group_id, contract_key)
                            local[frame - lo] &= contract
                            edge_additions[frame - lo] &= contract
                            if known_key in group['mask_keys']:
                                local[frame - lo] |= (reader.group_mask(group_id, known_key) & contract
                                    & final[frame, y0:y1, x0:x1].astype(bool))
                    attachments = []
                    for endpoint, frame in ((source, source_frame), (target, target_frame)):
                        original = reader.group_mask(group_id, f"endpoint:{endpoint['observation_id']}")
                        remaining = original & final[frame, y0:y1, x0:x1].astype(bool)
                        local[frame - lo] |= remaining
                        attachments.append((remaining, dict(observation_id=endpoint['observation_id'],
                            original_voxels=int(np.count_nonzero(original)),
                            retained_voxels=int(np.count_nonzero(remaining)))))
                    labels, _ = ndimage.label(local, structure=structure)
                    source_labels = set(map(int, np.unique(labels[source_frame - lo][attachments[0][0]]))) - {0}
                    target_labels = set(map(int, np.unique(labels[target_frame - lo][attachments[1][0]]))) - {0}
                    addition_labels = set(map(int, np.unique(labels[edge_additions]))) - {0}
                    connected = bool(source_labels & target_labels & addition_labels)
                    edges.append(dict(edge_id=edge['edge_id'], source_id=source['observation_id'],
                        target_id=target['observation_id'], connected=connected, local_native_interval=[lo, hi],
                        local_surviving_addition_voxels=int(np.count_nonzero(edge_additions)),
                        endpoint_survival=[item[1] for item in attachments]))
                connected = bool(edges) and all(edge['connected'] for edge in edges)
                group_receipts[group_id] = dict(group_id=group_id, stage=stage,
                    status='survived' if connected else 'connection_lost', edges=edges,
                    all_requested_edges_connected=connected, connectivity=connectivity,
                    selected_run_ids=selected, interpolation_policy_identity=policy_identity,
                    selected_mask_semantics=('receipt_controlled_component_filter' if selection and selection.get('mask_filter')
                                             else 'legacy_unfiltered_candidates'),
                    evidence_fingerprint=bundle.evidence_fingerprint,
                    attachment_contract='surviving_selected_additions_and_fixed_local_original_observation_masks')
        for group_receipt in group_receipts.values():
            group_receipt['mask_reader'] = dict(reader.stats)
        for identity, ref in scope['refs']:
            groups = sorted({str(bundle.runs[str(run_id)]['group_id'])
                             for run_id in getattr(ref, 'sam_run_ids', ())})
            receipts = [group_receipts[group_id] for group_id in groups]
            status = ('not_assessed' if any(item['status'] == 'not_assessed' for item in receipts) else
                      'survived' if receipts and all(item['status'] == 'survived' for item in receipts) else
                      'connection_lost')
            results[identity] = dict(status=status, groups=receipts,
                reason='Local connectivity evaluated after downstream filtering with fixed observed attachments.')
    return results


@dataclass(frozen=True)
class TtaOutputInputs:
    """Run-constant teardown policy."""

    keep_temp_artifacts: bool
    tile_slice_workers: int


@dataclass(frozen=True)
class TtaOutputOperations:
    """Injected teardown operations preserving pipeline monkeypatch seams."""

    close_memmap_array: Callable[[object], None]
    close_raw_store_or_memmap_volume: Callable[..., None]
    archive_or_delete_binary_volume_storage: Callable[..., None]
    unload_yolo_model: Callable[[object], None]
    trim_cuda_memory: Callable[[], None]
    collect_garbage: Callable[[], object]


@dataclass(frozen=True)
class TtaOutputResult:
    """Immutable confirmation of the completed ownership transaction."""

    close_memmap_calls: int
    retired_tile_accumulators: int
    unloaded_models: int
    processing_volume_was_distinct: bool


@dataclass
class TtaOutputArtifacts:
    """Identity-preserving owner of every artifact live at final output teardown."""

    final_output_mask_mm: Optional[np.ndarray]
    final_union_mm: Optional[np.ndarray]
    native_view_support_by_model: Dict[str, Dict[str, np.ndarray]]
    azimuthal_native_output_by_model: Dict[str, Dict[str, np.ndarray]]
    tilted_native_output_by_model: Dict[str, Dict[str, np.ndarray]]
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]]
    parent_mask_support_by_model: Dict[str, Dict[str, object]]
    parent_bridge_support_by_model: Dict[str, Dict[str, object]]
    tile_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    tile_parent_mask_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    tile_parent_bridge_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    baseline_union_by_model_view: Mapping[Tuple[str, str], np.ndarray]
    baseline_confmap_by_model_view: Mapping[
        Tuple[str, str], Optional[np.ndarray]
    ]
    yolo_models: Sequence[Tuple[str, Optional[object]]]
    volume_rgb: object
    input_volume_rgb: object
    _close_started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def close(
        self,
        *,
        inputs: TtaOutputInputs,
        operations: TtaOutputOperations,
    ) -> TtaOutputResult:
        """Retire all transferred artifacts exactly once in publication-safe order."""

        if self._close_started:
            raise RuntimeError("TTA output artifacts have already entered teardown")
        self._close_started = True

        close_memmap_calls = 0
        retired_tile_accumulators = 0
        unloaded_models = 0
        processing_volume_was_distinct = bool(
            self.volume_rgb is not self.input_volume_rgb
        )

        if self.final_output_mask_mm is not self.final_union_mm:
            operations.close_memmap_array(self.final_output_mask_mm)
            close_memmap_calls += 1
        operations.close_memmap_array(self.final_union_mm)
        close_memmap_calls += 1

        for model_support in self.native_view_support_by_model.values():
            for volume in model_support.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_support.clear()
        for model_views in self.azimuthal_native_output_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()
        for model_views in self.tilted_native_output_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()
        for model_views in self.view_volumes_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()

        for model_support in self.parent_mask_support_by_model.values():
            for support in model_support.values():
                operations.close_raw_store_or_memmap_volume(
                    support,
                    keep_temp=bool(inputs.keep_temp_artifacts),
                )
            model_support.clear()
        for model_support in self.parent_bridge_support_by_model.values():
            for support in model_support.values():
                operations.close_raw_store_or_memmap_volume(
                    support,
                    keep_temp=bool(inputs.keep_temp_artifacts),
                )
            model_support.clear()

        for accumulator in self.tile_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining consolidated tile accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_accumulator_by_set.clear()
        for accumulator in self.tile_parent_mask_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining parent-mask tile category accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_parent_mask_accumulator_by_set.clear()
        for accumulator in self.tile_parent_bridge_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining parent-bridge tile category accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_parent_bridge_accumulator_by_set.clear()

        for volume in self.baseline_union_by_model_view.values():
            operations.close_memmap_array(volume)
            close_memmap_calls += 1
        for volume in self.baseline_confmap_by_model_view.values():
            operations.close_memmap_array(volume)
            close_memmap_calls += 1
        for _model_name, model in self.yolo_models:
            if model is not None:
                operations.unload_yolo_model(model)
                unloaded_models += 1
        if processing_volume_was_distinct:
            operations.close_memmap_array(self.volume_rgb)
            close_memmap_calls += 1
        operations.close_memmap_array(self.input_volume_rgb)
        close_memmap_calls += 1
        operations.trim_cuda_memory()
        # A safe retirement request does not unmap while this owner still holds
        # an array. Drop the completed references before selected-run scratch
        # cleanup (and the complete-manifest publication callback) can begin.
        self.final_output_mask_mm = None
        self.final_union_mm = None
        if isinstance(self.baseline_union_by_model_view, dict):
            self.baseline_union_by_model_view.clear()
        if isinstance(self.baseline_confmap_by_model_view, dict):
            self.baseline_confmap_by_model_view.clear()
        self.baseline_union_by_model_view = {}
        self.baseline_confmap_by_model_view = {}
        self.yolo_models = ()
        self.volume_rgb = None
        self.input_volume_rgb = None
        operations.collect_garbage()

        self._closed = True
        return TtaOutputResult(
            close_memmap_calls=int(close_memmap_calls),
            retired_tile_accumulators=int(retired_tile_accumulators),
            unloaded_models=int(unloaded_models),
            processing_volume_was_distinct=bool(processing_volume_was_distinct),
        )


__all__ = [
    "measure_bridge_output_survival",
    "measure_sam_final_connections",
    "TtaOutputArtifacts",
    "TtaOutputInputs",
    "TtaOutputOperations",
    "TtaOutputResult",
]
