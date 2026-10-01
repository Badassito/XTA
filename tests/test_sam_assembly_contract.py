"""Controlled SAM proposals traverse production assembly and ordinary tile gates."""
from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation


def _view(shape):
    return geometry.ViewInfo(
        name='transverse__tta_a0', physical_view_name='transverse',
        summary_family='transverse__tta_a0', tta_aug_id='a0',
        num_slices=shape[0], src_h=shape[1], src_w=shape[2],
        full_t=shape[0], full_h=shape[1], full_w=shape[2],
        pad_mode='clamp', family='orthogonal', tta_angle_deg=0.,
    )


class _SelectedContext:
    def __init__(self, root, selected, *, ready_order=None):
        self.evidence_root = root
        self.detector_identity = 'detector-source'
        self.bundle_identity = 'sam-bundle'
        self.selected = selected
        self.calls = []
        self.ready_order = ready_order

    def interpolate(self, observations, *, view, scope, work_dir, **kwargs):
        if self.ready_order is not None:
            assert self.ready_order == ['original_parent_ready']
        self.calls.append((scope, observations.copy(), kwargs))
        # Exhausted later passes return the original observations. Assembly must
        # retain the first completed pass, not call again and erase its repairs.
        if len(self.calls) > 1 and self.calls[-2][0] == scope:
            return observations, dict(skipped=True, added_voxels=0), []
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for direction in ('forward', 'backward'):
            mask = self.selected if direction == 'forward' else np.zeros_like(self.selected)
            path = work_dir / f'{direction}.cvol'
            interpolation.write_raw_bbox_mask_store(
                mask, path, format_name=interpolation.CVOL_FORMAT, workers=1,
            )
            entries.append(dict(direction=direction, path=str(path),
                                storage_format=interpolation.CVOL_FORMAT,
                                voxel_count=int(mask.sum()), policy_hash='controlled-policy',
                                evidence_path=str(work_dir / 'controlled-evidence')))
        merged = np.asarray(observations != 0, dtype=np.uint8) | self.selected
        return merged, dict(skipped=False, added_voxels=int(self.selected.sum()),
                            sam_policy_hash='controlled-policy',
                            sam_evidence_path=str(work_dir / 'controlled-evidence'),
                            requested_passes=kwargs['interpolation_passes'],
                            completed_passes=1, skipped_passes=kwargs['interpolation_passes'] - 1), entries


def _prepare(tmp_path, observations, context, *, distance=5, backend='sam', callback=None):
    with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace'), \
         mock.patch.object(assembly, 'materialize_nrrd_view_layer', return_value=None), \
         mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
         mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                           side_effect=AssertionError('SAM invoked the SDF generator')):
        return assembly.prepare_view_volume_after_fullframe(
            model_name='detector', view=_view(observations.shape),
            union_mm=observations, confmap_mm=None,
            union_path=tmp_path / 'original.u8.dat', confmap_path=None,
            temp_dir=tmp_path, dense_tiling_active=True, min_conf=0., min_radius=0.,
            interpolate=distance, interpolation_walk_back=0, interpolation_candidates=2,
            interpolate_passes=3, interpolate_min_radius=0., interpolation_search_angle=-6.,
            keep_temp=True, slice_workers=1, interpolation_task_workers=1,
            nrrd_layers_enabled=True, interpolation_backend=backend, sam_context=context,
            parent_mask_ready_callback=callback,
        )


def _decode(store):
    return np.stack([store.decode_slice(index) for index in range(store.shape[0])])


def test_sam_fullframe_uses_selected_parent_support_and_two_slots_with_zero_walkback(tmp_path):
    observed = np.zeros((3, 7, 9), dtype=np.uint8)
    observed[0, 3, 4] = observed[2, 3, 4] = 1
    before = observed.copy()
    selected = np.zeros_like(observed)
    selected[1, 3, 4] = 1
    order = []
    context = _SelectedContext(tmp_path / 'evidence', selected, ready_order=order)

    def parent_ready(_model, _view_name, support):
        assert np.array_equal(_decode(support), before)
        order.append('original_parent_ready')

    result = _prepare(tmp_path, observed, context, callback=parent_ready)
    try:
        assert len(context.calls) == 1
        scope, seed_snapshot, settings = context.calls[0]
        assert scope.endswith('/fullframe')
        assert np.array_equal(seed_snapshot, before)
        assert np.array_equal(observed, before)
        assert settings['interpolation_walk_back'] == 0
        assert settings['interpolation_candidates'] == 2
        assert settings['interpolation_passes'] == 3
        assert settings['search_angle_deg'] == -6.
        assert np.array_equal(result.final_view_volume_mm, before | selected)
        assert np.array_equal(_decode(result.parent_mask_support_mm), before)
        assert np.array_equal(_decode(result.parent_bridge_support_mm), selected)
        bridges = [ref for ref in result.nrrd_layers if ref.mask_kind == 'bridge']
        assert len(bridges) == 2
        assert {ref.interpolation_direction for ref in bridges} == {'forward', 'backward'}
        assert all(ref.interpolation_backend == 'sam' for ref in bridges)
        assert all(ref.interpolation_walk_back_index == 0 for ref in bridges)
        assert all(ref.interpolation_candidate_index == 0 for ref in bridges)
        assert all(ref.seed_detector_identity == 'detector-source' for ref in bridges)
        assert result.parent_bridge_support_mm.meta['proposal_selection_status'] == 'policy_selected'
        assert result.parent_bridge_support_mm.meta['gate_support_identity']
    finally:
        result.parent_mask_support_mm.close()
        result.parent_bridge_support_mm.close()


def test_bridge_rescued_detector_component_reaches_consolidated_sam_with_gate_lineage(tmp_path):
    shape = (3, 7, 9)
    selected = np.zeros(shape, dtype=np.uint8)
    selected[1, 3, 4] = 1
    observed = np.zeros(shape, dtype=np.uint8)
    observed[0, 3, 4] = observed[2, 3, 4] = 1
    parent_context = _SelectedContext(tmp_path / 'parent-evidence', selected)
    parent_result = _prepare(tmp_path / 'parent', observed, parent_context)
    accepted = np.zeros(shape, dtype=np.uint8)
    category = np.zeros(shape, dtype=np.uint8)
    residual = np.zeros(shape, dtype=np.uint8)
    residual[1, 3:5, 4:6] = 1
    try:
        gate = assembly.gate_tile_components_against_support_inplace(
            residual, parent_result.parent_bridge_support_mm,
            parent_crop=(0, 7, 0, 9), accepted_total_mm=accepted,
            accepted_category_mm=category, retain_rejected_components=False,
        )
        assert gate['accepted_voxels'] == 4
        assert np.array_equal(accepted, category)
        destination = observed | selected
        added = np.zeros(shape, dtype=np.uint8)
        added[2, 4, 5] = 1
        context = _SelectedContext(tmp_path / 'tile-evidence', added)
        identity = parent_result.parent_bridge_support_mm.meta['gate_support_identity']
        lineage = dict(gate_support_identity=identity,
                       interpolation_policy_identity='controlled-policy',
                       gate_support_fingerprints={'parent': identity})
        with mock.patch.object(assembly, 'materialize_nrrd_view_layer', return_value=None), \
             mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
             mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                               side_effect=AssertionError('SAM invoked SDF for consolidated tiles')):
            result = assembly.finalize_consolidated_tile_volume_for_parent(
                model_name='detector', view=_view(shape), tile_accumulator_mm=accepted,
                destination_mm=destination, destination_lock=threading.Lock(),
                temp_dir=tmp_path / 'tile', interpolate=5, interpolation_walk_back=0,
                interpolation_candidates=2, interpolate_passes=3,
                interpolate_min_radius=0., interpolation_search_angle=0., keep_temp=True,
                slice_workers=1, interpolation_task_workers=1, nrrd_layers_enabled=True,
                tile_parent_bridge_accumulator_mm=category, config_id='s4_st2',
                interpolation_backend='sam', sam_context=context, sam_upstream_lineage=lineage,
            )
        assert len(context.calls) == 1
        scope, tile_observations, settings = context.calls[0]
        assert scope.endswith('/tile/s4_st2')
        assert np.array_equal(tile_observations, category)
        assert settings['upstream_lineage'] == lineage
        assert np.array_equal(result.final_accumulator_mm, category | added)
        assert destination[2, 4, 5] == 1
        bridges = [ref for ref in result.nrrd_layers if ref.mask_kind == 'bridge']
        assert len(bridges) == 2
        assert all(ref.tile_config_id == 's4_st2' for ref in bridges)
        assert all(ref.gate_support_identity == identity for ref in bridges)
    finally:
        parent_result.parent_mask_support_mm.close()
        parent_result.parent_bridge_support_mm.close()


def test_distance_zero_never_initializes_configured_sam(tmp_path):
    observed = np.ones((2, 3, 4), dtype=np.uint8)
    context = mock.Mock()
    result = _prepare(tmp_path, observed, context, distance=0)
    try:
        context.interpolate.assert_not_called()
        assert not result.interpolation_stats
        assert result.parent_bridge_support_mm is None
    finally:
        result.parent_mask_support_mm.close()


def test_sam_generation_failure_propagates_instead_of_empty_success(tmp_path):
    observed = np.ones((2, 3, 4), dtype=np.uint8)
    context = mock.Mock(evidence_root=tmp_path / 'evidence')
    context.interpolate.side_effect = RuntimeError('SAM worker failed')
    with pytest.raises(RuntimeError, match='SAM worker failed'):
        _prepare(tmp_path, observed, context)


def test_expanded_transverse_runtime_view_reaches_real_generator(tmp_path):
    from XTA import sam_interpolation

    class RepeatedSeedRuntime:
        def run(self, **request):
            frames = range(request['frame_start'], request['frame_stop'])
            return SimpleNamespace(
                frames={frame: request['seed_mask'].copy() for frame in frames},
                tracker_scores={}, observation_status={},
                receipt={'prediction_valid': True, 'coverage_complete': True},
            )

    observed = np.zeros((3, 7, 9), dtype=np.uint8)
    observed[0, 3, 4] = observed[2, 3, 4] = 1
    physical = geometry.ViewInfo(name='transverse', num_slices=3,
                                 src_h=7, src_w=9, family='orthogonal', pad_mode='clamp')
    expanded = geometry.expand_views_into_tta_variants([physical], [0.])[0]
    assert expanded.summary_family != 'transverse'
    with mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                           side_effect=AssertionError('SAM invoked SDF')):
        merged, stats, slots = sam_interpolation.interpolate_sam_view_volume_pass(
            observed, view=expanded, work_dir=tmp_path, runtime=RepeatedSeedRuntime(),
            gap_distance=3, min_radius=0., interpolation_walk_back=0,
            return_bridge_components=True,
        )
    try:
        assert stats['sam_selected_runs'] > 0
        assert stats['added_voxels'] == 1
        assert merged[1, 3, 4] == 1
        assert len(slots) == 2
    finally:
        merged._mmap.close()
