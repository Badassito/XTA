"""Controlled SAM proposals traverse production assembly and independent tile configurations."""
from __future__ import annotations

from pathlib import Path
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
    def __init__(self, root, selected):
        self.evidence_root = root
        self.detector_identity = 'detector-source'
        self.bundle_identity = 'sam-bundle'
        self.selected = selected
        self.calls = []

    def interpolate(self, observations, *, view, scope, work_dir, **kwargs):
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


def _prepare(tmp_path, observations, context, *, distance=5, backend='sam'):
    with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace'), \
         mock.patch.object(assembly, 'materialize_nrrd_view_layer', return_value=None), \
         mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
         mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                           side_effect=AssertionError('SAM invoked the SDF generator')):
        return assembly.prepare_view_volume_after_fullframe(
            model_name='detector', view=_view(observations.shape),
            union_mm=observations, confmap_mm=None,
            union_path=tmp_path / 'original.u8.dat', confmap_path=None,
            temp_dir=tmp_path, tiling_active=True, min_conf=0., min_radius=0.,
            interpolate=distance, interpolation_walk_back=0, interpolation_candidates=2,
            interpolate_passes=3, interpolate_min_radius=0., interpolation_search_angle=-6.,
            slice_workers=1, interpolation_task_workers=1,
            nrrd_layers_enabled=True, interpolation_backend=backend, sam_context=context,
        )


def test_sam_fullframe_keeps_selection_and_two_directional_slots_without_tile_support(tmp_path):
    observed = np.zeros((3, 7, 9), dtype=np.uint8)
    observed[0, 3, 4] = observed[2, 3, 4] = 1
    before = observed.copy()
    selected = np.zeros_like(observed)
    selected[1, 3, 4] = 1
    context = _SelectedContext(tmp_path / 'evidence', selected)
    result = _prepare(tmp_path, observed, context)
    assert len(context.calls) == 1
    scope, seed_snapshot, settings = context.calls[0]
    assert scope.endswith('/fullframe')
    np.testing.assert_array_equal(seed_snapshot, before)
    np.testing.assert_array_equal(observed, before)
    assert settings['interpolation_walk_back'] == 0
    assert settings['interpolation_candidates'] == 2
    assert settings['interpolation_passes'] == 3
    assert settings['search_angle_deg'] == -6.
    np.testing.assert_array_equal(result.final_view_volume_mm, before | selected)
    bridges = [ref for ref in result.nrrd_layers if ref.mask_kind == 'bridge']
    assert len(bridges) == 2
    assert {ref.interpolation_direction for ref in bridges} == {'forward', 'backward'}
    assert all(ref.interpolation_backend == 'sam' and ref.interpolation_walk_back_index == 0
               and ref.interpolation_candidate_index == 0 and ref.seed_detector_identity == 'detector-source'
               for ref in bridges)
    assert not (tmp_path / 'tile_support').exists()


def test_ungated_tile_observation_survives_and_only_config_seeds_sam(tmp_path):
    observed = np.zeros((3, 7, 9), dtype=np.uint8)
    observed[1, 3:5, 4:6] = 1
    added = np.zeros_like(observed)
    added[2, 4, 5] = 1
    context = _SelectedContext(tmp_path / 'tile-evidence', added)
    originals = []

    def publish_observed(volume, **kwargs):
        originals.append((volume.copy(), kwargs['tile_acceptance'], kwargs['stage']))

    with mock.patch.object(assembly, 'materialize_nrrd_view_layer', side_effect=publish_observed), \
            mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
            mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                              side_effect=AssertionError('SAM invoked SDF for consolidated tiles')):
        result = assembly.finalize_independent_tile_config(
            model_name='detector', view=_view(observed.shape), tile_accumulator_mm=observed,
            temp_dir=tmp_path, interpolate=5, interpolation_walk_back=0,
            interpolation_candidates=2, interpolate_passes=2, interpolate_min_radius=0.,
            interpolation_search_angle=-6., slice_workers=1, interpolation_task_workers=1,
            nrrd_layers_enabled=True, config_id='s4_st2', interpolation_backend='sam', sam_context=context)
    assert len(originals) == len(context.calls) == 1
    raw, acceptance, stage = originals[0]
    np.testing.assert_array_equal(raw, observed)
    assert (acceptance, stage) == ('independent', 's4_st2_pre_tile_interpolation')
    scope, tile_observations, settings = context.calls[0]
    assert scope.endswith('/tile/s4_st2')
    np.testing.assert_array_equal(tile_observations, observed)
    assert settings['upstream_lineage'] == dict(seed_detector_identity='detector-source',
                                              tile_config_id='s4_st2', tile_acceptance='independent')
    np.testing.assert_array_equal(result.final_accumulator_mm, observed | added)
    bridges = [ref for ref in result.nrrd_layers if ref.mask_kind == 'bridge']
    assert {ref.interpolation_direction for ref in bridges} == {'forward', 'backward'}
    assert all(ref.tile_config_id == 's4_st2' and not ref.gate_support_identity for ref in bridges)


def test_distance_zero_never_initializes_configured_sam(tmp_path):
    observed = np.ones((2, 3, 4), dtype=np.uint8)
    context = mock.Mock()
    result = _prepare(tmp_path, observed, context, distance=0)
    context.interpolate.assert_not_called()
    assert not result.interpolation_stats and not (tmp_path / 'tile_support').exists()


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
