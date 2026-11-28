"""The full-frame scientific branch runs without producing tile-gate support."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation


def _prepare(tmp_path, *, backend='sdf', distance=0, passes=1, augmentation_pass=0,
             initial=None, cleanup=False, extrapolation=0, sam_context=None):
    shape = (3, 7, 9)
    path = tmp_path / 'native.dat'
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    source[:] = 0
    source[0, 2:4, 3:6] = 1
    if initial is not None:
        source[:] = initial
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    view = replace(view, augmentation_pass=augmentation_pass)
    result = assembly.prepare_view_volume_after_fullframe(
        model_name='detector', view=view, union_mm=source, confmap_mm=None,
        union_path=path, confmap_path=None, temp_dir=tmp_path,
        tiling_active=True, min_conf=0., min_radius=0.,
        interpolate=distance, interpolation_walk_back=0, interpolation_candidates=1,
        interpolate_passes=passes, interpolate_min_radius=0., interpolation_search_angle=0.,
        slice_workers=1, interpolation_task_workers=1,
        nrrd_layers_enabled=False, interpolation_backend=backend,
        precleaned_slice_cleanup=not cleanup, hole_fill_done_on_device=not cleanup,
        extrapolation_distance=extrapolation,
        sam_context=sam_context)
    return result


@pytest.mark.parametrize('backend', ['sdf', 'sam'])
@pytest.mark.parametrize('distance,passes,augmentation_pass', [(0, 1, 0), (5, 0, 0), (5, 1, 1)])
def test_inactive_pass_allocates_no_fullframe_tile_gate(tmp_path, backend, distance, passes, augmentation_pass):
    expected = np.zeros((3, 7, 9), np.uint8)
    expected[0, 2:4, 3:6] = 1
    result = _prepare(tmp_path, backend=backend, distance=distance, passes=passes,
                      augmentation_pass=augmentation_pass)
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    assert not result.interpolation_stats and not (tmp_path / 'tile_support').exists()


@pytest.mark.parametrize('added', [False, True])
def test_fullframe_interpolation_keeps_its_additions_without_gate_snapshot(tmp_path, added):
    def interpolate(*, mask_mm, **_kwargs):
        if added:
            mask_mm[1, 3, 4] = 1
        return mask_mm, {'added_voxels': int(added)}

    with mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process', side_effect=interpolate):
        result = _prepare(tmp_path, distance=5)
    expected = np.zeros((3, 7, 9), np.uint8)
    expected[0, 2:4, 3:6] = 1
    expected[1, 3, 4] = int(added)
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    assert result.interpolation_stats[0]['added_voxels'] == int(added)
    assert not (tmp_path / 'tile_support').exists()


def test_real_cpu_hole_fill_preserves_fullframe_observations_without_tile_gate(tmp_path):
    from XTA import workspace
    initial = np.zeros((3, 7, 9), dtype=np.uint8)
    initial[0, 1:6, 2:7] = 1
    initial[0, 2:5, 3:6] = 0
    expected = initial.copy()
    expected[0, 2:5, 3:6] = 1
    with mock.patch.object(workspace, '_YOLO_HOLE_FILL_ACTIVE', True):
        result = _prepare(tmp_path, initial=initial, cleanup=True)
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    assert not (tmp_path / 'tile_support').exists()


def test_extrapolation_adds_to_fullframe_after_frozen_seed_baseline(tmp_path):
    tail = np.zeros((3, 7, 9), dtype=np.uint8)
    tail[2, 3, 4] = 1
    tail_path = tmp_path / 'tail.cvol'
    interpolation.write_raw_bbox_mask_store(tail, tail_path,
        format_name=interpolation.CVOL_FORMAT, desc='Controlled SAM tail', workers=1)
    observed = []
    def extrapolate(native_volume, **kwargs):
        observed.append(np.asarray(native_volume).copy())
        return native_volume, {'added_voxels': 1}, [{'path': str(tail_path)}]
    context = SimpleNamespace(evidence_root=tmp_path / 'evidence', extrapolate=extrapolate)
    result = _prepare(tmp_path/'actual', backend='sam', extrapolation=2, sam_context=context)
    assert len(observed) == 1 and not observed[0][2].any()
    np.testing.assert_array_equal(result.final_view_volume_mm, observed[0] | tail)
    assert [row['processing_role'] for row in result.interpolation_stats] == ['extrapolation']
    assert not (tmp_path/'actual'/'tile_support').exists()
