"""Explicit native conversion budgets include geometry and gathered numeric work."""
from __future__ import annotations

from dataclasses import asdict
import gc
import tracemalloc

import numpy as np
import pytest

from XTA import backprojection, geometry
from XTA.config import TiltedViewGroup
from XTA.confidence_evidence import write_block_confidence_evidence
from XTA.confidence_projection import score_projection_reader, score_projection_workspace


def _evidence(root, view, source_shape, values):
    return write_block_confidence_evidence(
        root, values.shape, lambda z: values[z], model_name='m', layer_key=view.name,
        coordinate_space='native_view_processing', source_shape_tyx=source_shape,
        provenance={'view': asdict(view)})


def test_real_conversion_cold_warm_and_distinct_geometries_fit_three_mib(tmp_path):
    cache = backprojection._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE
    original = cache.copy()
    cache.clear()
    try:
        for number, shape in enumerate(((3, 512, 512), (3, 512, 512), (3, 511, 509), (7, 320, 384))):
            view = geometry.get_view_infos(*shape, cartesian_views=(), azimuthal_views=('transverse',),
                azimuthal_azimuth_angles=(45.,), azimuthal_native_raster=0)[0]
            values = np.full((view.num_slices, view.src_h, view.src_w), 173, np.uint8)
            values[0, :, :view.src_w//3] = 0
            reference = _evidence(tmp_path/f'evidence{number}', view, shape, values)
            gc.collect()
            tracemalloc.start()
            try:
                with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=5) as reader:
                    actual, known = reader(0, 1)
                    assert actual.shape == (1, *shape[1:])
                    assert np.array_equal(known, actual > 0)
                    assert set(np.unique(actual)) == {0, 173}
                    del actual, known
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert peak < 3*1024**2, (number, peak)
            assert not cache
            assert not list((tmp_path/'work').iterdir())
            with pytest.raises(RuntimeError, match='closed'):
                reader(0, 1)
    finally:
        cache.clear()
        cache.update(original)


def _geometry_cases():
    views = geometry.get_view_infos(5, 7, 9,
        cartesian_views=('transverse', 'sagittal', 'coronal'),
        tilt_groups=(TiltedViewGroup(('transverse', 'sagittal', 'coronal'), (23.,), ('vertical', 'horizontal')),),
        azimuthal_views=('transverse', 'sagittal', 'coronal', 'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
        azimuthal_azimuth_angles=(45.,)*6, azimuthal_native_raster=0,
        radial_views=('transverse', 'sagittal', 'coronal', 'tilted_transverse'),
        radial_min_radius=.7, radial_patch_size=5,
        spherical_views=('transverse', 'tilted_transverse'), spherical_min_radius=.7, spherical_patch_size=5)
    selected, seen = [], set()
    for view in views:
        key = view.name if view.family not in ('radial', 'spherical') else (
            view.family, view.radial_base_view, view.radial_tilted_source,
            view.tilt_direction, view.spherical_face)
        if key not in seen:
            selected.append(view)
            seen.add(key)
    return selected


@pytest.mark.parametrize('view', _geometry_cases(), ids=lambda view: view.name)
@pytest.mark.parametrize('reduced', [False, True], ids=['native', 'reduced'])
def test_bounded_seven_pixel_strips_match_existing_projection(tmp_path, view, reduced):
    rng = np.random.default_rng(127)
    native_shape = (view.num_slices, 3, 3) if reduced else (view.num_slices, view.src_h, view.src_w)
    values = rng.choice(np.array([0, 0, 0, 32, 173, 229], np.uint8), size=native_shape)
    original = values.copy()
    for shape in ((4, 6, 8), (8, 10, 12)):
        with score_projection_reader(values, view, shape, tmp_path/'oracle') as read:
            expected = np.stack([read(z) for z in range(shape[0])])
        cache_keys = set(backprojection._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE)
        plan = score_projection_workspace(values.shape, view, shape, 16*1024**2)
        budget = plan.fixed_bytes + 7*plan.bytes_per_point
        with score_projection_reader(values, view, shape, tmp_path/'bounded', memory_bytes=budget) as read:
            actual = np.stack([read(z) for z in range(shape[0])])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values, original)
        assert set(backprojection._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE) == cache_keys
        assert not list((tmp_path/'bounded').iterdir())


def test_projection_minimum_rejection_precedes_native_staging(tmp_path):
    shape = (3, 512, 512)
    view = geometry.get_view_infos(*shape, cartesian_views=(), azimuthal_views=('transverse',),
        azimuthal_azimuth_angles=(45.,), azimuthal_native_raster=0)[0]
    values = np.full((view.num_slices, view.src_h, view.src_w), 173, np.uint8)
    reference = _evidence(tmp_path/'evidence', view, shape, values)
    plan = score_projection_workspace(values.shape, view, shape, 3*1024**2)
    too_small = 2*(plan.fixed_bytes+plan.bytes_per_point-1)/1024**2
    with pytest.raises(MemoryError, match='azimuthal projection needs at least'):
        with reference.source_reader(tmp_path/'work', memory_mib=too_small, max_staging_mib=5):
            pytest.fail('Impossible projection was admitted')
    assert not (tmp_path/'work').exists()


def test_projection_exception_retires_temporary_maps(tmp_path, monkeypatch):
    from XTA import confidence_projection
    shape = (5, 7, 9)
    view = geometry.get_view_infos(*shape, cartesian_views=(),
        tilt_groups=(TiltedViewGroup(('transverse',), (23.,), ('vertical',)),))[0]
    values = np.full((view.num_slices, view.src_h, view.src_w), 173, np.uint8)
    reference = _evidence(tmp_path/'evidence', view, shape, values)
    def fail(*args, **kwargs):
        raise OSError('injected strip failure')
    monkeypatch.setattr(confidence_projection, '_scatter_score_strip', fail)
    with pytest.raises(OSError, match='injected strip failure'):
        with reference.source_reader(tmp_path/'work', memory_mib=4, max_staging_mib=5):
            pass
    assert not list((tmp_path/'work').iterdir())


def test_slab_admission_accounts_for_known_output_and_preserves_reader(tmp_path):
    shape = (5, 512, 512)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    values = np.full(shape, 173, np.uint8)
    reference = _evidence(tmp_path/'evidence', view, shape, values)
    with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=5) as reader:
        with pytest.raises(MemoryError, match='slab'):
            reader(0, 5)
        scores, known = reader(0, 1)
        np.testing.assert_array_equal(scores, values[:1])
        assert known.all()


@pytest.mark.parametrize('output_hw', [(1048576, 1), (1, 1048576)], ids=['tall', 'wide'])
def test_real_narrow_bbox_cold_and_warm_fit_three_mib(tmp_path, output_hw):
    view = geometry.get_view_infos(1, 1, 1, cartesian_views=('transverse',))[0]
    values = np.full((1, 1, 1), 173, np.uint8)
    reference = _evidence(tmp_path/'evidence', view, (1, *output_hw), values)
    for attempt in range(2):
        gc.collect()
        tracemalloc.start()
        try:
            with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=4) as reader:
                crop = reader.read_crop(0)
                assert crop[:4] == (0, output_hw[0], 0, output_hw[1])
                assert crop[4].shape == output_hw
                assert crop[4].flags.owndata and crop[4].flags.c_contiguous
                assert np.all(crop[4] == 173)
                del crop
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 3*1024**2, (output_hw, attempt, peak)
        assert not list((tmp_path/'work').iterdir())


@pytest.mark.parametrize('transpose', [False, True], ids=['tall', 'wide'])
def test_narrow_bbox_preserves_unknowns_and_owns_only_its_returned_crop(tmp_path, transpose):
    native = np.array([0, 0, 32, 0, 173, 0, 0], np.uint8).reshape(1, 7, 1)
    output = (1, 70000, 1)
    if transpose:
        native = native.transpose(0, 2, 1).copy()
        output = (1, 1, 70000)
    view = geometry.get_view_infos(*native.shape, cartesian_views=('transverse',))[0]
    reference = _evidence(tmp_path/'evidence', view, output, native)
    with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=4) as reader:
        crop = reader.read_crop(0)
        assert crop[:4] == ((0, 1, 20000, 50000) if transpose else (20000, 50000, 0, 1))
        expected = np.repeat([32, 0, 173], 10000).astype(np.uint8)
        np.testing.assert_array_equal(crop[4].reshape(-1), expected)
        crop[4].fill(249)
        again = reader.read_crop(0)
        np.testing.assert_array_equal(again[4].reshape(-1), expected)
        assert not np.shares_memory(crop[4], again[4])
        with pytest.raises(IndexError):
            reader.read_crop(1)
    with pytest.raises(RuntimeError, match='closed'):
        reader.read_crop(0)


def test_unknown_bbox_returns_none_without_allocating_a_plane(tmp_path):
    view = geometry.get_view_infos(1, 1, 1, cartesian_views=('transverse',))[0]
    values = np.zeros((1, 1, 1), np.uint8)
    reference = _evidence(tmp_path/'evidence', view, (1, 1048576, 1), values)
    with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=4) as reader:
        assert reader.read_crop(0) is None
    assert not (tmp_path/'work').exists()


@pytest.mark.parametrize('transpose', [False, True], ids=['tall', 'wide'])
def test_projected_unknown_bbox_returns_none_for_nonempty_native_evidence(tmp_path, transpose):
    # Mixed resize axes select nearest XY samples. The only known sample is
    # outside that footprint, so the projected plane must have no bbox.
    values = np.array([0, 173], np.uint8).reshape(1, 1, 2)
    output = (1, 1048576, 1)
    if transpose:
        values = values.transpose(0, 2, 1).copy()
        output = (1, 1, 1048576)
    view = geometry.get_view_infos(*values.shape, cartesian_views=('transverse',))[0]
    reference = _evidence(tmp_path/'evidence', view, output, values)
    assert reference.metadata['known_voxels'] == 1
    with reference.source_reader(tmp_path/'work', memory_mib=3, max_staging_mib=4) as reader:
        assert reader.read_crop(0) is None
    assert not list((tmp_path/'work').iterdir())
