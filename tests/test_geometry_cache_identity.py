"""Pixel caches bind a live owner; geometry caches bind their consumed recipe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
from pathlib import Path
import threading
from unittest import mock
import warnings
import weakref

import numpy as np
import pytest

from XTA import backprojection as bp
from XTA import geometry as g
from XTA.config import TiltedViewGroup


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def clear_coronal():
    with g._CORONAL_BLOCK_CACHE_LOCK:
        assert not g._CORONAL_BLOCK_BUILDS_IN_FLIGHT
        g._CORONAL_BLOCK_CACHE.clear()


def clear_tilted():
    with g._TILTED_RENDER_PLAN_CACHE_LOCK:
        assert not g._TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT
        g._TILTED_RENDER_PLAN_CACHE.clear()
        g._TILTED_RENDER_PLAN_CACHE_BYTES = 0


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CORONAL_BLOCK_COLS', '8')
    monkeypatch.setenv('YOLO_TTA_CORONAL_BLOCK_CACHE', '2')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    clear_coronal()
    clear_tilted()
    bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
    yield
    clear_coronal()
    clear_tilted()
    bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()


def coronal_view(shape):
    return g.get_view_infos(*shape, cartesian_views=('coronal',),
                            azimuthal_views=(), azimuthal_azimuth_angles=())[0]


def tilted_view(shape, base='transverse', direction='vertical', sign=1):
    views = g.get_view_infos(*shape, cartesian_views=(), azimuthal_views=(),
        azimuthal_azimuth_angles=(), tilt_groups=(TiltedViewGroup(
            views=(base,), tilt_angles=(20.0,), tilt_directions=(direction,)),))
    return next(v for v in views if sign * v.tilt_angle_deg > 0)


def test_coronal_public_dtype_alias_both_orders():
    volume = np.full((8, 9, 10), 200, np.uint8)
    signed = volume.view(np.int8)
    view = coronal_view(volume.shape)
    for sources in ((volume, signed), (signed, volume)):
        clear_coronal()
        for source in sources:
            actual = g.get_view_frame_by_index(source, view, 1)
            assert actual.dtype == source.dtype
            np.testing.assert_array_equal(actual, source[:, :, 1])
            assert g.get_view_frame_by_index(source, view, 1).base is actual.base


def test_coronal_source_collection_removes_entry_and_keeps_returned_frame():
    source = np.full((8, 9, 10), 7, np.uint8)
    source_ref = weakref.ref(source)
    frame = g.get_view_frame_by_index(source, coronal_view(source.shape), 1)
    assert len(g._CORONAL_BLOCK_CACHE) == 1
    del source
    gc.collect()
    assert source_ref() is None
    assert not g._CORONAL_BLOCK_CACHE
    np.testing.assert_array_equal(frame, np.full((8, 9), 7, np.uint8))
    # A consecutive same-shaped volume must not inherit any former pixels,
    # regardless of whether this allocator happens to reuse the old address.
    replacement = np.full((8, 9, 10), 9, np.uint8)
    actual = g.get_view_frame_by_index(replacement, coronal_view(replacement.shape), 1)
    np.testing.assert_array_equal(actual, replacement[:, :, 1])


def test_coronal_stale_collection_callback_cannot_delete_replacement():
    source = np.ones((2, 3, 10), np.uint8)
    g._coronal_frame_from_block_cache(source, 1)
    key, original = next(iter(g._CORONAL_BLOCK_CACHE.items()))
    replacement = np.zeros_like(source)
    newer = g._CoronalBlockCacheEntry(weakref.ref(replacement), np.zeros((8, 2, 3), np.uint8))
    # Deterministically simulate a delayed callback after identity reuse.
    g._CORONAL_BLOCK_CACHE[key] = newer
    g._remove_coronal_block_for_collected_source(key, original.source_ref)
    assert g._CORONAL_BLOCK_CACHE[key] is newer
    g._CORONAL_BLOCK_CACHE.clear()


def test_coronal_captures_block_width_once_per_read():
    shape = (2, 3, 24)
    source = np.broadcast_to(np.arange(24, dtype=np.uint8), shape).copy()
    view = coronal_view(shape)
    with mock.patch.object(g, 'coronal_block_cols', side_effect=(8, 16)) as width:
        first = g.get_view_frame_by_index(source, view, 9)
        second = g.get_view_frame_by_index(source, view, 17)
    assert width.call_count == 2
    np.testing.assert_array_equal(first, source[:, :, 9])
    np.testing.assert_array_equal(second, source[:, :, 17])


def test_coronal_same_owner_layout_change_does_not_reuse_old_block():
    source = np.arange(8 ** 3, dtype=np.uint16).reshape(8, 8, 8).copy()
    g._coronal_frame_from_block_cache(source, 1)
    original_key = g._coronal_block_cache_key(source, 0)
    with warnings.catch_warnings(action='ignore', category=DeprecationWarning):
        source.strides = (source.strides[0], source.strides[2], source.strides[1])
    assert g._coronal_block_cache_key(source, 0) != original_key
    actual = g._coronal_frame_from_block_cache(source, 1)
    np.testing.assert_array_equal(actual, source[:, :, 1])


def test_coronal_lru_eviction_keeps_frame_alive(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CORONAL_BLOCK_CACHE', '1')
    source = np.arange(2 * 3 * 24, dtype=np.uint8).reshape(2, 3, 24)
    view = coronal_view(source.shape)
    retained = g.get_view_frame_by_index(source, view, 1)
    expected = source[:, :, 1].copy()
    g.get_view_frame_by_index(source, view, 17)
    assert len(g._CORONAL_BLOCK_CACHE) == 1
    np.testing.assert_array_equal(retained, expected)


@pytest.mark.parametrize('fail_first', [False, True])
def test_coronal_concurrent_build_and_failure_wakeup(fail_first):
    source = np.arange(2 * 3 * 10, dtype=np.uint8).reshape(2, 3, 10)
    started, release = threading.Event(), threading.Event()
    real_build = g._build_coronal_block
    calls = 0
    calls_lock = threading.Lock()

    def build(volume, first, stop):
        nonlocal calls
        with calls_lock:
            calls += 1
            attempt = calls
        if attempt == 1:
            started.set()
            assert release.wait(5)
            if fail_first:
                raise RuntimeError('injected block build failure')
        return real_build(volume, first, stop)

    with mock.patch.object(g, '_build_coronal_block', side_effect=build):
        with ThreadPoolExecutor(max_workers=6) as pool:
            first = pool.submit(g._coronal_frame_from_block_cache, source, 1)
            assert started.wait(5)
            others = [pool.submit(g._coronal_frame_from_block_cache, source, 1) for _ in range(5)]
            release.set()
            if fail_first:
                with pytest.raises(RuntimeError, match='injected'):
                    first.result(timeout=5)
            else:
                np.testing.assert_array_equal(first.result(timeout=5), source[:, :, 1])
            for future in others:
                np.testing.assert_array_equal(future.result(timeout=5), source[:, :, 1])
    assert calls == (2 if fail_first else 1)
    assert not g._CORONAL_BLOCK_BUILDS_IN_FLIGHT


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('sign', [1, -1])
def test_tilted_public_changed_shape_both_orders(base, direction, sign):
    # Change only the stack domain, retaining the same native in-plane raster.
    larger = {'transverse': (7, 6, 7), 'sagittal': (5, 8, 7), 'coronal': (5, 6, 9)}
    shapes = ((5, 6, 7), larger[base])
    views = [tilted_view(shape, base, direction, sign) for shape in shapes]
    volumes = [np.ones(shape, np.uint8) for shape in shapes]
    expected = []
    for volume, view in zip(volumes, views):
        clear_tilted()
        expected.append([g.get_view_frame_by_index(volume, view, frame)
                         for frame in range(view.num_slices)])
    for order in ((0, 1), (1, 0)):
        clear_tilted()
        for index in order:
            view, volume = views[index], volumes[index]
            for frame, wanted in enumerate(expected[index]):
                np.testing.assert_array_equal(g.get_view_frame_by_index(volume, view, frame), wanted)
    assert g._tilted_plan_cache_key(views[0], IDENTITY, 6, 7) != g._tilted_plan_cache_key(views[1], IDENTITY, 6, 7)


def test_tilted_half_pixel_affines_both_orders_and_exact_effective_reuse():
    view = tilted_view((5, 6, 7))
    mask = np.broadcast_to(np.arange(7, dtype=np.uint8) % 2, (5, 6, 7)).copy()
    matrices = [IDENTITY.copy(), IDENTITY.copy()]
    matrices[0][0, 2], matrices[1][0, 2] = np.float32(.4999998), np.float32(.5000002)
    expected = []
    for matrix in matrices:
        clear_tilted()
        expected.append(g.render_tilted_categorical_frame_on_grid(mask, view, 2, matrix, 6, 7))
    assert np.count_nonzero(expected[0] != expected[1]) == 24
    for order in ((0, 1), (1, 0)):
        clear_tilted()
        for index in order:
            matrix = matrices[index]
            actual = g.render_tilted_categorical_frame_on_grid(mask, view, 2, matrix, 6, 7)
            np.testing.assert_array_equal(actual, expected[index])
            plan = g.get_tilted_render_plan(view, matrix, 6, 7)
            assert g.get_tilted_render_plan(view, matrix.astype(np.float64), 6, 7) is plan
            assert g.get_tilted_render_plan(replace(view, display_name='cosmetic'), matrix, 6, 7) is plan
    assert not g._TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT


def test_tilted_lru_and_concurrent_failed_build_cleanup(monkeypatch):
    view = tilted_view((5, 6, 7))
    real_build = g._build_tilted_render_plan
    started, release = threading.Event(), threading.Event()
    calls = 0
    lock = threading.Lock()

    def build(*args):
        nonlocal calls
        with lock:
            calls += 1
            attempt = calls
        if attempt == 1:
            started.set()
            assert release.wait(5)
            raise RuntimeError('injected tilted build failure')
        return real_build(*args)

    with mock.patch.object(g, '_build_tilted_render_plan', side_effect=build):
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(g.get_tilted_render_plan, view, IDENTITY, 6, 7)
            assert started.wait(5)
            others = [pool.submit(g.get_tilted_render_plan, view, IDENTITY, 6, 7) for _ in range(3)]
            release.set()
            with pytest.raises(RuntimeError, match='injected'):
                first.result(timeout=5)
            plans = [future.result(timeout=5) for future in others]
    assert calls == 2 and all(plan is plans[0] for plan in plans)
    assert not g._TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT
    monkeypatch.setenv('YOLO_TTA_TILTED_PLAN_CACHE_GIB', '0')
    changed = IDENTITY.copy()
    changed[0, 2] = .5
    newest = g.get_tilted_render_plan(view, changed, 6, 7)
    assert list(g._TILTED_RENDER_PLAN_CACHE.values()) == [newest]
    assert g._TILTED_RENDER_PLAN_CACHE_BYTES == g._tilted_render_plan_nbytes(newest)


def dense_pair():
    # Use the builder's actual float32 theta on this NumPy/platform. The legal
    # spacing requests straddle the boundary between source planes 1 and 2.
    yy, xx = np.indices((9, 9), dtype=np.float32)
    theta = np.mod(np.degrees(np.arctan2(yy - 4., xx - 4.)).astype(np.float32), 180.).astype(np.float32)
    tie = float(theta[5, 8]) / 1.5
    views = [g.get_view_infos(9, 9, 9, cartesian_views=(), azimuthal_views=('transverse',),
             azimuthal_azimuth_angles=(tie + delta,))[0] for delta in (-1e-9, 1e-9)]
    plans = [bp.build_azimuthal_backprojection_plan(view)[0] for view in views]
    return views, plans


def dense_sink(view, masks):
    output = np.zeros((9, 9, 9), np.uint8)
    def consume(first, block):
        output[int(first):int(first) + block.shape[0]] = block
    result = bp.backproject_azimuthal_volume_to_volume(masks, view,
        Path('unused_sink_only.dat'), 'cache identity test', workers=1, reserve_bytes=0,
        projection_block_callback=consume, sink_only=True)
    assert tuple(result.shape) == output.shape
    return output


def test_dense_exact_angular_owner_public_backprojection_both_orders():
    views, plans = dense_pair()
    assert bp._azimuthal_plan_signature(plans[0]) != bp._azimuthal_plan_signature(plans[1])
    assert views[0].num_slices == views[1].num_slices
    masks = np.broadcast_to((np.arange(views[0].num_slices) % 2).astype(np.uint8)[:, None, None],
                            (views[0].num_slices, views[0].src_h, views[0].src_w)).copy()
    expected = []
    for view in views:
        bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
        expected.append(dense_sink(view, masks))
    assert np.count_nonzero(expected[0] != expected[1]) >= 9
    for order in ((0, 1), (1, 0)):
        bp._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE.clear()
        for index in order:
            np.testing.assert_array_equal(dense_sink(views[index], masks), expected[index])
            dense = bp.build_dense_azimuthal_backprojection_map(views[index], plans[index])
            assert bp.build_dense_azimuthal_backprojection_map(views[index], plans[index]) is dense


def test_dense_signature_binds_source_and_reversal():
    _, plans = dense_pair()
    original = plans[0]
    changed_source = [replace(original[0], source_index=1), *original[1:]]
    changed_reverse = [replace(original[0], reverse_u=True), *original[1:]]
    assert bp._azimuthal_plan_signature(original) != bp._azimuthal_plan_signature(changed_source)
    assert bp._azimuthal_plan_signature(original) != bp._azimuthal_plan_signature(changed_reverse)
