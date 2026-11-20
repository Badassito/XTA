"""Every intensity cube builder and the low-quality gray downbin share one center-aligned T grid."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import media, outputs
from XTA.yolo_cpu_input import _t_map


def _in_memory(**kwargs):
    return np.empty(kwargs['shape'], dtype=kwargs['dtype'])


@pytest.fixture(autouse=True)
def _fresh_streaming_state():
    media.wait_for_streaming_producers(timeout=30.0)
    media.reset_streaming_state_for_new_run()
    yield
    media.wait_for_streaming_producers(timeout=30.0)


def _native_renderer_reference(per_slice, out_t):
    """The CPU native renderer's center-aligned two-tap lerp over already-resized slices."""
    lo, hi, alpha = _t_map(per_slice.shape[0], out_t)
    first = per_slice[lo].astype(np.float32)
    second = per_slice[hi].astype(np.float32)
    weights = alpha.reshape((-1,) + (1,) * (per_slice.ndim - 1))
    return np.rint(first + weights * (second - first)).clip(0., 255.).astype(np.uint8)


def _difference(a, b):
    return int(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max())


@pytest.mark.parametrize('in_t,out_t', [(5, 28), (7, 9), (30, 31), (12, 40)])
def test_streaming_t_only_cube_matches_the_slab_and_native_renderer(in_t, out_t):
    source = np.random.default_rng(in_t * 101 + out_t).integers(0, 256, (in_t, 6, 7), dtype=np.uint8)
    out_shape = (out_t, 6, 7)
    with mock.patch.object(media, 'allocate_workspace_array', side_effect=_in_memory):
        slab = np.array(media.resize_volume_to_processing_cube_gray8(
            source, out_shape, Path('slab.uint8'), workers=1))
        streamed = media.resize_volume_to_processing_cube_gray8_streaming(
            source, out_shape, Path('streamed.uint8'), workers=1)
        media.wait_for_volume_ready(streamed)
    streamed = np.array(streamed)
    # OpenCV's fixed-point slab and the producer's addWeighted round differently by
    # at most one level; an endpoint T grid differs by tens of levels on these inputs.
    assert _difference(streamed, slab) <= 1
    assert _difference(streamed, _native_renderer_reference(source, out_t)) <= 1


@pytest.mark.parametrize('streaming', [False, True])
def test_xy_resizing_cube_takes_the_native_renderer_t_taps(streaming):
    # Constant slices make the XY resize exact, so each output row exposes its T taps.
    levels = np.asarray([0, 60, 120, 180, 240], dtype=np.uint8)
    source = np.broadcast_to(levels[:, None, None], (5, 6, 30)).copy()
    out_shape = (29, 29, 30)
    resize = (media.resize_volume_to_processing_cube_gray8_streaming if streaming
              else media.resize_volume_to_processing_cube_gray8)
    with mock.patch.object(media, 'allocate_workspace_array', side_effect=_in_memory):
        cube = resize(source, out_shape, Path('cube.uint8'), workers=1)
        media.wait_for_volume_ready(cube)
    rows = np.array(cube)[:, 0, 0]
    expected = _native_renderer_reference(levels, out_shape[0])
    assert _difference(rows, expected) <= 1
    assert np.all(np.array(cube) == rows[:, None, None])


@pytest.mark.parametrize('source_shape,target_shape', [((50, 8, 8), (17, 4, 4)), ((31, 6, 6), (11, 6, 6))])
def test_low_quality_gray_samples_the_mask_box_center(source_shape, target_shape):
    in_t, out_t = source_shape[0], target_shape[0]
    step = 255 // (in_t - 1)
    source = np.broadcast_to(
        (np.arange(in_t, dtype=np.uint8) * np.uint8(step))[:, None, None], source_shape,
    ).copy()
    with mock.patch.object(outputs, 'allocate_workspace_array', side_effect=_in_memory):
        gray = np.array(outputs.resize_gray_volume_to_shape(
            source, target_shape, Path('lq_gray.uint8'), workers=1))
    position = gray[:, 0, 0].astype(np.float64) / step
    coords = np.arange(out_t, dtype=np.float64)
    # resize_binary_mask_volume_to_shape ORs [floor(z*in/out), ceil((z+1)*in/out)),
    # a box whose center in index units is (z+0.5)*in/out - 0.5.
    box_center = (coords + 0.5) * in_t / out_t - 0.5
    assert float(np.abs(position - box_center).max()) <= 0.5 / step + 1e-9


def test_concurrent_cold_callers_build_each_remap_table_once(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(media, '_REMAP_TABLES', {})
    real_indices = media._center_aligned_nearest_source_indices
    calls = []

    def slow_indices(in_count, out_count):
        calls.append((in_count, out_count))
        time.sleep(0.02)  # widen the window in which other threads miss the cache
        return real_indices(in_count, out_count)

    monkeypatch.setattr(media, '_center_aligned_nearest_source_indices', slow_indices)
    plane = (np.arange(12 * 10).reshape(12, 10) % 3 == 0).astype(np.uint8)
    expected = plane.take(real_indices(12, 19), axis=0).take(real_indices(10, 17), axis=1)
    barrier = threading.Barrier(8)
    results = [None] * 8

    def restore(index):
        barrier.wait()
        results[index] = media.resize_binary_plane_to_output_xy(plane, 19, 17)

    threads = [threading.Thread(target=restore, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 2  # one build: one column table and one row table
    for result in results:
        np.testing.assert_array_equal(result, expected)
