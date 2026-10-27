"""Intensity/categorical QSC plans share owners without changing coordinates."""
from contextlib import nullcontext
import os
from unittest import mock

import numpy as np
import pytest

from XTA import pta_cuda_shells as shells, spherical_cuda as sc
from tests.test_spherical_cuda import resident_engine, spherical_view, rotation_xyz


@pytest.mark.parametrize('face', range(6))
@pytest.mark.parametrize('rotated', (False, True))
def test_exact_fp64_plan_reuses_native_owners_without_host_build_or_upload(face, rotated):
    engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
    view = spherical_view(face=face, size=19, intervals=16, origin=(-1, -2),
                          rotation=rotation_xyz() if rotated else None)
    key = sc.spherical_direction_key(view, engine.device)
    [(row, directions, valid)] = list(sc._direction_blocks(engine, key))
    before = shells.shell_direction_cache_stats(engine)
    with mock.patch.object(sc, '_build_host_direction_block', side_effect=AssertionError('duplicate QSC')):
        actual = shells._spherical_directions(view, engine.device, None, direction_renderer=engine)
    assert row == 0 and actual[0] is directions and actual[1] is valid
    expected_dirs, expected_valid = sc._build_host_direction_block(key, 0, view.src_h)
    np.testing.assert_array_equal(actual[0].numpy(), expected_dirs)
    np.testing.assert_array_equal(actual[1].numpy(), expected_valid)
    after = shells.shell_direction_cache_stats(engine)
    assert after['shared_hits'] == before['shared_hits'] + 1
    assert after['h2d_copies'] == before['h2d_copies']
    assert after['intensity_upload_calls'] == before['intensity_upload_calls'] == 2


def test_radius_independent_borrow_does_not_touch_native_lru_or_retain_evicted_entry():
    engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
    first = spherical_view()
    second = spherical_view(face=1)
    key = sc.spherical_direction_key(first, engine.device)
    list(sc._direction_blocks(engine, key))
    list(sc._direction_blocks(engine, sc.spherical_direction_key(second, engine.device)))
    order = list(engine._spherical_direction_cache)
    borrowed = sc.cached_spherical_directions(engine, key)
    assert borrowed is not None
    assert list(engine._spherical_direction_cache) == order
    sc.clear_spherical_render_cache(engine)
    assert not engine._spherical_direction_cache
    # The consumer's temporary references remain usable after producer eviction.
    assert borrowed[0].shape == (7, 7, 3)
    assert sc.cached_spherical_directions(engine, key) is None


def test_fp32_strips_and_different_geometry_are_never_borrowed_as_fp64():
    engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
    view = spherical_view()
    key = sc.spherical_direction_key(view, engine.device)
    list(sc._direction_blocks(engine, key, dtype=np.float32))
    assert sc.cached_spherical_directions(engine, key) is None
    sc.clear_spherical_render_cache(engine)
    with mock.patch.object(sc, '_DIRECTION_CACHE_BYTES', 7 * 25 * 2):
        list(sc._direction_blocks(engine, key))
    assert sc.cached_spherical_directions(engine, key) is None
    list(sc._direction_blocks(engine, key))
    for index, changed in ((0, 'cuda:1'), (1, 1), (2, 7), (3, 8), (4, 8), (5, -1), (6, -1),
                           (7, tuple(rotation_xyz().reshape(-1)))):
        other = list(key)
        other[index] = changed
        assert sc.cached_spherical_directions(engine, tuple(other)) is None
    entry = engine._spherical_direction_cache[key + ('<f8',)]
    engine._spherical_direction_cache[key + ('<f8',)] = (entry[0].float(), entry[1], entry[2])
    assert sc.cached_spherical_directions(engine, key) is None


def test_shared_miss_keeps_existing_categorical_cache_and_exact_fp64_builder():
    import torch
    shells.clear_shell_direction_cache()
    engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
    view = spherical_view(size=70, intervals=66, origin=(-1, -2))
    key = sc.spherical_direction_key(view, engine.device)
    expected_dirs, expected_valid = sc._build_host_direction_block(key, 0, view.src_h)
    try:
        with mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()):
            first = shells._spherical_directions(view, engine.device, None, direction_renderer=engine)
        with mock.patch.object(sc, '_build_host_direction_block', side_effect=AssertionError('duplicate QSC')):
            second = shells._spherical_directions(view, engine.device, None, direction_renderer=engine)
        assert second[0] is first[0] and second[1] is first[1]
        np.testing.assert_array_equal(first[0].numpy(), expected_dirs)
        np.testing.assert_array_equal(first[1].numpy(), expected_valid)
    finally:
        shells.clear_shell_direction_cache()


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_SPHERICAL_SHARING') != '1', reason='explicit GPU qualification')
@pytest.mark.parametrize('face', range(6))
@pytest.mark.parametrize('tilted', (False, True))
@pytest.mark.parametrize('tile', (False, True))
def test_cuda_shared_directions_match_categorical_oracle_across_streams_and_eviction(face, tilted, tile):
    import torch
    pytest.importorskip('cupy')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import cv2
    from XTA.spherical_geometry import render_shell_frame
    from tests.test_pta_cuda_shells import _spherical, _sources
    mask, coverage = _sources()
    view = _spherical(face, tilted=tilted)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    engine = resident_engine(mask)
    engine.device = torch.device('cuda:0')
    engine._stream = producer
    engine._volume_gpu = torch.as_tensor(mask, device=engine.device)
    key = sc.spherical_direction_key(view, engine.device)
    with torch.cuda.stream(producer):
        list(sc._direction_blocks(engine, key))
    forward = np.array([[2. / 3., 0, 2], [0, 2. / 3., -1]], np.float32) if tile else np.eye(2, 3, dtype=np.float32)
    expected = [cv2.warpAffine(render_shell_frame(source, view, 0, categorical=True), forward,
                              (19, 17), flags=cv2.INTER_NEAREST) for source in (mask, coverage)]
    before = shells.shell_direction_cache_stats()
    with mock.patch.object(sc, '_build_host_direction_block', side_effect=AssertionError('duplicate QSC')):
        actual = shells.render_shell_categorical_pair(
            torch.as_tensor(mask, device='cuda'), torch.as_tensor(coverage, device='cuda'),
            view, 0, M_src_to_out=forward, out_h=17, out_w=19,
            stream=consumer, direction_renderer=engine,
        )
    sc.clear_spherical_render_cache(engine)
    with torch.cuda.stream(producer):
        [torch.empty((17, 19, 3), dtype=torch.float64, device='cuda').fill_(123) for _ in range(4)]
    consumer.synchronize()
    for output, reference in zip(actual, expected):
        np.testing.assert_array_equal(output.cpu().numpy(), reference)
    after = shells.shell_direction_cache_stats()
    assert after['shared_hits'] == before['shared_hits'] + 1
    assert after['h2d_copies'] == before['h2d_copies']
