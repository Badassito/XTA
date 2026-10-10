"""Native shell futures are ordered, scratch-bounded and settled before exit."""
from dataclasses import replace
import gc
import threading
import time
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import geometry
from XTA.sam_canvas_rendering import (iter_prefetched_native_planes,
    native_shell_workspace_bytes, render_canonical_crop)
from XTA.sam_view_geometry import sam_native_transform_record


def setup(family='radial'):
    source = np.random.default_rng(155285).integers(0, 256, (13, 17, 19), np.uint8)
    views = geometry.get_view_infos(*source.shape, cartesian_views=(), radial_views=('transverse',),
        radial_patch_size=8, radial_min_radius=1., spherical_views=('transverse',), spherical_patch_size=8)
    return source, next(view for view in views if view.family == family)


def budget(view, jobs=2, remap=8192):
    return jobs*native_shell_workspace_bytes(view)+view.src_h*view.src_w+remap


@pytest.fixture(autouse=True)
def cpu_budget(monkeypatch):
    monkeypatch.setattr('XTA.workspace._cpu_count', lambda: 4)


@pytest.mark.parametrize('family', ('radial', 'spherical'))
def test_native_prefetch_and_cached_canonical_remap_are_exact(family):
    source, view = setup(family)
    frames = (0, view.num_slices//2, view.num_slices-1)
    shape = (view.num_slices, 37, 37)
    transform = sam_native_transform_record(view, shape, source.shape)
    affine = np.asarray(transform['M_native_to_canvas'], np.float32)
    inverse = np.asarray(transform['M_canvas_to_native'], np.float32)
    iterator = iter_prefetched_native_planes(source, view, frames,
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=2)
    try:
        observed = []
        for frame, plane, remaining in iterator:
            expected = geometry.render_intensity_frame_on_grid(source, view, frame,
                M_src_to_out=affine, M_out_to_src=inverse, output_height=37, output_width=37)
            actual = render_canonical_crop(source, view, frame, affine=affine, inverse=inverse,
                output_origin_yx=(5, 7), output_height=11, output_width=13, output_canvas_width=37,
                native_frame_cache={'plane': plane, 'origin_xy': (0, 0)}, max_workspace_bytes=remaining)
            np.testing.assert_array_equal(actual, expected[5:16, 7:20])
            observed.append(frame)
        assert observed == list(frames)
    finally:
        iterator.close()


@pytest.mark.parametrize('family', ('radial', 'spherical'))
@pytest.mark.parametrize('reverse', (False, True))
def test_native_prefetch_preserves_strided_and_reversed_sources(family, reverse):
    _source, view = setup(family)
    backing = np.random.default_rng(155286).integers(0, 256, (26, 17, 38), np.uint8)
    source = backing[::2, :, ::2]
    if reverse:
        source = source[::-1, ::-1, ::-1]
    assert source.shape == (13, 17, 19) and not source.flags.c_contiguous
    frames = (0, view.num_slices-1)
    iterator = iter_prefetched_native_planes(source, view, frames,
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=2)
    try:
        for frame, plane, _remaining in iterator:
            np.testing.assert_array_equal(plane, geometry.get_view_frame_by_index(source, view, frame))
    finally:
        iterator.close()


def test_pending_window_and_output_order_are_bounded():
    source, view = setup()
    release, second_done = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = peak = 0
    started, results, errors = [], [], []
    def getter(_source, _view, frame):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(frame)
        try:
            if frame == 0:
                assert release.wait(5)
            if frame == 1:
                second_done.set()
            return np.full((8, 8), frame, np.uint8)
        finally:
            with lock:
                active -= 1
    iterator = iter_prefetched_native_planes(source, view, range(4),
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=8)
    def consume():
        try:
            for frame, plane, remaining in iterator:
                results.append((frame, int(plane[0, 0]), remaining))
        except BaseException as error:
            errors.append(error)
    with mock.patch.object(geometry, 'get_view_frame_by_index', side_effect=getter):
        owner = threading.Thread(target=consume)
        owner.start()
        try:
            assert second_done.wait(5)
            assert sorted(started) == [0, 1] and not results
        finally:
            release.set()
            owner.join(5)
            iterator.close()
    assert not owner.is_alive() and not errors and active == 0 and peak == 2
    assert [(frame, value) for frame, value, _remaining in results] == [(i, i) for i in range(4)]
    assert all(remaining == 8192 for _frame, _value, remaining in results)


def test_early_close_waits_for_running_native_frame_and_drops_ready_results():
    source, view = setup()
    running, release, closed = threading.Event(), threading.Event(), threading.Event()
    references = []
    def getter(_source, _view, frame):
        if frame == 1:
            running.set()
            assert release.wait(5)
        plane = np.full((8, 8), frame, np.uint8)
        references.append(weakref.ref(plane))
        return plane
    iterator = iter_prefetched_native_planes(source, view, range(4),
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=2)
    with mock.patch.object(geometry, 'get_view_frame_by_index', side_effect=getter):
        first = next(iterator)
        assert running.wait(5)
        closer = threading.Thread(target=lambda: (iterator.close(), closed.set()))
        closer.start()
        try:
            assert not closed.wait(.05)
        finally:
            release.set()
            closer.join(5)
    assert closed.is_set() and not closer.is_alive()
    del first
    gc.collect()
    assert len(references) == 2 and all(reference() is None for reference in references)


def test_native_error_joins_other_work_and_clears_exception_array_aliases():
    source, view = setup()
    running, release, returned = threading.Event(), threading.Event(), threading.Event()
    references, errors = [], []
    def getter(_source, _view, frame):
        if frame == 0:
            assert running.wait(5)
            extra = np.zeros((300, 300), np.uint8)
            references.append(weakref.ref(extra))
            raise RuntimeError('native failure')
        running.set()
        assert release.wait(5)
        plane = np.full((8, 8), frame, np.uint8)
        references.append(weakref.ref(plane))
        return plane
    iterator = iter_prefetched_native_planes(source, view, range(4),
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=2)
    def consume():
        try:
            next(iterator)
        except BaseException as error:
            errors.append(error)
        finally:
            returned.set()
    with mock.patch.object(geometry, 'get_view_frame_by_index', side_effect=getter):
        owner = threading.Thread(target=consume)
        owner.start()
        try:
            assert running.wait(5) and not returned.wait(.05)
        finally:
            release.set()
            owner.join(5)
            iterator.close()
    assert returned.is_set() and len(errors) == 1 and 'native failure' in str(errors[0])
    gc.collect()
    assert all(reference() is None for reference in references), 'retained error cannot retain uncharged native arrays'


def test_callback_cancels_after_native_render_without_publishing_plane():
    source, view = setup()
    cancelled = threading.Event()
    references = []
    def check():
        if cancelled.is_set():
            raise RuntimeError('producer cancelled')
    def getter(_source, _view, _frame):
        plane = np.zeros((8, 8), np.uint8)
        references.append(weakref.ref(plane))
        cancelled.set()
        return plane
    iterator = iter_prefetched_native_planes(source, view, (0,),
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, check_cancel=check)
    with mock.patch.object(geometry, 'get_view_frame_by_index', side_effect=getter):
        with pytest.raises(RuntimeError, match='producer cancelled'):
            next(iterator)
    iterator.close()
    gc.collect()
    assert references[0]() is None


def test_sampling_guard_precedes_native_render():
    source, view = setup()
    iterator = iter_prefetched_native_planes(source, view, (0,),
        max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192)
    with mock.patch.object(geometry, 'require_forward_sampling', side_effect=RuntimeError('unapproved')), \
            mock.patch.object(geometry, 'get_view_frame_by_index') as getter:
        with pytest.raises(RuntimeError, match='unapproved'):
            next(iterator)
        getter.assert_not_called()
    iterator.close()


def test_two_active_iterators_make_independent_progress():
    source, view = setup()
    blocked, release, other_done = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def getter(_source, _view, frame):
        if frame < 2:
            blocked.set()
            assert release.wait(5)
        return np.full((8, 8), frame, np.uint8)
    def consume(frames, done=None):
        iterator = iter_prefetched_native_planes(source, view, frames,
            max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192, max_workers=2)
        try:
            for _frame, _plane, _remaining in iterator:
                pass
            if done is not None:
                done.set()
        except BaseException as error:
            errors.append(error)
        finally:
            iterator.close()
    with mock.patch.object(geometry, 'get_view_frame_by_index', side_effect=getter):
        first = threading.Thread(target=consume, args=((0, 1),))
        second = threading.Thread(target=consume, args=((2, 3), other_done))
        first.start()
        try:
            assert blocked.wait(5)
            second.start()
            assert other_done.wait(5), 'independent cohort must not share the blocked executor'
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
    assert not first.is_alive() and not second.is_alive() and not errors


def test_budget_and_ready_source_boundaries_precede_any_native_allocation():
    source, view = setup()
    assert native_shell_workspace_bytes(view) == 8*8+1024*8*8
    invalids = ((source.astype(np.float32), view, budget(view)),
                (source, replace(view, family='tilted'), budget(view)),
                (source, view, native_shell_workspace_bytes(view)))
    with mock.patch.object(geometry, 'get_view_frame_by_index') as getter:
        for array, descriptor, allowance in invalids:
            iterator = iter_prefetched_native_planes(array, descriptor, (0,),
                max_workspace_bytes=allowance, min_remap_workspace_bytes=8192)
            with pytest.raises(ValueError):
                next(iterator)
            iterator.close()
        readiness = SimpleNamespace(_all_event=threading.Event(), _exception=None)
        with mock.patch('XTA.media.volume_readiness', return_value=readiness):
            iterator = iter_prefetched_native_planes(source, view, (0,),
                max_workspace_bytes=budget(view), min_remap_workspace_bytes=8192)
            with pytest.raises(ValueError, match='cannot wait'):
                next(iterator)
            iterator.close()
        getter.assert_not_called()


def test_explicit_remap_cap_is_forwarded_and_enforced():
    source, view = setup()
    native = geometry.get_view_frame_by_index(source, view, 0)
    with pytest.raises(RuntimeError, match='bounded rendering memory budget'):
        render_canonical_crop(source, view, 0, affine=np.eye(2, 3), inverse=np.eye(2, 3),
            output_origin_yx=(0, 0), output_height=4, output_width=4,
            native_frame_cache={'plane': native, 'origin_xy': (0, 0)}, max_workspace_bytes=1)
