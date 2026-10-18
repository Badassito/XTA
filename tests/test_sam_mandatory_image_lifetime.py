"""Guaranteed mandatory images keep exact pixels and bounded live ownership."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_cache_transactions import proof_runtime
from tests.test_sam_view_image_cache import context_for, crop_from, demand


def case(tmp_path, monkeypatch, *, scratch=4096, cap=63):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(scratch))
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(cap))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    source = np.arange(3*7*9, dtype=np.uint8).reshape(3, 7, 9)
    context = context_for(tmp_path, source)
    context._runtime = proof_runtime()
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    pool = _ByteAdmissionPool(64*resources.GIB, 'mandatory-images')
    return context, view, pool


@pytest.mark.parametrize('allowance', (0, 128, 4*resources.GIB))
@pytest.mark.parametrize('scratch', (4096, 65536))
def test_current_and_inline_retry_need_no_optional_credit_under_pressure(
        tmp_path, monkeypatch, allowance, scratch):
    context, view, pool = case(tmp_path, monkeypatch, scratch=scratch)
    shape = context.source_volume.shape
    original = demand(shape, {1:(1, 1, 5, 6)})
    expanded = demand(shape, {1:(0, 0, 7, 9)})
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=allowance, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            initial_charge = pool.in_use
            assert profile.assigned_image_bytes >= 2*63+scratch
            profile._lease.headroom_probe = lambda:pool.capacity
            with mock.patch('XTA.sam_image_prefetch._try_image_profile',
                    side_effect=AssertionError('mandatory work requested optional image credit')):
                with context.resource_scope(profile):
                    with context.image_cohort_provider(view, shape, original) as first:
                        with context.image_cohort_provider(view, shape, expanded) as second:
                            assert first.path != second.path and first.path.exists()
                            assert context._parent_image_payload_bytes(profile) == 20+63
                            assert pool.in_use == initial_charge
                            np.testing.assert_array_equal(crop_from(first, 1, (1, 1, 5, 6)),
                                                          context.source_volume[1, 1:5, 1:6])
                            np.testing.assert_array_equal(crop_from(second, 1, (0, 0, 7, 9)),
                                                          context.source_volume[1])
                        assert not second.path.exists() and first.path.exists()
                    assert not first.path.exists() and not context._cache_owners
                    assert not context._retained_parent_image_credits
                    assert profile._lease.scope_holds == 0
        assert resources.sam_parent_promised_bytes(pool) == 0
    finally:
        context.close()


def test_cross_parent_pixel_donation_cannot_extend_the_funding_parent_lifetime(tmp_path, monkeypatch):
    context, view, pool = case(tmp_path, monkeypatch)
    shape = context.source_volume.shape
    planned = demand(shape, {1:(1, 1, 5, 6)})
    first_ready, second_ready, first_gone = (threading.Event() for _ in range(3))
    paths, charges = {}, {}
    def parent(index):
        try:
            with resources.admit_sam_parent_resources(pool, 4*resources.GIB, f'parent-{index}',
                    base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                    headroom_probe=lambda:128*resources.GIB) as profile:
                charges[index] = profile._lease.charged_bytes
                with context.resource_scope(profile):
                    if index:
                        assert first_ready.wait(5)
                    with context.image_cohort_provider(view, shape, planned) as reference:
                        paths[index] = reference.path
                        if not index:
                            first_ready.set()
                            assert second_ready.wait(5)
                        else:
                            assert paths[0] != paths[1]
                            assert context.image_cache_reused_pixels == 20
                            np.testing.assert_array_equal(crop_from(reference, 1, (1, 1, 5, 6)),
                                                          context.source_volume[1, 1:5, 1:6])
                            second_ready.set()
                            assert first_gone.wait(5)
                            assert not paths[0].exists() and reference.path.exists()
                            assert pool.in_use == charges[1]
            if not index:
                first_gone.set()
        finally:
            # Do not strand a sibling if an assertion fails.
            if not index:
                first_ready.set()
                first_gone.set()
            else:
                second_ready.set()
    try:
        with ThreadPoolExecutor(2) as workers:
            jobs = [workers.submit(parent, index) for index in range(2)]
            for job in jobs:
                job.result(timeout=10)
        assert not any(path.exists() for path in paths.values())
        assert not context._retained_parent_image_credits and pool.in_use == 0
    finally:
        context.close()


@pytest.mark.parametrize('failure', ('retirement', 'cancelled'))
def test_failed_or_cancelled_cache_cannot_refund_an_unproven_parent(tmp_path, monkeypatch, failure):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1:(1, 1, 5, 6)})
    runtime = context._runtime
    runtime.release_source_cache = mock.Mock(side_effect=RuntimeError('unproven gray reader'))
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile):
                with pytest.raises(RuntimeError, match='unproven gray reader|cancelled scope'):
                    with context.image_cohort_provider(view, context.source_volume.shape, planned) as reference:
                        if failure == 'cancelled':
                            context._cancel.set()
                            raise RuntimeError('cancelled scope')
            assert reference.path.exists() and profile._lease.scope_holds == 1
        assert pool.in_use == profile._lease.charged_bytes
        assert not profile._lease.active and not profile._lease.credit_returned
        context.close()
        assert not reference.path.exists() and pool.in_use == 0
        assert profile._lease.credit_returned and not context._retained_parent_image_credits
    finally:
        context.close()


def test_borrowed_native_backing_keeps_its_original_owner(tmp_path, monkeypatch):
    context, view, pool = case(tmp_path, monkeypatch)
    path = tmp_path/'native.gray8.dat'
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=context.source_volume.shape)
    source[:] = context.source_volume
    source.flush()
    context.source_volume = source
    planned = demand(source.shape, {frame:(0, 0, 7, 9) for frame in range(3)})
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile):
                with context.image_cohort_provider(view, source.shape, planned) as reference:
                    assert reference.path == path.resolve()
                    assert profile._lease.scope_holds == 0
                    assert not context._retained_parent_image_credits
        assert pool.in_use == 0
        context.close()
        assert path.exists()
    finally:
        context.close()
        source._mmap.close()


def test_injected_gray_reader_cannot_exceed_declared_scratch_before_allocation(tmp_path, monkeypatch):
    context, view, pool = case(tmp_path, monkeypatch)
    planned = demand(context.source_volume.shape, {1:(1, 1, 5, 6)})
    context._resource_local.gray_cache_reader = SimpleNamespace(crop_workspace_bytes=lambda box:4097)
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            with context.resource_scope(profile), mock.patch('numpy.memmap') as allocate:
                with pytest.raises(resources.SamMandatoryImageAdmissionError, match='gray_renderer_scratch_bound'):
                    context._image_provider(view, context.source_volume.shape, planned)
                allocate.assert_not_called()
                assert not context._image_builds and not context._cache_owners
    finally:
        context._resource_local.gray_cache_reader = None
        context.close()


@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
@pytest.mark.parametrize('startup_failure', (False, True))
def test_initial_pass_uses_lexical_guarantee_and_preserves_it_through_failed_startup(
        tmp_path, monkeypatch, operation, startup_failure):
    from XTA import sam_interpolation, sam_extrapolation
    context, view, pool = case(tmp_path, monkeypatch, cap=1024)
    context._runtime = None
    observed = np.zeros_like(context.source_volume)
    observed[0, 2:4, 2:4] = observed[2, 2:4, 2:4] = 1
    module = sam_interpolation if operation == 'interpolation' else sam_extrapolation
    execute_name = ('interpolate_sam_view_volume_pass' if operation == 'interpolation' else
                    'extrapolate_sam_view_volume_pass')
    seen = []
    def start():
        assert context._retained_parent_image_credits
        if startup_failure:
            raise RuntimeError('controlled predictor startup failure')
        context._runtime = proof_runtime()
    def execute(volume, **kwargs):
        reference = kwargs['image_provider']
        seen.append(reference.path)
        assert reference.path.exists()
        assert context._cache_owners[str(reference.path)]['persistent'] is False
        return volume, {}, []
    monkeypatch.setattr(context, '_start', start)
    monkeypatch.setattr(module, execute_name, execute)
    kwargs = (dict(gap_distance=3, interpolation_walk_back=0) if operation == 'interpolation'
              else dict(distance=1, walk_back=0, min_radius=0.))
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            profile._lease.headroom_probe = lambda:pool.capacity
            with context.resource_scope(profile):
                call = getattr(context, 'interpolate' if operation == 'interpolation' else 'extrapolate')
                if startup_failure:
                    with pytest.raises(RuntimeError, match='controlled predictor startup failure'):
                        call(observed, view=view, scope='parent', **kwargs)
                    retained_paths = [owner['reference'].path for owner in context._cache_owners.values()]
                    assert retained_paths and all(path.exists() for path in retained_paths)
                else:
                    call(observed, view=view, scope='parent', **kwargs)
                    assert seen and all(not path.exists() for path in seen)
        if startup_failure:
            assert pool.in_use == profile._lease.charged_bytes
            context.close()
            assert all(not path.exists() for path in retained_paths)
        assert pool.in_use == 0 and not context._retained_parent_image_credits
    finally:
        context.close()


def test_valid_captured_gray_workspace_uses_guarantee_without_source_render(tmp_path, monkeypatch):
    from XTA.sam_gpu_rendering import live_image_sampling
    from tests.test_sam_yolo_gray_cache import install_cache
    context, view, pool = case(tmp_path, monkeypatch, scratch=256*1024**2)
    shape = context.source_volume.shape
    pixels = install_cache(context, view, shape)
    planned = demand(shape, {1:(1, 1, 5, 6)})
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=0, image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda:128*resources.GIB) as profile:
            profile._lease.headroom_probe = lambda:pool.capacity
            with context.resource_scope(profile), mock.patch.object(context, '_render_demand_crop',
                    side_effect=AssertionError('gray capture fell back to source rendering')):
                with context.image_cohort_provider(view, shape, planned) as reference:
                    assert live_image_sampling(reference)['backend'] == 'canonical_yolo_gray_cache'
                    np.testing.assert_array_equal(crop_from(reference, 1, (1, 1, 5, 6)), pixels[1, 1:5, 1:6])
        assert not reference.path.exists() and pool.in_use == 0
    finally:
        context.close()
