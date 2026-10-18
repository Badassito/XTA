"""Excess image grants survive consumers and uncertain mappings, without CUDA."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_cache_transactions import proof_runtime
from tests.test_sam_view_image_cache import context_for, crop_from, demand


GIB = resources.GIB


@pytest.fixture
def images(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '4096')
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    source = np.arange(3*7*9, dtype=np.uint8).reshape(3, 7, 9)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {1:(1, 1, 5, 6)})
    pool = _ByteAdmissionPool(64*GIB, 'current-images')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent',
            base_allowance_bytes=256, headroom_probe=lambda:128*GIB) as parent:
        context._runtime = proof_runtime()
        try:
            with context.resource_scope(parent):
                yield context, view, planned, pool, parent, 4096+20
        finally:
            context.close()
    assert resources.sam_parent_promised_bytes(pool) == 0


def staged(pool):
    return resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes']


def test_fresh_grant_funds_build_and_restores_parent_before_tracking(images, monkeypatch):
    context, view, planned, pool, parent, needed = images
    original = context._render_demand_crop
    builders = []
    def render(*args, **kwargs):
        builders.append(context._resource_local.profile)
        assert builders[-1].metadata()['status'] == 'image_only_admission'
        assert builders[-1]._lease.scope_holds == 1, 'one build owns the deferred refund right'
        assert staged(pool) == needed
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    context._runtime = proof_runtime(lambda _reference: staged(pool) == needed or pytest.fail('early refund'))
    with context.image_cohort_provider(view, context.source_volume.shape, planned) as reference:
        assert context._resource_local.profile is parent
        assert staged(pool) == needed and len(builders) == 1
        np.testing.assert_array_equal(crop_from(reference, 1, (1, 1, 5, 6)),
                                      context.source_volume[1, 1:5, 1:6])
    assert staged(pool) == 0 and not reference.path.exists()
    assert not context._retained_parent_image_credits
    assert not builders[0]._lease.active and parent._lease.active


def test_shared_borrower_keeps_original_image_grant_until_last_retirement(images):
    context, view, planned, pool, parent, needed = images
    with ExitStack() as cleanup:
        first = ExitStack()
        cleanup.callback(first.close)
        reference = first.enter_context(context.image_cohort_provider(view, context.source_volume.shape, planned))
        second = cleanup.enter_context(context.image_cohort_provider(view, context.source_volume.shape, planned))
        assert second is reference and staged(pool) == needed
        first.close()
        assert reference.path.exists() and staged(pool) == needed
        assert len(context._retained_parent_image_credits) == 1
        assert context._cache_owners[str(reference.path)]['leases'] == 1
    assert staged(pool) == 0 and not reference.path.exists()
    assert not context._retained_parent_image_credits
    assert context._resource_local.profile is parent


@pytest.mark.parametrize('provider_failure', (False, True))
def test_primary_failure_refunds_only_after_proven_cache_retirement(images, monkeypatch, provider_failure):
    context, view, planned, pool, parent, _needed = images
    failure = ValueError('controlled image/SDK failure')
    references = []
    if provider_failure:
        original = context.image_provider
        def fail(*args, **kwargs):
            references.append(original(*args, **kwargs))
            raise failure
        monkeypatch.setattr(context, 'image_provider', fail)
    with pytest.raises(ValueError) as caught:
        with context.image_cohort_provider(view, context.source_volume.shape, planned) as reference:
            references.append(reference)
            raise failure
    assert caught.value is failure
    assert staged(pool) == 0 and not references[0].path.exists()
    assert context._resource_local.profile is parent
    assert not context._cache_owners and context._active_image_cohorts == 0


def test_unproven_retirement_keeps_image_grant_until_runtime_closes(images):
    context, view, planned, pool, parent, needed = images
    runtime = context._runtime
    runtime.release_source_cache = mock.Mock(side_effect=RuntimeError('uncertain gray mappings'))
    with pytest.raises(RuntimeError, match='uncertain gray mappings'):
        with context.image_cohort_provider(view, context.source_volume.shape, planned) as reference:
            assert staged(pool) == needed
    assert reference.path.exists() and staged(pool) == needed
    assert len(context._retained_parent_image_credits) == 1
    assert context._cache_owners[str(reference.path)]['retirement_unproven']
    assert context._resource_local.profile is parent
    context.close()
    assert staged(pool) == 0 and not context._retained_parent_image_credits


def test_partial_claim_failure_keeps_primary_error_and_unproven_image_credit(images, monkeypatch):
    context, view, planned, pool, _parent, needed = images
    failure = ValueError('provider failed after atomic claim')
    original = context.image_provider
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise failure
    monkeypatch.setattr(context, 'image_provider', fail)
    context._runtime.release_source_cache = mock.Mock(side_effect=RuntimeError('uncertain mapping'))
    with pytest.raises(ValueError) as caught:
        with context.image_cohort_provider(view, context.source_volume.shape, planned):
            pytest.fail('partial failed provider exposed its reference')
    assert caught.value is failure and staged(pool) == needed
    assert len(context._retained_parent_image_credits) == 1
    assert any(owner.get('retirement_unproven') for owner in context._cache_owners.values())
    context.close()
    assert staged(pool) == 0


def test_declined_fresh_grant_is_explicit_before_render_and_has_no_wait(images, monkeypatch):
    context, view, planned, pool, parent, _needed = images
    monkeypatch.setattr(parent._lease, 'headroom_probe', lambda:64*GIB)
    with mock.patch.object(context, '_render_demand_crop') as render:
        with pytest.raises(RuntimeError, match='no available image-only credit'):
            with context.image_cohort_provider(view, context.source_volume.shape, planned):
                pytest.fail('unfunded image grant reached consumer')
    render.assert_not_called()
    assert staged(pool) == 0 and context._active_image_cohorts == 0
    assert context._resource_local.profile is parent


def test_invalid_demand_and_native_borrow_do_not_request_extra_image_credit(images, tmp_path):
    context, view, planned, pool, _parent, _needed = images
    with mock.patch('XTA.sam_image_prefetch._try_image_profile') as grant:
        with pytest.raises(ValueError, match='outside'):
            with context.image_cohort_provider(view, context.source_volume.shape,
                    demand(context.source_volume.shape, {1:(0, 0, 100, 100)})):
                pytest.fail('invalid demand reached consumer')
        path = tmp_path/'native.gray8.dat'
        source = np.memmap(path, dtype=np.uint8, mode='w+', shape=context.source_volume.shape)
        source[:] = context.source_volume
        source.flush()
        context.source_volume = source
        try:
            whole = demand(source.shape, {frame:(0, 0, 7, 9) for frame in range(3)})
            with context.image_cohort_provider(view, source.shape, whole) as reference:
                assert reference.path == path and staged(pool) == 0
            assert path.exists()
        finally:
            context.source_volume = np.asarray(source).copy()
            source._mmap.close()
        grant.assert_not_called()


@pytest.mark.parametrize('fail_startup', (False, True))
@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
def test_initial_pass_builds_before_start_and_holds_grant_through_execution(
        tmp_path, monkeypatch, fail_startup, operation):
    from XTA import sam_extrapolation, sam_interpolation
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(2*GIB))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    source = np.full((3, 7, 9), 73, np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    observed = np.zeros_like(source)
    if operation == 'interpolation':
        observed[0, 2:4, 2:4] = observed[2, 2:4, 2:4] = 1
        module, prepare_name, execute_name, call = (sam_interpolation,
            'prepare_sam_interpolation_pass', 'interpolate_sam_view_volume_pass', context.interpolate)
        prepared = sam_interpolation.prepare_sam_interpolation_pass(observed, view=view,
            scope='parent', gap_distance=3, interpolation_walk_back=0)
    else:
        observed[1, 2:4, 2:4] = 1
        module, prepare_name, execute_name, call = (sam_extrapolation,
            'prepare_sam_extrapolation_pass', 'extrapolate_sam_view_volume_pass', context.extrapolate)
        prepared = sam_extrapolation.prepare_sam_extrapolation_pass(observed, view=view,
            scope='parent', distance=1, walk_back=0, min_radius=0.)
    assert prepared.needs_tracking
    pool = _ByteAdmissionPool(64*GIB, 'interpolation-images')
    order = []
    original = context._render_demand_crop
    def render(*args, **kwargs):
        order.append('image')
        assert context._resource_local.profile.metadata()['status'] == 'image_only_admission'
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    failure = RuntimeError('controlled predictor startup failure')
    try:
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent',
                base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as parent:
            def start():
                assert order and context._resource_local.profile is parent and staged(pool) > 2*GIB
                order.append('start')
                if fail_startup:
                    raise failure
                context._runtime = proof_runtime(lambda _reference: order.append('retire'))
            def track(volume, **kwargs):
                assert context._resource_local.profile is kwargs['resource_profile'] is parent
                assert staged(pool) > 2*GIB and kwargs['image_provider'].path.exists()
                order.append('track')
                return volume, {}, []
            monkeypatch.setattr(context, '_start', start)
            monkeypatch.setattr(module, prepare_name, lambda *args, **kwargs:prepared)
            monkeypatch.setattr(module, execute_name, track)
            with context.resource_scope(parent):
                if fail_startup:
                    with pytest.raises(RuntimeError) as caught:
                        call(observed, view=view, scope='parent')
                    assert caught.value is failure and staged(pool) > 2*GIB
                    context.close()
                else:
                    merged, _stats, _components = call(observed, view=view, scope='parent')
                    assert merged is observed and order[-3:] == ['start', 'track', 'retire']
                assert staged(pool) == 0 and context._active_passes == 0
    finally:
        context.close()
