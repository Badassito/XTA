"""Required extrapolation images fit live parent credit under image pressure."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sam_extrapolation as extrapolation, sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_image_prefetch import _try_image_profile
from tests.test_sam_extrapolation_family_dispatch import _baseline, _pixels, Runtime
from tests.test_sam_view_image_cache import context_for, crop_from, demand


def _payload(prepared):
    return sum((box[2]-box[0])*(box[3]-box[1]) for box in prepared.frame_crop_bounds.values())


@pytest.mark.parametrize('pressure', ('physical_headroom', 'live_physical_headroom', 'staging_capacity'))
@pytest.mark.parametrize('guaranteed', (False, True))
def test_extrapolation_keeps_complete_groups_and_exact_output_under_image_pressure(
        tmp_path, monkeypatch, pressure, guaranteed):
    observed = _baseline(count=3)
    frozen = observed.copy()
    source = (np.arange(observed.size, dtype=np.uint32) % 251).astype(np.uint8).reshape(observed.shape)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    prepared = extrapolation.prepare_sam_extrapolation_pass(observed, view=view, scope='parent',
        distance=3, walk_back=1, min_radius=3.)
    individual = [_payload(extrapolation._cohort_prepared(prepared, (group.group_id,)))
                  for group in prepared.groups]
    funded_cap = max(individual)
    assert 0 < funded_cap < _payload(prepared) < resources.GIB
    # Keep production's exact 1 GiB parent image partition; scale only the
    # configured scratch/payload boundary so this regression allocates no GiB.
    scratch = resources.GIB-funded_cap
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(scratch))
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', str(resources.GIB))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    reference_runtime = Runtime(prepared)
    _, expected_stats, expected_parts = extrapolation.extrapolate_sam_view_volume_pass(
        observed, prepared_plan=prepared, runtime=reference_runtime,
        work_dir=tmp_path/'reference', view=view, scope='parent',
        distance=3, walk_back=1, min_radius=3.)
    context = context_for(tmp_path/'current', source)
    runtime = Runtime(prepared)
    original_prepare = extrapolation.prepare_sam_extrapolation_pass
    def prepare(*args, **kwargs):
        runtime.prepared = original_prepare(*args, **kwargs)
        return runtime.prepared
    monkeypatch.setattr(extrapolation, 'prepare_sam_extrapolation_pass', prepare)
    retirements, references, declines, admitted_funding = [], [], [], []
    runtime.close = lambda:None
    runtime.dispatch_stats = {}
    def retire(reference):
        assert reference.path.exists()
        retirements.append(reference.path)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    runtime.release_source_cache = retire
    original_results = runtime.iter_results
    def results(requests, source_cache_ref=None, **kwargs):
        assert source_cache_ref is not None
        references.append(source_cache_ref)
        for frame, y0, x0, y1, x1, _offset in source_cache_ref.frame_crops:
            np.testing.assert_array_equal(crop_from(source_cache_ref, frame, (y0, x0, y1, x1)),
                                          source[frame, y0:y1, x0:x1])
        yield from original_results(requests, source_cache_ref=source_cache_ref, **kwargs)
    runtime.iter_results = results
    context._runtime = runtime
    monkeypatch.setattr(context, '_start', lambda:None)
    original_try = _try_image_profile
    @contextmanager
    def recorded_try(*args, **kwargs):
        with original_try(*args, **kwargs) as profile:
            declines.append(profile is None)
            if profile is not None:
                admitted_funding.append(profile._lease.parent_image_funding is not None)
            yield profile
    monkeypatch.setattr('XTA.sam_image_prefetch._try_image_profile', recorded_try)
    pool = _ByteAdmissionPool(64*resources.GIB, 'funded-extrapolation-images')
    retry_request = SimpleNamespace(settled=False)
    source_hold = context._hold_retry_source(retry_request)
    release_retained = None
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent', worker_count=4,
                base_allowance_bytes=4*resources.GIB, headroom_probe=lambda:128*resources.GIB,
                image_phase_bytes=context.mandatory_image_phase_bytes() if guaranteed else 0) as parent:
            # A previously closed producer still owns its image credit while
            # another reader retains the source. Admission must not refund it.
            with original_try(pool, 4096, 'retained-retry/images', lambda:128*resources.GIB) as retained:
                release_retained = resources.retain_sam_parent_image_credit(retained)
            staging = pool._sam_image_staging_pool
            if pressure == 'physical_headroom':
                # Hold physical availability fixed. Growing the sample with
                # funded staging would invent RAM and legitimately enable a
                # later standalone grant after the mixed-mode overlap fix.
                physical = pool.capacity+staging.in_use-1
                parent._lease.headroom_probe = lambda:physical
            elif pressure == 'live_physical_headroom':
                parent._lease.headroom_probe = lambda:resources.sam_parent_promised_bytes(pool)
            else:
                staging.capacity = staging.in_use
            with context.resource_scope(parent):
                # The former full-scope descriptor cannot enter under exactly
                # this pressure; the fix must change cohort admission, not
                # loosen the image pool or consume the retained retry credit.
                if guaranteed:
                    with context.image_cohort_provider(view, source.shape, prepared) as full:
                        assert full.path.exists()
                    assert not full.path.exists()
                    retirements.clear()
                else:
                    with pytest.raises(RuntimeError, match='no available image-only credit'):
                        with context.image_cohort_provider(view, source.shape, prepared):
                            pytest.fail('unfunded whole-scope image descriptor was admitted')
                assert staging.in_use == 4096 and not source_hold.closed
                returned, actual_stats, actual_parts = context.extrapolate(observed, view=view, scope='parent',
                    work_dir=tmp_path/'actual', distance=3, walk_back=1, min_radius=3.)
                assert returned is observed and context._resource_local.profile is parent
                assert not source_hold.closed and staging.in_use == 4096
                record = actual_stats['sam_image_cache_cohorts']
                assert record['configured_cache_bytes'] == resources.GIB
                assert record['effective_cache_bytes'] == (resources.GIB if guaranteed else funded_cap)
                assert record['renderer_scratch_bytes'] == scratch
                assert record['parent_image_allowance_bytes'] == parent.assigned_image_bytes
                assert (len(record['cohorts']) == 1) if guaranteed else (len(record['cohorts']) > 1)
                assert all(item['payload_bytes']+scratch <= parent.assigned_image_bytes
                           for item in record['cohorts'])
                assert {gid for item in record['cohorts'] for gid in item['group_ids']} == {
                    group.group_id for group in prepared.groups}
                if guaranteed:
                    assert not declines and not admitted_funding
                elif pressure == 'physical_headroom':
                    # The unchanged mandatory/current-image request still
                    # declines; optional overlap now spends real parent credit.
                    assert declines[0] and not all(declines)
                    assert admitted_funding and all(admitted_funding)
                else:
                    assert declines and all(declines) and not admitted_funding
                assert pool.in_use == parent._lease.charged_bytes
                assert resources.sam_image_staging_snapshot(pool)['image_parent_funded_bytes'] == 0
                assert references and len(retirements) == len(record['cohorts'])
                assert all(not path.exists() for path in retirements)
                assert not context._cache_owners and context._active_image_cohorts == 0
                assert not context._image_prefetches and not context._retained_parent_image_credits
            release_retained()
            release_retained = None
        assert resources.sam_parent_promised_bytes(pool) == 0
        actual = SamEvidenceBundle.open(actual_stats['sam_evidence_path'])
        expected = SamEvidenceBundle.open(expected_stats['sam_evidence_path'])
        assert set(actual.runs) == set(expected.runs) == {run.run_id for run in prepared.runs}
        with actual.reader() as left, expected.reader() as right:
            for run in prepared.runs:
                for frame in run.expected_frames:
                    np.testing.assert_array_equal(left.raw_mask(run.run_id, frame), right.raw_mask(run.run_id, frame))
        np.testing.assert_array_equal(_pixels(actual_parts, observed.shape), _pixels(expected_parts, observed.shape))
        np.testing.assert_array_equal(observed, frozen)
    finally:
        if release_retained is not None:
            release_retained()
        retry_request.settled = True
        source_hold.close()
        context.close()


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('patch_index', (0, -1))
def test_native_radial_gray8_cohort_subsets_preserve_exact_tracker_pixels(
        tmp_path, monkeypatch, base, patch_index):
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    source = np.random.default_rng(157400).integers(0, 256, (63, 71, 67), dtype=np.uint8)
    views = geometry.get_view_infos(*source.shape, cartesian_views=(),
        radial_views=(base,), radial_patch_size=32, radial_min_radius=.5)
    view = views[patch_index]
    shape = (view.num_slices, view.src_h, view.src_w)
    frames = sorted({0, view.num_slices//2, view.num_slices-1})
    broad = (0, 0, shape[1], shape[2])
    crops = ((1, 2, shape[1]-3, shape[2]-1),
             (shape[1]//2, 0, shape[1], shape[2]//2),
             (0, shape[2]//2, shape[1]//2, shape[2]))
    reference_context = context_for(tmp_path/'full', source)
    try:
        reference = reference_context.image_provider(view, shape,
            demand(shape, {frame:broad for frame in frames}))
        for index, crop in enumerate(crops):
            # Separate contexts prevent a superset cache/donor hit from hiding
            # a changed native sampler phase at the narrower demand origin.
            context = context_for(tmp_path/f'cohort-{index}', source)
            try:
                actual = context.image_provider(view, shape,
                    demand(shape, {frame:crop for frame in frames}))
                assert context.rendered_frames == len(frames)
                for frame in frames:
                    np.testing.assert_array_equal(crop_from(actual, frame, crop),
                                                  crop_from(reference, frame, crop))
            finally:
                context.close()
    finally:
        reference_context.close()


def test_indivisible_group_keeps_validated_separate_grant_and_original_runs(tmp_path, monkeypatch):
    from tests.test_sam_cache_transactions import proof_runtime
    source = np.full((3, 7, 9), 73, np.uint8)
    observed = np.zeros_like(source)
    observed[1, 2:4, 2:4] = 1
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(resources.GIB-100))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args, **kwargs:None)
    monkeypatch.setattr(context, '_start', lambda:None)
    context._runtime = proof_runtime()
    pool = _ByteAdmissionPool(64*resources.GIB, 'indivisible-images')
    def execute(volume, **kwargs):
        prepared = kwargs['prepared_plan']
        assert max(_payload(extrapolation._cohort_prepared(prepared, (group.group_id,)))
                   for group in prepared.groups) > 100
        assert {run.run_id for cohort in kwargs['image_cohorts'] for run in cohort.prepared.runs} == {
            run.run_id for run in prepared.runs}
        assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] > resources.GIB
        assert kwargs['image_provider'].path.exists()
        return volume, {}, []
    monkeypatch.setattr(extrapolation, 'extrapolate_sam_view_volume_pass', execute)
    try:
        with resources.admit_sam_parent_resources(pool, 4*resources.GIB, 'parent',
                base_allowance_bytes=4*resources.GIB, headroom_probe=lambda:128*resources.GIB) as parent:
            with context.resource_scope(parent):
                _, stats, _ = context.extrapolate(observed, view=view, scope='parent',
                    distance=1, walk_back=0, min_radius=0.)
                assert stats['sam_image_cache_cohorts']['effective_cache_bytes'] == resources.GIB
                assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 0
    finally:
        context.close()
