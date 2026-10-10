"""CPU-only retained-plan composition; no production gate or throughput claim."""
from concurrent.futures import CancelledError
import hashlib
import json
from threading import Event, Thread

import numpy as np
import pytest

from XTA import geometry, sam_resources as resources
from XTA.assembly import prepare_detector_view_cpu
from XTA.interpolation import _ByteAdmissionPool
from XTA.lta_rendering import render_native_tile_window
from XTA.sam_image_prefetch import _retain_image_credit, _try_image_profile
from XTA.sam_interpolation import (SamInterpolationInfrastructureError,
    interpolate_sam_view_volume_pass, prepare_sam_interpolation_pass)
from tests.test_sam_interpolation import RepeatedSeedTracker, _observations
from tests.test_sam_view_image_cache import context_for, crop_from


MIB = 1024**2
PARENT_CAPACITY = 3*1024**3
SCRATCH_BYTES = 8192


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class _BarrierSDK(RepeatedSeedTracker):
    def __init__(self, barrier, yolo_done, cache, *, failure=False):
        super().__init__(failure=failure)
        self.barrier, self.yolo_done, self.cache = barrier, yolo_done, cache
        self.clips = []

    def run(self, **request):
        assert self.barrier.is_set(), 'SDK must not consume before its start barrier'
        assert self.yolo_done.is_set(), 'SDK must wait for the other detector producer'
        clip = render_native_tile_window(self.cache, frame_start=request['frame_start'],
            frame_stop=request['frame_stop'], tile_xyxy=request['crop_xyxy'])
        assert clip.source_cache_mapping_retired
        self.clips.append([_digest(np.asarray(frame)) for frame in clip])
        return super().run(**request)


def _job(request):
    return dict(run_id=request['run_id'], crop_xyxy=list(request['crop_xyxy']),
        seed_frame=request['seed_frame'], frame_start=request['frame_start'],
        frame_stop=request['frame_stop'], direction=request['direction'],
        seed_sha256=_digest(request['seed_mask']))


def _compose(root, *, early, failure=None):
    if resources.physical_sam_headroom() < 2*PARENT_CAPACITY+2*MIB:
        pytest.skip('Actual RAM headroom cannot fund the independent CPU profile')
    mask = _observations()
    mask[0, 11, 12] = mask[4, 11, 12] = 0
    mask[1, 1, 1] = 1
    scores = np.full(mask.shape, 230, np.uint8)
    scores[1, 1, 1] = 20
    gray = np.arange(mask.size, dtype=np.uint16).astype(np.uint8).reshape(mask.shape)
    view = geometry.get_view_infos(*gray.shape, cartesian_views=('transverse',))[0]
    context = context_for(root, gray)
    pool = _ByteAdmissionPool(PARENT_CAPACITY, 'preplan composition')
    started, release_yolo, yolo_done, sdk_start = (Event() for _ in range(4))
    other_mask = np.zeros_like(mask)
    def other_producer():
        started.set()
        release_yolo.wait()
        other_mask[-1] = 1
        yolo_done.set()
    producer = Thread(target=other_producer, name='mock-other-yolo')
    producer.start()
    assert started.wait(2)
    if not early:
        release_yolo.set()
        producer.join(2)
        assert yolo_done.is_set()
        sdk_start.set()
    cache = release_images = merged = prepared = tracker = None
    evidence = dict(kind='retained CPU preplan composition', early=early,
        mocked=('other YOLO dependency', 'SDK start and repeated-seed outputs'),
        performance_benchmark=False, gpu_inference=False)
    try:
        with pool.reserve(MIB, 'protected other detector allowance'):
            with resources.admit_sam_parent_resources(pool, MIB, 'completed-view',
                    worker_count=1, execution_slots=1, base_allowance_bytes=MIB) as profile:
                try:
                    prepare_detector_view_cpu(mask, scores, view, .5, 0., workers=1)
                    assert mask[1, 1, 1] == 0 and mask[0].any() and mask[4].any()
                    mask.flags.writeable = False
                    metadata = dict(scope_id='fixture/completed-view/fullframe',
                        canvas_transform=context._canvas_transform(view, mask.shape)[2], sam_crop_mode='whole')
                    options = dict(view=view, scope=metadata, gap_distance=5, min_radius=0,
                        interpolation_walk_back=0, resource_profile=profile, crop_mode='whole')
                    prepared = prepare_sam_interpolation_pass(mask, **options)
                    assert prepared.needs_tracking and prepared.runs
                    payload = sum((box[2]-box[0])*(box[3]-box[1])
                        for box in prepared.frame_crop_bounds.values())
                    with _try_image_profile(pool, payload+SCRATCH_BYTES,
                            'completed-view/gray-bank', resources.physical_sam_headroom) as images:
                        if images is None:
                            pytest.skip('Actual image-bank RAM headroom is unavailable')
                        release_images = _retain_image_credit(images)
                        with context.resource_scope(images):
                            cache = context.image_provider(view, mask.shape, prepared)
                    assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == payload+SCRATCH_BYTES
                    assert context._cache_owners[str(cache.path)]['owned']
                    cache.revalidate()
                    crop_digests = []
                    for frame, box in prepared.frame_crop_bounds.items():
                        y0, x0, y1, x1 = box
                        pixels = crop_from(cache, frame, box)
                        np.testing.assert_array_equal(pixels, gray[frame, y0:y1, x0:x1])
                        crop_digests.append([frame, list(box), _digest(pixels)])
                    tracker = _BarrierSDK(sdk_start, yolo_done, cache, failure=failure=='sdk')
                    evidence['ready_bank'] = dict(sdk_calls=len(tracker.calls),
                        other_yolo_live=producer.is_alive(), other_yolo_done=yolo_done.is_set(),
                        real_detector_retirement_ready=context.detector_retirement_ready,
                        cache_bytes=cache.size_bytes, image_credit_bytes=payload+SCRATCH_BYTES,
                        parent_credit_bytes=pool.in_use, mask_bytes=mask.nbytes,
                        snapshot_sha256=prepared.observation_snapshot_sha256,
                        settings_sha256=prepared.settings_sha256, crops=crop_digests)
                    assert tracker.calls == [] and not context.detector_retirement_ready
                    if early:
                        assert producer.is_alive() and not yolo_done.is_set() and not sdk_start.is_set()
                        assert not other_mask.any()
                    if failure=='cancel':
                        raise CancelledError('controlled retained-bank cancellation')
                    release_yolo.set()
                    producer.join(2)
                    assert yolo_done.is_set() and not producer.is_alive()
                    sdk_start.set()
                    evidence['handoff'] = dict(other_yolo_joined=True, mocked_sdk_barrier_open=True)
                    merged, stats, _ = interpolate_sam_view_volume_pass(mask,
                        work_dir=root/'selected', image_provider=cache, runtime=tracker,
                        prepared_plan=prepared, return_bridge_components=True, **options)
                    result = np.asarray(merged).copy()
                    evidence['executed'] = dict(jobs=[_job(request) for request in tracker.calls],
                        clips=tracker.clips, selected_run_ids=stats['sam_selection_receipt']['selected_run_ids'],
                        added_voxels=stats['added_voxels'], output_sha256=_digest(result))
                    assert prepared.plan.contract_lease_budget.snapshot()['active_leases'] == 0
                finally:
                    if isinstance(merged, np.memmap):
                        merged._mmap.close()
                    prepared = merged = None
                    context.close()
                    if cache is not None:
                        assert cache.path.parent == (root/'runtime'/'sam_image_cache').resolve()
                        cache.path.unlink()
                        assert not cache.path.exists()
                    if release_images is not None:
                        release_images()
                    assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 0
        return result, evidence
    finally:
        context.close()
        release_yolo.set()
        producer.join(2)
        assert not producer.is_alive()
        assert resources.sam_parent_promised_bytes(pool) == 0
        evidence['cleanup'] = dict(parent_in_use_bytes=pool.in_use,
            image_in_use_bytes=resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'],
            other_producer_joined=True, image_bank_retired=cache is None or not cache.path.exists())
        root.mkdir(parents=True, exist_ok=True)
        (root/'protocol-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


def test_real_cpu_plan_and_gray_bank_precede_mock_sdk_while_other_yolo_is_pending(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGES', '0')
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(SCRATCH_BYTES))
    early, early_evidence = _compose(tmp_path/'early', early=True)
    control, control_evidence = _compose(tmp_path/'serial', early=False)
    np.testing.assert_array_equal(early, control)
    assert early_evidence['ready_bank']['sdk_calls'] == control_evidence['ready_bank']['sdk_calls'] == 0
    assert early_evidence['ready_bank']['other_yolo_live']
    for key in ('crops', 'snapshot_sha256', 'settings_sha256'):
        assert early_evidence['ready_bank'][key] == control_evidence['ready_bank'][key]
    assert early_evidence['executed'] == control_evidence['executed']
    assert early_evidence['executed']['added_voxels'] > 0


@pytest.mark.parametrize('failure', ('cancel', 'sdk'))
def test_retained_bank_cancel_or_mock_sdk_failure_retires_actual_credits(tmp_path, monkeypatch, failure):
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGES', '0')
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(SCRATCH_BYTES))
    error = CancelledError if failure=='cancel' else SamInterpolationInfrastructureError
    with pytest.raises(error, match='controlled'):
        _compose(tmp_path/failure, early=True, failure=failure)
    evidence = json.loads((tmp_path/failure/'protocol-evidence.json').read_text())
    assert evidence['ready_bank']['sdk_calls'] == 0 and evidence['ready_bank']['other_yolo_live']
    assert evidence['cleanup']['parent_in_use_bytes'] == evidence['cleanup']['image_in_use_bytes'] == 0
    assert evidence['cleanup']['image_bank_retired'] and evidence['cleanup']['other_producer_joined']
