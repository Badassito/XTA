"""Cross-parent pressure must not turn CPU publication into a GPU lease barrier."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sam_evidence as evidence
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.sam_gpu_rendering import SamGpuCropRenderer
from tests.test_sam_compute_ack_adversarial import _protocol
from tests.test_sam_parallel_mask_packing import admitted, immutable_empty
from tests.test_sam_view_image_cache import context_for, crop_from, demand


def reached(event, pending=None):
    # Watchdog only: ordering is proved by events and ownership, never elapsed speed.
    if not event.wait(10):
        if pending is not None and pending.done():
            pending.result()  # Surface the producer failure instead of hiding it as a timeout.
        pytest.fail('pressure scenario did not reach its required state')


@pytest.mark.parametrize('cache_barrier,outcome', [
    ('flush', 'success'), ('revalidate', 'success'), ('flush', 'flush_failure'),
    ('flush', 'cancel'), ('flush', 'packing_failure'),
])
def test_ready_tracking_progresses_during_cache_and_evidence_publication(
        tmp_path, monkeypatch, cache_barrier, outcome):
    tracker, request, worker, coordinator, resident, _aux, handbacks = _protocol(tmp_path, monkeypatch)
    source = np.arange(3*15*21, dtype=np.uint16).reshape(3,15,21).astype(np.uint8)
    context = context_for(tmp_path/'context', source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('coronal',))[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    box = (0, 0, shape[1], shape[2])
    planned = demand(shape, {frame: box for frame in range(3)})
    expected = source.transpose(2,0,1)[:3].copy()
    fake_torch = SimpleNamespace(device=lambda value:value,
        cuda=SimpleNamespace(device_count=lambda:2,
            mem_get_info=lambda _device:(8*1024**3,16*1024**3)))
    image_lease = resident.try_acquire_compute(fake_torch, 'SAM image preparation')
    assert image_lease is not None
    renderer = SamGpuCropRenderer(context, view, source, source.shape[0], image_lease,
                                 fake_torch, 1024**2)
    last_render, finish_render = threading.Event(), threading.Event()
    finalizing, finish_cache = threading.Event(), threading.Event()
    encoder_started, finish_encoder = threading.Event(), threading.Event()
    cached_queued, compute_retired = threading.Event(), threading.Event()
    retirement_calls = []

    def retire_assets():
        retirement_calls.append('source_and_stream_settled')
        compute_retired.set()

    renderer.engine = SimpleNamespace(release_inference_assets=retire_assets, _volume_mm=None)
    context._gpu_image_owners[1] = renderer
    renderer.sampling_identity = lambda:dict(contract='pressure-fixture', backend='cuda',
                                            implementation_sha256='pressure-fixture')

    def render(_view, frame, _inverse, *, output_origin_yx, output_height, output_width):
        assert renderer.lease is image_lease and not compute_retired.is_set()
        if frame == 2:
            last_render.set()
            reached(finish_render)
        y, x = output_origin_yx
        return expected[frame, y:y+output_height, x:x+output_width].copy()

    renderer.render = render
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *_a, **_k:renderer)
    monkeypatch.setattr('XTA.backprojection._trim_main_process_cuda_device', lambda *_a, **_k:None)
    original_flush, original_revalidate = np.memmap.flush, LtaPhysicalViewCacheRef.revalidate
    cache_failure = OSError('controlled image-cache publication failure')

    def publication_gate():
        finalizing.set()
        reached(finish_cache)
        if outcome == 'flush_failure':
            raise cache_failure

    def flush(array):
        if cache_barrier == 'flush' and 'sam_image_cache' in str(getattr(array, 'filename', '')):
            publication_gate()
        return original_flush(array)

    def revalidate(reference):
        if cache_barrier == 'revalidate' and 'sam_image_cache' in str(reference.path):
            publication_gate()
        return original_revalidate(reference)

    monkeypatch.setattr(np.memmap, 'flush', flush)
    monkeypatch.setattr(LtaPhysicalViewCacheRef, 'revalidate', revalidate)
    original_encode = evidence._encode_mask
    encoding_failure = ValueError('controlled pressure encoder failure')

    def encode(value):
        if value.all():
            encoder_started.set()
            reached(finish_encoder)
            if outcome == 'packing_failure':
                raise encoding_failure
        return original_encode(value)

    monkeypatch.setattr(evidence, '_encode_mask', encode)
    packing = {}

    def write_evidence():
        empty = immutable_empty()
        with evidence.SamEvidenceWriter(tmp_path/'pressure-evidence', {}) as writer:
            writer._put('seed', empty, empty.shape)
            with admitted() as (pool, _profile, _scope, permit):
                packing.update(writer=writer, pool=pool, permit=permit)
                with writer.parallel_packing(permit):
                    writer._put('slow', np.ones(empty.shape, bool), empty.shape)
                    reached(encoder_started)
                    writer._put('cached', empty, empty.shape)
                    packing['queued_bytes'] = writer._packing_bytes
                    packing['scratch_bytes'] = permit.scratch_bytes(empty.size)
                    packing['queued_count'] = len(writer._packing_pending)
                    packing['tracked_futures'] = len(permit._futures)
                    cached_queued.set()
                assert not writer._packing_pending and not permit._futures
                assert writer._packing_bytes == 0
            assert pool.in_use == 0
            bundle = writer.commit()
        assert list(bundle.records) == ['cached', 'seed', 'slow']  # Canonical JSON key order.
        assert (bundle.records['seed']['offset'] < bundle.records['slow']['offset']
                < bundle.records['cached']['offset'])
        assert not bundle.mask('seed').any() and not bundle.mask('cached').any()
        assert bundle.mask('slow').all()
        return bundle

    def track_ready_parent():
        found = []
        for index, result in tracker.iter_results((request,)):
            try:
                assert result.receipt['run_id'] == request['run_id']
                assert all(np.array_equal(mask, request['seed_mask']) for mask in result.frames.values())
                found.append((index, result.receipt['run_id']))
            finally:
                tracker.release_result(result)
        return found

    try:
        with ThreadPoolExecutor(max_workers=3) as threads:
            images = threads.submit(context.image_provider, view, shape, planned)
            tracking = packing_future = None
            try:
                reached(last_render)
                tracking = threads.submit(track_ready_parent)
                with tracker._state_condition:
                    assert tracker._state_condition.wait_for(
                        lambda:any(scope.ready for scope in tracker._scopes.values()), timeout=10)
                assert not worker.submissions, 'ready tracker borrowed an unretired image compute lease'
                packing_future = threads.submit(write_evidence)
                reached(encoder_started, packing_future)
                reached(cached_queued, packing_future)
                assert not finish_encoder.is_set() and not packing_future.done()
                assert packing['queued_count'] == packing['tracked_futures'] == 2
                assert 0 < packing['queued_bytes'] <= packing['scratch_bytes']
                assert packing['pool'].in_use > 0
                finish_render.set()
                reached(finalizing)
                assert compute_retired.is_set() and renderer.lease is None and renderer.engine is None
                assert tracking.result(timeout=10) == [(0, request['run_id'])]
                assert handbacks == ['compute_released'] and not worker.active
                assert not images.done() and not packing_future.done()
                assert context._image_build_bytes > 0 and not context._caches
                assert not coordinator.snapshot()['stage_leases']
                if outcome == 'cancel':
                    context.cancel('controlled pressure cancellation')
                finish_cache.set()
                if outcome == 'flush_failure':
                    with pytest.raises(OSError) as caught:
                        images.result(timeout=10)
                    assert caught.value is cache_failure
                elif outcome == 'cancel':
                    with pytest.raises(RuntimeError, match='cancel|closing|lifetime'):
                        images.result(timeout=10)
                else:
                    reference = images.result(timeout=10)
                    for frame in range(3):
                        np.testing.assert_array_equal(crop_from(reference, frame, box), expected[frame])
                finish_encoder.set()
                if outcome == 'packing_failure':
                    with pytest.raises(ValueError) as caught:
                        packing_future.result(timeout=10)
                    assert caught.value is encoding_failure
                    assert not (tmp_path/'pressure-evidence').exists()
                else:
                    packing_future.result(timeout=10)
                assert packing['pool'].in_use == 0
                assert not packing['permit']._futures and packing['writer']._packing_bytes == 0
                assert context._image_build_bytes == context._image_build_credit_bytes == 0
                assert context._active_image_calls == 0 and not context._image_builds
                assert retirement_calls == ['source_and_stream_settled']
                if outcome in ('flush_failure', 'cancel'):
                    assert not context._caches
                    assert not list((context.temp_dir/'sam_image_cache').glob('*.gray8.dat'))
            finally:
                finish_render.set()
                finish_cache.set()
                finish_encoder.set()
    finally:
        finish_render.set()
        finish_cache.set()
        finish_encoder.set()
        renderer.close()
        context.close()
        tracker.close()
        resident.release(residency_settled=True)
    assert not coordinator.snapshot()['stage_leases']
    assert not coordinator.snapshot()['resident_owners']
