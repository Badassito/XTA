"""SAM overlap must retain exact pixels, bounded publication and source owners."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation
from XTA.projection_queue import ComponentProjectionQueue, settle_prepared_view_components
from XTA.reconciliation_runtime import RuntimeLayer
from tests.test_sam_view_image_cache import context_for, crop_from, demand


def _native_cases(source):
    views = geometry.get_view_infos(*source.shape, cartesian_views=('sagittal', 'coronal'))
    expected = {'sagittal': source.transpose(1, 0, 2), 'coronal': source.transpose(2, 0, 1)}
    return [(view, (view.num_slices, view.src_h, view.src_w), expected[view.name]) for view in views]


def test_independent_native_render_progresses_while_another_scope_is_blocked(tmp_path, monkeypatch):
    source = np.arange(3 * 6 * 7, dtype=np.uint8).reshape(3, 6, 7)
    first_case, second_case = _native_cases(source)
    context = context_for(tmp_path, source)
    first_entered, release_first = threading.Event(), threading.Event()
    render = context._render_demand_crop

    def gated(view, index, *args, **kwargs):
        if view.name == first_case[0].name:
            first_entered.set()
            assert release_first.wait(10), 'test did not release its blocked renderer'
        return render(view, index, *args, **kwargs)

    monkeypatch.setattr(context, '_render_demand_crop', gated)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            view, shape, _expected = first_case
            first = workers.submit(context.image_provider, view, shape,
                demand(shape, {1: (0, 0, shape[1], shape[2])}))
            try:
                assert first_entered.wait(3)
                view, shape, _expected = second_case
                second = workers.submit(context.image_provider, view, shape,
                    demand(shape, {1: (0, 0, shape[1], shape[2])}))
                second_ref = second.result(timeout=3)
                assert not first.done(), 'independent renderer must finish before the first renderer is released'
                live = context.image_cache_lifetime_snapshot()
                assert live['active_image_builders'] == live['active_image_calls'] == 1
                assert live['image_build_peak_count'] == 2
                assert live['active_image_build_bytes'] == first_case[1][1] * first_case[1][2]
            finally:
                release_first.set()
            first_ref = first.result(timeout=5)
        for reference, (_view, shape, expected) in zip((first_ref, second_ref), (first_case, second_case)):
            reference.revalidate()
            np.testing.assert_array_equal(crop_from(reference, 1, (0, 0, shape[1], shape[2])), expected[1])
        total = sum(shape[1] * shape[2] for _view, shape, _expected in (first_case, second_case))
        assert context.rendered_frames == 2 and context.rendered_pixels == total
        assert context.cache_logical_bytes == total
        settled = context.image_cache_lifetime_snapshot()
        assert settled['protected_owned_cache_bytes'] == settled['image_build_peak_bytes'] == total
        assert settled['active_image_builders'] == settled['active_image_build_bytes'] == settled['active_image_calls'] == 0
    finally:
        release_first.set()
        context.close()


def test_parallel_identical_demand_has_one_immutable_descriptor_and_one_render(tmp_path, monkeypatch):
    source = np.arange(3 * 6 * 7, dtype=np.uint8).reshape(3, 6, 7)
    view, shape, expected = _native_cases(source)[0]
    context = context_for(tmp_path, source)
    entered, duplicate_entered, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    render = context._render_demand_crop
    planned = demand(shape, {1: (0, 0, shape[1], shape[2])})

    def gated(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(10), 'test did not release its duplicate-demand renderer'
        return render(*args, **kwargs)

    def duplicate():
        duplicate_entered.set()
        return context.image_provider(view, shape, planned)

    monkeypatch.setattr(context, '_render_demand_crop', gated)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(context.image_provider, view, shape, planned)
            try:
                assert entered.wait(3)
                second = workers.submit(duplicate)
                assert duplicate_entered.wait(3)
                assert not first.done() and not second.done()
            finally:
                release.set()
            first_ref, second_ref = first.result(timeout=5), second.result(timeout=5)
        assert first_ref is second_ref
        assert len(calls) == context.rendered_frames == 1
        assert context.cache_logical_bytes == shape[1] * shape[2]
        np.testing.assert_array_equal(crop_from(first_ref, 1, (0, 0, shape[1], shape[2])), expected[1])
    finally:
        release.set()
        context.close()


def test_close_cancels_inflight_render_and_waits_for_source_owner_before_return(tmp_path, monkeypatch):
    source = np.arange(3 * 6 * 7, dtype=np.uint8).reshape(3, 6, 7)
    view, shape, _expected = _native_cases(source)[0]
    context = context_for(tmp_path, source)
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    render = context._render_demand_crop

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'test did not release its cancelled renderer'
        return render(*args, **kwargs)

    def close():
        closing.set()
        context.close()

    monkeypatch.setattr(context, '_render_demand_crop', gated)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = workers.submit(context.image_provider, view, shape,
                demand(shape, {1: (0, 0, shape[1], shape[2])}))
            try:
                assert entered.wait(3)
                teardown = workers.submit(close)
                assert closing.wait(3)
                assert context._cancel.wait(3)
                assert not teardown.done()
                assert context.source_volume is source
            finally:
                release.set()
            with pytest.raises(RuntimeError, match='closing|cancel|lifetime|ended'):
                pending.result(timeout=5)
            teardown.result(timeout=5)
        assert context.source_volume is None and context._closed
        assert not context._caches and not context._cache_entries
        assert not list((tmp_path / 'runtime' / 'sam_image_cache').glob('*.gray8.dat'))
    finally:
        release.set()
        context.close()


def test_completed_scope_publication_does_not_block_next_cache_or_tracker_admission(tmp_path, monkeypatch):
    from XTA.view_prepare import SamLayerProjectionSubmitter
    from tests.test_sam_compute_ack_adversarial import _protocol

    tracker, request, worker, _coordinator, resident, _aux, handbacks = _protocol(tmp_path, monkeypatch)
    source = np.arange(3 * 15 * 21, dtype=np.uint16).reshape(3, 15, 21).astype(np.uint8)
    context = context_for(tmp_path / 'context', source, detector_identity='detector', bundle_identity='sam')
    source_owner = weakref.ref(source)
    source_shape = source.shape
    transverse, coronal = geometry.get_view_infos(*source.shape, cartesian_views=('transverse', 'coronal'))
    native = np.zeros(source.shape, np.uint8)
    original_results = list(tracker.iter_results((request,)))
    for frame, mask in original_results[0][1].frames.items():
        native[frame, 3:12, 2:15] = mask
    selected_store = tmp_path / 'selected.cvol'
    interpolation.write_raw_bbox_mask_store(native, selected_store,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    entry = dict(path=str(selected_store), direction='forward', voxel_count=int(native.sum()),
        policy_hash='policy', group_ids=['original-family'])
    publishing, release = threading.Event(), threading.Event()
    queue = ComponentProjectionQueue(workers=1, max_pending=2,
        max_source_bytes=1024**3, max_working_bytes=5 * 1024**3)
    publication_contexts = []

    def publish(entry, **kwargs):
        publication_contexts.append(kwargs['sam_context'])
        publishing.set()
        assert release.wait(10), 'test did not release its first scope publisher'
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    submitter = SamLayerProjectionSubmitter(queue=queue, source_shape=source_shape,
        materialize_directional=publish, materialize_extrapolation=assembly.materialize_sam_extrapolation_view_layer)
    monkeypatch.setattr(assembly, 'nrrd_layer_sink', lambda: None)
    monkeypatch.setattr(assembly, 'final_source_output_shape', lambda: source_shape)
    prepared = interpolation.PreparedViewResult('detector', transverse.name, '0', 0., None, None, [])
    try:
        child = submitter(entry, layer_kind='interpolation', model_name='detector', view=transverse,
            source='fullframe', pass_index=1, sam_context=context, workers=1)
        prepared.pending_component_layers.append(child)
        assert publishing.wait(3)
        entry['policy_hash'] = 'changed-after-submission'
        entry['group_ids'].append('changed-after-submission')
        context.detector_identity = context.bundle_identity = 'changed-after-submission'
        assert not settle_prepared_view_components(prepared) and not prepared.nrrd_layers
        shape = (coronal.num_slices, coronal.src_h, coronal.src_w)
        box = (0, 2, 3, 15)
        reference = context.image_provider(coronal, shape, demand(shape, {frame: box for frame in range(3)}))
        expected_images = source.transpose(2, 0, 1)[:3, :3, 2:15].copy()
        for frame in range(3):
            np.testing.assert_array_equal(crop_from(reference, frame, box), expected_images[frame])
        seed = np.zeros((3, 13), np.bool_)
        seed[1:, 2:10] = True
        next_request = dict(run_id='next-scope', seed_mask=seed, seed_frame=0,
            frame_start=0, frame_stop=3, direction='forward', crop_xyxy=(2, 0, 15, 3))
        results = list(tracker.iter_results((next_request,), source_cache_ref=reference))
        assert results[0][1].receipt['run_id'] == 'next-scope'
        assert all(np.array_equal(mask, seed) for mask in results[0][1].frames.values())
        assert len(handbacks) == 2 and not worker.active
        assert not child.done(), 'the next verified tracker completion must precede prior publication'
        snapshot = queue.snapshot()
        assert snapshot['pending'] == snapshot['active'] == 1
        assert snapshot['source_bytes'] > 0 and snapshot['working_bytes'] == 256 * 1024**2
        tracker.release_source_cache(reference)
        context.close()
        source = None
        gc.collect()
        assert source_owner() is None, 'detached publication retained a live source owner'
        assert publication_contexts[0] is not context
        assert publication_contexts[0].source_volume.shape == source_shape
        assert publication_contexts[0].detector_identity == 'detector'
        assert publication_contexts[0].bundle_identity == 'sam'
        release.set()
        child.result(timeout=5)
        assert settle_prepared_view_components(prepared)
        assert len(prepared.nrrd_layers) == 1 and not prepared.pending_component_layers
        assert prepared.nrrd_layers[0].interpolation_policy_identity == 'policy'
        assert prepared.nrrd_layers[0].sam_group_ids == ('original-family',)
        output = RuntimeLayer(prepared.nrrd_layers[0], source_shape)
        try:
            np.testing.assert_array_equal(output.read_slab(0, source_shape[0]), native)
        finally:
            output.close()
        queue.shutdown()
        settled = queue.snapshot()
        assert settled['pending'] == settled['source_bytes'] == settled['active'] == settled['working_bytes'] == 0
    finally:
        release.set()
        context.close()
        tracker.close()
        resident.release(residency_settled=True)
        queue.shutdown(cancel_futures=True)


def test_publication_abort_wakes_blocked_parent_and_never_commits_partial_view(tmp_path, monkeypatch):
    from XTA.view_prepare import SamLayerProjectionSubmitter

    source = np.arange(3 * 6 * 7, dtype=np.uint8).reshape(3, 6, 7)
    context = context_for(tmp_path, source, detector_identity='detector', bundle_identity='sam')
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    native = np.zeros(source.shape, np.uint8)
    native[1, 2:4, 3:5] = 1
    selected_store = tmp_path / 'selected.cvol'
    interpolation.write_raw_bbox_mask_store(native, selected_store,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    entry = dict(path=str(selected_store), direction='forward', voxel_count=int(native.sum()))
    working_bytes = 256 * 1024**2
    source_bytes = sum(path.stat().st_size for path in selected_store.iterdir() if path.is_file())
    queue = ComponentProjectionQueue(workers=2, max_pending=2,
        max_source_bytes=2 * source_bytes, max_working_bytes=working_bytes)
    publishing, release, producer_entered = threading.Event(), threading.Event(), threading.Event()
    materialized = []

    def publish(entry, **kwargs):
        materialized.append(entry['direction'])
        publishing.set()
        assert release.wait(10), 'test did not release its admitted publisher'
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    submitter = SamLayerProjectionSubmitter(queue=queue, source_shape=source.shape,
        materialize_directional=publish, materialize_extrapolation=assembly.materialize_sam_extrapolation_view_layer)
    monkeypatch.setattr(assembly, 'nrrd_layer_sink', lambda: None)
    monkeypatch.setattr(assembly, 'final_source_output_shape', lambda: native.shape)
    kwargs = dict(layer_kind='interpolation', model_name='detector', view=view,
        source='fullframe', pass_index=1, sam_context=context, workers=1)
    prepared = interpolation.PreparedViewResult('detector', view.name, '0', 0., None, None, [])

    def blocked_producer():
        producer_entered.set()
        return submitter(dict(entry, direction='backward'), **kwargs)

    try:
        first = submitter(entry, **kwargs)
        assert publishing.wait(3)
        second = submitter(dict(entry, direction='backward'), **kwargs)
        prepared.pending_component_layers.extend((first, second))
        with ThreadPoolExecutor(max_workers=1) as parents:
            blocked = parents.submit(blocked_producer)
            try:
                assert producer_entered.wait(3)
                assert not blocked.done()
                snapshot = queue.snapshot()
                assert snapshot['pending'] == 2 and snapshot['source_bytes'] == 2 * source_bytes
                assert snapshot['active'] == 1 and snapshot['working_bytes'] == working_bytes
                assert not settle_prepared_view_components(prepared)
                queue.abort()
                with pytest.raises(RuntimeError, match='closed|failed'):
                    blocked.result(timeout=3)
                with pytest.raises(RuntimeError, match='aborted before admission'):
                    second.result(timeout=3)
                assert not first.done()
                with pytest.raises(RuntimeError, match='aborted before admission'):
                    settle_prepared_view_components(prepared)
                assert not prepared.nrrd_layers
                assert selected_store.is_dir(), 'cancellation must preserve a running publication source'
            finally:
                release.set()
        first.result(timeout=5)
        queue.shutdown()
        assert materialized == ['forward']
        with pytest.raises(RuntimeError, match='aborted before admission'):
            settle_prepared_view_components(prepared)
        assert not prepared.nrrd_layers
        snapshot = queue.snapshot()
        assert snapshot['submitted'] == snapshot['completed'] == 2 and snapshot['failed'] == 1
        assert snapshot['peak_pending'] == 2 and snapshot['peak_source_bytes'] == 2 * source_bytes
        assert snapshot['peak_active'] == 1 and snapshot['peak_working_bytes'] == working_bytes
        assert snapshot['pending'] == snapshot['source_bytes'] == snapshot['active'] == snapshot['working_bytes'] == 0
    finally:
        release.set()
        context.close()
        queue.shutdown(cancel_futures=True)
