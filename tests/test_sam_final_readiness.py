"""Real parent preparation and publication drain under a saturated bounded queue."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from queue import Empty, SimpleQueue
from threading import Event, Lock, current_thread, get_ident
import time

import numpy as np

from XTA import assembly, geometry
from XTA.config import TiltedViewGroup
from XTA.interpolation import _ByteAdmissionPool, _DirectUnionBackingLease, RawBBoxMaskStore
from XTA.projection_queue import ComponentProjectionQueue, prepared_view_waitables, settle_prepared_view_components
from XTA.view_prepare import SamLayerProjectionSubmitter, ViewPrepareLeaseState
from tests.test_sam_async_layer_publication import SelectedContext, owned_task


def test_sixty_real_parents_drain_after_legacy_projection_consumer_recovers(tmp_path, monkeypatch):
    for flag in ('CUDA_VISIBLE_DEVICES',):
        monkeypatch.setenv(flag, '-1')
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT', 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'):
        monkeypatch.setenv(flag, '0')
    gib = 1024**3
    monkeypatch.setattr('XTA.sam_resources.physical_sam_headroom', lambda: 2*1024*gib)
    target = (7, 9, 11)
    views = geometry.get_view_infos(*target, cartesian_views=(),
        tilt_groups=[TiltedViewGroup(('transverse',), (20.,), ('vertical',))],
        radial_views=('transverse', 'tilted_transverse'), radial_min_radius=.5, radial_patch_size=5)
    bases = tuple(next(view for view in views if view.family == 'radial'
        and bool(view.radial_tilted_source) == tilted) for tilted in (False, True))
    # This is the guarded legacy Transverse path, never the new compact adapter.
    monkeypatch.setattr('XTA.sam_sparse_shell_projection.project_shell_sparse_store',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Transverse entered compact path')))
    release = Event()
    release.set()
    waiters, lock, all_producers_waiting = set(), Lock(), Event()
    projected = []
    queue = ComponentProjectionQueue(workers=2, max_pending=8, max_source_bytes=16*gib,
        max_working_bytes=96*gib, working_budget=lambda:96*gib)
    original_wait = queue._condition.wait
    def observe_wait(*args, **kwargs):
        if current_thread().name.startswith('readiness-parent') and not release.is_set():
            with lock:
                waiters.add(get_ident())
                if len(waiters) == 21:
                    all_producers_waiting.set()
        return original_wait(*args, **kwargs)
    queue._condition.wait = observe_wait
    def materialize(kind, operation):
        def run(entry, **kwargs):
            assert release.wait(30), 'Independent test consumer was not released'
            result = operation(entry, **kwargs)
            with lock:
                projected.append((kwargs['view'].name, kind, entry['direction']))
            return result
        return run
    submitter = SamLayerProjectionSubmitter(queue, target,
        materialize('interpolation', assembly.materialize_sam_directional_view_layer),
        materialize('extrapolation', assembly.materialize_sam_extrapolation_view_layer))
    admission = _ByteAdmissionPool(21*32*gib, 'final readiness parent grants')
    leases = ViewPrepareLeaseState({}, set(), {}, set(), {})
    retired_events, retired, contexts = SimpleQueue(), set(), []
    parents = ThreadPoolExecutor(max_workers=21, thread_name_prefix='readiness-parent')
    assembly.set_final_source_output_shape(target)
    def submit_batch(first, stop):
        futures = {}
        for index in range(first, stop):
            view = replace(bases[index % 2], name=f'readiness-{index:02d}')
            context = SelectedContext(tmp_path/f'evidence-{index:02d}', target)
            contexts.append(context)
            task = owned_task(tmp_path/f'parent-{index:02d}', view, context, submitter, admission)
            task.preinterpolation_layer_already_published = True
            key = ('detector', view.name)
            amount = task.union_mm.nbytes
            leases.leases[key] = _DirectUnionBackingLease(key, amount, phase='postprocess')
            task.backing_lease = leases.leases[key]
            leases.postprocess_views.add(key)
            leases.postprocess_bytes[key] = amount
            future = parents.submit(task)
            task.track_cancellation(future)
            futures[future] = key
        return futures
    def drain(futures):
        deadline = time.monotonic()+60
        while futures:
            while True:
                try:
                    key, lease = retired_events.get_nowait()
                except Empty:
                    break
                leases.settle_publication_retirement(key, lease)
            for future, key in list(futures.items()):
                if not future.done():
                    continue
                prepared = future.result()
                if not settle_prepared_view_components(prepared):
                    leases.retire_dense_for_publication(prepared, enabled=True, tiled=False,
                        keep_temp=False, retired_callback=lambda *args:retired_events.put(args))
                    continue
                expected = [('bridge', 'forward'), ('bridge', 'backward'),
                            ('extrapolation', 'forward'), ('extrapolation', 'backward')]
                assert [(ref.mask_kind, ref.interpolation_direction or ref.extrapolation_provenance['direction'])
                        for ref in prepared.nrrd_layers] == expected
                for ref in prepared.nrrd_layers:
                    store = RawBBoxMaskStore.open(ref.path)
                    try:
                        assert store.shape == target
                        for frame in range(target[0]):
                            assert set(np.unique(store.decode_slice(frame))) <= {0, 1}
                    finally:
                        store.close()
                assert key not in retired
                retired.add(key)
                leases.complete(key, retain_for_dense_retirement=False)
                assert not leases.complete(key, retain_for_dense_retirement=False)
                del futures[future]
            if futures:
                assert time.monotonic() < deadline, 'Production-linked parent drain stopped progressing'
                wait(prepared_view_waitables(futures), timeout=.1, return_when=FIRST_COMPLETED)
    try:
        drain(submit_batch(0, 28))
        assert len(retired) == 28 and queue.snapshot()['pending'] == 0
        release.clear()
        pending = submit_batch(28, 60)
        try:
            assert all_producers_waiting.wait(15), 'Did not reach the saturated 21-producer state'
            state = queue.snapshot()
            assert state['pending'] == 8 and state['active'] == 2
            assert len(retired) == 28 and len(pending) == 32
            assert any(not context.interpolation_entered.is_set() for context in contexts[28:])
            assert not all(future.done() for future in pending)
        finally:
            release.set()
        drain(pending)
        parents.shutdown()
        queue.shutdown()
        assert len(retired) == 60 and len(projected) == len(set(projected)) == 240
        assert all(context.extrapolation_entered.is_set() for context in contexts)
        assert not leases.leases and not leases.postprocess_views and not leases.postprocess_bytes
        assert admission.in_use == 0
        state = queue.snapshot()
        assert state['submitted'] == state['completed'] == 240
        assert state['peak_pending'] <= 8 and state['peak_active'] <= 2
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
    finally:
        release.set()
        queue.abort()
        parents.shutdown(cancel_futures=True)
        queue.shutdown(cancel_futures=True)
        assembly.set_final_source_output_shape(None)
