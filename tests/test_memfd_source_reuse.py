"""Source-transfer reuse must preserve descriptor identity and dispatch ownership."""
from __future__ import annotations

from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import runtime
from XTA.tta_scheduler import TtaScheduler, TtaSchedulerState, _WorkerMemfdSourceCache


class _LocalDupFd:
    """Real descriptor ownership with a local transport for non-Linux hosts."""

    def __init__(self, fd):
        self.fd = os.dup(fd)
        self.original_duplicate = self.fd

    def detach(self):
        if self.fd is None:
            raise RuntimeError('transfer already consumed')
        fd, self.fd = self.fd, None
        return fd


@contextmanager
def _owner(path, data=b'source pixels'):
    with tempfile.TemporaryFile() as backing:
        backing.write(data)
        backing.flush()
        owned = os.dup(backing.fileno())
        runtime._register_memfd_owner(path, owned, 'source reuse test')
        try:
            yield owned
        finally:
            runtime._release_memfd_owner_key(path)


def _scheduler(*, queue_obj=None, preflight=None):
    state = TtaSchedulerState(
        baseline_union_paths={}, baseline_confmap_paths={}, parent_mask_support_by_model={},
        fullframe_remaining={}, direct_union_inference_views=set(),
        direct_union_postprocess_views=set(), direct_union_inference_bytes={},
        direct_union_postprocess_bytes={}, direct_union_backing_leases={},
    )
    state.gpu_task_queues[0] = queue.Queue() if queue_obj is None else queue_obj
    state.gpu_task_queues[1] = queue.Queue()
    state.cpu_task_queues[0] = queue.Queue()
    operations = SimpleNamespace(
        _attach_memfd_transfers_to_task=runtime._attach_memfd_transfers_to_task,
        preflight_multiprocessing_payload=preflight or runtime.preflight_multiprocessing_payload,
    )
    return TtaScheduler(inputs=None, state=state, operations=operations)


def _spawn_materializer(tasks, results):
    persistent = {}
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            transient = []
            try:
                transient = runtime._materialize_worker_task_memfd_paths(task, persistent)
                with open(task['source_volume_path'], 'rb') as handle:
                    source = handle.read()
                if task.get('result_mask_path'):
                    with open(task['result_mask_path'], 'r+b') as handle:
                        handle.write(b'Z')
                results.put({'task_id': task['task_id'], 'source': source,
                             'path': task['source_volume_path'], 'sources': len(persistent)})
            finally:
                runtime._close_fd_list(transient)
    finally:
        fds = tuple(persistent.values())
        runtime._close_fd_list(fds)
        closed = 0
        for fd in fds:
            try:
                os.fstat(fd)
            except OSError:
                closed += 1
        results.put({'closed_sources': closed})


class MemfdSourceReuseTests(unittest.TestCase):
    def setUp(self):
        self.dup = mock.patch.object(runtime.mp_reduction, 'DupFd', _LocalDupFd, create=True)
        self.dup.start()
        self.addCleanup(self.dup.stop)
        self.persistent = {}
        self.addCleanup(lambda: runtime._close_fd_list(self.persistent.values()))

    def assertClosed(self, fd):
        with self.assertRaises(OSError):
            os.fstat(fd)

    def materialize(self, task):
        transient = runtime._materialize_worker_task_memfd_paths(task, self.persistent)
        self.addCleanup(runtime._close_fd_list, transient)
        return transient

    def test_preorder_nested_references_transfer_each_source_once_and_outputs_independently(self):
        with _owner('source') as source, _owner('result'):
            child = {'source_volume_path': 'source', 'result_mask_path': 'result',
                     'native_resize': {'path': 'source'}}
            task = {'source_volume_path': 'source', 'native_resize': {'path': 'source'},
                    'result_mask_path': 'result', 'augmentation_pass_tasks': [child]}
            batch = runtime._attach_memfd_transfers_to_task(task, known_sources=())
            self.assertEqual(len(batch.handles), 3)
            key = runtime._memfd_source_identity(source)
            self.assertEqual(batch.source_keys, (key,))
            self.assertNotIn('source_volume_fd', child)
            self.assertNotIn('path_fd_ref', child['native_resize'])
            self.assertEqual(task['augmentation_pass_tasks'][0]['source_volume_fd_ref'], key)
            transient = self.materialize(task)
            self.assertEqual(len(transient), 2)
            self.assertNotEqual(*transient)
            self.assertEqual(len(self.persistent), 1)
            self.assertEqual(task['source_volume_path'], task['native_resize']['path'])
            self.assertEqual(task['source_volume_path'], task['augmentation_pass_tasks'][0]['source_volume_path'])

    def test_acknowledged_source_reuses_owner_but_keeps_transient_transfers(self):
        with _owner('source'), _owner('result'):
            first = {'source_volume_path': 'source'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            second = {'source_volume_path': 'source', 'result_mask_path': 'result',
                      'result_conf_path': 'result', 'canvas_path': 'result', 'd1_bitset_path': 'result'}
            repeated = runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.assertEqual(len(repeated.handles), 4)
            self.assertEqual(len(self.materialize(second)), 4)
            self.assertEqual(second['source_volume_path'], first['source_volume_path'])
            os.fstat(next(iter(self.persistent.values())))

    def test_same_parent_fd_number_with_different_inode_gets_new_child_owner(self):
        with _owner('source', b'old') as owner_fd, _owner('other', b'new') as other_fd:
            first = {'source_volume_path': 'source'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            old_child_fd = next(iter(self.persistent.values()))
            os.dup2(other_fd, owner_fd)
            second = {'source_volume_path': 'source'}
            replaced = runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.assertNotEqual(batch.source_keys, replaced.source_keys)
            self.assertEqual(len(replaced.handles), 1)
            self.materialize(second)
            self.assertNotEqual(first['source_volume_path'], second['source_volume_path'])
            self.assertEqual(len(self.persistent), 2)
            os.lseek(old_child_fd, 0, os.SEEK_SET)
            self.assertEqual(os.read(old_child_fd, 3), b'old')

    def test_streamed_content_changes_do_not_invalidate_same_storage(self):
        with _owner('source', b'old') as fd:
            first = {'source_volume_path': 'source'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b'new')
            second = {'source_volume_path': 'source'}
            again = runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.assertEqual(again.handles, [])
            self.materialize(second)
            self.assertEqual(first['source_volume_path'], second['source_volume_path'])

    def test_resized_storage_gets_a_new_identity_without_closing_old_child_fd(self):
        with _owner('source', b'old') as fd:
            first = {'source_volume_path': 'source'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            old_child_fd = next(iter(self.persistent.values()))
            os.ftruncate(fd, 40)
            second = {'source_volume_path': 'source'}
            resized = runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.assertNotEqual(batch.source_keys, resized.source_keys)
            self.assertEqual(len(resized.handles), 1)
            self.materialize(second)
            self.assertNotEqual(first['source_volume_path'], second['source_volume_path'])
            os.fstat(old_child_fd)

    def test_missing_cache_reference_fails_and_drains_unvisited_transfer(self):
        with _owner('source') as fd, _owner('result'):
            task = {'source_volume_path': 'source', 'result_mask_path': 'result'}
            batch = runtime._attach_memfd_transfers_to_task(
                task, known_sources=(runtime._memfd_source_identity(fd),))
            transferred = batch.handles[0].original_duplicate
            with self.assertRaisesRegex(RuntimeError, 'no matching persistent'):
                self.materialize(task)
            self.assertClosed(transferred)
            self.assertFalse(self.persistent)

    def test_wrong_received_identity_is_closed_and_not_cached(self):
        with _owner('source') as fd, _owner('other') as other:
            handle = _LocalDupFd(other)
            task = {'source_volume_fd': handle, 'source_volume_fd_key': runtime._memfd_source_identity(fd)}
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                self.materialize(task)
            self.assertClosed(handle.original_duplicate)
            self.assertFalse(self.persistent)

    def test_reference_to_recycled_child_fd_fails_closed(self):
        with _owner('source') as fd, _owner('other') as other:
            cached = os.dup(fd)
            key = runtime._memfd_source_identity(fd)
            self.persistent[key] = cached
            os.dup2(other, cached)
            with self.assertRaisesRegex(RuntimeError, 'no matching persistent'):
                self.materialize({'source_volume_fd_ref': key})

    def test_transient_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid cached memfd reference'):
            self.materialize({'result_mask_fd_ref': 'memfd-source-v1:1:2:3'})

    def test_attach_failure_drains_handles_registered_before_failure(self):
        handles = []
        def duplicate(fd):
            if handles:
                raise OSError('registration failed')
            result = _LocalDupFd(fd)
            handles.append(result)
            return result
        with _owner('source'), _owner('result'), mock.patch.object(runtime.mp_reduction, 'DupFd', duplicate):
            with self.assertRaisesRegex(OSError, 'registration failed'):
                runtime._attach_memfd_transfers_to_task(
                    {'source_volume_path': 'source', 'result_mask_path': 'result'}, known_sources=())
        self.assertClosed(handles[0].original_duplicate)

    def test_materialization_failure_closes_new_owners_and_later_nested_handles(self):
        with _owner('source'), _owner('other'):
            first = {'source_volume_path': 'source'}
            original = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            existing = dict(self.persistent)
            task = {'source_volume_path': 'other', 'augmentation_pass_tasks': [
                {'source_volume_fd_ref': 'memfd-source-v1:missing'},
                {'result_mask_path': 'other'},
            ]}
            batch = runtime._attach_memfd_transfers_to_task(task, known_sources=original.source_keys)
            fds = [h.original_duplicate for h in batch.handles]
            with self.assertRaisesRegex(RuntimeError, 'no matching persistent'):
                self.materialize(task)
            self.assertEqual(self.persistent, existing)
            for fd in fds:
                self.assertClosed(fd)
            os.fstat(next(iter(existing.values())))

    def test_materialization_counters_distinguish_detach_from_reference_hits(self):
        telemetry = SimpleNamespace(add=mock.Mock())
        with _owner('source'), mock.patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry):
            first = {'source_volume_path': 'source'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            self.materialize(first)
            second = {'source_volume_path': 'source'}
            runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.materialize(second)
        totals = {}
        for call in telemetry.add.call_args_list:
            name, value = call.args
            totals[name] = totals.get(name, 0) + value
        self.assertEqual(totals['worker.memfd_materialize.calls'], 2)
        self.assertEqual(totals['worker.memfd_materialize.detach_calls'], 1)
        self.assertEqual(totals['worker.memfd_materialize.source_reference_hits'], 1)
        self.assertGreaterEqual(totals['worker.memfd_materialize.seconds'],
                                totals['worker.memfd_materialize.detach_seconds'])

    def test_dispatch_waits_for_success_ack_and_validates_worker(self):
        scheduler = _scheduler()
        with _owner('source'):
            for tid in (1, 2):
                scheduler._put_worker_inference_task({'task_id': tid, 'source_volume_path': 'source'}, 'gpu', 0)
            tasks = [scheduler.state.gpu_task_queues[0].get_nowait() for _ in range(2)]
            self.assertTrue(all('source_volume_fd' in task for task in tasks))
            for task in tasks:
                self.materialize(task)
            cache = scheduler._worker_memfd_source_cache('gpu', 0)
            self.assertFalse(cache.known)
            scheduler._record_worker_memfd_completion({'type': 'compute_released', 'gpu_index': 1,
                                                      'task_id': 1, 'ok': True})
            self.assertFalse(cache.known)
            scheduler._record_worker_memfd_completion({'type': 'result', 'gpu_index': 0,
                                                      'task_id': 1, 'ok': False})
            self.assertFalse(cache.known)
            scheduler._record_worker_memfd_completion({'type': 'compute_released', 'gpu_index': 0,
                                                      'task_id': 2, 'ok': True})
            self.assertTrue(cache.known)
            scheduler._put_worker_inference_task({'task_id': 3, 'source_volume_path': 'source'}, 'gpu', 0)
            third = scheduler.state.gpu_task_queues[0].get_nowait()
            self.assertIn('source_volume_fd_ref', third)
            self.materialize(third)
            scheduler._record_worker_memfd_completion({'type': 'result', 'gpu_index': 0,
                                                      'task_id': 3, 'ok': True})
            self.assertFalse(cache.pending)

    def test_cpu_final_result_also_acknowledges_sources(self):
        scheduler = _scheduler()
        with _owner('source'):
            scheduler._put_worker_inference_task({'task_id': 1, 'source_volume_path': 'source'}, 'cpu', 0)
            self.materialize(scheduler.state.cpu_task_queues[0].get_nowait())
            scheduler._record_worker_memfd_completion({'type': 'result', 'worker_kind': 'cpu',
                                                      'cpu_index': 0, 'task_id': 1, 'ok': True})
            self.assertTrue(scheduler._worker_memfd_source_cache('cpu', 0).known)
            self.assertFalse(scheduler._worker_memfd_source_cache('gpu', 0).known)

    def test_successful_compute_ack_is_available_before_immediate_scheduler_refill(self):
        scheduler = _scheduler()
        scheduler.operations._main_process_gpu_stage_finish_inference = mock.Mock()
        scheduler.refresh_gpu_aux_interpolation_leases = mock.Mock()
        with _owner('source'):
            task = {'task_id': 1, 'source_volume_path': 'source'}
            scheduler._put_worker_inference_task(task, 'gpu', 0)
            self.materialize(scheduler.state.gpu_task_queues[0].get_nowait())
            cache = scheduler._worker_memfd_source_cache('gpu', 0)
            scheduler.dispatch_inference_windows = mock.Mock(side_effect=lambda _preferred:
                                                             self.assertTrue(cache.known))
            scheduler.process_one_worker_result({'type': 'compute_released', 'gpu_index': 0,
                                                  'task_id': 1, 'ok': True})
            scheduler.dispatch_inference_windows.assert_called_once()
            self.assertFalse(cache.pending)

    def test_real_pickle_failure_then_retry_does_not_reuse_uninstalled_source(self):
        scheduler = _scheduler()
        with _owner('source'):
            failed = {'task_id': 1, 'source_volume_path': 'source', 'callback': lambda: None}
            with self.assertRaisesRegex(TypeError, 'not serializable'):
                scheduler._put_worker_inference_task(failed, 'gpu', 0)
            self.assertClosed(failed['source_volume_fd'].original_duplicate)
            retry = {'task_id': 1, 'source_volume_path': 'source'}
            scheduler._put_worker_inference_task(retry, 'gpu', 0)
            sent = scheduler.state.gpu_task_queues[0].get_nowait()
            self.assertIn('source_volume_fd', sent)
            self.assertNotIn('source_volume_fd_ref', sent)
            self.materialize(sent)
            scheduler._record_worker_memfd_completion({'type': 'result', 'gpu_index': 0,
                                                      'task_id': 1, 'ok': True})
            self.assertTrue(scheduler._worker_memfd_source_cache('gpu', 0).known)

    def test_queue_put_and_pickle_failure_roll_back_registration_without_ack(self):
        for fail_at in ('put', 'pickle'):
            with self.subTest(fail_at=fail_at), _owner('source'):
                receiver = queue.Queue()
                if fail_at == 'put':
                    receiver.put = mock.Mock(side_effect=RuntimeError('queue failure'))
                preflight = (mock.Mock(side_effect=RuntimeError('pickle failure'))
                             if fail_at == 'pickle' else None)
                scheduler = _scheduler(queue_obj=receiver, preflight=preflight)
                task = {'task_id': 1, 'source_volume_path': 'source'}
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    scheduler._put_worker_inference_task(task, 'gpu', 0)
                handle = task['source_volume_fd']
                self.assertClosed(handle.original_duplicate)
                cache = scheduler._worker_memfd_source_cache('gpu', 0)
                self.assertFalse(cache.known)
                self.assertFalse(cache.pending)
                scheduler._record_worker_memfd_completion({'type': 'result', 'gpu_index': 0,
                                                          'task_id': 1, 'ok': True})
                self.assertFalse(cache.known)

    def test_queue_pid_and_run_reset_do_not_reuse_old_worker_ack(self):
        scheduler = _scheduler()
        cache = scheduler._worker_memfd_source_cache('gpu', 0)
        cache.known['old'] = None
        cache.pending[1] = ('pending',)
        scheduler._record_worker_memfd_completion({'type': 'ready', 'gpu_index': 0, 'pid': 100})
        self.assertTrue(cache.known)
        scheduler._record_worker_memfd_completion({'type': 'ready', 'gpu_index': 0, 'pid': 101})
        self.assertFalse(cache.known)
        self.assertFalse(cache.pending)
        cache.known['old'] = None
        scheduler.state.gpu_task_queues[0] = queue.Queue()
        replacement = scheduler._worker_memfd_source_cache('gpu', 0)
        self.assertIsNot(cache, replacement)
        self.assertFalse(replacement.known)
        self.assertFalse(_scheduler().state.worker_memfd_sources)
        scheduler.shutdown_inference_worker_processes()
        self.assertFalse(scheduler.state.worker_memfd_sources)

    def test_ack_metadata_is_bounded_without_closing_child_sources(self):
        cache = _WorkerMemfdSourceCache(queue.Queue())
        with _owner('source') as fd:
            child = os.dup(fd)
            self.addCleanup(runtime._close_fd_list, (child,))
            for index in range(90):
                cache.pending[index] = (f'source-{index}',)
                cache.completed(index, ok=True)
            self.assertEqual(len(cache.known), 64)
            self.assertNotIn('source-0', cache.known)
            self.assertFalse(cache.pending)
            os.fstat(child)


@unittest.skipUnless(hasattr(os, 'memfd_create') and Path('/proc/self/fd').exists(),
                     'real resource_sharer/memfd spawn transport requires Linux procfs')
class LinuxMemfdSourceReuseTests(unittest.TestCase):
    @staticmethod
    def settled_fd_count():
        previous, stable = None, 0
        for _ in range(100):
            count = len(list(Path('/proc/self/fd').iterdir()))
            stable = stable + 1 if count == previous else 0
            if stable >= 3:
                return count
            previous = count
            time.sleep(0.01)
        raise AssertionError('resource-sharer descriptor count did not settle')

    def test_spawn_reuse_survives_parent_close_and_releases_child_descriptors(self):
        context = mp.get_context('spawn')
        tasks, results = context.Queue(), context.Queue()
        worker = context.Process(target=_spawn_materializer, args=(tasks, results))
        source = os.memfd_create('reuse-source', flags=getattr(os, 'MFD_CLOEXEC', 0))
        output = os.memfd_create('reuse-output', flags=getattr(os, 'MFD_CLOEXEC', 0))
        os.write(source, b'pixels')
        os.write(output, b'.')
        runtime._register_memfd_owner('spawn-source', source, 'spawn test')
        runtime._register_memfd_owner('spawn-output', output, 'spawn test')
        worker.start()
        try:
            first = {'task_id': 1, 'source_volume_path': 'spawn-source', 'result_mask_path': 'spawn-output'}
            batch = runtime._attach_memfd_transfers_to_task(first, known_sources=())
            runtime.preflight_multiprocessing_payload(first)
            tasks.put(first)
            initial = results.get(timeout=20)
            self.assertEqual(initial['source'], b'pixels')
            second = {'task_id': 2, 'source_volume_path': 'spawn-source', 'result_mask_path': 'spawn-output'}
            again = runtime._attach_memfd_transfers_to_task(second, known_sources=batch.source_keys)
            self.assertEqual(len(again.handles), 1)
            tasks.put(second)
            repeated = results.get(timeout=20)
            self.assertEqual(initial['path'], repeated['path'])
            self.assertEqual(repeated['sources'], 1)
            os.lseek(output, 0, os.SEEK_SET)
            self.assertEqual(os.read(output, 1), b'Z')
            runtime._release_memfd_owner_key('spawn-source')
            tasks.put({'task_id': 3, 'source_volume_fd_ref': batch.source_keys[0]})
            self.assertEqual(results.get(timeout=20)['source'], b'pixels')
            tasks.put(None)
            self.assertEqual(results.get(timeout=20)['closed_sources'], 1)
            worker.join(timeout=20)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
            runtime._release_memfd_owner_key('spawn-source')
            runtime._release_memfd_owner_key('spawn-output')
            tasks.close()
            results.close()
            tasks.join_thread()
            results.join_thread()

    def test_real_sharer_registration_rollback_has_no_fd_leak(self):
        descriptor = os.memfd_create('reuse-rollback', flags=getattr(os, 'MFD_CLOEXEC', 0))
        runtime._register_memfd_owner('rollback-source', descriptor, 'rollback test')
        try:
            # Warm the resource-sharer listener, whose own sockets are persistent.
            runtime._attach_memfd_transfers_to_task(
                {'source_volume_path': 'rollback-source'}, known_sources=()).rollback()
            baseline = self.settled_fd_count()
            for _ in range(20):
                batch = runtime._attach_memfd_transfers_to_task(
                    {'source_volume_path': 'rollback-source'}, known_sources=())
                batch.rollback()
                batch.rollback()
            self.assertEqual(self.settled_fd_count(), baseline)
            os.fstat(descriptor)
        finally:
            runtime._release_memfd_owner_key('rollback-source')
