"""Prepared input leases use proven procfs access without descriptor handshakes."""
from contextlib import ExitStack
import multiprocessing as mp
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import runtime
from tests.test_memfd_source_reuse import _LocalDupFd, _owner, _scheduler

_ORIGINAL_OPEN = os.open


def _local_open(path, flags, *args, **kwargs):
    match = re.fullmatch(r'/proc/' + str(os.getpid()) + r'/fd/([0-9]+)', str(path).replace('\\', '/'))
    if match is None and not str(path).startswith('/proc/'):
        return _ORIGINAL_OPEN(path, flags, *args, **kwargs)
    if match is None or flags & os.O_WRONLY:
        raise OSError('unsupported proc access')
    # The portable shim checks flags; the Linux spawn test proves OS read-only access.
    return os.dup(int(match.group(1)))


def _spawn_input_receiver(tasks, results):
    persistent = {}
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            transient = runtime._materialize_worker_task_memfd_paths(task, persistent)
            try:
                reply = dict(task_id=task['task_id'], capability=task.get('_memfd_parent_proc_capability'))
                if task.get('cpu_input_path'):
                    with open(task['cpu_input_path'], 'rb') as handle:
                        reply['pixels'] = handle.read()
                    try:
                        os.write(transient[0], b'!')
                    except OSError:
                        reply['read_only'] = True
                    else:
                        reply['read_only'] = False
            finally:
                runtime._close_fd_list(transient)
            reply['closed_inputs'] = 0
            for fd in transient:
                try:
                    os.fstat(fd)
                except OSError:
                    reply['closed_inputs'] += 1
            results.put(reply)
    finally:
        runtime._close_fd_list(persistent.values())


class MemfdInputHandoffTests(unittest.TestCase):
    def setUp(self):
        self.dups = mock.patch.object(runtime.mp_reduction, 'DupFd', side_effect=_LocalDupFd, create=True)
        self.duplicate = self.dups.start()
        self.addCleanup(self.dups.stop)
        self.persistent = {}
        self.addCleanup(lambda:runtime._close_fd_list(self.persistent.values()))
        self.scheduler = _scheduler()
        self.scheduler.state.gpu_worker_processes = [SimpleNamespace(name='gpu-worker-0', pid=os.getpid())]

    def _grant_capability(self, source_fd):
        task = dict(task_id=1, source_volume_path=runtime._memfd_proc_path(source_fd).as_posix())
        self.scheduler._put_worker_inference_task(task, 'gpu', 0)
        actual = self.scheduler.state.gpu_task_queues[0].get_nowait()
        runtime._close_fd_list(runtime._materialize_worker_task_memfd_paths(actual, self.persistent))
        self.scheduler._record_worker_memfd_completion(dict(type='compute_released', gpu_index=0,
            task_id=1, ok=True, memfd_parent_proc_capability=actual['_memfd_parent_proc_capability']))
        return self.scheduler._worker_memfd_source_cache('gpu', 0)

    def _assert_closed(self, fd):
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_160_packets_use_zero_additional_sharer_registrations_and_read_only_opens(self):
        telemetry = SimpleNamespace(add=mock.Mock(), enabled=False)
        with _owner(None) as source_fd, mock.patch.object(runtime.os, 'open', side_effect=_local_open) as opened:
            cache = self._grant_capability(source_fd)
            self.duplicate.reset_mock()
            opened.reset_mock()
            with mock.patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry):
                for task_id in range(2, 162):
                    with _owner(None, b'prepared pixels') as input_fd:
                        task = dict(task_id=task_id, source_volume_path=runtime._memfd_proc_path(source_fd).as_posix(),
                            cpu_input_path=runtime._memfd_proc_path(input_fd).as_posix())
                        self.scheduler._put_worker_inference_task(task, 'gpu', 0)
                        sent = self.scheduler.state.gpu_task_queues[0].get_nowait()
                        self.assertNotIn('cpu_input_fd', sent)
                        self.assertIn('cpu_input_fd_proc', sent)
                        transient = runtime._materialize_worker_task_memfd_paths(sent, self.persistent)
                        self.assertEqual(len(transient), 1)
                        self.assertEqual(os.fstat(transient[0]).st_size, len(b'prepared pixels'))
                        runtime._close_fd_list(transient)
                        self._assert_closed(transient[0])
                        os.fstat(input_fd)
                        self.scheduler._record_worker_memfd_completion(dict(type='compute_released', gpu_index=0,
                            task_id=task_id, ok=True, memfd_parent_proc_capability=sent['_memfd_parent_proc_capability']))
            self.duplicate.assert_not_called()
            read_flags = [call.args[1] for call in opened.call_args_list if call.args[1] & os.O_RDWR == 0]
            self.assertEqual(len(read_flags), 160)
            self.assertTrue(all(flags & os.O_WRONLY == 0 for flags in read_flags))
            self.assertEqual(len(self.persistent), 1)
            self.assertFalse(cache.pending)
            totals = {}
            for call in telemetry.add.call_args_list:
                name, value = call.args
                totals[name] = totals.get(name, 0) + value
            self.assertEqual(totals['worker.memfd_materialize.detach_calls'], 0)
            self.assertEqual(totals['worker.memfd_materialize.input_proc_open_calls'], 160)
            self.assertEqual(totals['worker.memfd_materialize.output_proc_open_calls'], 0)

    def test_unproven_or_replaced_worker_keeps_original_transfer_fallback(self):
        for invalid in ('unproven', 'parent', 'worker', 'pid'):
            with self.subTest(invalid=invalid), _owner(None, b'pixels') as input_fd:
                self.scheduler = _scheduler()
                self.scheduler.state.gpu_worker_processes = [SimpleNamespace(name='gpu-worker-0', pid=os.getpid())]
                cache = self.scheduler._worker_memfd_source_cache('gpu', 0)
                if invalid != 'unproven':
                    cache.direct_output_capability = dict(parent_pid=os.getpid(), worker_pid=os.getpid())
                    cache.direct_output_capability['parent_pid' if invalid=='parent' else 'worker_pid'] += 1
                    if invalid == 'pid':
                        self.scheduler.state.gpu_worker_processes[0].pid += 1
                task = dict(task_id=2, cpu_input_path=runtime._memfd_proc_path(input_fd).as_posix())
                self.scheduler._put_worker_inference_task(task, 'gpu', 0)
                sent = self.scheduler.state.gpu_task_queues[0].get_nowait()
                self.assertIn('cpu_input_fd', sent)
                self.assertNotIn('cpu_input_fd_proc', sent)
                transient = runtime._materialize_worker_task_memfd_paths(sent, self.persistent)
                self.assertEqual(len(transient), 1)
                runtime._close_fd_list(transient)
                self._assert_closed(transient[0])

    def test_recycled_input_identity_fails_closed_and_drains_later_transfer(self):
        opened = []
        def open_proc(path, flags):
            fd = _local_open(path, flags)
            opened.append(fd)
            return fd
        with _owner(None, b'original') as input_fd, _owner(None, b'wrong!!!') as other_fd:
            task = dict(cpu_input_path=runtime._memfd_proc_path(input_fd).as_posix(),
                d1_bitset_path=runtime._memfd_proc_path(other_fd).as_posix())
            batch = runtime._attach_memfd_transfers_to_task(task,
                direct_output_capability=dict(parent_pid=os.getpid(), worker_pid=os.getpid()))
            self.assertEqual(len(batch.handles), 1)
            os.dup2(other_fd, input_fd)
            with mock.patch.object(runtime.os, 'open', side_effect=open_proc):
                with self.assertRaisesRegex(RuntimeError, 'input identity changed'):
                    runtime._materialize_worker_task_memfd_paths(task, self.persistent)
            for fd in opened + [batch.handles[0].original_duplicate]:
                self._assert_closed(fd)
            os.lseek(other_fd, 0, os.SEEK_SET)
            self.assertEqual(os.read(other_fd, 8), b'wrong!!!')

    def test_input_open_failure_closes_previous_output_and_later_transfer(self):
        opened = []
        def open_proc(path, flags):
            if flags & os.O_RDWR == 0:
                raise PermissionError('input proc access lost')
            fd = _local_open(path, flags)
            opened.append(fd)
            return fd
        with _owner(None) as fd:
            path = runtime._memfd_proc_path(fd).as_posix()
            task = dict(result_mask_path=path, cpu_input_path=path, d1_bitset_path=path)
            batch = runtime._attach_memfd_transfers_to_task(task,
                direct_output_capability=dict(parent_pid=os.getpid(), worker_pid=os.getpid()))
            with mock.patch.object(runtime.os, 'open', side_effect=open_proc):
                with self.assertRaisesRegex(PermissionError, 'input proc access lost'):
                    runtime._materialize_worker_task_memfd_paths(task, self.persistent)
            for owned in opened + [batch.handles[0].original_duplicate]:
                self._assert_closed(owned)
            os.fstat(fd)

    def test_dispatch_failure_retains_parent_input_without_registering_a_transfer(self):
        for failure in ('pickle', 'queue'):
            with self.subTest(failure=failure), _owner(None) as source_fd, _owner(None) as input_fd, \
                    mock.patch.object(runtime.os, 'open', side_effect=_local_open):
                self.scheduler = _scheduler()
                self.scheduler.state.gpu_worker_processes = [SimpleNamespace(name='gpu-worker-0', pid=os.getpid())]
                self._grant_capability(source_fd)
                self.duplicate.reset_mock()
                descriptor = dict(task_id=2, cpu_input_path=runtime._memfd_proc_path(input_fd).as_posix())
                if failure == 'pickle':
                    descriptor['unpicklable'] = lambda:None
                    expected = TypeError
                else:
                    self.scheduler.state.gpu_task_queues[0].put = mock.Mock(side_effect=RuntimeError('queue stopped'))
                    expected = RuntimeError
                with self.assertRaises(expected):
                    self.scheduler._put_worker_inference_task(descriptor, 'gpu', 0)
                self.duplicate.assert_not_called()
                os.fstat(input_fd)
                self.assertFalse(self.scheduler._worker_memfd_source_cache('gpu', 0).pending)


@unittest.skipUnless(hasattr(os, 'memfd_create') and Path('/proc/self/fd').exists(),
    'actual parent procfs input access requires Linux')
class LinuxMemfdInputHandoffTests(unittest.TestCase):
    def test_spawn_reads_exact_inputs_and_enforces_read_only_fds(self):
        context = mp.get_context('spawn')
        tasks, results = context.Queue(), context.Queue()
        worker = context.Process(target=_spawn_input_receiver, args=(tasks, results))
        paths = []
        try:
            with ExitStack() as owners:
                for data in (b'source', b'frame 1', b'frame 2'):
                    fd = os.memfd_create('cpu-input-test', flags=getattr(os, 'MFD_CLOEXEC', 0))
                    os.write(fd, data)
                    path = runtime._memfd_proc_path(fd).as_posix()
                    runtime._register_memfd_owner(path, fd, 'CPU input spawn test')
                    owners.callback(runtime._release_memfd_owner_key, path)
                    paths.append(path)
                worker.start()
                initial = dict(task_id=1, source_volume_path=paths[0])
                batch = runtime._attach_memfd_transfers_to_task(initial, known_sources=())
                tasks.put(initial)
                reply = results.get(timeout=20)
                proof = reply['capability']
                self.assertEqual(proof['worker_pid'], worker.pid)
                self.assertEqual(proof['parent_pid'], os.getpid())
                for task_id, path, data in zip((2, 3), paths[1:], (b'frame 1', b'frame 2')):
                    task = dict(task_id=task_id, source_volume_path=paths[0], cpu_input_path=path)
                    direct = runtime._attach_memfd_transfers_to_task(task, known_sources=batch.source_keys,
                        direct_output_capability=dict(parent_pid=os.getpid(), worker_pid=worker.pid))
                    self.assertEqual(direct.handles, [])
                    tasks.put(task)
                    received = results.get(timeout=20)
                    self.assertEqual(received['pixels'], data)
                    self.assertTrue(received['read_only'])
                    self.assertEqual(received['closed_inputs'], 1)
                tasks.put(None)
                worker.join(timeout=20)
                self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
            tasks.close()
            results.close()
            tasks.join_thread()
            results.join_thread()


if __name__ == '__main__':
    unittest.main()
