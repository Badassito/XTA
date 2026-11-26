"""Exercise native descriptor ownership through spawned actors and the production wire."""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import pickle
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from XTA import runtime
from XTA.native_detector_handles import detector_handle_submission, install_detector_handles, planned_detector_source_pins
from XTA.native_scheduler import CPU, PreparedWorkScheduler, recv_dispatch, send_receipt
from XTA.tta_worker_transport import DetectorActorPort


def _submit(owner, task, **kwargs):
    with detector_handle_submission(task) as (packet, handles):
        owner.submit_prepared(task['task_id'], 1, pickle.dumps(packet), CPU, handles=handles, **kwargs)


@contextlib.contextmanager
def _memfd(label, data):
    fd = os.memfd_create(label, flags=os.MFD_CLOEXEC)
    os.write(fd, data)
    path = str(runtime._memfd_proc_path(fd))
    runtime._register_memfd_owner(path, fd, label)
    try:
        yield path
    finally:
        runtime._release_memfd_owner_key(path)


def _receiver(channel):
    port = DetectorActorPort(channel, None)
    sources = {}
    try:
        while (task := port.get()) is not None:
            task_id = task['task_id']
            transient = []
            try:
                # This goes through the same materialization + ACK as both detector actors.
                transient = port.materialize(task, sources)
                port.installed(task_id)
                source = Path(task['source_volume_path'])
                diagnostics = {'source': source.read_bytes(), 'path': str(source),
                               'source_count': len(sources), 'transient_count': len(transient)}
                if task.get('cpu_input_path'):
                    fd = int(Path(task['cpu_input_path']).name)
                    diagnostics['input'] = os.pread(fd, 1, 0)
                    try:
                        os.pwrite(fd, b'?', 0)
                    except OSError:
                        pass
                    else:
                        raise AssertionError('CPU input descriptor permitted a write')
                if task.get('result_mask_path'):
                    os.pwrite(int(Path(task['result_mask_path']).name), bytes([task_id % 256]), 0)
                port.compute_done(task_id)
            except BaseException as exc:
                port.put(dict(type='result', task_id=task_id, worker_kind='cpu', cpu_index=7,
                              ok=False, error=repr(exc)))
                raise
            finally:
                runtime._close_fd_list(transient)
            diagnostics['closed'] = all(not Path(f'/proc/self/fd/{fd}').exists() for fd in transient)
            port.put(dict(type='result', task_id=task_id, worker_kind='cpu', cpu_index=7,
                          ok=True, stats=diagnostics))
    finally:
        runtime._close_fd_list(sources.values())
        channel.close()


@unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux SCM_RIGHTS and memfd required')
class NativeMemfdInputHandoffTests(unittest.TestCase):
    def test_unfunded_source_only_plan_pins_survive_unregister_until_cancellation(self):
        channel, actor = socket.socketpair()
        with channel, actor, PreparedWorkScheduler(31, 1, 1) as owner:
            owner.register_worker(0, 1, os.getpid(), CPU, 1, channel)
            owner.declare_parent(1, 1, 1)
            self.assertEqual(owner.wait_preparation(5)[0], 1)
            owner.declare_parent(2, 1, 1, descriptor_only=True)
            with _memfd('native-planned-cancel', b'complete') as path:
                source = os.stat(path)
                with planned_detector_source_pins() as pin:
                    with detector_handle_submission(dict(task_id=2, source_volume_path=path)) as (packet, handles):
                        retained = pin(packet, handles)
                    self.assertEqual(len(retained), 1)
                    owner.submit_planned(2, 2, pickle.dumps(packet), CPU, handles=retained)
                with self.assertRaises(OSError):
                    os.fstat(retained[0][0])

            def surviving_pins():
                count = 0
                for fd in Path('/proc/self/fd').iterdir():
                    try:
                        witness = fd.stat()
                    except FileNotFoundError:
                        continue
                    count += (witness.st_dev, witness.st_ino) == (source.st_dev, source.st_ino)
                return count

            self.assertEqual(owner.snapshot()['dispatched'], 0)
            self.assertEqual(surviving_pins(), 1)
            owner.cancel('discard prepared input before funding')
            self.assertEqual(surviving_pins(), 0)

    def test_source_only_capture_requires_completed_lazy_cube_and_releases_partial_preflight(self):
        import fcntl
        with _memfd('cube', b'cube') as path, tempfile.TemporaryDirectory() as name:
            ready, failed = Path(name)/'ready', Path(name)/'failed'
            spec = dict(source_volume_path=path, native_resize=dict(path=path, sentinel=str(ready), failed=str(failed)))
            with self.assertRaisesRegex(ValueError, 'inventory rejection'):
                with planned_detector_source_pins() as pin:
                    with detector_handle_submission(spec) as (packet, handles):
                        self.assertIsNone(pin(packet, handles))
                        ready.touch()
                        fd, _, role = pin(packet, handles)[0]
                        self.assertEqual(role, 0)
                        self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
                        failed.touch()
                        self.assertIsNone(pin(packet, handles))
                    for field in ('cpu_input_path', 'result_mask_path', 'canvas_path', 'd1_bitset_path'):
                        with detector_handle_submission({field: path}) as (packet, handles):
                            self.assertIsNone(pin(packet, handles))
                    raise ValueError('inventory rejection')
            with self.assertRaises(OSError):
                os.fstat(fd)
            self.assertEqual(Path(path).read_bytes(), b'cube')

    def test_160_packets_install_one_source_with_zero_additional_sharer_registrations(self):
        source_channel, child = socket.socketpair()
        worker = mp.get_context('spawn').Process(target=_receiver, args=(child,))
        worker.start()
        child.close()
        try:
            with PreparedWorkScheduler(41, 8192, 1) as owner, source_channel:
                owner.register_worker(2**32 + 7, 4, worker.pid, CPU, 4, source_channel)
                owner.declare_parent(1, 160, 0)
                parent, generation = owner.wait_preparation(10)
                self.assertEqual(parent, 1)
                with _memfd('source', b'source') as source, _memfd('output', b'.') as output, \
                        mock.patch('multiprocessing.resource_sharer._resource_sharer.register',
                                   side_effect=AssertionError('Python descriptor broker')):
                    for task_id in range(160):
                        with _memfd('input', bytes([task_id])) as pixels:
                            _submit(owner, dict(task_id=task_id, source_volume_path=source,
                                                cpu_input_path=pixels, result_mask_path=output))
                    expected_path = None
                    for _ in range(160):
                        kind, lease, payload = owner.wait_event(20)
                        self.assertEqual(kind, 'final')
                        stats = pickle.loads(payload)['stats']
                        self.assertEqual((stats['input'], stats['source'], stats['source_count'],
                                          stats['transient_count'], stats['closed']),
                                         (bytes([lease.task_id]), b'source', 1, 2, True))
                        expected_path = expected_path or stats['path']
                        self.assertEqual(stats['path'], expected_path)
                    self.assertEqual(Path(output).read_bytes(), bytes([159]))
                self.assertEqual(owner.wait_publication(10), (1, generation))
                owner.retire_parent(1, generation)
                owner.send_control(2**32 + 7, pickle.dumps(None), stop=True)
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(10)

    def test_source_install_ack_is_required_per_worker_and_gates_overlap(self):
        with contextlib.ExitStack() as stack:
            owner = stack.enter_context(PreparedWorkScheduler(2, 100, 1))
            actors = []
            for worker in (5, 8):
                channel, actor = socket.socketpair()
                stack.enter_context(actor)
                with channel:
                    owner.register_worker(worker, worker + 10, os.getpid(), CPU, 3, channel)
                actors.append(actor)
            owner.declare_parent(1, 4, 1)
            owner.wait_preparation(10)
            with _memfd('source', b'abc') as path:
                for task, worker in enumerate((5, 5, 8, 8)):
                    _submit(owner, dict(task_id=task, source_volume_path=path), required_worker=worker)
            for actor in actors:
                installed = {}
                received = {}
                lease, data = recv_dispatch(actor, received_handles=received)
                self.assertEqual(len(received), 1)
                self.assertEqual(install_detector_handles(pickle.loads(data), installed, received), [])
                stack.callback(runtime._close_fd_list, tuple(installed.values()))
                actor.settimeout(0.03)
                with self.assertRaises(TimeoutError):
                    actor.recv(1)
                actor.settimeout(5)
                send_receipt(actor, lease, 'installed')
                next_lease, data = recv_dispatch(actor, received_handles=received)
                self.assertEqual(received, {})
                self.assertEqual(install_detector_handles(pickle.loads(data), installed, received), [])
                send_receipt(actor, next_lease, 'installed')
                for completed in (lease, next_lease):
                    send_receipt(actor, completed, 'compute')
                    send_receipt(actor, completed, 'final')
            self.assertIsNotNone(owner.wait_publication(10))

    def test_cancel_closes_undispatched_and_unacknowledged_pins(self):
        channel, actor = socket.socketpair()
        with channel, actor, PreparedWorkScheduler(7, 100, 1) as owner:
            owner.register_worker(4, 1, os.getpid(), CPU, 1, channel)
            owner.declare_parent(1, 2, 0)
            owner.wait_preparation(10)
            for task_id in (1, 2):
                with _memfd('native-cancel-test', b'body') as path:
                    _submit(owner, dict(task_id=task_id, source_volume_path=path))
            received = {}
            recv_dispatch(actor, received_handles=received)
            runtime._close_fd_list(received.values())
            def pins():
                result = []
                for fd in Path('/proc/self/fd').iterdir():
                    try:
                        if 'native-cancel-test' in os.readlink(fd):
                            result.append(fd)
                    except FileNotFoundError:
                        pass
                return result
            self.assertEqual(len(pins()), 2)
            owner.cancel('cancel uninstalled work')
            deadline = time.monotonic() + 5
            while pins() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(pins(), [])

    def test_forged_identity_and_role_never_install_or_modify_storage(self):
        with _memfd('identity', b'original') as path:
            for field, replacement in [('identity', '0:0:0:0:0:0'), ('role', 2)]:
                with detector_handle_submission(dict(source_volume_path=path)) as (packet, handles):
                    original = packet['_detector_handle_table'][0]
                    packet['_detector_handle_table'] = ((replacement, 0) if field == 'identity'
                                                         else (original[0], replacement),)
                    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                    received = {0: fd}
                    with self.assertRaises(RuntimeError):
                        install_detector_handles(packet, {}, received)
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                    self.assertFalse(received)
                    self.assertEqual(Path(path).read_bytes(), b'original')

    def test_nested_policy_packet_larger_than_one_kernel_descriptor_message(self):
        channel, actor = socket.socketpair()
        with channel, actor, PreparedWorkScheduler(3, 1, 1) as owner:
            owner.register_worker(0, 1, os.getpid(), CPU, 1, channel)
            owner.declare_parent(1, 1, 0)
            owner.wait_preparation(5)
            with contextlib.ExitStack() as parents:
                children = [dict(canvas_path=parents.enter_context(_memfd('policy', bytes([i % 256]))))
                            for i in range(260)]
                _submit(owner, dict(task_id=0, augmentation_pass_tasks=children))
                self.assertTrue(all(set(child) == {'canvas_path'} for child in children))
            actor.settimeout(5)
            received = {}
            lease, payload = recv_dispatch(actor, received_handles=received)
            task = pickle.loads(payload)
            self.assertEqual(len(received), 260)
            transient = install_detector_handles(task, {}, received)
            try:
                self.assertEqual([Path(child['canvas_path']).read_bytes() for child in task['augmentation_pass_tasks']],
                                 [bytes([i % 256]) for i in range(260)])
            finally:
                runtime._close_fd_list(transient)
            send_receipt(actor, lease, 'installed')
            send_receipt(actor, lease, 'compute')
            send_receipt(actor, lease, 'final')
            self.assertIsNotNone(owner.wait_publication(5))


class PortableDetectorHandleTests(unittest.TestCase):
    def test_ordinary_file_paths_preserve_their_mapping_without_handle_registration(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'ordinary.raw'
            path.write_bytes(b'bytes')
            task = dict(source_volume_path=str(path), native_resize={'path': str(path)})
            with detector_handle_submission(task) as (packet, handles):
                self.assertEqual(handles, ())
                self.assertEqual(install_detector_handles(packet, {}, {}), [])
                self.assertEqual(Path(packet['source_volume_path']).read_bytes(), b'bytes')
                self.assertEqual(packet, task)
