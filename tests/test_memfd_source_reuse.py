"""Backing identity, nested mappings and rollback at the native detector boundary."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
import pickle
import socket
import tempfile
import threading
import unittest

from XTA import runtime
from XTA.native_detector_handles import close_received_handles, detector_handle_submission, install_detector_handles
from XTA.native_scheduler import CPU, PreparedWorkScheduler, recv_dispatch


@contextmanager
def _owner(key, data=b'source'):
    """Register a real file description without requiring Linux memfd on the test host."""
    with tempfile.TemporaryFile() as backing:
        backing.write(data)
        backing.flush()
        fd = os.dup(backing.fileno())
        runtime._register_memfd_owner(key, fd, 'detector transport test')
        try:
            yield fd
        finally:
            runtime._release_memfd_owner_key(key)


@contextmanager
def native_dispatch(task):
    """One production scheduler dispatch addressed to this process as a test actor."""
    channel, actor = socket.socketpair()
    with channel, actor, PreparedWorkScheduler(81, 100, 1) as scheduler:
        scheduler.register_worker(0, 1, os.getpid(), CPU, 1, channel)
        scheduler.declare_parent(1, 1, 0)
        scheduler.wait_preparation(5)
        descriptor = dict(task, task_id=0)
        with detector_handle_submission(descriptor) as (packet, handles):
            scheduler.submit_prepared(0, 1, pickle.dumps(packet), CPU, handles=handles)
        received = {}
        actor.settimeout(5)
        _, payload = recv_dispatch(actor, received_handles=received)
        try:
            yield pickle.loads(payload), received
        finally:
            close_received_handles(received)


class DetectorSourcePreparationTests(unittest.TestCase):
    def test_nested_writable_and_read_only_aliases_have_distinct_roles(self):
        with _owner('shared') as fd:
            child = dict(source_volume_path='shared', result_conf_path='shared',
                         native_resize={'path': 'shared'})
            base = dict(source_volume_path='shared', result_mask_path='shared', cpu_input_path='shared',
                        augmentation_pass_tasks=[child])
            with detector_handle_submission(base) as (packet, handles):
                self.assertEqual(len(handles), 3)
                borrowed, = {entry[0] for entry in handles}
                self.assertNotEqual(borrowed, fd)
                self.assertEqual(os.fstat(borrowed), os.fstat(fd))
                self.assertFalse(os.get_inheritable(borrowed))
                self.assertEqual(len({entry[1] for entry in handles}), 1)
                self.assertEqual({entry[2] for entry in handles}, {0, 1, 2})
                copied = packet['augmentation_pass_tasks'][0]
                self.assertEqual(copied['_detector_handle_slots']['source_volume_path'],
                                 copied['native_resize']['_detector_handle_slots']['path'])
                self.assertEqual(copied['_detector_handle_slots']['result_conf_path'],
                                 packet['_detector_handle_slots']['result_mask_path'])
            with self.assertRaises(OSError):
                os.fstat(borrowed)
            self.assertFalse(any('_detector_handle_slots' in owner for owner in (base, child, child['native_resize'])))

    def test_owner_can_retire_during_submission_and_pin_closes_on_failure(self):
        with _owner('registered') as original:
            with self.assertRaisesRegex(RuntimeError, 'submit failed'):
                with detector_handle_submission(dict(source_volume_path='registered')) as (_, handles):
                    pin = handles[0][0]
                    release = threading.Thread(target=runtime._release_memfd_owner_key,
                                               args=('registered',))
                    release.start()
                    release.join(3)
                    self.assertFalse(release.is_alive(), 'descriptor preparation kept the registry locked')
                    with self.assertRaises(OSError):
                        os.fstat(original)
                    self.assertEqual(os.fstat(pin).st_size, len(b'source'))
                    raise RuntimeError('submit failed')
            with self.assertRaises(OSError):
                os.fstat(pin)

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires inherited descriptor table')
    def test_forked_child_drops_submission_pin(self):
        with _owner('registered'):
            with detector_handle_submission(dict(source_volume_path='registered')) as (_, handles):
                pin = handles[0][0]
                child = os.fork()
                if not child:
                    try:
                        os.fstat(pin)
                    except OSError as exc:
                        os._exit(0 if exc.errno == errno.EBADF else 1)
                    os._exit(2)
                _, result = os.waitpid(child, 0)
                self.assertEqual(os.waitstatus_to_exitcode(result), 0)
                self.assertEqual(os.fstat(pin).st_size, len(b'source'))

    def test_unregistered_proc_alias_and_pickle_failure_preserve_parent_ownership(self):
        with _owner('registered') as fd:
            for task, error in ((dict(source_volume_path='registered', canvas_path='/proc/123/fd/456'), RuntimeError),
                                (dict(source_volume_path='registered', callback=lambda: None), AttributeError)):
                with self.assertRaises(error):
                    with detector_handle_submission(task) as (packet, _):
                        pickle.dumps(packet)
                self.assertEqual(os.fstat(fd).st_size, len(b'source'))
                self.assertNotIn('_detector_handle_slots', task)


@unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux descriptor access modes and SCM_RIGHTS required')
class NativeDetectorSourceTests(unittest.TestCase):
    def test_content_changes_preserve_identity_and_size_changes_force_revalidation(self):
        with _owner('source') as original, native_dispatch(dict(source_volume_path='source')) as (packet, received):
            cached = {}
            os.pwrite(original, b'pixels', 0)
            self.assertEqual(install_detector_handles(packet, cached, received), [])
            self.addCleanup(runtime._close_fd_list, cached.values())
            installed = next(iter(cached.values()))
            self.assertEqual(os.pread(installed, 6, 0), b'pixels')
            with detector_handle_submission(dict(source_volume_path='source')) as (repeated, _):
                os.ftruncate(original, 2)
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    install_detector_handles(repeated, cached, {})
            self.assertEqual(os.fstat(installed).st_size, 2)

    def test_replaced_owner_fd_number_acquires_new_inode(self):
        with _owner('left', b'first') as original, _owner('right', b'other') as replacement:
            with native_dispatch(dict(source_volume_path='left')) as (packet, received):
                cached = {}
                install_detector_handles(packet, cached, received)
            self.addCleanup(runtime._close_fd_list, cached.values())
            old = next(iter(cached.values()))
            os.dup2(replacement, original)
            with native_dispatch(dict(source_volume_path='left')) as (packet, received):
                install_detector_handles(packet, cached, received)
            self.assertEqual(len(cached), 2)
            self.assertEqual(os.pread(old, 5, 0), b'first')
            self.assertEqual(sorted(os.pread(fd, 5, 0) for fd in cached.values()), [b'first', b'other'])

    def test_corrupt_nested_reference_closes_new_rights_and_preserves_prior_sources(self):
        with _owner('old'), _owner('new'), _owner('output'):
            with native_dispatch(dict(source_volume_path='old')) as (packet, received):
                cached = {}
                install_detector_handles(packet, cached, received)
            self.addCleanup(runtime._close_fd_list, cached.values())
            existing = dict(cached)
            child = dict(result_mask_path='output', native_resize={'path': 'new'})
            with native_dispatch(dict(source_volume_path='new', augmentation_pass_tasks=[child])) as (packet, received):
                fds = tuple(received.values())
                packet['augmentation_pass_tasks'][0]['_detector_handle_slots']['result_mask_path'] = -1
                with self.assertRaisesRegex(ValueError, 'access role'):
                    install_detector_handles(packet, cached, received)
            self.assertEqual(cached, existing)
            self.assertEqual(os.pread(next(iter(cached.values())), 6, 0), b'source')
            for fd in fds:
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_missing_duplicate_and_extra_rights_fail_before_source_commit(self):
        with _owner('source'), _owner('output'):
            for invalid in ('missing', 'duplicate', 'extra'):
                with self.subTest(invalid=invalid), native_dispatch(
                        dict(source_volume_path='source', result_mask_path='output')) as (packet, received):
                    if invalid == 'missing':
                        os.close(received.pop(1))
                    elif invalid == 'duplicate':
                        packet['_detector_handle_table'] += (packet['_detector_handle_table'][0],)
                    else:
                        received[4] = os.dup(received[1])
                    pins, cached = tuple(received.values()), {}
                    with self.assertRaises((RuntimeError, ValueError)):
                        install_detector_handles(packet, cached, received)
                    self.assertEqual(cached, {})
                    self.assertEqual(received, {})
                    for fd in pins:
                        with self.assertRaises(OSError):
                            os.fstat(fd)
