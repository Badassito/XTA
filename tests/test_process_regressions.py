from __future__ import annotations

import gc
import os
import signal
import tempfile
import threading
import types
import unittest
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock
import weakref

import numpy as np


from XTA import assembly, interpolation, media, pipeline, runtime


def _policy_memfd_transfer_members(owners: list[object]) -> list[dict[str, str]]:
    """Use stable owner paths for both portable and real Linux transfer tests."""
    return [{
        'source_volume_path': str(runtime._memfd_backing_path_from_array(owners[0])),
        'result_mask_path': str(runtime._memfd_backing_path_from_array(owners[1 + 2 * index])),
        'result_conf_path': str(runtime._memfd_backing_path_from_array(owners[2 + 2 * index])),
    } for index in range(3)]


class _DetachHandle:
    def __init__(self, fd: int) -> None:
        self.fd = int(fd)

    def detach(self) -> int:
        return int(self.fd)


class _SubmitFailureExecutor:
    def submit(self, _function: object, **kwargs: object) -> object:
        stage = np.memmap(
            Path(str(kwargs['mask_path'])),
            dtype=np.dtype(str(kwargs['mask_dtype'])),
            mode='r+',
            shape=tuple(int(value) for value in kwargs['mask_shape']),
        )
        stage[:] = np.uint8(9)
        stage.flush()
        runtime.close_memmap_array(stage)
        raise RuntimeError('submit failure sentinel')


class _SuccessfulExecutor:
    def submit(self, _function: object, **kwargs: object) -> object:
        stage = np.memmap(
            Path(str(kwargs['mask_path'])),
            dtype=np.dtype(str(kwargs['mask_dtype'])),
            mode='r+',
            shape=tuple(int(value) for value in kwargs['mask_shape']),
        )
        stage[:] = np.uint8(9)
        stage.flush()
        runtime.close_memmap_array(stage)

        class _Future:
            @staticmethod
            def result() -> dict[str, object]:
                return {'worker_completed': True}

        return _Future()


class _FailingAuxPool:
    @staticmethod
    def try_submit(_kwargs: object) -> object:
        return object()

    @staticmethod
    def wait(_handle: object) -> dict[str, object]:
        raise RuntimeError('aux wait failure sentinel')


class ProcessRuntimeRegressionTests(unittest.TestCase):
    def aux_pool(self, queues):
        from tests.native_stage_test_support import native_coordinator
        coordinator = native_coordinator(self)
        coordinator.configure_workers(tuple(queues))
        coordinator.set_inference_priority_active(False)
        return runtime._GpuWorkerAuxInterpolationPool(queues)

    def tearDown(self) -> None:
        runtime.set_interpolation_process_executor(None, 0)
        runtime.set_gpu_worker_aux_interpolation_pool(None)
        media.abort_streaming_producers('test teardown')
        media.wait_for_streaming_producers(timeout=5.0)
        media.reset_streaming_state_for_new_run()

    def assert_speculative_stage_retired(self, work_dir: Path) -> None:
        for path in work_dir.glob('*fallback-stage*'):
            runtime.wait_for_retired_memmap_unlinks(path=path, timeout_s=5.0)
        self.assertEqual(list(work_dir.glob('*fallback-stage*')), [])

    def test_policy_memfd_transfer_fixture_uses_reopenable_owner_paths_across_platforms(self) -> None:
        paths = [Path(f'/proc/12345/fd/{100 + index}') for index in range(7)]
        roots = [types.SimpleNamespace(_workspace_memfd_path=str(path),
            filename=f'/memfd:policy-owner-{index} (deleted)') for index, path in enumerate(paths)]
        owners = [types.SimpleNamespace(base=owner) for owner in roots]
        members = _policy_memfd_transfer_members(owners)
        self.assertEqual(len(members), 3)
        for index, member in enumerate(members):
            self.assertEqual(member['source_volume_path'], str(paths[0]))
            self.assertEqual(member['result_mask_path'], str(paths[1 + 2 * index]))
            self.assertEqual(member['result_conf_path'], str(paths[2 + 2 * index]))
            self.assertFalse(any('memfd:' in path for path in member.values()))
        self.assertIsNone(runtime._memfd_backing_path_from_array(np.zeros((2, 3), np.uint8)))

    def test_fork_start_method_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_INTERPOLATION_PROCESS_START_METHOD': 'fork'},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, 'spawn or forkserver'):
                runtime.interpolation_process_start_method()

    def test_preflight_surfaces_pickle_failure_synchronously(self) -> None:
        with self.assertRaisesRegex(TypeError, 'not serializable'):
            runtime.preflight_multiprocessing_payload({'callback': lambda: None})

    def test_interpolation_backing_requires_complete_root_memfd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backing = Path(directory) / 'volume.u8'
            original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(4, 2, 2))
            copy = np.memmap(backing, dtype=np.uint8, mode='c', shape=original.shape)
            with mock.patch.object(runtime, '_memfd_backing_path_from_array', return_value=backing):
                self.assertEqual(runtime._interpolation_array_backing_path(original), backing)
                self.assertIsNone(runtime._interpolation_array_backing_path(original[1:3]))
                self.assertIsNone(runtime._interpolation_array_backing_path(copy))
            original._mmap.close()
            copy._mmap.close()

    def test_aux_interpolation_transfers_only_after_preflight_and_rolls_back_put(self) -> None:
        events = []
        with tempfile.TemporaryFile() as data:
            descriptor = os.dup(data.fileno())
            pool = self.aux_pool({
                0: types.SimpleNamespace(put=mock.Mock(side_effect=RuntimeError('queue closed'))),
            })
            kwargs = {'mask_path': 'test-owner'}
            def preflight(envelope):
                self.assertNotIn('mask_fd', envelope['aux_kwargs'])
                events.append('preflight')
            def duplicate(_path):
                events.append('duplicate')
                return _DetachHandle(descriptor)
            with mock.patch.object(runtime, 'preflight_multiprocessing_payload', side_effect=preflight), \
                    mock.patch.object(runtime, '_duplicate_memfd_path_for_child', side_effect=duplicate):
                self.assertIsNone(pool.try_submit(kwargs))
                self.assertEqual(events, [])
                pool.enable_worker(0)
                self.assertIsNone(pool.try_submit(kwargs))
            self.assertEqual(events, ['preflight', 'duplicate'])
            self.assertEqual(pool.outstanding(), 0)
            self.assertNotIn('mask_fd', kwargs)
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_aux_interpolation_does_not_transfer_after_pool_invalidates_preflight_reservation(self) -> None:
        task_queue = mock.Mock()
        pool = self.aux_pool({0: task_queue})
        pool.enable_worker(0)
        with mock.patch.object(runtime, 'preflight_multiprocessing_payload',
                               side_effect=lambda _: pool.mark_failed('worker exited')), \
                mock.patch.object(runtime, '_duplicate_memfd_path_for_child') as duplicate:
            self.assertIsNone(pool.try_submit({'mask_path': 'test-owner'}))
        duplicate.assert_not_called()
        task_queue.put.assert_not_called()

    def test_aux_failure_rolls_back_committed_descriptor(self) -> None:
        with tempfile.TemporaryFile() as data:
            descriptor = os.dup(data.fileno())
            pool = self.aux_pool({0: mock.Mock()})
            pool.enable_worker(0)
            with mock.patch.object(runtime, '_duplicate_memfd_path_for_child',
                                   return_value=_DetachHandle(descriptor)):
                handle = pool.try_submit({'mask_path': 'test-owner'})
            self.assertIsNotNone(handle)
            pool.mark_failed('worker exited')
            with self.assertRaisesRegex(RuntimeError, 'worker exited'):
                pool.wait(handle)
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_broken_interpolation_pool_reclaims_transfer_and_preserves_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            backing = work_dir / 'volume.u8'
            original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
            descriptor = os.open(backing, os.O_RDWR | getattr(os, 'O_BINARY', 0))
            failure = BrokenProcessPool('worker exited')
            future = Future()
            future.set_exception(failure)
            executor = types.SimpleNamespace(submit=mock.Mock(return_value=future))
            runtime.set_interpolation_process_executor(executor, 1)
            try:
                with mock.patch.dict(os.environ, {
                        'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                        'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': '0',
                    }), mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False), \
                        mock.patch.object(runtime, '_duplicate_memfd_path_for_child', return_value=_DetachHandle(descriptor)):
                    with self.assertRaisesRegex(RuntimeError, 'Interpolation process worker failed') as raised:
                        runtime.interpolate_view_volume_pass_maybe_process(
                            original, types.SimpleNamespace(name='test-view'), work_dir,
                            'broken-pool', 3, 15.0, 1, 2, 0.0, workers=1,
                        )
                self.assertIs(raised.exception.__cause__, failure)
                self.assertIn('mask_fd', executor.submit.call_args.kwargs)
                self.assertTrue(backing.exists())
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
            finally:
                original._mmap.close()

    def test_interpolation_opens_transferred_descriptor_without_parent_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            backing = work_dir / 'volume.u8'
            original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
            descriptor = os.open(backing, os.O_RDWR | getattr(os, 'O_BINARY', 0))
            def interpolate(*, mask_mm, **_unused):
                mask_mm[:] = 7
                return {}
            try:
                with mock.patch.object(interpolation, 'interpolate_view_volume_pass_inplace', new=interpolate), \
                        mock.patch.object(runtime, '_INTERPOLATION_PROCESS_WORKER', False):
                    runtime._interpolation_process_entry(
                        mask_path=str(work_dir / 'missing-parent-file'), mask_shape=original.shape,
                        mask_dtype='uint8', work_dir=str(work_dir), pass_tag='fd-transfer',
                        max_slice_distance=1, search_angle_deg=10., interpolation_walk_back=1,
                        interpolation_candidates=2, interpolate_min_radius=0., reserve_bytes=0,
                        workers=1, wrap_axis=False, mask_fd=_DetachHandle(descriptor),
                    )
                np.testing.assert_array_equal(original, np.full(original.shape, 7, np.uint8))
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
            finally:
                original._mmap.close()

    def test_termination_watchdog_installs_without_unique_scratch_and_restores_handlers(self) -> None:
        original = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        called = threading.Event()
        with mock.patch.object(runtime, '_force_termination_watchdog', side_effect=lambda *_args: called.set()) as watchdog:
            with runtime.run_termination_watchdog():
                with self.assertRaises(KeyboardInterrupt):
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                for signum in original:
                    self.assertIsNot(signal.getsignal(signum), original[signum])
                self.assertTrue(called.wait(5))
        self.assertEqual(watchdog.call_args.args[0], signal.SIGTERM)
        self.assertEqual({signum: signal.getsignal(signum) for signum in original}, original)

    def test_run_and_sweep_watchdogs_preserve_ignored_signal(self) -> None:
        from XTA.confidence_sweep import _cancel_signals

        original = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for install in (runtime.run_termination_watchdog, _cancel_signals):
                with self.subTest(watchdog=install.__name__), install():
                    self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGINT, original)

    def test_failed_process_pass_retires_its_copied_input_with_live_exception(self) -> None:
        original = np.zeros((2, 2, 2), np.uint8)
        for fallback in ('0', '1'):
            with self.subTest(fallback=fallback), tempfile.TemporaryDirectory() as directory:
                work_dir = Path(directory)
                executor = types.SimpleNamespace(
                    submit=mock.Mock(side_effect=RuntimeError('submit failure sentinel')),
                )
                runtime.set_interpolation_process_executor(executor, 1)

                def fail_in_process(*, mask_mm, **_kwargs):
                    mask_mm[:] = 1
                    raise ValueError('fallback failure sentinel')

                with mock.patch.dict(os.environ, {
                        'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                        'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': fallback,
                    }), mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False), \
                        mock.patch.object(interpolation, 'interpolate_view_volume_pass_inplace', new=fail_in_process):
                    try:
                        runtime.interpolate_view_volume_pass_maybe_process(
                            original, types.SimpleNamespace(name='test-view'), work_dir,
                            'copied-input', 3, 15.0, 1, 2, 0.0, workers=1,
                        )
                    except (RuntimeError, ValueError) as failure:
                        detail = failure if fallback == '1' else failure.__cause__
                        self.assertIn('fallback failure' if fallback == '1' else 'submit failure', str(detail))
                        runtime.wait_for_retired_memmap_unlinks(under=work_dir, timeout_s=5.0)
                        self.assertEqual(list(work_dir.glob('*.u8.dat')), [])
                    else:
                        self.fail('interpolation failure did not propagate')
        self.assertFalse(original.any())

    @unittest.skipUnless(hasattr(os, 'memfd_create') and Path('/proc/self/fd').exists(),
                         'real memfd descriptor transfer requires Linux procfs')
    def test_policy_memfd_transfers_reopen_independent_parent_arrays(self) -> None:
        shape = (2, 3, 4)
        from XTA.native_detector_handles import install_detector_handles
        from tests.test_memfd_source_reuse import native_dispatch
        owners, reopened, transient, persistent = [], [], [], {}
        try:
            with mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES': '1'}):
                for index in range(7):
                    owners.append(runtime._allocate_memfd_workspace_array(
                        shape, np.uint8, f'policy-transfer-{index}', initialize_zero=True))
            members = _policy_memfd_transfer_members(owners)
            members[0]['augmentation_pass_tasks'] = members[1:]
            with native_dispatch(members[0]) as (local, received):
                transient = install_detector_handles(local, persistent, received)
            self.assertEqual(len(transient), 6)
            self.assertEqual(len(persistent), 1)
            for index, member in enumerate((local, *local['augmentation_pass_tasks'])):
                for offset, field in enumerate(('result_mask_path', 'result_conf_path')):
                    mapping = np.memmap(member[field], mode='r+', dtype=np.uint8, shape=shape)
                    reopened.append(mapping)
                    mapping[1, :, :] = np.uint8(10 * (index + 1) + offset)
            # The worker can release transferred handles while deferred copies own
            # their mmaps. Original parent owners remain independent and readable.
            runtime._close_fd_list(transient)
            transient = []
            for index in range(3):
                for offset in range(2):
                    parent = owners[1 + 2 * index + offset]
                    np.testing.assert_array_equal(parent[0], np.zeros(shape[1:], np.uint8))
                    np.testing.assert_array_equal(parent[1],
                        np.full(shape[1:], 10 * (index + 1) + offset, np.uint8))
            for member in members:
                self.assertNotIn('result_mask_fd', member)
                self.assertNotIn('result_conf_fd', member)
        finally:
            for mapping in reopened:
                runtime.close_memmap_array_without_flush(mapping)
            runtime._close_fd_list(transient)
            runtime._close_fd_list(persistent.values())
            for owner in owners:
                runtime.close_memmap_array_without_flush(owner)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux interpolation descriptor broker')
    def test_interpolation_sharer_rollback_releases_registered_descriptor(self):
        descriptor = os.memfd_create('interpolation-rollback', flags=os.MFD_CLOEXEC)
        path = str(runtime._memfd_proc_path(descriptor))
        runtime._register_memfd_owner(path, descriptor, 'interpolation rollback')
        try:
            for _ in range(20):
                batch = runtime._attach_interpolation_memfd_transfer({'mask_path': path})
                self.assertEqual(len(batch.handles), 1)
                batch.rollback()
                batch.rollback()
            expected = os.fstat(descriptor)
            remaining = []
            for entry in Path('/proc/self/fd').iterdir():
                try:
                    current = entry.stat()
                except FileNotFoundError:
                    continue
                if (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino):
                    remaining.append(int(entry.name))
            self.assertEqual(remaining, [descriptor])
        finally:
            runtime._release_memfd_owner_key(path)

    def _run_interpolation_case(self, executor: object) -> tuple[np.memmap, np.ndarray, dict[str, object], Path]:
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        work_dir = Path(temp_context.name)
        backing = work_dir / 'input.u8.dat'
        original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
        original[:] = np.uint8(1)
        original.flush()

        def _fallback(*, mask_mm: np.ndarray, **_kwargs: object) -> dict[str, object]:
            np.testing.assert_array_equal(np.asarray(mask_mm), np.ones((2, 2, 2), dtype=np.uint8))
            mask_mm[:] = np.uint8(7)
            return {'fallback_saw_clean_input': True}

        runtime.set_interpolation_process_executor(executor, 1)  # type: ignore[arg-type]
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                    'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': '1',
                },
                clear=False,
            ),
            mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False),
            mock.patch.object(interpolation, 'interpolate_view_volume_pass_inplace', side_effect=_fallback),
        ):
            result, stats = runtime.interpolate_view_volume_pass_maybe_process(
                original,
                types.SimpleNamespace(name='test-view'),
                work_dir,
                'fault-injection',
                3,
                15.0,
                1,
                2,
                0.0,
                workers=1,
            )
        return original, result, stats, work_dir

    def test_submit_failure_fallback_uses_clean_input_and_preserves_identity(self) -> None:
        original, result, stats, work_dir = self._run_interpolation_case(
            _SubmitFailureExecutor(),
        )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(np.asarray(result), np.full((2, 2, 2), 7, dtype=np.uint8))
            self.assertTrue(stats['fallback_saw_clean_input'])
            self.assertEqual(stats['process_backend'], 'fallback_in_process_after_worker_failure')
            self.assert_speculative_stage_retired(work_dir)
        finally:
            mapping_ref = weakref.ref(original._mmap)
            runtime.close_memmap_array(original)
            self.assertFalse(original._mmap.closed)
            result = original = None
            gc.collect()
            self.assertIsNone(mapping_ref())

    def test_successful_transaction_commits_back_to_original_mapping(self) -> None:
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        work_dir = Path(temp_context.name)
        backing = work_dir / 'input.u8.dat'
        original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
        original[:] = np.uint8(1)
        original.flush()
        runtime.set_interpolation_process_executor(_SuccessfulExecutor(), 1)  # type: ignore[arg-type]
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                    'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': '1',
                },
                clear=False,
            ),
            mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False),
            mock.patch.object(
                interpolation,
                'interpolate_view_volume_pass_inplace',
                side_effect=AssertionError('fallback must not run after success'),
            ),
        ):
            result, stats = runtime.interpolate_view_volume_pass_maybe_process(
                original,
                types.SimpleNamespace(name='test-view'),
                work_dir,
                'success-injection',
                3,
                15.0,
                1,
                2,
                0.0,
                workers=1,
            )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(np.asarray(original), np.full((2, 2, 2), 9, dtype=np.uint8))
            self.assertTrue(stats['worker_completed'])
            self.assert_speculative_stage_retired(work_dir)
        finally:
            mapping_ref = weakref.ref(original._mmap)
            runtime.close_memmap_array(original)
            self.assertFalse(original._mmap.closed)
            result = original = None
            gc.collect()
            self.assertIsNone(mapping_ref())

    def test_aux_failure_fallback_retains_fallback_backend_telemetry(self) -> None:
        runtime.set_gpu_worker_aux_interpolation_pool(_FailingAuxPool())  # type: ignore[arg-type]
        original, result, stats, _work_dir = self._run_interpolation_case(
            _SuccessfulExecutor(),
        )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(
                np.asarray(result), np.full((2, 2, 2), 7, dtype=np.uint8),
            )
            self.assertTrue(stats['fallback_saw_clean_input'])
            self.assertEqual(
                stats['process_backend'], 'fallback_in_process_after_aux_failure',
            )
            self.assert_speculative_stage_retired(_work_dir)
        finally:
            mapping_ref = weakref.ref(original._mmap)
            runtime.close_memmap_array(original)
            self.assertFalse(original._mmap.closed)
            result = original = None
            gc.collect()
            self.assertIsNone(mapping_ref())

    def test_aux_queue_pickle_failure_rolls_back_pending_lease(self) -> None:
        class _Queue:
            def __init__(self) -> None:
                self.items: list[object] = []

            def put(self, item: object) -> None:
                self.items.append(item)

        task_queue = _Queue()
        pool = self.aux_pool({0: task_queue})
        self.assertTrue(pool.enable_worker(0))
        handle = pool.try_submit({'unpickleable': lambda: None})
        self.assertIsNone(handle)
        self.assertEqual(pool.outstanding(), 0)
        self.assertEqual(task_queue.items, [])


class PipelineLifecycleRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        media.abort_streaming_producers('test teardown')
        media.wait_for_streaming_producers(timeout=5.0)
        media.reset_streaming_state_for_new_run()

    def test_failed_run_cleans_registered_resources_and_next_run_resets_abort(self) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **_kwargs: object) -> None:
                self.shutdown_calls += 1

        class _Queue:
            def __init__(self) -> None:
                self.cancel_calls = 0
                self.close_calls = 0
                self.join_calls = 0

            def cancel_join_thread(self) -> None:
                self.cancel_calls += 1

            def close(self) -> None:
                self.close_calls += 1

            def join_thread(self) -> None:
                self.join_calls += 1

        executor = _Executor()
        process_queue = _Queue()
        calls = 0

        def _implementation() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                pipeline._run_resources().track_executor(executor)
                pipeline._run_resources().track_queue(process_queue)
                raise RuntimeError('pipeline failure sentinel')
            self.assertFalse(media.streaming_producers_aborted())

        with mock.patch.object(pipeline, '_main_impl', side_effect=_implementation):
            with self.assertRaisesRegex(RuntimeError, 'pipeline failure sentinel'):
                pipeline.main()
            self.assertTrue(media.streaming_producers_aborted())
            pipeline.main()
        self.assertEqual(calls, 2)
        self.assertGreaterEqual(executor.shutdown_calls, 1)
        self.assertEqual(process_queue.cancel_calls, 1)
        self.assertEqual(process_queue.close_calls, 1)
        self.assertEqual(process_queue.join_calls, 0)

    def test_compressor_shutdown_failure_does_not_strand_pipeline_lock(self) -> None:
        implementation_calls = 0
        telemetry = mock.Mock()

        def _implementation() -> None:
            nonlocal implementation_calls
            implementation_calls += 1

        with (
            mock.patch.object(pipeline, '_main_impl', side_effect=_implementation),
            mock.patch.object(
                pipeline, 'shutdown_nrrd_gzip_executors',
                side_effect=RuntimeError('compressor shutdown sentinel'),
            ),
            mock.patch.object(pipeline, 'runtime_telemetry', return_value=telemetry),
            mock.patch('builtins.print'),
        ):
            pipeline.main()
            pipeline.main()

        self.assertEqual(implementation_calls, 2)
        self.assertEqual(telemetry.fallback.call_count, 2)

    def test_later_thread_pool_constructor_failure_closes_earlier_pool(self) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **_kwargs: object) -> None:
                self.shutdown_calls += 1

        first_executor = _Executor()
        constructor_calls = 0

        def _construct(**_kwargs: object) -> object:
            nonlocal constructor_calls
            constructor_calls += 1
            if constructor_calls == 1:
                return first_executor
            raise RuntimeError('later pool constructor failure sentinel')

        def _implementation() -> None:
            pipeline._create_tracked_thread_pool(
                max_workers=1,
                thread_name_prefix='first-test-pool',
            )
            pipeline._create_tracked_thread_pool(
                max_workers=1,
                thread_name_prefix='second-test-pool',
            )

        with (
            mock.patch.object(pipeline, 'ThreadPoolExecutor', side_effect=_construct),
            mock.patch.object(pipeline, '_main_impl', side_effect=_implementation),
        ):
            with self.assertRaisesRegex(
                RuntimeError, 'later pool constructor failure sentinel',
            ):
                pipeline.main()

        self.assertEqual(constructor_calls, 2)
        self.assertEqual(first_executor.shutdown_calls, 1)

    def test_streaming_reset_refuses_live_producer(self) -> None:
        release = threading.Event()
        media._start_streaming_producer(lambda: release.wait(), name='test-live-producer')
        try:
            with self.assertRaisesRegex(RuntimeError, 'prior producers remain active'):
                media.reset_streaming_state_for_new_run()
        finally:
            release.set()
        self.assertTrue(media.wait_for_streaming_producers(timeout=5.0))
        media.reset_streaming_state_for_new_run()
        self.assertFalse(media.streaming_producers_aborted())

    def test_volume_readiness_does_not_outlive_its_array(self) -> None:
        completed = media.VolumeReadiness(3)
        completed.mark_all_ready()
        array = np.zeros((3, 4, 4), dtype=np.uint8)
        key = id(array)
        media.register_volume_readiness(array, completed)
        self.assertIs(media.volume_readiness(array), completed)
        del array
        gc.collect()
        self.assertNotIn(key, media._VOLUME_READINESS_BY_ARRAY_ID)
        # A later array (possibly at the same address) starts without readiness.
        later = np.zeros((10, 4, 4), dtype=np.uint8)
        self.assertIsNone(media.volume_readiness(later))
        media.wait_for_volume_slice_ready(later, 5)


if __name__ == '__main__':
    unittest.main()
