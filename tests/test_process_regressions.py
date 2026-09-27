from __future__ import annotations

import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from XTA import assembly, interpolation, media, pipeline, runtime


class _DetachHandle:
    def __init__(self, fd: int) -> None:
        self.fd = int(fd)

    def detach(self) -> int:
        return int(self.fd)


class _BrokenDetachHandle:
    def detach(self) -> int:
        raise RuntimeError('detach failure sentinel')


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
    def tearDown(self) -> None:
        runtime.set_interpolation_process_executor(None, 0)
        runtime.set_gpu_worker_aux_interpolation_pool(None)
        media.abort_streaming_producers('test teardown')
        media.wait_for_streaming_producers(timeout=5.0)
        media.reset_streaming_state_for_new_run()

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

    def test_partial_memfd_materialization_closes_detached_descriptors(self) -> None:
        first_fd, first_writer = os.pipe()
        os.close(first_writer)
        task = {
            'result_mask_fd': _DetachHandle(first_fd),
            'result_mask_fd_key': 'first',
            'result_conf_fd': _BrokenDetachHandle(),
            'result_conf_fd_key': 'second',
        }
        with self.assertRaisesRegex(RuntimeError, 'detach failure sentinel'):
            runtime._materialize_worker_task_memfd_paths(task, {})
        with self.assertRaises(OSError):
            os.fstat(first_fd)

    def test_policy_memfd_attachment_copies_every_nested_dispatch_dictionary(self) -> None:
        grandchild = {'result_mask_path': 'grandchild-mask'}
        child = {
            'source_volume_path': 'source',
            'result_mask_path': 'policy-mask',
            'result_conf_path': 'policy-conf',
            'canvas_path': 'policy-canvas',
            'd1_bitset_path': 'policy-bitset',
            'native_resize': {'path': 'native-source'},
            'augmentation_pass_tasks': [grandchild],
        }
        canonical = {
            'result_mask_path': 'base-mask',
            'augmentation_pass_tasks': [child],
        }
        first, second = dict(canonical), dict(canonical)
        with mock.patch.object(runtime, '_duplicate_memfd_path_for_child',
                               side_effect=lambda path: object() if path is not None else None):
            runtime._attach_memfd_transfers_to_task(first)
            runtime._attach_memfd_transfers_to_task(second)
        for dispatch in (first, second):
            copied = dispatch['augmentation_pass_tasks'][0]
            self.assertIsNot(dispatch['augmentation_pass_tasks'], canonical['augmentation_pass_tasks'])
            self.assertIsNot(copied, child)
            self.assertIsNot(copied['native_resize'], child['native_resize'])
            self.assertIsNot(copied['augmentation_pass_tasks'][0], grandchild)
            for path_field, handle_field in (
                ('source_volume_path', 'source_volume_fd'),
                ('result_mask_path', 'result_mask_fd'),
                ('result_conf_path', 'result_conf_fd'),
                ('canvas_path', 'canvas_fd'),
                ('d1_bitset_path', 'd1_bitset_fd'),
            ):
                self.assertIn(handle_field, copied)
                self.assertEqual(copied[f'{handle_field}_key'], child[path_field])
                self.assertNotIn(handle_field, child)
            self.assertEqual(copied['native_resize']['path_fd_key'], 'native-source')
            self.assertNotIn('path_fd', child['native_resize'])
            self.assertIn('result_mask_fd', copied['augmentation_pass_tasks'][0])
            self.assertNotIn('result_mask_fd', grandchild)
        self.assertIsNot(first['augmentation_pass_tasks'][0]['result_mask_fd'],
                         second['augmentation_pass_tasks'][0]['result_mask_fd'])
        self.assertNotIn('result_mask_fd', canonical)

    def _read_descriptor(self) -> int:
        reader, writer = os.pipe()
        os.close(writer)
        self.addCleanup(runtime._close_fd_list, [reader])
        return reader

    def test_policy_memfd_materialization_returns_all_result_fds_and_reuses_sources(self) -> None:
        source_fd, repeated_source_fd = self._read_descriptor(), self._read_descriptor()
        native_fd, repeated_native_fd = self._read_descriptor(), self._read_descriptor()
        result_fds = [self._read_descriptor() for _ in range(5)]
        grandchild = {'result_mask_fd': _DetachHandle(result_fds[4]),
                      'result_mask_path': 'grandchild-mask'}
        child = {
            'source_volume_fd': _DetachHandle(repeated_source_fd), 'source_volume_fd_key': 'source',
            'result_mask_fd': _DetachHandle(result_fds[1]), 'result_mask_path': 'policy-mask',
            'result_conf_fd': _DetachHandle(result_fds[2]), 'result_conf_path': 'policy-conf',
            'canvas_fd': _DetachHandle(result_fds[3]), 'canvas_path': 'policy-canvas',
            'native_resize': {'path': 'native-source', 'path_fd': _DetachHandle(repeated_native_fd)},
            'augmentation_pass_tasks': [grandchild],
        }
        incoming = {
            'source_volume_fd': _DetachHandle(source_fd), 'source_volume_fd_key': 'source',
            'result_mask_fd': _DetachHandle(result_fds[0]),
            'native_resize': {'path': 'native-source', 'path_fd': _DetachHandle(native_fd)},
            'augmentation_pass_tasks': [child],
        }
        local = dict(incoming)
        persistent = {}
        transient = runtime._materialize_worker_task_memfd_paths(local, persistent)
        self.assertEqual(transient, result_fds)
        self.assertEqual(persistent, {'source': source_fd, 'native-source': native_fd})
        copied = local['augmentation_pass_tasks'][0]
        self.assertEqual(copied['source_volume_path'], local['source_volume_path'])
        self.assertEqual(copied['native_resize']['path'], local['native_resize']['path'])
        for holder, field, fd in (
            (local, 'result_mask_path', result_fds[0]),
            (copied, 'result_mask_path', result_fds[1]),
            (copied, 'result_conf_path', result_fds[2]),
            (copied, 'canvas_path', result_fds[3]),
            (copied['augmentation_pass_tasks'][0], 'result_mask_path', result_fds[4]),
        ):
            self.assertEqual(holder[field], str(runtime._memfd_proc_path(fd, owner_pid=os.getpid())))
            os.fstat(fd)
        self.assertEqual(child['result_mask_path'], 'policy-mask')
        self.assertEqual(child['native_resize']['path'], 'native-source')
        self.assertEqual(grandchild['result_mask_path'], 'grandchild-mask')
        self.assertIn('result_mask_fd', child)
        self.assertIn('result_mask_fd', grandchild)
        for fd in (repeated_source_fd, repeated_native_fd):
            with self.assertRaises(OSError):
                os.fstat(fd)
        runtime._close_fd_list(transient)
        for fd in result_fds:
            with self.assertRaises(OSError):
                os.fstat(fd)
        os.fstat(source_fd)
        os.fstat(native_fd)

    def test_policy_memfd_failure_rolls_back_whole_group_and_preserves_existing_sources(self) -> None:
        cached_fd = self._read_descriptor()
        source_duplicate = self._read_descriptor()
        new_source, new_native = self._read_descriptor(), self._read_descriptor()
        result_fds = [self._read_descriptor() for _ in range(3)]
        child = {
            'source_volume_fd': _DetachHandle(new_source), 'source_volume_fd_key': 'new-source',
            'result_mask_fd': _DetachHandle(result_fds[1]), 'result_mask_path': 'policy-mask',
            'native_resize': {'path': 'new-native', 'path_fd': _DetachHandle(new_native)},
            'augmentation_pass_tasks': [{
                'result_mask_fd': _DetachHandle(result_fds[2]),
                'result_conf_fd': _BrokenDetachHandle(),
            }],
        }
        local = {
            'source_volume_fd': _DetachHandle(source_duplicate), 'source_volume_fd_key': 'cached',
            'result_mask_fd': _DetachHandle(result_fds[0]),
            'augmentation_pass_tasks': [child],
        }
        persistent = {'cached': cached_fd}
        with self.assertRaisesRegex(RuntimeError, 'detach failure sentinel'):
            runtime._materialize_worker_task_memfd_paths(local, persistent)
        self.assertEqual(persistent, {'cached': cached_fd})
        os.fstat(cached_fd)
        for fd in (source_duplicate, new_source, new_native, *result_fds):
            with self.assertRaises(OSError):
                os.fstat(fd)
        self.assertEqual(child['result_mask_path'], 'policy-mask')
        self.assertEqual(child['native_resize']['path'], 'new-native')
        self.assertIn('result_mask_fd', child)

    @unittest.skipUnless(hasattr(os, 'memfd_create') and Path('/proc/self/fd').exists(),
                         'real memfd descriptor transfer requires Linux procfs')
    def test_policy_memfd_transfers_reopen_independent_parent_arrays(self) -> None:
        shape = (2, 3, 4)
        owners, reopened, transient, persistent = [], [], [], {}
        try:
            with mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES': '1'}):
                for index in range(7):
                    owners.append(runtime._allocate_memfd_workspace_array(
                        shape, np.uint8, f'policy-transfer-{index}', initialize_zero=True))
            members = []
            for index in range(3):
                members.append({
                    'source_volume_path': str(runtime._memmap_backing_path(owners[0])),
                    'result_mask_path': str(runtime._memmap_backing_path(owners[1 + 2 * index])),
                    'result_conf_path': str(runtime._memmap_backing_path(owners[2 + 2 * index])),
                })
            members[0]['augmentation_pass_tasks'] = members[1:]
            dispatch = dict(members[0])
            runtime._attach_memfd_transfers_to_task(dispatch)
            local = dict(dispatch)
            transient = runtime._materialize_worker_task_memfd_paths(local, persistent)
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
                keep_temp=False,
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
            self.assertEqual(list(work_dir.glob('*fallback-stage*')), [])
        finally:
            runtime.close_memmap_array(original)

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
                keep_temp=False,
                workers=1,
            )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(np.asarray(original), np.full((2, 2, 2), 9, dtype=np.uint8))
            self.assertTrue(stats['worker_completed'])
            self.assertEqual(list(work_dir.glob('*fallback-stage*')), [])
        finally:
            runtime.close_memmap_array(original)

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
        finally:
            runtime.close_memmap_array(original)

    def test_aux_queue_pickle_failure_rolls_back_pending_lease(self) -> None:
        class _Queue:
            def __init__(self) -> None:
                self.items: list[object] = []

            def put(self, item: object) -> None:
                self.items.append(item)

        task_queue = _Queue()
        pool = runtime._GpuWorkerAuxInterpolationPool({0: task_queue})
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


if __name__ == '__main__':
    unittest.main()
