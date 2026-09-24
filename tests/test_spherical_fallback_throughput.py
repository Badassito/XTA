"""Sparse-shell skips and promotion during a pending CPU block remain exact."""
from concurrent.futures import CancelledError, TimeoutError
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_projection as sp
from XTA import spherical_projection_cpu as compiled
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation


class SparseShellSkipsTests(unittest.TestCase):
    def test_empty_shell_skips_qsc_and_compiled_trigonometry(self):
        shape = (17, 19, 21)
        view = build_spherical_view_infos(*shape, targets=('transverse',),
            min_radius=.5, patch_size=15, tilted_views=())[0]
        source = np.ones((view.num_slices, 7, 11), np.uint8)
        radii = np.asarray(view.spherical_radii)
        rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
        boxes = np.zeros((len(radii), 4), np.int64)
        args = (source, view, radii, rotation, shape, shape[0] // 2, 0, shape[1] * shape[2])
        with mock.patch.object(sp, 'qsc_forward_face', side_effect=AssertionError('QSC of empty shell')):
            self.assertFalse(sp._pull_spherical_chunk(*args, boxes).any())
        kernel_args = compiled._spherical_cpu_kernel_arguments(*args, boxes)
        with mock.patch.object(compiled.math, 'atan2', side_effect=AssertionError('QSC of empty shell')):
            self.assertFalse(compiled._pull_spherical_f64(*kernel_args).any())

    def test_interior_shell_gaps_preserve_binary_scores_and_restored_geometry(self):
        shape = (23, 25, 27)
        rng = np.random.default_rng(2232)
        views = build_spherical_view_infos(*shape, targets=('transverse',),
            min_radius=.5, patch_size=19, tilted_views=())
        qsc_points = [0, 0]
        original = sp.qsc_forward_face
        for initial in views:
            view = replace(initial, spherical_rotation_xyz=cube_rotation('horizontal', -23))
            source = rng.integers(1, 256, (view.num_slices, 7, 11), dtype=np.uint8)
            source[1::2] = 0
            source[:, :1] = 0
            source[:, :, :2] = 0
            boxes = np.tile((1, 7, 2, 11), (view.num_slices, 1)).astype(np.int64)
            boxes[1::2] = 0
            radii = np.asarray(view.spherical_radii)
            rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
            restored = (19, 29, 23)
            for scalar_max in (False, True):
                for z in range(restored[0]):
                    args = (source, view, radii, rotation, restored, z, 0, restored[1] * restored[2])
                    def count_qsc(which):
                        def project(points, face):
                            qsc_points[which] += len(points)
                            return original(points, face)
                        return project
                    with mock.patch.object(sp, 'qsc_forward_face', side_effect=count_qsc(0)):
                        expected = sp._pull_spherical_chunk(*args, scalar_max=scalar_max)
                    with mock.patch.object(sp, 'qsc_forward_face', side_effect=count_qsc(1)):
                        actual = sp._pull_spherical_chunk(*args, boxes, scalar_max=scalar_max)
                    np.testing.assert_array_equal(actual, expected)
                    kernel_args = compiled._spherical_cpu_kernel_arguments(*args, boxes)
                    np.testing.assert_array_equal(
                        compiled._pull_spherical_f64(*kernel_args[:-1], scalar_max), expected)
        self.assertGreater(qsc_points[0], 0)
        self.assertLess(qsc_points[1], qsc_points[0] * .7)

    def test_empty_outer_shell_does_not_steal_exact_inward_midpoint(self):
        view = build_spherical_view_infos(9, 9, 9, targets=('transverse',),
            min_radius=1., patch_size=17, tilted_views=())[0]
        view = replace(view, num_slices=2, spherical_radii=(1.5, 2.5),
            spherical_min_radius=1.5, spherical_max_radius=2.5)
        source = np.zeros((2, 17, 17), np.uint8)
        source[0] = 203
        boxes = np.array(((0, 17, 0, 17), (0, 0, 0, 0)), np.int64)
        args = (source, view, np.asarray(view.spherical_radii), np.eye(3), (9, 9, 9), 4, 4 * 9 + 6, 4 * 9 + 7)
        self.assertEqual(sp._pull_spherical_chunk(*args, boxes, scalar_max=True)[0], 203)
        kernel_args = compiled._spherical_cpu_kernel_arguments(*args, boxes)
        self.assertEqual(compiled._pull_spherical_f64(*kernel_args[:-1], True)[0], 203)


class PendingCpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (7, 9, 11)
        self.view = build_spherical_view_infos(*self.shape, targets=('transverse',),
            min_radius=.7, patch_size=15, tilted_views=())[0]
        self.source = np.ones((self.view.num_slices, 15, 15), np.uint8)
        radii, rotation, _, _ = sp._validate_spherical_projection(self.source, self.view, self.shape, None)
        self.expected = sp._project_spherical_block(self.source, self.view, radii,
            rotation, self.shape, 0, self.shape[0])

    def test_promotes_before_pending_block_finishes_and_joins_all_readers(self):
        for workers in (1, 3):
            for prefix in (0, 1):
                with self.subTest(workers=workers, published_prefix=prefix):
                    self._run_pending_handoff(workers, prefix)

    def test_admission_failure_joins_readers_and_preserves_original_error(self):
        self._run_pending_handoff(3, 0, fail_admission=True)

    def test_compact_promotion_discards_unpublished_dense_or_encoded_cpu_block(self):
        from tests.test_spherical_cpu_compact import EncodedSink
        radii, rotation, _, _ = sp._validate_spherical_projection(
            self.source, self.view, self.shape, None)
        original_encoded = sp._project_spherical_encoded_block
        for cpu_compact in (False, True):
            for packed in (False, True):
                with self.subTest(cpu_compact=cpu_compact, packed=packed):
                    joined = threading.Event()
                    started = threading.Event()
                    def pending(*args, cancel_event, **kwargs):
                        started.set()
                        try:
                            if not cancel_event.wait(3):
                                raise AssertionError('CPU block prevented compact CUDA promotion')
                            raise CancelledError('CPU cancelled')
                        finally:
                            joined.set()
                    def gpu(first, count, *, packed):
                        self.assertTrue(joined.is_set())
                        return original_encoded(self.source, self.view, radii, rotation,
                            self.shape, first, count, packed=packed)
                    stage = SimpleNamespace(device_index=0, max_block_depth=2,
                        projector=SimpleNamespace(), project_encoded=gpu, close=mock.Mock())
                    attempts = 0
                    def admit(*args, **kwargs):
                        nonlocal attempts
                        attempts += 1
                        return stage if attempts > 1 and started.is_set() else None
                    sink = EncodedSink(self.shape, packed=packed)
                    with ExitStack() as stack:
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=admit))
                        stack.enter_context(mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None))
                        stack.enter_context(mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=True))
                        stack.enter_context(mock.patch.object(sp, 'spherical_cpu_compact_enabled', return_value=cpu_compact))
                        stack.enter_context(mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 1)))
                        stack.enter_context(mock.patch.object(sp, '_project_spherical_block', side_effect=pending))
                        stack.enter_context(mock.patch.object(sp, '_project_spherical_encoded_block', side_effect=pending))
                        stack.enter_context(mock.patch.object(sp, '_CUDA_RECHECK_SECONDS', .01))
                        sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'),
                            'pending compact', sink_only=True, projection_block_callback=sink)
                    np.testing.assert_array_equal(sink.result, self.expected)
                    self.assertEqual(sink.seen, list(range(self.shape[0])))
                    sink.abort.assert_not_called()
                    stage.close.assert_called_once()

    def _run_pending_handoff(self, workers, prefix, fail_admission=False):
        started = threading.Event()
        lock = threading.Lock()
        live = 0
        cancelled = []
        seen = []
        result = np.zeros(self.shape, np.uint8)
        failure = ValueError('admission failed while CPU was reading')
        attempts = 0

        def project(*args, cancel_event, **kwargs):
            nonlocal live
            first, count = args[5:7]
            if first < prefix:
                return self.expected[first:first + count].copy()
            with lock:
                live += 1
                if live == workers:
                    started.set()
            try:
                if not cancel_event.wait(3):
                    raise AssertionError('admission was blocked behind the CPU future')
                cancelled.append(first)
                raise CancelledError('CPU chunk cancelled')
            finally:
                with lock:
                    live -= 1

        def gpu(first, count):
            self.assertEqual(live, 0, 'GPU publication started before borrowed CPU readers joined')
            self.assertGreaterEqual(first, prefix)
            return self.expected[first:first + count].copy()

        stage = SimpleNamespace(device_index=0, max_block_depth=2,
            projector=SimpleNamespace(), project=mock.Mock(side_effect=gpu), close=mock.Mock())

        def admit(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return None
            if not started.is_set():
                return None
            self.assertEqual(live, workers)
            self.assertEqual(seen, list(range(prefix)))
            if fail_admission:
                raise failure
            return stage

        def consume(first, block):
            self.assertEqual(first, len(seen))
            seen.extend(range(first, first + len(block)))
            result[first:first + len(block)] = block

        consume.abort = mock.Mock()
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=admit))
            stack.enter_context(mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None))
            stack.enter_context(mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=True))
            stack.enter_context(mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, workers)))
            stack.enter_context(mock.patch.object(sp, '_project_spherical_block', side_effect=project))
            stack.enter_context(mock.patch.object(sp, '_CUDA_RECHECK_SECONDS', .01))
            stack.enter_context(mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1000))
            if fail_admission:
                with self.assertRaises(ValueError) as caught:
                    sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'),
                        'pending failure', workers=workers, sink_only=True, projection_block_callback=consume)
                self.assertIs(caught.exception, failure)
            else:
                sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'),
                    'pending promotion', workers=workers, sink_only=True, projection_block_callback=consume)
        self.assertEqual(live, 0)
        self.assertEqual(len(cancelled), workers)
        self.assertTrue(self.source.all())
        if fail_admission:
            stage.project.assert_not_called()
            stage.close.assert_not_called()
            consume.abort.assert_called_once()
        else:
            stage.close.assert_called_once()
            consume.abort.assert_not_called()
            self.assertEqual(seen, list(range(self.shape[0])))
            np.testing.assert_array_equal(result, self.expected)
            self.assertEqual(stage.project.call_args_list[0].args[0], prefix)
            self.assertIn(f'cpu_slices={prefix}', output.getvalue())
            self.assertGreaterEqual(attempts, 2)
            self.assertIn(f'admission_attempts={attempts}', output.getvalue())

    def test_worker_timeout_error_is_not_mistaken_for_admission_poll(self):
        failure = TimeoutError('source read timed out')
        while_waiting = mock.Mock(return_value=False)
        cancel = threading.Event()
        with mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 1)):
            blocks = sp._ordered_spherical_blocks(mock.Mock(side_effect=failure), 2, 1, 1,
                cancel_event=cancel, while_waiting=while_waiting)
            with self.assertRaises(TimeoutError) as caught:
                next(blocks)
        self.assertIs(caught.exception, failure)
        self.assertTrue(cancel.is_set())
        while_waiting.assert_not_called()


if __name__ == '__main__':
    unittest.main()
