"""Compiled float64 Spherical pull preserves categorical geometry and thin masks."""
from dataclasses import replace
from itertools import product
from concurrent.futures import CancelledError
import contextlib
import io
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_projection as reference
from XTA import spherical_projection_cpu as candidate
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import spherical_output_bounds
from tests.reference_backends import spherical as oracle


def project_with_pull(source, view, shape, pull, boxes=None):
    radii = np.asarray(view.spherical_radii, np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    output = np.empty(shape, np.uint8)
    for z in range(shape[0]):
        plane = output[z].reshape(-1)
        for first in range(0, len(plane), 2048):
            stop = min(len(plane), first + 2048)
            plane[first:stop] = pull(source, view, radii, rotation, shape, z, first, stop, boxes)
    return output


def thin_source(view, size):
    """One-pixel lines cross diagonals, patch edges and radius slices."""
    source = np.zeros((view.num_slices, size, size), np.uint8)
    source[:, size // 2, :] = 1
    source[:, :, size // 2] = 1
    source[:, np.arange(size), np.arange(size)] = 1
    source[:, np.arange(size), size - 1 - np.arange(size)] = 1
    source[:, (0, size - 1), :] = 1
    source[:, :, (0, size - 1)] = 1
    return source


class SphericalNativeAvailabilityTests(unittest.TestCase):
    def test_kernel_is_a_compiled_extension_entry_point(self):
        from types import BuiltinFunctionType
        from XTA.native_cpu import load_native_cpu

        extension = load_native_cpu()
        self.assertIsInstance(extension.spherical_pull, BuiltinFunctionType)
        self.assertIs(candidate.kernels.spherical_pull, extension.spherical_pull)


class SphericalCompiledPullTests(unittest.TestCase):
    def test_faces_rotations_padding_and_restoration_match_numpy_labels(self):
        views = build_spherical_view_infos(33, 35, 37, targets=('transverse',),
                                           min_radius=.5, patch_size=29, tilted_views=())
        rng = np.random.default_rng(312)
        for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
            for initial in views:
                view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                source = rng.integers(0, 4, (view.num_slices, 7, 11), dtype=np.uint8)
                boxes = np.zeros((view.num_slices, 4), np.int64)
                boxes[2:-2] = (1, 6, 2, 10)
                for shape, metadata in (((33, 35, 37), None), ((23, 39, 31), boxes)):
                    with self.subTest(face=view.spherical_face, origin=(view.spherical_u_origin,
                            view.spherical_v_origin), rotation=rotation, shape=shape):
                        expected = project_with_pull(source, view, shape, oracle.pull_spherical_chunk, metadata)
                        actual = project_with_pull(source, view, shape, candidate.pull_spherical_chunk_native, metadata)
                        np.testing.assert_array_equal(actual, expected)

    def test_thin_lines_edges_and_corners_preserve_every_voxel_and_connectivity(self):
        shape = (35, 37, 39)
        views = build_spherical_view_infos(*shape, targets=('transverse',),
                                           min_radius=.5, patch_size=55, tilted_views=())
        for rotation, view in product((None, cube_rotation('vertical', 30), cube_rotation('horizontal', -30)), views):
            if rotation:
                view = replace(view, spherical_rotation_xyz=rotation)
            source = thin_source(view, 55)
            expected = project_with_pull(source, view, shape, oracle.pull_spherical_chunk)
            actual = project_with_pull(source, view, shape, candidate.pull_spherical_chunk_native)
            with self.subTest(face=view.spherical_face, rotation=rotation):
                self.assertGreater(np.count_nonzero(expected), 0)
                # Exact binary pixels also preserve every connected component.
                np.testing.assert_array_equal(actual, expected)

    def test_large_face_lattice_uses_local_patch_indices(self):
        view = build_spherical_view_infos(9, 9, 9, targets=('transverse',),
                                          min_radius=1., patch_size=17, tilted_views=())[0]
        intervals = 1 << 40
        view = replace(view, full_t=1, full_h=9, full_w=9, num_slices=1,
                       src_h=7, src_w=7, spherical_face=0, spherical_face_intervals=intervals,
                       spherical_u_origin=intervals // 2 - 2, spherical_v_origin=intervals // 2 - 3,
                       spherical_min_radius=3., spherical_max_radius=3., spherical_radii=(3.,))
        source = np.zeros((1, 7, 7), np.uint8)
        source[0, 3, 2] = 117
        result = candidate.pull_spherical_chunk_native(
            source, view, np.array([3.]), np.eye(3), (1, 3, 3), 0, 5, 6, scalar_max=True)
        np.testing.assert_array_equal(result, np.array([117], np.uint8))

    def test_nearest_shell_midpoint_keeps_inward_tie(self):
        view = build_spherical_view_infos(9, 9, 9, targets=('transverse',),
                                          min_radius=1., patch_size=17, tilted_views=())[0]
        view = replace(view, num_slices=2, spherical_radii=(1.5, 2.5),
                       spherical_min_radius=1.5, spherical_max_radius=2.5)
        source = np.zeros((2, 17, 17), np.uint8)
        source[0] = 1
        expected = project_with_pull(source, view, (9, 9, 9), oracle.pull_spherical_chunk)
        actual = project_with_pull(source, view, (9, 9, 9), candidate.pull_spherical_chunk_native)
        self.assertEqual(actual[4, 4, 6], 1)
        np.testing.assert_array_equal(actual, expected)

    def test_flat_output_addresses_beyond_int32_do_not_wrap(self):
        shape = (3, 65537, 65537)
        view = build_spherical_view_infos(*shape, targets=('transverse',),
                                          min_radius=1., patch_size=7, tilted_views=())[0]
        source = np.ones((1, 1, 1), np.uint8)
        first = 32768 * shape[2] + 32769
        self.assertGreater(first, 2**31)
        args = (source, view, np.asarray(view.spherical_radii), np.eye(3), shape, 1, first, first + 1)
        np.testing.assert_array_equal(candidate.pull_spherical_chunk_native(*args), np.ones(1, np.uint8))
        np.testing.assert_array_equal(candidate.pull_spherical_chunk_native(*args), oracle.pull_spherical_chunk(*args))

    def test_borrowed_strided_readonly_masks_and_empty_strips(self):
        view = build_spherical_view_infos(15, 17, 19, targets=('transverse',),
                                          min_radius=.5, patch_size=25, tilted_views=())[0]
        source = np.ones((view.num_slices, 11, 13), np.uint8)[:, ::2, ::2]
        source.flags.writeable = False
        args = (source, view, np.asarray(view.spherical_radii), np.eye(3), (15, 17, 19), 7, 0, 17 * 19)
        np.testing.assert_array_equal(candidate.pull_spherical_chunk_native(*args), oracle.pull_spherical_chunk(*args))
        self.assertFalse(source.flags.writeable)
        self.assertEqual(candidate.pull_spherical_chunk_native(*args[:-2], 0, 0).size, 0)


class SphericalCompiledAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.shape = (15, 17, 19)
        self.view = build_spherical_view_infos(*self.shape, targets=('transverse',),
                                               min_radius=.5, patch_size=25, tilted_views=())[0]
        self.source = np.ones((self.view.num_slices, 11, 13), np.uint8)
        self.radii = np.asarray(self.view.spherical_radii)
        self.rotation = np.asarray(self.view.spherical_rotation_xyz).reshape(3, 3)
        self.arguments = (self.source, self.view, self.radii, self.rotation, self.shape, None)

    def test_default_selects_exact_compiled_cpu_when_available(self):
        pull = object()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(candidate, 'prepare_spherical_chunk_native', return_value=pull) as prepare:
            self.assertIs(reference._select_spherical_cpu_pull(*self.arguments), pull)
        prepare.assert_called_once()

    def test_fast_geometry_opt_out_cannot_disable_required_compiler(self):
        with mock.patch.dict(os.environ, {'YOLO_TTA_FAST_GEOMETRY': '0'}), \
                mock.patch.object(candidate, 'prepare_spherical_chunk_native', return_value=object()) as prepare:
            reference._select_spherical_cpu_pull(*self.arguments)
        prepare.assert_called_once()

    def test_missing_native_entry_point_fails_before_publication(self):
        with mock.patch.object(candidate, 'kernels', SimpleNamespace()):
            with self.assertRaisesRegex(candidate.SphericalCpuProjectionUnavailable, 'Native Spherical projection unavailable'):
                reference._select_spherical_cpu_pull(*self.arguments)

    def test_native_load_failure_precedes_evaluation(self):
        class UnavailableBindings:
            @property
            def spherical_pull(self):
                raise candidate.NativeCpuBuildError('binary unavailable')

        with mock.patch.object(candidate, 'kernels', UnavailableBindings()):
            with self.assertRaisesRegex(candidate.SphericalCpuProjectionUnavailable, 'binary unavailable'):
                reference._select_spherical_cpu_pull(*self.arguments)

    def test_pipeline_native_load_failure_precedes_sink_publication(self):
        sink = mock.Mock()
        failure = candidate.SphericalCpuProjectionUnavailable('binary unavailable')
        with mock.patch.object(reference, '_try_spherical_cuda_stage', return_value=None), \
                mock.patch.object(candidate, 'prepare_spherical_chunk_native', side_effect=failure), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(candidate.SphericalCpuProjectionUnavailable, 'binary unavailable'):
                reference.backproject_spherical_volume_to_volume(
                    self.source, self.view, Path('unused'), 'compile failure',
                    sink_only=True, projection_block_callback=sink)
        sink.assert_not_called()

    def test_metadata_errors_are_not_reclassified_as_compiler_fallback(self):
        with self.assertRaisesRegex(ValueError, 'inconsistent shapes'):
            reference._select_spherical_cpu_pull(self.source, self.view, self.radii[:-1],
                                                 self.rotation, self.shape, None)

    def test_unbounded_public_helper_uses_required_compiled_pull(self):
        compiled_pull = mock.Mock(wraps=candidate.pull_spherical_chunk_native)
        with mock.patch.object(reference, '_select_spherical_cpu_pull', return_value=compiled_pull) as select:
            actual = reference._project_spherical_block(self.source, self.view, self.radii,
                self.rotation, self.shape, 7, 1)
        select.assert_called_once()
        self.assertTrue(compiled_pull.called)
        np.testing.assert_array_equal(actual, oracle.project_spherical_block(
            self.source, self.view, self.radii, self.rotation, self.shape, 7, 1))
        self.assertGreater(np.count_nonzero(actual), 0)

    def test_enabled_bounded_projection_preserves_readonly_strided_source_and_order(self):
        source = self.source[::-1, ::-2, ::-2]
        source.flags.writeable = False
        expected = project_with_pull(source, self.view, self.shape, oracle.pull_spherical_chunk)
        blocks = []
        with mock.patch.object(reference, '_try_spherical_cuda_stage', return_value=None), \
                mock.patch.object(reference, 'spherical_cuda_backproject_enabled', return_value=False), \
                mock.patch.object(reference, '_OUTPUT_BLOCK_BYTES', self.shape[1] * self.shape[2] * 2), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            reference.backproject_spherical_volume_to_volume(source, self.view, Path('unused'), 'compiled test',
                workers=2, sink_only=True, projection_block_callback=lambda first, block: blocks.append((first, block.copy())))
        self.assertEqual([first for first, _ in blocks], list(range(0, self.shape[0], 2)))
        np.testing.assert_array_equal(np.concatenate([block for _, block in blocks]), expected)
        self.assertFalse(source.flags.writeable)
        self.assertIn('backend=cpu_native_f64_bounded', output.getvalue())
        self.assertIn('cpu_setup_seconds=', output.getvalue())

    def test_cancellation_still_stops_at_the_next_bounded_chunk(self):
        cancel = threading.Event()
        calls = []

        def pull(*args):
            calls.append((args[6], args[7]))
            result = candidate.pull_spherical_chunk_native(*args)
            cancel.set()
            return result

        with mock.patch.object(reference, '_PULL_CHUNK_VOXELS', 16):
            with self.assertRaises(CancelledError):
                reference._project_spherical_block(self.source, self.view, self.radii, self.rotation,
                    self.shape, 7, 1, output_bounds=spherical_output_bounds(self.view, self.shape),
                    cancel_event=cancel, cpu_pull=pull)
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0][1] - calls[0][0], 16)

    def test_runtime_numerical_failure_propagates_without_fallback_or_publication(self):
        publish = mock.Mock()
        pull = mock.Mock(side_effect=ValueError('numerical failure'))
        with mock.patch.object(reference, '_select_spherical_cpu_pull', return_value=pull), \
                mock.patch.object(reference, '_try_spherical_cuda_stage', return_value=None), \
                mock.patch.object(reference, 'spherical_cuda_backproject_enabled', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'numerical failure'):
                reference.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'failure',
                    sink_only=True, projection_block_callback=publish)
        publish.assert_not_called()

    def test_initial_cuda_admission_does_not_load_cpu_or_change_its_projection(self):
        stage = SimpleNamespace(max_block_depth=2, device_index=0, close=mock.Mock(),
            project=lambda first, count: np.zeros((count, *self.shape[1:]), np.uint8), projector=SimpleNamespace())
        with mock.patch.object(reference, '_try_spherical_cuda_stage', return_value=stage), \
                mock.patch.object(reference, '_select_spherical_cpu_pull', side_effect=AssertionError('CPU setup on GPU')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            reference.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'CUDA admission',
                sink_only=True, projection_block_callback=lambda *_args: None)
        stage.close.assert_called_once()
        self.assertIn('cpu_backend=unused', output.getvalue())


if __name__ == '__main__':
    unittest.main()
