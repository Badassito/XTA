"""Available-GPU qualification of certified, reduced projection sampling.

The tests skip when Torch CUDA or CuPy is absent. GPU geometry remains FP64;
native intensity comparisons use the registered CUDA one-gray-level tolerance.
Terminal binary masks must match exactly. Point CUPY_CACHE_DIR, NUMBA_CACHE_DIR,
and YOLO_CONFIG_DIR into the task's Scratch directory when running locally.
"""
from __future__ import annotations

from collections import defaultdict
import math
import os
import unittest
from unittest import mock

import numpy as np

from XTA import geometry
from XTA.config import resolve_tilted_view_groups


def _views(shape, size, *, expanded=False):
    bases = ('transverse', 'sagittal', 'coronal') if expanded else ('transverse',)
    tilted = ('tilted_transverse',) if expanded else ()
    return [v for v in geometry.get_view_infos(
        *shape, cartesian_views=(), radial_views=bases+tilted,
        radial_patch_size=size, radial_min_radius=2.,
        spherical_views=('transverse',)+tilted, spherical_patch_size=size,
        spherical_min_radius=2., sampling_policy='coverage',
        tilt_groups=resolve_tilted_view_groups(['transverse:30:vertical']) if expanded else (),
    ) if v.family in ('radial', 'spherical') and v.tilt_angle_deg >= 0.]


def _domain(view):
    """Independent source-voxel center membership, without sampled shell indices."""
    t, y, x = np.indices((view.full_t, view.full_h, view.full_w))
    if view.family == 'spherical':
        radius = np.sqrt((t-(view.full_t-1)/2.)**2 + (y-(view.full_h-1)/2.)**2
                         + (x-(view.full_w-1)/2.)**2)
        return (radius >= view.spherical_min_radius) & (radius <= view.spherical_max_radius)
    if view.radial_base_view == 'transverse':
        stack, py, px, length = t, y, x, view.full_t
    elif view.radial_base_view == 'sagittal':
        stack, py, px, length = y, t, x, view.full_h
    else:
        stack, py, px, length = x, t, y, view.full_w
    dx, dy = px-view.center_x, py-view.center_y
    radius = np.hypot(dx, dy)
    height = stack.astype(np.float64)
    if view.radial_tilted_source:
        height -= math.tan(math.radians(view.tilt_angle_deg)) * (
            dy if view.tilt_direction == 'vertical' else dx)
    return ((radius >= view.radial_min_radius) & (radius <= view.radial_max_radius)
            & (height >= 0.) & (height <= length-1))


def _project_pair(data, view, shape):
    """Actual CPU reference and CUDA projector, with three-slice output blocks."""
    if view.family == 'spherical':
        from XTA.spherical_projection import _project_spherical_block
        from XTA.spherical_projection_cuda import SphericalCudaProjector
        expected = _project_spherical_block(data, view, np.asarray(view.spherical_radii),
            np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], None)
        projector = SphericalCudaProjector(data, view, shape, reserve_bytes=0,
                                          block_bytes=shape[1]*shape[2]*3)
    else:
        from XTA import cylindrical_projection as reference
        from XTA.cylindrical_cuda_projection import RadialCudaProjector
        radii = np.asarray(geometry.radial_global_radii(view))
        expected = np.stack([reference._pull_radial_chunk(data, view, radii, shape, z,
            0, shape[1]*shape[2]).reshape(shape[1:]) for z in range(shape[0])])
        plan = reference._build_radial_plane_plan(view, radii, shape)
        metadata = reference._radial_projection_metadata(view, data.shape, shape, plan)
        projector = RadialCudaProjector(data, plan, metadata, view, shape, reserve_bytes=0,
                                        block_bytes=shape[1]*shape[2]*3)
    with projector:
        actual = np.concatenate([projector.project(z, min(3, shape[0]-z))
                                 for z in range(0, shape[0], 3)])
    return expected, actual


def _ideal_azimuthal_plane(volume, view, index):
    """Independent FP64 bilinear intensity before the CPU gray8 boundary.

    Upright native height rows coincide with integer source stack coordinates.
    CUDA texture output remains fractional, unlike CPU native-frame uint8.
    """
    base = geometry.azimuthal_base_view_name(view)
    stack = volume if base == 'transverse' else (
        volume.transpose(1, 0, 2) if base == 'sagittal' else volume.transpose(2, 0, 1))
    if view.src_h != stack.shape[0]:
        raise AssertionError('This independent oracle requires native integer height sampling')
    theta = math.radians(view.azimuths_deg[index])
    arc = np.linspace(-view.roi_radius, view.roi_radius, view.src_w)
    x, y = view.center_x + arc*math.cos(theta), view.center_y + arc*math.sin(theta)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    dx, dy = x-x0, y-y0
    result = np.zeros((view.src_h, view.src_w), np.float64)
    for iy in (0, 1):
        for ix in (0, 1):
            weight = (dy if iy else 1.-dy) * (dx if ix else 1.-dx)
            result += stack[:, np.clip(y0+iy, 0, stack.shape[1]-1),
                            np.clip(x0+ix, 0, stack.shape[2]-1)] * weight
    return result


class ProjectionSamplingCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import cupy
            if not torch.cuda.is_available() or cupy.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest('CUDA GPU unavailable')
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f'Torch CUDA/CuPy unavailable: {error}') from error
        except Exception as error:
            if isinstance(error, unittest.SkipTest):
                raise
            raise unittest.SkipTest(f'CUDA initialization unavailable: {error}') from error
        cls.torch = torch

    def setUp(self):
        flags = mock.patch.dict(os.environ, {
            'YOLO_TTA_FAST_GEOMETRY': '0', 'YOLO_TTA_GPU_SPHERICAL_FP32': '0',
            'YOLO_TTA_GPU_RADIAL_NATIVE_KERNEL': '1', 'YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL': '1',
        })
        flags.start()
        self.addCleanup(flags.stop)

    def test_optimized_spherical_and_radial_native_cuda_inputs_match_cpu(self):
        from XTA.cuda_backend import _GpuWorkerRenderEngine
        shape = (31, 33, 35)
        volume = np.random.default_rng(1206).integers(0, 256, shape, dtype=np.uint8)
        engine = _GpuWorkerRenderEngine('cuda:0')
        try:
            self.assertEqual(engine.ensure_volume_array(volume), 'resident')
            for view in _views(shape, 32, expanded=True):
                self.assertEqual(view.sampling_policy, 'coverage')
                step = view.radial_step if view.family == 'radial' else view.spherical_step
                self.assertGreater(step, 1.)
                for index in sorted({0, view.num_slices//2, view.num_slices-1}):
                    with self.subTest(view=view.name, index=index):
                        with self.torch.cuda.stream(engine._stream):
                            actual = engine._render_native_plane(view, index)
                        engine._stream.synchronize()
                        expected = geometry.get_view_frame_by_index(volume, view, index)
                        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1., rtol=0.)
            self.assertTrue(engine._radial_native_kernel_announced)
            self.assertTrue(engine._spherical_native_kernel_announced)
            self.assertFalse(getattr(engine, '_radial_native_kernel_disabled', False))
            self.assertFalse(getattr(engine, '_spherical_native_kernel_disabled', False))
        finally:
            engine._stream.synchronize()
            engine.release_inference_assets()

    def test_optimized_azimuthal_native_cuda_inputs_match_cpu_on_all_axes(self):
        from XTA.backprojection import azimuthal_full_coverage_angle_deg
        from XTA.cuda_backend import _GpuWorkerRenderEngine
        shape = (31, 33, 35)
        bases = ('transverse', 'sagittal', 'coronal')
        angles = tuple(azimuthal_full_coverage_angle_deg(geometry.azimuthal_target_diameter(base, *shape))
                       for base in bases)
        views = geometry.get_view_infos(*shape, cartesian_views=(), azimuthal_views=bases,
            azimuthal_azimuth_angles=angles, azimuthal_auto_views=bases, sampling_policy='coverage')
        # The same selected angles were already legal as an explicit user sweep.
        # They exercise the legacy forward path independently of coverage metadata.
        explicit = geometry.get_view_infos(*shape, cartesian_views=(), azimuthal_views=bases,
            azimuthal_azimuth_angles=tuple(v.azimuths_deg[1] for v in views), sampling_policy='dense')
        volume = np.random.default_rng(1307).integers(0, 256, shape, dtype=np.uint8)
        engine = _GpuWorkerRenderEngine('cuda:0')
        maximum_error = 0.
        compared = 0
        try:
            self.assertEqual(engine.ensure_volume_array(volume), 'resident')
            for view, legacy in zip(views, explicit):
                self.assertEqual(view.sampling_policy, 'coverage')
                self.assertEqual(legacy.sampling_policy, 'dense')
                self.assertEqual(view.azimuths_deg, legacy.azimuths_deg)
                for index in sorted({0, 1, view.num_slices//2, view.num_slices-1}):
                    with self.subTest(view=view.name, index=index):
                        with self.torch.cuda.stream(engine._stream):
                            actual = engine._render_native_plane(view, index)
                            old = engine._render_native_plane(legacy, index)
                        engine._stream.synchronize()
                        actual = actual.cpu().numpy()
                        expected = _ideal_azimuthal_plane(volume, view, index)
                        np.testing.assert_array_equal(actual, old.cpu().numpy())
                        np.testing.assert_allclose(actual, expected, atol=1., rtol=0.)
                        maximum_error = max(maximum_error, float(np.max(np.abs(actual-expected))))
                        compared += 1
            print(f'Azimuthal CUDA sampling: {compared} exact explicit-angle baseline frames; '
                  f'FP64 intensity max error={maximum_error:.6g} gray levels.')
        finally:
            engine._stream.synchronize()
            engine.release_inference_assets()

    def test_all_one_native_masks_project_to_exact_declared_domains(self):
        shape = (31, 33, 35)
        groups = defaultdict(list)
        for view in _views(shape, 32, expanded=True):
            groups[(view.family, view.radial_base_view, view.tilt_direction, view.tilt_angle_deg)].append(view)
        for key, views in groups.items():
            union = np.zeros(shape, np.uint8)
            for view in views:
                self.assertEqual(view.sampling_policy, 'coverage')
                data = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
                with self.subTest(group=key, view=view.name):
                    expected, actual = _project_pair(data, view, shape)
                    np.testing.assert_array_equal(actual, expected)
                    union |= actual
            with self.subTest(group=key):
                np.testing.assert_array_equal(union, _domain(views[0]).astype(np.uint8))

    def test_random_native_masks_match_cpu_for_coarse_shells_and_rescaled_outputs(self):
        rng = np.random.default_rng(1408)
        for shape, size in (((19, 21, 23), 32), ((31, 33, 35), 64)):
            for view in _views(shape, size):
                self.assertEqual(view.sampling_policy, 'coverage')
                step = view.radial_step if view.family == 'radial' else view.spherical_step
                self.assertGreater(step, 1.)
                data = (rng.random((view.num_slices, size-3, size-5)) < .31).astype(np.uint8)
                for output_shape in (shape, tuple(n+1 for n in shape)):
                    with self.subTest(view=view.name, size=size, output_shape=output_shape):
                        expected, actual = _project_pair(data, view, output_shape)
                        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
