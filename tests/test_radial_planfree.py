"""The bounded compiled Radial fallback preserves source-coordinate semantics."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from tests.reference_backends.radial import pull_radial_chunk
from XTA import cylindrical_projection as cp, geometry
from XTA.config import TiltedViewGroup


class RadialPlanfreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cpu_only = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_RADIAL_BACKPROJECT': '0'})
        self.cpu_only.start()
        self.addCleanup(self.cpu_only.stop)

    def _project(self, source, view, shape, *, boxes=None, sink=None):
        blocks = []
        def consume(first, block):
            blocks.append((first, block.copy()))
            if sink is not None:
                sink(first, block)
        with (
            mock.patch.object(cp, '_PLANE_PLAN_MAX_BYTES', 1),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = cp.backproject_radial_volume_to_volume(
                source, view, Path('unused.dat'), 'plan-free test',
                out_shape_tyx=shape, workers=2, sink_only=True,
                known_slice_bboxes=boxes, projection_block_callback=consume,
            )
        self.assertIn('backend=cpu_numba_planfree', output.getvalue())
        self.assertEqual(result.shape, shape)
        self.assertEqual([first for first, _ in blocks], sorted(first for first, _ in blocks))
        return np.concatenate([block for _, block in blocks], axis=0)

    def test_plan_budget_refusal_matches_independent_oracle(self) -> None:
        shape = (7, 9, 11)
        rng = np.random.default_rng(893)
        for base in ('transverse', 'sagittal', 'coronal'):
            for tilted in (False, True):
                groups = ([TiltedViewGroup((base,), (23.0,), ('vertical', 'horizontal'))]
                          if tilted else [])
                views = [view for view in geometry.get_view_infos(
                    *shape, cartesian_views=(),
                    radial_views=(('tilted_' if tilted else '') + base,),
                    radial_min_radius=.7, radial_patch_size=7, tilt_groups=groups,
                ) if view.family == 'radial']
                for view in views[::max(1, len(views) // 3)]:
                    view = replace(view, radial_arc_origin=view.radial_arc_origin + .37)
                    source = (rng.random((view.num_slices, 5, 6)) < .22).astype(np.uint8)
                    radii = np.asarray(geometry.radial_global_radii(view), np.float64)
                    for output_shape in (shape, (5, 7, 8)):
                        expected = np.stack([
                            pull_radial_chunk(source, view, radii, output_shape, z,
                                              0, output_shape[1] * output_shape[2])
                            .reshape(output_shape[1:])
                            for z in range(output_shape[0])
                        ])
                        with self.subTest(base=base, tilted=tilted,
                                          view=view.name, output_shape=output_shape):
                            actual = self._project(source, view, output_shape)
                            np.testing.assert_array_equal(actual, expected)

    def test_planfree_bbox_crop_matches_zeroed_reference(self) -> None:
        shape = (5, 7, 9)
        view = next(view for view in geometry.get_view_infos(
            *shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.7, radial_patch_size=6,
        ) if view.family == 'radial')
        source = np.ones((view.num_slices, 5, 7), np.uint8)
        boxes = np.tile(np.asarray((1, 4, 2, 6), np.int64), (view.num_slices, 1))
        cropped = np.zeros_like(source)
        cropped[:, 1:4, 2:6] = 1
        radii = np.asarray(geometry.radial_global_radii(view), np.float64)
        expected = np.stack([
            pull_radial_chunk(cropped, view, radii, shape, z, 0, shape[1] * shape[2])
            .reshape(shape[1:]) for z in range(shape[0])
        ])
        np.testing.assert_array_equal(self._project(source, view, shape, boxes=boxes), expected)

    def test_tiny_period_and_strided_source_match_oracle(self) -> None:
        shape = (4, 7, 7)
        view = next(view for view in geometry.get_view_infos(
            *shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.01, radial_patch_size=16,
        ) if view.family == 'radial')
        source = np.random.default_rng(94).integers(
            0, 2, (view.num_slices, 5, 14), dtype=np.uint8)[:, :, ::2]
        self.assertFalse(source.flags.c_contiguous)
        radii = np.asarray(geometry.radial_global_radii(view), np.float64)
        expected = np.stack([
            pull_radial_chunk(source, view, radii, shape, z, 0, shape[1] * shape[2])
            .reshape(shape[1:]) for z in range(shape[0])
        ])
        np.testing.assert_array_equal(self._project(source, view, shape), expected)

    def test_compiled_chunk_preserves_float32_scalar_max_values(self) -> None:
        shape = (5, 7, 9)
        view = next(view for view in geometry.get_view_infos(
            *shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.01, radial_patch_size=12,
        ) if view.family == 'radial')
        source = np.random.default_rng(95).uniform(
            0, 100, (view.num_slices, 5, 12)).astype(np.float32)
        source[:, ::3, ::4] = 0
        radii = np.asarray(geometry.radial_global_radii(view), np.float64)
        for z in range(shape[0]):
            for first in (0, 17, 35):
                stop = min(shape[1] * shape[2], first + 23)
                with self.subTest(z=z, first=first):
                    expected = pull_radial_chunk(
                        source, view, radii, shape, z, first, stop, scalar_max=True)
                    actual = cp._pull_radial_chunk_compiled(
                        source, view, radii, shape, z, first, stop, scalar_max=True)
                    self.assertEqual(actual.dtype, np.float32)
                    np.testing.assert_array_equal(actual, expected)

    def test_sink_failure_stops_publication_and_preserves_borrowed_source(self) -> None:
        shape = (5, 7, 9)
        view = next(view for view in geometry.get_view_infos(
            *shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.7, radial_patch_size=6,
        ) if view.family == 'radial')
        source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
        delivered = []
        failure = RuntimeError('sink stopped')
        def consume(first, block):
            delivered.append(first)
            raise failure
        with (
            mock.patch.object(cp, '_PLANE_PLAN_MAX_BYTES', 1),
            mock.patch.object(cp, '_radial_block_schedule', return_value=(1, 2)),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError) as caught:
                cp.backproject_radial_volume_to_volume(
                    source, view, Path('unused.dat'), 'plan-free cancelled',
                    sink_only=True, workers=2, projection_block_callback=consume,
                )
        self.assertIs(caught.exception, failure)
        self.assertEqual(delivered, [0])
        self.assertTrue(source.all())

    def test_planfree_compile_failure_precedes_allocation_and_sink(self) -> None:
        shape = (3, 5, 7)
        view = next(view for view in geometry.get_view_infos(
            *shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.7, radial_patch_size=6,
        ) if view.family == 'radial')
        source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
        failure = RuntimeError('compiled pull rejected')
        sink = mock.Mock()
        with (
            mock.patch.object(cp, '_PLANE_PLAN_MAX_BYTES', 1),
            mock.patch.object(cp, '_pull_radial_range_into', side_effect=failure) as kernel,
            mock.patch.object(cp, 'allocate_workspace_array') as allocate,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(RuntimeError) as caught:
                cp.backproject_radial_volume_to_volume(
                    source, view, Path('unused.dat'), 'preflight failure',
                    projection_block_callback=sink,
                )
        self.assertIs(caught.exception, failure)
        kernel.assert_called_once()
        self.assertEqual(kernel.call_args.args[0].size, 0)
        allocate.assert_not_called()
        sink.assert_not_called()


if __name__ == '__main__':
    unittest.main()
