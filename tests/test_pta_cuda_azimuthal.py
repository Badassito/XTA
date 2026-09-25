"""CUDA azimuthal categorical projection against the canonical CPU geometry."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from XTA import geometry
from XTA.config import TiltedViewGroup
from XTA.pta_cuda_azimuthal import (
    _AZIMUTHAL_CATEGORICAL_CUDA,
    _matrix_inverse_from_forward,
    render_azimuthal_categorical_pair,
)


class PTACudaAzimuthalLargeOffsetContractTests(unittest.TestCase):
    def test_opencv_float32_inverse_for_two_thirds_scale(self) -> None:
        forward = np.asarray(
            ((np.float32(1024 / 1536), 0, 0),
             (0, np.float32(1024 / 1536), 0)),
            dtype=np.float32,
        )
        inverse = _matrix_inverse_from_forward(forward, None)
        self.assertEqual(inverse[0, 0], 1.5)
        self.assertEqual(inverse[1, 1], 1.5)
        self.assertNotEqual(
            cv2.invertAffineTransform(forward.astype(np.float64))[0, 0],
            inverse[0, 0],
        )

    def test_production_cube_offsets_exceed_int32_and_use_shared_int64_helper(self) -> None:
        t_len, y_len, x_len = 1931, 2048, 2048
        cases = (
            (1930, 2047, 2047),  # transverse
            (1900, 1024, 2047),  # sagittal plane coordinate near final T
            (1024, 2047, 1945),  # coronal plane coordinate near final X
        )
        offsets = [((t * y_len + y) * x_len + x) for t, y, x in cases]
        self.assertTrue(all(offset > np.iinfo(np.int32).max for offset in offsets))
        self.assertEqual(offsets[0], t_len * y_len * x_len - 1)
        self.assertIn('return ((long long)t * (long long)y_len', _AZIMUTHAL_CATEGORICAL_CUDA)
        self.assertEqual(_AZIMUTHAL_CATEGORICAL_CUDA.count('idx0 = pta_tyx_offset('), 3)
        self.assertEqual(_AZIMUTHAL_CATEGORICAL_CUDA.count('idx1 = pta_tyx_offset('), 3)


class PTACudaAzimuthalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            import cupy
        except ImportError as exc:
            raise unittest.SkipTest(f'CUDA dependencies unavailable: {exc}') from exc
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        cls.torch = torch

    def test_large_offsets_without_large_allocation(self) -> None:
        import cupy as cp

        kernel = cp.RawKernel(
            _AZIMUTHAL_CATEGORICAL_CUDA,
            'pta_azimuthal_offset_probe',
            options=('--fmad=false',),
        )
        result = cp.empty((1,), dtype=cp.int64)
        cases = (
            (1930, 2047, 2047, 2048, 2048),
            (1900, 1024, 2047, 2048, 2048),
            (1024, 2047, 1945, 2048, 2048),
        )
        for t, y, x, y_len, x_len in cases:
            with self.subTest(t=t, y=y, x=x):
                kernel((1,), (1,), (
                    result,
                    np.int32(t), np.int32(y), np.int32(x),
                    np.int32(y_len), np.int32(x_len),
                ))
                expected = (t * y_len + y) * x_len + x
                self.assertGreater(expected, np.iinfo(np.int32).max)
                self.assertEqual(int(result.get()[0]), expected)

    def test_two_thirds_tile_scale_matches_opencv_for_upright_and_tilted(self) -> None:
        rng = np.random.default_rng(20260926)
        scenarios = (
            # A long native U axis tests 1536 -> 1024 output columns.
            ('transverse', (8, 4, 1536), 8, 1536, 6, 1024),
            # A long native stack axis tests 1536 -> 1024 output rows.
            ('sagittal', (8, 1536, 8), 1536, 8, 1024, 6),
        )
        for base, shape, native_h, native_w, out_h, out_w in scenarios:
            foreground = (rng.random(shape) > 0.5).astype(np.uint8)
            coverage = (rng.random(shape) > 0.4).astype(np.uint8)
            gpu_foreground = self.torch.as_tensor(foreground, device='cuda')
            gpu_coverage = self.torch.as_tensor(coverage, device='cuda')
            for tilted in (False, True):
                with self.subTest(base=base, tilted=tilted):
                    view = geometry.ViewInfo(
                        name=f'azimuthal_{"tilted_" if tilted else ""}{base}',
                        num_slices=1, src_h=native_h, src_w=native_w,
                        pad_mode='pad', family=geometry.AZIMUTHAL_VIEW_FAMILY,
                        azimuths_deg=(0.0,), diameter=native_w,
                        center_x=(shape[2] - 1) / 2,
                        center_y=(shape[1] - 1) / 2 if base == 'transverse' else (shape[0] - 1) / 2,
                        roi_radius=(native_w - 1) / 2,
                        full_t=shape[0], full_h=shape[1], full_w=shape[2],
                        tilt_angle_deg=0.1 if tilted else 0.0,
                        tilt_direction='horizontal' if tilted else '',
                        tilt_base_view=base,
                        azimuthal_base_view=base,
                        azimuthal_tilted_source=tilted,
                    )
                    forward = np.asarray(
                        ((np.float32(out_w / native_w), 0, 0),
                         (0, np.float32(out_h / native_h), 0)),
                        dtype=np.float32,
                    )
                    inverse = cv2.invertAffineTransform(forward)
                    expected_foreground = geometry.render_categorical_frame_on_grid(
                        foreground, view, 0,
                        M_src_to_out=forward, M_out_to_src=inverse,
                        output_height=out_h, output_width=out_w,
                    )
                    expected_coverage = geometry.render_categorical_frame_on_grid(
                        coverage, view, 0,
                        M_src_to_out=forward, M_out_to_src=inverse,
                        output_height=out_h, output_width=out_w,
                    )
                    pair = render_azimuthal_categorical_pair(
                        gpu_foreground, gpu_coverage, view, 0,
                        M_src_to_out=forward, M_grid_to_src=inverse,
                        out_h=out_h, out_w=out_w,
                    )
                    self.assertIsNotNone(pair)
                    got_foreground, got_coverage = pair
                    np.testing.assert_array_equal(
                        got_foreground.cpu().numpy(), expected_foreground,
                    )
                    np.testing.assert_array_equal(
                        got_coverage.cpu().numpy(), expected_coverage,
                    )

    def test_upright_and_tilted_bases_with_output_affine_and_coverage(self) -> None:
        rng = np.random.default_rng(412)
        mask = (rng.random((9, 10, 11)) > 0.55).astype(np.uint8) * 9
        coverage = (rng.random((9, 10, 11)) > 0.42).astype(np.uint8) * 255
        gpu_mask = self.torch.as_tensor(mask, device='cuda')
        gpu_coverage = self.torch.as_tensor(coverage, device='cuda')
        views = geometry.get_view_infos(
            9, 10, 11,
            cartesian_views=(),
            azimuthal_views=(
                'transverse', 'sagittal', 'coronal',
                'tilted_transverse', 'tilted_sagittal', 'tilted_coronal',
            ),
            azimuthal_azimuth_angles=(37.0,) * 6,
            tilt_groups=(TiltedViewGroup(
                views=('transverse', 'sagittal', 'coronal'),
                tilt_angles=(24.0,), tilt_directions=('vertical', 'horizontal'),
            ),),
            azimuthal_native_raster=8,
        )
        selected = [
            view for view in views
            if geometry.is_azimuthal_view(view)
            and (not geometry.is_tilted_azimuthal_view(view) or view.tilt_angle_deg > 0)
        ]
        self.assertGreaterEqual(len(selected), 9)
        for view in selected:
            with self.subTest(view=view.name):
                angle_idx = min(1, len(view.azimuths_deg) - 1)
                M = cv2.getRotationMatrix2D(
                    ((view.src_w - 1) / 2, (view.src_h - 1) / 2), 12.5, 0.93,
                ).astype(np.float32)
                M[:, 2] += np.asarray((0.22, -0.17), dtype=np.float32)
                inverse = cv2.invertAffineTransform(M)
                out_h, out_w = int(view.src_h) + 3, int(view.src_w) + 4
                expected_mask = geometry.render_categorical_frame_on_grid(
                    mask, view, angle_idx,
                    M_src_to_out=M, M_out_to_src=inverse,
                    output_height=out_h, output_width=out_w,
                )
                expected_coverage = geometry.render_categorical_frame_on_grid(
                    coverage, view, angle_idx,
                    M_src_to_out=M, M_out_to_src=inverse,
                    output_height=out_h, output_width=out_w,
                )
                pair = render_azimuthal_categorical_pair(
                    gpu_mask, gpu_coverage, view, angle_idx,
                    M_grid_to_src=inverse, M_src_to_out=M,
                    out_h=out_h, out_w=out_w,
                )
                self.assertIsNotNone(pair)
                got_mask, got_coverage = pair
                np.testing.assert_array_equal(got_mask.cpu().numpy(), expected_mask)
                np.testing.assert_array_equal(got_coverage.cpu().numpy(), expected_coverage)

    def test_uncovered_tilted_stack_is_zero_and_optional_coverage_is_none(self) -> None:
        mask = np.ones((9, 10, 11), dtype=np.uint8)
        view = geometry.ViewInfo(
            name='azimuthal_tilted_transverse_vertical_p60',
            num_slices=1, src_h=7, src_w=8, pad_mode='pad',
            family=geometry.AZIMUTHAL_VIEW_FAMILY,
            azimuths_deg=(90.0,), diameter=8,
            center_x=5.0, center_y=4.5, roi_radius=4.5,
            full_t=9, full_h=10, full_w=11,
            tilt_angle_deg=60.0, tilt_direction='vertical',
            tilt_base_view='transverse', azimuthal_base_view='transverse',
            azimuthal_tilted_source=True,
        )
        expected = geometry.get_categorical_view_frame_by_index(mask, view, 0)
        pair = render_azimuthal_categorical_pair(
            self.torch.as_tensor(mask, device='cuda'), None, view, 0,
        )
        self.assertIsNotNone(pair)
        got, coverage = pair
        self.assertIsNone(coverage)
        np.testing.assert_array_equal(got.cpu().numpy(), expected)


if __name__ == '__main__':
    unittest.main()
