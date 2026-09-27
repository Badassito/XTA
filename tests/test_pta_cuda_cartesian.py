from __future__ import annotations

import unittest

import cv2
import numpy as np

from XTA import geometry
from XTA.config import TiltedViewGroup
from XTA.pta_cuda_cartesian import (
    _cv_inverse_for_nearest,
    render_categorical_item,
    render_categorical_pair,
)


def _cuda_modules():
    try:
        import torch
        import cupy
    except ImportError:
        return None
    return torch, cupy


class CartesianNearestAffineCpuTests(unittest.TestCase):
    def test_fp32_inverse_matches_opencv_at_two_thirds_scale(self):
        h, w = 61, 67
        yy, xx = np.indices((h, w), dtype=np.float32)
        encoded_x = (xx + 1).astype(np.uint8)
        encoded_y = (yy + 1).astype(np.uint8)
        forward = cv2.getRotationMatrix2D((33.0, 30.0), 0.0, 2.0 / 3.0).astype(np.float32)
        oracle_x = cv2.warpAffine(encoded_x, forward, (71, 59), flags=cv2.INTER_NEAREST)
        oracle_y = cv2.warpAffine(encoded_y, forward, (71, 59), flags=cv2.INTER_NEAREST)
        inverse = _cv_inverse_for_nearest(forward).astype(np.float64)
        oy, ox = np.indices((59, 71), dtype=np.float64)
        sx = np.rint(inverse[0, 0] * ox + inverse[0, 1] * oy + inverse[0, 2]).astype(np.int32)
        sy = np.rint(inverse[1, 0] * ox + inverse[1, 1] * oy + inverse[1, 2]).astype(np.int32)
        predicted_x = np.zeros_like(oracle_x)
        predicted_y = np.zeros_like(oracle_y)
        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        predicted_x[valid] = encoded_x[sy[valid], sx[valid]]
        predicted_y[valid] = encoded_y[sy[valid], sx[valid]]
        np.testing.assert_array_equal(predicted_x, oracle_x)
        np.testing.assert_array_equal(predicted_y, oracle_y)


class PtaCudaCartesianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modules = _cuda_modules()
        if modules is None or not modules[0].cuda.is_available():
            raise unittest.SkipTest("CUDA Torch and CuPy required")
        cls.torch, _ = modules

    def setUp(self):
        rng = np.random.default_rng(481)
        self.mask = np.asarray(rng.random((7, 8, 9)) > 0.73, dtype=np.uint8)
        self.coverage = np.asarray(rng.random((7, 8, 9)) > 0.18, dtype=np.uint8)
        self.mask_t = self.torch.from_numpy(self.mask).cuda()
        self.coverage_t = self.torch.from_numpy(self.coverage).cuda()

    def _compare(self, view, frame_idx, matrix, h, w, *, max_mismatch=0):
        forward = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
        actual, actual_coverage = render_categorical_pair(
            self.mask_t, self.coverage_t, view, matrix, frame_idx, h, w,
            M_src_to_out=forward,
        )
        expected = geometry.render_categorical_frame_on_grid(
            self.mask, view, frame_idx,
            M_src_to_out=forward, M_out_to_src=matrix,
            output_height=h, output_width=w,
        )
        expected_coverage = geometry.render_categorical_frame_on_grid(
            self.coverage, view, frame_idx,
            M_src_to_out=forward, M_out_to_src=matrix,
            output_height=h, output_width=w,
        )
        got = actual.cpu().numpy()
        got_coverage = actual_coverage.cpu().numpy()
        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(got_coverage.dtype, np.uint8)
        self.assertLessEqual(int(np.count_nonzero(got != expected)), max_mismatch)
        self.assertLessEqual(int(np.count_nonzero(got_coverage != expected_coverage)), max_mismatch)
        single = render_categorical_item(
            self.mask_t, view, matrix, frame_idx, h, w, M_src_to_out=forward,
        )
        np.testing.assert_array_equal(single.cpu().numpy(), got)

    def test_cartesian_axes_identity_and_affine(self):
        views = geometry.get_view_infos(
            7, 8, 9,
            cartesian_views=("transverse", "sagittal", "coronal"),
            azimuthal_views=(), azimuthal_azimuth_angles=(),
        )
        for view in views:
            with self.subTest(view=view.name, transform="identity"):
                self._compare(view, 2, np.asarray([[1, 0, 0], [0, 1, 0]], np.float32),
                              view.src_h, view.src_w)
            with self.subTest(view=view.name, transform="affine"):
                matrix = np.asarray([[0.91, 0.13, -0.37], [-0.07, 1.04, 0.29]], np.float32)
                self._compare(view, 2, matrix, 11, 12)

    def test_cartesian_two_thirds_tie_matrix_matches_opencv(self):
        t, y, x = np.indices((7, 61, 67))
        mask = np.asarray((t + y + x) % 2, dtype=np.uint8)
        view = geometry.get_view_infos(
            7, 61, 67, cartesian_views=("transverse",),
            azimuthal_views=(), azimuthal_azimuth_angles=(),
        )[0]
        forward = cv2.getRotationMatrix2D((33.0, 30.0), 0.0, 2.0 / 3.0).astype(np.float32)
        inverse = cv2.invertAffineTransform(forward).astype(np.float32)
        actual = render_categorical_item(
            self.torch.from_numpy(mask).cuda(), view, inverse, 2, 59, 71,
            M_src_to_out=forward,
        ).cpu().numpy()
        expected = geometry.render_categorical_frame_on_grid(
            mask, view, 2, M_src_to_out=forward, M_out_to_src=inverse,
            output_height=59, output_width=71,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_tilted_all_axes_and_directions(self):
        group = TiltedViewGroup(
            views=("transverse", "sagittal", "coronal"),
            tilt_angles=(19.0,), tilt_directions=("vertical", "horizontal"),
        )
        views = geometry.get_view_infos(
            7, 8, 9,
            cartesian_views=(), azimuthal_views=(), azimuthal_azimuth_angles=(),
            tilt_groups=(group,),
        )
        matrix = np.asarray([[0.97, 0.04, -0.26], [-0.03, 1.06, 0.21]], np.float32)
        for view in views:
            if view.family != "tilted" or float(view.tilt_angle_deg) <= 0:
                continue
            with self.subTest(view=view.name):
                idx = min(2, view.num_slices - 1)
                self._compare(view, idx, matrix, 10, 11)


if __name__ == "__main__":
    unittest.main()
