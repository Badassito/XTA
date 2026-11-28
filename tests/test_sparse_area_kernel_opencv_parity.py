"""The sparse NRRD INTER_AREA kernel sets exactly the pixels of the dense cv2 path.

Low-quality mirrors and layer restores produce the same output from dense blocks
(``_resize_binary_mask_frame_to_output_shape``) or sparse crops
(``_resize_sparse_binary_crop_to_output_region``), so both must apply OpenCV's
INTER_AREA rule, including its 1e-3-px edge-sliver cutoff at near-1:1 ratios.
"""

from __future__ import annotations

import numpy as np
import pytest

from XTA.outputs import (
    _resize_binary_mask_frame_to_output_shape,
    _resize_sparse_binary_crop_to_output_region,
)

GEOMETRIES = [
    (1001, 1001, 1000, 1000),
    (1003, 1003, 1000, 1000),
    (999, 1003, 333, 1000),
    (2003, 1007, 1000, 1000),
    (1344, 1344, 1080, 1300),
    (517, 389, 500, 388),
    # 2/1019-px slivers weigh exactly 1/510: cv2's float32 value ties at 0.5 and rounds to 0.
    (1020, 1020, 1019, 1019),
]


def _planes(in_h, in_w, seed=11):
    rng = np.random.default_rng(seed)
    planes = np.zeros((3, in_h, in_w), dtype=np.uint8)
    planes[0] = rng.random((in_h, in_w)) < 0.3
    planes[1] = rng.random((in_h, in_w)) < 0.6
    planes[2, rng.integers(0, in_h, 60), :] = 1
    planes[2, :, rng.integers(0, in_w, 60)] = 1
    return planes


def _sparse_restore(plane, bbox, out_h, out_w):
    y0, x0, y1, x1 = bbox
    result = np.zeros((out_h, out_w), dtype=np.uint8)
    restored = _resize_sparse_binary_crop_to_output_region(
        np.ascontiguousarray(plane[y0:y1, x0:x1]),
        source_shape=plane.shape, source_bbox=bbox, output_shape=(out_h, out_w),
    )
    if restored is not None:
        oy0, ox0, oy1, ox1, crop = restored
        result[oy0:oy1, ox0:ox1] = crop
    return result


@pytest.mark.parametrize("in_h,in_w,out_h,out_w", GEOMETRIES)
@pytest.mark.parametrize("bbox_kind", ["full", "interior"])
def test_sparse_area_kernel_matches_dense_cv2_restore(in_h, in_w, out_h, out_w, bbox_kind):
    if bbox_kind == "full":
        bbox = (0, 0, in_h, in_w)
    else:
        bbox = (in_h // 5 + 1, in_w // 7 + 3, in_h - in_h // 4 - 2, in_w - in_w // 6 - 1)
    y0, x0, y1, x1 = bbox
    for plane in _planes(in_h, in_w):
        source = np.zeros_like(plane)
        source[y0:y1, x0:x1] = plane[y0:y1, x0:x1]
        expected = _resize_binary_mask_frame_to_output_shape(source, out_h, out_w)
        np.testing.assert_array_equal(_sparse_restore(source, bbox, out_h, out_w), expected)


@pytest.mark.parametrize("value", [1, 2, 255])
def test_binary_area_counts_each_positive_source_pixel_once(value):
    from XTA import nrrd_sparse_kernels as nk

    crop = np.zeros((30, 30), np.uint8)
    crop[15, 15] = value
    expected = _resize_binary_mask_frame_to_output_shape(crop, 1, 1)
    assert not expected.any()
    np.testing.assert_array_equal(_sparse_restore(crop, (0, 0, 30, 30), 1, 1), expected)
    mode, ys, ye, xs, xe = nk.resize_mode_and_maps(30, 30, 1, 1)
    zero = np.zeros(1, np.int64)
    one = np.ones(1, np.int64)
    restored, _ = nk.fill_restored_member_group(
        crop.ravel(), False, zero, zero, zero, one*30, one*30, one.astype(np.uint8),
        zero, one, 1, np.array([0, 1], np.int64), zero, 30, 30, 1, 1, ys, ye, xs, xe, mode,
    )
    np.testing.assert_array_equal(restored, expected)


def test_binary_area_dense_255_does_not_wrap_at_signed_int32():
    from XTA import nrrd_sparse_kernels as nk

    crop = np.full((3000, 3000), 255, np.uint8)
    expected = _resize_binary_mask_frame_to_output_shape(crop, 1500, 1500)
    assert np.all(expected == 1)
    np.testing.assert_array_equal(_sparse_restore(crop, (0, 0, 3000, 3000), 1500, 1500), expected)
    mode, ys, ye, xs, xe = nk.resize_mode_and_maps(3000, 3000, 1500, 1500)
    one = np.ones(1, np.int64)
    zero = np.zeros(1, np.int64)
    restored, _ = nk.fill_restored_member_group(
        crop.ravel(), False, zero, zero, zero, one*3000, one*3000, one.astype(np.uint8),
        zero, one, 1, np.array([0, 1], np.int64), zero, 3000, 3000, 1500, 1500,
        ys, ye, xs, xe, mode,
    )
    np.testing.assert_array_equal(restored, expected)


def test_binary_area_oversized_virtual_crop_keeps_workspace_bounded():
    from XTA.native_cpu import kernels

    crop = np.broadcast_to(np.uint8(255), (65536, 65536))
    out = np.zeros((1, 1), np.uint8)
    kernels.nrrd_area_crop_resize(crop, *crop.shape, 32768, 32768, 0, 0, *crop.shape,
                                  32767, 32767, out)
    np.testing.assert_array_equal(out, [[1]])


def test_legacy_explicit_signed_prefix_abi_still_uses_binary_weights():
    import cv2
    from XTA.native_cpu import kernels

    crop = np.zeros((30, 30), np.uint8)
    crop[5, 5] = 255
    prefix = cv2.integral(crop, sdepth=cv2.CV_32S)
    output = np.ones((1, 1), np.uint8)
    args = (crop, *crop.shape, 1, 1, 0, 0, *crop.shape, 0, 0, output)
    kernels.nrrd_area_crop_resize(prefix, *args)
    assert not output.any()
    with pytest.raises(TypeError):
        kernels.nrrd_area_crop_resize(prefix.view(np.uint32), *args)
