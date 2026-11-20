"""The GPU final-fusion area restore reproduces the cv2 INTER_AREA plane restore.

The fused final ORs CPU-restored (cv2) and GPU-restored groups into one output, so
the torch operator must set exactly the pixels ``_resize_union_plane_to_out_xy`` sets,
including the sub-1e-3-px edge slivers OpenCV drops. These checks run on CPU torch.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from XTA.finalization import (
    _opencv_area_axis_taps,
    _resize_union_plane_to_out_xy,
    _torch_area_restore_binary,
)

torch = pytest.importorskip("torch")

GEOMETRIES = [
    (1024, 1024, 1000, 1000),
    (1001, 1001, 1000, 1000),
    (1003, 1003, 1000, 1000),
    (999, 1003, 333, 1000),
    (2003, 1007, 1000, 1000),
    (1344, 1344, 1080, 1300),
    (700, 700, 512, 512),
    (640, 640, 600, 512),
    (1000, 1000, 500, 250),
    (517, 389, 500, 388),
    (1020, 1020, 1019, 1019),  # exact float32 ties at 0.5, which cv2 rounds to 0
]


def _planes(in_h, in_w, seed=11):
    rng = np.random.default_rng(seed)
    planes = np.zeros((4, in_h, in_w), dtype=np.uint8)
    planes[0] = rng.random((in_h, in_w)) < 0.003
    planes[1] = rng.random((in_h, in_w)) < 0.3
    planes[2] = rng.random((in_h, in_w)) < 0.6
    planes[3, rng.integers(0, in_h, 60), :] = 1
    planes[3, :, rng.integers(0, in_w, 60)] = 1
    return planes


def _device_taps(n_in, n_out, device="cpu", taps=None):
    index, weight = _opencv_area_axis_taps(n_in, n_out) if taps is None else taps
    return (torch.from_numpy(index.copy()).to(device), torch.from_numpy(weight.copy()).to(device))


def _cv2_reference(planes, out_h, out_w):
    return np.stack([_resize_union_plane_to_out_xy(plane, out_h, out_w) for plane in planes])


@pytest.mark.parametrize("in_h,in_w,out_h,out_w", GEOMETRIES)
def test_torch_area_restore_matches_cv2_plane_restore(in_h, in_w, out_h, out_w):
    planes = _planes(in_h, in_w)
    expected = _cv2_reference(planes, out_h, out_w)
    actual = _torch_area_restore_binary(
        torch, torch.from_numpy(planes), _device_taps(in_h, out_h), _device_taps(in_w, out_w),
    )
    assert actual.dtype == torch.uint8
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_former_equal_weight_area_pool_marks_extra_sliver_pixels():
    planes = _planes(1001, 1001)
    expected = _cv2_reference(planes, 1000, 1000)
    pooled = torch.nn.functional.interpolate(
        torch.from_numpy(planes).to(torch.float32).unsqueeze(1), size=(1000, 1000), mode="area",
    )
    former = (pooled.squeeze(1) >= (1.0 / 510.0)).to(torch.uint8).numpy()
    assert np.count_nonzero(former != expected) > 0
    assert np.all(former >= expected)


def _fractional_taps_without_sliver_cutoff(n_in, n_out):
    scale = n_in / n_out
    rows = []
    for dx in range(n_out):
        lo, hi = dx * scale, min(float(n_in), (dx + 1) * scale)
        rows.append([(sx, np.float32((min(hi, sx + 1) - max(lo, sx)) / scale))
                     for sx in range(int(math.floor(lo)), min(n_in, int(math.ceil(hi))))])
    taps = max(len(row) for row in rows)
    index = np.zeros((taps, n_out), dtype=np.int64)
    weight = np.zeros((taps, n_out), dtype=np.float32)
    for dx, row in enumerate(rows):
        for tap, (sx, alpha) in enumerate(row):
            index[tap, dx], weight[tap, dx] = sx, alpha
    return index, weight


def test_fractional_taps_need_opencv_sliver_cutoff():
    planes = _planes(1003, 1003)
    expected = _cv2_reference(planes, 1000, 1000)
    plain = _device_taps(1003, 1000, taps=_fractional_taps_without_sliver_cutoff(1003, 1000))
    without_cutoff = _torch_area_restore_binary(torch, torch.from_numpy(planes), plain, plain).numpy()
    assert np.count_nonzero(without_cutoff != expected) > 0


def test_axis_taps_drop_subthreshold_slivers_and_stay_normalized():
    index, weight = _opencv_area_axis_taps(1001, 1000)
    assert not index.flags.writeable and not weight.flags.writeable
    assert index.dtype == np.int64 and weight.dtype == np.float32
    assert index.shape == weight.shape and index.shape[1] == 1000
    assert index.shape[0] <= math.ceil(1001 / 1000) + 2
    # Output 0 covers [0, 1.001): the 0.001-px sliver of source pixel 1 is dropped.
    assert np.count_nonzero(weight[:, 0]) == 1
    assert index[0, 0] == 0 and weight[0, 0] == np.float32(1.0 / 1.001)
    sums = weight.astype(np.float64).sum(axis=0)
    assert np.all(sums <= 1.0 + 1e-6)
    assert np.all(sums >= 1.0 - 1.001e-3)
    assert np.all(index[weight == 0] == 0)


@pytest.mark.skipif(os.environ.get("XTA_TEST_FINALIZATION_AREA_GPU") != "1",
                    reason="explicit real CUDA qualification with an owned GPU lock only")
@pytest.mark.parametrize("in_h,in_w,out_h,out_w", GEOMETRIES)
def test_real_cuda_area_restore_matches_cv2_plane_restore(in_h, in_w, out_h, out_w):
    assert torch.cuda.is_available()
    planes = _planes(in_h, in_w)
    expected = _cv2_reference(planes, out_h, out_w)
    actual = _torch_area_restore_binary(
        torch, torch.from_numpy(planes).to("cuda"),
        _device_taps(in_h, out_h, "cuda"), _device_taps(in_w, out_w, "cuda"),
    )
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
