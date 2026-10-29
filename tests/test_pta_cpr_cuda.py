"""CPU contracts and explicitly opted-in device parity for resident CPR."""

from __future__ import annotations

import os
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from XTA.pta_cpr import CprGeometry, render_cpr_frame
from XTA import pta_cpr_cuda


def geometry(center=(3.0, 4.0, 5.0), normal=(0.0, 1.0, 0.0),
             binormal=(0.0, 0.0, 1.0), side=8):
    return CprGeometry(*(np.asarray([value], dtype=np.float32) for value in
                         (center, normal, binormal)), (side - 1) // 2, raster_size=side)


class FakeArray:
    def __init__(self, shape=(7, 9, 11), dtype=np.uint8, contiguous=True, device=0, pointer=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = np.dtype(dtype)
        self.flags = types.SimpleNamespace(c_contiguous=contiguous)
        self.device = types.SimpleNamespace(id=device)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        self.data = types.SimpleNamespace(ptr=id(self) * 4096 if pointer is None else pointer)


class TestCprCudaContract(unittest.TestCase):
    def setUp(self):
        self.cp = types.SimpleNamespace(
            ndarray=FakeArray, uint8=np.uint8,
            empty=mock.Mock(side_effect=lambda shape, dtype: FakeArray(shape, dtype)),
            cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDevice=lambda: 0)),
        )
        self.launch = mock.Mock()
        self.patches = (
            mock.patch.dict(sys.modules, cupy=self.cp),
            mock.patch.object(pta_cpr_cuda, "_kernel", return_value=self.launch),
        )
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    def test_launch_preserves_device_result_and_stream_without_copy(self):
        source = FakeArray()
        stream = object()
        result = pta_cpr_cuda.render_cpr_frame_cuda(source, geometry(), 0, stream=stream)
        self.assertIs(result, self.launch.call_args.args[2][1])
        self.assertEqual(result.shape, (8, 8))
        self.assertEqual(result.dtype, np.uint8)
        self.assertIs(self.launch.call_args.args[2][0], source)
        self.assertIs(self.launch.call_args.kwargs["stream"], stream)
        self.assertEqual(self.launch.call_args.args[:2], ((1,), (256,)))
        self.assertEqual(tuple(self.launch.call_args.args[2][2:8]), (7, 7, 9, 11, 8, 0))
        self.assertEqual(float(self.launch.call_args.args[2][8]), 3.5)

    def test_native_t_is_explicit_and_categorical_rejects_virtual_resize(self):
        pta_cpr_cuda.render_cpr_frame_cuda(FakeArray(), geometry(), 0, resize_to_t=13)
        self.assertEqual(tuple(self.launch.call_args.args[2][2:4]), (7, 13))
        with self.assertRaisesRegex(ValueError, "processing-grid"):
            pta_cpr_cuda.render_cpr_frame_cuda(FakeArray(), geometry(), 0,
                                             resize_to_t=13, categorical=True)

    def test_caller_owned_output_is_written_and_returned_without_allocation(self):
        source = FakeArray(pointer=1000)
        output = FakeArray((8, 8), pointer=10000)
        result = pta_cpr_cuda.render_cpr_frame_cuda(source, geometry(), 0, out=output)
        self.assertIs(result, output)
        self.assertIs(self.launch.call_args.args[2][1], output)
        self.cp.empty.assert_not_called()

    def test_invalid_or_overlapping_caller_outputs_are_rejected(self):
        source = FakeArray(pointer=1000)
        for output, error in ((np.empty((8, 8), np.uint8), TypeError),
                              (FakeArray((8, 7)), ValueError),
                              (FakeArray((8, 8), dtype=np.float32), ValueError),
                              (FakeArray((8, 8), contiguous=False), ValueError),
                              (FakeArray((8, 8), device=1), ValueError),
                              (FakeArray((8, 8), pointer=1000), ValueError),
                              (FakeArray((8, 8), pointer=1000 + source.nbytes - 1), ValueError),
                              (FakeArray((8, 8), pointer=1000 - 63), ValueError)):
            with self.subTest(output=output), self.assertRaises(error):
                pta_cpr_cuda.render_cpr_frame_cuda(source, geometry(), 0, out=output)
        self.launch.assert_not_called()
        self.cp.empty.assert_not_called()

    def test_rejects_nonresident_wrong_dtype_layout_device_and_shape(self):
        for source, error in ((np.zeros((7, 9, 11), np.uint8), TypeError),
                              (FakeArray(dtype=np.float32), ValueError),
                              (FakeArray(contiguous=False), ValueError),
                              (FakeArray(device=1), ValueError),
                              (FakeArray(shape=(7, 0, 11)), ValueError),
                              (FakeArray(shape=(7, 11)), ValueError)):
            with self.subTest(source=source), self.assertRaises(error):
                pta_cpr_cuda.render_cpr_frame_cuda(source, geometry(), 0)
        self.launch.assert_not_called()

    def test_rejects_invalid_geometry_before_allocating(self):
        for geo, index, error in ((geometry(), -1, IndexError), (geometry(), 1, IndexError),
                                  (geometry(center=(np.nan, 0, 0)), 0, ValueError)):
            with self.subTest(index=index), self.assertRaises(error):
                pta_cpr_cuda.render_cpr_frame_cuda(FakeArray(), geo, index)
        with self.assertRaisesRegex(ValueError, "positive"):
            pta_cpr_cuda.render_cpr_frame_cuda(FakeArray(), geometry(), 0, resize_to_t=0)
        self.cp.empty.assert_not_called()


@unittest.skipUnless(os.environ.get("XTA_CPR_CUDA_TEST") == "1",
                     "requires explicitly reserved GPU: XTA_CPR_CUDA_TEST=1")
class TestCprCudaDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reservation = Path(os.environ["XTA_GPU_LOCK_PATH"])
        owner = json.loads(reservation.read_text())
        if owner["pid"] != os.getpid() or owner["token"] != os.environ["XTA_GPU_LOCK_TOKEN"]:
            raise RuntimeError("Device CPR tests require this process's GPU reservation")
        import cupy as cp
        cls.cp = cp

    def assert_sampler_equal(self, volume, geo, *, categorical=False, resize_to_t=None):
        cp = self.cp
        logical = volume
        if resize_to_t is not None and resize_to_t != volume.shape[0]:
            # Reproduce the resident virtual gray8 integer-voxel contract. This
            # is deliberately separate from trilinear CPR coordinate sampling.
            mapped = (np.arange(resize_to_t, dtype=np.float64) + .5) * (
                volume.shape[0] / resize_to_t) - .5
            lower = np.clip(np.floor(mapped).astype(np.int64), 0, volume.shape[0]-1)
            upper = np.minimum(lower + 1, volume.shape[0]-1)
            alpha = np.clip(mapped - lower, 0, 1).astype(np.float32)[:, None, None]
            a, b = volume[lower].astype(np.float32), volume[upper].astype(np.float32)
            logical = np.rint(a + alpha * (b-a)).astype(np.uint8)
        expected = render_cpr_frame(logical, geo, 0, categorical=categorical)
        got = cp.asnumpy(pta_cpr_cuda.render_cpr_frame_cuda(
            cp.asarray(volume), geo, 0, categorical=categorical, resize_to_t=resize_to_t))
        np.testing.assert_array_equal(got, expected)

    def test_random_oblique_grayscale_and_categories_byte_exact(self):
        rng = np.random.default_rng(61028)
        for shape in ((9, 13, 17), (1, 9, 11), (8, 1, 7), (6, 8, 1)):
            for side in (7, 8, 17):
                for _ in range(4):
                    center = rng.uniform(-1, np.asarray(shape))
                    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    geo = geometry(center, basis[:, 0], basis[:, 1], side)
                    volume = rng.integers(0, 256, shape, dtype=np.uint8)
                    with self.subTest(shape=shape, side=side):
                        self.assert_sampler_equal(volume, geo)
                        self.assert_sampler_equal(volume, geo, categorical=True)

    def test_boundaries_half_ties_and_axis_aligned_planes(self):
        volume = np.arange(5*7*9, dtype=np.uint16).reshape(5, 7, 9).astype(np.uint8)
        for t in (-1e-7, 0, .5, 1.5, 3.5, 4, 4.000001):
            for side in (5, 6):
                with self.subTest(t=t, side=side):
                    geo = geometry((t, 3, 4), side=side)
                    self.assert_sampler_equal(volume, geo)
                    self.assert_sampler_equal(volume, geo, categorical=True)

    def test_virtual_native_t_quantizes_before_trilinear_sampling(self):
        rng = np.random.default_rng(19031)
        for nt, lt in ((3, 7), (7, 3), (13, 19), (1, 8), (17, 26)):
            volume = rng.integers(0, 256, (nt, 13, 17), dtype=np.uint8)
            for _ in range(6):
                basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                geo = geometry(rng.uniform((0, 0, 0), (lt-1, 12, 16)),
                               basis[:, 0], basis[:, 1], side=12)
                with self.subTest(nt=nt, lt=lt):
                    self.assert_sampler_equal(volume, geo, resize_to_t=lt)

    def test_external_stream(self):
        cp = self.cp
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            result = pta_cpr_cuda.render_cpr_frame_cuda(
                cp.full((7, 9, 11), 201, cp.uint8), geometry(), 0, stream=stream)
        stream.synchronize()
        self.assertEqual(int(result.max()), 201)

    def test_torch_owned_output_keeps_original_allocator_and_pointer(self):
        import torch
        cp = self.cp
        rng = np.random.default_rng(8273)
        source = rng.integers(0, 256, (9, 13, 17), dtype=np.uint8)
        geo = geometry((4.1, 6.2, 8.3), (.2, .8, .1), (.1, -.2, .9), side=12)
        expected = render_cpr_frame(source, geo, 0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), cp.cuda.ExternalStream(stream.cuda_stream) as cp_stream:
            resident = torch.from_numpy(source).to(device='cuda')
            output = torch.empty((12, 12), device='cuda', dtype=torch.uint8)
            view = cp.asarray(output)
            result = pta_cpr_cuda.render_cpr_frame_cuda(
                cp.asarray(resident), geo, 0, stream=cp_stream, out=view)
            self.assertIs(result, view)
            self.assertEqual(int(result.data.ptr), output.data_ptr())
        stream.synchronize()
        np.testing.assert_array_equal(output.cpu().numpy(), expected)


if __name__ == "__main__":
    unittest.main()
