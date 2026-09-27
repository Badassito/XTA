"""CPU-side contracts for PTA's resident categorical CUDA coordinator."""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np

from XTA import pta_cuda_masks


class _FakeCuda:
    def __init__(self, free_bytes: int):
        self.free_bytes = free_bytes
        self.probes = 0

    def mem_get_info(self, _device_id):
        self.probes += 1
        return self.free_bytes, self.free_bytes

    def Stream(self, *, device):
        return _FakeStream()

    def device(self, _device):
        return nullcontext()

    def stream(self, _stream):
        return nullcontext()


class _FakeDeviceTensor:
    def __init__(self, shape):
        self.shape = shape
        self.copies = 0

    def __getitem__(self, _slice):
        return self

    def copy_(self, _source, *, non_blocking):
        self.copies += 1


class _FakeTorch:
    uint8 = np.uint8

    def __init__(self, free_bytes: int):
        self.cuda = _FakeCuda(free_bytes)
        self.allocations = []

    def empty(self, shape, *, dtype, device):
        tensor = _FakeDeviceTensor(shape)
        self.allocations.append(tensor)
        return tensor

    def from_numpy(self, array):
        return array


class _FakeStream:
    def __init__(self):
        self.fences = 0

    def synchronize(self):
        self.fences += 1


class CategoricalCoordinatorTests(unittest.TestCase):
    def test_full_and_tile_keep_both_canonical_matrices(self):
        full_forward = np.array([[2, 0, 3], [0, 2, 5]], dtype=np.float32)
        full_inverse = np.array([[0.5, 0, -1.5], [0, 0.5, -2.5]], dtype=np.float32)
        tile_forward = np.array([[1, 0, -7], [0, 1, -9]], dtype=np.float32)
        tile_inverse = np.array([[1, 0, 7], [0, 1, 9]], dtype=np.float32)
        plan = SimpleNamespace(
            aff=SimpleNamespace(
                M_src_to_out=full_forward,
                M_out_to_src=full_inverse,
                out_h=2048,
                out_w=2048,
            ),
            tile_layout=(SimpleNamespace(
                tile_tag="tile_1", out_h=1024, out_w=1024,
                shared_job=SimpleNamespace(
                    M_src_to_out=tile_forward,
                    M_out_to_src=tile_inverse,
                ),
            ),),
        )
        full = pta_cuda_masks._item_grid(plan, "full")
        tile = pta_cuda_masks._item_grid(plan, "tile_1")
        np.testing.assert_array_equal(full.forward, full_forward)
        np.testing.assert_array_equal(full.inverse, full_inverse)
        np.testing.assert_array_equal(tile.forward, tile_forward)
        np.testing.assert_array_equal(tile.inverse, tile_inverse)
        self.assertEqual((tile.height, tile.width), (1024, 1024))
        self.assertIsNone(pta_cuda_masks._item_grid(plan, "missing"))

    def test_memory_rejection_happens_before_device_allocation(self):
        mask = np.zeros((4, 16, 16), dtype=np.uint8)
        coverage = np.ones_like(mask)
        fake_torch = _FakeTorch(1)
        stream = _FakeStream()
        owner = pta_cuda_masks.CategoricalVolumeOwner(fake_torch, 0, stream)
        with mock.patch.dict("os.environ", {"PTA_GPU_CATEGORICAL_RESERVE_MIB": "0"}):
            self.assertFalse(owner.ensure(
                mask, coverage, identity="generation-1", largest_output_bytes=256,
            ))
            self.assertEqual(owner.rejected_key[0], "generation-1")
            self.assertEqual(owner.key, None)
            # The same shared-memory generation is not repeatedly probed.
            self.assertFalse(owner.ensure(
                mask, coverage, identity="generation-1", largest_output_bytes=256,
            ))
            self.assertEqual(fake_torch.cuda.probes, 1)
            # A new generation, even at an identical address, is probed anew.
            self.assertFalse(owner.ensure(
                mask, coverage, identity="generation-2", largest_output_bytes=256,
            ))
            self.assertEqual(fake_torch.cuda.probes, 2)

    def test_retire_fences_before_dropping_resident_sources(self):
        stream = _FakeStream()
        owner = pta_cuda_masks.CategoricalVolumeOwner(_FakeTorch(1), 0, stream)
        owner.mask_gpu = object()
        owner.coverage_gpu = object()
        owner.key = ("generation-1",)
        owner.retire()
        self.assertEqual(stream.fences, 1)
        self.assertIsNone(owner.mask_gpu)
        self.assertIsNone(owner.coverage_gpu)
        self.assertIsNone(owner.key)
        owner.retire()
        self.assertEqual(stream.fences, 1)

    def test_one_owner_and_finalizer_under_concurrent_first_use(self):
        runtime = {"torch": _FakeTorch(1), "device_id": 0}
        with mock.patch("multiprocessing.util.Finalize") as finalizer:
            with ThreadPoolExecutor(max_workers=16) as pool:
                owners = list(pool.map(
                    lambda _: pta_cuda_masks._owner_for_runtime(runtime),
                    range(32),
                ))
        self.assertTrue(all(owner is owners[0] for owner in owners))
        self.assertIs(runtime["categorical_volume_owner"], owners[0])
        finalizer.assert_called_once()
        self.assertEqual(finalizer.call_args.kwargs["exitpriority"], 14)

    def test_foreground_uploaded_once_when_coverage_aliases_it(self):
        source = np.ones((2, 4, 4), dtype=np.uint8)
        torch = _FakeTorch(100 * 1024 ** 3)
        owner = pta_cuda_masks.CategoricalVolumeOwner(torch, 0, _FakeStream())
        self.assertTrue(owner.ensure(
            source, source, identity="same-object", largest_output_bytes=16,
        ))
        self.assertIs(owner.mask_gpu, owner.coverage_gpu)
        self.assertEqual(torch.allocations[-1].copies, 1)
        owner.retire()

        alias = source.view()
        self.assertIsNot(alias, source)
        self.assertEqual(alias.__array_interface__["data"][0],
                         source.__array_interface__["data"][0])
        self.assertTrue(owner.ensure(
            source, alias, identity="same-pointer", largest_output_bytes=16,
        ))
        self.assertIs(owner.mask_gpu, owner.coverage_gpu)
        self.assertEqual(torch.allocations[-1].copies, 1)


if __name__ == "__main__":
    unittest.main()
