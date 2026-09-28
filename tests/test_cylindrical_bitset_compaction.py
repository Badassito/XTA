"""CPU-only admission and failure-lifetime checks for Radial bitset compaction."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np

from XTA.cylindrical_bitset_compaction import (
    RadialBitsetCompactor,
    RadialBitsetCompactionUnavailable,
    _owned_readonly_payload,
    plan_bitset_compaction,
)
from XTA.cylindrical_cuda_projection import RadialCudaProjectionUnsafeFailure


class _FakeGpuArray(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


class _FakeStream:
    fail_fence = False
    failures_remaining = 0
    sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1
        if self.fail_fence or self.failures_remaining:
            self.failures_remaining = max(0, self.failures_remaining - 1)
            raise RuntimeError('lost stream fence')


class _FakeCp:
    ndarray = _FakeGpuArray
    uint32 = np.uint32
    uint8 = np.uint8
    uint64 = np.uint64

    def __init__(self, *, free=2**30, fail_alloc_at=None, fail_kernel=False):
        self.free = free
        self.fail_alloc_at = fail_alloc_at
        self.allocations = 0
        self.stream = _FakeStream()
        self.fail_kernel = fail_kernel
        self.cuda = SimpleNamespace(
            Device=lambda _index: nullcontext(),
            Stream=lambda **_kwargs: self.stream,
            runtime=SimpleNamespace(memGetInfo=lambda: (self.free, 2**30)),
        )

    def empty(self, size, dtype):
        self.allocations += 1
        if self.allocations == self.fail_alloc_at:
            raise MemoryError('test allocation failure')
        return np.empty(size, dtype=dtype).view(_FakeGpuArray)

    def RawModule(self, **_kwargs):
        source = self

        class Module:
            def get_function(self, _name):
                def launch(*_args, **_kwargs):
                    if source.fail_kernel:
                        raise RuntimeError('test kernel launch failure')
                return launch
        return Module()


def _words(shape):
    return np.zeros((int(np.prod(shape)) + 31) // 32, np.uint32).view(_FakeGpuArray)


class BitsetCompactionUnitTests(unittest.TestCase):
    def test_payload_view_detaches_but_owned_array_is_not_copied(self):
        source = np.arange(12, dtype=np.uint8)
        view = source[2:8]
        detached = _owned_readonly_payload(view)
        self.assertTrue(detached.flags.owndata)
        self.assertFalse(detached.flags.writeable)
        np.testing.assert_array_equal(detached, view)
        source[2] = 99
        self.assertEqual(int(detached[0]), 2)

        owned = np.arange(4, dtype=np.uint8)
        same = _owned_readonly_payload(owned)
        self.assertIs(same, owned)
        self.assertTrue(same.flags.owndata)
        self.assertFalse(same.flags.writeable)

    def test_real_shape_has_bounded_exact_workspace(self):
        plan = plan_bitset_compaction((1931, 3064, 3022))
        self.assertEqual(plan.block_slices, 7)
        self.assertEqual(1931 % plan.block_slices, 6)
        self.assertEqual(plan.dense_bytes, 64_815_856)
        self.assertEqual(plan.payload_bytes, 8_107_344)
        self.assertEqual(plan.workspace_bytes, 72_923_432)
        self.assertLess(plan.dense_bytes, 64 * 1024**2)
        self.assertLess(plan.payload_bytes, 16 * 1024**2)
        self.assertEqual(plan_bitset_compaction((1931, 3064, 3022),
                                               max_slices=6).block_slices, 6)

    def test_small_view_fills_one_bounded_block(self):
        plan = plan_bitset_compaction((128, 512, 512))
        self.assertEqual(plan.block_slices, 128)
        self.assertEqual(plan.dense_bytes, 32 * 1024**2)
        self.assertEqual(plan.payload_bytes, 4 * 1024**2)

    def test_valid_tall_volume_can_fall_back_before_cuda_setup(self):
        with self.assertRaisesRegex(RadialBitsetCompactionUnavailable, 'grid-y'):
            plan_bitset_compaction((1, 65535 * 8 + 1, 1))

    def test_rejects_invalid_source_before_allocation(self):
        cp = _FakeCp()
        with self.assertRaises(ValueError):
            RadialBitsetCompactor(_words((2, 3, 37))[:-1], (2, 3, 37), 0,
                                  cp_module=cp)
        self.assertEqual(cp.allocations, 0)
        with self.assertRaises(ValueError):
            RadialBitsetCompactor(_words((2, 3, 37)), (2, 3, 37), 1,
                                  cp_module=cp)
        self.assertEqual(cp.allocations, 0)

    def test_low_capacity_and_partial_allocation_are_safe_unavailable(self):
        shape = (2, 3, 37)
        cp = _FakeCp(free=8)
        with self.assertRaises(RadialBitsetCompactionUnavailable):
            RadialBitsetCompactor(_words(shape), shape, 0, cp_module=cp)
        self.assertEqual(cp.allocations, 0)
        cp = _FakeCp(fail_alloc_at=3)
        with self.assertRaisesRegex(RadialBitsetCompactionUnavailable,
                                    'setup failed'):
            RadialBitsetCompactor(_words(shape), shape, 0, cp_module=cp)
        self.assertEqual(cp.allocations, 3)

    def test_successful_close_preserves_borrowed_source(self):
        shape = (2, 3, 37)
        source = _words(shape)
        cp = _FakeCp()
        encoder = RadialBitsetCompactor(source, shape, 0, cp_module=cp)
        self.assertIs(encoder.words, source)
        encoder.close()
        self.assertIsNone(encoder.words)
        self.assertIsNone(encoder.dense)
        self.assertEqual(source.size, 7)
        encoder.close()

    def test_failed_fence_retains_all_gpu_buffers_and_owner(self):
        shape = (2, 3, 37)
        source = _words(shape)
        cp = _FakeCp()
        encoder = RadialBitsetCompactor(source, shape, 0, cp_module=cp)
        cp.stream.fail_fence = True
        with self.assertRaises(RadialCudaProjectionUnsafeFailure) as caught:
            encoder.close()
        self.assertIs(caught.exception.projector, encoder)
        self.assertIs(encoder.words, source)
        self.assertIsNotNone(encoder.dense)
        self.assertFalse(encoder._closed)
        cp.stream.fail_fence = False
        with self.assertRaises(RadialCudaProjectionUnsafeFailure):
            encoder.close()
        self.assertEqual(cp.stream.sync_calls, 1)
        self.assertIs(encoder.words, source)

    def test_context_exit_does_not_retry_transient_failed_fence(self):
        shape = (2, 3, 37)
        source = _words(shape)
        cp = _FakeCp()
        cp.stream.failures_remaining = 1
        encoder = RadialBitsetCompactor(source, shape, 0, cp_module=cp)
        with self.assertRaises(RadialCudaProjectionUnsafeFailure):
            with encoder:
                encoder._fence()
        self.assertEqual(cp.stream.sync_calls, 1)
        self.assertIs(encoder.words, source)
        self.assertIsNotNone(encoder.dense)

    def test_error_after_kernel_launch_never_becomes_setup_unavailable(self):
        shape = (2, 3, 37)
        cp = _FakeCp(fail_kernel=True)
        with self.assertRaisesRegex(RuntimeError, 'kernel launch failure'):
            with RadialBitsetCompactor(_words(shape), shape, 0, cp_module=cp) as encoder:
                encoder.encode_block(0, 1)


if __name__ == '__main__':
    unittest.main()
