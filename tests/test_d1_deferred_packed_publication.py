"""Qualify the packed CPU call boundary reached by a deferred D1 actor result."""
from concurrent.futures import wait
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from numba.core.registry import CPUDispatcher

from XTA import cuda_d1, geometry
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore
from XTA.workers import _DeferredGpuWorkerTaskResult


class DeferredPackedOwnerTests(unittest.TestCase):
    def test_owner_future_publishes_exact_crops_without_python_boxed_dispatch(self):
        original_call = CPUDispatcher.__call__

        def guarded_call(dispatcher, *args, **kwargs):
            if dispatcher.py_func.__module__ == "XTA.packed_publication":
                raise AssertionError("Deferred packed publication entered the Python Numba dispatcher")
            return original_call(dispatcher, *args, **kwargs)

        shape = (7, 13, 129)
        view = geometry.get_view_infos(*shape, cartesian_views=("transverse",))[0]
        masks = []
        deferred = []
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"YOLO_TTA_PACKED_OWNER_PUBLICATION": "1"}), \
                mock.patch.object(CPUDispatcher, "__call__", guarded_call):
            for index in range(3):
                volume = np.zeros(shape, np.uint8)
                volume[index, index, [0, 31, 32, 63, 64, 128]] = 1
                volume[-1, -1, 112 + index:129] = 1
                masks.append(volume)
                packed = np.packbits(volume.ravel(), bitorder="little")
                # A non-owning uint32 view mirrors the host D2H tensor's NumPy view.
                words = np.pad(packed, (0, -packed.size % 4)).view(np.uint32)
                state = cuda_d1._D1WorkerViewState(
                    key=(f"model_{index}", view.name), view=view, output_shape=shape, bitset=None,
                    coverage=np.ones(shape[0], np.bool_), store_dir=Path(directory) / f"view_{index}.cvol",
                    created_at=time.monotonic(),
                )
                future = cuda_d1._d1_submit_publication(words=words, state=state)
                deferred.append(_DeferredGpuWorkerTaskResult({"task_id": index}, future, ()))

            _, pending = wait([result.flush_future for result in deferred], timeout=60)
            self.assertFalse(pending, "D1 writer futures did not settle")
            for index, result in enumerate(deferred):
                stats = result.finish()
                ref = stats["d1_layer_ref"]
                self.assertEqual(stats["task_id"], index)
                self.assertEqual(ref.storage_format, INTERNAL_PACKED_CVOL_FORMAT)
                self.assertEqual(stats["d1_cvol_stats"]["foreground_voxels"], int(masks[index].sum()))
                store = RawBBoxMaskStore.open(ref.path, mmap_payload=True)
                try:
                    for z in range(shape[0]):
                        np.testing.assert_array_equal(store.decode_slice(z), masks[index][z])
                finally:
                    store.close()
                self.assertIsNone(result.flush_future)
