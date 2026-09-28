"""Ownership and lifetime of materialized prediction-volume backings."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from XTA.geometry import PredictionVolumeRef, close_prediction_volume_ref
from XTA.runtime import wait_for_retired_memmap_unlinks


def _ref(array: np.memmap, path: Path, *, owned: bool) -> PredictionVolumeRef:
    return PredictionVolumeRef(
        array=array, path=path, name="prediction", view_name="transverse",
        job_id="a0", owns_materialized_path=owned,
    )


class PredictionVolumeRetirementTests(unittest.TestCase):
    def test_owned_scratch_waits_for_last_view_then_unlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prediction.dat"
            mapping = np.memmap(path, dtype=np.uint8, mode="w+", shape=(2, 2))
            mapping[:] = 9
            retained_view = mapping[0]
            ref = _ref(mapping, path, owned=True)

            close_prediction_volume_ref(ref)
            self.assertIsNone(ref.array)
            self.assertTrue(path.exists())
            self.assertEqual(int(retained_view[0]), 9)
            del mapping
            self.assertTrue(path.exists())
            del retained_view
            gc.collect()
            wait_for_retired_memmap_unlinks(path=path)
            self.assertFalse(path.exists())

    def test_unowned_input_and_keep_temp_backing_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for owned, keep_temp in ((False, False), (True, True)):
                with self.subTest(owned=owned, keep_temp=keep_temp):
                    path = root / f"source-{owned}-{keep_temp}.dat"
                    path.write_bytes(b"original")
                    mapping = np.memmap(path, dtype=np.uint8, mode="r", shape=(8,))
                    ref = _ref(mapping, path, owned=owned)
                    close_prediction_volume_ref(ref, keep_temp=keep_temp)
                    self.assertIsNone(ref.array)
                    del mapping
                    gc.collect()
                    self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
