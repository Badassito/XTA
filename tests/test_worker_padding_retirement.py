"""CPU-only failure ownership checks for worker padding backings."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from XTA import workers
from XTA.geometry import ViewInfo
from XTA.runtime import wait_for_retired_memmap_unlinks


class WorkerPaddingRetirementTests(unittest.TestCase):
    @staticmethod
    def _task(root: Path) -> dict[str, object]:
        return {
            "view": ViewInfo("azimuthal", 1, 2, 2, "clamp"),
            "job": object(), "job_id": "a0", "task_id": 1,
            "kind": "fullframe", "out_size": 2, "processing_shape": (1, 2, 2),
            "source_volume_path": str(root / "input.dat"),
            "result_conf_path": str(root / "confidence.dat"),
            "azimuthal_padding_dir": str(root / "padding"),
        }

    @staticmethod
    def _patch_worker_memmap(memmap_fn):
        return (
            mock.patch.object(workers, "AugJob", object),
            mock.patch.object(workers, "_canonical_raster_plan_for_task", return_value=None),
            mock.patch.object(workers, "_canonical_image_sink_for_task", return_value=None),
            mock.patch.object(workers, "view_processing_volume_shape", return_value=(1, 2, 2)),
            mock.patch.object(workers, "azimuthal_batch_padding_count", return_value=1),
            mock.patch.object(workers, "np", SimpleNamespace(memmap=memmap_fn, uint8=np.uint8)),
        )

    def test_second_padding_allocation_failure_defers_first_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            opened = []

            def open_padding(*args, **kwargs):
                if opened:
                    raise OSError("intentional confidence padding failure")
                mapping = np.memmap(*args, **kwargs)
                opened.append(mapping)
                return mapping

            patches = self._patch_worker_memmap(open_padding)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaisesRegex(OSError, "intentional confidence padding failure"):
                    workers.run_prediction_volume_in_worker(
                        None, SimpleNamespace(input_channels=1, batch=1), self._task(root)
                    )

            path = next((root / "padding").glob("*.mask.u8.dat"))
            retained_view = opened[0][0]
            self.assertEqual(int(retained_view[0, 0]), 0)
            self.assertTrue(path.exists())
            opened.clear()
            self.assertTrue(path.exists())
            del retained_view
            gc.collect()
            wait_for_retired_memmap_unlinks(path=path)
            self.assertFalse(path.exists())

    def test_preexisting_padding_file_is_neither_overwritten_nor_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            padding_dir = root / "padding"
            padding_dir.mkdir()
            path = padding_dir / "fullframe-azimuthal-a0-task1.mask.u8.dat"
            path.write_bytes(b"original")
            patches = self._patch_worker_memmap(mock.Mock(side_effect=AssertionError("unexpected memmap")))
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaises(FileExistsError):
                    workers.run_prediction_volume_in_worker(
                        None, SimpleNamespace(input_channels=1, batch=1), self._task(root)
                    )
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
