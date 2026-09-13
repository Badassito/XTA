from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import lta_window_gpu_smoke as smoke
from XTA.lta_propagation import LtaMaskSeed
from XTA.lta_tile_tracking import LtaLineageId


class LtaWindowGpuSmokeControlsTests(unittest.TestCase):
    def fixture(self, root):
        cache_path = root / "cache.raw"
        np.zeros((3, 4, 4), dtype=np.uint8).tofile(cache_path)
        cache = smoke.reference_existing_physical_view_cache(
            cache_path, shape=(3, 4, 4), physical_view_id="transverse",
            source_identity="smoke-control-test",
        )
        grid = smoke.LtaTileGridPlan(
            "s4_st3", 4, 3,
            smoke.plan_tile_grid(source_width=4, source_height=4, tile_size=4, tile_stride=3),
        )
        view = SimpleNamespace(
            volume_id="volume", physical_view_id="transverse",
            runtime_view_id="transverse__tta_a0", frame_count=3,
            frame_height=4, frame_width=4, tile_grids=(grid,),
        )
        lineage = LtaLineageId(
            view.volume_id, view.physical_view_id, view.runtime_view_id,
            "object", tile_config_id=grid.config_id,
        )
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        seeds = (LtaMaskSeed(lineage, 1, 0, mask, visited_tile_indices=(0,)),)
        pool = SimpleNamespace(submit=mock.Mock(side_effect=AssertionError("no GPU tasks in control test")))
        return dict(pool=pool, cache=cache, view=view, seeds=seeds,
                    windowed=True, role="control", device=0)

    def test_default_accepts_legacy_driver_without_frontier_keyword(self):
        calls = []

        # Deliberately no **kwargs or canonical_frontier parameter: retained
        # source snapshots must remain callable with the default harness mode.
        def legacy_driver(*, scheduler, pool, initial, view_plan, cache_ref,
                          view_union, relay_mask_revisions, temp_root, conf,
                          empty_frame_limit, worker_task_timeout, trace):
            calls.append(len(initial))
            view_union[1, 1:3, 1:3] = 1
            return 0, {}, (), {"mode": "retained"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(smoke.execution, "_drive_workers_to_fixed_point", legacy_driver):
                summary = smoke.run_case(root / "default", **self.fixture(root))
            self.assertEqual(calls, [1])
            self.assertFalse(summary["canonical_frontier"])
            self.assertEqual(summary["foreground_pixels"], 4)
            self.assertEqual(summary["worker_audit"], {"mode": "retained"})

    def test_opt_in_reaches_driver_and_preserves_union_and_work_audit(self):
        modes = []

        def candidate_driver(*, canonical_frontier=False, **kwargs):
            modes.append(canonical_frontier)
            kwargs["view_union"][1, 1:3, 1:3] = 1
            return 2, {}, ({"work_id": "frontier-task"},), {"canonical_work_count": 1}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(smoke.execution, "_drive_workers_to_fixed_point", candidate_driver):
                summary = smoke.run_case(root / "candidate", canonical_frontier=True, **self.fixture(root))
            self.assertEqual(modes, [True])
            self.assertTrue(summary["canonical_frontier"])
            self.assertEqual(summary["foreground_pixels"], 4)
            self.assertEqual(summary["generation"], 2)
            self.assertEqual(summary["dispatches"], ({"work_id": "frontier-task"},))
            self.assertEqual(summary["worker_audit"], {"canonical_work_count": 1})


if __name__ == "__main__":
    unittest.main()
