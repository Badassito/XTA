from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA.lta_dynamic_crops import (DynamicCropSettings, DynamicObject, NativeMask,
                                  plan_dynamic_crops, split_dynamic_object,
                                  touches_interior_guard, resize_crop_mask)
from XTA.lta_dynamic_execution import (DynamicViewCompletion, DynamicTask, drive_dynamic_workers,
                                      followup_dynamic_tasks, plan_initial_dynamic_tasks)
from XTA.lta_propagation import LtaMaskSeed, read_seed_artifact
from XTA.lta_rendering import reference_existing_physical_view_cache
from XTA.lta_scheduler import LtaSessionWork, LtaViewKey
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import TilePlan
from XTA.lta_windows import WindowPlan


class ReorderedMaskPool:
    worker_slots = ((0, 0), (1, 0))
    pids = {0: 1000, 1: 1001}
    ready_events = ()

    def __init__(self, reverse=False):
        self.pending = []
        self.reverse = reverse

    def submit(self, task, *, execution_device_id, worker_index=0):
        self.pending.append((task, execution_device_id, worker_index))

    def check_liveness(self):
        pass

    def wait_result(self, timeout=None):
        from XTA.lta_coverage import LtaCoverageBuilder
        from XTA.lta_relay_episodes import write_relay_observations
        from XTA.lta_union_artifacts import LtaUnionWriter
        from XTA.lta_workers import LtaWorkerResult
        task, device, worker = self.pending.pop(-1 if self.reverse else 0)
        payload = task.payload
        output = Path(payload["output_dir"])
        output.mkdir(parents=True)
        seeds = read_seed_artifact(payload["seed_artifact_path"], expected_sha256=payload["seed_artifact_sha256"])
        tile = TilePlan(**payload["tile"])
        window = WindowPlan(**payload["window"])
        start, stop = payload["output_frame_start"], payload["output_frame_stop"]
        union = np.zeros((stop - start, tile.size, tile.size), dtype=np.uint8)
        coverage = LtaCoverageBuilder((tile.size, tile.size))
        coverage.mark_observed(tuple(seed.lineage for seed in seeds), frame_start=window.frame_start,
                                frame_stop=window.frame_stop, prompt_frame=window.prompt_frame,
                                direction=window.direction)
        scores = {}
        for seed in seeds:
            native = NativeMask.from_crop(seed.mask, left=tile.left, top=tile.top)
            x0, y0, x1, y1 = native.xyxy
            for frame in range(window.frame_start, window.frame_stop):
                mask = np.zeros((tile.size, tile.size), dtype=bool)
                dx = (frame - window.prompt_frame) * 35
                left, right = max(tile.left, x0 + dx), min(tile.left + tile.size, x1 + dx)
                if left < right:
                    mask[y0 - tile.top:y1 - tile.top, left - tile.left:right - tile.left] = True
                coverage.add_prediction(seed.lineage, frame, mask)
                scores[f"{seed.lineage.token}|{frame}"] = 0.9
                if start <= frame < stop:
                    union[frame - start] |= mask.astype(np.uint8)
        with LtaUnionWriter(output / "union.pack", shape=union.shape, frame_start=start) as writer:
            writer.append_chunk(start, union)
            union_receipt = writer.finish()
        coverage_receipt = coverage.write(output, work_id=task.work_id,
                                           tile_index=payload["tile_index"], tile_config_id="dynamic")
        manifest = {"status": "complete", "work_id": task.work_id, "tile": payload["tile"],
                    "tile_index": payload["tile_index"], "window": payload["window"],
                    "output_frame_range": [start, stop], "union": union_receipt,
                    "lineage_coverage": coverage_receipt,
                    "relay_observation_artifact": write_relay_observations(output, {}),
                    "dogfood_seed_artifacts": [], "relays": [], "task_granularity": "window",
                    "crop_observation_scores": scores,
                    "profile": {"name": "fixture"}, "sam_runtime": {"version": "controlled"},
                    "crop_transform": {"native_crop_xyxy": list(tile.xyxy), "model_shape_hw": [1008, 1008],
                                       "native_mask_shape_hw": [tile.size, tile.size],
                                       "prediction_and_dogfood_coordinates": "native_crop"},
                    "constrained_batches": None, "relay_gate": {}, "foreground_pixels": int(union.sum()),
                    "windows": [{"status": "complete", "prediction_count": coverage.stats()["mask_count"],
                                 "retained_prediction_count": 0, "adapter": {}}]}
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return LtaWorkerResult(task.work_id, task.attempt_token, task.kind, device, self.pids[device],
                               str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                               path.stat().st_size, worker_index=worker)


def obj(name="a", *, frame=14, top=700, left=700, mask=None, depth=0):
    native = NativeMask(top, left, np.ones((20, 20), dtype=bool) if mask is None else mask)
    lineage = LtaLineageId("fixture", "transverse", "transverse", name, "dynamic")
    return DynamicObject(LtaMaskSeed(lineage, frame, 0, native.mask), native, depth)


class DynamicCropTests(unittest.TestCase):
    def test_sealed_clustering_is_independent_of_input_order(self):
        objects = (obj("a", left=690), obj("b", left=810), obj("c", left=2200))
        a = plan_dynamic_crops(objects, height=3024, width=3064)
        b = plan_dynamic_crops(tuple(reversed(objects)), height=3024, width=3064)
        self.assertEqual([(v.tile.xyxy, tuple(o.seed.lineage.token for o in v.objects)) for v in a],
                         [(v.tile.xyxy, tuple(o.seed.lineage.token for o in v.objects)) for v in b])
        self.assertEqual(sorted(map(lambda v: len(v.objects), a)), [1, 2])

    def test_odd_edge_crop_snap_never_clips_seed(self):
        item = obj(top=15, left=37, mask=np.ones((999, 1007), dtype=bool))
        crop = plan_dynamic_crops((item,), height=1055, width=1057,
                                  settings=DynamicCropSettings(margin=0, max_scale=1))[0].tile
        self.assertEqual(int(item.native.to_crop(crop).sum()), 999 * 1007)
        self.assertLessEqual(crop.left, 37)
        self.assertGreaterEqual(crop.left + crop.size, 1044)

    def test_large_crop_bounded_and_too_large_seed_rejected(self):
        item = obj(top=200, left=300, mask=np.ones((1501, 2011), dtype=bool))
        crop = plan_dynamic_crops((item,), height=3024, width=3064)[0].tile
        self.assertGreater(crop.size, 2011)
        self.assertLessEqual(crop.size, 3024)
        self.assertEqual(item.native.to_crop(crop).sum(), item.native.mask.sum())
        with self.assertRaisesRegex(ValueError, "complete dynamic seed"):
            plan_dynamic_crops((item,), height=3024, width=3064,
                               settings=DynamicCropSettings(max_scale=1))

    def test_physical_edge_is_not_crop_escape(self):
        crop = TilePlan(0, 0, 1008, 1764, 1764)
        self.assertFalse(touches_interior_guard(NativeMask(0, 0, np.ones((10, 10))), crop, guard=24))
        self.assertTrue(touches_interior_guard(NativeMask(100, 995, np.ones((10, 10))), crop, guard=24))

    def test_split_preserves_small_island_and_deterministic_child_identity(self):
        mask = np.zeros((60, 100), dtype=bool)
        mask[1:20, 1:20] = True
        mask[30:49, 50:69] = True
        mask[55:57, 90:93] = True
        item = obj(mask=mask)
        pieces = split_dynamic_object(item, settings=DynamicCropSettings())
        self.assertEqual(len(pieces), 2)
        rebuilt = np.zeros_like(mask)
        for piece in pieces:
            y, x = piece.native.top - item.native.top, piece.native.left - item.native.left
            rebuilt[y:y + piece.native.mask.shape[0], x:x + piece.native.mask.shape[1]] |= piece.native.mask
        np.testing.assert_array_equal(rebuilt, mask)
        self.assertEqual([p.seed.lineage for p in pieces],
                         [p.seed.lineage for p in split_dynamic_object(item, settings=DynamicCropSettings())])
        self.assertEqual(split_dynamic_object(replace(item, split_depth=3),
                                              settings=DynamicCropSettings()), (replace(item, split_depth=3),))

    def test_mask_resize_restores_native_geometry_and_empty_shape_guard(self):
        original = np.zeros((1764, 1764), dtype=bool)
        original[350:850, 500:950] = True
        model = resize_crop_mask(original)
        restored = resize_crop_mask(model, side=1764, restore=True)
        self.assertEqual(model.shape, (1008, 1008))
        self.assertEqual(restored.shape, original.shape)
        intersection = int(np.count_nonzero(original & restored))
        union = int(np.count_nonzero(original | restored))
        self.assertGreater(intersection / union, 0.99)

    def test_dynamic_config_is_opt_in_and_rejects_invalid_geometry_settings(self):
        from XTA.lta_config import parse_lta_args
        base = ["--input", "in", "--output", "out", "--model", "model", "--device", "0",
                "--enable_cartesian", "transverse"]
        self.assertEqual(parse_lta_args(base).args.lta_crop_backend, "tiled")
        self.assertEqual(parse_lta_args(base + ["--crop_backend", "dynamic"]).args.lta_crop_backend, "dynamic")
        for flag, value in (("--lta_crop_margin", "-1"), ("--lta_crop_guard", "504"),
                            ("--lta_crop_max_scale", "3.1")):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_lta_args(base + [flag, value])

    def test_backprojection_admission_requires_complete_dynamic_queue(self):
        view = LtaViewKey("fixture", "transverse")
        owner = DynamicViewCompletion(view, (1, 0))
        self.assertIsNone(owner.claim_backprojection(0))
        owner.settled = True
        claim = owner.claim_backprojection(0)
        self.assertIsNotNone(claim)
        self.assertIsNone(owner.claim_backprojection(0))
        owner.complete_backprojection(claim)

    def test_simultaneous_split_and_escape_combines_flags_and_patches_do_not_recurse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "view.raw"
            with path.open("wb") as handle:
                handle.truncate(60 * 1764 * 1764)
            cache = reference_existing_physical_view_cache(path, shape=(60, 1764, 1764),
                        physical_view_id="transverse", source_identity="fixture")
            view = SimpleNamespace(volume_id="fixture", physical_view_id="transverse", runtime_view_id="transverse",
                                   frame_count=60, frame_height=1764, frame_width=1764)
            parent = obj(frame=14, top=500, left=700)
            crop = TilePlan(200, 0, 1008, 1764, 1764)
            window = WindowPlan("center", 0, 0, 30, 14, "both", "authoritative")
            work = LtaSessionWork("parent", LtaViewKey("fixture", "transverse"), "transverse", 0, 0, 30, 0,
                                  estimated_cost=30, projection_key=cache.identity_sha256)
            task = DynamicTask(work, {}, (parent,), crop, window, False, 0, "root")
            split_mask = np.zeros((50, 35), dtype=bool)
            split_mask[:15, :15] = True
            split_mask[30:45, 20:35] = True
            event = obj(frame=19, top=400, left=1170, mask=split_mask)
            boundary = obj(frame=29, top=400, left=1180)
            masks = {(parent.seed.lineage, 19): event, (parent.seed.lineage, 29): boundary}
            audit = {"events": [], "patch_budget_events": [], "empty_boundary_count": 0,
                     "boundary_split_count": 0}
            children = followup_dynamic_tasks(task, masks, view_plan=view, cache_ref=cache, temp_root=root,
                        conf=0.15, empty_frame_limit=30, settings=DynamicCropSettings(), audit=audit)
            self.assertTrue(children)
            self.assertTrue(all(child.patch for child in children))
            self.assertEqual(audit["events"][0]["action"], "split_and_expand")
            self.assertEqual((children[0].window.frame_start, children[0].window.frame_stop), (19, 30))
            patch = children[0]
            patched_parent = patch.objects[0]
            next_event = DynamicObject(replace(patched_parent.seed, frame_index=23),
                                        NativeMask(patch.crop.top + 100, patch.crop.left + 1,
                                                   np.ones((12, 12))), patched_parent.split_depth)
            next_boundary = DynamicObject(replace(patched_parent.seed, frame_index=29),
                                            NativeMask(500, 1000, np.ones((20, 20))), patched_parent.split_depth)
            next_masks = {(patched_parent.seed.lineage, 23): next_event,
                          (patched_parent.seed.lineage, 29): next_boundary}
            followups = followup_dynamic_tasks(patch, next_masks, view_plan=view, cache_ref=cache,
                         temp_root=root, conf=0.15, empty_frame_limit=30,
                         settings=DynamicCropSettings(), audit=audit)
            self.assertTrue(all(not child.patch for child in followups))
            self.assertTrue(audit["patch_budget_events"])

    def test_two_worker_completion_reordering_preserves_escape_jobs_and_native_union(self):
        outputs = []
        for reverse in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / "view.raw"
                with path.open("wb") as handle:
                    handle.truncate(35 * 1200 * 3024)
                cache = reference_existing_physical_view_cache(path, shape=(35, 1200, 3024),
                            physical_view_id="transverse", source_identity="fixture")
                view = SimpleNamespace(volume_id="fixture", physical_view_id="transverse", runtime_view_id="transverse",
                                       frame_count=35, frame_height=1200, frame_width=3024)
                polygons = tuple(SimpleNamespace(row_index=index, points=((left / 3024, 500 / 1200),
                            ((left + 20) / 3024, 500 / 1200), ((left + 20) / 3024, 520 / 1200),
                            (left / 3024, 520 / 1200))) for index, left in enumerate((900, 2500)))
                annotations = (SimpleNamespace(frame_position=14, polygons=polygons, label_sha256="fixture"),)
                initial, _ = plan_initial_dynamic_tasks(SimpleNamespace(volume_id="fixture"), annotations, view, cache,
                              temp_root=root, conf=0.15, empty_frame_limit=30)
                owner = DynamicViewCompletion(initial[0].work.view, (0, 1))
                union_path = root / "union.raw"
                union = np.memmap(union_path, dtype=np.uint8, mode="w+", shape=cache.shape)
                result = drive_dynamic_workers(scheduler=owner, pool=ReorderedMaskPool(reverse), initial=initial,
                            view_plan=view, cache_ref=cache, view_union=union, temp_root=root, conf=0.15,
                            empty_frame_limit=30, worker_task_timeout=30, trace=None)
                identity = tuple((row["work_id"], tuple(row["crop_xyxy"]), row["patch"],
                                  tuple(row["seed_lineages"])) for row in result[2])
                self.assertTrue(result[3]["dynamic_crops"]["events"])
                outputs.append((hashlib.sha256(union).hexdigest(), identity))
                union._mmap.close()
        self.assertEqual(outputs[0], outputs[1])

    def test_worker_scaled_context_injects_model_masks_and_returns_native_dogfood(self):
        from XTA.lta_propagation import LtaObjectPrediction, LtaPropagationResult, write_seed_artifact
        from XTA.lta_sam import SamFramePrediction
        from XTA.lta_union_artifacts import read_union_array
        from XTA.lta_worker_adapter import execute_worker_task
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "view.raw"
            with path.open("wb") as handle:
                handle.truncate(3 * 1764 * 1764)
            cache = reference_existing_physical_view_cache(path, shape=(3, 1764, 1764),
                        physical_view_id="transverse", source_identity="fixture")
            tile = TilePlan(0, 0, 1764, 1764, 1764)
            item = obj(frame=0, top=350, left=500, mask=np.ones((500, 450), dtype=bool))
            irregular = np.eye(150, 120, dtype=bool)
            irregular[30:110, 20:100] = True
            peer = obj(frame=0, top=1200, left=1300, mask=irregular)
            peer = replace(peer, seed=replace(peer.seed,
                lineage=replace(peer.seed.lineage, lineage_id="irregular-peer")))
            seeds = tuple(value.in_crop(tile, object_id=index) for index, value in enumerate((item, peer)))
            artifact = write_seed_artifact(root / "seed.npz", seeds)
            window = WindowPlan("center", 0, 0, 3, 0, "both", "authoritative")
            payload = {"work_id": "scaled", "chain_work_id": "scaled", "sequence_id": "fixture",
                       "cache_ref": cache.payload(), "tile_index": 0, "tile_config_id": "dynamic",
                       "tile": asdict(tile), "neighbors": [], "windows": [asdict(window)],
                       "window": asdict(window), "seed_artifact_path": str(artifact.path),
                       "seed_artifact_sha256": artifact.sha256, "conf": 0.15,
                       "empty_frame_limit": 30, "crop_model_side": 1008,
                       "output_frame_start": 0, "output_frame_stop": 3,
                       "output_dir": str(root / "output")}
            context = SimpleNamespace(predictor=object(), profile={"name": "fixture"},
                                      sam_runtime={"version": "controlled"}, constrained_batches=None)
            expected = {seed.lineage: resize_crop_mask(seed.mask) for seed in seeds}
            def propagate(_a, _b, *, resource, request, prediction_callback, **_kwargs):
                self.assertIsInstance(resource, list)
                self.assertEqual(len(resource), 3)
                self.assertEqual(np.asarray(resource[0]).shape, (1008, 1008, 3))
                self.assertTrue(resource.source_cache_mapping_retired)
                self.assertEqual(len(request.seeds), 2)
                for seed in request.seeds:
                    np.testing.assert_array_equal(seed.mask, expected[seed.lineage])
                    for frame in range(3):
                        prediction_callback(LtaObjectPrediction(seed.lineage, SamFramePrediction(
                            request.session.sequence_id, request.session.session_index, frame, seed.object_id, 1.0,
                            expected[seed.lineage], 0.9), 0, seed.provenance))
                boundary = tuple(replace(seed, frame_index=2) for seed in request.seeds)
                return LtaPropagationResult(request, (), boundary,
                                             {"model_visited_frame_ranges": [[0, 3]]}, 0)
            with mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=propagate):
                result = execute_worker_task(context, "propagation_window", payload)
            manifest = json.loads(Path(result["artifact_path"]).read_text())
            union = read_union_array(manifest["union"])
            restored_masks = {lineage: resize_crop_mask(mask, side=1764, restore=True)
                              for lineage, mask in expected.items()}
            restored = np.logical_or.reduce(tuple(restored_masks.values()))
            self.assertEqual(union.shape, (3, 1764, 1764))
            np.testing.assert_array_equal(union[1], restored)
            dogfood = read_seed_artifact(manifest["dogfood_seed_artifacts"][0]["path"])
            self.assertEqual(len(dogfood), 2)
            for seed in dogfood:
                np.testing.assert_array_equal(seed.mask, restored_masks[seed.lineage])
            self.assertEqual(manifest["crop_transform"]["prediction_and_dogfood_coordinates"], "native_crop")
            # A subpixel seed cannot be certified as a successful empty run
            # when area downsampling removes all of its model-space support.
            tiny = obj(frame=0, top=800, left=800, mask=np.ones((1, 1), dtype=bool))
            tiny_artifact = write_seed_artifact(root / "tiny-seed.npz", (tiny.in_crop(tile, object_id=0),))
            tiny_payload = {**payload, "seed_artifact_path": str(tiny_artifact.path),
                "seed_artifact_sha256": tiny_artifact.sha256, "output_dir": str(root / "tiny-output")}
            with mock.patch("XTA.lta_propagation.run_mask_injected_session") as tracker:
                with self.assertRaisesRegex(ValueError, "seed mask must contain foreground"):
                    execute_worker_task(context, "propagation_window", tiny_payload)
            tracker.assert_not_called()
            self.assertFalse((root / "tiny-output" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
