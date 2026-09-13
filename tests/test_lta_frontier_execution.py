from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import lta_execution
from XTA.lta_coverage import LtaCoverageLedger
from XTA.lta_propagation import (
    LtaMaskSeed,
    LtaSeedProvenance,
    read_seed_artifact,
    run_mask_injected_session,
    write_seed_artifact,
)
from XTA.lta_rendering import reference_existing_physical_view_cache
from XTA.lta_relay_episodes import read_relay_observations, write_relay_observations
from XTA.lta_runtime import LtaTileGridPlan
from XTA.lta_sam import SamFramePrediction
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import plan_tile_grid
from XTA.lta_worker_adapter import execute_worker_task
from XTA.lta_workers import LtaWorkerResult


class ArtifactWorkerPool:
    """Run the actual worker adapter while varying completion and device order."""

    def __init__(self, *, devices=(0,), workers_per_device=1, reverse=False, corrupt=None):
        self.device_ids = tuple(devices)
        self.workers_per_device = workers_per_device
        self.pids = {device: 1000 + device * 10 for device in devices}
        self.pids_by_slot = {
            (device, slot): self.pids[device] + slot
            for device in devices for slot in range(workers_per_device)
        }
        self.ready_events = ()
        self.reverse = reverse
        self.corrupt = corrupt
        self.pending = []
        self.submitted = []
        self.completed = []
        self.maximum_active = 0
        self.context = SimpleNamespace(
            predictor=object(), profile={"name": "cpu_frontier_protocol_test"},
            sam_runtime={"distribution_version": "fake"}, constrained_batches=None,
        )

    def submit(self, task, *, execution_device_id, worker_index=0):
        slot = execution_device_id, worker_index
        if any((device, worker) == slot for _task, device, worker in self.pending):
            raise AssertionError("worker slot received simultaneous tasks")
        seeds = read_seed_artifact(
            task.payload["seed_artifact_path"],
            expected_sha256=task.payload["seed_artifact_sha256"],
        )
        self.submitted.append({
            "task": task,
            "seeds": seeds,
            "completed_before_submit": len(self.completed),
        })
        self.pending.append((task, execution_device_id, worker_index))
        self.maximum_active = max(self.maximum_active, len(self.pending))

    def wait_result(self, timeout=None):
        task, device, worker = self.pending.pop(-1 if self.reverse else 0)
        output = execute_worker_task(self.context, task.kind, task.payload)
        artifact = Path(output["artifact_path"])
        manifest = json.loads(artifact.read_text())
        if self.corrupt == "coverage":
            with Path(manifest["lineage_coverage"]["path"]).open("ab") as handle:
                handle.write(b"corrupt coverage")
        elif self.corrupt == "dogfood" and manifest.get("dogfood_seed_artifacts"):
            with Path(manifest["dogfood_seed_artifacts"][0]["path"]).open("ab") as handle:
                handle.write(b"corrupt dogfood")
        self.completed.append(task.work_id)
        return LtaWorkerResult(
            work_id=task.work_id, attempt_token=task.attempt_token, kind=task.kind,
            execution_device_id=device, worker_index=worker,
            worker_pid=self.pids_by_slot[(device, worker)],
            artifact_path=str(artifact),
            artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            artifact_size_bytes=artifact.stat().st_size,
        )

    def check_liveness(self):
        pass


class RecordingTrace:
    def __init__(self):
        self.events = []

    def event(self, name, **fields):
        self.events.append((name, fields))

    @contextmanager
    def phase(self, name, **fields):
        self.event(name + ":start", **fields)
        yield
        self.event(name + ":end", **fields)

    def flush(self):
        pass


class LtaFrontierExecutionTests(unittest.TestCase):
    @staticmethod
    def mask(*points):
        result = np.zeros((4, 4), dtype=bool)
        for row, column in points:
            result[row, column] = True
        return result

    def run_frontier(self, root, offerings, *, count=120, devices=(0,),
                     workers_per_device=1, reverse=False, corrupt=None, transform=None):
        root.mkdir(parents=True)
        grid = LtaTileGridPlan(
            "s4_st2", 4, 2,
            plan_tile_grid(source_width=6, source_height=4, tile_size=4, tile_stride=2),
        )
        view = SimpleNamespace(
            volume_id="volume", physical_view_id="transverse",
            runtime_view_id="transverse__tta_a0", frame_count=count,
            frame_height=4, frame_width=6, tile_grids=(grid,),
        )
        cache_path = root / "cache.raw"
        np.zeros((count, 4, 6), dtype=np.uint8).tofile(cache_path)
        cache = reference_existing_physical_view_cache(
            cache_path, shape=(count, 4, 6), physical_view_id="transverse",
            source_identity="frontier-protocol-fixture",
        )
        records = []
        for index, (name, frame, direction, mask) in enumerate(offerings):
            lineage = LtaLineageId(
                "volume", "transverse", view.runtime_view_id, name,
                tile_config_id=grid.config_id,
            )
            seed = LtaMaskSeed(
                lineage, frame, index + 7, mask,
                provenance=LtaSeedProvenance.SPATIAL_RELAY,
                tracker_probability=0.9, relay_generation=1,
                visited_tile_indices=(0, 1),
            )
            artifact = write_seed_artifact(root / "incoming" / f"seed-{index:04d}.npz", (seed,))
            records.append({
                "lineage": asdict(lineage), "source_tile_index": 0,
                "destination_tile_index": 1, "frame_index": frame,
                "temporal_direction": direction, "generation": 1,
                "seed_artifact_path": str(artifact.path),
                "seed_artifact_sha256": artifact.sha256,
            })
        self.last_pool = pool = ArtifactWorkerPool(
            devices=devices, workers_per_device=workers_per_device,
            reverse=reverse, corrupt=corrupt,
        )
        trace = RecordingTrace()
        audit = lta_execution._new_worker_audit()
        audit.update(task_count=0, skipped_window_count=0, workers_per_gpu=workers_per_device)
        dispatched = []
        union = np.zeros((count, 4, 6), dtype=np.uint8)
        self.last_union = union

        def sam_adapter(_measured, _raw, **kwargs):
            session, prompt = kwargs["session"], kwargs["prompt_frame"]
            direction = kwargs["propagation_direction"]
            if direction == "forward":
                frames = range(prompt, session.frame_stop)
            elif direction == "backward":
                frames = range(prompt, session.frame_start - 1, -1)
            else:
                raise AssertionError("a relay frontier must remain directional")
            for frame in frames:
                for object_id, seed in enumerate(kwargs["object_masks"]):
                    mask = np.asarray(seed).copy()
                    if transform is not None:
                        mask = transform(frame, direction, mask)
                    kwargs["prediction_callback"](SamFramePrediction(
                        session.sequence_id, session.session_index, frame,
                        object_id, 1.0, mask, 0.9,
                    ))
            return {"propagation": (), "seed_roundtrip_policy": "overlap-aware",
                    "seed_roundtrip_passed": True, "anchor_integrity_passed": True}

        def propagate(measured, raw, **kwargs):
            return run_mask_injected_session(measured, raw, adapter=sam_adapter, **kwargs)

        with LtaCoverageLedger(root / "coverage.sqlite3", frame_count=count) as coverage:
            with mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=propagate), \
                    mock.patch("XTA.lta_execution.PRODUCTION_LTA_RELAY_MIN_PIXELS", 1), \
                    mock.patch("XTA.lta_execution._plan_window_tasks",
                               side_effect=AssertionError("frontier expanded a full chain")):
                from XTA.lta_frontier_execution import _drive_canonical_relay_frontier

                generation = _drive_canonical_relay_frontier(
                    records, pool=pool, view_plan=view, cache_ref=cache,
                    view_union=union, coverage=coverage, temp_root=root, conf=0.15,
                    empty_frame_limit=30, worker_task_timeout=30, trace=trace,
                    worker_audit=audit, dispatched=dispatched, first_plan_order=0,
                    max_relay_generations=12,
                )
            coverage_stats = coverage.stats()
        return SimpleNamespace(union=union, pool=pool, trace=trace, audit=audit,
                               generation=generation, dispatched=dispatched,
                               coverage_stats=coverage_stats)

    @staticmethod
    def logical_tasks(result):
        return [
            (
                item["task"].work_id,
                tuple(sorted(item["task"].payload["windows"][0].items())),
                tuple((seed.lineage.token, seed.frame_index,
                       np.packbits(seed.mask, bitorder="little").tobytes())
                      for seed in item["seeds"]),
            )
            for item in result.pool.submitted
        ]

    def test_many_offset_seeds_fan_in_before_one_boundary_continuation(self):
        mask = self.mask((1, 3))
        offerings = [("same", frame, "forward", mask) for frame in range(1, 21)]
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_frontier(Path(folder) / "fanin", offerings)
        self.assertEqual(int(result.union[:1].sum()), 0)
        np.testing.assert_array_equal(result.union[1:, 1, 5], np.ones(119, dtype=np.uint8))
        continuation = [item for item in result.pool.submitted
                        if item["task"].payload["windows"][0]["prompt_frame"] == 29]
        self.assertEqual(len(continuation), 1)
        self.assertEqual(continuation[0]["completed_before_submit"], 20)
        self.assertLessEqual(len(result.pool.submitted), 25)
        self.assertTrue(all(len(item["task"].payload["windows"]) == 1
                            for item in result.pool.submitted))

    def test_unmatched_lineage_and_novel_support_survive_boundary_fanin(self):
        base, novel, other = self.mask((0, 3)), self.mask((1, 3)), self.mask((0, 3), (3, 3))
        offerings = [("same", 2, "forward", base), ("same", 8, "forward", novel),
                     ("unmatched", 8, "forward", other)]
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_frontier(Path(folder) / "support", offerings, count=64)
        np.testing.assert_array_equal(result.union[8:, :, 5],
                                      np.tile([1, 1, 0, 1], (56, 1)))
        boundary = [seed for item in result.pool.submitted for seed in item["seeds"]
                    if seed.frame_index == 29]
        self.assertEqual({seed.lineage.lineage_id for seed in boundary}, {"same", "unmatched"})
        same = next(seed for seed in boundary if seed.lineage.lineage_id == "same")
        np.testing.assert_array_equal(same.mask, base | novel)

    def test_directions_and_later_reentry_remain_independent(self):
        mask = self.mask((1, 3))
        offerings = [("same", 10, "backward", mask), ("same", 10, "forward", mask),
                     ("same", 60, "forward", mask)]

        def empty_gap(frame, _direction, source):
            return np.zeros_like(source) if 29 <= frame < 60 else source

        with tempfile.TemporaryDirectory() as folder:
            result = self.run_frontier(Path(folder) / "reentry", offerings,
                                       count=91, transform=empty_gap)
        np.testing.assert_array_equal(result.union[:29, 1, 5], np.ones(29, dtype=np.uint8))
        self.assertFalse(np.any(result.union[29:60]))
        np.testing.assert_array_equal(result.union[60:, 1, 5], np.ones(31, dtype=np.uint8))
        directions = {item["task"].payload["windows"][0]["direction"] for item in result.pool.submitted}
        self.assertEqual(directions, {"forward", "backward"})

    def test_masks_and_logical_tasks_do_not_depend_on_completion_order_or_device_count(self):
        offerings = [(f"lineage-{index % 3}", index + 1, "forward",
                      self.mask((index % 3, 3))) for index in range(12)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            serial = self.run_frontier(root / "serial", offerings, count=64)
            reordered = self.run_frontier(root / "parallel", offerings, count=64,
                                          devices=(0, 1), workers_per_device=2, reverse=True)
        np.testing.assert_array_equal(serial.union, reordered.union)
        self.assertEqual(self.logical_tasks(serial), self.logical_tasks(reordered))
        self.assertEqual(serial.generation, reordered.generation)
        self.assertEqual(reordered.pool.maximum_active, 4)

    def test_more_than_one_wave_keeps_boundary_fanin_and_slot_independent_batches(self):
        offerings = [(f"lineage-{row}", frame, "forward", self.mask((row, 3)))
                     for row in (0, 2) for frame in range(1, 21)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            serial = self.run_frontier(root / "serial", offerings, count=64)
            reordered = self.run_frontier(root / "parallel", list(reversed(offerings)),
                                          count=64, devices=(0, 1),
                                          workers_per_device=2, reverse=True)
        np.testing.assert_array_equal(serial.union, reordered.union)
        self.assertEqual(self.logical_tasks(serial), self.logical_tasks(reordered))
        for result in (serial, reordered):
            waves = [fields for name, fields in result.trace.events if name == "frontier_wave_start"]
            self.assertEqual(waves[0]["input_revisions"], 32)
            self.assertTrue(all(wave["input_revisions"] <= 32 for wave in waves))
            boundary = [item for item in result.pool.submitted
                        if item["task"].payload["windows"][0]["prompt_frame"] == 29]
            self.assertEqual(len(boundary), 1)
            self.assertGreaterEqual(boundary[0]["task"].payload["frontier_wave"], 3)
            np.testing.assert_array_equal(result.union[1:, 0, 5], np.ones(63, dtype=np.uint8))
            np.testing.assert_array_equal(result.union[1:, 2, 5], np.ones(63, dtype=np.uint8))

    def test_spatial_relays_wait_for_temporal_frontier_and_settle_without_cycle_replay(self):
        mask = self.mask((1, 0), (1, 3))
        offerings = [("same", 3, "forward", mask), ("same", 11, "forward", mask)]
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_frontier(Path(folder) / "neighbors", offerings, count=64,
                                       devices=(0, 1), reverse=True)
        self.assertEqual({item["task"].payload["tile_index"]
                          for item in result.pool.submitted}, {0, 1})
        self.assertGreater(result.generation, 1)
        generations = [item["task"].payload["relay_generation"]
                       for item in result.pool.submitted]
        self.assertEqual(generations, sorted(generations))
        first_next_generation = next(item for item in result.pool.submitted
                                     if item["task"].payload["relay_generation"] > 1)
        initial_generation_count = generations.count(1)
        self.assertEqual(first_next_generation["completed_before_submit"], initial_generation_count)
        np.testing.assert_array_equal(result.union[3:, 1, 2], np.ones(61, dtype=np.uint8))
        np.testing.assert_array_equal(result.union[3:, 1, 5], np.ones(61, dtype=np.uint8))
        self.assertLess(len(result.pool.submitted), 30)

    def test_corrupt_receipt_never_admits_a_continuation_wave(self):
        for corrupt in ("coverage", "dogfood"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex((ValueError, RuntimeError), "(size|digest|hash)"):
                    self.run_frontier(Path(folder) / corrupt,
                                      [("same", 2, "forward", self.mask((1, 3)))],
                                      count=64, corrupt=corrupt)
                self.assertEqual(len(self.last_pool.submitted), 1)

    def test_self_consistent_misbound_receipts_fail_before_reduction_or_continuation(self):
        original_wait = ArtifactWorkerPool.wait_result
        for corruption in ("tile_origin", "tile_config", "work_id", "lineage", "neighbor",
                           "frame", "shape", "object_id", "visited"):
            def misbind(pool, timeout=None):
                result = original_wait(pool, timeout)
                path = Path(result.artifact_path)
                manifest = json.loads(path.read_text())
                if corruption == "tile_origin":
                    manifest["tile"]["left"] = 0
                elif corruption == "tile_config":
                    manifest["tile_config_id"] = "other-grid"
                elif corruption == "work_id":
                    manifest["work_id"] = "another-task"
                else:
                    observations = read_relay_observations(manifest["relay_observation_artifact"])
                    self.assertEqual(len(observations), 1)
                    record = next(iter(observations.values()))
                    if corruption == "lineage":
                        record["lineage"] = replace(record["lineage"], lineage_id="invented-lineage")
                    elif corruption == "neighbor":
                        record["destination_index"] = 1
                    elif corruption == "object_id":
                        record["seed"] = replace(record["seed"], object_id=record["seed"].object_id + 1)
                    elif corruption == "visited":
                        record["seed"] = replace(record["seed"], visited_tile_indices=())
                    else:
                        first, last = record["episodes"][0]
                        if corruption == "frame":
                            first = last = (4, *first[1:])
                        else:
                            wrong_shape = np.zeros((5, 5), dtype=bool)
                            wrong_shape[1, 2] = True
                            packed = np.packbits(wrong_shape.reshape(-1))
                            first = (first[0], packed, wrong_shape.shape, first[3])
                            last = (last[0], packed, wrong_shape.shape, last[3])
                        record["episodes"] = [(first, last)]
                    observations = {(record["lineage"].token, record["destination_index"]): record}
                    manifest["relay_observation_artifact"] = write_relay_observations(path.parent, observations)
                path.write_text(json.dumps(manifest, sort_keys=True))
                return replace(result, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               artifact_size_bytes=path.stat().st_size)

            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(ArtifactWorkerPool, "wait_result", misbind), \
                    mock.patch("XTA.lta_execution._consume_chain_manifest",
                               wraps=lta_execution._consume_chain_manifest) as reduce_union, \
                    mock.patch("XTA.lta_execution._ingest_lineage_coverage",
                               wraps=lta_execution._ingest_lineage_coverage) as ingest:
                with self.assertRaisesRegex(RuntimeError, "(frontier|manifest does not match)"):
                    self.run_frontier(Path(folder) / corruption,
                                      [("same", 1, "forward", self.mask((1, 0)))], count=4)
                reduce_union.assert_not_called()
                ingest.assert_not_called()
                self.assertFalse(np.any(self.last_union))
                self.assertEqual(len(self.last_pool.submitted), 1)


if __name__ == "__main__":
    unittest.main()
