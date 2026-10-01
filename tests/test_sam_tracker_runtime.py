from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from XTA.lta_experimental import run_mask_seed_session
from XTA.lta_sam import SamSessionPlan
from XTA.sam_tracker_runtime import (
    SamInterpolationTracker,
    execute_interpolation_tracker_task,
    load_tracker_run_result,
    materialize_interpolation_image_cache,
)
from tests.test_lta_experimental import _Measured, _Predictor, _Tracker


class _CombinedPredictor(_Predictor):
    def __init__(self, tracker):
        super().__init__(tracker)
        self.measured = _Measured()

    def handle_request(self, request):
        return self.measured.handle_request(request)


class _CompletionPool:
    """CPU protocol fixture delivering later submissions before the first."""

    def __init__(self, *, fail_run=None, shutdown_failure=False, corrupt_metadata=False):
        self.active = {}
        self.submissions = []
        self.peak_active = 0
        self.fail_run = fail_run
        self.shutdown_failure = shutdown_failure
        self.corrupt_metadata = corrupt_metadata
        self.closed = False
        self.shutdown_count = 0

    def submit(self, task, *, execution_device_id):
        if execution_device_id in self.active:
            raise AssertionError("a worker received more than one active endpoint session")
        self.active[execution_device_id] = task
        self.submissions.append((execution_device_id, task))
        self.peak_active = max(self.peak_active, len(self.active))

    def wait_result(self, *, timeout):
        import XTA.sam_tracker_runtime as runtime

        device = next(reversed(self.active))
        task = self.active.pop(device)
        if task.work_id == self.fail_run:
            raise RuntimeError("controlled raw worker failure")
        predictor = _CombinedPredictor(_Tracker())
        context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={})
        output = execute_interpolation_tracker_task(context, task.kind, task.payload)
        path = Path(output["artifact_path"])
        if self.corrupt_metadata:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt["request_metadata"]["parent_run_id"] = "wrong-original-hypothesis"
            path.write_text(json.dumps(receipt), encoding="utf-8")
        return SimpleNamespace(
            work_id=task.work_id, attempt_token=task.attempt_token,
            execution_device_id=device, worker_pid=100 + device,
            artifact_path=str(path), artifact_sha256=runtime._sha256(path),
        )

    def shutdown(self, *, timeout, force):
        self.shutdown_count += 1
        self.active.clear()
        self.closed = True
        if self.shutdown_failure:
            raise RuntimeError("controlled pool shutdown failure")

    @property
    def workers_settled(self):
        return self.closed and not self.active

    def force_close(self, *, timeout):
        self.active.clear()
        self.closed = True


class SamRawObservationTests(unittest.TestCase):
    def _run(self, *, scores=(-5.0,), direction="forward", prompt=10, callback=None):
        seed = np.zeros((9, 13), dtype=bool)
        seed[2:7, 3:9] = True
        predictor = _CombinedPredictor(_Tracker(score_logits=scores))
        result = run_mask_seed_session(
            predictor, predictor,
            resource=[Image.fromarray(np.zeros((9, 13, 3), dtype=np.uint8)) for _ in range(3)],
            session=SamSessionPlan("raw-test", 0, 10, 13),
            prompt_frame=prompt, ground_truth=seed, seed=None, object_masks=(seed,),
            conf=0.9, propagation_mode="tracker-only", propagation_direction=direction,
            raw_observation_callback=callback, retain_raw_observations=True,
        )
        return predictor, result

    def test_low_score_masks_remain_raw_through_terminal(self):
        streamed = []
        predictor, result = self._run(callback=streamed.append)
        self.assertEqual(result["propagation"], ())
        self.assertEqual([item.frame_index for item in streamed], [10, 11, 12])
        self.assertTrue(all(item.binary_mask.any() for item in streamed))
        self.assertTrue(all(item.detector_confidence is None for item in streamed))
        self.assertTrue(all(not item.binary_mask.flags.writeable for item in streamed))
        self.assertTrue(result["raw_observation_complete"])
        self.assertTrue(predictor.measured.closed)

    def test_removed_sentinel_retains_mask_and_unknown_score(self):
        _, result = self._run(scores=(-1e4,))
        observations = result["raw_observations"]
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(item.status == "removed" for item in observations))
        self.assertTrue(all(item.frame_tracker_score is None for item in observations))
        self.assertTrue(all(item.binary_mask.any() for item in observations))

    def test_backward_coverage_stops_at_seed_without_invented_suffix(self):
        _, result = self._run(direction="backward", prompt=11)
        self.assertEqual([item.frame_index for item in result["raw_observations"]], [10, 11])
        self.assertEqual(result["model_visited_frame_ranges"], ((10, 12),))

    def test_shape_mismatch_is_error_before_session_start(self):
        predictor = _CombinedPredictor(_Tracker())
        with self.assertRaisesRegex(ValueError, "geometry"):
            run_mask_seed_session(
                predictor, predictor, resource=[Image.fromarray(np.zeros((8, 13, 3), dtype=np.uint8))],
                session=SamSessionPlan("bad-geometry", 0, 0, 1), prompt_frame=0,
                ground_truth=np.ones((9, 13), dtype=bool), seed=None,
                object_masks=(np.ones((9, 13), dtype=bool),), conf=0.0,
                propagation_mode="tracker-only", raw_observation_callback=lambda item: None,
            )


class SamTrackerArtifactTests(unittest.TestCase):
    def _execute(self, directory, *, removed=False, output_name="result"):
        import XTA.sam_tracker_runtime as runtime

        root = Path(directory)
        images = np.arange(3 * 15 * 21, dtype=np.uint16).reshape(3, 15, 21).astype(np.uint8)
        cache = materialize_interpolation_image_cache(
            images, path=root / "image.bin", physical_view_id="transverse",
            source_identity="test-source",
        )
        seed = np.zeros((9, 13), dtype=bool)
        seed[2:7, 3:9] = True
        seed_path = root / "seed.npz"
        runtime._atomic_npz(seed_path, seed=seed)
        predictor = _CombinedPredictor(_Tracker(score_logits=(-1e4 if removed else -5.0,)))
        context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={})
        output = execute_interpolation_tracker_task(context, "sam_interpolation_run", {
            "run_id": "rect-test", "image_cache": cache.payload(),
            "seed_path": str(seed_path), "seed_sha256": runtime._sha256(seed_path),
            "crop_xyxy": [4, 2, 17, 11], "frame_start": 0, "frame_stop": 3,
            "seed_frame": 0, "direction": "forward", "output_dir": str(root / output_name),
        })
        return load_tracker_run_result(Path(output["artifact_path"]))

    def test_rectangular_raw_packet_roundtrip_independent_of_score(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._execute(directory)
            self.assertEqual(tuple(result.frames), (0, 1, 2))
            self.assertEqual(result.frames[2].shape, (9, 13))
            self.assertTrue(result.frames[2].any())
            self.assertTrue(result.receipt["prediction_valid"])
            self.assertTrue(all(value < 0.1 for value in result.tracker_scores.values()))
            self.assertTrue(all(value == "observed" for value in result.observation_status.values()))

    def test_removed_masks_are_invalid_evidence_instead_of_successful_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._execute(directory, removed=True)
            self.assertFalse(result.receipt["prediction_valid"])
            self.assertEqual(result.receipt["status"], "invalid_removed_object")
            self.assertTrue(result.receipt["coverage_complete"])
            self.assertTrue(result.frames[2].any())

    def test_transfer_corruption_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self._execute(directory)
            path = Path(directory) / "result" / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            masks = Path(manifest["raw_masks"]["path"])
            masks.write_bytes(masks.read_bytes() + b"corruption")
            with self.assertRaisesRegex(RuntimeError, "checksum/size"):
                load_tracker_run_result(path)

    def test_bad_seed_rejected_before_predictor_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = np.zeros((3, 15, 21), dtype=np.uint8)
            cache = materialize_interpolation_image_cache(
                frames, path=Path(directory) / "image.bin", physical_view_id="transverse",
                source_identity="test",
            )
            tracker = SamInterpolationTracker(
                model_path="unused", device_ids=(0,), artifact_root=Path(directory) / "runs",
                source_cache_ref=cache,
            )
            with patch.object(tracker, "start") as started:
                with self.assertRaisesRegex(ValueError, "boolean"):
                    tracker.run(run_id="invalid", seed_mask=np.ones((9, 13), dtype=np.uint8),
                                seed_frame=0, frame_start=0, frame_stop=3, direction="forward",
                                crop_xyxy=(4, 2, 17, 11))
                started.assert_not_called()
            tracker.close()


class SamConcurrentDispatchTests(unittest.TestCase):
    _execute = SamTrackerArtifactTests._execute

    def _tracker(self, directory, pool, *, devices=(0, 1)):
        cache = materialize_interpolation_image_cache(
            np.zeros((3, 15, 21), dtype=np.uint8), path=Path(directory) / "images.bin",
            physical_view_id="transverse", source_identity="immutable-source",
        )
        tracker = SamInterpolationTracker(
            model_path="unused", device_ids=devices,
            artifact_root=Path(directory) / "staging", source_cache_ref=cache,
        )
        tracker._pool = pool
        return tracker, cache

    def _request(self, index):
        seed = np.zeros((9, 13), dtype=bool)
        seed[2:7, 2 + index : 5 + index] = True
        backward = index % 2 == 1
        return dict(run_id=f"run-{index}", seed_mask=seed,
                    seed_frame=2 if backward else 0, frame_start=0, frame_stop=3,
                    direction="backward" if backward else "forward", crop_xyxy=(4, 2, 17, 11))

    def test_reverse_completion_keeps_identity_and_dispatches_next_before_yield(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            consumed = []
            requests = [self._request(index) for index in range(4)]
            with patch.object(tracker, "start", return_value=tracker) as started:
                for index, result in tracker.iter_results(requests, source_cache_ref=cache):
                    consumed.append(index)
                    self.assertEqual(result.receipt["run_id"], f"run-{index}")
                    self.assertEqual(result.receipt["dispatch"]["input_index"], index)
                    for mask in result.frames.values():
                        self.assertTrue(np.array_equal(mask, requests[index]["seed_mask"]))
                    if len(consumed) == 1:
                        self.assertEqual(len(pool.submissions), 3)
                    tracker.release_result(result)
                self.assertGreater(started.call_count, 0)
            self.assertEqual(consumed, [1, 2, 3, 0])
            self.assertEqual(pool.peak_active, 2)
            self.assertEqual(tracker.dispatch_stats["peak_in_flight"], 2)
            self.assertEqual(tracker.dispatch_stats["submitted"], 4)
            self.assertEqual(list(tracker.artifact_root.glob("run-*")), [])
            tracker.close()

    def test_empty_iterator_has_no_predictor_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start") as started:
                self.assertEqual(list(tracker.iter_results((), source_cache_ref=cache)), [])
                started.assert_not_called()
            tracker.close()

    def test_single_credit_preserves_crop_affinity_and_bounds_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start", return_value=tracker):
                for _index, result in tracker.iter_results(
                    (self._request(index) for index in range(4)), source_cache_ref=cache, max_in_flight=1,
                ):
                    self.assertLessEqual(len(list(tracker.artifact_root.glob("run-*"))), 2)
            self.assertEqual([device for device, _task in pool.submissions], [0, 0, 0, 0])
            self.assertEqual(pool.peak_active, 1)
            self.assertEqual(tracker.dispatch_stats["affinity_hits"], 3)
            tracker.close()

    def test_source_swap_during_iterator_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start", return_value=tracker):
                stream = tracker.iter_results((self._request(0), self._request(1)), source_cache_ref=cache)
                _index, result = next(stream)
                with self.assertRaisesRegex(RuntimeError, "while.*iterator"):
                    tracker.set_source_cache(cache)
                tracker.release_result(result)
                stream.close()
            self.assertTrue(tracker._closed)
            self.assertEqual(pool.shutdown_count, 1)
            self.assertEqual(list(tracker.artifact_root.glob("run-*")), [])

    def test_worker_failure_cleans_all_staging_and_releases_dispatch_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool(fail_run="run-1", shutdown_failure=True)
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start", return_value=tracker):
                with self.assertRaisesRegex(RuntimeError, "controlled raw worker failure"):
                    list(tracker.iter_results((self._request(0), self._request(1)), source_cache_ref=cache))
            self.assertTrue(tracker._closed)
            self.assertEqual(list(tracker.artifact_root.glob("run-*")), [])
            self.assertTrue(tracker._dispatch_lock.acquire(blocking=False))
            tracker._dispatch_lock.release()

    def test_duplicate_run_identity_invalidates_outstanding_work(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start", return_value=tracker):
                with self.assertRaisesRegex(ValueError, "duplicate SAM run_id"):
                    list(tracker.iter_results((self._request(0), self._request(0)), source_cache_ref=cache))
            self.assertTrue(tracker._closed)
            self.assertEqual(pool.shutdown_count, 1)
            self.assertEqual(list(tracker.artifact_root.glob("run-*")), [])

    def test_original_run_tile_metadata_is_bounded_and_echoed_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool()
            tracker, cache = self._tracker(directory, pool)
            request = self._request(0)
            request["metadata"] = {
                "parent_run_id": "original-endpoint-forward",
                "tile_id": "tile-r0-c0", "crop_mode": "tiled",
                "ownership_bbox_yx": (2, 4, 11, 17),
            }
            with patch.object(tracker, "start", return_value=tracker):
                results = list(tracker.iter_results((request,), source_cache_ref=cache))
            self.assertEqual(results[0][1].receipt["request_metadata"], {
                "parent_run_id": "original-endpoint-forward", "tile_id": "tile-r0-c0",
                "crop_mode": "tiled", "ownership_bbox_yx": [2, 4, 11, 17],
            })
            tracker.close()

    def test_corrupt_parent_tile_attribution_cannot_be_selected_as_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = _CompletionPool(corrupt_metadata=True)
            tracker, cache = self._tracker(directory, pool)
            request = self._request(0)
            request["metadata"] = {"parent_run_id": "real-original-hypothesis", "tile_id": "tile-0"}
            with patch.object(tracker, "start", return_value=tracker):
                with self.assertRaisesRegex(RuntimeError, "original-run/tile attribution"):
                    list(tracker.iter_results((request,), source_cache_ref=cache))
            self.assertTrue(tracker.residency_released)
            self.assertEqual(list(tracker.artifact_root.glob("run-*")), [])

    def test_metadata_rejects_array_transfers_and_unbounded_attribution(self):
        from XTA.sam_tracker_runtime import _request_metadata

        with self.assertRaisesRegex(TypeError, "SAM request metadata"):
            _request_metadata({"do_not_transfer_mask": np.ones((9, 13), dtype=bool)})
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            _request_metadata({"oversized": "x" * (64 * 1024)})
        with self.assertRaises(ValueError):
            _request_metadata({"invalid_measurement": float("nan")})

    def test_shutdown_error_releases_residency_only_after_process_exit_proof(self):
        class StickyPool(_CompletionPool):
            sticky = True

            def shutdown(self, *, timeout, force):
                self.closed = True
                raise RuntimeError("controlled close failure")

            @property
            def workers_settled(self):
                return not self.sticky

            def force_close(self, *, timeout):
                if self.sticky:
                    raise RuntimeError("controlled process still alive")

        with tempfile.TemporaryDirectory() as directory:
            pool = StickyPool()
            tracker, _cache = self._tracker(directory, pool)
            with self.assertRaisesRegex(RuntimeError, "controlled close failure"):
                tracker.close()
            self.assertFalse(tracker.residency_released)
            self.assertIs(tracker._pool, pool)
            pool.sticky = False
            with self.assertRaisesRegex(RuntimeError, "controlled close failure"):
                tracker.close()
            self.assertTrue(tracker.residency_released)
            self.assertIsNone(tracker._pool)

    def test_closed_queue_flag_does_not_hide_a_lingering_process(self):
        from XTA.lta_workers import LtaWorkerPool

        class Process:
            alive = True
            terminate_calls = 0
            kill_calls = 0

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminate_calls += 1

            def join(self, *, timeout):
                pass

            def kill(self):
                self.kill_calls += 1
                self.alive = False

        pool = object.__new__(LtaWorkerPool)
        process = Process()
        pool._processes = {(0, 0): process}
        pool._closed = True
        pool.device_ids = (0,)
        self.assertFalse(pool.workers_settled)
        self.assertEqual(pool.shutdown(timeout=0.0, force=True), (0,))
        self.assertTrue(pool.workers_settled)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)

    def test_consumer_close_surfaces_unproven_cleanup_and_retains_staging(self):
        class StickyPool(_CompletionPool):
            sticky = True

            def shutdown(self, *, timeout, force):
                if self.sticky:
                    self.closed = True
                    raise RuntimeError("controlled settlement failure")
                return super().shutdown(timeout=timeout, force=force)

            @property
            def workers_settled(self):
                return not self.sticky

            def force_close(self, *, timeout):
                if self.sticky:
                    raise RuntimeError("controlled child still alive")

        with tempfile.TemporaryDirectory() as directory:
            pool = StickyPool()
            tracker, cache = self._tracker(directory, pool)
            with patch.object(tracker, "start", return_value=tracker):
                stream = tracker.iter_results((self._request(0), self._request(1)), source_cache_ref=cache)
                _index, result = next(stream)
                with self.assertRaisesRegex(RuntimeError, "controlled settlement failure"):
                    stream.close()
            self.assertFalse(tracker.residency_released)
            self.assertTrue(Path(result.receipt["temporary_artifact_directory"]).exists())
            self.assertTrue(tracker._dispatch_lock.acquire(blocking=False))
            tracker._dispatch_lock.release()
            pool.sticky = False
            tracker.close()

    def test_runtime_cancel_prevents_model_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = SamInterpolationTracker(model_path="unused", device_ids=(0,), artifact_root=directory)
            tracker.cancel("controlled cancellation")
            with patch("XTA.lta_workers.LtaWorkerPool") as factory:
                with self.assertRaisesRegex(RuntimeError, "controlled cancellation"):
                    tracker.start()
                factory.assert_not_called()
            tracker.close()

    def test_parent_cancel_interrupts_worker_result_wait_without_empty_artifact(self):
        from XTA.lta_workers import LtaWorkerPool
        from tests.test_lta_workers import _fake_adapter, _init, _task

        with _fake_adapter() as (module, root, log):
            cancelled = threading.Event()
            pool = LtaWorkerPool((0,), _init(module, log), cancel_event=cancelled)
            artifact = root / "cancelled-result.json"
            pool.submit(_task("cancel-test", "cancel-token", artifact, kind="hang", seconds=10.0),
                        execution_device_id=0)
            timer = threading.Timer(0.1, cancelled.set)
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "cancelled.*completed evidence"):
                    pool.wait_result(timeout=5.0)
                self.assertLess(time.monotonic() - started, 2.0)
                self.assertFalse(artifact.exists())
            finally:
                timer.cancel()
                pool.shutdown(timeout=0.1, force=True)

    def test_staging_release_keeps_transferred_masks_and_bounds_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._execute(directory, output_name="run-000000-test")
            tracker = SamInterpolationTracker(
                model_path="unused", device_ids=(0,), artifact_root=directory,
            )
            tracker.release_result(result)
            self.assertFalse((Path(directory) / "run-000000-test").exists())
            self.assertTrue(result.frames[2].any())
            foreign = self._execute(directory, output_name="outside-staging")
            with self.assertRaisesRegex(RuntimeError, "outside"):
                tracker.release_result(foreign)
            self.assertTrue((Path(directory) / "outside-staging").is_dir())
            tracker.close()


if __name__ == "__main__":
    unittest.main()
