"""Retry-group grants compose with the real producer/scheduler/raw adapter.

The completion pool mocks CUDA/learned inference; its CPU SDK fixture still
executes the production mask-session adapter and artifact-retirement boundary.
"""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
import threading

import numpy as np

from XTA import sam_extrapolation as extrapolation, sam_resources as resources
from XTA.sam_integration import SamInterpolationContext
from XTA.sam_retry_groups import execute_retry_groups
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_tracker_runtime import _CompletionPool


def test_live_retry_groups_reach_real_independent_producer_and_raw_adapter(tmp_path, monkeypatch):
    observed = np.zeros((8, 64, 256), dtype=np.uint8)
    observed[2:5, 20:27, 30:37] = 1
    observed[2:5, 40:47, 190:197] = 2
    observed.flags.writeable = False
    gray = np.zeros_like(observed)
    cache = materialize_interpolation_image_cache(gray, path=tmp_path / "gray.bin",
        physical_view_id="transverse", source_identity="frozen-group-source")
    context = SamInterpolationContext(model_path="CPU-SDK-fixture", device_ids=(0, 1),
        temp_dir=tmp_path / "context", evidence_root=tmp_path / "evidence",
        source_volume=gray, source_identity="frozen-group-source")
    tracker = SamInterpolationTracker(model_path="CPU-SDK-fixture", device_ids=(0, 1),
        workers_per_device=1, artifact_root=tmp_path / "workers")
    records = []

    class AdmittedPool(_CompletionPool):
        def submit(self, task, *, execution_device_id):
            with tracker._state_condition:
                scope = next(scope for scope in tracker._scopes.values()
                    if (task.work_id, task.attempt_token) in scope.outstanding)
                assert scope.capacity == scope.admission.max_in_flight == 1
                assert scope.admission._profile._lease.retry_child_factory is not None
                assert scope.admission_limits["retry_group_affine"] is True
                assert scope.independent_preparation is True
                assert scope.preparation_thread_id is not None
                assert scope.preparation_thread_id != scope.producer_thread_id
                records.append((scope.token, scope.producer_thread_id, scope.preparation_thread_id))
            return super().submit(task, execution_device_id=execution_device_id)

    # No first completion is allowed until two distinct groups are in flight.
    pool = AdmittedPool(expected_initial=2)
    tracker._pool = pool
    monkeypatch.setattr(tracker, "start", lambda: tracker)
    context._runtime = tracker
    ledger = SimpleNamespace(capacity=128 * resources.GIB, in_use=0,
        condition=threading.Condition(threading.RLock()))
    try:
        with resources.admit_sam_parent_resources(ledger, 10 * resources.GIB, "parent",
                worker_count=2, execution_slots=2, base_allowance_bytes=4 * resources.GIB,
                headroom_probe=lambda: 256 * resources.GIB) as parent:
            prepared = extrapolation.prepare_sam_extrapolation_pass(observed, scope="group-boundary",
                distance=3, walk_back=1, min_radius=3., resource_profile=parent,
                eligible_terminals=lambda terminal, direction: terminal.frame_index == 4 and direction == 1)
            assert len(prepared.groups) == 2 and len(prepared.runs) == 4
            before = ledger.in_use

            def run(ordinal, profile, child):
                assert child and context._resource_local.profile is profile
                profile._validate_owner()
                group = prepared.groups[ordinal]
                subset = extrapolation._cohort_prepared(prepared, (group.group_id,))
                pixels = ((group.context_bbox_yx[2] - group.context_bbox_yx[0])
                          * (group.context_bbox_yx[3] - group.context_bbox_yx[1]))
                frames = max(len(run.expected_frames) for run in subset.runs)
                wave = resources.cpu_wave_admission(resources.cpu_session_bytes(frames, pixels)["estimated_peak_bytes"],
                    frames * pixels, profile.assigned_cpu_wave_bytes, 1)
                subset = replace(subset, cpu_wave_admission=MappingProxyType(wave))
                schedules, completed = [], []
                stream = extrapolation._iterate_extrapolation_tracker_results(tracker, subset, 1,
                    {group.group_id: group}, prepared.plan.by_id, cache, context._cancel, profile, schedules)
                try:
                    for index, result in stream:
                        run_plan = subset.runs[index]
                        try:
                            assert result.receipt["coverage_complete"] is True
                            assert result.receipt["cpu_task_retirement"]["known_owners_retired"] is True
                            assert tuple(sorted(result.frames)) == tuple(sorted(run_plan.expected_frames))
                            completed.append(result.receipt["run_id"])
                        finally:
                            tracker.release_result(result)
                finally:
                    stream.close()
                assert schedules[0]["mode"] == "exact_crop_family_fifo"
                assert completed == [run.run_id for run in subset.runs]
                return {"ordinal": ordinal, "runs": tuple(completed), "attempt_count": 1}

            outcomes, stats = execute_retry_groups(tuple(group.group_id for group in prepared.groups), run,
                parent_profile=parent, context=context, runtime=tracker,
                parent_scope="group-boundary", family="radial", max_in_flight=2)
            assert [result["ordinal"] for result in outcomes] == [0, 1]
            assert stats["child_groups"] == 2 and stats["serial_fallback_groups"] == 0
            assert pool.peak_active == 2 and len(pool.completions) == 4
            assert len({token for token, _creator, _producer in records}) == 2
            assert len({creator for _token, creator, _producer in records}) == 2
            assert len({producer for _token, _creator, producer in records}) == 2
            assert not pool.active and not tracker._scopes
            assert not list(tracker.artifact_root.glob("run-*"))
            assert all(proof["active_scopes"] == 0 for proof in tracker._source_cache_retirement_proofs.values())
            assert not context._retry_source_holds and context._active_passes == 0
            assert parent._lease.scope_holds == 0 and ledger.in_use == before
    finally:
        context.close()
    assert pool.closed and pool.workers_settled and tracker.residency_released
    assert ledger.in_use == 0
