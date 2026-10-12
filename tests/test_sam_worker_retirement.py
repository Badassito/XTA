"""CPU ownership witnesses for successful and malformed SAM session cleanup."""
from types import SimpleNamespace
import copy
from dataclasses import make_dataclass
import json
import os
from pathlib import Path
import weakref

import numpy as np
from PIL import Image
import pytest

from XTA.lta_experimental import run_mask_seed_session
from XTA.lta_sam import SamSessionPlan
from tests.test_lta_experimental import _Predictor, _Tracker


def run_session(*, keep_registry=False, keep_state=False, response=None,
                missing_registry=False, close_error=None, tracker_error=None):
    predictor = _Predictor(_Tracker())
    state = predictor._all_inference_states["mask-test"]["state"]
    state["normalized_cpu_clip"] = np.zeros((3, 3, 16, 16), dtype=np.float16)
    owners = [weakref.ref(state["normalized_cpu_clip"])]
    tracker_states = [state]

    def create_tracker_state(_inference_state):
        private = {"history": np.zeros((3, 9, 13), dtype=np.float32)}
        owners.append(weakref.ref(private["history"]))
        tracker_states.append(private)
        return private

    predictor.model._init_new_sam2_state = create_tracker_state
    if tracker_error is not None:
        def fail_tracking(*args, **kwargs):
            raise tracker_error
        predictor.model.tracker.propagate_in_video = fail_tracking

    def handle_request(request):
        if request["type"] == "start_session":
            return {"session_id": "mask-test"}
        assert request["type"] == "close_session"
        if close_error is not None:
            raise close_error
        if not keep_registry:
            predictor._all_inference_states.pop("mask-test")
        if not keep_state:
            state.clear()
        if missing_registry:
            predictor._all_inference_states = None
        return {"is_success": True} if response is None else response

    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True
    result = run_mask_seed_session(
        SimpleNamespace(handle_request=handle_request), predictor,
        resource=[Image.fromarray(np.zeros((9, 13, 3), dtype=np.uint8)) for _ in range(3)],
        session=SamSessionPlan("retirement-test", 0, 0, 3), prompt_frame=0,
        ground_truth=seed, seed=None, object_masks=(seed,), conf=0.,
        propagation_mode="tracker-only", retain_predictions=False,
        raw_observation_callback=lambda _item: None, retain_raw_observations=False,
    )
    return result["session_retirement"], owners, tracker_states


def test_pinned_close_proves_session_and_history_owners_retired_without_global_gc(monkeypatch):
    import torch  # Load the runtime before checking the task's cleanup behavior.
    def forbidden_gc(*args, **kwargs):
        raise AssertionError("A task must not run global garbage collection")
    monkeypatch.setattr("gc.collect", forbidden_gc)
    proof, owners, states = run_session()
    assert proof == dict(schema="xta.sam_session_retirement/1", session_id="mask-test",
        close_call_completed=True, propagation_iterator_settled=True,
        session_registry_removed=True, inference_state_cleared=True,
        private_tracker_state_cleared=True, verified=True)
    assert all(owner() is None for owner in owners)
    assert states and all(not state for state in states)


@pytest.mark.parametrize("retained", ["registry", "state"])
def test_pinned_close_cannot_claim_retirement_with_a_retained_owner(retained):
    proof, owners, _states = run_session(keep_registry=retained == "registry",
                                       keep_state=retained == "state")
    assert proof["verified"] is False
    assert proof["session_registry_removed" if retained == "registry"
                 else "inference_state_cleared"] is False
    if retained == "state":
        assert owners[0]() is not None


def test_custom_close_protocol_is_unavailable_even_when_known_owners_are_cleared():
    proof, _owners, _states = run_session(response={"success": True})
    assert proof["session_registry_removed"] is True
    assert proof["inference_state_cleared"] is True
    assert proof["verified"] is None


def test_missing_registry_is_unavailable_but_known_close_failure_remains_false():
    proof, _owners, _states = run_session(missing_registry=True)
    assert proof["session_registry_removed"] is None and proof["verified"] is None
    proof, _owners, _states = run_session(missing_registry=True, response={"is_success": False})
    assert proof["verified"] is False


def test_cleanup_exceptions_and_primary_failure_are_preserved():
    with pytest.raises(RuntimeError, match="close failed"):
        run_session(close_error=RuntimeError("close failed"))
    primary = RuntimeError("tracker failed")
    with pytest.raises(RuntimeError, match="tracker failed") as caught:
        run_session(tracker_error=primary, close_error=RuntimeError("close failed"))
    assert caught.value is primary
    assert any("close_session" in note and "close failed" in note for note in primary.__notes__)


def test_exhausted_custom_iterator_without_close_has_unavailable_retirement_proof():
    predictor = _Predictor(_Tracker())
    state = predictor._all_inference_states["mask-test"]["state"]
    private = {}
    predictor.model._get_sam2_inference_states_by_obj_ids = lambda *_: [private]
    predictor.model._cache_frame_outputs = lambda *_: None
    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True
    def handle_request(request):
        if request["type"] == "start_session":
            return {"session_id": "mask-test"}
        if request["type"] == "add_prompt":
            return {"outputs": {}}
        assert request["type"] == "close_session"
        predictor._all_inference_states.pop("mask-test")
        state.clear()
        return {"is_success": True}
    stream = iter([dict(frame_index=frame, outputs=dict(
        out_obj_ids=[0], out_probs=[1.], out_binary_masks=[seed.copy()])) for frame in range(3)])
    assert not hasattr(stream, "close")
    measured = SimpleNamespace(handle_request=handle_request, handle_stream_request=lambda _request: stream)
    result = run_mask_seed_session(measured, predictor,
        resource=[Image.fromarray(np.zeros((9, 13, 3), dtype=np.uint8)) for _ in range(3)],
        session=SamSessionPlan("custom-iterator", 0, 0, 3), prompt_frame=0,
        ground_truth=seed, seed=make_dataclass("Point", ("x", "y"))(.5, .5),
        object_masks=None, conf=0., propagation_mode="merged", retain_predictions=False)
    assert result["propagation_response_count"] == 3
    proof = result["session_retirement"]
    assert proof["session_registry_removed"] is True and proof["inference_state_cleared"] is True
    assert proof["propagation_iterator_settled"] is None and proof["verified"] is None


def worker_fixture(tmp_path, *, retain_resource=False, close_response=None):
    from XTA import sam_tracker_runtime as runtime
    images = np.arange(3 * 15 * 21, dtype=np.uint16).reshape(3, 15, 21).astype(np.uint8)
    cache = runtime.materialize_interpolation_image_cache(
        images, path=tmp_path / "images.bin", physical_view_id="transverse",
        source_identity="retirement-source")
    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True
    seed_path = tmp_path / "seed.npz"
    runtime._atomic_npz(seed_path, seed=seed)
    predictor = _Predictor(_Tracker())
    predictor.model._xta_prepared_video_loader = True
    predictor.model.image_size = 16
    predictor.model.image_mean = predictor.model.image_std = (.5, .5, .5)
    retained = []

    def handle_request(request):
        if request["type"] == "start_session":
            if retain_resource:
                retained.append(request["resource_path"])
            return {"session_id": "mask-test"}
        assert request["type"] == "close_session"
        state = predictor._all_inference_states.pop("mask-test")["state"]
        state.clear()
        return {"is_success": True} if close_response is None else close_response

    predictor.handle_request = handle_request
    context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={})
    payload = dict(run_id="owned-run", work_id="owned-run", attempt_token="owned-attempt",
        image_cache=cache.payload(), seed_path=str(seed_path), seed_sha256=runtime._sha256(seed_path),
        crop_xyxy=[4, 2, 17, 11], frame_start=0, frame_stop=3, seed_frame=0,
        direction="forward", output_dir=str(tmp_path / "output"))
    return context, payload, retained


def test_worker_seals_bound_proof_only_after_the_real_heavy_frame_exits(tmp_path, monkeypatch):
    import torch
    from XTA import sam_tracker_runtime as runtime
    context, payload, _retained = worker_fixture(tmp_path)
    owners = []
    def observe_owner(value):
        reference = weakref.ref(value)
        owners.append(reference)
        return reference
    monkeypatch.setattr(runtime, "weakref", SimpleNamespace(ref=observe_owner))
    monkeypatch.setenv("LTA_EXECUTION_DEVICE_ID", "7")
    monkeypatch.setenv("LTA_WORKER_INDEX", "1")
    def forbidden_gc(*args, **kwargs):
        raise AssertionError("The worker wrapper must not run global GC")
    monkeypatch.setattr("gc.collect", forbidden_gc)
    output = runtime.execute_interpolation_tracker_task(context, "sam_interpolation_run", payload)
    manifest = json.loads(Path(output["artifact_path"]).read_text(encoding="utf-8"))
    proof = manifest["cpu_task_retirement"]
    assert proof["verified"] is True and proof["known_owners_retired"] is True
    assert proof["known_owner_count"] == len(owners) >= 12
    assert all(owner() is None for owner in owners)
    assert proof["session_retirement"]["verified"] is True
    assert proof["run_id"] == proof["work_id"] == "owned-run"
    assert proof["attempt_token"] == "owned-attempt"
    assert proof["worker_pid"] == os.getpid()
    assert proof["execution_device_id"] == 7 and proof["worker_index"] == 1
    prepared = SimpleNamespace(task=SimpleNamespace(payload=payload,
        work_id=payload["work_id"], attempt_token=payload["attempt_token"]))
    event = SimpleNamespace(worker_pid=os.getpid())
    assert runtime._cpu_completion_retired(manifest, prepared, (7, 1), event) is True
    for key, value in (("attempt_token", "another-attempt"), ("worker_pid", os.getpid()+1),
                       ("execution_device_id", 6), ("worker_index", 0)):
        altered = copy.deepcopy(manifest)
        altered["cpu_task_retirement"][key] = value
        with pytest.raises(RuntimeError, match="bound task/worker proof"):
            runtime._cpu_completion_retired(altered, prepared, (7, 1), event)
    altered = copy.deepcopy(manifest)
    altered["cpu_task_retirement"]["session_retirement"]["session_registry_removed"] = False
    with pytest.raises(RuntimeError, match="bound task/worker proof"):
        runtime._cpu_completion_retired(altered, prepared, (7, 1), event)
    altered = copy.deepcopy(manifest)
    altered["cpu_task_retirement"]["verified"] = None
    assert runtime._cpu_completion_retired(altered, prepared, (7, 1), event) is False
    altered["cpu_task_retirement"]["known_owners_retired"] = False
    with pytest.raises(RuntimeError, match="retained CPU owners"):
        runtime._cpu_completion_retired(altered, prepared, (7, 1), event)
    for key, value, slot, process in (
        ("execution_device_id", False, (0, 1), event),
        ("worker_index", True, (7, 1), event),
        ("worker_pid", True, (7, 1), SimpleNamespace(worker_pid=1)),
    ):
        altered = copy.deepcopy(manifest)
        altered["cpu_task_retirement"][key] = value
        with pytest.raises(RuntimeError, match="bound task/worker proof"):
            runtime._cpu_completion_retired(altered, prepared, slot, process)
    for missing in (False, True):
        altered = copy.deepcopy(manifest)
        session = altered["cpu_task_retirement"]["session_retirement"]
        if missing:
            session.pop("session_id")
        else:
            session["session_id"] = ""
        with pytest.raises(RuntimeError, match="bound task/worker proof"):
            runtime._cpu_completion_retired(altered, prepared, (7, 1), event)


def test_worker_refuses_to_publish_if_predictor_retains_native_cpu_input(tmp_path):
    from XTA import sam_tracker_runtime as runtime
    context, payload, retained = worker_fixture(tmp_path, retain_resource=True)
    with pytest.raises(RuntimeError, match="retained CPU input/session owners"):
        runtime.execute_interpolation_tracker_task(context, "sam_interpolation_run", payload)
    assert retained and retained[0]
    assert not (tmp_path / "output" / "manifest.json").exists()


@pytest.mark.parametrize("response, expected", [({"success": True}, None), ({"is_success": False}, False)])
def test_worker_distinguishes_unavailable_protocol_from_known_failed_retirement(tmp_path, response, expected):
    from XTA import sam_tracker_runtime as runtime
    context, payload, _retained = worker_fixture(tmp_path, close_response=response)
    if expected is False:
        with pytest.raises(RuntimeError, match="retained CPU input/session owners"):
            runtime.execute_interpolation_tracker_task(context, "sam_interpolation_run", payload)
        assert not (tmp_path / "output" / "manifest.json").exists()
    else:
        output = runtime.execute_interpolation_tracker_task(context, "sam_interpolation_run", payload)
        manifest = json.loads(Path(output["artifact_path"]).read_text(encoding="utf-8"))
        assert manifest["cpu_task_retirement"]["verified"] is None
