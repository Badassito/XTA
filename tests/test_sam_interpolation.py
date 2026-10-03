"""Generator transaction, immutable anchors, and multi-owner output contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_interpolation import (SamInterpolationInfrastructureError,
                                   interpolate_sam_view_volume_pass,
                                   selected_sam_plane)


class RepeatedSeedTracker:
    def __init__(self, *, missing_terminal=False, nonbinary=False, failure=False):
        self.calls = []
        self.released = []
        self.missing_terminal = missing_terminal
        self.nonbinary = nonbinary
        self.failure = failure

    def run(self, **request):
        self.calls.append(request)
        if self.failure:
            raise RuntimeError("controlled worker failure")
        frames = range(request["seed_frame"], request["frame_stop"]) if request["direction"] == "forward" else range(request["seed_frame"], request["frame_start"] - 1, -1)
        masks = {frame: request["seed_mask"].copy() for frame in frames}
        if self.missing_terminal:
            masks.pop(list(masks)[-1])
        if self.nonbinary:
            for mask in masks.values():
                mask = mask.astype(np.uint8)
                mask[mask != 0] = 2
                masks[list(masks)[0]] = mask
                break
        return SimpleNamespace(frames=masks, tracker_scores={frame: .8 for frame in masks},
                               observation_status={}, receipt={"prediction_valid": True})

    def release_result(self, result):
        self.released.append(result)


def _observations():
    result = np.zeros((5, 25, 29), dtype=np.uint8)
    result[0, 9:15, 10:16] = 1
    result[4, 9:15, 10:16] = 1
    return result


def _generate(tmp_path, tracker, **overrides):
    options = dict(work_dir=tmp_path, runtime=tracker, gap_distance=5, min_radius=0,
                   interpolation_walk_back=0, return_bridge_components=True)
    options.update(overrides)
    return interpolate_sam_view_volume_pass(_observations(), **options)


def _close(array):
    if isinstance(array, np.memmap):
        array._mmap.close()


def test_disabled_sam_does_not_require_model_geometry_or_images(tmp_path):
    source = _observations()
    result, stats, components = interpolate_sam_view_volume_pass(
        source, work_dir=tmp_path, gap_distance=0, wrap_axis=True,
        view="unsupported", runtime=None)
    assert result is source
    assert stats["inactive_configured_backend"] == "sam"
    assert components == []
    assert list(tmp_path.iterdir()) == []


def test_sam_generates_selected_additions_without_mutating_detector_anchors(tmp_path):
    source = _observations()
    before = source.copy()
    tracker = RepeatedSeedTracker()
    merged, stats, components = interpolate_sam_view_volume_pass(
        source, work_dir=tmp_path, runtime=tracker, gap_distance=5, min_radius=0,
        interpolation_walk_back=0, return_bridge_components=True)
    try:
        np.testing.assert_array_equal(source, before)
        assert int(np.count_nonzero(merged)) == 5 * 36
        assert stats["added_voxels"] == 3 * 36
        assert stats["sam_selected_runs"] == 2
        assert {component["direction"] for component in components} == {"forward", "backward"}
        assert [component["voxel_count"] for component in components] == [108, 108]
        assert len(tracker.released) == 2
        assert stats['sam_selection_identity'] == stats['sam_selection_receipt']['selection_identity']
        for component in components:
            assert component['sam_selection_identity'] == stats['sam_selection_identity']
            assert component['metadata']['sam_selection_identity'] == stats['sam_selection_identity']
            assert component['metadata']['sam_selection_resources'] == stats['sam_selection_resources']
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        assert len(bundle.runs) == 2
        for run in bundle.runs.values():
            assert run["seed_ids"]
            assert run["held_out_ids"]
            assert len(run["observed_frames"]) == 5
    finally:
        _close(merged)


def test_quality_rejected_raw_tracks_never_enter_directional_outputs(tmp_path):
    def reject(context):
        assert len(context["runs"]) == 2
        return {"selected_run_ids": [], "reasons": {"controlled": "reject all"}}

    policy = {"proposal_api_version": 1, "select_proposals": reject}
    merged, stats, components = _generate(tmp_path, RepeatedSeedTracker(), policy=policy)
    try:
        np.testing.assert_array_equal(merged, _observations())
        assert stats["sam_generated_runs"] == 2
        assert stats["sam_selected_runs"] == 0
        assert stats["added_voxels"] == 0
        assert len(components) == 2
        assert all(component["voxel_count"] == 0 for component in components)
        assert Path(stats["sam_evidence_path"], "masks.bin").stat().st_size > 0
    finally:
        _close(merged)


def test_selected_plane_rebuild_preserves_shared_same_direction_owner():
    a = np.zeros((4, 5), bool)
    b = a.copy()
    a[1:3, 1:3] = True
    b[1:3, 2:4] = True
    bundle = SimpleNamespace(
        groups={"g": {"context_bbox_yx": (2, 3, 6, 8)}},
        runs={identifier: {"group_id": "g", "direction": "forward", "expected_frames": [1]}
              for identifier in ("a", "b")},
        candidate_mask=lambda identifier, frame: {"a": a, "b": b}[identifier],
        raw_mask=lambda identifier, frame: {"a": a, "b": b}[identifier],
    )
    plane = selected_sam_plane(bundle, {"selected_run_ids": ["b"]}, 1, (9, 11), direction=1)
    np.testing.assert_array_equal(plane[2:6, 3:8], b)
    assert plane[3, 5]  # Shared support remains justified by b after a rejection.
    assert not plane[3, 4]


def test_worker_failure_retains_explicit_incomplete_evidence_without_selected_support(tmp_path):
    tracker = RepeatedSeedTracker(failure=True)
    with pytest.raises(SamInterpolationInfrastructureError, match="controlled worker failure"):
        _generate(tmp_path, tracker)
    evidence = list(tmp_path.glob("sam_*/evidence/manifest.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text())["complete"] is False
    assert list(tmp_path.glob("sam_*/sam_bridge_*.cvol")) == []


def test_missing_terminal_cannot_be_selected_by_permissive_policy(tmp_path):
    merged, stats, components = _generate(
        tmp_path, RepeatedSeedTracker(missing_terminal=True),
        policy={"sam_bridge_policy": "permissive"})
    try:
        assert stats["sam_incomplete_runs"] == 2
        assert stats["sam_selected_runs"] == 0
        assert all(component["voxel_count"] == 0 for component in components)
    finally:
        _close(merged)


def test_malformed_raw_transfer_is_not_coerced_into_valid_binary_support(tmp_path):
    tracker = RepeatedSeedTracker(nonbinary=True)
    with pytest.raises(SamInterpolationInfrastructureError, match="binary"):
        _generate(tmp_path, tracker)
    assert len(tracker.released) == 1
    assert list(tmp_path.glob("sam_*/sam_bridge_*.cvol")) == []


def test_additional_pass_exhaustion_does_not_repeat_tracker_or_seed_generated_masks(tmp_path):
    tracker = RepeatedSeedTracker()
    result, stats, components = _generate(tmp_path, tracker, pass_index=2, interpolation_passes=3)
    np.testing.assert_array_equal(result, _observations())
    assert tracker.calls == []
    assert stats["requested_passes"] == 3
    assert stats["completed_passes"] == 1
    assert stats["skipped_passes"] == 2
    assert stats["skip_reason"] == "observed_anchor_hypotheses_exhausted"
    assert components == []


def test_no_candidate_first_round_publishes_two_empty_slots_and_portable_evidence(tmp_path):
    observations = np.zeros((3, 11, 13), bool)
    merged, stats, components = interpolate_sam_view_volume_pass(
        observations, work_dir=tmp_path, runtime=None, gap_distance=4,
        return_bridge_components=True)
    try:
        assert stats["skip_reason"] == "no_missing_connections"
        assert len(components) == 2
        assert all(component["voxel_count"] == 0 for component in components)
        assert SamEvidenceBundle.open(stats["sam_evidence_path"]).manifest["complete"]
    finally:
        _close(merged)


def test_unsupported_active_view_fails_before_tracker(tmp_path):
    tracker = RepeatedSeedTracker()
    with pytest.raises(ValueError, match="Unsupported SAM TTA view family"):
        _generate(tmp_path, tracker, view=SimpleNamespace(family="unknown", summary_family="coronal"))
    assert tracker.calls == []


def test_capped_family_records_unresolved_bounds_without_tracking_or_large_mask_contracts(tmp_path):
    from XTA.sam_bridge_planning import SamPlanningLimits
    tracker = RepeatedSeedTracker()
    merged, stats, components = _generate(
        tmp_path, tracker, planner_limits=SamPlanningLimits(max_crop_pixels=4))
    try:
        assert tracker.calls == []
        assert stats["skip_reason"] == "resource_limit_unresolved"
        assert stats["sam_unresolved_groups"] > 0
        assert len(components) == 2
        assert all(component["voxel_count"] == 0 for component in components)
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        assert all(not group["complete"] for group in bundle.groups.values())
        assert all("context_crop_pixel_limit" in group["reasons"] for group in bundle.groups.values())
        assert all(any(name.startswith("endpoint_local:") for name in group["mask_keys"])
                   for group in bundle.groups.values())
    finally:
        _close(merged)


def test_cancellation_keeps_completed_raw_run_but_never_publishes_partial_support(tmp_path):
    import threading
    cancelled = threading.Event()

    class CancelAfterRun(RepeatedSeedTracker):
        def run(self, **request):
            result = super().run(**request)
            cancelled.set()
            return result

    tracker = CancelAfterRun()
    with pytest.raises(SamInterpolationInfrastructureError, match="cancelled"):
        _generate(tmp_path, tracker, cancel_event=cancelled)
    assert len(tracker.calls) == 1
    assert len(tracker.released) == 1
    manifest_path = next(tmp_path.glob("sam_*/evidence/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert not manifest["complete"]
    assert manifest["run_count"] == 1
    assert list(tmp_path.glob("sam_*/sam_bridge_*.cvol")) == []


def test_invalid_policy_fails_with_complete_raw_evidence_and_no_selected_outputs(tmp_path):
    def bad_policy(context):
        return ["unknown-run"]

    tracker = RepeatedSeedTracker()
    with pytest.raises(SamInterpolationInfrastructureError, match="outside"):
        _generate(tmp_path, tracker,
                  policy={"proposal_api_version": 1, "select_proposals": bad_policy})
    manifest_path = next(tmp_path.glob("sam_*/evidence/manifest.json"))
    assert json.loads(manifest_path.read_text())["complete"]
    failure_path = next(tmp_path.glob("sam_*/failure.json"))
    assert json.loads(failure_path.read_text())["phase"] == "selection_or_publication"
    assert list(tmp_path.glob("sam_*/sam_bridge_*.cvol")) == []


def test_min_radius_filters_generated_sam_cross_sections_and_zero_disables_it(tmp_path):
    narrow, rejected, components = _generate(tmp_path / "radius3", RepeatedSeedTracker(), min_radius=3)
    try:
        assert rejected["sam_generated_runs"] == 2
        assert rejected["sam_selected_runs"] == 0
        # Filtering can remove every raw component and subsequently fail endpoint
        # agreement. It does not veto an entire otherwise valid run by width.
        assert rejected["skipped_by_min_radius"] == 0
        assert rejected["sam_radius_removed_component_observations"] == 10
        assert rejected["sam_radius_removed_raw_foreground_observations"] == 360
        assert rejected["sam_radius_removed_candidate_observations"] == 216
        assert all(component["voxel_count"] == 0 for component in components)
    finally:
        _close(narrow)
    accepted, selected, _ = _generate(tmp_path / "radius0", RepeatedSeedTracker(), min_radius=0)
    try:
        assert selected["sam_selected_runs"] == 2
        assert selected["skipped_by_min_radius"] == 0
        assert selected["sam_radius_removed_component_observations"] == 0
        assert selected["added_voxels"] > 0
    finally:
        _close(accepted)


def test_structurally_invalid_expected_object_retains_diagnostics_and_fails_generation(tmp_path):
    class InvalidObjectTracker(RepeatedSeedTracker):
        def run(self, **request):
            result = super().run(**request)
            result.receipt["prediction_valid"] = False
            return result

    tracker = InvalidObjectTracker()
    with pytest.raises(SamInterpolationInfrastructureError, match="structurally invalid"):
        _generate(tmp_path, tracker)
    manifest_path = next(tmp_path.glob("sam_*/evidence/manifest.json"))
    assert not json.loads(manifest_path.read_text())["complete"]
    assert len(tracker.released) == 1
    assert list(tmp_path.glob("sam_*/sam_bridge_*.cvol")) == []


def test_tiny_components_are_removed_before_write_and_do_not_veto_large_valid_bridge(tmp_path):
    class DotTracker(RepeatedSeedTracker):
        def run(self, **request):
            result = super().run(**request)
            result.frames[2][0, 0] = True  # Outside the immutable acceptance crop support.
            result.frames[2][18, 19] = True  # Inside write/acceptance, separate from bridge.
            return result

    tracker = DotTracker()
    merged, stats, components = _generate(tmp_path, tracker, min_radius=1)
    try:
        assert stats["sam_selected_runs"] == 2
        assert stats["added_voxels"] == 108
        assert stats["sam_radius_removed_component_observations"] == 4
        assert stats["sam_radius_removed_candidate_observations"] == 2
        assert not merged[2, 0, 0]
        assert not merged[2, 18, 19]
        assert all(component["voxel_count"] == 108 for component in components)
        bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
        for run_id in bundle.runs:
            assert bundle.raw_mask(run_id, 2)[0, 0]
            assert bundle.raw_mask(run_id, 2)[18, 19]
            assert bundle.candidate_mask(run_id, 2)[18, 19]
    finally:
        _close(merged)
