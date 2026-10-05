"""ABBA qualification of SAM tiled scheduling, with unchanged production geometry.

The legacy override changes only child ordering inside the current bounded
parent cohorts. It does not change seeds, crops, masks, quality policy, or stale
plan checks. GPU execution and heat soaking belong to the operator holding the
shared reservation; this tool reuses the qualified integration seam's lock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

TRIALS = (("A1", "legacy", 1), ("B1", "crop_local", 1),
          ("B2", "crop_local", 2), ("A2", "legacy", 2))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def legacy_parent_batches(prepared, current_batches, worker_count=1):
    """Reproduce the old single-worker order without changing cohort membership."""
    if int(worker_count) != 1:
        raise ValueError("This diagnostic legacy schedule is defined for exactly one worker")
    if prepared.crop_mode != "tiled":
        return current_batches
    output = []
    for batch in current_batches:
        ordered = tuple(sorted(batch, key=lambda index: (
            prepared.tracker_jobs[index].original_run_index, index)))
        if sorted(ordered) != sorted(batch):
            raise AssertionError("Diagnostic schedule changed its bounded child inventory")
        output.append(ordered)
    return tuple(output)


def _seed_sha(seed):
    mask = np.asarray(seed)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Schedule tracing requires the unchanged Boolean original seed")
    digest = hashlib.sha256(json.dumps(list(mask.shape)).encode())
    digest.update(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


@contextmanager
def diagnostic_schedule(schedule, trace):
    """Process-local diagnostic overrides; restore every seam after the trial."""
    from XTA import sam_interpolation
    from XTA.sam_tracker_runtime import SamInterpolationTracker
    from XTA.outputs import NrrdLayerSink

    if schedule not in {"legacy", "crop_local"}:
        raise ValueError("Unknown schedule")
    plan_class = sam_interpolation.SamPreparedInterpolationPass
    original_batches = plan_class.execution_batches
    original_iter = SamInterpolationTracker.iter_results
    original_publish = sam_interpolation._publish_directions
    original_wait = NrrdLayerSink.wait

    def batches(self, worker_count=1):
        current = original_batches(self, worker_count)
        actual = legacy_parent_batches(self, current, worker_count) if schedule == "legacy" else current
        signature = canonical_sha(dict(snapshot=self.observation_snapshot_sha256,
            settings=self.settings_sha256, tiling=self.tiling_sha256,
            order=[list(batch) for batch in actual]))
        if signature not in trace["batch_plans"]:
            trace["batch_plans"][signature] = dict(
                observation_snapshot=self.observation_snapshot_sha256,
                settings_sha256=self.settings_sha256, tiling_sha256=self.tiling_sha256,
                actual_batches=[list(batch) for batch in actual],
                diagnostic_override=schedule == "legacy")
        return actual

    def traced_iter(self, requests, **kwargs):
        def observed_requests():
            for request in requests:
                trace["actual_child_request_order"].append(dict(
                    run_id=str(request["run_id"]), crop_xyxy=list(request["crop_xyxy"]),
                    seed_frame=int(request["seed_frame"]), frame_start=int(request["frame_start"]),
                    frame_stop=int(request["frame_stop"]), direction=str(request["direction"]),
                    native_seed_sha256=_seed_sha(request["seed_mask"]),
                    metadata=dict(request.get("metadata") or {})))
                yield request
        yield from original_iter(self, observed_requests(), **kwargs)

    def timed_publish(*args, **kwargs):
        began = time.perf_counter()
        try:
            return original_publish(*args, **kwargs)
        finally:
            trace["directional_publication_seconds"] += time.perf_counter() - began

    def timed_wait(self, *args, **kwargs):
        began = time.perf_counter()
        try:
            return original_wait(self, *args, **kwargs)
        finally:
            trace["nrrd_wait_seconds"] += time.perf_counter() - began

    plan_class.execution_batches = batches
    SamInterpolationTracker.iter_results = traced_iter
    sam_interpolation._publish_directions = timed_publish
    NrrdLayerSink.wait = timed_wait
    try:
        yield
    finally:
        plan_class.execution_batches = original_batches
        SamInterpolationTracker.iter_results = original_iter
        sam_interpolation._publish_directions = original_publish
        NrrdLayerSink.wait = original_wait


def evidence_snapshot(path):
    """Validate every mask, then compare content rather than volatile file offsets."""
    from XTA.sam_evidence import SamEvidenceBundle

    bundle = SamEvidenceBundle.open(path)
    if not bundle.manifest.get('complete'):
        raise ValueError("Scheduling comparison requires complete generated evidence")
    masks = {}
    for key in sorted(bundle.records):
        mask = bundle.mask(key)
        masks[key] = dict(shape=list(mask.shape), foreground=int(mask.sum()),
            decoded_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
            packed_sha256=str(bundle.records[key]["sha256"]))
    parents, children = {}, {}
    counters = dict(child_runs=0, frames=0, encoder_preparations=0, cache_hits=0,
                    cache_misses=0, cache_evictions=0, tracker_seconds=0.,
                    render_seconds=0., pack_seconds=0., queue_seconds=0.)
    for identity, run in sorted(bundle.runs.items()):
        parents[str(identity)] = {key: _plain(run.get(key)) for key in (
            "group_id", "seed_ids", "held_out_ids", "expected_frames", "observed_frames", "direction",
            "generation_mode", "complete", "structurally_valid", "tracker_scores", "observation_status")}
        for tile in run.get("tile_evidence", ()):
            tile_id = str(tile["tile_id"])
            children[f"{identity}/{tile_id}"] = {key: _plain(tile.get(key)) for key in (
                "crop_bbox_yx", "ownership_bbox_yx", "seed_foreground", "expected_frames", "observed_frames",
                "attempted", "status", "structurally_valid", "tracker_scores", "observation_status")}
            receipt = tile.get("runtime_receipt") or {}
            if not receipt:
                continue
            counters["child_runs"] += 1
            counters["frames"] += len(receipt["expected_frames"])
            feature = receipt["adapter_receipt"]["tracker_feature_preparation"]
            counters["encoder_preparations"] += int(feature.get("feature_only_preparations", 0))
            before, after = receipt.get("feature_cache_before") or {}, receipt.get("feature_cache_after") or {}
            for name, field in (("cache_hits", "hits"), ("cache_misses", "misses"), ("cache_evictions", "evictions")):
                counters[name] += int(after.get(field, 0)) - int(before.get(field, 0))
            for name in ("tracker_seconds", "render_seconds", "pack_seconds", "queue_seconds"):
                counters[name] += float(receipt.get("timings", {}).get(name, 0.))
    content = dict(masks=masks, parents=parents, children=children)
    return dict(content=content, content_sha256=canonical_sha(content), counters=counters)


def _plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def collect_trial(runroot):
    report = json.loads((runroot / "qualification.json").read_text())
    if (report.get("status") != "passed" or report.get("unsettled_sam_workers")
            or report.get('context_closed') is not True or report.get('gpu_leases_released') is not True):
        raise ValueError("Scheduling trial did not settle the qualified production seam")
    scopes = [("parent", report["parent_stats"])]
    scopes.extend((f"consolidated_{index:02d}", value)
                  for index, value in enumerate(report.get("consolidated_stats", ())))
    snapshots, timings = {}, {}
    for name, stats in scopes:
        if not stats.get("sam_evidence_path"):
            if (name != 'parent' and type(stats.get('planner_plan_count')) is int
                    and stats['planner_plan_count'] == 0):
                continue  # Explicitly empty consolidated planning scope.
            raise ValueError(f"Scheduling comparison is missing generated evidence for {name}")
        snapshots[name] = evidence_snapshot(stats["sam_evidence_path"])
        timings[name] = {key: stats.get(key) for key in (
            "planner_wall_seconds", "observation_snapshot_wall_seconds", "sam_tracker_wall_seconds",
            "sam_policy_wall_seconds", "generator_wall_seconds")}
    added = np.load(runroot / "online_selected_additions.npy", allow_pickle=False)
    if added.ndim != 3 or not np.isin(added, (0, 1)).all():
        raise ValueError("Scheduling comparison requires binary selected parent additions")
    return dict(scopes=snapshots, selected_parent_additions=dict(shape=list(added.shape),
        foreground=int(added.sum()), decoded_sha256=hashlib.sha256(np.ascontiguousarray(added).tobytes()).hexdigest()),
        startup_seconds=float(report["context_runtime"]["start_seconds"]),
        seam_wall_seconds=float(report["wall_seconds"]), generator_phase_timings=timings,
        operational_default_schedule_label=report["parent_stats"].get("sam_execution_schedule"),
        parent_reported_execution_order=report["parent_stats"].get("sam_execution_order"))


def compare_trials(trials):
    if (len(trials) != 4 or [(trial.get('slot'), trial.get('schedule'), trial.get('repeat'))
            for trial in trials] != list(TRIALS)
            or any(type(trial.get('repeat')) is not int for trial in trials)):
        raise ValueError("Qualification requires the complete A1 B1 B2 A2 matrix")
    reference = trials[0]["result"]
    for trial in trials:
        scopes = trial['result'].get('scopes', {})
        if not scopes or 'parent' not in scopes:
            raise ValueError("Scheduling comparison requires the generated parent evidence scope")
        for snapshot in scopes.values():
            content = snapshot.get('content', {})
            if not all(isinstance(content.get(key), dict) and content[key]
                       for key in ('masks', 'parents', 'children')):
                raise ValueError("Scheduling comparison has incomplete generated masks/parent/child inventory")
    for trial in trials[1:]:
        candidate = trial["result"]
        if set(reference["scopes"]) != set(candidate["scopes"]):
            raise AssertionError("Schedules changed the generated evidence scopes")
        for scope in reference["scopes"]:
            if reference["scopes"][scope]["content"] != candidate["scopes"][scope]["content"]:
                raise AssertionError(f"Scheduling changed raw/owned/candidate masks, geometry, or scores in {scope}")
        if reference["selected_parent_additions"] != candidate["selected_parent_additions"]:
            raise AssertionError("Scheduling changed selected parent bridge support")
    for trial in trials:
        if not trial.get('output'):
            raise ValueError("Scheduling comparison requires persisted trial artifacts")
        current = collect_trial(Path(trial['output']))
        saved = trial['result']
        if set(current['scopes']) != set(saved['scopes']):
            raise AssertionError("Persisted trial artifacts changed the evidence scopes")
        for scope in current['scopes']:
            for key in ('content', 'content_sha256'):
                if current['scopes'][scope][key] != saved['scopes'][scope].get(key):
                    raise AssertionError(f"Persisted trial artifacts changed raw/owned/candidate masks or scores in {scope}")
        if current['selected_parent_additions'] != saved['selected_parent_additions']:
            raise AssertionError("Persisted trial artifacts changed selected parent additions")
    return dict(exact_all_child_raw_owned_candidate_availability_masks=True,
                exact_all_child_object_scores=True, exact_selected_parent_additions=True,
                compared_trial_slots=[trial["slot"] for trial in trials], artifact_coverage_verified=True,
                qualification_claim=False, accuracy_claim=False)


def validate_comparison_fixture(fixture_path, report):
    """Authenticate the saved source identities without executing a GPU seam."""
    if file_sha(fixture_path) != report.get('fixture_sha256'):
        raise ValueError("Scheduling comparison fixture identity differs")
    fixture = json.loads(Path(fixture_path).read_text())
    for path_key, hash_key, report_key in (('image_path', 'image_sha256', 'image_sha256'),
            ('parent_observations', 'parent_observations_sha256', 'observations_sha256')):
        if (file_sha(fixture[path_key]) != fixture[hash_key]
                or fixture[hash_key] != report.get(report_key)):
            raise ValueError("Scheduling comparison source image/observation identity differs")
    for trial in report['trials']:
        actual = json.loads((Path(trial['output']) / 'qualification.json').read_text())
        if (actual.get('image_sha256') != fixture['image_sha256']
                or actual.get('original_input_sha256') != fixture['parent_observations_sha256']):
            raise ValueError("Scheduling trial references a different source fixture")


def prepare_fixture_copy(fixture_path, output):
    fixture_path, output = Path(fixture_path).resolve(strict=True), Path(output).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("fixture.json", "actual_detector_0059_component.npy", "actual_detector_0064_component.npy"):
        source = fixture_path if name == "fixture.json" else fixture_path.parent / name
        destination = output / name
        if destination.exists() and file_sha(destination) != file_sha(source):
            raise ValueError("Schedule qualification output contains another fixture")
        if not destination.exists():
            shutil.copyfile(source, destination)
        if file_sha(destination) != file_sha(source):
            raise AssertionError("Qualification fixture copy changed bytes")
    fixture = json.loads(fixture_path.read_text())
    for key, expected in (("image_path", "image_sha256"), ("parent_observations", "parent_observations_sha256")):
        if file_sha(fixture[key]) != fixture[expected]:
            raise ValueError("Frozen native fixture changed")
    return fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--heatsoak-receipt", type=Path,
                        help="Required run evidence from the GPU operator; this tool does not claim to heat soak")
    parser.add_argument("--compare-only", action="store_true", help="Validate an existing ABBA matrix without GPU execution")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "schedule_qualification.json"
    if args.compare_only:
        report = json.loads(report_path.read_text())
        report.pop('exact_comparison', None)
        report.update(comparison_status='incomplete', comparison_helper_sha256=file_sha(__file__))
        try:
            comparison = compare_trials(report['trials'])
            validate_comparison_fixture(args.fixture, report)
            report.update(exact_comparison=comparison, comparison_status='passed')
            report.pop('comparison_error', None)
        except Exception as error:
            report.pop('exact_comparison', None)
            report.update(comparison_status='failed', comparison_error=str(error))
            raise
        finally:
            # Existing execution/qualification failure and its error stay intact.
            report_path.write_text(json.dumps(report, indent=2))
        return 0
    if args.model is None or args.heatsoak_receipt is None or not args.heatsoak_receipt.is_file():
        parser.error("GPU run requires --model and an existing --heatsoak-receipt from its operator")
    fixture = prepare_fixture_copy(args.fixture, args.output)
    from tools.qualify_sam_tiled_integration import seam

    report = dict(schema="xta.sam-tiled-schedule-abba/1", status="running", trials=[],
        fixture=str(args.fixture.resolve()), fixture_sha256=file_sha(args.fixture),
        image_sha256=fixture["image_sha256"], observations_sha256=fixture["parent_observations_sha256"],
        helper_sha256=file_sha(__file__), cache_mib=512, worker_count=1,
        heatsoak=dict(path=str(args.heatsoak_receipt.resolve()), sha256=file_sha(args.heatsoak_receipt)),
        protocol="ABBA fresh predictor/worker each trial; identical native pixels, observations, swept crops, quality and512MiBcache",
        legacy_override="Child order only: stable parent-contiguous sorting inside CURRENT bounded16-parent cohorts",
        accuracy_claim=False)
    try:
        for slot, schedule, repeat in TRIALS:
            label = f"schedule_{slot}_{schedule}"
            runroot = args.output / label
            if runroot.exists():
                raise FileExistsError(f"Trial directory already exists: {runroot}")
            trace = dict(batch_plans={}, actual_child_request_order=[],
                         directional_publication_seconds=0., nrrd_wait_seconds=0.)
            began = time.perf_counter()
            with diagnostic_schedule(schedule, trace):
                seam(SimpleNamespace(output=args.output, model=args.model, crop_mode="tiled",
                                     policy="raw", run_label=label))
            result = collect_trial(runroot)
            trial = dict(slot=slot, schedule=schedule, repeat=repeat, output=str(runroot),
                diagnostic_override=schedule == "legacy", actual_schedule=(
                    "diagnostic_legacy_parent_contiguous_within_current_bounded_cohorts" if schedule == "legacy"
                    else "production_single_worker_crop_local_within_bounded_cohorts"),
                actual_child_order=trace["actual_child_request_order"], batch_plans=trace["batch_plans"],
                output_wall_seconds=dict(directional_publication=trace["directional_publication_seconds"],
                                         nrrd_wait=trace["nrrd_wait_seconds"]),
                trial_with_fingerprint_wall_seconds=time.perf_counter()-began, result=result)
            report["trials"].append(trial)
            report_path.write_text(json.dumps(report, indent=2))
        report["exact_comparison"] = compare_trials(report["trials"])
        report['comparison_status'] = 'passed'
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report["status"], report=str(report_path),
        parent_counters={trial["slot"]: trial["result"]["scopes"]["parent"]["counters"] for trial in report["trials"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
