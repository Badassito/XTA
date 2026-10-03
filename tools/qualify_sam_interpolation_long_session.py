"""Real-SDK coverage check on a synthetic repetition of one retained real frame.

This is a tracker/session qualification, not an interpolation accuracy test or
a production resource-admission benchmark. Preparation imports no model runtime.
The executing helper owns GPU_LOCK; do not wrap it in another GPU reservation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SCHEMA = "xta.sam_long_session_qualification/1"
FIXTURE_SCHEMA = "xta.sam_static_repeated_real_frame/1"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def session_contract(frame_count):
    """Prove the TTA subtype accepts this interval and LTA keeps its own cap."""
    from XTA.lta_sam import (LTA_SESSION_FRAMES, SamInterpolationSessionPlan,
                             SamSessionPlan, plan_sam_sessions)
    if LTA_SESSION_FRAMES != 30:
        raise AssertionError("LTA's existing thirty-frame partition changed")
    try:
        SamSessionPlan("ordinary_lta_guard", 0, 0, 31)
    except ValueError as error:
        if "at most 30 frames" not in str(error):
            raise
    else:
        raise AssertionError("Ordinary LTA unexpectedly accepted a 31-frame session")
    admitted = SamInterpolationSessionPlan("tta_long_session", 0, 0, frame_count)
    partition = plan_sam_sessions("ordinary_lta_guard", frame_count)
    if (admitted.frame_count != frame_count or
            [frame for part in partition for frame in part.frame_indices] != list(range(frame_count)) or
            any(part.frame_count > 30 for part in partition)):
        raise AssertionError("Session coverage or ordinary LTA partition is invalid")
    return dict(tta_session_class=type(admitted).__name__, tta_frame_count=frame_count,
                ordinary_lta_rejects_31=True, ordinary_lta_cap=30,
                ordinary_lta_partitions=[[part.frame_start, part.frame_stop] for part in partition])


def input_descriptor(args):
    if args.fixture is not None:
        if any(value is not None for value in (args.images, args.shape, args.seed_mask)):
            raise ValueError("Choose --fixture or explicit --images/--shape/--seed-mask")
        fixture_path = args.fixture.resolve(strict=True)
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        parent = fixture_path.parent
        def locate(value):
            path = Path(value)
            return (path if path.is_absolute() else parent / path).resolve(strict=True)
        return dict(images=locate(fixture["image_path"]), shape=tuple(fixture["shape_tyx"]),
                    seed=locate(fixture["original_seed_file"]),
                    fixture_path=str(fixture_path), fixture_sha256=file_sha(fixture_path),
                    declared_image_sha256=fixture.get("image_sha256"),
                    native_source_frames=fixture.get("native_source_frames"),
                    original_seed_id=fixture.get("original_seed_id"))
    if args.images is None or args.shape is None or args.seed_mask is None:
        raise ValueError("Supply --fixture or all of --images, --shape, and --seed-mask")
    return dict(images=args.images.resolve(strict=True), shape=tuple(args.shape),
                seed=args.seed_mask.resolve(strict=True), fixture_path=None,
                fixture_sha256=None, declared_image_sha256=None,
                native_source_frames=None, original_seed_id=None)


def prepare(args):
    from scipy import ndimage
    if args.frames < 33:
        raise ValueError("This qualification requires at least 33 requested frames")
    if args.crop_margin < 0 or args.max_fixture_bytes <= 0:
        raise ValueError("Crop margin must be nonnegative and fixture byte budget positive")
    contract = session_contract(args.frames)
    source = input_descriptor(args)
    shape = tuple(map(int, source["shape"]))
    if len(shape) != 3 or min(shape) < 1 or not 0 <= args.seed_source_frame < shape[0]:
        raise ValueError("Real-image TYX shape or source seed frame is invalid")
    if source["images"].stat().st_size != math.prod(shape):
        raise ValueError("Real-image uint8 file size differs from its declared TYX shape")
    source_image_sha = file_sha(source["images"])
    if source["declared_image_sha256"] and source_image_sha != source["declared_image_sha256"]:
        raise ValueError("Retained real-image file differs from its fixture identity")
    saved = np.load(source["seed"], allow_pickle=False)
    if not isinstance(saved, np.ndarray) or saved.shape != shape[1:]:
        if hasattr(saved, "close"):
            saved.close()
        raise ValueError("Original seed must be a native-canvas HxW .npy array")
    if not np.all((saved == 0) | (saved == 1)):
        raise ValueError("Original seed must be binary; choose a prescribed detector component")
    original = saved != 0
    labels, count = ndimage.label(original, structure=np.ones((3, 3), bool))
    component = args.component_id
    if component is None:
        if count != 1:
            raise ValueError(f"Original seed contains {count} components; prescribe --component-id")
        component = 1
    if component < 1 or component > count:
        raise ValueError("Prescribed seed component is absent")
    selected = labels == component
    yy, xx = np.nonzero(selected)
    proposed = (int(xx.min())-args.crop_margin, int(yy.min())-args.crop_margin,
                int(xx.max())+1+args.crop_margin, int(yy.max())+1+args.crop_margin)
    x0, y0 = max(0, proposed[0]), max(0, proposed[1])
    x1, y1 = min(shape[2], proposed[2]), min(shape[1], proposed[3])
    seed = np.ascontiguousarray(selected[y0:y1, x0:x1])
    if int(seed.sum()) != int(selected.sum()):
        raise AssertionError("Fixed crop clipped the prescribed original component")
    fixture_shape = (args.frames, *seed.shape)
    if math.prod(fixture_shape) > args.max_fixture_bytes:
        raise ValueError("Synthetic fixture exceeds this diagnostic's staging byte budget")
    images = np.memmap(source["images"], dtype=np.uint8, mode="r", shape=shape)
    frame = np.ascontiguousarray(images[args.seed_source_frame, y0:y1, x0:x1])
    real_frame_sha = hashlib.sha256(images[args.seed_source_frame].tobytes()).hexdigest()
    images._mmap.close()
    args.output.mkdir(parents=True, exist_ok=True)
    image_path, seed_path = args.output/"repeated_real_frame.uint8.dat", args.output/"original_component_seed.npy"
    temporary = image_path.with_name(image_path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("wb") as stream:
            for _ in range(args.frames):
                stream.write(frame.tobytes())
        os.replace(temporary, image_path)
    finally:
        temporary.unlink(missing_ok=True)
    np.save(seed_path, seed, allow_pickle=False)
    native_frames = source["native_source_frames"]
    record = dict(schema=FIXTURE_SCHEMA, synthetic_temporal_fixture=True,
        temporal_recipe="Repeat one unchanged retained real image; both endpoint seeds are copies of its one original component",
        qualification_route="Direct persistent tracker, not production planner/resource admission or bridge selection",
        accuracy_claim=False, labels_used=False, image_interpolation_or_resize_during_preparation=False,
        source_images=str(source["images"]), source_images_sha256=source_image_sha,
        source_shape_tyx=list(shape), source_image_frame=args.seed_source_frame,
        source_native_frame=native_frames[args.seed_source_frame] if native_frames is not None else None,
        source_real_frame_sha256=real_frame_sha, source_seed=str(source["seed"]),
        source_seed_sha256=file_sha(source["seed"]), source_fixture=source["fixture_path"],
        source_fixture_sha256=source["fixture_sha256"], original_seed_id=source["original_seed_id"],
        original_component_count=int(count), prescribed_component_id=int(component),
        component_selection="sole connected component" if args.component_id is None else "explicit component ID in original 8-connected seed labeling",
        seed_connectivity=8, original_component_foreground=int(selected.sum()),
        crop_xyxy_in_source=[x0,y0,x1,y1], requested_crop_xyxy_in_source=list(proposed),
        crop_margin=args.crop_margin, diagnostic_staging_budget_bytes=args.max_fixture_bytes,
        crop_clamped_sides=dict(left=x0!=proposed[0],top=y0!=proposed[1],right=x1!=proposed[2],bottom=y1!=proposed[3]),
        image_path=str(image_path.resolve()), image_sha256=file_sha(image_path), shape_tyx=list(fixture_shape),
        seed_path=str(seed_path.resolve()), seed_sha256=file_sha(seed_path),
        cropped_seed_packed_sha256=hashlib.sha256(np.packbits(seed).tobytes()).hexdigest(),
        repeated_crop_sha256=hashlib.sha256(frame.tobytes()).hexdigest(), session_contract=contract)
    write_json(args.output/"prepared_fixture.json", record)
    return record


class GpuReservation:
    """Fail fast on another owner; retain our lock if worker exit is unproven."""
    def __init__(self, path):
        self.path = Path(path)
        self.token = uuid.uuid4().hex

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(dict(task="sam_interpolation_long_session_qualification", pid=os.getpid(),
                start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(), token=self.token), stream)

    def release(self, *, residency_released):
        if not residency_released:
            return False
        current = json.loads(self.path.read_text(encoding="utf-8"))
        if current.get("token") != self.token:
            raise RuntimeError("GPU_LOCK ownership changed during qualification")
        self.path.unlink()
        return True


def validate_result(result, *, frames, crop_shape, direction, run_id):
    expected = tuple(range(frames))
    receipt = result.receipt
    if (receipt.get("run_id") != run_id or receipt.get("direction") != direction or
            receipt.get("frame_range") != [0, frames] or receipt.get("expected_frames") != list(expected) or
            receipt.get("seed_frame") != (0 if direction == "forward" else frames-1)):
        raise AssertionError("Returned tracker run has incorrect interval, seed or direction attribution")
    if any(tuple(sorted(mapping)) != expected for mapping in
           (result.frames, result.tracker_scores, result.observation_status)):
        raise AssertionError("Tracker did not return exact full requested frame/mask/score/status coverage")
    if (receipt.get("status") != "complete" or not receipt.get("coverage_complete") or
            not receipt.get("prediction_valid") or any(v != "observed" for v in result.observation_status.values())):
        raise AssertionError("Incomplete/removed tracker observations are not complete empty evidence")
    adapter = receipt["adapter_receipt"]
    if (not adapter.get("raw_observation_complete") or adapter.get("raw_observation_callback_count") != frames or
            not adapter.get("seed_roundtrip_passed") or not adapter.get("seed_roundtrip_exact")):
        raise AssertionError("Raw SDK coverage or exact original-seed injection contract failed")
    if receipt.get("sam_model", {}).get("model_version") != "sam3.1":
        raise AssertionError("Qualification did not use the requested SAM3.1 model")
    if any(mask.dtype != np.bool_ or mask.shape != crop_shape for mask in result.frames.values()):
        raise AssertionError("Raw masks changed the fixed native crop geometry")
    if any(value is None or not math.isfinite(value) or not 0 <= value <= 1 for value in result.tracker_scores.values()):
        raise AssertionError("A raw tracker probability is missing or invalid")


def execute(args, prepared, *, runtime_factory=None, monitor_factory=None):
    from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
    if args.model is None or not args.model.exists():
        raise ValueError("Execution requires an existing --model bundle")
    if args.device < 0 or args.cache_mib < 0:
        raise ValueError("Device and cache budget must be nonnegative")
    runtime_factory = runtime_factory or SamInterpolationTracker
    if monitor_factory is None:
        from tools.diagnose_sam_interpolation import resource_monitor
        monitor_factory = resource_monitor
    shape = tuple(prepared["shape_tyx"])
    if file_sha(prepared["image_path"]) != prepared["image_sha256"] or file_sha(prepared["seed_path"]) != prepared["seed_sha256"]:
        raise RuntimeError("Prepared synthetic fixture identity changed")
    images = np.memmap(prepared["image_path"], dtype=np.uint8, mode="r", shape=shape)
    try:
        cache = materialize_interpolation_image_cache(images, path=args.output/"tracker_input.uint8.dat",
            physical_view_id="Transverse", source_identity=prepared["image_sha256"])
    finally:
        images._mmap.close()
    seed = np.load(prepared["seed_path"], allow_pickle=False)
    source_files = [Path(__file__), *(REPOSITORY/"XTA"/name for name in
        ("sam_tracker_runtime.py", "sam_resources.py", "lta_sam.py", "lta_experimental.py", "lta_worker_adapter.py",
         "lta_workers.py", "lta_tracker_features.py", "lta_feature_cache.py"))]
    source_hashes = {str(path.resolve()): file_sha(path) for path in source_files}
    report = dict(schema=SCHEMA, status="running", fixture=prepared, source_sha256=source_hashes,
        model=str(args.model.resolve()), worker_count=1, device=args.device, cache_mib=args.cache_mib,
        claim="Exact real-SDK requested-frame coverage on a synthetic static repeated-real-image fixture; no accuracy or throughput claim",
        accuracy_claim=False, production_admission_claim=False, runs=[], gpu_lock_released=False)
    destination = args.output/"qualification.json"
    reservation = GpuReservation(args.gpu_lock)
    runtime = None
    reservation.acquire()
    primary = None
    try:
        write_json(destination, report)
        with monitor_factory(args.output/"resources.json", args.device):
            runtime = runtime_factory(model_path=args.model, device_ids=(args.device,),
                artifact_root=args.output/"tracker_staging", source_cache_ref=cache,
                profile=args.profile, feature_cache_bytes=args.cache_mib*1024**2)
            start = time.perf_counter()
            runtime.start()
            report["predictor_start_seconds"] = time.perf_counter()-start
            for direction in ("forward", "backward"):
                run_id = "synthetic_long_session_"+direction
                began = time.perf_counter()
                result = runtime.run(run_id=run_id, seed_mask=seed, seed_frame=0 if direction=="forward" else shape[0]-1,
                    frame_start=0, frame_stop=shape[0], direction=direction, crop_xyxy=(0,0,shape[2],shape[1]),
                    metadata=dict(qualification_schema=SCHEMA, synthetic_temporal_fixture=True,
                        original_seed_sha256=prepared["source_seed_sha256"], original_component_id=prepared["prescribed_component_id"]))
                result_error = None
                retained = None
                try:
                    validate_result(result, frames=shape[0], crop_shape=shape[1:], direction=direction, run_id=run_id)
                    masks = np.stack([result.frames[index] for index in range(shape[0])])
                    packed = np.packbits(masks.reshape(shape[0], -1), axis=1)
                    packet = args.output/(direction+"_raw_masks.npz")
                    np.savez_compressed(packet, packed_masks=packed, shape=np.asarray(shape[1:], np.int64),
                        frame_indices=np.arange(shape[0]), tracker_scores=np.asarray([result.tracker_scores[index] for index in range(shape[0])]),
                        observation_status=np.asarray([result.observation_status[index] for index in range(shape[0])]))
                    retained = dict(run_id=run_id,direction=direction,frame_count=shape[0],
                        exact_requested_coverage=True,first_frame=0,terminal_frame=shape[0]-1,
                        mask_sha256_by_frame={str(index):hashlib.sha256(packed[index].tobytes()).hexdigest() for index in range(shape[0])},
                        tracker_scores={str(index):result.tracker_scores[index] for index in range(shape[0])},
                        raw_artifact=dict(path=str(packet.resolve()),sha256=file_sha(packet)),
                        tracker_receipt=result.receipt, transfer_staging_released=False,
                        tracker_and_transfer_wall_seconds=time.perf_counter()-began)
                    report["runs"].append(retained)
                    write_json(destination, report)
                except BaseException as error:
                    result_error = error
                    raise
                finally:
                    try:
                        runtime.release_result(result)
                        if retained is not None:
                            retained["transfer_staging_released"] = True
                    except BaseException as error:
                        if result_error is None:
                            raise
                        result_error.add_note("Transfer cleanup also failed: " + str(error))
            if {str(path.resolve()): file_sha(path) for path in source_files} != source_hashes:
                raise RuntimeError("Qualification source changed during execution")
            report["status"] = "passed"
    except BaseException as error:
        primary = error
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        try:
            if runtime is not None:
                runtime.close()
        except BaseException as error:
            report.update(status="failed", worker_cleanup_error=str(error))
            if primary is None:
                primary = error
        released = runtime is None or bool(runtime.residency_released)
        report["worker_residency_released"] = released
        try:
            report["gpu_lock_released"] = reservation.release(residency_released=released)
        except BaseException as error:
            report.update(status="failed",gpu_lock_cleanup_error=str(error))
            if primary is None:
                primary = error
        if not released:
            report.update(status="failed", gpu_lock_retained_reason="Worker exit/model residency release is unproven")
            if primary is None:
                primary = RuntimeError(report["gpu_lock_retained_reason"])
        write_json(destination, report)
    if primary is not None:
        raise primary
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fixture", type=Path, help="Retained fixture JSON containing image_path, shape_tyx and original_seed_file")
    value.add_argument("--images", type=Path)
    value.add_argument("--shape", nargs=3, type=int, metavar=("T","H","W"))
    value.add_argument("--seed-mask", type=Path)
    value.add_argument("--seed-source-frame", type=int, default=0)
    value.add_argument("--component-id", type=int, help="Prescribed original 8-connected seed component; required for multi-component inputs")
    value.add_argument("--frames", type=int, default=33)
    value.add_argument("--crop-margin", type=int, default=16)
    value.add_argument("--max-fixture-bytes", type=int, default=256*1024**2)
    value.add_argument("--model", type=Path)
    value.add_argument("--device", type=int, default=0)
    value.add_argument("--profile", default="egpu")
    value.add_argument("--cache-mib", type=int, default=512)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, default=REPOSITORY.parent/"Scratch"/"Temp"/"GPU_LOCK")
    value.add_argument("--prepare-only", action="store_true", help="Prepare synthetic fixture and CPU session checks without importing or constructing a model")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    if (args.output/"qualification.json").exists():
        raise FileExistsError("Use a new output directory; qualification evidence is not overwritten")
    try:
        prepared = prepare(args)
        if args.prepare_only:
            print(json.dumps(dict(status="prepared_only",prepared_fixture=str(args.output/"prepared_fixture.json"),
                frame_count=args.frames,model_loaded=False,gpu_lock_acquired=False),indent=2))
            return 0
        report = execute(args, prepared)
        print(json.dumps(dict(status=report["status"],frame_count=args.frames,runs=len(report["runs"]),
            gpu_lock_released=report["gpu_lock_released"]),indent=2))
        return 0
    except Exception as error:
        if not (args.output/"qualification.json").exists():
            write_json(args.output/"qualification.json", dict(schema=SCHEMA,status="failed",
                stage="preflight_or_preparation",error_type=type(error).__name__,error=str(error),
                accuracy_claim=False,model_loaded=False))
        print(f"Qualification failed: {type(error).__name__}: {error}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
