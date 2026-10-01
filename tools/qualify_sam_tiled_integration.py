"""Bounded real fixture for production SAM assembly/runtime qualification.

The assembly seam receives eleven exact native source planes. This does not
change the CLI's processing-cube behavior or claim a native processing flag.
Detector and model stages require the ordinary shared GPU reservation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()

def prepare(args):
    parent = json.loads(args.parent_plan.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    source = np.memmap(parent["source_images"], dtype=np.uint8, mode="r", shape=tuple(parent["source_shape_tyx"]))
    native_frames = list(range(54, 65))
    local = [frame-int(parent["source_frame_start"]) for frame in native_frames]
    shape = (len(local), *source.shape[1:])
    image_path = args.output / "native_images.uint8.dat"
    images = np.memmap(image_path, dtype=np.uint8, mode="w+", shape=shape)
    frame_hashes = []
    for index, source_index in enumerate(local):
        images[index] = source[source_index]
        expected = hashlib.sha256(source[source_index].tobytes()).hexdigest()
        actual = hashlib.sha256(images[index].tobytes()).hexdigest()
        if actual != expected:
            raise ValueError("Native source-pixel fixture copy changed")
        frame_hashes.append(dict(fixture_frame=index, native_source_frame=native_frames[index], sha256=actual))
    images.flush()
    del images, source
    seed_id = "native_f0054_component0006"
    descriptor = next(row for row in parent["source_observations"] if row["observation_id"] == seed_id)
    seed_file = Path(parent["seed_directory"]) / (seed_id+".npz")
    with np.load(seed_file, allow_pickle=False) as packet:
        mask, bbox = packet["mask"].copy(), list(map(int, packet["bbox_yx"]))
    if hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() != descriptor["mask_sha256"]:
        raise ValueError("Original detector component seed changed")
    observed = np.zeros(shape[1:], bool)
    y0, x0, y1, x1 = bbox
    observed[y0:y1, x0:x1] = mask
    np.save(args.output / "original_detector_0054_component.npy", observed)
    manifest = dict(schema="xta.native_tiled_integration_fixture/1",
        source_images=parent["source_images"], source_images_sha256=file_sha(parent["source_images"]),
        parent_plan=str(args.parent_plan.resolve()), parent_plan_sha256=parent["plan_sha256"],
        image_path=str(image_path.resolve()), image_sha256=file_sha(image_path), shape_tyx=list(shape),
        native_source_frames=native_frames, source_frame_start=54, original_source_frame_start=594,
        frame_hashes=frame_hashes, source_grid="Unchanged native3064x3024 pixels; contiguous original source frames54..64",
        qualification_route="Production assembly/context/runtime/publication seam; CLI processing-cube decoding is separate",
        native_processing_cli_switch=False, processing_mode_changed=False,
        original_seed_id=seed_id, original_seed_descriptor=descriptor,
        expected_original_detector_width=x1-x0,
        original_seed_file=str((args.output / "original_detector_0054_component.npy").resolve()),
        endpoint_native_frames=[54,64], interior_gate_probe_native_frame=59,
        next_stage="Actual detector64/59 observations after integration handoff; no annotation input",
        planned_flags=dict(interpolation_backend="sam", interpolation_distance=15,
            interpolation_walk_back=0, interpolation_candidates=1, interpolation_passes=1,
            interpolation_min_radius=3, interpolation_search_angle=30),
        delayed_native_expansion="Explicit0 for native working-canvas seam and separately recorded CLI smoke",
        labels_used=False, image_intervention=False,
        command=sys.argv)
    write(args.output / "fixture.json", manifest)
    print(json.dumps({"fixture": str(args.output / "fixture.json"), "shape_tyx": shape,
        "seed_width": x1-x0, "frames": native_frames}, indent=2))

def derive(args):
    from scipy import ndimage as ndi
    fixture = json.loads((args.output / "fixture.json").read_text())
    shape = tuple(fixture["shape_tyx"])
    first = np.load(fixture["original_seed_file"]).astype(bool)
    observations = np.memmap(args.output / "parent_observations.uint8.dat", dtype=np.uint8, mode="w+", shape=shape)
    observations[:] = 0
    observations[0] = first
    records = []
    for frame in (64, 59):
        union = np.load(args.output / "native_detector" / f"endpoint_{frame:04d}_union.npy")
        labels, _ = ndi.label(union != 0, structure=np.ones((3,3), bool))
        overlap = np.bincount(labels[first].ravel())
        overlap[0] = 0
        selected = int(np.argmax(overlap))
        if not selected or not overlap[selected]:
            raise ValueError("No actual detector component overlaps the original wide source observation")
        component = labels == selected
        np.save(args.output / f"actual_detector_{frame:04d}_component.npy", component)
        if frame == 64:
            observations[-1] = component
        records.append(dict(native_frame=frame, connected_component_index=selected,
            overlap_with_original54=int(overlap[selected]), foreground=int(component.sum()),
            selection="Maximum overlap with one prescribed actual original54 detector component; no annotation or SAM input"))
    observations.flush()
    del observations
    fixture.update(parent_observations=str(args.output / "parent_observations.uint8.dat"),
        parent_observations_sha256=file_sha(args.output / "parent_observations.uint8.dat"),
        new_detector_components=records,
        observation_source="Prescribed subset of actual3072-gray full-frame detector components; intermediate parent observations absent. Native59 actual detector is a separate gate probe.")
    write(args.output / "fixture.json", fixture)
    print(json.dumps(records, indent=2))

def native_view(shape):
    from XTA.geometry import ViewInfo
    return ViewInfo(name="transverse__tta_a0", physical_view_name="transverse",
        summary_family="transverse__tta_a0", tta_aug_id="a0", num_slices=shape[0],
        src_h=shape[1], src_w=shape[2], full_t=shape[0], full_h=shape[1], full_w=shape[2],
        pad_mode="clamp", family="orthogonal", tta_angle_deg=0.)

def store_volume(store):
    return np.stack([store.decode_slice(index) for index in range(store.shape[0])])

def seam(args):
    from XTA import assembly, backprojection, outputs, pipeline
    from XTA.sam_integration import SamInterpolationContext, _store_fingerprint, sam_workers_unsettled
    from XTA.sam_crop_tiling import resolve_sam_crop_mode, tile_grid
    from XTA.interpolation import TilePostprocessTask
    from tools.compare_sam_crop_strategies import helper
    fixture = json.loads((args.output / "fixture.json").read_text())
    shape = tuple(fixture["shape_tyx"])
    images = np.memmap(fixture["image_path"], dtype=np.uint8, mode="r", shape=shape)
    original = np.memmap(fixture["parent_observations"], dtype=np.uint8, mode="r", shape=shape)
    runroot = args.output / (args.run_label or f"native_{args.crop_mode}_{args.policy}")
    runroot.mkdir(parents=True, exist_ok=True)
    observations = np.memmap(runroot / "original.uint8.dat", dtype=np.uint8, mode="w+", shape=shape)
    observations[:] = original
    observations.flush()
    os.environ["YOLO_TTA_SAM_CROP_MODE"] = args.crop_mode
    os.environ["YOLO_TTA_DELAY_NATIVE_EXPANSION"] = "0"
    mode = resolve_sam_crop_mode()
    policy = None if args.policy == "stock" else {"sam_bridge_policy": "permissive"}
    view = native_view(shape)
    diagnostic = helper()
    context = None
    prepared = None
    sink = outputs.NrrdLayerSink(nrrd_dir=runroot / "nrrd", stem="native_seam",
        output_shape_tyx=shape, max_workers=1)
    outputs.set_nrrd_layer_sink(sink)
    began = time.perf_counter()
    report = dict(schema="xta.real_native_sam_assembly_qualification/1", command=sys.argv,
        diagnostic_source_sha256=file_sha(__file__),
        fixture=str(args.output / "fixture.json"), crop_mode=mode, policy=args.policy,
        source_grid_shape_tyx=shape, native_frames=fixture["native_source_frames"],
        processing_cli_cube_stage_executed=False, original_observation_source=fixture["observation_source"],
        image_sha256=fixture["image_sha256"], original_input_sha256=fixture["parent_observations_sha256"],
        labels_used=False, outer_crop_rules_changed=False)
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),
            "sam_tiled_real_native_assembly"), diagnostic.resource_monitor(runroot / "resources.json", 0):
        backprojection._configure_main_process_gpu_stage_workers((0,))
        backprojection._set_main_process_gpu_inference_priority_active(False)
        assembly.set_final_source_output_shape(shape)
        context = SamInterpolationContext(model_path=args.model, device_ids=(0,),
            temp_dir=runroot / "temporary", evidence_root=runroot / "sam_interpolation",
            source_volume=images, source_identity=fixture["image_sha256"], source_grid_shape=shape,
            source_resize_semantics="Exact native fixture; assembly seam excludes CLI processing-cube stage",
            detector_identity="actual3072_gray_bestpt_connected_component_subset", detector_device_ids=(0,),
            policy=policy, crop_mode=mode, delayed_native_expansion=False, feature_cache_mib=512)
        # Actual detector generation completed and its process exited before this stage.
        context.detector_assets_retired()
        original_interpolate = context.interpolate
        def traced_interpolate(volume, **kwargs):
            lineage = dict(kwargs.get("upstream_lineage") or {})
            lineage["observation_source"] = fixture["observation_source"]
            lineage.update(native_source_frames=fixture["native_source_frames"],
                original_source_frame_start=fixture["original_source_frame_start"],
                source_image_sha256=fixture["image_sha256"])
            kwargs["upstream_lineage"] = lineage
            return original_interpolate(volume, **kwargs)
        context.interpolate = traced_interpolate
        try:
            ready = []
            prepared = assembly.prepare_view_volume_after_fullframe(
                model_name="actual_detector", view=view, union_mm=observations, confmap_mm=None,
                union_path=runroot / "original.uint8.dat", confmap_path=None, temp_dir=runroot / "temporary",
                dense_tiling_active=True, min_conf=0., min_radius=0., interpolate=15,
                interpolation_walk_back=0, interpolation_candidates=1, interpolate_passes=1,
                interpolate_min_radius=3., interpolation_search_angle=30., keep_temp=True,
                slice_workers=1, interpolation_task_workers=1, nrrd_layers_enabled=True,
                interpolation_backend="sam", sam_context=context,
                parent_mask_ready_callback=lambda *_: ready.append("immutable_parent_ready"))
            if ready != ["immutable_parent_ready"]:
                raise AssertionError("Ordinary immutable parent support was not published")
            stats = prepared.interpolation_stats[0]
            if not stats.get("sam_generated_runs"):
                raise AssertionError("Native qualification must actually generate independently seeded SAM runs")
            if mode == "tiled" and stats.get("sam_tiled_multi_tile_runs",0) < 1:
                raise AssertionError("Requested tiled mode did not actually exercise multiple footprints")
            np.save(runroot / "online_selected_additions.npy", np.asarray(prepared.final_view_volume_mm != 0) & ~(observations != 0))
            report.update(parent_stats=stats, parent_support_immutable=bool(np.array_equal(store_volume(prepared.parent_mask_support_mm), observations)),
                immutable_postcleanup_detector_snapshot_sha256=stats["sam_scope_metadata"]["observation_snapshot_sha256"],
                postcleanup_detector_file_sha256=file_sha(runroot / "original.uint8.dat"),
                parent_bridge_support_path=(str(prepared.parent_bridge_support_mm.root) if prepared.parent_bridge_support_mm is not None else None),
                parent_bridge_identity=(_store_fingerprint(Path(prepared.parent_bridge_support_mm.root)) if prepared.parent_bridge_support_mm is not None else str(stats.get("parent_gate_support_identity", ""))),
                parent_nrrd_layers=[dict(key=layer.key, path=str(layer.path), direction=layer.interpolation_direction,
                    shape=list(layer.shape), native_transform=layer.native_transform) for layer in prepared.nrrd_layers])
            # Gate a prescribed crop of actual detector output. It is explicitly
            # not claimed to be independently inferred YOLO tile output.
            probe = np.load(args.output / "actual_detector_0059_component.npy")
            target = np.load(args.output / "actual_detector_0064_component.npy")
            from XTA.sam_evidence import SamEvidenceBundle
            bundle = SamEvidenceBundle.open(stats["sam_evidence_path"])
            group = next(iter(bundle.groups.values()))
            grid = tile_grid(group["context_bbox_yx"])
            total = np.zeros(shape, np.uint8)
            parent_category = np.zeros(shape, np.uint8)
            bridge_category = np.zeros(shape, np.uint8)
            gate_records = []
            for tile in grid:
                y0,x0,y1,x1 = tile.crop_bbox_yx
                path = runroot / (tile.tile_id + ".uint8.dat")
                tile_mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(shape[0],y1-y0,x1-x0))
                tile_mm[:] = 0
                tile_mm[5] = probe[y0:y1,x0:x1]
                tile_mm[10] = target[y0:y1,x0:x1]
                task = TilePostprocessTask("actual_detector", view.name, "a0", 0., "diagnostic_actual_fullframe_crops",
                    tile.tile_id, (y0,y1,x0,x1), tile_mm, None, path, None,
                    processing_shape=tuple(tile_mm.shape), threshold_plane_shape=shape[1:])
                result = assembly.postprocess_tile_volume_after_inference(task, view=view,
                    min_conf=0., min_radius=0., keep_temp=True, slice_workers=1)
                first_gate = assembly.gate_tile_result_against_parent_mask(result,
                    parent_mask_support_mm=prepared.parent_mask_support_mm,
                    tile_accumulator_mm=total, tile_accumulator_locks=None,
                    work_dir=runroot / "tile_gate", keep_temp=True, slice_workers=1,
                    tile_parent_mask_accumulator_mm=parent_category)
                second = None
                if first_gate.residual_result is not None and prepared.parent_bridge_support_mm is not None:
                    second = assembly.gate_tile_residual_against_parent_bridge(first_gate.residual_result,
                        parent_bridge_support_mm=prepared.parent_bridge_support_mm,
                        tile_accumulator_mm=total, tile_accumulator_locks=None,
                        work_dir=runroot / "tile_gate", keep_temp=True, slice_workers=1,
                        tile_parent_bridge_accumulator_mm=bridge_category)
                gate_records.append(dict(tile_id=tile.tile_id, crop_bbox_yx=tile.crop_bbox_yx,
                    source="Prescribed native crop of actual fullframe detector59/64 masks; no SAM seed handoff",
                    parent=first_gate.gate_stats, bridge=getattr(second,"gate_stats",None)))
                if first_gate.residual_result is not None and prepared.parent_bridge_support_mm is None:
                    assembly._delete_tile_result_storage(first_gate.residual_result, keep_temp=True)
                del task, tile_mm, result, first_gate, second
            report.update(gates=gate_records, bridge_rescued_native59_voxels=int(bridge_category[5].sum()),
                direct_parent_native64_voxels=int(parent_category[10].sum()))
            if args.policy == "raw" and not bridge_category[5].any():
                raise AssertionError("Explicit permissive path must exercise nonempty actual-detector bridge rescue")
            identity = report["parent_bridge_identity"]
            lineage = dict(parent_scope="parent_bridge", gate_support_identity=identity,
                interpolation_policy_identity=stats["sam_policy_hash"],
                gate_support_fingerprints={"parent_bridge":identity},
                observation_source="Only ordinary-gated prescribed crops of actual detector59/64 components")
            consolidated = assembly.finalize_consolidated_tile_volume_for_parent(
                model_name="actual_detector", view=view, tile_accumulator_mm=total,
                destination_mm=prepared.final_view_volume_mm, destination_lock=threading.Lock(),
                temp_dir=runroot / "temporary", interpolate=15, interpolation_walk_back=0,
                interpolation_candidates=1, interpolate_passes=1, interpolate_min_radius=3.,
                interpolation_search_angle=30., keep_temp=True, slice_workers=1,
                interpolation_task_workers=1, nrrd_layers_enabled=True,
                tile_parent_mask_accumulator_mm=parent_category, tile_parent_bridge_accumulator_mm=bridge_category,
                config_id="diagnostic_actual_fullframe_crops", interpolation_backend="sam",
                sam_context=context, sam_upstream_lineage=lineage)
            report["consolidated_stats"] = consolidated.interpolation_stats
            report["context_runtime"] = dict(start_seconds=context.start_seconds, render_seconds=context.render_seconds,
                wait_seconds=context.wait_seconds, rendered_frames=context.rendered_frames,
                rendered_pixels=context.rendered_pixels, exact_backing_reuses=context.exact_backing_reuses)
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if prepared is not None:
                for store in (prepared.parent_mask_support_mm, prepared.parent_bridge_support_mm):
                    if store is not None:
                        store.close()
            if context is not None:
                context.close()
                report["context_closed"] = context._closed
                report["gpu_leases_released"] = len(context._leases) == 0
            report["unsettled_sam_workers"] = sam_workers_unsettled()
            sink.wait()
            report["nrrd_manifest"] = str(sink.write_manifest())
            sink.shutdown()
            outputs.set_nrrd_layer_sink(None)
            report["wall_seconds"] = time.perf_counter()-began
            assembly.set_final_source_output_shape(None)
            pipeline._reset_gpu_stage_coordinator_if_sam_settled()
            write(runroot / "qualification.json", report)
    print(json.dumps({"status":report["status"], "crop_mode":mode, "policy":args.policy,
        "parent_generated":stats["sam_generated_runs"], "parent_added":stats["added_voxels"],
        "gate_rescued":report["bridge_rescued_native59_voxels"], "output":str(runroot)}, indent=2))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "derive", "seam"))
    p.add_argument("--parent-plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--crop-mode", choices=("whole","tiled"), default="whole")
    p.add_argument("--policy", choices=("stock","raw"), default="stock")
    p.add_argument("--run-label", help="Fresh artifact directory name; preserves prior attempts")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "derive":
        derive(args)
    elif args.stage == "seam":
        seam(args)

if __name__ == "__main__":
    main()
