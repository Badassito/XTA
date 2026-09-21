"""Compare external reconciliation policies on saved additive NRRD layers."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from XTA.reconciliation import EvidenceLayer, reconcile
import XTA as _xta_package
from XTA.reconciliation_components import component_statistics
from XTA.reconciliation_io import read_layer_manifest, write_seg_nrrd
from XTA.reconciliation_policy import load_reconciliation_policy, resolve_reconciliation

_POLICIES = Path(_xta_package.__file__).resolve().parent / "examples" / "external_reconciliation"
_DEFAULT_POLICIES = ("union", "quorum3", "largest_island", "hybrid_with_fill",
                     "confidence_anchored", "confidence_core_rescue")
_CONFIDENCE_POLICIES = ("confidence_anchored", "confidence_core_rescue")
_CONFIDENCE_UNAVAILABLE = "Saved binary layer readers provide no retained prediction confidence; the run's confidence threshold is not a confidence value."


def _progress(message):
    print(message, flush=True)


def _atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def discover_geometry_context(layer_manifest: str | Path) -> tuple[dict, Path | None, dict | None]:
    """Find the nearest completed-run manifest and retain both source T scales."""
    manifest = Path(layer_manifest).resolve()
    for directory in manifest.parents:
        candidate = directory / "manifest.json"
        if candidate == manifest or not candidate.is_file():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        geometry = data.get("geometry", {})
        inputs = data.get("inputs", {})
        if not isinstance(geometry, dict) or not any(key in geometry for key in ("physical_views", "inference_view_variants")):
            continue
        views = {}
        for key in ("physical_views", "inference_view_variants"):
            for view in geometry.get(key, []):
                if isinstance(view, dict) and view.get("name"):
                    views[view["name"]] = view
        context = {"views_by_name": views, "source_shape_tyx": inputs.get("source_shape_t_y_x"),
                   "processing_shape_tyx": inputs.get("processing_shape_t_y_x")}
        return context, candidate, data
    return {}, None, None


def _load_policies(paths, memory_mib):
    policies = []
    seen = set()
    for path in paths:
        settings = resolve_reconciliation(SimpleNamespace(reconciliation=str(path), reconciliation_memory_mib=memory_mib))
        if settings.sha256 in seen:
            continue
        seen.add(settings.sha256)
        policy = load_reconciliation_policy(settings)
        policies.append((settings, policy))
    if not any(policy["mode"] == "union" and policy["decide"] is None for _, policy in policies):
        union_settings = resolve_reconciliation(SimpleNamespace(reconciliation=str(_POLICIES / "union.py"), reconciliation_memory_mib=memory_mib))
        policies.insert(0, (union_settings, load_reconciliation_policy(union_settings)))
    policies.sort(key=lambda item: 0 if item[1]["mode"] == "union" and item[1]["decide"] is None else 1)
    return policies


def _name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "policy"


def _file_fingerprint(path: Path) -> dict:
    info = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return {"size_bytes": info.st_size, "modified_time_ns": info.st_mtime_ns, "sha256": hasher.hexdigest()}


def resolve_saved_confidence(collection, run_manifest_path, *, reader_cleanup=None):
    """Attach persisted scores using exact model/layer identities and grids."""
    predictions = [layer for layer in collection if layer.metadata.get("mask_kind") == "yolo"
                   and not layer.metadata.get("empty_segment", False)]
    info = {"available": False, "status": "unavailable", "reason": _CONFIDENCE_UNAVAILABLE,
            "manifest": None, "required_prediction_layers": len(predictions),
            "matched_prediction_layers": 0, "missing_prediction_layers": [], "unsupported_grids": [],
            "deferred_native_layers": []}
    if not predictions:
        info.update(available=True, status="available", reason=None)
        return {}, info, []
    if run_manifest_path is None:
        return {}, info, []
    manifest_path = Path(run_manifest_path).parent / "reconciliation_evidence" / "manifest.json"
    if not manifest_path.is_file():
        return {}, info, []
    from XTA.confidence_evidence import ConfidenceEvidenceRef, SCHEMA
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {SCHEMA, "xta.confidence_evidence/2"} or not isinstance(manifest.get("layers"), list):
        raise ValueError(f"Invalid confidence evidence manifest: {manifest_path}")
    info["manifest"] = str(manifest_path)
    entries = {}
    for entry in manifest["layers"]:
        key = (str(entry["model_name"]), str(entry["layer_key"]))
        if key in entries:
            raise ValueError(f"Duplicate saved confidence identity: {key}")
        entries[key] = entry
    files = [manifest_path]
    readers = {}
    geometry = collection.geometry
    canonical_grid = (geometry.space == "left-posterior-superior"
                      and np.allclose(geometry.directions_xyz, np.eye(3), rtol=0, atol=1e-9)
                      and np.allclose(geometry.origin_xyz, (0, 0, 0), rtol=0, atol=1e-9))
    evidence_root = manifest_path.parent.resolve()
    for layer in predictions:
        model = layer.metadata.get("model_name")
        layer_key = layer.metadata.get("layer_key")
        entry = entries.get((model, layer_key)) if isinstance(model, str) and isinstance(layer_key, str) else None
        if entry is None:
            info["missing_prediction_layers"].append(layer.layer_id)
            continue
        directory = (evidence_root / str(entry["directory"])).resolve()
        if not directory.is_relative_to(evidence_root):
            raise ValueError("Confidence directory escapes its evidence manifest")
        reference = ConfidenceEvidenceRef.open(directory)
        if (reference.model_name, reference.layer_key) != (model, layer_key):
            raise ValueError(f"Confidence metadata identity disagrees with its manifest: {directory}")
        if tuple(entry.get("output_shape_tyx", reference.shape)) != reference.shape:
            raise ValueError(f"Confidence manifest shape disagrees with its sidecar: {directory}")
        for path in directory.rglob("*"):
            if path.is_file():
                if not path.resolve().is_relative_to(directory):
                    raise ValueError("Confidence storage file escapes its evidence directory")
                files.append(path)
        if getattr(reference, "coordinate_space", "source") != "source":
            info["deferred_native_layers"].append(layer.layer_id)
            continue
        axes_match = re.sub(r"\s+", "", str(reference.metadata.get("exported_axes", ""))).lower() == "(x,y,t)"
        if reference.shape != collection.shape_tyx or not canonical_grid or not axes_match:
            info["unsupported_grids"].append({"layer_id": layer.layer_id,
                "mask_shape_tyx": list(collection.shape_tyx), "confidence_shape_tyx": list(reference.shape),
                "canonical_source_reference": canonical_grid, "exported_axes_match": axes_match})
            continue
        readers[layer.layer_id] = reference.reader()
        if reader_cleanup is not None and hasattr(readers[layer.layer_id], "close"):
            reader_cleanup.callback(readers[layer.layer_id].close)
    info["matched_prediction_layers"] = len(readers)
    if info["unsupported_grids"]:
        info.update(status="unsupported_confidence_grid", reason="Retained confidence and mask grids differ; saved-score replay requires the exact source reference grid and does not resize scores.")
    elif info["deferred_native_layers"]:
        info.update(status="native_confidence_requires_export", reason="Deferred native-view confidence requires explicit source-grid or compact export before this comparison; collection does not stage projection implicitly.")
    elif info["missing_prediction_layers"]:
        info["reason"] = f"Retained confidence is missing for {len(info['missing_prediction_layers'])} nonempty prediction layers; the run threshold cannot replace it."
    else:
        info.update(available=True, status="available", reason=None)
    return readers, info, files


def _confidence_summary(report):
    entries = [dataset["confidence_evidence"] for dataset in report["datasets"] if "confidence_evidence" in dataset]
    complete = bool(entries) and all(entry["available"] for entry in entries)
    any_available = any(entry["available"] for entry in entries)
    report["confidence_available"] = complete
    report["confidence_status"] = "available" if complete else "mixed" if any_available else "unavailable"
    report["confidence_unavailable_reason"] = None if complete else "; ".join(sorted({entry["reason"] for entry in entries if entry["reason"]}))
    report["unavailable_standard_policies"] = [] if complete else list(_CONFIDENCE_POLICIES)


def _preview_arrays(union: np.ndarray, result: np.ndarray) -> np.ndarray:
    image = np.zeros((*union.shape, 3), dtype=np.uint8)
    retained = (union != 0) & (result != 0)
    rejected = (union != 0) & (result == 0)
    image[retained] = (45, 190, 135)
    image[rejected] = (240, 85, 85)
    return image


def _roi_bounds(mask: np.ndarray, padding=12):
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return (0, mask.shape[0], 0, mask.shape[1])
    return (max(0, int(ys.min())-padding), min(mask.shape[0], int(ys.max())+padding+1),
            max(0, int(xs.min())-padding), min(mask.shape[1], int(xs.max())+padding+1))


def write_previews(directory: Path, methods: list[dict], shape, *, dataset_name: str,
                   preview_slices, preview_planes) -> dict:
    """Render retained preview planes after full raw output maps are retired."""
    if "matplotlib" not in sys.modules:
        os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    union_record = next(record for record in methods if record.get("is_union_reference"))
    union = preview_planes[union_record["output"]]
    slices = preview_slices
    areas = {z: int(np.count_nonzero(union[z])) for _, z in slices}
    roi_info = [{"label": label, "z": z, "roi_y0_y1_x0_x1": list(_roi_bounds(union[z])),
                 "union_slice_voxels": int(areas[z])} for label, z in slices]
    paths = []
    for detail in (False, True):
        fig, axes = plt.subplots(len(methods), len(slices), figsize=(10.8, max(3.3, 3.2*len(methods))), squeeze=False, facecolor="#151920")
        try:
            for row, method in enumerate(methods):
                current = preview_planes[method["output"]]
                for column, (label, z) in enumerate(slices):
                    left, right = union[z], current[z]
                    if method.get("is_union_reference"):
                        image = np.repeat((left[..., None] * 210), 3, axis=2)
                    else:
                        image = _preview_arrays(left, right)
                    if detail:
                        y0, y1, x0, x1 = roi_info[column]["roi_y0_y1_x0_x1"]
                        image = image[y0:y1, x0:x1]
                    axis = axes[row, column]
                    axis.imshow(image, interpolation="nearest", origin="upper")
                    axis.set_axis_off()
                    kept = int(np.count_nonzero(right))
                    axis.set_title(f"{method['policy_name']} | {label}, z={z}\n{kept:,} kept / {int(areas[z]):,} candidate pixels", color="white", fontsize=10)
            fig.suptitle(f"{dataset_name} — raw reconciliation before global postprocessing", color="white", fontsize=13)
            fig.legend(handles=[Patch(color=(.82,.82,.82),label="Raw union"), Patch(color=(45/255,190/255,135/255),label="Retained"),
                                Patch(color=(240/255,85/255,85/255),label="Rejected")], loc="lower center", ncol=3, facecolor="#151920", labelcolor="white")
            fig.tight_layout(rect=(0, .035, 1, .965))
            output = directory / ("comparison_detail.png" if detail else "comparison_slices.png")
            fig.savefig(output, dpi=140, facecolor=fig.get_facecolor())
            paths.append(str(output))
        finally:
            plt.close(fig)
    return {"paths": paths, "slices": roi_info, "slice_selection": "Central output z and maximum raw-union foreground area; ROI bounds enclose the raw-union slice."}


def _csv_report(output: Path, datasets: list[dict]) -> None:
    fields = ["dataset", "policy_name", "status", "candidate_voxels", "retained_voxels", "rejected_voxels", "retained_fraction",
              "component_count_6", "largest_component_voxels_6", "largest_fraction_6", "elapsed_seconds", "reason"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            for method in dataset["methods"]:
                row = {key: method.get(key, "") for key in fields}
                row["dataset"] = dataset["dataset"]
                writer.writerow(row)


def compare(inputs, policy_paths, output, *, memory_mib=2048, previews=True, progress=_progress):
    if isinstance(memory_mib, bool) or not math.isfinite(float(memory_mib)) or float(memory_mib) <= 0:
        raise ValueError("memory_mib must be positive and finite")
    paths = [Path(path).resolve() for path in inputs]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("--input requires existing NRRD layer manifests")
    if len(set(paths)) != len(paths):
        raise ValueError("Each input manifest must be listed only once")
    policies = _load_policies(policy_paths, memory_mib)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output must be a fresh directory: {output}")
    if any(output.is_relative_to(path.parent) for path in paths):
        raise ValueError("Comparison outputs must be outside the input layer directories")
    output.mkdir(parents=True)
    report = {"schema": "xta.saved-reconciliation-comparison/1", "memory_mib": memory_mib,
              "comparison_stage": "Raw additive reconciliation before all global postprocessing",
              "interpretation": "Descriptive behavior on arbitrary examples, not golden samples or accuracy measurements. Counts refer to the supplied voxel grid; header display spacing does not establish physical units.",
              "confidence_available": False, "confidence_unavailable_reason": _CONFIDENCE_UNAVAILABLE,
              "unavailable_standard_policies": list(_CONFIDENCE_POLICIES), "datasets": []}
    implementation_root = Path(_xta_package.__file__).resolve().parent
    report["implementation_sha256"] = {name: hashlib.sha256((implementation_root / name).read_bytes()).hexdigest() for name in
                                        ("reconciliation.py", "reconciliation_components.py", "reconciliation_geometry.py", "reconciliation_io.py", "confidence_evidence.py")}
    _atomic_json(output / "comparison.json", report)
    for dataset_index, manifest_path in enumerate(paths):
        context, run_manifest_path, run_manifest = discover_geometry_context(manifest_path)
        dataset_name = run_manifest_path.parent.name if run_manifest_path else manifest_path.stem
        directory = output / f"{dataset_index+1:02d}_{_name(dataset_name)}"
        directory.mkdir()
        work = directory / ".working"
        work.mkdir()
        dataset = {"dataset": dataset_name, "input_manifest": str(manifest_path),
                   "run_manifest": str(run_manifest_path) if run_manifest_path else None,
                   "run_confidence_threshold": (run_manifest or {}).get("resolved_configuration", {}).get("conf"),
                   "geometry_context": {k: v for k,v in context.items() if k != "views_by_name"},
                   "methods": [], "input_fingerprints": {}}
        report["datasets"].append(dataset)
        with ExitStack() as reader_cleanup, read_layer_manifest(manifest_path, workspace=work, memory_mib=max(1, float(memory_mib)/8), max_open=1024) as collection:
            shape = collection.shape_tyx
            dataset.update(shape_tyx=list(shape), additive_layer_count=len(collection), reference_geometry={
                "space": collection.geometry.space, "directions_xyz": collection.geometry.directions_xyz,
                "origin_xyz": collection.geometry.origin_xyz})
            confidence_readers, confidence_info, confidence_files = resolve_saved_confidence(
                collection, run_manifest_path, reader_cleanup=reader_cleanup)
            dataset["confidence_evidence"] = confidence_info
            _confidence_summary(report)
            for layer in collection:
                if layer.path is not None:
                    dataset["input_fingerprints"][str(layer.path)] = _file_fingerprint(layer.path)
            dataset["input_fingerprints"][str(manifest_path)] = _file_fingerprint(manifest_path)
            if run_manifest_path:
                dataset["input_fingerprints"][str(run_manifest_path)] = _file_fingerprint(run_manifest_path)
            for confidence_file in confidence_files:
                dataset["input_fingerprints"][str(confidence_file)] = _file_fingerprint(confidence_file)
            evidence = [EvidenceLayer(layer.layer_id, shape, layer.metadata, layer.read_slab,
                                      confidence_reader=confidence_readers.get(layer.layer_id)) for layer in collection]
            preview_slices, preview_planes = None, {}
            for policy_index, (settings, policy) in enumerate(policies):
                policy_name = str(policy["name"])
                method = {"policy_name": policy_name, "policy_file": settings.path, "policy_sha256": settings.sha256,
                          "status": "pending"}
                dataset["methods"].append(method)
                if policy["mode"] == "confidence" and not confidence_info["available"]:
                    method.update(status=confidence_info["status"], reason=confidence_info["reason"])
                    progress(f"{dataset_name}: {policy_name} {confidence_info['status']}: {confidence_info['reason']}")
                    continue
                label = f"{policy_index+1:02d}_{_name(policy_name)}"
                raw_path = work / f"{label}.uint8"
                array = None
                nrrd_path = directory / f"{label}.seg.nrrd"
                is_union_reference = policy["mode"] == "union" and policy["decide"] is None
                areas = np.zeros(shape[0], np.int64) if previews and preview_slices is None else None
                started = time.monotonic()
                last_progress = [0.0]
                def on_progress(stage, done, total):
                    now = time.monotonic()
                    if done == total or now-last_progress[0] >= 20:
                        progress(f"{dataset_name}: {policy_name} {stage} {done}/{total}")
                        last_progress[0] = now
                def write_output(z0, z1, slab):
                    array[z0:z1] = slab
                    if areas is not None:
                        areas[z0:z1] = np.count_nonzero(slab, axis=(1, 2))
                try:
                    array = np.memmap(raw_path, mode="w+", dtype=np.uint8, shape=shape)
                    result = reconcile(evidence, shape_tyx=shape, policy=policy,
                                       write_slab=write_output,
                                       memory_mib=memory_mib, geometry_context=context, progress=on_progress)
                    array.flush()
                    stats = component_statistics(lambda a,b: np.asarray(array[a:b]), shape, memory_mib=memory_mib)
                    artifact = write_seg_nrrd(nrrd_path, shape_tyx=shape, read_slab=lambda a,b: np.asarray(array[a:b]),
                                              geometry=collection.geometry, segment_name=f"{dataset_name} {policy_name}",
                                              memory_mib=memory_mib)
                    settings.assert_unchanged()
                    if previews:
                        if preview_slices is None:
                            if not is_union_reference:
                                raise RuntimeError("Preview selection requires the raw union first")
                            preview_slices = [("Central slice", shape[0]//2),
                                              ("Largest raw-union slice", int(np.argmax(areas)))]
                        preview_planes[str(nrrd_path)] = {z: np.array(array[z], dtype=np.uint8, copy=True)
                                                          for _, z in preview_slices}
                finally:
                    if array is not None:
                        array._mmap.close()
                    raw_path.unlink(missing_ok=True)
                    for layer in collection:
                        layer.close()
                counts = result["counts"]
                method.update(status="complete", **{k: counts[k] for k in ("candidate_voxels", "retained_voxels", "rejected_voxels")},
                              retained_fraction=counts["retained_voxels"]/counts["candidate_voxels"] if counts["candidate_voxels"] else 0,
                              component_count_6=stats["component_count"], largest_component_voxels_6=stats["largest_component_voxels"],
                              largest_fraction_6=stats["largest_fraction"], elapsed_seconds=round(time.monotonic()-started,3),
                              output=str(nrrd_path), artifact=artifact, component_statistics=stats, reconciliation=result,
                              is_union_reference=is_union_reference)
                progress(f"{dataset_name}: {policy_name} retained {counts['retained_voxels']:,}/{counts['candidate_voxels']:,} voxels")
                _atomic_json(output / "comparison.json", report)
                _csv_report(output, report["datasets"])
            completed = [m for m in dataset["methods"] if m["status"] == "complete"]
            if previews:
                dataset["previews"] = write_previews(directory, completed, shape, dataset_name=dataset_name,
                                                     preview_slices=preview_slices, preview_planes=preview_planes)
                dataset["preview_retained_bytes"] = sum(plane.nbytes for planes in preview_planes.values() for plane in planes.values())
            dataset["raw_output_staging"] = "one policy map at a time; retired immediately after publication and preview-plane capture"
            work.rmdir()
            for path, original in dataset["input_fingerprints"].items():
                if _file_fingerprint(Path(path)) != original:
                    raise RuntimeError(f"Input changed during comparison: {path}")
            dataset["inputs_unchanged"] = True
        _atomic_json(output / "comparison.json", report)
        _csv_report(output, report["datasets"])
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, metavar="NRRD_MANIFEST.json")
    parser.add_argument("--policy", nargs="+", metavar="POLICY.py", default=None,
                        help="Defaults: union, quorum3, largest_island, hybrid_with_fill, confidence_anchored, confidence_core_rescue. A raw-union reference is always included.")
    parser.add_argument("--output", required=True, metavar="FRESH_DIRECTORY")
    parser.add_argument("--memory_mib", type=int, default=2048)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = args.policy or [_POLICIES / f"{name}.py" for name in _DEFAULT_POLICIES]
    compare(args.input, paths, args.output, memory_mib=args.memory_mib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
