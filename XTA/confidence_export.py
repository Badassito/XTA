"""Portable compact confidence bundles from completed source-aligned evidence.

Only score crops and compact mask planes are decoded. Native NRRD masks are
never opened; optional deferred-native conversion is explicit and bounded by
the confidence reference's staging API.
"""
from __future__ import annotations

from contextlib import closing, nullcontext
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .confidence_evidence import ConfidenceEvidenceRef, write_confidence_evidence
from .reconciliation_io import read_layer_manifest, write_seg_nrrd


EXPORT_SCHEMA = "xta.compact_confidence_export/1"
SOURCE_SCHEMA = "xta.confidence_evidence/1"
SUPPORTED_EVIDENCE_SCHEMAS = {SOURCE_SCHEMA, "xta.confidence_evidence/2"}


def _shape(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) <= 0 for x in value
    ):
        raise ValueError(f"{name} requires three positive integer dimensions")
    return tuple(int(x) for x in value)


def _fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "sha256": digest.hexdigest()}


def _inside(base, relative, description):
    path = (Path(base) / str(relative)).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError(f"{description} escapes its manifest directory")
    return path


def _remove_stage(path, parent, prefix):
    resolved, expected = Path(path).resolve(), Path(parent).resolve()
    if resolved.parent != expected or not resolved.name.startswith(prefix):
        raise RuntimeError("Refusing cleanup outside the owned export staging directory")
    if resolved.exists():
        shutil.rmtree(resolved)


@lru_cache(maxsize=32)
def _axis_footprints(source_count, target_count, area):
    coordinates = np.arange(target_count, dtype=np.int64)
    first = coordinates * source_count // target_count
    stop = ((coordinates + 1) * source_count + target_count - 1) // target_count if area else first + 1
    first.flags.writeable = stop.flags.writeable = False
    return first, stop


def source_frames(source_t, target_t, z):
    """Match categorical temporal restoration, including small-axis upsampling."""
    if not 0 <= z < target_t:
        raise ValueError("Compact plane index is outside its grid")
    if source_t >= target_t:
        return range(z * source_t // target_t, ((z + 1) * source_t + target_t - 1) // target_t)
    position = 0. if target_t <= 1 else z * (source_t - 1) / (target_t - 1)
    return (int(round(position)),)


def max_crop_into(output, crop, *, source_hw):
    """Max one source crop into globally phased categorical output footprints."""
    y0, y1, x0, x1, values = crop
    values = np.asarray(values)
    source_h, source_w = map(int, source_hw)
    y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
    if (values.dtype != np.uint8 or values.shape != (y1-y0, x1-x0)
            or not 0 <= y0 <= y1 <= source_h or not 0 <= x0 <= x1 <= source_w):
        raise ValueError("Confidence crop has invalid source bounds, dtype, or shape")
    if not values.size:
        return
    target_h, target_w = output.shape
    area = source_h >= target_h and source_w >= target_w
    ys, ye = _axis_footprints(source_h, target_h, area)
    xs, xe = _axis_footprints(source_w, target_w, area)
    oy = np.flatnonzero((ys < y1) & (ye > y0))
    ox = np.flatnonzero((xs < x1) & (xe > x0))
    if not oy.size or not ox.size:
        return
    starts = np.maximum(xs[ox], x0) - x0
    ends = np.minimum(xe[ox], x1) - x0
    horizontal = np.zeros((values.shape[0], ox.size), np.uint8)
    for offset in range(int(np.max(ends-starts))):
        active = starts + offset < ends
        horizontal[:, active] = np.maximum(horizontal[:, active], values[:, (starts+offset)[active]])
    starts = np.maximum(ys[oy], y0) - y0
    ends = np.minimum(ye[oy], y1) - y0
    reduced = np.zeros((oy.size, ox.size), np.uint8)
    for offset in range(int(np.max(ends-starts))):
        active = starts + offset < ends
        reduced[active] = np.maximum(reduced[active], horizontal[(starts+offset)[active]])
    selection = np.ix_(oy, ox)
    output[selection] = np.maximum(output[selection], reduced)


def compact_score_plane(reader, mask_plane, *, source_shape, target_shape, z):
    """Pool known scores and intersect with the supplied compact binary mask."""
    mask = np.asarray(mask_plane)
    if mask.shape != tuple(target_shape[1:]) or mask.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise ValueError("Compact mask plane has an invalid shape or dtype")
    if np.any(mask > 1):
        raise ValueError("Compact masks must contain only zero and one")
    result = np.zeros(target_shape[1:], np.uint8)
    if not np.any(mask):
        return result
    for source_z in source_frames(source_shape[0], target_shape[0], int(z)):
        if hasattr(reader, "iter_crops"):
            crops = reader.iter_crops(source_z)
        else:
            crop = reader.read_crop(source_z)
            crops = () if crop is None else (crop,)
        for crop in crops:
            max_crop_into(result, crop, source_hw=source_shape[1:])
    result[mask == 0] = 0
    return result


def export_memory_plan(source_shape, target_shape, memory_mib):
    if isinstance(memory_mib, bool) or not math.isfinite(float(memory_mib)) or memory_mib <= 0:
        raise ValueError("memory_mib must be positive and finite")
    # A legacy bbox may cover an entire native plane. Account for decoded and
    # compressed crop buffers, separable gather temporaries, and compact planes.
    native_plane = math.prod(source_shape[1:])
    compact_plane = math.prod(target_shape[1:])
    workspace = 2*native_plane + 4*source_shape[1]*target_shape[2] + 8*compact_plane + 1024**2
    if workspace > int(float(memory_mib)*1024**2):
        raise ValueError(f"Compact export requires at least {math.ceil(workspace/1024**2)} MiB for its source-crop workspace")
    return {"memory_mib": memory_mib, "planned_crop_workspace_bytes": workspace,
            "native_plane_bytes": native_plane, "compact_plane_bytes": compact_plane,
            "scope": "Numerical crop/plane workspace; metadata and reader-specific projection staging are separate"}


def export_compact_evidence(run_manifest, compact_manifest, output, *, memory_mib=256,
                            allow_native_projection=False, max_staging_mib=32768, progress=None):
    """Publish a fresh, portable compact mask/score package without native masks."""
    run_manifest, compact_manifest, output = (Path(path).expanduser().resolve() for path in (run_manifest, compact_manifest, output))
    if output.exists():
        raise FileExistsError(f"Export output must be fresh: {output}")
    if isinstance(max_staging_mib, bool) or not math.isfinite(float(max_staging_mib)) or max_staging_mib <= 0:
        raise ValueError("max_staging_mib must be positive and finite")
    run = json.loads(run_manifest.read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError("Export requires a completed run manifest")
    source_shape = _shape(run.get("inputs", {}).get("source_shape_t_y_x"), "Run source_shape_t_y_x")
    compact = json.loads(compact_manifest.read_text(encoding="utf-8"))
    target_shape = _shape(compact.get("output_shape_tyx"), "Compact output_shape_tyx")
    if _shape(compact.get("full_quality_output_shape_tyx"), "Compact full_quality_output_shape_tyx") != source_shape:
        raise ValueError("Compact mask source grid disagrees with the completed run")
    scale = compact.get("downbin_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or not 0 < scale < 1:
        raise ValueError("Compact manifest must declare downbin_scale strictly between zero and one")
    expected = tuple(max(4, int(math.floor(max(1., size*float(scale))/4. + .5))*4) for size in source_shape)
    if target_shape != expected:
        raise ValueError("Compact dimensions disagree with the declared pipeline downbin scale")
    plan = export_memory_plan(source_shape, target_shape, memory_mib)
    evidence_manifest = run_manifest.parent / "reconciliation_evidence" / "manifest.json"
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    if evidence.get("schema") not in SUPPORTED_EVIDENCE_SCHEMAS or not isinstance(evidence.get("layers"), list):
        raise ValueError("Unsupported confidence evidence manifest")
    if output.is_relative_to(compact_manifest.parent) or output.is_relative_to(evidence_manifest.parent):
        raise ValueError("Export output must be outside input mask and confidence directories")
    entries = {}
    for entry in evidence["layers"]:
        key = (entry.get("model_name"), entry.get("layer_key"))
        if not all(isinstance(value, str) and value for value in key) or key in entries:
            raise ValueError("Confidence manifest identities must be nonempty and unique")
        entries[key] = entry
    watched = {path: _fingerprint(path) for path in (run_manifest, compact_manifest, evidence_manifest)}
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output.name}.export-"
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=output.parent))
    work = stage / ".workspace"
    work.mkdir()
    result = {"schema": EXPORT_SCHEMA, "status": "building", "source_shape_tyx": list(source_shape),
              "output_shape_tyx": list(target_shape), "downbin_scale": scale, "memory_plan": plan,
              "allow_native_projection": bool(allow_native_projection), "max_staging_mib": max_staging_mib,
              "layers": [], "confidence_layers": [], "source_inputs": {}}
    try:
        mask_output = stage / "nrrd"
        score_output = stage / "reconciliation_evidence"
        provenance_output = stage / "provenance"
        for directory in (mask_output, score_output, provenance_output):
            directory.mkdir()
        with read_layer_manifest(compact_manifest, workspace=work, memory_mib=memory_mib) as collection:
            geometry = collection.geometry
            if (geometry.space != "left-posterior-superior"
                    or not np.allclose(geometry.directions_xyz, np.eye(3), rtol=0, atol=1e-9)
                    or not np.allclose(geometry.origin_xyz, (0,0,0), rtol=0, atol=1e-9)):
                raise ValueError("Compact masks require the canonical complete source reference grid")
            refs, seen_prediction_keys = {}, set()
            for layer in collection:
                metadata = layer.metadata
                if metadata.get("downbin_scale", scale) != scale:
                    raise ValueError("Layer downbin scale disagrees with its manifest")
                if metadata.get("mask_kind") != "yolo":
                    continue
                key = (metadata.get("model_name"), metadata.get("layer_key"))
                if not all(isinstance(value, str) and value for value in key):
                    raise ValueError(f"Prediction layer lacks a stable model/layer identity: {layer.layer_id}")
                if key in seen_prediction_keys:
                    raise ValueError("Compact prediction identities must be unique")
                seen_prediction_keys.add(key)
                entry = entries.get(key)
                if entry is None:
                    if metadata.get("empty_segment") is True:
                        continue
                    raise ValueError(f"Retained confidence is missing for prediction layer {key}")
                directory = _inside(evidence_manifest.parent, entry["directory"], "Confidence directory")
                ref = ConfidenceEvidenceRef.open(directory)
                if (ref.model_name, ref.layer_key) != key or tuple(ref.shape) != source_shape:
                    raise ValueError(f"Confidence identity or source grid disagrees for {key}")
                if tuple(entry.get("output_shape_tyx", ref.shape)) != source_shape:
                    raise ValueError(f"Confidence manifest source shape disagrees for {key}")
                space = getattr(ref, "coordinate_space", "source")
                if space != "source" and not allow_native_projection and not metadata.get("empty_segment", False):
                    raise ValueError("Native-view confidence requires explicit --allow_native_projection and staging budget")
                if space not in {"source", "native_view_processing"}:
                    raise ValueError(f"Unsupported confidence coordinate space: {space}")
                if space == "source" and ref.metadata.get("exported_axes", "(X, Y, t)").replace(" ", "").lower() != "(x,y,t)":
                    raise ValueError("Confidence source axes disagree with compact masks")
                for path in directory.rglob("*"):
                    if path.is_file():
                        if not path.resolve().is_relative_to(directory):
                            raise ValueError("Confidence storage file escapes its directory")
                        watched[path] = _fingerprint(path)
                storage_metadata = [(directory, ref.metadata)]
                storage_metadata += [(path.parent, json.loads(path.read_text(encoding="utf-8")))
                                     for path in directory.rglob("metadata.json") if path.parent != directory]
                for base, description in storage_metadata:
                    for file_field, digest_field in (("payload", "payload_sha256"), ("index", "index_sha256")):
                        expected_digest = description.get(digest_field)
                        if expected_digest is not None:
                            stored = _inside(base, description[file_field], "Confidence storage path")
                            actual = watched.get(stored) or _fingerprint(stored)
                            if actual["sha256"] != expected_digest:
                                raise ValueError(f"Confidence storage checksum mismatch: {stored}")
                refs[layer.layer_id] = ref
            portable_layers = []
            for index, layer in enumerate(collection):
                metadata = dict(layer.metadata)
                destination = _inside(mask_output, metadata["filename"], "Output mask")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if layer.path is not None:
                    watched[layer.path] = _fingerprint(layer.path)
                    shutil.copyfile(layer.path, destination)
                    if _fingerprint(destination)["sha256"] != watched[layer.path]["sha256"]:
                        raise RuntimeError("Copied compact mask differs from its input")
                else:
                    write_seg_nrrd(destination, shape_tyx=target_shape, read_slab=layer.read_slab,
                                   geometry=geometry, segment_name=metadata.get("segment_name", layer.layer_id), memory_mib=memory_mib)
                    metadata["stored_shape_tyx"] = list(target_shape)
                if metadata.get("mask_kind") == "yolo":
                    key = (metadata["model_name"], metadata["layer_key"])
                    identity = hashlib.sha256((key[0]+"\0"+key[1]).encode()).hexdigest()
                    ref = refs.get(layer.layer_id)
                    is_empty = metadata.get("empty_segment", False)
                    native = ref is not None and getattr(ref, "coordinate_space", "source") != "source" and not is_empty
                    if native:
                        if not hasattr(ref, "source_reader"):
                            raise ValueError("Installed confidence reader cannot explicitly project native-view evidence")
                        reader_context = ref.source_reader(work / identity, memory_mib=memory_mib,
                                                           max_staging_mib=max_staging_mib)
                    else:
                        source_reader = None if ref is None or is_empty else ref.reader()
                        reader_context = closing(source_reader) if hasattr(source_reader, "close") else nullcontext(source_reader)
                    if native and progress:
                        progress("native projection", index+1, len(collection))
                    with reader_context as reader:
                        if reader is not None and tuple(reader.shape) != source_shape:
                            raise ValueError("Source confidence reader changed the advertised source grid")
                        def compact_plane(z):
                            mask = layer.read_slab(z, z+1)[0]
                            if reader is None:
                                if np.any(mask):
                                    raise ValueError("An empty prediction entry contains foreground without scores")
                                result_plane = np.zeros(target_shape[1:], np.uint8)
                            else:
                                result_plane = compact_score_plane(reader, mask, source_shape=source_shape, target_shape=target_shape, z=z)
                            if progress:
                                progress(f"layer {index+1} planes", z+1, target_shape[0])
                            return result_plane
                        written = write_confidence_evidence(score_output / identity, target_shape, compact_plane,
                            layer_key=key[1], model_name=key[0], provenance={
                                "export_schema": EXPORT_SCHEMA, "source_shape_tyx": list(source_shape),
                                "source_coordinate_space": getattr(ref, "coordinate_space", "source") if ref else "source",
                                "native_projection_applied": native, "source_run_sha256": watched[run_manifest]["sha256"],
                                "compact_mask_sha256": _fingerprint(destination)["sha256"],
                                "aggregation": "maximum over source-to-compact categorical footprints, intersected with compact mask",
                                "source_confidence_metadata": dict(ref.metadata) if ref else None})
                    result["confidence_layers"].append({"model_name":key[0],"layer_key":key[1],"directory":identity,
                        "output_shape_tyx":list(target_shape),"score_semantics":written.metadata["score_semantics"],
                        "unknown":"score_zero","coordinate_space":"source"})
                    result["layers"].append({"filename":metadata["filename"],"known_voxels":written.metadata["known_voxels"],
                                             "native_projection_applied":native,"model_name":key[0],"layer_key":key[1]})
                else:
                    for z in range(target_shape[0]):
                        layer.read_slab(z,z+1)
                    result["layers"].append({"filename":metadata["filename"],"confidence":"not a direct-prediction layer"})
                layer.close()
                metadata["confidence_export_schema"] = EXPORT_SCHEMA
                portable_layers.append(metadata)
                if progress:
                    progress("layers", index+1, len(collection))
            portable_manifest = {**compact,"layer_count":len(portable_layers),"layers":portable_layers}
            (mask_output / compact_manifest.name).write_text(json.dumps(portable_manifest,indent=2)+"\n",encoding="utf-8")
        (score_output / "manifest.json").write_text(json.dumps({"schema":SOURCE_SCHEMA,"layers":result["confidence_layers"]},indent=2)+"\n",encoding="utf-8")
        shutil.copyfile(run_manifest, provenance_output / "original_run_manifest.json")
        shutil.copyfile(compact_manifest, provenance_output / "original_compact_manifest.json")
        portable_run = {**run,"outputs":{"requested":["nrrd","confidence_evidence"],"paths":{
            "run_manifest":"manifest.json","nrrd_dir":"nrrd","nrrd_manifest":f"nrrd/{compact_manifest.name}",
            "confidence_evidence_manifest":"reconciliation_evidence/manifest.json"}},
            "compact_evidence_export":{"schema":EXPORT_SCHEMA,"source_shape_tyx":list(source_shape),
                "output_shape_tyx":list(target_shape),"original_manifest":"provenance/original_run_manifest.json",
                "input_paths_role":"historical source identities; replay artifacts use relative output paths"}}
        (stage / "manifest.json").write_text(json.dumps(portable_run,indent=2)+"\n",encoding="utf-8")
        (stage / "README.md").write_text(
            "# Portable compact reconciliation evidence\n\n"
            f"Compact grid (t, y, x): {target_shape}. Original source grid: {source_shape}.\n\n"
            f"Use `nrrd/{compact_manifest.name}` with `tools/compare_reconciliation.py` for offline policy replay. "
            "The package contains additive compact masks, matched source-aligned confidence sidecars, "
            "relative replay paths, and original metadata snapshots under `provenance/`.\n\n"
            "Confidence is the maximum known score over each source-to-compact categorical footprint, "
            "intersected with its compact binary mask. Zero remains unknown. "
            "Compact voting can differ from native voting followed by downsampling.\n\n"
            "Input and model paths retained in the run metadata describe historical source identities; "
            "the replay dependencies are the relative files listed under `outputs.paths`.\n", encoding="utf-8")
        for path, fingerprint in watched.items():
            if _fingerprint(path) != fingerprint:
                raise RuntimeError(f"Input changed during compact export: {path}")
        if work.exists():
            resolved_work = work.resolve()
            if resolved_work.parent != stage.resolve() or resolved_work.name != ".workspace":
                raise RuntimeError("Unsafe export workspace cleanup target")
            shutil.rmtree(resolved_work)
        result.update(status="complete", inputs_unchanged=True, layer_count=len(portable_layers),
                      confidence_layer_count=len(result["confidence_layers"]), paths={"run_manifest":"manifest.json",
                        "nrrd_manifest":f"nrrd/{compact_manifest.name}","confidence_manifest":"reconciliation_evidence/manifest.json"},
                      source_inputs={str(path): fingerprint for path,fingerprint in watched.items()})
        (stage / "export_manifest.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        if output.exists():
            raise FileExistsError("Export output appeared while conversion was running")
        os.rename(stage,output)
        return result
    finally:
        if stage.exists():
            _remove_stage(stage,output.parent,prefix)


__all__ = ["export_compact_evidence","compact_score_plane","max_crop_into","source_frames","export_memory_plan"]
