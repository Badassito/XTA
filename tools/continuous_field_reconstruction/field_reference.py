"""Portable complete-object scalar reference for the guarded cubic upscaler.

Only the complete foreground bounding box plus three zero context cells is
materialized. All EDT feature transforms and Lee skeletonization are global on
that complete context, never independently reconstructed from output blocks.
``memory_budget_mib`` is a conservative allocation preflight, not an OS RSS cap.
The caller's existing preview bytes are included in the estimate.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np

SCHEMA = "pta-radius-reference-v1"
PADDING_CELLS = 3
_MIB = 1024 ** 2
_RUNTIME_RESERVE = 192 * _MIB
_WORKSPACE_BYTES = 32 * _MIB


class FieldReferenceMemoryError(MemoryError):
    """The exact global build needs a larger declared resource budget."""

    def __init__(self, estimate, budget_bytes):
        self.required_memory_estimate = estimate
        self.budget_bytes = int(budget_bytes)
        needed = estimate["bytes"] / _MIB
        super().__init__(
            f"Exact global field build estimates {needed:.1f} MiB, above the "
            f"{budget_bytes / _MIB:.1f} MiB budget. Raise memory_budget_mib on a "
            "suitable machine; local-block EDT/skeleton substitution is not supported."
        )


def _shape3(value, name):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three positive integer TYX dimensions") from exc
    if (len(values) != 3 or any(isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in values)):
        raise ValueError(f"{name} must contain three positive integer TYX dimensions")
    return tuple(int(v) for v in values)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * _MIB), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _blocks(shape, max_voxels=_WORKSPACE_BYTES // 64):
    """Bound temporary buffers even when one TYX plane exceeds the workspace."""
    width = min(shape[2], max_voxels)
    height = 1 if width < shape[2] else max(1, min(shape[1], max_voxels // width))
    for t in range(shape[0]):
        for y in range(0, shape[1], height):
            for x in range(0, shape[2], width):
                yield (slice(t, t + 1), slice(y, min(shape[1], y + height)),
                       slice(x, min(shape[2], x + width)))


def _scan(preview):
    """Validate without a complete cast/copy and hash canonical binary C order."""
    shape = preview.shape
    lo = np.array(shape, np.int64)
    hi = np.zeros(3, np.int64)
    foreground = 0
    h = hashlib.sha256()
    # These blocks preserve C-order byte concatenation: rows are grouped only
    # when the complete X axis fits the block.
    for block in _blocks(shape, max_voxels=1 << 20):
        a = preview[block]
        if np.any((a != 0) & (a != 1)):
            raise ValueError("preview must contain only finite binary values 0 and 1")
        binary = np.asarray(a != 0, np.uint8)
        h.update(binary.tobytes(order="C"))
        count = int(np.count_nonzero(binary))
        foreground += count
        if count:
            for axis in range(3):
                projection = np.any(binary, axis=tuple(k for k in range(3) if k != axis))
                occupied = np.flatnonzero(projection)
                lo[axis] = min(lo[axis], block[axis].start + int(occupied[0]))
                hi[axis] = max(hi[axis], block[axis].start + int(occupied[-1]) + 1)
    return foreground, lo, hi, h.hexdigest()


def _estimate(n, input_bytes, skeleton_count=None):
    # Includes public SciPy EDT's int64 binary-conversion temporary, int8 input,
    # indices memmap working set and a conservative runtime/slab allowance.
    phases = {
        "global_feature_transform": 24 * n + input_bytes + _RUNTIME_RESERVE,
        "blockwise_inside_distance": 21 * n + input_bytes + _RUNTIME_RESERVE + _WORKSPACE_BYTES,
    }
    if skeleton_count is not None:
        phases["sparse_cap_lookup"] = (21 * n + 16 * skeleton_count + input_bytes
                                       + _RUNTIME_RESERVE + _WORKSPACE_BYTES)
    required = max(phases.values())
    return {"bytes": int(required), "mib": required / _MIB,
            "input_resident_bytes_charged": int(input_bytes), "coefficient_voxels": int(n),
            "runtime_reserve_bytes": _RUNTIME_RESERVE, "workspace_bytes": _WORKSPACE_BYTES,
            "skeleton_count": skeleton_count,
            "phase_estimates_bytes": {k: int(v) for k, v in phases.items()},
            "policy": "Conservative allocation estimate including caller preview; not an OS RSS limit. Exact global EDT and Lee reference only."}


def _require_budget(estimate, budget_bytes):
    if estimate["bytes"] > budget_bytes:
        raise FieldReferenceMemoryError(estimate, budget_bytes)


@contextmanager
def _mapped(path, *, dtype=None, shape=None, mode="r"):
    if mode == "w+":
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    else:
        array = np.load(path, mmap_mode=mode, allow_pickle=False)
    try:
        yield array
    finally:
        if mode != "r":
            array.flush()
        memory = getattr(array, "_mmap", None)
        if memory is not None:
            memory.close()


def _feature_transform(binary, path, shape):
    from scipy import ndimage as ndi
    with _mapped(path, mode="w+", dtype=np.int32, shape=(3, *shape)) as indices:
        ndi.distance_transform_edt(binary, return_distances=False,
                                  return_indices=True, indices=indices)


def _distance_block(indices, block):
    """Reproduce SciPy's int32 subtraction/float64 square-reduce-sqrt order."""
    shape = tuple(s.stop - s.start for s in block)
    coordinates = np.indices(shape, dtype=np.int32)
    for axis, s in enumerate(block):
        coordinates[axis] += s.start
    delta = indices[(slice(None), *block)] - coordinates
    del coordinates
    d = delta.astype(np.float64)
    del delta
    np.multiply(d, d, out=d)
    result = np.add.reduce(d, axis=0)
    del d
    np.sqrt(result, out=result)
    return result


def _delete_work(path, stage):
    """Delete only a private path in this build's newly created staging tree."""
    path, stage = Path(path).resolve(), Path(stage).resolve()
    if path.parent != stage and not path.is_relative_to(stage / "_work"):
        raise RuntimeError("Refusing cleanup outside private field-build staging directory")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def build_field_reference(preview, native_shape_tyx, cache_dir, *,
                          memory_budget_mib=2048, progress=None):
    """Build and atomically publish one portable complete-object field cache.

    ``preview`` is a3D ndarray containing binary0/1 values. It is never mutated.
    ``progress`` optionally receives dictionaries with stage/status metadata.
    Existing cache destinations are always rejected. Empty input produces an
    explicit constant-zero manifest, without undefined empty EDT arrays.
    """
    if not isinstance(preview, np.ndarray) or preview.ndim != 3:
        raise ValueError("preview must be a3D binary numpy ndarray")
    low_shape = _shape3(preview.shape, "preview.shape")
    source_dtype = preview.dtype.str
    native_shape = _shape3(native_shape_tyx, "native_shape_tyx")
    if preview.dtype.kind not in "buif":
        raise ValueError("preview must have a real binary numeric/bool dtype")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if (isinstance(memory_budget_mib, bool) or not isinstance(memory_budget_mib, (int, float))
            or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0):
        raise ValueError("memory_budget_mib must be finite and positive")
    budget_bytes = int(memory_budget_mib * _MIB)
    output = Path(cache_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Field cache already exists; nothing overwritten: {output}")
    started = time.monotonic()

    def notify(stage, status, **extra):
        if progress is not None:
            progress({"stage": stage, "status": status,
                      "elapsed_seconds": time.monotonic() - started, **extra})

    notify("validate", "started")
    foreground, lo, hi, source_hash = _scan(preview)
    if preview.shape != low_shape or preview.dtype.str != source_dtype:
        raise RuntimeError("preview geometry/dtype changed during validation")
    field_shape = tuple(int(v) for v in (hi - lo + 2 * PADDING_CELLS)) if foreground else None
    n = math.prod(field_shape) if field_shape else 0
    if field_shape and max(field_shape) > np.iinfo(np.int32).max:
        raise ValueError("Exact EDT coefficient dimensions exceed int32 index range")
    estimate = _estimate(n, preview.nbytes)
    _require_budget(estimate, budget_bytes)
    notify("validate", "completed", foreground_voxels=foreground,
           required_memory_estimate=estimate)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".field-build.lock")
    token = uuid.uuid4().hex
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, json.dumps({"pid": os.getpid(), "token": token,
                            "started_utc": datetime.now(timezone.utc).isoformat()}).encode())
    os.close(fd)
    stage = None
    published = False
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Field cache already exists; nothing overwritten: {output}")
        required_disk = n * 36 + 16 * min(foreground, n) + 8 * _MIB
        if shutil.disk_usage(output.parent).free < required_disk:
            raise OSError(f"Exact field build needs about {required_disk} free temporary bytes")
        stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".field-build-", dir=output.parent))
        work = stage / "_work"
        work.mkdir()
        from .policy import POLICY, ALGORITHM_VERSION, package_source_identity
        package_sources = package_source_identity()
        manifest = {
            "schema": SCHEMA, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": POLICY, "algorithm_version": ALGORITHM_VERSION,
            "package_sources": package_sources,
            "empty_foreground": foreground == 0, "constant_mask": 0 if not foreground else None,
            "constant_scalar": -1.0 if not foreground else None,
            "low_shape_tyx": list(low_shape), "native_shape_tyx": list(native_shape),
            "source_dtype": str(np.dtype(source_dtype)), "source_foreground_voxels": foreground,
            "source_binary_uint8_C_sha256": source_hash,
            "source_foreground_bounds_tyx": [[int(a), int(b)] for a, b in zip(lo, hi)] if foreground else None,
            "origin_tyx": [int(v) for v in lo - PADDING_CELLS] if foreground else [0, 0, 0],
            "shape_tyx": list(field_shape) if field_shape else [0, 0, 0],
            "padding_cells": PADDING_CELLS,
            "exterior_policy": "Zero geometric support outside original source and complete foreground; padding is not negative annotation authority.",
            "coefficient_scope": "Complete foreground bbox plus3zero cells; no output-block-local field reconstruction.",
            "phi_definition": "EDT(mask)-0.5 inside;0.5-EDT(~mask) outside; PREVIEW index units.",
            "cap_definition": "min(0.5,0.2*max(nearest_global_Lee_skeleton_EDT-0.5,0)); same6CC only; undefined0; nearest original foreground scalar extension.",
            "required_memory_estimate": estimate, "memory_budget_mib": float(memory_budget_mib),
            "required_temporary_disk_estimate_bytes": int(required_disk),
            "fields": {}, "builder_sha256": _digest(Path(__file__)),
        }
        if foreground:
            _build_nonempty(preview, lo, hi, field_shape, stage, estimate, budget_bytes, notify, manifest)
        _delete_work(work, stage)
        for name, dtype in (("mask", "bool"), ("phi", "float32"), ("cap_extended", "float32")) if foreground else ():
            path = stage / (name + ".npy")
            manifest["fields"][name] = {"file": path.name, "dtype": dtype,
                                        "shape_tyx": list(field_shape), "bytes": path.stat().st_size,
                                        "sha256": _digest(path)}
        # Run the last user callback before final validation/publication. Never
        # hand a mutable manifest to progress code after it has been written.
        notify("publish", "started")
        # Detect caller changes before publishing. Validation/hash remain bounded.
        after = _scan(preview)
        if (preview.shape != low_shape or preview.dtype.str != source_dtype
                or after[0] != foreground or after[3] != source_hash):
            raise RuntimeError("preview changed during field construction; no cache published")
        manifest["elapsed_seconds"] = time.monotonic() - started
        _json(stage / "COMPLETE.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Field cache appeared during build; nothing overwritten: {output}")
        os.rename(stage, output)
        published = True
        return {**manifest, "cache_dir": str(output.resolve())}
    finally:
        if stage is not None and not published and stage.exists():
            resolved = stage.resolve()
            if resolved.parent != output.parent.resolve() or not resolved.name.startswith("." + output.name + ".field-build-"):
                raise RuntimeError("Refusing cleanup outside owned field-build staging directory")
            shutil.rmtree(resolved)
        lock.unlink(missing_ok=True)


def validate_field_reference(preview, native_shape_tyx, cache_dir):
    """Validate explicit cache reuse against the currently supplied source array.

    No cache files or source values are modified. Data/geometry/recipe mismatches
    raise; code and library versions are retained as provenance, not reuse locks.
    """
    from .policy import POLICY, ALGORITHM_VERSION, package_source_identity
    if not isinstance(preview, np.ndarray) or preview.ndim != 3 or preview.dtype.kind not in "buif":
        raise ValueError("preview must be a3D real binary numpy ndarray")
    low = _shape3(preview.shape, "preview.shape")
    native = _shape3(native_shape_tyx, "native_shape_tyx")
    path = Path(cache_dir).expanduser().resolve()
    manifest_path = path / "COMPLETE.json"
    if not manifest_path.is_file():
        raise ValueError("Field cache has no COMPLETE.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "complete"
            or manifest.get("policy") != POLICY or manifest.get("algorithm_version") != ALGORITHM_VERSION):
        raise ValueError("Field cache recipe differs")
    foreground, lo, hi, source_hash = _scan(preview)
    if (manifest.get("low_shape_tyx") != list(low) or manifest.get("native_shape_tyx") != list(native)
            or manifest.get("source_binary_uint8_C_sha256") != source_hash
            or manifest.get("source_foreground_voxels") != foreground
            or manifest.get("empty_foreground") != (foreground == 0)):
        raise ValueError("Field cache source/native geometry identity differs")
    bounds = [[int(a), int(b)] for a, b in zip(lo, hi)] if foreground else None
    if manifest.get("source_foreground_bounds_tyx") != bounds:
        raise ValueError("Field cache foreground bounds differ")
    expected = {"COMPLETE.json"}
    names = {"mask", "phi", "cap_extended"} if foreground else set()
    if set(manifest.get("fields", {})) != names:
        raise ValueError("Field cache coefficient inventory differs")
    if foreground:
        shape = tuple(int(v) for v in hi - lo + 2 * PADDING_CELLS)
        if manifest.get("shape_tyx") != list(shape) or manifest.get("origin_tyx") != [int(v) for v in lo-PADDING_CELLS]:
            raise ValueError("Field cache coefficient geometry differs")
        for name, dtype in (("mask", np.bool_), ("phi", np.float32), ("cap_extended", np.float32)):
            record = manifest["fields"][name]
            filename = record.get("file")
            if filename != name + ".npy":
                raise ValueError("Unexpected field filename")
            field = path / filename
            expected.add(filename)
            if (not field.is_file() or field.is_symlink() or field.stat().st_size != record.get("bytes")
                    or _digest(field) != record.get("sha256")):
                raise ValueError("Field cache coefficient identity differs: " + name)
            with _mapped(field) as array:
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise ValueError("Field cache coefficient dtype/shape differs: " + name)
    if {p.name for p in path.iterdir()} != expected:
        raise ValueError("Field cache contains unexpected files")
    return {**manifest, "cache_dir": str(path)}


def _build_nonempty(preview, lo, hi, shape, stage, estimate, budget_bytes, notify, manifest):
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    import scipy
    import skimage
    work = stage / "_work"
    mask_path = stage / "mask.npy"
    features_path = work / "features.npy"
    inside_path = work / "inside_distance.npy"
    skeleton_path = work / "skeleton.npy"
    components_path = work / "components.npy"
    cap_inside_path = work / "cap_inside.npy"
    sparse_flat_path = work / "skeleton_flat.npy"
    sparse_cap_path = work / "skeleton_cap.npy"
    sparse_component_path = work / "skeleton_component.npy"
    n = math.prod(shape)
    manifest["libraries"] = {"numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__}
    notify("mask", "started")
    with _mapped(mask_path, mode="w+", dtype=np.bool_, shape=shape) as mask:
        mask[:] = False
        target = tuple(slice(PADDING_CELLS, PADDING_CELLS + int(b-a)) for a,b in zip(lo,hi))
        # Source conversion is blockwise, including noncontiguous ndarray inputs.
        crop_shape = tuple(int(b-a) for a,b in zip(lo,hi))
        for block in _blocks(crop_shape):
            src = tuple(slice(int(a)+s.start, int(a)+s.stop) for a,s in zip(lo,block))
            dst = tuple(slice(PADDING_CELLS+s.start, PADDING_CELLS+s.stop) for s in block)
            mask[dst] = preview[src] != 0
    notify("mask", "completed")

    notify("inside", "started")
    with _mapped(mask_path) as mask:
        _feature_transform(mask, features_path, shape)
    with _mapped(features_path) as features, _mapped(inside_path, mode="w+", dtype=np.float64, shape=shape) as inside:
        for block in _blocks(shape):
            inside[block] = _distance_block(features, block)
    _delete_work(features_path, stage)
    gc.collect()
    notify("inside", "completed")

    notify("skeleton", "started")
    with _mapped(mask_path) as mask:
        skeleton = np.asarray(skeletonize(mask, method="lee"), dtype=np.bool_)
        with _mapped(skeleton_path, mode="w+", dtype=np.bool_, shape=shape) as saved:
            saved[:] = skeleton
        skeleton_count = int(np.count_nonzero(skeleton))
        del skeleton
    refined_estimate = _estimate(n, preview.nbytes, skeleton_count)
    _require_budget(refined_estimate, budget_bytes)
    manifest["required_memory_estimate"] = refined_estimate
    notify("skeleton", "completed", skeleton_voxels=skeleton_count,
           required_memory_estimate=refined_estimate)

    notify("components", "started")
    with _mapped(mask_path) as mask, _mapped(components_path, mode="w+", dtype=np.int32, shape=shape) as labels:
        component_count = int(ndi.label(mask, structure=ndi.generate_binary_structure(3, 1), output=labels))
    has_skeleton = np.zeros(component_count + 1, dtype=np.bool_)
    with _mapped(skeleton_path) as sk, _mapped(components_path) as labels:
        for block in _blocks(shape):
            has_skeleton[np.asarray(labels[block])[np.asarray(sk[block])]] = True
    missing = np.flatnonzero(~has_skeleton[1:]) + 1
    manifest["reference_diagnostics"] = {
        "components6": component_count, "skeleton_voxels": skeleton_count,
        "components_without_skeleton_count": int(missing.size),
        "components_without_skeleton_examples": [int(v) for v in missing[:100]],
        "skeleton_method": "Lee", "radius_validity": "Nearest GLOBAL skeleton must belong to same6CC; no farther same-component fallback.",
    }
    del missing, has_skeleton
    notify("components", "completed", components6=component_count)

    if skeleton_count:
        notify("sparse_skeleton_caps", "started")
        with _mapped(sparse_flat_path, mode="w+", dtype=np.int64, shape=(skeleton_count,)) as flats, \
                _mapped(sparse_cap_path, mode="w+", dtype=np.float32, shape=(skeleton_count,)) as caps, \
                _mapped(sparse_component_path, mode="w+", dtype=np.int32, shape=(skeleton_count,)) as owners, \
                _mapped(skeleton_path) as sk, _mapped(inside_path) as inside, _mapped(components_path) as labels:
            position = 0
            # Flat contiguous chunks guarantee sorted global flat indices for
            # exact searchsorted lookup after the global skeleton FT.
            for start in range(0, n, _WORKSPACE_BYTES // 64):
                stop = min(n, start + _WORKSPACE_BYTES // 64)
                local = np.flatnonzero(sk.reshape(-1)[start:stop])
                indices = local + start
                stop_position = position + len(indices)
                flats[position:stop_position] = indices
                distance = inside.reshape(-1)[indices]
                caps[position:stop_position] = np.minimum(.5, .2*np.maximum(distance-.5, 0)).astype(np.float32)
                owners[position:stop_position] = labels.reshape(-1)[indices]
                position = stop_position
            assert position == skeleton_count
        notify("sparse_skeleton_caps", "completed")
        notify("cap", "started")
        with _mapped(skeleton_path) as sk:
            inverse = np.logical_not(sk)
            _feature_transform(inverse, features_path, shape)
            del inverse
        valid_count = 0
        with _mapped(features_path) as features, _mapped(components_path) as labels, _mapped(mask_path) as mask, \
                _mapped(sparse_flat_path) as flats, _mapped(sparse_cap_path) as caps, _mapped(sparse_component_path) as owners, \
                _mapped(cap_inside_path, mode="w+", dtype=np.float32, shape=shape) as cap_inside:
            for block in _blocks(shape):
                near = features[(slice(None), *block)]
                nearest_flat = (near[0].astype(np.int64)*shape[1] + near[1])*shape[2] + near[2]
                positions = np.searchsorted(flats, nearest_flat)
                if np.any(positions >= skeleton_count) or not np.array_equal(flats[positions], nearest_flat):
                    raise RuntimeError("Global skeleton FT did not resolve to original skeleton coordinates")
                own = labels[block]
                valid = mask[block] & (own != 0) & (owners[positions] == own)
                cap_inside[block] = np.where(valid, caps[positions], 0)
                valid_count += int(np.count_nonzero(valid))
        _delete_work(features_path, stage)
    else:
        notify("cap", "started")
        valid_count = 0
        with _mapped(cap_inside_path, mode="w+", dtype=np.float32, shape=shape) as cap_inside:
            for block in _blocks(shape):
                cap_inside[block] = 0
    manifest["reference_diagnostics"].update(valid_foreground_radius=valid_count,
        undefined_foreground_radius=manifest["source_foreground_voxels"]-valid_count)
    gc.collect()
    notify("cap", "completed")

    # Initialize positive phi before the nearest-foreground FT so no full
    # inside-distance mmap must stay resident alongside that feature transform.
    notify("phi_initialize", "started")
    with _mapped(inside_path) as inside, _mapped(mask_path) as mask, \
            _mapped(stage/"phi.npy", mode="w+", dtype=np.float32, shape=shape) as phi:
        for block in _blocks(shape):
            phi[block] = np.where(mask[block], inside[block]-.5, 0).astype(np.float32)
    for path in (inside_path, skeleton_path, components_path, sparse_flat_path,
                 sparse_cap_path, sparse_component_path):
        _delete_work(path, stage)
    notify("phi_initialize", "completed")

    notify("nearest_foreground", "started")
    with _mapped(mask_path) as mask:
        inverse = np.logical_not(mask)
        _feature_transform(inverse, features_path, shape)
        del inverse
    notify("nearest_foreground", "completed")
    notify("phi", "started")
    with _mapped(mask_path) as mask, _mapped(features_path) as features, _mapped(stage/"phi.npy", mode="r+") as phi:
        for block in _blocks(shape):
            outside = _distance_block(features, block)
            selected = ~mask[block]
            view = phi[block]
            view[selected] = (.5-outside[selected]).astype(np.float32)
    notify("phi", "completed")
    notify("extended_cap", "started")
    mismatches = 0
    with _mapped(mask_path) as mask, _mapped(features_path) as features, _mapped(cap_inside_path) as cap_inside, \
            _mapped(stage/"cap_extended.npy", mode="w+", dtype=np.float32, shape=shape) as extended:
        for block in _blocks(shape):
            near = tuple(features[(slice(None), *block)])
            values = cap_inside[near]
            extended[block] = values
            selected = mask[block]
            mismatches += int(np.count_nonzero(values[selected] != cap_inside[block][selected]))
    if mismatches:
        raise RuntimeError("Foreground cap changed during nearest-original-foreground extension")
    manifest["reference_diagnostics"]["extension_inside_mismatches"] = mismatches
    gc.collect()
    notify("extended_cap", "completed")
