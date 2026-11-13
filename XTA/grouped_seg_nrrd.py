"""Bounded stock Slicer segmentation NRRD publication for overlapping segments.

Callbacks return full-reference C-order (t,y,x) binary slabs. The file uses
NRRD axes (x,y,t,list), one ordinary label=1 layer per logical segment.
No custom payload codec or inference-runtime imports are required.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .reconciliation_io import ReferenceGeometry, _budget, _shape
from .gzip_spans import GzipSpanWriter


_EMPTY_EXTENT = (0, -1, 0, -1, 0, -1)


@dataclass(frozen=True)
class GroupedSegment:
    """One segment on the shared reference grid.

    extent_xyz is an optional trusted inclusive upper bound in reference IJK.
    An empty extent declares an empty segment without calling read_slab.
    The callback must not allocate a larger source slab than requested.
    """
    segment_id: str
    name: str
    color_rgb: tuple[float, float, float]
    read_slab: Callable[[int, int], np.ndarray]
    tags: Mapping[str, str] = field(default_factory=dict)
    extent_xyz: tuple[int, int, int, int, int, int] | None = None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\0"):
        raise ValueError(f"{name} must be nonempty single-line text")
    return value


def _extent(value, shape):
    if value is None:
        return tuple(v for size in reversed(shape) for v in (0, size - 1))
    if (not isinstance(value, (tuple, list)) or len(value) != 6
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in value)):
        raise ValueError("extent_xyz must contain six integers")
    value = tuple(int(v) for v in value)
    empty = [value[2*i+1] < value[2*i] for i in range(3)]
    if all(empty):
        return _EMPTY_EXTENT
    if any(empty) or any(not 0 <= value[2*i] <= value[2*i+1] < size
                         for i, size in enumerate(reversed(shape))):
        raise ValueError("extent_xyz falls outside the reference grid")
    return value


def _validated_segments(segments, shape):
    if not isinstance(segments, Sequence) or not segments:
        raise ValueError("segments must be a nonempty sequence")
    ids = set()
    result = []
    for segment in segments:
        if not isinstance(segment, GroupedSegment):
            raise TypeError("segments must contain GroupedSegment values")
        sid = _text(segment.segment_id, "segment_id")
        _text(segment.name, "segment name")
        if sid in ids:
            raise ValueError(f"Duplicate segment_id: {sid}")
        ids.add(sid)
        if not callable(segment.read_slab):
            raise ValueError("read_slab must be callable")
        color = tuple(segment.color_rgb)
        if len(color) != 3 or any(isinstance(v, bool) or not math.isfinite(float(v))
                                   or not 0 <= float(v) <= 1 for v in color):
            raise ValueError("color_rgb must contain three finite values in [0,1]")
        if not isinstance(segment.tags, Mapping):
            raise ValueError("tags must be a mapping")
        tags = {}
        for key, value in segment.tags.items():
            _text(key, "tag key")
            if ":" in key or "|" in key:
                raise ValueError("tag keys cannot contain ':' or '|'")
            if not isinstance(value, str) or any(c in value for c in "|\r\n\0"):
                raise ValueError("tag values must be single-line strings without '|'")
            tags[key] = value
        result.append((segment, _extent(segment.extent_xyz, shape), color, tags))
    return result


def _union_crop(extents, shape_tyx):
    active = [extent for extent in extents if extent != _EMPTY_EXTENT]
    if not active:
        return (1, 1, 1), (0, 0, 0)
    lo = [min(e[2*i] for e in active) for i in range(3)]
    hi = [max(e[2*i+1] for e in active) for i in range(3)]
    # Reserve the conventional one-voxel image for empty segmentations when
    # the source has a neighbor. The segment extent itself remains a singleton.
    if lo == hi:
        for axis, size in enumerate(reversed(shape_tyx)):
            if size > 1:
                if hi[axis] + 1 < size:
                    hi[axis] += 1
                else:
                    lo[axis] -= 1
                break
    return tuple(hi[i]-lo[i]+1 for i in (2, 1, 0)), tuple(lo)


def _slab_extent(slab, z0, foreground):
    if not foreground:
        return _EMPTY_EXTENT
    bounds = []
    for axis in (2, 1, 0):
        projection = np.any(slab, axis=tuple(i for i in range(3) if i != axis))
        lo = int(np.argmax(projection))
        hi = len(projection) - 1 - int(np.argmax(projection[::-1]))
        if axis == 0:
            lo += z0
            hi += z0
        bounds.extend((lo, hi))
        del projection
    return tuple(bounds)


def _merge_extent(first, second):
    if first == _EMPTY_EXTENT:
        return second
    if second == _EMPTY_EXTENT:
        return first
    return tuple(v for i in range(3)
                 for v in (min(first[2*i], second[2*i]), max(first[2*i+1], second[2*i+1])))


def _vector(values):
    return "(" + ",".join(format(float(value), ".17g") for value in values) + ")"


def _header(shape, stored_shape, offset_xyz, geometry, records):
    origin = np.asarray(geometry.origin_xyz) + np.asarray(offset_xyz) @ np.asarray(geometry.directions_xyz)
    fields = [
        "NRRD0005", "# Stock overlapping binary segmentation generated by XTA",
        "type: uint8", "dimension: 4",
        "sizes: " + " ".join(str(v) for v in (*reversed(stored_shape), len(records))),
        "space: " + geometry.space, "kinds: domain domain domain list",
        "space directions: " + " ".join(_vector(v) for v in geometry.directions_xyz) + " none",
        "space origin: " + _vector(origin), "encoding: gzip",
        "Segmentation_ContainedRepresentationNames:=Binary labelmap|",
        "Segmentation_MasterRepresentation:=Binary labelmap",
        "Segmentation_SourceRepresentation:=Binary labelmap",
        "Segmentation_ReferenceImageExtent:=" + " ".join(str(v) for n in reversed(shape) for v in (0, n-1)),
        "Segmentation_ReferenceImageExtentOffset:=" + " ".join(str(v) for v in offset_xyz),
    ]
    for index, record in enumerate(records):
        p = f"Segment{index}_"
        fields.extend([
            p + "ID:=" + record["segment_id"], p + "Name:=" + record["name"],
            p + "NameAutoGenerated:=0",
            p + "Color:=" + " ".join(format(v, ".17g") for v in record["color_rgb"]),
            p + "ColorAutoGenerated:=0", p + "LabelValue:=1", p + "Layer:=" + str(index),
            p + "Extent:=" + " ".join(str(v) for v in record["stored_extent_xyz"]),
        ])
        if record["tags"]:
            fields.append(p + "Tags:=" + "".join(f"{key}:{value}|" for key, value in record["tags"].items()))
    return ("\n".join(fields) + "\n\n").encode("utf-8")


def write_grouped_seg_nrrd(
    path: str | Path, *, shape_tyx: tuple[int, int, int],
    segments: Sequence[GroupedSegment], geometry: ReferenceGeometry | None = None,
    memory_mib: float = 64, chunk_slices: int = 16, overwrite: bool = False,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write overlapping ordinary segments with one source pass and bounded slabs.

    The numeric buffer budget admits a source slab, normalization/validation
    workspace, and bounded canonical-zero construction/write buffers. Codec control state and O(K) metadata
    are separate fixed overhead. Callbacks may borrow pre-existing source arrays
    but must not create a larger new slab than requested.

    Trusted extent bounds permit cropping and omission of known-empty source
    slices. Values outside those unrequested bounds are the caller's contract.
    Source data returned within a request is checked against its declared bound.
    The compressed spool uses temp_dir (system temporary storage by default);
    only the final assembled temporary file is placed beside its destination.
    """
    shape = _shape(shape_tyx, "shape_tyx")
    budget = _budget(memory_mib)
    geometry = geometry or ReferenceGeometry(shape)
    if geometry.shape_tyx != shape:
        raise ValueError("Writer shape disagrees with spatial reference")
    if isinstance(chunk_slices, bool) or not isinstance(chunk_slices, int) or chunk_slices <= 0:
        raise ValueError("chunk_slices must be a positive integer")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be bool")
    descriptors = _validated_segments(segments, shape)
    stored_shape, offset_xyz = _union_crop([extent for _, extent, _, _ in descriptors], shape)
    per_slice = math.prod(shape[1:]) if any(bound != _EMPTY_EXTENT for _, bound, _, _ in descriptors) else 1
    zero_bytes = min(64 * 1024, per_slice)
    minimum = 3 * per_slice + zero_bytes
    depth = min(chunk_slices, shape[0], (budget - zero_bytes) // max(1, 3 * per_slice))
    if depth < 1:
        raise ValueError(f"memory_mib must admit at least {minimum} numeric buffer bytes for one reference slice")
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_root = None if temp_dir is None else Path(temp_dir).resolve()
    if payload_root is not None:
        payload_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".grouped-seg-write-", dir=path.parent))
    final_stage = stage_dir / "output.seg.nrrd"
    payload_dir = None
    zero_member_bytes = 1 << (min(1024*1024, max(1, budget//8)).bit_length()-1)
    zero_write_buffer_bytes = min(64*1024, max(1, budget//16))
    crop_x, crop_y, crop_z = offset_xyz
    stored_z, stored_y, stored_x = stored_shape
    plane_bytes = stored_y * stored_x
    records = []

    try:
        payload_dir = Path(tempfile.mkdtemp(prefix=".grouped-seg-payload-", dir=payload_root))
        payload_path = payload_dir / "payload.gz"
        with payload_path.open("wb") as raw, GzipSpanWriter(
            raw, compresslevel=3, zero_member_bytes=zero_member_bytes,
            write_buffer_bytes=zero_write_buffer_bytes,
        ) as compressed:
            for segment, bound, color, tags in descriptors:
                foreground, exact = 0, _EMPTY_EXTENT
                if bound == _EMPTY_EXTENT:
                    first = stop = crop_z
                else:
                    first, stop = max(crop_z, bound[4]), min(crop_z+stored_z, bound[5]+1)
                compressed.write_zeros((first-crop_z) * plane_bytes)
                for z0 in range(first, stop, depth):
                    z1 = min(stop, z0+depth)
                    slab = np.asarray(segment.read_slab(z0, z1))
                    if slab.shape != (z1-z0, *shape[1:]) or slab.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
                        raise ValueError("read_slab must return the requested uint8 or bool reference TYX slab")
                    if np.any(slab > 1):
                        raise ValueError("Grouped segments require binary values 0/1")
                    count = int(np.count_nonzero(slab))
                    extent = _slab_extent(slab, z0, count)
                    if count and any(extent[2*i] < bound[2*i] or extent[2*i+1] > bound[2*i+1] for i in range(3)):
                        raise ValueError(f"Source data exceeds extent_xyz for {segment.segment_id}")
                    exact = _merge_extent(exact, extent)
                    foreground += count
                    if count:
                        payload = np.ascontiguousarray(
                            slab[:, crop_y:crop_y+stored_y, crop_x:crop_x+stored_x], dtype=np.uint8,
                        )
                        compressed.write(memoryview(payload).cast("B"))
                        del payload
                    else:
                        compressed.write_zeros((z1-z0) * plane_bytes)
                    del slab
                compressed.write_zeros((crop_z+stored_z-stop) * plane_bytes)
                local_extent = (_EMPTY_EXTENT if exact == _EMPTY_EXTENT else
                                tuple(v-offset_xyz[i] for i in range(3) for v in exact[2*i:2*i+2]))
                records.append({
                    "segment_id": segment.segment_id, "name": segment.name,
                    "color_rgb": [float(v) for v in color], "tags": tags,
                    "layer": len(records), "label_value": 1,
                    "extent_xyz": list(exact), "stored_extent_xyz": list(local_extent),
                    "foreground_voxels": foreground, "empty_segment": foreground == 0,
                })
        header = _header(shape, stored_shape, offset_xyz, geometry, records)
        with final_stage.open("wb") as destination, payload_path.open("rb") as payload:
            destination.write(header)
            shutil.copyfileobj(payload, destination, length=min(1024*1024, budget))
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(final_stage, path)
        return {
            "path": str(path), "shape_tyx": list(shape), "stored_shape_tyx": list(stored_shape),
            "offset_xyz": list(offset_xyz), "segment_count": len(records), "segments": records,
            "numeric_buffer_budget_bytes": budget, "minimum_numeric_buffer_bytes": minimum,
            "chunk_slices": depth, "payload_uncompressed_bytes": math.prod(stored_shape)*len(records),
            "gzip_spans": dict(compressed.stats),
            "bytes": path.stat().st_size,
        }
    finally:
        if payload_dir is not None:
            (payload_dir / "payload.gz").unlink(missing_ok=True)
            payload_dir.rmdir()
        final_stage.unlink(missing_ok=True)
        stage_dir.rmdir()
