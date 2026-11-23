"""Bounded stock Slicer segmentation NRRD publication for overlapping segments.

Callbacks return (t,y,x) binary slabs on the reference grid or a declared XY crop. The file uses
NRRD axes (x,y,t,list), one ordinary label=1 layer per logical segment.
No custom payload codec or inference-runtime imports are required.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .reconciliation_io import FileLayer, ReferenceGeometry, _budget, _shape
from .gzip_spans import GzipSpanWriter


_EMPTY_EXTENT = (0, -1, 0, -1, 0, -1)


@dataclass(frozen=True)
class GroupedSegment:
    """One segment on the shared reference grid.

    extent_xyz is an optional trusted inclusive upper bound in reference IJK.
    An empty extent declares an empty segment without calling read_slab.
    Provide either a callback or a FileLayer. A callback must not allocate a
    larger source slab than requested. The observer receives validated slabs.
    """
    segment_id: str
    name: str
    color_rgb: tuple[float, float, float]
    read_slab: Callable[[int, int], np.ndarray] | None
    tags: Mapping[str, str] = field(default_factory=dict)
    extent_xyz: tuple[int, int, int, int, int, int] | None = None
    # Optional reference coordinates (y0, x0, y1, x1), upper bounds exclusive.
    source_window_yx: tuple[int, int, int, int] | None = None
    source_layer: FileLayer | None = None
    observe_slab: Callable[[int, np.ndarray], None] | None = None


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
        source = segment.source_layer
        if source is None:
            if not callable(segment.read_slab):
                raise ValueError("read_slab must be callable")
            source_window = (0, 0, *shape[1:])
        else:
            if not isinstance(source, FileLayer) or segment.read_slab is not None or source.shape_tyx != shape:
                raise ValueError("source_layer must be a FileLayer on the reference grid with no read_slab")
            _, y, x = source.offset_tyx
            _, height, width = source.stored_shape_tyx
            source_window = (y, x, y + height, x + width)
        if segment.observe_slab is not None and not callable(segment.observe_slab):
            raise ValueError("observe_slab must be callable")
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
        window = segment.source_window_yx
        if window is None:
            window = source_window
        if (not isinstance(window, (tuple, list)) or len(window) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in window)
                or not 0 <= window[0] < window[2] <= shape[1] or not 0 <= window[1] < window[3] <= shape[2]):
            raise ValueError("source_window_yx must be a nonempty crop inside the reference plane")
        if source is not None and tuple(window) != source_window:
            raise ValueError("source_window_yx disagrees with source_layer")
        result.append((segment, _extent(segment.extent_xyz, shape), color, tags, tuple(map(int, window))))
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


def _slab_extent(slab, origin_tyx, foreground):
    if not foreground:
        return _EMPTY_EXTENT
    bounds = []
    for axis in (2, 1, 0):
        projection = np.any(slab, axis=tuple(i for i in range(3) if i != axis))
        lo = int(np.argmax(projection))
        hi = len(projection) - 1 - int(np.argmax(projection[::-1]))
        bounds.extend((lo + origin_tyx[axis], hi + origin_tyx[axis]))
        del projection
    return tuple(bounds)


def _merge_extent(first, second):
    if first == _EMPTY_EXTENT:
        return second
    if second == _EMPTY_EXTENT:
        return first
    return tuple(v for i in range(3)
                 for v in (min(first[2*i], second[2*i]), max(first[2*i+1], second[2*i+1])))


def _group_slab(slab, window, crop_y, crop_x, height, width):
    """Place a verified source slab directly into the group's shared XY grid."""
    y0, x0, y1, x1 = window
    if y0 <= crop_y and x0 <= crop_x and y1 >= crop_y + height and x1 >= crop_x + width:
        return np.ascontiguousarray(
            slab[:, crop_y-y0:crop_y-y0+height, crop_x-x0:crop_x-x0+width], dtype=np.uint8)
    top, left = max(y0, crop_y), max(x0, crop_x)
    bottom, right = min(y1, crop_y + height), min(x1, crop_x + width)
    payload = np.zeros((len(slab), height, width), np.uint8)
    payload[:, top-crop_y:bottom-crop_y, left-crop_x:right-crop_x] = slab[:, top-y0:bottom-y0, left-x0:right-x0]
    return payload


def _reusable_source(segment, bound, window, stored_shape, offset_xyz):
    source = segment.source_layer
    if source is None or bound == _EMPTY_EXTENT or not source.can_copy_gzip_payload:
        return None
    crop_x, crop_y, crop_z = offset_xyz
    depth, height, width = stored_shape
    if window != (crop_y, crop_x, crop_y + height, crop_x + width):
        return None
    source_z = source.offset_tyx[0]
    source_stop = source_z + source.stored_shape_tyx[0]
    # A source with extra stored Z planes must still obey the original
    # trusted-extent contract. Reuse only planes the callback would read.
    if (source_z != bound[4] or source_stop != bound[5] + 1
            or not crop_z <= source_z < source_stop <= crop_z + depth):
        return None
    return source


def _scan_segment(segment, bound, window, first, stop, depth, stored_shape, offset_xyz, compressed):
    """Validate source slabs and optionally write their materialized group bytes."""
    y0, x0, y1, x1 = window
    crop_x, crop_y, _ = offset_xyz
    _, height, width = stored_shape
    source = segment.source_layer
    foreground, exact = 0, _EMPTY_EXTENT
    for z0 in range(first, stop, depth):
        z1 = min(stop, z0 + depth)
        slab = (source.read_slab(z0, z1, stored_xy=True)
                if source is not None else np.asarray(segment.read_slab(z0, z1)))
        if slab.shape != (z1-z0, y1-y0, x1-x0) or slab.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
            raise ValueError("read_slab must return the requested uint8 or bool TYX slab")
        if source is None and slab.dtype == np.uint8 and slab.max() > 1:
            raise ValueError("Grouped segments require binary values 0/1")
        count = int(np.count_nonzero(slab))
        extent = _slab_extent(slab, (z0, y0, x0), count)
        if count and any(extent[2*i] < bound[2*i] or extent[2*i+1] > bound[2*i+1] for i in range(3)):
            raise ValueError(f"Source data exceeds extent_xyz for {segment.segment_id}")
        exact = _merge_extent(exact, extent)
        foreground += count
        if compressed is not None:
            if count:
                payload = _group_slab(slab, window, crop_y, crop_x, height, width)
                compressed.write(memoryview(payload).cast("B"))
                del payload
            else:
                compressed.write_zeros((z1-z0) * height * width)
        if segment.observe_slab is not None:
            observed = slab.view()
            observed.flags.writeable = False
            segment.observe_slab(z0, observed)
            del observed
        del slab
    return foreground, exact


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
    temp_dir: str | Path | None = None, index_segments: bool = False,
    member_pool=None,
) -> dict[str, Any]:
    """Write overlapping ordinary segments with one source pass and bounded slabs.

    The numeric buffer budget admits a source slab, normalization/validation
    workspace, queued compression members, and bounded canonical-zero
    construction/write buffers. Codec control state and O(K) metadata are
    separate fixed overhead. Callbacks may borrow pre-existing source arrays
    but must not create a larger new slab than requested.

    Trusted extent bounds permit cropping and omission of known-empty source
    slices. Values outside those unrequested bounds are the caller's contract.
    Source data returned within a request is checked against its declared bound.
    The compressed spool uses temp_dir (system temporary storage by default);
    only the final assembled temporary file is placed beside its destination.
    Indexed segments have complete gzip member ranges for direct channel reads.
    Aligned scalar payloads can be reused during validation. A supplied pool
    encodes materialized data in independently decodable members.
    """
    shape = _shape(shape_tyx, "shape_tyx")
    budget = _budget(memory_mib)
    geometry = geometry or ReferenceGeometry(shape)
    if geometry.shape_tyx != shape:
        raise ValueError("Writer shape disagrees with spatial reference")
    if isinstance(chunk_slices, bool) or not isinstance(chunk_slices, int) or chunk_slices <= 0:
        raise ValueError("chunk_slices must be a positive integer")
    if not isinstance(overwrite, bool) or not isinstance(index_segments, bool):
        raise ValueError("overwrite and index_segments must be bool")
    descriptors = _validated_segments(segments, shape)
    stored_shape, offset_xyz = _union_crop([row[1] for row in descriptors], shape)
    reusable_sources = [_reusable_source(segment, bound, window, stored_shape, offset_xyz)
                        for segment, bound, _, _, window in descriptors]
    plane_bytes = math.prod(stored_shape[1:])
    largest_input = max(((y1-y0)*(x1-x0) for _, bound, _, _, (y0,x0,y1,x1) in descriptors
                         if bound != _EMPTY_EXTENT), default=1)
    per_slice = max(largest_input, plane_bytes)
    zero_bytes = min(64 * 1024, per_slice)
    minimum = 3 * per_slice + zero_bytes
    pending = reserved = 0
    if member_pool is not None and budget >= minimum:
        channel_bytes = ((bound[5] - bound[4] + 1) * plane_bytes
                         for (_, bound, _, _, _), source in zip(descriptors, reusable_sources)
                         if bound != _EMPTY_EXTENT and source is None)
        # Indexed channels drain separately. An unindexed stream can coalesce
        # materialized bytes from consecutive channels into the same member.
        materialized_bytes = max(channel_bytes, default=0) if index_segments else sum(channel_bytes)
        pending, reserved = member_pool.plan(budget - minimum, materialized_bytes)
    depth = min(chunk_slices, shape[0], (budget - reserved - zero_bytes) // max(1, 3 * per_slice))
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
    crop_z = offset_xyz[2]
    stored_z = stored_shape[0]
    records = []

    try:
        payload_dir = Path(tempfile.mkdtemp(prefix=".grouped-seg-payload-", dir=payload_root))
        payload_path = payload_dir / "payload.gz"
        with payload_path.open("wb") as raw, GzipSpanWriter(
            raw, compresslevel=3, zero_member_bytes=zero_member_bytes,
            write_buffer_bytes=zero_write_buffer_bytes,
            member_pool=member_pool, max_pending_members=pending,
        ) as compressed:
            for (segment, bound, color, tags, window), source in zip(descriptors, reusable_sources):
                start = raw.tell() if index_segments else None
                if bound == _EMPTY_EXTENT:
                    first = stop = crop_z
                else:
                    first, stop = max(crop_z, bound[4]), min(crop_z+stored_z, bound[5]+1)
                compressed.write_zeros((first-crop_z) * plane_bytes)
                if source is None:
                    foreground, exact = _scan_segment(
                        segment, bound, window, first, stop, depth, stored_shape, offset_xyz, compressed)
                else:
                    with compressed.gzip_payload_sink((stop-first) * plane_bytes) as sink:
                        with source.copy_gzip_payload_to(sink, read_bytes=zero_write_buffer_bytes):
                            foreground, exact = _scan_segment(
                                segment, bound, window, first, stop, depth, stored_shape, offset_xyz, None)
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
                if start is not None:
                    compressed.finish_span()
                    records[-1].update(gzip_offset=start, gzip_bytes=raw.tell() - start)
        header = _header(shape, stored_shape, offset_xyz, geometry, records)
        digest = hashlib.sha256(header) if index_segments else None
        with final_stage.open("wb") as destination, payload_path.open("rb") as payload:
            destination.write(header)
            block_bytes = min(1024*1024, budget)
            if digest is None:
                shutil.copyfileobj(payload, destination, length=block_bytes)
            else:
                while block := payload.read(block_bytes):
                    destination.write(block)
                    digest.update(block)
                destination.flush()
                os.fsync(destination.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(final_stage, path)
        if digest is not None:
            for record in records:
                record["gzip_offset"] += len(header)
        return {
            "path": str(path), "shape_tyx": list(shape), "stored_shape_tyx": list(stored_shape),
            "offset_xyz": list(offset_xyz), "segment_count": len(records), "segments": records,
            "numeric_buffer_budget_bytes": budget, "minimum_numeric_buffer_bytes": minimum,
            "chunk_slices": depth, "payload_uncompressed_bytes": math.prod(stored_shape)*len(records),
            "gzip_spans": dict(compressed.stats),
            "bytes": path.stat().st_size,
            **({"sha256": digest.hexdigest()} if digest is not None else {}),
        }
    finally:
        if payload_dir is not None:
            (payload_dir / "payload.gz").unlink(missing_ok=True)
            payload_dir.rmdir()
        final_stage.unlink(missing_ok=True)
        stage_dir.rmdir()
