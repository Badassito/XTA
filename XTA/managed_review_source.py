"""Transient proof and sparse coverage for scalar NRRDs written by a sweep."""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np

from .file_hashing import sha256_file
from .run_transport import _stamp
from .sweep_quota import SweepViewCapacityError, _update_database


_GIB = 1024**3
_SCHEMA = "xta.managed_review_source/1"
_LEDGER_TIMEOUT_SECONDS = 1
_MAX_SOURCE_PLANE_BYTES = 16 * 1024**2
_BIT_COUNTS = np.frombuffer(bytes(value.bit_count() for value in range(256)), dtype=np.uint8)
_BITWISE_COUNT = getattr(np, "bitwise_count", None)


def _receipt_dir(path):
    path = Path(path)
    key = hashlib.sha256(path.name.encode("utf-8")).hexdigest()
    return path.parent / ".review_sources" / key


def _source_planes(directory, shape, extent, packed):
    from .interpolation import CTILE_INDEX_DTYPE

    depth, height, width = shape
    index_path = directory / "index.bin"
    if index_path.stat().st_size != depth * CTILE_INDEX_DTYPE.itemsize:
        raise ValueError("Review source index size disagrees with its grid")
    index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=depth)
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    with (directory / "chunks.bin").open("rb") as chunks:
        size = os.fstat(chunks.fileno()).st_size
        for z, record in enumerate(index):
            kind = int(record["kind"])
            if kind == 0:
                continue
            y, x, stop_y, stop_x = (int(record[key]) for key in ("y0", "x0", "y1", "x1"))
            rows, columns = stop_y - y, stop_x - x
            stride = (columns + 7) // 8 if packed else columns
            offset = int(record["offset"])
            length = rows * stride
            if (kind != 1 or not zmin <= z <= zmax or
                    not ymin <= y < stop_y <= min(height, ymax + 1) or
                    not xmin <= x < stop_x <= min(width, xmax + 1) or
                    int(record["payload_size"]) != length or
                    int(record["payload_nbytes"]) != rows * columns or
                    not 0 <= offset <= size - length):
                raise ValueError(f"Invalid review source record at slice {z}")
            if length > _MAX_SOURCE_PLANE_BYTES:
                raise ValueError("Review source crop exceeds the per-plane receipt workspace")
            chunks.seek(offset)
            payload = chunks.read(length)
            if len(payload) != length:
                raise ValueError(f"Short review source at slice {z}")
            yield z, np.frombuffer(payload, dtype=np.uint8).reshape(rows, stride), y, x, columns


def _source_stamp(directory):
    return tuple(_stamp((directory / name).stat()) for name in ("index.bin", "chunks.bin"))


@lru_cache(maxsize=16)
def _memory_backed_stage(path):
    from .runtime import path_is_memory_backed

    return path_is_memory_backed(path)


def _has_scratch_capacity(stage_root, amount):
    free = shutil.disk_usage(stage_root).free
    if _memory_backed_stage(stage_root):
        from .workspace import available_anon_work_bytes

        free = min(free, available_anon_work_bytes())
    return free >= 8 * _GIB + amount


def review_source_identity(ref, shape):
    """Sample the scalar writer's backing store before it starts reading."""
    from .interpolation import MASK_STORE_FORMATS

    if (ref.live_array is not None or ref.storage_format not in MASK_STORE_FORMATS
            or tuple(ref.shape) != tuple(shape)):
        return None
    source = Path(ref.path)
    files = [source / name for name in ("index.bin", "chunks.bin", "meta.json")]
    if source.is_symlink() or any(item.is_symlink() for item in files):
        return None
    try:
        # Link creation and retirement change ctime, but do not change the
        # inode, size or last write time.
        return tuple(_stamp(item.stat())[:4] for item in files)
    except OSError:
        return None


def _inspect_source(directory, shape, extent, packed):
    """Check exact per-plane boxes, binary voxel count and sparse byte identity."""
    digest = hashlib.sha256()
    count = 0
    union = [shape[2], -1, shape[1], -1, shape[0], -1]
    for z, crop, y, x, width in _source_planes(directory, shape, extent, packed):
        data = memoryview(crop).cast("B")
        digest.update(data)
        if packed:
            right_bit = 1 << ((width - 1) % 8)
            if right_bit < 128 and (crop[:, -1] > (2 * right_bit - 1)).any():
                raise ValueError("Packed review source has nonzero row padding")
            left, right = crop[:, 0] & 1, crop[:, -1] & right_bit
            counts = _BITWISE_COUNT(crop) if _BITWISE_COUNT is not None else _BIT_COUNTS[crop]
            count += int(counts.sum(dtype=np.uint64))
        else:
            if crop.max() > 1:
                raise ValueError("Raw review source is not binary")
            left, right = crop[:, 0], crop[:, -1]
            count += int(np.count_nonzero(crop))
        if not (crop[0].any() and crop[-1].any() and left.any() and right.any()):
            raise ValueError("Review source does not have exact plane bounds")
        union[0], union[1] = min(union[0], x), max(union[1], x + width - 1)
        union[2], union[3] = min(union[2], y), max(union[3], y + len(crop) - 1)
        union[4], union[5] = min(union[4], z), max(union[5], z)
    if tuple(union) != extent or count == 0:
        raise ValueError("Review source does not have the scalar layer extent")
    return count, digest.hexdigest()


def capture_native_review_source(ref, shape, path, plan, stage_root, *, expected_identity=None):
    """Keep a bounded, exact same-grid sparse source after the scalar write succeeds."""
    from .interpolation import INTERNAL_PACKED_CVOL_FORMAT

    shape = tuple(shape)
    if getattr(plan, "layout", None) != "cropped":
        return False
    original_identity = review_source_identity(ref, shape)
    if original_identity is None or (expected_identity is not None and expected_identity != original_identity):
        return False
    limit_gib = float(os.environ.get("XTA_YOLO_SWEEP_REVIEW_SOURCE_GIB", "64"))
    if not math.isfinite(limit_gib) or limit_gib < 0:
        raise ValueError("XTA_YOLO_SWEEP_REVIEW_SOURCE_GIB must be finite and nonnegative")
    if not limit_gib:
        return False
    path = Path(path)
    stage_root = Path(stage_root)
    directory = _receipt_dir(path)
    if directory.exists():
        return False
    extent = tuple(v for low, length in zip(reversed(plan.stored_offset_tyx),
                                          reversed(plan.stored_shape_tyx))
                   for v in (low, low + length - 1))
    expected_extent = getattr(ref, "segment_extent_ijk", None)
    if expected_extent is None or tuple(expected_extent) != extent:
        return False
    source = Path(ref.path)
    files = [source / name for name in ("index.bin", "chunks.bin")]
    database = stage_root / "review_source_quota.sqlite"
    lease = str(directory.relative_to(stage_root))
    charged = False
    try:
        amount = sum(item.stat().st_size for item in files) + 4096
        if not _has_scratch_capacity(stage_root, amount):
            return False
        limit = int(limit_gib * _GIB)
        _update_database(database, [(lease, amount)], create=True, limit=limit, timeout=_LEDGER_TIMEOUT_SECONDS)
        charged = True
        directory.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=directory.parent) as temp:
            temp = Path(temp)
            for index, item in enumerate(files):
                destination = temp / item.name
                try:
                    os.link(item, destination)
                except OSError:
                    if _memory_backed_stage(stage_root):
                        raise
                    shutil.copyfile(item, destination)
                else:
                    if _stamp(destination.stat())[:4] != original_identity[index]:
                        raise ValueError("Review source inode changed before capture")
            source_stamp = _source_stamp(temp)
            scalar_stamp = _stamp(path.stat())
            meta = json.loads((source / "meta.json").read_bytes())
            if meta.get("format") != ref.storage_format or tuple(meta.get("shape", ())) != shape:
                raise ValueError("Review source format or grid changed during capture")
            packed = ref.storage_format == INTERNAL_PACKED_CVOL_FORMAT
            count, crop_digest = _inspect_source(temp, shape, extent, packed)
            receipt = {
                "schema": _SCHEMA, "extent_xyz": extent, "shape_tyx": shape,
                "foreground_voxels": count, "packed": packed, "crops_sha256": crop_digest,
                "index_sha256": sha256_file(temp / "index.bin"), "scalar_sha256": sha256_file(path),
            }
            if (source_stamp != _source_stamp(temp) or scalar_stamp != _stamp(path.stat())
                    or review_source_identity(ref, shape) != original_identity):
                raise ValueError("Managed review source changed during capture")
            (temp / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            os.rename(temp, directory)
        return True
    except (OSError, ValueError, sqlite3.Error, SweepViewCapacityError):
        # Cross-filesystem links and storage pressure leave the original scalar
        # NRRD available for fully validated publication.
        if charged:
            with suppress(OSError, sqlite3.Error):
                _update_database(database, [(lease, 0)], create=False, limit=limit, timeout=_LEDGER_TIMEOUT_SECONDS)
        return False


@dataclass(frozen=True)
class ManagedReviewSource:
    directory: Path
    receipt: dict

    def copy_verified(self, source, sink, bound, observe_plane, observe_packed, read_bytes):
        """Verify the original scalar bytes while copying complete gzip members."""
        from .reconciliation_io import _read_header_stream

        receipt = self.receipt
        if (tuple(receipt["shape_tyx"]) != source.shape_tyx
                or tuple(receipt["extent_xyz"]) != bound):
            raise ValueError(f"Managed review geometry differs from scalar NRRD: {source.path}")
        count = receipt["foreground_voxels"]
        if type(count) is not int or not 0 < count <= math.prod(source.stored_shape_tyx):
            raise ValueError(f"Invalid managed review foreground count: {source.path}")
        digest = hashlib.sha256()
        with source.path.open("rb") as scalar:
            stamp = _stamp(os.fstat(scalar.fileno()))
            if _read_header_stream(scalar, source.path) != (source.header, source._payload_offset):
                raise ValueError(f"Scalar NRRD header changed since inspection: {source.path}")
            scalar.seek(0)
            digest.update(scalar.read(source._payload_offset))
            while data := scalar.read(read_bytes):
                digest.update(data)
                sink.write(data)
            if (stamp != _stamp(os.fstat(scalar.fileno())) or stamp != _stamp(source.path.stat())
                    or digest.hexdigest() != receipt["scalar_sha256"]):
                raise ValueError(f"Managed review scalar digest mismatch: {source.path}")
        if observe_plane is not None:
            stamp = _source_stamp(self.directory)
            if sha256_file(self.directory / "index.bin") != receipt["index_sha256"]:
                raise ValueError(f"Managed review index digest mismatch: {source.path}")
            digest = hashlib.sha256()
            packed = receipt["packed"]
            for z, crop, y, x, width in _source_planes(self.directory, source.shape_tyx, bound, packed):
                digest.update(memoryview(crop).cast("B"))
                if packed:
                    observe_packed(z, crop, y, x, width)
                else:
                    observe_plane(z, crop, y, x)
            if digest.hexdigest() != receipt["crops_sha256"] or stamp != _source_stamp(self.directory):
                raise ValueError(f"Managed review crop digest mismatch: {source.path}")
        return count, bound


def read_native_review_source(path):
    if path is None:
        return None
    directory = _receipt_dir(path)
    if not directory.exists():
        return None
    receipt = json.loads((directory / "receipt.json").read_bytes())
    if receipt["schema"] != _SCHEMA:
        raise ValueError(f"Unknown managed review receipt schema: {path}")
    return ManagedReviewSource(directory, receipt)
