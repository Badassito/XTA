"""Publish immutable, directly produced review masks without intermediary NRRDs."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, closing
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Callable

import numpy as np

from .native_review_overview import NativeOverviewAccumulator, native_ijk_to_ras, overview_layout
from .native_review_store import (
    _DEFAULT_CHUNK,
    _DIRECT_SCHEMA,
    _PREPARATION_BYTES,
    _archive_digest,
    _block_inventory,
    _chunk_shape,
    _direct_grid_role,
    _layer_catalogue,
    _triple,
    _write_band,
    NativeReviewStore,
)


@dataclass(frozen=True)
class ReviewLayer:
    """One final logical layer at one cutoff, with a bounded TYX slab reader.

    bounds_tyx describes the reader's covered domain as half-open pairs. It
    defaults to the full grid. Voxels outside that domain are background.
    A declared empty layer may have zero-width bounds and is still catalogued.
    """

    metadata: dict
    read_slab: Callable[[int, int], np.ndarray]
    bounds_tyx: tuple[tuple[int, int], ...] | None = None


def _bounds(bounds_tyx, shape):
    if bounds_tyx is None:
        return (0, 0, 0), shape
    pairs = tuple(tuple(pair) for pair in bounds_tyx)
    if len(pairs) != 3 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("Review slab bounds must have three half-open pairs")
    lo = _triple((pair[0] for pair in pairs), "review slab starts")
    hi = _triple((pair[1] for pair in pairs), "review slab stops")
    if any(not a <= b <= size for a, b, size in zip(lo, hi, shape)):
        raise ValueError("Review slab bounds lie outside the selected grid")
    return lo, hi


def _write_layer(db, layer, row, shape, chunks, *, native_overview=False):
    lo, hi = _bounds(layer.bounds_tyx, shape)
    z0, y0, x0 = lo
    z1, y1, x1 = hi
    bz, _, bx = chunks
    count = 0
    lower = list(shape)
    upper = [-1, -1, -1]
    overview = NativeOverviewAccumulator(shape) if native_overview else None
    if all(a < b for a, b in zip(lo, hi)):
        x_aligned = (x0 // bx) * bx
        x_shift = x0 - x_aligned
        height, width = y1 - y0, x1 - x0
        packed_width = (x_shift + width + 7) // 8
        band_bytes = bz * height * packed_width
        per_plane = height * (2 * width + x_shift + packed_width)
        overview_bytes = 0 if overview is None else overview.data.nbytes
        max_depth = min(bz, (_PREPARATION_BYTES - band_bytes - overview_bytes) // per_plane)
        if max_depth < 1:
            raise ValueError("Review slab exceeds its preparation memory bound")
        for band_z in range((z0 // bz) * bz, z1, bz):
            stop = min(band_z + bz, z1)
            band = np.zeros((min(bz, shape[0] - band_z), height, packed_width), np.uint8)
            for first in range(max(z0, band_z), stop, max_depth):
                last = min(first + max_depth, stop)
                slab = np.asarray(layer.read_slab(first, last))
                if (slab.shape != (last - first, height, width)
                        or slab.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8))
                        or (slab.dtype == np.uint8 and slab.max(initial=0) > 1)):
                    raise ValueError("Review layer reader must return an exact binary TYX slab")
                occupied = int(np.count_nonzero(slab))
                if not occupied:
                    continue
                count += occupied
                if overview is not None:
                    overview.add((first, y0, x0), slab)
                for axis, origin in enumerate((first, y0, x0)):
                    other_axes = tuple(i for i in range(3) if i != axis)
                    coordinates = np.flatnonzero(np.any(slab, axis=other_axes))
                    lower[axis] = min(lower[axis], origin + int(coordinates[0]))
                    upper[axis] = max(upper[axis], origin + int(coordinates[-1]))
                if x_shift:
                    padded = np.zeros((last - first, height, x_shift + width), np.uint8)
                    padded[..., x_shift:] = slab
                    slab = padded
                band[first - band_z:last - band_z] = np.packbits(slab, axis=-1, bitorder="little")
            _write_band(db, band, band_z, y0, x_aligned, shape, chunks, row["logical_id"], row["cutoff_cent"])
    row["foreground_voxels"] = count
    row["empty_segment"] = count == 0
    row["extent_xyz"] = (
        [value for pair in zip(reversed(lower), reversed(upper)) for value in pair]
        if count else [0, -1, 0, -1, 0, -1]
    )
    if overview is not None:
        overview.save(db, row["cutoff_cent"])


def _cutoffs(values, default):
    values = tuple(values)
    if (not values or any(type(value) is not int or not 0 <= value <= 100 for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("Review needs distinct integer-hundredth cutoff slots")
    values = tuple(sorted(values))
    default = values[0] if default is None else default
    if type(default) is not int or default not in values:
        raise ValueError("Default cutoff is outside the review")
    return values, default


def _metadata(review_id, grid_id, shape_tyx, native_shape_tyx, geometry, cutoffs_cent, default_cutoff_cent,
              chunk_shape, source_record, axial_crop):
    if not isinstance(review_id, str) or not review_id:
        raise ValueError("Review identity is required")
    shape = _triple(shape_tyx, "review dimensions", minimum=1)
    native = _triple(native_shape_tyx, "native target dimensions", minimum=1)
    role = _direct_grid_role(grid_id, shape, native, axial_crop)
    geometry = dict(
        space=geometry["space"],
        directions_xyz=tuple(tuple(float(v) for v in axis) for axis in geometry["directions_xyz"]),
        origin_xyz=tuple(float(v) for v in geometry["origin_xyz"]),
    )
    native_ijk_to_ras(geometry)
    cutoffs, default = _cutoffs(cutoffs_cent, default_cutoff_cent)
    if source_record is not None and not isinstance(source_record, dict):
        raise ValueError("Review source provenance must be an object")
    exact = (source_record or {}).get("confidence_threshold_exact")
    if exact is not None and (len(cutoffs) != 1 or isinstance(exact, bool)
                              or not isinstance(exact, (float, int)) or not 0 <= exact <= 1):
        raise ValueError("An exact scalar confidence applies only to a singleton review")
    return dict(schema=_DIRECT_SCHEMA, grid_id=grid_id, grid_role=role, shape=shape,
                native_shape_tyx=native, chunk_shape=_chunk_shape(chunk_shape), geometry=geometry,
                sweep_id=review_id, cutoffs=cutoffs, default_cutoff=default, source_record=source_record,
                axial_crop=axial_crop)


def _write_archive(destination, metadata, emit, expected_layer_keys):
    from .json_publication import _fsync_parent_directory
    from .run_transport import _publish_fresh

    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Review archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xta-review-build-", dir=destination.parent) as work:
        stage = Path(work) / "review.sqlite"
        with closing(sqlite3.connect(stage)) as db:
            # Only the private, rebuildable staging database is unsynchronized.
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE review(document TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE blocks(lid TEXT NOT NULL, cutoff INTEGER NOT NULL, z INTEGER NOT NULL,"
                "y INTEGER NOT NULL,x INTEGER NOT NULL,packed BLOB NOT NULL,digest BLOB NOT NULL,"
                "PRIMARY KEY(lid,cutoff,z,y,x)) WITHOUT ROWID"
            )
            groups, keys = defaultdict(list), set()
            for row in emit(db):
                key = row["logical_id"], row["cutoff_cent"]
                if key in keys or key[1] not in metadata["cutoffs"]:
                    raise ValueError(f"Review contains a duplicate or unexpected layer: {key}")
                keys.add(key)
                groups[key[1]].append(row)
            if expected_layer_keys is not None and keys != set(expected_layer_keys):
                raise ValueError("Review layer inventory is incomplete or differs from the requested outputs")
            layers = _layer_catalogue(
                ({"cutoff_cent": cutoff, "segments": rows} for cutoff, rows in groups.items()),
                metadata["cutoffs"],
            )
            globals_ = [layer for layer in layers.values() if layer["role"] == "global"]
            if len(globals_) != 1 or tuple(globals_[0]["cutoffs"]) != metadata["cutoffs"]:
                raise ValueError("A complete review needs one logical Global at every cutoff")
            metadata["layers"] = layers
            metadata["block_inventory"] = _block_inventory(db)
            # Normalize integer catalogue keys before hashing as JSON object keys.
            metadata = json.loads(json.dumps(metadata, allow_nan=False))
            metadata["identity"] = metadata["manifest_sha256"] = _archive_digest(metadata)
            db.execute("INSERT INTO review VALUES (?)", (json.dumps(metadata, allow_nan=False),))
            db.execute("CREATE INDEX blocks_region ON blocks(z,y,x,cutoff,lid)")
            db.commit()
        with stage.open("rb+") as stream:
            os.fsync(stream.fileno())
        _publish_fresh(stage, destination)
        _fsync_parent_directory(destination.parent)
    return destination


def write_review_archive(destination, *, review_id, grid_id, shape_tyx, native_shape_tyx, geometry,
                         cutoffs_cent, layers, source_record=None, axial_crop=None, default_cutoff_cent=None,
                         chunk_shape=_DEFAULT_CHUNK, expected_layer_keys=None):
    """Publish one portable preview or native archive from settled masks.

    The caller owns slab readers through each layer's yielded lifetime. For a
    scalar threshold not representable in hundredths, the sole cutoff is a
    display slot and the source record should preserve the exact threshold.
    """
    metadata = _metadata(review_id, grid_id, shape_tyx, native_shape_tyx, geometry, cutoffs_cent,
                         default_cutoff_cent, chunk_shape, source_record, axial_crop)

    def emit(db):
        seen = set()
        for layer in layers:
            row = dict(layer.metadata)
            lid, cutoff = row["logical_id"], row["cutoff_cent"]
            if (not isinstance(lid, str) or not lid or type(cutoff) is not int
                    or cutoff not in metadata["cutoffs"] or (lid, cutoff) in seen):
                raise ValueError("Invalid or duplicate review layer")
            seen.add((lid, cutoff))
            row.setdefault("segment_id", _archive_digest([review_id, grid_id, lid, cutoff]))
            _write_layer(db, layer, row, metadata["shape"], metadata["chunk_shape"],
                         native_overview=metadata["grid_role"] == "native" and row.get("role") == "global")
            yield row

    return _write_archive(destination, metadata, emit, expected_layer_keys)


def _source_without_scalar_cutoff(source):
    if not isinstance(source, dict):
        return source
    return {key: value for key, value in source.items() if key != "confidence_threshold_exact"}


def merge_review_archives(destination, branches, *, review_id, source_record=None, default_cutoff_cent=None,
                          expected_grid_id=None, expected_shape_tyx=None, expected_native_shape_tyx=None):
    """Assemble a family archive from its sealed private, single-cutoff archives."""
    cutoffs, default = _cutoffs(branches, default_cutoff_cent)
    with ExitStack() as stack:
        stores = {
            cutoff: stack.enter_context(NativeReviewStore(branches[cutoff])) for cutoff in cutoffs
        }
        first = stores[cutoffs[0]]
        if ((expected_grid_id is not None and first.grid_id != expected_grid_id)
                or (expected_shape_tyx is not None and first.shape != tuple(expected_shape_tyx))
                or (expected_native_shape_tyx is not None
                    and first.native_shape_tyx != tuple(expected_native_shape_tyx))):
            raise ValueError("Review archive geometry differs from the requested output grid")
        common_source = _source_without_scalar_cutoff(first.source_record)
        for cutoff, store in stores.items():
            if store.schema != _DIRECT_SCHEMA or store.cutoffs != (cutoff,):
                raise ValueError("Review branch does not bind exactly its declared cutoff")
            if "preview_edit" in (store.source_record or {}):
                raise ValueError("Edited preview snapshots cannot serve as original cutoff branches")
            exact = (store.source_record or {}).get("confidence_threshold_exact")
            if exact is not None and exact != cutoff / 100:
                raise ValueError("Scalar confidence does not match its declared family cutoff")
            if (store.grid_id, store.shape, store.native_shape_tyx, store.geometry, store.axial_crop,
                    store.chunk_shape, _source_without_scalar_cutoff(store.source_record)) != (
                    first.grid_id, first.shape, first.native_shape_tyx, first.geometry, first.axial_crop,
                    first.chunk_shape, common_source):
                raise ValueError("Review branches disagree on source or grid geometry")
        metadata = _metadata(review_id, first.grid_id, first.shape, first.native_shape_tyx, first.geometry,
                             cutoffs, default, first.chunk_shape,
                             common_source if source_record is None else _source_without_scalar_cutoff(source_record),
                             first.axial_crop)
        metadata["source_archive_ids"] = {cutoff: store.identity for cutoff, store in stores.items()}

        def emit(db):
            limit = math.prod(first.chunk_shape) // 8
            limit += limit // 100 + 128
            for cutoff, store in stores.items():
                with store.read_connection() as source:
                    for lid, cent, z, y, x, size, payload, sha in source.execute(
                        "SELECT lid,cutoff,z,y,x,length(packed),substr(packed,1,?),digest "
                        "FROM blocks ORDER BY lid,cutoff,z,y,x", (limit,),
                    ):
                        if (lid not in store.layers or cent != cutoff or size > limit
                                or hashlib.sha256(payload).digest() != sha):
                            raise ValueError("Review branch has an invalid compressed block")
                        store.block_bounds((z, y, x))
                        db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (lid, cent, z, y, x, payload, sha))
                    if store.grid_role == "native":
                        packed_size = (math.prod(overview_layout(first.shape)[1]) + 7) // 8
                        maximum = packed_size + packed_size // 100 + 128
                        row = source.execute(
                            "SELECT length(packed),substr(packed,1,?) FROM native_overviews WHERE cutoff=?",
                            (maximum, cutoff),
                        ).fetchone()
                        if row is None or row[0] != len(row[1]):
                            raise ValueError("Review branch has no complete native overview")
                        db.execute("CREATE TABLE IF NOT EXISTS native_overviews "
                                   "(cutoff INTEGER PRIMARY KEY, packed BLOB NOT NULL)")
                        db.execute("INSERT INTO native_overviews VALUES (?, ?)", (cutoff, row[1]))
                for lid, layer in store.layers.items():
                    row = dict(layer["segments_by_cutoff"][str(cutoff)])
                    row.update(logical_id=lid, cutoff_cent=cutoff, role=layer["role"],
                               segment_name=layer["name"], color_rgb=layer["color"],
                               empty_segment=row["foreground_voxels"] == 0,
                               extent_xyz=layer["extent_by_cutoff"][cutoff])
                    yield row

        return _write_archive(destination, metadata, emit, None)
