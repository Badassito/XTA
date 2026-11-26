"""Immutable, independently indexed native segmentation masks for Slicer review."""
from __future__ import annotations

from contextlib import closing, contextmanager
import gzip
import hashlib
from itertools import groupby, islice
import json
import math
import operator
import os
from pathlib import Path
import sqlite3
import tempfile
import zlib

import numpy as np

from .native_review_overview import NativeOverviewAccumulator, native_ijk_to_ras, overview_layout


_SCHEMA = "xta.native_review/1"
_DIRECT_SCHEMA = "xta.review_archive/2"
_DEFAULT_CHUNK = (32, 64, 64)
_MAX_CHUNK_BYTES = 8 * 1024**2
_PREPARATION_BYTES = 256 * 1024**2
_MAX_REGION_BYTES = 128 * 1024**2
_PACKED_UNION_BYTES = 64 * 1024**2


def _triple(values, name, *, minimum=0):
    try:
        values = tuple(operator.index(n) for n in values)
    except TypeError as error:
        raise ValueError(f"Expected three integer {name} values") from error
    if len(values) != 3 or any(n < minimum for n in values):
        raise ValueError(f"Invalid {name}: {values}")
    return values


def _chunk_shape(values):
    shape = _triple(values, "chunk dimensions", minimum=1)
    if shape[2] % 8 or math.prod(shape) > _MAX_CHUNK_BYTES:
        raise ValueError("Chunk X must be a multiple of 8 and each decoded chunk must fit in 8 MiB")
    return shape


def _positions(start, stop, step):
    return range(start // step, (stop - 1) // step + 1)


def _region_filter(bounds):
    """Constrain a small block box by index prefix rather than scanning a Z band."""
    conditions, parameters = [], []
    for axis, lo, hi in zip("zyx", bounds[::2], bounds[1::2]):
        if axis != "x" and hi - lo < 8:
            conditions.append(f"{axis} IN ({','.join('?' for _ in range(lo, hi + 1))})")
            parameters.extend(range(lo, hi + 1))
        else:
            conditions.append(f"{axis} BETWEEN ? AND ?")
            parameters.extend((lo, hi))
    return " AND ".join(conditions), tuple(parameters)


def _file_stamp(path):
    file = path.stat()
    return file.st_dev, file.st_ino, file.st_size, file.st_mtime_ns, file.st_ctime_ns


def _archive_digest(value):
    document = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _direct_grid_role(grid_id, shape, native_shape, crop):
    if grid_id == "native" and native_shape == shape and crop is None:
        return "native"
    if (not isinstance(grid_id, str) or not grid_id.startswith("low_quality/")
            or any(a > b for a, b in zip(shape, native_shape))):
        raise ValueError("Review archive grid role and geometry disagree")
    if crop is not None:
        if not isinstance(crop, dict):
            raise ValueError("Invalid preview axial crop record")
        original = _triple(crop.get("original_preview_shape_tyx"), "original preview dimensions", minimum=1)
        original_native = _triple(crop.get("original_full_quality_shape_tyx"), "original native dimensions", minimum=1)
        if (shape[:2] != original[:2] or native_shape[:2] != original_native[:2]
                or not shape[2] <= original[2] or not native_shape[2] < original_native[2]
                or shape[2] != crop.get("first_removed_preview_i")
                or native_shape[2] != crop.get("first_removed_native_i")
                or any(a > b for a, b in zip(original, original_native))):
            raise ValueError("Cropped review dimensions differ from the original lattice and crop record")
    return "preview"


def _block_inventory(db):
    digest = hashlib.sha256()
    count = 0
    for lid, cutoff, z, y, x, block_sha in db.execute(
        "SELECT lid,cutoff,z,y,x,digest FROM blocks ORDER BY lid,cutoff,z,y,x"
    ):
        if not isinstance(block_sha, bytes) or len(block_sha) != 32:
            raise ValueError("Invalid review block digest")
        digest.update(bytes.fromhex(_archive_digest([lid, cutoff, z, y, x, block_sha.hex()])))
        count += 1
    return {"blocks": count, "sha256": digest.hexdigest()}


class NativeReviewStore:
    """Open a read-only archive using NumPy and the Python standard library."""

    def __init__(self, path, *, expected_identity=None):
        self.path = Path(path).expanduser().resolve(strict=True)
        self.file_stamp = _file_stamp(self.path)
        self._db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            self._db.execute("PRAGMA trusted_schema=OFF")
            self._db.execute("PRAGMA query_only=ON")
            # Editors can retain this snapshot through multiple native reads,
            # even when a separate application attempts to write the file.
            self._db.execute("BEGIN")
            rows = self._db.execute("SELECT document FROM review LIMIT 2").fetchall()
            if len(rows) != 1:
                raise ValueError("Expected one native review catalogue")
            if _file_stamp(self.path) != self.file_stamp:
                raise ValueError("Native review archive changed while opening")
            metadata = json.loads(rows[0][0])
            if metadata["schema"] not in (_SCHEMA, _DIRECT_SCHEMA):
                raise ValueError("Unsupported native review archive schema")
            self.schema = metadata["schema"]
            self.shape = self.shape_zyx = _triple(metadata["shape"], "native dimensions", minimum=1)
            self.chunk_shape = self.chunk_shape_zyx = self.block_shape = _chunk_shape(metadata["chunk_shape"])
            self.cutoffs = tuple(metadata["cutoffs"])
            if (not self.cutoffs or tuple(sorted(set(self.cutoffs))) != self.cutoffs
                    or any(type(c) is not int or not 0 <= c <= 100 for c in self.cutoffs)):
                raise ValueError("Invalid native review cutoffs")
            self.default_cutoff = metadata.get("default_cutoff", self.cutoffs[0])
            if type(self.default_cutoff) is not int or self.default_cutoff not in self.cutoffs:
                raise ValueError("Default cutoff differs from native review catalogue")
            self.geometry = metadata["geometry"]
            self.space = self.geometry["space"]
            self.directions_xyz = tuple(map(tuple, self.geometry["directions_xyz"]))
            self.origin_xyz = tuple(self.geometry["origin_xyz"])
            self.affine_ijk_to_ras = native_ijk_to_ras(self.geometry)
            self.identity = self.source_identity = metadata["identity"]
            if expected_identity is not None and self.identity != expected_identity:
                raise ValueError("Archive belongs to a different native review")
            self.grid_id = "native"
            self.grid_role = "native"
            self.native_shape_tyx = self.shape
            self.axial_crop = None
            if self.schema == _DIRECT_SCHEMA:
                manifest = {key: value for key, value in metadata.items() if key not in ("identity", "manifest_sha256")}
                if metadata["manifest_sha256"] != self.identity or _archive_digest(manifest) != self.identity:
                    raise ValueError("Review catalogue failed its digest check")
                if _block_inventory(self._db) != metadata["block_inventory"]:
                    raise ValueError("Review archive block inventory changed or is incomplete")
                self.grid_id = metadata["grid_id"]
                self.grid_role = metadata["grid_role"]
                self.native_shape_tyx = _triple(metadata["native_shape_tyx"], "native target dimensions", minimum=1)
                self.axial_crop = metadata.get("axial_crop")
                if self.grid_role != _direct_grid_role(self.grid_id, self.shape, self.native_shape_tyx, self.axial_crop):
                    raise ValueError("Review archive grid role and geometry disagree")
            self.source_record = metadata.get("source_record")
            self.sweep_id = metadata["sweep_id"]
            self.review_manifest_sha256 = metadata["manifest_sha256"]
            self.layers = metadata["layers"]
            if not isinstance(self.layers, dict):
                raise ValueError("Invalid native review layer catalogue")
            global_layers = []
            for lid, record in self.layers.items():
                if (record["layer_id"] != lid or not frozenset(record["cutoffs"]).issubset(self.cutoffs)
                        or set(record["extent_by_cutoff"]) != set(map(str, record["cutoffs"]))):
                    raise ValueError("Invalid native review layer identity")
                record["extent_by_cutoff"] = {
                    int(c): tuple(box) for c, box in record["extent_by_cutoff"].items()
                }
                record["cutoffs"] = tuple(record["cutoffs"])
                if record["role"] == "global":
                    global_layers.append(lid)
            if len(global_layers) > 1:
                raise ValueError("Multiple Global identities in native review archive")
            self.global_layer_id = global_layers[0] if global_layers else None
        except (KeyError, TypeError, ValueError, sqlite3.Error) as error:
            self._db.close()
            raise ValueError(f"Invalid native review archive: {self.path}: {error}") from error

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    @contextmanager
    def read_connection(self):
        """Borrow the stable read-only snapshot for optional archive tables."""
        yield self._db

    def _check_layer(self, lid, cutoff):
        if lid not in self.layers:
            raise KeyError(lid)
        if not isinstance(cutoff, (int, np.integer)) or isinstance(cutoff, bool) or cutoff not in self.cutoffs:
            raise ValueError(f"Unknown native review cutoff {cutoff}")

    def block_bounds(self, index):
        index = _triple(index, "chunk coordinates")
        starts = tuple(i * b for i, b in zip(index, self.chunk_shape))
        stops = tuple(min(a + b, n) for a, b, n in zip(starts, self.chunk_shape, self.shape))
        if any(b <= a for a, b in zip(starts, stops)):
            raise ValueError(f"Chunk outside native review grid: {index}")
        return tuple(zip(starts, stops))

    def _decode_packed_block(self, lid, cutoff, index):
        self._check_layer(lid, cutoff)
        index = _triple(index, "chunk coordinates")
        depth, height, width = (hi - lo for lo, hi in self.block_bounds(index))
        packed_shape = (depth, height, (width + 7) // 8)
        size = math.prod(packed_shape)
        limit = size + size // 100 + 128
        row = self._db.execute(
            "SELECT length(packed), substr(packed, 1, ?), digest FROM blocks "
            "WHERE lid=? AND cutoff=? AND z=? AND y=? AND x=?",
            (limit, lid, int(cutoff), *index),
        ).fetchone()
        if row is None:
            return np.zeros(packed_shape, np.uint8), width
        return self._decode_packed_row(lid, cutoff, index, row, packed_shape, width), width

    @staticmethod
    def _decode_packed_row(lid, cutoff, index, row, packed_shape, width):
        size = math.prod(packed_shape)
        length, packed, digest = row
        if length > size + size // 100 + 128 or hashlib.sha256(packed).digest() != digest:
            raise ValueError(f"Native review chunk failed its checksum: {lid} {cutoff} {index}")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(packed, size + 1)
        except zlib.error as error:
            raise ValueError(f"Invalid compressed native review chunk: {lid} {cutoff} {index}") from error
        if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError(f"Invalid decoded native review chunk: {lid} {cutoff} {index}")
        rows = np.frombuffer(raw, np.uint8).reshape(packed_shape)
        if width % 8 and np.any(rows[..., -1] >> (width % 8)):
            raise ValueError(f"Nonzero native review padding bits: {lid} {cutoff} {index}")
        return rows

    def read_packed_block(self, lid, cutoff, index):
        """Read validated little-endian bit-packed X rows."""
        rows, _ = self._decode_packed_block(lid, cutoff, index)
        return rows

    def read_block(self, lid, cutoff, index):
        rows, width = self._decode_packed_block(lid, cutoff, index)
        return np.unpackbits(rows, axis=-1, count=width, bitorder="little").view(np.bool_)

    def iter_packed_union(self, layer_ids, cutoff):
        """Stream the immutable union of selected layers in native block order."""
        layer_ids = tuple(sorted(set(layer_ids)))
        if not layer_ids:
            return
        for lid in layer_ids:
            self._check_layer(lid, cutoff)
        cutoff = int(cutoff)
        max_size = math.prod(self.chunk_shape) // 8
        limit = max_size + max_size // 100 + 128
        max_blocks = _PACKED_UNION_BYTES // max_size
        for start, stop, blocks in self._union_bands(cutoff, max_blocks):
            if blocks <= max_blocks:
                yield from self._packed_union_band(layer_ids, cutoff, start, stop, limit)
            else:
                # Very wide archives may exceed the cap in even one XY
                # chunk plane. Stream those planes in coordinate order.
                yield from self._packed_union_plane(frozenset(layer_ids), cutoff, start, limit)

    def _union_bands(self, cutoff, max_blocks):
        coordinates = self._db.execute(
            "SELECT DISTINCT z,y,x FROM blocks INDEXED BY blocks_region "
            "WHERE cutoff=? ORDER BY z,y,x",
            (cutoff,),
        )
        try:
            start, stop, used = None, None, 0
            for z, plane in groupby(coordinates, key=lambda row: row[0]):
                count = sum(1 for _ in plane)
                if start is not None and (used + count > max_blocks or z - start >= 4):
                    yield start, stop, used
                    start, used = None, 0
                if start is None:
                    start = z
                stop, used = z + 1, used + count
            if start is not None:
                yield start, stop, used
        finally:
            try:
                coordinates.close()
            except sqlite3.ProgrammingError:
                pass  # A cancelled operation can close the archive before this iterator.

    def _union_block(self, index):
        depth, height, width = (hi - lo for lo, hi in self.block_bounds(index))
        return np.zeros((depth, height, (width + 7) // 8), dtype=np.uint8), width

    def _packed_union_band(self, layer_ids, cutoff, start, stop, limit):
        # Read by primary key so adjacent payloads are fetched together. The
        # independent spatial index funds the in-memory packed union.
        unions = {}
        identifiers = iter(layer_ids)
        while selected := tuple(islice(identifiers, 256)):
            placeholders = ",".join("?" for _ in selected)
            records = self._db.execute(
                f"SELECT z,y,x,lid,length(packed),substr(packed, 1, ?),digest FROM blocks "
                f"WHERE lid IN ({placeholders}) AND cutoff=? AND z>=? AND z<?",
                (limit, *selected, cutoff, start, stop),
            )
            with closing(records):
                for z, y, x, lid, length, packed, digest in records:
                    index = z, y, x
                    if index not in unions:
                        unions[index] = self._union_block(index)
                    combined, width = unions[index]
                    combined |= self._decode_packed_row(
                        lid, cutoff, index, (length, packed, digest), combined.shape, width,
                    )
        for index in sorted(unions):
            combined, _ = unions.pop(index)
            if combined.any():
                yield index, combined

    def _packed_union_plane(self, layer_ids, cutoff, z, limit):
        # The covering alias filters the cutoff before SQLite seeks a BLOB
        # using its primary key. A single non-covering index cursor can seek
        # payload pages for other cutoffs too.
        records = self._db.execute(
            "SELECT a.y,a.x,a.lid,length(b.packed),substr(b.packed, 1, ?),b.digest "
            "FROM blocks AS a INDEXED BY blocks_region CROSS JOIN blocks AS b "
            "ON (b.lid,b.cutoff,b.z,b.y,b.x)=(a.lid,a.cutoff,a.z,a.y,a.x) "
            "WHERE a.z=? AND a.cutoff=? ORDER BY a.y,a.x",
            (limit, z, cutoff),
        )
        try:
            for (y, x), chunks in groupby(records, key=lambda row: row[:2]):
                index = z, y, x
                combined, width = self._union_block(index)
                for _, _, lid, length, packed, digest in chunks:
                    if lid in layer_ids:
                        combined |= self._decode_packed_row(
                            lid, cutoff, index, (length, packed, digest), combined.shape, width,
                        )
                if combined.any():
                    yield index, combined
        finally:
            try:
                records.close()
            except sqlite3.ProgrammingError:
                pass

    def iter_blocks(self, lid, cutoff):
        self._check_layer(lid, cutoff)
        for row in self._db.execute(
            "SELECT z,y,x FROM blocks WHERE lid=? AND cutoff=? ORDER BY z,y,x", (lid, int(cutoff)),
        ):
            yield row

    def layers_at(self, index, cutoff=None):
        index = _triple(index, "chunk coordinates")
        self.block_bounds(index)
        if cutoff is None:
            yield from (row[0] for row in self._db.execute(
                "SELECT DISTINCT lid FROM blocks WHERE z=? AND y=? AND x=? ORDER BY lid", index,
            ))
        else:
            if not isinstance(cutoff, (int, np.integer)) or isinstance(cutoff, bool) or cutoff not in self.cutoffs:
                raise ValueError(f"Unknown native review cutoff {cutoff}")
            yield from (row[0] for row in self._db.execute(
                "SELECT lid FROM blocks WHERE z=? AND y=? AND x=? AND cutoff=? ORDER BY lid",
                (*index, int(cutoff)),
            ))

    def _region_layers(self, cutoff, region_filter):
        clause, bounds = region_filter
        return self._db.execute(
            f"SELECT DISTINCT lid FROM blocks INDEXED BY blocks_region "
            f"WHERE cutoff=? AND {clause}",
            (cutoff, *bounds),
        )

    def _region_rows(self, lid, cutoff, region_filter):
        """Read only stored chunks for one layer and cutoff in a block box."""
        max_size = math.prod(self.chunk_shape) // 8
        limit = max_size + max_size // 100 + 128
        clause, bounds = region_filter
        return self._db.execute(
            f"SELECT z,y,x,length(packed),substr(packed, 1, ?),digest FROM blocks "
            f"WHERE lid=? AND cutoff=? AND {clause} ORDER BY z,y,x",
            (limit, lid, cutoff, *bounds),
        )

    def read_region(self, lid, cutoff, bounds):
        """Read exact half-open ZYX bounds, assembling one block at a time."""
        from itertools import product

        self._check_layer(lid, cutoff)
        bounds = tuple(tuple(operator.index(n) for n in pair) for pair in bounds)
        if len(bounds) != 3 or any(len(pair) != 2 or not 0 <= pair[0] <= pair[1] <= n
                                  for pair, n in zip(bounds, self.shape)):
            raise ValueError("Expected a region within the native review grid")
        shape = tuple(hi - lo for lo, hi in bounds)
        if math.prod(shape) > _MAX_REGION_BYTES:
            raise ValueError("Native review region exceeds 128 MiB. Read smaller regions or iterate blocks")
        region = np.zeros(shape, np.bool_)
        if not region.size:
            return region
        for index in product(*(_positions(lo, hi, step) for (lo, hi), step in zip(bounds, self.chunk_shape))):
            source, dest = [], []
            for (rlo, rhi), (blo, bhi) in zip(bounds, self.block_bounds(index)):
                lo, hi = max(rlo, blo), min(rhi, bhi)
                source.append(slice(lo - blo, hi - blo))
                dest.append(slice(lo - rlo, hi - rlo))
            region[tuple(dest)] = self.read_block(lid, cutoff, index)[tuple(source)]
        return region


def _layer_catalogue(groups, cutoffs):
    layers = {}
    seen_segments = set()
    for group in groups:
        cutoff = group["cutoff_cent"]
        if type(cutoff) is not int or cutoff not in cutoffs:
            raise ValueError("Native group cutoff is missing from the review")
        for row in group["segments"]:
            lid, segment, role = row["logical_id"], row["segment_id"], row["role"]
            if (not isinstance(lid, str) or not lid or not isinstance(segment, str) or not segment
                    or role not in ("direct", "generated", "derived", "global") or segment in seen_segments):
                raise ValueError("Invalid or duplicate native review segment identity")
            seen_segments.add(segment)
            additive = role in ("direct", "generated") or (
                role == "derived" and row.get("layer_role") == "additive_component"
                and row.get("recomposition_op") == "union"
            )
            name = row.get("segment_name", row.get("display_name", lid))
            color = tuple(row.get("color_rgb", (0.9, 0.7, 0.2)))
            if (not isinstance(name, str) or not name or len(color) != 3
                    or not all(isinstance(n, (int, float)) and 0 <= n <= 1 for n in color)):
                raise ValueError("Invalid native review layer presentation")
            record = layers.setdefault(lid, {
                "layer_id": lid, "logical_id": lid, "name": name, "color": color,
                "role": role, "additive": additive, "visible": role != "global",
                "cutoffs": [], "extent_by_cutoff": {}, "segments_by_cutoff": {},
            })
            # Match the legacy store's lowest-cutoff display name and color.
            if not record["cutoffs"] or cutoff < min(record["cutoffs"]):
                record["name"], record["color"] = name, color
            if cutoff in record["extent_by_cutoff"] or (record["role"], record["additive"]) != (role, additive):
                raise ValueError("Inconsistent native layer identity across cutoffs")
            if type(row["foreground_voxels"]) is not int or row["foreground_voxels"] < 0:
                raise ValueError("Invalid native review foreground count")
            if row["empty_segment"] is not (row["foreground_voxels"] == 0):
                raise ValueError("Native review empty flag disagrees with foreground count")
            record["cutoffs"].append(cutoff)
            record["extent_by_cutoff"][cutoff] = tuple(row["extent_xyz"])
            record["segments_by_cutoff"][cutoff] = {
                "segment_id": segment, "foreground_voxels": row["foreground_voxels"],
                "view_name": row.get("view_name"), "layer_key": row.get("layer_key"),
                "model_name": row.get("model_name"), "source_filename": row.get("source_filename"),
                "layer_role": row.get("layer_role"), "recomposition_op": row.get("recomposition_op"),
                "physical_view_name": row.get("physical_view_name"), "mask_kind": row.get("mask_kind"),
            }
    if not layers or sum(layer["role"] == "global" for layer in layers.values()) > 1:
        raise ValueError("Expected native review layers with at most one Global identity")
    for record in layers.values():
        record["cutoffs"].sort()
    return layers


def _read_plane(decoded, plane):
    remaining = memoryview(plane).cast("B")
    while remaining:
        count = decoded.readinto(remaining)
        if not count:
            raise ValueError("Grouped gzip channel is shorter than its native extent")
        remaining = remaining[count:]


def _channel_reader(raw, offset, length):
    from .reconciliation_io import _ReadRange

    try:
        from isal.igzip import IGzipFile
    except (ImportError, OSError):
        IGzipFile = gzip.GzipFile
    return IGzipFile(fileobj=_ReadRange(raw, offset, length), mode="rb")


def _write_band(db, band, band_z, y0, aligned_x, shape, chunks, lid, cutoff):
    """Insert occupied blocks from one bit-packed native Z strip."""
    bz, by, bx = chunks
    width = band.shape[2] * 8
    for iy in _positions(y0, y0 + band.shape[1], by):
        y = iy * by
        first_y, end_y = max(y, y0), min(y + by, y0 + band.shape[1])
        for ix in _positions(aligned_x, aligned_x + width, bx):
            x = ix * bx
            first_x, end_x = max(x, aligned_x), min(x + bx, aligned_x + width)
            selected = band[:, first_y - y0:end_y - y0, (first_x - aligned_x)//8:(end_x - aligned_x)//8]
            if not selected.any():
                continue
            block = np.zeros((min(bz, shape[0] - band_z), min(by, shape[1] - y),
                              (min(bx, shape[2] - x) + 7)//8), np.uint8)
            block[:, first_y - y:end_y - y, (first_x - x)//8:(end_x - x)//8] = selected[:block.shape[0]]
            data = zlib.compress(block, level=1)
            db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (lid, cutoff, band_z // bz, iy, ix, data, hashlib.sha256(data).digest()))


def _write_channel(db, raw, row, stored, origin, shape, chunks):
    """Verify binary/count/extent in a single pass and archive native packed blocks."""
    if math.prod(stored[1:]) * 2 > _PREPARATION_BYTES:
        raise ValueError("Grouped plane exceeds the preparation memory bound")
    plane = np.empty(stored[1:], np.uint8)
    cutoff = row["cutoff_cent"]
    expected = row["foreground_voxels"]
    seen = 0
    faces = np.zeros(6, np.bool_)
    strip_bytes = plane.nbytes * 2

    if expected:
        x0, x1, y0, y1, z0, z1 = row["extent_xyz"]
        bz, _, bx = chunks
        aligned_x = (x0 // bx) * bx
        width = ((x1 + 8 - aligned_x) // 8) * 8
        height = y1 - y0 + 1
        area = height * width
        strip_bytes += area * 2 + bz * (area // 8)
        if strip_bytes > _PREPARATION_BYTES:
            raise ValueError("Native preparation strip exceeds its memory bound. Reduce the Z chunk size")
        gathered = np.zeros((height, width), np.uint8)
        band = np.zeros((bz, height, width // 8), np.uint8)
        band_z = (z0 // bz) * bz
        inside_y = slice(y0 - origin[1], y1 + 1 - origin[1])
        inside_x = slice(x0 - origin[2], x1 + 1 - origin[2])
        gathered_x = slice(x0 - aligned_x, x1 + 1 - aligned_x)

    overview = None
    if row["role"] == "global":
        presence_bytes = math.prod(overview_layout(shape)[1])
        overview = NativeOverviewAccumulator(
            shape, extent_xyz=row["extent_xyz"],
            max_count_bytes=max(0, _PREPARATION_BYTES - strip_bytes - presence_bytes),
        )

    with _channel_reader(raw, row["gzip_offset"], row["gzip_bytes"]) as decoded:
        for z in range(origin[0], origin[0] + stored[0]):
            _read_plane(decoded, plane)
            maximum = int(plane.max())
            if maximum > 1:
                raise ValueError("Nonbinary voxel in native review layer")
            if not maximum:
                continue
            if not expected or not z0 <= z <= z1:
                raise ValueError("Native review foreground outside declared extent")
            tight_plane = plane[inside_y, inside_x]
            count = int(np.count_nonzero(tight_plane))
            if count != np.count_nonzero(plane):
                raise ValueError("Native review foreground outside declared extent")
            seen += count
            faces[0] |= tight_plane[:, 0].any()
            faces[1] |= tight_plane[:, -1].any()
            faces[2] |= tight_plane[0].any()
            faces[3] |= tight_plane[-1].any()
            faces[4] |= z == z0
            faces[5] |= z == z1

            current_band = (z // bz) * bz
            if current_band != band_z:
                _write_band(db, band, band_z, y0, aligned_x, shape, chunks, row["logical_id"], cutoff)
                band.fill(0)
                band_z = current_band
            gathered[:, gathered_x] = tight_plane
            band[z - band_z] = np.packbits(gathered, axis=1, bitorder="little")
            if overview is not None:
                overview.add((z, y0, x0), tight_plane[None, :, :])
        if decoded.read(1):
            raise ValueError("Grouped gzip channel is longer than its native extent")
    if seen != expected or (expected and not faces.all()):
        raise ValueError("Native review foreground count or exact bounds differ from its index")
    if expected:
        _write_band(db, band, band_z, y0, aligned_x, shape, chunks, row["logical_id"], cutoff)
    if overview is not None:
        overview.save(db, cutoff)


def prepare_native_review(manifest_path, destination, *, chunk_shape=_DEFAULT_CHUNK, intensity_path=None,
                          progress=None):
    """Prepare one portable archive from the indexed native grid of an XTA review."""
    from .reconciliation_io import ReferenceGeometry
    from .run_transport import _publish_fresh
    from .sweep_grouped_reader import _family, _relative_file, _review, group_stamp, inspect_indexed_group

    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Review archive already exists: {destination}")
    chunks = _chunk_shape(chunk_shape)
    family = _family(manifest_path.parent.parent)
    manifest_bytes = manifest_path.read_bytes()
    review = _review(manifest_path, family)
    if manifest_bytes != manifest_path.read_bytes():
        raise ValueError("Review manifest changed while opening")
    grid = [g for g in review["grids"] if g["grid_id"] == "native" and g["payload"] == "grouped"]
    if len(grid) != 1:
        raise ValueError("Expected one published native grouped grid")
    grid = grid[0]
    groups = grid["groups"]
    shape = _triple(grid["shape_tyx"], "native dimensions", minimum=1)
    geometry = {name: grid[name] for name in ("space", "directions_xyz", "origin_xyz")}
    native_ijk_to_ras(geometry)
    reference = ReferenceGeometry(shape, **geometry)
    cutoffs = review["cutoffs_cent"]
    if (not cutoffs or any(type(c) is not int or not 0 <= c <= 100 for c in cutoffs)
            or len(cutoffs) != len(set(cutoffs))):
        raise ValueError("Invalid native review cutoffs")
    cutoffs = sorted(cutoffs)
    default_cutoff = review.get("default_cutoff_cent", cutoffs[0])
    if type(default_cutoff) is not int or default_cutoff not in cutoffs:
        raise ValueError("Default cutoff differs from native review catalogue")
    source_record = family.get("inputs", {}).get("source")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    unique_paths = set()
    layers = _layer_catalogue(groups, cutoffs)
    metadata = {
        "schema": _SCHEMA, "shape": shape, "chunk_shape": chunks, "cutoffs": cutoffs, "default_cutoff": default_cutoff,
        "geometry": geometry, "sweep_id": review["sweep_id"], "layers": layers,
        "identity": hashlib.sha256(json.dumps([_SCHEMA, manifest_sha, chunks]).encode()).hexdigest(),
        "manifest_sha256": manifest_sha, "source_record": source_record,
    }
    if progress is not None:
        progress("start", 0, len(groups))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xta-native-review-", dir=destination.parent) as work:
        stage = Path(work) / "review.sqlite"
        with closing(sqlite3.connect(stage)) as db:
            # Staging can be rebuilt after interruption. Sync before publication.
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE review(document TEXT NOT NULL)")
            db.execute("INSERT INTO review VALUES (?)", (json.dumps(metadata, allow_nan=False),))
            db.execute(
                "CREATE TABLE blocks(lid TEXT NOT NULL, cutoff INTEGER NOT NULL, z INTEGER NOT NULL,"
                "y INTEGER NOT NULL,x INTEGER NOT NULL,packed BLOB NOT NULL,digest BLOB NOT NULL,"
                "PRIMARY KEY(lid,cutoff,z,y,x)) WITHOUT ROWID"
            )
            for number, group in enumerate(groups, 1):
                path = _relative_file(manifest_path.parent, group["path"], manifest_path.parent)
                if path in unique_paths:
                    raise ValueError("Native review repeats the same group file")
                unique_paths.add(path)
                with path.open("rb") as raw:
                    stamp = group_stamp(os.fstat(raw.fileno()))
                    _, _, stored, offset, _ = inspect_indexed_group(
                        path, group, stamp, reference, group["cutoff_cent"],
                    )
                    digest = hashlib.sha256()
                    for data in iter(lambda: raw.read(1024 * 1024), b""):
                        digest.update(data)
                    if digest.hexdigest() != group["sha256"]:
                        raise ValueError(f"Native group failed its SHA-256 check: {path}")
                    for row in group["segments"]:
                        _write_channel(db, raw, row, stored, offset[::-1], shape, chunks)
                    if group_stamp(os.fstat(raw.fileno())) != stamp or group_stamp(path.stat()) != stamp:
                        raise ValueError(f"Native group changed during preparation: {path}")
                if progress is not None:
                    progress("groups", number, len(groups))
            if progress is not None:
                progress("index", len(groups), len(groups))
            db.execute("CREATE INDEX blocks_region ON blocks(z,y,x,cutoff,lid)")
            if intensity_path is not None:
                from .native_review_intensity import write_native_intensity

                if progress is not None:
                    progress("intensity", len(groups), len(groups))
                write_native_intensity(db, intensity_path, shape, geometry, chunks, source_record=source_record)
            db.commit()
        if manifest_bytes != manifest_path.read_bytes():
            raise ValueError("Review manifest changed during preparation")
        with stage.open("rb+") as stream:
            os.fsync(stream.fileno())
        _publish_fresh(stage, destination)
    return destination
