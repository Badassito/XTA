"""Accumulate direct-layer coverage while grouped review masks are being streamed."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import tempfile
from threading import Lock

import numpy as np

from .gzip_spans import GzipSpanWriter
from .grouped_seg_nrrd import _plane_bounds
from .reconciliation_io import _budget


# Previous record for this z plane, crop start Y/X, crop height/width.
_RECORD = struct.Struct("<qIIII")
_COLORS = ((1, (94, 1, 166)), (2, (165, 31, 153)), (4, (214, 85, 109)),
           (8, (247, 145, 67)), (16, (249, 220, 36)))


def write_coverage_colors(path: Path, maximum: int, *, durable=False) -> None:
    """Every label is the exact count even when adjacent counts share a color."""
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("# Color table file\n0 Background 0 0 0 0\n")
        for value in range(1, maximum + 1):
            color = next(rgb for floor, rgb in reversed(_COLORS) if floor <= value)
            stream.write(f"{value} {value}_direct_layers {color[0]} {color[1]} {color[2]} 255\n")
        if durable:
            stream.flush()
            os.fsync(stream.fileno())


def _count_dtype(layer_count):
    return np.dtype("u1" if layer_count <= 255 else "<u2" if layer_count <= 65535 else "<u4")


@dataclass(frozen=True)
class _CoveragePlan:
    shape_tyx: tuple[int, int, int]
    row_bounds: tuple[tuple[int, int] | None, ...]
    dtype: np.dtype
    size_bytes: int


def plan_direct_coverage(shape_tyx, direct_extents):
    """Bound in-memory count bands using one declared extent per direct layer.

    At most the layers whose extents intersect a Z plane can vote on that
    plane. This bound also selects the internal count width.
    """
    slices, height, width = shape_tyx
    starts = np.full(slices, height, dtype=np.intp)
    stops = np.zeros(slices, dtype=np.intp)
    layer_counts = np.zeros(slices, dtype=np.intp)
    for x0, x1, y0, y1, z0, z1 in direct_extents:
        if x0 > x1 or y0 > y1 or z0 > z1:
            continue
        if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height and 0 <= z0 <= z1 < slices):
            raise ValueError("Coverage extent exceeds the grid")
        planes = slice(z0, z1 + 1)
        np.minimum(starts[planes], y0, out=starts[planes])
        np.maximum(stops[planes], y1 + 1, out=stops[planes])
        layer_counts[planes] += 1
    bounds = tuple((first, stop) if first < stop else None
                   for first, stop in zip(starts.tolist(), stops.tolist()))
    dtype = _count_dtype(int(layer_counts.max(initial=0)))
    size_bytes = sum(stop - first for first, stop in filter(None, bounds)) * width * dtype.itemsize
    return _CoveragePlan(tuple(shape_tyx), bounds, dtype, size_bytes)


class DirectCoverage:
    """Accumulate complete direct layers in memory or a bounded packed spool.

    Inputs may arrive concurrently and in any layer order. An admitted plan
    reserves only the occupied row bands. The fallback keeps only offsets and
    observed row bounds in memory and reads its packed crops at publication.
    """

    def __init__(self, directory, geometry, direct_layer_count, *, dense_plan=None):
        self.geometry = geometry
        self.layer_count = direct_layer_count
        self.dtype = _count_dtype(direct_layer_count)
        slices = geometry.shape_tyx[0]
        self.row_bounds = [None] * slices
        self._plan = dense_plan
        if dense_plan is None:
            self._heads = [-1] * slices
            self._spool = tempfile.TemporaryFile(mode="w+b", dir=directory)
            self._spool_lock = Lock()
        else:
            if dense_plan.shape_tyx != tuple(geometry.shape_tyx):
                raise ValueError("Coverage plan has the wrong grid")
            self._spool = None
            self._bands = [None] * slices
            self._band_locks = tuple(Lock() for _ in range(min(slices, 64)))

    def close(self):
        if self._spool is not None:
            self._spool.close()
        else:
            self._bands.clear()

    def observe(self, first_z, slab, extent_xyz, *, origin_yx=(0, 0)):
        """Record a binary source slab using the declared layer extent as an upper bound."""
        x0, x1, y0, y1, _, _ = extent_xyz
        origin_y, origin_x = origin_yx
        for offset, source in enumerate(slab):
            crop = source[y0-origin_y:y1-origin_y+1, x0-origin_x:x1-origin_x+1]
            bounds = _plane_bounds(crop)
            if bounds is None:
                continue
            top, bottom, left, right = bounds
            self.observe_plane(first_z + offset, crop[top:bottom, left:right], y0 + top, x0 + left)

    def observe_plane(self, z, crop, y, x):
        """Accumulate a validated nonempty binary crop at its reference coordinates."""
        if self._plan is None:
            self.observe_packed_plane(z, np.packbits(crop, axis=1, bitorder="little"), y, x, crop.shape[1])
            return
        with self._band_locks[z % len(self._band_locks)]:
            height, width = crop.shape
            bounds = self._plan.row_bounds[z]
            if bounds is None or not (bounds[0] <= y and y + height <= bounds[1]
                                      and 0 <= x and x + width <= self.geometry.shape_tyx[2]):
                raise ValueError("Coverage crop exceeds its planned band")
            first, stop = bounds
            band = self._bands[z]
            if band is None:
                band = self._bands[z] = np.zeros(
                    (stop - first, self.geometry.shape_tyx[2]), self._plan.dtype)
            band[y-first:y-first+height, x:x+width] += crop
            self._record_rows(z, y, y + height)

    def observe_packed_plane(self, z, packed, y, x, width):
        """Accumulate a validated crop of little-bit-order packed binary rows."""
        stride = (width + 7) // 8
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != stride:
            raise ValueError("Packed coverage crop has the wrong row format")
        height = packed.shape[0]
        if self._plan is not None:
            self.observe_plane(z, np.unpackbits(packed, axis=1, count=width, bitorder="little"), y, x)
            return
        packed = np.ascontiguousarray(packed)
        with self._spool_lock:
            position = self._spool.tell()
            self._spool.write(_RECORD.pack(self._heads[z], y, x, height, width))
            self._spool.write(memoryview(packed))
            self._heads[z] = position
            self._record_rows(z, y, y + height)

    def _record_rows(self, z, first, stop):
        previous = self.row_bounds[z]
        self.row_bounds[z] = (first, stop) if previous is None else (
            min(first, previous[0]), max(stop, previous[1]))

    def _read_spooled_band(self, z, first, band, decode_budget):
        band.fill(0)
        head = self._heads[z]
        while head != -1:
            self._spool.seek(head)
            head, y, x, height, width = _RECORD.unpack(self._spool.read(_RECORD.size))
            stride = (width + 7) // 8
            block_rows = min(height, decode_budget // (width + stride))
            for row in range(0, height, block_rows):
                rows = min(block_rows, height - row)
                packed = np.frombuffer(self._spool.read(rows * stride), dtype=np.uint8).reshape(rows, stride)
                band[y-first+row:y-first+row+rows, x:x+width] += np.unpackbits(
                    packed, axis=1, count=width, bitorder="little")
                del packed

    def write(self, path, *, sweep_id, grid_id, cutoff_cent, memory_mib=128, durable=False,
              member_pool=None):
        _, height, width = self.geometry.shape_tyx
        nrtype = {"u1":"uint8", "u2":"uint16", "u4":"uint32"}[f"u{self.dtype.itemsize}"]
        def vector(values):
            return "(" + ",".join(format(float(value), ".17g") for value in values) + ")"
        header = ["NRRD0005", f"type: {nrtype}", "dimension: 3",
                  f"space: {self.geometry.space}",
                  "sizes: " + " ".join(str(i) for i in reversed(self.geometry.shape_tyx)),
                  "space directions: " + " ".join(vector(v) for v in self.geometry.directions_xyz),
                  "kinds: domain domain domain", "space origin: " + vector(self.geometry.origin_xyz),
                  "encoding: gzip", "xta_role:=direct_layer_count", f"xta_sweep_id:={sweep_id}",
                  f"xta_grid_id:={grid_id}", f"xta_cutoff_cent:={cutoff_cent}",
                  f"xta_direct_layers:={self.layer_count}"]
        if self.dtype.itemsize > 1:
            header.append("endian: little")
        maximum = positive = votes = 0
        budget = _budget(memory_mib)
        member_bytes = 1 << (min(1024**2, budget//8).bit_length()-1)
        codec_buffer = min(64*1024, budget//16)
        band_height = max((bounds[1]-bounds[0] for bounds in self.row_bounds if bounds), default=0)
        row_bytes = width * self.dtype.itemsize
        band_bytes = band_height * row_bytes
        decode_budget = budget - band_bytes - member_bytes - codec_buffer
        # Leave the same member queue budget in both storage modes so the
        # encoder uses identical gzip boundaries, including at small budgets.
        minimum_decode = width + (width+7)//8 if band_height else 0
        if band_height and decode_budget < minimum_decode:
            raise ValueError("Coverage memory_mib must fit the occupied row band and one decoded input row")
        pending = 0
        if member_pool is not None and band_height:
            occupied_bytes = row_bytes * sum(stop - first for first, stop in filter(None, self.row_bounds))
            pending, reserved = member_pool.plan(decode_budget - minimum_decode, occupied_bytes)
            decode_budget -= reserved
        resident = self._plan is not None
        use_count_bands = resident and self._plan.dtype == self.dtype
        buffer = None if use_count_bands else np.empty((band_height, width), self.dtype)
        if self._spool is not None:
            self._spool.flush()
        with path.open("xb") as raw:
            raw.write(("\n".join(header) + "\n\n").encode("ascii"))
            with GzipSpanWriter(raw, zero_member_bytes=member_bytes, write_buffer_bytes=codec_buffer,
                                member_pool=member_pool, max_pending_members=pending) as encoded:
                for z, bounds in enumerate(self.row_bounds):
                    if bounds is None:
                        encoded.write_zeros(height * row_bytes)
                        continue
                    first, stop = bounds
                    if resident:
                        plan_first = self._plan.row_bounds[z][0]
                        band = self._bands[z][first-plan_first:stop-plan_first]
                        if not use_count_bands:
                            buffer[:stop-first] = band
                            band = buffer[:stop-first]
                    else:
                        band = buffer[:stop-first]
                        self._read_spooled_band(z, first, band, decode_budget)
                    positive += int(np.count_nonzero(band))
                    maximum = max(maximum, int(band.max()))
                    votes += int(band.sum(dtype=np.uint64))
                    encoded.write_zeros(first * row_bytes)
                    encoded.write(memoryview(band).cast("B"))
                    encoded.write_zeros((height-stop) * row_bytes)
            if durable:
                raw.flush()
                os.fsync(raw.fileno())
        return {"dtype":nrtype, "direct_layer_count": self.layer_count,
                "maximum_count":maximum, "positive_voxels":positive, "voxel_votes":votes,
                "row_band_bytes":band_bytes, "gzip_spans":dict(encoded.stats)}
