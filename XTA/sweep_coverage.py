"""Accumulate direct-layer coverage while grouped review masks are being streamed."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .gzip_spans import GzipSpanWriter
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


class DirectCoverage:
    """One disk spool per cutoff, with only offsets and row bounds per plane in RAM.

    Inputs can arrive in any layer order. Recording only occupied, bit-packed
    rectangles avoids allocating a dense native volume per threshold or re-opening
    the source NRRDs for a separate sum. Private records are read once at publication.
    """

    def __init__(self, directory, geometry, direct_layer_count):
        self.geometry = geometry
        self.layer_count = direct_layer_count
        self.dtype = np.dtype("u1" if direct_layer_count <= 255 else
                              "<u2" if direct_layer_count <= 65535 else "<u4")
        self.heads = [-1] * geometry.shape_tyx[0]
        self.row_bounds = [None] * geometry.shape_tyx[0]
        self._spool = tempfile.TemporaryFile(mode="w+b", dir=directory)

    def close(self):
        self._spool.close()

    def observe(self, first_z, slab, extent_xyz, *, origin_yx=(0, 0)):
        """Record a binary source slab using the declared layer extent as an upper bound."""
        x0, x1, y0, y1, _, _ = extent_xyz
        origin_y, origin_x = origin_yx
        for offset, source in enumerate(slab):
            crop = source[y0-origin_y:y1-origin_y+1, x0-origin_x:x1-origin_x+1]
            rows = np.flatnonzero(np.any(crop, axis=1))
            if not rows.size:
                continue
            y = int(rows[0])
            crop = crop[y:int(rows[-1])+1]
            columns = np.flatnonzero(np.any(crop, axis=0))
            x = int(columns[0])
            crop = crop[:, x:int(columns[-1])+1]
            z = first_z + offset
            position = self._spool.tell()
            self._spool.write(_RECORD.pack(self.heads[z], y0+y, x0+x, *crop.shape))
            self._spool.write(memoryview(np.packbits(crop, axis=1, bitorder="little")))
            self.heads[z] = position
            first, stop = y0+y, y0+y+crop.shape[0]
            previous = self.row_bounds[z]
            self.row_bounds[z] = (first, stop) if previous is None else (
                min(first, previous[0]), max(stop, previous[1]))

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
        minimum_decode = width + (width+7)//8 if band_height else 0
        if band_height and decode_budget < minimum_decode:
            raise ValueError("Coverage memory_mib must fit the occupied row band and one decoded input row")
        pending = 0
        if member_pool is not None and band_height:
            occupied_bytes = row_bytes * sum(stop - first for first, stop in filter(None, self.row_bounds))
            pending, reserved = member_pool.plan(decode_budget - minimum_decode, occupied_bytes)
            decode_budget -= reserved
        buffer = np.empty((band_height, width), self.dtype)
        self._spool.flush()
        with path.open("xb") as raw:
            raw.write(("\n".join(header) + "\n\n").encode("ascii"))
            with GzipSpanWriter(raw, zero_member_bytes=member_bytes, write_buffer_bytes=codec_buffer,
                                member_pool=member_pool, max_pending_members=pending) as encoded:
                for head, bounds in zip(self.heads, self.row_bounds):
                    if head == -1:
                        encoded.write_zeros(height * row_bytes)
                        continue
                    first, stop = bounds
                    band = buffer[:stop-first]
                    band.fill(0)
                    while head != -1:
                        self._spool.seek(head)
                        head, y, x, h, w = _RECORD.unpack(self._spool.read(_RECORD.size))
                        stride = (w+7)//8
                        block_rows = min(h, decode_budget//(w+stride))
                        for row in range(0, h, block_rows):
                            rows = min(block_rows, h-row)
                            data = self._spool.read(rows * stride)
                            packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, stride)
                            dest = band[y-first+row:y-first+row+rows, x:x+w]
                            dest += np.unpackbits(packed, axis=1, count=w, bitorder="little")
                            del data, packed
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
