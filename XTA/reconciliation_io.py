"""Bounded, manifest-driven access to decomposed binary NRRD layers.

Arrays use C-order (t, y, x); NRRD's stored axes are (x, y, t). Each layer
keeps a bounded forward decoder for increasing slab reads. Backward access
reopens that layer, allowing a statistics pass followed by a voting pass.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np


_MIB = 1024 * 1024
_U8_TYPES = {"unsigned char", "uchar", "uint8", "uint8_t"}


def _shape(value: Any, field_name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three positive integers")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in value):
        raise ValueError(f"{field_name} must contain three positive integers")
    return tuple(int(v) for v in value)


def _budget(memory_mib: float) -> int:
    if isinstance(memory_mib, bool) or not math.isfinite(float(memory_mib)) or float(memory_mib) <= 0:
        raise ValueError("memory_mib must be positive and finite")
    result = int(float(memory_mib) * _MIB)
    if result < 1024:
        raise ValueError("memory_mib must provide at least 1024 bytes")
    return result


def _ints(value: str, count: int, name: str) -> tuple[int, ...]:
    fields = str(value).split()
    if len(fields) != count or any(re.fullmatch(r"[+-]?\d+", x) is None for x in fields):
        raise ValueError(f"Invalid {name}: {value!r}")
    return tuple(int(x) for x in fields)


def _vectors(value: str, count: int, name: str) -> tuple[tuple[float, float, float], ...]:
    parts = re.findall(r"\([^()]*\)", value)
    if len(parts) != count or re.sub(r"\([^()]*\)", "", value).strip():
        raise ValueError(f"Invalid {name}: {value!r}")
    result = []
    for part in parts:
        fields = part[1:-1].split(",")
        if len(fields) != 3:
            raise ValueError(f"Invalid {name}: {value!r}")
        vector = tuple(float(x) for x in fields)
        if not all(math.isfinite(x) for x in vector):
            raise ValueError(f"Non-finite {name}")
        result.append(vector)
    return tuple(result)


def _read_header(path: Path) -> tuple[dict[str, str], int]:
    """Parse the single embedded-data header without importing inference packages."""
    fields: dict[str, str] = {}
    with path.open("rb") as stream:
        magic = stream.readline(64).strip()
        if not re.fullmatch(rb"NRRD000[1-5]", magic):
            raise ValueError(f"Not a supported NRRD file: {path}")
        while True:
            line = stream.readline(1024 * 1024 + 1)
            if not line or len(line) > 1024 * 1024 or stream.tell() > 4 * _MIB:
                raise ValueError(f"Incomplete or excessive NRRD header: {path}")
            if not line.strip():
                return fields, stream.tell()
            if line.startswith(b"#"):
                continue
            text = line.decode("utf-8").rstrip("\r\n")
            if ":" not in text:
                raise ValueError(f"Invalid NRRD header field: {path}")
            key, value = text.split(":", 1)
            if value.startswith("="):
                value = value[1:]
            key, value = key.strip(), value.strip()
            if key in fields:
                raise ValueError(f"Duplicate NRRD field {key!r}: {path}")
            fields[key] = value


@dataclass(frozen=True)
class ReferenceGeometry:
    shape_tyx: tuple[int, int, int]
    space: str = "left-posterior-superior"
    directions_xyz: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    )
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _shape(self.shape_tyx, "reference shape")
        directions = np.asarray(self.directions_xyz, dtype=float)
        origin = np.asarray(self.origin_xyz, dtype=float)
        if directions.shape != (3, 3) or origin.shape != (3,) or not np.isfinite(directions).all() or not np.isfinite(origin).all():
            raise ValueError("Reference geometry needs finite 3D directions and origin")
        if float(np.linalg.det(directions)) == 0:
            raise ValueError("Reference geometry directions are singular")
        if not self.space or "\n" in self.space or "\r" in self.space:
            raise ValueError("Reference geometry requires a space name")


def _same_geometry(a: ReferenceGeometry, b: ReferenceGeometry) -> bool:
    return (a.shape_tyx == b.shape_tyx and a.space == b.space
            and np.allclose(a.directions_xyz, b.directions_xyz, rtol=0, atol=1e-9)
            and np.allclose(a.origin_xyz, b.origin_xyz, rtol=0, atol=1e-9))


def _inspect_layer(path: Path, metadata: dict[str, Any], reference_shape: tuple[int, int, int]):
    header, payload_offset = _read_header(path)
    if header.get("dimension") != "3" or header.get("type", "").lower() not in _U8_TYPES:
        raise ValueError(f"Expected a 3D uint8 binary NRRD: {path}")
    if header.get("kinds", "").split() != ["domain"] * 3:
        raise ValueError(f"Expected three domain axes: {path}")
    if any(k in header for k in ("data file", "datafile", "byte skip", "line skip")):
        raise ValueError(f"Detached or skipped NRRD payloads are not supported: {path}")
    encoding = header.get("encoding", "").lower()
    if encoding not in {"gzip", "gz", "raw"}:
        raise ValueError(f"Unsupported NRRD encoding {encoding!r}: {path}")
    stored_shape = tuple(reversed(_ints(header.get("sizes", ""), 3, "sizes")))
    _shape(stored_shape, "NRRD sizes")
    if metadata.get("stored_shape_tyx") is not None and _shape(metadata["stored_shape_tyx"], "stored_shape_tyx") != stored_shape:
        raise ValueError(f"Manifest stored shape disagrees with NRRD: {path}")
    if any(re.match(r"Segment[1-9]\d*_", key) for key in header):
        raise ValueError(f"Multiple segments are not supported: {path}")
    if header.get("Segment0_LabelValue", "1") != "1" or header.get("Segment0_Layer", "0") != "0":
        raise ValueError(f"Expected segment label 1 in layer 0: {path}")
    extent = _ints(header.get("Segment0_Extent", ""), 6, "Segment0_Extent")
    empty_extent = all(extent[2 * axis + 1] < extent[2 * axis] for axis in range(3))
    if not empty_extent:
        for axis, size in enumerate(reversed(stored_shape)):
            lo, hi = extent[2 * axis:2 * axis + 2]
            if lo < 0 or hi < lo or hi >= size:
                raise ValueError(f"Segment extent falls outside stored data: {path}")
    if metadata.get("empty_segment") is True and not empty_extent:
        raise ValueError(f"Empty manifest entry has a nonempty segment extent: {path}")
    offset_xyz = _ints(header.get("Segmentation_ReferenceImageExtentOffset", "0 0 0"), 3, "reference offset")
    offset_tyx = tuple(reversed(offset_xyz))
    if any(o < 0 or o + s > n for o, s, n in zip(offset_tyx, stored_shape, reference_shape)):
        raise ValueError(f"Stored crop lies outside the reference grid: {path}")
    if stored_shape != reference_shape and "Segmentation_ReferenceImageExtentOffset" not in header:
        raise ValueError(f"Cropped data require an explicit reference offset: {path}")
    if "Segmentation_ReferenceImageExtent" in header:
        reference_extent = _ints(header["Segmentation_ReferenceImageExtent"], 6, "reference extent")
        expected = tuple(v for size in reversed(reference_shape) for v in (0, size - 1))
        if reference_extent != expected:
            raise ValueError(f"NRRD reference extent disagrees with manifest: {path}")
    directions = _vectors(header.get("space directions", ""), 3, "space directions")
    stored_origin = _vectors(header.get("space origin", ""), 1, "space origin")[0]
    if "space" not in header:
        raise ValueError(f"NRRD space is missing: {path}")
    reference_origin = np.asarray(stored_origin) - np.asarray(offset_xyz) @ np.asarray(directions)
    geometry = ReferenceGeometry(reference_shape, header["space"], directions, tuple(float(x) for x in reference_origin))
    content_match = re.search(r"reference_shape_tyx=\((\d+),\s*(\d+),\s*(\d+)\)", header.get("content", ""))
    if content_match and tuple(int(x) for x in content_match.groups()) != reference_shape:
        raise ValueError(f"NRRD content reference shape disagrees with manifest: {path}")
    return header, payload_offset, stored_shape, offset_tyx, geometry, encoding, empty_extent


@dataclass
class FileLayer:
    layer_id: str
    metadata: dict[str, Any]
    shape_tyx: tuple[int, int, int]
    path: Path | None
    header: dict[str, str]
    stored_shape_tyx: tuple[int, int, int]
    offset_tyx: tuple[int, int, int]
    _payload_offset: int = field(repr=False)
    _encoding: str = field(repr=False)
    _empty: bool = field(repr=False)
    _owner: LayerCollection | None = field(default=None, repr=False)
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint8), init=False)
    _source: Any = field(default=None, init=False, repr=False)
    _payload: Any = field(default=None, init=False, repr=False)
    _position: int = field(default=0, init=False, repr=False)
    _eof_validated: bool = field(default=False, init=False, repr=False)

    def _open(self) -> None:
        if self._payload is not None:
            return
        assert self._owner is not None and self.path is not None
        if self._owner._open_count >= self._owner.max_open:
            raise OSError(f"Reconciliation exceeded max_open={self._owner.max_open}; increase it or close completed layer readers")
        try:
            self._source = self.path.open("rb")
            self._source.seek(self._payload_offset)
            self._payload = gzip.GzipFile(fileobj=self._source, mode="rb") if self._encoding in {"gzip", "gz"} else self._source
        except OSError as exc:
            if self._source is not None:
                self._source.close()
            self._source = None
            raise OSError(f"Cannot open reconciliation layer {self.path}: {exc}") from exc
        self._position = 0
        self._eof_validated = False
        self._owner._open_count += 1

    def _consume(self, stop: int, target: memoryview | None = None) -> None:
        self._open()
        assert self._owner is not None
        target_offset = 0
        while self._position < stop:
            block = self._payload.read(min(self._owner._chunk_bytes, stop - self._position))
            if not block:
                raise ValueError(f"NRRD payload is shorter than declared shape: {self.path}")
            values = np.frombuffer(block, dtype=np.uint8)
            if np.any(values > 1):
                raise ValueError(f"NRRD payload is not binary 0/1: {self.path}")
            if self._empty and np.any(values):
                raise ValueError(f"NRRD empty segment contains foreground: {self.path}")
            if target is not None:
                target[target_offset:target_offset + len(block)] = block
                target_offset += len(block)
            self._position += len(block)

    def _validate_eof(self) -> None:
        expected = math.prod(self.stored_shape_tyx)
        self._consume(expected)
        # Reading beyond the declared end also validates concatenated-member CRCs.
        if self._payload.read(1):
            raise ValueError(f"NRRD payload exceeds declared shape: {self.path}")
        self.close()
        self._eof_validated = True

    def read_slab(self, z0: int, z1: int) -> np.ndarray:
        """Return an owned uint8 slab in the collection's reference grid."""
        if self._owner is None or self._owner._closed:
            raise RuntimeError("Layer collection is closed")
        if (isinstance(z0, bool) or isinstance(z1, bool) or not isinstance(z0, (int, np.integer))
                or not isinstance(z1, (int, np.integer)) or not 0 <= z0 <= z1 <= self.shape_tyx[0]):
            raise ValueError("Slab bounds must satisfy 0 <= z0 <= z1 <= t")
        z0, z1 = int(z0), int(z1)
        shape = (z1 - z0, *self.shape_tyx[1:])
        if math.prod(shape) > self._owner._budget_bytes:
            raise ValueError("Requested slab exceeds memory_mib; request fewer slices")
        result = np.zeros(shape, dtype=np.uint8)
        if z0 == z1 or self.path is None:
            return result
        dz, dy, dx = self.offset_tyx
        sz, sy, sx = self.stored_shape_tyx
        lo, hi = max(z0, dz), min(z1, dz + sz)
        try:
            if hi > lo:
                byte_start = (lo - dz) * sy * sx
                if self._payload is not None and byte_start < self._position:
                    self.close()
                self._consume(byte_start)
                if (sy, sx) == self.shape_tyx[1:]:
                    target = memoryview(result[lo-z0:hi-z0]).cast("B")
                    self._consume((hi - dz) * sy * sx, target)
                else:
                    for z in range(lo, hi):
                        for y in range(sy):
                            target = memoryview(result[z-z0, dy+y, dx:dx+sx]).cast("B")
                            self._consume(self._position + sx, target)
            if z1 >= dz + sz and not self._eof_validated:
                self._validate_eof()
        except BaseException:
            self.close()
            raise
        return result

    def close(self) -> None:
        """Close the forward decoder; a later read opens it again."""
        if self._payload is not None:
            try:
                self._payload.close()
            finally:
                if self._source is not None:
                    self._source.close()
                self._source = self._payload = None
                if self._owner is not None:
                    self._owner._open_count -= 1
        self._position = 0


class LayerCollection(Sequence[FileLayer]):
    def __init__(self, manifest: dict[str, Any], layers: list[FileLayer], geometry: ReferenceGeometry,
                 workspace: Path, budget_bytes: int, max_open: int) -> None:
        self.manifest = manifest
        self.layers = tuple(layers)
        self.geometry = geometry
        self.shape_tyx = geometry.shape_tyx
        self.workspace = workspace
        self._budget_bytes = budget_bytes
        self._chunk_bytes = min(_MIB, max(1024, budget_bytes // 4))
        self.max_open = max_open
        self._open_count = 0
        self._closed = False
        for layer in self.layers:
            layer._owner = self

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

    def __iter__(self) -> Iterator[FileLayer]:
        return iter(self.layers)

    def __enter__(self) -> LayerCollection:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            for layer in self.layers:
                layer.close()
            self._closed = True


def read_layer_manifest(path: str | Path, *, workspace: str | Path, memory_mib: float = 256,
                        max_open: int = 512) -> LayerCollection:
    """Read additive union descriptors and validate their common reference grid.

    Confidence is never synthesized from run thresholds. Missing files are errors,
    except entries explicitly marked empty; unlisted views are not added.
    """
    budget = _budget(memory_mib)
    if isinstance(max_open, bool) or not isinstance(max_open, int) or max_open <= 0:
        raise ValueError("max_open must be a positive integer")
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("layers"), list):
        raise ValueError("Expected a NRRD layer manifest with a layers list")
    reference_shape = _shape(manifest.get("output_shape_tyx"), "output_shape_tyx")
    if re.sub(r"\s+", "", str(manifest.get("exported_axes", ""))).lower() != "(x,y,t)":
        raise ValueError("Manifest must declare exported_axes (X, Y, t)")
    if manifest.get("layer_count", len(manifest["layers"])) != len(manifest["layers"]):
        raise ValueError("Manifest layer_count disagrees with layers")
    layers = []
    filenames = set()
    common_geometry = None
    for metadata in manifest["layers"]:
        if not isinstance(metadata, dict):
            raise ValueError("Manifest layer entries must be objects")
        filename = metadata.get("filename")
        if not isinstance(filename, str) or not filename or filename in filenames:
            raise ValueError("Manifest filenames must be nonempty and unique")
        filenames.add(filename)
        if metadata.get("layer_role") != "additive_component" or metadata.get("recomposition_op") != "union":
            continue
        layer_path = (path.parent / filename).resolve()
        if not layer_path.is_relative_to(path.parent):
            raise ValueError(f"Layer path escapes its manifest directory: {filename}")
        if _shape(metadata.get("output_shape_tyx", list(reference_shape)), "layer output_shape_tyx") != reference_shape:
            raise ValueError(f"Layer output shape disagrees with reference: {filename}")
        if metadata.get("exported_axes") is not None and re.sub(r"\s+", "", str(metadata["exported_axes"])).lower() != "(x,y,t)":
            raise ValueError(f"Layer axes disagree with reference: {filename}")
        if not layer_path.is_file():
            if metadata.get("empty_segment") is not True:
                raise FileNotFoundError(f"Declared layer is missing: {layer_path}")
            layer = FileLayer(filename, dict(metadata), reference_shape, None, {}, reference_shape,
                              (0, 0, 0), 0, "raw", True)
        else:
            header, offset, stored, crop, geometry, encoding, empty = _inspect_layer(layer_path, metadata, reference_shape)
            if common_geometry is None:
                common_geometry = geometry
            elif not _same_geometry(common_geometry, geometry):
                raise ValueError(f"Layer spatial geometry disagrees with reference: {filename}")
            layer = FileLayer(filename, dict(metadata), reference_shape, layer_path, header, stored, crop, offset, encoding, empty)
        layers.append(layer)
    if common_geometry is None:
        raise ValueError("No additive layer has a readable spatial reference")
    return LayerCollection(manifest, layers, common_geometry, Path(workspace), budget, max_open)


def write_seg_nrrd(path: str | Path, *, shape_tyx: tuple[int, int, int],
                   read_slab: Callable[[int, int], np.ndarray], geometry: ReferenceGeometry | None = None,
                   segment_name: str = "Reconciled", color_rgb: tuple[float, float, float] = (0.2, 0.8, 0.4),
                   memory_mib: float = 256, chunk_slices: int = 16, overwrite: bool = False) -> dict[str, Any]:
    """Stream a binary single-segment output, publishing only after full validation."""
    shape = _shape(shape_tyx, "shape_tyx")
    budget = _budget(memory_mib)
    geometry = geometry or ReferenceGeometry(shape)
    if geometry.shape_tyx != shape:
        raise ValueError("Writer shape disagrees with spatial reference")
    if isinstance(chunk_slices, bool) or not isinstance(chunk_slices, int) or chunk_slices <= 0:
        raise ValueError("chunk_slices must be a positive integer")
    if len(color_rgb) != 3 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in color_rgb):
        raise ValueError("color_rgb must contain three values between zero and one")
    # Source slab, validation temporary, and serialization together stay bounded.
    per_slice = math.prod(shape[1:])
    depth = min(chunk_slices, budget // max(1, 3 * per_slice))
    if depth < 1:
        raise ValueError("memory_mib must fit at least three uint8 slices for writing")
    path = Path(path).resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".reconciliation-write-", dir=path.parent))
    payload_path = stage_dir / "payload.gz"
    final_stage = stage_dir / "output.seg.nrrd"
    extent_lo = np.asarray(shape, dtype=np.int64)
    extent_hi = np.full(3, -1, dtype=np.int64)
    foreground = 0
    try:
        with payload_path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) as compressed:
            for z0 in range(0, shape[0], depth):
                z1 = min(shape[0], z0 + depth)
                slab = np.asarray(read_slab(z0, z1))
                if slab.shape != (z1 - z0, *shape[1:]) or slab.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
                    raise ValueError("Writer read_slab must return the requested uint8 or bool TYX shape")
                if np.any(slab > 1):
                    raise ValueError("Writer requires binary values 0/1")
                foreground += int(np.count_nonzero(slab))
                for axis in range(3):
                    active = np.flatnonzero(np.any(slab, axis=tuple(i for i in range(3) if i != axis)))
                    if active.size:
                        offset = z0 if axis == 0 else 0
                        extent_lo[axis] = min(extent_lo[axis], int(active[0]) + offset)
                        extent_hi[axis] = max(extent_hi[axis], int(active[-1]) + offset)
                compressed.write(np.ascontiguousarray(slab, dtype=np.uint8).tobytes(order="C"))
        extent = tuple(v for axis in (2, 1, 0) for v in (int(extent_lo[axis]), int(extent_hi[axis]))) if foreground else (0, -1, 0, -1, 0, -1)
        clean_name = " ".join(str(segment_name).splitlines()).strip() or "Reconciled"
        vector = lambda v: "(" + ",".join(format(float(x), ".17g") for x in v) + ")"
        fields = ["NRRD0005", "type: uint8", "dimension: 3", "sizes: " + " ".join(str(x) for x in reversed(shape)),
                  f"space: {geometry.space}", "kinds: domain domain domain",
                  "space directions: " + " ".join(vector(v) for v in geometry.directions_xyz),
                  "space origin: " + vector(geometry.origin_xyz), "encoding: gzip",
                  "Segmentation_ContainedRepresentationNames:=Binary labelmap|",
                  "Segmentation_MasterRepresentation:=Binary labelmap", "Segmentation_SourceRepresentation:=Binary labelmap",
                  "Segment0_ID:=Segment_1", f"Segment0_Name:={clean_name}", "Segment0_NameAutoGenerated:=0",
                  "Segment0_Color:=" + " ".join(format(float(v), ".6g") for v in color_rgb),
                  "Segment0_ColorAutoGenerated:=0", "Segment0_LabelValue:=1", "Segment0_Layer:=0",
                  "Segment0_Extent:=" + " ".join(str(v) for v in extent)]
        with final_stage.open("wb") as destination, payload_path.open("rb") as payload:
            destination.write(("\n".join(fields) + "\n\n").encode("utf-8"))
            shutil.copyfileobj(payload, destination, length=min(_MIB, budget // 4))
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        os.replace(final_stage, path)
        return {"path": str(path), "shape_tyx": list(shape), "stored_shape_tyx": list(shape),
                "segment_extent_xyz": list(extent), "foreground_voxels": foreground,
                "empty_segment": foreground == 0, "bytes": path.stat().st_size}
    finally:
        for staged in (payload_path, final_stage):
            staged.unlink(missing_ok=True)
        stage_dir.rmdir()


__all__ = ["FileLayer", "LayerCollection", "ReferenceGeometry", "read_layer_manifest", "write_seg_nrrd"]
