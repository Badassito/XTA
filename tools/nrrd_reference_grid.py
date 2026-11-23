"""Reference-grid decoding of published XTA layer NRRDs, cropped or full.

Sink-written TTA layers (NrrdLayerSink's default ``cropped`` layout) store
only their segment bounding box: ``sizes`` is the stored box, ``space origin`` is
its first voxel, and Slicer's ``Segmentation_ReferenceImageExtentOffset`` /
``Segmentation_ReferenceImageExtent`` place it inside the reference grid. A
full-layout file, or any NRRD without that offset field, is its own reference
grid. Slicer's own seg.nrrd saves carry only the offset; their reference grid is
the extent in Slicer's ``Reference image geometry`` conversion parameter, and an
offset-only box at offset zero without that geometry is read as its own grid.
Every reader here presents C-order ``(t, Y, X)`` reference-grid data, so a
tool's hashes and verdicts are identical for both layouts.

Manifest-driven tools use ``open_manifest_layers``: the supported
``XTA.reconciliation_io.read_layer_manifest(include_all_layers=True)`` reader and
its offset-aware ``read_slab``. The single-file readers serve tools that receive a
bare layer path; they also accept plain (non-segmentation) binary NRRDs.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.reconciliation_io import _read_header, read_layer_manifest  # noqa: E402
from XTA.sweep_grouped_reader import (  # noqa: E402
    LayerCatalogueRef, locate_run_layer_catalog, read_layer_catalog,
)

MIB = 1024 * 1024
OFFSET_FIELD = 'Segmentation_ReferenceImageExtentOffset'
EXTENT_FIELD = 'Segmentation_ReferenceImageExtent'
_U8_TYPES = {'unsigned char', 'uchar', 'uint8', 'uint8_t'}
_EMPTY_EXTENT = (0, -1, 0, -1, 0, -1)


def _integers(text: str, count: int, name: str) -> tuple[int, ...]:
    parts = str(text).split()
    if len(parts) != count or any(re.fullmatch(r'[+-]?\d+', part) is None for part in parts):
        raise ValueError(f'Invalid NRRD {name}: {text!r}')
    return tuple(int(part) for part in parts)


def _vectors(text: str) -> list[tuple[float, ...]]:
    return [tuple(float(value) for value in group.split(','))
            for group in re.findall(r'\(([^()]*)\)', str(text))]


def _canonical_number(value: float) -> str:
    return format(float(value) + 0.0, '.17g')


def _slicer_conversion_parameters(fields: dict) -> dict[str, str]:
    """Slicer ``Segmentation_ConversionParameters`` as {name: value} (``name|value|help&...``)."""
    parameters = {}
    for entry in str(fields.get('Segmentation_ConversionParameters') or '').split('&'):
        name, _, rest = entry.partition('|')
        if name.strip():
            parameters[name.strip()] = rest.partition('|')[0].strip()
    return parameters


def _slicer_reference_extent(fields: dict, path: Path) -> tuple[int, ...] | None:
    """Extent ``x0 x1 y0 y1 z0 z1`` of Slicer's saved reference image geometry, or None.

    The value is the IJK-to-RAS matrix (16 numbers) followed by the reference extent in
    the stored labelmap's data-axis order (the frame of the reference offset), as read by
    ``XTA.pta_inputs._slicer_reference_geometry_shape``. An oversampled labelmap is not
    stored on that grid, so its geometry cannot place the box.
    """
    parameters = _slicer_conversion_parameters(fields)
    value = parameters.get('Reference image geometry', '')
    if not value:
        return None
    try:
        numbers = [float(part) for part in value.split(';') if part.strip()]
    except ValueError:
        numbers = []
    if (len(numbers) != 22 or not all(math.isfinite(number) for number in numbers)
            or not all(number.is_integer() for number in numbers[16:])):
        raise ValueError(f'Invalid Slicer reference image geometry {value!r}: {path}')
    if parameters.get('Oversampling factor', '1') not in ('1', '1.0'):
        raise ValueError(f'Oversampled Slicer segmentation geometry is not its stored grid: {path}')
    return tuple(int(number) for number in numbers[16:])


@dataclass(frozen=True)
class ReferenceRaster:
    """Placement of one NRRD's stored raster inside its reference grid.

    Shapes are C-order. ``stored_shape``/``reference_shape`` are ``(t, Y, X)``, or
    ``(n, t, Y, X)`` for a grouped file with a trailing NRRD list axis; files of
    other dimensionality are passed through uncropped. ``offset_tyx`` is the
    reference index of the first stored voxel. ``stored_shape`` is ``None`` only
    for an uncropped header without ``sizes``, which supports whole-payload hashing.
    """

    path: Path
    fields: dict
    payload_offset: int
    encoding: str
    stored_shape: tuple[int, ...] | None
    reference_shape: tuple[int, ...] | None
    offset_tyx: tuple[int, int, int]

    @property
    def cropped(self) -> bool:
        return self.stored_shape != self.reference_shape or any(self.offset_tyx)

    @property
    def spatial_reference_shape(self) -> tuple[int, int, int]:
        if self.reference_shape is None or len(self.reference_shape) not in (3, 4):
            raise ValueError(f'NRRD has no 3D reference grid: {self.path}')
        return tuple(self.reference_shape[-3:])

    @property
    def spatial_stored_shape(self) -> tuple[int, int, int]:
        if self.stored_shape is None or len(self.stored_shape) not in (3, 4):
            raise ValueError(f'NRRD has no 3D stored raster: {self.path}')
        return tuple(self.stored_shape[-3:])

    def directions_xyz(self) -> np.ndarray | None:
        vectors = _vectors(self.fields.get('space directions', ''))[:3]
        if len(vectors) != 3 or any(len(vector) != 3 for vector in vectors):
            return None
        return np.asarray(vectors, dtype=np.float64)

    def reference_origin_xyz(self) -> np.ndarray | None:
        """World position of reference voxel (0,0,0); a cropped origin is its first stored voxel."""
        origins = _vectors(self.fields.get('space origin', ''))
        if len(origins) != 1 or len(origins[0]) != 3:
            return None
        origin = np.asarray(origins[0], dtype=np.float64)
        if not any(self.offset_tyx):
            return origin
        directions = self.directions_xyz()
        if directions is None:
            raise ValueError(f'A cropped NRRD needs space directions to place its origin: {self.path}')
        return origin - np.asarray(tuple(reversed(self.offset_tyx)), dtype=np.float64) @ directions

    def reference_spatial_fields(self) -> dict[str, str]:
        """Canonical ``sizes`` and ``space origin`` of the reference grid (layout independent)."""
        result = {}
        if self.reference_shape is not None:
            result['sizes'] = ' '.join(str(value) for value in self._nrrd_order(self.reference_shape))
        origin = self.reference_origin_xyz()
        if origin is not None:
            result['space origin'] = '(' + ','.join(_canonical_number(value) for value in origin) + ')'
        return result

    def reference_segment_extent(self, index: int = 0) -> tuple[int, ...] | None:
        """``SegmentN_Extent`` (x0 x1 y0 y1 t0 t1) in reference indices; empty stays empty."""
        raw = self.fields.get(f'Segment{int(index)}_Extent')
        if raw is None:
            return None
        extent = _integers(raw, 6, f'Segment{int(index)}_Extent')
        if any(extent[2 * axis + 1] < extent[2 * axis] for axis in range(3)):
            return extent
        offset_xyz = tuple(reversed(self.offset_tyx))
        return tuple(extent[2 * axis + side] + offset_xyz[axis] for axis in range(3) for side in (0, 1))

    def _nrrd_order(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(shape) == 4:
            return (*reversed(shape[1:]), shape[0])
        return tuple(reversed(shape))


def raster_from_fields(path: str | Path, fields: dict, payload_offset: int, *,
                       reference_shape_tyx: Sequence[int] | None = None) -> ReferenceRaster:
    """Resolve the stored box and reference grid of already parsed NRRD ``fields``."""
    path = Path(path)
    if fields.get('type', 'uint8').lower() not in _U8_TYPES:
        raise ValueError(f'Expected a uint8 NRRD: {path}')
    encoding = fields.get('encoding', 'raw').lower()
    if encoding not in {'gzip', 'gz', 'raw'}:
        raise ValueError(f'Unsupported NRRD encoding {encoding!r}: {path}')
    if any(key in fields for key in ('data file', 'datafile', 'byte skip', 'byteskip',
                                     'line skip', 'lineskip')):
        raise ValueError(f'Detached or skipped NRRD payloads are unsupported: {path}')
    sizes = None
    if 'sizes' in fields:
        sizes = tuple(int(value) for value in fields['sizes'].split())
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError(f'Invalid NRRD sizes: {path}')
        if 'dimension' in fields and int(fields['dimension']) != len(sizes):
            raise ValueError(f'NRRD sizes disagree with its dimension: {path}')
    kinds = fields.get('kinds', '').split()
    grouped = sizes is not None and len(sizes) == 4 and len(kinds) == 4 and kinds[3] == 'list'
    stored = None
    if sizes is not None:
        stored = ((sizes[3], *reversed(sizes[:3])) if grouped else tuple(reversed(sizes)))
    raw_offset = fields.get(OFFSET_FIELD)
    raw_extent = fields.get(EXTENT_FIELD)
    if raw_offset is None:
        if reference_shape_tyx is not None and stored is not None and tuple(stored[-3:]) != tuple(
                int(value) for value in reference_shape_tyx):
            raise ValueError(f'NRRD shape {stored} differs from reference {tuple(reference_shape_tyx)}: {path}')
        return ReferenceRaster(path, dict(fields), int(payload_offset), encoding, stored, stored, (0, 0, 0))
    if stored is None or len(stored) not in (3, 4):
        raise ValueError(f'A cropped NRRD needs 3D sizes: {path}')
    offset_xyz = _integers(raw_offset, 3, OFFSET_FIELD)
    declared = None
    # XTA's explicit extent wins; Slicer's own saves declare theirs in the conversion parameters.
    extent = (_integers(raw_extent, 6, EXTENT_FIELD) if raw_extent is not None
              else _slicer_reference_extent(fields, path))
    if extent is not None:
        if any(extent[2 * axis + 1] < extent[2 * axis] for axis in range(3)):
            raise ValueError(f'Invalid NRRD reference extent {extent}: {path}')
        offset_xyz = tuple(offset_xyz[axis] - extent[2 * axis] for axis in range(3))
        declared = tuple(extent[2 * axis + 1] - extent[2 * axis] + 1 for axis in (2, 1, 0))
    if reference_shape_tyx is not None:
        reference = tuple(int(value) for value in reference_shape_tyx)
        if len(reference) != 3 or min(reference) <= 0:
            raise ValueError(f'Invalid reference shape {reference_shape_tyx!r}')
        if declared is not None and declared != reference:
            raise ValueError(f'NRRD reference extent {declared} differs from {reference}: {path}')
    elif declared is not None:
        reference = declared
    elif not any(offset_xyz):
        # An offset-only box at the origin with no declared grid is its own reference grid.
        reference = tuple(stored[-3:])
    else:
        raise ValueError(f'A cropped NRRD needs its reference extent or a reference shape: {path}')
    offset_tyx = tuple(reversed(offset_xyz))
    if any(o < 0 or o + s > n for o, s, n in zip(offset_tyx, stored[-3:], reference)):
        raise ValueError(f'Stored box {stored} at {offset_tyx} lies outside reference {reference}: {path}')
    reference_full = (stored[0], *reference) if len(stored) == 4 else reference
    return ReferenceRaster(path, dict(fields), int(payload_offset), encoding, stored,
                           reference_full, offset_tyx)


def inspect_reference_raster(path: str | Path, *,
                             reference_shape_tyx: Sequence[int] | None = None) -> ReferenceRaster:
    """Parse one inline NRRD header and place its stored raster in the reference grid."""
    fields, payload_offset = _read_header(Path(path))
    return raster_from_fields(path, fields, payload_offset, reference_shape_tyx=reference_shape_tyx)


def _as_raster(source, reference_shape_tyx=None) -> ReferenceRaster:
    if isinstance(source, ReferenceRaster):
        return source
    return inspect_reference_raster(source, reference_shape_tyx=reference_shape_tyx)


@contextmanager
def _payload(raster: ReferenceRaster):
    with raster.path.open('rb') as raw:
        raw.seek(raster.payload_offset)
        if raster.encoding in {'gzip', 'gz'}:
            # GzipFile decodes concatenated members and verifies every CRC.
            with gzip.GzipFile(fileobj=raw, mode='rb') as decoded:
                yield decoded
        else:
            yield raw


def _read_exact(stream, count: int, path: Path) -> bytearray:
    data = bytearray(count)
    view = memoryview(data)
    position = 0
    while position < count:
        received = stream.readinto(view[position:])
        if not received:
            raise ValueError(f'NRRD payload is shorter than its declared shape: {path}')
        position += received
    return data


def _skip(stream, count: int, path: Path) -> None:
    remaining = int(count)
    while remaining > 0:
        block = stream.read(min(remaining, 4 * MIB))
        if not block:
            raise ValueError(f'NRRD payload is shorter than its declared shape: {path}')
        remaining -= len(block)


def iter_reference_slabs(source, depth: int = 8, *, start: int = 0, stop: int | None = None,
                         reference_shape_tyx: Sequence[int] | None = None,
                         verify_end: bool | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(z0, uint8[count, Y, X])`` reference slabs of a 3D NRRD in increasing t.

    Only the stored payload up to ``stop`` is decoded; slabs outside a cropped box are
    zeros without payload reads. ``verify_end`` (default: when ``stop`` is the last
    slice) also requires the payload to end exactly at its declared size.
    """
    raster = _as_raster(source, reference_shape_tyx)
    if raster.stored_shape is None or len(raster.stored_shape) != 3:
        raise ValueError(f'Reference slabs require a 3D NRRD: {raster.path}')
    if isinstance(depth, bool) or int(depth) < 1:
        raise ValueError('depth must be a positive integer')
    t, height, width = raster.reference_shape
    stop = t if stop is None else int(stop)
    start = int(start)
    if not 0 <= start <= stop <= t:
        raise ValueError(f'Slab bounds must satisfy 0 <= start <= stop <= {t}')
    verify_end = (stop == t) if verify_end is None else bool(verify_end)
    stored_t, stored_h, stored_w = raster.stored_shape
    dz, dy, dx = raster.offset_tyx
    stored_plane = stored_h * stored_w
    full_rows = (stored_h, stored_w) == (height, width)
    with _payload(raster) as stream:
        position = 0
        for z0 in range(start, stop, int(depth)):
            count = min(int(depth), stop - z0)
            slab = np.zeros((count, height, width), np.uint8)
            lo, hi = max(z0, dz), min(z0 + count, dz + stored_t)
            if hi > lo:
                begin = (lo - dz) * stored_plane
                _skip(stream, begin - position, raster.path)
                data = _read_exact(stream, (hi - lo) * stored_plane, raster.path)
                position = (hi - dz) * stored_plane
                box = np.frombuffer(data, np.uint8).reshape(hi - lo, stored_h, stored_w)
                if full_rows:
                    slab[lo - z0:hi - z0] = box
                else:
                    slab[lo - z0:hi - z0, dy:dy + stored_h, dx:dx + stored_w] = box
            yield z0, slab
        if verify_end:
            _skip(stream, stored_t * stored_plane - position, raster.path)
            if stream.read(1):
                raise ValueError(f'NRRD payload exceeds its declared shape: {raster.path}')


def read_reference_planes(source, start: int, stop: int, *,
                          reference_shape_tyx: Sequence[int] | None = None
                          ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return reference slices ``[start, stop)`` and the reference ``(t, Y, X)`` shape."""
    raster = _as_raster(source, reference_shape_tyx)
    reference = raster.spatial_reference_shape
    if int(stop) <= int(start):
        if not 0 <= int(start) == int(stop) <= reference[0]:
            raise ValueError(f'Slab bounds must satisfy 0 <= start <= stop <= {reference[0]}')
        return np.zeros((0, *reference[1:]), np.uint8), reference
    slabs = list(iter_reference_slabs(raster, int(stop) - int(start), start=start, stop=stop))
    return slabs[0][1], reference


def read_reference_volume(source, *, reference_shape_tyx: Sequence[int] | None = None) -> np.ndarray:
    """Decode a whole NRRD onto its reference grid (``(t,Y,X)``, or ``(n,t,Y,X)`` grouped)."""
    raster = _as_raster(source, reference_shape_tyx)
    if raster.stored_shape is None:
        raise ValueError(f'NRRD sizes are required to decode a volume: {raster.path}')
    if len(raster.stored_shape) == 3:
        volume = np.zeros(raster.reference_shape, np.uint8)
        for z0, slab in iter_reference_slabs(raster, max(1, 64 * MIB // max(
                1, raster.reference_shape[1] * raster.reference_shape[2]))):
            volume[z0:z0 + len(slab)] = slab
        return volume
    with _payload(raster) as stream:
        data = _read_exact(stream, math.prod(raster.stored_shape), raster.path)
        if stream.read(1):
            raise ValueError(f'NRRD payload exceeds its declared shape: {raster.path}')
    stored = np.frombuffer(data, np.uint8).reshape(raster.stored_shape)
    if not raster.cropped:
        return stored
    volume = np.zeros(raster.reference_shape, np.uint8)
    dz, dy, dx = raster.offset_tyx
    st, sy, sx = raster.spatial_stored_shape
    volume[:, dz:dz + st, dy:dy + sy, dx:dx + sx] = stored
    return volume


def reference_payload_sha256(source, *, chunk_bytes: int = 8 * MIB) -> str:
    """SHA-256 of the C-order reference-grid bytes, equal to a full-layout payload hash.

    Uncropped files hash their decoded payload stream directly (no ``sizes`` needed);
    cropped files are reconstructed one bounded reference slab at a time.
    """
    raster = _as_raster(source)
    digest = hashlib.sha256()
    if not raster.cropped:
        with _payload(raster) as stream:
            while block := stream.read(int(chunk_bytes)):
                digest.update(block)
        return digest.hexdigest()
    if len(raster.stored_shape) != 3:
        digest.update(memoryview(read_reference_volume(raster)).cast('B'))
        return digest.hexdigest()
    _t, height, width = raster.reference_shape
    depth = max(1, int(chunk_bytes) // max(1, height * width))
    for _z0, slab in iter_reference_slabs(raster, depth):
        digest.update(memoryview(slab).cast('B'))
    return digest.hexdigest()


def resolve_layer_catalogue(source, *, grid_id=None, cutoff_cent=None):
    """Select layers from a physical catalogue, cutoff receipt or managed review."""
    if cutoff_cent is not None and (type(cutoff_cent) is not int or not 0 <= cutoff_cent <= 100):
        raise ValueError("Cutoff must be an integer from 0 through 100")
    if isinstance(source, LayerCatalogueRef):
        if grid_id is not None or cutoff_cent is not None:
            raise ValueError("The layer catalogue already selects a grid and cutoff")
        return source
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        if cutoff_cent is not None and not (path / "manifest.json").is_file():
            raise ValueError("A cutoff selector requires a sweep review manifest")
        path = path / "manifest.json"
        if not path.is_file():
            return locate_run_layer_catalog(path, grid_id or "native")
    if path.name == "manifest.json":
        document = json.loads(path.read_bytes())
        if document.get("schema") == "xta.sweep_review/1":
            if cutoff_cent is None:
                raise ValueError("Select a cutoff for a sweep review manifest")
            selected_grid = grid_id or "native"
            branches = [branch for grid in document["grids"] if grid["grid_id"] == selected_grid
                        for branch in grid["branches"] if branch["cutoff_cent"] == cutoff_cent]
            if len(branches) != 1:
                raise ValueError(f"Expected one {selected_grid} layer catalogue at cutoff {cutoff_cent}")
            branch = branches[0]
            if branch.get("source_catalog") == "embedded":
                return LayerCatalogueRef(path, selected_grid, cutoff_cent)
            run_path = (path.parent / branch["run_manifest"]).resolve()
            if not run_path.is_relative_to(path.parent.parent):
                raise ValueError("Cutoff receipt escaped the study containing the review")
            return locate_run_layer_catalog(run_path, selected_grid)
        if "layers" not in document:
            if cutoff_cent is not None:
                raise ValueError("A cutoff selector requires a sweep review manifest")
            return locate_run_layer_catalog(path, grid_id or "native")
    if grid_id is not None or cutoff_cent is not None:
        raise ValueError("Grid and cutoff selectors require a cutoff run or sweep review manifest")
    return path


def manifest_volume_memory_mib(manifest_path: str | Path | LayerCatalogueRef, *,
                               minimum_mib: float = 256.0) -> float:
    """A read_slab budget large enough for one whole reference volume of a layer manifest."""
    document = read_layer_catalog(manifest_path)
    shape = tuple(int(value) for value in document['output_shape_tyx'])
    return max(float(minimum_mib), math.prod(shape) / MIB + 1.0)


def open_manifest_layers(manifest_path: str | Path | LayerCatalogueRef, *, filenames: Sequence[str] | None = None,
                         memory_mib: float | None = None, workspace: str | Path | None = None,
                         max_open: int = 512):
    """Open manifest layers through the supported offset-aware reconciliation reader.

    Every listed layer is included (``include_all_layers=True``); ``filenames``
    restricts and orders them, so validation covers only the layers a tool reads.
    ``memory_mib=None`` permits whole-volume ``read_slab`` calls (a cap, not an
    allocation). Use as a context manager; ``collection.shape_tyx`` is the grid.
    """
    path = Path(manifest_path)
    manifest_bytes = None
    if filenames is not None or memory_mib is None:
        document = read_layer_catalog(manifest_path)
    if filenames is not None:
        rows = {row.get('filename'): row for row in document.get('layers', []) if isinstance(row, dict)}
        missing = [name for name in filenames if name not in rows]
        if missing:
            raise KeyError(f'Layers absent from {path}: {missing}')
        selected = [rows[name] for name in filenames]
        manifest_bytes = json.dumps(dict(document, layers=selected, layer_count=len(selected))).encode('utf-8')
    if memory_mib is None:
        shape = tuple(int(value) for value in document['output_shape_tyx'])
        memory_mib = max(256.0, math.prod(shape) / MIB + 1.0)
    return read_layer_manifest(manifest_path, workspace=Path(workspace) if workspace is not None else path.parent,
                               memory_mib=float(memory_mib), max_open=int(max_open),
                               include_all_layers=True, manifest_bytes=manifest_bytes)


__all__ = [
    'ReferenceRaster', 'inspect_reference_raster', 'iter_reference_slabs', 'manifest_volume_memory_mib',
    'open_manifest_layers', 'raster_from_fields', 'read_reference_planes', 'read_reference_volume',
    'reference_payload_sha256', 'resolve_layer_catalogue',
]
