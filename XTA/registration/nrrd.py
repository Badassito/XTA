"""Nearest-neighbor transfer of integer NRRD labels between registered grids.

NRRD data axes are XYZ on disk and the bounded working array is ZYX. The
registration packet maps the original scan's native voxel-center indices. A
source reference header connects a cropped or reoriented labelmap's physical
coordinates to that native scan grid. An explicit full-grid assertion is also
available for labelmaps already stored in the scan's native axes.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import gzip
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping
import warnings

import numpy as np

from .packet import RegistrationPacket, load_packet

try:
    from isal.igzip import IGzipFile as _gzip_reader
except (ImportError, OSError):
    _gzip_reader = gzip.GzipFile


_INTEGER_TYPES = {
    "signed char": "i1",
    "int8": "i1",
    "int8_t": "i1",
    "uchar": "u1",
    "unsigned char": "u1",
    "uint8": "u1",
    "uint8_t": "u1",
    "short": "i2",
    "short int": "i2",
    "signed short": "i2",
    "signed short int": "i2",
    "int16": "i2",
    "int16_t": "i2",
    "ushort": "u2",
    "unsigned short": "u2",
    "unsigned short int": "u2",
    "uint16": "u2",
    "uint16_t": "u2",
    "int": "i4",
    "signed int": "i4",
    "int32": "i4",
    "int32_t": "i4",
    "uint": "u4",
    "unsigned int": "u4",
    "uint32": "u4",
    "uint32_t": "u4",
    "longlong": "i8",
    "long long": "i8",
    "long long int": "i8",
    "signed long long": "i8",
    "signed long long int": "i8",
    "int64": "i8",
    "int64_t": "i8",
    "ulonglong": "u8",
    "unsigned long long": "u8",
    "unsigned long long int": "u8",
    "uint64": "u8",
    "uint64_t": "u8",
}
_CANONICAL_TYPES = {
    "i1": "int8",
    "u1": "uint8",
    "i2": "int16",
    "u2": "uint16",
    "i4": "int32",
    "u4": "uint32",
    "i8": "int64",
    "u8": "uint64",
}
_ANATOMICAL_SPACES = {
    "lps": ("left-posterior-superior", (1, 1, 1)),
    "left-posterior-superior": ("left-posterior-superior", (1, 1, 1)),
    "ras": ("left-posterior-superior", (-1, -1, 1)),
    "right-anterior-superior": ("left-posterior-superior", (-1, -1, 1)),
    "las": ("left-posterior-superior", (1, -1, 1)),
    "left-anterior-superior": ("left-posterior-superior", (1, -1, 1)),
}
_SEGMENT_FIELD = re.compile(r"Segment(\d+)_(\w+)")
_REFERENCE_OFFSET = "Segmentation_ReferenceImageExtentOffset"
_REFERENCE_EXTENT = "Segmentation_ReferenceImageExtent"
_EMPTY_EXTENT = (0, -1, 0, -1, 0, -1)


def _pynrrd():
    try:
        import nrrd
    except ImportError as exc:
        raise RuntimeError(
            "Scalar NRRD transfer requires pynrrd. Install xta[registration] or pynrrd."
        ) from exc
    return nrrd


def _header(path: Path) -> tuple[dict[str, Any], int]:
    """Use pynrrd's geometry parsing while bounding the embedded header size."""
    lines = []
    length = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline(1_048_577)
            length += len(line)
            if not line or len(line) > 1_048_576 or length > 4_194_304:
                raise ValueError(f"Incomplete or excessive NRRD header in {path}")
            lines.append(line.decode("utf-8", errors="ignore"))
            if not line.strip():
                break
    return dict(_pynrrd().read_header(lines)), length


def _shape(header: Mapping[str, Any]) -> tuple[int, int, int]:
    sizes = np.asarray(header.get("sizes", []))
    if int(header.get("dimension", 0)) != 3 or sizes.shape != (3,):
        raise ValueError("Registration NRRD transfer requires a scalar three-dimensional grid")
    if sizes.dtype.kind not in "iu" or np.any(sizes <= 0):
        raise ValueError("NRRD sizes must contain three positive integers")
    kinds = list(header.get("kinds", ["domain"] * 3))
    if len(kinds) != 3 or any(str(kind).lower() not in {"domain", "space"} for kind in kinds):
        raise ValueError("Registration NRRD transfer requires three scalar spatial axes")
    return tuple(int(value) for value in sizes[::-1])


def _segmentation_axes(header):
    sizes = np.asarray(header["sizes"])
    kinds = [str(kind).lower() for kind in header.get("kinds", ())]
    if int(header["dimension"]) == 3:
        _shape(header)
        return (0, 1, 2), None
    if int(header["dimension"]) != 4 or sizes.shape != (4,) or len(kinds) != 4:
        raise ValueError("Slicer segmentations need three spatial axes and an optional list axis")
    if sizes.dtype.kind not in "iu" or np.any(sizes <= 0) or kinds.count("list") != 1:
        raise ValueError("Expected positive NRRD sizes and exactly one segmentation list axis")
    list_axis = kinds.index("list")
    spatial = tuple(axis for axis in range(4) if axis != list_axis)
    if any(kinds[axis] not in {"domain", "space"} for axis in spatial):
        raise ValueError("Expected three spatial axes in the segmentation")
    return spatial, list_axis


@dataclass
class _Segment:
    fields: dict[str, Any]
    layer: int
    value: int
    box: list[int] = field(default_factory=lambda: list(_EMPTY_EXTENT))
    count: int = 0

    def observe(self, plane, z, first_row, first_column):
        selected = plane == self.value
        count = int(np.count_nonzero(selected))
        if count == 0:
            return
        y = np.flatnonzero(np.any(selected, axis=1))
        x = np.flatnonzero(np.any(selected, axis=0))
        bounds = [
            int(x[0]) + first_column, int(x[-1]) + first_column,
            int(y[0]) + first_row, int(y[-1]) + first_row, z, z,
        ]
        if self.count:
            for axis in range(0, 6, 2):
                self.box[axis] = min(self.box[axis], bounds[axis])
                self.box[axis + 1] = max(self.box[axis + 1], bounds[axis + 1])
        else:
            self.box[:] = bounds
        self.count += count


def _segments(header, layer_count, dtype):
    by_index = {}
    for key, value in header.items():
        match = _SEGMENT_FIELD.fullmatch(str(key))
        if match:
            by_index.setdefault(int(match[1]), {})[match[2]] = value
    if not by_index:
        raise ValueError("Missing segment-aware label metadata in the Slicer NRRD")
    for key in ("Segmentation_SourceRepresentation", "Segmentation_MasterRepresentation"):
        if key in header and header[key] != "Binary labelmap":
            raise ValueError("Slicer registration requires a binary labelmap representation")
    ids, categories, segments = set(), set(), []
    for _, values in sorted(by_index.items()):
        try:
            segment_id = str(values["ID"])
            layer = int(values["Layer"])
            value = int(values["LabelValue"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Incomplete segment-aware metadata: ID, Layer and LabelValue are required") from exc
        if (
            not segment_id.strip()
            or segment_id in ids
            or (layer, value) in categories
            or not 0 <= layer < layer_count
            or not np.iinfo(dtype).min <= value <= np.iinfo(dtype).max
            or value == 0
        ):
            raise ValueError("Invalid or duplicate ID, layer or label value in the segmentation")
        ids.add(segment_id)
        categories.add((layer, value))
        segments.append(_Segment(values, layer, value))
    return segments


def _reference_extent(shape):
    return " ".join(str(value) for size in shape[::-1] for value in (0, size - 1))


def _aligned_crop_matrix(header, stored_shape, source_shape, *, segmentation):
    matrix = np.eye(4)
    if segmentation and _REFERENCE_EXTENT in header:
        extent = " ".join(str(header[_REFERENCE_EXTENT]).split())
        if extent != _reference_extent(source_shape):
            raise ValueError("The segmentation reference extent disagrees with the packet source grid")
    if segmentation and _REFERENCE_OFFSET in header:
        try:
            offset = tuple(map(int, str(header[_REFERENCE_OFFSET]).split()))
        except ValueError as exc:
            raise ValueError("The segmentation crop requires a valid reference offset") from exc
        if len(offset) != 3 or any(
            start < 0 or start + size > full
            for start, size, full in zip(offset, stored_shape[::-1], source_shape[::-1])
        ):
            raise ValueError("The segmentation crop lies outside the packet source grid")
        matrix[:3, 3] = -np.asarray(offset, dtype=np.float64)
    elif stored_shape != source_shape:
        raise ValueError(
            "An explicitly aligned labelmap must cover the full source grid or contain a segmentation crop offset"
        )
    return matrix


def _transferred_crop(stored_shape, target_shape, target_to_label):
    """Bound the potentially nonzero target raster by warped source voxel edges."""
    label_to_target = np.linalg.inv(target_to_label)
    edges = np.asarray(list(product(*[(-0.5, size - 0.5) for size in stored_shape[::-1]])))
    corners = edges @ label_to_target[:3, :3].T + label_to_target[:3, 3]
    limit = np.asarray(target_shape[::-1])
    start = np.clip(np.floor(corners.min(axis=0)), 0, limit).astype(int)
    stop = np.clip(np.ceil(corners.max(axis=0)) + 1, 0, limit).astype(int)
    if np.any(stop <= start):
        return (0, 0, 0), (1, 1, 1)
    if np.all(stop - start == 1):
        # Slicer reserves a single stored voxel for an empty segmentation.
        for axis in range(3):
            if stop[axis] < limit[axis]:
                stop[axis] += 1
                break
            if start[axis] > 0:
                start[axis] -= 1
                break
    return tuple(map(int, start)), tuple(map(int, stop))


def _scalar_dtype(header: Mapping[str, Any]) -> np.dtype:
    name = str(header.get("type", "")).lower()
    if name not in _INTEGER_TYPES:
        raise ValueError("Categorical NRRD labels must have an integer scalar type")
    dtype = np.dtype(_INTEGER_TYPES[name])
    if dtype.itemsize > 1:
        endian = str(header.get("endian", "")).lower()
        if endian not in {"little", "big"}:
            raise ValueError("Multibyte integer NRRDs require an explicit endian field")
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def _geometry(header: Mapping[str, Any]) -> tuple[np.ndarray, str, tuple[str, ...] | None]:
    directions = np.asarray(header.get("space directions", []), dtype=np.float64)
    origin = np.asarray(header.get("space origin", []), dtype=np.float64)
    if (
        directions.shape != (3, 3)
        or origin.shape != (3,)
        or not np.isfinite(directions).all()
        or not np.isfinite(origin).all()
    ):
        raise ValueError(
            "Physical label alignment requires finite space directions and space origin"
        )
    if abs(float(np.linalg.det(directions))) < 1e-12:
        raise ValueError("NRRD space directions are singular")
    space = str(header.get("space", "")).strip().lower()
    canonical, signs = _ANATOMICAL_SPACES.get(space, (space, (1, 1, 1)))
    basis = np.diag(signs)
    matrix = np.eye(4)
    matrix[:3, :3] = basis @ directions.T
    matrix[:3, 3] = basis @ origin
    units = header.get("space units")
    units = tuple(str(value).strip() for value in units) if units is not None else None
    if units is not None and len(units) != 3:
        raise ValueError("NRRD space units must have three entries")
    return matrix, canonical, units


def _reference(value: Any, packet: RegistrationPacket, role: str) -> dict[str, Any]:
    header = dict(value) if isinstance(value, Mapping) else _header(Path(value))[0]
    if _shape(header) != packet.shape(role):
        raise ValueError(f"The {role} reference shape does not match the packet scan grid")
    _geometry(header)
    directions = np.asarray(header["space directions"], np.float64)
    if not np.allclose(
        np.linalg.norm(directions, axis=1), packet.spacing(role), rtol=1e-5, atol=1e-8
    ):
        raise ValueError(
            f"The {role} reference voxel sizes disagree with the packet; use matching physical units"
        )
    return header


def _axis_alignment(header: Mapping[str, Any]) -> dict[str, Any]:
    """Report the canonical world axis that each stored reference axis follows.

    XTA's own NRRD exports store movie column, row and frame along +X, +Y and +Z.
    Other orientations can be valid scanner geometry, but they are also what a
    reoriented or resampled copy looks like, so callers warn rather than reject.
    """
    linear = _geometry(header)[0][:3, :3]
    unit = linear / np.linalg.norm(linear, axis=0)
    dominant = np.argmax(np.abs(unit), axis=0)
    stored = [
        ("+" if unit[world, axis] > 0 else "-") + "XYZ"[world]
        for axis, world in enumerate(dominant)
    ]
    oblique = not np.allclose(np.abs(unit[dominant, np.arange(3)]), 1.0, atol=1e-6)
    return dict(
        stored_axes_world=stored,
        oblique=bool(oblique),
        canonical=bool(stored == ["+X", "+Y", "+Z"] and not oblique),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@contextmanager
def _label_array(path: Path, header: Mapping[str, Any], offset: int, shape, work_dir: Path | None):
    dtype = _scalar_dtype(header)
    if any(key in header for key in ("data file", "datafile", "byte skip", "line skip")):
        raise ValueError("Use an embedded raw or gzip NRRD without detached payloads or skips")
    encoding = str(header.get("encoding", "")).lower()
    if encoding not in {"raw", "gzip", "gz"}:
        raise ValueError("Registration NRRD input supports raw and gzip encodings")
    expected_bytes = int(np.prod(shape, dtype=object)) * dtype.itemsize
    temporary = None
    volume = None
    try:
        if encoding == "raw":
            if path.stat().st_size != offset + expected_bytes:
                raise ValueError("Raw NRRD payload length disagrees with its declared dimensions")
            array_path, array_offset = path, offset
        else:
            temporary = tempfile.TemporaryDirectory(prefix="xta-registration-labels-", dir=work_dir)
            array_path = Path(temporary.name) / "labels.raw"
            remaining = expected_bytes
            with path.open("rb") as source, array_path.open("wb") as target:
                source.seek(offset)
                with _gzip_reader(fileobj=source, mode="rb") as payload:
                    while remaining:
                        chunk = payload.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(
                                "Compressed NRRD payload ends before its declared dimensions"
                            )
                        target.write(chunk)
                        remaining -= len(chunk)
                    if payload.read(1):
                        raise ValueError("Compressed NRRD payload exceeds its declared dimensions")
            array_offset = 0
        volume = np.memmap(array_path, dtype=dtype, mode="r", offset=array_offset, shape=shape)
        yield volume
    finally:
        if volume is not None:
            volume._mmap.close()
        if temporary is not None:
            temporary.cleanup()


def _output_header(
    packet: RegistrationPacket, role: str, template: dict[str, Any] | None
) -> dict[str, Any]:
    if template is None:
        return {
            "space dimension": 3,
            "space directions": np.diag(packet.spacing(role)),
            "space origin": np.zeros(3),
            "content": "XTA registered scalar labels; coordinates relative to first scan voxel center",
        }
    keys = ("space", "space dimension", "space directions", "space origin", "space units")
    return {
        **{key: template[key] for key in keys if key in template},
        "content": "XTA registered scalar labels on target reference grid",
    }


def _write_header(
    stream, shape: tuple[int, int, int], dtype: np.dtype, geometry: Mapping[str, Any], encoding: str,
    *, segments=None, layers=None, reference_shape=None, offset_xyz=(0, 0, 0),
) -> None:
    def vector(values):
        return "(" + ",".join(format(float(value), ".17g") for value in values) + ")"

    sizes = tuple(shape[::-1])
    directions = " ".join(vector(row) for row in geometry["space directions"])
    kinds = "domain domain domain"
    if layers is not None:
        sizes = (layers, *sizes)
        kinds = "list " + kinds
        directions = "none " + directions
    fields = [
        "NRRD0005",
        "type: " + _CANONICAL_TYPES[dtype.kind + str(dtype.itemsize)],
        f"dimension: {len(sizes)}",
        "sizes: " + " ".join(str(value) for value in sizes),
        "kinds: " + kinds,
    ]
    if dtype.itemsize > 1:
        fields.append("endian: little")
    fields.extend(["encoding: " + encoding])
    if "space" in geometry:
        fields.append("space: " + str(geometry["space"]))
    else:
        fields.append("space dimension: 3")
    fields.extend(
        [
            "space directions: " + directions,
            "space origin: " + vector(geometry["space origin"]),
        ]
    )
    if "space units" in geometry:
        fields.append(
            "space units: "
            + " ".join(json.dumps(str(unit), ensure_ascii=True) for unit in geometry["space units"])
        )
    fields.append("content: " + str(geometry["content"]))
    if segments is not None:
        fields.extend([
            "Segmentation_ContainedRepresentationNames:=Binary labelmap|",
            "Segmentation_MasterRepresentation:=Binary labelmap",
            "Segmentation_SourceRepresentation:=Binary labelmap",
            _REFERENCE_EXTENT + ":=" + _reference_extent(reference_shape),
            _REFERENCE_OFFSET + ":=" + " ".join(map(str, offset_xyz)),
        ])
        for index, segment in enumerate(segments):
            extent = (
                [value - offset_xyz[axis // 2] for axis, value in enumerate(segment.box)]
                if segment.count else _EMPTY_EXTENT
            )
            values = {**segment.fields, "Extent": " ".join(map(str, extent))}
            fields.extend(f"Segment{index}_{key}:={value}" for key, value in values.items())
    if any(any(character in line for character in "\r\n\0") for line in fields):
        raise ValueError("NRRD header values must fit on one line")
    stream.write(("\n".join(fields) + "\n\n").encode("utf-8"))


def _mapped_block_indices(matrix, limits, xx, yy, z):
    indices = np.empty((3, yy.shape[0], xx.shape[1]), dtype=np.int64)
    for axis in range(3):
        coord = (
            matrix[axis, 0] * xx
            + matrix[axis, 1] * yy
            + matrix[axis, 2] * z
            + matrix[axis, 3]
        )
        indices[axis] = np.floor(np.clip(coord + 0.5, -1, limits[axis])).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < limits[:, None, None]), axis=0)
    return indices, valid


def _mask_samples(mask: np.ndarray, indices_xyz: np.ndarray) -> np.ndarray:
    indexes = indices_xyz[::-1] if mask.ndim == 3 else indices_xyz[1::-1]
    return mask[tuple(np.clip(axis, 0, size - 1) for axis, size in zip(indexes, mask.shape))]


def _publish_complete_file(stage: Path, destination: Path) -> None:
    """Publish without replacing another writer's output."""
    if os.name == "nt":
        os.rename(stage, destination)
    else:
        os.link(stage, destination)


def transfer_nrrd_labelmap(
    packet: RegistrationPacket | str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    direction: str = "fixed-to-moving",
    source_reference: Mapping[str, Any] | str | Path | None = None,
    target_reference: Mapping[str, Any] | str | Path | None = None,
    source_grid_aligned: bool = False,
    work_dir: str | Path | None = None,
    block_rows: int = 128,
    encoding: str = "gzip",
    coverage_output: str | Path | None = None,
    background_value: int = 0,
) -> dict[str, Any]:
    """Transfer categorical labels and write a separate known-observation mask.

    Physical alignment requires a source reference header whose axes correspond
    to the original scan's native XYZ grid. Its shape and voxel sizes must agree
    with the packet. The labelmap may have its own crop, origin and directions.
    Alternatively, ``source_grid_aligned=True`` asserts that the input uses the
    native scan's exact index axes. An aligned cropped segmentation supplies
    its native origin via the Slicer reference offset. A target reference supplies
    the output world geometry. Without one the output records the packet's
    spacing in an unnamed local coordinate system at origin zero. A reference
    whose stored axes do not follow +X, +Y, +Z is still used, with a warning;
    the provenance JSON records each reference's path, header and axis alignment.

    Input and output payloads are streamed. Compressed input uses a temporary
    disk-backed array because an arbitrary rotation requires random access.
    Supports embedded raw/gzip integer labelmaps and Slicer binary segmentations
    with three spatial axes and an optional list axis. Output segment metadata
    retains layer assignments and describes the warped target geometry.
    """
    packet = packet if isinstance(packet, RegistrationPacket) else load_packet(packet)
    if not packet.usable:
        raise ValueError("This packet has no validated registration mapping")
    direction = direction.replace("-", "_")
    if direction not in {"fixed_to_moving", "moving_to_fixed"}:
        raise ValueError("direction must be fixed-to-moving or moving-to-fixed")
    source_role, target_role = (
        ("fixed", "moving") if direction == "fixed_to_moving" else ("moving", "fixed")
    )
    source_shape, target_shape = packet.shape(source_role), packet.shape(target_role)
    source_path, destination = Path(input_path), Path(output_path)
    coverage = (
        Path(coverage_output)
        if coverage_output is not None
        else destination.with_name(destination.stem + "_coverage.nrrd")
    )
    sidecar = destination.with_suffix(destination.suffix + ".registration.json")
    outputs = [destination, coverage, sidecar]
    if len({path.resolve() for path in outputs}) != 3 or any(path.exists() for path in outputs):
        raise FileExistsError(
            "Choose three distinct, new output paths for labels, coverage and metadata"
        )
    if destination.suffix.lower() != ".nrrd" or coverage.suffix.lower() != ".nrrd":
        raise ValueError("Output label and coverage files must use the embedded .nrrd format")
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    if encoding not in {"raw", "gzip"}:
        raise ValueError("Output encoding must be raw or gzip")
    header, offset = _header(source_path)
    segmentation = any(re.match(r"^(Segment\d+_|Segmentation_)", str(key)) for key in header)
    spatial_axes, layer_axis = (
        _segmentation_axes(header) if segmentation else ((0, 1, 2), None)
    )
    label_shape = (
        tuple(int(header["sizes"][axis]) for axis in reversed(spatial_axes))
        if segmentation else _shape(header)
    )
    array_shape = tuple(int(size) for size in reversed(header["sizes"]))
    layers = None if layer_axis is None else int(header["sizes"][layer_axis])
    dtype = _scalar_dtype(header).newbyteorder("<")
    segments = _segments(header, layers if layers is not None else 1, dtype) if segmentation else None
    if (
        isinstance(background_value, bool)
        or not isinstance(background_value, (int, np.integer))
        or not np.iinfo(dtype).min <= background_value <= np.iinfo(dtype).max
    ):
        raise ValueError(
            "background_value must be an integer representable by the source label type"
        )
    if segmentation and background_value != 0:
        raise ValueError("Slicer segmentations use zero as background")
    if source_reference is None:
        if not source_grid_aligned:
            raise ValueError(
                "Provide source_reference or explicitly assert source_grid_aligned=True"
            )
        source_template = None
        native_to_label = _aligned_crop_matrix(header, label_shape, source_shape, segmentation=segmentation)
    else:
        if source_grid_aligned:
            raise ValueError(
                "Choose physical source_reference alignment or source_grid_aligned=True"
            )
        source_template = _reference(source_reference, packet, source_role)
        label_header = header
        if layer_axis is not None:
            directions = np.asarray(header.get("space directions", []))
            if directions.shape != (4, 3) or not np.isnan(directions[layer_axis]).all():
                raise ValueError("The segmentation's list axis must have no spatial direction")
            label_header = {**header, "space directions": directions[list(spatial_axes)]}
        label_world, label_space, label_units = _geometry(label_header)
        scan_world, scan_space, scan_units = _geometry(source_template)
        if label_space != scan_space or label_units != scan_units:
            raise ValueError(
                "Label and source reference must use compatible coordinate spaces and identical space units"
            )
        native_to_label = np.linalg.inv(label_world) @ scan_world
    target_template = (
        _reference(target_reference, packet, target_role) if target_reference is not None else None
    )
    geometry = _output_header(packet, target_role, target_template)
    if segmentation:
        geometry.setdefault("space", "left-posterior-superior")
        geometry["content"] = "XTA registered Slicer segmentation on target scan grid"
    references = {}
    for name, value, template, role in (
        ("source", source_reference, source_template, source_role),
        ("target", target_reference, target_template, target_role),
    ):
        if template is None:
            references[name] = None
            continue
        alignment = _axis_alignment(template)
        references[name] = dict(
            path=None if isinstance(value, Mapping) else str(Path(value).resolve()),
            header=_json_value(template),
            axis_alignment=alignment,
        )
        if not alignment["canonical"]:
            warnings.warn(
                f"The {name} reference stores its axes as {', '.join(alignment['stored_axes_world'])}"
                f"{' (oblique)' if alignment['oblique'] else ''}, not XTA's +X, +Y, +Z export "
                f"layout. It must store the {role} movie's column, row and frame in native order; "
                "a reoriented or resampled copy gives misplaced labels.",
                UserWarning,
                stacklevel=2,
            )
    target_to_source = np.linalg.inv(packet.matrix(direction))
    target_to_label = native_to_label @ target_to_source
    shared_source_grid = label_shape == source_shape and np.array_equal(
        target_to_label, target_to_source
    )
    if segmentation:
        warp_start, warp_stop = _transferred_crop(label_shape, target_shape, target_to_label)
        warp_shape = tuple(stop - start for start, stop in zip(warp_start[::-1], warp_stop[::-1]))
        warp_geometry = {
            **geometry,
            "space origin": np.asarray(geometry["space origin"])
            + np.asarray(warp_start) @ np.asarray(geometry["space directions"]),
        }
    else:
        warp_start, warp_stop = (0, 0, 0), target_shape[::-1]
    x0, y0, z0 = warp_start
    x1, y1, z1 = warp_stop
    # Coverage still maps rows at full target width, and sampling can retain
    # both source and label indices plus the output and gathered label bytes.
    index_bytes_per_row = 64 * target_shape[2] + 48 * (x1 - x0)
    label_bytes_per_row = 2 * (x1 - x0) * (layers or 1) * dtype.itemsize
    rows_per_block = min(block_rows, max(1, (64 * 1024 * 1024) // (
        index_bytes_per_row + label_bytes_per_row
    )))
    source_mask, target_mask = packet.mask(source_role), packet.mask(target_role)
    for role, mask in ((source_role, source_mask), (target_role, target_mask)):
        shape = packet.shape(role)
        if mask is not None and (mask.dtype != np.bool_ or mask.shape not in (shape, shape[1:])):
            raise ValueError(f"The {role} observation mask disagrees with its native scan grid")
    temporary_root = Path(work_dir) if work_dir is not None else None
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    stages = []
    published = []
    observed = 0
    foreground = 0
    label_limits = np.asarray(label_shape[::-1])
    native_limits = np.asarray(source_shape[::-1])
    try:
        for path in outputs:
            fd, stage = tempfile.mkstemp(
                prefix="." + path.name + ".", suffix=".partial", dir=path.parent
            )
            os.close(fd)
            stages.append(Path(stage))
        with (
            _label_array(source_path, header, offset, array_shape, temporary_root) as labels,
            ExitStack() as stack,
        ):
            label_output = stack.enter_context(stages[0].open("wb"))
            label_stream = label_output
            if segmentation:
                # Segment bounds are known only after sampling. Spool the
                # encoded raster without materializing the target volume.
                label_stream = stack.enter_context(tempfile.TemporaryFile(mode="w+b", dir=temporary_root))
                label_payload = label_stream
            else:
                _write_header(label_output, target_shape, dtype, geometry, encoding)
            if layer_axis is not None:
                labels = np.moveaxis(labels, 3 - layer_axis, -1)
            coverage_stream = stack.enter_context(stages[1].open("wb"))
            coverage_geometry = {
                **geometry,
                "content": "Known registration observation coverage; 1 observed and 0 unavailable",
            }
            _write_header(
                coverage_stream, target_shape, np.dtype("u1"), coverage_geometry, encoding
            )
            if encoding == "gzip":
                label_stream = stack.enter_context(
                    gzip.GzipFile(fileobj=label_stream, mode="wb", compresslevel=3, mtime=0)
                )
                coverage_stream = stack.enter_context(
                    gzip.GzipFile(fileobj=coverage_stream, mode="wb", compresslevel=3, mtime=0)
                )
            xx = np.arange(target_shape[2], dtype=np.float64)[None, :]
            for z in range(target_shape[0]):
                for start in range(0, target_shape[1], rows_per_block):
                    end = min(target_shape[1], start + rows_per_block)
                    yy = np.arange(start, end, dtype=np.float64)[:, None]
                    native_index, valid = _mapped_block_indices(
                        target_to_source, native_limits, xx, yy, z
                    )
                    if source_mask is not None:
                        valid &= _mask_samples(source_mask, native_index)
                    if target_mask is not None:
                        target_plane = target_mask if target_mask.ndim == 2 else target_mask[z]
                        valid &= target_plane[start:end]
                    first_row, last_row = max(start, y0), min(end, y1)
                    if z0 <= z < z1 and first_row < last_row:
                        rows = slice(first_row - start, last_row - start)
                        sampled = valid[rows, x0:x1]
                        if shared_source_grid:
                            label_index = native_index[:, rows, x0:x1]
                        else:
                            label_index, label_valid = _mapped_block_indices(
                                target_to_label, label_limits, xx[:, x0:x1], yy[rows], z
                            )
                            # Cropped segmentations implicitly label unstored
                            # native voxels as background. Plain cropped maps
                            # give no label information outside their raster.
                            if segmentation:
                                sampled = sampled & label_valid
                            else:
                                sampled &= label_valid
                        shape = sampled.shape if layers is None else (*sampled.shape, layers)
                        block = np.full(shape, background_value, dtype=dtype)
                        block[sampled] = labels[tuple(axis[sampled] for axis in label_index[::-1])]
                        label_stream.write(memoryview(block).cast("B"))
                        if layers is None:
                            foreground += int(np.count_nonzero(block != background_value))
                        else:
                            foreground += int(np.count_nonzero(np.any(block, axis=-1)))
                        if segmentation and np.any(sampled):
                            for segment in segments:
                                segment.observe(
                                    block if layers is None else block[..., segment.layer],
                                    z, first_row, x0,
                                )
                        del block
                    coverage_stream.write(memoryview(valid.astype(np.uint8)).cast("B"))
                    observed += int(np.count_nonzero(valid))
            if segmentation:
                if encoding == "gzip":
                    label_stream.close()
                label_payload.seek(0)
                _write_header(label_output, warp_shape, dtype, warp_geometry, encoding,
                              segments=segments, layers=layers, reference_shape=target_shape,
                              offset_xyz=warp_start)
                shutil.copyfileobj(label_payload, label_output, length=1024 * 1024)
        result = dict(
            input=str(source_path.resolve()),
            output=str(destination.resolve()),
            coverage_output=str(coverage.resolve()),
            direction=direction.replace("_", "-"),
            source_role=source_role,
            target_role=target_role,
            source_grid_shape_zyx=list(source_shape),
            label_stored_shape_zyx=list(label_shape),
            target_grid_shape_zyx=list(target_shape),
            input_dtype=str(_scalar_dtype(header)),
            output_dtype=str(dtype),
            interpolation="nearest neighbor with floor(index + 0.5)",
            label_grid_from_source_native_xyz=native_to_label.tolist(),
            target_native_to_source_native_xyz=target_to_source.tolist(),
            output_geometry=_json_value(geometry),
            source_header=_json_value(header),
            source_reference=references["source"],
            target_reference=references["target"],
            output_geometry_mode=(
                "Target reference world coordinates"
                if target_template is not None
                else "Local scan coordinates with packet voxel sizes and zero origin"
            ),
            packet_pair=packet.metadata["pair"],
            background_value=int(background_value),
            observed_voxels=observed,
            unavailable_voxels=int(np.prod(target_shape, dtype=object)) - observed,
            output_foreground_voxels=foreground,
            coverage_semantics="One only where the target is observed, the mapped source is observed, and the input labelmap has a voxel. Zero-filled unavailable voxels require this coverage mask.",
        )
        if segmentation:
            result["output_geometry"] = _json_value(warp_geometry)
            result["output_stored_shape_zyx"] = list(warp_shape)
            result["output_reference_offset_xyz"] = list(warp_start)
            if target_template is None:
                result["output_geometry_mode"] = (
                    "Local LPS scan coordinates with packet voxel sizes and zero reference origin"
                )
            result["coverage_semantics"] = (
                "One where the target and mapped native source are both observed. "
                "Outside the stored segmentation crop, observed voxels have background label zero."
            )
            result["segments"] = [
                dict(id=segment.fields["ID"], layer=segment.layer, label_value=segment.value,
                     extent_xyz=segment.box, foreground_voxels=segment.count)
                for segment in segments
            ]
        stages[2].write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        for stage, path in zip(stages, outputs):
            owned = stage.stat()
            _publish_complete_file(stage, path)
            published.append((path, owned))
        return result
    except BaseException:
        for path, owned in published:
            try:
                current = path.stat()
            except FileNotFoundError:
                continue
            if (current.st_ino, current.st_dev, current.st_size, current.st_mtime_ns) == (
                owned.st_ino,
                owned.st_dev,
                owned.st_size,
                owned.st_mtime_ns,
            ):
                path.unlink(missing_ok=True)
        raise
    finally:
        for stage in stages:
            stage.unlink(missing_ok=True)
