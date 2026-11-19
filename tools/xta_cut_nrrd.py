"""Stream an XTA unit-LPS NRRD through an axis-aligned right-side cut.

Arrays are stored in (frame, Y, X) order. The output retains columns before
``first_removed`` and either zeros the remaining columns or omits them.

A cropped segmentation stores only a box at ``Segmentation_ReferenceImageExtentOffset``
(equal to its integer space origin) inside ``Segmentation_ReferenceImageExtent``.
Its cut keeps every retained voxel at the same reference position as cutting the
full-grid file; crop mode narrows the reference extent to the cut grid.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import re

import numpy as np


_TYPE_BYTES = {
    "signed char": 1, "int8": 1, "int8_t": 1,
    "uchar": 1, "unsigned char": 1, "uint8": 1, "uint8_t": 1,
    "short": 2, "short int": 2, "signed short": 2,
    "signed short int": 2, "int16": 2, "int16_t": 2,
    "ushort": 2, "unsigned short": 2, "unsigned short int": 2,
    "uint16": 2, "uint16_t": 2,
    "int": 4, "signed int": 4, "int32": 4, "int32_t": 4,
    "uint": 4, "unsigned int": 4, "uint32": 4, "uint32_t": 4,
    "longlong": 8, "long long": 8, "long long int": 8,
    "signed long long": 8, "signed long long int": 8,
    "int64": 8, "int64_t": 8,
    "ulonglong": 8, "unsigned long long": 8,
    "unsigned long long int": 8, "uint64": 8, "uint64_t": 8,
    "float": 4, "double": 8,
}
_HEADER_LIMIT = 4 * 1024**2
_BLOCK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class NrrdInfo:
    path: Path
    width: int
    height: int
    frames: int
    sample_bytes: int
    origin_xyz: tuple[int, int, int]
    encoding: str
    payload_offset: int
    fields: dict[str, str]
    custom_keys: frozenset[str]
    file_size: int
    modified_ns: int
    # XYZ reference grid of a cropped segmentation (Segmentation_ReferenceImageExtent).
    reference_xyz: tuple[int, int, int] | None = None


def _integers(value, count, description):
    values = value.split()
    if len(values) != count or any(not re.fullmatch(r"-?\d+", v) for v in values):
        raise ValueError(f"Invalid NRRD {description}: {value!r}")
    return tuple(int(v) for v in values)


def _vectors(value, count, description):
    parts = re.findall(r"\([^()]*\)", value)
    if len(parts) != count or re.sub(r"\([^()]*\)", "", value).strip():
        raise ValueError(f"Invalid NRRD {description}: {value!r}")
    try:
        vectors = np.asarray([[float(v) for v in p[1:-1].split(",")] for p in parts])
    except ValueError as error:
        raise ValueError(f"Invalid NRRD {description}: {value!r}") from error
    if vectors.shape != (count, 3) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid NRRD {description}: {value!r}")
    return vectors


def probe_nrrd(path):
    """Read and validate geometry and storage without decoding the payload."""
    path = Path(path).resolve()
    stat = path.stat()
    fields, custom = {}, set()
    with path.open("rb") as stream:
        if not re.fullmatch(rb"NRRD000[1-5]\r?\n", stream.readline(32)):
            raise ValueError(f"Invalid NRRD signature: {path}")
        while True:
            raw = stream.readline(_HEADER_LIMIT + 1)
            if not raw or stream.tell() > _HEADER_LIMIT:
                raise ValueError(f"Missing or oversized NRRD header: {path}")
            if not raw.strip():
                offset = stream.tell()
                break
            line = raw.decode("utf-8").strip()
            if line.startswith("#"):
                continue
            colon = line.find(":")
            if colon < 0:
                raise ValueError(f"Invalid NRRD header line: {line!r}")
            separator = ":=" if line[colon:colon + 2] == ":=" else ":"
            key, value = line.split(separator, 1)
            key = key.strip() if separator == ":=" else " ".join(key.lower().split())
            if not key or key in fields:
                raise ValueError(f"Duplicate or empty NRRD header field: {key!r}")
            fields[key] = value.strip()
            if separator == ":=":
                custom.add(key)

    standard_keys = {"dimension", "sizes", "type", "encoding", "endian", "space",
                     "space directions", "space origin", "space dimension", "kinds", "spacings"}
    if standard_keys.intersection(custom):
        raise ValueError("Standard NRRD fields require the ': ' separator.")
    if fields.get("dimension") != "3":
        raise ValueError("The cut requires a scalar 3D NRRD.")
    width, height, frames = _integers(fields.get("sizes", ""), 3, "sizes")
    if min(width, height, frames) < 1:
        raise ValueError("NRRD dimensions must be positive.")
    sample_bytes = _TYPE_BYTES.get(fields.get("type", "").lower())
    if sample_bytes is None:
        raise ValueError(f"Unsupported NRRD scalar type: {fields.get('type')!r}")
    encoding = fields.get("encoding", "").lower()
    if encoding not in {"raw", "gzip", "gz"}:
        raise ValueError("The cut supports inline raw or gzip NRRDs.")
    forbidden = {"data file", "datafile", "byte skip", "byteskip", "line skip", "lineskip",
                 "block size", "blocksize", "axis mins", "axis maxs"}
    present = forbidden.intersection(fields)
    if present:
        raise ValueError(f"Unsupported NRRD storage/axis fields: {sorted(present)}")
    if sample_bytes > 1 and fields.get("endian", "").lower() not in {"little", "big"}:
        raise ValueError("Multibyte scalar NRRDs require an explicit little/big endian field.")
    if fields.get("space", "").lower() not in {"lps", "left-posterior-superior"}:
        raise ValueError("Expected the original XTA unit-LPS coordinate space.")
    directions = _vectors(fields.get("space directions", ""), 3, "space directions")
    origin = _vectors(fields.get("space origin", ""), 1, "space origin")
    if not np.allclose(directions, np.eye(3), rtol=0, atol=1e-7):
        raise ValueError("Expected XTA unit spacing and axis order.")
    if np.any(np.abs(origin) > 2**53 - 1) or not np.allclose(origin, np.rint(origin), rtol=0, atol=1e-7):
        raise ValueError("Expected an integer-translated XTA unit-LPS lattice.")
    origin_xyz = tuple(int(v) for v in np.rint(origin[0]))
    if "space dimension" in fields and fields["space dimension"] != "3":
        raise ValueError("Expected NRRD space dimension 3.")
    if "spacings" in fields and fields["spacings"].split() != ["1", "1", "1"]:
        try:
            spacing = np.asarray([float(v) for v in fields["spacings"].split()])
        except ValueError as error:
            raise ValueError("Invalid NRRD spacings.") from error
        if spacing.shape != (3,) or not np.allclose(spacing, 1, rtol=0, atol=1e-7):
            raise ValueError("Expected NRRD spacings of 1 along each axis.")
    if "kinds" in fields and any(v not in {"domain", "space"} for v in fields["kinds"].split()):
        raise ValueError("Expected three scalar spatial NRRD axes.")
    if "kinds" in fields and len(fields["kinds"].split()) != 3:
        raise ValueError("Expected three scalar spatial NRRD axes.")
    if any(re.match(r"Segment[1-9]\d*_", key) for key in fields):
        raise ValueError("Only single-segment XTA NRRDs are supported.")
    if fields.get("Segment0_Layer", "0") != "0":
        raise ValueError("Expected segment layer 0.")
    if "Segment0_Extent" in fields:
        extent = _integers(fields["Segment0_Extent"], 6, "segment extent")
        nonempty = all(extent[i] <= extent[i + 1] for i in (0, 2, 4))
        if nonempty and any(extent[i] < 0 or extent[i + 1] >= size
                            for i, size in zip((0, 2, 4), (width, height, frames))):
            raise ValueError("Segment extent exceeds the stored NRRD raster.")
    reference_offset = (0, 0, 0)
    if "Segmentation_ReferenceImageExtentOffset" in fields:
        reference_offset = _integers(fields["Segmentation_ReferenceImageExtentOffset"], 3, "reference offset")
    reference_xyz = None
    if "Segmentation_ReferenceImageExtent" in fields:
        reference_extent = _integers(fields["Segmentation_ReferenceImageExtent"], 6, "reference extent")
        if reference_extent[0::2] != (0, 0, 0) or min(reference_extent[1::2]) < 0:
            raise ValueError("A segmentation reference extent must start at voxel zero.")
        reference_xyz = tuple(v + 1 for v in reference_extent[1::2])
    if reference_offset != (0, 0, 0) or reference_xyz is not None:
        # Slicer stores a cropped box at reference origin + offset; XTA reference grids start at zero.
        if reference_offset != origin_xyz:
            raise ValueError("A cropped segmentation's reference offset must equal its integer space origin.")
        if reference_xyz is not None and any(
                start < 0 or start + size > bound
                for start, size, bound in zip(reference_offset, (width, height, frames), reference_xyz)):
            raise ValueError("The stored segmentation box exceeds its reference extent.")
    if fields.get("Segmentation_ConversionParameters", "").strip():
        raise ValueError("Segmentation conversion geometry requires reconstruction before cutting.")
    if encoding == "raw" and stat.st_size != offset + width * height * frames * sample_bytes:
        raise ValueError("Raw NRRD payload length differs from its declared dimensions.")
    return NrrdInfo(path, width, height, frames, sample_bytes, origin_xyz, encoding, offset,
                    fields, frozenset(custom), stat.st_size, stat.st_mtime_ns, reference_xyz)


def _reference_cut_column(info, first_removed, reference_first_removed):
    """First removed reference-grid column of a cropped segmentation, else None."""
    if info.reference_xyz is None:
        return None
    if reference_first_removed is None:
        if not 0 < first_removed < info.width:
            raise ValueError("A cut through a cropped segmentation needs its reference-grid column.")
        reference_first_removed = info.origin_xyz[0] + first_removed
    if isinstance(reference_first_removed, bool) or not isinstance(reference_first_removed, int):
        raise ValueError("The first removed reference column must be an integer.")
    if not 0 <= reference_first_removed <= info.reference_xyz[0]:
        raise ValueError("The reference cut column lies outside the reference extent.")
    if max(0, min(info.width, reference_first_removed - info.origin_xyz[0])) != first_removed:
        raise ValueError("The stored and reference cut columns disagree.")
    return reference_first_removed


def _cropped_reference_fields(fields, info, first_removed, reference_cut, mode):
    """Rewrite a cropped segmentation's placement for its cut grid.

    Returns ``(offset_xyz, stored_xyz, reference_xyz, zero_prefix_voxels,
    zero_suffix_voxels, discard_retained)``. Zero mode keeps the box and grid. Crop
    mode narrows the reference grid to ``reference_cut`` columns; a box with no
    retained segment extent becomes the XTA/Slicer empty one-voxel image at offset
    zero, and a nonempty one-voxel box is widened by one zero neighbour like the XTA
    writer, because Slicer aborts on nonempty 1x1x1 segmentations."""
    offset = list(info.origin_xyz)
    reference = list(info.reference_xyz)
    stored = [first_removed if mode == "crop" else info.width, info.height, info.frames]
    prefix = suffix = 0
    discard = False
    if mode == "crop":
        reference[0] = reference_cut
        extent = (list(_integers(fields["Segment0_Extent"], 6, "segment extent"))
                  if "Segment0_Extent" in fields else None)
        if first_removed == 0 or (extent is not None and any(extent[i] > extent[i + 1] for i in (0, 2, 4))):
            offset, stored = [0, 0, 0], [1, 1, 1]
            suffix, discard = 1, True
            if extent is not None:
                fields["Segment0_Extent"] = "0 -1 0 -1 0 -1"
        elif stored == [1, 1, 1] and any(size > 1 for size in reference):
            # Widen along the first axis with room, toward higher indices when possible.
            axis = next(a for a in range(3) if reference[a] > 1)
            stored[axis] = 2
            if offset[axis] + 1 < reference[axis]:
                suffix = 1
            else:
                offset[axis] -= 1
                prefix = 1
                if extent is not None:
                    extent[2 * axis] += 1
                    extent[2 * axis + 1] += 1
                    fields["Segment0_Extent"] = " ".join(str(v) for v in extent)
        fields["sizes"] = " ".join(str(v) for v in stored)
        fields["space origin"] = "(" + ",".join(str(v) for v in offset) + ")"
        fields["Segmentation_ReferenceImageExtentOffset"] = " ".join(str(v) for v in offset)
        fields["Segmentation_ReferenceImageExtent"] = " ".join(
            str(v) for size in reference for v in (0, size - 1))
    return tuple(offset), tuple(stored), tuple(reference), prefix, suffix, discard


def _read_exact(stream, size):
    data = bytearray(size)
    view = memoryview(data)
    position = 0
    while position < size:
        count = stream.readinto(view[position:])
        if not count:
            raise ValueError("Truncated NRRD payload.")
        position += count
    return data


def write_nrrd(source, destination, info, first_removed, mode, *, reference_first_removed=None):
    """Write a new gzip NRRD and return the actual output payload audit.

    ``first_removed`` is the first discarded column, in the input NRRD grid.
    ``reference_first_removed`` is the same plane in a cropped segmentation's
    reference grid; it is derived when the plane crosses the stored box.
    Output creation is exclusive. A partial output is removed after failure.
    """
    if mode not in {"zero", "crop"}:
        raise ValueError("NRRD cut mode must be 'zero' or 'crop'.")
    if isinstance(first_removed, bool) or not isinstance(first_removed, int):
        raise ValueError("The first removed NRRD column must be an integer.")
    # A cropped segmentation wholly beyond the plane still has a nonempty cut grid.
    if not 0 <= first_removed <= info.width or (
            mode == "crop" and first_removed == 0 and info.reference_xyz is None):
        raise ValueError("The NRRD cut must retain a valid output width.")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    current = probe_nrrd(source)
    if current != info:
        raise ValueError("The input NRRD changed after its cut was planned.")
    reference_cut = _reference_cut_column(info, first_removed, reference_first_removed)
    if mode == "crop" and reference_cut == 0:
        raise ValueError("The NRRD cut must retain a valid output width.")
    output_width = first_removed if mode == "crop" else info.width
    fields = dict(info.fields)
    fields["sizes"] = f"{output_width} {info.height} {info.frames}"
    fields["encoding"] = "gzip"
    for key in ("min", "max", "old min", "old max"):
        fields.pop(key, None)
    if "Segment0_Extent" in fields:
        extent = list(_integers(fields["Segment0_Extent"], 6, "segment extent"))
        extent[1] = min(extent[1], first_removed - 1)
        if any(extent[i] > extent[i + 1] for i in (0, 2, 4)):
            extent = [0, -1, 0, -1, 0, -1]
        fields["Segment0_Extent"] = " ".join(str(v) for v in extent)
    custom = set(info.custom_keys)
    prefix_voxels = suffix_voxels = 0
    discard_retained = False
    if reference_cut is not None:
        placement = _cropped_reference_fields(fields, info, first_removed, reference_cut, mode)
        offset_xyz, stored_xyz, reference_xyz, prefix_voxels, suffix_voxels, discard_retained = placement
        if mode == "crop":
            custom.update(("Segmentation_ReferenceImageExtentOffset", "Segmentation_ReferenceImageExtent"))
    if "Segment0_Extent" in fields:
        custom.add("Segment0_Extent")
    # Existing content descriptions can embed the old array dimensions.
    if "content" in fields:
        column = first_removed if reference_cut is None else reference_cut
        fields["content"] = f"XTA right-side {mode}; first_removed_column={column}"
    header = "NRRD0005\n" + "\n".join(
        f"{key}{':=' if key in custom else ': '}{value}" for key, value in fields.items()
    ) + "\n\n"
    row_bytes = info.width * info.sample_bytes
    rows_per_block = max(1, _BLOCK_BYTES // row_bytes)
    digest = hashlib.sha256()
    output_bytes = 0
    created = False
    try:
        with ExitStack() as stack:
            raw_input = stack.enter_context(source.open("rb"))
            raw_input.seek(info.payload_offset)
            payload = raw_input
            if info.encoding in {"gzip", "gz"}:
                payload = stack.enter_context(gzip.GzipFile(fileobj=raw_input, mode="rb"))
            raw_output = stack.enter_context(destination.open("xb"))
            created = True
            raw_output.write(header.encode("utf-8"))
            output = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw_output,
                                                       mode="wb", compresslevel=1, mtime=0))
            if prefix_voxels:
                written = bytes(prefix_voxels * info.sample_bytes)
                output.write(written)
                digest.update(written)
                output_bytes += len(written)
            remaining_rows = info.height * info.frames
            while remaining_rows:
                rows = min(rows_per_block, remaining_rows)
                block = _read_exact(payload, rows * row_bytes)
                array = np.frombuffer(block, dtype=np.uint8).reshape(rows, info.width, info.sample_bytes)
                if mode == "zero":
                    array[:, first_removed:, :] = 0
                    written = memoryview(block)
                elif discard_retained:
                    # The empty retained extent is checked, then stored as one zero voxel.
                    if array[:, :first_removed, :].any():
                        raise ValueError("Cropped segmentation foreground lies outside its Segment0_Extent.")
                    written = b""
                else:
                    written = array[:, :first_removed, :].tobytes()
                output.write(written)
                digest.update(written)
                output_bytes += len(written)
                remaining_rows -= rows
            if payload.read(1):
                raise ValueError("Extra bytes after the declared NRRD payload.")
            if suffix_voxels:
                written = bytes(suffix_voxels * info.sample_bytes)
                output.write(written)
                digest.update(written)
                output_bytes += len(written)
        final_stat = source.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (info.file_size, info.modified_ns):
            raise ValueError("The input NRRD changed while being cut.")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    audit = {"output_shape_tyx": [info.frames, info.height, output_width],
             "payload_sha256": digest.hexdigest(), "payload_bytes": output_bytes,
             "encoding": "gzip", "sample_bytes": info.sample_bytes}
    if reference_cut is not None:
        audit.update(output_shape_tyx=list(reversed(stored_xyz)),
                     stored_offset_tyx=list(reversed(offset_xyz)),
                     reference_shape_tyx=list(reversed(reference_xyz)),
                     first_removed_reference_i=reference_cut)
    return audit
