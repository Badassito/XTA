"""Portable, data-only mappings between two reconstructed voxel grids.

The affine maps use homogeneous XYZ column vectors whose coordinates are
zero-based voxel centers.  Array shapes are stored in ZYX order.  A packet can
be a ZIP archive or its extracted ``mapping.json`` and relative mask files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


SCHEMA = "xta.registration.packet"
SCHEMA_VERSION = 1
MAX_JSON_BYTES = 1_048_576
MAX_ARCHIVE_BYTES = 134_217_728
ROLES = ("fixed", "moving")


def _relative_member(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Packet file references must use relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("Packet file references cannot leave the packet directory")
    if path.as_posix() != value:
        raise ValueError("Packet file references must use a normalized relative path")
    return value


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ValueError(f"{field} must contain a SHA256 digest")
    return value.lower()


def _matrix(value: Any, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite affine 4 by 4 matrix") from exc
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must be a finite affine 4 by 4 matrix")
    if not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError(f"{field} must have homogeneous last row [0, 0, 0, 1]")
    if abs(np.linalg.det(result[:3, :3])) < 1e-12:
        raise ValueError(f"{field} must be invertible")
    return result


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject unsupported schemas, ambiguous geometry and inconsistent maps."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Mapping metadata must be a JSON object")
    if metadata.get("schema") != SCHEMA or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported XTA registration packet schema")
    if not isinstance(metadata.get("pair"), str) or not metadata["pair"]:
        raise ValueError("Packet pair identity is required")
    if metadata.get("status") not in ("validated", "unavailable"):
        raise ValueError("Packet status must be validated or unavailable")
    for role in ROLES:
        scan = metadata.get(role)
        if not isinstance(scan, Mapping):
            raise ValueError(f"Packet {role} scan identity is required")
        name = scan.get("filename")
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or ":" in name
            or name in ("", ".", "..")
        ):
            raise ValueError(f"{role}.filename must be an acquisition basename")
        _digest(scan.get("sha256"), f"{role}.sha256")
        shape = scan.get("shape_zyx")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape)
        ):
            raise ValueError(f"{role}.shape_zyx needs three positive integer dimensions")
        try:
            spacing = np.asarray(scan.get("spacing_xyz"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{role}.spacing_xyz needs three finite positive values") from exc
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError(f"{role}.spacing_xyz needs three finite positive values")
        if not isinstance(scan.get("spacing_units"), str) or not scan["spacing_units"]:
            raise ValueError(f"{role}.spacing_units must state the unit or its unknown status")
        member = scan.get("observed_mask")
        if member is not None:
            _relative_member(member)
            _digest(scan.get("observed_mask_sha256"), f"{role}.observed_mask_sha256")
    mapping = metadata.get("mapping")
    if metadata["status"] == "unavailable":
        if mapping is not None:
            raise ValueError("An unavailable mapping packet cannot contain a transfer matrix")
        if metadata.get("registered_video") is not None:
            raise ValueError("An unavailable mapping packet cannot identify a registered video")
        return
    if not isinstance(mapping, Mapping):
        raise ValueError("A validated packet requires mapping matrices")
    forward = _matrix(mapping.get("fixed_to_moving_xyz"), "fixed_to_moving_xyz")
    inverse = _matrix(mapping.get("moving_to_fixed_xyz"), "moving_to_fixed_xyz")
    if not np.allclose(forward @ inverse, np.eye(4), rtol=1e-8, atol=1e-8):
        raise ValueError("Forward and reverse registration matrices disagree")
    if "physical_fixed_to_moving_xyz" in mapping:
        fixed_spacing = np.diag([*metadata["fixed"]["spacing_xyz"], 1.0])
        moving_spacing = np.diag([*metadata["moving"]["spacing_xyz"], 1.0])
        expected = moving_spacing @ forward @ np.linalg.inv(fixed_spacing)
        physical = _matrix(mapping["physical_fixed_to_moving_xyz"], "physical_fixed_to_moving_xyz")
        if not np.allclose(physical, expected, rtol=1e-8, atol=1e-7):
            raise ValueError("Physical-coordinate and voxel-coordinate matrices disagree")
    video = metadata.get("registered_video")
    if video is not None:
        if not isinstance(video, Mapping):
            raise ValueError("registered_video must be a JSON object")
        name = video.get("output_filename")
        if (
            not isinstance(name, str)
            or "/" in name
            or "\\" in name
            or ":" in name
            or name in ("", ".", "..")
        ):
            raise ValueError("registered_video.output_filename must be a video basename")
        _digest(video.get("output_sha256"), "registered_video.output_sha256")
        source, destination = video.get("source_role"), video.get("destination_role")
        if source not in ROLES or destination not in ROLES or source == destination:
            raise ValueError(
                "Registered video source and destination must name different scan roles"
            )
        if video.get("shape_zyx") != metadata[destination]["shape_zyx"]:
            raise ValueError("Registered video shape must match the destination voxel grid")
        if (
            _digest(video.get("source_sha256"), "registered_video.source_sha256")
            != metadata[source]["sha256"].lower()
        ):
            raise ValueError("Registered video source identity disagrees with its acquisition")
        video_matrix = _matrix(
            video.get("output_to_input_xyz"), "registered_video.output_to_input_xyz"
        )
        expected = forward if destination == "fixed" else inverse
        if not np.allclose(video_matrix, expected, rtol=1e-8, atol=1e-8):
            raise ValueError("Registered video sampling matrix disagrees with the mapping")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key}")
        result[key] = value
    return result


def _read_metadata(payload: bytes) -> dict:
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("Mapping JSON exceeds the supported size")
    result = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    validate_metadata(result)
    return result


def _read_mask(payload: bytes, scan: Mapping[str, Any]) -> np.ndarray:
    if hashlib.sha256(payload).hexdigest() != scan["observed_mask_sha256"].lower():
        raise ValueError("Observed-field mask checksum does not match its packet")
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Observed-field masks must use NPY format version 1 or 2")
    if dtype != np.bool_ or shape != tuple(scan["shape_zyx"][1:]):
        raise ValueError("Observed-field mask must be boolean YX on its native scan grid")
    pixels = int(shape[0]) * int(shape[1])
    if pixels > MAX_ARCHIVE_BYTES or len(payload) - stream.tell() != pixels:
        raise ValueError("Observed-field mask size disagrees with its NPY header")
    stream.seek(0)
    mask = np.load(stream, allow_pickle=False)
    mask.flags.writeable = False
    return mask


@dataclass(frozen=True)
class RegistrationPacket:
    """Validated mapping metadata and optional native boolean acquisition masks."""

    metadata: dict[str, Any]
    masks: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        # Keep caller-owned dictionaries and arrays from changing a validated
        # packet after construction. Accessors validate again because these
        # public data dictionaries can still be deliberately edited by callers.
        self.validate()
        document = json.loads(json.dumps(self.metadata, allow_nan=False))
        fields = {}
        for role, mask in self.masks.items():
            owned = np.array(mask, copy=True)
            owned.flags.writeable = False
            fields[role] = owned
        object.__setattr__(self, "metadata", document)
        object.__setattr__(self, "masks", fields)

    def validate(self) -> None:
        """Check direct objects as well as objects loaded from an archive."""
        validate_metadata(self.metadata)
        if not isinstance(self.masks, Mapping):
            raise ValueError("Packet masks must map scan roles to boolean YX arrays")
        for role, mask in self.masks.items():
            if role not in ROLES:
                raise ValueError("Mask keys must be fixed or moving")
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != np.bool_
                or mask.shape != tuple(self.metadata[role]["shape_zyx"][1:])
            ):
                raise ValueError("Observed-field mask must be boolean YX on its native scan grid")
        for role in ROLES:
            if self.metadata[role].get("observed_mask") is not None and role not in self.masks:
                raise ValueError(f"Packet lacks its {role} observed-field mask")

    @property
    def usable(self) -> bool:
        self.validate()
        return self.metadata["status"] == "validated"

    @property
    def fixed_to_moving_xyz(self) -> np.ndarray:
        return self.matrix("fixed_to_moving")

    @property
    def moving_to_fixed_xyz(self) -> np.ndarray:
        return self.matrix("moving_to_fixed")

    def matrix(self, direction: str = "fixed_to_moving") -> np.ndarray:
        self.validate()
        if self.metadata["status"] != "validated":
            raise ValueError("This packet has no validated transfer mapping")
        key = direction.replace("-", "_")
        if key not in ("fixed_to_moving", "moving_to_fixed"):
            raise ValueError("Direction must be fixed-to-moving or moving-to-fixed")
        return np.asarray(self.metadata["mapping"][key + "_xyz"], dtype=np.float64)

    def shape(self, role: str) -> tuple[int, int, int]:
        return tuple(self._scan(role)["shape_zyx"])

    def spacing(self, role: str) -> tuple[float, float, float]:
        return tuple(float(n) for n in self._scan(role)["spacing_xyz"])

    def mask(self, role: str) -> np.ndarray | None:
        self._scan(role)
        return self.masks.get(role)

    def _scan(self, role: str) -> dict:
        self.validate()
        if role not in ROLES:
            raise ValueError("Scan role must be fixed or moving")
        return self.metadata[role]

    def transform_points(self, points, direction: str = "fixed_to_moving") -> np.ndarray:
        positions = np.asarray(points, dtype=np.float64)
        if positions.ndim == 0 or positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("Points must be finite XYZ coordinates with final dimension 3")
        matrix = self.matrix(direction)
        return positions @ matrix[:3, :3].T + matrix[:3, 3]


def load_packet(path: str | Path) -> RegistrationPacket:
    """Load a standalone ZIP or extracted mapping JSON without extracting files."""
    path = Path(path)
    if path.is_dir():
        path = path / "mapping.json"
    masks = {}
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) > 32 or len(names) != len(set(names)):
                raise ValueError("Packet archive contains duplicate or excessive members")
            for name in names:
                _relative_member(name)
            if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                raise ValueError("Packet archive exceeds the supported uncompressed size")
            if (
                "mapping.json" not in names
                or archive.getinfo("mapping.json").file_size > MAX_JSON_BYTES
            ):
                raise ValueError("Packet archive needs a bounded mapping.json")
            metadata = _read_metadata(archive.read("mapping.json"))
            for role in ROLES:
                member = metadata[role].get("observed_mask")
                if member is not None:
                    if member not in names:
                        raise ValueError(f"Packet lacks its {role} observed-field mask")
                    masks[role] = _read_mask(archive.read(member), metadata[role])
    else:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Mapping JSON exceeds the supported size")
        metadata = _read_metadata(path.read_bytes())
        base = path.parent.resolve()
        for role in ROLES:
            member = metadata[role].get("observed_mask")
            if member is not None:
                target = (base / member).resolve()
                if base not in target.parents or target.stat().st_size > MAX_ARCHIVE_BYTES:
                    raise ValueError(
                        "Observed-field mask leaves its packet directory or is too large"
                    )
                masks[role] = _read_mask(target.read_bytes(), metadata[role])
    return RegistrationPacket(metadata, masks)


def write_packet(
    path: str | Path,
    metadata: Mapping[str, Any],
    masks: Mapping[str, np.ndarray] | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Publish a verified ZIP atomically; ``overwrite=False`` reserves a new destination.

    The default replaces an existing destination after verification. No-replace
    publication uses Windows no-replace rename or a POSIX hard link from the
    same-directory temporary file. A concurrent destination is preserved.
    """
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Registration packets must use a .zip filename")
    # JSON serialization gives the writer a private copy and rejects foreign objects.
    document = json.loads(json.dumps(metadata, allow_nan=False))
    encoded_masks = {}
    for role, mask in (masks or {}).items():
        if role not in ROLES:
            raise ValueError("Mask keys must be fixed or moving")
        values = np.asarray(mask)
        if values.dtype != np.bool_ or values.shape != tuple(document[role]["shape_zyx"][1:]):
            raise ValueError("Observed-field mask must be boolean YX on its native scan grid")
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        payload = stream.getvalue()
        member = f"masks/{role}.npy"
        document[role]["observed_mask"] = member
        document[role]["observed_mask_sha256"] = hashlib.sha256(payload).hexdigest()
        encoded_masks[member] = payload
    validate_metadata(document)
    for role in ROLES:
        member = document[role].get("observed_mask")
        if member is not None and member not in encoded_masks:
            raise ValueError(f"A referenced {role} mask must be supplied to write_packet")
    encoded_json = json.dumps(document, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if (
        len(encoded_json) > MAX_JSON_BYTES
        or sum(map(len, encoded_masks.values())) + len(encoded_json) > MAX_ARCHIVE_BYTES
    ):
        raise ValueError("Packet exceeds the supported size")
    readme = (
        f"{document['pair']} registration mapping\n\n"
        f"Status is {document['status']}.\n"
        "mapping.json records both acquisition filenames, SHA256 identities, grid dimensions and voxel sizes.\n"
        "Coordinates are zero-based XYZ voxel-center column vectors, with x=column, y=row and z=frame.\n"
        "Array shapes use ZYX order. Video FPS is playback metadata.\n"
        "To reslice the moving image onto the fixed grid, evaluate the moving image at fixed_to_moving_xyz times each fixed coordinate.\n"
        "A fixed-grid label point maps into the original moving grid using the same forward matrix. Use the inverse for moving-to-fixed label points.\n"
        "Use linear interpolation for grayscale intensities and nearest-neighbor interpolation for discrete labels.\n"
        "masks/*.npy, when present, are boolean YX acquisition fields at native voxel centers and apply at every Z plane. False means unobserved.\n"
        "Physical coordinates use each array's voxel-zero center as local origin; acquisition world origins were unavailable.\n"
    )
    if document.get("registered_video") is not None:
        readme += "registered_video identifies the existing full movie, its pixel grid and the output-to-input sampling matrix.\n"
    if document["status"] == "unavailable":
        readme += "\nThis pair has no validated transfer matrix. See validation in mapping.json.\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".partial.zip", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("mapping.json", encoded_json)
            archive.writestr("README.txt", readme)
            for member, payload in encoded_masks.items():
                archive.writestr(member, payload)
        load_packet(temporary)
        if overwrite:
            os.replace(temporary, path)
        elif os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
