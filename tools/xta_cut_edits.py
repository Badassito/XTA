"""Crop exported Slicer NPZ masks on their original XTA preview lattice."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil

import numpy as np

if __package__:
    from . import prepare_pta_input as prep
else:
    import prepare_pta_input as prep


FORMAT = "xta-slicer-edits-axial-crop-v1"
EMPTY_EXTENT = np.asarray([0, -1, 0, -1, 0, -1], dtype=np.int64)


@dataclass(frozen=True)
class EditsInfo:
    bundle_dir: Path
    preview_xyz: tuple[int, int, int]
    native_xyz: tuple[int, int, int]
    stem: str
    edits_sha256: str
    manifest_sha256: str
    edits: dict
    manifest: dict


def inspect_edits(bundle_dir):
    """Validate source metadata and snapshot identities without loading masks."""
    given_root = Path(bundle_dir)
    root = given_root.resolve()
    if given_root.is_symlink() or not root.is_dir():
        raise ValueError("The edit bundle must be a regular directory.")
    edits_path = prep.child(root, "edits.json")
    manifest_path = prep.child(root, "source_manifest.json")
    edits_bytes, manifest_bytes = edits_path.read_bytes(), manifest_path.read_bytes()
    edits, manifest = json.loads(edits_bytes), json.loads(manifest_bytes)
    if edits.get("format") != "xta-slicer-edits-v1":
        raise ValueError("Select the original Slicer export directory containing edits.json.")
    if manifest.get("exported_axes", "(X, Y, t)") != "(X, Y, t)":
        raise ValueError("The original manifest must use (X, Y, t) axis order.")
    # Source segment names can include a channel suffix absent from the MKV.
    names = [row.get("segment_name", "") for row in manifest.get("layers", [])
             if row.get("suffix") == "Global_final_output"]
    basename = edits.get("source_manifest_name", "")
    suffix = "_nrrd_manifest.json"
    if len(names) == 1 and names[0].endswith("_Global_final_output"):
        stem = names[0].removesuffix("_Global_final_output")
    elif basename.endswith(suffix):
        stem = basename[:-len(suffix)]
    else:
        raise ValueError("Cannot determine the volume stem from the original manifest or Global.")
    low, full, _, identities = prep.load_bundle(root, stem, allow_empty_global=True)
    edits_hash = hashlib.sha256(edits_bytes).hexdigest()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if edits_hash != identities["edits_sha256"] or manifest_hash != identities["source_manifest_sha256"]:
        raise ValueError("The export metadata changed while it was being read.")
    for row in edits["segments"]:
        source_name = row["source_filename"]
        if not isinstance(source_name, str) or Path(source_name).name != source_name or "/" in source_name or "\\" in source_name:
            raise ValueError("Original source filenames must be plain filenames.")
        if not row["file"].endswith(".npz"):
            raise ValueError("Exported snapshots must use .npz filenames.")
        if not isinstance(row.get("source_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["source_sha256"]):
            raise ValueError(f"Invalid source SHA256 for {row['name']}")
        if type(row.get("foreground_voxels")) is not int or row["foreground_voxels"] < 0:
            raise ValueError(f"Invalid foreground count for {row['name']}")
    return EditsInfo(root, tuple(reversed(low)), tuple(reversed(full)), stem,
                     edits_hash, manifest_hash, edits, manifest)


def _snapshot(path, entry, preview_xyz):
    return prep.read_snapshot(path, entry, tuple(reversed(preview_xyz)))


def _write_global(path, mask, extent, preview_xyz):
    width, height, frames = preview_xyz
    header = ("NRRD0005\ntype: unsigned char\ndimension: 3\nspace: left-posterior-superior\n"
              f"sizes: {width} {height} {frames}\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n"
              "kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n\n")
    digest, count = hashlib.sha256(), 0
    with Path(path).open("xb") as raw:
        raw.write(header.encode("ascii"))
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
            plane = np.zeros((height, width), dtype=np.uint8)
            for z in range(frames):
                plane.fill(0)
                if mask.size and extent[4] <= z <= extent[5]:
                    plane[extent[2]:extent[3] + 1, extent[0]:extent[1] + 1] = mask[z - extent[4]]
                data = memoryview(plane)
                stream.write(data)
                digest.update(data)
                count += int(np.count_nonzero(plane))
    return {"shape_tyx": [frames, height, width], "foreground_voxels": count,
            "payload_sha256": digest.hexdigest(), "sha256": prep.digest(path)}


def write_edits(bundle_dir, dest_edits_dir, dest_preview_nrrd, info,
                first_preview, first_native, marker_ras):
    """Write cropped snapshots and a full-field cropped preview Global NRRD."""
    current = inspect_edits(bundle_dir)
    if (current.bundle_dir, current.edits_sha256, current.manifest_sha256) != (info.bundle_dir, info.edits_sha256, info.manifest_sha256):
        raise ValueError("The original edit bundle changed after inspection.")
    if type(first_preview) is not int or not 0 < first_preview <= info.preview_xyz[0]:
        raise ValueError("The preview crop must retain a valid width on the original grid.")
    if type(first_native) is not int or not 0 < first_native < info.native_xyz[0]:
        raise ValueError("The native crop must retain and remove at least one column.")
    marker = tuple(Fraction(value) for value in marker_ras)
    if len(marker) != 3:
        raise ValueError("The marker must contain three RAS coordinates.")
    if prep.preview_cut_columns(marker, info.preview_xyz, info.native_xyz) != (first_preview, first_native):
        raise ValueError("The crop widths differ from the marker and original sampling dimensions.")
    destination = Path(dest_edits_dir)
    if destination.exists() or Path(dest_preview_nrrd).exists():
        raise FileExistsError("Cropped bundle and preview destinations must be new.")
    destination.mkdir()
    cropped_xyz = (first_preview, *info.preview_xyz[1:])
    result = dict(info.edits)
    result.update(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
                  preview_shape_tyx=list(reversed(cropped_xyz)),
                  full_quality_shape_tyx=[info.native_xyz[2], info.native_xyz[1], first_native], segments=[])
    result["axial_crop"] = {
        "original_preview_shape_tyx": list(reversed(info.preview_xyz)),
        "original_full_quality_shape_tyx": list(reversed(info.native_xyz)),
        "first_removed_preview_i": first_preview, "first_removed_native_i": first_native,
        "marker_preview_ras": [float(v) for v in marker],
        "marker_preview_ras_exact": [str(value) for value in marker],
        "source_edits_sha256": info.edits_sha256,
    }
    by_file = {row["filename"]: row for row in info.manifest["layers"]}
    original_counts, retained_counts, global_audit = 0, 0, None
    for entry in info.edits["segments"]:
        path = prep.child(info.bundle_dir, entry["file"])
        mask, extent = _snapshot(path, entry, info.preview_xyz)
        if mask.size:
            keep = max(0, min(mask.shape[2], first_preview - int(extent[0])))
            mask = mask[:, :, :keep]
            extent[1] = int(extent[0]) + keep - 1
        count = int(np.count_nonzero(mask))
        if not count:
            mask, extent = np.empty((0, 0, 0), dtype=np.uint8), EMPTY_EXTENT.copy()
        output_path = destination / entry["file"]
        with output_path.open("xb") as stream:
            np.savez_compressed(stream, mask=mask, extent_xyz=extent, ijk_to_ras=prep.REFERENCE_RAS)
        cropped_entry = dict(entry)
        cropped_entry.update(sha256=prep.digest(output_path), foreground_voxels=count,
                             axial_crop_source_snapshot_sha256=entry["sha256"])
        result["segments"].append(cropped_entry)
        original_counts += entry["foreground_voxels"]
        retained_counts += count
        if by_file[entry["source_filename"]].get("suffix") == "Global_final_output":
            global_audit = _write_global(dest_preview_nrrd, mask, extent, cropped_xyz)
        del mask
        if prep.digest(path) != entry["sha256"]:
            raise ValueError(f"Source snapshot changed during cropping: {path}")
    shutil.copyfile(prep.child(info.bundle_dir, "source_manifest.json"), destination / "source_manifest.json")
    result["notes"] = list(info.edits.get("notes", [])) + [
        "Derived axial crop on the original preview lattice; original source manifest is retained byte-for-byte.",
        "Native reconstruction must use axial_crop's original sampling dimensions before applying the native crop.",
        "Source NRRD hashes are retained source claims; this operation verifies the exported NPZ payloads and source-manifest identities.",
    ]
    (destination / "edits.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if prep.digest(prep.child(info.bundle_dir, "edits.json")) != info.edits_sha256 or prep.digest(destination / "source_manifest.json") != info.manifest_sha256:
        raise ValueError("Source bundle metadata changed while cropping.")
    for entry in info.edits["segments"]:
        if prep.digest(prep.child(info.bundle_dir, entry["file"])) != entry["sha256"]:
            raise ValueError(f"Source snapshot changed during cropping: {entry['file']}")
    return {"format": FORMAT, "segments": len(result["segments"]),
            "original_segment_foreground_voxels_sum": original_counts,
            "cropped_segment_foreground_voxels_sum": retained_counts,
            "edits_sha256": prep.digest(destination / "edits.json"),
            "preview_global": global_audit, "volume_stem": info.stem}
