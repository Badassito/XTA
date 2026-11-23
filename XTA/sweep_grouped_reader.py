"""Read authenticated channels of the native and preview sweep NRRDs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .file_hashing import sha256_file
from .reconciliation_io import ReferenceGeometry, _ints, _read_header, _same_geometry, _shape, _vectors
from .run_transport import _stamp


GROUPED_CATALOG_LAYOUT = "grouped_sweep_segments"


def group_stamp(info):
    return _stamp(info)


def _relative_file(base, relative, boundary):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Sweep paths must be relative to their manifest")
    path = (base / relative).resolve()
    if not path.is_relative_to(boundary):
        raise ValueError(f"Sweep path escaped its study: {relative}")
    return path


def _one(values, description):
    values = list(values)
    if len(values) != 1:
        raise ValueError(f"Expected one {description} in the sweep, found {len(values)}")
    return values[0]


def _family(study):
    path = study / "manifest.json"
    if path.is_file():
        record = json.loads(path.read_bytes())
        if record.get("schema") == "xta.confidence_sweep/1":
            return record
    return {}


def _review(path, family=None):
    data = path.read_bytes()
    record = json.loads(data)
    if record.get("schema") != "xta.sweep_review/1" or record.get("status") != "complete":
        raise ValueError("Expected a published XTA sweep review manifest")
    study = path.parent.parent
    if family is None:
        family = _family(study)
    digest = family.get("review_manifest_sha256")
    if digest is not None and (_relative_file(study, family.get("review_manifest"), study) != path
                               or hashlib.sha256(data).hexdigest() != digest):
        raise ValueError("Sweep review manifest differs from the family receipt")
    return record


def run_layer_catalogs(run_manifest):
    """Find the native and low-quality catalogues belonging to one run."""
    run_manifest = Path(run_manifest).resolve()
    run_data = run_manifest.read_bytes() if run_manifest.is_file() else b"{}"
    run = json.loads(run_data)
    branch = run.get("confidence_sweep") or {}
    if branch.get("nrrd_storage") == "grouped":
        try:
            cent = branch["cutoff_hundredths"]
            study = run_manifest.parents[2]
        except (IndexError, KeyError) as exc:
            raise ValueError("Invalid grouped cutoff manifest") from exc
        if type(cent) is not int or not 0 <= cent <= 100:
            raise ValueError("Invalid grouped cutoff value")
        expected_run = study / "confidence_cutoffs" / f"conf_{cent:03d}" / "manifest.json"
        if run_manifest != expected_run:
            raise ValueError("Grouped cutoff manifest is outside its sweep layout")
        family = _family(study)
        review_path = _relative_file(study, family.get("review_manifest") or "review/manifest.json", study)
        review = _review(review_path, family)
        if branch.get("sweep_id") != review.get("sweep_id"):
            raise ValueError("Grouped review belongs to a different sweep")
        run_sha256 = hashlib.sha256(run_data).hexdigest()
        result = {}
        for grid in review["grids"]:
            grid_id = grid["grid_id"]
            if grid_id in result:
                raise ValueError(f"Duplicate grouped review grid: {grid_id}")
            metadata = _one((item for item in grid["branches"] if item["cutoff_cent"] == cent),
                            f"{grid_id} catalogue for cutoff {cent}")
            linked_run = _relative_file(review_path.parent, metadata["run_manifest"], study)
            if (grid.get("payload") != "grouped" or metadata.get("source_storage") != "grouped"
                    or linked_run != run_manifest or metadata.get("run_manifest_sha256") != run_sha256
                    or metadata.get("run_id") != run.get("run_id")):
                raise ValueError("Review catalogue belongs to a different cutoff run")
            path = _relative_file(review_path.parent, metadata["source_manifest"], review_path.parent)
            if sha256_file(path) != metadata["source_manifest_sha256"]:
                raise ValueError(f"Grouped catalogue checksum differs from the review: {path}")
            result[grid_id] = path
        return result
    root = run_manifest.parent
    result = {}
    paths = list((root / "nrrd").glob("*_nrrd_manifest.json"))
    paths += list((root / "low_quality").glob("*/nrrd/*_nrrd_manifest.json"))
    for path in sorted(paths):
        grid_id = "native" if path.parent.parent == root else path.parent.parent.relative_to(root).as_posix()
        if grid_id in result:
            raise ValueError(f"Multiple scalar catalogues for the {grid_id} grid")
        result[grid_id] = path
    return result


def locate_run_layer_catalog(run_manifest, grid_id="native"):
    """Find a specific cutoff layer catalogue by its grid identifier."""
    try:
        return run_layer_catalogs(run_manifest)[grid_id]
    except KeyError as exc:
        raise ValueError(f"Run has no NRRD catalogue for grid {grid_id}") from exc


@dataclass(frozen=True)
class GroupedSegmentReference:
    path: Path
    sha256: str
    segment_id: str
    layer: int
    label_value: int
    gzip_offset: int
    gzip_bytes: int
    stored_shape_tyx: tuple[int, int, int]
    offset_xyz: tuple[int, int, int]
    size_bytes: int
    header: dict[str, str]
    payload_offset: int
    stamp: tuple[int, ...]
    empty: bool


class GroupedCatalogue:
    def __init__(self, path, catalogue, data_sha256):
        self.path = path
        # The review is an ancestor of its per-grid source catalogues.
        self.review_path = _relative_file(path.parent, catalogue["review_manifest"], path.parents[-1])
        root = self.review_path.parent
        if self.review_path.name != "manifest.json" or root == path.parent or not path.is_relative_to(root):
            raise ValueError("Grouped catalogue does not identify its enclosing review")
        review = _review(self.review_path)
        self.grid_id = catalogue["grid_id"]
        self.cutoff_cent = catalogue["cutoff_cent"]
        if type(self.cutoff_cent) is not int or not 0 <= self.cutoff_cent <= 100:
            raise ValueError("Invalid grouped cutoff value")
        grid = _one((item for item in review["grids"] if item["grid_id"] == self.grid_id),
                    f"{self.grid_id} grid")
        branch = _one((item for item in grid["branches"] if item["cutoff_cent"] == self.cutoff_cent),
                      f"{self.grid_id} catalogue for cutoff {self.cutoff_cent}")
        if (grid["payload"] != "grouped" or branch.get("source_storage") != "grouped"
                or _relative_file(root, branch["source_manifest"], root) != path
                or branch["source_manifest_sha256"] != data_sha256
                or branch["run_id"] != catalogue.get("run_id")):
            raise ValueError("Grouped layer catalogue identity does not match the review")
        self.run_manifest = _relative_file(root, branch["run_manifest"], root.parent)
        self.run_manifest_sha256 = branch["run_manifest_sha256"]
        run_data = self.run_manifest.read_bytes()
        run = json.loads(run_data)
        identity = run.get("confidence_sweep") or {}
        if (hashlib.sha256(run_data).hexdigest() != self.run_manifest_sha256
                or run.get("run_id") != branch["run_id"]
                or identity.get("sweep_id") != review["sweep_id"]
                or identity.get("cutoff_hundredths") != branch["cutoff_cent"]):
            raise ValueError("Grouped catalogue and cutoff run manifest disagree")
        self.geometry = ReferenceGeometry(_shape(grid["shape_tyx"], "review shape"), grid["space"],
                                          tuple(map(tuple, grid["directions_xyz"])), tuple(grid["origin_xyz"]))
        if self.geometry.shape_tyx != _shape(catalogue["output_shape_tyx"], "catalogue shape"):
            raise ValueError("Grouped review and layer catalogue grids disagree")
        self.layers = {row["filename"]: row for row in catalogue["layers"]}
        if len(self.layers) != len(catalogue["layers"]):
            raise ValueError("Grouped catalogue layer names must be unique")
        self._locators = {}
        self._headers = {}
        self.verified_files = {}
        for group in grid["groups"]:
            if group["cutoff_cent"] != self.cutoff_cent:
                continue
            for segment in group["segments"]:
                filename = segment["source_filename"]
                if filename in self._locators or filename not in self.layers:
                    raise ValueError(f"Unexpected grouped channel: {filename}")
                self._locators[filename] = group, segment
        if self._locators.keys() != self.layers.keys():
            raise ValueError("Grouped review is missing channels listed in the catalogue")

    def segment(self, filename, *, verify_payload=True):
        group, row = self._locators[filename]
        root = self.review_path.parent
        path = _relative_file(root, group["path"], root)
        stamp = group_stamp(path.stat())
        if path not in self._headers:
            self._headers[path] = self._inspect(path, group, stamp)
        header, payload_offset, stored, offset_xyz, previous_stamp = self._headers[path]
        if stamp != previous_stamp:
            raise ValueError(f"Grouped NRRD changed during reading: {path}")
        if verify_payload and path not in self.verified_files:
            if sha256_file(path) != group["sha256"] or group_stamp(path.stat()) != stamp:
                raise ValueError(f"Grouped NRRD checksum differs from its review: {path}")
            self.verified_files[path] = (stamp, group["sha256"])
        return GroupedSegmentReference(path, group["sha256"], row["segment_id"], row["layer"],
                                       row["label_value"], row["gzip_offset"], row["gzip_bytes"],
                                       stored, offset_xyz, stamp[2], header, payload_offset, stamp,
                                       row["empty_segment"])

    def _inspect(self, path, group, stamp):
        header, payload_offset = _read_header(path)
        segments = group["segments"]
        shape = self.geometry.shape_tyx
        x, y, z, channels = _ints(header.get("sizes", ""), 4, "grouped sizes")
        stored = _shape((z, y, x), "grouped crop shape")
        offset_xyz = _ints(header.get("Segmentation_ReferenceImageExtentOffset", ""), 3, "grouped offset")
        expected_extent = tuple(v for size in reversed(shape) for v in (0, size - 1))
        directions, _, list_direction = header.get("space directions", "").rpartition(" ")
        axes = _vectors(directions, 3, "grouped space directions")
        origin = np.asarray(_vectors(header.get("space origin", ""), 1, "grouped origin")[0])
        geometry = ReferenceGeometry(shape, header["space"], axes,
                                     tuple(origin - np.asarray(offset_xyz) @ np.asarray(axes)))
        if (header.get("dimension") != "4" or header.get("type") != "uint8"
                or header.get("encoding") != "gzip" or list_direction != "none"
                or header.get("kinds", "").split() != ["domain", "domain", "domain", "list"]
                or any(field in header for field in ("data file", "datafile", "byte skip", "line skip"))
                or channels != len(segments) or tuple(group["stored_shape_tyx"]) != stored
                or tuple(group["offset_xyz"]) != offset_xyz
                or _ints(header.get("Segmentation_ReferenceImageExtent", ""), 6, "grouped extent") != expected_extent
                or any(o < 0 or o+n > s for o, n, s in zip(reversed(offset_xyz), stored, shape))
                or not _same_geometry(geometry, self.geometry)
                or group_stamp(path.stat()) != stamp or group["bytes"] != stamp[2]
                or not re.fullmatch(r"[0-9a-f]{64}", group["sha256"])):
            raise ValueError(f"Grouped NRRD header disagrees with its review: {path}")
        next_offset = payload_offset
        for channel, segment in enumerate(segments):
            prefix = f"Segment{channel}_"
            length = segment["gzip_bytes"]
            extent = _ints(header.get(prefix + "Extent", ""), 6, "channel extent")
            empty = all(extent[axis+1] < extent[axis] for axis in (0, 2, 4))
            expected = extent if empty else tuple(value + offset_xyz[axis//2] for axis, value in enumerate(extent))
            if (segment["cutoff_cent"] != self.cutoff_cent
                    or type(segment["layer"]) is not int or segment["layer"] != channel
                    or type(segment["label_value"]) is not int or segment["label_value"] != 1
                    or type(segment["gzip_offset"]) is not int or segment["gzip_offset"] != next_offset
                    or type(length) is not int or length <= 0 or next_offset + length > stamp[2]
                    or header.get(prefix + "ID") != segment["segment_id"]
                    or header.get(prefix + "Layer") != str(channel) or header.get(prefix + "LabelValue") != "1"
                    or segment["empty_segment"] is not empty or tuple(segment["extent_xyz"]) != expected
                    or not empty and any(lo < 0 or hi < lo or hi >= size for lo, hi, size in
                                         zip(extent[::2], extent[1::2], reversed(stored)))):
                raise ValueError(f"Grouped NRRD segment index disagrees with its header: {path}")
            next_offset += length
        if next_offset != stamp[2]:
            raise ValueError(f"Grouped NRRD index does not exhaust the file: {path}")
        return header, payload_offset, stored, offset_xyz, stamp


def open_grouped_catalog(path):
    """Resolve an exact cutoff layer catalogue to its canonical indexed groups."""
    path = Path(path).resolve()
    data = path.read_bytes()
    record = json.loads(data)
    if record.get("layout") != GROUPED_CATALOG_LAYOUT or record.get("source_storage") != "grouped":
        raise ValueError("Expected a canonical grouped sweep layer catalogue")
    return GroupedCatalogue(path, record, hashlib.sha256(data).hexdigest())
