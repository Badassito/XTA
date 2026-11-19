"""Carry sparse YOLO segmentation planes between registered voxel grids.

Each supplied source plane represents a one-voxel-thick annotation slab.
Missing planes remain unknown. Native target slices can consequently contain
partial annotations, so the export retains exact cropped masks and coverage.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..lta_inputs import split_indexed_stem

if TYPE_CHECKING:
    from .packet import RegistrationPacket


@dataclass(frozen=True)
class _AnnotationPlane:
    z: int
    path: Path
    sha256: str
    polygons: dict[int, list[np.ndarray]]
    bounds_xyxy: tuple[int, int, int, int] | None


def _read_planes(folder: Path, shape: tuple[int, int, int], index_base: int):
    depth, height, width = shape
    planes = {}
    classes = set()
    series = set()
    rows = 0
    for path in sorted(folder.glob("*.txt")):
        prefix, index = split_indexed_stem(path)
        if index is None:
            if path.stem.lower() in {"classes", "notes", "readme"}:
                continue
            raise ValueError(f"A slice label needs a final numeric suffix: {path.name}")
        z = index - index_base
        if z in planes or not 0 <= z < depth:
            raise ValueError(f"Duplicate or out-of-range label frame at {path.name}")
        series.add(prefix)
        payload = path.read_bytes()
        polygons = {}
        all_points = []
        for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                class_id = int(fields[0])
                coordinates = np.asarray(fields[1:], dtype=np.float64)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid YOLO segmentation row at {path}:{line_number}") from exc
            if (
                class_id < 0
                or coordinates.size < 6
                or coordinates.size % 2
                or not np.all(np.isfinite(coordinates))
                or np.any(coordinates < 0)
                or np.any(coordinates > 1)
            ):
                raise ValueError(f"Invalid normalized YOLO polygon at {path}:{line_number}")
            points = np.rint(coordinates.reshape(-1, 2) * [width, height]).astype(np.int32)
            points = np.clip(points, [0, 0], [width - 1, height - 1])
            polygons.setdefault(class_id, []).append(points)
            all_points.append(points)
            classes.add(class_id)
            rows += 1
        bounds = None
        if all_points:
            vertices = np.concatenate(all_points)
            lower = vertices.min(axis=0)
            upper = vertices.max(axis=0) + 1
            bounds = tuple(int(value) for value in (*lower, *upper))
        planes[z] = _AnnotationPlane(z, path, hashlib.sha256(payload).hexdigest(), polygons, bounds)
    if not planes:
        raise ValueError(f"No indexed slice annotations in {folder}")
    if len(series) != 1:
        raise ValueError("Select a directory containing one slice-label series")
    return planes, sorted(classes), rows


def _transformed_bounds(bounds, z, matrix, shape):
    x0, y0, x1, y1 = bounds
    corners = np.asarray(
        list(itertools.product([x0 - 0.5, x1 - 0.5], [y0 - 0.5, y1 - 0.5], [z - 0.5, z + 0.5]))
    )
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    maximum = np.asarray(shape)[::-1]
    lower = np.clip(np.floor(transformed.min(axis=0)), 0, maximum).astype(int)
    upper = np.clip(np.ceil(transformed.max(axis=0)) + 1, 0, maximum).astype(int)
    return lower, upper


def _polygon_rows(mask, origin_xy, shape_yx, class_id):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    reconstructed = np.zeros_like(mask)
    lines = []
    holes = 0
    degenerate = 0
    height, width = shape_yx
    for index, contour in enumerate(contours):
        if hierarchy is not None and hierarchy[0, index, 3] >= 0:
            holes += 1
            continue
        if len(contour) < 3 or abs(cv2.contourArea(contour)) == 0:
            degenerate += 1
            continue
        points = contour.reshape(-1, 2) + np.asarray(origin_xy)
        normalized = points.astype(np.float64) / [width, height]
        line = str(class_id) + " " + " ".join(f"{value:.12f}" for value in normalized.ravel())
        lines.append(line)
        decoded = np.asarray(line.split()[1:], dtype=np.float64).reshape(-1, 2)
        decoded = np.rint(decoded * [width, height]).astype(np.int32) - np.asarray(origin_xy)
        cv2.fillPoly(reconstructed, [decoded.astype(np.int32)], 1)
    return lines, dict(
        holes_in_exact_mask=holes,
        omitted_degenerate_components=degenerate,
        false_positive_pixels=int(np.count_nonzero((reconstructed != 0) & (mask == 0))),
        false_negative_pixels=int(np.count_nonzero((reconstructed == 0) & (mask != 0))),
    )


class _LabelReslicer:
    def __init__(
        self, planes, classes, source_shape, target_shape, matrix, source_mask, target_mask
    ):
        self.planes = planes
        self.classes = classes
        self.source_shape = source_shape
        self.target_shape = target_shape
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.known = np.zeros(source_shape[0], bool)
        self.known[list(planes)] = True
        self.bounds = []
        for z, plane in planes.items():
            if plane.bounds_xyxy is not None:
                lower, upper = _transformed_bounds(plane.bounds_xyxy, z, matrix, target_shape)
                if np.all(upper > lower):
                    self.bounds.append((lower, upper))
        self.source_raster = lru_cache(maxsize=32)(self._source_raster)

    def _source_raster(self, z, class_id):
        plane = self.planes[z]
        x0, y0, x1, y1 = plane.bounds_xyxy
        raster = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for polygon in plane.polygons.get(class_id, []):
            cv2.fillPoly(raster, [polygon - [x0, y0]], 1)
        return raster

    def active_planes(self):
        return sorted({z for lower, upper in self.bounds for z in range(lower[2], upper[2])})

    def sample_plane(self, z):
        boxes = [(lower, upper) for lower, upper in self.bounds if lower[2] <= z < upper[2]]
        lower = np.min([entry[0] for entry in boxes], axis=0)
        upper = np.max([entry[1] for entry in boxes], axis=0)
        x0, y0 = lower[:2]
        x1, y1 = upper[:2]
        foreground = np.zeros((len(self.classes), y1 - y0, x1 - x0), np.uint8)
        coverage = np.zeros((y1 - y0, x1 - x0), bool)
        xx = np.arange(x0, x1, dtype=np.float64)[None, :]
        maximum = np.asarray(self.source_shape)[::-1]
        for first in range(y0, y1, 96):
            last = min(y1, first + 96)
            yy = np.arange(first, last, dtype=np.float64)[:, None]
            indices = np.empty((3, last - first, x1 - x0), np.int32)
            for axis in range(3):
                mapped = (
                    self.inverse[axis, 0] * xx
                    + self.inverse[axis, 1] * yy
                    + self.inverse[axis, 2] * z
                    + self.inverse[axis, 3]
                )
                indices[axis] = np.clip(np.floor(mapped + 0.5), -1, maximum[axis]).astype(np.int32)
            valid = np.all((indices >= 0) & (indices < maximum[:, None, None]), axis=0)
            if self.target_mask is not None:
                valid &= self.target_mask[first:last, x0:x1]
            source_z = indices[2]
            valid &= self.known[np.clip(source_z, 0, len(self.known) - 1)]
            if self.source_mask is not None:
                flat = indices[1] * self.source_shape[2] + indices[0]
                np.clip(flat, 0, self.source_mask.size - 1, out=flat)
                valid &= self.source_mask.reshape(-1)[flat]
            coverage[first - y0 : last - y0] = valid
            for source_plane in np.unique(source_z[valid]):
                layer = self.planes[int(source_plane)]
                if layer.bounds_xyxy is None:
                    continue
                sx0, sy0, sx1, sy1 = layer.bounds_xyxy
                use = valid & (source_z == source_plane)
                use &= (indices[0] >= sx0) & (indices[0] < sx1)
                use &= (indices[1] >= sy0) & (indices[1] < sy1)
                if not use.any():
                    continue
                for channel, class_id in enumerate(self.classes):
                    raster = self.source_raster(int(source_plane), class_id)
                    foreground[channel, first - y0 : last - y0][use] = raster[
                        indices[1][use] - sy0, indices[0][use] - sx0
                    ]
        return (int(x0), int(y0)), foreground, coverage


def _sample_planes(engine, workers):
    planes = iter(engine.active_planes())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque(
            (z, executor.submit(engine.sample_plane, z))
            for z in itertools.islice(planes, workers + 1)
        )
        while pending:
            z, future = pending.popleft()
            sampled = future.result()
            next_z = next(planes, None)
            if next_z is not None:
                pending.append((next_z, executor.submit(engine.sample_plane, next_z)))
            yield z, sampled


def _write_transfer(engine, output, stem, provenance, workers):
    labels = output / "partial_yolo_labels"
    exact = output / "exact_resliced_masks"
    labels.mkdir()
    exact.mkdir()
    domain = dict(
        source_known_planes=engine.known,
        source_empty_planes=np.asarray(
            sorted(z for z, plane in engine.planes.items() if not plane.polygons), dtype=np.int64
        ),
        source_to_target_xyz=engine.matrix,
        source_shape_zyx=np.asarray(engine.source_shape),
        target_shape_zyx=np.asarray(engine.target_shape),
    )
    if engine.source_mask is not None:
        domain["source_observed_yx"] = engine.source_mask
    if engine.target_mask is not None:
        domain["target_observed_yx"] = engine.target_mask
    np.savez_compressed(output / "annotation_domain.npz", **domain)
    records = []
    for z, (origin, masks, coverage) in _sample_planes(engine, workers):
        positive = int(np.count_nonzero(masks))
        if not positive:
            continue
        name = f"{stem}_{z + 1:04d}"
        rows = []
        stats = []
        for class_id, mask in zip(engine.classes, masks):
            lines, changes = _polygon_rows(mask, origin, engine.target_shape[1:], class_id)
            rows.extend(lines)
            stats.append(dict(class_id=class_id, positive_pixels=int(mask.sum()), **changes))
        if rows:
            (labels / f"{name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        np.savez_compressed(
            exact / f"{name}.npz",
            class_ids=np.asarray(engine.classes),
            foreground=masks,
            known_source_coverage=coverage,
            origin_xy=np.asarray(origin),
            target_z=np.asarray(z),
            target_shape_zyx=np.asarray(engine.target_shape),
        )
        records.append(
            dict(
                native_target_z=z,
                encoded_index_one_based=z + 1,
                positive_class_pixels=positive,
                yolo_rows=len(rows),
                yolo_file=f"partial_yolo_labels/{name}.txt" if rows else None,
                exact_mask=f"exact_resliced_masks/{name}.npz",
                known_roi_pixels=int(coverage.sum()),
                roi_pixels=int(coverage.size),
                classes=stats,
            )
        )
    false_positive = sum(c["false_positive_pixels"] for row in records for c in row["classes"])
    false_negative = sum(c["false_negative_pixels"] for row in records for c in row["classes"])
    summary = dict(
        provenance,
        coordinate_convention="Native zero-based XYZ voxel centers; output suffixes are one-based.",
        source_to_target_xyz=engine.matrix.tolist(),
        source_shape_zyx=list(engine.source_shape),
        target_shape_zyx=list(engine.target_shape),
        resampling="Inverse nearest-neighbor sampling of one-voxel-thick source annotation slabs.",
        label_semantics="Partial positive annotations. Missing source planes remain unknown. A target TXT does not certify complete slice annotation.",
        source_semantics="Polygons rasterized with cv2.fillPoly and unioned by class. Cross-slice instance identity is unavailable.",
        exact_mask_scope="Each NPZ describes its cropped target ROI. known_source_coverage is valid only inside that ROI.",
        annotation_domain="annotation_domain.npz",
        annotation_domain_semantics="Source known/empty planes, observed fields, shapes and the matrix retain coverage information beyond the cropped positive ROIs.",
        source_known_planes=np.flatnonzero(engine.known).tolist(),
        source_empty_planes=sorted(z for z, plane in engine.planes.items() if not plane.polygons),
        source_unknown_plane_count=int((~engine.known).sum()),
        class_ids=engine.classes,
        positive_target_planes=len(records),
        positive_class_pixels=sum(row["positive_class_pixels"] for row in records),
        yolo_total_rows=sum(row["yolo_rows"] for row in records),
        yolo_roundtrip_false_positive_pixels=false_positive,
        yolo_roundtrip_false_negative_pixels=false_negative,
        yolo_roundtrip_exact=false_positive == 0 and false_negative == 0,
        slices=records,
    )
    (output / "transfer_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def transfer_yolo_labels(
    packet: RegistrationPacket | str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    direction: str = "fixed-to-moving",
    filename_index_base: int = 1,
    target_stem: str | None = None,
    workers: int = 1,
) -> dict:
    """Export source annotations on the other acquisition's native voxel grid.

    ``labels_dir`` contains one series of ``<stem>_<index>.txt`` files. Each
    supplied empty file denotes an explicitly annotated empty source plane.
    Missing files carry no annotation. The output path must be new. Publication
    happens only after every plane and the manifest have been written.
    """
    from .packet import load_packet, validate_metadata

    if isinstance(packet, (str, Path)):
        packet = load_packet(packet)
    validate_metadata(packet.metadata)
    if not packet.usable:
        raise ValueError("Registration packet has no validated mapping")
    normalized = direction.replace("_", "-")
    if normalized not in {"fixed-to-moving", "moving-to-fixed"}:
        raise ValueError("Direction must be fixed-to-moving or moving-to-fixed")
    if filename_index_base not in (0, 1) or isinstance(filename_index_base, bool):
        raise ValueError("Filename index base must be zero or one")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("Use between one and eight label sampling workers")
    source_role, target_role = (
        ("fixed", "moving") if normalized == "fixed-to-moving" else ("moving", "fixed")
    )
    source_shape = tuple(packet.shape(source_role))
    target_shape = tuple(packet.shape(target_role))
    matrix = np.asarray(packet.matrix(normalized), dtype=np.float64)
    source_mask = packet.mask(source_role)
    target_mask = packet.mask(target_role)
    for mask, shape in [(source_mask, source_shape), (target_mask, target_shape)]:
        if mask is not None and (mask.dtype != np.bool_ or mask.shape != shape[1:]):
            raise ValueError("Observed field mask must use native boolean YX coordinates")
    source = Path(labels_dir)
    output = Path(output_dir)
    if not source.is_dir():
        raise ValueError(f"Label directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    target_scan = packet.metadata[target_role]
    stem = target_stem or Path(target_scan["filename"]).stem
    if not stem or Path(stem).name != stem or stem in {".", ".."} or any(c in stem for c in "/\\:"):
        raise ValueError("Target stem must be a simple filename stem")
    planes, classes, row_count = _read_planes(source, source_shape, filename_index_base)
    engine = _LabelReslicer(
        planes, classes, source_shape, target_shape, matrix, source_mask, target_mask
    )
    provenance = dict(
        direction=normalized,
        packet_pair=packet.metadata["pair"],
        source_scan={k: v for k, v in packet.metadata[source_role].items() if k != "observed_mask"},
        target_scan={k: v for k, v in target_scan.items() if k != "observed_mask"},
        source_filename_index_base=filename_index_base,
        source_polygon_rows=row_count,
        source_files=[
            dict(
                native_z=z,
                filename=plane.path.name,
                sha256=plane.sha256,
                state="foreground" if plane.polygons else "known_background",
            )
            for z, plane in sorted(planes.items())
        ],
        source_visibility_applied=source_mask is not None,
        target_visibility_applied=target_mask is not None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".label_transfer_", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        summary = _write_transfer(engine, staged, stem, provenance, workers)
        staged.rename(output)
    return summary
