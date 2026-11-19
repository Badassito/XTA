"""Export flat registered image and YOLO-label pairs on the annotation grid."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from scipy import ndimage

from .labels import _polygon_rows, _read_planes
from .packet import load_packet, write_packet
from .video import probe_video


def _registered_metadata(video, supplied):
    if supplied is not None:
        if isinstance(supplied, (str, Path)):
            result = json.loads(Path(supplied).read_text(encoding="utf-8-sig"))
        elif isinstance(supplied, dict):
            result = dict(supplied)
        else:
            raise ValueError("Registered movie export metadata must be a JSON object")
        if not isinstance(result, dict):
            raise ValueError("Registered movie export metadata must be a JSON object")
        return result
    for suffix in (".registration.json", ".export.json"):
        candidate = video.with_suffix(suffix)
        if candidate.is_file():
            result = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if isinstance(result, dict) and {
                "output_sha256",
                "source_role",
                "destination_role",
                "output_to_input_xyz",
            }.issubset(result):
                return result
    return None


def _check_registered_movie(packet, video, direction, metadata, assume_registered_grid):
    if direction not in ("fixed-to-moving", "moving-to-fixed"):
        raise ValueError("Direction must be fixed-to-moving or moving-to-fixed")
    source, target = ("fixed", "moving") if direction == "fixed-to-moving" else ("moving", "fixed")
    info = probe_video(video)
    if tuple(info["shape_zyx"]) != packet.shape(source):
        raise ValueError("Registered movie dimensions must match the original label grid")
    matrix = packet.matrix(direction)
    if metadata is None:
        if not assume_registered_grid:
            raise ValueError(
                "Supply the registered movie's export metadata, or explicitly assert assume_registered_grid=True"
            )
    else:
        if metadata.get("output_sha256") != info["sha256"]:
            raise ValueError("Registered movie checksum disagrees with its export metadata")
        if (
            metadata.get("source_role") != target
            or metadata.get("destination_role") != source
            or metadata.get("source_sha256") != packet.metadata[target]["sha256"]
            or metadata.get("shape_zyx") != list(packet.shape(source))
        ):
            raise ValueError(
                "Registered movie export direction or source identity disagrees with this packet"
            )
        declared = np.asarray(metadata.get("output_to_input_xyz"), dtype=float)
        if declared.shape != (4, 4) or not np.allclose(declared, matrix, rtol=1e-9, atol=1e-8):
            raise ValueError("Registered movie was resampled using a different mapping")
    return source, target, matrix, info


def checked_registered_video(
    packet,
    video,
    *,
    direction="fixed-to-moving",
    registered_metadata=None,
    assume_registered_grid=False,
):
    """Verify a resliced movie's content identity and its sampling direction."""
    packet = load_packet(packet) if isinstance(packet, (str, Path)) else packet
    if not packet.usable:
        raise ValueError("This packet has no validated registration")
    video = Path(video)
    if registered_metadata is not None:
        metadata = _registered_metadata(video, registered_metadata)
    else:
        metadata = _registered_metadata(video, None)
        recorded = packet.metadata.get("registered_video")
        if metadata is None and isinstance(recorded, dict):
            metadata = recorded
    source, target, matrix, info = _check_registered_movie(
        packet, video, direction, metadata, assume_registered_grid
    )
    return dict(
        annotation_role=source,
        intensity_role=target,
        output_to_input_xyz=matrix,
        video_info=info,
        export_metadata=metadata,
    )


def _observed_roi(packet, source, target, matrix, z, bounds):
    x0, y0, x1, y1 = bounds
    xx = np.arange(x0, x1, dtype=float)[None, :]
    shape = (y1 - y0, x1 - x0)
    observed = np.empty(shape, bool)
    maximum = np.asarray(packet.shape(target)[::-1]) - 1
    source_mask, target_mask = packet.mask(source), packet.mask(target)
    for first in range(y0, y1, 128):
        last = min(y1, first + 128)
        yy = np.arange(first, last, dtype=float)[:, None]
        coords = np.empty((3, last - first, x1 - x0), float)
        for axis in range(3):
            coords[axis] = (
                matrix[axis, 0] * xx + matrix[axis, 1] * yy + matrix[axis, 2] * z + matrix[axis, 3]
            )
        good = np.all((coords >= 0) & (coords <= maximum[:, None, None]), axis=0)
        if source_mask is not None:
            good &= source_mask[first:last, x0:x1]
        if target_mask is not None:
            values = ndimage.map_coordinates(
                target_mask,
                coords[1::-1],
                output=np.float32,
                order=1,
                mode="constant",
                cval=0,
                prefilter=False,
            )
            good &= values >= 1 - 1e-6
        observed[first - y0 : last - y0] = good
    return observed


def _clip_annotation(packet, source, target, matrix, plane, exact_dir, stem):
    payload = plane.path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != plane.sha256:
        raise ValueError(f"Source annotation changed while exporting: {plane.path}")
    source_rows = sum(len(polygons) for polygons in plane.polygons.values())
    result = dict(
        source_polygon_rows=source_rows,
        output_polygon_rows=source_rows,
        removed_unobserved_pixels=0,
        clipped_to_observed_field=False,
        yolo_false_positive_pixels=0,
        yolo_false_negative_pixels=0,
        exact_mask=None,
    )
    if plane.bounds_xyxy is None:
        return payload, result
    x0, y0, x1, y1 = plane.bounds_xyxy
    observed = _observed_roi(packet, source, target, matrix, plane.z, plane.bounds_xyxy)
    raster_shape = (y1 - y0, x1 - x0)
    rows = []
    masks = {class_id: np.zeros(raster_shape, np.uint8) for class_id in plane.polygons}
    affected = False
    for class_id, polygons in plane.polygons.items():
        for polygon in polygons:
            raster = np.zeros(raster_shape, np.uint8)
            cv2.fillPoly(raster, [polygon - [x0, y0]], 1)
            missing = int(np.count_nonzero(raster & ~observed))
            result["removed_unobserved_pixels"] += missing
            affected |= missing > 0
            clipped = raster & observed
            masks[class_id] |= clipped
            # Separate source polygons have no persistent 3D instance identity,
            # but retain their per-plane rows when clipping creates components.
            lines, loss = _polygon_rows(clipped, (x0, y0), packet.shape(source)[1:], class_id)
            if loss["false_positive_pixels"]:
                reconstructed = np.zeros(raster_shape, np.uint8)
                height, width = packet.shape(source)[1:]
                for line in lines:
                    xy = np.asarray(line.split()[1:], float).reshape(-1, 2) * [width, height]
                    points = np.rint(xy).astype(np.int32) - [x0, y0]
                    cv2.fillPoly(reconstructed, [points.astype(np.int32)], 1)
                if np.any((reconstructed != 0) & ~observed):
                    raise ValueError(
                        "A clipped polygon would cover an unobserved hole. Crop that annotation region before exporting ordinary YOLO supervision."
                    )
            rows.extend(lines)
            result["yolo_false_positive_pixels"] += loss["false_positive_pixels"]
            result["yolo_false_negative_pixels"] += loss["false_negative_pixels"]
    if not affected:
        result["yolo_false_positive_pixels"] = 0
        result["yolo_false_negative_pixels"] = 0
        return payload, result
    exact_dir.mkdir(exist_ok=True)
    exact_name = stem + ".npz"
    np.savez_compressed(
        exact_dir / exact_name,
        origin_xy=np.asarray([x0, y0]),
        source_z=np.asarray(plane.z),
        class_ids=np.asarray(list(masks)),
        foreground=np.asarray(list(masks.values())),
        observed=observed,
    )
    result.update(
        output_polygon_rows=len(rows),
        clipped_to_observed_field=True,
        exact_mask="provenance/clipped_masks/" + exact_name,
    )
    if not rows:
        # A wholly unobserved positive must not become a training negative.
        result["omission_reason"] = (
            "unobserved_foreground"
            if not any(mask.any() for mask in masks.values())
            else "foreground_cannot_form_a_YOLO_polygon"
        )
        return None, result
    return ("\n".join(rows) + "\n").encode("utf-8"), result


def _frame_filter(indexes):
    runs = []
    for z in indexes:
        if runs and runs[-1][1] + 1 == z:
            runs[-1][1] = z
        else:
            runs.append([z, z])
    clauses = [f"eq(n,{lo})" if lo == hi else f"between(n,{lo},{hi})" for lo, hi in runs]
    return "select='" + "+".join(clauses) + "'"


def _read_exact(stream, count):
    parts = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise IOError("Registered movie ended before the requested annotation plane")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def _write_image(stage, stem, raw, shape, packet, source, target, matrix, z):
    original = np.frombuffer(raw, np.uint8).reshape(shape)
    observed = _observed_roi(packet, source, target, matrix, z, (0, 0, shape[1], shape[0]))
    image = np.where(observed, original, 0).astype(np.uint8)
    path = stage / (stem + ".png")
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"PNG encoding failed for {stem}")
    path.write_bytes(encoded.tobytes())
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint8 or not np.array_equal(decoded, image):
        raise ValueError(f"PNG pixel roundtrip failed for {stem}")
    return dict(
        raw_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        registered_frame_pixel_sha256=hashlib.sha256(raw).hexdigest(),
        annotation_observed_pixels=int(observed.sum()),
        masked_unobserved_pixels=int((~observed).sum()),
        removed_nonzero_pixels=int(np.count_nonzero(original[~observed])),
    )


def export_image_label_pairs(
    packet,
    registered_video,
    labels_dir,
    output_dir,
    *,
    direction="fixed-to-moving",
    filename_index_base=1,
    registered_metadata=None,
    assume_registered_grid=False,
    workers=2,
    artifact_start_z=0,
):
    """Pair resliced intensity PNGs with YOLO labels on their original planes.

    For ``fixed-to-moving``, annotations come from the original fixed scan and
    ``registered_video`` contains moving intensity resampled onto the fixed
    grid. The opposite direction uses moving labels and fixed intensity.
    Missing annotation files stay absent, and empty files remain explicit
    background only where the resliced image is observed. A clipping sidecar
    retains the exact masks when a polygon crosses acquisition coverage.
    """
    packet = load_packet(packet) if isinstance(packet, (str, Path)) else packet
    if not packet.usable:
        raise ValueError("This packet has no validated registration for image-label export")
    if filename_index_base not in (0, 1) or isinstance(filename_index_base, bool):
        raise ValueError("Filename index base must be zero or one")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Image worker count must be a positive integer")
    if (
        isinstance(artifact_start_z, bool)
        or not isinstance(artifact_start_z, int)
        or artifact_start_z < 0
    ):
        raise ValueError("Artifact cutoff must be a nonnegative native reference frame")
    video, labels, output = Path(registered_video), Path(labels_dir), Path(output_dir)
    if not labels.is_dir():
        raise ValueError("Select one existing indexed YOLO label directory")
    if output.exists():
        raise FileExistsError(output)
    checked = checked_registered_video(
        packet,
        video,
        direction=direction,
        registered_metadata=registered_metadata,
        assume_registered_grid=assume_registered_grid,
    )
    source, target = checked["annotation_role"], checked["intensity_role"]
    matrix, info, metadata = (
        checked["output_to_input_xyz"],
        checked["video_info"],
        checked["export_metadata"],
    )
    shape = packet.shape(source)
    planes, classes, input_rows = _read_planes(labels, shape, filename_index_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_stat = video.stat()
    with tempfile.TemporaryDirectory(prefix=".registration_pairs_", dir=output.parent) as work:
        stage = Path(work) / "pairs"
        provenance = stage / "provenance"
        provenance.mkdir(parents=True)
        records = []
        selected = []
        for z, plane in sorted(planes.items()):
            stem = f"{video.stem}_{z+1:04d}"
            content, clipped = _clip_annotation(
                packet, source, target, matrix, plane, provenance / "clipped_masks", stem
            )
            record = dict(
                native_z=z,
                encoded_index_one_based=z + 1,
                source_annotation_filename=plane.path.name,
                source_annotation_sha256=plane.sha256,
                early_reference_region=z < artifact_start_z,
                annotation="foreground" if plane.polygons else "explicit_background",
                **clipped,
            )
            if content is None:
                record.update(annotation="unobserved_positive_omitted", image=None, label=None)
            else:
                label_name = stem + ".txt"
                (stage / label_name).write_bytes(content)
                record.update(
                    image=stem + ".png",
                    label=label_name,
                    label_sha256=hashlib.sha256(content).hexdigest(),
                )
                selected.append((z, stem, record))
            records.append(record)
        if selected:
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-noautorotate",
                "-threads",
                "2",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-filter_threads",
                "1",
                "-vf",
                _frame_filter([z for z, _, _ in selected]),
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(selected)),
                "-pix_fmt",
                "gray",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            with (
                (provenance / "decode.log").open("wb") as log,
                subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as process,
            ):
                try:
                    with ThreadPoolExecutor(workers) as pool:
                        pending = deque()
                        for _, stem, record in selected:
                            raw = _read_exact(process.stdout, shape[1] * shape[2])
                            pending.append(
                                (
                                    record,
                                    pool.submit(
                                        _write_image,
                                        stage,
                                        stem,
                                        raw,
                                        shape[1:],
                                        packet,
                                        source,
                                        target,
                                        matrix,
                                        record["native_z"],
                                    ),
                                )
                            )
                            if len(pending) >= workers + 1:
                                earlier, future = pending.popleft()
                                earlier.update(future.result())
                        for record, future in pending:
                            record.update(future.result())
                    if process.stdout.read(1) or process.wait() != 0:
                        raise IOError(
                            "Registered movie extraction failed or produced unexpected frames"
                        )
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.stdout.close()
        if (video.stat().st_size, video.stat().st_mtime_ns) != (
            video_stat.st_size,
            video_stat.st_mtime_ns,
        ):
            raise ValueError("Registered movie changed while exporting image-label pairs")
        summary = dict(
            schema="xta.registration.image_label_pairs",
            schema_version=1,
            pair=packet.metadata["pair"],
            direction=direction,
            annotation_role=source,
            intensity_role=target,
            registered_video_filename=video.name,
            registered_video_sha256=info["sha256"],
            registered_export_metadata_checked=metadata is not None,
            grid_shape_zyx=list(shape),
            grid_spacing_xyz=list(packet.spacing(source)),
            class_ids=classes,
            provided_annotation_planes=len(planes),
            unknown_annotation_planes=shape[0] - len(planes),
            image_label_pairs=len(selected),
            foreground_pairs=sum(
                r["annotation"] == "foreground" and r["image"] is not None for r in records
            ),
            explicitly_empty_pairs=sum(r["annotation"] == "explicit_background" for r in records),
            omitted_unobserved_positive_planes=sum(r["image"] is None for r in records),
            source_polygon_rows=input_rows,
            output_polygon_rows=sum(r["output_polygon_rows"] for r in records),
            clipped_annotation_planes=sum(r["clipped_to_observed_field"] for r in records),
            removed_unobserved_pixels=sum(r["removed_unobserved_pixels"] for r in records),
            yolo_false_positive_pixels=sum(r["yolo_false_positive_pixels"] for r in records),
            yolo_false_negative_pixels=sum(r["yolo_false_negative_pixels"] for r in records),
            source_label_index_base=filename_index_base,
            output_label_index_base=1,
            artifact_start_z=artifact_start_z,
            pixel_validation="Every PNG decoded and compared exactly with its registered frame after applying annotation-observed coverage",
            image_preprocessing="Pixels outside the original annotation field or the mapped intensity acquisition are set to zero. Image and label pairs retain only their shared observed domain.",
            coverage="Black acquisition exterior and mapped coordinates outside either scan are unobserved. Empty source annotations apply only where image observations exist.",
            mapping_packet="mapping.zip",
            frames=records,
        )
        write_packet(provenance / "mapping.zip", packet.metadata, packet.masks)
        (provenance / "pairs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return summary
