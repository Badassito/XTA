"""Publish one exact cutoff-support overlay per requested sweep resolution."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .json_publication import _fsync_parent_directory, _staged_sweep_directory


HEATMAP_DIRECTORY = "heatmaps"
HEATMAP_STAGING_ENV = "XTA_YOLO_SWEEP_MEDIA_ROOT"
_GRID_NAME = re.compile(r"(?:native|low_quality/[A-Za-z0-9_.+\-]+)")
_PALETTE_STOPS = ((64, 64, 255), (0, 224, 160), (255, 240, 32))


def _packed_path(root, cutoff, grid):
    if not isinstance(grid, str) or not _GRID_NAME.fullmatch(grid) or any(
        part in (".", "..") for part in grid.split("/")
    ):
        raise ValueError(f"Invalid heatmap grid name: {grid!r}")
    if type(cutoff) is not int or not 0 <= cutoff <= 100:
        raise ValueError(f"Invalid heatmap cutoff: {cutoff!r}")
    return Path(root) / f"conf_{cutoff:03d}" / f"{grid}.packbits"


def heatmap_input_contract(ownership, save_options, *, root, cutoff, source_shape):
    """Check the accepted scalar receipt for each requested video grid."""
    description = ownership.get("heatmap_inputs")
    grids = description.get("grids") if isinstance(description, dict) else None
    if not isinstance(grids, dict) or type(description.get("streaming_decode")) is not bool:
        raise ValueError("Scalar cutoff has no verified heatmap inputs")
    if ("native" in grids) != ("high_quality" in save_options):
        raise ValueError("Scalar cutoff heatmap native grid differs from --save high_quality")
    if any(grid.startswith("low_quality/") for grid in grids) != ("low_quality" in save_options):
        raise ValueError("Scalar cutoff heatmap preview grids differ from --save low_quality")
    for grid, shape in grids.items():
        path = _packed_path(root, cutoff, grid)
        if not isinstance(shape, list) or len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f"Invalid dimensions for confidence heatmap grid {grid!r}")
        if grid == "native" and shape != source_shape:
            raise ValueError("The native confidence heatmap differs from the source geometry")
        frames, height, width = shape
        if path.is_symlink() or not path.is_file() or path.stat().st_size != frames * ((height * width + 7) // 8):
            raise ValueError(f"Scalar cutoff heatmap mask is incomplete: {path}")
    return description


def stage_heatmap_mask(root, cutoff, grid, mask):
    """Write a frame-addressable snapshot of the exact final binary mask."""
    from .cuda_d1 import _read_binary_volume_slice_u8

    path = _packed_path(root, cutoff, grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=path.name + ".", suffix=".partial", dir=path.parent, delete=False,
        ) as stream:
            staging = Path(stream.name)
            for index in range(mask.shape[0]):
                bits = np.packbits(_read_binary_volume_slice_u8(mask, index), bitorder="little")
                stream.write(memoryview(bits))
        staging.replace(path)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        remainder = bytearray(data)
        while len(remainder) < size:
            block = stream.read(size - len(remainder))
            if not block:
                raise ValueError("A confidence heatmap input ended before its declared frame count")
            remainder.extend(block)
        data = remainder
    return np.frombuffer(data, np.uint8)


@contextmanager
def _open_gray_frames(work, inputs, shape, *, streaming_decode):
    """Reuse verified gray8 frames, or read one frame at a time from FFmpeg."""
    frame_count, height, width = shape
    plane_bytes = height * width

    def read_frame(stream):
        return _read_exact(stream, plane_bytes).reshape(height, width)

    ready = work / "source.gray8.json"
    if ready.is_file():
        metadata = json.loads(ready.read_text(encoding="utf-8"))
        if (metadata.get("source") != inputs["source"] or metadata.get("shape_tyx") != list(shape)
                or metadata.get("dtype") != "uint8"):
            raise ValueError("The shared confidence sweep source differs from the branch receipt")
        with (work / "source.gray8").open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != frame_count * plane_bytes:
                raise ValueError("The shared confidence sweep source is incomplete")

            def get_frame(index):
                stream.seek(index * plane_bytes)
                return read_frame(stream)

            yield get_frame
        return

    from .media import (
        _require_bin, _spawn_decode_process, _spawn_subprocess_with_retry,
        _synchronous_decode_stdout, ffmpeg_decode_threads,
    )
    _require_bin("ffmpeg")
    command = [
        "ffmpeg", "-v", "error", "-threads", str(ffmpeg_decode_threads()),
        "-i", inputs["source"]["path"], *(["-map", "0:v:0"] if streaming_decode else []),
        "-frames:v", str(frame_count), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-",
    ]
    proc = _spawn_subprocess_with_retry(lambda: _spawn_decode_process(command), "confidence heatmap source decode")
    with _synchronous_decode_stdout(proc) as stream:
        loaded = -1
        previous = current = None

        def get_frame(index):
            nonlocal loaded, previous, current
            while loaded < index:
                previous, current = current, read_frame(stream)
                loaded += 1
            if index == loaded:
                return current
            if index == loaded - 1:
                return previous
            raise ValueError("Confidence heatmap decoding requires forward frame access")

        yield get_frame
        # Downsampling can leave the final source frames unused. Drain the
        # declared frames before checking the child exit status.
        get_frame(frame_count - 1)
        if stream.read(1):
            raise ValueError("Unexpected trailing frames in the confidence heatmap source")


def _gray_frames_for_grid(source, source_shape, shape):
    """Match the standard LQ intensity resampling with two cached source slices."""
    from ._deps import cv2
    from .media import _center_aligned_source_position, _resize_gray_slice_nearest_or_linear

    if source_shape == shape:
        for index in range(shape[0]):
            yield source(index)
        return
    depth, height, width = shape
    interpolation = cv2.INTER_AREA if height <= source_shape[1] and width <= source_shape[2] else cv2.INTER_LINEAR

    @lru_cache(maxsize=2)
    def resized(index):
        return _resize_gray_slice_nearest_or_linear(source(index), width, height, interpolation)

    for index in range(depth):
        position = _center_aligned_source_position(index, depth, source_shape[0])
        lower = int(math.floor(position))
        upper = min(source_shape[0] - 1, lower + 1)
        alpha = position - lower
        lower_frame = resized(lower)
        if lower == upper or alpha <= 1e-7:
            yield lower_frame
        else:
            yield np.clip(np.rint(
                (1.0 - alpha) * lower_frame.astype(np.float32) + alpha * resized(upper).astype(np.float32)
            ), 0, 255).astype(np.uint8)


def _heatmap_palette(cutoff_count):
    palette = np.zeros((cutoff_count + 1, 3), dtype=np.uint8)
    support = np.linspace(0.0, 1.0, cutoff_count) if cutoff_count > 1 else [1.0]
    for channel, values in enumerate(zip(*_PALETTE_STOPS)):
        palette[1:, channel] = np.rint(np.interp(support, (0.0, 0.5, 1.0), values)).astype(np.uint8)
    return palette


def _render_heatmap(frame, gray, counts, palette):
    frame[:] = gray[:, :, None]
    rows = max(1, (128 * 1024) // frame.shape[1])
    for start in range(0, frame.shape[0], rows):
        region = slice(start, start + rows)
        active = counts[region]
        if np.any(active):
            blended = (np.add(gray[region, :, None], palette[active], dtype=np.uint16) >> 1).astype(np.uint8)
            np.copyto(frame[region], blended, where=(active != 0)[:, :, None])


def _write_heatmap(path, packed_paths, work, inputs, shape, fps, palette, *, native, streaming_decode):
    from .media import close_ffmpeg_writer, ffmpeg_rawvideo_writer
    from .outputs import ffmpeg_h264_rgb_writer

    source_shape = tuple(inputs["source_shape_t_y_x"])
    frames, height, width = shape
    plane_size = height * width
    packed_plane_size = (plane_size + 7) // 8
    with ExitStack() as stack:
        masks = [stack.enter_context(filename.open("rb")) for filename in packed_paths]
        if any(os.fstat(stream.fileno()).st_size != frames * packed_plane_size for stream in masks):
            raise ValueError("A staged confidence heatmap mask has the wrong size")
        source = stack.enter_context(_open_gray_frames(
            work, inputs, source_shape, streaming_decode=streaming_decode,
        ))
        frame = np.empty((height, width, 3), dtype=np.uint8)
        counts = np.empty((height, width), dtype=np.uint8)
        if native:
            writer = ffmpeg_rawvideo_writer(
                path, width, height, fps, pix_fmt_in="rgb24", pix_fmt_out="bgr0",
            )
        else:
            writer = ffmpeg_h264_rgb_writer(path, width, height, fps)
        try:
            assert writer.stdin is not None
            for gray in _gray_frames_for_grid(source, source_shape, shape):
                counts.fill(0)
                for stream in masks:
                    packed = _read_exact(stream, packed_plane_size)
                    bits = np.unpackbits(packed, bitorder="little", count=plane_size).reshape(height, width)
                    np.add(counts, bits, out=counts)
                _render_heatmap(frame, gray, counts, palette)
                writer.stdin.write(memoryview(frame).cast("B"))
        except BaseException:
            if writer.poll() is None:
                try:
                    writer.kill()
                except OSError:
                    pass
            try:
                close_ffmpeg_writer(writer)
            except Exception:
                pass
            raise
        else:
            close_ffmpeg_writer(writer)


def publish_sweep_heatmaps(root, work, inputs, grids, cutoffs, incomplete_cutoffs, *, requested_cutoffs, streaming_decode):
    """Render verified branch masks and publish the entire video set atomically."""
    if not grids:
        return []
    source_shape = tuple(inputs["source_shape_t_y_x"])
    palette = _heatmap_palette(len(cutoffs))
    destination = root / HEATMAP_DIRECTORY
    records = []
    with _staged_sweep_directory(destination, prefix=".heatmaps.atomic-result-") as stage:
        for grid, dimensions in grids.items():
            shapes_valid = (isinstance(dimensions, list) and len(dimensions) == 3
                            and all(type(value) is int and value > 0 for value in dimensions))
            if not shapes_valid:
                raise ValueError(f"Invalid dimensions for confidence heatmap grid {grid!r}")
            shape = tuple(dimensions)
            native = grid == "native"
            if native and shape != source_shape:
                raise ValueError("The native confidence heatmap differs from the source geometry")
            paths = [_packed_path(work / "heatmap-input", cutoff, grid) for cutoff in cutoffs]
            name = f"{grid}.{'mkv' if native else 'mp4'}"
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            fps = max(1e-6, float(inputs["fps"]) * shape[0] / source_shape[0])
            _write_heatmap(path, paths, work, inputs, shape, fps, palette,
                           native=native, streaming_decode=streaming_decode)
            with path.open("rb+") as stream:
                os.fsync(stream.fileno())
            _fsync_parent_directory(path.parent)
            records.append({
                "grid": grid, "path": (destination / name).relative_to(root).as_posix(),
                "shape_t_y_x": list(shape), "fps": fps,
                "cutoffs_hundredths": list(cutoffs), "incomplete_cutoffs_hundredths": list(incomplete_cutoffs),
                "requested_cutoffs_hundredths": list(requested_cutoffs),
                "status": "complete" if not incomplete_cutoffs and cutoffs == requested_cutoffs else "incomplete",
                "semantics": "Count of final cutoff masks containing each voxel",
                "palette_rgb": [None, *palette[1:].tolist()], "overlay_opacity": 0.5,
            })
        _fsync_parent_directory(stage)
    return records
