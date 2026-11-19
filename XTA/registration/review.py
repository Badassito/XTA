"""Display every fixed-grid slice beside its registered moving image and red/cyan overlay."""

from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from ..file_hashing import sha256_file
from .packet import load_packet
from .pairs import checked_registered_video
from .video import _probe_stream, _publish_new, _read_exact, _stat_identity, probe_video


def _decoder_command(path, width, height, selection=None):
    filters = ([selection] if selection else []) + [f"scale={width}:{height}:flags=area"]
    return [
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
        str(path),
        "-map",
        "0:v:0",
        "-filter_threads",
        "1",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "gray",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _display_window(path, shape, supplied, log):
    if supplied is not None:
        values = np.asarray(supplied, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)) or values[1] <= values[0]:
            raise ValueError("Display windows need finite increasing low and high values")
        return values.tolist()
    indexes = np.unique(np.linspace(0, shape[0] - 1, min(7, shape[0]), dtype=int))
    selection = "select='" + "+".join(f"eq(n,{z})" for z in indexes) + "'"
    data = subprocess.run(
        _decoder_command(path, 96, 96, selection), stdout=subprocess.PIPE, stderr=log, check=True
    ).stdout
    if len(data) != len(indexes) * 96 * 96:
        raise IOError("Display sampling produced an unexpected number of frames")
    pixels = np.frombuffer(data, np.uint8)
    values = pixels[pixels > 0]
    if not len(values):
        return [0.0, 255.0]
    baseline = int(np.argmax(np.bincount(values, minlength=256)))
    return [float(max(0, baseline - 3)), float(max(baseline + 6, np.percentile(values, 99.8)))]


def _compose(raw, shape, windows, z, depth):
    height, width = shape
    images = [
        np.rint(
            np.clip(
                (np.frombuffer(buf, np.uint8).reshape(shape).astype(float) - low)
                * 255
                / (high - low),
                0,
                255,
            )
        ).astype(np.uint8)
        for buf, (low, high) in zip(raw, windows)
    ]
    a, b = images
    frame = np.zeros((height + 68 + (height % 2), width * 3, 3), np.uint8)
    font = min(0.6, width / 800)
    for i, (image, title) in enumerate(
        zip(
            [a[..., None], b[..., None], np.stack((a, b, b), -1)],
            ["Fixed reference", "Moving on fixed grid", "Fixed red / moving cyan"],
        )
    ):
        frame[32 : 32 + height, i * width : (i + 1) * width] = image
        cv2.putText(
            frame,
            title,
            (i * width + 5, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Reference slice {z + 1} / {depth} | Every native slice",
        (5, height + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        font,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return frame


def render_review(
    packet,
    fixed,
    registered_video,
    output,
    *,
    fixed_window=None,
    moving_window=None,
    panel_width=480,
    fps=30,
    registered_metadata=None,
):
    """Write a full-depth MP4 for moving intensities resampled onto the fixed grid.

    Each original fixed Z index appears exactly once in order. Only XY is
    downsampled. The packet or registered export sidecar must establish the
    registered movie's exact sampling map and content identity.
    """
    if (
        isinstance(panel_width, bool)
        or not isinstance(panel_width, int)
        or panel_width < 2
        or panel_width % 2
    ):
        raise ValueError("Panel width must be a positive even integer")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Review playback FPS must be finite and positive")
    packet = load_packet(packet) if isinstance(packet, (str, Path)) else packet
    fixed, registered_video, output = Path(fixed), Path(registered_video), Path(output)
    sidecar = output.with_suffix(".review.json")
    if output.suffix.lower() != ".mp4":
        raise ValueError("The visual review output must use an MP4 filename")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output if output.exists() else sidecar)
    checked = checked_registered_video(
        packet, registered_video, registered_metadata=registered_metadata
    )
    fixed_info, registered_info = probe_video(fixed), checked["video_info"]
    shape = packet.shape("fixed")
    if (
        fixed_info["sha256"] != packet.metadata["fixed"]["sha256"]
        or tuple(fixed_info["shape_zyx"]) != shape
    ):
        raise ValueError("Fixed reference identity or dimensions disagree with the packet")
    depth, native_height, native_width = shape
    height = max(1, round(panel_width * native_height / native_width))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".registration_review_", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        movie = stage / "review.mp4"
        with (stage / "ffmpeg.log").open("wb") as log:
            windows = [
                _display_window(path, shape, requested, log)
                for path, requested in [(fixed, fixed_window), (registered_video, moving_window)]
            ]
            first_shape = _compose(
                [bytes(height * panel_width)] * 2, (height, panel_width), windows, 0, depth
            ).shape
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{first_shape[1]}x{first_shape[0]}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(movie),
            ]
            with ExitStack() as stack:
                decoders = [
                    stack.enter_context(
                        subprocess.Popen(
                            _decoder_command(path, panel_width, height),
                            stdout=subprocess.PIPE,
                            stderr=log,
                        )
                    )
                    for path in (fixed, registered_video)
                ]
                encoder = stack.enter_context(
                    subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                )
                try:
                    for z in range(depth):
                        raw = [_read_exact(proc.stdout, height * panel_width) for proc in decoders]
                        encoder.stdin.write(
                            _compose(raw, (height, panel_width), windows, z, depth).tobytes()
                        )
                    for proc in decoders:
                        if proc.stdout.read(1) or proc.wait() != 0:
                            raise IOError("Review source decode failed or had unexpected frames")
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise IOError("Review movie encoding failed")
                finally:
                    for proc in [encoder, *decoders]:
                        if proc.poll() is None:
                            proc.kill()
            for path, info in [(fixed, fixed_info), (registered_video, registered_info)]:
                if _stat_identity(path) != info["_source_stat"]:
                    raise ValueError("A review source changed during rendering")
            written = _probe_stream(movie, "-count_frames")
            if int(written.get("nb_read_frames", 0)) != depth or [
                int(written["height"]),
                int(written["width"]),
            ] != list(first_shape[:2]):
                raise ValueError("Review output dimensions or frame count are incorrect")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(movie),
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=log,
                check=True,
            )
        summary = dict(
            schema="xta.registration.review",
            schema_version=1,
            pair=packet.metadata["pair"],
            fixed_filename=fixed.name,
            fixed_sha256=fixed_info["sha256"],
            registered_filename=registered_video.name,
            registered_sha256=registered_info["sha256"],
            output_filename=output.name,
            output_sha256=sha256_file(movie),
            output_bytes=movie.stat().st_size,
            frames=depth,
            fps=float(fps),
            native_z_stride=1,
            frame_labels="One-based fixed slice numbers",
            output_shape_yxc=list(first_shape),
            display_windows=dict(fixed=windows[0], moving=windows[1]),
            display_downsampling="XY area filtering only",
            full_decode_verified=True,
        )
        metadata_path = stage / "review.json"
        metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _publish_new(movie, output)
        try:
            _publish_new(metadata_path, sidecar)
        except BaseException:
            if output.is_file() and sha256_file(output) == summary["output_sha256"]:
                output.unlink()
            raise
    return summary
