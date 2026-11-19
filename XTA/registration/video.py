"""Bounded movie decoding and native-grid volume resampling for registration.

Frames are Z slices in presentation order. Coordinates refer to zero-based
voxel centers. Playback FPS is recorded independently of spatial sampling.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser, Error as ConfigError
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy import ndimage

from ..file_hashing import sha256_file


def _probe_stream(path, count_option=None):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_option is not None:
        command.append(count_option)
    command.extend(["-show_streams", "-of", "json", str(path)])
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", check=True
        )
        if completed.stderr.strip():
            raise ValueError(completed.stderr.strip()[:1000])
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
            raise ValueError("no video stream")
        return streams[0]
    except (subprocess.CalledProcessError, ValueError) as exc:
        detail = (
            str(exc.stderr or "").strip()[:1000]
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise ValueError(f"Could not read registration movie {path}: {detail}") from exc


def _stat_identity(path):
    stat = Path(path).stat()
    return dict(
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def _frame_rate(stream):
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            value = float(Fraction(stream.get(key, "0/1")))
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _spacing(value):
    spacing = np.asarray(value, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Voxel spacing must contain three finite positive XYZ values")
    return spacing.tolist()


def probe_video(path, *, header_path=None, spacing_xyz=None, include_hash=True):
    """Read exact frame bounds and optional reconstruction-header geometry.

    A header's scalar ``Pixel Size`` is isotropic. The header must agree with
    all three decoded dimensions. Explicit spacing may supply anisotropic
    sampling when no header is provided. Length units remain unspecified.
    """
    path = Path(path)
    initial = _stat_identity(path)
    stream = _probe_stream(path)
    if stream.get("pix_fmt") != "gray":
        raise ValueError("Registration movies must contain native eight-bit grayscale pixels")
    if stream.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A"):
        raise ValueError(
            "Movie pixel aspect ratio must be 1:1; provide spatial geometry through voxel spacing"
        )
    if "rotate" in stream.get("tags", {}) or any(
        "rotation" in item or "displaymatrix" in item for item in stream.get("side_data_list", [])
    ):
        raise ValueError("Display rotation metadata must be resolved before volume registration")
    # FFV1 has one frame per packet. Other codecs require decoded-frame
    # counting, since packets may contain multiple frames or only part of one.
    # Duration and nominal frame rate never determine a stack's Z extent.
    ffv1 = stream.get("codec_name") == "ffv1"
    counted = _probe_stream(path, "-count_packets" if ffv1 else "-count_frames")
    count_key = "nb_read_packets" if ffv1 else "nb_read_frames"
    try:
        shape = [int(counted[count_key]), int(stream["height"]), int(stream["width"])]
        if min(shape) <= 0:
            raise ValueError("empty dimensions")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Movie has no positive exact frame count or valid dimensions: {path}"
        ) from exc
    spacing = _spacing(spacing_xyz) if spacing_xyz is not None else None
    result = dict(
        filename=path.name,
        shape_zyx=shape,
        spacing_xyz=spacing,
        spacing_units="unspecified",
        fps=_frame_rate(stream),
        codec=stream.get("codec_name"),
        pixel_format="gray",
        bytes=initial["bytes"],
        frame_count_method="ffv1_packet_count" if ffv1 else "decoded_frame_count",
    )
    if header_path is not None:
        try:
            header = ConfigParser()
            with Path(header_path).open(encoding="utf-8-sig") as handle:
                header.read_file(handle)
            image = header["Image Info"]
            header_shape = [
                image.getint(key) for key in ("Images Taken", "Image Height", "Image Width")
            ]
            declared = _spacing([image.getfloat("Pixel Size")] * 3)
        except (ConfigError, KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid reconstruction header {header_path}: {exc}") from exc
        if header_shape != shape:
            raise ValueError("Reconstruction header dimensions disagree with the movie")
        if spacing is not None and not np.allclose(spacing, declared, rtol=1e-9, atol=0):
            raise ValueError("Explicit spacing disagrees with the reconstruction header")
        result.update(
            spacing_xyz=declared,
            header_filename=Path(header_path).name,
            header_sha256=sha256_file(header_path),
        )
    if include_hash:
        result["sha256"] = sha256_file(path)
    if initial != _stat_identity(path):
        raise ValueError("Source movie changed while its metadata was being read")
    result["_source_stat"] = initial
    return result


@contextmanager
def decoded_volume(path, work_dir, *, info=None, threads=4):
    """Decode a movie to a temporary read-only uint8 ZYX memmap.

    The context owns only its unique temporary directory. Exiting closes the
    mapping and removes the decoded cache, including after an exception.
    """
    path, work_dir = Path(path), Path(work_dir)
    if threads < 1:
        raise ValueError("Decode threads must be positive")
    info = probe_video(path, include_hash=False) if info is None else info
    before = _stat_identity(path)
    expected_stat = info.get("_source_stat")
    if expected_stat is not None:
        if expected_stat != before:
            raise ValueError("Source movie changed since it was probed")
    elif info.get("sha256"):
        if sha256_file(path) != info["sha256"] or _stat_identity(path) != before:
            raise ValueError("Source movie identity disagrees with its decoded input metadata")
    else:
        raise ValueError(
            "Decoded input metadata needs the original probe identity or source SHA256"
        )
    shape = tuple(int(x) for x in info["shape_zyx"])
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError("Decoded shape must contain three positive dimensions")
    expected = math.prod(shape)
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work_dir).free < expected:
        raise OSError(f"Decoding requires {expected} free bytes in {work_dir}")
    with tempfile.TemporaryDirectory(prefix="registration_decode_", dir=work_dir) as temporary:
        raw = Path(temporary) / "volume.gray"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-nostdin",
                "-noautorotate",
                "-threads",
                str(threads),
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "gray",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-n",
                str(raw),
            ],
            check=True,
        )
        if raw.stat().st_size != expected:
            raise ValueError("Decoded frame count or geometry disagrees with the probed movie")
        if before != _stat_identity(path):
            raise ValueError("Source movie changed during decoding")
        volume = np.memmap(raw, np.uint8, "r", shape=shape)
        try:
            yield volume
        finally:
            volume._mmap.close()


def prepare_pyramid(volume, step):
    """Return a low-pass-filtered uint8 ZYX grid sampled at native origin zero.

    ``step`` must be a power of two. Only reduced XY planes and one output
    volume are held in memory, so native 18 GB input stacks can remain mapped.
    """
    import cv2

    if isinstance(step, bool) or int(step) != step or step < 1 or int(step) & (int(step) - 1):
        raise ValueError("Pyramid step must be a positive power of two")
    if volume.ndim != 3 or volume.dtype != np.uint8 or min(volume.shape) < 1:
        raise ValueError("Pyramid input must be a nonempty uint8 ZYX volume")
    step = int(step)
    if step == 1:
        return np.asarray(volume)
    shape = tuple((n + step - 1) // step for n in volume.shape)
    output = np.empty(shape, np.uint8)
    kernel = np.array([1.0])
    stride = 1
    while stride < step:
        taps = np.zeros(4 * stride + 1)
        taps[::stride] = np.array([1, 4, 6, 4, 1]) / 16
        kernel = np.convolve(kernel, taps)
        stride *= 2
    radius = (len(kernel) - 1) // 2
    reduced = {}
    for oz in range(shape[0]):
        native_z = oz * step
        reduced = {z: a for z, a in reduced.items() if z >= native_z - radius}
        plane = np.zeros(shape[1:], np.float32)
        for offset, weight in enumerate(kernel):
            z = cv2.borderInterpolate(
                native_z + offset - radius, volume.shape[0], cv2.BORDER_REFLECT_101
            )
            if z not in reduced:
                small = volume[z]
                factor = step
                while factor > 1:
                    small = cv2.pyrDown(small)
                    factor //= 2
                reduced[z] = small
            plane += float(weight) * reduced[z]
        output[oz] = np.rint(plane)
    return output


def observed_field(volume, *, black_exterior=False):
    """Return a conservative native YX acquisition mask.

    With ``black_exterior=True``, several planes locate the gray-to-black
    acquisition boundary. Internal dark holes are retained as observed. With
    ordinary imagery the full rectangular array is considered observed.
    """
    if volume.ndim != 3 or volume.dtype != np.uint8 or min(volume.shape) <= 0:
        raise ValueError("Observed-field input must be a nonempty uint8 ZYX volume")
    if not black_exterior:
        return np.ones(volume.shape[1:], dtype=bool)
    masks = []
    for z in sorted({0, volume.shape[0] // 2, volume.shape[0] - 1}):
        plane = np.asarray(volume[z])
        positive = plane[plane > 0]
        if not positive.size:
            continue
        # Half the dominant nonzero gray level lies between the black exterior
        # and the uniform reconstruction background for these slice stacks.
        baseline = int(np.argmax(np.bincount(positive.astype(np.uint8), minlength=256)[1:])) + 1
        regions, count = ndimage.label(plane > baseline / 2)
        if count:
            sizes = np.bincount(regions.ravel())
            sizes[0] = 0
            masks.append(ndimage.binary_fill_holes(regions == sizes.argmax()))
    if not masks:
        raise ValueError("No observed reconstruction field was found")
    result = np.logical_and.reduce(masks)
    if not result.any():
        raise ValueError("Sampled acquisition fields have no stable common support")
    return result


def resample_slice(volume, matrix_xyz, z, output_yx):
    """Sample input intensities at output→input native XYZ coordinates."""
    matrix = np.asarray(matrix_xyz, float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1])
    ):
        raise ValueError("Resampling requires a finite affine 4 by 4 matrix")
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError("Intensity input must be uint8 ZYX")
    height, width = (int(x) for x in output_yx)
    if min(height, width) <= 0 or int(z) != z or z < 0:
        raise ValueError("Output geometry must be positive and Z must be a nonnegative integer")
    linear = matrix[:3, :3][::-1, ::-1]
    offset = matrix[:3, 3][::-1] + linear[:, 0] * z
    return ndimage.affine_transform(
        volume,
        linear,
        offset,
        output_shape=(1, height, width),
        output=np.uint8,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )[0]


def _read_exact(stream, count):
    chunks = bytearray(count)
    view = memoryview(chunks)
    filled = 0
    while filled < count:
        got = stream.readinto(view[filled:])
        if not got:
            raise EOFError("Movie decode ended before a requested verification frame")
        filled += got
    return chunks


def _verify_samples(path, volume, matrix, shape, indexes, log):
    expression = "+".join(f"eq(n,{z})" for z in indexes)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-nostdin",
        "-threads",
        "2",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-filter_threads",
        "1",
        "-vf",
        f"select='{expression}'",
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
    matches = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log) as process:
        try:
            for z in indexes:
                raw = _read_exact(process.stdout, shape[1] * shape[2])
                frame = np.frombuffer(raw, np.uint8).reshape(shape[1:])
                if not np.array_equal(frame, resample_slice(volume, matrix, z, shape[1:])):
                    raise ValueError(f"Registered movie frame {z} differs from direct resampling")
                matches.append(
                    dict(frame_zero_based=z, raw_pixel_sha256=hashlib.sha256(raw).hexdigest())
                )
            if process.stdout.read(1) or process.wait() != 0:
                raise ValueError("Registered movie verification decode failed")
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
    return matches


def _publish_new(source, destination):
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)


def export_registered_video(
    moving_path,
    packet,
    output,
    work_dir,
    *,
    direction="moving-to-fixed",
    fps=None,
    workers=4,
    threads=3,
    verify_samples=True,
    progress=None,
):
    """Write every destination-grid frame as lossless grayscale FFV1.

    ``moving_path`` is the input intensity movie. For reverse resampling use
    ``direction='fixed-to-moving'`` and pass the original fixed movie. The
    packet's hashes prevent accidentally applying a map to another scan.
    """
    from .packet import load_packet

    packet = load_packet(packet) if isinstance(packet, (str, Path)) else packet
    if not packet.usable:
        raise ValueError("This registration packet has no validated mapping")
    if direction not in ("moving-to-fixed", "fixed-to-moving"):
        raise ValueError("Direction must be moving-to-fixed or fixed-to-moving")
    if min(workers, threads) < 1:
        raise ValueError("Resampling and encoder thread counts must be positive")
    output = Path(output)
    if output.suffix.lower() != ".mkv":
        raise ValueError("Lossless registered stack output must use the .mkv suffix")
    if output.exists():
        raise FileExistsError(output)
    sidecar = output.with_suffix(".registration.json")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    source_role, destination_role = (
        ("moving", "fixed") if direction == "moving-to-fixed" else ("fixed", "moving")
    )
    source = packet.metadata[source_role]
    target = packet.metadata[destination_role]
    info = probe_video(moving_path)
    if info["sha256"] != source["sha256"] or info["shape_zyx"] != source["shape_zyx"]:
        raise ValueError("Input movie identity or geometry disagrees with the registration packet")
    matrix = packet.matrix("fixed_to_moving" if source_role == "moving" else "moving_to_fixed")
    shape = tuple(target["shape_zyx"])
    playback_value = fps if fps is not None else target.get("fps") or info["fps"]
    if playback_value is None:
        raise ValueError("Movie has no valid playback frame rate; supply an explicit FPS")
    playback = float(playback_value)
    if not math.isfinite(playback) or playback <= 0:
        raise ValueError("Playback FPS must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=".registration_export_", dir=output.parent
    ) as temporary:
        partial = Path(temporary) / "registered.mkv"
        with (
            decoded_volume(moving_path, work_dir, info=info, threads=threads) as volume,
            (Path(temporary) / "ffmpeg.log").open("w+b") as log,
        ):
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "gray",
                "-video_size",
                f"{shape[2]}x{shape[1]}",
                "-framerate",
                str(playback),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-g",
                "1",
                "-slices",
                "16",
                "-slicecrc",
                "1",
                "-threads",
                str(threads),
                "-pix_fmt",
                "gray",
                "-color_range",
                "pc",
                str(partial),
            ]
            with subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
            ) as encoder:
                try:
                    with ThreadPoolExecutor(workers) as pool:
                        pending = deque()
                        next_z = 0
                        while next_z < min(shape[0], workers + 1):
                            pending.append(
                                pool.submit(resample_slice, volume, matrix, next_z, shape[1:])
                            )
                            next_z += 1
                        for z in range(shape[0]):
                            frame = pending.popleft().result()
                            encoder.stdin.write(frame.tobytes())
                            if next_z < shape[0]:
                                pending.append(
                                    pool.submit(resample_slice, volume, matrix, next_z, shape[1:])
                                )
                                next_z += 1
                            if progress is not None:
                                progress(z + 1, shape[0])
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError("Registered movie encoding failed")
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
            written = probe_video(partial, include_hash=False)
            if written["shape_zyx"] != list(shape) or written["codec"] != "ffv1":
                raise ValueError(
                    "Registered movie output dimensions, frame count or codec are incorrect"
                )
            indexes = sorted(
                set(np.linspace(0, shape[0] - 1, min(shape[0], 9), dtype=int).tolist())
            )
            samples = (
                _verify_samples(partial, volume, matrix, shape, indexes, log)
                if verify_samples
                else []
            )
            result = dict(
                output_filename=output.name,
                source_filename=Path(moving_path).name,
                source_sha256=info["sha256"],
                source_role=source_role,
                destination_role=destination_role,
                shape_zyx=list(shape),
                spacing_xyz=target["spacing_xyz"],
                spacing_units=target.get("spacing_units", "unspecified"),
                output_to_input_xyz=matrix.tolist(),
                interpolation="trilinear",
                dtype="uint8",
                outside_input_array_value=0,
                codec="ffv1",
                fps=playback,
                fps_is_spatial_spacing=False,
                frames=shape[0],
                exact_pixel_samples=samples,
                output_sha256=sha256_file(partial),
                output_bytes=partial.stat().st_size,
                elapsed_seconds=time.monotonic() - started,
            )
            result.update(
                packet_fixed_sha256=packet.metadata["fixed"]["sha256"],
                packet_moving_sha256=packet.metadata["moving"]["sha256"],
            )
            staged_sidecar = Path(temporary) / "registered.registration.json"
            staged_sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
            # Each publication refuses replacement. If the second publication
            # fails, remove only the unchanged file published by this call.
            _publish_new(partial, output)
            published = output.stat()
            try:
                _publish_new(staged_sidecar, sidecar)
            except BaseException:
                current = output.stat() if output.exists() else None
                if current is not None and (
                    current.st_ino,
                    current.st_dev,
                    current.st_size,
                    current.st_mtime_ns,
                ) == (published.st_ino, published.st_dev, published.st_size, published.st_mtime_ns):
                    output.unlink()
                raise
    return result
