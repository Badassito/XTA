"""PTA source image/video readers and annotation input I/O.

This module owns grayscale normalization, bounded input decoding, and NRRD/YOLO
annotation parsing. It does not import the PTA orchestration module or own its
source/prepared-volume class identities.
"""

from __future__ import annotations

import json
import math
import queue
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

from . import media
from .runtime import close_memmap_array, runtime_telemetry, runtime_telemetry_phase
from .pta_workers import ArrayAllocator, SharedBlock, WarningLog, bind_current_thread_to_cpus


GIB = 1024 ** 3
YoloPolygon = Tuple[Tuple[float, float], ...]
YoloPolygons = Tuple[YoloPolygon, ...]


class MaskPlaneSummary(NamedTuple):
    """Per-t-plane facts of a 0/1 mask, recorded while the NRRD reader wrote it.

    Valid only for that unmodified array: has_foreground equals
    np.any(mask, axis=(1, 2)) and first_foreground equals np.argmax(mask[t])
    (0 for an empty plane).
    """
    has_foreground: np.ndarray
    first_foreground: np.ndarray



def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")



def probe_image_dimensions(path: Path) -> Tuple[int, int]:
    if shutil.which("ffprobe") is not None:
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=10,
            )
            width_text, height_text = proc.stdout.strip().split("x", 1)
            return int(height_text), int(width_text)
        except Exception:
            pass
    gray = _read_gray_image_opencv(path)
    return int(gray.shape[0]), int(gray.shape[1])



def parallel_for_indices(
    count: int,
    func: Callable[[int], None],
    *,
    workers: int,
    desc: str,
    show_progress: bool = True,
    worker_cpu_order: Sequence[int] = (),
) -> None:
    total = max(0, int(count))
    if total <= 0:
        return
    nworkers = max(1, min(int(workers), total))
    if nworkers <= 1:
        iterable: Iterable[int] = tqdm(range(total), desc=desc) if show_progress else range(total)
        for idx in iterable:
            func(int(idx))
        return
    affinity_queue: "queue.SimpleQueue[int]" = queue.SimpleQueue()
    affinity_values = tuple(int(x) for x in worker_cpu_order)
    for worker_index in range(nworkers):
        if affinity_values:
            affinity_queue.put(int(affinity_values[worker_index % len(affinity_values)]))

    def _initializer() -> None:
        if affinity_values:
            bind_current_thread_to_cpus((affinity_queue.get(),))

    with ThreadPoolExecutor(max_workers=nworkers, initializer=_initializer) as executor:
        futures = [executor.submit(func, int(i)) for i in range(total)]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        else:
            for fut in as_completed(futures):
                fut.result()



def to_gray8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(np.clip(np.rint(arr.astype(np.float32) / 257.0), 0, 255).astype(np.uint8))
    amin = float(np.nanmin(arr)) if arr.size else 0.0
    amax = float(np.nanmax(arr)) if arr.size else 0.0
    if not math.isfinite(amin) or not math.isfinite(amax) or amax <= amin:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = (arr.astype(np.float32) - amin) * (255.0 / (amax - amin))
    return np.ascontiguousarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))



def _release_failed_allocator_blocks(function):
    """Own new shared allocations until their complete input/preparation result returns."""
    from functools import wraps

    @wraps(function)
    def run(first, *args, allocator=None, **kwargs):
        created = []
        def allocate(shape):
            array, block = allocator(shape)
            if (block is not None and block is not getattr(first, 'volume_block', None)
                    and all(block is not existing for existing in created)):
                created.append(block)
            return array, block
        try:
            return function(first, *args, allocator=allocate if allocator is not None else None, **kwargs)
        except BaseException:
            for block in created:
                block.release()
            raise
    return run



def _polygon_area_xy(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(0.5 * sum(
        float(points[i][0]) * float(points[(i + 1) % len(points)][1])
        - float(points[(i + 1) % len(points)][0]) * float(points[i][1])
        for i in range(len(points))
    ))



def ffprobe_info(video_path: Path) -> Dict[str, object]:
    _require_bin("ffprobe")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,start_time:format=duration",
        "-of", "json", str(video_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=True,
    )
    info = json.loads(proc.stdout)
    if not info.get("streams"):
        raise RuntimeError(f"No video stream found in {video_path}")
    st = info["streams"][0]

    def parse_ratio(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        n, d = s.split("/")
        return float(n) / float(d) if int(d) else 0.0

    width = int(st["width"])
    height = int(st["height"])
    fps = parse_ratio(str(st.get("avg_frame_rate", "0/0"))) or parse_ratio(str(st.get("r_frame_rate", "0/0"))) or 30.0
    nf = st.get("nb_frames")
    if nf is None or str(nf) in {"", "N/A"}:
        # Matroska has no frame count. TTA's exact tail-timestamp count avoids
        # reading the whole file; YOLO_TTA_FRAME_COUNT_FULL_SCAN=1 disables it.
        from .media import _tail_frame_count_enabled, _tail_timestamp_frame_count
        if _tail_frame_count_enabled():
            nf = _tail_timestamp_frame_count(video_path, st, info.get("format") or {}, fps)
    if nf is None or str(nf) in {"", "N/A"}:
        cmd2 = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "json", str(video_path),
        ]
        with runtime_telemetry().span('startup.source_probe.packet_count'):
            proc2 = subprocess.run(
                cmd2, capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=True,
            )
        info2 = json.loads(proc2.stdout)
        nf = info2["streams"][0].get("nb_read_packets")
    if nf is None or str(nf) in {"", "N/A"}:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    return {"width": width, "height": height, "fps": fps, "num_frames": int(nf)}



@runtime_telemetry_phase('pta.input.decode_video')
@_release_failed_allocator_blocks
def decode_video_gray8_to_memory(video_path: Path, *, warnings: WarningLog, allocator: Optional[ArrayAllocator] = None,
                                 probe_info: Optional[Mapping[str, object]] = None) -> Tuple[np.ndarray, float, Optional[SharedBlock]]:
    _require_bin("ffmpeg")
    info = ffprobe_info(video_path) if probe_info is None else probe_info
    w = int(info["width"])
    h = int(info["height"])
    t = int(info["num_frames"])
    fps = float(info["fps"])
    if allocator is not None:
        arr, block = allocator((t, h, w))
    else:
        arr = np.empty((t, h, w), dtype=np.uint8)
        block = None
    frame_bytes = int(w) * int(h)
    chunk_frames = max(1, min(128, (256 * 1024 * 1024) // max(1, frame_bytes)))
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]
    try:
        with memoryview(arr).cast("B") as raw:
            proc = media._spawn_decode_process(cmd)
            with media._synchronous_decode_stdout(proc) as stdout:
                with tqdm(total=t, desc="Decoding video to in-memory gray8 volume") as pbar:
                    for start in range(0, t, chunk_frames):
                        nframes = min(chunk_frames, t - start)
                        need = nframes * frame_bytes
                        offset = start * frame_bytes
                        with raw[offset:offset + need] as view:
                            filled = 0
                            while filled < need:
                                with view[filled:] as pending:
                                    nread = stdout.readinto(pending)
                                if nread is None or nread <= 0:
                                    raise RuntimeError(f"Unexpected EOF while decoding {video_path} near frame {start + 1}")
                                filled += nread
                        pbar.update(nframes)
    except BaseException:
        if block is None:
            close_memmap_array(arr)
        arr = None  # Drop the allocation before the owning decorator releases shared memory.
        raise
    warnings.add("input_volume_loaded_in_memory", f"{arr.shape}, {arr.nbytes / GIB:.2f} GiB")
    return arr, fps, block



def contiguous_start_0_or_1(indices: Sequence[int]) -> Optional[int]:
    idxs = sorted(int(x) for x in indices)
    n = len(idxs)
    if n == 0:
        return None
    for start in (0, 1):
        if idxs == list(range(start, start + n)):
            return int(start)
    return None



def describe_index_problem(indices: Sequence[int]) -> str:
    idxs = sorted(int(x) for x in indices)
    if not idxs:
        return "no indexed files"
    if len(idxs) <= 24:
        return f"indices={idxs}"
    return f"count={len(idxs)}, first={idxs[:8]}, last={idxs[-8:]}"



def read_yolo_polygons_normalized(path: Path, *, warnings: WarningLog) -> YoloPolygons:
    """Parse one YOLO segmentation file without allocating an image mask."""
    try:
        text = Path(path).read_text().strip()
    except Exception as exc:
        warnings.add("label_read_error", f"{path}: {exc}")
        return tuple()
    if not text:
        return tuple()
    polygons: List[YoloPolygon] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            warnings.add("invalid_label_line", f"{path.name}:{line_no}")
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except Exception:
            warnings.add("invalid_label_coordinate", f"{path.name}:{line_no}")
            continue
        points: List[Tuple[float, float]] = []
        for x_n, y_n in zip(coords[0::2], coords[1::2]):
            if not math.isfinite(x_n) or not math.isfinite(y_n):
                points = []
                break
            points.append((min(1.0, max(0.0, float(x_n))), min(1.0, max(0.0, float(y_n)))))
        if len(points) < 3 or len(set(points)) < 3 or _polygon_area_xy(points) <= 1e-12:
            warnings.add("degenerate_input_polygon", f"{path.name}:{line_no}")
            continue
        polygons.append(tuple(points))
    return tuple(polygons)



def _read_gray_image_opencv(path: Path) -> np.ndarray:
    # Request luminance/gray at decode time.  This avoids materializing the
    # source RGB array and the subsequent cvtColor pass for the target command.
    img = cv2.imread(str(path), int(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return to_gray8(img)



def _nvimgcodec_decoder_for_device(nvimgcodec: object, *, device_id: int, cpu_threads: int, batch_size: int) -> object:
    backends: List[object] = []
    backend_type = getattr(nvimgcodec, "Backend", None)
    backend_kind = getattr(nvimgcodec, "BackendKind", None)
    if backend_type is not None and backend_kind is not None:
        for name in ("HW_GPU_ONLY", "GPU_ONLY", "HYBRID_CPU_GPU"):
            kind = getattr(backend_kind, name, None)
            if kind is not None:
                try:
                    backends.append(backend_type(kind))
                except TypeError:
                    try:
                        backends.append(backend_type(kind=kind))
                    except Exception:
                        pass
    options = (
        f":num_cuda_streams=4 "
        f"nvjpeg_cuda_decoder:hybrid_huffman_threshold=0 "
        f"nvjpeg_cuda_decoder:preallocate_buffers=1 "
        f"nvjpeg_hw_decoder:preallocate_batch_size={max(1, int(batch_size))}"
    )
    kwargs: Dict[str, object] = {
        "device_id": int(device_id),
        "max_num_cpu_threads": max(1, int(cpu_threads)),
        "options": options,
    }
    if backends:
        kwargs["backends"] = backends
    return getattr(nvimgcodec, "Decoder")(**kwargs)



def _nvimgcodec_gray_params(nvimgcodec: object) -> object:
    params_type = getattr(nvimgcodec, "DecodeParams")
    color_spec_type = getattr(nvimgcodec, "ColorSpec", None)
    gray = getattr(color_spec_type, "GRAY", None) if color_spec_type is not None else None
    if gray is not None:
        return params_type(color_spec=gray, allow_any_depth=False)
    return params_type(allow_any_depth=False)



def _decoded_nvimgcodec_batch_to_arrays(decoded: object, expected: int) -> List[np.ndarray]:
    if isinstance(decoded, (list, tuple)):
        items = list(decoded)
    else:
        try:
            items = list(decoded)  # nvImageCodec batch objects are iterable
        except TypeError:
            host = decoded.cpu() if callable(getattr(decoded, "cpu", None)) else decoded
            batch = np.asarray(host)
            if batch.ndim >= 3 and int(batch.shape[0]) == int(expected):
                items = [batch[i] for i in range(int(expected))]
            else:
                items = [host]
    arrays: List[np.ndarray] = []
    for item in items:
        host = item.cpu() if callable(getattr(item, "cpu", None)) else item
        arr = np.asarray(host)
        if arr.ndim == 3 and int(arr.shape[-1]) == 1:
            arr = arr[:, :, 0]
        arrays.append(to_gray8(arr))
    if len(arrays) != int(expected):
        raise RuntimeError(f"nvImageCodec returned {len(arrays)} image(s) for a batch of {expected}")
    return arrays



def _decode_jpeg_sequence_nvimgcodec(
    images: Sequence[Path],
    volume: np.ndarray,
    *,
    device_ids: Sequence[int],
    cpu_sets: Sequence[Sequence[int]],
    batch_size: int,
    workers: int,
) -> None:
    try:
        from nvidia import nvimgcodec  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "nvJPEG input decoding requires NVIDIA nvImageCodec; install "
            "nvidia-nvimgcodec-cu12[all] or nvidia-nvimgcodec-cu13[all] "
            "to match the active CUDA major version"
        ) from exc
    if not device_ids:
        raise RuntimeError("nvJPEG input decoding requested but no CUDA-visible GPU was discovered")
    batch = max(1, int(batch_size))
    batches = [list(range(start, min(len(images), start + batch))) for start in range(0, len(images), batch)]
    per_device: List[List[List[int]]] = [[] for _ in device_ids]
    for batch_index, indices in enumerate(batches):
        per_device[batch_index % len(device_ids)].append(indices)

    def _decode_device(slot: int) -> None:
        cpus = tuple(int(x) for x in (cpu_sets[slot] if slot < len(cpu_sets) else ()))
        bind_current_thread_to_cpus(cpus)
        cpu_threads = max(1, min(len(cpus) if cpus else int(workers), 8))
        decoder = _nvimgcodec_decoder_for_device(
            nvimgcodec,
            device_id=int(device_ids[slot]),
            cpu_threads=cpu_threads,
            batch_size=batch,
        )
        params = _nvimgcodec_gray_params(nvimgcodec)
        for indices in per_device[slot]:
            paths = [str(images[i]) for i in indices]
            decoded = decoder.read(paths, params=params)
            arrays = _decoded_nvimgcodec_batch_to_arrays(decoded, len(indices))
            for frame_index, gray in zip(indices, arrays):
                if gray.shape != tuple(int(x) for x in volume.shape[1:]):
                    raise ValueError(
                        f"Image dimension mismatch: {images[frame_index]} has {gray.shape}, "
                        f"expected {tuple(volume.shape[1:])}"
                    )
                volume[frame_index] = gray

    with ThreadPoolExecutor(max_workers=len(device_ids), thread_name_prefix="nvjpeg-decode") as executor:
        futures = [executor.submit(_decode_device, slot) for slot in range(len(device_ids)) if per_device[slot]]
        for future in futures:
            future.result()



@runtime_telemetry_phase('pta.input.decode_sequence')
@_release_failed_allocator_blocks
def load_image_sequence_to_memory(
    images: Sequence[Path],
    *,
    warnings: WarningLog,
    workers: int,
    allocator: Optional[ArrayAllocator] = None,
    jpeg_decode_backend: str = "auto",
    jpeg_batch_size: int = 64,
    jpeg_device_ids: Sequence[int] = (),
    jpeg_cpu_sets: Sequence[Sequence[int]] = (),
    load_cpu_order: Sequence[int] = (),
    required_positions: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Optional[SharedBlock]]:
    if not images:
        raise ValueError("No image sequence frames supplied")
    first_gray = _read_gray_image_opencv(images[0])
    h, w = first_gray.shape
    t = len(images)
    positions = list(range(t)) if required_positions is None else sorted({0, *(int(i) for i in required_positions)})
    if any(i < 0 or i >= t for i in positions):
        raise ValueError('Required image-sequence positions are outside the source volume')
    if allocator is not None:
        volume, block = allocator((t, h, w))
    else:
        volume = np.empty((t, h, w), dtype=np.uint8) if required_positions is None else np.zeros((t, h, w), dtype=np.uint8)
        block = None
    volume[0] = first_gray

    backend = str(jpeg_decode_backend).strip().lower()
    all_jpeg = all(path.suffix.lower() in {".jpg", ".jpeg"} for path in images)
    if backend == "nvjpeg" and not all_jpeg:
        raise ValueError("--jpeg_decode_backend nvjpeg requires every sequence frame to be .jpg/.jpeg")
    try_nvjpeg = all_jpeg and backend in {"auto", "nvjpeg"} and bool(jpeg_device_ids)
    if try_nvjpeg:
        try:
            _decode_jpeg_sequence_nvimgcodec(
                images,
                volume,
                device_ids=jpeg_device_ids,
                cpu_sets=jpeg_cpu_sets,
                batch_size=int(jpeg_batch_size),
                workers=max(1, int(workers)),
            )
            warnings.add(
                "jpeg_sequence_decoded_with_nvjpeg",
                f"frames={len(images)}, GPUs={list(jpeg_device_ids)}, batch_size={int(jpeg_batch_size)}",
            )
            warnings.add("input_volume_loaded_in_memory", f"{volume.shape}, {volume.nbytes / GIB:.2f} GiB")
            return volume, block
        except Exception as exc:
            if backend == "nvjpeg":
                raise
            warnings.add("nvjpeg_fallback_to_opencv", f"{type(exc).__name__}: {exc}")
    elif backend == "nvjpeg":
        raise RuntimeError("--jpeg_decode_backend nvjpeg requires at least one discovered CUDA-visible GPU")

    if required_positions is not None:
        import struct
        # shortcut: unused PNG payloads are not decoded; diagnostics retain full loading.
        selected_positions = set(positions)
        for position, path in enumerate(images):
            if position in selected_positions:
                continue
            with Path(path).open('rb') as source:
                header = source.read(24)
            if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
                    or header[12:16] != b'IHDR'):
                raise ValueError(f'Invalid PNG image header: {path}')
            image_width, image_height = struct.unpack('>II', header[16:24])
            if (image_height, image_width) != (h, w):
                raise ValueError(f'Image dimension mismatch: {path} has {(image_height, image_width)}, expected {(h, w)}')
        warnings.add('partial_sequence_required_frames', f'decoded={len(positions)}/{t}; annotation/channel context retained')

    def _read_image(job: int) -> None:
        i = positions[job]
        if i == 0:
            return
        gray = _read_gray_image_opencv(images[i])
        if gray.shape != (h, w):
            raise ValueError(f"Image dimension mismatch: {images[i]} has {gray.shape}, expected {(h, w)}")
        volume[i] = gray

    parallel_for_indices(
        len(positions),
        _read_image,
        workers=workers,
        desc="Reading image sequence into in-memory gray8 volume",
        worker_cpu_order=load_cpu_order,
    )
    warnings.add("image_sequence_decoded_direct_grayscale", f"frames={len(images)}, backend=opencv")
    warnings.add("input_volume_loaded_in_memory", f"{volume.shape}, {volume.nbytes / GIB:.2f} GiB")
    return volume, block



def nrrd_header_to_internal_permutation(header: Mapping[str, object]) -> Optional[Tuple[int, int, int]]:
    """Infer the array-axis permutation required for internal ``(t,Y,X)``.

    NRRD ``space directions`` contains one physical direction vector per data
    axis.  For an axis-aligned spatial volume, map the data axes representing
    physical Z, Y, and X into the pipeline's t/Y/X order.  This disambiguates
    cubic volumes, for which shape alone cannot distinguish XYZ from TYX.
    """
    raw_directions = header.get("space directions")
    if raw_directions is not None:
        try:
            directions = np.asarray(raw_directions, dtype=np.float64)
            if directions.shape == (3, 3) and np.all(np.isfinite(directions)):
                magnitudes = np.abs(directions)
                dominant = tuple(int(x) for x in np.argmax(magnitudes, axis=1).tolist())
                dominant_values = magnitudes[np.arange(3), np.asarray(dominant, dtype=np.int64)]
                if set(dominant) == {0, 1, 2} and np.all(dominant_values > 0.0):
                    # Physical coordinates are X=0, Y=1, Z/t=2.
                    return (
                        int(dominant.index(2)),
                        int(dominant.index(1)),
                        int(dominant.index(0)),
                    )
        except Exception:
            pass

    # Recognize snapshots written by this script even if a downstream tool
    # removes the space-directions field while preserving content metadata.
    content = str(header.get("content", ""))
    if "exported_axes=(X,Y,t)" in content:
        return (2, 1, 0)
    return None



def _force_fit_nearest_source_indices(source_count, target_count):
    """Endpoint-aligned nearest map for --force geometry reconciliation (within 3 voxels).

    This deliberately keeps ``round(i*(source-1)/(target-1))``: it pins both label
    ends onto the image ends, unlike the center-aligned processing-cube resize."""
    source_count, target_count = int(source_count), int(target_count)
    if target_count <= 1 or source_count <= 1:
        return np.zeros(target_count, dtype=np.intp)
    positions = np.arange(target_count, dtype=np.float64) * (source_count - 1) / (target_count - 1)
    return np.clip(np.rint(positions), 0, source_count - 1).astype(np.intp)


def _nrrd_mask_source_planes(
    shape_tyx, expected_shape_tyx, nrrd_path, *, force=False,
    source_contiguous=True, source_plane_indices=None,
):
    """Validate geometry and map each target frame to one source mask plane."""
    source_t, source_h, source_w = (int(value) for value in shape_tyx)
    target_t, target_h, target_w = (int(value) for value in expected_shape_tyx)
    tolerance = 3 if force else 0
    if abs(source_h - target_h) > tolerance or abs(source_w - target_w) > tolerance:
        raise ValueError(
            f"NRRD segmentation XY shape {(source_h, source_w)} does not match image XY shape "
            f"{(target_h, target_w)} within the {tolerance}-voxel per-axis tolerance: {nrrd_path}"
        )
    if source_plane_indices is not None:
        if len(source_plane_indices) != target_t:
            raise ValueError("NRRD source-plane mapping must contain one index per image frame")
        indices = np.asarray(source_plane_indices, dtype=np.intp)
        if not np.any((indices >= 0) & (indices < source_t)):
            raise ValueError(f"NRRD source-plane mapping has no matched image slices: {nrrd_path}")
        return indices
    if source_t != target_t and (not force or not source_contiguous or abs(source_t - target_t) > 3):
        raise ValueError(
            f"NRRD segmentation depth {source_t} does not match image depth {target_t}: {nrrd_path}. "
            "--force permits XYZ resizing within 3 voxels per axis for contiguous sources; "
            "2D correspondence requires --nrrd_z_start_index to establish a source-plane mapping."
        )
    if source_t == target_t:
        return np.arange(target_t, dtype=np.intp)
    return _force_fit_nearest_source_indices(source_t, target_t)



def _nrrd_mask_orientation(
    header, shape, expected_shape_tyx, nrrd_path, *, force=False,
    source_contiguous=True, source_plane_indices=None,
):
    if len(shape) != 3:
        raise ValueError(f"NRRD segmentation must be 3D, got shape {shape} from {nrrd_path}")
    t_dim, h, w = (int(expected_shape_tyx[0]), int(expected_shape_tyx[1]), int(expected_shape_tyx[2]))
    shape = tuple(int(x) for x in shape)
    permutations = (
        ("already_tyx", (0, 1, 2)),
        ("xyz_to_tyx", (2, 1, 0)),
        ("yxt_to_tyx", (2, 0, 1)),
        ("txy_to_tyx", (0, 2, 1)),
    )
    candidates = [
        (name, permutation) for name, permutation in permutations
        if tuple(shape[axis] for axis in permutation) == (t_dim, h, w)
    ]
    if not candidates and not force and source_plane_indices is None:
        raise ValueError(
            f"NRRD segmentation shape {shape} does not match image volume shape (t,Y,X)=({t_dim},{h},{w}) "
            f"or recognized spatial permutations (X,Y,t)=({w},{h},{t_dim}) / (Y,X,t)=({h},{w},{t_dim})"
        )
    header_permutation = nrrd_header_to_internal_permutation(header)
    if header_permutation is not None:
        internal_shape = tuple(shape[int(axis)] for axis in header_permutation)
        if not force and source_plane_indices is None and internal_shape != (t_dim, h, w):
            raise ValueError(
                f"NRRD header axis directions imply internal shape {internal_shape}, but image volume shape is "
                f"(t,Y,X)=({t_dim},{h},{w}) for {nrrd_path}"
            )
        _nrrd_mask_source_planes(
            internal_shape, expected_shape_tyx, nrrd_path, force=force,
            source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        )
        permutation = tuple(int(x) for x in header_permutation)
        orientation = "header_axes_" + "".join(str(x) for x in permutation) + "_to_tyx"
    else:
        if not candidates and (force or source_plane_indices is not None):
            for name, permutation in permutations:
                internal_shape = tuple(shape[axis] for axis in permutation)
                try:
                    _nrrd_mask_source_planes(
                        internal_shape, expected_shape_tyx, nrrd_path, force=force,
                        source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
                    )
                except ValueError:
                    continue
                candidates.append((name, permutation))
        if not candidates:
            raise ValueError(
                f"NRRD segmentation shape {shape} does not match image volume shape (t,Y,X)=({t_dim},{h},{w}) "
                f"or recognized spatial permutations (X,Y,t)=({w},{h},{t_dim}) / (Y,X,t)=({h},{w},{t_dim})"
            )
        unique_permutations = {tuple(permutation) for _, permutation in candidates}
        if len(unique_permutations) != 1:
            matches = ", ".join(name for name, _ in candidates)
            raise ValueError(
                f"NRRD axis order is ambiguous from shape {shape} ({matches}) and the header has no usable "
                f"space directions/content marker: {nrrd_path}. Add axis-aware NRRD metadata before import."
            )
        orientation, permutation = candidates[0]

    return orientation, permutation



_NRRD_REFERENCE_OFFSET_KEY = 'Segmentation_ReferenceImageExtentOffset'
_NRRD_REFERENCE_EXTENT_KEY = 'Segmentation_ReferenceImageExtent'



def _nrrd_header_integers(header, key, count, nrrd_path):
    raw = header.get(key)
    if raw is None:
        return None
    parts = str(raw).split()
    if len(parts) != count or any(re.fullmatch(r'-?[0-9]+', part) is None for part in parts):
        raise ValueError(f"Invalid NRRD {key} {raw!r}: {nrrd_path}")
    return tuple(int(part) for part in parts)



def _slicer_conversion_parameters(header):
    """Slicer ``Segmentation_ConversionParameters`` as {name: value} (``name|value|help&...``)."""
    raw = header.get('Segmentation_ConversionParameters')
    parameters = {}
    for entry in str(raw or '').split('&'):
        name, _, rest = entry.partition('|')
        if name.strip():
            parameters[name.strip()] = rest.partition('|')[0].strip()
    return parameters



def _slicer_reference_geometry_shape(parameters, nrrd_path):
    """Per-data-axis sizes of Slicer's saved reference image geometry, or None when unset.

    The value is the IJK-to-RAS matrix (16 numbers) followed by the reference extent
    ``0 X-1 0 Y-1 0 Z-1`` in the stored labelmap's axis order."""
    value = parameters.get('Reference image geometry', '')
    if not value:
        return None
    parts = [part for part in value.split(';') if part.strip()]
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        numbers = []
    if (len(numbers) != 22 or not all(math.isfinite(number) for number in numbers)
            or not all(number.is_integer() for number in numbers[16:])):
        raise ValueError(f"Invalid Slicer reference image geometry {value!r}: {nrrd_path}")
    extent = [int(number) for number in numbers[16:]]
    if extent[0::2] != [0, 0, 0] or any(high < 0 for high in extent[1::2]):
        raise ValueError(f"Slicer reference image geometry must start at voxel zero: {value!r}: {nrrd_path}")
    return tuple(high + 1 for high in extent[1::2])



def _nrrd_reference_crop(header, stored_shape, expected_shape_tyx, nrrd_path, *, indexed=False):
    """Place a cropped segmentation raster inside its reference grid.

    Returns None for a raster that already is its reference grid. Otherwise returns
    ``(reference_shape, offset)`` in NRRD data-axis order. Cropped TTA layers and
    Slicer's own seg.nrrd saves store only a box at
    ``Segmentation_ReferenceImageExtentOffset``. The reference grid comes from XTA's
    ``Segmentation_ReferenceImageExtent``, else Slicer's saved reference image
    geometry, else the expected image grid mapped through the header axis directions
    (an offset-only box is then pasted without --force resampling). Indexed 2D
    correspondence needs a declared reference depth. A zero offset that cannot describe
    a crop of the image grid (indexed mode, no axis directions, or a box larger than the
    grid) is read as a complete raster, as before cropped layers existed.
    """
    stored_shape = tuple(int(value) for value in stored_shape)
    offset = _nrrd_header_integers(header, _NRRD_REFERENCE_OFFSET_KEY, 3, nrrd_path)
    extent = _nrrd_header_integers(header, _NRRD_REFERENCE_EXTENT_KEY, 6, nrrd_path)
    if (offset is None and extent is None) or len(stored_shape) != 3:
        return None
    offset = (0, 0, 0) if offset is None else offset
    parameters = _slicer_conversion_parameters(header)
    if extent is not None:
        if extent[0::2] != (0, 0, 0) or any(high < 0 for high in extent[1::2]):
            raise ValueError(f"NRRD reference extent must start at voxel zero: {extent}: {nrrd_path}")
        reference = tuple(high + 1 for high in extent[1::2])
    else:
        reference = _slicer_reference_geometry_shape(parameters, nrrd_path)
    if reference is None:
        complete_raster_if_zero = offset == (0, 0, 0)
        if indexed:
            if complete_raster_if_zero:
                return None
            raise ValueError(
                f"A cropped NRRD segmentation without Segmentation_ReferenceImageExtent or a Slicer reference "
                f"image geometry has no source depth for --nrrd_z_start_index: {nrrd_path}"
            )
        permutation = nrrd_header_to_internal_permutation(header)
        if permutation is None:
            if complete_raster_if_zero:
                return None
            raise ValueError(
                f"Cropped NRRD segmentation has a reference offset but neither a reference extent nor "
                f"axis directions to place it in the image grid: {nrrd_path}"
            )
        placed = [0, 0, 0]
        for internal_axis, data_axis in enumerate(permutation):
            placed[int(data_axis)] = int(expected_shape_tyx[internal_axis])
        if complete_raster_if_zero and any(size > bound for size, bound in zip(stored_shape, placed)):
            return None
        reference = tuple(placed)
    if any(start < 0 or start + size > bound for start, size, bound in zip(offset, stored_shape, reference)):
        raise ValueError(
            f"NRRD segmentation box {stored_shape} at reference offset {offset} lies outside its reference "
            f"grid {reference} (NRRD axis order): {nrrd_path}"
        )
    if reference == stored_shape and offset == (0, 0, 0):
        return None
    if parameters.get('Oversampling factor', '1') not in ('1', '1.0'):
        raise ValueError(f"Oversampled Slicer segmentation geometry is not a PTA mask grid: {nrrd_path}")
    return reference, offset



def _read_cropped_nrrd_planes(
    stream, mask, summary, box_tyx, source_shape, source_planes, direct_read, nrrd_path,
):
    """Decode a stored box as if streaming its zero-filled reference grid.

    ``box_tyx`` is ``(offset_tyx, stored_shape_tyx)`` in the internal reference grid.
    Only box planes are read; all other reference voxels are zero."""
    (box_t, box_y, box_x), (stored_t, stored_h, stored_w) = box_tyx
    telemetry = runtime_telemetry()
    if direct_read:
        mask[:box_t] = 0
        mask[box_t + stored_t:] = 0
        step = max(1, (8 * 1024 * 1024) // max(1, stored_h * stored_w))
        block = np.empty((min(step, stored_t), stored_h, stored_w), np.uint8)
        for start in range(0, stored_t, step):
            count = min(step, stored_t - start)
            stored = block[:count]
            _read_nrrd_bytes_into(stream, memoryview(stored).cast('B'), nrrd_path)
            planes = mask[box_t + start:box_t + start + count]
            with telemetry.span('pta.mask.nrrd_normalize'):
                planes.fill(0)
                np.greater(stored, 0, out=planes[:, box_y:box_y + stored_h, box_x:box_x + stored_w].view(np.bool_))
                _record_plane_foreground(planes, summary, box_t + start)
        return
    xy_indices = _nrrd_mask_xy_indices(source_shape, mask.shape)
    target_planes_by_source: Dict[int, List[int]] = {}
    for target_z, source_z in enumerate(source_planes):
        target_planes_by_source.setdefault(int(source_z), []).append(target_z)
    plane = np.zeros(source_shape[1:], dtype=np.uint8)
    window = plane[box_y:box_y + stored_h, box_x:box_x + stored_w]
    stored_plane = np.empty((stored_h, stored_w), dtype=np.uint8)
    for source_z in range(source_shape[0]):
        if box_t <= source_z < box_t + stored_t:
            _read_nrrd_bytes_into(stream, memoryview(stored_plane).cast('B'), nrrd_path)
            window[:] = stored_plane
        elif source_z == box_t + stored_t:
            window[:] = 0
        targets = target_planes_by_source.get(source_z)
        if targets:
            with telemetry.span('pta.mask.nrrd_resample'):
                _write_nrrd_mask_targets(plane, mask, xy_indices, summary, targets)



def _read_canonical_uint8_nrrd(
    nrrd_path, nrrd, expected_shape_tyx, *, force=False,
    source_contiguous=True, source_plane_indices=None,
):
    """Bound gzip output in place instead of building and copying an entire bytearray."""
    try:
        from isal.igzip import IGzipFile as gzip_reader
    except (ImportError, OSError):
        from gzip import GzipFile as gzip_reader
    with Path(nrrd_path).open('rb') as source:
        header = nrrd.read_header(source)
        if (header.get('type') not in {'uint8', 'uint8_t', 'uchar', 'unsigned char'}
                or header.get('encoding') not in {'raw', 'gzip', 'gz'}
                or header.get('datafile', header.get('data file')) is not None
                or header.get('lineskip', header.get('line skip', 0)) != 0
                or header.get('byteskip', header.get('byte skip', 0)) != 0):
            return None
        shape = tuple(int(x) for x in header['sizes'])
        if int(header.get('dimension', 0)) != len(shape) or any(x <= 0 for x in shape):
            raise ValueError(f'Invalid NRRD dimensions: {nrrd_path}')
        # A cropped raster is oriented, resampled and summarized on its reference grid.
        crop = _nrrd_reference_crop(header, shape, expected_shape_tyx, nrrd_path,
                                    indexed=source_plane_indices is not None)
        stored_shape, shape = shape, (shape if crop is None else crop[0])
        orientation, permutation = _nrrd_mask_orientation(
            header, shape, expected_shape_tyx, nrrd_path, force=force,
            source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        )
        if tuple(permutation) != (2, 1, 0):
            return None
        source_shape = tuple(shape[axis] for axis in permutation)
        box_tyx = None if crop is None else (
            tuple(int(crop[1][axis]) for axis in permutation),
            tuple(int(stored_shape[axis]) for axis in permutation),
        )
        source_planes = _nrrd_mask_source_planes(
            source_shape, expected_shape_tyx, nrrd_path, force=force,
            source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        )
        mask = np.empty(tuple(expected_shape_tyx), np.uint8)
        summary = MaskPlaneSummary(np.zeros(len(mask), bool), np.zeros(len(mask), np.intp))
        for target_z, source_z in enumerate(source_planes):
            if source_z < 0 or source_z >= source_shape[0]:
                mask[target_z] = 0
        stream = gzip_reader(fileobj=source, mode='rb') if header['encoding'] in {'gzip', 'gz'} else source
        direct_read = source_shape == tuple(expected_shape_tyx) and source_plane_indices is None
        telemetry = runtime_telemetry()
        try:
            with telemetry.span('pta.mask.nrrd_payload_read'):
                if box_tyx is not None:
                    _read_cropped_nrrd_planes(
                        stream, mask, summary, box_tyx, source_shape, source_planes, direct_read, nrrd_path,
                    )
                elif direct_read:
                    # Normalize and summarize whole planes while they are still in cache.
                    step = max(1, (8 * 1024 * 1024) // mask[0].nbytes)
                    for start in range(0, len(mask), step):
                        planes = mask[start:start + step]
                        _read_nrrd_bytes_into(stream, memoryview(planes).cast('B'), nrrd_path)
                        with telemetry.span('pta.mask.nrrd_normalize'):
                            # A bool view writes the same 0/1 bytes without a cast loop.
                            np.greater(planes, 0, out=planes.view(np.bool_))
                            _record_plane_foreground(planes, summary, start)
                else:
                    xy_indices = _nrrd_mask_xy_indices(source_shape, expected_shape_tyx)
                    target_planes_by_source: Dict[int, List[int]] = {}
                    for target_z, source_z in enumerate(source_planes):
                        target_planes_by_source.setdefault(int(source_z), []).append(target_z)
                    plane = np.empty(source_shape[1:], dtype=np.uint8)
                    for source_z in range(source_shape[0]):
                        _read_nrrd_bytes_into(stream, memoryview(plane).cast('B'), nrrd_path)
                        targets = target_planes_by_source.get(source_z)
                        if targets:
                            with telemetry.span('pta.mask.nrrd_resample'):
                                _write_nrrd_mask_targets(plane, mask, xy_indices, summary, targets)
                if stream.read(1):
                    raise ValueError(f'NRRD segmentation payload exceeds declared dimensions: {nrrd_path}')
        finally:
            if stream is not source:
                stream.close()
        return mask, orientation, source_shape, source_planes, summary



def _record_plane_foreground(planes, summary, start):
    """Record any() and the first foreground flat index of consecutive 0/1 planes."""
    rows = planes.reshape(len(planes), -1).view(np.bool_)
    first = rows.argmax(axis=1)  # stops at the first True of each plane
    summary.first_foreground[start:start + len(rows)] = first
    summary.has_foreground[start:start + len(rows)] = rows[np.arange(len(rows)), first]



def _read_nrrd_bytes_into(stream, raw, nrrd_path):
    for start in range(0, len(raw), 8 * 1024 * 1024):
        chunk = raw[start:min(len(raw), start + 8 * 1024 * 1024)]
        filled = 0
        while filled < len(chunk):
            count = stream.readinto(chunk[filled:])
            if not count:
                raise ValueError(f'Unexpected EOF in NRRD segmentation payload: {nrrd_path}')
            filled += int(count)



# Each (Y run, X run) block costs one ufunc call. Near-unity scales (the usual --force
# case, a few pixels off) give a handful of blocks; a 2x rescale gives millions, where the
# single np.ix_ gather is hundreds of times faster.
_MAX_NRRD_RUN_BLOCKS = 64



def _nrrd_mask_xy_indices(source_shape, target_shape):
    """None for equal XY. Else ('runs', y_runs, x_runs) of (target start, target stop, source
    start) when few blocks cover a plane, or ('gather', np.ix_ index) otherwise."""
    if source_shape[1:] == target_shape[1:]:
        return None
    axes = [_force_fit_nearest_source_indices(source_size, target_size)
            for source_size, target_size in zip(source_shape[1:], target_shape[1:])]
    # A run continues while each target step advances the source by exactly one.
    breaks = [np.flatnonzero(np.diff(index) != 1) + 1 for index in axes]
    if (len(breaks[0]) + 1) * (len(breaks[1]) + 1) > _MAX_NRRD_RUN_BLOCKS:
        return ('gather', np.ix_(*axes))
    runs = []
    for index, cuts in zip(axes, breaks):
        starts, stops = np.concatenate(([0], cuts)), np.concatenate((cuts, [len(index)]))
        runs.append(tuple((int(a), int(b), int(index[a])) for a, b in zip(starts, stops)))
    return ('runs', *runs)



def _write_nrrd_mask_plane(source, destination, xy_indices):
    """Write source > 0 as 0/1 bytes; near-unity resizes copy each nearest-neighbour run as a view."""
    out = destination.view(np.bool_)
    if xy_indices is None:
        np.greater(source, 0, out=out)
        return
    if xy_indices[0] == 'gather':
        np.greater(source[xy_indices[1]], 0, out=out)
        return
    _, y_runs, x_runs = xy_indices
    for y0, y1, source_y in y_runs:
        for x0, x1, source_x in x_runs:
            np.greater(source[source_y:source_y + y1 - y0, source_x:source_x + x1 - x0], 0,
                       out=out[y0:y1, x0:x1])



def _write_nrrd_mask_targets(source, mask, xy_indices, summary, targets):
    """Normalize and summarize each shared source plane once."""
    first = targets[0]
    _write_nrrd_mask_plane(source, mask[first], xy_indices)
    _record_plane_foreground(mask[first:first + 1], summary, first)
    for target in targets[1:]:
        mask[target] = mask[first]
        summary.has_foreground[target] = summary.has_foreground[first]
        summary.first_foreground[target] = summary.first_foreground[first]


def load_nrrd_mask_for_volume(
    nrrd_path: Path, expected_shape_tyx: Tuple[int, int, int], *, warnings: WarningLog,
    force: bool = False, source_contiguous: bool = True,
    source_plane_indices: Optional[Sequence[int]] = None,
    plane_summary_out: Optional[List[MaskPlaneSummary]] = None,
) -> np.ndarray:
    """plane_summary_out, when given, receives the reader's MaskPlaneSummary if it made one."""
    mask, _ = _load_nrrd_mask_for_volume(
        nrrd_path, expected_shape_tyx, warnings=warnings, force=force,
        source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        plane_summary_out=plane_summary_out,
    )
    return mask



def load_indexed_nrrd_mask_for_volume(
    nrrd_path: Path, expected_shape_tyx: Tuple[int, int, int], *, warnings: WarningLog,
    source_plane_indices: Sequence[int], force: bool = False,
    plane_summary_out: Optional[List[MaskPlaneSummary]] = None,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Load known 2D correspondences and return the annotated target positions."""
    return _load_nrrd_mask_for_volume(
        nrrd_path, expected_shape_tyx, warnings=warnings, force=force,
        source_contiguous=False, source_plane_indices=source_plane_indices,
        plane_summary_out=plane_summary_out,
    )



@runtime_telemetry_phase('pta.mask.load_nrrd')
def _load_nrrd_mask_for_volume(
    nrrd_path, expected_shape_tyx, *, warnings, force,
    source_contiguous, source_plane_indices, plane_summary_out=None,
):
    try:
        import nrrd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pynrrd is required for NRRD segmentation input: pip install pynrrd") from exc
    expected_shape_tyx = tuple(int(value) for value in expected_shape_tyx)
    if len(expected_shape_tyx) != 3 or any(value <= 0 for value in expected_shape_tyx):
        raise ValueError(f"Image volume shape must contain three positive axes: {expected_shape_tyx}")
    if source_plane_indices is not None:
        source_plane_indices = tuple(int(value) for value in source_plane_indices)
        if len(source_plane_indices) != expected_shape_tyx[0]:
            raise ValueError("NRRD source-plane mapping must contain one index per image frame")
    fast = _read_canonical_uint8_nrrd(
        nrrd_path, nrrd, expected_shape_tyx, force=force,
        source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
    )
    if fast is not None:
        mask, orientation, source_shape, source_planes, plane_summary = fast
        if plane_summary_out is not None:
            plane_summary_out.append(plane_summary)
        warnings.add('segmentation_nrrd_bounded_readinto', f'{nrrd_path.name}: {mask.nbytes} bytes, bounded 8 MiB output chunks')
    else:
        data, header = nrrd.read(str(nrrd_path))
        arr = np.asarray(data)
        crop = _nrrd_reference_crop(header, arr.shape, expected_shape_tyx, nrrd_path,
                                    indexed=source_plane_indices is not None)
        reference_shape, offset = (arr.shape, (0, 0, 0)) if crop is None else crop
        orientation, permutation = _nrrd_mask_orientation(
            header, reference_shape, expected_shape_tyx, nrrd_path, force=force,
            source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        )
        internal = arr if permutation == (0, 1, 2) else np.transpose(arr, permutation)
        source_shape = tuple(int(reference_shape[axis]) for axis in permutation)
        box_offset = tuple(int(offset[axis]) for axis in permutation)
        box = tuple(slice(start, start + size) for start, size in zip(box_offset, internal.shape))
        source_planes = _nrrd_mask_source_planes(
            source_shape, expected_shape_tyx, nrrd_path, force=force,
            source_contiguous=source_contiguous, source_plane_indices=source_plane_indices,
        )
        mask = np.empty(expected_shape_tyx, dtype=np.uint8)
        if source_shape == expected_shape_tyx and source_plane_indices is None:
            if crop is not None:
                mask.fill(0)
            np.greater(internal, 0, out=mask[box].view(np.bool_))
        else:
            xy_indices = _nrrd_mask_xy_indices(source_shape, expected_shape_tyx)
            if crop is not None:
                reference_plane = np.zeros(source_shape[1:], dtype=internal.dtype)
                window = reference_plane[box[1:]]
            for target_z, source_z in enumerate(source_planes):
                stored_z = int(source_z) - box_offset[0]
                if 0 <= stored_z < len(internal):
                    plane = internal[stored_z]
                    if crop is not None:
                        window[:] = plane
                        plane = reference_plane
                    _write_nrrd_mask_plane(plane, mask[target_z], xy_indices)
                else:
                    mask[target_z] = 0
    matched_positions = tuple(
        target_z for target_z, source_z in enumerate(source_planes)
        if 0 <= source_z < source_shape[0]
    )
    if source_shape[1:] != expected_shape_tyx[1:] or (
        source_plane_indices is None and source_shape[0] != expected_shape_tyx[0]
    ):
        axes = "XY" if source_plane_indices is not None or not source_contiguous else "XYZ"
        message = (f"{nrrd_path.name}: source(t,Y,X)={source_shape}, target={expected_shape_tyx}, "
                   f"axes={axes}, tolerance=3 per axis inclusive, endpoint-aligned nearest sampling")
        warnings.add("segmentation_nrrd_forced_resize", message)
        print(f"Resized NRRD labels: {message}")
    if source_plane_indices is not None:
        warnings.add(
            "segmentation_nrrd_indexed_2d_mapping",
            f"{nrrd_path.name}: matched={len(matched_positions)}/{expected_shape_tyx[0]}; "
            "source-plane indices preserved",
        )
    warnings.add("segmentation_nrrd_loaded_in_memory", f"{nrrd_path.name}: {mask.shape}, {mask.nbytes / GIB:.2f} GiB, orientation={orientation}")
    return mask, matched_positions
