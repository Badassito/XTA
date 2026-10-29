"""PTA source image/video readers and annotation input I/O.

This module owns grayscale normalization, bounded input decoding, and NRRD/YOLO
annotation parsing. It does not import the PTA orchestration module or own its
source/prepared-volume class identities.
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

from .runtime import runtime_telemetry, runtime_telemetry_phase
from .pta_workers import ArrayAllocator, SharedBlock, WarningLog, bind_current_thread_to_cpus


GIB = 1024 ** 3
YoloPolygon = Tuple[Tuple[float, float], ...]
YoloPolygons = Tuple[YoloPolygon, ...]



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
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
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
        cmd2 = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "json", str(video_path),
        ]
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
    raw = memoryview(arr).cast("B")
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        with tqdm(total=t, desc="Decoding video to in-memory gray8 volume") as pbar:
            for start in range(0, t, chunk_frames):
                nframes = min(chunk_frames, t - start)
                need = nframes * frame_bytes
                offset = start * frame_bytes
                view = raw[offset:offset + need]
                filled = 0
                while filled < need:
                    nread = proc.stdout.readinto(view[filled:])
                    if nread is None or nread <= 0:
                        raise RuntimeError(f"Unexpected EOF while decoding {video_path} near frame {start + 1}")
                    filled += int(nread)
                pbar.update(nframes)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        _, err = proc.communicate()
    if proc.returncode not in (0, None):
        msg = err.decode("utf-8", errors="replace") if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"ffmpeg decode failed: {msg}")
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



def _nrrd_mask_orientation(header, shape, expected_shape_tyx, nrrd_path):
    if len(shape) != 3:
        raise ValueError(f"NRRD segmentation must be 3D, got shape {shape} from {nrrd_path}")
    t_dim, h, w = (int(expected_shape_tyx[0]), int(expected_shape_tyx[1]), int(expected_shape_tyx[2]))
    shape = tuple(int(x) for x in shape)
    candidates: List[Tuple[str, Tuple[int, int, int]]] = []
    for name, source_shape, permutation in (
        ("already_tyx", (t_dim, h, w), (0, 1, 2)),
        ("xyz_to_tyx", (w, h, t_dim), (2, 1, 0)),
        ("yxt_to_tyx", (h, w, t_dim), (2, 0, 1)),
        ("txy_to_tyx", (t_dim, w, h), (0, 2, 1)),
    ):
        if shape == source_shape:
            candidates.append((name, permutation))
    if not candidates:
        raise ValueError(
            f"NRRD segmentation shape {shape} does not match image volume shape (t,Y,X)=({t_dim},{h},{w}) "
            f"or recognized spatial permutations (X,Y,t)=({w},{h},{t_dim}) / (Y,X,t)=({h},{w},{t_dim})"
        )

    header_permutation = nrrd_header_to_internal_permutation(header)
    if header_permutation is not None:
        internal_shape = tuple(shape[int(axis)] for axis in header_permutation)
        if internal_shape != (t_dim, h, w):
            raise ValueError(
                f"NRRD header axis directions imply internal shape {internal_shape}, but image volume shape is "
                f"(t,Y,X)=({t_dim},{h},{w}) for {nrrd_path}"
            )
        permutation = tuple(int(x) for x in header_permutation)
        orientation = "header_axes_" + "".join(str(x) for x in permutation) + "_to_tyx"
    else:
        unique_permutations = {tuple(permutation) for _, permutation in candidates}
        if len(unique_permutations) != 1:
            matches = ", ".join(name for name, _ in candidates)
            raise ValueError(
                f"NRRD axis order is ambiguous from shape {shape} ({matches}) and the header has no usable "
                f"space directions/content marker: {nrrd_path}. Add axis-aware NRRD metadata before import."
            )
        orientation, permutation = candidates[0]

    return orientation, permutation



def _read_canonical_uint8_nrrd(nrrd_path, nrrd, expected_shape_tyx):
    """Bound gzip output in place instead of building and copying an entire bytearray."""
    import gzip
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
        orientation, permutation = _nrrd_mask_orientation(header, shape, expected_shape_tyx, nrrd_path)
        if tuple(permutation) != (2, 1, 0):
            return None
        mask = np.empty(tuple(expected_shape_tyx), np.uint8)
        raw = memoryview(mask).cast('B')
        stream = gzip.GzipFile(fileobj=source, mode='rb') if header['encoding'] in {'gzip', 'gz'} else source
        try:
            with runtime_telemetry().span('pta.mask.nrrd_payload_read'):
                for start in range(0, len(raw), 8 * 1024 * 1024):
                    chunk = raw[start:min(len(raw), start + 8 * 1024 * 1024)]
                    filled = 0
                    while filled < len(chunk):
                        count = stream.readinto(chunk[filled:])
                        if not count:
                            raise ValueError(f'Unexpected EOF in NRRD segmentation payload: {nrrd_path}')
                        filled += int(count)
                if stream.read(1):
                    raise ValueError(f'NRRD segmentation payload exceeds declared dimensions: {nrrd_path}')
        finally:
            if stream is not source:
                stream.close()
        with runtime_telemetry().span('pta.mask.nrrd_normalize'):
            np.greater(mask, 0, out=mask)
        return mask, orientation



@runtime_telemetry_phase('pta.mask.load_nrrd')
def load_nrrd_mask_for_volume(nrrd_path: Path, expected_shape_tyx: Tuple[int, int, int], *, warnings: WarningLog) -> np.ndarray:
    try:
        import nrrd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pynrrd is required for NRRD segmentation input: pip install pynrrd") from exc
    fast = _read_canonical_uint8_nrrd(nrrd_path, nrrd, expected_shape_tyx)
    if fast is not None:
        mask, orientation = fast
        warnings.add('segmentation_nrrd_bounded_readinto', f'{nrrd_path.name}: {mask.nbytes} bytes, bounded 8 MiB output chunks')
    else:
        data, header = nrrd.read(str(nrrd_path))
        arr = np.asarray(data)
        orientation, permutation = _nrrd_mask_orientation(header, arr.shape, expected_shape_tyx, nrrd_path)
        internal = arr if permutation == (0, 1, 2) else np.transpose(arr, permutation)
        mask = np.ascontiguousarray((np.asarray(internal) > 0).astype(np.uint8))
    warnings.add("segmentation_nrrd_loaded_in_memory", f"{nrrd_path.name}: {mask.shape}, {mask.nbytes / GIB:.2f} GiB, orientation={orientation}")
    return mask
