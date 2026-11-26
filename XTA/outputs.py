"""NRRD, TIFF, MKV, summary, and low-quality output writers."""

from __future__ import annotations

import colorsys
import contextlib
import copy
import gc
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zlib
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import (
    dataclass,
    field,
    replace as dataclasses_replace,
)
from functools import lru_cache
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
import numpy as np
from .json_publication import _fsync_parent_directory, write_json_atomic
from .nrrd_spans import stream_native_crop_spans
from . import nrrd_compression
# Retained import contracts for existing stream users and embedded shutdown callers.
from .nrrd_compression import (
    _MemberParallelGzipPayloadWriter,
    nrrd_gzip_workers,
    nrrd_gzip_chunk_bytes,
    nrrd_member_gzip_window_bytes,
    shutdown_nrrd_gzip_executors,
)

from ._deps import _numba, cv2, tifffile, tqdm
from .polygon_keyholes import holes_by_parent, splice_holes

from .config import (
    ChannelFormat,
    DEFAULT_CHANNEL_FORMAT,
    GIB,
    NRRD_SPACE,
    SCRIPT_BASENAME,
    SaveRequest,
    _parse_token_list,
    resolve_channel_format,
)
from .runtime import (
    _acquire_parallel_pool,
    _release_parallel_pool,
    _settle_parallel_futures,
    allocate_workspace_array,
    choose_parallel_chunk_size,
    choose_slice_parallel_workers,
    close_memmap_array,
    parallel_for_indices,
    parallel_for_indices_chunked,
    runtime_telemetry,
    runtime_telemetry_phase,
    _madvise_mmap_traced,
)

# Explicit lower-layer dependencies keep imports one-way.
from .workspace import (
    _cpu_count,
    _env_flag,
    _env_float,
    _env_int,
    available_anon_work_bytes,
)
from .media import (
    _center_aligned_nearest_source_index,
    _center_aligned_nearest_source_indices,
    _center_aligned_source_position,
    _memory_backed_encoded_chunk_path,
    _memory_backed_encoded_chunk_size,
    _open_memory_backed_encoded_chunk,
    _resize_gray_slice_nearest_or_linear,
    _spawn_subprocess_with_retry,
    close_ffmpeg_writer,
    ffmpeg_ffv1_gray_writer,
    ffmpeg_ffv1_rgb_writer,
    resize_binary_plane_to_output_xy,
)
from .geometry import (
    ChannelFormattedFrameRenderer,
    ViewInfo,
    get_view_frame_by_index,
)
from .render_batch import RenderBatch
from .interpolation import (
    CTILE_INDEX_DTYPE,
    MASK_STORE_FORMATS,
    NrrdLayerRef,
    NrrdRasterPlan,
    NrrdSegmentExtent,
    RawBBoxMaskStore,
    _coerce_segment_extent,
    _nrrd_empty_segment_extent,
    _release_raw_store_chunks_ram_cache,
)
from .cuda_d1 import (
    _read_binary_volume_slice_bool,
    _read_binary_volume_slice_u8,
    _sanitize_nrrd_layer_token,
    _volume_shape_tuple,
)
from .backprojection import (
    _MainProcessGpuStageLease,
    _announce_main_gpu_stage_skip_once,
    _trim_main_process_cuda_device,
    _try_acquire_main_process_gpu_stage,
)

_OVERLAY_BLUE_U16 = np.array([0, 0, 255], dtype=np.uint16)  # RGB blue in the uint16 blend domain

def _overlay_blend_blue_inplace(frame_rgb: np.ndarray, mask2d: np.ndarray) -> None:
    """50% blue blend on mask pixels, restricted to the mask's bbox, in place.

 Byte-identical to the legacy full-frame masked path ((frame + blue) // 2 in uint16 on
 mask pixels, untouched elsewhere) but allocates only bbox-sized temps: no full-frame
 bool cast, no masked gather/scatter over the whole frame."""
    m = np.asarray(mask2d)
    ys = np.flatnonzero(m.any(axis=1))
    if ys.size == 0:
        return
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    xs = np.flatnonzero(m[y0:y1].any(axis=0))
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    sub = frame_rgb[y0:y1, x0:x1]
    msub = m[y0:y1, x0:x1] != 0
    blended = ((sub.astype(np.uint16) + _OVERLAY_BLUE_U16) // 2).astype(np.uint8)
    np.copyto(sub, blended, where=msub[:, :, None])

def _gray_frame_into_rgb_buffer(frame_gray: np.ndarray, frame_buf: np.ndarray) -> np.ndarray:
    """Expand a gray frame into the reused (H, W, 3) buffer, no per-frame alloc.

 Returns the buffer actually written (cv2 falls back to allocating only if dst were
 incompatible, which the caller's preallocated buffer never is)."""
    gray = np.ascontiguousarray(np.asarray(frame_gray, dtype=np.uint8))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, dst=frame_buf)

def ffv1_segment_count(total_frames: int) -> int:
    """Number of concurrent contiguous FFV1 encode shards for a final video.

 Each shard has identical codec parameters and contains whole, independently decodable
 FFV1 frames. ``YOLO_TTA_FFV1_SEGMENTS=1`` preserves the original single-pipe path.
 Explicit values are bounded only by the number of frames; the automatic default targets
 one 30-thread FFV1 encoder per roughly 32 allocated CPUs, capped at six encoders."""
    total = max(0, int(total_frames))
    automatic = min(6, max(1, int(math.ceil(float(_cpu_count()) / 32.0))))
    requested = max(1, _env_int('YOLO_TTA_FFV1_SEGMENTS', int(automatic)))
    return 1 if total <= 1 else min(int(total), int(requested))

def _ffv1_contiguous_segments(total_frames: int, segment_count: int) -> List[Tuple[int, int]]:
    """Return balanced, gap-free ``[start, stop)`` frame ranges."""
    total = max(0, int(total_frames))
    count = max(1, min(int(segment_count), max(1, total)))
    base, extra = divmod(total, count)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(count):
        stop = start + int(base) + (1 if idx < int(extra) else 0)
        ranges.append((int(start), int(stop)))
        start = stop
    return ranges

def _atomic_publication_stage_path(
    out_path: Path,
    *,
    publication_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return a same-filesystem staging path outside the publication tree.

 The staging directory is a sibling of the run's publication root. Recursive rsync or
 watchers rooted at ``publication_root`` therefore see neither shards nor a growing
 ``.assembling`` inode; only the final atomic rename enters that tree."""
    final_path = Path(out_path)
    root = Path(publication_root) if publication_root is not None else final_path.parent
    final_path.parent.mkdir(parents=True, exist_ok=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    # Every publisher owns its directory. A shared empty staging directory can be
    # removed by one concurrent writer after another has selected (but not yet
    # created) its stage file.
    stage_dir = Path(tempfile.mkdtemp(
        prefix=f'.{root.name}.atomic-result-',
        dir=str(root.parent),
    ))
    if int(stage_dir.stat().st_dev) != int(final_path.parent.stat().st_dev):
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise RuntimeError(
            f'Atomic publication staging and destination are on different filesystems: '
            f'{stage_dir} vs {final_path.parent}'
        )
    token = f'{os.getpid()}.{threading.get_ident()}.{time.time_ns()}'
    stage_path = stage_dir / f'.{final_path.stem}.{token}.assembling{final_path.suffix}'
    return stage_path, stage_dir

def _publish_staged_file_atomically(stage_path: Path, out_path: Path) -> None:
    """Durably replace ``out_path`` with a completed same-filesystem staging file."""
    stage = Path(stage_path)
    final_path = Path(out_path)
    if not stage.is_file() or int(stage.stat().st_size) <= 0:
        raise RuntimeError(f'Atomic publication stage is missing or empty: {stage}')
    telemetry = runtime_telemetry() if final_path.suffix.lower() == '.nrrd' else None
    def phase(name):
        return telemetry.span('nrrd.publication.' + name) if telemetry is not None else contextlib.nullcontext()
    # Windows rejects FlushFileBuffers (os.fsync) on a read-only handle with EBADF.
    # Open update-capable without modifying the completed payload.
    with phase('file_durability'):
        with open(stage, 'rb+') as stage_fh:
            os.fsync(stage_fh.fileno())
    with phase('rename'):
        os.replace(stage, final_path)
    with phase('directory_durability'):
        _fsync_parent_directory(final_path.parent)


@contextlib.contextmanager
def _same_directory_atomic_output(out_path: Path) -> Iterator[Path]:
    """Yield a private sibling path and publish it only after a successful write.

    Keeping the temporary inode in the destination directory guarantees that the final
    ``os.replace`` cannot cross filesystems.  The previous output remains intact if the
    producer raises, and the private file is removed on every exit path.
    """
    final_path = Path(out_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    token = f'{os.getpid()}.{threading.get_ident()}.{time.time_ns()}'
    stage_path = final_path.with_name(
        f'.{final_path.name}.{token}.assembling'
    )
    try:
        yield stage_path
        _publish_staged_file_atomically(stage_path, final_path)
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_json_atomically(out_path: Path, payload: object) -> Path:
    """Serialize JSON completely before atomically replacing its public sidecar."""
    return write_json_atomic(out_path, payload)

def _run_sharded_ffv1_encode(
    *,
    out_path: Path,
    total_frames: int,
    description: str,
    show_progress: bool,
    encode_range: Callable[[int, int, Path, threading.Event, Callable[[subprocess.Popen], None], Callable[[subprocess.Popen], None], Callable[[], None]], None],
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> None:
    """Encode one FFV1 stream or concatenate memory-backed shards atomically.

 Each range callback owns one ffmpeg producer/process whose Matroska output is an anonymous
 memfd. On the first shard failure all registered encoders are terminated and every worker
 is reaped. The only filesystem write is the ordered concat stream into a same-filesystem
 atomic publication stage, so the previous destination remains untouched on failure.
    A single encoder writes directly to the private publication stage.
    ``scratch_dir`` is accepted for caller compatibility and ignored."""
    _ = scratch_dir
    total = int(total_frames)
    segments = int(ffv1_segment_count(total))
    cancel_event = threading.Event()
    active_lock = threading.Lock()
    active_processes: set[subprocess.Popen] = set()

    def _register(proc: subprocess.Popen) -> None:
        with active_lock:
            active_processes.add(proc)
            cancelled = cancel_event.is_set()
        if cancelled:
            try:
                proc.terminate()
            except Exception:
                pass

    def _unregister(proc: subprocess.Popen) -> None:
        with active_lock:
            active_processes.discard(proc)

    def _terminate_active() -> None:
        cancel_event.set()
        with active_lock:
            processes = list(active_processes)
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        deadline = time.monotonic() + 5.0
        for proc in processes:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                pass

    progress = tqdm(total=max(0, total), desc=description, disable=not show_progress)

    def _progress_one() -> None:
        progress.update(1)

    joined_path: Optional[Path] = None
    publication_stage_dir: Optional[Path] = None
    executor: Optional[ThreadPoolExecutor] = None
    futures: List[Future] = []
    shard_files: List[object] = []
    first_error: Optional[BaseException] = None
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if segments == 1 or not callable(getattr(os, 'memfd_create', None)):
            joined_path, publication_stage_dir = _atomic_publication_stage_path(
                Path(out_path), publication_root=publication_root,
            )
            encode_range(0, total, joined_path, cancel_event, _register, _unregister, _progress_one)
            _publish_staged_file_atomically(joined_path, Path(out_path))
            joined_path = None
            return
        ranges = _ffv1_contiguous_segments(total, segments)
        for idx in range(len(ranges)):
            shard_files.append(_open_memory_backed_encoded_chunk(
                f'{Path(out_path).stem}.ffv1.segment_{idx:04d}.mkv',
                require_fileno=True,
            ))
        shard_paths = [_memory_backed_encoded_chunk_path(chunk) for chunk in shard_files]
        executor = ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix='ffv1-segment')
        futures = [
            executor.submit(
                encode_range,
                int(start),
                int(stop),
                shard_path,
                cancel_event,
                _register,
                _unregister,
                _progress_one,
            )
            for (start, stop), shard_path in zip(ranges, shard_paths)
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    _terminate_active()
        if first_error is not None:
            raise RuntimeError(f'FFV1 segmented encode failed for {Path(out_path).name}') from first_error

        missing_indices: List[int] = []
        for idx, chunk in enumerate(shard_files):
            chunk.flush()
            if _memory_backed_encoded_chunk_size(chunk) <= 0:
                missing_indices.append(int(idx))
        if missing_indices:
            raise RuntimeError(
                f'FFV1 encoder did not produce {len(missing_indices)} in-memory shard(s): '
                f'{missing_indices[:3]}'
            )

        joined_path, publication_stage_dir = _atomic_publication_stage_path(
            Path(out_path), publication_root=publication_root,
        )
        for chunk in shard_files:
            chunk.seek(0)
        concat_text = 'ffconcat version 1.0\n' + ''.join(
            f"file 'file:/proc/self/fd/{int(chunk.fileno())}'\n"
            for chunk in shard_files
        )
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'concat', '-safe', '0',
            '-protocol_whitelist', 'file,pipe', '-i', 'pipe:0',
            '-map', '0:v:0', '-c', 'copy', '-f', 'matroska', str(joined_path),
        ]
        completed = _spawn_subprocess_with_retry(
            lambda: subprocess.run(
                cmd,
                input=concat_text.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                pass_fds=tuple(int(chunk.fileno()) for chunk in shard_files),
            ),
            f'ffmpeg FFV1 concat ({Path(out_path).name})',
        )
        if int(completed.returncode) != 0:
            stderr = (
                completed.stderr.decode('utf-8', errors='ignore')
                if isinstance(completed.stderr, (bytes, bytearray)) else str(completed.stderr)
            )
            raise RuntimeError(f'FFV1 lossless concat failed for {Path(out_path).name}: {stderr}')
        _publish_staged_file_atomically(joined_path, Path(out_path))
        joined_path = None
    except BaseException:
        _terminate_active()
        raise
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # Python < 3.9 compatibility
                executor.shutdown(wait=True)
        progress.close()
        for chunk in shard_files:
            try:
                chunk.close()
            except Exception:
                pass
        if joined_path is not None:
            try:
                joined_path.unlink(missing_ok=True)
            except Exception:
                pass
        if publication_stage_dir is not None:
            try:
                publication_stage_dir.rmdir()
            except OSError:
                pass

def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W) gray/luma
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
    show_progress: bool = True,
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> None:
    """Encode a source/mask overlay, using sharded FFV1 and memory-backed concatenation when admitted."""
    T, H, W = volume_rgb.shape
    assert mask_u8.shape == (T, H, W)

    def _encode_range(
        start: int,
        stop: int,
        shard_path: Path,
        cancel_event: threading.Event,
        register_proc: Callable[[subprocess.Popen], None],
        unregister_proc: Callable[[subprocess.Popen], None],
        progress_one: Callable[[], None],
    ) -> None:
        proc = ffmpeg_ffv1_rgb_writer(
            shard_path,
            width=W,
            height=H,
            fps=fps,
        )
        register_proc(proc)
        frame_buf = np.empty((int(H), int(W), 3), dtype=np.uint8)
        primary_error: Optional[BaseException] = None
        primary_traceback = None
        try:
            assert proc.stdin is not None
            for t in range(int(start), int(stop)):
                if cancel_event.is_set():
                    break
                frame = _gray_frame_into_rgb_buffer(volume_rgb[t], frame_buf)
                _overlay_blend_blue_inplace(frame, mask_u8[t])
                proc.stdin.write(memoryview(frame).cast('B'))
                progress_one()
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        try:
            close_ffmpeg_writer(proc)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                primary_traceback = exc.__traceback__
        finally:
            unregister_proc(proc)
        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)

    _run_sharded_ffv1_encode(
        out_path=Path(out_path),
        total_frames=int(T),
        description=f"Writing overlay video ({out_path.name})",
        show_progress=bool(show_progress),
        encode_range=_encode_range,
        scratch_dir=scratch_dir,
        publication_root=publication_root,
    )

def mask_to_yolo_polygons(mask01: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Convert a binary mask (H,W) to normalized polygons, one per outer contour.

    Enclosed holes join their outer polygon through zero-width keyhole bridges;
    an island inside a hole is its own polygon.
    """
    h, w = mask01.shape
    m = (mask01.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    children = holes_by_parent(hierarchy)

    polys: List[List[Tuple[float, float]]] = []
    for index, cnt in enumerate(contours):
        if hierarchy is not None and hierarchy[0, index, 3] >= 0:
            continue
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is None or len(approx) < 3:
            continue
        holes = [cv2.approxPolyDP(contours[child], epsilon=1.0, closed=True).reshape(-1, 2)
                 for child in children.get(index, ())]
        pts = splice_holes(approx.reshape(-1, 2), [ring for ring in holes if len(ring) >= 3])
        polys.append([(float(x) / float(w), float(y) / float(h)) for (x, y) in pts])
    return polys

DEFAULT_LABEL_PATTERN = "labels/{Filename}_%04d.txt"

DEFAULT_BINARY_PATTERN = "binary_masks/{Filename}_Binary_%04d.tiff"

DEFAULT_SEMANTIC_PATTERN = "semantic_masks/{Filename}_%04d.png"



def _format_frame_path(pattern_path: Path, frame_idx_1based: int) -> Path:
    as_str = str(pattern_path)
    if "%" in as_str:
        try:
            return Path(as_str % int(frame_idx_1based))
        except TypeError:
            pass
    return pattern_path.with_name(f"{pattern_path.stem}_{int(frame_idx_1based):04d}{pattern_path.suffix}")


_FRAME_NUMBER_PLACEHOLDER_RE = re.compile(r'%(?:0\d+)?d')


def _frame_sequence_name_matcher(
    pattern_path: Path,
    extensions: Sequence[str],
) -> re.Pattern[str]:
    """Match files belonging to a frame pattern, independent of frame count/extension."""
    pattern = Path(pattern_path)
    name = pattern.name
    placeholder = _FRAME_NUMBER_PLACEHOLDER_RE.search(name)
    suffix = pattern.suffix
    if placeholder is not None:
        prefix = name[:placeholder.start()]
        tail = name[placeholder.end():]
        tail_without_suffix = tail[:-len(suffix)] if suffix and tail.lower().endswith(suffix.lower()) else tail
    else:
        prefix = f'{pattern.stem}_'
        tail_without_suffix = ''

    def _normalize_extension(extension: object) -> str:
        value = str(extension)
        value = value if value.startswith('.') else f'.{value}'
        return value.lower() if os.name == 'nt' else value

    normalized_extensions = {
        _normalize_extension(ext)
        for ext in extensions
        if str(ext)
    }
    if suffix:
        normalized_extensions.add(_normalize_extension(suffix))
    extension_expr = (
        '(?:' + '|'.join(re.escape(ext) for ext in sorted(normalized_extensions)) + ')'
        if normalized_extensions else ''
    )
    return re.compile(
        rf'^{re.escape(prefix)}\d+{re.escape(tail_without_suffix)}{extension_expr}$',
        # Windows paths are conventionally case-insensitive; POSIX paths are not.  On
        # Linux, folding case here could delete a distinct sequence in the same directory.
        flags=(re.IGNORECASE if os.name == 'nt' else 0),
    )


def _publish_staged_frame_sequence(
    stage_pattern: Path,
    final_pattern: Path,
    total: int,
    *,
    stale_extensions: Sequence[str],
) -> None:
    """Publish a fully rendered sequence, then remove tails/obsolete extensions.

    Rendering happens in a hidden directory under the destination directory.  Therefore a
    rendering failure exposes none of the new generation and preserves every prior frame.
    Each completed frame enters the public sequence with an atomic same-filesystem replace.
    """
    count = max(0, int(total))
    pairs = [
        (
            _format_frame_path(Path(stage_pattern), idx),
            _format_frame_path(Path(final_pattern), idx),
        )
        for idx in range(1, count + 1)
    ]
    missing = [stage for stage, _final in pairs if not stage.is_file()]
    if missing:
        raise RuntimeError(f'Frame sequence staging is incomplete; missing {missing[0]}')

    for stage_path, final_path in pairs:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_path, final_path)

    expected = {
        os.path.normcase(os.path.abspath(str(final_path)))
        for _stage_path, final_path in pairs
    }
    matcher = _frame_sequence_name_matcher(Path(final_pattern), stale_extensions)
    for candidate in Path(final_pattern).parent.iterdir():
        if not candidate.is_file() or matcher.fullmatch(candidate.name) is None:
            continue
        normalized = os.path.normcase(os.path.abspath(str(candidate)))
        if normalized not in expected:
            candidate.unlink()


@contextlib.contextmanager
def _staged_frame_sequence(pattern_path: Path) -> Iterator[Path]:
    """Yield an equivalent frame pattern rooted in a private destination-side directory."""
    final_pattern = Path(pattern_path)
    final_pattern.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(
        prefix=f'.{final_pattern.parent.name}.frame-generation-',
        dir=str(final_pattern.parent),
    ))
    try:
        yield stage_dir / final_pattern.name
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

def _write_label_file_from_mask(mask2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask2d) > 0
    if not np.any(m):
        out_path.write_text("")
        return

    polys = mask_to_yolo_polygons(m.astype(np.uint8))
    if not polys:
        out_path.write_text("")
        return

    lines: List[str] = []
    for poly in polys:
        coords: List[str] = []
        for x, y in poly:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        lines.append("0 " + " ".join(coords))
    out_path.write_text("\n".join(lines) + "\n")

def _write_binary_tiff_frame(mask2d: np.ndarray, out_path: Path) -> None:
    """Write one true bilevel binary TIFF frame with DEFLATE compression."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_bool = np.asarray(mask2d, dtype=bool)
    tifffile.imwrite(
        str(out_path),
        mask_bool,
        photometric='minisblack',
        compression='deflate',
    )

def _write_semantic_png_frame(mask2d: np.ndarray, out_path: Path) -> None:
    """Write class IDs 0 (background) and 1 (foreground) as grayscale PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = np.ascontiguousarray(np.asarray(mask2d) != 0, dtype=np.uint8)
    if not cv2.imwrite(str(out_path), class_ids):
        raise RuntimeError(f'Failed to write semantic mask: {out_path}')

def write_yolo_labels_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def _write_frame(t: int) -> None:
            fp = _format_frame_path(stage_pattern, int(t) + 1)
            _write_label_file_from_mask(np.asarray(mask_u8[int(t)]), fp)

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f"Writing YOLO labels ({pattern_path.parent.name})",
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            pattern_path,
            total,
            stale_extensions=(pattern_path.suffix, '.txt'),
        )
    return pattern_path.parent

def write_binary_tiff_sequence_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def _write_frame(t: int) -> None:
            fp = _format_frame_path(stage_pattern, int(t) + 1)
            _write_binary_tiff_frame(np.asarray(mask_u8[int(t)]), fp)

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f"Writing binary TIFF sequence ({pattern_path.parent.name})",
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            pattern_path,
            total,
            stale_extensions=(pattern_path.suffix, '.tif', '.tiff'),
        )
    return pattern_path.parent

def write_semantic_png_sequence_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])
    with _staged_frame_sequence(pattern_path) as stage_pattern:
        def _write_frame(t: int) -> None:
            fp = _format_frame_path(stage_pattern, int(t) + 1)
            _write_semantic_png_frame(np.asarray(mask_u8[int(t)]), fp)

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f"Writing semantic PNG sequence ({pattern_path.parent.name})",
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            pattern_path,
            total,
            stale_extensions=(pattern_path.suffix, '.png'),
        )
    return pattern_path.parent

def write_binary_video_from_mask_volume(
    mask_u8: np.ndarray,
    video_path: Path,
    fps: float,
    show_progress: bool = True,
    scratch_dir: Optional[Path] = None,
    publication_root: Optional[Path] = None,
) -> Path:
    """Write a binary FFV1 MKV, optionally as concurrent losslessly concatenated t shards."""
    T, H, W = mask_u8.shape
    def _encode_range(
        start: int,
        stop: int,
        shard_path: Path,
        cancel_event: threading.Event,
        register_proc: Callable[[subprocess.Popen], None],
        unregister_proc: Callable[[subprocess.Popen], None],
        progress_one: Callable[[], None],
    ) -> None:
        proc = ffmpeg_ffv1_gray_writer(
            shard_path,
            width=W,
            height=H,
            fps=fps,
        )
        register_proc(proc)
        # One buffer per producer; the K producers only read the shared mask volume.
        frame_buf = np.empty((int(H), int(W)), dtype=np.uint8)
        primary_error: Optional[BaseException] = None
        primary_traceback = None
        try:
            assert proc.stdin is not None
            for t in range(int(start), int(stop)):
                if cancel_event.is_set():
                    break
                np.multiply(np.asarray(mask_u8[t]), 255, out=frame_buf)
                proc.stdin.write(memoryview(frame_buf).cast('B'))
                progress_one()
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__
        try:
            close_ffmpeg_writer(proc)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                primary_traceback = exc.__traceback__
        finally:
            unregister_proc(proc)
        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)

    _run_sharded_ffv1_encode(
        out_path=Path(video_path),
        total_frames=int(T),
        description=f"Writing binary MKV ({video_path.name})",
        show_progress=bool(show_progress),
        encode_range=_encode_range,
        scratch_dir=scratch_dir,
        publication_root=publication_root,
    )
    return video_path

def _nrrd_ascii_header_text(value: object) -> str:
    """Return an ASCII-safe representation for pynrrd header fields."""
    text = str(value)
    replacements = {
        '°': 'deg',
        '±': '+/-',
        'µ': 'u',
        '–': '-',
        '—': '-',
        '−': '-',
        '\u00a0': ' ',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def _nrrd_space_directions_matrix(
    spatial_axes: int = 3,
    list_axis: bool = False,
    list_axis_position: str = 'first',
) -> np.ndarray:
    """Return a NRRD ``space directions`` matrix with optional non-spatial list axis.

 Non-spatial axes are represented by a row of NaNs, which serializes as
 ``none``. ``list_axis_position='last'`` is used by decomposed segmentation
 NRRDs so the on-disk byte stream is one complete ``(t,Y,X)`` layer block
 followed by the next layer."""
    spatial_axes_i = max(1, int(spatial_axes))
    if bool(list_axis):
        position = str(list_axis_position).strip().lower()
        mat = np.full((spatial_axes_i + 1, spatial_axes_i), np.nan, dtype=np.float64)
        if position == 'last':
            mat[:spatial_axes_i, :] = np.eye(spatial_axes_i, dtype=np.float64)
        elif position == 'first':
            mat[1:, :] = np.eye(spatial_axes_i, dtype=np.float64)
        else:
            raise ValueError("list_axis_position must be 'first' or 'last'")
        return mat
    return np.eye(spatial_axes_i, dtype=np.float64)

def nrrd_slicer_header(mask_shape_zyx: Tuple[int, int, int]) -> Dict[str, object]:
    t_dim, h, w = (int(mask_shape_zyx[0]), int(mask_shape_zyx[1]), int(mask_shape_zyx[2]))
    return {
        "space": NRRD_SPACE,
        "kinds": ["domain", "domain", "domain"],
        "space directions": _nrrd_space_directions_matrix(spatial_axes=3, list_axis=False),
        "space origin": np.zeros((3,), dtype=np.float64),
        "content": f"binary segmentation mask; source_shape_tyx=({t_dim},{h},{w}); exported_axes=(X,Y,t)",
    }

_SLICER_SEGMENT_COLOR_PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.83, 0.16, 0.16),  # red
    (0.12, 0.47, 0.71),  # blue
    (1.00, 0.60, 0.07),  # orange
    (0.17, 0.63, 0.17),  # green
    (0.58, 0.40, 0.74),  # purple
    (0.09, 0.75, 0.81),  # cyan
    (0.89, 0.47, 0.76),  # pink
    (0.74, 0.74, 0.13),  # olive
    (0.55, 0.34, 0.29),  # brown
    (0.98, 0.75, 0.37),  # light orange
    (0.68, 0.78, 0.91),  # light blue
    (0.60, 0.87, 0.54),  # light green
    (1.00, 0.60, 0.59),  # salmon
    (0.77, 0.69, 0.84),  # lavender
    (0.62, 0.85, 0.90),  # pale cyan
    (0.95, 0.90, 0.45),  # yellow
    (0.78, 0.58, 0.45),  # tan
    (0.96, 0.71, 0.82),  # light pink
    (0.47, 0.05, 0.53),  # violet
    (0.10, 0.35, 0.20),  # dark green
)

def _stable_layer_color_index(token: str) -> int:
    """Deterministic FNV-1a hash of a layer token.

 Python's builtin hash is salted per process, which would shuffle segment colors
 between runs; the palette pick must be reproducible for the same layer suffix."""
    h = 0x811C9DC5
    for b in str(token).encode('utf-8', errors='ignore'):
        h = ((h ^ int(b)) * 0x01000193) & 0xFFFFFFFF
    return int(h)

def slicer_segment_palette_color(token: str) -> Tuple[float, float, float]:
    """Deterministic palette color for a layer token (no in-run collision probing)."""
    palette = _SLICER_SEGMENT_COLOR_PALETTE
    return palette[_stable_layer_color_index(str(token)) % len(palette)]

def _slicer_segment_name_for_out_path(out_path: Path) -> str:
    name = Path(out_path).name
    for ext in ('.seg.nrrd', '.nrrd'):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return Path(out_path).stem

def _slicer_segment_extent_for_output(
    ref: 'NrrdLayerRef',
    output_shape_tyx: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    """Map the layer's backing-store segment extent into the output geometry.

 Returns the inclusive Slicer (minX maxX minY maxY minT maxT) extent in the exported
 (X,Y,t) axis order. Extents are expanded outward when the output geometry is scaled
 (e.g. low-quality downbins) and fall back to the full output extent when the backing
 extent is unknown or empty, so the header never understates where mask voxels live."""
    out_t, out_h, out_w = (int(output_shape_tyx[0]), int(output_shape_tyx[1]), int(output_shape_tyx[2]))
    full: NrrdSegmentExtent = (0, max(0, out_w - 1), 0, max(0, out_h - 1), 0, max(0, out_t - 1))
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is None:
        return full
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return full
    in_t, in_h, in_w = (int(v) for v in getattr(ref, 'segment_extent_shape_tyx', (0, 0, 0)))
    if in_t <= 0 or in_h <= 0 or in_w <= 0:
        return full

    def _axis(lo: int, hi: int, n_in: int, n_out: int) -> Tuple[int, int]:
        if int(n_in) != int(n_out):
            scale = float(n_out) / float(n_in)
            lo = int(math.floor(float(lo) * scale))
            hi = int(math.ceil((float(hi) + 1.0) * scale)) - 1
        return (max(0, min(int(lo), int(n_out) - 1)), max(0, min(int(hi), int(n_out) - 1)))

    x_lo, x_hi = _axis(x0, x1, in_w, out_w)
    y_lo, y_hi = _axis(y0, y1, in_h, out_h)
    # The payload restore is center-aligned nearest-neighbour when temporal
    # upscaling and interval-OR when downscaling. Exact inversion keeps the
    # extent equal to the restored support, and Slicer may discard payload
    # outside Segment0_Extent.
    exact_t = _nrrd_output_z_extent_for_source_extent(
        int(in_t), int(out_t), int(t0), int(t1),
    )
    if exact_t is None:
        return full
    t_lo, t_hi = (int(exact_t[0]), int(exact_t[1]))
    return (x_lo, x_hi, y_lo, y_hi, t_lo, t_hi)

def _nrrd_output_z_extent_for_source_extent(
    in_t: int,
    out_t: int,
    source_z0: int,
    source_z1: int,
) -> Optional[Tuple[int, int]]:
    """Invert the exact payload restore map for a non-empty source-z interval.

 Temporal upscaling is center-aligned nearest-neighbour, while downscaling ORs a
 floor/ceil coverage range. Extent metadata uses the same mapping as the payload
 writer instead of a ratio-only scaling."""
    in_t_i = max(1, int(in_t))
    out_t_i = max(1, int(out_t))
    source_lo = max(0, min(int(source_z0), in_t_i - 1))
    source_hi = max(0, min(int(source_z1), in_t_i - 1))
    if source_hi < source_lo:
        return None
    # Every output z reads a nonempty source interval; the cached per-z bounds are the
    # restore map itself, so this mask selects exactly the z whose interval meets the
    # extent. Each sink layer plans several extents; the per-z Python walk held the
    # GIL for ~70 ms per full-resolution plan.
    first_source, last_source = _nrrd_restore_source_bounds(int(in_t_i), int(out_t_i))
    hits = np.flatnonzero((last_source >= int(source_lo)) & (first_source <= int(source_hi)))
    return None if hits.size == 0 else (int(hits[0]), int(hits[-1]))

@lru_cache(maxsize=64)
def _nrrd_restore_source_bounds(in_t: int, out_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """First and last source z that ``_restore_source_indices_for_output_z`` reads per output z."""
    first = np.empty(int(out_t), dtype=np.int64)
    last = np.empty(int(out_t), dtype=np.int64)
    for out_z in range(int(out_t)):
        source_indices = _restore_source_indices_for_output_z(int(in_t), int(out_t), int(out_z))
        first[out_z] = int(source_indices[0])
        last[out_z] = int(source_indices[-1])
    first.setflags(write=False)
    last.setflags(write=False)
    return first, last

@lru_cache(maxsize=32)
def _nrrd_mirror_targets_by_source(out_t: int, mirror_t: int) -> Tuple[Tuple[int, ...], ...]:
    """For each full-quality output z, the mirror slices whose restore interval contains it."""
    targets: List[List[int]] = [[] for _ in range(int(out_t))]
    for mz in range(int(mirror_t)):
        for sz in _restore_source_indices_for_output_z(int(out_t), int(mirror_t), int(mz)):
            if 0 <= int(sz) < int(out_t):
                targets[int(sz)].append(int(mz))
    return tuple(tuple(v) for v in targets)

NRRD_LAYER_LAYOUT_FULL = 'full'

NRRD_LAYER_LAYOUT_CROPPED = 'cropped'

def _normalized_nrrd_layer_layout(value: Optional[str]) -> str:
    """Validate a per-layer NRRD storage layout; ``None`` keeps the historical full raster."""
    if value is None:
        return NRRD_LAYER_LAYOUT_FULL
    text = str(value).strip().lower()
    if text in (NRRD_LAYER_LAYOUT_FULL, NRRD_LAYER_LAYOUT_CROPPED):
        return text
    raise ValueError(
        f'Unsupported NRRD layer layout {value!r}; expected '
        f'{NRRD_LAYER_LAYOUT_CROPPED!r} or {NRRD_LAYER_LAYOUT_FULL!r}.'
    )

@dataclass(frozen=True)
class NrrdStoredRasterPlan(NrrdRasterPlan):
    """Raster plan whose stored box is placed inside its reference grid.

    ``stored_offset_tyx`` is the reference index of the first stored voxel; it is
    (0, 0, 0) for the full layout. ``segment_extent_xyt`` is relative to the stored
    raster, matching Slicer's SegmentN_Extent convention for cropped files."""

    stored_offset_tyx: Tuple[int, int, int] = (0, 0, 0)
    reference_shape_tyx: Tuple[int, int, int] = (0, 0, 0)
    layout: str = NRRD_LAYER_LAYOUT_FULL

def _nrrd_plan_layout(plan: object) -> str:
    return str(getattr(plan, 'layout', NRRD_LAYER_LAYOUT_FULL))

def _nrrd_plan_offset(plan: object) -> Tuple[int, int, int]:
    offset = getattr(plan, 'stored_offset_tyx', (0, 0, 0))
    return (int(offset[0]), int(offset[1]), int(offset[2]))

def _nrrd_cropped_box_for_extent(
    segment_extent_xyt: NrrdSegmentExtent,
    reference_shape_tyx: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return ``(stored_shape_tyx, offset_tyx)`` for one nonempty reference extent.

    Slicer 5.12 aborts while loading a nonempty 1x1x1 cropped segmentation (a one-voxel
    image is its empty-segmentation convention), so an all-singleton box is widened by one
    neighbour voxel exactly like grouped_seg_nrrd._union_crop."""
    ref_t, ref_h, ref_w = (int(v) for v in reference_shape_tyx)
    sizes_xyt = (ref_w, ref_h, ref_t)
    lo = [int(segment_extent_xyt[2 * axis]) for axis in range(3)]
    hi = [int(segment_extent_xyt[2 * axis + 1]) for axis in range(3)]
    for axis in range(3):
        if not 0 <= lo[axis] <= hi[axis] < sizes_xyt[axis]:
            raise ValueError(
                f'NRRD segment extent {tuple(segment_extent_xyt)} lies outside the reference '
                f'grid (t,Y,X)={tuple(reference_shape_tyx)}'
            )
    if lo == hi:
        for axis, size in enumerate(sizes_xyt):
            if size > 1:
                if hi[axis] + 1 < size:
                    hi[axis] += 1
                else:
                    lo[axis] -= 1
                break
    stored_shape_tyx = (hi[2] - lo[2] + 1, hi[1] - lo[1] + 1, hi[0] - lo[0] + 1)
    return stored_shape_tyx, (lo[2], lo[1], lo[0])

def _nrrd_raster_plan(
    ref: 'NrrdLayerRef',
    output_shape_tyx: Tuple[int, int, int],
    *,
    layout: Optional[str] = None,
) -> NrrdRasterPlan:
    """Map a layer extent into its stored raster within the output reference grid.

    The full layout (default) stores the complete reference raster. The cropped layout
    stores the mapped Slicer segment extent, which bounds every payload voxel (the
    writer verifies this while streaming); an empty layer stores one zero voxel and an
    unknown extent stores the full grid."""
    out_t, out_h, out_w = (int(v) for v in output_shape_tyx)
    reference = (out_t, out_h, out_w)
    resolved_layout = _normalized_nrrd_layer_layout(layout)
    raw_extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    empty = bool(
        raw_extent is not None
        and (
            int(raw_extent[1]) < int(raw_extent[0])
            or int(raw_extent[3]) < int(raw_extent[2])
            or int(raw_extent[5]) < int(raw_extent[4])
        )
    )
    segment_extent = (
        _nrrd_empty_segment_extent()
        if bool(empty)
        else _slicer_segment_extent_for_output(ref, output_shape_tyx)
    )
    if resolved_layout == NRRD_LAYER_LAYOUT_FULL:
        return NrrdStoredRasterPlan(
            stored_shape_tyx=reference,
            segment_extent_xyt=segment_extent,
            empty_segment=bool(empty),
            stored_offset_tyx=(0, 0, 0),
            reference_shape_tyx=reference,
            layout=NRRD_LAYER_LAYOUT_FULL,
        )
    if bool(empty):
        return NrrdStoredRasterPlan(
            stored_shape_tyx=(1, 1, 1),
            segment_extent_xyt=_nrrd_empty_segment_extent(),
            empty_segment=True,
            stored_offset_tyx=(0, 0, 0),
            reference_shape_tyx=reference,
            layout=NRRD_LAYER_LAYOUT_CROPPED,
        )
    stored_shape, offset_tyx = _nrrd_cropped_box_for_extent(segment_extent, reference)
    off_t, off_y, off_x = offset_tyx
    local_extent: NrrdSegmentExtent = (
        int(segment_extent[0]) - off_x, int(segment_extent[1]) - off_x,
        int(segment_extent[2]) - off_y, int(segment_extent[3]) - off_y,
        int(segment_extent[4]) - off_t, int(segment_extent[5]) - off_t,
    )
    return NrrdStoredRasterPlan(
        stored_shape_tyx=stored_shape,
        segment_extent_xyt=local_extent,
        empty_segment=False,
        stored_offset_tyx=offset_tyx,
        reference_shape_tyx=reference,
        layout=NRRD_LAYER_LAYOUT_CROPPED,
    )

def _nrrd_layer_header(
    plan: object,
    reference_shape_tyx: Tuple[int, int, int],
    *,
    segment_name: str,
    color_rgb: Tuple[float, float, float],
) -> Dict[str, object]:
    """Complete single-layer Slicer header for one raster plan.

    Full-layout headers are byte-identical to the historical writer. Cropped headers
    place ``space origin`` at the first stored voxel and record Slicer's
    ``Segmentation_ReferenceImageExtentOffset`` plus the full reference extent."""
    ref_t, ref_h, ref_w = (int(v) for v in reference_shape_tyx)
    stored_t, stored_h, stored_w = (int(v) for v in getattr(plan, 'stored_shape_tyx'))
    cropped = _nrrd_plan_layout(plan) == NRRD_LAYER_LAYOUT_CROPPED
    off_t, off_y, off_x = _nrrd_plan_offset(plan)
    header = nrrd_slicer_header((stored_t, stored_h, stored_w))
    offset_note = f'stored_offset_tyx=({off_t},{off_y},{off_x}); ' if cropped else ''
    header['content'] = (
        'binary segmentation mask; '
        f'reference_shape_tyx=({ref_t},{ref_h},{ref_w}); '
        f'stored_shape_tyx=({stored_t},{stored_h},{stored_w}); '
        f'{offset_note}exported_axes=(X,Y,t)'
    )
    if cropped:
        directions = np.asarray(header['space directions'], dtype=np.float64)
        reference_origin = np.asarray(header['space origin'], dtype=np.float64)
        header['space origin'] = (
            reference_origin
            + np.asarray((off_x, off_y, off_t), dtype=np.float64) @ directions
        )
    header.update(slicer_segmentation_header_fields(
        segment_name=segment_name,
        color_rgb=color_rgb,
        extent_xyt=getattr(plan, 'segment_extent_xyt'),
    ))
    if cropped:
        header['Segmentation_ReferenceImageExtentOffset'] = f'{off_x} {off_y} {off_t}'
        header['Segmentation_ReferenceImageExtent'] = (
            f'0 {ref_w - 1} 0 {ref_h - 1} 0 {ref_t - 1}'
        )
    return header

class NrrdCropExtentViolation(RuntimeError):
    """Nonzero layer payload was found outside the planned stored box."""

_NRRD_CROP_SMALL_ZERO_RUN_BYTES = 64 * 1024

class _CroppedNrrdPayloadWriter:
    """Forward one stored box of a reference-geometry C-order ``(t,Y,X)`` uint8 stream.

    Producers keep emitting the reference stream (data, cached zero runs and
    whole-slice members) exactly as for the full layout. Every byte position is mapped
    into the box: zero runs collapse to their in-box length in O(1), data contributes
    only its in-box rows/columns, and any nonzero byte outside the box raises
    NrrdCropExtentViolation so the caller can republish the historical full raster.
    The stream must be consumed by one writer thread at a time, like the inner writer."""

    def __init__(
        self,
        inner: object,
        reference_shape_tyx: Tuple[int, int, int],
        offset_tyx: Tuple[int, int, int],
        stored_shape_tyx: Tuple[int, int, int],
        *,
        logical_start: int = 0,
        logical_stop: Optional[int] = None,
        label: str = '',
    ) -> None:
        ref_t, ref_h, ref_w = (int(v) for v in reference_shape_tyx)
        off_t, off_y, off_x = (int(v) for v in offset_tyx)
        stored_t, stored_h, stored_w = (int(v) for v in stored_shape_tyx)
        if (
            min(ref_t, ref_h, ref_w, stored_t, stored_h, stored_w) <= 0
            or min(off_t, off_y, off_x) < 0
            or off_t + stored_t > ref_t or off_y + stored_h > ref_h or off_x + stored_w > ref_w
        ):
            raise ValueError(
                f'Stored box offset={tuple(offset_tyx)} shape={tuple(stored_shape_tyx)} lies '
                f'outside reference {tuple(reference_shape_tyx)}'
            )
        self.inner = inner
        self.label = str(label)
        self.minimum_input_bytes = max(1, int(getattr(inner, 'minimum_input_bytes', 1)))
        self._height, self._width = ref_h, ref_w
        self._plane = ref_h * ref_w
        self._t0, self._t1 = off_t, off_t + stored_t
        self._y0, self._y1 = off_y, off_y + stored_h
        self._x0, self._x1 = off_x, off_x + stored_w
        self._box_plane = stored_h * stored_w
        total = ref_t * self._plane
        self._start = int(logical_start)
        self._stop = total if logical_stop is None else int(logical_stop)
        if not 0 <= self._start <= self._stop <= total:
            raise ValueError('Cropped NRRD stream range lies outside the reference payload')
        self._position = self._start
        self._expected_stored = self._box_bytes_before(self._stop) - self._box_bytes_before(self._start)
        self._stored = 0
        self._zero_pending = 0
        # Pending in-box data as owned contiguous pieces; flushed as one owned member.
        self._data_parts: List[np.ndarray] = []
        self._data_bytes = 0
        self._queued_nonzero = 0
        self._source_owned = False
        chunk = int(getattr(inner, 'chunk_bytes', 0) or nrrd_compression.nrrd_gzip_chunk_bytes())
        self._buffer_limit = int(np.clip(chunk, 1024 * 1024, 16 * 1024 * 1024))
        canonical = getattr(inner, 'write_canonical_zeros', None)
        plain_zeros = getattr(inner, 'write_zeros', None)
        if callable(canonical) and self.minimum_input_bytes == 1:
            self._inner_zeros = canonical
        elif callable(plain_zeros):
            self._inner_zeros = plain_zeros
        else:
            self._inner_zeros = self._write_literal_zeros
        self._finished = False

    def _write_literal_zeros(self, nbytes: int) -> int:
        remaining = int(nbytes)
        block = bytes(min(remaining, 1024 * 1024))
        while remaining > 0:
            step = min(remaining, len(block))
            self.inner.write(memoryview(block)[:step])
            remaining -= step
        return int(nbytes)

    def _box_bytes_before(self, position: int) -> int:
        """Number of stored-box bytes at reference stream positions below ``position``."""
        t, rest = divmod(int(position), self._plane)
        if t < self._t0:
            return 0
        if t >= self._t1:
            return (self._t1 - self._t0) * self._box_plane
        count = (t - self._t0) * self._box_plane
        y, x = divmod(rest, self._width)
        if y < self._y0:
            return count
        box_w = self._x1 - self._x0
        if y >= self._y1:
            return count + self._box_plane
        return count + (y - self._y0) * box_w + min(max(x - self._x0, 0), box_w)

    def _raise_outside_foreground(self) -> None:
        raise NrrdCropExtentViolation(
            f'NRRD layer {self.label or "<unnamed>"} has foreground outside its planned '
            f'box t[{self._t0},{self._t1}) y[{self._y0},{self._y1}) x[{self._x0},{self._x1})'
        )

    def _write_owned(self, owned: np.ndarray) -> None:
        """Hand one exclusively owned contiguous 1-D buffer to the inner writer.

        Ownership transfer avoids a second detachment copy; pieces stay at most one
        buffer limit so large in-box blocks still spread across compressor workers."""
        owned_writer = getattr(self.inner, 'write_owned_known_nonzero', None)
        step = int(self._buffer_limit)
        for start in range(0, int(owned.size), step):
            piece = owned[start:start + step]
            if callable(owned_writer):
                owned_writer(piece)
            else:
                self.inner.write(memoryview(piece))

    def _flush_data(self) -> None:
        if self._data_parts:
            parts, self._data_parts, self._data_bytes = self._data_parts, [], 0
            self._write_owned(parts[0] if len(parts) == 1 else np.concatenate(parts))

    def _flush_zeros(self) -> None:
        if self._zero_pending:
            pending, self._zero_pending = self._zero_pending, 0
            self._inner_zeros(int(pending))

    def _queue_zeros(self, nbytes: int) -> None:
        count = int(nbytes)
        if count <= 0:
            return
        self._stored += count
        if self._data_parts and count <= _NRRD_CROP_SMALL_ZERO_RUN_BYTES:
            # A short in-box gap between data rows compresses inline more cheaply than
            # an extra complete gzip member.
            self._data_parts.append(np.zeros(count, dtype=np.uint8))
            self._data_bytes += count
            if self._data_bytes >= self._buffer_limit:
                self._flush_data()
            return
        self._flush_data()
        self._zero_pending += count

    def _queue_data(self, values: np.ndarray) -> None:
        count = int(values.size)
        if count <= 0:
            return
        self._stored += count
        self._flush_zeros()
        if self._source_owned and values.flags.c_contiguous:
            # The producer transferred this buffer (write_owned_known_nonzero) and the
            # in-box piece is contiguous: hand the view on without copying.
            owned = values.reshape(-1)
        else:
            # Exactly one copy: an ordinary writer may reuse its block, and the box is
            # usually a strided view, so the compressor retains an owned contiguous copy.
            owned = np.array(values, dtype=np.uint8, order='C', copy=True).reshape(-1)
        self._queued_nonzero += int(np.count_nonzero(owned))
        if count >= self._buffer_limit:
            self._flush_data()
            self._write_owned(owned)
            return
        self._data_parts.append(owned)
        self._data_bytes += count
        if self._data_bytes >= self._buffer_limit:
            self._flush_data()

    # Routing only extracts in-box pieces. write() proves nothing was dropped by
    # comparing two fast contiguous nonzero counts instead of scanning strided
    # out-of-box strips.
    def _route_row_piece(self, y: int, x: int, piece: np.ndarray) -> None:
        if y < self._y0 or y >= self._y1:
            return
        lo, hi = max(int(x), self._x0), min(int(x) + int(piece.size), self._x1)
        if hi > lo:
            self._queue_data(piece[lo - int(x):hi - int(x)])

    def _route_rows(self, y: int, rows: np.ndarray) -> None:
        lo, hi = max(int(y), self._y0), min(int(y) + int(rows.shape[0]), self._y1)
        if hi > lo:
            self._queue_data(rows[lo - int(y):hi - int(y), self._x0:self._x1])

    def _route_planes(self, t: int, planes: np.ndarray) -> None:
        lo, hi = max(int(t), self._t0), min(int(t) + int(planes.shape[0]), self._t1)
        if hi > lo:
            self._queue_data(planes[lo - int(t):hi - int(t), self._y0:self._y1, self._x0:self._x1])

    def _route_partial_plane(self, t: int, rest: int, values: np.ndarray) -> None:
        if t < self._t0 or t >= self._t1:
            return
        width = self._width
        y, x = divmod(int(rest), width)
        index = 0
        total = int(values.size)
        if x:
            take = min(total, width - x)
            self._route_row_piece(y, x, values[:take])
            index, y = take, y + 1
        rows = (total - index) // width
        if rows:
            self._route_rows(y, values[index:index + rows * width].reshape(rows, width))
            index += rows * width
            y += rows
        if index < total:
            self._route_row_piece(y, 0, values[index:])

    def write(self, data: object) -> int:
        return self._write_reference_bytes(data, owned=False)

    def _write_reference_bytes(self, data: object, *, owned: bool) -> int:
        if self._finished:
            raise RuntimeError('Cannot write to a finished cropped NRRD payload stream')
        self._source_owned = bool(owned)
        if isinstance(data, np.ndarray):
            values = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
        else:
            values = np.frombuffer(memoryview(data).cast('B'), dtype=np.uint8)
        total = int(values.size)
        start = int(self._position)
        if start + total > self._stop:
            raise ValueError('Cropped NRRD payload stream received bytes beyond its declared range')
        plane = self._plane
        queued_before = int(self._queued_nonzero)
        index = 0
        while index < total:
            t, rest = divmod(start + index, plane)
            if rest == 0 and total - index >= plane:
                count = (total - index) // plane
                self._route_planes(
                    t, values[index:index + count * plane].reshape(count, self._height, self._width),
                )
                index += count * plane
                continue
            take = min(total - index, plane - rest)
            self._route_partial_plane(t, rest, values[index:index + take])
            index += take
        if int(np.count_nonzero(values)) != int(self._queued_nonzero) - queued_before:
            self._raise_outside_foreground()
        self._position = start + total
        return total

    def write_known_nonzero(self, data: object) -> int:
        return self.write(data)

    def write_owned_known_nonzero(self, data: object) -> int:
        return self._write_reference_bytes(data, owned=True)

    def try_write_crop_span_batch(
        self,
        data: np.ndarray,
        segments: Sequence[Tuple[int, int, int]],
        records: Sequence[Tuple[int, int, int, int, int, int]],
    ) -> Optional[int]:
        """Route validated bbox bands to the stored raster without rescanning pixels."""
        write_batch = getattr(self.inner, 'write_owned_member_batch', None)
        if not callable(write_batch) or any(
            not (self._t0 <= z < self._t1 and self._y0 <= y0 < y1 <= self._y1
                 and self._x0 <= x0 < x1 <= self._x1)
            for _, y0, x0, y1, x1, z in records
        ):
            return None
        if self._finished:
            raise RuntimeError('Cannot write to a finished cropped NRRD payload stream')
        position = self._position
        box_width = self._x1 - self._x0
        cropped_segments = []
        for zeros, first, stop in segments:
            data_start = position + zeros
            box_zeros = self._box_bytes_before(data_start) - self._box_bytes_before(position)
            cropped_segments.append(
                (box_zeros, first // self._width * box_width, stop // self._width * box_width),
            )
            position = data_start + stop - first
        if position > self._stop:
            raise ValueError('Cropped NRRD payload stream received bytes beyond its declared range')
        cropped = np.ascontiguousarray(data[:, self._x0:self._x1])
        self._flush_data()
        self._flush_zeros()
        self._stored += int(write_batch(cropped, cropped_segments))
        self._position = position
        return int(cropped.size)

    def write_zeros(self, nbytes: int) -> int:
        if self._finished:
            raise RuntimeError('Cannot write to a finished cropped NRRD payload stream')
        count = int(nbytes)
        if count < 0:
            raise ValueError('A zero run cannot have negative length')
        start = int(self._position)
        if start + count > self._stop:
            raise ValueError('Cropped NRRD payload stream received zeros beyond its declared range')
        self._queue_zeros(self._box_bytes_before(start + count) - self._box_bytes_before(start))
        self._position = start + count
        return count

    def write_aligned_zeros(self, nbytes: int) -> int:
        return self.write_zeros(nbytes)

    def write_canonical_zeros(self, nbytes: int) -> int:
        return self.write_zeros(nbytes)

    def finish(self) -> int:
        """Flush pending bytes and prove the whole declared reference range arrived."""
        if self._finished:
            return int(self._stored)
        if int(self._position) != int(self._stop) or int(self._stored) != int(self._expected_stored):
            raise RuntimeError(
                f'Cropped NRRD payload {self.label or "<unnamed>"} is incomplete: '
                f'{self._position - self._start} of {self._stop - self._start} reference bytes, '
                f'{self._stored} of {self._expected_stored} stored bytes'
            )
        self._flush_data()
        self._flush_zeros()
        self._finished = True
        telemetry = runtime_telemetry()
        telemetry.add('nrrd.layer_crop.reference_bytes', int(self._stop - self._start))
        telemetry.add('nrrd.layer_crop.stored_bytes', int(self._stored))
        return int(self._stored)

def _nrrd_payload_writer_for_plan(
    payload_writer: object,
    plan: object,
    reference_shape_tyx: Tuple[int, int, int],
    z_start: int,
    z_stop: int,
    *,
    label: str = '',
) -> object:
    """Return the inner writer for full plans, or a box-cropping adapter for cropped plans."""
    if _nrrd_plan_layout(plan) != NRRD_LAYER_LAYOUT_CROPPED:
        return payload_writer
    ref_t, ref_h, ref_w = (int(v) for v in reference_shape_tyx)
    plane = ref_h * ref_w
    return _CroppedNrrdPayloadWriter(
        payload_writer,
        (ref_t, ref_h, ref_w),
        _nrrd_plan_offset(plan),
        tuple(int(v) for v in getattr(plan, 'stored_shape_tyx')),
        logical_start=int(z_start) * plane,
        logical_stop=int(z_stop) * plane,
        label=label,
    )

def _record_nrrd_stored_plan(entry: Dict[str, object], plan: object, sink_layout: str) -> None:
    """Write one plan's stored-raster fields into a manifest row.

    Full-layout sinks keep the historical row keys exactly. Cropped sinks also record
    ``stored_offset_tyx`` (reference index of the first stored voxel) so XTA readers can
    place the box without parsing headers, plus the per-file ``stored_layout``."""
    entry['stored_shape_tyx'] = [int(v) for v in getattr(plan, 'stored_shape_tyx')]
    entry['empty_segment'] = bool(getattr(plan, 'empty_segment', False))
    if str(sink_layout) == NRRD_LAYER_LAYOUT_CROPPED:
        entry['stored_offset_tyx'] = [int(v) for v in _nrrd_plan_offset(plan)]
        entry['stored_layout'] = (
            'segment_bounding_box'
            if _nrrd_plan_layout(plan) == NRRD_LAYER_LAYOUT_CROPPED else 'full_reference'
        )

def _finish_nrrd_payload_writer(writer: object) -> None:
    if isinstance(writer, _CroppedNrrdPayloadWriter):
        writer.finish()

def _nrrd_cropped_zshard_bands(
    ref: 'NrrdLayerRef',
    out_t: int,
    box_t0: int,
    box_t1: int,
    shard_count: int,
) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
    """Partition the reference depth so every band owns part of the stored box.

    Indexed cvol weights already ignore empty slices. Other sources split the box depth
    evenly; the first/last bands absorb the (zero-skipped) slices outside the box."""
    depth = max(0, int(out_t))
    box_depth = max(1, int(box_t1) - int(box_t0))
    count = max(1, min(int(shard_count), box_depth))
    weighted, weights = _nrrd_layer_zshard_bands(ref, depth, count)
    if weights is not None and sum(int(w) for w in weights) > 0:
        return weighted, weights
    bands = [
        (int(box_t0) + i * box_depth // count, int(box_t0) + (i + 1) * box_depth // count)
        for i in range(count)
    ]
    bands[0] = (0, bands[0][1])
    bands[-1] = (bands[-1][0], depth)
    return bands, None

def _nrrd_mapped_extent_preserve_empty(
    ref: 'NrrdLayerRef', output_shape_tyx: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is not None and (
        int(extent[1]) < int(extent[0])
        or int(extent[3]) < int(extent[2])
        or int(extent[5]) < int(extent[4])
    ):
        return _nrrd_empty_segment_extent()
    return _slicer_segment_extent_for_output(ref, output_shape_tyx)

def slicer_segmentation_header_fields(
    *,
    segment_name: str,
    color_rgb: Tuple[float, float, float],
    extent_xyt: NrrdSegmentExtent,
) -> Dict[str, object]:
    """3D Slicer.seg.nrrd custom header fields for one single-segment binary labelmap.

 All keys are non-standard NRRD fields, so _write_nrrd_ascii_header emits them with the
 key:=value separator Slicer expects. Segment0_LabelValue matches the uint8 mask value 1."""
    r, g, b = (float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
    return {
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_SourceRepresentation': 'Binary labelmap',
        'Segment0_ID': 'Segment_1',
        'Segment0_Name': str(segment_name),
        'Segment0_NameAutoGenerated': '0',
        'Segment0_Color': f'{r:.6g} {g:.6g} {b:.6g}',
        'Segment0_ColorAutoGenerated': '0',
        'Segment0_LabelValue': '1',
        'Segment0_Layer': '0',
        'Segment0_Extent': ' '.join(str(int(v)) for v in extent_xyt),
        'Segment0_Tags': (
            'TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list'
            '~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'
        ),
    }

def nrrd_z_chunk_cap() -> int:
    """Cap on one NRRD payload block, in output t-slices.

 The RAM-scaled buffer previously resolved to the WHOLE payload on large-RAM nodes, so the
 writer filled an ~payload-sized block before compression began. Bounded
 blocks pipeline parallel fill with the selected gzip backend; two blocks
 per in-flight write are resident (double buffering)."""
    return max(1, _env_int('YOLO_TTA_NRRD_Z_CHUNK_SLICES', 128))

def nrrd_fill_workers() -> int:
    """Return the process-wide worker count used to fill NRRD payload blocks.

    Raw-bbox decoding and OpenCV restores release the GIL. The bounded shared pool prevents
    concurrent layer writers from multiplying memory-scanning threads.
    """
    return max(2, _env_int('YOLO_TTA_NRRD_FILL_WORKERS', max(2, min(32, _cpu_count() // 4))))

_NRRD_FILL_EXECUTOR: Optional[ThreadPoolExecutor] = None

_NRRD_FILL_EXECUTOR_LOCK = threading.Lock()

def _nrrd_fill_executor() -> ThreadPoolExecutor:
    global _NRRD_FILL_EXECUTOR
    with _NRRD_FILL_EXECUTOR_LOCK:
        if _NRRD_FILL_EXECUTOR is None:
            _NRRD_FILL_EXECUTOR = ThreadPoolExecutor(
                max_workers=int(nrrd_fill_workers()),
                thread_name_prefix='nrrd-fill',
            )
        return _NRRD_FILL_EXECUTOR

def _nrrd_parallel_fill_indices(
    count: int,
    func: Callable[[int], None],
    *,
    requested_workers: int,
) -> None:
    total = max(0, int(count))
    lanes = max(1, min(int(requested_workers), int(total))) if total > 0 else 0
    if lanes <= 1:
        for idx in range(total):
            func(int(idx))
        return
    # At most one range per requested lane prevents one sink writer from flooding the
    # shared FIFO with thousands of tiny tasks while preserving GIL-releasing slice work.
    chunk = max(1, int(math.ceil(float(total) / float(lanes))))

    def _run(start: int, stop: int) -> None:
        for idx in range(int(start), int(stop)):
            func(int(idx))

    executor = _nrrd_fill_executor()
    futures = [
        executor.submit(_run, int(start), int(min(total, start + chunk)))
        for start in range(0, total, chunk)
    ]
    first_error: Optional[BaseException] = None
    for fut in futures:
        try:
            fut.result()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error

def nrrd_stream_buffer_bytes(required_bytes: Optional[int] = None) -> int:
    """Resolve the RAM budget for one NRRD payload slab.
    
    An explicit environment limit wins; otherwise the budget is a bounded fraction of current anonymous-memory headroom."""
    explicit = os.environ.get('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', '').strip()
    if explicit:
        mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', 4096))
        target = int(mib) * 1024 * 1024
    else:
        min_mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIN_MIB', 4096))
        reserve_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_RESERVE_GIB', 192.0))
        max_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_MAX_GIB', 384.0))
        fraction = min(0.90, max(0.01, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_FRACTION', 0.35)))
        avail = int(available_anon_work_bytes())
        usable = max(0, int(avail) - int(reserve_gib * GIB))
        target = int(max(int(min_mib) * 1024 * 1024, min(float(max_gib) * float(GIB), float(usable) * float(fraction))))
    if required_bytes is not None and int(required_bytes) > 0:
        target = min(int(target), int(required_bytes))
    return max(1, int(target))

def _nrrd_full_slice_z_chunk(layer_count: int, width: int, height: int, depth: int) -> int:
    full_slice_bytes = max(1, int(layer_count) * int(width) * int(height) * np.dtype(np.uint8).itemsize)
    full_payload_bytes = full_slice_bytes * max(1, int(depth))
    target = nrrd_stream_buffer_bytes(full_payload_bytes)
    if target < full_slice_bytes:
        return 1
    # current behavior: bound the block so parallel
    # fill and the selected gzip backend pipeline instead of materializing the
    # entire payload before compression starts.
    return max(1, min(int(depth), int(target // full_slice_bytes), int(nrrd_z_chunk_cap())))

def nrrd_madvise_dontneed_interval() -> int:
    # projected cvol pages are reread by the final union. Run 126080
    # spent ~40 s on its first completion frontier and stalled again at z≈1280 after
    # concurrent NRRD readers had discarded those pages. Keep them warm by default;
    # memory-constrained jobs can restore the former interval with an explicit value.
    return max(0, _env_int('YOLO_TTA_NRRD_MADVISE_DONTNEED_INTERVAL', 0))

def _madvise_array_mmap(arr: object, advice_name: str) -> None:
    try:
        import mmap as _mmap_module  # local import so non-POSIX platforms remain unaffected
        advice = getattr(_mmap_module, str(advice_name), None)
        if advice is None:
            return
        mmap_obj = getattr(arr, '_mmap', None)
        if mmap_obj is None:
            base = getattr(arr, 'base', None)
            mmap_obj = getattr(base, '_mmap', None)
        madvise_fn = getattr(mmap_obj, 'madvise', None)
        if callable(madvise_fn):
            _madvise_mmap_traced(mmap_obj, advice, source='nrrd.write')
    except Exception:
        pass

def _nrrd_float_text(value: float) -> str:
    value_f = float(value)
    if math.isnan(value_f):
        return 'none'
    if math.isinf(value_f):
        return 'inf' if value_f > 0 else '-inf'
    return f'{value_f:.17g}'

def _nrrd_vector_text(values: Sequence[object]) -> str:
    return '(' + ','.join(_nrrd_float_text(float(v)) for v in values) + ')'

def _nrrd_space_directions_text(value: object) -> str:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return _nrrd_vector_text(arr.tolist())
    if arr.ndim != 2:
        raise ValueError(f'NRRD space directions must be 1D or 2D, got shape {arr.shape}')
    parts: List[str] = []
    for row in arr:
        if np.all(np.isnan(row)):
            parts.append('none')
        else:
            parts.append(_nrrd_vector_text(row.tolist()))
    return ' '.join(parts)

def _nrrd_header_value_text(key: str, value: object) -> str:
    key_l = str(key).strip().lower()
    if key_l == 'space directions':
        return _nrrd_space_directions_text(value)
    if key_l in {'space origin', 'measurement frame'}:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            return _nrrd_vector_text(arr.tolist())
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 1:
            return ' '.join(_nrrd_float_text(float(v)) for v in arr.tolist())
        return ' '.join(_nrrd_vector_text(row) for row in arr.tolist())
    if isinstance(value, (list, tuple)):
        return ' '.join(_nrrd_ascii_header_text(v) for v in value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _nrrd_ascii_header_text(value).replace('\n', ' ').replace('\r', ' ')

_NRRD_STANDARD_FIELDS = {
    'type', 'dimension', 'sizes', 'spacings', 'thicknesses', 'axis mins', 'axis maxs',
    'centers', 'labels', 'units', 'min', 'max', 'old min', 'old max', 'endian',
    'encoding', 'line skip', 'byte skip', 'data file', 'content', 'sample units',
    'space', 'space dimension', 'space units', 'space origin', 'space directions',
    'measurement frame', 'kinds', 'block size',
}

def _nrrd_field_separator(key: str) -> str:
    return ':' if str(key).strip().lower() in _NRRD_STANDARD_FIELDS else ':='

def _write_nrrd_ascii_header(
    fh: object,
    *,
    header: Dict[str, object],
    sizes: Sequence[int],
    dimension: int,
    data_type: str = 'uint8',
    encoding: str = 'gzip',
) -> None:
    header_copy: Dict[str, object] = dict(header)
    for reserved in ('type', 'dimension', 'sizes'):
        header_copy.pop(reserved, None)
    header_copy['encoding'] = str(encoding)

    standard_order = [
        'space', 'space dimension', 'kinds', 'space directions', 'space origin',
        'measurement frame', 'content', 'encoding', 'endian',
    ]

    lines: List[str] = [
        'NRRD0005',
        f'# Complete NRRD file generated by {SCRIPT_BASENAME}',
        f'type: {str(data_type)}',
        f'dimension: {int(dimension)}',
        'sizes: ' + ' '.join(str(int(v)) for v in sizes),
    ]

    emitted: set[str] = set()
    for key in standard_order:
        if key in header_copy:
            lines.append(f'{key}: {_nrrd_header_value_text(key, header_copy[key])}')
            emitted.add(key)

    for key, value in header_copy.items():
        if key in emitted or key in {'type', 'dimension', 'sizes'}:
            continue
        sep = _nrrd_field_separator(str(key))
        value_text = _nrrd_header_value_text(str(key), value)
        if sep == ':=':
            lines.append(f'{str(key)}{sep}{value_text}')
        else:
            lines.append(f'{str(key)}{sep} {value_text}')

    text = '\n'.join(lines) + '\n\n'
    fh.write(text.encode('ascii', errors='ignore'))

def nrrd_layer_file_basename(stem: str, unique_suffix: str) -> str:
    """Bound physical layer names while retaining full labels in metadata."""
    logical_name = f'{stem}_{unique_suffix}'
    extension = '.seg.nrrd'
    ordinary = logical_name + extension
    if len(ordinary) <= 120:
        return ordinary
    import hashlib
    digest = hashlib.sha256(logical_name.encode('utf-8')).hexdigest()
    prefix_budget = 120 - len(extension) - len(digest) - 2
    prefix = logical_name[:prefix_budget].rstrip(' ._-')
    return f'{prefix}__{digest}{extension}'


def nrrd_layer_output_suffix(
    *,
    view_token: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    interpolation_walk_back_index: int = 0,
    interpolation_candidate_index: int = 0,
    interpolation_backend: str = '',
    interpolation_direction: str = '',
    seed_detector_identity: str = '',
    sam_bundle_identity: str = '',
    interpolation_policy_identity: str = '',
    tile_config_id: str = '',
    tile_acceptance: str = '',
    stage: str = '',
) -> str:
    """Return the single-layer NRRD filename suffix for one component layer.

 Each component layer is written as one single-layer NRRD named
 ``{Filestem}_{suffix}.seg.nrrd``; the suffix carries no model name."""
    source_l = str(source).strip().lower()
    mask_kind_l = str(mask_kind).strip().lower()
    stage_l = str(stage).strip().lower()
    if source_l == 'global':
        if stage_l.startswith('final_output'):
            return 'Global_final_output'
        if mask_kind_l == 'smoothing_result':
            return f'Global_smoothing_pass{int(pass_index):02d}'
        return 'Global_union_presmoothing'
    vt = _sanitize_nrrd_layer_token(view_token) or 'view'
    if mask_kind_l == 'extrapolation':
        direction = str(stage).removeprefix('sam_extrapolation_')
        if direction not in {'forward', 'backward'} or source_l not in {'fullframe', 'tile'}:
            raise ValueError('SAM extrapolation output requires a direction and native source scope')
        config = _sanitize_nrrd_layer_token(tile_config_id) if tile_config_id else ''
        return f'{vt}_{source_l}{"_" + config if config else ""}_sam_extrapolation_{direction}'
    if mask_kind_l == 'bridge' and str(interpolation_backend).lower() == 'sam':
        direction = str(interpolation_direction).strip().lower()
        if direction not in {'forward', 'backward'}:
            raise ValueError('SAM bridge output direction must be forward or backward')
        if source_l not in {'fullframe', 'tile'}:
            raise ValueError('SAM bridge output requires fullframe or tile provenance')
        config = _sanitize_nrrd_layer_token(tile_config_id) if tile_config_id else ''
        scope = source_l + (f'_{config}' if config else '')
        # Human-readable native scope plus short stable identities, with full
        # values in the manifest. Directions/passes still share the same vote cap.
        import hashlib
        identities = []
        for label, identity in (('detector', seed_detector_identity),
                                ('bundle', sam_bundle_identity),
                                ('policy', interpolation_policy_identity)):
            if identity:
                token = hashlib.sha256(str(identity).encode('utf-8')).hexdigest()[:12]
                identities.append(f'{label}{token}')
        provenance = ('_' + '_'.join(identities)) if identities else ''
        return f'{vt}_{scope}_sam_bridge_pass{int(pass_index):02d}_{direction}{provenance}'
    if source_l == 'fullframe':
        if mask_kind_l == 'bridge':
            suffix = f'{vt}_fullframe_bridge_pass{int(pass_index):02d}'
            if int(interpolation_walk_back_index) > 0 and int(interpolation_candidate_index) > 0:
                suffix += (
                    f'_walkback{int(interpolation_walk_back_index):02d}'
                    f'_candidate{int(interpolation_candidate_index):02d}'
                )
            return suffix
        return f'{vt}_fullframe_yolo'
    if source_l == 'tile':
        config = _sanitize_nrrd_layer_token(tile_config_id) if tile_config_id else ''
        config_part = f'_{config}' if config else ''
        if mask_kind_l == 'bridge':
            suffix = f'{vt}_tile{config_part}_bridge_pass{int(pass_index):02d}'
            if int(interpolation_walk_back_index) > 0 and int(interpolation_candidate_index) > 0:
                suffix += (
                    f'_walkback{int(interpolation_walk_back_index):02d}'
                    f'_candidate{int(interpolation_candidate_index):02d}'
                )
            return suffix
        acceptance = _sanitize_nrrd_layer_token(tile_acceptance) or 'parent_support'
        return f'{vt}_tile{config_part}_yolo_{acceptance}'
    parts = [vt, source_l or 'layer', mask_kind_l or 'mask']
    if int(pass_index) > 0:
        parts.append(f'pass{int(pass_index):02d}')
    if tile_acceptance:
        parts.append(_sanitize_nrrd_layer_token(tile_acceptance))
    return '_'.join(p for p in parts if p)

def nrrd_layer_zshards_requested() -> int:
    """Requested per-global-layer z shards (0=auto, 1=disabled, >1=cap)."""
    return max(0, _env_int('YOLO_TTA_NRRD_LAYER_ZSHARDS', 0))

def nrrd_layer_zshard_min_slices() -> int:
    return max(1, _env_int('YOLO_TTA_NRRD_LAYER_ZSHARD_MIN_SLICES', 256))

_NRRD_ZSHARD_SEMAPHORE_LOCK = threading.Lock()

_NRRD_ZSHARD_SEMAPHORES: Dict[int, object] = {}

def _nrrd_zshard_capacity() -> int:
    """Process-wide concurrent band capacity (also the semaphore capacity)."""
    return max(1, int(nrrd_layer_sink_workers()))

class _WeightedNrrdZShardSemaphore:
    """Bound concurrent NRRD z-band compression with weighted process-wide permits.
    
    Ordinary independent spools consume one permit each."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.available = int(self.capacity)
        self.condition = threading.Condition()

    def acquire(self, weight: int) -> None:
        requested = max(1, int(weight))
        if requested > int(self.capacity):
            raise ValueError(
                f'NRRD z-shard reservation {requested} exceeds capacity {self.capacity}'
            )
        with self.condition:
            while int(self.available) < int(requested):
                self.condition.wait()
            self.available -= int(requested)

    def release(self, weight: int) -> None:
        returned = max(1, int(weight))
        with self.condition:
            if int(self.available) + int(returned) > int(self.capacity):
                raise RuntimeError('NRRD z-shard permit pool released too many permits')
            self.available += int(returned)
            self.condition.notify_all()

def _nrrd_zshard_semaphore() -> _WeightedNrrdZShardSemaphore:
    """Return the atomic weighted band budget for the configured sink capacity."""
    capacity = int(_nrrd_zshard_capacity())
    with _NRRD_ZSHARD_SEMAPHORE_LOCK:
        sem = _NRRD_ZSHARD_SEMAPHORES.get(int(capacity))
        if sem is None:
            sem = _WeightedNrrdZShardSemaphore(int(capacity))
            _NRRD_ZSHARD_SEMAPHORES[int(capacity)] = sem
        return sem  # type: ignore[return-value]

def _nrrd_layer_zshard_count(
    ref: 'NrrdLayerRef',
    out_t: int,
    capacity: Optional[int] = None,
) -> int:
    """Shard count resolved by the executing writer, never sink queue occupancy."""
    if str(getattr(ref, 'source', '')).strip().lower() != 'global':
        return 1
    min_slices = int(nrrd_layer_zshard_min_slices())
    by_depth = max(1, int(math.ceil(float(max(0, int(out_t))) / float(min_slices))))
    requested = int(nrrd_layer_zshards_requested())
    if requested == 1:
        return 1
    band_capacity = int(_nrrd_zshard_capacity() if capacity is None else max(1, int(capacity)))
    cap = int(band_capacity) if requested == 0 else min(int(band_capacity), requested)
    return max(1, min(int(out_t), int(by_depth), max(1, int(cap))))

def _nrrd_layer_zshard_bands(
    ref: 'NrrdLayerRef',
    out_t: int,
    shard_count: int,
) -> Tuple[List[Tuple[int, int]], Optional[List[int]]]:
    """Partition native cvol layers by indexed sparse payload bytes.

 Equal slice bands remain the conservative fallback for resampled/non-cvol layers.
 The optional second result reports each band's indexed payload weight for logging."""
    depth = max(0, int(out_t))
    count = max(1, min(int(shard_count), max(1, depth)))
    equal = [
        (int(i * depth // count), int((i + 1) * depth // count))
        for i in range(count)
    ]
    if (
        depth <= 0
        or not _nrrd_layer_ref_is_raw_bbox_store(ref)
        or int(getattr(ref, 'shape', (0, 0, 0))[0]) != int(depth)
    ):
        return equal, None
    try:
        index_path = Path(ref.path) / 'index.bin'
        index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=int(depth))
        if int(index.shape[0]) != int(depth):
            return equal, None
        weights = np.asarray(index['payload_nbytes'], dtype=np.uint64)
        total = int(np.sum(weights, dtype=np.uint64))
        if total <= 0:
            return equal, [0 for _ in equal]
        prefix = np.cumsum(weights, dtype=np.uint64)
        boundaries = [0]
        for i in range(1, count):
            target = int((int(total) * int(i) + int(count) - 1) // int(count))
            boundary = int(np.searchsorted(prefix, np.uint64(target), side='left')) + 1
            boundary = max(int(boundaries[-1]) + 1, int(boundary))
            boundary = min(int(boundary), int(depth) - int(count - i))
            boundaries.append(int(boundary))
        boundaries.append(int(depth))
        bands = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(count)]
        band_weights = [
            int(np.sum(weights[int(z0):int(z1)], dtype=np.uint64))
            for z0, z1 in bands
        ]
        return bands, band_weights
    except Exception:
        return equal, None

@runtime_telemetry_phase('nrrd.write_layer')
def write_single_layer_nrrd_from_ref(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    segment_name: Optional[str] = None,
    segment_color: Optional[Tuple[float, float, float]] = None,
    block_consumer: Optional[Callable[[int, np.ndarray], None]] = None,
    sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    z_shards: Optional[int] = None,
    layout: Optional[str] = None,
    raster_plan: Optional[NrrdRasterPlan] = None,
    plan_receipt: Optional[Dict[str, NrrdRasterPlan]] = None,
) -> Path:
    """Write one component layer as its own single-layer 3D Slicer segmentation NRRD (X, Y, t).

 The layer is restored from its backing-store geometry directly to the final output
 geometry while streaming, reusing the same per-layer restore/stream path as the legacy
 decomposed writer. Each file holds one uint8 binary mask and is compressed
 as KAT-validated complete gzip members, preferring hardware-only QAT before
 libdeflate, ISA-L, or stdlib zlib; ``cpu`` opts out of QAT and IAA remains
 explicit-only. The Slicer segmentation fields make the file import as a Segmentation
 node; segment_name defaults to the output filename stem and segment_color to
 the deterministic palette pick for that name.

 ``layout`` selects the stored raster (``None``/``full``: the historical full reference
 raster; ``cropped``: the segment bounding box). ``raster_plan`` pins a precomputed plan.
 A cropped write that meets foreground outside its box is republished with the full
 layout; ``plan_receipt[str(out_path)]`` records the plan actually written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    seg_name = str(segment_name) if segment_name else _slicer_segment_name_for_out_path(out_path)
    seg_color = segment_color if segment_color is not None else slicer_segment_palette_color(seg_name)
    # live-volume layers defer the segment-extent scan to THIS (background
    # sink) worker instead of blocking the producing thread with a store-encode pass.
    ref = _resolve_live_ref_extent(ref)
    plan = raster_plan if raster_plan is not None else _nrrd_raster_plan(
        ref, (out_t, out_h, out_w), layout=layout,
    )
    try:
        _write_single_layer_nrrd_file(
            ref, (out_t, out_h, out_w), out_path, plan,
            segment_name=seg_name, segment_color=seg_color,
            block_consumer=block_consumer, sparse_consumer=sparse_consumer,
            z_shards=z_shards,
        )
    except NrrdCropExtentViolation as exc:
        if _nrrd_plan_layout(plan) != NRRD_LAYER_LAYOUT_CROPPED:
            raise
        _announce_nrrd_crop_violation(out_path, exc)
        # Mirror consumers fold with OR/max, so replaying the producer is idempotent.
        plan = _nrrd_raster_plan(ref, (out_t, out_h, out_w), layout=NRRD_LAYER_LAYOUT_FULL)
        _write_single_layer_nrrd_file(
            ref, (out_t, out_h, out_w), out_path, plan,
            segment_name=seg_name, segment_color=seg_color,
            block_consumer=block_consumer, sparse_consumer=sparse_consumer,
            z_shards=z_shards,
        )
    if plan_receipt is not None:
        plan_receipt[str(out_path)] = plan
    return out_path

def _announce_nrrd_crop_violation(out_path: Path, exc: BaseException) -> None:
    runtime_telemetry().add('nrrd.layer_crop.extent_violations', 1)
    print(
        f'Warning: {Path(out_path).name} has foreground outside its recorded segment '
        f'extent ({exc}); republishing that layer on the full reference raster.',
        flush=True,
    )

def _write_single_layer_nrrd_file(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    raster_plan: object,
    *,
    segment_name: str,
    segment_color: Tuple[float, float, float],
    block_consumer: Optional[Callable[[int, np.ndarray], None]],
    sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]],
    z_shards: Optional[int],
) -> Path:
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    stored_t, stored_h, stored_w = (int(v) for v in raster_plan.stored_shape_tyx)
    cropped = _nrrd_plan_layout(raster_plan) == NRRD_LAYER_LAYOUT_CROPPED
    box_t0 = _nrrd_plan_offset(raster_plan)[0]
    header = _nrrd_layer_header(
        raster_plan, (out_t, out_h, out_w),
        segment_name=segment_name, color_rgb=segment_color,
    )
    # Producers emit reference-geometry blocks, so a cropped plan sizes them by the
    # reference slice; the full layout keeps its historical (identical) stored sizing.
    z_chunk = (
        _nrrd_full_slice_z_chunk(1, out_w, out_h, out_t)
        if cropped else _nrrd_full_slice_z_chunk(1, stored_w, stored_h, stored_t)
    )
    # resolve auto-sharding only after this sink task starts executing. Queue
    # occupancy at submit time says nothing about the child-band capacity by the time a
    # late global layer reaches the head of the sink pool. The shared semaphore remains
    # the hard cross-layer over-sharding guard.
    if z_shards is None or int(z_shards) <= 0:
        shard_count = _nrrd_layer_zshard_count(ref, int(stored_t))
    else:
        shard_count = max(
            1,
            min(int(stored_t), int(z_shards), int(_nrrd_zshard_capacity())),
        )
    bands: Optional[List[Tuple[int, int]]] = None
    band_weights: Optional[List[int]] = None
    expected_codec_request_bytes = int(stored_t) * int(stored_h) * int(stored_w)
    if shard_count > 1:
        if cropped:
            bands, band_weights = _nrrd_cropped_zshard_bands(
                ref, int(out_t), int(box_t0), int(box_t0) + int(stored_t), int(shard_count),
            )
            shard_count = len(bands)
        else:
            bands, band_weights = _nrrd_layer_zshard_bands(ref, int(out_t), int(shard_count))
        expected_codec_request_bytes = max(1, min(
            max(0, min(int(z1), int(box_t0) + int(stored_t)) - max(int(z0), int(box_t0)))
            * int(stored_h) * int(stored_w)
            for z0, z1 in bands
        ))
    # Resolve, preflight, and KAT exactly once before a staging file/spool is opened.
    # Every z shard of this logical NRRD is pinned to this immutable codec spec.
    member_codec = nrrd_compression._require_nrrd_member_codec(
        expected_input_bytes=int(expected_codec_request_bytes),
    )
    if shard_count <= 1:
        with _same_directory_atomic_output(out_path) as stage_path:
            with open(stage_path, 'wb') as fh:
                _write_nrrd_ascii_header(
                    fh,
                    header=header,
                    sizes=(stored_w, stored_h, stored_t),
                    dimension=3,
                    data_type='unsigned char',
                    encoding='gzip',
                )
                with nrrd_compression._open_nrrd_payload_writer(fh, codec_spec=member_codec) as payload_writer:
                    stream_writer = _nrrd_payload_writer_for_plan(
                        payload_writer, raster_plan, (out_t, out_h, out_w), 0, out_t,
                        label=out_path.name,
                    )
                    _write_one_decomposed_nrrd_layer_payload(
                        ref,
                        (out_t, out_h, out_w),
                        stream_writer,
                        z_chunk=int(z_chunk),
                        block_consumer=block_consumer,
                        sparse_consumer=sparse_consumer,
                    )
                    _finish_nrrd_payload_writer(stream_writer)
        return out_path

    # each z band compresses into an independent anonymous memory-backed chunk.
    # A band holds ONE process-global permit only while it is actually compressing. This
    # lets two 8-band global files use a 12-lane budget concurrently and removes the
    # ordered bounded-queue backpressure that previously kept later compressors idle.
    if bands is None:
        raise RuntimeError('NRRD z-shard bands were not resolved')
    token = f'{os.getpid()}.{threading.get_ident()}'
    final_tmp = out_path.with_name(f'.{out_path.name}.{token}.assembling')
    shard_z_chunk = max(1, int(z_chunk) // int(shard_count))
    shard_fill_workers = int(nrrd_fill_workers())
    weight_note = ''
    if band_weights is not None:
        weight_note = (
            f', sparse_payload={min(band_weights) / (1024 ** 2):.1f}..'
            f'{max(band_weights) / (1024 ** 2):.1f} MiB/shard'
        )
    print(
        f'{out_path.name} -> {shard_count} in-memory compressed z chunks '
        f'(min_slices={nrrd_layer_zshard_min_slices()}, z_chunk={shard_z_chunk}'
        f'{weight_note}; one global permit per active band).'
    )
    zshard_permits = _nrrd_zshard_semaphore()
    n19_started = time.perf_counter()
    n19_band_metrics: List[Optional[Tuple[float, int, int, int]]] = [None] * int(shard_count)

    def _write_band(i: int) -> object:
        z0, z1 = bands[int(i)]
        zshard_permits.acquire(1)
        spool = None
        band_started = time.perf_counter()
        try:
            # Keep the complete compressed band in an anonymous memory-backed file. No encoded
            # shard is written to the output or scratch filesystem before ordered assembly.
            spool = _open_memory_backed_encoded_chunk(
                f'{out_path.stem}.nrrd.z{int(z0):06d}-{int(z1):06d}.gz',
            )
            with nrrd_compression._open_nrrd_payload_writer(spool, codec_spec=member_codec) as payload_writer:
                stream_writer = _nrrd_payload_writer_for_plan(
                    payload_writer, raster_plan, (out_t, out_h, out_w), int(z0), int(z1),
                    label=out_path.name,
                )
                _write_one_decomposed_nrrd_layer_payload(
                    ref,
                    (out_t, out_h, out_w),
                    stream_writer,
                    z_chunk=shard_z_chunk,
                    block_consumer=block_consumer,
                    sparse_consumer=sparse_consumer,
                    z_start=int(z0),
                    z_stop=int(z1),
                    fill_workers_override=shard_fill_workers,
                )
                _finish_nrrd_payload_writer(stream_writer)
            spool.flush()
            spool.seek(0, os.SEEK_END)
            compressed_size = int(spool.tell())
            spool.seek(0)
            n19_band_metrics[int(i)] = (
                float(time.perf_counter() - band_started),
                int(compressed_size),
                int(z0),
                int(z1),
            )
            return spool
        except BaseException:
            if spool is not None:
                try:
                    spool.close()
                except Exception:
                    pass
            raise
        finally:
            zshard_permits.release(1)

    def _copy_spool(src: object, dst: object) -> int:
        src.flush()
        dst.flush()
        src.seek(0)
        total = 0
        try:
            src_fd = int(src.fileno())
            dst_fd = int(dst.fileno())
            while True:
                copied = os.copy_file_range(src_fd, dst_fd, 64 * 1024 * 1024)
                if int(copied) <= 0:
                    break
                total += int(copied)
            return int(total)
        except Exception:
            # copy_file_range may fail after making partial progress (cross-filesystem,
            # quota, filesystem implementation). Resume at the exact byte offsets rather
            # than appending the already-copied prefix a second time.
            src.seek(int(total))
            while True:
                block = src.read(16 * 1024 * 1024)
                if not block:
                    break
                view = memoryview(block)
                written = 0
                while written < len(view):
                    step = dst.write(view[written:])
                    if step is None:
                        step = len(view) - written
                    if int(step) <= 0:
                        raise OSError('NRRD in-memory chunk copy made no forward progress')
                    written += int(step)
                total += int(written)
            return int(total)

    shard_pool = _acquire_parallel_pool(int(shard_count))
    futures: List[Future] = []
    spools: List[object] = []
    try:
        futures = [shard_pool.submit(_write_band, i) for i in range(shard_count)]
        with open(final_tmp, 'wb', buffering=0) as fh:
            _write_nrrd_ascii_header(
                fh,
                header=header,
                sizes=(stored_w, stored_h, stored_t),
                dimension=3,
                data_type='unsigned char',
                encoding='gzip',
            )
            # Wait/copy in z order. Later bands continue compressing independently in memory
            # while an earlier completed chunk streams as already-compressed bytes to disk.
            for i, fut in enumerate(futures):
                spool = fut.result()
                spools.append(spool)
                _copy_spool(spool, fh)
                spool.close()
                spools[-1] = None
        _publish_staged_file_atomically(final_tmp, out_path)
        completed_metrics = [m for m in n19_band_metrics if m is not None]
        compressed_total = sum(int(m[1]) for m in completed_metrics)
        slowest = max(completed_metrics, key=lambda m: float(m[0])) if completed_metrics else None
        slowest_note = ''
        if slowest is not None:
            slowest_note = (
                f', slowest_band=z{int(slowest[2])}:{int(slowest[3])} '
                f'{float(slowest[0]):.2f}s/{int(slowest[1]) / (1024 ** 2):.1f} MiB'
            )
        print(
            f'{out_path.name} completed in '
            f'{time.perf_counter() - n19_started:.2f}s; compressed='
            f'{compressed_total / GIB:.2f} GiB{slowest_note}.'
        )
    finally:
        _settle_parallel_futures(futures)
        consumed_ids = {id(spool) for spool in spools if spool is not None}
        for fut in futures:
            if not fut.done() or fut.cancelled():
                continue
            try:
                pending_spool = fut.result()
            except BaseException:
                continue
            if pending_spool is not None and id(pending_spool) not in consumed_ids:
                try:
                    pending_spool.close()
                except Exception:
                    pass
        _release_parallel_pool(int(shard_count), shard_pool)
        for spool in spools:
            if spool is not None:
                try:
                    spool.close()
                except Exception:
                    pass
        try:
            final_tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path

_NRRD_GPU_MIRROR_TEE_ANNOUNCED = False

_NRRD_SPARSE_MIRROR_TEE_ANNOUNCED = False

def nrrd_gpu_mirror_tee_enabled() -> bool:
    """Derive low-quality NRRD mirror volumes on the GPU."""
    return _env_flag('YOLO_TTA_NRRD_GPU_MIRROR_TEE', True)

class _GpuMirrorTee:
    """Derive downbinned mirror volumes on an exclusively leased CUDA device."""

    def __init__(
        self,
        torch_mod: object,
        device: object,
        mirror_jobs: List[Dict[str, object]],
        out_t: int,
        gpu_lease: _MainProcessGpuStageLease,
    ) -> None:
        self.torch = torch_mod
        self.device = device
        self.out_t = int(out_t)
        self.failed = False
        self.sub_batch = max(1, _env_int('YOLO_TTA_NRRD_GPU_MIRROR_TEE_SUBBATCH', 32))
        self.specs: List[Dict[str, object]] = []
        self.stream: Optional[object] = None
        self._gpu_lease: Optional[_MainProcessGpuStageLease] = gpu_lease
        self._closed = False
        try:
            with torch_mod.cuda.device(device):
                self.stream = torch_mod.cuda.Stream(device=device)
                with torch_mod.cuda.stream(self.stream):
                    for job in mirror_jobs:
                        m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[misc]
                        self.specs.append({
                            'job': job,
                            'vol': torch_mod.zeros(
                                (m_t, m_h, m_w), dtype=torch_mod.uint8, device=device,
                            ),
                            'src_to_m': job['src_to_m'],
                            'm_h': int(m_h),
                            'm_w': int(m_w),
                        })
                self.stream.synchronize()
        except BaseException:
            self._release_resources()
            raise

    def _release_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except Exception:
            pass
        for spec in self.specs:
            spec['vol'] = None
        self.specs.clear()
        self.stream = None
        _trim_main_process_cuda_device(
            self.torch,
            self.device,
            desc='NRRD GPU mirror tee cleanup',
        )
        lease = self._gpu_lease
        self._gpu_lease = None
        if lease is not None:
            lease.release()

    def tee(self, z0: int, block: np.ndarray) -> None:
        if self.failed or self._closed:
            return
        torch = self.torch
        blk = blk_f = pooled = pooled_u8 = vol = None
        try:
            import torch.nn.functional as F  # type: ignore
            with torch.cuda.device(self.device):
                with torch.cuda.stream(self.stream):
                    z_count = int(block.shape[0])
                    for b0 in range(0, z_count, int(self.sub_batch)):
                        b1 = min(z_count, b0 + int(self.sub_batch))
                        blk = torch.from_numpy(np.ascontiguousarray(block[b0:b1])).to(self.device)
                        blk_f = blk.to(torch.float16).unsqueeze(1)
                        for spec in self.specs:
                            pooled = F.adaptive_max_pool2d(
                                blk_f, (int(spec['m_h']), int(spec['m_w'])),
                            )
                            pooled_u8 = (pooled.squeeze(1) > 0).to(torch.uint8)
                            vol = spec['vol']
                            src_to_m = spec['src_to_m']
                            for bi in range(int(b1 - b0)):
                                full_z = int(z0) + int(b0) + int(bi)
                                if full_z >= self.out_t:
                                    break
                                for mz in src_to_m[full_z]:  # type: ignore[index]
                                    torch.maximum(vol[int(mz)], pooled_u8[int(bi)], out=vol[int(mz)])
                        blk = blk_f = pooled = pooled_u8 = vol = None
        except Exception as exc:
            self.failed = True
            # Clear partially constructed per-block tensors before trimming the allocator.
            blk = blk_f = pooled = pooled_u8 = vol = None
            gc.collect()
            print(
                f'Warning: GPU mirror tee failed mid-stream ({exc}); '
                'mirrors re-encode from the layer store.'
            )
            self._release_resources()

    def discard(self) -> None:
        self._release_resources()

    def finalize_into_jobs(self) -> bool:
        if self.failed or self._closed:
            self._release_resources()
            return False
        try:
            self.stream.synchronize()
            for spec in self.specs:
                spec['job']['volume'] = spec['vol'].cpu().numpy()  # type: ignore[index]
                spec['vol'] = None
            return True
        except Exception as exc:
            print(
                f'Warning: GPU mirror tee finalize failed ({exc}); '
                'mirrors re-encode from the layer store.'
            )
            self.failed = True
            return False
        finally:
            self._release_resources()

    def __del__(self) -> None:  # pragma: no cover - final safety net
        try:
            self._release_resources()
        except Exception:
            pass

def _acquire_output_gpu_stage(torch_mod: object, purpose: str) -> Optional[_MainProcessGpuStageLease]:
    """Wait only for an occupied configured GPU in a later-cutoff sweep branch."""
    from .backprojection import _main_process_gpu_output_stage_wait_state, _wait_main_process_gpu_stage_change

    consume = os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE') == 'consume'
    deadline = time.monotonic() + max(0.0, _env_float('XTA_YOLO_SWEEP_OUTPUT_STAGE_WAIT_SECONDS', 300.0))
    while True:
        status, epoch = _main_process_gpu_output_stage_wait_state(torch_mod, purpose)
        if status == 0:
            return None
        if status == 1:
            lease = _try_acquire_main_process_gpu_stage(torch_mod, purpose)
            if lease is not None or not consume:
                return lease
            # A failed CUDA probe on an otherwise eligible device cannot be
            # resolved by waiting for a lease. A competing successful claim can.
            status, epoch = _main_process_gpu_output_stage_wait_state(torch_mod, purpose)
            if status != 2:
                return None
        remaining = deadline - time.monotonic()
        if not consume or remaining <= 0:
            return None
        _wait_main_process_gpu_stage_change(epoch, remaining)

def _try_create_gpu_mirror_tee(
    mirror_jobs: List[Dict[str, object]],
    out_t: int,
    source_hw: Tuple[int, int],
) -> Optional['_GpuMirrorTee']:
    in_h, in_w = source_hw
    if not nrrd_gpu_mirror_tee_enabled() or not mirror_jobs or any(
        int(job['shape'][1]) > in_h or int(job['shape'][2]) > in_w for job in mirror_jobs
    ):
        return None
    lease: Optional[_MainProcessGpuStageLease] = None
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return None
        lease = _acquire_output_gpu_stage(torch, 'NRRD low-quality GPU mirror tee')
        if lease is None:
            _announce_main_gpu_stage_skip_once(
                'nrrd-mirror-tee-inference-busy',
                'NRRD GPU mirror tee skipped while all eligible GPUs have active/queued '
                'inference or another output-stage lease; using the CPU mirror tee.',
            )
            return None
        device = lease.torch_device(torch)
        need = sum(
            int(np.prod([int(v) for v in job['shape']]))
            for job in mirror_jobs
        )  # type: ignore[misc]
        free_bytes, _total = torch.cuda.mem_get_info(device)
        if int(free_bytes) < int(need) + 2 * GIB:
            lease.release()
            lease = None
            return None
        tee = _GpuMirrorTee(torch, device, mirror_jobs, int(out_t), lease)
        lease = None  # ownership transferred to the tee
        return tee
    except Exception as exc:
        if lease is not None:
            try:
                device = lease.torch_device(torch)  # type: ignore[name-defined]
                _trim_main_process_cuda_device(
                    torch, device, desc='failed NRRD GPU mirror tee construction',  # type: ignore[name-defined]
                )
            except Exception:
                pass
            lease.release()
        print(f'Warning: NRRD GPU mirror tee setup failed ({exc}); using the CPU mirror tee.')
        return None

def write_layer_nrrd_with_low_quality_mirrors(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    mirrors: Sequence[Tuple[Tuple[int, int, int], Path]],
    *,
    segment_name: Optional[str] = None,
    segment_color: Optional[Tuple[float, float, float]] = None,
    z_shards: Optional[int] = None,
    layout: Optional[str] = None,
    raster_plan: Optional[NrrdRasterPlan] = None,
    mirror_raster_plans: Optional[Dict[str, NrrdRasterPlan]] = None,
    plan_receipt: Optional[Dict[str, NrrdRasterPlan]] = None,
    write_native: bool = True,
    native_review_path: Optional[Path] = None,
    mirror_review_paths: Optional[Dict[str, Path]] = None,
    mirror_nrrd_paths: Optional[Sequence[Path]] = None,
) -> Path:
    """Write a full-quality layer and all low-quality mirrors in one source pass.

    Independent mirror encoders may run concurrently while sharing the streamed layer data.
    ``layout``/``raster_plan`` select the full-quality stored raster as in
    write_single_layer_nrrd_from_ref; ``mirror_raster_plans`` maps ``str(mirror_path)`` to
    a pinned mirror plan, and ``plan_receipt`` records every plan actually written."""
    out_t = int(output_shape[0])
    review_paths = mirror_review_paths or {}
    nrrd_paths = None if mirror_nrrd_paths is None else set(map(Path, mirror_nrrd_paths))
    resolved_layout = _normalized_nrrd_layer_layout(layout)
    # Resolve a live layer's deferred extent once so full-quality and mirror headers agree.
    ref = _resolve_live_ref_extent(ref)
    # mirror payloads are composed in two geometry steps (backing -> full output ->
    # low quality). Compose the header extent through that same intermediate geometry;
    # mapping backing -> mirror directly can understate a positive-support boundary voxel.
    mirror_extent_ref = dataclasses_replace(
        ref,
        segment_extent_ijk=_nrrd_mapped_extent_preserve_empty(ref, output_shape),
        segment_extent_shape_tyx=tuple(int(v) for v in output_shape),
        segment_extent_source='composed_full_output_extent_for_low_quality_mirror',
    )
    pinned_mirror_plans: Dict[str, NrrdRasterPlan] = dict(mirror_raster_plans or {})
    # This function is itself the queued sink task, so z-shard resolution happens at
    # execution time, not submission time. Reuse the result for tee scheduling and the full writer.
    if z_shards is None or int(z_shards) <= 0:
        resolved_z_shards = _nrrd_layer_zshard_count(ref, int(out_t))
    else:
        resolved_z_shards = max(
            1,
            min(int(out_t), int(z_shards), int(_nrrd_zshard_capacity())),
        )
    mirror_jobs: List[Dict[str, object]] = []
    for m_shape, m_path in mirrors:
        m_t, m_h, m_w = (int(m_shape[0]), int(m_shape[1]), int(m_shape[2]))
        # Invert the mirror's t mapping once per geometry: for each full-quality output z,
        # the mirror slice(s) whose source range contains it (matches
        # _restore_source_indices_for_output_z). The per-layer list stays private.
        mirror_jobs.append({
            'shape': (m_t, m_h, m_w),
            'path': Path(m_path),
            'volume': None,  # allocated below only when the CPU tee runs
            'src_to_m': list(_nrrd_mirror_targets_by_source(int(out_t), int(m_t))),
            'locks': [threading.Lock() for _ in range(16)],
            'failed': False,
        })

    # raw bbox stores expose their restored output crops during the same
    # sparse member pass. Keep this tee sparse on the CPU: uploading thousands of tiny
    # crops is slower than the vectorized integral/gather resize, while uploading the old
    # dense 8.83 MiB plane would forfeit the bandwidth win. Other source types retain.
    sparse_mirror_tee = bool(
        mirror_jobs and _nrrd_layer_ref_is_raw_bbox_store(ref)
    )

    # derive the mirrors on a GPU when one is free — the per-slice resizes
    # and striped-lock ORs leave the CPU tee entirely; mirror volumes come back in one
    # small D2H after the layer streams. CPU tee (below) remains the fallback.
    gpu_tee = None if sparse_mirror_tee else _try_create_gpu_mirror_tee(mirror_jobs, out_t, output_shape[1:])
    if gpu_tee is not None:
        global _NRRD_GPU_MIRROR_TEE_ANNOUNCED
        if not _NRRD_GPU_MIRROR_TEE_ANNOUNCED:
            _NRRD_GPU_MIRROR_TEE_ANNOUNCED = True
            print(
                f'GPU low-quality mirror tee active on {gpu_tee.device} ('
                'YOLO_TTA_NRRD_GPU_MIRROR_TEE=0 disables).'
            )
    else:
        for job in mirror_jobs:
            m_t, m_h, m_w = (int(v) for v in job['shape'])  # type: ignore[misc]
            job['volume'] = np.zeros((m_t, m_h, m_w), dtype=np.uint8)
        if bool(sparse_mirror_tee):
            global _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED
            if not _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED:
                _NRRD_SPARSE_MIRROR_TEE_ANNOUNCED = True
                print(
                    'Crop-aware low-quality NRRD mirror tee active; '
                    'native/restored cvol layers fill their mirrors in the same pass '
                    '(no second dense store pass).'
                )

    # Z-sharding already supplies z-band concurrency. Avoid nesting another up-to-16-way CPU
    # pool inside every shard; the unsharded path keeps the full slice fan-out.
    tee_workers = 1 if int(resolved_z_shards) > 1 else max(1, min(int(nrrd_fill_workers()), 16))

    def _tee(z0: int, block: np.ndarray) -> None:
        live_jobs = [j for j in mirror_jobs if not bool(j['failed'])]
        if not live_jobs:
            return
        z_count = int(block.shape[0])

        def _tee_one(zi: int) -> None:
            full_z = int(z0) + int(zi)
            if full_z >= out_t:
                return
            frame = block[int(zi)]
            for job in live_jobs:
                if bool(job['failed']):
                    continue
                try:
                    targets = job['src_to_m'][full_z]
                    if not targets:
                        continue
                    m_t, m_h, m_w = job['shape']  # type: ignore[misc]
                    resized = _resize_binary_mask_frame_to_output_shape(frame, int(m_h), int(m_w))
                    vol = job['volume']
                    locks = job['locks']
                    for mz in targets:
                        # Adjacent full-z frames can map to the same mirror slice; stripe locks
                        # keep the read-modify-write OR race-free without serializing the block.
                        with locks[int(mz) % len(locks)]:  # type: ignore[index]
                            np.bitwise_or(vol[int(mz)], resized, out=vol[int(mz)])  # type: ignore[index]
                except Exception:
                    job['failed'] = True

        if z_count > 1 and tee_workers > 1:
            _nrrd_parallel_fill_indices(
                z_count, _tee_one, requested_workers=tee_workers,
            )
        else:
            for zi in range(z_count):
                _tee_one(int(zi))

    def _tee_sparse(out_z: int, y0: int, x0: int, crop: np.ndarray) -> None:
        """Fold one already-restored output crop into every low-quality mirror."""
        if int(out_z) < 0 or int(out_z) >= int(out_t):
            return
        crop_u8 = np.asarray(crop, dtype=np.uint8)
        if crop_u8.size <= 0:
            return
        y1 = int(y0) + int(crop_u8.shape[0])
        x1 = int(x0) + int(crop_u8.shape[1])
        out_h, out_w = int(output_shape[1]), int(output_shape[2])
        for job in mirror_jobs:
            if bool(job['failed']):
                continue
            try:
                targets = job['src_to_m'][int(out_z)]
                if not targets:
                    continue
                _m_t, m_h, m_w = job['shape']  # type: ignore[misc]
                resized = _resize_sparse_binary_crop_to_output_region(
                    crop_u8,
                    source_shape=(int(out_h), int(out_w)),
                    source_bbox=(int(y0), int(x0), int(y1), int(x1)),
                    output_shape=(int(m_h), int(m_w)),
                )
                if resized is None:
                    continue
                my0, mx0, my1, mx1, mirror_crop = resized
                vol = job['volume']
                locks = job['locks']
                for mz in targets:
                    with locks[int(mz) % len(locks)]:  # type: ignore[index]
                        dst = vol[int(mz), int(my0):int(my1), int(mx0):int(mx1)]  # type: ignore[index]
                        np.bitwise_or(dst, mirror_crop, out=dst)
            except Exception:
                job['failed'] = True

    if bool(sparse_mirror_tee) and gpu_tee is None:
        # Fused producers fold a whole z-range of crops per job in one nogil resize
        # call and one locked OR call, with _tee_sparse's per-job failure semantics.
        from .nrrd_sparse_kernels import FusedSparseMirrorTee
        _tee_sparse.fused_sparse_mirror_tee = FusedSparseMirrorTee(  # type: ignore[attr-defined]
            mirror_jobs, (int(output_shape[1]), int(output_shape[2])),
        )

    # shard consumers can arrive concurrently. CPU mirror updates use
    # per-output-z stripe locks; the GPU tee owns one stream and tensor set, so
    # serialize only its tee calls while member compression remains parallel.
    gpu_tee_call_lock = threading.Lock() if gpu_tee is not None else None

    def _consume_block(z0: int, block: np.ndarray) -> None:
        if gpu_tee is not None:
            with gpu_tee_call_lock:  # type: ignore[arg-type]
                was_ok = not gpu_tee.failed
                gpu_tee.tee(int(z0), block)
                if gpu_tee.failed and was_ok:
                    # Mid-stream GPU failure: earlier blocks never went through the CPU tee,
                    # so partial CPU teeing is useless — mark every mirror for the store
                    # re-encode fallback below.
                    for job in mirror_jobs:
                        job['failed'] = True
                    gpu_tee.discard()
            return
        _tee(int(z0), block)

    block_consumer = _consume_block if mirror_jobs and not sparse_mirror_tee else None
    sparse_consumer = _tee_sparse if sparse_mirror_tee else None
    try:
        if write_native:
            write_single_layer_nrrd_from_ref(
                ref, output_shape, out_path,
                segment_name=segment_name, segment_color=segment_color,
                block_consumer=block_consumer, sparse_consumer=sparse_consumer,
                z_shards=int(resolved_z_shards), layout=resolved_layout,
                raster_plan=raster_plan, plan_receipt=plan_receipt,
            )
            if native_review_path is not None:
                _stage_layer_raster(ref, output_shape, native_review_path)
        else:
            _stage_layer_raster(ref, output_shape, native_review_path,
                                block_consumer=block_consumer, sparse_consumer=sparse_consumer)
    except BaseException:
        if gpu_tee is not None:
            gpu_tee.discard()
        raise

    # one small D2H per mirror replaces the per-block CPU resize/OR work.
    if gpu_tee is not None and not gpu_tee.finalize_into_jobs():
        for job in mirror_jobs:
            job['failed'] = True

    def _encode_mirror_volume(
        vol: np.ndarray,
        m_path: Path,
        mirror_shape: Tuple[int, int, int],
        mirror_plan: object,
        seg_name: str,
        seg_color: Tuple[float, float, float],
    ) -> None:
        m_t, m_h, m_w = (int(v) for v in mirror_shape)
        stored_t, stored_h, stored_w = (int(v) for v in mirror_plan.stored_shape_tyx)
        cropped = _nrrd_plan_layout(mirror_plan) == NRRD_LAYER_LAYOUT_CROPPED
        header = _nrrd_layer_header(
            mirror_plan, (m_t, m_h, m_w), segment_name=seg_name, color_rgb=seg_color,
        )
        # A cropped plan streams the whole in-memory mirror through the box adapter,
        # which also proves that no mirror foreground falls outside the stored box.
        step = max(1, int(
            _nrrd_full_slice_z_chunk(1, m_w, m_h, m_t)
            if cropped else _nrrd_full_slice_z_chunk(1, stored_w, stored_h, stored_t)
        ))
        source_depth = int(m_t) if cropped else int(stored_t)
        member_codec = nrrd_compression._require_nrrd_member_codec(
            expected_input_bytes=int(stored_t) * int(stored_h) * int(stored_w),
        )
        with _same_directory_atomic_output(m_path) as stage_path:
            with open(stage_path, 'wb') as fh:
                _write_nrrd_ascii_header(
                    fh, header=header, sizes=(stored_w, stored_h, stored_t),
                    dimension=3, data_type='unsigned char', encoding='gzip',
                )
                with nrrd_compression._open_nrrd_payload_writer(fh, codec_spec=member_codec) as payload_writer:
                    stream_writer = _nrrd_payload_writer_for_plan(
                        payload_writer, mirror_plan, (m_t, m_h, m_w), 0, m_t, label=m_path.name,
                    )
                    for z0 in range(0, source_depth, step):
                        z1 = min(source_depth, int(z0) + step)
                        stream_writer.write(memoryview(vol[z0:z1]).cast('B'))  # type: ignore[index]
                    _finish_nrrd_payload_writer(stream_writer)

    def _encode_mirror(job: Dict[str, object]) -> None:
        from .review_layer_sources import stage_review_volume

        m_t, m_h, m_w = job['shape']  # type: ignore[misc]
        m_path: Path = job['path']  # type: ignore[assignment]
        review_path = review_paths.get(str(m_path))
        write_nrrd = nrrd_paths is None or m_path in nrrd_paths
        mirror_shape = (int(m_t), int(m_h), int(m_w))
        mirror_plan = pinned_mirror_plans.get(str(m_path))
        if mirror_plan is None:
            mirror_plan = _nrrd_raster_plan(mirror_extent_ref, mirror_shape, layout=resolved_layout)
        vol: Optional[np.ndarray] = job.get('volume')  # type: ignore[assignment]
        try:
            try:
                if bool(job['failed']) or vol is None:
                    raise RuntimeError('mirror tee produced no settled host payload')
                if write_nrrd:
                    m_path.parent.mkdir(parents=True, exist_ok=True)
                    seg_name = str(segment_name) if segment_name else _slicer_segment_name_for_out_path(m_path)
                    seg_color = segment_color if segment_color is not None else slicer_segment_palette_color(seg_name)
                    try:
                        _encode_mirror_volume(vol, m_path, mirror_shape, mirror_plan, seg_name, seg_color)
                    except NrrdCropExtentViolation as exc:
                        if _nrrd_plan_layout(mirror_plan) != NRRD_LAYER_LAYOUT_CROPPED:
                            raise
                        _announce_nrrd_crop_violation(m_path, exc)
                        mirror_plan = _nrrd_raster_plan(
                            mirror_extent_ref, mirror_shape, layout=NRRD_LAYER_LAYOUT_FULL,
                        )
                        _encode_mirror_volume(vol, m_path, mirror_shape, mirror_plan, seg_name, seg_color)
            except Exception as exc:
                print(f'Warning: low-quality mirror tee failed for {m_path.name} ({exc}); '
                      'restoring that mirror from the layer store.')
                if write_nrrd:
                    write_single_layer_nrrd_from_ref(
                        ref, mirror_shape, m_path,
                        segment_name=segment_name, segment_color=segment_color, layout=resolved_layout,
                        raster_plan=(mirror_plan if _nrrd_plan_layout(mirror_plan) == NRRD_LAYER_LAYOUT_CROPPED
                                     else None), plan_receipt=plan_receipt,
                    )
                if review_path is not None:
                    _stage_layer_raster(ref, mirror_shape, review_path)
            else:
                if review_path is not None:
                    stage_review_volume(review_path, vol)
                if plan_receipt is not None:
                    plan_receipt[str(m_path)] = mirror_plan
        finally:
            job['volume'] = None

    # the mirror files of one layer are independent — encode them side by
    # side instead of one after another at the tail of the layer task. Every task runs to
    # completion (a failing mirror re-encodes from the store, matching the serial per-mirror
    # fallback) and the first error is re-raised after all mirrors settle.
    if len(mirror_jobs) > 1:
        pool_size = min(4, len(mirror_jobs))
        mirror_pool = _acquire_parallel_pool(int(pool_size))
        first_error: Optional[BaseException] = None
        try:
            futures = [mirror_pool.submit(_encode_mirror, job) for job in mirror_jobs]
            for fut in futures:
                try:
                    fut.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        finally:
            _release_parallel_pool(int(pool_size), mirror_pool)
        if first_error is not None:
            raise first_error
    else:
        for job in mirror_jobs:
            _encode_mirror(job)
    return out_path


def _stage_layer_raster(ref, output_shape, review_path, *, block_consumer=None, sparse_consumer=None):
    from .review_layer_sources import DiscardedRaster, PackedReviewRaster

    writer = PackedReviewRaster(review_path, output_shape) if review_path is not None else DiscardedRaster()
    out_t, out_h, out_w = output_shape
    try:
        _write_one_decomposed_nrrd_layer_payload(
            ref, output_shape, writer, z_chunk=_nrrd_full_slice_z_chunk(1, out_w, out_h, out_t),
            block_consumer=block_consumer, sparse_consumer=sparse_consumer,
        )
        if review_path is not None:
            writer.finalize()
    except BaseException:
        if review_path is not None:
            writer.discard()
        raise

def write_sink_layer_files(
    ref: 'NrrdLayerRef',
    output_shape: Tuple[int, int, int],
    out_path: Path,
    mirrors: Sequence[Tuple[Tuple[int, int, int], Path]],
    *,
    segment_name: str,
    segment_color: Tuple[float, float, float],
    z_shards: int,
    layout: str,
    raster_plan: NrrdRasterPlan,
    mirror_raster_plans: Dict[str, NrrdRasterPlan],
    plan_receipt: Dict[str, NrrdRasterPlan],
    write_native: bool = True,
    native_review_path: Optional[Path] = None,
    mirror_review_paths: Optional[Dict[str, Path]] = None,
    mirror_nrrd_paths: Optional[Sequence[Path]] = None,
) -> Path:
    """Write one NrrdLayerSink layer: its native file and every low-quality mirror.

    The sink's writer thread and a sweep NRRD worker process run this same call, so
    both write identical bytes; ``plan_receipt`` records each plan actually written."""
    if mirrors:
        return write_layer_nrrd_with_low_quality_mirrors(
            ref, output_shape, Path(out_path), list(mirrors),
            segment_name=segment_name,
            segment_color=segment_color,
            z_shards=int(z_shards),
            layout=layout,
            raster_plan=raster_plan,
            mirror_raster_plans=mirror_raster_plans,
            plan_receipt=plan_receipt,
            write_native=write_native, native_review_path=native_review_path,
            mirror_review_paths=mirror_review_paths, mirror_nrrd_paths=mirror_nrrd_paths,
        )
    if native_review_path is not None:
        _stage_layer_raster(ref, output_shape, native_review_path)
    if not write_native:
        return out_path
    return write_single_layer_nrrd_from_ref(
        ref, output_shape, Path(out_path),
        segment_name=segment_name,
        segment_color=segment_color,
        z_shards=int(z_shards),
        layout=layout,
        raster_plan=raster_plan,
        plan_receipt=plan_receipt,
    )

_INTERPOLATION_PROVENANCE_FIELDS = (
    'interpolation_backend', 'interpolation_direction', 'seed_detector_identity',
    'sam_bundle_identity', 'interpolation_policy_identity', 'interpolation_connectivity', 'proposal_evidence_path',
    'proposal_selection_status', 'sam_group_ids', 'sam_run_ids', 'observation_roots',
    'gate_support_identity', 'upstream_interpolation_policy_identity', 'native_transform',
    'selected_bridge_connection_status', 'final_connection_survival',
)


def interpolation_layer_provenance(ref: NrrdLayerRef, *, relative_to: Optional[Path] = None) -> Dict[str, object]:
    """Return explicit SAM/gate lineage without changing legacy SDF metadata."""
    from .artifact_archive import split_reference

    def relative_reference(value):
        parsed = split_reference(value)
        physical, logical = parsed if parsed is not None else (Path(value).resolve(), None)
        relative = Path(os.path.relpath(str(physical), str(Path(relative_to).resolve()))).as_posix()
        return relative if logical is None else relative + '#/' + logical

    tail = dict(getattr(ref, 'extrapolation_provenance', {}) or {})
    if tail:
        if relative_to is not None and tail.get('evidence_path'):
            tail['evidence_path'] = relative_reference(str(tail['evidence_path']))
            tail['evidence_path_base'] = 'manifest_directory'
        return {'extrapolation': tail, 'native_transform': ref.native_transform}
    if (str(getattr(ref, 'interpolation_backend', '')).lower() != 'sam'
            and not getattr(ref, 'gate_support_identity', '')
            and not getattr(ref, 'upstream_interpolation_policy_identity', '')):
        return {}
    result = {name: getattr(ref, name, '') for name in _INTERPOLATION_PROVENANCE_FIELDS}
    # JSON lists are portable and keep direction/run identity separate from the
    # detector-confidence channels and geometric source voting groups.
    for name in ('sam_group_ids', 'sam_run_ids', 'observation_roots'):
        result[name] = list(result[name])
    if relative_to is not None and result.get('proposal_evidence_path'):
        result['proposal_evidence_path'] = relative_reference(str(result['proposal_evidence_path']))
        result['proposal_evidence_path_base'] = 'manifest_directory'
    return result


class NrrdLayerSink:
    """Write each component layer as its own NRRD as soon as the layer becomes available."""

    def __init__(
        self,
        *,
        nrrd_dir: Path,
        stem: str,
        output_shape_tyx: Tuple[int, int, int],
        max_workers: int,
        low_quality_specs: Optional[Sequence['LowQualityDownbinSpec']] = None,
        low_quality_root: Optional[Path] = None,
        run_id: str | None = None,
        executor: Optional[ThreadPoolExecutor] = None,
        layer_file_processes: Optional[object] = None,
        layer_layout: str = NRRD_LAYER_LAYOUT_CROPPED,
        relative_artifact_paths: bool = True,
        managed_review_root: Optional[Path] = None,
        full_quality_formats: Sequence[str] = ('nrrd',),
        cutoff: Optional[float] = None,
        source_path: Optional[Path] = None,
        output_root: Optional[Path] = None,
        review_scratch_dir: Optional[Path] = None,
    ) -> None:
        self.nrrd_dir = Path(nrrd_dir)
        self.output_root = Path(output_root) if output_root is not None else self.nrrd_dir.parent
        self.full_quality_formats = frozenset(full_quality_formats) & {'nrrd', 'xtareview'}
        self.write_native_nrrd = 'nrrd' in self.full_quality_formats
        self.cutoff = 0.25 if cutoff is None else float(cutoff)
        self.cutoff_cent = max(0, min(100, int(round(100 * self.cutoff))))
        self._review_stage = os.environ.get('XTA_YOLO_SWEEP_XTAREVIEW_STAGE_ROOT')
        self._review_sources_dir: Optional[Path] = None
        self._review_temp = None
        self._published_paths: Dict[str, Dict[str, Path]] = {}
        self._staged_archives: List[str] = []
        self._review_source_record: Dict[str, object] = {'confidence_threshold_exact': self.cutoff}
        if source_path is not None:
            source_file = Path(source_path)
            self._review_source_record.update(path=str(source_file), source_stem=source_file.stem)
        self._review_axial_crop = None
        self.run_id = run_id
        self.relative_artifact_paths = bool(relative_artifact_paths)
        self.managed_review_root = managed_review_root
        # Optional sweep_nrrd_processes.SweepNrrdProcessPool owned by the caller: it
        # writes eligible layer files outside this process's GIL. This sink keeps every
        # name, colour, manifest row and publication record either way.
        self.layer_file_processes = layer_file_processes
        if self.write_native_nrrd:
            self.nrrd_dir.mkdir(parents=True, exist_ok=True)
        self.stem = str(stem)
        self.output_shape = (int(output_shape_tyx[0]), int(output_shape_tyx[1]), int(output_shape_tyx[2]))
        # ``cropped`` stores each layer's segment bounding box with a Slicer reference-extent
        # offset; ``full`` writes the historical full-reference raster byte for byte (tests
        # and profiling tools compare the two). Fixed per sink so a run never mixes layouts;
        # an invalid value fails here, before any layer is queued.
        self.layer_layout = _normalized_nrrd_layer_layout(layer_layout)
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._futures: List[Future] = []
        self._source_refs: Dict[Future, NrrdLayerRef] = {}
        self._layer_publication_observer: Optional[Callable[[NrrdLayerRef], object]] = None
        # Keep completion/manifest identity without retaining immutable live arrays
        # after their writer settles. View-major sweeps transfer these exact scalar
        # rows into a later scalar tail rather than encoding the layer again.
        self._publication_records: Dict[Tuple[str, str], List[Tuple[Future, Dict[str, object]]]] = {}
        self._manifest: List[Dict[str, object]] = []
        self._suffix_counts: Dict[str, int] = {}
        self._physical_names: Dict[str, str] = {}
        self._sam_output_slots: set = set()
        # Slicer segment colors already assigned in this run, so two layers whose
        # suffix hashes collide still render distinctly (deterministic forward probing).
        self._segment_colors_in_use: set = set()
        # low-quality NRRDs now mirror the full-quality decomposition instead of being one
        # combined volume written at the tail. Each downbin spec gets its own
        # one-single-layer-NRRD-per-component decomposition under low_quality/<token>/nrrd/ restored
        # from the same NrrdLayerRef and submitted here as each view completes (identical scheduling).
        self.low_quality_specs: List['LowQualityDownbinSpec'] = [
            spec for spec in (low_quality_specs or ()) if self._layer_formats(spec)
        ]
        self.low_quality_root = Path(low_quality_root) if low_quality_root is not None else None
        self._lq_manifests: Dict[str, List[Dict[str, object]]] = {}
        if self.low_quality_specs:
            if self.low_quality_root is None:
                raise RuntimeError(
                    'low-quality NRRD specs require a low_quality_root'
                )
            for spec in self.low_quality_specs:
                if 'nrrd' in self._layer_formats(spec):
                    self._lq_nrrd_dir(spec).mkdir(parents=True, exist_ok=True)
        if self.has_review_archives:
            if self._review_stage:
                self._review_sources_dir = Path(self._review_stage) / f'conf_{self.cutoff_cent:03d}' / '.layers'
            else:
                scratch = Path(review_scratch_dir) if review_scratch_dir is not None else None
                if scratch is not None:
                    scratch.mkdir(parents=True, exist_ok=True)
                self._review_temp = tempfile.TemporaryDirectory(prefix='xta-review-layers-', dir=scratch)
                self._review_sources_dir = Path(self._review_temp.name)
            if 'xtareview' in self.full_quality_formats:
                (self._review_sources_dir / 'native').mkdir(parents=True, exist_ok=True)
            for spec in self.low_quality_specs:
                if 'xtareview' in self._layer_formats(spec):
                    (self._review_sources_dir / 'low_quality' / str(spec.token)).mkdir(parents=True, exist_ok=True)
        # Construct the executor only after every validating/construction-time filesystem
        # operation has succeeded. A failed mkdir can therefore never strand worker threads
        # in an object that the pipeline did not receive and cannot shut down.
        self._owns_executor = executor is None
        self.executor = executor if executor is not None else ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix='nrrd-layer',
        )

    def _lq_nrrd_dir(self, spec: 'LowQualityDownbinSpec') -> Path:
        if self.low_quality_root is None:
            raise RuntimeError('low-quality NRRD directory requested without a low_quality_root')
        return self.low_quality_root / str(spec.token) / 'nrrd'

    @staticmethod
    def _layer_formats(spec):
        return frozenset(getattr(spec, 'formats', ()) or ('nrrd',)) & {'nrrd', 'xtareview'}

    @property
    def has_review_archives(self):
        return ('xtareview' in self.full_quality_formats
                or any('xtareview' in self._layer_formats(spec) for spec in self.low_quality_specs))

    def published_paths(self):
        """Settled public output manifests and editable archives, grouped by grid."""
        return {grid: dict(paths) for grid, paths in self._published_paths.items()}

    def staged_review_grids(self):
        return tuple(self._staged_archives)

    def set_review_source(self, source_record, *, axial_crop=None):
        if source_record is not None:
            self._review_source_record.update(source_record)
        self._review_axial_crop = axial_crop

    def _review_layer_path(self, grid_id, basename):
        if self._review_sources_dir is None:
            raise RuntimeError('Review layer requested without a staging owner')
        return self._review_sources_dir / grid_id / f'{basename}.cvol'

    def _review_layer_paths(self, native, mirrors):
        paths = {}
        if 'xtareview' in self.full_quality_formats:
            paths['native'] = self._review_layer_path('native', native['filename'])
        for spec in self.low_quality_specs:
            token = str(spec.token)
            if token in mirrors and 'xtareview' in self._layer_formats(spec):
                grid = f'low_quality/{token}'
                paths[grid] = self._review_layer_path(grid, mirrors[token]['filename'])
        return paths

    def _write_review_archives(self, native, mirrors):
        if not self.has_review_archives:
            return
        from .review_archive import write_review_archive
        from .review_layer_sources import review_layers

        grid_specs = []
        if 'xtareview' in self.full_quality_formats:
            grid_specs.append(('native', self.output_root, self.output_shape, native))
        for spec in self.low_quality_specs:
            if 'xtareview' in self._layer_formats(spec):
                token = str(spec.token)
                grid_specs.append((f'low_quality/{token}', self.low_quality_root / token,
                                   tuple(spec.output_shape_t_y_x), mirrors.get(token, ())))
        try:
            for grid, public_dir, shape, entries in grid_specs:
                if self._review_stage:
                    path = Path(self._review_stage) / f'conf_{self.cutoff_cent:03d}' / f'{grid}.xtareview'
                else:
                    path = public_dir / f'{self.stem}.xtareview'
                header = nrrd_slicer_header(shape)
                geometry = dict(space=header['space'], directions_xyz=header['space directions'].tolist(),
                                origin_xyz=header['space origin'].tolist())
                review_id = self.run_id or self.stem
                layers = review_layers(entries, [self._review_layer_path(grid, row['filename']) for row in entries],
                                       shape, self.cutoff_cent, review_id, grid)
                try:
                    write_review_archive(path, review_id=review_id, grid_id=grid, shape_tyx=shape,
                                         native_shape_tyx=self.output_shape, geometry=geometry,
                                         cutoffs_cent=(self.cutoff_cent,), layers=layers,
                                         source_record=self._review_source_record, axial_crop=self._review_axial_crop)
                finally:
                    layers.close()
                if self._review_stage:
                    self._staged_archives.append(grid)
                else:
                    self._published_paths.setdefault(grid, {})['xtareview'] = path
        finally:
            if self._review_temp is not None:
                self._review_temp.cleanup()

    def _segment_color_for_suffix(self, unique_suffix: str) -> Tuple[float, float, float]:
        # Called with self._lock held. Deterministic palette pick keyed off the layer suffix
        # (stable across runs), probing forward past colors already used this run; beyond the
        # palette size, a golden-ratio HSV walk keeps every additional layer distinct.
        palette = _SLICER_SEGMENT_COLOR_PALETTE
        base = _stable_layer_color_index(str(unique_suffix))
        for probe in range(len(palette)):
            color = palette[(base + probe) % len(palette)]
            if color not in self._segment_colors_in_use:
                self._segment_colors_in_use.add(color)
                return color
        hue = (float(base % 4096) / 4096.0 + 0.61803398875 * float(len(self._segment_colors_in_use))) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        color = (round(float(r), 6), round(float(g), 6), round(float(b), 6))
        self._segment_colors_in_use.add(color)
        return color

    def _layer_provenance(self, ref, *, relative_to):
        # Grouped sweep sources are staged on scratch, which can be on another
        # volume. Their final catalog rebases these references on publication.
        return interpolation_layer_provenance(
            ref, relative_to=relative_to if self.relative_artifact_paths else None,
        )

    def submit_layer(self, ref: Optional['NrrdLayerRef'], suffix: str) -> Optional[Path]:
        if ref is None:
            return None
        requested_suffix = str(suffix)
        provenance = self._layer_provenance(ref, relative_to=self.nrrd_dir)
        if ref.mask_kind == 'extrapolation':
            tail = dict(getattr(ref, 'extrapolation_provenance', {}) or {})
            if tail.get('direction') not in {'forward', 'backward'} or tail.get('backend') != 'sam':
                raise ValueError('Extrapolation output requires explicit SAM tail provenance')
            suffix = nrrd_layer_output_suffix(view_token=ref.view_name, source=ref.source,
                mask_kind='extrapolation', tile_config_id=ref.tile_config_id,
                stage='sam_extrapolation_' + tail['direction'])
        if str(getattr(ref, 'interpolation_backend', '')).lower() == 'sam':
            if ref.mask_kind == 'bridge' and ref.proposal_selection_status != 'policy_selected':
                raise ValueError('SAM directional output requires completed proposal-quality selection')
            if ref.mask_kind == 'bridge' and ref.interpolation_direction not in {'forward', 'backward'}:
                raise ValueError('SAM directional output requires an explicit view-native direction')
            if ref.mask_kind == 'bridge' and '_sam_bridge_' not in str(suffix):
                # Guard older sink call sites: provenance lives on the reference
                # and must survive projection even when a legacy suffix is supplied.
                suffix = nrrd_layer_output_suffix(
                    view_token=ref.view_name, source=ref.source, mask_kind=ref.mask_kind,
                    pass_index=ref.pass_index, tile_config_id=ref.tile_config_id,
                    interpolation_backend=ref.interpolation_backend,
                    interpolation_direction=ref.interpolation_direction,
                    seed_detector_identity=ref.seed_detector_identity,
                    sam_bundle_identity=ref.sam_bundle_identity,
                    interpolation_policy_identity=ref.interpolation_policy_identity,
                )
        layer_role = str(getattr(ref, 'layer_role', 'additive_component'))
        recomposition_op = str(getattr(ref, 'recomposition_op', 'union'))
        low_quality_recomposition_op = str(
            getattr(ref, 'low_quality_recomposition_op', recomposition_op)
        )
        mirror_low_quality = bool(getattr(ref, 'mirror_low_quality', True))
        with self._lock:
            if ref.mask_kind == 'bridge' and ref.interpolation_backend == 'sam':
                slot = (ref.model_name, ref.view_name, ref.aug_id, ref.angle_deg, ref.source,
                        ref.tile_config_id, ref.pass_index, ref.interpolation_direction,
                        ref.sam_bundle_identity, ref.interpolation_policy_identity)
                if slot in self._sam_output_slots:
                    raise ValueError('Duplicate SAM directional output slot for the same interpolation scope/pass')
                self._sam_output_slots.add(slot)
            seen = int(self._suffix_counts.get(str(suffix), 0))
            self._suffix_counts[str(suffix)] = seen + 1
            unique_suffix = str(suffix) if seen == 0 else f'{suffix}_{seen + 1:02d}'
            # Keep full logical segment labels and palette identity even when a
            # long physical basename is shortened for portable loose transfer.
            segment_name = f'{self.stem}_{unique_suffix}'
            basename = nrrd_layer_file_basename(self.stem, unique_suffix)
            previous_name = self._physical_names.setdefault(basename.casefold(), segment_name)
            if previous_name != segment_name:
                raise ValueError('NRRD physical filename collision between distinct logical layer names')
            out_path = self.nrrd_dir / basename
            segment_color = self._segment_color_for_suffix(unique_suffix)
            manifest_entry: Dict[str, object] = {
                'filename': out_path.name,
                'layer_key': getattr(ref, 'key', ''),
                'model_name': getattr(ref, 'model_name', ''),
                'suffix': unique_suffix,
                'view_name': getattr(ref, 'view_name', ''),
                'physical_view_name': getattr(ref, 'physical_view_name', ''),
                'tta_aug_id': getattr(ref, 'aug_id', ''),
                'tta_angle_deg': float(getattr(ref, 'angle_deg', 0.0)),
                'view_family': getattr(ref, 'view_family', ''),
                'source': getattr(ref, 'source', ''),
                'mask_kind': getattr(ref, 'mask_kind', ''),
                'pass_index': int(getattr(ref, 'pass_index', 0)),
                'interpolation_walk_back_index': int(getattr(ref, 'interpolation_walk_back_index', 0)),
                'interpolation_candidate_index': int(getattr(ref, 'interpolation_candidate_index', 0)),
                'tile_config_id': getattr(ref, 'tile_config_id', ''),
                'tile_acceptance': getattr(ref, 'tile_acceptance', ''),
                'stage': getattr(ref, 'stage', ''),
                'description': getattr(ref, 'description', ''),
                'layer_role': layer_role,
                'recomposition_op': recomposition_op,
                'backing_shape_tyx': [int(v) for v in getattr(ref, 'shape', (0, 0, 0))],
                'output_shape_tyx': [int(v) for v in self.output_shape],
                'stored_shape_tyx': None,
                **({'stored_offset_tyx': None, 'stored_layout': None}
                   if self.layer_layout == NRRD_LAYER_LAYOUT_CROPPED else {}),
                'empty_segment': None,
                'exported_axes': '(X, Y, t)',
                'segment_name': segment_name,
                'segment_color_rgb': [round(float(c), 6) for c in segment_color],
                # resolves the actual count inside the executing writer. Keeping the
                # policy explicit avoids recording the stale submit-time pool occupancy.
                'z_shards': None,
                'z_shards_policy': 'execution_time',
            }
            manifest_entry.update(provenance)
            self._manifest.append(manifest_entry)
            # mirror this component layer into each low-quality downbin decomposition,
            # scheduled now (as the view completes) exactly like the full-quality layer and sharing
            # the same unique suffix so a low-quality layer maps 1:1 to its full-quality layer.
            # the mirrors are derived from the full-quality payload blocks inside ONE
            # combined write task (write_layer_nrrd_with_low_quality_mirrors) instead of each
            # re-decoding the full-resolution layer store as an independent task.
            lq_mirror_args: List[Tuple[Tuple[int, int, int], Path]] = []
            mirror_nrrd_paths: List[Path] = []
            lq_manifest_entries: List[Tuple[Tuple[int, int, int], Dict[str, object]]] = []
            for spec in self.low_quality_specs if mirror_low_quality else []:
                if self.low_quality_root is None:
                    break
                lq_shape = (
                    int(spec.output_shape_t_y_x[0]),
                    int(spec.output_shape_t_y_x[1]),
                    int(spec.output_shape_t_y_x[2]),
                )
                lq_path = self._lq_nrrd_dir(spec) / basename
                lq_mirror_args.append((lq_shape, lq_path))
                if 'nrrd' in self._layer_formats(spec):
                    mirror_nrrd_paths.append(lq_path)
                lq_manifest_entry: Dict[str, object] = {
                    'filename': lq_path.name,
                    'layer_key': getattr(ref, 'key', ''),
                    'model_name': getattr(ref, 'model_name', ''),
                    'suffix': unique_suffix,
                    'view_name': getattr(ref, 'view_name', ''),
                    'physical_view_name': getattr(ref, 'physical_view_name', ''),
                    'tta_aug_id': getattr(ref, 'aug_id', ''),
                    'tta_angle_deg': float(getattr(ref, 'angle_deg', 0.0)),
                    'view_family': getattr(ref, 'view_family', ''),
                    'source': getattr(ref, 'source', ''),
                    'mask_kind': getattr(ref, 'mask_kind', ''),
                    'pass_index': int(getattr(ref, 'pass_index', 0)),
                    'interpolation_walk_back_index': int(getattr(ref, 'interpolation_walk_back_index', 0)),
                    'interpolation_candidate_index': int(getattr(ref, 'interpolation_candidate_index', 0)),
                    'tile_config_id': getattr(ref, 'tile_config_id', ''),
                    'tile_acceptance': getattr(ref, 'tile_acceptance', ''),
                    'stage': getattr(ref, 'stage', ''),
                    'description': getattr(ref, 'description', ''),
                    'layer_role': layer_role,
                    'recomposition_op': low_quality_recomposition_op,
                    'full_quality_recomposition_op': recomposition_op,
                    'backing_shape_tyx': [int(v) for v in getattr(ref, 'shape', (0, 0, 0))],
                    'output_shape_tyx': [int(v) for v in lq_shape],
                    'stored_shape_tyx': None,
                    **({'stored_offset_tyx': None, 'stored_layout': None}
                       if self.layer_layout == NRRD_LAYER_LAYOUT_CROPPED else {}),
                    'empty_segment': None,
                    'downbin_value': str(spec.raw_value),
                    'downbin_token': str(spec.token),
                    'downbin_scale': float(spec.scale),
                    'exported_axes': '(X, Y, t)',
                    'segment_name': segment_name,
                    'segment_color_rgb': [round(float(c), 6) for c in segment_color],
                }
                lq_manifest_entry.update(self._layer_provenance(ref, relative_to=self._lq_nrrd_dir(spec)))
                self._lq_manifests.setdefault(str(spec.token), []).append(lq_manifest_entry)
                lq_manifest_entries.append((lq_shape, lq_manifest_entry))
            mirror_entries = {str(entry['downbin_token']): entry for _shape, entry in lq_manifest_entries}
            review_paths = self._review_layer_paths(manifest_entry, mirror_entries)

            def _execute_layer_write() -> Path:
                # This closure begins on the sink executor, so its count reflects execution
                # time rather than submission-time queue occupancy. Record that exact value
                # and pass it through to the mirror/full writer as one immutable decision.
                resolved_ref = _resolve_live_ref_extent(ref)
                layout = self.layer_layout
                full_plan = _nrrd_raster_plan(resolved_ref, self.output_shape, layout=layout)
                executed_z_shards = int(_nrrd_layer_zshard_count(
                    resolved_ref, int(full_plan.stored_shape_tyx[0]),
                ))
                lq_plans: Dict[str, NrrdRasterPlan] = {}
                with self._lock:
                    manifest_entry['z_shards'] = int(executed_z_shards)
                    _record_nrrd_stored_plan(manifest_entry, full_plan, layout)
                    composed_mirror_ref = dataclasses_replace(
                        resolved_ref,
                        segment_extent_ijk=_nrrd_mapped_extent_preserve_empty(
                            resolved_ref, self.output_shape,
                        ),
                        segment_extent_shape_tyx=tuple(int(v) for v in self.output_shape),
                        segment_extent_source='composed_full_output_extent_for_low_quality_mirror',
                    )
                    for (lq_shape, lq_entry), (_mirror_shape, lq_path) in zip(
                        lq_manifest_entries, lq_mirror_args,
                    ):
                        lq_plan = _nrrd_raster_plan(composed_mirror_ref, lq_shape, layout=layout)
                        _record_nrrd_stored_plan(lq_entry, lq_plan, layout)
                        lq_plans[str(lq_path)] = lq_plan
                written_plans: Dict[str, NrrdRasterPlan] = {}
                write_args = dict(
                    ref=resolved_ref, output_shape=self.output_shape, out_path=out_path,
                    mirrors=tuple(lq_mirror_args), segment_name=segment_name,
                    segment_color=segment_color, z_shards=int(executed_z_shards),
                    layout=layout, raster_plan=full_plan, mirror_raster_plans=lq_plans,
                )
                if not self.write_native_nrrd:
                    write_args['write_native'] = False
                if 'native' in review_paths:
                    write_args['native_review_path'] = review_paths['native']
                if len(mirror_nrrd_paths) != len(lq_mirror_args):
                    write_args['mirror_nrrd_paths'] = tuple(mirror_nrrd_paths)
                if review_paths and lq_mirror_args:
                    write_args['mirror_review_paths'] = {
                        str(path): review_paths[grid]
                        for (_shape, entry), (_mirror_shape, path) in zip(lq_manifest_entries, lq_mirror_args)
                        if (grid := f"low_quality/{entry['downbin_token']}") in review_paths
                    }
                review_identity = None
                if self.write_native_nrrd and self.managed_review_root is not None and layout == NRRD_LAYER_LAYOUT_CROPPED:
                    from .managed_review_source import capture_native_review_source, review_source_identity

                    review_identity = review_source_identity(resolved_ref, self.output_shape)
                # A worker process writes eligible files with identical bytes and returns
                # the plans it wrote; None (pool absent, broken, or layer ineligible)
                # writes them here.
                processes = self.layer_file_processes
                remote_plans = (
                    processes.write_layer_files(write_args)
                    if processes is not None and not review_paths and processes.accepts(resolved_ref, executed_z_shards)
                    else None
                )
                if remote_plans is not None:
                    written_plans.update(remote_plans)
                    result = out_path
                else:
                    result = write_sink_layer_files(**write_args, plan_receipt=written_plans)
                actual = written_plans.get(str(out_path))
                if layout == NRRD_LAYER_LAYOUT_CROPPED:
                    # An extent-violation fallback republished the full raster; the rows must
                    # describe the files actually on disk.
                    with self._lock:
                        if actual is not None:
                            _record_nrrd_stored_plan(manifest_entry, actual, layout)
                        for (_lq_shape, lq_entry), (_mirror_shape, lq_path) in zip(
                            lq_manifest_entries, lq_mirror_args,
                        ):
                            mirror_actual = written_plans.get(str(lq_path))
                            if mirror_actual is not None:
                                _record_nrrd_stored_plan(lq_entry, mirror_actual, layout)
                if review_identity is not None and actual is not None:
                    capture_native_review_source(
                        resolved_ref, self.output_shape, out_path, actual, self.managed_review_root,
                        expected_identity=review_identity,
                    )
                self.observe_published_layer(ref)
                return result

            fut = self.executor.submit(_execute_layer_write)
            self._futures.append(fut)
            self._source_refs[fut] = ref
            self._publication_records.setdefault((str(ref.key), requested_suffix), []).append((fut, {
                'schema': 'xta.nrrd_layer_publication/1',
                'stem': self.stem,
                'run_id': self.run_id,
                'output_shape_tyx': list(self.output_shape),
                'native': manifest_entry,
                'low_quality': mirror_entries,
                'allocator': {
                    'requested_suffix': str(suffix),
                    'occurrence': seen + 1,
                    'color_rgb': list(segment_color),
                },
            }))
        # add_done_callback runs immediately when the task already finished.
        # Register outside the source-map lock; the callback captures no layer.
        fut.add_done_callback(self._release_completed_source)
        return out_path

    def _release_completed_source(self, future: Future) -> None:
        with self._lock:
            self._source_refs.pop(future, None)

    def set_layer_publication_observer(
        self, observer: Optional[Callable[['NrrdLayerRef'], object]],
    ) -> None:
        """Observe successful final-path publication without resubmitting layers.

        The callback runs on the layer writer before its Future resolves. It must
        not wait for that same layer. Callback errors fail the writer transaction
        and remain visible to the normal sink drain.
        """
        with self._lock:
            self._layer_publication_observer = observer

    def observe_published_layer(self, ref: 'NrrdLayerRef') -> None:
        """Notify the current observer after native and every LQ mirror succeed."""
        with self._lock:
            observer = self._layer_publication_observer
        if observer is not None:
            observer(ref)

    def pending_layer_refs(self) -> Tuple[NrrdLayerRef, ...]:
        """Snapshot every unfinished export's source without opening its payload."""
        with self._lock:
            self._source_refs = {
                future: ref for future, ref in self._source_refs.items()
                if not future.done()
            }
            return tuple(self._source_refs.values())

    def published_layer_record(
        self,
        ref: Optional['NrrdLayerRef'],
        suffix: str,
        *,
        output_path: Optional[Path] = None,
        wait_for_completion: bool = False,
    ) -> Optional[Dict[str, object]]:
        """Snapshot one completed scalar native/LQ publication for later adoption.

        The caller drains its view first. This never waits while holding the sink
        lock, opens layer sources, or invents rows from submit-time geometry. A
        repeated key/suffix requires the actual returned path to disambiguate it.
        """
        if ref is None:
            return None
        with self._lock:
            matches = list(self._publication_records.get((str(ref.key), str(suffix)), ()))
            if output_path is not None:
                path = Path(output_path)
                matches = [(future, record) for future, record in matches
                           if path == self.nrrd_dir / str(record['native']['filename'])]
        if len(matches) != 1:
            raise ValueError('Published layer identity is unavailable or ambiguous')
        future, record = matches[0]
        if not wait_for_completion and not future.done():
            raise RuntimeError('Layer publication is still pending; drain the sink first')
        future.result()
        with self._lock:
            snapshot = copy.deepcopy(record)
        def file_identity(path: Path) -> Dict[str, int]:
            stat = path.stat()
            return dict(size_bytes=int(stat.st_size), modified_time_ns=int(stat.st_mtime_ns),
                        device=int(stat.st_dev), file_id=int(stat.st_ino))
        specs = {str(spec.token): spec for spec in self.low_quality_specs}
        snapshot['files'] = {
            'native': (file_identity(self.nrrd_dir / str(snapshot['native']['filename']))
                       if self.write_native_nrrd else None),
            'low_quality': {
                token: file_identity(self._lq_nrrd_dir(specs[token]) / str(row['filename']))
                for token, row in snapshot['low_quality'].items()
                if 'nrrd' in self._layer_formats(specs[token])
            },
        }
        if self.has_review_archives:
            from .review_layer_sources import layer_file_identities

            snapshot['files']['review'] = {
                grid: layer_file_identities(path)
                for grid, path in self._review_layer_paths(snapshot['native'], snapshot['low_quality']).items()
            }
        return snapshot

    def wait_published_layer(
        self,
        ref: Optional['NrrdLayerRef'],
        suffix: str,
        *,
        output_path: Optional[Path] = None,
    ) -> Optional[Dict[str, object]]:
        """Quietly fence one combined native/LQ writer and snapshot its rows."""
        return self.published_layer_record(ref, suffix, output_path=output_path,
                                          wait_for_completion=True)

    def layer_count(self) -> int:
        with self._lock:
            return len(self._manifest)

    def adopt_published_layer(self, ref, record) -> Optional[Path]:
        """Register an existing cutoff detector publication without rewriting it."""
        from .tta_sweep_branch_manifest import adopt_published_layer
        return adopt_published_layer(self,ref,record,
            basename_for_suffix=nrrd_layer_file_basename,
            provenance_for_directory=self._layer_provenance)

    def centerline_audit_layer_count(self) -> int:
        with self._lock:
            return int(sum(
                1 for entry in self._manifest
                if str(entry.get('source', '')) == 'centerline_filter'
            ))

    def low_quality_layer_count(self) -> int:
        with self._lock:
            return int(sum(len(entries) for entries in self._lq_manifests.values()))

    def low_quality_centerline_audit_layer_count(self) -> int:
        with self._lock:
            return int(sum(
                1
                for entries in self._lq_manifests.values()
                for entry in entries
                if str(entry.get('source', '')) == 'centerline_filter'
            ))

    def progress_counts(self) -> Tuple[int, int]:
        """Return completed and submitted write-task counts without waiting."""
        with self._lock:
            futures = list(self._futures)
        return int(sum(1 for fut in futures if fut.done())), int(len(futures))

    def wait(self) -> None:
        with self._lock:
            futures = list(self._futures)
        total = int(len(futures))
        if total <= 0:
            return
        pending: set[Future] = set()
        first_error: Optional[BaseException] = None
        completed = 0
        for fut in futures:
            if not fut.done():
                pending.add(fut)
                continue
            completed += 1
            try:
                fut.result()
            except BaseException as exc:  # pragma: no cover - surfaced to main
                if first_error is None:
                    first_error = exc
        status_seconds = max(5.0, _env_float('YOLO_TTA_NRRD_WAIT_STATUS_SECONDS', 30.0))
        last_status = time.monotonic()
        print(
            f'Single-layer NRRD write status: {int(completed)}/{total} complete, '
            f'{len(pending)} pending.'
        )
        while pending:
            done, pending_remainder = wait(
                pending,
                timeout=float(status_seconds),
                return_when=FIRST_COMPLETED,
            )
            pending = set(pending_remainder)
            for fut in done:
                completed += 1
                try:
                    fut.result()
                except BaseException as exc:  # pragma: no cover - surfaced to main
                    if first_error is None:
                        first_error = exc
            now = time.monotonic()
            if done or now - float(last_status) >= float(status_seconds):
                print(
                    f'Single-layer NRRD write status: {int(completed)}/{total} complete, '
                    f'{len(pending)} pending.'
                )
                last_status = float(now)
        if first_error is not None:
            raise RuntimeError('Single-layer NRRD writing failed') from first_error

    def shutdown(self) -> None:
        try:
            if self._owns_executor:
                self.executor.shutdown(wait=True)
            else:
                # Several branch sinks may share one bounded encoder pool. Its
                # coordinator owns pool shutdown; this sink still fences every
                # reader and reports failures before releasing source references.
                self.wait()
        finally:
            with self._lock:
                self._source_refs.clear()

    def record_final_bridge_survival(self, receipts: Sequence[Dict[str, object]]) -> None:
        """Attach the final output transaction without reusing native topology claims."""
        keyed = {(str(item.get('model_name', '')), str(item.get('layer_key', ''))): dict(item)
                 for item in receipts}
        with self._lock:
            for entry in self._manifest:
                receipt = keyed.get((str(entry.get('model_name', '')), str(entry.get('layer_key', ''))))
                if receipt is not None:
                    entry['final_output_survival'] = receipt
                    entry['final_connection_survival'] = receipt.get('connection_survival', 'not_assessed')
            # Downbin resampling changes topology. Keep the source receipt as a
            # labelled reference and leave downbin connection survival unverified.
            for entries in self._lq_manifests.values():
                for entry in entries:
                    receipt = keyed.get((str(entry.get('model_name', '')), str(entry.get('layer_key', ''))))
                    if receipt is not None:
                        entry['full_quality_final_output_survival'] = receipt
                        entry['final_connection_survival'] = 'not_assessed_after_downbin'

    def _stored_layout_manifest_fields(self) -> Dict[str, object]:
        """Top-level layout field; empty for the full layout so its manifests stay unchanged."""
        if self.layer_layout != NRRD_LAYER_LAYOUT_CROPPED:
            return {}
        return {'stored_layout': 'segment_bounding_box'}

    def _stored_layout_manifest_notes(self) -> List[str]:
        if self.layer_layout != NRRD_LAYER_LAYOUT_CROPPED:
            return []
        return [
            'Each file stores only its layer\'s segment bounding box (stored_shape_tyx) placed at '
            'stored_offset_tyx (t, Y, X) in output_shape_tyx: the header space origin is that '
            'voxel\'s position, Segmentation_ReferenceImageExtentOffset/Segmentation_ReferenceImageExtent '
            'give the Slicer reference extent, and Segment0_Extent is relative to the stored box. '
            'Voxels outside the box are zero. Empty layers store one zero voxel at offset (0, 0, 0). '
            'Rows whose stored_layout is full_reference store the complete output grid.',
        ]

    def write_manifest(self) -> Optional[Path]:
        with self._lock:
            manifest_layers = list(self._manifest)
            lq_manifests = {token: list(entries) for token, entries in self._lq_manifests.items()}
        if self.has_review_archives:
            self.wait()
            self._write_review_archives(manifest_layers, lq_manifests)
        # one manifest per low-quality downbin decomposition, mirroring the full-quality
        # sidecar so each low-quality nrrd/ folder is self-describing and recomposable on its own.
        if self.low_quality_root is not None:
            for spec in self.low_quality_specs:
                if 'nrrd' not in self._layer_formats(spec):
                    continue
                entries = lq_manifests.get(str(spec.token), [])
                if not entries:
                    continue
                lq_shape = (
                    int(spec.output_shape_t_y_x[0]),
                    int(spec.output_shape_t_y_x[1]),
                    int(spec.output_shape_t_y_x[2]),
                )
                lq_manifest_path = self._lq_nrrd_dir(spec) / f'{self.stem}_nrrd_manifest.json'
                _write_json_atomically(lq_manifest_path, {
                    'layout': 'one_single_layer_nrrd_per_component',
                    **({'run_id': self.run_id} if self.run_id is not None else {}),
                    **self._stored_layout_manifest_fields(),
                    'quality': 'low_quality',
                    'downbin_value': str(spec.raw_value),
                    'downbin_token': str(spec.token),
                    'downbin_scale': float(spec.scale),
                    'exported_axes': '(X, Y, t)',
                    'output_shape_tyx': [int(v) for v in lq_shape],
                    'full_quality_output_shape_tyx': [int(v) for v in self.output_shape],
                    'layer_count': len(entries),
                    'layers': entries,
                    'notes': [
                        'Low-quality distribution: eligible component layers and complete checkpoints '
                        'downbinned to this grid from their full-quality logical raster.',
                        'Each NRRD is one uint8 binary mask in source output geometry (X, Y, t), downbinned.',
                        'Each file is a 3D Slicer segmentation (.seg.nrrd) sharing its full-quality layer\'s logical segment name and color; physical basenames longer than 120 characters are shortened.',
                        'Layer suffixes match the corresponding full-quality layers. Centerline removed-component and watershed-candidate audit layers are mirrored as non-recomposable downbins (diagnostic_only and none, respectively).',
                        'Use each layer\'s recomposition_op. Only union entries marked union; select entries are complete checkpoints.',
                        *self._stored_layout_manifest_notes(),
                    ],
                })
                if self.relative_artifact_paths:
                    self._published_paths.setdefault(f'low_quality/{spec.token}', {})['nrrd'] = lq_manifest_path
        if not manifest_layers or not self.write_native_nrrd:
            return None
        manifest_path = self.nrrd_dir / f'{self.stem}_nrrd_manifest.json'
        manifest = {
            'layout': 'one_single_layer_nrrd_per_component',
            **({'run_id': self.run_id} if self.run_id is not None else {}),
            **self._stored_layout_manifest_fields(),
            'exported_axes': '(X, Y, t)',
            'output_shape_tyx': [int(v) for v in self.output_shape],
            'layer_count': len(manifest_layers),
            'layers': manifest_layers,
            'notes': [
                'Each NRRD is one uint8 binary mask in source output geometry (X, Y, t).',
                'Each file is a 3D Slicer segmentation (.seg.nrrd) holding one full logical segment label; segment_color_rgb records the assigned Slicer color. Physical basenames longer than 120 characters are shortened.',
                'Recomposition is role-aware: union only layers whose recomposition_op is union; select chooses a complete checkpoint; subtract_from_previous_checkpoint removes that full-quality delta; none is diagnostic only.',
                'YOLO layers are cleaned masks before interpolation bridges; bridge layers contain only voxels added by that pass.',
                'Global checkpoints, including centerline pass00_input/result and Global_final_output, are complete alternatives and must never be unioned with component or audit layers.',
                'The manifest lists only this run. In a reused output directory, similarly named files absent from this manifest are stale and must be ignored.',
                *self._stored_layout_manifest_notes(),
            ],
        }
        _write_json_atomically(manifest_path, manifest)
        if self.relative_artifact_paths:
            self._published_paths.setdefault('native', {})['nrrd'] = manifest_path
        return manifest_path

_NRRD_LAYER_SINK: Optional[NrrdLayerSink] = None

def set_nrrd_layer_sink(sink: Optional[NrrdLayerSink]) -> None:
    global _NRRD_LAYER_SINK
    _NRRD_LAYER_SINK = sink

def nrrd_layer_sink() -> Optional[NrrdLayerSink]:
    return _NRRD_LAYER_SINK

def nrrd_layer_sink_workers() -> int:
    # default cap 4 -> 12. With ~72 queued files (layers x low-quality mirrors) the
    # 4-slot pool left the box near-idle through the whole write tail; most of the write volume
    # lands after inference has drained, so a wider pool is safe.
    cores = max(1, _cpu_count())
    default_workers = max(1, min(12, max(1, cores // 8)))
    return max(1, _env_int('YOLO_TTA_NRRD_LAYER_SINK_WORKERS', int(default_workers)))

def _nrrd_layer_ref_is_raw_bbox_store(ref: NrrdLayerRef) -> bool:
    return str(getattr(ref, 'storage_format', 'raw_u8')) in MASK_STORE_FORMATS or Path(ref.path).is_dir()

class _LiveArrayLayerSource:
    """Read adapter over an in-process live layer volume.

 Duck-typed for the payload streamer (shape + slicing); NOT a RawBBoxMaskStore, so the
 zero-copy direct-native branch applies when shapes match. The wrapped array is owned by
 the caller (e.g. the final union volume) and must never be closed by the sink."""

    def __init__(self, arr: np.ndarray) -> None:
        self.array = np.asarray(arr)
        self.shape = tuple(int(x) for x in self.array.shape)

    def __getitem__(self, item: object) -> np.ndarray:
        return self.array[item]

def _resolve_live_ref_extent(ref: NrrdLayerRef) -> NrrdLayerRef:
    """Compute a live-volume layer's deferred segment extent (idempotent).

 Live refs carry segment_extent_ijk=None so the producing thread never pays the scan;
 the first sink worker that needs the header resolves it here (one GIL-releasing
 reduction pass over the live volume) and downstream uses the returned ref."""
    if (
        getattr(ref, 'live_array', None) is None
        or _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None)) is not None
    ):
        return ref
    live_arr = np.asarray(ref.live_array)
    return dataclasses_replace(
        ref,
        segment_extent_ijk=compute_segment_extent_zyx(live_arr),
        segment_extent_shape_tyx=tuple(int(x) for x in live_arr.shape),
        segment_extent_source='live_volume_sink_scan',
    )

def _open_nrrd_layer_ref(ref: NrrdLayerRef) -> object:
    live = getattr(ref, 'live_array', None)
    if live is not None:
        return _LiveArrayLayerSource(live)  #
    if _nrrd_layer_ref_is_raw_bbox_store(ref):
        return RawBBoxMaskStore.open(
            ref.path,
            cache_payload_in_ram=True,
        )
    return np.memmap(
        ref.path,
        dtype=np.dtype(ref.dtype),
        mode='r',
        shape=tuple(int(x) for x in ref.shape),
    )

def _close_nrrd_layer_source(src: object) -> None:
    if isinstance(src, _LiveArrayLayerSource):
        return  # the live volume belongs to the caller; never close it here
    if isinstance(src, RawBBoxMaskStore):
        src.close()
        return
    close_memmap_array(src)

def _drop_nrrd_raw_store_chunks_ram_cache(src: object) -> None:
    if not isinstance(src, RawBBoxMaskStore):
        return
    # release (refcount) rather than evict — sibling full-quality /
    # low-quality writers for the same layer may still hold the entry.
    if not bool(getattr(src, '_ram_cache_ref_held', False)):
        return
    src._ram_cache_ref_held = False
    _release_raw_store_chunks_ram_cache(src.chunks_path)

def _compute_segment_extent_zyx_serial(src: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """Return the exact Slicer segment extent through the serial fallback scan."""
    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    min_t, max_t = t_dim, -1
    min_y, max_y = h_dim, -1
    min_x, max_x = w_dim, -1

    for t_idx in range(t_dim):
        sl = np.asarray(src[int(t_idx)], dtype=bool)
        if not np.any(sl):
            continue
        row_has_fg = np.any(sl, axis=1)
        col_has_fg = np.any(sl, axis=0)
        ys = np.flatnonzero(row_has_fg)
        xs = np.flatnonzero(col_has_fg)
        if xs.size <= 0 or ys.size <= 0:
            continue
        min_t = min(min_t, int(t_idx))
        max_t = max(max_t, int(t_idx))
        min_y = min(min_y, int(ys[0]))
        max_y = max(max_y, int(ys[-1]))
        min_x = min(min_x, int(xs[0]))
        max_x = max(max_x, int(xs[-1]))

    if max_t < 0:
        return _nrrd_empty_segment_extent()
    return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))

def compute_segment_extent_zyx(
    mask_zyx: np.ndarray,
    *,
    workers: Optional[int] = None,
) -> Tuple[int, int, int, int, int, int]:
    """Return the exact Slicer segment extent through parallel per-slice reductions."""
    src = np.asarray(mask_zyx)
    if src.ndim != 3:
        raise ValueError(f'compute_segment_extent_zyx expects a 3D (t,Y,X) layer, got {src.shape}')

    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    requested_workers = max(1, int(_cpu_count() if workers is None else workers))
    scan_workers = choose_slice_parallel_workers(int(requested_workers), int(t_dim))
    if int(t_dim) <= 1 or int(scan_workers) <= 1:
        return _compute_segment_extent_zyx_serial(src)

    # [min_y, max_y, min_x, max_x] for each t slice. Empty slices retain the
    # max<min sentinel. Each task owns one row, so there is no shared reduction race.
    slice_bounds = np.empty((int(t_dim), 4), dtype=np.int64)
    slice_bounds[:, 0] = np.int64(h_dim)
    slice_bounds[:, 1] = np.int64(-1)
    slice_bounds[:, 2] = np.int64(w_dim)
    slice_bounds[:, 3] = np.int64(-1)

    def _scan_slice(t_idx: int) -> None:
        # Preserve the serial path's truth-value conversion exactly; the pipeline stores
        # uint8 masks, but this also keeps the fallback-equivalence contract for any caller.
        sl = np.asarray(src[int(t_idx)], dtype=bool)
        if not np.any(sl):
            return
        row_has_fg = np.any(sl, axis=1)
        col_has_fg = np.any(sl, axis=0)
        ys = np.flatnonzero(row_has_fg)
        xs = np.flatnonzero(col_has_fg)
        if xs.size <= 0 or ys.size <= 0:
            return
        slice_bounds[int(t_idx), :] = (
            int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1]),
        )

    parallel_for_indices_chunked(
        int(t_dim),
        _scan_slice,
        max_workers=int(scan_workers),
        desc='NRRD segment extent scan',
        show_progress=False,
        target_chunks_per_worker=2,
    )

    nonempty = slice_bounds[:, 1] >= 0
    nonempty_t = np.flatnonzero(nonempty)
    if nonempty_t.size <= 0:
        return _nrrd_empty_segment_extent()
    active = slice_bounds[nonempty]
    return (
        int(np.min(active[:, 2])),
        int(np.max(active[:, 3])),
        int(np.min(active[:, 0])),
        int(np.max(active[:, 1])),
        int(nonempty_t[0]),
        int(nonempty_t[-1]),
    )

def _restore_source_indices_for_output_z(in_t: int, out_t: int, out_z: int) -> List[int]:
    in_t_i = max(1, int(in_t))
    out_t_i = max(1, int(out_t))
    out_z_i = int(np.clip(int(out_z), 0, out_t_i - 1))
    if in_t_i >= out_t_i:
        src_start = int(math.floor(float(out_z_i) * float(in_t_i) / float(out_t_i)))
        src_stop = int(math.ceil(float(out_z_i + 1) * float(in_t_i) / float(out_t_i)))
        src_start = int(np.clip(src_start, 0, in_t_i - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t_i))
        return [int(v) for v in range(src_start, src_stop)]

    return [_center_aligned_nearest_source_index(out_z_i, out_t_i, in_t_i)]

def _resize_binary_mask_frame_to_output_shape(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return resize_binary_plane_to_output_xy(frame, int(out_h), int(out_w))

@lru_cache(maxsize=32)
def _nrrd_sparse_resize_axis_map(
    in_count: int,
    out_count: int,
    area: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cached global-coordinate source support for every output coordinate.

 The returned inclusive-start/exclusive-stop arrays hold the center-aligned nearest
 table shared with the dense restore, or the positive-support footprint of INTER_AREA.
 Keeping this map global is what lets resize a bbox independently without shifting
 its phase to the crop origin."""
    in_n = max(1, int(in_count))
    out_n = max(1, int(out_count))
    coords = np.arange(int(out_n), dtype=np.int64)
    if bool(area):
        starts = (coords * np.int64(in_n)) // np.int64(out_n)
        stops = (
            ((coords + np.int64(1)) * np.int64(in_n) + np.int64(out_n - 1))
            // np.int64(out_n)
        )
    else:
        starts = _center_aligned_nearest_source_indices(int(in_n), int(out_n)).astype(np.int64)
        stops = starts + np.int64(1)
    starts.setflags(write=False)
    stops.setflags(write=False)
    return starts, stops

@_numba.njit(cache=True, nogil=True, inline='always')  # type: ignore[misc]
def _numba_nrrd_continuous_integral_at(
    integral: np.ndarray,
    crop: np.ndarray,
    y_coord: float,
    x_coord: float,
) -> float:  # pragma: no cover - compiled implementation
    crop_h = int(crop.shape[0])
    crop_w = int(crop.shape[1])
    yy = min(float(crop_h), max(0.0, float(y_coord)))
    xx = min(float(crop_w), max(0.0, float(x_coord)))
    yy_round = round(yy)
    xx_round = round(xx)
    if abs(yy - yy_round) < 1e-12:
        yy = float(yy_round)
    if abs(xx - xx_round) < 1e-12:
        xx = float(xx_round)
    iy = int(math.floor(yy))
    ix = int(math.floor(xx))
    fy = float(yy - iy)
    fx = float(xx - ix)
    iy_cell = min(int(iy), int(crop_h - 1))
    ix_cell = min(int(ix), int(crop_w - 1))
    value = float(integral[int(iy), int(ix)])
    value += float(
        integral[int(iy), int(ix_cell + 1)]
        - integral[int(iy), int(ix_cell)]
    ) * fx
    value += float(
        integral[int(iy_cell + 1), int(ix)]
        - integral[int(iy_cell), int(ix)]
    ) * fy
    value += float(crop[int(iy_cell), int(ix_cell)]) * fy * fx
    return float(value)

@_numba.njit(cache=True, nogil=True, inline='always')  # type: ignore[misc]
def _numba_opencv_area_axis_tab(
    out_i: int,
    in_n: int,
    out_n: int,
) -> Tuple[float, float, float, int, int]:  # pragma: no cover - compiled implementation
    """cv::computeResizeAreaTab bounds ``(fs1, fs2, cellWidth, s1, s2)`` of one output pixel."""
    scale = 1.0 / (float(out_n) / float(in_n))
    fs1 = float(out_i) * scale
    fs2 = fs1 + scale
    cell = min(scale, float(in_n) - fs1)
    s2 = min(int(math.floor(fs2)), int(in_n) - 1)
    s1 = min(int(math.ceil(fs1)), s2)
    return fs1, fs2, cell, s1, s2

@_numba.njit(cache=True, nogil=True, inline='always')  # type: ignore[misc]
def _numba_opencv_area_axis_span(
    out_i: int,
    in_n: int,
    out_n: int,
) -> Tuple[float, float, float]:  # pragma: no cover - compiled implementation
    """Global source interval ``[lo, hi)`` and cellWidth of one INTER_AREA output pixel.

 Mirrors cv::computeResizeAreaTab: a leading or trailing fractional source pixel counts
 only when its sliver exceeds 1e-3 px, and weights are normalized by cellWidth."""
    fs1, fs2, cell, s1, s2 = _numba_opencv_area_axis_tab(out_i, in_n, out_n)
    lo = fs1 if float(s1) - fs1 > 1e-3 else float(s1)
    hi = float(s2)
    if fs2 - float(s2) > 1e-3:
        hi += min(fs2 - float(s2), 1.0, cell)
    return lo, hi, cell

@_numba.njit(cache=True, nogil=True, inline='always')  # type: ignore[misc]
def _numba_opencv_area_axis_alpha(
    source_i: int,
    fs1: float,
    fs2: float,
    cell: float,
    s1: int,
    s2: int,
) -> np.float32:  # pragma: no cover - compiled implementation
    """computeResizeAreaTab's float32 alpha of source pixel ``source_i`` in ``[s1-1, s2]``; 0 if no tap."""
    if source_i == s1 - 1:
        if float(s1) - fs1 > 1e-3:
            return np.float32((float(s1) - fs1) / cell)
        return np.float32(0.0)
    if source_i == s2:
        if fs2 - float(s2) > 1e-3:
            return np.float32(min(fs2 - float(s2), 1.0, cell) / cell)
        return np.float32(0.0)
    return np.float32(1.0 / cell)

@_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def _numba_opencv_area_binary_value(
    crop: np.ndarray,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    source_y0: int,
    source_x0: int,
    out_y: int,
    out_x: int,
) -> float:  # pragma: no cover - compiled implementation
    """cv2's float32 INTER_AREA value of one output pixel of the 255-scaled binary frame.

 Replays cv::resize's taps in OpenCV's order and float32 precision: each source row is
 reduced along X, then rows accumulate along Y. Source pixels outside the crop are zero
 and would only add exact zeros, so they are skipped."""
    fy1, fy2, cell_y, y1, y2 = _numba_opencv_area_axis_tab(out_y, in_h, out_h)
    fx1, fx2, cell_x, x1, x2 = _numba_opencv_area_axis_tab(out_x, in_w, out_w)
    total = np.float32(0.0)
    for sy in range(max(y1 - 1, source_y0), min(y2 + 1, source_y0 + crop.shape[0])):
        beta = _numba_opencv_area_axis_alpha(sy, fy1, fy2, cell_y, y1, y2)
        if beta == np.float32(0.0):
            continue
        row = sy - source_y0
        buf = np.float32(0.0)
        for sx in range(max(x1 - 1, source_x0), min(x2 + 1, source_x0 + crop.shape[1])):
            if crop[row, sx - source_x0] == 0:
                continue
            alpha = _numba_opencv_area_axis_alpha(sx, fx1, fx2, cell_x, x1, x2)
            buf = np.float32(buf + np.float32(np.float32(255.0) * alpha))
        total = np.float32(total + np.float32(beta * buf))
    return float(total)

@_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
def _numba_nrrd_area_crop_resize_kernel(
    integral: np.ndarray,
    crop: np.ndarray,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    source_y0: int,
    source_x0: int,
    source_y1: int,
    source_x1: int,
    out_y0: int,
    out_x0: int,
    restored: np.ndarray,
) -> None:  # pragma: no cover - compiled implementation
    # Column spans are row-invariant; compute them once instead of once per pixel.
    width = int(restored.shape[1])
    sx0s = np.empty(width, dtype=np.float64)
    sx1s = np.empty(width, dtype=np.float64)
    cells_x = np.empty(width, dtype=np.float64)
    for local_x in range(width):
        lo_x, hi_x, cell_x = _numba_opencv_area_axis_span(int(out_x0 + local_x), in_w, out_w)
        sx0s[local_x] = max(lo_x, float(source_x0)) - float(source_x0)
        sx1s[local_x] = max(sx0s[local_x], min(hi_x, float(source_x1)) - float(source_x0))
        cells_x[local_x] = cell_x
    for local_y in range(int(restored.shape[0])):
        out_y = int(out_y0 + local_y)
        lo_y, hi_y, cell_y = _numba_opencv_area_axis_span(out_y, in_h, out_h)
        sy0 = max(lo_y, float(source_y0)) - float(source_y0)
        sy1 = max(sy0, min(hi_y, float(source_y1)) - float(source_y0))
        for local_x in range(width):
            out_x = int(out_x0 + local_x)
            sx0 = sx0s[local_x]
            sx1 = sx1s[local_x]
            cell_x = cells_x[local_x]
            weighted = _numba_nrrd_continuous_integral_at(integral, crop, sy1, sx1)
            weighted -= _numba_nrrd_continuous_integral_at(integral, crop, sy0, sx1)
            weighted -= _numba_nrrd_continuous_integral_at(integral, crop, sy1, sx0)
            weighted += _numba_nrrd_continuous_integral_at(integral, crop, sy0, sx0)
            # cv2 rounds the 255-scaled float32 value half-to-even, so a tie stays 0. Near
            # the threshold the float64 integral cannot tell which side cv2's float32 sum
            # lands on (exact ties occur, e.g. 1020 -> 1019), so replay that sum there.
            threshold = cell_y * cell_x / 510.0
            if abs(float(weighted) - threshold) > 1e-4 * threshold:
                inside = float(weighted) > threshold
            else:
                inside = _numba_opencv_area_binary_value(
                    crop, in_h, in_w, out_h, out_w, source_y0, source_x0, out_y, out_x,
                ) > 0.5
            restored[int(local_y), int(local_x)] = np.uint8(1 if inside else 0)


def _resize_sparse_binary_crop_to_output_region(
    crop: np.ndarray,
    *,
    source_shape: Tuple[int, int],
    source_bbox: Tuple[int, int, int, int],
    output_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
    """Resize one binary source bbox in the *full frame's* coordinate system.

 Returns ``(out_y0, out_x0, out_y1, out_x1, crop)``. The output region includes every
 output pixel whose global sampling footprint can see the source bbox, including the
 one-pixel influence halo that an area downscale can create. Work is proportional to
 the affected region: OpenCV's integral-image kernel and the compiled area sampler
 avoid a full zero-plane resize."""
    in_h, in_w = (max(1, int(v)) for v in source_shape)
    out_h, out_w = (max(1, int(v)) for v in output_shape)
    y0, x0, y1, x1 = (int(v) for v in source_bbox)
    crop_arr = np.asarray(crop)
    # RawBBoxMaskStore and restored member planes are already canonical uint8 0/1.
    # Preserve their view (including a bbox-width row stride) instead of making the old
    # compare + astype copy. Non-pipeline callers retain normalization below.
    if crop_arr.dtype == np.uint8:
        crop_u8 = crop_arr
    else:
        crop_u8 = np.ascontiguousarray(crop_arr > 0, dtype=np.uint8)
    if (
        y0 < 0 or x0 < 0 or y1 > int(in_h) or x1 > int(in_w)
        or y1 <= y0 or x1 <= x0
        or tuple(int(v) for v in crop_u8.shape) != (int(y1 - y0), int(x1 - x0))
    ):
        raise ValueError(
            f'Invalid sparse resize bbox {(y0, x0, y1, x1)} / crop {crop_u8.shape} '
            f'for source {(in_h, in_w)}'
        )

    if int(in_h) == int(out_h) and int(in_w) == int(out_w):
        return int(y0), int(x0), int(y1), int(x1), np.ascontiguousarray(crop_u8)

    use_area = bool(int(in_h) >= int(out_h) and int(in_w) >= int(out_w))
    y_starts, y_stops = _nrrd_sparse_resize_axis_map(int(in_h), int(out_h), bool(use_area))
    x_starts, x_stops = _nrrd_sparse_resize_axis_map(int(in_w), int(out_w), bool(use_area))

    # Support intervals are monotonic. searchsorted finds the affected output bbox without
    # allocating an out_h/out_w boolean mask for every source crop.
    out_y0 = int(np.searchsorted(y_stops, int(y0), side='right'))
    out_y1 = int(np.searchsorted(y_starts, int(y1), side='left'))
    out_x0 = int(np.searchsorted(x_stops, int(x0), side='right'))
    out_x1 = int(np.searchsorted(x_starts, int(x1), side='left'))
    out_y0 = max(0, min(int(out_y0), int(out_h)))
    out_y1 = max(int(out_y0), min(int(out_y1), int(out_h)))
    out_x0 = max(0, min(int(out_x0), int(out_w)))
    out_x1 = max(int(out_x0), min(int(out_x1), int(out_w)))
    if int(out_y1) <= int(out_y0) or int(out_x1) <= int(out_x0):
        return None

    if not bool(use_area):
        src_y = y_starts[int(out_y0):int(out_y1)] - np.int64(y0)
        src_x = x_starts[int(out_x0):int(out_x1)] - np.int64(x0)
        restored = crop_u8[src_y[:, None], src_x[None, :]]
    else:
        # Reproduce INTER_AREA's global phase and uint8 rounding threshold without padding
        # a full source plane. cv2.integral is a compiled, GIL-releasing prefix pass over
        # the bbox; evaluating its continuous piecewise-constant integral accounts for
        # fractional first/last source pixels, and like cv::computeResizeAreaTab the
        # kernel drops edge slivers of at most 1e-3 source px.
        integral = cv2.integral(crop_u8, sdepth=cv2.CV_32S)
        restored = np.empty(
            (int(out_y1 - out_y0), int(out_x1 - out_x0)), dtype=np.uint8,
        )
        _numba_nrrd_area_crop_resize_kernel(
            integral, crop_u8,
            int(in_h), int(in_w), int(out_h), int(out_w),
            int(y0), int(x0), int(y1), int(x1),
            int(out_y0), int(out_x0), restored,
        )
    if not np.any(restored):
        return None
    return (
        int(out_y0), int(out_x0), int(out_y1), int(out_x1),
        np.ascontiguousarray(restored, dtype=np.uint8),
    )

@runtime_telemetry_phase('restore.read_slice')
def _read_layer_slice_in_output_shape(
    src: object,
    output_shape: Tuple[int, int, int],
    out_z: int,
) -> np.ndarray:
    """Return one output-geometry ``(Y,X)`` slice for an NRRD layer without materializing the full layer."""
    in_t, in_h, in_w = _volume_shape_tuple(src)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    out_z_i = int(np.clip(int(out_z), 0, out_t - 1))

    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return _read_binary_volume_slice_u8(src, out_z_i)

    source_indices = _restore_source_indices_for_output_z(in_t, out_t, out_z_i)
    if in_h == out_h and in_w == out_w:
        if len(source_indices) == 1:
            return _read_binary_volume_slice_u8(src, int(source_indices[0]))
        restored = np.zeros((out_h, out_w), dtype=np.uint8)
        for src_idx in source_indices:
            restored |= _read_binary_volume_slice_bool(src, int(src_idx)).astype(np.uint8, copy=False)
        return restored

    restored = np.zeros((out_h, out_w), dtype=np.uint8)
    for src_idx in source_indices:
        restored |= _resize_binary_mask_frame_to_output_shape(_read_binary_volume_slice_u8(src, int(src_idx)), out_h, out_w)
    return restored

def _nrrd_layer_zero_skip_window(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int]]:
    """Output-space t-window ``[lo, hi)`` that MAY contain foreground, from the layer extent.

 Returns None when no extent is recorded (nothing can be skipped) and ``(0, 0)`` when
 the recorded extent is the scanned-empty sentinel — every extent source
 (raw_bbox_cvol_index / raw_layer_materialization_scan / live_volume_sink_scan) records
 genuinely scanned data, so empty means the whole payload is zeros. Note the contrast
 with _slicer_segment_extent_for_output, which maps empty to FULL because Slicer needs
 a valid display extent; for zero-skipping the empty sentinel is trustworthy as-is.

 The t-axis window inverts the ACTUAL restore mappings rather than reusing the header's
 outward display scaling: for in_t >= out_t each output z reads sources
 [floor(z*in/out), ceil((z+1)*in/out)) — the floor/ceil bounds below are exact for that;
 for in_t < out_t each output z reads one center-aligned nearest source, inverted
 exactly by _nrrd_output_z_extent_for_source_extent. One slice of padding per side
 absorbs any residual rounding-rule differences; skipping must never reclassify a
 foreground slice as zero."""
    extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    if extent is None:
        return None
    out_t = int(output_shape[0])
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return (0, 0)
    in_t = int(getattr(ref, 'segment_extent_shape_tyx', (0, 0, 0))[0])
    if in_t <= 0 or out_t <= 0:
        return None
    if in_t >= out_t:
        # Downscale/identity restore: output z draws sources [floor(z*in/out), ceil((z+1)*in/out)).
        lo_f = int(math.floor(float(t0) * float(out_t) / float(in_t)))
        hi_f = int(math.ceil(float(t1 + 1) * float(out_t) / float(in_t))) - 1
    else:
        # Upscale restore: output z draws one center-aligned nearest source.
        exact = _nrrd_output_z_extent_for_source_extent(int(in_t), int(out_t), int(t0), int(t1))
        if exact is None:
            return None
        lo_f, hi_f = (int(exact[0]), int(exact[1]))
    lo = max(0, int(lo_f) - 1)
    hi = min(int(out_t), int(hi_f) + 2)
    if hi <= lo:
        return (0, 0)
    return (int(lo), int(hi))

def _write_one_decomposed_nrrd_layer_payload(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
    payload_writer: object,
    *,
    z_chunk: int,
    pbar: Optional[object] = None,
    layer_idx: int = 0,
    block_consumer: Optional[Callable[[int, np.ndarray], None]] = None,
    sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    z_start: int = 0,
    z_stop: Optional[int] = None,
    fill_workers_override: Optional[int] = None,
) -> None:
    """Stream one restored NRRD layer in bounded output-z blocks.
    
    Optional consumers observe each block, and known-zero ranges bypass source reads."""
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    z_begin = int(np.clip(int(z_start), 0, out_t))
    z_end = out_t if z_stop is None else int(np.clip(int(z_stop), z_begin, out_t))
    if z_end <= z_begin:
        return
    z_chunk_i = max(1, int(z_chunk))
    madvise_interval = nrrd_madvise_dontneed_interval()
    src: Optional[object] = None
    try:
        src = _open_nrrd_layer_ref(ref)
        _madvise_array_mmap(src, 'MADV_SEQUENTIAL')
        in_t, in_h, in_w = _volume_shape_tuple(src)
        direct_native_stream = (
            (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
            and not isinstance(src, RawBBoxMaskStore)
        )
        raw_store_stream = isinstance(src, RawBBoxMaskStore)
        raw_store_native_stream = bool(
            raw_store_stream
            and (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
        )
        if (raw_store_native_stream and block_consumer is None
                and callable(getattr(payload_writer, 'write_canonical_zeros', None))
                and int(getattr(payload_writer, 'minimum_input_bytes', 0)) == 1):
            encoded = stream_native_crop_spans(
                src, payload_writer, z_begin, z_end, nrrd_compression.nrrd_gzip_chunk_bytes(), sparse_consumer)
            if pbar is not None:
                pbar.update(z_end-z_begin)
            runtime_telemetry().add('nrrd.crop_row_codec_input_bytes', encoded)
            print(f'NRRD crop-row stream {Path(ref.path).name}: '
                  f'codec_input={encoded / GIB:.3f} GiB of '
                  f'{(z_end-z_begin)*out_h*out_w / GIB:.3f} GiB; '
                  'empty row spans use cached gzip members.', flush=True)
            return
        # z-blocks fully outside this window are emitted as cached zero
        # members/chunks through write_zeros — no fill, no tee, no source page reads.
        zero_window = _nrrd_layer_zero_skip_window(ref, (out_t, out_h, out_w))
        can_write_zeros = callable(getattr(payload_writer, 'write_zeros', None))

        def _block_is_known_zero(z0: int, z1: int) -> bool:
            return (
                zero_window is not None
                and can_write_zeros
                and (int(z1) <= int(zero_window[0]) or int(z0) >= int(zero_window[1]))
            )

        if bool(raw_store_stream):
            # native AND restored cvol sources share one sparse path.
            # Members end only at output-slice boundaries, large assembly is vectorized
            # in native libraries, and an owned member can transfer directly to the async
            # compressor. The sparse mirror consumer observes crops during this same pass.
            member_bytes = int(np.clip(
                int(nrrd_compression.nrrd_gzip_chunk_bytes()),
                8 * 1024 * 1024,
                16 * 1024 * 1024,
            ))
            write_nonzero = getattr(payload_writer, 'write_owned_known_nonzero', None)
            if not callable(write_nonzero):
                write_nonzero = getattr(payload_writer, 'write_known_nonzero', None)
            if not callable(write_nonzero):
                write_nonzero = payload_writer.write
            write_aligned_zeros = getattr(payload_writer, 'write_aligned_zeros', None)
            if bool(raw_store_native_stream):
                members = src.iter_native_sparse_members(
                    int(z_begin), int(z_end), member_bytes=int(member_bytes),
                    sparse_consumer=sparse_consumer,
                )
            else:
                members = src.iter_restored_sparse_members(
                    (int(out_t), int(out_h), int(out_w)),
                    int(z_begin), int(z_end), member_bytes=int(member_bytes),
                    sparse_consumer=sparse_consumer,
                )
            member_z = int(z_begin)
            for raw_len, member in members:
                member_slices = int(raw_len) // max(1, int(out_h) * int(out_w))
                if member is None and can_write_zeros:
                    if callable(write_aligned_zeros):
                        write_aligned_zeros(int(raw_len))
                    else:
                        payload_writer.write_zeros(int(raw_len))
                elif member is None:
                    payload_writer.write(bytes(int(raw_len)))
                else:
                    write_nonzero(member)
                    # Preserve the generic consumer contract without the old second store
                    # pass: the aligned member is already an output-geometry dense block.
                    if block_consumer is not None:
                        block_consumer(
                            int(member_z),
                            np.asarray(member, dtype=np.uint8).reshape(
                                (int(member_slices), int(out_h), int(out_w)),
                            ),
                        )
                member_z += int(member_slices)
            if pbar is not None:
                pbar.update(int(z_end - z_begin))
            return

        if bool(direct_native_stream):
            # The memmap/live volume is already a native (t,Y,X) C-order layer: chunks are
            # views written directly — no transpose, Fortran conversion, or layer interleave.
            # Compression runs on a one-thread writer pool while this thread feeds the mirror
            # tee from the same read-only block, overlapping tee and deflate.
            slice_bytes = int(out_h) * int(out_w)
            writer_pool = _acquire_parallel_pool(1)
            pending: Optional[Future] = None
            try:
                for z0 in range(z_begin, z_end, int(z_chunk_i)):
                    z1 = min(z_end, int(z0) + int(z_chunk_i))
                    if _block_is_known_zero(int(z0), int(z1)):
                        if pending is not None:
                            pending.result()
                        pending = writer_pool.submit(payload_writer.write_zeros, int(z1 - z0) * slice_bytes)
                        if pbar is not None:
                            pbar.update(int(z1 - z0))
                        continue
                    chunk = np.asarray(src[int(z0):int(z1)], dtype=np.uint8)
                    if not chunk.flags['C_CONTIGUOUS']:
                        chunk = np.ascontiguousarray(chunk)
                    if pending is not None:
                        pending.result()
                    pending = writer_pool.submit(payload_writer.write, memoryview(chunk).cast('B'))
                    if block_consumer is not None:
                        block_consumer(int(z0), chunk)
                    if pbar is not None:
                        pbar.update(int(z1 - z0))
                    if madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                        # The writer must be done with these pages before dropping them.
                        pending.result()
                        pending = None
                        _madvise_array_mmap(src, 'MADV_DONTNEED')
                if pending is not None:
                    pending.result()
                    pending = None
            finally:
                if pending is not None:
                    try:
                        pending.result()
                    except Exception:
                        pass
                _release_parallel_pool(1, writer_pool)
            return

        # shared bounded-block streamer — each block is filled by a small
        # GIL-releasing thread pool (store decode memcpys / cv2 restores) and handed to a
        # one-thread writer with double buffering, so decode/restore overlaps the
        # selected validated gzip backend instead of filling the whole payload first.
        fill_workers = (
            int(nrrd_fill_workers())
            if fill_workers_override is None
            else max(1, int(fill_workers_override))
        )

        def _stream_filled_blocks(fill_one: Callable[[int, np.ndarray], None], *, madvise_src: bool) -> None:
            local_z_chunk = int(z_chunk_i)
            buffers: List[np.ndarray] = []
            for _ in range(2):
                try:
                    buffers.append(np.empty((int(local_z_chunk), int(out_h), int(out_w)), dtype=np.uint8, order='C'))
                except MemoryError:
                    break
            if not buffers:
                if int(local_z_chunk) != 1:
                    print(
                        f'Warning: NRRD payload block allocation failed for layer {int(layer_idx)}; '
                        'falling back to one output t-slice at a time.'
                    )
                local_z_chunk = 1
                buffers = [np.empty((1, int(out_h), int(out_w)), dtype=np.uint8, order='C')]

            # checkout-cached single-thread writer pool (was one build per layer).
            writer_pool = _acquire_parallel_pool(1)
            pending: Optional[Future] = None
            # buffers only cycle for blocks that are actually filled;
            # known-zero blocks go through write_zeros with no buffer at all. The counter
            # (not the z index) keeps the double-buffer alternation valid across skips:
            # at most one write is outstanding, so the buffer being refilled always
            # finished its own previous write before the intervening submit happened.
            buf_cycle = 0
            try:
                for z0 in range(z_begin, z_end, int(local_z_chunk)):
                    z1 = min(z_end, int(z0) + int(local_z_chunk))
                    z_count = int(z1 - z0)
                    if _block_is_known_zero(int(z0), int(z1)):
                        if pending is not None:
                            pending.result()
                        pending = writer_pool.submit(
                            payload_writer.write_zeros, int(z_count) * int(out_h) * int(out_w),
                        )
                        if pbar is not None:
                            pbar.update(int(z_count))
                        continue
                    if len(buffers) < 2 and pending is not None:
                        # Single-buffer fallback: the previous write must finish before refill.
                        pending.result()
                        pending = None
                    block = buffers[int(buf_cycle) % len(buffers)][:z_count, :, :]
                    buf_cycle += 1

                    if int(fill_workers) > 1 and int(z_count) > 1:
                        def _fill(zi: int, _z0: int = int(z0), _block: np.ndarray = block) -> None:
                            fill_one(int(_z0 + int(zi)), _block[int(zi)])
                        _nrrd_parallel_fill_indices(
                            int(z_count),
                            _fill,
                            requested_workers=min(int(fill_workers), int(z_count)),
                        )
                    else:
                        for zi in range(int(z_count)):
                            fill_one(int(z0 + zi), block[int(zi)])

                    if pending is not None:
                        pending.result()
                    pending = writer_pool.submit(payload_writer.write, memoryview(block).cast('B'))
                    # the mirror tee reads the block on this thread while the
                    # writer thread compresses it — both are read-only; the next refill of this
                    # buffer is still gated on pending.result two iterations out.
                    if block_consumer is not None:
                        block_consumer(int(z0), block)
                    if pbar is not None:
                        pbar.update(int(z_count))
                    if bool(madvise_src) and madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                        _madvise_array_mmap(src, 'MADV_DONTNEED')
                if pending is not None:
                    pending.result()
                    pending = None
            finally:
                if pending is not None:
                    try:
                        pending.result()
                    except Exception:
                        pass
                _release_parallel_pool(1, writer_pool)

        def _fill_restored(z: int, out2d: np.ndarray) -> None:
            np.copyto(out2d, np.asarray(_read_layer_slice_in_output_shape(src, output_shape, int(z)), dtype=np.uint8))

        _stream_filled_blocks(_fill_restored, madvise_src=True)
    finally:
        if src is not None:
            _madvise_array_mmap(src, 'MADV_DONTNEED')
            _close_nrrd_layer_source(src)
            _drop_nrrd_raw_store_chunks_ram_cache(src)

@dataclass(frozen=True)
class LowQualityDownbinSpec:
    raw_value: str
    token: str
    scale: float
    output_shape_t_y_x: Tuple[int, int, int]
    warning: str = ''
    formats: Tuple[str, ...] = ()

def _nearest_multiple_of_four(value: float) -> int:
    return max(4, int(math.floor(float(value) / 4.0 + 0.5)) * 4)

def _round_low_quality_dimension(value: float) -> int:
    """Round one isotropically scaled dimension to the nearest positive multiple of four."""
    return _nearest_multiple_of_four(max(1.0, float(value)))

def _low_quality_token(raw: str, shape_t_y_x: Tuple[int, int, int]) -> str:
    safe_raw = str(raw).strip().replace('-', 'm').replace('.', 'p').replace(',', '_')
    t_dim, h_dim, w_dim = (int(shape_t_y_x[0]), int(shape_t_y_x[1]), int(shape_t_y_x[2]))
    return f'{safe_raw}_{int(w_dim)}x{int(h_dim)}x{int(t_dim)}'

def resolve_low_quality_downbin_specs(
    downbin_values: Sequence[str] | str | None,
    low_quality_requested: bool,
    source_shape_t_y_x: Tuple[int, int, int],
) -> Tuple[List[LowQualityDownbinSpec], List[str]]:
    """Resolve isotropic low-quality downbins in native input geometry."""
    if downbin_values is None and not bool(low_quality_requested):
        return [], []

    raw_tokens = _parse_token_list(downbin_values) if downbin_values is not None else []
    if not raw_tokens:
        raw_tokens = ['1024']

    in_t, in_h, in_w = (int(source_shape_t_y_x[0]), int(source_shape_t_y_x[1]), int(source_shape_t_y_x[2]))
    max_dim = max(1, int(in_t), int(in_h), int(in_w))
    specs: List[LowQualityDownbinSpec] = []
    warnings: List[str] = []
    seen_shapes: set[Tuple[int, int, int]] = set()

    for raw in raw_tokens:
        raw_s = str(raw).strip()
        if not raw_s:
            continue
        try:
            value = float(raw_s)
        except Exception as exc:
            raise ValueError(f'--save low_quality downbin is not numeric: {raw_s!r}') from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError('--save low_quality downbins must be positive finite numbers')

        warning = ''
        raw_lower = raw_s.lower()
        looks_float = ('.' in raw_s) or ('e' in raw_lower)
        if looks_float:
            if not (0.0 < float(value) < 1.0):
                raise ValueError(
                    '--save low_quality floating-point downbins must be less than 1.0; '
                    f'got {raw_s!r}'
                )
            scale = float(value)
        else:
            nearest_int = int(round(value))
            if abs(float(value) - float(nearest_int)) > 1e-6:
                raise ValueError('--save low_quality integer downbins must be whole numbers; use a fraction such as 0.5 for scale factors')
            if nearest_int <= 0:
                raise ValueError('--save low_quality integer targets must be positive')
            rounded_target = _nearest_multiple_of_four(float(nearest_int))
            if int(rounded_target) != int(nearest_int):
                warning = (
                    f'--save low_quality:{int(nearest_int)} is not a multiple of 4; '
                    f'rounded to {int(rounded_target)} for isotropic low-quality output.'
                )
                warnings.append(warning)
            scale = min(1.0, float(rounded_target) / float(max_dim))

        # Round down when the nearest block would exceed the source dimension.
        shape = tuple(
            min(dim, _round_low_quality_dimension(dim * scale), max(4, dim // 4 * 4))
            for dim in (in_t, in_h, in_w)
        )
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        specs.append(LowQualityDownbinSpec(
            raw_value=raw_s,
            token=_low_quality_token(raw_s, shape),
            scale=float(scale),
            output_shape_t_y_x=shape,
            warning=warning,
        ))

    return specs, warnings


def resolve_save_grids(
    request: SaveRequest, source_shape_t_y_x: Tuple[int, int, int],
) -> Tuple[List[LowQualityDownbinSpec], List[str]]:
    """Merge requested formats when different downbins resolve to the same raster."""
    grids: Dict[Tuple[int, int, int], LowQualityDownbinSpec] = {}
    warnings: List[str] = []
    for group in request.low_quality:
        (candidate,), notes = resolve_low_quality_downbin_specs(
            (group.downbin,), True, source_shape_t_y_x,
        )
        warnings.extend(notes)
        grid = grids.get(candidate.output_shape_t_y_x, candidate)
        formats = tuple(dict.fromkeys((*grid.formats, *group.formats)))
        grids[candidate.output_shape_t_y_x] = dataclasses_replace(grid, formats=formats)
    return list(grids.values()), warnings

def low_quality_gpu_downbin_enabled() -> bool:
    return _env_flag('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', True)

def low_quality_gpu_downbin_chunk_slices() -> int:
    return max(1, _env_int('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN_CHUNK', 32))

_LQ_GPU_DOWNBIN_ANNOUNCED = False

def _try_gpu_downbin_volume(src_vol: np.ndarray, out_mm: np.ndarray, mode: str) -> bool:
    """Fill ``out_mm`` on an exclusively leased GPU; False selects the CPU path."""
    if not low_quality_gpu_downbin_enabled():
        return False
    # PyTorch adaptive pooling does not reproduce OpenCV INTER_AREA's fractional-bin
    # support or its uint8 rounding.  Gray distribution assets are an external output
    # contract, so keep them on the reference CPU/OpenCV path until an exact CUDA kernel
    # is available.  Binary-mask max pooling retains its existing GPU acceleration.
    if str(mode) == 'gray':
        return False
    in_t, in_h, in_w = (int(src_vol.shape[0]), int(src_vol.shape[1]), int(src_vol.shape[2]))
    out_t, out_h, out_w = (int(out_mm.shape[0]), int(out_mm.shape[1]), int(out_mm.shape[2]))
    if out_h > in_h or out_w > in_w:
        return False
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        if not bool(torch.cuda.is_available()):
            return False
    except Exception:
        return False
    lease = _acquire_output_gpu_stage(torch, f'low-quality {mode} GPU downbin')
    if lease is None:
        _announce_main_gpu_stage_skip_once(
            'low-quality-downbin-inference-busy',
            'GPU low-quality downbin skipped while all eligible GPUs have active/queued '
            'inference or another output-stage lease; using the CPU resize path.',
        )
        return False
    try:
        return _try_gpu_downbin_volume_on_device(
            src_vol,
            out_mm,
            str(mode),
            torch=torch,
            F=F,
            device=lease.torch_device(torch),
        )
    finally:
        lease.release()

def _try_gpu_downbin_volume_on_device(
    src_vol: np.ndarray,
    out_mm: np.ndarray,
    mode: str,
    *,
    torch: object,
    F: object,
    device: object,
) -> bool:
    if str(mode) == 'gray':
        return False
    in_t, in_h, in_w = (int(src_vol.shape[0]), int(src_vol.shape[1]), int(src_vol.shape[2]))
    out_t, out_h, out_w = (int(out_mm.shape[0]), int(out_mm.shape[1]), int(out_mm.shape[2]))
    chunk = int(low_quality_gpu_downbin_chunk_slices())
    src_per_chunk = int(math.ceil(float(in_t) / float(max(1, out_t)) * float(chunk))) + 2
    fp_bytes = 2
    need = (
        src_per_chunk * in_h * in_w * (1 + fp_bytes)
        + (chunk + 2) * out_h * out_w * (fp_bytes + 1)
        + GIB
    )
    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return False
    if int(free_bytes) < int(need):
        return False

    blk = pooled = out_chunk = window = None
    rows: List[object] = []
    try:
        with torch.inference_mode():
            with torch.cuda.device(device):
                for o0 in range(0, out_t, chunk):
                    blk = pooled = out_chunk = window = None
                    rows = []
                    o1 = min(out_t, int(o0) + chunk)
                    if mode == 'mask':
                        starts: List[int] = []
                        stops: List[int] = []
                        for oz in range(o0, o1):
                            s0 = int(math.floor(float(oz) * float(in_t) / float(out_t)))
                            s1 = int(math.ceil(float(oz + 1) * float(in_t) / float(out_t)))
                            s0 = int(np.clip(s0, 0, in_t - 1))
                            s1 = int(np.clip(max(s0 + 1, s1), 1, in_t))
                            starts.append(s0)
                            stops.append(s1)
                        lo, hi = min(starts), max(stops)
                        blk = torch.from_numpy(np.ascontiguousarray(src_vol[lo:hi])).to(device)
                        pooled = F.adaptive_max_pool2d(
                            blk.to(torch.float16).unsqueeze(1), (int(out_h), int(out_w)),
                        ).squeeze(1)
                        for i in range(int(o1 - o0)):
                            window = pooled[int(starts[i] - lo):int(stops[i] - lo)]
                            rows.append(window.amax(dim=0) if int(window.shape[0]) > 1 else window[0])
                        out_chunk = (torch.stack(rows) > 0).to(torch.uint8)
                    else:
                        return False
                    out_mm[int(o0):int(o1)] = out_chunk.cpu().numpy()
        global _LQ_GPU_DOWNBIN_ANNOUNCED
        if not _LQ_GPU_DOWNBIN_ANNOUNCED:
            _LQ_GPU_DOWNBIN_ANNOUNCED = True
            print(
                f'GPU low-quality downbin active on {device} '
                '(YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN=0 disables).'
            )
        return True
    except Exception as exc:
        print(f'Warning: GPU low-quality downbin failed ({exc}); using the CPU resize path.')
        return False
    finally:
        rows.clear()
        blk = pooled = out_chunk = window = None
        _trim_main_process_cuda_device(
            torch,
            device,
            desc=f'low-quality {mode} GPU downbin cleanup',
        )

def resize_gray_volume_to_shape(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing gray volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    # downbin on an idle GPU when possible; the per-slice CPU path below is
    # the unchanged fallback (it rewrites every slice, so a failed GPU pass cannot leak).
    if _try_gpu_downbin_volume(volume_gray, out_mm, 'gray'):
        return out_mm

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    xy_interp = cv2.INTER_AREA if (out_h <= in_h and out_w <= in_w) else cv2.INTER_LINEAR

    def _render_target_slice(out_z: int) -> None:
        # Center-aligned T matches the cube and the low-quality mask's box center.
        src_z = _center_aligned_source_position(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))
        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, xy_interp)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            return
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, xy_interp)
        blended = np.clip(
            np.rint((1.0 - alpha) * f0.astype(np.float32, copy=False) + alpha * f1.astype(np.float32, copy=False)),
            0.0,
            255.0,
        ).astype(np.uint8)
        out_mm[int(out_z), :, :] = blended

    parallel_for_indices_chunked(
        int(out_t),
        _render_target_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    return out_mm

def resize_binary_mask_volume_to_shape(
    mask_u8: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing binary mask volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(mask_u8.shape[0]), int(mask_u8.shape[1]), int(mask_u8.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return mask_u8

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    # downbin on an idle GPU when possible; CPU fallback below is unchanged.
    if _try_gpu_downbin_volume(mask_u8, out_mm, 'mask'):
        return out_mm

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _resize_mask_xy(frame: np.ndarray) -> np.ndarray:
        return resize_binary_plane_to_output_xy(frame, int(out_h), int(out_w))

    def _restore_slice(out_z: int) -> None:
        src_start = int(math.floor(float(out_z) * float(in_t) / float(out_t)))
        src_stop = int(math.ceil(float(out_z + 1) * float(in_t) / float(out_t)))
        src_start = int(np.clip(src_start, 0, in_t - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t))
        restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
        for src_idx in range(src_start, src_stop):
            restored |= _resize_mask_xy(mask_u8[int(src_idx)])
        out_mm[int(out_z), :, :] = restored

    parallel_for_indices_chunked(
        int(out_t),
        _restore_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    return out_mm



def _mask_grid_paths(out_dir: Path, stem: str, formats: Sequence[str]) -> Dict[str, Path]:
    files = {
        'images': out_dir / 'images' / f'{stem}_%04d.png',
        'video': out_dir / f'{stem}_Overlay.mkv',
        'binary_video': out_dir / f'{stem}_Binary.mkv',
        'labels': out_dir / DEFAULT_LABEL_PATTERN.replace('{Filename}', stem),
        'binary_images': out_dir / DEFAULT_BINARY_PATTERN.replace('{Filename}', stem),
        'semantic': out_dir / DEFAULT_SEMANTIC_PATTERN.replace('{Filename}', stem),
    }
    return {name: files[name] for name in formats if name in files}


def _mask_grid_receipt(files: Dict[str, Path], prefix: str = '') -> Dict[str, Path]:
    keys = {
        'video': 'overlay', 'binary_video': 'binary_video', 'binary_images': 'binary_tiff_dir',
        'labels': 'labels_dir', 'semantic': 'semantic_png_dir', 'images': 'images_dir',
    }
    return {
        prefix + keys[name]: path if name in ('video', 'binary_video') else path.parent
        for name, path in files.items()
    }


def _submit_mask_grid_writers(
    executor: ThreadPoolExecutor, files: Dict[str, Path], volume_gray: Optional[np.ndarray],
    mask: Optional[np.ndarray], fps: float, workers: int, show_progress: bool, scratch: Path,
    publication_root: Path,
) -> List[Future]:
    jobs = []
    sequences = {
        'binary_images': write_binary_tiff_sequence_from_pattern,
        'labels': write_yolo_labels_from_pattern,
        'semantic': write_semantic_png_sequence_from_pattern,
    }
    for name, path in files.items():
        if name == 'images':
            from .source_images import write_source_images_from_pattern
            jobs.append(executor.submit(write_source_images_from_pattern, volume_gray, path, workers, show_progress))
        elif name in sequences:
            jobs.append(executor.submit(sequences[name], mask, path, workers, show_progress))
        else:
            kwargs = dict(show_progress=show_progress, scratch_dir=scratch, publication_root=publication_root)
            if name == 'video':
                jobs.append(executor.submit(write_overlay_video, volume_gray, mask, path, fps, **kwargs))
            else:
                jobs.append(executor.submit(write_binary_video_from_mask_volume, mask, path, fps, **kwargs))
    return jobs


def write_grid_voxel_count(mask: np.ndarray, destination: Path, *, grid: str, workers: int) -> int:
    from .unification.tta_manifest import TTA_VOXEL_COUNT_SCHEMA

    counts = np.zeros(mask.shape[0], dtype=np.int64)

    def count_slice(z):
        counts[z] = np.count_nonzero(mask[z])

    parallel_for_indices(len(counts), count_slice,
                         max_workers=choose_slice_parallel_workers(workers, len(counts)),
                         desc='Counting voxel_volume')
    count = int(np.sum(counts, dtype=np.int64))
    write_json_atomic(destination, {
        'schema': TTA_VOXEL_COUNT_SCHEMA,
        'voxel_count': count,
        'source_stage': 'final_output_after_all_postprocessing',
        'target': ('native_input_space_final_binary_prediction' if grid == 'native'
                   else 'downbinned_final_binary_prediction'),
        'grid': grid, 'units': 'foreground_voxels', 'shape_t_y_x': list(map(int, mask.shape)),
    })
    return count


def _low_quality_grid_outputs(out_dir, stem, spec, *, family_video, flat_output):
    grid_dir = out_dir / 'low_quality' / spec.token
    formats = tuple(f for f in spec.formats if f != 'video' or not family_video)
    files = _mask_grid_paths(grid_dir, stem, formats)
    if flat_output:
        for name in ('video', 'binary_video'):
            if name in files:
                files[name] = out_dir / f'{files[name].stem}_LowQuality_{spec.token}.mkv'
    prefix = f'low_quality_{spec.token}_'
    paths = _mask_grid_receipt(files, prefix)
    if 'voxel_volume' in formats:
        paths[prefix + 'voxel_volume'] = grid_dir / f'{stem}_VoxelVolume.json'
    return files, paths


def save_single_low_quality_output(
    *, volume_gray: np.ndarray, mask_u8: np.ndarray, spec: LowQualityDownbinSpec,
    out_dir: Path, stem: str, fps: float, temp_dir: Path, workers: int = 1,
    show_progress: bool = True,
    stage_mask: Optional[Callable[[LowQualityDownbinSpec, np.ndarray], None]] = None,
    flat_output: bool = False,
) -> Dict[str, Path]:
    """Resize once per requested grid and dispatch only its selected mask products."""
    family_video = stage_mask is not None and 'video' in spec.formats
    files, paths = _low_quality_grid_outputs(out_dir, stem, spec, family_video=family_video,
                                           flat_output=flat_output)
    if not paths and not family_video:
        return {}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    shape = spec.output_shape_t_y_x
    scratch = Path(temp_dir) / 'low_quality' / spec.token
    gray = None
    mask = None
    try:
        if family_video or set(files) - {'images'} or 'voxel_volume' in spec.formats:
            mask = resize_binary_mask_volume_to_shape(mask_u8, shape, scratch / 'mask.u8.dat',
                                                      workers=workers, desc=f'Low-quality mask {spec.token}')
        if family_video:
            stage_mask(spec, mask)
        if 'video' in files or 'images' in files:
            gray = resize_gray_volume_to_shape(volume_gray, shape, scratch / 'source.gray8.dat',
                                                workers=workers, desc=f'Low-quality source {spec.token}')
        if files:
            count = max(1, min(len(files), _env_int('YOLO_TTA_LOW_QUALITY_OUTPUT_WORKERS_PER_BIN', min(workers, 3))))
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix='grid-output') as pool:
                futures = _submit_mask_grid_writers(
                    pool, files, gray, mask, fps * shape[0] / mask_u8.shape[0], workers,
                    show_progress, scratch, out_dir,
                )
                for future in as_completed(futures):
                    future.result()
        if 'voxel_volume' in spec.formats:
            write_grid_voxel_count(mask, paths[f'low_quality_{spec.token}_voxel_volume'],
                                   grid=f'low_quality/{spec.token}', workers=workers)
        return paths
    finally:
        if gray is not None and gray is not volume_gray:
            close_memmap_array(gray)
        if mask is not None and mask is not mask_u8:
            close_memmap_array(mask)


def collect_low_quality_output_futures(
    executor: ThreadPoolExecutor, *, volume_gray: np.ndarray, mask_u8: np.ndarray,
    out_dir: Path, stem: str, fps: float, downbin_specs: Sequence[LowQualityDownbinSpec],
    temp_dir: Path, workers: int = 1, show_progress: bool = True,
    stage_mask: Optional[Callable[[LowQualityDownbinSpec, np.ndarray], None]] = None,
    flat_output: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    paths = {}
    futures = []
    for spec in downbin_specs:
        family_video = stage_mask is not None and 'video' in spec.formats
        _, selected = _low_quality_grid_outputs(out_dir, stem, spec, family_video=family_video,
                                               flat_output=flat_output)
        if selected or family_video:
            futures.append(executor.submit(
                save_single_low_quality_output, volume_gray=volume_gray, mask_u8=mask_u8,
                spec=spec, out_dir=out_dir, stem=stem, fps=fps, temp_dir=temp_dir,
                workers=workers, show_progress=show_progress, stage_mask=stage_mask,
                flat_output=flat_output,
            ))
            paths.update(selected)
    return paths, futures

@dataclass
class BackgroundOutputSubmission:
    label: str
    result_paths: Dict[str, Path]
    futures: List[Future] = field(default_factory=list)
    resources: List[object] = field(default_factory=list)

    def wait(self) -> Dict[str, Path]:
        error: Optional[BaseException] = None
        try:
            for fut in self.futures:
                fut.result()
        except BaseException as exc:
            error = exc
            _settle_parallel_futures(self.futures)
        finally:
            for resource in self.resources:
                close_memmap_array(resource)
        if error is not None:
            raise RuntimeError(f'Background output generation failed for {self.label}') from error
        return self.result_paths

class BackgroundOutputManager:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='yolo-output')
        self.pending: List[BackgroundOutputSubmission] = []

    def submit(self, submission: BackgroundOutputSubmission) -> Dict[str, Path]:
        self.pending.append(submission)
        return submission.result_paths

    def reap_completed(self, *, max_completed: Optional[int] = None) -> None:
        if max_completed is not None and int(max_completed) < 1:
            raise ValueError('Output completion limit must be positive')
        remaining: List[BackgroundOutputSubmission] = []
        reaped = 0
        for submission in self.pending:
            if ((max_completed is None or reaped < int(max_completed))
                    and all(fut.done() for fut in submission.futures)):
                submission.wait()
                reaped += 1
            else:
                remaining.append(submission)
        self.pending = remaining

    def wait(self) -> None:
        error: Optional[BaseException] = None
        try:
            while self.pending:
                submission = self.pending.pop(0)
                try:
                    submission.wait()
                except BaseException as exc:  # pragma: no cover - surfaced to main
                    if error is None:
                        error = exc
        finally:
            self.executor.shutdown(wait=True)
        if error is not None:
            raise error

def collect_pipeline_output_futures(
    executor: ThreadPoolExecutor, *, volume_rgb: np.ndarray, mask_u8: np.ndarray,
    out_dir: Path, stem: str, fps: float, formats: Sequence[str], frame_workers: int,
    scratch_dir: Path, show_progress: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    if volume_rgb.shape != mask_u8.shape:
        raise ValueError('Source intensity and final label grids must have the same shape')
    files = _mask_grid_paths(out_dir, stem, formats)
    futures = _submit_mask_grid_writers(
        executor, files, volume_rgb, mask_u8, fps, frame_workers, show_progress, scratch_dir, out_dir,
    )
    return _mask_grid_receipt(files), futures

def _write_multichannel_tiff(path: Path, frame_hwc: np.ndarray, channel_count: int) -> None:
    """Write one uint8 grayscale TIFF page per model-input channel."""
    writer = getattr(cv2, 'imwritemulti', None)
    if not callable(writer):
        raise RuntimeError(
            'Saving channel-formatted images with five or more channels requires an '
            'OpenCV build that provides cv2.imwritemulti(); upgrade opencv-python.'
        )
    frame = np.ascontiguousarray(np.asarray(frame_hwc), dtype=np.uint8)
    expected_channels = int(channel_count)
    if frame.ndim != 3 or int(frame.shape[2]) != expected_channels:
        raise ValueError(
            f'Multi-page TIFF output expected HxWx{expected_channels}, got {tuple(frame.shape)}'
        )
    pages = [
        np.ascontiguousarray(frame[:, :, channel_index])
        for channel_index in range(expected_channels)
    ]
    if not bool(writer(str(path), pages)):
        raise RuntimeError(f'Failed to write multi-page TIFF: {path}')

def _canonical_image_token(
    value: object, fallback: str, *, max_length: int = 80,
) -> str:
    token = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_.-')
    token = token or str(fallback)
    limit = max(16, int(max_length))
    if len(token) > limit:
        digest = f'{zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF:08x}'
        token = f'{token[:limit - len(digest) - 1]}-{digest}'
    return token

@dataclass(frozen=True)
class CanonicalRenderImageSink:
    """Persist the exact CPU canonical arrays shared with the model loader."""

    stage_root: Path
    stem: str
    model_name: str
    view_name: str
    kind: str
    aug_id: str
    channel_count: int
    config_id: Optional[str] = None
    tile_id: Optional[str] = None
    backend: str = 'cpu'

    def _directory(self) -> Path:
        root = Path(self.stage_root)
        parts = [
            _canonical_image_token(self.backend, 'cpu'),
            _canonical_image_token(self.model_name, 'shared'),
            _canonical_image_token(self.view_name, 'view'),
            _canonical_image_token(self.kind, 'fullframe'),
            _canonical_image_token(self.aug_id, 'a0'),
        ]
        if str(self.kind) == 'tile':
            parts.extend((
                _canonical_image_token(self.config_id, 'tile_config'),
                _canonical_image_token(self.tile_id, 'tile'),
            ))
        for part in parts:
            root = root / part
        return root

    def _path_for_center(self, center_index: int) -> Path:
        suffix = '.tif' if int(self.channel_count) >= 5 else '.png'
        identity = [
            f'view-{_canonical_image_token(self.view_name, "view", max_length=40)}',
            f'aug-{_canonical_image_token(self.aug_id, "a0", max_length=24)}',
        ]
        if str(self.kind) == 'tile':
            identity.extend((
                f'config-{_canonical_image_token(self.config_id, "tile_config", max_length=32)}',
                f'tile-{_canonical_image_token(self.tile_id, "tile", max_length=48)}',
            ))
        identity.append(f'frame-{int(center_index) + 1:06d}')
        return self._directory() / ('__'.join(identity) + suffix)

    def __call__(self, batch: RenderBatch) -> None:
        for item in batch.items:
            # Cartesian tail repeats and Azimuthal seam-extension slots are model batch
            # bookkeeping, not distinct canonical center rasters.
            if bool(item.synthetic_padding) or bool(item.azimuthal_padding):
                continue
            if item.center_index is None:
                continue
            frame = np.asarray(item.frame)
            if frame.dtype != np.uint8:
                raise TypeError(
                    f'Canonical model-input image must be uint8, got {frame.dtype}'
                )
            if frame.ndim != 3 or int(frame.shape[2]) != int(self.channel_count):
                raise ValueError(
                    f'Canonical model-input image expected HxWx{int(self.channel_count)}, '
                    f'got {tuple(frame.shape)}'
                )
            out_path = self._path_for_center(int(item.center_index))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = out_path.with_name(
                f'.{out_path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{out_path.suffix}'
            )
            try:
                if int(self.channel_count) >= 5:
                    _write_multichannel_tiff(temp_path, frame, int(self.channel_count))
                else:
                    encoded = frame
                    if int(self.channel_count) == 1:
                        encoded = np.asarray(frame[:, :, 0])
                    if not bool(cv2.imwrite(str(temp_path), encoded)):
                        raise RuntimeError(f'Failed to write canonical model-input image: {out_path}')
                os.replace(temp_path, out_path)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

def canonical_render_image_sink_from_spec(
    spec: Dict[str, object],
    *,
    backend: Optional[str] = None,
) -> CanonicalRenderImageSink:
    """Build a spawn-safe sink from a worker task's plain-data description."""
    return CanonicalRenderImageSink(
        stage_root=Path(str(spec['stage_root'])),
        stem=str(spec['stem']),
        model_name=str(spec.get('model_name', 'shared')),
        view_name=str(spec['view_name']),
        kind=str(spec['kind']),
        aug_id=str(spec['aug_id']),
        channel_count=max(1, int(spec['channel_count'])),
        config_id=(str(spec['config_id']) if spec.get('config_id') is not None else None),
        tile_id=(str(spec['tile_id']) if spec.get('tile_id') is not None else None),
        backend=str(backend if backend is not None else spec.get('backend', 'cpu')),
    )

def publish_canonical_render_images(stage_root: Path, final_root: Path) -> Path:
    """Publish one completed same-filesystem canonical image tree."""
    stage = Path(stage_root)
    final = Path(final_root)
    if not stage.is_dir():
        raise FileNotFoundError(f'Canonical model-input image stage is missing: {stage}')
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if final.is_dir():
            shutil.rmtree(final)
        else:
            final.unlink()
    os.replace(stage, final)
    return final

def write_view_images(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_dir: Path,
    stem: str,
    *,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    """Save the same channel-formatted center inputs supplied to inference."""
    fmt = resolve_channel_format(channel_format)
    view_dir = out_dir / 'images' / view.name
    view_dir.mkdir(parents=True, exist_ok=True)
    total = int(view.num_slices)
    multipage_tiff = int(fmt.channel_count) >= 5
    suffix = '.tif' if multipage_tiff else '.png'
    renderer = ChannelFormattedFrameRenderer(
        lambda source_idx: get_view_frame_by_index(volume_rgb, view, int(source_idx)),
        view,
        fmt,
    )
    final_pattern = view_dir / f'{stem}_{view.name}_%04d{suffix}'
    with _staged_frame_sequence(final_pattern) as stage_pattern:
        def _write_frame(idx: int) -> None:
            frame = np.ascontiguousarray(renderer(int(idx)), dtype=np.uint8)
            out_path = _format_frame_path(stage_pattern, int(idx) + 1)
            if multipage_tiff:
                _write_multichannel_tiff(out_path, frame, int(fmt.channel_count))
                return
            if frame.ndim == 3 and int(frame.shape[2]) == 1:
                frame = np.ascontiguousarray(frame[:, :, 0])
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f'Failed to write image: {out_path}')

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f'Writing {view.name} {fmt.token} image sequence',
            show_progress=show_progress,
        )
        _publish_staged_frame_sequence(
            stage_pattern,
            final_pattern,
            total,
            stale_extensions=('.png', '.tif', '.tiff'),
        )
    return view_dir

def collect_sam_planning_omissions(interpolation_stats, *, known_omissions=()):
    from .sam_planning_outcomes import sam_planning_omission
    records = list(known_omissions)
    for stats in interpolation_stats:
        operation = ('extrapolation' if stats.get('processing_role') == 'extrapolation'
                     or stats.get('extrapolation_backend') == 'sam' else 'interpolation')
        scope = str(stats.get('scope_id') or '/'.join(str(stats[key]) for key in
                    ('model', 'view', 'source', 'tile_config_id') if stats.get(key)))
        omission = stats.get('sam_planning_omission') or sam_planning_omission(
            stats, operation=operation, scope_id=scope, view_name=str(stats.get('view', '')))
        if omission is not None:
            records.append(omission)
    unique = {}
    for record in records:
        key = (record['operation'], str(record['scope_id']).removesuffix('/extrapolation'),
               record['view'], record['pass_index'])
        unique[key] = record
    return list(unique.values())


def write_summary_file(
    out_path: Path,
    *,
    command: str,
    input_path: Path,
    out_dir: Path,
    scratch_dir: Path,
    source_shape_x_y_t: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    fps: float,
    model_paths: Sequence[str],
    view_names: Sequence[str],
    view_prediction_stats: Dict[str, int],
    interpolation_stats: List[Dict[str, object]],
    enable_3d_void_fill: bool,
    gaussian_smoothing_stats: Optional[Dict[str, int | float]],
    keep_objects_stats: Optional[Dict[str, int | float]],
    voxel_volume: Optional[int],
    final_paths: Dict[str, Path],
    augmentation_workers: int,
    slice_postprocess_workers: int,
    interpolation_workers: int,
    output_workers: int,
    view_prediction_labels: Optional[Dict[str, str]] = None,
    sam_planning_omissions: Optional[Sequence[Dict[str, object]]] = None,
    encoding: Optional[str] = None,
) -> Path:
    lines: List[str] = []
    omissions = (collect_sam_planning_omissions(interpolation_stats)
                 if sam_planning_omissions is None else sam_planning_omissions)
    if omissions:
        lines.append('Run status: INCOMPLETE; requested SAM work was omitted. Valid outputs are retained.')
        for item in omissions:
            lines.append(f"  SAM {item['operation']} {item['scope_id'] or item['view']} "
                         f"pass {item['pass_index']}: unresolved_plan={item['unresolved_plan']}, "
                         f"unresolved_groups={item['unresolved_group_count']}, "
                         f"reasons={','.join(item['reasons'])}")
        lines.append('')
    lines.append(f'Command: {command}')
    lines.append(f'Input: {input_path}')
    lines.append(f'Output directory: {out_dir}')
    lines.append(f'Source dimensions before cubic resizing (X, Y, t): {source_shape_x_y_t}')
    lines.append(f'Processing volume shape (t, Y, X): {volume_shape}')
    lines.append(f'FPS: {fps}')
    lines.append(f'Scratch directory: {scratch_dir}')
    lines.append('Workspace policy: in-memory first with disk fallback when the working set exceeds available RAM/swap')
    lines.append(f'3D void fill: {"enabled" if bool(enable_3d_void_fill) else "disabled"}; when enabled, it is applied once after the final global union')
    if bool(enable_3d_void_fill):
        lines.append('3D void fill background connectivity: 6-connected (topological dual of the 26-connected foreground)')
    lines.append(f'Augmentation workers: {int(augmentation_workers)}')
    lines.append(f'Slice-parallel postprocess workers: {int(slice_postprocess_workers)}')
    lines.append(f'Interpolation workers: {int(interpolation_workers)}')
    lines.append(f'Output workers: {int(output_workers)}')
    lines.append('Worker oversubscription: inference-phase pools are bounded against process-local GPU/OpenVINO reservations; tail-only CPU pools may expand after every inference worker exits.')
    if model_paths:
        lines.append('Models:')
        for model_path in model_paths:
            lines.append(f'  {str(model_path)}')
    else:
        lines.append('Models: <none>')
    lines.append(f'Views: {", ".join(view_names)}')
    lines.append('')
    lines.append('View statistics:')
    total_prediction_count = 0
    labels = dict(view_prediction_labels or {})
    ordered_keys: List[str] = [k for k in ('transverse', 'sagittal', 'coronal', 'azimuthal_transverse') if k in view_prediction_stats]
    tilted_keys = [k for k in view_prediction_stats.keys() if str(k).startswith('tilted_')]
    other_keys = [k for k in view_prediction_stats.keys() if k not in set(ordered_keys) and k not in set(tilted_keys)]
    for view_key in ordered_keys + sorted(tilted_keys, key=lambda k: labels.get(k, k)) + sorted(other_keys):
        label = labels.get(view_key, str(view_key).replace('_', ' ').title())
        count = int(view_prediction_stats.get(view_key, 0))
        total_prediction_count += count
        lines.append(f'  {label}: predictions={count}')
    lines.append(f'  Total prediction count: {int(total_prediction_count)}')

    extrapolation_stats = [item for item in interpolation_stats
                           if item.get('processing_role') == 'extrapolation']
    interpolation_stats = [item for item in interpolation_stats
                           if item.get('processing_role') != 'extrapolation']
    if extrapolation_stats:
        lines.append('SAM extrapolation: remaining post-interpolation terminals; raw-empty/horizon stop; contacts continue.')
        for item in extrapolation_stats:
            lines.append(f"  {item.get('model', '?')}/{item.get('view', '?')}/{item.get('source', '?')}: "
                         f"added_voxels={int(item.get('added_voxels', 0))}, "
                         f"evidence={item.get('sam_evidence_path', item.get('evidence_path', ''))}")
    if interpolation_stats:
        lines.append('')
        sam_stats = [item for item in interpolation_stats if item.get('interpolation_backend') == 'sam']
        if sam_stats:
            lines.append('Interpolation backend: sam; directions are increasing/decreasing view-native frame index.')
            lines.append('SAM bridges are policy-selected additions; final connection survival is reported separately in the NRRD manifest.')
        lines.append('Interpolation statistics (per pass):')
        pass_indices = sorted({int(s.get('pass_index', 0)) for s in interpolation_stats})
        for pass_idx in pass_indices:
            stats_this_pass = [s for s in interpolation_stats if int(s.get('pass_index', 0)) == pass_idx]
            total_objects = sum(int(s.get('num_objects', 0)) for s in stats_this_pass)
            total_endpoints = sum(int(s.get('num_endpoints', 0)) for s in stats_this_pass)
            total_candidates = sum(int(s.get('candidate_connections', 0)) for s in stats_this_pass)
            total_accepted = sum(int(s.get('accepted_connections', 0)) for s in stats_this_pass)
            total_default_bridges = sum(int(s.get('default_bridges', 0)) for s in stats_this_pass)
            total_walk_back = sum(int(s.get('walk_back_bridges', 0)) for s in stats_this_pass)
            total_skipped = sum(int(s.get('skipped_by_min_radius', 0)) for s in stats_this_pass)
            total_added_voxels = sum(int(s.get('added_voxels', 0)) for s in stats_this_pass)
            lines.append(
                f'  Pass {pass_idx}: objects={total_objects}, endpoints={total_endpoints}, '
                f'candidate_connections={total_candidates}, accepted_connections={total_accepted}, '
                f'default_bridges={total_default_bridges}, walk_back_bridges={total_walk_back}, '
                f'bridges_skipped_by_--interpolation_min_radius={total_skipped}, added_voxels={total_added_voxels}'
            )
            for s in sorted(stats_this_pass, key=lambda d: (str(d.get('model', '')), str(d.get('view', '')))):
                lines.append(
                    f"    {s.get('model', '?')}/{s.get('view', '?')}: "
                    f"objects={int(s.get('num_objects', 0))}, "
                    f"endpoints={int(s.get('num_endpoints', 0))}, "
                    f"candidate_connections={int(s.get('candidate_connections', 0))}, "
                    f"accepted_connections={int(s.get('accepted_connections', 0))}, "
                    f"default_bridges={int(s.get('default_bridges', 0))}, "
                    f"walk_back_bridges={int(s.get('walk_back_bridges', 0))}, "
                    f"bridges_skipped_by_--interpolation_min_radius={int(s.get('skipped_by_min_radius', 0))}, "
                    f"added_voxels={int(s.get('added_voxels', 0))}, "
                    f"skipped={bool(s.get('skipped', False))}"
                )
                if s.get('interpolation_backend') == 'sam':
                    family_receipts = list(s.get('sam_group_planning_receipts') or ())
                    refusal_counts: Dict[str, int] = {}
                    for family in family_receipts:
                        if family.get('status') != 'planned':
                            for reason in set(family.get('reasons') or ()):
                                refusal_counts[str(reason)] = refusal_counts.get(str(reason), 0) + 1
                    lines.append(
                        f"      SAM planning: status={s.get('sam_planning_status', 'unknown')}, "
                        f"families={len(family_receipts)}, "
                        f"planned={sum(family.get('status') == 'planned' for family in family_receipts)}, "
                        f"unresolved={int(s.get('sam_unresolved_groups', 0))}, "
                        f"planned_runs={int(s.get('planner_plan_count', 0))}"
                    )
                    if refusal_counts:
                        lines.append('        Family refusal reasons (can overlap): ' + ', '.join(
                            f'{reason}={count}' for reason, count in sorted(refusal_counts.items())))
                    if s.get('sam_planning_reasons'):
                        lines.append('        Plan reasons: ' + ', '.join(
                            str(reason) for reason in s['sam_planning_reasons']))
                    if s.get('sam_session_frame_range'):
                        lines.append(f"        Session frame range (endpoints included): {s['sam_session_frame_range']}")
                    if 'sam_group_peak_budget_bytes' in s and 'sam_contract_resident_budget_bytes' in s:
                        lines.append(
                            f"        Planning limits: family_MiB={int(s['sam_group_peak_budget_bytes']) / (1024**2):.1f}, "
                            f"resident_contract_MiB={int(s['sam_contract_resident_budget_bytes']) / (1024**2):.1f}")
                    resource_profile = s.get('sam_resource_profile') or {}
                    if resource_profile:
                        resource_fields = [f"status={resource_profile.get('status', 'unknown')}"]
                        for key, label in (('assigned_contract_bytes', 'assigned_family_MiB'),
                                           ('assigned_live_contract_bytes', 'assigned_resident_contract_MiB'),
                                           ('assigned_topology_bytes', 'assigned_topology_MiB'),
                                           ('assigned_session_cpu_bytes', 'assigned_session_CPU_MiB'),
                                           ('assigned_cpu_wave_bytes', 'assigned_CPU_wave_MiB')):
                            if key in resource_profile:
                                resource_fields.append(f'{label}={int(resource_profile[key]) / (1024**2):.1f}')
                        lines.append('        SAM resources: ' + ', '.join(resource_fields))
                    if 'sam_effective_in_flight' in s:
                        lines.append(
                            f"        SAM scheduling: max_in_flight={int(s['sam_effective_in_flight'])}, "
                            f"defer_refill_until_consumed={bool(s.get('sam_defer_refill_until_consumed', False))}")
                    lines.append(
                        f"      SAM generated_runs={int(s.get('sam_generated_runs', 0))}, "
                        f"selected_runs={int(s.get('sam_selected_runs', 0))}, "
                        f"rescued_groups={int(s.get('sam_guarded_rescued_groups', 0))}, "
                        f"rescued_runs={int(s.get('sam_guarded_rescued_runs', 0))}, "
                        f"rescue_enabled={bool(s.get('sam_guarded_rescue', {}).get('enabled', False))}, "
                        f"incomplete_runs={int(s.get('sam_incomplete_runs', 0))}, "
                        f"passes_requested={int(s.get('requested_passes', 0))}, "
                        f"passes_completed={int(s.get('completed_passes', 0))}, "
                        f"passes_skipped={int(s.get('skipped_passes', 0))}, "
                        f"policy_version={int(s.get('sam_policy_version', 0))}, "
                        f"policy={s.get('sam_policy_hash', '')}, evidence={s.get('sam_evidence_path', '')}"
                    )

    lines.append('')
    if gaussian_smoothing_stats is not None and int(gaussian_smoothing_stats.get('enabled', 0)) > 0:
        lines.append(
            'Gaussian smoothing: enabled; '
            f"sigma={float(gaussian_smoothing_stats.get('sigma', 0.0)):g}, "
            f"passes_requested={int(gaussian_smoothing_stats.get('passes_requested', 0))}, "
            f"passes_completed={int(gaussian_smoothing_stats.get('passes_completed', 0))}"
        )
        lines.append(
            '  voxel changes after thresholding: '
            f"added={int(gaussian_smoothing_stats.get('total_added_voxels', 0))}, "
            f"removed={int(gaussian_smoothing_stats.get('total_removed_voxels', 0))}"
        )
    else:
        lines.append('Gaussian smoothing: disabled')

    if keep_objects_stats is not None:
        lines.append('')
        lines.append(
            'keep_objects: '
            f"enabled, objects={int(keep_objects_stats.get('num_objects', 0))}, "
            f"kept={int(keep_objects_stats.get('kept_objects', 0))}, "
            f"removed_objects={int(keep_objects_stats.get('removed_objects', 0))}, "
            f"removed_voxels={int(keep_objects_stats.get('removed_voxels', 0))}, "
            f"topology_slabs={int(keep_objects_stats.get('topology_slab_count', 0))}, "
            f"slab_workers={int(keep_objects_stats.get('topology_slab_workers', 0))}"
        )
        lines.append(
            '  phase_seconds: '
            f"label={float(keep_objects_stats.get('label_seconds', 0.0)):.3f}, "
            f"pair_extraction={float(keep_objects_stats.get('pair_extraction_seconds', 0.0)):.3f}, "
            f"local_union={float(keep_objects_stats.get('local_union_seconds', 0.0)):.3f}, "
            f"boundary_merge={float(keep_objects_stats.get('boundary_merge_seconds', 0.0)):.3f}, "
            f"root_expansion={float(keep_objects_stats.get('root_expansion_seconds', 0.0)):.3f}, "
            f"area_reduction={float(keep_objects_stats.get('area_reduction_seconds', 0.0)):.3f}, "
            f"decision={float(keep_objects_stats.get('decision_seconds', 0.0)):.3f}, "
            f"lut={float(keep_objects_stats.get('lut_seconds', 0.0)):.3f}, "
            f"apply={float(keep_objects_stats.get('apply_seconds', 0.0)):.3f}, "
            f"total={float(keep_objects_stats.get('total_seconds', 0.0)):.3f}"
        )
        if int(keep_objects_stats.get('gpu_resident_tail', 0)) > 0:
            lines.append(
                '  gpu_resident_tail: '
                f"gpus={int(keep_objects_stats.get('gpu_count', 0))}, "
                f"peer_bytes={int(keep_objects_stats.get('peer_bytes', 0))}, "
                f"peer_host_bounces={int(keep_objects_stats.get('peer_host_bounces', 0))}"
            )
    else:
        lines.append('')
        lines.append('keep_objects: disabled')

    if voxel_volume is not None:
        lines.append('')
        lines.append(f'voxel_volume_native_input_space: {int(voxel_volume)}')

    lines.append('')
    lines.append('Final outputs:')
    for key in sorted(final_paths.keys()):
        lines.append(f'{key}: {final_paths[key]}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return out_path
