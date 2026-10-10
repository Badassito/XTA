"""Bounded native shell crops and canonical OpenCV TTA image remapping.

Rebasing a float32 affine to a crop origin changes the inverse/interpolation
phase at awkward scales. Preserve GLOBAL output coordinates and OpenCV's
vector block phase, then remap only the requested rectangle plus bounded
alignment columns. Native TTA plane samplers remain owned by geometry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

CANONICAL_CROP_RENDER_CONTRACT = 'xta.sam_global_canonical_warp_crop/1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CANONICAL_PHASE_SELF_CHECK_CONTRACT = 'xta.sam_canonical_phase_self_check/1'
NATIVE_SHELL_CROP_CONTRACT = 'xta.sam_native_shell_demand_crop/1'
_PHASE_SELF_CHECK_LOCK = threading.Lock()
_PHASE_SELF_CHECKS = {}


def native_shell_workspace_bytes(view, bbox_yx=None):
    """Bound one native byte plane and the shell sampler's 32-row temporaries."""
    height, width = int(view.src_h), int(view.src_w)
    if str(view.family) not in {'radial', 'spherical'} or min(height, width) <= 0:
        raise ValueError('Native shell workspace requires positive Radial/Spherical geometry')
    if bbox_yx is not None:
        import operator
        try:
            bbox_yx = tuple(bbox_yx)
            y0, x0, y1, x1 = (operator.index(value) for value in bbox_yx)
        except (TypeError, ValueError) as error:
            raise ValueError('Native shell crop requires four integer bounds') from error
        if (any(isinstance(value, (bool, np.bool_)) for value in bbox_yx)
                or not 0 <= y0 < y1 <= height or not 0 <= x0 < x1 <= width):
            raise ValueError('Native shell crop is outside its physical view')
        height, width = y1-y0, x1-x0
    return height*width+1024*min(32, height)*width


def native_shell_crop_bbox(view, affine, output_bbox_yx):
    from ._deps import cv2
    from .sam_transverse_cache_rendering import native_crop_bbox
    matrix = np.asarray(affine, dtype=np.float32).reshape(2, 3).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('Native shell crop affine must be finite')
    return native_crop_bbox(output_bbox_yx, cv2.invertAffineTransform(matrix),
        (int(view.src_h), int(view.src_w)))


def _render_native_shell_crop(source, view, index, bbox):
    from .runtime import runtime_telemetry
    telemetry = runtime_telemetry()
    started, cpu_started = time.perf_counter(), time.thread_time()
    try:
        if bbox is None:
            plane = np.zeros((1, 1), np.uint8)
        else:
            if str(view.family) == 'radial':
                from .cylindrical_geometry import render_shell_frame
            else:
                from .spherical_geometry import render_shell_frame
            plane = render_shell_frame(source, view, int(index), bbox_yx=bbox)
        telemetry.add('sam.cpu_images.native_sampled_pixels', 0 if bbox is None else int(plane.size))
        telemetry.add('sam.cpu_images.native_full_frame_pixels', int(view.src_h)*int(view.src_w))
        return plane
    finally:
        telemetry.add('sam.cpu_images.native_sampling_host_seconds', time.perf_counter()-started)
        telemetry.add('sam.cpu_images.native_sampling_thread_cpu_seconds', time.thread_time()-cpu_started)


def prepare_native_shell_crop(source, view, index, *, affine, output_bbox_yx,
                              max_workspace_bytes):
    from . import geometry
    geometry.require_forward_sampling('cpu', geometry.DataRole.INTENSITY)
    if (str(view.family) not in {'radial', 'spherical'}
            or not isinstance(source, np.ndarray) or source.ndim != 3
            or tuple(source.shape) != (int(view.full_t), int(view.full_h), int(view.full_w))
            or not 0 <= int(index) < int(view.num_slices)):
        raise ValueError('Native shell crop requires its physical source and frame')
    bbox = native_shell_crop_bbox(view, affine, output_bbox_yx)
    needed = native_shell_workspace_bytes(view, bbox) if bbox is not None else 1
    if needed > int(max_workspace_bytes):
        raise RuntimeError('SAM native shell crop exceeds admitted rendering workspace')
    return _render_native_shell_crop(source, view, index, bbox), ((bbox[1], bbox[0]) if bbox is not None else (0, 0))


def iter_prefetched_native_planes(source, view, frames, *, max_workspace_bytes,
                                  min_remap_workspace_bytes, max_workers=8, check_cancel=None,
                                  native_crop_bounds=None):
    """Yield ordered native planes and their owner's remaining remap allowance.

    The caller owns one yielded plane; close this iterator before releasing its
    scratch credit. Workers only read a ready source and never own live profiles.
    """
    from . import geometry
    from .media import volume_readiness
    from .runtime import choose_slice_parallel_workers, parallel_map_in_order
    from .sam_gpu_rendering import clear_image_error_frames
    from .workspace import _cpu_count
    if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.dtype != np.uint8
            or tuple(source.shape) != (int(view.full_t), int(view.full_h), int(view.full_w))):
        raise ValueError('Native prefetch requires the ready materialized uint8 source geometry')
    readiness = volume_readiness(source)
    if readiness is not None and (not readiness._all_event.is_set() or readiness._exception is not None):
        raise ValueError('Native prefetch cannot wait for or materialize its source')
    frames = tuple(frames)
    if any(isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer))
           or not 0 <= int(frame) < int(view.num_slices) for frame in frames):
        raise ValueError('Native prefetch frame is outside its physical view')
    crops = (None if native_crop_bounds is None else
        {int(frame): None if native_crop_bounds[frame] is None else tuple(native_crop_bounds[frame])
         for frame in frames})
    if crops is None:
        native_bytes = int(view.src_h)*int(view.src_w)
        job_bytes = native_shell_workspace_bytes(view)
    else:
        job_bytes = max((native_shell_workspace_bytes(view, box) if box is not None else 1
            for box in crops.values()), default=1)
        native_bytes = max(((box[2]-box[0])*(box[3]-box[1]) if box is not None else 1
            for box in crops.values()), default=1)
    budget, minimum = int(max_workspace_bytes), int(min_remap_workspace_bytes)
    # One old yielded plane can survive normal next()/tuple assignment briefly.
    slots = (budget-minimum-native_bytes)//job_bytes
    if minimum <= native_bytes or slots < 1:
        raise ValueError('Native prefetch scratch cannot fit a job, transfer plane and remap lane')
    if not frames:
        return
    workers = choose_slice_parallel_workers(min(8, int(max_workers), max(1, int(_cpu_count())), slots), len(frames))
    remap_bytes = budget-workers*job_bytes-native_bytes
    from .runtime import runtime_telemetry
    runtime_telemetry().gauge('sam.cpu_images.last_native_prefetch', dict(
        workers=workers, frames=len(frames), native_plane_bytes=native_bytes,
        native_job_bound_bytes=job_bytes, remap_workspace_bytes=remap_bytes,
        admitted_workspace_bytes=budget, cropped=crops is not None))

    def render(frame):
        plane = None
        try:
            if check_cancel is not None:
                check_cancel()
            geometry.require_forward_sampling('cpu', geometry.DataRole.INTENSITY)
            if crops is None:
                plane = geometry.get_view_frame_by_index(source, view, int(frame))
                expected_shape = (int(view.src_h), int(view.src_w))
            else:
                bbox = crops[int(frame)]
                plane = _render_native_shell_crop(source, view, frame, bbox)
                expected_shape = (bbox[2]-bbox[0], bbox[3]-bbox[1]) if bbox is not None else (1, 1)
            if check_cancel is not None:
                check_cancel()
            if (not isinstance(plane, np.ndarray) or plane.dtype != np.uint8
                    or plane.shape != expected_shape or not plane.flags.c_contiguous):
                raise ValueError('Native prefetch sampler returned an incompatible plane')
            return plane
        except BaseException as error:
            plane = None
            clear_image_error_frames(error)
            raise

    pending = parallel_map_in_order(render, frames, max_workers=workers, max_pending=workers)
    plane = None
    try:
        for frame, plane in zip(frames, pending):
            if check_cancel is not None:
                check_cancel()
            yield int(frame), plane, remap_bytes
            plane = None
    except BaseException as error:
        plane = None
        clear_image_error_frames(error)
        raise
    finally:
        plane = None
        pending.close()


def cartesian_source_view(source, view):
    """Lend an exact materialized Cartesian axis recipe without copying pixels."""
    from .geometry import physical_view_name
    if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.dtype != np.uint8
            or str(view.family) != 'orthogonal'):
        return None
    axes = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
            'coronal': (2, 0, 1)}.get(physical_view_name(view))
    if axes is None:
        return None
    oriented = source.transpose(axes)
    if tuple(oriented.shape) != (int(view.num_slices), int(view.src_h), int(view.src_w)):
        return None
    from .media import volume_readiness
    readiness = volume_readiness(source)
    if readiness is not None and not readiness._all_event.is_set():
        return None
    return oriented


def prepare_cartesian_native_crop(source, view, index, *, affine, output_bbox_yx):
    """Gather only the native taps needed by the unchanged global affine remap."""
    from ._deps import cv2
    from .sam_transverse_cache_rendering import native_crop_bbox
    from .workspace import _env_int
    oriented = cartesian_source_view(source, view)
    if oriented is None:
        raise ValueError('SAM Cartesian crop requires a ready materialized uint8 axis recipe')
    if not 0 <= int(index) < int(oriented.shape[0]):
        raise IndexError('SAM Cartesian crop frame is outside its physical view')
    box = tuple(map(int, output_bbox_yx))
    if len(box) != 4 or min(box[:2]) < 0 or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('SAM Cartesian preparation crop must be positive and nonnegative')
    matrix = np.asarray(affine, dtype=np.float32).reshape(2, 3).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('SAM Cartesian crop affine must be finite')
    inverse = cv2.invertAffineTransform(matrix)
    roi = native_crop_bbox(box, inverse, oriented.shape[1:])
    if roi is None:
        # No source tap is reachable; a zero plane also preserves border output.
        return np.zeros((1, 1), np.uint8), (0, 0)
    y0, x0, y1, x1 = roi
    if (y1-y0)*(x1-x0) >= max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2)):
        raise RuntimeError('SAM Cartesian native crop exceeds the bounded rendering memory budget')
    return np.ascontiguousarray(oriented[int(index), y0:y1, x0:x1]), (x0, y0)


def render_canonical_crop(source, view, index, *, affine, inverse, output_origin_yx,
                          output_height, output_width, view_frames=None,
                          native_origin_xy=(0, 0), output_canvas_width=None,
                          native_frame_cache=None, max_workspace_bytes=None):
    """Apply the canonical global affine phase to the requested rectangle."""
    from . import geometry
    from .workspace import _env_int
    geometry.require_forward_sampling('cpu', geometry.DataRole.INTENSITY)
    cached = native_frame_cache is not None and 'plane' in native_frame_cache
    if not cached and view_frames is None:
        # Established samplers may need one native plane before resizing. Refuse
        # an oversized plane before its allocation, rather than building a full
        # view stack or growing an unbounded per-endpoint native cache.
        budget = max(1, int(max_workspace_bytes) if max_workspace_bytes is not None else
                     _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
        native_bytes = int(view.src_h)*int(view.src_w)
        if native_bytes+int(output_height)*int(output_width)+64*(int(output_width)+16) > budget:
            raise RuntimeError('SAM native plane and canonical crop exceed the bounded rendering memory budget')
    if cached:
        native = native_frame_cache['plane']
        native_origin_xy = native_frame_cache['origin_xy']
    elif geometry.is_tilted_view(view):
        # The established tilted intensity path first renders its exact native
        # integer grid, then warpAffine(... WARP_INVERSE_MAP). Preserve both.
        identity = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
        native = geometry.render_tilted_frame_on_grid(source, view, int(index), identity,
                                                       int(view.src_h), int(view.src_w))
    else:
        native = geometry.get_view_frame_by_index(source, view, int(index), view_frames=view_frames)
    if native_frame_cache is not None and not cached:
        native_frame_cache.update(plane=native, origin_xy=native_origin_xy)
    if geometry.is_tilted_view(view):
        matrix = np.asarray(inverse, dtype=np.float32).astype(np.float64)
    else:
        from ._deps import cv2
        # warpAffine converts its input float32 coefficients to double before
        # inversion. Inverting in float32 produces another noncanonical phase.
        matrix = cv2.invertAffineTransform(np.asarray(affine, dtype=np.float32).astype(np.float64))
    started = time.perf_counter()
    try:
        return _remap_global_affine_crop(native, matrix, output_origin_yx=output_origin_yx,
            output_height=output_height, output_width=output_width, native_origin_xy=native_origin_xy,
            output_canvas_width=output_canvas_width, sampling_stats=native_frame_cache,
            max_workspace_bytes=max_workspace_bytes)
    finally:
        if str(view.family) in {'radial', 'spherical'}:
            from .runtime import runtime_telemetry
            runtime_telemetry().add('sam.cpu_images.canonical_remap_host_seconds', time.perf_counter()-started)


def canonical_sampling_backend():
    """Bind the library and active numerical dispatch to image cache identity."""
    from ._deps import cv2
    return dict(opencv_version=str(cv2.__version__), optimized=bool(cv2.useOptimized()),
        cpu_features=str(cv2.getCPUFeaturesLine()),
        algorithm_hint=int(cv2.getDefaultAlgorithmHint()) if hasattr(cv2, 'getDefaultAlgorithmHint') else None,
        build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        ipp_enabled=bool(cv2.ipp.useIPP()) if hasattr(cv2, 'ipp') else None,
        ipp_not_exact=bool(cv2.ipp.useIPP_NotExact()) if hasattr(cv2, 'ipp') else None)


def ensure_canonical_phase_supported(backend=None):
    """Small synthetic check once per helper/library/active numerical backend.

    This never calibrates from real images, changes a rendering rule, or falls
    back to rendering a full real canonical frame. A mismatch stops admission.
    """
    from ._deps import cv2
    backend = dict(backend or canonical_sampling_backend())
    key = hashlib.sha256(json.dumps(dict(helper_sha256=IMPLEMENTATION_SHA256,
        backend=backend), sort_keys=True).encode()).hexdigest()
    with _PHASE_SELF_CHECK_LOCK:
        if key not in _PHASE_SELF_CHECKS:
            started = time.perf_counter()
            native = np.random.default_rng(337).integers(0, 256, size=(19, 31), dtype=np.uint8)
            failures, comparisons = [], 0
            # The largest synthetic canonical raster is 1535x73 (<113KiB),
            # independent of source/view size. Odd tails, awkward global x/y
            # offsets and both forward/inverse warp routes are exercised.
            for height, width in ((73, 1535), (29, 157)):
                forward = np.array([[width/31., 0., -.4375],
                                    [0., height/19., -.21875]], np.float32)
                inverted = cv2.invertAffineTransform(forward.astype(np.float64))
                boxes = ((5, width-171 if width>171 else 17, height-3, width),
                         (height//3+1, 19, height-1, min(width, 93)))
                for inverse_route in (False, True):
                    supplied = inverted.astype(np.float32) if inverse_route else forward
                    reference = cv2.warpAffine(native, supplied, (width, height),
                        flags=cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse_route else 0),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    matrix = supplied.astype(np.float64) if inverse_route else inverted
                    for y0, x0, y1, x1 in boxes:
                        actual = _remap_global_affine_crop(native, matrix, output_origin_yx=(y0, x0),
                            output_height=y1-y0, output_width=x1-x0, output_canvas_width=width,
                            max_workspace_bytes=2*1024**2)
                        mismatches = int(np.count_nonzero(actual != reference[y0:y1, x0:x1]))
                        comparisons += 1
                        if mismatches:
                            failures.append(dict(canvas_shape_yx=[height, width],
                                inverse_route=inverse_route, bbox_yx=[y0, x0, y1, x1],
                                mismatched_pixels=mismatches))
            _PHASE_SELF_CHECKS[key] = dict(contract=CANONICAL_PHASE_SELF_CHECK_CONTRACT,
                status='passed' if not failures else 'unsupported', helper_sha256=IMPLEMENTATION_SHA256,
                backend=backend, backend_key_sha256=key, synthetic_comparisons=comparisons,
                largest_synthetic_canvas_bytes=73*1535, failures=failures,
                wall_seconds=time.perf_counter()-started)
        # Operational receipts must not expose mutable cached guard state.
        receipt = json.loads(json.dumps(_PHASE_SELF_CHECKS[key]))
    if receipt['status'] != 'passed':
        raise RuntimeError('Unsupported OpenCV canonical SAM crop numerical phase; '
            f"OpenCV {backend['opencv_version']} failed the bounded synthetic self-check. "
            'Run the SAM provider CPU tests and use a compatible OpenCV build before retrying. '
            'SAM model admission was refused.')
    return receipt


def _vector_width_and_fma():
    from ._deps import cv2
    if not cv2.useOptimized():
        return 1, False
    features = str(cv2.getCPUFeaturesLine()).split()
    enabled = [feature for feature in features if not feature.endswith('?')]
    # OpenCV's warp universal intrinsics use 256-bit float lanes on x86,
    # including its AVX512-SKX dispatch (the kernel's unroll is twice 8 lanes).
    width = 16 if '*AVX2' in enabled else 8
    # OpenCV's CPU_FMA3 enumeration is 12. NEON/AArch64 uses fused multiply-add
    # in the same global vector coordinate expression.
    fused = bool(cv2.checkHardwareSupport(12)) or any('NEON' in feature for feature in enabled)
    return width, fused


def _remap_global_affine_crop(native, inverse_double, *, output_origin_yx,
                              output_height, output_width, native_origin_xy=(0, 0),
                              output_canvas_width=None, sampling_stats=None,
                              max_workspace_bytes=None):
    from ._deps import cv2
    from .workspace import _env_int
    y0, x0 = map(int, output_origin_yx)
    height, width = int(output_height), int(output_width)
    if height <= 0 or width <= 0 or min(x0, y0) < 0:
        raise ValueError('Canonical crop requires positive dimensions and nonnegative global origin')
    native = np.ascontiguousarray(native, dtype=np.uint8)
    if native.ndim != 2:
        raise ValueError('Canonical SAM crop requires one grayscale native plane')
    matrix = np.asarray(inverse_double, dtype=np.float64).reshape(2, 3)
    nx0, ny0 = map(int, native_origin_xy)
    if not np.isfinite(matrix).all():
        raise ValueError('Canonical sampling matrix must be finite')
    canvas_width = int(output_canvas_width or x0+width)
    if canvas_width < x0+width:
        raise ValueError('Canonical crop exceeds the full output canvas')
    vector_width, fused = _vector_width_and_fma()
    # Remap and warp use the same vector interpolation expression in OpenCV5.
    # Start on a global vector boundary and include its full right block so a
    # crop edge cannot turn canonical vector pixels into scalar-tail pixels.
    left = (x0//vector_width)*vector_width
    right = min(canvas_width, ((x0+width+vector_width-1)//vector_width)*vector_width)
    mapped_width = right-left
    budget = max(1, int(max_workspace_bytes) if max_workspace_bytes is not None else
                 _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
    output_bytes = height*width
    # Each block retains fixed maps and bounded int64 coordinate temporaries.
    row_bytes = max(1, mapped_width*64)
    retained_bytes = output_bytes+int(native.nbytes)
    if retained_bytes + row_bytes > budget:
        raise RuntimeError('SAM canonical crop exceeds the bounded rendering memory budget')
    block_rows = max(1, min(256, (budget-retained_bytes)//row_bytes))
    if sampling_stats is not None:
        sampling_stats['sampled_output_pixels'] = int(sampling_stats.get('sampled_output_pixels', 0))+height*mapped_width
    result = np.empty((height, width), np.uint8)
    if int(str(cv2.__version__).split('.')[0]) >= 5:
        matrix32 = matrix.astype(np.float32)
        xx = np.arange(left, right, dtype=np.float32)
        vector_pixels = xx < (canvas_width//vector_width)*vector_width if vector_width > 1 else np.zeros(xx.shape, bool)
        for row in range(0, height, block_rows):
            stop = min(height, row+block_rows)
            yy = np.arange(y0+row, y0+stop, dtype=np.float32)
            bx = matrix32[0, 1]*yy+matrix32[0, 2]
            by = matrix32[1, 1]*yy+matrix32[1, 2]
            if fused:
                map_x = (np.float64(matrix32[0, 0])*xx.astype(np.float64)[None, :]
                         +bx.astype(np.float64)[:, None]).astype(np.float32)
                map_y = (np.float64(matrix32[1, 0])*xx.astype(np.float64)[None, :]
                         +by.astype(np.float64)[:, None]).astype(np.float32)
            else:
                map_x = matrix32[0, 0]*xx[None, :]+bx[:, None]
                map_y = matrix32[1, 0]*xx[None, :]+by[:, None]
            if not vector_pixels.all():
                # The canonical scalar tail evaluates x-term, y-term, then
                # translation separately in float32 (no vector FMA).
                scalar_x = ((matrix32[0, 0]*xx[None, :]+matrix32[0, 1]*yy[:, None])+matrix32[0, 2])
                scalar_y = ((matrix32[1, 0]*xx[None, :]+matrix32[1, 1]*yy[:, None])+matrix32[1, 2])
                map_x[:, ~vector_pixels] = scalar_x[:, ~vector_pixels]
                map_y[:, ~vector_pixels] = scalar_y[:, ~vector_pixels]
            if nx0:
                map_x -= np.float32(nx0)
            if ny0:
                map_y -= np.float32(ny0)
            rendered = cv2.remap(native, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            result[row:stop] = rendered[:, x0-left:x0-left+width]
        return result
    # OpenCV's affine mapper uses AB_BITS=max(10,INTER_BITS), INTER_BITS=5.
    # It rounds the x delta and y base separately before summing and shifting.
    # np.rint agrees with cvRound's nearest-even host rounding.
    scale, shift, round_delta, interpolation_bits = 1024, 5, 16, 5
    int_min, int_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    xx = np.arange(left, right, dtype=np.float64)
    dx = np.clip(np.rint(matrix[0, 0]*xx*scale), int_min, int_max).astype(np.int64)
    dy = np.clip(np.rint(matrix[1, 0]*xx*scale), int_min, int_max).astype(np.int64)
    for row in range(0, height, block_rows):
        stop = min(height, row+block_rows)
        yy = np.arange(y0+row, y0+stop, dtype=np.float64)
        bx = np.clip(np.rint((matrix[0, 1]*yy+matrix[0, 2])*scale), int_min, int_max).astype(np.int64)+round_delta
        by = np.clip(np.rint((matrix[1, 1]*yy+matrix[1, 2])*scale), int_min, int_max).astype(np.int64)+round_delta
        fixed_x = (bx[:, None]+dx[None, :]) >> shift
        fixed_y = (by[:, None]+dy[None, :]) >> shift
        coordinates = np.empty((stop-row, mapped_width, 2), np.int16)
        # Native ROI origins are integer translations applied AFTER global
        # interpolation quantization; they cannot change the fractional phase.
        coordinates[:, :, 0] = np.clip((fixed_x >> interpolation_bits)-nx0, -32768, 32767)
        coordinates[:, :, 1] = np.clip((fixed_y >> interpolation_bits)-ny0, -32768, 32767)
        fractions = (((fixed_y & 31) << interpolation_bits)+(fixed_x & 31)).astype(np.uint16)
        rendered = cv2.remap(native, coordinates, fractions, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result[row:stop] = rendered[:, x0-left:x0-left+width]
    return result


__all__ = ['CANONICAL_CROP_RENDER_CONTRACT', 'IMPLEMENTATION_SHA256', 'render_canonical_crop',
           'NATIVE_SHELL_CROP_CONTRACT', 'native_shell_crop_bbox', 'prepare_native_shell_crop',
           'native_shell_workspace_bytes', 'iter_prefetched_native_planes',
           'cartesian_source_view', 'prepare_cartesian_native_crop',
           'canonical_sampling_backend', 'CANONICAL_PHASE_SELF_CHECK_CONTRACT',
           'ensure_canonical_phase_supported']
