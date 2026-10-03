"""Bounded crops of the exact canonical OpenCV TTA intensity raster.

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
_PHASE_SELF_CHECK_LOCK = threading.Lock()
_PHASE_SELF_CHECKS = {}


def render_canonical_crop(source, view, index, *, affine, inverse, output_origin_yx,
                          output_height, output_width, view_frames=None,
                          native_origin_xy=(0, 0), output_canvas_width=None,
                          native_frame_cache=None):
    """Match a full canonical render slice without rendering that full canvas."""
    from . import geometry
    from .workspace import _env_int
    geometry.require_forward_sampling('cpu', geometry.DataRole.INTENSITY)
    cached = native_frame_cache is not None and 'plane' in native_frame_cache
    if not cached and view_frames is None:
        # Established samplers may need one native plane before resizing. Refuse
        # an oversized plane before its allocation, rather than building a full
        # view stack or growing an unbounded per-endpoint native cache.
        budget = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
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
    return _remap_global_affine_crop(native, matrix, output_origin_yx=output_origin_yx,
        output_height=output_height, output_width=output_width, native_origin_xy=native_origin_xy,
        output_canvas_width=output_canvas_width, sampling_stats=native_frame_cache)


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
           'canonical_sampling_backend', 'CANONICAL_PHASE_SELF_CHECK_CONTRACT',
           'ensure_canonical_phase_supported']
