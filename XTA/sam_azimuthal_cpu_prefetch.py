"""Ordered, scratch-bounded CPU preparation of Azimuthal SAM demand crops.

The serial CPU SAM image route renders each planned Azimuthal record with
``geometry.get_view_frame_by_index`` (one full native plane) and then applies
the canonical global affine crop to every still-missing rectangle. Records of
different frames are independent and their numpy/OpenCV kernels release the
GIL, so a bounded worker pool may prepare upcoming records while the owner
copies earlier crops into the private cache in plan order. Workers call the
exact same native sampler on the same ready ndarray and the unchanged
canonical remap helper with its default workspace budget; they never sample
another grid, crop the native plane or see a live resource profile.
"""
from __future__ import annotations

import numpy as np

AZIMUTHAL_CPU_PREFETCH_CONTRACT = 'xta.sam_azimuthal_cpu_crop_prefetch/1'


def azimuthal_native_workspace_bytes(view) -> int:
    """Conservative peak bytes of one ``get_view_frame_by_index`` Azimuthal call.

    Covers the returned uint8 plane, the per-block gather/float/einsum/rounding
    temporaries of the non-tilted extractor (including the stack-fold buffers)
    and the per-row-block temporaries of the tilted extractor.
    """
    from . import geometry
    from .workspace import _env_int
    if not geometry.is_azimuthal_view(view):
        raise ValueError('Azimuthal native workspace requires an Azimuthal view')
    rows, width = int(view.src_h), int(view.src_w)
    if rows <= 0 or width <= 0:
        raise ValueError('Azimuthal native workspace requires positive native geometry')
    plane = rows*width
    taps = int(geometry.AZIMUTHAL_FILTER_TAP_COUNT)**2
    # Sampler/tap tables: int32/float32 indices and weights plus flattened int64 taps.
    tables = width*taps*48+4096
    if geometry.is_tilted_azimuthal_view(view):
        block = min(rows, max(1, _env_int('YOLO_TTA_TILTED_AZIMUTHAL_ROW_BLOCK', 16)))
        # Per tap: centers/stack/valid/s0/s1/alpha/f0/f1/values (<= 12 float32/int32
        # rows) plus acc and the final rounding temporaries.
        return plane+tables+block*width*4*16
    stack = int(geometry.azimuthal_stack_length(view))
    block = min(stack, geometry.choose_azimuthal_exact_block_frames(width))
    # Gather result + layout copy (uint8), float32 samples, einsum output and
    # rint/clip float32 temporaries for each stack block.
    block_bytes = block*width*taps*(2+4)+block*width*4*3
    fold = 0
    if rows != stack:
        # Full-stack float32 projection and four rows x width float32 fold temporaries.
        fold = stack*width*4+rows*width*4*4+rows*16
    return plane+tables+block_bytes+fold


def canonical_remap_workspace_bytes(height, width, x0, canvas_width) -> int:
    """Peak bytes of the unchanged canonical remap of one rectangle at its default budget.

    Mirrors ``_remap_global_affine_crop``: the result plus one block of at most
    256 rows of 64-byte map/coordinate temporaries over the vector-aligned span.
    """
    from .sam_canvas_rendering import _vector_width_and_fma
    vector_width, _fused = _vector_width_and_fma()
    left = (int(x0)//vector_width)*vector_width
    right = min(int(canvas_width), ((int(x0)+int(width)+vector_width-1)//vector_width)*vector_width)
    return int(height)*int(width)+min(256, int(height))*max(1, right-left)*64+64*(int(width)+16)


def render_azimuthal_record(source, view, frame, rects, *, affine, inverse, canvas_width):
    """Render one record's still-missing rectangles exactly as the serial route does.

    The serial route samples the full native plane on its first missing
    rectangle and reuses it, at origin (0, 0), for the remaining rectangles.
    Returns ``(images, native_plane_pixels, canonical_sampled_pixels)``.
    """
    from . import geometry
    from .sam_canvas_rendering import render_canonical_crop
    geometry.require_forward_sampling('cpu', geometry.DataRole.INTENSITY)
    plane = geometry.get_view_frame_by_index(source, view, int(frame))
    if (not isinstance(plane, np.ndarray) or plane.dtype != np.uint8
            or plane.shape != (int(view.src_h), int(view.src_w))):
        raise ValueError('Azimuthal prefetch sampler returned an incompatible plane')
    cache = dict(plane=plane, origin_xy=(0, 0))
    images = []
    for cy0, cx0, cy1, cx1 in rects:
        image = render_canonical_crop(source, view, int(frame), affine=affine, inverse=inverse,
            output_origin_yx=(cy0, cx0), output_height=cy1-cy0, output_width=cx1-cx0,
            output_canvas_width=canvas_width, native_frame_cache=cache, max_workspace_bytes=None)
        if image.dtype != np.uint8 or image.shape != (cy1-cy0, cx1-cx0):
            raise ValueError('Azimuthal prefetch remap returned a mismatched detector crop')
        images.append(image)
    pixels = int(plane.size)
    del plane
    cache.pop('plane', None)
    return images, pixels, int(cache.get('sampled_output_pixels', 0))


def iter_prefetched_azimuthal_records(source, view, jobs, *, workers, affine, inverse,
                                      canvas_width, check_cancel=None):
    """Yield ``(frame, images, plane_pixels, sampled_pixels)`` in job order.

    ``jobs`` is a sequence of ``(native_frame, rects)``. At most ``workers`` jobs
    are in flight and the caller owns one yielded result; close this iterator
    before releasing the build's scratch credit.
    """
    from . import geometry
    from .media import volume_readiness
    from .runtime import parallel_map_in_order
    from .sam_gpu_rendering import clear_image_error_frames
    if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.dtype != np.uint8
            or tuple(source.shape) != (int(view.full_t), int(view.full_h), int(view.full_w))
            or not geometry.is_azimuthal_view(view)):
        raise ValueError('Azimuthal prefetch requires the ready materialized uint8 source geometry')
    readiness = volume_readiness(source)
    if readiness is not None and (not readiness._all_event.is_set() or readiness._exception is not None):
        raise ValueError('Azimuthal prefetch cannot wait for or materialize its source')
    jobs = tuple((frame, tuple(tuple(int(value) for value in rect) for rect in rects)) for frame, rects in jobs)
    if any(isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer))
           or not 0 <= int(frame) < int(view.num_slices) or not rects for frame, rects in jobs):
        raise ValueError('Azimuthal prefetch job is outside its physical view or has no rectangle')
    workers = max(1, int(workers))

    def render(position):
        result = None
        try:
            if check_cancel is not None:
                check_cancel()
            frame, rects = jobs[position]
            result = render_azimuthal_record(source, view, frame, rects, affine=affine,
                inverse=inverse, canvas_width=canvas_width)
            if check_cancel is not None:
                check_cancel()
            return result
        except BaseException as error:
            result = None
            clear_image_error_frames(error)
            raise

    if not jobs:
        return
    pending = parallel_map_in_order(render, range(len(jobs)), max_workers=workers, max_pending=workers)
    result = None
    try:
        for (frame, _rects), result in zip(jobs, pending):
            if check_cancel is not None:
                check_cancel()
            images, plane_pixels, sampled = result
            result = None
            yield int(frame), images, plane_pixels, sampled
            images = None
    except BaseException as error:
        result = images = None
        clear_image_error_frames(error)
        raise
    finally:
        result = images = None
        pending.close()


__all__ = ['AZIMUTHAL_CPU_PREFETCH_CONTRACT', 'azimuthal_native_workspace_bytes',
           'canonical_remap_workspace_bytes', 'render_azimuthal_record',
           'iter_prefetched_azimuthal_records']
