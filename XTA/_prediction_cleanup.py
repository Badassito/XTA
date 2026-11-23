"""CPU prediction cleanup: component thresholds, confidence retirement and final topology.

This owner is independent of model execution and GPU process state. Streaming
inference and completed-view assembly share the same component-filtering rules.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ._deps import cv2, ndi
from .geometry import ViewInfo, view_processing_min_radius
from .runtime import (
    choose_parallel_chunk_size,
    choose_slice_parallel_workers,
    parallel_for_indices_chunked,
)
from .workspace import yolo_hole_fill_enabled

CONF_U8_MAX = 255


def min_conf_to_u8_threshold(min_conf: float) -> int:
    conf_clamped = min(1.0, max(0.0, float(min_conf)))
    return int(math.ceil(conf_clamped * float(CONF_U8_MAX) - 1e-9))

# OpenCV is the default because the hot operations used here release the GIL and scale
# better under Python thread pools. 'scipy' keeps the older scipy.ndimage cleanup as a
# differential reference for tests and strict regression comparison.
DEFAULT_CLEANUP_BACKEND = 'opencv'


def _normalize_cleanup_backend(backend: str) -> str:
    """Return the canonical per-slice cleanup backend name or raise ValueError."""
    name = str(backend).strip().lower()
    if name not in ('opencv', 'scipy'):
        raise ValueError(f"unknown prediction cleanup backend {backend!r}; expected 'opencv' or 'scipy'")
    return name

def _cv2_connected_components(mask_u8: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray]:
    return cv2.connectedComponents(
        np.ascontiguousarray(mask_u8, dtype=np.uint8),
        connectivity=int(connectivity),
        ltype=cv2.CV_32S,
    )

_PLANE_CONNECTIVITY = np.zeros((3, 3, 3), dtype=bool)
_PLANE_CONNECTIVITY[1, 1, :] = True
_PLANE_CONNECTIVITY[1, :, 1] = True


def _device_enclosed_background(cp, cpx_ndi, background):
    """Find enclosed 4-connected background independently in each device plane."""
    labels, count = cpx_ndi.label(background, structure=_PLANE_CONNECTIVITY)
    exterior = cp.zeros((int(count) + 1,), dtype=cp.bool_)
    exterior[0] = True
    exterior[labels[:, 0, :]] = True
    exterior[labels[:, -1, :]] = True
    exterior[labels[:, :, 0]] = True
    exterior[labels[:, :, -1]] = True
    return ~exterior[labels]


def _fill_holes_2d_scipy(mask_bool: np.ndarray) -> np.ndarray:
    return np.asarray(ndi.binary_fill_holes(np.asarray(mask_bool, dtype=bool)), dtype=bool)

def _fill_holes_2d_opencv_u8_inplace(
    arr_u8: np.ndarray,
    known_bbox: Optional[Sequence[int]] = None,
) -> None:
    """Fill enclosed 4-connected background inside the foreground bbox plus halo.

    A one-pixel exterior background halo gives the cropped connected-component problem
    exactly the same outside/background semantics as the full plane.  When foreground
    touches a global image edge the crop also touches that edge, so boundary-connected
    background remains outside.  Empty and full-foreground slices return without labels.
    """
    arr = np.asarray(arr_u8, dtype=np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return
    full_h, full_w = (int(arr.shape[0]), int(arr.shape[1]))
    if known_bbox is not None:
        try:
            y0_known, y1_known, x0_known, x1_known = (int(v) for v in known_bbox)
            y0_known = max(0, min(full_h, y0_known))
            y1_known = max(y0_known, min(full_h, y1_known))
            x0_known = max(0, min(full_w, x0_known))
            x1_known = max(x0_known, min(full_w, x1_known))
            x0, y0 = int(x0_known), int(y0_known)
            width, height = int(x1_known - x0_known), int(y1_known - y0_known)
        except Exception:
            x0, y0, width, height = (int(v) for v in cv2.boundingRect(arr))
    else:
        x0, y0, width, height = (int(v) for v in cv2.boundingRect(arr))
    if width <= 0 or height <= 0:
        return
    y0h = max(0, int(y0) - 1)
    x0h = max(0, int(x0) - 1)
    y1h = min(full_h, int(y0) + int(height) + 1)
    x1h = min(full_w, int(x0) + int(width) + 1)
    sub = arr[y0h:y1h, x0h:x1h]
    bg = np.equal(sub, 0, order='C').view(np.uint8)
    if not bool(np.any(bg)):
        return
    num_labels, labels2d = _cv2_connected_components(bg, connectivity=4)
    if int(num_labels) <= 1:
        return
    # With a single background component, an exterior corner proves that
    # the entire component is open. Otherwise classify component IDs first
    # and only expand the classification to pixels when a hole exists.
    if int(num_labels) == 2 and labels2d[0, 0] != 0:
        return
    enclosed = np.ones(int(num_labels), dtype=bool)
    enclosed[0] = False
    enclosed[labels2d[0, :]] = False
    enclosed[labels2d[-1, :]] = False
    enclosed[labels2d[:, 0]] = False
    enclosed[labels2d[:, -1]] = False
    if np.any(enclosed):
        sub[enclosed[labels2d]] = np.uint8(1)

def _filter_connected_components_by_min_radius_scipy(
    mask_bool: np.ndarray,
    structure2: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    if mask_bool.size and bool(mask_bool.all()):
        return np.full(mask_bool.shape, math.hypot(*mask_bool.shape) >= float(min_radius), dtype=bool)
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)

    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    dist = np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    radii = np.asarray(ndi.maximum(dist, labels=labels2d, index=label_ids), dtype=np.float32)
    keep_ids = label_ids[radii >= float(min_radius)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)

def _filter_connected_components_by_min_radius_opencv(
    mask_bool: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)
    if bool(mask_u8.all()):
        return np.full(mask_u8.shape, math.hypot(*mask_u8.shape) >= float(min_radius), dtype=bool)

    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = np.zeros((int(num_labels),), dtype=np.float32)
    np.maximum.at(radii, labels2d.ravel(), np.asarray(dist, dtype=np.float32).ravel())
    keep_lookup = radii >= float(min_radius)
    keep_lookup[0] = False
    return keep_lookup[labels2d]

def _filter_connected_components_by_min_conf_opencv(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(mask_bool, dtype=np.uint8)
    if mask_u8.size == 0:
        return np.zeros(mask_u8.shape, dtype=bool)

    region = None
    if (
        mask_u8.ndim == 2 and isinstance(conf_slice, np.ndarray)
        and conf_slice.dtype == np.uint8 and conf_slice.shape == mask_u8.shape
    ):
        x0, y0, width, height = cv2.boundingRect(mask_u8)
        if width == 0 or height == 0:
            return np.zeros(mask_u8.shape, dtype=bool)
        # Every foreground pixel is in this rectangle. Removing exterior zeros
        # preserves all 8-connected components and their confidence maxima.
        # Require a material reduction to offset copying strided crop rows.
        if 2 * width * height <= mask_u8.size:
            region = np.s_[y0:y0 + height, x0:x0 + width]
    elif not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    label_input = mask_u8 if region is None else mask_u8[region]
    num_labels, labels2d = _cv2_connected_components(label_input, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    confidence = np.asarray(conf_slice, dtype=np.uint8)
    if region is not None:
        confidence = confidence[region]
    maxima = np.zeros((int(num_labels),), dtype=np.uint8)
    np.maximum.at(maxima, labels2d.ravel(), confidence.ravel())
    keep_lookup = maxima >= int(min_conf_u8)
    keep_lookup[0] = False
    kept = keep_lookup[labels2d]
    if region is None:
        return kept
    del labels2d
    result = np.zeros(mask_u8.shape, dtype=bool)
    result[region] = kept
    return result

def _filter_connected_components_by_min_conf_scipy(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
    structure2: np.ndarray,
) -> np.ndarray:
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    maxima = np.asarray(ndi.maximum(conf_slice, labels=labels2d, index=label_ids), dtype=np.uint8)
    keep_ids = label_ids[maxima >= int(min_conf_u8)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)

def _cleanup_prediction_slice_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    idx: int,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    backend: str = DEFAULT_CLEANUP_BACKEND,
    structure2: Optional[np.ndarray] = None,
    min_conf_u8: Optional[int] = None,
) -> bool:
    """Gate components by confidence then radius; final view cleanup owns hole filling."""
    idx_i = int(idx)
    if confmap_mm is None and float(min_conf) <= 0.0 and float(min_radius) <= 0.0:
        # nothing to filter and no confidence map to zero — skip the full
        # slice bool-cast + unconditional write-back (a pure read+rewrite no-op at
        # min_conf 0 --min_radius 0) and report foreground from a read-only scan.
        return bool(np.any(np.asarray(mask_mm[idx_i])))
    backend_norm = _normalize_cleanup_backend(backend)
    structure = np.ones((3, 3), dtype=bool) if structure2 is None else structure2
    min_conf_u8_i = (
        int(min_conf_to_u8_threshold(float(min_conf)))
        if min_conf_u8 is None and float(min_conf) > 0.0
        else int(min_conf_u8 or 0)
    )

    mask_slice = np.asarray(mask_mm[idx_i], dtype=bool)
    conf_slice = None if confmap_mm is None else np.asarray(confmap_mm[idx_i], dtype=np.uint8)

    if np.any(mask_slice) and conf_slice is not None and float(min_conf) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_conf_opencv(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
            )
        else:
            mask_slice = _filter_connected_components_by_min_conf_scipy(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
                structure,
            )

    # apply --min_radius BEFORE hole filling. Spec items 5-6 require the order
    # min_conf -> --min_radius -> hole fill, with hole filling applied to the FINAL per-view
    # volume. Hole filling is therefore performed as a separate volume-level 2D pass in
    # cleanup_view_volume_after_prediction_inplace after the (: native per-view) radius
    # filter completes, rather than here in the per-slice unit (which previously hole-filled before radius).
    if np.any(mask_slice) and float(min_radius) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_radius_opencv(
                mask_slice,
                float(min_radius),
            )
        else:
            mask_slice = _filter_connected_components_by_min_radius_scipy(
                mask_slice,
                structure,
                float(min_radius),
            )

    has_foreground = bool(np.any(mask_slice))
    mask_mm[idx_i, :, :] = mask_slice.astype(np.uint8, copy=False)
    if conf_slice is not None:
        if has_foreground:
            conf_slice[~mask_slice] = np.uint8(0)
        else:
            conf_slice.fill(np.uint8(0))
        confmap_mm[idx_i, :, :] = conf_slice.astype(np.uint8, copy=False)
    return bool(has_foreground)

def fused_slice_cleanup_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray] = None,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    workers: int = 1,
    desc: str = 'Fused slice cleanup',
    backend: str = DEFAULT_CLEANUP_BACKEND,
) -> None:
    """Filter slices in chunks, using one distance transform per nonempty slice.

    Component confidence maxima gate first, then component radius maxima; hole
    filling is deliberately deferred until the completed-view boundary.
    """
    num_slices = int(mask_mm.shape[0])
    structure2 = np.ones((3, 3), dtype=bool)
    min_conf_u8 = int(min_conf_to_u8_threshold(float(min_conf))) if float(min_conf) > 0.0 else 0
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    backend = _normalize_cleanup_backend(backend)

    def _process(i: int) -> None:
        _cleanup_prediction_slice_inplace(
            mask_mm,
            confmap_mm,
            int(i),
            min_conf=float(min_conf),
            min_radius=float(min_radius),
            backend=backend,
            structure2=structure2,
            min_conf_u8=int(min_conf_u8),
        )

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
    )

def fill_view_volume_holes_2d_inplace(
    mask_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = '2D hole fill (per-view volume)',
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    backend: str = DEFAULT_CLEANUP_BACKEND,
) -> None:
    """Fill enclosed 2D background components in completed view slices.

    When an inference worker supplied valid per-slice foreground metadata, only
    known-nonempty slices are submitted and their half-open ``(y0,y1,x0,x1)`` bbox is
    passed directly to the topology primitive. Missing or malformed metadata falls back
    to the authoritative full-slice scan.
    """
    num_slices = int(mask_mm.shape[0])
    if num_slices <= 0:
        return
    backend = _normalize_cleanup_backend(backend)

    metadata_ok = False
    any_arr: Optional[np.ndarray] = None
    bbox_arr: Optional[np.ndarray] = None
    if known_slice_any is not None:
        try:
            any_arr = np.asarray(known_slice_any, dtype=bool).reshape(-1)
            if int(any_arr.shape[0]) != int(num_slices):
                raise ValueError('slice-any length mismatch')
            if known_slice_bboxes is not None:
                bbox_arr = np.asarray(known_slice_bboxes, dtype=np.int64)
                if tuple(int(v) for v in bbox_arr.shape) != (int(num_slices), 4):
                    raise ValueError('slice-bbox shape mismatch')
            metadata_ok = True
        except Exception:
            metadata_ok = False
            any_arr = None
            bbox_arr = None

    active_indices = (
        np.flatnonzero(any_arr).astype(np.int64, copy=False)
        if metadata_ok and any_arr is not None
        else np.arange(int(num_slices), dtype=np.int64)
    )
    active_count = int(active_indices.size)
    if metadata_ok:
        print(
            f'{desc}: inference-worker foreground metadata scheduled {active_count}/{num_slices} '
            'slice(s); empty full-plane scans skipped.'
        )
    if active_count <= 0:
        return

    worker_count = choose_slice_parallel_workers(int(workers), active_count)
    chunk_size = choose_parallel_chunk_size(active_count, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(position: int) -> None:
        idx_i = int(active_indices[int(position)])
        arr = np.asarray(mask_mm[idx_i])
        known_bbox: Optional[np.ndarray] = None
        if bbox_arr is not None:
            candidate = np.asarray(bbox_arr[idx_i], dtype=np.int64).reshape(4)
            y0c, y1c, x0c, x1c = (int(v) for v in candidate)
            if (
                0 <= y0c < y1c <= int(arr.shape[0])
                and 0 <= x0c < x1c <= int(arr.shape[1])
            ):
                known_bbox = candidate
        # Without trusted metadata, keep the authoritative emptiness test. With metadata,
        # the committed GPU union has already proved this slice nonempty. An invalid bbox
        # falls back to the backend's normal full-slice bbox discovery for this slice only.
        if not metadata_ok and not bool(arr.any()):
            return
        if backend == 'opencv':
            _fill_holes_2d_opencv_u8_inplace(arr, known_bbox=known_bbox)
        else:
            if known_bbox is None:
                filled = _fill_holes_2d_scipy(arr > 0)
                holes = filled & (arr == 0)
                if bool(np.any(holes)):
                    arr[holes] = np.uint8(1)
                return
            y0, y1, x0, x1 = (int(v) for v in known_bbox)
            y0 = max(0, y0 - 1); x0 = max(0, x0 - 1)
            y1 = min(int(arr.shape[0]), y1 + 1); x1 = min(int(arr.shape[1]), x1 + 1)
            if y1 <= y0 or x1 <= x0:
                return
            sub = arr[y0:y1, x0:x1]
            filled = _fill_holes_2d_scipy(sub > 0)
            holes = filled & (sub == 0)
            if bool(np.any(holes)):
                sub[holes] = np.uint8(1)

    parallel_for_indices_chunked(
        active_count,
        _process,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
    )

def cleanup_view_volume_after_prediction_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    view: ViewInfo,
    min_conf: float,
    min_radius: float,
    *,
    workers: int = 1,
    precleaned_slice_cleanup: bool = False,
    skip_hole_fill: bool = False,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    threshold_plane_shape: Optional[Tuple[int, int]] = None,
    backend: str = DEFAULT_CLEANUP_BACKEND,
) -> None:
    enable_hole_fill = yolo_hole_fill_enabled()
    if (not enable_hole_fill and (
            bool(precleaned_slice_cleanup) or (float(min_conf) <= 0.0 and float(min_radius) <= 0.0))):
        # With no requested filtering or topology work, preserve the resident/packed
        # prediction bytes without even scanning the parent or starting a worker pool.
        return
    # non-azimuthal masks can still be on the canonical inference grid here.
    # Convert the native-view radius once and perform every component operation on that smaller
    # raster. Azimuthal already owns a deliberately folded native raster and keeps its historical
    # threshold unchanged.
    threshold_shape = (
        tuple(int(v) for v in threshold_plane_shape)
        if threshold_plane_shape is not None
        else tuple(int(v) for v in np.asarray(mask_mm).shape[-2:])
    )
    native_min_radius = view_processing_min_radius(
        view, float(min_radius), threshold_shape,
    )

    if bool(precleaned_slice_cleanup):
        # angle-variant inference has already applied per-slice --min_conf and the full
        # view-native --min_radius as results streamed in (streaming_cleanup_min_radius now equals
        # the full radius for every view), so nothing radius-related remains before the final
        # hole-fill pass.
        pass
    else:
        effective_confmap_mm = confmap_mm if float(min_conf) > 0.0 else None

        fused_slice_cleanup_inplace(
            mask_mm,
            effective_confmap_mm,
            min_conf=float(min_conf),
            min_radius=float(native_min_radius),
            workers=int(workers),
            desc=f'Fused cleanup ({view.name})',
            backend=backend,
        )

    # 2D hole filling is the FINAL per-view step, applied after
    # min_conf and the view-native --min_radius filter. The previous order hole-filled before
    # min_radius inside the per-slice unit.
    # skipped when every slice of this view was already hole-filled on device by
    # the GPU workers (identical per-slice semantics; the CPU pass would be pure recompute).
    if not enable_hole_fill:
        return
    if int(view.augmentation_pass) > 0:
        # An inverse-invalid island is unknown, not an enclosed background hole.
        # Filtering/removal remains eligible; adding unsupported foreground does not.
        print(f'2D hole fill ({view.name}): skipped for inverse-mapped external-policy masks.')
    elif bool(skip_hole_fill):
        print(f'2D hole fill ({view.name}): done on device during accumulation; CPU pass skipped.')
    else:
        fill_view_volume_holes_2d_inplace(
            mask_mm,
            workers=int(workers),
            desc=f'2D hole fill ({view.name})',
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
            backend=backend,
        )
