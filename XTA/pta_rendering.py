"""PTA render-plan values and CPU geometry used by parent and worker processes.

This module is deliberately independent of :mod:`XTA.pta`.  Every class
serialized in a render-phase payload and every CPU rendering primitive used by
spawned workers has a canonical owner here, except ``TileConfig``, which
:mod:`XTA.unification.tiles` owns and this module re-exports.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

from . import geometry as shared_geometry
from .pta_cpr import CprGeometry, render_cpr_frame
from .unification.contracts import RasterPlan
from .unification.sampling import forward_sampling_policy, require_forward_sampling
from .unification.tiles import TileConfig


@dataclass(frozen=True)
class ChannelVariant:
    """One physical output set, including custom forward/reverse ordering."""

    format_token: str
    kind: str  # gray, rgb, custom
    channel_count: int
    stride: int
    reverse: bool
    offsets: Tuple[int, ...]

    @property
    def tag_token(self) -> str:
        if self.kind == "custom" and self.reverse:
            return f"{self.format_token}_reverse"
        return self.format_token

    @property
    def order_name(self) -> str:
        return "reverse" if self.reverse else "forward"


DEFAULT_CHANNEL_VARIANT = ChannelVariant(
    format_token="gray",
    kind="gray",
    channel_count=1,
    stride=1,
    reverse=False,
    offsets=(0,),
)


@dataclass(frozen=True)
class ViewInfo:
    name: str
    display_name: str
    # adapt_shared_view() token (transverse, sagittal, coronal, azimuthal,
    # tilted_transverse for any Tilted base, radial, spherical), or "cpr".
    family: str
    num_slices: int
    src_h: int
    src_w: int
    pad_mode: str  # clamp or pad
    full_t: int
    full_h: int
    full_w: int
    azimuths_deg: Tuple[float, ...] = ()
    diameter: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    roi_radius: float = 0.0
    tilt_angle_deg: float = 0.0
    tilt_direction: str = ""
    # Canonical shared physical view.  Legacy-compatible scalar fields above are
    # presentation/planner adapters only; all built-in pixels come from this
    # shared TTA geometry object when present.
    shared_view: Optional[shared_geometry.ViewInfo] = None
    cpr_geometry: Optional[CprGeometry] = None


@dataclass(frozen=True)
class AffineSpec:
    angle_deg: float
    src_w: int
    src_h: int
    canvas_w: int
    canvas_h: int
    out_w: int
    out_h: int
    pad_off_x: float
    pad_off_y: float
    M_src_to_canvas: np.ndarray
    M_canvas_to_src: np.ndarray
    M_src_to_out: np.ndarray
    M_out_to_src: np.ndarray
    # TTA's exact affine/canvas object.  ``native_output`` represents PTA's
    # mode-specific imgsz=0 default while retaining the shared physical view.
    shared_affine: Optional[shared_geometry.AffineSpec] = None
    native_output: bool = False


def get_native_view_image(volume: np.ndarray, view: ViewInfo, idx: int) -> np.ndarray:
    """Native CPR cross-section; every other family renders through ``shared_view``."""
    if view.family == "cpr" and view.cpr_geometry is not None:
        return render_cpr_frame(volume, view.cpr_geometry, int(idx))
    raise ValueError(f"Native image frame for {view.family} requires a canonical shared view")


def get_native_view_frame(volume: np.ndarray, mask: np.ndarray, view: ViewInfo, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    return get_native_view_image(volume, view, int(idx)), get_native_view_mask(mask, view, int(idx))


def warp_pair(native_img: np.ndarray, native_mask: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.warpAffine(
        np.ascontiguousarray(native_img), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    msk = cv2.warpAffine(
        np.ascontiguousarray(native_mask), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray(img), np.ascontiguousarray((msk > 0).astype(np.uint8))


def warp_image(native_img: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    image = cv2.warpAffine(
        np.ascontiguousarray(native_img), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray(image, dtype=np.uint8)


def _shared_render_full_intensity(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_azimuthal_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None:
        raise ValueError("shared intensity rendering requires view.shared_view")
    return np.ascontiguousarray(
        shared_geometry.render_intensity_frame_on_grid(
            volume,
            view.shared_view,
            int(idx),
            M_src_to_out=aff.M_src_to_out,
            M_out_to_src=aff.M_out_to_src,
            output_height=int(aff.out_h),
            output_width=int(aff.out_w),
            mirror_azimuthal_u=bool(mirror_azimuthal_u),
        ),
        dtype=np.uint8,
    )


def _shared_render_canvas_intensity(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_azimuthal_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None or aff.shared_affine is None:
        raise ValueError("shared canvas rendering requires view.shared_view and an affine")
    if shared_geometry.is_tilted_view(view.shared_view):
        return np.ascontiguousarray(
            shared_geometry.render_tilted_canvas_frame(
                volume, view.shared_view, int(idx), aff.shared_affine
            ),
            dtype=np.uint8,
        )
    native = np.ascontiguousarray(
        shared_geometry.get_view_frame_by_index(volume, view.shared_view, int(idx))
    )
    if bool(mirror_azimuthal_u):
        if not shared_geometry.is_azimuthal_view(view.shared_view):
            raise ValueError("azimuthal-u mirroring requested for a non-azimuthal view")
        native = np.ascontiguousarray(native[:, ::-1])
    return warp_image(
        native,
        aff.shared_affine.M_src_to_canvas,
        int(aff.shared_affine.canvas_w),
        int(aff.shared_affine.canvas_h),
    )


def _shared_render_full_mask(
    mask: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_azimuthal_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None:
        raise ValueError("shared categorical rendering requires view.shared_view")
    return np.ascontiguousarray(
        shared_geometry.render_categorical_frame_on_grid(
            mask,
            view.shared_view,
            int(idx),
            M_src_to_out=aff.M_src_to_out,
            M_out_to_src=aff.M_out_to_src,
            output_height=int(aff.out_h),
            output_width=int(aff.out_w),
            mirror_azimuthal_u=bool(mirror_azimuthal_u),
        ),
        dtype=np.uint8,
    )


def _shared_render_canvas_mask(
    mask: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
) -> np.ndarray:
    if view.shared_view is None or aff.shared_affine is None:
        raise ValueError("shared categorical canvas requires view.shared_view and an affine")
    if shared_geometry.is_tilted_view(view.shared_view):
        rendered = shared_geometry._render_tilted_array_on_grid(
            mask,
            view.shared_view,
            int(idx),
            aff.shared_affine.M_canvas_to_src,
            int(aff.shared_affine.canvas_h),
            int(aff.shared_affine.canvas_w),
            mask_mode=True,
        )
        return np.ascontiguousarray((np.asarray(rendered) > 0).astype(np.uint8))
    native = shared_geometry.get_categorical_view_frame_by_index(
        mask, view.shared_view, int(idx)
    )
    return warp_mask_only(
        native,
        aff.shared_affine.M_src_to_canvas,
        int(aff.shared_affine.canvas_w),
        int(aff.shared_affine.canvas_h),
    )


def render_image_full_and_optional_canvas(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    need_canvas: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render only the image side of one source-view slice."""
    if view.shared_view is not None:
        image_full = _shared_render_full_intensity(volume, view, int(idx), aff)
        image_canvas = (
            _shared_render_canvas_intensity(volume, view, int(idx), aff)
            if bool(need_canvas)
            else None
        )
        return image_full, image_canvas
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    native_image = get_native_view_image(volume, view, int(idx))
    image_full = warp_image(native_image, aff.M_src_to_out, aff.out_w, aff.out_h)
    if shared_full_canvas:
        return image_full, image_full
    if need_canvas:
        return image_full, warp_image(native_image, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return image_full, None


def render_full_and_optional_canvas(volume: np.ndarray, mask: np.ndarray, view: ViewInfo, idx: int, aff: AffineSpec, need_canvas: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if view.shared_view is not None:
        image_full = _shared_render_full_intensity(volume, view, int(idx), aff)
        mask_full = _shared_render_full_mask(mask, view, int(idx), aff)
        if need_canvas:
            return (
                image_full,
                mask_full,
                _shared_render_canvas_intensity(volume, view, int(idx), aff),
                _shared_render_canvas_mask(mask, view, int(idx), aff),
            )
        return image_full, mask_full, None, None
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    img_native, mask_native = get_native_view_frame(volume, mask, view, idx)
    img_full, mask_full = warp_pair(img_native, mask_native, aff.M_src_to_out, aff.out_w, aff.out_h)
    img_canvas = mask_canvas = None
    if shared_full_canvas:
        img_canvas, mask_canvas = img_full, mask_full
    elif need_canvas:
        img_canvas, mask_canvas = warp_pair(img_native, mask_native, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return img_full, mask_full, img_canvas, mask_canvas


def resize_centered(frame: np.ndarray, out_w: int, out_h: int, interpolation: int) -> np.ndarray:
    if int(frame.shape[1]) == int(out_w) and int(frame.shape[0]) == int(out_h):
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(cv2.resize(
        np.asarray(frame),
        dsize=(int(out_w), int(out_h)),
        interpolation=int(interpolation),
    ))


def extract_padded_tile(frame: np.ndarray, x0: int, y0: int, tile_size: int) -> np.ndarray:
    arr = np.asarray(frame)
    if (
        int(x0) >= 0
        and int(y0) >= 0
        and int(x0) + int(tile_size) <= int(arr.shape[1])
        and int(y0) + int(tile_size) <= int(arr.shape[0])
    ):
        # cv2.resize accepts non-contiguous ROI views, avoiding a tile-sized
        # zero-fill and copy for every normal tile_positions() result.
        return arr[int(y0):int(y0) + int(tile_size), int(x0):int(x0) + int(tile_size), ...]
    tile_shape = (int(tile_size), int(tile_size)) + tuple(int(x) for x in arr.shape[2:])
    tile = np.zeros(tile_shape, dtype=arr.dtype)
    sx0 = max(0, int(x0))
    sy0 = max(0, int(y0))
    sx1 = min(int(frame.shape[1]), int(x0) + int(tile_size))
    sy1 = min(int(frame.shape[0]), int(y0) + int(tile_size))
    if sx0 >= sx1 or sy0 >= sy1:
        return tile
    dx0 = sx0 - int(x0)
    dy0 = sy0 - int(y0)
    tile[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0), ...] = arr[sy0:sy1, sx0:sx1, ...]
    return tile


@dataclass(frozen=True)
class RenderTileItem:
    cfg: TileConfig
    x: int
    y: int
    tile_tag: str
    out_w: int
    out_h: int
    img_pattern: str
    lbl_pattern: str
    overlay_path: Optional[Path]
    label_enabled: bool
    channel_kind: str = "gray"
    shared_job: Optional[shared_geometry.TileJob] = None
    canonical_plan: Optional[RasterPlan] = None
    publish_images: bool = True
    publish_labels: bool = True


@dataclass(frozen=True)
class RenderPlan:
    view: ViewInfo
    aff: AffineSpec
    tag: str
    img_pattern: str
    lbl_pattern: str
    overlay_path: Optional[Path]
    label_enabled: bool
    tile_layout: Tuple[RenderTileItem, ...]
    stats: Dict[str, object]
    channel_variant: ChannelVariant = DEFAULT_CHANNEL_VARIANT
    # None means every view frame is eligible.  A tuple is used for partial
    # volumes so unannotated centers and discontinuous C...S... centers never
    # enter classification, splitting, augmentation, or rendering.
    eligible_frame_indices: Optional[Tuple[int, ...]] = None
    # Native encoded image indices for a partial transverse sequence.  Custom
    # channels resolve offsets against these values rather than compact array
    # positions, preserving gaps and true-volume edge clamping.
    source_encoded_indices: Tuple[int, ...] = ()
    shared_aug_job: Optional[shared_geometry.AugJob] = None
    canonical_plan: Optional[RasterPlan] = None
    publish_images: bool = True
    publish_labels: bool = True


@dataclass
class RenderFrameSource:
    img_full: Optional[np.ndarray]
    mask_full: Optional[np.ndarray]
    img_canvas: Optional[np.ndarray]
    mask_canvas: Optional[np.ndarray]
    # Canonical direct-to-output tile rasters.  These prevent PTA from
    # reintroducing a canvas-crop-resize interpolation layer after TTA's
    # collapsed tile transform.
    tile_arrays: Dict[str, Tuple[Optional[np.ndarray], np.ndarray]] = field(default_factory=dict)


def edge_clamped_view_index(view: ViewInfo, index: int) -> int:
    return max(0, min(int(view.num_slices) - 1, int(index)))


class EncodedSliceIndices(tuple):
    """Immutable sorted integer lattice shared by partial-sequence channel lookups."""

    __slots__ = ()

    def __new__(cls, indices, *, description="Encoded image indices"):
        if type(indices) is cls:
            return indices
        encoded = super().__new__(cls, map(int, indices))
        if any(encoded[i] >= encoded[i + 1] for i in range(len(encoded) - 1)):
            raise ValueError(f"{description} must be strictly increasing, got {encoded[:24]}")
        return encoded


def encoded_channel_source_positions(
    encoded_indices: Sequence[int],
    center_position: int,
    offsets: Sequence[int],
) -> Tuple[Optional[Tuple[int, ...]], Tuple[int, ...]]:
    """Resolve custom-channel offsets against original encoded image indices.

    Requests below the minimum or above the maximum encoded image index are
    true-volume boundary requests and clamp to the nearest edge.  A requested
    index *inside* those bounds must exist exactly; otherwise the center crosses
    a discontinuity and is ineligible.
    """
    encoded = EncodedSliceIndices(encoded_indices)
    if not encoded:
        raise ValueError("encoded_channel_source_positions requires at least one encoded index")
    center_pos = int(center_position)
    if center_pos < 0 or center_pos >= len(encoded):
        raise IndexError(f"Center position {center_pos} is outside encoded index mapping of length {len(encoded)}")
    lower = int(encoded[0])
    upper = int(encoded[-1])
    center_encoded = int(encoded[center_pos])
    contiguous = upper - lower + 1 == len(encoded)
    positions: List[int] = []
    missing: List[int] = []
    for offset in offsets:
        requested = center_encoded + int(offset)
        resolved = lower if requested < lower else (upper if requested > upper else requested)
        position = resolved - lower if contiguous else bisect_left(encoded, resolved)
        if position >= len(encoded) or int(encoded[position]) != int(resolved):
            missing.append(int(resolved))
        else:
            positions.append(int(position))
    if missing:
        return None, tuple(sorted(set(int(x) for x in missing)))
    return tuple(positions), ()


def _require_pta_canonical_plan(
    plan: RenderPlan,
    role: str,
    *,
    tile: Optional[RenderTileItem] = None,
    backend: str = "cpu",
) -> None:
    """Mechanically bind shared PTA work to one registered sampling backend."""

    if plan.view.shared_view is None:
        return
    canonical = tile.canonical_plan if tile is not None else plan.canonical_plan
    if canonical is None:
        raise RuntimeError(
            f"shared PTA geometry {plan.tag!r} has no canonical RasterPlan"
        )
    current_policy = forward_sampling_policy()
    if canonical.sampling_policy.digest != current_policy.digest:
        raise RuntimeError(
            f"PTA RasterPlan policy drift for {plan.tag!r}: "
            f"plan={canonical.sampling_policy.digest}, current={current_policy.digest}"
        )
    require_forward_sampling(str(backend), role)


def render_channel_formatted_images(
    *,
    volume: np.ndarray,
    plan: RenderPlan,
    idx: int,
    need_canvas: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render one center slice N into the plan's requested channel layout."""
    _require_pta_canonical_plan(plan, "intensity")
    variant = plan.channel_variant
    center_idx = int(idx)

    if variant.kind in {"gray", "rgb"}:
        image_full, image_canvas = render_image_full_and_optional_canvas(
            volume,
            plan.view,
            center_idx,
            plan.aff,
            bool(need_canvas),
        )
        if variant.kind == "gray":
            return image_full, image_canvas
        rgb_full = np.ascontiguousarray(np.repeat(image_full[:, :, None], 3, axis=2))
        rgb_canvas = (
            np.ascontiguousarray(np.repeat(image_canvas[:, :, None], 3, axis=2))
            if image_canvas is not None
            else None
        )
        return rgb_full, rgb_canvas

    if plan.source_encoded_indices:
        source_positions, missing = encoded_channel_source_positions(
            plan.source_encoded_indices,
            center_idx,
            variant.offsets,
        )
        if source_positions is None:
            center_encoded = int(plan.source_encoded_indices[center_idx])
            raise RuntimeError(
                f"Ineligible discontinuous channel center reached rendering for {plan.tag}: "
                f"center encoded index={center_encoded}, missing required index/indices={list(missing)}"
            )
        source_addresses = tuple((int(position), False) for position in source_positions)
    else:
        if plan.view.shared_view is not None:
            source_addresses = tuple(
                shared_geometry.channel_view_slice_source(
                    plan.view.shared_view, center_idx + int(offset)
                )
                for offset in variant.offsets
            )
        else:
            source_addresses = tuple(
                (edge_clamped_view_index(plan.view, center_idx + int(offset)), False)
                for offset in variant.offsets
            )

    full_planes: List[np.ndarray] = []
    canvas_planes: List[np.ndarray] = []
    for source_idx, mirror_u in source_addresses:
        if plan.view.shared_view is not None:
            image_full = _shared_render_full_intensity(
                volume,
                plan.view,
                int(source_idx),
                plan.aff,
                mirror_azimuthal_u=bool(mirror_u),
            )
            image_canvas = (
                _shared_render_canvas_intensity(
                    volume,
                    plan.view,
                    int(source_idx),
                    plan.aff,
                    mirror_azimuthal_u=bool(mirror_u),
                )
                if need_canvas
                else None
            )
        else:
            image_full, image_canvas = render_image_full_and_optional_canvas(
                volume,
                plan.view,
                int(source_idx),
                plan.aff,
                bool(need_canvas),
            )
        full_planes.append(image_full)
        if need_canvas:
            if image_canvas is None:
                raise RuntimeError(f"Channel stack requested a missing image canvas for {plan.tag}")
            canvas_planes.append(image_canvas)
    stacked_full = np.ascontiguousarray(np.stack(full_planes, axis=2), dtype=np.uint8)
    stacked_canvas = (
        np.ascontiguousarray(np.stack(canvas_planes, axis=2), dtype=np.uint8)
        if need_canvas
        else None
    )
    return stacked_full, stacked_canvas


def render_shared_tile_images(
    *,
    volume: np.ndarray,
    plan: RenderPlan,
    tile: RenderTileItem,
    idx: int,
) -> np.ndarray:
    """Render one PTA tile through TTA's collapsed direct-to-output transform."""

    _require_pta_canonical_plan(plan, "intensity", tile=tile)
    if plan.view.shared_view is None or tile.shared_job is None:
        raise ValueError("shared tile rendering requires canonical view and tile jobs")
    variant = plan.channel_variant
    center_idx = int(idx)

    def _plane(source_idx: int, mirror_u: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            shared_geometry.render_tile_frame_for_job(
                volume,
                plan.view.shared_view,
                tile.shared_job,
                int(source_idx),
                mirror_azimuthal_u=bool(mirror_u),
            ),
            dtype=np.uint8,
        )

    if variant.kind == "gray":
        source_idx, mirror_u = shared_geometry.channel_view_slice_source(
            plan.view.shared_view, center_idx
        )
        return _plane(source_idx, mirror_u)
    if variant.kind == "rgb":
        source_idx, mirror_u = shared_geometry.channel_view_slice_source(
            plan.view.shared_view, center_idx
        )
        plane = _plane(source_idx, mirror_u)
        return np.ascontiguousarray(np.repeat(plane[:, :, None], 3, axis=2))

    if plan.source_encoded_indices:
        source_positions, missing = encoded_channel_source_positions(
            plan.source_encoded_indices,
            center_idx,
            variant.offsets,
        )
        if source_positions is None:
            center_encoded = int(plan.source_encoded_indices[center_idx])
            raise RuntimeError(
                f"Ineligible discontinuous tile-channel center reached rendering for {plan.tag}: "
                f"center encoded index={center_encoded}, missing required index/indices={list(missing)}"
            )
        addresses = tuple((int(position), False) for position in source_positions)
    else:
        addresses = tuple(
            shared_geometry.channel_view_slice_source(
                plan.view.shared_view, center_idx + int(offset)
            )
            for offset in variant.offsets
        )
    return np.ascontiguousarray(
        np.stack([_plane(source_idx, mirror_u) for source_idx, mirror_u in addresses], axis=2),
        dtype=np.uint8,
    )


def render_plan_frame_source(
    *, volume: np.ndarray, mask: np.ndarray, plan: RenderPlan, idx: int,
    need_canvas: Optional[bool] = None, need_full: bool = True, need_image: bool = True,
) -> RenderFrameSource:
    """Render one center slice and its requested channel context.

    Tasks that only feed the full-frame item pass
    ``need_canvas=False`` so no canvas is rendered for them.
    Tile-only tasks skip the full raster when all tiles have canonical jobs.
    Legacy canvas tiles still use the combined full/canvas renderer.

    Image channels may come from neighboring view slices, but both
    full-frame and tile labels always use the center slice ``idx``.  Native
    partial sequences resolve custom-channel neighbors by encoded image index.
    """
    if need_canvas is None:
        need_canvas = bool(plan.tile_layout)
    shared_tiles = tuple(tile for tile in plan.tile_layout if tile.shared_job is not None)
    canvas_required = bool(need_canvas) and len(shared_tiles) != len(plan.tile_layout)
    img_full = img_canvas = mask_full = mask_canvas = None
    if need_full or canvas_required:
        if need_image:
            img_full, img_canvas = render_channel_formatted_images(
                volume=volume,
                plan=plan,
                idx=int(idx),
                need_canvas=bool(canvas_required),
            )
        mask_full, mask_canvas = render_plan_frame_mask_source(
            mask=mask,
            plan=plan,
            idx=int(idx),
            need_canvas=bool(canvas_required),
        )
    tile_arrays: Dict[str, Tuple[Optional[np.ndarray], np.ndarray]] = {}
    for tile in shared_tiles:
        assert plan.view.shared_view is not None and tile.shared_job is not None
        tile_image = None
        if need_image:
            tile_image = render_shared_tile_images(volume=volume, plan=plan, tile=tile, idx=int(idx))
        else:
            _require_pta_canonical_plan(plan, "categorical_ground_truth", tile=tile)
        tile_mask = shared_geometry.render_categorical_tile_for_job(
            mask,
            plan.view.shared_view,
            tile.shared_job,
            int(idx),
        )
        tile_arrays[str(tile.tile_tag)] = (
            tile_image,
            np.ascontiguousarray((np.asarray(tile_mask) > 0).astype(np.uint8)),
        )
    return RenderFrameSource(
        img_full=img_full,
        mask_full=mask_full,
        img_canvas=img_canvas,
        mask_canvas=mask_canvas,
        tile_arrays=tile_arrays,
    )


def get_native_view_mask(mask: np.ndarray, view: ViewInfo, idx: int) -> np.ndarray:
    """Native CPR categorical cross-section; every other family renders through ``shared_view``."""
    if view.family == "cpr" and view.cpr_geometry is not None:
        return render_cpr_frame(mask, view.cpr_geometry, int(idx), categorical=True)
    raise ValueError(f"Native mask frame for {view.family} requires a canonical shared view")


def warp_mask_only(native_mask: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    msk = cv2.warpAffine(
        np.ascontiguousarray(native_mask), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray((msk > 0).astype(np.uint8))


def render_plan_frame_mask_source(*, mask: np.ndarray, plan: RenderPlan, idx: int, need_canvas: Optional[bool] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render only the mask side of one view frame.

    Copy-0 foreground classification never needs image pixels, so
    the planning pass skips the image slice/warp entirely.
    """
    _require_pta_canonical_plan(plan, "categorical_ground_truth")
    view = plan.view
    aff = plan.aff
    if need_canvas is None:
        need_canvas = bool(plan.tile_layout)
    need_canvas = bool(need_canvas)
    if view.shared_view is not None:
        mask_full = _shared_render_full_mask(mask, view, int(idx), aff)
        return (
            mask_full,
            _shared_render_canvas_mask(mask, view, int(idx), aff)
            if need_canvas
            else None,
        )
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    native = get_native_view_mask(mask, view, int(idx))
    mask_full = warp_mask_only(native, aff.M_src_to_out, aff.out_w, aff.out_h)
    if shared_full_canvas:
        return mask_full, mask_full
    if need_canvas:
        return mask_full, warp_mask_only(native, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return mask_full, None


__all__ = [
    "AffineSpec",
    "ChannelVariant",
    "DEFAULT_CHANNEL_VARIANT",
    "RenderFrameSource",
    "RenderPlan",
    "RenderTileItem",
    "TileConfig",
    "ViewInfo",
    "edge_clamped_view_index",
    "encoded_channel_source_positions",
    "extract_padded_tile",
    "render_channel_formatted_images",
    "render_image_full_and_optional_canvas",
    "render_plan_frame_mask_source",
    "render_plan_frame_source",
    "render_shared_tile_images",
    "resize_centered",
]
