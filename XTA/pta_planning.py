"""PTA physical-view adapters and render-plan construction.

Planning uses canonical shared geometry and the value classes owned by
:mod:`XTA.pta_rendering`; it never imports the PTA command-line orchestrator.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import geometry as shared_geometry
from .geometry import _affine2x3_to_3x3 as affine2x3_to_3x3
from .pta_publication import output_image_suffix, parse_output_image_format
from .pta_rendering import (
    DEFAULT_CHANNEL_VARIANT, AffineSpec, ChannelVariant, RenderPlan,
    RenderTileItem, TileConfig, ViewInfo, encoded_channel_source_positions,
)
from .pta_workers import WarningLog
from .runtime import runtime_telemetry_phase
from .unification.contracts import RasterPlan
from .unification.runtime import compile_physical_views
from .unification.sampling import build_forward_raster_plan, forward_sampling_policy
from .unification.tta_manifest import (
    projection_sampling_record, radial_view_plan_metadata,
    spherical_view_plan_metadata,
)



def make_channel_tag(base_tag: str, variant: ChannelVariant) -> str:
    # Preserve v4 filenames for the default single-channel output.  Additional
    # formats receive explicit suffixes so multiple requested formats cannot
    # collide within one run.
    if variant.kind == "gray":
        return str(base_tag)
    return f"{base_tag}_{variant.tag_token}"


def adapt_shared_view(view: shared_geometry.ViewInfo) -> ViewInfo:
    """Expose a canonical TTA view to the unchanged PTA planner vocabulary."""

    physical_name = shared_geometry.physical_view_name(view)
    if shared_geometry.is_azimuthal_view(view):
        runtime_family = "azimuthal"
    elif str(view.family) == "radial":
        runtime_family = "radial"
    elif shared_geometry.is_spherical_view(view):
        runtime_family = "spherical"
    elif shared_geometry.is_tilted_view(view):
        # The old token described only transverse tilts.  It is retained solely
        # for PTA summary/filter call sites; ``shared_view`` owns the base axes.
        runtime_family = "tilted_transverse"
    else:
        runtime_family = str(physical_name)
    return ViewInfo(
        name=str(view.name),
        display_name=str(shared_geometry.view_output_token(view)),
        family=runtime_family,
        num_slices=int(view.num_slices),
        src_h=int(view.src_h),
        src_w=int(view.src_w),
        pad_mode=str(view.pad_mode),
        full_t=int(view.full_t),
        full_h=int(view.full_h),
        full_w=int(view.full_w),
        azimuths_deg=tuple(float(value) for value in view.azimuths_deg),
        diameter=int(view.diameter),
        center_x=float(view.center_x),
        center_y=float(view.center_y),
        roi_radius=float(view.roi_radius),
        tilt_angle_deg=float(view.tilt_angle_deg),
        tilt_direction=str(view.tilt_direction),
        shared_view=view,
    )


def compile_v18_pta_views(
    *,
    t_dim: int,
    h: int,
    w: int,
    config: object,
    azimuthal_native_raster: int,
) -> Tuple[List[ViewInfo], object]:
    """Compile PTA grouped requests with the exact shared TTA view compiler."""

    compiled = compile_physical_views(
        t_dim=int(t_dim),
        height=int(h),
        width=int(w),
        cartesian_views=tuple(getattr(config, "cartesian_views")),
        azimuthal_requests=tuple(getattr(config, "azimuthal_requests")),
        tilted_groups=tuple(getattr(config, "tilted_groups")),
        azimuthal_native_raster=int(azimuthal_native_raster),
        radial_requests=tuple(getattr(config, "radial_requests", ())),
        radial_min_radius=getattr(getattr(config, "args", None), "radial_min_radius", None),
        radial_patch_size=int(azimuthal_native_raster) if int(azimuthal_native_raster) > 0 else 3072,
        spherical_requests=tuple(getattr(config, "spherical_requests", ())),
        spherical_min_radius=getattr(getattr(config, "args", None), "spherical_min_radius", None),
        spherical_patch_size=int(azimuthal_native_raster),
        augmentation_angles=(0.0,),
    )
    return [adapt_shared_view(view) for view in compiled.views], compiled


def rotated_bbox_size(w: int, h: int, angle_deg: float) -> Tuple[int, int]:
    theta = math.radians(float(angle_deg) % 360.0)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    bw = int(math.ceil(float(w) * c + float(h) * s))
    bh = int(math.ceil(float(w) * s + float(h) * c))
    return max(1, bw), max(1, bh)


def center_scale_matrix(src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    cx_s = (int(src_w) - 1) / 2.0
    cy_s = (int(src_h) - 1) / 2.0
    cx_o = (int(out_w) - 1) / 2.0
    cy_o = (int(out_h) - 1) / 2.0
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    return np.array([[sx, 0.0, cx_o - sx * cx_s], [0.0, sy, cy_o - sy * cy_s], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_affine(
    src_w: int,
    src_h: int,
    angle_deg: float,
    pad_mode: str,
    imgsz: int,
    *,
    shared_view: Optional[shared_geometry.ViewInfo] = None,
) -> AffineSpec:
    if shared_view is not None:
        # TTA owns padding, rotation, center alignment, and square scaling.
        # PTA's imgsz=0 remains a native-raster publication default; the
        # canonical canvas matrices are still retained for grouped tiles.
        native_output = int(imgsz) == 0
        shared_out_size = int(imgsz) if int(imgsz) > 0 else max(int(src_w), int(src_h), 1)
        shared_affine = shared_geometry.build_affine(
            view=str(shared_view.name),
            src_w=int(src_w),
            src_h=int(src_h),
            out_size=int(shared_out_size),
            angle_deg=float(angle_deg),
            pad_mode=str(pad_mode),
        )
        if native_output:
            identity = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            )
            out_w, out_h = int(src_w), int(src_h)
            src_to_out = identity
            out_to_src = identity
        else:
            out_w = out_h = int(shared_affine.out_size)
            src_to_out = np.asarray(shared_affine.M_src_to_out, dtype=np.float32)
            out_to_src = np.asarray(shared_affine.M_out_to_src, dtype=np.float32)
        return AffineSpec(
            angle_deg=float(angle_deg),
            src_w=int(src_w),
            src_h=int(src_h),
            canvas_w=int(shared_affine.canvas_w),
            canvas_h=int(shared_affine.canvas_h),
            out_w=int(out_w),
            out_h=int(out_h),
            pad_off_x=float(shared_affine.pad_off_x),
            pad_off_y=float(shared_affine.pad_off_y),
            M_src_to_canvas=np.asarray(shared_affine.M_src_to_canvas, dtype=np.float32),
            M_canvas_to_src=np.asarray(shared_affine.M_canvas_to_src, dtype=np.float32),
            M_src_to_out=src_to_out,
            M_out_to_src=out_to_src,
            shared_affine=shared_affine,
            native_output=bool(native_output),
        )
    if pad_mode not in {"clamp", "pad"}:
        raise ValueError("pad_mode must be clamp or pad")
    if pad_mode == "pad":
        canvas_w, canvas_h = rotated_bbox_size(int(src_w), int(src_h), float(angle_deg))
    else:
        canvas_w, canvas_h = int(src_w), int(src_h)
    off_x = (canvas_w - int(src_w)) / 2.0
    off_y = (canvas_h - int(src_h)) / 2.0
    cx = (canvas_w - 1) / 2.0
    cy = (canvas_h - 1) / 2.0
    M_pad = np.array([[1.0, 0.0, off_x], [0.0, 1.0, off_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_rot = affine2x3_to_3x3(cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0))
    M_src_to_canvas3 = M_rot @ M_pad
    M_canvas_to_src3 = np.linalg.inv(M_src_to_canvas3)
    if int(imgsz) > 0:
        out_w = out_h = int(imgsz)
        M_scale = center_scale_matrix(canvas_w, canvas_h, out_w, out_h)
    else:
        out_w, out_h = int(canvas_w), int(canvas_h)
        M_scale = np.eye(3, dtype=np.float64)
    M_src_to_out3 = M_scale @ M_src_to_canvas3
    M_out_to_src3 = np.linalg.inv(M_src_to_out3)
    return AffineSpec(
        float(angle_deg), int(src_w), int(src_h), int(canvas_w), int(canvas_h), int(out_w), int(out_h), float(off_x), float(off_y),
        M_src_to_canvas3[:2, :].astype(np.float32), M_canvas_to_src3[:2, :].astype(np.float32),
        M_src_to_out3[:2, :].astype(np.float32), M_out_to_src3[:2, :].astype(np.float32),
    )


def dense_tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride)) if last > 0 else [0]
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return [int(x) for x in starts]


def build_render_plan(
    *,
    view: ViewInfo,
    aff: AffineSpec,
    tag: str,
    out_dir: Path,
    stem: str,
    tile_configs: Sequence[TileConfig],
    save_overlay: bool,
    imgsz: int,
    label_enabled: bool,
    image_format: str = "png",
    channel_variant: ChannelVariant = DEFAULT_CHANNEL_VARIANT,
    eligible_frame_indices: Optional[Sequence[int]] = None,
    source_encoded_indices: Sequence[int] = (),
    publish_images: bool = True,
    publish_labels: bool = True,
) -> RenderPlan:
    from .unification.geometry_identity import geometry_recipe_metadata

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    if publish_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if label_enabled and publish_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    image_suffix = output_image_suffix(image_format)
    img_pattern = str(images_dir / f"{stem}_{tag}_%04d{image_suffix}")
    lbl_pattern = str(labels_dir / f"{stem}_{tag}_%04d.txt")
    overlay_path = overlays_dir / f"{stem}_{tag}_Overlay.mkv" if save_overlay else None

    shared_aug_job: Optional[shared_geometry.AugJob] = None
    if view.shared_view is not None and aff.shared_affine is not None:
        shared_aug_job = shared_geometry.AugJob(
            aug_id="a0",
            angle_deg=0.0,
            meta_path=out_dir / ".v18_geometry" / f"{stem}_{tag}.meta.json",
            aff=aff.shared_affine,
        )
    canonical_plan: Optional[RasterPlan] = None
    if view.shared_view is not None:
        canonical_plan = build_forward_raster_plan(
            mode="pta",
            physical_view_id=shared_geometry.physical_view_name(view.shared_view),
            angle_deg=float(aff.angle_deg),
            channel_token=str(channel_variant.format_token),
            channel_kind=str(channel_variant.kind),
            channel_count=int(channel_variant.channel_count),
            channel_stride=int(channel_variant.stride),
            channel_offsets=tuple(int(value) for value in channel_variant.offsets),
            channel_direction=str(channel_variant.order_name),
            output_shape=(int(aff.out_h), int(aff.out_w)),
            metadata={
                "runtime_view_id": str(view.name),
                "runtime_job_id": str(tag),
                "runtime_kind": "fullframe",
                **geometry_recipe_metadata(view.shared_view, aff),
                **projection_sampling_record(view.shared_view),
                **radial_view_plan_metadata(view.shared_view),
                **spherical_view_plan_metadata(view.shared_view),
            },
        )

    tile_items: List[RenderTileItem] = []
    for cfg in tile_configs:
        out_side = int(imgsz) if int(imgsz) > 0 else int(cfg.tile_size)
        if view.shared_view is not None and shared_aug_job is not None:
            shared_jobs: Sequence[Optional[shared_geometry.DenseTileJob]] = tuple(
                shared_geometry.build_dense_tile_jobs_for_aug(
                    view.shared_view,
                    shared_aug_job,
                    shared_geometry.TileConfig(
                        tile_size=int(cfg.tile_size),
                        tile_stride=int(cfg.tile_stride),
                        config_id=str(cfg.config_id),
                    ),
                    out_size=int(out_side),
                    temp_dir=out_dir / ".v18_geometry",
                )
            )
        else:
            shared_jobs = tuple(
                None
                for _y in dense_tile_positions(aff.canvas_h, cfg.tile_size, cfg.tile_stride)
                for _x in dense_tile_positions(aff.canvas_w, cfg.tile_size, cfg.tile_stride)
            )
        if shared_jobs and shared_jobs[0] is not None:
            positions = (
                (int(job.tile_x), int(job.tile_y), job)
                for job in shared_jobs
                if job is not None
            )
        else:
            positions = (
                (int(x), int(y), None)
                for y in dense_tile_positions(aff.canvas_h, cfg.tile_size, cfg.tile_stride)
                for x in dense_tile_positions(aff.canvas_w, cfg.tile_size, cfg.tile_stride)
            )
        for x, y, shared_job in positions:
            tile_tag = f"{tag}_tile_{cfg.config_id}_x{int(x):04d}_y{int(y):04d}"
            canonical_tile_plan: Optional[RasterPlan] = None
            if view.shared_view is not None:
                canonical_tile_plan = build_forward_raster_plan(
                    mode="pta",
                    physical_view_id=shared_geometry.physical_view_name(
                        view.shared_view
                    ),
                    angle_deg=float(aff.angle_deg),
                    channel_token=str(channel_variant.format_token),
                    channel_kind=str(channel_variant.kind),
                    channel_count=int(channel_variant.channel_count),
                    channel_stride=int(channel_variant.stride),
                    channel_offsets=tuple(
                        int(value) for value in channel_variant.offsets
                    ),
                    channel_direction=str(channel_variant.order_name),
                    output_shape=(int(out_side), int(out_side)),
                    tile_size=int(cfg.tile_size),
                    tile_stride=int(cfg.tile_stride),
                    metadata={
                        "runtime_view_id": str(view.name),
                        "runtime_job_id": str(tile_tag),
                        "runtime_kind": "tile",
                        "tile_config_id": str(cfg.config_id),
                        "tile_x": int(x),
                        "tile_y": int(y),
                        **geometry_recipe_metadata(view.shared_view, shared_job if shared_job is not None else aff),
                        **projection_sampling_record(view.shared_view),
                        **radial_view_plan_metadata(view.shared_view),
                        **spherical_view_plan_metadata(view.shared_view),
                    },
                )
            tile_items.append(RenderTileItem(
                cfg=cfg,
                x=int(x),
                y=int(y),
                tile_tag=tile_tag,
                out_w=int(out_side),
                out_h=int(out_side),
                img_pattern=str(images_dir / f"{stem}_{tile_tag}_%04d{image_suffix}"),
                lbl_pattern=str(labels_dir / f"{stem}_{tile_tag}_%04d.txt"),
                overlay_path=(overlays_dir / f"{stem}_{tile_tag}_Overlay.mkv" if save_overlay else None),
                label_enabled=bool(label_enabled),
                channel_kind=str(channel_variant.kind),
                shared_job=shared_job,
                canonical_plan=canonical_tile_plan,
                publish_images=bool(publish_images),
                publish_labels=bool(publish_labels),
            ))

    normalized_eligible: Optional[Tuple[int, ...]]
    if eligible_frame_indices is None:
        normalized_eligible = None
        eligible_count = int(view.num_slices)
    else:
        normalized_eligible = tuple(sorted({int(x) for x in eligible_frame_indices}))
        if any(x < 0 or x >= int(view.num_slices) for x in normalized_eligible):
            raise ValueError(
                f"Eligible center index is outside view {view.name} bounds 0..{int(view.num_slices)-1}: "
                f"{normalized_eligible[:24]}"
            )
        eligible_count = len(normalized_eligible)
    normalized_encoded = tuple(int(x) for x in source_encoded_indices)
    if normalized_encoded and len(normalized_encoded) != int(view.num_slices):
        raise ValueError(
            f"Encoded-index mapping for {view.name} has {len(normalized_encoded)} entries, "
            f"expected {int(view.num_slices)}"
        )

    stats: Dict[str, object] = {
        "tag": tag,
        "view": view.name,
        "angle": float(aff.angle_deg),
        "frames": int(eligible_count),
        "view_frames": int(view.num_slices),
        "skipped_center_frames": int(view.num_slices) - int(eligible_count),
        "image_format": parse_output_image_format(image_format),
        "channel_format": str(channel_variant.format_token),
        "channel_kind": str(channel_variant.kind),
        "channel_count": int(channel_variant.channel_count),
        "channel_stride": int(channel_variant.stride),
        "channel_order": str(channel_variant.order_name),
        "channel_offsets": [int(x) for x in channel_variant.offsets],
        "full_output_size": [int(aff.out_w), int(aff.out_h)],
        "tiles": [
            {"tag": item.tile_tag, "tile_size": int(item.cfg.tile_size), "tile_stride": int(item.cfg.tile_stride), "x": int(item.x), "y": int(item.y), "output_size": [int(item.out_w), int(item.out_h)]}
            for item in tile_items
        ],
        "scheduler": "global_source_frame_and_output_frame_parallel",
        "forward_geometry": "XTA.geometry",
        "forward_backend": "runtime_selected_cpu_or_cuda_tta_canonical",
        "sampling_policy_digest": forward_sampling_policy().digest,
        "canonical_plan_digest": (
            None if canonical_plan is None else str(canonical_plan.digest)
        ),
        "canonical_tile_plan_digests": [
            str(item.canonical_plan.digest)
            for item in tile_items
            if item.canonical_plan is not None
        ],
        "label_enabled": bool(label_enabled),
        "publish_images": bool(publish_images),
        "publish_labels": bool(publish_labels),
    }
    return RenderPlan(
        view=view,
        aff=aff,
        tag=tag,
        img_pattern=img_pattern,
        lbl_pattern=lbl_pattern,
        overlay_path=overlay_path,
        label_enabled=bool(label_enabled),
        tile_layout=tuple(tile_items),
        stats=stats,
        channel_variant=channel_variant,
        eligible_frame_indices=normalized_eligible,
        source_encoded_indices=normalized_encoded,
        shared_aug_job=shared_aug_job,
        canonical_plan=canonical_plan,
        publish_images=bool(publish_images),
        publish_labels=bool(publish_labels),
    )


def render_plan_frame_indices(plan: RenderPlan) -> Sequence[int]:
    if plan.eligible_frame_indices is None:
        return range(int(plan.view.num_slices))
    return plan.eligible_frame_indices


def iter_render_source_frame_jobs_round_robin(plans: Sequence[RenderPlan]) -> Iterator[Tuple[int, int]]:
    """Yield only eligible source-frame jobs, frame-major across variants.

    Keep the original bounded-memory iteration pattern: eligibility is stored
    per plan rather than materializing every (plan, frame) job in a global map.
    """
    if not plans:
        return
    eligible_sets: List[Optional[set[int]]] = [
        None if plan.eligible_frame_indices is None else set(int(x) for x in plan.eligible_frame_indices)
        for plan in plans
    ]
    max_frames = max(int(plan.view.num_slices) for plan in plans)
    for frame_idx in range(max_frames):
        for plan_idx, plan in enumerate(plans):
            if frame_idx >= int(plan.view.num_slices):
                continue
            eligible = eligible_sets[plan_idx]
            if eligible is None or int(frame_idx) in eligible:
                yield int(plan_idx), int(frame_idx)


@runtime_telemetry_phase('pta.plan.geometry')
def build_plans_for_views(
    *,
    views: Sequence[ViewInfo],
    out_dir: Path,
    stem: str,
    tile_configs: Sequence[TileConfig],
    save_overlay: bool,
    imgsz: int,
    label_enabled: bool,
    image_format: str = "png",
    channel_variants: Sequence[ChannelVariant] = (DEFAULT_CHANNEL_VARIANT,),
    center_eligible_frame_indices: Optional[Sequence[int]] = None,
    transverse_source_encoded_indices: Sequence[int] = (),
    warnings: Optional[WarningLog] = None,
    publish_images: bool = True,
    publish_labels: bool = True,
) -> List[RenderPlan]:
    """Build plans with partial-center and continuity guards.

    ``center_eligible_frame_indices`` applies only to the native transverse view
    of an unforced Partially Labeled YOLO sequence.  Frames with no label file
    remain resident as image context but cannot become output centers.

    ``transverse_source_encoded_indices`` preserves the original numeric slice
    coordinates for custom C...S... stacks.  Requested indices inside the true
    [minimum, maximum] bounds must exist; requests outside those bounds retain
    edge clamping at the actual volume boundary.
    """
    plans: List[RenderPlan] = []
    normalized_base_centers = (
        None
        if center_eligible_frame_indices is None
        else tuple(sorted({int(x) for x in center_eligible_frame_indices}))
    )
    normalized_encoded = tuple(int(x) for x in transverse_source_encoded_indices)
    if normalized_encoded and any(
        normalized_encoded[i] >= normalized_encoded[i + 1]
        for i in range(len(normalized_encoded) - 1)
    ):
        raise ValueError(
            f"{stem}: transverse encoded image indices must be strictly increasing; "
            f"got {normalized_encoded[:24]}"
        )
    warned_discontinuity_formats: set[str] = set()

    for view in views:
        aff = build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            int(imgsz),
            shared_view=view.shared_view,
        )
        base_tag = view.display_name
        is_transverse = view.family == "transverse"
        if is_transverse and normalized_base_centers is not None:
            if any(x < 0 or x >= int(view.num_slices) for x in normalized_base_centers):
                raise ValueError(
                    f"{stem}: annotated center index is outside transverse bounds "
                    f"0..{int(view.num_slices)-1}: {normalized_base_centers[:24]}"
                )
        if is_transverse and normalized_encoded and len(normalized_encoded) != int(view.num_slices):
            raise ValueError(
                f"{stem}: transverse encoded-index mapping has {len(normalized_encoded)} entries, "
                f"expected {int(view.num_slices)}"
            )

        for channel_variant in channel_variants:
            tag = make_channel_tag(base_tag, channel_variant)
            eligible: Optional[Tuple[int, ...]] = (
                normalized_base_centers if is_transverse else None
            )
            source_indices_for_plan: Tuple[int, ...] = ()
            skipped_discontinuous: List[Tuple[int, Tuple[int, ...]]] = []

            if is_transverse and channel_variant.kind == "custom" and normalized_encoded:
                source_indices_for_plan = normalized_encoded
                candidate_centers: Sequence[int] = (
                    range(int(view.num_slices)) if eligible is None else eligible
                )
                continuous_centers: List[int] = []
                for frame_idx in candidate_centers:
                    positions, missing = encoded_channel_source_positions(
                        normalized_encoded,
                        int(frame_idx),
                        channel_variant.offsets,
                    )
                    if positions is None:
                        skipped_discontinuous.append((int(frame_idx), tuple(int(x) for x in missing)))
                    else:
                        continuous_centers.append(int(frame_idx))
                eligible = tuple(continuous_centers)

                warning_key = str(channel_variant.format_token)
                if skipped_discontinuous and warning_key not in warned_discontinuity_formats:
                    warned_discontinuity_formats.add(warning_key)
                    examples = [
                        f"center={int(normalized_encoded[frame_idx])}, missing={list(missing)}"
                        for frame_idx, missing in skipped_discontinuous[:8]
                    ]
                    msg = (
                        f"{stem} {channel_variant.format_token}: skipped "
                        f"{len(skipped_discontinuous)} annotated center slice(s) because required "
                        f"in-volume channel indices are missing across a discontinuity; "
                        f"examples: {'; '.join(examples)}. Requests beyond actual volume edges "
                        f"remain edge-clamped."
                    )
                    if warnings is not None:
                        warnings.add("channel_stack_discontinuity_centers_skipped", msg)
                    print(f"WARNING: {msg}", file=sys.stderr)

            plan = build_render_plan(
                view=view,
                aff=aff,
                tag=tag,
                out_dir=out_dir,
                stem=stem,
                tile_configs=tile_configs,
                save_overlay=bool(save_overlay),
                imgsz=int(imgsz),
                label_enabled=bool(label_enabled),
                image_format=image_format,
                channel_variant=channel_variant,
                eligible_frame_indices=eligible,
                source_encoded_indices=source_indices_for_plan,
                publish_images=bool(publish_images),
                publish_labels=bool(publish_labels),
            )
            base_count = (
                len(normalized_base_centers)
                if is_transverse and normalized_base_centers is not None
                else int(view.num_slices)
            )
            final_count = len(render_plan_frame_indices(plan))
            plan.stats["annotated_center_frames"] = int(base_count)
            plan.stats["skipped_unannotated_centers"] = (
                int(view.num_slices) - int(base_count)
                if is_transverse and normalized_base_centers is not None
                else 0
            )
            plan.stats["skipped_discontinuous_centers"] = int(len(skipped_discontinuous))
            plan.stats["eligible_center_frames"] = int(final_count)
            plans.append(plan)
    return plans
