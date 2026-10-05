"""Production-geometry rendering seam for LTA.

This CPU/reference path proves that canonical XTA view/TTA rendering and native
backprojection can surround a SAM mask without using YOLO.  CUDA tensor capture
can replace only :meth:`LtaRenderedView.render_frame_rgb` later while retaining
the same raster-plan and restoration contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from ._deps import cv2
from .assembly import project_view_volume_to_orthogonal_volume
from .config import resolve_channel_format
from .geometry import (
    AugJob,
    ViewInfo,
    build_aug_job_for_variant,
    build_view_frame_cache,
    physical_view_name,
    render_fullframe_frame_for_job,
)
from .unification.contracts import RasterPlan
from .unification.sampling import build_forward_raster_plan


@dataclass(frozen=True)
class LtaPhysicalViewCacheRef:
    """File-backed immutable native frames shared by all workers for one view."""

    path: Path
    shape: Tuple[int, int, int]
    dtype: str
    physical_view_id: str
    identity_sha256: str
    size_bytes: int
    mtime_ns: int
    # Optional compact SAM demand payload. Records are native frame index,
    # Y0/X0/Y1/X1, then the byte offset of that frame's cropped uint8 raster.
    # Ordinary LTA caches retain their full TYX storage and empty records.
    frame_crops: Tuple[Tuple[int, int, int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        path = Path(self.path).resolve(strict=True)
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("physical-view cache shape must contain three positive values")
        if str(self.dtype) != "uint8":
            raise ValueError("physical-view caches must use uint8")
        if path.stat().st_size != int(self.size_bytes):
            raise ValueError("physical-view cache size changed before publication")
        try:
            records = tuple(tuple(operator.index(value) for value in record) for record in self.frame_crops)
        except TypeError as error:
            raise ValueError('compact frame cache coordinates must be integers') from error
        if any(isinstance(value, (bool, np.bool_)) for record in self.frame_crops for value in record):
            raise ValueError('compact frame cache coordinates must not be booleans')
        expected_offset = 0
        previous_frame = -1
        for record in records:
            if len(record) != 6:
                raise ValueError('compact frame cache record must contain frame/bbox/offset')
            frame, y0, x0, y1, x1, offset = record
            if not (previous_frame < frame < shape[0] and 0 <= y0 < y1 <= shape[1]
                    and 0 <= x0 < x1 <= shape[2] and offset == expected_offset):
                raise ValueError('compact frame cache record is outside geometry or corrupt')
            expected_offset += (y1-y0) * (x1-x0)
            previous_frame = frame
        if records and expected_offset != int(self.size_bytes):
            raise ValueError('compact frame cache payload byte count differs from records')
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, 'frame_crops', records)

    def revalidate(self) -> None:
        stat = self.path.stat()
        if int(stat.st_size) != self.size_bytes or int(stat.st_mtime_ns) != self.mtime_ns:
            raise RuntimeError(f"physical-view cache changed after planning: {self.path}")

    def open(self, *, mode: str = "r") -> np.memmap:
        self.revalidate()
        return np.memmap(
            self.path,
            dtype=np.uint8,
            mode=str(mode),
            shape=(self.size_bytes,) if self.frame_crops else self.shape,
        )

    def payload(self) -> dict[str, object]:
        payload = {
            "path": str(self.path),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "physical_view_id": self.physical_view_id,
            "identity_sha256": self.identity_sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }
        if self.frame_crops:
            payload['frame_crops'] = [list(record) for record in self.frame_crops]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LtaPhysicalViewCacheRef":
        return cls(
            path=Path(str(payload["path"])),
            shape=tuple(int(value) for value in payload["shape"]),  # type: ignore[arg-type]
            dtype=str(payload["dtype"]),
            physical_view_id=str(payload["physical_view_id"]),
            identity_sha256=str(payload["identity_sha256"]),
            size_bytes=int(payload["size_bytes"]),
            mtime_ns=int(payload["mtime_ns"]),
            frame_crops=tuple(tuple(value for value in record)
                for record in payload.get('frame_crops', ())),
        )


def materialize_physical_view_cache(
    volume_u8: np.ndarray,
    view: ViewInfo,
    *,
    path: Path,
    workers: int = 1,
    source_identity: str = "",
) -> LtaPhysicalViewCacheRef:
    """Materialize exactly one file-backed native cache for a physical view."""

    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = build_view_frame_cache(
        volume_rgb=np.asarray(volume_u8),
        view=view,
        out_path=destination,
        desc=f"LTA {physical_view_name(view)} immutable frame cache",
        prefer_memory=False,
        workers=max(1, int(workers)),
    )
    flush = getattr(cache, "flush", None)
    if callable(flush):
        flush()
    stat = destination.stat()
    # Release our private references; retained worker views or tracebacks must
    # keep their mapping valid until they, too, are gone.
    del flush, cache
    identity_payload = {
        "source_identity": str(source_identity),
        "physical_view_id": physical_view_name(view),
        "shape": [int(view.num_slices), int(view.src_h), int(view.src_w)],
        "dtype": "uint8",
        "size_bytes": int(stat.st_size),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LtaPhysicalViewCacheRef(
        path=destination,
        shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
        dtype="uint8",
        physical_view_id=physical_view_name(view),
        identity_sha256=identity,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def reference_existing_physical_view_cache(
    path: Path,
    *,
    shape: Sequence[int],
    physical_view_id: str,
    source_identity: str,
) -> LtaPhysicalViewCacheRef:
    """Adopt an existing file-backed identity view without copying it."""

    resolved = Path(path).resolve(strict=True)
    resolved_shape = tuple(int(value) for value in shape)
    if len(resolved_shape) != 3 or any(value < 1 for value in resolved_shape):
        raise ValueError("existing physical-view cache shape must be positive TYX")
    expected_size = math.prod(resolved_shape)
    stat = resolved.stat()
    if int(stat.st_size) != int(expected_size):
        raise ValueError(
            f"existing uint8 cache size {stat.st_size} != expected {expected_size}"
        )
    identity = hashlib.sha256(
        json.dumps(
            {
                "source_identity": str(source_identity),
                "physical_view_id": str(physical_view_id),
                "shape": list(resolved_shape),
                "dtype": "uint8",
                "size_bytes": int(stat.st_size),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return LtaPhysicalViewCacheRef(
        path=resolved,
        shape=resolved_shape,
        dtype="uint8",
        physical_view_id=str(physical_view_id),
        identity_sha256=identity,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def render_native_tile_window(
    cache_ref: LtaPhysicalViewCacheRef,
    *,
    frame_start: int,
    frame_stop: int,
    tile_xyxy: Sequence[int],
) -> list[object]:
    """Read one native tile window from the shared cache as RGB PIL frames."""

    from PIL import Image

    start = int(frame_start)
    stop = int(frame_stop)
    if not 0 <= start < stop <= cache_ref.shape[0]:
        raise ValueError("requested cache window is outside the physical view")
    x0, y0, x1, y1 = (int(value) for value in tile_xyxy)
    if not 0 <= x0 < x1 <= cache_ref.shape[2] or not 0 <= y0 < y1 <= cache_ref.shape[1]:
        raise ValueError("tile_xyxy is outside the physical-view cache")
    import weakref
    from .runtime import close_memmap_array_without_flush
    cache = cache_ref.open(mode="r")
    mapping = weakref.ref(cache._mmap)
    records = {record[0]: record for record in cache_ref.frame_crops}
    class DetachedRgbFrames(list):
        source_cache_mapping_retired = False
    images = DetachedRgbFrames()
    frame = crop = None
    try:
        for index in range(start, stop):
            if records:
                if index not in records:
                    raise ValueError('SAM image cache does not contain a requested tracking frame')
                _, cy0, cx0, cy1, cx1, offset = records[index]
                if not (cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1):
                    raise ValueError('SAM tracker crop exceeds its immutable image demand')
                frame = cache[offset:offset+(cy1-cy0)*(cx1-cx0)].reshape(cy1-cy0, cx1-cx0)
                crop = frame[y0-cy0:y1-cy0, x0-cx0:x1-cx0]
            else:
                crop = cache[index, y0:y1, x0:x1]
            # A gray cache is always 2D here. implicit_rgb repeats into a new
            # RGB allocation, so PIL/model images never borrow this gray mmap.
            images.append(Image.fromarray(implicit_rgb(np.ascontiguousarray(crop)), mode='RGB'))
    finally:
        frame = crop = None
        close_memmap_array_without_flush(cache)
        cache = None
        # Inspect the shared mmap, not merely the outer NumPy wrapper. Error
        # traceback aliases remain valid and defer retirement naturally.
        images.source_cache_mapping_retired = mapping() is None
    return images


def union_tile_chunk_into_view(
    destination: np.ndarray,
    chunk: object,
    *,
    frame_start: int,
    tile_xyxy: Sequence[int],
) -> None:
    """OR one settled tile-local chunk into a mutable physical-view volume."""

    source = np.asarray(chunk, dtype=np.uint8)
    if source.ndim != 3:
        raise ValueError("tile chunk must have (frames,Y,X) shape")
    x0, y0, x1, y1 = (int(value) for value in tile_xyxy)
    if source.shape[1:] != (y1 - y0, x1 - x0):
        raise ValueError("tile chunk shape does not match tile_xyxy")
    start = int(frame_start)
    stop = start + int(source.shape[0])
    if not 0 <= start < stop <= int(destination.shape[0]):
        raise ValueError("tile chunk frame range is outside the destination view")
    if not 0 <= x0 < x1 <= int(destination.shape[2]):
        raise ValueError("tile chunk X range is outside the destination view")
    if not 0 <= y0 < y1 <= int(destination.shape[1]):
        raise ValueError("tile chunk Y range is outside the destination view")
    np.bitwise_or(
        destination[start:stop, y0:y1, x0:x1],
        source,
        out=destination[start:stop, y0:y1, x0:x1],
    )


def implicit_rgb(frame: object) -> np.ndarray:
    """Convert one canonical gray/RGB uint8 raster to contiguous RGB."""

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise ValueError(f"LTA canonical intensity frame must be uint8; got {array.dtype}")
    if array.ndim == 2:
        return np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
    if array.ndim == 3 and int(array.shape[2]) == 3:
        return np.ascontiguousarray(array)
    raise ValueError(f"LTA canonical intensity frame must be HxW or HxWx3; got {array.shape}")


@dataclass(frozen=True)
class LtaRenderedView:
    """One runtime view/angle using XTA's authoritative affine renderer."""

    volume_u8: np.ndarray
    view: ViewInfo
    aug_job: AugJob
    raster_plan: RasterPlan
    source_shape_tyx: Tuple[int, int, int]

    def __post_init__(self) -> None:
        volume = np.asarray(self.volume_u8)
        if volume.dtype != np.uint8 or volume.ndim != 3:
            raise ValueError("LTA source volume must be uint8 with (t,Y,X) shape")
        if tuple(int(value) for value in volume.shape) != tuple(self.source_shape_tyx):
            raise ValueError("LTA source volume shape does not match source_shape_tyx")
        if not isinstance(self.view, ViewInfo):
            raise TypeError("view must be a ViewInfo")
        if not isinstance(self.aug_job, AugJob):
            raise TypeError("aug_job must be an AugJob")
        if not isinstance(self.raster_plan, RasterPlan) or self.raster_plan.mode.value != "lta":
            raise TypeError("raster_plan must be an LTA RasterPlan")

    @property
    def frame_count(self) -> int:
        return int(self.view.num_slices)

    @property
    def model_shape_hw(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.raster_plan.output_shape)

    def render_frame_rgb(self, frame_index: int) -> np.ndarray:
        if int(frame_index) < 0 or int(frame_index) >= self.frame_count:
            raise IndexError(frame_index)
        frame = render_fullframe_frame_for_job(
            volume_rgb=self.volume_u8,
            view=self.view,
            job=self.aug_job,
            frame_idx=int(frame_index),
        )
        rgb = implicit_rgb(frame)
        if tuple(int(value) for value in rgb.shape[:2]) != self.model_shape_hw:
            raise RuntimeError(
                f"canonical renderer returned {rgb.shape[:2]}, expected {self.model_shape_hw}"
            )
        return rgb

    def restore_model_masks_to_view(
        self,
        masks_by_frame: Mapping[int, object],
    ) -> np.ndarray:
        """Invert only the in-plane model affine into one view-native volume."""

        restored = np.zeros(
            (self.frame_count, int(self.view.src_h), int(self.view.src_w)),
            dtype=np.uint8,
        )
        expected_model_shape = self.model_shape_hw
        for raw_index, raw_mask in masks_by_frame.items():
            frame_index = int(raw_index)
            if frame_index < 0 or frame_index >= self.frame_count:
                raise IndexError(frame_index)
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_ or tuple(int(value) for value in mask.shape) != expected_model_shape:
                raise ValueError(
                    f"SAM mask for frame {frame_index} must be bool {expected_model_shape}; "
                    f"got {mask.dtype}/{mask.shape}"
                )
            native = cv2.warpAffine(
                np.asarray(mask, dtype=np.uint8),
                np.asarray(self.aug_job.aff.M_out_to_src, dtype=np.float32),
                dsize=(int(self.view.src_w), int(self.view.src_h)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            restored[frame_index] = np.asarray(native != 0, dtype=np.uint8)
        return restored

    def project_view_masks_to_native(
        self,
        view_masks: np.ndarray,
        *,
        out_path: Path,
        workers: int = 1,
        prefer_memory: bool = True,
    ) -> np.ndarray:
        """Backproject one settled runtime-view mask volume to source TYX."""

        masks = np.asarray(view_masks)
        expected = (self.frame_count, int(self.view.src_h), int(self.view.src_w))
        if masks.dtype != np.uint8 or tuple(int(value) for value in masks.shape) != expected:
            raise ValueError(f"view mask volume must be uint8 {expected}; got {masks.dtype}/{masks.shape}")
        projected = project_view_volume_to_orthogonal_volume(
            masks,
            self.view,
            Path(out_path),
            desc=f"LTA native projection {self.view.name}",
            workers=max(1, int(workers)),
            prefer_memory=bool(prefer_memory),
            out_shape_tyx=self.source_shape_tyx,
            allow_transverse_passthrough=False,
        )
        if not isinstance(projected, np.ndarray):
            raise RuntimeError("LTA reference projection unexpectedly returned sink-only output")
        result = np.asarray(projected, dtype=np.uint8)
        if tuple(int(value) for value in result.shape) != self.source_shape_tyx:
            raise RuntimeError(
                f"LTA native projection returned {result.shape}, expected {self.source_shape_tyx}"
            )
        return result


def build_lta_rendered_view(
    volume_u8: np.ndarray,
    view: ViewInfo,
    *,
    temp_dir: Path,
    output_size: int = 1008,
) -> LtaRenderedView:
    """Bind one expanded LTA runtime view to canonical geometry and RGB policy."""

    from .unification.tta_manifest import radial_view_plan_metadata, spherical_view_plan_metadata
    from .unification.geometry_identity import geometry_recipe_metadata
    volume = np.asarray(volume_u8)
    if volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError("LTA source volume must be uint8 with (t,Y,X) shape")
    job = build_aug_job_for_variant(view, int(output_size), Path(temp_dir))
    fmt = resolve_channel_format("RGB")
    raster_plan = build_forward_raster_plan(
        mode="lta",
        physical_view_id=physical_view_name(view),
        angle_deg=float(job.angle_deg),
        channel_token=fmt.token,
        channel_kind=fmt.kind,
        channel_count=fmt.channel_count,
        channel_stride=fmt.stride,
        channel_offsets=fmt.offsets,
        channel_direction="ascending",
        output_shape=(int(output_size), int(output_size)),
        metadata={
            "runtime_view_id": str(view.name),
            "runtime_job_id": str(job.aug_id),
            "runtime_kind": "fullframe_sam",
            "channel_policy": "implicit_rgb_v1",
            "source_shape_tyx": [int(value) for value in volume.shape],
            **geometry_recipe_metadata(view, job.aff),
            **radial_view_plan_metadata(view),
            **spherical_view_plan_metadata(view),
        },
    )
    return LtaRenderedView(
        volume_u8=volume,
        view=view,
        aug_job=job,
        raster_plan=raster_plan,
        source_shape_tyx=tuple(int(value) for value in volume.shape),
    )


__all__ = (
    "LtaPhysicalViewCacheRef",
    "LtaRenderedView",
    "build_lta_rendered_view",
    "implicit_rgb",
    "materialize_physical_view_cache",
    "render_native_tile_window",
    "reference_existing_physical_view_cache",
    "union_tile_chunk_into_view",
)
