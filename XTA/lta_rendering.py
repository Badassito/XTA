"""Shared native-frame caches and tile-window rendering for LTA.

Each physical view is cached once as immutable file-backed uint8 frames shared
by all workers.  Tile windows read native gray crops from that cache and detach
model-sized RGB frames, and settled tile chunks are ORed back into a mutable
physical-view volume.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


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


def _detach_gray_rgb_image(crop: np.ndarray) -> object:
    """Expand gray bytes directly into an independently owned RGB PIL image."""

    from PIL import Image

    # L-mode fromarray may borrow a contiguous gray crop. Conversion owns its
    # RGB storage, so the returned image never keeps the source mapping alive.
    # Exceptional aliases retain their original backing until they retire.
    return Image.fromarray(np.ascontiguousarray(crop), mode="L").convert("RGB")


def render_native_tile_window(
    cache_ref: LtaPhysicalViewCacheRef,
    *,
    frame_start: int,
    frame_stop: int,
    tile_xyxy: Sequence[int],
    model_side: int | None = None,
) -> list[object]:
    """Read native gray crops and detach model-sized RGB PIL frames.

    Scaled dynamic crops resize the gray frame before expanding its identical
    RGB channels. This retains the same INTER_AREA pixels without building an
    entire window of larger RGB frames that would immediately be discarded.
    Native crop coordinates and the shared cache remain unchanged.
    """

    start = int(frame_start)
    stop = int(frame_stop)
    if not 0 <= start < stop <= cache_ref.shape[0]:
        raise ValueError("requested cache window is outside the physical view")
    x0, y0, x1, y1 = (int(value) for value in tile_xyxy)
    if not 0 <= x0 < x1 <= cache_ref.shape[2] or not 0 <= y0 < y1 <= cache_ref.shape[1]:
        raise ValueError("tile_xyxy is outside the physical-view cache")
    if model_side is not None:
        if isinstance(model_side, (bool, np.bool_)):
            raise TypeError("model_side must be a positive integer")
        model_side = operator.index(model_side)
        if model_side < 1:
            raise ValueError("model_side must be a positive integer")
        from .lta_dynamic_crops import resize_crop_frame
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
            if model_side is not None:
                crop = resize_crop_frame(crop, side=model_side)
            # The cache crop is uint8 gray. PIL expands directly into detached
            # RGB storage without an intermediate three-channel NumPy copy.
            images.append(_detach_gray_rgb_image(crop))
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


__all__ = (
    "LtaPhysicalViewCacheRef",
    "render_native_tile_window",
    "reference_existing_physical_view_cache",
    "union_tile_chunk_into_view",
)
