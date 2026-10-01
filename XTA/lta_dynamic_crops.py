"""Bounded object-following LTA crop geometry, in native view coordinates.

Crops stay fixed for one tracker window.  A seed is never clipped to make a
crop fit: a large seed receives a larger crop or an explicit planning error.
The model transform is separate from native geometry and uses the same
1008-square input contract as the production tiled tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import operator
from typing import Sequence

import numpy as np

from .lta_propagation import LtaMaskSeed
from .lta_tiles import TilePlan


DYNAMIC_CROP_POLICY = "lta.dynamic-crops/1"
DYNAMIC_MODEL_SIDE = 1008


@dataclass(frozen=True)
class DynamicCropSettings:
    margin: int = 96
    guard: int = 24
    snap: int = 84
    max_scale: float = 3.0
    split_min_pixels: int = 100
    max_split_depth: int = 3
    max_split_children: int = 16
    max_crop_objects: int = 16

    def __post_init__(self):
        for name in ("margin", "guard", "snap", "split_min_pixels", "max_split_depth", "max_split_children", "max_crop_objects"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise TypeError(f"{name} must be an integer")
            value = operator.index(raw)
            if value < (1 if name in {"snap", "split_min_pixels", "max_split_children", "max_crop_objects"} else 0):
                raise ValueError(f"{name} is outside its allowed range")
            object.__setattr__(self, name, int(value))
        scale = float(self.max_scale)
        if not math.isfinite(scale) or not 1 <= scale <= 3:
            raise ValueError("dynamic crop max_scale must be finite and in [1,3]")
        object.__setattr__(self, "max_scale", scale)
        if self.guard >= DYNAMIC_MODEL_SIDE // 2:
            raise ValueError("dynamic crop guard must be below half the model side")
        if self.max_crop_objects > 16 or self.max_split_children > 16:
            raise ValueError("dynamic crop object and split-child budgets must not exceed 16")


@dataclass(frozen=True)
class NativeMask:
    """A tight bitmap with a global top-left corner; no dense view allocation."""

    top: int
    left: int
    mask: object = field(repr=False, compare=False)

    def __post_init__(self):
        top, left = operator.index(self.top), operator.index(self.left)
        mask = np.ascontiguousarray(np.asarray(self.mask) != 0, dtype=np.bool_).copy()
        if top < 0 or left < 0 or mask.ndim != 2 or not bool(mask.any()):
            raise ValueError("native masks require nonnegative coordinates and nonempty 2D foreground")
        mask.setflags(write=False)
        object.__setattr__(self, "top", int(top))
        object.__setattr__(self, "left", int(left))
        object.__setattr__(self, "mask", mask)

    @classmethod
    def from_crop(cls, mask, *, left=0, top=0):
        binary = np.asarray(mask, dtype=np.bool_)
        if binary.ndim != 2:
            raise ValueError("crop mask must be 2D")
        rows = np.flatnonzero(binary.any(axis=1))
        if not len(rows):
            return None
        columns = np.flatnonzero(binary.any(axis=0))
        return cls(int(top) + int(rows[0]), int(left) + int(columns[0]),
                   binary[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1])

    @property
    def xyxy(self):
        return self.left, self.top, self.left + self.mask.shape[1], self.top + self.mask.shape[0]

    def to_crop(self, crop: TilePlan):
        x0, y0, x1, y1 = self.xyxy
        cx0, cy0, cx1, cy1 = crop.xyxy
        if not (cx0 <= x0 < x1 <= cx1 and cy0 <= y0 < y1 <= cy1):
            raise ValueError("dynamic crop would clip its seed mask")
        output = np.zeros((crop.size, crop.size), dtype=np.bool_)
        output[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = self.mask
        return output


@dataclass(frozen=True)
class DynamicObject:
    seed: LtaMaskSeed
    native: NativeMask
    split_depth: int = 0

    def in_crop(self, crop: TilePlan, *, object_id: int):
        return replace(self.seed, object_id=object_id, mask=self.native.to_crop(crop))


@dataclass(frozen=True)
class DynamicCrop:
    tile: TilePlan
    objects: tuple[DynamicObject, ...]


def _containing_origin(low, high, *, side, extent, snap):
    """Snap only inside the interval that preserves every seed pixel."""
    lower = max(0, int(high) - side)
    upper = min(int(low), extent - side)
    if lower > upper:
        raise ValueError("native seed bounds cannot fit inside the crop")
    preferred = int(round(((low + high - side) / 2.0) / snap)) * snap
    return min(max(preferred, lower), upper)


def plan_dynamic_crops(objects: Sequence[DynamicObject], *, height: int, width: int,
                       settings: DynamicCropSettings = DynamicCropSettings()):
    """Deterministic clustering with full-seed containment and a hard side cap."""
    if min(height, width) < DYNAMIC_MODEL_SIDE:
        raise ValueError("dynamic LTA crops require both native frame dimensions >=1008")
    limit = min(int(DYNAMIC_MODEL_SIDE * settings.max_scale), height, width)
    ordered = tuple(sorted(objects, key=lambda item: item.seed.lineage.token))
    if not ordered:
        return ()
    if len({item.seed.lineage for item in ordered}) != len(ordered):
        raise ValueError("dynamic crop objects must have distinct lineages")
    if len({item.seed.frame_index for item in ordered}) != 1:
        raise ValueError("dynamic crop objects must share their prompt frame")
    boxes = []
    for item in ordered:
        x0, y0, x1, y1 = item.native.xyxy
        if x1 > width or y1 > height:
            raise ValueError("dynamic seed mask lies outside its native view")
        if max(x1 - x0, y1 - y0) > limit:
            raise ValueError("complete dynamic seed exceeds the bounded crop side; increase coverage using tiled LTA")
        boxes.append((max(0, x0 - settings.margin), max(0, y0 - settings.margin),
                      min(width, x1 + settings.margin), min(height, y1 + settings.margin)))
    clusters = [[index] for index in range(len(ordered))]
    # Merge only groups fitting the standard model-size native crop. Large
    # objects retain independent contexts rather than forcing unrelated peers
    # into a heavily downscaled image.
    while True:
        candidates = []
        for i, a in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                b = clusters[j]
                members = a + b
                if len(members) > settings.max_crop_objects:
                    continue
                box = (min(boxes[k][0] for k in members), min(boxes[k][1] for k in members),
                       max(boxes[k][2] for k in members), max(boxes[k][3] for k in members))
                if max(box[2] - box[0], box[3] - box[1]) <= DYNAMIC_MODEL_SIDE:
                    candidates.append(((box[2] - box[0]) * (box[3] - box[1]), i, j))
        if not candidates:
            break
        _, i, j = min(candidates)
        clusters[i].extend(clusters.pop(j))
    output = []
    for members in clusters:
        actual = [ordered[index].native.xyxy for index in members]
        x0, y0 = min(box[0] for box in actual), min(box[1] for box in actual)
        x1, y1 = max(box[2] for box in actual), max(box[3] for box in actual)
        padded = [boxes[index] for index in members]
        extent = max(max(box[2] for box in padded) - min(box[0] for box in padded),
                     max(box[3] for box in padded) - min(box[1] for box in padded))
        side = min(limit, max(DYNAMIC_MODEL_SIDE, int(math.ceil(extent / settings.snap)) * settings.snap))
        left = _containing_origin(x0, x1, side=side, extent=width, snap=settings.snap)
        top = _containing_origin(y0, y1, side=side, extent=height, snap=settings.snap)
        crop = TilePlan(left, top, side, width, height)
        values = tuple(ordered[index] for index in members)
        for item in values:
            item.native.to_crop(crop)
        output.append(DynamicCrop(crop, values))
    return tuple(output)


def touches_interior_guard(native: NativeMask, crop: TilePlan, *, guard: int):
    """Physical view edges cannot trigger crop escape patches."""
    if guard <= 0:
        return False
    x0, y0, x1, y1 = native.xyxy
    native_guard = max(1, int(math.ceil(guard * crop.size / DYNAMIC_MODEL_SIDE)))
    return bool((crop.left > 0 and x0 - crop.left < native_guard)
                or (crop.top > 0 and y0 - crop.top < native_guard)
                or (crop.left + crop.size < crop.source_width and crop.left + crop.size - x1 < native_guard)
                or (crop.top + crop.size < crop.source_height and crop.top + crop.size - y1 < native_guard))


def split_dynamic_object(item: DynamicObject, *, settings: DynamicCropSettings):
    """Keep every component; threshold controls splitting, never mask deletion."""
    if item.split_depth >= settings.max_split_depth:
        return (item,)
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(item.native.mask.astype(np.uint8), connectivity=8)
    pieces = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= settings.split_min_pixels]
    if len(pieces) < 2:
        return (item,)
    # Small islands remain with the largest daughter so splitting never erases
    # existing foreground. Their raw geometry is still in the worker packet.
    pieces.sort(key=lambda index: (-int(stats[index, cv2.CC_STAT_AREA]), int(stats[index, cv2.CC_STAT_TOP]),
                                   int(stats[index, cv2.CC_STAT_LEFT])))
    retained_pieces = pieces[:settings.max_split_children]
    remainder = (labels != 0) & ~np.isin(labels, retained_pieces)
    omitted_piece_count = len(pieces) - len(retained_pieces)
    output = []
    for ordinal, index in enumerate(retained_pieces):
        mask = labels == index
        if ordinal == 0:
            mask |= remainder
        native = NativeMask.from_crop(mask, left=item.native.left, top=item.native.top)
        digest = hashlib.sha256(np.packbits(native.mask).tobytes()).hexdigest()[:10]
        lineage = replace(item.seed.lineage, lineage_id=(
            f"{item.seed.lineage.lineage_id}.s{item.seed.frame_index}-{ordinal}-{digest}"))
        seed = replace(item.seed, lineage=lineage, mask=native.mask,
                       source_receipt={**item.seed.source_receipt, "parent_lineage": item.seed.lineage.token,
                                       "split_frame": item.seed.frame_index, "split_piece": ordinal})
        if omitted_piece_count:
            seed = replace(seed, source_receipt={**seed.source_receipt,
                "split_budget": {"action": "excess_components_preserved_with_largest_child",
                                 "aggregated_component_count": omitted_piece_count,
                                 "child_limit": settings.max_split_children}})
        output.append(DynamicObject(seed, native, item.split_depth + 1))
    return tuple(output)


def split_depth_budget_reached(item: DynamicObject, *, settings: DynamicCropSettings):
    if item.split_depth < settings.max_split_depth:
        return False
    import cv2
    count, _, stats, _ = cv2.connectedComponentsWithStats(item.native.mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= settings.split_min_pixels
               for index in range(1, count)) > 1


def resize_crop_frame(frame, *, side=DYNAMIC_MODEL_SIDE):
    import cv2
    array = np.asarray(frame)
    if array.shape[:2] == (side, side):
        return np.ascontiguousarray(array)
    return cv2.resize(array, (side, side), interpolation=cv2.INTER_AREA)


def resize_crop_mask(mask, *, side=DYNAMIC_MODEL_SIDE, restore=False):
    import cv2
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.shape == (side, side):
        return np.ascontiguousarray(binary)
    method = cv2.INTER_LINEAR if restore else cv2.INTER_AREA
    return cv2.resize(binary.astype(np.uint8) * 255, (side, side), interpolation=method) >= 128


__all__ = ("DYNAMIC_CROP_POLICY", "DYNAMIC_MODEL_SIDE", "DynamicCropSettings", "NativeMask",
           "DynamicObject", "DynamicCrop", "plan_dynamic_crops", "touches_interior_guard",
           "split_dynamic_object", "split_depth_budget_reached", "resize_crop_frame", "resize_crop_mask")
