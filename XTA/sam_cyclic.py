"""Bounded view-native cyclic addresses; no source-grid or tracker state.

Azimuthal TTA covers a half turn. Crossing its frame boundary reverses the
working-canvas u axis. A full-turn address recipe is also explicit, never inferred
from an image shape. Stored masks retain their unfolded coordinates until selected
support is folded back to the original view.
"""
from __future__ import annotations

import operator
import hashlib
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

CYCLIC_FRAME_ADDRESSING_SCHEMA = "xta.sam_cyclic_view_frames/1"
CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA = "xta.sam_cyclic_extrapolation_frames/1"
_SOURCE_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()


def assert_cyclic_implementation_unchanged():
    if hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest() != IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM cyclic address implementation changed after loading")


@dataclass(frozen=True)
class CyclicFrameAddresses(MappingABC):
    """Constant-size frame mapping; callers materialize only bounded demands."""
    native_count: int
    evidence_count: int
    period_degrees: float
    addressing_schema: str = CYCLIC_FRAME_ADDRESSING_SCHEMA

    def __post_init__(self):
        native = _integer(self.native_count, "native frame count")
        evidence = _integer(self.evidence_count, "evidence frame count")
        maximum=(3*native-2 if self.addressing_schema==CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA else 2*native-1)
        if (self.addressing_schema not in {CYCLIC_FRAME_ADDRESSING_SCHEMA,CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA}
                or not 1 <= native <= evidence <= maximum):
            raise ValueError("Cyclic frame map exceeds its bounded alias prefix")
        address_for_unfolded_index(0, native, period_degrees=self.period_degrees)

    def __len__(self):
        return self.evidence_count

    def __iter__(self):
        return iter(range(self.evidence_count))

    def __getitem__(self, index):
        frame = _integer(index, "stored cyclic frame")
        if not 0 <= frame < self.evidence_count:
            raise KeyError(frame)
        return address_for_unfolded_index(frame, self.native_count, period_degrees=self.period_degrees)


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def address_for_unfolded_index(index: int, native_frame_count: int, *,
                               period_degrees: float = 180.0) -> Mapping[str, object]:
    """Address positive or negative neighbors outside a stored frame map."""
    assert_cyclic_implementation_unchanged()
    frame, count = _integer(index, "unfolded frame"), _integer(native_frame_count, "native frame count")
    if count < 1 or isinstance(period_degrees, bool) or period_degrees not in (180.0, 360.0):
        raise ValueError("Cyclic frame count must be positive and its period must be 180 or 360 degrees")
    cycle, native = divmod(frame, count)
    return MappingProxyType(dict(unfolded_index=frame, native_index=native, cycle_index=cycle,
                                 mirror_u=bool(period_degrees == 180.0 and cycle % 2)))


def mirror_bbox_yx(bbox_yx, canvas_width: int):
    width = _integer(canvas_width, "canvas width")
    y0, x0, y1, x1 = (_integer(value, "crop bound") for value in bbox_yx)
    if width < 1 or not (0 <= y0 < y1 and 0 <= x0 < x1 <= width):
        raise ValueError("Cyclic crop is outside its declared working canvas")
    return y0, width - x1, y1, width - x0


def transform_crop_between_frame_addresses(mask, bbox_yx, source_address, target_address,
                                           canvas_width: int):
    """Transform one exact cropped mask by the two unfolded phases' parity XOR."""
    assert_cyclic_implementation_unchanged()
    array = np.asarray(mask)
    bbox = tuple(_integer(value, "crop bound") for value in bbox_yx)
    mirrored_bbox = mirror_bbox_yx(bbox, canvas_width)
    if array.ndim != 2 or array.shape != (bbox[2] - bbox[0], bbox[3] - bbox[1]):
        raise ValueError("Cyclic cropped mask shape differs from its bbox")
    if int(source_address["native_index"]) != int(target_address["native_index"]):
        raise ValueError("Cyclic phase transform must describe the same original native frame")
    if bool(source_address["mirror_u"]) ^ bool(target_address["mirror_u"]):
        return array[:, ::-1], mirrored_bbox
    return array, bbox


def build_cyclic_frame_addressing(native_shape_tyx, alias_frames: int, *, period_degrees=180.0):
    shape = tuple(_integer(value, "native shape") for value in native_shape_tyx)
    aliases = _integer(alias_frames, "alias frame count")
    if len(shape) != 3 or any(value < 1 for value in shape) or not 0 <= aliases < shape[0]:
        raise ValueError("Cyclic aliases must be bounded below the positive native frame count")
    address_for_unfolded_index(0, shape[0], period_degrees=period_degrees)
    return MappingProxyType(dict(schema=CYCLIC_FRAME_ADDRESSING_SCHEMA,
        native_shape_tyx=shape, evidence_shape_tyx=(shape[0] + aliases, *shape[1:]),
        alias_frames=aliases, period_degrees=float(period_degrees)))


def validate_cyclic_frame_addressing(metadata, *, expected_frames=None):
    """Check shape, bounded closure, frame identity and mirror parity on reading."""
    assert_cyclic_implementation_unchanged()
    if not isinstance(metadata, Mapping) or metadata.get("schema") not in {CYCLIC_FRAME_ADDRESSING_SCHEMA,CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA}:
        raise ValueError("Unsupported cyclic SAM frame-address schema")
    native = tuple(_integer(value, "native shape") for value in metadata["native_shape_tyx"])
    evidence = tuple(_integer(value, "evidence shape") for value in metadata["evidence_shape_tyx"])
    alias = _integer(metadata["alias_frames"], "alias frame count")
    if len(native) != 3 or len(evidence) != 3 or any(value < 1 for value in native):
        raise ValueError("Malformed cyclic SAM native/evidence shape")
    maximum=2*(native[0]-1) if metadata['schema']==CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA else native[0]-1
    if not 0 <= alias <= maximum or evidence != (native[0] + alias, *native[1:]):
        raise ValueError("Cyclic SAM aliases violate their bounded native closure")
    period = metadata["period_degrees"]
    address_for_unfolded_index(0, native[0], period_degrees=period)
    if "addresses" not in metadata:
        if expected_frames is not None:
            raise ValueError("Cyclic SAM group must retain each declared frame address")
        return CyclicFrameAddresses(native[0], evidence[0], period,metadata['schema'])
    raw_addresses = metadata["addresses"]
    if not isinstance(raw_addresses, Mapping):
        raise ValueError("Cyclic SAM frame addresses must be a mapping")
    addresses = {}
    for key, value in raw_addresses.items():
        try:
            frame = int(key) if isinstance(key, str) and str(int(key)) == key else _integer(key, "frame key")
        except (TypeError, ValueError) as error:
            raise ValueError("Malformed cyclic SAM frame key") from error
        if frame in addresses or not 0 <= frame < evidence[0] or not isinstance(value, Mapping):
            raise ValueError("Cyclic SAM address is duplicated or outside its evidence canvas")
        expected = address_for_unfolded_index(frame, native[0], period_degrees=period)
        if set(value) != set(expected) or any(type(value[name]) is not type(expected[name])
                                              or value[name] != expected[name] for name in expected):
            raise ValueError("Cyclic SAM native frame or mirror closure is corrupted")
        addresses[frame] = MappingProxyType(dict(value))
    frames = range(evidence[0]) if expected_frames is None else expected_frames
    if set(addresses) != {_integer(value, "expected frame") for value in frames}:
        raise ValueError("Cyclic SAM address coverage differs from its declared frames")
    return MappingProxyType(addresses)


class CyclicObservationVolume:
    """Slice-only aliases of immutable original detector masks, never a dense copy."""
    def __init__(self, original, alias_frames: int, *, period_degrees=180.0):
        self.original = original
        self.frame_addressing = build_cyclic_frame_addressing(original.shape, alias_frames,
                                                             period_degrees=period_degrees)
        self.shape = self.frame_addressing["evidence_shape_tyx"]
        self.dtype = np.dtype(original.dtype)
        self.frame_addresses = CyclicFrameAddresses(self.frame_addressing["native_shape_tyx"][0],
                                                    self.shape[0], self.frame_addressing["period_degrees"])

    def __getitem__(self, index):
        frame = _integer(index, "cyclic observation frame")
        if not 0 <= frame < self.shape[0]:
            raise IndexError(frame)
        address = self.frame_addresses[frame]
        plane = np.asarray(self.original[int(address["native_index"])])
        if plane.shape != tuple(self.shape[1:]):
            raise ValueError("Original cyclic observation plane changed shape")
        return plane[:, ::-1] if address["mirror_u"] else plane

    def __array__(self, *args, **kwargs):
        raise RuntimeError("Cyclic SAM observations are slice-only; dense extended volumes are forbidden")


class ExtrapolationCyclicObservationVolume(CyclicObservationVolume):
    """Tail-scoped extra aliases preserve both inward seeds and full tail horizon."""
    def __init__(self,original,alias_frames,*,period_degrees=180.):
        shape=tuple(_integer(v,'native shape') for v in original.shape)
        aliases=_integer(alias_frames,'alias frame count')
        if len(shape)!=3 or any(v<1 for v in shape) or not 0<=aliases<=2*(shape[0]-1):
            raise ValueError('Extrapolation cyclic aliases exceed their separate bounded closure')
        self.original=original
        self.frame_addressing=MappingProxyType(dict(schema=CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA,
            native_shape_tyx=shape,evidence_shape_tyx=(shape[0]+aliases,*shape[1:]),
            alias_frames=aliases,period_degrees=float(period_degrees)))
        self.shape=self.frame_addressing['evidence_shape_tyx']
        self.dtype=np.dtype(original.dtype)
        self.frame_addresses=CyclicFrameAddresses(shape[0],self.shape[0],float(period_degrees),
                                                  CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA)


__all__ = ["CYCLIC_FRAME_ADDRESSING_SCHEMA", "IMPLEMENTATION_SHA256", "assert_cyclic_implementation_unchanged",
           "CyclicFrameAddresses", "CyclicObservationVolume", "address_for_unfolded_index",
           "build_cyclic_frame_addressing", "validate_cyclic_frame_addressing", "mirror_bbox_yx",
           "transform_crop_between_frame_addresses"]
