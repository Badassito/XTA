"""Literal pre-ROI observation discovery; preserve independently of optimized code."""
from __future__ import annotations
import hashlib
from types import MappingProxyType
import numpy as np
from scipy import ndimage as ndi
from XTA.sam_bridge_planning import SamObservation, _identity
from XTA.sam_observation_masks import (_PackedObservationMask, immutable_observation_mask, observation_mask_storage_bytes, observation_mask_storage_size)


def _discover_observations(volume, *, scope, labels, lineage, limits, inventory):
    result: list[SamObservation] = []
    by_frame: dict[int, list[int]] = {}
    shape = tuple(int(v) for v in volume.shape)  # type: ignore[attr-defined]
    labelled = labels is not None or np.dtype(volume.dtype).kind in "iu"  # type: ignore[attr-defined]
    charge = 0
    for frame in range(shape[0]):
        plane = np.asarray(volume[frame])  # type: ignore[index]
        canonical = np.asarray(labels[frame]) if labels is not None else plane  # type: ignore[index]
        frame_items: list[int] = []
        component_index = 0
        foreground = plane != 0
        values = np.unique(canonical[foreground]) if labelled else (0,)
        # One self-equal label reproduces the binary partition; NaN/NaT do not.
        # Multiple labels may touch; extrapolation must rediscover their union.
        reusable = (inventory is not None and len(values) <= 1
                    and (not len(values) or bool(values[0] == values[0])))
        address_metadata = getattr(volume, "frame_addressing", None)
        address = volume.frame_addresses[frame] if address_metadata else None
        key = (shape[1:], int(address['native_index']) if address else frame,
               bool(address['mirror_u']) if address else False)
        frame_digest = hashlib.sha256(np.packbits(foreground).tobytes()).digest() if reusable else None
        cached = inventory._take(key, frame_digest) if inventory is not None else None
        if not reusable:
            cached = None
        if cached is not None:
            inventory._counts['reused_frames'] += 1
        components = []
        remaining = sum(observation_mask_storage_bytes(crop) + 1024 for _, crop, _ in cached) if cached is not None else 0
        for value in values:
            if cached is None:
                mask = foreground & (canonical == value) if labelled else foreground
                if inventory is not None:
                    inventory._counts['label_calls'] += 1
                local, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
                candidates = ((index, slices, None) for index, slices in
                              enumerate(ndi.find_objects(local), start=1) if slices is not None)
            else:
                candidates = ((None, None, item) for item in cached)
            for local_index, slices, reused in candidates:
                if len(result) >= limits.max_observations:
                    return result, by_frame, "observation_inventory_limit"
                component_index += 1
                if reused is None:
                    sy, sx = slices
                    bbox = (int(sy.start), int(sx.start), int(sy.stop), int(sx.stop))
                else:
                    bbox = reused[0]
                pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                charge += (observation_mask_storage_size(pixels) if reused is None else
                           observation_mask_storage_bytes(reused[1])) + 1024
                if charge > limits.max_observation_bytes:
                    return result, by_frame, "observation_inventory_memory_limit"
                if reused is not None:
                    remaining -= observation_mask_storage_bytes(reused[1]) + 1024
                if inventory is not None:
                    inventory._trim(charge + remaining)
                if reused is None:
                    crop = immutable_observation_mask(local[slices] == local_index)
                    packed = crop.packed if isinstance(crop, _PackedObservationMask) else np.packbits(crop).tobytes()
                    digest = hashlib.sha256(packed).hexdigest()
                else:
                    _, crop, digest = reused
                    inventory._counts['reused_components'] += 1
                if reusable:
                    components.append((bbox, crop, digest))
                identifier = _identity("sam_obs", scope, frame, bbox, digest, int(value))
                original_identifier, native_frame, mirror_u = identifier, frame, False
                roots = dict(lineage or {})
                roots.setdefault("observation_source", "detector")
                roots.setdefault("scope_id", scope)
                if address_metadata:
                    native_frame, mirror_u = int(address["native_index"]), bool(address["mirror_u"])
                    original_bbox, original_mask = bbox, crop
                    if mirror_u:
                        from XTA.sam_cyclic import mirror_bbox_yx
                        original_bbox = mirror_bbox_yx(bbox, shape[2])
                        original_mask = crop[:, ::-1]
                    original_digest = (hashlib.sha256(np.packbits(original_mask).tobytes()).hexdigest()
                                       if mirror_u else digest)
                    original_identifier = _identity("sam_obs", scope, native_frame, original_bbox, original_digest, int(value))
                    identifier = (original_identifier if int(address["cycle_index"]) == 0 else
                                  _identity("sam_obs_alias", original_identifier, frame, address_metadata["schema"], mirror_u))
                    roots.update(original_observation_id=original_identifier, native_frame_index=native_frame,
                                 unfolded_frame_index=frame, cycle_index=int(address["cycle_index"]), mirror_u=mirror_u)
                frame_items.append(len(result))
                result.append(SamObservation(identifier, frame, int(value), component_index,
                                             bbox, crop, MappingProxyType(roots),
                                             original_observation_id=original_identifier,
                                             native_frame_index=native_frame, mirror_u=mirror_u))
        by_frame[frame] = frame_items
        if reusable:
            inventory._remember(key, frame_digest, components, charge)
    return result, by_frame, None
