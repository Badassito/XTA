"""Literal selected-native-plane traversal before bounded publication lookup."""
from collections.abc import Mapping
import numpy as np
from XTA.sam_interpolation import (_records_by_id, _publication_frame_addresses,
                                  SamInterpolationInfrastructureError)

def selected_sam_plane(bundle: object, receipt: Mapping[str, object], frame: int,
                       shape_yx: tuple[int, int], *, direction: int | None = None,
                       _records=None) -> np.ndarray:
    """Rebuild a selected native plane from owners, preserving shared voxels."""
    from XTA.sam_mask_reader import effective_candidate_mask
    groups, runs = ((_records_by_id(bundle.groups, "group_id"), _records_by_id(bundle.runs, "run_id"))
                    if _records is None else _records)
    plane = np.zeros(tuple(shape_yx), dtype=np.uint8)
    for run_id in receipt.get("selected_run_ids", ()):
        run = runs[str(run_id)]
        run_direction = run["direction"]
        sign = (1 if run_direction == "forward" else -1) if isinstance(run_direction, str) else int(run_direction)
        if direction is not None and sign != int(direction):
            continue
        group = groups[str(run["group_id"])]
        addresses = None
        if group.get("frame_addressing"):
            from XTA.sam_cyclic import address_for_unfolded_index, transform_crop_between_frame_addresses
            addresses = _publication_frame_addresses(bundle, group)
            native_shape = tuple(group["frame_addressing"]["native_shape_tyx"])
            if native_shape[1:] != tuple(shape_yx):
                raise SamInterpolationInfrastructureError("Cyclic native output shape differs from saved addresses")
        for stored_frame in run["expected_frames"]:
            stored_frame = int(stored_frame)
            address = addresses[stored_frame] if addresses is not None else None
            if int(address["native_index"] if address is not None else stored_frame) != int(frame):
                continue
            crop = effective_candidate_mask(bundle, str(run_id), stored_frame, receipt)
            if crop is None:
                continue
            crop = np.asarray(crop, dtype=bool)
            bbox = tuple(map(int, group["context_bbox_yx"]))
            if address is not None:
                target = address_for_unfolded_index(int(frame), native_shape[0],
                    period_degrees=group["frame_addressing"]["period_degrees"])
                crop, bbox = transform_crop_between_frame_addresses(crop, bbox, address, target, shape_yx[1])
            y0, x0, y1, x1 = bbox
            if crop.shape != (y1 - y0, x1 - x0):
                raise SamInterpolationInfrastructureError("Candidate mask shape differs from stored crop geometry")
            if not (0 <= y0 < y1 <= shape_yx[0] and 0 <= x0 < x1 <= shape_yx[1]):
                raise SamInterpolationInfrastructureError("Candidate crop is outside its native view canvas")
            plane[y0:y1, x0:x1] |= crop
    return plane
