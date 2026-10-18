"""Literal selection/publication functions before the run157372 reuse repair.

Keep these independent of optimized functions: exact decisions, coverage and
cyclic alias pixels are compared against this retained implementation.
"""
from collections.abc import Mapping
import numpy as np
from XTA.sam_evidence import _group_shape, _plain
from XTA.sam_mask_reader import effective_raw_mask
from XTA.sam_interpolation import _records_by_id, SamInterpolationInfrastructureError


def measure_family_agreement(bundle, group_id, selected_run_ids, *, mask_filter=None, edge_ids=None):
    """Compare independently seeded edge families, preserving unavailable coverage."""
    group = bundle.groups[group_id]
    endpoints = {v["observation_id"]: v for v in group["endpoints"]}
    selected = [bundle.runs[key] for key in selected_run_ids]
    edges = [edge for edge in group.get("edges", ()) if edge_ids is None or edge["edge_id"] in edge_ids]
    forward_by_edge, reverse_by_edge, missing = {}, {}, []
    for edge in edges:
        source, target = endpoints[edge["source_id"]], endpoints[edge["target_id"]]
        interval = set(range(min(source["frame_index"], target["frame_index"]), max(source["frame_index"], target["frame_index"]) + 1))
        forward = [run for run in selected if edge["source_id"] in run.get("seed_ids", ()) and edge["target_id"] in run.get("held_out_ids", ())
                   and interval.issubset(run["observed_frames"])]
        reverse = [run for run in selected if edge["target_id"] in run.get("seed_ids", ()) and edge["source_id"] in run.get("held_out_ids", ())
                   and interval.issubset(run["observed_frames"])]
        forward_by_edge[edge["edge_id"]], reverse_by_edge[edge["edge_id"]] = forward, reverse
        if not forward or not reverse:
            missing.append(edge["edge_id"])
    slices = []
    intersection_total, union_total = 0, 0
    for frame in group["frame_indices"]:
        active = [edge for edge in edges if min(endpoints[edge["source_id"]]["frame_index"], endpoints[edge["target_id"]]["frame_index"]) < frame
                  < max(endpoints[edge["source_id"]]["frame_index"], endpoints[edge["target_id"]]["frame_index"])]
        if not active:
            continue
        a, b = np.zeros(_group_shape(group), bool), np.zeros(_group_shape(group), bool)
        missing_frame = []
        for edge in active:
            for collection, plane in ((forward_by_edge[edge["edge_id"]], a), (reverse_by_edge[edge["edge_id"]], b)):
                valid = [run for run in collection if frame in run["observed_frames"] and frame not in run.get("injected_frames", ())]
                if not valid:
                    missing_frame.append(edge["edge_id"])
                coverage=np.zeros(_group_shape(group),bool)
                for run in valid:
                    plane |= (bundle.raw_mask(run["run_id"],frame) if mask_filter is None else
                              effective_raw_mask(bundle, run["run_id"], frame, mask_filter))
                    if run.get("generation_mode")=="tiled":
                        coverage |= bundle.availability_mask(run["run_id"],frame)
                    else:
                        coverage[:]=True
                edge_write=f"edge_write:{edge['edge_id']}:{frame}"
                required=bundle.group_mask(group_id,edge_write if edge_write in group["mask_keys"] else f"write:{frame}")
                if np.any(required & ~coverage):
                    missing_frame.append(edge["edge_id"])
        domain = bundle.group_mask(group_id, f"write:{frame}")
        a &= domain
        b &= domain
        intersection, union = int(np.count_nonzero(a & b)), int(np.count_nonzero(a | b))
        intersection_total += intersection
        union_total += union
        slices.append(dict(frame_index=int(frame), status="unknown_missing_coverage" if missing_frame else "both_empty" if not union else "measured",
                           missing_edges=sorted(set(missing_frame)), intersection=intersection, union=union,
                           iou=intersection / union if union else None))
    complete = bool(slices) and not missing and all(v["status"] == "measured" for v in slices)
    return dict(status="measured" if complete else "unknown_or_empty_coverage", complete=complete,
                missing_independent_edges=missing, slices=slices,
                iou=intersection_total / union_total if union_total else None,
                minimum_slice_iou=min((v["iou"] for v in slices if v["iou"] is not None), default=None))

def _cyclic_group_geometry(bundle,group):
    """Saved bounded metadata controls comparisons; never infer a cyclic view."""
    from XTA.sam_cyclic import validate_cyclic_frame_addressing
    metadata=group.get("frame_addressing")
    scope=bundle.scope.get("frame_addressing")
    if metadata is None:
        if group.get("frame_addresses"):
            raise ValueError("Cyclic SAM group addresses require their retained closure header")
        if scope is None: return None
        # A scope header establishes the recipe, but the actual stored group
        # addresses must be retained so missing spatial/phase ownership is not
        # silently interpreted as success.
        raise ValueError("Cyclic SAM scope requires retained frame addressing for every group")
    addresses=validate_cyclic_frame_addressing(metadata,expected_frames=group["frame_indices"])
    if scope is not None:
        validate_cyclic_frame_addressing(scope)
        for key in ("native_shape_tyx","evidence_shape_tyx","alias_frames","period_degrees"):
            if _plain(metadata[key])!=_plain(scope[key]):
                raise ValueError("Cyclic SAM group closure differs from its saved scope")
    if group.get("frame_addresses") and _plain(group["frame_addresses"])!=_plain(addresses):
        raise ValueError("Cyclic SAM group frame addresses disagree")
    return dict(addresses=addresses,native_shape_tyx=tuple(metadata["native_shape_tyx"]),period_degrees=metadata["period_degrees"])

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
            from XTA.sam_cyclic import (address_for_unfolded_index, transform_crop_between_frame_addresses,
                                     validate_cyclic_frame_addressing)
            addresses = validate_cyclic_frame_addressing(group["frame_addressing"], expected_frames=group["frame_indices"])
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
