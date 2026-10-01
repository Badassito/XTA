"""Frozen, observation-only geometry for paired native SAM crop strategies.

Full native detector components choose families and crops. Fixed overlapping
tiles see the same total pixels as the endpoint-union crop. Independent original
seed intersections condition every tile; predictions never condition other tiles.
This is a paired follow-up on seen data, not a blind validation protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

SCHEMA = "xta.sam_native_crop_strategy_plan/1"
TILE_MAX = 1008
HALO = 128
STRIDE = TILE_MAX-2*HALO


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    observation_id: str
    frame_native: int
    component_index: int
    bbox_yx: tuple[int, int, int, int]
    mask_crop: np.ndarray = field(compare=False, repr=False)

    def descriptor(self, source_shape_yx):
        y0, x0, y1, x1 = self.bbox_yx
        height, width = source_shape_yx
        return dict(observation_id=self.observation_id, frame_native=self.frame_native,
            component_index=self.component_index, bbox_yx=list(self.bbox_yx),
            foreground=int(self.mask_crop.sum()),
            native_source_boundary_sides=[side for side, hit in (("top", y0 == 0), ("left", x0 == 0), ("bottom", y1 == height), ("right", x1 == width)) if hit],
            mask_sha256=hashlib.sha256(np.packbits(self.mask_crop).tobytes()).hexdigest(),
            identity_scope="Full native slice-connected component; no biological cross-frame identity asserted")


def extract_components(mask, frame_native):
    foreground = np.asarray(mask) != 0
    if foreground.ndim != 2:
        raise ValueError("Endpoint masks must be native two-dimensional planes")
    labels, _ = ndi.label(foreground, structure=np.ones((3, 3), bool))
    result = []
    for index, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        y, x = slices
        crop = np.ascontiguousarray(labels[slices] == index)
        crop.setflags(write=False)
        result.append(Observation(f"native_f{frame_native:04d}_component{index:04d}", int(frame_native), index,
            (y.start, x.start, y.stop, x.stop), crop))
    return result


def mask_in_crop(observation, crop_yx):
    """Exact native full-component intersection; no resize or seed-radius filter."""
    y0, x0, y1, x1 = map(int, crop_yx)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid crop")
    output = np.zeros((y1-y0, x1-x0), bool)
    a0, b0, a1, b1 = observation.bbox_yx
    cy0, cx0, cy1, cx1 = max(a0, y0), max(b0, x0), min(a1, y1), min(b1, x1)
    if cy0 < cy1 and cx0 < cx1:
        output[cy0-y0:cy1-y0, cx0-x0:cx1-x0] = observation.mask_crop[cy0-a0:cy1-a0, cx0-b0:cx1-b0]
    return output


def _intersects_roi(observation, roi_xyxy):
    x0, y0, x1, y1 = map(int, roi_xyxy)
    a0, b0, a1, b1 = observation.bbox_yx
    cy0, cx0, cy1, cx1 = max(a0, y0), max(b0, x0), min(a1, y1), min(b1, x1)
    return cy0 < cy1 and cx0 < cx1 and bool(observation.mask_crop[cy0-a0:cy1-a0, cx0-b0:cx1-b0].any())


def axis_windows(start, stop, *, maximum=TILE_MAX, halo=HALO):
    """Minimal fixed cover with stride <= maximum-2*halo and midpoint owners."""
    start, stop, maximum, halo = map(int, (start, stop, maximum, halo))
    length = stop-start
    if length <= 0 or maximum <= 0 or halo < 0 or maximum <= 2*halo:
        raise ValueError("Invalid tile axis bounds")
    width = min(length, maximum)
    stride = maximum-2*halo
    last = stop-width
    starts = list(range(start, last+1, stride))
    if starts[-1] != last:
        starts.append(last)
    seams = [(a+width+b)//2 for a, b in zip(starts, starts[1:])]
    boundaries = [start, *seams, stop]
    return [dict(crop=[position, position+width], ownership=[boundaries[index], boundaries[index+1]])
            for index, position in enumerate(starts)]


def tile_plan(whole_crop_yx, *, maximum=TILE_MAX, halo=HALO):
    y0, x0, y1, x1 = map(int, whole_crop_yx)
    ys, xs = axis_windows(y0, y1, maximum=maximum, halo=halo), axis_windows(x0, x1, maximum=maximum, halo=halo)
    tiles = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            tiles.append(dict(tile_id=f"tile_r{row:02d}_c{column:02d}",
                crop_bbox_yx=[y["crop"][0], x["crop"][0], y["crop"][1], x["crop"][1]],
                ownership_bbox_yx=[y["ownership"][0], x["ownership"][0], y["ownership"][1], x["ownership"][1]]))
    return dict(mode="identical_crop_control" if len(tiles) == 1 else "independent_overlapping_tiles",
        max_tile_native_side=maximum, minimum_interior_halo_px=halo, maximum_stride_px=maximum-2*halo,
        ownership_rule="Fixed adjacent-overlap midpoints; outer owners extend to shared whole-crop edge",
        additional_context_outside_whole_crop_pixels=0, tiles=tiles)


def _geometric_edges(source, target, source_shape_yx, growth):
    height, width = source_shape_yx
    padding = int(math.ceil(growth))+1
    edges = []
    for observation in source:
        y0, x0, y1, x1 = observation.bbox_yx
        crop = (max(0, y0-padding), max(0, x0-padding), min(height, y1+padding), min(width, x1+padding))
        local = mask_in_crop(observation, crop)
        distance = ndi.distance_transform_edt(~local)
        for endpoint in target:
            a0, b0, a1, b1 = endpoint.bbox_yx
            cy0, cx0, cy1, cx1 = max(a0, crop[0]), max(b0, crop[1]), min(a1, crop[2]), min(b1, crop[3])
            if cy0 >= cy1 or cx0 >= cx1:
                continue
            support = endpoint.mask_crop[cy0-a0:cy1-a0, cx0-b0:cx1-b0]
            distances = distance[cy0-crop[0]:cy1-crop[0], cx0-crop[1]:cx1-crop[1]]
            if np.any(support & (distances <= growth+1e-9)):
                edges.append(dict(source_id=observation.observation_id, target_id=endpoint.observation_id,
                    geometry="Original silhouette overlap or fixed projection-cone growth", minimum_native_distance=float(distances[support].min())))
    return edges


def build_strategy_plan(endpoint_masks: Mapping[int, np.ndarray], reviews: Sequence[dict], *, context_margin=24,
                        search_angle_degrees=30., tile_max=TILE_MAX, tile_halo=HALO):
    frames = sorted(map(int, endpoint_masks))
    if len(frames) != 2 or frames[0] >= frames[1]:
        raise ValueError("Exactly two original endpoint frames are required")
    shape = tuple(map(int, np.asarray(endpoint_masks[frames[0]]).shape))
    if len(shape) != 2 or tuple(np.asarray(endpoint_masks[frames[1]]).shape) != shape:
        raise ValueError("Native endpoint shapes differ")
    if context_margin < 0 or not 0 <= search_angle_degrees < 90:
        raise ValueError("Invalid context/projection settings")
    source, target = (extract_components(endpoint_masks[frame], frame) for frame in frames)
    observations = {v.observation_id: v for v in source+target}
    growth = math.tan(math.radians(search_angle_degrees))*(frames[1]-frames[0])
    edges = _geometric_edges(source, target, shape, growth)
    parents = {key: key for key in observations}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for edge in edges:
        a, b = root(edge["source_id"]), root(edge["target_id"])
        parents[max(a, b)] = min(a, b)
    components = {}
    for key in observations:
        components.setdefault(root(key), []).append(key)
    families, review_records = [], []
    for nodes in components.values():
        own_edges = [edge for edge in edges if edge["source_id"] in nodes and edge["target_id"] in nodes]
        matches = [review for review in reviews if any(_intersects_roi(observations[key], review["global_roi_xyxy"]) for key in nodes)]
        if not matches or not own_edges:
            continue
        family_id = "family_"+_hash(sorted(nodes))[:16]
        y0, x0 = min(observations[key].bbox_yx[0] for key in nodes), min(observations[key].bbox_yx[1] for key in nodes)
        y1, x1 = max(observations[key].bbox_yx[2] for key in nodes), max(observations[key].bbox_yx[3] for key in nodes)
        crop = [max(0, y0-context_margin), max(0, x0-context_margin), min(shape[0], y1+context_margin), min(shape[1], x1+context_margin)]
        grid = tile_plan(crop, maximum=tile_max, halo=tile_halo)
        runs = []
        for key in sorted(nodes):
            observation = observations[key]
            direction = "forward" if observation.frame_native == frames[0] else "backward"
            opposite = sorted({edge["target_id"] if direction == "forward" else edge["source_id"] for edge in own_edges
                               if edge["source_id"] == key or edge["target_id"] == key})
            tile_seeds = []
            for tile in grid["tiles"]:
                count = int(mask_in_crop(observation, tile["crop_bbox_yx"]).sum())
                tile_seeds.append(dict(tile_id=tile["tile_id"], seed_foreground=count,
                    status="independently_original_seeded" if count else "unavailable_empty_original_seed",
                    ownership_bbox_yx=tile["ownership_bbox_yx"]))
            covered = sum((s["ownership_bbox_yx"][2]-s["ownership_bbox_yx"][0])*(s["ownership_bbox_yx"][3]-s["ownership_bbox_yx"][1]) for s in tile_seeds if s["seed_foreground"])
            runs.append(dict(run_id=family_id+"_"+key+"_"+direction, seed_observation_id=key,
                seed_frame_native=observation.frame_native, direction=direction, held_out_observation_ids=opposite,
                frame_start_native=frames[0], frame_stop_native=frames[1]+1,
                tile_seeds=tile_seeds, available_owner_pixels=covered,
                available_owner_fraction=covered/((crop[2]-crop[0])*(crop[3]-crop[1]))))
        families.append(dict(family_id=family_id, observation_ids=sorted(nodes), edges=own_edges,
            review_case_ids=[review["case"] for review in matches],
            priority=0 if any(review["case"] == "region01_case02" for review in matches) else 1,
            whole_crop_bbox_yx=crop, endpoint_union_bbox_yx=[y0, x0, y1, x1],
            native_source_edge_censored=any(observations[key].descriptor(shape)["native_source_boundary_sides"] for key in nodes),
            complete_visible_original_silhouettes=True, completeness_beyond_source_extent="unknown_if_source_boundary_touched",
            tile_strategy=grid, runs=runs))
    families.sort(key=lambda family: (family["priority"], family["family_id"]))
    for review in reviews:
        matches = [family["family_id"] for family in families if review["case"] in family["review_case_ids"]]
        review_records.append({**dict(review), "family_ids":matches, "status":"matched_full_native_family" if matches else "unresolved_no_original_endpoint_connection"})
    plan = dict(schema=SCHEMA, source_shape_yx=list(shape), endpoint_frames_native=frames,
        source_observations=[observation.descriptor(shape) for observation in observations.values()],
        observations_scope="Full native endpoint unions; all slice-connected components retained without review-ROI silhouette clipping",
        families=families, reviews=review_records,
        settings=dict(context_margin_native_px=context_margin, projection_search_angle_degrees=search_angle_degrees,
            projection_growth_native_px=growth, tile_max_native_side=tile_max, tile_halo_native_px=tile_halo,
            interpolation_distance=15, interpolation_min_radius=3., interpolation_walk_back=1,
            walk_back_availability="No adjacent detector observation frames provided; no generated walk-back seeds",
            boundary_f1_tolerance_native_px=2.),
        protocol=dict(kind="Paired seen-data follow-up; no blind-validation claim", labels_used_for_planning=False,
            seed_conditioning="Each original native component independently, then exact intersection with its tile; predictions never seed another tile",
            empty_seed_rule="Skip inference; owner core unavailable/unknown, never successful predicted background; full-domain scores count missing coverage as absent prediction",
            seam_assembly="Exclusive fixed midpoint ownership, assembled in global native coordinates before global component radius/topology checks",
            raw_halo_evidence="Retain complete raw masks for every tile, including discarded overlap, for global leakage diagnostics",
            shared_context="Tile footprints are entirely inside the same fixed whole-family crop; no extra image pixels outside it",
            controls="When whole crop fits1008 on both axes, tiled strategy is the exact same crop and upscale",
            cache_fairness="Reset feature cache between strategies and timing repeats; reuse within one strategy only",
            boundary_f1="Compute full native contours before ROI restriction; allow matches within2px anywhere in full image, including outside review ROI"))
    plan["plan_sha256"] = _hash(plan)
    return plan, observations


def assemble_owned_tiles(plan, tile_masks):
    """Assemble raw owner support; unavailable cores remain explicitly unknown."""
    y0, x0, y1, x1 = plan["whole_crop_bbox_yx"]
    output = np.zeros((y1-y0, x1-x0), bool)
    available = np.zeros(output.shape, bool)
    for tile in plan["tile_strategy"]["tiles"]:
        if tile["tile_id"] not in tile_masks:
            continue
        a0, b0, a1, b1 = tile["crop_bbox_yx"]
        mask = np.asarray(tile_masks[tile["tile_id"]], bool)
        if mask.shape != (a1-a0, b1-b0):
            raise ValueError("Raw tile mask shape differs from declared native footprint")
        c0, d0, c1, d1 = tile["ownership_bbox_yx"]
        output[c0-y0:c1-y0, d0-x0:d1-x0] = mask[c0-a0:c1-a0, d0-b0:d1-b0]
        available[c0-y0:c1-y0, d0-x0:d1-x0] = True
    return output, available


def boundary_f1(prediction, truth, roi_xyxy, *, tolerance_px=2.):
    """Full-native contour matching; ROI chooses scored points, never contours."""
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Boundary metrics require matched full-native planes")
    if not np.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("Invalid boundary tolerance")
    x0, y0, x1, y1 = map(int, roi_xyxy)
    height, width = truth.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Review ROI lies outside the native image")
    structure = np.ones((3, 3), bool)
    p = prediction & ~ndi.binary_erosion(prediction, structure=structure, border_value=0)
    g = truth & ~ndi.binary_erosion(truth, structure=structure, border_value=0)
    roi = np.s_[y0:y1, x0:x1]
    pc, gc = int(p[roi].sum()), int(g[roi].sum())
    pm = int((p & (ndi.distance_transform_edt(~g) <= tolerance_px))[roi].sum()) if g.any() else 0
    gm = int((g & (ndi.distance_transform_edt(~p) <= tolerance_px))[roi].sum()) if p.any() else 0
    precision, recall = pm/pc if pc else None, gm/gc if gc else None
    value = None if precision is None or recall is None else 0. if precision+recall == 0 else 2*precision*recall/(precision+recall)
    return dict(tolerance_native_px=float(tolerance_px), prediction_boundary_points=pc, truth_boundary_points=gc,
        matched_prediction_points=pm, matched_truth_points=gm, precision=precision, recall=recall, f1=value,
        status="undefined_zero_boundary_count" if value is None else "measured",
        contour_domain="Full native image before ROI restriction; matches may occur outside ROI")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--prior-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.endpoints.read_text("utf-8"))
    endpoints = {int(frame):np.load(args.endpoints.parent/record["union_file"], mmap_mode="r") for frame, record in metadata["frames"].items() if int(frame) in (54, 68)}
    reviews = []
    for region in sorted(args.prior_reviews.glob("region*")):
        inventory = json.loads((region/"cases.json").read_text("utf-8"))
        x, y, _, _ = inventory["source_crop_xyxy"]
        for case in inventory["cases"]:
            a, b, c, d = case["region_xyxy"]
            reviews.append(dict(case=case["id"], global_roi_xyxy=[a+x, b+y, c+x, d+y], original_region=region.name,
                evaluation_frame_native=inventory["evaluation_local_frame"]+inventory["input_native_frames"][0]))
    plan, observations = build_strategy_plan(endpoints, reviews)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = args.output/"original_native_component_seeds"
    seeds.mkdir(exist_ok=True)
    for observation in observations.values():
        path = seeds/(observation.observation_id+".npz")
        np.savez_compressed(path, mask=observation.mask_crop, bbox_yx=np.asarray(observation.bbox_yx), frame_native=observation.frame_native)
    plan["source_images"] = metadata["source_images"]
    plan["source_shape_tyx"] = metadata["source_shape_tyx"]
    plan["source_frame_start"] = metadata["source_frame_start"]
    plan["endpoint_metadata"] = str(args.endpoints)
    plan["endpoint_metadata_sha256"] = hashlib.sha256(args.endpoints.read_bytes()).hexdigest()
    plan["seed_directory"] = str(seeds)
    plan["geometry_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan["plan_sha256"] = _hash({key:value for key, value in plan.items() if key != "plan_sha256"})
    (args.output/"strategy_plan.json").write_text(json.dumps(plan, indent=2), "utf-8")
    (args.output/"protocol.json").write_text(json.dumps(dict(schema=SCHEMA, plan_sha256=plan["plan_sha256"], protocol=plan["protocol"], settings=plan["settings"], geometry_source_sha256=plan["geometry_source_sha256"]), indent=2), "utf-8")
    print(json.dumps(dict(families=len(plan["families"]), runs=sum(len(family["runs"]) for family in plan["families"]),
        tile_sessions=sum(sum(bool(seed["seed_foreground"]) for seed in run["tile_seeds"]) for family in plan["families"] for run in family["runs"]),
        families_summary=[dict(id=family["family_id"], reviews=family["review_case_ids"], crop=family["whole_crop_bbox_yx"], tiles=len(family["tile_strategy"]["tiles"]), runs=len(family["runs"])) for family in plan["families"]]), indent=2))


if __name__ == "__main__":
    main()
