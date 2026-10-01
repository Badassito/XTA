"""Offline native assembly and paired scoring of frozen SAM crop experiments.

Raw masks remain research predictions. Radius filtering occurs after native tile
assembly for EACH ORIGINAL SEED RUN, before independent hypotheses are united.
No output of this tool implies production-v2 acceptance.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sam_crop_strategy_geometry import Observation, assemble_owned_tiles, boundary_f1, mask_in_crop
from XTA.sam_filtering import filter_sam_components

METHODS = ["sdf", *[f"sam_{strategy}_{stage}_{direction}" for strategy in ("whole", "tiles")
                    for stage in ("raw", "radius3") for direction in ("forward", "backward", "union")]]


class RawRunReader:
    """Bounded compressed-payload cache; unpack only a requested native plane."""
    def __init__(self, directory, plan, *, max_cache_bytes=64*1024**2):
        self.directory = Path(directory)
        self.plan = plan
        self.rows = json.loads((self.directory/"raw_index.json").read_text("utf-8"))
        self.packet = np.load(self.directory/"raw_masks.npz", allow_pickle=False)
        self.families = {family["family_id"]: family for family in plan["families"]}
        self.by_original = {}
        for row in self.rows:
            family = self.families[row["family_id"]]
            expected = (dict(crop_bbox_yx=family["whole_crop_bbox_yx"], ownership_bbox_yx=family["whole_crop_bbox_yx"])
                        if row["tile_id"] == "whole" else next(tile for tile in family["tile_strategy"]["tiles"] if tile["tile_id"] == row["tile_id"]))
            if row["crop_bbox_yx"] != expected["crop_bbox_yx"] or row["ownership_bbox_yx"] != expected["ownership_bbox_yx"]:
                raise ValueError("Retained raw coordinates differ from the frozen geometry")
            self.by_original.setdefault(row["original_run_id"], []).append(row)
        self.cache, self.cache_bytes, self.max_cache_bytes = OrderedDict(), 0, max_cache_bytes

    def close(self):
        self.packet.close()
        self.cache.clear()

    def _packed(self, row):
        key = row["packed_key"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        value = self.packet[key]
        if hashlib.sha256(value.tobytes()).hexdigest() != row["binary_sha256"]:
            raise ValueError("Retained raw binary payload changed")
        if value.nbytes <= self.max_cache_bytes:
            while self.cache and self.cache_bytes+value.nbytes > self.max_cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.nbytes
            self.cache[key] = value
            self.cache_bytes += value.nbytes
        return value

    def tile_plane(self, row, frame):
        if frame not in row["native_frames"]:
            raise ValueError("Raw prediction is missing a declared native frame")
        position = row["native_frames"].index(frame)
        height, width = row["shape_yx"]
        packed = self._packed(row)
        return np.unpackbits(packed[position], count=height*width, bitorder="little").reshape(height, width).astype(bool)

    def assemble_frame(self, original_run_id, frame):
        rows = self.by_original.get(original_run_id, [])
        if not rows:
            raise ValueError("Original seed run has no retained inference results")
        family = self.families[rows[0]["family_id"]]
        if rows[0]["tile_id"] == "whole":
            if len(rows) != 1:
                raise ValueError("Whole-crop run has multiple owners")
            raw = self.tile_plane(rows[0], frame)
            return raw, np.ones(raw.shape, bool)
        masks = {row["tile_id"]: self.tile_plane(row, frame) for row in rows if frame in row["native_frames"]}
        return assemble_owned_tiles(family, masks)

    def assemble_original_run(self, original_run_id, *, include_halos=False):
        rows = self.by_original.get(original_run_id, [])
        if not rows:
            raise ValueError("Original seed run has no retained inference results")
        family = self.families[rows[0]["family_id"]]
        frames = list(range(self.plan["endpoint_frames_native"][0], self.plan["endpoint_frames_native"][1]+1))
        y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
        raw = np.zeros((len(frames), y1-y0, x1-x0), bool)
        available = np.zeros(raw.shape, bool)
        for index, frame in enumerate(frames):
            raw[index], available[index] = self.assemble_frame(original_run_id, frame)
        halo_records = []
        if include_halos:
            for row in rows:
                halo_records.append({**row, "masks":np.stack([self.tile_plane(row, frame) for frame in frames]), "native_frames":frames})
        return frames, raw, available, halo_records


def load_observations(plan):
    result = {}
    for descriptor in plan["source_observations"]:
        file = Path(plan["seed_directory"])/(descriptor["observation_id"]+".npz")
        with np.load(file, allow_pickle=False) as saved:
            mask, bbox, frame = saved["mask"].copy(), tuple(map(int, saved["bbox_yx"])), int(saved["frame_native"])
        if (list(bbox) != descriptor["bbox_yx"] or frame != descriptor["frame_native"]
                or hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() != descriptor["mask_sha256"]):
            raise ValueError("Frozen original observation changed")
        result[descriptor["observation_id"]] = Observation(descriptor["observation_id"], frame, descriptor["component_index"], bbox, mask)
    return result


def load_truth(path, source_shape_yx):
    height, width = source_shape_yx
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for line in Path(path).read_text("utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        points = list(map(float, fields[1:]))
        draw.polygon([(points[index]*width, points[index+1]*height) for index in range(0, len(points), 2)], fill=1)
    return np.asarray(image, bool)


def filter_original_run_planes(raw, *, min_radius=3.):
    """Keep independent seed hypotheses separate until component filtering."""
    raw = np.asarray(raw, bool)
    if raw.ndim != 3:
        raise ValueError("Original run must have frame,y,x shape")
    filtered = np.empty(raw.shape, bool)
    removed = 0
    for index in range(len(raw)):
        filtered[index], diagnostic = filter_sam_components(raw[index], min_radius)
        removed += diagnostic["removed_foreground"]
    return filtered, removed


def mask_metrics(prediction, truth, roi_xyxy, *, boundary_fields=None):
    x0, y0, x1, y1 = roi_xyxy
    pred, expected = prediction[y0:y1, x0:x1], truth[y0:y1, x0:x1]
    tp, fp, fn = int((pred & expected).sum()), int((pred & ~expected).sum()), int((~pred & expected).sum())
    boundary = boundary_f1(prediction, truth, roi_xyxy) if boundary_fields is None else boundary_from_fields(boundary_fields, roi_xyxy)
    return dict(tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
        predicted_foreground=int(pred.sum()), truth_foreground=int(expected.sum()), boundary_f1=boundary)


def contour(mask):
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3,3), bool), border_value=0)


def boundary_from_fields(fields, roi_xyxy):
    """Exact frozen boundary metric, reusing full-plane distances across ROIs."""
    predicted, ground, predicted_distance, ground_distance = fields
    x0, y0, x1, y1 = roi_xyxy
    roi = np.s_[y0:y1, x0:x1]
    pc, gc = int(predicted[roi].sum()), int(ground[roi].sum())
    pm = int((predicted[roi] & (ground_distance[roi] <= 2.)).sum()) if ground_distance is not None else 0
    gm = int((ground[roi] & (predicted_distance[roi] <= 2.)).sum()) if predicted_distance is not None else 0
    precision, recall = pm/pc if pc else None, gm/gc if gc else None
    value = None if precision is None or recall is None else 0. if precision+recall == 0 else 2*precision*recall/(precision+recall)
    return dict(tolerance_native_px=2., prediction_boundary_points=pc, truth_boundary_points=gc,
        matched_prediction_points=pm, matched_truth_points=gm, precision=precision, recall=recall, f1=value,
        status="undefined_zero_boundary_count" if value is None else "measured",
        contour_domain="Full native image before ROI restriction; matches may occur outside ROI")


def endpoint_connections(family, frames, masks, observations):
    """Local 26-connectivity with fixed original endpoint attachments."""
    local = masks.copy()
    crop = family["whole_crop_bbox_yx"]
    for key in family["observation_ids"]:
        observation = observations[key]
        local[frames.index(observation.frame_native)] |= mask_in_crop(observation, crop)
    labels, _ = ndi.label(local, structure=np.ones((3,3,3), bool))
    result = []
    for edge in family["edges"]:
        a, b = observations[edge["source_id"]], observations[edge["target_id"]]
        am, bm = mask_in_crop(a, crop), mask_in_crop(b, crop)
        left = set(map(int, np.unique(labels[frames.index(a.frame_native)][am]))) - {0}
        right = set(map(int, np.unique(labels[frames.index(b.frame_native)][bm]))) - {0}
        result.append(dict(source_id=a.observation_id, target_id=b.observation_id, connected=bool(left & right)))
    return dict(connectivity=26, edges=result, all_declared_edges_connected=all(edge["connected"] for edge in result),
        interpretation="Experimental raw connection diagnostic; does not imply stock-v2 acceptance")


def compare_controls(plan, whole, tiled):
    checks = []
    for family in plan["families"]:
        if family["tile_strategy"]["mode"] != "identical_crop_control":
            continue
        for run in family["runs"]:
            wa, ta = whole.by_original[run["run_id"]], tiled.by_original[run["run_id"]]
            if len(wa) != 1 or len(ta) != 1:
                raise ValueError("Identical control does not have exactly one footprint")
            wr, tr = wa[0], ta[0]
            binary_equal = wr["shape_yx"] == tr["shape_yx"] and wr["native_frames"] == tr["native_frames"] and np.array_equal(whole._packed(wr), tiled._packed(tr))
            checks.append(dict(family_id=family["family_id"], original_run_id=run["run_id"],
                exact_raw_binary_masks_all_frames=binary_equal,
                exact_tracker_object_probabilities=wr["tracker_probabilities"] == tr["tracker_probabilities"],
                native_frames=wr["native_frames"], per_pixel_probabilities="Not retained by the pinned tracker API"))
    return dict(runs=checks, total_runs=len(checks),
        all_binary_masks_exact=all(check["exact_raw_binary_masks_all_frames"] for check in checks),
        all_tracker_object_probabilities_exact=all(check["exact_tracker_object_probabilities"] for check in checks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--evaluation-frame", type=int, default=61)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    root = args.experiment
    plan_path = root/"strategy_plan.json"
    plan = json.loads(plan_path.read_text("utf-8"))
    geometry = Path(__file__).with_name("sam_crop_strategy_geometry.py")
    if hashlib.sha256(geometry.read_bytes()).hexdigest() != plan["geometry_source_sha256"]:
        raise ValueError("Frozen geometry implementation changed")
    report = json.loads((root/"inference_report.json").read_text("utf-8"))
    if report["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Inference did not use the frozen plan")
    begin = time.perf_counter()
    readers = {name:RawRunReader(root/f"repeat{args.repeat}"/strategy, plan) for name, strategy in (("whole","whole_crop"), ("tiles","independent_tiles"))}
    observations = load_observations(plan)
    frame = args.evaluation_frame
    frames = list(range(plan["endpoint_frames_native"][0], plan["endpoint_frames_native"][1]+1))
    frame_position = frames.index(frame)
    native_shape = tuple(plan["source_shape_yx"])
    globals_ = {method:np.zeros(native_shape, bool) for method in METHODS}
    global_available = {f"{strategy}_{direction}":np.zeros(native_shape, bool) for strategy in ("whole", "tiles") for direction in ("forward", "backward", "union")}
    predictions_dir = root/"family_predictions"
    predictions_dir.mkdir(exist_ok=True)
    family_results = []
    for family in plan["families"]:
        y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
        cropped_shape = (len(frames), y1-y0, x1-x0)
        local_eval = {}
        record = dict(family_id=family["family_id"], review_case_ids=family["review_case_ids"], crop_bbox_yx=family["whole_crop_bbox_yx"],
            native_source_edge_censored=family["native_source_edge_censored"], mode=family["tile_strategy"]["mode"], methods={}, original_runs=[])
        for strategy, reader in readers.items():
            directions = {key:np.zeros(cropped_shape, bool) for key in ("raw_forward", "raw_backward", "radius3_forward", "radius3_backward", "available_forward", "available_backward")}
            for run in family["runs"]:
                _, raw, available, _ = reader.assemble_original_run(run["run_id"])
                direction = run["direction"]
                filtered, removed = filter_original_run_planes(raw)
                directions["raw_"+direction] |= raw
                directions["radius3_"+direction] |= filtered
                directions["available_"+direction] |= available
                record["original_runs"].append(dict(strategy=strategy, run_id=run["run_id"], removed_foreground_across_frames=removed,
                    seed_observation_id=run["seed_observation_id"], direction=direction,
                    native_owner_coverage_at_evaluation=float(available[frame_position].mean()),
                    all_frames_complete_owner_coverage=bool(available.all())))
                del raw, available, filtered
            for stage in ("raw", "radius3"):
                directions[stage+"_union"] = directions[stage+"_forward"] | directions[stage+"_backward"]
                for direction in ("forward", "backward", "union"):
                    masks = directions[stage+"_"+direction]
                    method = f"sam_{strategy}_{stage}_{direction}"
                    local_eval[method] = masks[frame_position].copy()
                    globals_[method][y0:y1, x0:x1] |= masks[frame_position]
                    record["methods"][method] = dict(foreground_at_evaluation=int(masks[frame_position].sum()),
                        endpoint_connections=endpoint_connections(family, frames, masks, observations))
            directions["available_union"] = directions["available_forward"] | directions["available_backward"]
            for direction in ("forward", "backward", "union"):
                global_available[strategy+"_"+direction][y0:y1, x0:x1] |= directions["available_"+direction][frame_position]
            del directions
        sdf_file = root/"sdf_reference"/(family["family_id"]+".npz")
        with np.load(sdf_file, allow_pickle=False) as packet:
            sdf_frames = list(map(int, packet["frame_indices"]))
            sdf = packet["masks"][sdf_frames.index(frame)]
            if list(map(int, packet["bbox_yx"])) != family["whole_crop_bbox_yx"]:
                raise ValueError("Matched SDF native crop differs")
        local_eval["sdf"] = sdf
        globals_["sdf"][y0:y1, x0:x1] |= sdf
        prediction = predictions_dir/(family["family_id"]+".npz")
        np.savez_compressed(prediction, bbox_yx=family["whole_crop_bbox_yx"], frame_native=frame, **local_eval)
        record["prediction_file"] = str(prediction)
        family_results.append(record)
        print("Assembled "+family["family_id"], flush=True)
    control_checks = compare_controls(plan, readers["whole"], readers["tiles"])
    for reader in readers.values():
        reader.close()
    # All inference and native reconstructions precede annotation loading.
    global_file = root/"global_eval_predictions.npz"
    np.savez_compressed(global_file, frame_native=frame, **globals_)
    truth = load_truth(args.label, native_shape)
    roi_results = [dict(case=review["case"], global_roi_xyxy=review["global_roi_xyxy"], family_ids=review["family_ids"], methods={}, differences={}) for review in plan["reviews"]]
    supplemental_results = []
    for family in plan["families"]:
        if family["tile_strategy"]["mode"] != "independent_overlapping_tiles":
            continue
        y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
        definitions = [("whole_crop_semantic_foreground", [x0,y0,x1,y1], None)]
        xs = sorted({tile["ownership_bbox_yx"][3] for tile in family["tile_strategy"]["tiles"] if x0 < tile["ownership_bbox_yx"][3] < x1})
        ys = sorted({tile["ownership_bbox_yx"][2] for tile in family["tile_strategy"]["tiles"] if y0 < tile["ownership_bbox_yx"][2] < y1})
        definitions.extend(("fixed_vertical_seam_strip", [max(x0,x-16),y0,min(x1,x+16),y1], x) for x in xs)
        definitions.extend(("fixed_horizontal_seam_strip", [x0,max(y0,y-16),x1,min(y1,y+16)], y) for y in ys)
        for index, (kind, roi, seam) in enumerate(definitions):
            supplemental_results.append(dict(case=family["family_id"]+f"_supplement{index}", kind=kind,
                family_ids=[family["family_id"]], global_roi_xyxy=roi, seam_native_coordinate=seam,
                seam_half_width_native_px=16 if seam is not None else None,
                posthoc_seen_data_diagnostic=True,
                limitation="Absolute semantic foreground includes other annotated objects; combined global predictions are not pure large-instance segmentation. Geometry-derived domain, not label-selected.",
                methods={}, differences={}))
    all_roi_results = roi_results+supplemental_results
    truth_contour = contour(truth)
    truth_distance = ndi.distance_transform_edt(~truth_contour) if truth_contour.any() else None
    for method, prediction in globals_.items():
        prediction_contour = contour(prediction)
        prediction_distance = ndi.distance_transform_edt(~prediction_contour) if prediction_contour.any() else None
        fields = (prediction_contour, truth_contour, prediction_distance, truth_distance)
        for row in all_roi_results:
            roi = row["global_roi_xyxy"]
            x0, y0, x1, y1 = roi
            metric = mask_metrics(prediction, truth, roi, boundary_fields=fields)
            if method != "sdf":
                _, strategy, _, direction = method.split("_")
                coverage = global_available[strategy+"_"+direction][y0:y1, x0:x1]
                metric["native_owner_coverage_fraction"] = float(coverage.mean())
                pred_roi, truth_roi = prediction[y0:y1, x0:x1], truth[y0:y1, x0:x1]
                tp = int((pred_roi & truth_roi & coverage).sum())
                fp = int((pred_roi & ~truth_roi & coverage).sum())
                fn = int((~pred_roi & truth_roi & coverage).sum())
                metric["covered_roi_only"] = dict(tp=tp, fp=fp, fn=fn, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                    scope="Additional coverage diagnostic; full-ROI metrics count unavailable owner pixels as missed prediction")
            row["methods"][method] = metric
        print("Scored "+method, flush=True)
    for row in all_roi_results:
        x0, y0, x1, y1 = row["global_roi_xyxy"]
        for stage in ("raw", "radius3"):
            a, b = globals_[f"sam_whole_{stage}_union"][y0:y1, x0:x1], globals_[f"sam_tiles_{stage}_union"][y0:y1, x0:x1]
            intersection, union = int((a & b).sum()), int((a | b).sum())
            row["differences"][stage] = dict(whole_only=int((a & ~b).sum()), tiles_only=int((b & ~a).sum()),
                intersection=intersection, union=union, whole_tiles_iou=intersection/union if union else None)
    results = dict(schema="xta.sam_paired_crop_strategy_analysis/1", plan_sha256=plan["plan_sha256"], plan_file=str(plan_path),
        evaluation_frame_native=frame, evaluation_original_frame=594+frame,
        study_scope="Paired follow-up on previously seen annotation61; no blind-validation or independent-patient claim",
        production_acceptance="Raw and radius3 diagnostics only; neither implies conservative-v2 accepted output",
        filter_order="Native tiles assembled per original seed run, component radius3 filter per run, then independent seed hypotheses united per direction/family",
        label=str(args.label), label_sha256=hashlib.sha256(args.label.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        roi_results=roi_results, supplemental_results=supplemental_results, unique_family_results=family_results, same_crop_controls=control_checks,
        global_prediction_file=str(global_file), timing_report_file=str(root/"inference_report.json"),
        inference_sessions=report["sessions"], quality_audit_file=str(root/"quality_analysis.json"),
        quality_audit_available=(root/"quality_analysis.json").exists(), methods=METHODS,
        foreground_aggregates={}, analysis_seconds=time.perf_counter()-begin)
    for method in METHODS:
        totals = {field:sum(row["methods"][method][field] for row in roi_results) for field in ("tp", "fp", "fn")}
        totals.update(micro_iou=totals["tp"]/max(1,sum(totals.values())),
            macro_iou=float(np.mean([row["methods"][method]["iou"] for row in roi_results if row["methods"][method]["iou"] is not None])),
            precision=totals["tp"]/max(1,totals["tp"]+totals["fp"]), recall=totals["tp"]/max(1,totals["tp"]+totals["fn"]))
        results["foreground_aggregates"][method] = totals
    (root/"analysis.json").write_text(json.dumps(results, indent=2), "utf-8")
    print(json.dumps(dict(main_metrics={method:results["foreground_aggregates"][method] for method in ("sdf","sam_whole_raw_union","sam_tiles_raw_union","sam_whole_radius3_union","sam_tiles_radius3_union")},
        controls_binary_exact=control_checks["all_binary_masks_exact"], controls_tracker_probabilities_exact=control_checks["all_tracker_object_probabilities_exact"], seconds=results["analysis_seconds"]), indent=2))


if __name__ == "__main__":
    main()
