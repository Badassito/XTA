"""Replay retained SAM tracks with per-original-seed largest-island filtering.

This CPU-only diagnostic never changes a production receipt, tracker seed, or
raw prediction. Native tile owners are assembled before each original run is
filtered; independent seed runs are united afterwards. Largest means maximum
8-connected component area per frame, with deterministic row-major tie breaking.
It cannot promise biological lineage or split two daughters joined by foreground.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from XTA.sam_filtering import filter_sam_components
from tools.analyze_sam_crop_strategies import RawRunReader, load_observations, load_truth, endpoint_connections
from tools.sam_crop_strategy_geometry import mask_in_crop

METHODS = ("raw", "radius", "largest", "radius_largest")


def largest_island(mask):
    """Return one complete 8-connected component; empty input stays empty."""
    value = np.asarray(mask)
    if value.ndim != 2 or not np.isin(value, (0, 1)).all():
        raise ValueError("Largest-island filtering requires a binary 2D mask")
    labels, count = ndi.label(value, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count+1)[1:]
    chosen = int(np.argmax(sizes))+1 if count else None
    result = labels == chosen if chosen is not None else np.zeros(value.shape, bool)
    return result, dict(component_count=int(count), selected_component_id=chosen,
        foreground=int(value.sum()), selected_foreground=int(result.sum()),
        removed_foreground=int(value.sum())-int(result.sum()),
        tied_largest_count=int(np.count_nonzero(sizes == sizes.max())) if count else 0)


def replay_filters(raw, radius):
    """Apply all variants per original run before independent-run union."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or not np.isin(raw, (0, 1)).all():
        raise ValueError("Retained run must be a binary frame,y,x array")
    outputs = {key: np.zeros(raw.shape, bool) for key in METHODS}
    diagnostics = []
    for index, mask in enumerate(raw):
        effective, radius_measurement = filter_sam_components(mask, radius)
        largest, largest_measurement = largest_island(mask)
        radius_largest, joint_measurement = largest_island(effective)
        outputs["raw"][index] = mask
        outputs["radius"][index] = effective
        outputs["largest"][index] = largest
        outputs["radius_largest"][index] = radius_largest
        diagnostics.append(dict(frame_position=index, radius=radius_measurement,
            largest=largest_measurement, radius_largest=joint_measurement,
            largest_only_vs_radius=int(np.count_nonzero(largest & ~effective)),
            radius_only_vs_largest=int(np.count_nonzero(effective & ~largest))))
    return outputs, diagnostics


def terminal_reachability(family, run, frames, masks, observations):
    """Query terminal observations without adding them to propagated support.

    Only the original seed is attached to the tracked volume. Opposite endpoint
    masks are queries, so an observed target cannot repair a disconnected track.
    Nonzero overlap is an attachment diagnostic, not a production acceptance rule.
    """
    seed = observations[run["seed_observation_id"]]
    crop = family["whole_crop_bbox_yx"]
    seed_mask = mask_in_crop(seed, crop)
    local = np.array(masks, dtype=bool, copy=True)
    seed_index = frames.index(seed.frame_native)
    local[seed_index] |= seed_mask
    labels, _ = ndi.label(local, structure=np.ones((3, 3, 3), bool))
    roots = np.unique(labels[seed_index][seed_mask])
    roots = roots[roots != 0]
    targets = []
    for identity in run["held_out_observation_ids"]:
        observation = observations[identity]
        reference = mask_in_crop(observation, crop)
        index = frames.index(observation.frame_native)
        prediction = masks[index]
        connected = np.isin(labels[index], roots) & reference
        intersection = int(np.count_nonzero(prediction & reference))
        union = int(np.count_nonzero(prediction | reference))
        targets.append(dict(observation_id=identity, foreground=int(reference.sum()),
            intersection=intersection, recall=intersection/max(1, int(reference.sum())),
            iou=intersection/max(1, union), reachable_foreground=int(connected.sum()),
            reachable=bool(connected.any())))
    temporal_gaps = []
    for index in range(1, len(frames)):
        if masks[index-1].any() and masks[index].any() and not np.any(
                ndi.binary_dilation(masks[index-1], structure=np.ones((3, 3), bool)) & masks[index]):
            temporal_gaps.append([frames[index-1], frames[index]])
    return dict(targets=targets, reachable_target_count=sum(row["reachable"] for row in targets),
        nonzero_overlap_target_count=sum(row["intersection"] > 0 for row in targets),
        stock_default_endpoint_recall_threshold=.5,
        stock_all_held_out_recall_gate=bool(targets) and all(row["recall"] >= .5 for row in targets),
        research_any_target_recall_gate=bool(targets) and any(row["recall"] >= .5 for row in targets),
        recall_gate_scope="Endpoint recall only; no containment, excess, availability, or topology acceptance implied",
        reachable_all_targets=bool(targets) and all(row["reachable"] for row in targets),
        nonempty_consecutive_planes_without_26_adjacency=temporal_gaps)


def _metrics(prediction, reference):
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    return dict(tp=tp, fp=fp, fn=fn, iou=tp/max(1, tp+fp+fn),
        precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def study(experiment, output, *, repeats=(1, 2), evaluation_frame=61, radius=3., label=None):
    """Freeze coordinate-bound predictions before optional annotation scoring."""
    experiment, output = Path(experiment), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = experiment/"strategy_plan.json"
    plan = json.loads(plan_path.read_text("utf-8"))
    observations = load_observations(plan)
    frames = list(range(plan["endpoint_frames_native"][0], plan["endpoint_frames_native"][1]+1))
    evaluation_position = frames.index(evaluation_frame)
    shape = tuple(plan["source_shape_yx"])
    report = dict(schema="xta.sam_largest_island_study/1", tool_sha256=_sha(__file__),
        plan_path=str(plan_path.resolve()), plan_file_sha256=_sha(plan_path),
        plan_sha256=plan["plan_sha256"], radius_native_px=radius, evaluation_frame_native=evaluation_frame,
        methods=list(METHODS), connectivity_2d=8, connectivity_3d=26,
        tie_break="First row-major connected-component label among equal maximum areas",
        filter_order="Assemble fixed native tile owners per original seed; filter each run independently; unite independent runs by direction",
        evidence_kind="CPU replay of retained raw SAM masks; no tracker rerun or production acceptance receipt",
        planning_used_labels=False, limitations=["Previously seen development data, not a blind holdout",
            "Largest island selects area independently on every frame; it can switch branches",
            "Connected daughters remain one island", "Removing raw islands after tracking does not change recurrent model memory",
            "Stock selection requires every held-out endpoint; one-daughter parent runs can correctly fail its recall gate",
            "Endpoint attachment and observed-mask topology are diagnostics; they do not imply production acceptance"],
        sessions=[])
    frozen = []
    for repeat in repeats:
        for strategy, directory in (("whole", "whole_crop"), ("tiles", "independent_tiles")):
            source = experiment/f"repeat{repeat}"/directory
            reader = RawRunReader(source, plan)
            before = {name:_sha(source/name) for name in ("raw_index.json", "raw_masks.npz")}
            global_masks = {f"{method}_{direction}":np.zeros(shape, bool)
                for method in METHODS for direction in ("forward", "backward", "union")}
            family_masks = {}
            session = dict(repeat=repeat, strategy=strategy, source_directory=str(source.resolve()),
                source_sha256=before, families=[], skipped_families=[],
                planned_original_run_count=sum(len(family["runs"]) for family in plan["families"]),
                retained_original_run_count=len(reader.by_original))
            try:
                for family in plan["families"]:
                    missing = [run["run_id"] for run in family["runs"] if run["run_id"] not in reader.by_original]
                    if missing:
                        session["skipped_families"].append(dict(family_id=family["family_id"],
                            missing_original_run_ids=missing, reason="No complete paired original-run inventory; never fill missing runs with background"))
                        continue
                    y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
                    local_shape = (len(frames), y1-y0, x1-x0)
                    combined = {f"{method}_{direction}":np.zeros(local_shape, bool)
                        for method in METHODS for direction in ("forward", "backward")}
                    record = dict(family_id=family["family_id"], observation_ids=family["observation_ids"],
                        crop_bbox_yx=family["whole_crop_bbox_yx"], runs=[], methods={})
                    for run in family["runs"]:
                        _, raw, available, _ = reader.assemble_original_run(run["run_id"])
                        outputs, diagnostics = replay_filters(raw, radius)
                        run_record = dict(run_id=run["run_id"], seed_observation_id=run["seed_observation_id"],
                            direction=run["direction"], held_out_observation_ids=run["held_out_observation_ids"],
                            all_frames_complete_owner_coverage=bool(available.all()),
                            frame_diagnostics=[dict(row, frame_native=frames[index]) for index,row in enumerate(diagnostics)], methods={})
                        for method, masks in outputs.items():
                            combined[f"{method}_{run['direction']}"] |= masks
                            run_record["methods"][method] = dict(foreground_across_frames=int(masks.sum()),
                                terminal=terminal_reachability(family, run, frames, masks, observations))
                        record["runs"].append(run_record)
                    for method in METHODS:
                        combined[f"{method}_union"] = combined[f"{method}_forward"] | combined[f"{method}_backward"]
                        for direction in ("forward", "backward", "union"):
                            name = f"{method}_{direction}"
                            masks = combined[name]
                            global_masks[name][y0:y1,x0:x1] |= masks[evaluation_position]
                            family_masks[f"{family['family_id']}__{name}"] = masks[evaluation_position].copy()
                            record["methods"][name] = dict(foreground_at_evaluation=int(masks[evaluation_position].sum()),
                                topology=endpoint_connections(family, frames, masks, observations))
                    session["families"].append(record)
                    print(f"repeat{repeat} {strategy}: {family['family_id']}", flush=True)
                destination = output/f"repeat{repeat}_{strategy}_evaluation.npz"
                np.savez_compressed(destination, frame_native=evaluation_frame, **global_masks)
                session["frozen_prediction_file"] = str(destination.resolve())
                session["frozen_prediction_sha256"] = _sha(destination)
                family_destination = output/f"repeat{repeat}_{strategy}_family_evaluation.npz"
                np.savez_compressed(family_destination, **family_masks)
                session["frozen_family_prediction_file"] = str(family_destination.resolve())
                session["frozen_family_prediction_sha256"] = _sha(family_destination)
                frozen.append((session, destination, family_destination))
                if before != {name:_sha(source/name) for name in before}:
                    raise RuntimeError("Retained raw SAM evidence changed during replay")
                report["sessions"].append(session)
            finally:
                reader.close()
    # This is deliberately after all variant masks are frozen and hashed.
    if label is not None:
        truth = load_truth(label, shape)
        report["label_path"] = str(Path(label).resolve())
        report["label_sha256"] = _sha(label)
        for session, path, family_path in frozen:
            rows = []
            completed = {family["family_id"] for family in session["families"]}
            session["unscored_review_ids"] = []
            with np.load(path, allow_pickle=False) as saved:
                for review in plan["reviews"]:
                    if not set(review["family_ids"]).issubset(completed):
                        session["unscored_review_ids"].append(review["case"])
                        continue
                    x0, y0, x1, y1 = review["global_roi_xyxy"]
                    rows.append(dict(case=review["case"], roi_xyxy=review["global_roi_xyxy"],
                        methods={name:_metrics(saved[name][y0:y1,x0:x1], truth[y0:y1,x0:x1])
                            for name in saved.files if name != "frame_native"}))
            session["roi_metrics"] = rows
            with np.load(family_path, allow_pickle=False) as saved:
                for family in session["families"]:
                    y0,x0,y1,x1 = family["crop_bbox_yx"]
                    family["annotation_crop_metrics"] = dict(
                        scope="Semantic foreground in fixed family crop; other annotated objects may be present; no instance identity claim",
                        methods={name:_metrics(saved[f"{family['family_id']}__{name}"], truth[y0:y1,x0:x1])
                            for name in family["methods"]})
            session["foreground_aggregates"] = {}
            for name in rows[0]["methods"] if rows else ():
                totals = {key:sum(row["methods"][name][key] for row in rows) for key in ("tp","fp","fn")}
                tp,fp,fn = (totals[key] for key in ("tp","fp","fn"))
                session["foreground_aggregates"][name] = dict(totals, micro_iou=tp/max(1,tp+fp+fn),
                    precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn))
    report_path = output/"largest_island_study.json"
    report_path.write_text(json.dumps(report, indent=2), "utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--evaluation-frame", type=int, default=61)
    parser.add_argument("--radius", type=float, default=3.)
    parser.add_argument("--label", type=Path)
    args = parser.parse_args()
    study(args.experiment, args.output, repeats=args.repeats, evaluation_frame=args.evaluation_frame,
        radius=args.radius, label=args.label)


if __name__ == "__main__":
    main()
