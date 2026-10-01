"""Generate matched native SDF references and the paired crop-strategy report.

Research tooling only. References use the frozen original component masks and
production SDF geometry. Reports and all generated data belong in Scratch.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sdf_reference(args):
    for key in ("YOLO_TTA_GPU_INTERPOLATION", "YOLO_TTA_GPU_SLICE_LABELING",
                "YOLO_TTA_GPU_INTERPOLATION_RADIUS", "YOLO_TTA_GPU_INTERPOLATION_REQUIRED"):
        os.environ[key] = "0"
    os.environ["YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER"] = "0"
    os.environ["YOLO_TTA_INTERPOLATION_PLAN_BATCH_MIB"] = "128"
    import numpy as np
    from XTA import interpolation
    interpolation.cv2.setNumThreads(1)
    plan_path = args.experiment / "strategy_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seed_dir = Path(plan["seed_directory"])
    if not seed_dir.is_absolute():
        seed_dir = args.experiment / seed_dir
    first, last = map(int, plan["endpoint_frames_native"])
    if last - first > int(plan["settings"]["interpolation_distance"]):
        raise ValueError("Endpoint distance exceeds the frozen interpolation distance")
    frames = np.arange(first, last + 1, dtype=np.int32)
    destination = args.experiment / "sdf_reference"
    destination.mkdir(exist_ok=True)
    result = {"schema": "xta.native_crop_matched_sdf/1", "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": file_sha(plan_path), "labels_used": False,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": file_sha(interpolation.__file__), "tool_sha256": file_sha(__file__),
        "parameters": {"distance": int(plan["settings"]["interpolation_distance"]),
            "search_angle": float(plan["settings"]["projection_search_angle_degrees"]),
            "walk_back": int(plan["settings"]["interpolation_walk_back"]),
            "candidates": 1, "passes": 1, "min_radius": float(plan["settings"]["interpolation_min_radius"]),
            "workers": args.workers, "gpu": False},
        "walk_back_availability": "Only original endpoint frames supplied; no adjacent observations and no generated seeds",
        "geometry": "Complete original native family masks inside the same frozen whole-family crop; global coordinates restored by bbox_yx",
        "timing_scope": "Local serial CPU reference. SDF compute is separate from input loading, compression and connectivity audit; not GPU-model-only timing",
        "families": []}
    write_json(destination / "protocol.json", result)
    observation_info = {item["observation_id"]: item for item in plan["source_observations"]}
    for family in plan["families"]:
        started = time.perf_counter()
        fy0, fx0, fy1, fx1 = map(int, family["whole_crop_bbox_yx"])
        observed = np.zeros((len(frames), fy1 - fy0, fx1 - fx0), np.uint8)
        seeds, seed_hashes = {}, {}
        for identifier in family["observation_ids"]:
            path = seed_dir / (identifier + ".npz")
            seed_hashes[identifier] = file_sha(path)
            with np.load(path, allow_pickle=False) as archive:
                mask = np.asarray(archive["mask"], bool)
                sy0, sx0, sy1, sx1 = map(int, archive["bbox_yx"])
                frame = int(archive["frame_native"])
            if not (fy0 <= sy0 < sy1 <= fy1 and fx0 <= sx0 < sx1 <= fx1):
                raise ValueError("Frozen whole crop clips an original observation")
            if frame not in (first, last):
                raise ValueError("Reference may only use the frozen original endpoints")
            if int(mask.sum()) != int(observation_info[identifier]["foreground"]):
                raise ValueError("Original observation foreground changed")
            observed[frame - first, sy0 - fy0:sy1 - fy0, sx0 - fx0:sx1 - fx0] |= mask
            point = np.argwhere(mask)[0] + np.asarray((sy0 - fy0, sx0 - fx0))
            seeds[identifier] = (frame - first, int(point[0]), int(point[1]))
        data = observed.copy()
        compute_started = time.perf_counter()
        stats = interpolation.interpolate_view_volume_pass_inplace(
            data, destination / (family["family_id"] + "_work"), "native_reference",
            result["parameters"]["distance"], result["parameters"]["search_angle"],
            result["parameters"]["walk_back"], 1, result["parameters"]["min_radius"],
            keep_temp=False, prefer_memory=False, reserve_bytes=0, workers=args.workers)
        compute_seconds = time.perf_counter() - compute_started
        if not np.array_equal(data[[0, -1]], observed[[0, -1]]):
            raise ValueError("SDF repainted an original endpoint")
        additions = (data != 0) & ~(observed != 0)
        target = destination / (family["family_id"] + ".npz")
        np.savez_compressed(target, frame_indices=frames, bbox_yx=np.asarray(family["whole_crop_bbox_yx"], np.int32), masks=additions)
        labels, _ = interpolation.ndi.label(data != 0, structure=np.ones((3, 3, 3), bool))
        edges = []
        for edge in family["edges"]:
            a = int(labels[seeds[edge["source_id"]]])
            b = int(labels[seeds[edge["target_id"]]])
            edges.append({"source_id": edge["source_id"], "target_id": edge["target_id"], "connected": a > 0 and a == b})
        for identifier, digest in seed_hashes.items():
            if file_sha(seed_dir / (identifier + ".npz")) != digest:
                raise ValueError("Original seed changed during reference generation")
        row = {"family_id": family["family_id"], "review_case_ids": family["review_case_ids"],
            "prediction": str(target.relative_to(args.experiment)), "prediction_file_sha256": file_sha(target),
            "bbox_yx": family["whole_crop_bbox_yx"], "frame_indices": frames.tolist(),
            "source_seed_file_sha256": seed_hashes, "input_foreground_by_frame": observed.sum(axis=(1, 2)).tolist(),
            "added_foreground_by_frame": additions.sum(axis=(1, 2)).tolist(),
            "compute_wall_seconds": compute_seconds, "generation_save_and_audit_wall_seconds": time.perf_counter() - started,
            "native_edge_connections": edges, "statistics": stats, "model_inference_count": 0}
        result["families"].append(row)
        write_json(destination / "results.json", result)
        print(f"SDF {family['family_id']}: {compute_seconds:.3f}s, {int(additions.sum())} added voxels", flush=True)
        del labels, additions, data, observed
    result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if file_sha(plan_path) != result["plan_file_sha256"]:
        raise ValueError("Frozen plan changed during reference generation")
    write_json(destination / "results.json", result)
    print(f"SDF reference complete: {len(result['families'])} frozen native families", flush=True)


def report(args):
    os.environ.setdefault("MPLCONFIGDIR", str(args.experiment / "matplotlib_cache"))
    import html
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    root = args.experiment
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "strategy_plan.json").read_text(encoding="utf-8"))
    inference = json.loads((root / "inference_report.json").read_text(encoding="utf-8"))
    authority = json.loads((root / "native_crop_authority.json").read_text(encoding="utf-8")) if (root/"native_crop_authority.json").exists() else {}
    reference = json.loads((root / "sdf_reference/results.json").read_text(encoding="utf-8"))
    if analysis["plan_sha256"] != plan["plan_sha256"] or reference["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Report inputs do not share the same frozen native plan")
    output = args.output or root / "crop_strategy_report.html"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = output.parent / "crop_strategy_figures"
    figures.mkdir(exist_ok=True)
    escape = lambda value: html.escape(str(value))
    fmt = lambda value, digits=4: "—" if value is None else f"{value:.{digits}f}"
    methods = [("sdf", "Matched native SDF"), ("sam_whole_radius3_union", "Whole crop resize"),
               ("sam_tiles_radius3_union", "Independent tiles")]

    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join('<td>'+str(cell)+'</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'

    def relative(path):
        return os.path.relpath(path, output.parent).replace("\\", "/")

    def link(path, label):
        return '<a href="'+escape(relative(root/path))+'">'+escape(label)+'</a>'

    metric_rows, raw_rows = [], []
    for row in analysis["roi_results"]:
        for method, label in methods:
            metric = row["methods"][method]
            metric_rows.append([escape(row["case"]), escape(label), fmt(metric["iou"]), fmt(metric["precision"]),
                fmt(metric["recall"]), fmt(metric["boundary_f1"]["f1"]), f"{metric['tp']:,}", f"{metric['fp']:,}", f"{metric['fn']:,}"])
        for method, label in (("sam_whole_raw_union", "Whole crop raw"), ("sam_tiles_raw_union", "Independent tiles raw")):
            metric = row["methods"][method]
            raw_rows.append([escape(row["case"]), escape(label), fmt(metric["iou"]), fmt(metric["precision"]),
                             fmt(metric["recall"]), fmt(metric["boundary_f1"]["f1"])])
    timing_rows = []
    for session in inference["sessions"]:
        timing_rows.append([str(session["repeat"]), escape(session["strategy"]), str(len(session["families"])),
            str(session["request_count"]), fmt(session["predictor_start_seconds"],3),
            fmt(session["tracking_render_transfer_wall_seconds"],3), fmt(session["worker_tracker_seconds"],3),
            str(session["encoder_preparations"]), str(session["feature_cache_hits"])])
    compute_seconds = sum(row["compute_wall_seconds"] for row in reference["families"])
    reference_total = sum(row["generation_save_and_audit_wall_seconds"] for row in reference["families"])
    native_shape = tuple(plan["source_shape_yx"])
    frame = int(analysis["evaluation_frame_native"])
    images = np.memmap(plan["source_images"], mode="r", dtype=np.uint8, shape=tuple(plan["source_shape_tyx"]))
    image = images[frame-int(plan["source_frame_start"])]
    label_path = Path(analysis["label"])
    if file_sha(label_path) != analysis["label_sha256"]:
        raise ValueError("Evaluation annotation changed")
    truth_image = Image.new("1", native_shape[::-1], 0)
    draw = ImageDraw.Draw(truth_image)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        values = list(map(float, parts[1:]))
        draw.polygon([(values[i]*native_shape[1],values[i+1]*native_shape[0]) for i in range(0,len(values),2)],fill=1)
    truth = np.asarray(truth_image, bool)
    with np.load(analysis["global_prediction_file"], allow_pickle=False) as archive:
        if int(archive["frame_native"]) != frame:
            raise ValueError("Prediction frame differs from the scored frame")
        predictions = {method: np.asarray(archive[method], bool) for method,_ in methods}
    if any(value.shape != native_shape for value in predictions.values()):
        raise ValueError("Predictions are not on the native source canvas")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    selected_cases = ["region01_case02", "region01_case01", "region02_case03"]
    row_lookup = {row["case"]:row for row in analysis["roi_results"]}

    def error_overlay(axis, viewport, method):
        y0,x0,y1,x1 = viewport
        axis.imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
        pred,gt = predictions[method][y0:y1,x0:x1],truth[y0:y1,x0:x1]
        colors = np.zeros((*gt.shape,4),np.float32)
        colors[pred & gt] = (.15,.82,.43,.55)
        colors[pred & ~gt] = (1.,.22,.52,.85)
        colors[~pred & gt] = (.12,.77,1.,.75)
        axis.imshow(colors)
        axis.set_xticks([]);axis.set_yticks([])

    fig,axes = plt.subplots(3,4,figsize=(12.8,10),squeeze=False)
    for index,case in enumerate(selected_cases):
        row = row_lookup[case];x0,y0,x1,y1 = row["global_roi_xyxy"]
        axes[index,0].imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
        gt = truth[y0:y1,x0:x1]
        if gt.any() and not gt.all():
            axes[index,0].contour(gt,levels=[.5],colors=["#ffd84a"],linewidths=1)
        axes[index,0].set_title("Image + annotation")
        axes[index,0].set_ylabel(case.replace("region","R").replace("_case"," · "),fontweight="bold")
        axes[index,0].set_xticks([]);axes[index,0].set_yticks([])
        for column,(method,label) in enumerate(methods,1):
            error_overlay(axes[index,column],(y0,x0,y1,x1),method)
            metric=row["methods"][method]
            axes[index,column].set_title(label+f"\nIoU {fmt(metric['iou'],3)} · BF1 {fmt(metric['boundary_f1']['f1'],3)}")
    fig.legend(handles=[Patch(color="#ffd84a",label="Annotation"),Patch(color="#26d16e",label="True positive"),
        Patch(color="#ff3885",label="False positive"),Patch(color="#1fc4ff",label="False negative")],loc="lower center",ncol=4,frameon=False)
    fig.subplots_adjust(wspace=.10,hspace=.25,bottom=.065,top=.96)
    fig.savefig(figures/"roi_comparison.png",dpi=170,bbox_inches="tight");plt.close(fig)
    large = next(f for f in plan["families"] if "region01_case02" in f["review_case_ids"])
    y0,x0,y1,x1 = large["whole_crop_bbox_yx"]
    fig,axes = plt.subplots(4,1,figsize=(13,10))
    axes[0].imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
    axes[0].contour(truth[y0:y1,x0:x1],levels=[.5],colors=["#ffd84a"],linewidths=.6)
    for tile in large["tile_strategy"]["tiles"]:
        by0,bx0,by1,bx1 = tile["crop_bbox_yx"]
        axes[0].add_patch(Rectangle((bx0-x0,by0-y0),bx1-bx0,by1-by0,fill=False,edgecolor="#7ec9e3",linewidth=1.2))
        oy0,ox0,oy1,ox1 = tile["ownership_bbox_yx"]
        axes[0].add_patch(Rectangle((ox0-x0,oy0-y0),ox1-ox0,oy1-oy0,fill=False,edgecolor="white",linestyle="--",linewidth=1.3))
    axes[0].set_title("Full fixed family crop: annotation yellow, tile footprints blue, ownership seams white")
    axes[0].set_xticks([]);axes[0].set_yticks([])
    for axis,(method,label) in zip(axes[1:],methods):
        error_overlay(axis,(y0,x0,y1,x1),method);axis.set_title(label)
    fig.legend(handles=[Patch(color="#26d16e",label="True positive"),Patch(color="#ff3885",label="False positive"),
        Patch(color="#1fc4ff",label="False negative")],loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(figures/"large_family_comparison.png",dpi=170,bbox_inches="tight");plt.close(fig)
    supplements = analysis.get("supplemental_results", [])
    supplement_rows = []
    for row in supplements:
        for method,label in methods:
            metric = row["methods"][method]
            domain = "Whole fixed family crop" if row["kind"] == "whole_crop_semantic_foreground" else f"Seam x={row['seam_native_coordinate']} ±16px"
            supplement_rows.append([escape(domain),escape(label),fmt(metric["iou"]),fmt(metric["precision"]),fmt(metric["recall"]),
                fmt(metric["boundary_f1"]["f1"]),f"{metric['fp']:,}",f"{metric['fn']:,}"])
    seam_rows = [row for row in supplements if row["kind"] == "fixed_vertical_seam_strip"]
    if seam_rows:
        fig,axes = plt.subplots(len(seam_rows),4,figsize=(12.8,7.4),squeeze=False)
        cy=(y0+y1)//2
        for index,row in enumerate(seam_rows):
            seam=int(row["seam_native_coordinate"])
            viewport=(max(y0,cy-180),seam-96,min(y1,cy+180),seam+96)
            vy0,vx0,vy1,vx1=viewport
            axes[index,0].imshow(image[vy0:vy1,vx0:vx1],cmap="gray",vmin=0,vmax=255)
            axes[index,0].contour(truth[vy0:vy1,vx0:vx1],levels=[.5],colors=["#ffd84a"],linewidths=.8)
            axes[index,0].set_title(f"Seam x={seam} · fixed geometry zoom")
            axes[index,0].set_xticks([]);axes[index,0].set_yticks([])
            for column,(method,label) in enumerate(methods,1):
                error_overlay(axes[index,column],viewport,method);axes[index,column].set_title(label)
            for axis in axes[index]:
                axis.axvline(seam-vx0,color="white",linestyle="--",linewidth=1)
        fig.legend(handles=[Patch(color="#26d16e",label="True positive"),Patch(color="#ff3885",label="False positive"),
            Patch(color="#1fc4ff",label="False negative")],loc="lower center",ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.045,1,1));fig.savefig(figures/"seam_zooms.png",dpi=170,bbox_inches="tight");plt.close(fig)
    sessions = inference["sessions"]
    large_sessions = [s for s in sessions if len(s["families"])==1]
    large_whole = next(s for s in large_sessions if s["strategy"]=="whole_crop")
    large_tiles = next(s for s in large_sessions if s["strategy"]=="independent_tiles")
    timing_ratio = large_tiles["tracking_render_transfer_wall_seconds"]/large_whole["tracking_render_transfer_wall_seconds"]
    controls = analysis["same_crop_controls"]
    supplement_html = table(['Domain','Method','IoU','Precision','Recall','Boundary F1 @2px','FP','FN'],supplement_rows)
    supplement_html += '<p class="small">These supplemental domains were requested after inspecting the six fixed ROI results. Their bounds follow the already frozen full crop and ownership seams, with no reinference or tuning. Semantic full-crop truth may contain other untracked annotated objects; this is not pure tracked-instance accuracy. Seam metrics use the entire crop height and fixed ±16px bands; the visual zoom uses fixed crop-centered geometry.</p>'
    quality = json.loads((root/"quality_analysis.json").read_text(encoding="utf-8")) if (root/"quality_analysis.json").exists() else {}
    quality_rows = []
    for strategy,family in quality.get("primary_family",{}).items():
        for run in family["original_runs"]:
            roles = run["outer_context_contacts_by_role"]
            raw_contacts = sum(role["raw_outer_context_touch"] for role in roles.values())
            filtered_contacts = sum(role["filtered_outer_context_touch"] for role in roles.values())
            quality_rows.append([escape(strategy),escape(run["direction"]),str(raw_contacts),str(filtered_contacts),
                str(roles["intermediate"]["filtered_outer_context_touch"]),str(roles["held_out_endpoint"]["filtered_outer_context_touch"])])
    quality_html = table(['Strategy','Direction','Raw outer contacts','After radius3','Intermediate','Held-out endpoint'],quality_rows)
    if quality:
        quality_html += '<p>All surviving contacts above belong to the largest component; filtered dots do not explain them. These are cumulative foreground-frame contacts on the <strong>shared outer context edge</strong>, not unique voxels or doubled raw halos. Original endpoint silhouettes are complete, but intermediate prediction extent is still context limited. Native 26-connected local links survive; opposite detector-endpoint agreement is about 0.964 for the large whole-crop runs, which is not independent ground-truth accuracy.</p><p><strong>No stock-v2 production acceptance receipt exists.</strong> This research execution bypasses the integrated planner and tile gate and uses different diagnostic spatial contracts. Stock-style topology work was estimated at 326,600,400 bytes, exceeding the default 256 MiB budget; the audit used a bounded 512 MiB research limit. Neither strategy is presented as a production-accepted fix.</p>'
        quality_html += '<p>'+link('quality_analysis.json','Detailed quality summary')+' · '+link('quality_audit.md','Quality interpretation')+' · '+link('quality_mask_parity.json','All 54 filtered-mask comparisons matched the accuracy scorer')+'</p>'
    else:
        quality_html += '<p>Separate conservative-v2 quality diagnosis is pending. Research outputs shown here are not accepted production bridges.</p>'
    rim_html = ''
    rim = json.loads((root/"outer_rim_boundary_check.json").read_text(encoding="utf-8")) if (root/"outer_rim_boundary_check.json").exists() else {}
    if rim:
        a = rim["methods"]["sam_whole_radius3_union"]["excluding_outer_2px"]["f1"]
        b = rim["methods"]["sam_tiles_radius3_union"]["excluding_outer_2px"]["f1"]
        rim_html = f'<p class="small">A post-hoc robustness check excluding a fixed 2px outer-context rim retained the wider-crop boundary improvement: {a*100:.2f}% → {b*100:.2f}% (+{(b-a)*100:.2f} points). It uses the same combined global predictions and changes no primary metric or inference. '+link('outer_rim_boundary_check.json','Robustness evidence')+'</p>'
    sources = ["strategy_plan.json","protocol.json","endpoints.json","inference_report.json","analysis.json",
               "sdf_reference/results.json","independent_control_equivalence.json","independent_large_roi_check.json"]
    sources += [name for name in ("quality_analysis.json","quality_mask_parity.json","quality_audit.md","outer_rim_boundary_check.json","native_crop_authority.json") if (root/name).exists()]
    summary = {"schema":"xta.sam_crop_strategy_report/1","generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256":plan["plan_sha256"],"study_scope":analysis["study_scope"],"production_acceptance":analysis["production_acceptance"],
        "main_methods":[key for key,_ in methods],"roi_results":analysis["roi_results"],"inference_sessions":[{key:s[key] for key in (
            "repeat","strategy","families","request_count","predictor_start_seconds","tracking_render_transfer_wall_seconds","worker_tracker_seconds","encoder_preparations","feature_cache_hits")} for s in sessions],
        "sdf_cpu_compute_seconds":compute_seconds,"sdf_generation_save_audit_seconds":reference_total,
        "large_family_work_time_ratio":timing_ratio,"same_crop_controls":controls,"supplemental_results":supplements,
        "production_quality":quality,"outer_rim_robustness":rim,
        "matched_family_timing":authority.get("timing",{}).get("oversized_median_tracker_seconds"),
        "sources":[{"path":name,"sha256":file_sha(root/name)} for name in sources if (root/name).exists()]}
    write_json(output.with_suffix(".metrics.json"),summary)
    whole_domain = next((row for row in supplements if row["kind"] == "whole_crop_semantic_foreground"),None)
    lead = "Actual paired tracker results on the same complete native endpoint families, with a new matched SDF reference."
    if whole_domain:
        a,b = whole_domain["methods"]["sam_whole_radius3_union"],whole_domain["methods"]["sam_tiles_radius3_union"]
        local_a,local_b=row_lookup["region01_case02"]["methods"]["sam_whole_radius3_union"],row_lookup["region01_case02"]["methods"]["sam_tiles_radius3_union"]
        lead = f"Independent tiles improved boundary F1 across the oversized fixed crop from {a['boundary_f1']['f1']*100:.2f}% to {b['boundary_f1']['f1']*100:.2f}% (+{(b['boundary_f1']['f1']-a['boundary_f1']['f1'])*100:.2f} points), and its semantic IoU from {a['iou']*100:.4f}% to {b['iou']*100:.4f}%. The original smaller R01 Case 02 ROI moved the other way: IoU {local_a['iou']*100:.4f}% → {local_b['iou']*100:.4f}% and boundary F1 {local_a['boundary_f1']['f1']*100:.2f}% → {local_b['boundary_f1']['f1']*100:.2f}%. Extra inference therefore buys some wider-crop boundary detail, rather than a uniform accuracy gain."
    matched_timing = authority.get("timing",{}).get("oversized_median_tracker_seconds",{})
    median_timing_html = ''
    if matched_timing:
        ratio=matched_timing["independent_tiles"]/matched_timing["whole_crop"]
        median_timing_html=f'<p>Across both samples of the same oversized family, median worker tracker time was <strong>{matched_timing["whole_crop"]:.3f}s → {matched_timing["independent_tiles"]:.3f}s ({ratio:.2f}×)</strong>; the tile strategy makes six independent sessions instead of two and 45 encoder preparations instead of 15. This is the matched model-work comparison.</p>'
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paired native SAM crop strategies</title><style>
    body{{font:16px/1.55 system-ui,sans-serif;color:#223746;background:#f3f6f8;margin:0}}main{{max-width:1150px;margin:30px auto;padding:24px;background:white}}h1{{font-size:30px;line-height:1.2}}h2{{margin-top:30px}}.note{{background:#eef5f7;padding:16px;border-left:4px solid #23858a}}.small{{font-size:14px;color:#596d78}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce5eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f6}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}a{{color:#176f89}}figure{{margin:22px 0}}figcaption{{font-size:14px;color:#596d78}}details{{margin:20px 0}}code{{font-size:13px}}
    </style></head><body><main><h1>Whole-crop resize versus independent tiled SAM</h1><p>{lead}</p>{median_timing_html}<p>The oversized R01 Case 02 family used a {y1-y0} × {x1-x0} crop and three independent tiles. The eight smaller families are exact same-crop controls; their masks and tracker object probabilities matched across strategies.</p>
    <div class="note">This is a follow-up on already seen frame 61 / original scan frame 655, not blind or independent-patient validation. These are research tracker predictions with global radius-3 filtering, not production conservative-v2 selected bridges. The old artificially clipped SDF/SAM observations are not reused as the current baseline.</div>
    <h2>Matched methods and inputs</h2><p>All methods use the frozen full-native detector components at frames 54 and 68, within the same whole-family crop plus 24px context. Whole-crop SAM uses the standard model resize. Independent tiles retain at most 1008 native pixels per axis, 128px interior halos, fixed overlap-midpoint ownership, and no cross-tile propagation or generated seeds. Tiles receive no image context outside the shared whole crop. Native assembly occurs per original seed run before radius filtering and hypothesis union.</p><p>Interpolation distance 15, search angle 30°, minimum radius 3, one pass and candidate budget 1. Walk-back remains 1 but is exhausted because adjacent observations were not supplied. SDF uses those same full-native endpoint masks with production legacy geometry; the compute crop preserves every supplied foreground pixel.</p>
    <h2>Six fixed ROI results</h2>{table(['Case','Method','IoU','Precision','Recall','Boundary F1 @2px','TP','FP','FN'],metric_rows)}<p class="small">Boundary contours are computed on full-native masks before restricting the measurement ROI; matching may use contour points within 2 native pixels outside the ROI. Extra tiled detail is judged against this annotation, not merely foreground area. All six original ROIs remain included.</p>
    <details><summary>Unfiltered raw tracker comparison</summary>{table(['Case','Raw method','IoU','Precision','Recall','Boundary F1 @2px'],raw_rows)}</details>
    <h2>Measured cost and inference counts</h2>{table(['Repeat','Strategy','Families','Sessions','Startup s','Tracking/render/transfer s','Worker tracker s','Encoder calls','Feature hits'],timing_rows)}<p>On the repeated oversized family, three tiles took {large_tiles['tracking_render_transfer_wall_seconds']:.3f}s versus {large_whole['tracking_render_transfer_wall_seconds']:.3f}s for whole-crop resize ({timing_ratio:.2f}×). Startup is separate. The first complete nine-family pass and second oversized-family-only pass are different workloads and are not pooled.</p><p>SDF's serial CPU compute totaled {compute_seconds:.3f}s; generation, compression and connection audits totaled {reference_total:.3f}s. This is a CPU reference measurement, not GPU model-only timing or a production throughput claim. Cache state was reset between SAM strategies/repeats; reuse occurred only within a strategy.</p>
    <h2>Side-by-side native images</h2><figure><a href="{escape(relative(figures/'roi_comparison.png'))}"><img src="{escape(relative(figures/'roi_comparison.png'))}" alt="Native ROI images and error overlays for new matched SDF, whole-crop SAM and independent tiles"></a><figcaption>Yellow annotation; green true positive; pink false positive; cyan false negative. Click for full resolution.</figcaption></figure>
    <h2>Full oversized crop and seam behavior</h2><figure><a href="{escape(relative(figures/'large_family_comparison.png'))}"><img src="{escape(relative(figures/'large_family_comparison.png'))}" alt="Full fixed large family crop, tile footprints, ownership seams and prediction error overlays"></a><figcaption>The original R01 Case 02 ROI is inside the middle ownership core and does not intersect a seam. Green = true positive, magenta = false positive, cyan = false negative. Full-crop comparison is shown separately; semantic foreground can include unrelated annotated objects.</figcaption></figure>{supplement_html}{rim_html}<figure><a href="{escape(relative(figures/'seam_zooms.png'))}"><img src="{escape(relative(figures/'seam_zooms.png'))}" alt="Fixed geometry zooms around the two native ownership seams"></a><figcaption>White dashed line is the ownership seam. Zooms use fixed crop geometry, rather than picking locations from the annotation or prediction errors.</figcaption></figure>
    <h2>Continuity, coverage and production quality</h2><p>Full-ROI metrics count unavailable model/owner-domain pixels as absent prediction. Coverage also includes ROI pixels outside all declared family crops; it is not an empty-seed rate. No empty tile seeds occurred here. Missing seeds would remain unknown rather than successful predicted background. Complete raw halos are retained for leakage diagnosis. Endpoint connection and availability records are retained per family; raster connectivity does not establish biological identity.</p>{quality_html}
    <h2>Evidence and limits</h2><p>Same scan, one labeled plane; nine unique families, six possibly overlapping review ROIs, and only one genuinely oversized family. No accuracy default or production module changed. Actual timings use one local 4090 Laptop eGPU; no H100 or production-scale claim.</p><p>{' · '.join(link(name,name) for name in sources if (root/name).exists())}</p><p>{link(output.with_suffix('.metrics.json').name,'Concise report metrics')}</p><p class="small">Generated {escape(summary['generated_utc'])}. Reports and figures are generated in Scratch; source files, seed masks and frozen plans are not modified by this report.</p></main></body></html>'''
    output.write_text(document,encoding="utf-8")
    print(json.dumps({"html":str(output),"metrics":str(output.with_suffix('.metrics.json')),"sources":len(summary['sources'])},indent=2))


def two_tile_report(args):
    """Report the completed two-tile follow-up without modifying prior artifacts."""
    import html
    import statistics
    os.environ.setdefault("MPLCONFIGDIR", str(args.experiment / "matplotlib_cache"))
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    root = args.experiment
    comparison = json.loads((root/"comparison.json").read_text(encoding="utf-8"))
    plan = json.loads((root/"strategy_plan.json").read_text(encoding="utf-8"))
    reference_root = Path(comparison["baseline_analysis_file"]).parent
    reference = json.loads((reference_root/"native_crop_authority.json").read_text(encoding="utf-8"))
    inference = json.loads((root/"inference_report.json").read_text(encoding="utf-8"))
    if comparison["variant_plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Two-tile results do not match their frozen plan")
    output = (args.output or root/"two_tile_report.html").resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    figures = output.parent/"threeway_figures";figures.mkdir(exist_ok=True)
    escape = lambda value: html.escape(str(value))
    fmt = lambda value: "—" if value is None else f"{value:.4f}"
    methods = [("sdf","Retained matched SDF"),("sam_whole_radius3_union","Whole crop"),
               ("sam_two_radius3_union","Two tiles · 1260px"),("sam_tiles_radius3_union","Three tiles · 1008px")]
    def link(path,label):
        return '<a href="'+escape(os.path.relpath(path,output.parent).replace("\\","/"))+'">'+escape(label)+'</a>'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def metric_rows(domains):
        rows=[]
        for domain in domains:
            for method,label in methods:
                value=domain["methods"][method]
                rows.append([escape(domain["case"]),escape(label),fmt(value["iou"]),fmt(value["precision"]),fmt(value["recall"]),
                    fmt(value["boundary_f1"]["f1"]),f"{value['fp']:,}",f"{value['fn']:,}"])
        return rows
    two_seconds = statistics.median(s["worker_tracker_seconds"] for s in inference["sessions"])
    old_times = reference["timing"]["oversized_median_tracker_seconds"]
    timing_rows = [["Whole crop (retained)","2","15",f"{old_times['whole_crop']:.3f}"],
                  ["Two tiles (new)","4","30",f"{two_seconds:.3f}"],
                  ["Three tiles (retained)","6","45",f"{old_times['independent_tiles']:.3f}"]]
    supplement = comparison["supplemental_results"]
    whole = next(row for row in supplement if row["kind"] == "whole_crop_semantic_foreground")
    primary = next(row for row in comparison["roi_results"] if row["case"] == "region01_case02")
    a,b,c = [whole["methods"][method] for method,_ in methods[1:]]
    pa,pb,pc = [primary["methods"][method] for method,_ in methods[1:]]
    family = next(f for f in plan["families"] if f["family_id"] == comparison["variant_family_id"])
    y0,x0,y1,x1=map(int,family["whole_crop_bbox_yx"])
    source=np.memmap(plan["source_images"],mode="r",dtype=np.uint8,shape=tuple(plan["source_shape_tyx"]))
    frame=int(comparison["evaluation_frame_native"])
    image=source[frame-int(plan["source_frame_start"])]
    if file_sha(comparison["label"]) != comparison["label_sha256"]:
        raise ValueError("Scored annotation changed")
    truth_image=Image.new("1",tuple(plan["source_shape_yx"])[::-1],0);draw=ImageDraw.Draw(truth_image)
    for line in Path(comparison["label"]).read_text(encoding="utf-8").splitlines():
        parts=line.split()
        if len(parts)<7:continue
        values=list(map(float,parts[1:]));draw.polygon([(values[i]*plan["source_shape_yx"][1],values[i+1]*plan["source_shape_yx"][0]) for i in range(0,len(values),2)],fill=1)
    truth=np.asarray(truth_image,bool)
    with np.load(comparison["full_native_prediction_file"],allow_pickle=False) as packet:
        predictions={method:np.asarray(packet[method],bool) for method,_ in methods}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,axes=plt.subplots(5,1,figsize=(13,12))
    axes[0].imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
    axes[0].contour(truth[y0:y1,x0:x1],levels=[.5],colors=["#ffd84a"],linewidths=.6)
    for tile in family["tile_strategy"]["tiles"]:
        ty0,tx0,ty1,tx1=tile["crop_bbox_yx"]
        axes[0].add_patch(Rectangle((tx0-x0,ty0-y0),tx1-tx0,ty1-ty0,fill=False,color="#7ec9e3",linewidth=1.4))
    axes[0].axvline(1835-x0,color="white",linestyle="--",linewidth=1.4)
    for seam in (1683,2211):axes[0].axvline(seam-x0,color="#a4a4a4",linestyle=":",linewidth=1.5)
    axes[0].set_title("New 1260px footprints blue · two-tile seam white · retained three-tile seams grey · annotation yellow")
    for axis,(method,label) in zip(axes[1:],methods):
        axis.imshow(image[y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255)
        p,g=predictions[method][y0:y1,x0:x1],truth[y0:y1,x0:x1]
        rgba=np.zeros((*g.shape,4),np.float32);rgba[p&g]=(.15,.82,.43,.55);rgba[p&~g]=(1.,.22,.52,.85);rgba[~p&g]=(.12,.77,1.,.75)
        axis.imshow(rgba);axis.set_title(label)
    for axis in axes:axis.set_xticks([]);axis.set_yticks([])
    fig.legend(handles=[Patch(color="#26d16e",label="True positive"),Patch(color="#ff3885",label="False positive"),Patch(color="#1fc4ff",label="False negative")],loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.03,1,1));figure=figures/"threeway_fullcrop.png";fig.savefig(figure,dpi=160,bbox_inches="tight");plt.close(fig)
    sources=[root/"comparison.json",root/"strategy_plan.json",root/"inference_report.json",root/"quality_analysis.json",root/"quality_comparison.json",root/"two_tile_authority.json",root/"independent_raw_audit.json",root/"outer_rim_boundary_check.json",reference_root/"native_crop_authority.json",reference_root/"sdf_reference/results.json"]
    summary={"schema":"xta.sam_two_tile_report/1","generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant_plan_sha256":plan["plan_sha256"],"methods":methods,"roi_results":comparison["roi_results"],"supplemental_results":supplement,
        "median_tracker_seconds":{"whole":old_times["whole_crop"],"two_tiles":two_seconds,"three_tiles":old_times["independent_tiles"]},
        "inputs":"Eight unchanged family predictions inherited and OR-composed with new large-family outputs; no subtraction, detector rerun, SDF rerun or blind-validation claim",
        "sources":[{"path":os.path.relpath(p,root).replace("\\","/"),"sha256":file_sha(p)} for p in sources if p.exists()]}
    write_json(output.with_suffix(".metrics.json"),summary)
    footprints=table(['Tile','Native y0,x0,y1,x1','Owner y0,x0,y1,x1'],[[escape(t["tile_id"]),escape(t["crop_bbox_yx"]),escape(t["ownership_bbox_yx"])] for t in family["tile_strategy"]["tiles"]])
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-tile intermediate zoom comparison</title><style>body{{font:16px/1.55 system-ui,sans-serif;background:#f3f6f8;color:#223746}}main{{max-width:1120px;margin:28px auto;padding:26px;background:white}}h1{{font-size:30px}}h2{{margin-top:28px}}.note{{background:#eef5f7;padding:16px;border-left:4px solid #23858a}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce5eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f6}}img{{max-width:100%}}a{{color:#176f89}}.small,figcaption{{font-size:14px;color:#596d78}}</style></head><body><main>
    <h1>Two tiles at 1.25× native width: the measured middle option</h1><p>Two tiles reduced matched tracker time by {(1-two_seconds/old_times['independent_tiles'])*100:.1f}% versus three tiles ({two_seconds:.3f}s versus {old_times['independent_tiles']:.3f}s). Full-crop boundary F1 was {b['boundary_f1']['f1']*100:.2f}%, close to whole-crop {a['boundary_f1']['f1']*100:.2f}% and below three-tile {c['boundary_f1']['f1']*100:.2f}%. Its semantic IoU was {b['iou']*100:.4f}% versus {a['iou']*100:.4f}% whole and {c['iou']*100:.4f}% three tiles.</p>
    <p>The original smaller R01 Case 02 ROI has a different tradeoff: boundary F1 {pa['boundary_f1']['f1']*100:.2f}% / {pb['boundary_f1']['f1']*100:.2f}% / {pc['boundary_f1']['f1']*100:.2f}% for whole / two / three. All original ROIs and all old/new seam domains are retained below.</p><p class="small">A post-hoc check using the same fixed 2px outer-context inset gives boundary F1 83.17% / 83.04% / 86.20% for whole / two / three. The tiny two-tile full-crop gain does not persist after excluding that rim; whole and two are essentially similar here, while three retains its wider boundary improvement. Primary metrics and inference are unchanged.</p>
    <div class="note">Same seen annotation at native frame 61 / source frame 655, one oversized family. The eight unchanged small-family predictions and matched SDF are inherited from the prior run and combined by OR with the new family outputs. This is a research tracker follow-up, not blind validation, a production acceptance receipt, or a default change. The prior report is unchanged.</div>
    <h2>Native footprints and partial downsampling</h2><p>The shared 659×2065 crop is unchanged. Each new footprint is 659×1260; its width is resized to model width 1008, a 1.25× native-width reduction. The vertical extent remains the same in all methods. The two footprints overlap by 455px and fixed ownership meets at x=1835. No cross-tile propagation or new seeds are used; each session receives its original partial endpoint mask.</p>{footprints}
    <h2>Matched work and inference counts</h2>{table(['Strategy','Sessions per repeat','Encoder calls','Median worker tracker s'],timing_rows)}<p class="small">The new method has two measured repeats, four independent sessions and 60 raw frames per repeat. Startup (~8.8s) is separate; previous whole/three timing samples are retained. Cache state was reset per strategy/repeat. This is local single-eGPU work, not total pipeline or target-system throughput.</p>
    <h2>Full crop and fixed old/new seam strips</h2>{table(['Domain','Method','IoU','Precision','Recall','Boundary F1 @2px','FP','FN'],metric_rows(supplement))}<p class="small">Contours use the full-native plane before ROI restriction. Whole-crop semantic truth may include other untracked objects, so this is not isolated instance accuracy. Supplemental domains are post-hoc fixed geometry, not new inference or primary ROI tuning.</p>
    <figure><a href="{escape(os.path.relpath(figure,output.parent).replace(chr(92),'/'))}"><img src="{escape(os.path.relpath(figure,output.parent).replace(chr(92),'/'))}" alt="Same full crop comparing retained SDF, whole resize, new two tiles and retained three tiles"></a><figcaption>Combined global predictions include the same eight retained families. Green = true positive, magenta = false positive, cyan = false negative. Click for full resolution.</figcaption></figure>
    <details><summary>All six fixed original ROIs</summary>{table(['Case','Method','IoU','Precision','Recall','Boundary F1 @2px','FP','FN'],metric_rows(comparison['roi_results']))}</details>
    <h2>Quality remains a separate constraint</h2><p>Shared outer-context contacts still survive radius filtering in the main component (new two-tile forward 1,530; backward 2,370 cumulative foreground-frame contacts). Local 26-connected endpoint links survive, but the intermediate extent remains context limited. No stock-v2 receipt exists. The prior topology resource estimate exceeds the default 256 MiB budget; the diagnostic audit uses a bounded 512 MiB research limit. These are not accepted production bridges.</p>
    <p>{' · '.join(link(p,p.name) for p in sources if p.exists())} · {link(output.with_suffix('.metrics.json'),'Report metrics')}</p><p class="small">No new detector or SDF run, no production modules changed, no general accuracy claim from one scan/plane. Generated {escape(summary['generated_utc'])}.</p></main></body></html>'''
    output.write_text(document,encoding="utf-8")
    print(json.dumps({"html":str(output),"metrics":str(output.with_suffix('.metrics.json')),"methods":4},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sdf-reference", "report", "two-tile-report"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "sdf-reference":
        sdf_reference(args)
    elif args.stage == "two-tile-report":
        two_tile_report(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
