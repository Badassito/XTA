"""Freeze the requested 1260-native-pixel two-tile follow-up before inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sam_crop_strategy_geometry import _hash, tile_plan
from tools.compare_sam_crop_strategies import original_seed

FAMILY = "family_bd7b8fcafd364a36"

def prepare_variant(prior_plan, prior_path):
    plan = copy.deepcopy(prior_plan)
    family = next(f for f in plan["families"] if f["family_id"] == FAMILY)
    if family["whole_crop_bbox_yx"] != [831, 803, 1490, 2868]:
        raise ValueError("Requested original whole-family context changed")
    family["tile_strategy"] = tile_plan(family["whole_crop_bbox_yx"], maximum=1260, halo=128)
    tiles = family["tile_strategy"]["tiles"]
    if len(tiles) != 2:
        raise ValueError("Frozen geometry helper did not produce exactly two tiles")
    y0, x0, y1, x1 = family["whole_crop_bbox_yx"]
    for run in family["runs"]:
        seeds = []
        for tile in tiles:
            path = Path(plan["seed_directory"]) / (run["seed_observation_id"] + ".npz")
            foreground = int(original_seed(path, tile["crop_bbox_yx"]).sum())
            seeds.append(dict(tile_id=tile["tile_id"], seed_foreground=foreground,
                status="independently_original_seeded" if foreground else "unavailable_empty_original_seed",
                ownership_bbox_yx=tile["ownership_bbox_yx"]))
        run["tile_seeds"] = seeds
        run["available_owner_pixels"] = sum((s["ownership_bbox_yx"][2]-s["ownership_bbox_yx"][0]) *
            (s["ownership_bbox_yx"][3]-s["ownership_bbox_yx"][1]) for s in seeds if s["seed_foreground"])
        run["available_owner_fraction"] = run["available_owner_pixels"] / ((y1-y0)*(x1-x0))
    plan["families"] = [family]
    plan["settings"]["tile_max_native_side"] = 1260
    plan["protocol"]["controls"] = "No control reruns; retain previous identical-crop controls and whole/three-tile evidence."
    plan["follow_up"] = dict(parent_plan_path=str(prior_path), parent_plan_sha256=prior_plan["plan_sha256"],
        native_tile_side=1260, native_tile_zoom=1.25, nominal_model_image_side=1008,
        native_tile_shape_yx=[659,1260], model_image_scale_xy=[0.8,1008/659],
        zoom_scope="Horizontal zoom-out within unchanged shared family crop; per-tile context increases, total available image context unchanged.",
        same_original_family_context=True, labels_used_for_geometry=False, predictions_used_for_geometry=False,
        requested_variant="Two native 659x1260 footprints; existing SAM loader still resizes model images to1008.")
    plan["plan_sha256"] = _hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    return plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prior = json.loads(args.prior_plan.read_text(encoding="utf-8"))
    geometry = Path(__file__).with_name("sam_crop_strategy_geometry.py")
    if hashlib.sha256(geometry.read_bytes()).hexdigest() != prior["geometry_source_sha256"]:
        raise ValueError("Original frozen geometry implementation changed")
    if _hash({key: value for key, value in prior.items() if key != "plan_sha256"}) != prior["plan_sha256"]:
        raise ValueError("Original plan fingerprint changed")
    plan = prepare_variant(prior, args.prior_plan.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strategy_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    protocol = {"plan_sha256": plan["plan_sha256"], "follow_up": plan["follow_up"],
                "family": plan["families"][0], "settings": plan["settings"], "protocol": plan["protocol"]}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "tiles": plan["families"][0]["tile_strategy"]["tiles"]}, indent=2))

if __name__ == "__main__":
    main()
