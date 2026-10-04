"""Build bounded group-local SDF references from retained SAM observations.

This is a CPU-only geometric diagnostic. Original observations outside each
retained crop are unavailable, so it does not claim whole-production SDF parity.
Saved working-canvas and native/source transforms are recorded separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from unittest import mock

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0,str(REPOSITORY))

from XTA.sam_evidence import SamEvidenceBundle, _plain
from tools.evaluate_sam_reference_quality import sha


def reconstruct_group_observations(bundle,group):
    frames = list(map(int,group["frame_indices"]))
    if frames != list(range(frames[0],frames[-1]+1)):
        raise ValueError("Group-local SDF requires a consecutive source frame interval")
    y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
    observations = np.zeros((len(frames),y1-y0,x1-x0),np.uint8)
    for index,frame in enumerate(frames):
        for kind in ("known_foreground","unrelated"):
            key = f"{kind}:{frame}"
            if key not in group["mask_keys"]:
                raise ValueError("Group-local SDF requires exact known and unrelated observations")
            observations[index] |= bundle.group_mask(group["group_id"],key)
    for endpoint in group["endpoints"]:
        mask = bundle.group_mask(group["group_id"],f"endpoint:{endpoint['observation_id']}")
        if np.any(mask & ~(observations[frames.index(int(endpoint["frame_index"]))]!=0)):
            raise ValueError("Group endpoint differs from reconstructed original observations")
    return frames,observations


def generate(evidence,output,*,max_group_voxels=8_000_000,workers=1,max_distance=15,search_angle=30.,walk_back=0):
    for key in ("YOLO_TTA_GPU_INTERPOLATION","YOLO_TTA_GPU_SLICE_LABELING",
        "YOLO_TTA_GPU_SLICE_LABELING_PAIRS","YOLO_TTA_GPU_SLICE_LABELING_IN_CHILDREN",
        "YOLO_TTA_GPU_INTERPOLATION_RADIUS","YOLO_TTA_GPU_INTERPOLATION_REQUIRED",
        "YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER"):
        os.environ[key] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from XTA import interpolation
    interpolation.cv2.setNumThreads(1)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    bundle = SamEvidenceBundle.open(evidence)
    report = dict(schema="xta.sam_group_local_sdf/1",evidence_path=str(Path(evidence).resolve()),
        evidence_fingerprint=bundle.evidence_fingerprint,tool_sha256=sha(__file__),
        sdf_source_sha256=sha(interpolation.__file__),max_group_voxels=max_group_voxels,
        coordinate_domain="Retained SAM working-canvas crops; no native/source reprojection",
        canvas_shape_tyx=_plain(bundle.scope.get("shape_tyx")),
        canvas_transform=_plain(bundle.scope.get("canvas_transform",{})),
        interpretation="Same retained crop observations; original context outside the crop unavailable; not whole-production SDF parity or independent truth",
        settings=dict(max_slice_distance=max_distance,search_angle_deg=search_angle,walk_back=walk_back,
            interpolation_candidates=1,passes=1,forced_cpu=True,workers=workers),groups=[],skipped_groups=[])
    with bundle.reader() as reader:
        for identity,group in sorted(bundle.groups.items()):
            y0,x0,y1,x1 = map(int,group["context_bbox_yx"])
            voxels = len(group["frame_indices"])*(y1-y0)*(x1-x0)
            if voxels>max_group_voxels or group.get("status","planned")!="planned":
                report["skipped_groups"].append(dict(group_id=identity,voxels=voxels,
                    reason="bounded workspace limit" if voxels>max_group_voxels else "unpublished input contracts"))
                continue
            frames,original = reconstruct_group_observations(reader,group)
            prediction = original.copy()
            destination = output/identity
            destination.mkdir(exist_ok=True)
            threshold = float(group["interpolation_min_radius"])
            with mock.patch.object(interpolation,"create_cuda_interpolation_renderer",return_value=(None,"forced CPU group reference")):
                statistics = interpolation.interpolate_view_volume_pass_inplace(prediction,destination/"work","group_reference",
                    max_distance,search_angle,walk_back,1,threshold,keep_temp=False,prefer_memory=True,reserve_bytes=0,workers=workers)
            if np.any((original!=0)&(prediction==0)):
                raise ValueError("SDF removed a retained original observation")
            if any("cuda" in str(statistics.get(key,"")).lower() for key in ("interpolation_render_backend","interpolation_radius_backend")):
                raise RuntimeError("Unexpected GPU SDF reference path")
            additions = (prediction!=0)&(original==0)
            packet = destination/"reference.npz"
            np.savez_compressed(packet,shape_tyx=np.asarray(original.shape,np.int64),frames=np.asarray(frames,np.int64),
                bbox_yx=np.asarray(group["context_bbox_yx"],np.int64),
                original=np.packbits(original.reshape(-1),bitorder="little"),
                additions=np.packbits(additions.reshape(-1),bitorder="little"))
            report["groups"].append(dict(group_id=identity,reference_file=str(packet.resolve()),reference_sha256=sha(packet),
                stored_frame_start=frames[0],full_source_frame_start=frames[0]+int(bundle.scope.get("source_frame_start",0)),
                shape_tyx=list(original.shape),bbox_yx=list(group["context_bbox_yx"]),
                original_foreground=int(np.count_nonzero(original)),added_foreground=int(additions.sum()),
                interpolation_min_radius=threshold,statistics=statistics))
            print(f"SDF reference {identity}: {int(additions.sum())} added pixels",flush=True)
    bundle.assert_unchanged()
    (output/"group_references.json").write_text(json.dumps(report,indent=2),"utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-group-voxels",type=int,default=8_000_000)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--max-distance",type=int,default=15)
    parser.add_argument("--search-angle",type=float,default=30.)
    parser.add_argument("--walk-back",type=int,default=0)
    args = parser.parse_args()
    generate(args.evidence,args.output,max_group_voxels=args.max_group_voxels,workers=args.workers,
        max_distance=args.max_distance,search_angle=args.search_angle,walk_back=args.walk_back)


if __name__ == "__main__":
    main()
