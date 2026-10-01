"""Pre-register detector-only interpolation cases before reading held-out labels."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import numpy as np
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--detector",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--start",type=int,default=50)
    p.add_argument("--stop",type=int,default=73)
    p.add_argument("--anchors",nargs=2,type=int,default=(54,68))
    p.add_argument("--evaluation-frame",type=int,default=61)
    p.add_argument("--regions",type=int,default=2)
    p.add_argument("--cases-per-region",type=int,default=3)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("diagnostic",Path(__file__).parent/"diagnose_sam_interpolation.py")
    diagnostic=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    tools=Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Environment\tools")
    probe=json.loads(subprocess.run([str(tools/"ffprobe.exe"),"-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","json",str(args.input)],capture_output=True,text=True,check=True).stdout)["streams"][0]
    width,height=int(probe["width"]),int(probe["height"])
    full=args.output/"source_frames.uint8.dat"
    with full.open("wb") as out:
        subprocess.run([str(tools/"ffmpeg.exe"),"-v","error","-i",str(args.input),"-vf",f"select=between(n\\,{args.start}\\,{args.stop-1})","-fps_mode","passthrough","-f","rawvideo","-pix_fmt","gray","-"],stdout=out,check=True)
    images=np.memmap(full,dtype=np.uint8,mode="r",shape=(args.stop-args.start,height,width))
    with diagnostic.gpu_lock(Path(r"C:\Users\Bry\Documents\ChatGPT\Scratch\Temp\GPU_LOCK"),"sam_improvements_blind_detector_holdout"),diagnostic.resource_monitor(args.output/"resources.json",0):
        import torch
        from ultralytics import YOLO
        detector=YOLO(str(args.detector))
        assert int(next(detector.model.parameters()).shape[1])==1
        def predict(image,side):
            h,w=image.shape
            factor=side/max(h,w)
            nh,nw=round(h*factor),round(w*factor)
            resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
            canvas=np.full((side,side),114,np.uint8)
            y,x=(side-nh)//2,(side-nw)//2
            canvas[y:y+nh,x:x+nw]=resized
            tensor=torch.from_numpy(canvas.copy())[None,None].to(0,dtype=torch.float32)/255
            result=detector.predict(tensor,imgsz=side,device=0,conf=.15,retina_masks=True,verbose=False)[0]
            merged=np.zeros((h,w),np.uint8)
            if result.masks is not None:
                for mask in result.masks.data.detach().cpu().numpy():
                    merged|=(cv2.resize(mask[y:y+nh,x:x+nw],(w,h),interpolation=cv2.INTER_NEAREST)>.5).astype(np.uint8)
            return merged
        endpoints=[predict(images[frame-args.start],3072) for frame in args.anchors]
        components=[]
        for mask in endpoints:
            count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
            components.append((stats,centers))
        candidates=[]
        for a in range(1,len(components[0][0])):
            boxa=components[0][0][a]
            if boxa[4]<100 or max(boxa[2],boxa[3])>780:
                continue
            for b in range(1,len(components[1][0])):
                boxb=components[1][0][b]
                distance=float(np.linalg.norm(components[0][1][a]-components[1][1][b]))
                if boxb[4]>=100 and max(boxb[2],boxb[3])<=780 and distance<=150:
                    candidates.append((-min(int(boxa[4]),int(boxb[4])),distance,a,b))
        regions=[]
        for _,distance,a,b in sorted(candidates):
            center=(components[0][1][a]+components[1][1][b])/2
            x0=min(max(0,int(round((center[0]-504)/32)*32)),width-1008)
            y0=min(max(0,int(round((center[1]-504)/32)*32)),height-1008)
            box=(x0,y0,x0+1008,y0+1008)
            if any(max(x0,item[0])<min(x0+1008,item[2]) and max(y0,item[1])<min(y0+1008,item[3]) for item in regions):
                continue
            if not all(x0+40<=record[0] and y0+40<=record[1] and record[0]+record[2]<=x0+968 and record[1]+record[3]<=y0+968 for record in (components[0][0][a],components[1][0][b])):
                continue
            regions.append(box)
            if len(regions)>=args.regions:
                break
        metadata={"schema":"xta.detector_only_holdout/1","source":str(args.input),"source_native_frames":[args.start,args.stop],
                  "source_original_video_frame_offset":594,"anchors_native":args.anchors,"evaluation_frame_native":args.evaluation_frame,
                  "labels_read_during_preparation":False,"region_selection":"Largest matched full-frame detector endpoint components define disjoint 1008-pixel windows with 40px clearance; additional ROI-selected components may meet a window edge and require the separate context audit",
                  "detector":str(args.detector),"detector_sha256":hashlib.sha256(args.detector.read_bytes()).hexdigest(),
                  "full_frame_detector_imgsz":3072,"roi_detector_imgsz":1024,"conf":.15,"command":sys.argv,"regions":[]}
        for index,(x0,y0,x1,y1) in enumerate(regions):
            out=args.output/f"region{index+1:02d}"
            out.mkdir(exist_ok=True)
            native=np.ascontiguousarray(images[:,y0:y1,x0:x1])
            (out/"images.uint8.dat").write_bytes(native.tobytes())
            observations=np.zeros(native.shape,np.uint8)
            for frame,image in enumerate(native):
                observations[frame]=predict(image,1024)
            np.save(out/"detector_observations.npy",observations)
            first,last=[frame-args.start for frame in args.anchors]
            n,labels,stats,centers=cv2.connectedComponentsWithStats(observations[first],connectivity=8)
            n2,labels2,stats2,centers2=cv2.connectedComponentsWithStats(observations[last],connectivity=8)
            matches=[]
            for a in range(1,n):
                if stats[a,4]<100:
                    continue
                for b in range(1,n2):
                    d=float(np.linalg.norm(centers[a]-centers2[b]))
                    if stats2[b,4]>=100 and d<=75:
                        matches.append((d,-int(stats[a,4]),a,b))
            cases=[]
            for distance,_,a,b in sorted(matches):
                center=(centers[a]+centers2[b])/2
                if any(np.linalg.norm(center-np.asarray(row["center_xy"]))<180 for row in cases):
                    continue
                rx0,ry0=max(0,int(center[0]-120)),max(0,int(center[1]-120))
                rx1,ry1=min(1008,int(center[0]+120)+1),min(1008,int(center[1]+120)+1)
                case=diagnostic.select_whole_components(observations,(rx0,ry0,rx1,ry1))
                case[first+1:last]=0
                identifier=f"region{index+1:02d}_case{len(cases)+1:02d}"
                filename=f"case{len(cases)+1:02d}_observations.npy"
                np.save(out/filename,case)
                cases.append({"id":identifier,"observations":filename,"center_xy":center.tolist(),"region_xyxy":[rx0,ry0,rx1,ry1],
                              "anchor_local_frames":[first,last],"anchor_native_frames":args.anchors,"withheld_local_frames":list(range(first+1,last)),
                              "observation_sha256":hashlib.sha256(np.packbits(case!=0).tobytes()).hexdigest(),"case_selection":"Whole components of the ROI detector mask; silhouettes can already be truncated by the 1008-pixel input window, so context validity requires the separate label-free audit"})
                if len(cases)>=args.cases_per_region:
                    break
            inventory={"image_path":"images.uint8.dat","image_shape":list(native.shape),"input":str(args.input),"input_native_frames":[args.start,args.stop],
                       "source_crop_xyxy":[x0,y0,x1,y1],"cases":cases,"planning_labels_used":False,"evaluation_local_frame":args.evaluation_frame-args.start,
                       "detector":str(args.detector),"detector_sha256":metadata["detector_sha256"]}
            (out/"cases.json").write_text(json.dumps(inventory,indent=2))
            metadata["regions"].append({"path":str(out),"crop_xyxy":[x0,y0,x1,y1],"cases":len(cases)})
            print(f"Prepared blind region{index+1}: {len(cases)} cases {x0,y0,x1,y1}",flush=True)
        (args.output/"holdout_preregistration.json").write_text(json.dumps(metadata,indent=2))
        del detector
        torch.cuda.empty_cache()


if __name__=="__main__":
    main()
