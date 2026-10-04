"""Plot frozen SDF/old-SAM/new-SAM/annotation masks in one common native ROI.

Unknown SAM generation domains are shaded and striped. No predictions are
generated or selected here; every plotted mask and annotation is hash-checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0,str(REPOSITORY))

from tools.evaluate_sam_reference_quality import sha
from tools.analyze_sam_crop_strategies import load_truth


def frozen_plane(report,method,frame):
    row = next(row for row in report["frozen_predictions"] if row["frame_index"]==frame)
    if sha(row["file"])!=row["sha256"]:
        raise ValueError("Frozen evaluation predictions changed")
    with np.load(row["file"],allow_pickle=False) as saved:
        shape = tuple(map(int,saved["shape_yx"]))
        plane = np.unpackbits(saved[method],count=int(np.prod(shape)),bitorder="little").reshape(shape).astype(bool)
        domain = np.unpackbits(saved["domain"],count=int(np.prod(shape)),bitorder="little").reshape(shape).astype(bool)
    return plane,domain


def plot(reports,output,*,before_reports=None,new_method="anchor_context_censored",old_method="original_stock",roi=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    rows = []
    shape = None
    for name,path in reports.items():
        report = json.loads(Path(path).read_text("utf-8"))
        label = report["manual_labels"][0]
        frame = int(label["frame_index"])
        if sha(label["file"])!=label["sha256"]:
            raise ValueError("Evaluation annotation changed")
        new,new_domain = frozen_plane(report,new_method,frame)
        sdf,_ = frozen_plane(report,"sdf",frame)
        old_report_path = Path(before_reports[name]) if before_reports and name in before_reports else Path(path)
        old_report = json.loads(old_report_path.read_text("utf-8"))
        old,old_domain = frozen_plane(old_report,old_method,frame)
        current_shape = new.shape
        if shape is None:
            shape = current_shape
        if current_shape!=shape or old.shape!=shape or sdf.shape!=shape:
            raise ValueError("Common-ROI plotting requires identical native source shapes")
        truth = load_truth(label["file"],shape)
        source_paths = list(report["source_file_sha256"])
        original = next((Path(path) for path in source_paths if "observation" in Path(path).name.lower()),None)
        if original is None:
            raise ValueError("Plot requires the authenticated original observation volume")
        if sha(original)!=report["source_file_sha256"][str(original)]:
            raise ValueError("Original observation volume changed")
        observed = np.load(original,mmap_mode="r",allow_pickle=False)[frame]!=0
        truth = truth & ~observed
        rows.append(dict(name=name,source_frame=frame+report["source_frame_start"],
            masks=(sdf,old,new,truth),domains=(np.ones(shape,bool),old_domain,new_domain,np.ones(shape,bool)),
            report_path=str(Path(path).resolve()),report_sha256=sha(path),
            before_report_path=str(old_report_path.resolve()),before_report_sha256=sha(old_report_path),
            annotation_sha256=label["sha256"]))
    if roi is None:
        roi = (0,0,shape[1],shape[0])
    x0,y0,x1,y1 = map(int,roi)
    if not (0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):
        raise ValueError("Common ROI is outside the saved native source")
    height,width = y1-y0,x1-x0
    yy,xx = np.ogrid[:height,:width]
    stripes = ((xx+yy)%48)<3
    fig,axes = plt.subplots(len(rows),4,figsize=(13,3.8*len(rows)),layout="constrained",squeeze=False)
    titles = ("SDF reference","Previous selected SAM","New selected SAM","Existing annotation")
    for index,row in enumerate(rows):
        for column,(mask,domain) in enumerate(zip(row["masks"],row["domains"])):
            value = np.zeros((height,width),np.uint8)
            unknown = ~domain[y0:y1,x0:x1]
            value[unknown] = 1
            value[unknown & stripes] = 2
            value[mask[y0:y1,x0:x1]] = 3
            ax = axes[index,column]
            ax.imshow(value,cmap=ListedColormap(["#f7f8fa","#ece5d8","#c7bdad","#273d56"]),vmin=0,vmax=3,interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if index==0:
                ax.set_title(titles[column],fontsize=12)
            if column==0:
                ax.set_ylabel(f"{row['name']}\nsource slice {row['source_frame']}",fontsize=11)
    fig.suptitle(f"SAM bridge additions before and after selection changes\nCommon native ROI ({x0}, {y0})–({x1}, {y1}); original observations excluded",fontsize=14)
    fig.legend(handles=[Patch(facecolor="#273d56",label="Bridge / annotated foreground"),
        Patch(facecolor="#ece5d8",edgecolor="#c7bdad",hatch="///",label="Unknown or unattempted SAM generation domain")],
        loc="outside lower center",ncol=2,fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)
    metadata = dict(common_roi_xyxy=list(roi),shape_yx=list(shape),new_method=new_method,old_method=old_method,
        tool_sha256=sha(__file__),figure_file=str(output.resolve()),figure_sha256=sha(output),
        datasets=[{key:value for key,value in row.items() if key not in {"masks","domains"}} for row in rows])
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2),"utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",action="append",required=True,help="Dataset NAME=QUALITY_JSON")
    parser.add_argument("--before-report",action="append",default=[],help="Optional earlier-domain NAME=QUALITY_JSON")
    parser.add_argument("--new-method",default="anchor_context_censored")
    parser.add_argument("--old-method",default="original_stock")
    parser.add_argument("--roi",type=int,nargs=4)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    reports = dict(value.split("=",1) for value in args.report)
    before = dict(value.split("=",1) for value in args.before_report)
    plot(reports,args.output,before_reports=before,new_method=args.new_method,old_method=args.old_method,roi=args.roi)


if __name__ == "__main__":
    main()
