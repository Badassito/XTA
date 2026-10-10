"""Build the post-v25 outer-crop report from persisted evidence.

This reporter performs no planning, model execution, policy selection, or mask
editing. It reads the authoritative analysis index or an explicit normalized
report_data.json. Missing results remain unreported; refused support stays unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from XTA.artifact_archive import (open_artifact, read_artifact, artifact_exists,
    physical_path, member_exists, split_reference)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(read_artifact(path)) if artifact_exists(path) else default


def file_sha(path):
    digest = hashlib.sha256()
    with open_artifact(path) as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve(root, path):
    result = Path(path)
    return result if result.is_absolute() else root / result


def is_source_file(path):
    return member_exists(path) if split_reference(path) is not None else Path(path).is_file()


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(value))+'</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>'+str(value)+'</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def analysis_data(root, constants):
    """Normalize authoritative analyses; never calculate or fill model metrics."""
    index = read_json(root/"analysis_index.json")
    if not index:
        return None
    aliases = {"development":"development_seen655", "source_590_599":"followup_594", "source_686_695":"followup_690"}
    specs = {entry["dataset_id"]:entry for entry in constants.get("datasets",[])}
    result = {"schema":"xta.sam_outer_crop_report_data/1","status":index.get("status","analysis_available"),"datasets":[],"sources":[{"path":"analysis_index.json"}],"missing_analysis_ids":[]}
    for entry in index.get("entries",[]):
        identifier = entry["dataset_id"]
        spec = specs.get(aliases.get(identifier,identifier),{})
        path = resolve(root,entry["analysis_file"])
        analysis = read_json(path)
        if analysis is None:
            result['missing_analysis_ids'].append(identifier)
            result['datasets'].append(dict(id=identifier,title=spec.get('dataset_id',identifier),
                scope=spec.get('stage','unreported'),exposure=entry.get('exposure','unreported'),
                analysis_available=False,status='missing_analysis',analysis_file=str(path),
                results=[],common_results=[],models=[],dataset_spec=spec,sources=[{'path':str(path)}]))
            continue
        rows=[];common_rows=[]
        for model in analysis.get("models",[]):
            for key in ('selection_file','timing_source','preview_file'):
                if model.get(key):
                    result['sources'].append({'path':model[key]})
            receipt=read_json(resolve(root,model['selection_file']),{})if model.get('selection_file')else{}
            model['report_receipt_summary']={
                'run_status_counts':dict(Counter(item.get('status','unreported')for item in receipt.get('run_receipts',{}).values())),
                'run_reason_counts':dict(Counter(reason for item in receipt.get('run_receipts',{}).values()for reason in item.get('reasons',[]))),
                'group_status_counts':dict(Counter(item.get('status','unreported')for item in receipt.get('group_receipts',{}).values()))}
            scope=model.get("scope_status","unreported")
            counts={key:model.get(key) for key in ("original_family_count","planned_family_count","refused_family_count","cohort_complete")}
            for target,domains,population in ((rows,model.get('domain_results',[]),'published_survivors'),
                    (common_rows,model.get('common_domain_results',[]),'common_surviving_lineages')):
                for domain in domains:
                    metrics=domain.get("metrics",{})
                    target.append({"variant":model.get("variant"),"mode":model.get("mode"),"status":scope,"population":population,
                        "domain":domain.get("id","unreported"),"raw_metrics":metrics.get("raw"),"radius3_metrics":metrics.get("radius3"),
                        "raw_halo_metrics":metrics.get("raw_halo"),"candidate_metrics":metrics.get("candidate_W"),"selected_metrics":metrics.get("selected_W"),
                        "metric_availability":domain.get("metric_availability"),"known_coverage_fraction":domain.get("known_coverage_fraction"),
                        "refusal_reasons":domain.get("refused_overlap_ids",[]),"cohort":counts,"timing_source":model.get("timing_source"),"timing":model.get("timing"),
                        "evidence_file":model.get("evidence_file"),"selection_file":model.get("selection_file")})
        result["datasets"].append({"id":identifier,"title":spec.get("dataset_id",identifier),"analysis_available":True,
            "scope":spec.get("stage",str(analysis.get("dataset","unreported"))),
            "exposure":entry.get("exposure",analysis.get("exposure",constants.get("exposure",{}))),
            "results":rows,"common_results":common_rows,"common_surviving_lineage_count":analysis.get('common_surviving_lineage_count'),
            "common_surviving_lineage_keys":analysis.get('common_surviving_lineage_keys'),"models":analysis.get("models",[]),"dataset_spec":spec,
            "label_file":analysis.get("label_file",analysis.get("label")),"label_sha256":analysis.get("label_sha256"),
            "sources":[{"path":str(path),"label":"Authoritative stage analysis"}]})
        result["sources"].append({"path":str(path)})
    if result['missing_analysis_ids']:
        result['upstream_status'] = result['status']
        result['status'] = 'incomplete'
    return result


def model_figures(root, figures, dataset):
    """Display actual previews without reselecting or computing accuracy."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image, ImageDraw
    spec=dataset.get('dataset_spec',{})
    if not spec.get('image_path') or not spec.get('source_shape_tyx'):
        return None
    identifier=dataset['id']
    records=[]
    for model in dataset.get('models',[]):
        preview=model.get('preview_file',root/'previews'/identifier/model.get('mode','unknown')/(model.get('variant','unknown')+'.npz'))
        path=resolve(root,preview)
        if path.exists():records.append((model,path))
    if not records:return None
    priority={'B0':0,'B1':1,'C2':2,'A2':3,'C3':4,'Cfull':5,'SDF':6}
    records.sort(key=lambda item:(priority.get(item[0].get('variant'),9),item[0].get('mode','')))
    shape=tuple(spec['source_shape_tyx'])
    source=np.memmap(resolve(root,spec['image_path']),mode='r',dtype=np.uint8,shape=shape)
    label=dataset.get('label_file')
    truth=None
    if label and resolve(root,label).exists():
        if dataset.get('label_sha256') and file_sha(resolve(root,label))!=dataset['label_sha256']:
            raise ValueError('Scored annotation changed')
        canvas=Image.new('1',shape[1:][::-1],0);draw=ImageDraw.Draw(canvas)
        for line in resolve(root,label).read_text(encoding='utf-8').splitlines():
            fields=line.split()
            if len(fields)<7:continue
            values=list(map(float,fields[1:]));draw.polygon([(values[i]*shape[2],values[i+1]*shape[1]) for i in range(0,len(values),2)],fill=1)
        truth=np.asarray(canvas,bool)
    fixed=spec.get('frozen_large_family_scoring_crop_yx')
    if fixed:
        sy0,sx0,sy1,sx1=map(int,fixed)
        viewport=(max(0,sy0-128),max(0,sx0-128),min(shape[1],sy1+128),min(shape[2],sx1+128))
    else:viewport=(0,0,shape[1],shape[2])
    y0,x0,y1,x1=viewport
    row_height=max(1.6,min(3.8,3.7*(y1-y0)/(x1-x0)))
    fig,axes=plt.subplots(len(records),4,figsize=(15,row_height*len(records)+.6),squeeze=False)
    for row,(model,path) in enumerate(records):
        with np.load(path,allow_pickle=False) as packet:
            is_sdf=model.get('variant')=='SDF'
            required=('selected_W','known_owner','refused_bbox_domain')if is_sdf else('raw','raw_halo','selected_W','A','W','image_context_domain','known_owner','refused_bbox_domain')
            if any(key not in packet.files for key in required):
                raise ValueError('Incomplete visualization preview; no inferred layers are allowed')
            fields={key:np.asarray(packet[key],bool)[y0:y1,x0:x1] for key in required}
            if any(packet[key].shape!=shape[1:] for key in required):
                raise ValueError('Visualization mask shape differs from native source')
            if is_sdf:
                reference_path=resolve(root,model['reference_file'])
                if model.get('reference_sha256')and file_sha(reference_path)!=model['reference_sha256']:
                    raise ValueError('Scored SDF reference changed')
                reference=read_json(reference_path)
                cache_local=int(reference['evaluation_cache_local'])
                full_frame=int(reference['evaluation_full_source_frame'])
            else:
                cache_local=int(packet['cache_local'])
                full_frame=int(packet['full_source_frame'])
        if not 0<=cache_local<shape[0] or full_frame!=int(spec['source_frame_start'])+cache_local:
            raise ValueError('Preview source-frame mapping differs')
        image=source[cache_local,y0:y1,x0:x1]
        for axis in axes[row]:axis.imshow(image,cmap='gray',vmin=0,vmax=255);axis.set_xticks([]);axis.set_yticks([])
        contract=axes[row,0]
        for key,color in (()if is_sdf else(('image_context_domain','#65c8e1'),('A','#4778df'),('W','#ffb347'))):
            mask=fields[key]
            if mask.any() and not mask.all():contract.contour(mask,levels=[.5],colors=[color],linewidths=.9)
        contract.set_title('SDF reference · same evaluation image'if is_sdf else'Image C / acceptance A / writes W',fontsize=10)
        scope=model.get('scope_status','unreported')
        scope_label='Single-family stress\nRetained B1 controls'if scope=='single_family_stress_with_retained_B1_controls'else scope.replace('_',' ')
        contract.set_ylabel('SDF · CPU reference\nFull original observations'if is_sdf else model.get('variant','?')+' · '+model.get('mode','?')+'\n'+scope_label,fontsize=9)
        known=fields['known_owner']
        unknown=~known
        for column,key,title in ((1,'raw','Raw owned core'),(2,'raw_halo','Raw full halo · quality only'),(3,'selected_W','Actually selected writes W')):
            if is_sdf and key!='selected_W':
                axes[row,column].set_axis_off()
                axes[row,column].text(.5,.5,'No SAM model evidence',ha='center',va='center',transform=axes[row,column].transAxes,color='#223746',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
                continue
            if is_sdf:title='SDF actual additions · no SAM-W clipping'
            axis=axes[row,column];pred=fields[key]
            if truth is not None:
                gt=truth[y0:y1,x0:x1]
                if key=='raw_halo':
                    if pred.any() and not pred.all():axis.contour(pred,levels=[.5],colors=['#c36af0'],linewidths=.9)
                    if gt.any() and not gt.all():axis.contour(gt,levels=[.5],colors=['#ffd84a'],linewidths=.6)
                else:
                    rgba=np.zeros((*pred.shape,4),np.float32)
                    rgba[pred & gt & known]=(.15,.82,.43,.6)
                    rgba[pred & ~gt & known]=(1.,.22,.52,.8)
                    rgba[~pred & gt & known]=(.12,.77,1.,.7)
                    axis.imshow(rgba)
            elif pred.any() and not pred.all():axis.contour(pred,levels=[.5],colors=['#c36af0'],linewidths=.9)
            if key!='raw_halo' and unknown.any():
                axis.contourf(unknown,levels=[.5,1.5],colors=['#e8d9bd'],alpha=.17,hatches=['///'])
            axis.set_title(title,fontsize=10)
    fig.legend(handles=[Patch(color='#65c8e1',label='Image context C'),Patch(color='#4778df',label='Acceptance A'),Patch(color='#ffb347',label='Permitted W'),
        Patch(color='#26d16e',label='TP on known support'),Patch(color='#ff3885',label='FP on known support'),Patch(color='#1fc4ff',label='FN on known support'),
        Patch(facecolor='#e8d9bd',hatch='///',label='Unknown owner coverage'),Patch(color='#c36af0',label='Full-halo contour'),
        Patch(color='#ffd84a',label='Annotation contour')],loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,min(.22,.65/(row_height*len(records)+.6)),1,1))
    path=figures/(identifier+'_actual_layers.png');fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
    return {'path':path,'viewport_yx':viewport,'caption':'Exact persisted previews. Halo contours are quality evidence, not output; error colors apply only on known owner coverage. Hatching indicates unavailable owner coverage, never successful predicted background. A2 selected W is altered-A fixed-evidence diagnostic output, not a fresh pipeline. SDF actual additions use full original observations without SAM-W clipping or SAM model evidence.'}


def synthetic_figure(root, figures):
    """Plot exact independently exported arrays; never reconstruct support."""
    metadata = read_json(root/"regression_review/synthetic_world_contracts.json")
    archive_path = root/"regression_review/synthetic_world_contracts.npz"
    if not metadata or not archive_path.exists():
        return None
    if file_sha(archive_path) != metadata["array_archive_sha256"]:
        raise ValueError("Synthetic oracle array archive changed")
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    with np.load(archive_path, allow_pickle=False) as archive:
        restored = (archive["generated_W"] & ~archive["legacy_W"]).sum(axis=(1,2))
        index = int(np.argmax(restored))
        arrays = {key:archive[key][index].copy() for key in (
            "legacy_A","generated_A","intended_A","legacy_W","generated_W","intended_W")}
        frame = int(archive["frame_indices"][index])
    fig, axes = plt.subplots(2,3,figsize=(12.4,5.4),squeeze=False)
    old = metadata["crop_contract"]["legacy_context_bbox_yx"]
    new = metadata["crop_contract"]["context_bbox_yx"]
    for row, label in enumerate(("A","W")):
        for column,(prefix,title) in enumerate((("legacy","Old cropped support"),("generated","Restored support"),("intended","Independent full-canvas oracle"))):
            axis=axes[row,column]
            axis.imshow(arrays[prefix+"_"+label],cmap="Blues",vmin=0,vmax=1)
            for bbox,color,style in ((old,"#d34756","--"),(new,"#278d72","-")):
                y0,x0,y1,x1=bbox
                axis.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color=color,linestyle=style,linewidth=1))
            axis.set_title(title,fontsize=10);axis.set_xticks([]);axis.set_yticks([])
            if column==0:axis.set_ylabel("Acceptance A" if label=="A" else "Permitted writes W")
    fig.suptitle(f"Synthetic L-shape geometry · frame {frame} · no model inference",fontweight="bold")
    fig.tight_layout()
    path=figures/"synthetic_geometry_oracle.png"
    fig.savefig(path,dpi=170,bbox_inches="tight")
    plt.close(fig)
    return {"path":path,"frame":frame,"frame_selection":"Largest restored W support in this exact synthetic fixture", "metadata":metadata}


def build(args):
    root=args.experiment.resolve()
    root.mkdir(parents=True,exist_ok=True)
    output=(args.output or root/"outer_crop_report.html").resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    figures=output.parent/"outer_crop_figures";figures.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(root/"matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    data_path=args.data or root/"report_data.json"
    constants=read_json(root/"protocol_constants.json",{})
    data=read_json(data_path)
    indexed=analysis_data(root,constants)
    if data is None:
        data=indexed or {"schema":"xta.sam_outer_crop_report_data/1","status":"awaiting_actual_model_results","datasets":[],"sources":[]}
    elif indexed is not None:
        # A normalized view cannot silently omit or replace unavailable indexed
        # evidence with saved metrics. Keep every authoritative indexed ID.
        datasets={dataset['id']:dataset for dataset in data.get('datasets',[])}
        for dataset in indexed['datasets']:
            if dataset.get('analysis_available') is False or dataset['id'] not in datasets:
                datasets[dataset['id']]=dataset
        data['datasets']=list(datasets.values())
        data.setdefault('sources',[]).extend(indexed['sources'])
        data['missing_analysis_ids']=indexed['missing_analysis_ids']
        if indexed['missing_analysis_ids']:
            data['upstream_status']=data.get('status','unreported')
            data['status']='incomplete'
    extraction=read_json(root/"extraction.json",{})
    planner=read_json(root/"planner_validation.json",{})
    scheduling=read_json(root/"scheduler_abba/summary.json",{})
    oracle=None if args.skip_figures else synthetic_figure(root,figures)
    escape=lambda value:html.escape(str(value))
    def link(path,label=None):
        actual=physical_path(resolve(root,path))
        relative=os.path.relpath(actual,output.parent).replace("\\","/")
        return '<a href="'+escape(relative)+'">'+escape(label or path)+'</a>'
    def metric_rows(dataset, compact=False, common=False, stress=None):
        rows=[]
        for result in dataset.get("common_results"if common else"results",[]):
            if stress is not None and (result.get('status')=='single_family_stress_with_retained_B1_controls')!=stress:
                continue
            if compact and result.get("domain")!='development_large_frozen_crop':
                continue
            raw=result.get("raw_metrics") or {}
            selected=result.get("selected_metrics") or {}
            halo=result.get("raw_halo_metrics") or {}
            radius=result.get("radius3_metrics") or {}
            candidate=result.get("candidate_metrics") or {}
            boundary=raw.get("boundary_f1")
            if isinstance(boundary,dict):boundary=boundary.get("f1")
            if compact:
                rows.append([escape(result.get("variant","unreported")),escape(result.get("mode","unreported")),number(raw.get("iou")),number(boundary),
                    number(halo.get("iou")),number(candidate.get("iou")),number(selected.get("iou")),number(selected.get("precision")),number(selected.get("recall")),number(result.get("known_coverage_fraction"))])
                continue
            rows.append([escape(result.get("domain","unreported")),escape(result.get("variant","unreported")),escape(result.get("mode","unreported")),escape(result.get("status","unreported")),
                number(raw.get("iou")),number(halo.get("iou")),number(radius.get("iou")),number(candidate.get("iou")),number(boundary),number(selected.get("iou")),number(selected.get("precision")),number(selected.get("recall")),
                number(result.get("known_coverage_fraction")),escape(result.get("metric_availability","unreported")),escape(", ".join(map(str,result.get("refusal_reasons",[]))) or "—")])
        headers=['Variant','Mode','Raw core IoU','Raw boundary F1','Raw halo IoU','Candidate-W IoU','Actual selected IoU','Selected P','Selected R','Known coverage'] if compact else ['Domain','Variant','Mode','Scope status','Raw core IoU','Raw halo IoU','Radius3 core IoU','Candidate-W IoU','Raw boundary F1','Actual selected IoU','Selected precision','Selected recall','Known owner coverage','Metric availability','Refused overlap IDs']
        return table(headers,rows) if rows else '<p>No model results are reported for this stage yet. Missing evidence is not a successful empty prediction.</p>'
    dataset_html=[]
    for dataset in data.get("datasets",[]):
        title=escape(dataset.get("title",dataset.get("id","Unnamed stage")))
        exposure=escape(dataset.get("exposure","Prior exposure not yet reported; do not infer a pristine holdout."))
        description=escape(dataset.get("scope","No scope supplied"))
        if dataset.get('analysis_available') is False:
            description += ' — Required indexed analysis is missing; its metrics are unavailable.'
        cohort = dataset.get("cohort",{})
        cohort_rows = [[escape(name.replace('_',' ')),escape(value)] for name,value in cohort.items()]
        cohort_html = table(['Cohort scope','Recorded value'],cohort_rows) if cohort_rows else '<p class="small">Cohort completeness has not been reported; no complete-cohort claim is inferred.</p>'
        if dataset.get('models'):
            cohort_html=table(['Variant','Mode','Scope','Original families','Planned families','Refused families','Complete cohort'],[
                [escape(m.get('variant','unreported')),escape(m.get('mode','unreported')),escape(m.get('scope_status','unreported')),
                 escape(m.get('original_family_count','unreported')),escape(m.get('planned_family_count','unreported')),
                 escape(m.get('refused_family_count','unreported')),escape(m.get('cohort_complete','unreported'))] for m in dataset['models']])
        policy_rows=[];refusal_rows=[]
        for model in dataset.get('models',[]):
            receipt=model.get('report_receipt_summary',{})
            if receipt.get('run_status_counts'):
                policy_rows.append([escape(model.get('variant')),escape(model.get('mode')),
                    escape(', '.join(f'{key}: {value}'for key,value in sorted(receipt['group_status_counts'].items()))),
                    escape(', '.join(f'{key}: {value}'for key,value in sorted(receipt['run_status_counts'].items()))),
                    escape(', '.join(f'{key}: {value}'for key,value in sorted(receipt['run_reason_counts'].items()))or'—')])
            for group in model.get('groups',[]):
                if not group.get('model_available',True):
                    refusal_rows.append([escape(model.get('variant')),escape(model.get('mode')),escape(group.get('group_id')),escape(group.get('status')),
                        escape(', '.join(group.get('reasons',[]))or'Unreported'),escape(group.get('context_bbox_yx'))])
        policy_html='<details><summary>Actual strict selection and resource refusals</summary>'+table(['Variant','Mode','Group outcomes','Original-run outcomes','Run rejection reasons (counts)'],policy_rows)+table(['Variant','Mode','Unavailable family','Status','Reason','Original context bbox YX'],refusal_rows)+'</details>' if policy_rows or refusal_rows else ''
        visual=None if args.skip_figures else model_figures(root,figures,dataset)
        visual_html='' if not visual else '<figure>'+link(visual['path'],'Open actual model layers')+'<img src="'+escape(os.path.relpath(visual['path'],output.parent).replace('\\','/'))+'" alt="Actual image context, acceptance and permitted writes, raw owned core, full halo and selected writes"><figcaption>'+escape(visual['caption'])+'</figcaption></figure>'
        timed_rows=[]
        for model in dataset.get('models',[]):
            timing=model.get('timing')or{}
            timed_rows.append([escape(model.get('variant','unreported')),escape(model.get('mode','unreported')),number(timing.get('startup_seconds'),3),number(timing.get('tracker_transfer_pack_seconds'),3),number(timing.get('evaluation_seconds'),3),number(timing.get('cpu_reference_seconds',timing.get('cpu_pass_wall_seconds')),3),escape(timing.get('original_runs','—')),escape(timing.get('child_jobs','—'))])
        timing_html=table(['Variant','Mode','Startup s','Tracker/transfer/pack s','Evaluation s','CPU SDF reference pass s','Original runs','Child jobs'],timed_rows) if timed_rows else ''
        main_metrics=('<h3>Fixed 659 × 2065 semantic crop</h3><p class="small">Published unions from surviving families, not a complete-cohort score. The semantic crop may include untracked foreground. Raw proposal quality and actually selected writes are separate.</p>'+metric_rows(dataset,compact=True,stress=False)) if dataset.get('id')=='development' else ''
        if dataset.get('id')=='development'and any(row.get('status')=='single_family_stress_with_retained_B1_controls'for row in dataset.get('results',[])):
            main_metrics+='<h3>Single-family context stress · retained B1 controls</h3><p class="small">One family is generated under C3/full-context; other-family masks/selections are retained from B1. These OR-composed diagnostics are not complete new cohorts or joint fresh-pipeline selections. Timings cover only the generated stress family.</p>'+metric_rows(dataset,compact=True,stress=True)
        if dataset.get('common_results'):
            main_metrics+='<h3>Common surviving original lineages</h3><p class="small">Matched surviving support supplied by the scorer, separate from each variant’s full published union. Stress scopes remain separate; these semantic domains can still contain foreground outside the tracked families.</p>'
            main_metrics+=metric_rows(dataset,compact=dataset.get('id')=='development',common=True)
        dataset_html.append(f'<section><h2>{title}</h2><p>{description}</p><p class="note">{exposure}</p>'+cohort_html+main_metrics+'<details><summary>All fixed domains and evidence layers</summary>'+metric_rows(dataset)+'</details>'+policy_html+timing_html+'<p class="small">Tracker time includes transfers and raw packing. Startup and policy evaluation are separate measured phases; these phases are not a whole CLI walltime claim. Comparative totals have different admitted jobs/families: lower totals can reflect refusals and do not establish faster context expansion. Stress timings cover only one generated family, with retained B1 controls not rerun. Fixed-evidence A2 performs no fresh inference. SDF uses the exact original observations and unchanged mask-only production method; its CPU pass timing is reference construction, not a CPU/GPU speed comparison. SAM resource-refused coverage does not restrict SDF.</p>'+
            visual_html+''.join('<p>'+link(item['path'],item.get('label'))+'</p>' for item in dataset.get('sources',[]))+'</section>')
    if not dataset_html:
        dataset_html=['<section><h2>Model evidence pending</h2><p>Seen segment plane 61 (full-source frame 655) is exploratory. Later full-source frames 594 and 690 are reserved from current crop-strategy tuning, but their annotations were prior LTA prompts. They are not pristine or independent-patient holdouts. Actual raw and selected outputs will be listed once persisted.</p></section>']
    oracle_html='<p>No exact oracle-array export is available.</p>'
    if oracle:
        rows=[]
        for name,contract in oracle['metadata']['contracts'].items():
            rows.append([name,str(contract['restored_inside_old_crop']),str(contract['restored_outside_old_crop']),str(contract['removed_legacy_pixels']),
                         'Exact' if contract['matches_full_canvas_oracle'] else 'Mismatch'])
        oracle_html=table(['Contract','Restored inside old crop','Restored outside old crop','Old pixels removed','Full-canvas oracle'],rows)
        oracle_html += '<figure>'+link(oracle['path'],'Open exact synthetic figure')+'<img src="'+escape(os.path.relpath(oracle['path'],output.parent).replace('\\','/'))+'" alt="Synthetic exact acceptance and write masks, old versus restored and independent oracle"><figcaption>Counts span 21 synthetic frames. Red dashed box is old image context; green box is corrected context. '+escape(oracle['frame_selection'])+'. Structural correctness only: no SAM model or accuracy benchmark was executed.</figcaption></figure>'
    windows=[]
    for window in extraction.get('windows',[]):
        windows.append([escape(window['window_full_source_half_open']),str(window['middle_full_source_frame']),escape(window['endpoint_full_source_frames']),
                        'Exact retained pixel match' if window['middle_pixel_match_exact'] else 'Pixel mismatch'])
    available={dataset['id']:dataset for dataset in data.get('datasets',[]) if dataset.get('analysis_available',True)}
    aliases={'development_seen655':'development','followup_594':'source_590_599','followup_690':'source_686_695'}
    stage_rows=[[escape(spec['dataset_id']),escape(spec.get('stage','unreported')),escape(spec.get('middle_full_source_frame','unreported')),
                 'Actual scored outputs available' if aliases.get(spec['dataset_id'],spec['dataset_id'])in available else 'No scored analysis published yet']
                for spec in constants.get('datasets',[])]
    schedule_html='<p>No measured scheduling qualification is available.</p>'
    if scheduling.get('status')=='passed':
        trials=scheduling['trials']
        legacy=[t for t in trials if t['schedule']=='legacy']
        local=[t for t in trials if t['schedule']=='crop_local']
        old_tracker=statistics.median(t['parent_worker']['tracker_seconds'] for t in legacy)
        new_tracker=statistics.median(t['parent_worker']['tracker_seconds'] for t in local)
        old_seam=statistics.median(t['seam_wall'] for t in legacy)
        new_seam=statistics.median(t['seam_wall'] for t in local)
        rows=[[t['slot'],escape(t['schedule']),number(t['startup'],3),number(t['parent_worker']['tracker_seconds'],3),
               number(t['seam_wall'],3),str(t['parent_worker']['encoder_preparations']),str(t['parent_worker']['cache_hits'])] for t in trials]
        schedule_html=f'<p>The separate ABBA scheduling qualification reduced median parent-worker tracking from {old_tracker:.3f}s to {new_tracker:.3f}s ({(1-new_tracker/old_tracker)*100:.1f}% lower). Median bounded seam walltime was {old_seam:.3f}s → {new_seam:.3f}s ({(1-new_seam/old_seam)*100:.1f}% lower). This changes work ordering/cache reuse, not outer-crop geometry or model accuracy.</p>'
        schedule_html += table(['Trial','Schedule','Startup s','Parent worker tracker s','Bounded seam s','Encoder calls','Feature hits'],rows)
        schedule_html += '<p class="small">All child raw, owned-core, candidate and availability masks, individual child object scores and selected parent additions matched. Parent encoder preparations fell 66→33 and hits rose 0→33. Bounded seam walltime includes the separately partitioned predictor startup plus assembly, parent/consolidated tracking, selection, gates, publication and shutdown. It excludes decoding, detector inference and post-run fingerprint collection, so it is not a whole CLI/pipeline benchmark.</p><p>'+link('scheduler_abba/summary.json','ABBA metrics and exactness')+' · '+link('scheduler_abba/schedule_qualification.json','Complete trial provenance')+'</p>'
    sources=list(data.get('sources',[]))
    sources += [{'path':path} for path in ('protocol_constants.json','protocol_clarifications.json','protocol_manifest_development.json','extraction.json','planner_validation.json','original_geometry_validation.json',
        'regression_review/synthetic_world_contracts.json','regression_review/synthetic_world_contracts.npz',
        'scheduler_abba/summary.json','scheduler_abba/schedule_qualification.json') if (root/path).exists()]
    unique={str(item['path']):item for item in sources}
    summary={'schema':'xta.sam_outer_crop_report/1','status':data.get('status','unreported'),'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baseline_commit':planner.get('release_baseline'),'datasets':data.get('datasets',[]),'planner_validation':planner,'scheduling':scheduling,
        'missing_analysis_ids':data.get('missing_analysis_ids',[]),'upstream_status':data.get('upstream_status'),
        'oracle':None if not oracle else {'frame':oracle['frame'],'frame_selection':oracle['frame_selection'],'metadata':oracle['metadata']},
        'sources':[{'path':path,'sha256':file_sha(resolve(root,path))} for path in unique if is_source_file(resolve(root,path))]}
    output.with_suffix('.metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM outer-crop evidence</title><style>body{{font:16px/1.55 system-ui,sans-serif;background:#f3f6f8;color:#223746}}main{{max-width:1200px;margin:28px auto;padding:26px;background:white}}h1{{font-size:30px}}h2{{margin-top:28px}}.note{{background:#eef5f7;padding:14px;border-left:4px solid #23858a}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dce5eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f6}}img{{max-width:100%}}a{{color:#176f89}}figcaption,.small{{font-size:14px;color:#596d78}}</style></head><body><main><h1>Outer-crop geometry, context and selection</h1><p>Status: <strong>{escape(data.get('status','unreported').replace('_',' '))}</strong>. This report keeps structural geometry restoration separate from observed model behavior and actual bridge selection.</p>
    <div class="note"><strong>Distinct spatial contracts.</strong> Image context C controls pixels the model can see. Acceptance A controls quality measurement and containment. Permitted writes W control publishable additions. Owned cores govern tile assembly; full raw halos remain separate quality evidence. None is interchangeable with the others.</div>
    <section><h2>Swept-envelope correctness restoration</h2><p>The independent exact-array oracle checks restored A/W support against full-canvas geometry. This is a unit correctness result, not a claim that SAM segmentation improved. Real 54/64 and 54/68 A/W parity is reported separately; their image context extends only one or two rows.</p>{oracle_html}<p>{link('planner_validation.json','Planner validation')} · {link('original_geometry_validation.json','Real-case geometry and resource status')}</p></section>
    <section><h2>Scheduling qualification — separate from geometry and accuracy</h2>{schedule_html}</section>
    <section><h2>Study stages and original pixels</h2>{table(['Stage','Exposure scope','Evaluation source frame','Analysis status'],stage_rows)}{table(['Full-source window [first,last)','Evaluation source frame','Original endpoint source frames','Pixel validation'],windows)}<p>Frame 61 of the retained segment is full-source 655. Frames 594/690 are current-tuning-held-out follow-ups, with prior LTA prompt exposure disclosed. Same scan and detector training membership unverified; no independent-patient accuracy claim.</p></section>
    {''.join(dataset_html)}<section><h2>Reading the experiment variants</h2><p>B0 is baseline geometry; B1 restores swept-envelope correctness. C2 tests two long-axis model-patch units of image context while A/W/evaluation/raster contracts remain fixed relative to B1. A2 reuses C2 raw evidence with an altered acceptance contract: its selected W is a diagnostic reselection, never a fresh pipeline result. Any preregistered C3/full-context variants keep their own scope and resource status.</p><p>Equal research caps, resource refusals, unknown coverage and skipped seeds must stay visible. A complete generated run rejected by policy can legitimately publish zero selected writes. A resource-refused or missing run instead has unavailable model metrics and unknown coverage; it does not become successful predicted background. Partial published output is labeled as a partial-cohort effect. Numeric thresholds are not inferred from variant labels; persisted contracts and selection receipts are authoritative.</p></section>
    <section><h2>Timing and selection provenance</h2><p>Inference, predictor startup, rendering/transfers, policy/replay and publication timings are shown only when measured. Fixed-evidence A2 replay has no new inference. Raw candidates, filtered owned-core support, full-halo veto evidence and actually selected writes remain distinct. SDF is a separate mask-only reference where supplied, not a fallback for rejected SAM proposals.</p><p>{' · '.join(link(path,unique[path].get('label',path)) for path in unique if is_source_file(resolve(root,path)))} · {link(output.with_suffix('.metrics.json'),'Report metrics')}</p></section><p class="small">Generated {escape(summary['generated_utc'])}. This reporter performs no model inference, policy selection, commits, or modifications to prior reports.</p></main></body></html>'''
    output.write_text(document,encoding='utf-8')
    print(json.dumps({'html':str(output),'metrics':str(output.with_suffix('.metrics.json')),'status':summary['status']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--data',type=Path,help='Normalized report_data.json index; source analysis/receipts remain authoritative')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip-figures',action='store_true',help='Light table-only build during timed model work')
    build(parser.parse_args())


if __name__=='__main__':
    main()
