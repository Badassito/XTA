"""Reproduce original-seed SAM recovery for a resource-refused native family.

Research workflow: prepare detector-only geometry, generate independent native
whole/tile masks, package exact evidence, score existing labels, and replay
explicit branch policies. SDF is a reference; it never supplies output pixels.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import json
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _source(args):
    reference = json.loads((args.experiment/'sdf_references'/args.dataset/'reference.json').read_text('utf-8'))
    for role in ('original_observations', 'selected_additions'):
        if _sha(reference[role+'_file']) != reference[role+'_sha256']:
            raise ValueError('Retained source/reference bytes changed: '+role)
    return reference, int(reference['source_frame_start']), tuple(reference['source_shape_tyx']), int(reference['evaluation_cache_local'])


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_output_key(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or '..' in value:
        raise ValueError('Recovery identities must be safe file/directory names')
    return value


def _plan(args):
    from tools.sam_crop_strategy_geometry import _hash
    plan = json.loads((args.output/'strategy_plan.json').read_text('utf-8'))
    if _hash({key:value for key,value in plan.items() if key != 'plan_sha256'}) != plan['plan_sha256']:
        raise ValueError('Sealed recovery geometry plan changed')
    return plan


def select_refused_group(bundle, group_id=None):
    """Choose only retained original masks, never annotation or model pixels."""
    candidates = [group for group in bundle.groups.values() if not group.get('complete', True)]
    if group_id is not None:
        candidates = [group for group in candidates if group['group_id'] == group_id]
    if not candidates:
        raise ValueError('No retained refused original family matches the request')
    with bundle.reader() as reader:
        def area(group):
            if any('endpoint_local:'+e['observation_id'] not in group['mask_keys'] for e in group['endpoints']):
                raise ValueError('Refused family lacks its retained original seed silhouettes')
            return sum(int(reader.group_mask(group['group_id'], 'endpoint_local:'+endpoint['observation_id']).sum())
                       for endpoint in group['endpoints'])
        return max(candidates, key=lambda group: (area(group), group['group_id']))


def remap_contract(mask, old, new):
    """Preserve exact overlapping native pixels, with explicit cropped geometry."""
    arr = np.asarray(mask)
    if arr.dtype != np.bool_ or arr.ndim not in (2, 3) or arr.shape[-2:] != (old[2]-old[0], old[3]-old[1]):
        raise ValueError('Contract shape/dtype differs from its declared crop')
    if new[2] <= new[0] or new[3] <= new[1]:
        raise ValueError('Research context must be a nonempty rectangle')
    target = np.zeros((*arr.shape[:-2], new[2]-new[0], new[3]-new[1]), bool)
    y0, x0, y1, x1 = max(old[0],new[0]), max(old[1],new[1]), min(old[2],new[2]), min(old[3],new[3])
    if y0 < y1 and x0 < x1:
        target[..., y0-new[0]:y1-new[0], x0-new[1]:x1-new[1]] = arr[..., y0-old[0]:y1-old[0], x0-old[1]:x1-old[1]]
    return target


def load_proxy(path, reference, frame):
    if path is None:
        return np.load(reference['selected_additions_file'], mmap_mode='r')[frame] != 0
    with np.load(path, allow_pickle=False) as packet:
        return packet['prediction'][frame] != 0


def selected_bundles(root, families, modes):
    for mode in modes:
        for path in sorted((root/'bundles'/mode).glob('*/manifest.json')):
            family = path.parent.name
            if families is None or family in families:
                yield family, mode


def merge(args):
    """Replace a refused family, preserving all other raw owners for joint QA."""
    from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, _plain
    from XTA.sam_interpolation import _StreamingGroupMasks
    reference, offset, shape, frame = _source(args)
    if args.family is None or len(args.family) != 1:
        raise ValueError('Joint recovery merge requires one --family')
    records = []
    for mode in args.modes:
        old = SamEvidenceBundle.open(args.experiment/'runs'/args.dataset/mode/'B1'/'evidence')
        new = SamEvidenceBundle.open(args.output/'bundles'/mode/args.family[0])
        if tuple(old.scope['shape_tyx']) != tuple(new.scope['shape_tyx']) or tuple(new.scope['shape_tyx']) != shape:
            raise ValueError('Joint evidence shape differs from the original source')
        if old.scope['source_image_sha256'] != new.scope['source_image_sha256']:
            raise ValueError('Joint evidence was generated from different images')
        for key in ('sam_model', 'sam_runtime'):
            if key in old.scope and key in new.scope and old.scope[key] != new.scope[key]:
                raise ValueError('Joint evidence model/runtime identity differs: '+key)
        if len(new.groups) != 1:
            raise ValueError('Joint recovery currently replaces one original family')
        restored = next(iter(new.groups.values()))
        ids = {endpoint['observation_id'] for endpoint in restored['endpoints']}
        replaced = [gid for gid, group in old.groups.items()
                    if not group.get('complete', True) and ids == {e['observation_id'] for e in group['endpoints']}]
        if len(replaced) != 1:
            raise ValueError('Recovery does not uniquely replace an original refused family')
        output = args.output/'bundles'/mode/('joint_'+args.family[0])
        scope = _plain(old.scope)
        scope.update(scope_id=args.dataset+'/joint_refusal_recovery/'+args.family[0]+'/'+mode,
                     research_only=True, joint_recovery={'original_evidence_fingerprint':old.evidence_fingerprint,
                     'recovery_evidence_fingerprint':new.evidence_fingerprint,'replaced_refused_group':replaced[0],
                     'original_observations_sha256':reference['original_observations_sha256'],
                     'all_raw_owners_reselected_jointly':True,'saved_selections_imported':False})
        with SamEvidenceWriter(output, scope) as writer:
            for bundle, skip in ((old, set(replaced)), (new, set())):
                with bundle.reader() as reader:
                    for gid, group in bundle.groups.items():
                        if gid in skip:
                            continue
                        masks = _StreamingGroupMasks()
                        for name in group['mask_keys']:
                            masks[name] = lambda gid=gid, name=name: reader.group_mask(gid, name)
                        writer.add_group(_plain(group), masks)
                    for rid, run in bundle.runs.items():
                        if run['group_id'] in skip:
                            continue
                        descriptor = _plain(run)
                        for tile in run.get('tile_evidence', ()):
                            frames = {int(f):reader.tile_raw_mask(rid,tile['tile_id'],int(f))
                                      for f in tile['raw_mask_keys']}
                            writer.add_run_tile(rid, _plain(tile), frames)
                        raw = {int(f):reader.raw_mask(rid,int(f)) for f in run['raw_mask_keys']}
                        candidates = {int(f):reader.candidate_mask(rid,int(f)) for f in run['raw_mask_keys']}
                        available = ({int(f):reader.availability_mask(rid,int(f)) for f in run['raw_mask_keys']}
                                     if run.get('availability_mask_keys') else None)
                        writer.add_run(descriptor, raw, candidates, availability_masks=available)
            joint = writer.commit()
        record = dict(mode=mode, family='joint_'+args.family[0], evidence=str(output),
                      evidence_fingerprint=joint.evidence_fingerprint, original_groups=len(old.groups),
                      joint_groups=len(joint.groups), original_runs=len(old.runs), joint_runs=len(joint.runs),
                      original_refused_group_replaced=replaced[0], all_original_raw_owners_preserved=True)
        records.append(record)
        print('Joint evidence', mode, record['family'], len(joint.runs), 'original runs', flush=True)
    (args.output/'joint_bundle_index.json').write_text(json.dumps(records,indent=2), encoding='utf-8')


def generate(args):
    from tools.compare_sam_crop_strategies import run_strategies
    if args.model is None:
        raise ValueError('Generation requires --model with a local SAM bundle')
    plan_path = args.output/'strategy_plan.json'
    plan = _plan(args)
    margin_families = [family['family_id'] for family in plan['families'] if family['padding'] is not None]
    baseline_families = [family['family_id'] for family in plan['families'] if family['padding'] is None]
    common = dict(plan=plan_path, model=args.model, device=args.device, repeats=1, cache_mib=512)
    run_strategies(SimpleNamespace(**common, output=args.output/'margin_runs', family=margin_families,
                                  strategy=['whole_crop', 'independent_tiles']))
    run_strategies(SimpleNamespace(**common, output=args.output/'baseline_runs', family=baseline_families,
                                  strategy=['whole_crop']))


def prepare(args):
    root = out = args.output
    original = args.experiment
    repo = REPO
    ref, offset, shape, evaluation_frame = _source(args)
    from pathlib import Path
    import hashlib,json
    import numpy as np
    from XTA.sam_evidence import SamEvidenceBundle
    from tools.sam_crop_strategy_geometry import Observation,tile_plan,mask_in_crop,_hash

    if (out/'strategy_plan.json').exists():
        raise FileExistsError('Refusal recovery preparation requires a fresh plan destination')
    ref=json.loads((original/'sdf_references'/args.dataset/'reference.json').read_text())
    window=json.loads((original/args.dataset/'window.json').read_text())
    if _sha(window['image_path']) != window['image_sha256']:
        raise ValueError('Retained source images changed before geometry preparation')
    b=SamEvidenceBundle.open(original/'runs'/args.dataset/'whole'/'B1'/'evidence')
    g=select_refused_group(b,args.group)
    if set(int(e['frame_index']) for e in g['endpoints']) != {0, shape[0]-1}:
        raise ValueError('This recovery workflow requires original anchors at both ends of the retained window')
    seeds=out/'seeds';seeds.mkdir(exist_ok=True)
    observed=[]
    with b.reader() as r:
     for e in g['endpoints']:
      safe_output_key(e['observation_id'])
      mask=r.group_mask(g['group_id'],'endpoint_local:'+e['observation_id'])
      o=Observation(e['observation_id'],offset+int(e['frame_index']),int(e['canonical_label']),tuple(e['bbox_yx']),mask)
      observed.append(o)
      np.savez_compressed(seeds/(o.observation_id+'.npz'),mask=mask,bbox_yx=np.array(o.bbox_yx),frame_native=np.array(o.frame_native))
    union=[min(o.bbox_yx[0] for o in observed),min(o.bbox_yx[1] for o in observed),max(o.bbox_yx[2] for o in observed),max(o.bbox_yx[3] for o in observed)]
    families=[]
    for variant,padding in [(args.family_prefix+'_pair'+str(p),p) for p in args.margins]+[(args.family_prefix+'_B1',None)]:
     crop=list(g['context_bbox_yx']) if padding is None else [max(0,union[0]-padding),max(0,union[1]-padding),min(shape[1],union[2]+padding),min(shape[2],union[3]+padding)]
     grid=tile_plan(crop)
     runs=[]
     for o in observed:
      direction='forward' if o.frame_native==offset else 'backward'
      opposite=sorted({edge['target_id'] if direction=='forward' else edge['source_id'] for edge in g['edges'] if o.observation_id in (edge['source_id'],edge['target_id'])})
      tile_seeds=[]
      for t in grid['tiles']:
       area=int(mask_in_crop(o,t['crop_bbox_yx']).sum())
       tile_seeds.append({'tile_id':t['tile_id'],'seed_foreground':area,'status':'independently_original_seeded' if area else 'unavailable_empty_original_seed','ownership_bbox_yx':t['ownership_bbox_yx']})
      runs.append({'run_id':variant+'_'+o.observation_id+'_'+direction,'seed_observation_id':o.observation_id,'seed_frame_native':o.frame_native,
          'direction':direction,'held_out_observation_ids':opposite,'frame_start_native':offset,'frame_stop_native':offset+shape[0],'tile_seeds':tile_seeds})
     original_group=g.get('crop_contract',{}).get('outer_crop_experiment',{}).get('original_group_id',g['group_id'])
     families.append({'family_id':variant,'original_group_id':original_group,'refused_group_id':g['group_id'],
       'observation_ids':[o.observation_id for o in observed],'edges':[dict(e) for e in g['edges']],
       'whole_crop_bbox_yx':crop,'endpoint_union_bbox_yx':union,'tile_strategy':grid,'runs':runs,
       'crop_rule':'Retained corrected B1 swept image context' if padding is None else 'Original endpoint union rectangle plus '+str(padding)+' native pixels on all axes',
       'padding':padding,'original_mask_pixels_unchanged':True,'predictions_used_to_plan':False})
    plan={'schema':'xta.sam_native_crop_strategy_plan/1','source_images':window['image_path'],'source_shape_tyx':window['shape_tyx'],'source_frame_start':offset,
     'endpoint_frames_native':[offset,offset+shape[0]-1],'source_shape_yx':[shape[1],shape[2]],'seed_directory':str(seeds),'source_observations':[o.descriptor((shape[1],shape[2])) for o in observed],
     'families':families,'geometry_source_sha256':hashlib.sha256((repo/'tools'/'sam_crop_strategy_geometry.py').read_bytes()).hexdigest(),
     'source_image_sha256':window['image_sha256'],'source_original_observations_sha256':ref['original_observations_sha256'],
     'source_evidence_fingerprint':b.evidence_fingerprint,'original_refusal_reasons':list(g['reasons']),
     'original_refused_contract_bytes':g['crop_contract']['charged_contract_bytes'],'protocol':{'research_only':True,'labels_used_for_geometry':False,
     'family_selection':'Retained resource-refused family, selected by original seed area or explicit group identity',
     'conditioning':'Each original detector observation is injected independently; each tile intersects the original seed only; no prediction feedback',
     'whole_control':'Whole original family context versus two original-endpoint union margins; no policy selection during generation',
     'benchmark':False,'annotation_exposure':'Existing seen study labels; not a new blind holdout'},
     'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'tools'/'compare_sam_crop_strategies.py',repo/'XTA'/'sam_tracker_runtime.py',repo/'XTA'/'lta_experimental.py']}}
    plan['plan_sha256']=_hash(plan)
    (out/'strategy_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps([{'family':f['family_id'],'crop':f['whole_crop_bbox_yx'],'pixels':(f['whole_crop_bbox_yx'][2]-f['whole_crop_bbox_yx'][0])*(f['whole_crop_bbox_yx'][3]-f['whole_crop_bbox_yx'][1]),'tiles':len(f['tile_strategy']['tiles']),'attempted_tile_runs':sum(s['seed_foreground']>0 for r in f['runs'] for s in r['tile_seeds'])} for f in families],indent=2))


def package(args):
    root = out = args.output
    original = args.experiment
    repo = REPO
    ref, offset, shape, evaluation_frame = _source(args)
    from pathlib import Path
    from dataclasses import replace
    from types import SimpleNamespace,MappingProxyType
    import hashlib,json,gc
    import numpy as np
    from XTA.sam_bridge_planning import plan_sam_bridges,SamPlanningLimits
    from XTA.sam_evidence import SamEvidenceWriter
    from XTA.sam_interpolation import _write_group,_store_generated_parent_run
    from XTA.sam_crop_tiling import prepare_tiled_jobs,tile_descriptor,TiledRunAssembly
    from XTA.sam_tracker_runtime import SamTrackerRunResult
    from tools.analyze_sam_crop_strategies import RawRunReader
    strategy=_plan(args)
    observations=np.load(original/'plans'/args.dataset/'observations.npy',mmap_mode='r')
    settings=dict(interpolation_distance=15,interpolation_candidates=1,interpolation_walk_back=0,interpolation_passes=1,interpolation_min_radius=3.,interpolation_search_angle=30.,scope_id='outer_crop_'+args.dataset,observation_lineage=dict(kind='actual3072_gray_detector_endpoint_union',source_frame_start=offset,labels_used=False))
    plan=plan_sam_bridges(observations,limits=SamPlanningLimits(max_group_bytes=args.group_mib*1024**2,max_total_contract_bytes=args.total_mib*1024**2),**settings)
    base=next(g for g in plan.groups if set(g.observation_ids)==set(strategy['families'][0]['observation_ids']))
    assert base.status=='planned',base.reasons
    observed={o.observation_id:o for o in plan.observations}
    original_runs=[r for r in plan.runs if r.group_id==base.group_id]
    assert len(original_runs)==len(strategy['families'][0]['runs'])
    results=[]

    def result_from_row(reader,row):
     if (not row['runtime_receipt'].get('prediction_valid') or not row['runtime_receipt'].get('coverage_complete')
      or not row['runtime_receipt']['adapter_receipt'].get('raw_observation_complete')):raise ValueError('Invalid or incomplete raw session')
     return SamTrackerRunResult(frames={f-offset:reader.tile_plane(row,f) for f in row['native_frames']},tracker_scores={int(f)-offset:v for f,v in row['tracker_probabilities'].items()},observation_status={f-offset:'observed' for f in row['native_frames']},receipt=row['runtime_receipt'])

    locations=[('whole',root/'margin_runs'/'repeat1'/'whole_crop'),('tiled',root/'margin_runs'/'repeat1'/'independent_tiles'),('whole',root/'baseline_runs'/'repeat1'/'whole_crop')]
    for mode,directory in locations:
     reader=RawRunReader(directory,strategy)
     for family in strategy['families']:
      if family['family_id'] not in {r['family_id'] for r in reader.rows}:continue
      family_id=family['family_id'];new=tuple(family['whole_crop_bbox_yx']);old=tuple(base.context_bbox_yx)
      group_id='recovery_'+family_id+'_'+base.group_id
      recipe={'schema':'xta.sam_original_seed_context_recovery/1','variant':family_id,'actual_context_bbox_yx':new,'original_swept_context_bbox_yx':old,
       'original_group_id':base.group_id,'source_refused_group':strategy['families'][0]['refused_group_id'],'original_endpoint_union_bbox_yx':family['endpoint_union_bbox_yx'],
       'padding_px':family['padding'],'geometry_source':'Complete original observations only; no model or annotation pixels choose the rectangle',
       'legacy_domains':'Existing world-coordinate A/W/edge contracts intersected with actual research image context; known/unrelated recomputed from all original observations',
       'stock_B1_geometry_equivalent':family['padding'] is None,'live_resource_budget_qualified':False,'research_group_cap_bytes':args.group_mib*1024**2,'research_total_cap_bytes':args.total_mib*1024**2,
       'original_geometry_world_pixels_preserved':new==old,'raw_generation_plan_sha256':strategy['plan_sha256']}
      with base.materialize_contracts() as concrete:
       if family['padding'] is None:
        assert new==old,(new,old)
        group=replace(concrete,group_id=group_id,crop_contract={**dict(concrete.crop_contract),'research_context_recovery':recipe})
       else:
        updates={key:remap_contract(getattr(concrete,key),old,new) for key in ['acceptance_masks','write_masks']}
        for key in ['branch_evaluation_masks','edge_write_masks','branch_permitted_masks','edge_contract_masks']:
         updates[key]={k:remap_contract(v,old,new) for k,v in getattr(concrete,key).items()}
        y0,x0,y1,x1=new
        known=np.asarray(observations[list(concrete.frame_indices),y0:y1,x0:x1]!=0)
        own=np.zeros(known.shape,bool)
        for oid in concrete.observation_ids:
         o=observed[oid];own[list(concrete.frame_indices).index(o.frame_index)]|=o.mask_in_crop(new)
        updates['known_foreground_masks']=known;updates['unrelated_masks']=known&~own
        group=replace(concrete,group_id=group_id,context_bbox_yx=new,contract_recipe=None,
         crop_contract={'schema':'xta.sam_research_seed_pair_context/1','context_bbox_yx':new,'canvas_shape_yx':observations.shape[1:],
          'original_observed_family_bbox_yx':family['endpoint_union_bbox_yx'],'baseline_swept_contract':dict(concrete.crop_contract),'research_context_recovery':recipe},**updates)
       runs=[replace(r,group_id=group_id,run_id=family_id+'_'+r.run_id) for r in original_runs]
       out=root/'bundles'/mode/family_id
       out.parent.mkdir(parents=True,exist_ok=True)
       meta={'schema':'xta.sam_refusal_recovery_scope/1','scope_id':args.dataset+'/refusal_recovery/'+family_id+'/'+mode,'research_only':True,
        'backend':'sam','sam_crop_mode':mode,'shape_tyx':list(observations.shape),'source_frame_start':offset,'physical_view':'transverse','pass_index':1,
        'source_image_sha256':strategy['source_image_sha256'],'input_fingerprints':{'original_observations_sha256':strategy['source_original_observations_sha256'],'source_images_sha256':strategy['source_image_sha256'],'raw_generation_plan_sha256':strategy['plan_sha256']},
        'crop_contract_identity':recipe,'unrelated_observations':'Complete original source observations in the actual crop; no family-only masked substitute',
        'raw_generation_performed_by':'tools/compare_sam_crop_strategies.py','model_generation_finished_before_policy_selection':True}
       with SamEvidenceWriter(out,meta) as writer:
        _write_group(writer,group,observed,3.)
        if mode=='whole':
         for run in runs:
          row=next(r for r in reader.rows if r['family_id']==family_id and r['seed_observation_id']==run.seed_ids[0])
          _store_generated_parent_run(writer,run,result_from_row(reader,row),group,observed,writer.scope,{'research_geometry':recipe})
        else:
         jobs,inventory,tiling_hash,assembly_bytes=prepare_tiled_jobs(runs,{group_id:group},observed)
         writer.scope.update(tiling_plan_sha256=tiling_hash,assembly_logical_bytes=assembly_bytes)
         for i,run in enumerate(runs):
          for tile in inventory[i]:
           if not tile.attempted:writer.add_run_tile(run.run_id,tile_descriptor(run,tile),{})
          assembly=TiledRunAssembly(run,group,inventory[i],root/'assembly_temp'/run.run_id)
          for job in [j for j in jobs if j.original_run_index==i]:
           row=next(r for r in reader.rows if r['family_id']==family_id and r['seed_observation_id']==run.seed_ids[0] and r['tile_id']==job.tile.tile_id)
           if tuple(row['crop_bbox_yx'])!=job.tile.crop_bbox_yx or tuple(row['ownership_bbox_yx'])!=job.tile.ownership_bbox_yx:raise ValueError('Actual fixed tile recipe differs')
           descriptor,masks=assembly.consume(job,result_from_row(reader,row));writer.add_run_tile(run.run_id,descriptor,masks)
          assert assembly.ready
          _store_generated_parent_run(writer,run,assembly.result(),group,observed,writer.scope,{'research_geometry':recipe},assembly.availability())
          assembly.close()
        bundle=writer.commit()
        results.append({'family':family_id,'mode':mode,'evidence':str(out),'evidence_fingerprint':bundle.evidence_fingerprint,'original_runs':len(runs),'geometry':recipe})
        print('Packed',mode,family_id,bundle.evidence_fingerprint,flush=True)
       del group,runs
      gc.collect()
     reader.close()
    (root/'bundle_index.json').write_text(json.dumps({'schema':'xta.sam_recovery_bundle_index/1','records':results},indent=2),encoding='utf-8')


def score(args):
    root = out = args.output
    original = args.experiment
    repo = REPO
    ref, offset, shape, evaluation_frame = _source(args)
    from pathlib import Path
    import json,hashlib
    import numpy as np
    from scipy import ndimage as ndi
    from tools.analyze_sam_crop_strategies import RawRunReader,load_truth
    from tools.analyze_sam_crop_strategies import load_observations
    from XTA.sam_filtering import filter_sam_components

    plan=_plan(args)
    seeds=load_observations(plan)
    binding = None
    sdf = load_proxy(args.proxy, ref, evaluation_frame)
    analysis=json.loads((original/('analysis_'+args.dataset+'.json')).read_text())
    truth=load_truth(analysis['label'],(shape[1],shape[2]))
    obs=np.load(original/'plans'/args.dataset/'observations.npy',mmap_mode='r')[evaluation_frame]!=0
    with np.load(original/'previews'/args.dataset/'whole'/'B1.npz') as p: baseline=p['raw']&~obs
    truth = truth & ~obs
    ref=np.load(original/'sdf_references'/args.dataset/'selected_additions.npy',mmap_mode='r')[evaluation_frame]!=0

    def metric(a,b):
     tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum())
     return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(1,tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)}
    def largest(a):
     labs,n=ndi.label(a,np.ones((3,3),bool));counts=np.bincount(labs.ravel());counts[0]=0
     return labs==int(counts.argmax()) if n else np.zeros_like(a)
    records=[]
    locations=[('whole',root/'margin_runs'/'repeat1'/'whole_crop'),('tiled',root/'margin_runs'/'repeat1'/'independent_tiles'),('whole',root/'baseline_runs'/'repeat1'/'whole_crop')]
    for mode,directory in locations:
     if not (directory/'raw_index.json').exists(): continue
     reader=RawRunReader(directory,plan)
     families={f['family_id']:f for f in plan['families']}
     for family_id in sorted({r['family_id'] for r in reader.rows}):
      family=families[family_id];y0,x0,y1,x1=family['whole_crop_bbox_yx']
      masks={k:np.zeros((shape[1],shape[2]),bool) for k in ['raw','radius3','largest','forward','backward']}
      endpoint_agreements=[]
      for run in family['runs']:
       raw,available=reader.assemble_frame(run['run_id'],offset+evaluation_frame)
       filtered,_=filter_sam_components(raw,3.)
       masks['raw'][y0:y1,x0:x1]|=raw
       masks['radius3'][y0:y1,x0:x1]|=filtered
       masks['largest'][y0:y1,x0:x1]|=largest(raw)
       masks[run['direction']][y0:y1,x0:x1]|=raw
       held=offset if run['direction']=='backward' else offset+shape[0]-1
       terminal,coverage=reader.assemble_frame(run['run_id'],held)
       for target in run['held_out_observation_ids']:
        seed=seeds[target]
        ty0,tx0,ty1,tx1=seed.bbox_yx
        predicted=terminal[ty0-y0:ty1-y0,tx0-x0:tx1-x0]
        hit=int((predicted&seed.mask_crop).sum())
        endpoint_agreements.append({'seed':run['seed_observation_id'],'target':target,'direction':run['direction'],'reference_area':int(seed.mask_crop.sum()),'intersection':hit,'recall':hit/max(1,int(seed.mask_crop.sum()))})
      metrics={}
      for name,predicted in masks.items():
       metrics[name]={'lower_sdf_proxy_agreement_not_truth':metric(predicted,sdf),'all_semantic_truth':metric(predicted,truth),'combined_with_old_B1_raw':metric(predicted|baseline,truth)}
      packet=root/(family_id+'_'+mode+'_native_midpoint.npz');np.savez_compressed(packet,**masks)
      record={'family':family_id,'mode':mode,'crop':family['whole_crop_bbox_yx'],'metrics':metrics,'endpoint_agreement':endpoint_agreements,
          'native_midpoint_masks':str(packet),'native_midpoint_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),
          'new_raw_foreground':int(masks['raw'].sum()),'incremental_truth_over_old_B1':int((masks['raw']&~baseline&truth).sum()),'incremental_nontruth':int((masks['raw']&~baseline&~truth).sum())}
      records.append(record)
      print(family_id,mode,'lowerSAMvsSDF',round(metrics['raw']['lower_sdf_proxy_agreement_not_truth']['iou'],4),'combinedtruthIoU',round(metrics['raw']['combined_with_old_B1_raw']['iou'],4),'newTP',record['incremental_truth_over_old_B1'],'newFP',record['incremental_nontruth'],flush=True)
     reader.close()
    report={'schema':'xta.sam_refused_family_recovery/1','production_enabled':False,'plan_sha256':plan['plan_sha256'],'label_sha256':analysis['label_sha256'],
     'same_input_original_observations_sha256':plan['source_original_observations_sha256'],'original_refused_contract_bytes':plan['original_refused_contract_bytes'],
     'proxy_binding_file':None if args.proxy is None else str(args.proxy),'sdf_full_original_truth_metrics':metric(ref,truth),'old_B1_raw_full_truth_metrics':metric(baseline,truth),
     'records':records,'limitations':['One previously exposed annotation frame of one subject; functional crop and refusal recovery evidence, not blind qualification.',
     'SDF is a geometric reference, not ground truth. Semantic annotations can contain other objects beyond the recovered family.',
     'Combined outputs include retained unselected SAM from old B1 survivors and newly generated lower family, before production branch safety rules.',
     'Independent tile holes with no original seed are unavailable spatial coverage; aggregate tile probability is undefined.'],
     'no_SDF_pixels_in_output':True}
    (root/'recovery_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')



def select(args):
    root = out = args.output
    original = args.experiment
    repo = REPO
    ref, offset, shape, evaluation_frame = _source(args)
    from pathlib import Path
    import json,hashlib
    import numpy as np
    from XTA.sam_evidence import SamEvidenceBundle,selected_native_plane
    from XTA.sam_policy import select_sam_proposals
    from tools.analyze_sam_crop_strategies import load_truth
    a=json.loads((original/('analysis_'+args.dataset+'.json')).read_text())
    truth=load_truth(a['label'],(shape[1],shape[2]))
    obs=np.load(original/'plans'/args.dataset/'observations.npy',mmap_mode='r')
    truth=truth&~(obs[evaluation_frame]!=0)
    with np.load(original/'previews'/args.dataset/'whole'/'B1.npz') as p: oldraw=p['raw']&~(obs[evaluation_frame]!=0)
    sdf=np.load(original/'sdf_references'/args.dataset/'selected_additions.npy',mmap_mode='r')[evaluation_frame]!=0

    def metric(p,t):
     tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
     return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(1,tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)}
    records=[]
    for recall in args.recall:
     for family,mode in selected_bundles(root,args.family,args.modes):
      version=6 if mode=='whole' else 7
      b=SamEvidenceBundle.open(root/'bundles'/mode/family)
      policy={'sam_bridge_policy':{'kind':'conservative','version':version,'strict_containment':False,'guarded_rescue':False,'branch_aware_selection':True,'allow_paired_seed_tracks':True,'branch_write_domain':'fixed_context','branch_crop_boundary_policy':args.crop_boundary_policy,'min_endpoint_recall':recall,'max_group_bytes':args.group_mib*1024**2}}
      receipt=select_sam_proposals(b,policy=policy,frozen_evidence=True,reader_cache_bytes=32*1024**2,workers=1)
      folder=root/'selected'/args.selection_tag/family/('recall'+str(recall))/mode;folder.mkdir(parents=True,exist_ok=True)
      (folder/'selection.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
      planes={}
      for frame in range(shape[0]):
       plane=selected_native_plane(b,receipt,frame,shape_yx=(shape[1],shape[2]))
       if np.any(plane&(obs[frame]!=0)):raise ValueError('Selection repainted original observations')
       planes[str(frame+offset)]=plane
      payload=folder/'selected_native_planes.npz';np.savez_compressed(payload,**planes)
      p=planes[str(offset+evaluation_frame)]
      raw=np.zeros(p.shape,bool)
      with b.reader() as r:
       for rid,run in b.runs.items():
        g=b.groups[run['group_id']];y0,x0,y1,x1=g['context_bbox_yx'];raw[y0:y1,x0:x1]|=r.raw_mask(rid,evaluation_frame)
      if np.any(p&~raw):raise ValueError('Selection inserted non-SAM support')
      record={'family':family,'mode':mode,'version':version,'min_endpoint_recall':recall,'evidence_fingerprint':b.evidence_fingerprint,'policy':policy,
        'selected_run_ids':receipt['selected_run_ids'],'selected_branch_ids':sorted(receipt.get('branch_selection',{}).get('edges',{})),
        'censored_branch_count':sum(bool(edge.get('internal_crop_edge_pixels')) for edge in receipt.get('branch_selection',{}).get('edges',{}).values()),
        'selected_foreground_midpoint':int(p.sum()),'selected_foreground_all_frames':sum(int(v.sum()) for v in planes.values()),
        'lower_only_semantic_truth_metrics':metric(p,truth),'combined_with_original_B1_raw_ceiling':metric(p|oldraw,truth),
        'selection_file':str(folder/'selection.json'),'selected_payload':str(payload),'selected_payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),
        'all_selected_pixels_from_raw_SAM':True,'original_observations_preserved':True,
        'group_reasons':{k:v.get('reasons',[]) for k,v in receipt['group_receipts'].items()},
        'run_reasons':{k:v.get('reasons',[]) for k,v in receipt['run_receipts'].items()}}
      records.append(record)
      (root/('selection_evaluation_'+args.selection_tag+'.json')).write_text(json.dumps({'schema':'xta.sam_recovery_selected_quality/1','sdf_reference_truth_metrics':metric(sdf,truth),'records':records},indent=2),encoding='utf-8')
      print(mode,'recall',recall,'selected runs',len(receipt['selected_run_ids']),'midpoint',int(p.sum()),'combined raw control IoU',record['combined_with_original_B1_raw_ceiling']['iou'],flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'generate', 'package', 'merge', 'score', 'select'))
    parser.add_argument('--experiment', type=Path, required=True, help='Sealed SAM_Outer_Crop experiment root')
    parser.add_argument('--output', type=Path, required=True, help='Task-specific recovery directory')
    parser.add_argument('--dataset', default='source_590_599')
    parser.add_argument('--group', help='Retained refused group ID; otherwise largest original seed family')
    parser.add_argument('--family-prefix', default='recovery')
    parser.add_argument('--margins', type=int, nargs='+', default=(64, 128))
    parser.add_argument('--family', nargs='+', help='Packaged family IDs for policy replay')
    parser.add_argument('--modes', choices=('whole','tiled'), nargs='+', default=('whole','tiled'))
    parser.add_argument('--recall', type=float, nargs='+', default=(0., .5))
    parser.add_argument('--group-mib', type=int, default=1024)
    parser.add_argument('--total-mib', type=int, default=2048)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--proxy', type=Path, help='Optional exact-instance SDF NPZ with prediction key')
    parser.add_argument('--crop-boundary-policy', choices=('reject','retain_censored'), default='reject')
    parser.add_argument('--selection-tag', default='research', help='Distinct saved selection arm name')
    args = parser.parse_args(argv)
    for identity in (args.dataset, args.family_prefix, args.selection_tag, *(args.family or ())):
        safe_output_key(identity)
    if any(m < 0 or m > 1024 for m in args.margins) or len(set(args.margins)) != len(args.margins):
        parser.error('--margins must be unique native pixel margins in [0, 1024]')
    if args.group_mib <= 0 or args.total_mib < args.group_mib:
        parser.error('Research memory caps must be positive and total must cover one group')
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
