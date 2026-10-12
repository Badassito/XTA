"""Read retained metadata only; count upper bounds, not predicted speedups."""
import argparse
import collections
import json
import pathlib
import tarfile


def inspect_index(data, label, *, identity_sets=None):
    groups, runs = data.get('groups', {}), data.get('runs', {})
    exact = collections.defaultdict(list)
    families = collections.defaultdict(list)
    unique_features = set()
    frame_work = 0
    eligible = []
    feature_counts=collections.Counter()
    timing_counts=collections.Counter()
    cache_max=collections.Counter()
    cache_delta=collections.Counter()
    identity_features=set()
    worker_identity_features=set()
    family_workers=collections.defaultdict(set)
    for rid, run in runs.items():
        group = groups.get(run['group_id'], {})
        crop = tuple(group.get('context_bbox_yx', ()))
        frames = tuple(run.get('expected_frames', ()))
        if not crop or not frames:
            continue
        direction = run.get('direction')
        record = {'run_id': rid, 'group_id': run['group_id'], 'crop': crop,
                  'frames': frames, 'seed_ids': run.get('seed_ids'), 'direction': direction}
        receipt=run.get('runtime_receipt') or {}
        image=receipt.get('image_cache') or {}
        preprocessing=receipt.get('input_preparation') or {}
        quota=(receipt.get('worker_startup_receipt') or {}).get('cuda_allocator_quota') or {}
        worker=(quota.get('execution_device_id'),quota.get('worker_index'))
        family_workers[run['group_id']].add(worker)
        loader=(preprocessing.get('policy'),preprocessing.get('implementation_sha256'),
                preprocessing.get('image_size'),tuple(preprocessing.get('image_mean',())),
                tuple(preprocessing.get('image_std',())))
        source=(image.get('identity_sha256'),image.get('physical_view_id'),crop,loader)
        for frame in frames:
            key=(source,frame)
            identity_features.add(key)
            worker_identity_features.add((worker,key))
        adapter=receipt.get('adapter_receipt') or {}
        for key in ('shared_feature_cache_hits','shared_feature_cache_misses','feature_only_cache_hits','feature_only_preparations'):
            feature_counts[key]+=adapter.get('tracker_feature_preparation',{}).get(key,0)
        for key in ('tracker_seconds','input_preparation_seconds','render_seconds','pack_seconds','sdk_session_init_seconds'):
            timing_counts[key]+=receipt.get('timings',{}).get(key,0.)
        for key,value in (receipt.get('feature_cache_after') or {}).items():
            if isinstance(value,(float,int)):
                cache_max[key]=max(cache_max[key],value)
                if key in ('admissions','evictions','rejected_active_bytes','rejected_headroom','rejected_oversize'):
                    cache_delta[key]+=value-(receipt.get('feature_cache_before') or {}).get(key,0)
        exact[(crop, direction, frames)].append(record)
        families[(crop, direction, frames[0])].append(record)
        frame_work += len(frames)
        unique_features.update((crop, f) for f in frames)
        eligible.append(record)
    same_inventory = [bucket for bucket in exact.values() if len(bucket) > 1]
    same_start = [bucket for bucket in families.values() if len(bucket) > 1]
    # Equal crop/start/direction/end only. No change to scale or frame inventory.
    strict_reduction = sum((len(bucket)-1)*len(bucket[0]['frames']) for bucket in same_inventory)
    start_reduction=sum(sum(len(record['frames']) for record in bucket)-
                        max(len(record['frames']) for record in bucket) for bucket in same_start)
    # Greedy upper bounds: metadata proximity only, not mask interaction tests.
    nearby={}
    by_time=collections.defaultdict(list)
    for record in eligible:
        by_time[(record['direction'],record['frames'])].append(record)
    for stretch in (1.10,1.25,1.50,2.00):
        saved=0
        grouped_runs=0
        cohorts=[]
        for bucket in by_time.values():
            pending=sorted(bucket,key=lambda record:(record['crop'][0],record['crop'][1],record['run_id']))
            packed=[]
            for record in pending:
                crop=record['crop']
                for cohort in packed:
                    if len(cohort['members'])>=16:
                        continue
                    old=cohort['crop']
                    union=(min(old[0],crop[0]),min(old[1],crop[1]),max(old[2],crop[2]),max(old[3],crop[3]))
                    h,w=union[2]-union[0],union[3]-union[1]
                    if h>1008 or w>1008:
                        continue
                    members=(*cohort['members'],record)
                    if all(h<=(r['crop'][2]-r['crop'][0])*stretch and
                           w<=(r['crop'][3]-r['crop'][1])*stretch for r in members):
                        cohort['crop']=union
                        cohort['members'].append(record)
                        break
                else:
                    packed.append({'crop':crop,'members':[record]})
            for cohort in packed:
                if len(cohort['members'])>1:
                    count=len(cohort['members'])
                    saved+=(count-1)*len(cohort['members'][0]['frames'])
                    grouped_runs+=count
                    cohorts.append(cohort)
        nearby[str(stretch)]={'grouped_runs':grouped_runs,'saved_frame_steps_upper_bound':saved,
                             'fraction_upper_bound':saved/frame_work if frame_work else 0,
                             'examples':sorted(cohorts,key=lambda c:len(c['members']),reverse=True)[:3]}
    stats = {'label': label, 'scope': data.get('scope', {}).get('scope_id'),
             'groups': len(groups), 'runs': len(runs), 'tracker_frame_work': frame_work,
             'ideal_unique_crop_frames': len(unique_features),
             'exact_visual_reuse_upper_bound_fraction': 1-len(unique_features)/frame_work if frame_work else 0,
             'same_crop_direction_start_end_multi_buckets': len(same_inventory),
             'same_crop_direction_start_multi_buckets': len(same_start),
             'same_crop_direction_start_end_runs': sum(map(len,same_inventory)),
             'strict_grouped_frame_step_reduction': strict_reduction,
             'strict_grouped_frame_step_fraction': strict_reduction/frame_work if frame_work else 0,
             'same_crop_direction_start_varying_end_step_reduction':start_reduction,
             'same_crop_direction_start_varying_end_fraction':start_reduction/frame_work if frame_work else 0,
             'near_crop_same_frame_inventory_greedy_upper_bounds':nearby,
             'feature_preparation_summed_per_run':dict(feature_counts),
             'unique_full_image_crop_frame_loader_identities':len(identity_features),
             'unique_full_identities_per_recorded_worker':len(worker_identity_features),
             'families_seen_on_multiple_recorded_workers':sum(len(workers)>1 for workers in family_workers.values()),
             'feature_cache_counter_delta_sums':dict(cache_delta),
             'timings_summed_worker_seconds_not_walltime':dict(timing_counts),
             'cache_snapshot_maxima_not_delta_counts':dict(cache_max),
             'strict_grouped_examples': sorted(same_inventory,key=lambda b:len(b),reverse=True)[:8]}
    if identity_sets is not None:
        identity_sets[0].update(identity_features)
        identity_sets[1].update(worker_identity_features)
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    path=pathlib.Path(args.path)
    rows=[]
    identity_sets=(set(),set())
    error=None
    if path.suffix=='.tar':
        try:
            with tarfile.open(path,mode='r:') as archive:
                for member in archive:
                    if member.isfile() and member.name.endswith('/evidence/index.json') and not any(
                        token in member.name for token in ('crop_retries/','retry_attempts/','adaptive_final/')):
                        with archive.extractfile(member) as handle:
                            data=json.load(handle)
                        rows.append(inspect_index(data,member.name,identity_sets=identity_sets))
        except (tarfile.TarError, EOFError, ValueError) as exc:
            error=repr(exc)
    else:
        rows.append(inspect_index(json.loads(path.read_text(encoding='utf-8')),str(path),identity_sets=identity_sets))
    payload={'path':str(path),'archive_tail_error':error,'indices':rows}
    totals=collections.Counter()
    for row in rows:
        for key in ('groups','runs','tracker_frame_work','ideal_unique_crop_frames',
                    'strict_grouped_frame_step_reduction','same_crop_direction_start_end_runs'):
            totals[key]+=row[key]
        totals['same_crop_direction_start_varying_end_step_reduction']+=row['same_crop_direction_start_varying_end_step_reduction']
    payload['totals']=dict(totals)
    payload['totals']['global_unique_full_identities']=len(identity_sets[0])
    payload['totals']['global_unique_full_identities_per_recorded_worker']=len(identity_sets[1])
    if totals['tracker_frame_work']:
        payload['totals']['strict_grouped_frame_step_fraction']=totals['strict_grouped_frame_step_reduction']/totals['tracker_frame_work']
    pathlib.Path(args.out).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ('strict_grouped_examples','near_crop_same_frame_inventory_greedy_upper_bounds')}))
    print('TOTALS',json.dumps(payload['totals']))
    if error:
        print('Tail error:',error)


if __name__=='__main__':
    main()
