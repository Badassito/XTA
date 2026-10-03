"""Generate mask-only CPU SDF references from sealed original observations.

No images or annotations are read. This uses one unchanged production SDF pass
and records its native full-plane output separately from SAM resource coverage.
It is an accuracy reference, not a CPU/GPU performance comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_original_inputs(root,identifier):
    """Authenticate sealed metadata, then prove every original detector pixel."""
    import numpy as np
    root=Path(root).resolve()
    constants_path=root/'protocol_constants.json'
    constants=json.loads(constants_path.read_text('utf-8'))
    constants_digest=fingerprint({key:value for key,value in constants.items()if key!='constants_sha256'})
    if constants.get('constants_sha256')!=constants_digest:
        raise ValueError('Locked protocol constants changed')
    aliases={'development':'development_seen655','source_590_599':'followup_594','source_686_695':'followup_690'}
    specs={row['dataset_id']:row for row in constants['datasets']}
    spec=specs[aliases.get(identifier,identifier)]
    manifest_path=root/('protocol_manifest_development.json'if identifier=='development'else'protocol_manifest_followup.json')
    manifest=json.loads(manifest_path.read_text('utf-8'))
    if manifest.get('manifest_sha256')!=fingerprint({key:value for key,value in manifest.items()if key!='manifest_sha256'}):
        raise ValueError('Numeric plan seal changed')
    if manifest.get('constants_sha256')!=constants_digest or Path(manifest['constants_file']).resolve()!=constants_path:
        raise ValueError('Numeric plan seal has foreign protocol constants')
    plan_path=root/'plans'/identifier/'B0.plan.json'
    matches=[row for row in manifest['plan_files']if Path(row['file']).resolve()==plan_path]
    if len(matches)!=1 or matches[0]['sha256']!=sha(plan_path)or matches[0]['bytes']!=plan_path.stat().st_size:
        raise ValueError('Original detector plan is not authenticated by its numeric seal')
    plan=json.loads(plan_path.read_text('utf-8'))
    dataset_path=plan_path.parent/'dataset.json'
    dataset=json.loads(dataset_path.read_text('utf-8'))
    if dataset!=plan['dataset']:
        raise ValueError('Dataset metadata differs from sealed detector plan')
    shape=tuple(spec['source_shape_tyx'])
    first=int(spec['source_frame_start'])
    indices=[int(frame)-first for frame in spec['endpoint_full_source_frames']]
    if (dataset['dataset_id']!=identifier or tuple(dataset['shape_tyx'])!=shape or
        int(dataset['source_frame_start'])!=first or dataset['endpoint_local_frames']!=indices or
        int(dataset['evaluation_full_source_frame'])!=int(spec['middle_full_source_frame']) or
        Path(dataset['image_path']).resolve()!=Path(spec['image_path']).resolve()):
        raise ValueError('Dataset frame/shape identity differs from locked constants')
    if sha(spec['endpoint_metadata'])!=spec['endpoint_metadata_sha256']:
        raise ValueError('Original detector endpoint snapshot changed')
    files=dataset['endpoint_files']
    if len(files)!=len(indices)or len(set(indices))!=len(indices)or any(not 0<=index<shape[0]for index in indices):
        raise ValueError('Invalid original endpoint inventory')
    endpoint_hashes={path:sha(path)for path in files}
    if endpoint_hashes!=dataset['endpoint_sha256']:
        raise ValueError('Endpoint files differ from sealed original inputs')
    source=plan_path.parent/'observations.npy'
    input_sha=sha(source)
    observations=np.load(source,mmap_mode='r',allow_pickle=False)
    if observations.shape!=shape or observations.dtype not in (np.dtype('uint8'),np.dtype('bool')):
        raise ValueError('Original observations differ from sealed native source shape/type')
    endpoints={index:np.load(path,mmap_mode='r',allow_pickle=False)for index,path in zip(indices,files)}
    for index,endpoint in endpoints.items():
        if endpoint.shape!=shape[1:]:
            raise ValueError('Original endpoint union differs from native canvas')
    for frame in range(shape[0]):
        if frame in endpoints:
            if not np.array_equal(observations[frame],endpoints[frame]!=0):
                raise ValueError('Observation endpoint pixels differ from sealed detector union')
        elif observations[frame].any():
            raise ValueError('Non-endpoint observation frame must be empty')
    if sha(source)!=input_sha or {path:sha(path)for path in files}!=endpoint_hashes:
        raise ValueError('Original input changed during validation')
    proof=dict(schema='xta.sam_outer_crop_original_input_binding/1',dataset_id=identifier,
        constants_file=str(constants_path),constants_sha256=constants_digest,
        numeric_seal_file=str(manifest_path),numeric_seal_sha256=manifest['manifest_sha256'],
        sealed_plan_file=str(plan_path),sealed_plan_file_sha256=sha(plan_path),
        dataset_file=str(dataset_path),dataset_file_sha256=sha(dataset_path),
        original_observations_file=str(source),original_observations_sha256=input_sha,
        original_endpoint_file_sha256=endpoint_hashes,endpoint_local_frames=indices,
        endpoint_planes_exact=True,all_other_planes_empty=True,labels_read=False,images_read=False)
    return spec,dataset,proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',default=['development','source_590_599','source_686_695'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    for key in ('YOLO_TTA_GPU_INTERPOLATION','YOLO_TTA_GPU_SLICE_LABELING',
                'YOLO_TTA_GPU_SLICE_LABELING_PAIRS','YOLO_TTA_GPU_SLICE_LABELING_IN_CHILDREN',
                'YOLO_TTA_GPU_INTERPOLATION_RADIUS','YOLO_TTA_GPU_INTERPOLATION_REQUIRED'):
        os.environ[key]='0'
    os.environ['YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER']='0'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from XTA import interpolation
    interpolation.cv2.setNumThreads(1)
    root=args.experiment.resolve()
    settings=dict(max_slice_distance=15,search_angle_deg=30.,interpolation_walk_back=0,
        interpolation_candidates=1,interpolate_min_radius=3.,passes=1)
    for identifier in args.datasets:
        spec,dataset_metadata,input_binding=validate_original_inputs(root,identifier)
        source=root/'plans'/identifier/'observations.npy'
        output=root/'sdf_references'/identifier
        if(output/'reference.json').exists():
            raise FileExistsError('Reference already published; refusing to overwrite')
        input_sha=input_binding['original_observations_sha256']
        if sha(source)!=input_sha:
            raise ValueError('Validated original observation pin changed before allocation')
        endpoint_hashes={path:sha(path)for path in dataset_metadata['endpoint_files']}
        if endpoint_hashes!=dataset_metadata['endpoint_sha256']:
            raise ValueError('Endpoint files differ from sealed original inputs')
        observations=np.load(source,mmap_mode='r',allow_pickle=False)
        if tuple(observations.shape)!=tuple(spec['source_shape_tyx']):
            raise ValueError('Original observations differ from sealed native source shape')
        if sha(source)!=input_sha:
            raise ValueError('Validated original observation pin changed during opening')
        output.mkdir(parents=True,exist_ok=True)
        result_path=output/'prediction.npy'
        prediction=np.lib.format.open_memmap(result_path,mode='w+',dtype=observations.dtype,shape=observations.shape)
        prediction[:]=observations
        prediction.flush()
        started=time.perf_counter()
        # The normal creation call is unconditional. Prevent even lazy CuPy
        # probing; CPU SDF geometry/radius/render code remains unchanged.
        with mock.patch.object(interpolation,'create_cuda_interpolation_renderer',return_value=(None,'forced CPU-only reference')):
            stats=interpolation.interpolate_view_volume_pass_inplace(prediction,output/'work','sdf_reference',
                settings['max_slice_distance'],settings['search_angle_deg'],settings['interpolation_walk_back'],
                settings['interpolation_candidates'],settings['interpolate_min_radius'],keep_temp=False,
                prefer_memory=True,reserve_bytes=0,workers=args.workers)
        elapsed=time.perf_counter()-started
        prediction.flush()
        for key in ('interpolation_render_backend','interpolation_radius_backend'):
            if 'cuda'in str(stats.get(key,'')).lower():
                raise RuntimeError('Unexpected GPU reference backend')
        middle=int(spec['middle_full_source_frame'])-int(spec['source_frame_start'])
        np.save(output/'evaluation_plane.npy',prediction[middle]!=0,allow_pickle=False)
        added=0
        additions_path=output/'selected_additions.npy'
        additions=np.lib.format.open_memmap(additions_path,mode='w+',dtype=np.bool_,shape=observations.shape)
        for frame in range(len(observations)):
            if np.any((observations[frame]!=0)&(prediction[frame]==0)):
                raise ValueError('SDF removed an original observation')
            additions[frame]=(prediction[frame]!=0)&(observations[frame]==0)
            added+=int(np.count_nonzero(additions[frame]))
        additions.flush()
        if sha(source)!=input_sha:
            raise ValueError('Original observation file changed during generation')
        if {path:sha(path)for path in endpoint_hashes}!=endpoint_hashes:
            raise ValueError('Original endpoint files changed during generation')
        record=dict(schema='xta.sam_outer_crop_sdf_reference/1',dataset_id=identifier,
            settings=settings,workers=args.workers,labels_read=False,images_read=False,
            source_shape_tyx=list(observations.shape),source_frame_start=spec['source_frame_start'],
            evaluation_full_source_frame=spec['middle_full_source_frame'],evaluation_cache_local=middle,
            original_observations_file=str(source),original_observations_sha256=input_sha,
            original_input_binding=input_binding,
            original_endpoint_file_sha256=endpoint_hashes,original_endpoint_files_preserved=True,
            prediction_file=str(result_path),prediction_sha256=sha(result_path),
            evaluation_plane_file=str(output/'evaluation_plane.npy'),evaluation_plane_sha256=sha(output/'evaluation_plane.npy'),
            selected_additions_file=str(additions_path),selected_additions_sha256=sha(additions_path),
            added_voxels=added,original_observations_preserved=True,statistics=stats,
            cpu_pass_wall_seconds=elapsed,timing_scope='One CPU production SDF pass only; reference construction, not a CPU/GPU speed comparison',
            forced_cpu=True,renderer_creation_disabled=True,cuda_visible_devices='-1',
            source_sha256=sha(interpolation.__file__),tool_sha256=sha(__file__),
            command=sys.argv,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (output/'reference.json').write_text(json.dumps(record,indent=2),'utf-8')
        print(json.dumps(dict(dataset=identifier,added_voxels=added,cpu_pass_seconds=elapsed,reference=str(output/'reference.json'))),flush=True)
        del additions,prediction,observations


if __name__=='__main__':
    main()
