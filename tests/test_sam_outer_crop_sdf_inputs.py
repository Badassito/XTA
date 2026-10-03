"""Original detector observations must be bound before SDF output allocation."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.generate_sam_outer_crop_sdf import fingerprint,sha,validate_original_inputs
from tools import generate_sam_outer_crop_sdf as generator


def sealed_inputs(tmp_path):
    root=tmp_path/'study';directory=root/'plans'/'development';directory.mkdir(parents=True)
    files=[];volume=np.zeros((3,8,9),np.uint8)
    for frame,x in ((0,3),(2,4)):
        mask=np.zeros((8,9),bool);mask[3,x]=True
        path=root/f'endpoint_{frame}.npy';np.save(path,mask);files.append(str(path));volume[frame]=mask
    snapshot=root/'endpoints.json';snapshot.write_text('{}',encoding='utf-8')
    dataset=dict(dataset_id='development',shape_tyx=list(volume.shape),source_frame_start=100,
        endpoint_local_frames=[0,2],endpoint_files=files,endpoint_sha256={path:sha(path)for path in files},
        evaluation_full_source_frame=101,image_path=str(root/'not_read_image.dat'))
    spec=dict(dataset_id='development_seen655',source_shape_tyx=list(volume.shape),source_frame_start=100,
        endpoint_full_source_frames=[100,102],middle_full_source_frame=101,image_path=dataset['image_path'],
        endpoint_metadata=str(snapshot),endpoint_metadata_sha256=sha(snapshot))
    constants={'datasets':[spec]};constants['constants_sha256']=fingerprint(constants)
    cp=root/'protocol_constants.json';cp.write_text(json.dumps(constants),encoding='utf-8')
    plan=directory/'B0.plan.json';plan.write_text(json.dumps({'dataset':dataset}),encoding='utf-8')
    (directory/'dataset.json').write_text(json.dumps(dataset),encoding='utf-8')
    np.save(directory/'observations.npy',volume)
    manifest=dict(constants_file=str(cp),constants_sha256=constants['constants_sha256'],
        plan_files=[dict(file=str(plan),sha256=sha(plan),bytes=plan.stat().st_size)])
    manifest['manifest_sha256']=fingerprint(manifest)
    (root/'protocol_manifest_development.json').write_text(json.dumps(manifest),encoding='utf-8')
    return root


def test_exact_sealed_endpoint_volume_without_images_or_labels(tmp_path):
    root=sealed_inputs(tmp_path)
    _spec,_dataset,proof=validate_original_inputs(root,'development')
    assert proof['endpoint_planes_exact']and proof['all_other_planes_empty']
    assert proof['labels_read']is False and proof['images_read']is False


@pytest.mark.parametrize('corruption',['middle_voxel','endpoint_voxel','dataset','constants','plan','seal','endpoint_snapshot'])
def test_corruption_is_rejected_before_reference_output_exists(tmp_path,corruption):
    root=sealed_inputs(tmp_path);directory=root/'plans'/'development'
    if corruption in ('middle_voxel','endpoint_voxel'):
        path=directory/'observations.npy';volume=np.load(path);volume[1 if corruption=='middle_voxel'else 0,0,0]=1;np.save(path,volume)
    elif corruption=='dataset':
        path=directory/'dataset.json';data=json.loads(path.read_text());data['source_frame_start']=99;path.write_text(json.dumps(data))
    elif corruption=='constants':
        path=root/'protocol_constants.json';data=json.loads(path.read_text());data['datasets'][0]['middle_full_source_frame']=102;path.write_text(json.dumps(data))
    elif corruption=='plan':
        path=directory/'B0.plan.json';path.write_text(path.read_text()+' ')
    elif corruption=='seal':
        path=root/'protocol_manifest_development.json';data=json.loads(path.read_text());data['plan_files'][0]['sha256']='0'*64;path.write_text(json.dumps(data))
    else:(root/'endpoints.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):validate_original_inputs(root,'development')
    assert not(root/'sdf_references').exists()


def test_validated_hash_remains_the_generation_pin(tmp_path,monkeypatch):
    root=sealed_inputs(tmp_path)
    original=generator.validate_original_inputs
    def mutate_after_validation(*args):
        validated=original(*args)
        path=root/'plans'/'development'/'observations.npy'
        volume=np.load(path);volume[1,0,0]=1;np.save(path,volume)
        return validated
    monkeypatch.setattr(generator,'validate_original_inputs',mutate_after_validation)
    monkeypatch.setattr(generator.sys,'argv',['generate_sdf','--experiment',str(root),'--datasets','development'])
    for key in ('CUDA_VISIBLE_DEVICES','YOLO_TTA_GPU_INTERPOLATION','YOLO_TTA_GPU_SLICE_LABELING',
                'YOLO_TTA_GPU_SLICE_LABELING_PAIRS','YOLO_TTA_GPU_SLICE_LABELING_IN_CHILDREN',
                'YOLO_TTA_GPU_INTERPOLATION_RADIUS','YOLO_TTA_GPU_INTERPOLATION_REQUIRED','YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER'):
        monkeypatch.setenv(key,os.environ.get(key,''))
    original_open=np.lib.format.open_memmap
    def forbidden_allocation(*args,**kwargs):
        if kwargs.get('mode','r+')!='r':
            raise AssertionError('Corrupted observations reached output allocation')
        return original_open(*args,**kwargs)
    monkeypatch.setattr(np.lib.format,'open_memmap',forbidden_allocation)
    with pytest.raises(ValueError,match='pin changed before allocation'):generator.main()
    assert not(root/'sdf_references').exists()
