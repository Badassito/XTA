"""Opt-in real CUDA through packed-parent publication and TTA preparation."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import assembly,backprojection as bp,geometry,packed_tilted_projection as packed,runtime
from XTA.interpolation import RawBBoxMaskStore,materialize_raw_bbox_mask_store_workspace
from XTA.native_pull_cuda import NativePullCudaSession
from XTA.projection_coverage_cpu import prepare_native_pull_plan
from XTA.view_prepare import AdmittedViewPrepare,_array_lifetime_owner
from tests.reference_backends.native_pull import iter_destination_samples


pytestmark=pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA')!='1',
                            reason='Explicit GPU lock ownership and actual CUDA required')


def _views():
    result=[]
    work=(33,64,80)
    for base in ('transverse','sagittal','coronal'):
        tilted=next(v for v in geometry._build_tilted_view_infos(*work,tilt_views=(base,),
            tilt_angles=(30.,),tilt_directions=('horizontal',)) if v.tilt_angle_deg==-30.)
        result.extend((tilted,geometry._build_azimuthal_view_info(*work,base_view=base,
            azimuth_angle=40.,azimuthal_native_raster=0,request_token='packed-integration',
            tilted_source=tilted)))
    return result


def _oracle(source,view,target):
    result=np.zeros(target,np.uint8)
    for destination,frame,row,column in iter_destination_samples(view,source.shape,target,chunk_voxels=113):
        values=source[frame,row,column]
        result.reshape(-1)[destination[values!=0]]=1
    return result


def _decode(path):
    store=RawBBoxMaskStore.open(Path(path),mmap_payload=True)
    try:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])
    finally:
        store.close()


def _hashes(path):
    return {str(file.relative_to(path)):hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob('*') if file.is_file()}


@pytest.fixture
def actual_cuda(monkeypatch,tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    for flag in ('YOLO_TTA_GPU_BACKPROJECT','YOLO_TTA_NATIVE_PULL_CUDA',
                 'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT','YOLO_TTA_PACKED_PARENT'):
        monkeypatch.setenv(flag,'1')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND','compiled')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS','0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB','0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR',str(tmp_path/'telemetry'))
    monkeypatch.setattr(assembly,'nrrd_layer_sink',lambda:None)
    bp._reset_main_process_gpu_stage_coordinator()
    sessions=[]
    acquire=bp._try_native_pull_encoded_cuda_stage
    def track(*args,**kwargs):
        stage=acquire(*args,**kwargs)
        assert stage is not None,'Real packed-parent CUDA integration cannot fall back to CPU'
        sessions.append(stage.session)
        return stage
    monkeypatch.setattr(bp,'_try_native_pull_encoded_cuda_stage',track)
    yield sessions
    import torch
    torch.cuda.synchronize()
    assert sessions,'The real encoded CUDA factory was never called'
    assert all(session._closed and session._pool.total_bytes()==0 for session in sessions)
    snapshot=bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
    assert not snapshot['stage_leases'] and not snapshot['resident_owners']
    bp._reset_main_process_gpu_stage_coordinator()


def _assert_cuda_telemetry(view,target):
    telemetry=runtime.runtime_telemetry()
    with telemetry.lock:
        records=[value for key,value in telemetry.gauges.items()
                 if key.startswith('projection.native_destination_pull.live.'+view.name+'.encoded.')]
    assert records
    record=records[-1]
    assert record['state']=='complete'
    assert record['backend']=='cuda_native_encoded'
    assert record['gpu_planes']==target[0] and record['cuda_handoff_plane']==0


@pytest.mark.parametrize('view',_views(),ids=lambda v:v.name)
def test_real_packed_projector_cuda_preserves_frozen_oracle_and_input(tmp_path,monkeypatch,actual_cuda,view):
    source=np.random.default_rng(160381).integers(0,2,(view.num_slices,view.src_h,view.src_w),dtype=np.uint8)
    source[::4]=0
    source[:,:,0]=0
    target=(27,97,111)
    expected=_oracle(source,view,target)
    input_path=tmp_path/'native.cvol'
    pack_stats=packed.pack_terminal_native_source(source,input_path,desc='Actual CUDA packed parent')
    if os.name=='nt':
        assert pack_stats['payload_backing']!='planned_memfd'
    before=_hashes(input_path)
    # Decoder methods are forbidden while publishing; the result is decoded
    # only after the entire actual GPU transaction has closed its residency.
    with mock.patch.object(RawBBoxMaskStore,'decode_slice',side_effect=AssertionError('native dense decode')):
        stats=packed.project_tilted_sparse_store(input_path,view,tmp_path/'output.cvol',
                                               out_shape_tyx=target,workers=3)
    np.testing.assert_array_equal(_decode(tmp_path/'output.cvol'),expected)
    assert _hashes(input_path)==before
    assert stats['gpu_planes']==target[0] and stats['cuda_handoff_plane']==0
    assert stats['native_decode_bytes_avoided']==source.size
    _assert_cuda_telemetry(view,target)
    assert actual_cuda[-1].encoded_input
    runtime.release_memfd_owners_under(input_path)


def _task(root,view):
    shape=(view.num_slices,view.src_h,view.src_w)
    path=root/'native.u8.dat'
    volume=np.memmap(path,mode='w+',dtype=np.uint8,shape=shape)
    volume[:]=0
    volume[:,max(1,shape[1]//4):max(2,3*shape[1]//4),
             max(1,shape[2]//4):max(2,3*shape[2]//4)]=1
    expected=np.array(volume)
    owner_ref=weakref.ref(_array_lifetime_owner(volume))
    retired=[]
    task=AdmittedViewPrepare(admission=SimpleNamespace(reserve=lambda *_a:nullcontext()),
        transient_bytes=1,model_name='model',view=view,union_mm=volume,confmap_mm=None,
        d1_shadow_path=None,union_path=path,confmap_path=None,temp_dir=root,dense_tiling_active=False,
        min_conf=0.,min_radius=0.,interpolation_distance=0,interpolation_walk_back=1,
        interpolation_candidates=1,interpolation_passes=1,interpolation_min_radius=0.,
        interpolation_search_angle=0.,keep_temp_artifacts=False,slice_workers=2,
        interpolation_task_workers=1,component_layers_needed=True,precleaned_slice_cleanup=True,
        hole_fill_done_on_device=True,slice_meta=None,fuse_azimuthal_component_layers=lambda:False,
        component_ref_dense_retirement_active=True,preinterpolation_layer_already_published=False,
        parent_mask_ready_callback=None,submit_component_projection=lambda *_a,**_k:None,
        materialize_workspace=materialize_raw_bbox_mask_store_workspace,
        prepare=assembly.prepare_view_volume_after_fullframe,
        mask_retired_callback=lambda *args:retired.append(args))
    return task,expected,owner_ref,retired


@pytest.mark.parametrize('view',_views(),ids=lambda v:v.name)
def test_actual_tta_prepare_retires_dense_before_encoded_cuda_publication(tmp_path,monkeypatch,actual_cuda,view):
    task,source,owner_ref,retired=_task(tmp_path,view)
    target=(27,97,111)
    monkeypatch.setattr(assembly,'_FINAL_SOURCE_OUTPUT_SHAPE_TYX',target)
    expected=_oracle(source,view,target)
    original=packed.project_tilted_sparse_store
    def require_retired(*args,**kwargs):
        assert task.union_mm is None and owner_ref() is None and retired
        return original(*args,**kwargs)
    monkeypatch.setattr(packed,'project_tilted_sparse_store',require_retired)
    with mock.patch.object(RawBBoxMaskStore,'decode_slice',side_effect=AssertionError('native dense decode')):
        result=task()
    assert owner_ref() is None and len(retired)==1
    assert result.final_view_volume_mm is None and result.native_support_mm is None
    assert result._packed_native_source_retired and len(result.nrrd_layers)==1
    np.testing.assert_array_equal(_decode(result.nrrd_layers[0].path),expected)
    _assert_cuda_telemetry(view,target)
    assert actual_cuda[-1].encoded_input
    assert not (tmp_path/'packed_native_parents'/'model'/view.name).exists()


@pytest.mark.parametrize('base',['transverse','sagittal','coronal'])
def test_actual_encoded_cuda_uses_idle_worker_before_global_inference_drain(
        tmp_path,monkeypatch,actual_cuda,base):
    """Reproduce 160587's idle GPU + inference-priority + native packed parent."""
    work,target=(33,64,80),(27,97,111)
    view=geometry._build_azimuthal_view_info(*work,base_view=base,azimuth_angle=40.,
        azimuthal_native_raster=0,request_token='pre-drain-regression')
    source=np.random.default_rng(160587).integers(0,2,
        (view.num_slices,view.src_h,view.src_w),dtype=np.uint8)
    source[::4]=0
    source[:,:,:3]=0
    expected=_oracle(source,view,target)
    input_path,output_path=tmp_path/'native.cvol',tmp_path/'projected.cvol'
    monkeypatch.setenv('YOLO_TTA_MEMFD_WORKSPACES','1')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_RAM_RESERVE_MIB','0')
    pack_stats=packed.pack_terminal_native_source(source,input_path,
        desc='Actual pre-drain native source retirement')
    # Linux exercises the real memfd writer when supported. Windows has no
    # memfd syscall and uses the production packed mmap fallback explicitly.
    if runtime.memfd_workspace_enabled():
        assert pack_stats['payload_backing']=='planned_memfd'
    else:
        assert pack_stats['payload_backing']=='disk'
    before_hashes=_hashes(input_path)
    store=RawBBoxMaskStore.open(input_path,mmap_payload=True)
    assert store._packbits_payload
    coordinator=bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(False)
    monkeypatch.setattr(bp,'gpu_worker_aux_interpolation_pool',lambda:None)
    before=coordinator.snapshot()
    assert before['inference_priority_active'] and not before['inference_inflight']
    assert not before['inference_asset_retirement_pending'] and not before['retired_inference_devices']
    assert not before['pending_inference_backlog']
    admitted=[]
    acquire=bp._try_native_pull_encoded_cuda_stage
    def capture_admission(*args,**kwargs):
        stage=acquire(*args,**kwargs)
        assert stage is not None and stage.device_index==0
        assert stage.session.encoded_input
        snapshot=coordinator.snapshot()
        assert snapshot['inference_priority_active']
        assert not snapshot['pending_inference_backlog'] and not snapshot['inference_inflight']
        assert not snapshot['inference_asset_retirement_pending']
        assert not snapshot['retired_inference_devices']
        assert snapshot['stage_leases'][0].startswith('Native source projection ')
        assert not coordinator.can_dispatch_inference(0)
        assert not coordinator.begin_inference(0)
        admitted.append(snapshot)
        return stage
    monkeypatch.setattr(bp,'_try_native_pull_encoded_cuda_stage',capture_admission)
    try:
        with mock.patch.object(RawBBoxMaskStore,'decode_slice',side_effect=AssertionError('native dense decode')):
            stats=packed.project_tilted_sparse_store(store,view,output_path,
                out_shape_tyx=target,workers=3)
        assert admitted and len(actual_cuda)==1
        np.testing.assert_array_equal(_decode(output_path),expected)
        assert _hashes(input_path)==before_hashes
        assert stats['gpu_planes']==target[0] and stats['cuda_handoff_plane']==0
        _assert_cuda_telemetry(view,target)
        after=coordinator.snapshot()
        assert not after['stage_leases'] and not after['provisional_stage_devices']
        assert after['inference_priority_active'] and not after['retired_inference_devices']
        assert coordinator.can_dispatch_inference(0)
        assert coordinator.begin_inference(0), 'GPU stage did not return its inference device'
        coordinator.finish_inference(0)
        (tmp_path/'native-pre-drain-gpu-proof.json').write_text(json.dumps(dict(
            view=view.name,source_shape=source.shape,output_shape=target,
            source_payload_backing=pack_stats['payload_backing'],
            memfd_supported=runtime.memfd_workspace_enabled(),oracle_parity=True,
            before=before,during=admitted,after=after,
            gpu_planes=stats['gpu_planes'],cuda_handoff_plane=stats['cuda_handoff_plane'],
            inference_resumed=True),indent=2),encoding='utf-8')
    finally:
        store.close()
        runtime.release_memfd_owners_under(input_path)


@pytest.mark.parametrize('base',['transverse','sagittal','coronal'])
@pytest.mark.parametrize('angular',[False,True],ids=['tilted','tilted-azimuthal'])
def test_real_encoded_stage_int64_destination_span_without_full_output(tmp_path,actual_cuda,base,angular):
    work,target=(1693,2048,2048),(1123,3022,3064)
    tilted=next(v for v in geometry._build_tilted_view_infos(*work,tilt_views=(base,),
        tilt_angles=(23.,),tilt_directions=('vertical',)) if v.tilt_angle_deg==23.)
    z,y,x=901,2301,2501
    stack,vertical,_={'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(2,0,1)}[base]
    coords=[(i+.5)*n/o-.5 for i,n,o in zip((z,y,x),work,target)]
    center=coords[stack]-math.tan(math.radians(23.))*(coords[vertical]-(work[vertical]-1)/2)
    origin=math.floor(center)
    if angular:
        view=geometry._build_azimuthal_view_info(*work,base_view=base,azimuth_angle=45.,
            azimuthal_native_raster=7,request_token='encoded-int64',tilted_source=tilted)
        view=replace(view,tilt_frame_start=origin,azimuths_deg=(7.,19.,62.,144.),num_slices=4)
    else:
        view=replace(tilted,tilt_frame_start=origin,tilt_frame_stop=origin+1,num_slices=2)
    source=np.random.default_rng(772).integers(0,2,(view.num_slices,257,257),dtype=np.uint8)
    path=tmp_path/'native.cvol'
    packed.pack_terminal_native_source(source,path,desc='Actual CUDA int64 encoded source')
    store=RawBBoxMaskStore.open(path,mmap_payload=True)
    from XTA.sparse_projection import _SparseMaskSampler
    sampler=_SparseMaskSampler(store)
    plan=prepare_native_pull_plan(view,store.shape,target,max_plan_bytes=16*1024**2,cache_plane=False)
    first=z*target[1]*target[2]+y*target[2]+x
    assert first>2**32
    count,bbox=4097,(z,y,x,z+1,y+3,x+173)
    expected=np.zeros(count,np.uint8)
    for destination,frame,row,column in iter_destination_samples(view,source.shape,target,
            first_flat=first,stop_flat=first+count,chunk_voxels=113,destination_bbox_tyx=bbox):
        values=source[frame,row,column]
        expected[destination[values!=0]-first]=1
    actual=np.full(count,177,np.uint8)
    stage=None
    try:
        stage=bp._try_native_pull_encoded_cuda_stage(sampler.encoded,store.index,
            store._packbits_payload,plan,view,max_output_voxels=count)
        assert stage is not None
        assert stage.session.plan.source_shape==source.shape
        stats=stage.pull_flat_into(actual,first_flat=first,destination_bbox_tyx=bbox)
        assert stats['backend']=='cuda_native_encoded_packed'
        assert stats['indexed_input_byte_reads']>0
        np.testing.assert_array_equal(actual,expected)
    finally:
        if stage is not None:
            stage.close()
        sampler.close()
        store.close()
        runtime.release_memfd_owners_under(path)
