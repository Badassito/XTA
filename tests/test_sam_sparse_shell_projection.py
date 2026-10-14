"""Encoded SAM input must equal the unchanged dense CPU shell projectors."""
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.sam_sparse_shell_projection import project_shell_sparse_store, SparseShellProjectionUnavailable
from XTA.runtime import close_memmap_array_without_flush
from tests.test_sam_view_orientations import VIEWS, SOURCE_SHAPE

VIEWS=tuple(view for view in VIEWS if view.family in ('radial','spherical'))


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    monkeypatch.setenv('YOLO_TTA_GPU_RADIAL_BACKPROJECT','0')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT','0')


def decode(path):
    with closing(RawBBoxMaskStore.open(path,mmap_payload=True)) as store:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])


def dense_oracle(data,view,shape,path):
    from XTA.cylindrical_projection import backproject_radial_volume_to_volume
    from XTA.spherical_projection import backproject_spherical_volume_to_volume
    function=backproject_radial_volume_to_volume if view.family=='radial' else backproject_spherical_volume_to_volume
    result=function(data,view,path,'dense oracle',out_shape_tyx=shape,workers=1)
    try:
        return np.array(result,copy=True)
    finally:
        close_memmap_array_without_flush(result,unlink_path=path)


@pytest.mark.parametrize('view',VIEWS,ids=lambda view:view.name)
@pytest.mark.parametrize('case',('sparse','edges_rescaled','dense','empty'))
def test_encoded_matches_dense_geometry_for_every_shell_orientation(tmp_path,view,case):
    native=(view.src_h,view.src_w) if case!='edges_rescaled' else (5,7)
    data=np.zeros((view.num_slices,*native),np.uint8)
    if case=='dense':
        data[:]=255
    elif case=='sparse':
        data[len(data)//2,max(0,native[0]//2-1):native[0]//2+2,
             max(0,native[1]//2-1):native[1]//2+2]=1
    elif case=='edges_rescaled':
        data[0,:,0]=255
        data[-1,-1,:]=255
        data[:,0,-1]=255
    shape=(9,7,13) if case=='edges_rescaled' else SOURCE_SHAPE
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    expected=dense_oracle(data,view,shape,tmp_path/'expected.dat')
    result=project_shell_sparse_store(source,view,tmp_path/'result.cvol',out_shape_tyx=shape,workers=3)
    np.testing.assert_array_equal(decode(result['path']),expected)
    assert result['native_decode_bytes']==0
    assert result['native_logical_bytes']==data.size
    assert result['foreground_voxels']==np.count_nonzero(expected)


@pytest.mark.parametrize('family',('radial','spherical'))
@pytest.mark.parametrize('format_name',(CVOL_FORMAT,INTERNAL_PACKED_CVOL_FORMAT))
def test_raw_and_packed_input_require_no_dense_or_slice_decode(tmp_path,monkeypatch,family,format_name):
    view=next(view for view in VIEWS if view.family==family)
    data=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    data[:,1:3,1:4]=255
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=format_name,workers=1)
    expected=dense_oracle(data,view,SOURCE_SHAPE,tmp_path/'expected.dat')
    with closing(RawBBoxMaskStore.open(source,mmap_payload=True)) as store:
        for name in ('decode_slice','decode_slice_crop','fill_decoded_slice_into'):
            monkeypatch.setattr(store,name,lambda *a,**k:pytest.fail('Encoded projection decoded input'))
        result=project_shell_sparse_store(store,view,tmp_path/'result.cvol',out_shape_tyx=SOURCE_SHAPE)
        assert store.index is not None  # A borrowed store remains caller-owned.
    np.testing.assert_array_equal(decode(result['path']),expected)


@pytest.mark.parametrize('family',('radial','spherical'))
def test_failure_discards_partial_output_and_keeps_source_readable(tmp_path,monkeypatch,family):
    from XTA import sam_sparse_shell_projection as module
    view=next(view for view in VIEWS if view.family==family)
    data=np.ones((view.num_slices,view.src_h,view.src_w),np.uint8)
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    original=module.IncrementalRawBBoxMaskStoreWriter.finalize
    def failed(writer):
        raise OSError('controlled sealing failure')
    monkeypatch.setattr(module.IncrementalRawBBoxMaskStoreWriter,'finalize',failed)
    target=tmp_path/'result.cvol'
    with pytest.raises(OSError,match='controlled sealing failure'):
        project_shell_sparse_store(source,view,target,out_shape_tyx=SOURCE_SHAPE,workers=2)
    assert not target.exists()
    np.testing.assert_array_equal(decode(source),data)
    monkeypatch.setattr(module.IncrementalRawBBoxMaskStoreWriter,'finalize',original)


def test_radial_planfree_uses_same_kernel_without_dense_source(tmp_path,monkeypatch):
    from XTA import cylindrical_projection as radial
    view=next(view for view in VIEWS if view.family=='radial')
    data=(np.random.default_rng(9).random((view.num_slices,view.src_h,view.src_w))<.25).astype(np.uint8)
    def refuse(*args,**kwargs):
        raise radial._RadialPlanePlanTooLarge('bounded fixture')
    monkeypatch.setattr(radial,'_radial_plane_plan',refuse)
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    expected=dense_oracle(data,view,SOURCE_SHAPE,tmp_path/'expected.dat')
    result=project_shell_sparse_store(source,view,tmp_path/'result.cvol',out_shape_tyx=SOURCE_SHAPE)
    np.testing.assert_array_equal(decode(result['path']),expected)


def test_preflight_workspace_refusal_never_publishes(tmp_path,monkeypatch):
    view=next(view for view in VIEWS if view.family=='spherical')
    data=np.ones((view.num_slices,view.src_h,view.src_w),np.uint8)
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB','1')
    target=tmp_path/'result.cvol'
    with pytest.raises(SparseShellProjectionUnavailable,match='workspace'):
        project_shell_sparse_store(source,view,target,out_shape_tyx=SOURCE_SHAPE)
    assert not target.exists()


@pytest.mark.parametrize('route',('implicit','explicit_cpu','explicit_gpu','full_bbox','preflight','corrupt'))
def test_sam_dispatch_preserves_explicit_gpu_and_only_preflight_can_fallback(tmp_path,monkeypatch,route):
    from XTA import assembly, sam_sparse_shell_projection as module
    family='spherical'
    view=next(view for view in VIEWS if view.family==family and not view.spherical_tilted_source)
    data=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    data[:,1:3,1:4]=1
    if route=='full_bbox':data[:]=1
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    flag='YOLO_TTA_GPU_'+family.upper()+'_BACKPROJECT'
    monkeypatch.delenv(flag,raising=False)
    if route=='explicit_cpu':monkeypatch.setenv(flag,'0')
    if route=='explicit_gpu':monkeypatch.setenv(flag,'1')
    calls=[]
    original=module.project_shell_sparse_store
    def compact(*args,**kwargs):
        calls.append('compact')
        if route=='preflight':raise SparseShellProjectionUnavailable('controlled qualification refusal')
        if route=='corrupt':raise ValueError('controlled corrupt data')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'project_shell_sparse_store',compact)
    def dense(*args,**kwargs):
        calls.append('dense')
        raise LookupError('legacy dense route reached before any GPU query')
    monkeypatch.setattr('XTA.interpolation.materialize_raw_bbox_mask_store_workspace',dense)
    root=tmp_path/'work';root.mkdir()
    old=assembly.final_source_output_shape();assembly.set_final_source_output_shape(SOURCE_SHAPE)
    try:
        def invoke():
            return assembly._sam_directional_source_backing(source,view=view,stage='test',has_foreground=True,
                model_name='test',source='fullframe',pass_index=1,tile_config_id='',workers=1,_work_root=root)
        if route in ('explicit_gpu','full_bbox','preflight'):
            with pytest.raises(LookupError,match='legacy dense'):
                invoke()
            assert calls==(['compact','dense'] if route=='preflight' else ['dense'])
        elif route=='corrupt':
            with pytest.raises(ValueError,match='corrupt data'):invoke()
            assert calls==['compact']
        else:
            assert Path(invoke()['path']).exists()
            assert calls==['compact']
    finally:
        assembly.set_final_source_output_shape(old)


@pytest.mark.parametrize('view',tuple(view for view in VIEWS
    if view.family=='radial' or (view.family=='spherical' and view.spherical_tilted_source)),ids=lambda view:view.name)
@pytest.mark.parametrize('flag_value',(None,'0'))
def test_unqualified_shell_preserves_legacy_route_for_default_and_explicit_cpu(
        tmp_path,monkeypatch,view,flag_value):
    from XTA import assembly, sam_sparse_shell_projection as module
    data=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    data[:,1:3,1:4]=1
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    flag='YOLO_TTA_GPU_'+view.family.upper()+'_BACKPROJECT'
    if flag_value is None:monkeypatch.delenv(flag,raising=False)
    else:monkeypatch.setenv(flag,flag_value)
    def unexpected(*args,**kwargs):
        pytest.fail('Unqualified shell view entered automatic encoded projection')
    monkeypatch.setattr(module,'encoded_shell_has_unused_canvas',unexpected)
    monkeypatch.setattr(module,'project_shell_sparse_store',unexpected)
    calls=[]
    def dense(path,*args,**kwargs):
        calls.append(Path(path))
        raise LookupError('legacy backing reached without GPU query')
    monkeypatch.setattr('XTA.interpolation.materialize_raw_bbox_mask_store_workspace',dense)
    root=tmp_path/'work';root.mkdir()
    old=assembly.final_source_output_shape();assembly.set_final_source_output_shape(SOURCE_SHAPE)
    try:
        with pytest.raises(LookupError,match='legacy backing'):
            assembly._sam_directional_source_backing(source,view=view,stage='test',has_foreground=True,
                model_name='test',source='fullframe',pass_index=1,tile_config_id='',workers=1,_work_root=root)
        assert calls==[source]
    finally:
        assembly.set_final_source_output_shape(old)


@pytest.mark.parametrize('mutation',('kind','bounds','truncated'))
def test_malformed_encoded_input_is_not_a_capability_fallback(tmp_path,mutation):
    view=next(view for view in VIEWS if view.family=='radial')
    data=np.ones((view.num_slices,view.src_h,view.src_w),np.uint8)
    source=tmp_path/'source.cvol'
    write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    with closing(RawBBoxMaskStore.open(source,mmap_payload=True)) as store:
        store.index=store.index.copy()
        if mutation=='kind':store.index[0]['kind']=3
        elif mutation=='bounds':store.index[0]['x1']=store.shape[2]+1
        else:store.index[0]['offset']=2**50
        with pytest.raises((ValueError,OSError)) as error:
            project_shell_sparse_store(store,view,tmp_path/'output',out_shape_tyx=SOURCE_SHAPE)
        assert not isinstance(error.value,SparseShellProjectionUnavailable)
    assert not (tmp_path/'output').exists()
