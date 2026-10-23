"""Opt-in actual-device exact native pull parity, never a CPU fallback."""
from dataclasses import replace
import math
import os

import numpy as np
import pytest

from XTA import geometry
from XTA.native_pull_cuda import (NativePullCudaSession, NativePullCudaUnavailable,
                                  NativePullCudaUnsafeFailure)
from XTA.projection_coverage_cpu import prepare_native_pull_plan, pull_native_flat_into
from tests.reference_backends.native_pull import iter_destination_samples


def _views():
    views = []
    for base in ('transverse', 'sagittal', 'coronal'):
        for direction, angle in (('vertical', 23.), ('horizontal', -31.)):
            tilted = next(v for v in geometry._build_tilted_view_infos(9, 13, 17,
                tilt_views=(base,), tilt_angles=(abs(angle),), tilt_directions=(direction,))
                if v.tilt_angle_deg == angle)
            views.extend((tilted, geometry._build_azimuthal_view_info(9, 13, 17,
                base_view=base, azimuth_angle=45., azimuthal_native_raster=7,
                request_token='cuda-exact', tilted_source=tilted)))
        views.append(geometry._build_azimuthal_view_info(9, 13, 17, base_view=base,
            azimuth_angle=45., azimuthal_native_raster=7, request_token='cuda-exact'))
    return views


def _reference(source, view, shape, first, count, numeric, bbox):
    expected = np.zeros(count, np.uint8)
    for destination, frames, rows, columns in iter_destination_samples(view, source.shape,
            shape, first_flat=first, stop_flat=first+count, chunk_voxels=113,
            destination_bbox_tyx=bbox):
        selected = source[frames, rows, columns]
        if numeric:
            np.maximum.at(expected, destination-first, selected)
        else:
            expected[destination[selected != 0]-first] = 1
    return expected


def test_cuda_credit_refusal_precedes_import_or_allocation():
    view = _views()[0]
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    plan = prepare_native_pull_plan(view, source.shape, (14, 19, 23), max_plan_bytes=1024**2)
    with pytest.raises(NativePullCudaUnavailable) as caught:
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=1,
                              max_output_voxels=127)
    assert caught.value.reason_code=='device_credit'
    assert caught.value.details['required_device_bytes']>caught.value.details['allowed_device_bytes']==1


def test_numeric_packing_is_rejected():
    view = _views()[0]
    source = np.zeros((view.num_slices, 5, 5), np.uint8)
    plan = prepare_native_pull_plan(view, source.shape, (14, 19, 23), max_plan_bytes=1024**2)
    with pytest.raises(ValueError, match='Numeric max'):
        NativePullCudaSession(source, plan, device_index=0, max_device_bytes=1024**2,
                              max_output_voxels=127, scalar_max=True, packed_input=True)


def test_unsafe_failure_bypasses_recoverable_exception_fallback():
    assert issubclass(NativePullCudaUnsafeFailure, BaseException)
    assert not issubclass(NativePullCudaUnsafeFailure, Exception)


def test_optional_cupy_import_declines_before_device_ownership(monkeypatch):
    import builtins
    view=_views()[0]
    source=np.zeros((view.num_slices,5,5),np.uint8)
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    original_import=builtins.__import__
    def import_without_cupy(name,*args,**kwargs):
        if name=='cupy':
            raise ImportError('synthetic missing optional CuPy')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',import_without_cupy)
    with pytest.raises(NativePullCudaUnavailable,match='synthetic missing optional CuPy'):
        NativePullCudaSession(source,plan,device_index=0,max_device_bytes=1024**2,max_output_voxels=127)


_gpu = pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA') != '1',
                         reason='Explicit GPU lock ownership and CUDA qualification required')


@_gpu
@pytest.mark.parametrize('view', _views(), ids=lambda v: v.name)
@pytest.mark.parametrize('shape,reduced,cache', [((14,19,23),False,True),
                                               ((4,7,11),True,False),
                                               ((11,8,19),True,True)])
@pytest.mark.parametrize('numeric,packed', [(False,True),(False,False),(True,False)])
def test_actual_cuda_matches_compiled_and_frozen_numpy(view, shape, reduced, cache, numeric, packed):
    source_shape = (view.num_slices,5,5) if reduced else (view.num_slices,view.src_h,view.src_w)
    # Nonbinary values ensure packed input means any nonzero, including even values.
    source = np.random.default_rng(150772).choice(np.asarray([0,0,2,16,89,128,255],np.uint8),source_shape)
    source = source[:,::-1,::-1]  # Exercise bounded flattening of non-contiguous sources.
    original = source.copy()
    bbox = (1,1,2,shape[0],shape[1]-1,shape[2]-1)
    plan = prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=4*1024**2,cache_plane=cache)
    first, count = 13, math.prod(shape)-29
    expected = _reference(source,view,shape,first,count,numeric,bbox)
    cpu = np.empty(count,np.uint8)
    cpu_stats = pull_native_flat_into(source,plan,cpu,first_flat=first,scalar_max=numeric,destination_bbox_tyx=bbox)
    np.testing.assert_array_equal(cpu,expected)
    actual = np.empty(count,np.uint8)
    with NativePullCudaSession(source,plan,device_index=0,max_device_bytes=64*1024**2,
            max_output_voxels=127,scalar_max=numeric,packed_input=packed,max_upload_bytes=1032) as session:
        stats = [session.pull_flat_into(actual[offset:offset+127],first_flat=first+offset,
                                       destination_bbox_tyx=bbox) for offset in range(0,count,127)]
        assert session._pool.total_bytes() >= session.device_bytes
        assert session.device_bytes <= 64*1024**2
        assert sum(s['destination_voxels_visited'] for s in stats) == cpu_stats['destination_voxels_visited']
        assert sum(s['contribution_addresses'] for s in stats) == cpu_stats['contribution_addresses']
    assert session._pool.total_bytes() == 0
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(source,original)
    with pytest.raises(RuntimeError, match='closed'):
        session.pull_flat_into(np.empty(1,np.uint8),first_flat=0)


@_gpu
@pytest.mark.parametrize('family', ['tilted', 'tilted-azimuthal'])
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_actual_cuda_int64_high_coordinates_and_partial_spans(family, base):
    work, shape = (1693,2048,2048),(1123,3022,3064)
    tilted = next(v for v in geometry._build_tilted_view_infos(*work,tilt_views=(base,),
        tilt_angles=(23.,),tilt_directions=('vertical',)) if v.tilt_angle_deg == 23.)
    view = (tilted if family == 'tilted' else geometry._build_azimuthal_view_info(*work,
        base_view=base,azimuth_angle=45.,azimuthal_native_raster=7,request_token='int64',tilted_source=tilted))
    z,y,x = 901,2301,2501
    axes = {'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(2,0,1)}[base]
    coords = [(i+.5)*n/o-.5 for i,n,o in zip((z,y,x),work,shape)]
    stack,vertical,_ = axes
    center = coords[stack]-math.tan(math.radians(23.))*(coords[vertical]-(work[vertical]-1)/2)
    if family == 'tilted':
        view=replace(view,tilt_frame_start=math.floor(center),tilt_frame_stop=math.floor(center)+1,num_slices=2)
    else:
        view=replace(view,tilt_frame_start=math.floor(center),azimuths_deg=(7.,19.,62.,144.),num_slices=4)
    source=np.random.default_rng(772).integers(0,256,(view.num_slices,257,257),dtype=np.uint8)
    plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=8*1024**2,cache_plane=False)
    first=z*shape[1]*shape[2]+y*shape[2]+x
    assert first>2**32
    bbox=(z,y,x,z+1,y+3,x+173)
    expected=_reference(source,view,shape,first,4097,True,bbox)
    actual=np.empty_like(expected)
    with NativePullCudaSession(source,plan,device_index=0,max_device_bytes=16*1024**2,
            max_output_voxels=4097,scalar_max=True) as session:
        session.pull_flat_into(actual,first_flat=first,destination_bbox_tyx=bbox)
    np.testing.assert_array_equal(actual,expected)


@_gpu
def test_actual_cuda_constructor_failure_releases_owned_memory(monkeypatch):
    import cupy as cp
    view=_views()[0]
    source=np.ones((view.num_slices,5,5),np.uint8)
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    pools=[]
    pool_factory=cp.cuda.MemoryPool
    def record_pool():
        pool=pool_factory()
        pools.append(pool)
        return pool
    def failure(*args,**kwargs):
        raise RuntimeError('synthetic compiler failure')
    monkeypatch.setattr(cp.cuda,'MemoryPool',record_pool)
    monkeypatch.setattr(cp,'RawKernel',failure)
    with pytest.raises(NativePullCudaUnavailable,match='synthetic compiler failure'):
        NativePullCudaSession(source,plan,device_index=0,max_device_bytes=1024**2,max_output_voxels=127)
    assert len(pools)==1 and pools[0].total_bytes()==0


@_gpu
def test_actual_cuda_kernel_failure_retains_session_and_never_publishes(monkeypatch):
    view=_views()[0]
    source=np.ones((view.num_slices,5,5),np.uint8)
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    output=np.full(127,177,np.uint8)
    with NativePullCudaSession(source,plan,device_index=0,max_device_bytes=1024**2,
                              max_output_voxels=127) as session:
        def failure(*args,**kwargs):
            raise RuntimeError('synthetic launch failure')
        monkeypatch.setattr(session,'_kernel',failure)
        with pytest.raises(NativePullCudaUnsafeFailure) as caught:
            session.pull_flat_into(output,first_flat=0)
        assert caught.value.session is session
        assert session._pool.total_bytes()>0
        np.testing.assert_array_equal(output,np.full(127,177,np.uint8))
    assert session._pool.total_bytes()==0


@_gpu
def test_actual_cuda_close_failure_retains_owned_residency(monkeypatch):
    view=_views()[0]
    source=np.ones((view.num_slices,5,5),np.uint8)
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    session=NativePullCudaSession(source,plan,device_index=0,max_device_bytes=1024**2,max_output_voxels=127)
    real_stream=session.stream
    class FailedFence:
        def synchronize(self):
            raise RuntimeError('synthetic stream fence failure')
    monkeypatch.setattr(session,'stream',FailedFence())
    try:
        with pytest.raises(NativePullCudaUnsafeFailure) as caught:
            session.close()
        assert caught.value.session is session
        assert not session._closed
        assert session._source is not None and session._pool.total_bytes()>0
    finally:
        session.stream=real_stream
        session.close()
    assert session._pool.total_bytes()==0


def _encoded_fixture(source, packed):
    from XTA.interpolation import CTILE_INDEX_DTYPE
    index=np.zeros(source.shape[0],CTILE_INDEX_DTYPE)
    payload=[]
    offset=0
    for frame,plane in enumerate(source):
        y,x=np.nonzero(plane)
        if not y.size:
            continue
        y0,y1,x0,x1=int(y.min()),int(y.max())+1,int(x.min()),int(x.max())+1
        crop=np.ascontiguousarray(plane[y0:y1,x0:x1]!=0,dtype=np.uint8)
        encoded=np.packbits(crop,axis=1,bitorder='little') if packed else crop
        data=encoded.reshape(-1)
        index[frame]=(1,np.zeros(7,np.uint8),offset,data.size,y0,x0,y1,x1,crop.size)
        payload.append(data)
        offset+=data.size
    return (np.concatenate(payload) if payload else np.empty(0,np.uint8)),index


@pytest.mark.parametrize('invalid',['kind','offset','short','bbox','decoded-size'])
def test_encoded_index_invalid_input_is_rejected_before_device_upload(invalid):
    view=_views()[0]
    source=np.ones((view.num_slices,5,5),np.uint8)
    encoded,index=_encoded_fixture(source,True)
    if invalid=='kind': index[0]['kind']=7
    elif invalid=='offset': index[0]['offset']=np.iinfo(np.uint64).max
    elif invalid=='short': encoded=encoded[:-1]
    elif invalid=='bbox': index[0]['y1']=source.shape[1]+1
    else: index[0]['payload_nbytes']+=1
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    with pytest.raises(ValueError):
        NativePullCudaSession.from_encoded(encoded,index,True,plan,device_index=0,
            max_device_bytes=1024**2,max_output_voxels=127)


def test_encoded_numeric_max_declines_without_device_work():
    view=_views()[0]
    source=np.ones((view.num_slices,5,5),np.uint8)
    encoded,index=_encoded_fixture(source,True)
    plan=prepare_native_pull_plan(view,source.shape,(14,19,23),max_plan_bytes=1024**2)
    with pytest.raises(NativePullCudaUnavailable,match='binary OR'):
        NativePullCudaSession.from_encoded(encoded,index,True,plan,device_index=0,
            max_device_bytes=1024**2,max_output_voxels=127,scalar_max=True)


@_gpu
@pytest.mark.parametrize('view',_views(),ids=lambda v:v.name)
@pytest.mark.parametrize('shape,reduced,cache', [((14,19,23),False,True),
                                               ((4,7,11),True,False),
                                               ((11,8,19),True,True)])
@pytest.mark.parametrize('packed',[False,True],ids=['raw-crops','row-packed-crops'])
def test_actual_cuda_encoded_crops_match_dense_numpy_and_cpu(view,shape,reduced,cache,packed):
    source_shape=(view.num_slices,5,5) if reduced else (view.num_slices,view.src_h,view.src_w)
    source=np.random.default_rng(160381).integers(0,2,source_shape,dtype=np.uint8)
    source[::3]=0
    source[:,:,0]=0  # Crop begins at a nonzero column; bits use crop-local columns.
    source[:,0,:]=0
    encoded,index=_encoded_fixture(source,packed)
    encoded.flags.writeable=False
    index.flags.writeable=False
    plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=4*1024**2,cache_plane=cache)
    first,count=13,math.prod(shape)-29
    bbox=(1,1,2,shape[0],shape[1]-1,shape[2]-1)
    expected=_reference(source,view,shape,first,count,False,bbox)
    actual=np.empty(count,np.uint8)
    with NativePullCudaSession.from_encoded(encoded,index,packed,plan,device_index=0,
            max_device_bytes=4*1024**2,max_output_voxels=127,max_upload_bytes=24) as session:
        assert session._source_owner is encoded and session._index_owner is index
        assert session._source.size==encoded.size
        stats=[session.pull_flat_into(actual[lo:lo+127],first_flat=first+lo,
            destination_bbox_tyx=bbox) for lo in range(0,count,127)]
        assert session._pool.total_bytes()==session.device_bytes
        assert all(s['backend'].startswith('cuda_native_encoded_') for s in stats)
    np.testing.assert_array_equal(actual,expected)
    cpu=np.empty(count,np.uint8)
    dense_stats=pull_native_flat_into(source,plan,cpu,first_flat=first,destination_bbox_tyx=bbox)
    assert sum(s['contribution_addresses'] for s in stats)==dense_stats['contribution_addresses']
    assert sum(s['indexed_input_byte_reads'] for s in stats)<=dense_stats['contribution_addresses']
    np.testing.assert_array_equal(cpu,expected)
    if geometry.is_azimuthal_view(view):
        from XTA.projection_coverage_cpu import pull_native_encoded_flat_into
        encoded_stats=pull_native_encoded_flat_into(encoded,index,packed,plan,cpu,
            first_flat=first,destination_bbox_tyx=bbox)
        assert sum(s['indexed_input_byte_reads'] for s in stats)==encoded_stats['indexed_input_byte_reads']
        np.testing.assert_array_equal(cpu,expected)
    assert session._pool.total_bytes()==0


@_gpu
@pytest.mark.parametrize('packed',[False,True])
def test_actual_cuda_borrows_real_rawbbox_store_without_native_decode(tmp_path,monkeypatch,packed):
    from XTA.interpolation import (IncrementalRawBBoxMaskStoreWriter,RawBBoxMaskStore,
                                    INTERNAL_PACKED_CVOL_FORMAT,CVOL_FORMAT)
    from XTA.sparse_projection import _SparseMaskSampler
    view=_views()[0]
    source=np.random.default_rng(772).integers(0,2,(view.num_slices,5,5),dtype=np.uint8)
    source[::2]=0
    source[:,:,0]=0
    root=tmp_path/'source.cvol'
    writer=IncrementalRawBBoxMaskStoreWriter(shape=source.shape,store_dir=root,
        format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT,desc='encoded CUDA borrow')
    for z in range(source.shape[0]):
        writer.consume(z,source[z:z+1])
    writer.finalize()
    store=RawBBoxMaskStore.open(root,mmap_payload=True)
    sampler=_SparseMaskSampler(store)
    def forbid_decode(*args,**kwargs):
        raise AssertionError('Encoded CUDA expanded a source crop')
    monkeypatch.setattr(store,'decode_slice',forbid_decode,raising=False)
    shape=(11,8,19)
    plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=1024**2)
    expected=_reference(source,view,shape,0,math.prod(shape),False,None)
    actual=np.empty_like(expected)
    try:
        with NativePullCudaSession.from_encoded(sampler.encoded,store.index,store._packbits_payload,
                plan,device_index=0,max_device_bytes=4*1024**2,max_output_voxels=actual.size) as session:
            assert session._source_owner is sampler.encoded
            session.pull_flat_into(actual,first_flat=0)
        np.testing.assert_array_equal(actual,expected)
    finally:
        sampler.close()
        store.close()


@_gpu
def test_actual_cuda_empty_encoded_payload_never_reads_unallocated_source():
    view=_views()[0]
    source=np.zeros((view.num_slices,5,5),np.uint8)
    encoded,index=_encoded_fixture(source,True)
    shape=(11,8,19)
    plan=prepare_native_pull_plan(view,source.shape,shape,max_plan_bytes=1024**2)
    output=np.full(math.prod(shape),199,np.uint8)
    with NativePullCudaSession.from_encoded(encoded,index,True,plan,device_index=0,
            max_device_bytes=1024**2,max_output_voxels=output.size) as session:
        stats=session.pull_flat_into(output,first_flat=0)
        assert stats['indexed_input_byte_reads']==0
        assert stats['contribution_addresses']>0
    assert not output.any()
