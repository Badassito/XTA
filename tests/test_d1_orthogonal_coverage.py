"""D1 categorical coverage dispatch, chunk lifetime, and optional CUDA parity."""
from concurrent.futures import Future
import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_d1, pipeline
from XTA.geometry import ViewInfo
from XTA.interpolation import RawBBoxMaskStore
from XTA.d1_orthogonal_coverage import (build_orthogonal_coverage_plan,coverage_launch_groups,
    execute_orthogonal_coverage_reference,ORTHOGONAL_COVERAGE_CONTRACT)
from tests.test_spherical_runtime import _assigned


def view_for(base, frames, height=25, width=27):
    return ViewInfo(name=base,physical_view_name=base,family='orthogonal',num_slices=frames,
        src_h=height,src_w=width,full_t=17,full_h=25,full_w=27,pad_mode='clamp')


class HostWords:
    def __init__(self,words):self.array=words
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array


def cpu_launch(_grid,_block,args,**_kwargs):
    mask,ph,pw,specs,count,ot,oh,ow,stack,row,column,*remaining=args
    pairs=list(zip(remaining[:6:2],remaining[1:6:2]));words=remaining[6]
    for local,frame,z0,z1,y0,y1,x0,x1 in specs[:count]:
        for z in range(z0,z1):
            for y in range(y0,y1):
                for x in range(x0,x1):
                    bounds=[(int(first[index]),int(stop[index])) for index,(first,stop) in zip((z,y,x),pairs)]
                    if not bounds[stack][0]<=frame<bounds[stack][1]:continue
                    if np.any(mask[local,bounds[row][0]:bounds[row][1],bounds[column][0]:bounds[column][1]]):
                        linear=(z*int(oh)+y)*int(ow)+x
                        words[linear//32]|=np.uint32(1<<int(linear%32))


@pytest.mark.parametrize('base,frames',[('transverse',17),('sagittal',25),('coronal',27)])
def test_out_of_order_chunks_publish_only_after_all_required_input_and_match_cvol(tmp_path,base,frames):
    view=view_for(base,frames)
    source=np.zeros((frames,17,17),np.uint8)
    source[:,6:11,7:12]=1;source[0,0,0]=1;source[-1,-1,-1]=1
    shape=(13,25,27)
    words=HostWords(np.zeros((int(np.prod(shape))+31)//32,np.uint32))
    state=cuda_d1._D1WorkerViewState(('fixture',base),view,shape,words,np.zeros(frames,bool),
        np.zeros(1,np.float32),np.zeros(1,np.float32),tmp_path/'source.cvol',0.)
    stream=SimpleNamespace(synchronize=mock.Mock())
    cp=SimpleNamespace(asarray=lambda value:np.asarray(value.array if isinstance(value,HostWords) else value),
        cuda=SimpleNamespace(get_current_stream=lambda:stream))
    kernels=SimpleNamespace(cp=cp,d1_orthogonal_coverage_to_bits=cpu_launch,
        d1_backproject_bboxes_to_bits=mock.Mock(side_effect=AssertionError('old point scatter selected')))
    published=[]
    def publish(*,words,state):
        assert state.coverage.all()
        result=cuda_d1._d1_finalize_bitset_layer(words=words.copy(),output_shape=shape,
            store_dir=state.store_dir,model_name='fixture',view=view,projection_kind=state.projection_kind)
        published.append(result)
        future=Future();future.set_result(result);return future
    cuts=(0,frames//3,2*frames//3,frames)
    chunks=[(cuts[1],cuts[2]),(cuts[0],cuts[1]),(cuts[2],cuts[3])]
    with mock.patch.object(cuda_d1,'_d1_get_or_create_state',return_value=state), \
            mock.patch.object(cuda_d1,'_d1_backproject_kernels',return_value=kernels), \
            mock.patch.object(cuda_d1,'_d1_submit_publication',side_effect=publish), \
            mock.patch.dict(cuda_d1._D1_WORKER_VIEW_STATES,{state.key:state}):
        for ordinal,(first,stop) in enumerate(chunks):
            chunk=source[first:stop].copy()
            boxes=np.zeros((len(chunk),4),np.int64)
            for z,plane in enumerate(chunk):
                ys,xs=np.nonzero(plane)
                if len(ys):boxes[z]=(ys.min(),ys.max()+1,xs.min(),xs.max()+1)
            accumulator=SimpleNamespace(union_dev=chunk,host_written=False,
                _d1_confidence_slice_metadata=dict(slice_any=np.any(chunk,axis=(1,2)),slice_bboxes=boxes),
                conf_dev=None,prediction_counts_dev=None,slice_bboxes_dev=None,slice_bboxes_written=None)
            result=cuda_d1._d1_consume_device_union(dict(view=view,slice_start=first,slice_count=stop-first),accumulator)
            assert accumulator.union_dev is None
            assert bool(published)==(ordinal==len(chunks)-1)
    assert result['d1_view_complete'] and len(published)==1
    store=RawBBoxMaskStore.open(state.store_dir)
    try:
        actual=np.stack([store.decode_slice(z,dtype=np.uint8) for z in range(shape[0])])
        expected=execute_orthogonal_coverage_reference(source,view,shape)
        np.testing.assert_array_equal(actual,expected)
        assert store.meta['projection_payload_fusion']==ORTHOGONAL_COVERAGE_CONTRACT
    finally:store.close()


@pytest.mark.parametrize('cpu',[False,True])
def test_area_contraction_route_never_claims_d1_or_hybrid(cpu):
    tree=ast.parse(inspect.getsource(pipeline._main_impl))
    hybrid=next(node for node in ast.walk(tree) if _assigned(node,'hybrid_deferred'))
    routing=next(node for node in ast.walk(tree) if isinstance(node,ast.If)
        and isinstance(node.test,ast.Name) and node.test.id=='radial_owner'
        and any(_assigned(statement,'result_mode','d1_owner') for statement in node.body))
    program=compile(ast.fix_missing_locations(ast.Module(body=[hybrid,routing],type_ignores=[])),
        '<categorical area fallback routing>','exec')
    view=view_for('coronal',27)
    env=dict(view=view,kind='fullframe',v1613_d1_owner_active=True,legacy_d1_model_eligible=True,
        d1_categorical_coverage_eligible_by_view={view.name:False},worker_direct_union_active=True,
        cpu_eligible=cpu,gpu_eligible=True,azimuthal_parent_requires_seam_union=False,radial_owner=False,
        requires_native_pull=lambda _view:False,HYBRID_DEFERRED_RESULT_MODE=pipeline.HYBRID_DEFERRED_RESULT_MODE)
    exec(program,env)
    assert env['result_mode']=='direct_union' and not env['hybrid_deferred']


@pytest.mark.skipif(os.environ.get('YOLO_TTA_TEST_D1_ORTHOGONAL_CUDA')!='1',reason='explicit GPU_LOCK-owned CUDA qualification')
@pytest.mark.parametrize('base,frames,processing_hw,shape',[
    ('transverse',1,(2048,2048),(1,3064,3022)),
    ('transverse',17,(17,17),(13,25,27)),
    ('sagittal',25,(17,17),(13,25,27)),
    ('coronal',27,(17,17),(13,25,27)),
])
def test_cuda_packed_coverage_matches_native_reference(base,frames,processing_hw,shape):
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount()>0
    source=np.zeros((frames,*processing_hw),np.uint8)
    source[:,3:7,5:11]=1;source[0,0,0]=1;source[-1,-1,-1]=1
    view=view_for(base,frames,*processing_hw)
    plan=build_orthogonal_coverage_plan(view,source.shape,shape)
    kernels=cuda_d1._d1_backproject_kernels()
    bits=cp.zeros((int(np.prod(shape))+31)//32,cp.uint32)
    device_ranges=tuple(cp.asarray(value) for pair in plan.source_ranges_tyx for value in pair)
    chunks=((frames//2,frames),(0,frames//2)) if frames>1 else ((0,1),)
    for first,stop in chunks:
        mask=cp.asarray(source[first:stop])
        boxes=[]
        for local,plane in enumerate(source[first:stop]):
            ys,xs=np.nonzero(plane)
            if len(ys):boxes.append((local,ys.min(),xs.min(),ys.max()-ys.min()+1,xs.max()-xs.min()+1))
        for cells,specs in coverage_launch_groups(plan,np.asarray(boxes,np.int32),first):
            kernels.d1_orthogonal_coverage_to_bits(((cells+255)//256,len(specs)),(256,),
                (mask,np.int32(processing_hw[0]),np.int32(processing_hw[1]),cp.asarray(specs),
                 np.int32(len(specs)),*map(np.int32,shape),np.int32(plan.input_axes_tyx.index(0)),
                 np.int32(plan.input_axes_tyx.index(1)),np.int32(plan.input_axes_tyx.index(2)),
                 *device_ranges,bits))
    cp.cuda.get_current_stream().synchronize()
    actual=np.unpackbits(cp.asnumpy(bits).view(np.uint8),bitorder='little',count=int(np.prod(shape))).reshape(shape)
    expected=execute_orthogonal_coverage_reference(source,view,shape)
    np.testing.assert_array_equal(actual,expected)
