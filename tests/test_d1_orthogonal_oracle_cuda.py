"""Independent symbolic native FOV/gap oracles for the Cartesian CUDA kernel."""
import math
import os

import numpy as np
import pytest

from XTA import cuda_d1, geometry
from XTA.d1_orthogonal_coverage import build_orthogonal_coverage_plan,coverage_launch_groups


def symbolic_expected(source,base,shape):
    """Direct axis/interval definition; no production table or CPU projector."""
    axes={'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(1,2,0)}[base]
    canonical=tuple(source.shape[axis] for axis in axes)
    expected=np.zeros(shape,np.uint8)
    for z,y,x in np.ndindex(shape):
        if canonical[0]>=shape[0]:
            frames=range(z*canonical[0]//shape[0],
                ((z+1)*canonical[0]+shape[0]-1)//shape[0])
        else:
            frames=(round(z*(canonical[0]-1)/(shape[0]-1)) if shape[0]>1 else 0,)
        row=y*canonical[1]//shape[1]
        column=x*canonical[2]//shape[2]
        for frame in frames:
            original=[None,None,None]
            for axis,value in zip(axes,(frame,row,column)):original[axis]=value
            expected[z,y,x]|=source[tuple(original)]
    return expected


@pytest.mark.skipif(os.environ.get('YOLO_TTA_TEST_D1_ORTHOGONAL_CUDA')!='1',
    reason='explicit GPU_LOCK-owned independent CUDA qualification')
@pytest.mark.parametrize('base',('transverse','sagittal','coronal'))
@pytest.mark.parametrize('cube',(False,True),ids=('noncubic-native','t-rescaled'))
@pytest.mark.parametrize('pattern',('full-fov','sharp-gaps'))
def test_cuda_native_fov_and_sharp_gaps_match_symbolic_oracle(base,cube,pattern):
    import cupy as cp
    target=(13,25,27)
    work=(17 if cube else target[0],*target[1:])
    view=geometry.get_view_infos(*work,cartesian_views=(base,),azimuthal_views=())[0]
    source=np.ones((view.num_slices,17,17),np.uint8)
    if pattern=='sharp-gaps':
        source[:,5:8]=0
        source[:,:,9:11]=0
        start=view.num_slices//3
        source[start:start+2]=0
    expected=symbolic_expected(source,base,target)
    if pattern=='full-fov':assert expected.all()
    else:assert expected.any() and not expected.all()
    plan=build_orthogonal_coverage_plan(view,source.shape,target)
    ranges=tuple(cp.asarray(value) for pair in plan.source_ranges_tyx for value in pair)
    bits=cp.zeros((math.prod(target)+31)//32,cp.uint32)
    kernel=cuda_d1._d1_backproject_kernels().d1_orthogonal_coverage_to_bits
    cuts=(0,view.num_slices//3,2*view.num_slices//3,view.num_slices)
    # Neither stack order nor overlapping target temporal bins can discard data.
    for first,stop in ((cuts[1],cuts[2]),(cuts[2],cuts[3]),(cuts[0],cuts[1])):
        chunk=source[first:stop]
        mask=cp.asarray(chunk)
        boxes=[]
        for local,plane in enumerate(chunk):
            rows,cols=np.nonzero(plane)
            if len(rows):boxes.append((local,rows.min(),cols.min(),rows.max()-rows.min()+1,cols.max()-cols.min()+1))
        for count,specs in coverage_launch_groups(plan,np.asarray(boxes,np.int32),first):
            gpu_specs=cp.asarray(specs)
            kernel(((count+255)//256,len(specs)),(256,),
                (mask,np.int32(17),np.int32(17),gpu_specs,np.int32(len(specs)),
                 *map(np.int32,target),np.int32(plan.input_axes_tyx.index(0)),
                 np.int32(plan.input_axes_tyx.index(1)),np.int32(plan.input_axes_tyx.index(2)),*ranges,bits))
            cp.cuda.get_current_stream().synchronize()
    actual=np.unpackbits(cp.asnumpy(bits).view(np.uint8),bitorder='little',count=math.prod(target)).reshape(target)
    np.testing.assert_array_equal(actual,expected)
