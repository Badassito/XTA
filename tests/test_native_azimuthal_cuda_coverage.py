"""Opt-in actual upright transverse Azimuthal CUDA parity; no CPU fallback."""
import os
from pathlib import Path
import numpy as np
import pytest
from XTA import backprojection as bp, geometry as g


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_AZIMUTHAL_CUDA') != '1', reason='Explicit GPU coordinator authorization required')
@pytest.mark.parametrize('backend', ['resident', 'streaming'])
@pytest.mark.parametrize('spacing', [7., 40.])
@pytest.mark.parametrize('pattern', ['positive', 'gap'])
def test_real_upright_azimuthal_cuda_native_cells(tmp_path, backend, spacing, pattern):
    import torch
    assert torch.cuda.is_available(), 'Requested real CUDA qualification cannot fall back'
    view = g._build_azimuthal_view_info(9,11,13,base_view='transverse',azimuth_angle=spacing,
                                      azimuthal_native_raster=0,request_token='test')
    source = np.ones((view.num_slices,view.src_h,view.src_w),np.uint8)
    if pattern == 'gap':
        source[:,2:4,:] = 0
        source[:,:,4:7] = 0
    shape=(6,16,20)
    plan,_=bp.build_azimuthal_backprojection_plan(view)
    # Independent pointwise native angular ownership plus exact native-row OR.
    expected=np.zeros(shape,np.uint8)
    radius=view.roi_radius
    angles=np.asarray([float(p.angle_deg)%180. for p in plan],np.float32)
    for y in range(shape[1]):
        wy=np.float32(y+.5)*np.float32(view.full_h/shape[1])-np.float32(.5)
        for x in range(shape[2]):
            wx=np.float32(x+.5)*np.float32(view.full_w/shape[2])-np.float32(.5)
            dx,dy=wx-view.center_x,wy-view.center_y
            if np.sqrt(dx*dx+dy*dy)>radius+.5: continue
            theta=np.float32(np.degrees(np.arctan2(dy,dx)))%np.float32(180.)
            distance=np.abs((float(theta)-angles.astype(np.float64)+90.)%180.-90.)
            at=int(distance.argmin())
            angle=angles[at]
            signed=np.float32(dx)*np.cos(np.deg2rad(angle))+np.float32(dy)*np.sin(np.deg2rad(angle))
            if plan[at].reverse_u:signed=-signed
            u=int(np.clip(np.rint((signed+radius)/(2*radius)*(view.src_w-1)),0,view.src_w-1))
            for z in range(shape[0]):
                lo=z*view.src_h//shape[0]
                hi=((z+1)*view.src_h+shape[0]-1)//shape[0]
                expected[z,y,x]=bool(source[plan[at].source_index,lo:hi,u].any())
    dense=bp.build_dense_azimuthal_backprojection_map(view,plan,out_shape_hw=shape[1:])
    rows=lambda z:(z*view.src_h//shape[0],((z+1)*view.src_h+shape[0]-1)//shape[0])
    output=np.zeros(shape,np.uint8)
    with torch.cuda.device(0):
        device=torch.device('cuda:0')
        if backend=='resident':
            kernels=bp._azimuthal_resident_backproject_kernel()
            assert kernels is not None,'Actual resident CUDA kernels are required'
            actual=bp._azimuthal_backproject_gpu_resident_on_device(
                source,output,dense.valid_mask,dense.source_idx_map,dense.u_idx_map,
                rows,*shape,'actual native coverage',kernels=kernels,torch=torch,dev=device)
        else:
            valid=np.flatnonzero(dense.valid_mask.reshape(-1)).astype(np.int64)
            flat=dense.source_idx_map.reshape(-1)[valid].astype(np.int64)*source.shape[2]+dense.u_idx_map.reshape(-1)[valid]
            actual=bp._azimuthal_backproject_gpu_streaming_on_device(
                source,output,valid,flat,rows,*shape,'actual native coverage',torch=torch,dev=device)
        assert actual is True,'No CPU fallback is allowed in this qualification'
        torch.cuda.synchronize()
    np.testing.assert_array_equal(output,expected)


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_AZIMUTHAL_CUDA') != '1', reason='Explicit GPU coordinator authorization required')
@pytest.mark.parametrize('backend', ['resident', 'streaming'])
def test_real_upright_azimuthal_cuda_degenerate_disk(tmp_path, backend):
    import torch
    assert torch.cuda.is_available(), 'Requested real CUDA qualification cannot fall back'
    view=g._build_azimuthal_view_info(1,1,1,base_view='transverse',azimuth_angle=45.,
                                    azimuthal_native_raster=0,request_token='test')
    source=np.ones((view.num_slices,1,1),np.uint8)
    y,x=np.indices((5,5),dtype=np.float64)
    expected=((((x+.5)/5-.5)**2+((y+.5)/5-.5)**2)<=.25)[None].astype(np.uint8)
    assert expected.sum()==21
    plan,_=bp.build_azimuthal_backprojection_plan(view)
    dense=bp.build_dense_azimuthal_backprojection_map(view,plan,out_shape_hw=(5,5))
    output=np.zeros((1,5,5),np.uint8)
    rows=lambda z:(0,1)
    with torch.cuda.device(0):
        device=torch.device('cuda:0')
        if backend=='resident':
            kernels=bp._azimuthal_resident_backproject_kernel()
            assert kernels is not None,'Actual resident CUDA kernels are required'
            actual=bp._azimuthal_backproject_gpu_resident_on_device(
                source,output,dense.valid_mask,dense.source_idx_map,dense.u_idx_map,
                rows,1,5,5,'actual degenerate disk',kernels=kernels,torch=torch,dev=device)
        else:
            valid=np.flatnonzero(dense.valid_mask.reshape(-1)).astype(np.int64)
            flat=dense.source_idx_map.reshape(-1)[valid].astype(np.int64)
            actual=bp._azimuthal_backproject_gpu_streaming_on_device(
                source,output,valid,flat,rows,1,5,5,'actual degenerate disk',torch=torch,dev=device)
        assert actual is True,'No CPU fallback is allowed in this qualification'
        torch.cuda.synchronize()
    np.testing.assert_array_equal(output,expected)
