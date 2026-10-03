"""Opt-in actual CUDA native coverage against rational geometry oracles.

Run only under the task GPU lock with XTA_TEST_SPHERICAL_CUDA=1. When enabled,
runtime/admission failures fail these tests instead of becoming CPU fallbacks.
"""
import os
from dataclasses import replace

import numpy as np
import pytest

from XTA.spherical_geometry import build_spherical_view_infos,cube_rotation
from XTA.spherical_projection_cuda import SphericalCudaProjector
from tests.test_spherical_native_coverage import annulus,closed_face,exact_squared_radius,views
from tests.test_spherical_cpu_compact import decode


pytestmark=pytest.mark.skipif(os.environ.get('XTA_TEST_SPHERICAL_CUDA')!='1',reason='requires coordinated actual CUDA qualification')


def gpu_forms(source,view,output,boxes):
    """Exercise dense, raw-ROI and packed-ROI publication on the real GPU."""
    arrays=[np.zeros(output,np.uint8)for _ in range(3)]
    with SphericalCudaProjector(source,view,output,boxes,block_bytes=output[1]*output[2]*3,reserve_bytes=0)as projector:
        for first in range(0,output[0],3):
            count=min(3,output[0]-first)
            arrays[0][first:first+count]=projector.project(first,count)
            for index,packed in ((1,False),(2,True)):
                block=projector.project_encoded(first,count,packed=packed)
                arrays[index][first:first+count]=decode(block,output)
    return arrays


@pytest.mark.parametrize('work,output',[((7,7,7),(21,21,21)),((7,9,11),(21,27,33))])
@pytest.mark.parametrize('rotation',[cube_rotation(),cube_rotation('vertical',31),cube_rotation('horizontal',-23)])
def test_cuda_exact_rational_native_annulus(work,output,rotation):
    unions=[np.zeros(output,np.uint8)for _ in range(3)]
    for view in views(work,8,rotation):
        source=np.ones((view.num_slices,2,3),np.uint8)
        boxes=np.tile(np.array([0,2,0,3],np.int64),(view.num_slices,1))
        for union,result in zip(unions,gpu_forms(source,view,output,boxes)):union|=result
    expected,_,_=annulus(work,output,.5,3.)
    for union in unions:np.testing.assert_array_equal(union!=0,expected)


@pytest.mark.parametrize('face',[0,4])
def test_cuda_localized_face_keeps_negative_space(face):
    work,output=(7,9,11),(19,23,31)
    rotation=cube_rotation('vertical',31)
    unions=[np.zeros(output,np.uint8)for _ in range(3)]
    for view in views(work,8,rotation):
        if view.spherical_face!=face:continue
        source=np.ones((view.num_slices,2,3),np.uint8)
        boxes=np.tile(np.array([0,2,0,3],np.int64),(view.num_slices,1))
        for union,result in zip(unions,gpu_forms(source,view,output,boxes)):union|=result
    valid,points,_=annulus(work,output,.5,3.)
    expected=valid&closed_face(points,rotation,face)
    assert expected.any()and(valid&~expected).any()
    for union in unions:np.testing.assert_array_equal(union!=0,expected)


def test_cuda_radial_gap_and_exact_midpoints_survive_packed_roi():
    work,output=(9,11,12),(27,33,36)
    unions=[np.zeros(output,np.uint8)for _ in range(3)]
    trajectories=build_spherical_view_infos(*work,targets=('transverse',),min_radius=1.,patch_size=8,tilted_views=())
    for view in trajectories:
        source=np.zeros((view.num_slices,2,3),np.uint8);source[1]=1
        boxes=np.zeros((view.num_slices,4),np.int64);boxes[1]=(0,2,0,3)
        for union,result in zip(unions,gpu_forms(source,view,output,boxes)):union|=result
    squared,denominator=exact_squared_radius(work,output)
    expected=(4*squared>9*denominator**2)&(4*squared<=25*denominator**2)
    for union in unions:
        np.testing.assert_array_equal(union!=0,expected)
        assert union[13,16,22]==0  #1.5 belongs to empty inner shell.
        assert union[13,16,25]==1  #2.5 belongs to active inner shell.


def test_cuda_exact_inner_boundary_is_closed():
    work,output=(9,11,12),(27,33,36)
    unions=[np.zeros(output,np.uint8)for _ in range(3)]
    for view in build_spherical_view_infos(*work,targets=('transverse',),min_radius=1.5,patch_size=8,tilted_views=()):
        source=np.ones((view.num_slices,2,3),np.uint8)
        boxes=np.tile(np.array([0,2,0,3],np.int64),(view.num_slices,1))
        for union,result in zip(unions,gpu_forms(source,view,output,boxes)):union|=result
    expected,_,_=annulus(work,output,1.5,4.)
    for union in unions:
        np.testing.assert_array_equal(union!=0,expected)
        assert union[13,16,22]==1
        assert union[13,16,21]==0


def test_cuda_beyond_roundoff_limits_remain_excluded():
    for work,output,minimum,upper in (((7,7,7),(21,21,21),.5,3.-1e-12),
                                     ((9,9,9),(27,27,27),3.+1e-12,4.)):
        unions=[np.zeros(output,np.uint8)for _ in range(3)]
        trajectories=build_spherical_view_infos(*work,targets=('transverse',),min_radius=minimum,patch_size=15,tilted_views=())
        for original in trajectories:
            view=replace(original,spherical_max_radius=upper,spherical_radii=(*original.spherical_radii[:-1],upper))
            source=np.ones((view.num_slices,2,3),np.uint8)
            boxes=np.tile(np.array([0,2,0,3],np.int64),(view.num_slices,1))
            for union,result in zip(unions,gpu_forms(source,view,output,boxes)):union|=result
        squared,denominator=exact_squared_radius(work,output)
        expected=((4*squared>=denominator**2)&(squared<9*denominator**2))if minimum==.5 else((squared>9*denominator**2)&(squared<=16*denominator**2))
        for union in unions:np.testing.assert_array_equal(union!=0,expected)
