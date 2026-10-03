"""Native coverage oracles independent of the QSC forward equations.

All-positive charts must cover their analytic spherical domain. Single-face
objects and inactive radial cells preserve negative space; parity between two
projectors is not used as the coverage oracle.
"""
from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
import pytest

from XTA.spherical_geometry import build_spherical_view_infos,cube_rotation
from XTA.spherical_projection import backproject_spherical_volume_to_volume
from XTA.spherical_projection_cpu import pull_spherical_chunk_numba
from XTA import geometry
from XTA.config import TiltedViewGroup


def working_points(work,output):
    """Cartesian destination centers from physical extent, no QSC equations."""
    indices=np.indices(output,dtype=np.float64)
    points=[]
    for axis in range(3):
        points.append(((indices[axis]+.5)-output[axis]/2.)*work[axis]/output[axis])
    return np.stack(points[::-1],axis=-1)


def exact_squared_radius(work,output):
    """Integer Cartesian oracle for the small rational-scale fixtures."""
    reductions=[math.gcd(work[axis],2*output[axis])for axis in range(3)]
    denominators=[2*output[axis]//reductions[axis]for axis in range(3)]
    denominator=math.lcm(*denominators)
    indices=np.indices(output,dtype=np.int64)
    numerators=[(2*indices[axis]+1-output[axis])*(work[axis]//reductions[axis])*(denominator//denominators[axis])for axis in range(3)]
    return sum(numerator*numerator for numerator in numerators),denominator


def annulus(work,output,minimum,maximum):
    points=working_points(work,output)
    radius=np.linalg.norm(points,axis=-1)
    squared,denominator=exact_squared_radius(work,output)
    lower,upper=Fraction(minimum),Fraction(maximum)
    valid=(squared*lower.denominator**2>=lower.numerator**2*denominator**2)&(squared*upper.denominator**2<=upper.numerator**2*denominator**2)
    return valid,points,radius


def closed_face(points,rotation,face):
    local=points@np.asarray(rotation).reshape(3,3)
    axis,sign=((0,1),(1,1),(0,-1),(1,-1),(2,1),(2,-1))[face]
    normal=sign*local[...,axis]
    others=[index for index in range(3)if index!=axis]
    tolerance=8*np.finfo(np.float64).eps*np.max(np.abs(local),axis=-1)
    return (normal>0)&(normal+tolerance>=np.max(np.abs(local[...,others]),axis=-1))


def views(work,size,rotation,policy='dense'):
    built=build_spherical_view_infos(*work,targets=('transverse',),min_radius=.5,
        patch_size=size,tilted_views=(),sampling_policy=policy)
    return [replace(view,spherical_rotation_xyz=rotation)for view in built]


def project(data,view,output,boxes=None):
    blocks=[]
    result=backproject_spherical_volume_to_volume(data,view,Path('unused-native-spherical.dat'),
        'independent native coverage',out_shape_tyx=output,sink_only=True,
        known_slice_bboxes=boxes,projection_block_callback=lambda first,block:blocks.append((first,block.copy())))
    assert result.shape==output
    assert [first+i for first,block in blocks for i in range(len(block))]==list(range(output[0]))
    return np.concatenate([block for _,block in blocks])


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT','0')


ROTATIONS=[cube_rotation(),cube_rotation('vertical',31),cube_rotation('horizontal',-23),cube_rotation('vertical',90)]


@pytest.mark.parametrize('policy',['dense','coverage'])
@pytest.mark.parametrize('rotation',ROTATIONS)
@pytest.mark.parametrize('work,output,size',[((7,7,7),(21,21,21),5),((7,9,11),(21,27,33),8)])
def test_reduced_all_foreground_fills_only_native_annulus(work,output,size,rotation,policy):
    trajectories=views(work,size,rotation,policy)
    union=np.zeros(output,np.uint8)
    for view in trajectories:
        data=np.ones((view.num_slices,2,3),np.uint8)
        union|=project(data,view,output)
    expected,_,_=annulus(work,output,.5,(min(work)-1)/2.)
    np.testing.assert_array_equal(union!=0,expected)


@pytest.mark.parametrize('rotation',ROTATIONS)
@pytest.mark.parametrize('face',range(6))
def test_localized_one_face_object_keeps_opposite_face_gap(rotation,face):
    work,output=(7,9,11),(19,23,31)
    trajectories=[view for view in views(work,5,rotation)if view.spherical_face==face]
    union=np.zeros(output,np.uint8)
    for view in trajectories:
        union|=project(np.ones((view.num_slices,1,1),np.uint8),view,output)
    valid,points,_=annulus(work,output,.5,3.)
    expected=valid&closed_face(points,rotation,face)
    np.testing.assert_array_equal(union!=0,expected)
    assert expected.any()and(valid&~expected).any()


def test_enlarged_native_face_seams_corners_and_poles_are_closed():
    work,output=(7,7,7),(21,21,21)
    for view in views(work,15,cube_rotation()):
        result=project(np.ones((view.num_slices,2,3),np.uint8),view,output)
        assert result[10,13,13]==int(view.spherical_face in (0,1))
        assert result[13,13,13]==int(view.spherical_face in (0,1,4))
        assert result[16,10,10]==int(view.spherical_face==4)
        assert result[4,10,10]==int(view.spherical_face==5)
        assert result[10,10,10]==0


def test_localized_radial_cell_and_empty_gap_survive_roi_pruning():
    work,output=(9,11,12),(27,33,36)
    trajectories=build_spherical_view_infos(*work,targets=('transverse',),min_radius=1.,patch_size=7,tilted_views=())
    radii=np.asarray(trajectories[0].spherical_radii)
    assert np.array_equal(radii,[1.,2.,3.,4.])
    union=np.zeros(output,np.uint8)
    for view in trajectories:
        data=np.zeros((view.num_slices,2,3),np.uint8);data[1]=1
        boxes=np.zeros((view.num_slices,4),np.int64);boxes[1]=(0,2,0,3)
        actual=project(data,view,output,boxes)
        np.testing.assert_array_equal(actual,project(data,view,output))
        union|=actual
    valid,_,radius=annulus(work,output,1.,4.)
    squared,denominator=exact_squared_radius(work,output)
    # Active radius2 owns (1.5,2.5], using rational squared bounds rather
    # than a floating argmin that could share the original boundary error.
    expected=valid&(4*squared>9*denominator**2)&(4*squared<=25*denominator**2)
    np.testing.assert_array_equal(union!=0,expected)
    # Exact midpoint ties belong to the inner shell, even at the ROI limits.
    assert radius[13,16,22]==1.5 and union[13,16,22]==0
    assert radius[13,16,25]==2.5 and union[13,16,25]==1


def test_exact_inner_radius_is_closed_without_voxel_sized_expansion():
    work,output=(9,11,12),(27,33,36)
    union=np.zeros(output,np.uint8)
    for view in build_spherical_view_infos(*work,targets=('transverse',),min_radius=1.5,patch_size=7,tilted_views=()):
        union|=project(np.ones((view.num_slices,2,3),np.uint8),view,output)
    expected,_,_=annulus(work,output,1.5,4.)
    np.testing.assert_array_equal(union!=0,expected)
    assert union[13,16,22]==1
    assert union[13,16,21]==0


def test_points_outside_limit_beyond_roundoff_remain_excluded():
    work,output=(7,7,7),(21,21,21)
    upper=3.-1e-12
    union=np.zeros(output,np.uint8)
    for view in views(work,15,cube_rotation()):
        changed=replace(view,spherical_max_radius=upper,spherical_radii=(*view.spherical_radii[:-1],upper))
        union|=project(np.ones((view.num_slices,2,3),np.uint8),changed,output)
    assert union[10,10,19]==0  # Exactly radius3, now strictly outside.
    work,output=(9,9,9),(27,27,27)
    union=np.zeros(output,np.uint8)
    for view in build_spherical_view_infos(*work,targets=('transverse',),min_radius=3.+1e-12,patch_size=15,tilted_views=()):
        union|=project(np.ones((view.num_slices,2,3),np.uint8),view,output)
    assert union[13,13,22]==0  # Exactly radius3, strictly inside excluded hole.


def test_actual_cartesian_aliases_and_all_signed_tilt_groups_fill_native_domain():
    work,output=(7,9,11),(21,27,33)
    groups=(TiltedViewGroup(('transverse','sagittal','coronal'),(31.,),('vertical','horizontal')),)
    built=geometry.get_view_infos(*work,cartesian_views=(),
        spherical_views=('transverse','sagittal','coronal','tilted_transverse','tilted_sagittal','tilted_coronal'),
        spherical_min_radius=.5,spherical_patch_size=5,tilt_groups=groups)
    trajectories=[view for view in built if view.family=='spherical']
    keys={view.spherical_group for view in trajectories}
    assert len(keys)==5  # upright and both signs of both tilt directions.
    expected,_,_=annulus(work,output,.5,3.)
    for key in keys:
        union=np.zeros(output,np.uint8)
        for view in trajectories:
            if view.spherical_group==key:
                assert len(view.spherical_request_tokens)==3
                union|=project(np.ones((view.num_slices,2,3),np.uint8),view,output)
        np.testing.assert_array_equal(union!=0,expected)


def test_positive_scores_share_exact_rational_native_binary_support():
    work,output=(7,9,11),(21,27,33)
    score=np.zeros(output,np.uint8)
    for view in views(work,5,cube_rotation('horizontal',-23)):
        source=np.full((view.num_slices,2,3),217,np.uint8)
        radii=np.asarray(view.spherical_radii)
        rotation=np.asarray(view.spherical_rotation_xyz).reshape(3,3)
        for z in range(output[0]):
            plane=pull_spherical_chunk_numba(source,view,radii,rotation,output,z,0,output[1]*output[2],scalar_max=True).reshape(output[1:])
            score[z]=np.maximum(score[z],plane)
    expected,_,_=annulus(work,output,.5,3.)
    np.testing.assert_array_equal(score,np.where(expected,217,0).astype(np.uint8))


def test_compiled_pull_chunk_splits_keep_localized_native_support():
    work,output=(7,9,11),(3,360,365)
    view=views(work,15,cube_rotation())[0]
    data=np.ones((view.num_slices,1,1),np.uint8)
    rotation=np.asarray(view.spherical_rotation_xyz).reshape(3,3)
    radii=np.asarray(view.spherical_radii)
    splits=(0,17,127001,128001,output[1]*output[2])
    actual=np.concatenate([pull_spherical_chunk_numba(data,view,radii,rotation,output,1,a,b)
        for a,b in zip(splits,splits[1:])]).reshape(output[1:])
    valid,points,_=annulus(work,output,.5,3.)
    expected=valid&closed_face(points,cube_rotation(),0)
    np.testing.assert_array_equal(actual!=0,expected[1])
