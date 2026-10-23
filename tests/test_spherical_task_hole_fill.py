"""Spherical lease cleanup reuses the existing exact device-union fill."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from XTA.inference import device_union_task_hole_fill_eligible


@pytest.mark.parametrize('family,count,expected',[
    ('spherical',1,True),('spherical',2,True),('spherical',7,True),
    ('tilted',1,True),('tilted',2,False),('azimuthal',2,False),('radial',2,False)])
def test_spherical_split_leases_keep_existing_device_cleanup(family,count,expected):
    view=SimpleNamespace(family=family,augmentation_pass=0)
    assert device_union_task_hole_fill_eligible(view,enabled=True,kind='fullframe',
        result_mode='direct_union',lease_count=count) is expected


@pytest.mark.parametrize('change',[
    dict(enabled=False),dict(kind='tile'),dict(result_mode='file'),dict(result_mode='d1_owner'),
    dict(lease_count=0),dict(augmentation_pass=1)])
def test_device_cleanup_requires_original_order_and_coverage_proofs(change):
    args=dict(enabled=True,kind='fullframe',result_mode='direct_union',lease_count=2)
    view=SimpleNamespace(family='spherical',augmentation_pass=change.get('augmentation_pass',0))
    args.update({key:value for key,value in change.items() if key!='augmentation_pass'})
    assert not device_union_task_hole_fill_eligible(view,**args)


def _patterns():
    source=np.zeros((17,129,137),np.uint8)
    source[0]=1
    for z in range(1,source.shape[0]):
        source[z,8:121,11:125]=1
        source[z,20:39,25:57]=0
        source[z,55:88,67:103]=0
        if z%3==0:
            source[z,20:57,42]=0
            source[z,:24,42]=0  # Exterior channel remains background.
        if z%4==0:
            np.fill_diagonal(source[z],0)  # Diagonals do not connect 4-neighbor background.
    source[-1]=0
    return source


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA')!='1',reason='GPU lock and explicit actual CUDA required')
@pytest.mark.parametrize('leases',[(17,),(7,5,5)],ids=['whole-stack','disjoint-split-leases'])
def test_actual_device_fill_matches_scipy_and_reports_all_lease_frames(monkeypatch,leases):
    import torch
    from XTA.inference import _DeviceUnionAccumulator
    monkeypatch.setenv('YOLO_TTA_GPU_HOLE_FILL_BLOCK','4')
    source=_patterns()
    expected=np.stack([ndimage.binary_fill_holes(plane) for plane in source]).astype(np.uint8)
    actual=np.empty_like(source)
    first=filled=0
    for length in leases:
        accumulator=_DeviceUnionAccumulator(torch,torch.device('cuda:0'),length,*source.shape[1:],False)
        accumulator.union_dev.copy_(torch.from_numpy(source[first:first+length]).to('cuda:0'))
        count=accumulator.fill_holes_2d()
        assert count==length,'Actual CUDA cleanup must not fall back to CPU'
        filled+=count
        actual[first:first+length]=accumulator.union_dev.cpu().numpy()
        first+=length
        del accumulator
    assert filled==source.shape[0]
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_PULL_CUDA')!='1',reason='GPU lock and explicit actual CUDA required')
def test_device_host_fallback_returns_zero_so_completed_view_is_repaired_on_cpu():
    import torch
    from XTA.inference import _DeviceUnionAccumulator
    accumulator=_DeviceUnionAccumulator(torch,torch.device('cuda:0'),3,17,19,False)
    accumulator.mark_host_write()
    assert accumulator.fill_holes_2d()==0
