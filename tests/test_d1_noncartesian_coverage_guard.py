"""Foreign D1 oblique tasks fail before device allocation or state mutation."""
from dataclasses import replace
import inspect
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import cuda_d1, geometry, config


def oblique_views():
    return tuple(geometry.get_view_infos(17,19,23,cartesian_views=(),
        azimuthal_views=('transverse','tilted_transverse'),azimuthal_azimuth_angles=(30.,30.),
        tilt_groups=(config.TiltedViewGroup(('transverse',),(30.,),('vertical','horizontal')),),
        azimuthal_native_raster=13))


@pytest.mark.parametrize('view',oblique_views(),ids=lambda view:view.name)
@pytest.mark.parametrize('angle',(0.,120.))
@pytest.mark.parametrize('delayed',('0','1'))
def test_legacy_oblique_state_rejected_before_any_cuda_probe(view,angle,delayed,monkeypatch):
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION',delayed)
    view=replace(view,tta_angle_deg=angle)
    before=dict(cuda_d1._D1_WORKER_VIEW_STATES)
    with mock.patch.object(cuda_d1,'_d1_backproject_kernels',side_effect=AssertionError('CUDA probed')) as probe:
        with pytest.raises(ValueError,match='canonical native pull projectors'):
            cuda_d1._d1_get_or_create_state({'view':view,'result_mode':'d1_owner'},None)
        with pytest.raises(ValueError,match='canonical native pull projectors'):
            cuda_d1._d1_require_cartesian_native_coverage(view)
    probe.assert_not_called()
    assert cuda_d1._D1_WORKER_VIEW_STATES==before


@pytest.mark.parametrize('view',oblique_views(),ids=lambda view:view.name)
def test_consumer_cannot_bypass_family_guard_using_an_existing_state(view):
    accumulator=SimpleNamespace(union_dev=object(),host_written=False)
    with mock.patch.object(cuda_d1,'_d1_get_or_create_state',side_effect=AssertionError('state allocation attempted')) as state:
        with pytest.raises(ValueError,match='canonical native pull projectors'):
            cuda_d1._d1_consume_device_union({'view':view},accumulator)
    assert accumulator.union_dev is not None
    state.assert_not_called()


@pytest.mark.parametrize('base',('transverse','sagittal','coronal'))
def test_cartesian_bases_pass_the_native_coverage_guard(base):
    view=geometry.get_view_infos(17,19,23,cartesian_views=(base,),azimuthal_views=())[0]
    assert cuda_d1._d1_require_cartesian_native_coverage(view) is None


def test_kernel_module_carries_only_the_cartesian_coverage_kernel():
    source=inspect.getsource(cuda_d1._d1_backproject_kernels)
    assert 'd1_backproject_bboxes_to_bits' not in source
    assert source.count('extern "C" __global__')==1
    assert 'extern "C" __global__ void d1_orthogonal_coverage_to_bits(' in source


def test_unknown_cartesian_base_rejected_before_device_access():
    view=geometry.ViewInfo(name='foreign',physical_view_name='foreign',family='orthogonal',
        num_slices=17,src_h=19,src_w=23,pad_mode='clamp')
    with mock.patch.object(cuda_d1,'_d1_backproject_kernels',side_effect=AssertionError('CUDA probed')):
        with pytest.raises(ValueError,match='does not support base view'):
            cuda_d1._d1_get_or_create_state({'view':view},None)
