"""Numeric shell mode, compact transfer and crop-reader ownership contracts."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import os
import sys
import json

import numpy as np
import pytest

from XTA import backprojection, geometry, cylindrical_projection, spherical_projection
from XTA.config import resolve_tilted_view_groups
from XTA.cylindrical_cuda_projection import (
    RadialCudaProjector, RadialEncodedBlock, RadialEncodedSlice,
    RadialCudaProjectionUnavailable, validate_radial_numeric_preflight,
    rank_source_probe_planes, validate_rank_crop_samples,
)
from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnavailable
from XTA.sweep_projection_cuda import cuda_rank_projection_reader
from tests.reference_backends.radial import pull_radial_chunk
from tests.reference_backends.spherical import project_spherical_block


def _views():
    all_views = geometry.get_view_infos(7, 9, 11, cartesian_views=(),
        radial_views=('transverse', 'sagittal', 'coronal', 'tilted_transverse'),
        radial_patch_size=7, radial_min_radius=.7,
        spherical_views=('transverse', 'tilted_transverse'), spherical_patch_size=7,
        spherical_min_radius=.7, tilt_groups=resolve_tilted_view_groups(['transverse:23:vertical']))
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        result.append(next(v for v in all_views if v.family == 'radial'
            and v.radial_base_view == base and not v.radial_tilted_source))
    result.append(next(v for v in all_views if v.family == 'radial' and v.radial_tilted_source
                       and v.tilt_angle_deg > 0))
    for face in (0, 4):
        result.append(next(v for v in all_views if v.family == 'spherical'
                           and v.spherical_face == face and v.tilt_angle_deg == 0))
    result.append(next(v for v in all_views if v.family == 'spherical'
                       and v.spherical_face == 0 and v.tilt_angle_deg > 0))
    return result


def _boxes(source):
    boxes = np.zeros((source.shape[0], 4), np.int64)
    for i, plane in enumerate(source):
        yy, xx = np.nonzero(plane)
        if yy.size:
            boxes[i] = yy.min(), yy.max()+1, xx.min(), xx.max()+1
    return boxes


def _packet(first, block):
    records, payloads, offset = [], [], 0
    for local, plane in enumerate(block):
        yy, xx = np.nonzero(plane)
        if not len(yy):
            records.append(RadialEncodedSlice(first+local, 0, 0, 0, 0, 0, offset, 0))
            continue
        y0, y1, x0, x1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
        values = np.ascontiguousarray(plane[y0:y1, x0:x1]).reshape(-1)
        records.append(RadialEncodedSlice(first+local, y0, y1, x0, x1, len(yy), offset, values.size))
        payloads.append(values)
        offset += values.size
    payload = np.concatenate(payloads) if payloads else np.empty(0, np.uint8)
    payload.flags.writeable = False
    return RadialEncodedBlock(first, tuple(records), payload, False)


def _binary_oracle(source, view, shape):
    if view.family == 'radial':
        radii = np.asarray(geometry.radial_global_radii(view), np.float64)
        return np.stack([pull_radial_chunk(source, view, radii, shape, z, 0,
            shape[1]*shape[2]).reshape(shape[1:]) for z in range(shape[0])])
    return project_spherical_block(source, view, np.asarray(view.spherical_radii),
        np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], None)


@pytest.fixture
def shell_stage(monkeypatch):
    owners = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', '1')
    monkeypatch.setattr(backprojection, 'gpu_backproject_enabled', lambda: True)

    def make(source, view, shape, boxes, options):
        assert options['scalar_max'] is True and options['block_bytes'] > 0
        assert boxes is not None
        projector = SimpleNamespace(closed=False, max_block_depth=min(3, shape[0]),
            source_h2d_bytes=int(np.sum((boxes[:,1]-boxes[:,0])*(boxes[:,3]-boxes[:,2]))),
            source_layout='bbox_u8', geometry_bytes=123, metadata_d2h_bytes=0,
            output_bounds=SimpleNamespace(z0=0, z1=shape[0], y0=0, y1=shape[1], x0=0, x1=shape[2]))
        def crops(first, count):
            if view.family == 'radial':
                args = cylindrical_projection._radial_planfree_arguments(source, view,
                    np.asarray(geometry.radial_global_radii(view), np.float64), shape, boxes)
                values = []
                for z in range(first, first+count):
                    out = np.zeros(shape[1]*shape[2], np.uint8)
                    cylindrical_projection._pull_radial_range_into(out, args[0], args[1], z, 0, *args[2:], True)
                    values.append(out.reshape(shape[1:]))
            else:
                from XTA.spherical_projection_cpu import pull_spherical_chunk_numba
                values = [pull_spherical_chunk_numba(source, view, np.asarray(view.spherical_radii),
                    np.asarray(view.spherical_rotation_xyz).reshape(3,3), shape, z, 0,
                    shape[1]*shape[2], boxes, scalar_max=True).reshape(shape[1:])
                    for z in range(first, first+count)]
            projector.metadata_d2h_bytes += count*24
            return _packet(first, values)
        projector.project_rank_crops = crops
        projector.close = lambda: setattr(projector, 'closed', True)
        lease = SimpleNamespace(device_index=0, released=False)
        def release():
            assert projector.closed
            lease.released = True
        lease.release = release
        owners.append((projector, lease, source, boxes))
        stage_type = (cylindrical_projection._RadialCudaStage if view.family == 'radial'
                      else spherical_projection._SphericalCudaStage)
        return stage_type(projector, lease)
    monkeypatch.setattr(cylindrical_projection, '_try_radial_cuda_stage',
        lambda source, plan, metadata, view, shape, boxes, use_boxes, **options:
            make(source, view, shape, boxes, options))
    monkeypatch.setattr(spherical_projection, '_try_spherical_cuda_stage',
        lambda source, view, shape, boxes, **options: make(source, view, shape, boxes, options))
    return owners


@pytest.mark.parametrize('view', _views(), ids=lambda v: v.name)
def test_shell_rank_crops_equal_frozen_independent_binary_geometry(tmp_path, shell_stage, view):
    shape = (9, 12, 15)
    source = np.zeros((view.num_slices, 7, 7), np.uint8)
    source[:, 2:5, 2:5] = np.random.default_rng(168338).integers(1,6,(view.num_slices,3,3),dtype=np.uint8)
    boxes = _boxes(source)
    state = {}
    with cuda_rank_projection_reader(source, view, shape, tmp_path, level_count=5,
            memory_bytes=16*1024**2, known_native_boxes=boxes, source_max_rank=5,
            diagnostics=state) as read:
        assert read is not None, state
        actual = np.zeros(shape, np.uint8)
        for z in range(shape[0]):
            crop = read.read_crop(z)
            if crop is not None:
                y0,y1,x0,x1,values = crop
                assert values.flags.owndata
                actual[z,y0:y1,x0:x1] = values
        for cutoff in range(1,6):
            np.testing.assert_array_equal(actual>=cutoff,
                _binary_oracle((source>=cutoff).astype(np.uint8),view,shape).astype(bool))
        assert state['dense_plane_allocations'] == 0
        assert state['source_h2d_bytes'] == view.num_slices*9 < source.nbytes
    assert shell_stage[0][0].closed and shell_stage[0][1].released


def test_sparse_crop_consumption_never_allocates_full_job_shaped_output(tmp_path, monkeypatch, shell_stage):
    view = _views()[4]
    source = np.zeros((view.num_slices, 7, 7), np.uint8)
    source[:,2,3] = 5
    shape = (1123, 3022, 3064)
    def sparse(source, actual_view, actual_shape, boxes, **options):
        projector = SimpleNamespace(max_block_depth=2, source_h2d_bytes=view.num_slices,
            source_layout='bbox_u8', geometry_bytes=123, metadata_d2h_bytes=0,
            output_bounds=SimpleNamespace(z0=500,z1=502,y0=100,y1=102,x0=200,x1=203))
        def crops(first,count):
            records = tuple(RadialEncodedSlice(first+i,100,102,200,203,6,i*6,6) for i in range(count))
            return RadialEncodedBlock(first,records,np.full(count*6,5,np.uint8),False)
        projector.project_rank_crops = crops
        projector.close = lambda: None
        return spherical_projection._SphericalCudaStage(projector,
            SimpleNamespace(device_index=0,release=lambda:None))
    monkeypatch.setattr(spherical_projection,'_try_spherical_cuda_stage',sparse)
    original = np.zeros
    def forbid(shape_value,*args,**kwargs):
        if isinstance(shape_value,tuple) and shape_value == shape[1:]:
            pytest.fail('Sparse rank crop consumption allocated a full output plane')
        return original(shape_value,*args,**kwargs)
    monkeypatch.setattr(np,'zeros',forbid)
    state={}
    with cuda_rank_projection_reader(source,view,shape,tmp_path,level_count=5,
            memory_bytes=64*1024**2,known_native_boxes=_boxes(source),source_max_rank=5,
            diagnostics=state) as read:
        assert read is not None,state
        assert read.known_z_bounds==(500,502)
        assert read.known_output_bbox_tyx==(500,100,200,502,102,203)
        assert read.read_crop(0) is None
        assert read.read_crop(1122) is None
        assert read.read_crop(500)[-1].shape==(2,3)
        assert read.read_crop(501)[-1].flags.owndata
        assert state['payload_d2h_bytes']==12
        assert state['dense_plane_allocations']==0


def test_bounded_radial_plan_is_exact_and_declines_before_oversized_allocation():
    view=_views()[0]
    radii=np.asarray(geometry.radial_global_radii(view),np.float64)
    shape=(9,12,15)
    standard=cylindrical_projection._build_radial_plane_plan(view,radii,shape)
    bounded=cylindrical_projection._build_radial_plane_plan(view,radii,shape,
        max_plan_bytes=1024**2,build_chunk_pixels=7)
    for name in ('shell_index','column_offsets','native_columns'):
        np.testing.assert_array_equal(getattr(bounded,name),getattr(standard,name))
    with pytest.raises(cylindrical_projection._RadialPlanePlanTooLarge):
        cylindrical_projection._build_radial_plane_plan(view,radii,shape,max_plan_bytes=1)


def test_radial_numeric_preflight_checks_values_not_only_binary_support():
    view=_views()[0]
    source=np.full((view.num_slices,7,7),5,np.uint8)
    shape=(9,12,15)
    actual=5*_binary_oracle(np.ones_like(source),view,shape)[4]
    mode,count=validate_radial_numeric_preflight(actual,source,view,shape,z=4)
    assert count==shape[1]*shape[2]
    with pytest.raises(ValueError,match='numeric CUDA preflight'):
        validate_radial_numeric_preflight((actual>0).astype(np.uint8),source,view,shape,z=4)


@pytest.mark.parametrize('view',_views(),ids=lambda v:v.name)
def test_source_probe_planes_are_bounded_and_crop_samples_preserve_numeric_values(view):
    source=np.zeros((view.num_slices,7,7),np.uint8)
    source[:,2:4,3:5]=5
    probes=rank_source_probe_planes(source,view,(9,12,15),_boxes(source))
    assert len(probes)<=6 and all(0<=z<9 for z in probes)
    block=_packet(3,[np.full((2,3),5,np.uint8)])
    oracle=lambda z,first,stop:np.full(stop-first,5,np.uint8)
    checks,observed=validate_rank_crop_samples(block,oracle,3)
    assert checks==observed==2  # Uniform crop: maximum and first sample coincide.
    wrong=RadialEncodedBlock(block.first_z,block.records,np.ones_like(block.payload),False)
    with pytest.raises(ValueError,match='source-sample preflight'):
        validate_rank_crop_samples(wrong,oracle,3)


@pytest.mark.parametrize('projector_type',[RadialCudaProjector,SphericalCudaProjector])
def test_numeric_mode_cannot_enter_boolean_encoder(projector_type):
    projector=projector_type.__new__(projector_type)
    projector.scalar_max=True
    with pytest.raises(ValueError,match='never Boolean'):
        projector.project_encoded(0,1,False)
    projector.scalar_max=False
    with pytest.raises(ValueError,match='scalar_max=True'):
        projector.project_rank_crops(0,1)


@pytest.mark.parametrize('family',['radial','spherical'])
def test_sparse_numeric_device_credit_declines_before_any_cuda_array_allocation(monkeypatch,family):
    view=next(v for v in _views() if v.family==family)
    source=np.zeros((view.num_slices,7,7),np.uint8)
    source[:,2,3]=5
    boxes=_boxes(source)
    pool=lambda:SimpleNamespace(free_all_blocks=lambda:None)
    fake=SimpleNamespace(cuda=SimpleNamespace(Device=lambda *a:nullcontext(),
        Stream=lambda **k:SimpleNamespace(synchronize=lambda:None),MemoryPool=pool,PinnedMemoryPool=pool,
        runtime=SimpleNamespace(getDeviceCount=lambda:1,memGetInfo=lambda:(0,0))),
        empty=lambda *a,**k:pytest.fail('Unfunded CUDA array allocation'))
    monkeypatch.setitem(sys.modules,'cupy',fake)
    if family=='radial':
        radii=np.asarray(geometry.radial_global_radii(view),np.float64)
        plan=cylindrical_projection._build_radial_plane_plan(view,radii,(9,12,15))
        metadata=cylindrical_projection._radial_projection_metadata(view,source.shape,(9,12,15),plan)
        projector=RadialCudaProjector.__new__(RadialCudaProjector)
        with pytest.raises(RadialCudaProjectionUnavailable):
            projector.__init__(source,plan,metadata,view,(9,12,15),boxes,True,scalar_max=True,reserve_bytes=0)
    else:
        projector=SphericalCudaProjector.__new__(SphericalCudaProjector)
        with pytest.raises(SphericalCudaProjectionUnavailable):
            projector.__init__(source,view,(9,12,15),boxes,scalar_max=True,reserve_bytes=0)
    assert projector.source_layout=='bbox_u8'
    assert projector.source_h2d_bytes==view.num_slices < source.nbytes


@pytest.mark.skipif(os.environ.get('XTA_TEST_SWEEP_PROJECTION_CUDA')!='1',
                   reason='Coordinated actual-GPU qualification required')
@pytest.mark.parametrize('view',_views(),ids=lambda v:v.name)
def test_actual_shell_rank_crops_match_scalar_masks(tmp_path,monkeypatch,view):
    coordinator=backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0]); coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection,'_MAIN_PROCESS_GPU_STAGE_COORDINATOR',coordinator)
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT','1')
    monkeypatch.setenv('YOLO_TTA_GPU_PROJECTION_OVERLAP','0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB','1')
    source=np.zeros((view.num_slices,7,7),np.uint8)
    source[:,2:5,2:5]=np.random.default_rng(168338).integers(1,6,(view.num_slices,3,3),dtype=np.uint8)
    shape=(9,12,15)
    with cuda_rank_projection_reader(source,view,shape,tmp_path,level_count=5,
            memory_bytes=16*1024**2,known_native_boxes=_boxes(source),source_max_rank=5) as read:
        assert read is not None,'Actual GPU qualification cannot pass on a CPU fallback'
        actual=np.zeros(shape,np.uint8)
        for z in range(shape[0]):
            crop=read.read_crop(z)
            if crop is not None:
                y0,y1,x0,x1,values=crop; actual[z,y0:y1,x0:x1]=values
        for cutoff in range(1,6):
            np.testing.assert_array_equal(actual>=cutoff,
                _binary_oracle((source>=cutoff).astype(np.uint8),view,shape).astype(bool))
        assert read.projection_diagnostics()['source_h2d_bytes']<source.nbytes
    print('CUDA_RANK_METRICS '+json.dumps(dict(view=view.name,shape=shape,
        **read.projection_diagnostics()),default=int,sort_keys=True))
