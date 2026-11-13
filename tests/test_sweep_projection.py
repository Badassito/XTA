"""Every level of one rank projection matches actual scalar binary output."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import TiltedViewGroup
from XTA.outputs import _read_layer_slice_in_output_shape
from XTA.sweep_projection import sweep_rank_projection_reader


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","-1")
    for name in ("YOLO_TTA_GPU_BACKPROJECT","YOLO_TTA_GPU_BACKPROJECT_RESIDENT",
                 "YOLO_TTA_GPU_RADIAL_BACKPROJECT","YOLO_TTA_GPU_SPHERICAL_BACKPROJECT"):
        monkeypatch.setenv(name,"0")


def views():
    all_views=geometry.get_view_infos(5,7,9,
        cartesian_views=("transverse","sagittal","coronal"),
        tilt_groups=(TiltedViewGroup(("transverse","sagittal","coronal"),(23.,),("vertical","horizontal")),),
        azimuthal_views=("transverse","sagittal","coronal","tilted_transverse","tilted_sagittal","tilted_coronal"),
        azimuthal_azimuth_angles=(45.,)*6,azimuthal_native_raster=0,
        radial_views=("transverse","sagittal","coronal","tilted_transverse"),
        radial_min_radius=.7,radial_patch_size=5,
        spherical_views=("transverse",),spherical_min_radius=.7,spherical_patch_size=5)
    chosen=[];seen=set()
    for v in all_views:
        if v.family=="radial":
            key=(v.family,v.radial_base_view,v.radial_tilted_source,v.tilt_direction)
        elif v.family=="spherical":
            key=(v.family,v.spherical_face)
        elif v.family=="azimuthal":
            key=(v.family,v.azimuthal_base_view,v.azimuthal_tilted_source)
        else:key=v.name
        if key not in seen:
            chosen.append(v);seen.add(key)
    from XTA.spherical_geometry import cube_rotation
    for v in list(chosen):
        if v.family=="orthogonal":
            chosen.append(replace(v,name=v.name+"__tta_37",physical_view_name=v.name,tta_angle_deg=37.))
        if v.family=="spherical" and v.spherical_face in (0,3,5):
            chosen.append(replace(v,name=v.name+"_rotated",
                spherical_rotation_xyz=tuple(np.asarray(cube_rotation("vertical",31.)).reshape(-1))))
    return chosen


def scalar_binary(source,view,shape,path):
    cartesian=view.family not in ("radial","spherical","azimuthal") and not geometry.is_tilted_view(view)
    # Upright geometry is pure permutation; final publication owns restoration.
    result=assembly.project_view_volume_to_orthogonal_volume(source,view,path,"scalar sweep oracle",
        workers=1,prefer_memory=False,reserve_bytes=0,
        out_shape_tyx=None if cartesian else shape)
    try:
        return np.stack([_read_layer_slice_in_output_shape(result,shape,z) for z in range(shape[0])])
    finally:
        if isinstance(result,np.memmap):result._mmap.close()
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("view",views(),ids=lambda v:v.name)
@pytest.mark.parametrize("reduced",[False,True],ids=["native","model-raster"])
def test_every_level_matches_actual_scalar_pipeline_projection(tmp_path,view,reduced):
    source_shape=(view.num_slices,3,3) if reduced else (view.num_slices,view.src_h,view.src_w)
    source=np.random.default_rng(91).integers(0,4,source_shape,dtype=np.uint8)
    original=source.copy()
    for shape in ((3,4,5),(8,10,12)):
        with sweep_rank_projection_reader(source,view,shape,tmp_path/"rank",level_count=3,memory_bytes=4*1024**2) as read:
            projected=np.stack([read(z) for z in range(shape[0])])
            diagnostics=read.projection_diagnostics()
            assert diagnostics["read_planes"]==shape[0]
        with pytest.raises(RuntimeError,match="closed"):read(0)
        for level in range(1,4):
            expected=scalar_binary((source>=level).astype(np.uint8),view,shape,tmp_path/"scalar.dat")
            np.testing.assert_array_equal(projected>=level,expected>0)
    np.testing.assert_array_equal(source,original)
    assert not list((tmp_path/"rank").iterdir())


def independent_upright_restore(source,shape):
    """Scalar source-cell oracle, independent of production resize helpers."""
    iz,iy,ix=source.shape;oz,oy,ox=shape
    result=np.zeros(shape,bool)
    area=iy>=oy and ix>=ox
    for z,y,x in np.ndindex(shape):
        zs=(range(z*iz//oz,min(iz,max(z*iz//oz+1,((z+1)*iz+oz-1)//oz))) if iz>=oz
            else (min(iz-1,max(0,round(0. if iz<=1 or oz<=1 else z*(iz-1)/(oz-1)))),))
        ys=(range(y*iy//oy,((y+1)*iy+oy-1)//oy) if area else (y*iy//oy,))
        xs=(range(x*ix//ox,((x+1)*ix+ox-1)//ox) if area else (x*ix//ox,))
        result[z,y,x]=any(source[a,b,c] for a in zs for b in ys for c in xs)
    return result


@pytest.mark.parametrize("base",("transverse","sagittal","coronal"))
@pytest.mark.parametrize("shape",((7,8,9),(2,2,2),(7,2,9),(1,1,1)))
def test_upright_reduced_axes_and_endpoint_z_rounding_have_independent_oracle(tmp_path,base,shape):
    view=geometry.get_view_infos(5,7,9,cartesian_views=(base,))[0]
    source=np.random.default_rng(72).integers(0,7,(view.num_slices,3,3),dtype=np.uint8)
    orthogonal=source.transpose({"transverse":(0,1,2),"sagittal":(1,0,2),"coronal":(1,2,0)}[base])
    with sweep_rank_projection_reader(source,view,shape,tmp_path,level_count=6,memory_bytes=2*1024**2) as read:
        projected=np.stack([read(z) for z in range(shape[0])])
    for level in range(1,7):
        expected=independent_upright_restore(orthogonal>=level,shape)
        np.testing.assert_array_equal(projected>=level,expected)


@pytest.mark.parametrize("family",("radial","spherical"))
def test_sparse_rectangle_skips_actual_output_points_and_exposes_bounds(tmp_path,family):
    kwargs=({"radial_views":("transverse",),"radial_min_radius":.2,"radial_patch_size":5}
            if family=="radial" else
            {"spherical_views":("transverse",),"spherical_min_radius":.2,"spherical_patch_size":5})
    view=next(v for v in geometry.get_view_infos(11,13,15,cartesian_views=(),**kwargs) if v.family==family)
    source=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    source[0]=3
    shape=(21,25,29)
    with sweep_rank_projection_reader(source,view,shape,tmp_path/"work",level_count=3,memory_bytes=4*1024**2) as read:
        actual=np.stack([read(z) for z in range(shape[0])])
        diagnostics=read.projection_diagnostics()
        assert diagnostics["projected_points"]<np.prod(shape)
        assert len(read.known_z_bounds)==2
        assert len(read.known_output_bbox_tyx)==6
    for level in range(1,4):
        expected=scalar_binary((source>=level).astype(np.uint8),view,shape,tmp_path/"scalar.dat")
        np.testing.assert_array_equal(actual>=level,expected>0)


def test_memory_and_rank_admission_precede_workspace_and_no_source_copy(tmp_path):
    view=geometry.get_view_infos(3,5,7,cartesian_views=("transverse",))[0]
    source=np.full((3,5,7),4,np.uint8)
    with pytest.raises(ValueError,match="exceeds"):
        with sweep_rank_projection_reader(source,view,(3,5,7),tmp_path/"bad",level_count=3):pass
    assert not (tmp_path/"bad").exists()
    with pytest.raises(MemoryError):
        with sweep_rank_projection_reader(source,view,(3,512,512),tmp_path/"small",level_count=4,memory_bytes=1024):pass
    assert not (tmp_path/"small").exists()
    for invalid in (0,256,True,1.5):
        with pytest.raises(ValueError):
            with sweep_rank_projection_reader(source,view,(3,5,7),tmp_path,level_count=invalid):pass



def test_packer_certificate_skips_source_scan_and_is_exact(tmp_path,monkeypatch):
    from XTA import sweep_projection
    view=geometry.get_view_infos(3,5,7,cartesian_views=("transverse",))[0]
    source=np.random.default_rng(33).integers(0,4,(3,5,7),dtype=np.uint8)
    boxes=np.tile(np.array([0,5,0,7],np.int64),(3,1))
    def forbidden(*_args):
        raise AssertionError("Certified normal path rescanned the native rank mmap")
    monkeypatch.setattr(sweep_projection,"_rank_boxes",forbidden)
    with sweep_rank_projection_reader(source,view,(3,5,7),tmp_path,level_count=3,
            known_native_boxes=boxes) as read:
        actual=np.stack([read(z) for z in range(3)])
        state=read.projection_diagnostics()
    np.testing.assert_array_equal(actual,source)
    assert state["source_scan_voxels"]==0
    assert state["source_max_rank_is_upper_bound"]


@pytest.mark.parametrize("boxes",[
    np.zeros((3,4),np.float64), np.zeros((2,4),np.int64),
    np.array([[0,6,0,7]]*3,np.int64),np.array([[0,5,0,8]]*3,np.int64),
    np.array([[-1,5,0,7]]*3,np.int64),np.array([[2,1,0,7]]*3,np.int64),
    np.array([[0,0,1,2]]*3,np.int64),np.array([[1,1,2,2]]*3,np.int64),
])
def test_invalid_packer_bounds_rejected_before_work_directory(tmp_path,boxes):
    view=geometry.get_view_infos(3,5,7,cartesian_views=("transverse",))[0]
    source=np.zeros((3,5,7),np.uint8)
    with pytest.raises(ValueError,match="known_native_boxes"):
        with sweep_rank_projection_reader(source,view,(3,5,7),tmp_path/"bad",
                level_count=3,known_native_boxes=boxes):pass
    assert not (tmp_path/"bad").exists()



@pytest.mark.parametrize("first,count",[(0,35),(1,2),(3,13),(7,20),(32,3)])
def test_rectangle_copy_has_no_coordinate_gather_and_preserves_exact_flat_order(first,count):
    from XTA.sweep_projection import _copy_rectangle_strip
    result=np.zeros((9,13),np.uint8)
    values=np.arange(1,count+1,dtype=np.uint8)
    _copy_rectangle_strip(result,values,first,2,3,7)
    expected=np.zeros_like(result)
    for offset,value in enumerate(values):
        row,column=divmod(first+offset,7)
        expected[2+row,3+column]=value
    np.testing.assert_array_equal(result,expected)


@pytest.mark.parametrize("base",("transverse","sagittal","coronal"))
@pytest.mark.parametrize("numeric",(False,True))
def test_compiled_radial_roi_strips_preserve_actual_flat_kernel(base,numeric):
    from XTA.cylindrical_geometry import global_radii
    from XTA.cylindrical_projection import _pull_radial_chunk_compiled, _pull_radial_rectangle_compiled
    view=next(v for v in geometry.get_view_infos(5,7,9,cartesian_views=(),
        radial_views=(base,),radial_min_radius=.2,radial_patch_size=5) if v.family=="radial")
    source=np.random.default_rng(111).integers(0,7,(view.num_slices,3,3),dtype=np.uint8)
    shape=(8,10,12)
    radii=np.asarray(global_radii(view),np.float64)
    bounds=(1,9,2,11)
    for z in (1,4,7):
        full=_pull_radial_chunk_compiled(source,view,radii,shape,z,0,120,scalar_max=numeric).reshape(10,12)
        expected=full[1:9,2:11].reshape(-1)
        for first in range(0,len(expected),13):
            stop=min(len(expected),first+13)
            actual=_pull_radial_rectangle_compiled(source,view,radii,shape,z,first,stop,
                bounds_yx=bounds,scalar_max=numeric)
            np.testing.assert_array_equal(actual,expected[first:stop])



@pytest.mark.parametrize("family",("orthogonal","tilted","azimuthal","radial","spherical"))
def test_cpu_parallel_lanes_match_serial_outputs_with_bounded_queued_credit(tmp_path,family):
    from concurrent.futures import ThreadPoolExecutor
    view=next(v for v in views() if v.family==family)
    source=np.random.default_rng(314).integers(0,4,(view.num_slices,3,3),dtype=np.uint8)
    original=source.copy()
    shape=(8,10,12)
    with sweep_rank_projection_reader(source,view,shape,tmp_path/"serial",level_count=3,workers=1) as read:
        expected=[read(z) for z in range(shape[0])]
    with sweep_rank_projection_reader(source,view,shape,tmp_path/"parallel",level_count=3,
            workers=3,memory_bytes=4*1024**2) as read:
        assert read.parallel_workers==3
        with ThreadPoolExecutor(max_workers=3) as executor:
            actual=list(executor.map(read,range(shape[0])))
        state=read.projection_diagnostics()
        assert state["read_planes"]==shape[0]
        assert state["active_reads"]==0
        assert state["cpu_admitted_bytes"]<=state["budget_bytes"]
        assert state["queued_plane_credit_bytes"]==3*np.prod(shape[1:])
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(source,original)


def test_cpu_lane_count_reduces_to_fit_working_and_completed_output_planes(tmp_path):
    view=geometry.get_view_infos(5,64,64,cartesian_views=("transverse",))[0]
    source=np.ones((5,3,3),np.uint8)
    with sweep_rank_projection_reader(source,view,(5,512,512),tmp_path,level_count=1,
            workers=8,memory_bytes=2*1024**2) as read:
        state=read.projection_diagnostics()
        assert read.parallel_workers==2
        assert state["cpu_admitted_bytes"]<=2*1024**2
        assert state["cpu_lane_working_bytes"]+512*512 <= (
            state["budget_bytes"]-state["source_metadata_bytes"])//read.parallel_workers


@pytest.mark.parametrize("family",("radial","spherical"))
def test_curved_crop_does_not_allocate_dense_plane_and_matches_dense(tmp_path,monkeypatch,family):
    from XTA import sweep_projection
    kwargs=({"radial_views":("transverse",),"radial_min_radius":.2,"radial_patch_size":5}
            if family=="radial" else
            {"spherical_views":("transverse",),"spherical_min_radius":.2,"spherical_patch_size":5})
    view=next(v for v in geometry.get_view_infos(11,13,15,cartesian_views=(),**kwargs) if v.family==family)
    source=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    source[0]=3
    shape=(61,71,81)
    with sweep_rank_projection_reader(source,view,shape,tmp_path,level_count=3,workers=2) as read:
        z0,z1=read.known_z_bounds
        z=min(z1-1,max(z0,shape[0]//2))
        dense=read(z)
        original_zeros=np.zeros
        def checked_zeros(request,*args,**kwargs):
            if isinstance(request,(tuple,list)) and tuple(request)==shape[1:]:
                raise AssertionError("Sparse crop allocated the dense output plane")
            return original_zeros(request,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(sweep_projection.np,"zeros",checked_zeros)
            crop=read.read_crop(z)
        assert crop is not None
        y0,y1,x0,x1,data=crop
        assert data.shape==(y1-y0,x1-x0)
        rebuilt=np.zeros(shape[1:],np.uint8);rebuilt[y0:y1,x0:x1]=data
        np.testing.assert_array_equal(rebuilt,dense)


def test_parallel_lane_dispatch_really_overlaps_independent_curved_readers(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from XTA import sweep_projection
    factory=sweep_projection._curved_reader
    barrier=threading.Barrier(3)
    def synchronized_factory(*args,**kwargs):
        actual=factory(*args,**kwargs)
        first=True
        def synchronized(z):
            nonlocal first
            if first:
                first=False
                barrier.wait(timeout=10)
            return actual(z)
        synchronized.projection_diagnostics=lambda:dict(args[-1])
        return synchronized
    monkeypatch.setattr(sweep_projection,"_curved_reader",synchronized_factory)
    view=next(v for v in views() if v.family=="radial")
    source=np.full((view.num_slices,3,3),3,np.uint8)
    with sweep_rank_projection_reader(source,view,(8,10,12),tmp_path,level_count=3,workers=3) as read:
        with ThreadPoolExecutor(max_workers=3) as executor:
            actual=list(executor.map(read,(2,3,4)))
        assert len(actual)==3
        assert read.projection_diagnostics()["peak_active_reads"]==3


def test_cuda_attempt_gets_full_budget_one_lane_and_crop_forwarding(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from XTA import sweep_projection_cuda
    observed={}
    @contextmanager
    def fake(source,view,shape,work,**kwargs):
        observed.update(kwargs)
        def read(_z):
            raise AssertionError("Crop forwarding called dense CUDA reader")
        read.read_crop=lambda _z:(1,3,2,5,np.ones((2,3),np.uint8))
        read.known_z_bounds=(0,shape[0])
        read.known_output_bbox_tyx=(0,1,2,shape[0],3,5)
        read.projection_diagnostics=lambda:{"backend":"cuda_test"}
        yield read
    monkeypatch.setattr(sweep_projection_cuda,"cuda_rank_projection_reader",fake)
    view=geometry.get_view_infos(3,5,7,cartesian_views=("transverse",))[0]
    source=np.ones((3,5,7),np.uint8)
    with sweep_rank_projection_reader(source,view,(3,5,7),tmp_path,level_count=1,
            workers=8,memory_bytes=2*1024**2) as read:
        assert read.parallel_workers==1
        crop=read.read_crop(1)
        assert crop[-1].sum()==6
        state=read.projection_diagnostics()
        assert observed["memory_bytes"]==state["budget_bytes"]-state["source_metadata_bytes"]
