"""Production view-family projection and store ownership, including arbitrary nonnested cutoffs."""
from contextlib import closing

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import TiltedViewGroup
from XTA.interpolation import RawBBoxMaskStore
from XTA.outputs import _read_layer_slice_in_output_shape
from XTA.sweep_backprojection import SweepProjectionUnsupported


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    for name in ("YOLO_TTA_GPU_BACKPROJECT","YOLO_TTA_GPU_BACKPROJECT_RESIDENT",
                 "YOLO_TTA_GPU_RADIAL_BACKPROJECT","YOLO_TTA_GPU_SPHERICAL_BACKPROJECT"):
        monkeypatch.setenv(name, "0")


def make_views():
    result = geometry.get_view_infos(5,7,9,cartesian_views=("transverse","sagittal","coronal"),
        tilt_groups=(TiltedViewGroup(("transverse",),(23.,),("vertical",)),),
        azimuthal_views=("transverse","tilted_transverse"),azimuthal_azimuth_angles=(45.,45.),azimuthal_native_raster=0,
        radial_views=("transverse","tilted_transverse"),radial_min_radius=.7,radial_patch_size=5,
        spherical_views=("transverse","tilted_transverse"),spherical_min_radius=.7,spherical_patch_size=5)
    selected = {}
    for v in result:
        if v.family == "orthogonal":
            key = v.name
        elif v.family == "azimuthal":
            key = v.family, v.azimuthal_tilted_source
        elif v.family == "radial":
            key = v.family, v.radial_tilted_source
        elif v.family == "spherical":
            key = v.family, v.spherical_tilted_source
        else:
            key = v.family
        selected.setdefault(key, v)
    return list(selected.values())


def read(ref):
    with closing(RawBBoxMaskStore.open(ref.path, mmap_payload=True)) as store:
        return np.stack([store.decode_slice(z) for z in range(ref.shape[0])])


def reference(source, view, target, work):
    array = assembly.project_view_volume_to_orthogonal_volume(source,view,work,"scalar projection check",
        workers=1, prefer_memory=False, reserve_bytes=0, out_shape_tyx=None if view.family=="orthogonal" else target)
    try:
        return np.array(array, copy=True)
    finally:
        from XTA.runtime import close_memmap_array_without_flush
        close_memmap_array_without_flush(array,unlink_path=work)


@pytest.mark.parametrize("view", make_views(), ids=lambda v:v.name)
def test_all_families_use_one_projection_of_native_fullframe_cutoffs(view,tmp_path):
    shape = (view.num_slices,view.src_h,view.src_w)
    rank = np.random.default_rng(19).integers(0,4,size=shape,dtype=np.uint8)
    cuts = [50,60,70,100]
    masks = {cut:(rank>=i).astype(np.uint8) for i,cut in enumerate(cuts,1)}
    target = (5,7,9)
    metrics = {}
    views = assembly.project_sweep_view(view=view, model_name="test", cutoffs_hundredths=cuts,
        read_mask_plane=lambda cut,z:masks[cut][z], native_shape=shape, output_shape=target,
        work_dir=tmp_path/"group", metrics=metrics)
    assert metrics["geometry_passes"] == 1 and metrics["nested_chain_count"]==1
    assert views[100] is None
    assert set(views) == set(cuts)
    for cut in cuts[:3]:
        ref = views[cut]
        assert ref.path.is_relative_to(tmp_path/"group") and ref.path.is_dir()
        assert (ref.source,ref.mask_kind,ref.stage,ref.model_name,ref.view_name)==(
            "fullframe","yolo","pre_interpolation","test",view.name)
        actual = read(ref)
        expected = reference(masks[cut],view,target,tmp_path/f"oracle_{cut}.dat")
        np.testing.assert_array_equal(actual,expected)
        assert ref.shape == expected.shape
        pts = np.argwhere(expected)
        extent = tuple(v for lo,hi in zip(pts.min(0)[::-1],pts.max(0)[::-1]) for v in (lo,hi)) if pts.size else (0,-1,0,-1,0,-1)
        assert ref.segment_extent_ijk == extent


def test_arbitrary_nonnested_occupancy_and_overlapping_crops_remain_exact(tmp_path):
    shape = (3,8,11)
    v = geometry.ViewInfo("transverse",shape[0],shape[1],shape[2],"clamp",full_t=3,full_h=8,full_w=11)
    a = np.zeros(shape,np.uint8)
    a[:,1:6,2:9]=1
    b = a.copy(); b[:,1,:]=0
    c = np.zeros_like(a); c[1:,5:8,2:6]=1
    d = c.copy(); d[:,7,:]=0
    e = np.zeros_like(a); e[0,0,0]=1
    masks = dict(zip((0,10,20,30,40),(a,b,c,d,e)))
    def crops(cut,z):
        data = masks[cut][z]
        # Opaque sparse readers can send multiple overlapping contributions.
        yield 0,8,0,7,data[:,:7]
        yield 0,8,5,11,data[:,5:]
    metrics = {}
    result = assembly.project_sweep_view(view=v,model_name="m",cutoffs_hundredths=tuple(masks),
        native_shape=shape,output_shape=shape,work_dir=tmp_path/"group",read_mask_crops=crops,metrics=metrics)
    assert metrics["geometry_passes"] == 3 and metrics["nested_chain_count"] == 3
    for cent,ref in result.items():
        np.testing.assert_array_equal(read(ref),masks[cent])
    assert len(list((tmp_path/"group").iterdir())) == 1


def test_undefined_or_failed_view_never_publishes_partial_paths(tmp_path):
    v = geometry.ViewInfo("transverse",3,4,5,"clamp",full_t=3,full_h=4,full_w=5)
    shape = (3,4,5)
    kw = dict(view=v,model_name="m",cutoffs_hundredths=(50,75),native_shape=shape,
              output_shape=shape,work_dir=tmp_path/"group")
    empty = np.zeros(shape[1:],np.uint8)
    stats = {}
    assert assembly.project_sweep_view(**kw,read_mask_plane=lambda c,z:empty,metrics=stats)=={50:None,75:None}
    assert stats["geometry_passes"]==0
    def broken(c,z):
        if z==1: raise ValueError("incomplete native range")
        return empty
    with pytest.raises(ValueError,match="incomplete"):
        assembly.project_sweep_view(**kw,read_mask_plane=broken)
    with pytest.raises(ValueError,match="binary"):
        assembly.project_sweep_view(**kw,read_mask_plane=lambda c,z:np.full(shape[1:],2,np.uint8))
    with pytest.raises(SweepProjectionUnsupported,match="scratch"):
        assembly.project_sweep_view(**kw,read_mask_plane=lambda c,z:empty, max_workspace_bytes=1)
    assert list((tmp_path/"group").iterdir()) == []


def test_reduced_cartesian_ref_retains_original_terminal_source_restore(tmp_path):
    v = geometry.ViewInfo("sagittal",7,5,9,"clamp",full_t=5,full_h=7,full_w=9)
    shape = (7,3,4)
    target = (5,7,9)
    a = np.random.default_rng(21).integers(0,2,size=shape,dtype=np.uint8)
    refs=assembly.project_sweep_view(view=v,model_name="m",cutoffs_hundredths=(50,),native_shape=shape,
        output_shape=target,work_dir=tmp_path/"group",read_mask_plane=lambda c,z:a[z])
    actual=read(refs[50])
    assert actual.shape==(3,7,4)
    expected=a.transpose(1,0,2)
    for z in range(target[0]):
        np.testing.assert_array_equal(_read_layer_slice_in_output_shape(actual,target,z),
                                      _read_layer_slice_in_output_shape(expected,target,z))


@pytest.mark.parametrize("name", ("transverse","sagittal","coronal"))
def test_d1_empty_slot_uses_original_source_grid_without_projection(tmp_path, name):
    target = (5,7,9)
    depth = {"transverse":5,"sagittal":7,"coronal":9}[name]
    view = geometry.ViewInfo(name, depth, 7, 9, "clamp", full_t=5, full_h=7, full_w=9)
    kw = dict(view=view, model_name="model", cutoffs_hundredths=(100,), native_shape=(depth,3,4),
              output_shape=target, read_mask_crops=lambda *_:(), work_dir=tmp_path/"group")
    assert assembly.project_sweep_view(**kw) == {100:None}
    metrics = {}
    ref = assembly.project_sweep_view(**kw, retain_empty_orthogonal=True, metrics=metrics)[100]
    assert ref.shape == target and ref.segment_extent_ijk == (0,-1,0,-1,0,-1)
    assert ref.segment_extent_source == "d1_incremental_source_space_cvol_index"
    assert ref.key.endswith("__fullframe__yolo__pre_interpolation") and ref.mask_kind == "yolo"
    assert not np.any(read(ref)) and metrics["geometry_passes"] == 0


@pytest.mark.parametrize("name", ("transverse","sagittal","coronal"))
def test_nonempty_d1_source_grid_equals_independent_d1_reference(tmp_path, name):
    from tests.reference_backends.d1_orthogonal_coverage import execute_orthogonal_coverage_reference
    target = (5,7,9)
    depth = {"transverse":11,"sagittal":7,"coronal":9}[name]
    # Includes a temporal contraction (sagittal and coronal rows are time),
    # XY upsampling, and the real owner's edge/rounding behavior.
    native = (depth,11,4)
    view = geometry.ViewInfo(name,depth,11,4,"clamp",full_t=5,full_h=7,full_w=9)
    ranks = np.random.default_rng(158).integers(0,3,size=native,dtype=np.uint8)
    refs = assembly.project_sweep_view(view=view,model_name="m",cutoffs_hundredths=(50,75),
        native_shape=native, output_shape=target, read_mask_plane=lambda c,z: ranks[z] >= (1 if c==50 else 2),
        work_dir=tmp_path, retain_empty_orthogonal=True)
    for level, cent in enumerate((50,75),1):
        assert refs[cent].shape == target
        expected = execute_orthogonal_coverage_reference((ranks>=level).astype(np.uint8),view,target)
        np.testing.assert_array_equal(read(refs[cent]), expected)


def test_sparse_crop_pipeline_never_scans_or_requests_empty_dense_output(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from XTA import sweep_backprojection as module
    shape = (17,64,85)
    view = geometry.ViewInfo("transverse",17,64,85,"clamp",full_t=17,full_h=64,full_w=85)
    tiny = np.array([[2,1,0],[2,0,1]],np.uint8)
    def crops(c,z):
        if z == 5:
            yield 8,10,9,12,(tiny >= (1 if c == 50 else 2))
    @contextmanager
    def reader(*args, **kwargs):
        def forbidden(*_):
            raise AssertionError("dense output must not be materialized")
        forbidden.known_z_bounds = (4,7)
        forbidden.read_crop = lambda z:(8,10,9,12,tiny.copy()) if z==5 else None
        forbidden.parallel_workers = 2
        yield forbidden
    monkeypatch.setattr(module,"sweep_rank_projection_reader",reader)
    stats={}
    refs=module.project_sweep_view(view=view,model_name="m",cutoffs_hundredths=(50,75),
        native_shape=shape,output_shape=shape,read_mask_crops=crops,work_dir=tmp_path,workers=2,metrics=stats)
    info=stats["projectors"][0]
    assert info["rank_crop_reads"] == 3 and info["rank_dense_reads"] == 0
    assert info["rank_bbox_scan_pixels"] == 0 and info["rank_statistics_pixels"] == 6
    assert info["empty_crop_reads"] == 2 and info["execution_lanes"] == 2
    for i,cent in enumerate((50,75),1):
        dst=read(refs[cent])
        np.testing.assert_array_equal(dst[5,8:10,9:12],tiny>=i)
        assert dst.sum()==(tiny>=i).sum()


def test_ordered_projectors_overlap_without_read_ahead_past_lane_budget():
    from threading import Event, Lock
    from XTA.sweep_backprojection import _ordered_planes
    later_started=Event()
    lock=Lock()
    active=set()
    calls=[]
    def plane(z):
        with lock:
            calls.append(z)
            active.add(z)
        if z==0:
            assert later_started.wait(2)
        elif z==1:
            later_started.set()
        with lock:
            active.remove(z)
        return z
    with closing(_ordered_planes(plane,0,5,2)) as results:
        assert next(results) == (0,0)
        assert sorted(calls)==[0,1]
        assert list(results)==[(1,1),(2,2),(3,3),(4,4)]
    assert not active


def test_sparse_scratch_admission_charges_touched_pages_and_tmpfs_headroom(tmp_path,monkeypatch):
    from XTA import runtime, workspace
    shape=(3,512,512)
    view=geometry.ViewInfo("transverse",3,512,512,"clamp",full_t=3,full_h=512,full_w=512)
    kw=dict(view=view, model_name="m", cutoffs_hundredths=(50,), native_shape=shape,
            output_shape=shape,work_dir=tmp_path/"sparse",
            read_mask_crops=lambda c,z: [(1,2,3,4,np.ones((1,1),np.uint8))] if z==1 else [])
    stats={}
    ref=assembly.project_sweep_view(**kw,max_workspace_bytes=192*1024,metrics=stats)[50]
    assert ref.path.is_dir() and stats["native_sparse_page_upper_bound"]<=128*1024
    assert stats["native_rank_backing_bytes"]==np.prod(shape)
    assert stats["output_payload_bytes"]==1
    monkeypatch.setattr(runtime,"path_is_memory_backed",lambda path:True)
    monkeypatch.setattr(workspace,"available_anon_work_bytes",lambda:0)
    with pytest.raises(SweepProjectionUnsupported,match="tmpfs headroom"):
        assembly.project_sweep_view(**kw)


def test_lane_pack_kernel_matches_numpy_packbits_exactly():
    from XTA.sweep_backprojection import _pack_rank_level
    rng = np.random.default_rng(20261112)
    for _ in range(300):
        h, w = int(rng.integers(1, 40)), int(rng.integers(1, 70))
        region = rng.integers(0, 6, size=(h, w), dtype=np.uint8)
        y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        y1, x1 = int(rng.integers(y0 + 1, h + 1)), int(rng.integers(x0 + 1, w + 1))
        level = int(rng.integers(1, 6))
        expected = np.packbits(np.greater_equal(region[y0:y1, x0:x1], level), axis=1, bitorder="little")
        got = _pack_rank_level(region, y0, y1, x0, x1, level)
        assert got.dtype == np.uint8 and got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


def test_single_lane_reader_sees_planes_in_order_while_encodes_run_in_parallel(tmp_path, monkeypatch):
    """A CUDA-style reader (one serialized lane with a block cache) must be read in z order."""
    import contextlib
    import threading
    import time as _time
    from XTA import sweep_backprojection as sweep

    view = next(v for v in make_views() if v.family == "spherical")
    shape = (view.num_slices, view.src_h, view.src_w)
    rank = np.random.default_rng(23).integers(0, 4, size=shape, dtype=np.uint8)
    cuts = [50, 60, 70]
    masks = {cut: (rank >= i).astype(np.uint8) for i, cut in enumerate(cuts, 1)}
    real_reader = sweep.sweep_rank_projection_reader
    order, lock = [], threading.Lock()
    jitter = np.random.default_rng(5)

    @contextlib.contextmanager
    def single_lane(*args, **kwargs):
        with real_reader(*args, **{**kwargs, "workers": 1}) as project:
            def read_crop(z):
                with lock:
                    order.append(z)
                    delay = float(jitter.uniform(0, 0.004))
                _time.sleep(delay)
                return project.read_crop(z)
            read_crop.read_crop = read_crop
            for name in ("known_z_bounds", "known_output_bbox_tyx", "projection_diagnostics"):
                if hasattr(project, name):
                    setattr(read_crop, name, getattr(project, name))
            read_crop.parallel_workers = 1
            read_crop.serialized_device_reader = True
            yield read_crop

    def run(workers, folder):
        monkeypatch.setattr(sweep, "sweep_rank_projection_reader", single_lane)
        try:
            return assembly.project_sweep_view(view=view, model_name="m", cutoffs_hundredths=cuts,
                read_mask_plane=lambda c, z: masks[c][z], native_shape=shape, output_shape=(5, 7, 9),
                work_dir=tmp_path / folder, workers=workers)
        finally:
            monkeypatch.setattr(sweep, "sweep_rank_projection_reader", real_reader)

    serial = run(1, "serial")
    order.clear()
    parallel = run(4, "parallel")
    assert order == sorted(order), "a single-lane reader must be read in increasing z"
    for cut in cuts:
        if serial[cut] is None:
            assert parallel[cut] is None
            continue
        np.testing.assert_array_equal(read(serial[cut]), read(parallel[cut]))


def test_failed_single_lane_reader_is_not_called_for_later_planes(tmp_path, monkeypatch):
    """After a serialized device reader fails, no later plane may reach it."""
    import contextlib
    from XTA import sweep_backprojection as sweep

    view = next(v for v in make_views() if v.family == "spherical")
    shape = (view.num_slices, view.src_h, view.src_w)
    masks = {50: np.ones(shape, np.uint8), 60: np.ones(shape, np.uint8)}
    calls = []
    real_reader = sweep.sweep_rank_projection_reader

    @contextlib.contextmanager
    def failing(*args, **kwargs):
        with real_reader(*args, **{**kwargs, "workers": 1}) as project:
            z0 = getattr(project, "known_z_bounds", (0, shape[0]))[0]

            def read_crop(z):
                calls.append(z)
                if z == z0:
                    raise OSError("device session quarantined")
                return project.read_crop(z)
            read_crop.read_crop = read_crop
            for name in ("known_z_bounds", "known_output_bbox_tyx", "projection_diagnostics"):
                if hasattr(project, name):
                    setattr(read_crop, name, getattr(project, name))
            read_crop.parallel_workers = 1
            read_crop.serialized_device_reader = True
            yield read_crop

    monkeypatch.setattr(sweep, "sweep_rank_projection_reader", failing)
    with pytest.raises(OSError, match="quarantined"):
        assembly.project_sweep_view(view=view, model_name="m", cutoffs_hundredths=(50, 60),
            read_mask_plane=lambda c, z: masks[c][z], native_shape=shape, output_shape=(5, 7, 9),
            work_dir=tmp_path / "w", workers=4)
    assert len(calls) == 1, calls
