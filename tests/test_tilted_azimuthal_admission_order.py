"""The public tilted Azimuthal route must not build logging-only trajectories."""
import numpy as np
import pytest

from XTA import geometry as g, backprojection as bp, projection_coverage_cpu as cpu


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for key, value in dict(CUDA_VISIBLE_DEVICES='-1', YOLO_TTA_GPU_BACKPROJECT='0',
        YOLO_TTA_GPU_BACKPROJECT_RESIDENT='0', YOLO_TTA_NATIVE_PULL_BACKEND='compiled',
        YOLO_TTA_NATIVE_PULL_PLAN_MIB='8', YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB='32').items():
        monkeypatch.setenv(key, value)


def view_for(work, base='transverse', direction='horizontal', raster=7):
    tilted = next(v for v in g._build_tilted_view_infos(*work,
        tilt_views=(base,), tilt_angles=(23.,), tilt_directions=(direction,))
        if v.tilt_angle_deg > 0)
    return g._build_azimuthal_view_info(*work, base_view=base, azimuth_angle=45.,
        azimuthal_native_raster=raster, request_token='admission-test', tilted_source=tilted)


@pytest.mark.parametrize('diameter', [4096, 8192])
@pytest.mark.parametrize('reduced', [False, True])
def test_public_rejects_before_building_trajectory(tmp_path, monkeypatch, diameter, reduced):
    work = (64, diameter, diameter)
    view = view_for(work, raster=128)
    shape = (view.num_slices, 128, 128) if reduced else (view.num_slices, view.src_h, view.src_w)
    source = np.ones(shape, np.uint8)
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_PLAN_MIB', '1')
    def forbidden(*args, **kwargs):
        raise AssertionError('trajectory built before admission')
    monkeypatch.setattr(bp, 'build_azimuthal_backprojection_plan', forbidden)
    callbacks = []
    path = tmp_path / 'unused.raw'
    with pytest.raises(cpu.NativePullPlanUnavailable, match='initializer geometry'):
        bp.backproject_azimuthal_volume_to_volume(source, view, path, 'admission test',
            out_shape_tyx=work, sink_only=True,
            projection_block_callback=lambda *args: callbacks.append(args))
    assert not callbacks and not path.exists()


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('reduced', [False, True])
def test_public_matches_compiled_core_outputs_and_callback_order(tmp_path, monkeypatch, base, direction, reduced):
    work = (9, 13, 17)
    view = view_for(work, base, direction)
    shape = (view.num_slices, 5, 5) if reduced else (view.num_slices, view.src_h, view.src_w)
    source = np.random.default_rng(771).integers(0, 2, shape, dtype=np.uint8)
    original = source.copy()
    public_planes, core_planes = [], []
    def callback(target):
        return lambda first, value: target.append((first, value.copy()))
    build_counts = []
    real_build = bp.build_azimuthal_backprojection_plan
    def build(view):
        plan, stats = real_build(view)
        build_counts.append(len(plan))
        return plan, stats
    monkeypatch.setattr(bp, 'build_azimuthal_backprojection_plan', build)
    bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path/'public.raw', 'public',
        out_shape_tyx=work, workers=2, sink_only=True,
        projection_block_callback=callback(public_planes))
    assert len(build_counts) == 1
    bp._backproject_native_destination_pull(source, view, tmp_path/'core.raw', 'core',
        output_shape=work, workers=2, sink_only=True, callback=callback(core_planes))
    assert len(build_counts) == 2
    assert [i for i,_ in public_planes] == [i for i,_ in core_planes]
    assert len(public_planes) == work[0]
    for (_, public), (_, core) in zip(public_planes, core_planes):
        np.testing.assert_array_equal(public, core)
    np.testing.assert_array_equal(source, original)
    assert not (tmp_path/'public.raw').exists() and not (tmp_path/'core.raw').exists()
