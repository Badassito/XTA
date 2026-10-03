"""Plan admission, fallback and ownership of compiled native score readers."""
import gc
import sys
import types
import weakref

import numpy as np
import pytest

from XTA import geometry
from XTA.config import TiltedViewGroup
from XTA.confidence_projection import score_projection_reader, _destination_score_reader


class NativePullPlanUnavailable(MemoryError):
    pass


@pytest.fixture(autouse=True)
def compiled_backend(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled')


def _fixture():
    view = geometry.get_view_infos(5, 7, 9, cartesian_views=(),
        tilt_groups=(TiltedViewGroup(('transverse',), (23.,), ('vertical',)),))[0]
    return view, np.full((view.num_slices, 3, 3), 173, np.uint8), (8, 10, 12)


def _fake_backend(monkeypatch, prepare, pull):
    module = types.ModuleType('XTA.projection_coverage_cpu')
    module.prepare_native_pull_plan = prepare
    module.pull_native_flat_into = pull
    module.NativePullPlanUnavailable = NativePullPlanUnavailable
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_plan_credit_includes_only_explicit_remainder_and_outputs_are_owned(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    original = source.copy()
    budget = 3*1024**2
    plans, calls = [], []
    def prepare(v, source_shape, target_shape, *, max_plan_bytes, cache_plane):
        assert v is view and source_shape == source.shape and target_shape == shape
        assert max_plan_bytes == budget-np.prod(shape[1:])-256*1024
        assert cache_plane is True
        plan = types.SimpleNamespace(workspace_bytes=1024, persistent_bytes=512,
            temporary_strip_bytes=512, max_strip_voxels=120, backend='compiled-test')
        plans.append(plan)
        return plan
    def pull(values, plan, flat, *, first_flat, scalar_max):
        assert values is source and plan is plans[0] and scalar_max is True
        assert flat.flags.writeable and flat.flags.c_contiguous
        calls.append(first_flat)
        flat[:] = np.uint8(173)
        flat[0] = np.uint8(0)
    _fake_backend(monkeypatch, prepare, pull)
    with score_projection_reader(source, view, shape, tmp_path, memory_bytes=budget) as read:
        first, second = read(1), read(2)
        assert first.flags.owndata and second.flags.owndata
        assert not np.shares_memory(first, second)
        first[:] = 99
        assert second[0, 0] == 0 and second[-1, -1] == 173
        stats = read.projection_diagnostics()
        assert stats['backend'] == 'compiled-test' and stats['pull_calls'] == 2
        assert stats['fallback_reason'] is None
        assert stats['plan_workspace_bytes']+stats['control_bytes']+stats['output_bytes'] <= budget
        stats['plan_workspace_bytes'] = budget*2
        assert read.projection_diagnostics()['plan_workspace_bytes'] == 1024
    assert calls == [120, 240]
    np.testing.assert_array_equal(source, original)
    with pytest.raises(RuntimeError, match='closed'):
        read(0)


def test_impossible_plan_never_runs_and_advertises_exact_fallback(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    def prepare(*args, **kwargs):
        raise NativePullPlanUnavailable('plan minimum exceeds live credit')
    def pull(*args, **kwargs):
        pytest.fail('An unadmitted plan ran')
    _fake_backend(monkeypatch, prepare, pull)
    with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2) as read:
        value = read(3)
        stats = read.projection_diagnostics()
    assert stats['backend'] == 'native_numpy_sampler'
    assert stats['fallback_reason'] == 'NativePullPlanUnavailable: plan minimum exceeds live credit'
    assert stats['plan_workspace_bytes'] == 0 and value.shape == shape[1:]
    assert set(np.unique(value)).issubset({0, 173})


@pytest.mark.parametrize('fields', [dict(workspace_bytes=3*1024**2, persistent_bytes=1,
    temporary_strip_bytes=0), dict(workspace_bytes=512, persistent_bytes=400,
    temporary_strip_bytes=200), dict(workspace_bytes=512, persistent_bytes=400,
    temporary_strip_bytes=-1)])
def test_misreported_plan_charge_cannot_bypass_reader_budget(tmp_path, monkeypatch, fields):
    view, source, shape = _fixture()
    def prepare(*args, **kwargs):
        return types.SimpleNamespace(**fields, max_strip_voxels=120, backend='misreported')
    _fake_backend(monkeypatch, prepare, lambda *a, **k: pytest.fail('Invalid credit ran'))
    with pytest.raises(MemoryError, match='admitted workspace'):
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2):
            pytest.fail('Invalid charge was published')


def test_invalid_geometry_and_kernel_errors_propagate_without_fallback(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    def invalid(*args, **kwargs):
        raise ValueError('invalid native geometry')
    _fake_backend(monkeypatch, invalid, lambda *a, **k: None)
    with pytest.raises(ValueError, match='invalid native geometry'):
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2):
            pass
    def allocation_bug(*args, **kwargs):
        raise MemoryError('unexpected allocation failure')
    _fake_backend(monkeypatch, allocation_bug, lambda *a, **k: None)
    with pytest.raises(MemoryError, match='unexpected allocation failure'):
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2):
            pass
    def prepare(*args, **kwargs):
        return types.SimpleNamespace(workspace_bytes=512, persistent_bytes=512,
            temporary_strip_bytes=0, max_strip_voxels=120, backend='compiled-test')
    def failed(*args, **kwargs):
        raise OSError('kernel input read failed')
    _fake_backend(monkeypatch, prepare, failed)
    with pytest.raises(OSError, match='kernel input read failed'):
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2) as read:
            read(1)


def test_closed_reader_releases_plan_even_when_reader_is_retained(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    borrowed = []
    class Plan:
        workspace_bytes = persistent_bytes = 512
        temporary_strip_bytes = 0
        max_strip_voxels = 120
        backend = 'compiled-test'
    def prepare(*args, **kwargs):
        plan = Plan()
        borrowed.append(weakref.ref(plan))
        return plan
    _fake_backend(monkeypatch, prepare, lambda source, plan, output, **kw: output.fill(173))
    enabled = gc.isenabled()
    gc.disable()
    try:
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2) as read:
            result = read(0)
            assert borrowed[0]() is not None
        assert borrowed[0]() is None
        assert result.flags.owndata and (result == 173).all()
        with pytest.raises(RuntimeError, match='closed'):
            read(0)
    finally:
        if enabled:
            gc.enable()


def test_explicit_numpy_comparator_never_prepares_compiled_plan(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'numpy')
    _fake_backend(monkeypatch, lambda *a, **k: pytest.fail('Comparator prepared a plan'),
        lambda *a, **k: pytest.fail('Comparator ran a kernel'))
    with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2) as read:
        result = read(0)
        diagnostics = read.projection_diagnostics()
    assert result.shape == shape[1:]
    assert diagnostics['fallback_reason'] == 'explicit_numpy_backend'
    assert diagnostics['backend'] == 'native_numpy_sampler'


def test_unknown_backend_is_rejected(tmp_path, monkeypatch):
    view, source, shape = _fixture()
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'unknown')
    with pytest.raises(ValueError, match='Unsupported confidence native pull backend'):
        with score_projection_reader(source, view, shape, tmp_path, memory_bytes=1024**2):
            pass


@pytest.mark.parametrize('budget,expected_backend', [
    (1536*1024, 'compiled_azimuthal_strip'), (64*1024**2, 'compiled_azimuthal_cached'),
    (None, 'compiled_azimuthal_cached')])
def test_real_cached_and_strip_plans_preserve_exact_numeric_reader(tmp_path, budget, expected_backend):
    view = geometry.get_view_infos(3, 512, 512, cartesian_views=(), azimuthal_views=('transverse',),
        azimuthal_azimuth_angles=(45.,), azimuthal_native_raster=0)[0]
    values = np.random.default_rng(79).choice(np.array([0, 29, 173, 229], np.uint8),
        (view.num_slices, 3, 3))
    original = values.copy()
    shape = (3, 512, 512)
    old = _destination_score_reader(values, view, shape, 65536)
    expected = old(1)
    with score_projection_reader(values, view, shape, tmp_path, memory_bytes=budget) as read:
        actual = read(1)
        diagnostics = read.projection_diagnostics()
        np.testing.assert_array_equal(actual, expected)
        assert diagnostics['backend'] == expected_backend
        assert diagnostics['fallback_reason'] is None
        assert (diagnostics['plan_workspace_bytes']+diagnostics['output_bytes']
                +diagnostics['control_bytes']) <= diagnostics['budget_bytes']
        if budget is None:
            assert diagnostics['plan_credit_bytes'] == 64*1024**2
        else:
            assert diagnostics['budget_bytes'] == budget
        assert diagnostics['contribution_addresses'] > 0
        if expected_backend.endswith('_strip'):
            assert diagnostics['kernel_calls'] > 1
            assert diagnostics['max_strip_voxels'] < np.prod(shape[1:])
        actual[:] = 0
        np.testing.assert_array_equal(read(1), expected)
    np.testing.assert_array_equal(values, original)
