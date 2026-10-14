"""Funded CPU geometry overlaps model boot; projection waits for the cohort."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import gc
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import assembly, confidence_evidence, pipeline, sam_bridge_planning as planning
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_integration import SamConcurrentStartupResourceError
from XTA.runtime import wait_for_retired_memmap_unlinks
from tests.test_sam_progressive_startup import boot
from tests.test_terminal_component_refs import _function
from tests.test_view_prepare_input_retirement import _task


GIB = 1024**3


def parents_ready(context, future):
    namespace = dict(sam_context=context,
        sam_cpu_futures={'runtime': future, 'runtime_needed': True})
    return _function(Path(pipeline.__file__).read_text(), '_sam_parents_ready', namespace)()


def start_blocked(boot):
    boot.parent = _ByteAdmissionPool(64*GIB, 'SAM CPU prefix')
    started, admitted = threading.Event(), threading.Event()
    boot.blocks[0] = started, admitted
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    context.detector_device_assets_retired(0)
    executor = ThreadPoolExecutor(max_workers=2)
    future = executor.submit(context.prepare_runtime, boot.parent)
    assert started.wait(3.)
    assert context.cpu_prepare_ready and not context.runtime_ready
    assert parents_ready(context, future) and not future.done()
    return context, executor, future, admitted


def prefix_task(boot, context, monkeypatch, *, family='orthogonal', distance=3,
                extrapolation=1, multilabel=False):
    task, _expected = _task(boot.root/'parent', keep=True)
    task.union_mm[:] = 0
    task.union_mm[0, 1:3, 1:4] = task.union_mm[2, 1:3, 1:4] = 1
    if multilabel:
        task.union_mm[0, 1:3, 1] = 3
        task.union_mm[0, 1:3, 2:4] = 7
    expected = np.array(task.union_mm)
    task.view = replace(task.view, family=family,
        name=('tilted_transverse' if family == 'tilted' else family+'_transverse'))
    task.admission = boot.parent
    task.transient_bytes = task.sam_base_allowance_bytes = 4*GIB
    task.sam_context = context
    task.interpolation_backend = 'sam'
    task.interpolation_distance = distance
    task.extrapolation_distance = extrapolation
    task.prepare = assembly.prepare_view_volume_after_fullframe
    task.component_ref_dense_retirement_active = False
    warmed, projected = threading.Event(), threading.Event()
    owners, snapshots = [], []
    original = planning._observations

    def observe(*args, **kwargs):
        inventory = kwargs.get('inventory')
        if not context.runtime_ready and inventory is not None:
            profile = context._resource_local.profile
            assert profile.metadata()['base_non_cpu_allowance_bytes'] >= 2*GIB
            assert context._startup_fleet_funded and context._startup_fleet_credit_bytes > 0
            result = original(*args, **kwargs)
            owners.append(inventory)
            warmed.set()
            return result
        return original(*args, **kwargs)

    def projection(*args, **kwargs):
        assert context.runtime_ready
        projected.set()
        return None

    def capture(*args, **kwargs):
        assert context.runtime_ready
        return None

    def inspect_plan(mask, scope, inventory, *, wrap=False):
        options = dict(scope_id=scope, interpolation_distance=3,
            interpolation_min_radius=0, interpolation_search_angle=15,
            lazy_contracts=True, wrap_axis=wrap)
        actual = planning.plan_sam_bridges(mask, _observation_inventory=inventory, **options)
        expected_plan = planning.plan_sam_bridges(mask, **options)
        assert actual == expected_plan
        assert [x.observation_id for x in actual.observations] == [
            x.observation_id for x in expected_plan.observations]
        for left, right in zip(actual.observations, expected_plan.observations):
            np.testing.assert_array_equal(left.mask_crop, right.mask_crop)
        snapshots.append(inventory.snapshot() if inventory is not None else {})

    def interpolate(mask, *, scope, _observation_inventory=None, **kwargs):
        assert context.runtime_ready
        inspect_plan(mask, scope, _observation_inventory, wrap=family=='azimuthal')
        return mask, {'skipped': True}, []

    def extrapolate(mask, *, distance, _observation_inventory=None, **kwargs):
        if distance:
            assert context.runtime_ready
            inspect_plan(mask, 'post-interpolation', _observation_inventory)
        return {}, []

    monkeypatch.setattr(planning, '_observations', observe)
    monkeypatch.setattr(assembly, '_view_uses_interpolation', lambda _view, value: value > 0)
    monkeypatch.setattr(assembly, 'materialize_nrrd_view_layer', projection)
    monkeypatch.setattr(confidence_evidence, 'capture_prediction_confidence', capture)
    monkeypatch.setattr(context, 'interpolate', interpolate)
    monkeypatch.setattr(assembly, '_run_sam_extrapolation', extrapolate)
    return task, expected, warmed, projected, owners, snapshots


@pytest.mark.parametrize('family', ['orthogonal', 'tilted', 'radial', 'spherical', 'azimuthal'])
@pytest.mark.parametrize('distance,extrapolation', [(3, 1), (3, 0), (0, 1)])
def test_production_cpu_prefix_reuses_exact_geometry_before_cohort_ready(
        boot, monkeypatch, family, distance, extrapolation):
    context, executor, startup, admitted = start_blocked(boot)
    try:
        task, expected, warmed, projected, owners, snapshots = prefix_task(
            boot, context, monkeypatch, family=family, distance=distance,
            extrapolation=extrapolation)
        result = executor.submit(task)
        assert warmed.wait(3.)
        assert not projected.is_set() and not result.done() and not startup.done()
        assert owners[0].snapshot()['retained_bytes'] > 0
        assert boot.parent.in_use > context._startup_fleet_credit_bytes
        admitted.set()
        startup.result(timeout=3.)
        output = result.result(timeout=3.)
        np.testing.assert_array_equal(output.final_view_volume_mm, expected)
        assert projected.is_set() and owners[0].snapshot()['retained_bytes'] == 0
        if distance or extrapolation:
            assert snapshots and snapshots[0]['reused_frames'] > 0
    finally:
        admitted.set()
        executor.shutdown(wait=True)
        context.close()
    assert boot.parent.in_use == 0


@pytest.mark.parametrize('failure', ['cancel', 'worker_failure', 'observation_error'])
def test_startup_failure_retires_prefix_inventory_and_never_projects(boot, monkeypatch, failure):
    context, executor, startup, admitted = start_blocked(boot)
    try:
        task, _expected, warmed, projected, owners, _snapshots = prefix_task(boot, context, monkeypatch)
        task.keep_temp_artifacts = False
        task.confmap_path = task.temp_dir/'scores.dat'
        task.confmap_mm = np.memmap(task.confmap_path, mode='w+', dtype=np.uint8,
                                  shape=task.union_mm.shape)
        task.confmap_mm[:] = 255
        mapping = weakref.ref(task.confmap_mm._mmap)
        retired = []
        task.confidence_retired_callback = lambda *values:retired.append(values)
        if failure == 'observation_error':
            original = planning._observations
            def broken(*args, **kwargs):
                original(*args, **kwargs)
                raise ValueError('original observation data error')
            monkeypatch.setattr(planning, '_observations', broken)
        result = executor.submit(task)
        assert warmed.wait(3.)
        if failure == 'cancel':
            context.cancel('cancel while CPU prefix owns geometry')
        elif failure == 'worker_failure':
            boot.failures[0, 2] = lambda: ValueError('model startup failed')
        admitted.set()
        with pytest.raises((ValueError, RuntimeError)):
            result.result(timeout=3.)
        assert not projected.is_set() and owners[0].snapshot()['retained_bytes'] == 0
        gc.collect()
        assert mapping() is None and task.confmap_mm is None
        assert retired == [('model', task.view.name, int(np.prod(task.union_mm.shape)))]
        wait_for_retired_memmap_unlinks(path=task.confmap_path)
        assert not task.confmap_path.exists()
        try:
            startup.result(timeout=3.)
        except (ValueError, RuntimeError):
            pass
    finally:
        admitted.set()
        executor.shutdown(wait=True)
        context.close()
    assert boot.parent.in_use == 0


def test_multilabel_prestage_keeps_normal_label_partition(boot, monkeypatch):
    context, executor, startup, admitted = start_blocked(boot)
    try:
        task, expected, warmed, _projected, owners, snapshots = prefix_task(
            boot, context, monkeypatch, family='azimuthal', multilabel=True)
        result = executor.submit(task)
        assert warmed.wait(3.)
        admitted.set()
        startup.result(timeout=3.)
        output = result.result(timeout=3.)
        np.testing.assert_array_equal(output.final_view_volume_mm, expected)
        assert snapshots[0]['label_calls'] > 3
        assert owners[0].snapshot()['retained_bytes'] == 0
    finally:
        admitted.set()
        executor.shutdown(wait=True)
        context.close()


def test_sdf_interpolation_skips_prestage_scan_but_keeps_startup_barrier(boot, monkeypatch):
    context, executor, startup, admitted = start_blocked(boot)
    try:
        task, expected, warmed, projected, owners, _snapshots = prefix_task(boot, context, monkeypatch)
        task.interpolation_backend = 'sdf'
        task.interpolation_walk_back = 0
        reached_barrier = threading.Event()
        start = context._start
        def wait_for_runtime():
            reached_barrier.set()
            return start()
        monkeypatch.setattr(context, '_start', wait_for_runtime)
        monkeypatch.setattr(assembly, 'interpolate_view_volume_pass_maybe_process',
            lambda *, mask_mm, **kwargs:(mask_mm, {'skipped': True}))
        result = executor.submit(task)
        assert reached_barrier.wait(3.)
        assert not warmed.is_set() and not owners and not projected.is_set()
        assert not result.done() and not startup.done()
        admitted.set()
        startup.result(timeout=3.)
        output = result.result(timeout=3.)
        np.testing.assert_array_equal(output.final_view_volume_mm, expected)
        assert projected.is_set()
    finally:
        admitted.set()
        executor.shutdown(wait=True)
        context.close()
    assert boot.parent.in_use == 0


@pytest.mark.parametrize('family', ['orthogonal', 'tilted', 'radial', 'spherical', 'azimuthal'])
def test_no_sam_work_keeps_original_unwrapped_prepare(boot, monkeypatch, family):
    context, executor, startup, admitted = start_blocked(boot)
    try:
        task, expected, warmed, _projected, owners, snapshots = prefix_task(
            boot, context, monkeypatch, family=family, distance=0, extrapolation=0)
        monkeypatch.setattr(context, '_start', lambda:pytest.fail('No SAM work must not start SAM'))
        def cpu_only(*args, **kwargs):
            assert not context.runtime_ready
            assert getattr(context._resource_local, 'profile', None) is None
        monkeypatch.setattr(confidence_evidence, 'capture_prediction_confidence', cpu_only)
        monkeypatch.setattr(assembly, 'materialize_nrrd_view_layer', cpu_only)
        output = task()
        np.testing.assert_array_equal(output.final_view_volume_mm, expected)
        assert not warmed.is_set() and not owners and not snapshots
        assert not startup.done()
    finally:
        admitted.set()
        executor.shutdown(wait=True)
        context.close()


def test_cpu_parent_gate_requires_retirement_and_whole_fleet_funding(boot):
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    future = Future()
    try:
        assert not context.cpu_prepare_ready and not parents_ready(context, future)
        context._starting_runtime = SimpleNamespace()
        context._retired_devices.add(0)
        assert not context.cpu_prepare_ready and not parents_ready(context, future)
        context._starting_runtime = None
    finally:
        context.close()


def test_cold_host_refusal_never_opens_cpu_parent_gate(boot):
    boot.parent = _ByteAdmissionPool(64*GIB, 'unfunded SAM CPU prefix')
    boot.headroom[0] = GIB
    context = boot.make()
    context.configure_startup_parent_pool(boot.parent)
    context.detector_device_assets_retired(0)
    try:
        with pytest.raises(SamConcurrentStartupResourceError, match='cold host budget'):
            context.prepare_runtime(boot.parent)
        assert not context.cpu_prepare_ready
        assert not context._startup_fleet_funded
        assert not boot.trace
    finally:
        context.close()
    assert boot.parent.in_use == 0
