"""Actual scheduler and pipeline ownership boundaries for private canvas birth."""
import ast
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA.detector_workspaces import DetectorWorkspacePreparation
from XTA.geometry import ViewInfo
from XTA.interpolation import _ByteAdmissionPool
from XTA.view_prepare import ViewPrepareLeaseState
from tests.native_detector_test_support import DetectorRun, detector_inputs, detector_state, detector_view


def _pipeline_helpers(namespace):
    names = {'_detector_future_ram_rights', '_plan_baseline_workspace',
        '_discard_baseline_bundles', '_build_baseline_workspaces', '_adopt_baseline_workspaces',
        '_refresh_workspace_plans', '_rollback_workspace_plans', '_request_baseline_workspaces',
        '_ensure_baseline_workspaces', '_request_sweep_hydration'}
    tree = ast.parse(Path('XTA/pipeline.py').read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'pipeline-workspace-helpers', 'exec'), namespace)


@pytest.fixture
def workspace(tmp_path):
    import weakref
    from XTA.interpolation import _DirectUnionBackingLease
    state = detector_state()
    allocations, closed = [], []
    pool = _ByteAdmissionPool(1 << 24, 'test workspace promises')
    leases = ViewPrepareLeaseState(state.direct_union_backing_leases, state.direct_union_inference_views,
        state.direct_union_inference_bytes, state.direct_union_postprocess_views, state.direct_union_postprocess_bytes)
    namespace = dict(__name__='XTA.pipeline', __package__='XTA',
        np=np, Path=Path, ViewInfo=ViewInfo, weakref=weakref, threading=threading,
        _DirectUnionBackingLease=_DirectUnionBackingLease, temp_dir=tmp_path,
        args=SimpleNamespace(imgsz=8, min_conf=0, reconciliation_retain_confidence=False, interpolation_distance=0),
        tiling_active=False, component_layers_needed=False, policy_settings=SimpleNamespace(enabled=False),
        worker_direct_union_active=True, sam_context=None, sam_cpu_futures={'source':None},
        sam_parent_staging=None, sam_cpu_max_detector_parent=0, cpu_input_prefetch=None,
        gray_capture_reserve_bytes=0, cpu_input_pinned_reserve_bytes=0,
        input_volume_rgb=np.zeros((1,1,1), np.uint8),
        parent_transient_admission=pool, view_prepare_leases=leases,
        _sam_parent_requires_staging=lambda view:False,
        view_processing_volume_shape=lambda view,size:(view.num_slices,size,size),
        array_nbytes=lambda shape,dtype:int(np.prod(shape))*np.dtype(dtype).itemsize,
        interpolation_process_backend_enabled=lambda:False, _view_uses_interpolation=lambda *args:False,
        _memmap_backing_path=lambda value:Path('shared-test-path'), _memfd_owner_key_from_array=lambda value:None,
        path_is_memory_backed=lambda path:False, classify_dense_ram_backing=lambda *args,**kwargs:True,
        capture_memfd_owner_proofs=lambda arrays:(), scratch_unlink_path_for_memmap=lambda *args:None,
        runtime_telemetry=lambda:SimpleNamespace(add=lambda *args:None), bounded_policy_parent_keys=set(),
        baseline_union_by_model_view={}, baseline_confmap_by_model_view={}, baseline_slice_locks_by_model_view={},
        view_slice_meta={},
        baseline_union_paths=state.baseline_union_paths, baseline_confmap_paths=state.baseline_confmap_paths,
        direct_union_backing_leases=state.direct_union_backing_leases,
        direct_union_inference_views=state.direct_union_inference_views,
        direct_union_postprocess_views=state.direct_union_postprocess_views,
        direct_union_inference_bytes=state.direct_union_inference_bytes,
        scheduler_state=state)
    def allocate(*, shape, **kwargs):
        allocations.append(threading.current_thread().name)
        return np.zeros(shape, np.uint8)
    namespace['allocate_workspace_array'] = allocate
    namespace['close_memmap_array_without_flush'] = lambda value,**kwargs:closed.append(id(value))
    _pipeline_helpers(namespace)
    with DetectorRun(tmp_path, state=state, inputs=detector_inputs(
            tmp_path, gpu_device_count=2, v1613_d1_owner_active=True)) as run:
        namespace['scheduler'] = run.scheduler
        service = DetectorWorkspacePreparation(build=namespace['_build_baseline_workspaces'],
            refresh=namespace['_refresh_workspace_plans'], adopt=namespace['_adopt_baseline_workspaces'],
            discard=namespace['_discard_baseline_bundles'], rollback=namespace['_rollback_workspace_plans'],
            wake=run.scheduler.notify_gpu_stage_admission_change)
        namespace['detector_workspace_preparation'] = service
        run.scheduler.inputs = replace(run.scheduler.inputs,
            ensure_baseline_workspaces=namespace['_ensure_baseline_workspaces'],
            request_baseline_workspaces=namespace['_request_baseline_workspaces'],
            drain_baseline_workspaces=service.drain)
        try:
            yield SimpleNamespace(state=state, namespace=namespace, scheduler=run.scheduler, service=service,
                native_run=run, pool=pool, allocations=allocations, closed=closed)
        finally:
            service.shutdown()


def _task(workspace, task_id, *, name='curved', family='spherical', mode='direct_union'):
    view = detector_view(name=name, family=family)
    task = dict(task_id=task_id, model_name='model', view=view, kind='fullframe',
        result_mode=mode, slice_start=0, slice_count=16, processing_shape=(16,8,8), out_size=8,
        gpu_eligible=True, disable_runtime_split=True)
    state = workspace.state
    state.gpu_worker_tasks_by_id[task_id] = task
    state.fullframe_task_ids_by_parent[('model',name)] = [task_id]
    state.fullframe_remaining[('model',name)] = 1
    return task


def test_blocked_birth_does_not_claim_gpu_or_block_ready_d1_dispatch(workspace):
    entered, release = threading.Event(), threading.Event()
    original = workspace.namespace['allocate_workspace_array']
    def allocate(**kwargs):
        entered.set()
        assert release.wait(3)
        return original(**kwargs)
    workspace.namespace['allocate_workspace_array'] = allocate
    curved = _task(workspace, 10)
    cheap = _task(workspace, 20, name='cheap', family='orthogonal', mode='d1_owner')
    run = workspace.native_run
    actor = run.worker(0)
    run.start_transport()
    try:
        run.declare(curved, cheap)
        run.drain_until(lambda: bool(workspace.service.futures))
        assert entered.wait(3)
        assert workspace.state.direct_union_inference_bytes[('model','curved')] == 1024
        assert workspace.state.direct_union_backing_leases[('model','curved')].phase == 'preparing'
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        assert run.take(0)['task_id'] == 20
        assert run.native.snapshot()['active'] == 1
        assert not workspace.namespace['baseline_union_by_model_view']
        actor.installed(20)
        actor.compute_done(20)
        actor.put(dict(type='result', task_id=20, gpu_index=0, ok=True, stats={}))
        release.set()
        workspace.service.futures[0].result(timeout=3)
        assert not workspace.namespace['baseline_union_by_model_view']
        run.drain_until(lambda: ('model','curved') in workspace.namespace['baseline_union_by_model_view'])
        assert run.take(0)['task_id'] == 10
        assert ('model','curved') in workspace.namespace['baseline_union_by_model_view']
        assert workspace.state.direct_union_backing_leases[('model','curved')].phase == 'inference'
        assert workspace.allocations == ['detector-workspace-birth']
    finally:
        release.set()


def test_policy_group_is_funded_before_birth_and_published_atomically(workspace):
    entered, release = threading.Event(), threading.Event()
    original = workspace.namespace['allocate_workspace_array']
    def allocate(**kwargs):
        entered.set()
        assert release.wait(3)
        return original(**kwargs)
    workspace.namespace['allocate_workspace_array'] = allocate
    first = _task(workspace, 10, name='pass0')
    second = _task(workspace, 11, name='pass1')
    first.update(bounded_parent_admission=True, augmentation_pass_tasks=(second,))
    second['bounded_parent_admission'] = True
    try:
        assert not workspace.namespace['_request_baseline_workspaces'](first)
        assert entered.wait(3)
        assert sum(workspace.state.direct_union_inference_bytes.values()) == 2048
        assert set(workspace.state.direct_union_admission_group_by_parent.values()) == {('model','pass0')}
        assert len(workspace.service.records) == 1
        assert not workspace.namespace['baseline_union_by_model_view']
        release.set()
        workspace.service.futures[0].result(timeout=3)
        workspace.service.drain()
        assert set(workspace.namespace['baseline_union_by_model_view']) == {('model','pass0'),('model','pass1')}
        assert not workspace.service.pending
    finally:
        release.set()


def test_planning_failure_returns_only_this_unpublished_groups_credit(workspace):
    task = _task(workspace, 10)
    workspace.namespace['_plan_baseline_workspace'] = lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('bad plan'))
    with pytest.raises(ValueError, match='bad plan'):
        workspace.namespace['_request_baseline_workspaces'](task)
    assert not workspace.state.direct_union_backing_leases
    assert not workspace.state.direct_union_inference_bytes
    assert not workspace.state.direct_union_admission_group_by_parent
    assert not workspace.service.pending


@pytest.mark.parametrize('state', ['closed', 'failed_hydration', 'success'])
def test_sweep_hydration_overlaps_within_dense_credit_and_settles_ownership(workspace, monkeypatch, state):
    from XTA import sweep_views
    from XTA.interpolation import _DirectUnionBackingLease

    run = workspace.native_run
    run.scheduler.inputs = replace(run.scheduler.inputs, direct_union_total_dense_byte_limit=1700,
                                   direct_union_inference_byte_limit=1700)
    run.start_transport()
    run.declare()
    view = detector_view()
    key = ('model', view.name)
    busy = ('model', 'previous')
    postprocess = workspace.state.direct_union_postprocess_bytes
    postprocess[busy] = 700
    busy_lease = _DirectUnionBackingLease(busy, 700, phase='postprocess')
    workspace.state.direct_union_backing_leases[busy] = busy_lease
    workspace.state.direct_union_postprocess_views.add(busy)
    run.drain_until(lambda: run.scheduler.request_host_workspace(busy, 700))
    run.scheduler.host_workspace_adopted(busy)
    workspace.namespace.update(os=os, model_name='model', sweep_cutoff=50,
        sweep_fanout_plan=SimpleNamespace(cutoffs=(50, 60), identities={view.name: 'sweep-id'}),
        direct_union_postprocess_bytes=postprocess, direct_union_total_dense_byte_limit=1700,
        direct_union_inference_view_limit=2)
    request = workspace.namespace['_request_sweep_hydration']

    def hydrate(model, name, cutoff, union, scores, *, identity, stop):
        assert (model, name, cutoff, identity) == ('model', view.name, 50, 'sweep-id')
        assert threading.current_thread().name == 'detector-workspace-birth'
        assert np.all(union == 0) and scores is None and not stop.is_set()
        if state == 'failed_hydration':
            raise OSError('sweep read failed')
        union[0, 2, 3] = 1
        return dict(prediction_count=9, slice_any=np.any(union, axis=(1, 2)), slice_bboxes=None)
    monkeypatch.setattr(sweep_views, 'hydrate', hydrate)

    request(view, key)
    assert not workspace.service.pending and not workspace.state.direct_union_inference_bytes
    assert run.native.snapshot()['blocked_parents'] == 1
    credits = workspace.namespace['view_prepare_leases']
    assert credits.retire_input_bytes(busy, busy_lease, 24, token='confidence')

    def request_hydration():
        request(view, key)
        return workspace.service.pending

    if state == 'closed':
        workspace.service.shutdown()
        with pytest.raises(RuntimeError, match='preparation is closed'):
            run.drain_until(request_hydration)
    else:
        run.drain_until(request_hydration)
        assert workspace.state.direct_union_inference_bytes == {key: 1024}
        future = workspace.service.futures[0]
        if state == 'failed_hydration':
            with pytest.raises(RuntimeError, match='sweep read failed'):
                future.result(timeout=3)
            with pytest.raises(RuntimeError, match='sweep read failed'):
                workspace.service.drain()
        else:
            future.result(timeout=3)
            assert key not in workspace.namespace['baseline_union_by_model_view']
            workspace.service.drain()
            assert workspace.namespace['baseline_union_by_model_view'][key][0, 2, 3] == 1
            assert workspace.namespace['view_slice_meta'][key]['prediction_count'] == 9
            assert workspace.state.direct_union_backing_leases[key].phase == 'inference'
    if state != 'success':
        assert workspace.state.direct_union_backing_leases == {busy: busy_lease}
        assert not workspace.state.direct_union_inference_bytes
    assert postprocess == {busy: 676}
    assert not workspace.service.pending


def test_partial_group_birth_failure_closes_arrays_before_refunding(workspace):
    first = _task(workspace, 10, name='pass0')
    second = _task(workspace, 11, name='pass1')
    first.update(bounded_parent_admission=True, augmentation_pass_tasks=(second,))
    second['bounded_parent_admission'] = True
    original = workspace.namespace['allocate_workspace_array']
    def allocate(**kwargs):
        if workspace.allocations:
            raise OSError('second canvas failed')
        return original(**kwargs)
    workspace.namespace['allocate_workspace_array'] = allocate
    workspace.namespace['_request_baseline_workspaces'](first)
    with pytest.raises(RuntimeError, match='second canvas failed'):
        workspace.service.futures[0].result(timeout=3)
    with pytest.raises(RuntimeError, match='second canvas failed'):
        workspace.service.drain()
    assert len(workspace.closed) == 1
    assert not workspace.state.direct_union_backing_leases
    assert not workspace.namespace['baseline_union_by_model_view']


def test_adoption_validates_every_member_before_any_publication(workspace):
    first = _task(workspace, 10, name='pass0')
    second = _task(workspace, 11, name='pass1')
    first.update(bounded_parent_admission=True, augmentation_pass_tasks=(second,))
    second['bounded_parent_admission'] = True
    workspace.namespace['_request_baseline_workspaces'](first)
    record = next(iter(workspace.service.records.values()))
    bundles = record.future.result(timeout=3)
    adopt = workspace.namespace['_adopt_baseline_workspaces']
    with pytest.raises(RuntimeError, match='incomplete group'):
        adopt(record.plans, bundles[:1])
    workspace.state.direct_union_backing_leases[('model','pass1')].owner_count = 0
    with pytest.raises(RuntimeError, match='lost its preparing reservation'):
        adopt(record.plans, bundles)
    assert not workspace.namespace['baseline_union_by_model_view']
    assert workspace.state.direct_union_backing_leases[('model','pass0')].phase == 'preparing'
    workspace.state.direct_union_backing_leases[('model','pass1')].owner_count = 1


def test_request_metadata_does_not_wait_on_the_parent_physical_probe_condition(workspace):
    held, release, returned = threading.Event(), threading.Event(), threading.Event()
    def hold():
        with workspace.pool.condition:
            held.set()
            release.wait(3)
    holder = threading.Thread(target=hold)
    holder.start()
    task = _task(workspace, 10)
    caller = threading.Thread(target=lambda:(workspace.namespace['_request_baseline_workspaces'](task), returned.set()))
    try:
        assert held.wait(3)
        caller.start()
        assert returned.wait(.5)
    finally:
        release.set()
        holder.join(3)
        caller.join(3)


def test_shutdown_keeps_aliased_private_bytes_and_settles_them_on_repeat(workspace):
    import gc
    task = _task(workspace, 10)
    workspace.namespace['_request_baseline_workspaces'](task)
    record = next(iter(workspace.service.records.values()))
    bundles = record.future.result(timeout=3)
    alias = bundles[0]['union_mm']
    workspace.service.shutdown()
    assert workspace.service.pending
    assert workspace.state.direct_union_inference_bytes[('model','curved')] == 1024
    assert len(workspace.closed) == 1
    del alias
    gc.collect()
    workspace.service.shutdown()
    assert not workspace.service.pending
    assert not workspace.state.direct_union_backing_leases
    assert len(workspace.closed) == 1


def test_failed_adoption_leaves_private_bundle_for_background_shutdown_cleanup(workspace):
    task = _task(workspace, 10)
    workspace.namespace['_request_baseline_workspaces'](task)
    workspace.service.futures[0].result(timeout=3)
    workspace.service.adopt = lambda *args:(_ for _ in ()).throw(RuntimeError('adopter failed'))
    with pytest.raises(RuntimeError, match='adopter failed'):
        workspace.service.drain()
    assert workspace.service.pending and not workspace.closed
    assert workspace.state.direct_union_inference_bytes[('model','curved')] == 1024
    workspace.service.shutdown()
    assert not workspace.service.pending
    assert len(workspace.closed) == 1
    assert not workspace.state.direct_union_backing_leases


def test_refill_grant_is_yielded_before_initial_workspace_birth(workspace):
    class Grant:
        max_bytes = 2000
        reserved_bytes = 2000

        def try_yield_capacity(self, amount):
            self.reserved_bytes = amount
            return amount

        def attach_ready(self, _task):
            return None

    owner = Grant()
    workspace.namespace['cpu_input_prefetch'] = workspace.scheduler.cpu_input_prefetch = owner
    workspace.scheduler.inputs = replace(workspace.scheduler.inputs,
        direct_union_total_dense_byte_limit=2048, direct_union_inference_byte_limit=2048)
    curved = _task(workspace, 10)
    d1 = _task(workspace, 20, name='d1', family='orthogonal', mode='d1_owner')
    run = workspace.native_run
    run.worker(0)
    run.start_transport()
    run.declare(curved, d1)
    run.drain_until(lambda: ('model', 'curved') in workspace.state.direct_union_inference_bytes)
    assert owner.reserved_bytes == 1024
    assert run.native.snapshot()['byte_capacity'] == 1024
    assert set(workspace.state.direct_union_inference_bytes) == {('model','curved')}
    assert workspace.state.direct_union_inference_bytes[('model','curved')] + owner.reserved_bytes == 2048


def test_unmaterialized_native_gray_capture_and_pinned_rights_are_in_physical_plan(workspace, monkeypatch):
    from XTA import media
    ready = threading.Event()
    monkeypatch.setattr(media, 'volume_readiness', lambda _source:SimpleNamespace(_all_event=ready))
    workspace.namespace['input_volume_rgb'] = SimpleNamespace(nbytes=100)
    workspace.namespace['gray_capture_reserve_bytes'] = 200
    workspace.namespace['cpu_input_pinned_reserve_bytes'] = 50
    amount, basis = workspace.namespace['_detector_future_ram_rights']()
    assert amount == 350
    ready.set()
    new_amount, new_basis = workspace.namespace['_detector_future_ram_rights']()
    assert new_amount == 250 and new_basis != basis


def test_stale_external_promises_requalify_without_reallocating_or_publishing(workspace):
    task = _task(workspace, 10)
    workspace.namespace['_request_baseline_workspaces'](task)
    workspace.service.futures[0].result(timeout=3)
    workspace.namespace['gray_capture_reserve_bytes'] = 200
    workspace.service.drain()
    assert workspace.service.pending
    assert not workspace.namespace['baseline_union_by_model_view']
    assert workspace.state.direct_union_inference_bytes[('model','curved')] == 1024
    workspace.service.futures[0].result(timeout=3)
    workspace.service.drain()
    assert not workspace.service.pending
    assert len(workspace.allocations) == 1
