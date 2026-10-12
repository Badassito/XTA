"""Actual scheduler and pipeline ownership boundaries for private canvas birth."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA.detector_workspaces import DetectorWorkspacePreparation
from XTA.geometry import ViewInfo
from XTA.interpolation import _ByteAdmissionPool
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def _pipeline_helpers(namespace):
    names = {'_detector_future_ram_rights', '_plan_baseline_workspace',
        '_discard_baseline_bundles', '_build_baseline_workspaces', '_adopt_baseline_workspaces',
        '_refresh_workspace_plans', '_rollback_workspace_plans', '_request_baseline_workspaces',
        '_ensure_baseline_workspaces'}
    tree = ast.parse(Path('XTA/pipeline.py').read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'pipeline-workspace-helpers', 'exec'), namespace)


@pytest.fixture
def workspace(tmp_path):
    import weakref
    from XTA.interpolation import _DirectUnionBackingLease
    state = _state()
    allocations, closed = [], []
    pool = _ByteAdmissionPool(1 << 24, 'test workspace promises')
    leases = ViewPrepareLeaseState(state.direct_union_backing_leases, state.direct_union_inference_views,
        state.direct_union_inference_bytes, state.direct_union_postprocess_views, state.direct_union_postprocess_bytes)
    namespace = dict(__name__='XTA.pipeline', __package__='XTA',
        np=np, Path=Path, ViewInfo=ViewInfo, weakref=weakref, threading=threading,
        _DirectUnionBackingLease=_DirectUnionBackingLease, temp_dir=tmp_path,
        args=SimpleNamespace(imgsz=8, min_conf=0, reconciliation_retain_confidence=False, interpolation_distance=0),
        dense_tiling_active=False, component_layers_needed=False, policy_settings=SimpleNamespace(enabled=False),
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
    scheduler = _scheduler(tmp_path, state=state,
        input_overrides=dict(gpu_device_count=2, v1613_d1_owner_active=True))
    namespace['scheduler'] = scheduler
    service = DetectorWorkspacePreparation(build=namespace['_build_baseline_workspaces'],
        refresh=namespace['_refresh_workspace_plans'], adopt=namespace['_adopt_baseline_workspaces'],
        discard=namespace['_discard_baseline_bundles'], rollback=namespace['_rollback_workspace_plans'],
        wake=scheduler.notify_gpu_stage_admission_change)
    namespace['detector_workspace_preparation'] = service
    scheduler.inputs = replace(scheduler.inputs,
        ensure_baseline_workspaces=namespace['_ensure_baseline_workspaces'],
        request_baseline_workspaces=namespace['_request_baseline_workspaces'],
        drain_baseline_workspaces=service.drain)
    yield SimpleNamespace(state=state, namespace=namespace, scheduler=scheduler, service=service,
        pool=pool, allocations=allocations, closed=closed)
    service.shutdown()


def _task(workspace, task_id, *, name='curved', family='spherical', mode='direct_union'):
    view = replace(_view(), name=name, family=family)
    task = dict(task_id=task_id, model_name='model', view=view, kind='fullframe',
        result_mode=mode, slice_start=0, slice_count=16, processing_shape=(16,8,8), out_size=8,
        gpu_eligible=True, disable_runtime_split=True)
    state = workspace.state
    state.gpu_worker_tasks_by_id[task_id] = task
    state.gpu_worker_pending_task_ids.append(task_id)
    state.fullframe_task_ids_by_parent[('model',name)] = [task_id]
    state.fullframe_remaining[('model',name)] = 1
    return task


def test_blocked_birth_does_not_claim_gpu_or_block_ready_d1_dispatch(workspace):
    import queue
    entered, release = threading.Event(), threading.Event()
    original = workspace.namespace['allocate_workspace_array']
    def allocate(**kwargs):
        entered.set()
        assert release.wait(3)
        return original(**kwargs)
    workspace.namespace['allocate_workspace_array'] = allocate
    curved = _task(workspace, 10)
    cheap = _task(workspace, 20, name='cheap', family='orthogonal', mode='d1_owner')
    workspace.state.gpu_task_queues.update({0:queue.Queue(), 1:queue.Queue()})
    try:
        workspace.scheduler.prepare_baseline_backlog()
        assert entered.wait(3)
        assert workspace.state.direct_union_inference_bytes[('model','curved')] == 1024
        assert workspace.state.direct_union_backing_leases[('model','curved')].phase == 'preparing'
        workspace.scheduler.dispatch_gpu_worker_inference_window()
        actual = [tasks.get_nowait() for tasks in workspace.state.gpu_task_queues.values() if not tasks.empty()]
        assert [task['task_id'] for task in actual] == [20]
        assert workspace.state.d1_owner_by_parent == {('model','cheap'):0}
        assert 10 in workspace.state.gpu_worker_pending_task_ids
        assert not workspace.namespace['baseline_union_by_model_view']
        assert workspace.scheduler.process_quiescence_issues()['preparing_workspaces'] == [('model','curved')]
        release.set()
        workspace.service.futures[0].result(timeout=3)
        assert not workspace.namespace['baseline_union_by_model_view']
        workspace.scheduler.dispatch_gpu_worker_inference_window()
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
        assert workspace.scheduler._direct_union_admission_totals()[0] == 1
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
        reserved_bytes = 2000
        def can_yield_capacity(self, amount):
            return True
        def try_yield_capacity(self, amount):
            self.reserved_bytes = amount
            return amount
    owner = Grant()
    workspace.namespace['cpu_input_prefetch'] = workspace.scheduler.cpu_input_prefetch = owner
    workspace.scheduler.inputs = replace(workspace.scheduler.inputs,
        direct_union_total_dense_byte_limit=2048, direct_union_inference_byte_limit=2048)
    _task(workspace, 10)
    _task(workspace, 20, name='d1', family='orthogonal', mode='d1_owner')
    _task(workspace, 30, name='unresolved', mode='hybrid_deferred')
    workspace.scheduler.prepare_baseline_backlog()
    assert owner.reserved_bytes == 1024
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
