"""Preparing detector inputs remain future inference without probing RAM on the owner."""
import select
import threading

from XTA import sam_parent_staging as staging
from tests.native_scheduler_cases import finish_actor
from tests.native_stage_test_support import native_coordinator
from tests.test_detector_workspaces import _task, workspace
from tests.test_sam_parent_staging_adversarial import queue


def test_preparing_pending_task_blocks_terminal_gpu_asset_retirement(workspace, request, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = workspace.namespace['allocate_workspace_array']

    def allocate(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setitem(workspace.namespace, 'allocate_workspace_array', allocate)
    task = _task(workspace, 10)
    state = workspace.state
    run = workspace.native_run
    coordinator = native_coordinator(request, monkeypatch, owner=run.native)
    coordinator.configure_workers((0, 1))
    actor = run.worker(0)
    other_actor = run.worker(1)
    run.start_transport()
    try:
        run.declare(task)
        run.drain_until(lambda: bool(workspace.service.futures))
        assert entered.wait(5)
        snapshot = run.native.snapshot()
        assert snapshot['preparing_parents'] == 1
        assert snapshot['ready'] == snapshot['active'] == snapshot['dispatched'] == 0
        assert state.direct_union_backing_leases[('model', 'curved')].phase == 'preparing'
        assert workspace.scheduler.process_inference_outstanding()
        assert workspace.scheduler.process_quiescence_issues()['outstanding_tasks'] == 1
        assert not workspace.scheduler.request_gpu_inference_asset_release()
        assert not state.gpu_inference_asset_release_requested
        assert not state.gpu_inference_asset_release_pending_by_worker
        assert not select.select([actor.stream, other_actor.stream], [], [], 0)[0]

        release.set()
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        issued = run.take(0)
        assert issued['task_id'] == task['task_id']
        finish_actor(actor, issued)
        run.drain_until(lambda: len(run.results) == 1)
        assert not workspace.scheduler.process_inference_outstanding()
        coordinator.set_inference_asset_retirement_pending(True)
        assert not workspace.scheduler.request_gpu_inference_asset_release()
        assert not workspace.scheduler.gpu_inference_asset_release_complete()
        for index, worker in enumerate((actor, other_actor)):
            command = run.take(index)
            assert command['op'] == 'release_inference_assets'
            assert command['inference_drained'] is True
            worker.put(dict(type='inference_assets_released', op=command['op'],
                gpu_index=index, task_id=command['task_id'], ok=True, stats={'released': True}))
        run.drain_until(workspace.scheduler.gpu_inference_asset_release_complete)
        assert workspace.scheduler.request_gpu_inference_asset_release()
    finally:
        release.set()


def test_ram_headroom_probes_stay_off_owner_during_ready_gate_and_adoption(
        workspace, tmp_path, monkeypatch):
    owner = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    probes = []

    def headroom():
        probes.append(threading.get_ident())
        assert probes[-1] != owner, 'Owner thread performed a physical headroom probe'
        entered.set()
        assert release.wait(5)
        return staging.RAM_RESERVE_BYTES + 65536

    monkeypatch.setattr(staging, 'publication_ram_headroom', headroom)
    monkeypatch.setattr(staging, 'workspace_anon_cap_bytes', lambda: 0)
    stage, _writer, _prepare, _leases = queue(tmp_path / 'sam', cap=1 << 20,
        leases=workspace.namespace['view_prepare_leases'])
    monkeypatch.setattr(stage, '_publish_staging', lambda: None)
    namespace = workspace.namespace
    namespace.update(sam_parent_staging=stage, sam_cpu_max_detector_parent=1024,
        _sam_parent_requires_staging=lambda view: True)
    task = _task(workspace, 10)
    run = workspace.native_run
    actor = run.worker(0)
    run.start_transport()
    key = ('model', 'curved')
    try:
        run.declare(task)
        run.drain_until(lambda: bool(workspace.service.futures))
        assert entered.wait(5)
        snapshot = run.native.snapshot()
        assert snapshot['preparing_parents'] == 1
        assert snapshot['ready'] == snapshot['dispatched'] == 0
        assert not stage.owns_ram_first_parent(key)
        assert key not in namespace['baseline_union_by_model_view']

        birth = workspace.service.futures[0]
        release.set()
        birth.result(timeout=5)
        assert not stage.owns_ram_first_parent(key)
        probe_count = len(probes)
        run.drain_until(lambda: run.native.snapshot()['dispatched'] == 1)
        issued = run.take(0)
        assert issued['task_id'] == task['task_id']
        assert stage.owns_ram_first_parent(key)
        assert key in namespace['baseline_union_by_model_view']
        assert probe_count > 0 and len(probes) == probe_count
        finish_actor(actor, issued)
        run.drain_until(lambda: len(run.results) == 1)
    finally:
        release.set()
        workspace.service.shutdown()
        namespace['baseline_union_by_model_view'].clear()
        namespace['baseline_confmap_by_model_view'].clear()
        stage.release_ram_first(key)
        stage.close()
        stage.finalize_cleanup()
