"""Preparing detector inputs remain future inference without probing RAM on the owner."""
import queue as queue_module
import threading

from XTA import sam_parent_staging as staging
from tests.test_detector_workspaces import workspace, _task
from tests.test_sam_parent_staging_adversarial import queue


def test_preparing_pending_task_blocks_terminal_gpu_asset_retirement(workspace):
    entered, release = threading.Event(), threading.Event()
    original = workspace.namespace['allocate_workspace_array']
    def allocate(**kwargs):
        entered.set()
        assert release.wait(3)
        return original(**kwargs)
    workspace.namespace['allocate_workspace_array'] = allocate
    task = _task(workspace, 10)
    state = workspace.state
    state.gpu_task_queues.update({0: queue_module.Queue(), 1: queue_module.Queue()})
    state.gpu_worker_total_tasks = 1
    try:
        workspace.scheduler.prepare_baseline_backlog()
        assert entered.wait(3)
        assert not workspace.scheduler.direct_union_task_admissible(task)
        assert state.gpu_worker_pending_task_ids
        workspace.scheduler.publish_gpu_worker_admissible_backlog()
        # Isolate the pending-input guard from the result-count guard.
        state.gpu_worker_results_collected = state.gpu_worker_total_tasks
        assert not workspace.scheduler.request_gpu_inference_asset_release()
        assert not state.gpu_inference_asset_release_requested
        assert not state.gpu_inference_asset_release_pending_by_worker
        assert not workspace.scheduler.gpu_inference_asset_release_complete()
        assert all(tasks.empty() for tasks in state.gpu_task_queues.values())
        assert workspace.scheduler.process_quiescence_issues()['preparing_workspaces'] == [('model', 'curved')]
    finally:
        release.set()


def test_ram_headroom_probes_stay_off_owner_during_ready_gate_and_adoption(workspace, tmp_path, monkeypatch):
    owner = threading.get_ident()
    probes = []
    def headroom():
        probes.append(threading.get_ident())
        assert probes[-1] != owner, 'Owner thread performed a physical headroom probe'
        return staging.RAM_RESERVE_BYTES + 65536
    monkeypatch.setattr(staging, 'publication_ram_headroom', headroom)
    monkeypatch.setattr(staging, 'workspace_anon_cap_bytes', lambda: 0)
    stage, _writer, _prepare, _leases = queue(tmp_path/'sam', cap=1 << 20,
        leases=workspace.namespace['view_prepare_leases'])
    monkeypatch.setattr(stage, '_publish_staging', lambda: None)
    namespace = workspace.namespace
    namespace.update(sam_parent_staging=stage, sam_cpu_max_detector_parent=1024,
        _sam_parent_requires_staging=lambda view: True)
    task = _task(workspace, 10)
    try:
        assert workspace.scheduler.direct_union_task_admissible(task)
        assert not namespace['_request_baseline_workspaces'](task)
        workspace.service.futures[0].result(timeout=3)
        assert not workspace.scheduler.direct_union_task_admissible(task)
        workspace.service.drain()
        assert workspace.scheduler.direct_union_task_admissible(task)
        assert stage.owns_ram_first_parent(('model', 'curved'))
        assert probes and all(probe != owner for probe in probes)
    finally:
        workspace.service.shutdown()
        namespace['baseline_union_by_model_view'].clear()
        namespace['baseline_confmap_by_model_view'].clear()
        stage.release_ram_first(('model', 'curved'))
        stage.close()
        stage.finalize_cleanup()
