"""Scheduler diagnostics reveal CPU projection progress without changing credits."""
from __future__ import annotations

import ast
from concurrent.futures import Future
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from XTA import pipeline


def _finished(value=None, error=None):
    future = Future()
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
    return future


def test_summary_classifies_unsettled_prepares_without_waiting():
    queued = Future()
    running = Future()
    running.set_running_or_notify_cancel()
    child = Future()
    completed = _finished(SimpleNamespace(pending_component_layers=[_finished()]))
    publishing = _finished(SimpleNamespace(pending_component_layers=[child]))
    failed = _finished(error=RuntimeError('preserved failure'))
    cancelled = Future()
    cancelled.cancel()
    summary = pipeline._scheduler_wait_activity_summary(
        [queued, running, completed, publishing, failed, cancelled], {})
    assert summary['parent_prepares'] == dict(
        running=1, queued=1, awaiting_components=1, completed_awaiting_drain=3)
    assert summary['active_native_projections'] == 0
    assert not child.done() and not queued.done() and not running.done()
    assert failed.exception().args == ('preserved failure',)


def test_native_projection_summary_retains_oldest_live_work_and_copies_records():
    gauges = {
        f'projection.native_destination_pull.live.view{n}': {
            'state': 'running', 'view': f'view{n}', 'elapsed_seconds': n * 10.,
            'completed_planes': n, 'total_planes': 100}
        for n in range(5)
    }
    gauges.update({
        'projection.native_destination_pull.live.done': {'state': 'complete'},
        'projection.native_destination_pull.live.failed': {'state': 'failed'},
        'projection.native_destination_pull.live.invalid': None,
        'unrelated': {'state': 'running', 'elapsed_seconds': 9999},
    })
    summary = pipeline._scheduler_wait_activity_summary([], gauges)
    assert summary['active_native_projections'] == 5
    assert [record['view'] for record in summary['oldest_native_projections']] == [
        'view4', 'view3', 'view2']
    summary['oldest_native_projections'][0]['completed_planes'] = 999
    assert gauges['projection.native_destination_pull.live.view4']['completed_planes'] == 4


def _wait_logger(namespace):
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == '_log_scheduler_wait_state')
    wrapper = ast.parse('def factory():\n    last_scheduler_wait_log = 0.0\n').body[0]
    wrapper.body.extend([function, ast.Return(ast.Name('_log_scheduler_wait_state', ast.Load()))])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, str(pipeline.__file__), 'exec'), namespace)
    return namespace['factory']()


def test_actual_wait_logger_reports_credit_convoy_and_live_planes(capsys):
    telemetry = SimpleNamespace(lock=threading.RLock(), gauges={
        'projection.native_destination_pull.live.tilted': {
            'state': 'running', 'view': 'tilted', 'completed_planes': 321,
            'total_planes': 1931, 'elapsed_seconds': 500.,
            'backend': 'compiled_native_pull_cpu', 'workers': 8,
            'last_update_monotonic': 1000.,
        }}, gauge=Mock())
    running = []
    for _ in range(4):
        future = Future()
        future.set_running_or_notify_cancel()
        running.append(future)
    future_map = {future: None for future in running + [Future() for _ in range(13)]}
    state = SimpleNamespace(gpu_worker_results_collected=2243, gpu_worker_total_tasks=6111,
                            gpu_worker_tile_dense_result_bytes_reserved=0)
    namespace = dict(vars(pipeline))
    namespace.update(
        runtime_telemetry=lambda: telemetry,
        _MAIN_PROCESS_GPU_STAGE_COORDINATOR=SimpleNamespace(snapshot=lambda: {
            'spherical_retirement_pressure': True}),
        _env_float=lambda _name, default: default,
        parent_transient_admission=SimpleNamespace(condition=threading.Condition(),
                                                 in_use=88, capacity=192),
        postprocessed_tiles_waiting_by_parent={}, residual_tiles_waiting_by_parent={},
        view_processing_futures=future_map, gpu_worker_pending_task_ids=set(range(3868)),
        scheduler_state=state, direct_union_inference_bytes={},
        direct_union_postprocess_bytes={str(n): 10 for n in range(17)},
        direct_union_total_dense_byte_limit=256,
        direct_union_inference_views=set(), direct_union_postprocess_views=set(range(17)),
        gpu_worker_tile_dense_result_task_limit=4, gpu_worker_tile_dense_result_limit=100,
        gpu_worker_tile_dense_result_reservations={}, ready_fullframe=[], ready_tile_infer=[])
    for name in ('pending_prediction_volume_futures', 'pending_prediction_build_jobs',
                 'prediction_accumulation_futures', 'physical_view_finalization_futures',
                 'physical_view_union_futures', 'tile_cleanup_futures', 'tile_parent_gate_futures',
                 'tile_bridge_gate_futures', 'tile_consolidation_futures',
                 'tile_parent_finalization_futures'):
        namespace[name] = {}
    logger = _wait_logger(namespace)
    logger(force=True)
    output = capsys.readouterr().out
    assert 'parent_running=4, parent_queued=13' in output
    assert 'inference_received=2243/6111' in output
    assert 'native_projection_active=1' in output
    assert "'completed_planes': 321" in output
    key, receipt = telemetry.gauge.call_args.args
    assert key == 'scheduler.wait_activity'
    assert receipt['dense_retained_bytes'] == 170 and receipt['dense_limit_bytes'] == 256
    assert receipt['inference_pending_tasks'] == 3868
    assert receipt['oldest_native_projections'][0]['workers'] == 8
    assert len(future_map) == 17 and state.gpu_worker_results_collected == 2243
    # The regular status interval still suppresses repeated output/gauge updates.
    logger()
    assert capsys.readouterr().out == ''
    assert telemetry.gauge.call_count == 1
