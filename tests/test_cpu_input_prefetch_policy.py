"""Pipeline lookahead limits are per GPU and do not resize the render executor."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA import pipeline, yolo_cpu_input


@pytest.fixture(autouse=True)
def clear_input_overrides(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_CPU_INPUT_BUILDERS', raising=False)
    monkeypatch.delenv('YOLO_TTA_CPU_INPUT_PACKETS', raising=False)


@pytest.mark.parametrize('gpus,budget,builders,packets', [(1,128,1,2), (4,128,4,8),
                                                       (8,3,3,16), (0,1,1,2)])
def test_per_gpu_defaults_respect_existing_worker_budget(gpus, budget, builders, packets):
    assert pipeline._cpu_input_prefetch_policy(gpus, budget) == dict(
        max_builders=builders, max_packets=packets)


def test_positive_explicit_overrides_are_preserved_except_builder_budget(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_BUILDERS', '19')
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_PACKETS', '3')
    assert pipeline._cpu_input_prefetch_policy(4, 7) == dict(max_builders=7, max_packets=3)
    assert pipeline._cpu_input_prefetch_policy(4, 32) == dict(max_builders=19, max_packets=3)


def test_runtime_receipt_records_policy_and_explicit_requests(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_BUILDERS', '7')
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_PACKETS', '11')
    receipt = pipeline._execution_runtime_provenance()
    assert receipt['cpu_input_prefetch_policy_requests'] == dict(
        default_builders_per_gpu=1, default_packets_per_gpu=2,
        builders_override='7', packets_override='11')


@pytest.mark.parametrize('name', ['YOLO_TTA_CPU_INPUT_BUILDERS', 'YOLO_TTA_CPU_INPUT_PACKETS'])
@pytest.mark.parametrize('value', ['0', '-1', 'invalid', '1.5', ''])
def test_invalid_overrides_are_reported(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name + ' must be a positive integer'):
        pipeline._cpu_input_prefetch_policy(4, 128)


def test_actual_pipeline_constructor_passes_limits_and_keeps_shared_executor(monkeypatch):
    captured = {}
    owner = SimpleNamespace()
    def construct(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return owner
    monkeypatch.setattr(yolo_cpu_input, 'CpuInputPrefetch', construct)
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_BUILDERS', '20')
    render_executor, volume_executor = object(), object()
    tracked, gauges = [], []
    notify = lambda: None
    source = object()
    namespace = dict(vars(pipeline), cpu_input_prefetch_requested=True,
        gpu_task_queues={i: object() for i in range(4)}, worker_budget=3,
        scheduler=SimpleNamespace(notify_gpu_stage_admission_change=notify),
        gpu_worker_tasks_by_id={}, input_volume_rgb=source, volume_rgb=object(),
        native_resize_task_spec={}, prediction_render_executor=render_executor,
        prediction_volume_executor=volume_executor, cpu_input_budget=64 * 1024**3,
        gpu_device_count=4, args=SimpleNamespace(imgsz=2048),
        available_anon_work_bytes=lambda:64*1024**3,
        runtime_telemetry=lambda:SimpleNamespace(gauge=lambda *args:gauges.append(args)),
        _run_resources=lambda:SimpleNamespace(track_executor=tracked.append))
    tree = ast.parse(Path(pipeline.__file__).read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == 'cpu_input_prefetch_requested')
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<pipeline CPU input>', 'exec'), namespace)
    assert captured['args'][1:3] == (source, render_executor)
    assert captured['kwargs']['max_builders'] == 3
    assert captured['kwargs']['max_packets'] == 8
    assert captured['kwargs']['on_capacity_change'] is notify
    assert tracked == [owner]
    assert gauges == [('yolo.cpu_input.policy', dict(max_builders=3, max_packets=8))]
    assert namespace['scheduler'].cpu_input_prefetch is owner
