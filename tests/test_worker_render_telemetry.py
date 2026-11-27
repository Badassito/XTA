"""Render-route receipts report the final task and count only published work."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import workers
from XTA.inference import PredictConfig
from tests import test_worker_raster_plan_provenance as raster_tests


class ResidentSource:
    def __init__(self, _engine, _view, _job, *, num_frames, **_options):
        self.num_frames = num_frames
        self.closed = False

    def close(self):
        self.closed = True


class PreparedSource(ResidentSource):
    def finish_cpu_copies(self):
        self.copy_bytes = self.num_frames * 4
        self.copy_h2d_seconds = self.copy_submit_seconds = self.copy_wait_seconds = 0.0


class StreamingSource:
    def __init__(self, render, *, num_frames, **_options):
        self.render = render
        self.num_frames = num_frames
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "case,path,recovered",
    [
        ("resident", "gpu_resident", False),
        ("prepared", "cpu_prepared_trt_input", False),
        ("slab", "gpu_slab_then_cpu_affine", False),
        ("no_residency", "cpu", False),
        ("setup_failure", "cpu", True),
        ("lazy_retry", "cpu", True),
    ],
)
def test_gpu_worker_receipt_records_final_source(case, path, recovered, tmp_path, monkeypatch):
    task, _ = raster_tests.WorkerRasterPlanProvenanceTests._task(tmp_path, with_image_sink=False)
    task["slice_count"] = np.int64(1)  # One task frame from a larger view.
    cfg = PredictConfig(imgsz=2, conf=0.25, device="cuda:0", quantize=None, batch=1, input_channels=1)
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    cpu_frame = np.full((2, 2), 23, dtype=np.uint8)
    slab_frame = np.full((2, 2), 47, dtype=np.uint8)
    engine = SimpleNamespace(
        ensure_volume=Mock(return_value="streaming" if case in {"slab", "no_residency"} else "resident"),
        prerender_azimuthal_slab=Mock(return_value=cube),
        disable_resident_after_runtime_failure=Mock(),
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock())),
    )
    if case == "setup_failure":
        engine.ensure_volume.side_effect = RuntimeError("GPU setup failed")
    if case == "prepared":
        prepared_path = tmp_path / "prepared.u8"
        prepared_path.write_bytes(cube.tobytes())
        task.update(cpu_input_path=str(prepared_path), cpu_input_shape=cube.shape)

    seen = []

    def predict(_model, source, *, num_frames, **_options):
        assert num_frames == source.num_frames == 1
        seen.append(source)
        if case == "lazy_retry" and isinstance(source, ResidentSource):
            raise RuntimeError("GPU source failed lazily")
        if isinstance(source, StreamingSource):
            np.testing.assert_array_equal(source.render(0), slab_frame if case == "slab" else cpu_frame)
        return {}

    monkeypatch.delenv("XTA_YOLO_SWEEP_NATIVE_ROLE", raising=False)
    monkeypatch.setattr(workers, "_worker_gpu_render_engine", lambda: engine)
    monkeypatch.setattr(workers, "GpuRenderedYoloSource", ResidentSource)
    monkeypatch.setattr(workers, "CpuPreparedYoloSource", PreparedSource)
    monkeypatch.setattr(workers, "StreamingYoloVolumeSource", StreamingSource)
    monkeypatch.setattr(workers, "open_existing_gray_memmap", lambda *_args, **_kwargs: cube)
    monkeypatch.setattr(workers, "_worker_render_callable", lambda *_args, **_kwargs: lambda _: cpu_frame)
    monkeypatch.setattr(workers, "predict_source_and_accumulate", predict)
    if case == "slab":
        monkeypatch.setattr(workers, "is_azimuthal_view", lambda _: True)
        monkeypatch.setattr(workers, "azimuthal_streaming_gpu_render_supported", lambda _: True)
        monkeypatch.setattr(workers, "_azimuthal_slab_context_indices", lambda *_args, **_kwargs: (0,))
        monkeypatch.setattr(workers, "_azimuthal_slab_channel_renderer", lambda *_args, **_kwargs: lambda _: slab_frame)

    stats = workers.run_prediction_volume_in_worker(None, cfg, task)
    render = {name: stats[name] for name in ("render_path", "render_frames", "render_recovered_on_cpu")}
    assert json.loads(json.dumps(render)) == dict(
        render_path=path, render_frames=1, render_recovered_on_cpu=recovered,
    )
    assert type(render["render_frames"]) is int and render["render_recovered_on_cpu"] is recovered
    assert len(seen) == (2 if case == "lazy_retry" else 1)
    assert all(source.closed for source in seen)
    assert engine.disable_resident_after_runtime_failure.call_count == (1 if case == "lazy_retry" else 0)
    if case == "prepared":
        assert isinstance(seen[0], PreparedSource) and stats["cpu_input_frames"] == 1
    if case == "slab":
        engine.prerender_azimuthal_slab.assert_called_once()

    telemetry, outgoing, trace = Mock(), Mock(), Mock()
    monkeypatch.setattr(workers, "runtime_telemetry", lambda: telemetry)
    monkeypatch.setattr(workers, "runtime_trace_event", trace)
    receipt = dict(ok=True, stats=stats)
    workers._publish_detector_result(outgoing, receipt, device="cuda:0", task=task)
    outgoing.put.assert_called_once_with(receipt)
    trace.assert_called_once_with("worker_publication_done", task=task, device="cuda:0", **render)
    prefix = f"worker.render.{task['view'].family}.{path}"
    assert [(entry.args, entry.kwargs) for entry in telemetry.add.call_args_list] == [
        ((f"{prefix}.completed_tasks", 1), {}),
        ((f"{prefix}.completed_frames", 1), {}),
    ]


def test_failed_result_transport_does_not_count_completed_frames(monkeypatch):
    telemetry = Mock()
    outgoing = Mock()
    outgoing.put.side_effect = OSError("test publication transport failure")
    monkeypatch.setattr(workers, "runtime_telemetry", lambda: telemetry)
    monkeypatch.setattr(workers, "runtime_trace_event", Mock())
    receipt = dict(stats=dict(render_path="cpu", render_frames=3, render_recovered_on_cpu=True))

    with pytest.raises(OSError, match="test publication transport failure"):
        workers._publish_detector_result(outgoing, receipt, device="cuda:0", family="spherical")
    telemetry.add.assert_not_called()
