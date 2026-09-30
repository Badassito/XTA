"""GPU sample identity survives repeated telemetry flushes and sampling errors."""
import json
import threading
from types import SimpleNamespace
from unittest import mock

from XTA import runtime


class FakeNvml:
    def nvmlInit(self): pass
    def nvmlDeviceGetCount(self): return 2
    def nvmlDeviceGetHandleByIndex(self,index): return index
    def nvmlDeviceGetUtilizationRates(self,index):
        return SimpleNamespace(gpu=40+index,memory=10+index)
    def nvmlDeviceGetMemoryInfo(self,index):
        return SimpleNamespace(used=100+index,total=1000)
    def nvmlDeviceGetUUID(self,index): return f'GPU-{index}'.encode()
    def nvmlDeviceGetPciInfo(self,index): return SimpleNamespace(busId=f'0000:0{index}:00.0')


def telemetry(tmp_path):
    with mock.patch.dict('os.environ',{'YOLO_TTA_TELEMETRY':'1',
                                      'YOLO_TTA_TELEMETRY_DIR':str(tmp_path)}):
        return runtime.RuntimeTelemetry()


def test_unchanged_real_samples_have_new_times_but_flushes_do_not(tmp_path):
    sink=telemetry(tmp_path)
    sampler=runtime.RuntimeSystemSampler(sink)
    with mock.patch.object(runtime,'_load_nvidia_ml_py',return_value=FakeNvml()):
        with mock.patch.object(runtime.time,'monotonic_ns',side_effect=[100,110]):
            sampler._sample_gpu()
        sink.flush(); sink.flush()
        with mock.patch.object(runtime.time,'monotonic_ns',side_effect=[200,210]):
            sampler._sample_gpu()
        sink.flush()
    records=[json.loads(line)['gauges'] for line in sink.path.read_text().splitlines()]
    assert [row['system.gpu_sample_monotonic_ns'] for row in records]==[110,110,210]
    assert [row['system.gpu_sample_started_ns'] for row in records]==[100,100,200]
    assert records[0]['system.gpu_utilization']==records[2]['system.gpu_utilization']
    assert records[0]['system.gpu_devices']==[
        {'nvml_index':0,'uuid':'GPU-0','pci_bus_id':'0000:00:00.0'},
        {'nvml_index':1,'uuid':'GPU-1','pci_bus_id':'0000:01:00.0'}]
    sink.flush(final=True)


def test_failed_partial_poll_does_not_publish_mixed_gauges_or_new_timestamp(tmp_path):
    sink=telemetry(tmp_path)
    sampler=runtime.RuntimeSystemSampler(sink)
    nvml=FakeNvml()
    with mock.patch.object(runtime,'_load_nvidia_ml_py',return_value=nvml):
        sampler._sample_gpu()
        original=sink.snapshot()['gauges']
        with mock.patch.object(nvml,'nvmlDeviceGetUtilizationRates',side_effect=[
                SimpleNamespace(gpu=99,memory=99),RuntimeError('device poll failed')]):
            sampler._sample_gpu()
    assert sink.snapshot()['gauges']==original


def test_optional_identity_errors_preserve_utilization_sample(tmp_path):
    sink=telemetry(tmp_path)
    sampler=runtime.RuntimeSystemSampler(sink)
    nvml=FakeNvml()
    with mock.patch.object(runtime,'_load_nvidia_ml_py',return_value=nvml), \
         mock.patch.object(nvml,'nvmlDeviceGetUUID',side_effect=NotImplementedError), \
         mock.patch.object(nvml,'nvmlDeviceGetPciInfo',side_effect=NotImplementedError):
        sampler._sample_gpu()
    gauges=sink.snapshot()['gauges']
    assert gauges['system.gpu_devices']==[{'nvml_index':0},{'nvml_index':1}]
    assert gauges['system.gpu_utilization'][1]['gpu']==41
    assert gauges['system.gpu_sample_monotonic_ns']>=gauges['system.gpu_sample_started_ns']


def test_final_provider_sample_cannot_be_overwritten_by_older_sample(tmp_path):
    sink = telemetry(tmp_path)
    entered, release, final_done = threading.Event(), threading.Event(), threading.Event()
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return {'waiting_writers': 1 if calls == 1 else 0}

    sink.register_sample_provider('nrrd.member_stream.live', provider)
    periodic = threading.Thread(target=sink.sample_registered_providers)
    periodic.start()
    assert entered.wait(timeout=1)
    final = threading.Thread(target=lambda: (
        sink.sample_registered_providers(), final_done.set(),
    ))
    final.start()
    try:
        assert not final_done.wait(timeout=.05)
    finally:
        release.set()
        periodic.join(timeout=2)
        final.join(timeout=2)
    assert final_done.is_set()
    assert sink.snapshot()['gauges']['nrrd.member_stream.live']['waiting_writers'] == 0


def test_sample_provider_failure_and_fallback_failure_do_not_stop_other_samples(tmp_path):
    sink = telemetry(tmp_path)

    def broken():
        raise RuntimeError('diagnostic failed')

    sink.register_sample_provider('broken', broken)
    sink.register_sample_provider('healthy', lambda: {'sample_monotonic_ns': 123})
    with mock.patch.object(sink, 'fallback', side_effect=RuntimeError('fallback failed')):
        sink.sample_registered_providers()
    assert sink.snapshot()['gauges']['healthy'] == {'sample_monotonic_ns': 123}
