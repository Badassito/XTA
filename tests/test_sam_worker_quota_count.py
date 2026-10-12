"""The child allocator cap follows the actual admitted process count."""
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import sam_tracker_runtime as sam


@pytest.mark.parametrize('count', (2, 3, 4))
def test_worker_init_carries_cohort_count_and_fallback_uses_one(tmp_path, count):
    tracker = sam.SamInterpolationTracker(model_path='unused', device_ids=(0,),
        workers_per_device=count, artifact_root=tmp_path/'runs',
        progressive_pool=SimpleNamespace(cancel_event=None))
    try:
        config = tracker.worker_init({'0': 1/count}).adapter_config
        assert config['sam_workers_per_device'] == count
        assert config['cuda_allocator_fractions'] == {'0': 1/count}
        fallback = tracker.worker_init().adapter_config
        assert fallback['sam_workers_per_device'] == 1
        assert 'cuda_allocator_fractions' not in fallback
    finally:
        tracker._pool = None
        tracker.close()


@pytest.mark.parametrize('count', (2, 3, 4))
@pytest.mark.parametrize('oversubscribed', (False, True))
def test_child_rejects_oversubscribed_fraction_before_model_or_cuda_call(monkeypatch, count, oversubscribed):
    cuda = SimpleNamespace(set_per_process_memory_fraction=mock.Mock(),
        mem_get_info=mock.Mock(return_value=(1200, 1200)))
    monkeypatch.setitem(__import__('sys').modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setenv('LTA_EXECUTION_DEVICE_ID', '0')
    fraction = 1/count + (0.001 if oversubscribed else 0)
    config = dict(cuda_allocator_fractions={'0': fraction}, sam_workers_per_device=count)
    with mock.patch('XTA.lta_worker_adapter.build_worker_predictor', side_effect=RuntimeError('model boundary')) as build:
        if oversubscribed:
            with pytest.raises(ValueError, match='per-process CUDA allocator fraction'):
                sam.build_interpolation_predictor(config)
            build.assert_not_called()
            cuda.set_per_process_memory_fraction.assert_not_called()
            cuda.mem_get_info.assert_not_called()
        else:
            with pytest.raises(RuntimeError, match='model boundary'):
                sam.build_interpolation_predictor(config)
            cuda.set_per_process_memory_fraction.assert_called_once_with(fraction, 0)
            build.assert_called_once_with(config)


@pytest.mark.parametrize('count', (None, True, 1, 5))
def test_child_cannot_accept_quota_without_valid_shared_process_count(monkeypatch, count):
    monkeypatch.setitem(__import__('sys').modules, 'torch', SimpleNamespace())
    with pytest.raises((TypeError, ValueError)):
        sam.build_interpolation_predictor(dict(cuda_allocator_fractions={'0': .25}, sam_workers_per_device=count))
