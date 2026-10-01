"""An unkillable predictor cannot become an available GPU by resetting a run."""
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest

from XTA import pipeline, sam_integration


class RefusedWorkerShutdown:
    def __init__(self):
        self.alive = True
        self.dispatch_stats = {}

    @property
    def residency_released(self):
        return not self.alive

    def cancel(self, reason):
        pass

    def close(self):
        if self.alive:
            raise RuntimeError('forced worker termination did not settle residency')


def test_unsettled_model_blocks_reset_and_new_run_until_exit_is_proven(tmp_path, monkeypatch):
    context = sam_integration.SamInterpolationContext(model_path='model', device_ids=('0',),
        temp_dir=tmp_path, evidence_root=tmp_path/'evidence',
        source_volume=np.zeros((3, 4, 4), np.uint8), source_identity='pixels')
    runtime = RefusedWorkerShutdown()
    releases = []
    context._runtime = runtime
    context._leases.append(SimpleNamespace(release=lambda: releases.append('released')))
    reset_calls = []
    monkeypatch.setattr(pipeline, '_reset_main_process_gpu_stage_coordinator',
        lambda: reset_calls.append('reset'))
    owner = weakref.ref(context)
    try:
        with pytest.raises(RuntimeError, match='did not settle'):
            context.close()
        assert sam_integration.sam_workers_unsettled()
        assert not context._closed
        assert context.source_volume is not None
        assert releases == []
        assert not pipeline._reset_gpu_stage_coordinator_if_sam_settled()
        assert reset_calls == []
        with pytest.raises(RuntimeError, match='new GPU admission is refused'):
            sam_integration.retry_unsettled_sam_workers()
        del context
        gc.collect()
        assert owner() is not None  # Failed-run locals may disappear safely.
        context = owner()
        runtime.alive = False
        sam_integration.retry_unsettled_sam_workers()
        assert not sam_integration.sam_workers_unsettled()
        assert context._closed
        assert context.source_volume is None
        assert releases == ['released']
        assert pipeline._reset_gpu_stage_coordinator_if_sam_settled()
        assert reset_calls == ['reset']
    finally:
        runtime.alive = False
        retained = owner()
        if retained is not None:
            retained.close()


def test_shutdown_error_with_proven_dead_workers_releases_ownership(tmp_path):
    context = sam_integration.SamInterpolationContext(model_path='model', device_ids=('0',),
        temp_dir=tmp_path, evidence_root=tmp_path/'evidence',
        source_volume=np.zeros((3, 4, 4), np.uint8), source_identity='pixels')
    def closed_error():
        raise RuntimeError('adapter shutdown reported a diagnostic error')
    context._runtime = SimpleNamespace(residency_released=True, close=closed_error,
        cancel=lambda reason: None, dispatch_stats={})
    releases = []
    context._leases.append(SimpleNamespace(release=lambda: releases.append('released')))
    with pytest.raises(RuntimeError, match='diagnostic error'):
        context.close()
    assert context._closed
    assert not sam_integration.sam_workers_unsettled()
    assert releases == ['released']
    context.close()
    assert releases == ['released']

