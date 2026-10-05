"""CPU regression for normal detector-drained standalone GPU admission setup."""
from types import SimpleNamespace

import pytest

from XTA import backprojection as bp
from tools import smoke_sam_feature_dispatch as cache
from tools import smoke_sam_gpu_handoff as handoff


@pytest.fixture
def coordinator(monkeypatch):
    owner = bp._MainProcessGpuStageCoordinator()
    monkeypatch.setattr(bp, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', owner)
    monkeypatch.setattr(bp, 'gpu_worker_aux_interpolation_pool', lambda: None)
    return owner


@pytest.mark.parametrize('frontend', [handoff, cache])
def test_no_detector_setup_publishes_retirement_and_normal_sam_admission(frontend, coordinator):
    receipt = frontend.initialize_standalone_coordinator(bp, 0)
    snapshot = receipt['coordinator_after']
    assert receipt['detector_workers_started'] is False
    assert receipt['detector_assets_started'] is False
    assert snapshot['inference_priority_active'] is False
    assert snapshot['pending_inference_backlog'] is False
    assert snapshot['inference_asset_retirement_pending'] is False
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    lease = bp._try_acquire_specific_main_process_gpu_stage(fake, 0, 'TTA persistent SAM interpolation predictor')
    assert lease is not None
    lease.release()
    assert coordinator.snapshot()['stage_leases'] == {}


@pytest.mark.parametrize('frontend', [handoff, cache])
@pytest.mark.parametrize('kind', ['inflight', 'stage', 'resident'])
def test_setup_does_not_override_real_existing_owner(frontend, coordinator, kind):
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    lease = None
    resident = None
    if kind == 'inflight':
        assert coordinator.begin_inference(0)
    else:
        fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
        lease = coordinator.try_acquire_specific_stage(fake, 0, 'existing owner')
        assert lease is not None
        if kind == 'resident':
            resident = lease.promote_residency()
    before = coordinator.snapshot()
    try:
        with pytest.raises(AssertionError, match='cannot clear another'):
            frontend.initialize_standalone_coordinator(bp, 0)
        assert coordinator.snapshot() == before
    finally:
        if kind == 'inflight':
            coordinator.finish_inference(0)
        if resident is not None:
            resident.release(residency_settled=True)
        elif lease is not None:
            lease.release()


@pytest.mark.parametrize('frontend', [handoff, cache])
def test_receipt_serialization_preserves_real_writer_immutable_sdk_and_scores(frontend, tmp_path):
    import json
    import numpy as np
    from XTA.geometry import get_view_infos
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_extrapolation import extrapolate_sam_view_volume_pass
    from tests.test_sam_image_cohort_smoke_capture import Tracker, eligible

    baseline = np.zeros((23, 48, 64), np.uint8)
    baseline[4:19, 15:24, 20:31] = 1
    view = get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    _, stats, _ = extrapolate_sam_view_volume_pass(baseline, view=view,
        work_dir=tmp_path / 'evidence', runtime=Tracker(), distance=4, walk_back=0,
        min_radius=3., crop_mode='whole', eligible_terminals=eligible)
    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
    run = next(iter(bundle.runs.values()))
    path = tmp_path / 'result.json'
    frontend.write_receipt(path, {'sdk': run['runtime_receipt'], 'raw_scores': run['tracker_scores'],
                                'seed_ids': run['seed_ids'], 'nullable_parent_scores': None})
    saved = json.loads(path.read_text())
    assert saved['raw_scores'] == {str(frame): .25 + frame / 100 for frame in run['expected_frames']}
    assert saved['sdk']['adapter_receipt']['seed_roundtrip_exact'] is True
    assert saved['sdk']['sam_model']['checkpoint_sha256'] == 'synthetic'
    assert saved['nullable_parent_scores'] is None
    assert saved['seed_ids'] == list(run['seed_ids'])
