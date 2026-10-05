"""One enlarged family attempt uses the same original seed and global policy."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.geometry import get_view_infos
from XTA.sam_bridge_planning import SamPlanningLimits
from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_interpolation import interpolate_sam_view_volume_pass


class _GrowingRuntime:
    device_ids = (0,)
    def __init__(self, shape, *, fail_retry=False):
        self.shape = shape
        self.calls = []
        self.fail_retry = fail_retry
    def run(self, **request):
        x0, y0, x1, y1 = request['crop_xyxy']
        seed = np.zeros(self.shape[1:], bool)
        seed[y0:y1, x0:x1] = request['seed_mask']
        self.calls.append((request['run_id'], tuple(request['crop_xyxy']), seed.copy(),
                           request['frame_start'], request['frame_stop']))
        if '__retry1_' in request['run_id'] and self.fail_retry:
            raise RuntimeError('optional retry SDK failure')
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            raw = np.zeros(self.shape[1:], bool)
            raw[64:70, 64:100 if frame == 1 else 70] = True
            frames[frame] = raw[y0:y1, x0:x1].copy()
        return SimpleNamespace(frames=frames, tracker_scores={}, observation_status={},
            receipt={'run_id': request['run_id'], 'prediction_valid': True, 'coverage_complete': True})


def _execute(tmp_path, *, enabled=True, fail_retry=False, provider_failure=False, crop_mode='whole'):
    native = np.zeros((3, 160, 160), np.uint8)
    native[0, 64:70, 64:70] = native[2, 64:70, 64:70] = 1
    runtime = _GrowingRuntime(native.shape, fail_retry=fail_retry)
    providers = []
    def provider(prepared):
        providers.append(prepared)
        if provider_failure:
            raise RuntimeError('optional retry renderer failure')
        return SimpleNamespace(shape=prepared.native_shape, identity_sha256='a'*64,
            frame_crops=tuple((frame, *box, 0) for frame, box in prepared.frame_crop_bounds.items()))
    policy = {'sam_bridge_policy': dict(version=7 if crop_mode == 'tiled' else 6, kind='conservative',
        strict_containment=False, guarded_rescue=False, branch_aware_selection=True,
        branch_write_domain='fixed_context', branch_crop_boundary_policy='retain_censored',
        strict_family_agreement=False, min_endpoint_recall=.5)}
    merged, stats, components = interpolate_sam_view_volume_pass(native,
        view=get_view_infos(*native.shape, cartesian_views=('transverse',))[0],
        image_provider=SimpleNamespace(shape=native.shape, identity_sha256='a'*64),
        runtime=runtime, work_dir=tmp_path, gap_distance=3, min_radius=0.,
        interpolation_walk_back=0, search_angle_deg=-6., policy=policy,
        planner_limits=SamPlanningLimits(context_margin_px=8, curvature_margin_px=2,
                                        acceptance_margin_px=2),
        crop_retry_policy=SamCropRetryPolicy(enabled=enabled), retry_image_provider=provider,
        return_bridge_components=True, crop_mode=crop_mode)
    return native, runtime, providers, merged, stats, components


@pytest.mark.parametrize('crop_mode', ('whole', 'tiled'))
def test_one_group_retry_recovers_growth_and_preserves_initial_raw_evidence(tmp_path, crop_mode):
    native, runtime, providers, merged, stats, components = _execute(tmp_path, crop_mode=crop_mode)
    try:
        assert len(runtime.calls) == 4  # two original directions, one same-family retry
        assert len(providers) == 1
        for _, _, seed, start, stop in runtime.calls:
            np.testing.assert_array_equal(seed, native[0])
            assert (start, stop) == (0, 3)
        assert stats['added_voxels'] == 216
        assert merged[1, 65, 99] == 1
        ledger = stats['sam_crop_retry']
        assert sum(record['retry'] for record in ledger['attempts'].values()) == 1
        assert all(record['status'] == 'succeeded' for record in ledger['attempts'].values())
        initial = SamEvidenceBundle.open(ledger['initial_evidence_path'])
        final = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert initial.evidence_fingerprint == ledger['initial_evidence_fingerprint']
        assert set(initial.groups).isdisjoint(final.groups)
        assert all('__retry1_' in run_id for run_id in final.runs)
        assert all(run['crop_retry_of_run_id'] in initial.runs for run in final.runs.values())
        assert all(component['evidence_path'] == str(final.directory) for component in components)
        assert len(stats['sam_selection_receipt']['group_receipts']) == 1
        if crop_mode == 'tiled':
            child = next(iter(ledger['tiled_child_crop_contacts'].values()))
            assert child['initial']['coverage_proof'] is False
            assert child['retry']['coverage_proof'] is False
            assert child['retry']['internal_child_enlargement'].startswith('unsupported;')
    finally:
        merged._mmap.close()


@pytest.mark.parametrize('failure', ('renderer', 'sdk'))
def test_optional_retry_failure_records_attempt_and_preserves_original_selection(tmp_path, failure):
    native, runtime, providers, merged, stats, _ = _execute(tmp_path,
        provider_failure=failure == 'renderer', fail_retry=failure == 'sdk')
    try:
        assert len(providers) == 1
        assert stats['added_voxels'] == 108
        assert merged[1, 65, 81] == 1 and merged[1, 65, 99] == 0
        assert all(record['status'] == 'failed' for record in stats['sam_crop_retry']['attempts'].values())
        assert stats['sam_evidence_path'] == stats['sam_crop_retry']['initial_evidence_path']
        assert not any('__retry1_' in run_id for run_id in SamEvidenceBundle.open(stats['sam_evidence_path']).runs)
    finally:
        merged._mmap.close()


def test_default_off_never_censuses_or_rerenders_retry(tmp_path, monkeypatch):
    monkeypatch.setattr('XTA.sam_crop_retry.raw_crop_boundary_contacts',
        lambda *args, **kwargs: pytest.fail('default-off retry inspected raw masks'))
    _, runtime, providers, merged, stats, _ = _execute(tmp_path, enabled=False)
    try:
        assert len(runtime.calls) == 2 and not providers
        assert stats['added_voxels'] == 108
        assert 'sam_crop_retry' not in stats
    finally:
        merged._mmap.close()
