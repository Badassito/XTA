"""One enlarged family attempt uses the same original seed and global policy."""
import json
from contextlib import contextmanager
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
    def __init__(self, shape, *, fail_retry=False, growth_stop=100, malformed_retry=False):
        self.shape = shape
        self.calls = []
        self.fail_retry = fail_retry
        self.growth_stop = growth_stop
        self.malformed_retry = malformed_retry
        self.live=0
    def run(self, **request):
        x0, y0, x1, y1 = request['crop_xyxy']
        seed = np.zeros(self.shape[1:], bool)
        seed[y0:y1, x0:x1] = request['seed_mask']
        self.calls.append((request['run_id'], tuple(request['crop_xyxy']), seed.copy(),
                           request['frame_start'], request['frame_stop']))
        retry_index=(int(request['run_id'].split('__retry',1)[1].split('_',1)[0])
            if '__retry' in request['run_id'] else 0)
        if retry_index and retry_index==int(self.fail_retry):
            raise RuntimeError('optional retry SDK failure')
        frames = {}
        for frame in range(request['frame_start'], request['frame_stop']):
            raw = np.zeros(self.shape[1:], bool)
            raw[64:70, 64:self.growth_stop if frame == 1 else 70] = True
            frames[frame] = raw[y0:y1, x0:x1].copy()
        self.live+=1
        return SimpleNamespace(frames=frames, tracker_scores={}, observation_status={},
            receipt={'run_id': request['run_id'], 'prediction_valid': True,
                'coverage_complete': not (retry_index and self.malformed_retry)})
    def release_result(self,result):
        self.live-=1


def _execute(tmp_path, *, enabled=True, fail_retry=False, provider_failure=False, crop_mode='whole',
        growth_stop=100, source=None, retry_policy=None, lease_events=None, malformed_retry=False):
    native = np.zeros((3, 160, 160), np.uint8) if source is None else source
    native[0, 64:70, 64:70] = native[2, 64:70, 64:70] = 1
    runtime = _GrowingRuntime(native.shape, fail_retry=fail_retry,growth_stop=growth_stop,
        malformed_retry=malformed_retry)
    providers = []
    def provider(prepared):
        providers.append(prepared)
        if provider_failure:
            raise RuntimeError('optional retry renderer failure')
        descriptor=SimpleNamespace(shape=prepared.native_shape, identity_sha256='a'*64,
            frame_crops=tuple((frame, *box, 0) for frame, box in prepared.frame_crop_bounds.items()))
        if lease_events is None:
            return descriptor
        @contextmanager
        def lease():
            assert runtime.live==0
            lease_events.append('enter')
            try:
                yield descriptor
            finally:
                assert runtime.live==0
                lease_events.append('exit')
        return lease()
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
        crop_retry_policy=retry_policy or SamCropRetryPolicy(enabled=enabled), retry_image_provider=provider,
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
def test_required_retry_failure_stops_scope_and_preserves_raw_diagnostics(tmp_path, failure):
    with pytest.raises(RuntimeError,match='retry .* failure'):
        _execute(tmp_path,provider_failure=failure == 'renderer', fail_retry=failure == 'sdk')
    ledger=json.loads(next(tmp_path.glob('sam_*/crop_retry.json')).read_text())
    assert ledger['scope_failed'] is True
    assert all(record['status']=='failed' for record in ledger['attempts'].values())
    assert SamEvidenceBundle.open(ledger['initial_evidence_path']).manifest['complete'] is True
    assert not list(tmp_path.glob('sam_*/selection.json'))
    assert not list(tmp_path.glob('sam_*/sam_bridge_*.cvol'))
    failure_receipt=json.loads(next(tmp_path.glob('sam_*/failure.json')).read_text())
    assert failure_receipt['complete'] is False and failure_receipt['phase']=='crop_retry'


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


@pytest.mark.parametrize('mode',('whole','tiled'))
def test_repeated_enlargement_keeps_full_original_seeds_and_only_imports_last_attempt(tmp_path,mode):
    source=np.zeros((3,160,512),np.uint8)
    source[[0,2],64:70,64:70]=1
    before=source.copy()
    events=[]
    native,runtime,providers,merged,stats,_=_execute(tmp_path,source=source,growth_stop=300,
        crop_mode=mode,lease_events=events)
    try:
        expected=before.copy()
        expected[1,64:70,64:300]=1
        np.testing.assert_array_equal(merged,expected)
        np.testing.assert_array_equal(native,before)
        ledger=stats['sam_crop_retry']
        history=next(iter(ledger['attempt_history'].values()))
        assert len(history)>=2 and len(providers)==len(history)
        assert events==['enter','exit']*len(history) and runtime.live==0
        assert [row['attempt_index'] for row in history]==list(range(1,len(history)+1))
        assert history[-1]['completion_detail']['outer_context_resolved'] is True
        assert all(row['completion_detail']['extent_remains_censored'] for row in history[:-1])
        assert ledger['charged_tracker_frames']==6*len(history)
        assert ledger['charged_pixel_frames']==sum(row['retry_pixel_frames'] for row in history)
        for _,_,seed,start,stop in runtime.calls:
            np.testing.assert_array_equal(seed,before[0])
            assert (start,stop)==(0,3)
        final=SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert len(final.groups)==1 and len(final.runs)==2
        assert all('__retry'+str(len(history))+'_' in rid for rid in final.runs)
        for row in history:
            attempt=SamEvidenceBundle.open(row['completion_detail']['evidence_path'])
            assert attempt.manifest['complete'] is True
            if row is not history[-1]:
                assert set(attempt.runs).isdisjoint(final.runs)
        assert stats['added_voxels']==6*(300-64)
    finally:
        merged._mmap.close()


def test_expansion_can_reach_declared_canvas_edge_without_certifying_external_coverage(tmp_path):
    source=np.zeros((3,160,280),np.uint8)
    source[[0,2],64:70,64:70]=1
    _,_,providers,merged,stats,_=_execute(tmp_path,source=source,growth_stop=280)
    try:
        assert providers[-1].groups[0].context_bbox_yx[3]==280
        row=next(iter(stats['sam_crop_retry']['attempts'].values()))
        detail=row['completion_detail']
        assert detail['outer_context_resolved'] is True and detail['coverage_proof'] is False
        assert detail['outer_context_contacts']['canvas_edge_contacts']['right']>0
        assert merged[1,65,279]==1
    finally:
        merged._mmap.close()


@pytest.mark.parametrize('failure',('configured_resource','second_sdk','malformed_complete_flag'))
def test_needed_failure_after_growth_never_exports_a_clipped_success(tmp_path,failure):
    source=np.zeros((3,160,512),np.uint8)
    source[[0,2],64:70,64:70]=1
    before=source.copy()
    kwargs=dict(source=source,growth_stop=300)
    if failure=='configured_resource':
        kwargs['retry_policy']=SamCropRetryPolicy(enabled=True,max_extra_tracker_frames=6)
    elif failure=='second_sdk':
        kwargs['fail_retry']=2
    else:
        kwargs['malformed_retry']=True
    with pytest.raises(RuntimeError):
        _execute(tmp_path,**kwargs)
    np.testing.assert_array_equal(source,before)
    ledger=json.loads(next(tmp_path.glob('sam_*/crop_retry.json')).read_text())
    assert ledger['scope_failed'] is True
    history=next(iter(ledger['attempt_history'].values()))
    if failure!='malformed_complete_flag':
        assert history[0]['status']=='succeeded'
        assert history[0]['completion_detail']['extent_remains_censored'] is True
        assert SamEvidenceBundle.open(history[0]['completion_detail']['evidence_path']).manifest['complete'] is True
        assert history[-1]['status'] in {'failed','refused'}
    else:
        assert history[-1]['status']=='failed'
        assert 'complete valid original-seed' in history[-1]['completion_detail']['error']
    assert not list(tmp_path.glob('**/adaptive_final/evidence/manifest.json'))
    assert not list(tmp_path.glob('**/selection.json'))
    assert not list(tmp_path.glob('**/sam_bridge_*.cvol'))
    failure_receipt=json.loads(next(tmp_path.glob('sam_*/failure.json')).read_text())
    assert failure_receipt['complete'] is False and failure_receipt['phase']=='crop_retry'
