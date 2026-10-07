"""Outer-context retries replay frozen tails and publish only the final attempt."""
from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_extrapolation as extrapolation
from XTA.sam_crop_retry import SamCropRetryAdmissionError, SamCropRetryPolicy
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter
from tests.test_sam_dynamic_crop_retry_adversarial import _decode_components, _provider, _result


def _case(root, *, mode='whole', canvas_contacts=False, erase_final=False, early_empty=False,
          failure=None, policy=None, planner_limits=None, managed=True, resource_profile=None):
    observed=np.zeros((8,384,768),np.uint8)
    observed[2:5,140:147,240:247]=1
    original=observed.copy()
    prepared=extrapolation.prepare_sam_extrapolation_pass(observed,distance=3,walk_back=1,
        min_radius=3.,crop_mode=mode,planner_limits=planner_limits,resource_profile=resource_profile,
        eligible_terminals=lambda terminal,direction:terminal.frame_index==4 and direction==1)
    assert len(prepared.groups)==1
    group=prepared.groups[0]
    truth=observed.copy()
    truth[5,143,240:701]=1
    truth[6,143,650:657]=1
    truth[7,140:147,240:247]=1
    if canvas_contacts:
        truth[5]=1
    if early_empty:
        truth[5]=0
    calls,providers,leases=[],[],[]
    active=[False]
    runtime=SimpleNamespace(_cancel=Event(),_closed=False)

    def run(**request):
        x0,y0,x1,y1=request['crop_xyxy']
        index=len(providers)
        if index:
            assert active[0] if managed else True
        seed=observed[request['seed_frame'],y0:y1,x0:x1].astype(bool)
        np.testing.assert_array_equal(request['seed_mask'],seed)
        calls.append(dict(attempt=index,seed_frame=request['seed_frame'],
            interval=(request['frame_start'],request['frame_stop'],request['direction']),
            crop=(y0,x0,y1,x1)))
        if index==2 and failure is not None:
            if failure=='cancel':
                runtime._cancel.set()
            if failure=='worker':
                raise RuntimeError('genuine synthetic worker failure')
            if failure=='io':
                raise OSError('genuine synthetic evidence I/O failure')
            raise RuntimeError('synthetic cancellation during replay')
        masks={frame:truth[frame,y0:y1,x0:x1].astype(bool)
               for frame in range(request['frame_start'],request['frame_stop'])}
        if erase_final and x1>700:
            masks[5][:]=False
        return _result(masks)

    runtime.run=run
    def factory(retry):
        assert not active[0]
        assert retry.plan.observations is prepared.plan.observations
        assert retry.observation_snapshot_sha256==prepared.observation_snapshot_sha256
        assert {run.seed_ids for run in retry.runs}=={run.seed_ids for run in prepared.runs}
        assert [run.expected_frames for run in retry.runs]==[run.expected_frames for run in prepared.runs]
        providers.append(retry)
        if not managed:
            return _provider(retry)
        @contextmanager
        def leased():
            active[0]=True
            leases.append(('enter',len(providers)))
            try:
                yield _provider(retry)
            finally:
                active[0]=False
                leases.append(('exit',len(providers)))
        return leased()

    def execute():
        return extrapolation.extrapolate_sam_view_volume_pass(observed,work_dir=root,
            scope='sam',prepared_plan=prepared,image_provider=_provider(prepared),runtime=runtime,
            distance=3,walk_back=1,min_radius=3.,crop_mode=mode,planner_limits=planner_limits,
            resource_profile=resource_profile,
            crop_retry_policy=policy or SamCropRetryPolicy(enabled=True),retry_image_provider=factory)
    return SimpleNamespace(run=execute,observed=observed,original=original,truth=truth,prepared=prepared,
        group=group,calls=calls,providers=providers,leases=leases,active=active,root=root,runtime=runtime)


def _ledger(case):
    path=next(case.root.glob('sam_extrap_*/crop_retry.json'))
    return path.parent,json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_multiple_enlargements_replay_original_interval_and_keep_last_authority(tmp_path,mode):
    case=_case(tmp_path,mode=mode)
    returned,stats,components=case.run()
    assert returned is case.observed
    np.testing.assert_array_equal(case.observed,case.original)
    assert len(case.providers)>=2 and not case.active[0]
    assert case.leases==[event for index in range(1,len(case.providers)+1)
                        for event in (('enter',index),('exit',index))]
    boxes=[case.group.context_bbox_yx,*[item.groups[0].context_bbox_yx for item in case.providers]]
    for prior,current in zip(boxes,boxes[1:]):
        assert current!=prior and current[0]<=prior[0]<prior[2]<=current[2]
        assert current[1]<=prior[1]<prior[3]<=current[3]
    first=[row for row in case.calls if row['attempt']==0]
    for index in range(1,len(case.providers)+1):
        attempt=[row for row in case.calls if row['attempt']==index]
        assert [(row['seed_frame'],row['interval']) for row in attempt]==[
            (row['seed_frame'],row['interval']) for row in first]
    actual=_decode_components(components,case.observed.shape)
    np.testing.assert_array_equal(actual,case.truth & ~case.original)
    assert actual[6].sum()==7 and actual[7].sum()==49
    destination,ledger=_ledger(case)
    rows=ledger['attempt_history'][case.group.group_id]
    assert len(rows)==len(case.providers) and all(row['status']=='succeeded' for row in rows)
    assert [row['attempt_index'] for row in rows]==list(range(1,len(rows)+1))
    assert rows[-1]['resolution_status']=='outer_context_resolved' and not rows[-1]['extent_censored']
    assert ledger['charged_tracker_frames']==sum(row['retry_tracker_frames'] for row in rows)
    final=SamEvidenceBundle.open(stats['sam_evidence_path'])
    accepted=final.scope['crop_retry_final_selection']['accepted_attempts'][case.group.group_id]
    latest=SamEvidenceBundle.open(accepted['evidence_path'])
    assert set(final.runs)==set(latest.runs) and accepted['attempt_index']==len(rows)
    for run in final.runs.values():
        assert run['crop_retry_of_run_id'] in {item.run_id for item in case.prepared.runs}
    assert not list((destination/'retry_attempts').rglob('sam_extrapolation_*.cvol'))


def test_canvas_edges_finish_outer_resolution_without_claiming_global_coverage(tmp_path):
    case=_case(tmp_path,canvas_contacts=True)
    _,_,components=case.run()
    assert case.providers[-1].groups[0].context_bbox_yx==(0,0,384,768)
    _,ledger=_ledger(case)
    diagnostic=next(iter(ledger['final_reached_prefix_extent_diagnostics'][case.group.group_id].values()))
    assert diagnostic['outer_context_resolved'] and not diagnostic['coverage_proof']
    assert any(diagnostic['outer_context_contacts']['canvas_edge_contacts'].values())
    np.testing.assert_array_equal(_decode_components(components,case.observed.shape),case.truth & ~case.original)


def test_latest_empty_prefix_replaces_all_old_predictions_without_splice(tmp_path):
    case=_case(tmp_path,erase_final=True)
    _,_,components=case.run()
    assert len(case.providers)>=2
    assert not _decode_components(components,case.observed.shape).any()
    assert case.truth[7].any()  # Later reappearance cannot bypass the final raw-empty stop.
    np.testing.assert_array_equal(case.observed,case.original)


@pytest.mark.parametrize('mode',['whole','tiled'])
def test_unreached_suffix_contacts_do_not_start_a_retry(tmp_path,mode):
    case=_case(tmp_path,mode=mode,early_empty=True)
    _,_,components=case.run()
    assert not case.providers and not _decode_components(components,case.observed.shape).any()
    _,ledger=_ledger(case)
    assert ledger['charged_tracker_frames']==0


@pytest.mark.parametrize('failure',['worker','io','cancel'])
def test_later_failure_stops_scope_without_final_success_or_publication(tmp_path,failure):
    case=_case(tmp_path,failure=failure)
    with pytest.raises((RuntimeError,OSError),match='synthetic'):
        case.run()
    destination,ledger=_ledger(case)
    assert ledger['status']==('cancelled' if failure=='cancel' else 'failed')
    assert ledger['scope_error'] and not case.active[0]
    assert not (destination/'selection.json').exists()
    assert not list(destination.glob('sam_extrapolation_*.cvol'))
    assert (destination/'initial_selection.json').is_file()
    assert SamEvidenceBundle.open(ledger['original_evidence_path']).manifest['complete']
    previous=ledger['last_complete_attempts'][case.group.group_id]
    assert previous['attempt_index']==1 and SamEvidenceBundle.open(previous['evidence_path']).manifest['complete']
    rows=ledger['attempt_history'][case.group.group_id]
    assert rows[0]['status']=='succeeded' and rows[-1]['status']==('cancelled' if failure=='cancel' else 'failed')
    assert len(case.providers)==2
    np.testing.assert_array_equal(case.observed,case.original)


@pytest.mark.parametrize('resource',['memory','cache','planner'])
def test_needed_admission_failure_is_named_and_never_publishes_clipped_success(tmp_path,monkeypatch,resource):
    policy=SamCropRetryPolicy(enabled=True,max_retry_memory_bytes=1) if resource=='memory' else None
    limits=None
    if resource=='cache':
        monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES','1')
    if resource=='planner':
        from XTA.sam_bridge_planning import SamPlanningLimits
        limits=SamPlanningLimits(max_crop_pixels=263*263)
    case=_case(tmp_path,policy=policy,planner_limits=limits)
    with pytest.raises(SamCropRetryAdmissionError) as caught:
        case.run()
    assert case.group.group_id in str(caught.value) and 'scope sam' in str(caught.value)
    if resource=='cache':
        assert 'YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES' in str(caught.value)
    elif resource=='planner':
        assert 'max_crop_pixels' in str(caught.value)
    else:
        assert 'effective_limit_bytes=1' in str(caught.value)
    destination,ledger=_ledger(case)
    assert ledger['status']=='failed' and not case.providers
    assert not (destination/'selection.json').exists() and not list(destination.glob('sam_extrapolation_*.cvol'))
    assert ledger['last_complete_attempts'][case.group.group_id]['attempt_index']==0
    np.testing.assert_array_equal(case.observed,case.original)


def test_final_evidence_import_failure_persists_failed_ledger_and_complete_attempts(tmp_path,monkeypatch):
    case=_case(tmp_path)
    original_import=SamEvidenceWriter.import_group
    def fail_import(writer,*args,**kwargs):
        if writer.directory.name=='final_evidence':
            raise OSError('final evidence import failed')
        return original_import(writer,*args,**kwargs)
    monkeypatch.setattr(SamEvidenceWriter,'import_group',fail_import)
    with pytest.raises(OSError,match='final evidence import failed'):
        case.run()
    destination,ledger=_ledger(case)
    assert ledger['status']=='failed' and ledger['scope_error']['phase']=='final_import_or_selection'
    last=ledger['last_complete_attempts'][case.group.group_id]
    assert SamEvidenceBundle.open(last['evidence_path']).manifest['complete']
    assert not (destination/'selection.json').exists() and not list(destination.glob('sam_extrapolation_*.cvol'))


def test_legacy_bare_image_provider_remains_supported(tmp_path):
    case=_case(tmp_path,managed=False)
    _,_,components=case.run()
    assert len(case.providers)>=2 and not case.leases
    np.testing.assert_array_equal(_decode_components(components,case.observed.shape),case.truth & ~case.original)
