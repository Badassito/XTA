from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.qualify_sam_tiled_schedule import TRIALS, collect_trial, compare_trials, file_sha, legacy_parent_batches, main


def test_legacy_override_changes_only_order_inside_current_parent_cohorts():
    prepared = SimpleNamespace(crop_mode="tiled", tracker_jobs=tuple(
        SimpleNamespace(original_run_index=index // 3) for index in range(12)))
    current = ((0, 3, 1, 4, 2, 5), (6, 9, 7, 10, 8, 11))
    result = legacy_parent_batches(prepared, current, 1)
    assert result == ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))
    assert [set(batch) for batch in current] == [set(batch) for batch in result]
    assert current == ((0, 3, 1, 4, 2, 5), (6, 9, 7, 10, 8, 11))
    with pytest.raises(ValueError, match="exactly one worker"):
        legacy_parent_batches(prepared, current, 2)


def _matrix(tmp_path):
    from tests.test_sam_evidence_import import _bundle
    bundle = _bundle(tmp_path / 'evidence', tiled=True)
    (tmp_path / 'image.bin').write_bytes(b'synthetic canonical source images')
    np.save(tmp_path / 'observations.npy', np.zeros((5,12,16),np.uint8))
    fixture = dict(image_path=str(tmp_path/'image.bin'),image_sha256=file_sha(tmp_path/'image.bin'),
        parent_observations=str(tmp_path/'observations.npy'),parent_observations_sha256=file_sha(tmp_path/'observations.npy'))
    (tmp_path/'fixture.json').write_text(json.dumps(fixture))
    trials=[]
    for slot,schedule,repeat in TRIALS:
        output=tmp_path/slot;output.mkdir()
        report=dict(status='passed',unsettled_sam_workers=False,context_closed=True,gpu_leases_released=True,
            parent_stats={'sam_evidence_path':str(bundle.directory)},consolidated_stats=[],
            context_runtime={'start_seconds':1.},wall_seconds=2.,image_sha256=fixture['image_sha256'],
            original_input_sha256=fixture['parent_observations_sha256'])
        (output/'qualification.json').write_text(json.dumps(report))
        np.save(output/'online_selected_additions.npy',np.zeros((5,12,16),np.uint8))
        trials.append(dict(slot=slot,schedule=schedule,repeat=repeat,output=str(output),result=collect_trial(output)))
    return trials


def test_abba_comparison_ignores_performance_counters_but_rejects_changed_raw_or_scores(tmp_path):
    trials = _matrix(tmp_path)
    trials[1]["result"]["scopes"]["parent"]["counters"]["cache_hits"] = 33
    assert compare_trials(trials)["exact_all_child_raw_owned_candidate_availability_masks"]
    original=copy.deepcopy(trials)
    child=next(iter(trials[2]['result']['scopes']['parent']['content']['children'].values()))
    child['tracker_scores']['2']=0.76
    with pytest.raises(AssertionError, match="scores"):
        compare_trials(trials)
    trials = original
    mask=next(iter(trials[3]['result']['scopes']['parent']['content']['masks'].values()))
    mask['decoded_sha256']='changed'
    with pytest.raises(AssertionError, match="masks"):
        compare_trials(trials)


def test_abba_requires_complete_slots_and_selected_output_identity(tmp_path):
    trials=_matrix(tmp_path)
    with pytest.raises(ValueError, match="complete A1 B1 B2 A2"):
        compare_trials(trials[:3])
    trials[-1]["result"]["selected_parent_additions"]["decoded_sha256"] = "changed-output"
    with pytest.raises(AssertionError, match="selected parent"):
        compare_trials(trials)


@pytest.mark.parametrize('field,value', [('schedule','crop_local'),('repeat',99),('repeat',True)])
def test_abba_requires_exact_schedule_repeat_mapping(tmp_path,field,value):
    trials=_matrix(tmp_path)
    for trial in trials:trial[field]=value
    with pytest.raises(ValueError,match='complete A1 B1 B2 A2'):
        compare_trials(trials)


def test_no_generated_scope_cannot_claim_vacuous_exactness(tmp_path):
    trials=_matrix(tmp_path)
    for trial in trials:trial['result']['scopes']={}
    with pytest.raises(ValueError,match='generated parent evidence'):
        compare_trials(trials)


@pytest.mark.parametrize('missing', ['qualification.json','online_selected_additions.npy'])
def test_comparison_reopens_required_trial_artifacts(tmp_path,missing):
    trials=_matrix(tmp_path)
    (tmp_path/'B2'/missing).unlink()
    with pytest.raises(OSError):compare_trials(trials)


def test_changed_selected_artifact_rejected_even_when_snapshots_all_agree(tmp_path):
    trials=_matrix(tmp_path)
    added=np.zeros((5,12,16),np.uint8);added[2,4,4]=1
    np.save(tmp_path/'B2'/'online_selected_additions.npy',added)
    with pytest.raises(AssertionError,match='Persisted trial artifacts changed selected'):
        compare_trials(trials)


def _saved_report(tmp_path,trials,status='passed'):
    fixture=json.loads((tmp_path/'fixture.json').read_text())
    report=dict(schema='xta.sam-tiled-schedule-abba/1',status=status,trials=trials,
        fixture_sha256=file_sha(tmp_path/'fixture.json'),image_sha256=fixture['image_sha256'],
        observations_sha256=fixture['parent_observations_sha256'])
    if status=='failed':report['error']='previous qualification shutdown failed'
    (tmp_path/'schedule_qualification.json').write_text(json.dumps(report))
    return report


def test_compare_only_preserves_prior_failure_and_allows_valid_zero_selected_output(tmp_path):
    trials=_matrix(tmp_path)
    _saved_report(tmp_path,trials,status='failed')
    assert main(['--compare-only','--fixture',str(tmp_path/'fixture.json'),'--output',str(tmp_path)])==0
    report=json.loads((tmp_path/'schedule_qualification.json').read_text())
    assert report['status']=='failed' and report['error']=='previous qualification shutdown failed'
    assert report['comparison_status']=='passed'
    assert report['exact_comparison']['artifact_coverage_verified']
    assert report['exact_comparison']['qualification_claim'] is False
    assert all(trial['result']['selected_parent_additions']['foreground']==0 for trial in trials)


def test_compare_only_cannot_promote_failed_empty_receipt(tmp_path):
    trials=_matrix(tmp_path)
    for trial in trials:trial['result']['scopes']={}
    _saved_report(tmp_path,trials,status='failed')
    with pytest.raises(ValueError):
        main(['--compare-only','--fixture',str(tmp_path/'fixture.json'),'--output',str(tmp_path)])
    report=json.loads((tmp_path/'schedule_qualification.json').read_text())
    assert report['status']=='failed' and report['comparison_status']=='failed'
    assert 'exact_comparison' not in report


def test_compare_only_rechecks_source_fixture_identity(tmp_path):
    _saved_report(tmp_path,_matrix(tmp_path))
    (tmp_path/'image.bin').write_bytes(b'changed-source-image')
    with pytest.raises(ValueError,match='source image'):
        main(['--compare-only','--fixture',str(tmp_path/'fixture.json'),'--output',str(tmp_path)])
    report=json.loads((tmp_path/'schedule_qualification.json').read_text())
    assert report['status']=='passed' and report['comparison_status']=='failed'
