"""Indexed evidence remains accounted when an analysis is absent or empty."""
import json
from types import SimpleNamespace

from tools.report_sam_outer_crop import analysis_data, build


def _inputs(tmp_path, *, missing):
    constants={'datasets':[{'dataset_id':'development_seen655','stage':'development'},
                           {'dataset_id':'followup_594','stage':'followup'}]}
    (tmp_path/'protocol_constants.json').write_text(json.dumps(constants))
    (tmp_path/'analysis_index.json').write_text(json.dumps({'status':'passed','entries':[
        {'dataset_id':'development','analysis_file':'available.json'},
        {'dataset_id':'source_590_599','analysis_file':'second.json'}]}))
    (tmp_path/'available.json').write_text(json.dumps({'dataset':'development','models':[]}))
    if not missing:
        (tmp_path/'second.json').write_text(json.dumps({'dataset':'source_590_599','models':[]}))
    return constants


def test_missing_indexed_analysis_keeps_denominator_and_marks_full_report_incomplete(tmp_path):
    constants=_inputs(tmp_path,missing=True)
    data=analysis_data(tmp_path,constants)
    assert data['status']=='incomplete' and data['upstream_status']=='passed'
    assert [dataset['id']for dataset in data['datasets']]==['development','source_590_599']
    missing=data['datasets'][1]
    assert missing['analysis_available'] is False and missing['status']=='missing_analysis'
    assert missing['results']==[] and missing['models']==[]  # Never manufacture zero metrics.
    build(SimpleNamespace(experiment=tmp_path,data=None,output=tmp_path/'report.html',skip_figures=True))
    html=(tmp_path/'report.html').read_text()
    metrics=json.loads((tmp_path/'report.metrics.json').read_text())
    assert 'Status: <strong>incomplete</strong>'in html and 'Required indexed analysis is missing'in html
    assert 'No scored analysis published yet'in html
    assert metrics['status']=='incomplete' and metrics['missing_analysis_ids']==['source_590_599']
    assert len(metrics['datasets'])==2


def test_present_empty_analysis_is_available_and_does_not_imply_missing_evidence(tmp_path):
    constants=_inputs(tmp_path,missing=False)
    data=analysis_data(tmp_path,constants)
    assert data['status']=='passed' and data['missing_analysis_ids']==[]
    assert all(dataset['analysis_available']for dataset in data['datasets'])
    assert all(dataset['results']==[]for dataset in data['datasets'])


def test_normalized_view_cannot_hide_missing_indexed_analysis_or_publish_its_stale_scores(tmp_path):
    _inputs(tmp_path,missing=True)
    (tmp_path/'report_data.json').write_text(json.dumps({'status':'passed','datasets':[
        {'id':'source_590_599','results':[{'selected_metrics':{'iou':1.}}]}]}))
    build(SimpleNamespace(experiment=tmp_path,data=None,output=tmp_path/'report.html',skip_figures=True))
    metrics=json.loads((tmp_path/'report.metrics.json').read_text())
    assert metrics['status']=='incomplete'
    assert {dataset['id']for dataset in metrics['datasets']}=={'development','source_590_599'}
    missing=next(dataset for dataset in metrics['datasets']if dataset['id']=='source_590_599')
    assert missing['analysis_available'] is False and missing['results']==[]
