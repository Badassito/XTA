"""The throughput successor retains the strictly qualified performance history."""
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path('C:\\Users\\Bry\\Documents\\ChatGPT\\Scratch\\Experiments\\Job150772_Performance_20261002\\final_qualification_v2\\source\\XTA_v25.0.1_complete_source.zip')
KEY = 'v24_job150790_150798_throughput_development_review'
PREFIX = 'REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT'


@pytest.fixture(scope='module')
def predecessor():
    if ARCHIVE.is_file():
        files, metadata = prepare.qualified_development_archive(ARCHIVE, development='job150790-150798-throughput')
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    prior = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    prior.pop(KEY, None)
    metadata = dict(kind='qualified_development_source_zip', qualification_status='passed',
        full_qualification=True, package_version='25.0.1', source_identity_count=724,
        sha256=inventory.REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256,
        filename='XTA_v25.0.1_complete_source.zip')
    return None, metadata, prior


def review_fixture(predecessor):
    _files, metadata, prior = predecessor
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Retain the strictly qualified performance tool.')
             for row in prior['v24_job150772_performance_development_review']['validation_tools']]
    tools.extend(dict(path=path, previous_sha256=previous,
                      sha256=hashlib.sha256((prepare.ROOT/path).read_text(encoding='utf-8').encode()).hexdigest(),
                      reason='Add bounded confidence capture qualification.')
                 for path, previous in getattr(inventory, PREFIX+'_ADDED_VALIDATION_TOOLS').items())
    return dict(release='24.0.1', kind='development', development='job150790-150798-throughput',
        package_version='24.0.1', released=False, predecessor_tag='v24.0.0',
        feature='sam-job150790-150798-throughput-development',
        previous_review_sha256=inventory.REVIEWED_V24_JOB150772_PERFORMANCE_DEVELOPMENT_SHA256,
        predecessor_commit=inventory.REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(prior), predecessor_source_archive=metadata,
        definitions=[], statements=[], removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[], complete_modules=[],
        module_snapshots=[], validation_tools=tools)


def authenticate(prior, review):
    with mock.patch.object(inventory, PREFIX+'_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX+'_PREDECESSOR_MODULES', {}), \
         mock.patch.object(inventory, PREFIX+'_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v24_job150790_150798_throughput_development_contract(
            {**prior, KEY: review}, prior['v21_review'])


def test_throughput_predecessor_retains_exact_qualified_performance_identity(predecessor):
    files, metadata, prior = predecessor
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 724
    assert prepare.canonical(prior) == inventory.REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT_PREDECESSOR_SHA256
    assert prepare.canonical(prior['v24_job150772_performance_development_review']) == inventory.REVIEWED_V24_JOB150772_PERFORMANCE_DEVELOPMENT_SHA256
    assert metadata['full_qualification'] is True
    assert authenticate(prior, review_fixture(predecessor))['released'] is False


@pytest.mark.parametrize('field,value,error', [
    ('kind', 'reviewed_development_source_zip', 'predecessor archive changed'),
    ('qualification_status', 'incomplete', 'completed v24.0.1 qualification'),
    ('full_qualification', False, 'completed v24.0.1 qualification'),
    ('full_qualification', 1, 'completed v24.0.1 qualification'),
    ('sha256', '0'*64, 'predecessor archive changed'),
])
def test_throughput_predecessor_cannot_be_substituted_or_relabelled(predecessor, field, value, error):
    _files, _metadata, prior = predecessor
    review = review_fixture(predecessor)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match=error):
        authenticate(prior, review)


@pytest.mark.parametrize('key', ['v24_job150772_performance_development_review', 'v24_0_1_release_review',
                               'v24_0_0_release_review'])
def test_throughput_successor_cannot_rewrite_history(predecessor, key):
    _files, _metadata, prior = predecessor
    altered = copy.deepcopy(prior)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, review_fixture(predecessor))


def test_throughput_source_zip_is_required(tmp_path):
    with pytest.raises(ValueError, match='requires --predecessor-archive'):
        prepare.prepare(output_dir=tmp_path, development='job150790-150798-throughput')


@pytest.mark.parametrize('development,expected', [
    ('job150772-performance', '92fad99e47314c0c0cf95fda1daa5a7965322e3cba133597f8699a8fa9380256'),
    ('job150790-150798-throughput', '0'*64),
])
def test_candidate_amendment_cannot_target_another_review_or_digest(development, expected):
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='exact current unqualified'):
        prepare.validate_unqualified_candidate_amendment(current, development, expected, None)


def test_candidate_amendment_requires_bound_failed_proof(tmp_path):
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    if prepare.canonical(current[KEY]) != inventory.UNQUALIFIED_THROUGHPUT_CANDIDATE_SHA256:
        pytest.skip('The documented failed candidate has already been superseded')
    args = (current, 'job150790-150798-throughput', inventory.UNQUALIFIED_THROUGHPUT_CANDIDATE_SHA256)
    with pytest.raises(ValueError, match='failed qualification receipt'):
        prepare.validate_unqualified_candidate_amendment(*args, None)
    fake = tmp_path/'successful.json'
    fake.write_text(json.dumps(dict(success=True, kind='development-snapshot', steps=[])), encoding='utf-8')
    with pytest.raises(ValueError, match='independent pin'):
        prepare.validate_unqualified_candidate_amendment(*args, fake)


def test_candidate_amendment_is_explicit_write_only(tmp_path):
    with pytest.raises(ValueError, match='requires --write'):
        prepare.prepare(output_dir=tmp_path, development='job150790-150798-throughput',
            predecessor_archive=ARCHIVE,
            amend_unqualified_candidate=inventory.UNQUALIFIED_THROUGHPUT_CANDIDATE_SHA256)


def test_private_throughput_publication_preserves_all_earlier_records(predecessor, tmp_path, monkeypatch):
    files, _metadata, prior = predecessor
    if files is None:
        pytest.skip('Retained qualified archive is needed for the private source fixture')
    root = tmp_path/'repository'
    for relative, payload in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root/'XTA/throughput_fixture.py').write_text('VALUE = 1\n', encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py',
                     'tools/benchmark_confidence_capture.py', 'tools/qualify_sam_policy_throughput.py',
                     'tools/qualify_sam_family_dispatch.py'):
        (root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest = root/'release/_package_inventory.json'
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(inventory, 'ROOT', root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest)
    def fixture_git(command, **kwargs):
        assert command == ['git', 'rev-parse', 'v24.0.0^{commit}']
        return inventory.REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=tmp_path/'evidence', development='job150790-150798-throughput',
                             predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == prior
    assert result['written'] and published[KEY]['released'] is False
