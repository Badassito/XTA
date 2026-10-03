"""A reviewed source archive cannot acquire a false qualification claim."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path('C:\\Users\\Bry\\Documents\\ChatGPT\\Scratch\\Experiments\\SAM_Job150615_20261001\\unqualified_9af_review\\XTA_v25.0.0_complete_source.zip')
KEY = 'v24_job150615_headroom_development_review'
PREFIX = 'REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT'


@pytest.fixture(scope='module')
def reviewed():
    if ARCHIVE.is_file():
        files, metadata = prepare.reviewed_development_archive(ARCHIVE)
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    predecessor = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    predecessor.pop('v24_job150790_150798_throughput_development_review', None)
    predecessor.pop('v24_job150772_performance_development_review', None)
    predecessor.pop('v24_0_1_release_review', None)
    predecessor.pop(KEY, None)
    metadata = dict(kind='reviewed_development_source_zip', qualification_status='incomplete',
        full_qualification=False,
        sha256=inventory.REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256,
        source_identity_count=698, filename='XTA_v25.0.0_complete_source.zip')
    return None, metadata, predecessor


def review_fixture(reviewed):
    _files, metadata, predecessor = reviewed
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Preserve the previously reviewed tool.')
             for row in predecessor['v24_job150615_development_review']['validation_tools']]
    return dict(release='24.0.0', kind='development', development='job150615-headroom', package_version='24.0.0',
        released=False, predecessor_tag='v24.0.0', feature='sam-job150615-headroom-development',
        previous_review_sha256=inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_SHA256,
        predecessor_commit=inventory.REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(predecessor), predecessor_source_archive=metadata,
        definitions=[], statements=[], removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[], complete_modules=[],
        module_snapshots=[], validation_tools=tools)


def authenticate(predecessor, review):
    with mock.patch.object(inventory, PREFIX+'_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX+'_PREDECESSOR_MODULES', {}), \
         mock.patch.object(inventory, PREFIX+'_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v24_job150615_headroom_development_contract(
            {**predecessor, KEY: review}, predecessor['v21_review'])


def test_reviewed_archive_retains_exact_9af_source_and_incomplete_status(reviewed):
    files, metadata, predecessor = reviewed
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 698
    assert metadata['kind'] == 'reviewed_development_source_zip'
    assert metadata['qualification_status'] == 'incomplete' and metadata['full_qualification'] is False
    assert metadata['sha256'] == inventory.REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256
    assert prepare.canonical(predecessor) == inventory.REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT_PREDECESSOR_SHA256
    assert prepare.canonical(predecessor['v24_job150615_development_review']) == inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_SHA256
    assert authenticate(predecessor, review_fixture(reviewed))['released'] is False


@pytest.mark.parametrize('field,value,error', [
    ('kind', 'qualified_development_source_zip', 'predecessor archive changed'),
    ('qualification_status', 'complete', 'incomplete qualification status'),
    ('full_qualification', True, 'incomplete qualification status'),
    ('full_qualification', 0, 'incomplete qualification status'),
    ('sha256', '0'*64, 'predecessor archive changed'),
    ('source_identity_count', 699, 'predecessor archive changed'),
])
def test_reauthentication_cannot_upgrade_or_substitute_the_reviewed_archive(reviewed, field, value, error):
    _files, _metadata, predecessor = reviewed
    review = review_fixture(reviewed)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match=error):
        authenticate(predecessor, review)


@pytest.mark.parametrize('key', ['v24_job150615_development_review', 'v24_guarded_rescue_development_review',
                               'v24_outer_crop_development_review', 'v24_0_0_release_review'])
def test_headroom_successor_cannot_rewrite_any_prior_receipt(reviewed, key):
    _files, _metadata, predecessor = reviewed
    altered = copy.deepcopy(predecessor)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, review_fixture(reviewed))


def test_reviewed_archive_cannot_enter_the_qualified_loader(tmp_path):
    with pytest.raises(ValueError, match='no qualified archive predecessor'):
        prepare.qualified_development_archive(ARCHIVE, development='job150615-headroom')
    if not ARCHIVE.is_file():
        pytest.skip('Retained archive is needed for byte tampering')
    changed = tmp_path/'modified-reviewed.zip'
    changed.write_bytes(ARCHIVE.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='independent SHA256'):
        prepare.reviewed_development_archive(changed)


def test_headroom_preparation_requires_the_reviewed_archive(tmp_path):
    with pytest.raises(ValueError, match='job150615-headroom requires --predecessor-archive from reviewed'):
        prepare.prepare(output_dir=tmp_path, development='job150615-headroom')


def test_private_headroom_publication_appends_without_qualifying_the_predecessor(reviewed, tmp_path, monkeypatch):
    files, _metadata, predecessor = reviewed
    if files is None:
        pytest.skip('Retained reviewed archive is needed for the private publication fixture')
    root, evidence = tmp_path/'repository', tmp_path/'evidence'
    for relative, payload in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root/'XTA/headroom_fixture.py').write_text('VALUE = 1\n', encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py'):
        (root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest = root/'release/_package_inventory.json'
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(inventory, 'ROOT', root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest)
    def fixture_git(command, **kwargs):
        assert command == ['git', 'rev-parse', 'v24.0.0^{commit}']
        return inventory.REVIEWED_V24_JOB150615_HEADROOM_DEVELOPMENT_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=evidence, development='job150615-headroom',
                             predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == predecessor
    assert result['written'] is True and published[KEY]['released'] is False
    source = published[KEY]['predecessor_source_archive']
    assert source['kind'] == 'reviewed_development_source_zip'
    assert source['qualification_status'] == 'incomplete' and source['full_qualification'] is False
