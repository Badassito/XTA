"""The cluster-job successor preserves every qualified development receipt."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path('C:\\Users\\Bry\\Documents\\ChatGPT\\Scratch\\Experiments\\SAM_Guarded_Rescue_20261001\\final_qualification\\source\\XTA_v25.0.0_complete_source.zip')
KEY = 'v24_job150615_development_review'
PREFIX = 'REVIEWED_V24_JOB150615_DEVELOPMENT'


@pytest.fixture(scope='module')
def qualified():
    if ARCHIVE.is_file():
        files, metadata = prepare.qualified_development_archive(ARCHIVE, development='job150615')
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    predecessor = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    predecessor.pop('v24_job150790_150798_throughput_development_review', None)
    predecessor.pop('v24_0_2_sam_bridges_development_review', None)
    predecessor.pop('v24_job150772_performance_development_review', None)
    predecessor.pop('v24_0_1_release_review', None)
    predecessor.pop('v24_job150615_headroom_development_review', None)
    predecessor.pop(KEY, None)
    metadata = dict(kind='qualified_development_source_zip',
        sha256=inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256,
        source_identity_count=685, filename='XTA_v25.0.0_complete_source.zip')
    return None, metadata, predecessor


def review_fixture(qualified):
    _files, metadata, predecessor = qualified
    previous = predecessor['v24_guarded_rescue_development_review']
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Preserve the qualified guarded-rescue tool.') for row in previous['validation_tools']]
    tools.extend(dict(path=path, previous_sha256=prior,
                      sha256=hashlib.sha256((prepare.ROOT/path).read_text(encoding='utf-8').encode()).hexdigest(),
                      reason='Add the explicit long-session qualification helper.')
                 for path, prior in getattr(inventory, PREFIX+'_ADDED_VALIDATION_TOOLS').items())
    return dict(release='24.0.0', kind='development', development='job150615', package_version='24.0.0',
        released=False, predecessor_tag='v24.0.0', feature='sam-job150615-development',
        previous_review_sha256=inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_SHA256,
        predecessor_commit=inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(predecessor), predecessor_source_archive=metadata,
        definitions=[], statements=[], removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[], complete_modules=[],
        module_snapshots=[], validation_tools=tools)


def authenticate(predecessor, review, pins=None):
    removals = {
        'definitions': tuple(sorted((row['module'], row['name'], row['previous_index'], row['previous_sha256'])
                                    for row in review['removed_definitions'])),
        'statements': tuple(sorted((row['module'], row['previous_index'], row['previous_sha256'])
                                   for row in review['removed_statements'])),
    }
    with mock.patch.object(inventory, PREFIX+'_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX+'_PREDECESSOR_MODULES', pins or {}), \
         mock.patch.object(inventory, PREFIX+'_REMOVALS', removals):
        return inventory.reviewed_v24_job150615_development_contract(
            {**predecessor, KEY: review}, predecessor['v21_review'])


def test_qualified_guarded_archive_and_prior_receipts_have_independent_pins(qualified):
    files, metadata, predecessor = qualified
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 685
    assert metadata['sha256'] == inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256
    assert prepare.canonical(predecessor) == inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_PREDECESSOR_SHA256
    assert prepare.canonical(predecessor['v24_guarded_rescue_development_review']) == inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_SHA256
    assert authenticate(predecessor, review_fixture(qualified))['released'] is False


@pytest.mark.parametrize('key', ['v24_guarded_rescue_development_review', 'v24_outer_crop_development_review',
                               'v24_0_0_release_review'])
def test_job_successor_cannot_reauthenticate_changed_history(qualified, key):
    _files, _metadata, predecessor = qualified
    altered = copy.deepcopy(predecessor)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, review_fixture(qualified))


@pytest.mark.parametrize('field,value', [('sha256', '0'*64), ('source_identity_count', 684),
                                      ('kind', 'unqualified_working_directory')])
def test_job_successor_rejects_substituted_archive_metadata(qualified, field, value):
    _files, _metadata, predecessor = qualified
    review = review_fixture(qualified)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match='qualified predecessor archive changed'):
        authenticate(predecessor, review)


def test_job_archive_payload_tampering_is_rejected(tmp_path):
    if not ARCHIVE.is_file():
        pytest.skip('Retained qualified archive is unavailable')
    changed = tmp_path/'changed.zip'
    changed.write_bytes(ARCHIVE.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='independent SHA256'):
        prepare.qualified_development_archive(changed, development='job150615')
    with pytest.raises(ValueError, match='independent SHA256'):
        prepare.qualified_development_archive(ARCHIVE)


def test_job_tool_link_cannot_skip_the_guarded_predecessor(qualified):
    _files, _metadata, predecessor = qualified
    review = review_fixture(qualified)
    review['validation_tools'][0]['previous_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='invalid predecessor'):
        authenticate(predecessor, review)


def test_job_statement_positions_bind_the_qualified_source(qualified):
    _files, _metadata, predecessor = qualified
    pin, snapshot, records = prepare.review_module('job_fixture', 'VALUE = 1\n', 'VALUE = 2\n',
        complete=True, labels_by_hash={}, reason='Reviewed job fixture.')
    review = review_fixture(qualified)
    review.update(module_snapshots=[snapshot], complete_modules=['job_fixture'], **records)
    assert authenticate(predecessor, review, {'job_fixture': pin})['module_snapshots'] == [snapshot]
    altered = copy.deepcopy(review)
    altered['statements'][0]['previous_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='predecessor changed'):
        authenticate(predecessor, altered, {'job_fixture': pin})


def test_three_development_successors_rewind_to_the_tagged_source():
    values = ('VALUE = 0\n', 'VALUE = 1\n', 'VALUE = 2\n', 'VALUE = 3\n')
    chain = []
    for before, after in zip(values, values[1:]):
        _pin, snapshot, records = prepare.review_module('fixture', before, after,
            complete=True, labels_by_hash={}, reason='Independent successor.')
        chain.append({**records, 'module_snapshots': [snapshot]})
    restored = inventory.rewind_reviewed_statements('fixture', ast.parse(values[-1]).body, chain)
    assert [value for _node, value in restored] == chain[0]['module_snapshots'][0]['previous_top_level']


def test_published_guarded_review_cannot_be_overwritten(tmp_path):
    previous_archive = ARCHIVE.parent.parent.parent.parent/'SAM_Outer_Crop_20261001'/'final_qualification'/'source'/'XTA_v25.0.0_complete_source.zip'
    if not previous_archive.is_file():
        pytest.skip('Retained outer-crop archive is needed for preparation')
    with pytest.raises(ValueError, match='Published development reviews are immutable'):
        prepare.prepare(output_dir=tmp_path, development='guarded-rescue',
                        predecessor_archive=previous_archive, write=True)


def test_job_preparation_requires_its_qualified_archive(tmp_path):
    with pytest.raises(ValueError, match='job150615 requires --predecessor-archive'):
        prepare.prepare(output_dir=tmp_path, development='job150615')


def test_private_job_publication_only_appends_the_new_receipt(qualified, tmp_path, monkeypatch):
    files, _metadata, predecessor = qualified
    if files is None:
        pytest.skip('Retained archive is needed for a private source fixture')
    root, evidence = tmp_path/'repository', tmp_path/'evidence'
    for relative, payload in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root/'XTA/job_fixture.py').write_text('VALUE = 3\n', encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py',
                     'tools/qualify_release.py', 'tools/qualify_sam_interpolation_long_session.py'):
        (root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest = root/'release/_package_inventory.json'
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(inventory, 'ROOT', root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest)
    def fixture_git(command, **kwargs):
        assert command == ['git', 'rev-parse', 'v24.0.0^{commit}']
        return inventory.REVIEWED_V24_JOB150615_DEVELOPMENT_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=evidence, development='job150615', predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == predecessor
    assert result['written'] is True and published[KEY]['released'] is False
    previous_tools = {row['path']: row['sha256'] for row in predecessor['v24_guarded_rescue_development_review']['validation_tools']}
    qualifier = next(row for row in published[KEY]['validation_tools'] if row['path'] == 'tools/qualify_release.py')
    assert qualifier['previous_sha256'] == previous_tools[qualifier['path']]
    assert qualifier['sha256'] != qualifier['previous_sha256']
    long_session = next(row for row in published[KEY]['validation_tools']
                        if row['path'] == 'tools/qualify_sam_interpolation_long_session.py')
    assert long_session['previous_sha256'] is None
    with pytest.raises(ValueError, match='Published development reviews are immutable'):
        prepare.prepare(output_dir=tmp_path/'second-evidence', development='job150615',
                        predecessor_archive=ARCHIVE, write=True)
