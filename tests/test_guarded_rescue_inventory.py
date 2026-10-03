"""A development successor must preserve its independently qualified history."""
import ast
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path('C:\\Users\\Bry\\Documents\\ChatGPT\\Scratch\\Experiments\\SAM_Outer_Crop_20261001\\final_qualification\\source\\XTA_v25.0.0_complete_source.zip')
KEY = 'v24_guarded_rescue_development_review'
PREFIX = 'REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT'


@pytest.fixture(scope='module')
def qualified():
    if ARCHIVE.is_file():
        files, metadata = prepare.qualified_development_archive(ARCHIVE)
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    # Chain/tamper tests remain useful in portable source bundles. The current
    # inventory embeds the entire prior receipt; no host Scratch path is needed.
    predecessor = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    predecessor.pop('v24_job150790_150798_throughput_development_review', None)
    predecessor.pop('v24_job150772_performance_development_review', None)
    predecessor.pop('v24_0_1_release_review', None)
    predecessor.pop('v24_job150615_headroom_development_review', None)
    predecessor.pop('v24_job150615_development_review', None)
    predecessor.pop(KEY, None)
    metadata = dict(kind='qualified_development_source_zip',
        sha256=inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256,
        source_identity_count=670, filename='XTA_v25.0.0_complete_source.zip')
    return None, metadata, predecessor


def review_fixture(qualified):
    _files, metadata, predecessor = qualified
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Preserve an independently qualified development tool.')
             for row in predecessor['v24_outer_crop_development_review']['validation_tools']]
    return dict(release='24.0.0', kind='development', development='guarded-rescue', package_version='24.0.0',
        released=False, predecessor_tag='v24.0.0', feature='sam-guarded-rescue-development',
        previous_review_sha256=inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_SHA256,
        predecessor_commit=inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_PREDECESSOR_COMMIT,
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
        return inventory.reviewed_v24_guarded_rescue_development_contract(
            {**predecessor, KEY: review}, predecessor['v21_review'])


def test_qualified_archive_and_both_prior_receipts_are_independently_pinned(qualified):
    files, metadata, predecessor = qualified
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 670
    assert metadata['sha256'] == inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256
    assert prepare.canonical(predecessor) == inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_PREDECESSOR_SHA256
    assert prepare.canonical(predecessor['v24_outer_crop_development_review']) == inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_SHA256
    assert authenticate(predecessor, review_fixture(qualified))['released'] is False


def test_archive_tampering_is_rejected_before_source_review(qualified, tmp_path):
    if not ARCHIVE.is_file():
        pytest.skip('Actual predecessor ZIP is needed for archive mutation qualification')
    changed = tmp_path/'modified-predecessor.zip'
    changed.write_bytes(ARCHIVE.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='independent SHA256'):
        prepare.qualified_development_archive(changed)


def test_guarded_successor_cannot_reauthenticate_changed_outer_or_tagged_history(qualified):
    _files, _metadata, predecessor = qualified
    for key in ('v24_outer_crop_development_review', 'v24_0_0_release_review'):
        altered = copy.deepcopy(predecessor)
        altered[key]['feature'] = 'rewritten-history'
        with pytest.raises(RuntimeError, match='predecessor inventory changed'):
            authenticate(altered, review_fixture(qualified))


@pytest.mark.parametrize('field,value', [('sha256', '0'*64), ('source_identity_count', 669),
                                      ('kind', 'unqualified_working_directory')])
def test_successor_cannot_substitute_a_different_predecessor_archive(qualified, field, value):
    _files, _metadata, predecessor = qualified
    review = review_fixture(qualified)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match='qualified predecessor archive changed'):
        authenticate(predecessor, review)


def test_guarded_review_tools_have_exact_outer_crop_predecessors(qualified):
    _files, _metadata, predecessor = qualified
    review = review_fixture(qualified)
    review['validation_tools'][0]['previous_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='invalid predecessor'):
        authenticate(predecessor, review)


def test_successor_source_positions_are_checked_against_qualified_predecessor(qualified):
    _files, _metadata, predecessor = qualified
    old, new = 'VALUE = 1\n', 'VALUE = 2\n'
    pin, snapshot, records = prepare.review_module('guarded_fixture', old, new,
        complete=True, labels_by_hash={}, reason='Reviewed successor fixture.')
    review = review_fixture(qualified)
    review.update(module_snapshots=[snapshot], complete_modules=['guarded_fixture'], **records)
    assert authenticate(predecessor, review, {'guarded_fixture': pin})['module_snapshots'] == [snapshot]
    wrong = copy.deepcopy(review)
    wrong['statements'][0]['previous_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='predecessor changed'):
        authenticate(predecessor, wrong, {'guarded_fixture': pin})


def test_source_rewind_preserves_the_prior_development_and_release_sequence():
    old, outer, guarded = 'VALUE = 1\n', 'VALUE = 2\n', 'VALUE = 3\n'
    _, s1, r1 = prepare.review_module('fixture', old, outer, complete=True,
        labels_by_hash={}, reason='Original development.')
    _, s2, r2 = prepare.review_module('fixture', outer, guarded, complete=True,
        labels_by_hash={}, reason='Successor development.')
    chain = ({**r1, 'module_snapshots': [s1]}, {**r2, 'module_snapshots': [s2]})
    restored = inventory.rewind_reviewed_statements('fixture', ast.parse(guarded).body, chain)
    assert [value for _node, value in restored] == s1['previous_top_level']


def test_published_outer_crop_review_cannot_be_overwritten(tmp_path):
    with pytest.raises(ValueError, match='Published development reviews are immutable'):
        prepare.prepare(output_dir=tmp_path, development='outer-crop', write=True)


def test_guarded_preparation_requires_the_qualified_archive(tmp_path):
    with pytest.raises(ValueError, match='requires --predecessor-archive'):
        prepare.prepare(output_dir=tmp_path, development='guarded-rescue')


def test_successor_publication_appends_without_rewriting_qualified_receipts(qualified, tmp_path, monkeypatch):
    files, _metadata, predecessor = qualified
    if files is None:
        pytest.skip('Actual predecessor ZIP is needed to construct a private source fixture')
    fake_root, evidence = tmp_path/'repository', tmp_path/'scratch_evidence'
    for relative, payload in files.items():
        path = fake_root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (fake_root/'XTA'/'guarded_fixture.py').write_text('VALUE = 3\n', encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py'):
        (fake_root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest_path = fake_root/'release'/'_package_inventory.json'
    verifier_path = fake_root/'tools'/'verify_package_inventory.py'
    before = verifier_path.read_text(encoding='utf-8')
    monkeypatch.setattr(prepare, 'ROOT', fake_root)
    monkeypatch.setattr(inventory, 'ROOT', fake_root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest_path)
    def fixture_git(command, **kwargs):
        assert command == ['git', 'rev-parse', 'v24.0.0^{commit}']
        return inventory.REVIEWED_V24_GUARDED_RESCUE_DEVELOPMENT_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=evidence, development='guarded-rescue',
                             predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == predecessor
    assert published[KEY]['development'] == 'guarded-rescue' and published[KEY]['released'] is False
    assert result['written'] and result['kind'] == 'development'
    old_assignments = {node.targets[0].id: inventory.digest(node) for node in ast.parse(before).body
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name)}
    new_assignments = {node.targets[0].id: inventory.digest(node) for node in ast.parse(verifier_path.read_text()).body
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name)}
    changed_pins = {PREFIX+'_SHA256', PREFIX+'_PREDECESSOR_MODULES'}
    if getattr(inventory, PREFIX+'_REMOVALS') != {'definitions': (), 'statements': ()}:
        changed_pins.add(PREFIX+'_REMOVALS')
    assert {name for name in old_assignments if old_assignments[name] != new_assignments[name]} == changed_pins
