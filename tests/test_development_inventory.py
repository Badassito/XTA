"""Development source review must preserve a released tag and package identity."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory


KEY = 'v24_outer_crop_development_review'
PREFIX = 'REVIEWED_V24_OUTER_CROP_DEVELOPMENT'


@pytest.fixture(scope='module')
def released():
    source = prepare.git_file(prepare.ROOT, inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_COMMIT,
                              'release/_package_inventory.json')
    return json.loads(source)


def fixture_review(released):
    prior_tools = released['v24_0_0_release_review']['validation_tools']
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Retain independently authenticated tagged tool.') for row in prior_tools]
    tools.extend(dict(path=path, previous_sha256=value, sha256='a' * 64,
                      reason='Reviewed development tool.') for path, value in
                 inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_ADDED_VALIDATION_TOOLS.items())
    return dict(release='24.0.0', kind='development', development='outer-crop', package_version='24.0.0',
        released=False, predecessor_tag='v24.0.0', feature='sam-outer-crop-development',
        previous_review_sha256=inventory.REVIEWED_V24_0_0_RELEASE_SHA256,
        predecessor_commit=inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(released), definitions=[], statements=[],
        removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[],
        complete_modules=[], module_snapshots=[], validation_tools=tools)


def authenticate(released, review, *, pins=None, removals=None):
    with mock.patch.object(inventory, PREFIX + '_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX + '_PREDECESSOR_MODULES', pins or {}), \
         mock.patch.object(inventory, PREFIX + '_REMOVALS', removals or {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v24_outer_crop_development_contract(
            {**released, KEY: review}, released['v21_review'])


def test_development_predecessor_is_exact_tagged_release(released):
    assert prepare.canonical(released) == inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_SHA256
    assert inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_COMMIT == 'b0152ec399578070d7c2adcf7c81a939e8b9986e'
    assert 'v24_0_1_release_review' not in released
    assert released['v24_0_0_release_review']['release'] == '24.0.0'
    assert authenticate(released, fixture_review(released))['kind'] == 'development'


def test_reauthenticated_development_cannot_rewrite_released_history(released):
    altered = copy.deepcopy(released)
    altered['v24_0_0_release_review']['feature'] = 'unreviewed'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, fixture_review(released))


@pytest.mark.parametrize('field,value', [('kind', 'release'), ('released', True),
                                      ('package_version', '24.0.1'), ('development', 'other'),
                                      ('predecessor_tag', 'v24.0.1')])
def test_development_labels_cannot_claim_a_new_release(released, field, value):
    review = fixture_review(released)
    review[field] = value
    with pytest.raises(RuntimeError, match='labelled development'):
        authenticate(released, review)


@pytest.mark.parametrize('module,binding', [('__init__', '__version__'), ('cli', 'SCRIPT_VERSION'),
                                          ('cli', 'SCRIPT_BASENAME'), ('config', 'SCRIPT_VERSION'),
                                          ('config', 'SCRIPT_VERSION_COMPACT'), ('config', 'SCRIPT_BASENAME')])
def test_development_review_cannot_supersede_release_identity(released, module, binding):
    review = fixture_review(released)
    review['statements'].append(dict(module=module, label='unreviewed_release_change', binding=binding,
        sha256='a' * 64, previous_sha256=None, reason='Invalid attempted version bump.'))
    with pytest.raises(RuntimeError, match='cannot change the package or launcher'):
        authenticate(released, review)


def test_independent_tool_predecessor_links_are_not_self_reauthenticated(released):
    review = fixture_review(released)
    review['validation_tools'][0]['previous_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='validation-tool review has invalid predecessor'):
        authenticate(released, review)
    review = fixture_review(released)
    review['validation_tools'].pop()
    with pytest.raises(RuntimeError, match='missing or duplicate paths'):
        authenticate(released, review)


def test_new_source_pin_and_exact_statement_positions_are_independent(released):
    source = 'VALUE = 2\n'
    pin, snapshot, records = prepare.review_module('development_fixture', None, source,
        complete=True, labels_by_hash={}, reason='Reviewed new development fixture.')
    review = fixture_review(released)
    review.update(module_snapshots=[snapshot], complete_modules=['development_fixture'], **records)
    assert authenticate(released, review, pins={'development_fixture': pin})['module_snapshots'] == [snapshot]
    altered = copy.deepcopy(review)
    altered['module_snapshots'][0]['previous_ast_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='source predecessor changed'):
        authenticate(released, altered, pins={'development_fixture': pin})
    altered = copy.deepcopy(review)
    altered['statements'][0]['current_index'] = 99
    with pytest.raises(RuntimeError, match='statement position differs'):
        authenticate(released, altered, pins={'development_fixture': pin})


def test_tagged_release_write_fails_before_development_source_is_examined(tmp_path):
    with mock.patch.object(prepare, 'git_file', side_effect=AssertionError('Source must not be read')):
        with pytest.raises(ValueError, match='already tagged'):
            prepare.prepare(output_dir=tmp_path, release='24.0.0', write=True)


def test_development_tag_identity_must_match_independent_pin(tmp_path):
    with mock.patch.object(prepare.subprocess, 'check_output', return_value='0' * 40):
        with pytest.raises(ValueError, match='independently pinned'):
            prepare.prepare(output_dir=tmp_path, development='outer-crop')


def test_development_drafts_belong_outside_repository_and_have_separate_cli_selection():
    with pytest.raises(ValueError, match='outside the repository'):
        prepare.prepare(output_dir=prepare.ROOT / 'generated_dev', development='outer-crop')
    with pytest.raises(ValueError, match='not both'):
        prepare.prepare(output_dir=prepare.ROOT.parent / 'Scratch', release='23.0.5', development='outer-crop')


def test_only_development_pin_assignments_change():
    source = ("RELEASED_SHA256 = 'immutable'\nDEV_SHA256 = ''\nDEV_PREDECESSOR_MODULES = {}\n"
              "DEV_REMOVALS = {}\nPACKAGE_VERSION = '24.0.0'\n")
    updated = prepare._update_verifier_pins(source, 'DEV', 'a' * 64,
        {'new': {'ast_sha256': None}}, {'definitions': (), 'statements': ()})
    before, after = ast.parse(source), ast.parse(updated)
    assert inventory.digest(before.body[0]) == inventory.digest(after.body[0])
    assert inventory.digest(before.body[-1]) == inventory.digest(after.body[-1])


def test_development_spec_inherits_all_released_tool_checks_and_adds_prepare_audit(released):
    spec = prepare.DEVELOPMENTS['outer-crop']
    inherited = tuple(row['path'] for row in released['v24_0_0_release_review']['validation_tools'])
    assert spec['validation_tools'][:len(inherited)] == inherited
    assert spec['validation_tools'][len(inherited):] == ('tools/prepare_reconciliation_release.py',) + prepare.OUTER_CROP_DEVELOPMENT_TOOLS
    assert tuple(inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_ADDED_VALIDATION_TOOLS) == spec['validation_tools'][len(inherited):]
    tag_source = prepare.git_file(prepare.ROOT, inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_COMMIT,
                                 'tools/prepare_reconciliation_release.py')
    assert hashlib.sha256(tag_source.encode()).hexdigest() == inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_ADDED_VALIDATION_TOOLS[
        'tools/prepare_reconciliation_release.py']


def test_development_write_preserves_tagged_inventory_and_only_changes_dev_pins(released, tmp_path, monkeypatch):
    """Exercise publication in a private repository-shaped fixture, not the checkout."""
    real_root = prepare.ROOT
    real_git_file = prepare.git_file
    commit = inventory.REVIEWED_V24_OUTER_CROP_DEVELOPMENT_PREDECESSOR_COMMIT
    fake_root = tmp_path / 'fixture_repository'
    evidence = tmp_path / 'scratch_evidence'
    fake_manifest = fake_root / 'release' / '_package_inventory.json'
    fake_manifest.parent.mkdir(parents=True)
    fake_manifest.write_text(json.dumps(released), encoding='utf-8')
    package = fake_root / 'XTA'
    package.mkdir()
    (package / 'development_fixture.py').write_text('VALUE = 2\n', encoding='utf-8')

    earlier = tuple(value for name, value in released.items() if name != 'v21_review'
                    and isinstance(value, dict) and 'release' in value and 'definitions' in value)
    radial_modules = set(inventory.reviewed_radial_module_hashes(released['v21_review'], earlier))
    radial_modules.update(module for module, _ in inventory.reviewed_radial_definition_hashes(
        released['v21_review'], earlier))
    tagged_sources = {'release/_package_inventory.json': json.dumps(released)}
    for module in radial_modules:
        relative = f'XTA/{module}.py'
        text = real_git_file(real_root, commit, relative)
        tagged_sources[relative] = text
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    for relative in prepare.DEVELOPMENTS['outer-crop']['validation_tools']:
        text = real_git_file(real_root, commit, relative)
        tagged_sources[relative] = text
        target = fake_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text if text is not None else (real_root / relative).read_text(encoding='utf-8'), encoding='utf-8')
    # The builder itself is a development tool; its released hash remains the predecessor.
    (fake_root / 'tools' / 'prepare_reconciliation_release.py').write_text(
        (real_root / 'tools' / 'prepare_reconciliation_release.py').read_text(encoding='utf-8'), encoding='utf-8')
    verifier = fake_root / 'tools' / 'verify_package_inventory.py'
    original_verifier = (real_root / 'tools' / 'verify_package_inventory.py').read_text(encoding='utf-8')
    verifier.write_text(original_verifier, encoding='utf-8')

    def git_snapshot(_root, supplied_commit, relative):
        assert supplied_commit == commit
        return tagged_sources.get(relative)

    def git_command(command, **_kwargs):
        if command[:2] == ['git', 'rev-parse']:
            return commit
        if command[:3] == ['git', 'diff', '--name-only']:
            return 'XTA/development_fixture.py\n'
        if command[:2] == ['git', 'ls-files']:
            return ''
        raise AssertionError(f'Unexpected Git operation: {command}')

    monkeypatch.setattr(prepare, 'ROOT', fake_root)
    monkeypatch.setattr(inventory, 'ROOT', fake_root)
    monkeypatch.setattr(inventory, 'MANIFEST', fake_manifest)
    monkeypatch.setattr(prepare, 'git_file', git_snapshot)
    monkeypatch.setattr(prepare.subprocess, 'check_output', git_command)
    result = prepare.prepare(output_dir=evidence, development='outer-crop', write=True)
    published = json.loads(fake_manifest.read_text(encoding='utf-8'))
    assert result['kind'] == 'development' and result['written'] is True
    assert {key: value for key, value in published.items() if key != KEY} == released
    assert published[KEY]['package_version'] == '24.0.0' and published[KEY]['released'] is False
    assert (evidence / 'development_review_draft.json').is_file()
    assert not (evidence / 'release_review_draft.json').exists()
    before = {node.targets[0].id: inventory.digest(node) for node in ast.parse(original_verifier).body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    after = {node.targets[0].id: inventory.digest(node) for node in ast.parse(verifier.read_text(encoding='utf-8')).body
             if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {PREFIX + '_SHA256', PREFIX + '_PREDECESSOR_MODULES'}
