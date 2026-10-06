"""Release numbering uses actual lineage and cannot reuse an occupied identity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import check_release_version as guard


def git(root, *args):
    return subprocess.check_output(('git', *args), cwd=root, text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0')).strip()


def write_source(root, version):
    (root / 'XTA').mkdir(exist_ok=True)
    (root / 'XTA/__init__.py').write_text(f'__version__ = {version!r}\n', encoding='utf-8')
    for launcher in root.glob('*_SLURM.py'):
        launcher.unlink()
    (root / f'GPT-6-Astra-Ultra_v{version}_SLURM.py').write_text('# fixture launcher\n', encoding='utf-8')


def commit_source(root, version, *, tag=None, annotated=False):
    write_source(root, version)
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture ' + version)
    if tag:
        if annotated:
            git(root, 'tag', '-a', tag, '-m', 'fixture release')
        else:
            git(root, 'tag', tag)
    return git(root, 'rev-parse', 'HEAD')


def repository(tmp_path, version='1.2.3', *, tag=True, annotated=False):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q', '--initial-branch=main')
    git(root, 'config', 'user.name', 'Release guard fixture')
    git(root, 'config', 'user.email', 'fixture@example.com')
    git(root, 'config', 'core.autocrlf', 'false')
    commit_source(root, version, tag='v' + version if tag else None, annotated=annotated)
    return root


@pytest.mark.parametrize('target', ['1.2.4', '1.3.0', '2.0.0'])
def test_adjacent_planned_releases_use_head_inclusively(tmp_path, target):
    root = repository(tmp_path)
    result = guard.check_release_version(root, target_version=target)
    assert result['kind'] == 'planned-release' and result['release_ready'] is True
    assert result['previous_release'] == {'version': '1.2.3', 'tag': 'v1.2.3', 'commit': git(root, 'rev-parse', 'HEAD')}
    assert result['expected_next_versions'] == ['1.2.4', '1.3.0', '2.0.0']
    assert result['warnings'] == [] and result['acknowledgement'] is None
    assert result['target_version'] == target
    assert (root / 'XTA/__init__.py').read_text() == "__version__ = '1.2.3'\n"


@pytest.mark.parametrize('target', ['1.2.5', '1.4.0', '3.0.0', '2.0.1', '1.3.1'])
def test_missing_patch_minor_major_and_zero_releases_are_blocked(tmp_path, target):
    root = repository(tmp_path)
    with pytest.raises(guard.ReleaseVersionError, match='Numbering gap'):
        guard.check_release_version(root, target_version=target)


def test_gap_acknowledgement_names_exact_transition_and_reason(tmp_path):
    root = repository(tmp_path)
    result = guard.check_release_version(root, target_version='1.2.5',
        acknowledge_gap='v1.2.3:v1.2.5', reason='Retired an unpublished candidate')
    assert result['acknowledgement'] == {'transition': 'v1.2.3:v1.2.5', 'reason': 'Retired an unpublished candidate'}
    assert result['release_ready'] is True and 'acknowledged' in result['warnings'][0]
    assert result['numbering_gap']['missing_ranges'] == [
        {'level': 'patch', 'first': 'v1.2.4', 'last': 'v1.2.4', 'count': 1,
         'present_elsewhere': [], 'present_elsewhere_count': 0}]


def test_large_gaps_have_bounded_ranges_instead_of_enumerating_versions(tmp_path):
    root = repository(tmp_path)
    result = guard.check_release_version(root, target_version='1.999999999.0',
        acknowledge_gap='v1.2.3:v1.999999999.0', reason='Fixture exercises a large intentional gap')
    assert result['numbering_gap']['missing_ranges'] == [
        {'level': 'minor', 'first': 'v1.3.0', 'last': 'v1.999999998.0', 'count': 999999996,
         'present_elsewhere': [], 'present_elsewhere_count': 0}]
    assert len(result['warnings'][0]) < 400


def test_gap_names_skipped_major_and_the_target_major_zero_checkpoint(tmp_path):
    root = repository(tmp_path, version='22.3.2')
    result = guard.check_release_version(root, target_version='24.0.1',
        acknowledge_gap='v22.3.2:v24.0.1', reason='Fixture gap')
    assert 'major series v23.x' in result['warnings'][0]
    assert 'v23.0.0' in result['warnings'][0] and 'v24.0.0' in result['warnings'][0]
    assert result['scope'] == 'local-git-refs'


def test_skipped_checkpoint_present_on_another_lineage_is_reported_as_existing(tmp_path):
    root = repository(tmp_path, version='13.2.2')
    git(root, 'switch', '-qc', 'checkpoint')
    commit_source(root, '13.2.3', tag='v13.2.3')
    git(root, 'switch', '-q', 'main')
    result = guard.check_release_version(root, target_version='13.2.4',
        acknowledge_gap='v13.2.2:v13.2.4', reason='Different lineage intentionally')
    assert 'v13.2.3 exists elsewhere in local refs' in result['warnings'][0]
    assert 'no matching checkpoint' not in result['warnings'][0]
    assert result['numbering_gap']['missing_ranges'][0]['present_elsewhere'] == ['v13.2.3']


@pytest.mark.parametrize('ack,reason', [
    ('v1.2.2:v1.2.5', 'reason'), ('v1.2.3:v1.2.6', 'reason'),
    ('1.2.3:1.2.5', 'reason'), ('v1.2.3:v1.2.5', None), ('v1.2.3:v1.2.5', '  '),
])
def test_gap_acknowledgements_cannot_be_broad_or_reasonless(tmp_path, ack, reason):
    root = repository(tmp_path)
    with pytest.raises(guard.ReleaseVersionError, match='exact transition|nonempty reason'):
        guard.check_release_version(root, target_version='1.2.5', acknowledge_gap=ack, reason=reason)


def test_reason_requires_a_gap_and_backwards_versions_cannot_be_acknowledged(tmp_path):
    root = repository(tmp_path)
    with pytest.raises(guard.ReleaseVersionError, match='requires --acknowledge-gap'):
        guard.check_release_version(root, target_version='1.2.4', reason='reason')
    with pytest.raises(guard.ReleaseVersionError, match='only to a numbering gap'):
        guard.check_release_version(root, target_version='1.2.4', acknowledge_gap='v1.2.3:v1.2.4', reason='reason')
    with pytest.raises(guard.ReleaseVersionError, match='must advance'):
        guard.check_release_version(root, target_version='1.1.0', acknowledge_gap='v1.2.3:v1.1.0', reason='reason')


def test_existing_tagged_head_still_checks_its_transition(tmp_path):
    root = repository(tmp_path)
    head = commit_source(root, '1.2.5', tag='v1.2.5')
    with pytest.raises(guard.ReleaseVersionError, match='v1.2.3:v1.2.5'):
        guard.check_release_version(root)
    result = guard.check_release_version(root, acknowledge_gap='v1.2.3:v1.2.5', reason='intentional gap')
    assert result['kind'] == 'release' and result['head_commit'] == head
    assert result['previous_release']['version'] == '1.2.3'


def test_committed_untagged_release_passes_and_annotated_tags_are_peeled(tmp_path):
    root = repository(tmp_path, annotated=True)
    prior = git(root, 'rev-parse', 'HEAD')
    head = commit_source(root, '1.2.4')
    result = guard.check_release_version(root)
    assert result['head_commit'] == head and result['previous_release']['commit'] == prior
    git(root, 'tag', '-a', 'v1.2.4', '-m', 'release')
    assert guard.check_release_version(root)['head_commit'] == head


def test_nested_annotated_release_tag_resolves_to_commit(tmp_path):
    root = repository(tmp_path, tag=False)
    git(root, 'tag', '-a', 'annotation-object', '-m', 'inner')
    git(root, 'tag', '-a', 'v1.2.3', 'annotation-object', '-m', 'outer')
    assert guard.check_release_version(root, target_version='1.2.4')['previous_release']['commit'] == git(root, 'rev-parse', 'HEAD')


def test_planned_version_cannot_reuse_even_head_tag(tmp_path):
    root = repository(tmp_path)
    with pytest.raises(guard.ReleaseVersionError, match='identity collision'):
        guard.check_release_version(root, target_version='1.2.3')


def test_new_source_with_existing_version_is_an_identity_collision(tmp_path):
    root = repository(tmp_path)
    (root / 'source.py').write_text('NEW_SOURCE = True\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'changed source without a version bump')
    with pytest.raises(guard.ReleaseVersionError, match='identity collision'):
        guard.check_release_version(root, acknowledge_gap='v1.2.3:v1.2.3', reason='cannot override identity')
    result = guard.check_release_version(root, snapshot=True)
    assert result['release_ready'] is False and 'identity collision' in result['warnings'][0]


def test_branch_release_does_not_replace_first_parent_predecessor(tmp_path):
    root = repository(tmp_path)
    git(root, 'switch', '-qc', 'side')
    commit_source(root, '3.0.0', tag='v3.0.0')
    git(root, 'switch', '-q', 'main')
    result = guard.check_release_version(root, target_version='1.2.4')
    assert result['previous_release']['version'] == '1.2.3' and result['warnings'] == []


def test_branch_tag_collision_cannot_be_overridden_by_gap_acknowledgement(tmp_path):
    root = repository(tmp_path)
    git(root, 'switch', '-qc', 'side')
    commit_source(root, '1.2.5', tag='v1.2.5')
    git(root, 'switch', '-q', 'main')
    with pytest.raises(guard.ReleaseVersionError, match='identity collision'):
        guard.check_release_version(root, target_version='1.2.5', acknowledge_gap='v1.2.3:v1.2.5', reason='reason')


def test_case_variant_of_target_tag_is_an_identity_collision(tmp_path):
    root = repository(tmp_path)
    git(root, 'switch', '-qc', 'side')
    commit_source(root, '1.2.4', tag='V1.2.4')
    git(root, 'switch', '-q', 'main')
    with pytest.raises(guard.ReleaseVersionError, match='V1.2.4 .* identity collision'):
        guard.check_release_version(root, target_version='1.2.4')


def test_noncommit_target_tag_is_a_collision_without_rescanning_unrelated_tags(tmp_path):
    root = repository(tmp_path)
    blob = git(root, 'rev-parse', 'HEAD:XTA/__init__.py')
    git(root, 'tag', 'v9.0.0', blob)
    assert guard.check_release_version(root, target_version='1.2.4')['warnings'] == []
    git(root, 'tag', 'v1.2.4', blob)
    with pytest.raises(guard.ReleaseVersionError, match='non-commit object.*identity collision'):
        guard.check_release_version(root, target_version='1.2.4')


def test_merge_second_parent_tag_is_not_a_release_predecessor(tmp_path):
    root = repository(tmp_path)
    git(root, 'switch', '-qc', 'side')
    commit_source(root, '2.0.0', tag='v2.0.0')
    git(root, 'switch', '-q', 'main')
    commit_source(root, '1.2.4')
    git(root, 'merge', '-q', '--no-ff', '--strategy=ours', '-m', 'fixture merge', 'side')
    result = guard.check_release_version(root)
    assert result['previous_release']['version'] == '1.2.3'
    assert result['target_version'] == '1.2.4' and result['warnings'] == []


def test_nearest_commit_with_multiple_release_tags_is_ambiguous(tmp_path):
    root = repository(tmp_path)
    git(root, 'tag', 'v1.2.4')
    with pytest.raises(guard.ReleaseVersionError, match='Ambiguous predecessor'):
        guard.check_release_version(root, target_version='1.2.5', acknowledge_gap='v1.2.3:v1.2.5', reason='reason')


def test_predecessor_tag_must_agree_with_its_actual_source(tmp_path):
    root = repository(tmp_path, tag=False)
    git(root, 'tag', 'v1.2.2')
    with pytest.raises(guard.ReleaseVersionError, match='Invalid predecessor source'):
        guard.check_release_version(root, target_version='1.2.4', acknowledge_gap='v1.2.2:v1.2.4', reason='reason')
    with pytest.raises(guard.ReleaseVersionError, match='HEAD tag .* differs'):
        guard.check_release_version(root)


def test_old_numbering_gaps_and_nonrelease_tags_are_not_rescanned(tmp_path):
    root = repository(tmp_path, version='12.0.0')
    commit_source(root, '14.2.4', tag='v14.2.4')
    git(root, 'tag', 'v14-experiment-r1')
    git(root, 'tag', 'v014.2.4')
    result = guard.check_release_version(root, target_version='14.2.5')
    assert result['previous_release']['version'] == '14.2.4' and result['warnings'] == []


@pytest.mark.parametrize('version', ['0.0.0', '1.0.0'])
def test_initial_release_has_explicit_normal_starting_versions(tmp_path, version):
    root = repository(tmp_path, version=version, tag=False)
    result = guard.check_release_version(root)
    assert result['previous_release'] is None and result['expected_next_versions'] == ['0.0.0', '1.0.0']
    assert result['release_ready'] is True


def test_nonstandard_initial_release_requires_none_transition_acknowledgement(tmp_path):
    root = repository(tmp_path, tag=False)
    with pytest.raises(guard.ReleaseVersionError, match='NONE:v1.2.3'):
        guard.check_release_version(root)
    result = guard.check_release_version(root, acknowledge_gap='NONE:v1.2.3', reason='Importing an existing project')
    assert result['acknowledgement']['transition'] == 'NONE:v1.2.3'


def test_dirty_snapshot_can_retain_head_label_without_claiming_a_release(tmp_path):
    root = repository(tmp_path)
    (root / 'uncommitted.py').write_text('DIRTY = True\n')
    result = guard.check_release_version(root, snapshot=True)
    assert result['kind'] == 'development-snapshot' and result['release_ready'] is False
    assert result['previous_release']['commit'] == result['head_commit'] and result['warnings'] == []


def test_snapshot_numbering_gap_and_branch_collision_are_warnings(tmp_path):
    root = repository(tmp_path)
    git(root, 'switch', '-qc', 'side')
    commit_source(root, '1.2.5', tag='v1.2.5')
    git(root, 'switch', '-q', 'main')
    write_source(root, '1.2.5')
    result = guard.check_release_version(root, snapshot=True)
    assert result['release_ready'] is False
    assert any('identity collision' in warning for warning in result['warnings'])
    assert any('Numbering gap' in warning for warning in result['warnings'])


@pytest.mark.parametrize('version', ['01.0.0', '1.00.0', '1.0.00', 'v1.0.0', '1.0', '1.0.0-rc1'])
def test_invalid_source_versions_are_not_permitted_in_snapshots(tmp_path, version):
    root = repository(tmp_path)
    write_source(root, version)
    with pytest.raises(guard.ReleaseVersionError, match='Invalid release version'):
        guard.check_release_version(root, snapshot=True)


def test_source_requires_one_literal_version_and_matching_launcher(tmp_path):
    root = repository(tmp_path)
    (root / 'other_v1.2.3_SLURM.py').write_text('# duplicate\n')
    with pytest.raises(guard.ReleaseVersionError, match='exactly one versioned root launcher'):
        guard.check_release_version(root, snapshot=True)
    (root / 'other_v1.2.3_SLURM.py').unlink()
    (root / 'XTA/__init__.py').write_text("__version__ = '1.2.4'\n")
    with pytest.raises(guard.ReleaseVersionError, match='launcher .* differs'):
        guard.check_release_version(root, snapshot=True)
    (root / 'XTA/__init__.py').write_text("__version__ = '1.2.3'\n__version__ = '1.2.3'\n")
    with pytest.raises(guard.ReleaseVersionError, match='exactly one literal'):
        guard.check_release_version(root, snapshot=True)


def test_snapshot_and_planned_modes_cannot_be_combined(tmp_path):
    root = repository(tmp_path)
    with pytest.raises(guard.ReleaseVersionError, match='different source modes'):
        guard.check_release_version(root, snapshot=True, target_version='1.2.4')


def test_shallow_release_fails_closed_but_snapshot_records_limitation(tmp_path):
    root = repository(tmp_path)
    clone = tmp_path / 'shallow'
    git(tmp_path, 'clone', '-q', '--depth=1', '--branch=main', root.as_uri(), str(clone))
    with pytest.raises(guard.ReleaseVersionError, match='Shallow Git history'):
        guard.check_release_version(clone, acknowledge_gap='NONE:v1.2.3', reason='cannot override shallow history')
    result = guard.check_release_version(clone, snapshot=True)
    assert result['release_ready'] is False and any('Shallow' in warning for warning in result['warnings'])


def test_absent_git_or_repository_cannot_establish_a_release(tmp_path, monkeypatch):
    with pytest.raises(guard.ReleaseVersionError, match='Cannot read release Git history'):
        guard.check_release_version(tmp_path)
    def absent(*args, **kwargs):
        raise FileNotFoundError('Git is not installed')
    monkeypatch.setattr(guard.subprocess, 'run', absent)
    with pytest.raises(guard.ReleaseVersionError, match='Git is not installed'):
        guard.check_release_version(tmp_path)


def test_checker_git_commands_disable_optional_locks_and_do_not_mutate_repo(tmp_path, monkeypatch):
    root = repository(tmp_path)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    original_run = guard.subprocess.run
    calls = []
    def tracked(command, **kwargs):
        calls.append(command[1])
        assert kwargs['env']['GIT_OPTIONAL_LOCKS'] == '0'
        return original_run(command, **kwargs)
    monkeypatch.setattr(guard.subprocess, 'run', tracked)
    guard.check_release_version(root, target_version='1.2.4')
    after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert after == before
    assert set(calls) <= {'rev-parse', 'for-each-ref', 'cat-file', 'rev-list', 'ls-tree', 'show'}


def test_cli_json_reports_valid_proposal_and_exact_failure(tmp_path):
    root = repository(tmp_path)
    command = [sys.executable, '-B', str(Path(guard.__file__).resolve()), '--root', str(root), '--json']
    accepted = subprocess.run([*command, '--target-version', '1.3.0'], capture_output=True, text=True)
    assert accepted.returncode == 0 and json.loads(accepted.stdout)['kind'] == 'planned-release'
    rejected = subprocess.run([*command, '--target-version', '2.0.1'], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'v1.2.3:v2.0.1' in json.loads(rejected.stdout)['error']
