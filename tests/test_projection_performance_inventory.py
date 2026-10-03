"""Performance development preserves the fully qualified source predecessor."""
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path(r'C:\Users\Bry\Documents\ChatGPT\Scratch\Experiments\Projection_Coverage_v25_0_1_20261001\final_qualification_v4\source\XTA_v25.0.1_complete_source.zip')
KEY = 'v25_job150772_performance_development_review'
PREFIX = 'REVIEWED_V25_JOB150772_PERFORMANCE_DEVELOPMENT'


@pytest.fixture(scope='module')
def predecessor():
    if ARCHIVE.is_file():
        files, metadata = prepare.qualified_development_archive(ARCHIVE, development='job150772-performance')
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    prior = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    prior.pop('v25_job150790_150798_throughput_development_review', None)
    prior.pop(KEY, None)
    metadata = dict(kind='qualified_development_source_zip', qualification_status='passed',
        full_qualification=True, package_version='25.0.1', source_identity_count=715,
        sha256=inventory.REVIEWED_V25_JOB150772_PERFORMANCE_DEVELOPMENT_PREDECESSOR_ARCHIVE_SHA256,
        filename='XTA_v25.0.1_complete_source.zip')
    return None, metadata, prior


def review_fixture(predecessor):
    _files, metadata, prior = predecessor
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Preserve the qualified source tool.') for row in prior['v25_0_1_release_review']['validation_tools']]
    tools.extend(dict(path=path, previous_sha256=previous,
                      sha256=hashlib.sha256((prepare.ROOT/path).read_text(encoding='utf-8').encode()).hexdigest(),
                      reason='Add native projection performance qualification.')
                 for path, previous in getattr(inventory, PREFIX+'_ADDED_VALIDATION_TOOLS').items())
    return dict(release='25.0.1', kind='development', development='job150772-performance',
        package_version='25.0.1', released=False, predecessor_tag='v25.0.0',
        feature='sam-job150772-performance-development',
        previous_review_sha256=inventory.REVIEWED_V25_0_1_RELEASE_SHA256,
        predecessor_commit=inventory.REVIEWED_V25_JOB150772_PERFORMANCE_DEVELOPMENT_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(prior), predecessor_source_archive=metadata,
        definitions=[], statements=[], removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[], complete_modules=[],
        module_snapshots=[], validation_tools=tools)


def authenticate(prior, review):
    with mock.patch.object(inventory, PREFIX+'_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX+'_PREDECESSOR_MODULES', {}), \
         mock.patch.object(inventory, PREFIX+'_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v25_job150772_performance_development_contract(
            {**prior, KEY: review}, prior['v21_review'])


def test_performance_predecessor_has_completed_qualification_and_exact_source_identity(predecessor):
    files, metadata, prior = predecessor
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 715
    assert metadata['qualification_status'] == 'passed' and metadata['full_qualification'] is True
    assert prepare.canonical(prior) == inventory.REVIEWED_V25_JOB150772_PERFORMANCE_DEVELOPMENT_PREDECESSOR_SHA256
    assert prepare.canonical(prior['v25_0_1_release_review']) == inventory.REVIEWED_V25_0_1_RELEASE_SHA256
    assert authenticate(prior, review_fixture(predecessor))['kind'] == 'development'


@pytest.mark.parametrize('field,value,error', [
    ('kind', 'reviewed_development_source_zip', 'predecessor archive changed'),
    ('full_qualification', False, 'completed v25.0.1 qualification'),
    ('full_qualification', 1, 'completed v25.0.1 qualification'),
    ('qualification_status', 'incomplete', 'completed v25.0.1 qualification'),
    ('sha256', '0'*64, 'predecessor archive changed'),
])
def test_performance_predecessor_cannot_be_relabelled_or_substituted(predecessor, field, value, error):
    _files, _metadata, prior = predecessor
    review = review_fixture(predecessor)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match=error):
        authenticate(prior, review)


@pytest.mark.parametrize('key', ['v25_0_1_release_review', 'v25_job150615_headroom_development_review',
                               'v25_0_0_release_review'])
def test_performance_successor_cannot_rewrite_earlier_history(predecessor, key):
    _files, _metadata, prior = predecessor
    altered = copy.deepcopy(prior)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, review_fixture(predecessor))


def test_performance_archive_tampering_is_rejected(tmp_path):
    if not ARCHIVE.is_file():
        pytest.skip('Actual qualified source ZIP is needed for byte mutation')
    changed = tmp_path/'changed.zip'
    changed.write_bytes(ARCHIVE.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='independent SHA256'):
        prepare.qualified_development_archive(changed, development='job150772-performance')


def test_performance_audit_requires_the_qualified_source_zip(tmp_path):
    with pytest.raises(ValueError, match='requires --predecessor-archive'):
        prepare.prepare(output_dir=tmp_path, development='job150772-performance')


def seam_successor(before, after):
    return dict(definitions=[dict(module='fixture', name='replacement', previous_sha256=before,
        sha256=after, previous_index=0, current_index=0, reason='Explicit successor.')],
        module_snapshots=[dict(module='fixture', previous_top_level=[before], top_level=[after])])


def test_retired_seam_replacement_follows_continuous_position_bound_successors():
    old, middle, current = 'a'*64, 'b'*64, 'c'*64
    assert inventory._retired_seam_replacement_hash('fixture', 'replacement', old,
        (seam_successor(old, middle), seam_successor(middle, current))) == current
    # No successor keeps the old hash, so it cannot authorize a changed live function.
    assert inventory._retired_seam_replacement_hash('fixture', 'replacement', old, ()) == old != current


@pytest.mark.parametrize('mutation', ['missing', 'previous', 'current', 'duplicate', 'index', 'removed'])
def test_retired_seam_replacement_rejects_missing_forked_or_unbound_links(mutation):
    old, middle, current = 'a'*64, 'b'*64, 'c'*64
    chain = [seam_successor(old, middle), seam_successor(middle, current)]
    if mutation == 'missing':
        chain.pop(0)
    elif mutation == 'previous':
        chain[1]['definitions'][0]['previous_sha256'] = 'd'*64
    elif mutation == 'current':
        chain[1]['definitions'][0]['sha256'] = 'd'*64
    elif mutation == 'duplicate':
        chain[1]['definitions'].append(copy.deepcopy(chain[1]['definitions'][0]))
    elif mutation == 'index':
        chain[1]['definitions'][0]['current_index'] = 1
    else:
        chain[1]['removed_definitions'] = [dict(module='fixture', name='replacement')]
    with pytest.raises(RuntimeError, match='retired seam replacement'):
        inventory._retired_seam_replacement_hash('fixture', 'replacement', old, chain)


def test_private_performance_publication_keeps_every_release_record(predecessor, tmp_path, monkeypatch):
    files, _metadata, prior = predecessor
    if files is None:
        pytest.skip('Actual qualified archive is needed for the private source fixture')
    root = tmp_path/'repository'
    for relative, payload in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root/'XTA/performance_fixture.py').write_text('VALUE = 1\n', encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py',
                     'tools/benchmark_native_pull_projection.py'):
        (root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest = root/'release/_package_inventory.json'
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(inventory, 'ROOT', root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest)
    def fixture_git(command, **kwargs):
        assert command == ['git', 'rev-parse', 'v25.0.0^{commit}']
        return inventory.REVIEWED_V25_JOB150772_PERFORMANCE_DEVELOPMENT_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=tmp_path/'evidence', development='job150772-performance',
                             predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == prior
    assert result['written'] and published[KEY]['released'] is False
    assert published[KEY]['predecessor_source_archive']['full_qualification'] is True
