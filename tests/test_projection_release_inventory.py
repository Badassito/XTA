"""The patch release preserves its tagged and reviewed development parents."""
import copy
import ast
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory

ARCHIVE = Path(r'C:\Users\Bry\Documents\ChatGPT\Scratch\Experiments\SAM_Job150615_20261001\final_snapshot_validation\source\XTA_v25.0.0_complete_source.zip')
KEY = 'v25_0_1_release_review'
PREFIX = 'REVIEWED_V25_0_1_RELEASE'


@pytest.fixture(scope='module')
def predecessor():
    if ARCHIVE.is_file():
        files, metadata = prepare.reviewed_release_predecessor_archive(ARCHIVE)
        return files, metadata, json.loads(files['release/_package_inventory.json'])
    prior = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    prior.pop('v25_job150790_150798_throughput_development_review', None)
    prior.pop('v25_job150772_performance_development_review', None)
    prior.pop(KEY, None)
    metadata = dict(kind='reviewed_development_source_zip',
        qualification_status='strict_gate_failed_followup_audited', full_qualification=False,
        sha256=inventory.REVIEWED_V25_0_1_RELEASE_PREDECESSOR_ARCHIVE_SHA256,
        source_identity_count=699, filename='XTA_v25.0.0_complete_source.zip')
    return None, metadata, prior


def review_fixture(predecessor):
    _files, metadata, prior = predecessor
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Preserve the reviewed development tool.')
             for row in prior['v25_job150615_headroom_development_review']['validation_tools']]
    tools.extend(dict(path=path, previous_sha256=previous,
                      sha256=hashlib.sha256((prepare.ROOT/path).read_text(encoding='utf-8').encode()).hexdigest(),
                      reason='Add explicit all-view projection qualification.')
                 for path, previous in getattr(inventory, PREFIX+'_ADDED_VALIDATION_TOOLS').items())
    return dict(release='25.0.1', kind='release', package_version='25.0.1',
        predecessor_tag='v25.0.0', released_predecessor_version='25.0.0', feature='all-view-projection-coverage',
        previous_review_sha256=inventory.REVIEWED_V25_JOB150615_HEADROOM_DEVELOPMENT_SHA256,
        predecessor_commit=inventory.REVIEWED_V25_0_1_RELEASE_PREDECESSOR_COMMIT,
        predecessor_inventory_sha256=prepare.canonical(prior), predecessor_source_archive=metadata,
        definitions=[], statements=[], removed_definitions=[], removed_statements=[], local_import_seam_updates=[],
        preserved_radial_definition_updates=[], preserved_radial_module_updates=[], complete_modules=[],
        module_snapshots=[], validation_tools=tools)


def authenticate(prior, review):
    with mock.patch.object(inventory, PREFIX+'_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX+'_PREDECESSOR_MODULES', {}), \
         mock.patch.object(inventory, PREFIX+'_RETIRED_LOCAL_IMPORT_SEAMS', {}), \
         mock.patch.object(inventory, PREFIX+'_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v25_0_1_release_contract({**prior, KEY: review}, prior['v21_review'])


def test_patch_has_independent_tag_and_reviewed_archive_predecessors(predecessor):
    files, metadata, prior = predecessor
    if files is not None:
        assert len(files) == metadata['source_identity_count'] == 699
    assert prepare.canonical(prior) == inventory.REVIEWED_V25_0_1_RELEASE_PREDECESSOR_SHA256
    assert metadata['sha256'] == inventory.REVIEWED_V25_0_1_RELEASE_PREDECESSOR_ARCHIVE_SHA256
    assert metadata['full_qualification'] is False
    assert authenticate(prior, review_fixture(predecessor))['predecessor_tag'] == 'v25.0.0'


@pytest.mark.parametrize('field,value,error', [
    ('kind', 'qualified_development_source_zip', 'predecessor archive changed'),
    ('qualification_status', 'complete', 'failed strict-gate status'),
    ('full_qualification', True, 'failed strict-gate status'),
    ('full_qualification', 0, 'failed strict-gate status'),
    ('sha256', '0'*64, 'predecessor archive changed'),
])
def test_patch_cannot_upgrade_the_previous_strict_gate(predecessor, field, value, error):
    _files, _metadata, prior = predecessor
    review = review_fixture(predecessor)
    review['predecessor_source_archive'] = dict(review['predecessor_source_archive'], **{field: value})
    with pytest.raises(RuntimeError, match=error):
        authenticate(prior, review)


@pytest.mark.parametrize('key', ['v25_job150615_headroom_development_review', 'v25_job150615_development_review',
                               'v25_guarded_rescue_development_review', 'v25_outer_crop_development_review',
                               'v25_0_0_release_review'])
def test_patch_cannot_rewrite_any_release_or_development_receipt(predecessor, key):
    _files, _metadata, prior = predecessor
    altered = copy.deepcopy(prior)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, review_fixture(predecessor))


def test_patch_requires_reviewed_archive(tmp_path):
    with pytest.raises(ValueError, match='25.0.1 requires --predecessor-archive'):
        prepare.prepare(output_dir=tmp_path, release='25.0.1')


def test_seam_retirement_requires_exact_old_seam_and_replacement_review():
    old = ('def runner():\n    '+inventory.LOCAL_IMPORT_SEAM_MARKER+
           '\n    from .sample import helper\n    return helper()\n')
    new = 'def runner():\n    return 123\n'
    key = ('fixture', 'runner')
    previous = inventory.reviewed_local_import_seams('fixture', old, ast.parse(old))[key]
    replacement = inventory.digest(ast.parse(new).body[0])
    args = dict(complete=True, labels_by_hash={}, reason='Replace the reviewed local seam.')
    with pytest.raises(ValueError, match='seam ownership changed'):
        prepare.review_module('fixture', old, new, **args)
    with pytest.raises(ValueError, match='independent review'):
        prepare.review_module('fixture', old, new, seam_retirements={key: (*previous, '0'*64)}, **args)
    _pin, _snapshot, records = prepare.review_module('fixture', old, new,
        seam_retirements={key: (*previous, replacement)}, **args)
    record, = records['retired_local_import_seams']
    assert (record['previous_definition_sha256'], record['previous_seam_sha256'],
            record['definition_sha256']) == (*previous, replacement)


def test_patch_positional_rewind_preserves_retained_statement_reordering():
    old, new = 'A = 1\nB = 2\nC = 3\n', 'C = 3\nA = 10\nB = 2\n'
    _pin, snapshot, records = prepare.review_module('fixture', old, new,
        complete=False, labels_by_hash={}, reason='Review explicit statement movement.')
    review = dict(release='25.0.1', module_snapshots=[snapshot], **records)
    restored = inventory.rewind_reviewed_statements('fixture', ast.parse(new).body, (review,))
    assert [value for _node, value in restored] == snapshot['previous_top_level']
    altered = copy.deepcopy(review)
    altered['statements'][0]['previous_index'] = 2
    with pytest.raises(RuntimeError, match='predecessor'):
        inventory.rewind_reviewed_statements('fixture', ast.parse(new).body, (altered,))


def test_private_patch_publication_preserves_all_appendices(predecessor, tmp_path, monkeypatch):
    files, _metadata, prior = predecessor
    if files is None:
        pytest.skip('Retained reviewed archive is needed for private publication')
    root = tmp_path/'repository'
    for relative, payload in files.items():
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    initializer = files['XTA/__init__.py'].decode('utf-8').replace('__version__ = "25.0.0"', '__version__ = "25.0.1"')
    (root/'XTA/__init__.py').write_text(initializer, encoding='utf-8', newline='\n')
    for relative in ('tools/prepare_reconciliation_release.py', 'tools/verify_package_inventory.py',
                     'tools/qualify_projection_coverage.py'):
        (root/relative).write_text((prepare.ROOT/relative).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    manifest = root/'release/_package_inventory.json'
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(inventory, 'ROOT', root)
    monkeypatch.setattr(inventory, 'MANIFEST', manifest)
    monkeypatch.setattr(inventory, PREFIX+'_RETIRED_LOCAL_IMPORT_SEAMS', {})
    def fixture_git(command, **kwargs):
        if command == ['git', 'tag', '--list', 'v25.0.1']:
            return b''
        assert command == ['git', 'rev-parse', 'v25.0.0^{commit}']
        return inventory.REVIEWED_V25_0_1_RELEASE_PREDECESSOR_COMMIT
    monkeypatch.setattr(prepare.subprocess, 'check_output', fixture_git)
    result = prepare.prepare(output_dir=tmp_path/'evidence', release='25.0.1',
                             predecessor_archive=ARCHIVE, write=True)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    assert {key: value for key, value in published.items() if key != KEY} == prior
    assert result['written'] and published[KEY]['kind'] == 'release'
    assert published[KEY]['predecessor_source_archive']['full_qualification'] is False
