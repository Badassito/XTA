"""A versioned candidate extends tagged history without claiming qualification."""
from __future__ import annotations

import copy
import json
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory


KEY = 'v24_0_2_sam_bridges_development_review'
PREFIX = 'REVIEWED_V24_0_2_SAM_BRIDGES_DEVELOPMENT'
DEVELOPMENT = 'sam-bridges-v24.0.2'


@pytest.fixture(scope='module')
def predecessor():
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    current.pop('v24_1_0_sam_extrapolation_development_review', None)
    current.pop(KEY, None)
    return current


def fixture_review(prior):
    prior_tools = prior['v24_job150790_150798_throughput_development_review']['validation_tools']
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Retain the exact tagged validation tool.') for row in prior_tools]
    tools.extend(dict(path=path, previous_sha256=previous, sha256='a' * 64,
                      reason='Review the new research tool without claiming a workload passed.')
                 for path, previous in getattr(inventory, PREFIX + '_ADDED_VALIDATION_TOOLS').items())
    return dict(release='24.0.2', kind='development', development=DEVELOPMENT,
        package_version='24.0.2', released=False, predecessor_tag='v24.0.1',
        released_predecessor_version='24.0.1', source_review_only=True,
        feature='sam-bridges-v24.0.2-development',
        previous_review_sha256=inventory.REVIEWED_V24_JOB150790_150798_THROUGHPUT_DEVELOPMENT_SHA256,
        predecessor_commit=getattr(inventory, PREFIX + '_PREDECESSOR_COMMIT'),
        predecessor_inventory_sha256=prepare.canonical(prior),
        definitions=[], statements=[], removed_definitions=[], removed_statements=[],
        local_import_seam_updates=[], preserved_radial_definition_updates=[],
        preserved_radial_module_updates=[], complete_modules=[], module_snapshots=[], validation_tools=tools)


def authenticate(prior, review, *, pins=None):
    with mock.patch.object(inventory, PREFIX + '_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX + '_PREDECESSOR_MODULES', pins or {}), \
         mock.patch.object(inventory, PREFIX + '_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v24_0_2_sam_bridges_development_contract(
            {**prior, KEY: review}, prior['v21_review'])


def test_candidate_starts_from_exact_tagged_v24_0_1(predecessor):
    tagged = prepare.git_file(prepare.ROOT, 'v24.0.1', 'release/_package_inventory.json')
    assert json.loads(tagged) == predecessor
    assert prepare.canonical(predecessor) == getattr(inventory, PREFIX + '_PREDECESSOR_SHA256')
    review = authenticate(predecessor, fixture_review(predecessor))
    assert review['released'] is False and review['source_review_only'] is True


def test_candidate_version_bump_has_exact_source_predecessor_link(predecessor):
    previous = prepare.git_file(prepare.ROOT, 'v24.0.1', 'XTA/__init__.py')
    current = prepare.git_file(prepare.ROOT, 'v24.0.2', 'XTA/__init__.py')
    pin, snapshot, records = prepare.review_module('__init__', previous, current,
        complete=False, labels_by_hash={}, reason='Identify the separately reviewed v24.0.2 candidate.')
    review = fixture_review(predecessor)
    review.update(module_snapshots=[snapshot], **records)
    assert any(row.get('binding') == '__version__' for row in review['statements'])
    authenticate(predecessor, review, pins={'__init__': pin})
    review['module_snapshots'][0]['previous_ast_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='source predecessor changed'):
        authenticate(predecessor, review, pins={'__init__': pin})


@pytest.mark.parametrize('field,value', [
    ('kind', 'release'), ('released', True), ('source_review_only', False),
    ('package_version', '24.0.1'), ('predecessor_tag', 'v24.0.0'),
    ('full_qualification', True), ('qualification_status', 'passed'),
])
def test_candidate_cannot_claim_a_release_or_unexecuted_qualification(predecessor, field, value):
    review = fixture_review(predecessor)
    review[field] = value
    with pytest.raises(RuntimeError, match='source-reviewed development candidate'):
        authenticate(predecessor, review)


@pytest.mark.parametrize('key', ['v24_0_1_release_review',
                               'v24_job150790_150798_throughput_development_review'])
def test_candidate_cannot_reauthenticate_rewritten_history(predecessor, key):
    altered = copy.deepcopy(predecessor)
    altered[key]['feature'] = 'rewritten-history'
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, fixture_review(altered))


def test_candidate_tool_chain_keeps_inherited_order_and_independent_new_tool_pins(predecessor):
    spec = prepare.DEVELOPMENTS[DEVELOPMENT]
    inherited = tuple(row['path'] for row in predecessor['v24_job150790_150798_throughput_development_review']['validation_tools'])
    assert spec['validation_tools'] == inherited + tuple(getattr(inventory, PREFIX + '_ADDED_VALIDATION_TOOLS'))
    review = fixture_review(predecessor)
    review['validation_tools'][-1]['previous_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='validation-tool review has invalid predecessor'):
        authenticate(predecessor, review)


def test_candidate_tag_identity_is_verified_before_reading_source(tmp_path):
    with mock.patch.object(prepare.subprocess, 'check_output', return_value='0' * 40), \
         mock.patch.object(prepare, 'git_file', side_effect=AssertionError('Source must not be read')):
        with pytest.raises(ValueError, match='independently pinned v24.0.1'):
            prepare.prepare(output_dir=tmp_path, development=DEVELOPMENT)


def test_default_preparation_cannot_overwrite_an_existing_candidate(predecessor, tmp_path):
    current = {**predecessor, KEY: fixture_review(predecessor)}
    manifest = tmp_path / 'candidate.json'
    manifest.write_text(json.dumps(current), encoding='utf-8')
    with mock.patch.object(inventory, 'MANIFEST', manifest):
        with pytest.raises(ValueError, match='immutable; prepare a new successor'):
            prepare.prepare(output_dir=tmp_path, development=DEVELOPMENT, write=True)
