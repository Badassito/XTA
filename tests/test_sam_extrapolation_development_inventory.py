"""v24.1.0 development preserves the actual tagged v24.0.2 predecessor."""
from __future__ import annotations

import copy
import json
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory


KEY = 'v24_1_0_sam_extrapolation_development_review'
PREFIX = 'REVIEWED_V24_1_0_SAM_EXTRAPOLATION_DEVELOPMENT'
DEVELOPMENT = 'sam-extrapolation-v24.1.0'


@pytest.fixture(scope='module')
def predecessor():
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    current.pop(KEY, None)
    return current


def fixture_review(prior):
    previous = prior['v24_0_2_sam_bridges_development_review']
    tools = [dict(path=row['path'], previous_sha256=row['sha256'], sha256=row['sha256'],
                  reason='Retain the exact tagged predecessor tool.') for row in previous['validation_tools']]
    tools.extend(dict(path=path, previous_sha256=old, sha256='a' * 64,
                      reason='Review the new extrapolation validation tool.')
                 for path, old in getattr(inventory, PREFIX + '_ADDED_VALIDATION_TOOLS').items())
    return dict(release='24.1.0', kind='development', development=DEVELOPMENT,
        package_version='24.1.0', released=False, predecessor_tag='v24.0.2',
        released_predecessor_version='24.0.2', source_review_only=True,
        feature='sam-extrapolation-v24.1.0-development',
        previous_review_sha256=inventory.REVIEWED_V24_0_2_SAM_BRIDGES_DEVELOPMENT_SHA256,
        predecessor_commit=getattr(inventory, PREFIX + '_PREDECESSOR_COMMIT'),
        predecessor_inventory_sha256=prepare.canonical(prior),
        definitions=[], statements=[], removed_definitions=[], removed_statements=[],
        local_import_seam_updates=[], preserved_radial_definition_updates=[],
        preserved_radial_module_updates=[], complete_modules=[], module_snapshots=[], validation_tools=tools)


def authenticate(prior, review, *, pins=None):
    with mock.patch.object(inventory, PREFIX + '_SHA256', prepare.canonical(review)), \
         mock.patch.object(inventory, PREFIX + '_PREDECESSOR_MODULES', pins or {}), \
         mock.patch.object(inventory, PREFIX + '_REMOVALS', {'definitions': (), 'statements': ()}):
        return inventory.reviewed_v24_1_0_sam_extrapolation_development_contract(
            {**prior, KEY: review}, prior['v21_review'])


def test_predecessor_is_actual_tagged_v24_0_2_and_retains_9956_source_receipt(predecessor):
    tagged = prepare.git_file(prepare.ROOT, 'v24.0.2', 'release/_package_inventory.json')
    assert json.loads(tagged) == predecessor
    assert prepare.canonical(predecessor) == getattr(inventory, PREFIX + '_PREDECESSOR_SHA256')
    assert getattr(inventory, PREFIX + '_PREDECESSOR_COMMIT') == '38a142a4905f5809c511273e544102a03b80868c'
    assert prepare.canonical(predecessor['v24_0_2_sam_bridges_development_review']) == '068ae54034ad1d0a0b88714c1024e216ef860855e74c1c520cc2ec0f16d23532'
    # A later Git tag must not rewrite what the original source review claimed.
    assert predecessor['v24_0_2_sam_bridges_development_review']['released'] is False
    assert authenticate(predecessor, fixture_review(predecessor))['source_review_only'] is True


@pytest.mark.parametrize('field,value', [
    ('kind', 'release'), ('released', True), ('source_review_only', False),
    ('package_version', '24.0.2'), ('predecessor_tag', 'v24.0.1'),
    ('full_qualification', True), ('qualification_status', 'passed'),
])
def test_uncommitted_extrapolation_cannot_claim_release_or_qualification(predecessor, field, value):
    review = fixture_review(predecessor)
    review[field] = value
    with pytest.raises(RuntimeError, match='source-reviewed development candidate'):
        authenticate(predecessor, review)


def test_v24_0_2_history_cannot_be_rewritten_even_after_self_rehash(predecessor):
    altered = copy.deepcopy(predecessor)
    altered['v24_0_2_sam_bridges_development_review']['released'] = True
    with pytest.raises(RuntimeError, match='predecessor inventory changed'):
        authenticate(altered, fixture_review(altered))


def test_version_bump_has_exact_v24_0_2_source_predecessor(predecessor):
    previous = prepare.git_file(prepare.ROOT, 'v24.0.2', 'XTA/__init__.py')
    current = (prepare.ROOT / 'XTA/__init__.py').read_text(encoding='utf-8')
    pin, snapshot, records = prepare.review_module('__init__', previous, current,
        complete=False, labels_by_hash={}, reason='Identify the separately reviewed v24.1.0 candidate.')
    review = fixture_review(predecessor)
    review.update(module_snapshots=[snapshot], **records)
    assert any(row.get('binding') == '__version__' for row in review['statements'])
    authenticate(predecessor, review, pins={'__init__': pin})
    review['module_snapshots'][0]['previous_ast_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='source predecessor changed'):
        authenticate(predecessor, review, pins={'__init__': pin})


def test_new_tool_list_inherits_all_tagged_validation_paths(predecessor):
    inherited = tuple(row['path'] for row in predecessor['v24_0_2_sam_bridges_development_review']['validation_tools'])
    assert prepare.DEVELOPMENTS[DEVELOPMENT]['validation_tools'] == inherited + tuple(
        getattr(inventory, PREFIX + '_ADDED_VALIDATION_TOOLS'))


def test_real_tag_identity_is_checked_before_source_read(tmp_path):
    with mock.patch.object(prepare.subprocess, 'check_output', return_value='0' * 40), \
         mock.patch.object(prepare, 'git_file', side_effect=AssertionError('Source must not be read')):
        with pytest.raises(ValueError, match='independently pinned v24.0.2'):
            prepare.prepare(output_dir=tmp_path, development=DEVELOPMENT)


def test_published_extrapolation_candidate_is_not_overwritten_by_default(predecessor, tmp_path):
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps({**predecessor, KEY: fixture_review(predecessor)}), encoding='utf-8')
    with mock.patch.object(inventory, 'MANIFEST', path):
        with pytest.raises(ValueError, match='immutable; prepare a new successor'):
            prepare.prepare(output_dir=tmp_path, development=DEVELOPMENT, write=True)
