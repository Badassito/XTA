"""The publication throughput review preserves every committed release record."""
import ast
import copy
import hashlib
import json
from unittest import mock

import pytest

from tools import verify_package_inventory as inventory


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.fixture
def manifest():
    value = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    if 'v22_3_2_release_review' not in value:
        pytest.skip('Final publication throughput review has not been written')
    return value


def test_committed_predecessor_and_all_of_its_records_remain_authenticated(manifest):
    review = inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review'])
    prior = inventory._without_reviewed_v22_3_2_release(manifest)
    assert canonical(prior) == inventory.REVIEWED_V22_3_2_RELEASE_PREDECESSOR_SHA256
    assert review['predecessor_commit'] == 'a70e424d62916d2e3ee0ce2dcb3d02565a6ccb3e'
    assert review['previous_review_sha256'] == inventory.REVIEWED_V22_3_1_RELEASE_SHA256
    assert prior['v22_3_1_release_review']['release'] == '22.3.1'
    assert review['release'] == '22.3.2'
    assert not any(item.get('removed') for item in review['module_snapshots'])


def test_rewriting_any_earlier_record_is_rejected(manifest):
    for key in manifest.keys() - {'v22_3_2_release_review'}:
        altered = copy.deepcopy(manifest)
        if isinstance(altered[key], dict):
            altered[key]['tampered'] = True
        elif isinstance(altered[key], list):
            altered[key][0]['line'] = -1
        else:
            altered[key] += 1
        with pytest.raises(RuntimeError, match='v22.3.2 predecessor inventory changed'):
            inventory.reviewed_v22_3_2_release_contract(altered, altered['v21_review'])


@pytest.mark.parametrize('mutation, message', [
    ('source_anchor', 'source predecessor changed'),
    ('historical_statement', 'source predecessor changed'),
    ('removal', 'unreviewed module removal'),
    ('position', 'statement position differs'),
])
def test_reauthenticated_review_cannot_rewrite_independent_pins(manifest, mutation, message):
    review = manifest['v22_3_2_release_review']
    snapshot = next(item for item in review['module_snapshots'] if item['previous_top_level'])
    if mutation == 'source_anchor':
        snapshot['previous_ast_sha256'] = '0' * 64
    elif mutation == 'historical_statement':
        snapshot['previous_top_level'][0] = '0' * 64
    elif mutation == 'removal':
        snapshot['removed'] = True
    else:
        review['definitions'][0]['current_index'] = -1
    with mock.patch.object(inventory, 'REVIEWED_V22_3_2_RELEASE_SHA256', canonical(review)):
        with pytest.raises(RuntimeError, match=message):
            inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review'])


def test_current_sources_and_qualification_tools_match_the_review(manifest):
    review = inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review'])
    trees = {item['module']: ast.parse((inventory.PACKAGE / (item['module'] + '.py')).read_text(encoding='utf-8'))
             for item in review['module_snapshots']}
    inventory.verify_v22_3_source_snapshots(review, trees)
    inventory.verify_v22_3_validation_tools(review)
    for module in ('pipeline', 'confidence_consolidation', 'confidence_publication',
                   'd1_confidence_retirement', 'cuda_d1', 'inference', 'runtime', 'tta_scheduler',
                   'outputs', 'reconciliation_runtime', 'scheduler_diagnostics',
                   'backprojection', 'tta_background'):
        altered = dict(trees)
        altered[module] = copy.deepcopy(trees[module])
        altered[module].body.append(ast.Pass())
        with pytest.raises(RuntimeError, match='v22.3.2 reviewed source changed'):
            inventory.verify_v22_3_source_snapshots(review, altered)


@pytest.mark.parametrize('path', [
    'tools/compare_reconciliation.py', 'tools/qualify_tta_reconciliation.py',
    'tools/export_reconciliation_evidence.py', 'tools/qualify_confidence_consolidation.py',
    'tools/qualify_d1_confidence_bounds.py', 'tools/analyze_pipeline_trace.py',
])
def test_validation_tool_history_is_independently_preserved(manifest, path):
    review = manifest['v22_3_2_release_review']
    tool = next(item for item in review['validation_tools'] if item['path'] == path)
    tool['previous_sha256'] = '0' * 64
    with mock.patch.object(inventory, 'REVIEWED_V22_3_2_RELEASE_SHA256', canonical(review)):
        with pytest.raises(RuntimeError, match='validation-tool predecessor changed'):
            inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review'])


def test_older_release_contracts_admit_only_the_authenticated_successor(manifest):
    prior = inventory._without_reviewed_v22_3_2_release(manifest)
    assert inventory.reviewed_v22_3_1_release_contract(manifest, manifest['v21_review']) == prior['v22_3_1_release_review']
    assert inventory.reviewed_v22_3_release_contract(manifest, manifest['v21_review']) == prior['v22_3_release_review']
    manifest['v22_3_2_release_review']['definitions'][0]['reason'] = 'Unreviewed replacement'
    with pytest.raises(RuntimeError, match='v22.3.2 review digest mismatch'):
        inventory.reviewed_v22_3_1_release_contract(manifest, manifest['v21_review'])
