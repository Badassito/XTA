"""The semantic release preserves and extends the authenticated source inventory."""
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
    if 'v23_release_review' not in value:
        pytest.skip('Final semantic release review has not been written')
    return value


def test_historical_inventory_is_unchanged(manifest):
    review = inventory.reviewed_v23_release_contract(manifest, manifest['v21_review'])
    prior = inventory._without_reviewed_v23_release(manifest)
    assert canonical(prior) == inventory.REVIEWED_V23_RELEASE_PREDECESSOR_SHA256
    assert review['predecessor_commit'] == inventory.REVIEWED_V23_RELEASE_PREDECESSOR_COMMIT
    assert review['previous_review_sha256'] == inventory.REVIEWED_V22_3_2_RELEASE_SHA256
    assert review['release'] == '23.0.1'
    assert review['feature'] == 'yolo-semantic-segmentation'
    assert {'semantic_cuda', 'semantic_trt'} <= {
        item['module'] for item in review['module_snapshots']}
    semantic_export = next(item for item in review['validation_tools']
                           if item['path'] == 'tools/export_semantic_logits.py')
    assert semantic_export['previous_sha256'] is None
    ring_qualification = next(item for item in review['validation_tools']
                              if item['path'] == 'tools/qualify_semantic_trt.py')
    assert ring_qualification['previous_sha256'] is None
    pta_qualification = next(item for item in review['validation_tools']
                             if item['path'] == 'tools/qualify_pta_classification.py')
    assert pta_qualification['previous_sha256'] is None
    gpu_mask_qualification = next(item for item in review['validation_tools']
                                  if item['path'] == 'tools/qualify_pta_gpu_masks.py')
    assert gpu_mask_qualification['previous_sha256'] is None
    gpu_render_qualification = next(item for item in review['validation_tools']
                                    if item['path'] == 'tools/qualify_pta_gpu_render.py')
    assert gpu_render_qualification['previous_sha256'] is None


def test_earlier_records_cannot_be_rewritten(manifest):
    for key in manifest.keys() - {'v23_release_review'}:
        changed = copy.deepcopy(manifest)
        if isinstance(changed[key], dict):
            changed[key]['tampered'] = True
        elif isinstance(changed[key], list):
            changed[key][0]['line'] = -1
        else:
            changed[key] += 1
        with pytest.raises(RuntimeError, match='v23.0.1 predecessor inventory changed'):
            inventory.reviewed_v23_release_contract(changed, changed['v21_review'])


def test_independent_source_predecessor_pin_rejects_reauthenticated_review(manifest):
    review = manifest['v23_release_review']
    snapshot = next(item for item in review['module_snapshots'] if item['previous_top_level'])
    snapshot['previous_ast_sha256'] = '0' * 64
    with mock.patch.object(inventory, 'REVIEWED_V23_RELEASE_SHA256', canonical(review)):
        with pytest.raises(RuntimeError, match='v23.0.1 source predecessor changed'):
            inventory.reviewed_v23_release_contract(manifest, manifest['v21_review'])


def test_current_sources_and_tools_match_review(manifest):
    review = inventory.reviewed_v23_release_contract(manifest, manifest['v21_review'])
    trees = {item['module']: ast.parse((inventory.PACKAGE / (item['module'] + '.py')).read_text(encoding='utf-8'))
             for item in review['module_snapshots']}
    inventory.verify_v22_3_source_snapshots(review, trees)
    inventory.verify_v22_3_validation_tools(review)
    changed = dict(trees)
    module = review['module_snapshots'][0]['module']
    changed[module] = copy.deepcopy(trees[module])
    changed[module].body.append(ast.Pass())
    with pytest.raises(RuntimeError, match='v23.0.1 reviewed source changed'):
        inventory.verify_v22_3_source_snapshots(review, changed)


def test_historical_contracts_admit_only_authenticated_successor(manifest):
    prior = inventory._without_reviewed_v23_release(manifest)
    assert inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review']) == prior['v22_3_2_release_review']
    manifest['v23_release_review']['definitions'][0]['reason'] = 'Unreviewed replacement'
    with pytest.raises(RuntimeError, match='v23.0.1 review digest mismatch'):
        inventory.reviewed_v22_3_2_release_contract(manifest, manifest['v21_review'])
