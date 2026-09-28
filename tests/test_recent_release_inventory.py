"""Check the release chain and source pins with one shared set of cases.

Older, release-specific tests live in Scratch/Data/XTA/History. The active
inventory is authenticated by the verifier; these tests exercise its shared
failure modes and the few release-specific rules that still matter.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass
from unittest import mock

import pytest

from tools import verify_package_inventory as inventory


@dataclass(frozen=True)
class Release:
    key: str
    number: str
    contract: object
    previous_contract: object
    predecessor_digest: str
    previous_review_digest: str
    predecessor_commit: str


RELEASES = (
    Release('v22_3_release_review', '22.3.0', inventory.reviewed_v22_3_release_contract,
            inventory.reviewed_v22_2_release_contract,
            inventory.REVIEWED_V22_3_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V22_2_RELEASE_SHA256,
            'd336a66d75d7811a2c07d007ed40c46a2cfcfcb2'),
    Release('v22_3_1_release_review', '22.3.1', inventory.reviewed_v22_3_1_release_contract,
            inventory.reviewed_v22_3_release_contract,
            inventory.REVIEWED_V22_3_1_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V22_3_RELEASE_SHA256,
            '300cda53b477bdc2263a80d53bb354dc2ea8df46'),
    Release('v22_3_2_release_review', '22.3.2', inventory.reviewed_v22_3_2_release_contract,
            inventory.reviewed_v22_3_1_release_contract,
            inventory.REVIEWED_V22_3_2_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V22_3_1_RELEASE_SHA256,
            'a70e424d62916d2e3ee0ce2dcb3d02565a6ccb3e'),
    Release('v23_release_review', '23.0.1', inventory.reviewed_v23_release_contract,
            inventory.reviewed_v22_3_2_release_contract,
            inventory.REVIEWED_V23_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V22_3_2_RELEASE_SHA256,
            inventory.REVIEWED_V23_RELEASE_PREDECESSOR_COMMIT),
    Release('v23_0_2_release_review', '23.0.2', inventory.reviewed_v23_0_2_release_contract,
            inventory.reviewed_v23_release_contract,
            inventory.REVIEWED_V23_0_2_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V23_RELEASE_SHA256,
            inventory.REVIEWED_V23_0_2_RELEASE_PREDECESSOR_COMMIT),
    Release('v23_0_3_release_review', '23.0.3', inventory.reviewed_v23_0_3_release_contract,
            inventory.reviewed_v23_0_2_release_contract,
            inventory.REVIEWED_V23_0_3_RELEASE_PREDECESSOR_SHA256,
            inventory.REVIEWED_V23_0_2_RELEASE_SHA256,
            inventory.REVIEWED_V23_0_3_RELEASE_PREDECESSOR_COMMIT),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.fixture(scope='module')
def complete_manifest():
    return json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def current_trees(complete_manifest):
    modules = {
        item['module']
        for release in RELEASES
        for item in complete_manifest[release.key]['module_snapshots']
    }
    return {
        module: ast.parse(path.read_text(encoding='utf-8'))
        for module in modules
        if (path := inventory.PACKAGE / (module + '.py')).is_file()
    }


def _at_release(complete_manifest, release):
    manifest = copy.deepcopy(complete_manifest)
    for later in RELEASES[RELEASES.index(release) + 1:]:
        manifest.pop(later.key, None)
    return manifest


@pytest.mark.parametrize('release', RELEASES, ids=lambda item: item.number)
def test_review_authenticates_complete_predecessor(complete_manifest, release):
    manifest = _at_release(complete_manifest, release)
    review = release.contract(manifest, manifest['v21_review'])
    prior = {key: value for key, value in manifest.items() if key != release.key}
    assert _digest(prior) == release.predecessor_digest
    assert review['predecessor_commit'] == release.predecessor_commit
    assert review['previous_review_sha256'] == release.previous_review_digest
    assert review['release'] == release.number


@pytest.mark.parametrize('release', RELEASES, ids=lambda item: item.number)
def test_predecessor_and_review_changes_are_rejected(complete_manifest, release):
    manifest = _at_release(complete_manifest, release)
    for key, mutation in (('v21_review', lambda value: value.update(tampered=True)),
                          ('statements', lambda value: value[0].update(line=-1))):
        altered = copy.deepcopy(manifest)
        mutation(altered[key])
        with pytest.raises(RuntimeError, match=f'v{release.number} predecessor inventory changed'):
            release.contract(altered, altered['v21_review'])

    altered = copy.deepcopy(manifest)
    altered[release.key]['definitions'][0]['reason'] = 'Unreviewed replacement'
    with pytest.raises(RuntimeError, match=f'v{release.number} review digest mismatch'):
        release.previous_contract(altered, altered['v21_review'])


@pytest.mark.parametrize('release', RELEASES, ids=lambda item: item.number)
def test_reauthenticated_review_cannot_change_independent_source_pin(complete_manifest, release):
    manifest = _at_release(complete_manifest, release)
    review = manifest[release.key]
    snapshot = next(item for item in review['module_snapshots'] if item['previous_top_level'])
    snapshot['previous_ast_sha256'] = '0' * 64
    digest_name = {
        '22.3.0': 'REVIEWED_V22_3_RELEASE_SHA256',
        '22.3.1': 'REVIEWED_V22_3_1_RELEASE_SHA256',
        '22.3.2': 'REVIEWED_V22_3_2_RELEASE_SHA256',
        '23.0.1': 'REVIEWED_V23_RELEASE_SHA256',
        '23.0.2': 'REVIEWED_V23_0_2_RELEASE_SHA256',
        '23.0.3': 'REVIEWED_V23_0_3_RELEASE_SHA256',
    }[release.number]
    with mock.patch.object(inventory, digest_name, _digest(review)):
        with pytest.raises(RuntimeError, match='source predecessor changed'):
            release.contract(manifest, manifest['v21_review'])


@pytest.mark.parametrize('release', RELEASES, ids=lambda item: item.number)
def test_current_sources_and_tools_follow_authenticated_successors(complete_manifest, current_trees, release):
    review = complete_manifest[release.key]
    successors = tuple(complete_manifest[item.key] for item in RELEASES[RELEASES.index(release) + 1:])
    trees = {item['module']: current_trees[item['module']] for item in review['module_snapshots']
             if item['module'] in current_trees}
    inventory.verify_v22_3_source_snapshots(review, trees, successors)
    inventory.verify_v22_3_validation_tools(review, successors)

    module = next(item['module'] for item in review['module_snapshots'] if item['module'] in trees)
    altered = dict(trees)
    altered[module] = copy.deepcopy(trees[module])
    altered[module].body.append(ast.Pass())
    with pytest.raises(RuntimeError, match=f'v{release.number} reviewed source changed'):
        inventory.verify_v22_3_source_snapshots(review, altered, successors)


def test_removed_examples_retain_their_last_source_pins(complete_manifest, current_trees):
    manifest = _at_release(complete_manifest, RELEASES[1])
    review = manifest['v22_3_1_release_review']
    prior = manifest['v22_3_release_review']
    successors = tuple(complete_manifest[item.key] for item in RELEASES[2:])
    trees = {item['module']: current_trees[item['module']] for item in review['module_snapshots']
             if item['module'] in current_trees}
    for snapshot in review['module_snapshots']:
        if not snapshot.get('removed'):
            continue
        module = snapshot['module']
        historical = next(item for item in prior['module_snapshots'] if item['module'] == module)
        assert snapshot['previous_ast_sha256'] == historical['ast_sha256']
        assert snapshot['previous_top_level'] == historical['top_level']
        assert not (inventory.PACKAGE / (module + '.py')).exists()
        with pytest.raises(RuntimeError, match='v22.3.1 reviewed source changed'):
            inventory.verify_v22_3_source_snapshots(review, {**trees, module: ast.parse('')}, successors)


@pytest.mark.parametrize('mutation, message', [
    ('extra', 'reviewed module removals differ'),
    ('missing', 'reviewed module removals differ'),
    ('source', 'removed module retains current source records'),
    ('statements', 'removed module retains current source records'),
])
def test_removal_scope_cannot_be_reauthenticated(complete_manifest, mutation, message):
    manifest = _at_release(complete_manifest, RELEASES[1])
    review = manifest['v22_3_1_release_review']
    removed = next(item for item in review['module_snapshots'] if item.get('removed'))
    if mutation == 'extra':
        next(item for item in review['module_snapshots'] if item['module'] == 'pipeline')['removed'] = True
    elif mutation == 'missing':
        removed.pop('removed')
    elif mutation == 'source':
        removed['ast_sha256'] = inventory.digest(ast.parse(''))
    else:
        removed['top_level'] = ['0' * 64]
    with mock.patch.object(inventory, 'REVIEWED_V22_3_1_RELEASE_SHA256', _digest(review)):
        with pytest.raises(RuntimeError, match=message):
            RELEASES[1].contract(manifest, manifest['v21_review'])


@pytest.mark.parametrize('release_index,path', [
    (1, 'tools/compare_reconciliation.py'),
    (2, 'tools/qualify_confidence_consolidation.py'),
    (2, 'tools/qualify_d1_confidence_bounds.py'),
])
def test_validation_tool_predecessor_links_are_independent(complete_manifest, release_index, path):
    release = RELEASES[release_index]
    manifest = _at_release(complete_manifest, release)
    review = manifest[release.key]
    tool = next(item for item in review['validation_tools'] if item['path'] == path)
    tool['previous_sha256'] = '0' * 64
    digest_name = 'REVIEWED_V22_3_1_RELEASE_SHA256' if release_index == 1 else 'REVIEWED_V22_3_2_RELEASE_SHA256'
    with mock.patch.object(inventory, digest_name, _digest(review)):
        with pytest.raises(RuntimeError, match='validation-tool predecessor changed'):
            release.contract(manifest, manifest['v21_review'])


@pytest.mark.parametrize('release', RELEASES, ids=lambda item: item.number)
def test_validation_tool_source_changes_are_rejected(complete_manifest, release):
    review = copy.deepcopy(complete_manifest[release.key])
    successors = tuple(complete_manifest[item.key] for item in RELEASES[RELEASES.index(release) + 1:])
    review['validation_tools'][0]['sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='validation tool changed'):
        inventory.verify_v22_3_validation_tools(review, successors)


def test_prior_validation_tools_require_authenticated_successor_links(complete_manifest):
    prior = complete_manifest['v22_3_release_review']
    successors = tuple(complete_manifest[item.key] for item in RELEASES[1:])
    inventory.verify_v22_3_validation_tools(prior, successors)
    altered = list(copy.deepcopy(successors))
    inherited_path = prior['validation_tools'][0]['path']
    update = next(item for item in altered[0]['validation_tools'] if item['path'] == inherited_path)
    update['previous_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='validation tool changed: predecessor'):
        inventory.verify_v22_3_validation_tools(prior, altered)


def test_semantic_release_identifies_new_model_and_qualification_tools(complete_manifest):
    review = complete_manifest['v23_release_review']
    assert review['feature'] == 'yolo-semantic-segmentation'
    assert {'semantic_cuda', 'semantic_trt'} <= {item['module'] for item in review['module_snapshots']}
    for path in ('tools/export_semantic_logits.py', 'tools/qualify_semantic_trt.py',
                 'tools/qualify_pta_classification.py', 'tools/qualify_pta_gpu_masks.py',
                 'tools/qualify_pta_gpu_render.py'):
        assert next(item for item in review['validation_tools'] if item['path'] == path)['previous_sha256'] is None


@pytest.mark.parametrize('mutation, message', [
    ('missing', 'reviewed definition removals differ'),
    ('predecessor', 'retired definition predecessor changed'),
])
def test_retired_geometry_helper_keeps_authenticated_predecessor(complete_manifest, mutation, message):
    altered = copy.deepcopy(complete_manifest)
    altered.pop('v23_0_3_release_review', None)
    review = altered['v23_0_2_release_review']
    retired = review['removed_definitions']
    assert [(item['module'], item['name']) for item in retired] == [
        ('geometry', '_angle_from_aug_id')]
    assert not any(getattr(node, 'name', None) == '_angle_from_aug_id'
                   for node in ast.parse((inventory.PACKAGE / 'geometry.py').read_text(encoding='utf-8')).body)
    if mutation == 'missing':
        retired.clear()
    else:
        retired[0]['previous_index'] = -1
    with mock.patch.object(inventory, 'REVIEWED_V23_0_2_RELEASE_SHA256', _digest(review)):
        with pytest.raises(RuntimeError, match=message):
            inventory.reviewed_v23_0_2_release_contract(altered, altered['v21_review'])
