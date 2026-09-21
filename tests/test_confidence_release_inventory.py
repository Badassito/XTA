"""The performance release extends the committed reconciliation inventory."""
import ast
import copy
import hashlib
import json
from unittest import mock

import pytest
from tools import verify_package_inventory as inventory


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@pytest.fixture
def manifest():
    value=json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    if 'v22_3_1_release_review' not in value:
        pytest.skip('Final performance release review has not been written')
    return value


def test_commit_and_entire_predecessor_are_preserved(manifest):
    review=inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])
    prior=inventory._without_reviewed_v22_3_1_release(manifest)
    assert canonical(prior)==inventory.REVIEWED_V22_3_1_RELEASE_PREDECESSOR_SHA256
    assert review['predecessor_commit']=='300cda53b477bdc2263a80d53bb354dc2ea8df46'
    assert review['previous_review_sha256']==inventory.REVIEWED_V22_3_RELEASE_SHA256
    assert prior['v22_3_release_review']['release']=='22.3.0'
    assert review['release']=='22.3.1'


def test_earlier_record_changes_are_rejected(manifest):
    for key in manifest.keys()-{'v22_3_1_release_review'}:
        changed=copy.deepcopy(manifest)
        if isinstance(changed[key],dict): changed[key]['tampered']=True
        elif isinstance(changed[key],list): changed[key][0]['line']=-1
        else: changed[key]+=1
        with pytest.raises(RuntimeError,match='v22.3.1 predecessor inventory changed'):
            inventory.reviewed_v22_3_1_release_contract(changed,changed['v21_review'])


def test_reauthenticating_review_cannot_change_independent_predecessor_pin(manifest):
    review=manifest['v22_3_1_release_review']
    snapshot=next(item for item in review['module_snapshots'] if item['previous_top_level'])
    snapshot['previous_ast_sha256']='0'*64
    with mock.patch.object(inventory,'REVIEWED_V22_3_1_RELEASE_SHA256',canonical(review)):
        with pytest.raises(RuntimeError,match='v22.3.1 source predecessor changed'):
            inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])


def test_current_complete_sources_and_tools_match_review(manifest):
    review=inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])
    trees={item['module']:ast.parse((inventory.PACKAGE/(item['module']+'.py')).read_text(encoding='utf-8'))
           for item in review['module_snapshots'] if not item.get('removed')}
    inventory.verify_v22_3_source_snapshots(review,trees)
    inventory.verify_v22_3_validation_tools(review)
    for module in ('confidence_evidence','pipeline','reconciliation_runtime'):
        changed=dict(trees)
        changed[module]=copy.deepcopy(trees[module])
        changed[module].body.append(ast.Pass())
        with pytest.raises(RuntimeError,match='v22.3.1 reviewed source changed'):
            inventory.verify_v22_3_source_snapshots(review,changed)


def test_retired_examples_preserve_history_and_reject_reintroduced_source(manifest):
    review=inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])
    prior=manifest['v22_3_release_review']
    trees={item['module']:ast.parse((inventory.PACKAGE/(item['module']+'.py')).read_text(encoding='utf-8'))
           for item in review['module_snapshots'] if not item.get('removed')}
    for snapshot in review['module_snapshots']:
        if not snapshot.get('removed'):
            continue
        module=snapshot['module']
        historical=next(item for item in prior['module_snapshots'] if item['module']==module)
        assert snapshot['previous_ast_sha256']==historical['ast_sha256']
        assert snapshot['previous_top_level']==historical['top_level']
        assert not (inventory.PACKAGE/(module+'.py')).exists()
        changed={**trees,module:ast.parse('')}
        with pytest.raises(RuntimeError,match='v22.3.1 reviewed source changed'):
            inventory.verify_v22_3_source_snapshots(review,changed)


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'source', 'statements'])
def test_removal_scope_cannot_expand_or_retain_code_even_if_reauthenticated(manifest,mutation):
    review=manifest['v22_3_1_release_review']
    removed=next(item for item in review['module_snapshots'] if item.get('removed'))
    if mutation=='extra':
        next(item for item in review['module_snapshots'] if item['module']=='pipeline')['removed']=True
        expected='reviewed module removals differ'
    elif mutation=='missing':
        removed.pop('removed')
        expected='reviewed module removals differ'
    elif mutation=='source':
        removed['ast_sha256']=inventory.digest(ast.parse(''))
        expected='removed module retains current source records'
    else:
        removed['top_level']=['0'*64]
        expected='removed module retains current source records'
    with mock.patch.object(inventory,'REVIEWED_V22_3_1_RELEASE_SHA256',canonical(review)):
        with pytest.raises(RuntimeError,match=expected):
            inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])


@pytest.mark.parametrize('path', ['tools/compare_reconciliation.py', 'tools/qualify_tta_reconciliation.py',
                                 'tools/export_reconciliation_evidence.py'])
def test_validation_tool_links_cannot_be_rewritten_even_if_review_is_reauthenticated(manifest, path):
    review=manifest['v22_3_1_release_review']
    item=next(item for item in review['validation_tools'] if item['path']==path)
    item['previous_sha256']='0'*64
    with mock.patch.object(inventory,'REVIEWED_V22_3_1_RELEASE_SHA256',canonical(review)):
        with pytest.raises(RuntimeError,match='validation-tool predecessor changed'):
            inventory.reviewed_v22_3_1_release_contract(manifest,manifest['v21_review'])


def test_previous_release_tool_verification_follows_only_authenticated_links(manifest):
    prior=inventory._without_reviewed_v22_3_1_release(manifest)
    review=manifest['v22_3_1_release_review']
    old=prior['v22_3_release_review']
    inventory.verify_v22_3_validation_tools(old,(review,))
    altered=copy.deepcopy(review)
    altered['validation_tools'][0]['previous_sha256']='0'*64
    with pytest.raises(RuntimeError,match='validation tool changed: predecessor'):
        inventory.verify_v22_3_validation_tools(old,(altered,))
