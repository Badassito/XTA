"""Losing modern owner recipes must never restore rejected candidate pixels."""
from copy import deepcopy
import json

import numpy as np
import pytest

from XTA.sam_branch_selection import validate_branch_selection
from XTA.sam_evidence import fingerprint, iter_selected_planes, load_sam_online_selection
from XTA.sam_mask_reader import effective_candidate_mask
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_branch_selection_adversarial import _single_edge, _bundle, _run, _policy


@pytest.fixture
def branch_case(tmp_path):
    group, masks, raw = _single_edge()
    # This isolated proposal belongs to the old write domain. Only the retained
    # connected-owner recipe excludes it, so a radius-only fallback revives it.
    raw[2][10, 10] = True
    bundle = _bundle(tmp_path, group, masks, [(_run('F'), raw)])
    receipt = select_sam_proposals(bundle, _policy())
    assert bundle.candidate_mask('F', 2)[10, 10]
    assert not effective_candidate_mask(bundle, 'F', 2, receipt)[10, 10]
    return bundle, receipt


@pytest.mark.parametrize('marker', [
    {'resolved_policy': {'version': '6'}},
    {'resolved_policy': {'version': '7'}},
    {'resolved_policy': {'branch_aware_selection': True}},
    {'resolved_policy': {'allow_paired_seed_tracks': True}},
    {'resolved_policy': {'branch_write_domain': 'fixed_context'}},
    {'policy_name': 'sam_conservative_connected_branches_v6'},
    {'policy_name': 'sam_conservative_tiled_connected_branches_v7'},
    {'branch_selection_summary': {}},
])
def test_retained_branch_markers_require_owner_recipe_at_every_read_boundary(branch_case, marker):
    bundle, complete = branch_case
    lost = {key: deepcopy(complete[key]) for key in
        ('schema', 'evidence_fingerprint', 'selected_run_ids', 'mask_filter')}
    lost.update(marker)
    with pytest.raises(ValueError, match='retained branch selection recipe'):
        effective_candidate_mask(bundle, 'F', 2, lost)
    with pytest.raises(ValueError, match='retained branch selection recipe'):
        list(iter_selected_planes(bundle, lost))
    (bundle.directory.parent/'selection.json').write_text(json.dumps(lost), encoding='utf-8')
    with pytest.raises(ValueError, match='retained branch selection recipe'):
        load_sam_online_selection(bundle)
    with bundle.reader() as reader:
        wrapper = {**lost, 'mask_filter': reader.filter_snapshot(lost['mask_filter'])}
        with pytest.raises(ValueError, match='retained branch selection recipe'):
            reader.filter_snapshot(wrapper)
        with pytest.raises(ValueError, match='retained branch selection recipe'):
            reader.effective_candidate_mask('F', 2, wrapper)


@pytest.mark.parametrize('version', [1, 4, 5, '4', '5'])
def test_historical_nonbranch_receipts_preserve_candidate_unions(branch_case, version):
    bundle, complete = branch_case
    historical = dict(mask_filter=complete['mask_filter'], selected_run_ids=['F'],
        resolved_policy=dict(version=version, branch_aware_selection=False,
            allow_paired_seed_tracks=False, branch_write_domain='edge_write'))
    np.testing.assert_array_equal(effective_candidate_mask(bundle, 'F', 2, historical),
        bundle.candidate_mask('F', 2))
    with bundle.reader() as reader:
        historical['mask_filter'] = reader.filter_snapshot(historical['mask_filter'])
        assert reader.filter_snapshot(historical).branch_selection is None
        assert reader.effective_candidate_mask('F', 2, historical)[10, 10]


def test_pre_hardening_v25_1_receipt_keeps_exact_packed_owner_support(branch_case):
    bundle, receipt = branch_case
    historical = deepcopy(receipt)
    recipe = historical['branch_selection']
    recipe['implementation_sha256'] = '9473abe3904f033d5ea1597122494685e535aefd5f03fc04f409c0513a6caf7b'
    recipe['sha256'] = fingerprint({key: value for key, value in recipe.items() if key != 'sha256'})
    assert validate_branch_selection(historical, bundle,
        mask_filter_sha256=historical['mask_filter']['sha256']) is not None
    np.testing.assert_array_equal(effective_candidate_mask(bundle, 'F', 2, historical),
        effective_candidate_mask(bundle, 'F', 2, receipt))
