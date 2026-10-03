"""Modern receipts cannot downgrade published support to legacy raw masks."""
import json

import pytest

from XTA.sam_evidence import load_sam_online_selection
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_evidence_policy import fixture_group, fixture_run, build_bundle


def test_guarded_whole_receipt_requires_filter_even_without_scope_sentinel(tmp_path):
    group,masks,raw=fixture_group()
    bundle=build_bundle(tmp_path,[(fixture_run('F',group),raw)],group=group,masks=masks)
    receipt=select_sam_proposals(bundle)
    assert receipt['resolved_policy']['version']==4
    receipt.pop('mask_filter')
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='retained mask filter'):
        load_sam_online_selection(bundle)


def test_guarded_tiled_receipt_requires_filter_even_without_scope_sentinel(tmp_path):
    from tests.test_sam_tiled_evidence_policy import tiled_bundle
    bundle,_=tiled_bundle(tmp_path)
    receipt=select_sam_proposals(bundle)
    assert receipt['resolved_policy']['version']==5
    receipt.pop('mask_filter')
    (bundle.directory.parent/'selection.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='retained mask filter'):
        load_sam_online_selection(bundle)
