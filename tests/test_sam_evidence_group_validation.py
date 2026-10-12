"""Long-group validation indexes identities without changing mask checks."""
import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter
from tests.test_sam_evidence_policy import fixture_group


def test_long_group_checks_each_mask_name_once(tmp_path):
    calls = []
    class Name(str):
        def startswith(self, prefix, *args):
            calls.append(self)
            return super().startswith(prefix, *args)
    group, original, _ = fixture_group()
    group['frame_indices'] = list(range(257))
    group['endpoints'][-1]['frame_index'] = 256
    acceptance = original['acceptance:0']
    endpoint = original['endpoint:family:A']
    masks = {Name(key): value for key,value in original.items()
             if key.startswith(('endpoint:', 'evaluation:'))}
    for frame in group['frame_indices']:
        write = acceptance & ~endpoint if frame in (0, 256) else acceptance
        masks[Name(f'acceptance:{frame}')] = acceptance
        masks[Name(f'write:{frame}')] = write
        masks[Name(f'edge_write:family:edge:{frame}')] = write
    with SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        writer.add_group(group, masks)
        assert len(calls) == len(masks)
        bundle = writer.commit()
    for frame in (0, 128, 256):
        np.testing.assert_array_equal(bundle.group_mask('family', f'edge_write:family:edge:{frame}'),
                                      masks[f'write:{frame}'])


@pytest.mark.parametrize('name,rejected', [
    ('edge_write:branch:1', True), ('edge_write:1', True), ('edge_write::1', True),
    ('edge_write:has:colons:1', True), ('edge_write:branch:01', False),
    ('edge_write:branch:+1', False), ('edge_write:branch:1.0', False),
    ('edge_write:branch:99', False), ('edge_write:', False), ('not_edge_write:1', False)])
def test_edge_mask_suffix_keeps_exact_legacy_matching_and_failed_write_atomicity(tmp_path, name, rejected):
    group, masks, _ = fixture_group()
    outside = np.zeros((12, 16), bool)
    outside[0, 0] = True
    masks[name] = outside
    writer = SamEvidenceWriter(tmp_path/'bundle', {})
    try:
        if rejected:
            with pytest.raises(ValueError, match='Branch-specific SAM write domain'):
                writer.add_group(group, masks)
            assert not writer.groups and not writer.records and writer._stream.tell() == 0
        else:
            writer.add_group(group, masks)
            assert name in writer.groups['family']['mask_keys']
    finally:
        writer.abort()


def test_branch_checks_keep_frame_order_and_mapping_order_within_frame(tmp_path):
    group, original, _ = fixture_group()
    reads = []
    class Masks(dict):
        def __getitem__(self, key):
            if key.startswith('edge_write:'):
                reads.append(key)
            return super().__getitem__(key)
    masks = Masks(original)
    for name, frame in [('edge_write:z:3', 3), ('edge_write:z:1', 1),
                        ('edge_write:a:1', 1), ('edge_write:q:0', 0)]:
        masks[name] = original[f'write:{frame}']
    writer = SamEvidenceWriter(tmp_path/'bundle', {})
    try:
        writer.add_group(group, masks)
        assert reads[:4] == ['edge_write:q:0', 'edge_write:z:1', 'edge_write:a:1', 'edge_write:z:3']
    finally:
        writer.abort()


def test_earlier_invalid_frame_still_fails_before_later_branch_and_before_encoding(tmp_path):
    group, masks, _ = fixture_group()
    outside = np.zeros((12, 16), bool)
    outside[0, 0] = True
    masks['edge_write:later:3'] = outside
    masks['edge_write:earlier:1'] = np.full((12, 16), 2, np.uint8)
    writer = SamEvidenceWriter(tmp_path/'bundle', {})
    try:
        with pytest.raises(ValueError, match='must be binary'):
            writer.add_group(group, masks)
        assert not writer.groups and not writer.records and writer._stream.tell() == 0
    finally:
        writer.abort()
