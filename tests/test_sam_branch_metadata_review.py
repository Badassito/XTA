"""Independent pressure-transition and nested ownership adversaries."""
import numpy as np
import pytest

from XTA import sam_branch_selection as branch, sam_policy
from XTA.sam_evidence import SamEvidenceWriter, _plain
from XTA.sam_filtering import build_mask_filter
from XTA.sam_mask_reader import SamMaskReader
from tests.test_sam_branch_support import bundle_fixture
from tests.test_sam_branch_performance import two_groups


def three_geometry_families(tmp_path):
    """A real larger middle crop leaves less topology slack than its neighbors."""
    source, _ = bundle_fixture(tmp_path/'original', grown=True)
    with SamEvidenceWriter(tmp_path/'pressure', {'shape_tyx': [5, 40, 45]}) as writer:
        for suffix, x, height in (('', 0, 13), ('2', 15, 40), ('3', 30, 13)):
            group = _plain(source.groups['g'])
            group.update(group_id='g'+suffix, context_bbox_yx=[0, x, height, x+15])
            for endpoint in group['endpoints']:
                endpoint['observation_id'] += suffix
            for edge in group['edges']:
                for name in ('edge_id', 'source_id', 'target_id'):
                    edge[name] += suffix
            masks = {}
            for name in source.groups['g']['mask_keys']:
                parts = name.split(':')
                if parts[0] in ('edge_write', 'edge_contract', 'endpoint', 'evaluation', 'permitted'):
                    parts[1] += suffix
                masks[':'.join(parts)] = np.pad(source.group_mask('g', name),
                    ((0, height-13), (0, 0)))
            writer.add_group(group, masks)
            for identifier in ('f', 'r'):
                run = _plain(source.runs[identifier])
                run.update(run_id=identifier+suffix, group_id='g'+suffix,
                    seed_ids=[value+suffix for value in run['seed_ids']],
                    held_out_ids=[value+suffix for value in run['held_out_ids']],
                    edge_ids=['e'+suffix])
                writer.add_run(run, {frame: np.pad(source.raw_mask(identifier, frame),
                    ((0, height-13), (0, 0))) for frame in run['expected_frames']})
        return writer.commit()


@pytest.mark.parametrize('middle_admitted', (True, False))
def test_real_larger_family_pressure_retires_or_preserves_prefix_then_recovers(
        tmp_path, monkeypatch, middle_admitted):
    bundle = three_geometry_families(tmp_path)
    middle_bytes = branch.branch_workspace_bytes((5, 40, 15))
    policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0,
        'max_group_bytes': middle_bytes if middle_admitted else middle_bytes-1}}
    observed = []
    intrinsic = []
    original_overlay = SamMaskReader._branch_filter_overlay
    original_measure = sam_policy._measure_group_intrinsic

    def overlay(self, prefix, recipe, **kwargs):
        observed.append((tuple(prefix.branch_selection['edges']) if prefix.branch_selection else (),
            tuple(recipe['edges']), isinstance(prefix.branch_selection, branch._BranchPrefix),
            self._branch_index_bytes(prefix), kwargs['max_index_bytes']))
        return original_overlay(self, prefix, recipe, **kwargs)

    def measure(reader, group, *args, **kwargs):
        intrinsic.append((group['group_id'], kwargs['retained_index_bytes']))
        return original_measure(reader, group, *args, **kwargs)

    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay', overlay)
    monkeypatch.setattr(sam_policy, '_measure_group_intrinsic', measure)
    actual = sam_policy.select_sam_proposals(bundle, policy)
    third_trial = next(value for value in observed if value[1] == ('e3',))
    if middle_admitted:
        assert intrinsic == [('g', 0), ('g2', 0), ('g3', 0)]
        assert third_trial[:4] == (('e', 'e2'), ('e3',), False, 0)
        middle_trial = next(value for value in observed if value[1] == ('e2',))
        assert middle_trial == (('e',), ('e2',), False, 0, 0)
        assert actual['selected_run_ids'] == ['f', 'f2', 'f3', 'r', 'r2', 'r3']
    else:
        assert [group_id for group_id, _ in intrinsic] == ['g', 'g3']
        assert intrinsic[1][1] > 0
        assert third_trial[0] == ('e',) and third_trial[2] and third_trial[3] > 0
        assert actual['group_receipts']['g2']['status'] == 'not_assessed_resource_refused'
        assert actual['selected_run_ids'] == ['f', 'f3', 'r', 'r3']
    assert third_trial[4] > 0
    assert actual['selection_resources']['intrinsic_measurements']['branch_metadata']['overlay_snapshot_count'] >= 4

    # The previous full-merge route is an independent ownership-path control;
    # receipt decisions and exact packed masks retain the same quality policy.
    monkeypatch.setattr(SamMaskReader, '_branch_filter_overlay',
        lambda self, prefix, recipe, **kwargs: original_overlay(self, prefix, recipe, max_index_bytes=0))
    monkeypatch.setattr(sam_policy, '_measure_group_intrinsic', original_measure)
    expected = sam_policy.select_sam_proposals(bundle, policy)
    for key in ('branch_selection', 'group_receipts', 'run_receipts', 'selected_run_ids'):
        assert actual[key] == expected[key]


def test_grandchild_requires_each_borrow_and_cannot_retire_its_parent_prefix(tmp_path):
    bundle = two_groups(tmp_path)
    with bundle.reader(max_cache_bytes=0) as parent:
        radius = build_mask_filter(parent, enabled=False)
        recipe, _ = branch.build_connected_edge_selection(parent, radius, 'g',
            {'e': ['f', 'r']}, write_domain='fixed_context')
        original = parent._branch_filter_overlay(parent.filter_snapshot(radius), recipe, max_index_bytes=1024**2)
        reference = parent.effective_candidate_mask('f', 2, original)
        with parent.fork(max_cache_bytes=0) as child:
            child_snapshot = child.borrowed_filter_snapshot(original)
            with child.fork(max_cache_bytes=0) as grandchild:
                with pytest.raises(ValueError, match='active outer transaction'):
                    grandchild.borrowed_filter_snapshot(original)
                inherited = grandchild.borrowed_filter_snapshot(child_snapshot)
                with pytest.raises(ValueError, match='one reader transaction'):
                    parent._branch_filter_overlay(inherited, recipe, max_index_bytes=0)
                with pytest.raises(RuntimeError, match='active reader lanes'):
                    child.close()
                with pytest.raises(RuntimeError, match='active reader lanes'):
                    parent.close()
                assert parent.active and child.active and grandchild.active
                np.testing.assert_array_equal(grandchild.effective_candidate_mask('f', 2, inherited), reference)
            assert not grandchild.active and grandchild._stream is None
        assert not child.active and not parent._children
        np.testing.assert_array_equal(parent.effective_candidate_mask('f', 2, original), reference)
    assert parent.stats['transaction_complete'] and not parent.active
