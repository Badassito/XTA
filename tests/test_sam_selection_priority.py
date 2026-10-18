"""Restored families cannot reorder already-admissible conflicting proposals."""
import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter, _plain, iter_selected_planes
from XTA.sam_policy import _group_selection_key, select_sam_proposals
from tests.test_sam_evidence_policy import fixture_group, fixture_run


def _bundle(path, groups, *, modern):
    template = None
    if modern:
        from tests.test_sam_branch_support import bundle_fixture
        template, _ = bundle_fixture(path.parent / (path.name + '-template'), grown=True)
    with SamEvidenceWriter(path, {'shape_tyx': [5, 13, 15]} if modern else {}) as writer:
        for group_id, offset, priority in groups:
            if not modern:
                group, masks, raw = fixture_group(group_id, x=5 + offset)
                if priority is not None:
                    group['selection_priority'] = [priority, group_id]
                writer.add_group(group, masks)
                writer.add_run(fixture_run(group_id + ':f', group), raw)
                continue
            group = _plain(template.groups['g'])
            group['group_id'] = group_id
            if priority is not None:
                group['selection_priority'] = [priority, group_id]
            def identity(value):
                return group_id + ':' + value
            for endpoint in group['endpoints']:
                endpoint['observation_id'] = identity(endpoint['observation_id'])
            if 'endpoint_ids' in group:
                group['endpoint_ids'] = [identity(value) for value in group['endpoint_ids']]
            for edge in group['edges']:
                for key in ('edge_id', 'source_id', 'target_id'):
                    edge[key] = identity(edge[key])
            masks = {}
            for name in template.groups['g']['mask_keys']:
                parts = name.split(':')
                if parts[0] in ('edge_write', 'edge_contract', 'endpoint', 'evaluation', 'permitted'):
                    parts[1] = identity(parts[1])
                masks[':'.join(parts)] = np.roll(template.group_mask('g', name), offset, axis=1)
            writer.add_group(group, masks)
            for run_id in ('f', 'r'):
                run = _plain(template.runs[run_id])
                run.update(run_id=identity(run_id), group_id=group_id,
                    edge_ids=[identity(value) for value in run['edge_ids']],
                    seed_ids=[identity(value) for value in run['seed_ids']],
                    held_out_ids=[identity(value) for value in run['held_out_ids']])
                writer.add_run(run, {frame: np.roll(template.raw_mask(run_id, frame), offset, axis=1)
                                    for frame in run['expected_frames']})
        return writer.commit()


def _selected_pixels(bundle, receipt):
    planes = {}
    for direction in ('forward', 'backward'):
        for _run_id, frame, plane in iter_selected_planes(bundle, receipt, direction=direction):
            if frame not in planes:
                planes[frame] = np.zeros_like(plane)
            planes[frame] |= plane
    return planes


@pytest.mark.parametrize('modern', (False, True))
def test_restored_family_cannot_preempt_existing_conflicting_output(tmp_path, modern):
    policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0}} if modern else None
    baseline = _bundle(tmp_path / 'baseline', [('z_existing', 0, None)], modern=modern)
    reference = select_sam_proposals(baseline, policy)
    legacy_order = _bundle(tmp_path / 'unordered',
        [('z_existing', 0, None), ('a_restored', 3, None)], modern=modern)
    legacy = select_sam_proposals(legacy_order, policy)
    assert legacy['selected_run_ids'] and all(key.startswith('a_restored:') for key in legacy['selected_run_ids'])
    restored = _bundle(tmp_path / 'restored',
        [('a_restored', 3, 1), ('z_existing', 0, 0)], modern=modern)
    result = select_sam_proposals(restored, policy)
    assert result['selected_run_ids'] == reference['selected_run_ids']
    assert result['group_receipts']['z_existing'] == reference['group_receipts']['z_existing']
    actual, expected = _selected_pixels(restored, result), _selected_pixels(baseline, reference)
    assert actual.keys() == expected.keys()
    for frame in actual:
        np.testing.assert_array_equal(actual[frame], expected[frame])


def test_missing_priorities_keep_historical_id_order_and_restored_ties_are_stable():
    groups = {'z': {}, 'b': {'selection_priority': (1, 'b')},
              'a': {}, 'c': {'selection_priority': [1, 'c']}}
    assert sorted(groups, key=lambda key: _group_selection_key(key, groups[key])) == ['a', 'z', 'b', 'c']


@pytest.mark.parametrize('restored_is_stock', (False, True))
def test_rescue_ties_use_priority_without_changing_stock_precedence(tmp_path, restored_is_stock):
    from tests.test_sam_guarded_rescue import _family, _run
    with SamEvidenceWriter(tmp_path / 'rescue', {'shape_tyx': [5, 128, 320]}) as writer:
        for group_id, tier, flaw in (('a_restored', 1, 'none' if restored_is_stock else 'shallow'),
                                    ('z_existing', 0, 'shallow')):
            group, masks, raw = _family(group_id, flaw=flaw)
            group['selection_priority'] = [tier, group_id]
            writer.add_group(group, masks)
            for direction in (1, -1):
                writer.add_run(_run(group, f'{group_id}:{direction}', direction), raw)
        bundle = writer.commit()
    result = select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 4}})
    winner = 'a_restored' if restored_is_stock else 'z_existing'
    assert result['selected_run_ids'] == [winner + ':-1', winner + ':1']
    assert result['guarded_rescue']['rescued_group_ids'] == ([] if restored_is_stock else ['z_existing'])


@pytest.mark.parametrize('priority', (None, [], [0], [True, 'family'], [2, 'family'],
                                    [0, 'other'], [0, 'family', 'extra'], 'family'))
def test_malformed_priority_fails_before_selection(tmp_path, priority):
    group, masks, raw = fixture_group()
    group['selection_priority'] = priority
    with SamEvidenceWriter(tmp_path / 'bad', {}) as writer:
        writer.add_group(group, masks)
        writer.add_run(fixture_run('f', group), raw)
        bundle = writer.commit()
    with pytest.raises(ValueError, match='selection_priority'):
        select_sam_proposals(bundle)
