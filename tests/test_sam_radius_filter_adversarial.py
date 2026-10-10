"""Adversarial checks for component radius cleanup before SAM spatial clipping."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from XTA.sam_evidence import (SamEvidenceWriter, export_sam_evidence,
                              iter_selected_planes, load_sam_online_selection)
from XTA.sam_filtering import effective_raw_mask, filter_sam_components
from XTA.sam_interpolation import selected_sam_plane
from XTA.sam_policy import replay_sam_proposals, select_sam_proposals
from XTA.sam_replay import replay_sam_directional_nrrds


SHAPE = (64, 80)


def _body():
    result = np.zeros(SHAPE, dtype=bool)
    result[20:35, 20:35] = True
    return result


def _bundle(path, variants, *, threshold=2., reference=None, write=None,
            endpoints=None, edges=None, scope=None):
    reference = _body() if reference is None else reference
    acceptance = np.zeros(SHAPE, dtype=bool)
    acceptance[7:57, 7:73] = True
    write = acceptance if write is None else write
    if endpoints is None:
        endpoints = [dict(observation_id='A', frame_index=0, canonical_label=1),
                     dict(observation_id='B', frame_index=4, canonical_label=2)]
    edges = edges or [dict(edge_id='A-B', source_id='A', target_id='B')]
    group = dict(group_id='family', context_bbox_yx=(0, 0, *SHAPE),
                 frame_indices=list(range(5)), endpoints=endpoints, edges=edges,
                 complete=True, interpolation_min_radius=threshold)
    masks = {}
    for frame in range(5):
        masks[f'acceptance:{frame}'] = acceptance
        mask = write.copy()
        for endpoint in endpoints:
            if endpoint['frame_index'] == frame:
                mask &= ~reference[endpoint['observation_id']] if isinstance(reference, dict) else ~reference
        masks[f'write:{frame}'] = mask
    for endpoint in endpoints:
        silhouette = reference[endpoint['observation_id']] if isinstance(reference, dict) else reference
        masks[f"endpoint:{endpoint['observation_id']}"] = silhouette
        masks[f"evaluation:{endpoint['observation_id']}"] = acceptance
    with SamEvidenceWriter(path / 'bundle', {'shape_tyx': [5, *SHAPE], **(scope or {})}) as writer:
        writer.add_group(group, masks)
        for run_id, raw, direction in variants:
            forward = direction == 'forward'
            seed_ids = ['A'] if forward else [endpoints[-1]['observation_id']]
            held_out = [endpoint['observation_id'] for endpoint in endpoints
                        if endpoint['observation_id'] not in seed_ids]
            writer.add_run(dict(run_id=run_id, group_id='family', direction=direction,
                seed_ids=seed_ids, held_out_ids=held_out,
                expected_frames=list(range(5)) if forward else list(range(4, -1, -1)),
                injected_frames=[0 if forward else 4], complete=True, pass_index=1), raw)
        return writer.commit()


def _raw(mask=None):
    return {frame: (_body() if mask is None else mask.copy()) for frame in range(5)}


def _planes(bundle, receipt, direction='forward'):
    return {frame: mask for _, frame, mask in iter_selected_planes(bundle, receipt, direction=direction)}


@pytest.mark.parametrize('threshold,removed', ((0., False), (0.999, False), (1., True), (1.001, True)))
def test_component_radius_equality_and_zero_use_whole_components(threshold, removed):
    raw = _body()
    raw[45:47, 60:62] = True  # Its maximum inscribed radius is exactly one.
    original = raw.copy()
    filtered, diagnostic = filter_sam_components(raw, threshold)
    assert np.array_equal(raw, original)
    assert np.array_equal(filtered[20:35, 20:35], original[20:35, 20:35])
    assert bool(filtered[45:47, 60:62].any()) is not removed
    assert diagnostic['removed_foreground'] == (4 if removed else 0)
    with pytest.raises(ValueError):
        filtered.setflags(write=True)


@pytest.mark.parametrize('threshold,retained', ((4.999, True), (5., False)))
def test_full_crop_foreground_uses_padded_boundary_distance(threshold, retained):
    raw = np.ones((9, 9), dtype=bool)
    filtered, _ = filter_sam_components(raw, threshold)
    assert np.array_equal(filtered, raw if retained else np.zeros_like(raw))


def test_boolean_validation_fast_path_matches_binary_numeric_inputs_and_rejects_other_values():
    raw=_body()[::2,::2]
    expected,diagnostic=filter_sam_components(raw,2.)
    for dtype in (np.uint8,np.int32,np.float32):
        actual,details=filter_sam_components(raw.astype(dtype),2.)
        assert np.array_equal(actual,expected) and details==diagnostic
        invalid=raw.astype(dtype)
        invalid[0,0]=2
        with pytest.raises(ValueError,match='binary two-dimensional'):
            filter_sam_components(invalid,2.)
    with pytest.raises(ValueError,match='binary two-dimensional'):
        filter_sam_components(raw[None],2.)


def test_attached_thin_spurs_and_diagonal_pixels_are_not_eroded_or_reclassified():
    raw = _body()
    raw[27, 35:61] = True  # Thin but attached to a substantial body.
    raw[35, 35] = True  # Eight-connected to the body's lower-right corner.
    filtered, diagnostic = filter_sam_components(raw, 3.)
    assert np.array_equal(filtered, raw)
    assert diagnostic['raw_component_count'] == 1
    assert diagnostic['removed_component_count'] == 0


def test_multiple_branches_and_a_real_hole_survive_without_cleanup_mutation():
    raw = np.zeros(SHAPE, dtype=bool)
    raw[16:41, 15:40] = True
    raw[24:33, 23:32] = False
    raw[18:35, 49:66] = True
    filtered, diagnostic = filter_sam_components(raw, 3.)
    assert np.array_equal(filtered, raw)
    assert not filtered[28, 27]
    assert diagnostic['retained_component_count'] == 2


def test_small_island_outside_acceptance_and_write_is_removed_before_containment(tmp_path):
    raw = _raw()
    raw[2][2, 2] = True
    original = raw[2].copy()
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    fingerprint = bundle.evidence_fingerprint
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == ['F']
    measurement = receipt['run_receipts']['F']['measurements']
    assert measurement['raw_first_observed_violation'] == 2
    assert measurement['raw_containment'][2]['outside'] == 1
    assert measurement['first_observed_violation'] is None
    assert measurement['containment'][2]['outside'] == 0
    assert np.array_equal(_planes(bundle, receipt)[2], _body())
    assert np.array_equal(bundle.raw_mask('F', 2), original)
    assert bundle.evidence_fingerprint == fingerprint
    bundle.assert_unchanged()


def test_retained_large_outside_island_still_rejects_whole_run(tmp_path):
    raw = _raw()
    raw[2][1:6, 1:6] = True  # Radius three exceeds configured radius two.
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == []
    measurement = receipt['run_receipts']['F']['measurements']
    assert measurement['raw_containment'][2]['outside'] == 25
    assert measurement['containment'][2]['outside'] == 25
    assert measurement['first_observed_violation'] == 2
    assert not _planes(bundle, receipt)[1].any()


def test_zero_radius_leaves_outside_noise_visible_to_containment(tmp_path):
    raw = _raw()
    raw[2][2, 2] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')], threshold=0.)
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == []
    measurement = receipt['run_receipts']['F']['measurements']
    assert measurement['containment'][2]['outside'] == 1
    assert np.array_equal(effective_raw_mask(bundle, 'F', 2, receipt), raw[2])


def test_full_raw_radius_is_applied_before_thin_write_domain_clipping(tmp_path):
    raw = _raw()
    write = np.zeros(SHAPE, dtype=bool)
    write[20:35, 27] = True  # Candidate alone is one pixel thick.
    bundle = _bundle(tmp_path, [('F', raw, 'forward')], threshold=3., write=write)
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == ['F']
    assert np.array_equal(_planes(bundle, receipt)[2], write)
    assert np.array_equal(effective_raw_mask(bundle, 'F', 2, receipt), raw[2])


def test_attached_thin_leaking_spur_is_retained_and_therefore_rejected(tmp_path):
    raw = _raw()
    raw[2][27, 35:79] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')], threshold=3.)
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == []
    effective = effective_raw_mask(bundle, 'F', 2, receipt)
    assert np.array_equal(effective, raw[2])
    assert receipt['run_receipts']['F']['measurements']['containment'][2]['outside'] > 0


def test_removed_small_required_branch_cannot_inherit_raw_connection_claim(tmp_path):
    parent = np.zeros(SHAPE, dtype=bool)
    parent[20:35, 20:61] = True
    daughter = _body()
    small = np.zeros(SHAPE, dtype=bool)
    small[27, 55] = True
    raw = {0: parent, **{frame: daughter | small for frame in (1, 2, 3, 4)}}
    endpoints = [dict(observation_id='A', frame_index=0, canonical_label=1),
                 dict(observation_id='B', frame_index=4, canonical_label=2),
                 dict(observation_id='C', frame_index=4, canonical_label=3)]
    edges = [dict(edge_id='A-B', source_id='A', target_id='B'),
             dict(edge_id='A-C', source_id='A', target_id='C')]
    bundle = _bundle(tmp_path, [('F', raw, 'forward')], threshold=1.,
                     reference={'A': parent, 'B': daughter, 'C': small}, endpoints=endpoints, edges=edges)
    receipt = select_sam_proposals(bundle, {'sam_bridge_policy': {
        'min_endpoint_recall': 0., 'max_endpoint_excess': 1.,
    }})
    assert receipt['selected_run_ids'] == []
    topology = receipt['group_receipts']['family']['candidate_topology']
    connected = {edge['edge_id']: edge['connected'] for edge in topology['edges']}
    assert connected == {'A-B': True, 'A-C': False}
    assert 'all_requested_local_connections_required' in receipt['group_receipts']['family']['reasons']
    assert np.array_equal(bundle.raw_mask('F', 2), raw[2])


def test_rejecting_one_overlap_owner_preserves_other_after_component_filter(tmp_path):
    bad, good = _raw(), _raw()
    bad[1][42:49, 59:66] = True
    bad[2][1:6, 1:6] = True
    good[2][45, 60] = True
    bundle = _bundle(tmp_path, [('F1', bad, 'forward'), ('F2', good, 'forward')])
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == ['F2']
    selected = _planes(bundle, receipt)
    assert selected[2][27, 27]
    assert np.array_equal(selected[1], _body())
    assert not selected[1][45, 60]
    assert not selected[2][45, 60]
    assert bundle.raw_mask('F1', 2)[2, 2]
    assert bundle.raw_mask('F2', 2)[45, 60]


def _read_nrrd(path):
    header, payload = Path(path).read_bytes().split(b'\n\n', 1)
    sizes = next(line.split(b':', 1)[1] for line in header.splitlines() if line.startswith(b'sizes:'))
    shape = tuple(reversed(tuple(map(int, sizes.split()))))
    return np.frombuffer(gzip.decompress(payload), dtype=np.uint8).reshape(shape)


def test_online_packed_and_directional_nrrd_replay_never_restore_removed_noise(tmp_path):
    forward, reverse = _raw(), _raw()
    forward[2][2, 2] = forward[2][45, 60] = True
    reverse[2][3, 3] = reverse[2][46, 61] = True
    bundle = _bundle(tmp_path, [('F', forward, 'forward'), ('R', reverse, 'backward')])
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == ['F', 'R']
    assert np.array_equal(selected_sam_plane(bundle, receipt, 2, SHAPE), _body())
    packed = replay_sam_proposals(bundle, tmp_path / 'packed')
    assert packed['mask_filter'] == receipt['mask_filter']
    with np.load(tmp_path / 'packed' / 'selected_planes.npz') as archive:
        for record in packed['replay_outputs']['packed_plane_index']:
            plane = np.unpackbits(archive[record['key']], bitorder='little', count=np.prod(SHAPE)).reshape(SHAPE)
            expected = _body() if record['native_frame'] in (1, 2, 3) else np.zeros(SHAPE, dtype=bool)
            assert np.array_equal(plane, expected)
    exported = replay_sam_directional_nrrds(bundle, tmp_path / 'nrrd', memory_mib=1)
    for layer in exported['layers']:
        output = _read_nrrd(tmp_path / 'nrrd' / layer['path'])
        assert np.array_equal(output[2], _body())
        assert not output[:, 2, 2].any()
        assert not output[:, 45, 60].any()
    bundle.assert_unchanged()


def test_radius_override_replays_original_evidence_without_mutation(tmp_path):
    raw = _raw()
    raw[2][43:48, 59:64] = True  # Radius three survives two, disappears at three.
    bundle = _bundle(tmp_path, [('F', raw, 'forward')], threshold=2.)
    original = bundle.evidence_fingerprint
    first = select_sam_proposals(bundle)
    assert _planes(bundle, first)[2][45, 61]
    second = replay_sam_proposals(bundle, tmp_path / 'stricter', policy={
        'sam_bridge_policy': {'component_min_radius': 3.},
    })
    assert second['selected_run_ids'] == ['F']
    assert not _planes(bundle, second)[2][45, 61]
    assert second['mask_filter']['thresholds_by_group']['family'] == 3.
    assert first['policy_hash'] != second['policy_hash']
    assert bundle.groups['family']['interpolation_min_radius'] == 2.
    assert bundle.raw_mask('F', 2)[45, 61]
    assert bundle.evidence_fingerprint == original
    bundle.assert_unchanged()


def test_legacy_receipt_uses_original_unfiltered_owned_candidates(tmp_path):
    raw = _raw()
    raw[2][45, 60] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    legacy = {'selected_run_ids': ['F']}
    assert _planes(bundle, legacy)[2][45, 60]
    assert not _planes(bundle, select_sam_proposals(bundle))[2][45, 60]


def test_effective_helper_rejects_stale_loaded_filter_implementation(tmp_path, monkeypatch):
    from XTA import sam_filtering

    raw = _raw()
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    receipt = select_sam_proposals(bundle)
    modified = tmp_path / 'changed_filter.py'
    modified.write_bytes(sam_filtering._SOURCE_PATH.read_bytes() + b'\n# changed implementation\n')
    monkeypatch.setattr(sam_filtering, '_SOURCE_PATH', modified)
    with pytest.raises(RuntimeError, match='implementation changed after loading'):
        effective_raw_mask(bundle, 'F', 2, receipt)


def _resign_filter(spec):
    spec['sha256'] = hashlib.sha256(json.dumps(
        {key: value for key, value in spec.items() if key != 'sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')).hexdigest()


@pytest.mark.parametrize('field,value', (
    ('connectivity', 4),
    ('comparison', 'maximum_inscribed_radius<threshold'),
    ('radius_units', 'model_pixels'),
    ('implementation_sha256', 'different-implementation'),
))
def test_resigned_malformed_filter_contract_cannot_silently_change_geometry(tmp_path, field, value):
    raw = _raw()
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    receipt = select_sam_proposals(bundle)
    altered = json.loads(json.dumps(receipt))
    altered['mask_filter'][field] = value
    _resign_filter(altered['mask_filter'])
    with pytest.raises(ValueError, match='semantics|implementation'):
        selected_sam_plane(bundle, altered, 2, SHAPE)


def test_filter_threshold_mutation_without_matching_fingerprint_fails_closed(tmp_path):
    raw = _raw()
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    altered = select_sam_proposals(bundle)
    altered['mask_filter']['thresholds_by_group']['family'] = 0.
    with pytest.raises(ValueError, match='fingerprint'):
        _planes(bundle, altered)


def test_quality_v2_receipt_missing_filter_is_not_legacy_unfiltered(tmp_path):
    raw = _raw()
    raw[2][45, 60] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')])
    altered = select_sam_proposals(bundle)
    altered.pop('mask_filter')
    with pytest.raises(ValueError, match='missing.*filter'):
        selected_sam_plane(bundle, altered, 2, SHAPE)


def test_new_scope_cannot_restore_removed_pixels_by_downgrading_online_receipt(tmp_path):
    raw = _raw()
    raw[2][45, 60] = True
    bundle = _bundle(tmp_path, [('F', raw, 'forward')],
                     scope={'selection_receipt_required': True})
    with pytest.raises(ValueError, match='retained online selection'):
        load_sam_online_selection(bundle)
    altered = select_sam_proposals(bundle)
    altered.pop('mask_filter')
    altered['resolved_policy']['version'] = 1
    altered['policy_name'] = 'legacy_v1'
    (bundle.directory.parent / 'selection.json').write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='requires.*filter|missing.*filter'):
        load_sam_online_selection(bundle, policy_hash=altered['policy_hash'])
    with pytest.raises(ValueError, match='requires.*filter|missing.*filter'):
        export_sam_evidence(bundle, tmp_path / 'portable')
    assert not (tmp_path / 'portable').exists()
    bundle.assert_unchanged()
