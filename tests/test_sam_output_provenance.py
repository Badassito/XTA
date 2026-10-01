"""Directional SAM receipts remain distinct from final source-grid topology."""
from dataclasses import replace
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.interpolation import NrrdLayerRef
from XTA.outputs import NrrdLayerSink, nrrd_layer_output_suffix
from XTA.reconciliation import EvidenceLayer, reconcile
from XTA.reconciliation_io import read_layer_manifest
from XTA.reconciliation_policy import resolve_reconciliation
from XTA.reconciliation_runtime import reconcile_tta_layers
from XTA.tta_outputs import measure_bridge_output_survival
from XTA.sam_evidence import SamEvidenceWriter, SamEvidenceBundle, load_sam_online_selection


def ref(tmp_path, direction='forward', *, value=None, **extra):
    value = np.zeros((3, 5, 7), np.uint8) if value is None else value
    path = tmp_path / f'{direction}.dat'
    value.tofile(path)
    return NrrdLayerRef(key=direction, name=direction, path=path, shape=value.shape,
        model_name='detector', view_name='transverse', physical_view_name='transverse',
        view_family='orthogonal', source='fullframe', mask_kind='bridge', pass_index=1,
        interpolation_backend='sam', interpolation_direction=direction,
        seed_detector_identity='detector-sha256', sam_bundle_identity='sam-sha256',
        interpolation_policy_identity='policy-sha256', proposal_selection_status='policy_selected',
        sam_group_ids=('family1',), sam_run_ids=(f'run-{direction}',),
        observation_roots=('endpoint1', 'endpoint2'),
        native_transform={'axes': ['t', 'y', 'x'], 'shape_tyx': list(value.shape)},
        selected_bridge_connection_status='connected_in_native_write_region', **extra)


def cfg(tmp_path):
    path = tmp_path / 'policy.py'
    path.write_text("def build_reconciliation():\n    return dict(mode='union')\n")
    return resolve_reconciliation(SimpleNamespace(reconciliation=path, reconciliation_memory_mib=8))


def test_sdf_output_suffix_retains_walkback_candidate_layout():
    assert nrrd_layer_output_suffix(view_token='transverse', source='fullframe',
        mask_kind='bridge', pass_index=2, interpolation_walk_back_index=3,
        interpolation_candidate_index=4) == 'transverse_fullframe_bridge_pass02_walkback03_candidate04'


def test_sam_suffix_has_explicit_direction_and_stable_scope_identities():
    kwargs = dict(view_token='transverse', source='tile', tile_config_id='tile_384',
        mask_kind='bridge', pass_index=2, interpolation_backend='sam',
        seed_detector_identity='detector', sam_bundle_identity='bundle', interpolation_policy_identity='policy')
    forward = nrrd_layer_output_suffix(**kwargs, interpolation_direction='forward')
    backward = nrrd_layer_output_suffix(**kwargs, interpolation_direction='backward')
    assert forward.startswith('transverse_tile_tile_384_sam_bridge_pass02_forward_detector')
    assert '_bundle' in forward and '_policy' in forward
    assert backward != forward
    assert forward == nrrd_layer_output_suffix(**kwargs, interpolation_direction='forward',
        interpolation_walk_back_index=99, interpolation_candidate_index=99)
    with pytest.raises(ValueError, match='direction'):
        nrrd_layer_output_suffix(**kwargs)


def test_two_binary_directional_slots_roundtrip_with_empty_backward_and_native_metadata(tmp_path):
    value = np.zeros((3, 5, 7), np.uint8)
    value[1, 2, 2:5] = 1
    forward = ref(tmp_path, value=value)
    backward = ref(tmp_path, 'backward', segment_extent_ijk=(0, -1, 0, -1, 0, -1),
        segment_extent_shape_tyx=value.shape)
    sink = NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='case',
        output_shape_tyx=value.shape, max_workers=1)
    try:
        # A legacy caller's suffix cannot hide the backend or collapse directions.
        sink.submit_layer(forward, 'old_bridge_pass01')
        sink.submit_layer(backward, 'old_bridge_pass01')
        sink.wait()
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    saved = json.loads(manifest.read_text())
    assert len(saved['layers']) == 2
    assert {entry['interpolation_direction'] for entry in saved['layers']} == {'forward', 'backward'}
    assert all('_sam_bridge_' in entry['filename'] for entry in saved['layers'])
    assert all(entry['interpolation_walk_back_index'] == 0 for entry in saved['layers'])
    assert all(entry['native_transform'] == forward.native_transform for entry in saved['layers'])
    assert all(entry['sam_run_ids'] for entry in saved['layers'])
    with read_layer_manifest(manifest, workspace=tmp_path / 'read') as layers:
        by_direction = {layer.metadata['interpolation_direction']: layer for layer in layers}
        np.testing.assert_array_equal(by_direction['forward'].read_slab(0, 3), value)
        assert not by_direction['backward'].read_slab(0, 3).any()


def test_sam_selection_guard_covers_union_reuse_and_sink(tmp_path):
    bad = replace(ref(tmp_path), proposal_selection_status='generated_complete')
    with pytest.raises(ValueError, match='selection'):
        reconcile_tta_layers([bad], views=[], source_shape_tyx=bad.shape,
            processing_shape_tyx=bad.shape, settings=cfg(tmp_path),
            output_dir=tmp_path / 'published', workspace=tmp_path / 'work',
            assembled_union=np.zeros(bad.shape, np.uint8))
    sink = NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='case',
        output_shape_tyx=bad.shape, max_workers=1)
    try:
        with pytest.raises(ValueError, match='selection'):
            sink.submit_layer(bad, 'bad')
    finally:
        sink.shutdown()


def test_source_receipt_preserves_native_claim_but_reports_final_topology_unknown(tmp_path):
    value = np.zeros((3, 5, 7), np.uint8)
    value[1, 2, 2:5] = 1
    layer = ref(tmp_path, value=value)
    result, report = reconcile_tta_layers([layer], views=[], source_shape_tyx=value.shape,
        processing_shape_tyx=value.shape, settings=cfg(tmp_path),
        output_dir=tmp_path / 'published', workspace=tmp_path / 'work', assembled_union=value.copy())
    receipt = report['layers']['detector/forward']['source_reconciliation_survival']
    assert receipt['selected_voxels'] == receipt['retained_voxels'] == 3
    assert receipt['selected_bridge_connection_status'] == 'connected_in_native_write_region'
    assert receipt['connection_survival'] == 'not_assessed'
    assert report['layers']['detector/forward']['metadata']['interpolation_direction'] == 'forward'
    assert report['sam_connection_survival']['final_output_after_global_postprocessing'] == 'not_assessed'


def test_final_support_removal_never_inherits_native_connected_status(tmp_path):
    value = np.zeros((3, 5, 7), np.uint8)
    value[1, 2, 2:5] = 1
    layer = ref(tmp_path, value=value)
    final = value.copy()
    final[1, 2, 3] = 0
    receipts = measure_bridge_output_survival([layer], final)
    assert receipts[0]['selected_voxels'] == 3
    assert receipts[0]['retained_voxels'] == 2 and receipts[0]['removed_voxels'] == 1
    assert receipts[0]['support_survival'] == 'later_filtered'
    assert receipts[0]['connection_survival'] == 'not_assessed'


def test_sam_directions_are_one_geometry_vote(tmp_path):
    value = np.ones((3, 5, 7), np.uint8)
    refs = [ref(tmp_path, direction, value=value) for direction in ('forward', 'backward')]
    layers = [EvidenceLayer(item.key, item.shape, dict(mask_kind='bridge', physical_view_name='transverse',
        view_name='transverse', source='fullframe', layer_role='additive_component', recomposition_op='union',
        interpolation_backend='sam', interpolation_direction=item.interpolation_direction),
        lambda a, b: value[a:b]) for item in refs]
    output = np.zeros_like(value)
    report = reconcile(layers, shape_tyx=value.shape,
        policy=dict(mode='weighted', grouping='views', threshold=.1, min_sources=2, min_prediction_sources=0),
        write_slab=lambda a, b, v: output.__setitem__(slice(a, b), v), memory_mib=8)
    assert report['group_count'] == 1
    assert not output.any()


def test_legacy_sdf_refs_create_no_sam_survival_reads(tmp_path):
    layer = NrrdLayerRef(key='legacy', name='legacy', path=tmp_path / 'missing', shape=(3, 5, 7),
        source='fullframe', mask_kind='bridge')
    assert measure_bridge_output_survival([layer], np.zeros(layer.shape, np.uint8), memory_mib=.000001) == []


def final_fixture(tmp_path, *, continuation=False, detached=False, min_radius=0):
    shape = (5, 5, 7)
    silhouette = np.zeros(shape[1:], bool)
    silhouette[2, 3] = 1
    group = dict(group_id='group', context_bbox_yx=[0, 0, 5, 7], frame_indices=list(range(5)),
        endpoints=[dict(observation_id='source', frame_index=0), dict(observation_id='target', frame_index=4)],
        edges=[dict(edge_id='edge', source_id='source', target_id='target')],
        interpolation_min_radius=float(min_radius))
    masks = {'endpoint:source': silhouette, 'endpoint:target': silhouette,
        'evaluation:source': np.ones(shape[1:], bool), 'evaluation:target': np.ones(shape[1:], bool)}
    for frame in range(5):
        masks[f'acceptance:{frame}'] = np.ones(shape[1:], bool)
        masks[f'write:{frame}'] = np.ones(shape[1:], bool)
        if frame in (0, 4):
            masks[f'write:{frame}'][2, 3] = False
    raw = {frame: silhouette for frame in range(5)}
    if continuation or detached:
        observed_frames = (2, 3) if continuation else (1, 2, 3)
        for frame in range(5):
            masks[f'edge_contract:edge:{frame}'] = silhouette.copy()
            if frame in observed_frames:
                masks[f'known_foreground:{frame}'] = silhouette
                masks[f'write:{frame}'][2, 3] = False
        if detached:
            raw[2] = silhouette.copy()
            raw[2][2, 5] = True
            masks['edge_contract:edge:2'][2, 5] = True
    with SamEvidenceWriter(tmp_path / 'evidence', dict(shape_tyx=list(shape))) as writer:
        writer.add_group(group, masks)
        writer.add_run(dict(run_id='run', group_id='group', direction='forward',
            expected_frames=list(range(5)), seed_ids=['source'], held_out_ids=['target']),
            raw)
        bundle = writer.commit()
    additions = np.zeros(shape, np.uint8)
    for frame in range(5):
        additions[frame] = bundle.candidate_mask('run', frame)
    layer = replace(ref(tmp_path, value=additions), proposal_evidence_path=str(bundle.directory),
        sam_group_ids=('group',), sam_run_ids=('run',),
        native_transform=dict(kind='identity', view='transverse', shape_tyx=list(shape),
            source_shape_tyx=list(shape), angle_deg=0), interpolation_connectivity=6)
    final = additions.copy()
    final[[0, 4], 2, 3] = 1
    if continuation or detached:
        final[list(observed_frames), 2, 3] = 1
    return layer, final


def test_final_native_connection_survives_with_actual_endpoint_masks(tmp_path):
    layer, final = final_fixture(tmp_path)
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['connection_survival'] == 'survived'
    group = receipt['group_connections'][0]
    assert group['all_requested_edges_connected']
    assert group['edges'][0]['endpoint_survival'][0]['retained_voxels'] == 1
    assert group['connectivity'] == 6


def test_final_connection_audit_cannot_resurrect_radius_filtered_contributors(tmp_path):
    from XTA.sam_policy import select_sam_proposals
    layer, final = final_fixture(tmp_path, min_radius=1)
    bundle = SamEvidenceBundle.open(layer.proposal_evidence_path)

    def keep_complete_run(context):
        return [run['run_id'] for run in context['runs']]

    # The old fixture's one-pixel raw masks are removed by radius>=1. A custom
    # selection may keep its structurally valid run identity, but final topology
    # still has to assess the resulting empty selected additions.
    policy = {'proposal_api_version': 1, 'select_proposals': keep_complete_run}
    selection = select_sam_proposals(bundle, policy=policy)
    (bundle.directory.parent / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    layer = replace(layer, interpolation_policy_identity=selection['policy_hash'])
    # This directional slot publishes the empty filtered addition; unrelated
    # final foreground still happens to contain the old raw one-pixel route.
    np.zeros_like(final).tofile(layer.path)
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['connection_survival'] == 'connection_lost'
    assert receipt['group_connections'][0]['selected_mask_semantics'] == 'receipt_controlled_component_filter'
    assert receipt['group_connections'][0]['edges'][0]['local_surviving_addition_voxels'] == 0


def test_online_selection_loader_rejects_disappeared_or_wrong_filter_receipt(tmp_path):
    layer, _ = final_fixture(tmp_path)
    bundle = SamEvidenceBundle.open(layer.proposal_evidence_path)
    assert load_sam_online_selection(bundle) is None  # Explicit legacy semantics.
    # A new online bundle's required receipt cannot silently become legacy data.
    bundle.scope = {**dict(bundle.scope), 'selection_receipt_required': True}
    with pytest.raises(ValueError, match='retained online selection'):
        load_sam_online_selection(bundle)


def test_new_scope_cannot_downgrade_filtering_by_forging_legacy_selection_metadata(tmp_path):
    from XTA.sam_evidence import export_sam_evidence
    from XTA.sam_policy import select_sam_proposals
    layer, _ = final_fixture(tmp_path)
    bundle = SamEvidenceBundle.open(layer.proposal_evidence_path)
    selection = select_sam_proposals(bundle)
    selection.pop('mask_filter')
    selection['policy_name'] = 'sam_conservative_v1'
    selection['resolved_policy']['version'] = 1
    (bundle.directory.parent / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    bundle.scope = {**dict(bundle.scope), 'selection_receipt_required': True}
    with pytest.raises(ValueError, match='mask filter specification'):
        load_sam_online_selection(bundle)
    with pytest.raises(ValueError, match='mask filter specification'):
        export_sam_evidence(bundle, tmp_path / 'invalid_export')


def test_final_native_connection_cut_cannot_use_remote_existing_route(tmp_path):
    layer, final = final_fixture(tmp_path)
    final[2, 2, 3] = 0
    final[:, 2, 1] = 1
    final[[0, 4], 2, 1:4] = 1
    # The entire final volume is still connected through unchanged remote
    # foreground, which cannot count as survival of this local selected repair.
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['connection_survival'] == 'connection_lost'
    assert not receipt['group_connections'][0]['edges'][0]['connected']


def test_final_endpoint_removal_is_detected_even_when_all_bridge_voxels_survive(tmp_path):
    layer, final = final_fixture(tmp_path)
    final[0, 2, 3] = 0
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['support_survival'] == 'support_preserved'
    assert receipt['connection_survival'] == 'connection_lost'
    assert receipt['group_connections'][0]['edges'][0]['endpoint_survival'][0]['retained_voxels'] == 0


def test_source_resampling_cannot_inherit_native_connection_claim(tmp_path):
    layer, final = final_fixture(tmp_path)
    layer = replace(layer, native_transform={**layer.native_transform, 'kind': 'resampled'})
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['connection_survival'] == 'not_assessed'
    assert 'identity Transverse' in receipt['connection_survival_reason']


def test_saved_layer_proposal_bundle_is_portable_and_independent_of_raw_workspace(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    layer, final = final_fixture(original)
    sink = NrrdLayerSink(nrrd_dir=original / 'nrrd', stem='case', output_shape_tyx=final.shape, max_workers=1)
    try:
        sink.submit_layer(layer, 'selected')
        sink.wait()
        sink.record_final_bridge_survival(measure_bridge_output_survival([layer], final))
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    saved = json.loads(manifest.read_text())
    entry = saved['layers'][0]
    assert entry['proposal_evidence_path'] == '../evidence'
    assert entry['proposal_evidence_path_base'] == 'manifest_directory'
    assert entry['final_connection_survival'] == 'survived'
    portable = tmp_path / 'portable'
    shutil.copytree(original / 'nrrd', portable / 'nrrd')
    shutil.copytree(original / 'evidence', portable / 'evidence')
    # The public layer and packed evidence contain everything needed for replay;
    # the raw layer backing and original evidence location are removed.
    shutil.rmtree(original)
    with read_layer_manifest(portable / 'nrrd' / manifest.name, workspace=tmp_path / 'workspace') as layers:
        bundle = layers[0].proposal_bundle()
        assert list(bundle.runs) == ['run']
        assert bundle.candidate_mask('run', 2)[2, 3]
        assert layers[0].metadata['final_connection_survival'] == 'survived'


def test_output_count_cannot_scale_with_groups_within_a_directional_scope(tmp_path):
    layer = ref(tmp_path)
    sink = NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='case', output_shape_tyx=layer.shape, max_workers=1)
    try:
        sink.submit_layer(layer, 'first')
        with pytest.raises(ValueError, match='Duplicate SAM directional output slot'):
            sink.submit_layer(replace(layer, key='different-group', sam_group_ids=('other',)), 'second')
        sink.wait()
    finally:
        sink.shutdown()


def test_legacy_sdf_sink_has_no_phantom_sam_fields(tmp_path):
    layer = replace(ref(tmp_path), interpolation_backend='', interpolation_direction='',
        proposal_selection_status='', proposal_evidence_path='')
    sink = NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='case', output_shape_tyx=layer.shape, max_workers=1)
    try:
        sink.submit_layer(layer, 'transverse_fullframe_bridge_pass01_walkback01_candidate01')
        sink.wait()
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    entry = json.loads(manifest.read_text())['layers'][0]
    assert 'interpolation_backend' not in entry and 'interpolation_direction' not in entry
    assert 'sam_run_ids' not in entry and 'native_transform' not in entry


def test_final_fixed_observed_continuation_is_used_and_its_removal_is_detected(tmp_path):
    layer, final = final_fixture(tmp_path, continuation=True)
    before = measure_bridge_output_survival([layer], final)[0]
    assert before['selected_voxels'] == 1 and before['connection_survival'] == 'survived'
    final[2, 2, 3] = 0
    after = measure_bridge_output_survival([layer], final)[0]
    assert after['support_survival'] == 'support_preserved'
    assert after['connection_survival'] == 'connection_lost'


def test_detached_addition_does_not_certify_already_connected_observations(tmp_path):
    layer, final = final_fixture(tmp_path, detached=True)
    receipt = measure_bridge_output_survival([layer], final)[0]
    assert receipt['selected_voxels'] == 1 and receipt['support_survival'] == 'support_preserved'
    assert receipt['connection_survival'] == 'connection_lost'
