"""Independent tiled-generation halo, ownership, coverage, and replay contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA.config import activate_sam_crop_mode, resolve_sam_crop_mode
from XTA.sam_crop_tiling import tile_grid
from XTA.sam_evidence import SamEvidenceWriter, iter_selected_planes
from XTA.sam_policy import measure_family_agreement, replay_sam_proposals, select_sam_proposals


SHAPE = (48, 1512)
TILES = ((0, 0, 48, 1008, 0, 0, 48, 756),
         (0, 504, 48, 1512, 0, 756, 48, 1512))


def _fixture(tmp_path, *, leak='none', missing_frame=False, direction='forward'):
    body = np.zeros(SHAPE, dtype=bool)
    body[20:35, 20:1400] = True
    acceptance = np.zeros(SHAPE, dtype=bool)
    acceptance[8:40, 8:1504] = True
    masks = {f'acceptance:{frame}': acceptance for frame in range(3)}
    for frame in range(3):
        masks[f'write:{frame}'] = acceptance & ~body if frame in (0, 2) else acceptance
    for endpoint in ('A', 'B'):
        masks[f'endpoint:{endpoint}'] = body
        masks[f'evaluation:{endpoint}'] = acceptance
    group = dict(group_id='family', context_bbox_yx=(0, 0, *SHAPE), frame_indices=[0, 1, 2],
        endpoints=[dict(observation_id='A', frame_index=0, canonical_label=1),
                   dict(observation_id='B', frame_index=2, canonical_label=2)],
        edges=[dict(edge_id='A-B', source_id='A', target_id='B')],
        complete=True, interpolation_min_radius=2.)
    expected = [0, 1, 2] if direction == 'forward' else [2, 1, 0]
    seed_id, target_id = ('A', 'B') if direction == 'forward' else ('B', 'A')
    raw = {frame: np.zeros(SHAPE, dtype=bool) for frame in expected}
    availability = {frame: np.zeros(SHAPE, dtype=bool) for frame in expected}
    with SamEvidenceWriter(tmp_path / 'bundle', {'shape_tyx': [3, *SHAPE], 'sam_crop_mode': 'tiled'}) as writer:
        writer.add_group(group, masks)
        for number, values in enumerate(TILES):
            y0, x0, y1, x1, a0, b0, a1, b1 = values
            frames = {frame: body[y0:y1, x0:x1].copy() for frame in expected}
            if number == 0 and leak != 'none':
                if leak == 'small':
                    frames[1][2, 990-x0] = True
                else:
                    frames[1][2:7, 990-x0:995-x0] = True
            if number == 1 and missing_frame:
                del frames[1]
            descriptor = dict(group_id='family', tile_id=f'tile-{number}',
                crop_bbox_yx=(y0, x0, y1, x1), ownership_bbox_yx=(a0, b0, a1, b1),
                expected_frames=expected, seed_ids=[seed_id], injected_frames=[expected[0]],
                attempted=True, complete=True,
                tracker_scores={str(frame): .8+number*.1 for frame in frames})
            writer.add_run_tile('R', descriptor, frames)
            for frame, mask in frames.items():
                raw[frame][a0:a1, b0:b1] = mask[a0-y0:a1-y0, b0-x0:b1-x0]
                availability[frame][a0:a1, b0:b1] = True
        writer.add_run(dict(run_id='R', group_id='family', generation_mode='tiled',
            direction=direction, expected_frames=expected, seed_ids=[seed_id], held_out_ids=[target_id],
            injected_frames=[expected[0]], complete=True, pass_index=1),
            raw, availability_masks=availability)
        return writer.commit(), body


@pytest.mark.parametrize('shape', ((17, 1008), (659, 1009), (2065, 659), (1009, 1009), (3064, 3024)))
def test_production_tiles_partition_working_crop_with_1008_total_and_128_halo(shape):
    h, w = shape
    crop = (11, 19, 11+h, 19+w)
    tiles = tile_grid(crop)
    owners = np.zeros(shape, dtype=np.uint8)
    for tile in tiles:
        y0, x0, y1, x1 = tile.crop_bbox_yx
        a0, b0, a1, b1 = tile.ownership_bbox_yx
        assert y1-y0 <= 1008 and x1-x0 <= 1008
        assert 11 <= y0 <= a0 < a1 <= y1 <= 11+h
        assert 19 <= x0 <= b0 < b1 <= x1 <= 19+w
        if a0 > crop[0]:
            assert a0-y0 >= 128
        if a1 < crop[2]:
            assert y1-a1 >= 128
        if b0 > crop[1]:
            assert b0-x0 >= 128
        if b1 < crop[3]:
            assert x1-b1 >= 128
        owners[a0-11:a1-11, b0-19:b1-19] += 1
    assert np.all(owners == 1)


def test_env_snapshot_restores_after_nested_failure_without_outer_crop_change(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    with activate_sam_crop_mode('tiled'):
        monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'invalid')
        assert resolve_sam_crop_mode() == 'tiled'
        with pytest.raises(RuntimeError):
            with activate_sam_crop_mode('whole'):
                assert resolve_sam_crop_mode() == 'whole'
                raise RuntimeError('leave nested launch')
        assert resolve_sam_crop_mode() == 'tiled'
    with pytest.raises(ValueError, match='whole or tiled'):
        resolve_sam_crop_mode()


@pytest.mark.parametrize('mode', ('whole', 'tiled'))
def test_mode_uses_current_working_canvas_without_forcing_native_scale(tmp_path, mode):
    from XTA.geometry import ViewInfo
    from XTA.sam_integration import SamInterpolationContext
    source = SimpleNamespace(shape=(3, 3072, 3072))
    observed = np.zeros((3, 1024, 1024), dtype=np.uint8)
    view = ViewInfo(name='transverse__tta_a0', physical_view_name='transverse',
        num_slices=3, src_h=3072, src_w=3072, family='orthogonal', pad_mode='clamp')
    context = SamInterpolationContext(model_path='unused', device_ids=('0',),
        temp_dir=tmp_path / 'temp', evidence_root=tmp_path / 'evidence',
        source_volume=source, source_identity='same-input', crop_mode=mode,
        delayed_native_expansion=True)
    try:
        with mock.patch.object(context, 'image_provider', side_effect=AssertionError('no-job render')) as render, \
             mock.patch.object(context, '_start', side_effect=AssertionError('no-job model')) as start:
            merged, stats, slots = context.interpolate(observed, view=view, scope='working',
                work_dir=tmp_path / 'output', gap_distance=15, min_radius=0.,
                interpolation_walk_back=0, return_bridge_components=True)
        assert merged is observed
        assert stats['sam_crop_mode'] == mode
        assert stats['sam_working_canvas_kind'] == 'detector_processing'
        assert stats['sam_working_canvas_shape_tyx'] == [3, 1024, 1024]
        assert stats['sam_native_view_shape_tyx'] == [3, 3072, 3072]
        assert stats['delayed_native_expansion_at_launch'] is True
        assert len(slots) == 2
        render.assert_not_called()
        start.assert_not_called()
    finally:
        context.close()


def test_large_discarded_halo_leak_rejects_instead_of_hiding_in_owned_cores(tmp_path):
    bundle, body = _fixture(tmp_path, leak='large')
    assert np.array_equal(bundle.raw_mask('R', 1), body)
    assert bundle.halo_union_mask('R', 1)[3, 992]
    receipt = select_sam_proposals(bundle)
    assert receipt['resolved_policy']['version'] == 5
    assert receipt['selected_run_ids'] == []
    measurement = receipt['run_receipts']['R']['measurements']
    assert measurement['containment'][1]['outside'] == 0
    assert measurement['tile_halo_containment'][1]['outside'] == 25
    assert bundle.tile_raw_mask('R', 'tile-0', 1)[3, 992]


def test_small_halo_noise_filtered_for_quality_but_raw_halo_evidence_preserved(tmp_path):
    bundle, body = _fixture(tmp_path, leak='small')
    receipt = select_sam_proposals(bundle)
    assert receipt['selected_run_ids'] == ['R']
    measurement = receipt['run_receipts']['R']['measurements']
    assert measurement['tile_halo_containment'][1]['raw_outside'] == 1
    assert measurement['tile_halo_containment'][1]['outside'] == 0
    output = next(mask for _, frame, mask in iter_selected_planes(bundle, receipt) if frame == 1)
    assert np.array_equal(output, body)
    assert not output[2, 990]
    assert bundle.tile_raw_mask('R', 'tile-0', 1)[2, 990]
    assert bundle.runs['R']['tracker_scores'] is None
    assert bundle.runs['R']['tile_evidence'][0]['tracker_scores']['1'] == .8


def test_halo_never_becomes_output_even_under_permissive_tiled_policy(tmp_path):
    bundle, body = _fixture(tmp_path, leak='large')
    receipt = select_sam_proposals(bundle, {'sam_bridge_policy': 'permissive'})
    assert receipt['selected_run_ids'] == ['R']
    output = next(mask for _, frame, mask in iter_selected_planes(bundle, receipt) if frame == 1)
    assert np.array_equal(output, body)
    assert not output[3, 992]
    bundle.assert_unchanged()


def test_missing_attempted_tile_frame_is_infrastructure_incomplete_not_empty(tmp_path):
    bundle, _ = _fixture(tmp_path, missing_frame=True)
    receipt = select_sam_proposals(bundle, {'sam_bridge_policy': 'permissive'})
    assert receipt['selected_run_ids'] == []
    reasons = receipt['run_receipts']['R']['measurements']['infrastructure_errors']
    assert 'attempted_tile_coverage_incomplete' in reasons
    assert not bundle.availability_mask('R', 1)[:, 756:].any()
    assert bundle.availability_mask('R', 1)[:, :756].all()


def test_explicit_v2_cannot_score_tiled_evidence_as_legacy_whole(tmp_path):
    bundle, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='incompatible|requires v3'):
        select_sam_proposals(bundle, {'sam_bridge_policy': {'version': 2}})


def test_tiled_replay_uses_saved_mode_and_exact_core_masks_regardless_of_env(tmp_path, monkeypatch):
    bundle, body = _fixture(tmp_path, leak='small')
    original = bundle.evidence_fingerprint
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    online = select_sam_proposals(bundle)
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'invalid')
    replay = replay_sam_proposals(bundle, tmp_path / 'replay')
    assert replay['selected_run_ids'] == online['selected_run_ids']
    assert replay['policy_hash'] == online['policy_hash']
    assert replay['resolved_policy']['version'] == 5
    with np.load(tmp_path / 'replay' / 'selected_planes.npz') as archive:
        row = next(row for row in replay['replay_outputs']['packed_plane_index']
                   if row['direction'] == 'forward' and row['native_frame'] == 1)
        mask = np.unpackbits(archive[row['key']], bitorder='little', count=np.prod(SHAPE)).reshape(SHAPE)
        assert np.array_equal(mask, body)
    assert bundle.evidence_fingerprint == original
    bundle.assert_unchanged()


def test_legacy_whole_replay_does_not_read_tiled_env_or_create_v3(tmp_path, monkeypatch):
    from tests.test_sam_radius_filter_adversarial import _bundle, _raw
    bundle = _bundle(tmp_path, [('F', _raw(), 'forward')])
    baseline = select_sam_proposals(bundle)
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'tiled')
    selected = select_sam_proposals(bundle)
    assert selected['resolved_policy']['version'] == 4
    assert selected['selected_run_ids'] == baseline['selected_run_ids']
    assert selected['policy_hash'] == baseline['policy_hash']


def _unknown_endpoint_bundle(tmp_path):
    source, target = np.zeros(SHAPE, dtype=bool), np.zeros(SHAPE, dtype=bool)
    source[20:35, 20:35] = True
    target[20:35, 20:1400] = True
    acceptance = np.zeros(SHAPE, dtype=bool)
    acceptance[8:40, 8:1504] = True
    group = dict(group_id='family', context_bbox_yx=(0, 0, *SHAPE), frame_indices=[0, 1, 2],
        endpoints=[dict(observation_id='A', frame_index=0), dict(observation_id='B', frame_index=2)],
        edges=[dict(edge_id='A-B', source_id='A', target_id='B')], complete=True,
        interpolation_min_radius=2.)
    masks = {f'acceptance:{frame}': acceptance for frame in range(3)}
    masks.update({f'write:{frame}': source if frame == 1 else np.zeros(SHAPE, dtype=bool) for frame in range(3)})
    masks.update({'endpoint:A': source, 'endpoint:B': target,
                  'evaluation:A': acceptance, 'evaluation:B': acceptance})
    raw = {frame: source.copy() for frame in range(3)}
    available = np.zeros(SHAPE, dtype=bool)
    available[:, :756] = True
    with SamEvidenceWriter(tmp_path / 'bundle', {'shape_tyx': [3, *SHAPE], 'sam_crop_mode': 'tiled'}) as writer:
        writer.add_group(group, masks)
        for number, values in enumerate(TILES):
            y0, x0, y1, x1, a0, b0, a1, b1 = values
            attempted = number == 0
            frames = {frame: source[y0:y1, x0:x1].copy() for frame in range(3)} if attempted else {}
            writer.add_run_tile('R', dict(group_id='family', tile_id=f'tile-{number}',
                crop_bbox_yx=(y0, x0, y1, x1), ownership_bbox_yx=(a0, b0, a1, b1),
                expected_frames=[0, 1, 2], seed_ids=['A'], injected_frames=[0],
                attempted=attempted, complete=attempted), frames)
        writer.add_run(dict(run_id='R', group_id='family', generation_mode='tiled', direction='forward',
            expected_frames=[0, 1, 2], seed_ids=['A'], held_out_ids=['B'], injected_frames=[0],
            complete=True, pass_index=1), raw, availability_masks={frame: available for frame in range(3)})
        return writer.commit()


def test_custom_policy_cannot_select_unknown_heldout_endpoint_coverage(tmp_path):
    bundle = _unknown_endpoint_bundle(tmp_path)
    policy = {'proposal_api_version': 1, 'select_proposals': lambda context: ['R']}
    with pytest.raises(ValueError, match='incomplete|structurally invalid'):
        select_sam_proposals(bundle, policy)
    ordinary = select_sam_proposals(bundle, {'sam_bridge_policy': 'permissive'})
    assert ordinary['selected_run_ids'] == []
    assert ordinary['run_receipts']['R']['measurements']['endpoint_agreement'][0]['status'] == 'unknown_spatial_coverage'


@pytest.mark.parametrize('forgery', ('skip_nonempty_seed', 'zero_foreground', 'wrong_seed_frame'))
def test_tile_seed_metadata_cannot_forge_skip_or_change_original_observation(tmp_path, forgery):
    body = np.zeros(SHAPE, dtype=bool)
    body[20:35, 20:1400] = True
    acceptance = np.ones(SHAPE, dtype=bool)
    group = dict(group_id='family', context_bbox_yx=(0, 0, *SHAPE), frame_indices=[0, 1, 2],
        endpoints=[dict(observation_id='A', frame_index=0), dict(observation_id='B', frame_index=2)],
        edges=[dict(edge_id='A-B', source_id='A', target_id='B')], complete=True)
    masks = {f'acceptance:{frame}': acceptance for frame in range(3)}
    masks.update({f'write:{frame}': np.zeros(SHAPE, dtype=bool) for frame in range(3)})
    masks.update({'endpoint:A': body, 'endpoint:B': body,
                  'evaluation:A': acceptance, 'evaluation:B': acceptance})
    tile = dict(group_id='family', tile_id='tile-0', crop_bbox_yx=(0, 0, 48, 1008),
        ownership_bbox_yx=(0, 0, 48, 756), expected_frames=[0, 1, 2], seed_ids=['A'],
        attempted=True, complete=True)
    raw = {frame: body[:, :1008].copy() for frame in range(3)}
    if forgery == 'skip_nonempty_seed':
        tile['attempted'] = False
        raw = {}
    elif forgery == 'zero_foreground':
        tile['seed_foreground'] = 0
    else:
        tile['seed_ids'] = ['B']
    with SamEvidenceWriter(tmp_path / 'never-publish', {}) as writer:
        writer.add_group(group, masks)
        with pytest.raises(ValueError, match='seed|original'):
            writer.add_run_tile('R', tile, raw)
