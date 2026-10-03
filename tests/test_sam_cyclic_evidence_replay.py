"""Cyclic aliases remain attributable while publication folds exact native masks."""
from copy import deepcopy
import json

import numpy as np
import pytest

from XTA.sam_cyclic import build_cyclic_frame_addressing, address_for_unfolded_index
from XTA.sam_evidence import (SamEvidenceWriter, SamEvidenceBundle, iter_selected_planes,
    iter_selected_native_crops, native_output_shape_tyx, selected_native_plane, fingerprint)
from XTA.sam_policy import select_sam_proposals, replay_sam_proposals
from XTA.sam_replay import replay_sam_directional_nrrds


def fixture(period=180., *, mapping_mutation=None):
    native = (4, 12, 16)
    recipe = dict(build_cyclic_frame_addressing(native, 2, period_degrees=period))
    frames = (2,3,4)
    addresses = {frame: dict(address_for_unfolded_index(frame, 4, period_degrees=period)) for frame in frames}
    group = dict(group_id='seam', context_bbox_yx=(2,1,10,9), frame_indices=list(frames),
        frame_addressing={**recipe, 'addresses':deepcopy(addresses)},
        frame_addresses=deepcopy(addresses), native_shape_tyx=native, complete=True,
        interpolation_min_radius=0., endpoints=[dict(observation_id='A', original_observation_id='native-A',
            frame_index=2, native_frame_index=2, mirror_u=False),
        dict(observation_id='B-alias', original_observation_id='native-B', frame_index=4,
            native_frame_index=0, mirror_u=period==180.)],
        edges=[dict(edge_id='edge', source_id='A', target_id='B-alias')])
    masks = {}
    first, last = np.zeros((8,8), bool), np.zeros((8,8), bool)
    first[3:5,2:4] = True
    last[3:5,3:5] = True
    for frame in frames:
        masks[f'acceptance:{frame}'] = np.ones((8,8),bool)
        masks[f'write:{frame}'] = np.ones((8,8),bool)
    masks['write:2'] &= ~first
    masks['write:4'] &= ~last
    masks['endpoint:A'], masks['endpoint:B-alias'] = first, last
    masks['evaluation:A'] = masks['evaluation:B-alias'] = np.ones((8,8),bool)
    raw = {2:first.copy(), 3:first.copy(), 4:last.copy()}
    raw[3][6,6] = True
    raw[4][3,1] = True
    run = dict(run_id='forward', group_id='seam', seed_ids=['A'], held_out_ids=['B-alias'],
        direction='forward', expected_frames=list(frames), injected_frames=[2], complete=True, pass_index=1)
    scope = dict(shape_tyx=native, evidence_shape_tyx=recipe['evidence_shape_tyx'],
        frame_addressing=recipe, sam_crop_mode='whole', view_name='azimuthal',
        canvas_transform={'native_frame_axis':'half_turn_azimuthal', 'source_grid_shape_tyx':[20,30,40]})
    if mapping_mutation:
        mapping_mutation(scope,group)
    return scope,group,masks,run,raw


def bundle_at(root, **kwargs):
    scope, group, masks, run, raw = fixture(**kwargs)
    with SamEvidenceWriter(root/'evidence', scope) as writer:
        writer.add_group(group,masks)
        writer.add_run(run,raw)
        return writer.commit()


@pytest.mark.parametrize('period,alias_x', [(180.,13),(360.,2)])
def test_native_plane_fold_uses_saved_parity_and_preserves_unfolded_evidence(tmp_path,period,alias_x):
    bundle = bundle_at(tmp_path,period=period)
    receipt = select_sam_proposals(bundle,policy={'sam_bridge_policy':'permissive'})
    assert receipt['selected_run_ids']==['forward']
    assert native_output_shape_tyx(bundle)==(4,12,16)
    stored = {frame:plane for _,frame,plane in iter_selected_planes(bundle,receipt)}
    assert 4 in stored and stored[4][3,1]
    crops = list(iter_selected_native_crops(bundle,receipt))
    alias = next(row for row in crops if row[1]==4)
    assert alias[2]==0
    assert alias[3]==((2,7,10,15) if period==180. else (2,1,10,9))
    assert not alias[4].flags.writeable
    plane = selected_native_plane(bundle,receipt,0,direction='forward')
    assert plane[5,alias_x] and plane.sum()==1
    assert not plane[5:7,10:12].any() if period==180. else not plane[5:7,4:6].any()
    assert bundle.candidate_mask('forward',4)[3,1]
    assert selected_native_plane(bundle,receipt,1).sum()==0


def test_native_replay_nrrds_fold_aliases_and_retain_projection_metadata(tmp_path):
    from tests.test_sam_replay_tools import _read
    bundle = bundle_at(tmp_path)
    exported = replay_sam_directional_nrrds(bundle,tmp_path/'replay',
        policy={'sam_bridge_policy':'permissive'},memory_mib=1)
    assert exported['shape_tyx']==[4,12,16]
    assert exported['cyclic_aliases_folded']
    assert exported['evidence_coordinate_space']=='unfolded_cyclic_view_crop'
    assert not exported['source_grid_projected']
    assert exported['scope']['canvas_transform']['source_grid_shape_tyx']==[20,30,40]
    actual,_ = _read(tmp_path/'replay'/exported['layers'][0]['path'])
    selection = exported['selection']
    expected = np.stack([selected_native_plane(bundle,selection,frame,direction='forward') for frame in range(4)])
    np.testing.assert_array_equal(actual,expected)
    assert actual[0,5,13] and actual.shape[0]==4


@pytest.mark.parametrize('period', (180., 360.))
def test_packed_replay_index_uses_native_coordinates_and_saved_aliases(tmp_path,monkeypatch,period):
    bundle=bundle_at(tmp_path,period=period)
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE','tiled')
    replay=replay_sam_proposals(bundle,tmp_path/'packed',policy={'sam_bridge_policy':'permissive'})
    output=replay['replay_outputs']
    assert output['diagnostic_coordinate_space']=='view_native_crop'
    assert output['native_shape_tyx']==[4,12,16]
    assert output['frame_addressing']['period_degrees']==period
    assembled=np.zeros((4,12,16),np.uint8)
    rows=output['packed_plane_index']
    assert len({row['key'] for row in rows})==len(rows)
    with np.load(tmp_path/'packed'/'selected_planes.npz') as packed:
        for row in rows:
            if row['direction']!='forward':
                continue
            mask=np.unpackbits(packed[row['key']],bitorder='little',
                count=int(np.prod(row['shape']))).reshape(row['shape'])
            y0,x0,y1,x1=row['context_bbox_yx']
            assembled[row['native_frame'],y0:y1,x0:x1]|=mask
    alias=next(row for row in rows if row['stored_unfolded_frame']==4)
    assert alias['native_frame']==0
    assert alias['stored_frame_address']==dict(address_for_unfolded_index(4,4,period_degrees=period))
    assert alias['stored_unfolded_bbox_yx']==[2,1,10,9]
    expected=np.stack([selected_native_plane(bundle,replay,frame,direction='forward') for frame in range(4)])
    np.testing.assert_array_equal(assembled,expected)
    bundle.assert_unchanged()


@pytest.mark.parametrize('mutate', [
    lambda scope,group: group['frame_addresses'][4].update(native_index=1),
    lambda scope,group: group['frame_addressing']['addresses'][4].update(mirror_u=False),
    lambda scope,group: group['endpoints'][1].update(native_frame_index=1),
    lambda scope,group: group['endpoints'][1].update(mirror_u=False),
    lambda scope,group: group.update(frame_addresses={2:group['frame_addresses'][2]}),
    lambda scope,group: scope.update(shape_tyx=(5,12,16)),
    lambda scope,group: group.update(context_bbox_yx=(2,1,13,9)),
])
def test_corrupt_native_closure_rejected_before_publication(tmp_path,mutate):
    with pytest.raises(ValueError,match='[Cc]yclic|closure|outside'):
        bundle_at(tmp_path,mapping_mutation=mutate)


def test_valid_bundle_checksums_cannot_hide_corrupt_native_address_map(tmp_path):
    bundle = bundle_at(tmp_path)
    index_path=bundle.directory/'index.json'
    index=json.loads(index_path.read_text())
    index['groups']['seam']['frame_addresses']['4']['native_index']=1
    index_path.write_text(json.dumps(index))
    from XTA.sam_evidence import _file_hash
    manifest_path=bundle.directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['files']['index.json'].update(bytes=index_path.stat().st_size,sha256=_file_hash(index_path))
    manifest['evidence_fingerprint']=fingerprint({key:value for key,value in manifest.items() if key!='evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='[Cc]yclic'):
        SamEvidenceBundle.open(bundle.directory)


@pytest.mark.parametrize('field', ('raw_mask_keys','candidate_mask_keys'))
def test_valid_bundle_checksums_cannot_hide_mask_frame_alias_corruption(tmp_path,field):
    bundle=bundle_at(tmp_path)
    index_path=bundle.directory/'index.json'
    index=json.loads(index_path.read_text())
    masks=index['runs']['forward'][field]
    masks['5']=masks.pop('4')  # Valid evidence frame, outside this run's saved map.
    index_path.write_text(json.dumps(index))
    from XTA.sam_evidence import _file_hash
    manifest_path=bundle.directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['files']['index.json'].update(bytes=index_path.stat().st_size,sha256=_file_hash(index_path))
    manifest['evidence_fingerprint']=fingerprint({key:value for key,value in manifest.items() if key!='evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Cyclic SAM mask addresses'):
        SamEvidenceBundle.open(bundle.directory)


def test_cached_cyclic_geometry_still_checks_loaded_helper_source(tmp_path,monkeypatch):
    from XTA import sam_cyclic
    bundle = bundle_at(tmp_path)
    changed = tmp_path/'changed_helper.py'
    changed.write_bytes(sam_cyclic._SOURCE_PATH.read_bytes()+b'\n# changed after import\n')
    monkeypatch.setattr(sam_cyclic,'_SOURCE_PATH',changed)
    with pytest.raises(RuntimeError,match='cyclic.*changed|Cyclic.*changed'):
        native_output_shape_tyx(bundle)


def test_saved_helper_identity_cannot_silently_reinterpret_addresses(tmp_path):
    with pytest.raises(ValueError,match='implementation identity'):
        bundle_at(tmp_path,mapping_mutation=lambda scope,group:
            scope.update(cyclic_implementation_sha256='different-helper'))


def test_effective_radius_filtering_is_applied_before_native_fold(tmp_path):
    from XTA.sam_mask_reader import effective_candidate_mask
    scope,group,masks,run,raw=fixture()
    group['interpolation_min_radius']=1.
    raw[3][1:7,1:7]=True
    raw[4][1:6,2:7]=True
    raw[4][7,0]=True  # Floating one-pixel island removed before mirror/fold.
    with SamEvidenceWriter(tmp_path/'evidence',scope) as writer:
        writer.add_group(group,masks)
        writer.add_run(run,raw)
        bundle=writer.commit()
    receipt=select_sam_proposals(bundle,policy={'sam_bridge_policy':dict(kind='permissive',
        enforce_interpolation_min_radius=True,component_min_radius=1.)})
    assert receipt['selected_run_ids']==['forward']
    effective=effective_candidate_mask(bundle,'forward',4,receipt)
    assert not effective[7,0]
    expected=np.zeros((12,16),np.uint8)
    expected[2:10,7:15]=effective[:,::-1]
    np.testing.assert_array_equal(selected_native_plane(bundle,receipt,0),expected)
    assert bundle.raw_mask('forward',4)[7,0]

