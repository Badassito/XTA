"""Independent lookup review: native pixels, read order and trust boundaries."""
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

from XTA import sam_interpolation as publication, sam_mask_reader
from XTA.sam_cyclic import (CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA,
    address_for_unfolded_index, build_cyclic_frame_addressing)
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, _plain, fingerprint
from XTA.sam_filtering import build_mask_filter
from tests.reference_backends.sam_selection import selected_sam_plane as legacy_plane
from tests.test_sam_branch_performance import multiple_edges


def cyclic_owners(tmp_path, *, period=180., cycles=2):
    native = (4, 12, 16)
    count = 7 if cycles == 2 else 10
    if cycles == 2:
        closure = dict(build_cyclic_frame_addressing(native, count-native[0], period_degrees=period))
    else:
        closure = dict(schema=CYCLIC_EXTRAPOLATION_ADDRESSING_SCHEMA,
            native_shape_tyx=native, evidence_shape_tyx=(count, *native[1:]),
            alias_frames=count-native[0], period_degrees=float(period))
    frames = list(range(count))
    addresses = {frame: dict(address_for_unfolded_index(frame, native[0], period_degrees=period)) for frame in frames}
    endpoints = []
    for name, frame in (('A', 0), ('B', count-1)):
        endpoints.append(dict(observation_id=name, original_observation_id='native-'+name,
            frame_index=frame, native_frame_index=addresses[frame]['native_index'], mirror_u=addresses[frame]['mirror_u']))
    group = dict(group_id='g', context_bbox_yx=(2,1,10,9), frame_indices=frames,
        frame_addressing={**closure,'addresses':deepcopy(addresses)}, frame_addresses=deepcopy(addresses),
        native_shape_tyx=native, complete=True, interpolation_min_radius=0.,
        endpoints=endpoints, edges=[dict(edge_id='edge',source_id='A',target_id='B')])
    first, last = np.zeros((8,8),bool), np.zeros((8,8),bool)
    first[1:3,2:4] = True
    last[4:6,3:5] = True
    masks = {'endpoint:A':first, 'endpoint:B':last,
        'evaluation:A':np.ones((8,8),bool), 'evaluation:B':np.ones((8,8),bool)}
    for frame in frames:
        masks[f'acceptance:{frame}'] = np.ones((8,8),bool)
        masks[f'write:{frame}'] = np.ones((8,8),bool)
    masks['write:0'] &= ~first
    masks[f'write:{count-1}'] &= ~last
    raw = {frame: np.eye(8,dtype=bool) if frame % 2 else np.fliplr(np.eye(8,dtype=bool)) for frame in frames}
    raw[0], raw[count-1] = first.copy(), last.copy()
    scope = dict(shape_tyx=native, evidence_shape_tyx=closure['evidence_shape_tyx'],
        frame_addressing=closure, sam_crop_mode='whole')
    if cycles == 3:
        scope['evidence_purpose'] = 'sam_extrapolation'
    with SamEvidenceWriter(tmp_path/'evidence', scope) as writer:
        writer.add_group(group,masks)
        for identifier, reverse in (('f',False),('r',True),('f2',False)):
            expected = frames[::-1] if reverse else frames
            run = dict(run_id=identifier,group_id='g',seed_ids=['B' if reverse else 'A'],
                held_out_ids=['A' if reverse else 'B'],direction='backward' if reverse else 'forward',
                expected_frames=expected,injected_frames=[expected[0]],complete=True,pass_index=1)
            writer.add_run(run,raw)
        bundle = writer.commit()
    receipt = dict(selected_run_ids=['r','f2','f'], mask_filter=build_mask_filter(bundle,enabled=False,min_radius=0.),
        policy_hash='lookup-review-policy', selection_identity='lookup-review-selection')
    return bundle, receipt


def private_lookup(reader, receipt, *, reserved_bytes=None):
    view = {**receipt, 'selected_run_ids':tuple(receipt['selected_run_ids']), 'mask_filter':reader.filter_snapshot(receipt)}
    view.pop('branch_selection',None)
    charge = publication._publication_lookup_reservation(receipt)
    lookup = publication._build_publication_frame_lookup(reader,view,reader.scope['shape_tyx'][1:],
        reserved_bytes=charge if reserved_bytes is None else reserved_bytes)
    return view, lookup, charge


def compare_plane(reader, view, lookup, frame, *, direction=None):
    shape = tuple(reader.scope['shape_tyx'][1:])
    calls = []
    original = sam_mask_reader.effective_candidate_mask
    def observed(bundle, identifier, stored, receipt=None):
        calls.append((str(identifier),int(stored)))
        return original(bundle,identifier,stored,receipt)
    with mock.patch.object(sam_mask_reader,'effective_candidate_mask',observed):
        expected = legacy_plane(reader,view,frame,shape,direction=direction,_records=(reader.groups,reader.runs))
        expected_calls = calls[:]
        calls.clear()
        actual = publication.selected_sam_plane(reader,view,frame,shape,direction=direction,
            _records=(reader.groups,reader.runs),_lookup=lookup)
    np.testing.assert_array_equal(actual,expected)
    assert calls == expected_calls
    return calls


@pytest.mark.parametrize('period',(180.,360.))
@pytest.mark.parametrize('cycles',(2,3))
@pytest.mark.parametrize('direction',(None,1,-1))
def test_aliases_preserve_every_owner_and_stored_frame_access_order(tmp_path,period,cycles,direction):
    bundle,receipt = cyclic_owners(tmp_path,period=period,cycles=cycles)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_ = private_lookup(reader,receipt)
        assert lookup is not None
        for frame in (-1,0,1,2,3,4):
            calls = compare_plane(reader,view,lookup,frame,direction=direction)
            if frame == 0 and direction in (None,-1):
                assert [stored for owner,stored in calls if owner=='r'] == ([4,0] if cycles==2 else [8,4,0])


def test_repeated_selected_owner_is_not_deduplicated(tmp_path):
    bundle,receipt = cyclic_owners(tmp_path,cycles=3)
    receipt['selected_run_ids']=['r','f','r']
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        calls=compare_plane(reader,view,lookup,0)
        assert calls==[('r',8),('r',4),('r',0),('f',0),('f',4),('f',8),('r',8),('r',4),('r',0)]


def test_missing_branch_support_still_performs_the_candidate_access(tmp_path):
    from tests.test_sam_selection_reuse import _cyclic_branch_bundle
    from XTA.sam_policy import select_sam_proposals
    bundle=_cyclic_branch_bundle(tmp_path,180.)
    receipt=select_sam_proposals(bundle,{'sam_bridge_policy':{'version':6,'component_min_radius':0.}})
    assert receipt['selected_run_ids']
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        calls=compare_plane(reader,view,lookup,2)
        assert calls
        recipe=receipt['branch_selection']
        assert any(not any(str(frame) in recipe['edges'][edge]['owner_support'][owner]
                           for edge in recipe['selected_edge_ids_by_run'][owner])
                   for owner,frame in calls)


@pytest.mark.parametrize('field',('scope','groups','runs','records'))
@pytest.mark.parametrize('before_build',(False,True))
def test_replacement_proxy_cannot_borrow_original_binding(tmp_path,field,before_build):
    bundle,receipt=cyclic_owners(tmp_path,cycles=3)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        backing=dict(getattr(bundle,field))
        setattr(bundle,field,MappingProxyType(backing))
        if before_build:
            view,lookup,_=private_lookup(reader,receipt)
            assert lookup is None
        elif field=='runs':
            # The proxy identity remains stable while its external owner changes
            # expected-frame traversal order. Frozen inner records do not help.
            changed=dict(backing['r'])
            changed['expected_frames']=tuple(reversed(changed['expected_frames']))
            backing['r']=MappingProxyType(changed)
        compare_plane(reader,view,lookup,0)


def test_payload_record_proxy_backing_mutation_still_fails_on_empty_candidate(tmp_path):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        backing=dict(bundle.records)
        bundle.records=MappingProxyType(backing)
        key=bundle.runs['f']['candidate_mask_keys']['0']
        corrupt=dict(backing[key]);corrupt['sha256']='0'*64
        backing[key]=MappingProxyType(corrupt)
        for function,options in ((legacy_plane,{}),(publication.selected_sam_plane,{'_lookup':lookup})):
            with pytest.raises(ValueError,match='checksum'):
                function(reader,view,0,(12,16),direction=1,_records=(reader.groups,reader.runs),**options)


def test_real_empty_payload_corruption_is_checked_inside_fast_lookup_call(tmp_path):
    bundle,receipt=cyclic_owners(tmp_path,cycles=3)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        key=bundle.runs['f']['candidate_mask_keys']['0']
        record=bundle.records[key]
        assert record['foreground']==0
        path=bundle.directory/'masks.bin'
        original=path.read_bytes();offset=int(record['offset'])
        damaged=original[:offset]+bytes([original[offset]^1])+original[offset+1:]
        path.write_bytes(damaged)
        try:
            for function,options in ((legacy_plane,{}),(publication.selected_sam_plane,{'_lookup':lookup})):
                with pytest.raises(ValueError,match='checksum|compressed'):
                    function(reader,view,0,(12,16),direction=1,_records=(reader.groups,reader.runs),**options)
        finally:
            path.write_bytes(original)  # Failures must occur at access, not only outer exit.


@pytest.mark.parametrize('frames',((0,2,1,3,4,5,6),(0,1,1,2,3,4,5,6)))
def test_original_but_noncanonical_expected_frames_keep_legacy_order(tmp_path,frames):
    bundle,receipt=cyclic_owners(tmp_path)
    index_path,manifest_path=bundle.directory/'index.json',bundle.directory/'manifest.json'
    index=json.loads(index_path.read_text());manifest=json.loads(manifest_path.read_text())
    index['runs']['f']['expected_frames']=list(frames)
    encoded=json.dumps(index).encode();index_path.write_bytes(encoded)
    manifest['files']['index.json'].update(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    manifest['evidence_fingerprint']=fingerprint({key:value for key,value in manifest.items() if key!='evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    bundle=SamEvidenceBundle.open(bundle.directory)
    receipt['mask_filter']=build_mask_filter(bundle,enabled=False,min_radius=0.)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        for frame in range(4): compare_plane(reader,view,lookup,frame)


@pytest.mark.parametrize('method',('mask','raw_mask','candidate_mask','effective_candidate_mask','group_mask',
                                 'filter_snapshot','effective_raw_mask','measure_effective_raw_mask'))
def test_shadowed_mask_methods_do_not_create_fast_lookup(tmp_path,method):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        original=getattr(reader,method)
        setattr(reader,method,lambda *args,**kwargs:original(*args,**kwargs))
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is None
        compare_plane(reader,view,lookup,0)


def test_lookup_cannot_cross_reader_transaction(tmp_path):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as first:
        view,lookup,_=private_lookup(first,receipt)
        assert lookup is not None
    with bundle.reader(max_cache_bytes=0) as second:
        other={**view,'mask_filter':second.filter_snapshot(receipt)}
        compare_plane(second,other,lookup,0)


@pytest.mark.parametrize('replace_before_build',(False,True))
def test_rebound_geometry_cache_and_external_backing_cannot_supply_fast_addresses(tmp_path,replace_before_build):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        geometry=dict(bundle._frame_geometry)
        groups=dict(geometry['groups'])
        group=dict(groups['g'])
        external_addresses=dict(group['addresses'])
        group['addresses']=MappingProxyType(external_addresses)
        groups['g']=MappingProxyType(group)
        geometry['groups']=MappingProxyType(groups)
        bundle._frame_geometry=MappingProxyType(geometry)
        if replace_before_build:
            view,lookup,_=private_lookup(reader,receipt)
            assert lookup is None
        # Generic publication observes its current derived cache. A cached fast
        # lookup must not retain or trust an externally replaced address map.
        changed=dict(external_addresses[4]);changed['native_index']=1
        external_addresses[4]=MappingProxyType(changed)
        calls=[]
        original=sam_mask_reader.effective_candidate_mask
        def observed(owner,identifier,frame,filter_value=None):
            calls.append((str(identifier),frame))
            return original(owner,identifier,frame,filter_value)
        with mock.patch.object(sam_mask_reader,'effective_candidate_mask',observed):
            expected=publication.selected_sam_plane(reader,view,0,(12,16),_records=(reader.groups,reader.runs))
            expected_calls=calls[:];calls.clear()
            actual=publication.selected_sam_plane(reader,view,0,(12,16),_records=(reader.groups,reader.runs),_lookup=lookup)
        np.testing.assert_array_equal(actual,expected)
        assert calls==expected_calls


@pytest.mark.parametrize('missing',('_frame_geometry','_frame_geometry_source','_frame_geometry_sources','_mask_records_source'))
def test_missing_cache_or_original_binding_uses_generic_publication(tmp_path,missing):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        delattr(bundle,missing)
        fresh_view,fresh_lookup,_=private_lookup(reader,receipt)
        assert fresh_lookup is None
        compare_plane(reader,fresh_view,fresh_lookup,0)
        compare_plane(reader,view,lookup,0)


def test_view_owner_rebinding_and_mutable_outer_receipt_keep_public_semantics(tmp_path):
    bundle,receipt=cyclic_owners(tmp_path,cycles=3)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        original=tuple(view['selected_run_ids'])
        receipt['selected_run_ids'].clear()
        assert view['selected_run_ids']==original  # Private publication snapshot.
        compare_plane(reader,view,lookup,0)
        view['selected_run_ids']=('f','r')
        calls=compare_plane(reader,view,lookup,0)
        assert [owner for owner,_ in calls]==['f']*3+['r']*3
        # The ordinary public path still observes the caller's mutable receipt.
        np.testing.assert_array_equal(publication.selected_sam_plane(reader,receipt,0,(12,16)),np.zeros((12,16),np.uint8))


@pytest.mark.parametrize('frame',(False,0.,'0',np.int64(0)))
@pytest.mark.parametrize('direction',(None,1.,'1',np.int64(-1)))
def test_noncanonical_public_arguments_keep_legacy_conversion(tmp_path,frame,direction):
    bundle,receipt=cyclic_owners(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt)
        assert lookup is not None
        compare_plane(reader,view,lookup,frame,direction=direction)


def test_negative_stored_cyclic_address_is_rejected_at_bundle_open(tmp_path):
    bundle,_=cyclic_owners(tmp_path,cycles=3)
    index_path,manifest_path=bundle.directory/'index.json',bundle.directory/'manifest.json'
    index=json.loads(index_path.read_text());manifest=json.loads(manifest_path.read_text())
    group=index['groups']['g']
    group['frame_indices'][0]=-1
    address=dict(address_for_unfolded_index(-1,4,period_degrees=180.))
    for name in ('frame_addresses',):
        group[name]['-1']=address;group[name].pop('0')
    group['frame_addressing']['addresses']['-1']=address
    group['frame_addressing']['addresses'].pop('0')
    encoded=json.dumps(index).encode();index_path.write_bytes(encoded)
    manifest['files']['index.json'].update(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    manifest['evidence_fingerprint']=fingerprint({key:value for key,value in manifest.items() if key!='evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='outside|nonnegative|frame'):
        SamEvidenceBundle.open(bundle.directory)


def test_reservation_boundary_and_overflow_preserve_fallback_pixels(tmp_path):
    bundle,receipt=cyclic_owners(tmp_path)
    charge=publication._publication_lookup_reservation(receipt)
    assert charge==4096+512*len(receipt['selected_run_ids'])
    with bundle.reader(max_cache_bytes=0) as reader:
        view,lookup,_=private_lookup(reader,receipt,reserved_bytes=charge-1)
        assert lookup is None
        compare_plane(reader,view,lookup,0)
        view,lookup,_=private_lookup(reader,receipt,reserved_bytes=charge)
        assert lookup is not None
        compare_plane(reader,view,lookup,0)
    max_owners=(8*1024**2-4096)//512
    boundary={'selected_run_ids':[f'owner-{i}' for i in range(max_owners)]}
    assert publication._publication_lookup_reservation(boundary)==8*1024**2
    boundary['selected_run_ids'].append('over-budget')
    assert publication._publication_lookup_reservation(boundary)==0


@pytest.mark.parametrize('cyclic',(False,True))
def test_actual_publication_uses_lookup_within_original_reader_allowance(tmp_path,cyclic):
    from XTA.interpolation import RawBBoxMaskStore
    if cyclic:
        bundle,receipt=cyclic_owners(tmp_path/'source')
    else:
        bundle=multiple_edges(tmp_path/'source')
        receipt=dict(selected_run_ids=['r','f'],mask_filter=build_mask_filter(bundle,enabled=False,min_radius=0.),
            policy_hash='lookup-review-policy',selection_identity='lookup-review-selection')
    shape=tuple(bundle.scope['shape_tyx'])
    observations=np.zeros(shape,np.uint8)
    destination=tmp_path/'fast';destination.mkdir()
    slow_destination=tmp_path/'slow';slow_destination.mkdir()
    builds=[];matches=[]
    build=publication._build_publication_frame_lookup
    match=publication._PublicationFrameLookup.matches
    def counted_build(reader,view,shape_yx,**kwargs):
        result=build(reader,view,shape_yx,**kwargs)
        builds.append((reader.max_cache_bytes,kwargs['reserved_bytes'],result))
        return result
    def counted_match(*args,**kwargs):
        result=match(*args,**kwargs);matches.append(result);return result
    fast=slow=None
    try:
        with mock.patch.object(publication,'_build_publication_frame_lookup',counted_build), \
                mock.patch.object(publication._PublicationFrameLookup,'matches',counted_match):
            fast,fast_components,fast_added,stats=publication._publish_directions(bundle,receipt,observations,
                destination,_plain(bundle.scope),1,tmp_path/'fast-work')
        assert len(builds)==1 and builds[0][2] is not None
        assert matches and all(matches)
        cache,reserved,_=builds[0]
        assert cache+reserved==32*1024**2
        assert stats['max_cache_bytes']==cache and stats['peak_cache_bytes']<=cache
        with mock.patch.object(publication,'_build_publication_frame_lookup',return_value=None):
            slow,slow_components,slow_added,_=publication._publish_directions(bundle,receipt,observations,
                slow_destination,_plain(bundle.scope),1,tmp_path/'slow-work')
        np.testing.assert_array_equal(fast,slow)
        assert fast_added==slow_added
        for before,after in zip(slow_components,fast_components):
            for key in ('direction','voxel_count','run_ids','group_ids','observation_roots','connection_status'):
                assert before[key]==after[key]
            old_store,new_store=RawBBoxMaskStore.open(before['path']),RawBBoxMaskStore.open(after['path'])
            try:
                for frame in range(shape[0]):
                    np.testing.assert_array_equal(old_store.decode_slice(frame),new_store.decode_slice(frame))
            finally:
                old_store.close();new_store.close()
    finally:
        for result in (fast,slow):
            if isinstance(result,np.memmap):result._mmap.close()
