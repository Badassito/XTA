"""Native owner reuse preserves publication bits, bounded memory and failure gates."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_extrapolation as core, sam_extrapolation_policy as policy
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore
from XTA.sam_evidence import SamEvidenceBundle, _plain
from XTA.sam_mask_reader import SamMaskReader


class Tracker:
    def run(self,**request):
        frames={frame:np.ones_like(request['seed_mask'])
            for frame in range(request['frame_start'],request['frame_stop'])}
        frames[request['seed_frame']]=request['seed_mask'].copy()
        return SimpleNamespace(frames=frames,tracker_scores={},observation_status={},
            receipt={'run_id':request['run_id'],'coverage_complete':True})


def evidence(tmp_path,*,width=40,height=24,cyclic=True):
    volume=np.zeros((9,height,width),np.uint8)
    x=14 if width==40 else 140
    volume[3:5,7:14,x:x+9]=1
    _,stats,_=core.extrapolate_sam_view_volume_pass(volume,work_dir=tmp_path/'source',
        runtime=Tracker(),distance=7,walk_back=0,min_radius=3.,wrap_axis=cyclic,
        _evidence_only=True)
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    receipt=json.loads(Path(stats['sam_selection_receipt_path']).read_text())
    return volume,bundle,receipt


def pixels(components,shape):
    result=[]
    for component in components:
        store=RawBBoxMaskStore.open(component['path'],mmap_payload=False)
        try:
            result.append(np.stack([store.decode_slice(frame) for frame in range(shape[0])]))
        finally:
            store.close()
    return result


def expected(bundle,receipt):
    shape=tuple(bundle.scope['shape_tyx'])
    return [np.stack([policy.selected_extrapolation_plane(bundle,receipt,frame,
        shape_yx=shape[1:],direction=sign) for frame in range(shape[0])]) for sign in (1,-1)]


@pytest.mark.parametrize('cyclic,width',[(False,40),(True,40),(True,640)])
def test_one_owner_read_and_two_receipt_hashes_preserve_native_planes_and_union(tmp_path,monkeypatch,cyclic,width):
    volume,bundle,receipt=evidence(tmp_path,width=width,cyclic=cyclic)
    reference=expected(bundle,receipt)
    calls={'fingerprints':0,'raw':0,'integrity':0}
    old_fingerprint=policy.fingerprint
    def fingerprint(value):
        calls['fingerprints']+=1
        return old_fingerprint(value)
    old_raw=SamMaskReader.raw_mask
    def raw(*args,**kwargs):
        calls['raw']+=1
        return old_raw(*args,**kwargs)
    old_integrity=bundle.assert_unchanged
    def integrity():
        calls['integrity']+=1
        return old_integrity()
    monkeypatch.setattr(policy,'fingerprint',fingerprint)
    monkeypatch.setattr(SamMaskReader,'raw_mask',raw)
    monkeypatch.setattr(bundle,'assert_unchanged',integrity)
    components,added=core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope))
    actual=pixels(components,volume.shape)
    for left,right in zip(actual,reference):
        np.testing.assert_array_equal(left,right)
    assert added==np.count_nonzero(reference[0]|reference[1])
    if cyclic and width==40:
        assert added<sum(np.count_nonzero(plane) for plane in reference)
    assert calls=={'fingerprints':2,'integrity':2,
        'raw':sum(len(receipt['selected_frames_by_run'][rid]) for rid in receipt['selected_run_ids'])}


@pytest.mark.parametrize('empty_direction',['forward','both'])
def test_empty_direction_frames_are_elided_and_standalone_helpers_keep_validation(tmp_path,monkeypatch,empty_direction):
    volume,bundle,receipt=evidence(tmp_path)
    receipt['selected_run_ids']=[rid for rid in receipt['selected_run_ids']
        if empty_direction!='both' and bundle.runs[rid]['direction']!='forward']
    receipt['selection_identity']=policy.fingerprint({k:v for k,v in receipt.items() if k!='selection_identity'})
    seen=[]
    original=IncrementalRawBBoxMaskStoreWriter.consume
    def consume(writer,frame,block):
        seen.append(writer.extra_meta['direction'])
        return original(writer,frame,block)
    monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter,'consume',consume)
    components,added=core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope))
    actual=pixels(components,volume.shape)
    assert not actual[0].any() and 'forward' not in seen
    if empty_direction=='both':
        assert seen==[] and added==0
    receipt['policy_hash']='tampered'
    with pytest.raises(ValueError,match='modified'):
        policy.selected_extrapolation_plane(bundle,receipt,0)


@pytest.mark.parametrize('resign',[False,True])
def test_receipt_mutation_after_index_creation_aborts_both_outputs(tmp_path,monkeypatch,resign):
    _,bundle,receipt=evidence(tmp_path)
    original=SamMaskReader.raw_mask
    changed=False
    def raw(*args,**kwargs):
        nonlocal changed
        mask=original(*args,**kwargs)
        if not changed:
            changed=True
            receipt['policy_hash']='changed-during-publication'
            if resign:
                receipt['selection_identity']=policy.fingerprint({k:v for k,v in receipt.items() if k!='selection_identity'})
        return mask
    monkeypatch.setattr(SamMaskReader,'raw_mask',raw)
    with pytest.raises(ValueError,match='modified|changed during publication'):
        core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope))
    assert not list((tmp_path/'published').glob('*/meta.json'))
    assert not list((tmp_path/'published').glob('*/index.bin'))


def test_payload_mutation_after_cached_reads_is_detected_before_either_finalize(tmp_path,monkeypatch):
    _,bundle,receipt=evidence(tmp_path)
    owners=sum(len(receipt['selected_frames_by_run'][rid]) for rid in receipt['selected_run_ids'])
    original=SamMaskReader.raw_mask
    count=0
    def raw(*args,**kwargs):
        nonlocal count
        mask=original(*args,**kwargs)
        count+=1
        if count==owners:
            with (bundle.directory/'masks.bin').open('r+b') as stream:
                first=stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0]^1]))
        return mask
    monkeypatch.setattr(SamMaskReader,'raw_mask',raw)
    with pytest.raises(ValueError,match='checksum|changed'):
        core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope))
    assert not list((tmp_path/'published').glob('*/meta.json'))


def test_receipt_is_revalidated_after_source_reader_exit(tmp_path,monkeypatch):
    _,bundle,receipt=evidence(tmp_path)
    original=bundle.assert_unchanged
    checks=0
    def integrity():
        nonlocal checks
        original()
        checks+=1
        if checks==2:
            receipt['policy_hash']='changed-while-reader-exited'
    monkeypatch.setattr(bundle,'assert_unchanged',integrity)
    with pytest.raises(ValueError,match='modified'):
        core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope))
    assert not list((tmp_path/'published').glob('*/meta.json'))


@pytest.mark.parametrize('failure',['consume','second_finalize','cancel_between_finalizers','cancel_during_consume'])
def test_failure_or_cancellation_retracts_both_fresh_stores_and_closes_writers(tmp_path,monkeypatch,failure):
    _,bundle,receipt=evidence(tmp_path)
    event=Event()
    writers=[]
    old_abort=IncrementalRawBBoxMaskStoreWriter.abort
    def abort(writer,error):
        writers.append(writer)
        return old_abort(writer,error)
    monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter,'abort',abort)
    if failure in ('consume','cancel_during_consume'):
        original=IncrementalRawBBoxMaskStoreWriter.consume
        def consume(writer,*args,**kwargs):
            if failure=='consume' and writer.extra_meta['direction']=='backward':
                raise RuntimeError('controlled consume failure')
            result=original(writer,*args,**kwargs)
            if failure=='cancel_during_consume':
                event.set()
            return result
        monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter,'consume',consume)
    else:
        original=IncrementalRawBBoxMaskStoreWriter.finalize
        def finalize(writer):
            result=original(writer)
            if failure=='second_finalize' and writer.extra_meta['direction']=='backward':
                raise RuntimeError('controlled second finalizer failure')
            if failure=='cancel_between_finalizers':
                event.set()
            return result
        monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter,'finalize',finalize)
    with pytest.raises(RuntimeError,match='controlled|cancelled'):
        core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope),cancel_event=event)
    assert len(writers)==2 and all(writer._fd is None for writer in writers)
    assert all(not writer.store_dir.exists() for writer in writers)


def test_private_owner_index_is_immutable_and_bound_to_one_active_reader(tmp_path):
    _,bundle,receipt=evidence(tmp_path)
    with bundle.reader() as first:
        index=policy._publication_index(first,receipt,max_metadata_bytes=64*1024**2)
        frame=next(iter(index.owners_by_frame))
        with pytest.raises(TypeError):
            index.owners_by_frame[frame]=((),())
        with bundle.reader() as other:
            with pytest.raises(ValueError,match='another reader'):
                policy._indexed_publication_plane(other,index,frame,direction=0,shape_yx=(24,40))
    with pytest.raises(RuntimeError,match='active transaction'):
        policy._indexed_publication_plane(first,index,frame,direction=0,shape_yx=(24,40))


@pytest.mark.parametrize('kind',['metadata_limit','single_plane_credit'])
def test_reuse_limits_preserve_established_publication_without_omitting_owners(tmp_path,monkeypatch,kind):
    volume,bundle,receipt=(evidence(tmp_path,width=640,height=256) if kind=='single_plane_credit'
        else evidence(tmp_path))
    reference=expected(bundle,receipt)
    kwargs={}
    if kind=='metadata_limit':
        def unavailable(*args,**kwargs):
            raise policy._PublicationReuseLimit
        monkeypatch.setattr(policy,'_publication_index',unavailable)
    else:
        import XTA.sam_resources as resources
        plane=volume.shape[1]*volume.shape[2]
        crop=max((g['context_bbox_yx'][2]-g['context_bbox_yx'][0])
            *(g['context_bbox_yx'][3]-g['context_bbox_yx'][1]) for g in bundle.groups.values())
        # Reuse fits beside one plane; a second full plane exceeds the credit.
        budget=32*1024**2+volume.shape[0]*256+crop+3*plane+plane//2
        monkeypatch.setattr(resources,'validate_live_sam_resource_profile',lambda profile:{'assigned_plane_bytes':budget})
        kwargs['resource_profile']=object()
        calls=[]
        original=policy._indexed_publication_crop
        def crop_read(reader,owner):
            calls.append(owner)
            return original(reader,owner)
        monkeypatch.setattr(policy,'_indexed_publication_crop',crop_read)
    components,added=core._publish(bundle,receipt,tmp_path/'published',_plain(bundle.scope),**kwargs)
    actual=pixels(components,volume.shape)
    for left,right in zip(actual,reference):
        np.testing.assert_array_equal(left,right)
    assert added==np.count_nonzero(reference[0]|reference[1])
    if kind=='single_plane_credit':
        assert len(calls)==2*sum(len(receipt['selected_frames_by_run'][rid]) for rid in receipt['selected_run_ids'])


def test_selected_seed_frame_is_rejected_even_if_caller_resigns_malformed_receipt(tmp_path):
    _,bundle,receipt=evidence(tmp_path)
    rid=receipt['selected_run_ids'][0]
    malformed=deepcopy(receipt)
    malformed['selected_frames_by_run'][rid].append(bundle.runs[rid]['expected_frames'][0])
    malformed['selection_identity']=policy.fingerprint({k:v for k,v in malformed.items() if k!='selection_identity'})
    with pytest.raises(ValueError,match='declared output ownership'):
        core._publish(bundle,malformed,tmp_path/'published',_plain(bundle.scope))
    assert not list((tmp_path/'published').glob('*/meta.json'))


def test_owner_index_refuses_native_address_outside_the_canvas(tmp_path):
    _,bundle,receipt=evidence(tmp_path)
    groups=_plain(bundle.groups)
    rid=receipt['selected_run_ids'][0]
    stored=receipt['selected_frames_by_run'][rid][0]
    group=groups[bundle.runs[rid]['group_id']]
    group['frame_addresses'][str(stored)]['native_index']=bundle.scope['shape_tyx'][0]
    reader=SimpleNamespace(_require_active=lambda:None,scope=bundle.scope,
        evidence_fingerprint=bundle.evidence_fingerprint,runs=bundle.runs,groups=groups)
    with pytest.raises(ValueError,match='outside its native canvas'):
        policy._publication_index(reader,receipt,max_metadata_bytes=64*1024**2)
