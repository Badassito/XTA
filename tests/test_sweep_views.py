"""Exact native fanout staging, acknowledged integrity, and bounded sparse reads."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sweep_views as views


@pytest.fixture
def family(tmp_path,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_WORK_DIR',str(tmp_path))
    monkeypatch.setenv('XTA_YOLO_SWEEP_ID','family-a')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE','produce')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS','50,80')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_GIB','1')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY','0')
    return tmp_path/'native_fanout'


def writer(shape,start=0,count=None,padding=0):
    task={'view':SimpleNamespace(name='transverse__tta_a0'),'model_name':'detector','slice_start':start}
    return views.begin_task_fanout(task,SimpleNamespace(conf=.5),backend='openvino_cpu',
        expected_frames=shape[0]-start if count is None else count,padding_count=padding,native_shape=shape)


def seal(shape,receipts,**kwargs):
    return views.seal_view_family('detector','transverse__tta_a0',shape,(50,80),receipts=receipts,**kwargs)


def basic(shape=(4,12,16),start=0,count=None,score=True):
    w=writer(shape,start,count)
    n=shape[0]-start if count is None else count
    expected={c:np.zeros((n,*shape[1:]),np.uint8) for c in w.cutoffs}
    for cent in w.cutoffs:
        for local,z in enumerate(range(start,start+n)):
            plane=expected[cent][local]
            if z%3!=1:plane[2:5,3:7]=1
            scores=np.where(plane,cent+100,0).astype(np.uint8) if score else None
            w.put(cent,z,plane,scores,prediction_count=z+1,frames_with_predictions=int(plane.any()),bbox=(2,5,3,7))
    return w.finish(),expected


def test_exact_multi_range_hydration_and_stats_without_dense_bounds_scan(family):
    shape=(5,12,16)
    a,ma=basic(shape,0,2);b,mb=basic(shape,2,3)
    decision=seal(shape,[b,a])
    assert decision['status']=='ready'
    assert decision['prediction_counts']=={50:15,80:15}
    assert decision['frames_with_predictions']=={50:3,80:3}
    for cent in (50,80):
        target=np.zeros(shape,np.uint8);scores=np.zeros(shape,np.uint8)
        stats=views.hydrate('detector','transverse__tta_a0',cent,target,scores)
        expected=np.concatenate((ma[cent],mb[cent]))
        np.testing.assert_array_equal(target,expected)
        np.testing.assert_array_equal(scores,np.where(expected,cent+100,0))
        np.testing.assert_array_equal(stats['slice_any'],expected.any(axis=(1,2)))
        assert stats['prediction_count']==15 and stats['native_fanout_hydrated']
        assert views.has_complete_view('detector','transverse__tta_a0',cent,native_shape=shape)
    for receipt in (a,b):
        meta=json.loads((Path(receipt['path'])/'metadata.json').read_text())
        assert meta['metrics']['full_plane_bounds_scans']==0
        assert meta['metrics']['encoded_crop_pixels']<2*np.prod(shape)


def test_padding_or_max_contributions_and_real_statistics(family):
    shape=(2,8,10);w=writer(shape,padding=1)
    for cent in w.cutoffs:
        for z in range(2):
            m=np.ones((2,2),np.uint8)
            w.put(cent,z,m,np.full_like(m,150),bbox=(1,3,1,3),prediction_count=2,frames_with_predictions=1)
        w.put(cent,0,np.ones((2,2),np.uint8),np.full((2,2),230,np.uint8),
            bbox=(2,4,2,4),is_padding=True,padding_ordinal=0,prediction_count=999)
    seal(shape,[w.finish()])
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        mask=reader.read_mask_plane(0);score=reader.read_score_plane(0)
        assert mask.sum()==7 and score[2,2]==230 and score[1,1]==150
        assert reader.prediction_count==4 and reader.frames_with_predictions==2
        assert len(list(reader.iter_crops(0)))==2


@pytest.mark.parametrize('route',['python','native'])
def test_resuming_a_frame_after_close_fails_instead_of_dropping_padding(family,monkeypatch,route):
    if route=='python':
        monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_DECODE','python')
    shape=(2,8,10);w=writer(shape,padding=1)
    for cent in w.cutoffs:
        for z in range(2):
            w.put(cent,z,np.ones((2,2),np.uint8),np.full((2,2),150,np.uint8),bbox=(1,3,1,3),
                  prediction_count=1,frames_with_predictions=1)
        w.put(cent,0,np.ones((2,2),np.uint8),np.full((2,2),230,np.uint8),
            bbox=(2,4,2,4),is_padding=True,padding_ordinal=0,prediction_count=999)
    seal(shape,[w.finish()])
    reader=views.open_view('detector','transverse__tta_a0',50)
    crops=reader.iter_crops(0)
    next(crops)  # the range row's crop; the padding row's crop is still pending
    reader.close()
    with pytest.raises(RuntimeError,match='closed'):
        list(crops)


@pytest.mark.parametrize('score_lane',[False,True])
def test_completed_empty_and_present_zero_score_lane_are_distinct(family,score_lane):
    shape=(2,9,11);w=writer(shape)
    for c in w.cutoffs:
        for z in range(2):
            empty=np.empty((0,0),np.uint8)
            w.put(c,z,empty,empty if score_lane else None,bbox=(0,0,0,0),prediction_count=0)
    seal(shape,[w.finish()])
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        assert not reader.has_foreground and reader.has_score_lane==score_lane
        assert not reader.read_mask_plane(0).any()
        score=reader.read_score_plane(0)
        assert (score is not None)==score_lane
        if score_lane:assert not score.any()


def test_bounded_strips_are_read_only_and_stream_file_is_reused(family,monkeypatch):
    monkeypatch.setattr(views,'_CHUNK_BYTES',64)
    shape=(3,20,16);w=writer(shape)
    for c in w.cutoffs:
        for z in range(3):w.put(c,z,np.ones(shape[1:],np.uint8),np.full(shape[1:],200,np.uint8),bbox=(0,20,0,16))
    seal(shape,[w.finish()])
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        for z in range(3):
            crops=list(reader.iter_crops(z));assert len(crops)==5
            for *_,mask,score in crops:
                assert mask.size<=64 and not mask.flags.writeable and not score.flags.writeable
            assert reader.read_mask_plane(z).all()
        assert reader.io_metrics['payload_open_calls']==1


def test_thread_safe_out_of_order_frame_calls_and_duplicates(family):
    shape=(8,4,5);w=writer(shape)
    def put(pair):
        c,z=pair;w.put(c,z,np.ones((1,1),np.uint8),None,bbox=(1,2,2,3),prediction_count=1)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(put,[(c,z) for c in w.cutoffs for z in reversed(range(8))]))
    with pytest.raises(ValueError,match='Duplicate'):put((50,0))
    seal(shape,[w.finish()])
    with views.open_view('detector','transverse__tta_a0',80) as reader:
        assert reader.prediction_count==8 and not reader.has_score_lane


def test_missing_range_or_duplicate_coverage_is_fatal_not_fallback(family):
    a,_=basic((4,12,16),0,2)
    with pytest.raises(views.SweepViewIntegrityError,match='complete grid'):seal((4,12,16),[a])
    with pytest.raises(views.SweepViewIntegrityError):views.has_complete_view('detector','transverse__tta_a0',50)


def test_explicit_unsupported_is_view_wide_and_generic_observations_are_valid(family):
    decision=seal((4,12,16),[{'status':'unsupported','reason':'unqualified resident route'}])
    assert decision['status']=='fallback' and 'unqualified' in decision['reason']
    assert views.view_family_status('detector','transverse__tta_a0',50)=='fallback'
    assert not views.has_complete_view('detector','transverse__tta_a0',80)


def test_lost_acknowledged_file_is_fatal_even_with_an_unsupported_observation(family):
    receipt,_=basic()
    (Path(receipt['path'])/'payload.bin').unlink()
    with pytest.raises(views.SweepViewIntegrityError):seal((4,12,16),[receipt,{'status':'unsupported','reason':'other route'}])


def test_payload_corruption_is_rejected_when_crop_is_consumed(family):
    receipt,_=basic();seal((4,12,16),[receipt])
    path=Path(receipt['path'])/'payload.bin';raw=bytearray(path.read_bytes());raw[views._CHUNK.size]^=1;path.write_bytes(raw)
    assert views.has_complete_view('detector','transverse__tta_a0',50)
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        with pytest.raises(views.SweepViewIntegrityError):list(reader.iter_crops(0))


def test_identity_grid_and_family_mismatch_cannot_hydrate(family,monkeypatch):
    receipt,_=basic();seal((4,12,16),[receipt])
    with pytest.raises(views.SweepViewIntegrityError):views.open_view('detector','transverse__tta_a0',50,native_shape=(4,12,17))
    with pytest.raises(views.SweepViewIntegrityError):views.open_view('detector','transverse__tta_a0',50,identity={'different':True})
    monkeypatch.setenv('XTA_YOLO_SWEEP_ID','family-b')
    with pytest.raises(views.SweepViewIntegrityError):views.has_complete_view('detector','transverse__tta_a0',50)


def test_capacity_refusal_and_abort_are_explicit_pre_ack_fallback(family,monkeypatch):
    w=writer((1,4,5))
    def refuse(_):raise views.SweepViewCapacityError('test budget pressure')
    monkeypatch.setattr(w,'_fund',refuse)
    with pytest.raises(views.SweepViewCapacityError):w.put(50,0,np.ones((1,1),np.uint8),None,bbox=(0,1,0,1))
    receipt=w.abort('test budget pressure')
    assert not (Path(receipt['path'])/'payload.bin').exists()
    decision=seal((1,4,5),[receipt]);assert decision['status']=='fallback'


def test_zero_budget_rejects_before_any_payload_allocation(family,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_GIB','0')
    with pytest.raises(views.SweepViewCapacityError):writer((1,4,5))
    assert not list(family.glob('ranges/*/payload.bin'))


def test_missing_consumer_decision_is_fatal_and_disabled_mode_is_noop(family,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE','consume')
    with pytest.raises(views.SweepViewIntegrityError):views.has_complete_view('detector','transverse__tta_a0',50)
    assert writer((1,4,5)) is None


def test_incomplete_task_cannot_be_acknowledged(family):
    w=writer((2,4,5))
    w.put(50,0,np.empty((0,0),np.uint8))
    with pytest.raises(views.SweepViewIntegrityError,match='all expected'):w.finish()
    assert w.abort('producer stopped')['status']=='fallback'


def test_retirement_preserves_projected_decision_and_releases_payload(family):
    receipt,_=basic();seal((4,12,16),[receipt])
    result=views.retire_view_payloads('detector','transverse__tta_a0',(50,80))
    assert result['status']=='projected' and result['freed_bytes']>0
    assert views.view_family_status('detector','transverse__tta_a0',50)=='projected'
    assert not views.has_complete_view('detector','transverse__tta_a0',50)
    with pytest.raises(views.SweepViewRetiredError):views.open_view('detector','transverse__tta_a0',80)
    assert not (Path(receipt['path'])/'payload.bin').exists()


def test_hydrate_can_explicitly_clear_reused_targets(family):
    receipt,expected=basic();seal((4,12,16),[receipt])
    target=np.ones((4,12,16),np.uint8)
    views.hydrate('detector','transverse__tta_a0',50,target,targets_zeroed=False)
    np.testing.assert_array_equal(target,expected[50])


def test_task_identity_token_is_bound_once_and_checked_by_all_consumers(family):
    task={'view_name':'transverse__tta_a0','model_name':'detector','sweep_fanout_identity':{'input':'source-sha','model':'model-sha','geometry':'view-sha'}}
    w=views.begin_task_fanout(task,SimpleNamespace(),backend='cuda',expected_frames=1,native_shape=(1,4,5))
    for c in w.cutoffs:w.put(c,0,np.empty((0,0),np.uint8))
    receipt=w.finish()
    identity=task['sweep_fanout_identity']
    seal((1,4,5),[receipt],identity=identity)
    assert views.has_complete_view('detector','transverse__tta_a0',50,identity=identity)
    with pytest.raises(views.SweepViewIntegrityError):views.has_complete_view('detector','transverse__tta_a0',50)
    unsupported=views.unsupported_task_fanout(task,SimpleNamespace(),backend='cuda',reason='route',native_shape=(1,4,5))
    assert unsupported['status']=='unsupported' and unsupported['native_shape']==[1,4,5]


def test_nonempty_mask_with_zero_scores_has_a_present_unknown_lane(family):
    w=writer((1,3,4))
    for c in w.cutoffs:w.put(c,0,np.ones((1,2),np.uint8),np.zeros((1,2),np.uint8),bbox=(1,2,1,3))
    seal((1,3,4),[w.finish()])
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        assert reader.has_score_lane and reader.has_foreground
        assert reader.read_score_plane(0) is not None and not reader.read_score_plane(0).any()
        assert reader.read_mask_plane(0).sum()==2


def test_consumer_cutoffs_share_one_verified_index_allocation(family,monkeypatch):
    receipt,_=basic();seal((4,12,16),[receipt])
    original=Path.read_bytes;reads=[]
    def read(path):
        if path.name=='index.bin':reads.append(str(path))
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    with views.open_view('detector','transverse__tta_a0',50) as first, views.open_view('detector','transverse__tta_a0',80) as second:
        assert len(reads)==1
        assert first._ranges[0][1].base is second._ranges[0][1].base


def test_shared_index_structural_validation_runs_only_once(family,monkeypatch):
    receipt,_=basic();seal((4,12,16),[receipt])
    with views.open_view('detector','transverse__tta_a0',50):
        def fail(*_args,**_kwargs):raise AssertionError('revalidated all-cut logical index')
        monkeypatch.setattr(views.np,'arange',fail)
        with views.open_view('detector','transverse__tta_a0',80) as reader:assert reader.shape==(4,12,16)


def test_changed_crop_coordinates_are_not_accepted_as_valid_data(family):
    receipt,_=basic();seal((4,12,16),[receipt])
    path=Path(receipt['path'])/'payload.bin';raw=bytearray(path.read_bytes())
    # Shift the bbox without changing its area or any compressed data bytes.
    fields=list(views._CHUNK.unpack_from(raw));fields[0]+=1;fields[1]+=1
    views._CHUNK.pack_into(raw,0,*fields);path.write_bytes(raw)
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        with pytest.raises(views.SweepViewIntegrityError):list(reader.iter_crops(0))


def test_shared_quota_covers_concurrent_task_reservations(family,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_GIB',str(12/1024))
    first=writer((2,4,5),0,1)
    with pytest.raises(views.SweepViewCapacityError):writer((2,4,5),1,1)
    for c in first.cutoffs:first.put(c,0,np.empty((0,0),np.uint8))
    receipt=first.finish()
    second=writer((2,4,5),1,1)
    for c in second.cutoffs:second.put(c,1,np.empty((0,0),np.uint8))
    assert seal((2,4,5),[receipt,second.finish()])['status']=='ready'


def test_terminal_consumed_state_forbids_future_native_reads(family,monkeypatch):
    receipt,_=basic();seal((4,12,16),[receipt])
    views.retire_view_payloads('detector','transverse__tta_a0',(50,80),terminal_state='consumed')
    assert views.view_family_status('detector','transverse__tta_a0',80)=='consumed'
    assert not views.has_complete_view('detector','transverse__tta_a0',80)
    with pytest.raises(views.SweepViewRetiredError):views.open_view('detector','transverse__tta_a0',50)
    marker=next(family.glob('views/*/metadata.json'));marker.unlink()
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE','consume')
    with pytest.raises(views.SweepViewIntegrityError):views.view_family_status('detector','transverse__tta_a0',50)


def test_default_frame_io_does_not_create_hashes_or_duplicate_crc(family,monkeypatch):
    w=writer((1,4,5))
    def fail(*_args,**_kwargs):raise AssertionError('per-frame optional checksum ran in normal mode')
    with monkeypatch.context() as patch:
        patch.setattr(views.hashlib,'sha256',fail)
        patch.setattr(views.zlib,'crc32',fail)
        for c in w.cutoffs:w.put(c,0,np.ones((1,2),np.uint8),np.full((1,2),200,np.uint8),bbox=(1,2,1,3))
    receipt=w.finish();meta=json.loads((Path(receipt['path'])/'metadata.json').read_text())
    assert meta['payload_sha256'] is None and 'sha256' not in w.index.dtype.names
    seal((1,4,5),[receipt])
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        with monkeypatch.context() as patch:
            patch.setattr(views.hashlib,'sha256',fail)
            patch.setattr(views.zlib,'crc32',fail)
            assert len(list(reader.iter_crops(0)))==1


def test_opt_in_frame_integrity_remains_available(family,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY','1')
    receipt,_=basic();meta=json.loads((Path(receipt['path'])/'metadata.json').read_text())
    assert meta['verify_integrity'] and len(meta['payload_sha256'])==64
    seal((4,12,16),[receipt])
    with views.open_view('detector','transverse__tta_a0',50) as reader:assert list(reader.iter_crops(0))


def test_mask_only_reader_skips_score_bytes_and_inflation(family,monkeypatch):
    # Counts Python zlib calls, so pin the Python route; the native route's
    # mask-only reads are covered in test_sweep_view_native_decode.py.
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_DECODE','python')
    w=writer((1,4,5))
    for c in w.cutoffs:w.put(c,0,np.ones((2,3),np.uint8),np.full((2,3),200,np.uint8),bbox=(1,3,1,4))
    receipt=w.finish();seal((1,4,5),[receipt])
    fields=views._CHUNK.unpack_from((Path(receipt['path'])/'payload.bin').read_bytes())
    original=views.zlib.decompressobj;calls=[]
    def decoder():calls.append(1);return original()
    monkeypatch.setattr(views.zlib,'decompressobj',decoder)
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        crops=list(reader.iter_mask_crops(0));assert len(crops[0])==5 and len(calls)==1
        assert reader.io_metrics['payload_read_bytes']==views._CHUNK.size+fields[4]
    with views.open_view('detector','transverse__tta_a0',50) as reader:
        list(reader.iter_crops(0));assert len(calls)==3
        assert reader.io_metrics['payload_read_bytes']==views._CHUNK.size+fields[4]+fields[5]


def test_caller_certified_alias_reuses_bytes_without_compression_or_scan(family,monkeypatch):
    w=writer((1,4,5))
    w.put(50,0,np.ones((2,3),np.uint8),np.full((2,3),200,np.uint8),bbox=(1,3,1,4),prediction_count=2)
    size=w._used
    def fail(*_args,**_kwargs):raise AssertionError('alias encoded or scanned data')
    with monkeypatch.context() as patch:
        patch.setattr(views.zlib,'compress',fail)
        patch.setattr(views.np,'packbits',fail)
        w.put_alias(80,0,source_cutoff=50,prediction_count=7)
    assert w._used==size and w._metrics['alias_frames']==1
    decision=seal((1,4,5),[w.finish()]);assert decision['prediction_counts']=={50:2,80:7}
    with views.open_view('detector','transverse__tta_a0',50) as a,views.open_view('detector','transverse__tta_a0',80) as b:
        np.testing.assert_array_equal(a.read_mask_plane(0),b.read_mask_plane(0))
        np.testing.assert_array_equal(a.read_score_plane(0),b.read_score_plane(0))


def test_partial_fallback_releases_only_verified_completed_ranges(family):
    receipt,_=basic((4,12,16),0,2)
    pending=views.NativeTaskFanoutWriter(family,'other','other_view',(1,4,5),(50,80),frame_start=0,expected_frames=1)
    path=Path(receipt['path'])
    decision=seal((4,12,16),[receipt,{'status':'unsupported','reason':'other range unavailable'}])
    assert decision['status']=='fallback' and decision['stored_bytes']==0 and decision['discarded_bytes']>0
    assert not path.exists() and pending.path.exists()
    with sqlite3.connect(family/'quota.sqlite') as db:
        assert db.execute('SELECT bytes FROM leases WHERE id=?',(path.name,)).fetchone() is None
        assert db.execute('SELECT bytes FROM leases WHERE id=?',(pending.path.name,)).fetchone() is not None
    assert views.view_family_status('detector','transverse__tta_a0',50)=='fallback'
    pending.abort('test complete')


def test_empty_frame_runs_publish_without_pixel_or_payload_work(family,monkeypatch):
    w=writer((100,32,48))
    def fail(*_args,**_kwargs):raise AssertionError('empty run touched pixels or compression')
    with monkeypatch.context() as patch:
        patch.setattr(views.zlib,'compress',fail)
        patch.setattr(views.np,'packbits',fail)
        for c in w.cutoffs:w.put_empty_range(c,0,100,score_lane=True)
    receipt=w.finish();decision=seal((100,32,48),[receipt])
    assert decision['prediction_counts']=={50:0,80:0}
    with views.open_view('detector','transverse__tta_a0',80) as reader:
        assert not reader.has_foreground and reader.has_score_lane
        assert list(reader.iter_mask_crops(99))==[]
        assert reader.io_metrics['payload_open_calls']==0


def test_unicode_logical_names_keep_exact_product_identity(family):
    w=views.NativeTaskFanoutWriter(family,'模型','横断面',(1,3,4),(50,80),frame_start=0,expected_frames=1)
    for c in w.cutoffs:w.put_empty_range(c,0,1)
    views.seal_view_family('模型','横断面',(1,3,4),(50,80),receipts=[w.finish()])
    assert views.has_complete_view('模型','横断面',50)


def test_explicit_scalar_owned_floor_keeps_counts_without_any_pixel_lane(family,monkeypatch):
    task={'view_name':'transverse__tta_a0','model_name':'detector','slice_start':0}
    w=views.begin_task_fanout(task,SimpleNamespace(),backend='cuda',expected_frames=3,
        native_shape=(3,8,9),scalar_owned_cutoff=50)
    assert w.primary_cutoff==50 and w.scalar_owned_cutoff==50
    def fail(*_args,**_kwargs):raise AssertionError('scalar-owned floor touched pixels or compression')
    with monkeypatch.context() as patch:
        patch.setattr(views.zlib,'compress',fail)
        patch.setattr(views.np,'packbits',fail)
        w.put_scalar_owned_range(50,0,3,prediction_counts=np.array([2,0,3]),
            frames_with_predictions=np.array([True,False,True]))
    assert w._used==0 and (w.index[0]['state']==4).all()
    for z in range(3):w.put(80,z,np.ones((1,2),np.uint8),np.full((1,2),230,np.uint8),bbox=(1,2,1,3),prediction_count=1)
    with pytest.raises(views.SweepViewScalarOwnedError):w.put_alias(80,0,source_cutoff=50)
    with pytest.raises(views.SweepViewScalarOwnedError):w.put(50,0,np.ones((8,9),np.uint8))
    with pytest.raises(views.SweepViewScalarOwnedError):w.put_empty_range(50,0,3)
    decision=seal((3,8,9),[w.finish()])
    assert decision['status']=='ready' and decision['prediction_counts']=={50:5,80:3}
    assert decision['frames_with_predictions']=={50:2,80:3}
    assert decision['has_foreground'][50] is None and decision['has_foreground'][80]
    assert views.view_family_status('detector','transverse__tta_a0',50)=='scalar_owned'
    assert not views.has_complete_view('detector','transverse__tta_a0',50)
    assert views.has_complete_view('detector','transverse__tta_a0',80)
    with pytest.raises(views.SweepViewScalarOwnedError):views.open_view('detector','transverse__tta_a0',50)
    with views.open_view('detector','transverse__tta_a0',80) as reader:
        assert reader.prediction_count==3 and reader.read_mask_plane(1).sum()==2
    assert views.retire_view_payloads('detector','transverse__tta_a0',(50,80))['status']=='projected'


def test_scalar_owned_floor_must_be_explicit_and_fully_covered(family):
    w=writer((2,3,4))
    with pytest.raises(ValueError):w.put_scalar_owned_range(50,0,2,prediction_counts=np.array([1,2]),frames_with_predictions=np.array([1,1]))
    w.abort('test')
    owned=views.NativeTaskFanoutWriter(family,'detector','transverse__tta_a0',(2,3,4),(50,80),
        frame_start=0,expected_frames=2,scalar_owned_cutoff=50)
    owned.put_scalar_owned_range(50,0,1,prediction_counts=np.array([2]),frames_with_predictions=np.array([1]))
    owned.put_empty_range(80,0,2)
    with pytest.raises(views.SweepViewIntegrityError):owned.finish()
    owned.abort('incomplete floor counts')


def test_oversized_crop_row_is_refused_before_acknowledgment(family,monkeypatch):
    monkeypatch.setattr(views,'_CHUNK_BYTES',8)
    w=writer((1,2,9))
    with pytest.raises(views.SweepViewCapacityError):w.put(50,0,np.ones((2,9),np.uint8),bbox=(0,2,0,9))
    assert w.abort('row exceeds strip admission')['status']=='fallback'
