"""Independent bounded parent staging, physical ownership and continuation tests."""
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import gc
import os
import threading
import weakref

import numpy as np
import pytest

from XTA import sam_parent_staging as staging
from XTA.interpolation import _DirectUnionBackingLease,PreparedViewResult
from XTA.runtime import close_memmap_array_without_flush,wait_for_retired_memmap_unlinks
from XTA.view_prepare import ViewPrepareLeaseState


class ManualExecutor:
    def __init__(self, *, fail_submit=False):
        self.entries={}
        self.fail_submit=fail_submit

    def submit(self, function, *args, **kwargs):
        if self.fail_submit:raise RuntimeError('controlled submit failure')
        future=Future();self.entries[future]=(function,args,kwargs)
        return future

    def finish(self):
        for future,(function,args,kwargs) in list(self.entries.items()):
            del self.entries[future]
            if not future.set_running_or_notify_cancel():continue
            try:future.set_result(function(*args,**kwargs))
            except BaseException as error:future.set_exception(error)


class TinyTask:
    def __init__(self, root, name, mask, confidence=None, *, keep_temp=False):
        self.model_name='model';self.view=SimpleNamespace(name=name)
        self.union_mm=mask;self.confmap_mm=confidence
        self.union_path=root/(name+'.mask');self.confmap_path=root/(name+'.conf') if confidence is not None else None
        self.d1_shadow_path=None;self.keep_temp_artifacts=keep_temp
        self.close_dense=close_memmap_array_without_flush
        self.calls=0

    def __call__(self):
        self.calls+=1
        assert isinstance(self.union_mm,np.memmap)
        result=PreparedViewResult(self.model_name,self.view.name,'0',0.,self.union_mm,self.union_mm,[])
        result.confidence=self.confmap_mm
        return result


def lease_state():
    return ViewPrepareLeaseState({},set(),{},set(),{})


def admit(leases,task,amount):
    key=(task.model_name,task.view.name)
    leases.leases[key]=_DirectUnionBackingLease(key,amount,phase='postprocess')
    leases.postprocess_views.add(key);leases.postprocess_bytes[key]=amount
    return key


def queue(tmp_path, *, cap=120, ready=lambda:False, bounded=(), keep_temp=False, leases=None):
    check,prepare=ManualExecutor(),ManualExecutor()
    leases=leases or lease_state()
    result=staging.DeferredSamParentQueue(temp_dir=tmp_path/'temp',output_dir=tmp_path/'output',
        checkpoint_executor=check,prepare_executor=prepare,leases=leases,dense_limit=cap,
        ready=ready,keep_temp=keep_temp,bounded_parent_keys=bounded)
    return result,check,prepare,leases


def owning_mask():
    result=np.zeros((3,4,5),np.uint8);result.ravel()[::7]=1
    return result


def test_three_policy_groups_release_real_buffers_before_detector_drain(tmp_path):
    from tests.test_tta_policy_parent_admission import policy_task
    from tests.test_tta_scheduler_boundary import _state,_scheduler
    state=_state();arrays={}
    def ensure(model,view):
        key=(model,view.name)
        arrays[key]=np.zeros((1,1,30),np.uint8)
        state.baseline_union_paths[key]=tmp_path/(view.name+'.mask')
        state.direct_union_backing_leases[key]=_DirectUnionBackingLease(key,30)
        state.direct_union_inference_views.add(key);state.direct_union_inference_bytes[key]=30
    scheduler=_scheduler(tmp_path,state=state,input_overrides=dict(ensure_baseline_workspaces=ensure,
        direct_union_inference_view_limit=1,direct_union_inference_byte_limit=60,
        direct_union_total_dense_byte_limit=60))
    leases=ViewPrepareLeaseState(state.direct_union_backing_leases,state.direct_union_inference_views,
        state.direct_union_inference_bytes,state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    groups=[policy_task(index,ratio=2,size=30,task_id=index) for index in range(3)]
    all_keys={(str(member['model_name']),member['view'].name) for group in groups
              for member in (group,*group['augmentation_pass_tasks'])}
    staged,check,prepare,_=queue(tmp_path,cap=60,leases=leases,bounded=all_keys)
    references=[]
    try:
        for group in groups:
            assert scheduler.direct_union_task_admissible(group)
            scheduler.activate_direct_union_task(group)
            for member in (group,*group['augmentation_pass_tasks']):
                key=(str(member['model_name']),member['view'].name)
                mask=arrays.pop(key);references.append(weakref.ref(mask))
                task=TinyTask(tmp_path,member['view'].name,mask)
                assert leases.handoff(key)
                staged.defer(task,30)
                del mask,task
            check.finish();resumed,released=staged.pump();gc.collect()
            assert released and not resumed and not prepare.entries
            assert not leases.leases and not leases.postprocess_bytes
        assert all(reference() is None for reference in references)
        assert staged.snapshot()['deferred_parents']==6
    finally:staged.close()


@pytest.mark.parametrize('confidence',(False,True))
def test_roundtrip_confidence_and_nojob_dense_result_live_after_queue_close(tmp_path,confidence):
    ready=[False];staged,check,prepare,leases=queue(tmp_path,ready=lambda:ready[0])
    mask=owning_mask();expected=mask.copy()
    conf=np.zeros(mask.shape,np.uint8) if confidence else None
    if conf is not None:conf[:]=np.arange(conf.size,dtype=np.uint8).reshape(conf.shape)
    expected_conf=None if conf is None else conf.copy()
    task=TinyTask(tmp_path,'v',mask,conf);required=mask.nbytes+(0 if conf is None else conf.nbytes)
    admit(leases,task,required);staged.defer(task,required);del mask,conf
    check.finish();staged.pump()
    assert task.union_mm is None and task.confmap_mm is None
    assert not leases.leases and task.calls==0
    ready[0]=True;resumed,_=staged.pump()
    assert sum(leases.postprocess_bytes.values())==required
    prepare.finish();result=next(iter(resumed)).result()
    path=Path(result.final_view_volume_mm.filename)
    staged.close()
    try:
        assert path.exists()
        np.testing.assert_array_equal(result.final_view_volume_mm,expected)
        if confidence:np.testing.assert_array_equal(result.confidence,expected_conf)
        else:assert result.confidence is None
    finally:
        close_memmap_array_without_flush(result.final_view_volume_mm,unlink_path=path)
        confidence_path=None if result.confidence is None else Path(result.confidence.filename)
        if result.confidence is not None:close_memmap_array_without_flush(result.confidence,unlink_path=confidence_path)
        result.native_support_mm=result.final_view_volume_mm=result.confidence=None
        task.union_mm=task.confmap_mm=None
        leases.complete(('model','v'),retain_for_dense_retirement=False)
        gc.collect()
        wait_for_retired_memmap_unlinks(path=path,timeout_s=3)
        if confidence_path is not None:wait_for_retired_memmap_unlinks(path=confidence_path,timeout_s=3)
        staged.finalize_cleanup()
        assert not staged.root.exists()


def test_cap_shrink_refuses_oversized_policy_parent_before_reopening(tmp_path):
    cap=[120];ready=[False]
    staged,check,prepare,leases=queue(tmp_path,cap=lambda:cap[0],ready=lambda:ready[0],bounded={('model','v')})
    task=TinyTask(tmp_path,'v',owning_mask());admit(leases,task,60);staged.defer(task,60)
    check.finish();staged.pump();cap[0]=59;ready[0]=True
    with pytest.raises(RuntimeError,match='exceeds bounded dense limit'):staged.pump()
    assert not prepare.entries and task.union_mm is None and not leases.leases
    staged.close()


def test_rejected_memmap_alias_never_closes_or_unlinks_other_owner(tmp_path):
    path=tmp_path/'owner.dat';root=np.memmap(path,mode='w+',dtype=np.uint8,shape=(3,4,5))
    root[:]=7;root.flush();alias=root[:,:,:]
    task=TinyTask(tmp_path,'alias',alias);task.union_path=path
    calls=[];task.close_dense=lambda value,**kwargs:calls.append((value,kwargs))
    destination=tmp_path/'checkpoint';destination.mkdir()
    try:
        with pytest.raises(ValueError,match='derived/shared'):
            staging.checkpoint_parent(task,destination,tmp_path,60,threading.Event())
        assert calls==[]
        assert task.union_mm is alias and path.exists() and not root._mmap.closed
        assert root[0,0,0]==7
    finally:root._mmap.close()


def test_invalid_confidence_alias_cannot_retire_valid_mask_first(tmp_path):
    mask=owning_mask();conf=np.zeros(mask.shape,np.uint8);alias=conf.view()
    task=TinyTask(tmp_path,'pair',mask,alias);calls=[]
    task.close_dense=lambda value,**kwargs:calls.append(value)
    destination=tmp_path/'checkpoint';destination.mkdir()
    with pytest.raises(ValueError,match='derived/shared'):
        staging.checkpoint_parent(task,destination,tmp_path,120,threading.Event())
    assert calls==[] and task.union_mm is mask and task.confmap_mm is alias


def test_public_defer_rejects_alias_before_submission_or_credit_transfer(tmp_path):
    staged,check,_prepare,leases=queue(tmp_path)
    original=np.zeros((3,4,5),np.uint8);alias=original.view()
    task=TinyTask(tmp_path,'alias',alias);admit(leases,task,60)
    with pytest.raises(ValueError,match='derived/shared'):staged.defer(task,60)
    assert not check.entries and task.union_mm is alias
    assert leases.postprocess_bytes[('model','alias')]==60
    # The real owner remains responsible; the failed handoff grants no credit.
    leases.complete(('model','alias'),retain_for_dense_retirement=False)
    staged.close()


def test_prepare_submit_failure_releases_readmitted_maps_and_credit(tmp_path):
    ready=[False];staged,check,prepare,leases=queue(tmp_path,ready=lambda:ready[0])
    task=TinyTask(tmp_path,'v',owning_mask());admit(leases,task,60);staged.defer(task,60)
    check.finish();staged.pump();prepare.fail_submit=True;ready[0]=True
    with pytest.raises(RuntimeError,match='submit failure'):staged.pump()
    assert task.union_mm is None and not leases.leases and not leases.postprocess_bytes
    staged.close()


def test_cancelled_queue_never_readmits_even_when_ready_changes(tmp_path):
    ready=[False];staged,check,prepare,leases=queue(tmp_path,ready=lambda:ready[0])
    task=TinyTask(tmp_path,'v',owning_mask());admit(leases,task,60);staged.defer(task,60)
    check.finish();staged.pump();staged.cancel('pipeline resources failed');ready[0]=True
    resumed,_=staged.pump()
    assert not resumed and not prepare.entries
    staged.close()


def test_keep_temp_checkpoint_retires_mock_memfd_owner_before_credit(tmp_path,monkeypatch):
    from XTA import runtime
    source=tmp_path/'temporary-RAM-model.dat'
    mapping=np.memmap(source,mode='w+',dtype=np.uint8,shape=(3,4,5));mapping[:]=1;mapping.flush()
    fd=os.open(source,os.O_RDWR);owner_key='reviewer-memfd-'+str(fd)
    runtime._register_memfd_owner(owner_key,fd,'independent staging owner-FD proof')
    mapping._workspace_memfd_owner_key=owner_key
    reference=weakref.ref(mapping._mmap)
    monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:Path(path).resolve()==source.resolve())
    staged,check,_prepare,leases=queue(tmp_path,keep_temp=True)
    task=TinyTask(tmp_path,'v',mapping,keep_temp=True);task.union_path=source
    admit(leases,task,60);staged.defer(task,60);del mapping
    try:
        check.finish()
        # The model's extra descriptor must be gone before scheduler credit,
        # without relying on a later collection to hide retained tmpfs pages.
        assert reference() is None
        with pytest.raises(OSError):os.fstat(fd)
        staged.pump()
        assert owner_key not in runtime._MEMFD_OWNERS and not leases.leases
        saved=staged.deferred[('model','v')][1].mask
        assert saved.path!=source and saved.path.exists()
        raw=np.memmap(saved.path,mode='r',dtype=np.uint8,shape=(3,4,5))
        np.testing.assert_array_equal(raw,np.ones(raw.shape,np.uint8));raw._mmap.close();del raw
        staged.close();staged.finalize_cleanup()
        assert saved.path.exists() # debug evidence is on disk, not retained RAM
    finally:
        runtime._release_memfd_owner_key(owner_key)


def test_memory_shadow_relocation_does_not_expand_dense_canvas(tmp_path,monkeypatch):
    staged,check,prepare,leases=queue(tmp_path)
    source=tmp_path/'temp'/'private-shadow';source.mkdir()
    (source/'meta.json').write_text('{"compact":true}')
    (source/'payload.bin').write_bytes(b'exact sparse payload')
    monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:Path(path).resolve().is_relative_to(source.resolve()))
    task=TinyTask(tmp_path,'d1',None);task.d1_shadow_path=source
    admit(leases,task,60);staged.defer(task,60)
    check.finish();resumed,released=staged.pump()
    assert released and not resumed and not prepare.entries and task.union_mm is None
    assert not source.exists() and task.d1_shadow_path.exists()
    assert (task.d1_shadow_path/'payload.bin').read_bytes()==b'exact sparse payload'
    assert not leases.leases
    staged.close()


@pytest.mark.parametrize('no_progress', [False, True])
def test_shadow_checkpoint_short_writes_preserve_bytes_or_refuse_source_retirement(
        tmp_path, monkeypatch, no_progress):
    source = tmp_path / 'temp' / 'private-shadow'
    source.mkdir(parents=True)
    expected = b'exact sparse payload, including the terminal bytes'
    (source / 'payload.bin').write_bytes(expected)
    root = tmp_path / 'checkpoints'
    root.mkdir()
    task = TinyTask(tmp_path, 'd1', None)
    task.d1_shadow_path = source
    original_open = Path.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def write(self, data):
            if no_progress:
                return 0
            return self.stream.write(data[:max(1, len(data) // 2)])

    def short_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        if path.name == 'payload.bin' and path.is_relative_to(root) and mode == 'xb':
            return ShortWriter(stream)
        return stream

    monkeypatch.setattr(Path, 'open', short_open)
    if no_progress:
        with pytest.raises(OSError, match='write made no progress'):
            staging.checkpoint_parent(task, root, tmp_path / 'temp', 60, threading.Event())
        assert source.exists() and (source / 'payload.bin').read_bytes() == expected
        assert list(root.iterdir()) == []
    else:
        snapshot = staging.checkpoint_parent(task, root, tmp_path / 'temp', 60, threading.Event())
        assert not source.exists()
        assert snapshot.written_bytes == len(expected)
        assert (snapshot.shadow_path / 'payload.bin').read_bytes() == expected


def test_tmpfs_falls_back_to_output_and_both_ram_refuse(tmp_path,monkeypatch):
    temp=tmp_path/'temp';output=tmp_path/'output'
    monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:Path(path).is_relative_to(temp))
    selected=staging.select_checkpoint_root(temp,output)
    assert selected.is_relative_to(output)
    monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:True)
    with pytest.raises(RuntimeError,match='both destinations are memory-backed'):
        staging.select_checkpoint_root(tmp_path/'a',tmp_path/'b')


@pytest.mark.parametrize('foreground', [1, 255])
def test_compact_first_write_restores_exact_native_mask_scores_and_or_max(tmp_path, monkeypatch, foreground):
    ready = [False]
    staged, check, prepare, leases = queue(tmp_path, cap=200000, ready=lambda:ready[0])
    shape = (7, 33, 35)  # Odd native width, empty frames, unaligned crop starts.
    mask = np.zeros(shape, np.uint8)
    mask[0,0,-1] = mask[-1,-1,0] = foreground
    mask[2:5,3:29,5:33] = foreground
    rng = np.random.default_rng(3184)
    scores = rng.integers(0,256,shape,dtype=np.uint8)  # Include evidence outside labels.
    scores[0,0,-1] = 0
    scores[-1,-1,0] = 255
    expected_mask, expected_scores = mask.copy(), scores.copy()
    mask_ref, score_ref = weakref.ref(mask), weakref.ref(scores)
    task = TinyTask(tmp_path, 'compact', mask, scores)
    required = mask.nbytes + scores.nbytes
    key = admit(leases, task, required)
    workspace = []
    active_codec = [False]
    @contextmanager
    def reserve(amount, name):
        assert leases.postprocess_bytes[key] == required
        active_codec[0] = True
        workspace.append(amount)
        try:yield
        finally:active_codec[0] = False
    task.admission = SimpleNamespace(reserve=reserve)
    staged.defer(task, required)
    del mask, scores
    check.finish()
    assert leases.postprocess_bytes[key] == required  # No early credit before pump.
    assert mask_ref() is None and score_ref() is None
    staged.pump()
    assert not leases.leases
    snapshot = staged.deferred[key][1]
    assert snapshot.mask.encoding == 'packed_mask'
    assert snapshot.mask.foreground_value == foreground
    assert snapshot.confidence.encoding == 'score_blocks'
    assert not tuple(snapshot.owned_dir.rglob('*.dat'))  # First disk bytes are compact.
    assert snapshot.written_bytes == snapshot.mask.physical_bytes + snapshot.confidence.physical_bytes
    original_allocate = staging.allocate_workspace_array
    def admitted_allocate(*args, **kwargs):
        assert active_codec[0] and leases.postprocess_bytes[key] == required
        return original_allocate(*args, **kwargs)
    monkeypatch.setattr(staging, 'allocate_workspace_array', admitted_allocate)
    monkeypatch.setattr(staging, '_ram_fits', lambda *_args:False)  # Exact raw restore fallback.
    ready[0] = True
    resumed, _ = staged.pump()
    assert task.union_mm is None and task.confmap_mm is None  # Scheduler did not decode.
    assert leases.postprocess_bytes[key] == required
    prepare.finish()
    result = next(iter(resumed)).result()
    np.testing.assert_array_equal(result.final_view_volume_mm, expected_mask)
    np.testing.assert_array_equal(result.confidence, expected_scores)
    other_mask = np.roll(expected_mask, 1, axis=2)
    other_scores = np.flip(expected_scores, axis=2).copy()
    np.testing.assert_array_equal(np.bitwise_or(result.final_view_volume_mm, other_mask),
                                  np.bitwise_or(expected_mask, other_mask))
    np.testing.assert_array_equal(np.maximum(result.confidence, other_scores),
                                  np.maximum(expected_scores, other_scores))
    assert result.final_view_volume_mm[0,0,-1] == foreground and result.confidence[0,0,-1] == 0
    assert staged.restore_raw_bytes == required
    assert len(workspace) == 2 and all(value >= staging.CODEC_CONTROL_BYTES for value in workspace)
    staged.close()
    for array in (task.union_mm, task.confmap_mm):array._mmap.close()
    task.union_mm = task.confmap_mm = None
    leases.complete(key, retain_for_dense_retirement=False)
    staged.finalize_cleanup()


@pytest.mark.parametrize('kind', ['mixed-mask', 'float32', 'complex64'])
def test_compact_checkpoint_never_quantizes_unsupported_mask_or_confidence(tmp_path, kind):
    shape = (7,33,35)
    mask = np.zeros(shape, np.uint8)
    mask[0,0,0] = 1
    mask[-1,-1,-1] = 255 if kind == 'mixed-mask' else 1
    dtype = np.uint8 if kind == 'mixed-mask' else np.dtype(kind)
    values = np.arange(np.prod(shape),dtype=np.float32).reshape(shape).astype(dtype)
    if dtype != np.uint8:
        values[0,0,0] = -0.0
        values[1,1,1] = np.nan
    wanted_mask, wanted_values = mask.tobytes(), values.tobytes()
    task = TinyTask(tmp_path, kind, mask, values)
    root = tmp_path/'checkpoints';root.mkdir()
    snapshot = staging.checkpoint_parent(task,root,tmp_path,mask.nbytes+values.nbytes,threading.Event())
    assert snapshot.mask.encoding == ('raw' if kind == 'mixed-mask' else 'packed_mask')
    assert snapshot.confidence.encoding == ('score_blocks' if dtype == np.uint8 else 'raw')
    for saved, wanted in ((snapshot.mask,wanted_mask),(snapshot.confidence,wanted_values)):
        restored = saved.open()
        assert restored.tobytes() == wanted
        restored._mmap.close()


def test_ram_first_requires_physical_headroom_dense_limit_and_aggregate_anon_cap(monkeypatch):
    reserve = staging.RAM_RESERVE_BYTES
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:reserve+159)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
    assert not staging._ram_fits(120,40,256)
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:reserve+160)
    assert staging._ram_fits(120,40,256)
    assert not staging._ram_fits(120,40,159)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:159)
    assert not staging._ram_fits(120,40,256)


@pytest.mark.parametrize('failure', ['encode', 'restore', 'cancel-restore'])
def test_compact_failure_or_cancel_settles_owners_and_admission(tmp_path, monkeypatch, failure):
    ready = [False]
    staged, check, prepare, leases = queue(tmp_path, cap=200000, ready=lambda:ready[0])
    mask = np.zeros((7,33,35),np.uint8);mask[:,2:-2,3:-3] = 1
    task = TinyTask(tmp_path, 'v', mask)
    required = mask.nbytes;key = admit(leases,task,required)
    staged.defer(task,required);del mask
    if failure == 'encode':
        from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter
        def fail(*_args):raise RuntimeError('controlled compact encoder failure')
        monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter,'consume',fail)
        check.finish()
        with pytest.raises(RuntimeError,match='encoder failure'):staged.pump()
        assert leases.postprocess_bytes[key] == required  # Failure granted no reusable credit.
        assert task.union_mm is None and list(staged.root.iterdir()) == []
    else:
        check.finish();staged.pump();ready[0] = True
        resumed,_ = staged.pump()
        if failure == 'cancel-restore':
            staged.cancel()
            assert next(iter(resumed)).cancelled()
        else:
            from XTA.interpolation import RawBBoxMaskStore
            def fail(*_args):raise RuntimeError('controlled compact decoder failure')
            monkeypatch.setattr(RawBBoxMaskStore,'decode_slice_crop',fail)
            prepare.finish()
            with pytest.raises(RuntimeError,match='decoder failure'):staged.pump()
            assert task.union_mm is None and not leases.leases
    staged.close();staged.finalize_cleanup()
    assert not leases.leases and not staged.root.exists()


def test_compact_restore_refuses_changed_payload_before_allocating(tmp_path, monkeypatch):
    mask = np.zeros((7,33,35),np.uint8);mask[:,2:-2,3:-3] = 1
    task = TinyTask(tmp_path,'v',mask)
    root = tmp_path/'checkpoints';root.mkdir()
    snapshot = staging.checkpoint_parent(task,root,tmp_path,mask.nbytes,threading.Event())
    chunks = snapshot.mask.path/'chunks.bin'
    original = chunks.read_bytes();chunks.write_bytes(bytes([original[0]^1])+original[1:])
    monkeypatch.setattr(staging,'allocate_workspace_array',lambda *_a,**_k:pytest.fail('allocated before identity check'))
    with pytest.raises(RuntimeError,match='changed before resume'):snapshot.mask.open()


def test_busy_codec_pool_uses_raw_stream_without_waiting_or_extra_credit(tmp_path):
    from XTA.interpolation import _ByteAdmissionPool
    staged,check,_prepare,leases = queue(tmp_path,cap=200000)
    mask = np.ones((7,33,35),np.uint8)
    expected = mask.tobytes()
    task = TinyTask(tmp_path,'busy-codec',mask)
    amount = mask.nbytes;key = admit(leases,task,amount)
    workspace = staging._codec_workspace_bytes(mask.shape)
    pool = _ByteAdmissionPool(workspace,'controlled CPU preparation')
    task.admission = pool
    with pool.reserve(workspace,'already running CPU stage'):
        staged.defer(task,amount);check.finish();staged.pump()
        assert pool.in_use == workspace and not leases.leases
    snapshot = staged.deferred[key][1]
    assert snapshot.mask.encoding == 'raw' and snapshot.codec_fallback_bytes == amount
    assert snapshot.mask.path.read_bytes() == expected
    assert staged.snapshot()['codec_raw_fallback_bytes'] == amount and pool.in_use == 0
    staged.close();staged.finalize_cleanup()


@pytest.mark.parametrize('running', [False, True])
def test_checkpoint_cancel_waits_for_running_owner_before_cleanup(tmp_path,running):
    staged,check,_prepare,leases = queue(tmp_path,cap=200000)
    task = TinyTask(tmp_path,'cpu',np.ones((7,33,35),np.uint8))
    amount = task.union_mm.nbytes;key = admit(leases,task,amount)
    staged.defer(task,amount)
    cpu = next(iter(staged.checkpoint_futures))
    if running:assert cpu.set_running_or_notify_cancel()
    staged.abort()
    if running:
        assert task.union_mm is not None and key in leases.leases
        with pytest.raises(RuntimeError,match='must settle'):staged.close()
        cpu.set_result(None)
    else:
        assert cpu.cancelled() and task.union_mm is None
    if running:check.entries.pop(cpu)
    check.finish();staged.close();staged.finalize_cleanup()
    assert not leases.leases and not check.entries


def test_full_native_plane_codec_allocations_fit_separate_workspace(tmp_path):
    import tracemalloc
    shape = (2,3072,3073)
    mask = np.ones(shape,np.uint8)
    scores = np.full(shape,255,np.uint8);scores[:,::3,::5] = 0
    task = TinyTask(tmp_path,'native',mask,scores)
    root=tmp_path/'checkpoints';root.mkdir()
    workspace=staging._codec_workspace_bytes(shape)
    tracemalloc.start()
    try:
        snapshot=staging.checkpoint_parent(task,root,tmp_path,mask.nbytes+scores.nbytes,threading.Event())
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert peak < workspace, (peak,workspace)
    assert snapshot.mask.encoding=='packed_mask' and snapshot.confidence.encoding=='score_blocks'
    assert snapshot.written_bytes < mask.nbytes+scores.nbytes


def test_compact_mask_restore_drops_previous_bbox_before_decoding_next(tmp_path,monkeypatch):
    from XTA.interpolation import RawBBoxMaskStore
    mask=np.ones((7,33,35),np.uint8)
    task=TinyTask(tmp_path,'crop-lifetime',mask)
    root=tmp_path/'checkpoints';root.mkdir()
    snapshot=staging.checkpoint_parent(task,root,tmp_path,mask.nbytes,threading.Event())
    original=RawBBoxMaskStore.decode_slice_crop
    previous=[None]
    def one_live_crop(store,z):
        assert previous[0] is None or previous[0]() is None
        crop=original(store,z)
        if crop is not None:previous[0]=weakref.ref(crop[-1])
        return crop
    monkeypatch.setattr(RawBBoxMaskStore,'decode_slice_crop',one_live_crop)
    result=snapshot.mask.open()
    try:np.testing.assert_array_equal(result,np.ones(result.shape,np.uint8))
    finally:result._mmap.close()
    assert previous[0]() is None


def test_unusually_wide_row_codec_is_exact_and_separately_bounded(tmp_path):
    import tracemalloc
    shape=(2,1,1024**2+9)
    mask=np.full(shape,255,np.uint8)
    scores=np.full(shape,128,np.uint8);scores[:,:,::7]=0
    expected_mask,expected_scores=mask.tobytes(),scores.tobytes()
    task=TinyTask(tmp_path,'wide',mask,scores)
    root=tmp_path/'checkpoints';root.mkdir()
    charge=staging._codec_workspace_bytes(shape)
    assert charge >= staging.CODEC_CONTROL_BYTES + 5*shape[2]
    tracemalloc.start()
    try:
        snapshot=staging.checkpoint_parent(task,root,tmp_path,mask.nbytes+scores.nbytes,threading.Event())
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert peak < charge, (peak,charge)
    for saved,wanted in ((snapshot.mask,expected_mask),(snapshot.confidence,expected_scores)):
        restored=saved.open()
        try:assert restored.tobytes()==wanted
        finally:restored._mmap.close()


def test_failed_prepare_keeps_derived_restored_alias_readable_and_credited(tmp_path,monkeypatch):
    from XTA.runtime import wait_for_retired_memmap_unlinks
    aliases=[]
    class FailingPrepare(TinyTask):
        def __call__(self):
            aliases.append(self.union_mm[:,1:-1,2:-2])
            raise RuntimeError('controlled post-restore prepare failure')
    ready=[False]
    staged,check,prepare,leases=queue(tmp_path,cap=200000,ready=lambda:ready[0])
    task=FailingPrepare(tmp_path,'alias-failure',np.ones((7,33,35),np.uint8))
    amount=task.union_mm.nbytes;key=admit(leases,task,amount)
    staged.defer(task,amount);check.finish();staged.pump();ready[0]=True
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:False)
    resumed,_=staged.pump();prepare.finish()
    with pytest.raises(RuntimeError,match='post-restore prepare failure'):staged.pump()
    assert leases.postprocess_bytes[key]==amount
    assert task.union_mm is None and np.all(aliases[0]==1)
    owner=aliases[0].base
    assert isinstance(owner,np.memmap) and not owner._mmap.closed
    path=Path(owner.filename)
    staged.close()
    assert np.all(aliases[0]==1)  # Queue close cannot force-close prepared consumers.
    aliases.clear();del owner,resumed
    gc.collect();wait_for_retired_memmap_unlinks(path=path,timeout_s=3)
    leases.complete(key,retain_for_dense_retirement=False)
    staged.finalize_cleanup()


def test_checkpoint_submit_failure_retains_original_owner_and_admission(tmp_path):
    staged, check, _prepare, leases = queue(tmp_path, cap=200000)
    task = TinyTask(tmp_path, 'submit', np.ones((7,33,35),np.uint8))
    amount = task.union_mm.nbytes; key = admit(leases, task, amount)
    check.fail_submit = True
    with pytest.raises(RuntimeError, match='submit failure'):
        staged.defer(task, amount)
    assert task.union_mm is not None and key in leases.leases
    assert not staged.checkpoint_futures
    check.fail_submit = False
    staged.defer(task, amount); check.finish(); staged.pump()
    assert task.union_mm is None and not leases.leases
    staged.close(); staged.finalize_cleanup()


def test_ram_restore_allocates_only_after_fresh_lease_and_leaves_no_raw_restore_path(tmp_path,monkeypatch):
    from XTA import runtime
    ready=[False]
    staged,check,prepare,leases=queue(tmp_path,cap=200000,ready=lambda:ready[0])
    mask=np.ones((7,33,35),np.uint8)
    scores=np.full(mask.shape,255,np.uint8);scores[0,0,0]=0
    task=TinyTask(tmp_path,'ram-restore',mask,scores)
    amount=mask.nbytes+scores.nbytes;key=admit(leases,task,amount)
    old_lease=leases.leases[key]
    staged.defer(task,amount);check.finish();staged.pump()
    owned=staged.deferred[key][1].owned_dir
    calls=[]
    def simulated_memfd(shape,dtype,destination,_description,**kwargs):
        assert leases.leases[key] is not old_lease and leases.postprocess_bytes[key]==amount
        assert kwargs['prefer_memfd'] and not kwargs['prefer_memory']
        path=tmp_path/f'simulated-RAM-{len(calls)}'
        array=np.memmap(path,mode='w+',dtype=dtype,shape=shape)
        descriptor=os.open(path,os.O_RDWR)
        runtime._register_memfd_owner(str(path),descriptor,'independent restored RAM-fd ownership proof')
        array._workspace_memfd_owner_key=str(path)
        array._workspace_memfd_path=str(path)
        calls.append((descriptor,path))
        return array
    monkeypatch.setattr(staging,'allocate_workspace_array',simulated_memfd)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    ready[0]=True;resumed,_=staged.pump();prepare.finish()
    result=next(iter(resumed)).result()
    np.testing.assert_array_equal(result.final_view_volume_mm,mask)
    np.testing.assert_array_equal(result.confidence,scores)
    assert len(calls)==2 and staged.restore_raw_bytes==0
    assert not tuple(owned.rglob('*.restored.dat'))
    staged.close()
    result.native_support_mm=result.final_view_volume_mm=result.confidence=None
    for name in ('union_mm','confmap_mm'):
        array=getattr(task,name);setattr(task,name,None)
        runtime.close_memmap_array_without_flush(array)
    del array
    gc.collect()
    for descriptor,_path in calls:
        with pytest.raises(OSError):os.fstat(descriptor)
    leases.complete(key,retain_for_dense_retirement=False)
    staged.finalize_cleanup()


def test_ram_first_birth_charges_multiple_owners_and_leaves_detector_room(tmp_path,monkeypatch):
    staged,_check,_prepare,_leases=queue(tmp_path,cap=120)
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+1000)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
    first,second=('model','first'),('model','second')
    assert staged.claim_ram_first(first,50,0,50)
    assert staged.owns_ram_first_parent(first) and staged.ram_first_bytes==50
    assert staged.claim_ram_first(second,10,50,50)
    assert staged.ram_first_bytes==60 and len(staged.ram_first_owners)==2
    with pytest.raises(RuntimeError,match='claimed twice'):
        staged.claim_ram_first(second,10,60,50)
    assert not staged.release_ram_first(('model','foreign'))
    assert staged.release_ram_first(first)
    assert staged.ram_first_bytes==10
    assert staged.release_ram_first(second)
    assert not staged.claim_ram_first(second,50,50,50)  # Would leave no detector room.
    assert staged.claim_ram_first(second,50,10,50)
    assert staged.snapshot()['ram_first_grants']==3
    assert staged.release_ram_first(second)
    staged.close();staged.finalize_cleanup()


def test_ram_token_survives_running_codec_cancel_and_releases_after_source_close(tmp_path,monkeypatch):
    staged,_check,_prepare,leases=queue(tmp_path,cap=200000)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    task=TinyTask(tmp_path,'bank',np.ones((7,33,35),np.uint8))
    amount=task.union_mm.nbytes;key=admit(leases,task,amount)
    assert staged.claim_ram_first(key,amount,0,amount)
    staged.defer(task,amount)
    cpu=next(iter(staged.checkpoint_futures));cpu.set_running_or_notify_cancel()
    staged.abort()
    assert task.union_mm is not None and staged.owns_ram_first_parent(key)
    with pytest.raises(RuntimeError,match='must settle'):staged.close()
    cpu.set_result(False)
    staged.close()
    assert task.union_mm is None and not leases.leases and staged.ram_first_key is None
    assert staged.snapshot()['ram_first_releases']==1
    staged.finalize_cleanup()


def test_ram_token_held_until_complete_codec_future_and_retirement_pump(tmp_path,monkeypatch):
    staged,check,_prepare,leases=queue(tmp_path,cap=200000)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    task=TinyTask(tmp_path,'bank',np.ones((7,33,35),np.uint8))
    amount=task.union_mm.nbytes;key=admit(leases,task,amount)
    staged._dense_limit=amount
    assert staged.claim_ram_first(key,amount,0,amount)
    staged.defer(task,amount)
    assert not staged.claim_ram_first(('model','next'),amount,amount,amount)
    check.finish()
    assert task.union_mm is None and staged.owns_ram_first_parent(key)
    staged.pump()
    assert staged.ram_first_key is None and not leases.leases
    assert staged.claim_ram_first(('model','next'),amount,0,amount)
    staged.release_ram_first(('model','next'))
    staged.close();staged.finalize_cleanup()


def test_verified_raw_disk_retirement_bypasses_blocked_codec_and_refills_detector(tmp_path,monkeypatch):
    from tests.test_tta_scheduler_boundary import _state,_scheduler,_view
    state=_state()
    scheduler=_scheduler(tmp_path,state=state,input_overrides=dict(
        direct_union_inference_view_limit=4,direct_union_inference_byte_limit=120,
        direct_union_total_dense_byte_limit=120))
    leases=ViewPrepareLeaseState(state.direct_union_backing_leases,state.direct_union_inference_views,
        state.direct_union_inference_bytes,state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    staged,codec,fast,_=queue(tmp_path,cap=120,leases=leases)
    ram=TinyTask(tmp_path,'bank',np.ones((1,5,5),np.uint8),np.ones((1,5,5),np.uint8))
    key=admit(leases,ram,50)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    assert staged.claim_ram_first(key,50,0,50)
    staged.defer(ram,50)
    arrays=[];paths=[]
    for name in ('mask','confidence'):
        path=staged.source_root/(name+'.dat')
        array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(1,5,5));array[:]=255
        arrays.append(array);paths.append(path)
    disk=TinyTask(staged.source_root,'disk',*arrays)
    disk.union_path,disk.confmap_path=paths
    disk_key=admit(leases,disk,50);staged.defer(disk,50)
    with mock.patch('XTA.confidence_evidence.confidence_evidence_enabled',return_value=True):
        candidate=dict(kind='fullframe',model_name='model',view=_view(),
            result_mode='direct_union',processing_shape=(1,5,5))
        assert not scheduler.direct_union_task_admissible(candidate)
        assert len(codec.entries)==1 and len(fast.entries)==1
        fast.finish();staged.pump()
        assert disk_key not in leases.leases and key in leases.leases
        assert staged.owns_ram_first_parent(key) and not next(iter(codec.entries)).done()
        assert scheduler.direct_union_task_admissible(candidate)
    saved=staged.deferred[disk_key][1]
    assert saved.mask.reused and saved.confidence.reused and saved.written_bytes==0
    for snapshot in (saved.mask,saved.confidence):
        restored=snapshot.open()
        try:np.testing.assert_array_equal(restored,np.full(restored.shape,255,np.uint8))
        finally:restored._mmap.close()
    codec.finish();staged.pump()
    staged.close();staged.finalize_cleanup()


def test_two_ram_parents_drain_while_full_spool_routes_detector_to_disk(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from tests.test_tta_scheduler_boundary import _state,_scheduler,_view
    state=_state();shape=(7,33,35);amount=2*int(np.prod(shape));cap=3*amount
    leases=ViewPrepareLeaseState(state.direct_union_backing_leases,state.direct_union_inference_views,
        state.direct_union_inference_bytes,state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+cap)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
    gate=threading.Event();both_started=threading.Event();started=[];lock=threading.Lock()
    checkpoint=staging.checkpoint_parent
    def held_checkpoint(task,*args):
        if task.view.name in ('first','second'):
            with lock:
                started.append(task.view.name)
                if len(started)==2:both_started.set()
            assert gate.wait(10)
        return checkpoint(task,*args)
    monkeypatch.setattr(staging,'checkpoint_parent',held_checkpoint)
    expected={};disk_task=None
    with ThreadPoolExecutor(max_workers=2) as codec,ThreadPoolExecutor(max_workers=1) as fast:
        staged=staging.DeferredSamParentQueue(temp_dir=tmp_path/'temp',output_dir=tmp_path/'output',
            checkpoint_executor=codec,prepare_executor=fast,leases=leases,dense_limit=cap,ready=lambda:False)
        def ensure(model,view):
            key=(model,view.name);arrays=[]
            for label in ('mask','confidence'):
                path=staged.source_root/(label+'.dat')
                value=np.memmap(path,mode='w+',dtype=np.uint8,shape=shape);value[:]=0
                arrays.append(value)
            nonlocal disk_task
            disk_task=TinyTask(staged.source_root,view.name,*arrays)
            disk_task.union_path=Path(arrays[0].filename);disk_task.confmap_path=Path(arrays[1].filename)
            state.baseline_union_paths[key]=disk_task.union_path
            state.baseline_confmap_paths[key]=disk_task.confmap_path
            state.direct_union_backing_leases[key]=_DirectUnionBackingLease(key,amount)
            state.direct_union_inference_views.add(key);state.direct_union_inference_bytes[key]=amount
        scheduler=_scheduler(tmp_path,state=state,input_overrides=dict(ensure_baseline_workspaces=ensure,
            min_conf=.5,direct_union_inference_byte_limit=amount,direct_union_total_dense_byte_limit=cap),
            operation_overrides=dict(staged_ram_backlog_bytes=staged.detector_backlog_bytes))
        try:
            for name in ('first','second'):
                mask=np.zeros(shape,np.uint8);mask[:,2:-2,3:-3]=255
                scores=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape).copy()
                expected[name]=(mask.copy(),scores.copy())
                task=TinyTask(staged.source_root,name,mask,scores);key=('model',name)
                active=sum(leases.postprocess_bytes.values())
                assert staged.claim_ram_first(key,amount,active,amount)
                admit(leases,task,amount);staged.defer(task,amount)
                del mask,scores,task
            assert both_started.wait(5)
            assert staged.ram_first_bytes==2*amount and staged.detector_backlog_bytes()==2*amount
            next_key=('model','transverse__tta_a0')
            assert not staged.claim_ram_first(next_key,amount,2*amount,amount)
            candidate=dict(kind='fullframe',model_name='model',view=_view(),
                result_mode='direct_union',processing_shape=shape)
            assert scheduler.direct_union_task_admissible(candidate)
            scheduler.activate_direct_union_task(candidate)
            disk_task.union_mm[:]=expected['first'][0];disk_task.confmap_mm[:]=expected['first'][1]
            assert leases.handoff(next_key);staged.defer(disk_task,amount)
            assert staged.detector_backlog_bytes()==2*amount  # Disk owners remain in the active window.
            disk_future=next(future for future,saved in staged.checkpoint_futures.items() if saved[0]==next_key)
            disk_future.result(timeout=5);staged.pump()
            assert next_key not in leases.leases and staged.ram_first_bytes==2*amount
            assert scheduler.direct_union_task_admissible(candidate)
            gate.set()
            for future in list(staged.checkpoint_futures):future.result(timeout=10)
            staged.pump()
            assert not leases.leases and not staged.ram_first_owners
            assert staged.detector_backlog_bytes()==0
            assert staged.claim_ram_first(('model','next'),amount,0,amount)
            staged.release_ram_first(('model','next'))
            for name in ('first','second',next_key[1]):
                saved=staged.deferred[('model',name)][1]
                wanted=expected['first' if name==next_key[1] else name]
                assert saved.mask.encoding==('raw' if name==next_key[1] else 'packed_mask')
                for reference,values in zip((saved.mask,saved.confidence),wanted):
                    restored=reference.open()
                    try:np.testing.assert_array_equal(restored,values)
                    finally:restored._mmap.close()
        finally:
            gate.set();codec.shutdown(wait=True);fast.shutdown(wait=True)
            staged.close();staged.finalize_cleanup()


def test_ram_backlog_rejects_replaced_original_lease(tmp_path,monkeypatch):
    staged,check,_prepare,leases=queue(tmp_path,cap=200000)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    task=TinyTask(tmp_path,'v',owning_mask());amount=task.union_mm.nbytes;key=admit(leases,task,amount)
    assert staged.claim_ram_first(key,amount,0,amount)
    staged.defer(task,amount)
    assert staged.detector_backlog_bytes()==amount
    original=leases.leases[key]
    leases.leases[key]=_DirectUnionBackingLease(key,amount,phase='postprocess')
    with pytest.raises(RuntimeError,match='original postprocess lease'):staged.detector_backlog_bytes()
    leases.leases[key]=original
    check.finish();staged.pump();staged.close();staged.finalize_cleanup()


def test_unowned_ram_checkpoint_rejects_before_executor_acceptance(tmp_path,monkeypatch):
    staged,check,_prepare,_leases=queue(tmp_path,cap=200000)
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args:True)
    task=TinyTask(tmp_path,'v',owning_mask());amount=task.union_mm.nbytes;key=('model','v')
    assert staged.claim_ram_first(key,amount,0,amount)
    with pytest.raises(RuntimeError,match='original postprocess lease'):staged.defer(task,amount)
    assert not check.entries and task.union_mm is not None
    staged.release_ram_first(key);staged.close();staged.finalize_cleanup()

