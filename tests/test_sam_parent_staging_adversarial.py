"""Independent bounded parent staging, physical ownership and continuation tests."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
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

