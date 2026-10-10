"""Resident credit never removes unfaulted RAM, ownership or logical limits."""
from contextlib import contextmanager
import gc
import os
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import runtime, sam_parent_staging as staging
from XTA.interpolation import _DirectUnionBackingLease
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_sam_parent_staging_adversarial import queue


@contextmanager
def controlled_owner(tmp_path, *, size=1024):
    """Use real descriptor/map lifetime with controlled Linux occupancy below."""
    path=tmp_path/'owner.dat'
    array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(1,1,size))
    descriptor=os.open(path,os.O_RDWR)
    key=f'/proc/{os.getpid()}/fd/{descriptor}'
    array._workspace_memfd_owner_key=key
    array._workspace_memfd_owner_fd=descriptor
    runtime._register_memfd_owner(key,descriptor,'controlled raw-workspace authority')
    info=os.fstat(descriptor)
    try:
        yield array,key,descriptor,info
    finally:
        array._mmap.close()
        runtime._release_memfd_owner_key(key)


def linux_probe(monkeypatch, descriptor, info, *, blocks=1, swap=0):
    monkeypatch.setattr(runtime.sys,'platform','linux')
    monkeypatch.setattr(runtime,'_read_meminfo_bytes',lambda:{'SwapTotal':swap} if swap is not None else {})
    original=os.fstat
    def fstat(fd):
        if fd!=descriptor:
            return original(fd)
        return SimpleNamespace(st_dev=info.st_dev,st_ino=info.st_ino,st_size=info.st_size,st_blocks=blocks)
    monkeypatch.setattr(runtime.os,'fstat',fstat)


def promises(*rows):
    return tuple(_DirectUnionBackingLease(('test',str(index)),amount,ram_backed=True,
        memfd_owner_proofs=proofs) for index,(amount,proofs) in enumerate(rows))


def test_capture_is_weak_and_rejects_derived_or_copy_on_write_maps(tmp_path):
    with controlled_owner(tmp_path) as (array,_key,_descriptor,_info):
        reference=weakref.ref(array)
        proofs=runtime.capture_memfd_owner_proofs({'mask':array,'confidence':None})
        assert len(proofs)==1 and proofs[0][0]=='mask'
        assert runtime.capture_memfd_owner_proofs({'mask':array[:]})==()
        array.mode='c'
        assert runtime.capture_memfd_owner_proofs({'mask':array})==()
        array.mode='r+'
    array=None
    gc.collect()
    assert reference() is None and proofs  # Facts do not pin the original array.


def test_future_output_entry_clears_original_resident_basis_before_new_canvas(tmp_path):
    from tests.test_view_prepare_input_retirement import _task
    task,_expected=_task(tmp_path)
    lease=_DirectUnionBackingLease((task.model_name,task.view.name),task.union_mm.nbytes,
        ram_backed=True,memfd_owner_proofs=(('mask','old-source',1,2,task.union_mm.nbytes),))
    task.backing_lease=lease
    original=task.prepare
    def prepare(**kwargs):
        assert lease.ram_backed is None and lease.memfd_owner_proofs==()
        return original(**kwargs)
    task.prepare=prepare
    result=task()
    assert lease.memfd_owner_proofs==()
    task=result=None
    gc.collect()


@pytest.mark.parametrize('swap', [None,1,4096])
def test_unproven_or_positive_swap_preserves_full_guard(tmp_path,monkeypatch,swap):
    with controlled_owner(tmp_path) as (array,_key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'mask':array})
        linux_probe(monkeypatch,descriptor,info,blocks=2,swap=swap)
        headroom,resident,record=runtime.memfd_ram_headroom(promises((1024,proofs)),lambda:500)
        assert headroom==500 and resident==0
        assert record['memfd_resident_probe_status']==('swap_unproven' if swap is None else 'swap_enabled')


@pytest.mark.parametrize('invalid', ['unregistered','device','inode','size','error'])
def test_stale_recycled_or_unprobeable_owner_never_receives_credit(tmp_path,monkeypatch,invalid):
    with controlled_owner(tmp_path) as (array,key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'mask':array})
        linux_probe(monkeypatch,descriptor,info,blocks=2)
        if invalid=='unregistered':
            runtime._MEMFD_OWNERS.pop(key)
        elif invalid=='error':
            monkeypatch.setattr(runtime.os,'fstat',mock.Mock(side_effect=OSError('descriptor gone')))
        else:
            fields=dict(st_dev=info.st_dev,st_ino=info.st_ino,st_size=info.st_size,st_blocks=2)
            fields[{'device':'st_dev','inode':'st_ino','size':'st_size'}[invalid]]+=1
            monkeypatch.setattr(runtime.os,'fstat',lambda _fd:SimpleNamespace(**fields))
        assert runtime.memfd_ram_headroom(promises((1024,proofs)),lambda:500)[1]==0
        if invalid=='unregistered':
            runtime._MEMFD_OWNERS[key]=(descriptor,'controlled raw-workspace authority')


def test_dedup_clamp_partial_retirement_and_probe_order(tmp_path,monkeypatch):
    with controlled_owner(tmp_path) as (array,key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'confidence':array})
        linux_probe(monkeypatch,descriptor,info,blocks=100)
        sampled=mock.Mock(wraps=runtime.os.fstat)
        monkeypatch.setattr(runtime.os,'fstat',sampled)
        def physical():
            assert sampled.called  # Existing occupied pages are sampled first.
            assert runtime._MEMFD_OWNERS[key][0]==descriptor
            return 500
        headroom,resident,record=runtime.memfd_ram_headroom(promises((600,proofs),(1024,proofs)),physical)
        assert headroom==500 and resident==600  # Clip to promise and count the shared inode once.
        assert record['eligible_memfd_owner_count']==1
        lease=_DirectUnionBackingLease(('model','view'),2048,phase='postprocess',memfd_owner_proofs=proofs)
        state=ViewPrepareLeaseState({lease.key:lease},set(),{}, {lease.key},{lease.key:2048})
        assert state.retire_input_bytes(lease.key,lease,1024,token='confidence')
        assert lease.nbytes==1024 and lease.memfd_owner_proofs==()
        assert runtime.memfd_ram_headroom(state.ram_residency_promises(),lambda:500)[1]==0


def test_swap_proof_loss_and_headroom_error_are_conservative(tmp_path,monkeypatch):
    with controlled_owner(tmp_path) as (array,_key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'mask':array})
        linux_probe(monkeypatch,descriptor,info,blocks=2)
        monkeypatch.setattr(runtime,'_read_meminfo_bytes',mock.Mock(side_effect=[{'SwapTotal':0},{'SwapTotal':1}]))
        assert runtime.memfd_ram_headroom(promises((1024,proofs)),lambda:500)[1]==0
        monkeypatch.setattr(runtime,'_read_meminfo_bytes',lambda:{'SwapTotal':0})
        def unreadable():
            raise OSError('headroom unavailable')
        assert runtime.memfd_ram_headroom(promises((1024,proofs)),unreadable)[:2]==(0,0)


@pytest.mark.parametrize('change', ['retire','future_output','partial_credit','release','shrink','grow'])
def test_reentrant_headroom_changes_cannot_reuse_expired_or_newly_grown_credit(tmp_path,monkeypatch,change):
    with controlled_owner(tmp_path) as (array,key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'confidence':array})
        linux_probe(monkeypatch,descriptor,info,blocks=1 if change=='grow' else 2)
        lease=promises((1024,proofs))[0]
        state=ViewPrepareLeaseState({lease.key:lease},{lease.key},{lease.key:1024},set(),{})
        def physical():
            if change=='retire':
                array._mmap.close();runtime._release_memfd_owner_key(key)
            elif change=='future_output':
                lease.memfd_owner_proofs=();lease.ram_backed=None
            elif change=='partial_credit':
                state.handoff(lease.key)
                state.retire_input_bytes(lease.key,lease,512,token='confidence')
            elif change=='release':
                state.handoff(lease.key);state.complete(lease.key,retain_for_dense_retirement=False)
            else:
                fields=SimpleNamespace(st_dev=info.st_dev,st_ino=info.st_ino,st_size=info.st_size,
                    st_blocks=2 if change=='grow' else 0)
                monkeypatch.setattr(runtime.os,'fstat',lambda _fd:fields)
            return 2000
        headroom,resident,record=runtime.memfd_ram_headroom(state.ram_residency_promises(),physical)
        assert headroom==2000 and resident==(512 if change=='grow' else 0)
        if change!='grow':
            assert record['memfd_resident_probe_status']=='resident_basis_changed' or change=='shrink'


def test_staging_credits_only_resident_part_while_full_caps_and_future_promise_remain(tmp_path,monkeypatch):
    stage,_writer,_prepare,leases=queue(tmp_path,cap=1200)
    with controlled_owner(tmp_path) as (array,_key,descriptor,info):
        proofs=runtime.capture_memfd_owner_proofs({'mask':array})
        linux_probe(monkeypatch,descriptor,info,blocks=1)
        key=('model','existing')
        lease=_DirectUnionBackingLease(key,700,ram_backed=True,memfd_owner_proofs=proofs)
        leases.leases[key]=lease;leases.inference_views.add(key);leases.inference_bytes[key]=700
        monkeypatch.setattr(staging,'RAM_RESERVE_BYTES',16)
        monkeypatch.setattr(staging,'publication_ram_headroom',lambda:500)
        monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
        assert not staging._ram_fits(70,700,1200)  # Old full-promise comparison.
        assert stage.claim_ram_first(('model','new'),50,leases.ram_commitment_bytes,20)
        receipt={}
        assert staging._ram_fits(70,700,1200,receipt,leases.ram_residency_promises())
        assert receipt['proven_resident_ram_bytes']==512
        assert receipt['future_ram_commitments_bytes']==188
        assert receipt['total_checked_bytes']==770 and receipt['physical_required_bytes']==274
        assert lease.nbytes==700 and leases.inference_bytes[key]==700
        assert not staging._ram_fits(70,700,769,None,leases.ram_residency_promises())
        monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:769)
        assert not staging._ram_fits(70,700,1200,None,leases.ram_residency_promises())
        monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
        monkeypatch.setattr(staging,'publication_ram_headroom',lambda:273)
        assert not stage.claim_ram_first(('model','future-overcommit'),50,leases.ram_commitment_bytes,20)
        assert stage.last_ram_denial['reason']=='physical_headroom'
        stage.release_ram_first(('model','new'))
    leases.handoff(key);leases.complete(key,retain_for_dense_retirement=False)
    stage.close();stage.finalize_cleanup()


@pytest.mark.skipif(not hasattr(os,'memfd_create'),reason='Real memfd residency requires Linux')
def test_real_sparse_memfd_stays_lazy_and_occupied_pages_are_credited_without_prefault():
    if runtime._read_meminfo_bytes().get('SwapTotal')!=0:
        pytest.skip('Real resident block proof requires node swap disabled')
    array=runtime._allocate_memfd_workspace_array((8,1,4096),np.uint8,'resident functional proof',initialize_zero=True)
    try:
        proofs=runtime.capture_memfd_owner_proofs({'mask':array})
        descriptor=array._workspace_memfd_owner_fd
        assert os.fstat(descriptor).st_blocks==0
        assert runtime.memfd_ram_headroom(promises((array.nbytes,proofs)),lambda:10**8)[1]==0
        array[0]=173
        headroom,resident,_record=runtime.memfd_ram_headroom(promises((array.nbytes,proofs)),lambda:10**8)
        assert 0<resident<=array.nbytes and headroom==10**8
        assert min(array.nbytes,os.fstat(descriptor).st_blocks*512)==resident
    finally:
        key=array._workspace_memfd_owner_key
        array._mmap.close();runtime._release_memfd_owner_key(key)
