"""Dense backing births preserve owned progressive model-startup headroom."""
from contextlib import contextmanager
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import sam_parent_staging as staging
from XTA.interpolation import _ByteAdmissionPool, _DirectUnionBackingLease
from tests.test_memfd_resident_admission import controlled_owner, linux_probe
from tests.test_sam_parent_staging_adversarial import queue, TinyTask


def compact_checkpoint(tmp_path):
    mask = np.zeros((3,80,100), np.uint8)
    mask[:,10:20,15:25] = 1
    saved = staging._snapshot_array(mask, None, tmp_path/'mask.dat', tmp_path,
        threading.Event(), mask=True)
    assert saved.encoding == 'packed_mask'
    return saved, mask


def test_copy_promises_exclude_unused_capacity_and_disk_deferred_payloads(tmp_path, monkeypatch):
    stage, _writer, _prepare, leases = queue(tmp_path, cap=10**12)
    stage.deferred[('model','disk')] = (None, staging.ParentCheckpoint(None,None,10**11,0,0,0))
    for name, amount, backing in (('pending',700,None), ('disk',10000,False)):
        key = ('model',name)
        leases.leases[key] = _DirectUnionBackingLease(key,amount,ram_backed=backing)
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: 100000)
    assert stage.checkpoint_copy_promises() == 700
    # No unused 1-TB allowance or unadmitted 100-GB disk checkpoint is owed.
    stage.deferred.clear()
    leases.leases.clear()
    stage.close()


@pytest.mark.parametrize('swap,expected', [(0,512), (1,1024)])
def test_copy_promises_credit_only_authenticated_no_swap_resident_pages(
        tmp_path, monkeypatch, swap, expected):
    stage, _writer, _prepare, leases = queue(tmp_path, cap=10**12)
    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proofs = staging.capture_memfd_owner_proofs({'mask':array})
        linux_probe(monkeypatch, descriptor, info, blocks=1, swap=swap)
        key = ('model','partial')
        leases.leases[key] = _DirectUnionBackingLease(key,1024,memfd_owner_proofs=proofs)
        monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: 100000)
        assert stage.checkpoint_copy_promises() == expected
        leases.leases[key].memfd_owner_proofs = ()
        assert stage.checkpoint_copy_promises() == 1024
        leases.leases.clear()
    stage.close()


def test_progressive_restore_publishes_weak_owner_proof_before_decode(tmp_path,monkeypatch):
    from XTA.interpolation import RawBBoxMaskStore
    expected = np.zeros((1,1,24000),np.uint8)
    expected[0,0,101:230] = 1
    saved = staging._snapshot_array(expected,None,tmp_path/'mask.dat',tmp_path,
        threading.Event(),mask=True)
    stage, _writer, prepare, leases = queue(tmp_path,cap=10**6,ready=lambda:True)
    task = TinyTask(tmp_path,'restored',None)
    task.admission = _ByteAdmissionPool(2*1024**3,'parent')
    task.sam_context = SimpleNamespace(progressive_startup=True,
        _startup_pool=task.admission,startup_budget_snapshot_locked=lambda **_kwargs:
            dict(physical_headroom_bytes=100000,required_host_bytes=1000))
    key = ('model','restored')
    stage.deferred[key] = (task,staging.ParentCheckpoint(saved,None,saved.nbytes,0,0,0))
    monkeypatch.setattr(staging,'_ram_fits',lambda *_args,**_kwargs:True)
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:100000)
    with controlled_owner(tmp_path,size=saved.nbytes) as (array,_key,descriptor,info):
        linux_probe(monkeypatch,descriptor,info,blocks=20)
        def allocate(*_args,**kwargs):
            assert task.admission.condition._is_owned() and kwargs['prefer_memfd']
            return array
        monkeypatch.setattr(staging,'allocate_workspace_array',allocate)
        original_decode = RawBBoxMaskStore.decode_slice_crop
        def decode(self,*args,**kwargs):
            assert not task.admission.condition._is_owned()
            assert leases.leases[key].memfd_owner_proofs
            assert stage.checkpoint_copy_promises() == saved.nbytes-20*512
            return original_decode(self,*args,**kwargs)
        monkeypatch.setattr(RawBBoxMaskStore,'decode_slice_crop',decode)
        resumed,_released = stage.pump()
        prepare.finish()
        result = next(iter(resumed)).result()
        np.testing.assert_array_equal(result.final_view_volume_mm,expected)
        leases.complete(key,retain_for_dense_retirement=False)
        stage.close()


@pytest.mark.parametrize('required,prefer_memfd', [(800,True), (1200,False)])
def test_backing_birth_is_atomic_with_startup_but_decode_does_not_hold_condition(
        tmp_path, monkeypatch, required, prefer_memfd):
    from XTA.interpolation import RawBBoxMaskStore
    saved, expected = compact_checkpoint(tmp_path)
    pool = _ByteAdmissionPool(2048, 'controlled parent credit')
    entered_allocation, startup_entered = threading.Event(), threading.Event()
    attempted = threading.Event()
    def budget(**_kwargs):
        assert pool.condition._is_owned()
        return dict(physical_headroom_bytes=1000,required_host_bytes=required)
    task = SimpleNamespace(admission=pool, sam_context=SimpleNamespace(
        progressive_startup=True,startup_budget_snapshot_locked=budget))
    chosen = []
    def allocate(shape,dtype,path,_desc,**kwargs):
        assert pool.condition._is_owned()
        entered_allocation.set()
        assert attempted.wait(2)
        assert not startup_entered.wait(.05)
        chosen.append(kwargs['prefer_memfd'])
        return np.memmap(path, mode='w+', dtype=dtype, shape=shape)
    monkeypatch.setattr(staging, 'allocate_workspace_array', allocate)
    original_decode = RawBBoxMaskStore.decode_slice_crop
    def decode(self,*args,**kwargs):
        assert not pool.condition._is_owned()
        assert startup_entered.wait(2)
        return original_decode(self,*args,**kwargs)
    monkeypatch.setattr(RawBBoxMaskStore, 'decode_slice_crop', decode)
    def startup():
        assert entered_allocation.wait(2)
        attempted.set()
        with pool.condition:
            startup_entered.set()
    worker = threading.Thread(target=startup)
    worker.start()
    metrics = {}
    output = None
    try:
        output = saved.open(prefer_ram=True,metrics=metrics,
            allocation_guard=lambda: staging._restore_allocation_guard(task,metrics))
        np.testing.assert_array_equal(output,expected)
        assert chosen == [prefer_memfd]
        assert metrics['startup_ram_admitted'] is prefer_memfd
    finally:
        worker.join(3)
        assert not worker.is_alive()
        if output is not None:
            staging.close_memmap_array_without_flush(output)
            if not output._mmap.closed:
                output._mmap.close()


@pytest.mark.parametrize('progressive,with_snapshot,expected', [
    (False,True,True), (True,False,False)])
def test_legacy_allocation_and_missing_progressive_permission(
        tmp_path,monkeypatch,progressive,with_snapshot,expected):
    saved, _mask = compact_checkpoint(tmp_path)
    context = SimpleNamespace(progressive_startup=progressive)
    if with_snapshot:
        context.startup_budget_snapshot_locked = lambda: pytest.fail('legacy must not query startup debt')
    task = SimpleNamespace(admission=_ByteAdmissionPool(2048,'parent'),sam_context=context)
    chosen = []
    def allocate(shape,dtype,path,_desc,**kwargs):
        chosen.append(kwargs['prefer_memfd'])
        return np.memmap(path,mode='w+',dtype=dtype,shape=shape)
    monkeypatch.setattr(staging,'allocate_workspace_array',allocate)
    metrics = {}
    output = saved.open(prefer_ram=True,metrics=metrics,
        allocation_guard=lambda: staging._restore_allocation_guard(task,metrics))
    try:
        assert chosen == [expected]
    finally:
        output._mmap.close()


def test_callback_failure_retires_allocated_backing_and_releases_guard(tmp_path,monkeypatch):
    saved, _mask = compact_checkpoint(tmp_path)
    outputs = []
    def allocate(shape,dtype,path,_desc,**_kwargs):
        output = np.memmap(path,mode='w+',dtype=dtype,shape=shape)
        outputs.append(output)
        return output
    monkeypatch.setattr(staging,'allocate_workspace_array',allocate)
    pool = _ByteAdmissionPool(2048,'parent')
    @contextmanager
    def guard():
        with pool.condition:
            yield True
    def fail(_array):
        raise RuntimeError('controlled owner proof failure')
    with pytest.raises(RuntimeError,match='controlled owner proof failure'):
        saved.open(prefer_ram=True,allocation_guard=guard,allocation_callback=fail)
    assert outputs[0]._mmap.closed and not pool.condition._is_owned()


@pytest.mark.parametrize('headroom_gib,admitted', [(95,False), (96,True)])
def test_real_context_snapshot_accounts_fleet_and_actual_dense_debt_once(
        tmp_path,monkeypatch,headroom_gib,admitted):
    from XTA import sam_resources
    from XTA.sam_integration import SamInterpolationContext
    gib = sam_resources.GIB
    stage, _writer, _prepare, leases = queue(tmp_path,cap=10**12)
    key = ('model','pending')
    leases.leases[key] = _DirectUnionBackingLease(key,2*gib)
    pool = _ByteAdmissionPool(64*gib,'parent')
    context = object.__new__(SamInterpolationContext)
    context.progressive_startup = True
    context._startup_pool = pool
    context._startup_host_reserved = 0
    context._startup_fleet_funded = False
    context._startup_fleet_credit_bytes = 0
    context._startup_parent_envelope_bytes = 0
    context._startup_funding_lock = threading.Lock()
    context._startup_wait_timeout = 1.
    context._publish_startup_progress = lambda *_args,**_kwargs: None
    context._startup_future_peaks = {0:6*gib,1:6*gib}
    context._retired_devices = {0}
    context._startup_host_grants = {}
    context._startup_pending_host_bytes = stage.checkpoint_copy_promises
    context.source_volume = np.zeros((1,1,1),np.uint8)
    context._progressive_error = None
    context._cancel = threading.Event()
    context._closed = False
    context._failure = ''
    ordered = []
    monkeypatch.setattr(staging,'publication_ram_headroom',
        lambda: ordered.append('resident-proof') or headroom_gib*gib)
    monkeypatch.setattr(sam_resources,'physical_sam_headroom',
        lambda: ordered.append('physical') or headroom_gib*gib)
    task = SimpleNamespace(admission=pool,sam_context=context)
    receipt = {}
    with staging._restore_allocation_guard(task,receipt) as allowed:
        assert allowed is admitted
    assert ordered == ['resident-proof','physical']
    assert receipt['startup_required_host_bytes'] == 96*gib
    # Active peak is already one of the fleet promises, not added twice.
    if admitted:
        context._fund_progressive_fleet()
        context._reserve_progressive_host(0,6*gib,{})
        assert pool.in_use == 12*gib and context._startup_host_reserved == 6*gib
        with staging._restore_allocation_guard(task,{}) as allowed:
            assert allowed
        context._startup_future_peaks.pop(0)
        with pool.condition:
            context._sync_startup_fleet_credit_locked()
        context._release_progressive_host(0)
        assert pool.in_use == 6*gib
        with staging._restore_allocation_guard(task,receipt) as allowed:
            assert allowed
        assert receipt['startup_required_host_bytes'] == 90*gib
    leases.leases.clear()
    stage.close()


def test_cancelled_context_cannot_allocate_even_when_budget_fits(tmp_path):
    pool = _ByteAdmissionPool(2048,'parent')
    def cancelled():
        raise RuntimeError('controlled cancelled startup')
    task = SimpleNamespace(admission=pool,sam_context=SimpleNamespace(
        progressive_startup=True,_startup_pool=pool,check_startup=cancelled,
        startup_budget_snapshot_locked=lambda: pytest.fail('cancelled startup must not probe')))
    with pytest.raises(RuntimeError,match='controlled cancelled startup'):
        with staging._restore_allocation_guard(task,{}):
            pytest.fail('cancelled startup must not allocate')
    assert not pool.condition._is_owned()


def test_new_restore_lease_cannot_race_a_startup_budget_snapshot(tmp_path):
    stage, _writer, prepare, leases = queue(tmp_path,cap=120,ready=lambda:True)
    task = TinyTask(tmp_path,'deferred',None)
    task.admission = _ByteAdmissionPool(2048,'parent')
    key = ('model','deferred')
    stage.deferred[key] = (task,staging.ParentCheckpoint(None,None,60,0,0,0))
    started, finished = threading.Event(),threading.Event()
    result = []
    def pump():
        started.set()
        result.append(stage.pump())
        finished.set()
    with task.admission.condition:
        worker = threading.Thread(target=pump)
        worker.start()
        assert started.wait(2)
        assert not finished.wait(.05)
        assert stage.checkpoint_copy_promises() == 0
    worker.join(3)
    assert not worker.is_alive()
    assert stage.checkpoint_copy_promises() == 60
    assert len(result[0][0]) == len(prepare.entries) == 1
    for future in prepare.entries:
        future.cancel()
    leases.complete(key,retain_for_dense_retirement=False)
    stage.close()
