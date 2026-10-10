"""Startup takes priority over new dense promises without parked base credit."""
from concurrent.futures import CancelledError
from types import SimpleNamespace
import threading

import pytest

from XTA import runtime, sam_parent_staging as staging
from XTA.interpolation import _ByteAdmissionPool, _DirectUnionBackingLease
from tests.test_memfd_resident_admission import controlled_owner, linux_probe
from tests.test_sam_parent_staging_adversarial import queue, TinyTask
from tests.test_view_prepare_input_retirement import _task


@pytest.mark.parametrize('backing,proofs,expected', [
    (False,(),100), (None,(),0), (True,(('mask','old',1,2,100),),100),
    (None,(('mask','old',1,2,20),('confidence','other',1,3,30)),50),
    (True,(('mask','stale',1,2,1000),),100)])
def test_prospective_reset_charges_disk_and_all_possible_old_credit(backing,proofs,expected):
    lease = _DirectUnionBackingLease(('model','view'),100,ram_backed=backing,memfd_owner_proofs=proofs)
    assert staging.dense_proof_reset_bytes(lease) == expected


def test_malformed_old_proof_fails_before_dropping_any_debt():
    proof = ('mask','broken')
    lease = _DirectUnionBackingLease(('model','view'),100,ram_backed=True,memfd_owner_proofs=(proof,))
    with pytest.raises(RuntimeError,match='residency proof is malformed'):
        staging.dense_proof_reset_bytes(lease)
    assert lease.ram_backed is True and lease.memfd_owner_proofs == (proof,)


def test_residency_growth_between_hint_and_callback_cannot_undercharge_reset(tmp_path,monkeypatch):
    stage,_writer,_prepare,leases = queue(tmp_path,cap=10**6)
    with controlled_owner(tmp_path) as (array,_key,descriptor,info):
        proofs = runtime.capture_memfd_owner_proofs({'mask':array})
        lease = _DirectUnionBackingLease(('model','view'),1024,memfd_owner_proofs=proofs)
        leases.leases[lease.key] = lease
        monkeypatch.setattr(staging,'publication_ram_headroom',lambda:10000)
        linux_probe(monkeypatch,descriptor,info,blocks=0)
        assert runtime.memfd_ram_headroom((lease,),lambda:10000)[1] == 0
        additional = staging.dense_proof_reset_bytes(lease)
        # Decode faults all pages before startup's next resident sample.
        linux_probe(monkeypatch,descriptor,info,blocks=2)
        assert stage.checkpoint_copy_promises() == 0
        assert additional == 1024
        budget = dict(physical_headroom_bytes=1500,required_host_bytes=1000+additional,
            remaining_startup_bytes=1)
        assert not staging._dense_startup_fits(budget)
        assert lease.memfd_owner_proofs == proofs
        leases.leases.clear()
    stage.close()


def test_prospective_dense_birth_stays_deferred_without_any_lease_or_base_credit(tmp_path):
    stage,_writer,prepare,leases = queue(tmp_path,cap=120,ready=lambda:True)
    task = TinyTask(tmp_path,'deferred',None)
    task.admission = _ByteAdmissionPool(2048,'parent')
    physical = [100]
    additions = []
    def budget(*,additional_pending_bytes=0):
        assert task.admission.condition._is_owned()
        additions.append(additional_pending_bytes)
        return dict(physical_headroom_bytes=physical[0],required_host_bytes=90+additional_pending_bytes,
            remaining_startup_bytes=1)
    task.sam_context = SimpleNamespace(progressive_startup=True,_startup_pool=task.admission,
        _startup_fleet_funded=True,_startup_fleet_credit_bytes=1,startup_budget_snapshot_locked=budget)
    key = ('model','deferred')
    stage.deferred[key] = (task,staging.ParentCheckpoint(None,None,60,0,0,0))
    resumed,_released = stage.pump()
    assert not resumed and not prepare.entries and not leases.leases
    assert key in stage.deferred and task.admission.in_use == 0
    assert additions == [60]
    physical[0] = 200
    resumed,_released = stage.pump()
    assert len(resumed) == 1 and leases.leases[key].ram_commitment_bytes == 60
    for future in prepare.entries:
        future.cancel()
    leases.complete(key,retain_for_dense_retirement=False)
    stage.close()


@pytest.mark.parametrize('cancel', [False,True])
def test_prepare_reset_waits_before_base_credit_and_preserves_proofs_until_admitted(
        tmp_path,cancel):
    task,expected = _task(tmp_path)
    task.admission = _ByteAdmissionPool(2048,'parent')
    lease = _DirectUnionBackingLease((task.model_name,task.view.name),expected.nbytes,
        ram_backed=True,memfd_owner_proofs=(('mask','stale-owned-input',1,2,expected.nbytes),))
    task.backing_lease = lease
    waiting,allow = threading.Event(),threading.Event()
    cancelled = threading.Event()
    def budget(*,additional_pending_bytes=0):
        assert task.admission.condition._is_owned()
        assert additional_pending_bytes == expected.nbytes
        waiting.set()
        return dict(physical_headroom_bytes=200 if allow.is_set() else 100,
            required_host_bytes=100+additional_pending_bytes,remaining_startup_bytes=1)
    task.sam_context = SimpleNamespace(progressive_startup=True,_startup_pool=task.admission,
        _startup_fleet_funded=True,_startup_fleet_credit_bytes=1,startup_budget_snapshot_locked=budget,
        _cancel=cancelled,_failure='controlled cancellation')
    result,errors = [],[]
    def run():
        try:
            result.append(task())
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert waiting.wait(2)
        with task.admission.condition:
            assert task.admission.in_use == 0
            assert lease.ram_backed is True and lease.memfd_owner_proofs
            if cancel:
                cancelled.set()
            else:
                allow.set()
            task.admission.condition.notify_all()
        worker.join(3)
        assert not worker.is_alive() and task.admission.in_use == 0
        if cancel:
            assert len(errors) == 1 and isinstance(errors[0],CancelledError)
            assert task._cancelled_before_prepare and task.union_mm is None
            assert lease.ram_backed is True and lease.memfd_owner_proofs
        else:
            assert not errors and len(result) == 1
            assert lease.ram_backed is None and lease.memfd_owner_proofs == ()
    finally:
        cancelled.set()
        with task.admission.condition:
            task.admission.condition.notify_all()
        worker.join(3)
        if task.union_mm is not None and not task.union_mm._mmap.closed:
            task.union_mm._mmap.close()


def test_unfunded_empty_work_forecast_never_fences_prepare_or_restoration(tmp_path):
    task,_expected = _task(tmp_path/'parent')
    task.admission = _ByteAdmissionPool(2048,'parent')
    task.backing_lease = _DirectUnionBackingLease((task.model_name,task.view.name),60,ram_backed=False)
    task.sam_context = SimpleNamespace(progressive_startup=True,_startup_fleet_funded=False,
        _startup_pool=task.admission,_startup_future_peaks={0:10**12},_cancel=threading.Event(),
        startup_budget_snapshot_locked=lambda **_kwargs:pytest.fail('unfunded forecasts must not block empty work'))
    result = task()
    assert result is not None and task.admission.in_use == 0
    assert task.backing_lease.ram_backed is None
    stage,_writer,prepare,leases = queue(tmp_path/'staging',cap=120,ready=lambda:True)
    empty = TinyTask(tmp_path,'empty',None)
    empty.admission,empty.sam_context = task.admission,task.sam_context
    key = ('model','empty')
    stage.deferred[key] = (empty,staging.ParentCheckpoint(None,None,60,0,0,0))
    resumed,_released = stage.pump()
    assert len(resumed) == 1
    for future in prepare.entries:
        future.cancel()
    leases.complete(key,retain_for_dense_retirement=False)
    stage.close()


def test_completed_fleet_returns_before_any_proof_or_physical_probe(tmp_path):
    task,_expected = _task(tmp_path)
    task.admission = _ByteAdmissionPool(2048,'parent')
    task.backing_lease = _DirectUnionBackingLease((task.model_name,task.view.name),60,
        ram_backed=True,memfd_owner_proofs=(('mask','old-short-proof'),))
    task.sam_context = SimpleNamespace(progressive_startup=True,_startup_fleet_funded=True,
        _startup_fleet_credit_bytes=0,_startup_pool=task.admission,_cancel=threading.Event(),
        startup_budget_snapshot_locked=lambda **_kwargs:pytest.fail('settled fleet must not probe'))
    result = task()
    assert result is not None and task.admission.in_use == 0
    assert task.backing_lease.ram_backed is None and task.backing_lease.memfd_owner_proofs == ()
