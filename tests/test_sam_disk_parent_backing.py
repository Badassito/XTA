"""Blocked shared-SAM parents reuse exact ordinary-disk owners from birth."""
import gc
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pipeline, sam_parent_staging as staging
from XTA.geometry import ViewInfo
from XTA.runtime import wait_for_retired_memmap_unlinks
from XTA.interpolation import _DirectUnionBackingLease
from XTA.view_prepare import ViewPrepareLeaseState, classify_dense_ram_backing
from tests.test_sam_parent_staging_adversarial import TinyTask, queue
from tests.test_terminal_component_refs import _function
from tests.test_tta_scheduler_boundary import _state


@pytest.mark.parametrize('metric', ['ram_commitment_bytes', 'disk_backed_logical_bytes'])
def test_ram_totals_survive_scheduler_changes_during_restore_read(metric):
    entered, resume = threading.Event(), threading.Event()
    ram = metric == 'ram_commitment_bytes'

    class PausedLease:
        nbytes = 10

        @property
        def ram_commitment_bytes(self):
            entered.set()
            assert resume.wait(3)
            return self.nbytes

        @property
        def ram_backed(self):
            entered.set()
            assert resume.wait(3)
            return ram

    leases = ViewPrepareLeaseState({'first':PausedLease()}, set(), {}, set(), {})
    with ThreadPoolExecutor(max_workers=1) as executor:
        reading = executor.submit(getattr, leases, metric)
        try:
            assert entered.wait(2)
            leases.leases['next'] = _DirectUnionBackingLease(('model','next'), 20, ram_backed=ram)
        finally:
            resume.set()
        assert reading.result(timeout=3) == 10
    assert getattr(leases, metric) == 30


def allocation(tmp_path, *, backend='sam', extrapolation=False, ready=False, shared=True,
               stager=True, interpolation=True, tmpfs=False, monkeypatch=None, ram_fit=False, cap=120):
    staged,check,prepare,leases=queue(tmp_path,cap=cap,ready=lambda:ready)
    if tmpfs:
        monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:
            Path(path).resolve().is_relative_to(staged.source_root.resolve()))
    state=_state();shape=(3,4,5)
    view=ViewInfo(name='test__tta_a0',family='tilted',num_slices=3,src_h=4,src_w=5,pad_mode='clamp')
    ns=dict(vars(pipeline));telemetry=mock.Mock()
    ns.update(temp_dir=staged.source_root,worker_direct_union_active=True,
        sam_cpu_max_detector_parent=60,
        policy_settings=SimpleNamespace(enabled=False),bounded_policy_parent_keys=set(),
        interpolation_settings=SimpleNamespace(backend=backend,extrapolation_enabled=extrapolation),
        sam_context=SimpleNamespace(detector_retirement_ready=ready,
            shared_detector_devices=('cuda:0',)if shared else()),sam_parent_staging=staged if stager else None,
        args=SimpleNamespace(imgsz=8,min_conf=0.,interpolation_distance=5 if interpolation else 0,
            reconciliation_retain_confidence=True),dense_tiling_active=False,nrrd_layers_needed=False,
        component_layers_needed=False,baseline_union_by_model_view={},baseline_confmap_by_model_view={},
        baseline_union_paths=state.baseline_union_paths,baseline_confmap_paths=state.baseline_confmap_paths,
        baseline_slice_locks_by_model_view={},direct_union_backing_leases=state.direct_union_backing_leases,
        direct_union_inference_views=state.direct_union_inference_views,
        direct_union_postprocess_views=state.direct_union_postprocess_views,
        direct_union_inference_bytes=state.direct_union_inference_bytes,
        direct_union_postprocess_bytes=state.direct_union_postprocess_bytes,
        view_prepare_leases=ViewPrepareLeaseState(state.direct_union_backing_leases,
            state.direct_union_inference_views, state.direct_union_inference_bytes,
            state.direct_union_postprocess_views, state.direct_union_postprocess_bytes),
        view_processing_volume_shape=lambda *_args:shape,_view_uses_interpolation=lambda *_args:interpolation,
        runtime_telemetry=lambda:telemetry)
    source=Path(pipeline.__file__).read_text(encoding='utf-8')
    ns['_sam_parents_ready']=lambda:ns['sam_context'].detector_retirement_ready
    _function(source,'_sam_parent_requires_staging',ns)
    ensure=_function(source,'_ensure_baseline_workspaces',ns)
    allocate=mock.Mock(wraps=pipeline.allocate_workspace_array);ns['allocate_workspace_array']=allocate
    if ram_fit is None:
        with mock.patch.object(staging,'publication_ram_headroom',return_value=staging.RAM_RESERVE_BYTES+10000), \
             mock.patch.object(staging,'workspace_anon_cap_bytes',return_value=0):
            ensure('model',view)
    else:
        with mock.patch.object(staged,'claim_ram_first',return_value=ram_fit):
            ensure('model',view)
    return staged,check,prepare,state,ns,view,allocate,telemetry


def _close_allocated(ns):
    for registry in ('baseline_union_by_model_view','baseline_confmap_by_model_view'):
        for array in ns[registry].values():
            if isinstance(array,np.memmap) and not array._mmap.closed:array._mmap.close()
        ns[registry].clear()


def test_actual_uint8_allocations_start_zero_and_checkpoint_reuses_identical_mask_confidence(tmp_path):
    staged,check,prepare,state,ns,view,allocate,telemetry=allocation(tmp_path)
    key=('model',view.name)
    mask=ns['baseline_union_by_model_view'].pop(key);confidence=ns['baseline_confmap_by_model_view'].pop(key)
    assert isinstance(mask,np.memmap) and isinstance(confidence,np.memmap)
    assert mask.dtype==confidence.dtype==np.uint8
    assert not mask.any() and not confidence.any()
    assert all(call.kwargs['prefer_memory'] is False and call.kwargs['prefer_memfd'] is False
               for call in allocate.call_args_list)
    mask[1,1:3,2:4]=1;confidence[1,1:3,2:4]=np.array([[0,1],[128,255]],np.uint8)
    wanted_mask,wanted_conf=mask.copy(),confidence.copy()
    task=TinyTask(staged.source_root,view.name,mask,confidence)
    task.union_path=ns['baseline_union_paths'][key];task.confmap_path=ns['baseline_confmap_paths'][key]
    paths=(task.union_path,task.confmap_path)
    # The normal allocator's logical admission is identical to two uint8 owners.
    assert state.direct_union_inference_bytes[key]==120
    assert state.direct_union_backing_leases[key].ram_commitment_bytes == 0
    from XTA.view_prepare import ViewPrepareLeaseState
    leases=ViewPrepareLeaseState(state.direct_union_backing_leases,state.direct_union_inference_views,
        state.direct_union_inference_bytes,state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    staged.leases=leases;assert leases.handoff(key)
    staged.defer(task,120);del mask,confidence
    prepare.finish();resumed,released=staged.pump()
    assert released and not resumed and not leases.leases
    snapshot=staged.deferred[key][1]
    assert snapshot.mask.reused and snapshot.confidence.reused
    assert snapshot.mask.path==paths[0] and snapshot.confidence.path==paths[1]
    assert snapshot.written_bytes==0 and snapshot.reused_bytes==120
    assert staged.snapshot()['written_bytes']==0 and staged.snapshot()['reused_regular_disk_bytes']==120
    assert not tuple(staged.root.rglob('*.dat'))  # No duplicate checkpoint payload.
    for saved,wanted in ((snapshot.mask,wanted_mask),(snapshot.confidence,wanted_conf)):
        reopened=saved.open();np.testing.assert_array_equal(reopened,wanted)
        assert hashlib.sha256(reopened.tobytes()).digest()==hashlib.sha256(wanted.tobytes()).digest()
        reopened._mmap.close()
    telemetry.add.assert_called_with('sam_interpolation.staging.disk_from_birth_parents',1)
    staged.ready=lambda:True
    resumed,released=staged.pump();assert len(resumed)==1
    assert leases.leases[key].ram_commitment_bytes == 0
    prepare.finish();result=next(iter(resumed)).result()
    np.testing.assert_array_equal(result.final_view_volume_mm,wanted_mask)
    np.testing.assert_array_equal(result.confidence,wanted_conf)
    task.union_mm._mmap.close();task.confmap_mm._mmap.close()
    staged.close();staged.finalize_cleanup()
    assert all(path.exists()for path in paths)  # Original temp owners belong to normal pipeline cleanup.


def test_backing_classification_keeps_lazy_mixed_cow_unknown_and_future_ram(tmp_path):
    disk = np.memmap(tmp_path/'ordinary.dat', mode='w+', dtype=np.uint8, shape=(2,3,4))
    memory = np.zeros(disk.shape, np.uint8)
    cow = np.memmap(disk.filename, mode='c', dtype=np.uint8, shape=disk.shape)
    try:
        assert classify_dense_ram_backing((disk, None)) is False
        assert classify_dense_ram_backing((memory,)) is True  # Zero/unfaulted pages stay promised.
        assert classify_dense_ram_backing((cow,)) is True
        assert classify_dense_ram_backing((disk, memory)) is None
        assert classify_dense_ram_backing((disk,), future_ram_bytes=1) is None
        assert classify_dense_ram_backing((disk[:],)) is None  # A derived/unproven owner remains charged.
        assert classify_dense_ram_backing((None,)) is None
        with mock.patch('XTA.view_prepare._mount_fstype_for_path', return_value='tmpfs'):
            assert classify_dense_ram_backing((disk,)) is True
        with mock.patch('XTA.view_prepare._mount_fstype_for_path', return_value=None), \
             mock.patch('XTA.view_prepare._windows_volume_ram_backing', return_value=None):
            assert classify_dense_ram_backing((disk,)) is None
        with mock.patch.object(type(tmp_path),'is_symlink',return_value=True):
            assert classify_dense_ram_backing((disk,)) is None
        with mock.patch.object(type(tmp_path),'resolve',return_value=tmp_path/'other-backing'):
            assert classify_dense_ram_backing((disk,)) is None
        with mock.patch('XTA.view_prepare._memfd_owner_key_from_array', return_value='registered-memfd'):
            assert classify_dense_ram_backing((disk,)) is True
    finally:
        cow._mmap.close();disk._mmap.close()
    assert classify_dense_ram_backing((disk,)) is None


@pytest.mark.parametrize('drive_kind,expected', [(0,None),(1,None),(2,False),(3,False),(4,False),(5,False),(6,True)])
def test_native_windows_volume_probe_fails_closed_for_unknown_drives(tmp_path,drive_kind,expected):
    from XTA.view_prepare import _windows_volume_ram_backing
    def volume_path(_path, output, _length):
        output.value = str(tmp_path.anchor)
        return True
    kernel=SimpleNamespace(GetVolumePathNameW=volume_path,GetDriveTypeW=mock.Mock(return_value=drive_kind))
    with mock.patch('ctypes.WinDLL',create=True,return_value=kernel):
        assert _windows_volume_ram_backing(tmp_path) is expected
    kernel.GetVolumePathNameW=mock.Mock(return_value=False)
    kernel.GetDriveTypeW.reset_mock()
    with mock.patch('ctypes.WinDLL',create=True,return_value=kernel):
        assert _windows_volume_ram_backing(tmp_path) is None
    kernel.GetDriveTypeW.assert_not_called()


def test_disk_input_is_charged_before_preparation_can_replace_it_with_ram(tmp_path):
    from tests.test_view_prepare_input_retirement import _task
    task, expected = _task(tmp_path)
    key = (task.model_name,task.view.name)
    lease = _DirectUnionBackingLease(key,task.union_mm.nbytes,phase='postprocess',ram_backed=False)
    task.backing_lease = lease
    original_prepare = task.prepare
    def prepare(**kwargs):
        assert lease.ram_backed is None and lease.ram_commitment_bytes == expected.nbytes
        return original_prepare(**kwargs)  # Existing prepare fixture replaces disk input with an ndarray.
    task.prepare = prepare
    assert lease.ram_commitment_bytes == 0
    result = task()
    assert lease.nbytes == expected.nbytes and lease.ram_commitment_bytes == expected.nbytes
    np.testing.assert_array_equal(result.final_view_volume_mm,expected)
    task = result = None
    gc.collect()
    wait_for_retired_memmap_unlinks(path=tmp_path/'original.u8.dat')


def test_disk_logical_credit_does_not_block_ram_birth_and_partial_retirement_stays_exact(tmp_path, monkeypatch):
    stage,_writer,_prepare,leases = queue(tmp_path, cap=200)
    disk, memory = ('model','disk'), ('model','ram')
    leases.leases[disk] = _DirectUnionBackingLease(disk,140,ram_backed=False)
    leases.leases[memory] = _DirectUnionBackingLease(memory,40,ram_backed=None)
    leases.inference_views.update((disk,memory))
    leases.inference_bytes.update({disk:140,memory:40})
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+100)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
    assert not staging._ram_fits(40, sum(leases.inference_bytes.values()), 200)
    assert leases.ram_commitment_bytes == 40 and leases.disk_backed_logical_bytes == 140
    assert stage.claim_ram_first(('model','new'),20,leases.ram_commitment_bytes,20)
    assert leases.inference_bytes == {disk:140,memory:40}  # Detector/logical admission is unchanged.
    assert leases.handoff(disk) and leases.handoff(memory)
    assert leases.retire_input_bytes(disk,leases.leases[disk],60,token='confidence')
    assert leases.leases[disk].nbytes == 80 and leases.leases[disk].ram_commitment_bytes == 0
    assert leases.retire_input_bytes(memory,leases.leases[memory],10,token='confidence')
    assert leases.ram_commitment_bytes == 30 and leases.disk_backed_logical_bytes == 80
    # Remaining lazy memory still needs real headroom; excluding disk never invents RAM.
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+69)
    assert not stage.claim_ram_first(('model','too-much'),20,leases.ram_commitment_bytes,20)
    receipt=stage.snapshot()['last_ram_denial']
    assert receipt['reason']=='physical_headroom'
    assert receipt['active_ram_commitments_bytes']==30 and receipt['excluded_disk_logical_bytes']==80
    assert receipt['total_checked_bytes']==70 and receipt['physical_headroom_bytes']==staging.RAM_RESERVE_BYTES+69
    leases.complete(disk,retain_for_dense_retirement=False)
    leases.complete(memory,retain_for_dense_retirement=False)
    stage.release_ram_first(('model','new'))
    stage.close();stage.finalize_cleanup()


@pytest.mark.parametrize('guard', ['ram_owner_limit','dense_limit','anonymous_workspace_cap','physical_headroom','cancelled'])
def test_last_denial_retains_requested_reserve_and_exact_guard(tmp_path,monkeypatch,guard):
    stage,_writer,_prepare,_leases=queue(tmp_path,cap=100)
    monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+1000)
    monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:0)
    required, active, detector = 20, 30, 10
    if guard=='ram_owner_limit': required=101
    elif guard=='dense_limit': active=80
    elif guard=='anonymous_workspace_cap': monkeypatch.setattr(staging,'workspace_anon_cap_bytes',lambda:59)
    elif guard=='physical_headroom': monkeypatch.setattr(staging,'publication_ram_headroom',lambda:staging.RAM_RESERVE_BYTES+59)
    elif guard=='cancelled': stage.stop.set()
    assert not stage.claim_ram_first(('model','parent'),required,active,detector)
    receipt=stage.snapshot()['last_ram_denial']
    assert receipt['reason']==guard and receipt['requested_owner_bytes']==required
    assert receipt['detector_room_bytes']==detector and receipt['active_ram_commitments_bytes']==active
    stage.close();stage.finalize_cleanup()


@pytest.mark.parametrize('options', [
    {'backend':'sdf'}, {'ready':True}, {'shared':False}, {'stager':False},
    {'interpolation':False}, {'tmpfs':True},
])
def test_noneligible_and_tmpfs_paths_preserve_existing_allocation_choices(tmp_path,monkeypatch,options):
    staged,_check,_prepare,_state,ns,_view,allocate,telemetry=allocation(tmp_path,monkeypatch=monkeypatch,**options)
    try:
        assert all(call.kwargs['prefer_memfd'] is True for call in allocate.call_args_list)
        assert all(call.kwargs['prefer_memory'] is False for call in allocate.call_args_list)
        assert not any(call.args[0]=='sam_interpolation.staging.disk_from_birth_parents'
                       for call in telemetry.add.call_args_list)
        if options.get('tmpfs'):
            telemetry.add.assert_called_with('sam_interpolation.staging.checkpoint_copy_backing_parents',1)
    finally:_close_allocated(ns);staged.close();staged.finalize_cleanup()


def test_extrapolation_only_shared_parent_is_eligible(tmp_path):
    staged,_check,_prepare,_state,ns,_view,allocate,_telemetry=allocation(tmp_path,backend='sdf',
        interpolation=False,extrapolation=True)
    try:assert all(call.kwargs['prefer_memfd'] is False for call in allocate.call_args_list)
    finally:_close_allocated(ns);staged.close();staged.finalize_cleanup()


def test_admitted_ram_first_attempt_preserves_mask_score_and_dense_reservation(tmp_path):
    staged,_check,_prepare,state,ns,view,allocate,telemetry=allocation(tmp_path,ram_fit=True)
    try:
        assert all(call.kwargs['prefer_memfd'] is True for call in allocate.call_args_list)
        assert state.direct_union_inference_bytes[('model',view.name)] == 120
        assert not ns['baseline_union_by_model_view'][('model',view.name)].any()
        assert not ns['baseline_confmap_by_model_view'][('model',view.name)].any()
        telemetry.add.assert_any_call('sam_interpolation.staging.ram_first_attempt_parents',1)
    finally:_close_allocated(ns);staged.close();staged.finalize_cleanup()


def test_regular_backing_requires_run_owned_paths_and_stable_identity(tmp_path):
    staged,_check,_prepare,_leases=queue(tmp_path)
    root=staged.source_root;root.mkdir(exist_ok=True)
    assert staged.reusable_workspace_paths(root/'union'/'mask.dat',root/'union'/'conf.dat')
    assert not staged.reusable_workspace_paths(tmp_path/'foreign-input.dat')
    path=root/'mask.dat';array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(3,4,5));array[:]=1
    task=TinyTask(root,'test',array);task.union_path=path
    snapshot=staging.checkpoint_parent(task,staged.root,root,60,staged.stop)
    assert snapshot.written_bytes==0 and snapshot.mask.reused
    # Same length content mutation must still fail the frozen mtime identity.
    saved_mtime=path.stat().st_mtime_ns
    path.write_bytes(bytes([9])*60)
    os.utime(path,ns=(saved_mtime,saved_mtime+1_000_000_000))
    with pytest.raises(RuntimeError,match='changed before resume'):snapshot.mask.open()
    staged.close();staged.finalize_cleanup()
    assert path.exists()


def test_cancellation_keeps_original_regular_input_and_releases_owner(tmp_path):
    staged,check,prepare,leases=queue(tmp_path)
    original=tmp_path/'source-input.bin';original.write_bytes(b'immutable source input')
    path=staged.source_root/'owned-workspace.dat';array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(3,4,5));array[:]=7
    task=TinyTask(staged.source_root,'test',array);task.union_path=path
    from tests.test_sam_parent_staging_adversarial import admit
    admit(leases,task,60);staged.defer(task,60);staged.abort();check.finish()
    staged.close();staged.finalize_cleanup();del array;gc.collect()
    wait_for_retired_memmap_unlinks(path=path,timeout_s=5.)
    assert not path.exists() and not leases.leases
    assert original.read_bytes()==b'immutable source input'


def test_failed_confidence_allocation_cleans_only_owned_workspace(tmp_path):
    original=tmp_path/'source-input.bin';original.write_bytes(b'immutable source input')
    actual=pipeline.allocate_workspace_array;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==2:raise OSError('controlled confidence allocation failure')
        return actual(*args,**kwargs)
    with mock.patch.object(pipeline,'allocate_workspace_array',side_effect=fail_second):
        with pytest.raises(OSError,match='controlled confidence'):
            allocation(tmp_path)
    gc.collect()
    for path in (tmp_path/'temp'/'union').rglob('*.dat'):
        wait_for_retired_memmap_unlinks(path=path,timeout_s=5.)
    assert not list((tmp_path/'temp'/'union').rglob('*.dat'))
    assert original.read_bytes()==b'immutable source input'


def test_fsync_failure_cleans_owned_mapping_without_touching_source_input(tmp_path,monkeypatch):
    original=tmp_path/'source-input.bin';original.write_bytes(b'immutable source input')
    staged,check,prepare,leases=queue(tmp_path)
    path=staged.source_root/'owned-workspace.dat';array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(3,4,5));array[:]=1
    task=TinyTask(staged.source_root,'test',array);task.union_path=path
    from tests.test_sam_parent_staging_adversarial import admit
    admit(leases,task,60);staged.defer(task,60)
    error=OSError('controlled fsync failure')
    monkeypatch.setattr(staging.os,'fsync',mock.Mock(side_effect=error))
    prepare.finish()
    with pytest.raises(OSError,match='controlled fsync'):staged.pump()
    staged.close();staged.finalize_cleanup()
    assert task.union_mm is None and array._mmap.closed and not leases.leases
    # The deliberately retained exception keeps its array argument alive via
    # the traceback. Normal retirement unlinks once that last reference dies.
    error.__traceback__=None
    del array;gc.collect()
    wait_for_retired_memmap_unlinks(path=path,timeout_s=5.)
    assert not path.exists() and not leases.leases
    assert original.read_bytes()==b'immutable source input'


def test_real_birth_token_returns_after_all_regular_disk_allocation_fallback(tmp_path):
    actual=pipeline.allocate_workspace_array
    def disk_only(*args,**kwargs):
        kwargs['prefer_memory']=kwargs['prefer_memfd']=False
        return actual(*args,**kwargs)
    with mock.patch.object(pipeline,'allocate_workspace_array',side_effect=disk_only):
        staged,_check,_prepare,state,ns,view,_allocate,_telemetry=allocation(tmp_path,ram_fit=None,cap=240)
    try:
        assert staged.ram_first_key is None
        assert staged.ram_first_grants==staged.ram_first_releases==1
        assert state.direct_union_inference_bytes[('model',view.name)]==120
        assert all(isinstance(value,np.memmap) for registry in
            ('baseline_union_by_model_view','baseline_confmap_by_model_view') for value in ns[registry].values())
    finally:_close_allocated(ns);staged.close();staged.finalize_cleanup()


@pytest.mark.parametrize('retain_alias',[False,True])
def test_failed_birth_allocation_keeps_bank_until_real_ram_owner_dies(tmp_path,retain_alias):
    held=[];calls=[];captured=[]
    original_claim=staging.DeferredSamParentQueue.claim_ram_first
    def record_claim(stage,*args,**kwargs):
        captured.append(stage)
        return original_claim(stage,*args,**kwargs)
    def fail_confidence(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==2:raise OSError('controlled bank confidence allocation failure')
        value=np.zeros(kwargs['shape'],np.uint8)
        if retain_alias:held.append(value)
        return value
    with mock.patch.object(staging.DeferredSamParentQueue,'claim_ram_first',new=record_claim), \
         mock.patch.object(pipeline,'allocate_workspace_array',side_effect=fail_confidence):
        with pytest.raises(OSError,match='bank confidence allocation failure'):
            allocation(tmp_path,ram_fit=None,cap=240)
    staged=captured[0]
    assert staged.ram_first_grants==1
    assert (staged.ram_first_key is not None)==retain_alias
    assert staged.ram_first_releases==int(not retain_alias)
    if retain_alias:
        assert np.all(held[0]==0)
        held.clear();gc.collect()
        staged.release_ram_first(('model','test__tta_a0'))
    staged.close();staged.finalize_cleanup()
