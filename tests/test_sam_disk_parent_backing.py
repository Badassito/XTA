"""Blocked shared-SAM parents reuse exact ordinary-disk owners from birth."""
import gc
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pipeline, sam_parent_staging as staging
from XTA.geometry import ViewInfo
from XTA.runtime import wait_for_retired_memmap_unlinks
from tests.test_sam_parent_staging_adversarial import TinyTask, queue
from tests.test_terminal_component_refs import _function
from tests.test_tta_scheduler_boundary import _state


def allocation(tmp_path, *, backend='sam', extrapolation=False, ready=False, shared=True,
               stager=True, interpolation=True, tmpfs=False, monkeypatch=None):
    staged,check,prepare,leases=queue(tmp_path,cap=120,ready=lambda:ready)
    if tmpfs:
        monkeypatch.setattr(staging,'path_is_memory_backed',lambda path:
            Path(path).resolve().is_relative_to(staged.source_root.resolve()))
    state=_state();shape=(3,4,5)
    view=ViewInfo(name='test__tta_a0',family='tilted',num_slices=3,src_h=4,src_w=5,pad_mode='clamp')
    ns=dict(vars(pipeline));telemetry=mock.Mock()
    ns.update(temp_dir=staged.source_root,worker_direct_union_active=True,
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
        view_processing_volume_shape=lambda *_args:shape,_view_uses_interpolation=lambda *_args:interpolation,
        runtime_telemetry=lambda:telemetry)
    source=Path(pipeline.__file__).read_text(encoding='utf-8')
    _function(source,'_sam_parent_requires_staging',ns)
    ensure=_function(source,'_ensure_baseline_workspaces',ns)
    allocate=mock.Mock(wraps=pipeline.allocate_workspace_array);ns['allocate_workspace_array']=allocate
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
    from XTA.view_prepare import ViewPrepareLeaseState
    leases=ViewPrepareLeaseState(state.direct_union_backing_leases,state.direct_union_inference_views,
        state.direct_union_inference_bytes,state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    staged.leases=leases;assert leases.handoff(key)
    staged.defer(task,120);del mask,confidence
    check.finish();resumed,released=staged.pump()
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
    prepare.finish();result=next(iter(resumed)).result()
    np.testing.assert_array_equal(result.final_view_volume_mm,wanted_mask)
    np.testing.assert_array_equal(result.confidence,wanted_conf)
    task.union_mm._mmap.close();task.confmap_mm._mmap.close()
    staged.close();staged.finalize_cleanup()
    assert all(path.exists()for path in paths)  # Original temp owners belong to normal pipeline cleanup.


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
    staged,check,_prepare,leases=queue(tmp_path)
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
    staged,check,_prepare,leases=queue(tmp_path)
    path=staged.source_root/'owned-workspace.dat';array=np.memmap(path,mode='w+',dtype=np.uint8,shape=(3,4,5));array[:]=1
    task=TinyTask(staged.source_root,'test',array);task.union_path=path
    from tests.test_sam_parent_staging_adversarial import admit
    admit(leases,task,60);staged.defer(task,60)
    error=OSError('controlled fsync failure')
    monkeypatch.setattr(staging.os,'fsync',mock.Mock(side_effect=error))
    check.finish()
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
