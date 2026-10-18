"""RAM backlog policy cannot bypass physical birth or file-owner retirement."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import gc
import os
import threading

import numpy as np
import pytest

from XTA import pipeline, sam_parent_staging as staging
from XTA.detector_workspaces import WorkspaceBirthFailure
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_sam_parent_staging_adversarial import TinyTask, queue
from tests.test_sam_disk_parent_backing import allocation, _close_allocated


def ordinary_queue(tmp_path, monkeypatch, **kwargs):
    temp = (tmp_path/'temp').absolute()
    monkeypatch.setattr(staging, 'path_is_memory_backed',
        lambda path: Path(path).absolute().is_relative_to(temp))
    return queue(tmp_path, **kwargs)


@pytest.mark.parametrize('logical_limit', ('dense_limit', 'ram_owner_limit'))
def test_short_lived_birth_uses_physical_proof_not_backlog_quota(tmp_path, monkeypatch, logical_limit):
    stage, *_ = queue(tmp_path, cap=100)
    monkeypatch.setattr(staging, 'workspace_anon_cap_bytes', lambda: 0)
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES+117)
    if logical_limit == 'ram_owner_limit':
        stage.ram_first_owners[('older', 'parent')] = 100
    plan = stage.plan_ram_first(('model','next'),20,80,10,external_future_bytes=17)
    assert stage.qualify_ram_first(plan)[0] is False
    allowed, receipt = stage.qualify_detector_ram_birth(plan)
    assert allowed and receipt['physical_required_bytes'] == staging.RAM_RESERVE_BYTES+117
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES+116)
    allowed, receipt = stage.qualify_detector_ram_birth(plan)
    assert not allowed and receipt['reason'] == 'physical_headroom'
    assert stage.leases.leases == {}  # The proof does not invent a grant.
    stage.close();stage.finalize_cleanup()


def test_low_headroom_uses_owned_files_and_reuses_exact_mask_and_scores(tmp_path, monkeypatch):
    temp = (tmp_path/'temp').absolute()
    monkeypatch.setattr(staging, 'path_is_memory_backed',
        lambda path: Path(path).absolute().is_relative_to(temp))
    stage, checkpoint, prepare, state, ns, view, allocator, _ = allocation(tmp_path)
    key = ('model', view.name)
    mask = ns['baseline_union_by_model_view'].pop(key)
    scores = ns['baseline_confmap_by_model_view'].pop(key)
    paths = (ns['baseline_union_paths'][key], ns['baseline_confmap_paths'][key])
    assert all(Path(path).is_relative_to(stage.root) for path in paths)
    assert all(not Path(path).is_relative_to(temp) for path in paths)
    assert all(call.kwargs['reuse_existing'] and not call.kwargs['prefer_memfd']
               and not call.kwargs['prefer_memory'] for call in allocator.call_args_list)
    assert mask.mode == scores.mode == 'r+'
    assert not mask.any() and not scores.any()
    mask[1,1:3,2:4] = 1
    scores[1,1:3,2:4] = [[0,1],[128,255]]
    expected_mask, expected_scores = mask.copy(), scores.copy()
    task = TinyTask(stage.source_root, view.name, mask, scores)
    task.union_path, task.confmap_path = paths
    del mask,scores
    stage.leases = ViewPrepareLeaseState(state.direct_union_backing_leases,
        state.direct_union_inference_views,state.direct_union_inference_bytes,
        state.direct_union_postprocess_views,state.direct_union_postprocess_bytes)
    assert stage.leases.handoff(key)
    stage.defer(task,120)
    assert not checkpoint.entries and len(prepare.entries)==1
    prepare.finish()
    stage.pump()
    snapshot = stage.deferred[key][1]
    assert snapshot.owned_dir == Path(paths[0]).parent
    assert snapshot.mask.reused and snapshot.confidence.reused and snapshot.written_bytes==0
    assert task.union_mm is task.confmap_mm is None
    restored_mask, restored_scores = snapshot.mask.open(), snapshot.confidence.open()
    np.testing.assert_array_equal(restored_mask,expected_mask)
    np.testing.assert_array_equal(restored_scores,expected_scores)
    # Model the normal transfer of a raw checkpoint to its retained consumer.
    stage.transferred_roots.add(snapshot.owned_dir)
    stage.close()
    assert all(Path(path).exists() for path in paths)
    np.testing.assert_array_equal(restored_scores,expected_scores)
    restored_mask._mmap.close();restored_scores._mmap.close()
    stage.finalize_cleanup()
    assert not stage.root.exists()


def test_healthy_physical_ram_preserves_backlog_denied_fast_path(tmp_path, monkeypatch):
    stage, *_rest, ns, _view, allocator, _telemetry = allocation(
        tmp_path, ram_fit=False, birth_ram_fit=True)
    try:
        assert all(call.kwargs['prefer_memfd'] and not call.kwargs['reuse_existing']
                   for call in allocator.call_args_list)
        assert not stage._detector_disk_workspaces
    finally:
        _close_allocated(ns);stage.close();stage.finalize_cleanup()


@pytest.mark.parametrize('filesystem', (True, None))
def test_disk_fallback_refuses_memory_or_unknown_filesystem(tmp_path, monkeypatch, filesystem):
    stage, *_ = ordinary_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(staging, '_dense_mapping_ram_backing', lambda path: filesystem)
    with pytest.raises(RuntimeError, match='verified non-memory'):
        stage.detector_disk_workspace(('model','parent'),(60,60))
    assert not list(stage.root.iterdir())
    stage.close();stage.finalize_cleanup()


def test_space_check_covers_both_inputs_before_any_file_birth(tmp_path, monkeypatch):
    stage, *_ = ordinary_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(staging.shutil, 'disk_usage', lambda root:
        SimpleNamespace(free=staging.STREAM_BYTES+119))
    with pytest.raises(RuntimeError, match='mask and confidence'):
        stage.detector_disk_workspace(('model','parent'),(60,60))
    assert not list(stage.root.iterdir()) and not stage._detector_disk_workspaces
    stage.close();stage.finalize_cleanup()


def test_sparse_births_cannot_each_spend_the_same_filesystem_space(tmp_path, monkeypatch):
    stage, *_ = ordinary_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(staging.os,'posix_fallocate',None,raising=False)
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda root:SimpleNamespace(free=staging.STREAM_BYTES+200))
    with pytest.raises(RuntimeError,match='filesystem space'):
        with stage.detector_disk_birth_group(240):
            pytest.fail('Sibling admission was not checked before the first file birth')
    assert not list(stage.root.iterdir())
    first=stage.detector_disk_workspace(('m','first'),(60,60))
    assert stage._unwritten_detector_disk_bytes()==120
    with pytest.raises(RuntimeError,match='filesystem space'):
        stage.detector_disk_workspace(('m','second'),(60,60))
    assert len(stage._detector_disk_workspaces)==1
    stage.retire_detector_disk_workspace(first)
    assert not stage._detector_disk_workspaces
    second=stage.detector_disk_workspace(('m','second'),(60,60))
    stage.retire_detector_disk_workspace(second)
    stage.close();stage.finalize_cleanup()


def test_mapping_keeps_reserved_file_identity_extents_and_existing_bytes(tmp_path,monkeypatch):
    stage, *_ = ordinary_queue(tmp_path,monkeypatch)
    with mock.patch.object(staging.os,'open',wraps=os.open) as opened:
        proof=stage.detector_disk_workspace(('m','reserved'),(8192,))
    created=[call for call in opened.call_args_list if call.args[1]&os.O_CREAT]
    assert len(created)==1 and created[0].args[2]==0o600
    path,identity=proof['files'][0]
    with path.open('r+b') as stream:stream.write(bytes(range(64)))
    before=path.stat()
    array=pipeline.allocate_workspace_array((2,64,64),np.uint8,path,'reserved test',
        prefer_memory=False,prefer_memfd=False,reuse_existing=True)
    try:
        after=path.stat()
        assert array.mode=='r+'
        assert (after.st_dev,after.st_ino,after.st_size)==identity
        assert array.reshape(-1)[:64].tobytes()==bytes(range(64))
        if hasattr(before,'st_blocks'):
            assert after.st_blocks>=before.st_blocks
        stage.validate_detector_disk_workspace(proof)
    finally:array._mmap.close()
    stage.retire_detector_disk_workspace(proof)
    stage.close();stage.finalize_cleanup()


@pytest.mark.parametrize('method', ('close','finalize_cleanup'))
def test_cleanup_refuses_a_replaced_root_without_touching_foreign_contents(tmp_path,monkeypatch,method):
    stage, *_ = ordinary_queue(tmp_path,monkeypatch)
    original=stage.root.with_name(stage.root.name+'-original')
    assert stage.root.resolve().is_relative_to(tmp_path.resolve())
    assert original.parent.resolve().is_relative_to(tmp_path.resolve())
    stage.root.rename(original)
    stage.root.mkdir();foreign=stage.root/'foreign-marker';foreign.write_text('keep')
    try:
        with pytest.raises(RuntimeError,match='root identity changed'):getattr(stage,method)()
        assert foreign.read_text()=='keep'
    finally:
        foreign.unlink();stage.root.rmdir();original.rename(stage.root)
    stage.close();stage.finalize_cleanup()
    stage.close();stage.finalize_cleanup()  # Already-absent owned root is idempotent.


@pytest.mark.parametrize('failure_index,keep_alias', ((1,False),(2,False),(2,True)))
def test_actual_map_failure_retires_files_only_after_every_private_alias(tmp_path,monkeypatch,failure_index,keep_alias):
    real_stage=staging.DeferredSamParentQueue
    real_allocate=pipeline.allocate_workspace_array
    stages=[];aliases=[];calls=[]
    primary=OSError('controlled private map failure')
    foreign=tmp_path/'original-input';foreign.write_bytes(b'unchanged')
    def capture_stage(*args,**kwargs):
        value=real_stage(*args,**kwargs);stages.append(value);return value
    def allocate_or_fail(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==failure_index:raise primary
        array=real_allocate(*args,**kwargs)
        if keep_alias:aliases.append(array.view())
        return array
    monkeypatch.setattr(staging,'DeferredSamParentQueue',capture_stage)
    monkeypatch.setattr(pipeline,'allocate_workspace_array',allocate_or_fail)
    with pytest.raises(WorkspaceBirthFailure,match='controlled private map failure'):
        allocation(tmp_path)
    stage=stages[0]
    assert foreign.read_bytes()==b'unchanged'
    if keep_alias:
        assert len(stage._detector_disk_workspaces)==1
        proof=next(iter(stage._detector_disk_workspaces.values()))
        assert all(path.exists() for path,_identity in proof['files'])
        stage.close()
        assert all(path.exists() for path,_identity in proof['files'])
        aliases.clear();gc.collect()
    stage.close()
    assert not stage._detector_disk_workspaces and not stage._detector_disk_retirements
    assert not list(stage.root.rglob('*.dat'))
    stage.finalize_cleanup()


def test_partial_file_creation_preserves_primary_error_and_foreign_files(tmp_path, monkeypatch):
    stage, *_ = ordinary_queue(tmp_path, monkeypatch)
    foreign=tmp_path/'foreign.txt';foreign.write_text('keep')
    truncate=os.ftruncate;calls=[];primary=OSError('controlled second file failure')
    def fail_second(fd,size):
        calls.append(size)
        if len(calls)==2:raise primary
        truncate(fd,size)
    monkeypatch.setattr(staging.os,'ftruncate',fail_second)
    with pytest.raises(OSError) as caught:
        stage.detector_disk_workspace(('model','parent'),(60,60))
    assert caught.value is primary
    assert foreign.read_text()=='keep'
    assert not list(stage.root.iterdir()) and not stage._detector_disk_workspaces
    stage.close();stage.finalize_cleanup()


def test_existing_directory_or_replaced_target_never_becomes_owned(tmp_path, monkeypatch):
    stage, *_ = ordinary_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(staging.uuid,'uuid4',lambda:SimpleNamespace(hex='fixed'))
    existing=stage.root/'detector-fixed';existing.mkdir()
    foreign=existing/'mask.dat';foreign.write_bytes(b'keep')
    with pytest.raises(FileExistsError):stage.detector_disk_workspace(('m','v'),(60,))
    assert foreign.read_bytes()==b'keep' and not stage._detector_disk_workspaces
    foreign.unlink();existing.rmdir()
    proof=stage.detector_disk_workspace(('m','v'),(60,))
    path=proof['files'][0][0]
    replacement=path.with_suffix('.replacement');replacement.write_bytes(b'x'*60)
    os.replace(replacement,path)
    with pytest.raises(RuntimeError,match='backing changed'):
        stage.validate_detector_disk_workspace(proof)
    assert path.read_bytes()==b'x'*60
    stage.close()
    with pytest.raises(RuntimeError,match='private disk owners remain unsettled'):
        stage.finalize_cleanup()
    assert path.read_bytes()==b'x'*60


@pytest.mark.parametrize('disk', (False,True))
def test_stale_plan_refresh_keeps_actual_existing_backing_without_another_birth(tmp_path, monkeypatch, disk):
    stage, _check, _prepare, _state, ns, view, allocator, _ = allocation(tmp_path,ram_fit=True)
    new_view=replace(view,name='new-parent')
    with mock.patch.object(stage,'claim_ram_first',return_value=False):
        plan=ns['_plan_baseline_workspace']('model',new_view,legacy=True)
    with mock.patch.object(stage,'qualify_detector_ram_birth',return_value=(not disk,{})):
        bundles=ns['_build_baseline_workspaces']((plan,),None,threading.Event())
    count=allocator.call_count;owners=tuple(id(bundles[0][key]) for key in ('union_mm','conf_mm'))
    prior=dict(bundles[0])
    try:
        refresh_plan=dict(plan,ram_plan={'current-owner-snapshot':True})
        with mock.patch.object(stage,'qualify_ram_first',return_value=(False,{'reason':'physical_headroom'})), \
             mock.patch.object(stage,'qualify_detector_ram_birth',side_effect=AssertionError('second birth')):
            refreshed=ns['_build_baseline_workspaces']((refresh_plan,),bundles,threading.Event())
        assert refreshed is bundles and allocator.call_count==count
        assert tuple(id(bundles[0][key]) for key in ('union_mm','conf_mm'))==owners
        assert bundles[0]['disk_birth'] is disk
        assert bundles[0]['disk_workspace'] is prior['disk_workspace']
        assert bundles[0]['ram_qualification'] == (False,{'reason':'physical_headroom'})
    finally:
        prior.clear()
        ns['_discard_baseline_bundles'](bundles);bundles.clear();gc.collect()
        _close_allocated(ns);stage.close();stage.finalize_cleanup()
