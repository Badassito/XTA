"""Independent parent scopes can feed SAM while other scopes consume results."""
from concurrent.futures import ThreadPoolExecutor
import ast
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import runtime
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_parent_staging import DeferredSamParentQueue
from tests.test_sam_parent_staging_adversarial import ManualExecutor, TinyTask, admit, lease_state


def clear_memory_overrides(monkeypatch):
    for name in ('YOLO_TTA_DIRECT_UNION_INFERENCE_GIB', 'YOLO_TTA_DIRECT_UNION_TOTAL_GIB',
                 'YOLO_TTA_PARENT_TRANSIENT_GIB'):
        monkeypatch.delenv(name, raising=False)


def test_sam_memory_defaults_scale_with_one_ram_budget_and_keep_legacy_floors(monkeypatch):
    clear_memory_overrides(monkeypatch)
    gib = runtime.GIB
    limits = runtime.resolve_parent_memory_limits(1400*gib, sam_enabled=True, policy_enabled=False)
    assert limits == (128*gib, int(1336*gib*.40), 334*gib)
    assert limits[1]+limits[2] <= int(1336*gib*.65)
    assert runtime.resolve_parent_memory_limits(1400*gib, sam_enabled=False,
        policy_enabled=False) == (128*gib, 256*gib, 192*gib)
    for sam in (False, True):
        assert runtime.resolve_parent_memory_limits(10*gib, sam_enabled=sam,
            policy_enabled=False) == (64*gib, 128*gib, 64*gib)
        assert runtime.resolve_parent_memory_limits(10*gib, sam_enabled=sam,
            policy_enabled=True) == (64*gib, 384*gib, 64*gib)


def test_explicit_memory_overrides_remain_authoritative(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_DIRECT_UNION_INFERENCE_GIB', '15')
    monkeypatch.setenv('YOLO_TTA_DIRECT_UNION_TOTAL_GIB', '40')
    monkeypatch.setenv('YOLO_TTA_PARENT_TRANSIENT_GIB', '33')
    assert runtime.resolve_parent_memory_limits(1400*runtime.GIB, sam_enabled=True,
        policy_enabled=False) == (15*runtime.GIB,40*runtime.GIB,33*runtime.GIB)


def pipeline_statements():
    return list(ast.walk(ast.parse(Path(runtime.__file__).with_name('pipeline.py').read_text())))


def test_pipeline_uses_one_physical_probe_instead_of_available_ram_plus_swap(monkeypatch):
    from XTA import sam_resources
    clear_memory_overrides(monkeypatch)
    calls = []
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom',
        lambda: calls.append('physical') or 1400*runtime.GIB)
    nodes = pipeline_statements()
    probe = next(node for node in nodes if isinstance(node, ast.If) and any(
        isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == '_parent_startup_headroom' for target in child.targets)
        for child in node.body))
    limits = next(node for node in nodes if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Tuple) and any(isinstance(item, ast.Name)
            and item.id == '_parent_transient_bytes' for item in target.elts) for target in node.targets))
    namespace = dict(interpolation_settings=SimpleNamespace(sam_enabled=True),
        policy_settings=SimpleNamespace(enabled=False),
        available_anon_work_bytes=lambda: calls.append('RAM+swap') or 2800*runtime.GIB,
        resolve_parent_memory_limits=runtime.resolve_parent_memory_limits,
        __package__='XTA')
    exec(compile(ast.Module(body=[probe,limits],type_ignores=[]), '<actual pipeline budget>', 'exec'), namespace)
    assert calls == ['physical']
    assert namespace['direct_union_total_dense_byte_limit'] == int(1336*runtime.GIB*.40)


def test_sam_policy_plan_reserves_extra_credit_pool_before_clamping_dense(monkeypatch):
    from XTA.publication_memory import policy_parent_memory_plan
    clear_memory_overrides(monkeypatch)
    block = next(node for node in pipeline_statements() if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute) and node.test.attr == 'sam_enabled'
        and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'parent_reserve' for target in child.targets) for child in node.body))
    available = 1400*runtime.GIB
    capacity = available//8  # The unchanged policy physical-headroom clamp.
    namespace = dict(interpolation_settings=SimpleNamespace(sam_enabled=True),
        parent_working=4*runtime.GIB, parent_reserve=32*runtime.GIB,
        parent_transient_admission=SimpleNamespace(capacity=capacity))
    exec(compile(ast.Module(body=[block],type_ignores=[]), '<actual policy reserve>', 'exec'), namespace)
    assert namespace['parent_reserve'] == capacity
    _, dense, _ = runtime.resolve_parent_memory_limits(available, sam_enabled=True, policy_enabled=True)
    plan = policy_parent_memory_plan([], requested_dense_limit=dense, available_ram_bytes=available,
        source_shape=(1931,3064,3022), output_reserve_bytes=32*runtime.GIB,
        parent_transient_reserve_bytes=namespace['parent_reserve'], worker_buffer_reserve_bytes=4*runtime.GIB)
    assert plan['parent_transient_reserve_bytes'] == capacity
    assert plan['total_reserve_bytes'] <= available


@pytest.mark.parametrize('sam,expected_gib', [(False,1400), (True,240)])
def test_policy_preflight_honors_slurm_ram_without_a_cgroup(monkeypatch, sam, expected_gib):
    import psutil
    from XTA import publication_memory, workspace
    monkeypatch.setenv('SLURM_MEM_PER_NODE', str(256*1024))
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', lambda: 1400*runtime.GIB)
    monkeypatch.setattr(workspace, 'available_anon_work_bytes', lambda: 2800*runtime.GIB)
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=16*runtime.GIB), children=lambda **_kwargs: []))
    branch = next(node for node in pipeline_statements() if isinstance(node, ast.If)
        and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'policy_headroom' for target in child.targets) for child in node.body))
    from XTA.sam_resources import physical_sam_headroom
    namespace = dict(interpolation_settings=SimpleNamespace(sam_enabled=sam),
        publication_ram_headroom=publication_memory.publication_ram_headroom,
        physical_sam_headroom=physical_sam_headroom,
        parent_transient_admission=SimpleNamespace(capacity=334*runtime.GIB))
    # Execute the actual probe, SAM clamp and pool clamp before dispatch.
    exec(compile(ast.Module(body=branch.body[:3],type_ignores=[]), '<actual policy preflight>', 'exec'), namespace)
    assert namespace['policy_headroom'] == expected_gib*runtime.GIB
    assert namespace['parent_transient_admission'].capacity == expected_gib*runtime.GIB//8


def allocation(monkeypatch, slots, *, budget=128, count=20, headroom_gib=256):
    from XTA import sam_resources
    monkeypatch.delenv('YOLO_TTA_PARENT_POSTPROCESS_WORKERS', raising=False)
    monkeypatch.delenv('YOLO_TTA_PARENT_SLICE_WORKERS', raising=False)
    monkeypatch.delenv('YOLO_TTA_PARENT_POSTPROCESS_RESERVE_GIB', raising=False)
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', lambda: headroom_gib*runtime.GIB)
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom', lambda: headroom_gib*runtime.GIB)
    view = SimpleNamespace(name='coronal', family='orthogonal', src_h=4, src_w=5, num_slices=3)
    return runtime.resolve_parent_postprocess_worker_allocation(worker_budget=budget,
        views=[view]*count, nrrd_layers_enabled=True, interpolation_enabled=True,
        sam_execution_slots=slots)


@pytest.mark.parametrize('slots,budget,count,headroom,expected', [
    (0,128,20,256,4), (4,128,20,256,8), (8,128,20,256,16),
    (8,4,20,256,4), (8,128,3,256,3), (8,128,20,25,2),
    (8,512,64,256,32), (8,512,64,25,2), (0,512,64,256,4)])
def test_parent_default_tracks_sam_capacity_with_existing_memory_and_cpu_bounds(
        monkeypatch, slots, budget, count, headroom, expected):
    outer, slices, _bytes, _memory, default = allocation(monkeypatch, slots,
        budget=budget, count=count, headroom_gib=headroom)
    assert outer == default == expected
    assert slices == budget//expected
    assert outer*slices <= budget


def test_sam_parent_startup_memory_cap_excludes_swap(monkeypatch):
    allocation(monkeypatch, 8, budget=512, count=64, headroom_gib=25)
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', lambda: 2800*runtime.GIB)
    view = SimpleNamespace(name='coronal', family='orthogonal', src_h=4, src_w=5, num_slices=3)
    for slots, expected_outer, expected_memory in ((8,2,2), (0,4,64)):
        outer, _slices, _bytes, memory, _default = runtime.resolve_parent_postprocess_worker_allocation(
            worker_budget=512, views=[view]*64, nrrd_layers_enabled=True,
            interpolation_enabled=True, sam_execution_slots=slots)
        assert (outer, memory) == (expected_outer, expected_memory)


def test_explicit_parent_and_inner_overrides_keep_existing_behavior(monkeypatch):
    allocation(monkeypatch, 8)
    monkeypatch.setenv('YOLO_TTA_PARENT_POSTPROCESS_WORKERS', '3')
    monkeypatch.setenv('YOLO_TTA_PARENT_SLICE_WORKERS', '7')
    view = SimpleNamespace(name='coronal', family='orthogonal', src_h=4, src_w=5, num_slices=3)
    outer, slices, *_ = runtime.resolve_parent_postprocess_worker_allocation(worker_budget=128,
        views=[view]*20, nrrd_layers_enabled=True, interpolation_enabled=True, sam_execution_slots=8)
    assert (outer, slices) == (3, 7)


@pytest.mark.parametrize('slots,feeds_fifth', [(0, False), (8, True)])
def test_real_deferred_parent_queue_feeds_another_view_before_four_scopes_finish(
        tmp_path, monkeypatch, slots, feeds_fifth):
    workers, *_ = allocation(monkeypatch, slots)
    entered = [threading.Event() for _ in range(5)]
    release = threading.Event()
    pool = _ByteAdmissionPool(1000, 'tiny parent transient credit')

    class Task(TinyTask):
        def __init__(self, index):
            super().__init__(tmp_path, str(index), np.zeros((3,4,5), np.uint8))
            self.index = index
            self.admission = pool

        def __call__(self):
            with self.admission.reserve(100, self.view.name):
                entered[self.index].set()
                if self.index < 4:
                    # Existing scopes consume/validate completed GPU results.
                    assert release.wait(5)
                return super().__call__()

    check = ManualExecutor()
    leases = lease_state()
    ready = [False]
    with ThreadPoolExecutor(max_workers=workers) as prepare:
        stage = DeferredSamParentQueue(temp_dir=tmp_path/'temp', output_dir=tmp_path/'output',
            checkpoint_executor=check, prepare_executor=prepare, leases=leases,
            dense_limit=1000, ready=lambda: ready[0])
        tasks = [Task(index) for index in range(5)]
        for task in tasks:
            admit(leases, task, 60)
            stage.defer(task, 60)
        check.finish()
        stage.pump()
        ready[0] = True
        futures, _ = stage.pump()
        try:
            assert len(futures) == 5
            assert all(event.wait(2) for event in entered[:4])
            assert entered[4].wait(.1) == feeds_fifth
            assert pool.in_use == 400 < pool.capacity
            assert sum(leases.postprocess_bytes.values()) == 300 < stage.dense_limit
        finally:
            release.set()
        for future, key in futures.items():
            result = future.result(timeout=5)
            result.final_view_volume_mm._mmap.close()
            result.native_support_mm = result.final_view_volume_mm = None
            leases.complete(key, retain_for_dense_retirement=False)
        assert pool.in_use == 0 and not leases.leases
        stage.close()
        stage.finalize_cleanup()
