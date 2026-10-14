"""Static SAM repartition preserves the total promise and every dense birth floor."""
import ast
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA import pipeline, runtime, sam_resources
from XTA.interpolation import _ByteAdmissionPool
from XTA.tta_scheduler import TtaScheduler

GIB = runtime.GIB


@pytest.mark.parametrize('inference,dense,transient,parent,cpu', [
    (128,454,454,40,64), (64,128,64,4,32), (128,454,454,200,64),
    (128,454,454,0,64), (128,454,454,40,0)])
def test_partition_conserves_bytes_and_preserves_progress_floors(inference,dense,transient,parent,cpu):
    actual_d,actual_t = runtime.repartition_sam_parent_memory(
        inference*GIB,dense*GIB,transient*GIB,max_parent_bytes=parent*GIB,cpu_input_bytes=cpu*GIB)
    assert actual_d+actual_t == (dense+transient)*GIB
    assert actual_d <= dense*GIB and actual_t >= transient*GIB
    if actual_d < dense*GIB:
        assert actual_d >= (inference+cpu+2*parent)*GIB
        assert actual_d >= 2*inference*GIB
    else:
        assert (actual_d,actual_t) == (dense*GIB,transient*GIB)


@dataclass
class Inputs:
    direct_union_total_dense_byte_limit: int = 454*GIB
    direct_union_inference_byte_limit: int = 128*GIB
    min_conf: float = 0
    dense_tiling_active: bool = False
    nrrd_layers_needed: bool = True
    imgsz: int = 2048


def actual_partition(monkeypatch):
    for name in ('YOLO_TTA_DIRECT_UNION_TOTAL_GIB','YOLO_TTA_PARENT_TRANSIENT_GIB'):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_PREFETCH','1')
    monkeypatch.setattr('XTA.confidence_evidence.confidence_evidence_enabled',lambda:False)
    scheduler = SimpleNamespace(inputs=Inputs(),operations=SimpleNamespace(array_nbytes=runtime.array_nbytes))
    scheduler._direct_union_parent_bytes = lambda task:TtaScheduler._direct_union_parent_bytes(scheduler,task)
    namespace = dict(vars(pipeline))
    namespace.update(scheduler=scheduler,GIB=GIB,replace=replace,_env_flag=runtime._env_flag,
        interpolation_settings=SimpleNamespace(sam_enabled=True),sam_parent_staging=object(),
        policy_settings=SimpleNamespace(enabled=False),bounded_policy_parent_keys=set(),
        dense_tiling_active=False,keep_temp_artifacts=False,gpu_worker_process_active=True,
        cpu_worker_process_active=False,channel_format=SimpleNamespace(kind='gray'),
        args=SimpleNamespace(gpu_batch=1),memfd_workspace_enabled=lambda:True,
        available_anon_work_bytes=lambda:1400*GIB,
        gpu_worker_tasks_by_id={0:dict(kind='fullframe',processing_shape=(2,2,10*GIB))},
        direct_union_backing_leases={},direct_union_inference_bytes={},direct_union_postprocess_bytes={},
        sam_cpu_futures=dict(runtime=None,source=None),sam_context=SimpleNamespace(),
        parent_transient_admission=_ByteAdmissionPool(454*GIB,'test'),
        direct_union_total_dense_byte_limit=454*GIB,direct_union_inference_byte_limit=128*GIB)
    tree = ast.parse(Path(pipeline.__file__).read_text())
    start = next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='cpu_input_prefetch_requested' for target in node.targets))
    block = next(node for node in ast.walk(tree) if isinstance(node,ast.If)
        and ast.unparse(node.test).startswith('interpolation_settings.sam_enabled and sam_parent_staging'))
    budget = next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
        and node.lineno==start.end_lineno+1)
    code = compile(ast.Module(body=[start,budget,block],type_ignores=[]),'<actual static partition>','exec')
    return namespace,lambda:exec(code,namespace)


@pytest.mark.parametrize('confidence,expected_dense',[(False,272),(True,352)])
def test_actual_pipeline_uses_exact_mask_and_confidence_bytes(monkeypatch,confidence,expected_dense):
    ns,run = actual_partition(monkeypatch)
    monkeypatch.setattr('XTA.confidence_evidence.confidence_evidence_enabled',lambda:confidence)
    run()
    assert ns['cpu_input_budget'] == 64*GIB
    assert ns['direct_union_total_dense_byte_limit'] == expected_dense*GIB
    assert ns['scheduler'].inputs.direct_union_total_dense_byte_limit == expected_dense*GIB
    assert ns['parent_transient_admission'].capacity == (908-expected_dense)*GIB
    assert ns['scheduler'].inputs.direct_union_inference_byte_limit == 128*GIB


@pytest.mark.parametrize('reason',[
    'non_sam','cpu_detector','no_gpu','policy','policy_parent','tiles','retained','dense_override','transient_override',
    'empty_inventory','unknown_shape','tile_task','dense_owner','inference_owner','postprocess_owner',
    'runtime_started','source_started','live_transient','funded_startup','startup_hold','startup_credit','image_hold'])
def test_unsupported_or_already_owned_run_keeps_original_partition(monkeypatch,reason):
    ns,run = actual_partition(monkeypatch)
    if reason=='non_sam': ns['interpolation_settings'].sam_enabled=False
    elif reason=='cpu_detector': ns['cpu_worker_process_active']=True
    elif reason=='no_gpu': ns['gpu_worker_process_active']=False
    elif reason=='policy': ns['policy_settings'].enabled=True
    elif reason=='policy_parent': ns['bounded_policy_parent_keys'].add('parent')
    elif reason=='tiles': ns['dense_tiling_active']=True
    elif reason=='retained': ns['keep_temp_artifacts']=True
    elif reason.endswith('_override'):
        monkeypatch.setenv('YOLO_TTA_DIRECT_UNION_TOTAL_GIB' if reason=='dense_override'
            else 'YOLO_TTA_PARENT_TRANSIENT_GIB','454')
    elif reason=='empty_inventory': ns['gpu_worker_tasks_by_id'].clear()
    elif reason=='unknown_shape': ns['gpu_worker_tasks_by_id'][0].pop('processing_shape')
    elif reason=='tile_task': ns['gpu_worker_tasks_by_id'][0]['kind']='tile'
    elif reason in ('dense_owner','inference_owner','postprocess_owner'):
        key=dict(dense_owner='direct_union_backing_leases',inference_owner='direct_union_inference_bytes',
            postprocess_owner='direct_union_postprocess_bytes')[reason]
        ns[key]['parent']=1
    elif reason in ('runtime_started','source_started'): ns['sam_cpu_futures'][reason.split('_')[0]]=object()
    elif reason=='live_transient': ns['parent_transient_admission'].in_use=1
    elif reason=='image_hold': ns['parent_transient_admission']._sam_image_staging_pool=SimpleNamespace(in_use=1)
    else:
        setattr(ns['sam_context'],dict(funded_startup='_startup_fleet_funded',
            startup_hold='_startup_host_reserved',startup_credit='_startup_fleet_credit_bytes')[reason],1)
    run()
    assert ns['direct_union_total_dense_byte_limit'] == ns['parent_transient_admission'].capacity == 454*GIB


def test_repartition_funds_twenty_one_unchanged_parent_grants_and_returns_all_credit():
    dense,transient = runtime.repartition_sam_parent_memory(128*GIB,454*GIB,454*GIB,
        max_parent_bytes=40*GIB,cpu_input_bytes=64*GIB)
    pool = _ByteAdmissionPool(transient,'test')
    base = 10*GIB+GIB//2
    with ExitStack() as stack:
        for index in range(21):
            profile = stack.enter_context(sam_resources.admit_sam_parent_resources(pool,base,str(index),
                worker_count=4,execution_slots=16,base_allowance_bytes=4*GIB,
                headroom_probe=lambda:2000*GIB))
            assert profile.reserved_extra_bytes == 16*GIB
        assert pool.in_use == 21*(base+16*GIB) > 454*GIB
        assert pool.in_use+dense <= 908*GIB
    assert pool.in_use == 0
