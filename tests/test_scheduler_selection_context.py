"""Differential policy coverage for bounded, per-selection evaluation reuse."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import queue
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA.tta_scheduler import D1ParentGroup
from tests._scheduler_selection_reference import reference_select
from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def random_scheduler(seed):
    rng = random.Random(seed)
    state = _state()
    state.gpu_task_queues.update({i:queue.Queue() for i in range(4)})
    state.cpu_task_queues[0] = queue.Queue()
    parents = []
    active_cpu = None
    for p in range(10):
        view = replace(_view(), name=f'parent_{p}', family=rng.choice(
            ('orthogonal','tilted','azimuthal','radial','spherical')),
            src_h=rng.choice((32,2048,4096)), src_w=32)
        parent = ('model',view.name)
        mode = rng.choice(('d1_owner','direct_union','file','hybrid_deferred'))
        hybrid = rng.random()<.4
        if hybrid:
            if mode=='direct_union' and active_cpu is None:
                state.hybrid_view_mode_by_parent[parent]='direct_union'
                active_cpu=parent
            else:
                state.hybrid_view_mode_by_parent[parent]=rng.choice(('unclaimed','d1_owner','retired'))
            if rng.random()<.5:state.hybrid_cpu_reserved_parent_set.add(parent)
        indices=[]
        for _ in range(rng.randint(1,9)):
            tid=len(state.gpu_worker_tasks_by_id)
            kind='tile' if rng.random()<.16 else 'fullframe'
            task=dict(task_id=tid,kind=kind,model_name='model',view=view,
                result_mode=mode if kind=='fullframe' else 'file',
                slice_start=tid*4,slice_count=rng.choice((1,4,16,33)),
                processing_shape=(16,8,8),out_size=32,gpu_eligible=rng.random()>.12,
                hybrid_cpu_eligible_origin=hybrid, result_conf_path='scores' if rng.random()<.2 else None)
            state.gpu_worker_tasks_by_id[tid]=task
            state.gpu_worker_pending_task_ids.append(tid)
            if kind=='fullframe':indices.append(tid)
        state.fullframe_remaining[parent]=len(indices)+rng.randrange(4)
        state.fullframe_task_ids_by_parent[parent]=indices
        if mode=='d1_owner' and indices:parents.append(parent)
        if rng.random()<.4:state.parent_mask_support_by_model['model'][view.name]=True
        state.hybrid_view_tasks_by_backend[parent]=Counter(cpu=rng.randrange(5))
    shuffled=list(state.gpu_worker_pending_task_ids);rng.shuffle(shuffled)
    state.gpu_worker_pending_task_ids.clear();state.gpu_worker_pending_task_ids.extend(shuffled)
    for worker,parent in enumerate(parents[:rng.randrange(min(4,len(parents))+1)]):
        state.d1_owner_by_parent[parent]=worker
        state.d1_active_parent_by_worker[worker]=parent
    if seed%5==0 and parents:
        parent=parents[-1]
        ids=state.fullframe_task_ids_by_parent[parent]
        state.d1_groups_by_parent[parent]=D1ParentGroup(parent,'group',(0,1),0,
            {tid:tid%2 for tid in ids},{0:((0,1),),1:((1,2),)})
        state.d1_owner_by_parent[parent]=0
        state.d1_active_parent_by_worker.update({0:parent,1:parent})
    direct=[p for p,ids in state.fullframe_task_ids_by_parent.items() if ids and
            state.gpu_worker_tasks_by_id[ids[0]]['result_mode']=='direct_union']
    for parent in direct[:rng.randrange(min(2,len(direct))+1)]:
        state.direct_union_inference_views.add(parent)
        state.direct_union_inference_bytes[parent]=1024
        state.direct_union_backing_leases[parent]=SimpleNamespace(phase='inference')
    if seed%3==0:
        state.gpu_worker_tile_dense_result_reservations[-1]=1024
        state.gpu_worker_tile_dense_result_bytes_reserved=1024
    if seed%4==0:
        state.direct_union_postprocess_bytes[('model','completed')]=4000
        state.direct_union_backing_leases[('model','completed')]=SimpleNamespace(phase='postprocess')
    for worker in range(4):
        state.spherical_render_parent_by_worker[worker]=('model',f'parent_{rng.randrange(10)}')
    state.gpu_worker_cpu_assist_inflight_task_ids.update(range(rng.randrange(3)))
    def cost_key(task):
        # Deliberately task-sensitive: a memo must not infer that cost keys depend
        # only on parent, mode or the shared ViewInfo object.
        return task.get('kind'),task.get('task_id',0)%3
    if seed%2:
        state.gpu_worker_seconds_per_frame_ewma.update({('fullframe',0):.019,('fullframe',1):.071,('tile',2):.4})
    scheduler=_scheduler(Path('.'),state=state,input_overrides=dict(v1613_d1_owner_active=bool(seed%3),
        gpu_device_count=4,direct_union_inference_view_limit=2,direct_union_inference_byte_limit=2048,
        direct_union_total_dense_byte_limit=4096,gpu_worker_tile_dense_result_limit=2048),
        operation_overrides=dict(gpu_worker_task_cost_key=cost_key,
            gpu_worker_default_seconds_per_frame=lambda view:.04 if view.family=='spherical' else .06,
            hybrid_gpu_stealback_enabled=lambda:True,hybrid_gpu_stealback_min_cpu_samples=lambda:1,
            hybrid_gpu_stealback_max_fraction=lambda:.5,
            _env_int=lambda name,default: int(seed%2) if name=='YOLO_TTA_GPU_SPHERICAL_LOCALITY' else default))
    return scheduler


class SchedulerSelectionContextTests(unittest.TestCase):
    def test_randomized_reference_ids_worker_order_mutations_and_dynamic_state(self):
        with mock.patch('XTA.confidence_evidence.confidence_evidence_enabled',return_value=False), \
             mock.patch('builtins.print'):
            for seed in range(100):
                left,right=random_scheduler(seed),random_scheduler(seed)
                for step in range(8):
                    candidates=([3,1,0,2] if step%2 else [2,0])
                    preferred=('model',f'parent_{(seed+step)%10}')
                    expected=reference_select(left,preferred,candidates)
                    actual=right.pop_gpu_worker_pending_task_id(preferred,candidates)
                    with self.subTest(seed=seed,step=step):
                        self.assertEqual(actual,expected)
                        self.assertEqual(list(right.state.gpu_worker_pending_task_ids),list(left.state.gpu_worker_pending_task_ids))
                        self.assertEqual(right.state.hybrid_stealback_announced_parents,left.state.hybrid_stealback_announced_parents)
                        for tid in right.state.gpu_worker_tasks_by_id:
                            self.assertEqual(right.state.gpu_worker_tasks_by_id[tid].get('hybrid_gpu_assist_dispatch'),
                                             left.state.gpu_worker_tasks_by_id[tid].get('hybrid_gpu_assist_dispatch'))
                    # Ownership, measured rates and storage pressure can change after
                    # each selection; no value from its context may survive the call.
                    for scheduler in (left,right):
                        scheduler.state.gpu_worker_seconds_per_frame_ewma[('fullframe',step%3)]=.003+step*.011
                        scheduler.state.direct_union_postprocess_bytes[('model','completed')]=4000 if step%2 else 0
                    if actual is None:break

    def test_cost_key_called_once_per_task_and_cold_prior_once_per_shared_view(self):
        scheduler=random_scheduler(2)
        state=scheduler.state
        state.d1_owner_by_parent.clear();state.d1_active_parent_by_worker.clear()
        state.d1_groups_by_parent.clear();state.hybrid_view_mode_by_parent.clear()
        state.hybrid_cpu_reserved_parent_set.clear()
        for task in state.gpu_worker_tasks_by_id.values():
            task.update(kind='fullframe',result_mode='file',hybrid_cpu_eligible_origin=False,gpu_eligible=True)
        # Two leases per parent suppress the special single-pending unlock path.
        for parent,ids in list(state.fullframe_task_ids_by_parent.items()):
            if len(ids)<2:
                for tid in list(state.gpu_worker_pending_task_ids):
                    if state.gpu_worker_tasks_by_id[tid]['view'].name==parent[1]:state.gpu_worker_pending_task_ids.remove(tid)
        key=mock.Mock(side_effect=scheduler.operations.gpu_worker_task_cost_key)
        prior=mock.Mock(side_effect=scheduler.operations.gpu_worker_default_seconds_per_frame)
        scheduler.operations=replace(scheduler.operations,gpu_worker_task_cost_key=key,gpu_worker_default_seconds_per_frame=prior)
        count=len(state.gpu_worker_pending_task_ids)
        scheduler.pop_gpu_worker_pending_task_id(candidate_workers=[0,1,2,3])
        self.assertEqual(key.call_count,count)
        self.assertEqual(prior.call_count,len({id(task['view']) for task in state.gpu_worker_tasks_by_id.values()
                                             if task['task_id'] in state.gpu_worker_pending_task_ids or task.get('hybrid_gpu_assist_dispatch') is not None}))

    def test_same_parent_heterogeneous_shapes_and_policy_errors_match_reference(self):
        for policy in (False,True):
            left,right=random_scheduler(7),random_scheduler(7)
            for scheduler in (left,right):
                state=scheduler.state;state.gpu_worker_pending_task_ids.clear()
                state.hybrid_view_mode_by_parent.clear();state.d1_active_parent_by_worker.clear()
                view=_view()
                for tid,shape in enumerate(((16,8,8),(500,80,80))):
                    state.gpu_worker_tasks_by_id[tid]=dict(task_id=tid,kind='fullframe',model_name='model',view=view,
                        gpu_eligible=True,result_mode='direct_union',processing_shape=shape,slice_count=2,
                        bounded_parent_admission=policy)
                    state.gpu_worker_pending_task_ids.append(tid)
            def run(scheduler,method):
                try:return method(scheduler,candidate_workers=[0,1])
                except Exception as exc:return type(exc),str(exc)
            self.assertEqual(run(left,reference_select),run(right,type(right).pop_gpu_worker_pending_task_id))

    def test_parent_float_sums_and_position_ties_are_not_reassociated(self):
        left,right=random_scheduler(1),random_scheduler(1)
        for scheduler in (left,right):
            state=scheduler.state;state.gpu_worker_pending_task_ids.clear();state.hybrid_view_mode_by_parent.clear()
            state.d1_owner_by_parent.clear();state.d1_active_parent_by_worker.clear();state.d1_groups_by_parent.clear()
            for tid,count in enumerate((10**16,1,1,10**16,2,0)):
                state.gpu_worker_tasks_by_id[tid]=dict(task_id=tid,kind='fullframe',model_name='model',
                    view=replace(_view(),name=f'p{tid//3}'),slice_count=count,gpu_eligible=True,result_mode='file')
                state.gpu_worker_pending_task_ids.append(tid)
            scheduler.operations=replace(scheduler.operations,gpu_worker_task_cost_key=lambda task:('one',),
                                         gpu_worker_default_seconds_per_frame=lambda view:1.)
            state.gpu_worker_seconds_per_frame_ewma.clear()
        self.assertEqual(reference_select(left,candidate_workers=[3,1]),
                         right.pop_gpu_worker_pending_task_id(candidate_workers=[3,1]))

    def test_spherical_file_policy_and_ordinary_task_keep_distinct_locality(self):
        def build():
            state = _state()
            view = replace(_view(), name='spherical_same', family='spherical',
                           src_h=32, src_w=32)
            parent = ('model', view.name)
            state.gpu_task_queues.update({0: queue.Queue(), 1: queue.Queue()})
            state.spherical_render_parent_by_worker[0] = parent
            state.spherical_render_parent_by_worker[1] = ('model', 'other')
            state.fullframe_remaining[parent] = 2
            for task_id, bounded, count in ((0, False, 1), (1, True, 100)):
                state.gpu_worker_tasks_by_id[task_id] = dict(
                    task_id=task_id, kind='fullframe', model_name='model',
                    view=view, result_mode='file', bounded_parent_admission=bounded,
                    processing_shape=(2, 8, 8), slice_count=count, gpu_eligible=True,
                )
                state.gpu_worker_pending_task_ids.append(task_id)
            return _scheduler(Path('.'), state=state,
                              input_overrides=dict(gpu_device_count=2))

        old, current = build(), build()
        with mock.patch('XTA.confidence_evidence.confidence_evidence_enabled',
                        return_value=False):
            expected = reference_select(old, candidate_workers=[0, 1])
            actual = current.pop_gpu_worker_pending_task_id(candidate_workers=[0, 1])
        self.assertEqual(expected, (1, [0]))
        self.assertEqual(actual, expected)


if __name__=='__main__':unittest.main()
