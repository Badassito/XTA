# Frozen pre-optimization selector oracle for ordering and worker subsets.
from __future__ import annotations

def reference_select(self,
    preferred_parent: Optional[Tuple[str, str]] = None,
    candidate_workers: Optional[Sequence[int]] = None,
) -> Optional[Tuple[int, List[int]]]:
    """Pick an admissible GPU task and the worker subset allowed to own it.

    Unreserved hybrid parents are ordinary mandatory CUDA D1 work. Future CPU-reserved
    parents remain protected. The one active direct-union parent enters the candidate
    pool only when its active-view ETA quota has an open CUDA-assist slot.
    """
    pending_ids = [int(v) for v in self.state.gpu_worker_pending_task_ids]
    candidates = [int(v) for v in (candidate_workers or tuple(self.state.gpu_task_queues))]
    feasible_by_id: Dict[int, List[int]] = {}
    eligible: List[Tuple[int, int]] = []
    for position, task_id in enumerate(pending_ids):
        task = self.state.gpu_worker_tasks_by_id[int(task_id)]
        if not bool(task.get('gpu_eligible', self.inputs.gpu_worker_process_active)):
            continue
        if not self.direct_union_task_admissible(task):
            continue
        if not self.tile_dense_result_task_admissible(task):
            continue
        feasible = self.d1_feasible_workers(task, candidates)
        if not feasible:
            continue
        feasible_by_id[int(task_id)] = feasible
        eligible.append((int(position), int(task_id)))
    if not eligible:
        return None

    active_parent = self.active_cpu_shared_parent()
    mandatory_gpu: List[Tuple[int, int]] = []
    active_cpu_assist: List[Tuple[int, int]] = []
    for pair in eligible:
        task = self.state.gpu_worker_tasks_by_id[int(pair[1])]
        if self.hybrid_task_is_active_cpu_assist(task, active_parent):
            active_cpu_assist.append(pair)
        elif self.hybrid_task_is_gpu_mandatory(task):
            mandatory_gpu.append(pair)

    assist_ids: set[int] = set()
    quota = self.hybrid_gpu_stealback_quota(mandatory_gpu, active_cpu_assist)
    assist_slots_open = max(
        0,
        int(quota) - int(len(self.state.gpu_worker_cpu_assist_inflight_task_ids)),
    )
    if assist_slots_open > 0 and active_cpu_assist:
        selected_pool = list(active_cpu_assist)
        assist_ids = {int(task_id) for _position, task_id in active_cpu_assist}
    elif mandatory_gpu:
        selected_pool = list(mandatory_gpu)
    else:
        return None

    selected_pool, feasible_by_id = self.prefer_spherical_locality(
        selected_pool, feasible_by_id, preferred_parent,
    )
    parent_pending_counts: Dict[Tuple[str, str], int] = {}
    for _position, task_id in selected_pool:
        parent_key = self.gpu_worker_fullframe_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)])
        if parent_key is not None:
            parent_pending_counts[parent_key] = int(parent_pending_counts.get(parent_key, 0)) + 1
    unlock_candidates: List[Tuple[int, int, int, int, int]] = []
    for position, task_id in selected_pool:
        task = self.state.gpu_worker_tasks_by_id[int(task_id)]
        parent_key = self.gpu_worker_fullframe_parent_key(task)
        if parent_key is None or int(parent_pending_counts.get(parent_key, 0)) != 1:
            continue
        unlock_candidates.append((
            self.hybrid_gpu_selection_rank(task),
            0 if parent_key == preferred_parent else 1,
            int(self.state.fullframe_remaining.get(parent_key, 2 ** 31 - 1)),
            int(position),
            int(task_id),
        ))
    if unlock_candidates:
        _hybrid_rank, _preferred, _remaining, _position, selected_id = min(unlock_candidates)
    else:
        parent_seconds: Dict[Optional[Tuple[str, str]], float] = {}
        for _position_i, task_id_i in selected_pool:
            task_i = self.state.gpu_worker_tasks_by_id[int(task_id_i)]
            parent_i = self.gpu_worker_fullframe_parent_key(task_i)
            parent_seconds[parent_i] = float(parent_seconds.get(parent_i, 0.0)) + self.gpu_worker_task_seconds(task_i)
        selected_id = min(
            selected_pool,
            key=lambda pair: (
                self.hybrid_gpu_selection_rank(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                0 if self.d1_task_parent_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]) in self.state.d1_owner_by_parent else 1,
                0 if (self.direct_union_task_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]) in self.state.direct_union_inference_views) else 1,
                self.inference_storage_priority_rank(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                -float(parent_seconds.get(self.gpu_worker_fullframe_parent_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]), 0.0)),
                -float(self.gpu_worker_task_seconds(self.state.gpu_worker_tasks_by_id[int(pair[1])])),
                int(pair[0]),
            ),
        )[1]
    selected_task = self.state.gpu_worker_tasks_by_id[int(selected_id)]
    selected_task['hybrid_gpu_assist_dispatch'] = bool(int(selected_id) in assist_ids)
    self.state.gpu_worker_pending_task_ids.remove(int(selected_id))
    return int(selected_id), list(feasible_by_id[int(selected_id)])
