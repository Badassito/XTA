"""Controllable scientific endpoints around the actual native detector owner."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import queue
import select
import time
from types import SimpleNamespace

from XTA.geometry import ViewInfo
from XTA.native_scheduler import PreparedWorkScheduler
from XTA.tta_scheduler import TtaScheduler, TtaSchedulerCallbacks, TtaSchedulerInputs, TtaSchedulerState
from XTA.tta_worker_transport import DetectorActorPort


_RUN_IDS = itertools.count(20000)


def detector_view(**overrides):
    fields = dict(name='transverse__tta_a0', num_slices=16, src_h=8, src_w=8,
                  pad_mode='clamp', physical_view_name='transverse', tta_aug_id='a0')
    return ViewInfo(**dict(fields, **overrides))


def detector_state():
    return TtaSchedulerState(
        baseline_union_paths={}, baseline_confmap_paths={}, parent_mask_support_by_model={'model': {}},
        fullframe_remaining={}, direct_union_inference_views=set(), direct_union_postprocess_views=set(),
        direct_union_inference_bytes={}, direct_union_postprocess_bytes={}, direct_union_backing_leases={},
    )


def detector_inputs(root, **overrides):
    fields = dict(
        batch=1, conf=0.25, cpu_batch=1, cpu_precision='fp32', gpu_batch=1, gpu_quantize=None,
        imgsz=8, interpolation_distance=0, min_conf=0.0, min_radius=0.0, dense_tiling_active=False,
        nrrd_layers_needed=False, direct_union_inference_view_limit=2,
        direct_union_inference_byte_limit=1 << 30, direct_union_total_dense_byte_limit=1 << 31,
        gpu_worker_tile_dense_result_limit=1 << 30, gpu_worker_tile_dense_result_task_limit=4,
        gpu_worker_process_active=True, inference_worker_process_active=True, gpu_device_count=1,
        v1613_d1_owner_active=False, gpu_worker_result_dir=Path(root) / 'gpu_worker_results',
        ensure_baseline_workspaces=lambda _model, _view: None, gib=1 << 30,
        hybrid_deferred_result_mode='hybrid_deferred',
    )
    return TtaSchedulerInputs(**dict(fields, **overrides))


class DetectorRun:
    def __init__(self, root, *, state=None, inputs=None):
        self.state = state if state is not None else detector_state()
        self.inputs = inputs if inputs is not None else detector_inputs(root)
        self.state.gpu_result_queue = queue.Queue()
        self.native = PreparedWorkScheduler(next(_RUN_IDS), self.inputs.direct_union_total_dense_byte_limit, 8)
        self.scheduler = TtaScheduler(native=self.native, state=self.state, inputs=self.inputs)
        self.actors = {}
        self.receivers = []
        self.results = []
        self.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(
            handle_fullframe_worker_result=lambda task, stats: self.results.append((task, stats)),
            handle_tile_worker_result=lambda task, stats: self.results.append((task, stats)),
            announce_process_inference_drain_if_complete=lambda: None,
            check_parent_affinity=lambda: None,
        ))

    def worker(self, index, *, kind='gpu', pid=None, window=1):
        actor = DetectorActorPort(self.scheduler.open_worker(kind, index), self.state.gpu_result_queue)
        self.actors[kind, index] = actor
        pid = os.getpid() if pid is None else pid
        self.scheduler.register_worker(kind, index, pid, window=window)
        processes = self.state.gpu_worker_processes if kind == 'gpu' else self.state.cpu_worker_processes
        processes.append(SimpleNamespace(name=f'{kind}-worker-{index}', pid=pid, exitcode=None))
        return actor

    def start_transport(self):
        self.scheduler.configure_result_transport(
            push_drain_active=True, track_thread=lambda thread, _stop: self.receivers.append(thread))

    def declare(self, *tasks, ticket_limit=8):
        self.state.gpu_worker_tasks_by_id.update((task['task_id'], task) for task in tasks)
        self.state.gpu_worker_total_tasks = len(self.state.gpu_worker_tasks_by_id)
        self.scheduler.declare_inventory(ticket_limit=ticket_limit)

    def drain_until(self, predicate, *, timeout=3):
        deadline = time.monotonic() + timeout
        while True:
            self.scheduler.drain_process_inference_results()
            if predicate():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f'native detector condition did not settle: {self.native.snapshot()}')
            self.scheduler.wait_for_one_process_result(remaining)

    def take(self, index, *, kind='gpu', timeout=3):
        actor = self.actors[kind, index]
        if not select.select([actor.stream], [], [], timeout)[0]:
            raise AssertionError(f'no detector command for {(kind, index)}: {self.native.snapshot()}')
        return actor.get()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.state.push_drain_stop.set()
        self.native.close()
        for actor in self.actors.values():
            actor.stream.close()
        for thread in self.receivers:
            thread.join(3)
            assert not thread.is_alive()
