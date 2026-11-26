"""Scientific producer and consumer ports for the native prepared detector scheduler."""
from __future__ import annotations

import math
import os
import queue
import socket
import threading
import time
import weakref
from collections import Counter, defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .backprojection import GpuInferenceAssetsRetired
from .geometry import ViewInfo
from .native_detector_handles import detector_handle_submission, planned_detector_source_pins
from .native_scheduler import _MAX_PAYLOAD
from .runtime import runtime_telemetry, runtime_trace_event

@dataclass(frozen=True)
class TtaSchedulerInputs:
    """Run-constant process-inference configuration and synchronous callbacks."""

    batch: int
    conf: float
    cpu_batch: int
    cpu_precision: str
    gpu_batch: int
    gpu_quantize: object
    imgsz: int
    interpolation_distance: int
    min_conf: float
    min_radius: float
    dense_tiling_active: bool
    nrrd_layers_needed: bool
    direct_union_inference_view_limit: int
    direct_union_inference_byte_limit: int
    direct_union_total_dense_byte_limit: int
    gpu_worker_tile_dense_result_limit: int
    gpu_worker_tile_dense_result_task_limit: int
    gpu_worker_process_active: bool
    inference_worker_process_active: bool
    gpu_device_count: int
    v1613_d1_owner_active: bool
    gpu_worker_result_dir: Path
    ensure_baseline_workspaces: Callable[[str, ViewInfo], None]
    gib: int
    hybrid_deferred_result_mode: str
    request_baseline_workspaces: Optional[Callable[[Dict[str, object]], bool]] = None
    drain_baseline_workspaces: Optional[Callable[[], None]] = None
    d1_owner_groups_requested: bool = False
    d1_owner_group_size_limit: int = 8

@dataclass(frozen=True)
class TtaSchedulerCallbacks:
    """Main-thread callbacks into the still-inline view/tile completion owner."""

    handle_fullframe_worker_result: Callable[[Dict[str, object], Dict[str, object]], None]
    handle_tile_worker_result: Callable[[Dict[str, object], Dict[str, object]], None]
    announce_process_inference_drain_if_complete: Callable[[], None]
    check_parent_affinity: Callable[[], None]
    gpu_inference_assets_retired: Optional[Callable[[GpuInferenceAssetsRetired], None]] = None

@dataclass
class TtaSchedulerState:
    """Scientific workspace and result registries, plus read-only run reporting."""
    baseline_union_paths: Dict[Tuple[str, str], Path]
    baseline_confmap_paths: Dict[Tuple[str, str], Optional[Path]]
    parent_mask_support_by_model: Dict[str, Dict[str, object]]
    fullframe_remaining: Dict[Tuple[str, str], int]
    direct_union_inference_views: set[Tuple[str, str]]
    direct_union_postprocess_views: set[Tuple[str, str]]
    direct_union_inference_bytes: Dict[Tuple[str, str], int]
    direct_union_postprocess_bytes: Dict[Tuple[str, str], int]
    direct_union_backing_leases: Dict[Tuple[str, str], object]
    direct_union_admission_group_by_parent: Dict[Tuple[str, str], Tuple[str, str]] = field(
        default_factory=dict
    )
    cpu_task_queues: Dict[int, object] = field(default_factory=dict)
    gpu_task_queues: Dict[int, object] = field(default_factory=dict)
    cpu_worker_processes: List[object] = field(default_factory=list)
    gpu_worker_processes: List[object] = field(default_factory=list)
    gpu_worker_ready_details_by_id: Dict[int, Dict[str, object]] = field(default_factory=dict)
    cpu_worker_dispatched_by_id: Dict[int, int] = field(default_factory=dict)
    cpu_worker_results_by_id: Dict[int, int] = field(default_factory=dict)
    cpu_worker_ready_details_by_id: Dict[int, Dict[str, object]] = field(
        default_factory=dict
    )
    gpu_result_queue: object = None
    gpu_worker_tasks_by_id: Dict[int, Dict[str, object]] = field(default_factory=dict)
    gpu_worker_results_collected: int = 0
    gpu_worker_total_tasks: int = 0
    gpu_worker_dispatched_tasks: int = 0
    gpu_worker_dispatched_by_id: Dict[int, int] = field(default_factory=dict)
    gpu_worker_results_by_id: Dict[int, int] = field(default_factory=dict)
    gpu_worker_next_dynamic_task_id: int = 0
    gpu_worker_tile_dense_result_bytes_reserved: int = 0
    gpu_worker_tile_dense_result_memfd_bytes_reserved: int = 0
    gpu_worker_tile_dense_result_max_retention_seconds: float = 0.0
    gpu_worker_tile_dense_result_reservations: Dict[int, int] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_memfd_reservations: Dict[int, int] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_reserved_at: Dict[int, float] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_workspaces: Dict[
        int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    ] = field(default_factory=dict)
    gpu_worker_tile_task_id_by_key: Dict[Tuple[str, str, str], int] = field(
        default_factory=dict
    )
    gpu_worker_tile_pending_result_ids_by_task: Dict[int, set[str]] = field(
        default_factory=dict
    )
    gpu_inference_drained_at: Optional[float] = None
    gpu_inference_drain_announced: bool = False
    gpu_inference_asset_release_requested: bool = False
    gpu_inference_asset_release_pending_by_worker: Dict[int, int] = field(default_factory=dict)
    gpu_inference_asset_release_results_by_worker: Dict[int, Dict[str, object]] = field(default_factory=dict)
    gpu_inference_asset_release_epoch: Optional[int] = None
    gpu_inference_asset_release_proofs_by_worker: Dict[int, GpuInferenceAssetsRetired] = field(default_factory=dict)
    d1_layer_ref_by_parent: Dict[Tuple[str, str], object] = field(default_factory=dict)
    d1_view_shadow_path_by_parent: Dict[Tuple[str, str], Path] = field(
        default_factory=dict
    )
    hybrid_view_mode_by_parent: Dict[Tuple[str, str], str] = field(default_factory=dict)
    fullframe_task_ids_by_parent: Dict[Tuple[str, str], List[int]] = field(
        default_factory=dict
    )
    hybrid_cpu_reserved_parents: List[Tuple[str, str]] = field(default_factory=list)
    hybrid_cpu_reserved_parent_set: set[Tuple[str, str]] = field(default_factory=set)
    hybrid_cpu_reservation_rank_by_parent: Dict[Tuple[str, str], int] = field(
        default_factory=dict
    )
    gpu_worker_cpu_assist_inflight_task_ids: set[int] = field(default_factory=set)
    gpu_worker_cpu_assist_completed_task_ids: set[int] = field(default_factory=set)
    hybrid_stealback_announced_parents: set[Tuple[str, str]] = field(
        default_factory=set
    )
    hybrid_cpu_idle_reason_last: str = ""
    gpu_frames_completed_total: int = 0
    cpu_frames_completed_total: int = 0
    hybrid_gpu_frames_completed_total: int = 0
    hybrid_cpu_frames_completed_total: int = 0
    hybrid_view_frames_by_backend: Dict[Tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    hybrid_view_tasks_by_backend: Dict[Tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    gpu_worker_seed_task_count: int = 0
    result_transport_configured: bool = False
    push_drain_active: bool = False
    scheduler_wake: threading.Event = field(default_factory=threading.Event)
    push_drain_stop: threading.Event = field(default_factory=threading.Event)
    pushed_worker_results: deque[Dict[str, object]] = field(default_factory=deque)
    wake_hooked_futures: weakref.WeakSet[Future] = field(default_factory=weakref.WeakSet)


@dataclass(frozen=True)
class TtaSchedulerArtifacts:
    """Identity-preserving artifact registries handed to the post-scheduler tail."""

    d1_layer_ref_by_parent: Mapping[Tuple[str, str], object]
    d1_view_shadow_path_by_parent: Mapping[Tuple[str, str], Path]

@dataclass(frozen=True)
class TtaSchedulerResult:
    """Immutable process-scheduler telemetry and artifact handoff."""

    gpu_inference_drained_at: Optional[float]
    gpu_worker_results_collected: int
    gpu_worker_total_tasks: int
    gpu_frames_completed_total: int
    cpu_frames_completed_total: int
    hybrid_gpu_frames_completed_total: int
    hybrid_cpu_frames_completed_total: int
    gpu_dispatched_by_worker: Mapping[int, int]
    gpu_completed_by_worker: Mapping[int, int]
    cpu_dispatched_by_worker: Mapping[int, int]
    cpu_completed_by_worker: Mapping[int, int]
    cpu_worker_ready_details: Mapping[int, Mapping[str, object]]
    hybrid_view_mode_by_parent: Mapping[Tuple[str, str], str]
    hybrid_view_frames_by_backend: Mapping[Tuple[str, str], Mapping[str, int]]
    hybrid_view_tasks_by_backend: Mapping[Tuple[str, str], Mapping[str, int]]
    quiescence_issues: Mapping[str, object]
    artifacts: TtaSchedulerArtifacts

@dataclass
class _ParentInputs:
    key: tuple
    task_ids: tuple
    members: tuple
    dense_bytes: int
    exclusive_kind: int
    generation: int = 0
    publication_received: bool = False
    assembled: int = 0
    publications: list = field(default_factory=list)
    owners: list = field(default_factory=list)
    host_request: object = None
    host_last_use: bool = False
    host_adopted: bool = False
    declared_generation: int = 0
    admission_reserve: int = 0
    result_bytes: int = 0


class _WorkerCommands:
    def __init__(self, native, worker_id):
        self.native = native
        self.worker_id = worker_id

    def put(self, message):
        self.native.send_control(self.worker_id, bytes(ForkingPickler.dumps(message)),
                                 expect_reply=message is not None, stop=message is None)


class TtaScheduler:
    """Translate scientific work and actual ownership witnesses at the native boundary."""

    def __init__(self, *, native, state, inputs):
        self.native = native
        self.state = state
        self.inputs = inputs
        self.cpu_input_prefetch = None
        self._callbacks = None
        self._streams = {}
        self._parents = {}
        self._published = set()
        self._parent_by_key = {}
        self._host_requests = {}
        self._worker_by_id = {}
        self._worker_pids = {}
        self._next_parent_id = 1
        self._preparing = set()
        self._events = queue.SimpleQueue()
        self._started = False
        self._cpu_input_partition = 0
        self._control_seq = 0
        self._fault = None
        self._handling = False

    def open_worker(self, kind, worker_id):
        source, actor = socket.socketpair()
        self._streams[kind, worker_id] = source
        # CPU and CUDA indices have separate namespaces in scientific metadata.
        wire_id = int(worker_id) if kind == 'gpu' else 2**32 + int(worker_id)
        self._worker_by_id[wire_id] = kind, worker_id
        queues = self.state.gpu_task_queues if kind == 'gpu' else self.state.cpu_task_queues
        queues[worker_id] = _WorkerCommands(self.native, wire_id)
        return actor

    def register_worker(self, kind, worker_id, pid, window=1):
        source = self._streams.pop((kind, worker_id))
        with source:
            commands = (self.state.gpu_task_queues if kind == 'gpu' else self.state.cpu_task_queues)[worker_id]
            self.native.register_worker(commands.worker_id, 1, pid, 2 if kind == 'gpu' else 1,
                                        window, source, worker_id if kind == 'gpu' else -1)
            self._worker_pids[(kind, worker_id)] = pid

    def bind_result_callbacks(self, callbacks):
        self._callbacks = callbacks

    def configure_result_transport(self, *, push_drain_active, track_thread):
        if self.state.result_transport_configured:
            raise RuntimeError('Detector actor transport configured twice')
        self.state.result_transport_configured = True
        self.state.push_drain_active = True

        def receive(kind, take):
            try:
                while not self.state.push_drain_stop.is_set():
                    event = take(0.25)
                    if event is not None:
                        self._events.put((kind, event))
                        self.wake_scheduler()
            except BaseException as exc:
                if not self.state.push_drain_stop.is_set():
                    self._events.put(('error', exc))
                    self.wake_scheduler()

        ports = [('prepared', self.native.wait_preparation),
                 ('publication', self.native.wait_publication),
                 ('worker', self.native.wait_event)]
        if self.state.gpu_result_queue is not None:
            def startup_message(timeout):
                try:
                    return self.state.gpu_result_queue.get(timeout=timeout)
                except queue.Empty:
                    return None
            ports.append(('startup', startup_message))
        for kind, take in ports:
            thread = threading.Thread(target=receive, args=(kind, take),
                                      name=f'detector-{kind}-port', daemon=True)
            track_thread(thread, self.state.push_drain_stop)
            thread.start()

    def _direct_union_parent_bytes(self, task):
        from .confidence_evidence import confidence_evidence_enabled
        from .geometry import view_processing_volume_shape
        shape = task.get('processing_shape')
        if shape is None:
            shape = view_processing_volume_shape(task['view'], int(task.get('out_size', self.inputs.imgsz)))
        if (len(shape) != 3 or any(isinstance(value, (bool, np.bool_))
                                or not isinstance(value, (int, np.integer)) or value <= 0 for value in shape)):
            raise ValueError(f'Detector parent requires a positive three-dimensional shape, got {shape}')
        planes = 1 + int(self.inputs.min_conf > 0.0 or confidence_evidence_enabled())
        return math.prod(int(value) for value in shape) * planes

    def _validate_parent_ranges(self, key, task_ids):
        """Every declared scientific job must cover its intended output exactly once."""
        ranges = defaultdict(list)
        modes = set()
        for task_id in task_ids:
            task = self.state.gpu_worker_tasks_by_id[task_id]
            modes.add(task['result_mode'])
            for member in self._direct_union_admission_tasks(task):
                if 'union_num_slices' not in member:
                    continue  # Upstream may submit a standalone prepared range.
                start, length = member['slice_start'], member['slice_count']
                extent = member['union_num_slices']
                if (type(start) is not int or type(length) is not int or type(extent) is not int
                        or start < 0 or length <= 0 or start + length > extent or extent != member['view'].num_slices):
                    raise ValueError(f'Invalid detector range for {key}, task {task_id}')
                job = (self.gpu_worker_fullframe_parent_key(member), member['job_id'])
                ranges[job].append((start, start + length, extent))
        if len(modes) != 1:
            raise ValueError(f'Detector parent {key} changed output ownership between ranges')
        for job, windows in ranges.items():
            cursor = 0
            for start, stop, extent in sorted(windows):
                if start != cursor:
                    raise ValueError(f'Detector job {job} has a gap or duplicate near slice {cursor}')
                cursor = stop
            if cursor != extent:
                raise ValueError(f'Detector job {job} has incomplete slice coverage {cursor}/{extent}')

    def _parent_result_bytes(self, task_ids):
        """Prepay file results and wrapped seam slots that may outlive their kernels."""
        from .geometry import azimuthal_batch_padding_count
        from .confidence_evidence import confidence_evidence_enabled
        size = 0
        for task_id in task_ids:
            task = self.state.gpu_worker_tasks_by_id[task_id]
            for member in self._direct_union_admission_tasks(task):
                shape = member.get('processing_shape')
                if not shape:
                    continue
                count = member['slice_count']
                batch = member.get('prediction_batch', self.inputs.gpu_batch)
                slots = azimuthal_batch_padding_count(member['view'], count, batch,
                                                       slice_offset=member.get('slice_start', 0))
                if member['result_mode'] == 'file' and member.get('result_mask_path'):
                    slots += count
                if slots:
                    scores = bool(member.get('result_conf_path') or (member['result_mode'] == 'direct_union'
                                  and (self.inputs.min_conf > 0 or confidence_evidence_enabled())))
                    size += slots * math.prod(shape[-2:]) * (1 + int(scores))
        return size

    def _immutable_descriptors(self, parent, capture_sources):
        """Work with ready sources and no pending producer can enter the plan at declaration."""
        if parent.dense_bytes:
            return None
        from .yolo_cpu_input import cpu_input_eligible
        packets = []
        for task_id in parent.task_ids:
            task = self.state.gpu_worker_tasks_by_id[task_id]
            if self.cpu_input_prefetch is not None and cpu_input_eligible(task):
                return None
            with detector_handle_submission(task) as (descriptor, handles):
                sources = capture_sources(descriptor, handles)
                if sources is None:
                    return None
                allowed = int(bool(task.get('cpu_eligible'))) | (2 * int(bool(task.get('gpu_eligible'))))
                payload = bytes(ForkingPickler.dumps(descriptor))
                if len(payload) > _MAX_PAYLOAD:
                    raise ValueError(f'Prepared descriptor for task {task_id} exceeds the native wire limit')
                packets.append((task_id, payload, allowed, sources))
        return packets

    def declare_inventory(self, *, ticket_limit):
        """Validate the entire known route before native can fund its first parent."""
        if self._started:
            raise RuntimeError('Detector task inventory already declared')
        if self._streams:
            raise ValueError('Detector inventory contains an unregistered worker endpoint')
        groups = defaultdict(list)
        for task_id, task in self.state.gpu_worker_tasks_by_id.items():
            if (type(task_id) is not int or not 0 <= task_id < 2**63
                    or type(task.get('task_id')) is not int or task_id != task['task_id']):
                raise ValueError('Detector task inventory key and descriptor identity disagree')
            if task['kind'] != 'fullframe' or task['result_mode'] not in ('direct_union', 'd1_owner', 'file'):
                raise ValueError(f'Unsupported native detector task route {task["kind"]}/{task["result_mode"]}')
            if task['result_mode'] == 'd1_owner' and not (self.state.gpu_task_queues and task.get('gpu_eligible')):
                raise ValueError('D1 parent ownership requires a CUDA model actor')
            if not ((task.get('gpu_eligible') and self.state.gpu_task_queues)
                    or (task.get('cpu_eligible') and self.state.cpu_task_queues)):
                raise ValueError(f'No selected actor can execute detector task {task_id}')
            groups[self.gpu_worker_fullframe_parent_key(task)].append(task_id)
        from .scheduler_preflight import preflight_tta_scheduler
        preflight_tta_scheduler(gpu_device_count=len(self.state.gpu_task_queues),
                                cpu_enabled=bool(self.state.cpu_task_queues),
                                tile_configs=self.inputs.dense_tiling_active,
                                d1_owner_groups=self.inputs.d1_owner_groups_requested)
        cpu_reserve = self.cpu_input_prefetch.max_bytes if self.cpu_input_prefetch is not None else 0
        if self.cpu_input_prefetch is not None:
            largest = max((self.direct_union_task_bytes(self.state.gpu_worker_tasks_by_id[ids[0]])
                           for ids in groups.values()
                           if self.direct_union_task_key(self.state.gpu_worker_tasks_by_id[ids[0]]) is not None),
                          default=0)
            cpu_reserve = min(cpu_reserve, max(0, self.inputs.direct_union_total_dense_byte_limit - largest))
            if self.cpu_input_prefetch.try_yield_capacity(cpu_reserve) is None:
                raise ValueError('CPU input buffers exceed their funded partition before detector preparation')
        total = self.inputs.direct_union_total_dense_byte_limit - cpu_reserve
        self._cpu_input_partition = cpu_reserve
        inference = min(total, self.inputs.direct_union_inference_byte_limit)
        self.native.configure_capacity(total, ticket_limit, inference, self.inputs.direct_union_inference_view_limit,
                                       pressure_byte_capacity=self.inputs.direct_union_total_dense_byte_limit)
        for parent_id, (key, ids) in enumerate(groups.items(), 1):
            self._validate_parent_ranges(key, ids)
            first = self.state.gpu_worker_tasks_by_id[ids[0]]
            members = self._direct_union_admission_tasks(first)
            keys = tuple(self.gpu_worker_fullframe_parent_key(member) for member in members)
            dense_bytes = self.direct_union_task_bytes(first) if self.direct_union_task_key(first) is not None else 0
            results = self._parent_result_bytes(ids)
            if dense_bytes + results > total:
                raise ValueError(f'Native detector parent {key} needs {dense_bytes + results} bytes, '
                                 f'available inference/total budget is {inference}/{total}')
            if any(parent_key in self._parent_by_key for parent_key in keys):
                raise ValueError(f'Overlapping native detector parent groups for {key}')
            parent = _ParentInputs(key, tuple(ids), keys, dense_bytes,
                                   2 if first['result_mode'] == 'd1_owner' else 0, result_bytes=results)
            self._parents[parent_id] = parent
            self._next_parent_id = parent_id + 1
            self._parent_by_key.update((member, parent_id) for member in keys)
        self._started = True
        with planned_detector_source_pins() as capture_sources:
            packets_by_parent = {parent_id: self._immutable_descriptors(parent, capture_sources)
                                 for parent_id, parent in self._parents.items()}
            for parent_id, parent in self._parents.items():
                packets = packets_by_parent.pop(parent_id)
                generation = self.native.declare_parent(
                    parent_id, len(parent.task_ids), parent.dense_bytes + parent.result_bytes,
                    exclusive_worker_kind=parent.exclusive_kind, retirement_pressure_bytes=parent.dense_bytes,
                    descriptor_only=packets is not None)
                if packets is not None:
                    parent.generation = generation
                    for task_id, payload, allowed, handles in packets:
                        self.native.submit_planned(task_id, parent_id, payload, allowed, handles=handles)

    def try_resize_dense_capacity(self, dense_bytes):
        """Transfer only budget the native owner can actually surrender."""
        input_bytes = self._cpu_input_partition
        if self.cpu_input_prefetch is not None and self.cpu_input_prefetch.reserved_bytes == 0:
            input_bytes = 0
        capacity = dense_bytes - input_bytes
        if capacity < 0:
            return False
        if not self.native.try_resize_capacity(capacity, min(capacity, self.inputs.direct_union_inference_byte_limit),
                                               pressure_byte_capacity=dense_bytes):
            return False
        self._cpu_input_partition = input_bytes
        return True

    def request_host_workspace(self, key, nbytes, *, namespace='sweep', allow_oversize=False, admission_reserve=0):
        """Request a grant for hydrating previously sealed scientific input."""
        request = (namespace, key)
        parent_id = self._host_requests.get(request)
        if parent_id is None:
            parent_id = self._next_parent_id
            self._next_parent_id += 1
            declared = self.native.declare_parent(parent_id, 0, nbytes, allow_oversize_host=allow_oversize,
                                                  admission_reserve=admission_reserve)
            self._host_requests[request] = parent_id
            self._parents[parent_id] = _ParentInputs(key, (), (key,), nbytes, 0, host_request=request,
                                                     declared_generation=declared, admission_reserve=admission_reserve)
            return False
        parent = self._parents[parent_id]
        if parent.dense_bytes != nbytes or parent.owners:
            raise RuntimeError(f'Host workspace request changed after declaration for {namespace}/{key}')
        if not parent.generation and admission_reserve != parent.admission_reserve:
            if self.native.update_host_reserve(parent_id, parent.declared_generation, admission_reserve):
                parent.admission_reserve = admission_reserve
        return parent.generation != 0 and not parent.task_ids

    def host_workspace_bound(self, key, *, namespace, owner):
        """Bind the exact future owner before a host allocation actor starts."""
        parent = self._parents[self._host_requests[namespace, key]]
        if (not parent.generation or parent.owners or parent.dense_bytes != owner.nbytes
                or self.state.direct_union_backing_leases.get(key) is not owner):
            raise RuntimeError(f'Host workspace cannot bind an unfunded or changed owner for {namespace}/{key}')
        parent.owners = [(key, owner)]

    def host_workspace_adopted(self, key, *, namespace='sweep', owner=None):
        parent_id = self._host_requests[namespace, key]
        parent = self._parents[parent_id]
        if parent.host_adopted or not parent.generation:
            raise RuntimeError(f'Duplicate or unfunded host workspace adoption for {namespace}/{key}')
        if not parent.owners:
            current = self.state.direct_union_backing_leases[key]
            self.host_workspace_bound(key, namespace=namespace, owner=current if owner is None else owner)
        if owner is not None and owner is not parent.owners[0][1]:
            raise RuntimeError(f'Stale workspace owner for {namespace}/{key}')
        self.native.host_prepared(parent_id, parent.generation)
        parent.host_adopted = True

    def host_workspace_failed(self, key, error, *, namespace='sweep'):
        if (namespace, key) not in self._host_requests:
            raise RuntimeError(f'Unrequested host workspace failure for {namespace}/{key}')
        self._events.put(('error', error))
        self.native.cancel(f'Host workspace {namespace}/{key} failed: {error}')
        self.wake_scheduler()

    def host_workspace_retired(self, key, *, namespace, owner):
        """Accept the staging actor's last-use fact, independent of publication arrival."""
        parent_id = self._host_requests[namespace, key]
        parent = self._parents[parent_id]
        if (len(parent.owners) != 1 or parent.owners[0][1] is not owner
                or self.state.direct_union_backing_leases.get(key) is owner
                or owner.phase != 'released' or owner.owner_count != 0):
            raise RuntimeError(f'Unsettled or stale workspace retirement for {namespace}/{key}')
        parent.host_last_use = True
        self._retire_consumed()
        return parent_id not in self._parents

    def service_preparation(self, *_):
        """Encode already-funded scientific descriptors after workspace installation."""
        if not self._started:
            return
        if self.inputs.drain_baseline_workspaces is not None:
            self.inputs.drain_baseline_workspaces()
        for parent_id in sorted(self._preparing):
            parent = self._parents[parent_id]
            first = self.state.gpu_worker_tasks_by_id[parent.task_ids[0]]
            if parent.dense_bytes:
                if not self.inputs.request_baseline_workspaces(first):
                    continue
                parent.owners = [(key, self.state.direct_union_backing_leases[key]) for key in parent.members]
                if sum(owner.nbytes for _, owner in parent.owners) != parent.dense_bytes:
                    raise RuntimeError(f'Detector parent {parent.key} allocated outside its native byte grant')
            for task_id in parent.task_ids:
                task = self.state.gpu_worker_tasks_by_id[task_id]
                for member in self._direct_union_admission_tasks(task):
                    if member['result_mode'] == 'direct_union':
                        key = self.gpu_worker_fullframe_parent_key(member)
                        member['result_mask_path'] = str(self.state.baseline_union_paths[key])
                        conf = self.state.baseline_confmap_paths.get(key)
                        member['result_conf_path'] = str(conf) if conf is not None else None
                descriptor = dict(task)
                if self.cpu_input_prefetch is not None:
                    self.cpu_input_prefetch.attach_ready(descriptor)
                with detector_handle_submission(descriptor) as (packet, handles):
                    payload = bytes(ForkingPickler.dumps(packet))
                    kinds = int(bool(task.get('cpu_eligible'))) | (2 * int(bool(task.get('gpu_eligible'))))
                    self.native.submit_prepared(task_id, parent_id, payload, kinds, handles=handles)
            self._preparing.remove(parent_id)
        self._retire_consumed()

    def track_publication(self, parent_key, completion):
        if not isinstance(completion, Future):
            raise TypeError('Sweep publisher must return a last-use Future')
        self._parents[self._parent_by_key[parent_key]].publications.append(completion)

        def settled(future):
            if future.cancelled():
                error = RuntimeError(f'Detector publication for {parent_key} was cancelled')
            else:
                error = future.exception()
            if error is not None:
                self._events.put(('error', error))
                self.native.cancel(f'Publication for {parent_key} failed: {error}')
            self.wake_scheduler()

        completion.add_done_callback(settled)

    def _retire_consumed(self):
        for parent_id in tuple(self._published):
            parent = self._parents[parent_id]
            if not parent.publication_received or parent.assembled != len(parent.task_ids):
                continue
            for future in parent.publications:
                if future.done():
                    future.result()
            live = [owner for key, owner in parent.owners if
                    self.state.direct_union_backing_leases.get(key) is owner
                    or owner.phase != 'released' or owner.owner_count != 0]
            if not live and parent.host_request is not None and parent.host_request[0] != 'sweep' and not parent.host_last_use:
                continue
            remaining = sum(owner.nbytes for owner in live)
            if remaining < parent.dense_bytes + parent.result_bytes:
                self.native.set_parent_remaining_bytes(parent_id, parent.generation, remaining)
                parent.dense_bytes = remaining
                parent.result_bytes = 0
            if any(not future.done() for future in parent.publications):
                continue
            if live:
                continue
            self.native.retire_parent(parent_id, parent.generation)
            if parent.host_request is not None:
                del self._host_requests[parent.host_request]
            else:
                for key in parent.members:
                    del self._parent_by_key[key]
            del self._parents[parent_id]
            self._published.remove(parent_id)

    def _handle_event(self, channel, event):
        if channel == 'error':
            raise event
        if channel in ('prepared', 'publication'):
            parent_id, generation = event
            parent = self._parents[parent_id]
            if channel == 'prepared':
                parent.generation = generation
                if parent.task_ids:
                    self._preparing.add(parent_id)
            elif generation != parent.generation:
                raise RuntimeError(f'Unexpected publication generation for detector parent {parent_id}')
            else:
                parent.publication_received = True
                self._published.add(parent_id)
            return
        if channel == 'startup':
            if event['type'] not in ('ready', 'fatal'):
                raise RuntimeError('Scientific worker result bypassed native receipt validation')
            kind = event.get('worker_kind', 'gpu')
            worker = event['cpu_index' if kind == 'cpu' else 'gpu_index']
            if event['type'] == 'fatal':
                raise RuntimeError(f'{kind} worker {worker} failed: {event.get("error")}\n{event.get("traceback")}')
            records = (self.state.cpu_worker_ready_details_by_id if kind == 'cpu' else
                       self.state.gpu_worker_ready_details_by_id)
            if (type(worker) is not int or type(event.get('pid')) is not int
                    or self._worker_pids.get((kind, worker)) != event['pid']):
                raise RuntimeError('Detector ready receipt has an unregistered worker identity')
            uuid = event.get('cuda_device_uuid')
            records[worker] = dict(event, worker_pid=event['pid'],
                                   cuda_device_uuid=uuid if isinstance(uuid, str) and uuid else None)
            if kind == 'gpu':
                runtime_telemetry().gauge('gpu.device_identity', {
                    'parent_pid': os.getpid(), 'sample_monotonic_ns': time.monotonic_ns(),
                    'source': 'managed_worker_ready',
                    'devices': {f'cuda:{k}': {'worker_pid': v['worker_pid'],
                        'cuda_device_uuid': v.get('cuda_device_uuid')} for k, v in records.items()}})
            return
        kind, lease, payload = event
        if not payload:
            return
        message = ForkingPickler.loads(payload)
        backend, worker = self._worker_by_id[lease.worker_id]
        if (message.get('worker_kind', 'gpu') != backend
                or message.get('cpu_index' if backend == 'cpu' else 'gpu_index') != worker):
            raise RuntimeError('Scientific result backend conflicts with its authenticated native channel')
        if kind == 'control':
            if message.get('worker_pid') != lease.pid:
                raise RuntimeError('Unexpected actor PID in native detector control response')
            if message.get('type') == 'inference_assets_released':
                self._process_inference_asset_release_result(message)
            elif message.get('type') == 'aux_result':
                from .runtime import gpu_worker_aux_interpolation_pool
                pool = gpu_worker_aux_interpolation_pool()
                if pool is None:
                    raise RuntimeError('Unrequested detector auxiliary result')
                pool.complete(message['task_id'], worker, message['ok'], message.get('stats'), message.get('error',''))
                self._offer_auxiliary_workers()
            else:
                raise RuntimeError(f'Unrecognized detector control response {message.get("type")}')
            return
        if kind != 'final' or message['type'] != 'result' or message['task_id'] != lease.task_id or message['ok'] is not True:
            raise RuntimeError('Unrecognized detector science receipt')
        task = self.state.gpu_worker_tasks_by_id[lease.task_id]
        trace_device = f'{"cuda" if backend == "gpu" else "cpu"}:{worker}'
        runtime_trace_event('scheduler_result_received', task=task, device=trace_device,
                            task_id=lease.task_id, generation=lease.generation)
        if self.cpu_input_prefetch is not None:
            self.cpu_input_prefetch.retire(task)
        counts = self.state.cpu_worker_results_by_id if backend == 'cpu' else self.state.gpu_worker_results_by_id
        issued = self.state.cpu_worker_dispatched_by_id if backend == 'cpu' else self.state.gpu_worker_dispatched_by_id
        counts[worker] = counts.get(worker, 0) + 1
        issued[worker] = counts[worker]
        self.state.gpu_worker_results_collected += 1
        if backend == 'cpu':
            self.state.cpu_frames_completed_total += task['slice_count']
        else:
            self.state.gpu_frames_completed_total += task['slice_count']
        self._callbacks.handle_fullframe_worker_result(task, message['stats'])
        runtime_trace_event('scheduler_result_handled', task=task, device=trace_device,
                            task_id=lease.task_id, generation=lease.generation)
        parent = self._parents[lease.parent_id]
        parent.assembled += 1
        if parent.assembled == len(parent.task_ids):
            self.native.adopt_postprocess(lease.parent_id, parent.generation)
        self._callbacks.announce_process_inference_drain_if_complete()

    def drain_process_inference_results(self, *, max_messages=32):
        if self._handling:
            return 0
        count = 0
        self._handling = True
        try:
            while count < max_messages:
                try:
                    channel, event = self._events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(channel, event)
                count += 1
            self.service_preparation()
            return count
        except BaseException as exc:
            self.native.cancel(str(exc))
            raise
        finally:
            self._handling = False

    def has_pending_process_results(self):
        return not self._events.empty()

    def hook_scheduler_wake(self, futures):
        for future in futures:
            if future not in self.state.wake_hooked_futures:
                self.state.wake_hooked_futures.add(future)
                future.add_done_callback(self.wake_scheduler)

    def wake_scheduler(self, _future=None):
        self.state.scheduler_wake.set()

    notify_gpu_stage_admission_change = wake_scheduler

    def wait_for_one_process_result(self, timeout):
        if self._events.empty():
            self.state.scheduler_wake.wait(timeout)
            self.state.scheduler_wake.clear()

    def process_inference_outstanding(self):
        return (self.state.gpu_worker_results_collected < self.state.gpu_worker_total_tasks
                or bool(self.state.gpu_inference_asset_release_pending_by_worker))

    def publication_outstanding(self):
        return bool(self._parents)

    def check_inference_workers_alive(self):
        self._callbacks.check_parent_affinity()
        if self.process_inference_outstanding():
            for process in self.state.gpu_worker_processes + self.state.cpu_worker_processes:
                if not process.is_alive():
                    raise RuntimeError(f'Detector worker {process.name} exited with code {process.exitcode}')

    def request_gpu_inference_asset_release(self):
        state = self.state
        if not state.gpu_task_queues or not state.gpu_worker_total_tasks:
            return True
        if state.gpu_worker_results_collected != state.gpu_worker_total_tasks:
            return False
        if state.gpu_inference_asset_release_requested:
            return not state.gpu_inference_asset_release_pending_by_worker
        from .backprojection import _main_process_gpu_stage_epoch
        state.gpu_inference_asset_release_epoch = _main_process_gpu_stage_epoch()
        state.gpu_inference_asset_release_requested = True
        for worker, commands in state.gpu_task_queues.items():
            self._control_seq -= 1
            state.gpu_inference_asset_release_pending_by_worker[worker] = self._control_seq
            commands.put(dict(task_id=self._control_seq, task_type='control',
                              op='release_inference_assets', inference_drained=True))
        return False

    def gpu_inference_asset_release_complete(self):
        return not self.state.gpu_inference_asset_release_pending_by_worker

    def _offer_auxiliary_workers(self):
        from .runtime import gpu_worker_aux_interpolation_pool
        pool = gpu_worker_aux_interpolation_pool()
        if pool is not None and self.gpu_inference_asset_release_complete():
            for worker in self.state.gpu_task_queues:
                pool.enable_worker(worker, allow_full_cpu_affinity=True)

    def shutdown_inference_worker_processes(self):
        from .runtime import gpu_worker_aux_interpolation_pool, set_gpu_worker_aux_interpolation_pool
        pool = gpu_worker_aux_interpolation_pool()
        if pool is not None:
            pool.mark_failed('Inference workers are shutting down')
            set_gpu_worker_aux_interpolation_pool(None)
        for commands in (*self.state.gpu_task_queues.values(), *self.state.cpu_task_queues.values()):
            try:
                commands.put(None)
            except RuntimeError:
                pass  # A cancelled native owner already closed the worker stream.
        for process in self.state.gpu_worker_processes + self.state.cpu_worker_processes:
            self.reap_inference_worker_process(process, process.name)

    def process_quiescence_issues(self):
        issues = {}
        if self.state.gpu_worker_results_collected != self.state.gpu_worker_total_tasks:
            issues['outstanding_tasks'] = self.state.gpu_worker_total_tasks-self.state.gpu_worker_results_collected
        if self._parents:
            issues['outstanding_parent_tickets'] = sorted(self._parents)
        return issues

    def release_tile_dense_result_task_id(self, *_args, **_kwargs):
        raise RuntimeError('Dense tile ownership is unavailable with this native scheduler')

    release_tile_dense_result_for_key = release_tile_dense_result_task_id

    def describe_hybrid_cpu_idle_reason(self):
        return self.state.hybrid_cpu_idle_reason_last

    def direct_union_task_key(self, task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        mode = str(task.get('result_mode', 'file'))
        if (str(task.get('kind', '')) == 'fullframe'
                and bool(task.get('bounded_parent_admission', False))
                and mode not in ('file', 'direct_union')):
            raise RuntimeError(f'Policy parent admission cannot use result mode {mode}')
        admitted_file = mode == 'file' and bool(task.get('bounded_parent_admission', False))
        if str(task.get('kind', '')) != 'fullframe' or not (mode == 'direct_union' or admitted_file):
            return None
        view_obj = task.get('view')
        if view_obj is None:
            return None
        return (str(task.get('model_name', '')), str(getattr(view_obj, 'name', '')))

    def _direct_union_admission_tasks(self, task: Dict[str, object]) -> Tuple[Dict[str, object], ...]:
        if not bool(task.get('bounded_parent_admission', False)):
            return (task,)
        members = (task, *task.get('augmentation_pass_tasks', ()))
        group_mode = str(task.get('result_mode', 'file'))
        keys = set()
        for member in members:
            key = self.direct_union_task_key(member)
            if (key is None
                    or not bool(member.get('bounded_parent_admission', False))):
                raise RuntimeError('Policy parent admission requires independently marked full-frame passes')
            if str(member.get('result_mode', 'file')) != group_mode:
                raise RuntimeError('Policy parent admission requires one result mode for every sibling')
            if key in keys:
                raise RuntimeError(f'Policy parent admission contains a duplicate pass {key}')
            keys.add(key)
        return members

    def direct_union_task_bytes(self, task: Dict[str, object]) -> int:
        return sum(self._direct_union_parent_bytes(member)
                   for member in self._direct_union_admission_tasks(task))

    def tile_task_azimuthal_padding_count(self, task: Dict[str, object]) -> int:
        from .geometry import azimuthal_batch_padding_count
        if str(task.get('kind', '')) != 'tile':
            return 0
        view_obj = task.get('view')
        if not isinstance(view_obj, ViewInfo):
            return 0
        return azimuthal_batch_padding_count(
            view_obj,
            int(task.get('slice_count', view_obj.num_slices)),
            max(1, int(task.get('prediction_batch', self.inputs.batch))),
            slice_offset=int(task.get('slice_start', 0)),
        )

    def tile_dense_result_task_bytes(self, task: Dict[str, object]) -> int:
        if str(task.get('kind', '')) != 'tile':
            return 0
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            return 0
        planes = 2 if task.get('result_conf_path') else 1
        main_bytes = int(math.prod(shape)) * int(planes)
        policy_factor = 1 + len(task.get('augmentation_pass_tasks', ()))
        padding_count = int(self.tile_task_azimuthal_padding_count(task))
        if padding_count <= 0:
            return int(main_bytes) * int(policy_factor)
        plane_bytes = int(shape[1]) * int(shape[2]) * int(np.dtype(np.uint8).itemsize)
        compact_padding_bytes = int(padding_count) * int(plane_bytes) * int(planes)
        # Keep credit for both the worker's seam slots and one compact parity group.
        return (main_bytes + 2 * compact_padding_bytes) * policy_factor

    def gpu_worker_fullframe_parent_key(self, task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if str(task.get('kind', '')) != 'fullframe':
            return None
        view_obj = task.get('view')
        view_name = getattr(view_obj, 'name', None)
        if view_name is None:
            return None
        return (str(task.get('model_name', '')), str(view_name))

    @staticmethod
    def reap_inference_worker_process(process: object, backend_label: str) -> None:
        try:
            process.join(timeout=30)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not process.is_alive():  # type: ignore[attr-defined]
            return
        try:
            process.terminate()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            process.join(timeout=10)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not process.is_alive() or not hasattr(process, "kill"):  # type: ignore[attr-defined]
            return
        try:
            process.kill()  # type: ignore[attr-defined]
            process.join(timeout=30)  # type: ignore[attr-defined]
        except Exception:
            pass
        if process.is_alive():  # type: ignore[attr-defined]
            print(
                f"WARNING: {backend_label} worker pid={getattr(process, 'pid', '?')} "
                "survived SIGKILL after 30s and is being abandoned. It is most likely "
                "stuck in D state; this node may need manual cleanup before it accepts "
                "another job.",
                flush=True,
            )

    def result(self) -> TtaSchedulerResult:
        state = self.state
        return TtaSchedulerResult(
            gpu_inference_drained_at=state.gpu_inference_drained_at,
            gpu_worker_results_collected=int(state.gpu_worker_results_collected),
            gpu_worker_total_tasks=int(state.gpu_worker_total_tasks),
            gpu_frames_completed_total=int(state.gpu_frames_completed_total),
            cpu_frames_completed_total=int(state.cpu_frames_completed_total),
            hybrid_gpu_frames_completed_total=int(
                state.hybrid_gpu_frames_completed_total
            ),
            hybrid_cpu_frames_completed_total=int(
                state.hybrid_cpu_frames_completed_total
            ),
            gpu_dispatched_by_worker=dict(state.gpu_worker_dispatched_by_id),
            gpu_completed_by_worker=dict(state.gpu_worker_results_by_id),
            cpu_dispatched_by_worker=dict(state.cpu_worker_dispatched_by_id),
            cpu_completed_by_worker=dict(state.cpu_worker_results_by_id),
            cpu_worker_ready_details={
                int(key): dict(value)
                for key, value in state.cpu_worker_ready_details_by_id.items()
            },
            hybrid_view_mode_by_parent=dict(state.hybrid_view_mode_by_parent),
            hybrid_view_frames_by_backend={
                key: dict(value)
                for key, value in state.hybrid_view_frames_by_backend.items()
            },
            hybrid_view_tasks_by_backend={
                key: dict(value)
                for key, value in state.hybrid_view_tasks_by_backend.items()
            },
            quiescence_issues=self.process_quiescence_issues(),
            artifacts=TtaSchedulerArtifacts(
                d1_layer_ref_by_parent=state.d1_layer_ref_by_parent,
                d1_view_shadow_path_by_parent=state.d1_view_shadow_path_by_parent,
            ),
        )

    def _process_inference_asset_release_result(self, msg: Dict[str, object]) -> None:
        state = self.state
        worker = msg.get('gpu_index', -1)
        task_id = msg.get('task_id', 0)
        if (
            type(worker) is not int or type(task_id) is not int
            or type(msg.get('ok')) is not bool
            or str(msg.get('worker_kind', 'gpu')).lower() != 'gpu'
            or str(msg.get('op', '')) != 'release_inference_assets'
            or worker not in state.gpu_task_queues
            or task_id >= 0
        ):
            raise RuntimeError(f'Invalid inference-asset release acknowledgement: worker={worker}, task={task_id}')
        stats_value = msg.get('stats') or {}
        if not isinstance(stats_value, Mapping):
            raise RuntimeError('Invalid inference-asset release acknowledgement stats')
        stats = dict(stats_value)
        ok = msg['ok']
        if ok and (stats.get('released') is not True or stats.get('assets_intact', False) is not False
                   or stats.get('phase', 'released') != 'released'):
            raise RuntimeError('Successful inference-asset release did not prove completed retirement')
        safe_refusal = bool(
            stats.get('assets_intact') is True and stats.get('released', False) is False
            and str(stats.get('phase', '')) == 'validate_drain'
        )
        if not ok and not safe_refusal:
            raise RuntimeError(
                f'GPU worker {worker} failed during inference-asset release '
                f'(task {task_id}): {msg.get("error")}\n{msg.get("traceback")}'
            )
        completed = state.gpu_inference_asset_release_results_by_worker.get(worker)
        if completed is not None:
            if task_id == int(completed['task_id']) and bool(msg.get('ok')) == bool(completed['ok']):
                return
            raise RuntimeError(f'Conflicting inference-asset release acknowledgement from worker {worker}')
        expected = state.gpu_inference_asset_release_pending_by_worker.get(worker)
        if expected is None or int(expected) != task_id:
            raise RuntimeError(f'Unrequested inference-asset release acknowledgement: worker={worker}, task={task_id}')
        record = {'task_id': task_id, 'ok': ok, 'stats': stats, 'error': str(msg.get('error') or '')}
        state.gpu_inference_asset_release_results_by_worker[worker] = record
        del state.gpu_inference_asset_release_pending_by_worker[worker]
        runtime_telemetry().add('inference.asset_release.acknowledgements', 1)
        runtime_telemetry().gauge(f'inference.asset_release.worker.{worker}', record)
        runtime_telemetry().gauge('inference.asset_release.pending_workers', sorted(state.gpu_inference_asset_release_pending_by_worker))
        if not ok:
            runtime_telemetry().add('inference.asset_release.safe_refusals', 1)
            print(f'GPU worker {worker} retained inference assets: {record["error"]}; continuing with existing memory admission.')
        elif state.gpu_inference_asset_release_epoch is not None:
            proof = GpuInferenceAssetsRetired(worker, task_id, state.gpu_inference_asset_release_epoch)
            state.gpu_inference_asset_release_proofs_by_worker[worker] = proof
            callback = self._callbacks.gpu_inference_assets_retired
            if callback is not None:
                callback(proof)
        if self.gpu_inference_asset_release_complete():
            self._callbacks.announce_process_inference_drain_if_complete()
        self._offer_auxiliary_workers()
