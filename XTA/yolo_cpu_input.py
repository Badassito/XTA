"""Bounded CPU preparation of Cartesian detector inputs in shared RAM."""
from __future__ import annotations

from collections import defaultdict, deque
from bisect import bisect_right
from concurrent.futures import Executor
from dataclasses import dataclass, field
from itertools import zip_longest
import threading
import time
import errno
import os
import weakref

import numpy as np

from ._deps import cv2
from .geometry import (AugJob, ViewInfo, physical_view_name, tilted_base_view_name,
                       tilted_frame_center, tilted_stack_axis_length)
from .media import wait_for_volume_ready
from .runtime import (_allocate_memfd_workspace_array, _memfd_backing_path_from_array,
                      _memfd_owner_key_from_array, _memfd_owner_fd_for_path,
                      _root_memmap_for_array, close_memmap_array_without_flush, runtime_telemetry)
from .unification.contracts import DataRole
from .unification.sampling import require_forward_sampling
from .workspace import _env_flag


def cpu_input_family_eligible(view):
    """Share the prepared-input device policy with scheduler priority hints."""
    family = str(getattr(view, 'family', ''))
    # Tilted inputs have a resident CUDA renderer that samples directly on the
    # model grid. Full-native NumPy shear/gather preparation competed with
    # mandatory parent cleanup, and speculative packets were mostly revoked
    # before consumption. Keep that CPU qualification route opt-in.
    return family == 'orthogonal' or (family == 'tilted'
        and _env_flag('YOLO_TTA_CPU_INPUT_TILTED', False))


def cpu_input_eligible(task):
    view = task.get('view')
    fmt = task.get('channel_format')
    return bool(isinstance(view, ViewInfo) and isinstance(task.get('job'), AugJob)
        and cpu_input_family_eligible(view)
        and (tilted_base_view_name(view) if str(view.family) == 'tilted'
             else physical_view_name(view)) in {'transverse', 'sagittal', 'coronal'}
        and str(task.get('kind')) == 'fullframe'
        and int(task.get('prediction_batch', 1)) == 1
        and getattr(fmt, 'kind', None) == 'gray' and getattr(fmt, 'channel_count', 0) == 1
        and not task.get('augmentation_pass_tasks')
        and not task.get('sam_gray_cache_spec')
        and not task.get('canonical_image_spec'))


def _t_map(native_t, logical_t):
    positions = (np.arange(logical_t, dtype=np.float64) + .5) * (native_t / logical_t) - .5
    lo = np.clip(np.floor(positions).astype(np.int64), 0, native_t - 1)
    hi = np.minimum(lo + 1, native_t - 1)
    return lo, hi, np.clip(positions - lo, 0., 1.).astype(np.float32)


class NativeCartesianRenderer:
    """Render from decoded native t; never request the deferred processing cube."""
    def __init__(self, volume, view, job):
        require_forward_sampling('cpu', DataRole.INTENSITY)
        self.volume = np.asarray(volume)
        self.view, self.job = view, job
        if (self.volume.ndim != 3 or self.volume.dtype != np.uint8
                or tuple(self.volume.shape[1:]) != (int(view.full_h), int(view.full_w))):
            raise ValueError('CPU detector preparation requires the decoded uint8 native grid')
        self.base = tilted_base_view_name(view) if view.family == 'tilted' else physical_view_name(view)
        self.lo, self.hi, self.alpha = _t_map(self.volume.shape[0], int(view.full_t))

    def _values(self, t, y, x):
        a = self.volume[self.lo[t], y, x].astype(np.float32)
        b = self.volume[self.hi[t], y, x].astype(np.float32)
        return np.rint(a + self.alpha[t] * (b - a)).clip(0., 255.)

    def __call__(self, index):
        view, vol = self.view, self.volume
        if not 0 <= int(index) < int(view.num_slices):
            raise IndexError(index)
        if view.family == 'orthogonal':
            if self.base == 'transverse':
                a, b = vol[self.lo[index]].astype(np.float32), vol[self.hi[index]].astype(np.float32)
                native = np.rint(a + self.alpha[index] * (b - a)).clip(0., 255.).astype(np.uint8)
            else:
                plane = vol[:, index, :] if self.base == 'sagittal' else vol[:, :, index]
                # Same center-aligned linear t reconstruction as the resident renderer.
                a = plane[self.lo].astype(np.float32)
                native = np.rint(a + self.alpha[:, None] * (plane[self.hi].astype(np.float32) - a))
                native = native.clip(0., 255.).astype(np.uint8)
        else:
            native = np.zeros((int(view.src_h), int(view.src_w)), np.uint8)
            center = int(tilted_frame_center(view, index))
            stack_length = int(tilted_stack_axis_length(view))
            shear = np.float32(np.tan(np.deg2rad(float(view.tilt_angle_deg))))
            xs = np.arange(int(view.src_w), dtype=np.int64)[None, :]
            for row0 in range(0, int(view.src_h), 128):
                row1 = min(row0 + 128, int(view.src_h))
                ys = np.arange(row0, row1, dtype=np.int64)[:, None]
                axis = (ys - (int(view.src_h) - 1) / 2 if view.tilt_direction == 'vertical'
                        else xs - (int(view.src_w) - 1) / 2)
                stack = np.float32(center) + shear * axis.astype(np.float32)
                valid = (stack >= 0) & (stack <= stack_length - 1)
                s0 = np.clip(np.floor(stack).astype(np.int64), 0, stack_length - 1)
                s1 = np.minimum(s0 + 1, stack_length - 1)
                weight = stack - s0
                if self.base == 'transverse':
                    a, b = self._values(s0, ys, xs), self._values(s1, ys, xs)
                elif self.base == 'sagittal':
                    a, b = self._values(ys, s0, xs), self._values(ys, s1, xs)
                else:
                    a, b = self._values(ys, xs, s0), self._values(ys, xs, s1)
                native[row0:row1] = np.where(valid, np.rint(a + weight * (b - a)), 0).clip(0, 255)
        # uint8 inputs add at most half a grey level of output quantization. OpenCV's
        # established CPU affine sampler is also used by ordinary CPU detector inputs.
        return cv2.warpAffine(native, np.asarray(self.job.aff.M_src_to_out, np.float32),
            (int(self.job.aff.out_size), int(self.job.aff.out_size)), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@dataclass
class _Packet:
    task: dict
    remaining: set
    future: object = None
    array: object = None
    failed: bool = False
    charged_bytes: int = 0
    building: bool = False
    retiring: bool = False
    running: bool = False
    input_path: object = None
    input_shape: object = None
    built_array: object = None
    build_error: object = None
    borrowers: set = field(default_factory=set)
    revoked: bool = False
    retirement_mapping: object = None
    retirement_fd: object = None


class CpuInputPrefetch:
    """Prepare bounded leases; split siblings share one original coverage owner."""
    def __init__(self, tasks, volume, executor: Executor, *, max_bytes, headroom,
                 allocate=_allocate_memfd_workspace_array, backing=_memfd_backing_path_from_array,
                 close_array=close_memmap_array_without_flush, on_capacity_change=None):
        self.executor, self.volume = executor, volume
        self.max_bytes, self.headroom = max(0, int(max_bytes)), headroom
        self.capacity_bytes = self.max_bytes
        self.reclaim_target = None
        self.allocate, self.backing, self.close_array = allocate, backing, close_array
        self.on_capacity_change = on_capacity_change
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.closed, self.bytes = False, 0
        self.packets = [_Packet(dict(task), set(range(int(task['slice_start']),
            int(task['slice_start']) + int(task['slice_count']))))
            for task in tasks if cpu_input_eligible(task)]
        self.index = defaultdict(list)
        for packet in self.packets:
            self.index[self._parent(packet.task)].append(packet)
        self.starts = {}
        for key, packets in self.index.items():
            packets.sort(key=lambda packet: int(packet.task['slice_start']))
            self.starts[key] = [int(packet.task['slice_start']) for packet in packets]
        self.pending = deque(packet for row in zip_longest(*self.index.values())
                             for packet in row if packet is not None)
        self.retiring = deque()
        self.pump_scheduled = self.pump_requested = False
        self.completed_task_ids = set()
        # Executor.submit can start a thread under a process-wide lock. Keep it
        # on this prestarted controller, never on the GPU scheduler's thread.
        self.control_thread = threading.Thread(target=self._control, name='YOLO CPU input control', daemon=True)
        self.control_thread.start()
        self.pump()

    @staticmethod
    def _size(packet):
        return int(packet.task['slice_count']) * int(packet.task['out_size']) ** 2

    @property
    def reserved_bytes(self):
        # Keep the current grant funded while callbacks can refill it. Only the
        # scheduler can yield unused capacity or replenish it from mask headroom.
        with self.lock:
            return self.capacity_bytes if self.bytes or (not self.closed and any(
                packet.remaining and not packet.revoked and not packet.failed
                for packet in self.packets)) else 0

    def set_capacity(self, grant_bytes):
        with self.lock:
            previous = self.capacity_bytes
            target = min(self.max_bytes, max(0, int(grant_bytes)))
            if self.reclaim_target is not None:
                target = min(target, self.reclaim_target)
            self.capacity_bytes = max(self.bytes, target)
            expanded = self.capacity_bytes > previous
            granted = self.capacity_bytes
        if expanded:
            self.pump()
        return granted

    def try_yield_capacity(self, grant_bytes):
        with self.lock:
            granted = min(self.capacity_bytes, max(0, int(grant_bytes)))
            if self.bytes > granted:
                return None
            self.capacity_bytes = granted
            self.reclaim_target = None
            return granted

    def can_yield_capacity(self, grant_bytes):
        with self.lock:
            return self.bytes <= min(self.capacity_bytes, max(0, int(grant_bytes)))

    def can_reclaim_capacity(self, grant_bytes):
        """Preview optional retirement, never certify that bytes are already free."""
        with self.lock:
            retained = sum(packet.charged_bytes for packet in self.packets if packet.borrowers)
            return not self.closed and retained <= max(0, int(grant_bytes))

    def request_reclaim_capacity(self, grant_bytes, *, preferred_parent=None, preferred_task=None):
        """Ask the controller to retire unborrowed inputs; credit settles later.

        A ready packet may be needed by future detector work whose dense parent
        cannot yet be born. Such speculative inputs must not prevent that birth.
        Attached packets remain owned through the existing compute-release fence.
        """
        requested_packets = requested_bytes = 0
        with self.lock:
            target = min(self.capacity_bytes, max(0, int(grant_bytes)))
            if self.closed or not self.can_reclaim_capacity(target):
                return False
            if self.reclaim_target is not None:
                target = min(target, self.reclaim_target)
            self.reclaim_target = target
            self.capacity_bytes = max(self.bytes, target)
            reclaiming = sum(packet.charged_bytes for packet in self.packets
                             if packet.revoked or packet.retiring)
            candidates = [packet for packet in reversed(self.packets)
                          if packet.charged_bytes and not packet.borrowers
                          and not packet.revoked and not packet.retiring]
            # Preserve the selected parent's useful inputs when possible, and
            # prefer already-built speculative data over interrupting a renderer.
            preferred_packet = None if preferred_task is None else self._packet(preferred_task)
            candidates.sort(key=lambda packet: (
                packet is preferred_packet,
                self._parent(packet.task)[:2] == preferred_parent,
                packet.building, packet.running))
            for packet in candidates:
                if self.bytes - reclaiming <= target:
                    break
                packet.revoked = True
                reclaiming += packet.charged_bytes
                requested_packets += 1
                requested_bytes += packet.charged_bytes
                self._queue_retirement(packet)
            # No close, Future.cancel or executor operation runs on the caller.
        if requested_packets:
            self._record('reclaim_requested_packets', requested_packets)
            self._record('reclaim_requested_charged_bytes', requested_bytes)
        self.pump()
        return True

    def _settle_reclaim_capacity(self):
        if self.reclaim_target is not None:
            self.capacity_bytes = max(self.bytes, min(self.capacity_bytes, self.reclaim_target))

    def _record(self, name, value):
        telemetry = runtime_telemetry()
        if telemetry is not None:
            add = getattr(telemetry, 'add_scheduler_counter', telemetry.add)
            add('yolo.cpu_input.' + name, value)

    def snapshot(self):
        with self.lock:
            ready = [packet for packet in self.packets if packet.array is not None and not packet.retiring]
            return dict(maximum_bytes=self.max_bytes, capacity_bytes=self.capacity_bytes, charged_bytes=self.bytes,
                reserved_bytes=self.reserved_bytes, pending_packets=len(self.pending),
                building_packets=sum(packet.building for packet in self.packets),
                running_builders=sum(packet.running for packet in self.packets),
                ready_packets=len(ready), ready_frames=sum(len(packet.remaining) for packet in ready),
                retiring_packets=sum(packet.retiring for packet in self.packets),
                borrowed_packets=sum(bool(packet.borrowers) for packet in self.packets),
                revoked_packets=sum(packet.revoked for packet in self.packets),
                reclaim_target_bytes=self.reclaim_target, closed=self.closed)

    def _report(self):
        telemetry = runtime_telemetry()
        if telemetry is not None:
            telemetry.gauge('yolo.cpu_input.live', self.snapshot())

    def _capacity_changed(self):
        if not self.closed and self.on_capacity_change is not None:
            self.on_capacity_change()

    def _queue_retirement(self, packet):
        if not packet.building and not packet.retiring:
            packet.retiring = True
            self.retiring.append(packet)

    def _close_packet(self, packet):
        with self.lock:
            array = packet.array
        if array is not None:
            root = _root_memmap_for_array(array)
            mapping = None if root is None else getattr(root, '_mmap', None)
            if mapping is not None and packet.retirement_mapping is None:
                packet.retirement_mapping = weakref.ref(mapping, self._retirement_ready)
                fd = _memfd_owner_fd_for_path(_memfd_owner_key_from_array(array))
                if fd is not None:
                    identity = os.fstat(fd)
                    packet.retirement_fd = (fd, identity.st_dev, identity.st_ino)
            root = mapping = None
            self.close_array(array)
        with self.condition:
            packet.array = None
            array = None  # Drop the final local mapping before returning its credit.
            packet.input_path = packet.input_shape = None
            # close_array registers alias-aware retirement. It does not prove
            # that another NumPy view, callback or owner FD has actually retired.
            if packet.retirement_mapping is not None and packet.retirement_mapping() is not None:
                return False
            if packet.retirement_fd is not None:
                fd, device, inode = packet.retirement_fd
                try:
                    identity = os.fstat(fd)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
                else:
                    if (identity.st_dev, identity.st_ino) == (device, inode):
                        return False
            settled_bytes = packet.charged_bytes if packet.revoked else 0
            packet.retirement_mapping = packet.retirement_fd = None
            self.bytes -= packet.charged_bytes
            packet.charged_bytes = 0
            packet.retiring = False
            packet.borrowers.clear()
            self._settle_reclaim_capacity()
            self.condition.notify_all()
        if settled_bytes:
            self._record('reclaim_settled_packets', 1)
            self._record('reclaim_settled_charged_bytes', settled_bytes)
        self._capacity_changed()
        return True

    def _retirement_ready(self, _reference):
        with self.condition:
            if not self.closed:
                self.pump_requested = self.pump_scheduled = True
            self.condition.notify_all()

    def _build(self, packet):
        task = packet.task
        started = time.perf_counter()
        with self.lock:
            packet.running = True
            if self.closed or packet.revoked or not packet.remaining:
                return None
        wait_for_volume_ready(self.volume)
        with self.lock:
            if self.closed or packet.revoked or not packet.remaining:
                return None
        array = self.allocate((int(task['slice_count']), int(task['out_size']), int(task['out_size'])),
            np.uint8, 'YOLO CPU input', initialize_zero=False)
        try:
            if self.backing(array) is None:
                raise OSError('CPU input preparation requires transferable shared RAM')
            render = NativeCartesianRenderer(self.volume, task['view'], task['job'])
            for local, index in enumerate(range(int(task['slice_start']),
                    int(task['slice_start']) + int(task['slice_count']))):
                with self.lock:
                    if self.closed or packet.revoked or not packet.remaining:
                        return None
                array[local] = render(index)
            self._record('prepared_frames', int(task['slice_count']))
            self._record('prepare_seconds', time.perf_counter() - started)
            # The Future carries completion only. A returned ndarray would leave
            # an alias in Future._result after adoption, delaying memfd retirement.
            with self.lock:
                packet.built_array, array = array, None
            return None
        except Exception as error:
            # Do not retain array-bearing producer frames in Future._exception.
            # Preparation failure still follows the established GPU fallback.
            with self.lock:
                packet.build_error = repr(error)
            return None
        finally:
            if array is not None:
                # Keep failed/cancelled producer payloads in the same retirement
                # owner. The completed Future must retain no ndarray traceback.
                with self.lock:
                    packet.built_array, array = array, None

    def _done(self, packet, future):
        error = None
        path = shape = None
        try:
            future.result()
        except Exception as exc:
            error = repr(exc)
        with self.lock:
            array, packet.built_array = packet.built_array, None
            error = error or packet.build_error
            packet.build_error = None
        if array is not None:
            try:
                path, shape = self.backing(array), tuple(array.shape)
                if path is None:
                    raise OSError('CPU input preparation requires transferable shared RAM')
                path = str(path)
            except Exception as exc:
                error = repr(exc)
        with self.condition:
            packet.array = array
            packet.input_path, packet.input_shape = path, shape
            packet.failed = error is not None
            packet.future = None  # The adopted array, not a completed Future, owns the payload.
            packet.building = False
            packet.running = False
            if self.closed or packet.revoked or not packet.remaining or packet.array is None or error is not None:
                self._queue_retirement(packet)
            else:
                self.bytes -= packet.charged_bytes - self._size(packet)
                packet.charged_bytes = self._size(packet)
            self._settle_reclaim_capacity()
            array = None  # Only the packet retains the adopted mapping now.
            self.condition.notify_all()
        if error is not None and not future.cancelled():
            self._record('prepare_failures', 1)
            print(f'YOLO CPU input preparation unavailable for {packet.task["view"].name}: {error}; GPU rendering remains available.', flush=True)
        self.pump()
        self._report()
        self._capacity_changed()

    def pump(self):
        with self.condition:
            if self.closed:
                return
            self.pump_requested = True
            self.pump_scheduled = True
            self.condition.notify_all()

    def _control(self):
        while True:
            with self.condition:
                # Mapping death normally wakes us. The bounded retry also covers
                # a keepalive-FD finalizer that finishes after that notification.
                self.condition.wait_for(lambda: self.closed or self.pump_requested,
                    timeout=.05 if self.retiring else None)
                if self.closed:
                    return
                self.pump_scheduled = True
            try:
                self._pump()
                self._report()
            except Exception as error:
                if not self.closed:
                    self._record('pump_failures', 1)
                    print(f'YOLO CPU input lookahead unavailable: {error}; GPU rendering remains available.', flush=True)
            finally:
                with self.condition:
                    self.pump_scheduled = self.pump_requested
                    self.condition.notify_all()

    def _pump(self):
        while True:
            with self.lock:
                if self.closed:
                    return
                self.pump_requested = False
                cancelled = [packet.future for packet in self.packets
                             if packet.revoked and packet.building and not packet.running
                             and packet.future is not None]
            # Cancellation can invoke adoption callbacks. Keep it on the existing
            # controller and outside the owner lock, just like array retirement.
            for future in cancelled:
                future.cancel()
            with self.lock:
                retire_count = len(self.retiring)
            for _ in range(retire_count):
                with self.lock:
                    if self.closed or not self.retiring:
                        break
                    packet = self.retiring.popleft()
                try:
                    settled = self._close_packet(packet)
                except BaseException:
                    with self.lock:
                        self.retiring.appendleft(packet)
                    raise
                if not settled:
                    with self.lock:
                        self.retiring.append(packet)
            available = None
            while True:
                with self.lock:
                    while self.pending and (self.pending[0].revoked or not self.pending[0].remaining):
                        self.pending.popleft()
                    if self.closed or not self.pending:
                        break
                    packet = self.pending[0]
                    view = packet.task['view']
                    # Native planes and NumPy lerp temporaries are live only while building.
                    size = self._size(packet) + int(view.src_h) * int(view.src_w) * 24
                    size += min(128, int(view.src_h)) * int(view.src_w) * 128
                    limit = self.capacity_bytes if self.reclaim_target is None else self.reclaim_target
                    if self.bytes + size > limit:
                        break
                if available is None:
                    available = max(0, int(self.headroom()))
                    continue  # Recheck ownership after the slow probe.
                if available < size:
                    break
                with self.lock:
                    if self.closed or self.pending[0] is not packet or packet.revoked or not packet.remaining:
                        continue
                    limit = self.capacity_bytes if self.reclaim_target is None else self.reclaim_target
                    if self.bytes + size > limit:
                        break
                    self.pending.popleft()
                    self.bytes += size
                    packet.charged_bytes = size
                    packet.building = True  # Credit survives the submit/adoption window.
                available -= size
                try:
                    future = self.executor.submit(self._build, packet)
                except BaseException:
                    with self.condition:
                        packet.building = False
                        self._queue_retirement(packet)
                        self.pump_requested = True  # Settle this failed submission without another GPU event.
                        self.condition.notify_all()
                    raise
                with self.lock:
                    packet.future = future
                future.add_done_callback(lambda done, packet=packet: self._done(packet, done))
                # A reclaim request may have arrived while submit was outside the
                # lock; revisit this newly published Future for cancellation.
                if packet.revoked:
                    self.pump()
            with self.lock:
                if not self.pump_requested:
                    return

    def _packet(self, task):
        if not cpu_input_eligible(task):
            return None
        start, count = int(task['slice_start']), int(task['slice_count'])
        key = self._parent(task)
        position = bisect_right(self.starts.get(key, ()), start) - 1
        if position >= 0:
            packet = self.index[key][position]
            original = packet.task
            if start + count <= int(original['slice_start']) + int(original['slice_count']):
                return packet
        return None

    @staticmethod
    def _parent(task):
        return str(task.get('model_name')), str(task['view'].name), str(task['job_id'])

    def ready(self, task):
        packet = self._packet(task)
        if packet is None:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            return not self.closed and packet.input_path is not None and not packet.retiring and not packet.revoked
        finally:
            self.lock.release()

    def attach_ready(self, task):
        packet = self._packet(task)
        if packet is None:
            return
        attached = False
        if not self.lock.acquire(blocking=False):
            self._record('gpu_fallback_frames', int(task['slice_count']))
            return
        try:
            if (not self.closed and packet.input_path is not None and not packet.retiring
                    and not packet.revoked and int(task['task_id']) not in self.completed_task_ids):
                packet.borrowers.add(int(task['task_id']))
                task.update(cpu_input_path=packet.input_path, cpu_input_shape=packet.input_shape,
                    cpu_input_offset=int(task['slice_start']) - int(packet.task['slice_start']))
                attached = True
        finally:
            self.lock.release()
        self._record('offloaded_frames' if attached else 'gpu_fallback_frames', int(task['slice_count']))

    def retire(self, task):
        with self.lock:
            task_id = int(task['task_id'])
            if task_id in self.completed_task_ids:
                return
            self.completed_task_ids.add(task_id)
            packet = self._packet(task)
            if packet is None:
                return
            packet.borrowers.discard(task_id)
            packet.remaining.difference_update(range(int(task['slice_start']),
                int(task['slice_start']) + int(task['slice_count'])))
            if not packet.remaining and not packet.borrowers and not packet.building:
                self._queue_retirement(packet)
        self.pump()

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self.condition:
            self.closed = True
            self.pump_requested = False
            futures = [packet.future for packet in self.packets if packet.future is not None]
            self.condition.notify_all()
        for future in futures:
            future.cancel()
        self.control_thread.join()
        with self.condition:
            self.pump_scheduled = False
            # The controller may have been inside submit when shutdown began.
            futures = [packet.future for packet in self.packets if packet.future is not None]
        for future in futures:
            future.cancel()
        # Workers must have joined before this owner is shut down.
        for future in futures:
            try:
                future.result()
            except Exception:
                pass
        with self.condition:
            while any(packet.building for packet in self.packets):
                self.condition.wait()
        for packet in self.packets:
            while not self._close_packet(packet):
                with self.condition:
                    self.condition.wait(timeout=.05)
        self._report()
