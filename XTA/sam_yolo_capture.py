"""Bounded, event-fenced gray capture from actual YOLO model input pixels."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
import hashlib
import json
import os
import sys

import numpy as np

CONTROL_BYTES = 64*1024**2
CAPTURE_CONTRACT = 'xta.cuda_yolo_center_gray8/1'
MAX_PACKET_BYTES = 128*1024**2
MAX_ENCODERS = 16
_WORKER_POOLS = {}
_WORKER_POOLS_LOCK = threading.RLock()


def encoder_worker_count(cpu_workers):
    return max(1,min(MAX_ENCODERS,int(cpu_workers)))


def capture_host_capacity_bytes(batch_size, out_size, cpu_workers=MAX_ENCODERS):
    # Fund two depth-eight packets per encoder across live and deferred tasks.
    frames = max(32,int(batch_size),16*encoder_worker_count(cpu_workers))
    return 4*frames*int(out_size)**2+CONTROL_BYTES


def _metrics(*, counters=None, gauges=None):
    try:
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        for name,value in (counters or {}).items():
            telemetry.add('sam.gray_capture.'+name,value)
        for name,value in (gauges or {}).items():
            telemetry.gauge('sam.gray_capture.'+name,value)
    except Exception:
        pass  # Diagnostics cannot change capture ownership.


@dataclass(eq=False)
class _Copy:
    frames: tuple
    host: object
    copied: object
    read_done: object
    source: object
    gray: object
    charge: int
    copy_proven: bool = False
    event_recorded: bool = False
    stream: object = None


class GrayCapturePool:
    """One worker's shared cap includes every current and deferred capture."""
    def __init__(self, host_capacity_bytes, *, cpu_workers=MAX_ENCODERS):
        self.capacity = int(host_capacity_bytes)
        if self.capacity <= CONTROL_BYTES:
            raise ValueError('Gray capture capacity must include payload and codec control')
        self.condition = threading.Condition(threading.RLock())
        self.used = self.peak = 0
        self.wait_seconds = 0.
        self.wait_count = 0
        self.closed = False
        self.unsettled = set()
        self.sessions = set()
        self.disabled_views = {}
        self.encoder_workers = encoder_worker_count(cpu_workers)
        self.base_encoders = min(4,self.encoder_workers)
        self.started_encoders = self.boost_priority_applied = 0
        self.boost_priority_unavailable = self.boost_priority_errors = 0
        self.boost_nice_values = set()
        self.boost_priority_last_error = None
        self.queued_packets = self.active_encoders = 0
        self.executor = ThreadPoolExecutor(max_workers=self.encoder_workers,
            thread_name_prefix='yolo-gray-encode',initializer=self._initialize_encoder)
        self._report()

    def _initialize_encoder(self):
        with self.condition:
            extra = self.started_encoders >= self.base_encoders
            self.started_encoders += 1
        applied = unavailable = errors = 0
        nice = error = None
        if extra:
            try:
                if not sys.platform.startswith('linux') or not all(
                        hasattr(os,name) for name in ('setpriority','getpriority','PRIO_PROCESS')):
                    unavailable = 1
                else:
                    # Extra encoders yield to feeders; Linux applies nice to this TID only.
                    tid = threading.get_native_id()
                    target = min(19,max(10,os.getpriority(os.PRIO_PROCESS,tid)+5))
                    os.setpriority(os.PRIO_PROCESS,tid,target)
                    nice = os.getpriority(os.PRIO_PROCESS,tid)
                    if nice != target:
                        raise RuntimeError(f'Encoder priority readback {nice} differs from {target}')
                    applied = 1
            except Exception as failure:
                errors = 1
                error = f'{type(failure).__name__}: {failure}'
        with self.condition:
            self.boost_priority_applied += applied
            self.boost_priority_unavailable += unavailable
            self.boost_priority_errors += errors
            if nice is not None:self.boost_nice_values.add(nice)
            if error is not None:self.boost_priority_last_error = error
            self._report()

    def _report(self):
        _metrics(gauges={'pool':self.snapshot()})

    def reserve(self, raw_bytes, *, cancelled=lambda:False, view_key=None):
        charge = 4*int(raw_bytes)
        if charge > self.capacity-CONTROL_BYTES:
            raise MemoryError('Gray packet exceeds its declared worker capture capacity')
        with self.condition:
            if cancelled():
                raise RuntimeError('Gray capture admission cancelled')
            while self.used+charge > self.capacity-CONTROL_BYTES:
                if self.closed or cancelled():
                    raise RuntimeError('Gray capture pool closed while admission was waiting')
                if view_key in self.disabled_views:
                    return 0
                started = time.monotonic()
                self.wait_count += 1
                _metrics(counters={'backpressure_wait_count':1})
                self._report()
                self.condition.wait(.05)
                waited = time.monotonic()-started
                self.wait_seconds += waited
                _metrics(counters={'backpressure_wait_seconds':waited})
            if self.closed or cancelled():
                raise RuntimeError('Gray capture pool is closed')
            if view_key in self.disabled_views:
                return 0
            self.used += charge
            self.peak = max(self.peak,self.used)
            self._report()
        return charge

    def release(self, charge):
        with self.condition:
            if int(charge)>self.used:
                raise RuntimeError('Gray capture credit released twice')
            self.used -= int(charge)
            self.condition.notify_all()
            self._report()

    def snapshot(self):
        with self.condition:
            return dict(capacity_bytes=self.capacity,in_use_bytes=self.used,
                peak_payload_credit_bytes=self.peak,backpressure_wait_seconds=self.wait_seconds,
                backpressure_wait_count=self.wait_count,unsettled_copies=len(self.unsettled),
                encoder_workers=self.encoder_workers,queued_packets=self.queued_packets,
                active_encoders=self.active_encoders,base_encoders=self.base_encoders,
                boost_encoders=self.encoder_workers-self.base_encoders,
                started_encoders=self.started_encoders,
                boost_priority_applied=self.boost_priority_applied,
                boost_priority_unavailable=self.boost_priority_unavailable,
                boost_priority_errors=self.boost_priority_errors,
                boost_nice_values=sorted(self.boost_nice_values),
                boost_priority_last_error=self.boost_priority_last_error)

    def disable_view(self,key,reason):
        with self.condition:
            self.disabled_views.setdefault(key,str(reason))
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            sessions = tuple(self.sessions)
            self.condition.notify_all()
        failure = None
        for session in sessions:
            try:
                session.finish_capture().result()
            except BaseException as error:
                failure = failure or error
        self.executor.shutdown(wait=True)
        for copy in tuple(self.unsettled):
            if copy.stream is None:
                continue
            try:
                copy.stream.synchronize()
            except BaseException:
                continue
            copy.host = copy.source = copy.gray = None
            with self.condition:
                self.unsettled.discard(copy)
            self.release(copy.charge)
        with self.condition:
            if self.unsettled or self.used:
                raise RuntimeError('Gray capture owners remain unsettled; memory credit retained')
            self.closed = True
            self.condition.notify_all()
        if failure is not None:
            raise failure


class GrayCaptureSession:
    """Task-local grouping; the pool owns the shared memory/encoder bound."""
    def __init__(self,pool,payload,*,torch=None,device=None,publish_frames=None,seal=None):
        self.pool,self.payload,self.torch,self.device = pool,payload,torch,device
        self.spec = payload['spec']
        contract = self.spec['capture_contract']
        if contract.get('contract') != CAPTURE_CONTRACT:
            raise ValueError('Unknown YOLO gray capture contract')
        self.channel_index = int(contract['channel_index'])
        self.canvas_hw = tuple(map(int,self.spec['canvas_shape'][-2:]))
        self.native_centers = frozenset(map(int,self.spec['native_centers']))
        self.group_depth = min(8,MAX_PACKET_BYTES//max(1,int(np.prod(self.canvas_hw))))
        self.view_key = hashlib.sha256(json.dumps(self.spec,sort_keys=True,
            allow_nan=False).encode()).hexdigest()
        self.cancelled = threading.Event()
        self._publish,self._seal = publish_frames,seal
        self._state = threading.Condition(threading.RLock())
        self._producer = threading.Lock()
        self._copies = set()
        self._builder = []
        self._builder_frames = 0
        self._seen = set()
        self._pending = 0
        self._closed = self._finalize_queued = False
        self._failure = None
        self.future = Future()
        self.captured_frames = self.packets = 0
        self.encode_seconds = 0.
        self.copy_wait_seconds = self.encode_write_seconds = 0.
        self.committed_frames = 0
        self._stream = None
        if torch is not None:
            self._stream = torch.cuda.Stream(device=device)
        with pool.condition:
            if pool.closed:
                raise RuntimeError('Gray capture pool is closed')
            pool.sessions.add(self)
        if not self.group_depth:
            pool.disable_view(self.view_key,'gray frame exceeds cache packet bound')

    def _write(self,ids,gray):
        if self._publish is not None:
            return self._publish(ids,gray)
        from .gray_cache import GrayCacheWriter,gray_cache_path,GrayCacheUnavailable
        try:
            writer = GrayCacheWriter(gray_cache_path(self.payload['store_root'],self.spec),self.spec)
        except GrayCacheUnavailable as error:
            self.pool.disable_view(self.view_key,error)
            return False
        try:
            accepted = writer.write_frames(ids,gray)
            if not accepted:
                self.pool.disable_view(self.view_key,getattr(writer,'disabled_reason',None) or 'cache disk admission declined')
            return accepted
        finally:
            writer.close()

    def _seal_store(self):
        if self.view_key in self.pool.disabled_views:
            return False
        if self._seal is not None:
            return bool(self._seal())
        from .gray_cache import GrayCacheWriter,gray_cache_path,GrayCacheUnavailable
        try:
            writer = GrayCacheWriter(gray_cache_path(self.payload['store_root'],self.spec),self.spec)
        except GrayCacheUnavailable as error:
            self.pool.disable_view(self.view_key,error)
            return False
        try:
            return bool(writer.seal())
        finally:
            writer.close()

    def capture_gpu(self,tensor,frame_ids,ready_event,*,ring_slot=None):
        """Read input once; ring reuse waits only quantization's last source read."""
        torch = self.torch
        if torch is None:
            raise RuntimeError('GPU gray capture requires its worker torch context')
        ids = tuple(frame_ids)
        if len(tensor.shape)!=4 or len(ids)!=int(tensor.shape[0]):
            raise ValueError('Gray capture frame addresses differ from BCHW batch')
        if tensor.dtype not in (torch.float16,torch.float32):
            raise TypeError('Gray capture requires normalized float16/float32 input')
        if not 0<=self.channel_index<int(tensor.shape[1]):
            raise ValueError('Gray capture center channel is outside its model layout')
        h,w = map(int,tensor.shape[-2:])
        if (h,w)!=self.canvas_hw or any(value is not None and int(value) not in self.native_centers for value in ids):
            raise ValueError('Gray capture pixels/frames differ from declared canonical view')
        with self._producer:
            if self._closed:
                raise RuntimeError('Gray capture task already finished')
            if self.view_key in self.pool.disabled_views:
                return
            index = 0
            while index<len(ids):
                if ids[index] is None or int(ids[index]) in self._seen:
                    index += 1
                    continue
                end = index
                available = self.group_depth-self._builder_frames
                while (end<len(ids) and end-index<available and ids[end] is not None
                        and int(ids[end]) not in self._seen):
                    end += 1
                frames = tuple(int(value) for value in ids[index:end])
                raw_bytes = len(frames)*h*w
                free,total = map(int,torch.cuda.mem_get_info(self.device))
                available = free+max(0,int(torch.cuda.memory_reserved(self.device))
                    -int(torch.cuda.memory_allocated(self.device)))
                if available < 5*raw_bytes+max(2*1024**3,int(total*.05)):
                    self.pool.disable_view(self.view_key,'GPU capture workspace headroom')
                    return
                charge = self.pool.reserve(raw_bytes,cancelled=self.cancelled.is_set,view_key=self.view_key)
                if not charge:
                    return
                copy = None
                host = gray = read_done = None
                launched = False
                try:
                    host = torch.empty((len(frames),h,w),dtype=torch.uint8,device='cpu',pin_memory=True)
                    with torch.cuda.stream(self._stream):
                        if ready_event is not None:
                            self._stream.wait_event(ready_event)
                        tensor.record_stream(self._stream)
                        gray = tensor[index:end,self.channel_index].to(dtype=torch.float32,copy=True)
                        read_done = torch.cuda.Event()
                        read_done.record(self._stream)
                        if ring_slot is not None:
                            ring_slot._sam_gray_capture_read_done = read_done
                        gray.clamp_(0.,1.).mul_(255.).round_()
                        gray = gray.to(torch.uint8)
                        copied = torch.cuda.Event()
                        copy = _Copy(frames,host,copied,read_done,tensor,gray,charge,stream=self._stream)
                        with self._state:
                            self._copies.add(copy)
                        launched = True
                        host.copy_(gray,non_blocking=True)
                        copied.record(self._stream)
                        copy.event_recorded = True
                    self._seen.update(frames)
                    self._builder.append(copy)
                    self._builder_frames += len(frames)
                    self.captured_frames += len(frames)
                    if self._builder_frames==self.group_depth:
                        self._flush_builder()
                except BaseException as error:
                    oom = getattr(torch,'OutOfMemoryError',None)
                    if oom is not None and isinstance(error,oom):
                        try:
                            self._stream.synchronize()
                        except BaseException:
                            retained = copy or _Copy(frames,host,None,read_done,tensor,gray,charge,stream=self._stream)
                            with self.pool.condition:
                                self.pool.unsettled.add(retained)
                            self._failure = RuntimeError('Gray capture allocation cleanup remains unproven')
                            raise
                        if copy is not None:
                            with self._state:
                                self._copies.discard(copy)
                            copy.host = copy.source = copy.gray = None
                        host = gray = None
                        self.pool.release(charge)
                        self.pool.disable_view(self.view_key,'GPU capture allocation declined')
                        return
                    if launched:
                        with self.pool.condition:
                            self.pool.unsettled.add(copy)
                        self._failure = RuntimeError(f'Gray CUDA copy ownership uncertain: {error}')
                    else:
                        try:
                            self._stream.synchronize()
                        except BaseException:
                            retained = _Copy(frames,host,None,read_done,tensor,gray,charge,stream=self._stream)
                            with self.pool.condition:
                                self.pool.unsettled.add(retained)
                            raise
                        host = gray = None
                        self.pool.release(charge)
                    raise
                index = end

    def _flush_builder(self):
        if not self._builder:
            return
        copies,self._builder = self._builder,[]
        self._builder_frames = 0
        with self._state:
            self._pending += 1
            self.packets += 1
        with self.pool.condition:
            self.pool.queued_packets += 1
            self.pool._report()
        try:
            self.pool.executor.submit(self._encode,copies)
        except BaseException:
            with self.pool.condition:
                self.pool.unsettled.update(copies)
                self.pool.queued_packets -= 1
                self.pool._report()
            with self._state:
                self._pending -= 1
                self._failure = RuntimeError('Gray packet encoder submission failed')
            raise

    def _encode(self,copies):
        failure = None
        arrays = gray = None
        started = time.monotonic()
        copy_wait = encode_write = 0.
        with self.pool.condition:
            self.pool.queued_packets -= 1
            self.pool.active_encoders += 1
            self.pool._report()
        try:
            try:
                for copy in copies:
                    if not copy.event_recorded:
                        raise RuntimeError('Gray CUDA copy has no recorded completion event')
                    copy.copied.synchronize()
                    copy.copy_proven = True
                    copy.source = copy.gray = None
            finally:
                copy_wait = time.monotonic()-started
            ids = tuple(frame for copy in copies for frame in copy.frames)
            arrays = [copy.host.numpy() for copy in copies]
            gray = arrays[0] if len(arrays)==1 else np.concatenate(arrays,axis=0)
            encode_started = time.monotonic()
            try:
                accepted = self._write(ids,gray)
            finally:
                encode_write = time.monotonic()-encode_started
            if accepted is False:
                self.pool.disable_view(self.view_key,'cache packet admission declined')
            else:
                with self._state:
                    self.committed_frames += len(ids)
        except BaseException as error:
            # Do not pin released raw buffers in the exception's traceback.
            failure = RuntimeError(f'YOLO gray packet publication failed: {type(error).__name__}: {error}')
        finally:
            arrays = gray = None
            for copy in copies:
                if copy.copy_proven:
                    with self._state:
                        self._copies.discard(copy)
                    copy.host = copy.source = copy.gray = None
                    self.pool.release(copy.charge)
                else:
                    with self.pool.condition:
                        self.pool.unsettled.add(copy)
            with self._state:
                self._pending -= 1
                self.encode_seconds += time.monotonic()-started
                self.copy_wait_seconds += copy_wait
                self.encode_write_seconds += encode_write
                self._failure = self._failure or failure
                self._queue_finalize_locked()
            with self.pool.condition:
                self.pool.active_encoders -= 1
                self.pool._report()
            _metrics(counters={'device_copy_wait_seconds':copy_wait,
                'encode_write_seconds':encode_write,'encoder_packets':1,
                'encoder_failed_packets':int(failure is not None)})

    def _queue_finalize_locked(self):
        if self._closed and not self._pending and not self._finalize_queued:
            self._finalize_queued = True
            try:
                self.pool.executor.submit(self._finalize)
            except BaseException as error:
                self.future.set_exception(error)

    def _finalize(self):
        try:
            if self._failure is not None:
                raise self._failure
            sealed = self._seal_store()
            self.future.set_result(dict(gray_capture_frames=self.captured_frames,
                gray_capture_committed_frames=self.committed_frames,
                gray_capture_packets=self.packets,gray_capture_sealed=sealed,
                gray_capture_disabled_reason=self.pool.disabled_views.get(self.view_key),
                gray_capture_cancelled=self.cancelled.is_set(),
                gray_capture_encode_seconds=self.encode_seconds,
                gray_capture_device_copy_wait_seconds=self.copy_wait_seconds,
                gray_capture_encode_write_seconds=self.encode_write_seconds,
                gray_capture_pool=self.pool.snapshot()))
        except BaseException as error:
            self.future.set_exception(error)
        finally:
            with self.pool.condition:
                self.pool.sessions.discard(self)

    def finish_capture(self):
        """Enqueue the last <=8 frames; return without waiting for compression."""
        with self._producer:
            if not self._closed:
                self._flush_builder()
                with self._state:
                    self._closed = True
                    self._queue_finalize_locked()
        return self.future

    def wait_device_copies(self):
        with self._state:
            copies = tuple(self._copies)
        for copy in copies:
            if not copy.event_recorded:
                raise RuntimeError('Gray copy ownership remains unproven')
            copy.copied.synchronize()
            copy.copy_proven = True
            copy.source = copy.gray = None

    def cancel(self):
        self.cancelled.set()
        self.pool.disable_view(self.view_key,'capture source cancelled or replayed on CPU')
        with self.pool.condition:
            self.pool.condition.notify_all()
        return self.finish_capture()


def worker_capture_session(payload,*,torch,device,cpu_workers=MAX_ENCODERS):
    """Reuse one declared pool across every task/deferred tail in this worker."""
    key = (str(payload['store_root']),str(device))
    capacity = int(payload['host_capacity_bytes'])
    with _WORKER_POOLS_LOCK:
        pool = _WORKER_POOLS.get(key)
        if pool is None:
            pool = GrayCapturePool(capacity,cpu_workers=cpu_workers)
            _WORKER_POOLS[key] = pool
        elif pool.capacity!=capacity:
            raise RuntimeError('Gray capture cannot change a worker-owned memory cap')
    return GrayCaptureSession(pool,payload,torch=torch,device=device)


def close_worker_capture_pools():
    with _WORKER_POOLS_LOCK:
        pools = tuple(_WORKER_POOLS.items())
    failure = None
    for key,pool in pools:
        try:
            pool.close()
        except BaseException as error:
            failure = failure or error
        else:
            with _WORKER_POOLS_LOCK:
                _WORKER_POOLS.pop(key,None)
    if failure is not None:
        raise failure


def merge_publication_futures(*futures):
    """Preserve one deferred task owner until all CPU publication tails finish."""
    inputs = tuple(value for value in futures if isinstance(value,Future))
    result = Future()
    lock = threading.Lock()
    def completed(_future):
        with lock:
            if result.done() or not all(value.done() for value in inputs):
                return
            try:
                merged = {}
                for value in inputs:
                    fields = value.result()
                    if isinstance(fields,dict):
                        merged.update(fields)
                result.set_result(merged)
            except BaseException as error:
                result.set_exception(error)
    for value in inputs:
        value.add_done_callback(completed)
    if not inputs:
        result.set_result({})
    return result
