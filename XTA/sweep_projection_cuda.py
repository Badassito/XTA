"""Admitted numeric-rank CUDA projection for a group of nested cutoff masks.

One uint8 rank source replaces repeated binary uploads. Maximum over the exact
production source addresses commutes with each rank threshold. Bit-lane bytes
are not ranks and must not be passed to this adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import operator
import os
import threading
import time

import numpy as np

_GIB = 1024**3
_CUDA_MODULE_BYTES = 64 * 1024**2
_RANK_STAGE_STATE = {'timeouts': 0}
_RANK_STAGE_STATE_LOCK = threading.Lock()
# Sibling purposes carry this suffix; floor projections never do.
_SIBLING_PURPOSE_SUFFIX = ' sibling ranks'


def rank_projection_gpu_enabled():
    """``XTA_YOLO_SWEEP_RANK_GPU=0`` restores the historical admission exactly."""
    from .workspace import _env_flag
    return _env_flag('XTA_YOLO_SWEEP_RANK_GPU', True)


def rank_projection_gpu_policy():
    """(overlap cap, extra headroom, wait seconds) for sibling rank projection.

    Rank projection may share a GPU with live inference through the existing
    one-per-device overlap lease. By default a sibling lease never exceeds the
    ordinary overlap slice (``gpu_projection_overlap_reserve_bytes``), which
    inference-side sizing already excludes; a larger session uses the CPU.
    ``XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB`` explicitly opts in to a larger
    lease of exactly the funded requirement. Such a lease is kept only while
    ``XTA_YOLO_SWEEP_RANK_GPU_HEADROOM_GIB`` (default 8) more stays free on
    that device, because inference-side sizing excludes only the ordinary
    slice. A busy device is retried for at most
    ``XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS`` (default 60) before the CPU path;
    ``rank_projection_gpu_wait`` shortens that per family.
    """
    from .cuda_backend import gpu_projection_overlap_reserve_bytes
    from .workspace import _env_float
    cap = _env_float('XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB', math.nan)
    cap = int(max(0.0, cap) * _GIB) if math.isfinite(cap) else gpu_projection_overlap_reserve_bytes()
    headroom = _env_float('XTA_YOLO_SWEEP_RANK_GPU_HEADROOM_GIB', 8.0)
    headroom = int(max(0.0, headroom if math.isfinite(headroom) else 8.0) * _GIB)
    wait = _env_float('XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS', 60.0)
    return cap, headroom, max(0.0, wait if math.isfinite(wait) else 60.0)


def rank_projection_gpu_wait(family):
    """(wait seconds, decline reason or None) for one sibling family.

    Overlap slots are scarce (one per GPU, shared with floor projections and
    every publisher), so a sibling waits only as long as its CPU route costs:

    - ``tilted`` (non-azimuthal): exact CPU row spans evaluate ~7-10% of the
      grid (~11-21 s per view on 4 lanes) while the GPU evaluates the whole
      union box, so the GPU is used only when a slot is free right now (zero
      wait), unless ``XTA_YOLO_SWEEP_RANK_GPU_TILTED=1``.
    - ``radial``: its tight-bound CPU route costs ~15-30 s, so it waits at
      most ``XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS`` (default 15).
    - ``azimuthal`` (and anything else): its CPU route costs ~300 s per view,
      so it keeps the full ``XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS`` window.

    The reason names a policy-limited decline in diagnostics and telemetry.
    """
    from .workspace import _env_flag, _env_float
    _, _, wait = rank_projection_gpu_policy()
    if family == 'tilted':
        if _env_flag('XTA_YOLO_SWEEP_RANK_GPU_TILTED', False):
            return wait, None
        return 0.0, 'tilted_prefers_cpu'
    if family == 'radial':
        radial = _env_float('XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS', 15.0)
        radial = max(0.0, radial if math.isfinite(radial) else 15.0)
        if radial < wait:
            return radial, 'radial_wait_budget'
    return wait, None


def _inference_active():
    """Read queued/running inference from the native owner without a CUDA call."""
    from . import backprojection
    coordinator = backprojection._MAIN_PROCESS_GPU_STAGE_COORDINATOR
    return coordinator.inference_active()


def _telemetry(name, value=1):
    try:
        from .runtime import runtime_telemetry
        runtime_telemetry().add(name, value)
    except Exception:
        pass  # Diagnostics never decide admission.


def _floor_retirement_pending(purpose):
    """Whether a live floor projection waits for a GPU turn (coordinator state, no CUDA).

    Floor projections keep priority over sibling projections: while one has
    a live retirement request, siblings stay off the scarce overlap slots, so
    the floor's next poll finds a slot and no sibling lease blocks its drain.
    Floor CPU readers keep re-probing (so requests stay live for a whole CPU
    projection); a request a sibling cannot delay is ignored: one whose floor
    is too large for the slice (overlap-declined) and is not yet due a
    retirement turn (memory pressure, or 30 s age), exactly as the
    coordinator's FIFO grants turns.
    """
    from . import backprojection
    coordinator = backprojection._MAIN_PROCESS_GPU_STAGE_COORDINATOR
    return coordinator.floor_retirement_pending(purpose)


def _acquire_rank_overlap_lease(purpose, required_bytes, state):
    """A shared lease beside inference funded for ``required_bytes``, or None."""
    from . import backprojection
    from .cuda_backend import gpu_projection_overlap_reserve_bytes
    reserve = gpu_projection_overlap_reserve_bytes()
    if reserve <= 0:
        state.update(reason='overlap_disabled', retryable=False)
        return None
    cap, headroom, _ = rank_projection_gpu_policy()
    cap, required = max(reserve, cap), int(required_bytes)
    if required > cap:
        state.update(reason='rank_device_credit', retryable=False, required_device_bytes=required,
                     overlap_cap_bytes=cap)
        return None
    granule = 256 * 1024**2
    budget = reserve if required <= reserve else min(cap, -(-required // granule) * granule)
    if _floor_retirement_pending(purpose):
        state.update(reason='floor_retirement_pending', retryable=True, overlap_budget_bytes=budget)
        return None
    try:
        import torch
    except ImportError:
        state.update(reason='torch_unavailable', retryable=False)
        return None
    lease = backprojection._MAIN_PROCESS_GPU_STAGE_COORDINATOR.try_acquire_overlap_stage(torch, purpose, budget)
    if lease is None:
        state.update(reason='overlap_busy', retryable=True, overlap_budget_bytes=budget)
        return None
    if _floor_retirement_pending(purpose):
        # A floor request arrived between the check and the grant. Requests
        # registered after the grant already avoid this device; earlier ones
        # are seen here, so the floor keeps priority without a coordinator lock.
        lease.release()
        state.update(reason='floor_retirement_pending', retryable=True, overlap_budget_bytes=budget)
        return None
    if budget > reserve:
        try:
            free, _ = torch.cuda.mem_get_info(torch.device(f'cuda:{int(lease.device_index)}'))
        except Exception:
            free = 0
        if int(free) < budget + headroom:
            lease.release()
            state.update(reason='overlap_headroom', retryable=True, overlap_budget_bytes=budget,
                         free_device_bytes=int(free), headroom_bytes=headroom)
            return None
    _telemetry('gpu.projection_overlap.acquired')
    state.update(reason='admitted_overlap', overlap_budget_bytes=budget, required_device_bytes=required)
    return lease


def _admit_rank_stage(attempt, update, family=None):
    """Retry busy-device declines within the family's wait budget, then CPU.

    ``rank_projection_gpu_wait`` sets the budget (zero for tilted: one attempt).
    After one unsuccessful full-length wait the process-wide window shrinks to
    two seconds until a stage is admitted again, so a device that never frees
    cannot stall views; a policy-shortened wait proves nothing about that.
    """
    wait, policy_reason = rank_projection_gpu_wait(family)
    with _RANK_STAGE_STATE_LOCK:
        if _RANK_STAGE_STATE['timeouts']:
            wait = min(wait, 2.0)
    started = time.monotonic()
    deadline, delay, attempts = started + wait, 0.01, 0
    while True:
        attempts += 1
        state = {}
        stage = attempt(state)
        if stage is not None or not state.get('retryable') or time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, min(delay, deadline - time.monotonic())))
        delay = min(0.2, 2 * delay)
    waited = time.monotonic() - started
    busy = stage is None and bool(state.get('retryable'))
    with _RANK_STAGE_STATE_LOCK:
        if stage is not None:
            _RANK_STAGE_STATE['timeouts'] = 0
        elif busy and policy_reason is None:
            _RANK_STAGE_STATE['timeouts'] += 1
    if stage is not None:
        # The admitted lease decides the reason; an earlier failed overlap
        # attempt must not leave its decline reason on an exclusive turn.
        lease = getattr(stage, 'lease', None)
        if lease is not None:
            state['reason'] = 'admitted_overlap' if getattr(lease, 'overlap', False) else 'admitted_exclusive'
    elif busy and policy_reason is not None:
        # The family policy, not a lost device, chose the CPU route.
        state.update(stage_busy_reason=state.get('reason'), reason=policy_reason)
        _telemetry(f'sweep.rank_gpu.policy.{policy_reason}')
    update(stage_attempts=attempts, stage_wait_seconds=round(waited, 4),
           stage_wait_budget_seconds=round(wait, 4), stage_reason=state.get('reason'),
           **{k: v for k, v in state.items() if k in ('overlap_budget_bytes', 'required_device_bytes',
                                                      'unavailable_code', 'stage_busy_reason')})
    _telemetry('sweep.rank_gpu.stage_wait_seconds', waited)
    return stage, state


class _NativeRankCropStage:
    """Retire private CUDA storage before returning the device lease."""

    def __init__(self, session, lease):
        self.session, self.lease = session, lease
        self.device_index = int(lease.device_index)
        self.max_block_depth = int(session.max_rank_block_depth)

    def project_rank_crops(self, first, count, **kwargs):
        try:
            return self.session.project_rank_crops(first, count, **kwargs)
        except BaseException as exc:
            exc.stage_lease = self.lease
            raise

    def close(self):
        if self.session is None:
            return
        try:
            self.session.close()
        except BaseException as exc:
            # A failed fence keeps both the session and its lease owned.
            exc.stage_lease = self.lease
            raise
        self.session = None
        self.lease.release()


def _block_crop_reader(project_block, depth, shape, output_bbox, update, snapshot, is_closed, on_block):
    """Serve ordered rank crops from one cached device block at a time.

    A failed block poisons the reader: its exception (for example
    ``RadialCudaProjectionUnsafeFailure`` with its quarantine owner) is
    re-raised on every later read, never a stale or half-replaced block.
    """
    z_start, z_stop = output_bbox[0], output_bbox[3]
    cache = {'first': None, 'block': None, 'failure': None}

    def read_crop(z):
        if is_closed():
            raise RuntimeError('CUDA rank projection reader is closed')
        if isinstance(z, bool):
            raise IndexError('Projection plane is outside its output grid')
        try:
            index = operator.index(z)
        except TypeError as exc:
            raise IndexError('Projection plane is outside its output grid') from exc
        if not 0 <= index < shape[0]:
            raise IndexError('Projection plane is outside its output grid')
        update(rank_crop_reads=int(snapshot()['rank_crop_reads']) + 1)
        if not z_start <= index < z_stop or output_bbox[1] == output_bbox[4] or output_bbox[2] == output_bbox[5]:
            update(empty_crop_reads=int(snapshot()['empty_crop_reads']) + 1)
            return None
        failure = cache.get('failure')
        if failure is not None:
            raise failure
        first = z_start + (index - z_start) // depth * depth
        if cache['first'] != first:
            # Retire the previous owned packet before another download can
            # coexist with the pinned buffer and the caller's planes. The
            # cache names no block until the new one has fully arrived.
            cache.update(first=None, block=None)
            try:
                block = project_block(first, min(depth, z_stop - first))
            except BaseException as exc:
                cache['failure'] = exc
                raise
            cache.update(first=first, block=block)
            on_block(block)
        block = cache['block']
        record = block.records[index - first]
        if not record.size:
            update(empty_crop_reads=int(snapshot()['empty_crop_reads']) + 1)
            return None
        values = block.payload[record.offset:record.offset + record.size]
        crop = values.reshape(record.y1 - record.y0, record.x1 - record.x0).copy()
        update(cuda_planes=int(snapshot().get('cuda_planes', 0)) + 1)
        return record.y0, record.y1, record.x0, record.x1, crop

    def read_plane(z):
        crop = read_crop(z)
        output = np.zeros(shape[1:], np.uint8)
        update(dense_plane_allocations=int(snapshot()['dense_plane_allocations']) + 1)
        if crop is not None:
            y0, y1, x0, x1, values = crop
            output[y0:y1, x0:x1] = values
        return output

    read_plane.projection_diagnostics = snapshot
    read_plane.read_crop = read_crop
    read_plane.known_z_bounds = (z_start, z_stop)
    read_plane.known_output_bbox_tyx = output_bbox
    return read_plane, cache


def _certificate_is_empty(output_bbox):
    return output_bbox is not None and any(output_bbox[i] == output_bbox[i + 3] for i in range(3))


@contextmanager
def _empty_certificate_reader(shape, output_bbox, update, snapshot):
    """An empty certified output box proves every plane is zero: no GPU turn.

    Serves exactly what an admitted reader would (an empty crop for every
    plane) without admission, source upload or a device lease.
    """
    closed = False

    def unreachable(first, count):
        raise AssertionError('An empty output certificate never projects a block')

    update(backend='certified_empty_rank_crops', reason='empty_output_certificate',
           known_output_bbox_tyx=output_bbox, dense_plane_allocations=0, rank_crop_reads=0, empty_crop_reads=0)
    _telemetry('sweep.rank_gpu.empty_certificate')
    read_plane, _ = _block_crop_reader(unreachable, 1, shape, output_bbox, update, snapshot,
                                       lambda: closed, lambda block: None)
    read_plane.known_z_bounds = (output_bbox[0], output_bbox[0])
    try:
        yield read_plane
    finally:
        closed = True
        update(closed=True)


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if result < 1:
        raise ValueError(f'{name} must be a positive integer')
    return result


@contextmanager
def _shell_rank_projection_reader(source, view, shape, budget, boxes, output_bbox, update, snapshot):
    """Use existing leased shell projectors and preserve numeric ROI payloads."""
    family = str(view.family)
    if output_bbox is not None:
        output_bbox = tuple(output_bbox)
        if (len(output_bbox) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                for v in output_bbox) or any(not 0 <= output_bbox[i] <= output_bbox[i+3] <= shape[i]
                                             for i in range(3))):
            raise ValueError('Shell output certificate exceeds its output grid')
    if not source.flags.c_contiguous:
        update(reason='shell_source_not_contiguous')
        yield None
        return
    if boxes is not None:
        boxes = np.asarray(boxes)
        if (boxes.dtype != np.int64 or boxes.shape != (source.shape[0], 4)
                or np.any(boxes < 0) or np.any(boxes[:, 0] > boxes[:, 1])
                or np.any(boxes[:, 2] > boxes[:, 3])
                or np.any(boxes[:, 1] > source.shape[1]) or np.any(boxes[:, 3] > source.shape[2])):
            raise ValueError('Shell rank bounds must be valid int64 native slice rectangles')
        boxes = np.ascontiguousarray(boxes)
        if not np.any((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2])):
            update(reason='empty_rank_source')
            yield None
            return
    if _certificate_is_empty(output_bbox) and rank_projection_gpu_enabled():
        with _empty_certificate_reader(shape, tuple(int(v) for v in output_bbox), update, snapshot) as read:
            yield read
        return
    from .workspace import _env_int
    upload = max(1, _env_int('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', 8)) * 1024**2
    upload = min(upload, 64 * 1024**2)
    plane = math.prod(shape[1:])
    control = 512 * 1024
    count = source.shape[0]
    # Source rectangles and rank bytes are borrowed. Count every geometry array
    # conservatively, including metadata buffers and a bounded CPU preflight.
    geometry_estimate = 48 * count + 80
    if family == 'radial':
        base = ('transverse', 'sagittal', 'coronal').index(str(view.radial_base_view))
        geometry_estimate += (8 * (sum(shape) + count * int(view.src_w))
                              + 4 * (int(view.src_h) + int(view.src_w)))
    minimum = 4 * plane + upload + geometry_estimate + control
    if plane > 64 * 1024**2 or minimum >= budget:
        update(reason='shell_host_credit', required_nonplan_bytes=minimum)
        yield None
        return
    plan = metadata = None
    build_peak = minimum
    if family == 'radial':
        from . import cylindrical_projection as projection
        if not projection.radial_cuda_backproject_enabled():
            update(reason='radial_cuda_disabled')
            yield None
            return
        from .geometry import radial_global_radii
        available = budget - minimum
        strip = max(1, min(128 * 1024, available // 1024))
        # At most three copies of admitted CSR bytes plus bounded NumPy
        # coordinate strips coexist during production plan construction.
        plan_limit = (available - 256 * strip) // 3
        if plan_limit < 1:
            update(reason='radial_plan_credit', radial_plan_limit_bytes=plan_limit)
            yield None
            return
        try:
            plan, plan_reused = projection._radial_plane_plan(view,
                np.asarray(radial_global_radii(view), np.float64), shape,
                max_plan_bytes=plan_limit, build_chunk_pixels=strip)
        except projection._RadialPlanePlanTooLarge:
            update(reason='radial_plan_credit', radial_plan_limit_bytes=plan_limit)
            yield None
            return
        meta_chunk = max(int(view.src_w), min(128 * 1024, max(1, available // 256)))
        build_peak = minimum + max(0 if plan_reused else 3 * plan_limit + 256 * strip,
                                   int(plan.nbytes) + 64 * meta_chunk)
        update(radial_plan_reused=bool(plan_reused))
        _telemetry(f'sweep.rank_gpu.radial_plan.{"reused" if plan_reused else "built"}')
        if build_peak > budget:
            update(reason='radial_metadata_credit', host_peak_bytes=build_peak)
            yield None
            return
        metadata = projection._radial_projection_metadata(view, source.shape, shape, plan,
                                                           chunk_values=meta_chunk)
        geometry_bytes = (int(plan.nbytes) + sum(a.nbytes for a in metadata if isinstance(a, np.ndarray))
                          + 32 * count + 8 * (count + 1))
        purpose = f'Radial source projection {view.name}'
    else:
        from . import spherical_projection as projection
        if not projection.spherical_cuda_backproject_enabled():
            update(reason='spherical_cuda_disabled')
            yield None
            return
        geometry_bytes = geometry_estimate
        purpose = f'Spherical source projection {view.name}{_SIBLING_PURPOSE_SUFFIX}'
    available_blocks = (budget - upload - 2 * plane - geometry_bytes - control) // (2 * plane)
    if available_blocks < 1 or build_peak > budget:
        update(reason='shell_host_credit', host_peak_bytes=build_peak)
        yield None
        return
    block_bytes = min(64 * 1024**2 // plane, shape[0], available_blocks, 4096) * plane
    host_peak = max(build_peak, geometry_bytes + upload + 2 * block_bytes + 2 * plane + control)
    update(host_peak_bytes=host_peak, plan_bytes=geometry_bytes,
           shell_block_bytes=block_bytes, source_snapshot_bytes=int(source.nbytes))
    if family == 'radial' and rank_projection_gpu_enabled():
        # Beside live inference only the shared overlap lease is used; an
        # exclusive idle-device turn is requested only when no inference is
        # queued or running, so a sibling never parks an inference GPU.
        purpose = f'Radial source projection {view.name} sibling ranks'
        crop_bytes = int(source.nbytes)
        if boxes is not None:
            crop_bytes = min(crop_bytes, int(np.sum((boxes[:, 1]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 2]))))
        depth = max(1, min(block_bytes, 64 * 1024**2) // plane)
        required = (max(1, crop_bytes) + geometry_bytes + 2 * depth * plane + 32 * depth + 12
                    + 256 * 1024**2)  # Matches RadialCudaProjector.required_device_bytes.
        update(required_device_bytes=required)

        def attempt(state):
            return projection._try_radial_cuda_stage(source, plan, metadata, view, shape, boxes,
                boxes is not None, quiet=True, scalar_max=True, block_bytes=block_bytes, upload_bytes=upload,
                retry_state=state, overlap_lease=lambda: _acquire_rank_overlap_lease(purpose, required, state),
                exclusive=not _inference_active(), purpose=purpose)
        stage, admission = _admit_rank_stage(attempt, update, family='radial')
        decline = admission.get('reason') or 'shell_stage_unavailable'
    elif family == 'radial':
        stage = projection._try_radial_cuda_stage(source, plan, metadata, view, shape, boxes,
            boxes is not None, quiet=True, scalar_max=True, block_bytes=block_bytes, upload_bytes=upload)
        decline = 'shell_stage_unavailable'
    else:
        stage = projection._try_spherical_cuda_stage(source, view, shape, boxes, quiet=True,
            scalar_max=True, block_bytes=block_bytes, upload_bytes=upload, purpose=purpose)
        decline = 'shell_stage_unavailable'
    if stage is None:
        from .backprojection import _cancel_main_process_spherical_retirement_request
        _cancel_main_process_spherical_retirement_request(purpose)
        update(reason=decline)
        _telemetry(f'sweep.rank_gpu.declined.{family}')
        yield None
        return
    closed = False
    cache = {}
    try:
        projector = stage.projector
        if output_bbox is None and family == 'spherical':
            bounds = projector.output_bounds
            output_bbox = (bounds.z0, bounds.y0, bounds.x0, bounds.z1, bounds.y1, bounds.x1)
        output_bbox = (0, 0, 0, *shape) if output_bbox is None else tuple(output_bbox)
        if (len(output_bbox) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                for v in output_bbox) or any(not 0 <= output_bbox[i] <= output_bbox[i+3] <= shape[i]
                                             for i in range(3))):
            raise ValueError('Shell output certificate exceeds its output grid')
        output_bbox = tuple(int(v) for v in output_bbox)
        update(backend=f'cuda_{family}_uint8_max_rank_crops', reason='admitted',
               device_index=int(stage.device_index), native_source_bytes=int(source.nbytes),
               source_h2d_bytes=int(projector.source_h2d_bytes),
               source_snapshot_bytes=int(projector.source_h2d_bytes),
               source_layout=str(projector.source_layout), geometry_h2d_bytes=int(projector.geometry_bytes),
               numeric_sample_checks=int(getattr(projector, 'numeric_sample_checks', 0)),
               numeric_observed_samples=int(getattr(projector, 'numeric_observed_samples', 0)),
               source_probe_planes=tuple(getattr(projector, 'source_probe_planes', ())),
               known_output_bbox_tyx=output_bbox, dense_plane_allocations=0,
               rank_crop_reads=0, empty_crop_reads=0)
        _telemetry(f'sweep.rank_gpu.admitted.{family}')

        def on_block(block):
            update(shell_blocks=int(snapshot().get('shell_blocks', 0)) + 1,
                   payload_d2h_bytes=int(snapshot().get('payload_d2h_bytes', 0)) + block.payload.nbytes,
                   metadata_d2h_bytes=int(getattr(projector, 'metadata_d2h_bytes', 0)))
        read_plane, cache = _block_crop_reader(stage.project_rank_crops, int(stage.max_block_depth), shape,
                                               output_bbox, update, snapshot, lambda: closed, on_block)
        yield read_plane
    finally:
        closed = True
        update(closed=True)
        stage.close()
        cache.clear()
        stage = plan = metadata = source = None


@contextmanager
def _native_rank_crop_reader(source, view, shape, budget, boxes, output_bbox, update, snapshot):
    """Tilted/Azimuthal numeric ranks from certified source crops on the GPU.

    Only the certified native rectangles are uploaded (raw rank bytes). The
    existing native-pull kernel takes the exact maximum over the production
    addresses; a device reduction then returns tight nonzero crops only.
    """
    from types import SimpleNamespace
    from . import backprojection
    from .workspace import _env_float, _env_int
    from .native_encoded_cuda import estimate_native_pull_encoded_output
    from .native_pull_cuda import (NativePullCudaSession, NativePullCudaUnavailable, NativePullCudaUnsafeFailure,
                                   estimate_native_pull_cuda_bytes, estimate_native_rank_crop_output,
                                   native_rank_crop_block_depth)
    count = source.shape[0]
    areas = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    occupied = int(np.sum(areas))
    if not occupied:
        update(reason='empty_rank_source')
        yield None
        return
    if output_bbox is None:
        output_bbox = (0, 0, 0, *shape)
    output_bbox = tuple(output_bbox)
    if (len(output_bbox) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
            for v in output_bbox) or any(not 0 <= output_bbox[i] <= output_bbox[i+3] <= shape[i]
                                         for i in range(3))):
        raise ValueError('Native output certificate exceeds its output grid')
    output_bbox = tuple(int(v) for v in output_bbox)
    if _certificate_is_empty(output_bbox):
        with _empty_certificate_reader(shape, output_bbox, update, snapshot) as read:
            yield read
        return
    from .geometry import is_azimuthal_view
    family = 'azimuthal' if is_azimuthal_view(view) else 'tilted'
    upload = max(1, _env_int('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', 8)) * 1024**2
    plane_bytes = math.prod(shape[1:])
    depth = native_rank_crop_block_depth(SimpleNamespace(output_shape=shape))
    if not depth:
        update(reason='rank_crop_plane_credit')
        yield None
        return
    rank = estimate_native_rank_crop_output(SimpleNamespace(output_shape=shape), depth)
    # Pinned upload window, crop records, rank compaction window plus two owned
    # result payloads, and two caller planes; then the geometry plan.
    fixed = upload + 512 * 1024 + 48 * count + rank['host_bytes'] + 2 * plane_bytes
    plan_budget = min(64 * 1024**2, budget - fixed)
    if plan_budget <= 0:
        update(reason='host_credit', required_nonplan_bytes=fixed)
        yield None
        return
    from .projection_coverage_cpu import prepare_native_pull_plan, NativePullPlanUnavailable
    try:
        plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=plan_budget)
    except NativePullPlanUnavailable:
        update(reason='plan_credit')
        yield None
        return
    if not backprojection._native_pull_cuda_stage_enabled(view, plan):
        update(reason='cuda_disabled_or_unsupported')
        yield None
        return
    staging = estimate_native_pull_encoded_output(plan, block_depth=0, slots=2,
                                                  max_upload_bytes=upload, encoded_input=True)
    session_host = int(staging['host_bytes']) + int(rank['host_bytes'])
    host_peak = 2 * plane_bytes + max(int(plan.workspace_bytes), int(plan.persistent_bytes) + session_host)
    max_output = min(depth * plane_bytes, int(plan.max_strip_voxels))
    required = estimate_native_pull_cuda_bytes(source.shape, plan, max_output_voxels=max_output,
        packed_input=False, encoded_nbytes=occupied, rank_crop_block_depth=depth) + _CUDA_MODULE_BYTES
    update(plan_bytes=int(plan.persistent_bytes), plan_build_peak_bytes=int(plan.workspace_bytes),
           stage_host_bytes=session_host, host_peak_bytes=host_peak, native_source_bytes=int(source.nbytes),
           certified_crop_bytes=occupied, required_device_bytes=required, rank_block_depth=depth)
    if host_peak > budget:
        update(reason='host_credit')
        yield None
        return
    purpose = f'Native source projection {view.name} sibling ranks'

    def attempt(state):
        lease = _acquire_rank_overlap_lease(purpose, required, state)
        if lease is not None:
            allowed = int(lease.budget_bytes)
        else:
            if _inference_active():
                return None
            try:
                import torch
            except ImportError:
                state.update(reason='torch_unavailable', retryable=False)
                return None
            lease = backprojection._try_acquire_main_process_gpu_stage(torch, purpose)
            if lease is None:
                state.update(reason='exclusive_busy', retryable=True)
                return None
            try:
                free, _ = torch.cuda.mem_get_info(lease.torch_device(torch))
                reserve = int(max(0., _env_float('YOLO_TTA_NATIVE_PULL_GPU_RESERVE_GIB', 1.)) * _GIB)
                allowed = max(0, int(free) - reserve)
            except BaseException:
                lease.release()
                raise
        try:
            session = NativePullCudaSession(source, plan, device_index=int(lease.device_index),
                max_device_bytes=allowed, max_output_voxels=max_output, scalar_max=True,
                packed_input=False, max_upload_bytes=upload, max_host_bytes=session_host,
                source_boxes=boxes, rank_crop_block_depth=depth)
        except NativePullCudaUnsafeFailure as exc:
            exc.stage_lease = lease
            raise
        except NativePullCudaUnavailable as exc:
            lease.release()
            state.update(reason='cuda_unavailable', retryable=False,
                         unavailable_code=getattr(exc, 'reason_code', 'unavailable'))
            return None
        except BaseException:
            lease.release()
            raise
        return _NativeRankCropStage(session, lease)

    stage, admission = _admit_rank_stage(attempt, update, family=family)
    if stage is None:
        backprojection._cancel_main_process_spherical_retirement_request(purpose)
        update(reason=admission.get('reason', 'stage_unavailable'))
        _telemetry(f'sweep.rank_gpu.declined.{view.family}')
        yield None
        return
    closed = False
    cache = {}
    try:
        session = stage.session
        update(backend='cuda_native_uint8_max_rank_crops', reason='admitted',
               device_index=int(stage.device_index), source_h2d_bytes=int(session.source_h2d_bytes),
               source_snapshot_bytes=int(session.source_h2d_bytes), source_layout='bbox_u8',
               device_bytes=int(session.device_bytes), known_output_bbox_tyx=output_bbox,
               dense_plane_allocations=0, rank_crop_reads=0, empty_crop_reads=0)
        _telemetry(f'sweep.rank_gpu.admitted.{view.family}')

        def project(first, block_count):
            return stage.project_rank_crops(first, block_count, destination_bbox_tyx=output_bbox)

        def on_block(block):
            stats = block.stats or {}
            current = snapshot()
            update(shell_blocks=int(current.get('shell_blocks', 0)) + 1,
                   payload_d2h_bytes=int(current.get('payload_d2h_bytes', 0)) + block.payload.nbytes,
                   metadata_d2h_bytes=int(current.get('metadata_d2h_bytes', 0)) + int(stats.get('metadata_d2h_bytes', 0)),
                   kernel_calls=int(current['kernel_calls']) + int(stats.get('kernel_calls', 0)),
                   contribution_addresses=int(current['contribution_addresses'])
                       + int(stats.get('contribution_addresses', 0)))
        read_plane, cache = _block_crop_reader(project, int(stage.max_block_depth), shape, output_bbox,
                                               update, snapshot, lambda: closed, on_block)
        yield read_plane
    finally:
        closed = True
        update(closed=True)
        # A failed fence keeps the session and lease owned (stage_lease attached).
        stage.close()
        cache.clear()
        stage = plan = source = None


@contextmanager
def cuda_rank_projection_reader(source_rank_u8, view, output_shape_tyx, work_dir, *,
                                level_count, memory_bytes=64 * 1024**2,
                                diagnostics=None, source_max_rank=None,
                                known_native_boxes=None, known_output_bbox_tyx=None):
    """Yield an owned-plane reader, or None for a safe CPU admission fallback.

    The caller keeps the source immutable and owns returned planes. Host credit
    covers plan construction/runtime, production upload staging, and two output
    planes. The existing coordinator separately admits all device residency.
    A successful context owns exactly one production native CUDA stage; its
    stream is fenced and storage retired before the device lease is released.
    Device execution/settlement failures propagate with their quarantine owner.
    ``work_dir`` matches the CPU group-reader contract; no file is allocated.
    ``source_max_rank`` can reuse a certified immutable-source maximum or upper
    bound from the rank packer/occupancy scan, avoiding another volume traversal.
    A trusted packer's ``level_count`` is a sufficient conservative ceiling.
    """
    del work_dir
    if not isinstance(source_rank_u8, np.ndarray):
        raise ValueError('CUDA rank projection borrows an existing uint8 TYX array')
    source = np.asarray(source_rank_u8)
    levels = _positive_integer(level_count, 'level_count')
    budget = _positive_integer(memory_bytes, 'memory_bytes')
    if levels > 255:
        raise ValueError('CUDA rank projection supports at most 255 levels')
    if source.dtype != np.uint8 or source.ndim != 3 or min(source.shape) < 1:
        raise ValueError('CUDA rank projection requires a nonempty uint8 TYX source')
    shape = tuple(_positive_integer(v, 'output dimension') for v in output_shape_tyx)
    if len(shape) != 3:
        raise ValueError('CUDA rank projection requires three output dimensions')
    if source_max_rank is None:
        maximum = int(np.max(source))
    else:
        if isinstance(source_max_rank, bool):
            raise ValueError('source_max_rank must be a certified rank bound in [0,255]')
        try:
            maximum = operator.index(source_max_rank)
        except TypeError as exc:
            raise ValueError('source_max_rank must be a certified rank bound in [0,255]') from exc
        if not 0 <= maximum <= 255:
            raise ValueError('source_max_rank must be a certified rank bound in [0,255]')
    if maximum > levels:
        raise ValueError('Source rank exceeds level_count')
    state = dict(backend='cpu_fallback', reason=None, level_count=levels,
                 cuda_planes=0, kernel_calls=0, contribution_addresses=0,
                 dense_d2h_bytes=0, memory_bytes=budget, closed=False)

    def snapshot():
        return dict(state)

    def update(**values):
        state.update(values)
        if diagnostics is not None:
            diagnostics.update(state)

    from .geometry import is_azimuthal_view, is_tilted_view
    shell = str(view.family) in ('radial', 'spherical')
    if not (shell or is_azimuthal_view(view) or is_tilted_view(view)):
        update(reason='unsupported_family')
        yield None
        return
    from . import backprojection
    from .workspace import _env_flag, _env_int
    if (not backprojection.gpu_backproject_enabled()
            or os.environ.get('CUDA_VISIBLE_DEVICES', '').strip() == '-1'):
        update(reason='cuda_disabled')
        yield None
        return
    if shell:
        with _shell_rank_projection_reader(source, view, shape, budget, known_native_boxes,
                                            known_output_bbox_tyx, update, snapshot) as read:
            yield read
        return
    if known_native_boxes is not None:
        boxes = np.asarray(known_native_boxes)
        if (boxes.dtype != np.int64 or boxes.shape != (source.shape[0], 4)
                or np.any(boxes < 0) or np.any(boxes[:, 0] > boxes[:, 1])
                or np.any(boxes[:, 2] > boxes[:, 3])
                or np.any(boxes[:, 1] > source.shape[1]) or np.any(boxes[:, 3] > source.shape[2])):
            raise ValueError('Native rank bounds must be valid int64 slice rectangles')
        occupied_bytes = int(np.sum((boxes[:, 1]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 2])))
        if source.flags.c_contiguous and rank_projection_gpu_enabled():
            if not _env_flag('YOLO_TTA_NATIVE_PULL_CUDA', True):
                update(reason='cuda_disabled')
                yield None
                return
            # Upload only certified rectangles and return device-compacted
            # rank crops: no dense native upload and no dense plane D2H.
            with _native_rank_crop_reader(source, view, shape, budget, np.ascontiguousarray(boxes),
                                          known_output_bbox_tyx, update, snapshot) as read:
                yield read
            return
        if occupied_bytes * 2 < source.nbytes:
            # Existing encoded native CUDA rejects scalar_max. Do not turn a
            # sparse numeric group into a full dense-view upload while that
            # separate encoded-rank route remains unqualified.
            update(reason='sparse_native_requires_encoded_rank',
                   native_source_bytes=int(source.nbytes), certified_crop_bytes=occupied_bytes)
            yield None
            return
    if not _env_flag('YOLO_TTA_NATIVE_PULL_CUDA', True):
        update(reason='cuda_disabled')
        yield None
        return

    # The production stage uses this exact configured upload window. Reserve
    # pinned staging plus its worst-case bounded dense flattening temporaries
    # before allocating the source-independent geometry plan.
    upload = max(1, _env_int('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', 8)) * 1024**2
    plane_bytes = math.prod(shape[1:])
    host_staging_base = upload + 2 * upload + (upload + 7) // 8 + 256 * 1024
    plan_budget = min(64 * 1024**2, budget - 2 * plane_bytes - host_staging_base)
    if plan_budget <= 0:
        update(reason='host_credit', required_nonplan_bytes=2 * plane_bytes + host_staging_base)
        yield None
        return
    from .projection_coverage_cpu import prepare_native_pull_plan, NativePullPlanUnavailable
    try:
        plan = prepare_native_pull_plan(view, source.shape, shape, max_plan_bytes=plan_budget)
    except NativePullPlanUnavailable:
        update(reason='plan_credit')
        yield None
        return
    if not backprojection._native_pull_cuda_stage_enabled(view, plan):
        update(reason='cuda_disabled_or_unsupported')
        yield None
        return
    from .native_encoded_cuda import estimate_native_pull_encoded_output
    host = estimate_native_pull_encoded_output(plan, block_depth=0, slots=2,
        max_upload_bytes=upload, encoded_input=False)
    host_peak = 2 * plane_bytes + max(int(plan.workspace_bytes),
                                    int(plan.persistent_bytes) + int(host['host_bytes']))
    update(plan_bytes=int(plan.persistent_bytes), plan_build_peak_bytes=int(plan.workspace_bytes),
           stage_host_bytes=int(host['host_bytes']), host_peak_bytes=host_peak,
           max_output_voxels=min(plane_bytes, int(plan.max_strip_voxels)))
    if host_peak > budget:
        update(reason='host_credit')
        yield None
        return

    retry = {}
    stage = backprojection._try_native_pull_cuda_stage(source, view, plan, scalar_max=True,
        max_output_voxels=min(plane_bytes, int(plan.max_strip_voxels)),
        retry_state=retry, max_host_bytes=int(host['host_bytes']))
    if stage is None:
        # This view will finish on CPU rather than retrying mid-reader. Do not
        # leave its fair-device request ahead of later eligible views.
        backprojection._cancel_native_pull_cuda_request(view)
        update(reason=retry.get('reason', 'stage_unavailable'))
        yield None
        return
    try:
        update(backend='cuda_native_uint8_max_rank', reason='admitted',
               device_index=int(stage.device_index), source_snapshot_bytes=int(source.nbytes))
    except BaseException:
        stage.close()
        raise

    def read_plane(z):
        if state['closed']:
            raise RuntimeError('CUDA rank projection reader is closed')
        if isinstance(z, bool):
            raise IndexError('Projection plane is outside its output grid')
        try:
            index = operator.index(z)
        except TypeError as exc:
            raise IndexError('Projection plane is outside its output grid') from exc
        if not 0 <= index < shape[0]:
            raise IndexError('Projection plane is outside its output grid')
        plane = np.empty(shape[1:], np.uint8)
        flat = plane.reshape(-1)
        strip = int(stage.max_output_voxels)
        for first in range(0, plane_bytes, strip):
            # Production pull synchronizes each download before acknowledging
            # it. A partial plane is never returned after a device failure.
            stats = stage.pull_flat_into(flat[first:first + strip],
                first_flat=index * plane_bytes + first)
            update(kernel_calls=state['kernel_calls'] + int((stats or {}).get('kernel_calls', 0)),
                   contribution_addresses=state['contribution_addresses']
                       + int((stats or {}).get('contribution_addresses', 0)),
                   dense_d2h_bytes=state['dense_d2h_bytes']
                       + int((stats or {}).get('dense_d2h_bytes', 0)))
        update(cuda_planes=state['cuda_planes'] + 1)
        return plane

    read_plane.projection_diagnostics = snapshot
    try:
        yield read_plane
    finally:
        update(closed=True)
        # Do not release or discard the stage if stream settlement fails:
        # NativePullCudaUnsafeFailure retains the session and admission lease.
        stage.close()
        stage = plan = source = None
