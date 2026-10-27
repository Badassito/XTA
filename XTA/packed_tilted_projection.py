"""Project immutable cropped Tilted masks with bounded destination-plane storage.

The categorical geometry matches the native destination-pull contract. Source
bits are indexed in their encoded crop; neither a native nor destination volume
is expanded. The caller retains the packed source until this transaction ends.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from ._deps import _numba
from .projection_coverage_cpu import _axis_bounds, _axis_value


def terminal_native_source_packing_eligible(view, *, native_bytes, retire_dense,
        component_layers, tiled, keep_temp, prepublished, interpolate,
        interpolation_passes, extrapolate, mask_callback_available=True,
        confidence_can_retire=True):
    """One admission/execution predicate for the terminal packed continuation."""
    from .workspace import _env_flag, _env_int
    from .geometry import is_tilted_view, is_azimuthal_view
    from .interpolation import _view_uses_interpolation
    return bool(_env_flag('YOLO_TTA_PACKED_PARENT', True)
        and int(native_bytes) >= max(0, _env_int('YOLO_TTA_PACKED_PARENT_MIN_MIB', 64)) * 1024**2
        and os.environ.get('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled').strip().lower() == 'compiled'
        and retire_dense and component_layers and not tiled and not keep_temp and not prepublished
        and mask_callback_available and confidence_can_retire
        and (is_tilted_view(view) or is_azimuthal_view(view))
        and not (_view_uses_interpolation(view, int(interpolate)) and int(interpolation_passes) > 0)
        and int(extrapolate) <= 0)


def pack_terminal_native_source(volume, store_dir, *, desc):
    """Seal cropped row bits, using physical-headroom guarded memfd storage.

    Source admission is retained throughout packing. The caller can then return
    its dense credit minus ``resident_bytes`` only after the dense owner dies.
    """
    started = time.perf_counter()
    from .interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT
    from .runtime import (memfd_workspace_enabled, _register_memfd_owner,
                          release_memfd_owners_under, runtime_telemetry)
    from .publication_memory import publication_ram_headroom
    from .workspace import _env_int
    from .packed_dense_source import (encode_dense_u8_packed_block,
        DENSE_PACKED_PAYLOAD_BYTES, DENSE_PACKED_CONTROL_BYTES, MAX_DENSE_PACKED_BLOCK_SLICES)

    imports_seconds = time.perf_counter() - started
    ram_probe_seconds = progress_seconds = setup_seconds = finalize_seconds = 0.
    ram_probe_calls = progress_calls = 0
    setup_complete = False
    def ram_headroom():
        nonlocal ram_probe_seconds, ram_probe_calls
        phase_started = time.perf_counter()
        try:
            return publication_ram_headroom()
        finally:
            ram_probe_seconds += time.perf_counter() - phase_started
            ram_probe_calls += 1

    shape = tuple(map(int, volume.shape))
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('Packed native source requires positive TYX dimensions')
    path = Path(store_dir).absolute()
    if path.exists():
        raise FileExistsError(f'Immutable packed source already exists: {path}')
    reserve = max(0, _env_int('YOLO_TTA_PACKED_PARENT_RAM_RESERVE_MIB', 1024)) * 1024**2
    backing = None
    owner_fd = None
    if memfd_workspace_enabled() and ram_headroom() > reserve + 8 * 1024**2:
        try:
            owner_fd = os.memfd_create('xta-packed-parent', flags=getattr(os, 'MFD_CLOEXEC', 0))
            _register_memfd_owner(str(path / 'chunks.bin'), owner_fd, 'transient packed Tilted parent')
            backing = f'/proc/{os.getpid()}/fd/{owner_fd}'
        except OSError:
            if owner_fd is not None:
                os.close(owner_fd)
            owner_fd = None
    writer = None
    encode_seconds = writer_seconds = 0.
    packed_plane_bound = shape[1] * ((shape[2] + 7) // 8)
    block_encoder = (isinstance(volume, np.ndarray) and volume.dtype == np.uint8
                     and packed_plane_bound <= DENSE_PACKED_PAYLOAD_BYTES)
    block_count = (min(MAX_DENSE_PACKED_BLOCK_SLICES, shape[0],
                       DENSE_PACKED_PAYLOAD_BYTES // max(1, packed_plane_bound))
                   if block_encoder else 1)
    payload_slot = None
    complete = False
    completed_frames = 0
    def progress(state):
        nonlocal progress_seconds, progress_calls
        phase_started = time.perf_counter()
        record = dict(
            description=desc, state=state, completed_frames=completed_frames,
            total_frames=shape[0], elapsed_seconds=time.perf_counter() - started,
            last_update_monotonic=time.monotonic(),
            timing_contract='xta.native_source_pack_timing/2',
            backend='nogil_u8_blocks' if block_encoder else 'bounded_row_slabs',
            block_slices=block_count, encode_seconds=encode_seconds, writer_seconds=writer_seconds,
            ram_probe_seconds=ram_probe_seconds, ram_probe_calls=ram_probe_calls,
            progress_seconds=progress_seconds, progress_calls=progress_calls,
            setup_seconds=setup_seconds, imports_seconds=imports_seconds,
            finalize_seconds=finalize_seconds,
            payload_workspace_bytes=0 if payload_slot is None else payload_slot.nbytes,
            control_workspace_bytes=DENSE_PACKED_CONTROL_BYTES if block_encoder else 0)
        try:
            runtime_telemetry().gauge('projection.native_source_pack.live.' + path.name, record)
        finally:
            # Running gauges contain the cost of preceding progress updates.
            progress_seconds += time.perf_counter() - phase_started
            progress_calls += 1
        return record
    def final_timing(record):
        # Include the final ordinary gauge's own wait without recursively
        # waiting on that same diagnostic lock to measure its publication.
        record.update(progress_seconds=progress_seconds, progress_calls=progress_calls,
            elapsed_seconds=time.perf_counter() - started, last_update_monotonic=time.monotonic())
        runtime_telemetry().set_scheduler_gauge('projection.native_source_pack.live.' + path.name, record)
    try:
        if block_encoder:
            # Production admission funds these exact fixed maxima. The reusable
            # slot is never larger than 32 MiB; metadata/records fit its 64 KiB
            # control allowance. No copied source or full-volume bitset is made.
            payload_slot = np.empty(block_count * packed_plane_bound, np.uint8)
        progress('running')
        writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=path,
            format_name=INTERNAL_PACKED_CVOL_FORMAT, payload_backing=backing,
            desc=desc, extra_meta={'producer': 'terminal_native_parent_row_packbits'})
        setup_seconds = max(0., time.perf_counter() - started - ram_probe_seconds - progress_seconds)
        setup_complete = True
        # Spill only between complete blocks. Oversized planes and compatible
        # non-uint8 callers retain the established <=1 MiB slab fallback.
        for z in range(0, shape[0], block_count):
            count = min(block_count, shape[0] - z)
            growth = count * packed_plane_bound
            scratch = 0 if payload_slot is None else payload_slot.nbytes + DENSE_PACKED_CONTROL_BYTES
            if writer._ram_payload and ram_headroom() < reserve + growth + scratch:
                writer.spill_payload_to_disk()
                release_memfd_owners_under(path)
            if block_encoder:
                phase_started = time.perf_counter()
                records, payload = encode_dense_u8_packed_block(volume, z, count,
                    payload_buffer=payload_slot)
                encode_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                writer.consume_encoded_block(z, records, payload, packed=True)
                writer_seconds += time.perf_counter() - phase_started
                records = payload = None
            else:
                phase_started = time.perf_counter()
                writer.consume(z, volume[z:z + 1])
                writer_seconds += time.perf_counter() - phase_started
            completed_frames += count
            progress('running')
        phase_started = time.perf_counter()
        try:
            stats = writer.finalize()
        finally:
            finalize_seconds += time.perf_counter() - phase_started
        resident = int(stats['index_bytes'])
        if stats['payload_backing'] == 'planned_memfd':
            resident += int(stats['raw_payload_bytes'])
        else:
            release_memfd_owners_under(path)
        stats['resident_bytes'] = resident
        stats.update(packing_backend='nogil_u8_blocks' if block_encoder else 'bounded_row_slabs',
            packing_block_slices=block_count,
            packing_payload_workspace_bytes=0 if payload_slot is None else payload_slot.nbytes,
            packing_control_workspace_bytes=DENSE_PACKED_CONTROL_BYTES if block_encoder else 0,
            packing_encode_seconds=encode_seconds, packing_writer_seconds=writer_seconds)
        complete = True
        final_timing(progress('complete'))
        stats.update(packing_ram_probe_seconds=ram_probe_seconds, packing_ram_probe_calls=ram_probe_calls,
            packing_progress_seconds=progress_seconds, packing_progress_calls=progress_calls,
            packing_setup_seconds=setup_seconds, packing_imports_seconds=imports_seconds,
            packing_finalize_seconds=finalize_seconds,
            packing_elapsed_seconds=time.perf_counter() - started)
        return stats
    except BaseException as error:
        if writer is not None:
            writer.abort(error)
            writer.discard()
        release_memfd_owners_under(path)
        raise
    finally:
        if not setup_complete:
            setup_seconds = max(0., time.perf_counter() - started - ram_probe_seconds - progress_seconds)
        if not complete:
            final_timing(progress('failed'))


@_numba.njit(nogil=True, cache=False)
def _pull_tilted_encoded(encoded, index, packed, source_shape, output, first,
                         integer, coefficient, bbox):
    base, vertical = integer[0], integer[1]
    wt, wh, ww, ot, oh, ow = integer[2:8]
    nh, nw = integer[8], integer[9]
    tangent, origin, shear_center = coefficient[0:3]
    addresses = reads = 0
    local_start = 0
    while local_start < output.size:
        at = first + local_start
        z, y, row_x = at // (oh * ow), (at // ow) % oh, at % ow
        row_count = min(ow - row_x, output.size - local_start)
        if not (bbox[0] <= z < bbox[3] and bbox[1] <= y < bbox[4]):
            output[local_start:local_start + row_count] = 0
            local_start += row_count
            continue
        z0, z1 = _axis_bounds(z, wt, ot)
        y0, y1 = _axis_bounds(y, wh, oh)
        for row_offset in range(row_count):
            x = row_x + row_offset
            if not bbox[2] <= x < bbox[5]:
                output[local_start + row_offset] = 0
                continue
            value = 0
            x0, x1 = _axis_bounds(x, ww, ow)
            for iz in range(z0, z1):
                cz = _axis_value(z, iz, wt, ot, coefficient[9])
                for iy in range(y0, y1):
                    cy = _axis_value(y, iy, wh, oh, coefficient[10])
                    for ix in range(x0, x1):
                        cx = _axis_value(x, ix, ww, ow, coefficient[11])
                        nv = cy if base == 0 else cz
                        nu = cx if base != 2 else cy
                        stack = cz if base == 0 else (cy if base == 1 else cx)
                        axis = nv if vertical else nu
                        frame = stack - tangent * (axis - shear_center) - origin
                        if frame < -.5 or frame >= source_shape[0] - .5:
                            continue
                        if nv < -.5 or nv >= nh - .5 or nu < -.5 or nu >= nw - .5:
                            continue
                        row = coefficient[6] * nu + coefficient[7] * nv + coefficient[8]
                        column = coefficient[3] * nu + coefficient[4] * nv + coefficient[5]
                        f = int(math.floor(frame + .5))
                        r = min(source_shape[1] - 1, max(0, int(math.floor(row + .5))))
                        c = min(source_shape[2] - 1, max(0, int(math.floor(column + .5))))
                        addresses += 1
                        record = index[f]
                        if (record['kind'] == 0 or r < record['y0'] or r >= record['y1']
                                or c < record['x0'] or c >= record['x1']):
                            continue
                        width = np.int64(record['x1']) - np.int64(record['x0'])
                        local_column = c - np.int64(record['x0'])
                        stride = (width + 7) // 8 if packed else width
                        address = (np.int64(record['offset'])
                                   + (r - np.int64(record['y0'])) * stride)
                        address += local_column // 8 if packed else local_column
                        pixel = encoded[address]
                        reads += 1
                        if packed:
                            pixel = (pixel >> (local_column % 8)) & 1
                        if pixel:
                            value = 1
            output[local_start + row_offset] = value
        local_start += row_count
    return addresses, reads


def _encoded_destination_bbox(store, view, output_shape, *, trajectory=None):
    from .geometry import is_azimuthal_view, tilted_base_view_name
    if is_azimuthal_view(view):
        from .sparse_projection import _destination_bbox_for_sparse_input
        return _destination_bbox_for_sparse_input(store, view, output_shape, trajectory=trajectory)
    active = np.flatnonzero(store.index['kind'] != 0)
    if not active.size:
        return None
    stack, v, u = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                    'coronal': (2, 0, 1)}[tilted_base_view_name(view)]
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    low, high = float(active[0]) - .5, float(active[-1]) + .5
    shear = v if view.tilt_direction == 'vertical' else u
    tangent = math.tan(math.radians(float(view.tilt_angle_deg)))
    center = (work[shear] - 1) / 2.
    shifts = (tangent * (-.5 - center), tangent * (work[shear] - .5 - center))
    low += int(view.tilt_frame_start) + min(shifts)
    high += int(view.tilt_frame_start) + max(shifts)
    lower, upper = [0, 0, 0], list(output_shape)
    lower[stack] = max(0, math.floor((low + .5) * output_shape[stack] / work[stack] - .5) - 2)
    upper[stack] = min(output_shape[stack],
        math.ceil((high + .5) * output_shape[stack] / work[stack] - .5) + 3)
    return (*lower, *upper)


def _native_gpu_output_workspace(plan, *, host_budget_bytes, packed_output, max_upload_bytes):
    """Fund constructor staging and two immutable results before CPU admission."""
    from .native_pull_cuda import estimate_native_pull_encoded_output
    options = dict(slots=2, max_upload_bytes=int(max_upload_bytes), encoded_input=True)
    base = estimate_native_pull_encoded_output(plan, block_depth=0, **options)
    if int(base['host_bytes']) > int(host_budget_bytes):
        return dict(base, allowed=False, reason='constructor_host_workspace')
    selected = base
    if packed_output:
        plane = int(plan.output_shape[1]) * int(plan.output_shape[2])
        packed_plane = int(plan.output_shape[1]) * ((int(plan.output_shape[2]) + 7) // 8)
        maximum = min(64, int(plan.output_shape[0]), (64 * 1024**2) // max(1, plane),
                      (32 * 1024**2) // max(1, packed_plane))
        for depth in range(maximum, 0, -1):
            try:
                candidate = estimate_native_pull_encoded_output(plan, block_depth=depth, **options)
            except ValueError:
                # The admitted native plan may exceed the crop encoder's
                # bounded grid/int32 geometry. Its dense CUDA route is valid.
                continue
            if int(candidate['host_bytes']) <= int(host_budget_bytes):
                selected = candidate
                break
    return dict(selected, allowed=True, reason=None)


def project_tilted_sparse_store(source, view, store_dir: Path, *, out_shape_tyx,
                               workers=1, format_name=None, extra_meta=None,
                               output_memory=None):
    """Seal one source-grid CVOL, preserving borrowed immutable source stores."""
    from .geometry import is_tilted_view, is_azimuthal_view
    from .interpolation import (RawBBoxMaskStore, INTERNAL_PACKED_CVOL_FORMAT,
                                IncrementalRawBBoxMaskStoreWriter)
    from .projection_coverage_cpu import (prepare_native_pull_plan, NativePullPlanUnavailable,
                                          pull_native_encoded_flat_into)
    from .sparse_projection import _SparseMaskSampler
    from .backprojection import (_native_pull_ordered_planes,
                                 _native_pull_gpu_ordered_planes,
                                 _native_pull_gpu_ordered_blocks,
                                 _native_pull_cuda_stage_enabled,
                                 _try_native_pull_encoded_cuda_stage,
                                 _cancel_native_pull_cuda_request)
    from .native_pull_scheduling import get_native_pull_scheduler
    from .workspace import _env_int
    from .artifact_archive import physical_path
    from .runtime import (runtime_telemetry, _memfd_owner_fd_for_path,
                          release_memfd_owners_under)
    from .publication_memory import publication_ram_headroom

    if not (is_tilted_view(view) or is_azimuthal_view(view)):
        raise ValueError('Encoded native projection requires a Tilted or Azimuthal view')
    output_shape = tuple(map(int, out_shape_tyx))
    if len(output_shape) != 3 or min(output_shape) < 1:
        raise ValueError('Encoded Tilted output requires positive TYX dimensions')
    target = Path(store_dir).absolute()
    if target.exists():
        raise FileExistsError(f'Immutable projected store already exists: {target}')
    payload_backing = None
    payload_limit = memory_reserve = 0
    if output_memory:
        descriptor = dict(output_memory)
        owner_fd = _memfd_owner_fd_for_path(target / 'chunks.bin')
        if (Path(descriptor['path']).absolute() != target
                or tuple(map(int, descriptor['shape'])) != output_shape
                or (format_name or INTERNAL_PACKED_CVOL_FORMAT) != INTERNAL_PACKED_CVOL_FORMAT
                or int(descriptor['payload_limit']) <= 0 or owner_fd is None
                or str(descriptor['payload_backing']) != f'/proc/{os.getpid()}/fd/{owner_fd}'):
            raise ValueError('Projected output RAM descriptor does not bind this packed layer and owner')
        payload_backing = str(descriptor['payload_backing'])
        payload_limit = int(descriptor['payload_limit'])
        memory_reserve = max(0, int(descriptor['reserve_bytes']))
    owned = not isinstance(source, RawBBoxMaskStore)
    store = RawBBoxMaskStore.open(Path(source), mmap_payload=True) if owned else source
    sampler = None
    succeeded = False
    started = time.perf_counter()
    timings = dict(source_validation=0., plan_preparation=0., destination_bounds=0.,
                   cpu_warmup=0., cpu_warmup_calls=0, gpu_admission=0., cpu_reader_join=0.,
                   gpu_projection=0., gpu_producer_wait=0., writer_consume=0., writer_finalize=0.)
    telemetry = runtime_telemetry()
    prepare_key = ('projection.native_prepare.live.' + str(view.name)
                   + '.encoded.' + str(threading.get_ident()))
    def preparing(phase):
        telemetry.gauge(prepare_key, dict(view=str(view.name), phase=phase,
            elapsed_seconds=time.perf_counter() - started,
            last_update_monotonic=time.monotonic(), timings_seconds=dict(timings)))
    try:
        preparing('source_validation')
        phase_started = time.perf_counter()
        source_path = physical_path(store.root).resolve()
        if (target.resolve() == source_path or source_path in target.resolve().parents
                or target.resolve() in source_path.parents):
            raise ValueError('Projected store must not replace its input')
        sampler = _SparseMaskSampler(store)  # Validate every record before workers launch.
        timings['source_validation'] = time.perf_counter() - phase_started
        preparing('native_plan')
        phase_started = time.perf_counter()
        plan = prepare_native_pull_plan(view, store.shape, output_shape,
            max_plan_bytes=max(1, _env_int('YOLO_TTA_NATIVE_PULL_PLAN_MIB', 64)) * 1024**2)
        timings['plan_preparation'] = time.perf_counter() - phase_started
        preparing('destination_bounds')
        phase_started = time.perf_counter()
        angular = is_azimuthal_view(view)
        trajectory = ((plan.raw_angles_f64, plan.plan_sources, plan.plan_reverses)
                      if angular and plan.raw_angles_f64 is not None else None)
        destination_bbox = _encoded_destination_bbox(store, view, output_shape, trajectory=trajectory)
        bbox_array = np.asarray(destination_bbox or (0, 0, 0, 0, 0, 0), np.int64)
        timings['destination_bounds'] = time.perf_counter() - phase_started
        plane_bytes = output_shape[1] * output_shape[2]
        # The writer scans the borrowed plane and normalizes <=1 MiB row slabs.
        # One consumer plane can overlap the bounded unfinished worker window.
        per_worker = plane_bytes + 3 * max(1024**2, output_shape[2]) + plan.temporary_strip_bytes
        allowance = max(1, _env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', 256)) * 1024**2
        if per_worker + plane_bytes > allowance:
            raise NativePullPlanUnavailable('Encoded Tilted worker exceeds its workspace allowance')
        gpu_workspace = dict(allowed=True, host_bytes=0, block_depth=0, max_payload_bytes=0,
                             pinned_bytes=0, reason=None)
        gpu_candidate = False
        remaining_planes = (0 if destination_bbox is None else
            max(0, min(output_shape[0], destination_bbox[3]) - max(0, destination_bbox[0])))
        minimum_gpu_voxels = max(0, _env_int('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', 64 * 1024**2))
        if (_native_pull_cuda_stage_enabled(view, plan)
                and remaining_planes * plane_bytes >= minimum_gpu_voxels and destination_bbox is not None):
            gpu_candidate = True
            gpu_workspace = _native_gpu_output_workspace(plan,
                host_budget_bytes=allowance - per_worker - plane_bytes,
                packed_output=(format_name or INTERNAL_PACKED_CVOL_FORMAT) == INTERNAL_PACKED_CVOL_FORMAT,
                max_upload_bytes=max(1, _env_int('YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB', 8)) * 1024**2)
        gpu_host_reserve = int(gpu_workspace['host_bytes']) if gpu_workspace['allowed'] else 0
        scheduler = get_native_pull_scheduler()
        limit = scheduler.max_workers if scheduler is not None else max(1, int(workers))
        count = max(1, min(limit, output_shape[0],
                          (allowance - plane_bytes - gpu_host_reserve) // per_worker))
        source_shape = np.asarray(store.shape, np.int64)
        def warm_cpu():
            # A CUDA-ready parent must not wait for an unused CPU signature's
            # global compiler lock. Warm only after GPU admission declines.
            preparing('cpu_warmup')
            phase_started = time.perf_counter()
            try:
                if angular:
                    pull_native_encoded_flat_into(sampler.encoded, store.index, store._packbits_payload,
                        plan, np.empty(0, np.uint8), first_flat=0, destination_bbox_tyx=destination_bbox)
                else:
                    _pull_tilted_encoded(sampler.encoded, store.index, store._packbits_payload,
                        source_shape, np.empty(0, np.uint8), 0,
                        plan.integers, plan.coefficients, bbox_array)
            finally:
                timings['cpu_warmup'] += time.perf_counter() - phase_started
                timings['cpu_warmup_calls'] += 1
                preparing('ready')
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = gpu_planes = 0
        gpu_encoded_blocks = gpu_payload_d2h_bytes = gpu_dense_d2h_bytes = 0
        active_encoded_depth = 0
        handoff_plane = None
        active_gpu = False
        progress_key = ('projection.native_destination_pull.live.' + str(view.name)
                        + '.encoded.' + str(threading.get_ident()))
        def progress(state):
            telemetry.gauge(progress_key, dict(view=str(view.name),
                description=f'Encoded native parent projection {view.name}', state=state,
                backend='cuda_native_encoded' if active_gpu else 'encoded_' + plan.backend,
                workers=count, completed_planes=completed, total_planes=output_shape[0],
                elapsed_seconds=time.perf_counter() - started,
                last_update_monotonic=time.monotonic(),
                plan_bytes=int(plan.persistent_bytes), worker_workspace_bytes=per_worker,
                consumer_plane_bytes=plane_bytes, native_logical_bytes=math.prod(store.shape),
                encoded_payload_bytes=int(sampler.encoded.nbytes),
                gpu_planes=gpu_planes, cuda_handoff_plane=handoff_plane,
                gpu_pipeline_planes=2 * (active_encoded_depth or 1),
                gpu_pipeline_plane_bytes=0 if active_encoded_depth else 2 * plane_bytes,
                gpu_pipeline_blocks=2 if active_encoded_depth else 0,
                gpu_host_reserve_bytes=gpu_host_reserve, gpu_encoded_block_depth=active_encoded_depth,
                gpu_encoded_requested_block_depth=int(gpu_workspace['block_depth']),
                gpu_workspace_decline=gpu_workspace['reason'], gpu_encoded_blocks=gpu_encoded_blocks,
                gpu_payload_d2h_bytes=gpu_payload_d2h_bytes, gpu_dense_d2h_bytes=gpu_dense_d2h_bytes,
                timings_seconds=dict(timings),
                shared_scheduler=None if scheduler is None else scheduler.snapshot()))
        preparing('ready')
        progress('running')
        with tempfile.TemporaryDirectory(prefix='.tilted-encoded-', dir=target.parent) as temporary:
            staged = Path(temporary) / 'projected.cvol'
            writer = IncrementalRawBBoxMaskStoreWriter(shape=output_shape, store_dir=staged,
                format_name=format_name or INTERNAL_PACKED_CVOL_FORMAT,
                payload_backing=payload_backing,
                desc=f'Encoded Tilted projection {view.name}', extra_meta={
                    'projection_geometry_contract': 'xta.native_destination_pull/1',
                    'projection_output_shape_tyx': list(output_shape),
                    'projection_payload_fusion': 'tilted_native_destination_pull',
                    'projection_sink_only': True, **dict(extra_meta or {})})
            stream = None
            try:
                def project_plane(z, stage=None):
                    nonlocal gpu_dense_d2h_bytes
                    projection_started = time.perf_counter()
                    if destination_bbox is None or not destination_bbox[0] <= z < destination_bbox[3]:
                        return None, 0, 0
                    plane = np.empty(output_shape[1:], np.uint8)
                    flat = plane.reshape(-1)
                    addresses = reads = 0
                    strip = int(stage.max_output_voxels if stage is not None else plan.max_strip_voxels)
                    for offset in range(0, plane_bytes, strip):
                        output = flat[offset:min(plane_bytes, offset + strip)]
                        if stage is None and angular:
                            sample_stats = pull_native_encoded_flat_into(sampler.encoded, store.index,
                                store._packbits_payload, plan, output,
                                first_flat=z * plane_bytes + offset,
                                destination_bbox_tyx=destination_bbox)
                            a = sample_stats['contribution_addresses']
                            r = sample_stats['indexed_input_byte_reads']
                        elif stage is None:
                            a, r = _pull_tilted_encoded(sampler.encoded, store.index,
                                store._packbits_payload, source_shape, output,
                                z * plane_bytes + offset, plan.integers, plan.coefficients, bbox_array)
                        else:
                            sample_stats = stage.pull_flat_into(output,
                                first_flat=z * plane_bytes + offset, destination_bbox_tyx=destination_bbox)
                            gpu_dense_d2h_bytes += output.size
                            a = int((sample_stats or {}).get('contribution_addresses', 0))
                            r = int((sample_stats or {}).get('indexed_input_byte_reads', 0))
                        addresses += a
                        reads += r
                    if stage is not None:
                        timings['gpu_projection'] += time.perf_counter() - projection_started
                    return plane, int(addresses), int(reads)
                def project_encoded(first, depth, stage):
                    projection_started = time.perf_counter()
                    try:
                        block = stage.project_encoded(first, depth, destination_bbox_tyx=destination_bbox)
                        if (int(block.first_z) != first or len(block.records) != depth
                                or not bool(block.packed)):
                            raise RuntimeError('Native CUDA encoded block changed its admitted destination range')
                        return block
                    finally:
                        timings['gpu_projection'] += time.perf_counter() - projection_started
                def ordered_projection():
                    nonlocal active_gpu, gpu_planes, handoff_plane, gpu_encoded_blocks, gpu_payload_d2h_bytes
                    nonlocal active_encoded_depth
                    cpu_stream = gpu_stream = stage = None
                    retry = {'retryable': False}
                    def acquire(first_plane):
                        if destination_bbox is None:
                            return None
                        remaining = max(0, min(output_shape[0], destination_bbox[3])
                            - max(first_plane, destination_bbox[0]))
                        minimum = max(0, _env_int('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', 64 * 1024**2))
                        if remaining * plane_bytes < minimum:
                            retry['retryable'] = False
                            _cancel_native_pull_cuda_request(view)
                            return None
                        if not gpu_workspace['allowed']:
                            retry['retryable'] = False
                            _cancel_native_pull_cuda_request(view)
                            return None
                        phase_started = time.perf_counter()
                        try:
                            admission = ({'encoded_output_block_depth': int(gpu_workspace['block_depth']),
                                          'encoded_output_slots': 2, 'max_host_bytes': gpu_host_reserve}
                                         if gpu_candidate else {})
                            return _try_native_pull_encoded_cuda_stage(sampler.encoded, store.index,
                                store._packbits_payload, plan, view,
                                max_output_voxels=min(plane_bytes, int(plan.max_strip_voxels)),
                                retry_state=retry, **admission)
                        finally:
                            timings['gpu_admission'] += time.perf_counter() - phase_started
                    first_gpu = 0
                    try:
                        stage = acquire(0)
                        if stage is None:
                            if destination_bbox is not None:
                                warm_cpu()
                            cpu_stream = _native_pull_ordered_planes(output_shape[0], project_plane,
                                count, scheduler=scheduler)
                            next_probe = time.monotonic() + 1.
                            for z, result in cpu_stream:
                                yield z, result
                                first_gpu = z + 1
                                if first_gpu == output_shape[0]:
                                    break
                                if retry['retryable'] and time.monotonic() >= next_probe:
                                    stage = acquire(first_gpu)
                                    next_probe = time.monotonic() + 1.
                                    if stage is not None:
                                        break
                            # Workers only computed planes. Join unconsumed work
                            # before GPU restarts this uncommitted suffix.
                            phase_started = time.perf_counter()
                            cpu_stream.close()
                            timings['cpu_reader_join'] += time.perf_counter() - phase_started
                            cpu_stream = None
                            result = None
                        if stage is not None:
                            active_gpu = True
                            handoff_plane = first_gpu
                            encoded_depth = int(getattr(stage, 'max_encoded_block_depth', 0))
                            if encoded_depth:
                                if (not gpu_candidate or encoded_depth > int(gpu_workspace['block_depth'])
                                        or int(stage.encoded_host_bytes) > gpu_host_reserve):
                                    raise RuntimeError('Native CUDA encoded staging exceeds its CPU-window reservation')
                                gpu_stream = _native_pull_gpu_ordered_blocks(first_gpu, output_shape[0],
                                    encoded_depth, lambda first, depth: project_encoded(first, depth, stage))
                            else:
                                gpu_stream = _native_pull_gpu_ordered_planes(first_gpu, output_shape[0],
                                    lambda z: project_plane(z, stage))
                            active_encoded_depth = encoded_depth
                            progress('running')
                            while True:
                                wait_started = time.perf_counter()
                                try:
                                    z, result = next(gpu_stream)
                                except StopIteration:
                                    break
                                timings['gpu_producer_wait'] += time.perf_counter() - wait_started
                                if encoded_depth:
                                    block_stats = result.stats
                                    gpu_planes += len(result.records)
                                    gpu_encoded_blocks += 1
                                    gpu_payload_d2h_bytes += int(block_stats['payload_d2h_bytes'])
                                    yield z, (result, int(block_stats['contribution_addresses']),
                                               int(block_stats['indexed_input_byte_reads']))
                                else:
                                    gpu_planes += 1
                                    yield z, result
                                result = None
                    finally:
                        try:
                            if gpu_stream is not None:
                                gpu_stream.close()
                        finally:
                            try:
                                if cpu_stream is not None:
                                    cpu_stream.close()
                            finally:
                                if stage is not None:
                                    stage.close()
                stream = ordered_projection()
                total_addresses = total_reads = 0
                last_progress = time.perf_counter()
                packed_plane_bound = output_shape[1] * ((output_shape[2] + 7) // 8)
                memory_probe_planes = max(1, (32 * 1024**2) // max(1, packed_plane_bound))
                next_memory_probe = 0
                for z, (plane, addresses, reads) in stream:
                    consume_started = time.perf_counter()
                    encoded_block = plane is not None and not isinstance(plane, np.ndarray)
                    consumed_planes = len(plane.records) if encoded_block else 1
                    if plane is None:
                        writer.consume_empty_range(z, 1)
                    else:
                        if writer._ram_payload:
                            exceeded = writer._next_offset + consumed_planes * packed_plane_bound > payload_limit
                            pressure = False
                            if z >= next_memory_probe:
                                # Like Radial publication, fund a bounded growth
                                # window before touching its pages. All future
                                # worst-case payload grants already share one plan.
                                growth = max(consumed_planes,
                                    min(memory_probe_planes, output_shape[0] - z)) * packed_plane_bound
                                pressure = publication_ram_headroom() < memory_reserve + growth
                                next_memory_probe = z + memory_probe_planes
                            if exceeded or pressure:
                                # The ordered consumer is the sole writer. Spill
                                # only between complete callback transactions.
                                writer.spill_payload_to_disk()
                        if encoded_block:
                            writer.consume_encoded_block(z, plane.records, plane.payload, packed=True)
                        else:
                            writer.consume(z, plane[None])
                    timings['writer_consume'] += time.perf_counter() - consume_started
                    plane = None
                    total_addresses += addresses
                    total_reads += reads
                    completed += consumed_planes
                    if time.perf_counter() - last_progress >= 2.:
                        progress('running')
                        last_progress = time.perf_counter()
                finalize_started = time.perf_counter()
                stats = writer.finalize()
                timings['writer_finalize'] += time.perf_counter() - finalize_started
                staged.rename(target)
                if payload_backing and stats['payload_backing'] != 'planned_memfd':
                    release_memfd_owners_under(target)
                succeeded = True
                progress('complete')
            except BaseException as error:
                try:
                    if stream is not None:
                        stream.close()  # Join every encoded source reader before retirement.
                except BaseException as cleanup_error:
                    error = cleanup_error
                writer.abort(error)
                writer.discard()
                progress('failed')
                raise error
            finally:
                if stream is not None:
                    stream.close()
        stats.update(native_logical_bytes=math.prod(store.shape),
            native_decode_bytes_avoided=math.prod(store.shape),
            output_volume_bytes_avoided=math.prod(output_shape),
            worker_workspace_bytes=per_worker, worker_window=count,
            consumer_plane_bytes=plane_bytes,
            indexed_input_byte_reads=total_reads, contribution_addresses=total_addresses,
            gpu_planes=gpu_planes, cuda_handoff_plane=handoff_plane,
            gpu_pipeline_plane_bytes=0 if active_encoded_depth else 2 * plane_bytes,
            gpu_pipeline_blocks=2 if active_encoded_depth else 0, timings_seconds=dict(timings),
            gpu_host_reserve_bytes=gpu_host_reserve, gpu_encoded_block_depth=active_encoded_depth,
            gpu_encoded_requested_block_depth=int(gpu_workspace['block_depth']),
            gpu_encoded_blocks=gpu_encoded_blocks, gpu_payload_d2h_bytes=gpu_payload_d2h_bytes,
            gpu_dense_d2h_bytes=gpu_dense_d2h_bytes,
            elapsed_seconds=time.perf_counter() - started)
        runtime_telemetry().add('projection.tilted_encoded.native_decode_bytes_avoided',
                                stats['native_logical_bytes'])
        return stats
    finally:
        preparing('complete' if succeeded else 'failed')
        _cancel_native_pull_cuda_request(view, failed=not succeeded)
        if sampler is not None:
            sampler.close()
        if owned:
            store.close()
