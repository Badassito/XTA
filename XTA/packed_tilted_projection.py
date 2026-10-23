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
    from .interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT
    from .runtime import (memfd_workspace_enabled, _register_memfd_owner,
                          release_memfd_owners_under)
    from .publication_memory import publication_ram_headroom
    from .workspace import _env_int

    shape = tuple(map(int, volume.shape))
    path = Path(store_dir).absolute()
    if path.exists():
        raise FileExistsError(f'Immutable packed source already exists: {path}')
    reserve = max(0, _env_int('YOLO_TTA_PACKED_PARENT_RAM_RESERVE_MIB', 1024)) * 1024**2
    backing = None
    owner_fd = None
    if memfd_workspace_enabled() and publication_ram_headroom() > reserve + 8 * 1024**2:
        try:
            owner_fd = os.memfd_create('xta-packed-parent', flags=getattr(os, 'MFD_CLOEXEC', 0))
            _register_memfd_owner(str(path / 'chunks.bin'), owner_fd, 'transient packed Tilted parent')
            backing = f'/proc/{os.getpid()}/fd/{owner_fd}'
        except OSError:
            if owner_fd is not None:
                os.close(owner_fd)
            owner_fd = None
    writer = None
    try:
        writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=path,
            format_name=INTERNAL_PACKED_CVOL_FORMAT, payload_backing=backing,
            desc=desc, extra_meta={'producer': 'terminal_native_parent_row_packbits'})
        # Spill only between callbacks. No whole-plane normalization is needed;
        # the established writer scans a view and packs <=1 MiB row slabs.
        plane_bound = shape[1] * ((shape[2] + 7) // 8)
        for z in range(shape[0]):
            if writer._ram_payload and publication_ram_headroom() < reserve + plane_bound:
                writer.spill_payload_to_disk()
                release_memfd_owners_under(path)
            writer.consume(z, volume[z:z + 1])
        stats = writer.finalize()
        resident = int(stats['index_bytes'])
        if stats['payload_backing'] == 'planned_memfd':
            resident += int(stats['raw_payload_bytes'])
        else:
            release_memfd_owners_under(path)
        stats['resident_bytes'] = resident
        return stats
    except BaseException as error:
        if writer is not None:
            writer.abort(error)
            writer.discard()
        release_memfd_owners_under(path)
        raise


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


def _encoded_destination_bbox(store, view, output_shape):
    from .geometry import is_azimuthal_view, tilted_base_view_name
    if is_azimuthal_view(view):
        from .sparse_projection import _destination_bbox_for_sparse_input
        return _destination_bbox_for_sparse_input(store, view, output_shape)
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
    try:
        source_path = physical_path(store.root).resolve()
        if (target.resolve() == source_path or source_path in target.resolve().parents
                or target.resolve() in source_path.parents):
            raise ValueError('Projected store must not replace its input')
        sampler = _SparseMaskSampler(store)  # Validate every record before workers launch.
        plan = prepare_native_pull_plan(view, store.shape, output_shape,
            max_plan_bytes=max(1, _env_int('YOLO_TTA_NATIVE_PULL_PLAN_MIB', 64)) * 1024**2)
        angular = is_azimuthal_view(view)
        destination_bbox = _encoded_destination_bbox(store, view, output_shape)
        bbox_array = np.asarray(destination_bbox or (0, 0, 0, 0, 0, 0), np.int64)
        plane_bytes = output_shape[1] * output_shape[2]
        # The writer scans the borrowed plane and normalizes <=1 MiB row slabs.
        # One consumer plane can overlap the bounded unfinished worker window.
        per_worker = plane_bytes + 3 * max(1024**2, output_shape[2]) + plan.temporary_strip_bytes
        allowance = max(1, _env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', 256)) * 1024**2
        if per_worker + plane_bytes > allowance:
            raise NativePullPlanUnavailable('Encoded Tilted worker exceeds its workspace allowance')
        scheduler = get_native_pull_scheduler()
        limit = scheduler.max_workers if scheduler is not None else max(1, int(workers))
        count = max(1, min(limit, output_shape[0], (allowance - plane_bytes) // per_worker))
        source_shape = np.asarray(store.shape, np.int64)
        # Compile once; empty spans read no source byte.
        if angular:
            pull_native_encoded_flat_into(sampler.encoded, store.index, store._packbits_payload,
                plan, np.empty(0, np.uint8), first_flat=0, destination_bbox_tyx=destination_bbox)
        else:
            _pull_tilted_encoded(sampler.encoded, store.index, store._packbits_payload,
                                source_shape, np.empty(0, np.uint8), 0,
                                plan.integers, plan.coefficients, bbox_array)
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = gpu_planes = 0
        handoff_plane = None
        active_gpu = False
        telemetry = runtime_telemetry()
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
                shared_scheduler=None if scheduler is None else scheduler.snapshot()))
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
                            a = int((sample_stats or {}).get('contribution_addresses', 0))
                            r = int((sample_stats or {}).get('indexed_input_byte_reads', 0))
                        addresses += a
                        reads += r
                    return plane, int(addresses), int(reads)
                def ordered_projection():
                    nonlocal active_gpu, gpu_planes, handoff_plane
                    cpu_stream = stage = None
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
                        return _try_native_pull_encoded_cuda_stage(sampler.encoded, store.index,
                            store._packbits_payload, plan, view,
                            max_output_voxels=min(plane_bytes, int(plan.max_strip_voxels)),
                            retry_state=retry)
                    first_gpu = 0
                    try:
                        stage = acquire(0)
                        if stage is None:
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
                            cpu_stream.close()
                            cpu_stream = None
                        if stage is not None:
                            active_gpu = True
                            handoff_plane = first_gpu
                            progress('running')
                            for z in range(first_gpu, output_shape[0]):
                                gpu_planes += 1
                                yield z, project_plane(z, stage)
                    finally:
                        if cpu_stream is not None:
                            cpu_stream.close()
                        if stage is not None:
                            stage.close()
                stream = ordered_projection()
                total_addresses = total_reads = 0
                last_progress = time.perf_counter()
                packed_plane_bound = output_shape[1] * ((output_shape[2] + 7) // 8)
                memory_probe_planes = max(1, (32 * 1024**2) // max(1, packed_plane_bound))
                next_memory_probe = 0
                for z, (plane, addresses, reads) in stream:
                    if plane is None:
                        writer.consume_empty_range(z, 1)
                    else:
                        if writer._ram_payload:
                            exceeded = writer._next_offset + packed_plane_bound > payload_limit
                            pressure = False
                            if z >= next_memory_probe:
                                # Like Radial publication, fund a bounded growth
                                # window before touching its pages. All future
                                # worst-case payload grants already share one plan.
                                growth = min(memory_probe_planes, output_shape[0] - z) * packed_plane_bound
                                pressure = publication_ram_headroom() < memory_reserve + growth
                                next_memory_probe = z + memory_probe_planes
                            if exceeded or pressure:
                                # The ordered consumer is the sole writer. Spill
                                # only between complete callback transactions.
                                writer.spill_payload_to_disk()
                        writer.consume(z, plane[None])
                    plane = None
                    total_addresses += addresses
                    total_reads += reads
                    completed += 1
                    if time.perf_counter() - last_progress >= 2.:
                        progress('running')
                        last_progress = time.perf_counter()
                stats = writer.finalize()
                staged.rename(target)
                if payload_backing and stats['payload_backing'] != 'planned_memfd':
                    release_memfd_owners_under(target)
                succeeded = True
                progress('complete')
            except BaseException as error:
                if stream is not None:
                    stream.close()  # Join every encoded source reader before retirement.
                writer.abort(error)
                writer.discard()
                progress('failed')
                raise
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
            elapsed_seconds=time.perf_counter() - started)
        runtime_telemetry().add('projection.tilted_encoded.native_decode_bytes_avoided',
                                stats['native_logical_bytes'])
        return stats
    finally:
        _cancel_native_pull_cuda_request(view, failed=not succeeded)
        if sampler is not None:
            sampler.close()
        if owned:
            store.close()
