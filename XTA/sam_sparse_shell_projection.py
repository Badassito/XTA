"""CPU SAM publication from encoded masks through unchanged shell kernels.

The native source borrows the validated CVOL index and packed payload. Geometry remains owned by
the ordinary Radial/Spherical projectors.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import numpy as np

from .native_cpu import kernels
from .interpolation import CTILE_INDEX_DTYPE, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, IncrementalRawBBoxMaskStoreWriter
from .sparse_projection import _SparseMaskSampler


class SparseShellProjectionUnavailable(RuntimeError):
    """CPU adapter qualification failed before output publication began."""


def encoded_shell_has_unused_canvas(store):
    """Use storage-proven empty regions; full bbox coverage keeps the old route."""
    occupied = 0
    for record in store.index:
        kind = int(record['kind'])
        if kind == 0:
            continue
        y0, y1, x0, x1 = (int(record[name]) for name in ('y0','y1','x0','x1'))
        if kind != 1 or not (0 <= y0 < y1 <= store.shape[1] and 0 <= x0 < x1 <= store.shape[2]):
            raise ValueError('Invalid encoded shell input bounds')
        occupied += (y1-y0)*(x1-x0)
    return occupied < math.prod(store.shape)


class _EncodedSource(tuple):
    """Lend a read-only CVOL index and payload to the native shell kernels."""

    def __new__(cls, shape, encoded, index, packed):
        wire_index = index.view(np.uint8).reshape(len(index), CTILE_INDEX_DTYPE.itemsize)
        wire_index.flags.writeable = False
        return super().__new__(cls, (tuple(map(int, shape)), encoded, wire_index, bool(packed)))

    @property
    def shape(self):
        return self[0]


def _require_shell_kernel(name):
    try:
        return getattr(kernels, name)
    except AttributeError as error:
        raise SparseShellProjectionUnavailable('Native encoded shell kernel unavailable') from error


def _prepare_spherical(source, shape_only, view, shape, boxes):
    from .spherical_projection import _validate_spherical_projection, _project_spherical_encoded_block
    from .spherical_projection_bounds import spherical_output_bounds
    from .spherical_projection_cpu import _spherical_cpu_kernel_arguments
    radii, rotation, shape, boxes = _validate_spherical_projection(shape_only, view, shape, boxes)
    bounds = spherical_output_bounds(view, shape, boxes)
    arguments = _spherical_cpu_kernel_arguments(shape_only, view, radii, rotation, shape, 0, 0, 0, boxes)
    fixed = (source, *arguments[1:-7])
    compiled_pull = _require_shell_kernel('spherical_pull')

    def pull(_source, _view, _radii, _rotation, _shape, z, first, stop, _boxes):
        return compiled_pull(*fixed, z, first, stop, 0, 0, 0, False)
    def rectangle(_source, _view, _radii, _rotation, _shape, z, first, stop, _boxes, *, bounds_yx):
        y0, y1, x0, x1 = bounds_yx
        return compiled_pull(*fixed, z, first, stop, x1-x0, x0, y0, False)
    pull.rectangle = rectangle
    def project(z):
        return _project_spherical_encoded_block(source, view, radii, rotation, shape, z, 1,
            boxes, packed=True, output_bounds=bounds, cpu_pull=pull)
    return project


def _prepare_radial(source, shape_only, view, shape, boxes):
    from .geometry import radial_global_radii
    from .cylindrical_projection import (_radial_plane_plan, _RadialPlanePlanTooLarge,
        _radial_projection_metadata, _project_radial_block, _radial_planfree_arguments,
        _pull_radial_range_into, _PULL_CHUNK_VOXELS)
    radii = np.asarray(radial_global_radii(view), np.float64)
    start = int(view.radial_shell_start)
    local = np.asarray(view.radial_radii, np.float64)
    if (radii.ndim != 1 or not radii.size or not np.isfinite(radii).all()
            or np.any(radii < 0) or np.any(np.diff(radii) <= 0)
            or start < 0 or start+source.shape[0] > radii.size
            or local.shape != (source.shape[0],)
            or not np.array_equal(local, radii[start:start+source.shape[0]])):
        raise ValueError('Radial trajectory does not match its global radius grid')
    if view.radial_tilted_source and view.tilt_direction not in ('vertical', 'horizontal'):
        raise ValueError('Unsupported Radial tilt direction')
    try:
        plan, _hit = _radial_plane_plan(view, radii, shape)
    except _RadialPlanePlanTooLarge:
        plan = None
    if plan is not None:
        centers, ideal, sampled, rows, columns, stack, vertical = _radial_projection_metadata(
            view, source.shape, shape, plan)
        arguments = (source, plan.shell_index, plan.column_offsets, plan.native_columns,
            sampled, rows, columns, centers, ideal, int(stack), int(view.radial_height_origin),
            int(view.src_h), plan.base_id, bool(vertical), int(plan.plane_shape[1]), shape[1], shape[2])
        _require_shell_kernel('project_radial_block')
        # Existing integer shell ownership can prove sagittal/coronal Z planes empty.
        occupied = (boxes[:,1] > boxes[:,0]) & (boxes[:,3] > boxes[:,2])
        plane_shells = plan.shell_index.reshape(plan.plane_shape)
        def project(z):
            if plan.base_id != 0:
                shells = plane_shells[z]
                valid = shells >= 0
                if not np.any(occupied[shells[valid]]):
                    return None
            return _project_radial_block(*arguments, z, 1, boxes, True)
        return project
    _require_shell_kernel('pull_radial_range_into')
    arguments = _radial_planfree_arguments(shape_only, view, radii, shape, boxes)
    def project(z):
        plane = np.zeros((1, shape[1], shape[2]), np.uint8)
        flat = plane.reshape(-1)
        for first in range(0, flat.size, _PULL_CHUNK_VOXELS):
            _pull_radial_range_into(flat[first:first+_PULL_CHUNK_VOXELS], source, arguments[1],
                z, first, *arguments[2:], False)
        return plane
    return project


def project_shell_sparse_store(source, view, store_dir, *, out_shape_tyx, workers=1):
    """Publish a packed store using bounded CPU planes and no native decode."""
    from .artifact_archive import physical_path
    from .backprojection import _native_pull_ordered_planes
    from .workspace import _env_int, _cpu_count
    family = str(view.family)
    shape = tuple(map(int, out_shape_tyx))
    if family not in ('radial', 'spherical') or len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Encoded shell projection requires a valid Radial/Spherical output')
    target = Path(store_dir).resolve()
    if target.exists():
        raise FileExistsError('Projected store must be fresh')
    owned = not isinstance(source, RawBBoxMaskStore)
    store = RawBBoxMaskStore.open(Path(source), mmap_payload=True) if owned else source
    sampler = encoded = index = adapter = project = iterator = writer = None
    started = time.perf_counter()
    progress = None
    try:
        original = physical_path(store.root).resolve()
        if target == original or target in original.parents or original in target.parents:
            raise ValueError('Projected store must not replace its input')
        if min(store.shape) <= 0 or store.shape[0] != int(view.num_slices):
            raise ValueError('Shell input shape differs from its view')
        sampler = _SparseMaskSampler(store)
        encoded, index = sampler.encoded.view(), store.index.view()
        encoded.flags.writeable = index.flags.writeable = False
        boxes = np.zeros((store.shape[0], 4), np.int64)
        present = store.index['kind'] == 1
        for column, field in enumerate(('y0','y1','x0','x1')):
            boxes[present,column] = store.index[field][present]
        # The shape-only broadcast owns one byte and is used only for validation.
        shape_only = np.broadcast_to(np.array(0, np.uint8), tuple(store.shape))
        adapter = _EncodedSource(store.shape, encoded, index, store._packbits_payload)
        project = (_prepare_spherical if family == 'spherical' else _prepare_radial)(
            adapter, shape_only, view, shape, boxes)
        workspace = max(1,_env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB',256))*1024**2
        per_plane = 2*shape[1]*shape[2]+shape[1]*((shape[2]+7)//8)+1024**2
        count = min(max(1,int(workers)), _cpu_count(), shape[0], max(0,workspace//per_plane-1))
        if count < 1:
            raise SparseShellProjectionUnavailable('One encoded shell plane exceeds CPU workspace credit')
        writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=target,
            format_name=INTERNAL_PACKED_CVOL_FORMAT, desc=f'Encoded SAM {family} projection',
            extra_meta={'projection_geometry_contract':'xta.native_destination_pull/1',
                        'projection_output_shape_tyx':list(shape),
                        'projection_payload_fusion':'encoded_input_existing_shell_kernel'})
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        progress_key = 'projection.native_destination_pull.live.'+str(view.name)+'.encoded.'+str(threading.get_ident())
        completed = 0
        def progress(state):
            telemetry.gauge(progress_key,dict(view=str(view.name),state=state,
                backend='cpu_encoded_existing_'+family,workers=count,
                completed_planes=completed,total_planes=shape[0],native_decode_bytes=0,
                elapsed_seconds=time.perf_counter()-started,last_update_monotonic=time.monotonic()))
        progress('running')
        last_progress = time.perf_counter()
        if not np.any(present):
            writer.consume_empty_range(0,shape[0])
            completed = shape[0]
        else:
            iterator = _native_pull_ordered_planes(shape[0], project, count)
            for z, block in iterator:
                if block is None:
                    writer.consume_empty_range(z,1)
                elif family=='spherical':
                    writer.consume_encoded_block(z,block.records,block.payload,packed=True)
                else:
                    writer(z,block)
                del block
                completed += 1
                if time.perf_counter()-last_progress >= 2.:
                    progress('running')
                    last_progress = time.perf_counter()
        stats = writer.finalize()
        progress('complete')
        return dict(stats, path=str(target), storage_format=INTERNAL_PACKED_CVOL_FORMAT,
            shape=shape, backend='cpu_encoded_existing_'+family, projection_workers=count,
            native_decode_bytes=0, native_logical_bytes=math.prod(store.shape),
            input_encoded_bytes=int(encoded.size), worker_workspace_bytes=per_plane,
            workspace_accounting='per output worker; shared Radial plan/cache and metadata excluded',
            seconds=time.perf_counter()-started)
    except BaseException as error:
        if progress is not None:
            progress('failed')
        if writer is not None:
            try:
                writer.abort(error)
                writer.discard()
            except BaseException as cleanup:
                if hasattr(error,'add_note'):
                    error.add_note(f'Encoded shell staging cleanup failed: {cleanup}')
        raise
    finally:
        if iterator is not None:
            iterator.close()  # Join every reader before retiring mapped input.
        iterator = project = adapter = encoded = index = None
        if sampler is not None:
            sampler.close()
        if owned:
            store.close()
