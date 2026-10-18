"""Bounded exact-operator SAM batches within an existing renderer lease.

Borrow one live SamGpuCropRenderer and consume returned host images before the
next chunk. All requests have one view, physical crop, inverse, and output shape.
No persistent data cache, source owner or publication point is added.
"""
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
import hashlib
import operator
import time

import numpy as np

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class BatchDeclined(RuntimeError):
    """Explicit pre-batch decline; never a successful empty image result."""
    def __init__(self, reason, receipt=None):
        super().__init__(f'Azimuthal batch declined before batch work: {reason}')
        self.reason = str(reason)
        self.receipt = dict(receipt or {})


@dataclass(frozen=True)
class BatchPlan:
    eligible: bool
    reason: str
    batch_size: int
    bbox_yx: tuple | None = None
    row_block: int = 0
    projection_bytes: int = 0
    warp_bytes: int = 0
    host_bytes: int = 0
    device_cap_bytes: int = 0
    host_cap_bytes: int = 0
    selected_rows: int = 0
    virtual_t: bool = False
    retained_cache_bytes: int | None = None
    existing_gpu_allowance_bytes: int = 0
    total_device_bound_bytes: int | None = None
    cold_cache_growth_bound_bytes: int = 0
    cold_cache_work_bound_bytes: int = 0
    memory_eligible: bool = False
    scalar_row_block: int = 0
    scalar_gather_iterations_per_image: int = 0
    scalar_gather_iterations: int = 0
    batch_gather_iterations: int | None = None
    gather_iteration_guard_passed: bool = False


@dataclass(frozen=True)
class BatchResult:
    images: tuple
    receipt: dict


def batch_workspace_ledger(*, batch_size, native_shape_yx, logical_t,
                          roi_shape_yx, output_shape_yx, selected_rows,
                          scalar_row_block, scratch_bytes, retained_cache_bytes):
    """Pure integer ledger; no XTA, Torch, device, tensor or allocator access.

    A caller must explicitly supply retained bytes, including zero for a stated
    cold-cache scenario. This helper does not prove that scenario is actual.
    """
    size = operator.index(batch_size)
    native_h, native_w = map(operator.index, native_shape_yx)
    roi_h, roi_w = map(operator.index, roi_shape_yx)
    height, width = map(operator.index, output_shape_yx)
    logical_t, selected_rows, scalar_row_block, scratch = map(operator.index,
        (logical_t, selected_rows, scalar_row_block, scratch_bytes))
    retained = operator.index(retained_cache_bytes)
    if (not 1 <= size <= 4 or min(native_h,native_w,roi_h,roi_w,height,width,
            logical_t,selected_rows,scalar_row_block,scratch) < 1 or retained < 0):
        raise ValueError('Invalid integer batch workspace dimensions or cache bytes')
    area, pixels, roi_area = native_h*native_w, height*width, roi_h*roi_w
    allowance = 256*area+96*pixels
    device_cap = min(scratch, max(0,allowance-retained))
    fixed = 4*native_w+20*logical_t+32*native_h
    projection_fixed = fixed+256*size*native_w+size*area+32*size*roi_area
    warp_bytes = fixed+6*size*area+16*size*pixels+32*pixels
    host_bytes = 2*size*pixels+128*(native_h+logical_t+native_w+roi_w)+64*size
    denominator = 256*size*roi_w
    row_block = min(scalar_row_block, selected_rows, max(0,device_cap-projection_fixed)//denominator)
    projection_bytes = projection_fixed+denominator*row_block
    memory_eligible = warp_bytes <= device_cap and host_bytes <= scratch and row_block >= 1
    scalar_iterations_per_image = (selected_rows+scalar_row_block-1)//scalar_row_block
    scalar_iterations = size*scalar_iterations_per_image
    batch_iterations = (selected_rows+row_block-1)//row_block if row_block else None
    gather_guard = batch_iterations is not None and batch_iterations <= scalar_iterations
    eligible = memory_eligible and gather_guard
    reason = ('scratch_or_cache_bound' if not memory_eligible else
              'gather_iteration_regression' if not gather_guard else 'compatible')
    return dict(eligible=eligible, reason=reason,
        row_block=row_block, projection_bytes=projection_bytes, warp_bytes=warp_bytes,
        host_bytes=host_bytes, device_cap_bytes=device_cap, host_cap_bytes=scratch,
        retained_cache_bytes=retained, existing_gpu_allowance_bytes=allowance,
        total_device_bound_bytes=retained+max(projection_bytes,warp_bytes),
        cold_cache_growth_bound_bytes=20*logical_t+20*native_h+8*pixels,
        cold_cache_work_bound_bytes=20*logical_t+32*native_h+32*pixels,
        memory_eligible=memory_eligible,scalar_row_block=scalar_row_block,
        scalar_gather_iterations_per_image=scalar_iterations_per_image,
        scalar_gather_iterations=scalar_iterations,batch_gather_iterations=batch_iterations,
        gather_iteration_guard_passed=gather_guard)


def retained_cache_snapshot(engine):
    """Count full Torch backing storages under the borrowed exclusive owner.

    The final call occurs after renderer._start selects its actual engine/device.
    Same-device cache growth by another owner is excluded by the existing whole-
    GPU lease. The affine lock protects only the metadata snapshot; it is never
    held while a grid is built. Other-device evictions can only reduce retention.
    """
    from XTA.inference import _AFFINE_GRID_CACHE, _AFFINE_GRID_CACHE_LOCK
    torch, device = engine.torch, str(engine.device)
    opaque = ('_tilted_plans','_fused_azimuthal_taps','_spherical_direction_cache',
        '_spherical_rotation_cache','_standalone_render_meta','_standalone_render_meta_ref',
        '_azimuthal_texture_ref','_fused_volume_ref')
    for name in opaque:
        value = getattr(engine,name,None)
        if value is None or isinstance(value,(dict,list,tuple,set,frozenset)) and len(value)==0:
            continue
        if isinstance(value,(str,bytes,int,float,bool)):
            continue  # Proven storage-free metadata.
        raise BatchDeclined('opaque_retained_cache',dict(cache_field=name))
    def storage_key(tensor):
        storage=tensor.untyped_storage()
        return str(tensor.device),int(storage.data_ptr()),int(storage.nbytes())
    source=getattr(engine,'_volume_gpu',None)
    if not isinstance(source,torch.Tensor) or str(source.device)!=device:
        raise BatchDeclined('unproven_source_storage')
    source_key=storage_key(source)
    flat=getattr(engine,'_volume_flat',None)
    if flat is not None and (not isinstance(flat,torch.Tensor) or storage_key(flat)!=source_key):
        raise BatchDeclined('unproven_source_alias')
    storages=set()
    def visit(value):
        if isinstance(value,torch.Tensor):
            key=storage_key(value)
            if key[0]!=device:
                raise BatchDeclined('unexpected_cache_device')
            if key!=source_key and key[2]:
                storages.add(key)
        elif isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif value is None or isinstance(value,(str,bytes,int,float,bool)):
            return
        else:
            raise BatchDeclined('opaque_retained_cache_value')
    for name in ('_native_t_map_cache','_native_plane_cache','_native_u8_plane_cache','_fold_cache'):
        visit(getattr(engine,name,{}))
    with _AFFINE_GRID_CACHE_LOCK:
        for value in _AFFINE_GRID_CACHE.values():
            # Keys retain the requested spelling (e.g. 'cuda'); tensors expose
            # the resolved device (e.g. 'cuda:0'). Never infer storage ownership
            # from an alias key or trust an opaque value's claimed key.
            if not isinstance(value,torch.Tensor):
                raise BatchDeclined('opaque_affine_grid')
            if str(value.device)==device:
                visit(value)
    return dict(retained_cache_bytes=sum(key[2] for key in storages),
        retained_cache_storage_count=len(storages),device=device,
        cache_baseline_source='actual_engine_after_start')


def plan_batch(renderer, view, frames, inverse, *, output_origin_yx,
               output_height, output_width, retained_cache_bytes=None):
    """Metadata-only validation and stage-wise ledger; no source/GPU allocation."""
    from XTA import geometry
    from XTA.cuda_backend import _azimuthal_gather_stack_block
    from XTA.sam_gpu_rendering import crop_inverse
    from XTA.sam_transverse_cache_rendering import native_crop_bbox

    frames = tuple(islice(frames, 5))
    size = len(frames)
    def decline(reason):
        return BatchPlan(False, reason, size)
    if not 2 <= size <= 4:
        return decline('batch_size')
    if getattr(renderer, 'view', view) is not view:
        return decline('different_owner_view')
    source_t = int(renderer.source.shape[0])
    if tuple(getattr(renderer, 'source_t_window', (0, source_t))) != (0, source_t):
        return decline('partial_source_owner')
    if str(view.family) != 'azimuthal' or geometry.is_tilted_azimuthal_view(view):
        return decline('unsupported_view')
    if not geometry.azimuthal_resident_gpu_render_supported(view):
        return decline('unsupported_geometry')
    if any(isinstance(f, (bool, np.bool_)) for f in frames):
        return decline('invalid_frame')
    try:
        checked_frames = tuple(operator.index(f) for f in frames)
        height, width = operator.index(output_height), operator.index(output_width)
    except TypeError:
        return decline('noninteger_request')
    if len(set(checked_frames)) != size or any(not 0 <= f < int(view.num_slices) for f in checked_frames):
        return decline('invalid_or_duplicate_frame')
    native_h, native_w = int(view.src_h), int(view.src_w)
    if min(native_h, native_w, height, width) < 1 or max(native_h, native_w, height, width) > 8192:
        return decline('unqualified_shape')
    try:
        matrix = crop_inverse(inverse, output_origin_yx)
    except (ValueError, TypeError):
        return decline('invalid_affine')
    if np.max(np.abs(matrix[:, :2])) > 4. or np.max(np.abs(matrix[:, 2])) > 32768.:
        return decline('unqualified_affine')
    bbox = native_crop_bbox((0, 0, height, width), matrix, (native_h, native_w))
    if bbox is None:
        return decline('empty_native_roi')
    y0, x0, y1, x1 = bbox
    roi_h, roi_w = y1-y0, x1-x0
    area, pixels, roi_area = native_h*native_w, height*width, roi_h*roi_w
    if 2*roi_area >= area:
        return decline('full_plane_route')
    native_t = int(renderer.source.shape[0])
    logical_t = int(renderer.logical_t)
    base = geometry.azimuthal_base_view_name(view)
    stack_len = native_t if base == 'transverse' else int(geometry.azimuthal_stack_length(view))
    fold = native_h != stack_len or base == 'transverse' and logical_t != stack_len
    selected_rows = roi_h*(2 if fold else 1)
    scalar_block = 512 if base == 'transverse' else _azimuthal_gather_stack_block(
        stack_len, native_h, native_w, logical_t)
    ledger=batch_workspace_ledger(batch_size=size,native_shape_yx=(native_h,native_w),
        logical_t=logical_t,roi_shape_yx=(roi_h,roi_w),output_shape_yx=(height,width),
        selected_rows=selected_rows,scalar_row_block=scalar_block,
        scratch_bytes=int(renderer.scratch_bytes),
        retained_cache_bytes=0 if retained_cache_bytes is None else retained_cache_bytes)
    if retained_cache_bytes is None:
        # Zero is only a lower bound used to prove impossibility, never admission.
        reason=('cache_measurement_required' if ledger['eligible'] else
            'gather_iteration_regression_even_with_empty_cache'
            if ledger['reason']=='gather_iteration_regression' else 'scratch_bound_even_with_empty_cache')
        ledger.update(eligible=False,reason=reason,retained_cache_bytes=None,total_device_bound_bytes=None)
    return BatchPlan(batch_size=size,bbox_yx=tuple(bbox),selected_rows=selected_rows,
        virtual_t=native_t!=logical_t,**ledger)


def choose_batch_prefix(renderer, view, frames, inverse, **crop):
    """Largest budget-fitting prefix that does not add native gather iterations."""
    frames = tuple(islice(frames, 4))
    for count in range(min(4, len(frames)), 1, -1):
        plan = plan_batch(renderer, view, frames[:count], inverse, **crop)
        if plan.eligible:
            return plan
    return BatchPlan(False, 'scalar_tail_or_ineligible', min(1, len(frames)))


def project_native_batch(engine, view, frames, bbox_yx, row_block):
    """Existing scalar native arithmetic with a leading independent batch axis."""
    from XTA import geometry
    torch, volume = engine.torch, engine._volume_gpu
    frames = tuple(islice(frames, 5))
    if not 1 <= len(frames) <= 4:
        raise ValueError('A borrowed batch call requires one to four frames')
    y0, x0, y1, x1 = map(int, bbox_yx)
    rows, columns = int(view.src_h), int(view.src_w)
    if (not frames or not 0 <= y0 < y1 <= rows or not 0 <= x0 < x1 <= columns
            or int(row_block) < 1):
        raise ValueError('Invalid native batch rectangle or row block')
    base = geometry.azimuthal_base_view_name(view)
    stack_len = int(volume.shape[0]) if base == 'transverse' else int(geometry.azimuthal_stack_length(view))
    pairs = []
    for frame in frames:
        angle = float(view.azimuths_deg[int(frame)])
        pair = engine._try_sam_azimuthal_taps_gpu(view, angle, x0, x1)
        if pair is None:
            flat, weights = engine._azimuthal_taps_gpu(view, angle)
            pair = flat[x0:x1], weights[x0:x1]
        pairs.append(pair)
    flat = torch.stack([pair[0] for pair in pairs])
    weights = torch.stack([pair[1] for pair in pairs])
    pairs.clear()
    # Drop the last reference slices too before gathering large row blocks.
    pair = None
    fold = rows != stack_len or base == 'transverse' and int(engine._logical_t) != stack_len
    if fold:
        lower, upper, alpha = engine._azimuthal_fold_indices(
            stack_len, rows, logical_t=int(engine._logical_t) if base == 'transverse' else None)
        selected_rows = torch.cat((lower[y0:y1], upper[y0:y1]))
    else:
        selected_rows = torch.arange(y0, y1, device=engine.device, dtype=torch.int64)
    projected = torch.empty((len(frames), int(selected_rows.numel()), x1-x0),
                            dtype=torch.float32, device=engine.device)
    if base == 'transverse':
        flat_volume = volume.view(stack_len, -1)
    else:
        plane_width = int(geometry.azimuthal_plane_shape(view)[1])
        logical_taps = torch.div(flat, plane_width, rounding_mode='floor')
        source_columns = torch.remainder(flat, plane_width)
        source_taps, next_taps, temporal_alpha = logical_taps, None, None
        if int(volume.shape[0]) != int(engine._logical_t):
            lower_t, upper_t, alpha_t = engine._native_t_indices()
            source_taps, next_taps = lower_t[logical_taps], upper_t[logical_taps]
            temporal_alpha = alpha_t[logical_taps]
    for start in range(0, int(selected_rows.numel()), int(row_block)):
        stop = min(int(selected_rows.numel()), start+int(row_block))
        stack = selected_rows[start:stop].view(1, -1, 1, 1)
        if base == 'transverse':
            samples = flat_volume[stack, flat[:, None]].to(torch.float32)
        else:
            if base == 'sagittal':
                indices = (source_taps[:, None], stack, source_columns[:, None])
            elif base == 'coronal':
                indices = (source_taps[:, None], source_columns[:, None], stack)
            else:
                raise ValueError(f'Unsupported resident Azimuthal base: {base}')
            samples = volume[indices].to(torch.float32)
            if next_taps is not None:
                following = volume[(next_taps[:, None], *indices[1:])].to(torch.float32)
                samples = torch.lerp(samples, following, temporal_alpha[:, None]).round_().clamp_(0.0, 255.0)
                del following
        projected[:, start:stop] = (samples*weights[:, None]).sum(dim=-1)
        del samples, stack
    if not fold:
        return projected
    height = y1-y0
    return projected[:, :height]*(1.0-alpha[y0:y1])[None]+projected[:, height:]*alpha[y0:y1][None]


def warp_native_batch(engine, native_u8, matrix, out_h, out_w):
    """Unchanged single-grid coordinates, expanded across independent images."""
    from XTA.cuda_backend import _affine_theta_from_dst_to_src, _get_cached_affine_grid
    torch = engine.torch
    if native_u8.ndim != 3 or native_u8.dtype != torch.uint8:
        raise ValueError('Batch warp requires BxHxW uint8 native images')
    batch, rows, columns = map(int, native_u8.shape)
    matrix = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    identity = np.asarray([[1., 0., 0.], [0., 1., 0.]], np.float32)
    if rows == int(out_h) and columns == int(out_w) and np.allclose(matrix, identity, rtol=0., atol=2e-6):
        return native_u8.contiguous()
    theta = _affine_theta_from_dst_to_src(matrix, rows, columns, int(out_h), int(out_w))
    grid = _get_cached_affine_grid(theta, int(out_h), int(out_w), engine.device)
    warped = engine.F.grid_sample(native_u8.to(torch.float32).reshape(batch, 1, rows, columns),
        grid.expand(batch, -1, -1, -1), mode='bilinear', padding_mode='zeros',
        align_corners=False).reshape(batch, int(out_h), int(out_w))
    return warped.round_().clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def render_compatible_batch(renderer, view, frames, inverse, *, output_origin_yx,
                            output_height, output_width, consume=None):
    """Borrow a live owner; unsupported metadata uses its ordinary scalar route.

    The caller owns admission, ordered private-mmap copies and final publication.
    With consume(position,image), copy one result at a time into that private
    sink; do not retain host views here. The collecting default is intended for
    external qualification only, and its views share one fully charged BHW base.
    A launched failure is never retried here. Caller must close the renderer via
    its established finally block, retaining source/lease until fencing succeeds.
    """
    from XTA.sam_gpu_rendering import crop_inverse, SamGpuRenderingUnavailable
    frames = tuple(islice(frames, 5))
    if not 1 <= len(frames) <= 4:
        raise ValueError('A borrowed batch call requires one to four frames')
    renderer.context._check_image_lifetime()
    # Keep caller-owned matrix/origin mutation outside the admitted request.
    inverse = np.asarray(inverse,dtype=np.float32).reshape(2,3).copy()
    output_origin_yx = tuple(output_origin_yx)
    plan = plan_batch(renderer, view, frames, inverse, output_origin_yx=output_origin_yx,
                      output_height=output_height, output_width=output_width)
    def make_receipt(current):
        receipt=asdict(current)
        receipt.update(requested_batch_size=len(frames),
            effective_batch_size=len(frames) if current.eligible else 0,
            batch_path_used=current.eligible, fallback_reason=None if current.eligible else current.reason,
            planned_device_scratch_peak_bytes=max(current.projection_bytes,current.warp_bytes),
            planned_host_scratch_peak_bytes=current.host_bytes,
            scratch_bound_semantics='new live batch tensors; retained caches are separately added to G; source and allocator reserve excluded')
        return receipt
    receipt=make_receipt(plan)
    if plan.reason!='cache_measurement_required':
        if consume is None:
            raise BatchDeclined(plan.reason,receipt)
        for position, frame in enumerate(frames):
            image = renderer.render(view, frame, inverse, output_origin_yx=output_origin_yx,
                output_height=output_height, output_width=output_width)
            try:
                consume(position, image)
            finally:
                image=None
            renderer.context._check_image_lifetime()
        receipt.update(effective_batch_size=1,scalar_fallback_used=True,
            completed_images=len(frames),host_result_mode='sink')
        return BatchResult((),receipt)
    native = patch = pixels = None
    sink_error = False
    try:
        engine = renderer._start()
        if (tuple(engine._volume_gpu.shape)!=tuple(renderer.source.shape)
                or int(engine._logical_t)!=int(renderer.logical_t)):
            raise BatchDeclined('actual_source_geometry_changed')
        cache=retained_cache_snapshot(engine)
        # Actual-device storage is measured only after source/lease selection;
        # no batch tensor or sink write has happened at this point.
        plan=plan_batch(renderer,view,frames,inverse,output_origin_yx=output_origin_yx,
            output_height=output_height,output_width=output_width,
            retained_cache_bytes=cache['retained_cache_bytes'])
        receipt=make_receipt(plan)
        receipt.update(cache)
        if not plan.eligible:
            raise BatchDeclined(plan.reason,receipt)
        matrix = crop_inverse(inverse, output_origin_yx)
        torch = renderer.torch
        with torch.cuda.device(engine.device), torch.cuda.stream(engine._stream):
            started = time.monotonic()
            patch = project_native_batch(engine, view, frames, plan.bbox_yx, plan.row_block)
            native = torch.zeros((len(frames), int(view.src_h), int(view.src_w)),
                                 dtype=torch.uint8, device=engine.device)
            y0, x0, y1, x1 = plan.bbox_yx
            native[:, y0:y1, x0:x1] = patch.round().clamp_(0, 255).to(torch.uint8)
            patch = None
            pixels = warp_native_batch(engine, native, matrix, output_height, output_width)
            native = None
            enqueued = time.monotonic()
            if consume is None:
                images = pixels.cpu().numpy()
                result = tuple(images[i] for i in range(len(frames)))
            else:
                for position in range(len(frames)):
                    renderer.context._check_image_lifetime()
                    image = pixels[position].cpu().numpy()
                    try:
                        consume(position, image)
                    except BaseException:
                        sink_error = True
                        raise
                    finally:
                        image = None
                result = ()
            completed = time.monotonic()
        engine.clear_native_plane_cache()
        renderer.context._check_image_lifetime()
        receipt.update(enqueue_host_seconds=enqueued-started,
                       copy_wait_host_seconds=(completed-enqueued if consume is None else None),
                       copy_and_sink_host_seconds=(completed-enqueued if consume is not None else None),
                       completed_images=len(frames), host_result_mode='collected' if consume is None else 'sink')
        return BatchResult(result, receipt)
    except BatchDeclined:
        raise
    except (RuntimeError, MemoryError) as error:
        if sink_error:
            raise  # A private-cache write error is not a CUDA capability miss.
        renderer._render_failed = True
        renderer.context._check_image_lifetime()
        if isinstance(error, SamGpuRenderingUnavailable):
            raise
        raise SamGpuRenderingUnavailable(str(error)) from error
    finally:
        native = patch = pixels = None
