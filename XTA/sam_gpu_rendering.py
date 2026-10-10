"""Idle-GPU SAM image preparation through the existing TTA intensity renderer.

Only demanded grayscale crops leave the device. One existing compute lease
fences a transaction; source, projection and grid allocations retire before
that lease is returned. CPU preparation remains the admission fallback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import threading
import time
import traceback
import weakref

import numpy as np


SAM_GPU_IMAGE_CONTRACT = 'xta.sam_tta_cuda_gray8_crop/1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LIVE_IMAGES = {}
_LIVE_IMAGES_LOCK = threading.RLock()


def _record_image_timings(**timings):
    try:
        from .runtime import runtime_telemetry
        for name, seconds in timings.items():
            runtime_telemetry().add('sam.gpu_images.'+name, seconds)
    except Exception:
        pass  # Timing cannot invalidate pixels or completed ownership transfers.


def register_live_image(context, reference, geometry_identity, sampling):
    """Authenticate live producer descriptors; serialized fields confer no trust."""
    token = id(reference)
    def retire(_reference):
        with _LIVE_IMAGES_LOCK:
            _LIVE_IMAGES.pop(token, None)
    with _LIVE_IMAGES_LOCK:
        proof = (str(geometry_identity), str(sampling['contract']))
        context._image_sampling_proofs[reference.identity_sha256] = proof
        _LIVE_IMAGES[token] = (weakref.ref(reference, retire), weakref.ref(context), proof,
            dict(sampling, source_geometry_identity_sha256=str(geometry_identity)))


def live_image_sampling(reference):
    with _LIVE_IMAGES_LOCK:
        entry = _LIVE_IMAGES.get(id(reference))
        return dict(entry[3]) if entry is not None and entry[0]() is reference else None


def same_live_image_geometry(pinned_identity, reference):
    """Permit an authenticated CPU/GPU retry, keeping byte/feature keys distinct."""
    from .sam_canvas_rendering import CANONICAL_CROP_RENDER_CONTRACT
    with _LIVE_IMAGES_LOCK:
        entry = _LIVE_IMAGES.get(id(reference))
        if entry is None or entry[0]() is not reference:
            return False
        context = entry[1]()
        previous = None if context is None else context._image_sampling_proofs.get(str(pinned_identity))
        approved = {CANONICAL_CROP_RENDER_CONTRACT, SAM_GPU_IMAGE_CONTRACT}
        return (context is not None and not context._closed and previous is not None and previous[0] == entry[2][0]
            and previous[1] in approved and entry[2][1] in approved)


def record_live_image(metadata, reference):
    """Portable pixel identity and sampler provenance for each actual input."""
    sampling = live_image_sampling(reference)
    if sampling is not None:
        metadata.setdefault('image_sampling_backend', sampling)
        sources = dict(metadata.get('image_sampling_sources', {}))
        sources[reference.identity_sha256] = sampling
        metadata['image_sampling_sources'] = sources
    return sampling


class SamGpuRenderingUnavailable(RuntimeError):
    """A private GPU image transaction must be discarded and rebuilt on CPU."""


def clear_image_error_frames(error):
    """Failed futures may retain nested GPU sampler locals through causes."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


def crop_inverse(inverse, origin_yx):
    """Keep global native coordinates when translating a local crop raster."""
    matrix = np.asarray(inverse, dtype=np.float32).reshape(2, 3).copy()
    if not np.isfinite(matrix).all():
        raise ValueError('SAM crop affine must be finite')
    y, x = map(int, origin_yx)
    if min(y, x) < 0:
        raise ValueError('SAM crop origin must be nonnegative')
    matrix[:, 2] = (matrix.astype(np.float64)[:, 2]
        + matrix.astype(np.float64)[:, :2] @ np.array([x, y], np.float64)).astype(np.float32)
    return matrix


def radial_source_t_window(view, source_shape, logical_t, frames):
    """Enclose full native Radial patch taps within the qualified axis envelope."""
    from .geometry import cartesian_view_axis_spec
    from .geometry_quality import spherical_fp32_shape_eligible
    from .cylindrical_geometry import shell_coordinates
    native_t, height, width = map(int, source_shape)
    full = (0, native_t)
    try:
        frames = tuple(frames)
        if (str(view.family) != 'radial' or not frames
                or not spherical_fp32_shape_eligible((native_t, logical_t, height, width,
                                                      view.src_h, view.src_w, view.num_slices))
                or len(view.radial_radii) != int(view.num_slices)
                or (int(view.full_t), int(view.full_h), int(view.full_w)) != (logical_t, height, width)
                or any(isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer))
                       or not 0 <= frame < len(view.radial_radii) for frame in frames)):
            return full
        spec = cartesian_view_axis_spec(str(view.radial_base_view), logical_t, height, width)
        if (float(view.center_x) != (int(spec['src_w'])-1)/2.
                or float(view.center_y) != (int(spec['src_h'])-1)/2.
                or not 0 <= int(view.radial_height_origin) < int(spec['num_slices'])
                or not np.isfinite(float(view.radial_arc_origin)) or float(view.radial_arc_origin) < 0
                or bool(view.radial_tilted_source) and (
                    str(view.tilt_direction) not in {'vertical', 'horizontal'}
                    or not np.isfinite(float(view.tilt_angle_deg)) or abs(float(view.tilt_angle_deg)) > 45)):
            return full
        radii = np.asarray([view.radial_radii[int(frame)] for frame in frames], np.float64)
        if (not np.isfinite(radii).all() or np.any(radii < 1.)
                or np.any(radii > (min(int(spec['src_h']), int(spec['src_w']))-1)/2.)
                or np.any(float(view.radial_arc_origin) >= 2*np.pi*radii)):
            return full
        columns = np.arange(int(view.src_w), dtype=np.float64)[None, :]
        endpoints = np.array([0, int(view.src_h)-1], np.float64)[:, None]
        low, high = np.inf, -np.inf
        for frame in frames:
            # T is affine in height; enumerate every angular pixel, not just corners.
            tt, _yy, _xx, _valid = shell_coordinates(view, int(frame), x=columns, y=endpoints)
            low, high = min(low, float(tt.min())), max(high, float(tt.max()))
            if low <= 0 and high >= int(logical_t)-1:
                return full
        if not np.isfinite((low, high)).all():
            return full
        # Within the <=4096-axis, bounded radius/shear recipe, outward integer
        # guards include CPU/CUDA FP64 trig phase and both logical outer taps.
        first, stop = max(0, int(np.floor(low))-2), min(int(logical_t), int(np.ceil(high))+3)
        if first >= stop:
            return full
        logical = np.array([first, stop-1], np.float64)
        positions = (logical+.5)*(native_t/float(logical_t))-.5
        lower = np.clip(np.floor(positions).astype(np.int64), 0, native_t-1)
        upper = np.minimum(lower+1, native_t-1)
        return max(0, int(lower.min())-2), min(native_t, int(upper.max())+3)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return full


class SamGpuCropRenderer:
    def __init__(self, context, view, source, logical_t, lease, torch, scratch_bytes, *, required_gpu=0,
                 source_t_window=None, source_frames=()):
        self.context, self.view, self.source = context, view, source
        self.logical_t, self.lease, self.torch = int(logical_t), lease, torch
        self.scratch_bytes = int(scratch_bytes)
        self.required_gpu = int(required_gpu)
        self.engine = None
        self.device_index = int(lease.device_index) if lease is not None else None
        self._image_burst_count = 1
        self._gpu_wait_deadline = None
        self._gpu_prefetch_cancel = None
        self._uploaded_source_key = None
        self._render_failed = False
        self.source_t_window = tuple(source_t_window or (0, int(source.shape[0])))
        self.source_frames = frozenset(map(int, source_frames))
        start, stop = self.source_t_window
        if (not 0 <= start < stop <= int(source.shape[0]) or self.source_t_window != (0, int(source.shape[0]))
                and (str(view.family) != 'radial' or not self.source_frames)):
            raise ValueError('SAM partial source requires a frozen Radial frame window')
        self.source_planned_bytes = (stop-start)*int(source.shape[1])*int(source.shape[2])
        self.resident_t_window = self.source_t_window
        self.source_resident_bytes = self.source_planned_bytes

    def _source_handoff_key(self):
        """Bind the selected immutable representation, not merely its shape."""
        from .runtime import _interpolation_array_backing_path
        backing = _interpolation_array_backing_path(self.source)
        stamp = None
        if backing is not None:
            path = Path(backing).resolve(strict=True)
            stat = path.stat()
            stamp = str(path), int(stat.st_size), int(stat.st_mtime_ns)
        return (id(self.context), self.context.source_identity, id(self.source),
            int(self.source.__array_interface__['data'][0]), tuple(self.source.shape),
            str(self.source.dtype), self.logical_t, stamp, self.resident_t_window)

    def sampling_identity(self):
        from . import cuda_backend, spherical_cuda
        from .geometry_quality import geometry_quality_request_record
        return dict(contract=SAM_GPU_IMAGE_CONTRACT, backend='cuda',
            implementation_sha256=IMPLEMENTATION_SHA256,
            tta_renderer_sha256=hashlib.sha256(Path(cuda_backend.__file__).read_bytes()).hexdigest(),
            spherical_renderer_sha256=hashlib.sha256(Path(spherical_cuda.__file__).read_bytes()).hexdigest(),
            torch_version=str(self.torch.__version__), cuda_version=str(self.torch.version.cuda),
            geometry_quality=geometry_quality_request_record(),
            azimuthal_projector='resident_torch_no_second_source_texture',
            tilted_projector='tta_fused_crop_with_quantized_native_taps_and_torch_fallback',
            pixel_contract='registered_TTA_CUDA_intensity_then_gray8',
            native_shape_tyx=list(self.source.shape), logical_t=self.logical_t,
            planned_source_t_window=list(self.source_t_window),
            temporal_sampling=('existing_processing_t' if int(self.source.shape[0])==self.logical_t
                else 'tta_center_aligned_virtual_processing_t_gray8'))

    def _start(self):
        self.context._check_image_lifetime()
        if self.engine is not None:
            return self.engine
        if self.resident_t_window != self.source_t_window:
            self.required_gpu += self.source_planned_bytes-self.source_resident_bytes
            self.resident_t_window = self.source_t_window
            self.source_resident_bytes = self.source_planned_bytes
        from .cuda_backend import _GpuWorkerRenderEngine
        from .media import wait_for_volume_ready
        from .runtime import _interpolation_array_backing_path
        wait_for_volume_ready(self.source)
        if self.lease is None:
            from .runtime import runtime_telemetry
            self._gpu_wait_deadline = None  # Only the FIFO head starts its 30s drain turn.
            self._gpu_prefetch_cancel = getattr(self.context._resource_local, 'image_prefetch_cancel', None)
            self.context._queue_gpu_image(self)
            busy, remaining = False, 30.
            try:
                while self.lease is None:
                    self.context._check_image_lifetime()
                    # The context grants only its concrete FIFO head. A
                    # handoff may already have assigned this exact owner.
                    lease = self.context._try_gpu_image_lease(self, self.torch)
                    if lease is None:
                        busy = True
                    else:
                        self.lease, self.device_index = lease, int(lease.device_index)
                        if self.engine is not None:
                            break  # Source/lease were handed over atomically.
                        admitted = False
                        try:
                            free, _total = self.torch.cuda.mem_get_info(self.device_index)
                            admitted = int(free) >= self.required_gpu
                        except RuntimeError:
                            pass
                        finally:
                            if not admitted:
                                self.context._release_sam_compute_lease(lease)
                                lease.release()
                                # Keep the returned lease marker until the
                                # context atomically rejects/clears it; another
                                # renderer must not hand over into this gap.
                                more_devices = self.context._reject_gpu_image_device(self, lease)
                        if admitted:
                            break
                        if more_devices:
                            continue
                        busy = False
                    deadline = self._gpu_wait_deadline
                    remaining = None if deadline is None else deadline-time.monotonic()
                    if not busy or remaining is not None and remaining <= 0:
                        break
                    started = time.monotonic()
                    self.context._cancel.wait(.05 if remaining is None else min(.05, remaining))
                    runtime_telemetry().add('sam.gpu_images.admission_wait_seconds', time.monotonic()-started)
                self.context._check_image_lifetime()
                if self.lease is None:
                    if busy and remaining is not None and remaining <= 0:
                        runtime_telemetry().add('sam.gpu_images.admission_wait_timeouts', 1)
                    raise SamGpuRenderingUnavailable('No idle GPU with admitted SAM source/render headroom')
            finally:
                self.context._finish_gpu_image_wait(self, granted=self.lease is not None)
        if self.engine is not None:
            return self.engine
        started = time.monotonic()
        engine = self.engine = _GpuWorkerRenderEngine(f'cuda:{self.device_index}')
        _record_image_timings(renderer_init_host_seconds=time.monotonic()-started)
        source_key = self._source_handoff_key()
        backing = _interpolation_array_backing_path(self.source)
        if self.source_t_window != (0, int(self.source.shape[0])):
            start, stop = self.source_t_window
            mode = engine.ensure_volume_array(self.source[start:stop], identity=str(source_key))
            engine._radial_source_t = (int(self.source.shape[0]), start)
            engine._logical_t = self.logical_t
            engine._native_t_map_cache.clear()
        elif backing is not None:
            mode = engine.ensure_volume(str(backing), self.source.shape, 'uint8',
                resize_to_t=self.logical_t)
        elif int(self.source.shape[0]) == self.logical_t:
            mode = engine.ensure_volume_array(self.source, identity=self.context.source_identity)
        else:
            raise SamGpuRenderingUnavailable('Native-T GPU rendering requires the existing source backing')
        # Optional CUDA textures own a second complete source copy. Use the
        # established pointer/Torch projector under the single-source budget.
        engine._azimuthal_texture_admitted = False
        if mode != 'resident':
            raise SamGpuRenderingUnavailable('SAM source did not fit GPU residency admission')
        self._uploaded_source_key = source_key
        _record_image_timings(**{'source_'+name:seconds
            for name,seconds in getattr(engine, '_source_residency_timings', {}).items()})
        from .runtime import runtime_telemetry
        runtime_telemetry().add('sam.gpu_images.gpu_admissions', 1)
        runtime_telemetry().add('sam.gpu_images.source_uploads', 1)
        runtime_telemetry().add('sam.gpu_images.source_upload_bytes', self.source_resident_bytes)
        if self.source_resident_bytes < int(self.source.nbytes):
            runtime_telemetry().add('sam.gpu_images.source_window_uploads', 1)
            runtime_telemetry().add('sam.gpu_images.source_window_upload_bytes_avoided',
                int(self.source.nbytes)-self.source_resident_bytes)
            runtime_telemetry().gauge('sam.gpu_images.source_window_resident_bytes', self.source_resident_bytes)
            runtime_telemetry().gauge('sam.gpu_images.source_window_headroom_bytes_saved',
                int(self.source.nbytes)-self.source_resident_bytes)
        return engine

    def render(self, view, frame, inverse, *, output_origin_yx, output_height, output_width):
        from .geometry import is_tilted_view
        self.context._check_image_lifetime()
        height, width = int(output_height), int(output_width)
        if min(height, width) <= 0 or height*width > self.scratch_bytes:
            raise SamGpuRenderingUnavailable('SAM crop exceeds its admitted render workspace')
        if not 0 <= int(frame) < int(view.num_slices):
            raise IndexError('SAM GPU frame is outside its physical view')
        if self.source_t_window != (0, int(self.source.shape[0])) and (
                view is not self.view or int(frame) not in self.source_frames):
            raise SamGpuRenderingUnavailable('SAM partial source does not cover this physical frame demand')
        native = pixels = None
        try:
            engine = self._start()
            matrix = crop_inverse(inverse, output_origin_yx)
            with self.torch.cuda.device(engine.device), self.torch.cuda.stream(engine._stream):
                started = time.monotonic()
                if is_tilted_view(view):
                    pixels = engine.render_tilted_grid_resident(view, matrix, int(frame), height, width)
                    pixels = pixels.round().clamp_(0, 255).to(self.torch.uint8)
                else:
                    native = engine._render_native_plane(view, int(frame))
                    native = native.round().clamp_(0, 255).to(self.torch.uint8)
                    pixels = engine.warp_native_uint8_frame(native, matrix, height, width)
                # This blocking copy completes the render before immutable
                # publication or the next crop may reuse projection storage.
                enqueued = time.monotonic()
                image = pixels.cpu().numpy()
                _record_image_timings(render_enqueue_host_seconds=enqueued-started,
                    crop_copy_wait_host_seconds=time.monotonic()-enqueued)
            engine.clear_native_plane_cache()
            self.context._check_image_lifetime()
            return image
        except (RuntimeError, MemoryError) as error:
            self._render_failed = True
            self.context._check_image_lifetime()
            if isinstance(error, SamGpuRenderingUnavailable):
                raise
            raise SamGpuRenderingUnavailable(str(error)) from error
        finally:
            native = pixels = None

    def close(self):
        if self.lease is None:
            return
        from .backprojection import _trim_main_process_cuda_device
        try:
            if (self.engine is not None and not self._render_failed
                    and sys.exc_info()[1] is None and self._try_handoff()):
                return
            started = time.monotonic()
            real_engine = False
            if self.engine is not None:
                from .cuda_backend import _GpuWorkerRenderEngine
                real_engine = type(self.engine) is _GpuWorkerRenderEngine
                self.engine.release_inference_assets()
                mapping = self.engine._volume_mm
                if isinstance(mapping, np.memmap):
                    mapping._mmap.close()
                self.engine._volume_mm = None
                self.engine = None
            from .inference import _AFFINE_GRID_CACHE, _AFFINE_GRID_CACHE_LOCK
            with _AFFINE_GRID_CACHE_LOCK:
                for key in tuple(_AFFINE_GRID_CACHE):
                    if key[0] == f'cuda:{self.device_index}':
                        _AFFINE_GRID_CACHE.pop(key)
            _trim_main_process_cuda_device(self.torch,
                self.torch.device(f'cuda:{self.device_index}'), desc='SAM image preparation',
                repeat_garbage_collection=not (real_engine and not self._render_failed
                    and sys.exc_info()[1] is None and not self.context._cancel.is_set()
                    and self not in self.context._unsettled_image_renderers))
            self.context._release_sam_compute_lease(self.lease)
            # A pre-start global stage is not registered as predictor compute.
            self.lease.release()
            self.lease = None
            self.context._finish_gpu_image(self)
            _record_image_timings(retirement_host_seconds=time.monotonic()-started)
        except BaseException:
            with self.context._idle:
                if self not in self.context._unsettled_image_renderers:
                    self.context._unsettled_image_renderers.append(self)
            from .sam_integration import _retain_unsettled_context
            _retain_unsettled_context(self.context)
            self.context.cancel('SAM GPU image preparation could not prove CUDA retirement')
            raise
        with self.context._idle:
            if self in self.context._unsettled_image_renderers:
                self.context._unsettled_image_renderers.remove(self)

    def _try_handoff(self):
        """Share one upload with an already-ready FIFO head, never idle cache it."""
        engine = self.engine
        if self._image_burst_count >= 2 and not self.context._can_extend_gpu_image_burst():
            return False
        with self.context._gpu_lease_lock:
            if not self.context._gpu_image_waiters or self.context._cancel.is_set() or self.context._closed:
                return False
        if (self._uploaded_source_key is None or getattr(engine, '_volume_gpu', None) is None
                or getattr(engine, '_azimuthal_texture_ref', None) is not None):
            return False
        try:
            if self._source_handoff_key() != self._uploaded_source_key:
                return False
        except (OSError, ValueError):
            return False
        # These are view workspaces, not retained source storage. Complete the
        # stream before dropping all aliases; keep only source and its T maps.
        started = time.monotonic()
        from .cuda_backend import _GpuWorkerRenderEngine
        real_engine = type(engine) is _GpuWorkerRenderEngine
        engine._stream.synchronize()
        engine.clear_native_plane_cache()
        from .spherical_cuda import clear_spherical_render_cache
        clear_spherical_render_cache(engine)
        for name in ('_tilted_plans', '_fold_cache', '_fused_azimuthal_taps'):
            getattr(engine, name).clear()
        engine._fused_volume_ref = None
        engine._standalone_render_meta = engine._standalone_render_meta_ref = None
        from .inference import _AFFINE_GRID_CACHE, _AFFINE_GRID_CACHE_LOCK
        with _AFFINE_GRID_CACHE_LOCK:
            for key in tuple(_AFFINE_GRID_CACHE):
                if key[0] == f'cuda:{self.device_index}':
                    _AFFINE_GRID_CACHE.pop(key)
        owned_retirement = (real_engine and not self._render_failed
            and sys.exc_info()[1] is None and not self.context._cancel.is_set()
            and self not in self.context._unsettled_image_renderers)
        with self.context._gpu_lease_lock:
            head = self.context._gpu_image_waiters[0] if self.context._gpu_image_waiters else None
            projector_bytes = (max(0, head.required_gpu-head.source_resident_bytes)
                if head is not None else None)
        free = None
        if owned_retirement and projector_bytes is not None:
            try:
                free, _total = self.torch.cuda.mem_get_info(self.device_index)
            except Exception:
                pass
        # Driver-free bytes exclude uncollected aliases and cached allocations;
        # a fenced source transfer needs no process-wide trim when they suffice.
        trimmed = free is None or int(free) < projector_bytes
        if trimmed:
            from .backprojection import _trim_main_process_cuda_device
            _trim_main_process_cuda_device(self.torch,
                self.torch.device(f'cuda:{self.device_index}'), desc='SAM source handoff',
                repeat_garbage_collection=not owned_retirement)
            free, _total = self.torch.cuda.mem_get_info(self.device_index)
        source_bytes = self.source_resident_bytes
        saved_upload_bytes = 0

        def transfer(next_renderer):
            nonlocal saved_upload_bytes
            if (next_renderer.context is not self.context or next_renderer.lease is not None
                    or next_renderer.engine is not None or next_renderer._gpu_wait_deadline is None
                    or time.monotonic() >= next_renderer._gpu_wait_deadline
                    or (next_renderer._gpu_prefetch_cancel is not None
                        and next_renderer._gpu_prefetch_cancel.is_set())):
                return False
            try:
                if next_renderer._source_handoff_key()[:-1] != self._uploaded_source_key[:-1]:
                    return False
            except (OSError, ValueError):
                return False
            start, stop = next_renderer.source_t_window
            if not self.resident_t_window[0] <= start < stop <= self.resident_t_window[1]:
                return False
            projector_bytes = max(0, next_renderer.required_gpu-next_renderer.source_resident_bytes)
            if int(free) < projector_bytes:
                return False
            saved_upload_bytes = next_renderer.source_planned_bytes
            next_renderer.resident_t_window = self.resident_t_window
            next_renderer.source_resident_bytes = source_bytes
            next_renderer.required_gpu = source_bytes+projector_bytes
            next_renderer.engine, next_renderer.lease = engine, self.lease
            next_renderer.device_index = self.device_index
            next_renderer._image_burst_count = self._image_burst_count+1
            next_renderer._uploaded_source_key = self._uploaded_source_key
            self.engine = self.lease = None
            return True

        handed = self.context._try_gpu_image_handoff(self, transfer)
        _record_image_timings(handoff_prepare_host_seconds=time.monotonic()-started)
        if handed:
            try:
                from .runtime import runtime_telemetry
                runtime_telemetry().add('sam.gpu_images.source_handoffs', 1)
                runtime_telemetry().add('sam.gpu_images.source_upload_bytes_saved', saved_upload_bytes)
                if not trimmed:
                    runtime_telemetry().add('sam.gpu_images.source_handoff_trim_skips', 1)
                if self._image_burst_count >= 2:
                    runtime_telemetry().add('sam.gpu_images.idle_sdk_extra_source_handoffs', 1)
            except Exception:
                pass  # Diagnostics cannot undo the recipient's live ownership.
        return handed


def try_gpu_crop_renderer(context, view, shape, prepared_plan):
    """Defer compute admission until pixels are needed; reject unsupported sources."""
    from .workspace import _env_flag, _env_int
    from .runtime import runtime_telemetry
    def decline(reason):
        runtime_telemetry().add('sam.gpu_images.cpu_admissions', 1)
        runtime_telemetry().gauge('sam.gpu_images.last_cpu_reason', reason)
        return None
    if not _env_flag('YOLO_TTA_SAM_GPU_IMAGES', True) or not getattr(context, 'gpu_image_ready', context.detector_retirement_ready):
        return decline('disabled_or_detector_assets_not_retired')
    if str(view.family) not in {'orthogonal', 'tilted', 'azimuthal', 'radial', 'spherical'}:
        return decline('unsupported_view_family')
    # A credited producer cannot park an idle GPU behind aggregate CPU image
    # admission. Legacy callers keep the established CPU provider.
    profile = getattr(context._resource_local, 'profile', None)
    if profile is None:
        return decline('no_live_parent_credit')
    from .sam_resources import validate_live_sam_resource_profile
    assigned = validate_live_sam_resource_profile(profile)
    required = dict(getattr(prepared_plan, 'frame_crop_bounds', {}) or {})
    if not required:
        return decline('no_compact_demand')
    payload = sum((int(box[2])-int(box[0]))*(int(box[3])-int(box[1])) for box in required.values())
    scratch = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
    if payload+scratch > int(assigned['base_non_cpu_allowance_bytes']):
        return decline('insufficient_parent_image_credit')
    source = context.source_volume
    logical_t = int(source.shape[0])
    if bool(getattr(source, '_is_lazy_processing_cube', False)):
        if source.materialized:
            source = source._array
        else:
            if source.streaming_backend:
                # Streaming preprocessing uses endpoint-aligned T; the TTA
                # virtual native-T renderer uses center-aligned coordinates.
                # GPU rounding permission does not authorize another grid.
                return decline('streamed_endpoint_processing_cube_not_materialized')
            if tuple(source.source.shape[1:]) != tuple(source.shape[1:]):
                return decline('unsupported_unmaterialized_source_xy_resize')
            source = source.source
    if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.dtype != np.uint8
            or not source.flags.c_contiguous):
        return decline('unsupported_source_layout')
    from .media import volume_readiness
    readiness = volume_readiness(source)
    if readiness is not None and not readiness._all_event.is_set():
        return decline('source_decode_incomplete')
    from .sam_canvas_rendering import cartesian_source_view
    if cartesian_source_view(source, view) is not None:
        return decline('bounded_cartesian_source_crop')
    from .runtime import _interpolation_array_backing_path
    backing = _interpolation_array_backing_path(source)
    if int(source.shape[0]) != logical_t and backing is None:
        return decline('native_t_source_has_no_backing')
    from .geometry import physical_view_name
    if (str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
            and tuple(shape) == tuple(source.shape) and backing is not None):
        return decline('exact_transverse_backing_reuse')
    started = time.monotonic()
    source_window = radial_source_t_window(view, source.shape, logical_t, required)
    source_bytes = (source_window[1]-source_window[0])*int(source.shape[1])*int(source.shape[2])
    _record_image_timings(source_window_plan_host_seconds=time.monotonic()-started)
    import torch
    if not torch.cuda.is_available():
        return decline('cuda_unavailable')
    from .cuda_backend import gpu_render_reserve_bytes
    # Conservative allowance covers native projector plans/taps, one output
    # grid and per-crop temporaries. No optional full-volume texture is kept.
    required_gpu = source_bytes+256*int(view.src_h)*int(view.src_w)+96*max(
        (int(box[2])-int(box[0]))*(int(box[3])-int(box[1])) for box in required.values())
    required_gpu += gpu_render_reserve_bytes()
    # No compute ownership while waiting for an image ticket or CPU credit.
    # The first missing crop waits for compute without holding a CUDA allocation.
    return SamGpuCropRenderer(context, view, source, logical_t, None, torch, scratch,
        required_gpu=required_gpu, source_t_window=source_window, source_frames=required)
