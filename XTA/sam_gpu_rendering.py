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
# Match the existing admission/drain turn, without charging source uploads to
# render service: a large source must not force retirement after every frame.
_IMAGE_COMPUTE_SERVICE_SECONDS = 30.0
_LIVE_IMAGES = {}
_LIVE_IMAGES_LOCK = threading.RLock()


def _record_image_timings(**timings):
    try:
        from .runtime import runtime_telemetry
        for name, seconds in timings.items():
            runtime_telemetry().add('sam.gpu_images.'+name, seconds)
    except Exception:
        pass  # Timing cannot invalidate pixels or completed ownership transfers.


def _record_image_phase(renderer, phase, *, frame=None, crop_shape=None):
    """Latest owned-device boundary only; diagnostics never control rendering."""
    try:
        lease = getattr(renderer, 'lease', None)
        if lease is None:
            return
        from .runtime import runtime_telemetry
        operation, scope = renderer._phase_scope
        runtime_telemetry().set_scheduler_gauge(
            f'sam.gpu_images.phase.cuda:{int(lease.device_index)}',
            dict(phase=str(phase), since_monotonic_ns=time.monotonic_ns(), owner_token=id(renderer),
                scope_id=scope, sam_operation=operation, frame=frame, crop_shape=crop_shape,
                thread_id=threading.get_ident(), thread=threading.current_thread().name))
    except Exception:
        pass  # Telemetry failures cannot change pixels, cleanup or handoffs.


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
        approved = {CANONICAL_CROP_RENDER_CONTRACT, SAM_GPU_IMAGE_CONTRACT,
                    'xta.sam_canonical_yolo_gray8/1'}
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
                 source_t_window=None, source_frames=(), opportunistic=False):
        self.context, self.view, self.source = context, view, source
        self.logical_t, self.lease, self.torch = int(logical_t), lease, torch
        self.scratch_bytes = int(scratch_bytes)
        self.required_gpu = int(required_gpu)
        self.opportunistic = bool(opportunistic)
        self.engine = None
        self.device_index = int(lease.device_index) if lease is not None else None
        self._image_burst_count = 1
        self._gpu_wait_deadline = None
        self._gpu_prefetch_cancel = None
        self._uploaded_source_key = None
        self._compute_service_started = None
        self._resume_gpu_after_yield = False
        self._render_failed = False
        self._phase_scope = getattr(getattr(context, '_resource_local', None), 'sam_phase_scope', ('', ''))
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
        from .sam_azimuthal_batch import IMPLEMENTATION_SHA256 as batch_sha256
        from .geometry_quality import geometry_quality_request_record
        return dict(contract=SAM_GPU_IMAGE_CONTRACT, backend='cuda',
            implementation_sha256=IMPLEMENTATION_SHA256,
            tta_renderer_sha256=hashlib.sha256(Path(cuda_backend.__file__).read_bytes()).hexdigest(),
            spherical_renderer_sha256=hashlib.sha256(Path(spherical_cuda.__file__).read_bytes()).hexdigest(),
            azimuthal_batch_sha256=batch_sha256,
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
            if self._compute_service_started is None:
                self._compute_service_started = time.monotonic()
            self._resume_gpu_after_yield = False
            return self.engine
        if self.resident_t_window != self.source_t_window:
            self.required_gpu += self.source_planned_bytes-self.source_resident_bytes
            self.resident_t_window = self.source_t_window
            self.source_resident_bytes = self.source_planned_bytes
        from .cuda_backend import _GpuWorkerRenderEngine
        from .media import wait_for_volume_ready
        from .runtime import _interpolation_array_backing_path
        wait_for_volume_ready(self.source)
        if self.lease is None and self.opportunistic and not self._resume_gpu_after_yield:
            self.lease = self.context._try_idle_gpu_image_lease(self.torch)
            if self.lease is None:
                raise SamGpuRenderingUnavailable('No already-idle GPU for prefetched SAM images')
            self.device_index = int(self.lease.device_index)
            try:
                free, _total = self.torch.cuda.mem_get_info(self.device_index)
            except RuntimeError:
                free = 0
            if int(free) < self.required_gpu:
                self.lease.release()
                self.lease = None
                raise SamGpuRenderingUnavailable('Idle GPU lacks admitted SAM source/render headroom')
            self.context._finish_gpu_image_wait(self, granted=True)
            self.context._check_image_lifetime()
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
                    if busy and remaining is not None and remaining <= 0 and self._resume_gpu_after_yield:
                        # A voluntary turn must not change the cohort's sampler.
                        # Stay cancellable in the FIFO while tracker work drains;
                        # real device/headroom refusals still take the failure path.
                        self._gpu_wait_deadline = time.monotonic()+30.
                        remaining = 30.
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
            self._compute_service_started = time.monotonic()
            self._resume_gpu_after_yield = False
            return self.engine
        started = time.monotonic()
        _record_image_phase(self, 'renderer_init')
        engine = self.engine = _GpuWorkerRenderEngine(f'cuda:{self.device_index}')
        _record_image_timings(renderer_init_host_seconds=time.monotonic()-started)
        _record_image_phase(self, 'source_upload')
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
        self._compute_service_started = time.monotonic()
        self._resume_gpu_after_yield = False
        return engine

    def _azimuthal_native_u8_crop(self, engine, view, frame, matrix, height, width):
        """Restrict projection work while keeping the full-plane warp coordinates.

        A small zero canvas preserves the original grid_sample normalization and
        border handling. Cropping that canvas and translating its affine would
        change float32 rounding. The existing two-pixel guard is used only in a
        bounded coordinate range; unusual geometries keep the full projection.
        """
        if str(view.family) != 'azimuthal':
            return None
        project = getattr(engine, '_render_azimuthal_native_crop_resident_torch', None)
        if not callable(project):
            return None
        native_h, native_w = int(view.src_h), int(view.src_w)
        transform = np.asarray(matrix, dtype=np.float32)
        if (min(native_h, native_w, height, width) < 1
                or max(native_h, native_w, height, width) > 8192
                or not np.isfinite(transform).all()
                or np.max(np.abs(transform[:, :2])) > 4.
                or np.max(np.abs(transform[:, 2])) > 32768.):
            return None
        from .sam_transverse_cache_rendering import native_crop_bbox
        bbox = native_crop_bbox((0, 0, height, width), transform, (native_h, native_w))
        # Keep projection+row-fold scratch below the old full-plane allowance.
        if bbox is not None and 2*(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) >= native_h*native_w:
            return None
        native = self.torch.zeros((native_h, native_w), dtype=self.torch.uint8, device=engine.device)
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            patch = project(view, int(frame), bbox)
            native[y0:y1, x0:x1] = patch.round().clamp_(0, 255).to(self.torch.uint8)
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        telemetry.add('sam.gpu_images.azimuthal_native_roi_frames', 1)
        telemetry.add('sam.gpu_images.azimuthal_native_roi_pixels',
                      0 if bbox is None else (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        telemetry.add('sam.gpu_images.azimuthal_native_full_pixels_avoided',
                      native_h*native_w-(0 if bbox is None else (bbox[2]-bbox[0])*(bbox[3]-bbox[1])))
        return native

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
                _record_image_phase(self, 'native_render', frame=int(frame), crop_shape=(height, width))
                if is_tilted_view(view):
                    pixels = engine.render_tilted_grid_resident(view, matrix, int(frame), height, width)
                    _record_image_phase(self, 'quantize', frame=int(frame), crop_shape=(height, width))
                    pixels = pixels.round().clamp_(0, 255).to(self.torch.uint8)
                else:
                    native = self._azimuthal_native_u8_crop(engine, view, frame, matrix, height, width)
                    if native is None:
                        native = engine._render_native_plane(view, int(frame))
                        _record_image_phase(self, 'quantize', frame=int(frame), crop_shape=(height, width))
                        native = native.round().clamp_(0, 255).to(self.torch.uint8)
                    _record_image_phase(self, 'warp', frame=int(frame), crop_shape=(height, width))
                    pixels = engine.warp_native_uint8_frame(native, matrix, height, width)
                # This blocking copy completes the render before immutable
                # publication or the next crop may reuse projection storage.
                enqueued = time.monotonic()
                _record_image_phase(self, 'd2h', frame=int(frame), crop_shape=(height, width))
                image = pixels.cpu().numpy()
                _record_image_timings(render_enqueue_host_seconds=enqueued-started,
                    crop_copy_wait_host_seconds=time.monotonic()-enqueued)
            _record_image_phase(self, 'native_cache_clear', frame=int(frame), crop_shape=(height, width))
            engine.clear_native_plane_cache()
            self.context._check_image_lifetime()
            _record_image_phase(self, 'host_cache_copy', frame=int(frame), crop_shape=(height, width))
            return image
        except (RuntimeError, MemoryError) as error:
            self._render_failed = True
            self.context._check_image_lifetime()
            if isinstance(error, SamGpuRenderingUnavailable):
                raise
            raise SamGpuRenderingUnavailable(str(error)) from error
        finally:
            native = pixels = None

    def render_compatible_prefix(self, view, frames, inverse, *, output_origin_yx,
                                 output_height, output_width, consume):
        """Render an ordered, funded prefix into the caller's private cache sink.

        A BatchDeclined is actionable only before the caller's sink is entered.
        The existing owner retains source/compute through success or retirement.
        """
        from itertools import islice
        import operator
        from .workspace import _env_flag
        from .sam_azimuthal_batch import (BatchDeclined, choose_batch_prefix,
            render_compatible_batch, retained_cache_snapshot)
        from .runtime import runtime_telemetry
        self.context._check_image_lifetime()
        if not _env_flag('YOLO_TTA_SAM_AZIMUTHAL_BATCH', True):
            raise BatchDeclined('disabled')
        if not callable(consume):
            raise TypeError('SAM image batch requires a private-cache sink')
        frames = tuple(islice(frames, 4))
        inverse = np.asarray(inverse, dtype=np.float32).reshape(2, 3).copy()
        crop = dict(output_origin_yx=tuple(output_origin_yx),
                    output_height=output_height, output_width=output_width)
        # An empty-cache lower bound can reject impossible work, but cannot
        # authorize allocation. Select again from the actual owner below.
        cold = choose_batch_prefix(self, view, frames, inverse, retained_cache_bytes=0, **crop)
        if not cold.eligible:
            raise BatchDeclined('no_eligible_prefix')
        try:
            engine = self._start()
        except (RuntimeError, MemoryError) as error:
            self._render_failed = True
            self.context._check_image_lifetime()
            if isinstance(error, SamGpuRenderingUnavailable):
                raise
            raise SamGpuRenderingUnavailable(str(error)) from error
        cache = retained_cache_snapshot(engine)
        plan = choose_batch_prefix(self, view, frames, inverse,
            retained_cache_bytes=cache['retained_cache_bytes'], **crop)
        if not plan.eligible:
            raise BatchDeclined('no_funded_prefix', cache)
        selected = frames[:plan.batch_size]
        delivered = 0
        telemetry = runtime_telemetry()
        telemetry.add('sam.gpu_images.batch_attempts', 1)
        _record_image_phase(self, 'batch_render', frame=int(selected[0]),
                            crop_shape=(int(output_height), int(output_width)))

        def ordered_sink(position, image):
            nonlocal delivered
            if (isinstance(position, (bool, np.bool_))
                    or operator.index(position) != delivered or delivered >= len(selected)):
                raise RuntimeError('SAM GPU batch returned an unordered or duplicate image')
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.shape != (int(output_height), int(output_width))):
                raise ValueError('SAM GPU batch returned a mismatched detector canvas')
            self.context._check_image_lifetime()
            # Count completed GPU outputs, just as the scalar demand route does.
            telemetry.add('sam.gpu_images.frames', 1)
            telemetry.add('sam.gpu_images.pixels', int(image.size))
            telemetry.add('sam.gpu_images.batch_frames', 1)
            _record_image_phase(self, 'batch_host_cache_copy', frame=int(selected[position]),
                                crop_shape=image.shape)
            consume(position, image)
            delivered += 1

        result = render_compatible_batch(self, view, selected, inverse, consume=ordered_sink, **crop)
        receipt = dict(result.receipt)
        try:
            count = operator.index(receipt.get('completed_images'))
            effective = operator.index(receipt.get('effective_batch_size'))
        except TypeError as error:
            raise RuntimeError('SAM GPU batch has an invalid completion count') from error
        if (isinstance(receipt.get('completed_images'), (bool, np.bool_))
                or isinstance(receipt.get('effective_batch_size'), (bool, np.bool_))
                or count != len(selected) or delivered != len(selected) or effective != len(selected)
                or receipt.get('batch_path_used') is not True or receipt.get('host_result_mode') != 'sink'
                or result.images):
            raise RuntimeError('SAM GPU batch did not complete its exact selected prefix')
        telemetry.add('sam.gpu_images.batches', 1)
        telemetry.add(f'sam.gpu_images.batches_size_{len(selected)}', 1)
        # Enqueue time remains comparable across scalar and batched outputs.
        # Sink time includes D2H and private mmap copying, so name it separately.
        _record_image_timings(render_enqueue_host_seconds=receipt['enqueue_host_seconds'],
            batch_render_enqueue_host_seconds=receipt['enqueue_host_seconds'],
            batch_copy_and_sink_host_seconds=receipt['copy_and_sink_host_seconds'])
        _record_image_phase(self, 'batch_complete', frame=int(selected[-1]),
                            crop_shape=(int(output_height), int(output_width)))
        try:
            telemetry.set_scheduler_gauge(f'sam.gpu_images.batch.cuda:{self.device_index}', dict(
                state='completed', batch_size=len(selected), first_frame=int(selected[0]),
                last_frame=int(selected[-1]), row_block=receipt['row_block'],
                batch_gather_iterations=receipt['batch_gather_iterations'],
                scalar_gather_iterations=receipt['scalar_gather_iterations'],
                retained_cache_bytes=receipt['retained_cache_bytes'],
                planned_device_scratch_peak_bytes=receipt['planned_device_scratch_peak_bytes'],
                scope_id=self._phase_scope[1], completed_monotonic_ns=time.monotonic_ns()))
        except Exception:
            pass
        return receipt

    def yield_compute_if_due(self):
        """Lend compute only after the caller has copied a completed crop/batch.

        Private host-cache and image-credit ownership stay with the caller.
        Reacquisition/upload occurs on its next render through the usual FIFO;
        this method never waits while holding a scheduler or image-cache lock.
        """
        if (self.lease is None or self.engine is None or self._compute_service_started is None
                or time.monotonic()-self._compute_service_started < _IMAGE_COMPUTE_SERVICE_SECONDS
                or not self.context._sam_sdk_ready()):
            return False
        self.context._check_image_lifetime()
        self.close(allow_handoff=False)
        self._image_burst_count = 1
        self._resume_gpu_after_yield = True
        try:
            from .runtime import runtime_telemetry
            runtime_telemetry().add('sam.gpu_images.compute_yields', 1)
        except Exception:
            pass  # Diagnostics cannot undo completed CUDA retirement.
        return True

    def close(self, *, allow_handoff=True):
        if self.lease is None:
            return
        from .backprojection import _trim_main_process_cuda_device
        try:
            if (allow_handoff and self.engine is not None and not self._render_failed
                    and sys.exc_info()[1] is None and self._try_handoff()):
                return
            _record_image_phase(self, 'retirement')
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
            # Publish before releasing: a subsequent owner may immediately overwrite this key.
            _record_image_phase(self, 'release_compute')
            self.context._release_sam_compute_lease(self.lease)
            # A pre-start global stage is not registered as predictor compute.
            self.lease.release()
            self.lease = None
            self._compute_service_started = None
            self._resume_gpu_after_yield = False
            self._uploaded_source_key = None
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
        _record_image_phase(self, 'handoff_cleanup')
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
            _record_image_phase(next_renderer, 'handoff_awaiting_render')
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


def try_gpu_crop_renderer(context, view, shape, prepared_plan, *, opportunistic=False):
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
    if payload+scratch > int(assigned.get('assigned_image_bytes',
            assigned['base_other_non_cpu_allowance_bytes'])):
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
        required_gpu=required_gpu, source_t_window=source_window, source_frames=required,
        opportunistic=opportunistic)
