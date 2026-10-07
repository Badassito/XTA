"""Idle-GPU SAM image preparation through the existing TTA intensity renderer.

Only demanded grayscale crops leave the device. One existing compute lease
fences a transaction; source, projection and grid allocations retire before
that lease is returned. CPU preparation remains the admission fallback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time
import traceback
import weakref

import numpy as np


SAM_GPU_IMAGE_CONTRACT = 'xta.sam_tta_cuda_gray8_crop/1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LIVE_IMAGES = {}
_LIVE_IMAGES_LOCK = threading.RLock()


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


class SamGpuCropRenderer:
    def __init__(self, context, view, source, logical_t, lease, torch, scratch_bytes, *, required_gpu=0):
        self.context, self.view, self.source = context, view, source
        self.logical_t, self.lease, self.torch = int(logical_t), lease, torch
        self.scratch_bytes = int(scratch_bytes)
        self.required_gpu = int(required_gpu)
        self.engine = None
        self.device_index = int(lease.device_index) if lease is not None else None

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
            temporal_sampling=('existing_processing_t' if int(self.source.shape[0])==self.logical_t
                else 'tta_center_aligned_virtual_processing_t_gray8'))

    def _start(self):
        if self.engine is not None:
            return self.engine
        from .cuda_backend import _GpuWorkerRenderEngine
        from .media import wait_for_volume_ready
        from .runtime import _interpolation_array_backing_path
        wait_for_volume_ready(self.source)
        if self.lease is None:
            from .backprojection import _try_acquire_specific_main_process_gpu_stage
            from .runtime import runtime_telemetry
            # ponytail: coordinator claims are not fair; cap waits at 30s.
            # Add fair image admission only if busy timeouts remain significant.
            deadline = time.monotonic() + 30.
            while self.lease is None:
                self.context._check_image_lifetime()
                busy = False
                for token in self.context.device_ids:
                    device = int(token.split(':')[-1])
                    with self.context._gpu_lease_lock:
                        resident = self.context._resident_leases.get(device)
                    # Predictor ACKs cannot settle this parent's render stream.
                    # Waiting owns neither a compute lease nor a render engine.
                    runtime_telemetry().add('sam.gpu_images.admission_attempts', 1)
                    lease = (resident.try_acquire_compute(self.torch, 'SAM image preparation') if resident is not None else
                        _try_acquire_specific_main_process_gpu_stage(self.torch, device, 'SAM image preparation'))
                    if lease is None:
                        busy = True
                        continue
                    self.lease, self.device_index = lease, device
                    admitted = False
                    try:
                        free, _total = self.torch.cuda.mem_get_info(device)
                        if int(free) >= self.required_gpu:
                            admitted = True
                            break
                    except RuntimeError:
                        pass
                    finally:
                        if not admitted:
                            self.context._release_sam_compute_lease(lease)
                            lease.release()
                            self.lease = None
                remaining = deadline - time.monotonic()
                if self.lease is not None or not busy or remaining <= 0:
                    break
                started = time.monotonic()
                self.context._cancel.wait(min(.05, remaining))
                runtime_telemetry().add('sam.gpu_images.admission_wait_seconds', time.monotonic()-started)
            if self.lease is None:
                if busy and remaining <= 0:
                    runtime_telemetry().add('sam.gpu_images.admission_wait_timeouts', 1)
                raise SamGpuRenderingUnavailable('No idle GPU with admitted SAM source/render headroom')
        engine = self.engine = _GpuWorkerRenderEngine(f'cuda:{self.device_index}')
        backing = _interpolation_array_backing_path(self.source)
        if backing is not None:
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
        from .runtime import runtime_telemetry
        runtime_telemetry().add('sam.gpu_images.gpu_admissions', 1)
        runtime_telemetry().add('sam.gpu_images.source_uploads', 1)
        runtime_telemetry().add('sam.gpu_images.source_upload_bytes', int(self.source.nbytes))
        return engine

    def render(self, view, frame, inverse, *, output_origin_yx, output_height, output_width):
        from .geometry import is_tilted_view
        self.context._check_image_lifetime()
        height, width = int(output_height), int(output_width)
        if min(height, width) <= 0 or height*width > self.scratch_bytes:
            raise SamGpuRenderingUnavailable('SAM crop exceeds its admitted render workspace')
        if not 0 <= int(frame) < int(view.num_slices):
            raise IndexError('SAM GPU frame is outside its physical view')
        native = pixels = None
        try:
            engine = self._start()
            matrix = crop_inverse(inverse, output_origin_yx)
            with self.torch.cuda.device(engine.device), self.torch.cuda.stream(engine._stream):
                if is_tilted_view(view):
                    pixels = engine.render_tilted_grid_resident(view, matrix, int(frame), height, width)
                    pixels = pixels.round().clamp_(0, 255).to(self.torch.uint8)
                else:
                    native = engine._render_native_plane(view, int(frame))
                    native = native.round().clamp_(0, 255).to(self.torch.uint8)
                    pixels = engine.warp_native_uint8_frame(native, matrix, height, width)
                # This blocking copy completes the render before immutable
                # publication or the next crop may reuse projection storage.
                image = pixels.cpu().numpy()
            engine.clear_native_plane_cache()
            self.context._check_image_lifetime()
            return image
        except (RuntimeError, MemoryError) as error:
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
            if self.engine is not None:
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
                self.torch.device(f'cuda:{self.device_index}'), desc='SAM image preparation')
            self.context._release_sam_compute_lease(self.lease)
            # A pre-start global stage is not registered as predictor compute.
            self.lease.release()
            self.lease = None
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


def try_gpu_crop_renderer(context, view, shape, prepared_plan):
    """Defer compute admission until pixels are needed; reject unsupported sources."""
    from .workspace import _env_flag, _env_int
    from .runtime import runtime_telemetry
    def decline(reason):
        runtime_telemetry().add('sam.gpu_images.cpu_admissions', 1)
        runtime_telemetry().gauge('sam.gpu_images.last_cpu_reason', reason)
        return None
    if not _env_flag('YOLO_TTA_SAM_GPU_IMAGES', True) or not context.detector_retirement_ready:
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
    from .runtime import _interpolation_array_backing_path
    backing = _interpolation_array_backing_path(source)
    if int(source.shape[0]) != logical_t and backing is None:
        return decline('native_t_source_has_no_backing')
    from .geometry import physical_view_name
    if (str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
            and tuple(shape) == tuple(source.shape) and backing is not None):
        return decline('exact_transverse_backing_reuse')
    import torch
    if not torch.cuda.is_available():
        return decline('cuda_unavailable')
    from .cuda_backend import gpu_render_reserve_bytes
    # Conservative allowance covers native projector plans/taps, one output
    # grid and per-crop temporaries. No optional full-volume texture is kept.
    required_gpu = int(source.nbytes)+256*int(view.src_h)*int(view.src_w)+96*max(
        (int(box[2])-int(box[0]))*(int(box[3])-int(box[1])) for box in required.values())
    required_gpu += gpu_render_reserve_bytes()
    # No compute ownership while waiting for an image ticket or CPU credit.
    # The first missing crop waits for compute without holding a CUDA allocation.
    return SamGpuCropRenderer(context, view, source, logical_t, None, torch, scratch,
        required_gpu=required_gpu)
