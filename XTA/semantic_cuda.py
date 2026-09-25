"""CUDA semantic-logit decoding and exact native component confidence filtering.

The CUDA entry points enqueue work on the caller's stream.  They never read a
per-frame scalar on the host; callers must keep borrowed TensorRT output storage
alive until that stream has completed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
import threading

import numpy as np


_KERNEL_SOURCE = r'''
extern "C" __global__ void semantic_decode_native(
    const float* logits, unsigned char* mask, unsigned char* confidence,
    int* foreground_count, int* weak_foreground,
    int channels, int input_h, int input_w, int output_size,
    int native_h, int native_w, float conf_threshold, int min_conf_u8,
    float t00, float t01, float t02, float t10, float t11, float t12
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = native_h * native_w;
    if (i >= n) return;
    int y = i / native_w;
    int x = i - y * native_w;
    // These are affine_grid/grid_sample's align_corners=False pixel centres.
    float nx = (2.0f * x + 1.0f) / native_w - 1.0f;
    float ny = (2.0f * y + 1.0f) / native_h - 1.0f;
    float gx = t00 * nx + t01 * ny + t02;
    float gy = t10 * nx + t11 * ny + t12;
    int ox = (int)nearbyintf(((gx + 1.0f) * output_size - 1.0f) * 0.5f);
    int oy = (int)nearbyintf(((gy + 1.0f) * output_size - 1.0f) * 0.5f);
    if (ox < 0 || oy < 0 || ox >= output_size || oy >= output_size) {
        mask[i] = 0;
        confidence[i] = 0;
        return;
    }

    // F.interpolate(..., align_corners=False) samples input pixel centres and
    // replicates edge values.  Identity size avoids any interpolation.
    int plane = input_h * input_w;
    float z0, z1 = 0.0f;
    if (input_h == output_size && input_w == output_size) {
        // F.interpolate is an identity at equal raster sizes.  A weighted sum
        // would incorrectly turn an infinite logit into 0*Inf = NaN.
        int p = oy * input_w + ox;
        z0 = logits[p];
        if (channels == 2) z1 = logits[plane + p];
    } else {
        float sx = ((ox + 0.5f) * input_w / output_size) - 0.5f;
        float sy = ((oy + 0.5f) * input_h / output_size) - 0.5f;
        sx = fminf(fmaxf(sx, 0.0f), input_w - 1.0f);
        sy = fminf(fmaxf(sy, 0.0f), input_h - 1.0f);
        int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
        int x1 = x0 + 1 < input_w ? x0 + 1 : input_w - 1;
        int y1 = y0 + 1 < input_h ? y0 + 1 : input_h - 1;
        float wx = sx - x0, wy = sy - y0;
        int p00 = y0 * input_w + x0;
        int p01 = y0 * input_w + x1;
        int p10 = y1 * input_w + x0;
        int p11 = y1 * input_w + x1;
        z0 = (1.0f - wy) * ((1.0f - wx) * logits[p00] + wx * logits[p01])
           + wy * ((1.0f - wx) * logits[p10] + wx * logits[p11]);
        if (channels == 2) {
            z1 = (1.0f - wy) * ((1.0f - wx) * logits[plane + p00] + wx * logits[plane + p01])
               + wy * ((1.0f - wx) * logits[plane + p10] + wx * logits[plane + p11]);
        }
    }
    float difference;
    if (channels == 1) {
        difference = -z0;
    } else {
        // PyTorch softmax yields NaN if either class is +Inf or both are -Inf.
        // A NaN probability never satisfies the foreground threshold.
        const float finite_max = 3.402823466e38f;
        if (z0 != z0 || z1 != z1 || z0 > finite_max || z1 > finite_max ||
            (z0 < -finite_max && z1 < -finite_max)) {
            mask[i] = 0;
            confidence[i] = 0;
            return;
        }
        difference = z0 - z1;
    }
    if (difference != difference) {
        mask[i] = 0;
        confidence[i] = 0;
        return;
    }
    difference = fminf(fmaxf(difference, -80.0f), 80.0f);
    float probability = 1.0f / (1.0f + expf(difference));
    bool foreground = probability >= conf_threshold;
    unsigned char score = foreground ? (unsigned char)nearbyintf(probability * 255.0f) : 0;
    mask[i] = foreground ? 1 : 0;
    confidence[i] = score;
    // One atomic per active warp rather than one per foreground pixel.  Both
    // flags are reset on this stream before decode, including direct output counts.
    unsigned live = __activemask();
    unsigned foreground_votes = __ballot_sync(live, foreground);
    unsigned weak_votes = __ballot_sync(live, foreground && score < min_conf_u8);
    if ((threadIdx.x & 31) == (__ffs(live) - 1)) {
        if (foreground_votes) atomicOr(foreground_count, 1);
        if (weak_votes) atomicOr(weak_foreground, 1);
    }
}

extern "C" __global__ void semantic_reset_flags(int* foreground_count, int* weak_foreground) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        foreground_count[0] = 0;
        weak_foreground[0] = 0;
    }
}

extern "C" __global__ void semantic_scan_weak(
    const unsigned char* mask, const unsigned char* confidence,
    int* foreground_count, int* weak_foreground, int threshold, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool foreground = mask[i] != 0;
    unsigned live = __activemask();
    unsigned foreground_votes = __ballot_sync(live, foreground);
    unsigned weak_votes = __ballot_sync(live, foreground && confidence[i] < threshold);
    if ((threadIdx.x & 31) == (__ffs(live) - 1)) {
        if (foreground_votes) atomicOr(foreground_count, 1);
        if (weak_votes) atomicOr(weak_foreground, 1);
    }
}

extern "C" __global__ void semantic_parent_init(
    const unsigned char* mask, int* parent, int* component_max,
    int* foreground_count, const int* weak_foreground, int n
) {
    if (!weak_foreground[0]) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) foreground_count[0] = 0;
    parent[i] = mask[i] ? i : -1;
    component_max[i] = 0;
}

__device__ __forceinline__ int semantic_root(int* parent, int i) {
    int p = parent[i];
    while (p != i) {
        int grandparent = parent[p];
        if (grandparent != p) atomicMin(parent + i, grandparent);
        i = p;
        p = grandparent;
    }
    return i;
}

__device__ __forceinline__ void semantic_join(int* parent, int a, int b) {
    while (true) {
        int ra = semantic_root(parent, a);
        int rb = semantic_root(parent, b);
        if (ra == rb) return;
        int high = ra > rb ? ra : rb;
        int low = ra < rb ? ra : rb;
        // Roots only move to lower indices, so this forest cannot cycle.  Retry
        // if a concurrent edge already changed the candidate root.
        if (atomicMin(parent + high, low) == high) return;
    }
}

extern "C" __global__ void semantic_union8(
    const unsigned char* mask, int* parent, const int* weak_foreground, int h, int w
) {
    if (!weak_foreground[0]) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h * w || !mask[i]) return;
    int y = i / w, x = i - y * w;
    if (x > 0 && mask[i - 1]) semantic_join(parent, i, i - 1);
    if (y > 0) {
        if (x > 0 && mask[i - w - 1]) semantic_join(parent, i, i - w - 1);
        if (mask[i - w]) semantic_join(parent, i, i - w);
        if (x + 1 < w && mask[i - w + 1]) semantic_join(parent, i, i - w + 1);
    }
}

extern "C" __global__ void semantic_component_max(
    const unsigned char* mask, const unsigned char* confidence,
    int* parent, int* component_max, const int* weak_foreground, int n
) {
    if (!weak_foreground[0]) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    int root = semantic_root(parent, i);
    parent[i] = root;
    int score = (int)confidence[i];
    // Scores only increase.  A stale low read causes an extra atomic, while a
    // current/higher value lets dense components avoid a hot atomic per pixel.
    if (score > ((volatile int*)component_max)[root]) {
        atomicMax(component_max + root, score);
    }
}

extern "C" __global__ void semantic_component_filter(
    unsigned char* mask, unsigned char* confidence,
    const int* parent, const int* component_max,
    int* foreground_count, const int* weak_foreground, int threshold, int n
) {
    if (!weak_foreground[0]) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !mask[i]) return;
    if (component_max[parent[i]] < threshold) {
        mask[i] = 0;
        confidence[i] = 0;
    } else {
        unsigned live = __activemask();
        unsigned kept = __ballot_sync(live, true);
        if ((threadIdx.x & 31) == (__ffs(live) - 1) && kept) {
            atomicOr(foreground_count, 1);
        }
    }
}
'''


@lru_cache(maxsize=16)
def _kernels_for_device(device_index: int):
    import cupy as cp  # type: ignore

    if int(cp.cuda.runtime.getDevice()) != int(device_index):
        raise RuntimeError('semantic CUDA kernel cache requested for the wrong device')
    module = cp.RawModule(
        code=_KERNEL_SOURCE, options=('-std=c++11',),
        name_expressions=(
            'semantic_decode_native', 'semantic_reset_flags', 'semantic_scan_weak',
            'semantic_parent_init', 'semantic_union8',
            'semantic_component_max', 'semantic_component_filter',
        ),
    )
    return cp, {name: module.get_function(name) for name in (
        'semantic_decode_native', 'semantic_reset_flags', 'semantic_scan_weak',
        'semantic_parent_init', 'semantic_union8',
        'semantic_component_max', 'semantic_component_filter',
    )}


def _kernels():
    import cupy as cp  # type: ignore

    return _kernels_for_device(int(cp.cuda.runtime.getDevice()))


def semantic_cuda_available(device: object = None) -> bool:
    """Precompile all kernels before selecting a device-only accumulation path."""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False
        selected = torch.device(device) if device is not None else torch.cuda.current_device()
        with torch.cuda.device(selected):
            _kernels()
        return True
    except Exception:
        return False


def _on_stream(stream: Optional[object], device: object):
    import torch  # type: ignore

    with torch.cuda.device(device):
        cp, _ = _kernels()
        torch_stream = stream if stream is not None else torch.cuda.current_stream(device)
        if torch.device(torch_stream.device) != torch.device(device):
            raise ValueError('semantic CUDA stream and logits must be on the same device')
        return cp, torch_stream, cp.cuda.ExternalStream(int(torch_stream.cuda_stream))


_COMPONENT_WORKSPACES: dict[tuple[int, int, int], tuple[Optional[object], Optional[object], object]] = {}
_COMPONENT_WORKSPACES_LOCK = threading.Lock()


def _component_workspace(cp, n: int, stream, *, need_components: bool = True):
    """Reuse scratch on one thread/stream; CUDA ordering protects reuse."""
    key = (threading.get_ident(), int(cp.cuda.runtime.getDevice()), int(stream.ptr))
    workspace = _COMPONENT_WORKSPACES.get(key)
    if workspace is None or (need_components and (
        workspace[0] is None or int(workspace[0].size) < int(n)
    )):
        # The fast hit is GIL-safe.  Allocation and retirement serialize here;
        # retirement is permitted only after the worker's CUDA completion fence.
        with _COMPONENT_WORKSPACES_LOCK:
            workspace = _COMPONENT_WORKSPACES.get(key)
            if workspace is None:
                flags = cp.empty((2,), dtype=cp.int32)
                parent = maxima = None
                workspace = parent, maxima, flags
                _COMPONENT_WORKSPACES[key] = workspace
            if need_components and (workspace[0] is None or int(workspace[0].size) < int(n)):
                parent = cp.empty((int(n),), dtype=cp.int32)
                maxima = cp.empty((int(n),), dtype=cp.int32)
                workspace = parent, maxima, workspace[2]
                _COMPONENT_WORKSPACES[key] = workspace
    return (
        None if workspace[0] is None else workspace[0][:n],
        None if workspace[1] is None else workspace[1][:n],
        workspace[2],
    )


def clear_semantic_cuda_cache() -> int:
    """Release resident CCL scratch after the caller's global CUDA retirement fence.

    Returns the number of live scratch bytes released.  This must not run while
    any semantic decode or component filter is in flight.
    """
    with _COMPONENT_WORKSPACES_LOCK:
        workspaces = list(_COMPONENT_WORKSPACES.items())
        _COMPONENT_WORKSPACES.clear()
    released = sum((0 if parent is None else int(parent.nbytes))
                   + (0 if maxima is None else int(maxima.nbytes)) + int(flags.nbytes)
                   for _key, (parent, maxima, flags) in workspaces)
    devices = {int(key[1]) for key, _workspace in workspaces}
    workspaces.clear()
    if devices:
        import cupy as cp  # type: ignore

        for device_index in devices:
            with cp.cuda.Device(device_index):
                cp.get_default_memory_pool().free_all_blocks()
    return released


def _filter_cupy(
    mask, confidence, threshold: int, stream, *,
    foreground_count=None, weak_foreground=None, preclassified: bool = False,
) -> None:
    cp, kernels = _kernels()
    if not 1 <= int(threshold) <= 255:
        raise ValueError('min_conf_u8 must be in 1..255 for component filtering')
    h, w = (int(mask.shape[0]), int(mask.shape[1]))
    n = h * w
    parent, maxima, flags = _component_workspace(cp, n, stream)
    count = foreground_count if foreground_count is not None else flags[:1]
    weak = weak_foreground if weak_foreground is not None else flags[1:2]
    grid = ((n + 255) // 256,)
    block = (256,)
    if not preclassified:
        kernels['semantic_reset_flags']((1,), (1,), (count, weak), stream=stream)
        kernels['semantic_scan_weak'](
            grid, block, (mask, confidence, count, weak, np.int32(threshold), np.int32(n)),
            stream=stream,
        )
    kernels['semantic_parent_init'](
        grid, block, (mask, parent, maxima, count, weak, np.int32(n)), stream=stream,
    )
    kernels['semantic_union8'](grid, block, (mask, parent, weak, np.int32(h), np.int32(w)), stream=stream)
    kernels['semantic_component_max'](
        grid, block, (mask, confidence, parent, maxima, weak, np.int32(n)), stream=stream,
    )
    kernels['semantic_component_filter'](
        grid, block, (mask, confidence, parent, maxima, count, weak,
                      np.int32(threshold), np.int32(n)),
        stream=stream,
    )
    # Scratch remains owned by this thread and stream.  A following frame on the
    # same stream cannot overwrite it before this filter completes.


def filter_semantic_components_cuda(
    mask_t: object, conf_u8_t: object, min_conf_u8: int, stream: Optional[object] = None,
) -> tuple[object, object]:
    """Apply the native 8-connected component confidence rule entirely on CUDA.

    The arrays are filtered in place.  A component is retained if its highest
    uint8 foreground confidence is at least ``min_conf_u8``.
    """
    import torch  # type: ignore

    if not isinstance(mask_t, torch.Tensor) or not isinstance(conf_u8_t, torch.Tensor):
        raise TypeError('semantic component filtering requires torch tensors')
    if not mask_t.is_cuda or not conf_u8_t.is_cuda or mask_t.device != conf_u8_t.device:
        raise ValueError('semantic component filtering requires colocated CUDA tensors')
    if mask_t.dtype != torch.uint8 or conf_u8_t.dtype != torch.uint8:
        raise ValueError('semantic component filtering requires uint8 mask and confidence')
    if mask_t.ndim != 2 or mask_t.shape != conf_u8_t.shape:
        raise ValueError('semantic mask and confidence must have the same 2-D shape')
    if int(min_conf_u8) <= 0:
        return mask_t, conf_u8_t
    cp, torch_stream, cp_stream = _on_stream(stream, mask_t.device)
    with torch.cuda.device(mask_t.device), torch.cuda.stream(torch_stream), cp_stream:
        mask_t.record_stream(torch_stream)
        conf_u8_t.record_stream(torch_stream)
        mask = cp.from_dlpack(mask_t)
        conf = cp.from_dlpack(conf_u8_t)
        _filter_cupy(mask, conf, int(min_conf_u8), cp_stream)
    return mask_t, conf_u8_t


@lru_cache(maxsize=128)
def _cached_affine_theta(
    matrix_values: tuple[float, ...], output_size: int, native_h: int, native_w: int,
) -> np.ndarray:
    m = np.asarray(matrix_values, dtype=np.float64).reshape(2, 3)
    inverse = np.linalg.inv(np.vstack((m, [0.0, 0.0, 1.0])))
    out_pixels = np.array([
        [native_w / 2.0, 0.0, native_w / 2.0 - 0.5],
        [0.0, native_h / 2.0, native_h / 2.0 - 0.5],
        [0.0, 0.0, 1.0],
    ])
    normalize = np.array([
        [2.0 / output_size, 0.0, 1.0 / output_size - 1.0],
        [0.0, 2.0 / output_size, 1.0 / output_size - 1.0],
        [0.0, 0.0, 1.0],
    ])
    return (normalize @ inverse @ out_pixels)[:2].astype(np.float32)


def _affine_theta(M_out_to_native: np.ndarray, output_size: int, native_h: int, native_w: int) -> np.ndarray:
    values = tuple(float(value) for value in np.asarray(M_out_to_native, dtype=np.float64).reshape(-1))
    if len(values) != 6:
        raise ValueError('semantic native affine must be a 2x3 matrix')
    return _cached_affine_theta(values, int(output_size), int(native_h), int(native_w))


def semantic_native_cuda(
    logits: object,
    *,
    output_size: int,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    conf_threshold: float,
    min_conf_u8: int,
    stream: Optional[object] = None,
    out_mask: Optional[object] = None,
    out_conf: Optional[object] = None,
    foreground_count: Optional[object] = None,
) -> tuple[object, object]:
    """Decode one raw semantic frame and filter its native mask on CUDA.

    Returns uint8 torch tensors ``(mask, confidence)`` of native shape.  Optional
    outputs are written in place; each must be exclusive to this frame and
    contiguous. ``foreground_count`` is one CUDA int32 element, overwritten with
    0 or 1 after component filtering. The caller owns stream ordering and
    borrowed-logit lifetime until completion.
    """
    import torch  # type: ignore

    if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.ndim != 3:
        raise ValueError('semantic CUDA decoder requires [C,H,W] CUDA torch logits')
    if int(logits.shape[0]) not in (1, 2):
        raise ValueError('binary semantic CUDA logits require one or two channels')
    if not logits.dtype.is_floating_point:
        raise ValueError('semantic CUDA logits must be floating point')
    output_size = int(output_size)
    native_h, native_w = int(native_h), int(native_w)
    if min(output_size, native_h, native_w, int(logits.shape[1]), int(logits.shape[2])) <= 0:
        raise ValueError('semantic CUDA raster dimensions must be positive')
    if not 0.0 <= float(conf_threshold) <= 1.0:
        raise ValueError('semantic CUDA confidence threshold must be in [0,1]')
    if not 0 <= int(min_conf_u8) <= 255:
        raise ValueError('semantic CUDA min_conf_u8 must be in [0,255]')
    for label, value, shape, dtype in (
        ('out_mask', out_mask, (native_h, native_w), torch.uint8),
        ('out_conf', out_conf, (native_h, native_w), torch.uint8),
        ('foreground_count', foreground_count, None, torch.int32),
    ):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or not value.is_cuda or value.device != logits.device:
            raise ValueError(f'{label} must be a CUDA tensor on the logits device')
        if value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f'{label} must be contiguous {dtype}')
        if (shape is not None and tuple(int(x) for x in value.shape) != shape) or (
            shape is None and int(value.numel()) != 1
        ):
            raise ValueError(f'{label} has the wrong shape')
    output_ptrs = [int(value.data_ptr()) for value in (out_mask, out_conf, foreground_count) if value is not None]
    if len(output_ptrs) != len(set(output_ptrs)) or any(int(logits.data_ptr()) == ptr for ptr in output_ptrs):
        raise ValueError('semantic CUDA outputs must not alias each other or logits')
    theta = _affine_theta(M_out_to_native, output_size, native_h, native_w).reshape(-1)
    cp, torch_stream, cp_stream = _on_stream(stream, logits.device)
    with torch.cuda.device(logits.device), torch.cuda.stream(torch_stream), cp_stream:
        # The established path converts logits to FP32 before interpolation.
        source = logits.contiguous().float()
        source.record_stream(torch_stream)
        source_cp = cp.from_dlpack(source)
        mask = cp.from_dlpack(out_mask) if out_mask is not None else cp.empty((native_h, native_w), dtype=cp.uint8)
        confidence = cp.from_dlpack(out_conf) if out_conf is not None else cp.empty_like(mask)
        _, kernels = _kernels()
        n = native_h * native_w
        _parent, _maxima, flags = _component_workspace(
            cp, n, cp_stream, need_components=int(min_conf_u8) > 0,
        )
        count = cp.from_dlpack(foreground_count) if foreground_count is not None else flags[:1]
        weak = flags[1:2]
        kernels['semantic_reset_flags']((1,), (1,), (count, weak), stream=cp_stream)
        kernels['semantic_decode_native'](
            ((n + 255) // 256,), (256,),
            (source_cp, mask, confidence, count, weak, np.int32(logits.shape[0]),
             np.int32(logits.shape[1]), np.int32(logits.shape[2]), np.int32(output_size),
             np.int32(native_h), np.int32(native_w), np.float32(conf_threshold),
             np.int32(min_conf_u8),
             *(np.float32(value) for value in theta)),
            stream=cp_stream,
        )
        if int(min_conf_u8) > 0:
            _filter_cupy(
                mask, confidence, int(min_conf_u8), cp_stream,
                foreground_count=count, weak_foreground=weak, preclassified=True,
            )
        mask_t = out_mask if out_mask is not None else torch.from_dlpack(mask)
        conf_t = out_conf if out_conf is not None else torch.from_dlpack(confidence)
        mask_t.record_stream(torch_stream)
        conf_t.record_stream(torch_stream)
        if foreground_count is not None:
            foreground_count.record_stream(torch_stream)
    return mask_t, conf_t


def semantic_native_reference(
    logits: object,
    *,
    output_size: int,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    conf_threshold: float,
    min_conf_u8: int,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference using the existing PyTorch/grid_sample and component rule."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from scipy import ndimage as ndi  # type: ignore

    from .semantic_inference import semantic_foreground_probability

    frame = torch.as_tensor(logits).float()
    if frame.ndim != 3:
        raise ValueError('semantic reference requires [C,H,W] logits')
    probability = semantic_foreground_probability(frame.unsqueeze(0), output_size=int(output_size))[0]
    union = (probability >= float(conf_threshold)).float()
    theta = torch.from_numpy(_affine_theta(M_out_to_native, int(output_size), int(native_h), int(native_w)))
    grid = F.affine_grid(theta.unsqueeze(0), [1, 1, int(native_h), int(native_w)], align_corners=False)
    warped = F.grid_sample(torch.stack((union, probability))[None], grid, mode='nearest',
                           padding_mode='zeros', align_corners=False)[0]
    mask = (warped[0] > 0.5).numpy().astype(np.uint8)
    confidence = torch.where(warped[0] > 0.5, (warped[1].clamp(0, 1) * 255).round(), 0)
    conf = confidence.numpy().astype(np.uint8)
    if int(min_conf_u8) > 0:
        labels, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
        maxima = np.asarray(ndi.maximum(conf, labels=labels, index=np.arange(1, n + 1)), dtype=np.uint8)
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = maxima >= int(min_conf_u8)
        mask = keep[labels].astype(np.uint8)
        conf = np.where(mask, conf, 0).astype(np.uint8)
    return mask, conf
