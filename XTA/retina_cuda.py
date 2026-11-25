"""Exact ROI retina decoding with bounded per-frame CUDA storage."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


# Each prototype cell has one owner, or is marked contested. The owner map
# covers every output pixel's ceil-clipped bounding box, independently of logit
# signs and scores. Contested cells evaluate all intersecting instance ROIs.
_SOURCE = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float value(const float* p, int i) { return p[i]; }
__device__ __forceinline__ float value(const half* p, int i) { return __half2float(p[i]); }

__device__ __forceinline__ float sample_coord(int output, int proto_size, int output_size) {
    const float scale = __fdiv_rn((float)proto_size, (float)output_size);
    const float source = __fsub_rn(__fmul_rn((float)output + 0.5f, scale), 0.5f);
    return fminf((float)(proto_size - 1), fmaxf(0.0f, source));
}

__device__ __forceinline__ int first_output(int cell, int proto_size, int output_size) {
    if (cell <= 0) return 0;
    if (cell >= proto_size) return output_size;
    // Invert the output-centre mapping, then correct the rounded boundary
    // against the exact forward mapping used by the ownership pass.
    int first = (int)ceilf(((float)cell + 0.5f) * ((float)output_size / proto_size) - 0.5f);
    first = max(0, min(output_size, first));
    while (first > 0 && sample_coord(first - 1, proto_size, output_size) >= cell) --first;
    while (first < output_size && sample_coord(first, proto_size, output_size) < cell) ++first;
    return first;
}

__device__ __forceinline__ int first_output_4x(int cell, int proto_size) {
    if (cell == 0) return 0;
    if (cell == proto_size) return 4 * proto_size;
    return 4 * cell + 2;
}

__device__ __forceinline__ float interpolation_weight_4x(int output, int cell, int proto_size) {
    if (cell + 1 == proto_size) return 0.f;
    return fmaxf(0.f, (output - 4 * cell) * .25f - .375f);
}

__device__ __forceinline__ int clip_bound(float coord, int limit) {
    return (int)ceilf(fminf((float)limit, fmaxf(0.0f, coord)));
}

template <typename H>
__device__ void mark_cells(
    const H* head, int anchors, const int* indices, const int* count, int xyxy,
    int ph, int pw, int ih, int iw, int4* boxes, int* owners) {
    const int n = *count;
    __shared__ int4 cell_box;
    for (int d = blockIdx.x; d < n; d += gridDim.x) {
        if (threadIdx.x == 0) {
            const int anchor = indices[d];
            float x0 = value(head, anchor), y0 = value(head, anchors + anchor);
            float x1 = value(head, 2 * anchors + anchor), y1 = value(head, 3 * anchors + anchor);
            if (!xyxy) {
                const float half_width = 0.5f * x1, half_height = 0.5f * y1;
                x1 = x0 + half_width; x0 -= half_width;
                y1 = y0 + half_height; y0 -= half_height;
            }
            int4 roi = make_int4(0, 0, 0, 0);
            if (isfinite(x0) && isfinite(y0) && isfinite(x1) && isfinite(y1)) {
                roi = make_int4(clip_bound(x0, iw), clip_bound(y0, ih),
                                clip_bound(x1, iw), clip_bound(y1, ih));
            }
            boxes[d] = roi;
            cell_box = (roi.x < roi.z && roi.y < roi.w)
                ? make_int4((int)sample_coord(roi.x, pw, iw), (int)sample_coord(roi.y, ph, ih),
                            1 + (int)sample_coord(roi.z - 1, pw, iw), 1 + (int)sample_coord(roi.w - 1, ph, ih))
                : make_int4(0, 0, 0, 0);
        }
        __syncthreads();
        const int4 cells = cell_box;
        const int width = cells.z - cells.x;
        const int length = width * (cells.w - cells.y);
        for (int j = threadIdx.x; j < length; j += blockDim.x) {
            const int y = cells.y + j / width;
            const int x = cells.x + j % width;
            int* owner = owners + y * pw + x;
            if (atomicCAS(owner, 0, d + 1) > 0) atomicExch(owner, -1);
        }
        __syncthreads();
    }
}

template <typename H, typename P>
__device__ void decode_cells(
    const H* head, const P* proto, int anchors, int channels, int coeff_first,
    int proto_h, int proto_w, int ph, int pw, int py, int px, int ih, int iw,
    const int* indices, const int* count, const int4* boxes, const int* owners,
    float* mask, float* scores) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ph * pw) return;
    const int owner = owners[cell];
    if (owner == 0) return;
    const int y = cell / pw, x = cell % pw;
    // This is the common stride-4 prototype grid. Half-pixel sampling becomes
    // (output - 4 * cell) / 4 - 3/8, whose operations are exact in this range.
    const bool fourfold = ih <= (1 << 22) && iw <= (1 << 22) && ih == 4 * ph && iw == 4 * pw;
    const int oy0 = fourfold ? first_output_4x(y, ph) : first_output(y, ph, ih);
    const int oy1 = fourfold ? first_output_4x(y + 1, ph) : first_output(y + 1, ph, ih);
    const int ox0 = fourfold ? first_output_4x(x, pw) : first_output(x, pw, iw);
    const int ox1 = fourfold ? first_output_4x(x + 1, pw) : first_output(x + 1, pw, iw);
    if (ox0 == ox1 || oy0 == oy1) return;

    // For uncontested cells, compute these four uncropped corner logits for
    // exactly one instance and share them over all pixels of the cell. Even
    // a sub-cell ROI is clipped at the final output pixels.
    const int y1 = min(y + 1, ph - 1), x1 = min(x + 1, pw - 1);
    const int p00 = (y + py) * proto_w + x + px;
    const int p01 = (y + py) * proto_w + x1 + px;
    const int p10 = (y1 + py) * proto_w + x + px;
    const int p11 = (y1 + py) * proto_w + x1 + px;
    const int area = proto_h * proto_w;
    const bool identity = ph == ih && pw == iw;
    const int begin = owner > 0 ? owner - 1 : 0;
    const int end = owner > 0 ? owner : *count;
    for (int d = begin; d < end; ++d) {
        const int4 box = boxes[d];
        const int xmin = max(ox0, box.x), xmax = min(ox1, box.z);
        const int ymin = max(oy0, box.y), ymax = min(oy1, box.w);
        if (xmin >= xmax || ymin >= ymax) continue;
        const int anchor = indices[d];
        float a = 0, b = 0, c = 0, e = 0;
        #pragma unroll 4
        for (int channel = 0; channel < channels; ++channel) {
            const float coeff = value(head, (coeff_first + channel) * anchors + anchor);
            const P* plane = proto + channel * area;
            a = fmaf(coeff, value(plane, p00), a);
            if (!identity) {
                b = fmaf(coeff, value(plane, p01), b);
                c = fmaf(coeff, value(plane, p10), c);
                e = fmaf(coeff, value(plane, p11), e);
            }
        }
        float score = 0.f;
        if (scores != nullptr) {
            const float confidence = fminf(1.f, fmaxf(0.f, value(head, 4 * anchors + anchor)));
            // Match Python's float32-to-byte rule without another full raster pass.
            score = __double2uint_rn((double)confidence * 255.0) * (1.f / 255.f);
        }
        for (int oy = ymin; oy < ymax; ++oy) {
            const float wy = fourfold ? interpolation_weight_4x(oy, y, ph)
                                      : sample_coord(oy, ph, ih) - (float)y;
            for (int ox = xmin; ox < xmax; ++ox) {
                const float wx = fourfold ? interpolation_weight_4x(ox, x, pw)
                                          : sample_coord(ox, pw, iw) - (float)x;
                // Match CPU retina's weighted-sum evaluation at the zero
                // crossing. Explicit rounding avoids contraction into FMA.
                float logit = a;
                if (!identity) {
                    const float top = __fadd_rn(__fmul_rn(a, 1.f - wx), __fmul_rn(b, wx));
                    const float bottom = __fadd_rn(__fmul_rn(c, 1.f - wx), __fmul_rn(e, wx));
                    logit = __fadd_rn(__fmul_rn(top, 1.f - wy), __fmul_rn(bottom, wy));
                }
                if (logit > 0.f) {
                    const int pixel = oy * iw + ox;
                    mask[pixel] = 1.f;
                    if (scores != nullptr) scores[pixel] = fmaxf(scores[pixel], score);
                }
            }
        }
    }
}

#define MARK(NAME, T) \
extern "C" __global__ void NAME( \
    const T* head, int anchors, const int* indices, const int* count, int xyxy, \
    int ph, int pw, int ih, int iw, int4* boxes, int* owners) { \
    mark_cells(head, anchors, indices, count, xyxy, ph, pw, ih, iw, boxes, owners); \
}
MARK(mark_f16, half)
MARK(mark_f32, float)

#define DECODE(NAME, H, P) \
extern "C" __global__ void NAME( \
    const H* head, const P* proto, int anchors, int channels, int coeff_first, \
    int proto_h, int proto_w, int ph, int pw, int py, int px, int ih, int iw, \
    const int* indices, const int* count, const int4* boxes, const int* owners, \
    float* mask, float* scores) { \
    decode_cells(head, proto, anchors, channels, coeff_first, proto_h, proto_w, \
                 ph, pw, py, px, ih, iw, indices, count, boxes, owners, mask, scores); \
}
DECODE(decode_f16_f16, half, half)
DECODE(decode_f16_f32, half, float)
DECODE(decode_f32_f16, float, half)
DECODE(decode_f32_f32, float, float)
"""


@lru_cache(maxsize=16)
def _kernels(device_id):
    import cupy

    names = ('mark_f16', 'mark_f32', 'decode_f16_f16', 'decode_f16_f32',
             'decode_f32_f16', 'decode_f32_f32')
    with cupy.cuda.Device(device_id):
        module = cupy.RawModule(code=_SOURCE, options=('--std=c++11',), name_expressions=names)
        return {name: module.get_function(name) for name in names}


class RetinaWorkspace:
    """Own fixed-size decoder storage independently of the number of mask pixels."""

    def __init__(self, head, proto, indices, count, image_hw, proto_crop, *, track_confidence):
        import cupy as cp
        import torch

        if head.dtype not in (torch.float16, torch.float32) or proto.dtype not in (torch.float16, torch.float32):
            raise ValueError('Exact CUDA retina requires FP16 or FP32 head and prototype tensors')
        if not head.is_cuda or any(value.device != head.device or not value.is_contiguous()
                                  for value in (head, proto, indices, count)):
            raise ValueError('Exact CUDA retina requires contiguous tensors on one CUDA device')
        self.head_tag = 'f16' if head.dtype == torch.float16 else 'f32'
        self.proto_tag = 'f16' if proto.dtype == torch.float16 else 'f32'
        self.i32 = np.int32
        self.ih, self.iw = map(int, image_hw)
        self.channels, self.proto_h, self.proto_w = map(int, proto.shape)
        crop_y, crop_x = proto_crop
        self.py, self.px = crop_y.start, crop_x.start
        self.ph, self.pw = crop_y.stop - self.py, crop_x.stop - self.px
        self.anchors = int(head.shape[1])
        if min(self.ph, self.pw, self.anchors, self.ih, self.iw) <= 0:
            raise ValueError('Exact CUDA retina requires nonempty dimensions')
        self._boxes = torch.empty((self.anchors, 4), dtype=torch.int32, device=head.device)
        self._owners = torch.empty((self.ph, self.pw), dtype=torch.int32, device=head.device)
        self.mask = torch.empty(image_hw, dtype=torch.float32, device=head.device)
        self.scores = torch.empty_like(self.mask) if track_confidence else None
        self.device_id = int(head.device.index)
        self._cp = cp
        self._inputs = (head, proto, indices, count)
        with cp.cuda.Device(self.device_id):
            self._buffers = tuple(cp.asarray(value) for value in (self._boxes, self._owners, self.mask))
            self._input_ptrs = tuple(cp.asarray(value) for value in self._inputs)
            self._score_ptr = cp.asarray(self.scores) if self.scores is not None else np.uintp(0)

    def decode(self, *, xyxy=False, stream):
        """Decode selected head columns without a host-side count transfer."""
        kernels = _kernels(self.device_id)
        self._owners.zero_()
        self.mask.zero_()
        if self.scores is not None:
            self.scores.zero_()
        i = self.i32
        boxes, owners, mask = self._buffers
        head_ptr, proto_ptr, idx, count_ptr = self._input_ptrs
        with self._cp.cuda.Device(self.device_id):
            kernels['mark_' + self.head_tag](
                (min(128, self.anchors),), (128,),
                (head_ptr, i(self.anchors), idx, count_ptr, i(xyxy), i(self.ph), i(self.pw),
                 i(self.ih), i(self.iw), boxes, owners), stream=stream,
            )
            kernels[f'decode_{self.head_tag}_{self.proto_tag}'](
                ((self.ph * self.pw + 127) // 128,), (128,),
                (head_ptr, proto_ptr, i(self.anchors), i(self.channels), i(6 if xyxy else 5),
                 i(self.proto_h), i(self.proto_w), i(self.ph), i(self.pw), i(self.py), i(self.px),
                 i(self.ih), i(self.iw), idx, count_ptr, boxes, owners, mask, self._score_ptr), stream=stream,
            )
        return self.mask, self.scores
