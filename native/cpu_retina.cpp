#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Byte = std::uint8_t;

PyObject* magnitude_bits(PyObject*, PyObject* input) {
    return protect([&]() -> PyObject* {
        const Buffer words(input, "prototype magnitude words", 1);
        words.require<std::uint32_t>();
        std::uint32_t largest = 0;
        {
            GilRelease release;
            for (Index i = 0; i < words.shape(0); ++i) {
                largest = std::max(largest, words.load<std::uint32_t>(i) & std::uint32_t{0x7fffffffU});
            }
        }
        return PyLong_FromUnsignedLong(largest);
    });
}

struct Coordinate {
    Index first;
    Index second;
    float weight;
};

Coordinate source_coordinate(Index index, Index size, float scale) {
    const float value = (static_cast<float>(index) + 0.5f) * scale - 0.5f;
    const float clipped = std::min(static_cast<float>(size - 1), std::max(0.0f, value));
    // A huge last valid index can round up to the first invalid index in binary32.
    if (static_cast<long double>(clipped) >= static_cast<long double>(size)) {
        throw std::overflow_error("Retina source coordinate exceeds its input extent");
    }
    const Index first = static_cast<Index>(clipped);
    return {first, std::min(first + 1, size - 1), clipped - static_cast<float>(first)};
}

float bilinear(float a, float b, float c, float d, float wx, float wy) noexcept {
    // The baseline rounds each multiply and add here. Only the channel dot is fused.
    const float top = a * (1.0f - wx) + b * wx;
    const float bottom = c * (1.0f - wx) + d * wx;
    return top * (1.0f - wy) + bottom * wy;
}

Index clipped_box_edge(float edge, Index extent) noexcept {
    if (edge <= 0.0f) return 0;
    if (static_cast<double>(edge) >= static_cast<double>(extent)) return extent;
    return static_cast<Index>(std::ceil(static_cast<double>(edge)));
}

Byte retina_confidence_score(float confidence) noexcept {
    // The public Python quantizer clamps NaN to zero and scales in binary64.
    if (!(confidence > 0.0f)) return 0;
    if (confidence >= 1.0f) return 255;
    const double value = static_cast<double>(confidence) * 255.0;
    const auto whole = static_cast<unsigned>(value);
    const double fraction = value - whole;
    return static_cast<Byte>(whole + (fraction > 0.5 || (fraction == 0.5 && (whole & 1u))));
}

void accumulate_logit_row(float coefficient, const char* proto_row, Index proto_step,
                          float* logits, Index width) noexcept {
    for (Index x = 0; x < width; ++x) {
        float value;
        std::memcpy(&value, proto_row + x * proto_step, sizeof(value));
        logits[x] = std::fma(coefficient, value, logits[x]);
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx,fma")))
void accumulate_logit_row_fma(float coefficient, const char* proto_row,
                              float* logits, Index width) noexcept {
    const auto weight = _mm256_set1_ps(coefficient);
    Index x = 0;
    for (; x <= width - 8; x += 8) {
        __m256 source;
        std::memcpy(&source, proto_row + x * sizeof(float), sizeof(source));
        const auto values = _mm256_loadu_ps(logits + x);
        _mm256_storeu_ps(logits + x, _mm256_fmadd_ps(weight, source, values));
    }
    accumulate_logit_row(coefficient, proto_row + x * sizeof(float), sizeof(float),
                         logits + x, width - x);
}
#endif

PyObject* accumulate_retina_rois(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *proto_input, *coeff_input, *boxes_input, *confs_input;
        Index crop_y, crop_x, crop_h, crop_w, out_h, out_w;
        int track_confidence;
        if (!PyArg_ParseTuple(args, "OOOOnnnnnnp:accumulate_retina_rois",
                              &proto_input, &coeff_input, &boxes_input, &confs_input,
                              &crop_y, &crop_x, &crop_h, &crop_w, &out_h, &out_w,
                              &track_confidence)) {
            throw PythonError{};
        }
        const Buffer proto(proto_input, "Retina prototypes", 3);
        const Buffer coefficients(coeff_input, "Retina coefficients", 2);
        const Buffer boxes(boxes_input, "Retina bounding boxes", 2);
        const Buffer confidences(confs_input, "Retina confidences", 1);
        proto.require<float>();
        coefficients.require<float>();
        boxes.require<float>();
        confidences.require<float>();
        if (proto.shape(0) <= 0 || coefficients.shape(1) != proto.shape(0)
            || (boxes.shape(0) && boxes.shape(1) < 4) || out_h <= 0 || out_w <= 0
            || crop_y < 0 || crop_x < 0 || crop_h <= 0 || crop_w <= 0
            || crop_y > proto.shape(1) || crop_h > proto.shape(1) - crop_y
            || crop_x > proto.shape(2) || crop_w > proto.shape(2) - crop_x) {
            value_error("Retina crop, detection and prototype shapes are inconsistent");
        }

        auto union_array = zeros({out_h, out_w}, "uint8");
        Object score_array;
        const Buffer frame_union(union_array.get(), "Retina union", 2, true);
        std::optional<Buffer> frame_scores;
        if (track_confidence) {
            score_array = zeros({out_h, out_w}, "uint8");
            frame_scores.emplace(score_array.get(), "Retina scores", 2, true);
        }
        Index kept = 0;
        {
            GilRelease release;
            const bool identity = crop_h == out_h && crop_w == out_w;
            const float scale_y = static_cast<float>(static_cast<double>(crop_h) / out_h);
            const float scale_x = static_cast<float>(static_cast<double>(crop_w) / out_w);
            std::vector<float> logits;
            std::vector<Coordinate> x_coordinates;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            const bool vector_rows = proto.stride(2) == static_cast<Index>(sizeof(float))
                && __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
#endif
            const Index detections = std::min(coefficients.shape(0), boxes.shape(0));
            for (Index detection = 0; detection < detections; ++detection) {
                const float left = boxes.load<float>(detection, 0);
                const float top = boxes.load<float>(detection, 1);
                const float right = boxes.load<float>(detection, 2);
                const float bottom = boxes.load<float>(detection, 3);
                bool finite = std::isfinite(left) && std::isfinite(top)
                    && std::isfinite(right) && std::isfinite(bottom);
                for (Index field = 4; finite && field < boxes.shape(1); ++field) {
                    finite = std::isfinite(boxes.load<float>(detection, field));
                }
                if (!finite) continue;

                const Index first_x = clipped_box_edge(left, out_w);
                const Index first_y = clipped_box_edge(top, out_h);
                const Index stop_x = clipped_box_edge(right, out_w);
                const Index stop_y = clipped_box_edge(bottom, out_h);
                if (stop_x <= first_x || stop_y <= first_y) continue;

                // Even identity resizing uses the canonical binary32 pixel-center
                // calculation, which can round onto an adjacent cell at wide extents.
                const Index proto_first_y = source_coordinate(first_y, crop_h, scale_y).first;
                const Index proto_first_x = source_coordinate(first_x, crop_w, scale_x).first;
                const auto last_y = source_coordinate(stop_y - 1, crop_h, scale_y);
                const auto last_x = source_coordinate(stop_x - 1, crop_w, scale_x);
                const Index proto_last_y = identity ? last_y.first : last_y.second;
                const Index proto_last_x = identity ? last_x.first : last_x.second;
                const Index width = proto_last_x - proto_first_x + 1;
                const Index height = proto_last_y - proto_first_y + 1;
                logits.assign(static_cast<std::size_t>(checked_extent(height, width)), 0.0f);

                // Vectorize across spatial cells only. Every cell still sees
                // the canonical sequence of float32 fused channel updates.
                for (Index channel = 0; channel < proto.shape(0); ++channel) {
                    const float coefficient = coefficients.load<float>(detection, channel);
                    for (Index y = 0; y < height; ++y) {
                        const auto* src = proto.ptr(channel, crop_y + proto_first_y + y,
                                                    crop_x + proto_first_x);
                        auto* dst = logits.data() + y * width;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
                        if (vector_rows && width >= 8) {
                            accumulate_logit_row_fma(coefficient, src, dst, width);
                            continue;
                        }
#endif
                        accumulate_logit_row(coefficient, src, proto.stride(2), dst, width);
                    }
                }

                x_coordinates.clear();
                x_coordinates.reserve(stop_x - first_x);
                for (Index x = first_x; x < stop_x; ++x) {
                    auto coordinate = source_coordinate(x, crop_w, scale_x);
                    coordinate.first -= proto_first_x;
                    coordinate.second -= proto_first_x;
                    x_coordinates.push_back(coordinate);
                }
                const Byte score = frame_scores && detection < confidences.shape(0)
                    ? retina_confidence_score(confidences.load<float>(detection)) : 0;
                bool foreground = false;
                for (Index y = first_y; y < stop_y; ++y) {
                    const auto coordinate = source_coordinate(y, crop_h, scale_y);
                    const auto* row0 = logits.data() + (coordinate.first - proto_first_y) * width;
                    const auto* row1 = identity ? row0
                        : logits.data() + (coordinate.second - proto_first_y) * width;
                    for (Index x = first_x; x < stop_x; ++x) {
                        const auto sample = x_coordinates[x - first_x];
                        float value = row0[sample.first];
                        if (!identity) {
                            value = bilinear(value, row0[sample.second],
                                             row1[sample.first], row1[sample.second],
                                             sample.weight, coordinate.weight);
                        }
                        if (!(value > 0.0f)) continue;
                        foreground = true;
                        frame_union.store<Byte>(1, y, x);
                        if (frame_scores && score > frame_scores->load<Byte>(y, x)) {
                            frame_scores->store<Byte>(score, y, x);
                        }
                    }
                }
                kept += foreground;
            }
        }
        Object count(PyLong_FromSsize_t(kept));
        return PyTuple_Pack(3, union_array.get(),
                            track_confidence ? score_array.get() : Py_None, count.get());
    });
}

struct CanonicalLogits {
    const Buffer& prototype;
    const Buffer& coefficients;
    Index stride;
    Index crop_y;
    Index crop_x;
    Index width;
    Index area;
    std::unique_ptr<float[]> cache;
    std::vector<Byte> known;

    float get(Index y, Index x) {
        if (!cache) {
            cache.reset(new float[area]);
            known.resize(area, 0);
        }
        const Index key = y * width + x;
        if (!known[key]) {
            const Index address = (y + crop_y) * stride + x + crop_x;
            float value = 0.0f;
            for (Index channel = 0; channel < coefficients.shape(0); ++channel) {
                value = std::fma(coefficients.load<float>(channel),
                                 prototype.load<float>(channel, address), value);
            }
            cache[key] = value;
            known[key] = 1;
        }
        return cache[key];
    }
};

PyObject* threshold_roi(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *low_input, *proto_input, *coeff_input;
        double bound;
        Index proto_width, crop_y, crop_x, out_h, out_w, first_y, stop_y, first_x, stop_x;
        if (!PyArg_ParseTuple(args, "OdOOnnnnnnnnn:threshold_retina_roi",
                              &low_input, &bound, &proto_input, &coeff_input,
                              &proto_width, &crop_y, &crop_x, &out_h, &out_w,
                              &first_y, &stop_y, &first_x, &stop_x)) {
            throw PythonError{};
        }
        const Buffer low(low_input, "retina low logits", 2);
        const Buffer proto(proto_input, "retina prototypes", 2);
        const Buffer coeff(coeff_input, "retina coefficients", 1);
        low.require<float>();
        proto.require<float>();
        coeff.require<float>();
        const Index low_h = low.shape(0);
        const Index low_w = low.shape(1);
        if (out_h <= 0 || out_w <= 0 || first_y < 0 || first_x < 0
            || stop_y < first_y || stop_x < first_x || stop_y > out_h || stop_x > out_w) {
            value_error("Retina ROI must lie inside a positive output raster");
        }
        const Index height = stop_y - first_y;
        const Index width = stop_x - first_x;
        if (height == 0 || width == 0) {
            return empty({height, width}, "bool").release();
        }
        if (low_h == 0 || low_w == 0 || crop_y < 0 || crop_x < 0
            || proto_width < low_w || crop_x > proto_width - low_w
            || coeff.shape(0) > proto.shape(0)) {
            value_error("Retina logit crop and prototype shapes are inconsistent");
        }
        const Index last_row = checked_add(crop_y, low_h - 1);
        const Index last_column = checked_add(crop_x, low_w - 1);
        const Index last_address = checked_add(checked_extent(last_row, proto_width), last_column);
        if (coeff.shape(0) != 0 && last_address >= proto.shape(1)) {
            value_error("Retina logit crop extends beyond the prototypes");
        }
        const Index low_area = checked_extent(low_h, low_w);
        Object result = empty({height, width}, "bool");
        {
            const Buffer mask(result.get(), "retina ROI mask", 2, true);
            GilRelease release;
            const float scale_y = static_cast<float>(static_cast<double>(low_h) / out_h);
            const float scale_x = static_cast<float>(static_cast<double>(low_w) / out_w);
            const bool identity = low_h == out_h && low_w == out_w;
            std::vector<Coordinate> x_coordinates;
            x_coordinates.reserve(width);
            for (Index x = first_x; x < stop_x; ++x) {
                x_coordinates.push_back(source_coordinate(x, low_w, scale_x));
            }
            CanonicalLogits replay{proto, coeff, proto_width, crop_y, crop_x, low_w, low_area, {}, {}};
            for (Index dy = 0; dy < height; ++dy) {
                const auto y = source_coordinate(first_y + dy, low_h, scale_y);
                for (Index dx = 0; dx < width; ++dx) {
                    const auto x = x_coordinates[dx];
                    float value = low.load<float>(y.first, x.first);
                    if (!identity) {
                        value = bilinear(value, low.load<float>(y.first, x.second),
                                         low.load<float>(y.second, x.first), low.load<float>(y.second, x.second),
                                         x.weight, y.weight);
                    }
                    if (!std::isfinite(value) || !(std::abs(value) > bound)) {
                        value = replay.get(y.first, x.first);
                        if (!identity) {
                            const float b = replay.get(y.first, x.second);
                            const float c = replay.get(y.second, x.first);
                            const float d = replay.get(y.second, x.second);
                            value = bilinear(value, b, c, d, x.weight, y.weight);
                        }
                    }
                    mask.store<bool>(value > 0.0f, dy, dx);
                }
            }
        }
        return result.release();
    });
}

#if defined(_M_X64) || defined(__SSE2__)
bool union_rows_do_not_overlap(const std::array<const Buffer*, 4>& planes,
                               Index height, Index width) noexcept {
    if (height == 0 || width < 16) return false;
    std::array<std::uintptr_t, 4> begin, end;
    for (std::size_t i = 0; i < planes.size(); ++i) {
        const Buffer& plane = *planes[i];
        if (plane.stride(1) != 1 || plane.stride(0) < width) return false;
        begin[i] = reinterpret_cast<std::uintptr_t>(plane.data());
        // Buffer has checked that the last element's offset is representable.
        const auto last = static_cast<std::uintptr_t>((height - 1) * plane.stride(0) + (width - 1));
        if (last > std::numeric_limits<std::uintptr_t>::max() - begin[i]) return false;
        end[i] = begin[i] + last;
        for (std::size_t j = 0; j < i; ++j) {
            if (begin[i] <= end[j] && begin[j] <= end[i]) return false;
        }
    }
    return true;
}

Index union_conf_vector_row(char* dst_mask, char* dst_conf, const char* src_mask,
                            const char* src_conf, Index width) noexcept {
    const auto zero = _mm_setzero_si128();
    const auto sign = _mm_set1_epi8(-128);
    Index x = 0;
    for (; x <= width - 16; x += 16) {
        __m128i mask;
        std::memcpy(&mask, src_mask + x, sizeof(mask));
        const auto background = _mm_cmpeq_epi8(mask, zero);
        if (_mm_movemask_epi8(background) == 0xffff) continue;

        __m128i previous_mask, previous_conf, confidence;
        std::memcpy(&previous_mask, dst_mask + x, sizeof(previous_mask));
        std::memcpy(&previous_conf, dst_conf + x, sizeof(previous_conf));
        std::memcpy(&confidence, src_conf + x, sizeof(confidence));
        const auto higher = _mm_cmpgt_epi8(_mm_xor_si128(confidence, sign),
                                          _mm_xor_si128(previous_conf, sign));
        const auto replace = _mm_andnot_si128(background,
            _mm_or_si128(_mm_cmpeq_epi8(previous_mask, zero), higher));
        const auto next_mask = _mm_or_si128(_mm_and_si128(replace, mask),
                                          _mm_andnot_si128(replace, previous_mask));
        const auto next_conf = _mm_or_si128(_mm_and_si128(replace, confidence),
                                          _mm_andnot_si128(replace, previous_conf));
        std::memcpy(dst_mask + x, &next_mask, sizeof(next_mask));
        std::memcpy(dst_conf + x, &next_conf, sizeof(next_conf));
    }
    return x;
}
#endif

PyObject* union_conf_slice(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *dst_mask_input, *dst_conf_input, *src_mask_input, *src_conf_input;
        if (!PyArg_ParseTuple(args, "OOOO:union_conf_slice_inplace",
                              &dst_mask_input, &dst_conf_input, &src_mask_input, &src_conf_input)) {
            throw PythonError{};
        }
        const Buffer dst_mask(dst_mask_input, "union destination mask", 2, true);
        const Buffer dst_conf(dst_conf_input, "union destination confidence", 2, true);
        const Buffer src_mask(src_mask_input, "union source mask", 2);
        const Buffer src_conf(src_conf_input, "union source confidence", 2);
        const Index height = dst_mask.shape(0);
        const Index width = dst_mask.shape(1);
        for (const Buffer* plane : {&dst_mask, &dst_conf, &src_mask, &src_conf}) {
            plane->require<Byte>();
            if (plane->shape(0) != height || plane->shape(1) != width) {
                value_error("Union mask and confidence planes must have the same shape");
            }
        }
#if defined(_M_X64) || defined(__SSE2__)
        const bool vector_rows = union_rows_do_not_overlap(
            {&dst_mask, &dst_conf, &src_mask, &src_conf}, height, width);
#endif
        {
            GilRelease release;
            for (Index y = 0; y < height; ++y) {
                Index first = 0;
#if defined(_M_X64) || defined(__SSE2__)
                if (vector_rows) {
                    first = union_conf_vector_row(dst_mask.ptr(y, 0), dst_conf.ptr(y, 0),
                                                  src_mask.ptr(y, 0), src_conf.ptr(y, 0), width);
                }
#endif
                for (Index x = first; x < width; ++x) {
                    const auto mask = src_mask.load<Byte>(y, x);
                    const auto confidence = src_conf.load<Byte>(y, x);
                    if (mask != 0 && (dst_mask.load<Byte>(y, x) == 0 || confidence > dst_conf.load<Byte>(y, x))) {
                        dst_mask.store(mask, y, x);
                        dst_conf.store(confidence, y, x);
                    }
                }
            }
        }
        Py_RETURN_NONE;
    });
}

template<class Label, class Value, class Maximum, bool UnitStride>
void reduce_component_row(const char* labels, Index label_step, const char* values, Index value_step,
                          const Buffer& maxima, Index count, Index width) noexcept {
    const Index label_stride = UnitStride ? static_cast<Index>(sizeof(Label)) : label_step;
    const Index value_stride = UnitStride ? static_cast<Index>(sizeof(Value)) : value_step;
    for (Index x = 0; x < width; ++x) {
        Label label;
        std::memcpy(&label, labels + x * label_stride, sizeof(label));
        if (label <= 0 || static_cast<std::uint64_t>(label) > static_cast<std::uint64_t>(count)) {
            continue;
        }
        const Index slot = static_cast<Index>(label) - 1;
        Value sample;
        std::memcpy(&sample, values + x * value_stride, sizeof(sample));
        const auto value = static_cast<Maximum>(sample);
        if (value > maxima.load<Maximum>(slot)) {
            maxima.store(value, slot);
        }
    }
}

template<class Label, class Value, class Maximum>
void reduce_component_maxima(const Buffer& labels, const Buffer& values,
                             const Buffer& maxima, Index count) noexcept {
    const Index width = labels.shape(1);
    if (width == 0 || count == 0) return;
    const Index label_step = labels.stride(1), value_step = values.stride(1);
    // The inner row may be contiguous even when a crop retains the larger pitch.
    // memcpy handles unaligned buffers in both paths.
    const auto reduce = label_step == static_cast<Index>(sizeof(Label))
                         && value_step == static_cast<Index>(sizeof(Value))
        ? reduce_component_row<Label, Value, Maximum, true>
        : reduce_component_row<Label, Value, Maximum, false>;
    for (Index y = 0; y < labels.shape(0); ++y) {
        reduce(labels.ptr(y, 0), label_step, values.ptr(y, 0), value_step, maxima, count, width);
    }
}

using MaximaReducer = void (*)(const Buffer&, const Buffer&, const Buffer&, Index);

template<class Value, class Maximum>
MaximaReducer component_maxima_reducer(const Buffer& labels) {
    if (labels.is<std::int32_t>()) return reduce_component_maxima<std::int32_t, Value, Maximum>;
    if (labels.is<std::int64_t>()) return reduce_component_maxima<std::int64_t, Value, Maximum>;
    if (labels.is<std::uint32_t>()) return reduce_component_maxima<std::uint32_t, Value, Maximum>;
    if (labels.is<std::uint64_t>()) return reduce_component_maxima<std::uint64_t, Value, Maximum>;
    if (labels.is<std::int16_t>()) return reduce_component_maxima<std::int16_t, Value, Maximum>;
    if (labels.is<std::uint16_t>()) return reduce_component_maxima<std::uint16_t, Value, Maximum>;
    if (labels.is<std::int8_t>()) return reduce_component_maxima<std::int8_t, Value, Maximum>;
    if (labels.is<std::uint8_t>()) return reduce_component_maxima<std::uint8_t, Value, Maximum>;
    type_error("Component labels must be native integers");
}

template<class Value, class Maximum>
PyObject* component_maxima_result(const Buffer& labels, const Buffer& values, Index count) {
    const auto reduce = component_maxima_reducer<Value, Maximum>(labels);
    constexpr auto dtype = std::is_same_v<Maximum, Byte> ? "uint8"
        : std::is_same_v<Maximum, float> ? "float32" : "float64";
    auto result = zeros({count}, dtype);
    {
        const Buffer maxima(result.get(), "component maxima", 1, true);
        maxima.require<Maximum>();
        if (maxima.shape(0) != count) value_error("Component maxima allocator returned the wrong shape");
        GilRelease release;
        reduce(labels, values, maxima, count);
    }
    return result.release();
}

template<bool PreserveValueDtype>
PyObject* component_maxima(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *label_input, *value_input;
        Index count;
        if (!PyArg_ParseTuple(args, "OOn:component_maxima", &label_input, &value_input, &count)) {
            throw PythonError{};
        }
        const Buffer labels(label_input, "component labels", 2);
        const Buffer values(value_input, "component values", 2);
        if (count < 0 || labels.shape(0) != values.shape(0) || labels.shape(1) != values.shape(1)) {
            value_error("Component count and array shapes are inconsistent");
        }
        if (values.is<double>()) {
            return component_maxima_result<double, double>(labels, values, count);
        }
        if (values.is<Byte>()) {
            return component_maxima_result<Byte, std::conditional_t<PreserveValueDtype, Byte, double>>(
                labels, values, count);
        }
        values.require<float>();
        return component_maxima_result<float, std::conditional_t<PreserveValueDtype, float, double>>(
            labels, values, count);
    });
}

PyMethodDef methods[] = {
    {"accumulate_retina_rois", accumulate_retina_rois, METH_VARARGS, "Accumulate canonical Retina ROIs."},
    {"max_float32_magnitude_bits", magnitude_bits, METH_O, "Find the maximum unsigned float32 magnitude word."},
    {"threshold_retina_roi", threshold_roi, METH_VARARGS, "Threshold a retina ROI with canonical replay."},
    {"union_conf_slice_inplace", union_conf_slice, METH_VARARGS, "Merge foreground and confidence into a slice."},
    {"component_maxima", component_maxima<true>, METH_VARARGS, "Reduce nonnegative values by one-based component label."},
    {"sam_component_maxima", component_maxima<false>, METH_VARARGS, "Reduce component maxima to float64."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_retina(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
