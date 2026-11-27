#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

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
        {
            GilRelease release;
            for (Index y = 0; y < height; ++y) {
                for (Index x = 0; x < width; ++x) {
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

template<class Label, class Distance>
void reduce_component_maxima(const Buffer& labels, const Buffer& distances,
                             const Buffer& maxima, Index count) noexcept {
    for (Index y = 0; y < labels.shape(0); ++y) {
        for (Index x = 0; x < labels.shape(1); ++x) {
            const auto label = labels.load<Label>(y, x);
            if (label <= 0 || static_cast<std::uint64_t>(label) > static_cast<std::uint64_t>(count)) {
                continue;
            }
            const Index slot = static_cast<Index>(label) - 1;
            const double distance = static_cast<double>(distances.load<Distance>(y, x));
            if (distance > maxima.load<double>(slot)) {
                maxima.store(distance, slot);
            }
        }
    }
}

using MaximaReducer = void (*)(const Buffer&, const Buffer&, const Buffer&, Index);

template<class Distance>
MaximaReducer component_maxima_reducer(const Buffer& labels) {
    if (labels.is<std::int32_t>()) return reduce_component_maxima<std::int32_t, Distance>;
    if (labels.is<std::int64_t>()) return reduce_component_maxima<std::int64_t, Distance>;
    if (labels.is<std::uint32_t>()) return reduce_component_maxima<std::uint32_t, Distance>;
    if (labels.is<std::uint64_t>()) return reduce_component_maxima<std::uint64_t, Distance>;
    if (labels.is<std::int16_t>()) return reduce_component_maxima<std::int16_t, Distance>;
    if (labels.is<std::uint16_t>()) return reduce_component_maxima<std::uint16_t, Distance>;
    if (labels.is<std::int8_t>()) return reduce_component_maxima<std::int8_t, Distance>;
    if (labels.is<std::uint8_t>()) return reduce_component_maxima<std::uint8_t, Distance>;
    type_error("SAM component labels must be native integers");
}

PyObject* component_maxima(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *label_input, *distance_input;
        Index count;
        if (!PyArg_ParseTuple(args, "OOn:sam_component_maxima", &label_input, &distance_input, &count)) {
            throw PythonError{};
        }
        const Buffer labels(label_input, "SAM component labels", 2);
        const Buffer distances(distance_input, "SAM component distances", 2);
        if (count < 0 || labels.shape(0) != distances.shape(0) || labels.shape(1) != distances.shape(1)) {
            value_error("SAM component count and array shapes are inconsistent");
        }
        MaximaReducer reduce;
        if (distances.is<double>()) {
            reduce = component_maxima_reducer<double>(labels);
        } else {
            distances.require<float>();
            reduce = component_maxima_reducer<float>(labels);
        }
        Object result = zeros({count}, "float64");
        {
            const Buffer maxima(result.get(), "SAM component maxima", 1, true);
            GilRelease release;
            reduce(labels, distances, maxima, count);
        }
        return result.release();
    });
}

PyMethodDef methods[] = {
    {"max_float32_magnitude_bits", magnitude_bits, METH_O, "Find the maximum unsigned float32 magnitude word."},
    {"threshold_retina_roi", threshold_roi, METH_VARARGS, "Threshold a retina ROI with canonical replay."},
    {"union_conf_slice_inplace", union_conf_slice, METH_VARARGS, "Merge foreground and confidence into a slice."},
    {"sam_component_maxima", component_maxima, METH_VARARGS, "Reduce the maximum distance of each SAM component."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_retina(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
