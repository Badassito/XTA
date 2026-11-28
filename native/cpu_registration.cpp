#include "cpu_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Shape = std::array<Index, 3>;

struct IntegerLabelType {
    const char *dtype;
    bool swap;
};

IntegerLabelType label_type(const Buffer &data) {
    const char *format = data.format();
    const std::uint16_t one = 1;
    bool little = *reinterpret_cast<const unsigned char *>(&one) == 1;
    if (*format == '<' || *format == '>' || *format == '!') {
        little = *format++ == '<';
    } else if (*format == '=' || *format == '@') {
        ++format;
    }
    if (!format[0] || format[1])
        type_error("Registration requires scalar integer labels");
    const bool signed_type = std::strchr("bhilqn", *format) != nullptr;
    if (!signed_type && !std::strchr("BHILQN", *format))
        type_error("Registration requires integer labels");
    switch (data.itemsize()) {
    case 1:
        return {signed_type ? "i1" : "u1", false};
    case 2:
        return {signed_type ? "<i2" : "<u2", !little};
    case 4:
        return {signed_type ? "<i4" : "<u4", !little};
    case 8:
        return {signed_type ? "<i8" : "<u8", !little};
    default:
        type_error("Unsupported registration label width");
    }
}

using Affine = std::array<std::array<double, 4>, 3>;

Affine coefficients(PyObject *object) {
    Buffer matrix(object, "registration transform", 2);
    matrix.require<double>();
    if (matrix.shape(0) != 4 || matrix.shape(1) != 4)
        value_error("Registration requires a 4-by-4 transform");
    Affine values{};
    for (Index axis = 0; axis < 3; ++axis) {
        for (Index column = 0; column < 4; ++column)
            values[axis][column] = matrix.load<double>(axis, column);
    }
    return values;
}

struct Mask {
    std::unique_ptr<Buffer> view;
    Mask(PyObject *object, const Shape &shape) {
        if (object == Py_None)
            return;
        view = std::make_unique<Buffer>(object, "registration observation mask");
        view->require<bool>();
        const int rank = view->ndim();
        if (rank != 2 && rank != 3)
            value_error("Registration mask must be YX or ZYX");
        for (int axis = 0; axis < rank; ++axis)
            if (view->shape(axis) != shape[3 - rank + axis])
                value_error("Registration mask dimensions differ from scan");
    }
    bool observed(Index z, Index y, Index x) const noexcept {
        return !view || (view->ndim() == 2 ? view->load<bool>(y, x) : view->load<bool>(z, y, x));
    }
};

struct RowMap {
    const Affine &matrix;
    double my[3], mz[3];
    RowMap(const Affine &affine, Index y, Index z) : matrix(affine) {
        for (int axis = 0; axis < 3; ++axis) {
            my[axis] = matrix[axis][1] * double(y);
            mz[axis] = matrix[axis][2] * double(z);
        }
    }
    bool sample(double x, const Shape &shape, Shape &index) const {
        for (int axis = 0; axis < 3; ++axis) {
            // Match the separate NumPy multiply/add operations in XYZ order.
            const double shifted = (((matrix[axis][0] * x + my[axis]) + mz[axis]) + matrix[axis][3]) + .5;
            const auto limit = shape[2 - axis];
            if (!(shifted >= 0. && shifted < double(limit)) || shifted >= 0x1p63)
                return false;
            index[2 - axis] = static_cast<Index>(std::floor(shifted));
            if (index[2 - axis] >= limit)
                return false;
        }
        return true;
    }
};

struct Warp {
    const Buffer &labels;
    const Buffer &output;
    const Buffer &coverage;
    const Mask &source_observed;
    const Mask &target_observed;
    const Affine &native;
    const Affine &label;
    Shape source_shape, target_shape;
    const char *background;
    Index first;
    bool shared_grid;
};

struct Counts {
    Index foreground = 0, observed = 0;
};

template <std::size_t N> Counts gather_rows(const Warp &job, bool swap) {
    Counts counts;
    const Shape label_shape{job.labels.shape(0), job.labels.shape(1), job.labels.shape(2)};
    for (Index row = 0; row < job.output.shape(0); ++row) {
        const Index global = job.first + row, z = global / job.target_shape[1],
                    y = global % job.target_shape[1];
        const RowMap native(job.native, y, z), label(job.label, y, z);
        char *destination = job.output.ptr(row, 0);
        char *coverage = job.coverage.ptr(row, 0);
        for (Index x = 0; x < job.target_shape[2]; ++x) {
            Shape index{};
            bool known = job.target_observed.observed(z, y, x) &&
                         native.sample(double(x), job.source_shape, index) &&
                         job.source_observed.observed(index[0], index[1], index[2]);
            if (known && !job.shared_grid)
                known = label.sample(double(x), label_shape, index);
            char *pixel = destination + N * x;
            if (known) {
                const char *src = job.labels.ptr(index[0], index[1], index[2]);
                if (swap)
                    std::reverse_copy(src, src + N, pixel);
                else
                    std::memcpy(pixel, src, N);
                ++counts.observed;
                counts.foreground += std::memcmp(pixel, job.background, N) != 0;
            } else {
                std::memcpy(pixel, job.background, N);
            }
            coverage[x] = static_cast<char>(known);
        }
    }
    return counts;
}

PyObject *warp_rows(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        PyObject *data, *source_mask, *target_mask, *to_native, *to_label, *background;
        Shape source{}, target{};
        Index first, count;
        if (!PyArg_ParseTuple(args, "O(nnn)(nnn)OOOOnnO:registration_warp_rows", &data, &source[0],
                              &source[1], &source[2], &target[0], &target[1], &target[2], &source_mask,
                              &target_mask, &to_native, &to_label, &first, &count, &background))
            return nullptr;
        Buffer labels(data, "registration labels", 3);
        auto type = label_type(labels);
        if (*std::min_element(target.begin(), target.end()) <= 0 ||
            *std::min_element(source.begin(), source.end()) <= 0)
            value_error("Registration grid dimensions must be positive");
        const Index rows = checked_extent(target[0], target[1]);
        if (first < 0 || count < 0 || first > rows || count > rows - first)
            value_error("Invalid registration slab");
        if (!PyBytes_Check(background) || PyBytes_GET_SIZE(background) != labels.itemsize())
            value_error("Registration background must be one little-endian label");
        const Mask sm(source_mask, source), tm(target_mask, target);
        const bool shared = to_label == Py_None;
        if (shared &&
            (labels.shape(0) != source[0] || labels.shape(1) != source[1] || labels.shape(2) != source[2]))
            value_error("Shared registration grids require matching shapes");
        const auto native = coefficients(to_native), label = shared ? native : coefficients(to_label);
        auto output = empty({count, target[2]}, type.dtype), coverage = empty({count, target[2]}, "uint8");
        Buffer out(output.get(), "registered labels", 2, true),
            covered(coverage.get(), "registration coverage", 2, true);
        const Warp job{labels, out,   covered, sm,     tm,
                       native, label, source,  target, PyBytes_AS_STRING(background),
                       first,  shared};
        Counts counts;
        {
            GilRelease unlock;
            switch (labels.itemsize()) {
            case 1:
                counts = gather_rows<1>(job, false);
                break;
            case 2:
                counts = gather_rows<2>(job, type.swap);
                break;
            case 4:
                counts = gather_rows<4>(job, type.swap);
                break;
            case 8:
                counts = gather_rows<8>(job, type.swap);
                break;
            }
        }
        return Py_BuildValue("(NNnn)", output.release(), coverage.release(), counts.foreground,
                             counts.observed);
    });
}

PyMethodDef methods[] = {{"registration_warp_rows", warp_rows, METH_VARARGS,
                          "Gather an ordered row slab of categorical labels and observation coverage."},
                         {nullptr, nullptr, 0, nullptr}};

} // namespace
int register_registration(PyObject *module) { return PyModule_AddFunctions(module, methods); }
} // namespace xta::cpu
