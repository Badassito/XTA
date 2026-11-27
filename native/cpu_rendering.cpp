#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xta::cpu {
namespace {

using Size = Py_ssize_t;
using Byte = std::uint8_t;
constexpr double pi = 3.141592653589793238462643383279502884;

double positive_remainder(double value, double period) {
    const double rest = std::fmod(value, period);
    return rest == 0. ? 0. : (rest < 0. ? rest + period : rest);
}

struct Box {
    Size top, left, bottom, right;

    void validate() const {
        if (top < 0 || left < 0 || top >= bottom || left >= right) {
            value_error("Shell crop bounds must describe a nonempty, nonnegative rectangle");
        }
    }

    Size height() const { return bottom - top; }
    Size width() const { return right - left; }
};

void validate_source(const Buffer& source) {
    source.require<Byte>();
    if (!source.shape(0) || !source.shape(1) || !source.shape(2)) {
        value_error("Shell rendering requires a nonempty physical source");
    }
}

Byte gray8_trilinear(const Buffer& source, double time, double row, double column) {
    if (!(-1. < time && time < source.shape(0) && -1. < row && row < source.shape(1)
          && -1. < column && column < source.shape(2))) {
        return 0;
    }
    const Size t0 = static_cast<Size>(std::floor(time));
    const Size y0 = static_cast<Size>(std::floor(row));
    const Size x0 = static_cast<Size>(std::floor(column));
    const float dt = static_cast<float>(time - t0);
    const float dy = static_cast<float>(row - y0);
    const float dx = static_cast<float>(column - x0);
    float accumulator = 0.f;
    for (Size it = 0; it < 2; ++it) {
        const auto ti = t0 + it;
        const float wt = it ? dt : 1.f - dt;
        if (ti < 0 || ti >= source.shape(0)) continue;
        for (Size iy = 0; iy < 2; ++iy) {
            const auto yi = y0 + iy;
            const float wy = iy ? dy : 1.f - dy;
            if (yi < 0 || yi >= source.shape(1)) continue;
            for (Size ix = 0; ix < 2; ++ix) {
                const auto xi = x0 + ix;
                const float wx = ix ? dx : 1.f - dx;
                if (xi < 0 || xi >= source.shape(2)) continue;
                const float weight = (wt * wy) * wx;
                const float term = static_cast<float>(source.load<Byte>(ti, yi, xi)) * weight;
                accumulator = accumulator + term;
            }
        }
    }
    // All taps are nonnegative, and the FP32 sum can exceed 255 slightly.
    const float value = std::min(255.f, accumulator);
    const auto whole = static_cast<unsigned>(value);
    const float fraction = value - static_cast<float>(whole);
    return static_cast<Byte>(whole + (fraction > .5f || (fraction == .5f && (whole & 1))));
}

struct RadialColumn {
    double x, y, offset;
};

PyObject* render_radial_gray8(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* array;
        int base, tilted, vertical;
        double radius, center_x, center_y, arc, height_origin, tangent;
        Box crop;
        if (!PyArg_ParseTuple(args, "Oidddddppdnnnn", &array, &base, &radius, &center_x, &center_y,
                              &arc, &height_origin, &tilted, &vertical, &tangent,
                              &crop.top, &crop.left, &crop.bottom, &crop.right)) return nullptr;
        const Buffer source(array, "Radial source", 3);
        validate_source(source);
        crop.validate();
        if (base < 0 || base > 2 || !std::isfinite(radius) || !std::isfinite(center_x)
            || !std::isfinite(center_y) || !std::isfinite(arc) || !std::isfinite(height_origin)
            || !std::isfinite(tangent)) {
            value_error("Radial source axes and geometry must be valid");
        }
        auto result = zeros({crop.height(), crop.width()}, "uint8");
        if (radius == 0.) return result.release();
        const Buffer output(result.get(), "Radial crop", 2, true);
        {
            GilRelease unlocked;
            std::vector<RadialColumn> columns(static_cast<std::size_t>(crop.width()));
            for (Size column = 0; column < crop.width(); ++column) {
                const double theta = positive_remainder((arc + static_cast<double>(crop.left + column)) / radius, 2. * pi);
                const double x = center_x + radius * std::cos(theta);
                const double y = center_y + radius * std::sin(theta);
                const double offset = tilted ? tangent * (vertical ? y - center_y : x - center_x) : 0.;
                columns[column] = {x, y, offset};
            }
            const Size last_height = source.shape(base) - 1;
            for (Size row = 0; row < crop.height(); ++row) {
                const double height = height_origin + static_cast<double>(crop.top + row);
                if (height < 0. || height > last_height) continue;
                for (Size column = 0; column < crop.width(); ++column) {
                    const auto& point = columns[column];
                    const double stack = tilted ? height + point.offset : height;
                    Byte value;
                    if (base == 0) value = gray8_trilinear(source, stack, point.y, point.x);
                    else if (base == 1) value = gray8_trilinear(source, point.y, stack, point.x);
                    else value = gray8_trilinear(source, point.y, point.x, stack);
                    output.store(value, row, column);
                }
            }
        }
        return result.release();
    });
}

std::array<double, 3> spherical_direction(int face, double u, double v) {
    const double inverse_sqrt_two = 1. / std::sqrt(2.);
    const double abs_u = std::abs(u), abs_v = std::abs(v);
    const int area = abs_u >= abs_v ? (u >= 0. ? 0 : 2) : (v >= 0. ? 1 : 3);
    const double p = std::max(abs_u, abs_v);
    const double minor_axis = area == 0 ? v : (area == 1 ? -u : (area == 2 ? -v : u));
    const double ratio = p == 0. ? 0. : minor_axis / p;
    const double tau = (pi / 12.) * ratio;
    const double numerator = std::sin(tau), denominator = std::cos(tau) - inverse_sqrt_two;
#ifdef _WIN32
    const double norm = _hypot(numerator, denominator);
#else
    const double norm = std::hypot(numerator, denominator);
#endif
    double cosine = denominator / norm, sine = numerator / norm;
    const bool diagonal = std::abs(ratio) == 1.;
    if (diagonal) {
        cosine = inverse_sqrt_two;
        sine = ratio > 0. ? inverse_sqrt_two : -inverse_sqrt_two;
    }
    const double d = 1. - cosine / std::sqrt(1. + cosine * cosine);
    const double normal = 1. - p * p * d;
    const double tangent = p * std::sqrt(d * (2. - p * p * d));
    double major = tangent * cosine, minor = tangent * sine;
    if (p == 1.) {
        major = normal;
        if (diagonal) minor = ratio > 0. ? normal : -normal;
    }
    const double right = area == 0 ? major : (area == 1 ? -minor : (area == 2 ? -major : minor));
    const double up = area == 0 ? minor : (area == 1 ? major : (area == 2 ? -minor : -major));
    switch (face) {
        case 0: return {normal, right, up};
        case 1: return {-right, normal, up};
        case 2: return {-normal, -right, up};
        case 3: return {right, -normal, up};
        case 4: return {-up, right, normal};
        default: return {up, right, -normal};
    }
}

Size chart_index(Size index, Size origin, Size intervals) {
    // Skip pixels beyond the chart before adding an origin near the index limit.
    return origin > intervals - index ? -1 : index + origin;
}

PyObject* render_spherical_gray8(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *array, *rotation_array;
        int face;
        Size intervals, origin_u, origin_v;
        double radius;
        Box crop;
        if (!PyArg_ParseTuple(args, "OinnnOdnnnn", &array, &face, &intervals, &origin_u, &origin_v,
                              &rotation_array, &radius, &crop.top, &crop.left, &crop.bottom, &crop.right)) return nullptr;
        const Buffer source(array, "Spherical source", 3), rotation(rotation_array, "QSC rotation", 2);
        validate_source(source);
        rotation.require<double>();
        crop.validate();
        if (face < 0 || face > 5 || intervals <= 0 || !std::isfinite(radius)
            || rotation.shape(0) != 3 || rotation.shape(1) != 3) {
            value_error("Spherical face, chart, rotation and radius must be valid");
        }
        double matrix[3][3];
        for (Size axis = 0; axis < 3; ++axis) {
            for (Size component = 0; component < 3; ++component) {
                matrix[axis][component] = rotation.load<double>(axis, component);
                if (!std::isfinite(matrix[axis][component])) value_error("Spherical rotation must be finite");
            }
        }
        auto result = zeros({crop.height(), crop.width()}, "uint8");
        const Buffer output(result.get(), "Spherical crop", 2, true);
        const double centers[] = {(source.shape(2) - 1) / 2., (source.shape(1) - 1) / 2., (source.shape(0) - 1) / 2.};
        {
            GilRelease unlocked;
            for (Size row = crop.top; row < crop.bottom; ++row) {
                const Size chart_row = chart_index(row, origin_v, intervals);
                if (chart_row < 0) continue;
                const double v = 1. - 2. * chart_row / intervals;
                for (Size column = crop.left; column < crop.right; ++column) {
                    const Size chart_column = chart_index(column, origin_u, intervals);
                    if (chart_column < 0) continue;
                    const double u = -1. + 2. * chart_column / intervals;
                    const auto direction = spherical_direction(face, u, v);
                    const double lx = direction[0] + 0., ly = direction[1] + 0., lz = direction[2] + 0.;
                    double position[3];
                    for (Size axis = 0; axis < 3; ++axis) {
                        const auto& coefficients = matrix[axis];
                        position[axis] = centers[axis] + radius * ((lx * coefficients[0] + ly * coefficients[1]) + lz * coefficients[2]);
                    }
                    output.store(gray8_trilinear(source, position[2], position[1], position[0]),
                                 row - crop.top, column - crop.left);
                }
            }
        }
        return result.release();
    });
}

PyObject* nearest_azimuthal_sources(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *target_array, *source_array;
        if (!PyArg_ParseTuple(args, "OO", &target_array, &source_array)) return nullptr;
        const Buffer targets(target_array, "Azimuthal targets", 1), sources(source_array, "Azimuthal sources", 1);
        targets.require<double>();
        sources.require<double>();
        if (!sources.shape(0)) value_error("No azimuthal source angles are available for backprojection");
        if (sources.shape(0) > std::numeric_limits<std::int32_t>::max()) {
            value_error("Azimuthal trajectory exceeds int32 source addressing");
        }
        auto owner_array = empty({targets.shape(0)}, "int32");
        auto reverse_array = empty({targets.shape(0)}, "bool");
        const Buffer owners(owner_array.get(), "Azimuthal owners", 1, true);
        const Buffer reverses(reverse_array.get(), "Azimuthal reversals", 1, true);
        {
            GilRelease unlocked;
            for (Size target_index = 0; target_index < targets.shape(0); ++target_index) {
                const double target = targets.load<double>(target_index);
                Size best_index = 0;
                double best_distance = std::numeric_limits<double>::infinity(), best_raw = 0.;
                for (Size source_index = 0; source_index < sources.shape(0); ++source_index) {
                    const double raw = target - sources.load<double>(source_index);
                    const double distance = std::abs(positive_remainder(raw + 90., 180.) - 90.);
                    // NumPy argmin returns the first NaN, even after a finite minimum.
                    if (std::isnan(distance)) {
                        best_index = source_index;
                        best_raw = raw;
                        break;
                    }
                    if (distance < best_distance) {
                        best_index = source_index;
                        best_distance = distance;
                        best_raw = raw;
                    }
                }
                owners.store(static_cast<std::int32_t>(best_index), target_index);
                reverses.store(std::abs(best_raw) > 90., target_index);
            }
        }
        return PyTuple_Pack(2, owner_array.get(), reverse_array.get());
    });
}

PyMethodDef methods[] = {
    {"render_radial_gray8", render_radial_gray8, METH_VARARGS, nullptr},
    {"render_spherical_gray8", render_spherical_gray8, METH_VARARGS, nullptr},
    {"nearest_azimuthal_sources", nearest_azimuthal_sources, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_rendering(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
