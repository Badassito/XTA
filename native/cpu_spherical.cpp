#include "cpu_shell_source.hpp"

#include <algorithm>
#include <cmath>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double roundoff = 8 * std::numeric_limits<double>::epsilon();

struct Projection {
    Index out_t, out_h, out_w, work_t, work_h, work_w;
    Index face, intervals, origin_u, origin_v, native_h, native_w;
    double minimum, maximum;
    Index z, first, stop, rectangle_width, rectangle_x, rectangle_y;
    bool use_boxes, scalar_max;
};

// Cache the radius interval in which the binary search takes its previous
// branch sequence. This works even with repeated or unsorted radii because
// every visited comparison contributes its own open/closed bound.
struct ShellSearch {
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    Index inner = 0, outer = 0;
    double inner_radius = 0., outer_radius = 0.;

    Index nearest(const Buffer& radii, double radius) {
        if (!(lower < radius && radius <= upper)) {
            lower = -std::numeric_limits<double>::infinity();
            upper = std::numeric_limits<double>::infinity();
            Index lo = 0, hi = radii.shape(0);
            while (lo < hi) {
                const auto middle = lo + (hi - lo) / 2;
                const auto candidate = radii.load<double>(middle);
                if (candidate < radius) {
                    lower = std::max(lower, candidate);
                    lo = middle + 1;
                } else {
                    upper = std::min(upper, candidate);
                    hi = middle;
                }
            }
            outer = std::min(lo, radii.shape(0) - 1);
            inner = std::max(Index(0), outer - 1);
            inner_radius = radii.load<double>(inner);
            outer_radius = radii.load<double>(outer);
        }
        return radius - inner_radius <= outer_radius - radius ? inner : outer;
    }
};

Index round_even(double value) {
    const auto lower = static_cast<Index>(std::floor(value));
    const double fraction = value - static_cast<double>(lower);
    return lower + (fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0));
}

std::array<double, 3> face_direction(Index face, double lx, double ly, double lz) {
    switch (face) {
        case 0: return {lx, ly, lz};
        case 1: return {ly, -lx, lz};
        case 2: return {-lx, -ly, lz};
        case 3: return {-ly, lx, lz};
        case 4: return {lz, ly, -lx};
        default: return {-lz, ly, lx};
    }
}

std::array<double, 2> face_coordinates(double normal, double right, double up) {
    int area;
    double major, minor;
    if (std::abs(right) >= std::abs(up)) {
        if (right >= 0.0) {
            area = 0; major = right; minor = up;
        } else {
            area = 2; major = -right; minor = -up;
        }
    } else if (up >= 0.0) {
        area = 1; major = up; minor = -right;
    } else {
        area = 3; major = -up; minor = right;
    }
    const double theta = std::atan2(minor, major);
    double ratio = (12.0 / pi) * (theta - std::asin(std::sin(theta) * (1.0 / std::sqrt(2.0))));
    if (std::abs(minor) == major) {
        ratio = minor > 0.0 ? 1.0 : (minor < 0.0 ? -1.0 : 0.0);
    }
    const double cosine = std::cos(theta);
    const double d = 1.0 - cosine / std::sqrt(1.0 + cosine * cosine);
#ifdef _WIN32
    const double tangent = ::_hypot(right, up);
#else
    const double tangent = ::hypot(right, up);
#endif
    const double p = major == normal ? 1.0 : tangent / std::sqrt((1.0 + normal) * d);
    const double m = p * ratio;
    switch (area) {
        case 0: return {p, m};
        case 1: return {-m, p};
        case 2: return {-p, -m};
        default: return {m, -p};
    }
}

Index source_pixel(Index pixel, Index native_length, Index source_length) {
    if (native_length == source_length) {
        return pixel;
    }
    const auto scaled = (double(pixel) + 0.5) * double(source_length) / double(native_length);
    return scaled >= double(source_length) ? source_length - 1 : Index(scaled);
}

void project(const ShellSource& source, const Buffer& radii, const Buffer& rotation,
             const Buffer& boxes, const Projection& cfg, const Buffer& output) {
    const auto source_h = source.shape(1);
    const auto source_w = source.shape(2);
    const auto dz = ((double(cfg.z) + 0.5) - double(cfg.out_t) / 2.0) * double(cfg.work_t) / double(cfg.out_t);
    const auto width = cfg.rectangle_width ? cfg.rectangle_width : cfg.out_w;
    if (cfg.first == cfg.stop) return;
    double matrix[3][3];
    for (Index axis = 0; axis < 3; ++axis) {
        for (Index component = 0; component < 3; ++component) {
            matrix[axis][component] = rotation.load<double>(axis, component);
        }
    }
    const double dz_rotation[] = {dz * matrix[2][0], dz * matrix[2][1], dz * matrix[2][2]};
    const double dz_squared = dz * dz;
    ShellSearch shell_search;
    Index pixel = cfg.first, x = pixel % width + cfg.rectangle_x, y = pixel / width + cfg.rectangle_y;
    while (pixel < cfg.stop) {
        const Index row_end = pixel + std::min(cfg.stop - pixel, width - (x - cfg.rectangle_x));
        const auto dy = ((double(y) + 0.5) - double(cfg.out_h) / 2.0) * double(cfg.work_h) / double(cfg.out_h);
        const auto least_radius = std::sqrt(dy * dy + dz_squared);
        // Nonnegative additions make least_radius a lower bound in binary64.
        // The extra margin keeps the original upper-shell tolerance intact.
        if (least_radius > cfg.maximum && least_radius - cfg.maximum >
                2. * roundoff * std::max(least_radius, cfg.maximum)) {
            pixel = row_end;
        } else {
            const double dy_rotation[] = {dy * matrix[1][0], dy * matrix[1][1], dy * matrix[1][2]};
            for (; pixel < row_end; ++pixel, ++x) {
                const auto dx = ((double(x) + 0.5) - double(cfg.out_w) / 2.0) * double(cfg.work_w) / double(cfg.out_w);
                const auto radius = std::sqrt((dx * dx + dy * dy) + dz_squared);
                if ((radius < cfg.minimum && cfg.minimum - radius > roundoff * std::max(radius, cfg.minimum))
                    || (radius > cfg.maximum && radius - cfg.maximum > roundoff * std::max(radius, cfg.maximum))) {
                    continue;
                }

                Index shell = -1;
                if (cfg.use_boxes) {
                    shell = shell_search.nearest(radii, radius);
                    if (boxes.load<std::int64_t>(shell, 1) <= boxes.load<std::int64_t>(shell, 0)
                        || boxes.load<std::int64_t>(shell, 3) <= boxes.load<std::int64_t>(shell, 2)) {
                        continue;
                    }
                }
                auto local = [&](int col) {
                    return (dx * matrix[0][col] + dy_rotation[col]) + dz_rotation[col];
                };
                double lx = local(0), ly = local(1), lz = local(2);
                const double scale = std::max({std::abs(lx), std::abs(ly), std::abs(lz)});
                if (scale == 0.0 || !std::isfinite(scale)) {
                    continue;
                }
                lx /= scale;
                ly /= scale;
                lz /= scale;
                auto [normal, right, up] = face_direction(cfg.face, lx, ly, lz);
                if (normal <= 0.0 || normal + roundoff < std::max(std::abs(right), std::abs(up))) {
                    continue;
                }
                const double norm = std::sqrt((lx * lx + ly * ly) + lz * lz);
                auto [u, v] = face_coordinates(normal / norm, right / norm, up / norm);
                u = std::min(1.0, std::max(-1.0, u));
                v = std::min(1.0, std::max(-1.0, v));
                const Index column = round_even((u + 1.0) * double(cfg.intervals) / 2.0) - cfg.origin_u;
                const Index row = round_even((1.0 - v) * double(cfg.intervals) / 2.0) - cfg.origin_v;
                if (row < 0 || row >= cfg.native_h || column < 0 || column >= cfg.native_w) {
                    continue;
                }
                const auto pr = source_pixel(row, cfg.native_h, source_h);
                const auto pc = source_pixel(column, cfg.native_w, source_w);
                if (!cfg.use_boxes) {
                    shell = shell_search.nearest(radii, radius);
                } else if (pr < boxes.load<std::int64_t>(shell, 0) || pr >= boxes.load<std::int64_t>(shell, 1)
                           || pc < boxes.load<std::int64_t>(shell, 2) || pc >= boxes.load<std::int64_t>(shell, 3)) {
                    continue;
                }
                const auto value = source.uint8(shell, pr, pc);
                output.store(cfg.scalar_max ? value : std::uint8_t(value != 0), pixel - cfg.first);
            }
        }
        x = cfg.rectangle_x;
        ++y;
    }
}

void validate(const ShellSource& source, const Buffer& radii, const Buffer& rotation,
              const Buffer& boxes, const Projection& cfg) {
    if (source.dense() && !source.dense()->is<std::uint8_t>() && !source.dense()->is<bool>()) {
        type_error("Spherical projection requires a uint8/bool source");
    }
    if (radii.shape(0) != source.shape(0) || rotation.shape(0) != 3 || rotation.shape(1) != 3
        || boxes.shape(1) != 4 || boxes.shape(0) != (cfg.use_boxes ? radii.shape(0) : 0)) {
        value_error("Spherical projection geometry arrays have inconsistent shapes");
    }
    if (cfg.out_t <= 0 || cfg.out_h <= 0 || cfg.out_w <= 0 || cfg.work_t <= 0 || cfg.work_h <= 0
        || cfg.work_w <= 0 || cfg.native_h <= 0 || cfg.native_w <= 0 || cfg.z < 0 || cfg.z >= cfg.out_t
        || cfg.first < 0 || cfg.stop < cfg.first || cfg.face < 0 || cfg.face > 5 || cfg.intervals <= 0
        || cfg.origin_u <= -cfg.native_w || cfg.origin_v <= -cfg.native_h
        || cfg.origin_u > cfg.intervals || cfg.origin_v > cfg.intervals
        || cfg.rectangle_width < 0 || cfg.rectangle_x < 0
        || cfg.rectangle_y < 0 || cfg.rectangle_y >= cfg.out_h) {
        value_error("Invalid Spherical projection geometry");
    }
    // The final floating lattice coordinate must safely convert to Index.
    if (double(cfg.intervals) >= std::ldexp(1.0, std::numeric_limits<Index>::digits)) {
        overflow_error("Spherical lattice exceeds its native index range");
    }
    const auto lattice_edge = round_even(double(cfg.intervals));
    checked_add(lattice_edge, -cfg.origin_u);
    checked_add(lattice_edge, -cfg.origin_v);
    const auto width = cfg.rectangle_width ? cfg.rectangle_width : cfg.out_w;
    if (cfg.rectangle_x > cfg.out_w || width > cfg.out_w - cfg.rectangle_x
        || cfg.stop > checked_extent(cfg.out_h - cfg.rectangle_y, width)) {
        value_error("Spherical projection strip exceeds its output grid");
    }
}

PyObject* pull_spherical(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        if (PyTuple_GET_SIZE(args) != 26) {
            type_error("spherical_pull requires 26 arguments");
        }
        auto arg = [&](int index) { return PyTuple_GET_ITEM(args, index); };
        auto integer = [&](int index) {
            const auto value = PyLong_AsSsize_t(arg(index));
            if (PyErr_Occurred()) {
                throw PythonError{};
            }
            return value;
        };
        auto number = [&](int index) {
            const auto value = PyFloat_AsDouble(arg(index));
            if (PyErr_Occurred()) {
                throw PythonError{};
            }
            return value;
        };
        auto boolean = [&](int index) {
            const auto value = PyObject_IsTrue(arg(index));
            if (value < 0) {
                throw PythonError{};
            }
            return value == 1;
        };
        const ShellSource source(arg(0));
        const Buffer radii(arg(1), "Spherical radii", 1);
        const Buffer rotation(arg(2), "Spherical rotation", 2);
        const Buffer boxes(arg(3), "Spherical bounds", 2);
        radii.require<double>();
        rotation.require<double>();
        boxes.require<std::int64_t>();
        const Projection cfg{
            integer(5), integer(6), integer(7), integer(8), integer(9), integer(10),
            integer(11), integer(12), integer(13), integer(14), integer(15), integer(16),
            number(17), number(18), integer(19), integer(20), integer(21), integer(22),
            integer(23), integer(24), boolean(4), boolean(25),
        };
        validate(source, radii, rotation, boxes, cfg);
        auto result = zeros({cfg.stop - cfg.first}, "uint8");
        const Buffer output(result.get(), "Spherical output", 1, true);
        {
            GilRelease unlocked;
            project(source, radii, rotation, boxes, cfg, output);
        }
        return result.release();
    });
}

PyMethodDef methods[] = {
    {"spherical_pull", pull_spherical, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_spherical(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
