#ifndef XTA_CPU_COVERAGE_PULL_HPP
#define XTA_CPU_COVERAGE_PULL_HPP

#include "cpu_coverage_geometry.hpp"

#include <tuple>

namespace xta::cpu::coverage {
using coverage_geometry::axis_bounds;
using coverage_geometry::axis_value;
using coverage_geometry::I;
using coverage_geometry::U;

struct Stats {
    I visited = 0, addresses = 0, reads = 0, positive = 0;
};

struct Geometry {
    I base, vertical, wt, wh, ww, ot, oh, ow, nh, nw, plane, total;
    std::array<double, 12> c;
    std::array<I, 6> bbox;

    Geometry(const Buffer &integer, const Buffer &coefficient, const Buffer &box) {
        integer.require<I>();
        coefficient.require<double>();
        box.require<I>();
        if (integer.ndim() != 1 || integer.shape(0) < 10 || coefficient.ndim() != 1 ||
            coefficient.shape(0) < 12 || box.ndim() != 1 || box.shape(0) != 6)
            value_error("Invalid native pull geometry arrays");
        base = integer.load<I>(0);
        vertical = integer.load<I>(1);
        wt = integer.load<I>(2);
        wh = integer.load<I>(3);
        ww = integer.load<I>(4);
        ot = integer.load<I>(5);
        oh = integer.load<I>(6);
        ow = integer.load<I>(7);
        nh = integer.load<I>(8);
        nw = integer.load<I>(9);
        if (base < 0 || base > 2 || std::min({wt, wh, ww, ot, oh, ow, nh, nw}) < 1)
            value_error("Invalid native pull geometry dimensions");
        plane = checked_extent(oh, ow);
        total = checked_extent(ot, plane);
        for (int i = 0; i < 12; ++i) {
            c[i] = coefficient.load<double>(i);
            if (!std::isfinite(c[i]))
                value_error("Nonfinite native pull geometry coefficient");
        }
        const std::array<I, 3> dims{ot, oh, ow};
        for (int i = 0; i < 3; ++i) {
            bbox[i] = box.load<I>(i);
            bbox[i + 3] = box.load<I>(i + 3);
            if (bbox[i] < 0 || bbox[i + 3] < bbox[i] || bbox[i + 3] > dims[i])
                value_error("Native pull crop exceeds its destination grid");
        }
    }
    void check_span(I first, I count) const {
        if (first < 0 || count < 0 || first > total || count > total - first)
            throw std::invalid_argument("Native pull output span exceeds its declared geometry");
    }
};

inline I clamp_floor(double value, I last) {
    if (!(value > 0.))
        return 0;
    if (value >= double(last))
        return last;
    return static_cast<I>(std::floor(value));
}

inline bool frame_independent_of_x(const Geometry &g) noexcept {
    // X is the stack coordinate on the coronal plane. Even with zero slope,
    // keep the original arithmetic for extreme horizontal coordinates where
    // a finite coefficient could generate an infinite delta and 0 * Inf.
    const auto limit = std::numeric_limits<double>::max() / 4.;
    return g.base != 2 && (g.vertical || (g.c[0] == 0. && std::abs(g.c[2]) < limit
            && std::abs(g.c[11]) < limit / double(g.ow)));
}

inline double tilted_frame(const Geometry &g, double stack, double nv, double nu) noexcept {
    return stack - g.c[0] * ((g.vertical ? nv : nu) - g.c[2]) - g.c[1];
}

template <class Read>
inline void tilted(const Geometry &g, I frames, I height, I width, Read &&read, char *output,
                   Py_ssize_t stride, I count, I first, bool scalar_max, Stats &stats) {
    g.check_span(first, count);
    if (std::min({frames, height, width}) <= 0)
        throw std::invalid_argument("Empty tilted pull source");
    const bool fixed_frame = frame_independent_of_x(g);
    I local = 0;
    while (local < count) {
        I at = first + local, z = at / g.plane, y = at / g.ow % g.oh, x0 = at % g.ow;
        I stop = std::min(count - local, g.ow - x0);
        const bool row_in_box = g.bbox[0] <= z && z < g.bbox[3] && g.bbox[1] <= y && y < g.bbox[4];
        I begin = row_in_box ? std::clamp(g.bbox[2] - x0, I(0), stop) : stop;
        I end = row_in_box ? std::clamp(g.bbox[5] - x0, begin, stop) : stop;
        for (I i = 0; i < begin; ++i)
            output[(local + i) * stride] = 0;
        for (I i = end; i < stop; ++i)
            output[(local + i) * stride] = 0;
        stats.visited += end - begin;
        if (begin < end) {
            auto [z0, z1] = axis_bounds(z, g.wt, g.ot);
            auto [y0, y1] = axis_bounds(y, g.wh, g.oh);
            const bool single_frame = fixed_frame && z1 - z0 == 1 && y1 - y0 == 1;
            double row_nv = 0., row_frame = 0.;
            if (single_frame) {
                const double cz = axis_value(z, z0, g.wt, g.ot, g.c[9]);
                const double cy = axis_value(y, y0, g.wh, g.oh, g.c[10]);
                row_nv = g.base == 0 ? cy : cz;
                row_frame = tilted_frame(g, g.base == 0 ? cz : cy, row_nv, 0.);
            }
            for (I i = begin; i < end; ++i) {
                const I x = x0 + i;
                auto [a, b] = axis_bounds(x, g.ww, g.ow);
                std::uint8_t result = 0;
                for (I iz = z0; iz < z1; ++iz) {
                    const double cz = axis_value(z, iz, g.wt, g.ot, g.c[9]);
                    for (I iy = y0; iy < y1; ++iy) {
                        const double cy = axis_value(y, iy, g.wh, g.oh, g.c[10]);
                        const double nv = single_frame ? row_nv : (g.base == 0 ? cy : cz);
                        for (I ix = a; ix < b; ++ix) {
                            const double cx = axis_value(x, ix, g.ww, g.ow, g.c[11]);
                            const double nu = g.base == 2 ? cy : cx;
                            const double stack = g.base == 0 ? cz : (g.base == 1 ? cy : cx);
                            const double frame = single_frame ? row_frame : tilted_frame(g, stack, nv, nu);
                            if (!(frame >= -.5 && frame < double(frames) - .5 && nv >= -.5 &&
                                  nv < double(g.nh) - .5 && nu >= -.5 && nu < double(g.nw) - .5))
                                continue;
                            // A valid frame just below the last half-step can round up on addition.
                            const I f = clamp_floor(frame + .5, frames - 1);
                            const I r = clamp_floor(g.c[6] * nu + g.c[7] * nv + g.c[8] + .5, height - 1);
                            const I c = clamp_floor(g.c[3] * nu + g.c[4] * nv + g.c[5] + .5, width - 1);
                            const std::uint8_t pixel = read(f, r, c);
                            ++stats.addresses;
                            if (scalar_max)
                                result = std::max(result, pixel);
                            else if (pixel)
                                result = 1;
                        }
                    }
                }
                output[(local + i) * stride] = static_cast<char>(result);
            }
        }
        local += stop;
    }
}

inline void dense_tilted(const Geometry &g, const Buffer &source, char *output, Py_ssize_t stride, I count,
                         I first, bool scalar_max, Stats &stats) {
    tilted(
        g, source.shape(0), source.shape(1), source.shape(2),
        [&](I f, I r, I c) { return source.load<std::uint8_t>(f, r, c); }, output, stride, count, first,
        scalar_max, stats);
}

template <class Read>
inline void azimuthal(const Geometry &g, I frames, I source_width, Read &&read, char *output,
                      Py_ssize_t stride, I count, I first, bool scalar_max, const Buffer &keys,
                      I key_offset, bool plane_keys, U invalid, const Buffer &rows, Stats &stats) {
    g.check_span(first, count);
    const bool narrow = keys.is<std::uint32_t>();
    const I plane_size =
        g.base == 0 ? g.plane : (g.base == 1 ? checked_extent(g.ot, g.ow) : checked_extent(g.ot, g.oh));
    if (source_width <= 0 || rows.shape(0) < g.nh || (!narrow && !keys.is<U>()) ||
        (plane_keys ? keys.shape(0) < plane_size
                    : (key_offset < 0 || key_offset > keys.shape(0) || count > keys.shape(0) - key_offset)))
        throw std::invalid_argument("Invalid azimuthal pull lookup geometry");
    const I stack_size = g.base == 0 ? g.wt : (g.base == 1 ? g.wh : g.ww);
    const I output_stack = g.base == 0 ? g.ot : (g.base == 1 ? g.oh : g.ow);
    const bool wide_rows = rows.is<I>();
    if (!wide_rows && !rows.is<std::int32_t>())
        throw std::invalid_argument("Invalid azimuthal row lookup dtype");
    const bool narrow_division = narrow && U(source_width) <= std::numeric_limits<std::uint32_t>::max();
    const bool fixed_frame = frame_independent_of_x(g);
    const auto row_range = [&](I destination, double stack, double nv, double nu) -> std::pair<I, I> {
        const double center = stack - g.c[0] * ((g.vertical ? nv : nu) - g.c[2]);
        const double frame = center - g.c[1];
        if (g.nh > output_stack) {
            const double shift = center - stack - g.c[1];
            const double left = double(checked_extent(destination, g.nh)) / output_stack +
                                shift * g.nh / stack_size;
            const double right = double(checked_extent(destination + 1, g.nh)) / output_stack +
                                 shift * g.nh / stack_size;
            return {clamp_floor(left, g.nh), clamp_floor(std::ceil(right), g.nh)};
        }
        if (frame >= -.5 && frame < double(stack_size) - .5) {
            const double native_row = (frame + .5) * g.nh / stack_size - .5;
            const I lower = clamp_floor(std::nearbyint(native_row), g.nh - 1);
            return {lower, lower + 1};
        }
        return {0, 0};
    };
    I local = 0;
    while (local < count) {
        I at = first + local, z = at / g.plane, y = at / g.ow % g.oh, x0 = at % g.ow;
        I stop = std::min(count - local, g.ow - x0);
        const bool row_in_box = g.bbox[0] <= z && z < g.bbox[3] && g.bbox[1] <= y && y < g.bbox[4];
        I begin = row_in_box ? std::clamp(g.bbox[2] - x0, I(0), stop) : stop;
        I end = row_in_box ? std::clamp(g.bbox[5] - x0, begin, stop) : stop;
        for (I i = 0; i < begin; ++i)
            output[(local + i) * stride] = 0;
        for (I i = end; i < stop; ++i)
            output[(local + i) * stride] = 0;
        stats.visited += end - begin;
        const double cz = (z + .5) * g.c[9] - .5, cy = (y + .5) * g.c[10] - .5;
        const I key_start = plane_keys
            ? (g.base == 2 ? z * g.oh + y : (g.base == 0 ? y : z) * g.ow + x0)
            : key_offset + local;
        const bool fixed_key = plane_keys && g.base == 2;
        const double nv = g.base == 0 ? cy : cz;
        std::pair<I, I> fixed_range;
        bool range_ready = false;
        for (I i = begin; i < end; ++i) {
            const I key_index = key_start + (fixed_key ? 0 : i);
            const U key = narrow ? keys.load<std::uint32_t>(key_index) : keys.load<U>(key_index);
            std::uint8_t result = 0;
            if (key != invalid) {
                I lower, upper;
                if (fixed_frame) {
                    // Preserve lazy validation on rows containing only invalid keys.
                    if (!range_ready) {
                        fixed_range = row_range(g.base == 0 ? z : y, g.base == 0 ? cz : cy, nv, 0.);
                        range_ready = true;
                    }
                    std::tie(lower, upper) = fixed_range;
                } else {
                    const I x = x0 + i;
                    const double cx = (x + .5) * g.c[11] - .5;
                    std::tie(lower, upper) = row_range(g.base == 0 ? z : (g.base == 1 ? y : x),
                        g.base == 0 ? cz : (g.base == 1 ? cy : cx), nv, g.base == 2 ? cy : cx);
                }
                if (lower < upper) {
                    U angle, column;
                    if (narrow_division) {
                        const auto small_key = static_cast<std::uint32_t>(key);
                        const auto small_width = static_cast<std::uint32_t>(source_width);
                        angle = small_key / small_width;
                        column = small_key % small_width;
                    } else {
                        angle = key / U(source_width);
                        column = key % U(source_width);
                    }
                    if (angle >= U(frames))
                        throw std::invalid_argument("Azimuthal source key is out of bounds");
                    for (I r = lower; r < upper; ++r) {
                        I row = wide_rows ? rows.load<I>(r) : rows.load<std::int32_t>(r);
                        const std::uint8_t pixel = read(I(angle), row, I(column));
                        ++stats.addresses;
                        if (scalar_max)
                            result = std::max(result, pixel);
                        else if (pixel)
                            result = 1;
                    }
                }
            }
            output[(local + i) * stride] = static_cast<char>(result);
        }
        local += stop;
    }
}

inline void dense_azimuthal(const Geometry &g, const Buffer &source, char *output, Py_ssize_t stride,
                            I count, I first, bool scalar_max, const Buffer &keys, I key_offset,
                            bool plane_keys, U invalid, const Buffer &rows, Stats &stats) {
    azimuthal(
        g, source.shape(0), source.shape(2),
        [&](I f, I r, I c) {
            if (r < 0 || r >= source.shape(1))
                throw std::invalid_argument("Azimuthal source row is out of bounds");
            return source.load<std::uint8_t>(f, r, c);
        },
        output, stride, count, first, scalar_max, keys, key_offset, plane_keys, invalid, rows, stats);
}

} // namespace xta::cpu::coverage
#endif
