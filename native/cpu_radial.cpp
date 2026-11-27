#include "cpu_kernels.hpp"
#include "cpu_shell_source.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;

class Arguments {
public:
    Arguments(PyObject* args, Index required, Index optional = 0) : args_(args) {
        const auto size = PyTuple_GET_SIZE(args_);
        if (size < required || size > required + optional) {
            type_error("Incorrect number of native Radial kernel arguments");
        }
    }

    PyObject* operator[](Index position) const noexcept {
        return PyTuple_GET_ITEM(args_, position);
    }

    Index index(Index position) const {
        if (position >= PyTuple_GET_SIZE(args_)) {
            return 0;
        }
        Object integer(PyNumber_Index((*this)[position]));
        const auto value = PyLong_AsSsize_t(integer.get());
        if (value == -1 && PyErr_Occurred()) {
            throw PythonError{};
        }
        return value;
    }

    double real(Index position) const {
        const auto value = PyFloat_AsDouble((*this)[position]);
        if (value == -1.0 && PyErr_Occurred()) {
            throw PythonError{};
        }
        return value;
    }

    bool flag(Index position) const {
        if (position >= PyTuple_GET_SIZE(args_)) {
            return false;
        }
        const int value = PyObject_IsTrue((*this)[position]);
        if (value == -1) {
            throw PythonError{};
        }
        return value != 0;
    }

private:
    PyObject* args_;
};

template<class Function>
void dispatch_source(const ShellSource& source, Function&& run) {
    const auto* dense = source.dense();
    if (!dense || dense->is<std::uint8_t>()) {
        run(std::uint8_t{});
    } else if (dense->is<bool>()) {
        run(bool{});
    } else if (dense->is<float>()) {
        run(float{});
    } else if (dense->is<double>()) {
        run(double{});
    } else if (dense->is<std::uint16_t>()) {
        run(std::uint16_t{});
    } else if (dense->is<std::int16_t>()) {
        run(std::int16_t{});
    } else if (dense->is<std::int8_t>()) {
        run(std::int8_t{});
    } else if (dense->is<std::uint32_t>()) {
        run(std::uint32_t{});
    } else if (dense->is<std::int32_t>()) {
        run(std::int32_t{});
    } else if (dense->is<std::uint64_t>()) {
        run(std::uint64_t{});
    } else if (dense->is<std::int64_t>()) {
        run(std::int64_t{});
    } else {
        type_error("Radial source requires a native real-valued numeric or Boolean dtype");
    }
}

class Bboxes {
public:
    Bboxes(PyObject* object, Index depth, bool use)
        : bounds_(object, "Radial bounding boxes", 2), use_(use) {
        bounds_.require<std::int64_t>();
        if (use_ && (bounds_.shape(0) != depth || bounds_.shape(1) != 4)) {
            value_error("Radial bounding boxes do not match their source");
        }
    }

    bool empty(Index z) const noexcept {
        return use_ && (bounds_.load<std::int64_t>(z, 1) <= bounds_.load<std::int64_t>(z, 0)
            || bounds_.load<std::int64_t>(z, 3) <= bounds_.load<std::int64_t>(z, 2));
    }

    bool contains(Index z, Index y, Index x) const noexcept {
        return !use_ || (y >= bounds_.load<std::int64_t>(z, 0)
            && y < bounds_.load<std::int64_t>(z, 1)
            && x >= bounds_.load<std::int64_t>(z, 2)
            && x < bounds_.load<std::int64_t>(z, 3));
    }

private:
    Buffer bounds_;
    bool use_;
};

Index radial_row(double height, Index stack_length, Index origin, Index native_height) {
    const double rounded = std::nearbyint(std::min(std::max(height, 0.0),
        static_cast<double>(stack_length - 1)));
    if (!std::isfinite(rounded) || rounded >= static_cast<double>(PY_SSIZE_T_MAX)) {
        return -1;
    }
    const auto index = static_cast<Index>(rounded);
    if (index < origin || (origin < 0 && index > PY_SSIZE_T_MAX + origin)) {
        return -1;
    }
    const auto row = index - origin;
    return row < native_height ? row : -1;
}

template<class T>
std::uint8_t rank_byte(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        // Keep the factored path's legacy x86 scalar conversion: truncate to
        // a signed 32-bit index, then take its low byte. Invalid conversion
        // produced the signed minimum sentinel, whose low byte is zero.
        const double truncated = std::trunc(static_cast<double>(value));
        if (!std::isfinite(truncated) || truncated < std::numeric_limits<std::int32_t>::min()
            || truncated > std::numeric_limits<std::int32_t>::max()) {
            return 0;
        }
        return static_cast<std::uint8_t>(static_cast<std::int32_t>(truncated));
    } else {
        return static_cast<std::uint8_t>(value);
    }
}

class FactoredProjection {
public:
    FactoredProjection(const Arguments& a)
        : source(a[0]), out_h(a.index(15)), out_w(a.index(16)), count(a.index(18)),
          shells(a[1], "Radial shells", 1), offsets(a[2], "Radial offsets", 1),
          columns(a[3], "Radial columns", 1), sampled(a[4], "Radial sample shear", 2),
          rows(a[5], "Radial row map", 1), mapped_columns(a[6], "Radial column map", 1),
          centers(a[7], "Radial stack centers", 1), ideal(a[8], "Radial ideal shear", 1),
          stack_length(a.index(9)), height_origin(a.index(10)), native_height(a.index(11)),
          base(a.index(12)), vertical(a.flag(13)), plane_width(a.index(14)),
          first(a.index(17)),
          boxes(a[19], source.shape(0), a.flag(20)), scalar_max(a.flag(21)) {
        shells.require<std::int32_t>();
        offsets.require<std::uint32_t>();
        columns.require<std::int32_t>();
        sampled.require<double>();
        rows.require<std::int32_t>();
        mapped_columns.require<std::int32_t>();
        centers.require<double>();
        ideal.require<double>();
        if (base < 0 || base > 2 || out_h <= 0 || out_w <= 0 || stack_length <= 0
            || native_height <= 0 || rows.shape(0) != native_height || first < 0 || count < 0
            || sampled.shape(0) != source.shape(0) || sampled.shape(1) != mapped_columns.shape(0)
            || offsets.shape(0) != checked_add(shells.shape(0), 1)
            || plane_width != (base == 2 ? out_h : out_w)) {
            value_error("Invalid factored Radial projection dimensions");
        }
        const auto stop = checked_add(first, count);
        const auto stack_stop = base == 0 ? stop : (base == 1 ? out_h : out_w);
        const auto plane_height = base == 0 ? out_h : stop;
        if (centers.shape(0) < stack_stop || shells.shape(0) < checked_extent(plane_height, plane_width)
            || ideal.shape(0) < (vertical ? plane_height : plane_width)) {
            value_error("Radial projection block exceeds its factored plan");
        }
    }

    template<class T>
    void project(const Buffer& result) const {
        for (Index dz = 0; dz < count; ++dz) {
            const Index z = first + dz;
            for (Index y = 0; y < out_h; ++y) {
                for (Index x = 0; x < out_w; ++x) {
                    const Index v = base == 0 ? y : z;
                    const Index u = base == 2 ? y : x;
                    const Index stack_index = base == 0 ? z : (base == 1 ? y : x);
                    const double stack = centers.load<double>(stack_index);
                    const double ideal_shear = ideal.load<double>(vertical ? v : u);
                    result.store(gather<T>(v * plane_width + u, stack, ideal_shear), dz, y, x);
                }
            }
        }
    }

    const ShellSource source;
    Index out_h, out_w, count;

private:
    template<class T>
    std::uint8_t gather(Index pixel, double stack, double ideal_shear) const {
        const Index shell = shells.load<std::int32_t>(pixel);
        if (shell < 0) {
            return 0;
        }
        if (shell >= source.shape(0)) {
            throw std::invalid_argument("Radial plan shell is outside the source");
        }
        const double ideal_height = stack - ideal_shear;
        if (boxes.empty(shell) || ideal_height < -0.5 || ideal_height >= stack_length - 0.5) {
            return 0;
        }
        const Index first_at = offsets.load<std::uint32_t>(pixel);
        const Index stop_at = offsets.load<std::uint32_t>(pixel + 1);
        if (first_at > stop_at || stop_at > columns.shape(0)) {
            throw std::invalid_argument("Radial plan offsets are outside the column array");
        }
        std::uint8_t best = 0;
        for (Index at = first_at; at < stop_at; ++at) {
            const Index column = columns.load<std::int32_t>(at);
            if (column < 0 || column >= sampled.shape(1)) {
                throw std::invalid_argument("Radial plan column is outside the sample table");
            }
            const Index row = radial_row(stack - sampled.load<double>(shell, column),
                stack_length, height_origin, native_height);
            if (row < 0) {
                continue;
            }
            const Index pr = rows.load<std::int32_t>(row);
            const Index pc = mapped_columns.load<std::int32_t>(column);
            if (pr < 0 || pr >= source.shape(1) || pc < 0 || pc >= source.shape(2)) {
                throw std::invalid_argument("Radial processing index is outside the source");
            }
            if (!boxes.contains(shell, pr, pc)) {
                continue;
            }
            const auto value = source.value<T>(shell, pr, pc);
            if (value != T{}) {
                if (!scalar_max) {
                    return 1;
                }
                best = std::max(best, rank_byte(value));
            }
        }
        return best;
    }

    // Declarations follow initialization dependencies; dimensions are kept near the data they size.
    Buffer shells, offsets, columns, sampled, rows, mapped_columns, centers, ideal;
    Index stack_length, height_origin, native_height, base;
    bool vertical;
    Index plane_width, first;
    Bboxes boxes;
    bool scalar_max;
};

PyObject* project_radial_block(PyObject*, PyObject* args) {
    return protect([&] {
        const FactoredProjection projection(Arguments(args, 21, 1));
        Object output = empty({projection.count, projection.out_h, projection.out_w}, "uint8");
        const Buffer result(output.get(), "Radial output block", 3, true);
        dispatch_source(projection.source, [&](auto type) {
            GilRelease release;
            projection.project<decltype(type)>(result);
        });
        return output.release();
    });
}

double centered_fraction(Index index, Index working, Index output) noexcept {
    return ((2.0 * index + 1.0 - output) * working) / (2.0 * output);
}

double physical_radius(double x, double y) noexcept {
#ifdef _WIN32
    // The legacy Windows float64 lowering calls the CRT _hypot symbol.
    return ::_hypot(x, y);
#else
    return std::hypot(x, y);
#endif
}

double positive_remainder(double value, double period) noexcept {
    const double remainder = std::fmod(value, period);
    return remainder < 0.0 ? remainder + period : remainder;
}

class PlanFreeProjection {
public:
    PlanFreeProjection(const Arguments& a)
        : source(a[1]), result(a[0], "Radial output strip", 1, true),
          radii(a[2], "Radial radii", 1), z(a.index(3)), first(a.index(4)),
          out_t(a.index(5)), out_h(a.index(6)), out_w(a.index(7)),
          work_t(a.index(8)), work_h(a.index(9)), work_w(a.index(10)), base(a.index(11)),
          center_x(a.real(12)), center_y(a.real(13)), min_radius(a.real(14)), max_radius(a.real(15)),
          shell_start(a.index(16)), arc_origin(a.real(17)),
          native_width(a.index(18)), native_height(a.index(19)), height_origin(a.index(20)),
          tilted(a.flag(21)), tangent(a.real(22)), vertical(a.flag(23)),
          boxes(a[24], source.shape(0), a.flag(25)), scalar_max(a.flag(26)),
          rectangle_width(a.index(27)), rectangle_x(a.index(28)), rectangle_y(a.index(29)) {
        radii.require<double>();
        if (base < 0 || base > 2 || std::min({out_t, out_h, out_w, work_t, work_h, work_w,
            native_width, native_height, source.shape(0), source.shape(1), source.shape(2)}) <= 0
            || z < 0 || z >= out_t || first < 0 || radii.shape(0) == 0 || shell_start < 0
            || !std::isfinite(center_x) || !std::isfinite(center_y)
            || !std::isfinite(arc_origin) || (tilted && !std::isfinite(tangent))) {
            value_error("Invalid plan-free Radial projection geometry");
        }
        Index height = out_h, width = out_w;
        if (rectangle_width != 0) {
            if (rectangle_width < 0 || rectangle_x < 0 || rectangle_y < 0 || rectangle_y >= out_h
                || rectangle_width > out_w || rectangle_x > out_w - rectangle_width) {
                value_error("Radial rectangle is outside the output plane");
            }
            height -= rectangle_y;
            width = rectangle_width;
        }
        const Index available = checked_extent(height, width);
        if (first > available || result.shape(0) > available - first) {
            value_error("Radial output strip is outside the destination bounds");
        }
    }

    template<class T>
    void project() const {
        if (scalar_max) {
            result.require<T>();
        } else {
            result.require<std::uint8_t>();
        }
        GilRelease release;
        constexpr double two_pi = 6.283185307179586476925286766559;
        constexpr double bound_eps = 8.0 * std::numeric_limits<double>::epsilon();
        const Index stack_length = base == 0 ? work_t : (base == 1 ? work_h : work_w);
        const double t_fraction = centered_fraction(z, work_t, out_t);
        const double t_center = (work_t - 1.0) * 0.5;
        const double h_center = (work_h - 1.0) * 0.5;
        const double w_center = (work_w - 1.0) * 0.5;
        const double wt = t_fraction + t_center;
        for (Index offset = 0; offset < result.shape(0); ++offset) {
            const Index pixel = first + offset;
            const Index y = rectangle_width ? pixel / rectangle_width + rectangle_y : pixel / out_w;
            const Index x = rectangle_width ? pixel % rectangle_width + rectangle_x : pixel % out_w;
            const double y_fraction = centered_fraction(y, work_h, out_h);
            const double x_fraction = centered_fraction(x, work_w, out_w);
            const double stack = base == 0 ? wt : (base == 1 ? y_fraction + h_center : x_fraction + w_center);
            const double dx = base == 2 ? y_fraction + (h_center - center_x)
                                       : x_fraction + (w_center - center_x);
            const double dy = base == 0 ? y_fraction + (h_center - center_y)
                                       : t_fraction + (t_center - center_y);
            const double radius = physical_radius(dx, dy);
            const double lower_tolerance = bound_eps * std::max(std::abs(radius), std::abs(min_radius));
            const double upper_tolerance = bound_eps * std::max(std::abs(radius), std::abs(max_radius));
            if (radius < min_radius - lower_tolerance || radius > max_radius + upper_tolerance) {
                continue;
            }
            const double ideal_height = tilted ? stack - tangent * (vertical ? dy : dx) : stack;
            if (ideal_height < -0.5 || ideal_height >= stack_length - 0.5) {
                continue;
            }
            const Index global_shell = nearest_shell(radius);
            if (global_shell < shell_start) {
                continue;
            }
            const Index shell = global_shell - shell_start;
            if (shell >= source.shape(0) || boxes.empty(shell)) {
                continue;
            }
            const double selected_radius = radii.load<double>(global_shell);
            const double period = two_pi * selected_radius;
            double column_float = 0.0;
            if (period > 1.0) {
                const double theta = positive_remainder(std::atan2(dy, dx), two_pi);
                column_float = positive_remainder(theta * selected_radius - arc_origin + 0.5, period) - 0.5;
            }
            if (tilted && selected_radius == 0.0) {
                continue;
            }
            for (Index candidate = 0; candidate < native_width; ++candidate) {
                Index column = candidate;
                if (period > 1.0) {
                    const double rounded = std::nearbyint(column_float);
                    if (!std::isfinite(rounded) || rounded >= static_cast<double>(native_width)) {
                        break;
                    }
                    column = static_cast<Index>(rounded);
                    column_float += period;
                }
                if (column < 0 || column >= native_width) {
                    continue;
                }
                double height = stack;
                if (tilted) {
                    const double angle = positive_remainder((arc_origin + column) / selected_radius, two_pi);
                    const double sample_axis = selected_radius * (vertical ? std::sin(angle) : std::cos(angle));
                    height -= tangent * sample_axis;
                }
                const Index row = radial_row(height, stack_length, height_origin, native_height);
                if (row < 0) {
                    continue;
                }
                const auto pr = processing_index(row, source.shape(1), native_height);
                const auto pc = processing_index(column, source.shape(2), native_width);
                if (!boxes.contains(shell, pr, pc)) {
                    continue;
                }
                const auto value = source.value<T>(shell, pr, pc);
                if (scalar_max) {
                    if (value > result.load<T>(offset)) {
                        result.store(value, offset);
                    }
                } else if (value != T{}) {
                    result.store(std::uint8_t{1}, offset);
                    break;
                }
            }
        }
    }

    const ShellSource source;

private:
    Index nearest_shell(double radius) const noexcept {
        Index low = 0, high = radii.shape(0);
        while (low < high) {
            const Index middle = low + (high - low) / 2;
            if (radii.load<double>(middle) < radius) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        const Index right = std::min(low, radii.shape(0) - 1);
        const Index left = std::max(Index{0}, right - 1);
        // An exactly tied radius belongs to the inner shell.
        return radius - radii.load<double>(left) <= radii.load<double>(right) - radius ? left : right;
    }

    static Index processing_index(Index index, Index processing, Index native) noexcept {
        const double projected = (index + 0.5) * processing / native;
        return projected >= static_cast<double>(processing) ? processing - 1 : static_cast<Index>(projected);
    }

    const Buffer result, radii;
    Index z, first, out_t, out_h, out_w, work_t, work_h, work_w, base;
    double center_x, center_y, min_radius, max_radius;
    Index shell_start;
    double arc_origin;
    Index native_width, native_height, height_origin;
    bool tilted;
    double tangent;
    bool vertical;
    Bboxes boxes;
    bool scalar_max;
    Index rectangle_width, rectangle_x, rectangle_y;
};

PyObject* pull_radial_range_into(PyObject*, PyObject* args) {
    return protect([&] {
        const PlanFreeProjection projection(Arguments(args, 27, 3));
        dispatch_source(projection.source, [&](auto type) { projection.project<decltype(type)>(); });
        Py_RETURN_NONE;
    });
}

PyObject* bucket_shell_pixels(PyObject*, PyObject* args) {
    return protect([&] {
        const Arguments a(args, 2);
        const Buffer shells(a[0], "Radial shells", 1);
        shells.require<std::int32_t>();
        const Index count = a.index(1);
        if (count < 0 || shells.shape(0) > std::numeric_limits<std::int32_t>::max()) {
            value_error("Invalid Radial shell bucket dimensions");
        }
        Object offsets_object = zeros({checked_add(count, 1)}, "int64");
        const Buffer offsets(offsets_object.get(), "Radial shell bucket offsets", 1, true);
        std::vector<std::int64_t> cursors(static_cast<std::size_t>(count));
        std::int64_t total = 0;
        {
            GilRelease release;
            for (Index pixel = 0; pixel < shells.shape(0); ++pixel) {
                const Index shell = shells.load<std::int32_t>(pixel);
                if (shell >= count) {
                    throw std::invalid_argument("Radial shell bucket index is out of range");
                }
                if (shell >= 0) {
                    ++cursors[shell];
                }
            }
            for (Index shell = 0; shell < count; ++shell) {
                const auto size = cursors[shell];
                cursors[shell] = total;
                total += size;
                offsets.store(total, shell + 1);
            }
        }
        Object pixels_object = empty({static_cast<Index>(total)}, "int32");
        const Buffer pixels(pixels_object.get(), "Radial shell bucket pixels", 1, true);
        {
            GilRelease release;
            for (Index pixel = 0; pixel < shells.shape(0); ++pixel) {
                const Index shell = shells.load<std::int32_t>(pixel);
                if (shell >= 0) {
                    pixels.store(static_cast<std::int32_t>(pixel), cursors[shell]++);
                }
            }
        }
        return PyTuple_Pack(2, offsets_object.get(), pixels_object.get());
    });
}

PyObject* pack_radial_source_block(PyObject*, PyObject* args) {
    return protect([&] {
        const Arguments a(args, 5);
        const Buffer source(a[0], "Radial pack source", 3);
        const Buffer boxes(a[1], "Radial pack boxes", 2);
        const Buffer offsets(a[2], "Radial pack offsets", 1);
        const Buffer destination(a[4], "Radial pack destination", 1, true);
        source.require<std::uint8_t>();
        boxes.require<std::int64_t>();
        offsets.require<std::uint64_t>();
        destination.require<std::uint8_t>();
        const Index first = a.index(3);
        if (boxes.shape(0) != source.shape(0) || boxes.shape(1) != 4
            || offsets.shape(0) != checked_add(source.shape(0), 1) || first < 0
            || offsets.load<std::uint64_t>(0) != 0) {
            value_error("Invalid Radial pack bounds or offsets");
        }
        const auto total = offsets.load<std::uint64_t>(source.shape(0));
        if (static_cast<std::uint64_t>(first) > total
            || static_cast<std::uint64_t>(destination.shape(0)) > total - first) {
            value_error("Radial pack interval lies outside the concatenated rectangles");
        }
        {
            GilRelease release;
            auto position = static_cast<std::uint64_t>(first);
            Index low = 0, high = source.shape(0);
            while (low < high) {
                const Index middle = low + (high - low) / 2;
                if (offsets.load<std::uint64_t>(middle + 1) <= position) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            for (Index cursor = 0, shell = low; cursor < destination.shape(0);) {
                while (shell < source.shape(0) && offsets.load<std::uint64_t>(shell + 1) <= position) {
                    ++shell;
                }
                if (shell == source.shape(0)) {
                    throw std::invalid_argument("Radial pack interval exhausts the source");
                }
                const Index y0 = static_cast<Index>(boxes.load<std::int64_t>(shell, 0));
                const Index y1 = static_cast<Index>(boxes.load<std::int64_t>(shell, 1));
                const Index x0 = static_cast<Index>(boxes.load<std::int64_t>(shell, 2));
                const Index x1 = static_cast<Index>(boxes.load<std::int64_t>(shell, 3));
                const auto start = offsets.load<std::uint64_t>(shell);
                const auto stop = offsets.load<std::uint64_t>(shell + 1);
                if (y0 < 0 || y1 <= y0 || y1 > source.shape(1)
                    || x0 < 0 || x1 <= x0 || x1 > source.shape(2) || start > position
                    || stop - start != static_cast<std::uint64_t>(checked_extent(y1 - y0, x1 - x0))) {
                    throw std::invalid_argument("Radial pack rectangle does not match its offsets");
                }
                const Index width = x1 - x0;
                const auto within = static_cast<Index>(position - start);
                const Index row = y0 + within / width;
                const Index column = x0 + within % width;
                const Index length = std::min(destination.shape(0) - cursor, x1 - column);
                const auto* input = source.ptr(shell, row, column);
                auto* output = destination.ptr(cursor);
                if (source.stride(2) == 1 && destination.stride(0) == 1) {
                    const auto input_address = reinterpret_cast<std::uintptr_t>(input);
                    const auto output_address = reinterpret_cast<std::uintptr_t>(output);
                    if (input_address >= output_address || output_address - input_address >= static_cast<std::uintptr_t>(length)) {
                        std::memmove(output, input, static_cast<std::size_t>(length));
                    } else {
                        // Preserve forward-copy semantics for overlapping direct callers.
                        for (Index i = 0; i < length; ++i) {
                            output[i] = input[i];
                        }
                    }
                } else {
                    for (Index i = 0; i < length; ++i) {
                        destination.store(source.load<std::uint8_t>(shell, row, column + i), cursor + i);
                    }
                }
                cursor += length;
                position += static_cast<std::uint64_t>(length);
            }
        }
        Py_RETURN_NONE;
    });
}

PyMethodDef methods[] = {
    {"project_radial_block", project_radial_block, METH_VARARGS, nullptr},
    {"pull_radial_range_into", pull_radial_range_into, METH_VARARGS, nullptr},
    {"bucket_shell_pixels", bucket_shell_pixels, METH_VARARGS, nullptr},
    {"pack_radial_source_block", pack_radial_source_block, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_radial(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}
} // namespace xta::cpu
