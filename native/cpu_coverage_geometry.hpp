#ifndef XTA_CPU_COVERAGE_GEOMETRY_HPP
#define XTA_CPU_COVERAGE_GEOMETRY_HPP

#include "cpu_kernels.hpp"
#include <algorithm>
#include <cmath>

namespace xta::cpu::coverage_geometry {

using I = std::int64_t;
using U = std::uint64_t;

inline double axis_value(I destination, I contributor, I working, I output, double scale) {
    return output < working ? static_cast<double>(contributor) : (destination + .5) * scale - .5;
}

inline std::pair<I, I> axis_bounds(I destination, I working, I output) {
    if (output >= working)
        return {0, 1};
    // The input grid is normally well inside 32 bits. Keep the exact integer
    // contraction free of checked-multiply divisions in that common case.
    if (working <= std::numeric_limits<std::int32_t>::max()) {
        return {destination * working / output, ((destination + 1) * working + output - 1) / output};
    }
    const auto first = checked_extent(destination, working);
    const auto stop = checked_extent(destination + 1, working);
    return {first / output, stop / output + (stop % output != 0)};
}

// These wrappers accept the positional calls used by the prepared plan. Shape and
// dtype validation precede the bulk call; all views retain the exporting object.
struct Args {
    PyObject *args;
    Args(PyObject *arguments, Py_ssize_t count) : args(arguments) {
        if (PyTuple_GET_SIZE(args) != count)
            type_error("Incorrect number of native projection arguments");
    }
    PyObject *operator[](int index) const { return PyTuple_GET_ITEM(args, index); }
    I integer(int i) const {
        auto value = PyLong_AsLongLong((*this)[i]);
        if (value == -1 && PyErr_Occurred())
            throw PythonError{};
        return value;
    }
    U unsigned_integer(int i) const {
        Object index(PyNumber_Index((*this)[i]));
        auto value = PyLong_AsUnsignedLongLong(index.get());
        if (PyErr_Occurred())
            throw PythonError{};
        return value;
    }
    double number(int i) const {
        double value = PyFloat_AsDouble((*this)[i]);
        if (PyErr_Occurred())
            throw PythonError{};
        return value;
    }
    bool flag(int i) const {
        int result = PyObject_IsTrue((*this)[i]);
        if (result < 0)
            throw PythonError{};
        return result != 0;
    }
};

inline void require_size(const Buffer &buffer, I length) {
    if (buffer.shape(0) != length)
        value_error("Inconsistent native projection array lengths");
}

struct AngularGrid {
    I first, width;
    Buffer vertical, horizontal;
    bool indexed;
    double work_h, work_w, out_h, out_w, center_y, center_x;
    AngularGrid(const Args &a, int start)
        : first(a.integer(start)), width(a.integer(start + 1)),
          vertical(a[start + 2], "vertical indices", 1), horizontal(a[start + 3], "horizontal indices", 1),
          indexed(a.flag(start + 4)), work_h(a.number(start + 5)), work_w(a.number(start + 6)),
          out_h(a.number(start + 7)), out_w(a.number(start + 8)), center_y(a.number(start + 9)),
          center_x(a.number(start + 10)) {
        vertical.require<I>();
        horizontal.require<I>();
        if (first < 0 || width <= 0 || !(out_h > 0 && out_w > 0))
            value_error("Invalid angular plane dimensions");
        require_size(horizontal, vertical.shape(0));
    }
    void validate_count(I count) const {
        if (indexed)
            require_size(vertical, count);
        else
            checked_add(first, count);
    }
    std::pair<double, double> position(I i) const {
        if (indexed) {
            if (i < 0 || i >= vertical.shape(0))
                throw std::invalid_argument("Angular point index is outside its table");
            return {double(vertical.load<I>(i)), double(horizontal.load<I>(i))};
        }
        const I at = checked_add(first, i);
        return {double(at / width), double(at % width)};
    }
    std::pair<double, double> offsets(I i) const {
        auto [v, h] = position(i);
        return {(2. * v + 1. - out_h) * work_h / (2. * out_h) + ((work_h - 1.) * .5 - center_y),
                (2. * h + 1. - out_w) * work_w / (2. * out_w) + ((work_w - 1.) * .5 - center_x)};
    }
};

inline double roi_bound(double distance, double limit) {
    return limit + 1.7763568394002505e-15 * std::max(distance, std::abs(limit));
}
inline double modulo180(double a) {
    const double mod = std::fmod(a, 180.);
    return mod == 0. ? 0. : mod < 0. ? mod + 180. : mod;
}

struct ColumnError {};
struct AngularKeys {
    double radius;
    I diameter;
    Buffer theta, dy, dx, sorted, order, sources, reversed, cosines, sines, columns;
    I source_width;
    U invalid;
    Buffer output;
    bool small;
    AngularKeys(const Args &a, int start)
        : radius(a.number(start)), diameter(a.integer(start + 1)), theta(a[start + 2], "theta", 1),
          dy(a[start + 3], "dy", 1), dx(a[start + 4], "dx", 1), sorted(a[start + 5], "sorted angles", 1),
          order(a[start + 6], "owner order", 1), sources(a[start + 7], "angle sources", 1),
          reversed(a[start + 8], "angle reversals", 1), cosines(a[start + 9], "cosines", 1),
          sines(a[start + 10], "sines", 1), columns(a[start + 11], "column table", 1),
          source_width(a.integer(start + 12)), invalid(a.unsigned_integer(start + 13)),
          output(a[start + 14], "keys", 1, true), small(output.is<std::uint32_t>()) {
        theta.require<float>();
        dy.require<float>();
        dx.require<float>();
        sorted.require<double>();
        order.require<I>();
        sources.require<I>();
        reversed.require<bool>();
        cosines.require<float>();
        sines.require<float>();
        columns.require<I>();
        if (!small)
            output.require<U>();
        const I n = output.shape(0), angles = sources.shape(0);
        for (auto *b : {&theta, &dy, &dx})
            require_size(*b, n);
        for (auto *b : {&sorted, &order, &reversed, &cosines, &sines})
            require_size(*b, angles);
        if (!angles || diameter < 1 || source_width < 1)
            value_error("Invalid angular owner geometry");
    }
    void store_key(I i, U key) const {
        if (small)
            output.store<std::uint32_t>(static_cast<std::uint32_t>(key), i);
        else
            output.store<U>(key, i);
    }
    U owner(I i) const {
        const double angle = theta.load<float>(i);
        I low = 0, high = sorted.shape(0);
        while (low < high) {
            I middle = low + (high - low) / 2;
            if (sorted.load<double>(middle) < angle)
                low = middle + 1;
            else
                high = middle;
        }
        const I right = low == sorted.shape(0) ? 0 : low, left = right ? right - 1 : sorted.shape(0) - 1;
        const double dl = std::abs(modulo180(angle - sorted.load<double>(left) + 90.) - 90.);
        const double dr = std::abs(modulo180(angle - sorted.load<double>(right) + 90.) - 90.);
        const I ol = order.load<I>(left), oright = order.load<I>(right);
        const I nearest = dl < dr || (dl == dr && ol < oright) ? ol : oright;
        if (nearest < 0 || nearest >= sources.shape(0))
            throw std::invalid_argument("Invalid angular owner");
        float projection = dx.load<float>(i) * cosines.load<float>(nearest) +
                           dy.load<float>(i) * sines.load<float>(nearest);
        if (reversed.load<bool>(nearest))
            projection = -projection;
        I column = 0;
        if (diameter != 1) {
            const float last = float(diameter - 1);
            const float raw = (projection + float(radius)) / float(std::max(1e-6, 2. * radius)) * last;
            const float q = std::min(std::max(std::nearbyint(raw), 0.f), last);
            if (!(q >= 0.f && static_cast<double>(q) < 0x1p63))
                throw ColumnError{};
            column = static_cast<I>(q);
        }
        if (column >= columns.shape(0))
            throw ColumnError{};
        return static_cast<U>(sources.load<I>(nearest)) * static_cast<U>(source_width) +
               static_cast<U>(columns.load<I>(column));
    }
};

template <class Fn> PyObject *angular_protect(Fn fn) {
    return protect([&]() -> PyObject * {
        try {
            return fn();
        } catch (const ColumnError &) {
            PyErr_SetString(PyExc_IndexError, "Angular key column is out of bounds for the column lookup");
            return nullptr;
        }
    });
}

inline PyObject *angular_coordinates(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 13);
        AngularGrid grid(a, 0);
        Buffer dy(a[11], "dy", 1, true), dx(a[12], "dx", 1, true);
        dy.require<float>();
        dx.require<float>();
        I size = dy.shape(0);
        require_size(dx, size);
        grid.validate_count(size);
        const double sy = grid.work_h / grid.out_h, sx = grid.work_w / grid.out_w;
        const float cy = float(grid.center_y), cx = float(grid.center_x);
        {
            GilRelease release;
            for (I i = 0; i < size; ++i) {
                auto [v, h] = grid.position(i);
                dy.store<float>(float((v + .5) * sy - .5) - cy, i);
                dx.store<float>(float((h + .5) * sx - .5) - cx, i);
            }
        }
        Py_RETURN_NONE;
    });
}

inline PyObject *angular_keys(PyObject *, PyObject *args) {
    return angular_protect([&]() -> PyObject * {
        Args a(args, 29);
        AngularGrid grid(a, 0);
        AngularKeys keys(a, 12);
        Buffer ambiguous(a[27], "ambiguous point indices", 1, true);
        ambiguous.require<I>();
        const I start = a.integer(28), size = keys.output.shape(0);
        const double margin = a.number(11), limit = keys.radius + .5;
        if (start < 0 || start > size)
            value_error("Angular key span start is out of bounds");
        grid.validate_count(size);
        I pending = 0, stop = start;
        {
            GilRelease release;
            for (; stop < size; ++stop) {
                auto [dy, dx] = grid.offsets(stop);
                double approx = std::sqrt(dx * dx + dy * dy);
                double lower = approx * (1. - margin), upper = approx * (1. + margin);
                if (approx == 0. || (1e-100 < approx && approx < 1e100)) {
                    if (upper <= roi_bound(lower, limit)) {
                        keys.store_key(stop, keys.owner(stop));
                        continue;
                    }
                    if (lower > roi_bound(upper, limit)) {
                        keys.store_key(stop, keys.invalid);
                        continue;
                    }
                }
                if (pending == ambiguous.shape(0))
                    break;
                ambiguous.store<I>(stop, pending++);
            }
        }
        return Py_BuildValue("(LL)", static_cast<long long>(pending), static_cast<long long>(stop));
    });
}

inline PyObject *ambiguous_offsets(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 14);
        Buffer indices(a[0], "indices", 1);
        indices.require<I>();
        AngularGrid grid(a, 1);
        Buffer dy(a[12], "dy", 1, true), dx(a[13], "dx", 1, true);
        dy.require<double>();
        dx.require<double>();
        require_size(dy, indices.shape(0));
        require_size(dx, indices.shape(0));
        {
            GilRelease release;
            for (I j = 0; j < indices.shape(0); ++j) {
                I i = indices.load<I>(j);
                if (i < 0)
                    throw std::invalid_argument("Negative ambiguous point index");
                auto [v, h] = grid.offsets(i);
                dy.store<double>(v, j);
                dx.store<double>(h, j);
            }
        }
        Py_RETURN_NONE;
    });
}

inline PyObject *ambiguous_keys(PyObject *, PyObject *args) {
    return angular_protect([&]() -> PyObject * {
        Args a(args, 17);
        Buffer indices(a[0], "indices", 1), distances(a[1], "distances", 1);
        indices.require<I>();
        distances.require<double>();
        require_size(distances, indices.shape(0));
        AngularKeys keys(a, 2);
        {
            GilRelease release;
            for (I j = 0; j < indices.shape(0); ++j) {
                I i = indices.load<I>(j);
                if (i < 0 || i >= keys.output.shape(0))
                    throw std::invalid_argument("Ambiguous point index out of bounds");
                double d = distances.load<double>(j);
                keys.store_key(i, d <= roi_bound(d, keys.radius + .5) ? keys.owner(i) : keys.invalid);
            }
        }
        Py_RETURN_NONE;
    });
}

} // namespace xta::cpu::coverage_geometry
#endif
