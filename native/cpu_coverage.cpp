#include "cpu_cvol_source.hpp"

namespace xta::cpu {
namespace {
using coverage::CvolSource;
using coverage::Geometry;
using coverage::Stats;
using coverage_geometry::Args;
using coverage_geometry::I;
using coverage_geometry::U;

PyObject *pull_tilted(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 7);
        Buffer source(a[0], "source", 3), out(a[1], "output", 1, true), ip(a[3], "integer geometry", 1),
            fp(a[4], "coefficients", 1), box(a[5], "bounds", 1);
        source.require<std::uint8_t>();
        out.require<std::uint8_t>();
        Geometry geometry(ip, fp, box);
        Stats stats;
        I first = a.integer(2);
        bool scalar_max = a.flag(6);
        {
            GilRelease release;
            dense_tilted(geometry, source, out.data(), out.stride(0), out.shape(0), first, scalar_max,
                         stats);
        }
        return Py_BuildValue("(LL)", (long long)stats.visited, (long long)stats.addresses);
    });
}

PyObject *pull_azimuthal(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 11);
        Buffer source(a[0], "source", 3), out(a[1], "output", 1, true), ip(a[3], "integer geometry", 1),
            fp(a[4], "coefficients", 1), box(a[5], "bounds", 1), keys(a[7], "keys", 1),
            rows(a[10], "rows", 1);
        source.require<std::uint8_t>();
        out.require<std::uint8_t>();
        Geometry geometry(ip, fp, box);
        Stats stats;
        I first = a.integer(2);
        U invalid = a.unsigned_integer(9);
        bool scalar_max = a.flag(6), cached = a.flag(8);
        {
            GilRelease release;
            dense_azimuthal(geometry, source, out.data(), out.stride(0), out.shape(0), first, scalar_max,
                            keys, 0, cached, invalid, rows, stats);
        }
        return Py_BuildValue("(LL)", (long long)stats.visited, (long long)stats.addresses);
    });
}

PyObject *pull_azimuthal_encoded(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 13);
        Buffer bytes(a[0], "encoded", 1), index(a[1], "records", 2), out(a[4], "output", 1, true),
            ip(a[6], "integer geometry", 1), fp(a[7], "coefficients", 1), box(a[8], "bounds", 1),
            keys(a[9], "keys", 1), rows(a[12], "rows", 1);
        out.require<std::uint8_t>();
        CvolSource source(bytes, index, a.flag(2));
        Geometry geometry(ip, fp, box);
        Stats stats;
        I width = a.integer(3), first = a.integer(5);
        U invalid = a.unsigned_integer(11);
        bool cached = a.flag(10);
        {
            GilRelease release;
            coverage::azimuthal(
                geometry, index.shape(0), width, [&](I f, I r, I c) { return source.read(f, r, c, stats); },
                out.data(), out.stride(0), out.shape(0), first, false, keys, 0, cached, invalid, rows,
                stats);
        }
        return Py_BuildValue("(LLLL)", (long long)stats.visited, (long long)stats.addresses,
                             (long long)stats.reads, (long long)stats.positive);
    });
}

PyObject *pull_tilted_encoded(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 9);
        Buffer bytes(a[0], "encoded", 1), index(a[1], "records", 2), shape(a[3], "source shape", 1),
            out(a[4], "output", 1, true), ip(a[6], "integer geometry", 1), fp(a[7], "coefficients", 1),
            box(a[8], "bounds", 1);
        out.require<std::uint8_t>();
        shape.require<I>();
        coverage_geometry::require_size(shape, 3);
        CvolSource source(bytes, index, a.flag(2));
        Geometry geometry(ip, fp, box);
        Stats stats;
        I first = a.integer(5), frames = shape.load<I>(0), height = shape.load<I>(1),
          width = shape.load<I>(2);
        coverage_geometry::require_size(index, frames);
        {
            GilRelease release;
            coverage::tilted(
                geometry, frames, height, width, [&](I f, I r, I c) { return source.read(f, r, c, stats); },
                out.data(), out.stride(0), out.shape(0), first, false, stats);
        }
        return Py_BuildValue("(LL)", (long long)stats.addresses, (long long)stats.reads);
    });
}

inline int popcount_byte(std::uint8_t byte) {
    unsigned value = byte;
    value -= (value >> 1) & 0x55;
    value = (value & 0x33) + ((value >> 2) & 0x33);
    return (value + (value >> 4)) & 0x0f;
}

PyObject *encoded_input_totals(PyObject *, PyObject *args) {
    return protect([&]() -> PyObject * {
        Args a(args, 3);
        Buffer bytes(a[0], "encoded", 1), index(a[1], "records", 2);
        CvolSource source(bytes, index, a.flag(2));
        I total = 0, foreground = 0;
        {
            GilRelease release;
            for (I i = 0; i < index.shape(0); ++i) {
                if (!source.field<std::uint8_t>(i, 0))
                    continue;
                U offset = source.field<U>(i, 8), count = source.field<U>(i, 16),
                  length = U(bytes.shape(0));
                if (offset > length || count > length - offset)
                    throw std::invalid_argument("Invalid CVOL payload extent");
                total = checked_add(total, static_cast<I>(count));
                if (source.packed) {
                    U left = source.field<std::uint32_t>(i, 28), right = source.field<std::uint32_t>(i, 36);
                    if (right <= left)
                        throw std::invalid_argument("Invalid packed CVOL row width");
                    U width = right - left, stride = (width + 7) / 8;
                    if (count % stride)
                        throw std::invalid_argument("Truncated packed CVOL row");
                    const unsigned mask = (1u << ((width - 1) % 8 + 1)) - 1;
                    for (U row = offset; row < offset + count; row += stride) {
                        for (U j = 0; j + 1 < stride; ++j)
                            foreground += popcount_byte(bytes.load<std::uint8_t>(row + j));
                        foreground += popcount_byte(bytes.load<std::uint8_t>(row + stride - 1) & mask);
                    }
                } else {
                    for (U j = offset; j < offset + count; ++j)
                        foreground += bytes.load<std::uint8_t>(j) != 0;
                }
            }
        }
        return Py_BuildValue("(LL)", (long long)total, (long long)foreground);
    });
}

PyMethodDef methods[] = {
    {"_angular_coordinates", coverage_geometry::angular_coordinates, METH_VARARGS, nullptr},
    {"_angular_keys_kernel", coverage_geometry::angular_keys, METH_VARARGS, nullptr},
    {"_ambiguous_offsets", coverage_geometry::ambiguous_offsets, METH_VARARGS, nullptr},
    {"_ambiguous_keys", coverage_geometry::ambiguous_keys, METH_VARARGS, nullptr},
    {"_pull_tilted", pull_tilted, METH_VARARGS, nullptr},
    {"_pull_azimuthal", pull_azimuthal, METH_VARARGS, nullptr},
    {"_pull_azimuthal_encoded", pull_azimuthal_encoded, METH_VARARGS, nullptr},
    {"_pull_tilted_encoded", pull_tilted_encoded, METH_VARARGS, nullptr},
    {"_encoded_input_totals", encoded_input_totals, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
} // namespace
int register_coverage(PyObject *module) { return PyModule_AddFunctions(module, methods); }
} // namespace xta::cpu
