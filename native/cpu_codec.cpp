#include "cpu_kernels.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

namespace xta::cpu {
namespace {

using Byte = std::uint8_t;
using Index = Py_ssize_t;
using Compress = int (*)(Byte*, unsigned long*, const Byte*, unsigned long, int);
using Inflate = int (*)(Byte*, unsigned long*, const Byte*, unsigned long*);

constexpr int invalid_bounds = -100;
constexpr int invalid_length = -101;
constexpr Index record_bytes = 28;

template<class Function>
Function callback(PyObject* address) {
    void* pointer = PyLong_AsVoidPtr(address);
    if (PyErr_Occurred()) throw PythonError{};
    if (!pointer) value_error("Missing native zlib entry point");
    return reinterpret_cast<Function>(pointer);
}

PyObject* encode(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *address, *mask_obj, *score_obj, *bounds_obj;
        Index z, cell_bytes, capacity, block;
        if (!PyArg_ParseTuple(args, "OOOnOnnn:encode_confidence_cells",
                              &address, &mask_obj, &score_obj, &z, &bounds_obj,
                              &cell_bytes, &capacity, &block)) return nullptr;
        const auto compress = callback<Compress>(address);
        Buffer mask(mask_obj, "mask", 3), scores(score_obj, "scores", 3), bounds(bounds_obj, "bounds", 2);
        scores.require<Byte>();
        bounds.require<std::int64_t>();
        if ((!mask.is<bool>() && !mask.is<Byte>()) || z < 0 || z >= mask.shape(0)
                || mask.shape(0) != scores.shape(0) || mask.shape(1) != scores.shape(1)
                || mask.shape(2) != scores.shape(2) || bounds.shape(1) != 5
                || bounds.shape(0) > 16 || block < 1 || block > 128 || cell_bytes < 0
                || cell_bytes > block * block || capacity <= 0
                || static_cast<std::uint64_t>(capacity) > ULONG_MAX) {
            value_error("Invalid native confidence cell buffers or workspace");
        }
        const Index cells = bounds.shape(0);
        auto encoded_array = empty({capacity}, "uint8");
        auto offsets_array = empty({cells + 1}, "int64");
        Buffer encoded(encoded_array.get(), "output", 1, true), offsets(offsets_array.get(), "offsets", 1, true);
        std::vector<Byte> scratch(static_cast<std::size_t>(cell_bytes));
        Index used = 0, completed = 0;
        int status = 0;
        {
            GilRelease unlock;
            offsets.store<std::int64_t>(0, 0);
            for (Index i = 0; i < cells; ++i) {
                const auto y0 = bounds.load<std::int64_t>(i, 0), y1 = bounds.load<std::int64_t>(i, 1);
                const auto x0 = bounds.load<std::int64_t>(i, 2), x1 = bounds.load<std::int64_t>(i, 3);
                if (y0 < 0 || y0 >= y1 || y1 > mask.shape(1) || x0 < 0 || x0 >= x1
                        || x1 > mask.shape(2) || y1 - y0 > block || x1 - x0 > block
                        || (y1 - y0) * (x1 - x0) > cell_bytes) {
                    throw std::invalid_argument("Native confidence crop exceeds its bounded cell");
                }
                Index at = 0;
                for (Index y = y0; y < y1; ++y) {
                    for (Index x = x0; x < x1; ++x) {
                        scratch[at++] = mask.load<Byte>(z, y, x) ? scores.load<Byte>(z, y, x) : 0;
                    }
                }
                unsigned long written = static_cast<unsigned long>(capacity - used);
                status = compress(reinterpret_cast<Byte*>(encoded.ptr(used)), &written,
                                  scratch.data(), static_cast<unsigned long>(at), 3);
                if (status != 0) break;
                if (written > static_cast<unsigned long>(capacity - used))
                    throw std::runtime_error("Native zlib exceeded its declared output capacity");
                used += written;
                offsets.store<std::int64_t>(used, i + 1);
                ++completed;
            }
        }
        return Py_BuildValue("OOni", encoded_array.get(), offsets_array.get(), completed, status);
    });
}

// The 28-byte on-disk records use little-endian fields and no trailing
// padding. The retained record view may be unaligned and strided.
template<class T>
T little_endian(const Byte* bytes) noexcept {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
        value |= static_cast<T>(bytes[i]) << (i * CHAR_BIT);
    return value;
}

struct Record {
    std::uint32_t z, y, x;
    std::uint16_t h, w;
    std::uint64_t offset;
    std::uint32_t length;

    explicit Record(const char* data) {
        const auto* p = reinterpret_cast<const Byte*>(data);
        z = little_endian<std::uint32_t>(p);
        y = little_endian<std::uint32_t>(p + 4);
        x = little_endian<std::uint32_t>(p + 8);
        h = little_endian<std::uint16_t>(p + 12);
        w = little_endian<std::uint16_t>(p + 14);
        offset = little_endian<std::uint64_t>(p + 16);
        length = little_endian<std::uint32_t>(p + 24);
    }

    std::uint64_t area() const noexcept { return static_cast<std::uint64_t>(h) * w; }

    bool fits(const Buffer& output) const noexcept {
        return z < static_cast<std::uint64_t>(output.shape(0))
               && static_cast<std::uint64_t>(y) + h <= static_cast<std::uint64_t>(output.shape(1))
               && static_cast<std::uint64_t>(x) + w <= static_cast<std::uint64_t>(output.shape(2));
    }
};

struct PayloadBase {
    std::uint64_t origin = 0, prefix = 0;

    explicit PayloadBase(PyObject* value) {
        int overflow = 0;
        const auto number = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) throw PythonError{};
        if (overflow != 0) {
            origin = PyLong_AsUnsignedLongLong(value);
            if (PyErr_Occurred()) throw PythonError{};
        } else if (number < 0) {
            prefix = static_cast<std::uint64_t>(-(number + 1)) + 1;
        } else {
            origin = static_cast<std::uint64_t>(number);
        }
    }
};

int inflate_record(Inflate inflate, const Buffer& payload, const Record& record,
                   const PayloadBase& base, Byte* destination) noexcept {
    const auto area = record.area();
    if (!area || area >= ULONG_MAX || !record.length || record.offset < base.origin) return invalid_bounds;
    const auto relative = record.offset - base.origin;
    const auto size = static_cast<std::uint64_t>(payload.shape(0));
    if (base.prefix > size || relative >= size - base.prefix) return invalid_bounds;
    const auto at = relative + base.prefix;
    if (record.length > size - at) return invalid_bounds;
    unsigned long written = static_cast<unsigned long>(area + 1), consumed = record.length;
    const int status = inflate(destination, &written,
                               reinterpret_cast<const Byte*>(payload.data()) + at, &consumed);
    if (status != 0) return status;
    return written == area && consumed == record.length ? 0 : invalid_length;
}

void check_input(const Buffer& payload, const Buffer& records) {
    payload.require<Byte>();
    payload.require_contiguous();
    if (records.itemsize() != record_bytes) value_error("Invalid confidence block record width");
}

PyObject* inflate_blocks(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *address, *payload_obj, *records_obj, *decoded_obj, *starts_obj, *base_obj;
        if (!PyArg_ParseTuple(args, "OOOOOO:inflate_confidence_blocks", &address, &payload_obj,
                              &records_obj, &base_obj, &decoded_obj, &starts_obj)) return nullptr;
        const auto inflate = callback<Inflate>(address);
        const PayloadBase base(base_obj);
        Buffer payload(payload_obj, "payload", 1), records(records_obj, "records", 1);
        Buffer decoded(decoded_obj, "decoded", 1, true), starts(starts_obj, "starts", 1);
        check_input(payload, records);
        decoded.require<Byte>().require_contiguous();
        starts.require<std::int64_t>();
        if (starts.shape(0) < records.shape(0)) value_error("Missing confidence block starts");
        Index completed = 0;
        int status = 0;
        {
            GilRelease unlock;
            for (; completed < records.shape(0); ++completed) {
                const Record rec(records.ptr(completed));
                const auto start = starts.load<std::int64_t>(completed);
                if (start < 0 || start >= decoded.shape(0)
                        || rec.area() >= static_cast<std::uint64_t>(decoded.shape(0) - start)) {
                    status = invalid_bounds;
                    break;
                }
                status = inflate_record(inflate, payload, rec, base, reinterpret_cast<Byte*>(decoded.ptr(start)));
                if (status) break;
            }
        }
        return Py_BuildValue("ni", completed, status);
    });
}

PyObject* inflate_scatter(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *address, *payload_obj, *records_obj, *base_obj, *output_obj;
        if (!PyArg_ParseTuple(args, "OOOOO:inflate_confidence_scatter", &address, &payload_obj,
                              &records_obj, &base_obj, &output_obj)) return nullptr;
        const auto inflate = callback<Inflate>(address);
        const PayloadBase base(base_obj);
        Buffer payload(payload_obj, "payload", 1), records(records_obj, "records", 1);
        Buffer output(output_obj, "output", 3, true);
        check_input(payload, records);
        output.require<Byte>();
        Index completed = 0;
        std::uint64_t assigned = 0;
        int status = 0;
        {
            GilRelease unlock;
            // Allocate once for the immutable index used by production readers.
            std::size_t capacity = 1;
            for (Index i = 0; i < records.shape(0); ++i) {
                const Record rec(records.ptr(i));
                if (!rec.fits(output) || !rec.area() || rec.area() >= ULONG_MAX) break;
                capacity = std::max(capacity, static_cast<std::size_t>(rec.area() + 1));
            }
            std::unique_ptr<Byte[]> scratch(new Byte[capacity]);
            for (; completed < records.shape(0); ++completed) {
                const Record rec(records.ptr(completed));
                const auto area = rec.area();
                if (!rec.fits(output) || !area || area >= ULONG_MAX) {
                    status = invalid_bounds;
                    break;
                }
                const auto required = static_cast<std::size_t>(area + 1);
                // A writable index can change between records, including through an
                // aliased output. Check the snapshot actually passed to zlib.
                if (required > capacity) {
                    scratch.reset();
                    scratch.reset(new Byte[required]);
                    capacity = required;
                }
                status = inflate_record(inflate, payload, rec, base, scratch.get());
                if (status) break;
                for (Index y = 0; y < rec.h; ++y) {
                    const auto* row = scratch.get() + y * rec.w;
                    if (output.stride(2) == 1) {
                        std::memcpy(output.ptr(rec.z, rec.y + y, rec.x), row, rec.w);
                    } else {
                        for (Index x = 0; x < rec.w; ++x)
                            output.store<Byte>(row[x], rec.z, rec.y + y, rec.x + x);
                    }
                }
                assigned += rec.area();
            }
        }
        return Py_BuildValue("niK", completed, status, static_cast<unsigned long long>(assigned));
    });
}

PyMethodDef methods[] = {
    {"encode_confidence_cells", encode, METH_VARARGS, "Encode bounded masked confidence cells with zlib."},
    {"inflate_confidence_blocks", inflate_blocks, METH_VARARGS, "Inflate contiguous block segments with a guard byte."},
    {"inflate_confidence_scatter", inflate_scatter, METH_VARARGS, "Inflate and assign blocks in index order."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_codec(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
