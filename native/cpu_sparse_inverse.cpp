#include "cpu_cvol_source.hpp"

#include <algorithm>
#include <limits>

namespace xta::cpu {
namespace {
using coverage_geometry::Args;
using Index = Py_ssize_t;
using Key = std::int64_t;
using Owner = std::uint32_t;
using Count = std::uint64_t;

// The public sampler supplies int32 addresses. Keep signed int64 views usable
// as well, without coercing strides or copying a bounded strip.
class SignedIndices {
public:
    SignedIndices(PyObject* object, const char* label) : values_(object, label, 1) {
        narrow_ = values_.is<std::int32_t>();
        if (!narrow_) values_.require<Key>();
    }

    Index size() const noexcept { return values_.shape(0); }
    Key operator[](Index position) const noexcept {
        if (narrow_) return values_.load<std::int32_t>(position);
        return values_.load<Key>(position);
    }

private:
    Buffer values_;
    bool narrow_;
};

void require_mask(const Buffer& mask) {
    if (!mask.is<bool>()) mask.require<std::uint8_t>();
}

PyObject* sample_encoded_mask(PyObject*, PyObject* arguments) {
    return protect([&]() -> PyObject* {
        const Args args(arguments, 7);
        const Buffer encoded(args[0], "encoded mask payload", 1);
        const Buffer records(args[1], "encoded mask records", 2);
        const SignedIndices angles(args[2], "sample angles");
        const SignedIndices rows(args[3], "sample rows");
        const SignedIndices columns(args[4], "sample columns");
        const Buffer valid(args[5], "sample validity", 1);
        require_mask(valid);
        if (rows.size() != angles.size() || columns.size() != angles.size()
            || valid.shape(0) != angles.size()) {
            value_error("Sparse projection address arrays differ in shape");
        }
        const coverage::CvolSource source(encoded, records, args.flag(6));
        auto output = zeros({angles.size()}, "uint8");
        const Buffer result(output.get(), "sampled mask", 1, true);
        coverage::Stats stats{};
        {
            const GilRelease release;
            for (Index position = 0; position < angles.size(); ++position) {
                if (valid.load<std::uint8_t>(position)) {
                    const auto value = source.read(angles[position], rows[position], columns[position], stats);
                    result.store<std::uint8_t>(value != 0, position);
                }
            }
        }
        return Py_BuildValue("(NL)", output.release(), static_cast<long long>(stats.reads));
    });
}

Index key_index(Key key, Index length) {
    if (key < 0 || key >= length) {
        throw std::invalid_argument("Sparse projection map key exceeds its owner table");
    }
    return static_cast<Index>(key);
}

PyObject* count_keys(PyObject*, PyObject* arguments) {
    return protect([&]() -> PyObject* {
        const Args args(arguments, 2);
        const SignedIndices keys(args[0], "map keys");
        const Buffer counts(args[1], "key counts", 1, true);
        counts.require<Owner>();
        {
            const GilRelease release;
            for (Index position = 0; position < keys.size(); ++position) {
                const Index key = key_index(keys[position], counts.shape(0));
                counts.store<Owner>(counts.load<Owner>(key) + Owner{1}, key);
            }
        }
        Py_RETURN_NONE;
    });
}

PyObject* prefix_counts(PyObject*, PyObject* arguments) {
    return protect([&]() -> PyObject* {
        const Args args(arguments, 1);
        const Buffer counts(args[0], "key counts", 1);
        counts.require<Owner>();
        auto output = empty({checked_add(counts.shape(0), 1)}, "uint32");
        const Buffer offsets(output.get(), "key offsets", 1, true);
        {
            const GilRelease release;
            Count total = 0;
            offsets.store<Owner>(0, 0);
            for (Index key = 0; key < counts.shape(0); ++key) {
                total += counts.load<Owner>(key);
                if (total > std::numeric_limits<Owner>::max()) {
                    throw std::invalid_argument("Sparse projection map exceeds uint32 owner capacity");
                }
                offsets.store<Owner>(static_cast<Owner>(total), key + 1);
            }
        }
        return output.release();
    });
}

PyObject* fill_owners(PyObject*, PyObject* arguments) {
    return protect([&]() -> PyObject* {
        const Args args(arguments, 4);
        const SignedIndices keys(args[0], "map keys");
        const Buffer positions(args[1], "owner positions", 1);
        const Buffer cursors(args[2], "owner cursors", 1, true);
        const Buffer owners(args[3], "owners", 1, true);
        positions.require<Owner>();
        cursors.require<Owner>();
        owners.require<Owner>();
        if (positions.shape(0) != keys.size()) {
            value_error("Sparse projection owner positions differ from map keys in shape");
        }
        {
            const GilRelease release;
            for (Index position = 0; position < keys.size(); ++position) {
                const Index key = key_index(keys[position], cursors.shape(0));
                const Owner cursor = cursors.load<Owner>(key);
                if (cursor >= static_cast<Count>(owners.shape(0))
                    || cursor == std::numeric_limits<Owner>::max()) {
                    throw std::invalid_argument("Sparse projection owner cursor exceeds its table");
                }
                owners.store<Owner>(positions.load<Owner>(position), cursor);
                cursors.store<Owner>(cursor + Owner{1}, key);
            }
        }
        Py_RETURN_NONE;
    });
}

struct OwnerRange {
    Index first;
    Index stop;
};

OwnerRange owner_range(const Buffer& offsets, Index key, Index payload_size) {
    const Owner first = offsets.load<Owner>(key);
    const Owner stop = offsets.load<Owner>(key + 1);
    if (first > stop || stop > static_cast<Count>(payload_size)) {
        throw std::invalid_argument("Sparse projection offsets exceed their owner table");
    }
    return {static_cast<Index>(first), static_cast<Index>(stop)};
}

Index index_argument(const Args& args, int position) {
    const Index value = PyNumber_AsSsize_t(args[position], PyExc_OverflowError);
    if (PyErr_Occurred()) throw PythonError{};
    return value;
}

PyObject* scatter_upright_crop(PyObject*, PyObject* arguments) {
    return protect([&]() -> PyObject* {
        const Args args(arguments, 16);
        const Buffer crop(args[0], "source crop", 2);
        const Index azimuth = index_argument(args, 1);
        const Index first_row = index_argument(args, 2);
        const Index first_u = index_argument(args, 3);
        const Buffer key_offsets(args[4], "key offsets", 1);
        const Buffer owners(args[5], "owners", 1);
        const Buffer row_offsets(args[6], "row offsets", 1);
        const Buffer frames(args[7], "frames", 1);
        const Index plane_width = index_argument(args, 8);
        const Index processing_width = index_argument(args, 9);
        const Index base = index_argument(args, 10);
        const Index out_h = index_argument(args, 11);
        const Index out_w = index_argument(args, 12);
        const Buffer packed(args[13], "packed destination", 1, true);
        const Buffer bounds(args[14], "destination slice bounds", 2, true);
        const Buffer slice_counts(args[15], "destination slice counts", 1, true);
        require_mask(crop);
        key_offsets.require<Owner>();
        owners.require<Owner>();
        row_offsets.require<Owner>();
        frames.require<Owner>();
        packed.require<std::uint8_t>();
        bounds.require<std::int32_t>();
        slice_counts.require<Count>();

        const Index out_t = slice_counts.shape(0);
        const Index source_rows = row_offsets.shape(0) - 1;
        if (base < 0 || base > 2 || out_h <= 0 || out_w <= 0 || out_t <= 0
            || out_h > std::numeric_limits<std::int32_t>::max()
            || out_w > std::numeric_limits<std::int32_t>::max()
            || bounds.shape(0) != out_t || bounds.shape(1) != 4
            || plane_width != (base == 2 ? out_h : out_w)
            || first_row < 0 || first_row > source_rows || crop.shape(0) > source_rows - first_row
            || azimuth < 0 || processing_width <= 0 || first_u < 0 || first_u > processing_width
            || crop.shape(1) > processing_width - first_u) {
            value_error("Invalid sparse projection crop or destination geometry");
        }
        const Index first_key = checked_add(checked_extent(azimuth, processing_width), first_u);
        const Index key_count = key_offsets.shape(0) - 1;
        if (first_key > key_count || crop.shape(1) > key_count - first_key) {
            value_error("Sparse projection crop exceeds its map keys");
        }
        const Index packed_w = (out_w - 1) / 8 + 1;
        if (packed.shape(0) != checked_extent(out_t, checked_extent(out_h, packed_w))) {
            value_error("Sparse projection packed destination differs from its declared geometry");
        }

        const Index frame_count = base == 0 ? out_t : base == 1 ? out_h : out_w;
        const Index plane_area = checked_extent(base == 0 ? out_h : out_t, plane_width);
        Count foreground = 0;
        Count contributions = 0;
        Count unique = 0;
        {
            const GilRelease release;
            for (Index row = 0; row < crop.shape(0); ++row) {
                OwnerRange mapped_frames{};
                bool checked_row = false;
                for (Index column = 0; column < crop.shape(1); ++column) {
                    if (!crop.load<std::uint8_t>(row, column)) continue;
                    ++foreground;
                    if (!checked_row) {
                        mapped_frames = owner_range(row_offsets, first_row + row, frames.shape(0));
                        checked_row = true;
                    }
                    if (mapped_frames.first == mapped_frames.stop) continue;
                    const auto mapped_owners = owner_range(key_offsets, first_key + column, owners.shape(0));
                    for (Index frame_at = mapped_frames.first; frame_at < mapped_frames.stop; ++frame_at) {
                        const Owner frame = frames.load<Owner>(frame_at);
                        if (frame >= static_cast<Count>(frame_count)) {
                            throw std::invalid_argument("Sparse projection frame exceeds its destination");
                        }
                        for (Index at = mapped_owners.first; at < mapped_owners.stop; ++at) {
                            const Owner owner = owners.load<Owner>(at);
                            if (owner >= static_cast<Count>(plane_area)) {
                                throw std::invalid_argument("Sparse projection owner exceeds its destination plane");
                            }
                            const Index v = owner / plane_width;
                            const Index u = owner % plane_width;
                            const Index z = base == 0 ? static_cast<Index>(frame) : v;
                            const auto y = static_cast<std::int32_t>(base == 0 ? v : base == 1 ? static_cast<Index>(frame) : u);
                            const auto x = static_cast<std::int32_t>(base == 2 ? static_cast<Index>(frame) : u);
                            const Index address = (z * out_h + y) * packed_w + (x >> 3);
                            const auto bit = static_cast<std::uint8_t>(1u << (x & 7));
                            ++contributions;
                            const auto byte = packed.load<std::uint8_t>(address);
                            if (byte & bit) continue;
                            packed.store<std::uint8_t>(byte | bit, address);
                            ++unique;
                            slice_counts.store<Count>(slice_counts.load<Count>(z) + 1, z);
                            bounds.store<std::int32_t>(std::min(bounds.load<std::int32_t>(z, 0), y), z, 0);
                            bounds.store<std::int32_t>(std::max(bounds.load<std::int32_t>(z, 1), y + 1), z, 1);
                            bounds.store<std::int32_t>(std::min(bounds.load<std::int32_t>(z, 2), x), z, 2);
                            bounds.store<std::int32_t>(std::max(bounds.load<std::int32_t>(z, 3), x + 1), z, 3);
                        }
                    }
                }
            }
        }
        return Py_BuildValue("(KKK)", static_cast<unsigned long long>(foreground),
                             static_cast<unsigned long long>(contributions),
                             static_cast<unsigned long long>(unique));
    });
}

PyMethodDef methods[] = {
    {"sparse_inverse_sample_encoded_mask", sample_encoded_mask, METH_VARARGS, nullptr},
    {"sparse_inverse_count_keys", count_keys, METH_VARARGS, nullptr},
    {"sparse_inverse_prefix_counts", prefix_counts, METH_VARARGS, nullptr},
    {"sparse_inverse_fill_owners", fill_owners, METH_VARARGS, nullptr},
    {"sparse_inverse_scatter_upright_crop", scatter_upright_crop, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
} // namespace

int register_sparse_inverse(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}
} // namespace xta::cpu
