#include "cpu_kernels.hpp"

#include <algorithm>
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <immintrin.h>
#define XTA_PACKING_TARGETS 1
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace xta::cpu {
namespace {

using Size = Py_ssize_t;
using Word = std::uint32_t;

Word low_bits(unsigned count) noexcept {
    return count == 32 ? ~Word{0} : (Word{1} << count) - 1;
}

unsigned popcount(Word bits) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcount(bits));
#else
    // No POPCNT requirement on a generic x86-64 build.
    bits -= (bits >> 1) & 0x55555555u;
    bits = (bits & 0x33333333u) + ((bits >> 2) & 0x33333333u);
    return (((bits + (bits >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
#endif
}

unsigned trailing(Word bits) noexcept {
    if (!bits) return 32;
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, bits);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctz(bits));
#endif
}

unsigned leading(Word bits) noexcept {
    if (!bits) return 32;
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, bits);
    return 31 - static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_clz(bits));
#endif
}

struct DenseRow {
    const char* data;
    Size stride;

    Size group(Size, Size remaining) const noexcept { return std::min<Size>(16, remaining); }

    Word bits(Size x, unsigned count) const noexcept {
        const char* begin = data + x * stride;
        Word result = 0;
        for (unsigned bit = 0; bit < count; ++bit) {
            result |= static_cast<Word>(begin[static_cast<Size>(bit) * stride] != 0) << bit;
        }
        return result;
    }
};

struct BitsetRow {
    const char* data;
    Size start_bit;

    Word bits(Size x, unsigned count) const noexcept {
        const Size at = start_bit + x;
        const unsigned shift = static_cast<unsigned>(at % 32);
        const char* address = data + (at / 32) * sizeof(Word);
        Word first;
        std::memcpy(&first, address, sizeof(first));
        Word result = first >> shift;
        if (shift + count > 32) {
            Word next;
            std::memcpy(&next, address + sizeof(Word), sizeof(next));
            result |= next << (32 - shift);
        }
        return result & low_bits(count);
    }
};

struct DenseSource {
    const Buffer& source;
    Size first;

    DenseRow row(Size z, Size y) const noexcept {
        return {source.ptr(first + z, y, 0), source.stride(2)};
    }
};

struct BitsetSource {
    const Buffer& words;
    Size height, width, first_bit;

    BitsetRow row(Size z, Size y) const noexcept {
        return {words.data(), first_bit + (z * height + y) * width};
    }
};

void check_dense(const Buffer& volume, Size first, Size count) {
    volume.require<std::uint8_t>();
    if (volume.shape(1) < 1 || volume.shape(2) < 1 || first < 0 || count < 0
            || first > volume.shape(0) || count > volume.shape(0) - first) {
        value_error("Invalid dense packing geometry");
    }
}

void check_words(const Buffer& words, Size height, Size width, Size start, Size count) {
    words.require<Word>().require_contiguous();
    if (height < 1 || width < 1 || count < 0 || start < 0) {
        value_error("Invalid packed source geometry");
    }
    const Size bits = checked_extent(checked_extent(count, height), width);
    const Size stop = checked_add(start, bits);
    if (stop / 32 + (stop % 32 != 0) > words.shape(0)) {
        value_error("Packed source is shorter than the requested slab");
    }
}

struct RowBounds {
    Size first, stop, count;
};

template<class Row>
RowBounds scan_groups(const Row& row, Size width) noexcept {
    RowBounds bounds{width, 0, 0};
    for (Size x = 0; x < width;) {
        const auto step = row.group(x, width - x);
        const auto mask = row.bits(x, static_cast<unsigned>(step));
        if (mask) {
            bounds.count += popcount(mask);
            if (bounds.first == width) bounds.first = x + trailing(mask);
            bounds.stop = x + (32 - leading(mask));
        }
        x += step;
    }
    return bounds;
}

RowBounds scan_row(const BitsetRow& row, Size width) noexcept {
    RowBounds bounds{width, 0, 0};
    for (Size x = 0; x < width;) {
        const Size at = row.start_bit + x;
        const unsigned shift = static_cast<unsigned>(at % 32);
        const unsigned take = static_cast<unsigned>(std::min<Size>(64 - shift, width - x));
        const char* address = row.data + (at / 32) * sizeof(Word);
        Word low;
        std::memcpy(&low, address, sizeof(low));
        std::uint64_t pair = low;
        if (shift + take > 32) {
            Word high;
            std::memcpy(&high, address + sizeof(Word), sizeof(high));
            pair |= std::uint64_t(high) << 32;
        }
        pair >>= shift;
        if (take < 64) pair &= (std::uint64_t{1} << take) - 1;
        if (pair) {
            const auto a = static_cast<Word>(pair);
            const auto b = static_cast<Word>(pair >> 32);
            bounds.count += popcount(a) + popcount(b);
            if (bounds.first == width) bounds.first = x + (a ? trailing(a) : 32 + trailing(b));
            bounds.stop = x + (b ? 64 - leading(b) : 32 - leading(a));
        }
        x += take;
    }
    return bounds;
}

template<class Row>
RowBounds scan_row(const Row& row, Size width) noexcept {
    return scan_groups(row, width);
}

#if defined(_M_X64) || defined(__SSE2__)
__m128i load_16_pixels(const char* data) noexcept {
    __m128i values;
    std::memcpy(&values, data, sizeof(values));
    return values;
}

Word nonzero_bits_16(const char* data) noexcept {
    const auto zeros = _mm_cmpeq_epi8(load_16_pixels(data), _mm_setzero_si128());
    return static_cast<Word>(_mm_movemask_epi8(zeros)) ^ 0xffffu;
}

#if defined(XTA_PACKING_TARGETS)
__attribute__((target("avx2,popcnt")))
RowBounds scan_dense_avx2(const char* row, Size width) noexcept {
    RowBounds bounds{width, 0, 0};
    Size x = 0;
    for (; x <= width - 32; x += 32) {
        __m256i pixels;
        std::memcpy(&pixels, row + x, sizeof(pixels));
        const auto zero = _mm256_cmpeq_epi8(pixels, _mm256_setzero_si256());
        const Word mask = ~static_cast<Word>(_mm256_movemask_epi8(zero));
        if (mask) {
            bounds.count += __builtin_popcount(mask);
            if (bounds.first == width) bounds.first = x + trailing(mask);
            bounds.stop = x + (32 - leading(mask));
        }
    }
    for (; x < width; ++x) {
        if (row[x]) {
            ++bounds.count;
            if (bounds.first == width) bounds.first = x;
            bounds.stop = x + 1;
        }
    }
    return bounds;
}
#endif

RowBounds scan_row(const DenseRow& row, Size width) noexcept {
    if (row.stride != 1 || width < 16) return scan_groups(row, width);
#if defined(XTA_PACKING_TARGETS)
    // GCC and Clang's feature check includes OS support for YMM state.
    if (width >= 64 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt")) {
        return scan_dense_avx2(row.data, width);
    }
#endif
    // Sum 0/1 bytes in two independent 64-bit lanes. Bound searches then
    // inspect only the empty margins and one occupied group at each edge.
    __m128i counts = _mm_setzero_si128();
    const auto ones = _mm_set1_epi8(1);
    const Size vector_stop = width - width % 16;
    for (Size x = 0; x < vector_stop; x += 16) {
        const auto membership = _mm_min_epu8(load_16_pixels(row.data + x), ones);
        counts = _mm_add_epi64(counts, _mm_sad_epu8(membership, _mm_setzero_si128()));
    }
    std::uint64_t lanes[2];
    std::memcpy(lanes, &counts, sizeof(lanes));
    Size count = static_cast<Size>(lanes[0] + lanes[1]);
    for (Size x = vector_stop; x < width; ++x) count += row.data[x] != 0;
    if (!count) return {width, 0, 0};

    Size first = 0;
    while (first <= width - 16) {
        const Word bits = nonzero_bits_16(row.data + first);
        if (bits) {
            first += trailing(bits);
            break;
        }
        first += 16;
    }
    // If all full groups were empty, count guarantees an occupied tail byte.
    if (first >= vector_stop) {
        while (row.data[first] == 0) ++first;
    }
    Size stop = width;
    while (stop >= 16) {
        const Word bits = nonzero_bits_16(row.data + stop - 16);
        if (bits) return {first, stop - 16 + (32 - leading(bits)), count};
        stop -= 16;
    }
    while (row.data[stop - 1] == 0) --stop;
    return {first, stop, count};
}
#endif

template<class Row>
void pack_row(const Row& row, Size first, Size stop, std::uint8_t* out) noexcept {
#if defined(_M_X64) || defined(__SSE2__)
    if constexpr (std::is_same_v<Row, DenseRow>) {
        if (row.stride == 1) {
            for (; stop - first >= 16; first += 16, out += 2) {
                const Word bits = nonzero_bits_16(row.data + first);
                out[0] = static_cast<std::uint8_t>(bits);
                out[1] = static_cast<std::uint8_t>(bits >> 8);
            }
        }
    }
#endif
    while (first < stop) {
        const Size step = std::min<Size>(8, stop - first);
        *out++ = static_cast<std::uint8_t>(row.bits(first, static_cast<unsigned>(step)));
        first += step;
    }
}

template<bool FullFields, class Source>
PyObject* scan(const Source& source, Size height, Size width, Size count, Size output_first = 0) {
    if (output_first < 0) value_error("Negative packed destination slice");
    if (count) checked_add(output_first, count - 1);
    constexpr int box_column = FullFields ? 1 : 0;
    Size payload_size = 0;
    Object result = zeros({count, FullFields ? 8 : 5}, "int64");
    Buffer bounds(result.get(), "crop bounds", 2, true);
    {
        GilRelease unlocked;
        for (Size z = 0; z < count; ++z) {
            Size top = height, bottom = 0, left = width, right = 0, foreground = 0;
            for (Size y = 0; y < height; ++y) {
                const auto row = scan_row(source.row(z, y), width);
                if (row.count) {
                    foreground += row.count;
                    top = std::min(top, y);
                    bottom = y + 1;
                    left = std::min(left, row.first);
                    right = std::max(right, row.stop);
                }
            }
            if (foreground) {
                bounds.store<std::int64_t>(top, z, box_column);
                bounds.store<std::int64_t>(bottom, z, box_column + 1);
                bounds.store<std::int64_t>(left, z, box_column + 2);
                bounds.store<std::int64_t>(right, z, box_column + 3);
                bounds.store<std::int64_t>(foreground, z, box_column + 4);
            }
            if constexpr (FullFields) {
                const Size columns = foreground ? right - left : 0;
                const Size row_bytes = columns / 8 + (columns % 8 != 0);
                const Size size = foreground ? checked_extent(bottom - top, row_bytes) : 0;
                bounds.store<std::int64_t>(output_first + z, z, 0);
                bounds.store<std::int64_t>(payload_size, z, 6);
                bounds.store<std::int64_t>(size, z, 7);
                payload_size = checked_add(payload_size, size);
            }
        }
    }
    if constexpr (FullFields) return Py_BuildValue("On", result.get(), payload_size);
    return result.release();
}

struct Crop {
    Size y0, y1, x0, x1;
};

template<class Source>
PyObject* encode(const Source& source, Size height, Size width, const Buffer& metadata,
                 const Buffer& payload) {
    metadata.require<std::int64_t>();
    payload.require<std::uint8_t>().require_contiguous();
    if (metadata.shape(1) != 5) value_error("Packed metadata must contain five fields");
    // Pin a validated snapshot. No malformed later row can leave the first row
    // committed, and changing an exported metadata view cannot extend a write.
    std::vector<Crop> crops;
    crops.reserve(metadata.shape(0));
    Size required = 0;
    for (Size z = 0; z < metadata.shape(0); ++z) {
        const auto y0 = metadata.load<std::int64_t>(z, 0);
        const auto y1 = metadata.load<std::int64_t>(z, 1);
        const auto x0 = metadata.load<std::int64_t>(z, 2);
        const auto x1 = metadata.load<std::int64_t>(z, 3);
        if (y0 < 0 || y0 > y1 || y1 > height || x0 < 0 || x0 > x1 || x1 > width) {
            value_error("Packed metadata crop is outside the source plane");
        }
        const Size crop_width = static_cast<Size>(x1 - x0);
        const Size row_bytes = crop_width / 8 + (crop_width % 8 != 0);
        required = checked_add(required, checked_extent(static_cast<Size>(y1 - y0), row_bytes));
        crops.push_back({static_cast<Size>(y0), static_cast<Size>(y1),
                         static_cast<Size>(x0), static_cast<Size>(x1)});
    }
    if (required != payload.shape(0)) value_error("Packed payload size differs from its metadata");
    {
        GilRelease unlocked;
        Size cursor = 0;
        for (Size z = 0; z < metadata.shape(0); ++z) {
            const auto& crop = crops[z];
            const Size columns = crop.x1 - crop.x0;
            const Size row_bytes = columns / 8 + (columns % 8 != 0);
            for (Size y = crop.y0; y < crop.y1 && row_bytes; ++y) {
                const auto row = source.row(z, y);
                auto* out = reinterpret_cast<std::uint8_t*>(payload.data()) + cursor;
                pack_row(row, crop.x0, crop.x1, out);
                cursor += row_bytes;
            }
        }
    }
    Py_RETURN_NONE;
}

template<bool FullFields>
PyObject* dense_metadata(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* volume;
        Size first, count, output_first = 0;
        if constexpr (FullFields) {
            if (!PyArg_ParseTuple(args, "Onnn", &volume, &first, &count, &output_first)) return nullptr;
        } else {
            if (!PyArg_ParseTuple(args, "Onn", &volume, &first, &count)) return nullptr;
        }
        Buffer source(volume, "source", 3);
        check_dense(source, first, count);
        return scan<FullFields>(DenseSource{source, first}, source.shape(1), source.shape(2), count, output_first);
    });
}

PyObject* dense_encode(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *volume, *meta, *out;
        Size first;
        if (!PyArg_ParseTuple(args, "OnOO", &volume, &first, &meta, &out)) return nullptr;
        Buffer source(volume, "source", 3), metadata(meta, "metadata", 2), payload(out, "payload", 1, true);
        check_dense(source, first, metadata.shape(0));
        return encode(DenseSource{source, first}, source.shape(1), source.shape(2), metadata, payload);
    });
}

template<bool FullFields>
PyObject* words_metadata(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* data;
        Size height, width, start, count, output_first = 0;
        if constexpr (FullFields) {
            if (!PyArg_ParseTuple(args, "Onnnnn", &data, &height, &width, &start, &count, &output_first)) return nullptr;
        } else {
            if (!PyArg_ParseTuple(args, "Onnnn", &data, &height, &width, &start, &count)) return nullptr;
        }
        Buffer words(data, "bitset", 1);
        check_words(words, height, width, start, count);
        return scan<FullFields>(BitsetSource{words, height, width, start}, height, width, count, output_first);
    });
}

PyObject* words_encode(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *data, *meta, *out;
        Size height, width, start;
        if (!PyArg_ParseTuple(args, "OnnnOO", &data, &height, &width, &start, &meta, &out)) return nullptr;
        Buffer words(data, "bitset", 1), metadata(meta, "metadata", 2), payload(out, "payload", 1, true);
        check_words(words, height, width, start, metadata.shape(0));
        return encode(BitsetSource{words, height, width, start}, height, width, metadata, payload);
    });
}

struct UnpackedByte {
    std::array<std::uint8_t, 8> pixels{};
    unsigned foreground = 0;
};

constexpr auto unpacked_bytes() {
    std::array<UnpackedByte, 256> values{};
    for (unsigned value = 0; value < values.size(); ++value) {
        auto& entry = values[value];
        for (unsigned bit = 0; bit < entry.pixels.size(); ++bit) {
            entry.pixels[bit] = static_cast<std::uint8_t>((value >> bit) & 1);
            entry.foreground += entry.pixels[bit];
        }
    }
    return values;
}

PyObject* unpack_plane_and_count(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *packed_object, *destination_object;
        if (!PyArg_ParseTuple(args, "OO:unpack_packed_plane_and_count", &packed_object, &destination_object))
            return nullptr;
        const Buffer packed(packed_object, "packed union plane", 2);
        const Buffer destination(destination_object, "expanded union plane", 2, true);
        packed.require<std::uint8_t>().require_contiguous();
        destination.require<std::uint8_t>().require_contiguous();
        const Size height = destination.shape(0), width = destination.shape(1);
        const Size full = width / 8;
        const auto tail = static_cast<unsigned>(width % 8);
        if (packed.shape(0) != height || packed.shape(1) != full + (tail != 0))
            value_error("Packed union plane differs from its destination shape");
        if (!height || !width) return PyLong_FromLong(0);
        const auto from = reinterpret_cast<std::uintptr_t>(packed.data());
        const auto to = reinterpret_cast<std::uintptr_t>(destination.data());
        if ((from <= to && to - from < std::uintmax_t(packed.nbytes()))
            || (to < from && from - to < std::uintmax_t(destination.nbytes())))
            value_error("Packed union input and output must be disjoint");
        static constexpr auto lut = unpacked_bytes();
        Size foreground = 0;
        {
            GilRelease release;
            for (Size y = 0; y < height; ++y) {
                const auto* input = reinterpret_cast<const std::uint8_t*>(packed.ptr(y, 0));
                auto* output = destination.ptr(y, 0);
                for (Size x = 0; x < full; ++x, output += 8) {
                    const auto& entry = lut[input[x]];
                    std::memcpy(output, entry.pixels.data(), entry.pixels.size());
                    foreground += entry.foreground;
                }
                if (tail) {
                    const auto& entry = lut[input[full] & low_bits(tail)];
                    std::memcpy(output, entry.pixels.data(), tail);
                    foreground += entry.foreground;
                }
            }
        }
        return PyLong_FromSsize_t(foreground);
    });
}

bool has_foreground_word(const char* data) noexcept {
    std::uint64_t word;
    std::memcpy(&word, data, sizeof(word));
    return word != 0;
}

Size first_foreground(const DenseRow& row, Size width) noexcept {
    Size x = 0;
    if (row.stride == 1) {
        while (width - x >= 8 && !has_foreground_word(row.data + x)) x += 8;
    }
    while (x < width && row.data[x * row.stride] == 0) ++x;
    return x;
}

Size foreground_stop(const DenseRow& row, Size lower, Size width) noexcept {
    Size x = width;
    if (row.stride == 1) {
        while (x - lower >= 8 && !has_foreground_word(row.data + x - 8)) x -= 8;
    }
    while (x > lower && row.data[(x - 1) * row.stride] == 0) --x;
    return x;
}

PyObject* binary_slice_metadata(PyObject*, PyObject* input) {
    return protect([&]() -> PyObject* {
        const Buffer source(input, "worker mask volume", 3);
        if (!source.is<std::uint8_t>() && !source.is<bool>()) {
            type_error("Worker binary metadata requires a uint8 or bool mask volume");
        }
        const Size depth = source.shape(0);
        const Size height = source.shape(1);
        const Size width = source.shape(2);
        const Size packed_width = height / 8 + (height % 8 != 0);
        auto any_array = zeros({depth}, "bool");
        auto bbox_array = zeros({depth, 4}, "int64");
        auto rows_array = zeros({depth, packed_width}, "uint8");
        if (height && width && depth) {
            const Buffer any(any_array.get(), "worker occupied slices", 1, true);
            const Buffer bounds(bbox_array.get(), "worker slice bounds", 2, true);
            const Buffer rows(rows_array.get(), "worker occupied rows", 2, true);
            const DenseSource volume{source, 0};
            GilRelease release;
            for (Size z = 0; z < depth; ++z) {
                Size top = height, bottom = 0, left = width, right = 0;
                for (Size y = 0; y < height; ++y) {
                    const auto row = volume.row(z, y);
                    const Size first = first_foreground(row, width);
                    if (first == width) continue;

                    top = std::min(top, y);
                    bottom = y + 1;
                    left = std::min(left, first);
                    // An earlier row already covers every column below right.
                    const Size stop = foreground_stop(row, std::max(first, right), width);
                    right = std::max(right, stop);

                    // Worker receipts use NumPy's MSB-first row occupancy.
                    const auto bit = static_cast<std::uint8_t>(0x80u >> (y % 8));
                    rows.store<std::uint8_t>(rows.load<std::uint8_t>(z, y / 8) | bit, z, y / 8);
                }
                if (bottom != 0) {
                    any.store<bool>(true, z);
                    bounds.store<std::int64_t>(top, z, 0);
                    bounds.store<std::int64_t>(bottom, z, 1);
                    bounds.store<std::int64_t>(left, z, 2);
                    bounds.store<std::int64_t>(right, z, 3);
                }
            }
        }
        return PyTuple_Pack(3, any_array.get(), bbox_array.get(), rows_array.get());
    });
}

PyMethodDef methods[] = {
    {"unpack_packed_plane_and_count", unpack_plane_and_count, METH_VARARGS,
     "Expand one packed union plane into a disjoint byte destination and count foreground."},
    {"binary_slice_metadata_from_volume", binary_slice_metadata, METH_O, "Scan worker slice bounds and row occupancy."},
    {"dense_mask_fields", dense_metadata<true>, METH_VARARGS, "Build CVOL records and their packed byte count."},
    {"dense_mask_metadata", dense_metadata<false>, METH_VARARGS, "Return exact per-plane bounds and foreground counts."},
    {"dense_mask_encode", dense_encode, METH_VARARGS, "Pack nonzero uint8 members into a bounded crop payload."},
    {"bitset_mask_fields", words_metadata<true>, METH_VARARGS, "Build CVOL records directly from source bits."},
    {"bitset_mask_metadata", words_metadata<false>, METH_VARARGS, "Scan contiguous uint32 source mask bits."},
    {"bitset_mask_encode", words_encode, METH_VARARGS, "Pack crop rows from contiguous source mask bits."},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace

int register_packing(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
