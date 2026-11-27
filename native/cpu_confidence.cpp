#include "cpu_kernels.hpp"
#include "cpu_coverage_pull.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Byte = std::uint8_t;
using Word = std::int64_t;

constexpr Index max_copy_cells = 16;
constexpr Index max_copy_bytes = 256 * 1024;

const Buffer& require_mask(const Buffer& mask) {
    return mask.is<bool>() ? mask : mask.require<Byte>();
}

void equal_shape(const Buffer& lhs, const Buffer& rhs) {
    for (int axis = 0; axis < lhs.ndim(); ++axis) {
        if (lhs.shape(axis) != rhs.shape(axis)) {
            value_error("Confidence buffer shapes differ");
        }
    }
}

bool in_range(Index first, Index stop, Index limit) {
    return first >= 0 && first <= stop && stop <= limit;
}

void require_frame(const Buffer& source, Index z) {
    if (z < 0 || z >= source.shape(0)) {
        value_error("Confidence frame lies outside the source");
    }
}

void require_rect(const Buffer& source, Index y0, Index y1, Index x0, Index x1) {
    const int vertical = source.ndim() - 2;
    if (!in_range(y0, y1, source.shape(vertical)) || !in_range(x0, x1, source.shape(vertical + 1))) {
        value_error("Confidence crop lies outside the source");
    }
}

Object prefix(const Object& array, Index stop) {
    Object end(PyLong_FromSsize_t(stop));
    Object slice(PySlice_New(nullptr, end.get(), nullptr));
    return Object(PyObject_GetItem(array.get(), slice.get()));
}

PyObject* masked_cell_bounds(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg;
        Index z, y0, y1, x0, x1, block;
        if (!PyArg_ParseTuple(args, "OOnnnnnn", &mask_arg, &scores_arg, &z, &y0, &y1, &x0, &x1, &block)) {
            throw PythonError{};
        }
        Buffer mask(mask_arg, "mask", 3), scores(scores_arg, "scores", 3);
        require_mask(mask);
        scores.require<Byte>();
        equal_shape(mask, scores);
        require_frame(mask, z);
        require_rect(mask, y0, y1, x0, x1);
        if (block <= 0) {
            value_error("Confidence block size must be positive");
        }
        const Index first_y = y0 / block * block, first_x = x0 / block * block;
        const Index row_cells = (y1 - first_y) / block + ((y1 - first_y) % block != 0);
        const Index col_cells = (x1 - first_x) / block + ((x1 - first_x) % block != 0);
        Object records = empty({checked_extent(row_cells, col_cells), 5}, "int64");
        Buffer output(records.get(), "cell bounds", 2, true);
        Index count = 0;
        {
            GilRelease unlocked;
            for (Index y = first_y; y < y1;) {
                const Index a = std::max(y, y0), b = y + std::min(y1 - y, block);
                for (Index x = first_x; x < x1;) {
                    const Index c = std::max(x, x0), d = x + std::min(x1 - x, block);
                    Index top = b, bottom = a, left = d, right = c, known = 0;
                    for (Index r = a; r < b; ++r) {
                        for (Index p = c; p < d; ++p) {
                            if (mask.load<Byte>(z, r, p) && scores.load<Byte>(z, r, p)) {
                                top = std::min(top, r);
                                bottom = std::max(bottom, r + 1);
                                left = std::min(left, p);
                                right = std::max(right, p + 1);
                                ++known;
                            }
                        }
                    }
                    if (known) {
                        output.store<Word>(top, count, 0);
                        output.store<Word>(bottom, count, 1);
                        output.store<Word>(left, count, 2);
                        output.store<Word>(right, count, 3);
                        output.store<Word>(known, count++, 4);
                    }
                    x = d;
                }
                y = b;
            }
        }
        return prefix(records, count).release();
    });
}

PyObject* copy_masked_cell(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg;
        Index z, y0, y1, x0, x1;
        if (!PyArg_ParseTuple(args, "OOnnnnn", &mask_arg, &scores_arg, &z, &y0, &y1, &x0, &x1)) {
            throw PythonError{};
        }
        Buffer mask(mask_arg, "mask", 3), scores(scores_arg, "scores", 3);
        require_mask(mask);
        scores.require<Byte>();
        equal_shape(mask, scores);
        require_frame(mask, z);
        require_rect(mask, y0, y1, x0, x1);
        Object result = empty({y1 - y0, x1 - x0}, "uint8");
        Buffer output(result.get(), "masked cell", 2, true);
        {
            GilRelease unlocked;
            for (Index y = y0; y < y1; ++y)
                for (Index x = x0; x < x1; ++x)
                    output.store<Byte>(mask.load<Byte>(z, y, x) ? scores.load<Byte>(z, y, x) : 0,
                                       y - y0, x - x0);
        }
        return result.release();
    });
}

PyObject* copy_masked_cells_batch(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg, *bounds_arg;
        Index z, max_bytes;
        if (!PyArg_ParseTuple(args, "OOnOn", &mask_arg, &scores_arg, &z, &bounds_arg, &max_bytes)) {
            throw PythonError{};
        }
        Buffer mask(mask_arg, "mask", 3), scores(scores_arg, "scores", 3), bounds(bounds_arg, "bounds", 2);
        require_mask(mask);
        scores.require<Byte>();
        bounds.require<Word>();
        equal_shape(mask, scores);
        require_frame(mask, z);
        const Index count = bounds.shape(0);
        if (count > max_copy_cells || bounds.shape(1) != 5 || !in_range(0, max_bytes, max_copy_bytes)) {
            value_error("Confidence copy batch exceeds its source/control bounds");
        }
        Object offsets = empty({count + 1}, "int64");
        Buffer boundaries(offsets.get(), "cell offsets", 1, true);
        boundaries.store<Word>(0, 0);
        std::array<std::array<Index, 4>, max_copy_cells> crops;
        Index total = 0;
        for (Index item = 0; item < count; ++item) {
            const Index y0 = bounds.load<Word>(item, 0), y1 = bounds.load<Word>(item, 1);
            const Index x0 = bounds.load<Word>(item, 2), x1 = bounds.load<Word>(item, 3);
            if (y0 >= y1 || x0 >= x1 || !in_range(y0, y1, mask.shape(1))
                || !in_range(x0, x1, mask.shape(2))) {
                value_error("Confidence copy cell exceeds its source geometry");
            }
            total = checked_add(total, checked_extent(y1 - y0, x1 - x0));
            boundaries.store<Word>(total, item + 1);
            crops[item] = {y0, y1, x0, x1};
        }
        if (total > max_bytes) {
            PyErr_SetString(PyExc_MemoryError, "Confidence copy batch exceeds its admitted byte credit");
            throw PythonError{};
        }
        Object result = empty({total}, "uint8");
        Buffer output(result.get(), "masked cells", 1, true);
        {
            GilRelease unlocked;
            Index at = 0;
            for (Index item = 0; item < count; ++item) {
                const auto [y0, y1, x0, x1] = crops[item];
                for (Index y = y0; y < y1; ++y)
                    for (Index x = x0; x < x1; ++x)
                        output.store<Byte>(mask.load<Byte>(z, y, x) ? scores.load<Byte>(z, y, x) : 0, at++);
            }
        }
        return Py_BuildValue("NN", result.release(), offsets.release());
    });
}

PyObject* zero_unmasked_scores_flat(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg;
        Index first, stop;
        if (!PyArg_ParseTuple(args, "OOnn", &mask_arg, &scores_arg, &first, &stop)) {
            throw PythonError{};
        }
        Buffer mask(mask_arg, "mask", 1), scores(scores_arg, "scores", 1, true);
        require_mask(mask);
        scores.require<Byte>();
        equal_shape(mask, scores);
        if (!in_range(first, stop, scores.shape(0))) {
            value_error("Confidence flat range lies outside the source");
        }
        {
            GilRelease unlocked;
            if (mask.stride(0) == 1 && scores.stride(0) == 1 && first < stop) {
                const auto* support = reinterpret_cast<const Byte*>(mask.ptr(first));
                auto* output = reinterpret_cast<Byte*>(scores.ptr(first));
                for (Index k = 0; k < stop - first; ++k)
                    output[k] &= static_cast<Byte>(0 - (support[k] != 0));
            } else {
                for (Index k = first; k < stop; ++k)
                    scores.store<Byte>(scores.load<Byte>(k) & static_cast<Byte>(0 - (mask.load<Byte>(k) != 0)), k);
            }
        }
        Py_RETURN_NONE;
    });
}

PyObject* zero_unmasked_scores_boxes(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg, *active_arg, *boxes_arg;
        Index z0, z1;
        if (!PyArg_ParseTuple(args, "OOOOnn", &mask_arg, &scores_arg, &active_arg, &boxes_arg, &z0, &z1)) {
            throw PythonError{};
        }
        Buffer mask(mask_arg, "mask", 3), scores(scores_arg, "scores", 3, true);
        Buffer active(active_arg, "active frames", 1), boxes(boxes_arg, "boxes", 2);
        require_mask(mask);
        scores.require<Byte>();
        require_mask(active);
        boxes.require<Word>();
        equal_shape(mask, scores);
        if (!in_range(z0, z1, mask.shape(0)) || z1 > active.shape(0) || z1 > boxes.shape(0)
            || boxes.shape(1) < 4) {
            value_error("Confidence box range lies outside the source");
        }
        {
            GilRelease unlocked;
            for (Index z = z0; z < z1; ++z) {
                if (!active.load<Byte>(z)) continue;
                const Index y0 = boxes.load<Word>(z, 0), y1 = boxes.load<Word>(z, 1);
                const Index x0 = boxes.load<Word>(z, 2), x1 = boxes.load<Word>(z, 3);
                if (!in_range(y0, y1, mask.shape(1)) || !in_range(x0, x1, mask.shape(2))) {
                    throw std::invalid_argument("Confidence box lies outside the source");
                }
                for (Index y = y0; y < y1; ++y) {
                    if (mask.stride(2) == 1 && scores.stride(2) == 1 && x0 < x1) {
                        const auto* support = reinterpret_cast<const Byte*>(mask.ptr(z, y, x0));
                        auto* output = reinterpret_cast<Byte*>(scores.ptr(z, y, x0));
                        for (Index k = 0; k < x1 - x0; ++k)
                            output[k] &= static_cast<Byte>(0 - (support[k] != 0));
                    } else {
                        for (Index x = x0; x < x1; ++x)
                            scores.store<Byte>(scores.load<Byte>(z, y, x)
                                & static_cast<Byte>(0 - (mask.load<Byte>(z, y, x) != 0)), z, y, x);
                    }
                }
            }
        }
        Py_RETURN_NONE;
    });
}

#if defined(_M_X64) || defined(__SSE2__)
std::pair<Index, Index> mask_eager_unit_stride_row(const Byte* support, Byte* scores, Index width) {
    const __m128i zero = _mm_setzero_si128();
    Index x = 0, first_block = 0, last_block = 0;
    unsigned first_bits = 0, last_bits = 0;
    for (; x <= width - 16; x += 16) {
        __m128i mask_values, score_values;
        std::memcpy(&mask_values, support + x, 16);
        std::memcpy(&score_values, scores + x, 16);
        const __m128i kept = _mm_andnot_si128(_mm_cmpeq_epi8(mask_values, zero), score_values);
        std::memcpy(scores + x, &kept, 16);
        const auto bits = static_cast<unsigned>(~_mm_movemask_epi8(_mm_cmpeq_epi8(kept, zero))) & 0xffffu;
        if (bits) {
            if (!first_bits) {
                first_block = x;
                first_bits = bits;
            }
            last_block = x;
            last_bits = bits;
        }
    }
    Index first = width, stop = 0;
    if (first_bits) {
#if defined(_MSC_VER)
        unsigned long low, high;
        _BitScanForward(&low, first_bits);
        _BitScanReverse(&high, last_bits);
        first = first_block + low;
        stop = last_block + high + 1;
#else
        first = first_block + __builtin_ctz(first_bits);
        stop = last_block + 32 - __builtin_clz(last_bits);
#endif
    }
    for (; x < width; ++x) {
        const Byte kept = scores[x] & static_cast<Byte>(0 - (support[x] != 0));
        scores[x] = kept;
        first = std::min(first, kept ? x : width);
        stop = std::max(stop, kept ? x + 1 : Index{0});
    }
    return {first, stop};
}
#endif

PyObject* mask_eager_scores_and_bound(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_arg, *scores_arg;
        if (!PyArg_ParseTuple(args, "OO", &mask_arg, &scores_arg)) throw PythonError{};
        Buffer mask(mask_arg, "mask", 3), scores(scores_arg, "scores", 3, true);
        require_mask(mask);
        scores.require<Byte>();
        equal_shape(mask, scores);
        const Index depth = scores.shape(0), height = scores.shape(1), width = scores.shape(2);
        Object result = zeros({depth, 4}, "int64");
        Buffer bounds(result.get(), "score bounds", 2, true);
        {
            GilRelease unlocked;
#if defined(_M_X64) || defined(__SSE2__)
            const bool vector_rows = width >= 16 && mask.stride(2) == 1 && scores.stride(2) == 1;
#endif
            for (Index z = 0; z < depth; ++z) {
                Index top = height, bottom = 0, left = width, right = 0;
                for (Index y = 0; y < height; ++y) {
                    Index first = width, stop = 0;
#if defined(_M_X64) || defined(__SSE2__)
                    bool vector_safe = false;
                    if (vector_rows) {
                        const auto* support = reinterpret_cast<const Byte*>(mask.ptr(z, y, 0));
                        auto* output = reinterpret_cast<Byte*>(scores.ptr(z, y, 0));
                        const auto a = reinterpret_cast<std::uintptr_t>(support);
                        const auto b = reinterpret_cast<std::uintptr_t>(output);
                        // A shifted alias inside this row can change a future read.
                        const auto separation = std::max(a, b) - std::min(a, b);
                        vector_safe = separation == 0 || separation >= static_cast<std::uintptr_t>(width);
                        if (vector_safe) {
                            const auto row = mask_eager_unit_stride_row(support, output, width);
                            first = row.first;
                            stop = row.second;
                        }
                    }
                    if (!vector_safe)
#endif
                    {
                        for (Index x = 0; x < width; ++x) {
                            const Byte kept = scores.load<Byte>(z, y, x)
                                & static_cast<Byte>(0 - (mask.load<Byte>(z, y, x) != 0));
                            scores.store<Byte>(kept, z, y, x);
                            first = std::min(first, kept ? x : width);
                            stop = std::max(stop, kept ? x + 1 : Index{0});
                        }
                    }
                    if (stop) {
                        top = std::min(top, y);
                        bottom = y + 1;
                        left = std::min(left, first);
                        right = std::max(right, stop);
                    }
                }
                if (bottom) {
                    bounds.store<Word>(top, z, 0);
                    bounds.store<Word>(bottom, z, 1);
                    bounds.store<Word>(left, z, 2);
                    bounds.store<Word>(right, z, 3);
                }
            }
        }
        return result.release();
    });
}

PyObject* merge_rank_crop(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *plane_arg, *mask_arg;
        Index top, left, level;
        int check_previous;
        if (!PyArg_ParseTuple(args, "OOnnnp", &plane_arg, &mask_arg, &top, &left, &level, &check_previous)) {
            throw PythonError{};
        }
        Buffer plane(plane_arg, "rank plane", 2, true), mask(mask_arg, "mask", 2);
        plane.require<Byte>();
        require_mask(mask);
        require_rect(plane, top, checked_add(top, mask.shape(0)), left, checked_add(left, mask.shape(1)));
        bool populated = false, gap = false;
        const bool boolean = mask.is<bool>();
        {
            GilRelease unlocked;
            for (Index row = 0; row < mask.shape(0); ++row) {
                for (Index col = 0; col < mask.shape(1); ++col) {
                    const Byte value = boolean ? mask.load<bool>(row, col) : mask.load<Byte>(row, col);
                    if (value > 1) {
                        throw std::invalid_argument("Native cutoff masks must be binary");
                    }
                    if (value) {
                        populated = true;
                        if (check_previous && level > 0 && plane.load<Byte>(top + row, left + col) < level - 1)
                            gap = true;
                        plane.store<Byte>(static_cast<Byte>(level), top + row, left + col);
                    }
                }
            }
        }
        return Py_BuildValue("OO", populated ? Py_True : Py_False, gap ? Py_True : Py_False);
    });
}

void measure_rank_extents(const Buffer& region, Index levels, const Buffer& stats) {
    const Index h = region.shape(0), w = region.shape(1);
    for (Index i = 0; i <= levels; ++i) {
        stats.store<Word>(h, i, 0);
        stats.store<Word>(w, i, 2);
    }
    for (Index y = 0; y < h; ++y) {
        for (Index x = 0; x < w; ++x) {
            const Index rank = region.load<Byte>(y, x);
            if (!rank) continue;
            if (rank > levels) {
                throw std::invalid_argument("Projected value exceeds its declared cutoff rank");
            }
            stats.store<Word>(std::min<Word>(stats.load<Word>(rank, 0), y), rank, 0);
            stats.store<Word>(y + 1, rank, 1);
            stats.store<Word>(std::min<Word>(stats.load<Word>(rank, 2), x), rank, 2);
            stats.store<Word>(std::max<Word>(stats.load<Word>(rank, 3), x + 1), rank, 3);
            stats.store<Word>(stats.load<Word>(rank, 4) + 1, rank, 4);
        }
    }
    for (Index i = levels - 1; i > 0; --i) {
        for (int field = 0; field < 4; ++field) {
            const Word a = stats.load<Word>(i, field), b = stats.load<Word>(i + 1, field);
            stats.store<Word>((field & 1) ? std::max(a, b) : std::min(a, b), i, field);
        }
        stats.store<Word>(stats.load<Word>(i, 4) + stats.load<Word>(i + 1, 4), i, 4);
    }
}

PyObject* cutoff_extents(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* region_arg;
        Index levels;
        if (!PyArg_ParseTuple(args, "On", &region_arg, &levels)) throw PythonError{};
        Buffer region(region_arg, "rank region", 2);
        region.require<Byte>();
        if (levels < 0) value_error("Negative cutoff rank count");
        Object result = zeros({checked_add(levels, 1), 5}, "int64");
        Buffer stats(result.get(), "cutoff extents", 2, true);
        {
            GilRelease unlocked;
            measure_rank_extents(region, levels, stats);
        }
        return result.release();
    });
}

Index rank_row_bytes(Index width) {
    return width / 8 + (width % 8 != 0);
}

void pack_rank_bits(const Buffer& region, Index y0, Index y1, Index x0, Index x1, Index level,
                    Byte* output) noexcept {
    const Index width = x1 - x0;
    if (!width) return;
    const Index stride = region.stride(1);
#if defined(_M_X64) || defined(__SSE2__)
    const bool vectorize = stride == 1 && width >= 16 && level >= 0 && level <= 255;
    const __m128i threshold = _mm_set1_epi8(static_cast<char>(static_cast<Byte>(level)));
#endif
    for (Index y = y0; y < y1; ++y) {
        const auto* row = reinterpret_cast<const Byte*>(region.ptr(y, x0));
        Index x = 0;
#if defined(_M_X64) || defined(__SSE2__)
        if (vectorize) {
            const Index stop = width - width % 16;
            for (; x < stop; x += 16) {
                __m128i values;
                std::memcpy(&values, row + x, sizeof(values));
                // Unsigned min equals the threshold precisely where rank >= level.
                const auto keep = _mm_cmpeq_epi8(_mm_min_epu8(values, threshold), threshold);
                const unsigned bits = static_cast<unsigned>(_mm_movemask_epi8(keep));
                *output++ = static_cast<Byte>(bits);
                *output++ = static_cast<Byte>(bits >> 8);
            }
        }
#endif
        const Index byte_stop = width - width % 8;
        for (; x < byte_stop; x += 8) {
            unsigned bits = 0;
            for (Index bit = 0; bit < 8; ++bit)
                bits |= static_cast<unsigned>(row[(x + bit) * stride] >= level) << bit;
            *output++ = static_cast<Byte>(bits);
        }
        if (x < width) {
            unsigned bits = 0;
            for (Index bit = 0; bit < width - x; ++bit)
                bits |= static_cast<unsigned>(row[(x + bit) * stride] >= level) << bit;
            *output++ = static_cast<Byte>(bits);
        }
    }
}

PyObject* pack_rank_level(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* region_arg;
        Index y0, y1, x0, x1, level;
        if (!PyArg_ParseTuple(args, "Onnnnn", &region_arg, &y0, &y1, &x0, &x1, &level)) throw PythonError{};
        Buffer region(region_arg, "rank region", 2);
        region.require<Byte>();
        require_rect(region, y0, y1, x0, x1);
        Object result = empty({y1 - y0, rank_row_bytes(x1 - x0)}, "uint8");
        Buffer output(result.get(), "packed rank bits", 2, true);
        {
            GilRelease unlocked;
            pack_rank_bits(region, y0, y1, x0, x1, level, reinterpret_cast<Byte*>(output.data()));
        }
        return result.release();
    });
}

PyObject* pack_rank_levels(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *region_arg, *selected_arg;
        Index z, top, left, levels, payload_limit;
        if (!PyArg_ParseTuple(args, "OnnnnOn", &region_arg, &z, &top, &left, &levels,
                              &selected_arg, &payload_limit)) throw PythonError{};
        Buffer region(region_arg, "rank region", 2), selected(selected_arg, "selected ranks", 1);
        region.require<Byte>();
        selected.require<Word>();
        const Index requested = selected.shape(0);
        if (z < 0 || top < 0 || left < 0 || levels < 1 || levels > 255
            || requested > levels || payload_limit < 0) {
            value_error("Invalid multilevel rank packing bounds");
        }
        checked_add(top, region.shape(0));
        checked_add(left, region.shape(1));
        // Pin the input region and use a private ordered copy of the controls
        // across both GIL-free sections.
        std::vector<Word> active(requested);
        Word previous = 0;
        for (Index i = 0; i < requested; ++i) {
            const auto level = selected.load<Word>(i);
            if (level <= previous || level > levels) value_error("Selected ranks must be ascending and in bounds");
            active[i] = previous = level;
        }
        Object extents = zeros({levels + 1, 5}, "int64");
        Buffer stats(extents.get(), "cutoff extents", 2, true);
        Index count = 0, total = 0;
        {
            GilRelease unlocked;
            measure_rank_extents(region, levels, stats);
            for (; count < requested; ++count) {
                const auto level = active[count];
                if (!stats.load<Word>(level, 4)) break;
                const Index height = stats.load<Word>(level, 1) - stats.load<Word>(level, 0);
                const Index width = stats.load<Word>(level, 3) - stats.load<Word>(level, 2);
                total = checked_add(total, checked_extent(height, rank_row_bytes(width)));
            }
        }
        if (total > payload_limit) {
            PyErr_SetString(PyExc_MemoryError, "Packed rank plane exceeds its payload credit");
            throw PythonError{};
        }
        Object fields = empty({count, 8}, "int64");
        Object offsets = empty({count + 1}, "int64");
        Object payload = empty({total}, "uint8");
        Buffer records(fields.get(), "cutoff records", 2, true);
        Buffer boundaries(offsets.get(), "cutoff payload offsets", 1, true);
        Buffer bytes(payload.get(), "packed ranks", 1, true);
        {
            GilRelease unlocked;
            Index cursor = 0;
            boundaries.store<Word>(0, 0);
            for (Index i = 0; i < count; ++i) {
                const auto level = active[i];
                const Index y0 = stats.load<Word>(level, 0), y1 = stats.load<Word>(level, 1);
                const Index x0 = stats.load<Word>(level, 2), x1 = stats.load<Word>(level, 3);
                const Index size = (y1 - y0) * rank_row_bytes(x1 - x0);
                pack_rank_bits(region, y0, y1, x0, x1, level, reinterpret_cast<Byte*>(bytes.ptr(cursor)));
                records.store<Word>(z, i, 0);
                records.store<Word>(top + y0, i, 1);
                records.store<Word>(top + y1, i, 2);
                records.store<Word>(left + x0, i, 3);
                records.store<Word>(left + x1, i, 4);
                records.store<Word>(stats.load<Word>(level, 4), i, 5);
                records.store<Word>(0, i, 6);  // Each writer receives its own payload subview.
                records.store<Word>(size, i, 7);
                cursor += size;
                boundaries.store<Word>(cursor, i + 1);
            }
        }
        return Py_BuildValue("NNN", fields.release(), offsets.release(), payload.release());
    });
}

PyObject* changed_scores(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *old_arg, *new_arg;
        if (!PyArg_ParseTuple(args, "OO", &old_arg, &new_arg)) throw PythonError{};
        Buffer original(old_arg, "original scores", 1), current(new_arg, "current scores", 1);
        original.require<Byte>();
        current.require<Byte>();
        equal_shape(original, current);
        int largest_dropped = 0, smallest_kept = 256;
        bool mismatch = false;
        {
            GilRelease unlocked;
            for (Index k = 0; k < original.shape(0); ++k) {
                const int before = original.load<Byte>(k), after = current.load<Byte>(k);
                if (after) {
                    smallest_kept = std::min(smallest_kept, after);
                    mismatch |= before != after;
                } else {
                    largest_dropped = std::max(largest_dropped, before);
                }
            }
        }
        return Py_BuildValue("iiO", largest_dropped, smallest_kept, mismatch ? Py_True : Py_False);
    });
}

PyObject* merge_masked_crop(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *plane_arg, *touched_arg, *mask_arg, *scores_arg;
        Index y0, x0, block;
        if (!PyArg_ParseTuple(args, "OOOOnnn", &plane_arg, &touched_arg, &mask_arg, &scores_arg,
                              &y0, &x0, &block)) throw PythonError{};
        Buffer plane(plane_arg, "score plane", 2, true), touched(touched_arg, "touched cells", 2, true);
        Buffer mask(mask_arg, "mask", 2), scores(scores_arg, "scores", 2);
        plane.require<Byte>();
        touched.require<Byte>();
        require_mask(mask);
        scores.require<Byte>();
        equal_shape(mask, scores);
        const Index rows = mask.shape(0), columns = mask.shape(1);
        const Index y1 = checked_add(y0, rows), x1 = checked_add(x0, columns);
        require_rect(plane, y0, y1, x0, x1);
        if (block <= 0) value_error("Confidence block size must be positive");
        const Index cell_y = y0 / block, cell_x = x0 / block;
        const Index stop_y = y1 ? (y1 - 1) / block + 1 : 0;
        const Index stop_x = x1 ? (x1 - 1) / block + 1 : 0;
        if (cell_y < stop_y && cell_x < stop_x) require_rect(touched, cell_y, stop_y, cell_x, stop_x);
        {
            GilRelease unlocked;
            if (columns) {
                const bool unit_stride = plane.stride(1) == 1 && mask.stride(1) == 1 && scores.stride(1) == 1;
                for (Index y = 0; y < rows; ++y) {
                    if (unit_stride) {
                        const auto* support = reinterpret_cast<const Byte*>(mask.ptr(y, 0));
                        const auto* source = reinterpret_cast<const Byte*>(scores.ptr(y, 0));
                        auto* target = reinterpret_cast<Byte*>(plane.ptr(y0 + y, x0));
                        for (Index x = 0; x < columns; ++x) {
                            const Byte value = source[x] & static_cast<Byte>(0 - (support[x] != 0));
                            target[x] = std::max(value, target[x]);
                        }
                    } else {
                        for (Index x = 0; x < columns; ++x) {
                            const Byte value = mask.load<Byte>(y, x) ? scores.load<Byte>(y, x) : 0;
                            plane.store<Byte>(std::max(value, plane.load<Byte>(y0 + y, x0 + x)), y0 + y, x0 + x);
                        }
                    }
                }
            }
            for (Index y = cell_y; y < stop_y; ++y)
                for (Index x = cell_x; x < stop_x; ++x)
                    touched.store<Byte>(1, y, x);
        }
        Py_RETURN_NONE;
    });
}

PyObject* take_frame_cells(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *plane_arg, *touched_arg;
        Index block;
        if (!PyArg_ParseTuple(args, "OOn", &plane_arg, &touched_arg, &block)) throw PythonError{};
        Buffer plane(plane_arg, "score plane", 2, true), touched(touched_arg, "touched cells", 2, true);
        plane.require<Byte>();
        touched.require<Byte>();
        if (block <= 0) value_error("Confidence block size must be positive");
        const Index h = plane.shape(0), w = plane.shape(1);
        const Index rows = touched.shape(0), columns = touched.shape(1);
        checked_extent(rows, block);
        checked_extent(columns, block);
        Index marked = 0;
        {
            GilRelease unlocked;
            for (Index row = 0; row < rows; ++row)
                for (Index col = 0; col < columns; ++col)
                    marked += touched.load<Byte>(row, col) != 0;
        }
        Object table = empty({marked, 6}, "int64");
        Buffer records(table.get(), "frame crop records", 2, true);
        Index count = 0, total = 0;
        {
            GilRelease unlocked;
            for (Index row = 0; row < rows; ++row) {
                for (Index col = 0; col < columns; ++col) {
                    if (!touched.load<Byte>(row, col)) continue;
                    touched.store<Byte>(0, row, col);
                    const Index a = row * block, b = std::min(a + block, h);
                    const Index c = col * block, d = std::min(c + block, w);
                    Index top = b, bottom = a, left = d, right = c, known = 0;
                    for (Index y = a; y < b; ++y) {
                        for (Index x = c; x < d; ++x) {
                            if (plane.load<Byte>(y, x)) {
                                top = std::min(top, y);
                                bottom = y + 1;
                                left = std::min(left, x);
                                right = std::max(right, x + 1);
                                ++known;
                            }
                        }
                    }
                    if (!known) continue;
                    if (count == marked) throw std::invalid_argument("Confidence cell marks changed during scan");
                    records.store<Word>(top, count, 0);
                    records.store<Word>(left, count, 1);
                    records.store<Word>(bottom - top, count, 2);
                    records.store<Word>(right - left, count, 3);
                    records.store<Word>(known, count, 4);
                    records.store<Word>(total, count++, 5);
                    total += (bottom - top) * (right - left);
                }
            }
        }
        Object raw = empty({total}, "uint8");
        Buffer packed(raw.get(), "raw score cells", 1, true);
        {
            GilRelease unlocked;
            for (Index i = 0; i < count; ++i) {
                const Index y0 = records.load<Word>(i, 0), x0 = records.load<Word>(i, 1);
                const Index height = records.load<Word>(i, 2), width = records.load<Word>(i, 3);
                Index start = records.load<Word>(i, 5);
                for (Index y = y0; y < y0 + height; ++y) {
                    if (plane.stride(1) == 1) {
                        auto* source = plane.ptr(y, x0);
                        std::memcpy(packed.ptr(start), source, width);
                        std::memset(source, 0, width);
                        start += width;
                    } else {
                        for (Index x = x0; x < x0 + width; ++x) {
                            packed.store<Byte>(plane.load<Byte>(y, x), start++);
                            plane.store<Byte>(0, y, x);
                        }
                    }
                }
            }
        }
        Object used_table = prefix(table, count);
        return Py_BuildValue("NN", used_table.release(), raw.release());
    });
}

unsigned long cell_bound(Index size) {
    const auto bound = checked_add(size, (size >> 12) + (size >> 14) + (size >> 25) + 13);
    if (static_cast<std::uint64_t>(bound) > ULONG_MAX) {
        throw std::overflow_error("Confidence cell exceeds zlib's ABI limit");
    }
    return static_cast<unsigned long>(bound);
}

std::pair<Index, Index> compressed_cell(const Buffer& raw, const Buffer& table, Index item) {
    const Index height = table.load<Word>(item, 2), width = table.load<Word>(item, 3);
    const Index size = checked_extent(height, width), start = table.load<Word>(item, 5);
    if (width < 0 || !in_range(start, checked_add(start, size), raw.shape(0))) {
        throw std::invalid_argument("Confidence cell exceeds its raw buffer");
    }
    return {start, size};
}

PyObject* sweep_compress_cells(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *pointer_arg, *raw_arg, *table_arg;
        if (!PyArg_ParseTuple(args, "OOO", &pointer_arg, &raw_arg, &table_arg)) throw PythonError{};
        using Compress = int (*)(Byte*, unsigned long*, const Byte*, unsigned long, int);
        auto compress = reinterpret_cast<Compress>(PyLong_AsVoidPtr(pointer_arg));
        if (PyErr_Occurred()) throw PythonError{};
        if (!compress) value_error("Missing qualified confidence compressor");
        Buffer raw(raw_arg, "raw score cells", 1), table(table_arg, "frame crop records", 2);
        raw.require<Byte>().require_contiguous();
        table.require<Word>();
        if (table.shape(1) != 6) value_error("Invalid confidence cell index");
        const Index count = table.shape(0);
        Index capacity = 0;
        {
            GilRelease unlocked;
            for (Index i = 0; i < count; ++i) {
                capacity = checked_add(capacity, cell_bound(compressed_cell(raw, table, i).second));
            }
        }
        Object payload = empty({std::max<Index>(1, capacity)}, "uint8");
        Object sizes = zeros({count}, "int64");
        Buffer output(payload.get(), "compressed cells", 1, true), lengths(sizes.get(), "cell lengths", 1, true);
        Index done = 0, used = 0;
        int status = 0;
        {
            GilRelease unlocked;
            for (; done < count; ++done) {
                const auto [start, size] = compressed_cell(raw, table, done);
                const unsigned long limit = cell_bound(size);
                if (limit > static_cast<std::uint64_t>(capacity - used)) {
                    throw std::invalid_argument("Confidence cells exceed the allocated compression bound");
                }
                unsigned long written = limit;
                const Byte zero = 0;
                const auto* input = size ? reinterpret_cast<const Byte*>(raw.ptr(start)) : &zero;
                status = compress(reinterpret_cast<Byte*>(output.ptr(used)), &written, input,
                                  static_cast<unsigned long>(size), 3);
                if (status) break;
                if (written > limit) throw std::invalid_argument("Confidence compressor exceeded its output bound");
                lengths.store<Word>(written, done);
                used += written;
            }
        }
        Object used_payload = prefix(payload, used);
        return Py_BuildValue("NNni", used_payload.release(), sizes.release(), done, status);
    });
}

PyObject* restore_d1_crop(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *source_arg, *output_arg, *ys_arg, *xs_arg;
        Index z0, z1, y0, x0;
        if (!PyArg_ParseTuple(args, "OOnnOOnn", &source_arg, &output_arg, &z0, &z1, &ys_arg, &xs_arg, &y0, &x0)) {
            throw PythonError{};
        }
        Buffer projected(source_arg, "projected ranks", 3), output(output_arg, "restored ranks", 2, true);
        Buffer ys(ys_arg, "source rows", 1), xs(xs_arg, "source columns", 1);
        projected.require<Byte>();
        output.require<Byte>();
        const bool rows_wide = ys.is<Word>(), columns_wide = xs.is<Word>();
        if (!rows_wide) ys.require<std::int32_t>();
        if (!columns_wide) xs.require<std::int32_t>();
        if (!in_range(z0, z1, projected.shape(0))
            || !in_range(y0, checked_add(y0, output.shape(0)), ys.shape(0))
            || !in_range(x0, checked_add(x0, output.shape(1)), xs.shape(0))) {
            value_error("D1 restore crop lies outside its plan");
        }
        {
            GilRelease unlocked;
            for (Index row = 0; row < output.shape(0); ++row) {
                const Index y = rows_wide ? ys.load<Word>(y0 + row) : ys.load<std::int32_t>(y0 + row);
                if (z0 < z1 && (y < 0 || y >= projected.shape(1))) {
                    throw std::invalid_argument("D1 restore row lies outside the source");
                }
                for (Index col = 0; col < output.shape(1); ++col) {
                    const Index x = columns_wide ? xs.load<Word>(x0 + col) : xs.load<std::int32_t>(x0 + col);
                    if (z0 < z1 && (x < 0 || x >= projected.shape(2))) {
                        throw std::invalid_argument("D1 restore column lies outside the source");
                    }
                    Byte best = 0;
                    for (Index z = z0; z < z1; ++z) best = std::max(best, projected.load<Byte>(z, y, x));
                    output.store<Byte>(best, row, col);
                }
            }
        }
        Py_RETURN_NONE;
    });
}

std::pair<double, double> contributor(Index position, Index working, Index output, double scale) {
    if (output < working) {
        const auto [first, stop] = coverage_geometry::axis_bounds(position, working, output);
        return {static_cast<double>(first), static_cast<double>(stop - 1)};
    }
    const double value = (position + .5) * scale - .5;
    return {value, value};
}

// Exact preimage of a closed interval under rational contraction or
// centered expansion. Recheck the floating-point boundaries as the pull does.
std::pair<Index, Index> preimage(double lower, double upper, Index working, Index output, double scale) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    if (std::isnan(lower)) lower = -inf;
    if (std::isnan(upper)) upper = inf;
    if (lower > upper) return {0, 0};
    if (output < working) {
        if (lower > working - 1 || upper < 0) return {0, 0};
        const Index first = lower <= 0 ? 0 : static_cast<Index>(std::ceil(lower));
        const Index last = upper >= working - 1 ? working - 1 : static_cast<Index>(std::floor(upper));
        if (first > last) return {0, 0};
        return {checked_extent(first, output) / working,
                (checked_extent(last + 1, output) - 1) / working + 1};
    }
    const double smallest = .5 * scale - .5, largest = (output - .5) * scale - .5;
    if (lower > largest || upper < smallest) return {0, 0};
    Index first = 0, stop = output;
    if (lower > smallest) {
        const double estimate = std::ceil((lower + .5) / scale - .5);
        first = estimate >= output ? output : estimate <= 0 ? 0 : static_cast<Index>(estimate);
        while (first > 0 && (first - 1 + .5) * scale - .5 >= lower) --first;
        while (first < output && (first + .5) * scale - .5 < lower) ++first;
    }
    if (upper < largest) {
        const double estimate = std::floor((upper + .5) / scale - .5);
        Index last = estimate >= output - 1 ? output - 1 : estimate <= -1 ? -1 : static_cast<Index>(estimate);
        while (last < output - 1 && (last + 1 + .5) * scale - .5 <= upper) ++last;
        while (last >= 0 && (last + .5) * scale - .5 > upper) --last;
        stop = last + 1;
    }
    return {first, std::max(first, stop)};
}

void require_axis(Index working, Index out, double scale) {
    if (working <= 0 || out <= 0 || (out >= working && (!std::isfinite(scale) || scale <= 0)))
        value_error("Invalid contributor geometry");
}

PyObject* contributor_hull(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        Index position, working, output;
        double scale;
        if (!PyArg_ParseTuple(args, "nnnd", &position, &working, &output, &scale)) throw PythonError{};
        require_axis(working, output, scale);
        if (position < 0 || position >= output) value_error("Contributor index lies outside the output");
        const auto [lower, upper] = contributor(position, working, output, scale);
        return Py_BuildValue("dd", lower, upper);
    });
}

PyObject* output_index_range(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        Index working, output;
        double lower, upper, scale;
        if (!PyArg_ParseTuple(args, "ddnnd", &lower, &upper, &working, &output, &scale)) throw PythonError{};
        require_axis(working, output, scale);
        const auto [first, stop] = preimage(lower, upper, working, output, scale);
        return Py_BuildValue("nn", first, stop);
    });
}

void update_tree(const Buffer& trees, Index kind, Index leaves, Index first, Index stop, double value) {
    first += leaves;
    stop += leaves;
    const auto update = [&](Index i) {
        const auto old = trees.load<double>(kind, i);
        trees.store<double>(kind & 1 ? std::max(old, value) : std::min(old, value), kind, i);
    };
    while (first < stop) {
        if (first & 1) update(first++);
        if (stop & 1) update(--stop);
        first >>= 1;
        stop >>= 1;
    }
}

void require_rectangle(const coverage::Geometry& g) {
    if (!in_range(g.bbox[0], g.bbox[3], g.ot) || !in_range(g.bbox[1], g.bbox[4], g.oh)
        || !in_range(g.bbox[2], g.bbox[5], g.ow)) value_error("Rank rectangle lies outside the output");
}

std::array<Index, 4> slice_spans(Index z, const Buffer& frames, const Buffer& regions,
                                const coverage::Geometry& g, double margin, const Buffer& spans,
                                const Buffer& trees, Index leaves) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    const auto [s_lo, s_hi] = contributor(z, g.wt, g.ot, g.c[9]);
    const double tangent = g.c[0], origin = g.c[1], center = g.c[2];
    for (Index row = 0; row < 4; ++row)
        for (Index node = 0; node < trees.shape(1); ++node)
            trees.store<double>(row & 1 ? -inf : inf, row, node);
    for (Index y = 0; y < spans.shape(0); ++y) {
        spans.store<Word>(0, y, 0);
        spans.store<Word>(0, y, 1);
    }

    double slope = 0;
    if (!g.vertical && tangent != 0.) {
        if (g.base == 1) slope = 1. / tangent;
        else if (g.base == 2) slope = tangent;
    }
    bool touched = false;
    for (Index i = 0; i < frames.shape(0); ++i) {
        const double low = frames.load<Word>(i) + origin - .5, high = frames.load<Word>(i) + origin + .5;
        const double v_lo = regions.load<double>(i, 0), v_hi = regions.load<double>(i, 1);
        const double u_lo = regions.load<double>(i, 2), u_hi = regions.load<double>(i, 3);
        double y_lo = v_lo, y_hi = v_hi, x_lo = u_lo, x_hi = u_hi;
        double band_lo = -inf, band_hi = inf;
        if (g.base == 0) {
            if (tangent == 0) {
                if (high < s_lo - margin || low > s_hi + margin) continue;
            } else {
                const double p = (s_lo - high) / tangent, q = (s_hi - low) / tangent;
                auto& lower = g.vertical ? y_lo : x_lo;
                auto& upper = g.vertical ? y_hi : x_hi;
                lower = std::max(lower, center + std::min(p, q) - margin);
                upper = std::min(upper, center + std::max(p, q) + margin);
            }
        } else {
            const double gate_lo = std::max(v_lo, s_lo - margin), gate_hi = std::min(v_hi, s_hi + margin);
            if (gate_lo > gate_hi) continue;
            if (g.vertical) {
                const double p = tangent * (gate_lo - center), q = tangent * (gate_hi - center);
                const double stack_lo = low + std::min(p, q), stack_hi = high + std::max(p, q);
                if (g.base == 1) {
                    y_lo = stack_lo;
                    y_hi = stack_hi;
                } else {
                    y_lo = u_lo;
                    y_hi = u_hi;
                    x_lo = stack_lo;
                    x_hi = stack_hi;
                }
            } else if (g.base == 1) {
                const double p = tangent * (u_lo - center), q = tangent * (u_hi - center);
                y_lo = low + std::min(p, q);
                y_hi = high + std::max(p, q);
                if (tangent != 0) {
                    const double left = center - low / tangent, right = center - high / tangent;
                    band_lo = std::min(left, right);
                    band_hi = std::max(left, right);
                }
            } else {
                y_lo = u_lo;
                y_hi = u_hi;
                x_lo = -inf;
                x_hi = inf;
                band_lo = low - tangent * center;
                band_hi = high - tangent * center;
            }
        }
        if (y_lo - margin > y_hi + margin || x_lo - margin > x_hi + margin) continue;
        auto [first, stop] = preimage(y_lo - margin, y_hi + margin, g.wh, g.oh, g.c[10]);
        first = std::max<Index>(first, g.bbox[1]);
        stop = std::min<Index>(stop, g.bbox[4]);
        if (first >= stop) continue;
        touched = true;
        update_tree(trees, 0, leaves, first, stop, x_lo - margin);
        update_tree(trees, 1, leaves, first, stop, x_hi + margin);
        update_tree(trees, 2, leaves, first, stop, band_lo - margin);
        update_tree(trees, 3, leaves, first, stop, band_hi + margin);
    }
    if (!touched) return {};
    for (Index node = 1; node < leaves; ++node) {
        for (Index kind = 0; kind < 4; ++kind) {
            const double value = trees.load<double>(kind, node);
            for (Index child : {2 * node, 2 * node + 1}) {
                const double old = trees.load<double>(kind, child);
                trees.store<double>(kind & 1 ? std::max(old, value) : std::min(old, value), kind, child);
            }
        }
    }
    Index y0 = g.oh, y1 = 0, x0 = g.ow, x1 = 0;
    for (Index y = g.bbox[1]; y < g.bbox[4]; ++y) {
        double lower = trees.load<double>(0, leaves + y), upper = trees.load<double>(1, leaves + y);
        if (lower > upper) continue;
        const auto [w_lo, w_hi] = contributor(y, g.wh, g.oh, g.c[10]);
        const double p = slope * w_lo, q = slope * w_hi;
        lower = std::max(lower, std::min(p, q) + trees.load<double>(2, leaves + y) - margin);
        upper = std::min(upper, std::max(p, q) + trees.load<double>(3, leaves + y) + margin);
        auto [first, stop] = preimage(lower, upper, g.ww, g.ow, g.c[11]);
        first = std::max<Index>(first, g.bbox[2]);
        stop = std::min<Index>(stop, g.bbox[5]);
        if (first >= stop) continue;
        spans.store<Word>(first, y, 0);
        spans.store<Word>(stop, y, 1);
        y0 = std::min(y0, y);
        y1 = y + 1;
        x0 = std::min(x0, first);
        x1 = std::max(x1, stop);
    }
    return y0 < y1 ? std::array<Index, 4>{y0, y1, x0, x1} : std::array<Index, 4>{};
}

PyObject* tilted_slice_spans(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *frame_arg, *regions_arg, *int_arg, *coeff_arg, *box_arg, *span_arg, *tree_arg;
        Index z;
        double margin;
        if (!PyArg_ParseTuple(args, "nOOOOdOOO", &z, &frame_arg, &regions_arg, &int_arg, &coeff_arg,
                              &margin, &box_arg, &span_arg, &tree_arg)) throw PythonError{};
        Buffer frames(frame_arg, "active source frames", 1), regions(regions_arg, "frame regions", 2);
        Buffer integers(int_arg, "geometry integers", 1), coefficients(coeff_arg, "geometry coefficients", 1);
        Buffer bbox(box_arg, "geometry bounds", 1), spans(span_arg, "output spans", 2, true);
        Buffer trees(tree_arg, "span search trees", 2, true);
        const coverage::Geometry g(integers, coefficients, bbox);
        require_rectangle(g);
        require_axis(g.wt, g.ot, g.c[9]);
        require_axis(g.wh, g.oh, g.c[10]);
        require_axis(g.ww, g.ow, g.c[11]);
        frames.require<Word>();
        regions.require<double>();
        spans.require<Word>();
        trees.require<double>();
        const Index leaves = trees.shape(1) / 2;
        if (z < 0 || z >= g.ot || regions.shape(0) != frames.shape(0) || regions.shape(1) != 4
            || spans.shape(0) < g.oh || spans.shape(1) != 2
            || trees.shape(0) != 4 || leaves < g.oh || (leaves & (leaves - 1)))
            value_error("Invalid Tilted span workspace");
        std::array<Index, 4> result;
        {
            GilRelease unlocked;
            result = slice_spans(z, frames, regions, g, margin, spans, trees, leaves);
        }
        return Py_BuildValue("nnnn", result[0], result[1], result[2], result[3]);
    });
}

PyObject* tilted_span_rows(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *source_arg, *out_arg, *span_arg, *int_arg, *coeff_arg, *box_arg;
        Index z, y0, x0;
        if (!PyArg_ParseTuple(args, "OOnnnOOOO", &source_arg, &out_arg, &z, &y0, &x0, &span_arg,
                              &int_arg, &coeff_arg, &box_arg)) throw PythonError{};
        Buffer source(source_arg, "rank source", 3), out(out_arg, "rank crop", 2, true);
        Buffer spans(span_arg, "output spans", 2), integers(int_arg, "geometry integers", 1);
        Buffer coefficients(coeff_arg, "geometry coefficients", 1), bbox(box_arg, "geometry bounds", 1);
        source.require<Byte>();
        out.require<Byte>();
        spans.require<Word>();
        const coverage::Geometry g(integers, coefficients, bbox);
        require_rectangle(g);
        const Index y1 = checked_add(y0, out.shape(0)), x1 = checked_add(x0, out.shape(1));
        if (spans.shape(1) != 2 || z < 0 || z >= g.ot || !in_range(y0, y1, std::min<Index>(g.oh, spans.shape(0)))
            || !in_range(x0, x1, g.ow)) value_error("Tilted spans lie outside the output");
        coverage::Stats stats;
        {
            GilRelease unlocked;
            for (Index y = y0; y < y1; ++y) {
                const Index first = spans.load<Word>(y, 0), stop = spans.load<Word>(y, 1);
                if (first >= stop) continue;
                if (first < x0 || stop > x1) throw std::invalid_argument("Tilted span lies outside the crop");
                coverage::dense_tilted(g, source, out.ptr(y - y0, first - x0), out.stride(1),
                                       stop - first, z * g.plane + y * g.ow + first, true, stats);
            }
        }
        return Py_BuildValue("LL", static_cast<long long>(stats.visited), static_cast<long long>(stats.addresses));
    });
}

template<bool Azimuthal>
PyObject* rank_rows(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *source_arg, *out_arg, *box_arg, *int_arg, *coeff_arg;
        PyObject *keys_arg = nullptr, *invalid_arg = nullptr, *rows_arg = nullptr;
        Index first, z;
        int cached = 0;
        if constexpr (Azimuthal) {
            if (!PyArg_ParseTuple(args, "OOnnOOOOpOO", &source_arg, &out_arg, &first, &z, &box_arg,
                                  &int_arg, &coeff_arg, &keys_arg, &cached, &invalid_arg, &rows_arg))
                throw PythonError{};
        } else {
            if (!PyArg_ParseTuple(args, "OOnnOOO", &source_arg, &out_arg, &first, &z, &box_arg,
                                  &int_arg, &coeff_arg)) throw PythonError{};
        }
        Buffer source(source_arg, "rank source", 3), out(out_arg, "rank row", 1, true);
        Buffer bbox(box_arg, "geometry bounds", 1), integers(int_arg, "geometry integers", 1);
        Buffer coefficients(coeff_arg, "geometry coefficients", 1);
        source.require<Byte>();
        out.require<Byte>();
        const coverage::Geometry g(integers, coefficients, bbox);
        require_rectangle(g);
        const Index rw = g.bbox[5] - g.bbox[2], size = out.shape(0);
        if (z < 0 || z >= g.ot || !in_range(first, checked_add(first, size), checked_extent(g.bbox[4] - g.bbox[1], rw)))
            value_error("Rank rows lie outside the output rectangle");
        coverage::Stats stats;
        const auto fill = [&](const auto& pull) {
            GilRelease unlocked;
            for (Index pos = 0; pos < size;) {
                const Index relative = first + pos;
                const Index y = g.bbox[1] + relative / rw, x = g.bbox[2] + relative % rw;
                const Index take = std::min<Index>(size - pos, g.bbox[5] - x);
                pull(out.ptr(pos), pos, take, z * g.plane + y * g.ow + x);
                pos += take;
            }
        };
        if constexpr (Azimuthal) {
            Buffer keys(keys_arg, "angular keys", 1), rows(rows_arg, "azimuthal row lookup", 1);
            Object invalid_index(PyNumber_Index(invalid_arg));
            const auto invalid = PyLong_AsUnsignedLongLong(invalid_index.get());
            if (PyErr_Occurred()) throw PythonError{};
            fill([&](char* dest, Index pos, Index count, Index absolute) {
                coverage::dense_azimuthal(g, source, dest, out.stride(0), count, absolute, true,
                                          keys, cached ? 0 : pos, cached, invalid, rows, stats);
            });
        } else {
            fill([&](char* dest, Index, Index count, Index absolute) {
                coverage::dense_tilted(g, source, dest, out.stride(0), count, absolute, true, stats);
            });
        }
        return PyLong_FromLongLong(stats.addresses);
    });
}

PyMethodDef methods[] = {
    {"masked_cell_bounds", masked_cell_bounds, METH_VARARGS, "Find globally ordered score cell bounds."},
    {"copy_masked_cell", copy_masked_cell, METH_VARARGS, "Copy one bounded masked cell."},
    {"copy_masked_cells_batch", copy_masked_cells_batch, METH_VARARGS, "Copy bounded cells and offsets."},
    {"zero_unmasked_scores_flat", zero_unmasked_scores_flat, METH_VARARGS, "Mask a flat score range."},
    {"zero_unmasked_scores_boxes", zero_unmasked_scores_boxes, METH_VARARGS, "Mask selected score boxes."},
    {"mask_eager_scores_and_bound", mask_eager_scores_and_bound, METH_VARARGS, "Mask scores and bound the result."},
    {"merge_rank_crop", merge_rank_crop, METH_VARARGS, "Merge a cutoff mask into ranks."},
    {"cutoff_extents", cutoff_extents, METH_VARARGS, "Measure cumulative cutoff boxes and counts."},
    {"pack_rank_level", pack_rank_level, METH_VARARGS, "Pack a cutoff's rank mask."},
    {"pack_rank_levels", pack_rank_levels, METH_VARARGS, "Pack selected rank levels into one bounded payload."},
    {"changed_scores", changed_scores, METH_VARARGS, "Compare two score packets."},
    {"merge_masked_crop", merge_masked_crop, METH_VARARGS, "Max-merge masked scores."},
    {"take_frame_cells", take_frame_cells, METH_VARARGS, "Take and clear tight score cells in grid order."},
    {"sweep_compress_cells", sweep_compress_cells, METH_VARARGS, "Compress a frame's exact score cells."},
    {"restore_d1_crop", restore_d1_crop, METH_VARARGS, "Restore a rational D1 rank crop."},
    {"contributor_hull", contributor_hull, METH_VARARGS, "Compute exact one-axis contributor bounds."},
    {"output_index_range", output_index_range, METH_VARARGS, "Compute a contributor interval preimage."},
    {"tilted_slice_spans", tilted_slice_spans, METH_VARARGS, "Certify Tilted rank spans."},
    {"tilted_span_rows", tilted_span_rows, METH_VARARGS, "Pull ranks on certified Tilted spans."},
    {"tilted_rank_rows", rank_rows<false>, METH_VARARGS, "Pull ranks on rectangular Tilted rows."},
    {"azimuthal_rank_rows", rank_rows<true>, METH_VARARGS, "Pull ranks on rectangular Azimuthal rows."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_confidence(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
