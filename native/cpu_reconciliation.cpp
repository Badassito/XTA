#include "cpu_kernels.hpp"

#include <algorithm>
#include <optional>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Count = std::uint16_t;

bool same_vote_shape(const Buffer& values, const Buffer& reference) noexcept {
    for (int axis = 0; axis < 4; ++axis) {
        if (values.shape(axis) != reference.shape(axis)) return false;
    }
    return true;
}

Index tile_count(Index dimension, Index tile) noexcept {
    return dimension / tile + (dimension % tile != 0);
}

#if defined(_M_X64) || defined(__SSE2__)
template<class Lane>
Lane load_lane(const char* input) noexcept {
    Lane value;
    std::memcpy(&value, input, sizeof(value));
    return value;
}

template<class Lane>
void store_lane(char* output, Lane value) noexcept {
    std::memcpy(output, &value, sizeof(value));
}

void add_bool_votes(char* count, const char* values) noexcept {
    auto bytes = _mm_setzero_si128();
    std::memcpy(&bytes, values, 8);
    const auto bounded = _mm_min_epu8(bytes, _mm_set1_epi8(1));
    const auto units = _mm_unpacklo_epi8(bounded, _mm_setzero_si128());
    store_lane(count, _mm_add_epi16(load_lane<__m128i>(count), units));
}

Index reduce_contiguous_row(const char* scores, const char* direct, const char* extrapolated,
                            char* support, char* direct_support, char* total, char* extra_support,
                            Index count) noexcept {
    Index x = 0;
    const auto zero = _mm_setzero_ps();
    for (; x <= count - 8; x += 8) {
        const auto first = load_lane<__m128>(scores + x * sizeof(float));
        const auto last = load_lane<__m128>(scores + (x + 4) * sizeof(float));
        const auto members = _mm_packs_epi32(_mm_castps_si128(_mm_cmpgt_ps(first, zero)),
                                            _mm_castps_si128(_mm_cmpgt_ps(last, zero)));
        auto* count_row = support + x * sizeof(Count);
        store_lane(count_row, _mm_sub_epi16(load_lane<__m128i>(count_row), members));
        add_bool_votes(direct_support + x * sizeof(Count), direct + x);
        auto* sum_first = total + x * sizeof(float);
        auto* sum_last = sum_first + 4 * sizeof(float);
        store_lane(sum_first, _mm_add_ps(load_lane<__m128>(sum_first), first));
        store_lane(sum_last, _mm_add_ps(load_lane<__m128>(sum_last), last));
        if (extrapolated) add_bool_votes(extra_support + x * sizeof(Count), extrapolated + x);
    }
    return x;
}
#endif

void reduce_tiles(const Buffer& scores, const Buffer& direct, const Buffer& tiles,
                  const Buffer* extrapolated, Index tile_size, const Buffer& support,
                  const Buffer& direct_support, const Buffer& total, const Buffer* extra_support) {
    const Index height = scores.shape(2), width = scores.shape(3);
#if defined(_M_X64) || defined(__SSE2__)
    const bool contiguous = scores.stride(3) == sizeof(float) && direct.stride(3) == 1
        && support.stride(2) == sizeof(Count) && direct_support.stride(2) == sizeof(Count)
        && total.stride(2) == sizeof(float)
        && (!extrapolated || (extrapolated->stride(3) == 1 && extra_support->stride(2) == sizeof(Count)));
#endif
    // Group order is the same as the strided-axis float32 summation. Keep
    // this order for edge tiles as well, even when just one voxel is covered.
    for (Index group = 0; group < scores.shape(0); ++group) {
        for (Index z = 0; z < scores.shape(1); ++z) {
            for (Index ty = 0; ty < tiles.shape(2); ++ty) {
                const Index top = ty * tile_size;
                const Index bottom = top + std::min(tile_size, height - top);
                Index tx = 0;
                while (tx < tiles.shape(3)) {
                    if (!tiles.load<bool>(group, z, ty, tx)) {
                        ++tx;
                        continue;
                    }
                    const Index first = tx++;
                    while (tx < tiles.shape(3) && tiles.load<bool>(group, z, ty, tx)) ++tx;
                    const Index left = first * tile_size;
                    const Index last = (tx - 1) * tile_size;
                    const Index right = last + std::min(tile_size, width - last);
                    for (Index y = top; y < bottom; ++y) {
                        Index start = left;
#if defined(_M_X64) || defined(__SSE2__)
                        if (contiguous) {
                            start += reduce_contiguous_row(scores.ptr(group, z, y, left),
                                direct.ptr(group, z, y, left),
                                extrapolated ? extrapolated->ptr(group, z, y, left) : nullptr,
                                support.ptr(z, y, left), direct_support.ptr(z, y, left),
                                total.ptr(z, y, left), extra_support ? extra_support->ptr(z, y, left) : nullptr,
                                right - left);
                        }
#endif
                        for (Index x = start; x < right; ++x) {
                            const float score = scores.load<float>(group, z, y, x);
                            support.store<Count>(support.load<Count>(z, y, x) + (score > 0.f), z, y, x);
                            direct_support.store<Count>(direct_support.load<Count>(z, y, x)
                                + direct.load<bool>(group, z, y, x), z, y, x);
                            total.store<float>(total.load<float>(z, y, x) + score, z, y, x);
                            if (extrapolated) {
                                extra_support->store<Count>(extra_support->load<Count>(z, y, x)
                                    + extrapolated->load<bool>(group, z, y, x), z, y, x);
                            }
                        }
                    }
                }
            }
        }
    }
}

PyObject* reduce_vote_groups(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *score_object, *direct_object, *extra_object, *tile_object;
        Index tile_size;
        if (!PyArg_ParseTuple(args, "OOOOn:reconciliation_reduce_vote_groups",
                              &score_object, &direct_object, &extra_object, &tile_object, &tile_size)) {
            return nullptr;
        }
        const Buffer scores(score_object, "reconciliation group scores", 4);
        const Buffer direct(direct_object, "reconciliation direct support", 4);
        const Buffer tiles(tile_object, "reconciliation occupied tiles", 4);
        scores.require<float>();
        direct.require<bool>();
        tiles.require<bool>();
        if (tile_size <= 0 || !same_vote_shape(direct, scores)
            || tiles.shape(0) != scores.shape(0) || tiles.shape(1) != scores.shape(1)
            || tiles.shape(2) != tile_count(scores.shape(2), tile_size)
            || tiles.shape(3) != tile_count(scores.shape(3), tile_size)) {
            value_error("Reconciliation tile grid differs from its score volume");
        }
        std::optional<Buffer> extrapolated;
        if (extra_object != Py_None) {
            extrapolated.emplace(extra_object, "reconciliation extrapolation support", 4);
            extrapolated->require<bool>();
            if (!same_vote_shape(*extrapolated, scores))
                value_error("Reconciliation extrapolation support differs from its score volume");
        }
        const auto shape = {scores.shape(1), scores.shape(2), scores.shape(3)};
        auto support_array = zeros(shape, "uint16");
        auto direct_array = zeros(shape, "uint16");
        auto total_array = zeros(shape, "float32");
        auto extra_array = extrapolated ? zeros(shape, "uint16") : Object();
        const Buffer support(support_array.get(), "reconciliation support counts", 3, true);
        const Buffer direct_support(direct_array.get(), "reconciliation direct counts", 3, true);
        const Buffer total(total_array.get(), "reconciliation total scores", 3, true);
        std::optional<Buffer> extra_support;
        if (extra_array.get()) extra_support.emplace(extra_array.get(), "reconciliation extrapolation counts", 3, true);
        {
            GilRelease release;
            reduce_tiles(scores, direct, tiles, extrapolated ? &*extrapolated : nullptr, tile_size,
                         support, direct_support, total, extra_support ? &*extra_support : nullptr);
        }
        return PyTuple_Pack(4, support_array.get(), direct_array.get(), total_array.get(),
                            extra_array.get() ? extra_array.get() : Py_None);
    });
}

PyMethodDef methods[] = {
    {"reconciliation_reduce_vote_groups", reduce_vote_groups, METH_VARARGS,
     "Reduce covered source-grid votes in group order."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_reconciliation(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
