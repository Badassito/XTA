#include "cpu_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Byte = std::uint8_t;
using Label = std::uint32_t;
using Code = std::uint64_t;
using Signed = std::int64_t;

PyObject* arg(PyObject* args, Index index) noexcept { return PyTuple_GET_ITEM(args, index); }

void arguments(PyObject* args, Index count) {
    if (PyTuple_GET_SIZE(args) != count) type_error("Incorrect topology kernel argument count");
}

Index index_argument(PyObject* args, Index i) {
    Object number(PyNumber_Index(arg(args, i)));
    const Index value = PyLong_AsSsize_t(number.get());
    if (PyErr_Occurred()) throw PythonError{};
    return value;
}

Code code_argument(PyObject* args, Index i) {
    Object number(PyNumber_Index(arg(args, i)));
    const auto value = PyLong_AsUnsignedLongLong(number.get());
    if (PyErr_Occurred()) throw PythonError{};
    return static_cast<Code>(value);
}

void same_shape(const Buffer& a, const Buffer& b) {
    for (int axis = 0; axis < a.ndim(); ++axis) {
        if (a.shape(axis) != b.shape(axis)) value_error("Topology buffer shapes differ");
    }
}

bool in_range(Signed index, Index length) noexcept {
    return index >= 0 && index < length;
}

enum class LabelType { narrow, wide, signed_wide };

LabelType label_type(const Buffer& buffer) {
    if (buffer.is<std::uint16_t>()) return LabelType::narrow;
    if (buffer.is<Label>()) return LabelType::wide;
    if (buffer.is<std::int32_t>()) return LabelType::signed_wide;
    type_error("Topology labels must be uint16, uint32 or int32");
}

template<class Function>
decltype(auto) with_labels(LabelType type, Function&& function) {
    switch (type) {
    case LabelType::narrow: return function(std::uint16_t{});
    case LabelType::wide: return function(Label{});
    default: return function(std::int32_t{});
    }
}

bool flag_type(const Buffer& buffer) {
    if (buffer.is<bool>()) return true;
    buffer.require<Byte>();
    return false;
}

template<class... Indices>
Byte read_flag(const Buffer& buffer, bool is_bool, Indices... indices) noexcept {
    return is_bool ? static_cast<Byte>(buffer.load<bool>(indices...)) : buffer.load<Byte>(indices...);
}

template<class... Indices>
void write_flag(const Buffer& buffer, bool is_bool, Byte value, Indices... indices) noexcept {
    if (is_bool) buffer.store<bool>(value != 0, indices...);
    else buffer.store<Byte>(value, indices...);
}

Index find_halving(const Buffer& parent, Signed node) {
    const Index size = parent.shape(0);
    for (Index step = 0; step < size && in_range(node, size); ++step) {
        const Signed next = parent.load<Signed>(node);
        if (next == node) return static_cast<Index>(node);
        if (!in_range(next, size)) break;
        const Signed grandparent = parent.load<Signed>(next);
        if (!in_range(grandparent, size)) break;
        parent.store<Signed>(grandparent, node);
        node = grandparent;
    }
    throw std::invalid_argument("Topology union-find contains an invalid parent or member");
}

std::pair<std::uintptr_t, std::uintptr_t> address_span(const Buffer& buffer) noexcept {
    const auto base = reinterpret_cast<std::uintptr_t>(buffer.data());
    auto lo = base, hi = base;
    if (buffer.nbytes()) {
        hi += buffer.itemsize();
        for (int axis = 0; axis < buffer.ndim(); ++axis) {
            const Index span = (buffer.shape(axis) - 1) * buffer.stride(axis);
            if (span < 0) lo += span;
            else hi += span;
        }
    }
    return {lo, hi};
}

bool spans_overlap(const Buffer& a, const Buffer& b) noexcept {
    if (!a.nbytes() || !b.nbytes()) return false;
    const auto [alo, ahi] = address_span(a);
    const auto [blo, bhi] = address_span(b);
    return alo < bhi && blo < ahi;
}

Signed union_id_argument(PyObject* object) {
    Object number(PyNumber_Index(object));
    const auto value = PyLong_AsLongLong(number.get());
    if (PyErr_Occurred()) throw PythonError{};
    constexpr Signed maximum = std::numeric_limits<Label>::max();
    if (value < -maximum || value > maximum) value_error("Topology union-find ID adjustment exceeds uint32");
    return value;
}

std::pair<Signed, Signed> union_id_pair(PyObject* argument, bool allow_scalar) {
    if (PyTuple_Check(argument)) {
        if (PyTuple_GET_SIZE(argument) != 2) value_error("Expected two topology union-find ID values");
        return {union_id_argument(PyTuple_GET_ITEM(argument, 0)),
                union_id_argument(PyTuple_GET_ITEM(argument, 1))};
    }
    if (!allow_scalar) type_error("Expected a pair of topology component counts");
    const Signed value = union_id_argument(argument);
    return {value, value};
}

PyObject* union_find_batch(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        const Index count = PyTuple_GET_SIZE(args);
        if (count < 4 || count > 6) type_error("Incorrect topology union-find argument count");
        const Buffer parent(arg(args, 0), "union-find parents", 1, true);
        const Buffer rank(arg(args, 1), "union-find ranks", 1, true);
        const Buffer touches(arg(args, 2), "union-find boundaries", 1, true);
        const Buffer codes(arg(args, 3), "union-find pair codes", 1);
        parent.require<Signed>();
        rank.require<std::int32_t>();
        touches.require<bool>();
        codes.require<Code>();
        same_shape(parent, rank);
        same_shape(parent, touches);
        const auto [base_a, base_b] = count >= 5 ? union_id_pair(arg(args, 4), true)
                                                  : std::pair<Signed, Signed>{0, 0};
        const bool validate_members = count == 6 && arg(args, 5) != Py_None;
        const auto [limit_a, limit_b] = validate_members ? union_id_pair(arg(args, 5), false)
                                                         : std::pair<Signed, Signed>{0, 0};
        if (limit_a < 0 || limit_b < 0) value_error("Negative topology component count");

        // Preserve the former split arrays' snapshot guarantee if the forest
        // could modify a later packed pair in its own backing store.
        std::vector<Code> snapshot;
        const Index size = codes.shape(0);
        if (spans_overlap(codes, parent) || spans_overlap(codes, rank) || spans_overlap(codes, touches)) {
            snapshot.reserve(static_cast<std::size_t>(size));
            for (Index i = 0; i < size; ++i) snapshot.push_back(codes.load<Code>(i));
        }
        const auto read_code = [&](Index i) {
            return snapshot.empty() ? codes.load<Code>(i) : snapshot[i];
        };
        int invalid_member = 0;
        {
            GilRelease release;
            if (validate_members) {
                // Shard metadata is checked for the whole batch before any
                // mutation, preserving high-member errors ahead of low ones.
                for (Index i = 0; i < size; ++i) {
                    const Code code = read_code(i);
                    const auto left = static_cast<Signed>(code >> 32);
                    const auto right = static_cast<Label>(code);
                    if (left == 0 || left > limit_a) {
                        invalid_member = 1;
                        break;
                    }
                    if (right == 0 || right > limit_b) invalid_member = 2;
                }
            }
            for (Index i = 0; !invalid_member && i < size; ++i) {
                const Code code = read_code(i);
                Index ra = find_halving(parent, static_cast<Signed>(code >> 32) - base_a);
                Index rb = find_halving(parent, static_cast<Signed>(static_cast<Label>(code)) - base_b);
                if (ra == rb) continue;
                auto rank_a = rank.load<std::int32_t>(ra), rank_b = rank.load<std::int32_t>(rb);
                if (rank_a < rank_b) {
                    std::swap(ra, rb);
                    std::swap(rank_a, rank_b);
                }
                if (rank_a == rank_b && rank_a == std::numeric_limits<std::int32_t>::max()) {
                    throw std::overflow_error("Topology union-find rank overflow");
                }
                parent.store<Signed>(ra, rb);
                if (touches.load<bool>(rb)) touches.store<bool>(true, ra);
                if (rank_a == rank_b) rank.store<std::int32_t>(rank_a + 1, ra);
            }
        }
        if (invalid_member) return PyLong_FromLong(invalid_member);
        Py_RETURN_NONE;
    });
}

Code hash_slot(Code code, Code mask) noexcept {
    code = (code ^ (code >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    code = (code ^ (code >> 27)) * UINT64_C(0x94d049bb133111eb);
    return (code ^ (code >> 31)) & mask;
}

Index table_mask(const Buffer& table) {
    table.require<Code>();
    const Index size = table.shape(0);
    if (size < 2 || (size & (size - 1))) value_error("Invalid topology hash table capacity");
    return size - 1;
}

struct ScanState {
    Index used;
    Index position;
    Index neighbor;
    Code last;
};

template<class A, class B>
ScanState scan_hash(const Buffer& prev, const Buffer& curr, const Buffer& offsets,
                    Code prev_offset, Code curr_offset, const Buffer& table, Index mask,
                    ScanState state, Index total) {
    const Index width = prev.shape(1), height = prev.shape(0), threshold = table.shape(0) / 2;
    if (state.position == total) return {state.used, total, 0, state.last};
    Index y = state.position / width, x = state.position % width;
    for (; y < height; ++y, x = 0) {
        for (; x < width; ++x, state.neighbor = 0) {
            const auto value = prev.load<A>(y, x);
            if (value <= 0) continue;
            const Code high = (static_cast<Code>(value) + prev_offset) << 32;
            for (; state.neighbor < offsets.shape(0);) {
                const Index n = state.neighbor++;
                const Signed dy = offsets.load<Signed>(n, 0), dx = offsets.load<Signed>(n, 1);
                if (dy < -y || dy >= height - y || dx < -x || dx >= width - x) continue;
                const auto other = curr.load<B>(y + dy, x + dx);
                if (other <= 0) continue;
                const Code code = high | (static_cast<Code>(other) + curr_offset);
                if (code == state.last) continue;
                state.last = code;
                if (code == 0) continue;
                auto slot = static_cast<Index>(hash_slot(code, mask));
                for (Index probed = 0;; ++probed) {
                    const Code present = table.load<Code>(slot);
                    if (present == code) break;
                    if (present == 0) {
                        table.store<Code>(code, slot);
                        if (++state.used >= threshold) {
                            return {state.used, y * width + x, state.neighbor, state.last};
                        }
                        break;
                    }
                    if (probed == mask) throw std::invalid_argument("Topology hash table is full");
                    slot = (slot + 1) & mask;
                }
            }
        }
    }
    return {state.used, total, 0, state.last};
}

PyObject* adjacency_scan(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 10);
        const Buffer prev(arg(args, 0), "previous labels", 2);
        const Buffer curr(arg(args, 1), "current labels", 2);
        const Buffer offsets(arg(args, 2), "adjacency offsets", 2);
        const auto dtype_a = label_type(prev), dtype_b = label_type(curr);
        const Code prev_offset = code_argument(args, 3), curr_offset = code_argument(args, 4);
        const Buffer table(arg(args, 5), "adjacency hash", 1, true);
        const Index mask = table_mask(table), total = checked_extent(prev.shape(0), prev.shape(1));
        ScanState state{index_argument(args, 6), index_argument(args, 7), index_argument(args, 8),
                        code_argument(args, 9)};
        offsets.require<Signed>();
        same_shape(prev, curr);
        if (offsets.shape(1) != 2 || !in_range(state.used, table.shape(0))
            || state.position < 0 || state.position > total
            || state.neighbor < 0 || state.neighbor > offsets.shape(0)) {
            value_error("Invalid topology adjacency scan cursor");
        }
        {
            GilRelease release;
            state = with_labels(dtype_a, [&](auto a) {
                return with_labels(dtype_b, [&](auto b) {
                    return scan_hash<decltype(a), decltype(b)>(prev, curr, offsets, prev_offset,
                        curr_offset, table, mask, state, total);
                });
            });
        }
        return Py_BuildValue("(nnnK)", state.used, state.position, state.neighbor,
                             static_cast<unsigned long long>(state.last));
    });
}

PyObject* adjacency_rehash(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 2);
        const Buffer source(arg(args, 0), "previous adjacency hash", 1);
        const Buffer destination(arg(args, 1), "enlarged adjacency hash", 1, true);
        source.require<Code>();
        const Index mask = table_mask(destination);
        {
            GilRelease release;
            for (Index i = 0; i < source.shape(0); ++i) {
                const Code code = source.load<Code>(i);
                if (!code) continue;
                auto slot = static_cast<Index>(hash_slot(code, mask));
                for (Index probed = 0; destination.load<Code>(slot) != 0; ++probed) {
                    if (probed == mask) throw std::invalid_argument("Enlarged topology hash table is full");
                    slot = (slot + 1) & mask;
                }
                destination.store<Code>(code, slot);
            }
        }
        Py_RETURN_NONE;
    });
}

PyObject* adjacency_extract(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 2);
        const Buffer table(arg(args, 0), "adjacency hash", 1);
        table.require<Code>();
        const Index count = index_argument(args, 1);
        if (count < 0 || count > table.shape(0)) value_error("Invalid topology adjacency count");
        auto result = empty({count}, "uint64");
        const Buffer output(result.get(), "adjacency keys", 1, true);
        Index index = 0;
        {
            GilRelease release;
            for (Index i = 0; i < table.shape(0); ++i) {
                const Code code = table.load<Code>(i);
                if (!code) continue;
                if (index == count) throw std::invalid_argument("Topology adjacency count is too small");
                output.store<Code>(code, index++);
            }
        }
        if (index != count) value_error("Topology adjacency count exceeds occupied slots");
        return result.release();
    });
}

template<class Emit>
Index merge_keys(const Buffer& a, const Buffer& b, Emit emit) {
    Index i = 0, j = 0, at = 0;
    while (i < a.shape(0) && j < b.shape(0)) {
        const Code left = a.load<Code>(i), right = b.load<Code>(j);
        emit(at++, std::min(left, right));
        i += left <= right;
        j += right <= left;
    }
    while (i < a.shape(0)) emit(at++, a.load<Code>(i++));
    while (j < b.shape(0)) emit(at++, b.load<Code>(j++));
    return at;
}

PyObject* adjacency_merge(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 2);
        const Buffer a(arg(args, 0), "first adjacency keys", 1);
        const Buffer b(arg(args, 1), "second adjacency keys", 1);
        a.require<Code>();
        b.require<Code>();
        if (!a.shape(0)) return Py_NewRef(arg(args, 1));
        if (!b.shape(0)) return Py_NewRef(arg(args, 0));
        checked_add(a.shape(0), b.shape(0));
        Index count;
        {
            GilRelease release;
            count = merge_keys(a, b, [](Index, Code) {});
        }
        auto result = empty({count}, "uint64");
        const Buffer output(result.get(), "merged adjacency keys", 1, true);
        {
            GilRelease release;
            const Index written = merge_keys(a, b, [&](Index at, Code code) {
                if (at >= count) {
                    throw std::invalid_argument("Topology adjacency key count changed during merge");
                }
                output.store<Code>(code, at);
            });
            if (written != count) {
                throw std::invalid_argument("Topology adjacency key count changed during merge");
            }
        }
        return result.release();
    });
}

template<class T>
Index write_label_runs(const Buffer& plane, const Buffer& runs, const Buffer& rows) noexcept {
    Index count = 0;
    for (Index y = 0; y < plane.shape(0); ++y) {
        rows.store<Signed>(count, y);
        for (Index x = 0; x < plane.shape(1);) {
            const auto value = plane.load<T>(y, x);
            const Index first = x++;
            if (value <= 0) continue;
            while (x < plane.shape(1) && plane.load<T>(y, x) == value) ++x;
            if (count == runs.shape(0)) return -1;
            runs.store<Label>(static_cast<Label>(first), count, 0);
            runs.store<Label>(static_cast<Label>(x), count, 1);
            runs.store<Label>(static_cast<Label>(value), count++, 2);
        }
    }
    rows.store<Signed>(count, plane.shape(0));
    return count;
}

PyObject* topology_label_runs(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 2);
        const Buffer plane(arg(args, 0), "run labels", 2);
        const auto dtype = label_type(plane);
        const Index capacity = index_argument(args, 1);
        if (capacity < 0 || static_cast<Code>(plane.shape(1)) > std::numeric_limits<Label>::max()) {
            value_error("Invalid topology label-run geometry");
        }
        auto runs = empty({capacity, 3}, "uint32");
        auto rows = empty({checked_add(plane.shape(0), 1)}, "int64");
        const Buffer run_data(runs.get(), "label runs", 2, true);
        const Buffer row_data(rows.get(), "run row offsets", 1, true);
        Index count;
        {
            GilRelease release;
            count = with_labels(dtype, [&](auto label) {
                return write_label_runs<decltype(label)>(plane, run_data, row_data);
            });
        }
        return Py_BuildValue("(OOn)", runs.get(), rows.get(), count);
    });
}

void validate_run_offsets(const Buffer& offsets, Index count) {
    Signed previous = 0;
    for (Index i = 0; i < offsets.shape(0); ++i) {
        const auto value = offsets.load<Signed>(i);
        if (value < previous || value > count) throw std::invalid_argument("Invalid topology run row offsets");
        previous = value;
    }
}

Index run_intersections(const Buffer& a, const Buffer& ar, const Buffer& b, const Buffer& br,
                        const Buffer& groups, Code ao, Code bo, const Buffer& codes) {
    const Index height = ar.shape(0) - 1;
    Index count = 0;
    Code last = 0;
    for (Index group = 0; group < groups.shape(0); ++group) {
        const Signed dy = groups.load<Signed>(group, 0), dx0 = groups.load<Signed>(group, 1),
            dx1 = groups.load<Signed>(group, 2);
        if (dx0 > dx1 || dx1 > std::numeric_limits<Signed>::max() - std::numeric_limits<Label>::max()) {
            throw std::invalid_argument("Invalid topology run shift");
        }
        for (Index y = 0; y < height; ++y) {
            if (dy < -y || dy >= height - y) continue;
            const Index cy = y + dy;
            Index first_b = br.load<Signed>(cy);
            const Index stop_b = br.load<Signed>(cy + 1), stop_a = ar.load<Signed>(y + 1);
            for (Index i = ar.load<Signed>(y); i < stop_a; ++i) {
                const Signed left = static_cast<Signed>(a.load<Label>(i, 0)) + dx0;
                const Signed right = static_cast<Signed>(a.load<Label>(i, 1)) + dx1;
                while (first_b < stop_b && b.load<Label>(first_b, 1) <= left) ++first_b;
                for (Index j = first_b; j < stop_b && b.load<Label>(j, 0) < right; ++j) {
                    const Code code = ((static_cast<Code>(a.load<Label>(i, 2)) + ao) << 32)
                        | (static_cast<Code>(b.load<Label>(j, 2)) + bo);
                    if (code == last) continue;
                    if (count == codes.shape(0)) return -1;
                    codes.store<Code>(code, count++);
                    last = code;
                }
            }
        }
    }
    return count;
}

PyObject* topology_run_intersections(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 8);
        const Buffer a(arg(args, 0), "first label runs", 2);
        const Buffer ar(arg(args, 1), "first run row offsets", 1);
        const Buffer b(arg(args, 2), "second label runs", 2);
        const Buffer br(arg(args, 3), "second run row offsets", 1);
        const Buffer groups(arg(args, 4), "adjacent run shifts", 2);
        a.require<Label>();
        b.require<Label>();
        ar.require<Signed>();
        br.require<Signed>();
        groups.require<Signed>();
        same_shape(ar, br);
        const Code ao = code_argument(args, 5), bo = code_argument(args, 6);
        const Index capacity = index_argument(args, 7);
        if (!ar.shape(0) || a.shape(1) != 3 || b.shape(1) != 3 || groups.shape(1) != 3 || capacity < 0) {
            value_error("Invalid topology run intersection geometry");
        }
        auto result = empty({capacity}, "uint64");
        const Buffer output(result.get(), "run intersection keys", 1, true);
        Index count;
        {
            GilRelease release;
            validate_run_offsets(ar, a.shape(0));
            validate_run_offsets(br, b.shape(0));
            count = run_intersections(a, ar, b, br, groups, ao, bo, output);
        }
        return Py_BuildValue("(On)", result.get(), count);
    });
}

PyObject* void_fill_slab(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 3);
        const Buffer mask(arg(args, 0), "void-fill mask", 3, true);
        const Buffer labels(arg(args, 1), "void-fill labels", 3);
        const Buffer lookup(arg(args, 2), "void-fill lookup", 1);
        const bool mask_bool = flag_type(mask), lookup_bool = flag_type(lookup);
        const auto dtype = label_type(labels);
        same_shape(mask, labels);
        {
            GilRelease release;
            with_labels(dtype, [&](auto label) {
                for (Index z = 0; z < mask.shape(0); ++z) {
                    for (Index y = 0; y < mask.shape(1); ++y) {
                        for (Index x = 0; x < mask.shape(2); ++x) {
                            const Signed v = labels.load<decltype(label)>(z, y, x);
                            if (!in_range(v, lookup.shape(0))) {
                                throw std::invalid_argument("Void-fill label exceeds topology lookup");
                            }
                            if (read_flag(lookup, lookup_bool, v)) write_flag(mask, mask_bool, 1, z, y, x);
                        }
                    }
                }
            });
        }
        Py_RETURN_NONE;
    });
}

struct SliceWork {
    Index z, y0, y1, x0, x1, lookup, label, max_label;
};

// A sufficient nonoverlap test admitting C/F order and reversed axes. A
// deliberately overlapping or unusual view uses serial writes.
bool disjoint_elements(const Buffer& output) noexcept {
    std::array<std::pair<Code, Code>, 3> axes{};
    for (int i = 0; i < 3; ++i) {
        const Index stride = output.stride(i);
        const Code magnitude = stride < 0 ? Code{0} - static_cast<Code>(stride) : static_cast<Code>(stride);
        axes[i] = {magnitude, static_cast<Code>(output.shape(i))};
    }
    std::sort(axes.begin(), axes.end());
    Code span = output.itemsize();
    for (const auto& [stride, dimension] : axes) {
        if (dimension <= 1) continue;
        if (stride < span) return false;
        span += (dimension - 1) * stride;
    }
    return true;
}

template<class Apply>
void apply_slices(const std::vector<SliceWork>& work, Index requested, Apply apply) {
    const auto count = static_cast<Index>(work.size());
    const auto lanes = std::min(count, requested);
    if (lanes <= 1) {
        for (const auto& slice : work) {
            if (!apply(slice)) throw std::invalid_argument("Topology label exceeds its lookup");
        }
        return;
    }
    // One caller-owned batch. Threads borrow only pinned buffers and join
    // before any view can be released, including after thread creation fails.
    std::atomic<bool> valid{true};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(lanes - 1));
    const auto run = [&](Index lane) {
        const Index first = lane * (count / lanes) + std::min(lane, count % lanes);
        const Index stop = first + count / lanes + (lane < count % lanes);
        for (Index i = first; i < stop && valid.load(std::memory_order_relaxed); ++i) {
            if (!apply(work[i])) {
                valid.store(false, std::memory_order_relaxed);
                break;
            }
        }
    };
    try {
        for (Index lane = 1; lane < lanes; ++lane) threads.emplace_back(run, lane);
        run(0);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();
    if (!valid) throw std::invalid_argument("Topology label exceeds its lookup");
}

std::vector<SliceWork> plan_slices(const Buffer& output, const Buffer& lookup, const Buffer& offsets,
                                  const Buffer& boxes, const Buffer& selected, bool compact,
                                  const Buffer* arena, const Buffer* label_offsets) {
    const Index count = output.shape(0);
    offsets.require<Signed>();
    boxes.require<Signed>();
    bool flag_bool = false;
    if (compact) selected.require<Label>();
    else flag_bool = flag_type(selected);
    if (label_offsets) label_offsets->require<Signed>();
    if (offsets.shape(0) != count || boxes.shape(0) != count || boxes.shape(1) != 4
        || selected.shape(0) != count || (label_offsets && label_offsets->shape(0) < count)) {
        value_error("Invalid topology slice lookup geometry");
    }
    std::vector<SliceWork> work;
    for (Index z = 0; z < count; ++z) {
        const Index active = compact ? selected.load<Label>(z) : read_flag(selected, flag_bool, z);
        if (!active) continue;
        const Signed y0 = boxes.load<Signed>(z, 0), y1 = boxes.load<Signed>(z, 1);
        const Signed x0 = boxes.load<Signed>(z, 2), x1 = boxes.load<Signed>(z, 3);
        const Signed off = offsets.load<Signed>(z);
        if (y0 < 0 || y1 < y0 || y1 > output.shape(1) || x0 < 0 || x1 < x0 || x1 > output.shape(2)) {
            value_error("Invalid topology slice bounds");
        }
        if (y0 == y1 || x0 == x1) continue;
        if (!in_range(off, lookup.shape(0)) || (compact && active >= lookup.shape(0) - off)) {
            value_error("Invalid topology slice lookup offset");
        }
        Index label = 0;
        if (arena) {
            const auto area = checked_extent(y1 - y0, x1 - x0);
            label = label_offsets->load<Signed>(z);
            if (label < 0 || label > arena->shape(0) || area > arena->shape(0) - label) {
                value_error("Invalid sparse topology label span");
            }
        }
        work.push_back({z, static_cast<Index>(y0), static_cast<Index>(y1),
                        static_cast<Index>(x0), static_cast<Index>(x1), static_cast<Index>(off),
                        label, compact ? active : lookup.shape(0) - off - 1});
    }
    return work;
}

template<class T>
bool relabel_slice(const Buffer& labels, const Buffer& lookup, const SliceWork& slice) noexcept {
    for (Index y = slice.y0; y < slice.y1; ++y) {
        for (Index x = slice.x0; x < slice.x1; ++x) {
            const Signed v = labels.load<T>(slice.z, y, x);
            if (!v) continue;
            if (v < 0 || v > slice.max_label) return false;
            labels.store<T>(static_cast<T>(lookup.load<Label>(slice.lookup + v)), slice.z, y, x);
        }
    }
    return true;
}

PyObject* compact_relabel(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 6);
        const Buffer labels(arg(args, 0), "relabel labels", 3, true);
        const Buffer lookup(arg(args, 1), "compact lookup", 1);
        const Buffer offsets(arg(args, 2), "compact offsets", 1);
        const Buffer boxes(arg(args, 3), "compact boxes", 2);
        const Buffer counts(arg(args, 4), "compact counts", 1);
        const auto dtype = label_type(labels);
        lookup.require<Label>();
        const Index workers = index_argument(args, 5);
        if (workers <= 0) value_error("Topology worker count must be positive");
        const auto work = plan_slices(labels, lookup, offsets, boxes, counts, true, nullptr, nullptr);
        const bool concurrent = disjoint_elements(labels) && !spans_overlap(labels, lookup);
        {
            GilRelease release;
            with_labels(dtype, [&](auto label) {
                apply_slices(work, concurrent ? workers : 1, [&](const SliceWork& slice) {
                    return relabel_slice<decltype(label)>(labels, lookup, slice);
                });
            });
        }
        Py_RETURN_NONE;
    });
}

template<class T, bool Sparse>
bool keep_slice(const Buffer& labels, const Buffer& lookup, bool lookup_bool, const Buffer& output,
                bool output_bool, const SliceWork& slice) noexcept {
    Index at = slice.label;
    for (Index y = slice.y0; y < slice.y1; ++y) {
        for (Index x = slice.x0; x < slice.x1; ++x) {
            Signed v;
            if constexpr (Sparse) v = labels.load<T>(at++);
            else v = labels.load<T>(slice.z, y, x);
            if (v < 0 || v > slice.max_label) return false;
            const Byte value = read_flag(lookup, lookup_bool, slice.lookup + v);
            write_flag(output, output_bool, value, slice.z, y, x);
        }
    }
    return true;
}

template<bool Sparse>
PyObject* keep_lut_apply(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        constexpr Index start = Sparse ? 2 : 1;
        arguments(args, start + 6);
        const Buffer labels(arg(args, 0), "keep labels", Sparse ? 1 : 3);
        const Buffer lookup(arg(args, start), "keep lookup", 1);
        const Buffer offsets(arg(args, start + 1), "keep lookup offsets", 1);
        const Buffer boxes(arg(args, start + 2), "keep boxes", 2);
        const Buffer selected(arg(args, start + 3), "keep slice flags", 1);
        const Buffer output(arg(args, start + 4), "kept mask", 3, true);
        const auto dtype = label_type(labels);
        const bool lookup_bool = flag_type(lookup), output_bool = flag_type(output);
        const Index workers = index_argument(args, start + 5);
        if (workers <= 0) value_error("Topology worker count must be positive");
        std::vector<SliceWork> work;
        if constexpr (Sparse) {
            const Buffer label_offsets(arg(args, 1), "sparse label offsets", 1);
            work = plan_slices(output, lookup, offsets, boxes, selected, false, &labels, &label_offsets);
        } else {
            same_shape(labels, output);
            work = plan_slices(output, lookup, offsets, boxes, selected, false, nullptr, nullptr);
        }
        const bool concurrent = disjoint_elements(output) && !spans_overlap(output, labels)
                                && !spans_overlap(output, lookup);
        {
            GilRelease release;
            with_labels(dtype, [&](auto label) {
                apply_slices(work, concurrent ? workers : 1, [&](const SliceWork& slice) {
                    return keep_slice<decltype(label), Sparse>(labels, lookup, lookup_bool, output, output_bool, slice);
                });
            });
        }
        Py_RETURN_NONE;
    });
}

PyMethodDef methods[] = {
    {"topology_union_find_batch", union_find_batch, METH_VARARGS, "Merge packed topology pairs in input order."},
    {"topology_adjacency_scan", adjacency_scan, METH_VARARGS, "Scan topology adjacency up to a bounded hash spill."},
    {"topology_adjacency_rehash", adjacency_rehash, METH_VARARGS, "Move existing topology keys into a larger hash."},
    {"topology_adjacency_extract", adjacency_extract, METH_VARARGS, "Extract occupied topology hash keys."},
    {"topology_adjacency_merge", adjacency_merge, METH_VARARGS, "Merge sorted topology key batches."},
    {"topology_label_runs", topology_label_runs, METH_VARARGS, "Extract bounded positive label runs."},
    {"topology_run_intersections", topology_run_intersections, METH_VARARGS, "Intersect shifted topology label runs."},
    {"topology_void_fill_slab", void_fill_slab, METH_VARARGS, "Apply the void lookup to a topology slab."},
    {"topology_compact_relabel", compact_relabel, METH_VARARGS, "Apply compact-label lookups by slice."},
    {"topology_keep_lut_apply", keep_lut_apply<false>, METH_VARARGS, "Apply keep lookups from dense label slices."},
    {"topology_sparse_keep_lut_apply", keep_lut_apply<true>, METH_VARARGS, "Apply keep lookups from sparse label slices."},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_topology(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
