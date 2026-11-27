// Packed sweep-journal decoding with exact zlib consumption and ordered rank merge.
#include "cpu_kernels.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Word = std::int64_t;
using Byte = std::uint8_t;
using Uncompress = int (*)(Byte*, unsigned long*, const Byte*, unsigned long*);

constexpr Index header_bytes = 40;
enum Status { ok = 0, bad_payload = -1, short_header = -2, bad_geometry = -3,
              bad_lengths = -4, unindexed = -5, bad_stream = -6, bad_plane = -7 };

struct Row {
    Word offset, length, chunks, top, bottom, left, right, height, width, scores, limit;
};

struct Chunk {
    Index at, y0, y1, x0, x1, mask_bytes, score_bytes, mask_raw, score_raw;
    Index packed_start = 0;
};

Uncompress inflater_address(PyObject* object) {
    void* address = PyLong_AsVoidPtr(object);
    if (PyErr_Occurred()) throw PythonError{};
    if (address == nullptr) value_error("Native zlib uncompress2 has no entry address");
    return reinterpret_cast<Uncompress>(address);
}

template<class Get>
Row read_row(Get get) {
    return {get(0), get(1), get(2), get(3), get(4), get(5),
            get(6), get(7), get(8), get(9), get(10)};
}

bool fits(const Row& row, Index payload_bytes) {
    return row.offset >= 0 && row.offset <= payload_bytes && row.length >= 0
        && row.length <= payload_bytes - row.offset && row.chunks >= 0
        && row.chunks <= row.length / header_bytes;
}

Word read_word(const Buffer& payload, Index offset) {
    return Word(payload.load<Byte>(offset)) | (Word(payload.load<Byte>(offset+1)) << 8)
        | (Word(payload.load<Byte>(offset+2)) << 16) | (Word(payload.load<Byte>(offset+3)) << 24);
}

Status scan_row(const Buffer& payload, const Row& row, std::vector<Chunk>& headers, Index base) {
    Index consumed = 0;
    for (Index i = 0; i < row.chunks; ++i) {
        if (row.length - consumed < header_bytes) return short_header;
        const Index at = static_cast<Index>(row.offset) + consumed;
        const Word y0 = read_word(payload, at), y1 = read_word(payload, at+4);
        const Word x0 = read_word(payload, at+8), x1 = read_word(payload, at+12);
        const Word ml = read_word(payload, at+16), sl = read_word(payload, at+20);
        const Word mr = read_word(payload, at+32), sr = read_word(payload, at+36);
        if (row.top > y0 || y0 >= y1 || y1 > row.bottom || row.bottom > row.height
            || row.left > x0 || x0 >= x1 || x1 > row.right || row.right > row.width
            || row.limit < 1 || y1-y0 > row.limit / (x1-x0)) return bad_geometry;
        const Word area = (y1-y0)*(x1-x0);
        if (mr != area/8 + (area%8 != 0) || sr != (row.scores ? area : 0)) return bad_geometry;
        if (ml > mr+1024 || sl > sr+1024 || (sl != 0) != (row.scores != 0)
            || header_bytes+ml+sl > row.length-consumed) return bad_lengths;
        if (area > PY_SSIZE_T_MAX || mr > PY_SSIZE_T_MAX || y1 > PY_SSIZE_T_MAX || x1 > PY_SSIZE_T_MAX)
            return bad_geometry;
        headers[base+i] = {at, Index(y0), Index(y1), Index(x0), Index(x1), Index(ml), Index(sl),
                           Index(mr), Index(sr)};
        consumed += header_bytes + static_cast<Index>(ml+sl);
    }
    return consumed == row.length ? ok : unindexed;
}

// The qualified provider remains owned by the calling decoder. Stack lengths
// use the library's C unsigned long, including its 32-bit width on Windows.
bool inflate_stream(Uncompress inflate, const Buffer& payload, Index offset, Index size,
                    Byte* output, Index raw, std::vector<Byte>& strided_input) {
    if (size <= 0 || static_cast<std::uintmax_t>(size) > ULONG_MAX
        || static_cast<std::uintmax_t>(raw) >= ULONG_MAX) return false;
    const Byte* input = reinterpret_cast<const Byte*>(payload.ptr(offset));
    if (payload.stride(0) != 1) {
        strided_input.resize(size);
        for (Index i = 0; i < size; ++i) strided_input[i] = payload.load<Byte>(offset+i);
        input = strided_input.data();
    }
    unsigned long written = static_cast<unsigned long>(raw)+1;
    unsigned long consumed = static_cast<unsigned long>(size);
    return inflate(output, &written, input, &consumed) == 0
        && written == static_cast<unsigned long>(raw) && consumed == static_cast<unsigned long>(size);
}

constexpr auto unpacked_bytes() {
    std::array<std::array<Byte, 8>, 256> values{};
    for (unsigned value = 0; value < 256; ++value)
        for (unsigned bit = 0; bit < 8; ++bit) values[value][bit] = Byte((value >> bit) & 1);
    return values;
}
constexpr auto bits = unpacked_bytes();

PyObject* decode_row(PyObject*, PyObject* args) {
    PyObject *address, *payload_object, *params_object;
    int include_scores;
    if (!PyArg_ParseTuple(args, "OOOp:journal_decode_row", &address, &payload_object,
                          &params_object, &include_scores)) return nullptr;
    return protect([&]() -> PyObject* {
        const auto inflate = inflater_address(address);
        Buffer payload(payload_object, "payload", 1), params(params_object, "row parameters", 1);
        payload.require<Byte>();
        params.require<Word>();
        if (params.shape(0) < 11) value_error("Incomplete journal row parameters");
        const Row row = read_row([&](Index col) { return params.load<Word>(col); });
        Status status = fits(row, payload.shape(0)) ? ok : bad_payload;
        const Index count = status == ok ? static_cast<Index>(row.chunks) : 0;
        std::vector<Chunk> headers(count);
        auto bounds = empty({count, 4}, "int64"), starts = zeros({count}, "int64");
        Buffer boxes(bounds.get(), "bounds", 2, true), positions(starts.get(), "starts", 1, true);
        const auto failure = [&]() {
            auto masks = empty({0}, "uint8"), scores = empty({0}, "uint8");
            return Py_BuildValue("(iOOOOn)", int(status), bounds.get(), starts.get(),
                                 masks.get(), scores.get(), Index(0));
        };
        Index slots = 0, largest = 1;
        if (status == ok) {
            GilRelease unlocked;
            status = scan_row(payload, row, headers, 0);
            if (status == ok) {
                for (Index i = 0; i < count; ++i) {
                    positions.store<Word>(slots, i);
                    slots = checked_add(slots, checked_extent(headers[i].mask_raw, 8));
                    largest = std::max(largest, checked_add(headers[i].mask_raw, 1));
                }
            }
        }
        if (status != ok) return failure();
        const bool read_scores = include_scores && row.scores != 0;
        auto masks = empty({slots}, "uint8");
        auto scores = empty({read_scores ? checked_add(slots, 1) : 0}, "uint8");
        Buffer mask_data(masks.get(), "masks", 1, true), score_data(scores.get(), "scores", 1, true);
        Index read = 0;
        {
            GilRelease unlocked;
            std::vector<Byte> scratch(largest), strided_input;
            for (Index i = 0; i < count; ++i) {
                const auto& h = headers[i];
                const Index at = h.at + header_bytes, start = positions.load<Word>(i);
                if (!inflate_stream(inflate, payload, at, h.mask_bytes, scratch.data(), h.mask_raw,
                                    strided_input)) {
                    status = bad_stream;
                    break;
                }
                for (Index byte = 0; byte < h.mask_raw; ++byte)
                    std::memcpy(mask_data.ptr(start+8*byte), bits[scratch[byte]].data(), 8);
                if (read_scores && !inflate_stream(inflate, payload, at+h.mask_bytes, h.score_bytes,
                        reinterpret_cast<Byte*>(score_data.ptr(start)), h.score_raw, strided_input)) {
                    status = bad_stream;
                    break;
                }
                boxes.store<Word>(h.y0, i, 0);
                boxes.store<Word>(h.y1, i, 1);
                boxes.store<Word>(h.x0, i, 2);
                boxes.store<Word>(h.x1, i, 3);
                read += header_bytes + h.mask_bytes + (include_scores ? h.score_bytes : 0);
            }
        }
        if (status != ok) return failure();
        return Py_BuildValue("(iOOOOn)", int(ok), bounds.get(), starts.get(),
                             masks.get(), scores.get(), read);
    });
}

struct Box {
    Index top, bottom, left, right;
};

// Merge only after every stream of every row has passed validation; otherwise
// the Python reader can resume without any partial mutation of the caller's plane.
void merge_chunk(const Chunk& h, const Byte* packed, const Buffer& plane, Word level,
                 bool check, bool& populated, bool& gap) {
    Index bit = 0;
    const Word previous = level == std::numeric_limits<Word>::min()
        ? std::numeric_limits<Word>::max() : level-1;
    for (Index y = h.y0; y < h.y1; ++y) {
        for (Index x = h.x0; x < h.x1;) {
            const Index shift = bit & 7;
            const Index take = std::min<Index>(8-shift, h.x1-x);
            const unsigned ones = (1u << take)-1;
            const unsigned value = (packed[bit >> 3] >> shift) & ones;
            if (value) {
                populated = true;
                for (Index k = 0; k < take; ++k) {
                    if (value != ones && !((value >> k) & 1)) continue;
                    if (check && plane.load<Byte>(y, x+k) < previous) gap = true;
                    plane.store<Byte>(static_cast<Byte>(level), y, x+k);
                }
            }
            bit += take;
            x += take;
        }
    }
}

template<class RowAt, class RankAt>
PyObject* merge_rows(Uncompress inflate, const Buffer& payload, const Buffer& plane, Index rows,
                     RowAt row_at, RankAt rank_at) {
    auto report = zeros({rows, 3}, "int64");
    Buffer outcome(report.get(), "rank outcome", 2, true);
    Box box{plane.shape(0), 0, plane.shape(1), 0};
    Status status = ok;
    {
        GilRelease unlocked;
        std::vector<Index> first(checked_add(rows, 1), 0);
        std::vector<Row> parameters(rows);
        for (Index r = 0; r < rows; ++r) {
            const Row row = row_at(r);
            if (!fits(row, payload.shape(0))) { status = bad_payload; break; }
            if (row.height != plane.shape(0) || row.width != plane.shape(1)) { status = bad_plane; break; }
            first[r+1] = checked_add(first[r], Index(row.chunks));
            parameters[r] = row;
        }
        std::vector<Chunk> chunks(status == ok ? first[rows] : 0);
        for (Index r = 0; status == ok && r < rows; ++r)
            status = scan_row(payload, parameters[r], chunks, first[r]);
        Index bytes = 0;
        if (status == ok) {
            for (auto& h : chunks) {
                h.packed_start = bytes;
                bytes = checked_add(bytes, h.mask_raw);
            }
        }
        std::vector<Byte> packed(checked_add(bytes, 1)), strided_input;
        for (Index r = 0; status == ok && r < rows; ++r) {
            Word read = 0;
            for (Index c = first[r]; c < first[r+1]; ++c) {
                const auto& h = chunks[c];
                if (!inflate_stream(inflate, payload, h.at+header_bytes, h.mask_bytes,
                        packed.data()+h.packed_start, h.mask_raw, strided_input)) {
                    status = bad_stream;
                    break;
                }
                read += header_bytes + h.mask_bytes;
            }
            outcome.store<Word>(read, r, 2);
        }
        for (Index r = 0; status == ok && r < rows; ++r) {
            const auto [level, check] = rank_at(r);
            bool row_populated = false, row_gap = false;
            for (Index c = first[r]; c < first[r+1]; ++c) {
                bool populated = false;
                const auto& h = chunks[c];
                merge_chunk(h, packed.data()+h.packed_start, plane, level, check, populated, row_gap);
                if (populated) {
                    row_populated = true;
                    box.top = std::min(box.top, h.y0);
                    box.bottom = std::max(box.bottom, h.y1);
                    box.left = std::min(box.left, h.x0);
                    box.right = std::max(box.right, h.x1);
                }
            }
            outcome.store<Word>(row_populated, r, 0);
            outcome.store<Word>(row_gap, r, 1);
        }
    }
    return Py_BuildValue("(iOnnnn)", int(status), report.get(), box.top, box.bottom, box.left, box.right);
}

PyObject* merge_rank_rows(PyObject*, PyObject* args) {
    PyObject *address, *payload_object, *table_object, *plane_object;
    if (!PyArg_ParseTuple(args, "OOOO:journal_merge_rank_rows", &address, &payload_object,
                          &table_object, &plane_object)) return nullptr;
    return protect([&]() -> PyObject* {
        const auto inflate = inflater_address(address);
        Buffer payload(payload_object, "payload", 1), table(table_object, "rank table", 2);
        Buffer plane(plane_object, "rank plane", 2, true);
        payload.require<Byte>();
        table.require<Word>();
        plane.require<Byte>();
        if (table.shape(1) < 13) value_error("Incomplete journal rank table");
        const Index rows = table.shape(0);
        return merge_rows(inflate, payload, plane, rows,
            [&](Index r) { return read_row([&](Index col) { return table.load<Word>(r, col); }); },
            [&](Index r) { return std::make_pair(table.load<Word>(r, 11), table.load<Word>(r, 12) != 0); });
    });
}

// The ordinary INDEX wire layout is little endian. Each row can start unaligned
// or many records away from the next selected cutoff. No header table is boxed
// in Python; the validated index and the backing mapping stay exported here.
std::uint64_t wire_integer(const Byte* p, unsigned bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= std::uint64_t(p[i]) << (8*i);
    return value;
}

PyObject* merge_family_rank_rows(PyObject*, PyObject* args) {
    PyObject *address, *payload_object, *records_object, *selection_object, *checks_object, *plane_object;
    Index limit;
    if (!PyArg_ParseTuple(args, "OOOOOOn:journal_merge_family_rank_rows", &address, &payload_object,
                          &records_object, &selection_object, &checks_object, &plane_object, &limit))
        return nullptr;
    return protect([&]() -> PyObject* {
        const auto inflate = inflater_address(address);
        Buffer payload(payload_object, "payload", 1), records(records_object, "journal wire records", 2);
        Buffer selected(selection_object, "cutoff positions", 1), checks(checks_object, "gap checks", 1);
        Buffer plane(plane_object, "rank plane", 2, true);
        payload.require<Byte>();
        records.require<Byte>();
        selected.require<Word>();
        checks.require<bool>();
        plane.require<Byte>();
        const Index count = selected.shape(0);
        if (records.shape(1) != 60 || records.stride(1) != 1 || count < 1 || count > 101
            || checks.shape(0) != count || limit < 1)
            value_error("Invalid journal family rank arguments");
        std::array<Word, 101> positions{};
        std::array<bool, 101> gap_checks{};
        Word last = -1;
        for (Index r = 0; r < count; ++r) {
            const Word pos = selected.load<Word>(r);
            if (pos <= last || pos >= records.shape(0))
                value_error("Journal cutoff positions must be in range and strictly increasing");
            last = pos;
            positions[r] = pos;
            gap_checks[r] = checks.load<bool>(r);
        }
        return merge_rows(inflate, payload, plane, count, [&](Index r) -> Row {
            const auto* p = reinterpret_cast<const Byte*>(records.ptr(positions[r], 0));
            const auto offset = wire_integer(p+40, 8), length = wire_integer(p+48, 8);
            if ((p[0] != 1 && p[0] != 2) || offset > std::uint64_t(PY_SSIZE_T_MAX)
                || length > std::uint64_t(PY_SSIZE_T_MAX)) return Row{-1};
            return {Word(offset), Word(length), Word(wire_integer(p+56, 4)),
                    Word(wire_integer(p+24, 4)), Word(wire_integer(p+28, 4)),
                    Word(wire_integer(p+32, 4)), Word(wire_integer(p+36, 4)),
                    plane.shape(0), plane.shape(1), p[2], limit};
        }, [&](Index r) { return std::make_pair(Word(r+1), gap_checks[r]); });
    });
}

PyObject* store_rank_region(PyObject*, PyObject* args) {
    PyObject *target_object, *plane_object;
    Index top, bottom, left, right;
    if (!PyArg_ParseTuple(args, "OOnnnn:journal_store_rank_region", &target_object, &plane_object,
                          &top, &bottom, &left, &right)) return nullptr;
    return protect([&]() -> PyObject* {
        Buffer target(target_object, "rank destination", 2, true), plane(plane_object, "rank plane", 2, true);
        target.require<Byte>();
        plane.require<Byte>();
        // The original kernel takes Python column slices from each row. Keep
        // their clipping and negative-index behavior for public callers.
        Index src_begin = left, src_end = right, dst_begin = left, dst_end = right;
        const Index count = PySlice_AdjustIndices(plane.shape(1), &src_begin, &src_end, 1);
        const Index capacity = PySlice_AdjustIndices(target.shape(1), &dst_begin, &dst_end, 1);
        if (top < bottom && count > 0) {
            if (top < -std::min(target.shape(0), plane.shape(0))
                || bottom > std::min(target.shape(0), plane.shape(0)) || capacity < count)
                value_error("Rank region exceeds its source or destination");
            GilRelease unlocked;
            const bool contiguous_rows = target.stride(1) == 1 && plane.stride(1) == 1;
            const auto width = static_cast<std::size_t>(count);
            for (Index y = top; y < bottom; ++y) {
                const Index sy = y >= 0 ? y : plane.shape(0)+y;
                const Index dy = y >= 0 ? y : target.shape(0)+y;
                if (contiguous_rows) {
                    char* from = plane.ptr(sy, src_begin);
                    char* to = target.ptr(dy, dst_begin);
                    const auto src = reinterpret_cast<std::uintptr_t>(from);
                    const auto dst = reinterpret_cast<std::uintptr_t>(to);
                    if (dst > src && dst-src < width) {
                        // Aliased views retain the original forward element-copy order.
                        for (std::size_t i = 0; i < width; ++i) to[i] = from[i];
                    } else {
                        std::memmove(to, from, width);
                    }
                    std::memset(from, 0, width);
                } else {
                    for (Index x = 0; x < count; ++x)
                        target.store<Byte>(plane.load<Byte>(sy, src_begin+x), dy, dst_begin+x);
                    for (Index x = 0; x < count; ++x) plane.store<Byte>(0, sy, src_begin+x);
                }
            }
        }
        Py_RETURN_NONE;
    });
}

PyMethodDef methods[] = {
    {"journal_decode_row", decode_row, METH_VARARGS, "Decode one bounded native sweep journal row."},
    {"journal_merge_family_rank_rows", merge_family_rank_rows, METH_VARARGS, "Merge selected lanes directly from a sealed journal frame index."},
    {"journal_merge_rank_rows", merge_rank_rows, METH_VARARGS, "Validate and merge journal rows into a rank plane."},
    {"journal_store_rank_region", store_rank_region, METH_VARARGS, "Move a rank region and clear its source."},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace

int register_journal(PyObject* module) { return PyModule_AddFunctions(module, methods); }
} // namespace xta::cpu
