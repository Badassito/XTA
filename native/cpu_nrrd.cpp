#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Byte = std::uint8_t;

constexpr auto make_bit_rows() {
    std::array<std::array<Byte, 8>, 256> rows{};
    for (std::size_t value = 0; value < rows.size(); ++value) {
        for (std::size_t bit = 0; bit < 8; ++bit) {
            rows[value][bit] = static_cast<Byte>((value >> bit) & 1);
        }
    }
    return rows;
}

constexpr auto bit_rows = make_bit_rows();

class Args {
public:
    Args(PyObject* tuple, Index count) : tuple_(tuple) {
        if (PyTuple_GET_SIZE(tuple) != count) {
            type_error("Unexpected number of NRRD kernel arguments");
        }
    }

    PyObject* object(Index i) const { return PyTuple_GET_ITEM(tuple_, i); }

    Index index(Index i) const {
        const Index value = PyNumber_AsSsize_t(object(i), PyExc_OverflowError);
        if (value == -1 && PyErr_Occurred()) {
            throw PythonError{};
        }
        return value;
    }

    bool boolean(Index i) const {
        const int value = PyObject_IsTrue(object(i));
        if (value < 0) {
            throw PythonError{};
        }
        return value != 0;
    }

private:
    PyObject* tuple_;
};

struct Box {
    Index y0, x0, y1, x1;

    Index height() const { return y1 - y0; }
    Index width() const { return x1 - x0; }
    bool nonempty() const { return y1 > y0 && x1 > x0; }

    bool within(Index height, Index width) const {
        return y0 >= 0 && x0 >= 0 && nonempty() && y1 <= height && x1 <= width;
    }
};

template <class T>
struct Plane {
    char* bytes;
    Index row_stride, col_stride;

    T load(Index y, Index x) const {
        T value;
        std::memcpy(&value, bytes + y * row_stride + x * col_stride, sizeof(T));
        return value;
    }

    void store(Index y, Index x, T value) const {
        std::memcpy(bytes + y * row_stride + x * col_stride, &value, sizeof(T));
    }
};

template <class T>
Plane<T> plane(const Buffer& buffer) {
    return {buffer.data(), buffer.stride(0), buffer.stride(1)};
}

template <class T>
Plane<T> dense_plane(T* bytes, Index width) {
    return {reinterpret_cast<char*>(bytes), checked_extent(width, sizeof(T)), sizeof(T)};
}

struct Geometry {
    Index in_h, in_w, out_h, out_w;

    void validate() const {
        if (std::min({in_h, in_w, out_h, out_w}) < 1) {
            throw std::invalid_argument("NRRD resize dimensions must be positive");
        }
    }
};

struct AreaTab {
    double first, stop, cell;
    Index first_full, stop_full;
};

struct AreaSpan {
    double low, high, cell;
};

AreaTab area_tab(Index out_i, Index in_n, Index out_n) {
    const double scale = 1.0 / (static_cast<double>(out_n) / static_cast<double>(in_n));
    const double first = static_cast<double>(out_i) * scale;
    const double stop = first + scale;
    const Index stop_full = stop >= static_cast<double>(in_n - 1)
        ? in_n - 1 : static_cast<Index>(std::floor(stop));
    const double first_full = std::ceil(first);
    return {first, stop, std::min(scale, static_cast<double>(in_n) - first),
            first_full >= static_cast<double>(stop_full) ? stop_full : static_cast<Index>(first_full), stop_full};
}

AreaSpan area_span(Index out_i, Index in_n, Index out_n, Index first, Index stop) {
    const auto tab = area_tab(out_i, in_n, out_n);
    const double start_full = static_cast<double>(tab.first_full);
    const double stop_full = static_cast<double>(tab.stop_full);
    const double low = start_full - tab.first > 1e-3 ? tab.first : start_full;
    const double last_weight = tab.stop - stop_full;
    const double high = last_weight > 1e-3
        ? stop_full + std::min({last_weight, 1.0, tab.cell}) : stop_full;
    const auto start = static_cast<double>(first);
    const double local_low = std::max(low, start) - start;
    return {local_low, std::max(local_low, std::min(high, static_cast<double>(stop)) - start), tab.cell};
}

float area_alpha(Index source_i, const AreaTab& tab) {
    if (source_i == tab.first_full - 1) {
        const double weight = static_cast<double>(tab.first_full) - tab.first;
        return weight > 1e-3 ? static_cast<float>(weight / tab.cell) : 0.0f;
    }
    if (source_i == tab.stop_full) {
        const double weight = tab.stop - static_cast<double>(tab.stop_full);
        return weight > 1e-3 ? static_cast<float>(std::min({weight, 1.0, tab.cell}) / tab.cell) : 0.0f;
    }
    return static_cast<float>(1.0 / tab.cell);
}

double continuous_integral(const Plane<std::int32_t>& integral, const Plane<Byte>& crop,
                           Index height, Index width, double y, double x) {
    y = std::clamp(y, 0.0, static_cast<double>(height));
    x = std::clamp(x, 0.0, static_cast<double>(width));
    const double y_round = std::round(y);
    const double x_round = std::round(x);
    if (std::abs(y - y_round) < 1e-12) {
        y = y_round;
    }
    if (std::abs(x - x_round) < 1e-12) {
        x = x_round;
    }
    const auto iy = static_cast<Index>(std::floor(y));
    const auto ix = static_cast<Index>(std::floor(x));
    const double fy = y - static_cast<double>(iy);
    const double fx = x - static_cast<double>(ix);
    const Index iy_cell = std::min(iy, height - 1);
    const Index ix_cell = std::min(ix, width - 1);
    double value = static_cast<double>(integral.load(iy, ix));
    value += static_cast<double>(static_cast<std::int64_t>(integral.load(iy, ix_cell + 1))
                                - integral.load(iy, ix_cell)) * fx;
    value += static_cast<double>(static_cast<std::int64_t>(integral.load(iy_cell + 1, ix))
                                - integral.load(iy_cell, ix)) * fy;
    value += static_cast<double>(crop.load(iy_cell, ix_cell)) * fy * fx;
    return value;
}

float area_binary_value(const Plane<Byte>& crop, const Box& source, const Geometry& geometry,
                        Index out_y, Index out_x) {
    const auto y_tab = area_tab(out_y, geometry.in_h, geometry.out_h);
    const auto x_tab = area_tab(out_x, geometry.in_w, geometry.out_w);
    float total = 0.0f;
    for (Index sy = std::max(y_tab.first_full - 1, source.y0);
         sy < std::min(y_tab.stop_full + 1, source.y1); ++sy) {
        const float beta = area_alpha(sy, y_tab);
        if (beta == 0.0f) {
            continue;
        }
        float row = 0.0f;
        for (Index sx = std::max(x_tab.first_full - 1, source.x0);
             sx < std::min(x_tab.stop_full + 1, source.x1); ++sx) {
            if (crop.load(sy - source.y0, sx - source.x0) != 0) {
                const float term = 255.0f * area_alpha(sx, x_tab);
                row += term;
            }
        }
        const float term = beta * row;
        total += term;
    }
    return total;
}

bool area_resize(const Plane<std::int32_t>& integral, const Plane<Byte>& crop,
                 const Box& source, const Geometry& geometry, const Box& target,
                 const Plane<Byte>& restored) {
    std::vector<AreaSpan> x_spans;
    x_spans.reserve(static_cast<std::size_t>(target.width()));
    for (Index x = target.x0; x < target.x1; ++x) {
        x_spans.push_back(area_span(x, geometry.in_w, geometry.out_w, source.x0, source.x1));
    }
    bool any = false;
    for (Index y = target.y0; y < target.y1; ++y) {
        const auto sy = area_span(y, geometry.in_h, geometry.out_h, source.y0, source.y1);
        for (Index x = 0; x < target.width(); ++x) {
            const auto sx = x_spans[static_cast<std::size_t>(x)];
            double weighted = continuous_integral(integral, crop, source.height(), source.width(), sy.high, sx.high);
            weighted -= continuous_integral(integral, crop, source.height(), source.width(), sy.low, sx.high);
            weighted -= continuous_integral(integral, crop, source.height(), source.width(), sy.high, sx.low);
            weighted += continuous_integral(integral, crop, source.height(), source.width(), sy.low, sx.low);
            // A half-even uint8 tie stays zero. Reproduce the ordered OpenCV
            // float32 taps near the threshold, where the integral is insufficient.
            const double threshold = sy.cell * sx.cell / 510.0;
            const bool inside = std::abs(weighted - threshold) > 1e-4 * threshold
                ? weighted > threshold
                : area_binary_value(crop, source, geometry, y, target.x0 + x) > 0.5f;
            restored.store(y - target.y0, x, static_cast<Byte>(inside));
            any |= inside;
        }
    }
    return any;
}

Index search_sorted(const Buffer& map, Index value, bool right) {
    Index first = 0;
    Index stop = map.shape(0);
    while (first < stop) {
        const Index mid = first + (stop - first) / 2;
        const auto entry = map.load<std::int64_t>(mid);
        if (entry < value || (right && entry == value)) {
            first = mid + 1;
        } else {
            stop = mid;
        }
    }
    return first;
}

enum ResizeMode { identity, nearest, area };

class ResizePlan {
public:
    ResizePlan(const Args& args, Index start)
        : grid{args.index(start), args.index(start + 1), args.index(start + 2), args.index(start + 3)},
          ys(args.object(start + 4), "NRRD row starts", 1), ye(args.object(start + 5), "NRRD row stops", 1),
          xs(args.object(start + 6), "NRRD column starts", 1), xe(args.object(start + 7), "NRRD column stops", 1),
          mode(args.index(start + 8)) {
        grid.validate();
        for (auto* map : {&ys, &ye, &xs, &xe}) {
            map->require<std::int64_t>();
        }
        if (mode < identity || mode > area
            || (mode == identity && (grid.in_h != grid.out_h || grid.in_w != grid.out_w))
            || (mode == area && (grid.in_h < grid.out_h || grid.in_w < grid.out_w))
            || (mode != identity && (ys.shape(0) != grid.out_h || ye.shape(0) != grid.out_h
                                     || xs.shape(0) != grid.out_w || xe.shape(0) != grid.out_w))) {
            value_error("Invalid NRRD resize plan");
        }
    }

    Box region(const Box& source) const {
        if (!source.within(grid.in_h, grid.in_w)) {
            throw std::invalid_argument("NRRD crop is outside its source plane");
        }
        if (mode == identity) {
            return source;
        }
        const Index y0 = search_sorted(ye, source.y0, true);
        const Index x0 = search_sorted(xe, source.x0, true);
        return {y0, x0, std::max(y0, search_sorted(ys, source.y1, false)),
                std::max(x0, search_sorted(xs, source.x1, false))};
    }

    bool resize(const Plane<Byte>& crop, const Box& source, const Box& region,
                const Plane<Byte>& output) const {
        if (mode == area) {
            const Index height = checked_add(source.height(), 1);
            const Index width = checked_add(source.width(), 1);
            std::vector<std::int32_t> sums(static_cast<std::size_t>(checked_extent(height, width)), 0);
            const auto integral = dense_plane(sums.data(), width);
            for (Index y = 0; y < source.height(); ++y) {
                std::uint32_t row = 0;
                for (Index x = 0; x < source.width(); ++x) {
                    row += crop.load(y, x);
                    const std::uint32_t value = static_cast<std::uint32_t>(integral.load(y, x + 1)) + row;
                    std::int32_t encoded;
                    std::memcpy(&encoded, &value, sizeof(encoded));
                    integral.store(y + 1, x + 1, encoded);
                }
            }
            return area_resize(integral, crop, source, grid, region, output);
        }
        bool any = mode == identity;
        for (Index y = 0; y < region.height(); ++y) {
            const Index sy = mode == identity ? source.y0 + y : ys.load<std::int64_t>(region.y0 + y);
            if (sy < source.y0 || sy >= source.y1) {
                throw std::invalid_argument("NRRD resize row map is outside the crop");
            }
            for (Index x = 0; x < region.width(); ++x) {
                const Index sx = mode == identity ? source.x0 + x : xs.load<std::int64_t>(region.x0 + x);
                if (sx < source.x0 || sx >= source.x1) {
                    throw std::invalid_argument("NRRD resize column map is outside the crop");
                }
                const auto value = crop.load(sy - source.y0, sx - source.x0);
                output.store(y, x, value);
                any |= value != 0;
            }
        }
        return any;
    }

    const Geometry grid;

private:
    Buffer ys, ye, xs, xe;
    Index mode;
};

class CropStore {
public:
    explicit CropStore(const Args& args)
        : chunks(args.object(0), "NRRD crop payload", 1), packed(args.boolean(1)),
          offsets(args.object(2), "NRRD crop offsets", 1), y0s(args.object(3), "NRRD crop start rows", 1),
          x0s(args.object(4), "NRRD crop start columns", 1), y1s(args.object(5), "NRRD crop stop rows", 1),
          x1s(args.object(6), "NRRD crop stop columns", 1), kinds(args.object(7), "NRRD crop kinds", 1) {
        chunks.require<Byte>();
        chunks.require_contiguous();
        kinds.require<Byte>();
        for (auto* field : {&offsets, &y0s, &x0s, &y1s, &x1s}) {
            field->require<std::int64_t>();
            if (field->shape(0) != slices()) {
                value_error("NRRD crop index lengths differ");
            }
        }
    }

    Index slices() const { return kinds.shape(0); }

    bool active(Index z) const {
        if (z < 0 || z >= slices() || kinds.load<Byte>(z) > 1) {
            throw std::invalid_argument("Invalid NRRD crop index");
        }
        return kinds.load<Byte>(z) == 1;
    }

    Box box(Index z) const {
        return {y0s.load<std::int64_t>(z), x0s.load<std::int64_t>(z),
                y1s.load<std::int64_t>(z), x1s.load<std::int64_t>(z)};
    }

    void decode(Index z, const Box& box, const Plane<Byte>& output) const {
        const Index width = box.width();
        const Index encoded_width = packed ? width / 8 + (width % 8 != 0) : width;
        const auto offset = offsets.load<std::int64_t>(z);
        const auto size = checked_extent(box.height(), encoded_width);
        if (offset < 0 || offset > chunks.shape(0) || size > chunks.shape(0) - offset) {
            throw std::invalid_argument("NRRD crop payload is outside its buffer");
        }
        const auto* bytes = reinterpret_cast<const Byte*>(chunks.data()) + offset;
        for (Index y = 0; y < box.height(); ++y) {
            const auto* row = bytes + y * encoded_width;
            if (!packed) {
                std::memcpy(output.bytes + y * output.row_stride, row, static_cast<std::size_t>(width));
                continue;
            }
            const Index full = width / 8;
            for (Index byte = 0; byte < full; ++byte) {
                std::memcpy(output.bytes + y * output.row_stride + byte * 8,
                            bit_rows[row[byte]].data(), 8);
            }
            for (Index x = full * 8; x < width; ++x) {
                output.store(y, x, static_cast<Byte>((row[x / 8] >> (x % 8)) & 1));
            }
        }
    }

private:
    Buffer chunks;
    bool packed;
    Buffer offsets, y0s, x0s, y1s, x1s, kinds;
};

using Record = std::array<std::int64_t, 6>;

PyObject* with_records(Object& buffer, const std::vector<Record>& records) {
    auto array = empty({static_cast<Index>(records.size()), 6}, "int64");
    Buffer values(array.get(), "NRRD output records", 2, true);
    if (!records.empty()) {
        GilRelease unlock;
        std::memcpy(values.data(), records.data(), records.size() * sizeof(Record));
    }
    return PyTuple_Pack(2, buffer.get(), array.get());
}

template <class Function>
void member_slices(const Buffer& firsts, const Buffer& counts, Index slices, Index total, Function&& visit) {
    Index slot = 0;
    for (Index m = 0; m < firsts.shape(0); ++m) {
        const auto first = firsts.load<std::int64_t>(m);
        const auto count = counts.load<std::int64_t>(m);
        if (first < 0 || first > slices || count < 0 || count > slices - first || count > total - slot) {
            throw std::invalid_argument("NRRD member slice range is invalid");
        }
        for (Index z = first; z < first + count; ++z) {
            visit(z, slot++);
        }
    }
    if (slot != total) {
        throw std::invalid_argument("NRRD member slice count differs from the allocation");
    }
}

struct IndexRange { Index first, stop; };

IndexRange destinations(const Buffer& ptr, Index z, Index count) {
    if (z < 0 || z >= ptr.shape(0) - 1) {
        throw std::invalid_argument("NRRD slice is outside its resize plan");
    }
    const auto first = ptr.load<std::int64_t>(z);
    const auto stop = ptr.load<std::int64_t>(z + 1);
    if (first < 0 || stop < first || stop > count) {
        throw std::invalid_argument("Invalid NRRD slice pointer range");
    }
    return {first, stop};
}

Box record_box(const Buffer& records, Index i) {
    return {records.load<std::int64_t>(i, 1), records.load<std::int64_t>(i, 2),
            records.load<std::int64_t>(i, 3), records.load<std::int64_t>(i, 4)};
}

void or_plane(const Plane<Byte>& input, const Plane<Byte>& output, Index height, Index width) {
    for (Index y = 0; y < height; ++y) {
        for (Index x = 0; x < width; ++x) {
            output.store(y, x, static_cast<Byte>(output.load(y, x) | input.load(y, x)));
        }
    }
}

PyObject* area_crop_resize(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 13);
        const Buffer integral(args.object(0), "NRRD integral image", 2);
        const Buffer crop(args.object(1), "NRRD crop", 2);
        const Geometry grid{args.index(2), args.index(3), args.index(4), args.index(5)};
        grid.validate();
        const Box source{args.index(6), args.index(7), args.index(8), args.index(9)};
        const Buffer out(args.object(12), "NRRD crop resize output", 2, true);
        const Index y = args.index(10), x = args.index(11);
        const Box target{y, x, checked_add(y, out.shape(0)), checked_add(x, out.shape(1))};
        crop.require<Byte>();
        integral.require<std::int32_t>();
        out.require<Byte>();
        if (!source.within(grid.in_h, grid.in_w) || source.height() != crop.shape(0) || source.width() != crop.shape(1)
            || integral.shape(0) != checked_add(crop.shape(0), 1)
            || integral.shape(1) != checked_add(crop.shape(1), 1)
            || y < 0 || x < 0 || target.y1 > grid.out_h || target.x1 > grid.out_w) {
            value_error("Invalid NRRD area crop resize bounds");
        }
        if (target.nonempty()) {
            GilRelease unlock;
            area_resize(plane<std::int32_t>(integral), plane<Byte>(crop), source, grid, target, plane<Byte>(out));
        }
        Py_RETURN_NONE;
    });
}

PyObject* fill_crop_row_batch(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 12);
        const CropStore crops(args);
        const Index first = args.index(8), stop = args.index(9);
        const Index width = args.index(10), rows = args.index(11);
        if (first < 0 || stop < first || stop > crops.slices() || width < 0 || rows < 0) {
            value_error("Invalid NRRD row batch shape or range");
        }
        auto array = empty({rows, width}, "uint8");
        const Buffer output(array.get(), "NRRD crop row batch", 2, true);
        {
            GilRelease unlock;
            Index row = 0;
            for (Index z = first; z < stop; ++z) {
                if (!crops.active(z)) {
                    continue;
                }
                const auto box = crops.box(z);
                if (!box.within(PY_SSIZE_T_MAX, width) || box.height() > rows - row) {
                    throw std::invalid_argument("NRRD crop exceeds row batch bounds");
                }
                for (Index r = 0; r < box.height(); ++r) {
                    std::memset(output.ptr(row + r, 0), 0, static_cast<std::size_t>(box.x0));
                    std::memset(output.ptr(row + r, box.x1), 0, static_cast<std::size_t>(width - box.x1));
                }
                crops.decode(z, box, {output.ptr(row, box.x0), output.stride(0), output.stride(1)});
                row += box.height();
            }
            if (row != rows) {
                throw std::invalid_argument("NRRD crop rows differ from the allocation");
            }
        }
        return array.release();
    });
}

PyObject* fill_native_member_group(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 13);
        const CropStore crops(args);
        const Buffer first(args.object(8), "NRRD member starts", 1), count(args.object(9), "NRRD member counts", 1);
        const Index total = args.index(10), height = args.index(11), width = args.index(12);
        first.require<std::int64_t>();
        count.require<std::int64_t>();
        if (first.shape(0) != count.shape(0) || total < 0 || height < 1 || width < 1) {
            value_error("Invalid native NRRD member dimensions");
        }
        auto array = zeros({checked_extent(total, height), width}, "uint8");
        const Buffer output(array.get(), "NRRD native members", 2, true);
        std::vector<Record> records;
        {
            GilRelease unlock;
            member_slices(first, count, crops.slices(), total, [&](Index z, Index slot) {
                if (!crops.active(z)) {
                    return;
                }
                const auto box = crops.box(z);
                if (!box.within(height, width)) {
                    throw std::invalid_argument("NRRD crop is outside its member plane");
                }
                const Index row = slot * height + box.y0;
                crops.decode(z, box, {output.ptr(row, box.x0), output.stride(0), output.stride(1)});
                records.push_back({row, box.y0, box.x0, box.y1, box.x1, z});
            });
        }
        return with_records(array, records);
    });
}

PyObject* fill_restored_member_group(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 22);
        const CropStore crops(args);
        const Buffer first(args.object(8), "NRRD member starts", 1), count(args.object(9), "NRRD member counts", 1);
        const Index total = args.index(10);
        const Buffer source_ptr(args.object(11), "NRRD source pointers", 1);
        const Buffer source_index(args.object(12), "NRRD source indices", 1);
        const ResizePlan plan(args, 13);
        for (const auto* indices : {&first, &count, &source_ptr, &source_index}) {
            indices->require<std::int64_t>();
        }
        if (first.shape(0) != count.shape(0) || total < 0 || source_ptr.shape(0) < 1) {
            value_error("Invalid restored NRRD member dimensions");
        }
        auto array = zeros({checked_extent(total, plan.grid.out_h), plan.grid.out_w}, "uint8");
        const Buffer output(array.get(), "NRRD restored members", 2, true);
        std::vector<Record> records;
        {
            GilRelease unlock;
            std::vector<Byte> crop_bytes, restored_bytes;
            member_slices(first, count, source_ptr.shape(0) - 1, total, [&](Index z, Index slot) {
                const Index row = slot * plan.grid.out_h;
                Box combined{plan.grid.out_h, plan.grid.out_w, 0, 0};
                auto range = destinations(source_ptr, z, source_index.shape(0));
                for (Index t = range.first; t < range.stop; ++t) {
                    const Index source_z = source_index.load<std::int64_t>(t);
                    if (!crops.active(source_z)) {
                        continue;
                    }
                    const auto box = crops.box(source_z);
                    const auto region = plan.region(box);
                    if (!region.nonempty()) {
                        continue;
                    }
                    crop_bytes.resize(static_cast<std::size_t>(checked_extent(box.height(), box.width())));
                    const auto crop = dense_plane(crop_bytes.data(), box.width());
                    crops.decode(source_z, box, crop);
                    restored_bytes.resize(static_cast<std::size_t>(checked_extent(region.height(), region.width())));
                    const auto restored = dense_plane(restored_bytes.data(), region.width());
                    if (!plan.resize(crop, box, region, restored)) {
                        continue;
                    }
                    or_plane(restored, {output.ptr(row + region.y0, region.x0), output.stride(0), output.stride(1)},
                             region.height(), region.width());
                    combined.y0 = std::min(combined.y0, region.y0);
                    combined.x0 = std::min(combined.x0, region.x0);
                    combined.y1 = std::max(combined.y1, region.y1);
                    combined.x1 = std::max(combined.x1, region.x1);
                }
                if (combined.nonempty()) {
                    records.push_back({row + combined.y0, combined.y0, combined.x0, combined.y1, combined.x1, z});
                }
            });
        }
        return with_records(array, records);
    });
}

PyObject* resize_records_for_tee(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 12);
        const Buffer source(args.object(0), "NRRD tee source", 2), records(args.object(1), "NRRD crop records", 2);
        const Buffer ptr(args.object(2), "NRRD target pointers", 1);
        const ResizePlan plan(args, 3);
        source.require<Byte>();
        records.require<std::int64_t>();
        ptr.require<std::int64_t>();
        if (records.shape(1) != 6) {
            value_error("Invalid NRRD crop record width");
        }
        auto array = zeros({records.shape(0), 5}, "int64");
        const Buffer regions(array.get(), "NRRD tee regions", 2, true);
        struct StagedCrop { Index record, row; Box source; };
        std::vector<StagedCrop> staged;
        Index total = 0;
        {
            GilRelease unlock;
            for (Index i = 0; i < records.shape(0); ++i) {
                const auto z = records.load<std::int64_t>(i, 5);
                const auto range = destinations(ptr, z, PY_SSIZE_T_MAX);
                if (range.first == range.stop) {
                    continue;
                }
                const auto box = record_box(records, i);
                const auto region = plan.region(box);
                if (!region.nonempty()) {
                    continue;
                }
                const auto row = records.load<std::int64_t>(i, 0);
                if (row < 0 || row > source.shape(0) || box.height() > source.shape(0) - row || box.x1 > source.shape(1)) {
                    throw std::invalid_argument("NRRD tee crop is outside its member buffer");
                }
                regions.store<std::int64_t>(region.y0, i, 0);
                regions.store<std::int64_t>(region.x0, i, 1);
                regions.store<std::int64_t>(region.y1, i, 2);
                regions.store<std::int64_t>(region.x1, i, 3);
                regions.store<std::int64_t>(total, i, 4);
                total = checked_add(total, checked_extent(region.height(), region.width()));
                staged.push_back({i, row, box});
            }
        }
        auto pixels = empty({total}, "uint8");
        const Buffer results(pixels.get(), "NRRD tee payload", 1, true);
        {
            GilRelease unlock;
            for (const auto& item : staged) {
                const auto i = item.record;
                const Box region{regions.load<std::int64_t>(i, 0), regions.load<std::int64_t>(i, 1),
                                 regions.load<std::int64_t>(i, 2), regions.load<std::int64_t>(i, 3)};
                const auto offset = regions.load<std::int64_t>(i, 4);
                const Plane<Byte> crop{source.ptr(item.row, item.source.x0), source.stride(0), source.stride(1)};
                if (!plan.resize(crop, item.source, region,
                                 dense_plane(reinterpret_cast<Byte*>(results.ptr(offset)), region.width()))) {
                    regions.store<std::int64_t>(region.y0, i, 2);
                    regions.store<std::int64_t>(region.x0, i, 3);
                }
            }
        }
        return PyTuple_Pack(2, array.get(), pixels.get());
    });
}

PyObject* or_tee_records(PyObject*, PyObject* raw) {
    return protect([&]() -> PyObject* {
        const Args args(raw, 6);
        const Buffer regions(args.object(0), "NRRD tee regions", 2), results(args.object(1), "NRRD tee payload", 1);
        const Buffer records(args.object(2), "NRRD crop records", 2), ptr(args.object(3), "NRRD target pointers", 1);
        const Buffer indices(args.object(4), "NRRD target indices", 1), volume(args.object(5), "NRRD target volume", 3, true);
        for (const auto* item : {&regions, &records, &ptr, &indices}) {
            item->require<std::int64_t>();
        }
        results.require<Byte>();
        volume.require<Byte>();
        if (records.shape(1) != 6 || regions.shape(1) != 5 || regions.shape(0) != records.shape(0)) {
            value_error("Invalid NRRD tee record dimensions");
        }
        {
            GilRelease unlock;
            for (bool commit : {false, true}) {
                for (Index i = 0; i < records.shape(0); ++i) {
                    const Box region{regions.load<std::int64_t>(i, 0), regions.load<std::int64_t>(i, 1),
                                     regions.load<std::int64_t>(i, 2), regions.load<std::int64_t>(i, 3)};
                    if (region.y1 <= region.y0) {
                        continue;
                    }
                    const auto offset = regions.load<std::int64_t>(i, 4);
                    if (!region.within(volume.shape(1), volume.shape(2)) || offset < 0 || offset > results.shape(0)
                        || checked_extent(region.height(), region.width()) > results.shape(0) - offset) {
                        throw std::invalid_argument("NRRD tee region is outside its bounds");
                    }
                    const auto range = destinations(ptr, records.load<std::int64_t>(i, 5), indices.shape(0));
                    for (Index t = range.first; t < range.stop; ++t) {
                        const auto z = indices.load<std::int64_t>(t);
                        if (z < 0 || z >= volume.shape(0)) {
                            throw std::invalid_argument("NRRD tee target is outside the volume");
                        }
                        if (commit) {
                            const Index row_stride = region.height() > 1 ? region.width() * results.stride(0) : 0;
                            const Plane<Byte> input{results.ptr(offset), row_stride, results.stride(0)};
                            const Plane<Byte> output{volume.ptr(z, region.y0, region.x0), volume.stride(1), volume.stride(2)};
                            or_plane(input, output, region.height(), region.width());
                        }
                    }
                }
            }
        }
        Py_RETURN_NONE;
    });
}

PyMethodDef methods[] = {
    {"nrrd_area_crop_resize", area_crop_resize, METH_VARARGS, nullptr},
    {"nrrd_fill_crop_row_batch", fill_crop_row_batch, METH_VARARGS, nullptr},
    {"nrrd_fill_native_member_group", fill_native_member_group, METH_VARARGS, nullptr},
    {"nrrd_fill_restored_member_group", fill_restored_member_group, METH_VARARGS, nullptr},
    {"nrrd_resize_records_for_tee", resize_records_for_tee, METH_VARARGS, nullptr},
    {"nrrd_or_tee_records", or_tee_records, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_nrrd(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
