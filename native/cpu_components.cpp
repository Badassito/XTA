#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;

bool foreground(const Buffer& mask, Index y, Index x) {
    return mask.load<std::uint8_t>(y, x) != 0;
}

Index packed_width(Index width) {
    return width / 8 + (width % 8 != 0);
}

void require_mask(const Buffer& crop) {
    if (!crop.is<bool>() && !crop.is<std::uint8_t>()) {
        type_error("Component crop must be boolean or uint8");
    }
}

void validate_component_crop(const Buffer& crop, const Buffer& bounds,
                             Index slice, Index first_t, Index first_x) {
    require_mask(crop);
    bounds.require<std::int64_t>();
    if (bounds.shape(1) != 4 || first_t < 0 || first_t > bounds.shape(0)
        || crop.shape(0) > bounds.shape(0) - first_t || first_x < 0 || slice < 0) {
        value_error("Component crop coordinates exceed the destination bounds");
    }
    checked_add(first_x, crop.shape(1));
    checked_add(slice, 1);
}

PyObject* sparse_component_accumulate_bounds(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *crop_obj, *bounds_obj;
        Index slice, first_t, first_x;
        int coronal;
        if (!PyArg_ParseTuple(args, "OnnnpO:sparse_component_accumulate_bounds",
                             &crop_obj, &slice, &first_t, &first_x, &coronal, &bounds_obj)) {
            return nullptr;
        }
        Buffer crop(crop_obj, "component crop", 2);
        Buffer bounds(bounds_obj, "component bounds", 2, true);
        validate_component_crop(crop, bounds, slice, first_t, first_x);
        Index count = 0;
        {
            GilRelease release;
            for (Index row = 0; row < crop.shape(0); ++row) {
                Index left = crop.shape(1);
                Index right = -1;
                for (Index col = 0; col < crop.shape(1); ++col) {
                    if (foreground(crop, row, col)) {
                        left = std::min(left, col);
                        right = col;
                        ++count;
                    }
                }
                if (right < 0) {
                    continue;
                }
                const Index t = first_t + row;
                const std::int64_t y0 = coronal ? first_x + left : slice;
                const std::int64_t x0 = coronal ? slice : first_x + left;
                const std::int64_t y1 = coronal ? first_x + right + 1 : slice + 1;
                const std::int64_t x1 = coronal ? slice + 1 : first_x + right + 1;
                bounds.store(std::min(bounds.load<std::int64_t>(t, 0), y0), t, 0);
                bounds.store(std::min(bounds.load<std::int64_t>(t, 1), x0), t, 1);
                bounds.store(std::max(bounds.load<std::int64_t>(t, 2), y1), t, 2);
                bounds.store(std::max(bounds.load<std::int64_t>(t, 3), x1), t, 3);
            }
        }
        return PyLong_FromSsize_t(count);
    });
}

PyObject* sparse_component_scatter_crop(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *crop_obj, *bounds_obj, *offsets_obj, *payload_obj;
        Index slice, first_t, first_x;
        int coronal;
        if (!PyArg_ParseTuple(args, "OnnnpOOO:sparse_component_scatter_crop",
                             &crop_obj, &slice, &first_t, &first_x, &coronal,
                             &bounds_obj, &offsets_obj, &payload_obj)) {
            return nullptr;
        }
        Buffer crop(crop_obj, "component crop", 2);
        Buffer bounds(bounds_obj, "component bounds", 2);
        Buffer offsets(offsets_obj, "component offsets", 1);
        Buffer payload(payload_obj, "component payload", 1, true);
        validate_component_crop(crop, bounds, slice, first_t, first_x);
        offsets.require<std::int64_t>();
        payload.require<std::uint8_t>();
        if (offsets.shape(0) != bounds.shape(0)) {
            value_error("Component offsets do not match the destination depth");
        }
        {
            GilRelease release;
            for (Index row = 0; row < crop.shape(0); ++row) {
                const Index t = first_t + row;
                const auto y0 = bounds.load<std::int64_t>(t, 0);
                const auto x0 = bounds.load<std::int64_t>(t, 1);
                const auto y1 = bounds.load<std::int64_t>(t, 2);
                const auto x1 = bounds.load<std::int64_t>(t, 3);
                const auto offset = offsets.load<std::int64_t>(t);
                Index stride = 0;
                for (Index col = 0; col < crop.shape(1); ++col) {
                    if (!foreground(crop, row, col)) {
                        continue;
                    }
                    const Index y = coronal ? first_x + col : slice;
                    const Index x = coronal ? slice : first_x + col;
                    if (!stride) {
                        if (y0 < 0 || x0 < 0 || y1 <= y0 || x1 <= x0
                            || y1 > PY_SSIZE_T_MAX || x1 > PY_SSIZE_T_MAX
                            || offset < 0 || offset > payload.shape(0)) {
                            throw std::invalid_argument("Invalid component payload bounds");
                        }
                        stride = packed_width(static_cast<Index>(x1 - x0));
                        if (y1 - y0 > (payload.shape(0) - offset) / stride) {
                            throw std::invalid_argument("Component payload is shorter than its bounds");
                        }
                    }
                    if (y < y0 || y >= y1 || x < x0 || x >= x1) {
                        throw std::invalid_argument("Component voxel falls outside its packed bounds");
                    }
                    const Index local_x = static_cast<Index>(x - x0);
                    const Index at = static_cast<Index>(offset + (y - y0) * stride + local_x / 8);
                    const auto pixel = static_cast<std::uint8_t>(1u << (local_x % 8));
                    payload.store<std::uint8_t>(payload.load<std::uint8_t>(at) | pixel, at);
                }
            }
        }
        Py_RETURN_NONE;
    });
}

PyObject* or_packed_crop_into_plane(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *destination_obj, *crop_obj;
        Index y0, x0, width;
        if (!PyArg_ParseTuple(args, "OOnnn:or_packed_crop_into_plane", &destination_obj,
                             &crop_obj, &y0, &x0, &width)) {
            return nullptr;
        }
        Buffer destination(destination_obj, "packed destination", 2, true);
        Buffer crop(crop_obj, "packed crop", 2);
        destination.require<std::uint8_t>();
        crop.require<std::uint8_t>();
        if (y0 < 0 || x0 < 0 || width < 0 || crop.shape(1) != packed_width(width) ||
            y0 > destination.shape(0) || crop.shape(0) > destination.shape(0) - y0 ||
            packed_width(checked_add(x0, width)) > destination.shape(1)) {
            value_error("Packed crop coordinates exceed the destination");
        }
        const Index first_byte = x0 / 8;
        const Index stop_byte = packed_width(x0 + width);
        const int shift = static_cast<int>(x0 % 8);
        const unsigned last_mask = width % 8 ? (1u << (width % 8)) - 1u : 255u;
        auto merge = [&](Index y, Index x, unsigned value) {
            destination.store<std::uint8_t>(destination.load<std::uint8_t>(y, x) | value, y, x);
        };
        {
            GilRelease release;
            for (Index row = 0; row < crop.shape(0); ++row) {
                unsigned carry = 0;
                for (Index col = 0; col < crop.shape(1); ++col) {
                    unsigned value = crop.load<std::uint8_t>(row, col);
                    if (col + 1 == crop.shape(1)) {
                        // Ignore padding bits, including in older packed stores.
                        value &= last_mask;
                    }
                    const unsigned shifted = (value << shift) | carry;
                    merge(y0 + row, first_byte + col, shifted & 255u);
                    carry = shifted >> 8;
                }
                if (first_byte + crop.shape(1) < stop_byte) {
                    merge(y0 + row, first_byte + crop.shape(1), carry);
                }
            }
        }
        Py_RETURN_NONE;
    });
}

PyObject* containment_counts(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *raw_obj, *effective_obj, *acceptance_obj;
        if (!PyArg_ParseTuple(args, "OOO:containment_counts", &raw_obj,
                             &effective_obj, &acceptance_obj)) {
            return nullptr;
        }
        Buffer raw(raw_obj, "raw mask", 2);
        Buffer effective(effective_obj, "effective mask", 2);
        Buffer acceptance(acceptance_obj, "acceptance mask", 2);
        const auto height = raw.shape(0);
        const auto width = raw.shape(1);
        if (!raw.is<bool>() || !effective.is<bool>() || !acceptance.is<bool>()
            || effective.shape(0) != height || effective.shape(1) != width
            || acceptance.shape(0) != height || acceptance.shape(1) != width) {
            value_error("SAM containment requires matching boolean planes");
        }
        Index raw_foreground = 0, foreground_count = 0, raw_outside = 0;
        Index outside = 0, raw_touch = 0, touch = 0;
        {
            GilRelease release;
            for (Index y = 0; y < height; ++y) {
                for (Index x = 0; x < width; ++x) {
                    const bool r = raw.load<bool>(y, x);
                    const bool e = effective.load<bool>(y, x);
                    raw_foreground += r;
                    foreground_count += e;
                    if (!r && !e) {
                        continue;
                    }
                    if (!acceptance.load<bool>(y, x)) {
                        raw_outside += r;
                        outside += e;
                        continue;
                    }
                    const bool boundary = y == 0 || x == 0 || y == height - 1 || x == width - 1
                        || !(acceptance.load<bool>(y - 1, x - 1) && acceptance.load<bool>(y - 1, x)
                             && acceptance.load<bool>(y - 1, x + 1) && acceptance.load<bool>(y, x - 1)
                             && acceptance.load<bool>(y, x + 1) && acceptance.load<bool>(y + 1, x - 1)
                             && acceptance.load<bool>(y + 1, x) && acceptance.load<bool>(y + 1, x + 1));
                    if (boundary) {
                        raw_touch += r;
                        touch += e;
                    }
                }
            }
        }
        return Py_BuildValue("(nnnnnn)", raw_foreground, foreground_count, raw_outside,
                             outside, raw_touch, touch);
    });
}

class IntegerVector {
public:
    IntegerVector(PyObject* object, const char* label) : values_(object, label, 1) {
        if (values_.is<std::int8_t>()) type_ = Type::i8;
        else if (values_.is<std::uint8_t>()) type_ = Type::u8;
        else if (values_.is<std::int16_t>()) type_ = Type::i16;
        else if (values_.is<std::uint16_t>()) type_ = Type::u16;
        else if (values_.is<std::int32_t>()) type_ = Type::i32;
        else if (values_.is<std::uint32_t>()) type_ = Type::u32;
        else if (values_.is<std::int64_t>()) type_ = Type::i64;
        else if (values_.is<std::uint64_t>()) type_ = Type::u64;
        else type_error("Sparse bounds require native-endian integer vectors");
    }

    Index size() const noexcept { return values_.shape(0); }

    std::int64_t at(Index at) const {
        switch (type_) {
            case Type::i8: return values_.load<std::int8_t>(at);
            case Type::u8: return values_.load<std::uint8_t>(at);
            case Type::i16: return values_.load<std::int16_t>(at);
            case Type::u16: return values_.load<std::uint16_t>(at);
            case Type::i32: return values_.load<std::int32_t>(at);
            case Type::u32: return values_.load<std::uint32_t>(at);
            case Type::i64: return values_.load<std::int64_t>(at);
            case Type::u64: {
                const auto value = values_.load<std::uint64_t>(at);
                if (value > static_cast<std::uint64_t>(INT64_MAX)) {
                    throw std::overflow_error("Sparse bounds integer exceeds its signed coordinate range");
                }
                return static_cast<std::int64_t>(value);
            }
        }
        throw std::logic_error("Unknown sparse bounds integer type");
    }

private:
    enum class Type { i8, u8, i16, u16, i32, u32, i64, u64 };
    Buffer values_;
    Type type_;
};

using Span = std::pair<Index, Index>;

bool is_monotone(const IntegerVector& values) {
    for (Index at = 1; at < values.size(); ++at) {
        if (values.at(at) < values.at(at - 1)) {
            return false;
        }
    }
    return true;
}

Index lower_bound(const IntegerVector& values, std::int64_t wanted) {
    Index first = 0, stop = values.size();
    while (first < stop) {
        const Index middle = first + (stop - first) / 2;
        if (values.at(middle) < wanted) {
            first = middle + 1;
        } else {
            stop = middle;
        }
    }
    return first;
}

Span native_span(const IntegerVector& values, std::int64_t lower, std::int64_t upper,
                 bool monotone) {
    if (monotone) {
        return {lower_bound(values, lower), lower_bound(values, upper) - 1};
    }
    Span result{values.size(), -1};
    for (Index at = 0; at < values.size(); ++at) {
        const auto value = values.at(at);
        if (lower <= value && value < upper) {
            result.first = std::min(result.first, at);
            result.second = at;
        }
    }
    return result;
}

PyObject* native_span_call(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* values_obj;
        long long lower, upper;
        int monotone;
        if (!PyArg_ParseTuple(args, "OLLp:native_span", &values_obj, &lower, &upper, &monotone)) {
            return nullptr;
        }
        IntegerVector values(values_obj, "native lookup");
        Span range;
        {
            GilRelease release;
            range = native_span(values, lower, upper, monotone != 0);
        }
        return Py_BuildValue("(nn)", range.first, range.second);
    });
}

using RowExtents = std::vector<std::pair<std::int32_t, std::int32_t>>;

RowExtents physical_row_extents(const IntegerVector& rows, Index processing_h, Index stack_size) {
    const Index native_h = rows.size();
    if (processing_h < 0 || stack_size < 0 || stack_size > INT32_MAX
        || (stack_size && !native_h)) {
        throw std::invalid_argument("Invalid sparse physical row geometry");
    }
    RowExtents result(static_cast<std::size_t>(processing_h),
                      {static_cast<std::int32_t>(stack_size), -1});
    for (Index frame = 0; frame < stack_size; ++frame) {
        Index start, stop;
        if (native_h >= stack_size) {
            start = static_cast<Index>(std::floor(static_cast<double>(frame) * native_h / stack_size));
            stop = static_cast<Index>(std::ceil(static_cast<double>(frame + 1) * native_h / stack_size));
            start = std::clamp<Index>(start, 0, native_h - 1);
            stop = std::clamp<Index>(std::max(start + 1, stop), 1, native_h);
        } else {
            const double coord = (static_cast<double>(frame) + .5)
                                 * (static_cast<double>(native_h) / stack_size) - .5;
            const double floor = std::floor(coord);
            const auto integer = static_cast<Index>(floor);
            const double fraction = coord - floor;
            const bool round_up = fraction > .5 || (fraction == .5 && integer % 2 != 0);
            start = std::clamp<Index>(integer + round_up, 0, native_h - 1);
            stop = start + 1;
        }
        for (Index row = start; row < stop; ++row) {
            const auto mapped = rows.at(row);
            if (mapped < 0 || mapped >= processing_h) {
                throw std::invalid_argument("Sparse physical row lookup exceeds its range");
            }
            auto& entry = result[static_cast<std::size_t>(mapped)];
            entry.first = std::min(entry.first, static_cast<std::int32_t>(frame));
            entry.second = std::max(entry.second, static_cast<std::int32_t>(frame));
        }
    }
    return result;
}

PyObject* physical_row_extents_call(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* rows_obj;
        Index processing_h, stack_size;
        if (!PyArg_ParseTuple(args, "Onn:physical_row_extents", &rows_obj, &processing_h, &stack_size)) {
            return nullptr;
        }
        IntegerVector rows(rows_obj, "native row lookup");
        if (processing_h < 0) {
            value_error("Invalid sparse physical row count");
        }
        Object first_obj = empty({processing_h}, "int32");
        Object last_obj = empty({processing_h}, "int32");
        Buffer first(first_obj.get(), "first physical row", 1, true);
        Buffer last(last_obj.get(), "last physical row", 1, true);
        {
            GilRelease release;
            const auto extents = physical_row_extents(rows, processing_h, stack_size);
            for (Index row = 0; row < processing_h; ++row) {
                first.store(extents[row].first, row);
                last.store(extents[row].second, row);
            }
        }
        return PyTuple_Pack(2, first_obj.get(), last_obj.get());
    });
}

constexpr auto inf = std::numeric_limits<double>::infinity();

std::pair<double, double> disk_axis(double radius, double lower, double upper,
                                    double along, double across) {
    double low = inf, high = -inf;
    const double points[] = {lower, upper, -radius * along, radius * along};
    for (int point = 0; point < 4; ++point) {
        const double coordinate = points[point];
        if (point >= 2 && !(lower <= coordinate && coordinate <= upper)) {
            continue;
        }
        const double spread = std::abs(across)
                              * std::sqrt(std::max(0., radius * radius - coordinate * coordinate));
        low = std::min(low, coordinate * along - spread);
        high = std::max(high, coordinate * along + spread);
    }
    return {low, high};
}

struct FrameBounds {
    double row_lower = inf, row_upper = 0, signed_lower = 0, signed_upper = 0;
};

PyObject* sparse_native_bounds(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *records, *rows_obj, *columns_obj, *sources_obj, *reverses_obj;
        PyObject *coefficient_ids_obj, *cosines_obj, *sines_obj, *limits_obj;
        Index processing_h, diameter, stack_size, plane_h, plane_w;
        double radius, center_y, center_x, tangent, origin;
        int base, tilted, vertical;
        if (!PyArg_ParseTuple(args, "OOOnOOOOOOdnnnnddippdd:sparse_native_bounds",
                             &records, &rows_obj, &columns_obj, &processing_h, &sources_obj,
                             &reverses_obj, &coefficient_ids_obj, &cosines_obj, &sines_obj,
                             &limits_obj, &radius, &diameter, &stack_size, &plane_h, &plane_w,
                             &center_y, &center_x, &base, &tilted, &vertical, &tangent, &origin)) {
            return nullptr;
        }
        if (!PyTuple_Check(records) || PyTuple_GET_SIZE(records) != 5) {
            value_error("Sparse bounds require five integer index fields");
        }
        IntegerVector kinds(PyTuple_GET_ITEM(records, 0), "record kinds");
        IntegerVector starts_y(PyTuple_GET_ITEM(records, 1), "record y0");
        IntegerVector starts_x(PyTuple_GET_ITEM(records, 2), "record x0");
        IntegerVector stops_y(PyTuple_GET_ITEM(records, 3), "record y1");
        IntegerVector stops_x(PyTuple_GET_ITEM(records, 4), "record x1");
        IntegerVector rows(rows_obj, "native row lookup");
        IntegerVector columns(columns_obj, "native column lookup");
        IntegerVector sources(sources_obj, "sample frames");
        IntegerVector coefficient_ids(coefficient_ids_obj, "sample coefficient ids");
        Buffer reverses(reverses_obj, "sample reversals", 1);
        Buffer cosines(cosines_obj, "cosines", 1);
        Buffer sines(sines_obj, "sines", 1);
        Buffer limits(limits_obj, "cosine limits", 1);
        reverses.require<bool>();
        cosines.require<double>();
        sines.require<double>();
        limits.require<double>();
        if (starts_y.size() != kinds.size() || starts_x.size() != kinds.size() ||
            stops_y.size() != kinds.size() || stops_x.size() != kinds.size() ||
            reverses.shape(0) != sources.size() || coefficient_ids.size() != sources.size() ||
            sines.shape(0) != cosines.shape(0) || limits.shape(0) != cosines.shape(0) ||
            diameter <= 0 || plane_h <= 0 || plane_w <= 0 || base < 0 || base > 2) {
            value_error("Invalid sparse native bounds geometry");
        }
        Object lower_obj = empty({3}, "float64");
        Object upper_obj = empty({3}, "float64");
        Buffer lower_bounds(lower_obj.get(), "lower sparse bounds", 1, true);
        Buffer upper_bounds(upper_obj.get(), "upper sparse bounds", 1, true);
        {
            GilRelease release;
            const auto physical = physical_row_extents(rows, processing_h, stack_size);
            const bool row_monotone = is_monotone(rows);
            const bool column_monotone = is_monotone(columns);
            const double extent_radius = radius + .5;
            const double diameter_span = static_cast<double>(std::max<Index>(1, diameter - 1));
            std::vector<FrameBounds> frames(static_cast<std::size_t>(kinds.size()));
            for (Index frame = 0; frame < kinds.size(); ++frame) {
                if (!kinds.at(frame)) {
                    continue;
                }
                const auto y0 = starts_y.at(frame), y1 = stops_y.at(frame);
                const auto x0 = starts_x.at(frame), x1 = stops_x.at(frame);
                const auto [col_first, col_last] = native_span(columns, x0, x1, column_monotone);
                const auto [native_first, native_last] = native_span(rows, y0, y1, row_monotone);
                Index first = stack_size, last = -1;
                const auto first_row = std::clamp<std::int64_t>(y0, 0, processing_h);
                const auto stop_row = std::min<std::int64_t>(processing_h, y1);
                for (Index row = static_cast<Index>(first_row); row < stop_row; ++row) {
                    first = std::min<Index>(first, physical[row].first);
                    last = std::max<Index>(last, physical[row].second);
                }
                if (last < first || native_last < native_first || col_last < col_first) {
                    continue;
                }
                const auto u_low = std::max<Index>(0, col_first - 1);
                const auto u_high = std::min<Index>(diameter - 1, col_last + 1);
                frames[frame] = {
                    std::min(static_cast<double>(first) - 1.,
                             static_cast<double>(native_first) * stack_size / rows.size() - .5),
                    std::max(static_cast<double>(last) + 1.,
                             static_cast<double>(native_last + 1) * stack_size / rows.size() - .5),
                    u_low == 0 ? -extent_radius : (u_low - .5) * 2. * radius / diameter_span - radius,
                    u_high == diameter - 1 ? extent_radius : (u_high + .5) * 2. * radius / diameter_span - radius,
                };
            }
            std::array<double, 3> lower_result{inf, inf, inf};
            std::array<double, 3> upper_result{-inf, -inf, -inf};
            for (Index sample = 0; sample < sources.size(); ++sample) {
                const auto frame = sources.at(sample);
                if (frame < 0 || frame >= kinds.size()) {
                    continue;
                }
                const auto& bounds = frames[static_cast<std::size_t>(frame)];
                if (bounds.row_lower == inf) {
                    continue;
                }
                double lower = bounds.signed_lower, upper = bounds.signed_upper;
                if (reverses.load<bool>(sample)) {
                    lower = -bounds.signed_upper;
                    upper = -bounds.signed_lower;
                }
                const auto id = coefficient_ids.at(sample);
                if (id < 0 || id >= cosines.shape(0)) {
                    throw std::invalid_argument("Sparse bounds coefficient is outside its table");
                }
                const double limit = limits.load<double>(id);
                const double owned_radius = limit > 0
                    ? std::min(extent_radius, std::max(std::abs(lower), std::abs(upper)) / limit)
                    : extent_radius;
                lower = std::max(-owned_radius, lower);
                upper = std::min(owned_radius, upper);
                if (lower > upper) {
                    continue;
                }
                const double cosine = cosines.load<double>(id), sine = sines.load<double>(id);
                const auto [dx0, dx1] = disk_axis(owned_radius, lower, upper, cosine, -sine);
                const auto [dy0, dy1] = disk_axis(owned_radius, lower, upper, sine, cosine);
                const double v0 = std::max(0., center_y + dy0 - 1.);
                const double v1 = std::min(static_cast<double>(plane_h - 1), center_y + dy1 + 1.);
                const double u0 = std::max(0., center_x + dx0 - 1.);
                const double u1 = std::min(static_cast<double>(plane_w - 1), center_x + dx1 + 1.);
                double stack0 = bounds.row_lower, stack1 = bounds.row_upper;
                if (tilted) {
                    const double center = static_cast<double>((vertical ? plane_h : plane_w) - 1) / 2.;
                    const double shift0 = tangent * ((vertical ? v0 : u0) - center);
                    const double shift1 = tangent * ((vertical ? v1 : u1) - center);
                    stack0 += origin + std::min(shift0, shift1) - 1.;
                    stack1 += origin + std::max(shift0, shift1) + 1.;
                }
                const double low[] = {base == 0 ? stack0 : v0, base == 1 ? stack0 : base == 0 ? v0 : u0,
                                      base == 2 ? stack0 : u0};
                const double high[] = {base == 0 ? stack1 : v1, base == 1 ? stack1 : base == 0 ? v1 : u1,
                                       base == 2 ? stack1 : u1};
                for (int axis = 0; axis < 3; ++axis) {
                    lower_result[axis] = std::min(lower_result[axis], std::floor(low[axis]));
                    upper_result[axis] = std::max(upper_result[axis], std::ceil(high[axis]));
                }
            }
            for (int axis = 0; axis < 3; ++axis) {
                lower_bounds.store(lower_result[axis], axis);
                upper_bounds.store(upper_result[axis], axis);
            }
        }
        return PyTuple_Pack(2, lower_obj.get(), upper_obj.get());
    });
}

PyMethodDef methods[] = {
    {"sparse_component_accumulate_bounds", sparse_component_accumulate_bounds, METH_VARARGS, nullptr},
    {"sparse_component_scatter_crop", sparse_component_scatter_crop, METH_VARARGS, nullptr},
    {"or_packed_crop_into_plane", or_packed_crop_into_plane, METH_VARARGS, nullptr},
    {"containment_counts", containment_counts, METH_VARARGS, nullptr},
    {"native_span", native_span_call, METH_VARARGS, nullptr},
    {"physical_row_extents", physical_row_extents_call, METH_VARARGS, nullptr},
    {"sparse_native_bounds", sparse_native_bounds, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_components(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
