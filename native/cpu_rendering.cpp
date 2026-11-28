#include "cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_M_X64) || (defined(__x86_64__) && defined(__SSE2__))
#include <emmintrin.h>
#endif

namespace xta::cpu {
namespace {

using Size = Py_ssize_t;
using Byte = std::uint8_t;
constexpr double pi = 3.141592653589793238462643383279502884;

double positive_remainder(double value, double period) {
    const double rest = std::fmod(value, period);
    return rest == 0. ? 0. : (rest < 0. ? rest + period : rest);
}

struct Box {
    Size top, left, bottom, right;

    void validate() const {
        if (top < 0 || left < 0 || top >= bottom || left >= right) {
            value_error("Shell crop bounds must describe a nonempty, nonnegative rectangle");
        }
    }

    Size height() const { return bottom - top; }
    Size width() const { return right - left; }
};

void validate_source(const Buffer& source) {
    source.require<Byte>();
    if (!source.shape(0) || !source.shape(1) || !source.shape(2)) {
        value_error("Shell rendering requires a nonempty physical source");
    }
}

struct CartesianPair {
    Size first, second;
    float weight;
};

CartesianPair cartesian_pair(const Buffer& lo, const Buffer& hi, const Buffer& alpha,
                              Size logical, Size native_depth) {
    const auto first = lo.load<std::int64_t>(logical);
    const auto second = hi.load<std::int64_t>(logical);
    const auto weight = alpha.load<float>(logical);
    if (first < 0 || first >= native_depth || second < 0 || second >= native_depth
        || !(weight >= 0.f && weight <= 1.f)) {
        throw std::invalid_argument("Cartesian time map must name valid slices and weights");
    }
    return {static_cast<Size>(first), static_cast<Size>(second), weight};
}

Byte round_gray8(float value) noexcept {
    const auto whole = static_cast<unsigned>(value);
    const float fraction = value - static_cast<float>(whole);
    return static_cast<Byte>(whole + (fraction > .5f || (fraction == .5f && (whole & 1u))));
}

#if defined(_M_X64) || (defined(__x86_64__) && defined(__SSE2__))
__m128i cartesian_lerp_round4(__m128i a32, __m128i b32, __m128 weight) noexcept {
    const auto a = _mm_cvtepi32_ps(a32);
    const auto b = _mm_cvtepi32_ps(b32);
    const auto value = _mm_add_ps(a, _mm_mul_ps(weight, _mm_sub_ps(b, a)));
    // Truncation followed by explicit ties-to-even works independently of the
    // SIMD conversion rounding mode. The source interpolation stays unfused.
    const auto whole = _mm_cvttps_epi32(value);
    const auto fraction = _mm_sub_ps(value, _mm_cvtepi32_ps(whole));
    const auto half = _mm_set1_ps(.5f);
    const auto odd = _mm_and_si128(whole, _mm_set1_epi32(1));
    const auto above = _mm_and_si128(_mm_castps_si128(_mm_cmpgt_ps(fraction, half)), _mm_set1_epi32(1));
    const auto tie = _mm_and_si128(_mm_castps_si128(_mm_cmpeq_ps(fraction, half)), odd);
    return _mm_add_epi32(whole, _mm_or_si128(above, tie));
}

Size lerp_cartesian_row_simd(const char* first, const char* second, char* destination,
                             Size width, float weight) noexcept {
    const auto zero = _mm_setzero_si128();
    const auto w = _mm_set1_ps(weight);
    Size x = 0;
    for (; x <= width - 8; x += 8) {
        // Copy eight bytes from each source without alignment or overread assumptions.
        std::int64_t first_bytes, second_bytes;
        std::memcpy(&first_bytes, first + x, sizeof(first_bytes));
        std::memcpy(&second_bytes, second + x, sizeof(second_bytes));
        const auto a16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(first_bytes), zero);
        const auto b16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(second_bytes), zero);
        const auto low = cartesian_lerp_round4(_mm_unpacklo_epi16(a16, zero), _mm_unpacklo_epi16(b16, zero), w);
        const auto high = cartesian_lerp_round4(_mm_unpackhi_epi16(a16, zero), _mm_unpackhi_epi16(b16, zero), w);
        const auto u16 = _mm_packs_epi32(low, high);
        const auto result = _mm_cvtsi128_si64(_mm_packus_epi16(u16, zero));
        std::memcpy(destination + x, &result, sizeof(result));
    }
    return x;
}
#endif

void lerp_cartesian_row(const char* first, const char* second, Size step, char* destination,
                        Size width, float weight) noexcept {
    const bool copy = first == second || weight == 0.f || weight == 1.f;
    if (copy && step == 1) {
        std::memcpy(destination, weight == 1.f ? second : first, static_cast<std::size_t>(width));
        return;
    }
    Size x = 0;
#if defined(_M_X64) || (defined(__x86_64__) && defined(__SSE2__))
    if (step == 1) {
        x = lerp_cartesian_row_simd(first, second, destination, width, weight);
    }
#endif
    auto* output = reinterpret_cast<Byte*>(destination);
    for (; x < width; ++x) {
        const auto a = static_cast<float>(static_cast<Byte>(first[x * step]));
        const auto b = static_cast<float>(static_cast<Byte>(second[x * step]));
        output[x] = round_gray8(a + weight * (b - a));
    }
}

bool cartesian_output_overlaps(const Buffer& output, const Buffer& input) noexcept {
    if (input.nbytes() == 0) return false;
    Size lowest = 0, highest = input.itemsize() - 1;
    for (int axis = 0; axis < input.ndim(); ++axis) {
        const Size extent = (input.shape(axis) - 1) * input.stride(axis);
        lowest += std::min<Size>(0, extent);
        highest += std::max<Size>(0, extent);
    }
    const auto start = reinterpret_cast<std::uintptr_t>(input.data() + lowest);
    const auto stop = reinterpret_cast<std::uintptr_t>(input.data() + highest);
    const auto output_start = reinterpret_cast<std::uintptr_t>(output.data());
    const auto output_stop = reinterpret_cast<std::uintptr_t>(output.data() + output.nbytes() - 1);
    return start <= output_stop && output_start <= stop;
}

PyObject* render_cartesian_t_lerp(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *source_input, *lo_input, *hi_input, *alpha_input, *destination_input;
        int axis;
        Size index;
        if (!PyArg_ParseTuple(args, "OOOOinO:render_cartesian_t_lerp_gray8",
                              &source_input, &lo_input, &hi_input, &alpha_input,
                              &axis, &index, &destination_input)) {
            throw PythonError{};
        }
        const Buffer source(source_input, "Cartesian source", 3);
        const Buffer lo(lo_input, "Cartesian lower time indices", 1);
        const Buffer hi(hi_input, "Cartesian upper time indices", 1);
        const Buffer alpha(alpha_input, "Cartesian time weights", 1);
        const Buffer destination(destination_input, "Cartesian native plane", 2, true);
        validate_source(source);
        lo.require<std::int64_t>();
        hi.require<std::int64_t>();
        alpha.require<float>();
        destination.require<Byte>();
        destination.require_contiguous();
        const Size depth = lo.shape(0);
        if (axis < 0 || axis > 2 || index < 0 || depth <= 0
            || hi.shape(0) != depth || alpha.shape(0) != depth
            || index >= (axis == 0 ? depth : source.shape(axis))) {
            value_error("Cartesian slice axis, index or time-map shape is invalid");
        }
        const int column_axis = axis == 2 ? 1 : 2;
        const Size height = axis == 0 ? source.shape(1) : depth;
        const Size width = source.shape(column_axis);
        if (destination.shape(0) != height || destination.shape(1) != width) {
            value_error("Cartesian native plane shape does not match the source and time map");
        }
        for (const Buffer* input : {&source, &lo, &hi, &alpha}) {
            if (cartesian_output_overlaps(destination, *input)) {
                value_error("Cartesian native plane cannot overlap the source or time map");
            }
        }
        {
            GilRelease release;
            // Retain checked controls before touching the output. In particular,
            // unchecked row pointers must never depend on rereading a map entry.
            const CartesianPair transverse = axis == 0
                ? cartesian_pair(lo, hi, alpha, index, source.shape(0)) : CartesianPair{};
            std::vector<CartesianPair> rows;
            if (axis != 0) {
                rows.reserve(depth);
                for (Size logical = 0; logical < depth; ++logical) {
                    rows.push_back(cartesian_pair(lo, hi, alpha, logical, source.shape(0)));
                }
            }
            const Size column_stride = source.stride(column_axis);
            for (Size row = 0; row < height; ++row) {
                const auto& pair = axis == 0 ? transverse : rows[row];
                const Size y = axis == 0 ? row : (axis == 1 ? index : 0);
                const Size x = axis == 2 ? index : 0;
                lerp_cartesian_row(source.ptr(pair.first, y, x), source.ptr(pair.second, y, x),
                                   column_stride, destination.ptr(row, 0), width, pair.weight);
            }
        }
        Py_RETURN_NONE;
    });
}

Byte gray8_trilinear(const Buffer& source, double time, double row, double column) {
    if (!(-1. < time && time < source.shape(0) && -1. < row && row < source.shape(1)
          && -1. < column && column < source.shape(2))) {
        return 0;
    }
    const Size t0 = static_cast<Size>(std::floor(time));
    const Size y0 = static_cast<Size>(std::floor(row));
    const Size x0 = static_cast<Size>(std::floor(column));
    const float dt = static_cast<float>(time - t0);
    const float dy = static_cast<float>(row - y0);
    const float dx = static_cast<float>(column - x0);
    float accumulator = 0.f;
    if (t0 >= 0 && t0 + 1 < source.shape(0) && y0 >= 0 && y0 + 1 < source.shape(1)
        && x0 >= 0 && x0 + 1 < source.shape(2)) {
        const Size step = source.stride(2);
        auto add_row = [&](const char* pixels, float weight) {
            accumulator = accumulator + static_cast<Byte>(pixels[0]) * (weight * (1.f - dx));
            accumulator = accumulator + static_cast<Byte>(pixels[step]) * (weight * dx);
        };
        add_row(source.ptr(t0, y0, x0), (1.f - dt) * (1.f - dy));
        add_row(source.ptr(t0, y0 + 1, x0), (1.f - dt) * dy);
        add_row(source.ptr(t0 + 1, y0, x0), dt * (1.f - dy));
        add_row(source.ptr(t0 + 1, y0 + 1, x0), dt * dy);
    } else {
        for (Size it = 0; it < 2; ++it) {
            const auto ti = t0 + it;
            const float wt = it ? dt : 1.f - dt;
            if (ti < 0 || ti >= source.shape(0)) continue;
            for (Size iy = 0; iy < 2; ++iy) {
                const auto yi = y0 + iy;
                const float wy = iy ? dy : 1.f - dy;
                if (yi < 0 || yi >= source.shape(1)) continue;
                for (Size ix = 0; ix < 2; ++ix) {
                    const auto xi = x0 + ix;
                    const float wx = ix ? dx : 1.f - dx;
                    if (xi < 0 || xi >= source.shape(2)) continue;
                    const float weight = (wt * wy) * wx;
                    const float term = static_cast<float>(source.load<Byte>(ti, yi, xi)) * weight;
                    accumulator = accumulator + term;
                }
            }
        }
    }
    // All taps are nonnegative, and the FP32 sum can exceed 255 slightly.
    const float value = std::min(255.f, accumulator);
    const auto whole = static_cast<unsigned>(value);
    const float fraction = value - static_cast<float>(whole);
    return static_cast<Byte>(whole + (fraction > .5f || (fraction == .5f && (whole & 1))));
}

struct RadialColumn {
    double x, y, offset;
};

PyObject* render_radial_gray8(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* array;
        int base, tilted, vertical;
        double radius, center_x, center_y, arc, height_origin, tangent;
        Box crop;
        if (!PyArg_ParseTuple(args, "Oidddddppdnnnn", &array, &base, &radius, &center_x, &center_y,
                              &arc, &height_origin, &tilted, &vertical, &tangent,
                              &crop.top, &crop.left, &crop.bottom, &crop.right)) return nullptr;
        const Buffer source(array, "Radial source", 3);
        validate_source(source);
        crop.validate();
        if (base < 0 || base > 2 || !std::isfinite(radius) || !std::isfinite(center_x)
            || !std::isfinite(center_y) || !std::isfinite(arc) || !std::isfinite(height_origin)
            || !std::isfinite(tangent)) {
            value_error("Radial source axes and geometry must be valid");
        }
        auto result = zeros({crop.height(), crop.width()}, "uint8");
        if (radius == 0.) return result.release();
        const Buffer output(result.get(), "Radial crop", 2, true);
        {
            GilRelease unlocked;
            std::vector<RadialColumn> columns(static_cast<std::size_t>(crop.width()));
            for (Size column = 0; column < crop.width(); ++column) {
                const double theta = positive_remainder((arc + static_cast<double>(crop.left + column)) / radius, 2. * pi);
                const double x = center_x + radius * std::cos(theta);
                const double y = center_y + radius * std::sin(theta);
                const double offset = tilted ? tangent * (vertical ? y - center_y : x - center_x) : 0.;
                columns[column] = {x, y, offset};
            }
            const Size last_height = source.shape(base) - 1;
            for (Size row = 0; row < crop.height(); ++row) {
                const double height = height_origin + static_cast<double>(crop.top + row);
                if (height < 0. || height > last_height) continue;
                for (Size column = 0; column < crop.width(); ++column) {
                    const auto& point = columns[column];
                    const double stack = tilted ? height + point.offset : height;
                    Byte value;
                    if (base == 0) value = gray8_trilinear(source, stack, point.y, point.x);
                    else if (base == 1) value = gray8_trilinear(source, point.y, stack, point.x);
                    else value = gray8_trilinear(source, point.y, point.x, stack);
                    output.store(value, row, column);
                }
            }
        }
        return result.release();
    });
}

std::array<double, 3> spherical_direction(int face, double u, double v) {
    const double inverse_sqrt_two = 1. / std::sqrt(2.);
    const double abs_u = std::abs(u), abs_v = std::abs(v);
    const int area = abs_u >= abs_v ? (u >= 0. ? 0 : 2) : (v >= 0. ? 1 : 3);
    const double p = std::max(abs_u, abs_v);
    const double minor_axis = area == 0 ? v : (area == 1 ? -u : (area == 2 ? -v : u));
    const double ratio = p == 0. ? 0. : minor_axis / p;
    const double tau = (pi / 12.) * ratio;
    const double numerator = std::sin(tau), denominator = std::cos(tau) - inverse_sqrt_two;
#ifdef _WIN32
    const double norm = _hypot(numerator, denominator);
#else
    const double norm = std::hypot(numerator, denominator);
#endif
    double cosine = denominator / norm, sine = numerator / norm;
    const bool diagonal = std::abs(ratio) == 1.;
    if (diagonal) {
        cosine = inverse_sqrt_two;
        sine = ratio > 0. ? inverse_sqrt_two : -inverse_sqrt_two;
    }
    const double d = 1. - cosine / std::sqrt(1. + cosine * cosine);
    const double normal = 1. - p * p * d;
    const double tangent = p * std::sqrt(d * (2. - p * p * d));
    double major = tangent * cosine, minor = tangent * sine;
    if (p == 1.) {
        major = normal;
        if (diagonal) minor = ratio > 0. ? normal : -normal;
    }
    const double right = area == 0 ? major : (area == 1 ? -minor : (area == 2 ? -major : minor));
    const double up = area == 0 ? minor : (area == 1 ? major : (area == 2 ? -minor : -major));
    switch (face) {
        case 0: return {normal, right, up};
        case 1: return {-right, normal, up};
        case 2: return {-normal, -right, up};
        case 3: return {right, -normal, up};
        case 4: return {-up, right, normal};
        default: return {up, right, -normal};
    }
}

Size chart_index(Size index, Size origin, Size intervals) {
    // Skip pixels beyond the chart before adding an origin near the index limit.
    return origin > intervals - index ? -1 : index + origin;
}

PyObject* render_spherical_gray8(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *array, *rotation_array;
        int face;
        Size intervals, origin_u, origin_v;
        double radius;
        Box crop;
        if (!PyArg_ParseTuple(args, "OinnnOdnnnn", &array, &face, &intervals, &origin_u, &origin_v,
                              &rotation_array, &radius, &crop.top, &crop.left, &crop.bottom, &crop.right)) return nullptr;
        const Buffer source(array, "Spherical source", 3), rotation(rotation_array, "QSC rotation", 2);
        validate_source(source);
        rotation.require<double>();
        crop.validate();
        if (face < 0 || face > 5 || intervals <= 0 || !std::isfinite(radius)
            || rotation.shape(0) != 3 || rotation.shape(1) != 3) {
            value_error("Spherical face, chart, rotation and radius must be valid");
        }
        double matrix[3][3];
        for (Size axis = 0; axis < 3; ++axis) {
            for (Size component = 0; component < 3; ++component) {
                matrix[axis][component] = rotation.load<double>(axis, component);
                if (!std::isfinite(matrix[axis][component])) value_error("Spherical rotation must be finite");
            }
        }
        auto result = zeros({crop.height(), crop.width()}, "uint8");
        const Buffer output(result.get(), "Spherical crop", 2, true);
        const double centers[] = {(source.shape(2) - 1) / 2., (source.shape(1) - 1) / 2., (source.shape(0) - 1) / 2.};
        {
            GilRelease unlocked;
            for (Size row = crop.top; row < crop.bottom; ++row) {
                const Size chart_row = chart_index(row, origin_v, intervals);
                if (chart_row < 0) continue;
                const double v = 1. - 2. * chart_row / intervals;
                for (Size column = crop.left; column < crop.right; ++column) {
                    const Size chart_column = chart_index(column, origin_u, intervals);
                    if (chart_column < 0) continue;
                    const double u = -1. + 2. * chart_column / intervals;
                    const auto direction = spherical_direction(face, u, v);
                    const double lx = direction[0] + 0., ly = direction[1] + 0., lz = direction[2] + 0.;
                    double position[3];
                    for (Size axis = 0; axis < 3; ++axis) {
                        const auto& coefficients = matrix[axis];
                        position[axis] = centers[axis] + radius * ((lx * coefficients[0] + ly * coefficients[1]) + lz * coefficients[2]);
                    }
                    output.store(gray8_trilinear(source, position[2], position[1], position[0]),
                                 row - crop.top, column - crop.left);
                }
            }
        }
        return result.release();
    });
}

struct AzimuthalMatch {
    Size index = 0;
    double distance = std::numeric_limits<double>::infinity();
    double raw = 0.;

    bool consider(double target, double source, Size candidate) noexcept {
        const double difference = target - source;
        const double wrapped = std::abs(positive_remainder(difference + 90., 180.) - 90.);
        // The exact search visits source indices in order and stops at the first NaN.
        const bool nan = std::isnan(wrapped);
        if (nan || wrapped < distance || (wrapped == distance && candidate < index)) {
            index = candidate;
            distance = wrapped;
            raw = difference;
        }
        return !nan;
    }
};

constexpr double bounded_angle_limit = 1024.;
struct AzimuthalPhase {
    double angle;
    Size index;
};

double azimuthal_phase(double angle) noexcept {
    const double phase = positive_remainder(angle, 180.);
    return phase < 180. ? phase : 0.;
}

AzimuthalMatch nearest_bounded_azimuthal_source(double target, const Buffer& sources,
                                                const std::vector<AzimuthalPhase>& phases) {
    auto lower = [&](double angle) {
        return std::lower_bound(phases.begin(), phases.end(), angle,
            [](const AzimuthalPhase& phase, double value) { return phase.angle < value; });
    };
    const double phase = azimuthal_phase(target);
    const auto forward = lower(phase);
    const double right = forward == phases.end() ? phases.front().angle + 180. : forward->angle;
    const double left = forward == phases.begin() ? phases.back().angle - 180. : (forward - 1)->angle;
    // Within the angle limit, intermediates of the legacy and phase arithmetic
    // stay below 4096 degrees. Allow 32 roundoff units at that scale for both
    // distances and the interval endpoints. The exact tie and reversal still
    // come from target - source in the original order of FP operations.
    constexpr double phase_slack = 32. * std::numeric_limits<double>::epsilon() * 4096.;
    const double reach = std::min(phase - left, right - phase) + phase_slack;
    const double first = phase - reach, last = phase + reach;
    AzimuthalMatch best;
    auto search_range = [&](double start, double stop) {
        for (auto item = lower(start); item != phases.end() && item->angle <= stop; ++item) {
            best.consider(target, sources.load<double>(item->index), item->index);
        }
    };
    search_range(first, last);
    if (first < 0.) search_range(first + 180., 180.);
    if (last >= 180.) search_range(0., last - 180.);
    return best;
}

PyObject* nearest_azimuthal_sources(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *target_array, *source_array;
        if (!PyArg_ParseTuple(args, "OO", &target_array, &source_array)) return nullptr;
        const Buffer targets(target_array, "Azimuthal targets", 1), sources(source_array, "Azimuthal sources", 1);
        targets.require<double>();
        sources.require<double>();
        if (!sources.shape(0)) value_error("No azimuthal source angles are available for backprojection");
        if (sources.shape(0) > std::numeric_limits<std::int32_t>::max()) {
            value_error("Azimuthal trajectory exceeds int32 source addressing");
        }
        auto owner_array = empty({targets.shape(0)}, "int32");
        auto reverse_array = empty({targets.shape(0)}, "bool");
        const Buffer owners(owner_array.get(), "Azimuthal owners", 1, true);
        const Buffer reverses(reverse_array.get(), "Azimuthal reversals", 1, true);
        {
            GilRelease unlocked;
            std::vector<AzimuthalPhase> phases;
            if (sources.shape(0) >= 8 && targets.shape(0) >= 8) {
                for (Size index = 0; index < sources.shape(0); ++index) {
                    const double angle = sources.load<double>(index);
                    if (!(std::abs(angle) <= bounded_angle_limit)) {
                        phases.clear();
                        break;
                    }
                    phases.push_back({azimuthal_phase(angle), index});
                }
                std::sort(phases.begin(), phases.end(),
                    [](const AzimuthalPhase& left, const AzimuthalPhase& right) {
                        return left.angle < right.angle;
                    });
            }
            for (Size target_index = 0; target_index < targets.shape(0); ++target_index) {
                const double target = targets.load<double>(target_index);
                AzimuthalMatch best;
                if (!phases.empty() && std::abs(target) <= bounded_angle_limit) {
                    best = nearest_bounded_azimuthal_source(target, sources, phases);
                } else {
                    for (Size source_index = 0; source_index < sources.shape(0); ++source_index) {
                        if (!best.consider(target, sources.load<double>(source_index), source_index)) break;
                    }
                }
                owners.store(static_cast<std::int32_t>(best.index), target_index);
                reverses.store(std::abs(best.raw) > 90., target_index);
            }
        }
        return PyTuple_Pack(2, owner_array.get(), reverse_array.get());
    });
}

PyMethodDef methods[] = {
    {"render_cartesian_t_lerp_gray8", render_cartesian_t_lerp, METH_VARARGS, nullptr},
    {"render_radial_gray8", render_radial_gray8, METH_VARARGS, nullptr},
    {"render_spherical_gray8", render_spherical_gray8, METH_VARARGS, nullptr},
    {"nearest_azimuthal_sources", nearest_azimuthal_sources, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_rendering(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
