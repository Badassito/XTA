#include "cpu_kernels.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Pixel = std::uint8_t;
using Coord = std::int64_t;

struct Point {
    Coord x;
    Coord y;
};

Point point(const Buffer& points, Index index) noexcept {
    return {points.load<Coord>(index, 0), points.load<Coord>(index, 1)};
}

bool inside(Point p, const Buffer& foreground) noexcept {
    return p.x >= 0 && p.x < foreground.shape(1) && p.y >= 0 && p.y < foreground.shape(0);
}

bool line8_foreground(Point first, Point last, const Buffer& foreground) noexcept {
    // cv2.line's LINE_8 path traverses left to right even on steep lines.
    // Reversing the traversal without swapping the endpoints' tie rule
    // selects a different midpoint when the error is exactly zero.
    if (first.x > last.x) std::swap(first, last);
    const auto dx = last.x - first.x;
    const auto dy = last.y >= first.y ? last.y - first.y : first.y - last.y;
    const auto ystep = last.y >= first.y ? Coord{1} : Coord{-1};
    const bool steep = dy > dx;
    const auto major = steep ? dy : dx;
    const auto minor = steep ? dx : dy;
    auto error = major - 2 * minor;
    auto x = first.x;
    auto y = first.y;
    for (Coord step = 0;; ++step) {
        // Treat bool exports as bytes as well: any nonzero pixel is foreground.
        if (foreground.load<Pixel>(y, x) == 0) return false;
        if (step == major) return true;
        const bool crossed = error < 0;
        error += (crossed ? 2 * major : 0) - 2 * minor;
        if (!steep || crossed) ++x;
        if (steep || crossed) y += ystep;
    }
}

PyObject* first_foreground_bridge(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject *mask_object, *holes_object, *points_object, *starts_object, *targets_object;
        if (!PyArg_ParseTuple(args, "OOOOO:polygon_first_foreground_bridge", &mask_object, &holes_object,
                              &points_object, &starts_object, &targets_object)) return nullptr;
        const Buffer mask(mask_object, "foreground", 2), hole(holes_object, "hole vertices", 2);
        const Buffer points(points_object, "ring vertices", 2), starts(starts_object, "hole indices", 1);
        const Buffer targets(targets_object, "ring indices", 1);
        if (!mask.is<Pixel>() && !mask.is<bool>())
            type_error("Polygon foreground must be uint8 or bool");
        hole.require<Coord>();
        points.require<Coord>();
        starts.require<Index>();
        targets.require<Index>();
        if (hole.shape(1) != 2 || points.shape(1) != 2 || starts.shape(0) != targets.shape(0))
            value_error("Invalid polygon bridge candidate layout");
        Index result = -1;
        {
            GilRelease unlocked;
            for (Index rank = 0; rank < starts.shape(0); ++rank) {
                const auto j = starts.load<Index>(rank), c = targets.load<Index>(rank);
                if (j < 0 || j >= hole.shape(0) || c < 0 || c >= points.shape(0))
                    throw std::invalid_argument("Polygon bridge vertex index is out of bounds");
                const auto first = point(hole, j), last = point(points, c);
                if (!inside(first, mask) || !inside(last, mask) ||
                    std::max(first.x, last.x) - std::min(first.x, last.x) > std::numeric_limits<int>::max() / 2 ||
                    std::max(first.y, last.y) - std::min(first.y, last.y) > std::numeric_limits<int>::max() / 2) {
                    // Preserve the existing OpenCV behavior on unusual inputs.
                    result = -2;
                    break;
                }
                if (line8_foreground(first, last, mask)) {
                    result = rank;
                    break;
                }
            }
        }
        return PyLong_FromSsize_t(result);
    });
}

void format_axis(char* output, Coord coordinate, Index size) {
    if (coordinate < 0 || coordinate >= size)
        throw std::out_of_range("Polygon coordinate lies outside the label frame");
    const double normalized = static_cast<double>(coordinate) / static_cast<double>(size);
    const auto result = std::to_chars(output, output + 8, normalized, std::chars_format::fixed, 6);
    if (result.ec != std::errc{} || result.ptr != output + 8)
        throw std::runtime_error("Could not format a normalized polygon coordinate");
}

PyObject* format_yolo_polygon_line(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        PyObject* points_object;
        Index width, height;
        if (!PyArg_ParseTuple(args, "Onn:format_yolo_polygon_line", &points_object, &width, &height))
            return nullptr;
        const Buffer points(points_object, "polygon vertices", 2);
        points.require<Coord>();
        if (points.shape(1) != 2 || width <= 0 || height <= 0) value_error("Invalid polygon label frame");
        const auto count = points.shape(0);
        if (count == 0) return PyUnicode_FromStringAndSize("0 ", 2);
        const auto length = checked_add(1, checked_extent(count, 18));
        std::string line(static_cast<std::size_t>(length), ' ');
        line[0] = '0';
        {
            GilRelease unlocked;
            for (Index i = 0; i < count; ++i) {
                auto* output = line.data() + 2 + 18 * i;
                format_axis(output, points.load<Coord>(i, 0), width);
                format_axis(output + 9, points.load<Coord>(i, 1), height);
            }
        }
        return PyUnicode_FromStringAndSize(line.data(), length);
    });
}

PyMethodDef methods[] = {
    {"polygon_first_foreground_bridge", first_foreground_bridge, METH_VARARGS, nullptr},
    {"format_yolo_polygon_line", format_yolo_polygon_line, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_polygons(PyObject* module) { return PyModule_AddFunctions(module, methods); }

} // namespace xta::cpu
