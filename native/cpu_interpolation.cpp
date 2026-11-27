#include "cpu_kernels.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace xta::cpu {
namespace {

using Index = Py_ssize_t;
using Signed = std::int64_t;
using Unsigned = std::uint64_t;
using Byte = std::uint8_t;

struct Point {
    Index y = -1;
    Index x = -1;
};

void arguments(PyObject* args, Index count) {
    if (PyTuple_GET_SIZE(args) != count) {
        type_error("Incorrect number of interpolation kernel arguments");
    }
}

PyObject* arg(PyObject* args, Index index) {
    return PyTuple_GET_ITEM(args, index);
}

Signed integer(PyObject* args, Index index) {
    const auto value = PyLong_AsLongLong(arg(args, index));
    if (value == -1 && PyErr_Occurred()) {
        throw PythonError{};
    }
    return static_cast<Signed>(value);
}

double real(PyObject* args, Index index) {
    const double value = PyFloat_AsDouble(arg(args, index));
    if (value == -1.0 && PyErr_Occurred()) {
        throw PythonError{};
    }
    return value;
}

bool flag(PyObject* args, Index index) {
    const int result = PyObject_IsTrue(arg(args, index));
    if (result < 0) {
        throw PythonError{};
    }
    return result != 0;
}

template<class Type>
struct Dtype {
    using type = Type;
};

template<class Function>
decltype(auto) integer_dtype(const Buffer& buffer, Function&& call) {
    if (buffer.is<std::uint8_t>()) return call(Dtype<std::uint8_t>{});
    if (buffer.is<std::uint16_t>()) return call(Dtype<std::uint16_t>{});
    if (buffer.is<std::uint32_t>()) return call(Dtype<std::uint32_t>{});
    if (buffer.is<std::uint64_t>()) return call(Dtype<std::uint64_t>{});
    if (buffer.is<std::int8_t>()) return call(Dtype<std::int8_t>{});
    if (buffer.is<std::int16_t>()) return call(Dtype<std::int16_t>{});
    if (buffer.is<std::int32_t>()) return call(Dtype<std::int32_t>{});
    if (buffer.is<std::int64_t>()) return call(Dtype<std::int64_t>{});
    type_error("Interpolation requires a native-endian integer buffer");
}

void require_mask(const Buffer& mask) {
    if (!mask.is<bool>() && !mask.is<Byte>()) {
        type_error("Interpolation mask must have bool or uint8 dtype");
    }
}

Signed difference(Signed left, Signed right) {
    if ((right > 0 && left < std::numeric_limits<Signed>::min() + right)
        || (right < 0 && left > std::numeric_limits<Signed>::max() + right)) {
        throw std::overflow_error("Interpolation coordinate difference overflow");
    }
    return left - right;
}

Signed signed_bits(Unsigned bits) {
    Signed result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// The original distance calculation is signed int64 arithmetic. Do its
// products in unsigned arithmetic to retain the wrapped bit pattern.
Signed squared_distance(Index y, Index x, Signed ref_y, Signed ref_x) {
    const Unsigned dy = static_cast<Unsigned>(y) - static_cast<Unsigned>(ref_y);
    const Unsigned dx = static_cast<Unsigned>(x) - static_cast<Unsigned>(ref_x);
    return signed_bits(dy * dy + dx * dx);
}

Point nearest_true(const Buffer& mask, Signed ref_y, Signed ref_x, bool center) {
    Point best;
    Signed best_distance = 0;
    for (Index y = 0; y < mask.shape(0); ++y) {
        for (Index x = 0; x < mask.shape(1); ++x) {
            if (!mask.load<Byte>(y, x)) {
                continue;
            }
            const Signed distance = squared_distance(y, x, ref_y, ref_x);
            if (best.y < 0 || (center && best_distance < 0) || distance < best_distance) {
                best = {y, x};
                best_distance = distance;
            }
        }
    }
    return best;
}

PyObject* component_crop_anchor(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 4);
        const Buffer labels(arg(args, 0), "component labels", 2);
        const Signed label = integer(args, 1);
        const float centroid_y = static_cast<float>(real(args, 2));
        const float centroid_x = static_cast<float>(real(args, 3));
        auto mask = empty({labels.shape(0), labels.shape(1)}, "bool");
        const Buffer output(mask.get(), "component crop", 2, true);
        Point first, best;
        float best_distance = 0;
        integer_dtype(labels, [&](auto dtype) {
            using Label = typename decltype(dtype)::type;
            GilRelease release;
            for (Index y = 0; y < labels.shape(0); ++y) {
                const float dy = static_cast<float>(y) - centroid_y;
                for (Index x = 0; x < labels.shape(1); ++x) {
                    const bool member = static_cast<Unsigned>(labels.load<Label>(y, x)) == static_cast<Unsigned>(label);
                    output.store<bool>(member, y, x);
                    if (!member) {
                        continue;
                    }
                    const float dx = static_cast<float>(x) - centroid_x;
                    const float distance = dy * dy + dx * dx;
                    if (first.y < 0) {
                        first = best = {y, x};
                        best_distance = distance;
                    } else if (distance < best_distance) {
                        best = {y, x};
                        best_distance = distance;
                    }
                }
            }
        });
        return Py_BuildValue("Onnnn", mask.get(), first.y, first.x, best.y, best.x);
    });
}

PyObject* nearest_true_pixel(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 3);
        const Buffer mask(arg(args, 0), "nearest-pixel mask", 2);
        require_mask(mask);
        const Signed ref_y = integer(args, 1), ref_x = integer(args, 2);
        Point best;
        {
            GilRelease release;
            best = nearest_true(mask, ref_y, ref_x, false);
        }
        return Py_BuildValue("nni", best.y, best.x, static_cast<int>(best.y >= 0));
    });
}

PyObject* dilated_overlap_count(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 10);
        const Buffer previous(arg(args, 0), "previous mask", 2);
        const Buffer candidate(arg(args, 3), "candidate mask", 2);
        require_mask(previous);
        require_mask(candidate);
        const Signed py0 = integer(args, 1), px0 = integer(args, 2);
        const Signed cy0 = integer(args, 4), cx0 = integer(args, 5);
        const Signed y0 = integer(args, 6), x0 = integer(args, 7);
        const Signed y1 = integer(args, 8), x1 = integer(args, 9);
        if (y1 <= y0 || x1 <= x0) {
            return PyLong_FromLong(0);
        }
        if (difference(y0, cy0) < 0 || difference(x0, cx0) < 0
            || difference(y1, cy0) > candidate.shape(0)
            || difference(x1, cx0) > candidate.shape(1)) {
            value_error("Interpolation overlap is outside its candidate mask");
        }
        const Index prev_first_y = difference(y0, py0);
        const Index prev_first_x = difference(x0, px0);
        const Index rows = difference(y1, y0), cols = difference(x1, x0);
        checked_add(prev_first_y, -1);
        checked_add(prev_first_x, -1);
        checked_add(prev_first_y, rows);
        checked_add(prev_first_x, cols);
        Index count = 0;
        {
            GilRelease release;
            for (Index y = 0; y < rows; ++y) {
                for (Index x = 0; x < cols; ++x) {
                    if (!candidate.load<Byte>(y0 - cy0 + y, x0 - cx0 + x)) {
                        continue;
                    }
                    bool hit = false;
                    for (Index dy = -1; dy <= 1 && !hit; ++dy) {
                        const Index py = prev_first_y + y + dy;
                        if (py < 0 || py >= previous.shape(0)) {
                            continue;
                        }
                        for (Index dx = -1; dx <= 1; ++dx) {
                            const Index px = prev_first_x + x + dx;
                            if (0 <= px && px < previous.shape(1) && previous.load<Byte>(py, px)) {
                                hit = true;
                                break;
                            }
                        }
                    }
                    count += hit;
                }
            }
        }
        return PyLong_FromSsize_t(count);
    });
}

void flood_center(const Buffer& mask, const Buffer& output, bool fill_holes) {
    const Index height = mask.shape(0), width = mask.shape(1);
    if (!height || !width) {
        return;
    }
    const Point center{height / 2, width / 2};
    const Point seed = mask.load<Byte>(center.y, center.x)
        ? center : nearest_true(mask, center.y, center.x, true);
    if (seed.y < 0) {
        return;
    }
    if (height > std::numeric_limits<std::int32_t>::max()
        || width > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("Interpolation flood exceeds coordinate workspace");
    }
    const Index area = checked_extent(height, width);
    struct FloodPoint { std::int32_t y, x; };
    // Preserve the existing one-byte state and eight-byte coordinate stack.
    std::vector<Byte> state(static_cast<std::size_t>(area), 0);
    std::unique_ptr<FloodPoint[]> stack(new FloodPoint[area]);
    Index used = 0;
    const auto push = [&](Index y, Index x, Byte mark) {
        state[y * width + x] = mark;
        stack[used++] = {static_cast<std::int32_t>(y), static_cast<std::int32_t>(x)};
    };
    push(seed.y, seed.x, 1);
    while (used) {
        const auto [y, x] = stack[--used];
        for (Index dy = -1; dy <= 1; ++dy) {
            const Index ny = y + dy;
            if (ny < 0 || ny >= height) {
                continue;
            }
            for (Index dx = -1; dx <= 1; ++dx) {
                const Index nx = x + dx;
                if (0 <= nx && nx < width && state[ny * width + nx] == 0
                    && mask.load<Byte>(ny, nx)) {
                    push(ny, nx, 1);
                }
            }
        }
    }

    if (fill_holes) {
        const auto outside = [&](Index y, Index x) {
            if (state[y * width + x] == 0) {
                push(y, x, 2);
            }
        };
        for (Index x = 0; x < width; ++x) {
            outside(0, x);
            outside(height - 1, x);
        }
        for (Index y = 0; y < height; ++y) {
            outside(y, 0);
            outside(y, width - 1);
        }
        while (used) {
            const auto [y, x] = stack[--used];
            if (y > 0) outside(y - 1, x);
            if (y + 1 < height) outside(y + 1, x);
            if (x > 0) outside(y, x - 1);
            if (x + 1 < width) outside(y, x + 1);
        }
    }
    for (Index y = 0; y < height; ++y) {
        for (Index x = 0; x < width; ++x) {
            const auto mark = state[y * width + x];
            output.store<bool>(fill_holes ? mark != 2 : mark == 1, y, x);
        }
    }
}

PyObject* keep_center_fill(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 2);
        const Buffer mask(arg(args, 0), "center mask", 2);
        require_mask(mask);
        const bool fill_holes = flag(args, 1);
        auto result = zeros({mask.shape(0), mask.shape(1)}, "bool");
        const Buffer output(result.get(), "center component", 2, true);
        {
            GilRelease release;
            flood_center(mask, output, fill_holes);
        }
        return result.release();
    });
}


template<bool Packed, class Word>
Index paste_mask(const Buffer& destination, const Buffer& local, Index dy0, Index dx0,
                 Index sy0, Index sy1, Index sx0, Index sx1, Word value) {
    GilRelease release;
    Index added = 0;
    for (Index y = sy0; y < sy1; ++y) {
        const Index dy = dy0 + (y - sy0);
        for (Index x = sx0; x < sx1; ++x) {
            if (!local.load<Byte>(y, x)) {
                continue;
            }
            const Index dx = dx0 + (x - sx0);
            const Word previous = destination.load<Word>(dy, dx);
            if constexpr (Packed) {
                added += (previous & value) == 0;
                // Composite paint values still set all of their bits on a partial hit.
                destination.store<Word>(static_cast<Word>(previous | value), dy, dx);
            } else if (!previous) {
                destination.store<Word>(Word{1}, dy, dx);
                ++added;
            }
        }
    }
    return added;
}

template<bool Packed>
PyObject* paste_or(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, Packed ? 9 : 8);
        const Buffer destination(arg(args, 0), "interpolation destination", 2, true);
        const Buffer local(arg(args, 1), "interpolation source", 2);
        require_mask(local);
        const Index dy0 = integer(args, 2), dx0 = integer(args, 3);
        const Index sy0 = integer(args, 4), sy1 = integer(args, 5);
        const Index sx0 = integer(args, 6), sx1 = integer(args, 7);
        if (sy1 <= sy0 || sx1 <= sx0) {
            return PyLong_FromLong(0);
        }
        if (std::min({dy0, dx0, sy0, sx0}) < 0 || sy1 > local.shape(0) || sx1 > local.shape(1)
            || dy0 > destination.shape(0) || sy1 - sy0 > destination.shape(0) - dy0
            || dx0 > destination.shape(1) || sx1 - sx0 > destination.shape(1) - dx0) {
            value_error("Interpolation paste is outside its source or destination");
        }
        if (destination.is<bool>()) {
            const bool value = !Packed || flag(args, 8);
            return PyLong_FromSsize_t(
                paste_mask<Packed>(destination, local, dy0, dx0, sy0, sy1, sx0, sx1, value));
        }
        Unsigned value = 1;
        if constexpr (Packed) {
            Object index(PyNumber_Index(arg(args, 8)));
            value = PyLong_AsUnsignedLongLong(index.get());
            if (PyErr_Occurred()) {
                throw PythonError{};
            }
        }
        return integer_dtype(destination, [&](auto dtype) -> PyObject* {
            using Word = typename decltype(dtype)::type;
            if constexpr (std::is_unsigned_v<Word>) {
                return PyLong_FromSsize_t(paste_mask<Packed>(
                    destination, local, dy0, dx0, sy0, sy1, sx0, sx1, static_cast<Word>(value)));
            } else {
                type_error("Interpolation paste destination must have bool or unsigned integer dtype");
            }
        });
    });
}

struct ProjectionWalk {
    Index slices;
    Index width;
    Index crop_y;
    Index crop_x;
    Index source_slice;
    Signed anchor_y;
    Signed anchor_x;
    Signed seed_label;
    Index direction;
    Index max_steps;
    double slope;
    Index max_candidates;
    bool wrap;
    bool negative_angle;
    Index capacity;
};

struct Candidate {
    Signed label;
    Signed y;
    Signed x;
    Signed distance;
};

struct ProjectionCount {
    Index found = 0;
    int overflow = 0;
};

template<class Label, bool Sparse>
ProjectionCount project_candidates(
    const Buffer& labels, const Buffer& offsets, const Buffer& boxes,
    const Buffer& lut, const Buffer& lut_offsets, const Buffer& sdf,
    const ProjectionWalk& walk, const std::array<Buffer, 6>& output) {
    const bool use_lut = lut_offsets.shape(0) == walk.slices;
    std::unique_ptr<Candidate[]> step_candidates(new Candidate[walk.capacity]);
    ProjectionCount count;
    for (Index step = 1; step <= walk.max_steps; ++step) {
        Index slice = walk.source_slice + walk.direction * step;
        const bool mirrored = slice < 0 || slice >= walk.slices;
        if (mirrored) {
            if (!walk.wrap) {
                break;
            }
            slice %= walk.slices;
            if (slice < 0) slice += walk.slices;
        }

        Index sy0 = 0, sy1 = 0, sx0 = 0, sx1 = 0, start = 0;
        if constexpr (Sparse) {
            sy0 = boxes.load<Signed>(slice, 0);
            sy1 = boxes.load<Signed>(slice, 1);
            sx0 = boxes.load<Signed>(slice, 2);
            sx1 = boxes.load<Signed>(slice, 3);
            // Empty slices may use a sentinel offset and must not read the arena.
            if (sy1 > sy0 && sx1 > sx0) {
                start = offsets.load<Signed>(slice);
                const Index area = checked_extent(difference(sy1, sy0), difference(sx1, sx0));
                if (start < 0 || start > labels.shape(0) || area > labels.shape(0) - start) {
                    throw std::invalid_argument("Interpolation sparse label slice exceeds its arena");
                }
            }
        }

        const double threshold = -walk.slope * static_cast<double>(step);
        Index step_count = 0;
        bool any_projection = false;
        for (Index y = 0; y < sdf.shape(0); ++y) {
            const Index gy = walk.crop_y + y;
            for (Index x = 0; x < sdf.shape(1); ++x) {
                if (sdf.load<float>(y, x) < threshold) {
                    continue;
                }
                any_projection = true;
                const Index gx = walk.crop_x + x;
                const Index sample_x = mirrored ? walk.width - 1 - gx : gx;
                Signed raw;
                if constexpr (Sparse) {
                    if (gy < sy0 || gy >= sy1 || sample_x < sx0 || sample_x >= sx1) {
                        continue;
                    }
                    raw = static_cast<Signed>(labels.load<Label>(
                        start + (gy - sy0) * (sx1 - sx0) + (sample_x - sx0)));
                } else {
                    raw = static_cast<Signed>(labels.load<Label>(slice, gy, sample_x));
                }
                if (raw <= 0) {
                    continue;
                }
                Signed target = raw;
                if (use_lut) {
                    const Index at = checked_add(lut_offsets.load<Signed>(slice), raw);
                    if (at < 0 || at >= lut.shape(0)) {
                        throw std::invalid_argument("Interpolation local label exceeds its LUT");
                    }
                    target = lut.load<std::uint32_t>(at);
                }
                if (target <= 0 || target == walk.seed_label) {
                    continue;
                }

                Index prior = 0;
                while (prior < count.found && output[0].load<Signed>(prior) != target) {
                    ++prior;
                }
                if (prior < count.found) {
                    continue;
                }
                Index local = 0;
                while (local < step_count && step_candidates[local].label != target) {
                    ++local;
                }
                // The returned x is mirrored when needed, but distance is in
                // the original projection coordinates. Equal distances retain
                // the first pixel in traversal order.
                const Signed distance = squared_distance(gy, gx, walk.anchor_y, walk.anchor_x);
                if (local == step_count) {
                    if (step_count >= walk.capacity) {
                        count.overflow = 1;
                        return count;
                    }
                    step_candidates[step_count++] = {target, gy, sample_x, distance};
                } else if (distance < step_candidates[local].distance) {
                    step_candidates[local] = {target, gy, sample_x, distance};
                }
            }
        }
        if (!any_projection) {
            if (walk.negative_angle) {
                break;
            }
            continue;
        }
        for (Index local = 0; local < step_count; ++local) {
            if (count.found >= walk.capacity) {
                count.overflow = 1;
                return count;
            }
            const auto& candidate = step_candidates[local];
            output[0].store<Signed>(candidate.label, count.found);
            output[1].store<Signed>(slice, count.found);
            output[2].store<Signed>(candidate.y, count.found);
            output[3].store<Signed>(candidate.x, count.found);
            output[4].store<Signed>(step, count.found);
            output[5].store<Signed>(candidate.distance, count.found);
            ++count.found;
        }
        if (count.found >= walk.max_candidates) {
            break;
        }
    }
    return count;
}


PyObject* find_projection_candidates(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 23);
        const Buffer dense(arg(args, 0), "dense interpolation labels", 3);
        const Buffer sparse(arg(args, 1), "sparse interpolation labels", 1);
        const Buffer offsets(arg(args, 2), "sparse interpolation offsets", 1);
        const Buffer boxes(arg(args, 3), "sparse interpolation boxes", 2);
        const bool sparse_mode = flag(args, 4);
        const Buffer lut(arg(args, 7), "interpolation label LUT", 1);
        const Buffer lut_offsets(arg(args, 8), "interpolation LUT offsets", 1);
        const Buffer sdf(arg(args, 9), "interpolation signed distance field", 2);
        offsets.require<Signed>();
        boxes.require<Signed>();
        lut.require<std::uint32_t>();
        lut_offsets.require<Signed>();
        sdf.require<float>();
        const ProjectionWalk walk{
            integer(args, 5), integer(args, 6), integer(args, 10), integer(args, 11),
            integer(args, 12), integer(args, 13), integer(args, 14), integer(args, 15),
            integer(args, 16), integer(args, 17), real(args, 18), integer(args, 19),
            flag(args, 20), flag(args, 21), integer(args, 22),
        };
        if (walk.slices <= 0 || walk.width < 0 || walk.crop_y < 0 || walk.crop_x < 0
            || walk.source_slice < 0 || walk.source_slice >= walk.slices || walk.capacity < 0
            || (walk.direction != -1 && walk.direction != 1)
            || walk.crop_x > walk.width || sdf.shape(1) > walk.width - walk.crop_x) {
            value_error("Invalid interpolation projection geometry");
        }
        if (walk.max_steps >= std::numeric_limits<Index>::max()) {
            overflow_error("Interpolation projection step count exceeds its index range");
        }
        if (walk.wrap && walk.direction > 0 && walk.max_steps > 0) {
            checked_add(walk.source_slice, walk.max_steps);
        }
        checked_add(walk.crop_y, sdf.shape(0));
        if (sparse_mode) {
            if (offsets.shape(0) < walk.slices || boxes.shape(0) != walk.slices || boxes.shape(1) != 4) {
                value_error("Invalid sparse interpolation metadata shape");
            }
        } else if (dense.shape(0) != walk.slices || dense.shape(2) != walk.width
            || walk.crop_y > dense.shape(1) || sdf.shape(0) > dense.shape(1) - walk.crop_y) {
            value_error("Interpolation projection crop is outside the dense label volume");
        }
        std::array<Object, 6> result{
            zeros({walk.capacity}, "int64"), zeros({walk.capacity}, "int64"),
            zeros({walk.capacity}, "int64"), zeros({walk.capacity}, "int64"),
            zeros({walk.capacity}, "int64"), zeros({walk.capacity}, "int64"),
        };
        const std::array<Buffer, 6> output{{
            {result[0].get(), "candidate labels", 1, true},
            {result[1].get(), "candidate slices", 1, true},
            {result[2].get(), "candidate rows", 1, true},
            {result[3].get(), "candidate columns", 1, true},
            {result[4].get(), "candidate steps", 1, true},
            {result[5].get(), "candidate distances", 1, true},
        }};
        const Buffer& labels = sparse_mode ? sparse : dense;
        const auto count = integer_dtype(labels, [&](auto dtype) {
            using Label = typename decltype(dtype)::type;
            GilRelease release;
            if (sparse_mode) {
                return project_candidates<Label, true>(labels, offsets, boxes, lut, lut_offsets, sdf, walk, output);
            }
            return project_candidates<Label, false>(labels, offsets, boxes, lut, lut_offsets, sdf, walk, output);
        });
        return Py_BuildValue("OOOOOOni", result[0].get(), result[1].get(), result[2].get(),
                             result[3].get(), result[4].get(), result[5].get(), count.found, count.overflow);
    });
}

struct EncodedCrop {
    Signed z;
    Signed y0;
    Signed y1;
    Signed x0;
    Signed x1;
    Signed foreground;
    Signed offset;
    Signed size;

    EncodedCrop(const Buffer& fields, Index row)
        : z(fields.load<Signed>(row, 0)),
          y0(fields.load<Signed>(row, 1)),
          y1(fields.load<Signed>(row, 2)),
          x0(fields.load<Signed>(row, 3)),
          x1(fields.load<Signed>(row, 4)),
          foreground(fields.load<Signed>(row, 5)),
          offset(fields.load<Signed>(row, 6)),
          size(fields.load<Signed>(row, 7)) {}
};

void require_encoded_fields(const Buffer& fields) {
    fields.require<Signed>();
    if (fields.shape(1) != 8) {
        value_error("Encoded interpolation block requires eight fields per record");
    }
}

int validate_encoded(const Buffer& fields, const Buffer& data, Signed first,
                     Signed height, Signed width, bool packed) {
    Signed cursor = 0;
    for (Index i = 0; i < fields.shape(0); ++i) {
        const EncodedCrop crop(fields, i);
        if (first > std::numeric_limits<Signed>::max() - i || crop.z != first + i
            || crop.y0 < 0 || crop.y0 > crop.y1 || crop.y1 > height
            || crop.x0 < 0 || crop.x0 > crop.x1 || crop.x1 > width) {
            return 1;
        }
        const Signed rows = crop.y1 - crop.y0;
        const Signed columns = crop.x1 - crop.x0;
        if (rows && columns > std::numeric_limits<Signed>::max() / rows) {
            return 2;
        }
        const Signed area = rows * columns;
        const Signed tail = columns % 8;
        const Signed row_bytes = packed ? columns / 8 + (tail != 0) : columns;
        if (crop.offset != cursor || crop.size != rows * row_bytes
            || crop.foreground < 0 || crop.foreground > area
            || (!crop.foreground && (crop.y0 || crop.y1 || crop.x0 || crop.x1))) {
            return 2;
        }
        if (crop.size > data.shape(0) - cursor) {
            return 3;
        }
        cursor += crop.size;
        if (packed && crop.size && tail) {
            const Byte tail_mask = static_cast<Byte>(0xffu << tail);
            for (Signed row = 0; row < rows; ++row) {
                const Signed last = crop.offset + row * row_bytes + (row_bytes - 1);
                if (data.load<Byte>(last) & tail_mask) {
                    return 4;
                }
            }
        }
    }
    return cursor == data.shape(0) ? 0 : 5;
}

PyObject* validate_encoded_block(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 6);
        const Buffer fields(arg(args, 0), "encoded interpolation fields", 2);
        const Buffer data(arg(args, 1), "encoded interpolation data", 1);
        require_encoded_fields(fields);
        data.require<Byte>();
        const Signed first = integer(args, 2), height = integer(args, 3), width = integer(args, 4);
        const bool packed = flag(args, 5);
        int status;
        {
            GilRelease release;
            status = validate_encoded(fields, data, first, height, width, packed);
        }
        return PyLong_FromLong(status);
    });
}

// CTILE_INDEX_DTYPE on disk uses this packed, little-endian layout.
constexpr Index index_bytes = 48;
constexpr Index kind_offset = 0, payload_offset = 8, payload_size = 16;
constexpr Index y0_offset = 24, x0_offset = 28, y1_offset = 32, x1_offset = 36;
constexpr Index payload_unpacked_bytes = 40;

template<class Field>
void store_little_endian(char* record, Index offset, Field value) {
    for (Index byte = 0; byte < static_cast<Index>(sizeof(Field)); ++byte) {
        record[offset + byte] = static_cast<char>(static_cast<Byte>(value >> (byte * 8)));
    }
}

PyObject* commit_encoded_block(PyObject*, PyObject* args) {
    return protect([&]() -> PyObject* {
        arguments(args, 4);
        const Buffer fields(arg(args, 0), "encoded interpolation fields", 2);
        const Buffer index(arg(args, 1), "encoded interpolation index", 2, true);
        const Buffer state(arg(args, 2), "encoded interpolation slice state", 1, true);
        require_encoded_fields(fields);
        index.require<Byte>();
        state.require<Byte>();
        if (index.shape(0) != state.shape(0) || index.shape(1) != index_bytes || index.stride(1) != 1) {
            value_error("Encoded interpolation index has an invalid layout");
        }
        const Signed base_offset = integer(args, 3);
        auto stats = empty({8}, "int64");
        const Buffer output(stats.get(), "encoded interpolation statistics", 1, true);
        constexpr Signed maximum = std::numeric_limits<Signed>::max();
        std::array<Signed, 8> sums{0, 0, maximum, -1, maximum, -1, maximum, -1};
        {
            GilRelease release;
            for (Index i = 0; i < fields.shape(0); ++i) {
                const EncodedCrop crop(fields, i);
                if (crop.z < 0 || crop.z >= index.shape(0)) {
                    throw std::invalid_argument("Encoded interpolation slice exceeds its index");
                }
                if (crop.foreground) {
                    char* record = index.ptr(crop.z, 0);
                    record[kind_offset] = 1;
                    store_little_endian(record, payload_offset,
                                        static_cast<Unsigned>(base_offset) + static_cast<Unsigned>(crop.offset));
                    store_little_endian(record, payload_size, static_cast<Unsigned>(crop.size));
                    store_little_endian(record, y0_offset, static_cast<std::uint32_t>(crop.y0));
                    store_little_endian(record, x0_offset, static_cast<std::uint32_t>(crop.x0));
                    store_little_endian(record, y1_offset, static_cast<std::uint32_t>(crop.y1));
                    store_little_endian(record, x1_offset, static_cast<std::uint32_t>(crop.x1));
                    store_little_endian(record, payload_unpacked_bytes,
                                        (static_cast<Unsigned>(crop.y1) - static_cast<Unsigned>(crop.y0))
                                        * (static_cast<Unsigned>(crop.x1) - static_cast<Unsigned>(crop.x0)));
                    ++sums[0];
                    sums[1] = signed_bits(static_cast<Unsigned>(sums[1]) + static_cast<Unsigned>(crop.foreground));
                    sums[2] = std::min(sums[2], crop.z);
                    sums[3] = std::max(sums[3], crop.z);
                    sums[4] = std::min(sums[4], crop.y0);
                    sums[5] = std::max(sums[5], signed_bits(static_cast<Unsigned>(crop.y1) - 1));
                    sums[6] = std::min(sums[6], crop.x0);
                    sums[7] = std::max(sums[7], signed_bits(static_cast<Unsigned>(crop.x1) - 1));
                }
                state.store<Byte>(2, crop.z);
            }
            for (Index i = 0; i < static_cast<Index>(sums.size()); ++i) {
                output.store<Signed>(sums[i], i);
            }
        }
        return stats.release();
    });
}

PyMethodDef methods[] = {
    {"interpolation_component_crop_anchor", component_crop_anchor, METH_VARARGS, nullptr},
    {"interpolation_nearest_true_pixel", nearest_true_pixel, METH_VARARGS, nullptr},
    {"interpolation_dilated_overlap_count", dilated_overlap_count, METH_VARARGS, nullptr},
    {"interpolation_keep_center_fill", keep_center_fill, METH_VARARGS, nullptr},
    {"interpolation_paste_masked_or", paste_or<false>, METH_VARARGS, nullptr},
    {"interpolation_paste_packed_or", paste_or<true>, METH_VARARGS, nullptr},
    {"interpolation_find_projection_candidates", find_projection_candidates, METH_VARARGS, nullptr},
    {"interpolation_validate_encoded_block", validate_encoded_block, METH_VARARGS, nullptr},
    {"interpolation_commit_encoded_block", commit_encoded_block, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

} // namespace

int register_interpolation(PyObject* module) {
    return PyModule_AddFunctions(module, methods);
}

} // namespace xta::cpu
