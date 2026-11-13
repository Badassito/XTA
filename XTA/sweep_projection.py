"""Project one nested confidence family by numeric maximum on actual geometry.

Rank zero is absent foreground; rank j admits one-based levels <= j. Threshold
classification and mask/native-affine processing happen before this reader.
Numeric maximum commutes with categorical OR and never interpolates scores.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import queue
import threading

import numpy as np


def _shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in value)):
        raise ValueError("Projection output shape must have three positive integers")
    return tuple(int(v) for v in value)


def _rank_boxes(source, level_count):
    boxes = np.zeros((source.shape[0], 4), np.int64)
    maximum = 0
    for index, plane in enumerate(source):
        maximum = max(maximum, int(plane.max(initial=0)))
        if maximum > level_count:
            raise ValueError("Source rank exceeds level_count")
        rows, columns = np.any(plane, axis=1), np.any(plane, axis=0)
        if rows.any():
            boxes[index] = (int(np.argmax(rows)), len(rows)-int(np.argmax(rows[::-1])),
                            int(np.argmax(columns)), len(columns)-int(np.argmax(columns[::-1])))
    return boxes, maximum


def _certified_boxes(value, source_shape):
    boxes = np.asarray(value)
    if boxes.dtype != np.int64 or boxes.shape != (source_shape[0], 4):
        raise ValueError("known_native_boxes must be int64 N-by-4 native bounds")
    if (np.any(boxes < 0) or np.any(boxes[:, 0] > boxes[:, 1])
            or np.any(boxes[:, 2] > boxes[:, 3])
            or np.any(boxes[:, 1] > source_shape[1]) or np.any(boxes[:, 3] > source_shape[2])):
        raise ValueError("known_native_boxes exceed native geometry")
    empty_y, empty_x = boxes[:, 0] == boxes[:, 1], boxes[:, 2] == boxes[:, 3]
    if np.any(empty_y != empty_x) or np.any(boxes[empty_y] != 0):
        raise ValueError("known_native_boxes empty frames must use four zeros")
    return np.ascontiguousarray(boxes)


def _output_axis_bounds(lower, upper, working, output):
    # Conservative broad-phase only. Exact existing pull equations retain every
    # final categorical test; extra working/output voxels cover centered rounding.
    first = math.floor((lower+.5)*output/working-.5)-1
    stop = math.ceil((upper+.5)*output/working-.5)+2
    return max(0, min(output, first)), max(0, min(output, stop))


def _radial_disk_bounds(view, shape, radii, boxes):
    """Coarse superset: the outer active disk square and the whole stack axis."""
    active = np.flatnonzero((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2]))
    if not len(active):
        return (0, 0, 0, 0, 0, 0)
    last = int(view.radial_shell_start) + int(active[-1])
    outer = float(view.radial_max_radius)
    if last+1 < len(radii):
        outer = min(outer, float(radii[last]+(radii[last+1]-radii[last])*.5))
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    margin = 1. + 32*np.finfo(np.float64).eps*max(*work, abs(outer))
    outer += margin
    stack, vertical, horizontal = _RADIAL_AXES[str(view.radial_base_view)]
    limits = [(0, shape[i]) for i in range(3)]
    limits[vertical] = _output_axis_bounds(float(view.center_y)-outer, float(view.center_y)+outer,
                                         work[vertical], shape[vertical])
    limits[horizontal] = _output_axis_bounds(float(view.center_x)-outer, float(view.center_x)+outer,
                                           work[horizontal], shape[horizontal])
    return (*[limits[i][0] for i in range(3)], *[limits[i][1] for i in range(3)])


# Output axes (stack, vertical dy, horizontal dx) of each Radial base plane.
_RADIAL_AXES = {"transverse": (0, 1, 2), "sagittal": (1, 0, 2), "coronal": (2, 0, 1)}
# Working-voxel dilation of every certified interval before conservative output
# rounding. Exact float64 geometry deviates by far less than 1e-6 voxel here.
_RADIAL_BOUND_MARGIN = 1.0
# Closed acceptance tolerance (radians / relative radius) for near-degenerate
# candidate points. Accepting extra points can only enlarge a superset.
_RADIAL_BOUND_ANGLE_TOLERANCE = 1e-9
_TWO_PI = 2.0*math.pi


def _output_axis_ranges(lower, upper, working, output):
    """Array form of :func:`_output_axis_bounds` with identical rounding."""
    first = np.floor((np.asarray(lower, np.float64)+.5)*output/working-.5)-1
    stop = np.ceil((np.asarray(upper, np.float64)+.5)*output/working-.5)+2
    return (np.clip(first, 0, output).astype(np.int64),
            np.clip(stop, 0, output).astype(np.int64))


def _native_index_ranges(native, source_count, starts, stops):
    """Inclusive native indices whose kernel source index lies in [start, stop).

    This repeats the projection kernel's exact float64 expression
    ``min(int((i+0.5)*source_count/native), source_count-1)``. The map is
    nondecreasing, so every certified box selects one contiguous index run.
    """
    mapped = np.minimum(((np.arange(native, dtype=np.float64)+.5)*source_count/native)
                        .astype(np.int64), source_count-1)
    first = np.searchsorted(mapped, starts, side="left").astype(np.int64)
    last = np.searchsorted(mapped, stops, side="left").astype(np.int64)-1
    return first, last


def _contains_phase(alpha, beta, phase):
    """Whether some phase+2*pi*k lies in each closed interval [alpha, beta]."""
    tolerance = _RADIAL_BOUND_ANGLE_TOLERANCE
    return (np.floor((beta-phase)/_TWO_PI+tolerance)
            >= np.ceil((alpha-phase)/_TWO_PI-tolerance))


def _arc_trig_ranges(alpha, beta, full):
    """Closed cosine and sine ranges over angle intervals (endpoints plus extrema)."""
    cos_a, cos_b, sin_a, sin_b = np.cos(alpha), np.cos(beta), np.sin(alpha), np.sin(beta)
    cos_min = np.where(full | _contains_phase(alpha, beta, math.pi), -1., np.minimum(cos_a, cos_b))
    cos_max = np.where(full | _contains_phase(alpha, beta, 0.), 1., np.maximum(cos_a, cos_b))
    sin_min = np.where(full | _contains_phase(alpha, beta, 1.5*math.pi), -1., np.minimum(sin_a, sin_b))
    sin_max = np.where(full | _contains_phase(alpha, beta, .5*math.pi), 1., np.maximum(sin_a, sin_b))
    return cos_min, cos_max, sin_min, sin_max


def _scaled_range(inner, outer, low, high):
    """Range of r*v for r in [inner, outer] >= 0 and v in [low, high]."""
    return (np.where(low < 0, outer*low, inner*low), np.where(high > 0, outer*high, inner*high))


def _sector_band_extent(inner, outer, alpha, beta, full, low, high):
    """Closed horizontal extent of one annular sector inside each band low <= y <= high.

    A linear maximum over the compact intersection lies at a sector vertex, a
    circle point with y=0, a circle/band-line crossing, or a ray/band-line
    crossing. Candidates use small closed acceptance tolerances; an empty
    result (min > max) certifies that the band misses the sector.
    """
    tolerance = _RADIAL_BOUND_ANGLE_TOLERANCE
    radial_tolerance = tolerance*(1.+outer)
    span = beta-alpha
    minimum = np.full(low.shape, np.inf)
    maximum = np.full(low.shape, -np.inf)

    def in_arc(angle):
        if full:
            return True
        offset = np.mod(angle-alpha, _TWO_PI)
        return (offset <= span+tolerance) | (offset >= _TWO_PI-tolerance)

    def admit(x, valid):
        nonlocal minimum, maximum
        minimum = np.where(valid, np.minimum(minimum, x), minimum)
        maximum = np.where(valid, np.maximum(maximum, x), maximum)

    in_band = lambda y: (y >= low-radial_tolerance) & (y <= high+radial_tolerance)
    radii = (inner, outer)
    if not full:
        for angle in (alpha, beta):
            cosine, sine = math.cos(angle), math.sin(angle)
            for radius in radii:
                admit(radius*cosine, in_band(radius*sine))
            if sine != 0.:
                for line in (low, high):
                    distance = line/sine
                    admit(distance*cosine, (distance >= inner-radial_tolerance)
                          & (distance <= outer+radial_tolerance))
    crosses_axis = in_band(0.)
    for radius in radii:
        for x, angle in ((radius, 0.), (-radius, math.pi)):
            if np.all(in_arc(np.float64(angle))):
                admit(x, crosses_axis)
        for line in (low, high):
            touching = np.abs(line) <= radius+radial_tolerance
            root = np.sqrt(np.maximum(radius*radius-line*line, 0.))
            for x in (root, -root):
                admit(x, touching & in_arc(np.arctan2(line, x)))
    return minimum, maximum


def _radial_output_bounds(view, shape, radii, boxes, source_shape):
    """Certified output support of one Radial rank projection (``RadialOutputBounds``).

    Each active native shell frame is bounded independently from the exact
    projection-kernel contract: nearest-shell radius band, the periodic arc
    of the certified box columns, and the stack interval admitted by its box
    rows (shifted by the tilted sample axis). Every interval is dilated by a
    working voxel and rounded outward, then intersected with the coarse disk
    superset. An output voxel reads only its nearest shell, so the union of
    shell footprints covers every nonzero output. The exact kernel still
    performs every final membership test.
    """
    rectangles = np.zeros((shape[0], 4), np.int64)
    legacy = _radial_disk_bounds(view, shape, radii, boxes)
    stack_axis, vertical_axis, horizontal_axis = _RADIAL_AXES[str(view.radial_base_view)]
    if legacy[3] <= legacy[0] or legacy[4] <= legacy[1] or legacy[5] <= legacy[2]:
        return RadialOutputBounds((0, 0, 0, 0, 0, 0), rectangles, stack_axis, *_EMPTY_FOOTPRINTS)
    native_h, native_w = int(source_shape[1]), int(source_shape[2])
    src_h, src_w = int(view.src_h), int(view.src_w)
    active = np.flatnonzero((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2]))
    row_first, row_last = _native_index_ranges(src_h, native_h, boxes[active, 0], boxes[active, 1])
    column_first, column_last = _native_index_ranges(src_w, native_w, boxes[active, 2], boxes[active, 3])
    keep = (row_first <= row_last) & (column_first <= column_last)
    # One extra native row/column is a free guard around the exact index maps.
    row_first, row_last = np.maximum(row_first-1, 0), np.minimum(row_last+1, src_h-1)
    column_first, column_last = np.maximum(column_first-1, 0), np.minimum(column_last+1, src_w-1)
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    stack_length = work[stack_axis]
    height_origin = int(view.radial_height_origin)
    # Height is clamped to the stack, then rint(height)-origin selects a row.
    height_first = np.maximum(height_origin+row_first, 0)
    height_last = np.minimum(height_origin+row_last, stack_length-1)
    keep &= height_first <= height_last

    # Nearest global shell: an exact midpoint belongs to the inner shell, and
    # radii beyond either end belong to the end shell. Closed min/max limits.
    radii = np.asarray(radii, np.float64)
    count = len(radii)
    global_index = int(view.radial_shell_start)+active
    selected = radii[global_index]
    previous = radii[np.maximum(global_index-1, 0)]
    following = radii[np.minimum(global_index+1, count-1)]
    minimum_radius, maximum_radius = float(view.radial_min_radius), float(view.radial_max_radius)
    radius_slack = 16*np.finfo(np.float64).eps*(1.+max(maximum_radius, float(radii[-1])))
    inner = np.where(global_index > 0, previous+(selected-previous)*.5, 0.)
    outer = np.where(global_index < count-1, selected+(following-selected)*.5, maximum_radius)
    inner = np.maximum(np.maximum(inner, minimum_radius)-radius_slack, 0.)
    outer = np.minimum(outer, maximum_radius)+radius_slack
    keep &= inner <= outer

    # Column c is read only when theta*r lies within half a column of
    # arc_origin+c modulo the shell period; periods <= 1 read every column.
    arc_origin = float(view.radial_arc_origin)
    period = _TWO_PI*selected
    column_slack = 1e-6+64*np.finfo(np.float64).eps*(src_w+4)*(1.+abs(arc_origin)+src_w+period)
    alpha = (arc_origin+column_first-.5-column_slack)/selected
    beta = (arc_origin+column_last+.5+column_slack)/selected
    full = (period <= 1.) | (beta-alpha >= _TWO_PI)
    cos_min, cos_max, sin_min, sin_max = _arc_trig_ranges(alpha, beta, full)
    dx_min, dx_max = _scaled_range(inner, outer, cos_min, cos_max)
    dy_min, dy_max = _scaled_range(inner, outer, sin_min, sin_max)

    stack_low = height_first-.5
    stack_high = height_last+.5
    ideal_low = np.full(len(active), -.5)
    ideal_high = np.full(len(active), stack_length-.5)
    tangent = 0.
    if bool(view.radial_tilted_source):
        tangent = math.tan(math.radians(float(view.tilt_angle_deg)))
        vertical = str(view.tilt_direction) == "vertical"
        # Sample axis r*sin/cos((arc_origin+c)/r) over the certified columns.
        sample_alpha = (arc_origin+column_first-column_slack)/selected
        sample_beta = (arc_origin+column_last+column_slack)/selected
        sample_full = full | (sample_beta-sample_alpha >= _TWO_PI)
        sample_trig = _arc_trig_ranges(sample_alpha, sample_beta, sample_full)
        sample_low, sample_high = (sample_trig[2:] if vertical else sample_trig[:2])
        sheared = (tangent*selected*sample_low, tangent*selected*sample_high)
        shear_low, shear_high = np.minimum(*sheared), np.maximum(*sheared)
        pixel = (tangent*dy_min, tangent*dy_max) if vertical else (tangent*dx_min, tangent*dx_max)
        pixel_low, pixel_high = np.minimum(*pixel), np.maximum(*pixel)
        ideal_low, ideal_high = ideal_low+pixel_low, ideal_high+pixel_high
        stack_low = stack_low+shear_low
        stack_high = stack_high+shear_high
        # A clamped height reaches the first/last row from any stack value
        # that still passes the ideal-height test.
        stack_low = np.where(height_first == 0, np.minimum(stack_low, ideal_low), stack_low)
        stack_high = np.where(height_last == stack_length-1, np.maximum(stack_high, ideal_high), stack_high)
    stack_low, stack_high = np.maximum(stack_low, ideal_low), np.minimum(stack_high, ideal_high)
    height_slack = 1e-9*(1.+stack_length+abs(tangent)*outer)
    keep &= stack_low <= stack_high+height_slack

    margin = _RADIAL_BOUND_MARGIN
    stack_first, stack_stop = _output_axis_ranges(stack_low-margin-height_slack, stack_high+margin+height_slack,
                                                  stack_length, shape[stack_axis])
    center_y, center_x = float(view.center_y), float(view.center_x)
    vertical_first, vertical_stop = _output_axis_ranges(center_y+dy_min-margin, center_y+dy_max+margin,
                                                        work[vertical_axis], shape[vertical_axis])
    horizontal_first, horizontal_stop = _output_axis_ranges(center_x+dx_min-margin, center_x+dx_max+margin,
                                                            work[horizontal_axis], shape[horizontal_axis])
    lows = [legacy[i] for i in range(3)]
    highs = [legacy[i+3] for i in range(3)]
    stack_first, stack_stop = np.maximum(stack_first, lows[stack_axis]), np.minimum(stack_stop, highs[stack_axis])
    vertical_first = np.maximum(vertical_first, lows[vertical_axis])
    vertical_stop = np.minimum(vertical_stop, highs[vertical_axis])
    horizontal_first = np.maximum(horizontal_first, lows[horizontal_axis])
    horizontal_stop = np.minimum(horizontal_stop, highs[horizontal_axis])
    keep &= ((stack_first < stack_stop) & (vertical_first < vertical_stop)
             & (horizontal_first < horizontal_stop))

    big = np.iinfo(np.int64).max
    # Per output slice: [stack or vertical range, horizontal range] unions.
    first_a = np.full(shape[0], big, np.int64)
    stop_a = np.zeros(shape[0], np.int64)
    first_b = np.full(shape[0], big, np.int64)
    stop_b = np.zeros(shape[0], np.int64)
    kept = np.flatnonzero(keep)
    row_ranges = None
    if stack_axis == 0:
        footprint_first, footprint_stop = stack_first[kept], stack_stop[kept]
        shell_ranges = np.stack((vertical_first, vertical_stop, horizontal_first, horizontal_stop),
                                axis=1)[kept].astype(np.int32)
        offsets = np.zeros(len(kept), np.int64)
        for k in kept:
            z0, z1 = int(stack_first[k]), int(stack_stop[k])
            np.minimum(first_a[z0:z1], vertical_first[k], out=first_a[z0:z1])
            np.maximum(stop_a[z0:z1], vertical_stop[k], out=stop_a[z0:z1])
            np.minimum(first_b[z0:z1], horizontal_first[k], out=first_b[z0:z1])
            np.maximum(stop_b[z0:z1], horizontal_stop[k], out=stop_b[z0:z1])
    else:
        # Axis 0 is the vertical in-plane axis: intersect each shell sector
        # with the band of every output row it can reach.
        out_t, work_t = shape[0], work[0]
        horizontal_work, horizontal_out = work[horizontal_axis], shape[horizontal_axis]
        footprint_first = np.zeros(len(kept), np.int64)
        footprint_stop = np.zeros(len(kept), np.int64)
        offsets = np.zeros(len(kept), np.int64)
        shell_ranges = np.zeros((len(kept), 4), np.int32)
        parts, total = [], 0
        for position, k in enumerate(kept):
            z0, z1 = int(vertical_first[k]), int(vertical_stop[k])
            rows = np.arange(z0, z1, dtype=np.float64)
            dy = ((2.*rows+1.-out_t)*work_t)/(2.*out_t)+((work_t-1.)*.5-center_y)
            low_x, high_x = _sector_band_extent(float(inner[k]), float(outer[k]), float(alpha[k]),
                                                float(beta[k]), bool(full[k]), dy-margin, dy+margin)
            touched = low_x <= high_x
            if not np.any(touched):
                continue
            x0, x1 = _output_axis_ranges(np.where(touched, center_x+low_x-margin, 0.),
                                         np.where(touched, center_x+high_x+margin, 0.),
                                         horizontal_work, horizontal_out)
            x0 = np.maximum(x0, horizontal_first[k])
            x1 = np.minimum(x1, horizontal_stop[k])
            touched &= x0 < x1
            hits = np.flatnonzero(touched)
            if not hits.size:
                continue
            index = hits+z0  # distinct rows
            first_b[index] = np.minimum(first_b[index], x0[touched])
            stop_b[index] = np.maximum(stop_b[index], x1[touched])
            first_a[index] = np.minimum(first_a[index], stack_first[k])
            stop_a[index] = np.maximum(stop_a[index], stack_stop[k])
            lead, tail = int(hits[0]), int(hits[-1])+1
            span = np.where(touched, 1, 0)[lead:tail]
            parts.append(np.stack((x0[lead:tail]*span, x1[lead:tail]*span), axis=1).astype(np.int32))
            footprint_first[position], footprint_stop[position] = z0+lead, z0+tail
            offsets[position] = total
            total += tail-lead
            shell_ranges[position, :2] = (stack_first[k], stack_stop[k])
        row_ranges = np.concatenate(parts) if parts else np.zeros((0, 2), np.int32)
    nonempty = (first_a < stop_a) & (first_b < stop_b)
    if not np.any(nonempty):
        return RadialOutputBounds((0, 0, 0, 0, 0, 0), rectangles, stack_axis, *_EMPTY_FOOTPRINTS)
    if stack_axis == 2:
        # Coronal: output axis 1 is horizontal (dx), axis 2 is the stack.
        first_a, stop_a, first_b, stop_b = first_b, stop_b, first_a, stop_a
    rectangles[nonempty] = np.stack((first_a, stop_a, first_b, stop_b), axis=1)[nonempty]
    slices = np.flatnonzero(nonempty)
    bbox = (int(slices[0]), int(rectangles[nonempty, 0].min()), int(rectangles[nonempty, 2].min()),
            int(slices[-1])+1, int(rectangles[nonempty, 1].max()), int(rectangles[nonempty, 3].max()))
    used = footprint_first < footprint_stop
    return RadialOutputBounds(bbox, rectangles, stack_axis, footprint_first[used], footprint_stop[used],
                              offsets[used], shell_ranges[used], row_ranges)


_EMPTY_FOOTPRINTS = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
                     np.zeros((0, 4), np.int32), None)
# Output-voxel tile edge of the per-slice footprint cover.
_RADIAL_COVER_TILE = 32
# Evaluate the whole union rectangle when the cover saves less than this.
_RADIAL_COVER_MIN_SAVING = .1


@dataclass(frozen=True)
class RadialOutputBounds:
    """Certified Radial rank-projection support on the output grid.

    ``rectangles[z]`` is the half-open ``(y0, y1, x0, x1)`` union rectangle of
    axis-0 output slice ``z`` (all zero when nothing can project there) and
    ``bbox_tyx`` is their 3D union. The per-shell footprints let each slice
    evaluate only output tiles that one of its native shells can reach.
    """
    bbox_tyx: tuple
    rectangles: np.ndarray
    stack_axis: int
    footprint_first: np.ndarray
    footprint_stop: np.ndarray
    footprint_offset: np.ndarray
    shell_ranges: np.ndarray
    row_ranges: object

    @property
    def nbytes(self):
        arrays = (self.rectangles, self.footprint_first, self.footprint_stop, self.footprint_offset,
                  self.shell_ranges, self.row_ranges)
        return sum(int(a.nbytes) for a in arrays if a is not None)

    def footprints(self, z):
        """Every shell footprint rectangle on output slice ``z`` as an M-by-4 int64 array."""
        inside = np.flatnonzero((self.footprint_first <= z) & (z < self.footprint_stop))
        ranges = self.shell_ranges[inside].astype(np.int64)
        if self.row_ranges is None:
            return ranges
        rows = self.row_ranges[self.footprint_offset[inside]+(z-self.footprint_first[inside])].astype(np.int64)
        stack, horizontal = ranges[:, :2], rows
        pieces = np.concatenate((horizontal, stack) if self.stack_axis == 2 else (stack, horizontal), axis=1)
        return pieces[(pieces[:, 0] < pieces[:, 1]) & (pieces[:, 2] < pieces[:, 3])]

    def cover(self, z):
        """Disjoint rectangles covering slice ``z`` footprints, or ``()`` when it is empty."""
        y0, y1, x0, x1 = (int(v) for v in self.rectangles[z])
        if y0 >= y1 or x0 >= x1:
            return ()
        return _footprint_cover(self.footprints(z), (y0, y1, x0, x1), _RADIAL_COVER_TILE)


def _footprint_cover(footprints, rectangle, tile):
    """Tile-aligned disjoint rectangles inside ``rectangle`` covering every footprint."""
    y0, y1, x0, x1 = rectangle
    if len(footprints) <= 1:
        if not len(footprints):
            return (rectangle,)  # Defensive superset; certified unions are never footprint-free.
        a, b, c, d = (int(v) for v in footprints[0])
        piece = (max(a, y0), min(b, y1), max(c, x0), min(d, x1))
        return (piece,) if piece[0] < piece[1] and piece[2] < piece[3] else (rectangle,)
    rows, columns = -(-(y1-y0)//tile), -(-(x1-x0)//tile)
    if rows*columns <= 1:
        return (rectangle,)
    top = (np.maximum(footprints[:, 0], y0)-y0)//tile
    bottom = (np.minimum(footprints[:, 1], y1)-y0-1)//tile+1
    left = (np.maximum(footprints[:, 2], x0)-x0)//tile
    right = (np.minimum(footprints[:, 3], x1)-x0-1)//tile+1
    valid = (top < bottom) & (left < right)
    marks = np.zeros((rows+1, columns+1), np.int32)
    top, bottom, left, right = top[valid], bottom[valid], left[valid], right[valid]
    np.add.at(marks, (top, left), 1)
    np.add.at(marks, (top, right), -1)
    np.add.at(marks, (bottom, left), -1)
    np.add.at(marks, (bottom, right), 1)
    covered = np.cumsum(np.cumsum(marks, axis=0), axis=1)[:rows, :columns] > 0
    if covered.sum() >= (1.-_RADIAL_COVER_MIN_SAVING)*covered.size:
        return (rectangle,)
    edges = np.diff(np.pad(covered.view(np.int8), ((0, 0), (1, 1))), axis=1)
    pieces = []
    band_runs, band_start = None, 0
    for row in range(rows+1):
        if row < rows:
            starts = np.flatnonzero(edges[row] == 1)
            runs = tuple(zip(starts.tolist(), np.flatnonzero(edges[row] == -1).tolist()))
        else:
            runs = None
        if runs == band_runs:
            continue
        if band_runs:
            top_y, bottom_y = y0+band_start*tile, min(y1, y0+row*tile)
            pieces.extend((top_y, bottom_y, x0+a*tile, min(x1, x0+b*tile)) for a, b in band_runs)
        band_runs, band_start = runs, row
    return tuple(pieces)



def _copy_rectangle_strip(destination, values, first, y0, x0, width):
    """Copy one ROI-flat strip with at most three contiguous slice operations."""
    row, column = divmod(int(first), int(width))
    consumed = 0
    if column:
        take = min(len(values), width-column)
        destination[y0+row, x0+column:x0+column+take] = values[:take]
        consumed = take
        row += 1
    rows = (len(values)-consumed)//width
    if rows:
        end = consumed+rows*width
        destination[y0+row:y0+row+rows, x0:x0+width] = values[consumed:end].reshape(rows,width)
        consumed = end
        row += rows
    if consumed < len(values):
        destination[y0+row, x0:x0+len(values)-consumed] = values[consumed:]


def _curved_reader(source, view, shape, boxes, workspace, diagnostics, *, radial_bounds=None):
    if str(view.family) == "spherical":
        from .spherical_projection import _validate_spherical_projection
        from .spherical_projection_bounds import spherical_output_bounds
        from .spherical_projection_cpu import prepare_spherical_chunk_numba, pull_spherical_rectangle_numba
        radii, rotation, resolved, boxes = _validate_spherical_projection(source, view, shape, boxes)
        if tuple(resolved) != shape:
            raise ValueError("Spherical projection changed output geometry")
        prepare_spherical_chunk_numba(source, view, radii, rotation, shape, boxes)
        bound = spherical_output_bounds(view, shape, boxes)
        bounds = (bound.z0, bound.y0, bound.x0, bound.z1, bound.y1, bound.x1)
        plane_rectangle = (bound.y0, bound.y1, bound.x0, bound.x1)
        def rectangle(z):
            inside = (bound.z0 <= z < bound.z1 and bound.y0 < bound.y1 and bound.x0 < bound.x1)
            return (plane_rectangle, (plane_rectangle,)) if inside else None
        def pull(z, first, stop, _rectangle):
            return pull_spherical_rectangle_numba(source, view, radii, rotation, shape,
                z, first, stop, boxes, bounds_yx=plane_rectangle, scalar_max=True)
        diagnostics["backend"] = "cpu_spherical_rank_rectangle"
    else:
        from .cylindrical_geometry import global_radii
        from .cylindrical_projection import _radial_planfree_arguments, _pull_radial_range_into
        radii = np.asarray(global_radii(view), np.float64)
        arguments = _radial_planfree_arguments(source, view, radii, shape, boxes)
        if radial_bounds is None:
            radial_bounds = _radial_output_bounds(view, shape, radii, boxes, source.shape)
        bounds, rectangles = radial_bounds.bbox_tyx, radial_bounds.rectangles
        def rectangle(z):
            y0, y1, x0, x1 = (int(v) for v in rectangles[z])
            if y0 >= y1 or x0 >= x1:
                return None
            return (y0, y1, x0, x1), radial_bounds.cover(z)
        def pull(z, first, stop, piece):
            result = np.zeros(stop-first, np.uint8)
            # Every output value is the full exact kernel value; a piece only
            # chooses which certified-support voxels are evaluated.
            _pull_radial_range_into(result, arguments[0], arguments[1], z, first, *arguments[2:],
                                    True, piece[3]-piece[2], piece[2], piece[0])
            return result
        diagnostics["backend"] = "cpu_radial_rank_rectangle"
        diagnostics["radial_rectangle_scope"] = "certified_shell_footprint_tiles"
        pull(0, 0, 0, (0, 1, 0, 1))  # Prepare this immutable CPU signature before lane threads start.
    z0, z1 = bounds[0], bounds[3]
    # Both compiled kernels own only a uint8 output strip and scalar geometry.
    # Their strip credit is bytes, unlike the NumPy coordinate-array fallback's
    # conservative per-point allowance. Bound individual native calls to 8 MiB.
    chunk = max(1,min(8*1024**2,int(workspace.memory_bytes-workspace.fixed_bytes)))
    diagnostics["strip_limit_points"] = chunk
    diagnostics["rectangle_copy_strategy"] = "contiguous_slices"
    def project_into(z,plane,result,destination_y,destination_x):
        (top,_bottom,left,_right),pieces=plane
        for piece in pieces:
            y0,y1,x0,x1=piece
            width=x1-x0
            total=(y1-y0)*width
            for first in range(0,total,chunk):
                stop=min(total,first+chunk)
                values=pull(z,first,stop,piece)
                _copy_rectangle_strip(result,values,first,destination_y+y0-top,destination_x+x0-left,width)
                diagnostics["kernel_calls"]+=1
                diagnostics["projected_points"]+=stop-first
    def read(z):
        result=np.zeros(shape[1:],np.uint8)
        plane=rectangle(z)
        if plane is None:
            diagnostics["skipped_planes"]+=1
            return result
        project_into(z,plane,result,plane[0][0],plane[0][2])
        return result
    def read_crop(z):
        plane=rectangle(z)
        if plane is None:
            diagnostics["skipped_planes"]+=1
            return None
        y0,y1,x0,x1=plane[0]
        result=np.zeros((y1-y0,x1-x0),np.uint8)
        project_into(z,plane,result,0,0)
        return y0,y1,x0,x1,result
    read.read_crop=read_crop
    read.known_z_bounds = (z0, z1)
    read.known_output_bbox_tyx = tuple(bounds)
    return read


@contextmanager
def sweep_rank_projection_reader(source_rank_u8,view,output_shape_tyx,work_dir,*,
                                 level_count,memory_bytes=64*1024**2,
                                 known_native_boxes=None,source_max_rank=None,workers=1):
    """Yield bounded owned planes/crops for all nested threshold levels.

    Caller-owned source ranks stay immutable. Without a packer certificate one
    scan validates ranks and gathers boxes; supplied int64 native boxes certify
    support and rank range. CPU lanes own independent immutable geometry/readers.
    One full queued-result plane per lane is reserved beyond working credit.
    CUDA first receives the complete available shared budget and stays one lane.
    """
    if (isinstance(level_count,bool) or not isinstance(level_count,(int,np.integer))
            or not 1<=int(level_count)<=255):
        raise ValueError("level_count must be an integer from 1 through 255")
    if (isinstance(memory_bytes,bool) or not isinstance(memory_bytes,(int,np.integer))
            or int(memory_bytes)<=0):
        raise ValueError("memory_bytes must be a positive integer")
    if isinstance(workers,bool) or not isinstance(workers,(int,np.integer)) or int(workers)<1:
        raise ValueError("workers must be a positive integer")
    requested=int(workers)
    source=np.asarray(source_rank_u8)
    if source.dtype!=np.uint8 or source.ndim!=3 or min(source.shape)<=0:
        raise ValueError("Sweep projection requires a nonempty three-dimensional uint8 rank array")
    if source.shape[0]!=int(view.num_slices):
        raise ValueError("Source rank frame count differs from declared view")
    shape=_shape(output_shape_tyx)
    from .confidence_projection import score_projection_reader,score_projection_workspace
    metadata_bytes=40*source.shape[0]+source.shape[1]+source.shape[2]
    if str(view.family)=="radial":
        # Shared immutable support certificate: slice rectangles, per-shell
        # footprints and bound temporaries, plus per-output-row footprint
        # extents when axis 0 is in-plane (sagittal/coronal bases).
        metadata_bytes+=32*shape[0]+256*source.shape[0]
        if str(view.radial_base_view)!="transverse":
            metadata_bytes+=8*source.shape[0]*shape[0]
    available=int(memory_bytes)-metadata_bytes
    score_projection_workspace(source.shape,view,shape,available)
    if source_max_rank is not None:
        if (isinstance(source_max_rank,bool) or not isinstance(source_max_rank,(int,np.integer))
                or not 0<=int(source_max_rank)<=int(level_count)):
            raise ValueError("source_max_rank must be a certified rank ceiling within level_count")
    certified=known_native_boxes is not None
    if certified:
        boxes=_certified_boxes(known_native_boxes,source.shape)
        active=bool(np.any(boxes[:,1]>boxes[:,0]))
        maximum=(int(level_count) if source_max_rank is None else int(source_max_rank)) if active else 0
    else:
        boxes,maximum=_rank_boxes(source,int(level_count))
        if source_max_rank is not None and maximum>int(source_max_rank):
            raise ValueError("Measured source rank exceeds source_max_rank certificate")
    plane_bytes=math.prod(shape[1:])
    diagnostics={"backend":"cpu_rank_projection","level_count":int(level_count),
        "budget_bytes":int(memory_bytes),"source_metadata_bytes":metadata_bytes,
        "output_plane_bytes":plane_bytes,"source_max_rank":maximum,
        "source_max_rank_is_upper_bound":certified and maximum!=0,
        "source_scan_voxels":0 if certified else int(source.size),
        "requested_workers":requested,"parallel_workers":1,"read_planes":0,
        "crop_reads":0,"failed_reads":0,"active_reads":0,"peak_active_reads":0,
        "queued_plane_credit_bytes":0,"cpu_lane_working_bytes":0,"cpu_admitted_bytes":0}
    condition=threading.Condition()
    closing=False
    closed=False
    active_calls=0
    lanes=[]
    pool=queue.SimpleQueue()
    final_state=None
    def snapshot():
        with condition:
            if final_state is not None:
                return dict(final_state)
            state=dict(diagnostics)
            states=[]
            for lane in lanes:
                callback=getattr(lane,"projection_diagnostics",None)
                states.append(callback() if callback else {})
            if states:
                state["backend"]=states[0].get("backend",state["backend"])
                for key in ("kernel_calls","projected_points","skipped_planes","contribution_addresses",
                            "output_voxels","cuda_planes","dense_d2h_bytes"):
                    state[key]=sum(int(item.get(key,0)) for item in states)
                for key in ("strip_limit_points","rectangle_copy_strategy"):
                    if key in states[0]:
                        state[key]=states[0][key]
                state["lane_diagnostics"]=states
            return state
    def dispatch(z,crop):
        nonlocal active_calls
        if isinstance(z,bool) or not isinstance(z,(int,np.integer)) or not 0<=int(z)<shape[0]:
            raise IndexError("Projection output slice is outside its grid")
        with condition:
            if closing or closed:
                raise RuntimeError("Sweep rank projection reader is closed")
            active_calls+=1
            diagnostics["read_planes"]+=1
            diagnostics["crop_reads"]+=int(crop)
            diagnostics["active_reads"]=active_calls
            diagnostics["peak_active_reads"]=max(diagnostics["peak_active_reads"],active_calls)
        lane=None
        try:
            lane=pool.get()
            return lane.read_crop(int(z)) if crop else lane(int(z))
        except BaseException:
            with condition:
                diagnostics["failed_reads"]+=1
            raise
        finally:
            if lane is not None:
                pool.put(lane)
            with condition:
                active_calls-=1
                diagnostics["active_reads"]=active_calls
                condition.notify_all()
    def read(z):
        return dispatch(z,False)
    work=Path(work_dir)
    work.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="sweep-rank-project-",dir=work) as private:
            from .sweep_projection_cuda import cuda_rank_projection_reader
            output_certificate=None
            radial_bounds=None
            if str(view.family)=="radial":
                from .cylindrical_geometry import global_radii
                radial_bounds=_radial_output_bounds(view,shape,np.asarray(global_radii(view),np.float64),
                                                    boxes,source.shape)
                # CUDA consumes only this union box, as a superset z/empty-plane skip.
                output_certificate=radial_bounds.bbox_tyx
                rectangles=radial_bounds.rectangles
                diagnostics["certified_rectangle_voxels"]=int(np.sum(
                    (rectangles[:,1]-rectangles[:,0])*(rectangles[:,3]-rectangles[:,2])))
                diagnostics["certificate_bytes"]=radial_bounds.nbytes
            else:
                from .geometry import is_azimuthal_view,is_tilted_view
                if is_azimuthal_view(view) or is_tilted_view(view):
                    from .interpolation import CTILE_INDEX_DTYPE
                    bound_workspace=source.shape[0]*(CTILE_INDEX_DTYPE.itemsize+64)
                    diagnostics["native_bounds_workspace_bytes"]=bound_workspace
                    if (bound_workspace<=available and
                            max(source.shape[1:])<=np.iinfo(np.uint32).max):
                        from .sweep_native_projection import native_rank_bounds
                        try:
                            output_certificate=native_rank_bounds(view,source.shape,shape,boxes)
                        except NotImplementedError:
                            diagnostics["native_bounds_decline"]="unsupported_contract"
                    else:
                        diagnostics["native_bounds_decline"]="metadata_credit_or_bounds_dtype"
            cuda_diagnostics={}
            with cuda_rank_projection_reader(source,view,shape,private,
                    level_count=int(level_count),memory_bytes=available,
                    source_max_rank=maximum,known_native_boxes=boxes,
                    known_output_bbox_tyx=output_certificate,
                    diagnostics=cuda_diagnostics) as cuda_read:
                # Record why a CUDA admission declined so progress shows CPU fallbacks.
                diagnostics["cuda_reason"]=str(cuda_diagnostics.get("reason") or "unrecorded")
                with ExitStack() as stack:
                    if cuda_read is not None:
                        lanes.append(cuda_read)
                    else:
                        for count in range(requested,0,-1):
                            lane_total=available//count
                            working=lane_total-plane_bytes
                            try:
                                workspace=score_projection_workspace(source.shape,view,shape,working)
                            except MemoryError:
                                continue
                            break
                        else:
                            raise MemoryError("CPU rank projection budget cannot fit one working and queued output plane")
                        diagnostics["parallel_workers"]=count
                        diagnostics["queued_plane_credit_bytes"]=count*plane_bytes
                        diagnostics["cpu_lane_working_bytes"]=working
                        diagnostics["cpu_admitted_bytes"]=metadata_bytes+count*(working+plane_bytes)
                        for index in range(count):
                            if str(view.family) in ("radial","spherical"):
                                lane_state={"backend":"cpu_rank_projection","kernel_calls":0,
                                            "projected_points":0,"skipped_planes":0}
                                lane=_curved_reader(source,view,shape,boxes,workspace,lane_state,
                                                    radial_bounds=radial_bounds)
                                lane.projection_diagnostics=lambda state=lane_state:dict(state)
                            else:
                                lane_work=Path(private)/f"lane-{index}"
                                lane_work.mkdir()
                                from .sweep_native_projection import native_rank_crop_reader
                                lane=stack.enter_context(native_rank_crop_reader(
                                    source,view,shape,lane_work,memory_bytes=working,
                                    known_native_boxes=boxes))
                                if lane is None:
                                    lane=stack.enter_context(score_projection_reader(
                                        source,view,shape,lane_work,memory_bytes=working))
                            lanes.append(lane)
                    for lane in lanes:
                        pool.put(lane)
                    read.parallel_workers=len(lanes)
                    # A CUDA reader is one device lane; callers may encode its
                    # crops on more host threads than it has lanes.
                    read.serialized_device_reader=cuda_read is not None
                    diagnostics["parallel_workers"]=len(lanes)
                    for name in ("known_z_bounds","known_output_bbox_tyx"):
                        if hasattr(lanes[0],name):
                            setattr(read,name,getattr(lanes[0],name))
                    if maximum==0:
                        read.known_z_bounds=(0,0)
                        for lane in lanes:
                            lane.read_crop=lambda _z:None
                        read.read_crop=lambda z:dispatch(z,True)
                    elif all(hasattr(lane,"read_crop") for lane in lanes):
                        read.read_crop=lambda z:dispatch(z,True)
                    read.projection_diagnostics=snapshot
                    try:
                        yield read
                    finally:
                        with condition:
                            closing=True
                            while active_calls:
                                condition.wait()
                        final_state=snapshot()
    finally:
        with condition:
            closing=closed=True
        lanes.clear()
        pool=None
        source=boxes=None


__all__=["sweep_rank_projection_reader"]
