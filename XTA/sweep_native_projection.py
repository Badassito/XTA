"""Crop-first categorical rank pulls for Cartesian, Tilted and Azimuthal sweep views.

Planar Tilted views evaluate only certified per-row output spans. Every active
native frame box is mapped through the exact destination-pull contract (axis
contraction, native validity, affine row/column rounding with edge clamping and
the sheared nearest-frame band) to closed working-coordinate intervals, which
are dilated slightly and rounded outward with the kernel's own contributor
tables. Voxels outside those spans cannot read a nonzero rank; voxels inside
are evaluated by the unchanged kernel, so outputs stay bit-identical.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import math
import os

import numpy as np

from ._deps import _numba
from .env_flags import env_flag
from .geometry import physical_view_name, is_tilted_view
from .projection_coverage_cpu import (
    _angular_keys, _geometry, _pull_azimuthal, _pull_tilted, NativePullPlanUnavailable,
    prepare_native_pull_plan,
)


# Output axis of the (stack, native vertical, native horizontal) working
# coordinates for the transverse, sagittal and coronal Tilted bases.
_TILTED_OUTPUT_AXES = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
# Working-voxel dilation of every certified interval. The float64 equations
# deviate by far less; division by the shear tangent is scaled separately.
_TILTED_BOUND_MARGIN = 1e-6


def _tilted_row_spans_enabled():
    """Certified Tilted row spans; ``0`` restores the stack-only full-plane bounds."""
    return env_flag("YOLO_TTA_SWEEP_TILTED_ROW_SPANS", True)


def _tilted_margin(tangent):
    magnitude = abs(float(tangent))
    return _TILTED_BOUND_MARGIN*(1.+magnitude+(1./magnitude if magnitude > 0. else 0.))


@_numba.njit(nogil=True, cache=True)
def _contributor_hull(index, working, output, scale):
    """Closed range of the kernel's working coordinates for one output index."""
    if output < working:
        first = index*working//output
        stop = ((index+1)*working+output-1)//output
        return float(first), float(stop-1)
    value = (index+.5)*scale-.5
    return value, value


@_numba.njit(nogil=True, cache=True)
def _output_index_range(lower, upper, working, output, scale):
    """Half-open output indices whose contributor set meets closed ``[lower, upper]``.

    Contraction uses the kernel's integer contributor tables exactly. Centered
    expansion re-evaluates the kernel's float expression at both ends, so the
    result is the exact preimage of the supplied closed interval. A NaN end is
    unbounded, keeping the result a superset.
    """
    if lower != lower:
        lower = -np.inf
    if upper != upper:
        upper = np.inf
    if lower > upper:
        return 0, 0
    if output < working:
        if lower > working-1 or upper < 0:
            return 0, 0
        first = 0 if lower <= 0 else int(math.ceil(lower))
        last = working-1 if upper >= working-1 else int(math.floor(upper))
        if first > last:
            return 0, 0
        return first*output//working, ((last+1)*output-1)//working+1
    smallest = .5*scale-.5
    largest = (output-.5)*scale-.5
    if lower > largest or upper < smallest:
        return 0, 0
    if lower <= smallest:
        first = 0
    else:
        first = min(output, max(0, int(math.ceil((lower+.5)/scale-.5))))
        while first > 0 and (first-1+.5)*scale-.5 >= lower:
            first -= 1
        while first < output and (first+.5)*scale-.5 < lower:
            first += 1
    if upper >= largest:
        stop = output
    else:
        last = min(output-1, max(-1, int(math.floor((upper+.5)/scale-.5))))
        while last < output-1 and (last+1+.5)*scale-.5 <= upper:
            last += 1
        while last >= 0 and (last+.5)*scale-.5 > upper:
            last -= 1
        stop = last+1
    return first, max(first, stop)


@_numba.njit(nogil=True, cache=True)
def _tree_update(tree, leaves, first, stop, value, minimum):
    first += leaves
    stop += leaves
    while first < stop:
        if first & 1:
            tree[first] = min(tree[first], value) if minimum else max(tree[first], value)
            first += 1
        if stop & 1:
            stop -= 1
            tree[stop] = min(tree[stop], value) if minimum else max(tree[stop], value)
        first >>= 1
        stop >>= 1


@_numba.njit(nogil=True, cache=True)
def _tilted_slice_spans(z, frames, regions, integers, coefficients, margin, bbox, spans, trees):
    """Certified half-open x span of every row of output slice ``z``.

    Each active frame contributes, within this slice, an output-row interval
    and a closed horizontal working interval ``[x_lo, x_hi]`` intersected with
    an optional sheared band ``slope*row + [band_lo, band_hi]``. A row keeps the
    hull ``max(min x_lo, slope*row + min band_lo)`` .. ``min(max x_hi, ...)``
    over frames that may reach it, which contains every frame's exact interval.
    Returns the slice hull ``(y0, y1, x0, x1)``; all zeros when nothing projects.
    """
    base, vertical = integers[0], integers[1]
    wt, wh, ww = integers[2], integers[3], integers[4]
    ot, oh, ow = integers[5], integers[6], integers[7]
    tangent, origin, center = coefficients[0], coefficients[1], coefficients[2]
    s_lo, s_hi = _contributor_hull(z, wt, ot, coefficients[9])
    leaves = trees.shape[1]//2
    trees[0, :] = np.inf
    trees[1, :] = -np.inf
    trees[2, :] = np.inf
    trees[3, :] = -np.inf
    spans[:, :] = 0
    slope = 0.
    if not vertical and tangent != 0.:
        if base == 1:
            slope = 1./tangent
        elif base == 2:
            slope = tangent
    touched = False
    for k in range(frames.size):
        low = frames[k]+origin-.5
        high = frames[k]+origin+.5
        v_lo, v_hi, u_lo, u_hi = regions[k, 0], regions[k, 1], regions[k, 2], regions[k, 3]
        band_lo, band_hi = -np.inf, np.inf
        if base == 0:
            # Slice z owns the stack; the nearest frame selects a shear-axis band.
            y_lo, y_hi, x_lo, x_hi = v_lo, v_hi, u_lo, u_hi
            if tangent == 0.:
                if high < s_lo-margin or low > s_hi+margin:
                    continue
            else:
                p = (s_lo-high)/tangent
                q = (s_hi-low)/tangent
                if vertical:
                    y_lo = max(y_lo, center+min(p, q)-margin)
                    y_hi = min(y_hi, center+max(p, q)+margin)
                else:
                    x_lo = max(x_lo, center+min(p, q)-margin)
                    x_hi = min(x_hi, center+max(p, q)+margin)
        else:
            # Slice z owns the native vertical coordinate: it gates the box rows.
            gate_lo = max(v_lo, s_lo-margin)
            gate_hi = min(v_hi, s_hi+margin)
            if gate_lo > gate_hi:
                continue
            if vertical:
                p = tangent*(gate_lo-center)
                q = tangent*(gate_hi-center)
                stack_lo, stack_hi = low+min(p, q), high+max(p, q)
                if base == 1:
                    y_lo, y_hi, x_lo, x_hi = stack_lo, stack_hi, u_lo, u_hi
                else:
                    y_lo, y_hi, x_lo, x_hi = u_lo, u_hi, stack_lo, stack_hi
            elif base == 1:
                # Rows own the stack; columns are the sheared native horizontal.
                p = tangent*(u_lo-center)
                q = tangent*(u_hi-center)
                y_lo, y_hi, x_lo, x_hi = low+min(p, q), high+max(p, q), u_lo, u_hi
                if tangent != 0.:
                    p = center-low/tangent
                    q = center-high/tangent
                    band_lo, band_hi = min(p, q), max(p, q)
            else:
                # Rows own the native horizontal; columns are the sheared stack.
                y_lo, y_hi, x_lo, x_hi = u_lo, u_hi, -np.inf, np.inf
                band_lo, band_hi = low-tangent*center, high-tangent*center
        if y_lo-margin > y_hi+margin or x_lo-margin > x_hi+margin:
            continue
        first, stop = _output_index_range(y_lo-margin, y_hi+margin, wh, oh, coefficients[10])
        first, stop = max(first, bbox[1]), min(stop, bbox[4])
        if first >= stop:
            continue
        touched = True
        _tree_update(trees[0], leaves, first, stop, x_lo-margin, True)
        _tree_update(trees[1], leaves, first, stop, x_hi+margin, False)
        _tree_update(trees[2], leaves, first, stop, band_lo-margin, True)
        _tree_update(trees[3], leaves, first, stop, band_hi+margin, False)
    if not touched:
        return 0, 0, 0, 0
    for node in range(1, leaves):
        for tree in range(4):
            value = trees[tree, node]
            for child in (2*node, 2*node+1):
                if tree & 1:
                    trees[tree, child] = max(trees[tree, child], value)
                else:
                    trees[tree, child] = min(trees[tree, child], value)
    y0, y1, x0, x1 = oh, 0, ow, 0
    for y in range(bbox[1], bbox[4]):
        lower, upper = trees[0, leaves+y], trees[1, leaves+y]
        if lower > upper:
            continue
        w_lo, w_hi = _contributor_hull(y, wh, oh, coefficients[10])
        p, q = slope*w_lo, slope*w_hi
        lower = max(lower, min(p, q)+trees[2, leaves+y]-margin)
        upper = min(upper, max(p, q)+trees[3, leaves+y]+margin)
        first, stop = _output_index_range(lower, upper, ww, ow, coefficients[11])
        first, stop = max(first, bbox[2]), min(stop, bbox[5])
        if first >= stop:
            continue
        spans[y, 0], spans[y, 1] = first, stop
        y0, y1 = min(y0, y), y+1
        x0, x1 = min(x0, first), max(x1, stop)
    if y0 >= y1:
        return 0, 0, 0, 0
    return y0, y1, x0, x1


@_numba.njit(nogil=True, cache=True)
def _tilted_span_rows(source, out, z, y0, x0, spans, integers, coefficients, bbox):
    """Exact kernel values on each certified span; the rest of ``out`` stays zero."""
    oh, ow = integers[6], integers[7]
    visited = addresses = 0
    for row in range(out.shape[0]):
        y = y0+row
        first, stop = spans[y, 0], spans[y, 1]
        if first >= stop:
            continue
        seen, count = _pull_tilted(source, out[row, first-x0:stop-x0], (z*oh+y)*ow+first,
                                   integers, coefficients, bbox, True)
        visited += seen
        addresses += count
    return visited, addresses


def _tilted_frame_regions(integers, coefficients, source_shape, boxes, margin):
    """Closed native (v, u) working intervals whose kernel address lies in each box.

    Rows/columns are rounded half up and clamped to the source edge, so an edge
    box also owns every address beyond that edge. Production Tilted affines are
    axis-aligned scalings; any other affine conservatively keeps the whole
    native domain for every active frame.
    """
    boxes = np.asarray(boxes, np.int64)
    active = np.flatnonzero((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2]))
    native_h, native_w = int(integers[8]), int(integers[9])
    height, width = int(source_shape[1]), int(source_shape[2])
    selected = boxes[active].astype(np.float64)
    count = len(active)
    v_lo, v_hi = np.full(count, -.5), np.full(count, native_h-.5)
    u_lo, u_hi = np.full(count, -.5), np.full(count, native_w-.5)
    c00, c01, c02, c10, c11, c12 = (float(value) for value in coefficients[3:9])
    if c01 == 0. and c10 == 0. and c00 != 0. and c11 != 0.:
        row_lo = np.where(boxes[active, 0] > 0, selected[:, 0]-.5, -np.inf)
        row_hi = np.where(boxes[active, 1] < height, selected[:, 1]-.5, np.inf)
        col_lo = np.where(boxes[active, 2] > 0, selected[:, 2]-.5, -np.inf)
        col_hi = np.where(boxes[active, 3] < width, selected[:, 3]-.5, np.inf)
        first, second = (row_lo-c12)/c11, (row_hi-c12)/c11
        v_lo = np.maximum(v_lo, np.minimum(first, second)-margin)
        v_hi = np.minimum(v_hi, np.maximum(first, second)+margin)
        first, second = (col_lo-c02)/c00, (col_hi-c02)/c00
        u_lo = np.maximum(u_lo, np.minimum(first, second)-margin)
        u_hi = np.minimum(u_hi, np.maximum(first, second)+margin)
    keep = (v_lo <= v_hi) & (u_lo <= u_hi)
    regions = np.ascontiguousarray(np.stack((v_lo, v_hi, u_lo, u_hi), axis=1)[keep])
    return np.ascontiguousarray(active[keep].astype(np.int64)), regions


def _tilted_union_bbox(integers, coefficients, frames, regions, margin):
    """3D union of every frame footprint over all stack positions."""
    if not len(frames):
        return (0, 0, 0, 0, 0, 0)
    base, vertical = int(integers[0]), int(integers[1])
    tangent, origin, center = (float(value) for value in coefficients[0:3])
    v_lo, v_hi, u_lo, u_hi = regions.T
    shear_lo, shear_hi = (v_lo, v_hi) if vertical else (u_lo, u_hi)
    shifts = tangent*(shear_lo-center), tangent*(shear_hi-center)
    stack_lo = frames+origin-.5+np.minimum(*shifts)
    stack_hi = frames+origin+.5+np.maximum(*shifts)
    lows, highs = [0., 0., 0.], [0., 0., 0.]
    for axis, low, high in zip(_TILTED_OUTPUT_AXES[base], (stack_lo, v_lo, u_lo), (stack_hi, v_hi, u_hi)):
        lows[axis], highs[axis] = float(low.min())-margin, float(high.max())+margin
    first, stop = [], []
    for axis in range(3):
        a, b = _output_index_range(lows[axis], highs[axis], int(integers[2+axis]),
                                   int(integers[5+axis]), float(coefficients[9+axis]))
        first.append(int(a))
        stop.append(int(b))
    if any(a >= b for a, b in zip(first, stop)):
        return (0, 0, 0, 0, 0, 0)
    return (*first, *stop)


def _tilted_certificate(view, native_shape, output_shape, boxes):
    """Exact geometry, active frames, native regions and union bbox for one Tilted view."""
    _, _, integers, coefficients, _ = _geometry(view, native_shape, output_shape)
    margin = _tilted_margin(coefficients[0])
    frames, regions = _tilted_frame_regions(integers, coefficients, native_shape, boxes, margin)
    bbox = _tilted_union_bbox(integers, coefficients, frames, regions, margin)
    return SimpleNamespace(integers=integers, coefficients=coefficients, margin=margin,
                           frames=frames, regions=regions, bbox=bbox)


@_numba.njit(nogil=True, cache=True)
def _tilted_rows(source, out, first, z, rect, integers, coeff):
    ow, oh = integers[7], integers[6]
    rw = rect[5] - rect[2]
    pos = addresses = 0
    while pos < out.size:
        relative = first + pos
        y, x = rect[1] + relative // rw, rect[2] + relative % rw
        take = min(out.size-pos, rect[5]-x)
        _, count = _pull_tilted(source, out[pos:pos+take], (z*oh+y)*ow+x, integers, coeff, rect, True)
        addresses += count
        pos += take
    return addresses


@_numba.njit(nogil=True, cache=True)
def _azimuthal_rows(source, out, first, z, rect, integers, coeff, keys, cached, invalid, rows):
    ow, oh = integers[7], integers[6]
    rw = rect[5] - rect[2]
    pos = addresses = 0
    while pos < out.size:
        relative = first + pos
        y, x = rect[1] + relative // rw, rect[2] + relative % rw
        take = min(out.size-pos, rect[5]-x)
        row_keys = keys if cached else keys[pos:pos+take]
        _, count = _pull_azimuthal(source, out[pos:pos+take], (z*oh+y)*ow+x, integers,
                                   coeff, rect, True, row_keys, cached, invalid, rows)
        addresses += count
        pos += take
    return addresses


def native_rank_bounds(view, native_shape, output_shape, boxes):
    """Reuse the complete geometry's conservative bounds without opening source payloads.

    Planar Tilted views return the 3D union of every certified frame footprint;
    the previous bound limited only the stack axis.
    """
    if is_tilted_view(view) and _tilted_row_spans_enabled():
        try:
            return _tilted_certificate(view, tuple(map(int, native_shape)),
                                       tuple(map(int, output_shape)), boxes).bbox
        except ValueError:
            pass  # Invalid geometry keeps its established stack-only bound and later failure.
    from .interpolation import CTILE_INDEX_DTYPE
    from .packed_tilted_projection import _encoded_destination_bbox
    index = np.zeros(native_shape[0], CTILE_INDEX_DTYPE)
    index["kind"] = boxes[:,1] != boxes[:,0]
    for col, name in enumerate(("y0","y1","x0","x1")):
        index[name] = boxes[:,col]
    proxy = SimpleNamespace(shape=native_shape, index=index)
    result = _encoded_destination_bbox(proxy, view, output_shape)
    return tuple(map(int,result)) if result is not None else (0,0,0,0,0,0)


@_numba.njit(nogil=True, cache=True)
def _restore_d1_crop(projected, output, z0, z1, ys, xs, y0, x0):
    for row in range(output.shape[0]):
        y = ys[y0+row]
        for col in range(output.shape[1]):
            x = xs[x0+col]
            best = np.uint8(0)
            for z in range(z0,z1):
                best = max(best, projected[z,y,x])
            output[row,col] = best


def _upright(source, view, target, boxes):
    axes = {"transverse":(0,1,2), "sagittal":(1,0,2), "coronal":(1,2,0)}.get(physical_view_name(view))
    if axes is None:
        return None
    projected = source.transpose(axes)
    active = np.flatnonzero(boxes[:,1] > boxes[:,0])
    if active.size:
        selected = boxes[active]
        low = (int(active[0]), int(selected[:,0].min()), int(selected[:,2].min()))
        high = (int(active[-1])+1, int(selected[:,1].max()), int(selected[:,3].max()))
        rect = (*[low[i] for i in axes], *[high[i] for i in axes])
    else:
        rect = (0,0,0,0,0,0)
    if projected.shape != target:
        from .d1_orthogonal_coverage import build_orthogonal_coverage_plan, orthogonal_coverage_supported
        if not orthogonal_coverage_supported(view, source.shape, target):
            return None  # Generic area contraction stays on the established bounded score restore.
        plan = build_orthogonal_coverage_plan(view, source.shape, target)
        if active.size:
            first, stop = [], []
            for i, (starts, stops) in enumerate(plan.source_ranges_tyx):
                first.append(int(np.searchsorted(stops, rect[i], side="right")))
                stop.append(int(np.searchsorted(starts, rect[i+3], side="left")))
            rect = (*first,*stop)
        (zs,ze), (ys,_), (xs,_) = plan.source_ranges_tyx
        def crop_d1(z):
            if not rect[0] <= z < rect[3] or rect[4] <= rect[1] or rect[5] <= rect[2]:
                return None
            dst = np.empty((rect[4]-rect[1],rect[5]-rect[2]),np.uint8)
            _restore_d1_crop(projected,dst,zs[z],ze[z],ys,xs,rect[1],rect[2])
            return rect[1],rect[4],rect[2],rect[5],dst
        return crop_d1,rect
    def crop(z):
        if not rect[0] <= z < rect[3]:
            return None
        if axes == (0,1,2):
            y0, y1, x0, x1 = map(int,boxes[z])
        else:
            y0, y1, x0, x1 = rect[1],rect[4],rect[2],rect[5]
        if not (y0 < y1 and x0 < x1):
            return None
        return y0,y1,x0,x1,projected[z,y0:y1,x0:x1].copy(order="C")
    return crop, rect


@contextmanager
def native_rank_crop_reader(source, view, shape, work_dir, *, memory_bytes, known_native_boxes):
    """Read only a conservative output rectangle, with no per-pixel or per-row Python dispatch.

    Planar Tilted crops evaluate only the certified per-row spans of each slice
    (``projected_points`` counts those voxels); the rest of the crop is zero.
    """
    stats = {"backend":"cpu_cartesian_rank_crop", "projected_points":0, "kernel_calls":0,
             "contribution_addresses":0}
    if view.family == "orthogonal":
        selected = _upright(source,view,shape,known_native_boxes)
        if selected is None:
            yield None
            return
        select, rect = selected
        def crop(z):
            result = select(z)
            if result is not None:
                stats["projected_points"] += result[4].size
            return result
    else:
        if (view.family != "azimuthal" and not is_tilted_view(view)) or os.environ.get(
                "YOLO_TTA_NATIVE_PULL_BACKEND", "compiled").strip().lower() == "numpy":
            yield None
            return
        from .confidence_projection import score_projection_workspace
        workspace = score_projection_workspace(source.shape,view,shape,memory_bytes)
        # The ordinary score reader reserves angular preparation and the result plane.
        # Also charge the short-lived bbox index and compiled strip coordinate tables.
        from .interpolation import CTILE_INDEX_DTYPE
        available = memory_bytes - workspace.fixed_bytes - source.shape[0]*(CTILE_INDEX_DTYPE.itemsize+64)
        if available < 8192:
            yield None
            return
        # Compact plan-free key generation needs its own local/global coordinate
        # vectors in addition to the original plan's admitted temporary arrays.
        # Planar Tilted crops instead use it for the per-slice span search.
        coordinate_bytes = min(4*1024**2, available//8)
        leaves = 1 << max(0, int(shape[1])-1).bit_length()
        certificate = None
        if (is_tilted_view(view) and _tilted_row_spans_enabled()
                and 64*leaves+16*int(shape[1]) <= coordinate_bytes):
            try:
                certificate = _tilted_certificate(view, source.shape, shape, known_native_boxes)
            except ValueError:
                certificate = None
        rect = (certificate.bbox if certificate is not None
                else native_rank_bounds(view,source.shape,shape,known_native_boxes))
        try:
            plan = prepare_native_pull_plan(view, source.shape, shape,
                                            max_plan_bytes=min(64*1024**2,available-coordinate_bytes),cache_plane=True)
        except (NativePullPlanUnavailable, NotImplementedError):
            yield None
            return
        bounds = np.asarray(rect,np.int64)
        rw = rect[5]-rect[2]
        limit = min(plan.max_strip_voxels, workspace.chunk_points, max(1,coordinate_bytes//48))
        cached = plan.backend == "compiled_azimuthal_cached"
        stats["backend"] = "cpu_rank_rectangle_" + plan.backend
        if certificate is not None and plan.backend == "compiled_native_tilted" and (
                np.array_equal(certificate.integers, plan.integers)
                and np.array_equal(certificate.coefficients, plan.coefficients)):
            stats.update(skipped_planes=0, tilted_bounds_scope="certified_frame_row_spans")
            frames, regions, margin = certificate.frames, certificate.regions, certificate.margin
            def crop(z):
                if not rect[0] <= z < rect[3] or not rw or rect[4] == rect[1]:
                    stats["skipped_planes"] += 1
                    return None
                spans = np.empty((shape[1], 2), np.int64)
                trees = np.empty((4, 2*leaves), np.float64)
                y0, y1, x0, x1 = _tilted_slice_spans(z, frames, regions, plan.integers, plan.coefficients,
                                                     margin, bounds, spans, trees)
                if y0 >= y1:
                    stats["skipped_planes"] += 1
                    return None
                arr = np.zeros((y1-y0, x1-x0), np.uint8)
                visited, count = _tilted_span_rows(source, arr, z, y0, x0, spans,
                                                   plan.integers, plan.coefficients, bounds)
                stats["contribution_addresses"] += int(count)
                stats["projected_points"] += int(visited)
                stats["kernel_calls"] += 1
                return y0, y1, x0, x1, arr
        else:
            crop = None
        def rectangle_crop(z):
            if not rect[0] <= z < rect[3] or not rw or rect[4] == rect[1]:
                return None
            arr = np.empty((rect[4]-rect[1],rw),np.uint8)
            flat = arr.ravel()
            for start in range(0,flat.size,limit):
                out = flat[start:start+limit]
                if plan.backend == "compiled_native_tilted":
                    count = _tilted_rows(source,out,start,z,bounds,plan.integers,plan.coefficients)
                else:
                    if cached:
                        keys = plan.keys.reshape(-1)
                    else:
                        where = np.arange(start,start+out.size,dtype=np.int64)
                        y = rect[1]+where//rw
                        x = rect[2]+where%rw
                        base = int(plan.integers[0])
                        vertical = y if base == 0 else np.full(out.size,z,np.int64)
                        horizontal = x if base != 2 else y
                        keys = _angular_keys(plan,vertical,horizontal)
                    count = _azimuthal_rows(source,out,start,z,bounds,plan.integers,plan.coefficients,
                                             keys,cached,plan.invalid_key,plan.row_lookup)
                stats["contribution_addresses"] += int(count)
                stats["projected_points"] += out.size
                stats["kernel_calls"] += 1
            return rect[1],rect[4],rect[2],rect[5],arr
        if crop is None:
            crop = rectangle_crop
    def read(z):
        out = np.zeros(shape[1:],np.uint8)
        item = crop(z)
        if item is not None:
            y0,y1,x0,x1,data = item
            out[y0:y1,x0:x1] = data
        return out
    read.read_crop = crop
    read.known_z_bounds = rect[0],rect[3]
    read.known_output_bbox_tyx = rect
    read.projection_diagnostics = lambda: dict(stats)
    yield read
