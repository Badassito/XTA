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

from .native_cpu import kernels as _cpu
from .env_flags import env_flag
from .geometry import physical_view_name, is_tilted_view
from .projection_coverage_cpu import (
    _angular_keys, _geometry, NativePullPlanUnavailable,
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


def _contributor_hull(index, working, output, scale):
    """Closed range of the kernel's working coordinates for one output index."""
    return _cpu.contributor_hull(index, working, output, scale)


def _output_index_range(lower, upper, working, output, scale):
    """Half-open output indices whose contributor set meets closed ``[lower, upper]``."""
    return _cpu.output_index_range(lower, upper, working, output, scale)


def _tilted_slice_spans(z, frames, regions, integers, coefficients, margin, bbox, spans, trees):
    """Certified half-open x span of every row of output slice ``z``."""
    return _cpu.tilted_slice_spans(
        z, frames, regions, integers, coefficients, margin, bbox, spans,
        trees)


def _tilted_span_rows(source, out, z, y0, x0, spans, integers, coefficients, bbox):
    """Exact kernel values on each certified span; the rest of ``out`` stays zero."""
    return _cpu.tilted_span_rows(source, out, z, y0, x0, spans, integers, coefficients, bbox)


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


def _tilted_rows(source, out, first, z, rect, integers, coeff):
    return _cpu.tilted_rank_rows(source, out, first, z, rect, integers, coeff)


def _azimuthal_rows(source, out, first, z, rect, integers, coeff, keys, cached, invalid, rows):
    return _cpu.azimuthal_rank_rows(
        source, out, first, z, rect, integers, coeff, keys, cached, invalid,
        rows)


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


def _restore_d1_crop(projected, output, z0, z1, ys, xs, y0, x0):
    return _cpu.restore_d1_crop(projected, output, z0, z1, ys, xs, y0, x0)


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
