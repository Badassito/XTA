"""Independent, vectorized NumPy Spherical projection oracle for tests/tools.

Production Spherical CPU projection requires the compiled float64 kernel. This
reference intentionally remains outside XTA so a benchmark can compare the
retained geometry rules without re-enabling a costly runtime backend.
"""
from __future__ import annotations

import numpy as np

from XTA.qsc import qsc_forward_face


def nearest_global_shell(radius: np.ndarray, radii: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(radii, radius, side='left'), 0, len(radii) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(radius - radii[left] <= radii[right] - radius, left, right)


def _processing_index(native: np.ndarray, native_count: int, processing_count: int) -> np.ndarray:
    if native_count == processing_count:
        return native
    return np.minimum(((native.astype(np.float64) + .5) * processing_count / native_count).astype(np.int64),
                      processing_count - 1)


def pull_spherical_chunk(source, view, radii, rotation, output_shape, z, first, stop, bboxes=None,
                         *, scalar_max=False):
    """Evaluate one flattened source XY strip with production's exact geometry."""
    out_t, out_h, out_w = output_shape
    work_t, work_h, work_w = int(view.full_t), int(view.full_h), int(view.full_w)
    flat = np.arange(first, stop, dtype=np.int64)
    dx = ((flat % out_w + .5) * work_w / out_w - .5) - (work_w - 1) / 2.0
    dy = ((flat // out_w + .5) * work_h / out_h - .5) - (work_h - 1) / 2.0
    dz = ((float(z) + .5) * work_t / out_t - .5) - (work_t - 1) / 2.0
    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    valid = ((radius >= float(view.spherical_min_radius))
             & (radius <= float(view.spherical_max_radius)))
    result = np.zeros(flat.shape, dtype=np.uint8)
    positions = np.flatnonzero(valid)
    if not positions.size:
        return result
    shell = None
    if bboxes is not None:
        shell = nearest_global_shell(radius[positions], radii)
        boxes = bboxes[shell]
        occupied = ((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2]))
        positions, shell = positions[occupied], shell[occupied]
        if not positions.size:
            return result
    dx, dy = dx[positions], dy[positions]
    local = np.stack(tuple(dx * rotation[0, a] + dy * rotation[1, a] + dz * rotation[2, a]
                           for a in range(3)), axis=-1)
    u, v, member = qsc_forward_face(local, int(view.spherical_face))
    count = int(view.spherical_face_intervals)
    columns = np.rint((u + 1.0) * count / 2.0).astype(np.int64) - int(view.spherical_u_origin)
    rows = np.rint((1.0 - v) * count / 2.0).astype(np.int64) - int(view.spherical_v_origin)
    member &= ((columns >= 0) & (columns < int(view.src_w))
               & (rows >= 0) & (rows < int(view.src_h)))
    if not np.any(member):
        return result
    positions = positions[member]
    shell = (nearest_global_shell(radius[positions], radii) if shell is None else shell[member])
    pr = _processing_index(rows[member], int(view.src_h), int(source.shape[1]))
    pc = _processing_index(columns[member], int(view.src_w), int(source.shape[2]))
    if bboxes is not None:
        boxes = bboxes[shell]
        inside = ((pr >= boxes[:, 0]) & (pr < boxes[:, 1])
                  & (pc >= boxes[:, 2]) & (pc < boxes[:, 3]))
        positions, shell, pr, pc = (a[inside] for a in (positions, shell, pr, pc))
    values = np.asarray(source[shell, pr, pc])
    result[positions] = values if scalar_max else np.asarray(values != 0, dtype=np.uint8)
    return result


def project_spherical_block(source, view, radii, rotation, shape, first_z, count, bboxes=None,
                            output_bounds=None):
    """Bounded oracle for small fixtures and dev benchmarks."""
    block = np.zeros((count, shape[1], shape[2]), dtype=np.uint8)
    if output_bounds is None:
        z0, z1, first_pixel, stop_pixel = first_z, first_z + count, 0, shape[1] * shape[2]
    else:
        z0, z1, y0, y1, x0, x1 = output_bounds.block(first_z, count)
        if z0 == z1 or y0 == y1 or x0 == x1:
            return block
        first_pixel, stop_pixel = y0 * shape[2], y1 * shape[2]
    for z in range(z0, z1):
        flat = block[z - first_z].reshape(-1)
        for first in range(first_pixel, stop_pixel, 128 * 1024):
            stop = min(first + 128 * 1024, stop_pixel)
            flat[first:stop] = pull_spherical_chunk(
                source, view, radii, rotation, shape, z, first, stop, bboxes)
    return block
