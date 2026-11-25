"""Bounded priority-basin selection for native review marker edits."""

from __future__ import annotations

from collections.abc import Callable
import math
import operator

import numpy as np


class BasinTooLarge(RuntimeError):
    """The effect of an all-cutoff edit could not be certified in the budget."""


def _sitk_flood(priorities, markers, domain):
    import SimpleITK as sitk

    # Match the production Slicer cutoff store, including FIFO ties.
    image = np.where(domain, 511 - priorities, 1023).astype(np.int16)
    labels = sitk.MorphologicalWatershedFromMarkers(
        sitk.GetImageFromArray(image),
        sitk.GetImageFromArray(markers),
        markWatershedLine=False,
        fullyConnected=False,
    )
    return sitk.GetArrayFromImage(labels)


def _domain_holding_seeds(before, footprint):
    from scipy import ndimage

    labels, count = ndimage.label(before > 0)
    selected = np.zeros(count + 1, dtype=bool)
    selected[labels[footprint]] = True
    selected[0] = False
    return selected[labels]


def _flood_arrays(before, after, footprint, kept):
    domain = _domain_holding_seeds(before, footprint)
    kept_seeds = domain & kept
    priorities = 2 * after.astype(np.int16)
    priorities[kept_seeds] += 1
    priorities[footprint] = 2 * before[footprint].astype(np.int16)
    markers = kept_seeds.astype(np.uint8)
    markers[footprint] = 2
    return domain, kept_seeds, priorities, markers


def _face(axis, side):
    face = [slice(None)] * 3
    face[axis] = 0 if side == 0 else -1
    return tuple(face)


def claim_window(
    shape,
    seed_bounds,
    read_window: Callable,
    flood: Callable | None = None,
    *,
    max_voxels,
    margin=8,
):
    """Return `(removed, origin_zyx, stop_zyx)` for a certified local basin.

    The half-open seed box contains all removed support at the displayed
    cutoff. Each read returns the visible union levels before and after the
    tentative edit, the removed-support mask and the kept-voxel mask. Levels
    are uint8 and masks are bool. Set the shape to one voxel on the fixed axis
    for a single Remove Slices plane.

    The primary flood must leave every internal face unclaimed. A second
    flood tests the competing exterior by seeding those faces as kept, using
    the existing cutoff editor's boundary rule. Increase failed margins
    before reading again. The requested voxel cap applies even when a
    connected component extends across the entire native scan.
    """
    shape = tuple(map(operator.index, shape))
    seed_lo, seed_hi = (tuple(map(operator.index, point)) for point in seed_bounds)
    if (len(shape) != 3 or len(seed_lo) != 3 or len(seed_hi) != 3
            or not all(0 <= lo < hi <= size for lo, hi, size in zip(seed_lo, seed_hi, shape))):
        raise ValueError("Basin seed bounds must lie inside a three-dimensional grid")
    max_voxels, margin = operator.index(max_voxels), operator.index(margin)
    if max_voxels < 1 or margin < 1:
        raise ValueError("Basin voxel budget and starting margin must be positive")
    flood = _sitk_flood if flood is None else flood
    if math.prod(min(size, stop + margin) - max(0, start - margin)
                 for size, start, stop in zip(shape, seed_lo, seed_hi)) > max_voxels:
        margin = 1
    lower = [margin] * 3
    upper = [margin] * 3

    while True:
        lo = tuple(max(0, start - pad) for start, pad in zip(seed_lo, lower))
        hi = tuple(min(size, stop + pad) for stop, pad, size in zip(seed_hi, upper, shape))
        local_shape = tuple(b - a for a, b in zip(lo, hi))
        wanted = math.prod(local_shape)
        if wanted > max_voxels:
            raise BasinTooLarge(
                f"All Cutoffs needs a larger connected review window "
                f"({wanted:,} voxels, budget {max_voxels:,}). Try Displayed+Stricter."
            )

        before, after, footprint, kept = read_window(lo, hi)
        for values, dtype in ((before, np.uint8), (after, np.uint8), (footprint, np.bool_), (kept, np.bool_)):
            if values.shape != local_shape or values.dtype != dtype:
                raise ValueError("Basin reader returned a mask or level grid with invalid shape or dtype")
        if not footprint.any():
            return footprint, lo, hi
        if np.any(footprint & (before == 0)):
            raise ValueError("Basin seed is outside the original visible support")
        domain, kept_seeds, priorities, markers = _flood_arrays(before, after, footprint, kept)
        if kept_seeds.any():
            claimed = (flood(priorities, markers, domain) == 2) & domain & ~kept
        else:
            claimed = domain & ~kept

        faces = [
            (axis, side, _face(axis, side)) for axis in range(3) for side in range(2)
            if (lo[axis] > 0 if side == 0 else hi[axis] < shape[axis])
        ]
        failed = [(axis, side) for axis, side, face in faces if claimed[face].any()]
        competing_priorities = competing_markers = labels = checked = None
        if not failed:
            competing_faces = {}
            for axis, side, face in faces:
                edge = domain[face] & (markers[face] == 0)
                if not edge.any():
                    continue
                if competing_markers is None:
                    competing_priorities, competing_markers = priorities.copy(), markers.copy()
                label = 3 + axis + 3 * side
                competing_faces[label] = (axis, side)
                competing_markers[face][edge] = label
                competing_priorities[face][edge] = 2 * after[face][edge].astype(np.int16) + 1
            if competing_markers is None:
                return claimed, lo, hi

            labels = flood(competing_priorities, competing_markers, domain)
            checked = (labels == 2) & domain & ~kept
            if np.array_equal(checked, claimed):
                return claimed, lo, hi
            failed = [
                competing_faces[int(label)] for label in np.unique(labels[checked ^ claimed])
                if label in competing_faces
            ] or [(axis, side) for axis, side, _ in faces]

        for axis, side in failed:
            pads = lower if side == 0 else upper
            pads[axis] *= 2
        del before, after, footprint, kept, domain, kept_seeds, priorities, markers, claimed
        del competing_priorities, competing_markers, labels, checked
