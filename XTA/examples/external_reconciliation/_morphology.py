"""Shared plane bounds and structuring elements for editable policies."""
from functools import lru_cache
from math import isqrt
from numbers import Integral

import numpy as np
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:
    cv2 = None


CROSS = ndi.generate_binary_structure(2, 1)
_CROSS_U8 = CROSS.astype(np.uint8)
_CROSS_U8.flags.writeable = False
_MAX_BOX_WIDTH = isqrt(np.iinfo(np.int32).max)


def candidate_window(plane):
    """Find the smallest rectangle containing all candidate pixels."""
    rows = np.flatnonzero(np.any(plane, axis=1))
    if not rows.size:
        return None
    row = slice(int(rows[0]), int(rows[-1]) + 1)
    columns = np.flatnonzero(np.any(plane[row], axis=0))
    return row, slice(int(columns[0]), int(columns[-1]) + 1)


@lru_cache(maxsize=16)
def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = x * x + y * y <= radius * radius
    footprint.flags.writeable = False
    return footprint


@lru_cache(maxsize=16)
def _disk_u8(radius):
    footprint = disk(radius).astype(np.uint8)
    footprint.flags.writeable = False
    return footprint


def open_disk(mask, radius):
    """Open a plane with the integer disk and a constant zero border."""
    if (cv2 is None or mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size
            or not isinstance(radius, Integral) or radius < 0):
        return ndi.binary_opening(mask, structure=disk(radius), border_value=0)
    source = np.ascontiguousarray(mask, dtype=np.uint8)
    opened = cv2.morphologyEx(
        source, cv2.MORPH_OPEN, _disk_u8(radius), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return opened.view(np.bool_)


def grow_cores(cores, domain, radius):
    """Apply bounded cross dilation, freezing pixels outside the domain."""
    if (cv2 is None or cores.ndim != 2 or not cores.size or cores.dtype != np.bool_
            or domain.dtype != np.bool_ or domain.shape != cores.shape
            or not isinstance(radius, Integral) or radius <= 0):
        return ndi.binary_dilation(cores, structure=CROSS, iterations=radius, mask=domain)
    x, y, width, height = cv2.boundingRect(cores.view(np.uint8))
    result = np.zeros_like(cores, dtype=bool)
    if width == 0:
        return result
    # Each cross step can advance by at most one pixel along either coordinate.
    radius = int(radius)
    region = (slice(max(0, y - radius), min(cores.shape[0], y + height + radius)),
              slice(max(0, x - radius), min(cores.shape[1], x + width + radius)))
    current = np.ascontiguousarray(cores[region], dtype=np.uint8)
    eligible = np.ascontiguousarray(domain[region], dtype=np.uint8)
    expanded = np.empty_like(current)
    for _ in range(radius):
        cv2.dilate(current, _CROSS_U8, dst=expanded,
                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.copyTo(expanded, eligible, current)
    result[region] = current.view(np.bool_)
    return result


def box_count(mask, width):
    """Count foreground in a square window with a constant zero border."""
    if (cv2 is not None and mask.ndim == 2 and mask.size and mask.dtype == np.bool_
            and isinstance(width, Integral) and 0 < width <= _MAX_BOX_WIDTH):
        width = int(width)
        source = np.ascontiguousarray(mask, dtype=np.uint8)
        counts = cv2.boxFilter(source, cv2.CV_32S, (width, width), normalize=False,
                               borderType=cv2.BORDER_CONSTANT)
        return counts.astype(np.int64)
    filtered = ndi.uniform_filter(mask.astype(np.float64), size=width, mode='constant')
    return np.rint(filtered * width * width).astype(np.int64)
