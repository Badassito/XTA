"""Shared plane bounds and structuring elements for editable policies."""
from functools import lru_cache

import numpy as np


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
