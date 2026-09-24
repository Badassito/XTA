"""Memory-map advice without holding the Python GIL during the kernel call."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import mmap
import os

import numpy as np


@lru_cache(maxsize=1)
def _libc_madvise():
    if os.name != 'posix':
        return None
    try:
        function = ctypes.CDLL(None, use_errno=True).madvise
    except (AttributeError, OSError):
        return None
    function.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    function.restype = ctypes.c_int
    return function


def madvise_mmap(mmap_obj: mmap.mmap, advice: int) -> str:
    """Advise the whole map, keeping its buffer exported until madvise returns.

    CPython's mmap.madvise calls the kernel while holding the GIL. A CDLL call
    releases it; the zero-copy NumPy view pins the mapping so another thread
    cannot close it while the raw pointer is in use.
    """
    function = _libc_madvise()
    if function is None:
        mmap_obj.madvise(advice)
        return 'python'
    view = np.frombuffer(mmap_obj, dtype=np.uint8)
    try:
        if function(view.ctypes.data, view.nbytes, int(advice)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        del view
    return 'libc'
