"""NumPy premises behind the azimuthal-keys GIL fix (numpy-only, host-checked).

The nogil angular key plane (``projection_coverage_cpu._angular_keys_into``)
keeps every transcendental in NumPy and changes only the call *shape*:

* ``np.degrees(np.arctan2(dy, dx))`` and ``% np.float32(180)`` now run on
  ~778k-point chunks instead of 65,536-point strips (old: out-of-place
  ``.astype(np.float32) % ...``; new: in-place ``%=``);
* ``np.cos/np.sin(np.deg2rad(angle))`` now run once on the per-angle float32
  table and are gathered, instead of on the per-point gathered angles.

Bit-identity therefore requires these NumPy float32 ufuncs to be
position-independent (no dependence on array length, split point, vector tail
or start alignment) on the host's dispatch target -- e.g. AVX512_SKX/SVML on
the Linux cluster. This module checks exactly that, needs only NumPy, and can
run where pytest/numba/XTA are absent::

    python -B tests/gil_azimuthal_numpy_premises.py

It exits 0 and prints ``OK`` when every premise holds on this host.
"""
import sys

import numpy as np


def _dispatch():
    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # NumPy 1.x
        from numpy.core import _multiarray_umath as umath
    found = [name for name in umath.__cpu_dispatch__ if umath.__cpu_features__.get(name)]
    return dict(numpy=np.__version__, baseline=list(umath.__cpu_baseline__),
                dispatch=list(umath.__cpu_dispatch__), active=found)


def _offsets(rng, size):
    """float32 centered offsets as the sampler forms them, plus edge values."""
    special = np.asarray([0., -0., 1., -1., 1e-30, -1e-30, 1e-45, 3000., -3000., .5, -.5,
                          1511.5, -1511.5, 2.4e-7, 65504., 1e20], np.float32)
    dy = (rng.integers(-3100, 3100, size) + rng.choice([0., .25, .5, .75, .1], size))
    dx = (rng.integers(-3100, 3100, size) + rng.choice([0., .25, .5, .75, .1], size))
    dy, dx = dy.astype(np.float32), dx.astype(np.float32)
    grid_y, grid_x = np.meshgrid(special, special, indexing='ij')
    dy = np.concatenate([grid_y.ravel(), dy, dx[:size // 8], -dy[:size // 8]])
    dx = np.concatenate([grid_x.ravel(), dx, dx[:size // 8], dy[size // 8:size // 4]])
    return np.ascontiguousarray(dy), np.ascontiguousarray(dx)


def _old_theta(dy, dx):
    # projection_coverage.azimuthal_plane_samples, verbatim.
    return np.degrees(np.arctan2(dy, dx)).astype(np.float32) % np.float32(180.)


def _new_theta(dy, dx):
    # projection_coverage_cpu._angular_keys_into, verbatim.
    theta = np.degrees(np.arctan2(dy, dx))
    theta %= np.float32(180.)
    return theta


def _bits(array):
    assert array.dtype == np.float32, array.dtype
    return np.ascontiguousarray(array).view(np.uint32)


def check_numpy_premises(size=300_000, seed=20261102):
    """Return a list of violated premises (empty when all hold)."""
    rng = np.random.default_rng(seed)
    failures = []
    dy, dx = _offsets(rng, size)
    reference = _bits(_old_theta(dy, dx))
    if not np.array_equal(reference, _bits(_new_theta(dy, dx))):
        failures.append('in-place theta %= differs from out-of-place astype %')
    # Strip/chunk splits at many lengths (vector tails) and start offsets
    # (misaligned starts inside an allocation), old shape vs new shape.
    lengths = list(range(1, 70)) + [127, 128, 129, 255, 256, 257, 4095, 4096, 4097, 65536]
    for start in range(0, 19):
        for length in lengths:
            stop = min(dy.size, start + length)
            for function, label in ((_old_theta, 'old'), (_new_theta, 'new')):
                part = _bits(function(dy[start:stop], dx[start:stop]))
                if not np.array_equal(part, reference[start:stop]):
                    failures.append(f'{label} theta depends on position: start={start} len={length}')
    # Fresh (aligned) copies of single elements: the shortest masked tail.
    for i in rng.choice(dy.size, 512, replace=False):
        one = _bits(_new_theta(dy[i:i + 1].copy(), dx[i:i + 1].copy()))
        if one[0] != reference[i]:
            failures.append(f'single-element theta differs at {i}')
            break
    # The production chunking: 65,536-point strips vs ~778k-point chunks.
    for strip in (65536, 778_000, 997):
        rebuilt = np.concatenate([_bits(_new_theta(dy[b:b + strip], dx[b:b + strip]))
                                  for b in range(0, dy.size, strip)])
        if not np.array_equal(rebuilt, reference):
            failures.append(f'chunked theta ({strip}) differs from whole-array theta')
    # cos/sin per-angle table gathered vs per-point on gathered angles.
    for spacing in (.25, .3, 1., 3., 7.5, 45.):
        raw = np.arange(0., 360., spacing)
        angles = np.asarray([float(a) % 180.0 for a in raw], np.float64).astype(np.float32)
        angles = np.concatenate([angles, rng.uniform(0, 180, 257).astype(np.float32),
                                 np.asarray([0., 90., 180., 45., 135., 1e-7], np.float32)])
        for function in (np.cos, np.sin):
            table = _bits(function(np.deg2rad(angles)))
            for count in list(range(1, 40)) + [65536, 65537, 778_000]:
                nearest = rng.integers(0, angles.size, count).astype(np.int32)
                per_point = _bits(function(np.deg2rad(angles[nearest])))
                if not np.array_equal(per_point, table[nearest]):
                    failures.append(f'{function.__name__} table gather differs '
                                    f'(spacing={spacing}, count={count})')
                    break
    return failures


def main():
    info = _dispatch()
    print('numpy', info['numpy'], 'baseline', info['baseline'], 'active dispatch', info['active'])
    failures = check_numpy_premises()
    for failure in failures[:50]:
        print('FAIL', failure)
    print('OK' if not failures else f'{len(failures)} premise violations')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
