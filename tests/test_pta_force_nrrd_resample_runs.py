"""Forced NRRD XY resampling by contiguous runs writes the same bytes as the np.ix_ gather."""
import itertools
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_inputs
from XTA.pta_inputs import _force_fit_nearest_source_indices as nearest


def gather_plane(source, target_yx):
    """The previous implementation: one np.ix_ gather, then a uint8-out np.greater."""
    out = np.empty(target_yx, np.uint8)
    if tuple(source.shape) == tuple(target_yx):
        np.greater(source, 0, out=out)
    else:
        rows, cols = nearest(source.shape[0], target_yx[0]), nearest(source.shape[1], target_yx[1])
        np.greater(source[np.ix_(rows, cols)], 0, out=out)
    return out


def run_plane(source, target_yx):
    out = np.full(target_yx, 7, np.uint8)  # every byte must be written
    runs = pta_inputs._nrrd_mask_xy_indices((1, *source.shape), (1, *target_yx))
    pta_inputs._write_nrrd_mask_plane(source, out, runs)
    return out


def random_plane(rng, shape, dtype=np.uint8):
    values = rng.integers(-3, 256, shape) * (rng.random(shape) < 0.3)
    return values.astype(dtype)


DELTAS = (-3, -2, -1, 0, 1, 2, 3)


@pytest.mark.parametrize('size_y', [1, 2, 3, 4, 5, 7, 12, 31])
def test_small_shapes_match_gather_for_every_delta_both_ways(size_y):
    rng = np.random.default_rng(size_y)
    for size_x in (1, 2, 3, 4, 6, 9, 17, 64):
        source = random_plane(rng, (size_y, size_x))
        for dy, dx in itertools.product(DELTAS, DELTAS):
            target = (size_y + dy, size_x + dx)
            if min(target) < 1:
                continue
            assert run_plane(source, target).tobytes() == gather_plane(source, target).tobytes(), target


@pytest.mark.parametrize('source_yx, target_yx', [
    ((3064, 3024), (3064, 3022)),
    ((3064, 3024), (3064, 3023)),
    ((3064, 3024), (3064, 3025)),
    ((3064, 3024), (3064, 3026)),
    ((3064, 3024), (3061, 3027)),
    ((3064, 3027), (3067, 3024)),
    ((3061, 3024), (3064, 3022)),
    ((1021, 1007), (1018, 1010)),
])
def test_real_variant_shapes_match_gather(source_yx, target_yx):
    rng = np.random.default_rng(sum(source_yx) + sum(target_yx))
    source = np.zeros(source_yx, np.uint8)
    source[200:-200, 150:-150] = rng.integers(0, 3, (source_yx[0] - 400, source_yx[1] - 300), dtype=np.uint8)
    source[0, ::2] = source[-1, 1::2] = source[::3, 0] = source[1::3, -1] = 255  # edges carry foreground
    kind, runs_y, runs_x = pta_inputs._nrrd_mask_xy_indices((1, *source_yx), (1, *target_yx))
    assert kind == 'runs' and len(runs_y) <= 4 and len(runs_x) <= 4  # a few view copies per plane
    assert run_plane(source, target_yx).tobytes() == gather_plane(source, target_yx).tobytes()


@pytest.mark.parametrize('kind', ['int16', 'float_nan', 'bool', 'transposed', 'strided'])
def test_general_reader_sources_match_gather(kind):
    rng = np.random.default_rng(3)
    base = random_plane(rng, (23, 29), np.int16)
    source = {
        'int16': base,
        'float_nan': np.where(rng.random(base.shape) < 0.1, np.nan, base.astype(np.float32)),
        'bool': base > 0,
        'transposed': np.ascontiguousarray(base.T).T,
        'strided': random_plane(rng, (46, 87), np.int16)[::2, ::3],
    }[kind]
    for target in ((20, 26), (23, 32), (26, 29), (21, 31)):
        assert run_plane(source, target).tobytes() == gather_plane(source, target).tobytes()


@pytest.mark.parametrize('source_yx, target_yx', [
    ((306, 302), (153, 151)),   # 2x down
    ((153, 151), (306, 302)),   # 2x up
    ((306, 302), (459, 453)),   # 1.5x up
    ((306, 302), (102, 101)),   # 3x down
    ((306, 302), (306, 151)),   # one axis rescaled, the other equal
    ((300, 300), (309, 309)),   # 10 runs per axis: 100 blocks, past the run limit
])
def test_far_from_unity_uses_the_gather_and_matches(source_yx, target_yx):
    rng = np.random.default_rng(sum(source_yx) * 7 + sum(target_yx))
    source = random_plane(rng, source_yx)
    xy = pta_inputs._nrrd_mask_xy_indices((1, *source_yx), (1, *target_yx))
    assert xy[0] == 'gather'  # run copies would be one ufunc call per tiny block
    assert run_plane(source, target_yx).tobytes() == gather_plane(source, target_yx).tobytes()


def test_run_limit_boundary():
    # 8 runs per axis is 64 blocks: still runs. 9 per axis is 81: gather.
    assert pta_inputs._nrrd_mask_xy_indices((1, 300, 300), (1, 307, 307))[0] == 'runs'
    assert pta_inputs._nrrd_mask_xy_indices((1, 300, 300), (1, 308, 308))[0] == 'gather'
    rng = np.random.default_rng(11)
    source = random_plane(rng, (300, 300))
    for target in ((307, 307), (308, 308), (293, 293), (292, 292)):
        assert run_plane(source, target).tobytes() == gather_plane(source, target).tobytes()


def test_equal_xy_returns_no_runs_and_normalizes():
    source = np.asarray([[0, 1, 255], [3, 0, 0]], np.uint8)
    assert pta_inputs._nrrd_mask_xy_indices((4, 2, 3), (5, 2, 3)) is None
    assert run_plane(source, (2, 3)).tolist() == [[0, 1, 1], [1, 0, 0]]


def write_mask(tmp_path, source, encoding):
    nrrd = pytest.importorskip('nrrd')
    path = tmp_path / f'mask_{encoding}.nrrd'
    nrrd.write(str(path), np.ascontiguousarray(source).transpose(2, 1, 0),
               header={'encoding': encoding, 'space directions': np.eye(3)})
    return path


def gather_volume(source, target):
    """Reference for a forced contiguous load: endpoint-aligned nearest on every axis."""
    planes, rows, cols = (nearest(size, out) for size, out in zip(source.shape, target))
    return (source[np.ix_(planes, rows, cols)] > 0).astype(np.uint8)


@pytest.mark.parametrize('encoding', ['raw', 'gzip'])
@pytest.mark.parametrize('deltas', [(0, 0, 1), (0, 0, -3), (1, -2, 3), (-3, 3, -1), (2, 1, 0), (3, 3, 3), (-3, -3, -3)])
def test_forced_loads_match_gather_reference_on_both_readers(tmp_path, encoding, deltas):
    rng = np.random.default_rng([delta + 3 for delta in deltas])
    source = random_plane(rng, (9, 14, 19))
    source[4] = 0
    target = tuple(size + delta for size, delta in zip(source.shape, deltas))
    path = write_mask(tmp_path, source, encoding)
    expected = gather_volume(source, target)
    fast = pta_inputs.load_nrrd_mask_for_volume(path, target, warnings=pta.WarningLog(), force=True)
    with mock.patch.object(pta_inputs, '_read_canonical_uint8_nrrd', return_value=None):
        general = pta_inputs.load_nrrd_mask_for_volume(path, target, warnings=pta.WarningLog(), force=True)
    for actual in (fast, general):
        assert actual.dtype == np.uint8 and actual.flags.c_contiguous
        assert actual.tobytes() == expected.tobytes()
