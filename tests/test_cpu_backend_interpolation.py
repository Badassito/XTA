"""CPU-only contracts for real interpolation candidate-planning benchmark cases."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from XTA import interpolation
from tests.reference_backends.interpolation import _find_slice_projection_candidates_python
from tools.cpu_backend_benchmarks import interpolation as benchmark


def test_benchmark_cases_use_equal_nonempty_compiled_and_python_candidates() -> None:
    cases = benchmark.make_cases('smoke')
    assert {case.metadata['fragmented'] for case in cases} == {False, True}
    for case in cases:
        reference, compiled = case.reference(), case.compiled()
        assert reference.dtype == compiled.dtype == np.int64
        assert reference.ndim == compiled.ndim == 2
        assert reference.shape[1] == compiled.shape[1] == 9
        assert reference.shape[0] > 0
        np.testing.assert_array_equal(compiled, reference)
    with mock.patch.object(interpolation, '_find_slice_projection_candidates_native', return_value=None):
        with pytest.raises(RuntimeError, match='refusing to time the Python fallback'):
            cases[0].compiled()


def test_compiled_workspace_growth_preserves_all_fragmented_candidates() -> None:
    labels = np.zeros((3, 64, 64), dtype=np.uint16)
    labels[0, 20:44, 20:44] = 1
    for row in range(4):
        for col in range(4):
            labels[1, 22 + row * 5, 22 + col * 5] = 2 + row * 4 + col
    seed = interpolation.SliceEndpointSeed(label=1, point=(0, 32, 32), direction_sign=1)
    cache = interpolation.SliceComponentTableCache(labels)
    kwargs = dict(
        labels_real=labels, seed=seed, max_slice_distance=2, search_angle_deg=20.0,
        max_candidates=32, wrap_axis=False, component_cache=cache,
    )
    try:
        with (
            mock.patch.dict(os.environ, {
                'YOLO_TTA_INTERPOLATION_NUMBA_MAX_TRACKED_CANDIDATES': '8',
            }),
            mock.patch.object(interpolation.kernels, 'interpolation_find_projection_candidates',
                              wraps=interpolation.kernels.interpolation_find_projection_candidates) as kernel,
        ):
            assert interpolation.interpolation_projection_max_tracked() == 8
            compiled = interpolation._find_slice_projection_candidates_native(**kwargs)
            assert [int(call.args[-1]) for call in kernel.call_args_list] == [8, 16]
            python_candidates = _find_slice_projection_candidates_python(**kwargs)
            assert len(python_candidates) == 16
            np.testing.assert_array_equal(benchmark._rows(compiled), benchmark._rows(python_candidates))
            dispatched = interpolation._find_slice_projection_candidates(**kwargs)
            np.testing.assert_array_equal(benchmark._rows(dispatched), benchmark._rows(python_candidates))
    finally:
        cache.clear()


@pytest.mark.parametrize('capacity, expected_size, overflow', [(64, 0, 1), (79, 79, 0), (1024, 79, 0)])
def test_compiled_candidates_allocate_exact_arrays_after_a_wide_step(capacity, expected_size, overflow):
    labels = np.zeros((2, 1, 79), np.uint16)
    labels[0] = 1
    labels[1] = np.arange(2, 81, dtype=np.uint16)
    sdf = np.ones((1, 79), np.float32)
    result = interpolation.kernels.interpolation_find_projection_candidates(
        labels, np.empty(0, np.uint16), np.empty(0, np.int64), np.empty((0, 4), np.int64),
        False, 2, 79, np.zeros(1, np.uint32), np.empty(0, np.int64), sdf,
        0, 0, 0, 0, 39, 1, 1, 1, 0., 100, False, False, capacity,
    )
    assert result[-2:] == (expected_size, overflow)
    for column in result[:-2]:
        assert column.shape == (expected_size,) and column.dtype == np.int64
    if expected_size:
        np.testing.assert_array_equal(result[0], np.arange(2, 81))
        np.testing.assert_array_equal(result[1], np.ones(79))
        np.testing.assert_array_equal(result[3], np.arange(79))
        np.testing.assert_array_equal(result[5], (np.arange(79) - 39) ** 2)


def test_compiled_candidate_kernel_error_propagates() -> None:
    labels = np.zeros((3, 32, 32), dtype=np.uint16)
    labels[0, 10:20, 10:20] = 1
    labels[1, 14, 14] = 2
    seed = interpolation.SliceEndpointSeed(1, (0, 15, 15), 1)
    with mock.patch.object(
        interpolation.kernels, 'interpolation_find_projection_candidates',
        side_effect=RuntimeError('injected compiled failure'),
    ):
        with pytest.raises(RuntimeError, match='injected compiled failure'):
            interpolation._find_slice_projection_candidates(
                labels, seed, max_slice_distance=2, search_angle_deg=20.0,
                max_candidates=1,
            )
