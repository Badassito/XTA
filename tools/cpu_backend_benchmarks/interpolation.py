"""Bounded, real interpolation projection-candidate planner cases.

The same warmed component/SDF cache is passed to the explicit Python and Numba
entry points. Benchmark trials therefore measure candidate scanning, not table
construction or first-call compilation. A compiled refusal is an error here:
the production dispatcher may fall back, but timing that fallback as Numba
would mislabel the result.
"""
from __future__ import annotations

import numpy as np

from XTA import interpolation
from tests.reference_backends.interpolation import _find_slice_projection_candidates_python

from .contracts import Case


def _rows(candidates: list[interpolation.SliceProjectionCandidate]) -> np.ndarray:
    """Canonical semantic fields in stable candidate order, including empty output."""
    return np.asarray([
        (
            int(candidate.source_label), int(candidate.target_label),
            *(int(value) for value in candidate.source_point),
            *(int(value) for value in candidate.target_point),
            int(candidate.slice_distance),
        )
        for candidate in candidates
    ], dtype=np.int64).reshape(-1, 9)


def _fixture(workload: str, *, fragmented: bool) -> tuple[np.ndarray, interpolation.SliceEndpointSeed, int, int]:
    if workload == 'smoke':
        slices, side, source_start, source_stop, grid, step = 7, 64, 20, 44, (4 if fragmented else 2), 5
    else:
        slices, side, source_start, source_stop, grid, step = 13, 256, 64, 192, (12 if fragmented else 2), 8
    labels = np.zeros((slices, side, side), dtype=np.uint16)
    labels[0, source_start:source_stop, source_start:source_stop] = 1
    # Isolated one-pixel components model heavily fragmented target slices.
    # Coherent targets use four filled 3x3 components instead.
    first = source_start + 2
    for row in range(grid):
        for col in range(grid):
            y, x = first + row * step, first + col * step
            label = 2 + row * grid + col
            if fragmented:
                labels[1, y, x] = label
            else:
                labels[1, y:y + 3, x:x + 3] = label
    seed = interpolation.SliceEndpointSeed(
        label=1, point=(0, (source_start + source_stop) // 2, (source_start + source_stop) // 2),
        direction_sign=1,
    )
    return labels, seed, min(4, slices - 1), grid * grid


def _case(workload: str, *, fragmented: bool) -> Case:
    if interpolation._numba_find_projection_candidates_kernel is None:
        raise RuntimeError('Interpolation compiled CPU benchmark requires Numba')
    labels, seed, max_distance, max_candidates = _fixture(workload, fragmented=fragmented)
    cache = interpolation.SliceComponentTableCache(labels)
    source, _anchor = cache.find_record_for_point(seed.point[0], seed.label, seed.point[1:])
    if source is None:
        raise RuntimeError('Interpolation benchmark source component was not found')
    sdf = cache.get_projection_sdf(
        source, max_slice_distance=max_distance, search_angle_deg=20.0,
    )
    kwargs = dict(
        labels_real=labels, seed=seed, max_slice_distance=max_distance,
        search_angle_deg=20.0, max_candidates=max_candidates, wrap_axis=False,
        component_cache=cache,
    )

    def reference() -> np.ndarray:
        return _rows(_find_slice_projection_candidates_python(**kwargs))

    def compiled() -> np.ndarray:
        candidates = interpolation._find_slice_projection_candidates_numba(**kwargs)
        if candidates is None:
            raise RuntimeError(
                'Interpolation compiled projection-candidate planner declined this fixture; '
                'refusing to time the Python fallback as compiled'
            )
        return _rows(candidates)

    # Warm the actual signature and reject any accidentally empty or inequivalent
    # fixture before the runner starts timing trials.
    expected = reference()
    observed = compiled()
    if expected.shape[0] != max_candidates or not np.array_equal(expected, observed):
        raise RuntimeError('Interpolation projection-candidate benchmark fixture is invalid')

    return Case(
        name=f'interpolation_{workload}_{"fragmented" if fragmented else "coherent"}_python_vs_numba',
        reference=reference,
        compiled=compiled,
        work_units=int(sdf.sdf.size),
        unit='candidate_scan_pixels',
        metadata={
            'family': 'interpolation', 'workload': workload,
            'reference_oracle': 'tests.reference_backends.interpolation._find_slice_projection_candidates_python',
            'production_compiled': 'XTA.interpolation._find_slice_projection_candidates_numba',
            'setup_included_in_trials': False,
            'labels_shape': list(labels.shape), 'labels_dtype': str(labels.dtype),
            'seed_point': list(seed.point), 'source_component_area': int(source.area),
            'sdf_window_shape': list(sdf.sdf.shape), 'max_slice_distance': int(max_distance),
            'scan_steps': 1,
            'search_angle_deg': 20.0, 'max_candidates': int(max_candidates),
            'candidate_count': int(expected.shape[0]), 'fragmented': bool(fragmented),
            'numba_workspace_cap': int(interpolation.interpolation_projection_numba_max_tracked()),
        },
    )


def make_cases(workload: str = 'smoke') -> list[Case]:
    """Return coherent and fragmented candidate-planning cases without timers or GPU work."""
    if workload not in ('smoke', 'scaled'):
        raise ValueError(f'Unknown interpolation benchmark workload {workload!r}')
    return [_case(workload, fragmented=False), _case(workload, fragmented=True)]
