"""Bounded production CPU projection cases; this module does not time them.

Factories build deterministic mask geometry and compile optional kernels before
the runner starts trials. Every case returns one small output block. Neither a
full native output volume nor a GPU is allocated here.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from XTA import cylindrical_projection as radial
from XTA import geometry
from XTA import spherical_projection as spherical
from XTA.spherical_geometry import build_spherical_view_infos
from XTA.spherical_projection_bounds import spherical_output_bounds
from XTA.spherical_projection_cpu import (
    SphericalCpuProjectionUnavailable,
    prepare_spherical_chunk_numba,
)
from tests.reference_backends.spherical import project_spherical_block as reference_spherical_block
from tests.reference_backends.radial import pull_radial_chunk as reference_radial_chunk

from .contracts import Case


def _filled_rectangles(view, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sparse, nonempty, C-contiguous uint8 masks with exact per-shell bounds."""
    shape = (int(view.num_slices), int(view.src_h), int(view.src_w))
    source = np.zeros(shape, dtype=np.uint8)
    boxes = np.zeros((shape[0], 4), dtype=np.int64)
    height = max(1, min(shape[1], shape[1] // 5))
    width = max(1, min(shape[2], shape[2] // 4))
    for shell in range(shape[0]):
        y0 = (shell * 7 + seed) % (shape[1] - height + 1)
        x0 = (shell * 11 + seed * 3) % (shape[2] - width + 1)
        source[shell, y0:y0 + height, x0:x0 + width] = 1
        boxes[shell] = (y0, y0 + height, x0, x0 + width)
    return source, boxes


def _spherical_case(workload: str) -> Case:
    if workload == 'smoke':
        working_shape, output_shape, patch_size, count = (43, 65, 67), (37, 69, 71), 32, 1
    else:
        working_shape, output_shape, patch_size, count = (513, 547, 571), (389, 563, 557), 96, 4
    views = build_spherical_view_infos(
        *working_shape, targets=('transverse',),
        min_radius=patch_size / (4 * math.pi), patch_size=patch_size,
        tilted_views=(),
    )
    view = next((candidate for candidate in views if candidate.spherical_face == 5), views[0])
    source, boxes = _filled_rectangles(view, seed=19)
    # This face occupies only part of the output volume. A filled source makes
    # the bounded sample exercise real QSC reads rather than an empty fast exit.
    source.fill(1)
    boxes[:] = (0, int(view.src_h), 0, int(view.src_w))
    radii = np.asarray(view.spherical_radii, dtype=np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, dtype=np.float64).reshape(3, 3)
    bounds = spherical_output_bounds(view, output_shape, boxes)
    first = min(max(0, (int(bounds.z0) + int(bounds.z1) - count) // 2), output_shape[0] - count)
    try:
        compiled_pull = prepare_spherical_chunk_numba(
            source, view, radii, rotation, output_shape, boxes,
        )
    except SphericalCpuProjectionUnavailable as exc:
        raise RuntimeError(f'Spherical compiled CPU benchmark unavailable: {exc}') from exc

    def reference() -> np.ndarray:
        return reference_spherical_block(
            source, view, radii, rotation, output_shape, first, count,
            boxes, bounds,
        )

    def compiled() -> np.ndarray:
        return spherical._project_spherical_block(
            source, view, radii, rotation, output_shape, first, count,
            boxes, bounds, cpu_pull=compiled_pull,
        )

    return Case(
        name=f'spherical_{workload}_numpy_vs_numba',
        reference=reference,
        compiled=compiled,
        work_units=count * output_shape[1] * output_shape[2],
        unit='output_voxels',
        metadata={
            'family': 'spherical', 'workload': workload,
            'reference': 'tests.reference_backends.spherical.project_spherical_block',
            'production_compiled': 'XTA.spherical_projection._project_spherical_block(cpu_pull=prepare_spherical_chunk_numba(...))',
            'setup_included_in_trials': False,
            'working_shape': list(working_shape), 'output_shape': list(output_shape),
            'source_shape': list(source.shape), 'first_z': first, 'block_depth': count,
            'bounded_yx': [int(bounds.y0), int(bounds.y1), int(bounds.x0), int(bounds.x1)],
            'known_slice_bboxes': True, 'mask_pattern': 'filled_face',
            'spherical_face': int(view.spherical_face),
        },
    )


def _spherical_native_plane_case() -> Case:
    """One native-coordinate output plane with a broadcast input mask.

    This follows ``tools/benchmark_spherical_cpu_pull.py``: one allocated
    3072-square source plane is read across all shells. It preserves the real
    source strides and label boundaries without allocating the logical 11 GiB
    input or the full 18 GiB output volume.
    """
    working_shape, output_shape, patch_size = (2911, 3064, 3022), (1931, 3064, 3022), 3072
    views = build_spherical_view_infos(
        *working_shape, targets=('transverse',), min_radius=None,
        patch_size=patch_size, tilted_views=(),
    )
    view = next(candidate for candidate in views if candidate.spherical_face == 0)
    rng = np.random.default_rng(142828)
    source_plane = (rng.random((patch_size, patch_size), dtype=np.float32) < 0.06).astype(np.uint8)
    source_plane[::37, :] = 1
    source_plane[:, ::41] = 1
    source_plane[np.arange(patch_size), np.arange(patch_size)] = 1
    source = np.broadcast_to(source_plane, (int(view.num_slices), patch_size, patch_size))
    radii = np.asarray(view.spherical_radii, dtype=np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, dtype=np.float64).reshape(3, 3)
    bounds = spherical_output_bounds(view, output_shape)
    first = output_shape[0] // 2
    if not int(bounds.z0) <= first < int(bounds.z1):
        raise RuntimeError('Native Spherical sample plane lies outside analytic output bounds')
    try:
        compiled_pull = prepare_spherical_chunk_numba(
            source, view, radii, rotation, output_shape,
        )
    except SphericalCpuProjectionUnavailable as exc:
        raise RuntimeError(f'Native Spherical compiled CPU benchmark unavailable: {exc}') from exc

    def reference() -> np.ndarray:
        return reference_spherical_block(
            source, view, radii, rotation, output_shape, first, 1,
            output_bounds=bounds,
        )

    def compiled() -> np.ndarray:
        return spherical._project_spherical_block(
            source, view, radii, rotation, output_shape, first, 1,
            output_bounds=bounds, cpu_pull=compiled_pull,
        )

    return Case(
        name='spherical_native_plane_numpy_vs_numba',
        reference=reference,
        compiled=compiled,
        work_units=output_shape[1] * output_shape[2],
        unit='output_voxels',
        metadata={
            'family': 'spherical', 'workload': 'scaled_native_plane',
            'reference': 'tests.reference_backends.spherical.project_spherical_block',
            'production_compiled': 'XTA.spherical_projection._project_spherical_block(cpu_pull=prepare_spherical_chunk_numba(...))',
            'setup_included_in_trials': False,
            'working_shape': list(working_shape), 'output_shape': list(output_shape),
            'source_shape': list(source.shape), 'source_allocated_bytes': int(source_plane.nbytes),
            'source_logical_bytes': int(source.nbytes), 'source_strides': list(source.strides),
            'source_broadcast_across_shells': True,
            'first_z': first, 'block_depth': 1,
            'output_block_bytes': int(output_shape[1] * output_shape[2]),
            'bounded_yx': [int(bounds.y0), int(bounds.y1), int(bounds.x0), int(bounds.x1)],
            'known_slice_bboxes': False, 'spherical_face': int(view.spherical_face),
            'limitations': 'Synthetic broadcast source reuses one mask plane across shells; no full-volume memory traffic.',
        },
    )


def _radial_case(workload: str) -> Case:
    if workload == 'smoke':
        working_shape, output_shape, patch_size, count = (43, 65, 67), (37, 69, 71), 32, 1
    else:
        working_shape, output_shape, patch_size, count = (513, 547, 571), (389, 563, 557), 256, 4
    if radial._numba is None or not hasattr(radial._project_radial_block, 'signatures'):
        raise RuntimeError('Radial compiled CPU benchmark requires Numba')
    views = geometry.get_view_infos(
        *working_shape, cartesian_views=(), radial_views=('transverse',),
        radial_patch_size=patch_size,
    )
    view = views[len(views) // 2]
    source, boxes = _filled_rectangles(view, seed=23)
    radii = np.asarray(geometry.radial_global_radii(view), dtype=np.float64)
    try:
        plan, _ = radial._radial_plane_plan(view, radii, output_shape)
    except radial._RadialPlanePlanTooLarge as exc:
        raise RuntimeError(f'Radial factored CPU benchmark has no bounded plan: {exc}') from exc
    centers, ideal, sampled, row_map, column_map, stack_length, vertical = (
        radial._radial_projection_metadata(view, source.shape, output_shape, plan)
    )
    arguments = (
        source, plan.shell_index, plan.column_offsets, plan.native_columns,
        sampled, row_map, column_map, centers, ideal, int(stack_length),
        int(view.radial_height_origin), int(view.src_h), plan.base_id,
        bool(vertical), int(plan.plane_shape[1]),
        int(output_shape[1]), int(output_shape[2]),
    )
    first = output_shape[0] // 2
    # The pipeline compiles this exact signature before publishing the first slice.
    radial._project_radial_block(*arguments, first, 0, boxes, True)

    def reference() -> np.ndarray:
        block = np.empty((count, *output_shape[1:]), dtype=np.uint8)
        for local_z in range(count):
            flat = block[local_z].reshape(-1)
            for pixel in range(0, flat.size, radial._PULL_CHUNK_VOXELS):
                stop = min(flat.size, pixel + radial._PULL_CHUNK_VOXELS)
                flat[pixel:stop] = reference_radial_chunk(
                    source, view, radii, output_shape, first + local_z, pixel, stop,
                )
        return block

    def compiled() -> np.ndarray:
        return radial._project_radial_block(*arguments, first, count, boxes, True)

    return Case(
        name=f'radial_{workload}_numpy_reference_vs_numba_factored',
        reference=reference,
        compiled=compiled,
        work_units=count * output_shape[1] * output_shape[2],
        unit='output_voxels',
        metadata={
            'family': 'radial', 'workload': workload,
            'reference': 'tests.reference_backends.radial.pull_radial_chunk',
            'production_compiled': 'XTA.cylindrical_projection._project_radial_block',
            'setup_included_in_trials': False,
            'working_shape': list(working_shape), 'output_shape': list(output_shape),
            'source_shape': list(source.shape), 'first_z': first, 'block_depth': count,
            'known_slice_bboxes': True, 'plan_bytes': int(plan.nbytes),
            'radial_patch_index': int(view.radial_patch_index),
        },
    )


def make_cases(workload: str = 'smoke') -> list[Case]:
    """Build cases without timers, GPUs, or any full-volume output allocation."""
    if workload not in ('smoke', 'scaled'):
        raise ValueError(f'Unknown projection benchmark workload {workload!r}')
    cases = [_spherical_case(workload), _radial_case(workload)]
    if workload == 'scaled':
        cases.append(_spherical_native_plane_case())
    return cases
