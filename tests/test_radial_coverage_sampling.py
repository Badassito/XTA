"""Independent source-tap and terminal ownership checks for fewer Radial shells."""
from __future__ import annotations

from dataclasses import replace
import itertools
import math
import unittest

import numpy as np

from XTA import cylindrical_geometry as cg, cylindrical_projection as cp, geometry
from XTA.config import resolve_tilted_view_groups


def radial_views(shape, *, base='transverse', patch=8, minimum=2., tilt=None,
                 sampling_policy='coverage'):
    sources = () if tilt is None else geometry.get_view_infos(
        *shape, cartesian_views=(),
        tilt_groups=resolve_tilted_view_groups([f'{base}:{min(tilt, 45)}:both']),
    )
    if tilt is not None and tilt > 45:
        # The CLI already rejects these; exercise defensive metadata admission.
        sources = tuple(replace(v, tilt_angle_deg=math.copysign(tilt, v.tilt_angle_deg))
                        for v in sources)
    return cg.build_radial_view_infos(
        *shape, targets=(('tilted_' if tilt is not None else '') + base,),
        min_radius=minimum, patch_size=patch, tilted_views=sources,
        sampling_policy=sampling_policy,
    )


def source_domain(view):
    """Membership from Cartesian voxel centers, independent of sampled radii."""
    t, y, x = np.indices((view.full_t, view.full_h, view.full_w))
    if view.radial_base_view == 'transverse':
        stack, py, px, length = t, y, x, view.full_t
    elif view.radial_base_view == 'sagittal':
        stack, py, px, length = y, t, x, view.full_h
    else:
        stack, py, px, length = x, t, y, view.full_w
    dx, dy = px-view.center_x, py-view.center_y
    radius = np.hypot(dx, dy)
    height = stack.astype(np.float64)
    if view.radial_tilted_source:
        height -= math.tan(math.radians(view.tilt_angle_deg)) * (
            dy if view.tilt_direction == 'vertical' else dx)
    return ((radius >= view.radial_min_radius) & (radius <= view.radial_max_radius)
            & (height >= 0.) & (height <= length-1)), radius


def native_source_taps(views):
    first = views[0]
    shape = (first.full_t, first.full_h, first.full_w)
    touched = np.zeros(shape, bool)
    for view in views:
        for index in range(view.num_slices):
            *coords, valid = cg.shell_coordinates(view, index)
            lower = [np.floor(c).astype(np.intp) for c in coords]
            delta = [c-a for c, a in zip(coords, lower)]
            for corner in itertools.product((0, 1), repeat=3):
                ids = [a+k for a, k in zip(lower, corner)]
                weight = np.prod([d if k else 1.-d for d, k in zip(delta, corner)], axis=0)
                include = valid & (weight > 1e-12)
                for axis, length in zip(ids, shape):
                    include &= (axis >= 0) & (axis < length)
                touched[tuple(axis[include] for axis in ids)] = True
    return touched


class RadialCoverageSamplingTests(unittest.TestCase):
    def test_large_plan_saves_over_forty_percent_with_strict_certificate(self):
        dense = radial_views((3072, 3072, 3072), patch=3072, minimum=None,
                             sampling_policy='dense')
        reduced = radial_views((3072, 3072, 3072), patch=3072, minimum=None)
        dense_count = sum(v.num_slices for v in dense)
        count = sum(v.num_slices for v in reduced)
        self.assertEqual(dense_count, 2969)
        self.assertLess(count, .60*dense_count)
        self.assertEqual(sum(v.sampling_reference_frames for v in reduced), dense_count)
        for view in reduced:
            self.assertEqual(view.sampling_certificate, 'radial-positive-trilinear-v1')
            self.assertLessEqual(view.sampling_error_bound_sq, .99 + 1e-12)
            self.assertGreater(view.radial_step, 1.)
            self.assertEqual(view.radial_radii[-1], 1535.5)
            full = cg.global_radii(view)
            max_gap = float(np.diff(full).max())
            self.assertEqual(view.sampling_error_bound_sq,
                             (max_gap/2.)**2 + .25 + max_gap/(8.*view.radial_min_radius))
            self.assertEqual(full[view.radial_shell_start:], view.radial_radii)
            self.assertEqual(full[0], 3072/(4*math.pi))
            self.assertEqual(len(full), view.radial_global_count)
        for view in dense:
            self.assertEqual(view.radial_global_count, 0)
            self.assertEqual(view.sampling_policy, 'dense')
            self.assertEqual(view.sampling_certificate, '')
            self.assertEqual(view.sampling_reference_frames, 0)

    def test_native_positive_taps_cover_all_axes_and_thin_tilted_boundaries(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            for height, plane in ((1, 19), (3, 20)):
                shape = ((height, plane, plane+2) if base == 'transverse' else
                         (plane, height, plane+2) if base == 'sagittal' else
                         (plane, plane+2, height))
                for tilt in (None, 30, 45):
                    views = radial_views(shape, base=base, tilt=tilt)
                    groups = {(v.tilt_angle_deg, v.tilt_direction) for v in views}
                    for key in groups:
                        group = [v for v in views if (v.tilt_angle_deg, v.tilt_direction) == key]
                        self.assertTrue(all(v.sampling_certificate for v in group))
                        touched = native_source_taps(group)
                        wanted, _ = source_domain(group[0])
                        with self.subTest(base=base, height=height, plane=plane, tilt=key):
                            self.assertTrue(np.all(touched[wanted]),
                                            np.argwhere(wanted & ~touched).tolist())

    def test_projector_uses_new_global_grid_for_all_patch_suffixes(self):
        shape = (3, 19, 21)
        views = radial_views(shape)
        combined = np.zeros(shape, np.uint8)
        for view in views:
            native = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
            for index in range(view.num_slices):
                native[index] = (index + view.radial_shell_start) % 2 == 0
            radii = np.asarray(cg.global_radii(view))
            for z in range(shape[0]):
                combined[z] |= cp._pull_radial_chunk(native, view, radii, shape,
                                                     z, 0, shape[1]*shape[2]).reshape(shape[1:])
        wanted, radius = source_domain(views[0])
        # Scalar argmin independently assigns exact midpoints to the inner shell.
        selected = np.argmin(np.abs(radius[..., None] - np.asarray(cg.global_radii(views[0]))), axis=-1)
        expected = wanted & (selected % 2 == 0)
        np.testing.assert_array_equal(combined, expected.astype(np.uint8))

    def test_steep_tilts_and_unhelpful_bounds_keep_exact_dense_geometry(self):
        for kwargs in ({'tilt': 60}, {'minimum': .001}, {'minimum': 9.}):
            dense = radial_views((3, 19, 21), sampling_policy='dense', **kwargs)
            requested = radial_views((3, 19, 21), **kwargs)
            self.assertEqual(len(dense), len(requested))
            for old, new in zip(dense, requested):
                with self.subTest(kwargs=kwargs, view=new.name):
                    self.assertEqual(old.radial_radii, new.radial_radii)
                    self.assertEqual(old.radial_shell_start, new.radial_shell_start)
                    self.assertEqual(new.radial_global_count, 0)
                    self.assertEqual(new.sampling_policy, 'dense')
                    self.assertEqual(new.sampling_certificate, '')
                    self.assertTrue(new.sampling_reason)
                    self.assertEqual(new.sampling_reference_frames, 0)
                    self.assertEqual(new, replace(old, sampling_reason=new.sampling_reason))

    def test_count_validation_and_legacy_global_grid(self):
        view = radial_views((3, 19, 21))[0]
        for value in (-1, 1, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cg.global_radii(replace(view, radial_global_count=value))
        legacy = replace(view, radial_global_count=0, sampling_policy='dense')
        self.assertEqual(cg.global_radii(legacy), cg.radius_grid(2., 9.))
        for minimum in (0., -1., math.nan, math.inf):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                cg.certified_radial_gap(minimum)
        with self.assertRaisesRegex(ValueError, 'sampling policy'):
            radial_views((3, 19, 21), sampling_policy='skip')

    def test_certificate_checks_actual_grid_and_rejects_tampered_metadata(self):
        view = radial_views((3, 19, 21))[0]
        mutations = (
            dict(radial_global_count=0),
            dict(sampling_certificate=''),
            dict(sampling_certificate='radial-positive-trilinear-v0'),
            dict(tilt_angle_deg=46.),
            dict(tilt_angle_deg=math.nan),
            dict(tilt_angle_deg=math.inf),
            dict(sampling_error_bound_sq=0.),
            dict(sampling_error_bound_sq=math.nan),
            dict(sampling_error_bound_sq=view.sampling_error_bound_sq + .001),
            dict(radial_global_count=view.radial_global_count + 1),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cg.global_radii(replace(view, **mutation))
        # Even a matching declared bound cannot certify an over-wide grid.
        gap = view.radial_max_radius - view.radial_min_radius
        declared = (gap/2.)**2 + .25 + gap/(8.*view.radial_min_radius)
        with self.assertRaisesRegex(ValueError, 'error budget'):
            cg.global_radii(replace(view, radial_global_count=2,
                                   sampling_error_bound_sq=declared))


if __name__ == '__main__':
    unittest.main()
