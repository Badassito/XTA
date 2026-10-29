"""Validate the admission prototype's hard bounds and starvation counterexample."""
from __future__ import annotations

import random
import math
from collections import Counter
import sys
import unittest

from tools.simulate_policy_parent_admission import GIB, Group, cluster_workload, simulate


class PolicyParentAdmissionSimulatorTests(unittest.TestCase):
    def test_current_geometry_inventory_for_historical_workload_preset(self):
        watched_modules = ('cv2', 'scipy', 'scipy.ndimage', 'tifffile', 'tqdm', 'numba',
                           'XTA._deps', 'XTA.geometry', 'XTA.unification.runtime')
        before = {name: sys.modules.get(name) for name in watched_modules}
        groups, metadata = cluster_workload()
        for name, module in before.items():
            with self.subTest(module=name):
                self.assertIs(sys.modules.get(name), module)
        self.assertEqual(metadata['geometry_source'], 'current_physical_view_compiler')
        self.assertEqual(metadata['shape_tyx'], (2911, 3064, 3022))
        self.assertEqual(metadata['raster'], 3072)
        # Current coverage geometry for this shape/preset, not job 144736's old
        # sampler. Derive the inventory without importing the physical compiler.
        # Three axes each have an upright pose and four signed 30-degree tilts.
        diameters = (3022, 2911, 2911)
        upright_angles = [math.ceil(math.pi * d / (4 * math.sqrt(.99 - .25))) for d in diameters]
        conservative_angles = [math.ceil(math.pi * d / 2) for d in diameters]
        # Radial positive-tap certificate: g²/4 + 1/4 + g/(8*r_min) <= .99,
        # r_min=3072/(4*pi), r_max=(diameter-1)/2. Endpoint-inclusive shells
        # retain these suffix lengths at arc origins 0,3072,6144[,9216].
        radial_frames = [738, 595, 311, 26, 706, 563, 278, 706, 563, 278]
        # QSC 41/50 certificate at raster 3072: eight arcsine-spaced bands,
        # n=3070, radii counts 85,85,85,85,86,86,86,86. Each band has six
        # faces and each face one patch, for five distinct cube poses.
        spherical_frames = ([85] * 4 + [86] * 4) * (6 * 5)
        expected = {
            'orthogonal': [2911, 3064, 3022],
            'azimuthal': upright_angles,
            'tilted': [2911, 3064, 3022] * 4,
            'tilted_azimuthal': conservative_angles * 4,
            'radial': radial_frames * 5,
            'spherical': spherical_frames,
        }
        self.assertEqual(
            Counter((group.family, group.frames) for group in groups),
            Counter((family, frames) for family, counts in expected.items() for frames in counts),
        )
        family_counts = {family: len(counts) for family, counts in expected.items()}
        self.assertEqual(family_counts, dict(orthogonal=3, azimuthal=3, tilted=12,
                                           tilted_azimuthal=12, radial=50, spherical=240))
        self.assertEqual(metadata['group_counts_by_family'], family_counts)
        self.assertEqual(metadata['group_count'], sum(family_counts.values()))
        self.assertEqual(metadata['rendered_frames'], sum(map(sum, expected.values())))
        for group in groups:
            self.assertEqual(group.pass_bytes, math.prod(group.processing_shape))
            self.assertEqual(group.processing_shape[0], group.frames)
        oversized = [group.pass_bytes * 4 / GIB for group in groups if group.pass_bytes * 4 > 128 * GIB]
        self.assertEqual(len(oversized), 12)
        self.assertAlmostEqual(min(oversized), 149.8642305508256)
        self.assertAlmostEqual(max(oversized), 155.56647767871618)

    def test_finite_small_backlog_defers_large_groups_until_tail_without_reservation(self):
        groups = [Group(f'small-{index}', 'radial', 128, GIB) for index in range(20)]
        groups += [Group(f'large-{index}', 'tilted_azimuthal', 1024, 40 * GIB) for index in range(2)]
        current = simulate(groups, policy='current')
        fair = simulate(groups, policy='reserve_oversized', small_group_quota=4)
        self.assertEqual(current['oversized_admissions'][0]['admission_position'], 20)
        self.assertEqual(fair['oversized_admissions'][0]['admission_position'], 4)
        self.assertLess(fair['oversized_admissions'][0]['time_units'], current['oversized_admissions'][0]['time_units'])
        for result in (current, fair):
            self.assertEqual(result['completed_independent_parents'], 88)
            self.assertEqual(result['completed_model_frames'], sum(group.frames for group in groups) * 4)
            self.assertLessEqual(result['peak_total_dense_bytes'], 384 * GIB)

    def test_reservation_drains_existing_group_instead_of_stranding_its_chunks(self):
        groups = [Group('already-open', 'orthogonal', 8193, 20 * GIB),
                  Group('small', 'radial', 129, GIB),
                  Group('oversized', 'tilted_azimuthal', 500, 40 * GIB)]
        result = simulate(groups, policy='reserve_oversized', small_group_quota=1, lease_frames=64)
        self.assertEqual(result['admission_order'], ['already-open', 'oversized', 'small'])
        self.assertGreater(result['reservation_drain_time_units'], 0)
        self.assertEqual(result['completed_independent_parents'], 12)
        self.assertEqual(result['completed_model_frames'], (8193 + 129 + 500) * 4)

    def test_no_oversized_groups_means_identical_policy_scheduling(self):
        groups = [Group(f'group-{index}', 'radial', 81 + index, (index + 1) * GIB) for index in range(10)]
        for ratio in (1, 2):
            baseline = simulate(groups, policy='current', ratio=ratio)
            fair = simulate(groups, policy='reserve_oversized', ratio=ratio)
            baseline.pop('policy')
            fair.pop('policy')
            self.assertEqual(baseline, fair)

    def test_varied_work_costs_and_pass_counts_always_finish_within_hard_cap(self):
        rng = random.Random(144736)
        groups = [Group(f'group-{index}', 'synthetic', rng.randrange(1, 1000), rng.randrange(1, 46) * GIB)
                  for index in range(24)]
        for ratio in (1, 2, 4):
            for postprocess_factor in (.01, .5, 8.):
                for policy in ('current', 'reserve_oversized'):
                    with self.subTest(ratio=ratio, postprocess=postprocess_factor, policy=policy):
                        result = simulate(groups, policy=policy, ratio=ratio,
                                          postprocess_factor=postprocess_factor, small_group_quota=3)
                        self.assertEqual(len(set(result['admission_order'])), len(groups))
                        self.assertEqual(result['completed_independent_parents'], len(groups) * ratio)
                        self.assertEqual(result['completed_model_frames'], sum(group.frames for group in groups) * ratio)
                        self.assertLessEqual(result['peak_total_dense_bytes'], 384 * GIB)
                        self.assertLessEqual(result['peak_inference_groups'], 4)

    def test_impossible_group_and_invalid_cost_fail_before_simulation(self):
        with self.assertRaisesRegex(ValueError, 'exceeds the hard total dense cap'):
            simulate([Group('too-big', 'synthetic', 1, 100 * GIB)], policy='current')
        for value in (0, -1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                simulate([Group('valid', 'synthetic', 1, GIB)], policy='current', postprocess_factor=value)


if __name__ == '__main__':
    unittest.main()
