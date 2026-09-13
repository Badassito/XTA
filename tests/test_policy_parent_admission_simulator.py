"""Validate the admission prototype's hard bounds and starvation counterexample."""
from __future__ import annotations

import random
import unittest

from tools.simulate_policy_parent_admission import GIB, Group, cluster_workload, simulate


class PolicyParentAdmissionSimulatorTests(unittest.TestCase):
    def test_real_geometry_matches_cluster_group_and_frame_inventory(self):
        groups, metadata = cluster_workload()
        self.assertEqual(len(groups), 110)
        self.assertEqual(metadata['rendered_frames'], 160593)
        self.assertEqual(metadata['group_counts_by_family']['tilted_azimuthal'], 12)
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
