"""Independent edge-case and resource invariants for reconciliation policies."""
from __future__ import annotations

import tracemalloc
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np

from XTA.reconciliation import EvidenceLayer, reconcile
from XTA.reconciliation_policy import validate_policy, resolve_reconciliation, load_reconciliation_policy
from XTA.reconciliation_geometry import section_codes, section_descriptor
from XTA import reconciliation_geometry


def evidence(name, mask, *, view=None, scores=None, observed=None, **metadata):
    info = dict(source='fullframe', mask_kind='yolo', view_name=view or name,
                physical_view_name=view or name, view_family='orthogonal', **metadata)
    confidence_reader = None
    if scores is not None:
        confidence_reader = lambda a, b: (scores[a:b].copy(), observed[a:b].copy())
    return EvidenceLayer(name, mask.shape, info, lambda a, b: mask[a:b].copy(), confidence_reader)


def execute(layers, shape, policy, *, context=None, memory_mib=8):
    output = np.zeros(shape, dtype=np.uint8)
    writes = []

    def write(first, stop, mask):
        writes.append((first, stop))
        output[first:stop] = mask

    report = reconcile(layers, shape_tyx=shape, policy=policy, write_slab=write,
                       geometry_context=context, memory_mib=memory_mib)
    return output, report, writes


class ReconciliationInvariantTests(unittest.TestCase):
    def test_policy_executes_verified_source_after_same_size_same_timestamp_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.py'
            path.write_text('def build_reconciliation():\n    return dict(name="first")\n')
            original_stat = path.stat()
            args = SimpleNamespace(reconciliation=str(path), reconciliation_memory_mib=64)
            first = resolve_reconciliation(args)
            self.assertEqual(load_reconciliation_policy(first)['name'], 'first')
            path.write_text('def build_reconciliation():\n    return dict(name="other")\n')
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            second = resolve_reconciliation(args)
            self.assertNotEqual(first.sha256, second.sha256)
            self.assertEqual(load_reconciliation_policy(second)['name'], 'other')

    def test_repeated_same_view_evidence_cannot_change_island_normalization(self):
        shape = (1, 1, 100)
        masks = [np.zeros(shape, np.uint8) for _ in range(3)]
        masks[0][..., :4] = 1
        masks[1][..., :16] = 1
        masks[2][..., 32:96] = 1
        original = [evidence(name, mask) for name, mask in zip(('a', 'b', 'c'), masks)]
        repeated = original + [evidence(f'a_repeat_{i}', masks[0], view='a') for i in range(20)]
        for grouping in ('views', 'sections'):
            with self.subTest(grouping=grouping):
                policy = dict(grouping=grouping, island_weighting=True, threshold=2., min_sources=2)
                expected, base_report, _ = execute(original, shape, policy)
                actual, repeated_report, _ = execute(repeated, shape, policy)
                np.testing.assert_array_equal(actual, expected)
                for name in ('a', 'b', 'c'):
                    self.assertEqual(repeated_report['layers'][name]['island_weight'],
                                     base_report['layers'][name]['island_weight'])

    def test_permissive_builtin_thresholds_still_return_only_candidate_voxels(self):
        shape = (2, 3, 4)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[:, 1, 1:3] = 1
        policy = dict(grouping='views', threshold=0, min_sources=0, min_prediction_sources=0)
        output, _, _ = execute([evidence('one', mask)], shape, policy)
        np.testing.assert_array_equal(output, mask)
        empty, report, _ = execute([], shape, policy)
        self.assertFalse(empty.any())
        self.assertEqual(report['counts']['retained_voxels'], 0)

    def test_validated_angular_range_matches_geometry_code_range(self):
        descriptor = section_descriptor(dict(physical_view_name='transverse', view_family='orthogonal'))
        for tolerance in (0.001, 0.009, 0.01, 1., 15.):
            with self.subTest(tolerance=tolerance):
                try:
                    policy = validate_policy(dict(angular_tolerance_deg=tolerance))
                except ValueError:
                    continue
                # A policy accepted before inference must not fail solely because
                # the geometry helper accepts a narrower parameter range.
                section_codes(descriptor, 0, 1, (1, 3, 4),
                              angular_tolerance_deg=policy['angular_tolerance_deg'])

    def test_only_overlapping_positive_votes_count_as_removed_duplicates(self):
        shape = (1, 1, 3)
        first = evidence('transverse', np.array([[[1, 1, 0]]], np.uint8))
        second = evidence('tilted_transverse_horizontal_p0p1',
                          np.array([[[0, 1, 1]]], np.uint8))
        output, report, _ = execute([first, second], shape, dict(threshold=1., min_sources=1))
        self.assertTrue(output.all())
        self.assertEqual(report['group_count'], 2)
        self.assertEqual(report['counts']['duplicate_group_support_removed'], 1)

    def test_section_geometry_reuses_xy_codes_and_recomputes_z_dependent_codes(self):
        shape = (3, 5, 5)
        mask = np.ones(shape, np.uint8)
        layers = [EvidenceLayer(name, shape, dict(physical_view_name=name, view_family='azimuthal',
                  mask_kind='yolo', azimuths_deg=(0., 45., 90., 135.)), lambda a, b: mask[a:b])
                  for name in ('azimuthal_transverse', 'azimuthal_sagittal')]
        context = dict(processing_shape_tyx=shape, source_shape_tyx=shape)
        policy = dict(threshold=1., min_sources=2, min_prediction_sources=0)
        expected, _, _ = execute(layers, shape, policy, context=context)
        with patch.object(reconciliation_geometry, 'section_codes', wraps=section_codes) as code:
            output, _, writes = execute(layers, shape, policy, context=context, memory_mib=.005)
        self.assertEqual(writes, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual([(call.args[0]['base_view'], *call.args[1:3]) for call in code.call_args_list],
                         [('transverse', 0, 1), ('sagittal', 0, 1), ('sagittal', 1, 2), ('sagittal', 2, 3)])
        np.testing.assert_array_equal(output, expected)

    def test_score_slab_keeps_boundaries_at_policy_threshold_precision(self):
        shape = (1, 1, 1)
        layer = evidence('transverse', np.ones(shape, np.uint8))
        policy = dict(threshold=np.nextafter(1., 2.), min_sources=1)
        output, _, _ = execute([layer], shape, policy)
        self.assertFalse(output.any())

    def test_active_anchor_bonus_keeps_custom_score_precision(self):
        shape = (1, 1, 1)
        mask = np.ones(shape, np.uint8)
        layers = [evidence(name, mask, scores=np.full(shape, 255, np.uint8), observed=mask.astype(bool))
                  for name in ('transverse', 'sagittal')]

        def decide(block):
            self.assertEqual(block['score'].dtype, np.float64)
            self.assertEqual(block['score'].item(), 2. + 1e-8)
            return block['score'] > 2.

        output, _, _ = execute(layers, shape, dict(mode='confidence', anchor_bonus=1e-8, decide=decide))
        self.assertTrue(output.all())

    def test_builtin_anchor_bonus_reuses_one_plane_at_float64_policy_precision(self):
        shape = (17, 2, 3)
        mask = np.ones(shape, np.uint8)
        observed = mask.astype(bool)
        observed[:, 0, 0] = False
        layers = [
            evidence('transverse', mask, scores=np.full(shape, 255, np.uint8), observed=observed),
            evidence('sagittal', mask, scores=np.full(shape, 255, np.uint8), observed=mask.astype(bool)),
        ]
        anchor_bonus = 1e-8
        threshold = np.nextafter(2. + anchor_bonus, 2.)
        policy = dict(mode='confidence', anchor_bonus=anchor_bonus, threshold=threshold)
        with patch.object(np, 'copyto', wraps=np.copyto) as copy:
            output, _, writes = execute(layers, shape, policy)
        self.assertEqual(writes, [(0, 16), (16, 17)])
        self.assertEqual(copy.call_count, shape[0])
        plane = copy.call_args_list[0].args[0]
        self.assertEqual((plane.shape, plane.dtype), (shape[1:], np.float64))
        self.assertTrue(all(call.args[0] is plane for call in copy.call_args_list))
        np.testing.assert_array_equal(output, observed)

    def test_shared_azimuth_cartesian_section_cannot_create_an_independent_anchor(self):
        shape = (3, 5, 5)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[:, 2, 4] = 1  # Same sagittal section as azimuth zero.
        mask[:, 3, 4] = 1  # Nearest sampled azimuth is 45 degrees, independently oriented.
        known = mask.astype(bool)
        high, low = np.full(shape, 230, np.uint8), np.full(shape, 70, np.uint8)
        layers = [evidence('cartesian', mask, view='sagittal', scores=high, observed=known),
                  evidence('azimuth', mask, view='azimuthal_transverse', scores=low, observed=known,
                           azimuths_deg=(0., 45., 90., 135.))]
        layers[1].metadata = {**layers[1].metadata, 'view_family': 'azimuthal'}
        policy = dict(mode='confidence', grouping='sections', threshold=1.2,
                      min_sources=2, min_prediction_sources=2,
                      anchor_confidence=.8, anchor_bonus=.2)
        context = dict(processing_shape_tyx=shape, source_shape_tyx=shape)
        output, report, _ = execute(layers, shape, policy, context=context)
        self.assertFalse(output[:, 2, 4].any())
        self.assertTrue(output[:, 3, 4].all())
        self.assertEqual(report['counts']['anchored_voxels'], 3)

    def test_confidence_zero_is_unknown_even_if_observation_bitmap_is_true(self):
        shape = (1, 2, 3)
        mask = np.ones(shape, np.uint8)
        observed = np.ones(shape, bool)
        low = evidence('quantized_zero', mask, scores=np.zeros(shape, np.uint8), observed=observed)
        high = evidence('high', mask, scores=np.full(shape, 230, np.uint8), observed=observed)
        output, report, _ = execute([low, high], shape,
            dict(mode='confidence', grouping='views', threshold=.8,
                 min_sources=1, min_prediction_sources=1))
        self.assertTrue(output.all())
        self.assertEqual(report['counts']['confidence_unknown_prediction_voxels'], 6)
        self.assertEqual(report['counts']['confidence_known_voxels'], 6)
        self.assertEqual(report['counts']['anchored_voxels'], 0)

    def test_dynamic_geometry_working_memory_is_within_accepted_budget(self):
        shape = (1, 256, 256)
        context = dict(processing_shape_tyx=shape, source_shape_tyx=shape,
                       views_by_name={
                           'azimuthal_transverse': dict(name='azimuthal_transverse', family='azimuthal',
                               azimuthal_base_view='transverse', azimuths_deg=(0., 45., 90., 135.)),
                           'sagittal': dict(name='sagittal', family='orthogonal', horizontal_axis='x',
                                           vertical_axis='t', stack_axis='y'),
                       })
        layers = [EvidenceLayer(name, shape,
                    dict(source='fullframe', mask_kind='yolo', physical_view_name=name),
                    lambda a, b: np.ones((b-a, *shape[1:]), dtype=np.uint8))
                  for name in context['views_by_name']]
        # Imports/context allocation happen before measuring algorithm workspace.
        section_codes(section_descriptor(layers[0].metadata, geometry_context=context),
                      0, 1, (1, 2, 2), context)
        for memory_mib in (5, 64):
            budget = memory_mib * 1024 * 1024
            with self.subTest(memory_mib=memory_mib):
                tracemalloc.start()
                try:
                    try:
                        report = reconcile(layers, shape_tyx=shape, policy={}, write_slab=lambda *args: None,
                                           geometry_context=context, memory_mib=memory_mib)
                    except ValueError as exc:
                        self.assertEqual(memory_mib, 5, 'ample budget was rejected')
                        self.assertRegex(str(exc), 'MiB|memory|budget')
                        continue  # Rejecting an insufficient exact workspace is appropriate.
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLessEqual(report['planned_working_bytes'], budget)
                # Small fixed Python allocation allowance; NumPy payloads dominate this fixture.
                self.assertLessEqual(peak, budget + 128 * 1024)
                self.assertLessEqual(peak, report['planned_working_bytes'] + 128 * 1024)


if __name__ == '__main__':
    unittest.main()
