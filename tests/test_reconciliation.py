"""Evidence invariants for external reconciliation, independent of a frontend."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from XTA.reconciliation import EvidenceLayer, reconcile
from XTA.reconciliation_policy import resolve_reconciliation, load_reconciliation_policy


def layer(name, array, *, view=None, kind='yolo', role='additive_component', scores=None, known=None):
    metadata = dict(source='fullframe', mask_kind=kind, physical_view_name=view or name,
                    view_name=view or name, view_family='orthogonal', layer_role=role, recomposition_op='union')
    confidence = None if scores is None else lambda a, b: (scores[a:b].copy(), known[a:b].copy())
    return EvidenceLayer(name, array.shape, metadata, lambda a, b: array[a:b].copy(), confidence)


def execute(layers, policy=None, memory_mib=8):
    shape = layers[0].shape_tyx if layers else (2, 7, 9)
    result = np.zeros(shape, np.uint8)
    writes = []
    def write(a, b, value):
        writes.append((a, b))
        result[a:b] = value
    report = reconcile(layers, shape_tyx=shape, policy=policy or dict(grouping='views'),
                       write_slab=write, memory_mib=memory_mib)
    return result, report, writes


class ReconciliationTests(unittest.TestCase):
    def test_union_matches_inputs_and_excludes_checkpoint(self):
        a = np.zeros((3, 7, 9), np.uint8)
        b = a.copy()
        a[:, 2:5, 2:4] = 1
        b[:, 3:6, 4:6] = 1
        checkpoint = np.ones_like(a)
        out, report, _ = execute([layer('a', a), layer('b', b), layer('final', checkpoint, role='checkpoint')],
                                 dict(mode='union', grouping='views'))
        np.testing.assert_array_equal(out, a | b)
        self.assertEqual(report['layer_count'], 2)

    def test_empty_evidence_is_empty(self):
        output, report, _ = execute([], dict(mode='union', grouping='views'))
        self.assertFalse(output.any())
        self.assertEqual(report['counts']['retained_voxels'], 0)

    def test_file_multiplicity_does_not_add_view_votes(self):
        mask = np.ones((2, 7, 9), np.uint8)
        duplicates = [layer(str(i), mask, view='transverse') for i in range(15)]
        output, report, _ = execute(duplicates)
        self.assertFalse(output.any())
        self.assertEqual(report['group_count'], 1)
        output, _, _ = execute([*duplicates, layer('independent', mask, view='sagittal')])
        np.testing.assert_array_equal(output, mask)

    def test_provenance_and_prediction_requirement(self):
        mask = np.ones((1, 7, 9), np.uint8)
        predictions = [layer('a', mask), layer('b', mask)]
        bridges = [layer('a', mask, kind='bridge'), layer('b', mask, kind='bridge')]
        self.assertTrue(execute(predictions)[0].all())
        self.assertFalse(execute(bridges)[0].any())
        self.assertFalse(execute([predictions[0], bridges[1]])[0].any())

    def test_confidence_is_actual_score_and_unknown_is_not_fabricated(self):
        mask = np.ones((1, 2, 4), np.uint8)
        high = np.full_like(mask, 230)
        low = np.full_like(mask, 50)
        known = np.ones_like(mask, bool)
        known[:, :, -1] = False
        policy = dict(mode='confidence', grouping='views', threshold=1., min_sources=2, min_prediction_sources=2)
        output, report, _ = execute([layer('a', mask, scores=high, known=known),
                                     layer('b', mask, scores=low, known=known)], policy)
        self.assertTrue(output[:, :, :-1].all())
        self.assertFalse(output[:, :, -1].any())
        self.assertEqual(report['counts']['confidence_unknown_prediction_voxels'], 4)
        with self.assertRaisesRegex(ValueError, 'Confidence evidence is unavailable'):
            execute([layer('missing', mask)], policy)

    def test_anchor_needs_independent_prediction_and_is_local(self):
        mask = np.ones((1, 2, 4), np.uint8)
        known = np.ones_like(mask, bool)
        high = np.full_like(mask, 210)
        high[:, :, 2:] = 50
        low = np.full_like(mask, 70)
        policy = dict(mode='confidence', grouping='views', threshold=1.2, anchor_bonus=.2,
                      anchor_confidence=.8, min_prediction_sources=2)
        output, _, _ = execute([layer('a', mask, scores=high, known=known),
                                layer('b', mask, scores=low, known=known)], policy)
        self.assertTrue(output[:, :, :2].all())
        self.assertFalse(output[:, :, 2:].any())
        output, _, _ = execute([layer('a', mask, view='same', scores=high, known=known),
                                layer('b', mask, view='same', scores=low, known=known)], policy)
        self.assertFalse(output.any())

    def test_editing_a_source_mask_removes_its_confidence_anchor(self):
        mask = np.ones((1, 2, 4), np.uint8)
        edited = mask.copy()
        edited[:, :, 2:] = 0
        known = np.ones_like(mask, bool)
        policy = dict(mode='confidence', grouping='views', threshold=1.2, anchor_bonus=.2,
                      anchor_confidence=.8, min_prediction_sources=2)
        output, _, _ = execute([
            layer('anchor', edited, scores=np.full_like(mask, 230), known=known),
            layer('weak1', mask, scores=np.full_like(mask, 128), known=known),
            layer('weak2', mask, scores=np.full_like(mask, 128), known=known),
        ], policy)
        self.assertTrue(output[:, :, :2].all())
        self.assertFalse(output[:, :, 2:].any())

    def test_order_and_slab_size_do_not_change_results(self):
        rng = np.random.default_rng(48)
        inputs = [layer(str(i), (rng.random((31, 32, 40)) > .65).astype(np.uint8)) for i in range(5)]
        before = [item.read_slab(0, 31) for item in inputs]
        a = execute(inputs, memory_mib=.25)[0]
        b = execute(list(reversed(inputs)), memory_mib=8)[0]
        np.testing.assert_array_equal(a, b)
        for item, original in zip(inputs, before):
            np.testing.assert_array_equal(item.read_slab(0, 31), original)

    def test_island_weights_are_bounded_and_reported(self):
        a = np.zeros((4, 8, 10), np.uint8)
        a[:, 2:5, 2:5] = 1
        b, c = a.copy(), a.copy()
        b[:, 2:6, 2:8] = 1
        c[:] = 1
        _, report, _ = execute([layer('a', a), layer('b', b), layer('c', c)],
                               dict(grouping='views', island_weighting=True))
        self.assertEqual(len(report['components']), 3)
        self.assertTrue(all(.75 <= record['weight'] <= 1.25 for record in report['components'].values()))

    def test_duplicate_ids_and_nonbinary_inputs_are_rejected(self):
        value = np.ones((1, 2, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            execute([layer('same', value), layer('same', value)])
        with self.assertRaisesRegex(ValueError, 'binary'):
            execute([layer('a', value * 2)])

    def test_empty_groups_do_not_change_island_weights_or_output(self):
        shape = (1, 8, 8)
        inputs = []
        for index, count in enumerate((4, 16, 64)):
            mask = np.zeros(shape, np.uint8)
            mask.ravel()[:count] = 1
            inputs.append(layer(f'view{index}', mask))
        policy = dict(grouping='views', island_weighting=True, threshold=2.)
        first, first_report, _ = execute(inputs, policy)
        empty = [layer(f'empty{i}', np.zeros(shape, np.uint8)) for i in range(20)]
        second, second_report, _ = execute([*inputs, *empty], policy)
        np.testing.assert_array_equal(first, second)
        for item in inputs:
            self.assertEqual(first_report['components'][item.layer_id]['weight'],
                             second_report['components'][item.layer_id]['weight'])

    def test_custom_decision_cannot_add_foreground(self):
        value = np.zeros((1, 2, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, 'outside candidate'):
            execute([layer('a', value)], dict(grouping='views', decide=lambda block: np.ones_like(block['candidate'])))

    def test_policy_files_load_and_hash_guard(self):
        root = Path(__file__).resolve().parents[1] / 'XTA/examples/external_reconciliation'
        for path in root.glob('*.py'):
            if path.name.startswith('_'):
                continue
            settings = resolve_reconciliation(SimpleNamespace(reconciliation=path, reconciliation_memory_mib=8))
            policy = load_reconciliation_policy(settings)
            self.assertEqual(policy['name'], path.stem)
            settings.assert_unchanged()


if __name__ == '__main__':
    unittest.main()
