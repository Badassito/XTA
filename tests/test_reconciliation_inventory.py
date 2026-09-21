"""Authenticate the reconciliation release without rewriting earlier reviews."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import unittest
from unittest import mock

from tools import verify_package_inventory as inventory


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ReconciliationInventoryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
        if 'v22_3_release_review' not in self.manifest:
            self.skipTest('Reconciliation release source review is awaiting final qualification')

    def test_complete_committed_predecessor_and_reconciliation_tools_are_authenticated(self):
        review = inventory.reviewed_v22_3_release_contract(self.manifest, self.manifest['v21_review'])
        prior = {key: value for key, value in self.manifest.items() if key != 'v22_3_release_review'}
        self.assertEqual(canonical(prior), inventory.REVIEWED_V22_3_RELEASE_PREDECESSOR_SHA256)
        self.assertEqual(review['predecessor_commit'], 'd336a66d75d7811a2c07d007ed40c46a2cfcfcb2')
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_2_RELEASE_SHA256)
        self.assertEqual(prior['v22_2_release_review']['release'], '22.2.0')
        self.assertEqual(review['release'], '22.3.0')
        self.assertEqual({item['module'] for item in review['module_snapshots']},
                         set(inventory.REVIEWED_V22_3_RELEASE_PREDECESSOR_MODULES))
        for module in ('reconciliation', 'reconciliation_policy', 'reconciliation_components',
                       'reconciliation_geometry', 'reconciliation_io', 'reconciliation_runtime',
                       'confidence_evidence', 'confidence_projection', 'confidence_tiles'):
            self.assertIn(module, review['complete_modules'])
        inventory.verify_v22_3_validation_tools(review)

    def test_every_predecessor_record_remains_immutable(self):
        for key in self.manifest.keys() - {'v22_3_release_review'}:
            manifest = copy.deepcopy(self.manifest)
            if isinstance(manifest[key], dict):
                manifest[key]['rewritten_history'] = True
            elif isinstance(manifest[key], list):
                manifest[key][0]['line'] = -1
            else:
                manifest[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'v22.3.0 predecessor inventory changed'):
                inventory.reviewed_v22_3_release_contract(manifest, manifest['v21_review'])

    def test_historical_release_accepts_only_the_authenticated_successor(self):
        prior = {key: value for key, value in self.manifest.items() if key != 'v22_3_release_review'}
        self.assertEqual(inventory.reviewed_v22_2_release_contract(self.manifest, self.manifest['v21_review']),
                         inventory.reviewed_v22_2_release_contract(prior, prior['v21_review']))
        self.manifest['v22_3_release_review']['definitions'][0]['reason'] = 'unreviewed replacement'
        with self.assertRaisesRegex(RuntimeError, 'v22.3.0 review digest mismatch'):
            inventory.reviewed_v22_2_release_contract(self.manifest, self.manifest['v21_review'])

    def test_reauthenticated_manifest_cannot_rewrite_independent_source_anchors(self):
        review = self.manifest['v22_3_release_review']
        snapshot = next(item for item in review['module_snapshots'] if item['previous_top_level'])
        snapshot['previous_ast_sha256'] = '0' * 64
        with mock.patch.object(inventory, 'REVIEWED_V22_3_RELEASE_SHA256', canonical(review)):
            with self.assertRaisesRegex(RuntimeError, 'v22.3.0 source predecessor changed'):
                inventory.reviewed_v22_3_release_contract(self.manifest, self.manifest['v21_review'])

    def test_current_code_and_external_policies_match_complete_source_snapshots(self):
        review = self.manifest['v22_3_release_review']
        trees = {item['module']: ast.parse((inventory.PACKAGE / (item['module'] + '.py')).read_text(encoding='utf-8'))
                 for item in review['module_snapshots']}
        inventory.verify_v22_3_source_snapshots(review, trees)
        for module in ('reconciliation', 'confidence_evidence', 'cuda_d1', 'pipeline',
                       'examples/external_reconciliation/baseline'):
            modified = dict(trees)
            modified[module] = copy.deepcopy(trees[module])
            modified[module].body.append(ast.Pass())
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, 'v22.3.0 reviewed source changed'):
                inventory.verify_v22_3_source_snapshots(review, modified)

    def test_comparison_and_qualification_tool_sources_are_pinned(self):
        review = self.manifest['v22_3_release_review']
        for index in range(len(review['validation_tools'])):
            changed = copy.deepcopy(review)
            changed['validation_tools'][index]['sha256'] = '0' * 64
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, 'v22.3.0 validation tool changed'):
                inventory.verify_v22_3_validation_tools(changed)


if __name__ == '__main__':
    unittest.main()
