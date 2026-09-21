from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools import verify_package_inventory as inventory
from tools.verify_package_inventory import (
    INTENTIONALLY_CHANGED_BINDINGS,
    LOCAL_IMPORT_SEAM_MARKER,
    MANIFEST,
    REVIEWED_V20_ADDED_DEFINITIONS,
    REVIEWED_V20_ADDED_STATEMENTS,
    azimuthal_rename_replacements,
    digest,
    main as verify_inventory,
    reviewed_local_import_seams,
    stable_ast_dump,
)


def _release_22_2_inventory():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    manifest.pop('v22_3_1_release_review', None)
    manifest.pop('v22_3_release_review', None)
    return manifest


def _reconciliation_successors():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    return tuple(manifest[key] for key in ('v22_3_release_review','v22_3_1_release_review') if key in manifest)


def _release_22_1_inventory():
    manifest = _release_22_2_inventory()
    manifest.pop('v22_2_release_review', None)
    return manifest


def _pta_inventory():
    # PTA contract tests retain the exact pre-promotion inventory snapshot.
    manifest = _release_22_1_inventory()
    manifest.pop('v22_1_release_review', None)
    return manifest


def _historical_inventory():
    # Existing contract tests exercise their original authenticated snapshot.
    # The PTA successor and its admission by older contracts are tested below.
    manifest = _pta_inventory()
    manifest.pop('v22_pta_throughput_review', None)
    return manifest


def inspect_seams(source: str):
    source = textwrap.dedent(source)
    return reviewed_local_import_seams("sample", source, ast.parse(source))


def augmentation_contract(manifest):
    return inventory.reviewed_v22_augmentation_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
    )


def coverage_contract(manifest):
    return inventory.reviewed_v22_coverage_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'], manifest['v22_augmentation_review'],
    )


def release_contract(manifest):
    return inventory.reviewed_v22_release_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'],
    )


def policy_memory_contract(manifest):
    return inventory.reviewed_v22_policy_memory_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'], manifest['v22_release_review'],
    )


def policy_throughput_contract(manifest):
    return inventory.reviewed_v22_policy_throughput_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'], manifest['v22_release_review'],
        manifest['v22_policy_memory_review'],
    )


def radial_retirement_contract(manifest):
    return inventory.reviewed_v22_radial_retirement_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'], manifest['v22_release_review'],
        manifest['v22_policy_memory_review'], manifest['v22_policy_throughput_review'],
    )


def policy_window_contract(manifest):
    return inventory.reviewed_v22_policy_window_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'], manifest['v22_release_review'],
        manifest['v22_policy_memory_review'], manifest['v22_policy_throughput_review'],
        manifest['v22_radial_retirement_review'],
    )


def tilted_azimuthal_gpu_contract(manifest):
    return inventory.reviewed_v22_tilted_azimuthal_gpu_contract(
        manifest, manifest['v21_review'],
        *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
        manifest['v21_1_review'], manifest['v21_1_1_review'],
        manifest['v21_1_2_review'], manifest['v22_augmentation_review'],
        manifest['v22_coverage_review'], manifest['v22_release_review'],
        manifest['v22_policy_memory_review'], manifest['v22_policy_throughput_review'],
        manifest['v22_radial_retirement_review'], manifest['v22_policy_window_review'],
    )


class PackageInventoryTests(unittest.TestCase):
    def test_v22_2_release_authenticates_the_complete_released_predecessor(self):
        manifest = _release_22_2_inventory()
        review = inventory.reviewed_v22_2_release_contract(manifest, manifest['v21_review'])
        prior = _release_22_1_inventory()
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_2_RELEASE_PREDECESSOR_SHA256)
        self.assertEqual(review['release'], '22.2.0')
        self.assertEqual(review['predecessor_commit'], 'd99f15a4f9dcedf008324f510227518afdfabb98')
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_1_RELEASE_SHA256)
        self.assertEqual({item['module'] for item in review['module_snapshots']},
                         set(inventory.REVIEWED_V22_2_RELEASE_PREDECESSOR_MODULES))
        for module in ('pta_binary', 'tta_augmentation_cpu', 'tta_augmentation_cpu_runtime'):
            self.assertIn(module, review['complete_modules'])
        for backend in ('CPU', 'GPU'):
            for profile in ('light', 'baseline', 'heavy', 'superheavy'):
                self.assertIn(f'examples/external_augmentations/{backend}_{profile}', review['complete_modules'])

    def test_v22_2_release_rejects_rewritten_history_and_unauthenticated_additions(self):
        baseline = _release_22_2_inventory()
        for key in baseline.keys() - {'v22_2_release_review'}:
            manifest = copy.deepcopy(baseline)
            if isinstance(manifest[key], dict):
                manifest[key]['rewritten_history'] = True
            elif isinstance(manifest[key], list):
                manifest[key][0]['line'] = -1
            else:
                manifest[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'v22.2.0 predecessor inventory changed'):
                inventory.reviewed_v22_2_release_contract(manifest, manifest['v21_review'])
        baseline['v22_2_release_review']['definitions'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v22.2.0 review digest mismatch'):
            inventory.reviewed_v22_2_release_contract(baseline, baseline['v21_review'])

    def test_historical_contracts_admit_only_the_authenticated_v22_2_successor(self):
        manifest = _release_22_2_inventory()
        prior = _release_22_1_inventory()
        contracts = (release_contract, policy_memory_contract, policy_throughput_contract,
                     radial_retirement_contract, policy_window_contract, tilted_azimuthal_gpu_contract,
                     lambda value: inventory.reviewed_v22_pta_throughput_contract(value, value['v21_review']),
                     lambda value: inventory.reviewed_v22_1_release_contract(value, value['v21_review']))
        for contract in contracts:
            self.assertEqual(contract(manifest), contract(prior))
        manifest['v22_2_release_review']['statements'][0]['reason'] = 'unreviewed replacement'
        for contract in contracts:
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'review digest mismatch'):
                contract(manifest)

    def test_v22_2_release_checks_independent_source_anchors_and_statement_positions(self):
        for mutation, expected in (
            ('commit', 'unexpected predecessor or source scope'),
            ('source_anchor', 'source predecessor changed'),
            ('historical_statements', 'source predecessor changed'),
            ('position', 'statement position differs'),
            ('previous_position', 'statement predecessor changed'),
        ):
            manifest = _release_22_2_inventory()
            review = manifest['v22_2_release_review']
            snapshot = next(item for item in review['module_snapshots'] if item['previous_top_level'])
            record = next(item for item in review['definitions'] + review['statements'] if item['previous_index'] is not None)
            if mutation == 'commit':
                review['predecessor_commit'] = '0' * 40
            elif mutation == 'source_anchor':
                snapshot['previous_ast_sha256'] = '0' * 64
            elif mutation == 'historical_statements':
                snapshot['previous_top_level'][0] = '0' * 64
            elif mutation == 'position':
                record['current_index'] = -1
            else:
                record['previous_index'] = -1
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(mutation=mutation), mock.patch.object(inventory, 'REVIEWED_V22_2_RELEASE_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, expected):
                    inventory.reviewed_v22_2_release_contract(manifest, manifest['v21_review'])

    def test_v22_2_full_source_snapshots_reject_unreviewed_code_and_policy_changes(self):
        manifest = _release_22_2_inventory()
        review = inventory.reviewed_v22_2_release_contract(manifest, manifest['v21_review'])
        trees = {item['module']: ast.parse((inventory.PACKAGE / (item['module'] + '.py')).read_text(encoding='utf-8'))
                 for item in review['module_snapshots']}
        inventory.verify_v22_2_source_snapshots(review, trees, _reconciliation_successors())
        for module in ('pipeline', 'pta_binary', 'tta_augmentation_cpu', 'examples/external_augmentations/CPU_baseline'):
            changed = dict(trees)
            changed[module] = copy.deepcopy(trees[module])
            changed[module].body.append(ast.Assign(targets=[ast.Name(id='UNREVIEWED_BINDING', ctx=ast.Store())],
                                                   value=ast.Constant(1)))
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, 'v22.2.0 reviewed source changed: ' + module):
                inventory.verify_v22_2_source_snapshots(review, changed, _reconciliation_successors())

    def test_v22_1_release_authenticates_exact_pta_candidate_bundle(self):
        manifest = _release_22_1_inventory()
        review = inventory.reviewed_v22_1_release_contract(manifest, manifest['v21_review'])
        prior = {key: value for key, value in manifest.items() if key != 'v22_1_release_review'}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_1_RELEASE_PREDECESSOR_SHA256)
        self.assertEqual(review['release'], '22.1.0')
        self.assertEqual(review['previous_review_sha256'],
                         'a9f548390c5d6881bb244cf0ee7f4333323b7e1c923079eb682f4248d779c46e')
        self.assertEqual(review['predecessor_bundle'], {
            'name': 'XTA_v22.0.0_complete_source.zip',
            'sha256': '5c3cbcdf7310293e92c8bd7386959d0be9d529acfb9d726a77896c7f520055ef',
        })
        self.assertEqual(len(review['statements']), 5)
        self.assertFalse(review['definitions'])
        self.assertNotIn('predecessor_commit', review)

    def test_v22_1_release_preserves_all_historical_records(self):
        baseline = _release_22_1_inventory()
        for key in baseline.keys() - {'v22_1_release_review'}:
            manifest = copy.deepcopy(baseline)
            if isinstance(manifest[key], dict):
                manifest[key]['rewritten_history'] = True
            elif isinstance(manifest[key], list):
                manifest[key][0]['line'] = -1
            else:
                manifest[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'release predecessor inventory changed'):
                inventory.reviewed_v22_1_release_contract(manifest, manifest['v21_review'])

    def test_historical_contracts_accept_only_authenticated_v22_1_release(self):
        manifest = _release_22_1_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_1_release_review']
        contracts = (release_contract, policy_memory_contract, policy_throughput_contract,
                     radial_retirement_contract, policy_window_contract, tilted_azimuthal_gpu_contract,
                     lambda value: inventory.reviewed_v22_pta_throughput_contract(value, value['v21_review']))
        for contract in contracts:
            self.assertEqual(contract(prior), contract(manifest))
        manifest['v22_1_release_review']['statements'][0]['sha256'] = '0' * 64
        for contract in contracts:
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'review digest mismatch'):
                contract(manifest)

    def test_v22_1_release_rejects_wrong_predecessor_binding_or_functional_scope(self):
        for mutation, expected in (
                ('bundle', 'unexpected bundle predecessor'),
                ('previous_hash', 'supersession does not match its historical pin'),
                ('binding', 'exactly its five release identity bindings'),
                ('module_scope', 'unexpected bundle predecessor')):
            manifest = _release_22_1_inventory()
            review = manifest['v22_1_release_review']
            if mutation == 'bundle':
                review['predecessor_bundle']['sha256'] = '0' * 64
            elif mutation == 'previous_hash':
                review['statements'][0]['previous_sha256'] = '0' * 64
            elif mutation == 'binding':
                review['statements'][0]['binding'] = 'UNREVIEWED_VERSION'
            else:
                review['preserved_module_statements_sha256']['cli'] = '0' * 64
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(mutation=mutation), mock.patch.object(inventory, 'REVIEWED_V22_1_RELEASE_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, expected):
                    inventory.reviewed_v22_1_release_contract(manifest, manifest['v21_review'])

    def test_v22_1_release_keeps_nonversion_runtime_statements_unchanged(self):
        manifest = _release_22_1_inventory()
        review = inventory.reviewed_v22_1_release_contract(manifest, manifest['v21_review'])
        successor = json.loads(MANIFEST.read_text(encoding='utf-8'))['v22_2_release_review']
        top_level = {module: ast.parse((inventory.PACKAGE / f'{module}.py').read_text()).body
                     for module in ('__init__', 'cli', 'config')}
        inventory.verify_v22_1_release_scope(review, top_level, (successor, *_reconciliation_successors()))
        for module in top_level:
            changed = copy.deepcopy(top_level)
            changed[module].append(ast.Assign(targets=[ast.Name(id='UNREVIEWED_RELEASE_BEHAVIOR', ctx=ast.Store())],
                                              value=ast.Constant(1)))
            with self.subTest(module=module), self.assertRaisesRegex(RuntimeError, 'changed non-version module statements: ' + module):
                inventory.verify_v22_1_release_scope(review, changed, (successor, *_reconciliation_successors()))

    def test_pta_successor_authenticates_complete_distributed_inventory_and_sources(self):
        manifest = _pta_inventory()
        review = inventory.reviewed_v22_pta_throughput_contract(manifest, manifest['v21_review'])
        prior = {key: value for key, value in manifest.items() if key != 'v22_pta_throughput_review'}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_SHA256)
        self.assertEqual(review['predecessor_bundle'], {
            'name': 'XTA_v22.0.0_complete_source.zip',
            'sha256': '44c5c5be060ade84e03e2660b1a1ef1e270bd474f68f32b7f81ed787d0db1b16',
        })
        self.assertEqual(review['previous_review_sha256'],
                         'efb3bb2ae63be95d3faac73b6b28b28279aa8fb495f08d0cb4c8d6b1dd89ff95')
        self.assertNotIn('predecessor_commit', review)
        self.assertEqual(set(review['complete_modules']), {'pta_batch_pipeline', 'pta_gpu_publication'})
        self.assertEqual({item['module']: item['previous_ast_sha256'] for item in review['module_snapshots']},
                         inventory.REVIEWED_V22_PTA_PREDECESSOR_MODULES)

    def test_pta_successor_rejects_changes_to_every_historical_record(self):
        baseline = _pta_inventory()
        for key in baseline.keys() - {'v22_pta_throughput_review'}:
            manifest = copy.deepcopy(baseline)
            value = manifest[key]
            if isinstance(value, dict):
                value['rewritten_history'] = True
            elif isinstance(value, list):
                value[0]['line'] = -1
            else:
                manifest[key] = int(value) + 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'PTA throughput predecessor inventory changed'):
                inventory.reviewed_v22_pta_throughput_contract(manifest, manifest['v21_review'])

    def test_historical_contracts_allow_only_the_authenticated_pta_successor(self):
        manifest = _pta_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_pta_throughput_review']
        contracts = (release_contract, policy_memory_contract, policy_throughput_contract,
                     radial_retirement_contract, policy_window_contract, tilted_azimuthal_gpu_contract)
        for contract in contracts:
            self.assertEqual(contract(prior), contract(manifest))
        manifest['v22_pta_throughput_review']['definitions'][0]['sha256'] = '0' * 64
        for contract in contracts:
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'review digest mismatch'):
                contract(manifest)

    def test_pta_successor_requires_exact_bundle_scope_and_original_module_anchors(self):
        for mutation, expected in (
                ('bundle', 'unexpected bundle predecessor'),
                ('definition', 'exactly its publication, CPU budget and format contracts'),
                ('complete_module', 'exactly its publication, CPU budget and format contracts'),
                ('module_scope', 'source snapshot coverage differs'),
                ('module_predecessor', 'source predecessor changed')):
            manifest = _pta_inventory()
            review = manifest['v22_pta_throughput_review']
            if mutation == 'bundle':
                review['predecessor_bundle']['sha256'] = '0' * 64
            elif mutation == 'definition':
                review['definitions'].pop()
            elif mutation == 'complete_module':
                review['complete_modules'].pop()
            elif mutation == 'module_scope':
                review['module_snapshots'].pop()
            else:
                review['module_snapshots'][0]['previous_ast_sha256'] = '0' * 64
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(mutation=mutation), mock.patch.object(inventory, 'REVIEWED_V22_PTA_THROUGHPUT_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, expected):
                    inventory.reviewed_v22_pta_throughput_contract(manifest, manifest['v21_review'])

    def test_pta_parser_supersedes_latest_augmentation_review_not_older_pin(self):
        manifest = _pta_inventory()
        review = manifest['v22_pta_throughput_review']
        record = next(item for item in review['definitions']
                      if (item['module'], item['name']) == ('pta_config', 'build_pta_argparser'))
        prior = next(item for item in manifest['v22_augmentation_review']['definitions']
                     if (item['module'], item['name']) == ('pta_config', 'build_pta_argparser'))
        self.assertEqual(record['previous_sha256'], prior['sha256'])
        record['previous_sha256'] = prior['previous_sha256']
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_PTA_THROUGHPUT_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'supersession does not match its historical pin: pta_config.build_pta_argparser'):
                inventory.reviewed_v22_pta_throughput_contract(manifest, manifest['v21_review'])

    def test_pta_new_modules_require_complete_statement_coverage(self):
        original_parse = inventory.ast.parse
        for module in ('pta_batch_pipeline', 'pta_gpu_publication'):
            def add_statement(source, filename='<unknown>', *args, **kwargs):
                tree = original_parse(source, filename, *args, **kwargs)
                if str(filename).replace('\\', '/').endswith('/' + module + '.py'):
                    tree.body.append(ast.Assign(targets=[ast.Name(id='UNREVIEWED_PTA_BINDING', ctx=ast.Store())],
                                                value=ast.Constant(1)))
                return tree
            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=add_statement):
                with self.assertRaisesRegex(RuntimeError, 'complete-module statement coverage differs: ' + module):
                    verify_inventory()

    def test_pta_existing_modules_pin_unmodified_definitions_and_top_level_bindings(self):
        original_parse = inventory.ast.parse
        for module in ('pta', 'pta_scheduler', 'pta_config', 'pta_runtime',
                       'pta_publication', 'pta_workers', 'nvtiff_backend'):
            def add_statement(source, filename='<unknown>', *args, **kwargs):
                tree = original_parse(source, filename, *args, **kwargs)
                if str(filename).replace('\\', '/').endswith('/' + module + '.py'):
                    tree.body.append(ast.Assign(targets=[ast.Name(id='UNREVIEWED_EXISTING_PTA_BINDING', ctx=ast.Store())],
                                                value=ast.Constant(1)))
                return tree
            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=add_statement):
                expected = ('complete-module statement coverage differs: ' if module == 'pta_runtime'
                            else 'PTA throughput complete source changed: ') + module
                with self.assertRaisesRegex(RuntimeError, expected):
                    verify_inventory()

    def test_tilted_gpu_review_authenticates_complete_committed_predecessor(self):
        manifest = _historical_inventory()
        review = tilted_azimuthal_gpu_contract(manifest)
        prior = {key: value for key, value in manifest.items() if key != 'v22_tilted_azimuthal_gpu_review'}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_SHA256)
        self.assertEqual(review['predecessor_commit'], '6365f0c0a75d704d9453696e9070cafa22a26434')
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_POLICY_WINDOW_SHA256)
        self.assertEqual(review['feature'], 'tilted-azimuthal-gpu-projection')
        self.assertEqual(set(review['complete_modules']), {'tilted_azimuthal_projection', 'tilted_azimuthal_projection_cuda'})
        self.assertEqual(len(review['definitions']), 19)

    def test_tilted_gpu_review_preserves_all_earlier_records(self):
        baseline = _historical_inventory()
        for key in baseline.keys() - {'v22_tilted_azimuthal_gpu_review'}:
            manifest = copy.deepcopy(baseline)
            value = manifest[key]
            if isinstance(value, dict):
                value['rewritten_history'] = True
            elif isinstance(value, list):
                value[0]['line'] = -1
            else:
                manifest[key] = int(value) + 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'Tilted Azimuthal GPU predecessor inventory changed'):
                tilted_azimuthal_gpu_contract(manifest)

    def test_historical_contracts_accept_only_authenticated_tilted_gpu_successor(self):
        manifest = _historical_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_tilted_azimuthal_gpu_review']
        contracts = (release_contract, policy_memory_contract, policy_throughput_contract,
                     radial_retirement_contract, policy_window_contract)
        for contract in contracts:
            self.assertEqual(contract(prior), contract(manifest))
        manifest['v22_tilted_azimuthal_gpu_review']['definitions'][0]['sha256'] = '0' * 64
        for contract in contracts:
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'review digest mismatch'):
                contract(manifest)
        manifest = _historical_inventory()
        del manifest['v22_policy_window_review']
        with self.assertRaisesRegex(RuntimeError, 'requires the reviewed policy window predecessor'):
            radial_retirement_contract(manifest)

    def test_tilted_gpu_requires_exact_identity_and_full_source_scope(self):
        for mutation in ('commit', 'definition', 'complete_module'):
            manifest = _historical_inventory()
            review = manifest['v22_tilted_azimuthal_gpu_review']
            if mutation == 'commit':
                review['predecessor_commit'] = '0' * 40
                message = 'unexpected predecessor or feature'
            elif mutation == 'definition':
                review['definitions'].pop()
                message = 'exactly its operator, routing and provenance contracts'
            else:
                review['complete_modules'].pop()
                message = 'exactly its operator, routing and provenance contracts'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(mutation=mutation), mock.patch.object(inventory, 'REVIEWED_V22_TILTED_AZIMUTHAL_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    tilted_azimuthal_gpu_contract(manifest)

    def test_tilted_gpu_coordinator_requires_latest_radial_retirement_pin(self):
        manifest = _historical_inventory()
        review = manifest['v22_tilted_azimuthal_gpu_review']
        record = next(item for item in review['definitions']
                      if (item['module'], item['name']) == ('backprojection', '_MainProcessGpuStageCoordinator'))
        prior = next(item for item in manifest['v22_radial_retirement_review']['definitions']
                     if (item['module'], item['name']) == ('backprojection', '_MainProcessGpuStageCoordinator'))
        self.assertEqual(record['previous_sha256'], prior['sha256'])
        record['previous_sha256'] = prior['previous_sha256']
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_TILTED_AZIMUTHAL_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'supersession does not match its historical pin: backprojection._MainProcessGpuStageCoordinator'):
                tilted_azimuthal_gpu_contract(manifest)

    def test_tilted_gpu_modules_have_complete_statement_coverage(self):
        original_parse = inventory.ast.parse
        for module in ('tilted_azimuthal_projection', 'tilted_azimuthal_projection_cuda'):
            def add_statement(source, filename='<unknown>', *args, **kwargs):
                tree = original_parse(source, filename, *args, **kwargs)
                if str(filename).replace('\\', '/').endswith('/' + module + '.py'):
                    tree.body.append(ast.Assign(targets=[ast.Name(id='UNREVIEWED_TILTED_BINDING', ctx=ast.Store())], value=ast.Constant(1)))
                return tree
            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=add_statement):
                with self.assertRaisesRegex(RuntimeError, 'complete-module statement coverage differs: ' + module):
                    verify_inventory()

    def test_tilted_gpu_kernel_has_an_independent_statement_pin(self):
        original_parse = inventory.ast.parse
        def change_kernel(source, filename='<unknown>', *args, **kwargs):
            tree = original_parse(source, filename, *args, **kwargs)
            if str(filename).replace('\\', '/').endswith('/tilted_azimuthal_projection_cuda.py'):
                node = next(node for node in tree.body if isinstance(node, ast.Assign)
                            and any(getattr(target, 'id', '') == '_KERNEL_SOURCE' for target in node.targets))
                node.value = ast.Constant(node.value.value + '\n// unreviewed kernel mutation\n')
            return tree
        with mock.patch.object(inventory.ast, 'parse', side_effect=change_kernel):
            with self.assertRaisesRegex(RuntimeError, 'reviewed statement changed or is missing: tilted_azimuthal_projection_cuda'):
                verify_inventory()

    def test_policy_window_scope_reconstructs_only_approved_provenance_change(self):
        manifest = _historical_inventory()
        review = tilted_azimuthal_gpu_contract(manifest)
        statements = ast.parse((inventory.PACKAGE / 'pipeline.py').read_text(encoding='utf-8')).body
        window = manifest['v22_policy_window_review']
        successor = json.loads(MANIFEST.read_text(encoding='utf-8'))['v22_2_release_review']
        inventory.verify_policy_window_runtime_scope(window, statements, (review, successor, *_reconciliation_successors()))
        with self.assertRaisesRegex(RuntimeError, 'policy window changed unreviewed pipeline statements or imports'):
            inventory.verify_policy_window_runtime_scope(window, statements)
        altered = copy.deepcopy(statements)
        node = next(node for node in altered if getattr(node, 'name', None) == '_execution_runtime_provenance')
        node.body.append(ast.Pass())
        with self.assertRaisesRegex(RuntimeError, 'policy window changed unreviewed pipeline statements or imports'):
            inventory.verify_policy_window_runtime_scope(window, altered, (review, successor, *_reconciliation_successors()))

    def test_policy_window_authenticates_complete_distributed_predecessor(self):
        manifest = _historical_inventory()
        review = policy_window_contract(manifest)
        prior = {key: value for key, value in manifest.items() if key not in ('v22_policy_window_review', 'v22_tilted_azimuthal_gpu_review')}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, '39ed77ed4f861333ef9252ab559c99eeb5b519944a32d80fc7f223c2703fd94a')
        self.assertEqual(authenticated, inventory.REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_SHA256)
        self.assertEqual(review['predecessor_bundle'], {
            'name': 'XTA_v22.0.0_complete_source.zip',
            'sha256': '9443efd0468c59d3b39b232339a680556eb40d02ff78566480cce6fd1e840c23',
        })
        self.assertNotIn('predecessor_commit', review)
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_RADIAL_RETIREMENT_SHA256)
        self.assertEqual([(item['module'],item['name']) for item in review['definitions']], [('pipeline','_main_impl')])
        self.assertFalse(review['statements'])

    def test_policy_window_preserves_every_historical_inventory_record(self):
        baseline = _historical_inventory()
        for key in baseline.keys() - {'v22_policy_window_review', 'v22_tilted_azimuthal_gpu_review'}:
            manifest = copy.deepcopy(baseline)
            value = manifest[key]
            if isinstance(value, dict):
                value['rewritten_history'] = True
            elif isinstance(value, list):
                value[0]['line'] = -1
            else:
                manifest[key] = int(value) + 1
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'policy window predecessor inventory changed'):
                policy_window_contract(manifest)

    def test_historical_reviews_accept_only_authenticated_policy_window_successor(self):
        manifest = _historical_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_policy_window_review']
        del prior['v22_tilted_azimuthal_gpu_review']
        contracts = (release_contract, policy_memory_contract, policy_throughput_contract, radial_retirement_contract)
        for contract in contracts:
            self.assertEqual(contract(prior), contract(manifest))
        manifest['v22_policy_window_review']['definitions'][0]['sha256'] = '0'*64
        for contract in contracts:
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'review digest mismatch'):
                contract(manifest)
        manifest = _historical_inventory()
        del manifest['v22_radial_retirement_review']
        with self.assertRaisesRegex(RuntimeError, 'requires the reviewed Radial retirement predecessor'):
            policy_throughput_contract(manifest)

    def test_policy_window_rejects_wrong_bundle_or_expanded_runtime_scope(self):
        for mutation in ('bundle', 'commit', 'definition', 'statement', 'complete_module'):
            manifest = _historical_inventory()
            review = manifest['v22_policy_window_review']
            if mutation == 'bundle':
                review['predecessor_bundle']['sha256'] = '0'*64
                message = 'unexpected bundle predecessor or feature'
            elif mutation == 'commit':
                review['predecessor_commit'] = '0'*40
                message = 'unexpected bundle predecessor or feature'
            else:
                message = 'must cover only pipeline._main_impl'
                if mutation == 'definition':
                    review['definitions'].append(dict(module='pipeline', name='unreviewed',
                        previous_sha256=None, sha256='1'*64, reason='unexpected runtime addition'))
                elif mutation == 'statement':
                    review['statements'].append(dict(module='pipeline', label='unreviewed_import',
                        previous_sha256=None, sha256='1'*64, reason='unexpected import'))
                else:
                    review['complete_modules'].append('pipeline')
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(mutation=mutation), mock.patch.object(inventory,'REVIEWED_V22_POLICY_WINDOW_SHA256',authenticated):
                with self.assertRaisesRegex(RuntimeError,message):
                    policy_window_contract(manifest)

    def test_policy_window_requires_latest_pipeline_predecessor_definition(self):
        manifest = _historical_inventory()
        review = manifest['v22_policy_window_review']
        previous = next(item for item in manifest['v22_policy_throughput_review']['definitions']
                        if (item['module'],item['name']) == ('pipeline','_main_impl'))
        self.assertEqual(review['definitions'][0]['previous_sha256'],previous['sha256'])
        review['definitions'][0]['previous_sha256'] = previous['previous_sha256']
        authenticated = hashlib.sha256(json.dumps(review,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory,'REVIEWED_V22_POLICY_WINDOW_SHA256',authenticated):
            with self.assertRaisesRegex(RuntimeError,'supersession does not match its historical pin: pipeline._main_impl'):
                policy_window_contract(manifest)

    def test_policy_window_current_body_and_unreviewed_imports_are_both_pinned(self):
        original_parse = inventory.ast.parse
        for target in ('body', 'import'):
            def changed_parse(source,filename='<unknown>',*args,**kwargs):
                tree = original_parse(source,filename,*args,**kwargs)
                if str(filename).replace('\\','/').endswith('/pipeline.py'):
                    if target == 'import':
                        tree.body.append(ast.Import(names=[ast.alias(name='unreviewed_import')]))
                    else:
                        node = next(node for node in tree.body if getattr(node,'name',None) == '_main_impl')
                        node.body.append(ast.Pass())
                return tree
            message = ('policy window changed unreviewed pipeline statements or imports'
                       if target == 'import' else 'reviewed (?:added )?definition changed or is missing: pipeline._main_impl')
            with self.subTest(target=target), mock.patch.object(inventory.ast,'parse',side_effect=changed_parse):
                with self.assertRaisesRegex(RuntimeError,message):
                    verify_inventory()

    def test_radial_retirement_authenticates_the_complete_throughput_bundle(self):
        manifest = _historical_inventory()
        review = radial_retirement_contract(manifest)
        prior = {key: value for key, value in manifest.items() if key not in ('v22_radial_retirement_review', 'v22_policy_window_review', 'v22_tilted_azimuthal_gpu_review')}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_SHA256)
        self.assertEqual(review['predecessor_bundle'], {
            'name': 'XTA_v22.0.0_complete_source.zip',
            'sha256': 'fede9e0d4a8a3e32786cd2a62e79b540f1e7dd0cd34630bc5e0a83f9b721cf8c',
        })
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_POLICY_THROUGHPUT_SHA256)
        self.assertNotIn('predecessor_commit', review)
        self.assertEqual(review['feature'], 'radial-gpu-retirement')
        self.assertEqual(len(review['definitions']), 5)
        self.assertEqual(len(review['statements']), 3)
        self.assertEqual([item['module'] for item in review['preserved_radial_module_updates']], ['cylindrical_projection'])

    def test_radial_retirement_preserves_all_predecessor_review_records(self):
        for key in ('v22_policy_throughput_review', 'v22_policy_memory_review', 'v22_release_review', 'v21_review'):
            manifest = _historical_inventory()
            manifest[key]['rewritten_history'] = True
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'Radial retirement predecessor inventory changed'):
                radial_retirement_contract(manifest)

    def test_historical_contracts_accept_only_authenticated_radial_successor(self):
        manifest = _historical_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_radial_retirement_review']
        del prior['v22_policy_window_review']
        del prior['v22_tilted_azimuthal_gpu_review']
        for contract in (release_contract, policy_memory_contract, policy_throughput_contract):
            with self.subTest(contract=contract.__name__):
                self.assertEqual(contract(prior), contract(manifest))
        manifest['v22_radial_retirement_review']['definitions'][0]['sha256'] = '0' * 64
        for contract in (release_contract, policy_memory_contract, policy_throughput_contract):
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'v22.0.0 review digest mismatch'):
                contract(manifest)
        manifest = _historical_inventory()
        del manifest['v22_policy_throughput_review']
        with self.assertRaisesRegex(RuntimeError, 'requires the reviewed throughput predecessor'):
            policy_memory_contract(manifest)

    def test_radial_retirement_requires_exact_bundle_and_projection_scope(self):
        for category in ('bundle', 'definitions', 'statements', 'preserved_radial_module_updates'):
            manifest = _historical_inventory()
            review = manifest['v22_radial_retirement_review']
            if category == 'bundle':
                review['predecessor_bundle']['sha256'] = '0' * 64
                message = 'unexpected bundle predecessor or feature'
            else:
                review[category].pop()
                message = 'exactly its scheduler, projection and retry contracts'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_RADIAL_RETIREMENT_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    radial_retirement_contract(manifest)

    def test_radial_retry_import_requires_the_unchanged_v20_predecessor_pin(self):
        manifest = _historical_inventory()
        review = manifest['v22_radial_retirement_review']
        record = next(item for item in review['statements'] if item['label'] == 'shared_future_import')
        key = ('cylindrical_projection', 'shared_future_import')
        self.assertEqual(record['previous_sha256'], REVIEWED_V20_ADDED_STATEMENTS[key][0])
        effective = inventory.reviewed_v20_statement_hashes((review,))
        self.assertEqual(effective[key], record['sha256'])
        record['previous_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v20 statement successor does not match its historical pin'):
            inventory.reviewed_v20_statement_hashes((review,))
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_RADIAL_RETIREMENT_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'supersession does not match its historical pin: cylindrical_projection.shared_future_import'):
                radial_retirement_contract(manifest)

    def test_radial_retry_import_and_interval_have_independent_statement_pins(self):
        original_parse = inventory.ast.parse
        for target in ('import', 'retry_interval'):
            def changed_parse(source, filename='<unknown>', *args, **kwargs):
                tree = original_parse(source, filename, *args, **kwargs)
                if str(filename).replace('\\', '/').endswith('/cylindrical_projection.py'):
                    for node in tree.body:
                        if target == 'import' and isinstance(node, ast.ImportFrom) and node.module == 'concurrent.futures':
                            node.names = [alias for alias in node.names if alias.name != 'CancelledError']
                        elif (target == 'retry_interval' and isinstance(node, ast.Assign)
                              and any(getattr(name, 'id', '') == '_CUDA_RECHECK_SECONDS' for name in node.targets)):
                            node.value = ast.Constant(value=2.0)
                return tree
            message = ('reviewed added statement changed or is missing: cylindrical_projection.shared_future_import'
                       if target == 'import' else 'reviewed statement changed or is missing: cylindrical_projection.binding__CUDA_RECHECK_SECONDS')
            with self.subTest(target=target), mock.patch.object(inventory.ast, 'parse', side_effect=changed_parse):
                with self.assertRaisesRegex(RuntimeError, message):
                    verify_inventory()

    def test_policy_throughput_authenticates_distributed_bundle_without_inventing_commit(self):
        manifest = _historical_inventory()
        review = policy_throughput_contract(manifest)
        prior = {key: value for key, value in manifest.items()
                 if key not in ('v22_policy_throughput_review', 'v22_radial_retirement_review', 'v22_policy_window_review', 'v22_tilted_azimuthal_gpu_review')}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_SHA256)
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_POLICY_MEMORY_SHA256)
        self.assertEqual(review['predecessor_bundle'], {
            'name': 'XTA_v22.0.0_complete_source.zip',
            'sha256': '687421acc062cfc7c67a60d98a5cce61e74be1ad90ed609c06fac7201b76d292',
        })
        self.assertNotIn('predecessor_commit', review)
        self.assertEqual(review['feature'], 'shared-policy-union-throughput')
        self.assertEqual(len(review['definitions']), 13)
        self.assertEqual(len(review['statements']), 1)
        self.assertEqual(len(review['local_import_seam_updates']), 1)

    def test_policy_throughput_cannot_rewrite_memory_or_earlier_inventory_records(self):
        for key in ('v22_policy_memory_review', 'v22_release_review', 'v22_coverage_review', 'v21_1_2_review'):
            manifest = _historical_inventory()
            manifest[key]['rewritten_history'] = True
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'throughput predecessor inventory changed'):
                policy_throughput_contract(manifest)

    def test_historical_contracts_accept_only_authenticated_throughput_successor(self):
        manifest = _historical_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_policy_throughput_review']
        del prior['v22_radial_retirement_review']
        del prior['v22_policy_window_review']
        del prior['v22_tilted_azimuthal_gpu_review']
        self.assertEqual(release_contract(prior), release_contract(manifest))
        self.assertEqual(policy_memory_contract(prior), policy_memory_contract(manifest))
        manifest['v22_policy_throughput_review']['definitions'][0]['sha256'] = '0' * 64
        for contract in (release_contract, policy_memory_contract):
            with self.subTest(contract=contract.__name__), self.assertRaisesRegex(RuntimeError, 'v22.0.0 review digest mismatch'):
                contract(manifest)
        manifest = _historical_inventory()
        del manifest['v22_policy_memory_review']
        with self.assertRaisesRegex(RuntimeError, 'requires the reviewed memory predecessor'):
            release_contract(manifest)

    def test_policy_throughput_requires_bundle_identity_and_exact_changed_scope(self):
        for category in ('bundle', 'commit', 'definitions', 'statements', 'local_import_seam_updates'):
            manifest = _historical_inventory()
            review = manifest['v22_policy_throughput_review']
            if category == 'bundle':
                review['predecessor_bundle']['sha256'] = '0' * 64
                message = 'unexpected bundle predecessor or feature'
            elif category == 'commit':
                review['predecessor_commit'] = '0' * 40
                message = 'unexpected bundle predecessor or feature'
            else:
                review[category].pop()
                message = 'exactly its shared-policy definitions, import and seam'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_POLICY_THROUGHPUT_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    policy_throughput_contract(manifest)

    def test_policy_throughput_requires_latest_memory_definition_and_review_pins(self):
        for category in ('definition', 'review'):
            manifest = _historical_inventory()
            review = manifest['v22_policy_throughput_review']
            if category == 'definition':
                record = next(item for item in review['definitions']
                              if (item['module'], item['name']) == ('tta_scheduler', 'TtaScheduler'))
                memory = next(item for item in manifest['v22_policy_memory_review']['definitions']
                              if (item['module'], item['name']) == ('tta_scheduler', 'TtaScheduler'))
                self.assertEqual(record['previous_sha256'], memory['sha256'])
                record['previous_sha256'] = memory['previous_sha256']
                message = 'supersession does not match its historical pin: tta_scheduler.TtaScheduler'
            else:
                review['previous_review_sha256'] = inventory.REVIEWED_V22_RELEASE_SHA256
                message = 'unexpected release or predecessor identity'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_POLICY_THROUGHPUT_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    policy_throughput_contract(manifest)

    def test_shared_policy_outputs_and_memfd_transfers_have_independent_definition_pins(self):
        original_digest = inventory.digest
        for target in ('_open_policy_sibling_outputs', '_materialize_worker_task_memfd_paths', '_DeviceUnionAccumulator'):
            def changed_digest(node):
                return '0' * 64 if getattr(node, 'name', None) == target else original_digest(node)
            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, 'reviewed (?:added )?definition changed or is missing:'):
                    verify_inventory()

    def test_policy_memory_review_authenticates_the_complete_reconciled_inventory(self):
        manifest = _historical_inventory()
        review = policy_memory_contract(manifest)
        prior = {key: value for key, value in manifest.items()
                 if key not in ('v22_policy_memory_review', 'v22_policy_throughput_review', 'v22_radial_retirement_review', 'v22_policy_window_review', 'v22_tilted_azimuthal_gpu_review')}
        authenticated = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(authenticated, inventory.REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_SHA256)
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_RELEASE_SHA256)
        self.assertEqual(review['predecessor_commit'], 'a328bc0c03b81ef4afcab9f9f360b16c38c4e6cc')
        self.assertEqual(review['feature'], 'bounded-policy-parent-memory')
        self.assertEqual(len(review['definitions']), 6)
        self.assertEqual(len(review['statements']), 2)
        self.assertEqual(release_contract(manifest), manifest['v22_release_review'])

    def test_policy_memory_review_cannot_rewrite_any_previous_record(self):
        for key in ('v21_1_2_review', 'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review'):
            manifest = _historical_inventory()
            manifest[key]['unreviewed_reason'] = 'rewritten history'
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'policy memory predecessor inventory changed'):
                policy_memory_contract(manifest)
        manifest = _historical_inventory()
        manifest['statements'][0]['line'] += 1
        with self.assertRaisesRegex(RuntimeError, 'policy memory predecessor inventory changed'):
            policy_memory_contract(manifest)

    def test_release_ignores_only_an_authenticated_memory_successor(self):
        manifest = _historical_inventory()
        prior = copy.deepcopy(manifest)
        del prior['v22_policy_memory_review']
        del prior['v22_policy_throughput_review']
        del prior['v22_radial_retirement_review']
        del prior['v22_policy_window_review']
        del prior['v22_tilted_azimuthal_gpu_review']
        self.assertEqual(release_contract(prior), release_contract(manifest))
        manifest['v22_policy_memory_review']['definitions'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v22.0.0 review digest mismatch'):
            release_contract(manifest)

    def test_policy_memory_requires_latest_definition_and_release_predecessors(self):
        for category in ('definition', 'release'):
            manifest = _historical_inventory()
            review = manifest['v22_policy_memory_review']
            if category == 'definition':
                record = next(item for item in review['definitions']
                              if (item['module'], item['name']) == ('tta_scheduler', 'TtaScheduler'))
                record['previous_sha256'] = '0' * 64
                message = 'supersession does not match its historical pin: tta_scheduler.TtaScheduler'
            else:
                review['previous_review_sha256'] = '0' * 64
                message = 'unexpected release or predecessor identity'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_POLICY_MEMORY_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    policy_memory_contract(manifest)

    def test_policy_memory_requires_exact_feature_identity_and_changed_source_scope(self):
        for category in ('feature', 'definitions', 'statements'):
            manifest = _historical_inventory()
            review = manifest['v22_policy_memory_review']
            if category == 'feature':
                review['feature'] = 'unreviewed-feature'
                message = 'unexpected predecessor or feature'
            else:
                review[category].pop()
                message = 'exactly its parent-admission definitions and imports'
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_POLICY_MEMORY_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    policy_memory_contract(manifest)

    def test_policy_memory_safety_helpers_have_independent_definition_pins(self):
        original_digest = inventory.digest
        for target in ('policy_parent_memory_plan', 'native_fullframe_dense_reserve', 'TtaScheduler'):
            def changed_digest(node):
                return '0' * 64 if getattr(node, 'name', None) == target else original_digest(node)
            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, 'reviewed (?:added )?definition changed or is missing:'):
                    verify_inventory()

    def test_v22_release_preserves_and_authenticates_both_parent_snapshots(self):
        manifest = _historical_inventory()
        review = release_contract(manifest)
        self.assertEqual(review['parent_snapshots'], list(inventory.REVIEWED_V22_RELEASE_PARENTS))
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_COVERAGE_SHA256)
        self.assertEqual(len(review['statements']), 5)
        self.assertEqual(review['definitions'], [])

    def test_v22_release_rejects_historical_changes_on_either_parent(self):
        for key in ('v21_1_2_review', 'v22_augmentation_review', 'v22_coverage_review', 'v21_review'):
            manifest = _historical_inventory()
            manifest[key]['definitions'][0]['reason'] = 'Rewritten parent record'
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'release parent inventory changed'):
                release_contract(manifest)

    def test_v22_release_rejects_missing_and_unreviewed_inventory_appendices(self):
        manifest = _historical_inventory()
        del manifest['v21_1_2_review']
        with self.assertRaisesRegex(RuntimeError, 'release parent inventory changed'):
            inventory.reviewed_v22_release_contract(manifest, manifest['v21_review'])
        manifest = _historical_inventory()
        manifest['unreviewed_appendix'] = {}
        with self.assertRaisesRegex(RuntimeError, 'release inventory keyset differs'):
            release_contract(manifest)

    def test_v22_coverage_uses_its_own_pre_merge_keyset(self):
        manifest = _historical_inventory()
        expected = copy.deepcopy(coverage_contract(manifest))
        del manifest['v21_1_2_review']
        del manifest['v22_release_review']
        self.assertEqual(coverage_contract(manifest), expected)

    def test_v22_release_authenticates_parent_identity_resolution_and_binding_digests(self):
        for category in ('parent_snapshots', 'merge_resolution', 'statements'):
            manifest = _historical_inventory()
            review = manifest['v22_release_review']
            if category == 'parent_snapshots':
                review[category][0]['commit'] = '0' * 40
            elif category == 'merge_resolution':
                review[category]['main_reviews'] = []
            else:
                review[category][0]['sha256'] = '0' * 64
            with self.subTest(category=category), self.assertRaisesRegex(RuntimeError, 'v22.0.0 review digest mismatch'):
                release_contract(manifest)

    def test_v22_release_requires_version_successors_of_the_main_parent(self):
        manifest = _historical_inventory()
        review = manifest['v22_release_review']
        record = review['statements'][0]
        old_version = next(item for item in manifest['v21_1_1_review']['statements']
                           if item['module'] == '__init__')
        record['previous_sha256'] = old_version['sha256']
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_RELEASE_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'supersession does not match its historical pin: __init__'):
                release_contract(manifest)

    def test_v22_release_rejects_reauthenticated_parent_and_binding_omissions(self):
        for category in ('parent_snapshots', 'statements'):
            manifest = _historical_inventory()
            review = manifest['v22_release_review']
            review[category].pop()
            authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            message = 'unexpected parent snapshots' if category == 'parent_snapshots' else 'exactly its five release identity bindings'
            with self.subTest(category=category), mock.patch.object(inventory, 'REVIEWED_V22_RELEASE_SHA256', authenticated):
                with self.assertRaisesRegex(RuntimeError, message):
                    release_contract(manifest)

    def test_frontier_release_authenticates_new_modules_and_prior_release(self):
        manifest = _historical_inventory()
        review = manifest['v21_1_2_review']
        self.assertEqual(review['complete_modules'], ['lta_frontier', 'lta_frontier_execution'])
        self.assertEqual(review['previous_review_sha256'],
                         '84426381fa7b386607ed8e79a993cf0b51bbbe46b1cfa3db4bc71307385694a4')
        next(item for item in review['definitions'] if item['name'] == 'CanonicalFrontier')['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.1.2 review digest mismatch'):
            inventory.reviewed_v21_1_2_contract(
                manifest, manifest['v21_review'],
                *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
                manifest['v21_1_review'], manifest['v21_1_1_review'],
            )

    def test_frontier_release_requires_latest_version_predecessor(self):
        manifest = _historical_inventory()
        review = manifest['v21_1_2_review']
        next(item for item in review['statements'] if item['module'] == '__init__')['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_1_2_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.1.2 supersession does not match its historical pin'):
                inventory.reviewed_v21_1_2_contract(
                    manifest, manifest['v21_review'],
                    *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
                    manifest['v21_1_review'], manifest['v21_1_1_review'],
                )

    def test_frontier_controller_and_driver_have_independent_definition_pins(self):
        original_digest = inventory.digest
        for target in ('CanonicalFrontier', '_drive_canonical_relay_frontier'):
            def changed_digest(node):
                return '0' * 64 if getattr(node, 'name', None) == target else original_digest(node)

            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, 'reviewed definition changed or is missing:'):
                    verify_inventory()


    def test_coverage_review_follows_the_complete_augmentation_predecessor(self):
        manifest = _historical_inventory()
        review = coverage_contract(manifest)
        self.assertEqual(review['previous_review_sha256'], inventory.REVIEWED_V22_AUGMENTATION_SHA256)
        self.assertEqual(review['predecessor_commit'], 'f6557bf52822c8e8d0a752deb81fa26652b8a4e4')
        self.assertEqual(review['feature'], 'coverage-sampling')
        self.assertEqual(manifest['v22_augmentation_review']['release'], '22.0.0')
        self.assertEqual(manifest['v21_1_1_review']['release'], '21.1.1')

    def test_coverage_review_cannot_rewrite_any_historical_record(self):
        for key, field in (
            ('v22_augmentation_review', 'reason'), ('v21_review', 'reason'),
        ):
            manifest = _historical_inventory()
            manifest[key]['definitions'][0][field] = 'Rewritten historical review'
            with self.subTest(review=key), self.assertRaisesRegex(RuntimeError, 'coverage predecessor inventory changed'):
                coverage_contract(manifest)
        manifest = _historical_inventory()
        manifest['statements'][0]['line'] += 1
        with self.assertRaisesRegex(RuntimeError, 'coverage predecessor inventory changed'):
            coverage_contract(manifest)

    def test_coverage_review_authenticates_geometry_bindings_modules_and_proof(self):
        for category in ('definitions', 'statements', 'preserved_radial_module_updates', 'validation_tools'):
            manifest = _historical_inventory()
            manifest['v22_coverage_review'][category][0]['sha256'] = '0' * 64
            with self.subTest(category=category), self.assertRaisesRegex(RuntimeError, 'v22.0.0 review digest mismatch'):
                coverage_contract(manifest)

    def test_coverage_review_requires_the_latest_augmentation_definition_pin(self):
        manifest = _historical_inventory()
        review = manifest['v22_coverage_review']
        record = next(item for item in review['definitions']
                      if (item['module'], item['name']) == ('config', 'build_argparser'))
        prior = next(item for item in manifest['v22_augmentation_review']['definitions']
                     if (item['module'], item['name']) == ('config', 'build_argparser'))
        self.assertEqual(record['previous_sha256'], prior['sha256'])
        record['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_COVERAGE_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'supersession does not match its historical pin: config.build_argparser'):
                coverage_contract(manifest)

    def test_coverage_review_requires_exact_radial_module_predecessor(self):
        manifest = _historical_inventory()
        patches = [*(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
                   manifest['v21_1_review'], manifest['v21_1_1_review'],
                   manifest['v22_augmentation_review'], manifest['v22_coverage_review']]
        record = next(item for item in patches[-1]['preserved_radial_module_updates']
                      if item['module'] == 'cylindrical_geometry')
        record['previous_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Radial module review does not match its preserved predecessor'):
            inventory.reviewed_radial_module_hashes(manifest['v21_review'], patches)

    def test_coverage_new_modules_have_complete_statement_coverage(self):
        manifest = _historical_inventory()
        review = coverage_contract(manifest)
        self.assertEqual(set(review['complete_modules']), {'spherical_sampling', 'azimuthal_coverage'})
        original_parse = inventory.ast.parse

        for module in review['complete_modules']:
            def add_unreviewed_statement(source, filename='<unknown>', *args, **kwargs):
                tree = original_parse(source, filename, *args, **kwargs)
                if str(filename).replace('\\', '/').endswith(f'/{module}.py'):
                    tree.body.extend(original_parse('UNREVIEWED_COVERAGE_BINDING = 1').body)
                return tree

            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=add_unreviewed_statement):
                with self.assertRaisesRegex(RuntimeError, f'complete-module statement coverage differs: {module}'):
                    verify_inventory()

    def test_coverage_qsc_unit_bound_has_an_independent_statement_pin(self):
        original_digest = inventory.digest

        def change_bound(node):
            if isinstance(node, ast.Assign) and any(getattr(item, 'id', '') == 'QSC_INVERSE_LIPSCHITZ' for item in node.targets):
                return '0' * 64
            return original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=change_bound):
            with self.assertRaisesRegex(RuntimeError, 'reviewed statement changed or is missing: qsc[.]'):
                verify_inventory()

    def test_coverage_rational_certificate_source_cannot_change_without_review(self):
        manifest = _historical_inventory()
        review = coverage_contract(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'tools/certify_qsc_lipschitz.py'
            path.parent.mkdir()
            path.write_text('raise RuntimeError("unreviewed proof replacement")\n', encoding='utf-8')
            with mock.patch.object(inventory, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'coverage validation tool changed or is missing'):
                    inventory.verify_coverage_validation_tools(review)

    def test_augmentation_review_authenticates_definitions_bindings_and_relocations(self):
        for category in ('definitions', 'statements', 'definition_relocations'):
            manifest = _historical_inventory()
            manifest['v22_augmentation_review'][category][0]['sha256'] = '0' * 64
            with self.subTest(category=category), self.assertRaisesRegex(
                    RuntimeError, 'v22.0.0 review digest mismatch'):
                augmentation_contract(manifest)

    def test_augmentation_review_requires_latest_definition_predecessor(self):
        manifest = _historical_inventory()
        review = manifest['v22_augmentation_review']
        record = next(item for item in review['definitions']
                      if (item['module'], item['name']) == ('config', 'build_argparser'))
        record['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(
            review, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_AUGMENTATION_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'v22.0.0 supersession does not match its historical pin'):
                augmentation_contract(manifest)

    def test_augmentation_relocation_requires_independent_preserved_pta_predecessor(self):
        manifest = _historical_inventory()
        review = manifest['v22_augmentation_review']
        review['definition_relocations'][0]['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(
            review, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V22_AUGMENTATION_SHA256', authenticated):
            with self.assertRaisesRegex(RuntimeError, 'relocation does not match its preserved predecessor'):
                augmentation_contract(manifest)

    def test_augmentation_relocation_rejects_a_replacement_pta_wrapper(self):
        manifest = _historical_inventory()
        review = augmentation_contract(manifest)
        top_level = {
            module: ast.parse((inventory.PACKAGE / f'{module}.py').read_text(encoding='utf-8')).body
            for module in ('pta_augmentation', 'augmentation_policy')
        }
        top_level['pta_augmentation'].extend(ast.parse(
            'def inspect_augmentation_definition(path):\n    return None\n',
        ).body)
        with self.assertRaisesRegex(RuntimeError, 'shared-owner relocation changed: pta_augmentation.inspect_augmentation_definition'):
            inventory.verify_augmentation_relocations(review, top_level)

    def test_augmentation_new_modules_have_complete_statement_coverage(self):
        manifest = _historical_inventory()
        review = augmentation_contract(manifest)
        self.assertEqual(set(review['complete_modules']), {
            'augmentation_policy', 'tta_augmentation', 'tta_augmentation_config',
            'tta_augmentation_cuda', 'tta_augmentation_retirement', 'tta_augmentation_runtime',
        })
        original_parse = inventory.ast.parse

        def add_unreviewed_statement(source, filename='<unknown>', *args, **kwargs):
            tree = original_parse(source, filename, *args, **kwargs)
            if str(filename).replace('\\', '/').endswith('/tta_augmentation_cuda.py'):
                tree.body.extend(original_parse('UNREVIEWED_AUGMENTATION_BINDING = 1').body)
            return tree

        with mock.patch.object(inventory.ast, 'parse', side_effect=add_unreviewed_statement):
            with self.assertRaisesRegex(RuntimeError, 'complete-module statement coverage differs: tta_augmentation_cuda'):
                verify_inventory()

    def test_augmentation_cuda_source_has_an_independent_statement_pin(self):
        original_digest = inventory.digest

        def change_cuda_source(node):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and 'extern "C" __global__ void policy_grids(' in node.value.value):
                return '0' * 64
            return original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=change_cuda_source):
            with self.assertRaisesRegex(RuntimeError, 'reviewed statement changed or is missing: tta_augmentation_cuda'):
                verify_inventory()

    def test_overlap_release_authenticates_the_explicit_concurrency_option(self):
        manifest = _historical_inventory()
        manifest['v21_1_1_review']['definitions'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.1.1 review digest mismatch'):
            inventory.reviewed_v21_1_1_contract(
                manifest, manifest['v21_review'],
                *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)), manifest['v21_1_review'],
            )

    def test_lta_release_authenticates_runtime_version_bindings(self) -> None:
        manifest = _historical_inventory()
        manifest['v21_1_review']['statements'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.1.0 review digest mismatch'):
            inventory.reviewed_v21_1_contract(
                manifest, manifest['v21_review'],
                *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
            )

    def test_lta_release_requires_latest_version_predecessor(self) -> None:
        manifest = _historical_inventory()
        review = manifest['v21_1_review']
        review['statements'][0]['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(
            review, sort_keys=True, separators=(',', ':'),
        ).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_1_SHA256', authenticated):
            with self.assertRaisesRegex(
                RuntimeError, 'v21.1.0 supersession does not match its historical pin',
            ):
                inventory.reviewed_v21_1_contract(
                    manifest, manifest['v21_review'],
                    *(manifest[f'v21_0_{i}_review'] for i in range(1, 7)),
                )

    def test_release_authenticates_upload_and_observability(self) -> None:
        for category, target in (
            ('definitions', 'RadialCudaProjector'),
            ('definitions', 'backproject_spherical_volume_to_volume'),
            ('definitions', '_execution_runtime_provenance'),
            ('preserved_radial_definition_updates', None),
            ('preserved_radial_module_updates', None),
        ):
            manifest = _historical_inventory()
            records = manifest['v21_0_6_review'][category]
            record = records[0] if target is None else next(item for item in records if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(category=category, target=target), self.assertRaisesRegex(
                    RuntimeError, 'v21.0.6 review digest mismatch'):
                inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_requires_the_preserved_radial_class_predecessor(self) -> None:
        manifest = _historical_inventory()
        review = manifest['v21_0_6_review']
        next(item for item in review['definitions'] if item['name'] == 'RadialCudaProjector')['previous_sha256'] = '0' * 64
        authenticated = hashlib.sha256(json.dumps(review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_6_SHA256', authenticated), self.assertRaisesRegex(
                RuntimeError, 'v21.0.6 supersession does not match its historical pin'):
            inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_requires_exact_preserved_module_and_upload_predecessors(self) -> None:
        for category, validate in (
            ('preserved_radial_module_updates', inventory.reviewed_radial_module_hashes),
            ('preserved_radial_definition_updates', inventory.reviewed_radial_definition_hashes),
        ):
            manifest = _historical_inventory()
            patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 7)]
            patches[-1][category][0]['previous_sha256'] = '0' * 64
            with self.subTest(category=category), self.assertRaisesRegex(RuntimeError, 'preserved predecessor'):
                validate(manifest['v21_review'], patches)

    def test_release_does_not_allow_unknown_preserved_modules(self) -> None:
        manifest = _historical_inventory()
        patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 7)]
        patches[-1]['preserved_radial_module_updates'][0]['module'] = 'unreviewed_module'
        with self.assertRaisesRegex(RuntimeError, 'unknown, duplicate or unexplained Radial module'):
            inventory.reviewed_radial_module_hashes(manifest['v21_review'], patches)

    def test_source_upload_method_has_an_independent_review_pin(self) -> None:
        original_digest = inventory.digest
        def altered_upload(node):
            return '0' * 64 if getattr(node, 'name', None) == '_upload_cropped_source' else original_digest(node)
        with mock.patch.object(inventory, 'digest', side_effect=altered_upload), self.assertRaisesRegex(
                RuntimeError, 'preserved Radial arithmetic: RadialCudaProjector._upload_cropped_source'):
            verify_inventory()

    def test_release_pins_native_ring_and_observability_contracts(self) -> None:
        for target in ('_ResidentTensorRTRingExecutor', 'GpuRenderedYoloSource',
                       '_claim_specialized_prediction_targets', 'RuntimeTelemetry', 'TtaScheduler'):
            manifest = _historical_inventory()
            review = manifest['v21_0_6_review']
            record = next(item for item in review['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.6 review digest mismatch'):
                inventory.reviewed_v21_0_6_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 6)))

    def test_release_authenticates_compact_projection_and_age_admission(self) -> None:
        for target in ('_MainProcessGpuStageCoordinator', '_project_spherical_encoded_block',
                       '_pull_spherical_f64'):
            manifest = _historical_inventory()
            review = manifest['v21_0_5_review']
            record = next(item for item in review['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_release_checks_current_compiled_kernel(self) -> None:
        manifest = _historical_inventory()
        review = manifest['v21_0_5_review']
        record = next(item for item in review['definitions'] if item['name'] == '_pull_spherical_f64')
        self.assertIsNone(record['previous_sha256'])
        original_digest = inventory.digest

        def altered_kernel(node):
            return '0' * 64 if getattr(node, 'name', None) == '_pull_spherical_f64' else original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=altered_kernel), self.assertRaisesRegex(
                RuntimeError, 'v21 patch reviewed definition changed or is missing: spherical_projection_cpu._pull_spherical_f64'):
            verify_inventory()

    def test_release_authenticates_fast_geometry_and_preserved_radial_update(self) -> None:
        manifest = _historical_inventory()
        review = manifest['v21_0_5_review']
        update = review['preserved_radial_definition_updates'][0]
        update['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
            inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_preserved_radial_update_requires_the_exact_predecessor(self) -> None:
        manifest = _historical_inventory()
        patches = [manifest[f'v21_0_{i}_review'] for i in range(1, 6)]
        patches[-1]['preserved_radial_definition_updates'][0]['previous_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'preserved predecessor'):
            inventory.reviewed_radial_definition_hashes(manifest['v21_review'], patches)

    def test_release_authenticates_scheduling_and_cancelled_reader_lifetime(self) -> None:
        for target in ('TtaScheduler', '_MainProcessGpuStageCoordinator', '_ordered_spherical_blocks'):
            manifest = _historical_inventory()
            record = next(item for item in manifest['v21_0_5_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.5 review digest mismatch'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_release_cannot_skip_the_latest_retirement_coordinator(self) -> None:
        manifest = _historical_inventory()
        fifth = manifest['v21_0_5_review']
        record = next(item for item in fifth['definitions'] if item['name'] == '_MainProcessGpuStageCoordinator')
        record['previous_sha256'] = '0' * 64
        reauthenticated = hashlib.sha256(json.dumps(fifth, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_5_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.5 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_5_contract(
                    manifest, manifest['v21_review'], *(manifest[f'v21_0_{i}_review'] for i in range(1, 5)))

    def test_fourth_patch_authenticates_geometry_projection_and_compaction(self) -> None:
        for target in ('qsc_inverse', '_project_spherical_block', '_resident_mask_kernels'):
            manifest = _historical_inventory()
            record = next(item for item in manifest['v21_0_4_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.4 review digest mismatch'):
                inventory.reviewed_v21_0_4_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'],
                    manifest['v21_0_2_review'], manifest['v21_0_3_review'],
                )

    def test_fourth_patch_cannot_skip_the_latest_diagnostic_revision(self) -> None:
        manifest = _historical_inventory()
        fourth = manifest['v21_0_4_review']
        record = next(item for item in fourth['definitions'] if item['name'] == '_announce_direct_compaction_layout')
        record['previous_sha256'] = '0' * 64
        reauthenticated = hashlib.sha256(json.dumps(
            fourth, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_4_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.4 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_4_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'],
                    manifest['v21_0_2_review'], manifest['v21_0_3_review'],
                )

    def test_third_patch_cannot_skip_the_latest_coordinator_revision(self) -> None:
        manifest = _historical_inventory()
        prior = inventory.reviewed_v21_contract(manifest)
        first = inventory.reviewed_v21_patch_contract(manifest, prior)
        second = inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        third = inventory.reviewed_v21_0_3_contract(manifest, prior, first, second)
        self.assertEqual(third['previous_review_sha256'], inventory.REVIEWED_V21_0_2_SHA256)
        earlier = next(item for item in first['definitions']
                       if item['name'] == '_MainProcessGpuStageCoordinator')
        current = next(item for item in third['definitions']
                       if item['name'] == '_MainProcessGpuStageCoordinator')
        current['previous_sha256'] = earlier['sha256']
        reauthenticated = hashlib.sha256(json.dumps(
            third, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_3_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.3 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_3_contract(manifest, prior, first, second)

    def test_third_patch_authenticates_preflight_and_layout_diagnostics(self) -> None:
        for target in ('validate_spherical_preflight_plane', '_announce_direct_compaction_layout'):
            manifest = _historical_inventory()
            record = next(item for item in manifest['v21_0_3_review']['definitions'] if item['name'] == target)
            record['sha256'] = '0' * 64
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, 'v21.0.3 review digest mismatch'):
                inventory.reviewed_v21_0_3_contract(
                    manifest, manifest['v21_review'], manifest['v21_0_1_review'], manifest['v21_0_2_review'],
                )

    def test_third_patch_checks_the_runtime_preflight_pixel_budget(self) -> None:
        original_digest = inventory.digest

        def changed_budget(node):
            targets = getattr(node, 'targets', ())
            if any(getattr(target, 'id', None) == '_PROBE_PIXELS' for target in targets):
                return '0' * 64
            return original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=changed_budget):
            with self.assertRaisesRegex(RuntimeError, 'reviewed statement changed or is missing: spherical_preflight.binding__PROBE_PIXELS'):
                verify_inventory()

    def test_second_patch_cannot_skip_the_intermediate_release(self) -> None:
        manifest = _historical_inventory()
        prior = inventory.reviewed_v21_contract(manifest)
        first = inventory.reviewed_v21_patch_contract(manifest, prior)
        second = inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        self.assertEqual(second['previous_review_sha256'], inventory.REVIEWED_V21_0_1_SHA256)
        old_record = next(item for item in first['definitions']
                          if item['name'] == '_MainProcessGpuStageCoordinator')
        record = next(item for item in second['definitions']
                      if item['name'] == '_MainProcessGpuStageCoordinator')
        record['previous_sha256'] = old_record['previous_sha256']
        reauthenticated = hashlib.sha256(json.dumps(
            second, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        with mock.patch.object(inventory, 'REVIEWED_V21_0_2_SHA256', reauthenticated):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.2 supersession does not match its historical pin'):
                inventory.reviewed_v21_0_2_contract(manifest, prior, first)
        manifest['v21_0_1_review']['release'] = '21.0.2'
        with self.assertRaisesRegex(RuntimeError, 'v21.0.1 review digest mismatch'):
            inventory.reviewed_v21_patch_contract(manifest, prior)

    def test_second_patch_pins_current_compaction_and_bounds_definitions(self) -> None:
        original_digest = inventory.digest
        for target in ('_build_direct_device_compacted_payload', 'GpuFlattenedRetinaPayload', 'spherical_output_bounds'):
            def changed_digest(node):
                return '0' * 64 if getattr(node, 'name', None) == target else original_digest(node)

            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, 'reviewed definition changed or is missing:'):
                    verify_inventory()

    def test_new_complete_modules_reject_an_unreviewed_top_level_statement(self) -> None:
        original_parse = ast.parse

        def parse_with_extra_statement(source, filename='<unknown>', *args, **kwargs):
            tree = original_parse(source, filename, *args, **kwargs)
            if str(filename).replace('\\', '/').endswith(f'/{module}.py'):
                tree.body.append(ast.Assign(
                    targets=[ast.Name(id='UNREVIEWED_POLICY', ctx=ast.Store())],
                    value=ast.Constant(value=True),
                ))
            return tree

        for module in ('spherical_projection_bounds', 'spherical_preflight',
                       'geometry_quality', 'spherical_projection_cpu', 'spherical_sampling_cuda',
                       'lta_frontier', 'lta_frontier_execution'):
            with self.subTest(module=module), mock.patch.object(inventory.ast, 'parse', side_effect=parse_with_extra_statement):
                with self.assertRaisesRegex(RuntimeError, f'complete-module statement coverage differs: {module}'):
                    verify_inventory()

    def test_patch_review_keeps_the_prior_release_authenticated(self) -> None:
        manifest = _historical_inventory()
        prior = inventory.reviewed_v21_contract(manifest)
        patch = inventory.reviewed_v21_patch_contract(manifest, prior)
        self.assertEqual(prior['release'], '21.0.0')
        self.assertEqual(patch['release'], '21.0.1')
        self.assertEqual(patch['previous_review_sha256'], inventory.REVIEWED_V21_SHA256)
        manifest['v21_review']['definitions'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'v21 review digest mismatch'):
            inventory.reviewed_v21_contract(manifest)

    def test_patch_admission_and_memory_contracts_cannot_change_without_review(self) -> None:
        for module, name in (
            ('backprojection', '_MainProcessGpuStageCoordinator'),
            ('publication_memory', 'native_fullframe_dense_reserve'),
        ):
            manifest = _historical_inventory()
            record = next(item for item in manifest['v21_0_1_review']['definitions']
                          if (item['module'], item['name']) == (module, name))
            record['sha256'] = '0' * 64
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'v21.0.1 review digest mismatch'):
                inventory.reviewed_v21_patch_contract(manifest, manifest['v21_review'])

    def test_patch_supersessions_chain_from_both_v21_and_v20_pins(self) -> None:
        for module, name in (
            ('pipeline', '_main_impl'),
            ('publication_memory', 'plan_native_publication_memory'),
        ):
            manifest = _historical_inventory()
            patch = manifest['v21_0_1_review']
            record = next(item for item in patch['definitions']
                          if (item['module'], item['name']) == (module, name))
            record['previous_sha256'] = '0' * 64
            reauthenticated = hashlib.sha256(json.dumps(
                patch, sort_keys=True, separators=(',', ':'),
            ).encode('utf-8')).hexdigest()
            with self.subTest(name=name), mock.patch.object(inventory, 'REVIEWED_V21_0_1_SHA256', reauthenticated):
                with self.assertRaisesRegex(RuntimeError, f'supersession does not match its historical pin: {module}.{name}'):
                    inventory.reviewed_v21_patch_contract(manifest, manifest['v21_review'])

    def test_patch_checks_current_added_definitions_and_retry_policy(self) -> None:
        original_digest = inventory.digest
        for target, expected_error in (
            ('native_fullframe_dense_reserve', 'reviewed definition changed or is missing: publication_memory.native_fullframe_dense_reserve'),
            ('_CUDA_RECHECK_SLICES', 'reviewed statement changed or is missing: spherical_projection.binding__CUDA_RECHECK_SLICES'),
        ):
            def changed_digest(node):
                names = [getattr(node, 'name', None)]
                if isinstance(node, ast.Assign):
                    names.extend(getattr(item, 'id', None) for item in node.targets)
                return '0' * 64 if target in names else original_digest(node)

            with self.subTest(target=target), mock.patch.object(inventory, 'digest', side_effect=changed_digest):
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    verify_inventory()

    def test_numba_compile_policy_is_pinned_outside_function_bodies(self) -> None:
        key = ('cylindrical_projection', 'numba_compile_policy')
        _expected_hash, reason = REVIEWED_V20_ADDED_STATEMENTS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_STATEMENTS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'reviewed added statement changed or is missing: cylindrical_projection.numba_compile_policy'):
                verify_inventory()

    def test_scalar_cuda_kernel_has_an_independent_review_pin(self) -> None:
        key = ('cuda_backend', '_radial_native_kernels')
        _expected_hash, reason = REVIEWED_V20_ADDED_DEFINITIONS[key]
        with mock.patch.dict(REVIEWED_V20_ADDED_DEFINITIONS, {key: ('0' * 64, reason)}):
            with self.assertRaisesRegex(RuntimeError, 'v21.0.5 supersession does not match its historical pin: cuda_backend._radial_native_kernels'):
                verify_inventory()

        original_digest = inventory.digest

        def changed_kernel(node):
            return '0' * 64 if getattr(node, 'name', None) == key[1] else original_digest(node)

        with mock.patch.object(inventory, 'digest', side_effect=changed_kernel):
            with self.assertRaisesRegex(RuntimeError, 'v20 reviewed added definition changed or is missing: cuda_backend._radial_native_kernels'):
                verify_inventory()

    def test_azimuthal_rename_pins_exact_ast_without_accepting_shell_names_or_changed_math(self) -> None:
        old = ast.parse('def radial_sample(value):\n    return value + 1\n').body[0]
        renamed = ast.parse('def azimuthal_sample(value):\n    return value + 1\n').body[0]
        altered = ast.parse('def azimuthal_sample(value):\n    return value + 2\n').body[0]
        baseline_hash = digest(old)
        record = {
            'module': 'sample', 'baseline_sha256': baseline_hash,
            'renamed_sha256': digest(renamed), 'baseline_name': 'radial_sample',
            'current_name': 'azimuthal_sample',
        }
        manifest = {
            'statements': [{'module': 'sample', 'sha256': baseline_hash, 'name': 'radial_sample'}],
            'v20_azimuthal_rename': {'statements': [record]},
        }
        replacement = azimuthal_rename_replacements(manifest)['sample', baseline_hash]
        self.assertEqual(replacement, digest(renamed))
        self.assertNotEqual(replacement, digest(old), 'A new shell function reusing the old name is a different declaration')
        self.assertNotEqual(replacement, digest(altered), 'The rename map must not normalize arithmetic changes')
        for updates, message in (
            ({'baseline_sha256': '0' * 64}, 'absent immutable'),
            ({'current_name': 'radial_sample'}, 'identity mismatch'),
            ({'renamed_sha256': 'not-a-digest'}, 'invalid reviewed'),
        ):
            broken = copy.deepcopy(manifest)
            broken['v20_azimuthal_rename']['statements'][0].update(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(RuntimeError, message):
                azimuthal_rename_replacements(broken)
        duplicated = copy.deepcopy(manifest)
        duplicated['v20_azimuthal_rename']['statements'].append(record)
        with self.assertRaisesRegex(RuntimeError, 'duplicate Azimuthal rename'):
            azimuthal_rename_replacements(duplicated)

    def test_original_inventory_cannot_be_silently_rebaselined(self) -> None:
        manifest = _historical_inventory()
        manifest['statements'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'inventory.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with mock.patch('tools.verify_package_inventory.MANIFEST', path):
                with self.assertRaisesRegex(RuntimeError, 'immutable inventory digest mismatch'):
                    verify_inventory()

    def test_ast_dump_keeps_empty_fields_across_python_versions(self) -> None:
        function = ast.parse("def callback():\n    pass\n").body[0]

        dumped = stable_ast_dump(function)

        self.assertIn("decorator_list=[]", dumped)

    def test_marker_move_changes_seam_digest_even_when_definition_ast_is_unchanged(self) -> None:
        first_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                {LOCAL_IMPORT_SEAM_MARKER}
                from .first import run_first
                # The other callback remains local for unrelated reasons.
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )
        second_import_reviewed = inspect_seams(
            f"""
            def callback(flag):
                # The first callback remains local for unrelated reasons.
                from .first import run_first
                {LOCAL_IMPORT_SEAM_MARKER}
                from .second import run_second
                return run_first() if flag else run_second()
            """
        )

        first_definition, first_seam = first_import_reviewed[("sample", "callback")]
        second_definition, second_seam = second_import_reviewed[("sample", "callback")]
        self.assertEqual(first_definition, second_definition)
        self.assertNotEqual(first_seam, second_seam)

    def test_marker_inside_method_pins_the_enclosing_top_level_class(self) -> None:
        reviewed = inspect_seams(
            f"""
            class CallbackOwner:
                def callback(self):
                    {LOCAL_IMPORT_SEAM_MARKER}
                    from .dependency import run
                    return run()
            """
        )
        self.assertEqual(set(reviewed), {("sample", "CallbackOwner")})

    def test_marker_must_immediately_precede_a_relative_function_local_import(self) -> None:
        invalid_sources = (
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}

                from .dependency import run
            """,
            f"""
            def callback():
                {LOCAL_IMPORT_SEAM_MARKER}
                from dependency import run
            """,
            f"""
            {LOCAL_IMPORT_SEAM_MARKER}
            from .dependency import run
            """,
        )
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                inspect_seams(source)

    def test_changed_binding_review_must_reference_the_immutable_inventory(self) -> None:
        unknown = ("config", "0" * 64)
        with mock.patch.dict(
            INTENTIONALLY_CHANGED_BINDINGS,
            {unknown: "SAVE_OPTION_TOKENS"},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "changed-binding entries are absent from the immutable inventory",
            ):
                verify_inventory()


if __name__ == "__main__":
    unittest.main()
