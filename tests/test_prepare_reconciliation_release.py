"""Review preparation preserves statement identity and independent history pins."""
from __future__ import annotations

import ast
import hashlib
import json

import pytest

from tools import prepare_reconciliation_release as prepare
from tools import verify_package_inventory as inventory


def review(old, new, *, complete=False, labels=None):
    return prepare.review_module('sample', old, new, complete=complete,
                                 labels_by_hash=labels or {}, reason='Reviewed fixture change.')


def test_definition_and_binding_changes_reconstruct_the_exact_predecessor():
    old = 'import math\nVALUE = 1\ndef calculate():\n    return VALUE\n'
    new = 'import math\nVALUE = 2\ndef calculate():\n    return VALUE + 1\ndef extra():\n    return 3\n'
    pin, snapshot, records = review(old, new)
    assert pin['ast_sha256'] == inventory.digest(ast.parse(old))
    assert pin['statements_sha256'] == hashlib.sha256(json.dumps(snapshot['previous_top_level'], separators=(',', ':')).encode()).hexdigest()
    assert {item['name'] for item in records['definitions']} == {'calculate', 'extra'}
    assert [item['binding'] for item in records['statements']] == ['VALUE']
    payload = {**records, 'module_snapshots': [snapshot]}
    restored = inventory.rewind_reviewed_statements('sample', ast.parse(new).body, (payload,))
    assert [value for _node, value in restored] == snapshot['previous_top_level']


def test_new_module_accounts_for_all_statements_and_has_no_fabricated_predecessor():
    source = '"""A new module."""\nimport math\nVALUE = 1\ndef calculate():\n    return VALUE\n'
    pin, snapshot, records = review(None, source, complete=True)
    assert pin['ast_sha256'] is None
    assert snapshot['previous_top_level'] == []
    entries = records['definitions'] + records['statements']
    assert len(entries) == len(ast.parse(source).body)
    assert all(item['previous_sha256'] is None and item['previous_index'] is None for item in entries)
    assert sorted(item['current_index'] for item in entries) == list(range(len(entries)))
    assert inventory.rewind_reviewed_statements('sample', ast.parse(source).body,
                                              ({**records, 'module_snapshots': [snapshot]},)) == []


def test_predecessor_definition_deletion_cannot_disappear_from_the_review():
    with pytest.raises(ValueError, match='unaccounted predecessor statements'):
        review('def required():\n    return 1\n', 'VALUE = 2\n')


def test_explicit_example_module_removal_keeps_the_exact_historical_snapshot():
    module = 'examples/external_reconciliation/baseline'
    source = '"""A retired example."""\ndef build_reconciliation():\n    return {"mode": "confidence"}\n'
    pin, snapshot, records = prepare.review_removed_module(module, source, reason='Reviewed example retirement.')
    assert pin['ast_sha256'] == inventory.digest(ast.parse(source))
    assert snapshot['removed'] is True
    assert snapshot['ast_sha256'] is None
    assert snapshot['top_level'] == []
    assert not any(records.values())
    payload = {**records, 'module_snapshots': [snapshot]}
    restored = inventory.rewind_reviewed_statements(module, (), (payload,))
    assert [value for _node, value in restored] == snapshot['previous_top_level']
    _, previous_snapshot, previous_records = prepare.review_module(
        module, None, source, complete=True, labels_by_hash={}, reason='Original example.')
    previous_review = {**previous_records, 'module_snapshots': [previous_snapshot], 'release': '22.3.0'}
    removal_review = {**payload, 'release': '22.3.1'}
    inventory.verify_v22_3_source_snapshots(previous_review, {}, (removal_review,))
    inventory.verify_v22_3_source_snapshots(removal_review, {})
    assert inventory.rewind_reviewed_statements(module, (), (previous_review, removal_review)) == []
    with pytest.raises(RuntimeError, match='v22.3.0 reviewed source changed'):
        inventory.verify_v22_3_source_snapshots(previous_review, {})
    with pytest.raises(RuntimeError, match='reviewed source statements changed'):
        inventory.rewind_reviewed_statements(module, ast.parse(source).body, (payload,))


@pytest.mark.parametrize('module, source', [
    ('pipeline', 'VALUE = 1\n'),
    ('examples/external_reconciliation/union', 'VALUE = 1\n'),
    ('examples/external_reconciliation/baseline', None),
])
def test_module_removal_requires_an_explicitly_reviewed_existing_example(module, source):
    with pytest.raises(ValueError, match='Module deletion needs a separate review'):
        prepare.review_removed_module(module, source, reason='Unreviewed removal.')


def test_reviewed_statement_labels_and_duplicate_binding_positions_are_preserved():
    old = 'VALUE = 1\nVALUE = 2\n'
    first, second = [inventory.digest(node) for node in ast.parse(old).body]
    labels = {('sample', first): 'first_value', ('sample', second): 'second_value'}
    _, _, records = review(old, 'VALUE = 3\nVALUE = 2\n', labels=labels)
    assert len(records['statements']) == 1
    changed = records['statements'][0]
    assert changed['label'] == 'first_value'
    assert changed['previous_index'] == changed['current_index'] == 0
    assert changed['previous_sha256'] == first


def test_pin_update_changes_only_the_requested_release_bindings():
    source = "HISTORY = 'authenticated'\nNEW_SHA256 = ''\nNEW_PREDECESSOR_MODULES = {}\nOTHER = 5\n"
    result = prepare._update_verifier_pins(source, 'NEW', 'a' * 64, {'sample': {'ast_sha256': None}})
    before, after = ast.parse(source), ast.parse(result)
    assert inventory.digest(before.body[0]) == inventory.digest(after.body[0])
    assert inventory.digest(before.body[-1]) == inventory.digest(after.body[-1])
    assert ast.literal_eval(after.body[1].value) == 'a' * 64
    assert ast.literal_eval(after.body[2].value) == {'sample': {'ast_sha256': None}}
    with pytest.raises(ValueError, match='missing'):
        prepare._update_verifier_pins("NEW_SHA256 = ''\n", 'NEW', 'a' * 64, {})


def test_generated_review_evidence_cannot_be_written_inside_the_repository():
    with pytest.raises(ValueError, match='outside the repository'):
        prepare.prepare(output_dir=prepare.ROOT / 'generated_review')
