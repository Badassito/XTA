"""Reuse the exact producer selection, preserving evidence and receipt checks."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from XTA import sam_extrapolation as core
from XTA.artifact_archive import reference
from XTA.sam_extrapolation_policy import fingerprint
from tests.test_sam_iterative_extrapolation import _case, _decode_components


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
@pytest.mark.parametrize('archived', [False, True])
def test_each_retry_selects_once_and_reuses_exact_receipt(tmp_path, monkeypatch, mode, archived):
    root = Path(reference(tmp_path/'retry.tar', 'run')) if archived else tmp_path/'run'
    case = _case(root, mode=mode)
    selected, loaded = {}, []
    counts = Counter()
    original_select, original_read = core.select_sam_extrapolation, core.read_artifact
    def select(bundle, **kwargs):
        result = original_select(bundle, **kwargs)
        counts[bundle.evidence_fingerprint] += 1
        selected[result['selection_identity']] = deepcopy(result)
        return result
    def read(path, **kwargs):
        result = original_read(path, **kwargs)
        if 'retry_attempts' in str(path) and str(path).endswith('selection.json'):
            receipt = json.loads(result)
            assert receipt == selected[receipt['selection_identity']]
            loaded.append(receipt['selection_identity'])
        return result
    monkeypatch.setattr(core, 'select_sam_extrapolation', select)
    monkeypatch.setattr(core, 'read_artifact', read)
    _, _, components = case.run()
    assert len(case.providers) >= 2
    assert len(loaded) == len(case.providers)
    assert len(counts) == len(case.providers) + 2 and set(counts.values()) == {1}
    np.testing.assert_array_equal(_decode_components(components, case.observed.shape), case.truth & ~case.original)
    np.testing.assert_array_equal(case.observed, case.original)


@pytest.mark.parametrize('mutation', ['missing', 'json', 'not_object',
                                     'unsigned', 'resigned_policy', 'resigned_frames', 'wrong_evidence'])
def test_changed_retry_receipt_fails_without_reselection_or_final_publication(tmp_path, monkeypatch, mutation):
    case = _case(tmp_path)
    original_write, original_select = core.write_json_atomic, core.select_sam_extrapolation
    calls = Counter()
    def select(bundle, **kwargs):
        calls[bundle.evidence_fingerprint] += 1
        return original_select(bundle, **kwargs)
    def write(path, payload, **kwargs):
        result = original_write(path, payload, **kwargs)
        if 'retry_attempts' not in str(path) or Path(path).name != 'selection.json':
            return result
        path = Path(path)
        if mutation == 'missing':
            path.unlink()
        elif mutation == 'json':
            path.write_text('{' + ' '*(path.stat().st_size-1))
        elif mutation == 'not_object':
            path.write_text('[]' + ' '*(path.stat().st_size-2))
        else:
            changed = deepcopy(payload)
            if mutation == 'resigned_frames':
                changed['selected_frames_by_run'][changed['selected_run_ids'][0]][0] += 1
            elif mutation == 'wrong_evidence':
                changed['evidence_fingerprint'] = '0'*64
            else:
                changed['policy_hash'] = 'f'*64
            if mutation != 'unsigned':
                changed['selection_identity'] = fingerprint({k:v for k,v in changed.items() if k != 'selection_identity'})
            path.write_text(json.dumps(changed, indent=2, ensure_ascii=True, allow_nan=False), newline='\n')
        return result
    monkeypatch.setattr(core, 'select_sam_extrapolation', select)
    monkeypatch.setattr(core, 'write_json_atomic', write)
    with pytest.raises((ValueError, FileNotFoundError)):
        case.run()
    assert len(case.providers) == 1 and set(calls.values()) == {1}
    destination = next(tmp_path.glob('sam_extrap_*'))
    assert not (destination/'selection.json').exists()
    assert not list(destination.glob('sam_extrapolation_*.cvol'))
    assert json.loads((destination/'crop_retry.json').read_text())['status'] == 'failed'
    np.testing.assert_array_equal(case.observed, case.original)


def test_contact_reader_still_detects_evidence_changed_after_selection_reuse(tmp_path, monkeypatch):
    case = _case(tmp_path)
    original = core.read_artifact
    def read(path, **kwargs):
        result = original(path, **kwargs)
        if 'retry_attempts' in str(path) and Path(path).name == 'selection.json':
            with (Path(path).parent/'evidence'/'masks.bin').open('r+b') as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
        return result
    monkeypatch.setattr(core, 'read_artifact', read)
    with pytest.raises(ValueError, match='checksum|changed'):
        case.run()
    destination = next(tmp_path.glob('sam_extrap_*'))
    assert not (destination/'selection.json').exists()
    assert not list(destination.glob('sam_extrapolation_*.cvol'))


def test_archived_replacement_cannot_reauthorize_another_selection(tmp_path, monkeypatch):
    case = _case(Path(reference(tmp_path/'retry.tar', 'run')))
    original = core.write_json_atomic
    changed = []
    def write(path, payload, **kwargs):
        result = original(path, payload, **kwargs)
        if 'retry_attempts' in str(path) and Path(path).name == 'selection.json':
            replacement = deepcopy(payload)
            replacement['policy_hash'] = 'f'*64
            replacement['selection_identity'] = fingerprint({k:v for k,v in replacement.items() if k != 'selection_identity'})
            original(path, replacement, **kwargs)
            changed.append(path)
        return result
    monkeypatch.setattr(core, 'write_json_atomic', write)
    with pytest.raises(ValueError, match='differs from its generated receipt'):
        case.run()
    assert len(changed) == 1 and len(case.providers) == 1


@pytest.mark.parametrize('field', ['selection_identity', 'policy_hash', 'selection_receipt_bytes'])
def test_missing_trusted_producer_identity_fails_closed(tmp_path, monkeypatch, field):
    case = _case(tmp_path)
    original = core.extrapolate_sam_view_volume_pass
    def generate(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('_evidence_only'):
            result[1].pop(field)
        return result
    monkeypatch.setattr(core, 'extrapolate_sam_view_volume_pass', generate)
    with pytest.raises(ValueError, match='generated receipt|generated byte bound'):
        case.run()
    assert len(case.providers) == 1


def test_receipt_growth_after_production_exceeds_trusted_byte_bound(tmp_path, monkeypatch):
    case = _case(tmp_path)
    original = core.extrapolate_sam_view_volume_pass
    def generate(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('_evidence_only'):
            stats = result[1]
            path = Path(stats['sam_selection_receipt_path'])
            assert path.stat().st_size == stats['selection_receipt_bytes']
            with path.open('ab') as stream:
                stream.write(b' ')
        return result
    monkeypatch.setattr(core, 'extrapolate_sam_view_volume_pass', generate)
    with pytest.raises(ValueError, match='bounded read budget'):
        case.run()
    assert len(case.providers) == 1


def test_large_producer_authored_receipt_has_no_new_fixed_size_limit(tmp_path, monkeypatch):
    case = _case(tmp_path)
    original = core.select_sam_extrapolation
    padded = False
    def select(bundle, **kwargs):
        nonlocal padded
        receipt = original(bundle, **kwargs)
        if not padded and 'retry_attempts' in str(bundle.directory):
            padded = True
            receipt['retained_diagnostic'] = 'x'*(64*1024**2+1)
            receipt['selection_identity'] = fingerprint({k:v for k,v in receipt.items() if k != 'selection_identity'})
        return receipt
    monkeypatch.setattr(core, 'select_sam_extrapolation', select)
    _, _, components = case.run()
    assert padded
    np.testing.assert_array_equal(_decode_components(components, case.observed.shape), case.truth & ~case.original)
