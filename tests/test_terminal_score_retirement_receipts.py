"""Independent mask/score death receipts preserve exact scheduler byte ownership."""
from concurrent.futures import Future
import gc
import itertools
from pathlib import Path
import queue
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA.interpolation import _DirectUnionBackingLease
from XTA.view_prepare import ViewPrepareLeaseState


MASK_BYTES = SCORE_BYTES = 1024
KEY = ('model', 'completed')


def _ledger(packed_bytes=0):
    proofs = (('mask', 'mask-proof'), ('confidence', 'score-proof'))
    if packed_bytes:
        proofs += (('packed', 'packed-proof'),)
    lease = _DirectUnionBackingLease(KEY, MASK_BYTES + SCORE_BYTES,
        phase='postprocess', memfd_owner_proofs=proofs)
    ledger = ViewPrepareLeaseState({KEY: lease}, set(), {}, {KEY}, {KEY: lease.nbytes})
    return ledger, lease


def _assert_charge(ledger, lease, expected):
    assert not ledger.inference_views and not ledger.inference_bytes
    if expected:
        assert ledger.leases[KEY] is lease
        assert ledger.postprocess_views == {KEY}
        assert ledger.postprocess_bytes == {KEY: expected}
        assert lease.nbytes == expected and lease.phase == 'postprocess'
        assert lease.owner_count == 1
    else:
        assert not ledger.leases and not ledger.postprocess_views
        assert not ledger.postprocess_bytes
        assert lease.phase == 'released' and lease.owner_count == 0


@pytest.mark.parametrize('packed_bytes', [0, 64])
@pytest.mark.parametrize('order', [('mask', 'confidence'), ('confidence', 'mask')])
def test_independent_input_receipts_refund_once_in_either_order(packed_bytes, order):
    ledger, lease = _ledger(packed_bytes)
    amounts = {'mask': MASK_BYTES - packed_bytes, 'confidence': SCORE_BYTES}
    expected = MASK_BYTES + SCORE_BYTES
    for token in order:
        assert ledger.retire_input_bytes(KEY, lease, amounts[token], token=token)
        expected -= amounts[token]
        _assert_charge(ledger, lease, expected)
        assert token not in {proof[0] for proof in lease.memfd_owner_proofs}
        assert not ledger.retire_input_bytes(KEY, lease, amounts[token], token=token)
        _assert_charge(ledger, lease, expected)
    assert expected == packed_bytes
    assert lease.memfd_owner_proofs == ((('packed', 'packed-proof'),) if packed_bytes else ())


@pytest.mark.parametrize('invalid_bytes', [0, -1, MASK_BYTES + SCORE_BYTES,
                                         MASK_BYTES + SCORE_BYTES + 1])
def test_score_receipt_cannot_remove_a_still_charged_mask(invalid_bytes):
    ledger, lease = _ledger()
    original_proofs = lease.memfd_owner_proofs
    with pytest.raises(RuntimeError, match='invalid dense credit'):
        ledger.retire_input_bytes(KEY, lease, invalid_bytes, token='confidence')
    _assert_charge(ledger, lease, MASK_BYTES + SCORE_BYTES)
    assert lease.memfd_owner_proofs == original_proofs
    assert not ledger.retired_inputs


@pytest.mark.parametrize('mask_identity', ['different_key', 'different_lease'])
def test_final_score_refund_requires_mask_receipt_for_the_exact_lease(mask_identity):
    ledger, lease = _ledger()
    if mask_identity == 'different_key':
        ledger.retired_inputs.add((('model', 'other'), id(lease), 'mask'))
    else:
        other = _DirectUnionBackingLease(KEY, MASK_BYTES, phase='released', owner_count=0)
        ledger.retired_inputs.add((KEY, id(other), 'mask'))
    with pytest.raises(RuntimeError, match='invalid dense credit'):
        ledger.retire_input_bytes(KEY, lease, MASK_BYTES + SCORE_BYTES, token='confidence')
    _assert_charge(ledger, lease, MASK_BYTES + SCORE_BYTES)
    assert {proof[0] for proof in lease.memfd_owner_proofs} == {'mask', 'confidence'}


@pytest.mark.parametrize('packed_bytes', [0, 64])
@pytest.mark.parametrize('order', list(itertools.permutations(('mask', 'confidence', 'publication'))))
def test_real_owner_death_receipts_and_publication_settle_in_every_order(
        tmp_path, packed_bytes, order):
    ledger, lease = _ledger(packed_bytes)
    owners = [np.zeros(MASK_BYTES, np.uint8), np.zeros(SCORE_BYTES, np.uint8)]
    refs = tuple(weakref.ref(owner) for owner in owners)
    receipts = {}
    weakref.finalize(owners[0], receipts.__setitem__, 'mask', (MASK_BYTES - packed_bytes,))
    weakref.finalize(owners[1], receipts.__setitem__, 'confidence', (SCORE_BYTES,))
    publication = Future()
    publication._xta_dense_independent_publication_paths = (tmp_path / 'published.cvol',)
    prepared = SimpleNamespace(model_name=KEY[0], view_name=KEY[1],
        native_support_mm=None, final_view_volume_mm=None,
        parent_mask_support_mm=None, parent_bridge_support_mm=None,
        pending_component_layers=[publication], nrrd_layers=[],
        _dense_input_owner_ref=refs[0], _dense_input_nbytes=MASK_BYTES,
        _packed_native_source_retired=True, _dense_confidence_owner_ref=refs[1])

    def publication_retired(key, expected_lease):
        receipts['publication'] = (key, expected_lease)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert not ledger.retire_dense_for_publication(prepared, enabled=True,
            tiled=False, keep_temp=False, retired_callback=publication_retired)
        assert KEY in ledger.retired_for_publication
        owners[0] = None
        assert refs[0]() is None and refs[1]() is not None
        assert set(receipts) == {'mask'}
        _assert_charge(ledger, lease, MASK_BYTES + SCORE_BYTES)
        owners[1] = None
        assert refs[1]() is None
        assert set(receipts) == {'mask', 'confidence', 'publication'}
        assert not publication.done()
        # Source death only queues receipts. The control thread applies a chosen
        # ordering, including whole-parent settlement before either input refund.
        expected = MASK_BYTES + SCORE_BYTES
        for kind in order:
            if kind == 'publication':
                applied = ledger.settle_publication_retirement(*receipts[kind])
                assert applied is bool(expected)
                expected = 0
                assert not ledger.settle_publication_retirement(*receipts[kind])
            else:
                amount, = receipts[kind]
                applied = ledger.retire_input_bytes(KEY, lease, amount, token=kind)
                assert applied is bool(expected)
                if applied:
                    expected -= amount
                assert not ledger.retire_input_bytes(KEY, lease, amount, token=kind)
            _assert_charge(ledger, lease, expected)
        assert expected == 0 and not publication.done()
    finally:
        owners.clear()
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize('packed_bytes', [0, 64])
@pytest.mark.parametrize('order', [('mask', 'confidence'), ('confidence', 'mask')])
def test_old_input_and_publication_receipts_cannot_refund_replacement(packed_bytes, order):
    ledger, old = _ledger(packed_bytes)
    amounts = {'mask': MASK_BYTES - packed_bytes, 'confidence': SCORE_BYTES}
    for token in order:
        assert ledger.retire_input_bytes(KEY, old, amounts[token], token=token)
    if packed_bytes:
        assert ledger.complete(KEY, retain_for_dense_retirement=False)
    ledger.retired_for_publication.add(KEY)
    replacement = _DirectUnionBackingLease(KEY, 4096, phase='postprocess',
        memfd_owner_proofs=(('mask', 'replacement-mask'), ('confidence', 'replacement-score')))
    ledger.leases[KEY] = replacement
    ledger.postprocess_views.add(KEY)
    ledger.postprocess_bytes[KEY] = replacement.nbytes
    proofs = replacement.memfd_owner_proofs
    for token in order:
        assert not ledger.retire_input_bytes(KEY, old, amounts[token], token=token)
        _assert_charge(ledger, replacement, 4096)
    assert not ledger.settle_publication_retirement(KEY, old)
    _assert_charge(ledger, replacement, 4096)
    assert replacement.memfd_owner_proofs == proofs


def test_pipeline_input_callbacks_wake_without_mutating_ledger_before_drain():
    from XTA import pipeline
    from tests.test_terminal_component_refs import _function

    ledger, lease = _ledger()
    mask_events, score_events, wake = queue.SimpleQueue(), queue.SimpleQueue(), Event()
    namespace = dict(vars(pipeline), parent_mask_retired_events=mask_events,
        parent_confidence_retired_events=score_events,
        scheduler_state=SimpleNamespace(scheduler_wake=wake))
    source = Path(pipeline.__file__).read_text(encoding='utf-8')
    mask_callback = _function(source, '_publish_parent_mask_retired', namespace)
    score_callback = _function(source, '_publish_parent_confidence_retired', namespace)
    mask_callback(*KEY, MASK_BYTES, expected_lease=lease)
    assert wake.is_set()
    wake.clear()
    score_callback(*KEY, SCORE_BYTES, expected_lease=lease)
    assert wake.is_set()
    _assert_charge(ledger, lease, MASK_BYTES + SCORE_BYTES)
    for events, token, expected in ((mask_events, 'mask', SCORE_BYTES),
                                    (score_events, 'confidence', 0)):
        key, expected_lease, nbytes = events.get_nowait()
        assert key == KEY and expected_lease is lease
        assert ledger.retire_input_bytes(key, expected_lease, nbytes, token=token)
        _assert_charge(ledger, lease, expected)
