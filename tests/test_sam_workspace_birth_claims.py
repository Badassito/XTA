"""RAM workspace birth probes cannot mutate scheduler ownership or half-adopt groups."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading

import pytest

from XTA import sam_parent_staging as staging
from XTA.interpolation import _DirectUnionBackingLease
from tests.test_sam_parent_staging_adversarial import queue


@pytest.fixture
def birth_queue(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setattr(staging, 'workspace_anon_cap_bytes', lambda: 0)
    monkeypatch.setattr(staging, 'publication_ram_headroom',
                        lambda: staging.RAM_RESERVE_BYTES + 1000)
    stage, _writer, _prepare, leases = queue(tmp_path, cap=100)
    published = []
    monkeypatch.setattr(stage, '_publish_staging', lambda: published.append(threading.get_ident()))
    try:
        yield stage, leases, published
    finally:
        for key in tuple(stage.ram_first_owners):
            stage.release_ram_first(key)
        for key in tuple(leases.leases):
            if key in leases.inference_views:
                leases.handoff(key)
            leases.complete(key, retain_for_dense_retirement=False)
        stage.close()
        stage.finalize_cleanup()


def peer(leases, name='peer', amount=40, *, ram_backed=None):
    key = ('model', name)
    lease = _DirectUnionBackingLease(key, amount, ram_backed=ram_backed)
    leases.leases[key] = lease
    leases.inference_views.add(key)
    leases.inference_bytes[key] = amount
    return lease


def qualified(stage, key, amount, active=0, room=0):
    plan = stage.plan_ram_first(key, amount, active, room)
    return plan, stage.qualify_ram_first(plan)


def test_plan_and_receipt_are_immutable_detached_snapshots(birth_queue):
    stage, leases, published = birth_queue
    live = peer(leases)
    plan = stage.plan_ram_first(('model', 'new'), 20, 40, 10)
    assert not published and not stage.ram_first_owners
    assert plan['promises'][0] is not live
    with pytest.raises((TypeError, AttributeError)):
        plan['required_bytes'] = 99
    with pytest.raises((TypeError, AttributeError)):
        plan['receipt']['requested_owner_bytes'] = 99
    with pytest.raises((TypeError, AttributeError, FrozenInstanceError)):
        plan['promises'][0].nbytes = 99
    live.nbytes = 80
    assert plan['promises'][0].nbytes == 40
    assert plan['required_bytes'] == 20
    assert plan['receipt']['requested_owner_bytes'] == 20
    assert not stage.ram_first_owners and not published


def test_background_qualification_never_reads_or_writes_scheduler_dicts(birth_queue, monkeypatch):
    stage, leases, published = birth_queue
    owner = threading.get_ident()
    live = peer(leases)
    plan = stage.plan_ram_first(('model', 'new'), 20, 40, 10)
    class OwnerDictionary(dict):
        def _check(self):
            assert threading.get_ident() == owner, 'Background qualifier touched scheduler ownership'
        def __iter__(self):
            self._check()
            return super().__iter__()
        def __setitem__(self, key, value):
            self._check()
            return super().__setitem__(key, value)
        def items(self):
            self._check()
            return super().items()
        def values(self):
            self._check()
            return super().values()
        def get(self, key, default=None):
            self._check()
            return super().get(key, default)
    leases.leases = OwnerDictionary(leases.leases)
    stage.ram_first_owners = OwnerDictionary(stage.ram_first_owners)
    probes = []
    def probe(promises, _headroom):
        probes.append(threading.get_ident())
        assert promises[0] is not live and promises[0].nbytes == 40
        return staging.RAM_RESERVE_BYTES + 1000, 0, {}
    monkeypatch.setattr(staging, 'memfd_ram_headroom', probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(stage.qualify_ram_first, plan).result(timeout=3)
    assert result[0] and probes != [owner]
    assert not published and not stage.ram_first_owners
    assert stage.ram_first_grants == stage.ram_first_denials == 0
    assert result[1] is not plan['receipt']
    assert result[1]['total_checked_bytes'] == 70
    assert stage.adopt_ram_first(((plan, result),)) == (True,)
    assert published == [owner]


def test_valid_batch_adopts_exact_credit_once_and_retires_each_claim(birth_queue):
    stage, leases, published = birth_queue
    live = peer(leases, amount=10)
    before = dict(leases.inference_bytes)
    first = qualified(stage, ('model', 'one'), 20, 10, 20)
    second = qualified(stage, ('model', 'two'), 30, 30, 20)
    assert stage.adopt_ram_first((first, second)) == (True, True)
    assert stage.ram_first_owners == {('model', 'one'): 20, ('model', 'two'): 30}
    assert stage.ram_first_bytes == 50 and stage.ram_first_grants == 2
    assert live.owner_count == 1 and leases.inference_bytes == before
    assert len(published) == 1
    assert stage.release_ram_first(('model', 'one'))
    assert stage.ram_first_bytes == 30 and stage.ram_first_releases == 1
    assert not stage.release_ram_first(('model', 'one'))
    assert stage.release_ram_first(('model', 'two'))
    assert stage.ram_first_bytes == 0 and stage.ram_first_releases == 2


def test_duplicate_keys_are_rejected_before_any_batch_grant(birth_queue):
    stage, _leases, published = birth_queue
    claim = qualified(stage, ('model', 'one'), 20)
    with pytest.raises((RuntimeError, ValueError), match='twice|duplicate'):
        stage.adopt_ram_first((claim, claim))
    assert not stage.ram_first_owners and stage.ram_first_grants == 0
    assert not published


def test_aggregate_batch_capacity_decisions_commit_exact_total(birth_queue):
    stage, _leases, _published = birth_queue
    claims = tuple(qualified(stage, ('model', name), 60) for name in ('one', 'two'))
    assert all(result[0] for _plan, result in claims)
    assert stage.adopt_ram_first(claims) == (True, False)
    assert stage.ram_first_owners == {('model', 'one'): 60}
    assert stage.ram_first_bytes == 60 and stage.ram_first_grants == 1
    assert stage.ram_first_denials == 1


def test_qualification_failure_keeps_denied_member_out_of_credit(birth_queue, monkeypatch):
    stage, _leases, _published = birth_queue
    first = qualified(stage, ('model', 'one'), 20)
    second_plan = stage.plan_ram_first(('model', 'two'), 20, 20, 10)
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES + 49)
    second = stage.qualify_ram_first(second_plan)
    assert not second[0] and second[1]['reason'] == 'physical_headroom'
    assert stage.adopt_ram_first((first, (second_plan, second))) == (True, False)
    assert stage.ram_first_owners == {('model', 'one'): 20}
    assert stage.ram_first_grants == stage.ram_first_denials == 1


def test_invalid_later_result_cannot_commit_an_earlier_member(birth_queue):
    stage, _leases, published = birth_queue
    first = qualified(stage, ('model', 'one'), 20)
    second = stage.plan_ram_first(('model', 'two'), 20, 20)
    with pytest.raises((TypeError, ValueError, IndexError)):
        stage.adopt_ram_first((first, (second, (True,))))
    assert not stage.ram_first_owners and stage.ram_first_grants == 0
    assert not published


def test_invalid_later_denial_receipt_is_validated_before_any_mutation(birth_queue):
    stage, _leases, published = birth_queue
    first = qualified(stage, ('model', 'one'), 20)
    second = stage.plan_ram_first(('model', 'two'), 20, 20)
    with pytest.raises((TypeError, ValueError)):
        stage.adopt_ram_first((first, (second, (False, None))))
    assert not stage.ram_first_owners and stage.ram_first_grants == 0
    assert stage.ram_first_denials == 0 and not published


def test_qualification_exception_has_no_scheduler_side_effect(birth_queue, monkeypatch):
    stage, _leases, published = birth_queue
    plan = stage.plan_ram_first(('model', 'new'), 20, 0)
    def fail(*_args):
        raise RuntimeError('failed RAM qualification')
    monkeypatch.setattr(staging, 'memfd_ram_headroom', fail)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='failed RAM qualification'):
            executor.submit(stage.qualify_ram_first, plan).result(timeout=3)
    assert not stage.ram_first_owners and not published
    assert stage.ram_first_grants == stage.ram_first_denials == 0


def test_external_future_rights_charge_physical_ram_without_inflating_dense_cap(birth_queue, monkeypatch):
    stage, leases, _published = birth_queue
    peer(leases)
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES + 120)
    plan = stage.plan_ram_first(('model', 'new'), 20, 40, 10, external_future_bytes=50)
    admitted, receipt = stage.qualify_ram_first(plan)
    assert admitted
    assert receipt['external_future_ram_bytes'] == 50
    assert receipt['physical_required_bytes'] == staging.RAM_RESERVE_BYTES + 120
    assert receipt['future_ram_commitments_bytes'] == 90
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES + 119)
    denied, receipt = stage.qualify_ram_first(plan)
    assert not denied and receipt['reason'] == 'physical_headroom'


def test_resident_peer_credit_does_not_erase_external_future_rights(birth_queue, monkeypatch):
    stage, leases, _published = birth_queue
    peer(leases)
    monkeypatch.setattr(staging, 'memfd_ram_headroom',
        lambda promises, probe: (staging.RAM_RESERVE_BYTES + 41, 40, {}))
    plan = stage.plan_ram_first(('model', 'new'), 20, 40, 10, external_future_bytes=12)
    admitted, receipt = stage.qualify_ram_first(plan)
    assert not admitted and receipt['reason'] == 'physical_headroom'
    assert receipt['physical_required_bytes'] == staging.RAM_RESERVE_BYTES + 42
    assert receipt['proven_resident_ram_bytes'] == 40
    assert receipt['future_ram_commitments_bytes'] == 12


def test_negative_external_future_rights_are_rejected_before_snapshot(birth_queue):
    stage, _leases, published = birth_queue
    with pytest.raises(ValueError, match='positive owner bytes'):
        stage.plan_ram_first(('model', 'new'), 20, 0, external_future_bytes=-1)
    assert not stage.ram_first_owners and not published


@pytest.mark.parametrize('change', ('new_peer', 'unresident_growth', 'ram_conversion', 'proof', 'owner', 'phase', 'replace'))
def test_stale_peer_basis_refuses_adoption_without_credit_or_diagnostics(birth_queue, change):
    stage, leases, published = birth_queue
    live = peer(leases, ram_backed=False if change == 'ram_conversion' else None)
    claim = qualified(stage, ('model', 'new'), 20, leases.ram_commitment_bytes, 10)
    if change == 'new_peer':
        peer(leases, name='later', amount=10)
    elif change == 'unresident_growth':
        live.nbytes += 40
        leases.inference_bytes[live.key] = live.nbytes
    elif change == 'ram_conversion':
        live.ram_backed = None
    elif change == 'proof':
        live.memfd_owner_proofs = (('mask', 3, 4, 5, 40),)
    elif change == 'owner':
        live.owner_count = 0
    elif change == 'phase':
        live.phase = 'postprocess'
    else:
        leases.leases[live.key] = _DirectUnionBackingLease(live.key, live.nbytes)
    assert not stage.ram_first_plans_current((claim,))
    assert stage.adopt_ram_first((claim,)) is None
    assert not stage.ram_first_owners and not published
    assert stage.ram_first_grants == stage.ram_first_denials == 0
    if change == 'owner':
        live.owner_count = 1
    if change == 'phase':
        live.phase = 'inference'


def test_shrinking_dense_limit_invalidates_qualified_total_not_just_owner_bytes(birth_queue):
    stage, leases, published = birth_queue
    peer(leases)
    claim = qualified(stage, ('model', 'new'), 20, 40, 20)
    assert claim[1][0]
    stage._dense_limit = 50
    assert not stage.ram_first_plans_current((claim,))
    assert stage.adopt_ram_first((claim,)) is None
    assert not stage.ram_first_owners and not published


def test_changed_claim_ledger_invalidates_older_plan(birth_queue):
    stage, _leases, published = birth_queue
    old = qualified(stage, ('model', 'old'), 20)
    assert stage.claim_ram_first(('model', 'other'), 20, 0)
    count = len(published)
    assert stage.adopt_ram_first((old,)) is None
    assert stage.ram_first_owners == {('model', 'other'): 20}
    assert stage.ram_first_grants == 1 and len(published) == count


def test_stale_member_rejects_whole_adoption_before_a_current_peer_grants(birth_queue):
    stage, leases, published = birth_queue
    old = qualified(stage, ('model', 'old'), 20)
    peer(leases)
    current = qualified(stage, ('model', 'current'), 20, 40)
    assert stage.adopt_ram_first((current, old)) is None
    assert not stage.ram_first_owners and not published
    assert stage.ram_first_grants == stage.ram_first_denials == 0


def test_cancelled_adoption_grants_nothing_and_existing_claim_retires_once(birth_queue):
    stage, _leases, _published = birth_queue
    assert stage.claim_ram_first(('model', 'existing'), 10, 0)
    claims = (qualified(stage, ('model', 'one'), 20, 10),
              qualified(stage, ('model', 'two'), 20, 30))
    stage.stop.set()
    assert stage.adopt_ram_first(claims) == (False, False)
    assert stage.ram_first_owners == {('model', 'existing'): 10}
    assert stage.last_ram_denial['reason'] == 'cancelled'
    assert stage.ram_first_grants == 1
    assert stage.release_ram_first(('model', 'existing'))
    assert not stage.release_ram_first(('model', 'existing'))


@pytest.mark.parametrize('arguments', ((0, 0, 0), (20, -1, 0), (20, 0, -1)))
def test_invalid_plan_request_has_no_registry_side_effect(birth_queue, arguments):
    stage, _leases, published = birth_queue
    with pytest.raises(ValueError, match='positive owner bytes'):
        stage.plan_ram_first(('model', 'new'), *arguments)
    assert not stage.ram_first_owners and not published


def test_legacy_synchronous_claim_preserves_disk_exclusion_room_and_denial(birth_queue, monkeypatch):
    stage, leases, _published = birth_queue
    peer(leases, 'disk', 140, ram_backed=False)
    peer(leases, 'ram', 40)
    assert stage.claim_ram_first(('model', 'new'), 20, leases.ram_commitment_bytes, 20)
    with pytest.raises(RuntimeError, match='claimed twice'):
        stage.claim_ram_first(('model', 'new'), 20, 40, 20)
    monkeypatch.setattr(staging, 'publication_ram_headroom', lambda: staging.RAM_RESERVE_BYTES + 79)
    assert not stage.claim_ram_first(('model', 'denied'), 20, 40, 20)
    receipt = stage.last_ram_denial
    assert receipt['reason'] == 'physical_headroom'
    assert receipt['excluded_disk_logical_bytes'] == 140
    assert receipt['total_checked_bytes'] == 80
    assert stage.ram_first_owners == {('model', 'new'): 20}


def test_legacy_claim_with_changed_peer_during_probe_declines_without_retry_loop(birth_queue, monkeypatch):
    stage, leases, published = birth_queue
    live = peer(leases)
    probes = []
    def probe(promises, _headroom):
        probes.append(promises[0].nbytes)
        live.nbytes += 1
        leases.inference_bytes[live.key] = live.nbytes
        return staging.RAM_RESERVE_BYTES + 1000, 0, {}
    monkeypatch.setattr(staging, 'memfd_ram_headroom', probe)
    assert not stage.claim_ram_first(('model', 'new'), 20, 40, 10)
    assert probes == [40] and not stage.ram_first_owners
    assert stage.ram_first_denials == 1
    assert stage.last_ram_denial['reason'] == 'ram_promise_basis_changed'
    assert len(published) == 1
