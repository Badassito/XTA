"""Release checks may share work only within one immutable verification pass."""
from __future__ import annotations

import copy
import json
from unittest import mock

import pytest

from tools import verify_package_inventory as inventory


def test_real_successor_contract_is_checked_once_per_context():
    manifest = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    predecessor = inventory.reviewed_v23_0_3_release_contract
    latest = inventory.reviewed_v23_0_4_release_contract
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        with mock.patch.object(inventory, 'reviewed_v23_0_3_release_contract', wraps=predecessor) as prior_check, \
             mock.patch.object(inventory, 'reviewed_v23_0_4_release_contract', wraps=latest) as latest_check:
            inventory._without_reviewed_v23_0_3_release(manifest)
            inventory._without_reviewed_v23_0_3_release(manifest)
            assert prior_check.call_count == 1
            assert latest_check.call_count == 1
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)


def test_latest_successor_tampering_is_rechecked_in_a_new_pass():
    manifest = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    latest = ('v24_0_0_release_review' if 'v24_0_0_release_review' in manifest
              else 'v23_0_5_release_review' if 'v23_0_5_release_review' in manifest
              else 'v23_0_4_release_review')
    without_latest = (inventory._without_reviewed_v24_0_0_release
                      if latest == 'v24_0_0_release_review'
                      else inventory._without_reviewed_v23_0_5_release
                      if latest == 'v23_0_5_release_review'
                      else inventory._without_reviewed_v23_0_4_release)
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        without_latest(manifest)
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)

    altered = copy.deepcopy(manifest)
    altered[latest]['feature'] = 'tampered'
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        with pytest.raises(RuntimeError, match='review digest mismatch'):
            without_latest(altered)
        with pytest.raises(RuntimeError, match='review digest mismatch'):
            without_latest(altered)
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)


def test_successful_review_is_reused_only_for_the_same_full_context():
    calls = []

    def authenticate(manifest, v21):
        calls.append((manifest['ancestor']['value'], v21['value']))
        return manifest['candidate']

    manifest = {'v21_review': {'value': 1}, 'ancestor': {'value': 2},
                'candidate': {'sha256': 'reviewed'}}
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        inventory._authenticate_successor_once(manifest, 'candidate', authenticate)
        inventory._authenticate_successor_once(dict(manifest), 'candidate', authenticate)
        assert calls == [(2, 1)]
        changed_predecessor = {**manifest, 'ancestor': {'value': 3}}
        inventory._authenticate_successor_once(changed_predecessor, 'candidate', authenticate)
        assert calls == [(2, 1), (3, 1)]
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)


def test_new_pass_rejects_tampering_and_failed_checks_are_never_cached():
    calls = []

    def authenticate(manifest, _v21):
        calls.append(manifest['candidate']['sha256'])
        if manifest['candidate']['sha256'] != 'reviewed':
            raise RuntimeError('review digest mismatch')
        return manifest['candidate']

    manifest = {'v21_review': {}, 'ancestor': {}, 'candidate': {'sha256': 'reviewed'}}
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        inventory._authenticate_successor_once(manifest, 'candidate', authenticate)
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)
    manifest['candidate']['sha256'] = 'tampered'
    token = inventory._ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        with pytest.raises(RuntimeError, match='digest mismatch'):
            inventory._authenticate_successor_once(manifest, 'candidate', authenticate)
        with pytest.raises(RuntimeError, match='digest mismatch'):
            inventory._authenticate_successor_once(manifest, 'candidate', authenticate)
    finally:
        inventory._ACTIVE_REVIEW_AUTH_CACHE.reset(token)
    assert calls == ['reviewed', 'tampered', 'tampered']


def test_main_clears_both_caches_when_verification_fails():
    with mock.patch.object(inventory, '_verify_main', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError, match='failed'):
            inventory.main()
    assert inventory._ACTIVE_REVIEW_AUTH_CACHE.get() is None
    assert inventory._ACTIVE_DIGEST_CACHE.get() is None
