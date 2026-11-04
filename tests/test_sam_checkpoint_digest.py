"""Coordinator checkpoint-digest prefetch reuse is identity-bound and opt-in."""
import hashlib
import os
import pickle
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import sam_checkpoint_digest as digest_mod
from XTA import sam_tracker_runtime


def _checkpoint(tmp_path, payload=b'fake checkpoint bytes; no model' * 1000):
    path = tmp_path / 'sam3.1_multiplex.pt'
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_hash_receipt_matches_existing_file_hash(tmp_path):
    path, expected = _checkpoint(tmp_path)
    receipt = digest_mod.hash_checkpoint(path, block_bytes=4096)
    assert receipt['schema'] == digest_mod.RECEIPT_SCHEMA
    assert receipt['sha256'] == expected == sam_tracker_runtime._sha256(path)
    assert receipt['hashed_bytes'] == path.stat().st_size
    assert receipt['identity'] == digest_mod.checkpoint_stat_identity(path)
    assert digest_mod.validated_digest(receipt, path) == expected
    # Spawn-safe primitive payload for LtaWorkerInit.
    pickle.dumps(receipt)


def test_identity_change_rejects_reuse(tmp_path):
    path, _ = _checkpoint(tmp_path)
    receipt = digest_mod.hash_checkpoint(path)
    path.write_bytes(path.read_bytes() + b'!')
    assert digest_mod.validated_digest(receipt, path) is None
    stat = path.stat()
    receipt = digest_mod.hash_checkpoint(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert digest_mod.validated_digest(receipt, path) is None
    assert digest_mod.validated_digest(receipt, tmp_path / 'missing.pt') is None


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(schema='other'),
    lambda r: r.update(sha256='0' * 63),
    lambda r: r.update(sha256='Z' * 64),
    lambda r: r['identity'].pop('inode'),
    lambda r: r['identity'].update(size_bytes=float(r['identity']['size_bytes'])),
    lambda r: r['identity'].update(extra=1),
])
def test_malformed_receipt_rejected(tmp_path, mutate):
    path, _ = _checkpoint(tmp_path)
    receipt = digest_mod.hash_checkpoint(path)
    receipt['identity'] = dict(receipt['identity'])
    mutate(receipt)
    assert digest_mod.validated_digest(receipt, path) is None
    assert digest_mod.validated_digest(None, path) is None


def test_cancel_and_unstable_identity_publish_no_receipt(tmp_path):
    path, _ = _checkpoint(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    assert digest_mod.hash_checkpoint(path, cancel_event=cancelled) is None
    real = digest_mod.checkpoint_stat_identity
    calls = []

    def changing(value):
        calls.append(value)
        identity = real(value)
        return identity if len(calls) == 1 else dict(identity, mtime_ns=identity['mtime_ns'] + 1)

    with mock.patch.object(digest_mod, 'checkpoint_stat_identity', side_effect=changing):
        assert digest_mod.hash_checkpoint(path) is None


def test_prefetch_is_opt_in_and_lookup_never_blocks(tmp_path, monkeypatch):
    path, expected = _checkpoint(tmp_path)
    model = tmp_path
    monkeypatch.delenv(digest_mod.ENV_FLAG, raising=False)
    assert digest_mod.start_checkpoint_digest_prefetch(model, path) is None
    assert digest_mod.checkpoint_digest_receipt(model) is None
    monkeypatch.setenv(digest_mod.ENV_FLAG, '1')
    gate = threading.Event()
    real_hash = digest_mod.hash_checkpoint

    def gated(*args, **kwargs):
        gate.wait(10)
        return real_hash(*args, **kwargs)

    try:
        with mock.patch.object(digest_mod, 'hash_checkpoint', side_effect=gated):
            prefetch = digest_mod.start_checkpoint_digest_prefetch(model, path)
            assert digest_mod.start_checkpoint_digest_prefetch(model, path) is prefetch
            assert digest_mod.checkpoint_digest_receipt(model) is None  # still hashing
            gate.set()
            assert prefetch.receipt(timeout=10)['sha256'] == expected
        assert digest_mod.checkpoint_digest_receipt(model)['sha256'] == expected
    finally:
        assert digest_mod.cancel_checkpoint_digest_prefetch(model)
    assert digest_mod.checkpoint_digest_receipt(model) is None


def test_failed_prefetch_leaves_workers_hashing(tmp_path, monkeypatch):
    monkeypatch.setenv(digest_mod.ENV_FLAG, '1')
    prefetch = digest_mod.start_checkpoint_digest_prefetch(tmp_path / 'm', tmp_path / 'missing.pt')
    try:
        assert prefetch.receipt(timeout=10) is None and 'FileNotFoundError' in prefetch.error
        assert digest_mod.checkpoint_digest_receipt(tmp_path / 'm') is None
    finally:
        digest_mod.cancel_checkpoint_digest_prefetch(tmp_path / 'm')


def _tracker(tmp_path, model_path):
    return sam_tracker_runtime.SamInterpolationTracker(
        model_path=model_path, device_ids=(0,), artifact_root=tmp_path / 'runtime')


def test_worker_init_forwards_only_a_completed_receipt(tmp_path, monkeypatch):
    path, expected = _checkpoint(tmp_path)
    tracker = _tracker(tmp_path, tmp_path)
    assert 'checkpoint_digest' not in tracker.worker_init().adapter_config
    monkeypatch.setenv(digest_mod.ENV_FLAG, '1')
    try:
        assert digest_mod.start_checkpoint_digest_prefetch(tmp_path, path).receipt(timeout=10)
        init = tracker.worker_init()
        assert init.adapter_config['checkpoint_digest']['sha256'] == expected
        pickle.dumps(init)
    finally:
        digest_mod.cancel_checkpoint_digest_prefetch(tmp_path)


def _build(tmp_path, checkpoint, config_extra):
    bundle = SimpleNamespace(checkpoint_path=checkpoint, checkpoint_identity_sha256='identity',
                             model_version='sam3.1', bpe_path=None)
    cuda = SimpleNamespace(synchronize=mock.Mock(), empty_cache=mock.Mock())
    context = SimpleNamespace(
        predictor=SimpleNamespace(model=SimpleNamespace(tracker=SimpleNamespace(fill_hole_area=0)),
                                  shutdown=mock.Mock()),
        sam_runtime={}, profile={}, torch_module=SimpleNamespace(cuda=cuda), restore_sdpa=None)
    config = dict(model_path='unused', feature_cache_bytes=0, **config_extra)
    with mock.patch('XTA.lta_worker_adapter.build_worker_predictor', return_value=context), \
            mock.patch('XTA.lta_sam.resolve_local_sam_bundle', return_value=bundle):
        return sam_tracker_runtime.build_interpolation_predictor(config)


def test_worker_reuses_verified_digest_without_rehashing(tmp_path):
    path, expected = _checkpoint(tmp_path)
    receipt = digest_mod.hash_checkpoint(path)
    with mock.patch.object(sam_tracker_runtime, '_sha256', side_effect=AssertionError('rehash')):
        context = _build(tmp_path, path, dict(checkpoint_digest=receipt))
    assert context.sam_model['checkpoint_sha256'] == expected
    assert context.sam_runtime['checkpoint_digest_basis'] == 'coordinator_prefetch_identity_verified'


def test_worker_rehashes_changed_checkpoint_and_default_is_unchanged(tmp_path):
    path, _ = _checkpoint(tmp_path)
    receipt = digest_mod.hash_checkpoint(path)
    path.write_bytes(b'replaced checkpoint')
    changed = _build(tmp_path, path, dict(checkpoint_digest=receipt))
    assert changed.sam_model['checkpoint_sha256'] == hashlib.sha256(b'replaced checkpoint').hexdigest()
    assert changed.sam_runtime['checkpoint_digest_basis'] == 'worker_hash'
    default = _build(tmp_path, path, {})
    assert default.sam_model == changed.sam_model
    assert 'checkpoint_digest_basis' not in default.sam_runtime


def _context(tmp_path, model_path):
    import numpy as np
    from XTA import sam_integration
    return sam_integration.SamInterpolationContext(
        model_path=model_path, device_ids=('cuda:0',),
        temp_dir=tmp_path / 'temporary', evidence_root=tmp_path / 'retained',
        source_volume=np.zeros((3, 4, 5), dtype=np.uint8),
        source_identity='source-identity', detector_identity='detector-identity')


def test_context_prefetch_is_explicit_opt_in_and_cancelled_on_close(tmp_path, monkeypatch):
    path, expected = _checkpoint(tmp_path)
    monkeypatch.delenv(digest_mod.ENV_FLAG, raising=False)
    context = _context(tmp_path, tmp_path)
    assert context.start_checkpoint_digest_prefetch() is None
    context.close()
    monkeypatch.setenv(digest_mod.ENV_FLAG, '1')
    context = _context(tmp_path, tmp_path)
    try:
        # Construction alone starts nothing; the pipeline starts it after detector readiness.
        assert digest_mod.checkpoint_digest_receipt(tmp_path) is None
        prefetch = context.start_checkpoint_digest_prefetch()
        assert context.start_checkpoint_digest_prefetch() is prefetch
        assert prefetch.receipt(timeout=10)['sha256'] == expected
    finally:
        context.close()
    assert digest_mod.checkpoint_digest_receipt(tmp_path) is None
    # Invalid bundles leave startup to report the error and workers to hash.
    broken = _context(tmp_path, tmp_path / 'missing-bundle')
    try:
        assert broken.start_checkpoint_digest_prefetch() is None
    finally:
        broken.close()
