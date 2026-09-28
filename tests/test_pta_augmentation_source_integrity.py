"""External PTA policies execute the bytes named by their SHA-256 digest."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock

import pytest

from XTA import pta_augmentation
from XTA.augmentation_policy import AugmentationDefinition


def test_gpu_loader_rejects_change_between_inspection_and_import(tmp_path: Path) -> None:
    policy = tmp_path / 'policy.py'
    original = b'def build_gpu_augmentation(*, device, batch_size):\n    return "original"\n'
    changed = b'def build_gpu_augmentation(*, device, batch_size):\n    return "changed"\n'
    policy.write_bytes(original)
    inspected = AugmentationDefinition(policy, hashlib.sha256(original).hexdigest(), 'build_gpu_augmentation')
    with mock.patch.object(pta_augmentation, 'inspect_augmentation_definition', return_value=inspected):
        policy.write_bytes(changed)
        with pytest.raises(RuntimeError, match='changed while loading'):
            pta_augmentation.load_gpu_augmentation_definition(str(policy))


def test_loader_ignores_timestamp_valid_stale_bytecode(tmp_path: Path) -> None:
    policy = tmp_path / 'policy.py'
    old = b'VALUE = "old"\n'
    new = b'VALUE = "new"\n'
    policy.write_bytes(old)
    spec = importlib.util.spec_from_file_location('_xta_stale_policy_probe', policy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # creates a timestamp-based pyc for the old bytes
    assert module.VALUE == 'old'
    before = policy.stat()
    policy.write_bytes(new)
    os.utime(policy, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert policy.stat().st_size == len(old)

    digest = hashlib.sha256(new).hexdigest()
    loaded = pta_augmentation._load_external_python_module(policy, digest)
    try:
        assert loaded.VALUE == 'new'
    finally:
        sys.modules.pop(loaded.__name__, None)
