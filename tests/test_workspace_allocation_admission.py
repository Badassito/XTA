"""Workspace diagnostics never repeat or replace fresh memory admission."""
from unittest import mock

import numpy as np
import pytest

from XTA import runtime


@pytest.fixture(autouse=True)
def no_numa_or_configured_cap(monkeypatch):
    monkeypatch.setattr(runtime, 'numa_interleave_memory', lambda *_a, **_k: False)
    monkeypatch.setattr(runtime, 'workspace_anon_cap_bytes', lambda: 0)


def test_forced_disk_and_reuse_do_not_probe_ram_for_logging(tmp_path, monkeypatch):
    probe = mock.Mock(side_effect=AssertionError('disk diagnostics queried RAM'))
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', probe)
    path = tmp_path / 'disk.dat'
    first = runtime.allocate_workspace_array((2, 4), np.uint8, path, 'disk',
        prefer_memory=False, prefer_memfd=False)
    first[:] = 7
    runtime.close_memmap_array(first)
    second = runtime.allocate_workspace_array((2, 4), np.uint8, path, 'reuse',
        prefer_memory=False, prefer_memfd=False, reuse_existing=True)
    try:
        np.testing.assert_array_equal(second, np.full((2, 4), 7, np.uint8))
        probe.assert_not_called()
    finally:
        runtime.close_memmap_array(second)


@pytest.mark.parametrize('headroom,cap,in_memory', [(18, 0, True), (17, 0, False), (100, 7, False)])
def test_one_fresh_probe_preserves_reserve_and_cap(tmp_path, monkeypatch, headroom, cap, in_memory):
    probe = mock.Mock(return_value=headroom)
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', probe)
    monkeypatch.setattr(runtime, 'workspace_anon_cap_bytes', lambda: cap)
    result = runtime.allocate_workspace_array((2, 4), np.uint8, tmp_path / 'map.dat', 'checked',
        reserve_bytes=10)
    try:
        assert isinstance(result, np.memmap) == (not in_memory)
        np.testing.assert_array_equal(result, np.zeros((2, 4), np.uint8))
        probe.assert_called_once_with()
    finally:
        runtime.close_memmap_array(result)


def test_memfd_attempt_has_one_probe_and_a_later_allocation_rechecks(tmp_path, monkeypatch):
    probe = mock.Mock(side_effect=[18, 17])
    expected = np.empty((2, 4), np.uint8)
    allocator = mock.Mock(return_value=expected)
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', probe)
    monkeypatch.setattr(runtime, '_allocate_memfd_workspace_array', allocator)
    first = runtime.allocate_workspace_array((2, 4), np.uint8, tmp_path / 'first.dat', 'memfd',
        prefer_memory=False, prefer_memfd=True, reserve_bytes=10)
    second = runtime.allocate_workspace_array((2, 4), np.uint8, tmp_path / 'second.dat', 'fallback',
        prefer_memory=False, prefer_memfd=True, reserve_bytes=10)
    try:
        assert first is expected
        assert isinstance(second, np.memmap)
        assert probe.call_count == 2
        allocator.assert_called_once()
    finally:
        runtime.close_memmap_array(second)


def test_failed_anonymous_allocation_rechecks_before_memfd(tmp_path, monkeypatch):
    probe = mock.Mock(side_effect=[100, 0])
    allocator = mock.Mock(side_effect=AssertionError('stale RAM admission reused'))
    monkeypatch.setattr(runtime, 'available_anon_work_bytes', probe)
    monkeypatch.setattr(runtime, '_allocate_memfd_workspace_array', allocator)
    monkeypatch.setattr(runtime.np, 'zeros', mock.Mock(side_effect=MemoryError('pressure')))
    result = runtime.allocate_workspace_array((2, 4), np.uint8, tmp_path / 'fallback.dat', 'retry',
        prefer_memory=True, prefer_memfd=True, reserve_bytes=10)
    try:
        assert isinstance(result, np.memmap)
        assert probe.call_count == 2
        allocator.assert_not_called()
    finally:
        runtime.close_memmap_array(result)
