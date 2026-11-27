"""Route a memory observation once, before any dynamic counters are read."""
from unittest.mock import Mock

import pytest

from XTA import workspace


@pytest.mark.parametrize('platform,route', [('linux', '_native_memory_sample'),
    ('win32', '_portable_memory_sample'), ('darwin', '_portable_memory_sample')])
def test_sample_selects_one_backend_before_taking_an_observation(monkeypatch, platform, route):
    native = Mock(return_value=({'MemAvailable': 1000}, 0, None))
    portable = Mock(return_value=({'MemAvailable': 500}, 45, None))
    backends = {'_native_memory_sample': native, '_portable_memory_sample': portable}
    for name, backend in backends.items():
        monkeypatch.setattr(workspace, name, backend)
    with monkeypatch.context() as platform_context:
        platform_context.setattr(workspace.sys, 'platform', platform)
        assert workspace.available_anon_work_bytes() == (0 if platform == 'linux' else 45)
    assert backends[route].call_count == 1
    assert sum(backend.call_count for backend in backends.values()) == 1


def test_native_adapter_uses_one_ordered_transaction_and_never_rereads_after_failure(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from XTA import native_cpu

    events = []
    failed = False
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == workspace._CGROUP_MEMBERSHIP_PATH:
            events.append('membership')
            return '0::/job/step\n'
        if path in (workspace._MEMINFO_PATH, workspace._MOUNTINFO_PATH):
            pytest.fail(f'Duplicate Python read of {path}')
        return original_read(path, *args, **kwargs)

    def capture(meminfo, plan, mountinfo, _encoding, _digits):
        assert meminfo == workspace._MEMINFO_PATH
        assert mountinfo == workspace._MOUNTINFO_PATH
        events.append('meminfo')
        groups = plan() if callable(plan) else plan
        base = workspace._CGROUP_V2_BASE
        expected_nodes = tuple(map(str, (base / 'job' / 'step', base / 'job', base)))
        assert groups == ((True, False, expected_nodes),)
        events.append('cgroup')
        levels = ((b'100', b'10', True, None),)
        return b'MemAvailable: 50 kB\nSwapFree: 40 kB\n', ((None, levels),), failed

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(native_cpu, 'kernels', SimpleNamespace(sample_memory_files=capture))
    monkeypatch.setattr(workspace, '_portable_memory_sample',
                        lambda **_kwargs: pytest.fail('Native sample was retried through the portable reader'))
    with monkeypatch.context() as platform_context:
        platform_context.setattr(workspace.sys, 'platform', 'linux')
        assert workspace.available_anon_work_bytes() == 90
        failed = True  # A later ancestor failed after the native sample started.
        assert workspace.available_anon_work_bytes() == 0
    assert events == ['meminfo', 'membership', 'cgroup', 'meminfo', 'cgroup']


def test_cgroup_sample_cannot_adopt_a_mount_proof_published_during_capture(monkeypatch):
    base = workspace._CGROUP_V2_BASE
    pid = workspace.os.getpid()
    mount_cache = {}
    monkeypatch.setattr(workspace, '_CGROUP_MOUNT_VERIFIED', mount_cache)
    monkeypatch.setattr(workspace, '_cgroup_layout',
                        lambda: (((True, base, (str(base),)),), False))
    mount_requests = []

    def capture(meminfo, plan, mountinfo, _encoding, _digits):
        assert meminfo is None
        assert mountinfo == workspace._MOUNTINFO_PATH
        groups = plan() if callable(plan) else plan
        ((v2, need_mount, nodes),) = groups
        assert v2 and nodes == (str(base),)
        mount_requests.append(need_mount)
        if need_mount:
            # Another sample authenticated the mount while this sample's read failed.
            mount_cache[(pid, base, v2)] = True
        levels = ((b'100', b'70', True, None),)
        return None, ((None, levels),), False

    def enforcing_sample():
        return workspace._sample_with(capture, include_meminfo=False, include_cgroup=True,
                                      verify_enforcement=True)

    assert enforcing_sample() == ({}, 30, None)
    assert enforcing_sample() == ({}, 30, 100)
    assert mount_requests == [True, False]
