"""Fresh RAM sampling policy and its public admission boundaries."""
from types import SimpleNamespace
import threading

import pytest

from XTA import publication_memory, runtime, sam_resources, workspace


def test_meminfo_and_cgroup_callers_request_only_their_counters(monkeypatch):
    calls = []

    def sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
        calls.append((include_meminfo, include_cgroup, verify_enforcement))
        return {'SwapTotal': 8192}, 100, 200 if verify_enforcement else None

    monkeypatch.setattr(workspace, '_memory_sample', sample)
    assert workspace._read_meminfo_bytes() == {'SwapTotal': 8192}
    assert workspace._cgroup_memory_headroom_bytes() == 100
    assert workspace._cgroup_memory_budget() == (100, 200)
    assert calls == [
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ]


def test_anonymous_headroom_uses_one_fresh_sample_per_call(monkeypatch):
    calls = []
    cgroup_limit = None
    physical_available = 64

    def sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
        calls.append((include_meminfo, include_cgroup, verify_enforcement, threading.get_ident()))
        assert include_meminfo and include_cgroup
        return {'MemAvailable': physical_available, 'SwapFree': 16}, cgroup_limit, None

    monkeypatch.setattr(workspace, '_memory_sample', sample)
    assert workspace.available_anon_work_bytes() == 80
    cgroup_limit = 50
    assert workspace.available_anon_work_bytes() == 50
    physical_available = 10
    assert workspace.available_anon_work_bytes() == 26
    cgroup_limit = 0
    assert workspace.available_anon_work_bytes() == 0
    assert calls == [(True, True, False, threading.get_ident())] * 4


def test_supplied_meminfo_refreshes_cgroup_without_rereading_the_node(monkeypatch):
    calls = []
    headrooms = iter((None, 20, 0))

    def sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
        calls.append((include_meminfo, include_cgroup, verify_enforcement))
        return {}, next(headrooms), None

    monkeypatch.setattr(workspace, '_memory_sample', sample)
    snapshot = {'MemAvailable': 10, 'SwapFree': 100}
    assert [workspace.available_anon_work_bytes(meminfo=snapshot) for _ in range(3)] == [110, 20, 0]
    assert calls == [(False, True, False)] * 3
    assert snapshot == {'MemAvailable': 10, 'SwapFree': 100}


@pytest.mark.parametrize('slurm', [False, True])
def test_sam_preserves_separate_live_samples_and_enforcement_check(monkeypatch, slurm):
    calls = []
    observed = iter((
        ({'MemAvailable': 90, 'SwapFree': 60}, None, None),
        ({'MemAvailable': 80, 'SwapFree': 60}, 40, None),
        ({}, 15, 1024**2),
    ))

    def sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
        calls.append((include_meminfo, include_cgroup, verify_enforcement))
        return next(observed)

    monkeypatch.setattr(publication_memory, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(workspace, '_memory_sample', sample)
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    if slurm:
        monkeypatch.setenv('SLURM_MEM_PER_NODE', '1')
    else:
        monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    assert sam_resources.physical_sam_headroom() == (15 if slurm else 40)
    expected = [(True, True, False), (True, True, False)]
    if slurm:
        expected.append((False, True, True))
    assert calls == expected


def test_memfd_credit_samples_owned_blocks_around_fresh_publication_headroom(tmp_path, monkeypatch):
    from tests.test_memfd_resident_admission import controlled_owner, linux_probe, promises

    with controlled_owner(tmp_path) as (array, _key, descriptor, info):
        proof = runtime.capture_memfd_owner_proofs({'mask': array})
        assert len(proof) == 1
        read_meminfo = runtime._read_meminfo_bytes
        linux_probe(monkeypatch, descriptor, info, blocks=1)
        monkeypatch.setattr(runtime, '_read_meminfo_bytes', read_meminfo)
        monkeypatch.setattr(publication_memory, 'os', SimpleNamespace(name='posix'))
        observed = []
        stat = runtime.os.fstat

        def sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
            assert include_meminfo and not verify_enforcement
            observed.append('headroom' if include_cgroup else 'swap')
            return {'MemAvailable': 4096, 'SwapTotal': 0}, None, None

        def owner_stat(fd):
            result = stat(fd)
            if (result.st_dev, result.st_ino) == (info.st_dev, info.st_ino):
                observed.append('blocks')
            return result

        monkeypatch.setattr(workspace, '_memory_sample', sample)
        monkeypatch.setattr(runtime.os, 'fstat', owner_stat)
        headroom, resident, receipt = runtime.memfd_ram_headroom(
            promises((1024, proof)), publication_memory.publication_ram_headroom)
        assert (headroom, resident) == (4096, 512)
        assert receipt['validated_memfd_owner_count'] == 1
        assert observed == ['swap', 'blocks', 'headroom', 'swap', 'blocks']
