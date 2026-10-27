"""Small CPU-only fixtures for kernel-backed SAM allocation headroom."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from XTA import sam_resources, workspace


V2_MOUNT = '29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
V1_MOUNT = '29 23 0:26 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n'


def proc_files(monkeypatch, *, v2=True, membership='/job/step', mount=None):
    base = '/sys/fs/cgroup' if v2 else '/sys/fs/cgroup/memory'
    files = {
        '/proc/self/cgroup': ('0::' if v2 else '5:memory:') + membership + '\n',
        '/proc/self/mountinfo': (V2_MOUNT if v2 else V1_MOUNT) if mount is None else mount,
    }

    def read(path, *args, **kwargs):
        value = files.get(path.as_posix(), FileNotFoundError(str(path)))
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(Path, 'read_text', read)
    return files, base


def finite(files, base, relative, limit, usage, *, v2=True, stat=None, hierarchical='1'):
    node = base + relative
    limit_name, usage_name = ('memory.max', 'memory.current') if v2 else (
        'memory.limit_in_bytes', 'memory.usage_in_bytes')
    files[node + '/' + limit_name] = str(limit)
    files[node + '/' + usage_name] = usage if isinstance(usage, Exception) else str(usage)
    if stat is not None:
        files[node + '/memory.stat'] = stat
    if not v2:
        files[node + '/memory.use_hierarchy'] = hierarchical


@pytest.mark.parametrize('v2', [True, False])
def test_nested_caps_keep_tightest_headroom_and_smallest_enforced_cap(monkeypatch, v2):
    files, base = proc_files(monkeypatch, v2=v2)
    finite(files, base, '/job/step', 400, 50, v2=v2)
    finite(files, base, '/job', 600, 500, v2=v2)
    assert workspace._cgroup_memory_budget() == (100, 400)
    assert workspace._cgroup_memory_headroom_bytes() == 100


@pytest.mark.parametrize('usage', ['0', '400', '401'])
def test_zero_usage_and_over_limit_are_not_missing(monkeypatch, usage):
    files, base = proc_files(monkeypatch)
    finite(files, base, '/job', 400, usage)
    assert workspace._cgroup_memory_budget() == (max(0, 400-int(usage)), 400)


def test_zero_limit_is_finite(monkeypatch):
    files, base = proc_files(monkeypatch)
    finite(files, base, '/job', 0, 0)
    assert workspace._cgroup_memory_budget() == (0, 0)


@pytest.mark.parametrize('usage', ['', '-1', 'max', 'broken', FileNotFoundError(), PermissionError()])
@pytest.mark.parametrize('bad_level', ['/job', '/job/step'])
def test_any_finite_ancestor_with_unknown_usage_grants_zero(monkeypatch, usage, bad_level):
    files, base = proc_files(monkeypatch)
    finite(files, base, '/job', 600, 100)
    finite(files, base, '/job/step', 400, 50)
    files[base + bad_level + '/memory.current'] = usage
    assert workspace._cgroup_memory_budget() == (0, None)


@pytest.mark.parametrize('limit', ['', '-1', 'broken', PermissionError()])
def test_invalid_limit_fails_closed(monkeypatch, limit):
    files, base = proc_files(monkeypatch)
    files[base + '/job/memory.max'] = limit
    assert workspace._cgroup_memory_budget() == (0, None)


@pytest.mark.parametrize('v2,unlimited', [(True, 'max'), (False, str(2**63-4096))])
def test_unlimited_and_absent_levels_supply_no_enforcement_proof(monkeypatch, v2, unlimited):
    files, base = proc_files(monkeypatch, v2=v2)
    finite(files, base, '/job', unlimited, 10, v2=v2)
    assert workspace._cgroup_memory_budget() == (None, None)


@pytest.mark.parametrize('mount', ['', V2_MOUNT.replace(' / /sys', ' /other /sys'),
    V2_MOUNT.replace('cgroup2', 'tmpfs'), V2_MOUNT + V2_MOUNT,
    V2_MOUNT + '30 29 0:30 / /sys/fs/cgroup/job rw - tmpfs tmpfs rw\n',
    V2_MOUNT + '30 29 0:30 / /sys/fs/cgroup/job/memory.max rw - tmpfs tmpfs rw\n'])
def test_unverified_mount_preserves_clamp_but_not_cap_proof(monkeypatch, mount):
    files, base = proc_files(monkeypatch, mount=mount)
    finite(files, base, '/job', 600, 100)
    assert workspace._cgroup_memory_budget() == (500, None)


@pytest.mark.parametrize('membership', ['relative', '/job/../other', '/job/./step', '/job\\step'])
def test_unsafe_membership_cannot_establish_scope(monkeypatch, membership):
    files, base = proc_files(monkeypatch, membership=membership)
    finite(files, base, '/job', 600, 100)
    assert workspace._cgroup_memory_budget() == (None, None)


@pytest.mark.parametrize('hierarchical', ['0', '', '-1', 'broken'])
def test_v1_nonhierarchical_limit_keeps_rss_fallback(monkeypatch, hierarchical):
    files, base = proc_files(monkeypatch, v2=False)
    finite(files, base, '/job', 600, 100, v2=False, hierarchical=hierarchical)
    assert workspace._cgroup_memory_budget() == (500, None)


@pytest.mark.parametrize('v2', [True, False])
@pytest.mark.parametrize('stat,expected', [
    ('inactive_file 200\nfile_dirty 30\nfile_writeback 20\n', 650),
    ('inactive_file 900\nfile_dirty 0\nfile_writeback 0\n', 1000),
    ('inactive_file 200\nfile_dirty 300\nfile_writeback 20\n', 500),
    ('inactive_file 200\nfile_dirty -1\nfile_writeback 20\n', 500),
    ('inactive_file 200\nfile_dirty 0\n', 500),
    ('inactive_file 200\nfile_dirty 0\nfile_writeback broken\n', 500),
    ('inactive_file 200\nfile_dirty 0\nfile_dirty 0\nfile_writeback 0\n', 500),
])
def test_reclaim_requires_complete_valid_stats_and_never_exceeds_usage(monkeypatch, v2, stat, expected):
    files, base = proc_files(monkeypatch, v2=v2)
    if not v2:
        stat = stat.replace('inactive_file', 'total_inactive_file').replace(
            'file_dirty', 'total_dirty').replace('file_writeback', 'total_writeback')
    finite(files, base, '/job', 1000, 500, v2=v2, stat=stat)
    assert workspace._cgroup_memory_budget() == (expected, 1000)


def physical_fixture(monkeypatch, budget, *, resident=900, physical=900, anon=900):
    # Avoid loading runtime/accelerator dependencies in these accounting tests.
    import psutil
    mib = 1024**2
    monkeypatch.setitem(sys.modules, 'XTA.publication_memory', SimpleNamespace(
        publication_ram_headroom=lambda: physical*mib))
    monkeypatch.setattr(workspace, 'available_anon_work_bytes', lambda: anon*mib)
    monkeypatch.setattr(workspace, '_cgroup_memory_budget', lambda: tuple(
        None if item is None else item*mib for item in budget))
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '600')
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    calls = []

    def process():
        calls.append('RSS')
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=resident*mib),
                               children=lambda recursive: [])

    monkeypatch.setattr(psutil, 'Process', process)
    return calls, mib


@pytest.mark.parametrize('cap', [600, 500, 0])
def test_enforcing_cgroup_avoids_shared_mapping_rss_double_count(monkeypatch, cap):
    calls, mib = physical_fixture(monkeypatch, (min(400, cap), cap))
    assert sam_resources.physical_sam_headroom() == min(400, cap)*mib
    assert calls == []


@pytest.mark.parametrize('budget,expected', [((400, 601), 300), ((400, None), 300),
    ((None, None), 300), ((0, None), 0), ((200, 1000), 200)])
def test_unknown_or_larger_cgroup_keeps_slurm_rss_fallback(monkeypatch, budget, expected):
    calls, mib = physical_fixture(monkeypatch, budget, resident=300)
    assert sam_resources.physical_sam_headroom() == expected*mib
    assert calls == ['RSS']


def test_verified_cap_preserves_physical_and_anon_clamps(monkeypatch):
    calls, mib = physical_fixture(monkeypatch, (400, 600), physical=300, anon=200)
    assert sam_resources.physical_sam_headroom() == 200*mib
    assert calls == []


def test_smallest_slurm_cap_requires_independent_enforcement(monkeypatch):
    calls, mib = physical_fixture(monkeypatch, (400, 600), resident=100)
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '50')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '10')
    assert sam_resources.physical_sam_headroom() == 400*mib
    assert calls == ['RSS']


def test_available_anon_api_still_applies_kernel_headroom(monkeypatch):
    files, base = proc_files(monkeypatch)
    finite(files, base, '/job', 600, 100)
    monkeypatch.setattr(workspace, '_read_meminfo_bytes', lambda: {
        'MemAvailable': 900, 'SwapFree': 1000})
    assert workspace.available_anon_work_bytes() == 500


@pytest.mark.parametrize('v2', [True, False])
def test_publication_shares_one_meminfo_but_refreshes_all_cgroup_limits(monkeypatch, v2):
    from XTA import publication_memory

    files, base = proc_files(monkeypatch, v2=v2)
    finite(files, base, '/job/step', 400, 50, v2=v2)
    finite(files, base, '/job', 600, 500, v2=v2)
    snapshots = []
    mem_available = 900

    def read_meminfo():
        snapshot = {'MemAvailable': mem_available, 'SwapFree': 10000}
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(workspace, '_read_meminfo_bytes', read_meminfo)
    monkeypatch.setattr(publication_memory, '_read_meminfo_bytes', read_meminfo)
    assert publication_memory.publication_ram_headroom() == 100
    finite(files, base, '/job', 1000, 500, v2=v2)
    assert publication_memory.publication_ram_headroom() == 350
    finite(files, base, '/job/step', 100, 75, v2=v2)
    assert publication_memory.publication_ram_headroom() == 25
    mem_available = 10
    assert publication_memory.publication_ram_headroom() == 10
    assert len(snapshots) == 4  # One fresh physical snapshot for every call.
    assert len({id(snapshot) for snapshot in snapshots}) == 4


def test_shared_meminfo_keeps_swap_for_anon_but_not_physical_publication(monkeypatch):
    from XTA import publication_memory

    proc_files(monkeypatch)
    snapshot = {'MemAvailable': 20, 'SwapFree': 1000}
    monkeypatch.setattr(workspace, '_read_meminfo_bytes', lambda: snapshot)
    monkeypatch.setattr(publication_memory, '_read_meminfo_bytes', lambda: snapshot)
    assert workspace.available_anon_work_bytes() == 1020
    assert workspace.available_anon_work_bytes(meminfo=snapshot) == 1020
    assert publication_memory.publication_ram_headroom() == 20


def test_shared_meminfo_does_not_hide_unreadable_finite_cgroup_usage(monkeypatch):
    from XTA import publication_memory

    files, base = proc_files(monkeypatch)
    finite(files, base, '/job', 600, PermissionError())
    snapshot = {'MemAvailable': 900, 'SwapFree': 1000}
    monkeypatch.setattr(publication_memory, '_read_meminfo_bytes', lambda: snapshot)
    assert publication_memory.publication_ram_headroom() == 0


@pytest.mark.parametrize('v2', [True, False])
def test_headroom_only_probe_preserves_value_without_mount_discovery(monkeypatch, v2):
    files, base = proc_files(monkeypatch, v2=v2)
    finite(files, base, '/job/step', 400, 50, v2=v2)
    finite(files, base, '/job', 600, 500, v2=v2)
    assert workspace._cgroup_memory_budget() == (100, 400)
    read = Path.read_text
    reads = []

    def counted_read(path, *args, **kwargs):
        reads.append(path.as_posix())
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', counted_read)
    assert workspace._cgroup_memory_budget(verify_enforcement=False) == (100, None)
    assert workspace._cgroup_memory_headroom_bytes() == 100
    assert '/proc/self/mountinfo' not in reads
