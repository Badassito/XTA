"""Independent procfs/controller oracles for SAM allocation headroom."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA import publication_memory, sam_resources, workspace

GIB = 1024**3
V2_NODE = '/sys/fs/cgroup/slurm/job_42/step_0'
V1_NODE = '/sys/fs/cgroup/memory/slurm/job_42/step_0'


def fake_procfs(monkeypatch, *, v2=True, limit=600*GIB, usage=200*GIB):
    node = V2_NODE if v2 else V1_NODE
    base = '/sys/fs/cgroup' if v2 else '/sys/fs/cgroup/memory'
    limit_name = 'memory.max' if v2 else 'memory.limit_in_bytes'
    usage_name = 'memory.current' if v2 else 'memory.usage_in_bytes'
    files = {
        '/proc/self/cgroup': ('0::' if v2 else '5:memory:')+'/slurm/job_42/step_0\n',
        '/proc/self/mountinfo': (f'32 20 0:28 / {base} rw,nosuid,nodev,noexec - '
                                 + ('cgroup2 cgroup rw\n' if v2 else 'cgroup cgroup rw,memory\n')),
        node+'/'+limit_name: str(limit),
        node+'/'+usage_name: str(usage),
        node+'/memory.stat': ('inactive_file 0\nfile_dirty 0\nfile_writeback 0\n' if v2 else
                              'total_inactive_file 0\ntotal_dirty 0\ntotal_writeback 0\n'),
        node+'/memory.use_hierarchy': '1',
    }
    original_read = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        name = path.as_posix()
        if name.startswith(('/proc/', '/sys/')):
            reads.append(name)
            if name not in files:
                raise FileNotFoundError(name)
            value = files[name]
            if isinstance(value, Exception):
                raise value
            return value
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    return files, node, reads


def slurm_and_host(monkeypatch, *, cap_gib=600, physical_gib=400, resident_gib=750):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', str(cap_gib*1024))
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    monkeypatch.setattr(publication_memory, 'publication_ram_headroom', lambda: physical_gib*GIB)
    monkeypatch.setattr(workspace, 'available_anon_work_bytes', lambda: physical_gib*GIB)
    calls = []

    def process():
        calls.append(True)
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=resident_gib*GIB//5),
            children=lambda recursive: [SimpleNamespace(
                memory_info=lambda: SimpleNamespace(rss=resident_gib*GIB//5)) for _ in range(4)])

    monkeypatch.setattr('psutil.Process', process)
    return calls


def test_verified_job_charge_avoids_shared_process_rss_phantom_zero(monkeypatch):
    fake_procfs(monkeypatch)
    calls = slurm_and_host(monkeypatch)
    assert workspace._cgroup_memory_budget() == (400*GIB, 600*GIB)
    assert sam_resources.physical_sam_headroom() == 400*GIB
    assert not calls


def test_parent_scope_headroom_and_smallest_enforced_cap_are_both_retained(monkeypatch):
    files, node, _ = fake_procfs(monkeypatch, limit=400*GIB, usage=100*GIB)
    parent = node.rsplit('/', 1)[0]
    files.update({parent+'/memory.max': str(600*GIB), parent+'/memory.current': str(550*GIB)})
    calls = slurm_and_host(monkeypatch, physical_gib=500)
    assert workspace._cgroup_memory_budget() == (50*GIB, 400*GIB)
    assert sam_resources.physical_sam_headroom() == 50*GIB
    assert not calls


@pytest.mark.parametrize('mount', ['missing', 'different_root', 'wrong_filesystem'])
def test_unverified_mount_cannot_remove_slurm_rss_fallback(monkeypatch, mount):
    files, _node, _ = fake_procfs(monkeypatch)
    if mount == 'missing':
        del files['/proc/self/mountinfo']
    elif mount == 'different_root':
        files['/proc/self/mountinfo'] = files['/proc/self/mountinfo'].replace(' / /sys/', ' /other-job /sys/')
    else:
        files['/proc/self/mountinfo'] = files['/proc/self/mountinfo'].replace(' - cgroup2 ', ' - ext4 ')
    calls = slurm_and_host(monkeypatch)
    headroom, cap = workspace._cgroup_memory_budget()
    assert cap is None and headroom == 400*GIB
    assert sam_resources.physical_sam_headroom() == 0 and calls


@pytest.mark.parametrize('membership', ['/slurm/../other', 'slurm/job_42', '/slurm\\job_42'])
def test_unsafe_membership_is_never_an_enforcement_proof(monkeypatch, membership):
    files, _node, reads = fake_procfs(monkeypatch)
    files['/proc/self/cgroup'] = '0::'+membership+'\n'
    assert workspace._cgroup_memory_budget()[1] is None
    assert not any('/../' in path or '\\' in path for path in reads)


@pytest.mark.parametrize('v2', [False, True])
@pytest.mark.parametrize('covering', ['membership', 'ancestor'])
def test_more_specific_mount_cannot_forge_current_group_enforcement(monkeypatch, v2, covering):
    files, node, _ = fake_procfs(monkeypatch, v2=v2)
    mountpoint = node if covering == 'membership' else node.rsplit('/', 1)[0]
    filesystem = 'cgroup2 cgroup rw' if v2 else 'cgroup cgroup rw,memory'
    files['/proc/self/mountinfo'] += (
        f'40 32 0:28 /slurm/other_job {mountpoint} rw - {filesystem}\n')
    calls = slurm_and_host(monkeypatch)
    assert workspace._cgroup_memory_budget()[1] is None
    assert sam_resources.physical_sam_headroom() == 0 and calls


@pytest.mark.parametrize('bad_usage', [None, '-1', 'invalid', 'max', PermissionError('unreadable')])
def test_known_finite_usage_failure_closes_credit_even_with_another_valid_ancestor(monkeypatch, bad_usage):
    files, node, _ = fake_procfs(monkeypatch)
    if bad_usage is None:
        del files[node+'/memory.current']
    else:
        files[node+'/memory.current'] = bad_usage
    parent = node.rsplit('/', 1)[0]
    files.update({parent+'/memory.max': str(800*GIB), parent+'/memory.current': str(100*GIB)})
    assert workspace._cgroup_memory_budget() == (0, None)
    assert workspace._cgroup_memory_headroom_bytes() == 0


@pytest.mark.parametrize('limit,usage,expected', [(600*GIB, 0, 600*GIB), (0, 0, 0), (600*GIB, 700*GIB, 0)])
def test_valid_zero_and_over_limit_values_are_not_unlimited(monkeypatch, limit, usage, expected):
    fake_procfs(monkeypatch, limit=limit, usage=usage)
    assert workspace._cgroup_memory_budget() == (expected, limit)


@pytest.mark.parametrize('stat', [
    f'inactive_file {100*GIB}\n',
    f'inactive_file {100*GIB}\nfile_dirty invalid\nfile_writeback 0\n',
    f'inactive_file {100*GIB}\nfile_dirty {-100*GIB}\nfile_writeback 0\n',
])
def test_incomplete_or_invalid_reclaim_stats_grant_no_reclamation_credit(monkeypatch, stat):
    files, node, _ = fake_procfs(monkeypatch, usage=300*GIB)
    files[node+'/memory.stat'] = stat
    assert workspace._cgroup_memory_budget() == (300*GIB, 600*GIB)


def test_clean_inactive_reclaim_and_usage_clamp_are_preserved(monkeypatch):
    files, node, _ = fake_procfs(monkeypatch, usage=300*GIB)
    files[node+'/memory.stat'] = f'inactive_file {100*GIB}\nfile_dirty {20*GIB}\nfile_writeback {10*GIB}\n'
    assert workspace._cgroup_memory_budget() == (370*GIB, 600*GIB)
    files[node+'/memory.stat'] = f'inactive_file {1000*GIB}\nfile_dirty 0\nfile_writeback 0\n'
    assert workspace._cgroup_memory_budget() == (600*GIB, 600*GIB)


@pytest.mark.parametrize('hierarchical', ['0', '1'])
def test_v1_parent_limit_is_proof_only_when_hierarchical(monkeypatch, hierarchical):
    files, node, _ = fake_procfs(monkeypatch, v2=False, limit=(1 << 63)-4096)
    parent = node.rsplit('/', 1)[0]
    files.update({parent+'/memory.limit_in_bytes': str(600*GIB),
                  parent+'/memory.usage_in_bytes': str(200*GIB),
                  parent+'/memory.use_hierarchy': hierarchical})
    _headroom, cap = workspace._cgroup_memory_budget()
    assert cap == (600*GIB if hierarchical == '1' else None)


def test_cgroup_limit_larger_than_slurm_does_not_replace_slurm_bound(monkeypatch):
    fake_procfs(monkeypatch, limit=700*GIB, usage=200*GIB)
    calls = slurm_and_host(monkeypatch, physical_gib=500, resident_gib=150)
    assert sam_resources.physical_sam_headroom() == 450*GIB and calls


def test_slurm_per_cpu_cap_and_invalid_declaration_remain_enforced(monkeypatch):
    fake_procfs(monkeypatch)
    calls = slurm_and_host(monkeypatch)
    monkeypatch.delenv('SLURM_MEM_PER_NODE')
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '6144')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '100')
    assert sam_resources.physical_sam_headroom() == 400*GIB and not calls
    monkeypatch.setenv('SLURM_MEM_PER_NODE', 'invalid')
    assert sam_resources.physical_sam_headroom() == 0 and not calls


def test_headroom_only_probe_never_reads_mounts_or_supplies_enforcement_proof(monkeypatch):
    _files, _node, reads = fake_procfs(monkeypatch)
    assert workspace._cgroup_memory_budget(verify_enforcement=False) == (400*GIB, None)
    assert workspace._cgroup_memory_headroom_bytes() == 400*GIB
    assert '/proc/self/mountinfo' not in reads
    assert workspace._cgroup_memory_budget() == (400*GIB, 600*GIB)


def test_headroom_only_probe_still_fails_closed_on_missing_usage(monkeypatch):
    files, node, reads = fake_procfs(monkeypatch)
    del files[node+'/memory.current']
    assert workspace._cgroup_memory_budget(verify_enforcement=False) == (0, None)
    assert workspace._cgroup_memory_headroom_bytes() == 0
    assert '/proc/self/mountinfo' not in reads


@pytest.mark.parametrize('slurm', [False, True])
def test_public_sam_probe_verifies_mounts_at_most_once_when_needed(monkeypatch, slurm):
    files, _node, reads = fake_procfs(monkeypatch)
    files['/proc/meminfo'] = f'MemAvailable: {500*GIB//1024} kB\nSwapFree: 0 kB\n'
    monkeypatch.delenv('SLURM_MEM_PER_CPU', raising=False)
    if slurm:
        monkeypatch.setenv('SLURM_MEM_PER_NODE', str(600*1024))
    else:
        monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    assert sam_resources.physical_sam_headroom() == 400*GIB
    assert reads.count('/proc/self/mountinfo') == int(slurm)
