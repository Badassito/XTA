"""CPU-only fixtures for job-scoped memory observations; no hardware queries."""
import io
import json
from pathlib import Path
import threading

import pytest

from XTA import runtime


@pytest.fixture
def files(monkeypatch):
    values = {
        '/proc/self/cgroup': '0::/slurm/job/step\n',
        '/proc/self/mountinfo': '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
    }
    original = Path.open

    class SmallReads(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4096
            return super().read(size)

    def opened(path, mode='r', *args, **kwargs):
        name = path.as_posix()
        if name.startswith(('/proc/', '/sys/fs/cgroup', '/memory view')):
            value = values.get(name, FileNotFoundError(name))
            if isinstance(value, Exception):
                raise value
            assert mode == 'rb' and kwargs['buffering'] == 0
            return SmallReads(value.encode())
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    return values


def level(files, path, *, current='100', maximum='400', v2=True):
    files[path + ('/memory.current' if v2 else '/memory.usage_in_bytes')] = current
    files[path + ('/memory.max' if v2 else '/memory.limit_in_bytes')] = maximum
    files[path + ('/memory.high' if v2 else '/memory.soft_limit_in_bytes')] = '300'
    files[path + '/memory.stat'] = ('anon 50\nfile 40\nshmem 10\npgfault 90\n' if v2 else
                                   'rss 50\ncache 40\ntotal_rss 70\ntotal_cache 60\n')
    if v2:
        files[path + '/memory.events'] = 'high 2\noom 1\noom_kill 1\n'
        files[path + '/memory.pressure'] = 'some avg10=1.00 avg60=0.00 avg300=0.00 total=8\n'


def test_job_and_finite_ancestors_remain_separate_and_root_is_not_the_job(files):
    level(files, '/sys/fs/cgroup/slurm/job/step', maximum='max')
    level(files, '/sys/fs/cgroup/slurm/job', current='800', maximum='900')
    level(files, '/sys/fs/cgroup', current='9000', maximum='max')
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['error'] is None
    assert snapshot['current'] == 100 and snapshot['limit'] is None
    assert snapshot['compatibility_path'].replace('\\', '/') == '/sys/fs/cgroup/slurm/job/step'
    assert snapshot['levels'][1]['current']['value'] == 800
    assert snapshot['levels'][1]['max']['value'] == 900
    assert snapshot['levels'][0]['stat']['value'] == {'anon': 50, 'file': 40, 'shmem': 10}
    assert snapshot['levels'][0]['events']['value']['oom_kill'] == 1
    assert snapshot['levels'][0]['pressure']['value'].startswith('some avg10=')
    assert snapshot['visible_root_reached'] is True
    assert snapshot['above_visible_root'].startswith('unknown')
    assert runtime.RuntimeSystemSampler._read_cgroup_memory() == (None, 100)


def test_bind_mount_root_offset_and_escaped_mountpoint_are_respected(files):
    files['/proc/self/mountinfo'] = '1 0 0:1 /slurm/job /memory\\040view rw - cgroup2 cgroup rw\n'
    level(files, '/memory view/step', current='123', maximum='456')
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['mount_root'] == '/slurm/job'
    assert snapshot['current'] == 123 and snapshot['limit'] == 456
    assert len(snapshot['levels']) == 2
    assert snapshot['levels'][-1]['path'].replace('\\', '/') == '/memory view'


@pytest.mark.parametrize('member', ['/slurm/../escape', '/slurm/./job', '/slurm//job', 'relative', '/slurm\\job'])
def test_invalid_membership_never_falls_back_to_root_memory(files, member):
    files['/proc/self/cgroup'] = '0::' + member
    level(files, '/sys/fs/cgroup', current='999', maximum='9999')
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['error'] and snapshot['levels'] == []
    assert snapshot['current'] is None and snapshot['limit'] is None


@pytest.mark.parametrize('mounts', ['', '1 0 0:1 /other /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
                                  '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
                                  '2 0 0:1 / /memory\\040view rw - cgroup2 cgroup rw\n'])
def test_missing_uncovered_or_ambiguous_mount_is_unknown(files, mounts):
    files['/proc/self/mountinfo'] = mounts
    level(files, '/sys/fs/cgroup/slurm/job/step')
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['error'] and snapshot['current'] is None


def test_nested_counter_overlay_does_not_authenticate_host_values_as_job_values(files):
    level(files, '/sys/fs/cgroup/slurm/job/step')
    files['/proc/self/mountinfo'] += '2 0 0:2 / /sys/fs/cgroup/slurm/job/memory.max rw - tmpfs tmpfs rw\n'
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert 'nested mount' in snapshot['error']
    assert snapshot['levels'] == [] and snapshot['current'] is None


@pytest.mark.parametrize('field,value', [('memory.current', PermissionError('denied')),
                                        ('memory.max', 'oops'), ('memory.high', '-1'),
                                        ('memory.events', 'oom 1\noom 2'), ('memory.stat', 'anon NaN'),
                                        ('memory.pressure', 'some avg10=nan avg60=0 avg300=0 total=1'),
                                        ('memory.pressure', '')])
def test_bad_fields_are_individually_unknown_and_do_not_remove_other_levels(files, field, value):
    job = '/sys/fs/cgroup/slurm/job/step'
    level(files, job)
    level(files, '/sys/fs/cgroup/slurm/job', current='700', maximum='800')
    files[job + '/' + field] = value
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    key = field.removeprefix('memory.')
    assert snapshot['error'] is None
    assert snapshot['levels'][0][key]['value'] is None
    assert snapshot['levels'][0][key]['error']
    assert snapshot['levels'][1]['max']['value'] == 800


def test_missing_stat_parts_are_unknown_not_zero(files):
    job = '/sys/fs/cgroup/slurm/job/step'
    level(files, job, current='0', maximum='0')
    files[job + '/memory.stat'] = 'anon 0\n'
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['current'] == snapshot['limit'] == 0
    assert snapshot['levels'][0]['stat']['value'] == {'anon': 0, 'file': None, 'shmem': None}


def test_ancestor_depth_and_field_bytes_are_bounded(files):
    member = '/' + '/'.join(['deep'] * 30)
    files['/proc/self/cgroup'] = '0::' + member
    level(files, '/sys/fs/cgroup' + member)
    files['/sys/fs/cgroup' + member + '/memory.stat'] = 'x' * 65537
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert len(snapshot['levels']) == 16
    assert snapshot['visible_root_reached'] is False
    assert snapshot['omitted_visible_levels'] == 15
    assert 'observation bound' in snapshot['levels'][0]['stat']['error']


def test_v1_job_pair_is_preserved_with_unlimited_sentinel(files):
    files['/proc/self/cgroup'] = '0::/unified\n5:cpu,memory:/slurm/job\n'
    files['/proc/self/mountinfo'] = '1 0 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n'
    level(files, '/sys/fs/cgroup/memory/slurm/job', current='123', maximum=str(1 << 63), v2=False)
    snapshot = runtime.RuntimeSystemSampler._read_cgroup_memory_snapshot()
    assert snapshot['version'] == 1 and snapshot['current'] == 123 and snapshot['limit'] is None
    assert snapshot['levels'][0]['max']['value'] == 'max'
    assert snapshot['levels'][0]['stat']['value']['total_rss'] == 70
    assert snapshot['levels'][0]['events']['value'] is None
    assert 'v1' in snapshot['levels'][0]['events']['error']


def test_sampler_clears_disappearing_current_and_limit_without_real_gpu_calls(files, monkeypatch):
    level(files, '/sys/fs/cgroup/slurm/job/step')
    records = []

    class Sink:
        lock = threading.RLock()
        gauges = {}
        def gauge(self, name, value): self.gauges[name] = value
        def gauge_many(self, values): self.gauges.update(values)
        def sample_registered_providers(self): pass
        def maybe_flush(self):
            records.append(json.loads(json.dumps(self.gauges)))
            files['/proc/self/cgroup'] = PermissionError('membership gone')

    class Stop:
        values = iter([False, False, True])
        def wait(self, _): return next(self.values)
        def is_set(self): return False

    sampler = runtime.RuntimeSystemSampler(Sink())
    sampler.stop_event = Stop()
    monkeypatch.setattr(sampler, '_sample_gpu', lambda: None)
    sampler._run()
    assert records[0]['system.cgroup_memory_current'] == 100
    assert records[0]['system.cgroup_memory_limit'] == 400
    assert records[1]['system.cgroup_memory_current'] is None
    assert records[1]['system.cgroup_memory_limit'] is None
    assert records[1]['system.cgroup_memory']['error'].startswith('PermissionError')
