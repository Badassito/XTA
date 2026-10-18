"""CPU-only fixtures: never invoke nvidia-smi, NVML, torch or a CUDA context."""
import io
import json
from pathlib import Path
import subprocess

import pytest

from tools import capture_gpu_stall as probe


@pytest.mark.parametrize('payload,binary', [
    (b'', False), (b' \xffid\x00\n', False), (b' \xffid\x00\n', True),
    (b'x' * 4096, True), (b'x' * probe.READ_LIMIT, True),
    (b'x' * (probe.READ_LIMIT + 1), True),
], ids=['empty', 'short-text', 'short-binary', 'chunk', 'limit', 'overflow'])
def test_read_uses_small_requests_until_eof_or_total_bound(payload, binary):
    requests = []
    returned = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            if not 0 < size <= 4096:
                raise OSError(12, 'oversized sysctl read request')
            requests.append(size)
            chunk = super().read(min(size, 3) if len(payload) < 20 else size)
            returned.append(len(chunk))
            return chunk

    class Source:
        def open(self, mode, *, buffering):
            assert (mode, buffering) == ('rb', 0)
            return Stream(payload)

    if len(payload) > probe.READ_LIMIT:
        with pytest.raises(ValueError, match='exceeds .* read bound'):
            probe.read(Source(), binary)
    else:
        result = probe.read(Source(), binary)
        assert result == (payload if binary else payload.decode('utf-8', errors='replace').strip())
        assert isinstance(result, bytes if binary else str)
    assert requests and sum(returned) == min(len(payload), probe.READ_LIMIT + 1)


def test_read_keeps_midstream_oserror_as_unknown():
    class Stream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError(12, 'real read failure')
            return super().read(3)

    class Source:
        def open(self, mode, *, buffering):
            return Stream(b'partial data')

    result = probe.observed(lambda: probe.read(Source()))
    assert result['value'] is None and 'real read failure' in result['error']


def xml(pids=(123,)):
    processes = ''.join(f'<process_info><pid>{pid}</pid><type>C</type></process_info>' for pid in pids)
    return (f'<nvidia_smi_log><gpu id="00000000:01:00.0"><uuid>GPU-test</uuid>'
            f'<minor_number>0</minor_number><processes>{processes}</processes>'
            '</gpu></nvidia_smi_log>').encode()


def stat(pid, ticks=100, cpu=7):
    fields = ['0'] * 50
    fields[0], fields[19], fields[36] = 'S', str(ticks), str(cpu)
    return f'{pid} (name (with spaces)) ' + ' '.join(fields)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture
def proc(tmp_path, monkeypatch):
    root, sys = tmp_path / 'proc', tmp_path / 'sys'
    monkeypatch.setattr(probe, 'PROC', root)
    monkeypatch.setattr(probe, 'SYS', sys)
    real_read = probe.read
    def fixture_read(path, binary=False):
        # PCI colons are valid on Linux but not in Windows fixture filenames.
        if path.parent.name == '0000:01:00.0':
            path = path.parent.with_name('0000_01_00.0') / path.name
        return real_read(path, binary)
    monkeypatch.setattr(probe, 'read', fixture_read)
    put(root / 'sys/kernel/random/boot_id', 'boot-test')
    put(root / 'self/mountinfo', f'1 0 0:1 / {sys.as_posix()}/fs/cgroup rw - cgroup2 cgroup rw')
    for pid in (123, 124, 125):
        base = root / str(pid)
        put(base / 'stat', stat(pid))
        put(base / 'status', 'Name:\tpython\nPPid:\t10\nState:\tS\nCpus_allowed_list:\t4-7\nMems_allowed_list:\t0-1\n')
        put(base / 'schedstat', '0 0 0')
        put(base / 'wchan', '0')
        put(base / 'environ', b'SECRET_TOKEN=never-save-this\0SLURM_JOB_ID=157683\0OTHER_SECRET=private\0')
        put(base / 'numa_maps', '1 default N0=2 N1=1 kernelpagesize_kB=4\n2 huge N1=1 kernelpagesize_kB=2048')
        put(base / 'cgroup', '0::/slurm/job_157683')
        for tid in (pid, pid + 1000, pid + 2000):
            put(base / 'task' / str(tid) / 'stat', stat(tid, ticks=tid + 100))
            for filename in ('status', 'schedstat', 'wchan'):
                put(base / 'task' / str(tid) / filename, (base / filename).read_bytes())
    put(sys / 'fs/cgroup/slurm/job_157683/cpu.stat', 'usage_usec 0\nthrottled_usec 7')
    put(sys / 'fs/cgroup/slurm/job_157683/cpu.max', 'max 100000')
    for name, value in {'numa_node': '1', 'local_cpulist': '4-7', 'current_link_speed': '16.0 GT/s PCIe',
                        'current_link_width': '16', 'max_link_speed': '32.0 GT/s PCIe', 'max_link_width': '16'}.items():
        put(sys / 'bus/pci/devices/0000_01_00.0' / name, value)
    return root


@pytest.fixture
def fake_commands(monkeypatch):
    class Clock:
        now = 0.0

        def sleep(self, duration):
            self.now += duration

    clock = Clock()
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(probe.time, 'sleep', clock.sleep)
    calls = []

    def run(args):
        calls.append(args)
        clock.now += .1
        return {'returncode': 0, 'error': None, 'duration_seconds': .1}, xml() if args[-1] == '-x' else b'fake topology'

    monkeypatch.setattr(probe, 'command', run)
    return clock, calls, run


def test_parse_uuid_pci_compute_only_and_unknown_fields():
    data = xml().replace(b'</processes>', b'<process_info><pid>77</pid><type>G</type></process_info>'
                        b'<process_info><pid>78</pid><type>C+G</type></process_info>'
                        b'<process_info><pid>N/A</pid><type>C</type></process_info>'
                        b'<process_info><pid>79</pid></process_info></processes>')
    result = probe.parse_gpus(data)
    assert [p['pid'] for p in result['compute_processes']] == [123, 78]
    assert result['compute_processes'][0]['uuid'] == 'GPU-test'
    assert result['compute_processes'][0]['pci_bus_id'] == '00000000:01:00.0'
    assert len(result['errors']) == 2
    assert probe.parse_gpus(xml(()))['compute_processes'] == []
    assert probe.parse_gpus(xml(()))['inventory_complete'] is True
    assert probe.observed(lambda: probe.parse_gpus(b'broken XML'))['value'] is None
    assert probe.observed(lambda: probe.parse_gpus(b'<other/>'))['error']


@pytest.mark.parametrize('outcome', ['ok', 'nonzero', 'timeout', 'missing'])
def test_subprocess_timeout_and_errors_are_explicit(monkeypatch, outcome):
    def run(args, **kwargs):
        assert kwargs == {'capture_output': True, 'timeout': 5.0, 'check': False}
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(args, 5, output=b'partial', stderr=b'late')
        if outcome == 'missing':
            raise FileNotFoundError('not installed')
        return subprocess.CompletedProcess(args, 0 if outcome == 'ok' else 1, b'full', b'x' * 5000)
    monkeypatch.setattr(probe.subprocess, 'run', run)
    metadata, data = probe.command(['nvidia-smi', '-q', '-x'])
    assert (metadata['error'] is None) == (outcome == 'ok')
    assert metadata['duration_seconds'] >= 0
    if outcome in ('ok', 'nonzero'):
        assert data == b'full' and metadata['stderr_truncated']
        assert len(metadata['stderr']) == 4096
    elif outcome == 'timeout':
        assert data == b'partial' and metadata['returncode'] is None


def test_process_metadata_numa_cgroup_and_no_environment_leak(proc):
    result = probe.process_snapshot(123, 100, [4096])
    assert result['state'] == 'observed'
    assert result['stat']['value'] == {'start_ticks': 100, 'last_processor': 7}
    assert result['status']['value']['PPid'] == '10'
    assert result['threads'][0]['status']['value']['Cpus_allowed_list'] == '4-7'
    assert result['threads'][0]['stat']['value']['start_ticks'] == 223
    assert result['threads'][0]['identity_stable'] is True
    assert result['numa_mapped_bytes']['value'] == {'N0': 8192, 'N1': 4096 + 2097152}
    assert result['cgroup_cpu']['value']['cpu.stat']['value']['usage_usec'] == '0'
    assert result['slurm_job_id']['value'] == '157683'
    assert 'SECRET' not in json.dumps(result) and 'never-save-this' not in json.dumps(result)
    assert result['wchan'] == {'value': '0', 'error': None}  # Literal kernel value, not evidence of no waiting.


def test_unknown_is_not_zero_and_permission_error_is_retained(proc, monkeypatch):
    (proc / '123/schedstat').unlink()
    real = probe.read
    def denied(path, binary=False):
        if path.name == 'environ':
            raise PermissionError('denied')
        return real(path, binary)
    monkeypatch.setattr(probe, 'read', denied)
    result = probe.process_snapshot(123, 100, [0])
    assert result['schedstat']['value'] is None and result['schedstat']['error']
    assert result['slurm_job_id']['value'] is None and 'PermissionError' in result['slurm_job_id']['error']
    assert result['omitted_threads'] == 3
    assert result['wchan']['value'] == '0'


def test_reused_disappeared_and_malformed_process_identities(proc):
    assert probe.process_snapshot(123, 99, [10])['state'] == 'pid_reused'
    (proc / '123/stat').unlink()
    missing = probe.process_snapshot(123, 100, [10])
    assert missing['state'] == 'unavailable' and 'FileNotFoundError' in missing['start_ticks']['error']
    put(proc / '123/stat', 'malformed')
    assert probe.observed(lambda: probe.start_ticks(123))['value'] is None


def test_identity_change_during_snapshot_marks_all_fields_unstable(proc, monkeypatch):
    values = iter([100, 200])
    monkeypatch.setattr(probe, 'start_ticks', lambda pid: next(values))
    assert probe.process_snapshot(123, 100, [1])['state'] == 'changed_or_disappeared_during_snapshot'


def test_pci_and_cgroup_paths_are_bounded(proc):
    assert probe.pci_files('00000000:01:00.0')['numa_node']['value'] == '1'
    assert probe.pci_files('../../secret')['error']
    put(proc / '123/cgroup', '0::/../../secret')
    assert probe.observed(lambda: probe.cgroup_cpu(123))['error']
    put(proc / '123/cgroup', '2:cpu:/slurm/job')
    assert 'unified cgroup' in probe.observed(lambda: probe.cgroup_cpu(123))['error']


@pytest.mark.parametrize('data', [b'<nvidia_smi_log/>', xml().replace(b'<processes>', b'<processes>N/A'),
                                 xml().replace(b'<processes>', b'<missing>').replace(b'</processes>', b'</missing>')])
def test_missing_inventory_is_unknown_not_empty(data):
    result = probe.parse_gpus(data)
    assert result['inventory_complete'] is False and result['errors']


def test_explicit_zero_gpu_inventory_is_known_empty():
    result = probe.parse_gpus(b'<nvidia_smi_log><attached_gpus>0</attached_gpus></nvidia_smi_log>')
    assert result['inventory_complete'] is True and result['gpus'] == []


@pytest.mark.parametrize('kind', ['N/A', 'MPS', 'future-type'])
def test_unknown_process_kind_does_not_establish_absence(kind):
    result = probe.parse_gpus(xml().replace(b'<type>C</type>', f'<type>{kind}</type>'.encode()))
    assert result['inventory_complete'] is False


@pytest.mark.parametrize('field', ['uuid', 'pci'])
def test_unsupported_gpu_identity_is_partial(field):
    data = xml().replace(b'GPU-test', b'N/A') if field == 'uuid' else xml().replace(b'00000000:01:00.0', b'N/A')
    assert probe.parse_gpus(data)['inventory_complete'] is False


@pytest.mark.parametrize('count,complete', [(0, False), (1, True), (2, False)])
def test_attached_gpu_count_must_match_inventory(count, complete):
    data = xml().replace(b'<nvidia_smi_log>', f'<nvidia_smi_log><attached_gpus>{count}</attached_gpus>'.encode())
    assert probe.parse_gpus(data)['inventory_complete'] is complete


def test_cgroup_mount_root_is_applied_before_reading_quota(proc):
    mount = probe.SYS / 'fs/cgroup'
    put(proc / 'self/mountinfo', f'1 0 0:1 /slurm {mount.as_posix()} rw - cgroup2 cgroup rw')
    put(mount / 'job_157683/cpu.stat', 'usage_usec 123')
    put(mount / 'job_157683/cpu.max', '200000 100000')
    result = probe.cgroup_cpu(123)
    assert result['mount_root'] == '/slurm'
    assert result['cpu.stat']['value'] == {'usage_usec': '123'}
    assert result['cpu.max']['value'] == '200000 100000'
    put(proc / 'self/mountinfo', f'1 0 0:1 /different {mount.as_posix()} rw - cgroup2 cgroup rw')
    assert 'mapping unknown' in probe.observed(lambda: probe.cgroup_cpu(123))['error']


def test_thread_reuse_inside_snapshot_is_marked_unstable(proc, monkeypatch):
    real = probe.process_stat
    calls = 0
    def changed(pid, base=None):
        nonlocal calls
        value = real(pid, base)
        if base is not None:
            calls += 1
            value['start_ticks'] = 1 if calls == 1 else 2
        return value
    monkeypatch.setattr(probe, 'process_stat', changed)
    row = probe.process_snapshot(123, 100, [1])['threads'][0]
    assert row['identity_stable'] is False


def test_visible_ancestor_quota_and_throttle_are_not_missed(proc, monkeypatch):
    mount = probe.SYS / 'fs/cgroup'
    put(mount / 'slurm/cpu.max', '100000 100000')
    put(mount / 'slurm/cpu.stat', 'throttled_usec 999')
    result = probe.cgroup_cpu(123)
    assert result['cpu.max']['value'] == 'max 100000'
    assert result['ancestors'][0]['cpu.max']['value'] == '100000 100000'
    assert result['ancestors'][0]['cpu.stat']['value']['throttled_usec'] == '999'
    assert result['visible_root_reached'] is True and result['omitted_visible_levels'] == 0
    monkeypatch.setattr(probe, 'MAX_CGROUP_DEPTH', 1)
    capped = probe.cgroup_cpu(123)
    assert capped['visible_root_reached'] is False and capped['omitted_visible_levels'] == 2
    assert capped['above_visible_root'].startswith('unknown')


def test_cgroup_memory_counters_and_finite_ancestor_limit_are_retained(proc):
    mount = probe.SYS / 'fs/cgroup'
    fields = {
        'memory.current': '800', 'memory.max': 'max', 'memory.high': '900',
        'memory.peak': '850', 'memory.events': 'oom 0\noom_kill 0',
        'memory.stat': 'anon 700\nfile 100',
        'memory.pressure': 'some avg10=0.00 avg60=0.00 avg300=0.00 total=12',
        'memory.swap.current': '0', 'memory.swap.max': 'max',
        'memory.swap.events': 'high 0\nmax 0\nfail 0',
    }
    for name, value in fields.items():
        put(mount / 'slurm/job_157683' / name, value)
    put(mount / 'slurm/memory.max', '1024')
    put(mount / 'slurm/memory.events', 'oom 2\noom_kill 1')
    result = probe.cgroup_cpu(123)
    for name, value in fields.items():
        assert result[name] == {'value': value, 'error': None}
    assert result['ancestors'][0]['memory.max']['value'] == '1024'
    assert result['ancestors'][0]['memory.events']['value'] == 'oom 2\noom_kill 1'
    assert result['ancestors'][0]['memory.current']['value'] is None
    assert 'FileNotFoundError' in result['ancestors'][0]['memory.current']['error']
    assert result['cpu.max']['value'] == 'max 100000'
    assert result['cpu.stat']['value']['throttled_usec'] == '7'
    assert result['visible_root_reached'] is True
    assert result['above_visible_root'].startswith('unknown')


def test_unreadable_memory_keeps_cpu_snapshot_evidence(proc, monkeypatch):
    real = probe.read

    def denied(path, binary=False):
        if path.name == 'memory.current':
            raise PermissionError('memory denied')
        return real(path, binary)

    monkeypatch.setattr(probe, 'read', denied)
    snapshot = probe.process_snapshot(123, 100, [0])['cgroup_cpu']
    assert snapshot['error'] is None
    assert snapshot['value']['cpu.stat']['value']['throttled_usec'] == '7'
    assert snapshot['value']['cpu.max']['value'] == 'max 100000'
    assert snapshot['value']['memory.current']['value'] is None
    assert 'PermissionError' in snapshot['value']['memory.current']['error']
    assert snapshot['value']['memory.max']['value'] is None
    assert 'FileNotFoundError' in snapshot['value']['memory.max']['error']


@pytest.mark.parametrize('seconds', [0, -1, 121])
def test_capture_bounds_create_nothing(tmp_path, seconds):
    with pytest.raises(ValueError):
        probe.capture(tmp_path, seconds)
    assert list(tmp_path.iterdir()) == []


def test_output_must_exist_and_unique_children_never_overwrite(tmp_path, proc, fake_commands):
    with pytest.raises(ValueError, match='must not be empty'):
        probe.capture('', 1)
    with pytest.raises(FileNotFoundError):
        probe.capture(tmp_path / 'missing', 1)
    file = tmp_path / 'file'
    file.write_text('keep')
    with pytest.raises(ValueError):
        probe.capture(file, 1)
    first, second = probe.capture(tmp_path, 1), probe.capture(tmp_path, 1)
    assert first != second and first.parent == second.parent == tmp_path
    assert file.read_text() == 'keep'
    assert (first / 'gpu-0000.xml').read_bytes() == xml()
    assert json.loads((first / 'manifest.json').read_text())['samples'] == 1
    assert json.loads((first / 'process-end.json').read_text())[0]['expected_start_ticks'] == 100
    assert json.loads((first / 'pci-first-seen.json').read_text())['00000000:01:00.0']['files']['max_link_width']['value'] == '16'
    assert fake_commands[1].count(['nvidia-smi', 'topo', '-m']) == 4


def test_process_and_thread_caps_are_visible_in_capture(tmp_path, proc, fake_commands, monkeypatch):
    monkeypatch.setattr(probe, 'MAX_PIDS', 1)
    monkeypatch.setattr(probe, 'MAX_THREADS', 1)
    run = fake_commands[2]
    def many(args):
        result, data = run(args)
        return result, xml((123, 124, 125)) if args[-1] == '-x' else data
    monkeypatch.setattr(probe, 'command', many)
    folder = probe.capture(tmp_path, 1)
    sample = json.loads((folder / 'samples.jsonl').read_text())
    assert sample['omitted_pids'] == [124, 125]
    process = json.loads((folder / 'process-first-seen.json').read_text())[0]
    assert len(process['threads']) == 1 and process['omitted_threads'] == 2


def test_pid_reuse_creates_separate_generations(tmp_path, proc, fake_commands, monkeypatch):
    run = fake_commands[2]
    seen = 0
    def reuse(args):
        nonlocal seen
        result, data = run(args)
        if args[-1] == '-x':
            seen += 1
            if seen == 2:
                put(proc / '123/stat', stat(123, ticks=200))
        return result, data
    monkeypatch.setattr(probe, 'command', reuse)
    folder = probe.capture(tmp_path, 2)
    first = json.loads((folder / 'process-first-seen.json').read_text())
    end = json.loads((folder / 'process-end.json').read_text())
    assert [p['expected_start_ticks'] for p in first] == [100, 200]
    assert [p['state'] for p in end] == ['pid_reused', 'observed']


def test_failed_gpu_query_retains_partial_and_unknown_coverage(tmp_path, proc, fake_commands, monkeypatch):
    run = fake_commands[2]
    def timeout(args):
        result, data = run(args)
        return ({'returncode': None, 'error': 'timeout'}, b'<partial') if args[-1] == '-x' else (result, data)
    monkeypatch.setattr(probe, 'command', timeout)
    folder = probe.capture(tmp_path, 1)
    sample = json.loads((folder / 'samples.jsonl').read_text())
    assert (folder / 'gpu-0000.partial.xml').read_bytes() == b'<partial'
    assert sample['parsed']['value'] is None
    assert sample['process_coverage'] == 'unknown' and sample['previous_pids_not_reported'] is None
    assert json.loads((folder / 'manifest.json').read_text())['gpu_query_coverage'] == 'failed'


def test_main_reports_nonzero_when_all_queries_failed(tmp_path, proc, fake_commands, monkeypatch, capsys):
    run = fake_commands[2]
    def missing(args):
        metadata, data = run(args)
        return ({'returncode': None, 'error': 'missing'}, b'') if args[-1] == '-x' else (metadata, data)
    monkeypatch.setattr(probe, 'command', missing)
    with pytest.raises(SystemExit) as exc:
        probe.main(['--output-dir', str(tmp_path), '--seconds', '1'])
    assert exc.value.code == 1 and 'all GPU queries failed' in capsys.readouterr().err


def test_partial_process_inventory_never_claims_prior_pid_disappeared(tmp_path, proc, fake_commands, monkeypatch):
    run = fake_commands[2]
    queries = 0
    def partial(args):
        nonlocal queries
        metadata, data = run(args)
        if args[-1] == '-x':
            queries += 1
            if queries == 2:
                data = b'<nvidia_smi_log/>'
        return metadata, data
    monkeypatch.setattr(probe, 'command', partial)
    folder = probe.capture(tmp_path, 2)
    rows = [json.loads(line) for line in (folder / 'samples.jsonl').read_text().splitlines()]
    assert rows[1]['process_coverage'] == 'partial'
    assert rows[1]['previous_pids_not_reported'] is None


def test_help_does_not_query_hardware(monkeypatch, capsys):
    monkeypatch.setattr(probe, 'command', lambda args: pytest.fail('hardware query during help'))
    with pytest.raises(SystemExit) as exc:
        probe.main(['--help'])
    assert exc.value.code == 0 and '--output-dir' in capsys.readouterr().out
