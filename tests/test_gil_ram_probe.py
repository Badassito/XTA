"""RAM headroom probe (#6 GIL sweep): fresh probes whose /proc and cgroupfs reads keep the GIL.

Every probe is fresh (no cached usage or limit), so it starts after its caller's call;
the GIL-holding reader returns the same text and errors as Path.read_text.
"""
import os
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA import publication_memory, runtime, workspace


@pytest.fixture(autouse=True)
def fresh_gil_read_state(monkeypatch):
    """Isolate the process-wide GIL-held read classes, counters and demotions."""
    monkeypatch.setattr(workspace, '_GIL_HELD_CLASSES',
                        set(workspace._LOCK_FREE_PROBE_CLASSES | workspace._LOCK_PRONE_PROBE_CLASSES))
    monkeypatch.setattr(workspace, '_GIL_HELD_READ_CALLS', {})
    monkeypatch.setattr(workspace, '_GIL_HELD_READ_MAX_SECONDS', {})
    monkeypatch.setattr(workspace, '_GIL_HELD_DEMOTED', {})
    monkeypatch.setattr(workspace, '_GIL_HOLD_LIMIT_SECONDS', 0.005)


# --- Verbatim pre-change bodies (HEAD) used as the reference -----------------
def old_available_anon_work_bytes(*, meminfo=None):
    info = workspace._read_meminfo_bytes() if meminfo is None else meminfo
    mem_avail = int(info.get('MemAvailable', 0))
    swap_free = int(info.get('SwapFree', 0))
    node_avail = max(0, mem_avail + swap_free)
    cgroup_headroom = workspace._cgroup_memory_headroom_bytes()
    if cgroup_headroom is None:
        return node_avail
    return min(node_avail, int(cgroup_headroom))


def old_publication_ram_headroom():
    info = workspace._read_meminfo_bytes()
    if 'MemAvailable' not in info and os.name == 'nt':
        try:
            import psutil
            return max(0, int(psutil.virtual_memory().available))
        except (ImportError, OSError, ValueError, AttributeError):
            return 0
    return max(0, min(int(info.get('MemAvailable', 0)),
                      int(old_available_anon_work_bytes(meminfo=info))))


def dict_reader(files, log=None):
    def read(path, *args, **kwargs):
        if log is not None:
            log.append(path.as_posix())
        value = files.get(path.as_posix(), FileNotFoundError(2, 'No such file', str(path)))
        if isinstance(value, Exception):
            raise value
        return value
    return read


def fake_proc(monkeypatch, files, log=None):
    read = dict_reader(files, log)
    monkeypatch.setattr(Path, 'read_text', read)
    return read


def meminfo_text(avail, swap, extra=True):
    lines = ['MemTotal:       99999999 kB']
    if avail is not None:
        lines.append(f'MemAvailable:   {avail} kB')
    if extra:
        lines.append('Cached:         1234 kB')
    if swap is not None:
        lines.append(f'SwapFree:       {swap} kB')
    return '\n'.join(lines) + '\n'


def random_tree(rng):
    v2 = rng.random() < .6
    base = '/sys/fs/cgroup' if v2 else '/sys/fs/cgroup/memory'
    limit_name, usage_name = ('memory.max', 'memory.current') if v2 else (
        'memory.limit_in_bytes', 'memory.usage_in_bytes')
    files = {'/proc/meminfo': meminfo_text(
        rng.choice([None, 0, rng.randrange(1, 10**8)]), rng.choice([None, 0, rng.randrange(1, 10**7)]))}
    if rng.random() < .15:
        return files
    files['/proc/self/cgroup'] = ('0::' if v2 else '5:memory:') + '/job/step/task\n'
    for node in ('', '/job', '/job/step', '/job/step/task'):
        kind = rng.random()
        if kind < .3:
            continue
        path = base + node
        if kind < .5:
            files[path + '/' + limit_name] = 'max' if v2 else str(1 << 62)
            continue
        limit = rng.randrange(0, 10**11)
        files[path + '/' + limit_name] = str(limit)
        usage = rng.choice([rng.randrange(0, 2 * limit + 1), PermissionError(13, 'no')]) \
            if rng.random() < .1 else rng.randrange(0, 2 * limit + 1)
        files[path + '/' + usage_name] = usage if isinstance(usage, Exception) else str(usage)
        if rng.random() < .7:
            keys = ('inactive_file', 'file_dirty', 'file_writeback') if v2 else (
                'total_inactive_file', 'total_dirty', 'total_writeback')
            files[path + '/memory.stat'] = ''.join(
                f'{key} {rng.randrange(0, 10**10)}\n' for key in keys if rng.random() < .95)
        if not v2:
            files[path + '/memory.use_hierarchy'] = rng.choice(['1', '0'])
    return files


def _gil_holding_libc():
    import ctypes
    if sys.platform.startswith('linux'):
        return None  # the production default: PyDLL(None)
    if os.name == 'nt':
        ucrt = ctypes.PyDLL('ucrtbase', use_errno=True)
        return SimpleNamespace(open=ucrt._open, read=ucrt._read, close=ucrt._close)
    pytest.skip('no GIL-holding C runtime hook for this platform')


def crt_reader():
    reader = workspace._gil_holding_text_reader(_gil_holding_libc())
    if reader is None:
        pytest.skip('GIL-holding reader unavailable here (non-kernel /proc mounts?)')
    return reader


def assert_membership_resolved_for_this_tree(files, reads):
    """After a layout reset, membership is read once when present and retried while missing."""
    count = reads.count('/proc/self/cgroup')
    assert count >= 1 and ('/proc/self/cgroup' not in files or count == 1)


def write_tree(root, files):
    for name, value in files.items():
        if isinstance(value, Exception):
            continue  # unreadable entries are covered by the dict-backed tests
        path = Path(str(root) + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode())


def test_path_read_text_probe_matches_previous_formula_on_randomized_trees(monkeypatch):
    rng = random.Random(1234)
    files, reads = {}, []
    fake_proc(monkeypatch, files, reads)
    for _ in range(400):
        files.clear()
        files.update(random_tree(rng))
        workspace._reset_cgroup_layout_cache()
        reads.clear()
        assert workspace.available_anon_work_bytes() == old_available_anon_work_bytes()
        info = workspace._read_meminfo_bytes()
        assert workspace.available_anon_work_bytes(meminfo=info) == old_available_anon_work_bytes(meminfo=info)
        if 'MemAvailable' in info or os.name != 'nt':
            assert publication_memory.publication_ram_headroom() == old_publication_ram_headroom()
        assert_membership_resolved_for_this_tree(files, reads)


def test_gil_reader_probe_matches_path_read_text_probe_on_randomized_trees(monkeypatch):
    """With the reader active (genuine Path.read_text), results equal the old Path.read_text probe."""
    rng = random.Random(4321)
    files = {}
    reader_log, released_log = [], []
    monkeypatch.setattr(workspace, '_GIL_READER', dict_reader(files, reader_log))
    monkeypatch.setattr(workspace, '_gil_releasing_read_text', dict_reader(files, released_log))
    old_reader = dict_reader(files)
    for _ in range(400):
        files.clear()
        files.update(random_tree(rng))
        workspace._reset_cgroup_layout_cache()
        reader_log.clear()
        released_log.clear()
        anon = workspace.available_anon_work_bytes()
        info = workspace._read_meminfo_bytes()
        anon_snapshot = workspace.available_anon_work_bytes(meminfo=info)
        physical = (publication_memory.publication_ram_headroom()
                    if 'MemAvailable' in info or os.name != 'nt' else None)
        budget = workspace._cgroup_memory_budget()
        assert reader_log  # the reader served this caller's own reads
        assert_membership_resolved_for_this_tree(files, reader_log + released_log)
        with monkeypatch.context() as patched:
            patched.setattr(Path, 'read_text', old_reader)
            workspace._reset_cgroup_layout_cache()  # the reference resolves membership itself
            assert anon == old_available_anon_work_bytes()
            assert anon_snapshot == old_available_anon_work_bytes(meminfo=info)
            if physical is not None:
                assert physical == old_publication_ram_headroom()
            assert budget == workspace._cgroup_memory_budget()


def test_real_files_through_crt_reader_match_path_read_text(monkeypatch, tmp_path):
    """End to end over real files: the C-runtime reader gives the old probe's values."""
    reader = crt_reader()
    root = tmp_path.as_posix()
    rng = random.Random(99)
    reads = []

    def held(path):
        reads.append(path.as_posix())
        return reader(Path(root + path.as_posix()))

    def released(path):
        reads.append(path.as_posix())
        return Path(root + path.as_posix()).read_text()
    monkeypatch.setattr(workspace, '_GIL_READER', held)
    monkeypatch.setattr(workspace, '_gil_releasing_read_text', released)
    original = Path.read_text

    def rerooted(self, *args, **kwargs):
        return original(Path(root + self.as_posix()), *args, **kwargs)
    for index in range(40):
        files = {name: value for name, value in random_tree(rng).items() if not isinstance(value, Exception)}
        root = (tmp_path / str(index)).as_posix()
        write_tree(root, files)
        workspace._reset_cgroup_layout_cache()
        reads.clear()
        new = (workspace.available_anon_work_bytes(), workspace._cgroup_memory_budget())
        info = workspace._read_meminfo_bytes()
        new_physical = publication_memory.publication_ram_headroom() if 'MemAvailable' in info else None
        assert_membership_resolved_for_this_tree(files, reads)
        with monkeypatch.context() as patched:
            patched.setattr(Path, 'read_text', rerooted)
            workspace._reset_cgroup_layout_cache()  # the reference resolves membership itself
            assert new ==(old_available_anon_work_bytes(), workspace._cgroup_memory_budget())
            if new_physical is not None:
                assert new_physical == old_publication_ram_headroom()


def test_reader_only_serves_proc_and_sys_paths(monkeypatch, tmp_path):
    served = []
    monkeypatch.setattr(workspace, '_GIL_READER', lambda path: served.append(path.as_posix()) or 'x')
    assert workspace._probe_read_text(Path('/proc/meminfo')) == 'x'
    assert workspace._probe_read_text(Path('/sys/fs/cgroup/memory.max')) == 'x'
    other = tmp_path / 'proc_like'
    other.write_text('7\n')
    assert workspace._probe_read_text(other) == '7\n'
    assert served == ['/proc/meminfo', '/sys/fs/cgroup/memory.max']


def test_replaced_path_read_text_wins_over_reader(monkeypatch):
    calls = []
    monkeypatch.setattr(workspace, '_GIL_READER', lambda path: calls.append(path) or meminfo_text(1, 1))
    fake_proc(monkeypatch, {'/proc/meminfo': meminfo_text(5, 0)})
    assert workspace.available_anon_work_bytes() == 5 * 1024
    assert calls == []


def test_reader_disabled_off_linux_by_env_and_on_fuse(monkeypatch):
    for platform in ('win32', 'darwin', 'cygwin'):
        monkeypatch.setattr(workspace.sys, 'platform', platform)
        assert workspace._gil_holding_text_reader() is None
    monkeypatch.setattr(workspace.sys, 'platform', 'linux')
    monkeypatch.setenv('YOLO_TTA_RAM_PROBE_GIL_READS', '0')
    assert workspace._gil_holding_text_reader() is None
    monkeypatch.delenv('YOLO_TTA_RAM_PROBE_GIL_READS')
    fake_proc(monkeypatch, {'/proc/self/mountinfo':
                            '22 1 0:21 / /proc rw - proc proc rw\n'
                            '40 22 0:50 / /proc/meminfo rw - fuse.lxcfs lxcfs rw\n'})
    assert workspace._gil_holding_text_reader() is None
    fake_proc(monkeypatch, {})  # unreadable mountinfo
    assert workspace._gil_holding_text_reader() is None


def test_off_linux_probe_uses_path_read_text(monkeypatch):
    """Windows/macOS keep the exact old path: the reader resolves to None."""
    if sys.platform.startswith('linux'):
        pytest.skip('checks non-Linux behaviour')
    monkeypatch.setattr(workspace, '_GIL_READER', workspace._GIL_READER_UNRESOLVED)
    workspace.available_anon_work_bytes()
    assert workspace._GIL_READER is None
    original = Path.read_text
    seen = []

    def spy(self, *args, **kwargs):
        seen.append(self.as_posix())
        return original(self, *args, **kwargs)
    # Rebind the module's notion of the genuine method so the spy counts as unpatched.
    monkeypatch.setattr(Path, 'read_text', spy)
    monkeypatch.setattr(workspace, '_PATH_READ_TEXT_ORIGINAL', spy)
    workspace.available_anon_work_bytes()
    assert '/proc/meminfo' in seen


def test_reader_resolves_once(monkeypatch):
    built = []
    monkeypatch.setattr(workspace, '_GIL_READER', workspace._GIL_READER_UNRESOLVED)
    monkeypatch.setattr(workspace, '_gil_holding_text_reader',
                        lambda: built.append(1) or dict_reader({'/proc/meminfo': meminfo_text(3, 0)}))
    for _ in range(5):
        assert workspace._read_meminfo_bytes()['MemAvailable'] == 3 * 1024
    assert built == [1]


def test_every_probe_is_fresh_and_starts_after_the_call(monkeypatch):
    """Regression for memfd_ram_headroom's "block sample before headroom" rule: pages that
    become resident between the caller's block sample and its headroom call are already out of
    the headroom it receives. No cached or earlier sample may be returned."""
    state = {'avail_kib': 64 * 1024 * 1024}
    events = []

    def reader(path):
        events.append(('read', path.as_posix()))
        if path.as_posix() == '/proc/meminfo':
            return meminfo_text(state['avail_kib'], 0)
        raise FileNotFoundError(2, 'No such file', str(path))
    monkeypatch.setattr(workspace, '_GIL_READER', reader)
    monkeypatch.setattr(workspace, '_gil_releasing_read_text', reader)
    rng = random.Random(5)
    for step in range(200):
        events.append(('block_sample', step))
        # Pages fault into a memfd after the block sample, before the headroom probe.
        state['avail_kib'] -= rng.randrange(0, 256 * 1024)  # up to 256 MiB per step
        fresh = state['avail_kib'] * 1024
        marker = len(events)
        assert publication_memory.publication_ram_headroom() == fresh
        assert workspace.available_anon_work_bytes() == fresh
        assert any(kind == 'read' for kind, _ in events[marker:])
        # Credit as memfd_ram_headroom does: headroom already excludes the faulted pages.
        headroom, _resident, _proof = runtime.memfd_ram_headroom((), publication_memory.publication_ram_headroom)
        assert headroom <= fresh


def test_no_background_sampler_thread(monkeypatch):
    fake_proc(monkeypatch, {'/proc/meminfo': meminfo_text(1, 0)})
    for _ in range(10):
        workspace.available_anon_work_bytes()
        publication_memory.publication_ram_headroom()
    assert not any('ram' in thread.name and 'sampler' in thread.name for thread in threading.enumerate())


def test_gil_holding_reader_matches_path_read_text(tmp_path):
    reader = crt_reader()
    cases = {
        'meminfo': b'MemTotal:       1 kB\nMemAvailable:   2 kB\n',
        'crlf': b'a 1\r\nb 2\rc 3\n',
        'empty': b'',
        'large': b''.join(b'inactive_file %d\n' % i for i in range(20000)),  # > 64 KiB
    }
    for name, payload in cases.items():
        path = tmp_path / name
        path.write_bytes(payload)
        assert reader(path) == path.read_text()
    with pytest.raises(FileNotFoundError):
        reader(tmp_path / 'missing')
    (tmp_path / 'bad').write_bytes(b'\xff\xfe\xfa')
    try:
        expected = (tmp_path / 'bad').read_text()
    except UnicodeDecodeError:
        with pytest.raises(UnicodeDecodeError):
            reader(tmp_path / 'bad')
    else:
        assert reader(tmp_path / 'bad') == expected


def test_gil_holding_reader_is_thread_safe(tmp_path):
    """One reader serves every caller thread; each read uses its own buffer."""
    reader = crt_reader()
    paths = []
    for index in range(8):
        path = tmp_path / f'f{index}'
        path.write_bytes(b''.join(b'k%d %d\n' % (index, i) for i in range(index * 3000)))
        paths.append(path)
    expected = [path.read_text() for path in paths]
    failures = []

    def worker(offset):
        for turn in range(200):
            index = (offset + turn) % len(paths)
            if reader(paths[index]) != expected[index]:
                failures.append(index)
    threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_gil_holding_reads_refuse_fuse_overlays():
    base = ('22 1 0:21 / /proc rw,nosuid - proc proc rw\n'
            '23 1 0:22 / /sys rw - sysfs sysfs rw\n'
            '29 23 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n'
            '30 1 8:1 / / rw - ext4 /dev/sda1 rw\n')
    assert workspace._probe_mounts_kernel_backed(base)
    lxcfs = base + '40 22 0:50 / /proc/meminfo rw - fuse.lxcfs lxcfs rw\n'
    assert not workspace._probe_mounts_kernel_backed(lxcfs)
    escaped = base + '41 23 0:51 / /sys/a\\040b rw - fuse.x x rw\n'
    assert not workspace._probe_mounts_kernel_backed(escaped)
    assert not workspace._probe_mounts_kernel_backed(base.replace('cgroup2 cgroup2', 'nfs4 srv:/x'))
    assert not workspace._probe_mounts_kernel_backed(base.splitlines()[1] + '\n')  # no /proc
