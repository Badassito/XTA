"""RAM probe review fix: GIL-held reads of lock-prone kernel files are bounded.

/proc/self/cgroup (cgroup_mutex before Linux 6.8), /proc/self/mountinfo
(namespace_sem) and memory.stat (rstat flush) can wait in the kernel. Each GIL-held
read is timed per class; a class whose read exceeds the hold limit is demoted to the
GIL-releasing Path.read_text for the rest of the process. YOLO_TTA_RAM_PROBE_GIL_READS
=counters never holds the GIL for those classes. Values never change.
"""
import random
import sys
from pathlib import Path

import pytest

from XTA import workspace
from tests.test_gil_ram_probe import (assert_membership_resolved_for_this_tree, dict_reader,  # noqa: F401
                                      fresh_gil_read_state, old_available_anon_work_bytes, random_tree)

CLASSES = {'/proc/meminfo': 'meminfo',
           '/sys/fs/cgroup/memory.max': 'cgroup_counter',
           '/sys/fs/cgroup/job/step/memory.current': 'cgroup_counter',
           '/sys/fs/cgroup/memory/job/memory.limit_in_bytes': 'cgroup_counter',
           '/sys/fs/cgroup/memory/job/memory.usage_in_bytes': 'cgroup_counter',
           '/sys/fs/cgroup/memory/job/memory.use_hierarchy': 'cgroup_counter',
           '/proc/self/cgroup': 'cgroup_membership',
           '/proc/self/mountinfo': 'mountinfo',
           '/sys/fs/cgroup/memory.stat': 'memory_stat',
           '/sys/fs/cgroup/memory/job/memory.stat': 'memory_stat'}
NOT_PROBE = ['/proc/self/status', '/sys/kernel/mm/memory.max', '/proc/memory.max',
             '/sys/fs/cgroup/job/memory.pressure', '/sys/fs/cgroup/job/memory.events',
             '/proc/self/cgroupx', '/sys/fs/cgroupx/memory.max']
LOCK_PRONE = {'cgroup_membership', 'mountinfo', 'memory_stat'}


@pytest.fixture
def fresh_state(fresh_gil_read_state):
    return None


@pytest.fixture
def split_readers(monkeypatch, fresh_state):
    held, released = [], []
    monkeypatch.setattr(workspace, '_GIL_READER', lambda p: held.append(p.as_posix()) or 'h')
    monkeypatch.setattr(workspace, '_gil_releasing_read_text',
                        lambda p: released.append(p.as_posix()) or 'r')
    return held, released


@pytest.mark.parametrize('name', sorted(CLASSES))
def test_probe_files_are_classified(name):
    assert workspace._probe_read_class(name) == CLASSES[name]
    assert (CLASSES[name] in LOCK_PRONE) == (CLASSES[name] in workspace._LOCK_PRONE_PROBE_CLASSES)


@pytest.mark.parametrize('name', NOT_PROBE)
def test_other_files_always_release_the_gil(split_readers, name):
    held, released = split_readers
    assert workspace._probe_read_class(name) is None
    assert workspace._probe_read_text(Path(name)) == 'r'
    assert held == [] and released == [name]


@pytest.mark.parametrize('name', sorted(CLASSES))
def test_fast_reads_stay_gil_held(split_readers, name):
    held, released = split_readers
    for _ in range(3):
        assert workspace._probe_read_text(Path(name)) == 'h'
    assert held == [name] * 3 and released == []
    stats = workspace.ram_probe_gil_read_stats()
    assert stats['ram_probe_gil_held_reads_by_class'] == {CLASSES[name]: 3}
    assert stats['ram_probe_gil_held_demoted'] == {}


@pytest.mark.parametrize('slow', ['/proc/self/cgroup', '/proc/self/mountinfo',
                                  '/sys/fs/cgroup/job/memory.stat', '/proc/meminfo'])
def test_slow_gil_held_read_demotes_only_its_class(monkeypatch, fresh_state, slow):
    """Regression for the review: a kernel wait with the GIL held happens at most once per class."""
    clock = [0.0]
    monkeypatch.setattr(workspace, '_perf_counter', lambda: clock[0])
    held, released = [], []

    def reader(path):
        held.append(path.as_posix())
        if path.as_posix() == slow:
            clock[0] += 0.050  # e.g. cgroup_mutex held by a concurrent cgroup rmdir
        return 'h'
    monkeypatch.setattr(workspace, '_GIL_READER', reader)
    monkeypatch.setattr(workspace, '_gil_releasing_read_text',
                        lambda p: released.append(p.as_posix()) or 'r')
    assert workspace._probe_read_text(Path(slow)) == 'h'  # the slow read still returns its text
    kind = workspace._probe_read_class(slow)
    assert kind not in workspace._GIL_HELD_CLASSES
    for _ in range(3):
        assert workspace._probe_read_text(Path(slow)) == 'r'
    assert held == [slow] and released == [slow] * 3
    others = [name for name, other in CLASSES.items() if other != kind]
    for name in others:
        assert workspace._probe_read_text(Path(name)) == 'h'
    stats = workspace.ram_probe_gil_read_stats()
    assert stats['ram_probe_gil_held_demoted'] == {kind: pytest.approx(0.050)}
    assert stats['ram_probe_gil_held_max_seconds'] == pytest.approx(0.050)


def test_read_at_the_limit_is_not_demoted(monkeypatch, fresh_state):
    clock = [0.0]
    monkeypatch.setattr(workspace, '_perf_counter', lambda: clock[0])

    def reader(path):
        clock[0] += 0.004  # below the 5 ms limit (exact-limit float steps are not stable)
        return 'h'
    monkeypatch.setattr(workspace, '_GIL_READER', reader)
    for _ in range(5):
        assert workspace._probe_read_text(Path('/proc/self/cgroup')) == 'h'
    assert 'cgroup_membership' in workspace._GIL_HELD_CLASSES


def test_gil_held_errors_propagate_count_and_time(monkeypatch, fresh_state):
    clock = [0.0]
    monkeypatch.setattr(workspace, '_perf_counter', lambda: clock[0])

    def failing(path):
        clock[0] += 1.0
        raise PermissionError(13, 'denied', str(path))
    monkeypatch.setattr(workspace, '_GIL_READER', failing)
    monkeypatch.setattr(workspace, '_gil_releasing_read_text', failing)
    with pytest.raises(PermissionError):
        workspace._probe_read_text(Path('/sys/fs/cgroup/memory.stat'))
    stats = workspace.ram_probe_gil_read_stats()
    assert stats['ram_probe_gil_held_reads_by_class'] == {'memory_stat': 1}
    assert 'memory_stat' in stats['ram_probe_gil_held_demoted']
    with pytest.raises(PermissionError):  # now the releasing path; same error
        workspace._probe_read_text(Path('/sys/fs/cgroup/memory.stat'))
    assert workspace.ram_probe_gil_read_stats()['ram_probe_gil_held_reads'] == 1


def _resolve(monkeypatch, env):
    monkeypatch.setattr(workspace, '_GIL_READER', workspace._GIL_READER_UNRESOLVED)
    monkeypatch.setattr(workspace, '_gil_holding_text_reader', lambda: (lambda p: 'h'))
    for key in ('YOLO_TTA_RAM_PROBE_GIL_READS', 'YOLO_TTA_RAM_PROBE_GIL_HOLD_LIMIT_MS'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return workspace._resolve_gil_holding_reader()


def test_counters_mode_never_holds_the_gil_for_lock_prone_files(monkeypatch, fresh_state):
    _resolve(monkeypatch, {'YOLO_TTA_RAM_PROBE_GIL_READS': 'counters'})
    assert workspace._GIL_HELD_CLASSES == set(workspace._LOCK_FREE_PROBE_CLASSES)
    released = []
    monkeypatch.setattr(workspace, '_gil_releasing_read_text',
                        lambda p: released.append(p.as_posix()) or 'r')
    for name, kind in CLASSES.items():
        assert workspace._probe_read_text(Path(name)) == ('r' if kind in LOCK_PRONE else 'h')
    assert sorted(released) == sorted(n for n, k in CLASSES.items() if k in LOCK_PRONE)


def test_hold_limit_defaults_to_switch_interval_and_env_overrides(monkeypatch, fresh_state):
    _resolve(monkeypatch, {})
    assert workspace._GIL_HOLD_LIMIT_SECONDS == sys.getswitchinterval()
    assert workspace._GIL_HELD_CLASSES == set(workspace._LOCK_FREE_PROBE_CLASSES
                                              | workspace._LOCK_PRONE_PROBE_CLASSES)
    _resolve(monkeypatch, {'YOLO_TTA_RAM_PROBE_GIL_HOLD_LIMIT_MS': '2.5'})
    assert workspace._GIL_HOLD_LIMIT_SECONDS == pytest.approx(0.0025)
    _resolve(monkeypatch, {'YOLO_TTA_RAM_PROBE_GIL_HOLD_LIMIT_MS': 'junk'})
    assert workspace._GIL_HOLD_LIMIT_SECONDS == sys.getswitchinterval()


def test_disabled_reader_leaves_classes_untouched(monkeypatch, fresh_state):
    monkeypatch.setattr(workspace, '_GIL_READER', workspace._GIL_READER_UNRESOLVED)
    monkeypatch.setattr(workspace, '_gil_holding_text_reader', lambda: None)
    monkeypatch.setenv('YOLO_TTA_RAM_PROBE_GIL_READS', 'counters')
    assert workspace._resolve_gil_holding_reader() is None
    released = []
    monkeypatch.setattr(workspace, '_gil_releasing_read_text',
                        lambda p: released.append(p.as_posix()) or 'r')
    assert workspace._probe_read_text(Path('/proc/meminfo')) == 'r'
    assert released == ['/proc/meminfo']


def test_full_probe_values_unchanged_under_random_demotions(monkeypatch, fresh_state):
    """Values equal the old probe whichever path (held or released) serves each file."""
    rng = random.Random(77)
    clock = [0.0]
    monkeypatch.setattr(workspace, '_perf_counter', lambda: clock[0])
    files, reads = {}, []
    base_held, base_released = dict_reader(files, reads), dict_reader(files, reads)

    def held(path):
        if rng.random() < .02:
            clock[0] += 1.0  # random over-limit read
        return base_held(path)
    monkeypatch.setattr(workspace, '_GIL_READER', held)
    monkeypatch.setattr(workspace, '_gil_releasing_read_text', base_released)
    old = dict_reader(files)
    for _ in range(300):
        files.clear()
        files.update(random_tree(rng))
        files['/proc/self/mountinfo'] = '29 1 0:26 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n'
        workspace._reset_cgroup_layout_cache()
        reads.clear()
        new = (workspace.available_anon_work_bytes(), workspace._cgroup_memory_budget())
        assert_membership_resolved_for_this_tree(files, reads)
        with monkeypatch.context() as patched:
            patched.setattr(Path, 'read_text', old)
            workspace._reset_cgroup_layout_cache()  # the reference resolves membership itself
            assert new ==(old_available_anon_work_bytes(), workspace._cgroup_memory_budget())
    assert workspace.ram_probe_gil_read_stats()['ram_probe_gil_held_demoted']
