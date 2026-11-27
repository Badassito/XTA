"""CPU-only integration tests for the synchronous native RAM file sampler."""
from contextlib import contextmanager
import errno
import locale
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

import pytest

import XTA
from XTA.native_cpu import kernels


requires_linux = pytest.mark.skipif(sys.platform != "linux", reason="uses Linux procfs and FIFO contracts")
_TIMEOUT = 10.0
_ENCODING = locale.getpreferredencoding(False)
_DIGIT_LIMIT = getattr(sys, "get_int_max_str_digits", lambda: 0)()


def _sample(meminfo, groups=(), mountinfo="/proc/self/mountinfo"):
    return kernels.sample_memory_files(meminfo, groups, mountinfo, _ENCODING, _DIGIT_LIMIT)


def _only_group(sample):
    info, groups, failed = sample
    assert not failed
    [group] = groups
    mounts, levels = group
    return info, mounts, tuple(levels)


def test_malformed_membership_path_fails_at_the_limit_read(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(b"MemAvailable: 17 kB\n")
    malformed = str(tmp_path / "bad") + "\0node"
    sample = _sample(meminfo, ((True, False, (malformed,)),))
    assert sample == (b"MemAvailable: 17 kB\n", ((None, ()),), True)


def test_path_conversion_preserves_exception_from_path_provider():
    error = UnicodeEncodeError("ascii", "\N{LATIN SMALL LETTER E WITH ACUTE}", 0, 1, "path provider failed")

    class FailingPath:
        def __fspath__(self):
            raise error

    with pytest.raises(UnicodeEncodeError) as raised:
        _sample(None, ((True, False, (FailingPath(),)),))
    assert raised.value is error


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX process locale and filesystem encodings")
def test_unencodable_cgroup_path_keeps_earlier_meminfo_and_fails_closed():
    # The filesystem encoding is fixed at Python startup; procfs text follows
    # the caller's current locale, which may later switch from ASCII to UTF-8.
    program = r"""
import locale, sys, tempfile
from pathlib import Path
from XTA import workspace

assert sys.getfilesystemencoding() == "ascii"
for utf8_locale in ("C.UTF-8", "C.utf8"):
    try:
        locale.setlocale(locale.LC_CTYPE, utf8_locale)
        break
    except locale.Error:
        pass
else:
    raise SystemExit(77)
with tempfile.TemporaryDirectory(prefix="xta-memory-encoding-", dir="/tmp") as folder:
    root = Path(folder)
    workspace._MEMINFO_PATH = root / "meminfo"
    workspace._MEMINFO_PATH.write_bytes(b"MemAvailable: 10 kB\n")
    workspace._CGROUP_MEMBERSHIP_PATH = root / "membership"
    workspace._CGROUP_MEMBERSHIP_PATH.write_bytes(b"0::/job-\xc3\xa9\n")
    workspace._CGROUP_V2_BASE = root / "cgroups"
    workspace._CGROUP_V2_BASE.mkdir()
    (workspace._CGROUP_V2_BASE / "memory.max").write_bytes(b"16777216\n")
    (workspace._CGROUP_V2_BASE / "memory.current").write_bytes(b"0\n")
    for calculate in (workspace._portable_memory_sample, workspace._native_memory_sample):
        workspace._reset_cgroup_layout_cache()
        assert calculate(include_meminfo=True, include_cgroup=True) == ({"MemAvailable": 10240}, 0, None)
"""
    environment = dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("No UTF-8 C locale is installed")
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


@pytest.mark.parametrize("v2", [False, True])
def test_native_sample_reads_dynamic_files_on_every_call(tmp_path, v2):
    leaf, root = tmp_path / "root" / "leaf", tmp_path / "root"
    leaf.mkdir(parents=True)
    meminfo, mountinfo = tmp_path / "meminfo", tmp_path / "mountinfo"
    limit_name, usage_name = (("memory.max", "memory.current") if v2 else
                             ("memory.limit_in_bytes", "memory.usage_in_bytes"))
    stat = (b"inactive_file 100\nfile_dirty 1\nfile_writeback 2\n" if v2 else
            b"total_inactive_file 100\ntotal_dirty 1\ntotal_writeback 2\n")
    meminfo.write_bytes(b"MemAvailable: 10 kB\n")
    mountinfo.write_bytes(b"initial mount snapshot\n")
    for node, limit, usage in ((leaf, b"125\n", b"10\n"), (root, b"200\n", b"70\n")):
        (node / limit_name).write_bytes(limit)
        (node / usage_name).write_bytes(usage)
        (node / "memory.stat").write_bytes(stat)
        if not v2:
            (node / "memory.use_hierarchy").write_bytes(b"1\n")
    layout = ((v2, True, (leaf, root)),)
    assert _only_group(_sample(meminfo, layout, mountinfo)) == (
        b"MemAvailable: 10 kB\n", b"initial mount snapshot\n",
        ((b"125\n", b"10\n", True, stat), (b"200\n", b"70\n", True, stat)),
    )

    # A finite zero survives. The unlimited leaf's missing usage does not fail
    # the sample. The same layout is reused after every underlying value changes.
    meminfo.write_bytes(b"MemAvailable: 7 kB\n")
    mountinfo.write_bytes(b"updated mount snapshot\n")
    (leaf / limit_name).write_bytes(b"max\n" if v2 else b"9223372036854771712\n")
    (leaf / usage_name).unlink()
    (root / limit_name).write_bytes(b"0\n")
    (root / usage_name).write_bytes(b"0\n")
    (root / "memory.stat").unlink()
    if not v2:
        (root / "memory.use_hierarchy").write_bytes(b"0\n")
    assert _only_group(_sample(meminfo, layout, mountinfo)) == (
        b"MemAvailable: 7 kB\n", b"updated mount snapshot\n",
        ((b"0\n", b"0\n", v2, None),),
    )


def test_native_sample_reads_meminfo_before_cold_layout_discovery(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(b"before layout\n")
    calls = []

    def discover():
        calls.append("discover")
        meminfo.write_bytes(b"after layout\n")
        return ()

    assert _sample(meminfo, discover) == (b"before layout\n", (), False)
    assert calls == ["discover"]
    assert _sample(meminfo) == (b"after layout\n", (), False)


def test_native_sample_handles_long_readonly_content_and_optional_read_errors(tmp_path):
    meminfo = tmp_path / "mem info Ω"
    mountinfo = tmp_path / "unreadable-mountinfo"
    mountinfo.mkdir()  # Directories cannot be read as sampling files, even as root.
    long_content = b"MemAvailable: 42 kB\r\n" + b"counter 1\n" * 8000
    meminfo.write_bytes(long_content)
    meminfo.chmod(0o444)
    layout = ((True, True, (tmp_path,)),)
    try:
        assert _only_group(_sample(meminfo, layout, mountinfo)) == (long_content, None, ())
        assert _only_group(_sample(mountinfo, layout, mountinfo)) == (None, None, ())
        assert _sample(None) == (None, (), False)
        assert meminfo.read_bytes() == long_content
    finally:
        meminfo.chmod(0o644)


def test_native_sample_resumes_fresh_after_unreadable_finite_usage(tmp_path):
    cap, usage = tmp_path / "memory.max", tmp_path / "memory.current"
    cap.write_bytes(b"4096\n")
    usage.mkdir()
    layout = ((True, False, (tmp_path,)),)
    assert _sample(None, layout)[2]

    usage.rmdir()
    usage.write_bytes(b"32\n")
    assert _only_group(_sample(None, layout)) == (None, None, ((b"4096\n", b"32\n", True, None),))


@pytest.mark.parametrize("v2", [False, True])
def test_native_and_portable_policy_agree_on_unicode_and_large_limits(tmp_path, monkeypatch, v2):
    from XTA import workspace

    # Select a known text encoding for both readers even on an ASCII-only host.
    read_text = Path.read_text

    def utf8_read(path, encoding=None, *args, **kwargs):
        return read_text(path, encoding or "utf-8", *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", utf8_read)
    monkeypatch.setattr(workspace, "_default_encoding", lambda: "utf-8")
    monkeypatch.setattr(workspace, "_cgroup_layout", lambda: (((v2, tmp_path, (str(tmp_path),)),), False))
    monkeypatch.setattr(workspace, "_CGROUP_LAYOUT", None)
    monkeypatch.setattr(workspace, "_CGROUP_MOUNT_VERIFIED", {(os.getpid(), tmp_path, v2): True})
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(b"MemAvailable: 17 kB\n")
    monkeypatch.setattr(workspace, "_MEMINFO_PATH", meminfo)

    if v2:
        cap = (1 << 80) + 17
        usage = (1 << 70) + 6
        (tmp_path / "memory.max").write_bytes(f"\u00a0{cap}\u00a0\n".encode())
        (tmp_path / "memory.current").write_bytes(f"\u00a0{usage}\u00a0\n".encode())
        (tmp_path / "memory.stat").write_bytes(b"inactive_file 7\nfile_dirty 1\nfile_writeback 2\n")
        expected_headroom, expected_cap = cap - usage + 4, cap
    else:
        # A v1 value above the unlimited sentinel must stop before reading usage.
        sentinel = (1 << 60) + 4096
        (tmp_path / "memory.limit_in_bytes").write_bytes(f"\u00a0{sentinel}\u00a0\n".encode())
        (tmp_path / "memory.usage_in_bytes").mkdir()
        expected_headroom = expected_cap = None

    args = dict(include_meminfo=True, include_cgroup=True, verify_enforcement=True)
    expected = ({"MemAvailable": 17 * 1024}, expected_headroom, expected_cap)
    assert workspace._portable_memory_sample(**args) == expected
    assert workspace._native_memory_sample(**args) == expected


@contextmanager
def _linux_fifo():
    # Use a Linux filesystem even when the test worktree and TMPDIR are on a
    # mounted Windows drive. The fixture owns only this isolated FIFO.
    with tempfile.TemporaryDirectory(prefix="xta-native-memory-", dir="/tmp") as directory:
        path = Path(directory) / "sample"
        os.mkfifo(path)
        yield path


@contextmanager
def _sample_child(fifo, code):
    bootstrap = (
        "import sys\n"
        "from pathlib import Path\n"
        "from XTA.native_cpu import kernels\n"
        "fifo = Path(sys.argv[1])\n"
        "sample = kernels.sample_memory_files\n"
    )
    with subprocess.Popen(
        [sys.executable, "-B", "-c", bootstrap + code, os.fspath(fifo)],
        cwd=Path(XTA.__file__).resolve().parents[1],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as child:
        try:
            yield child
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=_TIMEOUT)


@contextmanager
def _open_fifo_writer(child, path):
    deadline = time.monotonic() + _TIMEOUT
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as error:
            if error.errno != errno.ENXIO:
                raise
            assert child.poll() is None, child.stderr.read()
            assert time.monotonic() < deadline, "native child never opened the FIFO reader"
            time.sleep(0.01)
    try:
        yield fd
    finally:
        os.close(fd)


def _line(child, expected):
    assert select.select([child.stdout], [], [], _TIMEOUT)[0], (
        f"timed out waiting for child output {expected!r}"
    )
    assert child.stdout.readline() == expected


def _finish(child):
    output, errors = child.communicate(timeout=_TIMEOUT)
    assert child.returncode == 0, errors.decode(errors="replace")
    return output


@requires_linux
def test_blocked_native_read_leaves_interpreter_available():
    script = """
from threading import Thread
results = []
reader = Thread(target=lambda: results.append(sample(fifo, (), fifo, "utf-8", 0)))
reader.start()
assert sys.stdin.buffer.read(1) == b"g"
assert reader.is_alive()
print("progress", flush=True)
reader.join()
assert results == [(b"sampled data\\n", (), False)]
print("finished", flush=True)
"""
    with _linux_fifo() as fifo, _sample_child(fifo, script) as child:
        with _open_fifo_writer(child, fifo) as writer:
            # Opening the writer proves the native reader is present. Keep the
            # FIFO dry until another thread in the child makes Python progress.
            child.stdin.write(b"g")
            child.stdin.flush()
            _line(child, b"progress\n")
            os.write(writer, b"sampled data\n")
        assert _finish(child) == b"finished\n"


@requires_linux
def test_finite_limit_with_unreadable_usage_uses_first_observation():
    script = """
layout = ((True, False, (fifo.parent,)),)
_, _, failed = sample(None, layout, fifo.parent / "mountinfo", "utf-8", 0)
assert failed
print("failed closed", flush=True)
"""
    with _linux_fifo() as fifo:
        limit = fifo.rename(fifo.with_name("memory.max"))
        (limit.parent / "memory.current").mkdir()
        with _sample_child(limit, script) as child:
            with _open_fifo_writer(child, limit) as writer:
                os.write(writer, b"4096\n")
            # Any attempt to reopen the already observed finite limit needs a
            # second writer, which this test never provides.
            assert _finish(child) == b"failed closed\n"


@pytest.mark.parametrize("raises", [False, True])
@requires_linux
def test_native_read_handles_interrupted_syscall(raises):
    script = f"""
import signal
observed = []
class ProbeInterrupted(Exception):
    pass
def handler(signum, frame):
    observed.append(signum)
    if {raises!r}:
        raise ProbeInterrupted("handler ran")
signal.signal(signal.SIGUSR1, handler)
signal.siginterrupt(signal.SIGUSR1, True)
try:
    result = sample(fifo, (), fifo, "utf-8", 0)
except ProbeInterrupted:
    assert {raises!r}
else:
    assert not {raises!r}
    assert result == (b"data after signal\\n", (), False)
assert observed
print("signal handled", flush=True)
"""
    with _linux_fifo() as fifo, _sample_child(fifo, script) as child:
        with _open_fifo_writer(child, fifo) as writer:
            # Disable SA_RESTART explicitly in the child. For the raising
            # handler, retry delivery if the first signal lands before read.
            deadline = time.monotonic() + _TIMEOUT
            while child.poll() is None:
                try:
                    os.kill(child.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    break
                if select.select([child.stdout], [], [], 0.05)[0]:
                    break
                if not raises:
                    os.write(writer, b"data after signal\n")
                    break
                assert time.monotonic() < deadline, "raising handler did not interrupt the native read"
        assert _finish(child) == b"signal handled\n"


def _finite_limits(nodes, is_v2):
    name = "memory.max" if is_v2 else "memory.limit_in_bytes"
    limits = []
    for node in nodes:
        try:
            text = (node / name).read_bytes()
        except FileNotFoundError:
            continue
        raw = text.strip()
        if is_v2 and raw == b"max":
            continue
        limit = int(raw)
        if is_v2 or limit < 1 << 60:
            limits.append(limit)
    return tuple(limits)


@requires_linux
def test_native_sample_real_linux_proc_and_cgroup():
    from XTA import workspace

    meminfo = Path("/proc/meminfo")
    entries, _ = workspace._cgroup_layout()
    layout = tuple((v2, False, tuple(map(Path, nodes))) for v2, _, nodes in entries)
    stable_keys = (b"MemTotal:", b"SwapTotal:")

    def stable_values(data):
        return {fields[0]: int(fields[1]) for line in data.splitlines()
                if len(fields := line.split()) >= 2 and fields[0] in stable_keys}

    before = tuple(_finite_limits(nodes, v2) for v2, _, nodes in layout)
    info, groups, failed = _sample(meminfo, layout)
    assert not failed
    assert len(groups) == len(layout)
    stable = stable_values(info)
    assert stable[b"MemTotal:"] > 0
    assert stable == stable_values(meminfo.read_bytes())
    assert any(line.startswith(b"MemAvailable:") and int(line.split()[1]) >= 0
               for line in info.splitlines())
    after = tuple(_finite_limits(nodes, v2) for v2, _, nodes in layout)
    for ((mount, levels), limits_before, limits_after) in zip(groups, before, after):
        assert mount is None
        if limits_before == limits_after:
            assert tuple(int(limit) for limit, _, _, _ in levels) == limits_after
        # Usage and reclaimable bytes can move between observational reads.
        for _, usage, hierarchical, stat in levels:
            assert int(usage) >= 0
            assert isinstance(hierarchical, bool)
            assert stat is None or isinstance(stat, bytes)
