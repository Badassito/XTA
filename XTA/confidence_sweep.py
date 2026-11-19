"""Isolated scalar TTA runs for an exact requested confidence family.

The producer fans out native prediction products to all requested cutoffs.
Later fresh scalar processes consume validated native products while retaining
independent cleanup, interpolation, SAM and publication state. Unsupported
upstream routes use an explicitly recorded scalar fallback.

When the producer recorded no detector scalar fallback, consumers spawn no
detector workers, so up to XTA_YOLO_SWEEP_CONSUMER_PARALLEL of them (a positive
integer, or auto for a memory-based limit) run at once with prefixed output.
Otherwise, and whenever fanout is disabled, children run one after another.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import locale
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from uuid import uuid4

from . import __version__
from .json_publication import write_json_atomic
from .unification.tta_manifest import (
    TTA_RUN_MANIFEST_SCHEMA,
    _file_identity,
    assert_tta_artifacts_unchanged,
)


SCHEMA = 'xta.confidence_sweep/1'
_REPLACED_OPTIONS = {'--conf', '--output', '--temp', '--min_conf', '--capture_component_replay'}
_IDENTITY_FIELDS = ('path', 'size_bytes', 'modified_time_ns',
                    'change_or_creation_time_ns', 'device', 'file_id')
_OPTIONAL_IDENTITIES = ('sam_checkpoint_identity_sha256', 'augmentation_policy_sha256',
                        'reconciliation_policy_sha256')
_SCALAR_OUTPUTS = ('confidence_cutoffs','review','manifest.json')
_RESERVED_OUTPUTS = (*_SCALAR_OUTPUTS,'fanout_progress.json','fanout_storage.json')
_OPTIONAL_FANOUT_COUNTS = ('native_frames','stored_native_bytes','native_views_retired_after_projection')
_CHILD_SETTLE_SECONDS = 2.
_CHILD_GRACE_SECONDS = 10.
_CHILD_KILL_SECONDS = 5.
_CONSUMER_PARALLEL_ENV = 'XTA_YOLO_SWEEP_CONSUMER_PARALLEL'
_DEFER_NATIVE_RETIREMENT_ENV = 'XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT'
# A consumer's union and final tail can hold about three source-sized volumes.
_CONSUMER_SOURCE_VOLUME_FACTOR = 3
_CONSUMER_POLL_SECONDS = .1
_OUTPUT_CHUNK_BYTES = 64*1024
_OUTPUT_LINE_LIMIT = 1024*1024
_OUTPUT_JOIN_SECONDS = 10.
_OUTPUT_LOCK = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _cutoffs(values):
    result = tuple(values)
    if not result or any(isinstance(value, bool) or not isinstance(value, int)
                         or not 0 <= value <= 100 for value in result):
        raise ValueError('Confidence sweep cutoffs must be integer hundredths in [0,100]')
    if len(set(result)) != len(result):
        raise ValueError('Confidence sweep cutoffs must be unique')
    return result


def _base_arguments(arguments):
    """Remove exact single-token scalar options, including every = variant."""
    result, capture = [], None
    tokens = tuple(str(value) for value in arguments)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name, separator, value = token.partition('=')
        if name not in _REPLACED_OPTIONS:
            result.append(token)
            index += 1
            continue
        if not separator:
            index += 1
            if index >= len(tokens) or tokens[index].startswith('--'):
                raise ValueError(f'{name} requires one value')
            value = tokens[index]
        if not value:
            raise ValueError(f'{name} requires one value')
        if name == '--capture_component_replay':
            capture = Path(value).expanduser().resolve()
        index += 1
    return result, capture


def _check_work_destination(path, output_root, description, reserved_outputs=_RESERVED_OUTPUTS):
    destination = path.resolve()
    for name in reserved_outputs:
        reserved = output_root / name
        try:
            destination.relative_to(reserved.resolve())
        except ValueError:
            continue
        raise ValueError(f'{description} cannot be inside reserved confidence sweep output {reserved}')


def _child_command(arguments, package_parent):
    if sys.version_info >= (3,11):
        return [sys.executable,'-m','XTA','--mode','tta',*arguments]
    bootstrap = ("import sys; sys.path[0] = sys.argv.pop(1); import runpy; "
                 "runpy.run_module('XTA', run_name='__main__', alter_sys=True)")
    return [sys.executable,'-c',bootstrap,package_parent,'--mode','tta',*arguments]


@contextmanager
def _cancel_signals():
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def cancel(signum, _frame):
            raise KeyboardInterrupt(f'Confidence sweep received signal {int(signum)}')
        for name in ('SIGINT', 'SIGTERM'):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous[signum] = signal.signal(signum, cancel)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class _ChildTreeUnsettled(RuntimeError):
    """Shared scratch must remain owned when a process tree cannot be settled."""


class _PosixProcessTree:
    def __init__(self, process):
        self.process = process
        self.pgid = process.pid

    def active(self):
        self.process.poll()
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        # Zombies cannot read/write shared work. Linux containers can retain an
        # adopted zombie until their init reaps it; do not confuse it with work.
        proc = Path('/proc')
        if proc.is_dir():
            for directory in proc.iterdir():
                if not directory.name.isdigit():
                    continue
                try:
                    fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == self.pgid and fields[0] not in {'Z', 'X'}:
                        return True
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, IndexError):
                    return True
            return False
        return True

    def signal(self, force=False):
        try:
            os.killpg(self.pgid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def close(self):
        pass


class _WindowsProcessTree:
    """A kill-on-close Job Object, assigned before the scalar thread resumes."""
    creationflags = 0x00000004 | 0x00000200  # CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP

    def __init__(self):
        import ctypes
        from ctypes import wintypes as wt
        self.ctypes, self.wt = ctypes, wt
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        class BasicLimits(ctypes.Structure):
            _fields_ = [('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                ('flags',wt.DWORD),('minimum_working_set',ctypes.c_size_t),
                ('maximum_working_set',ctypes.c_size_t),('active_limit',wt.DWORD),
                ('affinity',ctypes.c_size_t),('priority',wt.DWORD),('scheduling',wt.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name,ctypes.c_ulonglong) for name in (
                'read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('basic',BasicLimits),('io',IoCounters),
                ('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                ('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]
        class Accounting(ctypes.Structure):
            _fields_ = [(name,ctypes.c_longlong) for name in (
                'user_time','kernel_time','period_user_time','period_kernel_time')] + [
                ('faults',wt.DWORD),('total_processes',wt.DWORD),
                ('active_processes',wt.DWORD),('terminated_processes',wt.DWORD)]
        class ThreadEntry(ctypes.Structure):
            _fields_ = [('size',wt.DWORD),('usage',wt.DWORD),('thread_id',wt.DWORD),
                ('process_id',wt.DWORD),('base_priority',wt.LONG),
                ('delta_priority',wt.LONG),('flags',wt.DWORD)]
        self.Accounting, self.ThreadEntry = Accounting, ThreadEntry
        signatures = {
            'CreateJobObjectW':([ctypes.c_void_p,wt.LPCWSTR],wt.HANDLE),
            'SetInformationJobObject':([wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD],wt.BOOL),
            'QueryInformationJobObject':([wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD,ctypes.c_void_p],wt.BOOL),
            'AssignProcessToJobObject':([wt.HANDLE,wt.HANDLE],wt.BOOL),
            'TerminateJobObject':([wt.HANDLE,wt.UINT],wt.BOOL),
            'OpenProcess':([wt.DWORD,wt.BOOL,wt.DWORD],wt.HANDLE),
            'CreateToolhelp32Snapshot':([wt.DWORD,wt.DWORD],wt.HANDLE),
            'Thread32First':([wt.HANDLE,ctypes.POINTER(ThreadEntry)],wt.BOOL),
            'Thread32Next':([wt.HANDLE,ctypes.POINTER(ThreadEntry)],wt.BOOL),
            'OpenThread':([wt.DWORD,wt.BOOL,wt.DWORD],wt.HANDLE),
            'ResumeThread':([wt.HANDLE],wt.DWORD),
            'CloseHandle':([wt.HANDLE],wt.BOOL),
        }
        for name,(arguments,result) in signatures.items():
            function = getattr(self.kernel,name)
            function.argtypes, function.restype = arguments,result
        self.job = self.kernel.CreateJobObjectW(None,None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error
        self.process = None
        self.attached = False
        self.resumed = False

    def attach_and_resume(self, process):
        self.process = process
        c, k = self.ctypes, self.kernel
        handle = k.OpenProcess(0x0100 | 0x0001,False,process.pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise c.WinError(c.get_last_error())
        try:
            if not k.AssignProcessToJobObject(self.job,handle):
                raise c.WinError(c.get_last_error())
            self.attached = True
        finally:
            k.CloseHandle(handle)
        # CREATE_SUSPENDED guarantees no Python/worker code has run before job
        # assignment. Enumerate its primary thread using documented Toolhelp API.
        snapshot = k.CreateToolhelp32Snapshot(0x00000004,0)  # TH32CS_SNAPTHREAD
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        resumed = False
        try:
            entry = self.ThreadEntry()
            entry.size = c.sizeof(entry)
            available = k.Thread32First(snapshot,c.byref(entry))
            while available:
                if entry.process_id == process.pid:
                    thread = k.OpenThread(0x0002,False,entry.thread_id)  # SUSPEND_RESUME
                    if not thread:
                        raise c.WinError(c.get_last_error())
                    try:
                        if k.ResumeThread(thread) == 0xffffffff:
                            raise c.WinError(c.get_last_error())
                        resumed = True
                        self.resumed = True
                    finally:
                        k.CloseHandle(thread)
                    break
                available = k.Thread32Next(snapshot,c.byref(entry))
        finally:
            k.CloseHandle(snapshot)
        if not resumed:
            raise RuntimeError('Suspended scalar primary thread was not found')

    def active(self):
        accounting = self.Accounting()
        if not self.kernel.QueryInformationJobObject(self.job,1,self.ctypes.byref(accounting),self.ctypes.sizeof(accounting),None):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return accounting.active_processes != 0

    def signal(self, force=False):
        if force or not self.resumed:
            if not self.kernel.TerminateJobObject(self.job,1):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
        elif self.process is not None and self.process.poll() is None:
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                pass  # No attached console: the bounded job deadline remains.

    def close(self):
        if self.job:
            if not self.kernel.CloseHandle(self.job):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            self.job = None


def _wait_tree(tree, timeout):
    deadline = time.monotonic()+timeout
    while tree.active():
        if time.monotonic() >= deadline:
            return False
        time.sleep(.05)
    return True


def _stop_child(process, tree):
    """Prove the owned tree is quiescent before sharing scratch can be deleted."""
    tree.signal()
    if not _wait_tree(tree,_CHILD_GRACE_SECONDS):
        tree.signal(force=True)
        if not _wait_tree(tree,_CHILD_KILL_SECONDS):
            raise _ChildTreeUnsettled(f'Scalar process tree {process.pid} did not stop; owned work retained')
    process.wait(timeout=5.)


@contextmanager
def _settling_signals():
    """Repeated parent signals cannot interrupt the bounded tree shutdown."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ('SIGINT','SIGTERM'):
            signum = getattr(signal,name,None)
            if signum is not None:
                previous[signum] = signal.signal(signum,signal.SIG_IGN)
    try:
        yield
    finally:
        for signum,handler in previous.items():
            signal.signal(signum,handler)


def _child_popen_options(env, tree, *, capture_output=False):
    options = {'env': env}
    if tree is not None:
        options['creationflags'] = tree.creationflags
    elif os.name == 'posix':
        options['start_new_session'] = True
    if capture_output:
        options.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return options


def _tree_owned(tree):
    """False until a Windows job holds the suspended child (it has no descendants yet)."""
    return tree is not None and not (isinstance(tree,_WindowsProcessTree) and not tree.attached)


def _abort_child(process, tree):
    """Stop a started scalar tree after an error; unproved shutdown retains owned work."""
    try:
        with _settling_signals():
            if isinstance(tree,_WindowsProcessTree) and not tree.attached:
                # Assignment failed while CREATE_SUSPENDED still held
                # the only thread; this process has no descendants.
                process.terminate()
                process.wait(timeout=5.)
            elif tree is not None:
                _stop_child(process,tree)
            elif process.poll() is None:
                process.terminate()
                process.wait(timeout=5.)
    except BaseException as stop_error:
        raise _ChildTreeUnsettled(f'Cannot establish shutdown of scalar tree {process.pid}; owned work retained') from stop_error


def _release_tree(tree):
    if tree is not None:
        try:
            tree.close()
        except Exception as error:
            raise _ChildTreeUnsettled('Scalar process tree ownership could not be released; owned work retained') from error


def _run_child(command, env, *, started=None):
    tree = _WindowsProcessTree() if os.name == 'nt' else None
    options = _child_popen_options(env, tree)
    process = None
    try:
        process = subprocess.Popen(command, **options)
        if tree is not None:
            tree.attach_and_resume(process)
        else:
            tree = _PosixProcessTree(process)
        if started is not None:
            started(process.pid)
        launched = time.monotonic()
        while True:
            try:
                returncode = process.wait(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                continue
        exited = time.monotonic()
        if not _wait_tree(tree,_CHILD_SETTLE_SECONDS):
            _stop_child(process,tree)
            raise RuntimeError('Scalar cutoff exited with live descendants; refusing completion')
        # Compare with the child's own "End-to-end pipeline walltime" line: the
        # difference is interpreter/resource teardown after its pipeline ended.
        print(f'Scalar cutoff process {process.pid} lived {exited - launched:.1f}s; '
              f'descendants settled {time.monotonic() - exited:.1f}s after exit.', flush=True)
        return returncode
    except BaseException:
        if process is not None:
            _abort_child(process, tree)
        raise
    finally:
        _release_tree(tree)


def _emit_prefixed_lines(stream_name, prefix, lines, encoding):
    text = ''.join(prefix + (line[:-1] if line.endswith(b'\r') else line).decode(encoding, errors='replace') + '\n'
                   for line in lines)
    with _OUTPUT_LOCK:
        target = getattr(sys, stream_name, None)
        if target is None:
            return
        try:
            target.write(text)
            target.flush()
        except UnicodeEncodeError:
            target_encoding = getattr(target, 'encoding', None) or 'ascii'
            try:
                target.write(text.encode(target_encoding, errors='replace').decode(target_encoding))
                target.flush()
            except Exception:
                pass
        except Exception:
            pass  # A broken parent stream must not stop draining the child pipe.


def _forward_prefixed_output(stream, stream_name, prefix, encoding):
    """Drain one child pipe to EOF and copy each line to the parent stream with a prefix."""
    pending = bytearray()
    split_line = False
    read = getattr(stream, 'read1', stream.read)
    try:
        while True:
            try:
                chunk = read(_OUTPUT_CHUNK_BYTES)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            pending += chunk
            if b'\n' not in chunk and len(pending) < _OUTPUT_LINE_LIMIT:
                continue
            *lines, rest = bytes(pending).split(b'\n')
            if lines and split_line:
                # This newline ends an overlong line whose start was already forwarded.
                if lines[0] in (b'', b'\r'):
                    lines.pop(0)
                split_line = False
            pending = bytearray(rest)
            if len(pending) >= _OUTPUT_LINE_LIMIT:
                lines.append(bytes(pending))
                pending.clear()
                split_line = True
            if lines:
                _emit_prefixed_lines(stream_name, prefix, lines, encoding)
        if pending:
            _emit_prefixed_lines(stream_name, prefix, [bytes(pending)], encoding)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _say(text, *, file=None):
    """Coordinator lines stay whole while concurrent children are forwarded."""
    with _OUTPUT_LOCK:
        print(text, file=sys.stdout if file is None else file, flush=True)


class _ConcurrentChild:
    """One consumer running beside others; its owned tree is stopped as a unit.

    Output goes through pipes drained by daemon threads, so a full pipe can
    never block the child while the coordinator waits on another consumer.
    """

    def __init__(self, branch):
        self.branch = branch
        self.process = None
        self.tree = None
        self.readers = []

    def start(self, started=None):
        tree = _WindowsProcessTree() if os.name == 'nt' else None
        self.tree = tree
        self.process = subprocess.Popen(self.branch['command'],
            **_child_popen_options(self.branch['env'], tree, capture_output=True))
        if tree is None:
            self.tree = _PosixProcessTree(self.process)
        encoding = locale.getpreferredencoding(False)
        cent = self.branch['cent']
        for stream, name in ((self.process.stdout, 'stdout'), (self.process.stderr, 'stderr')):
            reader = threading.Thread(target=_forward_prefixed_output,
                args=(stream, name, self.branch['prefix'], encoding),
                name=f'confidence-sweep-{cent:03d}-{name}', daemon=True)
            reader.start()
            self.readers.append(reader)
        if tree is not None:
            tree.attach_and_resume(self.process)
        if started is not None:
            started(self.process.pid)

    def active(self):
        if _tree_owned(self.tree):
            return self.tree.active()
        return self.process.poll() is None

    def settle(self):
        """Return the exit code once the whole exited tree is quiescent."""
        returncode = self.process.wait(timeout=5.)
        if not _wait_tree(self.tree,_CHILD_SETTLE_SECONDS):
            _stop_trees([self])
            raise RuntimeError('Scalar cutoff exited with live descendants; refusing completion')
        return returncode

    def close(self):
        """Finish forwarding output, then release tree ownership; call only once quiescent."""
        readers, self.readers = self.readers, []
        deadline = time.monotonic()+_OUTPUT_JOIN_SECONDS
        for reader in readers:
            reader.join(timeout=max(0., deadline-time.monotonic()))
        if any(reader.is_alive() for reader in readers):
            try:
                _say(f"Warning: output of confidence cutoff {self.branch['cent'] / 100:.2f} is still open after "
                     'its process tree stopped; forwarding continues in the background.', file=sys.stderr)
            except Exception:
                pass
        tree, self.tree = self.tree, None
        _release_tree(tree)


def _await_quiescence(children, timeout):
    deadline = time.monotonic()+timeout
    live = list(children)
    while True:
        live = [child for child in live if child.active()]
        if not live or time.monotonic() >= deadline:
            return live
        time.sleep(.05)


def _stop_trees_unchecked(children):
    first_error = None
    signalled = []
    for child in children:
        if child.process is None:
            continue
        try:
            if _tree_owned(child.tree):
                child.tree.signal()
            elif child.process.poll() is None:
                child.process.terminate()
            signalled.append(child)
        except Exception as error:
            first_error = first_error or error
    live = _await_quiescence(signalled,_CHILD_GRACE_SECONDS)
    for child in live:
        try:
            if _tree_owned(child.tree):
                child.tree.signal(force=True)
            else:
                child.process.kill()
        except Exception as error:
            first_error = first_error or error
    live = _await_quiescence(live,_CHILD_KILL_SECONDS)
    for child in signalled:
        if child not in live:
            try:
                child.process.wait(timeout=5.)
            except Exception as error:
                first_error = first_error or error
    if live:
        pids = ', '.join(str(child.process.pid) for child in live)
        raise _ChildTreeUnsettled(f'Scalar process trees {pids} did not stop; owned work retained') from first_error
    if first_error is not None:
        raise first_error


def _stop_trees(children):
    """Stop several owned trees with _stop_child's grace and kill deadlines, shared by all."""
    with _settling_signals():
        try:
            _stop_trees_unchecked(children)
        except _ChildTreeUnsettled:
            raise
        except BaseException as error:
            raise _ChildTreeUnsettled('Cannot establish shutdown of concurrent scalar trees; owned work retained') from error


def _requested_consumer_parallel():
    """Return the explicit consumer limit, or None for the memory-based automatic limit."""
    value = os.environ.get(_CONSUMER_PARALLEL_ENV, 'auto')
    if value == 'auto':
        return None
    if not value.isascii() or not value.isdigit() or value[0] == '0' or len(value) > 9:
        raise ValueError(f'{_CONSUMER_PARALLEL_ENV} must be a positive integer or auto')
    return int(value)


def _windows_available_physical_bytes():
    import ctypes
    from ctypes import wintypes as wt
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length',wt.DWORD),('memory_load',wt.DWORD)] + [
            (name,ctypes.c_ulonglong) for name in ('total_physical','available_physical',
                'total_page_file','available_page_file','total_virtual','available_virtual',
                'available_extended_virtual')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.available_physical)


def _available_memory():
    """Return (bytes, sources) of physical memory new processes may use, or (None, ())."""
    available, source = None, None
    try:
        import psutil
        available, source = int(psutil.virtual_memory().available), 'psutil'
    except Exception:
        pass
    if available is None:
        try:
            for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
                name, _, value = line.partition(':')
                if name == 'MemAvailable':
                    available, source = int(value.split()[0])*1024, 'MemAvailable'
                    break
        except (OSError, ValueError, IndexError):
            pass
    if available is None and os.name == 'nt':
        try:
            available = _windows_available_physical_bytes()
            source = 'GlobalMemoryStatusEx' if available is not None else None
        except (OSError, AttributeError, ValueError):
            available = None
    if available is None:
        return None, ()
    sources = [source]
    if os.name == 'posix':
        # Batch schedulers enforce a cgroup limit that node-wide figures do not show.
        try:
            from .workspace import _cgroup_memory_headroom_bytes
            headroom = _cgroup_memory_headroom_bytes()
        except Exception:
            headroom = None
        if headroom is not None:
            sources.append('cgroup headroom')
            available = min(available, int(headroom))
    return max(0, available), tuple(sources)


def _shared_cube_missing(work, inputs):
    """Return True when every consumer needs a processing cube the producer did not share.

    A cube that resizes Y or X is never built lazily, so each consumer reads
    ``cube.gray8``. A T-only cube may be lazy (GPU residency) and then never
    publishes one; the parent cannot tell that case apart, and sweep_input keeps
    concurrent consumers branch-local there.
    """
    source, processing = inputs['source_shape_t_y_x'], inputs['processing_shape_t_y_x']
    return list(processing[1:]) != list(source[1:]) and not (work / 'cube.gray8.json').is_file()


def _consumer_execution(requested, consumer_count, producer_evidence, source_shape, *,
                        shared_source=True, shared_cube=True):
    """Decide how many consumer cutoffs may run at once after a validated producer."""
    record = {'consumers': consumer_count,
              'requested': 'auto' if requested is None else requested}
    if producer_evidence['scalar_views']:
        # Fallback views make consumers spawn detector workers on the same GPUs.
        record.update(limit=1, reason=f"producer recorded {producer_evidence['scalar_views']} "
                                      'detector scalar-fallback views')
    elif not shared_source:
        # The producer found too little scratch for one shared copy. Concurrent
        # consumers never build shared inputs (sweep_input), so each would
        # prepare its own private copy in that same scratch at once.
        record.update(limit=1, reason='producer published no shared decoded source volume')
    elif not shared_cube:
        # Same reasoning for a processing cube every consumer must materialise.
        record.update(limit=1, reason='producer published no shared processing cube')
    elif requested is not None:
        record.update(limit=min(requested,consumer_count), reason=f'{_CONSUMER_PARALLEL_ENV}={requested}')
    else:
        per_consumer = max(1,_CONSUMER_SOURCE_VOLUME_FACTOR*math.prod(source_shape))
        available, sources = _available_memory()
        record['estimated_bytes_per_consumer'] = per_consumer
        if available is None:
            record.update(limit=1, reason='auto: available physical memory is unknown')
        else:
            admitted = available//per_consumer
            record.update(limit=max(1,min(consumer_count,admitted)), available_memory_bytes=available,
                memory_sources=list(sources),
                reason=(f'auto: memory admits {admitted} at about {per_consumer / 1024**3:.1f} GiB each '
                        f'({_CONSUMER_SOURCE_VOLUME_FACTOR}x source volume) of '
                        f"{available / 1024**3:.1f} GiB available [{', '.join(sources)}]"))
    record['mode'] = 'concurrent' if record['limit'] > 1 else 'sequential'
    return record


def _identity_record(value, *, model=False):
    if not isinstance(value, dict) or any(field not in value for field in _IDENTITY_FIELDS):
        raise ValueError('Scalar confidence receipt is missing a complete artifact identity')
    if not isinstance(value['path'], str) or not value['path']:
        raise ValueError('Scalar artifact identity has no path')
    for field in _IDENTITY_FIELDS[1:]:
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ValueError(f'Scalar artifact identity has invalid {field}')
    if model:
        digest = value.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Scalar model identity is missing its SHA256')


def _receipt(path, *, sweep_id, cent, min_conf):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != TTA_RUN_MANIFEST_SCHEMA:
        raise ValueError(f'Invalid scalar TTA receipt: {path}')
    omissions = value.get('sam_planning_omissions')
    expected_complete = not omissions
    if (not isinstance(omissions, list) or
            not all(isinstance(item, dict) and item for item in omissions) or
            value.get('sam_requested_work_complete') is not expected_complete or
            value.get('status') != ('complete' if expected_complete else 'incomplete')):
        raise ValueError(f'Scalar cutoff {cent / 100:.2f} has an inconsistent or unfinished SAM status')
    launch = value.get('launcher', {})
    if launch.get('mode') != 'tta' or launch.get('version') != __version__ or launch.get('pipeline_version') != __version__:
        raise ValueError('Scalar cutoff launcher/pipeline identity differs from the sweep')
    ownership = value.get('confidence_sweep', {})
    if ownership.get('sweep_id') != sweep_id or ownership.get('cutoff_hundredths') != cent:
        raise ValueError('Scalar cutoff receipt belongs to another confidence sweep')
    config = value.get('resolved_configuration', {})
    for key, expected in (('conf', cent / 100), ('min_conf', min_conf)):
        observed = config.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (float, int)) or not math.isfinite(observed) or observed != expected:
            raise ValueError(f'Scalar cutoff receipt has incorrect {key}')
    inputs = value.get('inputs')
    if not isinstance(inputs, dict) or inputs.get('captured_before_execution') is not True:
        raise ValueError('Scalar cutoff receipt has no pre-execution input identity')
    _identity_record(inputs.get('source'))
    models = inputs.get('models')
    if not isinstance(models, dict) or not {'gpu', 'cpu'} <= set(models) or not any(models.values()):
        raise ValueError('Scalar cutoff receipt has incomplete detector model identities')
    for model in models.values():
        if model is None:
            continue
        if not isinstance(model, dict) or not isinstance(model.get('requested_path'), str) or not model.get('artifacts'):
            raise ValueError('Scalar cutoff receipt has incomplete model artifacts')
        for artifact in model['artifacts']:
            _identity_record(artifact, model=True)
    for name in ('source_shape_t_y_x', 'processing_shape_t_y_x'):
        shape = inputs.get(name)
        if not isinstance(shape, list) or len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
            raise ValueError(f'Scalar cutoff receipt has invalid {name}')
    if isinstance(inputs.get('fps'), bool) or not isinstance(inputs.get('fps'), (int, float)) or not math.isfinite(inputs['fps']) or inputs['fps'] <= 0:
        raise ValueError('Scalar cutoff receipt has invalid source FPS')
    paths = value.get('outputs', {}).get('paths')
    if not isinstance(paths, dict) or Path(str(paths.get('run_manifest', ''))).resolve() != path.resolve():
        raise ValueError('Scalar cutoff receipt has an incorrect output manifest path')
    normalized = {
        'inputs': inputs,
        'configuration': {key: item for key, item in config.items()
                          if key not in {'conf', 'output', 'temp', 'capture_component_replay'}},
        'geometry': value.get('geometry'),
        'additional_identities': {key: ownership.get(key) for key in _OPTIONAL_IDENTITIES},
    }
    return value, normalized, hashlib.sha256(raw).hexdigest()


def _fanout_evidence(receipt, *, role, target_count):
    """Accept upstream work only from a validated scalar product receipt."""
    evidence = receipt['confidence_sweep'].get('upstream_fanout')
    if not isinstance(evidence,dict) or evidence.get('role') != role or evidence.get('validated') is not True:
        raise ValueError('Scalar cutoff has no validated native fanout receipt for its assigned role')
    counts = {}
    for name in ('produced_views','consumed_views','scalar_views','target_count'):
        value = evidence.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f'Native fanout receipt has an invalid {name}')
        counts[name] = value
    if counts['target_count'] != target_count:
        raise ValueError('Native fanout receipt target count differs from the confidence family')
    reasons = evidence.get('fallback_reasons')
    if not isinstance(reasons,list) or any(not isinstance(reason,str) or not reason.strip() for reason in reasons):
        raise ValueError('Native fanout fallback reasons must be nonempty strings')
    active = counts['produced_views'] if role == 'produce' else counts['consumed_views']
    if (not active or counts['scalar_views']) and not reasons:
        raise ValueError('Native fanout scalar fallback or unsupported work requires an explicit reason')
    if role == 'consume' and counts['produced_views']:
        raise ValueError('A native fanout consumer unexpectedly produced new upstream views')
    result = {'role':role,'validated':True,**counts,'fallback_reasons':list(reasons)}
    for name in _OPTIONAL_FANOUT_COUNTS:
        if name in evidence:
            value = evidence[name]
            if type(value) is not int or value < 0:
                raise ValueError(f'Native fanout receipt has an invalid {name}')
            result[name] = value
    for name in ('work','projection_fallbacks'):
        if name in evidence:
            value = evidence[name]
            if (not isinstance(value,dict) or any(not isinstance(key,str) or not key.strip()
                    or type(count) is not int or count < 0 for key,count in value.items())):
                raise ValueError(f'Native fanout receipt has an invalid {name} counter map')
            result[name] = dict(value)
    publication = evidence.get('early_publication')
    if publication is not None:
        # Diagnostic only: shows whether publication, not inference, paced the producer.
        if (not isinstance(publication, dict) or any(not isinstance(key, str) or isinstance(value, bool)
                or not isinstance(value, (int, float)) or value < 0 for key, value in publication.items())):
            raise ValueError('Native fanout receipt has an invalid early_publication record')
        result['early_publication'] = dict(publication)
    return result


def _update_fanout_summary(manifest):
    fanout = manifest['upstream_fanout']
    evidence = [row['upstream_fanout'] for row in manifest['branches'] if 'upstream_fanout' in row]
    fanout['validated_arms'] = len(evidence)
    for name in ('produced_views','consumed_views','scalar_views'):
        fanout[name] = sum(item[name] for item in evidence)
    optional_arms = {}
    for name in _OPTIONAL_FANOUT_COUNTS:
        measured = [item[name] for item in evidence if name in item]
        if measured:
            fanout[name] = sum(measured)
            optional_arms[name] = len(measured)
    if optional_arms:
        fanout['optional_counter_reported_arms'] = optional_arms
    for name in ('work','projection_fallbacks'):
        maps = [item[name] for item in evidence if name in item]
        if maps:
            keys = dict.fromkeys(key for item in maps for key in item)
            fanout[name] = {key:sum(item.get(key,0) for item in maps) for key in keys}
            fanout[name+'_reported_arms'] = {key:sum(key in item for item in maps) for key in keys}
    producer = next((item for item in evidence if item['role']=='produce'),None)
    if producer is not None:
        fanout['producer_evidence'] = dict(producer)
        if 'early_publication' in producer:
            fanout['early_publication'] = dict(producer['early_publication'])
    fanout['fallback_reasons'] = list(dict.fromkeys(reason for item in evidence for reason in item['fallback_reasons']))
    produced = fanout['produced_views']
    consumed = fanout['consumed_views']
    if consumed and not produced:
        raise ValueError('Native products were consumed without a validated producer in this family')
    if not evidence:
        fanout['status'] = 'awaiting_producer'
    elif fanout['scalar_views'] or fanout['fallback_reasons']:
        fanout['status'] = 'partial_fallback' if produced or consumed else 'unsupported'
    elif len(evidence) < len(manifest['cutoffs_hundredths']):
        fanout['status'] = 'native_products_ready'
    else:
        fanout['status'] = 'shared_native_products'
    fanout['validated'] = len(evidence) == len(manifest['cutoffs_hundredths'])


def _finalize_storage_pointer(manifest,root):
    fanout=manifest['upstream_fanout']
    if 'storage_file' in fanout:
        measured=(root/'fanout_storage.json').is_file()
        fanout['storage_measured']=measured
        fanout['storage_file']='fanout_storage.json' if measured else None


def _check_receipt_versions(versions):
    for path, identity in versions.items():
        if _file_identity(path) != identity:
            raise RuntimeError(f'Scalar cutoff receipt changed before sweep completion: {path}')


def _assert_artifact_stats_unchanged(inputs):
    """Cheap pre-review drift check; the final authoritative check hashes once."""
    records = [inputs['source']]
    for model in inputs['models'].values():
        if model is not None:
            records.extend(model['artifacts'])
    for record in records:
        current = _file_identity(record['path'])
        if any(current[field] != record[field] for field in _IDENTITY_FIELDS[1:]):
            raise RuntimeError(f"Input/model artifact changed before confidence review: {record['path']}")


def _remove_owned_work(work, parent, sweep_id):
    if work.is_symlink() or work.resolve().parent != parent.resolve() or not work.name.startswith('xta-confidence-sweep-'):
        raise RuntimeError('Confidence sweep work directory escaped its owner')
    marker = json.loads((work / 'owner.json').read_text(encoding='utf-8'))
    if marker.get('sweep_id') != sweep_id or marker.get('pid') != os.getpid():
        raise RuntimeError('Confidence sweep work ownership changed; refusing cleanup')
    shutil.rmtree(work)


def _release_owned_lock(path, sweep_id):
    owner = json.loads(path.read_text(encoding='utf-8'))
    if owner.get('sweep_id') != sweep_id or owner.get('pid') != os.getpid():
        raise RuntimeError('Confidence sweep root lock ownership changed')
    path.unlink()


def _work_storage(work_parent):
    """Record whether --temp keeps the transient sweep files in memory (manifest ``work_storage``)."""
    from .runtime import path_is_memory_backed
    try:
        memory_backed = bool(path_is_memory_backed(work_parent))
    except Exception:
        memory_backed = None
    return {'root': str(work_parent), 'memory_backed': memory_backed}


def run_confidence_sweep(mode_arguments, *, cutoffs_hundredths, output_root,
                         temp_root=None, min_conf=0.0):
    """Run independent scalar branches and atomically publish one family receipt.

    Existing sweep-owned paths are rejected without replacing their receipts.
    Unrelated files, including live scheduler logs, remain unmanaged.
    Failed outputs are retained for diagnosis; retries require a fresh output root.
    """
    mode_arguments = tuple(str(value) for value in mode_arguments)
    cents = _cutoffs(cutoffs_hundredths)
    view_major = os.environ.get('XTA_YOLO_SWEEP_VIEW_MAJOR','1')
    if view_major not in ('0','1'):
        raise ValueError('XTA_YOLO_SWEEP_VIEW_MAJOR must be 0 or 1')
    fanout_enabled = len(cents)>1 and view_major=='1'
    requested_parallel = _requested_consumer_parallel() if fanout_enabled else None
    reserved_outputs = _RESERVED_OUTPUTS if fanout_enabled else _SCALAR_OUTPUTS
    if isinstance(min_conf, bool) or not math.isfinite(float(min_conf)) or not 0 <= float(min_conf) <= 1:
        raise ValueError('Fixed sweep min_conf must be finite and in [0,1]')
    min_conf = float(min_conf)
    base_arguments, capture_root = _base_arguments(mode_arguments)
    cache_gib = os.environ.get('XTA_YOLO_SWEEP_CACHE_GIB', '64')
    if not math.isfinite(float(cache_gib)) or float(cache_gib) < 0:
        raise ValueError('XTA_YOLO_SWEEP_CACHE_GIB must be finite and nonnegative')
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sweep_id = uuid4().hex
    lock_path = root / '.confidence_sweep.lock'
    with lock_path.open('x', encoding='utf-8') as lock:
        json.dump({'sweep_id': sweep_id, 'pid': os.getpid(), 'started': _now()}, lock)
        lock.flush()
        os.fsync(lock.fileno())
    manifest_path = root / 'manifest.json'
    work = None
    manifest = None
    lock_held = True
    committed = False
    try:
        occupied = [root/name for name in reserved_outputs if os.path.lexists(root/name)]
        if occupied:
            raise FileExistsError(f'Confidence sweep output is already reserved; use a fresh output directory: {occupied[0]}')
        work_parent = Path(temp_root).expanduser().resolve() if temp_root is not None else Path(tempfile.gettempdir()).resolve()
        _check_work_destination(work_parent,root,'--temp',reserved_outputs)
        work_storage = _work_storage(work_parent)
        print(f"Confidence sweep workspace: {work_parent} "
              f"({'memory-backed' if work_storage['memory_backed'] else 'disk-backed'}).", flush=True)
        if capture_root is not None:
            _check_work_destination(capture_root,root,'--capture_component_replay',reserved_outputs)
        work_parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='xta-confidence-sweep-', dir=work_parent))
        write_json_atomic(work / 'owner.json', {'sweep_id': sweep_id, 'pid': os.getpid()})
        cache = work / 'forward_cache'
        manifest = {
            'schema': SCHEMA, 'status': 'in_progress', 'sweep_id': sweep_id,
            'launcher_version': __version__, 'parent_pid': os.getpid(), 'started': _now(),
            'cutoffs_hundredths': list(cents), 'fixed_min_conf': min_conf,
            'branch_policy': ('independent_scalar_suffix_with_native_fanout' if fanout_enabled
                              else 'independent_scalar_full_pipeline'),
            'mode_arguments': [str(value) for value in mode_arguments],
            'work_directory': str(work), 'work_storage': work_storage,
            'forward_cache': {'requested_budget_gib': float(cache_gib),
                'budget_gib': float(cache_gib) if len(cents)>1 else 0.,
                'lifetime': 'family_owned_transient'},
            'branches': [], 'review_manifest': None,
            'upstream_fanout': ({'status':'awaiting_producer','validated':False,
                'producer_cutoff_hundredths':cents[0], 'target_cutoffs_hundredths':sorted(cents),
                'progress_file':'fanout_progress.json','storage_file':'fanout_storage.json',
                'storage_measured':False,'validated_arms':0,
                'produced_views':0,'consumed_views':0,'scalar_views':0,'fallback_reasons':[]}
                if fanout_enabled else {'status':'disabled_singleton' if len(cents)==1 else 'disabled'}),
        }
        write_json_atomic(manifest_path, manifest)
        env = dict(os.environ)
        env['XTA_YOLO_SWEEP_CACHE_DIR'] = str(cache)
        env['XTA_YOLO_SWEEP_CACHE_GIB'] = cache_gib if len(cents)>1 else '0'
        env['XTA_YOLO_SWEEP_ID'] = sweep_id
        if len(cents)>1:
            env['XTA_YOLO_SWEEP_WORK_DIR'] = str(work)
        else:
            env.pop('XTA_YOLO_SWEEP_WORK_DIR',None)
        if fanout_enabled:
            env['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'] = ','.join(str(cent) for cent in sorted(cents))
            env['XTA_YOLO_SWEEP_OUTPUT_ROOT'] = str(root)
        else:
            env.pop('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS',None)
            env.pop('XTA_YOLO_SWEEP_OUTPUT_ROOT',None)
        env.pop('XTA_YOLO_SWEEP_NATIVE_ROLE',None)
        env['XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS'] = str(min(cents))
        env.pop('XTA_YOLO_SWEEP_FIXED_MIN_CONF', None)
        package_parent = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (package_parent, env.get('PYTHONPATH', ''))))
        if sys.version_info >= (3,11):
            env['PYTHONSAFEPATH'] = '1'
        # The coordinator owns this marker: only concurrently running consumers defer.
        env.pop(_DEFER_NATIVE_RETIREMENT_ENV, None)
        reference = None
        rows = []
        receipt_versions = {}
        # Only the coordinator thread changes the family receipt; output readers never do.
        manifest_lock = threading.Lock()

        def open_branch(index, *, concurrent=False):
            cent = cents[index]
            branch_out = root / 'confidence_cutoffs' / f'conf_{cent:03d}'
            branch_temp = work / 'branches' / f'conf_{cent:03d}'
            arguments = [*base_arguments, '--conf', f'{cent / 100:.2f}',
                         '--output', str(branch_out), '--temp', str(branch_temp),
                         '--min_conf', repr(min_conf)]
            if capture_root is not None:
                arguments.extend(('--capture_component_replay', str(capture_root / f'sweep_{sweep_id}' / f'conf_{cent:03d}')))
            command = _child_command(arguments,package_parent)
            child_env = {**env, 'XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS': str(cent)}
            native_role = 'produce' if index == 0 else 'consume'
            if fanout_enabled:
                child_env['XTA_YOLO_SWEEP_NATIVE_ROLE'] = native_role
            if concurrent:
                # A lower-cutoff consumer may still read shared native journal payloads.
                child_env[_DEFER_NATIVE_RETIREMENT_ENV] = '1'
            row = {'cutoff_hundredths': cent, 'confidence': f'{cent / 100:.2f}',
                   'status': 'in_progress', 'output': branch_out.relative_to(root).as_posix(),
                   'manifest': (branch_out / 'manifest.json').relative_to(root).as_posix(),
                   'command': command}
            if fanout_enabled:
                row['upstream_role'] = native_role
            with manifest_lock:
                manifest['branches'].append(row)
                write_json_atomic(manifest_path, manifest)
            _say(f'\n=== Confidence sweep {cent / 100:.2f} ({index + 1}/{len(cents)}) ===')
            return {'cent': cent, 'command': command, 'env': child_env, 'row': row,
                    'output': branch_out, 'role': native_role, 'prefix': f'[conf {cent / 100:.2f}] '}

        def record_started(row):
            def started(pid):
                with manifest_lock:
                    row['pid'] = pid
                    write_json_atomic(manifest_path,manifest)
            return started

        def accept_branch(branch, returncode):
            nonlocal reference
            cent, row, command = branch['cent'], branch['row'], branch['command']
            row['returncode'] = returncode
            if returncode != 0:
                row['status'] = 'failed'
                raise subprocess.CalledProcessError(returncode, command)
            receipt_path = branch['output'] / 'manifest.json'
            validation_started = time.monotonic()
            receipt_versions[receipt_path] = _file_identity(receipt_path)
            receipt, identity, digest = _receipt(receipt_path,
                sweep_id=sweep_id, cent=cent, min_conf=min_conf)
            print(f'Validated cutoff {cent / 100:.2f} receipt in {time.monotonic() - validation_started:.1f}s.',
                  flush=True)
            if reference is not None and identity != reference:
                raise ValueError('Input/model/configuration identity changed between confidence cutoffs')
            reference = identity
            with manifest_lock:
                manifest['inputs'] = receipt['inputs']
                if fanout_enabled:
                    row['upstream_fanout'] = _fanout_evidence(receipt,role=branch['role'],target_count=len(cents))
                    _update_fanout_summary(manifest)
            if fanout_enabled:
                evidence = row['upstream_fanout']
                _say(f"Validated upstream {cent / 100:.2f}: produced native views={evidence['produced_views']}, "
                     f"consumed native views={evidence['consumed_views']}, scalar fallback views={evidence['scalar_views']}.")
                if evidence['fallback_reasons']:
                    _say('Upstream scalar fallback: '+'; '.join(evidence['fallback_reasons']))
            with manifest_lock:
                row.update(status=receipt['status'], manifest_sha256=digest)
                if receipt['sam_planning_omissions']:
                    row['sam_planning_omissions'] = receipt['sam_planning_omissions']
                rows.append((cent, branch['output']))
                write_json_atomic(manifest_path, manifest)

        def run_sequential(index):
            branch = open_branch(index)
            returncode = _run_child(branch['command'], branch['env'], started=record_started(branch['row']))
            accept_branch(branch, returncode)

        def abandon_concurrent(running, current, error):
            """Record why unfinished consumers ended, then stop every owned tree together."""
            cancelled = isinstance(error, KeyboardInterrupt)
            failed = None if cancelled or current is None else current.branch['cent']
            unsettled = None
            with _settling_signals():
                with manifest_lock:
                    if failed is not None and current.branch['row']['status'] == 'in_progress':
                        current.branch['row']['status'] = 'failed'
                    for child in running:
                        row = child.branch['row']
                        if not cancelled and child is not current and row['status'] == 'in_progress':
                            row['status'] = 'stopped'
                            if failed is not None:
                                row['stopped_after_failed_cutoff_hundredths'] = failed
                try:
                    _stop_trees(running)
                except _ChildTreeUnsettled as stop_error:
                    unsettled = stop_error
                for child in running:
                    try:
                        child.close()
                    except _ChildTreeUnsettled as close_error:
                        unsettled = unsettled or close_error
                    if child.process is not None and child.process.returncode is not None:
                        child.branch['row'].setdefault('returncode', child.process.returncode)
            if unsettled is not None:
                raise unsettled

        def run_concurrent(indices, limit):
            """Launch consumers in cutoff order and validate each receipt as it exits."""
            pending = deque(indices)
            running = []
            current = None
            try:
                while pending or running:
                    while pending and len(running) < limit:
                        current = _ConcurrentChild(open_branch(pending.popleft(), concurrent=True))
                        running.append(current)
                        current.start(started=record_started(current.branch['row']))
                        current = None
                    finished = next((child for child in running if child.process.poll() is not None), None)
                    if finished is None:
                        time.sleep(_CONSUMER_POLL_SECONDS)
                        continue
                    current = finished
                    returncode = finished.settle()
                    running.remove(finished)
                    finished.close()
                    accept_branch(finished.branch, returncode)
                    current = None
            except BaseException as error:
                abandon_concurrent(running, current, error)
                raise

        with _cancel_signals():
            run_sequential(0)
            consumers = range(1, len(cents))
            limit = 1
            if fanout_enabled:
                execution = _consumer_execution(requested_parallel, len(consumers),
                    manifest['branches'][0]['upstream_fanout'], manifest['inputs']['source_shape_t_y_x'],
                    shared_source=(work / 'source.gray8.json').is_file(),
                    shared_cube=not _shared_cube_missing(work, manifest['inputs']))
                limit = execution['limit']
                with manifest_lock:
                    manifest['consumer_execution'] = execution
                    write_json_atomic(manifest_path, manifest)
                if limit > 1:
                    _say(f"Confidence sweep consumers: up to {limit} of {len(consumers)} run concurrently "
                         f"({execution['reason']}); their output lines are prefixed with [conf X.XX].")
                else:
                    _say(f"Confidence sweep consumers run sequentially ({execution['reason']}).")
            if limit > 1:
                run_concurrent(consumers, limit)
            else:
                for index in consumers:
                    run_sequential(index)
            rows.sort(key=lambda item: cents.index(item[0]))
            _check_receipt_versions(receipt_versions)
            _assert_artifact_stats_unchanged(manifest['inputs'])
            from .forward_replay import replay_cache_summary
            manifest['forward_cache']['summary'] = replay_cache_summary(cache)
            _remove_owned_work(work, work_parent, sweep_id)
            work = None
            incomplete = [row['cutoff_hundredths'] for row in manifest['branches'] if row['status'] == 'incomplete']
            if incomplete:
                manifest['incomplete_cutoffs_hundredths'] = incomplete
                manifest['review_omitted_reason'] = 'incomplete_requested_sam_work'
            else:
                from .sweep_publication import publish_sweep_review
                review = publish_sweep_review(root, rows, sweep_id=sweep_id, temp_dir=work_parent)
                if review is not None:
                    review = Path(review).resolve()
                    if not review.is_file():
                        raise RuntimeError('Confidence review publisher did not leave its manifest')
                    manifest['review_manifest'] = review.relative_to(root).as_posix()
            _check_receipt_versions(receipt_versions)
            # The coordinator can inherit an unrelated scalar run's environment.
            # At this point its own child workspace has already been removed.
            assert_tta_artifacts_unchanged(manifest['inputs'], use_sweep_cache=False)
            _finalize_storage_pointer(manifest,root)
            manifest.update(status='incomplete' if incomplete else 'complete',
                            finished=_now(), work_cleanup_complete=True)
            write_json_atomic(manifest_path, manifest)
            committed = True
            if incomplete:
                affected = ', '.join(f'{cent / 100:.2f}' for cent in incomplete)
                print(f'WARNING: SAM work was incomplete at {affected}. All scalar outputs were retained. '
                      f'Aggregate review was skipped. See {manifest_path}.', file=sys.stderr, flush=True)
            try:
                _release_owned_lock(lock_path, sweep_id)
            except Exception as lock_error:
                # The validated family commit is authoritative. A stale owner
                # file cannot invalidate settled products after that commit.
                print(f'Warning: completed confidence sweep root lock could not be retired: {lock_path}: {lock_error}',
                      file=sys.stderr,flush=True)
            lock_held = False
        return manifest_path
    except BaseException as error:
        if committed:
            raise
        if work is not None and not isinstance(error,_ChildTreeUnsettled):
            if manifest is not None:
                try:
                    from .forward_replay import replay_cache_summary
                    manifest['forward_cache']['summary'] = replay_cache_summary(cache)
                except Exception as summary_error:
                    manifest['forward_cache']['summary_error'] = str(summary_error)
            try:
                _remove_owned_work(work, work_parent, sweep_id)
                work = None
            except Exception as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Confidence sweep owned work cleanup failed: {cleanup_error}')
        if manifest is not None:
            _finalize_storage_pointer(manifest,root)
            manifest.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                            finished=_now(), error={'type': type(error).__name__, 'message': str(error)},
                            work_cleanup_complete=work is None)
            for row in manifest['branches']:
                if row['status'] == 'in_progress':
                    row['status'] = manifest['status']
            write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if lock_held:
            _release_owned_lock(lock_path, sweep_id)


__all__ = ['SCHEMA', 'run_confidence_sweep']
