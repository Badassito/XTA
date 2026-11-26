"""Prepare and install detector mapping descriptors on a private native actor channel."""
from __future__ import annotations

import array
import contextlib
import os
import socket
import stat
import struct


_SOURCE, _INPUT, _OUTPUT = range(3)
_FIELDS = {'source_volume_path': _SOURCE, 'cpu_input_path': _INPUT,
           'result_mask_path': _OUTPUT, 'result_conf_path': _OUTPUT,
           'canvas_path': _OUTPUT, 'd1_bitset_path': _OUTPUT}
_SLOTS = '_detector_handle_slots'
_TABLE = '_detector_handle_table'
_CHUNK_SIZE = 253


def _identity(fd):
    status = os.fstat(fd)
    if not stat.S_ISREG(status.st_mode):
        raise ValueError('Detector mapping requires a regular backing file')
    return ':'.join(map(str, (status.st_dev, status.st_ino, status.st_size, status.st_mode,
                              status.st_uid, status.st_gid)))


def _holders(task, *, copy):
    yield task, _FIELDS
    native = task.get('native_resize')
    if isinstance(native, dict):
        if copy:
            task['native_resize'] = native = dict(native)
        yield native, {'path': _SOURCE}
    children = task.get('augmentation_pass_tasks')
    if children:
        if copy:
            task['augmentation_pass_tasks'] = children = [dict(child) for child in children]
        for child in children:
            yield from _holders(child, copy=copy)


@contextlib.contextmanager
def detector_handle_submission(task):
    """Borrow stable registered FDs through native submit, which acquires its own pins."""
    from . import runtime

    packet = dict(task)
    slots, handles = {}, []
    with runtime._MEMFD_OWNER_LOCK:
        for holder, fields in _holders(packet, copy=True):
            if _SLOTS in holder or _TABLE in holder:
                raise ValueError('Detector task already carries a handle manifest')
            for field, role in fields.items():
                path = holder.get(field)
                if path is None:
                    continue
                fd = runtime._memfd_owner_fd_for_path(path)
                if fd is None:
                    if str(path).startswith('/proc/') and '/fd/' in str(path):
                        raise RuntimeError(f'Detector {field} refers to an unregistered proc descriptor')
                    continue
                key = _identity(fd), role
                if key not in slots:
                    slots[key] = len(handles)
                    handles.append((fd, *key))
                holder.setdefault(_SLOTS, {})[field] = slots[key]
        if handles:
            packet[_TABLE] = tuple(slots)
        yield packet, tuple(handles)


def close_received_handles(received):
    while received:
        _, fd = received.popitem()
        os.close(fd)


@contextlib.contextmanager
def planned_detector_source_pins():
    """Hold one read-only pin per ready backing through whole-inventory validation and submit."""
    pins = {}

    def capture(task, handles):
        if not handles:
            return ()
        if any(role != _SOURCE for _, _, role in handles):
            return None
        # Ordinary source files have passed the pipeline's synchronous decode gate.
        # Native-resize descriptors may also name a cube whose producer is still writing.
        for holder, _ in _holders(task, copy=False):
            resize = holder.get('native_resize')
            if isinstance(resize, dict) and resize.get('sentinel'):
                if (not os.path.isfile(resize['sentinel'])
                        or (resize.get('failed') and os.path.exists(resize['failed']))):
                    return None
        import fcntl
        prepared = []
        for borrowed, identity, role in handles:
            mode = fcntl.fcntl(borrowed, fcntl.F_GETFL)
            if (mode & os.O_ACCMODE) == os.O_WRONLY or mode & getattr(os, 'O_PATH', 0):
                raise ValueError('Detector source owner lacks read access')
            if identity not in pins:
                owned = os.open(f'/proc/self/fd/{borrowed}', os.O_RDONLY | os.O_CLOEXEC)
                try:
                    _validate_file(owned, identity, _SOURCE)
                    pins[identity] = owned
                except BaseException:
                    os.close(owned)
                    raise
            prepared.append((pins[identity], identity, role))
        return tuple(prepared)

    try:
        yield capture
    finally:
        close_received_handles(pins)


def receive_detector_files(channel, count, read_exact, received):
    """Handle the Linux ancillary-message limit and short stream reads independently."""
    if not hasattr(socket, 'MSG_CMSG_CLOEXEC'):
        raise RuntimeError('Native detector file transfer requires Linux')
    item_size = array.array('i').itemsize
    while count:
        chunk = min(count, _CHUNK_SIZE)
        data, ancillary, flags, _ = channel.recvmsg(
            4 * chunk, socket.CMSG_SPACE(chunk * item_size), socket.MSG_CMSG_CLOEXEC)
        fds = []
        try:
            for level, kind, raw in ancillary:
                if (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                    entries = array.array('i')
                    entries.frombytes(raw[:len(raw) // item_size * item_size])
                    fds.extend(entries)
            if not data or flags & socket.MSG_CTRUNC or len(fds) != chunk:
                raise ValueError('Truncated detector descriptor table')
            slots = struct.unpack(f'<{chunk}I', data + read_exact(channel, 4 * chunk - len(data)))
            if len(set(slots)) != chunk or any(slot in received for slot in slots):
                raise ValueError('Invalid detector descriptor slot indices')
            received.update(zip(slots, fds))
            fds.clear()
        finally:
            for fd in fds:
                os.close(fd)
        count -= chunk


def _validate_file(fd, identity, role):
    import fcntl
    if _identity(fd) != identity:
        raise RuntimeError('Detector mapping identity changed before installation')
    access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
    if access != (os.O_RDWR if role == _OUTPUT else os.O_RDONLY) or os.get_inheritable(fd):
        raise RuntimeError('Detector mapping has incorrect access mode or inheritance')


def install_detector_handles(task, persistent_sources, received):
    """Commit new source mappings together after validating the complete task graph."""
    new_sources, transient, resolved = {}, [], []
    try:
        manifest = task.pop(_TABLE, ())
        if not isinstance(manifest, tuple) or len(set(manifest)) != len(manifest):
            raise ValueError('Invalid detector mapping manifest')
        for slot, (identity, role) in enumerate(manifest):
            if not isinstance(identity, str) or type(role) is not int or role not in (_SOURCE, _INPUT, _OUTPUT):
                raise ValueError('Invalid detector descriptor identity or role')
            cached = persistent_sources.get(identity) if role == _SOURCE else None
            if cached is None:
                if slot not in received:
                    raise RuntimeError('Detector mapping lacks an installed file descriptor')
                fd = received[slot]
            else:
                if slot in received:
                    raise RuntimeError('Duplicate installation of a persistent detector source')
                fd = cached
            _validate_file(fd, identity, role)
            resolved.append((f'/proc/{os.getpid()}/fd/{fd}', role))
        referenced = set()
        for holder, fields in _holders(task, copy=False):
            for field, slot in holder.pop(_SLOTS, {}).items():
                if (field not in fields or type(slot) is not int or not 0 <= slot < len(resolved)
                        or resolved[slot][1] != fields[field]):
                    raise ValueError('Invalid detector mapping field or access role')
                holder[field] = resolved[slot][0]
                referenced.add(slot)
        if len(referenced) != len(manifest) or any(slot not in referenced for slot in received):
            raise ValueError('Unreferenced detector file descriptor')
        for slot, fd in received.items():
            identity, role = manifest[slot]
            if role == _SOURCE:
                new_sources[identity] = fd
            else:
                transient.append(fd)
        persistent_sources.update(new_sources)
        received.clear()
        return transient
    finally:
        close_received_handles(received)
