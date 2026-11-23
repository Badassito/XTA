"""Sweep-scoped replay of pre-threshold YOLO backend outputs.

The orchestrator owns a fresh private directory for one source and configuration.
Only backend outputs are cached. Every cutoff still runs its own decoder. Packet
files use plain numeric arrays and are published atomically after a disk lease.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import struct
import tempfile
import threading
import time
import weakref
import zlib

import numpy as np

_GIB = 1024 ** 3
_MAX_PACKET = 512 * 1024 ** 2
_HEADER_LIMIT = 64 * 1024
_INLINE_LIMIT = 4096
_DISK_FLOOR = 256 * 1024 ** 2
_MAGIC = b"XTARAW1\0"
_CACHE_ERRORS = (OSError, ValueError, OverflowError, TypeError, MemoryError, sqlite3.Error)
_LOCAL_CACHES = weakref.WeakSet()


def replay_enabled():
    if not os.environ.get("XTA_YOLO_SWEEP_CACHE_DIR"):
        return False
    try:
        gib = float(os.environ.get("XTA_YOLO_SWEEP_CACHE_GIB", "64"))
        return math.isfinite(gib) and gib > 0
    except ValueError:
        return False


def _json(value):
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported replay identity field {type(item).__name__}")

    return json.dumps(value, default=convert, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _file_stamp(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sweep_source_stamp(path):
    """Identify the original acquisition once in a scalar child, independently of temp paths."""
    if not replay_enabled():
        return None
    try:
        return _file_stamp(path)
    except OSError:
        return None


@lru_cache(maxsize=128)
def _known_model_digest(stamp, startup_directory):
    from .unification.tta_manifest import _file_identity
    return _file_identity(stamp[0], content_digest=True)["sha256"]


def bind_worker_replay_source(source, task, render_path):
    """Only unmodified built-in render jobs receive transfer-free logical input identities."""
    stamp = task.get("sweep_source_stamp")
    if not stamp or task.get("augmentation_pass_tasks") or not replay_enabled():
        return
    try:
        rendering = {key: task.get(key) for key in (
            "kind", "view", "job_id", "out_size", "channel_format", "slice_start",
            "slice_count", "source_shape", "source_dtype", "M_out_to_src", "M_out_to_processing",
            "processing_shape", "parent_crop", "azimuthal_texture_required",
        )}
        engine = getattr(source, "engine", None)
        job = task.get("job")
        if not is_dataclass(job) or not is_dataclass(task.get("view")):
            return
        rendering["job"] = asdict(job)
        rendering["backend"] = (type(source).__module__, type(source).__name__, str(render_path),
                                 str(getattr(engine, "_mode", "")))
        rendering["source"] = stamp
        resize = task.get("native_resize")
        rendering["native_resize"] = ({k: resize.get(k) for k in ("shape", "dtype")}
                                      if resize is not None else None)
        source._xta_forward_identity = _digest(rendering)
    except (TypeError, ValueError, AttributeError):
        return


def _db(root):
    conn = sqlite3.connect(root / "packets.sqlite", timeout=5, isolation_level=None)
    try:
        # Replay may live on shared scratch, where SQLite WAL shared memory is unsupported.
        if conn.execute("PRAGMA journal_mode=DELETE").fetchone() != ("delete",):
            raise sqlite3.OperationalError("Forward replay cache requires rollback journaling")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    except BaseException:
        conn.close()
        raise


@lru_cache(maxsize=16)
def _initialize(root, pid):
    root.mkdir(parents=True, exist_ok=True)
    conn = _db(root)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS packets (
                key TEXT PRIMARY KEY, backend TEXT NOT NULL, bytes INTEGER NOT NULL, ready INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inline_packets (key TEXT PRIMARY KEY, packet BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS capacity (
                id INTEGER PRIMARY KEY, bytes INTEGER NOT NULL, pending INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS counters (
                backend TEXT NOT NULL, event TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY(backend,event)
            );
            CREATE TABLE IF NOT EXISTS device_counters (
                backend TEXT NOT NULL, device TEXT NOT NULL, event TEXT NOT NULL, n INTEGER NOT NULL,
                PRIMARY KEY(backend,device,event)
            );
        """)
        with _transaction(conn):
            columns = {row[1] for row in conn.execute("PRAGMA table_info(capacity)")}
            if "pending" not in columns:
                conn.execute("ALTER TABLE capacity ADD COLUMN pending INTEGER NOT NULL DEFAULT 0")
            conn.execute("INSERT OR IGNORE INTO capacity VALUES (1, 0, 0)")
    finally:
        conn.close()


def replay_for_model(model_files, *, backend, settings):
    """Return an optional cache for this backend configuration and immutable model content."""
    directory = os.environ.get("XTA_YOLO_SWEEP_CACHE_DIR")
    if not directory:
        return None
    try:
        gib = float(os.environ.get("XTA_YOLO_SWEEP_CACHE_GIB", "64"))
        if not math.isfinite(gib) or gib <= 0:
            return None
        root = Path(directory).resolve()
        pid = os.getpid()
        _initialize(root, pid)
        startup = os.environ.get("XTA_YOLO_SWEEP_WORK_DIR")
        identities = [_known_model_digest(_file_stamp(path), startup) for path in model_files]
        if not identities:
            return None
        floor = os.environ.get("XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS")
        namespace = _digest(["xta.forward_replay/1", backend, settings, identities, floor])
        device = settings.get("runtime", {}).get("device_identity", settings.get("device", "cpu"))
        return _get_cache(root, pid, backend, str(device), namespace, min(int(gib * _GIB), 2**62))
    except _CACHE_ERRORS:
        return None


@lru_cache(maxsize=128)
def _get_cache(root, pid, backend, device, namespace, limit):
    return ForwardReplay(root, backend, namespace, limit, device=device)


def torch_replay_for_predictor(predictor, cfg, route):
    if not replay_enabled():
        return None
    backend = getattr(predictor, "model", None)
    path = getattr(predictor, "_xta_forward_model_path", None)
    if path is None:
        path = getattr(getattr(predictor, "args", None), "model", None)
    if not isinstance(path, (str, Path)) or Path(path).suffix.lower() not in (".pt", ".engine"):
        return None
    import torch
    if route in ("gpu-segmentation-direct", "gpu-segmentation-trt-ring"):
        floor = os.environ.get("XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS")
        if floor is not None:
            try:
                cents = int(floor)
                if not 0 <= cents <= 100 or float(cfg.conf) < cents / 100:
                    return None
            except (TypeError, ValueError):
                return None
    device = getattr(backend, "device", cfg.device)
    device_name = str(device)
    if device_name.isdigit():
        device_name = "cuda:" + device_name
    if device_name == "cuda" and torch.cuda.is_available():
        device_name = f"cuda:{torch.cuda.current_device()}"
    settings = dict(task=cfg.task, channels=cfg.input_channels, quantize=cfg.quantize,
                    device=device_name, runtime=_torch_runtime(device_name,os.getpid()),
                    fp16=bool(getattr(backend, "fp16", False)),
                    backend_class=type(backend).__qualname__, torch=str(torch.__version__),
                    cuda=torch.version.cuda)
    return replay_for_model([path], backend=route, settings=settings)


@lru_cache(maxsize=32)
def _torch_runtime(device_name, pid):
    import torch
    hardware = None
    if device_name.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device_name)
        uuid = getattr(p, "uuid", None)
        hardware = (p.name, p.major, p.minor, str(uuid) if uuid else "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = hardware[3] if hardware and hardware[3] else f"{device_name} visible={visible}"
    identity = f"{socket.gethostname()}/{physical}" if device_name.startswith("cuda") else device_name
    return {"device_identity":identity, "visible_devices":visible, "hardware":hardware,
            "ultralytics":importlib.metadata.version("ultralytics")}


@contextmanager
def _transaction(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


class ForwardReplay:
    def __init__(self, root, backend, namespace, limit, *, device="cpu"):
        self.root, self.backend, self.namespace, self.limit = root, backend, namespace, limit
        self.device = device
        self._local = threading.local()
        self._pending = Counter()
        self._stats_lock = threading.Lock()
        self._events = 0
        self._flushed = time.monotonic()
        self.verify = os.environ.get("XTA_YOLO_SWEEP_VERIFY", "0") == "1"
        self._pid = os.getpid()
        _LOCAL_CACHES.add(self)
        from multiprocessing.util import Finalize
        self._finalizer = Finalize(None, self.flush, exitpriority=5)

    def _connection(self):
        if getattr(self._local, "conn", None) is None:
            self._local.conn = _db(self.root)
        return self._local.conn

    def note(self, event, amount=1):
        # Buffered counters avoid a SQLite transaction on every cache hit.
        with self._stats_lock:
            self._pending[event] += int(amount)
            self._events += 1
            due = self._events >= 256 or time.monotonic()-self._flushed >= 10.0
        if due:
            self.flush()

    def flush(self):
        if self._pid != os.getpid():
            return
        with self._stats_lock:
            if not self._pending:
                return
            pending, self._pending = self._pending, Counter()
            self._events = 0
            self._flushed = time.monotonic()
        try:
            conn = self._connection()
            with _transaction(conn):
                conn.executemany(
                    "INSERT INTO counters VALUES (?,?,?) ON CONFLICT(backend,event) DO UPDATE SET n=n+excluded.n",
                    [(self.backend, event, n) for event,n in pending.items()])
                conn.executemany(
                    "INSERT INTO device_counters VALUES (?,?,?,?) ON CONFLICT(backend,device,event) "
                    "DO UPDATE SET n=n+excluded.n",
                    [(self.backend, self.device, event, n) for event,n in pending.items()])
        except _CACHE_ERRORS:
            with self._stats_lock:
                self._pending.update(pending)

    def input_key(self, batch, *, source=None, batch_index=None):
        """Use a qualified logical key, or an existing CPU buffer for an unknown renderer."""
        try:
            identity = getattr(source, "_xta_forward_identity", None)
            if identity is None:
                identity = getattr(getattr(source, "base_source", None), "_xta_forward_identity", None)
            shape = tuple(int(x) for x in batch.shape)
            dtype = str(batch.dtype)
            if identity is not None and batch_index is not None:
                self.note("logical_input_keys")
                source_type = (type(source).__module__, type(source).__qualname__)
                wrapped = getattr(source, "base_source", None)
                wrapped_type = (type(wrapped).__module__, type(wrapped).__qualname__) if wrapped is not None else None
                return _digest([self.namespace, shape, dtype, identity, source_type, wrapped_type, int(batch_index)])
            if bool(getattr(batch, "is_cuda", False)):
                self.note("unidentified_device_input")
                return None
            local = batch.detach().contiguous().numpy() if hasattr(batch, "detach") else batch
            array = np.ascontiguousarray(local)
            if array.dtype.hasobject or array.nbytes > _MAX_PACKET:
                return None
            digest = hashlib.sha256(_json([self.namespace, shape, dtype, "host_content"]))
            digest.update(memoryview(array).cast("B"))
            self.note("host_input_bytes_hashed", array.nbytes)
            return digest.hexdigest()
        except (AttributeError, RuntimeError, *_CACHE_ERRORS):
            return None

    def _packet_path(self, key):
        return self.root / key[:2] / (key + ".raw")

    def _reserve(self, key, size):
        if size > min(self.limit, _MAX_PACKET):
            return False
        free = shutil.disk_usage(self.root).free
        conn = self._connection()
        with _transaction(conn):
            if conn.execute("SELECT 1 FROM packets WHERE key=?", (key,)).fetchone():
                return False
            used, pending = conn.execute("SELECT bytes,pending FROM capacity WHERE id=1").fetchone()
            if used + size > self.limit or free < pending + size + _DISK_FLOOR:
                return False
            conn.execute("INSERT INTO packets VALUES (?,?,?,0)", (key, self.backend, size))
            conn.execute("UPDATE capacity SET bytes=bytes+?,pending=pending+? WHERE id=1", (size,size))
        return True

    def _settle(self, key, reserved, actual=None, inline=None):
        conn = self._connection()
        with _transaction(conn):
            if actual is None:
                removed = conn.execute("DELETE FROM packets WHERE key=? AND ready=0 AND bytes=?", (key,reserved))
                if not removed.rowcount:
                    return
            else:
                if inline is not None:
                    conn.execute("INSERT INTO inline_packets VALUES (?,?)", (key,inline))
                conn.execute("UPDATE packets SET ready=1,bytes=? WHERE key=?", (actual, key))
            # Fund allocation granularity and per-entry database metadata as well as payload.
            charged = ((actual + 1023)//1024 + 1)*1024 if actual is not None else 0
            conn.execute("UPDATE capacity SET bytes=bytes+?,pending=pending-? WHERE id=1",
                         (charged-reserved, reserved))

    def store(self, key, arrays):
        if key is None:
            return False
        lease = False
        staging = None
        published = None
        started = time.perf_counter_ns()
        try:
            if not arrays or len(arrays) > 32:
                return False
            sizes = [int(a.numel())*int(a.element_size()) if hasattr(a, "numel") else int(a.nbytes) for a in arrays]
            reserved = sum(sizes) + _HEADER_LIMIT
            lease = self._reserve(key, reserved)
            if not lease:
                self.note("not_admitted_or_already_stored")
                return False
            output = self._packet_path(key)
            output.parent.mkdir(exist_ok=True)
            records, buffers = [], []
            for item in arrays:
                strides = None
                if hasattr(item, "detach"):
                    strides = [int(v) for v in item.stride()]
                    torch_dtype = str(item.dtype)
                    host = item.detach().contiguous().cpu()
                    if torch_dtype == "torch.bfloat16":
                        import torch
                        host = host.view(torch.uint16)
                    array = host.numpy()
                else:
                    torch_dtype = None
                    array = np.asarray(item)
                    if not array.flags.c_contiguous:
                        array = np.ascontiguousarray(array)
                if array.dtype.kind not in "buif" or array.ndim > 8:
                    return False
                buf = memoryview(array).cast("B") if array.size else b""
                records.append([array.dtype.str, list(array.shape), strides, torch_dtype,
                                zlib.crc32(buf) if self.verify else None])
                buffers.append(buf)
            header = _json(records)
            prefix = _MAGIC + struct.pack("<I", len(header)) + header
            if len(prefix)+sum(sizes) <= _INLINE_LIMIT:
                packet = b"".join((prefix, *buffers))
                self._settle(key, reserved, len(packet), inline=packet)
                lease = False
                self.note("inline_stored")
                self.note("payload_bytes_written", len(packet))
                self.note("stored")
                return True
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".raw-", delete=False) as f:
                staging = Path(f.name)
                f.write(prefix)
                for buf in buffers:
                    f.write(buf)
                actual = f.tell()
            if actual > reserved:
                return False
            staging.replace(output)
            staging = None
            published = output
            self._settle(key, reserved, actual)
            lease = False
            self.note("payload_bytes_written", actual)
            self.note("stored")
            return True
        except (RuntimeError, AttributeError, *_CACHE_ERRORS):
            self.note("store_error")
            return False
        finally:
            if staging is not None:
                try:
                    staging.unlink(missing_ok=True)
                except OSError:
                    pass
            if lease:
                if published is not None:
                    try:
                        published.unlink(missing_ok=True)
                    except OSError:
                        pass
                try:
                    self._settle(key, reserved)
                except _CACHE_ERRORS:
                    pass
            self.note("store_ns", time.perf_counter_ns()-started)

    def store_instance(self, key, arrays, *, input_shape_hw, proto_support):
        """Persist an exact floor-qualified direct-decoder packet when its encoding is smaller."""
        if key is None:
            return False
        floor = os.environ.get("XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS")
        if floor is not None:
            from .forward_replay_sparse import pack_instance_packet
            started = time.perf_counter_ns()
            packed = pack_instance_packet(*arrays, floor_hundredths=int(floor),
                                          input_shape_hw=input_shape_hw, proto_support=proto_support)
            self.note("sparse_pack_host_ns", time.perf_counter_ns()-started)
            if packed is not None:
                stored = self.store(key, packed)
                if stored:
                    self.note("sparse_instance_stored")
                return stored
        return self.store(key, arrays)

    def _read(self, key):
        if key is None:
            return None
        started = time.perf_counter_ns()
        try:
            try:
                stream = self._packet_path(key).open("rb")
                inline_packet = None
            except FileNotFoundError:
                row = self._connection().execute("SELECT packet FROM inline_packets WHERE key=?", (key,)).fetchone()
                if row is None:
                    self.note("miss")
                    return None
                inline_packet = row[0]
                stream = io.BytesIO(inline_packet)
            with stream as f:
                size = len(inline_packet) if inline_packet is not None else os.fstat(f.fileno()).st_size
                if size > min(self.limit, _MAX_PACKET) or f.read(8) != _MAGIC:
                    raise ValueError("Invalid replay packet")
                head_size, = struct.unpack("<I", f.read(4))
                if head_size > _HEADER_LIMIT or head_size > size-12:
                    raise ValueError("Invalid replay packet header")
                records = json.loads(f.read(head_size))
                if not isinstance(records, list) or not 1 <= len(records) <= 32:
                    raise ValueError("Invalid output record count")
                result = []
                for dtype, shape, strides, torch_dtype, checksum in records:
                    dtype = np.dtype(dtype)
                    if (dtype.kind not in "buif" or (torch_dtype is not None and not dtype.isnative)
                            or not isinstance(shape,list) or len(shape) > 8
                            or any(type(n) is not int or n < 0 for n in shape)):
                        raise ValueError("Invalid replay dtype/shape")
                    n = math.prod(shape)
                    if n*dtype.itemsize > size-f.tell():
                        raise ValueError("Truncated replay array")
                    if inline_packet is not None:
                        offset = f.tell()
                        value = np.frombuffer(inline_packet, dtype=dtype, count=n, offset=offset).copy().reshape(shape)
                        f.seek(offset + n*dtype.itemsize)
                    else:
                        value = np.fromfile(f, dtype=dtype, count=n).reshape(shape)
                    if (self.verify and checksum is not None
                            and zlib.crc32(memoryview(value).cast("B") if value.size else b"") != checksum):
                        raise ValueError("Replay checksum mismatch")
                    result.append((value, strides, torch_dtype))
                if f.tell() != size:
                    raise ValueError("Trailing replay bytes")
            self.note("payload_bytes_read", size)
            return result
        except FileNotFoundError:
            self.note("miss")
        except (struct.error, *_CACHE_ERRORS):
            self.note("invalid_packet")
        finally:
            self.note("packet_lookup_and_read_ns", time.perf_counter_ns()-started)
        return None

    def load(self, key):
        rows = self._read(key)
        if rows is None:
            return None
        values = tuple(row[0] for row in rows)
        if self._is_encoded_instance(rows):
            values = self._unpack_instance(values, device=None)
            if values is None:
                return None
        self.note("hit")
        return values

    def _is_encoded_instance(self, rows):
        return self.backend in ("gpu-segmentation-direct", "gpu-segmentation-trt-ring") and len(rows) != 2

    def _unpack_instance(self, values, device):
        from .forward_replay_sparse import unpack_instance_packet
        try:
            return unpack_instance_packet(values, device=device)
        except (ValueError, IndexError, RuntimeError, TypeError, OverflowError):
            self.note("invalid_packet")
            return None

    def load_tensors(self, key, *, device):
        rows = self._read(key)
        if rows is None:
            return None
        started = time.perf_counter_ns()
        if self._is_encoded_instance(rows):
            # Validate small headers and indices on the host. Transfer only selected
            # anchors and prototype cells, then scatter directly onto the owning device.
            tensors = self._unpack_instance(tuple(row[0] for row in rows), device)
            if tensors is not None:
                self.note("hit")
                self.note("tensor_restore_host_ns", time.perf_counter_ns()-started)
            return tensors
        import torch
        result = []
        for value, strides, torch_dtype in rows:
            tensor = torch.from_numpy(value)
            if torch_dtype == "torch.bfloat16":
                tensor = tensor.view(torch.bfloat16)
            if torch_dtype != str(tensor.dtype):
                self.note("unsupported_tensor_layout")
                return None
            if strides is not None and tuple(strides) != tensor.stride():
                if (len(strides) != value.ndim or any(type(x) is not int or x < 0 for x in strides)
                        or (1+sum((n-1)*x for n,x in zip(value.shape,strides)))*value.itemsize > _MAX_PACKET):
                    self.note("unsupported_tensor_layout")
                    return None
                try:
                    target = torch.empty_strided(tensor.shape, strides, dtype=tensor.dtype, device=device)
                    target.copy_(tensor)
                except RuntimeError:
                    self.note("unsupported_tensor_layout")
                    return None
                result.append(target)
            else:
                try:
                    result.append(tensor.to(device=device))
                except torch.OutOfMemoryError:
                    self.note("device_restore_oom")
                    return None
        self.note("hit")
        self.note("tensor_restore_host_ns", time.perf_counter_ns()-started)
        return tuple(result)


def replay_cache_summary(root):
    """Aggregate workers' cache events even after scalar subprocesses have exited."""
    result = {"entries": 0, "raw_disk_bytes": 0, "by_backend": {}, "by_device": {}}
    if not root or not (Path(root)/"packets.sqlite").is_file():
        return result
    for cache in list(_LOCAL_CACHES):
        if cache.root == Path(root).resolve():
            cache.flush()
    conn = None
    try:
        conn = _db(Path(root))
        for backend, count, size in conn.execute(
                "SELECT backend,count(*),sum(bytes) FROM packets WHERE ready=1 GROUP BY backend"):
            result["entries"] += count
            result["raw_disk_bytes"] += size
            result["by_backend"][backend] = {"entries": count, "raw_disk_bytes": size}
        for backend, event, count in conn.execute("SELECT backend,event,n FROM counters"):
            result["by_backend"].setdefault(backend, {})[event] = count
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='device_counters'").fetchone():
            for backend, device, event, count in conn.execute("SELECT backend,device,event,n FROM device_counters"):
                result["by_device"].setdefault(device, {}).setdefault(backend, {})[event] = count
    except _CACHE_ERRORS:
        result["status"] = "cache metadata unavailable"
    finally:
        if conn is not None:
            conn.close()
    return result
