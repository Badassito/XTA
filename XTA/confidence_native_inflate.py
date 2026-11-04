"""Batched nogil zlib inflate for confidence score blocks.

The per-block ``zlib.decompressobj`` reader re-acquires the GIL several times
for every 128x128 block. In a busy main process each re-acquisition waits about
one switch interval, so one parent's restore took tens of seconds. This module
calls zlib's ``uncompress2`` from one ``@njit(nogil=True)`` kernel per batch of
blocks; the GIL is released once per batch.

Acceptance is the exact ``BlockScoreReader`` contract: a block is accepted
only when its zlib stream (default 15-bit window, no preset dictionary) ends
exactly at the end of its stored bytes and yields exactly ``h*w`` bytes. Any
other native status is never interpreted here: callers re-run the original
Python decoder for that frame, which owns the exact exception and partial-yield
behaviour. Library resolution follows ``confidence_native_codec``: Python's
zlib extension, then the system zlib on POSIX; Windows uses the Python reader
unless an explicit qualification override names a library.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
import zlib

import numpy as np

from ._deps import _numba
from .confidence_native_codec import _void_pointer


_LOCK = threading.Lock()
_CACHE = {}
_STATUS_BOUNDS = -100
_STATUS_LENGTH = -101


def _make_kernels(inflater, ulong_dtype):
    @_numba.njit(nogil=True, cache=False)
    def inflate_blocks(payload, records, base, decoded, starts):
        """Inflate index records into contiguous segments (one guard byte at the end).

        ``records`` is the block index slice; ``payload`` holds its stored
        bytes starting at absolute offset ``base``.
        """
        dest_length = np.empty(1, ulong_dtype)
        source_length = np.empty(1, ulong_dtype)
        for item in range(records.shape[0]):
            record = records[item]
            area = np.int64(record['h'])*np.int64(record['w'])
            offset = np.int64(record['offset'])-base
            length, start = np.int64(record['length']), starts[item]
            if (area <= 0 or length <= 0 or offset < 0 or offset+length > payload.shape[0]
                    or start < 0 or start+area+1 > decoded.shape[0]):
                return item, _STATUS_BOUNDS
            dest_length[0] = area+1
            source_length[0] = length
            status = inflater(_void_pointer(decoded.ctypes.data+np.uintp(start)),
                _void_pointer(dest_length.ctypes.data),
                _void_pointer(payload.ctypes.data+np.uintp(offset)),
                _void_pointer(source_length.ctypes.data))
            if status != 0:
                return item, np.int64(status)
            if np.int64(dest_length[0]) != area or np.int64(source_length[0]) != length:
                return item, _STATUS_LENGTH
        return records.shape[0], 0

    @_numba.njit(nogil=True, cache=False)
    def inflate_scatter(payload, records, base, output):
        """Inflate records and assign each crop into output in record order.

        Returns (completed records, status, decoded crop bytes assigned).
        """
        largest = 1
        for item in range(records.shape[0]):
            largest = max(largest, np.int64(records[item]['h'])*np.int64(records[item]['w'])+1)
        scratch = np.empty(largest, np.uint8)
        dest_length = np.empty(1, ulong_dtype)
        source_length = np.empty(1, ulong_dtype)
        decoded = 0
        for item in range(records.shape[0]):
            record = records[item]
            z, y, x = np.int64(record['z']), np.int64(record['y']), np.int64(record['x'])
            h, w = np.int64(record['h']), np.int64(record['w'])
            area = h*w
            offset = np.int64(record['offset'])-base
            length = np.int64(record['length'])
            if (h <= 0 or w <= 0 or length <= 0 or offset < 0 or offset+length > payload.shape[0]
                    or not 0 <= z < output.shape[0] or y < 0 or y+h > output.shape[1]
                    or x < 0 or x+w > output.shape[2]):
                return item, _STATUS_BOUNDS, decoded
            dest_length[0] = area+1
            source_length[0] = length
            status = inflater(_void_pointer(scratch.ctypes.data),
                _void_pointer(dest_length.ctypes.data),
                _void_pointer(payload.ctypes.data+np.uintp(offset)),
                _void_pointer(source_length.ctypes.data))
            if status != 0:
                return item, np.int64(status), decoded
            if np.int64(dest_length[0]) != area or np.int64(source_length[0]) != length:
                return item, _STATUS_LENGTH, decoded
            at = 0
            for row in range(y, y+h):
                for column in range(x, x+w):
                    output[z, row, column] = scratch[at]
                    at += 1
            decoded += area
        return records.shape[0], 0, decoded

    return inflate_blocks, inflate_scatter


class NativeConfidenceInflater:
    """Retain one probed zlib library and its two nogil batch kernels."""

    def __init__(self, library, *, source):
        self._library = library
        self.source = str(source)
        version = library.zlibVersion
        version.argtypes = []
        version.restype = ctypes.c_char_p
        self.version = version().decode('ascii')
        if not self.version or len(self.version) > 64:
            raise ValueError('Native zlib returned an invalid runtime version')
        inflater = library.uncompress2
        inflater.argtypes = [ctypes.c_void_p]*4
        inflater.restype = ctypes.c_int
        self._inflater = inflater
        self._ulong = np.uint32 if ctypes.sizeof(ctypes.c_ulong) == 4 else np.uint64
        self._inflate_blocks, self._inflate_scatter = _make_kernels(inflater, self._ulong)
        self._probe()

    def _call(self, encoded, capacity):
        source = ctypes.create_string_buffer(encoded, max(1, len(encoded)))
        output = (ctypes.c_ubyte*(capacity+16))(*([167]*(capacity+16)))
        dest_length, source_length = ctypes.c_ulong(capacity), ctypes.c_ulong(len(encoded))
        status = self._inflater(output, ctypes.byref(dest_length), source, ctypes.byref(source_length))
        if bytes(output[capacity:]) != bytes([167])*16:
            raise ValueError('Native zlib wrote outside its declared output capacity')
        return status, bytes(output[:dest_length.value]), dest_length.value, source_length.value

    def _accepts(self, encoded, area):
        status, raw, written, consumed = self._call(encoded, area+1)
        return (status == 0 and written == area and consumed == len(encoded)), raw

    def _probe(self):
        """Bounded ABI probes: every native accept must equal Python's accept."""
        def python_accepts(encoded, area):
            decoder = zlib.decompressobj()
            try:
                raw = decoder.decompress(encoded, area+1)
            except zlib.error:
                return False, b''
            return (len(raw) == area and decoder.eof and not decoder.unused_data
                    and not decoder.unconsumed_tail), raw
        patterned = bytes((index*37+index//11) % 256 for index in range(16384))
        sparse = bytes(value if index % 3 else 0 for index, value in enumerate(patterned))
        valid = []
        for raw in (b'\0', bytes(range(256)), b'\0'*16384, patterned, sparse):
            for level in (0, 1, 3, 6, 9):
                valid.append((zlib.compress(raw, level), raw))
        dictionary = zlib.compressobj(3, zdict=b'abcdefgh'*8)
        with_dictionary = dictionary.compress(patterned[:512])+dictionary.flush()
        cases = []
        for encoded, raw in valid:
            cases.append((encoded, len(raw)))
        sample, raw = valid[-3]
        cases += [(sample[:-1], len(raw)), (sample[:-4], len(raw)), (sample+b'\0', len(raw)),
                  (sample, len(raw)-1), (sample, len(raw)+1), (sample[:2]+bytes([sample[2] ^ 1])+sample[3:], len(raw)),
                  (sample[:-1]+bytes([sample[-1] ^ 1]), len(raw)), (b'\x78\x9c', len(raw)),
                  (bytes([0x79, 0x9c])+sample[2:], len(raw)), (with_dictionary, 512)]
        for encoded, area in cases:
            native, native_raw = self._accepts(encoded, area)
            expected, expected_raw = python_accepts(encoded, area)
            if native != expected or (native and native_raw != expected_raw):
                raise ValueError('Native zlib inflate differs from Python zlib on a bounded probe')
        # A zero-capacity call must not write and must report an error. Use a
        # 256-byte stream: zlib's uncompress2 decodes into its own 1-byte scratch
        # buffer when destLen is 0, so a 1-byte stream legitimately returns Z_OK.
        encoded, raw = valid[5]
        status, _, written, _ = self._call(encoded, 0)
        if (status == 0 or written) and len(raw) > 1:
            raise ValueError('Native zlib accepted a stream into a zero-byte buffer')

    def inflate_blocks(self, payload, records, base, starts, decoded_bytes):
        """Return (decoded, count, status) for index records stored from ``base``.

        Record ``i`` decodes to ``decoded[starts[i]:starts[i]+h*w]``; the caller
        supplies the segment starts and their total so no small numpy
        operation (each a GIL round trip under load) runs here.
        """
        decoded = np.empty(int(decoded_bytes)+1, np.uint8)
        count, status = self._inflate_blocks(payload, records, int(base), decoded, starts)
        return decoded, int(count), int(status)

    def inflate_scatter(self, payload, records, base, output):
        """Return (count, status, decoded bytes); records before count were assigned."""
        count, status, decoded = self._inflate_scatter(payload, records, int(base), output)
        return int(count), int(status), int(decoded)

    def warm(self):
        """Compile both kernels for the only argument types production passes.

        Records are a readonly C-contiguous slice of the BLOCK_DTYPE index
        memmap (a readonly ndarray types identically); payload and decode
        buffers are fresh uint8 arrays; output is a writable C uint8 3-D grid.
        """
        from .confidence_storage import BLOCK_DTYPE
        raw = np.arange(6, dtype=np.uint8)+11
        encoded = zlib.compress(raw.tobytes(), 3)
        payload = np.empty(len(encoded)+5, np.uint8)
        payload[5:] = np.frombuffer(encoded, np.uint8)
        records = np.zeros(1, BLOCK_DTYPE)
        records[0] = (0, 0, 0, 2, 3, 105, len(encoded))
        records.setflags(write=False)
        decoded, count, status = self.inflate_blocks(payload, records, 100, np.zeros(1, np.int64), 6)
        if status or count != 1 or decoded[:6].tobytes() != raw.tobytes():
            raise ValueError('Native nogil inflate kernel failed its qualification probe')
        output = np.zeros((1, 2, 3), np.uint8)
        count, status, assigned = self.inflate_scatter(payload, records, 100, output)
        if status or count != 1 or assigned != 6 or output.tobytes() != raw.tobytes():
            raise ValueError('Native nogil inflate scatter kernel failed its qualification probe')


def _load_backend(key):
    mode, qualification, path = key
    base = dict(backend='python_zlib', python_zlib_version=zlib.ZLIB_RUNTIME_VERSION)
    if mode in ('0', 'false', 'off', 'python'):
        return None, {**base, 'fallback_reason': 'disabled_by_configuration'}
    candidates = []
    if qualification == '1' and path:
        candidates.append((path, 'explicit_qualification'))
    else:
        extension = getattr(zlib, '__file__', None)
        if extension:
            candidates.append((extension, 'python_zlib_extension'))
        if os.name != 'nt':
            shared = ctypes.util.find_library('z')
            if shared and shared != extension:
                candidates.append((shared, 'system_zlib'))
    errors = []
    for filename, source in candidates:
        try:
            inflater = NativeConfidenceInflater(ctypes.CDLL(filename), source=source)
            inflater.warm()
            return inflater, dict(backend='native_uncompress2', codec_version=inflater.version,
                python_zlib_version=zlib.ZLIB_RUNTIME_VERSION, library_source=source,
                fallback_reason=None)
        except Exception as exc:
            errors.append(f'{source}:{type(exc).__name__}:{str(exc)[:160]}')
    reason = ';'.join(errors) if errors else 'native_zlib_library_not_available'
    return None, {**base, 'fallback_reason': reason}


def native_confidence_inflater():
    """Return a probed inflater (or None) and its backend/version/fallback receipt."""
    key = (os.environ.get('YOLO_TTA_CONFIDENCE_NATIVE_INFLATE', 'auto').strip().lower(),
           os.environ.get('XTA_QUALIFY_NATIVE_ZLIB', ''),
           os.environ.get('XTA_TEST_NATIVE_ZLIB_LIBRARY', ''))
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = _load_backend(key)
            _, receipt = _CACHE[key]
            print('Confidence block inflate: '+', '.join(f'{name}={value}' for name, value in receipt.items()),
                  flush=True)
        inflater, receipt = _CACHE[key]
    return inflater, dict(receipt)


def warm_confidence_inflate_kernels():
    """Load, probe and compile the score-block inflate kernels before the process is busy.

    Production passes a readonly BLOCK_DTYPE index-memmap slice, fresh uint8
    payload/decode buffers, int64 starts, and a writable C uint8 3-D restore
    destination (np.memmap and memfd-backed ndarray type identically).
    """
    started = time.perf_counter()
    inflater, receipt = native_confidence_inflater()
    if inflater is not None:
        inflater.warm()
        receipt.update(blocks_signatures=len(inflater._inflate_blocks.nopython_signatures),
                       scatter_signatures=len(inflater._inflate_scatter.nopython_signatures))
    receipt['seconds'] = time.perf_counter()-started
    return receipt


__all__ = ['NativeConfidenceInflater', 'native_confidence_inflater', 'warm_confidence_inflate_kernels']
