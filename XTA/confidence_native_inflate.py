"""Batched nogil zlib inflate for confidence score blocks.

The per-block ``zlib.decompressobj`` reader re-acquires the GIL several times
for every 128x128 block. In a busy main process each re-acquisition waits about
one switch interval, so one parent's restore took tens of seconds. This module
calls zlib's ``uncompress2`` from a compiled loop that releases the GIL once
for a complete batch of blocks.

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
import zlib

import numpy as np

from .confidence_storage import BLOCK_DTYPE
from .native_cpu import kernels
from .native_cpu_build import NativeCpuBuildError


_LOCK = threading.Lock()
_CACHE = {}
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
        self._address = ctypes.cast(inflater, ctypes.c_void_p).value
        self._probe()
        self._probe_batches()

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
        if records.dtype != BLOCK_DTYPE:
            raise ValueError('Invalid confidence block record layout')
        decoded = np.empty(int(decoded_bytes)+1, np.uint8)
        count, status = kernels.inflate_confidence_blocks(self._address, payload, records, int(base), decoded, starts)
        return decoded, int(count), int(status)

    def inflate_scatter(self, payload, records, base, output):
        """Return (count, status, decoded bytes); records before count were assigned."""
        if records.dtype != BLOCK_DTYPE:
            raise ValueError('Invalid confidence block record layout')
        count, status, decoded = kernels.inflate_confidence_scatter(self._address, payload, records, int(base), output)
        return int(count), int(status), int(decoded)

    def _probe_batches(self):
        """Check the selected provider through each compiled batch entry point."""
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
            return inflater, dict(backend='native_uncompress2', codec_version=inflater.version,
                python_zlib_version=zlib.ZLIB_RUNTIME_VERSION, library_source=source,
                fallback_reason=None)
        except NativeCpuBuildError:
            raise
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


__all__ = ['NativeConfidenceInflater', 'native_confidence_inflater']
