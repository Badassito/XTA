"""GIL-free decode of acknowledged native sweep journal rows.

A journal row is the byte range ``[offset, offset+length)`` of one range's
``payload.bin``: ``chunks`` consecutive strips, each a little-endian ``<10I``
header (y0, y1, x0, x1, mask bytes, score bytes, mask crc, score crc, packed
mask bytes, raw score bytes), the zlib stream of the strip's little-endian
``packbits`` mask and, with a score lane, the zlib stream of its masked scores.
The Python reader in ``sweep_views.NativeViewReader`` is the specification.

Each row is decoded by one ``@njit(nogil=True)`` call that validates every
header and length exactly as that reader does, inflates through the native
``uncompress2`` qualified by ``confidence_native_inflate`` and unpacks the
bits. The projection rank input instead merges all cutoff levels of a frame
straight from the packed bits into its rank plane in one call, then moves the
frame's rank region out in a second. A row is accepted only when the Python
reader accepts it with identical output. Any other outcome returns a nonzero
status before a single caller-visible byte is written, and the caller re-runs
the Python reader, which owns the exact exception and partial-yield behaviour.

The C entry point is passed as an address argument, so the kernels are
ordinary cached numba functions. The route is used only when the native
library reports Python's own zlib runtime version (on Linux, Python's zlib
extension or the system ``libz``); otherwise, and on Windows without an
explicit qualification override, readers keep the Python route.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
import zlib

import numpy as np
from llvmlite import ir
from numba import types
from numba.extending import intrinsic

from ._deps import _numba
from .confidence_native_codec import _void_pointer


HEADER_BYTES = 40
# Row parameter vector (int64) handed to the kernels.
ROW_PARAMETERS = ('offset', 'length', 'chunks', 'y0', 'y1', 'x0', 'x1',
                  'height', 'width', 'score', 'chunk_bytes')
STATUS_PAYLOAD = -1    # row range outside the mapped payload, or impossible chunk count
STATUS_HEADER = -2     # a header does not fit in the row
STATUS_GEOMETRY = -3   # strip bounds/area/raw sizes differ from the row
STATUS_LENGTHS = -4    # encoded lengths differ from the row
STATUS_UNINDEXED = -5  # strips do not end exactly at the row end
STATUS_INFLATE = -6    # a zlib stream is not exactly one stream of the raw size
STATUS_PLANE = -7      # caller rank plane differs from the native grid

_LOCK = threading.Lock()
_CACHE = {}


@intrinsic
def _call_uncompress2(typingctx, address, dest, dest_length, source, source_length):
    """``int uncompress2(Bytef*, uLongf*, const Bytef*, uLong*)`` at a runtime address."""
    signature = types.int32(address, dest, dest_length, source, source_length)

    def codegen(context, builder, signature, args):
        function_address, *pointers = args
        pointer_type = context.get_value_type(types.voidptr)
        function_type = ir.FunctionType(ir.IntType(32), [pointer_type]*4)
        function = builder.inttoptr(function_address, function_type.as_pointer())
        return builder.call(function, pointers)
    return signature, codegen


@_numba.njit(nogil=True, cache=True)
def _word(payload, at):
    return (np.int64(payload[at]) | (np.int64(payload[at+1]) << 8)
            | (np.int64(payload[at+2]) << 16) | (np.int64(payload[at+3]) << 24))


@_numba.njit(nogil=True, cache=True)
def _row_fits(payload, params):
    """The row range lies in the payload and could hold its chunk headers."""
    offset, length, chunks = params[0], params[1], params[2]
    return not (offset < 0 or length < 0 or chunks < 0 or offset > payload.shape[0]
                or length > payload.shape[0]-offset or chunks > length//HEADER_BYTES)


@_numba.njit(nogil=True, cache=True)
def _scan_row(payload, params, headers, base):
    """Validate a row's chunk headers like the Python reader; return (status, total area).

    ``headers[base+c]`` receives (position, y0, y1, x0, x1, mask bytes, score
    bytes, packed mask bytes, raw score bytes) of chunk ``c``.
    """
    offset, length, chunks = params[0], params[1], params[2]
    row_y0, row_y1, row_x0, row_x1 = params[3], params[4], params[5], params[6]
    height, width, lane, limit = params[7], params[8], params[9], params[10]
    consumed = 0
    total = 0
    for chunk in range(chunks):
        if HEADER_BYTES > length-consumed:
            return STATUS_HEADER, 0
        at = offset+consumed
        y0, y1 = _word(payload, at), _word(payload, at+4)
        x0, x1 = _word(payload, at+8), _word(payload, at+12)
        mask_bytes, score_bytes = _word(payload, at+16), _word(payload, at+20)
        mask_raw, score_raw = _word(payload, at+32), _word(payload, at+36)
        if not (row_y0 <= y0 < y1 <= row_y1 <= height and row_x0 <= x0 < x1 <= row_x1 <= width):
            return STATUS_GEOMETRY, 0
        # Either side above the limit already fails area <= limit (both are >= 1),
        # and checking first keeps the product inside int64.
        if y1-y0 > limit or x1-x0 > limit:
            return STATUS_GEOMETRY, 0
        area = (y1-y0)*(x1-x0)
        if area > limit or mask_raw != (area+7)//8 or score_raw != (area if lane != 0 else 0):
            return STATUS_GEOMETRY, 0
        if (mask_bytes > mask_raw+1024 or score_bytes > score_raw+1024
                or (score_bytes != 0) != (lane != 0)
                or mask_bytes+score_bytes+HEADER_BYTES > length-consumed):
            return STATUS_LENGTHS, 0
        entry = base+chunk
        headers[entry, 0] = at
        headers[entry, 1], headers[entry, 2], headers[entry, 3], headers[entry, 4] = y0, y1, x0, x1
        headers[entry, 5], headers[entry, 6] = mask_bytes, score_bytes
        headers[entry, 7], headers[entry, 8] = mask_raw, score_raw
        consumed += HEADER_BYTES+mask_bytes+score_bytes
        total += area
    if consumed != length:
        return STATUS_UNINDEXED, 0
    return 0, total


@_numba.njit(nogil=True, cache=True)
def _row_headers(payload, params):
    """Allocate and fill the header table; returns (status, total area, headers)."""
    if not _row_fits(payload, params):
        return STATUS_PAYLOAD, 0, np.empty((0, 9), np.int64)
    headers = np.empty((params[2], 9), np.int64)
    status, total = _scan_row(payload, params, headers, 0)
    return status, total, headers


@_numba.njit(nogil=True, cache=True)
def _inflate_exact(inflate, payload, at, encoded, output, start, raw, lengths):
    """True only when ``payload[at:at+encoded]`` is exactly one zlib stream of ``raw`` bytes.

    ``output[start:start+raw+1]`` must be writable; the extra byte detects a
    longer stream exactly as the Python reader's ``max_length=raw+1``.
    """
    if encoded <= 0:
        return False
    lengths[0] = raw+1
    lengths[1] = encoded
    status = _call_uncompress2(inflate, _void_pointer(output.ctypes.data+np.uintp(start)),
                               _void_pointer(lengths.ctypes.data),
                               _void_pointer(payload.ctypes.data+np.uintp(at)),
                               _void_pointer(lengths.ctypes.data+np.uintp(lengths.itemsize)))
    return status == 0 and np.int64(lengths[0]) == raw and np.int64(lengths[1]) == encoded


# _BIT_WORDS[v] holds the eight bytes (v >> k) & 1, k = 0..7, in memory order,
# so one 64-bit store unpacks a whole byte exactly as unpackbits(bitorder='little').
_BIT_WORDS = np.frombuffer(np.unpackbits(np.array(range(256), np.uint8).reshape(256, 1), axis=1,
                                         bitorder='little').tobytes(), np.uint64).copy()


@_numba.njit(nogil=True, cache=True)
def _unpack_words(packed, start, words, word, count):
    """``np.unpackbits(packed[start:], bitorder='little', count=count)`` into ``words[word:]`` bytes.

    The last partial byte writes up to seven pad bytes after ``count``; callers
    give every strip an 8-aligned slot, so they land in that slot's own pad.
    """
    for byte in range((count+7) >> 3):
        words[word+byte] = _BIT_WORDS[packed[start+byte]]


@_numba.njit(nogil=True, cache=True)
def decode_row(inflate, lengths, payload, params, include_scores):
    """Decode one row; returns (status, bounds, starts, masks, scores, read bytes).

    Chunk ``c`` is ``masks[starts[c]:starts[c]+area]`` (and the same slice of
    ``scores`` when ``include_scores`` and the row has a score lane) with
    ``bounds[c] = (y0, y1, x0, x1)``; starts are 8-aligned. Score streams are
    neither read nor validated when ``include_scores`` is false, as in the
    Python reader.
    """
    status, _total, headers = _row_headers(payload, params)
    chunks = headers.shape[0]
    bounds = np.empty((chunks, 4), np.int64)
    starts = np.zeros(chunks, np.int64)
    if status != 0:
        return status, bounds, starts, np.empty(0, np.uint8), np.empty(0, np.uint8), 0
    with_scores = include_scores and params[9] != 0
    slots = 0
    largest = 1
    for chunk in range(chunks):
        starts[chunk] = slots
        area = (headers[chunk, 2]-headers[chunk, 1])*(headers[chunk, 4]-headers[chunk, 3])
        slots += (area+7) & ~7
        largest = max(largest, headers[chunk, 7]+1)
    words = np.empty(slots >> 3, np.uint64)
    masks = words.view(np.uint8)
    # One trailing byte lets the last stream's extra-output probe land in bounds.
    scores = np.empty(slots+1 if with_scores else 0, np.uint8)
    scratch = np.empty(largest, np.uint8)
    read = 0
    for chunk in range(chunks):
        position = headers[chunk, 0]
        mask_bytes, score_bytes = headers[chunk, 5], headers[chunk, 6]
        mask_raw, score_raw = headers[chunk, 7], headers[chunk, 8]
        area = (headers[chunk, 2]-headers[chunk, 1])*(headers[chunk, 4]-headers[chunk, 3])
        if not _inflate_exact(inflate, payload, position+HEADER_BYTES, mask_bytes,
                              scratch, 0, mask_raw, lengths):
            return STATUS_INFLATE, bounds, starts, np.empty(0, np.uint8), np.empty(0, np.uint8), 0
        _unpack_words(scratch, 0, words, starts[chunk] >> 3, area)
        if with_scores and not _inflate_exact(inflate, payload, position+HEADER_BYTES+mask_bytes,
                                              score_bytes, scores, starts[chunk], score_raw, lengths):
            return STATUS_INFLATE, bounds, starts, np.empty(0, np.uint8), np.empty(0, np.uint8), 0
        bounds[chunk, 0], bounds[chunk, 1] = headers[chunk, 1], headers[chunk, 2]
        bounds[chunk, 2], bounds[chunk, 3] = headers[chunk, 3], headers[chunk, 4]
        read += HEADER_BYTES+mask_bytes+(score_bytes if include_scores else 0)
    return 0, bounds, starts, masks, scores, read


# Rank-merge table columns: the row parameters, then the rank level and whether
# that level checks the previous level for gaps.
RANK_LEVEL, RANK_CHECK = len(ROW_PARAMETERS), len(ROW_PARAMETERS)+1


@_numba.njit(nogil=True, cache=True)
def merge_rank_rows(inflate, lengths, payload, table, plane):
    """Merge rows into a rank plane exactly as ``_merge_rank_crop`` per strip, in table order.

    Returns (status, outcome, top, bottom, left, right): ``outcome[r]`` is
    (populated, gap, read bytes) of row ``r`` and the box covers populated
    strips. Every row is validated and every mask stream inflated before the
    plane is touched, so a nonzero status leaves the plane unchanged.
    """
    rows = table.shape[0]
    outcome = np.zeros((rows, 3), np.int64)
    height, width = plane.shape[0], plane.shape[1]
    total = 0
    for row in range(rows):
        if not _row_fits(payload, table[row]):
            return STATUS_PAYLOAD, outcome, height, 0, width, 0
        if table[row, 7] != height or table[row, 8] != width:
            return STATUS_PLANE, outcome, height, 0, width, 0
        total += table[row, 2]
    headers = np.empty((total, 9), np.int64)
    firsts = np.empty(rows+1, np.int64)
    firsts[0] = 0
    for row in range(rows):
        status, _area = _scan_row(payload, table[row], headers, firsts[row])
        if status != 0:
            return status, outcome, height, 0, width, 0
        firsts[row+1] = firsts[row]+table[row, 2]
    starts = np.empty(total+1, np.int64)
    starts[0] = 0
    for chunk in range(total):
        starts[chunk+1] = starts[chunk]+headers[chunk, 7]
    packed = np.empty(starts[total]+1, np.uint8)
    for row in range(rows):
        for chunk in range(firsts[row], firsts[row+1]):
            if not _inflate_exact(inflate, payload, headers[chunk, 0]+HEADER_BYTES, headers[chunk, 5],
                                  packed, starts[chunk], headers[chunk, 7], lengths):
                return STATUS_INFLATE, outcome, height, 0, width, 0
            outcome[row, 2] += HEADER_BYTES+headers[chunk, 5]
    top, bottom, left, right = height, 0, width, 0
    for row in range(rows):
        level, check = table[row, RANK_LEVEL], table[row, RANK_CHECK] != 0
        previous = level-1
        for chunk in range(firsts[row], firsts[row+1]):
            y0, y1, x0, x1 = headers[chunk, 1], headers[chunk, 2], headers[chunk, 3], headers[chunk, 4]
            start, columns = starts[chunk], x1-x0
            strip = gap = False
            bit = 0
            # Walk each strip row through its packed bytes: empty bytes are
            # skipped and full bytes skip the bit tests. Writes and gap tests
            # stay per pixel in row-major order, as in the per-crop merge.
            for line in range(y0, y1):
                stop = bit+columns
                column = x0
                while bit < stop:
                    shift = bit & 7
                    take = min(8-shift, stop-bit)
                    ones = (1 << take)-1
                    value = (np.int64(packed[start+(bit >> 3)]) >> shift) & ones
                    if value != 0:
                        strip = True
                        if value == ones:
                            for offset in range(take):
                                if check and plane[line, column+offset] < previous:
                                    gap = True
                                plane[line, column+offset] = level
                        else:
                            for offset in range(take):
                                if (value >> offset) & 1:
                                    if check and plane[line, column+offset] < previous:
                                        gap = True
                                    plane[line, column+offset] = level
                    bit += take
                    column += take
            if strip:
                outcome[row, 0] = 1
                top, bottom = min(top, y0), max(bottom, y1)
                left, right = min(left, x0), max(right, x1)
            if gap:
                outcome[row, 1] = 1
    return 0, outcome, top, bottom, left, right


@_numba.njit(nogil=True, cache=True)
def store_rank_region(target, plane, top, bottom, left, right):
    """``target[top:bottom, left:right] = plane[...]`` then zero that plane region."""
    # Row views keep the inner loops on unit-stride, unsigned-free indexing so
    # they vectorize (an element loop over 2-D signed indices runs ~8x slower).
    for line in range(top, bottom):
        source = plane[line, left:right]
        destination = target[line, left:right]
        for column in range(source.shape[0]):
            destination[column] = source[column]
        for column in range(source.shape[0]):
            source[column] = 0


class NativeViewDecoder:
    """One qualified zlib ``uncompress2`` entry point and the row kernels bound to it."""

    def __init__(self, inflater):
        self.inflater = inflater  # owns the loaded library for the process lifetime
        self.version = inflater.version
        self.source = inflater.source
        self.address = int(ctypes.cast(inflater._inflater, ctypes.c_void_p).value or 0)
        if not self.address:
            raise ValueError('Native zlib uncompress2 has no entry address')
        self.ulong_dtype = np.dtype(inflater._ulong)
        self._local = threading.local()
        self._probe()

    def _lengths(self):
        # uLong in/out lengths; per thread because the kernels write them.
        lengths = getattr(self._local, 'lengths', None)
        if lengths is None:
            lengths = self._local.lengths = np.zeros(2, self.ulong_dtype)
        return lengths

    def decode_row(self, payload, params, include_scores):
        return decode_row(self.address, self._lengths(), payload, params, bool(include_scores))

    def merge_rank_rows(self, payload, table, plane):
        return merge_rank_rows(self.address, self._lengths(), payload, table, plane)

    def _probe(self):
        """Every kernel result must equal the Python reader on a bounded row set."""
        payload, rows = _probe_rows()
        for params, lane, crops, mask_valid, valid in rows:
            for include_scores in (False, True):
                accepted = valid or (mask_valid and not include_scores)
                status, bounds, starts, masks, scores, _ = self.decode_row(payload, params, include_scores)
                if not accepted:
                    if status == 0:
                        raise ValueError('Native view decode accepted a malformed probe row')
                    continue
                if status != 0 or bounds.tolist() != [list(crop[:4]) for crop in crops]:
                    raise ValueError('Native view decode rejected or misplaced a valid probe row')
                for chunk, (_, _, _, _, mask, score) in enumerate(crops):
                    part = slice(int(starts[chunk]), int(starts[chunk])+mask.size)
                    if masks[part].tobytes() != mask.tobytes():
                        raise ValueError('Native view decode mask differs from numpy unpackbits')
                    if include_scores and lane and scores[part].tobytes() != score.tobytes():
                        raise ValueError('Native view decode scores differ from Python zlib')
            plane = np.zeros((int(params[7]), int(params[8])), np.uint8)
            plane[4:, :] = 2
            reference = plane.copy()
            table = np.concatenate((params, np.array((3, 1), np.int64))).reshape(1, -1)
            status, outcome, *box = self.merge_rank_rows(payload, table, plane)
            if not mask_valid:
                if status == 0 or plane.tobytes() != reference.tobytes():
                    raise ValueError('Native rank merge accepted or wrote a malformed probe row')
                continue
            gap = populated = False
            for y0, y1, x0, x1, mask, _ in crops:
                support = mask.astype(bool)
                gap |= bool(np.any(reference[y0:y1, x0:x1][support] < 2))
                populated |= bool(support.any())
                reference[y0:y1, x0:x1][support] = 3
            if (status != 0 or bool(outcome[0, 0]) != populated or bool(outcome[0, 1]) != gap
                    or plane.tobytes() != reference.tobytes()):
                raise ValueError('Native rank merge differs from the per-crop reference merge')
            target, expected = np.zeros_like(plane), np.zeros_like(plane)
            region = (slice(box[0], box[1]), slice(box[2], box[3]))
            expected[region] = reference[region]
            reference[region] = 0
            store_rank_region(target, plane, *box)
            if target.tobytes() != expected.tobytes() or plane.tobytes() != reference.tobytes():
                raise ValueError('Native rank region store differs from numpy assignment')

    def warm(self):
        self._probe()


def _probe_rows():
    """A payload of valid and malformed rows, written exactly as the journal writer does.

    Rows are (params, score lane, crops, mask-only valid, valid).
    """
    height, width = 9, 13
    pattern = np.zeros((height, width), np.uint8)
    values = np.zeros((height, width), np.uint8)
    for y in range(height):
        for x in range(width):
            pattern[y, x] = 1 if (y*7+x*3) % 5 < 2 else 0
            values[y, x] = (y*31+x*17) % 251
    payload = bytearray()
    rows = []

    def strip(y0, y1, x0, x1, lane, mask_tail, score_tail):
        piece = np.ascontiguousarray(pattern[y0:y1, x0:x1])
        packed = np.packbits(piece, axis=None, bitorder='little').tobytes()
        mask = zlib.compress(packed, 1)+mask_tail
        raw = np.where(piece != 0, values[y0:y1, x0:x1], np.uint8(0)).tobytes() if lane else b''
        score = zlib.compress(raw, 1)+score_tail if lane else b''
        head = np.array((y0, y1, x0, x1, len(mask), len(score), 0, 0, len(packed), len(raw)), '<u4').tobytes()
        crop = (y0, y1, x0, x1, piece, np.frombuffer(raw, np.uint8) if lane else None)
        return head+mask+score, crop

    def add(strips, lane, mask_valid=True, valid=True, mask_tail=b'', score_tail=b''):
        start = len(payload)
        crops = []
        for y0, y1 in strips:
            data, crop = strip(y0, y1, 2, 11, lane, mask_tail, score_tail)
            payload.extend(data)
            crops.append(crop)
        params = np.array((start, len(payload)-start, len(strips), strips[0][0], strips[-1][1], 2, 11,
                           height, width, int(lane), 1024), np.int64)
        rows.append((params, lane, crops, mask_valid, valid))

    add(((1, 4), (4, 7), (7, 9)), True)            # three odd-area strips with scores
    add(((1, 9),), False)                          # one 72-pixel strip, no score lane
    add(((1, 3), (3, 9)), True, mask_valid=False, valid=False, mask_tail=bytes([0]))
    add(((1, 4),), True, valid=False, score_tail=bytes([1]))   # only score reads see it
    return np.frombuffer(bytes(payload), np.uint8), rows


def native_view_decoder():
    """Return (decoder or None, receipt) for the journal reader's row kernels."""
    mode = os.environ.get('XTA_YOLO_SWEEP_NATIVE_DECODE', 'auto').strip().lower()
    if mode in ('0', 'false', 'off', 'python'):
        return None, dict(backend='python_zlib', python_zlib_version=zlib.ZLIB_RUNTIME_VERSION,
                          fallback_reason='disabled_by_configuration')
    from .confidence_native_inflate import native_confidence_inflater
    inflater, inflate_receipt = native_confidence_inflater()
    key = id(inflater)
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is None or entry[0] is not inflater:
            receipt = dict(backend='python_zlib', python_zlib_version=zlib.ZLIB_RUNTIME_VERSION)
            decoder = None
            if inflater is None:
                receipt['fallback_reason'] = inflate_receipt.get('fallback_reason') or 'native_inflate_unavailable'
            elif inflater.version != zlib.ZLIB_RUNTIME_VERSION:
                receipt.update(codec_version=inflater.version,
                               fallback_reason='native_zlib_version_differs_from_python_zlib')
            else:
                try:
                    decoder = NativeViewDecoder(inflater)
                    receipt.update(backend='native_uncompress2', codec_version=decoder.version,
                                   library_source=decoder.source, fallback_reason=None)
                except Exception as exc:
                    receipt['fallback_reason'] = f'{type(exc).__name__}:{str(exc)[:160]}'
            entry = (inflater, decoder, receipt)
            _CACHE[key] = entry
            print('Native view journal decode: '+', '.join(f'{name}={value}' for name, value in receipt.items()),
                  flush=True)
        _, decoder, receipt = entry
    return decoder, dict(receipt)


def warm_native_view_decode_kernels():
    """Load, probe and compile the journal row kernels before the process is busy."""
    started = time.perf_counter()
    decoder, receipt = native_view_decoder()
    if decoder is not None:
        decoder.warm()
        receipt.update(decode_signatures=len(decode_row.nopython_signatures),
                       merge_signatures=len(merge_rank_rows.nopython_signatures))
    receipt['seconds'] = time.perf_counter()-started
    return receipt


__all__ = ['NativeViewDecoder', 'decode_row', 'merge_rank_rows', 'native_view_decoder',
           'store_rank_region', 'warm_native_view_decode_kernels']
