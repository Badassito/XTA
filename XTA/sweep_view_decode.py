"""Decode bounded sweep-journal rows and merge rank planes with a qualified zlib.

The AOT CPU kernels validate strip geometry and compressed input lengths. A
failed merge leaves the caller's plane untouched so the stream reader can
replay the row with its original exception and partial-yield behavior.
"""
from __future__ import annotations

import ctypes
import os
import threading
import zlib

import numpy as np

from .native_cpu import kernels, load_native_cpu

HEADER_BYTES = 40
ROW_PARAMETERS = ('offset', 'length', 'chunks', 'y0', 'y1', 'x0', 'x1',
                  'height', 'width', 'score', 'chunk_bytes')
RANK_LEVEL, RANK_CHECK = len(ROW_PARAMETERS), len(ROW_PARAMETERS)+1
STATUS_PAYLOAD, STATUS_HEADER, STATUS_GEOMETRY = -1, -2, -3
STATUS_LENGTHS, STATUS_UNINDEXED, STATUS_INFLATE, STATUS_PLANE = -4, -5, -6, -7

_LOCK = threading.Lock()
_CACHE = {}


def decode_row(inflate, lengths, payload, params, include_scores):
    """Decode one row. Native code owns the per-call zlib lengths on its stack."""
    return kernels.journal_decode_row(inflate, payload, params, include_scores)


def merge_rank_rows(inflate, lengths, payload, table, plane):
    """Merge validated rows in table order using the existing decoder call contract."""
    return kernels.journal_merge_rank_rows(inflate, payload, table, plane)


def store_rank_region(target, plane, top, bottom, left, right):
    """Copy the selected rectangular rank region and zero its source."""
    return kernels.journal_store_rank_region(target, plane, top, bottom, left, right)


class NativeViewDecoder:
    """Retain the qualified library while its function address is in native use."""

    def __init__(self, inflater):
        self.inflater = inflater
        self.version = inflater.version
        self.source = inflater.source
        self.address = int(ctypes.cast(inflater._inflater, ctypes.c_void_p).value or 0)
        if not self.address:
            raise ValueError('Native zlib uncompress2 has no entry address')
        self._decode = kernels.journal_decode_row
        self._merge = kernels.journal_merge_rank_rows
        self._probe()

    def decode_row(self, payload, params, include_scores):
        return self._decode(self.address, payload, params, bool(include_scores))

    def merge_rank_rows(self, payload, table, plane):
        return self._merge(self.address, payload, table, plane)

    def merge_family_rank_rows(self, payload, records, positions, checks, plane, chunk_limit):
        return kernels.journal_merge_family_rank_rows(
            self.address, payload, records, positions, checks, plane, chunk_limit)

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
                # A missing build is a workload setup failure, independent of zlib qualification.
                load_native_cpu()
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


__all__ = ['NativeViewDecoder', 'decode_row', 'merge_rank_rows', 'native_view_decoder', 'store_rank_region']
