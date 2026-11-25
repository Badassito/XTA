"""Certify when a stricter cutoff can reuse its family's floor projection."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from threading import Lock
import time
import zlib

import numpy as np

from ._deps import _numba
from .confidence_storage import ConfidenceStageLimit
from .json_publication import write_json_atomic
from .run_transport import _stamp


_MIB = 1024 ** 2
_SCORE_FILES = ("metadata.json", "index.bin", "scores.u8.zlib")
_NATIVE_RECORD = struct.Struct("<IBB")
_NATIVE_CHUNK_BYTES = _MIB


def _signature(path):
    return {
        name: list(_stamp((path / name).stat()))
        for name in _SCORE_FILES
        if (path / name).is_file() and not (path / name).is_symlink()
    }


def _native_chunks(shape, read_native):
    """Visit fixed-size native chunks in raster order with no dense copy."""
    plane_size = math.prod(shape[1:])
    for z in range(shape[0]):
        plane = read_native(z)
        if plane is not None:
            plane = np.asarray(plane)
            if plane.shape != shape[1:] or plane.dtype != np.uint8:
                raise ValueError("Native scores differ from their declared grid")
            flat = plane.reshape(-1) if plane.flags.c_contiguous else plane.flat
        for first in range(0, plane_size, _NATIVE_CHUNK_BYTES):
            stop = min(plane_size, first + _NATIVE_CHUNK_BYTES)
            yield stop - first, None if plane is None else flat[first:stop]


def _write_native_proof(path, shape, read_native, limit):
    """Stage a sequential certificate source optimized for whole-cutoff scans."""
    used = 0
    with path.open("wb") as stream:
        for _, chunk in _native_chunks(shape, read_native):
            maximum = 0 if chunk is None else int(chunk.max(initial=0))
            minimum, encoded = 255, b""
            if maximum:
                minimum = int(chunk.min(where=chunk != 0, initial=255))
                encoded = zlib.compress(chunk, level=1)
            used += _NATIVE_RECORD.size + len(encoded)
            if used > limit:
                raise ConfidenceStageLimit("Floor native score proof exceeds its reservation")
            stream.write(_NATIVE_RECORD.pack(len(encoded), minimum, maximum))
            stream.write(encoded)
    return used


@_numba.njit(cache=True, nogil=True)
def _changed_scores(original, current):
    """Compare a nonempty packet in one pass without temporary mask arrays."""
    largest_dropped, smallest_kept = 0, 256
    mismatch = False
    for index in range(original.size):
        before, after = int(original[index]), int(current[index])
        largest_dropped = max(largest_dropped, before if after == 0 else 0)
        smallest_kept = min(smallest_kept, after if after != 0 else 256)
        mismatch |= after != 0 and before != after
    return largest_dropped, smallest_kept, mismatch


def _threshold_for_native_scores(path, shape, read_native):
    """Return an exact threshold for the already masked native score arrays.

    Positive stricter bytes must equal the floor at the same coordinates. All
    dropped floor bytes must be lower than every retained byte. In particular,
    a quantized score split across two cutoff decisions fails this certificate.
    """
    largest_dropped, smallest_kept = 0, 256
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        for size, current in _native_chunks(shape, read_native):
            length, minimum, maximum = _NATIVE_RECORD.unpack(stream.read(_NATIVE_RECORD.size))
            if (bool(length) != bool(maximum) or (length and not 0 < minimum <= maximum)
                    or length > size + (size >> 8) + 64 or stream.tell() + length > file_size):
                raise ValueError("Invalid floor native score record")
            active = current is not None and bool(np.any(current))
            if not active:
                stream.seek(length, os.SEEK_CUR)
                largest_dropped = max(largest_dropped, maximum)
            elif not length:
                return None
            else:
                decoder = zlib.decompressobj()
                raw = decoder.decompress(stream.read(length), size + 1)
                if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError("Invalid compressed floor native scores")
                original = np.frombuffer(raw, np.uint8)
                if np.array_equal(original, current):
                    smallest_kept = min(smallest_kept, minimum)
                else:
                    dropped, kept, mismatch = _changed_scores(original, current)
                    if mismatch:
                        return None
                    smallest_kept = min(smallest_kept, kept)
                    largest_dropped = max(largest_dropped, dropped)
            if largest_dropped >= smallest_kept:
                return None
        if stream.tell() != file_size:
            raise ValueError("Unexpected trailing floor native scores")
    return largest_dropped + 1


class _ThresholdedProjection:
    def __init__(self, source, threshold, copy_from):
        self.source = source
        self.threshold = threshold
        self.copy_from = copy_from

    def iter_crops(self, z):
        if self.threshold > 255:
            return
        for y0, y1, x0, x1, values in self.source.iter_crops(z):
            if self.threshold > 1:
                values = np.where(values >= self.threshold, values, np.uint8(0))
            yield y0, y1, x0, x1, values


class SweepScoreProjection:
    """Private, capacity-bounded floor cache; the coordinator retires it."""

    def __init__(self, work, *, sweep_id, cutoff, floor, budget_bytes):
        self.work = Path(work)
        self.root = self.work / "confidence-projection-floor"
        self.sweep_id = sweep_id
        self.cutoff = cutoff
        self.floor = floor
        self.budget_bytes = budget_bytes
        self._lock = Lock()
        self._reserved = 0
        self._retained = 0

    @classmethod
    def from_environment(cls, sweep_id, cutoff):
        work = os.environ.get("XTA_YOLO_SWEEP_WORK_DIR")
        if not work or sweep_id is None or cutoff is None:
            return None
        work = Path(work)
        if work.is_symlink() or not work.name.startswith("xta-confidence-sweep-"):
            return None
        try:
            owner = json.loads((work / "owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(owner, dict) or owner.get("sweep_id") != sweep_id:
            return None
        try:
            floor = int(os.environ.get("XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS", "-1"))
            # Use the configured sweep cache ceiling. Each layer also needs
            # local scratch headroom before it can reserve storage.
            budget = float(os.environ.get("XTA_YOLO_SWEEP_CACHE_GIB", "64"))
        except ValueError:
            return None
        if not math.isfinite(budget) or budget <= 0 or not 0 <= floor <= cutoff <= 100:
            return None
        if cutoff == floor:
            from .runtime import path_is_memory_backed
            from .workspace import available_anon_work_bytes
            budget_bytes = int(budget * 1024 ** 3)
            if path_is_memory_backed(work):
                # Bound files retained until family completion by the current
                # memory quota. Most of the remaining headroom belongs to the
                # active pipeline and its changing set of view workspaces.
                budget_bytes = min(budget_bytes, available_anon_work_bytes() // 8)
        else:
            budget_bytes = 0
        return cls(work, sweep_id=sweep_id, cutoff=cutoff, floor=floor, budget_bytes=budget_bytes)

    def _path(self, model_name, layer_key):
        key = hashlib.sha256((str(model_name) + "\0" + str(layer_key)).encode()).hexdigest()
        return self.root / key

    def _reserve(self, wanted):
        with self._lock:
            available = shutil.disk_usage(self.work).free
            booked = min(wanted, self.budget_bytes - self._retained - self._reserved,
                         available - max(64 * _MIB, available // 4) - self._reserved)
            if booked <= 0:
                return 0
            self._reserved += booked
            return booked

    def _settle(self, reserved, retained):
        with self._lock:
            self._reserved -= reserved
            self._retained += retained

    def remember(self, native_shape, read_native, source, *, provenance, minimum_projected_score=1):
        """Copy the already projected floor once, with its native score proof."""
        from .confidence_publication import copy_staged_blocks

        if self.cutoff != self.floor or self.budget_bytes <= 0:
            return "not_floor"
        reserved = retained = 0
        try:
            files = _signature(source.path)
            if tuple(files) != _SCORE_FILES:
                return "unsupported_floor_evidence"
            source_bytes = sum((source.path / name).stat().st_size for name in files)
            plane_size = math.prod(native_shape[1:])
            chunks_per_plane = (plane_size + _NATIVE_CHUNK_BYTES - 1) // _NATIVE_CHUNK_BYTES
            chunks = native_shape[0] * chunks_per_plane
            allowance = 64 * 1024
            # Includes the record header and zlib's worst-case output overhead.
            native_bytes = math.prod(native_shape)
            wanted = source_bytes + native_bytes + native_bytes // 256 + chunks * (64 + _NATIVE_RECORD.size) + allowance
            reserved = self._reserve(wanted)
            if reserved <= source_bytes + allowance:
                return "capacity"
            self.root.mkdir(exist_ok=True)
            if self.root.is_symlink():
                return "redirected_floor_cache"
            destination = self._path(source.model_name, source.layer_key)
            with tempfile.TemporaryDirectory(dir=self.root, prefix=".floor-") as directory:
                stage = Path(directory) / "layer"
                projected = stage / "projected"
                stage.mkdir()
                proof = stage / "native.scores"
                proof_bytes = _write_native_proof(
                    proof, native_shape, read_native, reserved - source_bytes - allowance)
                copy_staged_blocks(source.path, projected)
                if files != _signature(source.path):
                    return "floor_evidence_changed"
                receipt_path = stage / "receipt.json"
                write_json_atomic(receipt_path, {
                    "sweep_id": self.sweep_id, "floor": self.floor,
                    "native_shape": list(native_shape), "source_shape": list(source.shape),
                    "projection": provenance,
                    "minimum_projected_score": minimum_projected_score,
                    "proof_file": _stamp(proof.stat()),
                    "projected_files": _signature(projected),
                })
                used = source_bytes + proof_bytes + receipt_path.stat().st_size
                if used > reserved:
                    return "capacity"
                stage.rename(destination)
                retained = used
        except (OSError, ValueError, ConfidenceStageLimit):
            return "capacity_or_io"
        finally:
            self._settle(reserved, retained)
        return "stored"

    @contextmanager
    def reuse(self, native_shape, read_native, shape, *, model_name, layer_key, provenance, metrics):
        """Reuse numeric max only after establishing equal projected bytes."""
        from .confidence_evidence import ConfidenceEvidenceRef

        path = self._path(model_name, layer_key)
        metrics["source_projection_reused"] = False
        receipt_path = path / "receipt.json"
        if self.cutoff <= self.floor:
            yield None
            return
        if not receipt_path.is_file():
            metrics["score_reuse_reason"] = "floor_cache_unavailable"
            yield None
            return
        try:
            proof, projected_path = path / "native.scores", path / "projected"
            if any(member.is_symlink() for member in (self.root, path, receipt_path, proof, projected_path)):
                raise ValueError("Floor score cache contains a redirected path")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt["sweep_id"] != self.sweep_id or receipt["floor"] != self.floor
                    or receipt["native_shape"] != list(native_shape) or receipt["source_shape"] != list(shape)
                    or receipt["projection"] != provenance):
                raise ValueError("Floor score cache belongs to another geometry or sweep")
            identity = receipt["projected_files"]
            proof_identity = tuple(receipt["proof_file"])
            if (set(identity) != set(_SCORE_FILES) or identity != _signature(projected_path)
                    or proof_identity != _stamp(proof.stat())):
                raise ValueError("Floor score cache changed")
            projected = ConfidenceEvidenceRef.open(projected_path)
            if (projected.coordinate_space != "source" or projected.shape != shape
                    or (projected.model_name, projected.layer_key) != (str(model_name), str(layer_key))):
                raise ValueError("Floor score cache differs from the requested layer")
            started = time.perf_counter()
            threshold = _threshold_for_native_scores(proof, native_shape, read_native)
            metrics["score_reuse_proof_seconds"] = time.perf_counter() - started
            if proof_identity != _stamp(proof.stat()):
                raise ValueError("Floor native scores changed during certification")
            can_copy = threshold is not None and threshold <= int(receipt.get("minimum_projected_score", 1))
            reader = projected.reader() if threshold is not None and not can_copy and threshold <= 255 else nullcontext(None)
        except (OSError, ValueError, KeyError, TypeError, struct.error, zlib.error):
            metrics["score_reuse_reason"] = "invalid_floor_cache"
            yield None
            return
        if threshold is None:
            metrics["score_reuse_reason"] = "native_scores_are_not_an_exact_threshold"
            yield None
            return
        with reader as source:
            metrics.update(source_projection_reused=True, score_reuse_native_threshold=threshold)
            yield _ThresholdedProjection(source, threshold, projected_path if can_copy else None)
            if receipt["projected_files"] != _signature(path / "projected"):
                raise ValueError("Floor projection changed while reading its cached scores")
