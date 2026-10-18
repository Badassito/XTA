"""Copy compact SAM timing receipts, never masks, from a running job.

This standalone, standard-library tool opens the artifact TAR read-only without
its writer lock. It authenticates complete commits within the initial file size,
then verifies selected JSON payloads. An incomplete/racing tail is reported and
ignored. Optional live polling captures short-lived raw manifests concurrently.
Times are host stage durations, not GPU kernel durations; some timers overlap.
Use --format compact for lossless shared metadata and --resume-from PREVIOUS.zip
when its summary reports a next_cursor. Each part retains the same byte budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile

MIB = 1024 ** 2
MAX_COMMIT = 64 * MIB
MAX_PAX_TOTAL = 64 * MIB
MAX_PAX_CHAIN = 16
MAX_INDEX = 128 * MIB  # Same encoded-index limit as SamEvidenceBundle.
MAX_MANIFEST = 4 * MIB
MAX_MEMBERS = 200_000
MAX_JSON_TOTAL = 1024 * MIB
MAX_OUTPUT = 128 * MIB  # Per JSONL stream; rows are never truncated.
MAX_ROWS = 100_000
MAX_OBJECTS = 200_000
OBJECT_THRESHOLD = 512
SCHEMA = "xta.sam-runtime-receipt-capture/1"
COMPACT_SCHEMA = "xta.sam-runtime-receipt-capture/2"
CURSOR_SCHEMA = "xta.sam-runtime-receipt-cursor/1"
RECEIPT_KEYS = (
    "schema", "run_id", "status", "prediction_valid", "coverage_complete",
    "crop_xyxy", "seed_frame", "frame_range", "expected_frames", "direction",
    "image_cache", "image_cache_lifetime", "input_preparation",
    "request_metadata", "timings", "cpu_task_retirement", "dispatch",
    "feature_cache_before", "feature_cache_after", "adapter_receipt",
    "profile", "sam_runtime", "worker_startup_receipt", "session_cpu_admission",
    "cuda_quiescence", "sam_model", "seed_artifact_sha256",
)
RUN_KEYS = (
    "run_id", "group_id", "parent_run_id", "direction", "expected_frames",
    "observed_frames", "seed_ids", "held_out_ids", "edge_ids", "pass_index",
    "walk_back_index", "status", "generation_mode", "tile_id", "tile_index",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def safe_name(value):
    if not isinstance(value, str) or "\\" in value or "\0" in value:
        raise ValueError("Invalid artifact name")
    parts = value.split("/")
    if not value or any(p in ("", ".", "..") for p in parts) or ":" in parts[0]:
        raise ValueError("Unsafe artifact name")
    return value


class SnapshotReader(io.RawIOBase):
    """A fixed-length view; appends after open cannot extend this scan."""
    def __init__(self, stream, size):
        self.stream, self.size = stream, size

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=0):
        if whence == 2:
            offset, whence = self.size + offset, 0
        return self.stream.seek(offset, whence)

    def read(self, size=-1):
        remaining = max(0, self.size - self.tell())
        return self.stream.read(remaining if size < 0 else min(size, remaining))


class IncompleteTail(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


def validate_pax(data):
    """Only the plain metadata emitted by TarInfo.tobuf(PAX_FORMAT) is needed.

    In particular, do not let GNU sparse directives make tarfile read payloads
    internally while it is still parsing headers.
    """
    position = 0
    while position < len(data):
        space = data.find(b" ", position, position + 32)
        if space < 0 or not data[position:space].isdigit():
            raise ValueError("Invalid PAX record length")
        length = int(data[position:space])
        end = position + length
        if length <= space - position + 1 or end > len(data) or data[end - 1:end] != b"\n":
            raise ValueError("Invalid PAX record extent")
        key, separator, _value = data[space + 1:end - 1].partition(b"=")
        if not separator or key not in (b"path", b"size", b"hdrcharset"):
            raise ValueError("Unsupported PAX metadata field")
        position = end


def header_available(stream, offset, size, metadata_budget=None):
    budget = metadata_budget if metadata_budget is not None else {"pax_bytes": 0}
    chain = 0
    while size - offset >= 512:
        stream.seek(offset)
        block = stream.read(512)
        if block == bytes(512):
            return False
        header = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        if header.size < 0:
            raise ValueError("Negative TAR size")
        if header.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
            return True
        if header.type not in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            raise ValueError("Unsupported raw TAR header type")
        if header.size > MAX_COMMIT:
            raise ValueError("PAX metadata budget exceeded")
        chain += 1
        extent = 512 + ((header.size + 511) // 512) * 512
        if chain > MAX_PAX_CHAIN or budget["pax_bytes"] + extent > MAX_PAX_TOTAL:
            raise ValueError("PAX chain/cumulative metadata budget exceeded")
        if offset + extent > size:
            raise IncompleteTail("incomplete_pax_payload")
        budget["pax_bytes"] += extent
        validate_pax(stream.read(header.size))
        offset += extent
    if offset != size or chain:
        raise IncompleteTail("incomplete_pax_header" if chain else "incomplete_header")
    return False


def scan_committed(stream):
    """Track only metadata locations; skip every binary member by seeking."""
    size = os.fstat(stream.fileno()).st_size
    frozen = SnapshotReader(stream, size)
    selected, known, pending = {}, set(), {}
    transaction, end, commits, members = None, 0, 0, 0
    state = {"initial_bytes": size, "committed_end": 0, "commits": 0,
             "members": 0, "tail_status": "end_of_snapshot"}
    metadata_budget = {"pax_bytes": 0}
    try:
        if not header_available(frozen, 0, size, metadata_budget):
            return selected, state
        frozen.seek(0)
        with tarfile.open(fileobj=frozen, mode="r:") as archive:
            first = True
            while first or header_available(frozen, archive.offset, size, metadata_budget):
                first = False
                item = archive.next()
                archive.members.clear()  # tarfile otherwise retains every header.
                if item is None:
                    break
                if item.offset_data + item.size > size:
                    state["tail_status"] = "incomplete_payload"
                    break
                if not item.isfile() or item.size < 0:
                    raise ValueError("Nonregular artifact member")
                name = safe_name(item.name)
                members += 1
                if members > MAX_MEMBERS:
                    raise ValueError("Archive member-count budget exceeded")
                if name.startswith("DATA/"):
                    parts = name.split("/", 2)
                    if len(parts) != 3 or not re.fullmatch("[0-9a-f]{32}", parts[1]):
                        raise ValueError("Invalid DATA member")
                    current, logical = parts[1], safe_name(parts[2])
                    if (transaction is not None and current != transaction) or logical in pending:
                        raise ValueError("Interleaved or duplicated transaction")
                    transaction = current
                    pending[logical] = {"offset": item.offset_data, "bytes": item.size}
                    continue
                if name != "COMMIT/" + str(transaction) + ".json" or item.size > MAX_COMMIT:
                    raise ValueError("Invalid or oversized COMMIT member")
                frozen.seek(item.offset_data)
                commit = json.loads(frozen.read(item.size))
                if not isinstance(commit, dict):
                    raise ValueError("COMMIT JSON root is not an object")
                digest = commit.pop("sha256")
                rows = commit["members"]
                if (commit["schema"] != "xta.artifact-archive/1"
                        or commit["transaction"] != transaction
                        or type(commit["replace"]) is not bool
                        or hashlib.sha256(encoded(commit)).hexdigest() != digest
                        or not isinstance(rows, list) or len(rows) != len(pending)):
                    raise ValueError("COMMIT identity/checksum mismatch")
                published = {}
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError("COMMIT inventory row is not an object")
                    logical = safe_name(row["name"])
                    if (logical not in pending or logical in published
                            or type(row["bytes"]) is not int
                            or row["bytes"] != pending[logical]["bytes"]
                            or not isinstance(row["sha256"], str)
                            or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
                            or (not commit["replace"] and logical in known)):
                        raise ValueError("COMMIT inventory mismatch")
                    published[logical] = dict(pending[logical], sha256=row["sha256"],
                                             transaction=transaction)
                for logical, entry in published.items():
                    known.add(logical)
                    parts = PurePosixPath(logical).parts
                    if parts[0] in ("sam_interpolation", "sam_extrapolation") and parts[-1] in ("index.json", "manifest.json"):
                        selected[logical] = entry
                end = item.offset_data + ((item.size + 511) // 512) * 512
                commits += 1
                pending, transaction = {}, None
    except IncompleteTail as error:
        state["tail_status"] = error.status
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError,
            tarfile.TarError, EOFError, OSError) as error:
        # Preserve authenticated earlier commits even if the writer is midway
        # through a header at the captured tail. This is never called complete.
        state["tail_status"] = "stopped_before_unverified_tail"
        state["tail_error"] = f"{type(error).__name__}: {error}"
    if pending and state["tail_status"] == "end_of_snapshot":
        state["tail_status"] = "uncommitted_tail"
    state.update(committed_end=end, commits=commits, members=members,
                 pax_metadata_bytes=metadata_budget["pax_bytes"])
    return selected, state


def verified_json(stream, entry, limit):
    if entry["bytes"] > limit:
        raise ValueError("JSON member exceeds encoded-byte budget")
    stream.seek(entry["offset"])
    data = stream.read(entry["bytes"])
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("Committed JSON payload checksum mismatch")
    return json.loads(data)


class CaptureBudgetExceeded(ValueError):
    pass


def row_device(row):
    if not isinstance(row, dict):
        raise ValueError("Receipt row is not an object")
    receipt = row.get("runtime_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Runtime receipt is not an object")
    dispatch = receipt.get("dispatch", {})
    retirement = receipt.get("cpu_task_retirement", {})
    if not isinstance(dispatch, dict):
        raise ValueError("Receipt dispatch is not an object")
    if not isinstance(retirement, dict):
        raise ValueError("Receipt CPU retirement is not an object")
    return dispatch.get("execution_device_id", retirement.get("execution_device_id"))


class RowWriter:
    def __init__(self, stream):
        self.stream, self.rows, self.bytes = stream, 0, 0
        self.devices = {}

    def write(self, row):
        device = row_device(row)
        data = encoded(row) + b"\n"
        if self.rows >= MAX_ROWS or self.bytes + len(data) > MAX_OUTPUT:
            raise CaptureBudgetExceeded("Receipt output budget exceeded; no partial row written")
        self.stream.write(data)
        self._record(len(data), device)

    def _record(self, size, device):
        self.rows += 1
        self.bytes += size
        key = str(device) if device is not None else "unknown"
        self.devices[key] = self.devices.get(key, 0) + 1

    def summary(self):
        return {"rows": self.rows, "bytes": self.bytes, "rows_by_execution_device": self.devices}


class CompactRowWriter(RowWriter):
    """Lossless content-addressed JSON; all new objects and a row share one cap.

    References are separate from original values, so original keys cannot be
    mistaken for reference markers. Objects precede the rows that use them.
    """
    def __init__(self, stream):
        super().__init__(stream)
        self.known = set()

    def write(self, row):
        device = row_device(row)
        if self.rows >= MAX_ROWS:
            raise CaptureBudgetExceeded("Receipt row-count budget exceeded")
        staged, staged_bytes = {}, 0

        def intern(value):
            nonlocal staged_bytes
            original = encoded(value)
            if len(original) > MAX_OUTPUT:
                raise CaptureBudgetExceeded("Expanded receipt/object exceeds output budget")
            digest = hashlib.sha256(original).hexdigest()
            del original  # Do not retain a full ancestor encoding while descending.
            if digest in self.known or digest in staged:
                return digest
            reduced, references = value.copy(), []
            pairs = sorted(value.items()) if isinstance(value, dict) else enumerate(value)
            for key, child in pairs:
                if isinstance(child, (dict, list)) and len(encoded(child)) >= OBJECT_THRESHOLD:
                    references.append([key, intern(child)])
                    if isinstance(value, dict):
                        del reduced[key]
                    else:
                        reduced[key] = None
            if len(self.known) + len(staged) >= MAX_OBJECTS:
                raise CaptureBudgetExceeded("Shared object-count budget exceeded")
            data = encoded(dict(kind="object", sha256=digest, value=reduced,
                                references=references)) + b"\n"
            if self.bytes + staged_bytes + len(data) > MAX_OUTPUT:
                raise CaptureBudgetExceeded("Receipt output budget exceeded; no partial row written")
            staged[digest] = data
            staged_bytes += len(data)
            return digest

        digest = intern(row)
        marker = encoded(dict(kind="row", sha256=digest)) + b"\n"
        if self.bytes + staged_bytes + len(marker) > MAX_OUTPUT:
            raise CaptureBudgetExceeded("Receipt output budget exceeded; no partial row written")
        # Planning changes no persistent state; a rejected row leaves no dangling
        # reference, orphan object, or falsely remembered hash in this stream.
        for data in staged.values():
            self.stream.write(data)
        self.stream.write(marker)
        self.known.update(staged)
        self._record(staged_bytes + len(marker), device)

    def summary(self):
        return dict(super().summary(), objects=len(self.known), format="compact")


def iter_compact_rows(stream):
    """Verify and expand compact records; reject missing/corrupt references.

    Node values share immutable children internally. Each yielded row makes a
    bounded canonical JSON round trip, matching the writer's supported nesting
    depths and restoring independent values both within and between rows.
    """
    objects, sizes, total, rows = {}, {}, 0, 0
    while True:
        line = stream.readline(MAX_OUTPUT + 1)
        if not line:
            return
        total += len(line)
        if total > MAX_OUTPUT:
            raise ValueError("Compact input exceeds output-byte budget")
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError("Malformed compact record")
        digest = record["sha256"]
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Malformed compact object hash")
        if record.get("kind") == "row":
            if set(record) != {"kind", "sha256"} or digest not in objects:
                raise ValueError("Missing compact row object")
            rows += 1
            if rows > MAX_ROWS:
                raise ValueError("Compact row-count budget exceeded")
            row_device(objects[digest])
            yield json.loads(encoded(objects[digest]))
            continue
        if (record.get("kind") != "object" or set(record) != {"kind", "sha256", "value", "references"}
                or digest in objects or len(objects) >= MAX_OBJECTS
                or not isinstance(record["value"], (dict, list))
                or not isinstance(record["references"], list)):
            raise ValueError("Malformed or duplicate compact object")
        value = record["value"].copy()
        expanded_size = len(encoded(value))
        seen = set()
        for reference in record["references"]:
            if not isinstance(reference, list) or len(reference) != 2:
                raise ValueError("Malformed compact reference")
            key, child = reference
            if not isinstance(child, str) or child not in objects:
                raise ValueError("Missing compact child object")
            if isinstance(value, dict):
                if not isinstance(key, str) or key in value or key in seen:
                    raise ValueError("Invalid compact dictionary reference")
                expanded_size += len(encoded(key)) + 1 + sizes[child] + int(bool(value))
            else:
                if type(key) is not int or not 0 <= key < len(value) or value[key] is not None or key in seen:
                    raise ValueError("Invalid compact list reference")
                expanded_size += sizes[child] - 4  # Replace the null placeholder.
            if expanded_size > MAX_OUTPUT:
                raise ValueError("Expanded compact object exceeds output-byte budget")
            value[key] = objects[child]
            seen.add(key)
        original = encoded(value)
        if len(original) != expanded_size or hashlib.sha256(original).hexdigest() != digest:
            raise ValueError("Compact object checksum differs")
        objects[digest], sizes[digest] = value, expanded_size


def compact_receipt(value):
    return {key: value[key] for key in RECEIPT_KEYS if key in value}


def run_receipts(run, trail=""):
    if not isinstance(run, dict):
        return
    receipt = run.get("runtime_receipt")
    if isinstance(receipt, dict) and receipt:
        yield trail, run, receipt
    tiles = run.get("tile_evidence", ())
    pairs = tiles.items() if isinstance(tiles, dict) else enumerate(tiles) if isinstance(tiles, list) else ()
    for key, tile in pairs:
        yield from run_receipts(tile, trail + "/tile_evidence/" + str(key))


def inventory_identity(entries):
    digest = hashlib.sha256()
    for name, entry in sorted(entries.items()):
        digest.update(encoded([name, entry["bytes"], entry["sha256"], entry["transaction"]]) + b"\n")
    return digest.hexdigest()


def collect_archive(path, writer, cursor=None):
    report = {"path": str(path), "errors": [], "indexes_read": 0, "json_bytes_read": 0,
              "indexes_total": 0, "indexes_completed": 0, "indexes_failed": 0,
              "indexes_remaining": 0, "rows_skipped_for_cursor": 0,
              "coverage_complete": False, "next_cursor": None, "cursor_used": cursor}
    try:
        with path.open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Archive is not a regular file")
            entries, scan = scan_committed(stream)
            report.update(scan)
            identity = inventory_identity(entries)
            indexes = [(name, entry) for name, entry in sorted(entries.items()) if name.endswith("/index.json")]
            report.update(inventory_sha256=identity, indexes_total=len(indexes), indexes_remaining=len(indexes),
                          selected_index_bytes=sum(entry["bytes"] for _, entry in indexes))
            first_index, first_row = 0, 0
            if cursor is not None:
                if (not isinstance(cursor, dict) or cursor.get("schema") != CURSOR_SCHEMA
                        or cursor.get("inventory_sha256") != identity):
                    raise ValueError("Resume cursor inventory does not match this archive snapshot")
                first_index, first_row = cursor.get("index"), cursor.get("row")
                if (type(first_index) is not int or type(first_row) is not int
                        or not 0 <= first_index <= len(indexes) or first_row < 0
                        or (first_index == len(indexes) and first_row != 0)):
                    raise ValueError("Malformed resume cursor position")
            report["indexes_preceding_cursor"] = first_index
            report["indexes_remaining"] = len(indexes) - first_index

            def checkpoint(index_number, row_number, reason):
                report["stop_reason"] = reason
                report["next_cursor"] = dict(schema=CURSOR_SCHEMA, inventory_sha256=identity,
                                             index=index_number, row=row_number)
                report["indexes_remaining"] = len(indexes) - index_number

            for number in range(first_index, len(indexes)):
                logical, entry = indexes[number]
                skip_rows = first_row if number == first_index else 0
                row_number = 0
                try:
                    manifest_name = logical.rsplit("/", 1)[0] + "/manifest.json"
                    manifest_entry = entries.get(manifest_name)
                    if entry["bytes"] > MAX_INDEX or (manifest_entry and manifest_entry["bytes"] > MAX_MANIFEST):
                        raise ValueError("JSON member exceeds encoded-byte budget")
                    needed = entry["bytes"] + (manifest_entry["bytes"] if manifest_entry else 0)
                    if report["json_bytes_read"] + needed > MAX_JSON_TOTAL:
                        checkpoint(number, skip_rows, "json_read_budget")
                        break
                    report["json_bytes_read"] += entry["bytes"]
                    index = verified_json(stream, entry, MAX_INDEX)
                    if not isinstance(index, dict) or not isinstance(index.get("runs"), dict):
                        raise ValueError("Index has no SAM runs object")
                    scope = None
                    if manifest_entry is not None:
                        report["json_bytes_read"] += manifest_entry["bytes"]
                        manifest = verified_json(stream, manifest_entry, MAX_MANIFEST)
                        if not isinstance(manifest, dict):
                            raise ValueError("Evidence manifest JSON root is not an object")
                        scope = manifest.get("scope")
                    source = {"kind": "committed_archive", "member": logical,
                              "sha256": entry["sha256"], "transaction": entry["transaction"]}
                    for key, run in sorted(index["runs"].items()):
                        for trail, descriptor, receipt in run_receipts(run):
                            if row_number < skip_rows:
                                report["rows_skipped_for_cursor"] += 1
                                row_number += 1
                                continue
                            writer.write({"source": dict(source, run_key=key, run_path=trail),
                                "scope": scope,
                                "run": {k: descriptor[k] for k in RUN_KEYS if k in descriptor},
                                "runtime_receipt": compact_receipt(receipt)})
                            row_number += 1
                    if row_number < skip_rows:
                        raise ValueError("Resume cursor row exceeds index receipt inventory")
                    report["indexes_read"] += 1
                    report["indexes_completed"] += 1
                    report["indexes_remaining"] = len(indexes) - number - 1
                    del index
                except CaptureBudgetExceeded as error:
                    if writer.rows == 0:
                        # A fresh part cannot fit even its first appearance.
                        # Reissuing this cursor with identical limits would loop.
                        report["stop_reason"] = "single_receipt_exceeds_capture_budget"
                        report["errors"].append({"member": logical, "row": row_number,
                            "error": str(error), "resumable": False})
                        report["indexes_remaining"] = len(indexes) - number
                    else:
                        checkpoint(number, row_number, str(error))
                    break
                except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
                    report["errors"].append({"member": logical, "error": str(error)})
                    report["indexes_failed"] += 1
                    report["indexes_remaining"] = len(indexes) - number - 1
                    if len(report["errors"]) >= 100:
                        checkpoint(number + 1, 0, "error_count_budget")
                        break
            report["coverage_complete"] = (not report["errors"] and report["next_cursor"] is None
                and report["indexes_remaining"] == 0 and scan["tail_status"] == "end_of_snapshot")
    except (OSError, ValueError) as error:
        report["errors"].append({"error": f"{type(error).__name__}: {error}"})
    report.update(writer.summary())
    return report


def collect_live(root, writer, seconds, interval=.02):
    report = {"path": str(root), "seconds_requested": seconds, "errors": [],
              "polls": 0, "max_run_directories": 0, "json_bytes_read": 0}
    seen = set()
    started = time.monotonic()
    while True:
        try:
            directories = []
            for path in root.iterdir():
                if path.name.startswith("run-") and not path.is_symlink():
                    directories.append(path)
                    if len(directories) > MAX_MEMBERS:
                        raise ValueError("Live directory-count budget exceeded")
            report["max_run_directories"] = max(report["max_run_directories"], len(directories))
            for directory in directories:
                if str(directory) in seen:
                    continue
                path = directory / "manifest.json"
                try:
                    if path.is_symlink():
                        continue
                    with path.open("rb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST:
                            raise ValueError("Live manifest exceeds budget or is not regular")
                        data = stream.read(MAX_MANIFEST + 1)
                    if len(data) > MAX_MANIFEST:
                        raise ValueError("Live manifest grew beyond budget")
                    if report["json_bytes_read"] + len(data) > MAX_JSON_TOTAL:
                        raise ValueError("Total live JSON read budget exceeded")
                    report["json_bytes_read"] += len(data)
                    receipt = json.loads(data)
                    if not isinstance(receipt, dict):
                        raise ValueError("Live manifest JSON root is not an object")
                    if receipt.get("schema") != "xta.sam-raw-tracker-run/1":
                        continue
                    writer.write({"source": {"kind": "live_manifest", "path": str(path),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "captured_wall_time": time.time(), "captured_monotonic": time.monotonic()},
                        "runtime_receipt": compact_receipt(receipt)})
                    seen.add(str(directory))
                    if len(seen) >= MAX_ROWS:
                        raise ValueError("Live manifest-count budget reached")
                except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
                    continue  # Atomic publication/cleanup can race this observer.
            report["polls"] += 1
        except (OSError, ValueError, TypeError) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            break
        if time.monotonic() - started >= seconds:
            break
        time.sleep(min(interval, max(0., seconds - (time.monotonic() - started))))
    report["elapsed_seconds"] = time.monotonic() - started
    report.update(writer.summary())
    return report


def previous_cursor(path):
    with zipfile.ZipFile(path) as saved:
        info = saved.getinfo("summary.json")
        if info.file_size > MAX_MANIFEST:
            raise ValueError("Previous capture summary exceeds metadata budget")
        summary = json.loads(saved.read(info))
    if not isinstance(summary, dict) or not isinstance(summary.get("archive"), dict):
        raise ValueError("Previous capture has no archive coverage summary")
    cursor = summary["archive"].get("next_cursor")
    if not isinstance(cursor, dict):
        raise ValueError("Previous capture has no resumable cursor")
    return cursor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Run's sam-artifacts.tar")
    parser.add_argument("--live-root", type=Path, help="Node-local temp/sam_runtime")
    parser.add_argument("--watch-seconds", type=float, default=60.)
    parser.add_argument("--output", type=Path, help="New ZIP path (default: unique file under HOME)")
    parser.add_argument("--format", choices=("inline", "compact"), default="inline",
                        help="compact stores repeated metadata once by SHA256; inline preserves the original format")
    parser.add_argument("--resume-from", type=Path,
                        help="Previous capture ZIP whose archive next_cursor continues the same inventory")
    args = parser.parse_args(argv)
    if not args.archive and not args.live_root:
        parser.error("Provide --archive and/or --live-root")
    if not 0 <= args.watch_seconds <= 120:
        parser.error("--watch-seconds must be between 0 and 120")
    if args.resume_from and not args.archive:
        parser.error("--resume-from requires --archive")
    cursor = previous_cursor(args.resume_from) if args.resume_from else None
    writer_type = CompactRowWriter if args.format == "compact" else RowWriter
    suffix = ".compact.jsonl" if args.format == "compact" else ".jsonl"
    output = args.output or Path.home() / ("sam-runtime-receipts-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".zip")
    summary = {"schema": COMPACT_SCHEMA if args.format == "compact" else SCHEMA,
               "receipt_format": args.format, "started_wall_time": time.time(),
               "started_monotonic": time.monotonic(),
               "notes": ["Only authenticated committed archive metadata is used; masks are skipped.",
                         "Live polling may miss short-lived manifests; rows are not a complete run census.",
                         "Archive and live rows can overlap; match run_id, PID and worker_started_monotonic.",
                         "Host timing fields overlap and are not GPU kernel times.",
                         "coverage_complete applies to this part's suffix; retain earlier parts when resuming.",
                         "A nonnull next_cursor requires another part using --resume-from; errors remain explicit omissions.",
                         "Each index is bounded to 128 MiB encoded; decoded JSON uses additional memory."],
               "limits": {"index_bytes": MAX_INDEX, "manifest_bytes": MAX_MANIFEST,
                          "commit_bytes": MAX_COMMIT, "archive_members": MAX_MEMBERS,
                          "pax_total_bytes": MAX_PAX_TOTAL, "pax_chain": MAX_PAX_CHAIN,
                          "json_read_bytes_per_source": MAX_JSON_TOTAL,
                          "shared_objects_per_source": MAX_OBJECTS,
                          "output_bytes_per_source": MAX_OUTPUT, "rows_per_source": MAX_ROWS}}
    # Exclusive output creation prevents overwriting any input or earlier capture.
    with output.open("x+b") as destination, tempfile.TemporaryDirectory(prefix="sam-receipts-", dir=output.parent) as temporary:
        directory = Path(temporary)
        workers = []
        def capture_live():
            try:
                with (directory / ("live_receipts" + suffix)).open("wb") as stream:
                    summary["live"] = collect_live(args.live_root, writer_type(stream), args.watch_seconds)
            except Exception as error:
                summary["live"] = {"errors": [f"{type(error).__name__}: {error}"]}
        if args.live_root:
            worker = threading.Thread(target=capture_live, name="receipt-manifest-observer")
            worker.start()
            workers.append(worker)
        if args.archive:
            with (directory / ("archive_receipts" + suffix)).open("wb") as stream:
                summary["archive"] = collect_archive(args.archive, writer_type(stream), cursor=cursor)
        for worker in workers:
            worker.join()
        summary["finished_wall_time"] = time.time()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as result:
            for path in sorted(directory.iterdir()):
                result.write(path, path.name)
            result.writestr("summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output.absolute()), "archive": summary.get("archive"),
                      "live": summary.get("live")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
