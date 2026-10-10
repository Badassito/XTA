"""Lossless, checked transport envelopes for XTA run folders.

This module uses only the standard library. It changes transport layout, never
measurement or scientific evidence schemas. Loose producer files remain intact.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import zipfile

SCHEMA = "xta.run-transport/1"
INDEX = "META/transport.json"
CHUNK = 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_RECEIPTS = {"selection.json", "initial_selection.json", "generation.json", "crop_retry.json",
             "failure.json", "context_preparation_failure.json", "preflight.json"}


class TransportError(ValueError):
    pass


def _canonical_path(path) -> Path:
    value = os.fspath(path)
    if os.name == "nt":
        # Extended paths bypass Win32's dot-segment normalization. Strip that
        # prefix first, including when a staging helper already supplied it.
        if value.upper().startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(os.path.abspath(os.path.normpath(value)))


def _api_path(path: Path) -> str:
    """Normalize dot segments before applying the extended Windows API prefix."""
    value = str(_canonical_path(path))
    if os.name == "nt":
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def _lstat(path):
    return os.lstat(_api_path(Path(path)))


def _linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & _REPARSE)


def _plain_ancestors(path):
    for current in (path, *path.parents):
        try:
            if _linked(_lstat(current)):
                raise TransportError(f"Link/reparse path refused: {current}")
        except FileNotFoundError:
            continue


def _safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise TransportError(f"Unsafe archive path: {value!r}")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part[-1:] in {".", " "}
                or any(ord(char) < 32 or char in ':<>"|?*' for char in part)
                or re.fullmatch(r"(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise TransportError(f"Unsafe/nonportable archive path: {value!r}")
    return PurePosixPath(value)


def _json_file(path):
    try:
        info = _lstat(path)
        if _linked(info) or info.st_size > MAX_INDEX_BYTES:
            return {}
        with open(_api_path(path), "rb") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _run_metadata(root, requested_status):
    metadata = _json_file(root / "manifest.json")
    lta = _json_file(root / "lta_execution_identity.json")
    command = metadata.get("launcher", {}).get("command", ()) if isinstance(metadata.get("launcher"), dict) else ()
    owned = bool(lta.get("schema") == "lta.execution-identity/1" or
                 (command and any("GPT-6-Astra-Ultra" in str(token) or str(token) == "xta" for token in command[:1])))
    if requested_status != "unknown":
        return {"status": requested_status, "origin": "caller", "partial": requested_status != "complete"}, owned
    status = metadata.get("status")
    if status not in {"complete", "completed", "failed", "in_progress"}:
        status = "complete" if metadata.get("complete") is True else "unknown"
    status = "complete" if status == "completed" else status
    return {"status": status, "origin": "source_manifest" if status != "unknown" else "unavailable",
            "partial": None if status == "unknown" else status != "complete"}, owned


def _temporary_reason(relative, info, owned_run):
    parts = relative.parts
    # RuntimeTelemetry never traverses scratch. The output launcher explicitly
    # owns its root temp entry (including a symlink to PID-owned --temp storage).
    if owned_run and parts == ("temp",):
        return "verified XTA launcher-owned runtime temp entry; target not followed"
    name = parts[-1]
    if (stat.S_ISDIR(info.st_mode) and parts[0] in {"sam_interpolation", "sam_extrapolation"}
            and re.fullmatch(r"\.(?:evidence|final_evidence|initial_evidence)\.(?:stage|export)-[0-9a-f]{32}", name)):
        return "private SAM evidence staging/export directory (producer UUID naming contract)"
    if stat.S_ISDIR(info.st_mode) and re.fullmatch(r"\.nrrd\.atomic-result-[0-9A-Za-z_]{8}", name):
        return "private atomic-result publication staging directory (producer naming contract)"
    return None


def _category(relative):
    parts = relative.parts
    if parts == ("sam-artifacts.tar",):
        return "scientific_evidence"
    if parts[0] in {"telemetry", "lta_diagnostics"} and relative.suffix == ".jsonl":
        return "telemetry"
    if parts[0] in {"sam_interpolation", "sam_extrapolation", "reconciliation_evidence"}:
        return "scientific_receipt" if relative.name in _RECEIPTS else "scientific_evidence"
    if parts[0] in {"nrrd", "low_quality", "images", "labels", "overlays"}:
        return "output"
    return "diagnostic" if relative.suffix in {".json", ".jsonl", ".log", ".txt"} else "other"


def _public_nrrd(relative):
    """Scientific evidence stores remain inside the diagnostics envelope."""
    return (relative.suffix.lower() == ".nrrd" and relative.parts[0] not in {
        "sam_interpolation", "sam_extrapolation", "reconciliation_evidence", "sam_evidence", "evidence",
    })


def _stamp(info):
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    # Windows lstat/fstat can expose different deprecated ctime values even
    # for the same unchanged file. File ID, size and write time are consistent.
    return stamp if os.name == "nt" else (*stamp, info.st_ctime_ns)


def _scan(root, scope, owned_run):
    files, omissions, external = [], [], []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(_api_path(directory)) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = directory / entry.name
                relative = PurePosixPath(path.relative_to(root).as_posix())
                # Windows DirEntry.stat can omit file IDs; use the same lstat
                # API as the later stability checks so identities agree.
                info = _lstat(path)
                reason = _temporary_reason(relative, info, owned_run)
                if reason:
                    omissions.append({"path": str(relative), "reason": reason, "kind": "owned_temporary"})
                    continue
                if _linked(info):
                    raise TransportError(f"Link/reparse entry refused (not followed): {path}")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    _safe_relative(str(relative))
                    category = _category(relative)
                    if scope == "diagnostics" and _public_nrrd(relative):
                        omissions.append({"path": str(relative), "reason": "public NRRD payload transferred separately",
                                          "kind": "external_nrrd", "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
                        external.append((path, str(relative), "output", info))
                    else:
                        files.append((path, str(relative), category, info))
                else:
                    raise TransportError(f"Nonregular run entry refused: {path}")
    return (sorted(files, key=lambda row: row[1]), sorted(omissions, key=lambda row: row["path"]),
            sorted(external, key=lambda row: row[1]))


def _fresh_target(source, destination):
    source = _canonical_path(source)
    destination = _canonical_path(destination)
    _plain_ancestors(source)
    _plain_ancestors(destination)
    if destination == source or source in destination.parents:
        raise TransportError("Destination must be outside the input tree")
    try:
        _lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise TransportError(f"Refusing to overwrite destination: {destination}")
    os.makedirs(_api_path(destination.parent), exist_ok=True)
    return source, destination


def _publish_fresh(stage, destination, *, directory=False):
    # Windows rename refuses an existing destination. POSIX hard-link publication
    # is also exclusive for files; directories are checked immediately before rename.
    if os.name == "nt":
        if os.path.lexists(_api_path(destination)):
            raise TransportError(f"Destination appeared during operation: {destination}")
        os.rename(_api_path(stage), _api_path(destination))
    elif directory:
        # An ordinary POSIX rename can replace an existing empty directory.
        # Linux's exclusive rename closes that publication race.
        import ctypes
        import errno
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise TransportError("Exclusive directory publication is unavailable on this host")
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise TransportError(f"Destination appeared during operation: {destination}")
            raise OSError(code, os.strerror(code), str(destination))
    else:
        os.link(stage, destination)
        os.unlink(stage)


def _copy_hash(source, destination):
    digest, count = hashlib.sha256(), 0
    while True:
        data = source.read(CHUNK)
        if not data:
            return digest.hexdigest(), count
        destination.write(data)
        digest.update(data)
        count += len(data)


def pack_run(source, destination, *, scope="diagnostics", run_status="unknown"):
    """Pack a stable run snapshot atomically; changes abort without publishing."""
    if scope not in {"all", "diagnostics"} or run_status not in {"unknown", "in_progress", "failed", "complete"}:
        raise TransportError("Unsupported transport scope or run status")
    root, target = _fresh_target(source, destination)
    if not stat.S_ISDIR(_lstat(root).st_mode):
        raise TransportError("Pack input must be a directory")
    status, owned = _run_metadata(root, run_status)
    files, omissions, external = _scan(root, scope, owned)
    handle, temporary = tempfile.mkstemp(prefix=".xta-pack-", suffix=".zip", dir=_api_path(target.parent))
    os.close(handle)
    stage = Path(temporary)
    manifest = {"schema": SCHEMA, "scope": scope, "created_unix_ns": time.time_ns(),
                "source_name": root.name, "source_run": status,
                "source_inventory_completeness": "unknown; available local files only",
                "source_changes": [], "omitted": omissions, "external_outputs": [], "files": []}
    try:
        with zipfile.ZipFile(_api_path(stage), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, relative, category, captured in files:
                current = _lstat(path)
                if _linked(current) or _stamp(current) != _stamp(captured):
                    raise TransportError(f"Source changed before packing: {relative}")
                with open(_api_path(path), "rb") as source_file:
                    if _stamp(os.fstat(source_file.fileno())) != _stamp(captured):
                        raise TransportError(f"Source changed while opening: {relative}")
                    with archive.open("DATA/" + relative, "w", force_zip64=True) as output:
                        digest, size = _copy_hash(source_file, output)
                    if _stamp(os.fstat(source_file.fileno())) != _stamp(captured):
                        raise TransportError(f"Source changed while packing: {relative}")
                if _stamp(_lstat(path)) != _stamp(captured) or size != captured.st_size:
                    raise TransportError(f"Source changed after packing: {relative}")
                manifest["files"].append({"path": relative, "category": category, "bytes": size,
                                           "sha256": digest, "mtime_ns": captured.st_mtime_ns})
            for path, relative, category, captured in external:
                current = _lstat(path)
                if _linked(current) or _stamp(current) != _stamp(captured):
                    raise TransportError(f"External NRRD changed before hashing: {relative}")
                with open(_api_path(path), "rb") as source_file:
                    if _stamp(os.fstat(source_file.fileno())) != _stamp(captured):
                        raise TransportError(f"External NRRD changed while opening: {relative}")
                    digest, size = _copy_hash(source_file, _Discard())
                    if _stamp(os.fstat(source_file.fileno())) != _stamp(captured):
                        raise TransportError(f"External NRRD changed while hashing: {relative}")
                if _stamp(_lstat(path)) != _stamp(captured) or size != captured.st_size:
                    raise TransportError(f"External NRRD changed after hashing: {relative}")
                manifest["external_outputs"].append({"path": relative, "category": category, "bytes": size,
                                                      "sha256": digest, "mtime_ns": captured.st_mtime_ns,
                                                      "transport": "separate public NRRD file"})
            # Detect new/deleted files and completed publication directories too.
            after, after_omissions, after_external = _scan(root, scope, owned)
            if ([(row[1], _stamp(row[3])) for row in after] != [(row[1], _stamp(row[3])) for row in files]
                    or [(row[1], _stamp(row[3])) for row in after_external] != [(row[1], _stamp(row[3])) for row in external]
                    or after_omissions != omissions):
                raise TransportError("Source inventory changed while packing")
            archive.writestr(INDEX, json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False))
        verify_bundle(stage)
        _publish_fresh(stage, target)
        return manifest
    finally:
        if os.path.lexists(_api_path(stage)):
            os.unlink(_api_path(stage))


def _bundle_index(archive):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TransportError("Duplicate ZIP members")
    for info in infos:
        _safe_relative(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in {0, stat.S_IFREG} or info.is_dir() or info.flag_bits & 1 or (info.external_attr & _REPARSE):
            raise TransportError(f"Link, directory, encrypted or special ZIP member refused: {info.filename}")
    try:
        index_info = archive.getinfo(INDEX)
        if index_info.file_size > MAX_INDEX_BYTES:
            raise TransportError("Transport inventory exceeds bounded metadata budget")
        manifest = json.loads(archive.read(INDEX))
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise TransportError("Missing or invalid transport inventory") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("scope") not in {"all", "diagnostics"}
            or not isinstance(manifest.get("files"), list)):
        raise TransportError("Unsupported transport inventory")
    paths, prefixes = set(), {}
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise TransportError("Invalid inventory entry")
        relative = _safe_relative(row.get("path"))
        category = _category(relative)
        if row.get("category") != category:
            raise TransportError(f"Inventory file category mismatch: {relative}")
        if manifest["scope"] == "diagnostics" and _public_nrrd(relative):
            raise TransportError("Public NRRD payload is outside declared diagnostics scope")
        folded = str(relative).casefold()
        if folded in paths:
            raise TransportError("Duplicate/case-colliding inventory paths")
        paths.add(folded)
        if (type(row.get("bytes")) is not int or row["bytes"] < 0
                or not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise TransportError("Invalid inventory size/hash")
        for i in range(1, len(relative.parts) + 1):
            prefix = "/".join(relative.parts[:i])
            key = prefix.casefold()
            previous = prefixes.setdefault(key, (prefix, i == len(relative.parts)))
            if previous != (prefix, i == len(relative.parts)):
                raise TransportError("Case-colliding or file/directory-conflicting inventory")
        member = "DATA/" + str(relative)
        try:
            if archive.getinfo(member).file_size != row["bytes"]:
                raise TransportError(f"Inventory/member size mismatch: {member}")
        except KeyError as exc:
            raise TransportError(f"Missing inventoried member: {member}") from exc
    expected = {INDEX, *("DATA/" + row["path"] for row in manifest["files"])}
    if set(names) != expected:
        raise TransportError("ZIP members differ from declared inventory")
    externals = manifest.get("external_outputs", [])
    if not isinstance(externals, list):
        raise TransportError("Invalid external NRRD inventory")
    for row in externals:
        if not isinstance(row, dict):
            raise TransportError("Invalid external NRRD entry")
        relative = _safe_relative(row.get("path"))
        folded = str(relative).casefold()
        if (folded in paths or not _public_nrrd(relative) or row.get("category") != "output"
                or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise TransportError("Invalid/duplicate external NRRD identity")
        paths.add(folded)
    return manifest


def _verify_member(archive, row, output=None):
    with archive.open("DATA/" + row["path"]) as source:
        digest, size = _copy_hash(source, output or _Discard())
    if digest != row["sha256"] or size != row["bytes"]:
        raise TransportError(f"Member checksum/size mismatch: {row['path']}")


class _Discard:
    def write(self, data):
        return len(data)


def verify_bundle(path):
    """Validate structure, every CRC, declared byte count and SHA-256."""
    _plain_ancestors(_canonical_path(path))
    with zipfile.ZipFile(_api_path(Path(path))) as archive:
        manifest = _bundle_index(archive)
        for row in manifest["files"]:
            _verify_member(archive, row)
        return manifest


def verify_external_outputs(manifest, outputs_root):
    """Check separate NRRD transfer availability without changing run status."""
    root = _canonical_path(outputs_root)
    _plain_ancestors(root)
    rows = []
    for entry in manifest.get("external_outputs", []):
        relative = _safe_relative(entry["path"])
        path = root.joinpath(*relative.parts)
        row = {"path": str(relative)}
        try:
            _plain_ancestors(path)
            before = _lstat(path)
            if _linked(before) or not stat.S_ISREG(before.st_mode) or before.st_size != entry["bytes"]:
                row.update(status="corrupt", reason="type/link/size mismatch")
            else:
                with open(_api_path(path), "rb") as handle:
                    digest, count = _copy_hash(handle, _Discard())
                    stable = _stamp(os.fstat(handle.fileno())) == _stamp(before)
                stable = stable and _stamp(_lstat(path)) == _stamp(before)
                if not stable or count != entry["bytes"] or digest != entry["sha256"]:
                    row.update(status="corrupt", reason="changed file or SHA-256 mismatch")
                else:
                    row.update(status="verified")
        except FileNotFoundError:
            row.update(status="missing", reason="not available below the chosen outputs root")
        except (OSError, TransportError) as error:
            row.update(status="corrupt", reason=str(error))
        rows.append(row)
    missing = sum(row["status"] == "missing" for row in rows)
    corrupt = sum(row["status"] == "corrupt" for row in rows)
    return {"status": "incomplete" if missing or corrupt else "verified", "outputs_root": str(root),
            "verified": len(rows) - missing - corrupt, "missing": missing, "corrupt": corrupt,
            "source_run": manifest["source_run"], "files": rows,
            "meaning": "Separate transfer availability/integrity; missing files are not attributed to producer failure."}


def unpack_run(source, destination):
    """Extract into a private sibling, verify fully, then publish a fresh root."""
    source, target = _fresh_target(source, destination)
    # Extracting into an existing directory, even an empty one, is refused.
    stage = Path(tempfile.mkdtemp(prefix=".xta-unpack-", dir=_api_path(target.parent)))
    try:
        with zipfile.ZipFile(_api_path(source)) as archive:
            manifest = _bundle_index(archive)
            for row in manifest["files"]:
                output = stage.joinpath(*PurePosixPath(row["path"]).parts)
                os.makedirs(_api_path(output.parent), exist_ok=True)
                with open(_api_path(output), "xb") as handle:
                    _verify_member(archive, row, handle)
                if type(row.get("mtime_ns")) is int:
                    os.utime(_api_path(output), ns=(row["mtime_ns"], row["mtime_ns"]))
        _publish_fresh(stage, target, directory=True)
        return manifest
    finally:
        if os.path.lexists(_api_path(stage)):
            # stage was freshly created below the explicit destination parent;
            # all members passed path/link validation before any publication.
            shutil.rmtree(_api_path(stage))


class _VerifiedRaw(io.RawIOBase):
    def __init__(self, stream, row):
        self.stream, self.row = stream, row
        self.digest, self.count, self.checked = hashlib.sha256(), 0, False

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        if data:
            buffer[:len(data)] = data
            self.digest.update(data)
            self.count += len(data)
            return len(data)
        if not self.checked:
            self.checked = True
            if self.count != self.row["bytes"] or self.digest.hexdigest() != self.row["sha256"]:
                raise TransportError(f"Telemetry member checksum mismatch: {self.row['path']}")
        return 0


def telemetry_streams(paths, *, lta=False):
    """Yield (display path, text handle) for loose JSONL or verified ZIP streams."""
    selected = sorted({Path(path) for path in paths}, key=str)
    for path in selected:
        if path.is_dir():
            pattern = "*.jsonl" if lta else "telemetry-*.jsonl"
            yield from telemetry_streams(sorted(path.glob(pattern)), lta=lta)
        elif path.suffix.lower() == ".zip":
            _plain_ancestors(_canonical_path(path))
            with zipfile.ZipFile(_api_path(path)) as archive:
                manifest = _bundle_index(archive)
                for row in manifest["files"]:
                    if row.get("category") != "telemetry":
                        continue
                    if lta != (PurePosixPath(row["path"]).parts[0] == "lta_diagnostics"):
                        continue
                    with archive.open("DATA/" + row["path"]) as raw:
                        checked = _VerifiedRaw(raw, row)
                        with io.TextIOWrapper(io.BufferedReader(checked), encoding="utf-8") as text:
                            yield f"{path}!{row['path']}", text
                            # Enforce verification even if a consumer stops early.
                            while text.read(CHUNK):
                                pass
        else:
            with open(_api_path(path), encoding="utf-8") as handle:
                yield str(path), handle
