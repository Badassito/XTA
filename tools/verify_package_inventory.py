"""Verify the current maintained source set, version and canonical byte digests.

The enclosing Git commit/tag or complete-source ZIP authenticates this record.
It is an identity guard, not a numerical correctness or qualification claim.
Each release replaces one flat record; historical certificates live in History.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = 'release/_package_inventory.json'
MANIFEST = ROOT / INVENTORY_PATH
SCHEMA = 'xta-current-source-inventory-v1'
DIGEST_REPRESENTATION = 'utf8-text-crlf-to-lf;binary-exact-v1'
_CACHE_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
_ARCHIVE_METADATA = {'RELEASE_MANIFEST.json', 'READ_ME_FIRST.txt'}
_ARCHIVE_METADATA_CASEFOLD = {name.casefold() for name in _ARCHIVE_METADATA}
_VERSION = re.compile(r'(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)')
_SHA256 = re.compile(r'[0-9a-f]{64}')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_source_bytes(data: bytes) -> bytes:
    """Normalize CRLF in NUL-free UTF-8 text; retain lone CR and binary bytes."""
    if b'\0' in data:
        return data
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return data
    return data.replace(b'\r\n', b'\n')


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(('git', *args), cwd=root, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))
    if result.returncode:
        raise RuntimeError('Cannot enumerate maintained source: ' + result.stderr.decode('utf-8', errors='replace').strip())
    return result.stdout


def _generated_source_name(name: str) -> bool:
    parts = tuple(part.casefold() for part in PurePosixPath(name).parts)
    return (any(part in _CACHE_DIRECTORIES or part.endswith(('.pyc', '.pyo', '.nbc', '.nbi')) for part in parts)
            or parts[0] in {'build', 'dist'} or parts[0].endswith('.egg-info'))


def _source_names(root: Path) -> tuple[list[str], str | None, str | None]:
    if (root / '.git').exists():
        git_root = Path(os.fsdecode(_git(root, 'rev-parse', '--show-toplevel')).strip()).resolve()
        if git_root != root:
            raise RuntimeError('Source root must be the Git repository root')
        names = os.fsdecode(_git(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z')).split('\0')
        return sorted(set(names) - {''}), _git(root, 'rev-parse', 'HEAD').decode('ascii').strip(), os.fsdecode(_git(root, 'status', '--porcelain=v1', '-z'))
    names = []
    for directory, children, files in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        children[:] = [name for name in children if not _generated_source_name((relative / name).as_posix())]
        for name in children:
            _safe_path(root, (relative / name).as_posix())
        for name in files:
            member = (relative / name).as_posix()
            if member in _ARCHIVE_METADATA or _generated_source_name(member):
                continue
            names.append(member)
    return sorted(names), None, None


def _validate_name(name: str) -> None:
    path = PurePosixPath(name)
    if (not name or any(ord(character) < 32 for character in name) or '\\' in name or ':' in name or path.is_absolute()
            or any(part in {'.', '..'} for part in name.split('/')) or path.as_posix() != name):
        raise RuntimeError(f'Invalid source member path: {name!r}')


def validate_source_member(name: str) -> None:
    """Shared Git/bundle policy: source cannot use generated or metadata paths."""
    _validate_name(name)
    if name.casefold() in _ARCHIVE_METADATA_CASEFOLD:
        raise RuntimeError(f'Reserved source-bundle metadata cannot be maintained source: {name}')
    if _generated_source_name(name):
        raise RuntimeError(f'Generated/cache path cannot be maintained source: {name}')


def _safe_path(root: Path, name: str) -> Path:
    _validate_name(name)
    path = root
    for part in PurePosixPath(name).parts:
        path /= part
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
            raise RuntimeError(f'Source member uses a symbolic link or reparse point: {name}')
    if not path.resolve().is_relative_to(root):
        raise RuntimeError(f'Source member escapes the source root: {name}')
    return path


def _stat_key(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _read_source(root: Path, name: str) -> bytes | None:
    path = _safe_path(root, name)
    try:
        before = path.stat()
    except FileNotFoundError:
        return None  # A pending tracked deletion stays visible in raw identity.
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f'Source member is not a regular file: {name}')
    data = path.read_bytes()
    try:
        after = _safe_path(root, name).stat()
    except OSError as error:
        raise RuntimeError(f'Current source changed while reading {name}') from error
    if _stat_key(before) != _stat_key(after):
        raise RuntimeError(f'Current source changed while reading {name}')
    return data


def _capture(root: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    root = root.resolve()
    names, commit, status = _source_names(root)
    if len({name.casefold() for name in names}) != len(names):
        raise RuntimeError('Source members have ambiguous case-insensitive paths')
    payloads, files = {}, {}
    for name in names:
        raw = _read_source(root, name)
        files[name] = None if raw is None else digest(raw)
        if raw is not None:
            validate_source_member(name)
            payloads[name] = raw
    return payloads, {'commit': commit, 'status': status, 'files': files}


def capture_source_identity(root: Path = ROOT) -> dict[str, object]:
    """Capture raw bytes, manifest included, for before/after qualification guards."""
    return _capture(root)[1]


def _assert_unchanged(root: Path, identity: dict[str, object]) -> None:
    if capture_source_identity(root) != identity:
        raise RuntimeError('Current source changed during inventory verification; retry from a stable tree')


def read_source_files(root: Path = ROOT) -> dict[str, bytes]:
    """Read stable RAW maintained bytes, manifest included, for source bundles."""
    payloads, identity = _capture(root)
    _assert_unchanged(root, identity)
    return payloads


def collect_source_files(root: Path = ROOT, *, exclude_inventory: bool = True) -> dict[str, bytes]:
    """Collect every tracked/nonignored source, or a portable extracted source tree."""
    payloads = read_source_files(root)
    return {name: canonical_source_bytes(raw) for name, raw in payloads.items()
            if not exclude_inventory or name != INVENTORY_PATH}


def source_version(payloads: dict[str, bytes]) -> str:
    try:
        tree = ast.parse(payloads['XTA/__init__.py'].decode('utf-8-sig'))
        values = [ast.literal_eval(node.value) for node in tree.body
                  if ((isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == '__version__' for target in node.targets))
                      or (isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == '__version__'))]
    except (KeyError, SyntaxError, UnicodeError, ValueError, TypeError) as error:
        raise RuntimeError(f'Cannot read literal XTA.__version__: {error}') from error
    if len(values) != 1 or not isinstance(values[0], str) or _VERSION.fullmatch(values[0]) is None:
        raise RuntimeError('Expected exactly one semantic-version XTA.__version__ assignment')
    return values[0]


def _inventory_for(payloads: dict[str, bytes]) -> dict[str, object]:
    return {'schema': SCHEMA, 'digest_representation': DIGEST_REPRESENTATION,
            'version': source_version(payloads),
            'files': {name: digest(canonical_source_bytes(raw)) for name, raw in sorted(payloads.items())
                      if name != INVENTORY_PATH}}


def build_inventory(root: Path = ROOT) -> tuple[dict[str, object], dict[str, object]]:
    payloads, identity = _capture(root)
    inventory = _inventory_for(payloads)
    _assert_unchanged(root, identity)
    return inventory, identity


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f'Duplicate inventory JSON key: {key}')
        result[key] = value
    return result


def load_inventory(data: bytes) -> dict[str, object]:
    try:
        inventory = json.loads(data.decode('utf-8'), object_pairs_hook=_unique_pairs)
    except (UnicodeError, ValueError) as error:
        raise RuntimeError(f'Cannot read source inventory: {error}') from error
    if not isinstance(inventory, dict) or set(inventory) != {'schema', 'digest_representation', 'version', 'files'}:
        raise RuntimeError('Unexpected current source inventory fields')
    if inventory['schema'] != SCHEMA or inventory['digest_representation'] != DIGEST_REPRESENTATION:
        raise RuntimeError('Unsupported current source inventory representation')
    if not isinstance(inventory['version'], str) or _VERSION.fullmatch(inventory['version']) is None:
        raise RuntimeError('Invalid inventory version')
    files = inventory['files']
    if not isinstance(files, dict) or not files or INVENTORY_PATH in files:
        raise RuntimeError('Inventory must contain source files and exclude its own self digest')
    for name, value in files.items():
        validate_source_member(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeError(f'Invalid inventory source digest: {name}')
    if len({name.casefold() for name in files}) != len(files):
        raise RuntimeError('Inventory members have ambiguous case-insensitive paths')
    return inventory


def verify(root: Path = ROOT) -> dict[str, object]:
    payloads, identity = _capture(root)
    if INVENTORY_PATH not in payloads:
        raise RuntimeError('Current source inventory is missing')
    inventory = load_inventory(payloads[INVENTORY_PATH])
    actual = _inventory_for(payloads)
    if inventory['version'] != actual['version']:
        raise RuntimeError(f"Inventory version differs: recorded={inventory['version']}, source={actual['version']}")
    expected_files, actual_files = inventory['files'], actual['files']
    missing, extra = sorted(expected_files.keys() - actual_files.keys()), sorted(actual_files.keys() - expected_files.keys())
    if missing or extra:
        raise RuntimeError(f'Source file set differs: missing={missing}, extra={extra}')
    changed = [name for name in actual_files if expected_files[name] != actual_files[name]]
    if changed:
        raise RuntimeError(f'Source digest differs: {changed}')
    _assert_unchanged(root, identity)
    return {'version': inventory['version'], 'files': len(actual_files),
            'inventory_sha256': digest(payloads[INVENTORY_PATH]), 'source_commit': identity['commit']}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    result = verify(parser.parse_args(argv).root)
    print(f"package inventory verified: v{result['version']}, {result['files']} maintained source files; inventory SHA256 {result['inventory_sha256']}")


if __name__ == '__main__':
    main()
