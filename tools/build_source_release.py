"""Build a source bundle from a clean Git commit and recorded release context.

Use --snapshot only to inspect an uncommitted development tree. Snapshot bundles
are labelled as such and must not be treated as release artifacts.
Reproducibility assumes the same inputs, options and local release-tag context;
the manifest preserves the version check and any acknowledged-gap reason.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

if __package__:
    from .check_release_version import check_release_version
    from .verify_package_inventory import read_source_files, validate_source_member
else:
    from check_release_version import check_release_version
    from verify_package_inventory import read_source_files, validate_source_member


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(('git', *args), cwd=root)


def source_payloads(root: Path, *, snapshot: bool = False) -> tuple[dict[str, bytes], str]:
    """Read tracked commit bytes, or explicitly requested development bytes."""
    root = root.resolve()
    git_root = Path(os.fsdecode(_git(root, 'rev-parse', '--show-toplevel')).strip()).resolve()
    if git_root != root:
        raise ValueError(f'{root} is not the Git repository root')
    commit = _git(root, 'rev-parse', 'HEAD').decode('ascii').strip()
    if snapshot:
        try:
            return read_source_files(root), 'working-tree-snapshot'
        except RuntimeError as error:
            raise ValueError(str(error)) from error
    if _git(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all'):
        raise ValueError('Release bundles require a clean Git tree; use --snapshot for development validation')
    entries = []
    for raw in _git(root, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
        if not raw:
            continue
        metadata, raw_name = raw.split(b'\t', 1)
        mode, kind, object_id = metadata.split(b' ')
        name = raw_name.decode('utf-8', errors='surrogateescape')
        try:
            validate_source_member(name)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        if kind != b'blob' or mode == b'120000':
            raise ValueError(f'Source member is not a regular Git blob: {name}')
        entries.append((name, object_id))
    if len({name.casefold() for name, _ in entries}) != len(entries):
        raise ValueError('Source members have ambiguous case-insensitive paths')
    requested = b''.join(object_id + b'\n' for _name, object_id in entries)
    batch = subprocess.check_output(('git', 'cat-file', '--batch'), input=requested, cwd=root)
    payloads = {}
    offset = 0
    for name, object_id in entries:
        line_end = batch.find(b'\n', offset)
        if line_end < 0:
            raise ValueError(f'Git blob response ended before {name}')
        returned_id, kind, raw_size = batch[offset:line_end].split(b' ')
        if returned_id != object_id or kind != b'blob':
            raise ValueError(f'Git blob identity differs for {name}')
        size = int(raw_size)
        start, end = line_end + 1, line_end + 1 + size
        if batch[end:end + 1] != b'\n':
            raise ValueError(f'Git blob response is truncated for {name}')
        payloads[name] = batch[start:end]
        offset = end + 1
    if offset != len(batch):
        raise ValueError('Unexpected bytes after Git blob batch')
    return payloads, commit


def build(*, root: Path, output: Path, readme: Path | None = None,
          wheel: Path | None = None, snapshot: bool = False,
          acknowledge_gap: str | None = None, reason: str | None = None) -> Path:
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Generated releases belong in task Scratch, outside the repository')
    payloads, source = source_payloads(root, snapshot=snapshot)
    version_check = check_release_version(root, snapshot=snapshot,
                                         acknowledge_gap=acknowledge_gap, reason=reason)
    for warning in version_check['warnings']:
        print(f'WARNING: {warning}', file=sys.stderr)
    tree = ast.parse(payloads['XTA/__init__.py'].decode('utf-8'))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(
                       getattr(target, 'id', '') == '__version__' for target in node.targets))
    if version_check['target_version'] != version or (
            not snapshot and version_check['head_commit'] != source):
        raise ValueError('Source identity changed during the release-version check; retry from a stable checkout')
    launcher = f'GPT-6-Astra-Ultra_v{version}_SLURM.py'
    if launcher not in payloads:
        raise ValueError(f'Expected versioned launcher is absent: {launcher}')
    launchers = [name for name in payloads if name.endswith('_SLURM.py')]
    if launchers != [launcher]:
        raise ValueError(f'Expected exactly one versioned launcher: {launchers}')
    if readme:
        payloads['READ_ME_FIRST.txt'] = readme.read_bytes()
    manifest = {'version': version, 'launcher': launcher, 'source': source,
                'release_version_check': version_check,
                'files': {name: digest(data) for name, data in sorted(payloads.items())}}
    payloads['RELEASE_MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    if wheel:
        with zipfile.ZipFile(wheel) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f'Wheel has a corrupt member: {bad_member}')
            package = {name for name in archive.namelist() if name.startswith('XTA/')
                       and name.endswith(('.py', '.json', '.md'))}
            expected = {name for name in payloads if name.startswith('XTA/')}
            if package != expected:
                raise ValueError(f'Wheel package files differ: extra={package - expected}, missing={expected - package}')
            for name in expected:
                if archive.read(name) != payloads[name]:
                    raise ValueError(f'Wheel package bytes differ: {name}')
            wheel_launchers = [name for name in archive.namelist() if name.endswith('_SLURM.py')]
            if len(wheel_launchers) != 1 or not wheel_launchers[0].endswith('/' + launcher):
                raise ValueError(f'Wheel launcher differs: {wheel_launchers}')
            if archive.read(wheel_launchers[0]) != payloads[launcher]:
                raise ValueError('Wheel launcher bytes differ')
            metadata = archive.read(f'xta-{version}.dist-info/METADATA').decode()
            if f'Version: {version}' not in metadata.splitlines():
                raise ValueError('Wheel metadata version differs')
            if any(name.endswith(('.pyc', '.nbc', '.nbi')) for name in archive.namelist()):
                raise ValueError('Wheel contains generated bytecode or native caches')
        print(f'Wheel verified against {len(expected)} current package files.')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'XTA_v{version}_complete_source.zip'
    prefix = f'XTA_v{version}/'
    with tempfile.NamedTemporaryFile(dir=output, prefix='.release-', suffix='.zip', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, 'w') as target:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
                target.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with zipfile.ZipFile(temporary) as built:
            bad_member = built.testzip()
            if bad_member is not None:
                raise ValueError(f'Source archive has a corrupt member: {bad_member}')
            if set(built.namelist()) != {prefix + name for name in payloads}:
                raise ValueError('Source archive member list differs from its manifest')
            for name, data in payloads.items():
                if built.read(prefix + name) != data:
                    raise ValueError(f'Source archive bytes differ: {name}')
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    for artifact in (archive, *((wheel,) if wheel else ())):
        checksum = digest(artifact.read_bytes())
        artifact.with_suffix(artifact.suffix + '.sha256').write_text(
            f'{checksum}  {artifact.name}\n', encoding='ascii')
        print(f'{artifact.name}: {artifact.stat().st_size:,} bytes; SHA256 {checksum}')
    print(f'Verified {len(payloads)} source members from {source}.')
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--readme', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--snapshot', action='store_true',
                        help='validate the uncommitted working tree as a labelled development snapshot')
    parser.add_argument('--acknowledge-gap', metavar='FROM:TO',
                        help='Acknowledge this exact version gap only, e.g. v24.1.0:v26.0.0; requires --reason')
    parser.add_argument('--reason', help='Record the user-approved reason for --acknowledge-gap')
    args = parser.parse_args()
    try:
        build(root=ROOT, output=args.output_dir, readme=args.readme,
              wheel=args.wheel, snapshot=args.snapshot,
              acknowledge_gap=args.acknowledge_gap, reason=args.reason)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
