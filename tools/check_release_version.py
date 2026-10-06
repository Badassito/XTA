"""Check one release transition and local tag identity without changing Git.

The predecessor is the nearest semantic-version tag on the intended first-parent
lineage. Earlier transitions are not rescanned. Development snapshots are
explicitly labelled and never establish a release identity.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
_NUMBER = r'(?:0|[1-9][0-9]*)'
_VERSION = re.compile(rf'{_NUMBER}\.{_NUMBER}\.{_NUMBER}')
_TAG = re.compile(rf'v({_NUMBER}\.{_NUMBER}\.{_NUMBER})')
_LAUNCHER = re.compile(rf'.+_v({_NUMBER}\.{_NUMBER}\.{_NUMBER})_SLURM\.py')


class ReleaseVersionError(ValueError):
    """The requested source cannot establish an admissible release identity."""


def _version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise ReleaseVersionError(f'Invalid release version {value!r}; use MAJOR.MINOR.PATCH without leading zeroes')
    try:
        return tuple(int(part) for part in value.split('.'))
    except ValueError as error:
        raise ReleaseVersionError(f'Invalid release version: numeric component cannot be parsed: {error}') from error


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    try:
        result = subprocess.run(('git', *args), cwd=root, env=env, input=input,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseVersionError(f'Cannot read release Git history: {error}') from error
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise ReleaseVersionError(f'Cannot read release Git history ({args[0]}): {detail}')
    return result.stdout


def _source_version(root: Path, commit: str | None) -> str:
    """Read literal metadata and the sole root launcher without importing XTA."""
    if commit is None:
        initializer = root / 'XTA/__init__.py'
        if not initializer.is_file() or initializer.is_symlink() or not initializer.resolve().is_relative_to(root):
            raise ReleaseVersionError('Invalid release source: XTA/__init__.py must be a regular file inside the repository')
        try:
            source = initializer.read_bytes()
            launchers = [path.name for path in root.iterdir() if path.name.endswith('_SLURM.py')]
        except OSError as error:
            raise ReleaseVersionError(f'Cannot read release source: {error}') from error
        for name in launchers:
            path = root / name
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ReleaseVersionError(f'Invalid release launcher: {name}')
    else:
        entry = _git(root, 'ls-tree', '-z', commit, '--', 'XTA/__init__.py')
        if not entry or entry.split(b' ', 2)[0] not in {b'100644', b'100755'}:
            raise ReleaseVersionError(f'Invalid release source at {commit}: XTA/__init__.py is not a regular Git blob')
        source = _git(root, 'show', commit + ':XTA/__init__.py')
        launchers = []
        for row in _git(root, 'ls-tree', '-z', commit).split(b'\0'):
            if not row:
                continue
            metadata, raw_name = row.split(b'\t', 1)
            name = raw_name.decode('utf-8', errors='strict')
            if name.endswith('_SLURM.py'):
                mode, kind, _object_id = metadata.split(b' ')
                if kind != b'blob' or mode not in {b'100644', b'100755'}:
                    raise ReleaseVersionError(f'Invalid release launcher at {commit}: {name}')
                launchers.append(name)
    try:
        tree = ast.parse(source.decode('utf-8-sig'))
        values = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == '__version__' for target in node.targets):
                values.append(ast.literal_eval(node.value))
            elif isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', '') == '__version__':
                values.append(ast.literal_eval(node.value))
    except (SyntaxError, UnicodeError, ValueError, TypeError) as error:
        raise ReleaseVersionError(f'Invalid release source: cannot read literal XTA.__version__: {error}') from error
    if len(values) != 1:
        raise ReleaseVersionError('Invalid release source: expected exactly one literal XTA.__version__ assignment')
    version = values[0]
    _version(version)
    if len(launchers) != 1:
        raise ReleaseVersionError(f'Invalid release source: expected exactly one versioned root launcher, found {launchers}')
    match = _LAUNCHER.fullmatch(launchers[0])
    if match is None or match.group(1) != version:
        raise ReleaseVersionError(f'Invalid release source: launcher {launchers[0]!r} differs from package version {version}')
    return version


def _version_tags(root: Path) -> dict[str, dict[str, str | None]]:
    names = _git(root, 'for-each-ref', '--format=%(refname:strip=2)', 'refs/tags').decode('utf-8').splitlines()
    # Windows refs can differ in case across packed and loose storage. Inspect
    # uppercase aliases for collisions without treating them as lineage tags.
    selected = [(name, _TAG.fullmatch(name.casefold())) for name in names]
    selected = [(name, match.group(1)) for name, match in selected if match is not None]
    if not selected:
        return {}
    request = ''.join(f'refs/tags/{name}^{{commit}}\n' for name, _version_name in selected).encode('ascii')
    rows = _git(root, 'cat-file', '--batch-check=%(objectname) %(objecttype)', input=request).decode('ascii').splitlines()
    if len(rows) != len(selected):
        raise ReleaseVersionError('Cannot resolve all semantic-version tags to Git commits')
    tags = {}
    for (name, version), row in zip(selected, rows):
        object_id, kind = row.rsplit(' ', 1)
        # A malformed unrelated historical tag cannot be an ancestor release.
        # An occupied target with no commit remains an identity collision.
        tags[name] = {'version': version, 'tag': name, 'commit': object_id if kind == 'commit' else None}
    return tags


def _nearest_release(root: Path, lineage: list[str], tags: dict[str, dict[str, str | None]],
                     versions: dict[str, str]) -> dict[str, str] | None:
    by_commit = {}
    for tag in tags.values():
        if tag['commit'] is not None and tag['tag'] == 'v' + tag['version']:
            by_commit.setdefault(tag['commit'], []).append(tag)
    for commit in lineage:
        candidates = by_commit.get(commit, [])
        if not candidates:
            continue
        if len(candidates) != 1:
            names = sorted(tag['tag'] for tag in candidates)
            raise ReleaseVersionError(f'Ambiguous predecessor: multiple semantic-version tags at {commit}: {names}')
        release = candidates[0]
        declared = versions.get(commit)
        if declared is None:
            declared = versions[commit] = _source_version(root, commit)
        if declared != release['version']:
            raise ReleaseVersionError(f'Invalid predecessor source: {release["tag"]} declares package version {declared}')
        return dict(release)
    return None


def _acknowledgement(acknowledge_gap: str | None, reason: str | None,
                     transition: str, gap: bool) -> dict[str, str] | None:
    if acknowledge_gap is None:
        if reason is not None:
            raise ReleaseVersionError('A gap reason requires --acknowledge-gap FROM:TO')
        return None
    if not gap:
        raise ReleaseVersionError('A numbering acknowledgement applies only to a numbering gap')
    if acknowledge_gap != transition:
        raise ReleaseVersionError(f'Gap acknowledgement must match the exact transition {transition}')
    if not isinstance(reason, str) or not reason.strip():
        raise ReleaseVersionError('A numbering gap acknowledgement requires a nonempty reason')
    return {'transition': transition, 'reason': reason.strip()}


def _missing_ranges(previous: dict[str, str] | None, target: tuple[int, int, int]) -> list[dict[str, object]]:
    """Describe at most three missing ranges, without enumerating release names."""
    if previous is None:
        return []
    major, minor, patch = _version(previous['version'])
    new_major, new_minor, new_patch = target
    ranges = []
    def add(level, first, last, count):
        if count > 0:
            ranges.append({'level': level, 'first': first, 'last': last, 'count': count})
    if new_major == major and new_minor == minor:
        add('patch', f'v{major}.{minor}.{patch + 1}', f'v{major}.{minor}.{new_patch - 1}', new_patch - patch - 1)
    elif new_major == major:
        add('minor', f'v{major}.{minor + 1}.0', f'v{major}.{new_minor - 1}.0', new_minor - minor - 1)
        add('patch', f'v{major}.{new_minor}.0', f'v{major}.{new_minor}.{new_patch - 1}', new_patch)
    else:
        add('major', f'v{major + 1}.0.0', f'v{new_major - 1}.0.0', new_major - major - 1)
        add('minor', f'v{new_major}.0.0', f'v{new_major}.{new_minor - 1}.0', new_minor)
        add('patch', f'v{new_major}.{new_minor}.0', f'v{new_major}.{new_minor}.{new_patch - 1}', new_patch)
    return ranges


def _describe_ranges(ranges: list[dict[str, object]], tags: dict[str, dict[str, str | None]]) -> str:
    descriptions = []
    for item in ranges:
        first, last = _version(item['first'][1:]), _version(item['last'][1:])
        present = []
        for tag in tags.values():
            numbers = _version(tag['version'])
            if item['level'] == 'major':
                belongs = numbers[1:] == (0, 0) and first[0] <= numbers[0] <= last[0]
            elif item['level'] == 'minor':
                belongs = numbers[0] == first[0] and numbers[2] == 0 and first[1] <= numbers[1] <= last[1]
            else:
                belongs = numbers[:2] == first[:2] and first[2] <= numbers[2] <= last[2]
            if belongs:
                present.append(tag['tag'])
        # Ranges and exact present checkpoints remain bounded even for a very
        # large intentional jump. Existence is a local-ref fact, not a claim
        # that an off-lineage source is an admissible predecessor.
        present = sorted(present)
        item['present_elsewhere'] = present[:8]
        item['present_elsewhere_count'] = len(present)
        if item['level'] == 'major':
            series = f'v{first[0]}.x' if item['count'] == 1 else f'v{first[0]}.x through v{last[0]}.x'
            description = 'major series ' + series + ' (required starts '
        elif item['level'] == 'minor':
            series = f'v{first[0]}.{first[1]}.x' if item['count'] == 1 else f'v{first[0]}.{first[1]}.x through v{last[0]}.{last[1]}.x'
            description = 'minor series ' + series + ' (required starts '
        else:
            description = 'patch checkpoints '
        description += item['first'] if item['count'] == 1 else f'{item["first"]} through {item["last"]} ({item["count"]})'
        if item['level'] != 'patch':
            description += ')'
        if present:
            description += '; ' + ', '.join(present[:8]) + ' exists elsewhere in local refs'
            if len(present) > 8:
                description += f' ({len(present)} matching tags total)'
        else:
            description += '; no matching checkpoint tag in local refs'
        descriptions.append(description)
    return 'first-parent transition skips ' + '; '.join(descriptions)


def check_release_version(root: Path, *, target_version: str | None = None,
                          snapshot: bool = False, acknowledge_gap: str | None = None,
                          reason: str | None = None) -> dict[str, object]:
    """Check HEAD, a planned next version, or an explicitly labelled snapshot.

    A proposed version need not be written to the working tree yet. Numbering-gap
    acknowledgements match one exact vFROM:vTO transition (NONE for a first
    release), and cannot override source or tag identity failures.
    """
    if snapshot and target_version is not None:
        raise ReleaseVersionError('--snapshot and --target-version describe different source modes')
    root = Path(root).resolve()
    git_root = Path(os.fsdecode(_git(root, 'rev-parse', '--show-toplevel')).strip()).resolve()
    if git_root != root:
        raise ReleaseVersionError(f'{root} is not the Git repository root')
    head = _git(root, 'rev-parse', '--verify', 'HEAD^{commit}').decode('ascii').strip()
    shallow = _git(root, 'rev-parse', '--is-shallow-repository').strip() == b'true'
    warnings = []
    if shallow:
        if not snapshot:
            raise ReleaseVersionError('Shallow Git history cannot establish a release predecessor; obtain complete history')
        warnings.append('Shallow Git history does not establish a complete release lineage')
    versions = {}
    if snapshot:
        version = _source_version(root, None)
        kind = 'development-snapshot'
    else:
        versions[head] = _source_version(root, head)
        version = target_version if target_version is not None else versions[head]
        kind = 'planned-release' if target_version is not None else 'release'
    numbers = _version(version)
    tags = _version_tags(root)
    target_tags = [tag for name, tag in tags.items() if name.casefold() == 'v' + version]
    collision = bool(target_tags) and any(target_version is not None
        or tag['tag'] != 'v' + version or tag['commit'] != head for tag in target_tags)
    if collision:
        occupied = ', '.join(f'{tag["tag"]} at {tag["commit"] or "a non-commit object"}' for tag in target_tags)
        message = f'Release tag v{version} is already occupied by {occupied}; version reuse is an identity collision'
        if not snapshot:
            raise ReleaseVersionError(message)
        warnings.append(message)
    if not snapshot and target_version is None:
        head_tags = [tag for tag in tags.values() if tag['commit'] == head and tag['tag'] == 'v' + tag['version']]
        if len(head_tags) > 1:
            raise ReleaseVersionError(f'Ambiguous HEAD release tags: {sorted(tag["tag"] for tag in head_tags)}')
        if head_tags and head_tags[0]['version'] != version:
            raise ReleaseVersionError(f'HEAD tag {head_tags[0]["tag"]} differs from package version {version}')
    lineage = _git(root, 'rev-list', '--first-parent', head).decode('ascii').splitlines()
    if kind == 'release':
        lineage = lineage[1:]
    previous = _nearest_release(root, lineage, tags, versions)
    if previous is None:
        expected = ['0.0.0', '1.0.0']
        transition = 'NONE:v' + version
    else:
        major, minor, patch = _version(previous['version'])
        expected = [f'{major}.{minor}.{patch + 1}', f'{major}.{minor + 1}.0', f'{major + 1}.0.0']
        transition = previous['tag'] + ':v' + version
        # A snapshot may retain its baseline label while its bytes are dirty.
        retained_snapshot_label = snapshot and version == previous['version']
        if numbers <= (major, minor, patch) and not retained_snapshot_label:
            message = f'Release version must advance beyond {previous["tag"]}; requested v{version}'
            if not snapshot:
                raise ReleaseVersionError(message)
            warnings.append(message)
    retained_snapshot_label = snapshot and previous is not None and version == previous['version']
    gap = (version not in expected and not retained_snapshot_label
           and (previous is None or numbers > _version(previous['version'])))
    acknowledgement = _acknowledgement(acknowledge_gap, reason, transition, gap)
    numbering_gap = None
    if gap:
        ranges = _missing_ranges(previous, numbers)
        numbering_gap = {'transition': transition, 'scope': 'first-parent-transition', 'missing_ranges': ranges}
        message = f'Numbering gap for {transition}; expected ' + ', '.join('v' + item for item in expected)
        if ranges:
            message += '; ' + _describe_ranges(ranges, tags)
        if acknowledgement is None and not snapshot:
            raise ReleaseVersionError(message + '; acknowledge this exact gap with a reason only if intentional')
        warnings.append(message if acknowledgement is None else message + '; acknowledged: ' + acknowledgement['reason'])
    return {'kind': kind, 'scope': 'local-git-refs', 'head_commit': head, 'target_version': version,
            'previous_release': previous, 'expected_next_versions': expected,
            'warnings': warnings, 'acknowledgement': acknowledgement,
            'numbering_gap': numbering_gap, 'release_ready': not snapshot}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--target-version', help='check a proposed next MAJOR.MINOR.PATCH before editing or committing it')
    mode.add_argument('--snapshot', action='store_true', help='inspect the working tree as a development snapshot')
    parser.add_argument('--acknowledge-gap', metavar='FROM:TO', help='exact vFROM:vTO transition, or NONE:vTO for a first release')
    parser.add_argument('--reason', help='reason for the explicitly acknowledged numbering gap')
    parser.add_argument('--json', action='store_true', help='emit a machine-readable result')
    args = parser.parse_args(argv)
    try:
        result = check_release_version(args.root, target_version=args.target_version,
            snapshot=args.snapshot, acknowledge_gap=args.acknowledge_gap, reason=args.reason)
    except ReleaseVersionError as error:
        if args.json:
            print(json.dumps({'release_ready': False, 'error': str(error)}, indent=2))
        else:
            print(f'Release version check failed: {error}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        predecessor = result['previous_release']
        baseline = predecessor['tag'] if predecessor else 'no prior semantic-version release'
        readiness = 'numbering and tag identity passed' if result['release_ready'] else 'development snapshot; not a release'
        print(f'{result["kind"]}: v{result["target_version"]}, after {baseline}: {readiness}')
        for warning in result['warnings']:
            print('Warning: ' + warning)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
