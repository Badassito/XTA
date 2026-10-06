"""Draft or replace the single current-source inventory after source review.

Draft/evidence files belong outside the source tree. --write replaces only the
active inventory atomically; it does not alter source, versions, Git or History.
An inventory is source identity, not a numerical qualification claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__:
    from . import verify_package_inventory as inventory
else:
    import verify_package_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.inventory-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            failed = sys.exc_info()[0] is not None
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                if not failed:
                    raise


def prepare(*, output_dir: Path, write: bool = False, root: Path = ROOT) -> dict[str, object]:
    root, output_dir = root.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(root):
        raise ValueError('Generated inventory evidence belongs outside the source tree')
    draft, source_identity = inventory.build_inventory(root)
    draft_bytes = _json_bytes(draft)
    summary = {'version': draft['version'], 'source_files': len(draft['files']),
               'inventory_sha256': inventory.digest(draft_bytes), 'written': write,
               'source_identity': source_identity,
               'qualification_claim': 'Source identity only; numerical qualification is separate'}
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / 'current_source_inventory.json', draft_bytes)
    inventory._assert_unchanged(root, source_identity)
    if write:
        target = inventory._safe_path(root, inventory.INVENTORY_PATH)
        previous = target.read_bytes() if target.exists() else None
        _atomic_write(target, draft_bytes)
        try:
            after = inventory.capture_source_identity(root)
            expected_files = dict(source_identity['files'])
            expected_files[inventory.INVENTORY_PATH] = inventory.digest(draft_bytes)
            if after['commit'] != source_identity['commit'] or after['files'] != expected_files:
                raise RuntimeError('Current source changed while publishing the inventory')
            inventory.verify(root)
        except BaseException:
            if target.read_bytes() == draft_bytes:
                if previous is None:
                    target.unlink()
                else:
                    _atomic_write(target, previous)
            raise
    _atomic_write(output_dir / 'current_source_inventory_receipt.json', _json_bytes(summary))
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    result = prepare(root=args.root, output_dir=args.output_dir, write=args.write)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_identity'}))


if __name__ == '__main__':
    main()
