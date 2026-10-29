"""Lightweight path and evidence helpers for read-only PTA diagnostics."""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def external_output(path: Path, *, inputs: tuple[Path, ...] = ()) -> Path:
    """Require generated evidence outside both maintained source and inputs."""
    output = Path(path).expanduser().resolve()
    if output.is_relative_to(REPO_ROOT):
        raise ValueError('Generated diagnostic evidence must be outside the repository')
    for source in inputs:
        source = Path(source).expanduser().resolve()
        if output.is_relative_to(source):
            raise ValueError(f'Generated diagnostic evidence must be outside the input directory: {source}')
    return output


def receipt_child(directory: Path, name: str) -> Path:
    directory = Path(directory).resolve()
    candidate = Path(name)
    if (not name or candidate.is_absolute() or len(candidate.parts) != 1
            or name in {'.', '..'} or '\\' in name or '/' in name or ':' in name):
        raise ValueError(f'Expected a plain receipt member filename: {name!r}')
    path = (directory / candidate).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f'Receipt member escapes its input directory: {name!r}')
    return path


def write_json(output: Path, name: str, value: object) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    destination = receipt_child(output, name)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return destination
