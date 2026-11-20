"""Dependency-light content hashing for local artifact files."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hash a local file in bounded chunks without loading its full payload."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source_bytes(data: bytes) -> bytes:
    """Normalize CRLF in NUL-free UTF-8 text; retain lone CR and binary bytes.

    Matches tools/verify_package_inventory.canonical_source_bytes, so a source
    identity equals that file's release-inventory digest on LF and CRLF trees.
    """
    if b"\0" in data:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n")


def source_sha256(path: str | Path) -> str:
    """Line-ending-stable implementation identity of one source file."""
    return hashlib.sha256(canonical_source_bytes(Path(path).read_bytes())).hexdigest()
