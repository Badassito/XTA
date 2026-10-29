"""Streaming gzip readers for native-mask preparation.

ISA-L is optional (the project's ``acceleration`` extra). Both readers retain
gzip member/trailer validation and use the caller's bounded ``readinto`` buffers.
Callers must read through EOF to validate the complete stream, as with GzipFile.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import os

try:
    from isal.igzip import IGzipFile as _IGzipFile
except ImportError:
    _IGzipFile = None


def gzip_backend() -> str:
    """Return the selected reader; the environment override permits comparisons."""
    selected = os.environ.get('XTA_NATIVE_GZIP_BACKEND', 'auto').lower()
    if selected not in ('auto', 'stdlib', 'isal'):
        raise ValueError('XTA_NATIVE_GZIP_BACKEND must be auto, stdlib, or isal.')
    if selected == 'isal' and _IGzipFile is None:
        raise RuntimeError('XTA_NATIVE_GZIP_BACKEND=isal requires the optional isal package.')
    return 'isal' if _IGzipFile is not None and selected != 'stdlib' else 'stdlib'


@contextmanager
def open_gzip_reader(raw):
    """Wrap an already positioned binary file without taking ownership of it.

    Backend selection happens before reading. Corrupt payload exceptions are
    propagated directly; a failed ISA-L read is never retried with another codec.
    Concatenated members are consumed by the selected gzip-compatible reader.
    """
    reader_type = _IGzipFile if gzip_backend() == 'isal' else gzip.GzipFile
    with reader_type(fileobj=raw, mode='rb') as reader:
        yield reader
