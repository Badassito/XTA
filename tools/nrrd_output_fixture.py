"""Stream a saved binary NRRD into an exact, full-geometry packed CVOL fixture.

The converter holds one decoded TYX plane at a time. It refuses to replace an
existing fixture and verifies the CVOL by decoding one plane at a time again.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from XTA.interpolation import (  # noqa: E402
    INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter,
    RawBBoxMaskStore,
)

SCRATCH = Path.home() / 'Documents' / 'ChatGPT' / 'Scratch'
MAX_HEADER_BYTES = 1024 * 1024


def read_header(handle) -> tuple[dict[str, str], int]:
    """Read only the inline NRRD header, leaving the handle at its gzip payload."""
    lines: list[bytes] = []
    size = 0
    while True:
        line = handle.readline(MAX_HEADER_BYTES - size + 1)
        size += len(line)
        if not line or size > MAX_HEADER_BYTES:
            raise ValueError('NRRD header is missing its blank-line delimiter or is too large')
        if line in (b'\n', b'\r\n'):
            break
        lines.append(line.rstrip(b'\r\n'))
    if not lines or lines[0] != b'NRRD0005':
        raise ValueError('Expected an inline NRRD0005 header')
    fields: dict[str, str] = {}
    for raw in lines[1:]:
        if raw.startswith(b'#') or b':' not in raw:
            continue
        key, value = raw.split(b':', 1)
        fields[key.decode('ascii').strip()] = value.decode('ascii').strip().lstrip('=').strip()
    if fields.get('type', '').lower() not in ('unsigned char', 'uint8', 'uint8_t'):
        raise ValueError(f"Expected uint8 NRRD, got {fields.get('type')!r}")
    if fields.get('dimension') != '3' or fields.get('encoding', '').lower() != 'gzip':
        raise ValueError('Expected a 3D gzip NRRD')
    if any(key in fields for key in ('data file', 'byte skip', 'line skip')):
        raise ValueError('Detached or skipped NRRD payloads are unsupported')
    sizes = tuple(int(token) for token in fields.get('sizes', '').split())
    if len(sizes) != 3 or any(value <= 0 for value in sizes):
        raise ValueError(f'Invalid NRRD sizes: {sizes}')
    return fields, handle.tell()


def read_exact_into(source: gzip.GzipFile, target: memoryview) -> None:
    offset = 0
    while offset < len(target):
        count = source.readinto(target[offset:])
        if not count:
            raise EOFError(f'Gzip NRRD ended after {offset}/{len(target)} bytes of a plane')
        offset += int(count)


def convert(source_path: Path, output_dir: Path) -> dict[str, object]:
    source_path = source_path.resolve(strict=True)
    scratch = SCRATCH.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(scratch):
        raise ValueError(f'Fixture output must be under Scratch: {scratch}')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to replace an existing fixture: {output_dir}')
    output_dir.mkdir(parents=True)
    store_dir = output_dir / 'fixture.cvol'
    receipt_path = output_dir / 'receipt.json'

    writer = None
    with source_path.open('rb') as handle:
        fields, payload_offset = read_header(handle)
        x, y, t = (int(value) for value in fields['sizes'].split())
        shape_tyx = (t, y, x)
        plane_bytes = x * y
        decoded_plane = bytearray(plane_bytes)
        plane_view = memoryview(decoded_plane)
        plane_array = np.frombuffer(decoded_plane, dtype=np.uint8).reshape((y, x))
        decoded_hash = hashlib.sha256()
        writer = IncrementalRawBBoxMaskStoreWriter(
            shape=shape_tyx,
            store_dir=store_dir,
            format_name=INTERNAL_PACKED_CVOL_FORMAT,
            desc=f'production NRRD fixture {source_path.name}',
            extra_meta={'fixture_source_name': source_path.name},
        )
        started = time.perf_counter()
        try:
            with gzip.GzipFile(fileobj=handle, mode='rb') as payload:
                for z in range(t):
                    read_exact_into(payload, plane_view)
                    decoded_hash.update(plane_view)
                    if int(plane_array.max()) > 1:
                        raise ValueError(f'NRRD plane {z} contains values outside binary 0/1')
                    writer.consume(z, plane_array[np.newaxis, :, :])
                if payload.read(1):
                    raise ValueError('NRRD gzip payload contains more voxels than its header declares')
            stats = writer.finalize()
        except BaseException:
            writer.discard()
            raise
        decode_write_seconds = time.perf_counter() - started

    source_extent = fields.get('Segment0_Extent')
    parsed_source_extent = (
        [int(value) for value in source_extent.split()]
        if source_extent is not None else None
    )
    produced_extent = [int(value) for value in stats['segment_extent_ijk']]
    if parsed_source_extent is not None and parsed_source_extent != produced_extent:
        raise ValueError(
            f'Fixture foreground extent {produced_extent} differs from NRRD header '
            f'{parsed_source_extent}'
        )

    verify_started = time.perf_counter()
    cv_hash = hashlib.sha256()
    verify_plane = np.empty((y, x), dtype=np.uint8)
    store = RawBBoxMaskStore.open(store_dir, mmap_payload=True)
    try:
        for z in range(t):
            store.fill_decoded_slice_into(z, verify_plane)
            cv_hash.update(memoryview(verify_plane).cast('B'))
    finally:
        store.close()
    cvol_digest = cv_hash.hexdigest()
    source_digest = decoded_hash.hexdigest()
    if cvol_digest != source_digest:
        raise RuntimeError('CVOL decoded SHA-256 differs from the source gzip NRRD')

    receipt: dict[str, object] = {
        'source_nrrd': str(source_path),
        'source_name': source_path.name,
        'source_bytes': source_path.stat().st_size,
        'source_header_bytes': payload_offset,
        'shape_tyx': list(shape_tyx),
        'logical_bytes': t * plane_bytes,
        'segment_extent_ijk': produced_extent,
        'source_segment_extent_ijk': parsed_source_extent,
        'decoded_sha256': source_digest,
        'expected_decoded_sha256': source_digest,
        'cvol_decoded_sha256': cvol_digest,
        'cvol_path': str(store_dir),
        'cvol_format': INTERNAL_PACKED_CVOL_FORMAT,
        'cvol_stats': stats,
        'decode_write_seconds': decode_write_seconds,
        'verify_seconds': time.perf_counter() - verify_started,
        'gpu_used': False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Saved full-quality uint8 gzip NRRD')
    parser.add_argument('--output-dir', required=True, type=Path, help='New task-specific Scratch directory')
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output_dir), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
