"""Copy one saved D1 lease layer and verify every crop without dense volumes.

The input must be a completed native_pieces layer with disjoint frame leases.
The output directory must be new; input files are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import struct
import sys
import time
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.confidence_evidence import ConfidenceEvidenceRef
from XTA.confidence_native import write_native_disjoint_leases
from XTA.confidence_storage import PIECE_LAYOUT


def _require(value, message):
    if not value:
        raise ValueError(message)


def _update_digest(path, digest):
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)


def qualify(source, output_dir):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    _require(not output_dir.is_relative_to(source) and not source.is_relative_to(output_dir),
             'Qualification output must be separate from the input')
    before = ConfidenceEvidenceRef.open(source)
    _require(before.metadata.get('layout') == PIECE_LAYOUT
             and before.metadata.get('piece_contract') == 'disjoint_frame_leases'
             and before.metadata.get('pieces_disjoint') is True, 'Expected disjoint native frame leases')
    pieces = []
    for piece in sorted(before.metadata['pieces'], key=lambda value:value['offset_tyx'][0]):
        directory = (source/piece['directory']).resolve()
        _require(directory != source and directory.is_relative_to(source), 'Input piece escapes parent')
        reference = ConfidenceEvidenceRef.open(directory)
        _require(list(reference.storage_shape) == piece['stored_shape_tyx']
                 and reference.layer_key == piece['layer_key'], 'Input piece descriptor differs')
        pieces.append(dict(reference=reference, offset_tyx=piece['offset_tyx']))
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir/'consolidated'
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    peak = process.memory_info().rss if process is not None else None
    started = time.perf_counter()
    with mock.patch('zlib.compress', side_effect=AssertionError('recompression forbidden')), mock.patch(
            'zlib.decompressobj', side_effect=AssertionError('decompression forbidden')):
        after = write_native_disjoint_leases(destination, pieces, native_shape=before.storage_shape,
            source_shape=before.source_shape, layer_key=before.layer_key, model_name=before.model_name,
            provenance=before.metadata['provenance'])
    copy_seconds = time.perf_counter()-started
    print(f'Copied {len(pieces)} leases in {copy_seconds:.3f}s; validating every crop.', flush=True)
    for field in ('coordinate_space','stored_shape_tyx','source_shape_tyx','output_shape_tyx','dtype','unknown',
                  'layer_key','model_name','score_semantics','known_voxels','known_contributions','provenance'):
        _require(before.metadata[field] == after.metadata[field], f'Output metadata differs: {field}')
    _require(len(list(destination.iterdir())) == 3, 'Output does not contain exactly three files')
    source_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    for piece in pieces:
        _update_digest(piece['reference'].path/'scores.u8.zlib', source_digest)
    _update_digest(destination/'scores.u8.zlib', target_digest)
    _require(source_digest.hexdigest() == target_digest.hexdigest() == after.metadata['payload_sha256'],
             'Compressed payload is not exact source concatenation')
    started = time.perf_counter()
    missing, crop_digest = object(), hashlib.sha256()
    known, total_crops, total_raw = 0, 0, 0
    with before.native_reader() as left_reader, after.native_reader() as right_reader:
        for z in range(before.storage_shape[0]):
            for left, right in zip_longest(left_reader.iter_crops(z), right_reader.iter_crops(z), fillvalue=missing):
                _require(left is not missing and right is not missing, f'Crop count differs at frame {z}')
                _require(left[:4] == right[:4] and np.array_equal(left[4], right[4]),
                         f'Crop coordinates or score bytes differ at frame {z}')
                crop_digest.update(struct.pack('<IIIII',z,*left[:4]))
                crop_digest.update(left[4].tobytes())
                known += int(np.count_nonzero(left[4]))
                total_crops += 1
                total_raw += left[4].nbytes
            if process is not None and z%256 == 0:
                peak = max(peak, process.memory_info().rss)
    _require(known == before.metadata['known_voxels'] == after.metadata['known_voxels'],
             'Known-support count differs')
    verification_seconds = time.perf_counter()-started
    source_count = sum(path.is_file() for path in source.rglob('*'))
    memory = process.memory_info() if process is not None else None
    result = dict(source=str(source), destination=str(destination), source_file_count=source_count,
        target_file_count=3, pieces=len(pieces), native_shape=before.storage_shape, source_shape=before.source_shape,
        compressed_copy_seconds=copy_seconds, full_crop_verification_seconds=verification_seconds,
        payload_bytes=after.metadata['payload_bytes'], block_count=after.metadata['block_count'],
        exact_crops=total_crops, decoded_crop_bytes_per_side=total_raw, known_voxels=known,
        crop_sequence_sha256=crop_digest.hexdigest(), compressed_payload_sha256=target_digest.hexdigest(),
        index_sha256=after.metadata['index_sha256'],
        peak_process_rss_bytes=getattr(memory,'peak_wset',peak), sampled_peak_rss_bytes=peak,
        verification='Every crop coordinate and score byte matched; numeric zero remains unknown. '
                     'Geometry and provenance are unchanged. Compressed payload is exact source concatenation.',
        limits='CPU-only copy of one saved layer. No full planes, volumes, native NRRDs or projection. '
               'Copy timings do not measure pipeline throughput.')
    (output_dir/'validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Completed native_pieces layer directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='New validation directory under Scratch')
    args = parser.parse_args(argv)
    print(json.dumps(qualify(args.source, args.output_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
