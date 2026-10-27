#!/usr/bin/env python3
"""Verify a single F2 native-to-preview downbin and edited preview subset."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_planes(path, start, stop):
    with path.open('rb') as f:
        fields = {}
        for raw in iter(f.readline, b''):
            if not raw.strip():
                break
            line = raw.decode().strip()
            if ':' in line:
                k, v = line.split(':', 1)
                fields[k] = v.strip()
        shape = tuple(map(int, fields['sizes'].split()[::-1]))
        count = shape[1] * shape[2]
        with gzip.GzipFile(fileobj=f) as payload:
            payload.seek(start * count)
            data = payload.read((stop - start) * count)
    return np.frombuffer(data, np.uint8).reshape(stop - start, *shape[1:]), shape


def comparison(original, candidate):
    original, candidate = original != 0, candidate != 0
    return {'original_pixels': int(original.sum()), 'candidate_pixels': int(candidate.sum()),
            'added_pixels': int(np.count_nonzero(candidate & ~original)),
            'removed_pixels': int(np.count_nonzero(original & ~candidate)),
            'equal': bool(np.array_equal(original, candidate))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--original-low', type=Path, required=True)
    p.add_argument('--low-t', type=int, default=228)
    a = p.parse_args()
    metadata = json.loads((a.audit / 'prepper_f2_sampled_geometry.json').read_text())
    bundle = Path(metadata['bundle'])
    edits = json.loads((bundle / 'edits.json').read_text())
    global_entry = next(s for s in edits['segments'] if s['recomposition_op'] == 'select')
    full = tuple(edits['full_quality_shape_tyx'])
    low = tuple(edits['preview_shape_tyx'])
    start = math.floor(a.low_t * full[0] / low[0])
    stop = math.ceil((a.low_t + 1) * full[0] / low[0])
    original_low_hash = digest(a.original_low)
    print('Original low source hashed; reading bounded native contributors.', flush=True)
    native, shape = read_planes(a.audit / 'f2_original_native_global.nrrd', start, stop)
    if shape != full:
        raise ValueError('Native shape mismatch')
    reconstructed = np.zeros(low[1:], np.uint8)
    for frame in native:
        reconstructed |= (cv2.resize(np.ascontiguousarray((frame > 0).astype(np.uint8) * 255),
            (low[2], low[1]), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    actual_low, actual_low_shape = read_planes(a.original_low, a.low_t, a.low_t + 1)
    if actual_low_shape != low:
        raise ValueError('Low shape mismatch')
    with np.load(bundle / global_entry['file']) as snapshot:
        crop = snapshot['mask']
        affine = snapshot['ijk_to_ras']
        relative = np.diag([-1., -1., 1., 1.]) @ affine
        if not np.allclose(relative[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError('Edited snapshot orientation mismatch')
        offset = np.rint(relative[:3, 3]).astype(np.int64)
        extent = snapshot['extent_xyz'] + np.repeat(offset, 2)
        x0, x1, y0, y1, z0, z1 = map(int, extent)
        edited = np.zeros(low[1:], np.uint8)
        if z0 <= a.low_t <= z1:
            ys, ye = max(0, y0), min(low[1], y1 + 1)
            xs, xe = max(0, x0), min(low[2], x1 + 1)
            if ys < ye and xs < xe:
                edited[ys:ye, xs:xe] = crop[a.low_t - z0, ys-y0:ye-y0, xs-x0:xe-x0]
    result = {'source_paths': {'original_low': str(a.original_low), 'edited_bundle': str(bundle),
                             'original_native': str(a.audit / 'f2_original_native_global.nrrd')},
              'low_t': a.low_t, 'native_contributors_t_exclusive_stop': [start, stop],
              'low_shape_tyx': list(low), 'native_shape_tyx': list(full),
              'original_low_sha256': original_low_hash,
              'edited_global_expected_original_low_sha256': global_entry['source_sha256'],
              'original_low_hash_matches_edited_receipt_source': original_low_hash == global_entry['source_sha256'],
              'native_downbin_vs_original_low': comparison(actual_low[0], reconstructed),
              'original_low_vs_edited_global': comparison(actual_low[0], edited),
              'method': 'XY OpenCV INTER_AREA of uint8 binary255 threshold>0, OR all contributing native T slices; same current production CPU downbin contract.',
              'limitation': 'One preview plane only. Original native and original low masks predate manual deletions. Exact agreement at this plane establishes this bounded downbin correspondence, not the full source run identity.'}
    (a.audit / 'f2_native_downbin_t228_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    cv2.imwrite(str(a.audit / 'f2_original_low_t228.png'), actual_low[0] * 255)
    cv2.imwrite(str(a.audit / 'f2_edited_low_t228.png'), edited * 255)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
