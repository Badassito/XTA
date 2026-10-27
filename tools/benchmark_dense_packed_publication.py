"""Isolated CPU timings for legacy and bounded native packed publication.

All pixel masks, CVOL outputs, profiles, and results stay under --output.
The regular-file backing keeps old/new I/O identical on Windows; this is a
component sanity check, not a target-system throughput qualification.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import packed_dense_source as dense
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter
from XTA.publication_memory import publication_ram_headroom


def fixture(shape, seed):
    volume = np.zeros(shape, np.uint8)
    h0, h1 = shape[1] // 4, 3 * shape[1] // 4
    w0, w1 = shape[2] // 3, 2 * shape[2] // 3
    rng = np.random.default_rng(seed)
    volume[:, h0:h1, w0:w1] = rng.integers(0, 2, (shape[0], h1-h0, w1-w0), dtype=np.uint8) * np.uint8(255)
    volume.setflags(write=False)
    return volume


def write_variant(volume, path, variant, *, output_planes=False):
    publication_started = time.perf_counter()
    writer = IncrementalRawBBoxMaskStoreWriter(shape=volume.shape, store_dir=path,
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc=f'CPU packed publication {variant}')
    plane_bound = volume.shape[1] * ((volume.shape[2] + 7) // 8)
    block = 1 if output_planes else max(1, min(dense.MAX_DENSE_PACKED_BLOCK_SLICES,
        dense.DENSE_PACKED_PAYLOAD_BYTES // plane_bound))
    slot = np.empty(block * plane_bound, np.uint8) if variant == 'native_encoded' else None
    headroom_calls = 0
    headroom_seconds = 0.
    def probe():
        nonlocal headroom_calls, headroom_seconds
        headroom_calls += 1
        started = time.perf_counter()
        publication_ram_headroom()
        headroom_seconds += time.perf_counter()-started
    try:
        if variant == 'native_encoded':
            for first in range(0, volume.shape[0], block):
                count = min(block, volume.shape[0]-first)
                probe()
                records, payload = dense.encode_dense_u8_packed_block(volume, first, count,
                    payload_buffer=slot)
                writer.consume_encoded_block(first, records, payload, packed=True)
        else:
            next_probe = 0
            probe_window = max(1, (32 * 1024**2) // max(1, plane_bound))
            for z in range(volume.shape[0]):
                if variant == 'legacy_per_slice_guard' or z >= next_probe:
                    probe()
                    next_probe = z + probe_window
                writer.consume(z, volume[z:z+1])
        stats = writer.finalize()
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    publication_seconds = time.perf_counter()-publication_started
    return dict(stats=stats, headroom_calls=headroom_calls, headroom_seconds=headroom_seconds,
        publication_seconds=publication_seconds,
        chunks_sha256=hashlib.sha256((path/'chunks.bin').read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256((path/'index.bin').read_bytes()).hexdigest())


def top_profile(profile):
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
        rows.append(dict(file=filename, line=line, name=name, calls=calls,
                         own_seconds=own, cumulative_seconds=cumulative))
    return sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)[:30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    root = args.output.resolve() / ('dense-packing-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    started_wall_time = time.time()
    source = fixture((64, 2048, 2048), 719)
    output = fixture((4, 3064, 3022), 737)
    started = time.perf_counter()
    dense.encode_dense_u8_packed_block(source, 0, 1)
    jit_seconds = time.perf_counter()-started
    slot = np.empty(dense.DENSE_PACKED_PAYLOAD_BYTES, np.uint8)
    started = time.perf_counter()
    heat_calls = 0
    while time.perf_counter()-started < args.heatsoak_seconds:
        dense.encode_dense_u8_packed_block(source, 0, 64, payload_buffer=slot)
        heat_calls += 1
        if heat_calls % 100 == 0:
            print(f'CPU heat soak {time.perf_counter()-started:.1f}s', flush=True)
    heat_seconds = time.perf_counter()-started
    report = dict(platform=platform.platform(), processor=platform.processor(),
        python=sys.version, numpy=np.__version__, jit_seconds=jit_seconds,
        started_wall_time=started_wall_time,
        started_utc=datetime.fromtimestamp(started_wall_time, timezone.utc).isoformat(),
        cpu_heatsoak_seconds=heat_seconds, heatsoak_calls=heat_calls,
        note='Identical regular-file CVOL backing; host headroom probes; no GPU. Component sanity check.',
        cases={})
    for name, volume in (('source_block', source), ('projection_output_planes', output)):
        variants = (('legacy_per_slice_guard', 'legacy_window_guard', 'native_encoded')
                    if name == 'source_block' else ('legacy_window_guard', 'native_encoded'))
        case = dict(shape=list(volume.shape), variants={})
        reference = None
        for variant in variants:
            runs = []
            for repeat in range(args.repeat):
                started = time.perf_counter()
                measured_wall_start = time.time()
                result = write_variant(volume, root/f'{name}-{variant}-{repeat}.cvol', variant,
                    output_planes=name=='projection_output_planes')
                result['seconds'] = time.perf_counter()-started
                result['measured_wall_start'] = measured_wall_start
                result['measured_wall_end'] = time.time()
                digest = (result['chunks_sha256'],result['index_sha256'])
                if reference is None:
                    reference = digest
                if digest != reference:
                    raise RuntimeError('Encoded crop payload/index differs from legacy publication')
                runs.append(result)
            profile = cProfile.Profile()
            profile.enable()
            write_variant(volume, root/f'{name}-{variant}-profile.cvol', variant,
                output_planes=name=='projection_output_planes')
            profile.disable()
            profile.dump_stats(str(root/f'{name}-{variant}.prof'))
            case['variants'][variant] = dict(median_seconds=statistics.median(r['publication_seconds'] for r in runs),
                median_with_validation_seconds=statistics.median(r['seconds'] for r in runs),
                runs=runs, profile=top_profile(profile))
        report['cases'][name] = case
        print(name, {k:round(v['median_seconds'],5) for k,v in case['variants'].items()}, flush=True)
    report['source_files'] = {str(path.relative_to(Path(__file__).resolve().parents[1])):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (Path(__file__), Path(dense.__file__))}
    report['finished_wall_time'] = time.time()
    report['finished_utc'] = datetime.fromtimestamp(report['finished_wall_time'], timezone.utc).isoformat()
    (root/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(root/'results.json', flush=True)


if __name__ == '__main__':
    main()
