"""Standalone qualification of bounded GPU bitset -> packed CVOL crops.

This deliberately does not alter the production Radial owner.  The source
bitset remains on its GPU; only crop metadata and packed crop bytes return to
the host.  Run ``--gpu`` explicitly to exercise CUDA and compare every output
byte with the production CPU bitset encoder.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from XTA.cylindrical_bitset_compaction import (
    RadialBitsetCompactor as BitsetCudaPackedEncoder,
    export_owner_bitset_blocks,
    plan_bitset_compaction,
)
from XTA.packed_publication import encode_owner_packed_block


MIB = 1024 * 1024
DEFAULT_DENSE_LIMIT = 64 * MIB
DEFAULT_PAYLOAD_LIMIT = 16 * MIB
PHASE_DEBUG = False


def _phase(label):
    if PHASE_DEBUG:
        print(f'phase: {label}', file=sys.stderr, flush=True)

def block_capacity(shape, *, dense_limit=DEFAULT_DENSE_LIMIT,
                   payload_limit=DEFAULT_PAYLOAD_LIMIT, max_slices=None):
    """Compatibility wrapper for the standalone CPU contract tests."""
    options = dict(dense_limit=dense_limit, payload_limit=payload_limit)
    if max_slices is not None:
        options['max_slices'] = max_slices
    plan = plan_bitset_compaction(shape, **options)
    return {name: getattr(plan, name) for name in
            ('block_slices', 'dense_bytes', 'payload_bytes', 'metadata_bytes', 'offset_bytes')}


def _fixture(shape, pattern):
    depth, height, width = shape
    dense = np.zeros(shape, np.uint8)
    if pattern == 'empty':
        pass
    elif pattern == 'edges':
        dense[0, 0, 0] = 1
        dense[-1, -1, -1] = 1
        dense[1 % depth, -1, min(1, width - 1)] = 1
    elif pattern == 'stripes':
        dense[:, :, 0::31] = 1
        dense[::2, 1::3, -1] = 1
    elif pattern == 'random':
        dense = (np.random.default_rng(20260924).random(shape) < .13).astype(np.uint8)
        dense[0] = 0
    else:
        raise ValueError(pattern)
    bits = np.packbits(dense.reshape(-1), bitorder='little')
    padded = np.pad(bits, (0, (-bits.size) % 4))
    return padded.view('<u4').astype(np.uint32, copy=False)


def _compare_block(actual, expected_records, expected_payload):
    if actual.payload.flags.writeable:
        raise AssertionError('GPU export payload is not read-only')
    fields = ('z', 'y0', 'y1', 'x0', 'x1', 'foreground', 'offset', 'size')
    a = [tuple(int(getattr(r, field)) for field in fields) for r in actual.records]
    e = [tuple(int(getattr(r, field)) for field in fields) for r in expected_records]
    if a != e or not np.array_equal(actual.payload, expected_payload):
        raise AssertionError(f'GPU packed block differs: records={a != e}, '
                             f'payload={not np.array_equal(actual.payload, expected_payload)}')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * MIB), b''):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_real_fixture(fixture_path, shape, output_dir, baseline_cvol):
    """Encode a mmap-backed owner bitset, checking exact stored index/payload.

    Uploading the source is setup for this standalone prototype.  After that,
    the encoder never downloads or duplicates the complete owner bitset.
    """
    import cupy as cp
    from XTA.interpolation import (
        IncrementalRawBBoxMaskStoreWriter,
        INTERNAL_PACKED_CVOL_FORMAT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(map(int, shape))
    words_expected = (math.prod(shape) + 31) // 32
    if fixture_path.stat().st_size != words_expected * 4:
        raise ValueError('fixture size differs from exact source bitset shape')
    sizes = block_capacity(shape)
    free, _total = cp.cuda.runtime.memGetInfo()
    source_bytes = words_expected * 4
    scratch_bytes = sum(sizes[name] for name in
                        ('dense_bytes', 'payload_bytes', 'metadata_bytes', 'offset_bytes'))
    reserve = 1024**3
    if free < source_bytes + scratch_bytes + reserve:
        raise RuntimeError(f'GPU free bytes {free} cannot hold source {source_bytes}, '
                           f'scratch {scratch_bytes}, and 1 GiB reserve')
    words = np.memmap(fixture_path, dtype='<u4', mode='r', shape=(words_expected,))
    _phase('real fixture full source H2D upload start (setup, not encoded output)')
    gpu_words = cp.asarray(words)
    _phase('real fixture full source H2D upload done')
    store_path = output_dir / 'gpu-compact.cvol'
    writer = IncrementalRawBBoxMaskStoreWriter(
        shape=shape, store_dir=store_path,
        format_name=INTERNAL_PACKED_CVOL_FORMAT,
        desc='Real owner bitset GPU compaction qualification')
    blocks = payload_bytes = nonempty = 0
    try:
        exported = export_owner_bitset_blocks(gpu_words, shape, 0)
        for encoded in exported.blocks:
            writer.consume_encoded_block(encoded.first_z, encoded.records,
                                         encoded.payload, packed=True)
            blocks += 1
            payload_bytes += encoded.payload.size
            nonempty += sum(r.foreground > 0 for r in encoded.records)
            if blocks % 32 == 0:
                _phase(f'real fixture encoded {blocks} blocks')
        if (exported.nonempty_slices != nonempty or exported.payload_bytes != payload_bytes):
            raise AssertionError('GPU export aggregate counts differ from its blocks')
        stats = dict(writer.finalize())
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    finally:
        del gpu_words
        del words
    hashes = {}
    for name in ('index.bin', 'chunks.bin'):
        current = _sha256(store_path / name)
        baseline = _sha256(baseline_cvol / name)
        hashes[name] = dict(gpu=current, baseline=baseline, exact=current == baseline)
    result = dict(shape=list(shape), fixture=str(fixture_path), baseline=str(baseline_cvol),
                  output=str(store_path), blocks=blocks, nonempty_slices=nonempty,
                  payload_bytes=payload_bytes, stats=stats, scratch=sizes,
                  free_gpu_bytes_before=free, hashes=hashes,
                  exact=all(item['exact'] for item in hashes.values()))
    (output_dir / 'real-fixture-result.json').write_text(json.dumps(result, indent=2),
                                                         encoding='utf-8')
    if not result['exact']:
        raise AssertionError('Real GPU compact CVOL index/payload differs from CPU baseline')
    return result


def _publish_owned_payload(shape, store_path, *, words=None, blocks=None):
    """Run the actual CPU packed encoder or append already-encoded GPU blocks."""
    from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT

    started = time.perf_counter()
    cpu_started = time.thread_time()
    writer = IncrementalRawBBoxMaskStoreWriter(
        shape=shape, store_dir=store_path,
        format_name=INTERNAL_PACKED_CVOL_FORMAT,
        desc='Owner bitset ABBA publication')
    try:
        if words is not None:
            plane = shape[1] * shape[2]
            block_z = max(1, 256 * MIB // plane)
            for first in range(0, shape[0], block_z):
                count = min(block_z, shape[0] - first)
                start, stop = first * plane, (first + count) * plane
                if not np.any(words[start // 32:(stop + 31) // 32]):
                    writer.consume_empty_range(first, count)
                    continue
                records, payload = encode_owner_packed_block(words, shape, first, count)
                writer.consume_encoded_block(first, records, payload, packed=True)
        else:
            for block in blocks:
                writer.consume_encoded_block(block.first_z, block.records,
                                             block.payload, packed=True)
        stats = dict(writer.finalize())
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    return dict(wall_seconds=time.perf_counter() - started,
                thread_cpu_seconds=time.thread_time() - cpu_started,
                stats=stats)


def benchmark_real_abba(fixture_path, shape, output_dir, baseline_cvol, heat_seconds=60):
    """Warm ABBA: owner-ready boundary separated from async CPU publication."""
    import cupy as cp
    from tools.benchmark_radial_setup import heatsoak

    output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(map(int, shape))
    expected_words = (math.prod(shape) + 31) // 32
    if fixture_path.stat().st_size != expected_words * 4:
        raise ValueError('fixture size differs from exact source bitset shape')
    words = np.memmap(fixture_path, dtype='<u4', mode='r', shape=(expected_words,))
    sizes = block_capacity(shape)
    # Warm both JIT paths and CUDA kernels before heatsoak/timing.  The CPU
    # warmup stays tiny; it exercises both Numba signatures without scanning
    # the full fixture.  Existing CUDA qualifier already established parity.
    warm_shape = (2, 3, 37)
    encode_owner_packed_block(_fixture(warm_shape, 'edges'), warm_shape, 0, 2)
    warm_gpu = cp.asarray(_fixture(warm_shape, 'edges'))
    with BitsetCudaPackedEncoder(warm_gpu, warm_shape, 0) as warm_encoder:
        list(warm_encoder.blocks())
    del warm_gpu
    cp.get_default_memory_pool().free_all_blocks()
    heat_actual = heatsoak(heat_seconds, 0)

    rounds = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='packed-publication') as pool:
        for index, mode in enumerate(('cpu', 'gpu', 'gpu', 'cpu'), start=1):
            free_before, total = cp.cuda.runtime.memGetInfo()
            required = expected_words * 4 + sum(sizes[k] for k in
                ('dense_bytes', 'payload_bytes', 'metadata_bytes', 'offset_bytes')) + 1024**3
            if free_before < required:
                raise RuntimeError(f'round {index}: GPU has {free_before} free bytes; needs {required}')
            _phase(f'ABBA {index} {mode}: source H2D setup start')
            gpu_words = cp.asarray(words)
            cp.cuda.get_current_stream().synchronize()
            _phase(f'ABBA {index} {mode}: source H2D setup done')
            started = time.perf_counter()
            cpu_started = time.thread_time()
            if mode == 'cpu':
                owned_words = cp.asnumpy(gpu_words)
                owned_blocks = None
                retained_bytes = owned_words.nbytes
            else:
                owned_words = None
                with BitsetCudaPackedEncoder(gpu_words, shape, 0) as encoder:
                    owned_blocks = list(encoder.blocks())
                retained_bytes = sum(block.payload.nbytes for block in owned_blocks)
            del gpu_words
            cp.get_default_memory_pool().free_all_blocks()
            ready_at = time.perf_counter()
            ready_cpu_at = time.thread_time()
            store_path = output_dir / f'{index}-{mode}.cvol'
            submitted = time.perf_counter()
            future = pool.submit(_publish_owned_payload, shape, store_path,
                                 words=owned_words, blocks=owned_blocks)
            submit_done = time.perf_counter()
            publication = future.result()
            complete_at = time.perf_counter()
            hashes = {name: dict(candidate=_sha256(store_path / name),
                                 baseline=_sha256(baseline_cvol / name))
                      for name in ('index.bin', 'chunks.bin')}
            exact = all(pair['candidate'] == pair['baseline'] for pair in hashes.values())
            if not exact:
                raise AssertionError(f'ABBA round {index} {mode} CVOL differs from baseline')
            rounds.append(dict(round=index, mode=mode, free_gpu_bytes_before=free_before,
                               total_gpu_bytes=total, retained_host_bytes=retained_bytes,
                               source_h2d_excluded=True,
                               owner_ready_wall_seconds=ready_at - started,
                               owner_ready_thread_cpu_seconds=ready_cpu_at - cpu_started,
                               submit_wall_seconds=submit_done - submitted,
                               publication_wall_seconds=publication['wall_seconds'],
                               publication_thread_cpu_seconds=publication['thread_cpu_seconds'],
                               ready_to_completion_seconds=complete_at - ready_at,
                               owner_to_completion_seconds=complete_at - started,
                               stats=publication['stats'], hashes=hashes, exact=True))
            _phase(f'ABBA {index} {mode}: ready={ready_at-started:.3f}s, '
                   f'complete={complete_at-started:.3f}s')
            del owned_words, owned_blocks
    result = dict(shape=list(shape), fixture=str(fixture_path),
                  baseline=str(baseline_cvol), heatsoak_seconds=heat_actual,
                  scratch=sizes, rounds=rounds,
                  caveat='Local GPU and disk; owner setup H2D excluded. No four-worker contention.')
    (output_dir / 'abba-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def qualify_gpu_case(shape, pattern, *, output_dir=None, max_slices=None):
    import cupy as cp
    from XTA.interpolation import (
        IncrementalRawBBoxMaskStoreWriter,
        INTERNAL_PACKED_CVOL_FORMAT,
        RawBBoxMaskStore,
    )

    _phase(f'case {shape} {pattern} fixture start')
    words = _fixture(shape, pattern)
    _phase(f'case {shape} {pattern} cp.asarray start')
    gpu_words = cp.asarray(words)
    _phase(f'case {shape} {pattern} cp.asarray done')
    payload_bytes = blocks = nonempty = 0
    with tempfile.TemporaryDirectory() as temporary:
        store_path = Path(temporary) / 'packed.cvol'
        writer = IncrementalRawBBoxMaskStoreWriter(
            shape=shape, store_dir=store_path,
            format_name=INTERNAL_PACKED_CVOL_FORMAT,
            desc='GPU bitset compaction qualification')
        try:
            options = {} if max_slices is None else {'max_slices': max_slices}
            exported = export_owner_bitset_blocks(gpu_words, shape, 0, **options)
            scratch = block_capacity(shape, max_slices=max_slices)
            for actual in exported.blocks:
                first = actual.first_z
                count = len(actual.records)
                _phase(f'block {first}:{first+count} CPU oracle start')
                expected = encode_owner_packed_block(words, shape, first, count)
                _compare_block(actual, *expected)
                _phase(f'block {first}:{first+count} CVOL writer start')
                writer.consume_encoded_block(first, actual.records, actual.payload, packed=True)
                payload_bytes += actual.payload.size
                blocks += 1
                nonempty += sum(r.foreground > 0 for r in actual.records)
            if exported.nonempty_slices != nonempty or exported.payload_bytes != payload_bytes:
                raise AssertionError('GPU export aggregate counts differ from its blocks')
            writer.finalize()
            _phase(f'case {shape} {pattern} CVOL decode start')
            store = RawBBoxMaskStore.open(store_path)
            try:
                count = math.prod(shape)
                dense = np.unpackbits(words.view(np.uint8), bitorder='little')[:count].reshape(shape)
                for z in range(shape[0]):
                    np.testing.assert_array_equal(store.decode_slice(z), dense[z])
            finally:
                store.close()
        finally:
            writer.discard()
    if not np.array_equal(gpu_words.get(), words):
        raise AssertionError('borrowed source bitset changed during GPU encoding')
    result = dict(shape=list(shape), pattern=pattern, blocks=blocks,
                  nonempty_slices=nonempty, payload_bytes=payload_bytes,
                  scratch=scratch, max_slices_override=max_slices,
                  borrowed_source_unchanged=True, exact=True)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tail = f'-max{max_slices}' if max_slices is not None else ''
        (output_dir / f'{pattern}-{shape[0]}x{shape[1]}x{shape[2]}{tail}.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='explicitly run CUDA parity cases')
    parser.add_argument('--debug-phases', action='store_true', help='log phase boundaries and 30-second stack dumps')
    parser.add_argument('--first-case-only', action='store_true', help='run only the first empty unaligned case')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--real-fixture', type=Path,
                        help='mmap a production owner bitset for whole-store qualification')
    parser.add_argument('--baseline-cvol', type=Path,
                        help='CPU reference CVOL to compare index.bin and chunks.bin')
    parser.add_argument('--shape', type=int, nargs=3, metavar=('T', 'H', 'W'),
                        default=(1931, 3064, 3022))
    parser.add_argument('--benchmark-abba', action='store_true',
                        help='heatsoak and compare owner-ready latency with async CPU publication')
    parser.add_argument('--heat-seconds', type=float, default=60)
    args = parser.parse_args()
    if not args.gpu:
        parser.error('CUDA qualification requires explicit --gpu')
    global PHASE_DEBUG
    PHASE_DEBUG = args.debug_phases
    if PHASE_DEBUG:
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
    if args.real_fixture is not None:
        if args.output_dir is None or args.baseline_cvol is None:
            parser.error('--real-fixture requires --output-dir and --baseline-cvol')
        if args.benchmark_abba:
            print(json.dumps(benchmark_real_abba(args.real_fixture, args.shape,
                                                 args.output_dir, args.baseline_cvol,
                                                 args.heat_seconds), indent=2))
        else:
            print(json.dumps(qualify_real_fixture(args.real_fixture, args.shape,
                                                  args.output_dir, args.baseline_cvol), indent=2))
        return
    shapes = ((9, 3, 3022), (7, 11, 37), (3, 9, 8), (5, 7, 1))
    cases = [(shape, pattern) for shape in shapes
             for pattern in ('empty', 'edges', 'stripes', 'random')]
    if args.first_case_only:
        cases = cases[:1]
    results = [qualify_gpu_case(shape, pattern, output_dir=args.output_dir)
               for shape, pattern in cases]
    if not args.first_case_only:
        results.extend(qualify_gpu_case((9, 3, 3022), pattern,
            output_dir=args.output_dir, max_slices=6)
            for pattern in ('empty', 'edges', 'stripes', 'random'))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
