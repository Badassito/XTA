"""Paired bounded GPU pipeline probe against the committed 160624 baseline.

The large in-plane coordinates exercise D2H and packing costs, while the short
physical stack bounds local memory and runtime. This does not estimate run ETA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_native_pull_cuda import gpu_lock, heatsoak, source_fingerprints


def payload_hashes(path):
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ('index.bin', 'chunks.bin')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    args = parser.parse_args()
    if args.repetitions < 1 or args.heatsoak_seconds < 60:
        parser.error('At least one pair and 60 seconds heat soak are required')
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('cuda-temp', 'cuda-cache'):
        (output / name).mkdir(exist_ok=True)
    os.environ.update(TEMP=str(output / 'cuda-temp'), TMP=str(output / 'cuda-temp'),
        CUPY_CACHE_DIR=str(output / 'cuda-cache'), CUDA_VISIBLE_DEVICES='0',
        YOLO_TTA_GPU_BACKPROJECT='1', YOLO_TTA_NATIVE_PULL_CUDA='1',
        YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT='1',
        YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS='0', YOLO_TTA_NATIVE_PULL_BACKEND='compiled')
    from XTA import geometry, packed_tilted_projection as current, backprojection as bp
    from XTA.interpolation import RawBBoxMaskStore
    baseline_commit = '15a667c'
    frozen = subprocess.check_output(['git', 'show',
        f'{baseline_commit}:XTA/packed_tilted_projection.py'], cwd=repo)
    frozen_path = output / 'baseline_packed_tilted_projection.py'
    frozen_path.write_bytes(frozen)
    module_name = 'XTA._benchmark_native_pipeline_baseline'
    spec = importlib.util.spec_from_file_location(module_name, frozen_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = baseline
    spec.loader.exec_module(baseline)
    work, target = (32, 3064, 3023), (24, 3064, 3023)
    view = geometry._build_azimuthal_view_info(*work, base_view='transverse',
        azimuth_angle=15., azimuthal_native_raster=0, request_token='paired-pipeline')
    source = np.random.default_rng(160624).integers(0, 16,
        (view.num_slices, 256, 256), dtype=np.uint8)
    source[:] = source == 0
    source[:, :, :3] = 0
    source.flags.writeable = False
    source_path = output / 'source.cvol'
    pack_stats = current.pack_terminal_native_source(source, source_path,
        desc='Independent reduced source for paired native pipeline probe')
    initial_source_hashes = payload_hashes(source_path)
    records, warmup = [], []
    with gpu_lock(repo.parent / 'Scratch' / 'Temp' / 'GPU_LOCK'):
        before = source_fingerprints(repo)
        soak = heatsoak(args.heatsoak_seconds)
        expected_hashes = None
        for repetition in range(-1, args.repetitions):
            arms = [('baseline', baseline), ('current', current)]
            if repetition % 2:
                arms.reverse()
            stores = []
            for name, module in arms:
                bp._reset_main_process_gpu_stage_coordinator()
                path = output / f'{name}-{repetition + 1}.cvol'
                started = time.perf_counter()
                stats = module.project_tilted_sparse_store(source_path, view, path,
                    out_shape_tyx=target, workers=8)
                elapsed = time.perf_counter() - started
                assert stats['gpu_planes'] == target[0], stats
                hashes = payload_hashes(path)
                if expected_hashes is None:
                    expected_hashes = hashes
                assert hashes == expected_hashes, (name, repetition, hashes, expected_hashes)
                snapshot = bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
                assert not snapshot['stage_leases'] and not snapshot['resident_owners']
                record = dict(arm=name, repetition=repetition, seconds=elapsed,
                    stats=stats, hashes=hashes, actual_gpu=True)
                (warmup if repetition < 0 else records).append(record)
                stores.append(RawBBoxMaskStore.open(path, mmap_payload=True))
                print(json.dumps(record), flush=True)
            try:
                for z in range(target[0]):
                    np.testing.assert_array_equal(stores[0].decode_slice(z), stores[1].decode_slice(z))
            finally:
                for store in stores:
                    store.close()
        after = source_fingerprints(repo)
        changed = [path for path in before if before[path] != after[path]]
        assert not changed, changed
        assert payload_hashes(source_path) == initial_source_hashes
    medians = {arm: statistics.median(item['seconds'] for item in records if item['arm'] == arm)
        for arm in ('baseline', 'current')}
    report = dict(scope='Local bounded pipeline sanity; short stack and reduced source; no full-run ETA',
        baseline_commit=baseline_commit, baseline_module_sha256=hashlib.sha256(frozen).hexdigest(),
        working_shape=work, output_shape=target, source_shape=source.shape,
        source_bytes=source.nbytes, source_pack=pack_stats, heatsoak_seconds=soak,
        cold_warmup=warmup, measurements=records, median_seconds=medians,
        speedup=medians['baseline'] / medians['current'],
        payload_index_byte_parity=True, decoded_parity=True,
        source_before=before, source_after=after, changed_during_probe=changed,
        windows_disk_fallback=os.name == 'nt')
    (output / 'paired-native-pipeline.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(medians=medians, speedup=report['speedup'], parity=True)), flush=True)


if __name__ == '__main__':
    main()
