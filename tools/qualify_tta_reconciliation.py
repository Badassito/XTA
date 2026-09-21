"""Qualify real CPU/GPU/hybrid TTA reconciliation using tiny local models.

This verifies transport, geometry and publication, not segmentation accuracy.
All models, logs, caches and decoded validation artifacts stay under --output.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICIES = ROOT / 'XTA/examples/external_reconciliation'
CASES = ('cpu_off', 'cpu_union', 'gpu_off', 'gpu_union', 'cpu_confidence',
         'gpu_confidence', 'hybrid_augmented_tiles', 'hybrid_augmented_tiles_union',
         'gpu_native_views', 'gpu_native_views_confidence')
BRIDGE_CASES = ('cpu_bridges_off', 'cpu_bridges_union', 'cpu_bridges_provenance')


def create_cpu_fixtures(root, *, size=64, frames=16, missing_indices=()):
    """Create the CPU-only equivalent without importing CUDA or TensorRT."""
    import cv2
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    size = int(size)
    image = ops.parameter([1, 1, size, size], np.float32, name='images')
    proto = ops.subtract(image, ops.constant(np.float32(.35)))
    head = np.zeros((1, 6, 8), np.float32)
    head[:, :4, 0] = [size / 2, size / 2, size, size]
    head[:, 4:, 0] = [.9, 1.]
    head_node = ops.constant(head)
    head_node.output(0).get_tensor().set_names({'output0'})
    proto.output(0).get_tensor().set_names({'output1'})
    model = ov.Model([head_node, proto], [image], 'reconciliation_cpu_fixture')
    ov.save_model(model, root / 'model.xml')
    writer = cv2.VideoWriter(str(root / 'input.mkv'), cv2.VideoWriter_fourcc(*'FFV1'),
                             8., (size, size), False)
    if not writer.isOpened():
        raise RuntimeError('FFV1 fixture writer unavailable')
    try:
        yy, xx = np.mgrid[:size, :size]
        scale = size / 64.
        for index in range(int(frames)):
            frame = np.zeros((size, size), dtype=np.uint8)
            if index not in missing_indices:
                frame[(xx-(30+index % 4)*scale)**2 + (yy-29*scale)**2 < 220*scale*scale] = 220
                frame[int(28*scale):int(34*scale), int(29*scale):int(35*scale)] = 0
            writer.write(frame)
    finally:
        writer.release()


def runtime_environment(root):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMBA_NUM_THREADS='2',
        SLURM_CPUS_PER_TASK='8', YOLO_TTA_CPU_SOCKET_RESERVE_CORES='0',
        YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0', YOLO_TTA_TELEMETRY='0',
        YOLO_CONFIG_DIR=str(root / 'ultralytics-config'), NUMBA_CACHE_DIR=str(root / 'numba-cache'),
        CUPY_CACHE_DIR=str(root / 'cupy-cache'), TORCHINDUCTOR_CACHE_DIR=str(root / 'torchinductor-cache'),
        TRITON_CACHE_DIR=str(root / 'triton-cache'), NO_ALBUMENTATIONS_UPDATE='1',
        PTA_GPU_TORCH_COMPILE='0', YOLO_AUTOINSTALL='false')
    return env


def invocation(root, output, name):
    hybrid = name in ('hybrid_augmented_tiles', 'hybrid_augmented_tiles_union')
    cpu = name.startswith('cpu_')
    mode = 'cpu' if cpu else 'gpu'
    models = [f'{mode}:{root / ("model.xml" if cpu else "model.engine")}']
    if hybrid:
        models.append(f'cpu:{root / "model.xml"}')
    argv = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta', '--input', str(root / 'input.mkv'),
        '--model', *models, '--device', '0:cpu' if hybrid else ('cpu' if cpu else '0'),
        '--quantize', f'{mode}:fp32', *(['cpu:fp32'] if hybrid else []),
        '--batch', f'{mode}:1', *(['cpu:1'] if hybrid else []),
        '--channel_format', 'gray', '--imgsz', '64', '--cpu_threads', '2',
        '--cpu_streams', '1', '--cpu_infer_requests', '2', '--enable_cartesian',
        'transverse', 'sagittal', 'coronal', '--angle', '0', '--conf', '.1',
        '--min_conf', '0', '--min_radius', '0', '--interpolation_distance', '0',
        '--output', str(output / 'outputs'), '--temp', str(output / 'runtime'),
        '--save', 'nrrd', 'summary']
    if not name.endswith('_off'):
        policy = ('confidence_core_rescue' if hybrid and not name.endswith('_union')
                  else 'hybrid_with_fill' if name.endswith('_provenance')
                  else 'confidence_core_rescue' if name.endswith('_confidence') else 'union')
        argv.extend(['--reconciliation', str(POLICIES / f'{policy}.py'), '--reconciliation_memory_mib', '64'])
    if hybrid:
        argv.extend(['--enable_tile', '32:32', '--augmentation_ratio', '3', '--augmentation',
            f'gpu:{ROOT / "XTA/examples/external_augmentations/GPU_light.py"}',
            f'cpu:{ROOT / "XTA/examples/external_augmentations/CPU_light.py"}',
            '--augmentation_coverage', 'packed'])
    if name.startswith('gpu_native_views'):
        argv.extend(['--enable_azimuthal', 'transverse:30', '--enable_radial', 'transverse',
                     '--enable_spherical', 'transverse'])
    if name in BRIDGE_CASES:
        argv[argv.index('--imgsz') + 1] = '32'
        argv[argv.index('--interpolation_distance') + 1] = '6'
        argv.extend(['--interpolation_walk_back', '1', '--interpolation_candidates', '1',
                     '--interpolation_passes', '1', '--interpolation_min_radius', '0',
                     '--interpolation_search_angle', '45'])
    return argv


def decode_nrrd(path):
    import nrrd
    import numpy as np
    header = nrrd.read_header(str(path))
    raw = path.read_bytes()
    boundary = re.search(b'\r?\n\r?\n', raw)
    if not boundary or header['encoding'] != 'gzip' or header['type'] != 'unsigned char':
        raise ValueError(f'Unsupported qualification NRRD: {path}')
    values = np.frombuffer(gzip.decompress(raw[boundary.end():]), dtype=np.uint8)
    xyz = values.reshape(tuple(header['sizes']), order='F')
    if not np.all((xyz == 0) | (xyz == 1)):
        raise AssertionError('NRRD values are not binary')
    if not np.allclose(header['space directions'], np.eye(3)) or not np.allclose(header['space origin'], 0):
        raise AssertionError('Qualification fixture must retain its common source grid')
    return np.ascontiguousarray(xyz.transpose(2, 1, 0))


def inspect_case(output, name):
    import numpy as np
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    run = json.loads((output / 'manifest.json').read_text())
    if run['status'] != 'complete':
        raise AssertionError('Pipeline did not publish a complete run manifest')
    manifest = next((output / 'nrrd').glob('*_nrrd_manifest.json'))
    layers = json.loads(manifest.read_text())['layers']
    components, final = {}, None
    for layer in layers:
        values = decode_nrrd(manifest.parent / layer['filename'])
        if layer['source'] == 'global' and layer['stage'] == 'final_output_after_all_postprocessing':
            final = values
        elif layer.get('layer_role', 'additive_component') == 'additive_component' and layer['recomposition_op'] == 'union':
            key = '|'.join(str(layer.get(field, '')) for field in
                           ('view_name', 'source', 'mask_kind', 'tile_config_id', 'tile_acceptance', 'stage'))
            if key in components:
                raise AssertionError(f'Duplicate component identity: {key}')
            components[key] = values
    if final is None or not np.any(final):
        raise AssertionError('Fixture produced no nonempty final checkpoint')
    union = np.zeros(final.shape, dtype=np.uint8)
    for value in components.values():
        if value.shape != final.shape:
            raise AssertionError('Component and final source grids differ')
        union |= value
    if np.any(final & ~union):
        raise AssertionError('Reconciled final is not a subset of the additive components')
    receipt = dict(name=name, output=str(output), shape_tyx=list(final.shape),
        component_count=len(components), component_union_voxels=int(union.sum()),
        final_voxels=int(final.sum()), final_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
        component_hashes={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in components.items()},
        final_is_candidate_subset=True)
    if not name.endswith('_off'):
        report = json.loads((output / 'reconciliation/manifest.json').read_text())
        score_manifest = json.loads((output / 'reconciliation_evidence/manifest.json').read_text())
        saved = {(entry['model_name'], entry['layer_key']): entry for entry in score_manifest['layers']}
        values_present, known_total, prediction_layers = set(), 0, 0
        for layer in report['layers'].values():
            metadata = layer['metadata']
            if metadata['mask_kind'] != 'yolo':
                continue
            prediction_layers += 1
            entry = saved[(metadata['model_name'], metadata['layer_key'])]
            ref = ConfidenceEvidenceRef.open(output / 'reconciliation_evidence' / entry['directory'])
            if ref.shape != final.shape:
                raise AssertionError('Retained confidence has the wrong source grid')
            if Path(metadata['confidence_evidence']).resolve() != ref.path.resolve():
                raise AssertionError('Runtime layer confidence descriptor does not match saved sidecar')
            work = output.parent / 'score_verification' / entry['directory']
            with ref.source_reader(work) as reader:
                for z in range(final.shape[0]):
                    scores, known = reader(z, z+1)
                    if not np.array_equal(known, scores > 0):
                        raise AssertionError('Score known-support semantics changed')
                    known_total += int(known.sum())
                    values_present.update(map(int, np.unique(scores[known])))
        # The actual fixture score is float32 .9; backend arithmetic may round
        # its u8 tie to 229 or 230. Neither the .1 selection threshold nor a
        # fabricated binary score of 255 can satisfy this assertion.
        if known_total <= 0 or not values_present or not values_present <= {229, 230}:
            raise AssertionError(f'Unexpected observed instance confidence: {sorted(values_present)}')
        receipt.update(confidence_layers=prediction_layers, known_score_voxels=known_total,
                       known_score_values=sorted(values_present), reconciliation_counts=report['counts'])
        if report['policy']['mode'] == 'confidence' and report['counts']['confidence_known_voxels'] <= 0:
            raise AssertionError('Confidence reconciliation consumed no actual scores')
        if report['policy']['mode'] == 'union' and not np.array_equal(union, final):
            raise AssertionError('Union reconciliation changed the exact additive union')
        if report['policy']['mode'] == 'union':
            execution = report.get('execution', {})
            if (execution.get('strategy') != 'reuse_assembled_union'
                    or execution.get('component_payload_reads') != 0
                    or execution.get('confidence_payload_reads') != 0
                    or execution.get('new_source_volume_bytes') != 0):
                raise AssertionError('Union policy did not reuse the already assembled source output')
            receipt['union_reused_without_payload_reads'] = True
    elif (output / 'reconciliation_evidence').exists():
        raise AssertionError('Retention-off run unexpectedly wrote score sidecars')
    if name in ('hybrid_augmented_tiles', 'hybrid_augmented_tiles_union'):
        augmentation = json.loads((output / 'augmentation_manifest.json').read_text())
        backends = sorted({entry['backend'] for entry in augmentation['execution_records']})
        if backends != ['cpu', 'gpu'] or not all(entry['pass_count'] == 3 for entry in augmentation['execution_records']):
            raise AssertionError(f'Hybrid three-pass augmentation did not execute both backends: {backends}')
        if not any(layer['source'] == 'tile' for layer in layers):
            raise AssertionError('Hybrid fixture did not publish accepted tile predictions')
        receipt['augmentation_backends'] = backends
    log = (output.parent / 'pipeline.log').read_text(encoding='utf-8')
    if name in ('gpu_off', 'gpu_union', 'gpu_confidence'):
        if 'D1 owner' not in log or 'source-space layer' not in log:
            raise AssertionError('GPU qualification did not exercise the D1 owner path')
        receipt['d1_owner_exercised'] = True
    return receipt


def verify_bridge_evidence(output):
    import numpy as np
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    manifest = next((output / 'nrrd').glob('*_nrrd_manifest.json'))
    layers = json.loads(manifest.read_text())['layers']
    predictions = bridges = None
    bridge_count = 0
    for layer in layers:
        if layer.get('layer_role', 'additive_component') != 'additive_component' or layer['recomposition_op'] != 'union':
            continue
        mask = decode_nrrd(manifest.parent / layer['filename'])
        if predictions is None:
            predictions, bridges = np.zeros_like(mask), np.zeros_like(mask)
        if layer['mask_kind'] == 'bridge':
            bridge_count += int(bool(np.any(mask)))
            bridges |= mask
        elif layer['mask_kind'] == 'yolo':
            predictions |= mask
    if predictions is None or bridge_count == 0:
        raise AssertionError('Interpolation fixture produced no nonempty bridge layers')
    bridge_only = (bridges != 0) & (predictions == 0)
    if not np.any(bridge_only):
        raise AssertionError('Bridge layers add no foreground beyond predictions')
    result = dict(nonempty_bridge_layers=bridge_count, bridge_voxels=int(bridges.sum()),
                  bridge_only_voxels=int(bridge_only.sum()))
    report_path = output / 'reconciliation/manifest.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        known_any = np.zeros_like(bridge_only)
        prediction_count = bridge_roles = 0
        for record in report['layers'].values():
            metadata = record['metadata']
            if metadata['mask_kind'] == 'bridge':
                bridge_roles += 1
                if record['role'] != 'bridge' or metadata.get('confidence_evidence'):
                    raise AssertionError('Bridge received prediction confidence/provenance')
            elif metadata['mask_kind'] == 'yolo':
                prediction_count += 1
                ref = ConfidenceEvidenceRef.open(metadata['confidence_evidence'])
                work = output.parent / 'bridge_score_verification' / ref.path.name
                with ref.source_reader(work) as reader:
                    for z in range(ref.shape[0]):
                        scores, known = reader(z, z+1)
                        known_any[z] |= known[0]
        if not bridge_roles or not prediction_count or np.any(known_any & bridge_only):
            raise AssertionError('Bridge-only voxels received invented observed confidence')
        result.update(bridge_roles=bridge_roles, prediction_roles=prediction_count,
                      bridge_only_confidence_unknown=True)
        if report['policy']['name'] == 'hybrid_with_fill':
            weights = report['policy']['provenance_weights']
            if not weights['bridge'] < weights['prediction']:
                raise AssertionError('Provenance policy did not retain lower bridge weight')
            result['provenance_weights'] = weights
    return result


def verify_parity(before, after):
    if before['component_hashes'] != after['component_hashes']:
        raise AssertionError('Confidence retention changed additive component masks')
    if before['final_sha256'] != after['final_sha256']:
        raise AssertionError('Union reconciliation changed the final mask')
    return dict(component_masks_identical=True, final_mask_identical=True,
                retention_off=before['output'], retention_on=after['output'])


def run_case(root, name):
    index = 1
    while (root / f'{name}_{index:02d}').exists():
        index += 1
    case = root / f'{name}_{index:02d}'
    case.mkdir()
    argv = invocation(root, case, name)
    (case / 'invocation.json').write_text(json.dumps(argv, indent=2))
    print(f'Running {name}: {case}', flush=True)
    started = time.monotonic()
    with (case / 'pipeline.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=runtime_environment(root), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} pipeline failed ({result.returncode}); see {case / "pipeline.log"}')
    receipt = inspect_case(case / 'outputs', name)
    if name in BRIDGE_CASES:
        receipt['bridges'] = verify_bridge_evidence(case / 'outputs')
    receipt['elapsed_seconds'] = time.monotonic() - started
    (case / 'verification.json').write_text(json.dumps(receipt, indent=2))
    (root / f'{name}_latest.json').write_text(json.dumps(receipt, indent=2))
    print(f'Passed {name}: {receipt["component_count"]} components, {receipt["final_voxels"]} final voxels', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path)
    parser.add_argument('--cases', nargs='+', choices=CASES + BRIDGE_CASES, default=list(CASES))
    parser.add_argument('--reuse-fixtures', action='store_true')
    args = parser.parse_args()
    bridge_fixture = any(name in BRIDGE_CASES for name in args.cases)
    if bridge_fixture and not all(name in BRIDGE_CASES for name in args.cases):
        parser.error('Run the 32-pixel bridge fixture separately from the 64-pixel standard cases')
    needs_gpu = any(not name.startswith('cpu_') for name in args.cases)
    if needs_gpu and args.gpu_lock is None:
        parser.error('--gpu-lock is required for GPU qualification')
    root = args.output.resolve()
    if root.is_relative_to(ROOT) or (root.exists() and not args.reuse_fixtures):
        parser.error('Use a fresh directory outside the repository, or --reuse-fixtures')
    root.mkdir(parents=True, exist_ok=True)
    lock = args.gpu_lock.resolve() if needs_gpu else None
    if lock is not None:
        lock.parent.mkdir(parents=True, exist_ok=True)
    while lock is not None:
        try:
            with lock.open('x') as stream:
                json.dump(dict(task='reconciliation pipeline qualification', pid=os.getpid(),
                    start_time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), stream)
            break
        except FileExistsError:
            print('Waiting for GPU_LOCK', flush=True)
            time.sleep(5)
    try:
        os.environ.update(runtime_environment(root))
        if needs_gpu and not (root / 'model.engine').exists():
            from tools.qualify_tta_hybrid_augmentation import fixtures
            fixtures(root)
            print('Built real TensorRT and OpenVINO segmentation fixtures.', flush=True)
        elif not (root / 'model.xml').exists():
            create_cpu_fixtures(root, **dict(size=32, frames=32, missing_indices=(10, 11, 12, 13))
                                if bridge_fixture else {})
            print('Built real OpenVINO segmentation fixture.', flush=True)
        for name in args.cases:
            run_case(root, name)
        parity = {}
        for backend in ('cpu', 'gpu'):
            paths = [root / f'{backend}_{state}_latest.json' for state in ('off', 'union')]
            if all(path.exists() for path in paths):
                parity[backend] = verify_parity(*(json.loads(path.read_text()) for path in paths))
        if bridge_fixture:
            paths = [root / f'cpu_bridges_{state}_latest.json' for state in ('off', 'union')]
            if all(path.exists() for path in paths):
                parity['cpu_bridges'] = verify_parity(*(json.loads(path.read_text()) for path in paths))
        (root / 'parity.json').write_text(json.dumps(parity, indent=2))
        print(json.dumps(dict(parity=parity), indent=2), flush=True)
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
