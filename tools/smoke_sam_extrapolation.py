"""Prepare and run a bounded real-model SAM extrapolation functional smoke.

Historical input masks define an explicitly scoped fixture, not an accuracy
reference. Preparation is CPU-only. Run requires an atomic shared GPU lock and
an unchanged protocol/source tree. Outputs stay in the requested Scratch folder.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_pins():
    paths = sorted((REPO / 'XTA').rglob('*.py')) + [Path(__file__).resolve()]
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def terminal_selector(frame):
    return lambda observation, direction: observation.frame_index == frame and direction == 1


def plan_record(baseline, mode, terminal, distance, radius, scope):
    from XTA.geometry import get_view_infos
    from XTA.sam_extrapolation import prepare_sam_extrapolation_pass
    view = get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    prepared = prepare_sam_extrapolation_pass(baseline, view=view, scope=scope,
        distance=distance, walk_back=0, min_radius=radius, crop_mode=mode,
        eligible_terminals=terminal_selector(terminal))
    if len(prepared.runs) != 1 or len(prepared.runs[0].seed_ids) != 1:
        raise ValueError('Smoke fixture must yield exactly one independently seeded forward run')
    group = prepared.groups[0]
    expected = tuple(range(terminal, terminal + distance + 1))
    if prepared.runs[0].expected_frames != expected or group.terminal_frame != terminal:
        raise ValueError('Smoke horizon must fit beyond the actual remaining terminal')
    rejected = prepare_sam_extrapolation_pass(baseline, view=view, scope=scope + '/radius-reject',
        distance=distance, walk_back=0, min_radius=group.terminal_radius + .5,
        crop_mode=mode, eligible_terminals=terminal_selector(terminal))
    if rejected.needs_tracking or rejected.plan.skipped_by_min_radius != 1:
        raise AssertionError('Terminal-radius rejection did not exclude the original seed')
    return dict(terminal_frame=terminal, terminal_radius=group.terminal_radius,
        context_bbox_yx=list(group.context_bbox_yx), expected_frames=list(expected),
        output_frames=list(prepared.runs[0].output_frames), seed_ids=list(prepared.runs[0].seed_ids),
        tracker_jobs=len(prepared.tracker_jobs) if mode == 'tiled' else len(prepared.runs),
        radius_gate_rejected=True)


def prepare(args):
    import numpy as np
    from XTA.lta_sam import resolve_local_sam_bundle
    directory = args.case_directory.resolve()
    metadata_path = directory / 'cases.json'
    metadata = json.loads(metadata_path.read_text('utf-8'))
    case = next(row for row in metadata['cases'] if row['id'] == args.case_id)
    selected_path = directory / case['id'] / 'sam_selected.npy'
    detector_path = directory / case['observations']
    selected = np.load(selected_path, mmap_mode='r', allow_pickle=False)
    detector = np.load(detector_path, mmap_mode='r', allow_pickle=False)
    detector_packed_sha = hashlib.sha256(np.packbits(detector != 0).tobytes()).hexdigest()
    if case.get('observation_sha256') and detector_packed_sha != case['observation_sha256']:
        raise ValueError('Original detector pixels differ from their retained packed-mask identity')
    crop_shape = tuple(metadata['image_shape'])
    if selected.shape != crop_shape or detector.shape != crop_shape:
        raise ValueError('Retained masks differ from their declared native crop geometry')
    first, terminal = args.first_baseline_frame, args.terminal_frame
    if not (0 <= first <= terminal < crop_shape[0] and 1 <= args.distance <= 64
            and terminal + args.distance < crop_shape[0]):
        raise ValueError('Baseline window and complete tail horizon must fit the retained images')
    if not np.all((detector[first:terminal + 1] == 0) | (selected[first:terminal + 1] != 0)):
        raise ValueError('The retained interpolation baseline removed an in-scope original observation')
    if args.full_native_images is None:
        image_path = (directory / metadata['image_path']).resolve()
        shape = crop_shape
        bbox = (0, 0, shape[1], shape[2])
        coordinate_scope = 'native-resolution local crop chart'
    else:
        if args.full_native_shape is None:
            raise ValueError('--full-native-images requires --full-native-shape')
        image_path = args.full_native_images.resolve()
        shape = tuple(args.full_native_shape)
        x0, y0, x1, y1 = metadata['source_crop_xyxy']
        bbox = (y0, x0, y1, x1)
        if shape[0] != crop_shape[0] or not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
            raise ValueError('Native crop lies outside the declared full native images')
        crop = np.memmap(directory / metadata['image_path'], dtype='uint8', mode='r', shape=crop_shape)
        full = np.memmap(image_path, dtype='uint8', mode='r', shape=shape)
        if not np.array_equal(full[:, y0:y1, x0:x1], crop):
            raise ValueError('Retained crop pixels differ from full native source pixels')
        coordinate_scope = 'full native source chart; retained crop embedded without resampling'
        del full, crop
    if image_path.stat().st_size != int(np.prod(shape)):
        raise ValueError('Native image payload does not match its declared uint8 TYX shape')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_path = args.output / 'frozen_post_interpolation_baseline.npy'
    baseline = np.lib.format.open_memmap(baseline_path, mode='w+', dtype='uint8', shape=shape)
    baseline[:] = 0
    y0, x0, y1, x1 = bbox
    baseline[first:terminal + 1, y0:y1, x0:x1] = selected[first:terminal + 1]
    baseline.flush()
    plans = {mode: plan_record(baseline, mode, terminal, args.distance, args.min_radius,
                              'real-smoke/' + mode) for mode in args.crop_modes}
    bundle = resolve_local_sam_bundle(args.model)
    pins = source_pins()
    protocol = dict(schema='xta.sam_extrapolation_real_smoke/1', functional_smoke=True,
        benchmark=False, accuracy_claim=False, dynamic_retry_exercised=False,
        image_path=str(image_path), image_sha256=sha(image_path), shape_tyx=list(shape),
        coordinate_scope=coordinate_scope, crop_bbox_yx=list(bbox),
        crop_image_path=str((directory / metadata['image_path']).resolve()),
        crop_image_sha256=sha(directory / metadata['image_path']),
        metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path),
        source_video=metadata['input'], source_video_frames=metadata['input_native_frames'],
        historical_selected_file=str(selected_path), historical_selected_sha256=sha(selected_path),
        detector_observations_file=str(detector_path), detector_observations_sha256=sha(detector_path),
        detector_declared_packed_mask_sha256=detector_packed_sha,
        baseline_path=str(baseline_path.resolve()), baseline_sha256=sha(baseline_path),
        baseline_frame_half_open=[first, terminal + 1],
        baseline_recipe='Exact historical selected pixels in the declared window only; '
            'other historical detector observations are outside this artificial functional fixture scope',
        original_observations_in_scope_preserved=True, terminal_frame=terminal,
        terminal_mask_area=int(baseline[terminal].sum()), distance=args.distance,
        walk_back=0, min_radius=args.min_radius, crop_modes=args.crop_modes, plans=plans,
        model_path=str(args.model.resolve()), model_version=bundle.model_version,
        checkpoint_path=str(bundle.checkpoint_path), checkpoint_sha256=sha(bundle.checkpoint_path),
        device=args.device, gpu_lock_path=str(args.gpu_lock.resolve()),
        feature_cache_mib=args.feature_cache_mib, source_sha256=pins,
        invocation=sys.argv, predicted_mask_filters='none',
        scope_limit='One frozen original terminal per mode; no full CLI workload, blind holdout, '
            'target-system throughput or segmentation-quality claim')
    protocol['protocol_sha256'] = identity(protocol)
    write(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=str(args.output / 'protocol.json'), plans=plans)), flush=True)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text('utf-8'))
    unsigned = {key: value for key, value in protocol.items() if key != 'protocol_sha256'}
    if identity(unsigned) != protocol.get('protocol_sha256'):
        raise ValueError('Sealed functional-smoke protocol changed')
    for path_key, hash_key in (('image_path', 'image_sha256'), ('baseline_path', 'baseline_sha256'),
                              ('checkpoint_path', 'checkpoint_sha256')):
        if sha(protocol[path_key]) != protocol[hash_key]:
            raise ValueError('Sealed input/model changed: ' + path_key)
    if source_pins() != protocol['source_sha256']:
        raise ValueError('Production Python source changed; prepare a fresh protocol after source freeze')
    return protocol


@contextlib.contextmanager
def gpu_lock(path, state, task='sam-extrapolation-real-functional-smoke'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = dict(task=task, pid=os.getpid(),
                 start_time=dt.datetime.now(dt.timezone.utc).isoformat(), token=uuid.uuid4().hex)
    announced = None
    while True:
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(owner, handle)
            break
        except FileExistsError:
            holder = path.read_text('utf-8')
            if holder != announced:
                print('Waiting for GPU_LOCK: ' + holder, flush=True)
                announced = holder
            time.sleep(5)
    print('GPU_LOCK acquired', flush=True)
    try:
        yield owner
    finally:
        current = json.loads(path.read_text('utf-8')) if path.exists() else {}
        if current.get('token') == owner['token']:
            if state['contexts_settled']:
                path.unlink()
                print('GPU_LOCK released after worker/model retirement', flush=True)
            else:
                print('GPU_LOCK retained: worker/model retirement is unproven', flush=True)


def verify_mode(protocol, stats, components, baseline, destination, context, view):
    import numpy as np
    from XTA import assembly, outputs
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_extrapolation_policy import select_sam_extrapolation
    from XTA.reconciliation import evidence_role
    from XTA.reconciliation_runtime import RuntimeLayer
    from XTA.reconciliation_io import read_layer_manifest
    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
    receipt = select_sam_extrapolation(bundle)
    if len(bundle.runs) != 1 or stats['generated_runs'] != 1:
        raise AssertionError('Real smoke generated more than one original-seeded hypothesis')
    run = next(iter(bundle.runs.values()))
    expected = list(range(protocol['terminal_frame'], protocol['terminal_frame'] + protocol['distance'] + 1))
    if (list(run['expected_frames']) != expected or len(run['seed_ids']) != 1
            or run.get('held_out_ids') or run.get('edge_ids')
            or list(run['injected_frames']) != [protocol['terminal_frame']]):
        raise AssertionError('Real tracker violated the one original seed and fixed horizon contract')
    group = bundle.groups[run['group_id']]
    run_receipt = receipt['run_receipts'][run['run_id']]
    if run_receipt['stop_reason'] not in {'raw_empty', 'distance_limit'}:
        raise AssertionError('Tail stopped for a condition other than raw empty or horizon')
    raw_records = []
    with bundle.reader() as reader:
        for frame in expected:
            raw = reader.raw_mask(run['run_id'], frame)
            raw_records.append(dict(frame=frame, area=int(raw.sum()),
                sha256=hashlib.sha256(np.packbits(raw, bitorder='little').tobytes()).hexdigest()))
        model = dict(bundle.scope.get('sam_model', {}))
        sdk_runtime = dict(bundle.scope.get('sam_runtime', {}))
        if (model.get('checkpoint_sha256') != protocol['checkpoint_sha256']
                or model.get('model_version') != protocol['model_version']):
            raise AssertionError('Real worker model differs from the sealed local checkpoint')
        if (not sdk_runtime or run['runtime_receipt'].get('sam_model') != model
                or run['runtime_receipt'].get('sam_runtime') != sdk_runtime):
            raise AssertionError('Aggregate model/runtime identity differs from its actual original run')
        for tile in run.get('tile_evidence', ()):
            if tile.get('attempted') and (tile['runtime_receipt'].get('sam_model') != model
                    or tile['runtime_receipt'].get('sam_runtime') != sdk_runtime):
                raise AssertionError('Tile model/runtime identity differs from its aggregate evidence')
        if model.get('sdk_output_hole_fill_area') != 0:
            raise AssertionError('SDK hole filling changed authoritative raw observations')
        old_shape = assembly.final_source_output_shape()
        old_sink = outputs.nrrd_layer_sink()
        sink = outputs.NrrdLayerSink(nrrd_dir=destination / 'nrrd', stem='functional-smoke',
                                    output_shape_tyx=tuple(baseline.shape), max_workers=1)
        publications = []
        try:
            assembly.set_final_source_output_shape(tuple(baseline.shape))
            outputs.set_nrrd_layer_sink(sink)
            for component in components:
                ref = assembly.materialize_sam_extrapolation_view_layer(component,
                    model_name='retained-detector-fixture', view=view, source='fullframe',
                    sam_context=context, distance=protocol['distance'], walk_back=0,
                    min_radius=protocol['min_radius'], workers=1)
                sign = 1 if component['direction'] == 'forward' else -1
                layer = RuntimeLayer(ref, tuple(baseline.shape))
                digest = hashlib.sha256()
                foreground = 0
                try:
                    for frame in range(baseline.shape[0]):
                        actual = layer.read_slab(frame, frame + 1)[0] != 0
                        expected_plane = np.zeros(baseline.shape[1:], bool)
                        if sign == 1 and frame in run_receipt['effective_output_frames']:
                            y0, x0, y1, x1 = group['context_bbox_yx']
                            expected_plane[y0:y1, x0:x1] = reader.raw_mask(run['run_id'], frame)
                            expected_plane &= baseline[frame] == 0
                        np.testing.assert_array_equal(actual, expected_plane,
                            err_msg='Published source tail differs from raw mask outside frozen baseline')
                        digest.update(np.packbits(actual, bitorder='little').tobytes())
                        foreground += int(actual.sum())
                finally:
                    layer.close()
                if ref.mask_kind != 'extrapolation' or ref.interpolation_backend or ref.proposal_selection_status:
                    raise AssertionError('Tail was published with paired interpolation provenance')
                publications.append(dict(direction=component['direction'], path=str(ref.path),
                    shape_tyx=list(ref.shape), foreground=foreground, native_mask_sha256=digest.hexdigest()))
            sink.wait()
            manifest = sink.write_manifest()
        finally:
            outputs.set_nrrd_layer_sink(old_sink)
            assembly.set_final_source_output_shape(old_shape)
            sink.shutdown()
    with read_layer_manifest(manifest, workspace=destination / 'manifest-check') as layers:
        if any(evidence_role(layer.metadata) != 'extrapolation' or layer.proposal_bundle() is not None for layer in layers):
            raise AssertionError('Published manifest confused single-seed tails with paired bridges')
    adapter = run.get('runtime_receipt', {}).get('adapter_receipt', {})
    return dict(status='passed', stats=stats, evidence_path=str(bundle.directory),
        evidence_fingerprint=bundle.evidence_fingerprint, run_id=run['run_id'],
        raw_records=raw_records, run_receipt=run_receipt, publications=publications,
        nrrd_manifest=str(manifest), model=model, sdk_runtime=sdk_runtime,
        adapter_integrity=dict(seed_roundtrip_exact=adapter.get('seed_roundtrip_exact'),
            raw_observation_complete=adapter.get('raw_observation_complete')),
        baseline_unchanged=True, source_publication_matches_unfiltered_raw=True)


def run(args, *, held_gpu_lock=None, lock_state=None):
    import numpy as np
    from XTA.geometry import get_view_infos
    from XTA.sam_integration import SamInterpolationContext
    from tools.diagnose_sam_interpolation import resource_monitor
    protocol = load_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'protocol.json', protocol)
    baseline = np.load(protocol['baseline_path'], mmap_mode='r', allow_pickle=False)
    images = np.memmap(protocol['image_path'], dtype='uint8', mode='r', shape=tuple(protocol['shape_tyx']))
    view = get_view_infos(*baseline.shape, cartesian_views=('transverse',))[0]
    state = lock_state if lock_state is not None else dict(contexts_settled=True)
    if held_gpu_lock is not None:
        current = json.loads(Path(protocol['gpu_lock_path']).read_text('utf-8'))
        if current.get('token') != held_gpu_lock.get('token') or current.get('pid') != os.getpid():
            raise ValueError('A shared GPU lock must be owned by this same orchestrating process')
    report = dict(schema='xta.sam_extrapolation_real_smoke_result/1', status='running',
        functional_smoke=True, accuracy_claim=False, benchmark=False, dynamic_retry_exercised=False,
        protocol_sha256=protocol['protocol_sha256'], source_sha256_before=source_pins(),
        baseline_sha256_before=sha(protocol['baseline_path']), modes=[], invocation=sys.argv)
    os.environ['YOLO_TTA_NRRD_MEMBER_CODEC'] = 'cpu'
    try:
        lock = (gpu_lock(protocol['gpu_lock_path'], state) if held_gpu_lock is None
                else contextlib.nullcontext(held_gpu_lock))
        with lock as owner, resource_monitor(args.output / 'resources.json', protocol['device']):
            report['gpu_lock_owner'] = owner
            for mode in protocol['crop_modes']:
                destination = args.output / mode
                destination.mkdir()
                context = SamInterpolationContext(model_path=protocol['model_path'],
                    device_ids=(protocol['device'],), detector_device_ids=(),
                    temp_dir=destination / 'runtime', evidence_root=destination / 'evidence',
                    source_volume=images, source_identity=protocol['image_sha256'],
                    source_grid_shape=images.shape, crop_mode=mode,
                    bundle_identity=protocol['checkpoint_sha256'], interpolation_policy_enabled=False,
                    feature_cache_mib=protocol['feature_cache_mib'], adaptive_crop=False)
                state['contexts_settled'] = False
                try:
                    rejected, gate_stats, gate_components = context.extrapolate(baseline, view=view,
                        scope='real-smoke/' + mode + '/radius-reject',
                        work_dir=destination / 'radius-gate', distance=protocol['distance'],
                        walk_back=0, min_radius=protocol['plans'][mode]['terminal_radius'] + .5,
                        eligible_terminals=terminal_selector(protocol['terminal_frame']), workers=1)
                    if (rejected is not baseline or gate_components or not gate_stats.get('skipped')
                            or context._runtime is not None or context._leases):
                        raise AssertionError('Rejected terminal created GPU/image/model work')
                    returned, stats, components = context.extrapolate(baseline, view=view,
                        scope='real-smoke/' + mode, work_dir=destination / 'retained',
                        distance=protocol['distance'], walk_back=0, min_radius=protocol['min_radius'],
                        eligible_terminals=terminal_selector(protocol['terminal_frame']), workers=1)
                    if returned is not baseline or sha(protocol['baseline_path']) != protocol['baseline_sha256']:
                        raise AssertionError('Frozen post-interpolation baseline changed during real tracking')
                    mode_report = verify_mode(protocol, stats, components, baseline,
                                              destination, context, view)
                    mode_report['terminal_radius_gate_no_tracking'] = True
                    mode_report['radius_gate_stats'] = gate_stats
                    report['modes'].append(dict(crop_mode=mode, **mode_report))
                    print(json.dumps(dict(mode=mode, status='passed', added_voxels=stats['added_voxels'])), flush=True)
                finally:
                    try:
                        context.close()
                    finally:
                        state['contexts_settled'] = context._runtime is None and not context._leases
            report['source_sha256_after'] = source_pins()
            report['baseline_sha256_after'] = sha(protocol['baseline_path'])
            if report['source_sha256_before'] != report['source_sha256_after']:
                raise AssertionError('Production Python source changed during real-model proof')
            report.update(status='passed', gpu_used=True, worker_model_owners_retired=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      worker_model_owners_retired=state['contexts_settled'])
        write(args.output / 'result.json', report)
        raise
    write(args.output / 'result.json', report)
    return report


def check(args):
    report = json.loads(args.report.read_text('utf-8'))
    if (report.get('status') != 'passed' or not report.get('worker_model_owners_retired')
            or report['source_sha256_before'] != report['source_sha256_after']
            or report['baseline_sha256_before'] != report['baseline_sha256_after']):
        raise AssertionError('Real-model proof lacks unchanged inputs/source or completed owners')
    for mode in report['modes']:
        if not mode['source_publication_matches_unfiltered_raw']:
            raise AssertionError('Real-model publication parity failed')
    print(json.dumps(dict(status='passed', modes=[mode['crop_mode'] for mode in report['modes']],
                          accuracy_claim=False, benchmark=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    p = stages.add_parser('prepare')
    p.add_argument('--case-directory', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--full-native-images', type=Path)
    p.add_argument('--full-native-shape', type=int, nargs=3)
    p.add_argument('--first-baseline-frame', type=int, default=4)
    p.add_argument('--terminal-frame', type=int, default=18)
    p.add_argument('--distance', type=int, default=4)
    p.add_argument('--min-radius', type=float, default=3.)
    p.add_argument('--crop-modes', nargs='+', choices=('whole', 'tiled'), default=('whole', 'tiled'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpu-lock', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--feature-cache-mib', type=int, default=512)
    p.add_argument('--output', type=Path, required=True)
    p = stages.add_parser('run')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = stages.add_parser('check')
    p.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'check': check}[args.stage](args)


if __name__ == '__main__':
    main()
