"""Bounded real SAM cache/flat-versus-family functional parity qualification.

CPU preparation decodes a declared local-video continuation and authenticates
its overlap with canonical historical pixels. Frozen post-interpolation masks
are independently injected in complete 31/32-frame sessions. Four arms vary
only explicit feature-cache capacity and exact-crop dispatch. No biological
quality, timing, or multi-GPU speed claim is made.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import smoke_sam_extrapolation as shared
from tools.smoke_sam_gpu_handoff import initialize_standalone_coordinator, write_receipt


def pins():
    extra = (Path(__file__).resolve(), REPO / 'tools/smoke_sam_gpu_handoff.py',
             REPO / 'tools/smoke_sam_image_cohorts.py', REPO / 'tests/reference_backends/spherical.py')
    return {**shared.source_pins(), **{str(p.relative_to(REPO)): shared.sha(p) for p in extra}}


def eligible(observation, direction):
    return observation.frame_index == 18 and direction == 1


def pixel_hash(mask):
    import numpy as np
    return hashlib.sha256(np.packbits(mask != 0, bitorder='little').tobytes()).hexdigest()


def prepare(args):
    import cv2
    import numpy as np
    from XTA.geometry import get_view_infos, render_categorical_frame_on_grid
    from XTA.sam_view_geometry import sam_native_transform_record
    from XTA.sam_extrapolation import prepare_sam_extrapolation_pass
    from XTA.lta_sam import resolve_local_sam_bundle
    case_dir = args.case_directory.resolve()
    cases_path = case_dir / 'cases.json'
    metadata = json.loads(cases_path.read_text('utf-8'))
    selected_path = case_dir / 'region02_case02' / 'sam_selected.npy'
    selected = np.load(selected_path, mmap_mode='r', allow_pickle=False)
    original_shape = (23, 3064, 3024)
    if tuple(selected.shape) != (23, 1008, 1008):
        raise ValueError('The declared case02 historical baseline is required')
    if args.canonical_images.stat().st_size != int(np.prod(original_shape)):
        raise ValueError('Canonical native image fixture differs from its declared shape')
    original = np.memmap(args.canonical_images, dtype='uint8', mode='r', shape=original_shape)
    args.output.mkdir(parents=True, exist_ok=False)
    images_path = args.output / 'native_images.uint8.dat'
    source_shape = (49, 3064, 3024)
    images = np.memmap(images_path, dtype='uint8', mode='w+', shape=source_shape)
    capture = cv2.VideoCapture(str(args.video))
    positions = []
    try:
        if (not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != source_shape[2]
                or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != source_shape[1]
                or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) < args.first_video_frame + source_shape[0]):
            raise ValueError('Local video cannot supply the complete declared native continuation')
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, args.first_video_frame):
            raise ValueError('Local video seek failed')
        for index in range(source_shape[0]):
            ok, decoded = capture.read()
            if not ok or tuple(decoded.shape[:2]) != source_shape[1:]:
                raise ValueError('Local video decode omitted a declared frame')
            if not (np.array_equal(decoded[:, :, 0], decoded[:, :, 1])
                    and np.array_equal(decoded[:, :, 0], decoded[:, :, 2])):
                raise ValueError('Declared 8bit_Y source is not exact replicated grayscale')
            positions.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if positions[-1] != args.first_video_frame + index + 1:
                raise ValueError('Local video decoder frame address differs from declared ordering')
            images[index] = decoded[:, :, 0]
            if index < 23:
                np.testing.assert_array_equal(images[index], original[index])
        images.flush()
    finally:
        capture.release()
    native_baseline = np.zeros(source_shape, np.uint8)
    x0, y0, x1, y1 = metadata['source_crop_xyxy']
    native_baseline[4:19, y0:y1, x0:x1] = selected[4:19]
    view = get_view_infos(*source_shape, cartesian_views=('transverse',))[0]
    working_shape = (49, 2048, 2048)
    transform = sam_native_transform_record(view, working_shape, source_shape,
                                            source_processing_shape_tyx=source_shape)
    baseline_path = args.output / 'working_baseline.npy'
    baseline = np.lib.format.open_memmap(baseline_path, mode='w+', dtype='uint8', shape=working_shape)
    baseline[:] = 0
    for frame in range(4, 19):
        baseline[frame] = render_categorical_frame_on_grid(native_baseline, view, frame,
            M_src_to_out=np.asarray(transform['M_native_to_canvas'], np.float32),
            M_out_to_src=np.asarray(transform['M_canvas_to_native'], np.float32),
            output_height=2048, output_width=2048)
    baseline.flush()
    prepared = prepare_sam_extrapolation_pass(baseline, view=view,
        scope='feature-dispatch/fixed-original-baseline', distance=30, walk_back=1,
        min_radius=3., crop_mode='whole', eligible_terminals=eligible)
    if len(prepared.groups) != 1 or len(prepared.runs) != 2:
        raise AssertionError('Declared terminal/inward baseline must yield exactly two independent original seeds')
    group = prepared.groups[0]
    y0, x0, y1, x1 = group.context_bbox_yx
    bindings = {}
    for run in prepared.runs:
        seed_frame = run.expected_frames[0]
        bindings[run.run_id] = dict(seed_ids=list(run.seed_ids), injected_frames=[seed_frame],
            expected_frames=list(run.expected_frames), context_bbox_yx=list(group.context_bbox_yx),
            seed_pixels_sha256=pixel_hash(baseline[seed_frame, y0:y1, x0:x1]))
    if sorted(len(row['expected_frames']) for row in bindings.values()) != [31, 32]:
        raise AssertionError('Complete original-seeded sessions must be precisely31 and32 frames')
    model = resolve_local_sam_bundle(args.model)
    images_bytes = sum((box[2]-box[0])*(box[3]-box[1]) for box in prepared.frame_crop_bounds.values())
    protocol = dict(schema='xta.sam_feature_dispatch_functional_smoke/1', functional_smoke=True,
        benchmark=False, accuracy_claim=False, multi_gpu_speed_claim=False,
        video_path=str(args.video.resolve()), video_sha256=shared.sha(args.video),
        video_frame_indices=list(range(args.first_video_frame, args.first_video_frame + 49)),
        decoder='OpenCV exact replicated BGR grayscale, checked sequential frame positions',
        canonical_images_path=str(args.canonical_images.resolve()), canonical_images_sha256=shared.sha(args.canonical_images),
        canonical_overlap_all_23_frames_exact=True, synthetic_continuation=False,
        source_image_path=str(images_path.resolve()), source_image_sha256=shared.sha(images_path),
        source_shape_tyx=list(source_shape), working_shape_tyx=list(working_shape),
        selected_source_path=str(selected_path), selected_source_sha256=shared.sha(selected_path),
        cases_metadata_path=str(cases_path), cases_metadata_sha256=shared.sha(cases_path),
        baseline_path=str(baseline_path.resolve()), baseline_sha256=shared.sha(baseline_path),
        canonical_transform=transform,
        baseline_recipe='Historical selected original post-interpolation masksf4..18 embedded native, '
            'then existing canonical categorical nearest/threshold render to2048²; seedsf18/f17 frozen independently',
        scope='feature-dispatch/fixed-original-baseline', terminal_frame=18, direction=1,
        distance=30, walk_back=1, min_radius=3., adaptive_crop=False, crop_mode='whole',
        exact_original_run_bindings=bindings, image_cache_cap_bytes=images_bytes + 1,
        arms=[dict(cache_mib=cache, exact_crop_family_dispatch=family)
              for cache in (512, 1024) for family in (False, True)],
        tracker_jobs_per_arm=2, tracker_frames_per_arm=63, total_tracker_frames=252,
        model_path=str(args.model.resolve()), checkpoint_path=str(model.checkpoint_path),
        checkpoint_sha256=shared.sha(model.checkpoint_path), model_version=model.model_version,
        gpu_lock_path=str(args.gpu_lock.resolve()), device=args.device, source_sha256=pins(), invocation=sys.argv,
        claim='Functional parity/cache counters on a scoped real-video continuation; no biological-quality, '
              'timing or multi-GPU performance claim')
    protocol['protocol_sha256'] = shared.identity(protocol)
    write_receipt(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(status='prepared', protocol=str(args.output / 'protocol.json'),
        arms=protocol['arms'], session_frames=[31, 32], canonical_overlap_exact=True,
        synthetic_continuation=False)), flush=True)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text('utf-8'))
    if shared.identity({k: v for k, v in protocol.items() if k != 'protocol_sha256'}) != protocol['protocol_sha256']:
        raise ValueError('Sealed feature/dispatch protocol changed')
    for key in ('video', 'canonical_images', 'source_image', 'baseline', 'selected_source', 'cases_metadata', 'checkpoint'):
        if shared.sha(protocol[key + '_path']) != protocol[key + '_sha256']:
            raise ValueError('Sealed feature/dispatch input/model changed: ' + key)
    if pins() != protocol['source_sha256']:
        raise ValueError('Production source changed; prepare again after global freeze')
    return protocol


def capture(bundle, baseline, protocol, context, view, components):
    import numpy as np
    from XTA import assembly
    from XTA.reconciliation_runtime import RuntimeLayer
    records = {}
    sdk_rows = []
    with bundle.reader() as reader:
        for rid, run in bundle.runs.items():
            if rid not in protocol['exact_original_run_bindings']:
                raise AssertionError('Dispatch introduced a different original run identity')
            binding = protocol['exact_original_run_bindings'][rid]
            group = bundle.groups[run['group_id']]
            seed = reader.group_mask(run['group_id'], 'endpoint:' + run['seed_ids'][0])
            if pixel_hash(seed) != binding['seed_pixels_sha256']:
                raise AssertionError('Dispatch changed an independently frozen original baseline seed')
            receipt = run['runtime_receipt']
            if (receipt['sam_model']['checkpoint_sha256'] != protocol['checkpoint_sha256']
                    or receipt['adapter_receipt'].get('seed_roundtrip_exact') is not True
                    or receipt['adapter_receipt'].get('raw_observation_complete') is not True):
                raise AssertionError('Actual SAM SDK seed/full-interval/model provenance is incomplete')
            record = dict(seed_ids=list(run['seed_ids']), injected_frames=list(run['injected_frames']),
                expected_frames=list(run['expected_frames']), context_bbox_yx=list(group['context_bbox_yx']),
                seed_pixels_sha256=pixel_hash(seed))
            if record != binding:
                raise AssertionError('Dispatch changed a sealed original seed, crop or complete interval')
            record.update(raw_hashes={str(frame): pixel_hash(reader.raw_mask(rid, frame))
                                      for frame in run['expected_frames']},
                          tracker_scores=None if run['tracker_scores'] is None else dict(run['tracker_scores']))
            records[rid] = record
            sdk_rows.append(dict(run_id=rid, dispatch=receipt.get('dispatch'),
                feature_cache_before=receipt.get('feature_cache_before'),
                feature_cache_after=receipt.get('feature_cache_after'),
                feature_preparation=receipt['adapter_receipt']['tracker_feature_preparation'],
                cuda_quiescence=receipt.get('cuda_quiescence')))
    if set(records) != set(protocol['exact_original_run_bindings']):
        raise AssertionError('Dispatch omitted a complete independently seeded run')
    old_shape = assembly.final_source_output_shape()
    exports = {}
    try:
        assembly.set_final_source_output_shape(tuple(protocol['source_shape_tyx']))
        for component in components:
            ref = assembly.materialize_sam_extrapolation_view_layer(component,
                model_name='declared-feature-dispatch-fixture', view=view, source='fullframe', sam_context=context,
                distance=30, walk_back=1, min_radius=3., workers=1)
            layer = RuntimeLayer(ref, tuple(protocol['source_shape_tyx']))
            digest = hashlib.sha256()
            count = 0
            try:
                for frame in range(protocol['source_shape_tyx'][0]):
                    mask = layer.read_slab(frame, frame + 1)[0] != 0
                    digest.update(np.packbits(mask, bitorder='little').tobytes())
                    count += int(mask.sum())
            finally:
                layer.close()
            exports[component['direction']] = dict(mask_sha256=digest.hexdigest(), foreground=count, path=str(ref.path))
    finally:
        assembly.set_final_source_output_shape(old_shape)
    return records, sdk_rows, exports


def run(args):
    import numpy as np
    import torch
    from XTA import backprojection as bp
    from XTA.geometry import get_view_infos
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_integration import SamInterpolationContext
    from tools.diagnose_sam_interpolation import resource_monitor
    protocol = load_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write_receipt(args.output / 'protocol.json', protocol)
    baseline = np.load(protocol['baseline_path'], mmap_mode='r', allow_pickle=False)
    source = np.memmap(protocol['source_image_path'], mode='r', dtype='uint8',
                       shape=tuple(protocol['source_shape_tyx']))
    view = get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    state = dict(contexts_settled=True)
    result = dict(schema='xta.sam_feature_dispatch_functional_result/1', status='running', benchmark=False,
        accuracy_claim=False, multi_gpu_speed_claim=False, protocol_sha256=protocol['protocol_sha256'],
        source_pins_before=pins(), arms=[])
    try:
        with shared.gpu_lock(protocol['gpu_lock_path'], state, 'sam-real-cache-and-family-functional-parity'), \
                resource_monitor(args.output / 'resources.json', protocol['device']), ExitStack() as stack:
            result['standalone_detector_lifecycle'] = initialize_standalone_coordinator(bp, protocol['device'])
            stack.enter_context(mock.patch.dict(os.environ, {
                'YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES': str(protocol['image_cache_cap_bytes'])}))
            control = None
            for arm in protocol['arms']:
                label = 'cache' + str(arm['cache_mib']) + '_' + ('family' if arm['exact_crop_family_dispatch'] else 'flat')
                destination = args.output / label
                destination.mkdir()
                context = SamInterpolationContext(model_path=protocol['model_path'],
                    device_ids=(protocol['device'],), detector_device_ids=(), source_volume=source,
                    source_identity=protocol['source_image_sha256'], source_grid_shape=source.shape,
                    temp_dir=destination / 'runtime', evidence_root=destination / 'evidence',
                    feature_cache_mib=arm['cache_mib'], crop_mode='whole',
                    interpolation_policy_enabled=False, adaptive_crop=False)
                state['contexts_settled'] = False
                try:
                    returned, stats, components = context.extrapolate(baseline, view=view,
                        scope=protocol['scope'], work_dir=destination / 'retained',
                        distance=30, walk_back=1, min_radius=3., eligible_terminals=eligible, workers=1,
                        exact_crop_family_dispatch=arm['exact_crop_family_dispatch'])
                    if returned is not baseline:
                        raise AssertionError('Dispatch changed the frozen post-interpolation baseline')
                    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
                    raw, sdk, exports = capture(bundle, baseline, protocol, context, view, components)
                    batches = stats.get('sam_exact_crop_family_batches', ())
                    expected_mode = 'exact_crop_family_fifo' if arm['exact_crop_family_dispatch'] else 'flat'
                    if (stats.get('sam_exact_crop_family_dispatch_requested') != arm['exact_crop_family_dispatch']
                            or len(batches) != 1 or batches[0]['mode'] != expected_mode
                            or batches[0]['job_count'] != 2 or batches[0]['exact_crop_family_count'] != 1
                            or set(batches[0]['dispatch_run_ids']) != set(protocol['exact_original_run_bindings'])):
                        raise AssertionError('Actual dispatch path differs from explicit flat/family original-job protocol')
                    family_ids = {row['dispatch'].get('family_id') for row in sdk}
                    if ((arm['exact_crop_family_dispatch'] and (len(family_ids) != 1 or None in family_ids))
                            or (not arm['exact_crop_family_dispatch'] and family_ids != {None})):
                        raise AssertionError('Actual per-run family attribution differs from explicit dispatch arm')
                    dispatch_stats = dict(context._runtime.dispatch_stats)
                    if arm['exact_crop_family_dispatch']:
                        assigned = dispatch_stats.get('family_assigned')
                        if (assigned != batches[0]['exact_crop_family_count']
                                or dispatch_stats.get('family_completed') != assigned):
                            raise AssertionError('Successful exhausted dispatch did not retire every assigned family')
                    if control is None:
                        control = dict(raw=raw, exports=exports)
                    elif raw != control['raw'] or any(exports[d]['mask_sha256'] != old['mask_sha256']
                                                       for d, old in control['exports'].items()):
                        raise AssertionError('Cache/dispatch changed exact masks/scores/seeds/intervals/source exports')
                    after = sdk[-1]['feature_cache_after']
                    if after is None or after['max_bytes'] > arm['cache_mib'] * 1024**2:
                        raise AssertionError('Actual worker feature-cache capacity exceeds explicit byte allowance')
                    if after['live_bytes'] > after['max_bytes']:
                        raise AssertionError('Actual live feature tensors exceed the worker cache byte cap')
                    free_bytes, total_bytes = torch.cuda.mem_get_info(protocol['device'])
                    row = dict(**arm, status='passed', stats=stats, original_raw_records=raw,
                        sdk_cache_receipts=sdk, source_exports=exports,
                        dispatch_stats=dispatch_stats,
                        live_vram_after=dict(free_bytes=free_bytes, total_bytes=total_bytes),
                        cache_counters={key: after[key] for key in ('hits', 'misses', 'admissions', 'evictions',
                            'max_bytes', 'entries', 'resident_bytes', 'live_bytes', 'peak_live_bytes',
                            'headroom_bytes', 'rejected_headroom', 'rejected_active_bytes')})
                    result['arms'].append(row)
                    write_receipt(destination / 'result.json', row)
                    print(json.dumps(dict(arm=label, status='passed', cache_counters=row['cache_counters'])), flush=True)
                finally:
                    context.close()
                    owner_snapshot = bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
                    state['contexts_settled'] = (context._runtime is None and not context._leases
                        and not owner_snapshot['stage_leases'] and not owner_snapshot.get('resident_owners'))
                    if not state['contexts_settled']:
                        raise AssertionError('Model/worker ownership retirement is unproven')
            result['source_pins_after'] = pins()
            if result['source_pins_after'] != result['source_pins_before']:
                raise AssertionError('Source changed during functional cache/dispatch qualification')
            for key in ('video', 'source_image', 'baseline', 'selected_source', 'canonical_images', 'cases_metadata'):
                if shared.sha(protocol[key + '_path']) != protocol[key + '_sha256']:
                    raise AssertionError('A canonical input or frozen continuation changed')
            result.update(status='passed', exact_masks_scores_original_seeds_intervals_exports=True,
                          source_inputs_survived=True, owners_retired=state['contexts_settled'])
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      owners_retired=state['contexts_settled'])
        raise
    finally:
        write_receipt(args.output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    p = stages.add_parser('prepare')
    p.add_argument('--case-directory', type=Path, required=True)
    p.add_argument('--canonical-images', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--first-video-frame', type=int, default=644)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpu-lock', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=prepare)
    r = stages.add_parser('run')
    r.add_argument('--protocol', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
