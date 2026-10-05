"""Paired real-model functional check of owned gray-cache cohort lifetimes.

Preparation uses a declared historical baseline window and the existing
categorical TTA renderer. No quality or performance claim is made. The same
frozen hypotheses run once with a whole-scope cap and once with a forced split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import smoke_sam_extrapolation as shared


def pins():
    return {**shared.source_pins(), str(Path(__file__).resolve().relative_to(REPO)): shared.sha(__file__)}


def eligible(observation, direction):
    return (observation.frame_index, direction) in {(4, -1), (18, 1)}


def prepare(args):
    import hashlib
    import numpy as np
    from XTA.geometry import get_view_infos, render_categorical_frame_on_grid
    from XTA.sam_view_geometry import sam_native_transform_record
    from XTA.sam_extrapolation import prepare_sam_extrapolation_pass, plan_sam_extrapolation_image_cohorts
    from XTA.lta_sam import resolve_local_sam_bundle
    case_dir = args.case_directory.resolve()
    metadata = json.loads((case_dir / 'cases.json').read_text('utf-8'))
    selected_path = case_dir / 'region02_case02' / 'sam_selected.npy'
    selected = np.load(selected_path, mmap_mode='r', allow_pickle=False)
    source_shape = tuple(args.source_shape)
    if source_shape[0] != 23 or tuple(selected.shape) != (23, 1008, 1008):
        raise ValueError('This bounded fixture requires the declared 23-frame native case')
    if args.images.stat().st_size != int(np.prod(source_shape)):
        raise ValueError('Source image payload differs from declared uint8 TYX shape')
    x0, y0, x1, y1 = metadata['source_crop_xyxy']
    native = np.zeros(source_shape, np.uint8)
    native[4:19, y0:y1, x0:x1] = selected[4:19]
    shape = (23, 2048, 2048)
    view = get_view_infos(*source_shape, cartesian_views=('transverse',))[0]
    transform = sam_native_transform_record(view, shape, source_shape,
                                            source_processing_shape_tyx=source_shape)
    forward = np.asarray(transform['M_native_to_canvas'], np.float32)
    inverse = np.asarray(transform['M_canvas_to_native'], np.float32)
    if np.array_equal(forward, np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)):
        raise ValueError('Owned-cache probe must use a nonidentity canonical image transform')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_path = args.output / 'working_baseline.npy'
    baseline = np.lib.format.open_memmap(baseline_path, mode='w+', dtype='uint8', shape=shape)
    baseline[:] = 0
    for frame in range(4, 19):
        baseline[frame] = render_categorical_frame_on_grid(native, view, frame,
            M_src_to_out=forward, M_out_to_src=inverse, output_height=2048, output_width=2048)
    baseline.flush()
    records = {}
    for mode in ('whole', 'tiled'):
        prepared = prepare_sam_extrapolation_pass(baseline, view=view,
            scope='owned-cache-cohort-smoke/' + mode, distance=4, walk_back=0,
            min_radius=3., crop_mode=mode, eligible_terminals=eligible)
        if len(prepared.groups) != 2 or len(prepared.runs) != 2:
            raise ValueError('Working baseline must retain exactly the two predeclared terminal directions')
        groups = prepared.groups
        if set(groups[0].frame_indices) & set(groups[1].frame_indices):
            raise ValueError('Declared complete groups must have disjoint image frame intervals')
        individual = [len(group.frame_indices) * (group.context_bbox_yx[2] - group.context_bbox_yx[0])
                      * (group.context_bbox_yx[3] - group.context_bbox_yx[1]) for group in groups]
        total = sum((box[2] - box[0]) * (box[3] - box[1]) for box in prepared.frame_crop_bounds.values())
        cap = (max(individual) + total) // 2
        split = plan_sam_extrapolation_image_cohorts(prepared, cap)
        control = plan_sam_extrapolation_image_cohorts(prepared, total + 1)
        if len(split) != 2 or len(control) != 1:
            raise AssertionError('Test cache cap does not force precisely two complete-group cohorts')
        records[mode] = dict(group_payload_bytes=individual, total_payload_bytes=total,
            forced_cap_bytes=cap, control_cap_bytes=total + 1,
            frame_intervals=[list(group.frame_indices) for group in groups],
            contexts=[list(group.context_bbox_yx) for group in groups],
            tracker_jobs=len(prepared.tracker_jobs) if mode == 'tiled' else len(prepared.runs))
    model = resolve_local_sam_bundle(args.model)
    protocol = dict(schema='xta.sam_owned_image_cohort_smoke/1', functional_smoke=True,
        benchmark=False, accuracy_claim=False, source_image_path=str(args.images.resolve()),
        source_image_sha256=shared.sha(args.images), source_shape_tyx=list(source_shape),
        selected_source_path=str(selected_path), selected_source_sha256=shared.sha(selected_path),
        cases_metadata_path=str(case_dir / 'cases.json'), cases_metadata_sha256=shared.sha(case_dir / 'cases.json'),
        baseline_path=str(baseline_path.resolve()), baseline_sha256=shared.sha(baseline_path),
        working_shape_tyx=list(shape), canonical_transform=transform,
        baseline_recipe='Preserve exact historical selected f4..18 crop masks in native chart, then existing '
            'render_categorical_frame_on_grid nearest/threshold transformation to2048-square; artificial bounded fixture',
        known_terminals=[[4, -1], [18, 1]], distance=4, walk_back=0, min_radius=3., adaptive_crop=False,
        mode_records=records, model_path=str(args.model.resolve()), checkpoint_path=str(model.checkpoint_path),
        checkpoint_sha256=shared.sha(model.checkpoint_path), model_version=model.model_version,
        gpu_lock_path=str(args.gpu_lock.resolve()), device=args.device, source_sha256=pins(), invocation=sys.argv)
    protocol['protocol_sha256'] = shared.identity(protocol)
    shared.write(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=str(args.output / 'protocol.json'), cohorts=records)), flush=True)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text('utf-8'))
    if shared.identity({key: value for key, value in protocol.items() if key != 'protocol_sha256'}) != protocol['protocol_sha256']:
        raise ValueError('Sealed image-cohort protocol changed')
    for path_key, hash_key in (('source_image_path', 'source_image_sha256'), ('baseline_path', 'baseline_sha256'),
                              ('checkpoint_path', 'checkpoint_sha256')):
        if shared.sha(protocol[path_key]) != protocol[hash_key]:
            raise ValueError('Sealed cohort input/model changed')
    if pins() != protocol['source_sha256']:
        raise ValueError('Source changed; prepare again after final source freeze')
    return protocol


def capture(bundle, baseline, protocol, destination, context, view, components):
    import hashlib
    import numpy as np
    from XTA import assembly
    from XTA.sam_extrapolation_policy import select_sam_extrapolation
    from XTA.reconciliation_runtime import RuntimeLayer
    receipt = select_sam_extrapolation(bundle)
    records = {}
    with bundle.reader() as reader:
        for rid, run in bundle.runs.items():
            group = bundle.groups[run['group_id']]
            key = str(run['terminal_frame']) + '_' + run['direction']
            if len(run['seed_ids']) != 1 or len(run['expected_frames']) != 5:
                raise AssertionError('Cohort staging changed an original seed or complete interval')
            seed = reader.group_mask(run['group_id'], 'endpoint:' + run['seed_ids'][0])
            y0, x0, y1, x1 = group['context_bbox_yx']
            np.testing.assert_array_equal(seed, baseline[run['injected_frames'][0], y0:y1, x0:x1] != 0)
            parent = run['runtime_receipt']
            if parent.get('sam_model') != bundle.scope.get('sam_model') or parent.get('sam_runtime') != bundle.scope.get('sam_runtime'):
                raise AssertionError('Cohort evidence model/runtime identity differs from actual run')
            children = [tile['runtime_receipt'] for tile in run.get('tile_evidence', ()) if tile.get('attempted')]
            sdk_receipts = children or [parent]
            for sdk in sdk_receipts:
                if (sdk['sam_model']['checkpoint_sha256'] != protocol['checkpoint_sha256']
                        or sdk['adapter_receipt'].get('seed_roundtrip_exact') is not True
                        or sdk['adapter_receipt'].get('raw_observation_complete') is not True
                        or sdk.get('image_cache_lifetime', {}).get('gray_mapping_retired_after_render') is not True):
                    raise AssertionError('Real SDK seed/raw/gray mapping lifetime proof is incomplete')
            records[key] = dict(seed_ids=list(run['seed_ids']), injected_frames=list(run['injected_frames']),
                expected_frames=list(run['expected_frames']), context_bbox_yx=list(group['context_bbox_yx']),
                parent_scores=None if run['tracker_scores'] is None else dict(run['tracker_scores']),
                tile_scores={tile['tile_id']: None if tile['tracker_scores'] is None else dict(tile['tracker_scores'])
                    for tile in run.get('tile_evidence', ()) if tile.get('attempted')},
                tile_raw_hashes={tile['tile_id']: {str(frame): hashlib.sha256(np.packbits(
                    reader.tile_raw_mask(rid, tile['tile_id'], frame), bitorder='little').tobytes()).hexdigest()
                    for frame in run['expected_frames']} for tile in run.get('tile_evidence', ())
                    if tile.get('attempted')},
                raw_hashes={str(frame): hashlib.sha256(
                    np.packbits(reader.raw_mask(rid, frame), bitorder='little').tobytes()).hexdigest()
                    for frame in run['expected_frames']}, prefix=receipt['run_receipts'][rid]['effective_output_frames'],
                raw_sdk_gray_mappings_retired=True)
    exports = {}
    old_shape = assembly.final_source_output_shape()
    try:
        assembly.set_final_source_output_shape(tuple(protocol['source_shape_tyx']))
        for component in components:
            ref = assembly.materialize_sam_extrapolation_view_layer(component,
                model_name='declared-cohort-fixture', view=view, source='fullframe', sam_context=context,
                distance=4, walk_back=0, min_radius=3., workers=1)
            layer = RuntimeLayer(ref, tuple(protocol['source_shape_tyx']))
            digest = hashlib.sha256(); count = 0
            try:
                for frame in range(protocol['source_shape_tyx'][0]):
                    plane = layer.read_slab(frame, frame + 1)[0] != 0
                    digest.update(np.packbits(plane, bitorder='little').tobytes()); count += int(plane.sum())
            finally:
                layer.close()
            exports[component['direction']] = dict(mask_sha256=digest.hexdigest(), foreground=count,
                                                  path=str(ref.path), source_shape_tyx=protocol['source_shape_tyx'])
    finally:
        assembly.set_final_source_output_shape(old_shape)
    return records, exports


def run(args):
    import numpy as np
    from XTA.geometry import get_view_infos
    from XTA.sam_integration import SamInterpolationContext
    from XTA.sam_evidence import SamEvidenceBundle
    from tools.diagnose_sam_interpolation import resource_monitor
    protocol = load_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    shared.write(args.output / 'protocol.json', protocol)
    baseline = np.load(protocol['baseline_path'], mmap_mode='r', allow_pickle=False)
    source = np.memmap(protocol['source_image_path'], dtype='uint8', mode='r', shape=tuple(protocol['source_shape_tyx']))
    view = get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    previous = os.environ.get('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES')
    state = dict(contexts_settled=True)
    result = dict(schema='xta.sam_owned_image_cohort_smoke_result/1', status='running',
        functional_smoke=True, benchmark=False, accuracy_claim=False,
        protocol_sha256=protocol['protocol_sha256'], source_pins_before=pins(), arms=[])
    try:
        with shared.gpu_lock(protocol['gpu_lock_path'], state, 'sam-owned-gray-cache-cohort-functional-smoke'), \
                resource_monitor(args.output / 'resources.json', protocol['device']):
            for mode in ('whole', 'tiled'):
                controls = {}
                for arm in ('control', 'forced'):
                    cap = protocol['mode_records'][mode][arm + '_cap_bytes']
                    os.environ['YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES'] = str(cap)
                    destination = args.output / mode / arm
                    destination.mkdir(parents=True)
                    context = SamInterpolationContext(model_path=protocol['model_path'],
                        device_ids=(protocol['device'],), detector_device_ids=(), source_volume=source,
                        source_identity=protocol['source_image_sha256'], source_grid_shape=source.shape,
                        temp_dir=destination / 'runtime', evidence_root=destination / 'evidence',
                        crop_mode=mode, interpolation_policy_enabled=False, adaptive_crop=False)
                    state['contexts_settled'] = False
                    try:
                        returned, stats, components = context.extrapolate(baseline, view=view,
                            scope='owned-cache-cohort-smoke/' + mode, work_dir=destination / 'retained',
                            distance=4, walk_back=0, min_radius=3., eligible_terminals=eligible, workers=1)
                        if returned is not baseline or context.exact_backing_reuses:
                            raise AssertionError('Probe accidentally used borrowed native fast path or changed baseline')
                        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
                        records, exports = capture(bundle, baseline, protocol, destination, context, view, components)
                        lifetime = context.image_cache_lifetime_snapshot()
                        retirements = list(context.image_cohort_retirement_receipts)
                        if arm == 'forced':
                            if (len(stats['sam_image_cache_cohorts']['cohorts']) != 2
                                    or lifetime['owned_cohort_current_bytes'] != 0
                                    or lifetime['owned_cohort_peak_bytes'] > cap
                                    or lifetime['retirement_unproven_count'] != 0
                                    or len(retirements) != 2
                                    or any(r['status'] != 'retired' or not r['workers_finished']
                                           or not r['gray_mappings_retired']
                                           or not r['model_and_feature_cache_retained']
                                           or Path(r['path']).exists() for r in retirements)):
                                raise AssertionError('Owned cohort file/map retirement or bounded persistent-model proof failed')
                            if records != controls['records'] or exports.keys() != controls['exports'].keys():
                                raise AssertionError('Cohort split changed raw masks, scores, original seed or interval identities')
                            for direction, export in exports.items():
                                if export['mask_sha256'] != controls['exports'][direction]['mask_sha256']:
                                    raise AssertionError('Cohort split changed exported native source support')
                        else:
                            if len(stats['sam_image_cache_cohorts']['cohorts']) != 1 or lifetime['protected_owned_cache_bytes'] <= 0:
                                raise AssertionError('Control did not create an owned whole-scope gray cache')
                            controls = dict(records=records, exports=exports)
                        arm_result = dict(mode=mode, arm=arm, status='passed', cap_bytes=cap,
                            stats=stats, raw_records=records, source_exports=exports,
                            lifetime=lifetime, cohort_retirements=retirements,
                            exact_backing_reuses=context.exact_backing_reuses,
                            one_persistent_runtime_per_context=True)
                        result['arms'].append(arm_result)
                        shared.write(destination / 'result.json', arm_result)
                        print(json.dumps(dict(mode=mode, arm=arm, status='passed', cohorts=len(
                            stats['sam_image_cache_cohorts']['cohorts']))), flush=True)
                    finally:
                        try:
                            context.close()
                        finally:
                            state['contexts_settled'] = context._runtime is None and not context._leases
            result['source_pins_after'] = pins()
            if result['source_pins_before'] != result['source_pins_after']:
                raise AssertionError('Source changed during the real paired cohort proof')
            if shared.sha(protocol['source_image_path']) != protocol['source_image_sha256'] or shared.sha(
                    protocol['baseline_path']) != protocol['baseline_sha256']:
                raise AssertionError('Borrowed native image input or frozen working baseline changed')
            result.update(status='passed', source_input_survived=True, inputs_unchanged=True,
                          owners_retired=state['contexts_settled'])
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      owners_retired=state['contexts_settled'])
        shared.write(args.output / 'result.json', result)
        raise
    finally:
        if previous is None:
            os.environ.pop('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', None)
        else:
            os.environ['YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES'] = previous
    shared.write(args.output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    p = stages.add_parser('prepare')
    p.add_argument('--case-directory', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--source-shape', type=int, nargs=3, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpu-lock', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p = stages.add_parser('run')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
