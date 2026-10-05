"""Prepare/run a real SAM resident-idle GPU stage handoff functional proof.

The fixed historical terminal fixture is scoped evidence, not an accuracy
reference. Normal SAM and Spherical stage admission remain in control. This
tool observes real leases and rejects projection CPU fallback; it never resets
the coordinator while a model is resident. No benchmark claim is made.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import smoke_sam_extrapolation as shared
from tools import smoke_sam_image_cohorts as cohorts


def json_receipt(value):
    """Preserve immutable evidence values while normalizing JSON containers."""
    if isinstance(value, Mapping):
        return {key: json_receipt(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_receipt(item) for item in value]
    return value


def write_receipt(path, value):
    shared.write(path, json_receipt(value))


def pins():
    extra = (Path(__file__).resolve(), Path(cohorts.__file__).resolve(),
             REPO / 'tests/reference_backends/spherical.py')
    return {**shared.source_pins(), **{str(p.relative_to(REPO)): shared.sha(p) for p in extra}}


def initialize_standalone_coordinator(bp, device):
    """Publish the real drained-detector lifecycle for this detector-free probe."""
    before = bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
    if (before['stage_leases'] or before['inference_inflight'] or before.get('resident_owners')
            or before['provisional_stage_devices'] or before['pending_inference_backlog']
            or before['inference_asset_retirement_pending']):
        raise AssertionError('Standalone setup cannot clear another queued, active or resident GPU owner')
    bp._configure_main_process_gpu_stage_workers((device,))
    # Configure declares available logical devices and marks inference priority.
    # This frontend starts no detector workers/assets, so publish their normal
    # terminal lifecycle before the first SAM predictor requests admission.
    bp._set_main_process_gpu_pending_inference(False)
    bp._set_main_process_gpu_inference_priority_active(False)
    bp._set_main_process_gpu_asset_retirement_pending(False)
    after = bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()
    if (after['stage_leases'] or after['inference_inflight'] or after.get('resident_owners')
            or after['provisional_stage_devices'] or after['inference_priority_active']
            or after['pending_inference_backlog'] or after['inference_asset_retirement_pending']):
        raise AssertionError('Detector-free standalone setup did not publish clean drained ownership')
    return dict(detector_workers_started=False, detector_assets_started=False,
                published_normal_detector_drained_lifecycle=True, coordinator_before=before,
                coordinator_after=after)


def projection_fixture():
    import numpy as np
    from XTA.spherical_geometry import build_spherical_view_infos
    from tests.reference_backends.spherical import project_spherical_block
    shape = (9, 11, 13)
    view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.2,
                                     patch_size=15, tilted_views=())[0]
    source = np.random.default_rng(345).integers(0, 2,
        size=(view.num_slices, 15, 15), dtype=np.uint8)
    expected = project_spherical_block(source, view, np.asarray(view.spherical_radii),
        np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0])
    return source, view, expected


def prepare(args):
    import hashlib
    import numpy as np
    cohorts.prepare(args)
    path = args.output / 'protocol.json'
    protocol = json.loads(path.read_text('utf-8'))
    source, view, expected = projection_fixture()
    np.save(args.output / 'projection_input.npy', source, allow_pickle=False)
    np.save(args.output / 'projection_expected.npy', expected, allow_pickle=False)
    protocol.update(schema='xta.sam_gpu_stage_handoff_smoke/1', crop_mode='whole',
        sequence=[dict(name='back_first', terminal_frame=4, direction=-1),
                  dict(name='back_repeat', terminal_frame=4, direction=-1),
                  dict(name='forward', terminal_frame=18, direction=1)],
        arms=['baseline', 'interposed'], total_tracker_frames=30,
        feature_cache_mib=args.feature_cache_mib,
        projection=dict(family='spherical', native_shape_tyx=list(expected.shape),
            patch_size=15, min_radius=.2, view_name=view.name, seed=345,
            input_path=str((args.output / 'projection_input.npy').resolve()),
            input_sha256=shared.sha(args.output / 'projection_input.npy'),
            expected_path=str((args.output / 'projection_expected.npy').resolve()),
            expected_sha256=shared.sha(args.output / 'projection_expected.npy'),
            expected_pixels_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            foreground=int(expected.sum()), normal_stage_admission=True,
            normal_live_vram_admission=True, cpu_fallback_permitted=False,
            oracle='tests.reference_backends.spherical.project_spherical_block'),
        claim='Bounded real-model lease handoff, feature continuity, exact raw/score/export parity; '
              'no production-quality, throughput, or timing claim', source_sha256=pins(), invocation=sys.argv)
    from XTA.geometry import get_view_infos
    from XTA.sam_extrapolation import prepare_sam_extrapolation_pass
    baseline = np.load(protocol['baseline_path'], mmap_mode='r', allow_pickle=False)
    native_view = get_view_infos(*protocol['source_shape_tyx'], cartesian_views=('transverse',))[0]
    for item in protocol['sequence']:
        selected = lambda observation, direction: (
            observation.frame_index == item['terminal_frame'] and direction == item['direction'])
        prepared = prepare_sam_extrapolation_pass(baseline, view=native_view,
            scope='gpu-handoff/fixed-terminals', distance=4, walk_back=0, min_radius=3.,
            crop_mode='whole', eligible_terminals=selected)
        if len(prepared.runs) != 1 or len(prepared.groups) != 1:
            raise AssertionError('Predeclared handoff terminal must yield exactly one original-seed job')
        run, group = prepared.runs[0], prepared.groups[0]
        item['binding'] = dict(seed_ids=list(run.seed_ids),
            injected_frames=[run.expected_frames[0]], expected_frames=list(run.expected_frames),
            context_bbox_yx=list(group.context_bbox_yx))
    protocol['protocol_sha256'] = shared.identity({k: v for k, v in protocol.items() if k != 'protocol_sha256'})
    write_receipt(path, protocol)
    print(json.dumps(dict(protocol=str(path), status='prepared', benchmark=False,
                          tracker_frames=30, projection_shape=list(expected.shape))), flush=True)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text('utf-8'))
    if shared.identity({k: v for k, v in protocol.items() if k != 'protocol_sha256'}) != protocol['protocol_sha256']:
        raise ValueError('Sealed GPU handoff protocol changed')
    for key in ('source_image', 'baseline', 'checkpoint', 'selected_source', 'cases_metadata'):
        if shared.sha(protocol[key + '_path']) != protocol[key + '_sha256']:
            raise ValueError('Sealed GPU handoff input/model changed: ' + key)
    for key in ('input', 'expected'):
        if shared.sha(protocol['projection'][key + '_path']) != protocol['projection'][key + '_sha256']:
            raise ValueError('Sealed synthetic projection fixture changed')
    if pins() != protocol['source_sha256']:
        raise ValueError('Source changed; prepare again after global source freeze')
    return protocol


class LeaseObserver:
    """Transparent observation of actual coordinator leases and worker submits."""
    def __init__(self, bp):
        self.bp = bp
        self.phase = 'not_started'
        self.events = []

    def snapshot(self):
        return self.bp._MAIN_PROCESS_GPU_STAGE_COORDINATOR.snapshot()

    def event(self, kind, **details):
        self.events.append(dict(kind=kind, phase=self.phase, monotonic=time.monotonic(),
                                coordinator=self.snapshot(), **details))

    def observe(self, stack):
        from XTA.lta_workers import LtaWorkerPool
        init = self.bp._MainProcessGpuStageLease.__init__
        release = self.bp._MainProcessGpuStageLease.release
        promote = self.bp._MainProcessGpuStageLease.promote_residency
        submit = LtaWorkerPool.submit

        def actual_init(lease, *values, **keywords):
            init(lease, *values, **keywords)
            self.event('lease_acquired', purpose=lease.purpose, device=lease.device_index)

        def actual_release(lease):
            was_released = lease._released
            release(lease)
            if not was_released:
                self.event('lease_released', purpose=lease.purpose, device=lease.device_index)

        def actual_promote(lease):
            resident = promote(lease)
            self.event('startup_stage_promoted_to_idle_residency', purpose=lease.purpose,
                       device=lease.device_index)
            return resident

        def actual_submit(pool, task, *values, **keywords):
            device = int(keywords.get('execution_device_id', 0))
            snapshot = self.snapshot()
            owner = snapshot['stage_leases'].get(device)
            if not owner or 'sam' not in owner.lower():
                raise AssertionError('Actual SAM worker submit lacks exclusive active-stage protection')
            self.event('sam_worker_submit', device=device, work_id=task.work_id, purpose=owner)
            returned = submit(pool, task, *values, **keywords)
            import torch
            probe = self.bp._try_acquire_specific_main_process_gpu_stage(
                torch, device, 'Spherical source projection active SAM exclusion probe')
            self.event('active_sam_projection_admission_probe', device=device,
                       work_id=task.work_id, admitted=probe is not None)
            if probe is not None:
                probe.release()
                raise AssertionError('Foreign Spherical stage admitted while an actual SAM job owns compute')
            return returned

        stack.enter_context(mock.patch.object(self.bp._MainProcessGpuStageLease, '__init__', actual_init))
        stack.enter_context(mock.patch.object(self.bp._MainProcessGpuStageLease, 'release', actual_release))
        stack.enter_context(mock.patch.object(self.bp._MainProcessGpuStageLease, 'promote_residency', actual_promote))
        stack.enter_context(mock.patch.object(LtaWorkerPool, 'submit', actual_submit))


def resident_idle(context, observer, device):
    runtime = context._runtime
    if runtime is None or runtime._pool is None or runtime._closed:
        raise AssertionError('SAM persistent runtime/model pool was not retained')
    pool = runtime._pool
    if not pool.is_alive(device) or not pool.pids:
        raise AssertionError('SAM resident worker did not remain alive')
    snapshot = observer.snapshot()
    if (snapshot['stage_leases'] or snapshot['provisional_stage_devices']
            or snapshot['inference_inflight'] or runtime._iteration_active):
        raise AssertionError('Completed SAM iterator still monopolizes an active GPU stage')
    owners = snapshot.get('resident_owners', {})
    owner = owners.get(device)
    if (len(owners) != 1 or owner is None or owner['lendable'] is not True
            or owner['quarantined'] is not False or not runtime.startup_cuda_quiescent):
        raise AssertionError('Resident SAM allocator lacks a proven lendable, nonquarantined CUDA owner')
    observer.event('sam_resident_idle', worker_pids=dict(pool.pids))
    return runtime, dict(pool.pids)


def project_interposer(protocol, destination, observer, state):
    import hashlib
    import numpy as np
    from XTA import spherical_projection as sp
    from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnsafeFailure
    from XTA.interpolation import (IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
                                   INTERNAL_PACKED_CVOL_FORMAT)
    source, view, expected = projection_fixture()
    np.testing.assert_array_equal(source, np.load(protocol['projection']['input_path'], allow_pickle=False))
    np.testing.assert_array_equal(expected, np.load(protocol['projection']['expected_path'], allow_pickle=False))
    destination.mkdir(parents=True, exist_ok=False)
    actual_stages = []
    admit = sp._try_spherical_cuda_stage

    def normal_admission(*values, **keywords):
        stage = admit(*values, **keywords)
        if stage is None:
            raise AssertionError('Resident-idle real Spherical CUDA admission declined; CPU fallback forbidden')
        if type(stage.projector) is not SphericalCudaProjector:
            raise AssertionError('Interposed stage is not the actual Spherical CUDA projector')
        actual_stages.append(stage)
        observer.event('spherical_cuda_admitted', device=stage.device_index,
            required_device_bytes=stage.projector.required_device_bytes,
            reserve_bytes=stage.projector.reserve_bytes)
        return stage

    writer = IncrementalRawBBoxMaskStoreWriter(shape=expected.shape, store_dir=destination / 'projection.cvol',
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='SAM resident-idle real Spherical handoff')
    state['projection_settled'] = False
    try:
        with mock.patch.object(sp, '_try_spherical_cuda_stage', normal_admission):
            sp.backproject_spherical_volume_to_volume(source, view, destination / 'unused.dat',
                'SAM resident-idle interposer', workers=1, out_shape_tyx=expected.shape,
                sink_only=True, projection_block_callback=writer)
        metadata = writer.finalize()
        if len(actual_stages) != 1 or actual_stages[0].lease is not None:
            raise AssertionError('Spherical stage did not retire its real exclusive lease')
        with closing(RawBBoxMaskStore.open(writer.store_dir)) as store:
            actual = np.stack([store.decode_slice(z) for z in range(expected.shape[0])])
        np.testing.assert_array_equal(actual, expected)
        stage = actual_stages[0]
        projector = stage.projector
        if not projector._closed or projector.payload_d2h_bytes <= 0:
            raise AssertionError('Spherical projector did not execute/fence/retire actual CUDA publication')
        result = dict(actual_projector=type(projector).__module__ + '.' + type(projector).__name__,
            device=stage.device_index, cpu_fallback=False, exact_numpy_oracle=True,
            pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), foreground=int(actual.sum()),
            required_device_bytes=projector.required_device_bytes, reserve_bytes=projector.reserve_bytes,
            source_h2d_bytes=projector.source_h2d_bytes, payload_d2h_bytes=projector.payload_d2h_bytes,
            store_path=str(writer.store_dir), metadata=metadata, stage_retired=True)
        state['projection_settled'] = True
        return result
    except SphericalCudaProjectionUnsafeFailure:
        # The production exception retains the unfenced projector/lease. Keep
        # the shared GPU_LOCK as well; never claim safe model/stage retirement.
        state['projection_settled'] = False
        raise
    except BaseException:
        state['projection_settled'] = all(stage.lease is None for stage in actual_stages)
        raise


def sdk_receipts(bundle):
    rows = []
    for run in bundle.runs.values():
        children = [t['runtime_receipt'] for t in run.get('tile_evidence', ()) if t.get('attempted')]
        for receipt in children or [run['runtime_receipt']]:
            proof = receipt.get('cuda_quiescence', {})
            if (proof.get('synchronized') is not True or proof.get('worker_local_device') != 0
                    or proof.get('run_id') != receipt['run_id']
                    or proof.get('execution_device_id') != receipt['dispatch']['execution_device_id']
                    or receipt['sam_runtime'].get('startup_cuda_quiescence', {}).get('synchronized') is not True):
                raise AssertionError('Real SAM startup/job CUDA-quiescence proof is incomplete')
            rows.append({key: receipt.get(key) for key in ('run_id', 'dispatch', 'sam_model', 'sam_runtime',
                'feature_cache_before', 'feature_cache_after', 'adapter_receipt', 'image_cache', 'cuda_quiescence')})
    return rows


def run(args):
    import numpy as np
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
    state = dict(contexts_settled=True, projection_settled=True)
    observer = LeaseObserver(bp)
    result = dict(schema='xta.sam_gpu_stage_handoff_smoke_result/1', status='running', benchmark=False,
        accuracy_claim=False, protocol_sha256=protocol['protocol_sha256'], source_pins_before=pins(), arms=[])
    try:
        with shared.gpu_lock(protocol['gpu_lock_path'], state, 'sam-resident-idle-real-gpu-handoff'), \
                resource_monitor(args.output / 'resources.json', protocol['device']), ExitStack() as stack:
            if (observer.snapshot()['stage_leases'] or observer.snapshot()['inference_inflight']
                    or observer.snapshot().get('resident_owners')):
                raise AssertionError('Functional process entered with another GPU stage owner')
            result['standalone_detector_lifecycle'] = initialize_standalone_coordinator(bp, protocol['device'])
            observer.observe(stack)
            stack.enter_context(mock.patch.dict(os.environ, {
                'YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES': str(protocol['mode_records']['whole']['control_cap_bytes']),
                'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT': '1',
                'CUPY_CACHE_DIR': str(args.output / 'cupy-cache'),
                'NUMBA_CACHE_DIR': str(args.output / 'numba-cache')}))
            for arm in protocol['arms']:
                destination = args.output / arm
                destination.mkdir()
                context = SamInterpolationContext(model_path=protocol['model_path'],
                    device_ids=(protocol['device'],), detector_device_ids=(), source_volume=source,
                    source_identity=protocol['source_image_sha256'], source_grid_shape=source.shape,
                    temp_dir=destination / 'runtime', evidence_root=destination / 'evidence',
                    feature_cache_mib=protocol['feature_cache_mib'], crop_mode='whole',
                    interpolation_policy_enabled=False, adaptive_crop=False)
                state['contexts_settled'] = False
                arm_record = dict(arm=arm, passes=[])
                resident = worker_pids = None
                try:
                    for item in protocol['sequence']:
                        observer.phase = arm + '/' + item['name']
                        selected = lambda observation, direction: (
                            observation.frame_index == item['terminal_frame'] and direction == item['direction'])
                        pass_dir = destination / item['name']
                        pass_dir.mkdir()
                        returned, stats, components = context.extrapolate(baseline, view=view,
                            scope='gpu-handoff/fixed-terminals', work_dir=pass_dir / 'evidence',
                            distance=4, walk_back=0, min_radius=3., eligible_terminals=selected, workers=1)
                        if returned is not baseline:
                            raise AssertionError('Handoff changed the frozen post-interpolation baseline')
                        runtime, pids_now = resident_idle(context, observer, protocol['device'])
                        if resident is not None and (resident is not runtime or pids_now != worker_pids):
                            raise AssertionError('GPU handoff restarted the SAM runtime or model worker')
                        resident, worker_pids = runtime, pids_now
                        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
                        raw, exports = cohorts.capture(bundle, baseline, protocol, pass_dir, context, view, components)
                        if len(raw) != 1 or any(next(iter(raw.values()))[key] != value
                                              for key, value in item['binding'].items()):
                            raise AssertionError('Actual SAM seed, complete interval or crop differs from sealed planning')
                        row = dict(name=item['name'], raw_records=raw, source_exports=exports,
                                   sdk_receipts=sdk_receipts(bundle), worker_pids=pids_now, stats=stats)
                        arm_record['passes'].append(row)
                        if item['name'] == 'back_first' and arm == 'interposed':
                            observer.phase = arm + '/resident_idle_projection'
                            arm_record['projection'] = project_interposer(protocol,
                                destination / 'interposed_projection', observer, state)
                            runtime, after = resident_idle(context, observer, protocol['device'])
                            if runtime is not resident or after != worker_pids:
                                raise AssertionError('Projection retired the resident SAM worker/model')
                    first, repeat = arm_record['passes'][:2]
                    if first['raw_records'] != repeat['raw_records']:
                        raise AssertionError('Repeated frozen SAM seed changed exact raw masks/scores/intervals')
                    original_sdk, resumed_sdk = first['sdk_receipts'][0], repeat['sdk_receipts'][0]
                    if resumed_sdk['feature_cache_before'] != original_sdk['feature_cache_after']:
                        raise AssertionError('Idle handoff changed the resident feature cache before SAM resumed')
                    audit = resumed_sdk['adapter_receipt']['tracker_feature_preparation']
                    if audit.get('shared_feature_cache_hits', 0) <= 0:
                        raise AssertionError('Repeated original seed did not reuse retained SAM features')
                    arm_record.update(status='passed', persistent_runtime=True,
                                      repeat_feature_cache_hits=audit['shared_feature_cache_hits'])
                    result['arms'].append(arm_record)
                    write_receipt(destination / 'result.json', arm_record)
                    print(json.dumps(dict(arm=arm, status='passed', repeat_feature_cache_hits=
                                         audit['shared_feature_cache_hits'])), flush=True)
                finally:
                    observer.phase = arm + '/model_retirement'
                    context.close()
                    owner_snapshot = observer.snapshot()
                    state['contexts_settled'] = (context._runtime is None and not context._leases
                                                  and state['projection_settled']
                                                  and not owner_snapshot['stage_leases']
                                                  and not owner_snapshot.get('resident_owners'))
                    if not state['contexts_settled']:
                        raise AssertionError('Real GPU handoff ownership/model retirement is unproven')
            control, treatment = result['arms']
            for left, right in zip(control['passes'], treatment['passes']):
                if left['raw_records'] != right['raw_records']:
                    raise AssertionError('Interposed CUDA stage changed exact SAM masks/scores/seeds/intervals')
                for direction, export in left['source_exports'].items():
                    if export['mask_sha256'] != right['source_exports'][direction]['mask_sha256']:
                        raise AssertionError('Interposed CUDA stage changed native source export support')
            submits = [e for e in observer.events if e['kind'] == 'sam_worker_submit']
            starts = [e for e in observer.events if e['kind'] == 'lease_acquired' and 'sam' in e['purpose'].lower()]
            if len(submits) != 6 or len(starts) < 6:
                raise AssertionError('Lease receipts omit actual protected SAM startup/inference events')
            probes = [e for e in observer.events if e['kind'] == 'active_sam_projection_admission_probe']
            if len(probes) != 6 or any(e['admitted'] for e in probes):
                raise AssertionError('Actual active SAM jobs did not exclude foreign projection admission')
            result['source_pins_after'] = pins()
            if result['source_pins_after'] != result['source_pins_before']:
                raise AssertionError('Source changed during real GPU handoff qualification')
            for key in ('source_image', 'baseline', 'selected_source', 'cases_metadata'):
                if shared.sha(protocol[key + '_path']) != protocol[key + '_sha256']:
                    raise AssertionError('A canonical original input or frozen baseline changed')
            result.update(status='passed', exact_paired_raw_masks_scores_seeds_intervals_exports=True,
                          source_inputs_survived=True, owners_retired=True, coordinator_final=observer.snapshot())
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      owners_retired=state['contexts_settled'], projection_settled=state['projection_settled'])
        raise
    finally:
        result['lease_events'] = observer.events
        write_receipt(args.output / 'result.json', result)


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
    p.add_argument('--feature-cache-mib', type=int, default=512)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=prepare)
    r = stages.add_parser('run')
    r.add_argument('--protocol', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.set_defaults(function=run)
    args = parser.parse_args()
    if getattr(args, 'feature_cache_mib', 1) <= 0:
        parser.error('Feature continuity proof requires a positive cache budget')
    args.function(args)


if __name__ == '__main__':
    main()
