"""Capture selected cleaned native detector canvases without constructing SAM.

Run with --output FRESH_DIAGNOSTIC_DIR --view EXACT_RUNTIME_NAME (repeatable),
then -- and the original TTA arguments. The output is diagnostic evidence, never
a completed TTA run. Only gray, unaugmented, untiled Azimuthal/Radial parents are
supported. New YOLO inference is not asserted to reproduce an earlier run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.config import InterpolationSettings


class _CaptureFinished(BaseException):
    """Unwind the real pipeline's failure teardown before diagnostic publication."""


class _CaptureSettings(InterpolationSettings):
    @property
    def sam_enabled(self):
        return False


def _without_sam(settings):
    return _CaptureSettings(**asdict(settings))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


class PlannerInputCapture:
    def __init__(self, output, targets, metadata):
        self.output = Path(output)
        self.targets = frozenset(targets)
        self.metadata = metadata
        self.captured = {}
        self.error = None
        self.lock = threading.Lock()

    def capture(self, volume, *, model_name, view, prepare):
        import numpy as np
        from XTA.interpolation import (INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore,
                                       write_raw_bbox_mask_store)
        from XTA.sam_interpolation import observation_snapshot_sha256

        if view.name not in self.targets:
            return
        # Serialize capture I/O: extra memory is bounded by one native plane.
        with self.lock:
            if self.error:
                raise RuntimeError('Another planner capture failed')
            try:
                if view.name in self.captured:
                    raise RuntimeError(f'Duplicate planner input: {view.name}')
                array = np.asarray(volume)
                if array.ndim != 3 or array.dtype != np.uint8 or any(n <= 0 for n in array.shape):
                    raise ValueError('Planner capture requires a nonempty uint8 frame,y,x canvas')
                if array.shape[0] != view.num_slices:
                    raise ValueError('Native canvas frame count disagrees with ViewInfo')
                encoding = None
                literal = hashlib.sha256()
                for plane in array:
                    values = set(np.unique(plane).tolist()) - {0}
                    if values and (len(values) != 1 or not values <= {1, 255}):
                        raise ValueError('Capture refuses nonbinary or mixed 1/255 detector masks')
                    if values:
                        value = next(iter(values))
                        if encoding is not None and value != encoding:
                            raise ValueError('Capture refuses mixed 1/255 detector masks')
                        encoding = value
                    literal.update(np.ascontiguousarray(plane).tobytes())
                encoding = encoding or 1
                destination = self.output / ('scope-' + hashlib.sha256(
                    f'{model_name}/{view.name}/fullframe'.encode()).hexdigest()[:20])
                destination.mkdir()
                store_path = destination / 'observations.cvol'
                write_raw_bbox_mask_store(array, store_path, workers=1,
                    format_name=INTERNAL_PACKED_CVOL_FORMAT,
                    desc=f'DIAGNOSTIC native planner input {model_name}/{view.name}')
                # Verify original literal bytes, not only foreground parity, before the marker.
                restored = hashlib.sha256()
                store = RawBBoxMaskStore.open(store_path)
                try:
                    for index in range(array.shape[0]):
                        plane = (store.decode_slice(index) != 0).astype(array.dtype)
                        plane *= encoding
                        restored.update(plane.tobytes())
                finally:
                    store.close()
                if restored.hexdigest() != literal.hexdigest():
                    raise RuntimeError('Packed native observation round trip changed literal bytes')
                record = dict(schema='xta.sam_planner_input/1', status='captured',
                    diagnostic_only=True, scientific_tta_complete=False,
                    scope=f'{model_name}/{view.name}/fullframe', view=asdict(view),
                    shape=list(array.shape), dtype=array.dtype.str, nonzero_value=encoding,
                    working_canvas_kind=('native_view_grid' if tuple(array.shape) == (
                        int(view.num_slices), int(view.src_h), int(view.src_w)) else 'detector_processing_grid'),
                    literal_bytes_sha256=literal.hexdigest(),
                    observation_snapshot_sha256=observation_snapshot_sha256(array),
                    storage_format=INTERNAL_PACKED_CVOL_FORMAT, store='observations.cvol',
                    files={str(path.relative_to(destination)): _sha256_file(path)
                           for path in sorted(store_path.rglob('*')) if path.is_file()},
                    detector_prepare=prepare, provenance=self.metadata)
                _atomic_json(destination / 'manifest.json', record)
                self.captured[view.name] = str(destination.relative_to(self.output) / 'manifest.json')
            except BaseException as error:
                self.error = f'{type(error).__name__}: {error}'
                raise
            finally:
                volume = array = plane = store = None
            if self.targets == self.captured.keys():
                raise _CaptureFinished()

    def prepare(self, **kwargs):
        try:
            return self._prepare(**kwargs)
        except _CaptureFinished:
            raise
        except BaseException as error:
            with self.lock:
                self.error = f'{type(error).__name__}: {error}'
            raise
        finally:
            kwargs['union_mm'] = kwargs['confmap_mm'] = None

    def _prepare(self, **kwargs):
        from XTA.assembly import prepare_detector_view_cpu
        from XTA.interpolation import PreparedViewResult
        from XTA.runtime import close_memmap_array_without_flush
        from XTA.view_prepare import scratch_unlink_path_for_memmap

        scores = kwargs['confmap_mm']
        owner = kwargs.get('confidence_owner')
        if owner is not None:
            if scores is not None or len(owner) != 1:
                raise RuntimeError('Confidence must have exactly one transferred owner')
            scores = owner.pop()
        view = kwargs['view']
        try:
            if kwargs.get('sam_context') is not None:
                raise RuntimeError('Diagnostic unexpectedly received a SAM context')
            prepare_detector_view_cpu(kwargs['union_mm'], scores, view,
                kwargs['min_conf'], kwargs['min_radius'], workers=kwargs['slice_workers'],
                precleaned_slice_cleanup=kwargs['precleaned_slice_cleanup'],
                hole_fill_done_on_device=kwargs['hole_fill_done_on_device'],
                slice_meta=kwargs['slice_meta'])
            self.capture(kwargs['union_mm'], model_name=kwargs['model_name'], view=view,
                prepare={key: kwargs[key] for key in ('min_conf', 'min_radius',
                    'precleaned_slice_cleanup', 'hole_fill_done_on_device', 'slice_workers',
                    'preinterpolation_layer_already_published')})
        finally:
            callback = kwargs.get('confidence_retired_callback')
            if scores is not None and callback is not None:
                lifetime_owner = scores
                seen = set()
                while getattr(lifetime_owner, 'base', None) is not None and id(lifetime_owner) not in seen:
                    seen.add(id(lifetime_owner))
                    lifetime_owner = lifetime_owner.base
                finalizer = weakref.finalize(lifetime_owner, callback,
                    str(kwargs['model_name']), str(view.name), int(scores.nbytes))
                finalizer.atexit = False
                lifetime_owner = None
            try:
                close_memmap_array_without_flush(scores, unlink_path=(
                    scratch_unlink_path_for_memmap(scores, kwargs['confmap_path'])
                    if scores is not None else None))
            finally:
                scores = None
                kwargs['union_mm'] = kwargs['confmap_mm'] = None
        # AdmittedViewPrepare owns and retires the dense mask; no global union is needed.
        return PreparedViewResult(model_name=kwargs['model_name'], view_name=view.name,
            aug_id=view.tta_aug_id, angle_deg=view.tta_angle_deg,
            native_support_mm=None, final_view_volume_mm=None, interpolation_stats=[])


def run_capture(output, targets, tta_arguments, *, pipeline=None):
    from XTA import config
    from XTA.geometry import physical_view_name
    from XTA.reconciliation_policy import resolve_reconciliation
    from XTA.sam_bridge_planning import SamPlanningLimits
    from XTA.unification.tta_manifest import (capture_tta_artifact_identities,
                                            assert_tta_artifacts_unchanged)

    arguments = list(tta_arguments)
    if arguments[:2] == ['--mode', 'tta']:
        arguments = arguments[2:]
    original = config.build_argparser().parse_args(arguments)
    models = config.resolve_backend_models(original.model)
    devices = config.resolve_backend_devices(original.device)
    settings = config.resolve_interpolation_settings(original, models, devices)
    if not original.output:
        raise ValueError('Provide the original TTA --output explicitly for disjoint-output validation')
    channel = config.resolve_channel_format(original.channel_format)
    if (not models.gpu or models.cpu or devices.cpu or not devices.gpu_devices
            or original.enable_tile or original.augmentation
            or original.augmentation_ratio != 1 or channel.kind != 'gray' or channel.channel_count != 1
            or settings.sam_crop_mode != 'whole' or settings.backend != 'sam' or not settings.sam_enabled
            or original.interpolation_distance <= 0 or original.centerline_filter_passes
            or original.capture_component_replay):
        raise ValueError('Capture supports one GPU detector, gray, whole SAM crops, positive interpolation, '
                         'and no tiles or policy augmentation')
    target_set = frozenset(targets)
    if not target_set or len(target_set) != len(targets):
        raise ValueError('Provide distinct exact runtime --view names')
    if any(not name.startswith(('azimuthal_', 'radial_')) or not name.endswith('__tta_a0')
           for name in target_set):
        raise ValueError('Capture supports only angle-zero fullframe Azimuthal/Radial runtime views')
    if set(config.resolve_save_request(original.save).options) - {'low_quality', 'nrrd', 'summary'}:
        raise ValueError('Capture supports only original low_quality, nrrd, summary output requests')
    output = Path(output).expanduser().resolve()
    protected = [Path(original.input).expanduser().resolve(),
                 Path(original.output).expanduser().resolve(), ROOT]
    protected.extend(Path(value).expanduser().resolve() for value in (models.gpu, models.sam) if value)
    if output.exists() or any(_overlaps(output, path) for path in protected):
        raise ValueError('Capture --output must be fresh and disjoint from source, models, repository, and original output')
    identities = capture_tta_artifact_identities(input_path=original.input,
        gpu_model_path=models.gpu, cpu_model_path=None)
    identities['source']['sha256'] = _sha256_file(original.input)
    reconciliation = resolve_reconciliation(original)
    overrides = dict(output=str(output / 'pipeline-diagnostic'), temp=str(output / 'runtime-temp'),
        capture_component_replay=None)
    effective = argparse.Namespace(**{**vars(original), **overrides})
    metadata = dict(launcher_version=config.SCRIPT_VERSION, original_tta_arguments=arguments,
        original_arguments=vars(original), diagnostic_overrides={**overrides, 'sam_enabled': False,
            'nrrd_layer_submissions': 'suppressed', 'scientific_output_tail': 'forbidden'},
        interpolation=asdict(settings), planning_limits=asdict(SamPlanningLimits()),
        source_identities=identities, reconciliation=asdict(reconciliation),
        environment={key: value for key, value in os.environ.items() if key.startswith('YOLO_TTA_')},
        code_sha256={str(path.relative_to(ROOT)): _sha256_file(path)
            for path in sorted((ROOT / 'XTA').rglob('*.py'))},
        capture_tool_sha256=_sha256_file(__file__),
        same_as_previous_inference=False)
    output.mkdir(parents=True)
    if reconciliation.enabled:
        (output / 'reconciliation-policy.py').write_bytes(Path(reconciliation.path).read_bytes())
        metadata['reconciliation_policy_capture'] = 'reconciliation-policy.py'
    capture = PlannerInputCapture(output, target_set, metadata)
    report = dict(schema='xta.sam_planner_capture/1', diagnostic_only=True,
        scientific_tta_complete=False, sam_executed=False, status='in_progress',
        requested_views=sorted(target_set), captures={}, provenance=metadata)
    _atomic_json(output / 'capture-manifest.json', report)
    if pipeline is None:
        from XTA import pipeline
    from XTA import sam_integration
    original_compile = pipeline.compile_physical_views
    original_expand = pipeline.expand_views_into_tta_variants
    original_scratch = pipeline.choose_scratch_dir
    scratch_paths = []

    def compile_selected(**kwargs):
        compiled = original_compile(**kwargs)
        variants = original_expand(compiled.views, kwargs['augmentation_angles'])
        selected = [view for view in variants if view.name in target_set]
        missing = target_set - {view.name for view in selected}
        if missing:
            raise ValueError(f'Unknown exact runtime capture views: {sorted(missing)}')
        physical = {physical_view_name(view) for view in selected}
        return replace(compiled, views=tuple(view for view in compiled.views if view.name in physical))

    def expand_selected(views, angles):
        return [view for view in original_expand(views, angles) if view.name in target_set]

    def choose_scratch(*args, **kwargs):
        path = original_scratch(*args, **kwargs)
        if not Path(path).resolve().is_relative_to(output / 'runtime-temp'):
            raise RuntimeError('Pipeline scratch escaped the diagnostic-owned root')
        scratch_paths.append(Path(path))
        return path

    def forbidden(*args, **kwargs):
        raise RuntimeError('Diagnostic forbids SAM construction and scientific output publication')

    failure = None
    finished = False
    try:
        with ExitStack() as stack:
            diagnostic_parser = config.build_argparser()
            stack.enter_context(patch.object(diagnostic_parser, 'parse_args', return_value=effective))
            stack.enter_context(patch.object(pipeline, 'build_argparser', return_value=diagnostic_parser))
            stack.enter_context(patch.object(pipeline, 'resolve_interpolation_settings',
                return_value=_without_sam(settings)))
            stack.enter_context(patch.object(pipeline, 'prepare_view_volume_after_fullframe', capture.prepare))
            stack.enter_context(patch.object(pipeline, 'compile_physical_views', compile_selected))
            stack.enter_context(patch.object(pipeline, 'expand_views_into_tta_variants', expand_selected))
            stack.enter_context(patch.object(pipeline, 'choose_scratch_dir', choose_scratch))
            stack.enter_context(patch.object(sam_integration, 'SamInterpolationContext', forbidden))
            stack.enter_context(patch.object(pipeline.NrrdLayerSink, 'submit_layer', return_value=None))
            for name in ('assemble_final_union_after_view_union', '_finalize_physical_view_volume_group',
                         'build_tta_run_manifest', '_publish_complete_tta_manifest'):
                stack.enter_context(patch.object(pipeline, name, forbidden))
            pipeline.main()
    except _CaptureFinished:
        finished = True
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        (output / 'failure-traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        # main() has already joined its owners. Remove only the scratch we allocated.
        gc.collect()
        for path in scratch_paths:
            try:
                if path.exists():
                    pipeline._cleanup_tta_selected_run_scratch(temp_dir=path,
                        out_dir=output / 'pipeline-diagnostic')
            except BaseException as error:
                failure = f'Scratch cleanup failed: {type(error).__name__}: {error}'
                with (output / 'failure-traceback.txt').open('a', encoding='utf-8') as stream:
                    stream.write('\nScratch cleanup failure:\n' + traceback.format_exc())
        try:
            assert_tta_artifacts_unchanged(identities)
            for relative, expected in metadata['code_sha256'].items():
                if _sha256_file(ROOT / relative) != expected:
                    raise RuntimeError(f'Running source changed: {relative}')
            if _sha256_file(__file__) != metadata['capture_tool_sha256']:
                raise RuntimeError('Capture tool changed during execution')
            if reconciliation.enabled:
                if (_sha256_file(reconciliation.path) != reconciliation.sha256
                        or _sha256_file(output / 'reconciliation-policy.py') != reconciliation.sha256):
                    raise RuntimeError('Reconciliation policy changed during execution')
        except BaseException as error:
            failure = f'Source identity changed: {type(error).__name__}: {error}'
            with (output / 'failure-traceback.txt').open('a', encoding='utf-8') as stream:
                stream.write('\nSource identity revalidation failure:\n' + traceback.format_exc())
        missing = sorted(target_set - capture.captured.keys())
        report.update(captures=capture.captured, missing_views=missing,
            status=('capture_complete' if finished and not missing and not capture.error and not failure
                    else 'capture_failed' if capture.error or failure else 'capture_incomplete'),
            error=capture.error or failure,
            failure_traceback=('failure-traceback.txt'
                if (output / 'failure-traceback.txt').exists() else None),
            note='SAM was suppressed. This diagnostic is not a scientific TTA completion record.')
        _atomic_json(output / 'capture-manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--view', required=True, action='append', help='Exact runtime view name; repeat for each target')
    parser.add_argument('tta_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.tta_arguments or args.tta_arguments[0] != '--':
        parser.error('Place -- before the original TTA arguments')
    report = run_capture(args.output, args.view, args.tta_arguments[1:])
    print(f"DIAGNOSTIC {report['status']}: {len(report['captures'])}/{len(report['requested_views'])} "
          f"native planner inputs; SAM was not executed. Manifest: {args.output / 'capture-manifest.json'}")
    return 0 if report['status'] == 'capture_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
