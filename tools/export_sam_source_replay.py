"""Export an existing SAM selection through the production source/NRRD route.

No video, detector, SAM inference or dense observation volume is required.
The first supported route is retained canonical Transverse full-frame evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import __version__, assembly, outputs
from XTA.geometry import ViewInfo
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT
from XTA.sam_evidence import SamEvidenceBundle, iter_selected_native_crops, native_output_shape_tyx
from XTA.artifact_archive import read_artifact


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def cpu_export_environment():
    requested = {'YOLO_TTA_NRRD_GPU_MIRROR_TEE':'0',
                 'YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN':'0',
                 'YOLO_TTA_NRRD_LAYER_ZSHARDS':'1'}
    previous = {key:os.environ.get(key) for key in requested}
    os.environ.update(requested)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_native_direction(crops, destination, shape, *, memory_bytes):
    """Spool packed contributors, then OR one bounded native slice at a time."""
    destination = Path(destination)
    spool_path = destination.parent / f'{destination.name}.contributors.bin'
    by_frame = defaultdict(list)
    peak_raster_bytes = 0
    shape = tuple(map(int, shape))
    if destination.exists():
        raise FileExistsError('Native replay backing must be fresh')
    if 4*math.prod(shape[1:]) > int(memory_bytes):
        raise ValueError('Replay memory budget cannot fit one native slice and crop workspaces')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with spool_path.open('xb') as spool:
        for _group, _stored_frame, frame, bbox, mask in crops:
            if not np.any(mask):
                continue
            y0, x0, y1, x1 = map(int, bbox)
            if not (0 <= int(frame) < shape[0] and 0 <= y0 < y1 <= shape[1]
                    and 0 <= x0 < x1 <= shape[2] and mask.shape == (y1-y0, x1-x0)):
                raise ValueError('Selected native crop differs from its declared frame/canvas')
            rows = np.flatnonzero(np.any(mask, axis=1))
            a0, a1 = int(rows[0]), int(rows[-1])+1
            columns = np.flatnonzero(np.any(mask[a0:a1], axis=0))
            b0, b1 = int(columns[0]), int(columns[-1])+1
            tight = np.ascontiguousarray(mask[a0:a1, b0:b1], dtype=np.bool_)
            packed = np.packbits(tight.ravel(), bitorder='little')
            offset = spool.tell()
            spool.write(memoryview(packed).cast('B'))
            by_frame[int(frame)].append((y0+a0, x0+b0, y0+a1, x0+b1, offset, packed.size))
            peak_raster_bytes = max(peak_raster_bytes, int(mask.nbytes+tight.nbytes+packed.nbytes))
    writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=destination,
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='Retained selected SAM directional replay',
        extra_meta={'replay_source':'immutable_selected_native_crops'})
    try:
        with spool_path.open('rb') as spool:
            previous = 0
            for frame, records in sorted(by_frame.items()):
                if frame > previous:
                    writer.consume_empty_range(previous, frame-previous)
                y0, x0 = min(r[0] for r in records), min(r[1] for r in records)
                y1, x1 = max(r[2] for r in records), max(r[3] for r in records)
                union = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
                for a0, b0, a1, b1, offset, size in records:
                    spool.seek(offset)
                    raw = spool.read(size)
                    mask = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little',
                        count=(a1-a0)*(b1-b0)).reshape(a1-a0, b1-b0)
                    union[a0-y0:a1-y0, b0-x0:b1-x0] |= mask
                    peak_raster_bytes = max(peak_raster_bytes, int(union.nbytes+mask.nbytes+len(raw)))
                writer.consume_sparse_slice(frame, y0, y1, x0, x1, union)
                previous = frame+1
            if previous < shape[0]:
                writer.consume_empty_range(previous, shape[0]-previous)
        stats = writer.finalize()
        stats['replay_peak_local_raster_bytes'] = peak_raster_bytes
        stats['replay_packed_contributor_bytes'] = spool_path.stat().st_size
        return stats
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    finally:
        spool_path.unlink(missing_ok=True)


def reference_scope(reference_run, bundle):
    reference_run = Path(reference_run)
    manifest_path = next((reference_run/'nrrd').glob('*_nrrd_manifest.json'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    matches = [layer for layer in manifest['layers'] if layer.get('interpolation_backend') == 'sam'
        and layer['view_name'] == bundle.scope['view_name'] and layer['source'] == 'fullframe']
    if not matches:
        raise ValueError('Reference output has no SAM layer for this evidence scope')
    layer = matches[0]
    transform = layer['native_transform']
    if (transform['view_family'] != 'orthogonal' or transform['view_name'] != 'transverse'
            or layer.get('tile_config_id')):
        raise ValueError('Source replay currently requires canonical full-frame Transverse evidence')
    source_shape = tuple(map(int, manifest['output_shape_tyx']))
    native_shape = tuple(native_output_shape_tyx(bundle))
    if (source_shape != tuple(transform['source_shape_tyx'])
            or native_shape != tuple(transform['native_shape_tyx'])):
        raise ValueError('Reference source/native shapes differ from retained evidence')
    accepted_fields = {item.name for item in fields(ViewInfo)}
    recipe = {key:value for key,value in transform['sampler_recipe'].items() if key in accepted_fields}
    recipe.update(name=transform['runtime_view_name'], physical_view_name='transverse',
        tta_aug_id=layer['tta_aug_id'], tta_angle_deg=float(layer['tta_angle_deg']))
    view = ViewInfo(**recipe)
    detector = next(item for item in manifest['layers'] if item.get('mask_kind') == 'yolo'
        and item['view_name'] == layer['view_name'] and item['source'] == layer['source']
        and item.get('tile_config_id', '') == layer.get('tile_config_id', ''))
    return manifest_path, manifest, layer, detector, view, source_shape


def append_reference_detector(manifest_path, source_dir, detector):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = Path(source_dir)/detector['filename']
    destination = manifest_path.parent/detector['filename']
    shutil.copy2(source, destination)
    manifest['layers'].insert(0, dict(detector, replay_reference=True))
    manifest['layer_count'] = len(manifest['layers'])
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return {'source':str(source.resolve()), 'sha256':file_sha256(destination)}


def annotate_replay_manifest(path, provenance, selection_copy):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    manifest['replay'] = dict(provenance, selection_receipt=Path(os.path.relpath(
        selection_copy, Path(path).parent)).as_posix())
    for layer in manifest['layers']:
        if layer.get('interpolation_backend') == 'sam':
            layer['replay'] = dict(provenance)
            layer['description'] = ('Fixed-proposal replay from retained SAM observations; '
                'no detector or SAM inference was rerun. '+layer['description'])
    Path(path).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def export_source_replay(evidence, selection_path, reference_run, output, *, downbin='0.20', memory_mib=256):
    if not math.isfinite(float(memory_mib)) or float(memory_mib) <= 0:
        raise ValueError('Replay memory_mib must be finite and positive')
    budget = int(float(memory_mib)*1024**2)
    bundle = SamEvidenceBundle.open(evidence)
    selection_path = Path(selection_path).resolve()
    selection_bytes = read_artifact(selection_path)
    selection = json.loads(selection_bytes)
    if (selection.get('evidence_fingerprint') != bundle.evidence_fingerprint
            or not selection.get('policy_hash')):
        raise ValueError('Selection receipt does not match the retained proposal evidence')
    run_ids = list(selection['selected_run_ids'])
    if len(run_ids) != len(set(run_ids)) or set(run_ids)-set(bundle.runs):
        raise ValueError('Selection receipt references duplicate or unknown SAM owners')
    manifest_path, _manifest, old_layer, detector, view, source_shape = reference_scope(reference_run, bundle)
    run_manifest_path = Path(reference_run)/'manifest.json'
    generation_manifest = (json.loads(run_manifest_path.read_text(encoding='utf-8'))
                           if run_manifest_path.is_file() else {})
    launcher = generation_manifest.get('launcher', {})
    policy_implementation = selection.get('policy_implementation_sha256')
    current_policy_hash = file_sha256(Path(__file__).resolve().parents[1]/'XTA/sam_policy.py')
    provenance = dict(status='fixed_proposal_source_replay',
        original_generation_run=Path(reference_run).name,
        original_generation_pipeline_version=launcher.get('pipeline_version', launcher.get('version')),
        export_pipeline_version=__version__,
        selection_pipeline_version=__version__ if policy_implementation == current_policy_hash else None,
        selection_policy_implementation_sha256=policy_implementation,
        selection_branch_implementation_sha256=selection.get('branch_selection', {}).get('implementation_sha256'),
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        proposal_evidence_fingerprint=bundle.evidence_fingerprint,
        fresh_detector_run=False, fresh_sam_run=False)
    native_shape = tuple(native_output_shape_tyx(bundle))
    specs, _warnings = outputs.resolve_low_quality_downbin_specs([downbin], True, source_shape)
    if any(math.prod(spec.output_shape_t_y_x)+4*math.prod(native_shape[1:]) > budget for spec in specs):
        raise ValueError('Replay memory budget cannot fit one CPU downbin mirror and native crop workspaces')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Source replay output directory must be fresh')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ('.'+output.name+'.source-replay-'+uuid.uuid4().hex)
    staging.mkdir()
    prior_shape, prior_sink = assembly.final_source_output_shape(), outputs.nrrd_layer_sink()
    sink = None
    entries = []
    try:
        (staging/'selection.json').write_bytes(selection_bytes)
        with cpu_export_environment():
            stem = manifest_path.name.removesuffix('_nrrd_manifest.json')
            sink = outputs.NrrdLayerSink(nrrd_dir=staging/'nrrd', stem=stem,
                output_shape_tyx=source_shape, max_workers=1, low_quality_specs=specs,
                low_quality_root=staging/'low_quality')
            outputs.set_nrrd_layer_sink(sink)
            assembly.set_final_source_output_shape(source_shape)
            context = SimpleNamespace(detector_identity=old_layer['seed_detector_identity'],
                bundle_identity=old_layer['sam_bundle_identity'],
                source_volume=SimpleNamespace(shape=tuple(old_layer['native_transform']['source_processing_shape_tyx'])))
            passes = sorted({int(run['pass_index']) for run in bundle.runs.values()}) or [int(bundle.scope.get('pass_index', 1))]
            with bundle.reader(max_cache_bytes=min(32*1024**2, budget//8)) as reader:
                for pass_index in passes:
                    for direction in ('forward', 'backward'):
                        path = staging/'native'/f'sam_bridge_pass{pass_index:02d}_{direction}.cvol'
                        stats = write_native_direction(iter_selected_native_crops(reader, selection,
                            direction=direction, pass_index=pass_index), path, native_shape, memory_bytes=budget)
                        owners = sorted(key for key in run_ids if bundle.runs[key]['direction'] == direction
                            and int(bundle.runs[key]['pass_index']) == pass_index)
                        groups = sorted({bundle.runs[key]['group_id'] for key in owners})
                        roots = sorted({str(identifier) for key in owners for identifier in
                            (*bundle.runs[key]['seed_ids'], *bundle.runs[key]['held_out_ids'])})
                        entry = dict(direction=direction, path=str(path), voxel_count=int(stats['foreground_voxels']),
                            policy_hash=selection['policy_hash'], evidence_path=str(bundle.directory.resolve()),
                            run_ids=owners, group_ids=groups, observation_roots=roots,
                            topology_connectivity=selection.get('resolved_policy', {}).get('connectivity', 26),
                            connection_status='selected_native_replay_not_assessed_after_source_restore')
                        ref = assembly.materialize_sam_directional_view_layer(entry,
                            model_name=old_layer['model_name'], view=view, source='fullframe',
                            pass_index=pass_index, sam_context=context)
                        sink.wait()
                        entries.append(dict(direction=direction, pass_index=pass_index,
                            native_path=str(path.relative_to(staging)), native_stats=stats,
                            public_shape_tyx=list(ref.shape), native_transform=ref.native_transform))
            sink.wait()
            exported_manifest = sink.write_manifest()
            sink.shutdown()
            sink = None
        detector_receipts = [append_reference_detector(exported_manifest, manifest_path.parent, detector)]
        annotate_replay_manifest(exported_manifest, provenance, staging/'selection.json')
        for spec in specs:
            candidates = list((Path(reference_run)/'low_quality').glob('*/nrrd/*_nrrd_manifest.json'))
            match = next((path for path in candidates if tuple(json.loads(path.read_text(encoding='utf-8'))['output_shape_tyx']) == tuple(spec.output_shape_t_y_x)), None)
            if match is None:
                raise ValueError('Reference run lacks the requested low-quality detector geometry')
            low_manifest = json.loads(match.read_text(encoding='utf-8'))
            low_detector = next(item for item in low_manifest['layers'] if item['layer_key'] == detector['layer_key'])
            low_detector = dict(low_detector, downbin_value=str(spec.raw_value), downbin_token=str(spec.token),
                downbin_scale=float(spec.scale))
            target = staging/'low_quality'/spec.token/'nrrd'/exported_manifest.name
            detector_receipts.append(append_reference_detector(target, match.parent, low_detector))
            annotate_replay_manifest(target, provenance, staging/'selection.json')
        result = dict(schema='xta.sam_source_replay/1', complete=True, coordinate_space='source_grid',
            evidence_directory=str(bundle.directory.resolve()), evidence_fingerprint=bundle.evidence_fingerprint,
            selection_source=str(selection_path), selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
            selection_identity=selection.get('selection_identity'), policy_hash=selection['policy_hash'],
            replay=provenance,
            source_shape_tyx=list(source_shape), native_shape_tyx=list(native_shape),
            source_processing_shape_tyx=list(context.source_volume.shape), downbin=str(downbin),
            layers=entries, detector_reference_copies=detector_receipts,
            limitation='Fixed proposals and preserved source detector; no SAM regeneration, final voting or changed upstream tile gates.')
        bundle.assert_unchanged()
        if read_artifact(selection_path) != selection_bytes:
            raise RuntimeError('Selection receipt changed during source export')
        (staging/'replay_manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        if output.exists():
            raise FileExistsError('Source replay output directory appeared during export')
        os.replace(staging, output)
        return result
    except BaseException:
        if sink is not None:
            sink.shutdown()
        resolved_staging = staging.resolve()
        if (resolved_staging.parent != output.parent
                or not resolved_staging.name.startswith('.'+output.name+'.source-replay-')):
            raise RuntimeError('Source replay staging path changed before cleanup')
        shutil.rmtree(resolved_staging)
        raise
    finally:
        assembly.set_final_source_output_shape(prior_shape)
        outputs.set_nrrd_layer_sink(prior_sink)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('selection', type=Path)
    parser.add_argument('reference_run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--downbin', default='0.20')
    parser.add_argument('--memory-mib', type=float, default=256)
    args = parser.parse_args()
    result = export_source_replay(args.evidence, args.selection, args.reference_run,
        args.output, downbin=args.downbin, memory_mib=args.memory_mib)
    print(json.dumps({key:value for key,value in result.items() if key != 'layers'}, indent=2))


if __name__ == '__main__':
    main()
