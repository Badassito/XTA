"""Stream source-grid detector/SAM masks without allocating a dense volume."""
from pathlib import Path
import gzip
import argparse
import json
import sys
import time
from collections import OrderedDict

import numpy as np


def open_payload(path):
    stream = path.open('rb')
    fields = {}
    while line := stream.readline():
        if not line.strip():
            break
        if b':' in line and not line.startswith(b'#'):
            key, value = line.decode('ascii').split(':', 1)
            fields[key] = value.strip()
    assert fields['encoding'] == 'gzip', fields
    shape = tuple(reversed(tuple(map(int, fields['sizes'].split()))))
    return stream, gzip.GzipFile(fileobj=stream), shape


class NewVoxelDiagnostic:
    """Keep the optional inspection delta separate from production contributions."""

    def __init__(self, path, shape):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT
        self.path = Path(path).resolve()
        self.shape = tuple(shape)
        self.backing = self.path.parent/(self.path.name+'.cvol')
        if self.path.exists() or self.backing.exists() or self.path.with_suffix('.json').exists():
            raise FileExistsError('SAM new-voxel diagnostic destination must be fresh')
        self.writer = IncrementalRawBBoxMaskStoreWriter(shape=self.shape, store_dir=self.backing,
            format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='Diagnostic SAM union minus matching detector',
            extra_meta={'diagnostic_only':True, 'recomposition_op':'none',
                'operation':'OR restored directional SAM masks, then subtract detector at the same resolution'})

    def consume(self, frame, mask):
        self.writer.consume_sparse_slice(frame, 0, self.shape[1], 0, self.shape[2],
            mask.reshape(self.shape[1:]))

    def finalize(self, *, files, policy_ids):
        from XTA.interpolation import NrrdLayerRef, INTERNAL_PACKED_CVOL_FORMAT
        from XTA.outputs import write_single_layer_nrrd_from_ref
        stats = self.writer.finalize()
        ref = NrrdLayerRef(key='sam_new_voxels_diagnostic', name='SAM new voxels diagnostic',
            path=self.backing, shape=self.shape, dtype='uint8', storage_format=INTERNAL_PACKED_CVOL_FORMAT,
            model_name='diagnostic', view_name='transverse_source_grid', physical_view_name='transverse',
            aug_id='a0', angle_deg=0., view_family='orthogonal', source='diagnostic',
            mask_kind='diagnostic', pass_index=0, stage='same_resolution_subtraction',
            layer_role='diagnostic_only', recomposition_op='none',
            segment_extent_ijk=tuple(stats['segment_extent_ijk']), segment_extent_shape_tyx=self.shape,
            segment_extent_source='exact_same_resolution_diagnostic',
            description='Diagnostic only: restored SAM union minus detector at the identical output resolution.')
        write_single_layer_nrrd_from_ref(ref, self.shape, self.path,
            segment_name='Diagnostic SAM new voxels outside matching detector', z_shards=1)
        metadata = dict(schema='xta.sam_new_voxel_diagnostic/1', diagnostic_only=True,
            recomposition_op='none', nrrd=str(self.path), backing_store=str(self.backing),
            output_shape_tyx=list(self.shape), input_files=list(map(str, files)),
            interpolation_policy_identities=list(policy_ids),
            subtraction_order='Restore/downbin each input to this output resolution; OR directional SAM masks; subtract the matching detector.',
            interpretation='Inspection delta only; the original SAM contribution files are preserved unchanged.',
            voxel_count=int(stats['foreground_voxels']))
        self.path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run or low_quality output containing an nrrd directory.')
    parser.add_argument('output', type=Path, help='JSON report destination outside the source repository.')
    parser.add_argument('--audit-native', action='store_true',
                        help='Also compare retained native masks/confidence and reconstruct detector restoration.')
    parser.add_argument('--new-voxels-output', type=Path,
                        help='Also write a separate diagnostic SAM union minus detector at this exact output resolution.')
    args = parser.parse_args()
    root, destination = args.run, args.output
    manifest_path = next((root / 'nrrd').glob('*_nrrd_manifest.json'))
    layers = json.loads(manifest_path.read_text())['layers']
    bridge_layers = [layer for layer in layers if layer.get('interpolation_backend') == 'sam']
    if not bridge_layers:
        raise ValueError('This output contains no selected SAM bridge layers')
    scopes = {(layer['view_name'], layer['source'], layer.get('tile_config_id', '')) for layer in bridge_layers}
    if len(scopes) != 1:
        raise ValueError('Audit one SAM view/source scope at a time')
    detector = next(layer for layer in layers if layer['mask_kind'] == 'yolo' and
                    layer['view_name'] == bridge_layers[0]['view_name'] and
                    layer['source'] == bridge_layers[0]['source'] and
                    layer.get('tile_config_id', '') == bridge_layers[0].get('tile_config_id', ''))
    names = [detector['filename'], *(layer['filename'] for layer in bridge_layers)]
    native_audit = None
    native_reader = None
    native_owners = []
    restored_original = None
    if args.audit_native:
        transform = bridge_layers[0]['native_transform']
        if transform['view_family'] != 'orthogonal' or transform['view_name'] != 'transverse':
            raise ValueError('Native-to-source restoration audit currently requires verified Transverse geometry')
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from XTA.confidence_evidence import ConfidenceEvidenceRef
        from XTA.interpolation import RawBBoxMaskStore
        from XTA.outputs import _restore_source_indices_for_output_z, _resize_binary_mask_frame_to_output_shape
        confidence_manifest = json.loads((root/'reconciliation_evidence'/'manifest.json').read_text())
        confidence_entry = next(layer for layer in confidence_manifest['layers']
                                if layer['layer_key'] == detector['layer_key'])
        confidence_ref = ConfidenceEvidenceRef.open(root/'reconciliation_evidence'/confidence_entry['directory'])
        native_reader = confidence_ref.native_reader()
        native_shape = native_reader.shape
        for layer in bridge_layers:
            bundle = (root/'nrrd'/layer['proposal_evidence_path']).resolve()
            path = bundle.parent / f"sam_bridge_pass{int(layer['pass_index']):02d}_{layer['interpolation_direction']}.cvol"
            native_owners.append(RawBBoxMaskStore.open(path, mmap_payload=True))
        assert all(owner.shape == native_shape for owner in native_owners)
        native_audit = dict(shape_tyx=native_shape, confidence_support_voxels=0,
                            bridge_union_voxels=0, overlap_voxels=0,
                            source_restored_original_voxels=0, source_restored_overlap_voxels=0,
                            source_detector_original_intersection_voxels=0)
        for frame in range(native_shape[0]):
            original = native_reader(frame, frame+1)[1][0]
            union = np.logical_or.reduce([owner.decode_slice(frame) != 0 for owner in native_owners])
            native_audit['confidence_support_voxels'] += int(np.count_nonzero(original))
            native_audit['bridge_union_voxels'] += int(np.count_nonzero(union))
            native_audit['overlap_voxels'] += int(np.count_nonzero(original & union))
        cache = OrderedDict()
        def restored_original(frame, shape):
            output = np.zeros(shape[1:], dtype=np.uint8)
            for index in _restore_source_indices_for_output_z(native_shape[0], shape[0], frame):
                if index not in cache:
                    cache[index] = native_reader(index, index+1)[1][0].astype(np.uint8)
                    if len(cache) > 4:
                        cache.popitem(last=False)
                output |= _resize_binary_mask_frame_to_output_shape(cache[index], *shape[1:])
            return output != 0
    owners = [open_payload(root / 'nrrd' / name) for name in names]
    shape = owners[0][2]
    assert all(owner[2] == shape for owner in owners)
    diagnostic = NewVoxelDiagnostic(args.new_voxels_output, shape) if args.new_voxels_output else None
    count = shape[1] * shape[2]
    totals = dict(detector_voxels=0, bridge_union_voxels=0, overlap_union_voxels=0,
                  directions=[dict(voxels=0, overlap_voxels=0) for _ in bridge_layers])
    rows = []
    started = time.monotonic()
    for frame in range(shape[0]):
        planes = [np.frombuffer(owner[1].read(count), dtype=np.uint8) for owner in owners]
        assert all(plane.size == count for plane in planes), frame
        original = planes[0] != 0
        bridges = [plane != 0 for plane in planes[1:]]
        union = np.logical_or.reduce(bridges)
        detector_count = int(np.count_nonzero(original))
        bridge_count = int(np.count_nonzero(union))
        overlap_count = int(np.count_nonzero(original & union))
        if diagnostic is not None:
            diagnostic.consume(frame, union & ~original)
        if restored_original is not None:
            restored = restored_original(frame, shape).ravel()
            native_audit['source_restored_original_voxels'] += int(np.count_nonzero(restored))
            native_audit['source_restored_overlap_voxels'] += int(np.count_nonzero(restored & union))
            native_audit['source_detector_original_intersection_voxels'] += int(np.count_nonzero(restored & original))
        totals['detector_voxels'] += detector_count
        totals['bridge_union_voxels'] += bridge_count
        totals['overlap_union_voxels'] += overlap_count
        for mask, total in zip(bridges, totals['directions']):
            total['voxels'] += int(np.count_nonzero(mask))
            total['overlap_voxels'] += int(np.count_nonzero(mask & original))
        if bridge_count:
            rows.append(dict(frame=frame, detector=detector_count, bridge=bridge_count, overlap=overlap_count))
        if frame % 100 == 0:
            print(json.dumps(dict(frame=frame, seconds=round(time.monotonic()-started, 2))), flush=True)
    for stream, payload, _ in owners:
        payload.close()
        stream.close()
    for owner in native_owners:
        owner.close()
    if native_reader is not None:
        native_reader.close()
    result = dict(run=root.name, shape_tyx=shape, files=names, totals=totals, slices=rows,
                  seconds=time.monotonic()-started)
    if native_audit is not None:
        result['native_audit'] = native_audit
    if diagnostic is not None:
        result['new_voxels_diagnostic'] = diagnostic.finalize(
            files=[root/'nrrd'/name for name in names],
            policy_ids=sorted({layer['interpolation_policy_identity'] for layer in bridge_layers}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'slices'}), flush=True)


if __name__ == '__main__':
    main()
