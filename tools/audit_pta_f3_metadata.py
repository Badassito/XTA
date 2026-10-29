"""Audit native-cleanup provenance/counts using explicit receipts and input paths.

The original F3 report schema and evidence field names are retained. Counts scaled
by voxel-grid ratio do not establish spatial agreement. Outputs require an
explicit directory outside the repository and all supplied input directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .pta_audit_common import external_output, receipt_child, write_json
else:
    from pta_audit_common import external_output, receipt_child, write_json


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def percent_change(value, reference):
    return 100 * (value / reference - 1) if reference else None


def audit(native: Path, input_dir: Path, edits_path: Path, output_dir: Path):
    native, input_dir, edits_path = (Path(p).expanduser().resolve() for p in (native, input_dir, edits_path))
    output = external_output(output_dir, inputs=(native, input_dir, edits_path.parent))
    report_path = native / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    edits = json.loads(edits_path.read_text(encoding='utf-8'))
    global_edits = [s for s in edits['segments'] if s['recomposition_op'] == 'select']
    if len(global_edits) != 1:
        raise ValueError('Expected exactly one global select segment in edits receipt')
    global_edit = global_edits[0]
    preview_count = global_edit['foreground_voxels']
    grid_ratio = math.prod(report['shape_tyx']) / math.prod(report['preview_shape_tyx'])
    native_equivalent = preview_count * grid_ratio
    filename = report['pta_label_file']
    old_file, input_file = receipt_child(native, filename), receipt_child(input_dir, filename)
    old_hash, input_hash = sha256(old_file), sha256(input_file)
    result = {
        'scope': 'Metadata and compressed source-byte provenance audit; no image modifications or GPU use.',
        'source_paths': {'native_report': str(report_path), 'edited_preview_receipt': str(edits_path),
                         'old_native_result': str(old_file), 'foundation_f3_input': str(input_file)},
        'source_sha256': {'native_report': sha256(report_path), 'edited_preview_receipt': sha256(edits_path),
                          'old_native_result': old_hash, 'foundation_f3_input': input_hash},
        'input_and_old_native_bytes_equal_sha256': input_hash == old_hash,
        'file_sizes_bytes': {'old_native_result': old_file.stat().st_size,
                             'foundation_f3_input': input_file.stat().st_size},
        'native_report': {k: report[k] for k in ('tool_version', 'status', 'created_utc', 'shape_tyx',
            'preview_shape_tyx', 'preview_manifest_sha256', 'full_manifest_sha256',
            'initial_global_voxels', 'final_global_voxels', 'unresolved_components',
            'auto_resolved_components', 'auto_removed_voxels', 'union_cleanup', 'warnings')},
        'native_layer_count': len(report['layers']),
        'native_conflicting_voxels_preserved': sum(l.get('conflicting_voxels_preserved', 0) or 0 for l in report['layers']),
        'native_unobserved_components_preserved': sum(l.get('unobserved_components_preserved', 0) or 0 for l in report['layers']),
        'edited_preview': {'global_mask_file': global_edit['file'], 'global_name': global_edit['name'],
            'global_foreground_voxels': preview_count, 'segment_count': len(edits['segments']),
            'preview_shape_tyx': edits['preview_shape_tyx'],
            'full_quality_shape_tyx': edits['full_quality_shape_tyx'],
            'source_manifest_sha256': edits['source_manifest_sha256'], 'notes': edits['notes']},
        'count_comparison': {'exact_full_to_preview_grid_voxel_ratio': grid_ratio,
            'preview_global_voxels_times_exact_grid_ratio': native_equivalent,
            'native_output_change_vs_scaled_preview_percent': percent_change(report['final_global_voxels'], native_equivalent),
            'native_output_change_vs_initial_native_percent': percent_change(report['final_global_voxels'], report['initial_global_voxels'])},
        'limitations': [
            'Preview count is from the edits receipt; its full 3D mask was not recomputed by this audit.',
            'Volume-normalized foreground count does not establish spatial alignment or boundary agreement.',
            'No final native-to-preview footprint overlap or boundary/noise measurement was performed.',
            'No Foundationv5 label TXT/image pair or Roboflow-imported artifact was inspected in this F3 metadata audit.',
            'Native recovery selects original native components using preview evidence; its report explicitly disclaims exact low-resolution equality.'
        ]
    }
    write_json(output, 'f3_metadata_evidence.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True, help='Directory containing cleanup report.json and native output')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing the matching PTA input NRRD')
    parser.add_argument('--edits', type=Path, required=True, help='Edited preview edits.json receipt')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.native_dir, args.input_dir, args.edits, args.output_dir)
    print(json.dumps({'evidence_path': str(args.output_dir.resolve() / 'f3_metadata_evidence.json'),
                      'input_matches_old_native': result['input_and_old_native_bytes_equal_sha256'],
                      'input_sha256': result['source_sha256']['foundation_f3_input'],
                      'count_comparison': result['count_comparison']}, indent=2))


if __name__ == '__main__':
    main()
