"""Read-only F3 provenance/count audit; writes evidence to the task Scratch directory."""
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


native = Path('H:/F3_10_4_30_2026_8bit_Y_pta_158077_20261019_processfast/clean-native')
input_dir = Path('H:/XTA_TRAINING/v1/F3_10_4_30_2026_8bit_Y_Input')
edits_path = Path('H:/XTA_TRAINING/v1/F3_10_4_30_2026_8bit_Y_Filtered/edits.json')
report_path = native / 'report.json'
report = json.loads(report_path.read_text())
edits = json.loads(edits_path.read_text())
global_edit = next(s for s in edits['segments'] if s['recomposition_op'] == 'select')
preview_count = global_edit['foreground_voxels']
grid_ratio = math.prod(report['shape_tyx']) / math.prod(report['preview_shape_tyx'])
native_equivalent = preview_count * grid_ratio
filename = report['pta_label_file']
old_file, input_file = native / filename, input_dir / filename
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
        'native_output_change_vs_scaled_preview_percent': 100 * (report['final_global_voxels'] / native_equivalent - 1),
        'native_output_change_vs_initial_native_percent': 100 * (report['final_global_voxels'] / report['initial_global_voxels'] - 1)},
    'limitations': [
        'Preview count is from the edits receipt; its full 3D mask was not recomputed by this audit.',
        'Volume-normalized foreground count does not establish spatial alignment or boundary agreement.',
        'No final native-to-preview footprint overlap or boundary/noise measurement was performed.',
        'No Foundationv5 label TXT/image pair or Roboflow-imported artifact was inspected in this F3 metadata audit.',
        'Native recovery selects original native components using preview evidence; its report explicitly disclaims exact low-resolution equality.'
    ]
}
destination = Path('C:/Users/Bry/Documents/ChatGPT/Scratch/Experiments/PTA_ENHANCEMENT/label_size_audit_20261024/f3_metadata_evidence.json')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'evidence_path': str(destination), 'input_matches_old_native': input_hash == old_hash,
                  'input_sha256': input_hash, 'count_comparison': result['count_comparison']}, indent=2))
