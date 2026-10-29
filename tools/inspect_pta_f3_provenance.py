"""Inspect native-cleanup receipts and paired NRRD headers using explicit inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .pta_audit_common import external_output, receipt_child, write_json
else:
    from pta_audit_common import external_output, receipt_child, write_json


def read_header(path: Path):
    lines = []
    with path.open('rb') as stream:
        for _ in range(80):
            line = stream.readline()
            if not line.strip():
                break
            lines.append(line.decode('utf-8', errors='replace').rstrip())
    return {'path': str(path), 'bytes': path.stat().st_size, 'header_lines': lines}


def inspect(native: Path, input_dir: Path, edits_path: Path, output_dir: Path):
    native, input_dir, edits_path = (Path(p).expanduser().resolve() for p in (native, input_dir, edits_path))
    output = external_output(output_dir, inputs=(native, input_dir, edits_path.parent))
    report_path = native / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    edits = json.loads(edits_path.read_text(encoding='utf-8'))
    filename = report['pta_label_file']
    result = {
        'native_report_path': str(report_path),
        'report_top': {key: value for key, value in report.items() if key not in ('layers', 'source_full_sha256')},
        'layer_count': len(report['layers']),
        'conflicting_total': sum(layer.get('conflicting_voxels_preserved', 0) or 0 for layer in report['layers']),
        'unobserved_components': sum(layer.get('unobserved_components_preserved', 0) or 0 for layer in report['layers']),
        'headers': [read_header(receipt_child(base, filename)) for base in (native, input_dir)],
        'edits_path': str(edits_path),
        'edits_meta': {key: value for key, value in edits.items() if key != 'segments'},
        'segment_count': len(edits['segments']),
        'foreground_sum': sum(segment['foreground_voxels'] for segment in edits['segments']),
    }
    write_json(output, 'f3_provenance_inspection.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--edits', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(inspect(args.native_dir, args.input_dir, args.edits, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
