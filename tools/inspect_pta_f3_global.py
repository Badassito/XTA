"""Inspect Global segment metadata and NPZ arrays without modifying input files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .pta_audit_common import external_output, receipt_child, write_json
else:
    from pta_audit_common import external_output, receipt_child, write_json


def inspect(edits_path: Path, output_dir: Path):
    import numpy as np

    edits_path = Path(edits_path).expanduser().resolve()
    output = external_output(output_dir, inputs=(edits_path.parent,))
    edits = json.loads(edits_path.read_text(encoding='utf-8'))
    records = []
    for index, segment in enumerate(edits['segments']):
        if 'global' not in segment['name'].casefold():
            continue
        path = receipt_child(edits_path.parent, segment['file'])
        arrays = np.load(path, allow_pickle=False)
        try:
            def describe(key, value):
                return {'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
                        'preview': str(value)[:300]}

            # Read NPZ arrays one at a time, retaining only small descriptions.
            descriptions = ([describe(key, arrays[key]) for key in arrays.files]
                            if isinstance(arrays, np.lib.npyio.NpzFile)
                            else [describe('array', arrays)])
            records.append({'segment_index': index, 'segment': segment,
                            'arrays': descriptions})
        finally:
            if isinstance(arrays, np.lib.npyio.NpzFile):
                arrays.close()
    if not records:
        raise ValueError('No Global-named segments in the edits receipt')
    result = {'edits_path': str(edits_path), 'global_segments': records}
    write_json(output, 'f3_global_inspection.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edits', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(inspect(args.edits, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
