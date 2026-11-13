"""Identity validation for cutoff-specific Slicer edits and legacy native transfer.

Only standard-library modules are imported here. Each exported cutoff remains an
ordinary legacy edit bundle; its additional binding checks the cutoff's original
branch tree and run metadata. No mask or source-grid conversion happens
until these identities have been checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _xta_sweep_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _xta_sweep_json(path, *, exact_numbers=False):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    options = {}
    if exact_numbers:
        from decimal import Decimal
        options['parse_float'] = Decimal
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid, **options)


def _xta_sweep_cent(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError('cutoff_cent must be an integer from 0 through 100.')
    return value


def _xta_sweep_shape(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(item) is not int or item < 1 for item in value)):
        raise ValueError('Expected three positive integer TYX dimensions.')
    return tuple(value)


def _xta_sweep_child(base, relative, *, boundary=None):
    base = Path(base).resolve()
    boundary = base if boundary is None else Path(boundary).resolve()
    raw = Path(relative)
    if raw.is_absolute() or raw.drive or (boundary == base and '..' in raw.parts):
        raise ValueError(f'Unsafe sweep member path: {relative!r}')
    result = (base / raw).resolve()
    if not result.is_relative_to(boundary):
        raise ValueError(f'Sweep member escapes its source root: {relative!r}')
    return result


def _xta_sweep_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string.')
    return value


def _xta_sweep_run(path, expected_hash, cent, sweep_id, source_mode):
    if source_mode not in {'confidence_sweep','independent_scalar'}:
        raise ValueError('Unsupported sweep edit source mode.')
    if _xta_sweep_digest(path) != expected_hash:
        raise ValueError('Sweep cutoff run-manifest checksum mismatch.')
    run = _xta_sweep_json(path,exact_numbers=True)
    if source_mode == 'independent_scalar':
        from decimal import Decimal, InvalidOperation
        value = run.get('resolved_configuration',{}).get('conf')
        if isinstance(value,bool) or value is None:
            raise ValueError('Independent scalar run needs its original resolved confidence cutoff.')
        try:
            conf = value if isinstance(value,Decimal) else Decimal(str(value))
        except (InvalidOperation,ValueError) as error:
            raise ValueError('Independent scalar confidence cutoff is invalid.') from error
        if (not conf.is_finite() or not 0 <= conf <= 1 or conf != Decimal(cent).scaleb(-2)
                or run.get('status') != 'complete' or run.get('launcher',{}).get('mode') != 'tta'):
            raise ValueError('Independent scalar run must be completed TTA with the exact declared cutoff.')
    else:
        identity = run.get('confidence_sweep',{})
        if (identity.get('sweep_id') != sweep_id
                or _xta_sweep_cent(identity.get('cutoff_hundredths')) != cent):
            raise ValueError('Sweep cutoff run-manifest identity mismatch.')
    return run


def validate_sweep_source_pair(bundle_dir, bundle, low_manifest, full_manifest, *, verify_original_paths=True):
    """Accept one bound legacy child before any native image payload is read.

    Ordinary legacy bundles return None and keep their existing behavior. Native
    and preview manifests may have extra layers and different recomposition roles;
    only the exported logical layers must match their pinned source identities.
    Portable index selection may skip access to the original tree; the native
    transfer consumer always uses the default original-path verification.
    """
    if 'sweep_context' not in bundle:
        return None
    if bundle.get('format') != 'xta-slicer-edits-v1':
        raise ValueError('Sweep source transfer requires an ordinary legacy cutoff edit bundle.')
    binding = bundle['sweep_context']
    if not isinstance(binding, dict) or binding.get('schema') != 'xta.sweep_edit_binding/1':
        raise ValueError('Unsupported sweep edit binding.')
    _xta_sweep_text(binding.get('sweep_id'), 'sweep_id')
    cent = _xta_sweep_cent(binding.get('cutoff_cent'))
    grid_id = _xta_sweep_text(binding.get('grid_id'), 'grid_id')
    if not grid_id.startswith('low_quality/'):
        raise ValueError('A sweep edit must retain its preview grid identity.')
    root = Path(bundle_dir).resolve()
    original_low = _xta_sweep_child(root, 'source_manifest.json')
    original_full = _xta_sweep_child(root, binding.get('native_manifest_file', ''))
    original_run = _xta_sweep_child(root, binding.get('run_manifest_file', ''))
    run_sha = binding.get('run_manifest_sha256')
    source_mode = binding.get('source_mode','confidence_sweep')
    _xta_sweep_run(original_run,run_sha,cent,binding['sweep_id'],source_mode)
    for key in ('preview_manifest_relative','native_manifest_relative','run_manifest_relative','branch_dir_relative'):
        _xta_sweep_child(root,binding[key])
    if verify_original_paths:
        relative_low = Path(binding['preview_manifest_relative'])
        # Relocation preserves the declared branch tree, not a particular machine path.
        source_root = Path(low_manifest).resolve()
        for _ in relative_low.parts:
            source_root = source_root.parent
        branch = _xta_sweep_child(source_root,binding['branch_dir_relative'])
        expected_low = _xta_sweep_child(source_root,binding['preview_manifest_relative'])
        expected_full = _xta_sweep_child(source_root,binding['native_manifest_relative'])
        expected_run = _xta_sweep_child(source_root,binding['run_manifest_relative'])
        standard_branch = branch.name == f'conf_{cent:03d}' and branch.parent.name == 'confidence_cutoffs'
        if ((source_mode == 'confidence_sweep' and not standard_branch) or branch == source_root
                or not all(path.is_relative_to(branch) for path in (expected_low,expected_full,expected_run))
                or expected_run.parent != branch or expected_run.name != 'manifest.json'
                or expected_low != Path(low_manifest).resolve() or expected_full != Path(full_manifest).resolve()):
            raise ValueError('Sweep source paths do not belong to the bound cutoff branch.')
        if _xta_sweep_digest(expected_run) != run_sha:
            raise ValueError('Original sweep cutoff run manifest changed since export.')
    low_sha, full_sha = binding.get('preview_manifest_sha256'), binding.get('native_manifest_sha256')
    if (bundle.get('source_manifest_sha256') != low_sha
            or _xta_sweep_digest(original_low) != low_sha
            or _xta_sweep_digest(low_manifest) != low_sha):
        raise ValueError('Sweep cutoff preview source-manifest mismatch.')
    if _xta_sweep_digest(original_full) != full_sha or _xta_sweep_digest(full_manifest) != full_sha:
        raise ValueError('Sweep cutoff native source-manifest mismatch.')
    low, full = _xta_sweep_json(original_low), _xta_sweep_json(original_full)
    if (_xta_sweep_shape(bundle.get('preview_shape_tyx')) != _xta_sweep_shape(low['output_shape_tyx'])
            or _xta_sweep_shape(bundle.get('full_quality_shape_tyx')) != _xta_sweep_shape(full['output_shape_tyx'])
            or _xta_sweep_shape(low['full_quality_output_shape_tyx']) != _xta_sweep_shape(full['output_shape_tyx'])):
        raise ValueError('Sweep edit source-grid dimensions disagree.')
    low_rows = {row['filename']: row for row in low.get('layers', [])}
    full_rows = {row['filename']: row for row in full.get('layers', [])}
    if len(low_rows) != len(low.get('layers', [])) or len(full_rows) != len(full.get('layers', [])):
        raise ValueError('Duplicate sweep scalar source filenames.')
    identities = binding.get('logical_segments', [])
    if (not identities or len({row.get('logical_id') for row in identities}) != len(identities)
            or len({row.get('segment_id') for row in identities}) != len(identities)
            or len({row.get('source_filename') for row in identities}) != len(identities)):
        raise ValueError('Duplicate or missing sweep logical edit identity.')
    edits = bundle.get('segments', [])
    by_file = {row['source_filename']: row for row in edits}
    if len(by_file) != len(edits) or set(by_file) != {row['source_filename'] for row in identities}:
        raise ValueError('Sweep edited layers differ from their logical source binding.')
    for row in identities:
        _xta_sweep_text(row.get('logical_id'), 'logical_id')
        _xta_sweep_text(row.get('segment_id'), 'segment_id')
        original = low_rows.get(row['source_filename'])
        native = full_rows.get(row['source_filename'])
        if original is None or native is None:
            raise ValueError('Bound sweep layer is absent from a scalar source manifest.')
        edit = by_file[row['source_filename']]
        if (edit.get('logical_id') != row['logical_id'] or _xta_sweep_cent(edit.get('cutoff_cent')) != cent
                or edit.get('sweep_segment_id') != row['segment_id']):
            raise ValueError('Sweep edited segment cutoff/logical identity mismatch.')
        for key in ('suffix', 'segment_name', 'model_name', 'layer_key'):
            if original.get(key) != row.get(key) or native.get(key) != row.get(key):
                raise ValueError(f'Sweep source logical identity mismatch: {key}')
        if edit.get('name') != row['segment_name'] or edit.get('recomposition_op') != original.get('recomposition_op'):
            raise ValueError('Sweep edited name/operation differs from its original scalar layer.')
    return dict(schema=binding['schema'], sweep_id=binding['sweep_id'], cutoff_cent=cent,
                source_mode=source_mode,grid_id=grid_id, preview_manifest_sha256=low_sha, native_manifest_sha256=full_sha,
                run_manifest_sha256=run_sha,logical_layer_count=len(identities),
                status='source_pair_metadata_verified_before_native_payload')


def resolve_cut_bundle(root, cutoff_cent, *, grid_id=None):
    """Select exactly one child; never choose a cutoff or union cutoffs implicitly."""
    root = Path(root).resolve()
    index = _xta_sweep_json(_xta_sweep_child(root, 'sweep_edits.json'))
    if index.get('schema') != 'xta.sweep_edits/1':
        raise ValueError('Unsupported grouped sweep edit export.')
    cent = _xta_sweep_cent(cutoff_cent)
    candidates = [row for row in index.get('cuts', []) if row.get('cutoff_cent') == cent
                  and (grid_id is None or row.get('grid_id') == grid_id)]
    if len(candidates) != 1:
        raise ValueError('Select exactly one cutoff and preview grid from the grouped edits.')
    row = candidates[0]
    child = _xta_sweep_child(root, row['directory'])
    edits_path = _xta_sweep_child(child, 'edits.json')
    if _xta_sweep_digest(edits_path) != row.get('edits_sha256'):
        raise ValueError('Grouped cutoff edit checksum mismatch.')
    bundle = _xta_sweep_json(edits_path)
    binding = bundle.get('sweep_context', {})
    if (binding.get('sweep_id') != index.get('sweep_id') or binding.get('cutoff_cent') != cent
            or binding.get('source_mode','confidence_sweep') != index.get('source_mode','confidence_sweep')
            or binding.get('grid_id') != row.get('grid_id')):
        raise ValueError('Grouped cutoff child identity mismatch.')
    validate_sweep_source_pair(child, bundle, child/'source_manifest.json', child/'source_native_manifest.json',
                               verify_original_paths=False)
    return child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edits', required=True, type=Path, help='Grouped Slicer export directory')
    parser.add_argument('--cutoff-cent', required=True, type=int, help='Explicit integer cutoff, 0 through 100')
    parser.add_argument('--grid-id', help='Preview grid, required when the cutoff appears on several grids')
    args = parser.parse_args(argv)
    child = resolve_cut_bundle(args.edits, args.cutoff_cent, grid_id=args.grid_id)
    print(json.dumps({'status':'verified_cutoff_edit_metadata', 'edits':str(child)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
