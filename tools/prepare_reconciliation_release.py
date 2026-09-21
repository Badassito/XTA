"""Prepare an append-only release review from an authenticated Git predecessor.

The default invocation writes review evidence outside the repository. Use --write
only after the reviewed runtime and validation tools are frozen.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_package_inventory as inventory


RELEASES = {
    '22.3.1': dict(token='22_3_1', previous_token='22_3',
                  feature='efficient-union-and-native-confidence',
                  validation_tools=(
                      'tools/compare_reconciliation.py',
                      'tools/qualify_tta_reconciliation.py',
                      'tools/export_reconciliation_evidence.py')),
}
REASONS = {
    '__init__': 'Publish the package release identity as {release}.',
    'cli': 'Use the sole {release} launcher and current release identity.',
    'config': 'Publish the current release identity and reconciliation configuration.',
    'pipeline': 'Reuse the assembled additive union, release replaced buffer owners, defer unused confidence projection, and drain owned evidence futures with completion progress and propagated failures.',
    'reconciliation_runtime': 'Reuse borrowed union buffers without component rescans and publish truthful metadata while closing explicit reader owners.',
    'confidence_evidence': 'Persist bounded block and native confidence with publication start/completion progress, without unnecessary source-grid projection.',
    'confidence_storage': 'Store validated, bounded confidence blocks with checked codec metadata and periodic write progress.',
    'confidence_native': 'Preserve native confidence pieces and make bounded source-grid conversion an explicit operation.',
    'confidence_export': 'Export retained native confidence through explicit source-grid conversion and checked output metadata.',
    'confidence_projection': 'Project observed confidence only when requested, preserving geometry and reporting progress during tilted-azimuthal staging.',
    'confidence_tiles': 'Preserve gated tile confidence as native evidence with explicit parent support provenance.',
    'cuda_d1': 'Retain native D1 confidence shards through bounded storage and retirement.',
    'cuda_backend': 'Fuse confidence composition in an optional compiled 2D loop with conservative layout and alias admission, safe compile preflight, and unchanged NumPy fallback semantics.',
    'packed_publication': 'Accelerate packed metadata scans with LLVM population count, bit scans, and vectorizable interior counting while preserving the existing payload encoder.',
    'assembly': 'Preserve immutable confidence publication and release score workspaces on success or failure.',
    'reconciliation_io': 'Read persisted source or native confidence references without implicit projection.',
    'outputs': 'Preserve output publication and reconciliation evidence metadata.',
    'workers': 'Carry current confidence transport and ownership configuration into inference workers.',
    'publication_memory': 'Account for bounded confidence and publication ownership.',
    'inference': 'Preserve confidence transport without changing inference masks or morphology.',
    'backprojection': 'Preserve native confidence evidence and the existing binary projection contract.',
    'examples/external_reconciliation/_confidence_core': 'Provide the curated confidence core and spatial rescue decision with the validated two-dimensional behavior.',
    'examples/external_reconciliation/_hybrid': 'Provide the curated hybrid consensus, rescue and enclosed-hole filling decision.',
    'examples/external_reconciliation/confidence_core_rescue': 'Publish the confidence core rescue preset selected through visual review.',
    'examples/external_reconciliation/quorum3': 'Publish the independent section quorum preset as a compact voting alternative.',
    'examples/external_reconciliation/hybrid_with_fill': 'Publish the hybrid consensus and limited enclosed-hole filling preset.',
}
REMOVAL_REASONS = {
    'examples/external_reconciliation/' + name:
        'Retire this preset from the curated package selection while preserving its authenticated release history.'
    for name in ('baseline', 'confidence_voxel', 'cross_sections', 'provenance')
}
TOOL_REASONS = {
    'tools/compare_reconciliation.py': 'Compare persisted evidence with explicit native conversion, bounded readers and unchanged source artifacts.',
    'tools/qualify_tta_reconciliation.py': 'Qualify unchanged masks and complete source or native confidence across CPU, GPU and hybrid inference.',
    'tools/export_reconciliation_evidence.py': 'Explicitly export retained native confidence into a checked source-grid companion.',
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git_file(root, predecessor, relative):
    result = subprocess.run(['git', 'show', predecessor + ':' + relative], cwd=root, capture_output=True)
    return None if result.returncode else result.stdout.decode('utf-8')


def identity(node):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return ('definition', node.name)
    if isinstance(node, ast.Assign):
        return ('binding', tuple(n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)))
    if isinstance(node, ast.AnnAssign):
        return ('binding', (getattr(node.target, 'id', ast.dump(node.target)),))
    if isinstance(node, ast.ImportFrom):
        return ('from', node.level, node.module)
    if isinstance(node, ast.Import):
        return ('import', tuple(alias.name for alias in node.names))
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return ('docstring',)
    return (type(node).__name__,)


def statement_label(node):
    key = identity(node)
    if key[0] == 'binding':
        return 'binding_' + '_'.join(key[1])
    if key[0] == 'from':
        return 'import_' + '.' * key[1] + (key[2] or '')
    if key[0] == 'import':
        return 'import_' + '_'.join(key[1])
    if key[0] == 'docstring':
        return 'module_docstring'
    return f'{type(node).__name__}:{node.lineno}'


def review_module(module, old_source, new_source, *, complete, labels_by_hash, reason):
    """Account for every predecessor statement without silently dropping one."""
    old = ast.parse(old_source) if old_source is not None else None
    new = ast.parse(new_source)
    old_hashes = [inventory.digest(node) for node in old.body] if old is not None else []
    new_hashes = [inventory.digest(node) for node in new.body]
    pin = dict(ast_sha256=inventory.digest(old) if old is not None else None,
               statements_sha256=hashlib.sha256(json.dumps(old_hashes, separators=(',', ':')).encode()).hexdigest())
    snapshot = dict(module=module, previous_ast_sha256=pin['ast_sha256'], ast_sha256=inventory.digest(new),
                    previous_top_level=old_hashes, top_level=new_hashes, reason=reason)
    records = dict(definitions=[], statements=[], local_import_seam_updates=[])
    unmatched, used_labels = set(range(len(old_hashes))), set()
    for index, node in enumerate(new.body):
        previous = None
        if old is not None:
            previous = next((i for i in sorted(unmatched) if identity(old.body[i]) == identity(node)
                             and old_hashes[i] == new_hashes[index]), None)
            if previous is None:
                previous = next((i for i in sorted(unmatched) if identity(old.body[i]) == identity(node)), None)
        if previous is not None:
            unmatched.remove(previous)
        previous_hash = old_hashes[previous] if previous is not None else None
        if previous_hash == new_hashes[index] and not complete:
            continue
        item = dict(module=module, previous_sha256=previous_hash, sha256=new_hashes[index],
                    previous_index=previous, current_index=index, reason=reason)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            records['definitions'].append({**item, 'name': node.name})
        else:
            label = labels_by_hash.get((module, previous_hash), statement_label(node))
            if label in used_labels:
                label += f':{index}'
            used_labels.add(label)
            item['label'] = label
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                item['binding'] = node.targets[0].id
            records['statements'].append(item)
    if unmatched:
        raise ValueError(f'{module}: unaccounted predecessor statements: '
                         f'{[(i, identity(old.body[i])) for i in sorted(unmatched)]}')
    old_seams = inventory.reviewed_local_import_seams(module, old_source, old) if old is not None else {}
    new_seams = inventory.reviewed_local_import_seams(module, new_source, new)
    if set(new_seams) != set(old_seams):
        raise ValueError(f'{module}: local-import seam ownership changed')
    for key, current in new_seams.items():
        previous = old_seams[key]
        if current != previous:
            records['local_import_seam_updates'].append(dict(module=module, name=key[1],
                previous_definition_sha256=previous[0], previous_seam_sha256=previous[1],
                definition_sha256=current[0], seam_sha256=current[1], reason=reason))
    return pin, snapshot, records


def review_removed_module(module, old_source, *, reason):
    """Record an explicitly reviewed module retirement with its exact predecessor."""
    if module not in REMOVAL_REASONS or old_source is None or not reason:
        raise ValueError('Module deletion needs a separate review: ' + module)
    old = ast.parse(old_source)
    historical = [inventory.digest(node) for node in old.body]
    pin = dict(ast_sha256=inventory.digest(old),
               statements_sha256=hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest())
    snapshot = dict(module=module, previous_ast_sha256=pin['ast_sha256'], ast_sha256=None,
                    previous_top_level=historical, top_level=[], removed=True, reason=reason)
    records = dict(definitions=[], statements=[], local_import_seam_updates=[])
    return pin, snapshot, records


def _qualified_definition(source, name):
    node = ast.parse(source)
    for part in name.split('.'):
        node = next(value for value in node.body if getattr(value, 'name', None) == part)
    return node


def _update_verifier_pins(source, prefix, digest, pins):
    updates = {prefix + '_SHA256': repr(digest),
               prefix + '_PREDECESSOR_MODULES': pprint.pformat(pins, width=110, sort_dicts=True)}
    lines = source.splitlines(keepends=True)
    changes = [(node, target.id) for node in ast.parse(source).body if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Name) and target.id in updates]
    if len(changes) != len(updates) or {name for _node, name in changes} != set(updates):
        raise ValueError('Verifier is missing the release digest or predecessor-module pin binding')
    for node, name in sorted(changes, key=lambda pair: pair[0].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [name + ' = ' + updates[name] + '\n']
    return ''.join(lines)


def prepare(*, output_dir, release='22.3.1', write=False):
    root, output_dir = ROOT, Path(output_dir).resolve()
    if output_dir.is_relative_to(root):
        raise ValueError('Generated release-review evidence belongs outside the repository')
    spec = RELEASES[release]
    prefix = 'REVIEWED_V' + spec['token'] + '_RELEASE'
    key = 'v' + spec['token'] + '_release_review'
    predecessor_commit = getattr(inventory, prefix + '_PREDECESSOR_COMMIT')
    predecessor = json.loads(git_file(root, predecessor_commit, 'XTA/_package_inventory.json'))
    if canonical(predecessor) != getattr(inventory, prefix + '_PREDECESSOR_SHA256'):
        raise ValueError('Git predecessor differs from the independently authenticated inventory')
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    if {k: v for k, v in current.items() if k != key} != predecessor:
        raise ValueError('Preserve every predecessor inventory record before adding this review')
    audited = {item['module'] for item in predecessor['statements']}
    labels_by_hash = {}
    for value in predecessor.values():
        if isinstance(value, dict):
            for category in ('definitions', 'statements'):
                for item in value.get(category, ()):
                    audited.add(item['module'])
                    if category == 'statements' and 'label' in item:
                        labels_by_hash[item['module'], item['sha256']] = item['label']
    for (module, label), (value, _reason) in inventory.REVIEWED_V20_ADDED_STATEMENTS.items():
        labels_by_hash.setdefault((module, value), label)
    paths = subprocess.check_output(['git', 'diff', '--name-only', predecessor_commit, '--', 'XTA'], cwd=root, text=True).splitlines()
    paths += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'XTA'], cwd=root, text=True).splitlines()
    review = dict(release=release, feature=spec['feature'],
        previous_review_sha256=getattr(inventory, 'REVIEWED_V' + spec['previous_token'] + '_RELEASE_SHA256'),
        predecessor_commit=predecessor_commit, predecessor_inventory_sha256=canonical(predecessor),
        definitions=[], statements=[], local_import_seam_updates=[], preserved_radial_definition_updates=[],
        preserved_radial_module_updates=[], complete_modules=[], module_snapshots=[])
    source_pins = {}
    for relative in sorted(set(paths)):
        if not relative.endswith('.py'):
            continue
        path = root / relative
        module = relative.removeprefix('XTA/').removesuffix('.py')
        old_source = git_file(root, predecessor_commit, relative)
        if not path.is_file():
            pin, snapshot, records = review_removed_module(module, old_source,
                reason=REMOVAL_REASONS.get(module, ''))
            complete = False
        else:
            new_source = path.read_text(encoding='utf-8')
            if old_source is not None and inventory.digest(ast.parse(old_source)) == inventory.digest(ast.parse(new_source)):
                continue
            reason = REASONS[module].format(release=release)
            complete = module not in audited
            pin, snapshot, records = review_module(module, old_source, new_source,
                complete=complete, labels_by_hash=labels_by_hash, reason=reason)
        source_pins[module] = pin
        review['module_snapshots'].append(snapshot)
        if complete:
            review['complete_modules'].append(module)
        for category, values in records.items():
            review[category].extend(values)
    patches = [value for name, value in predecessor.items() if name != 'v21_review'
               and isinstance(value, dict) and 'release' in value and 'definitions' in value]
    for module, previous_hash in inventory.reviewed_radial_module_hashes(predecessor['v21_review'], patches).items():
        text = (root / 'XTA' / f'{module}.py').read_text(encoding='utf-8')
        new_hash = hashlib.sha256(text.encode()).hexdigest()
        if new_hash != previous_hash:
            if hashlib.sha256(git_file(root, predecessor_commit, f'XTA/{module}.py').encode()).hexdigest() != previous_hash:
                raise ValueError(f'Preserved module predecessor differs: {module}')
            review['preserved_radial_module_updates'].append(dict(module=module,
                previous_sha256=previous_hash, sha256=new_hash, reason=REASONS[module].format(release=release)))
    for (module, name), previous_hash in inventory.reviewed_radial_definition_hashes(predecessor['v21_review'], patches).items():
        text = (root / 'XTA' / f'{module}.py').read_text(encoding='utf-8')
        new_hash = inventory.digest(_qualified_definition(text, name))
        if new_hash != previous_hash:
            old_text = git_file(root, predecessor_commit, f'XTA/{module}.py')
            if inventory.digest(_qualified_definition(old_text, name)) != previous_hash:
                raise ValueError(f'Preserved definition predecessor differs: {module}.{name}')
            review['preserved_radial_definition_updates'].append(dict(module=module, qualified_name=name,
                previous_sha256=previous_hash, sha256=new_hash, reason=REASONS[module].format(release=release)))
    previous_review = predecessor['v' + spec['previous_token'] + '_release_review']
    previous_tools = {item['path']: item['sha256'] for item in previous_review.get('validation_tools', ())}
    review['validation_tools'] = [dict(path=path, previous_sha256=previous_tools.get(path),
        sha256=hashlib.sha256((root / path).read_text(encoding='utf-8').encode()).hexdigest(),
        reason=TOOL_REASONS[path]) for path in spec['validation_tools']]
    payload = {**predecessor, key: review}
    digest = canonical(review)
    # Validate draft structure using its proposed pins without publishing them.
    original_digest = getattr(inventory, prefix + '_SHA256')
    original_pins = getattr(inventory, prefix + '_PREDECESSOR_MODULES')
    try:
        setattr(inventory, prefix + '_SHA256', digest)
        setattr(inventory, prefix + '_PREDECESSOR_MODULES', source_pins)
        getattr(inventory, 'reviewed_v' + spec['token'] + '_release_contract')(payload, predecessor['v21_review'])
    finally:
        setattr(inventory, prefix + '_SHA256', original_digest)
        setattr(inventory, prefix + '_PREDECESSOR_MODULES', original_pins)
    if write:
        trees = {item['module']: ast.parse((root / 'XTA' / (item['module'] + '.py')).read_text(encoding='utf-8'))
                 for item in review['module_snapshots'] if not item.get('removed')}
        inventory.verify_v22_3_source_snapshots(review, trees)
        inventory.verify_v22_3_validation_tools(review)
        verifier = root / 'tools/verify_package_inventory.py'
        verifier_source = _update_verifier_pins(verifier.read_text(encoding='utf-8'), prefix, digest, source_pins)
        inventory.MANIFEST.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
        verifier.write_text(verifier_source, encoding='utf-8', newline='\n')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'release_review_draft.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'release_predecessor_source_pins.json').write_text(json.dumps(source_pins, indent=2) + '\n', encoding='utf-8')
    summary = dict(written=write, release=release, modules=len(source_pins), definitions=len(review['definitions']),
                   statements=len(review['statements']), seams=len(review['local_import_seam_updates']),
                   removed_modules=sum(item.get('removed') is True for item in review['module_snapshots']), sha256=digest)
    (output_dir / 'release_review_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=tuple(RELEASES), default='22.3.1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output_dir=args.output_dir, release=args.release, write=args.write)))


if __name__ == '__main__':
    main()
