"""Explicit, audited source unions for deletion-only native XTA recovery.

This module does not infer confidence from Slicer IDs or alter an edit bundle.
Its source specification declares an inventory-based reconstruction, whose
original manifest hashes and exact-name member inventories are checked here.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import numpy as np


class UnionSources:
    def __init__(self, receipt_path, specification_path, low_shape, full_shape,
                 primary_low, primary_full, edits, global_edit):
        from xta_native_filter import (Nrrd, child, digest, load_json,
                                       manifest_entries, shape3)
        self.low_shape, self.full_shape = tuple(low_shape), tuple(full_shape)
        self.receipt_path = Path(receipt_path).resolve()
        self.specification_path = Path(specification_path).resolve()
        self.receipt = load_json(self.receipt_path)
        spec = load_json(self.specification_path)
        expected_receipt = spec.get('union_receipt_sha256')
        if expected_receipt is not None and digest(self.receipt_path) != expected_receipt:
            raise ValueError('Union receipt checksum mismatch.')
        if self.receipt.get('format') != 'xta-preview-exact-name-union/1':
            raise ValueError('Unsupported exact-name union receipt format.')
        if spec.get('format') != 'xta-native-union-sources-v1':
            raise ValueError('Unsupported native union source specification.')
        basis = spec.get('identification_basis')
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError('Union specification must state its identification_basis.')
        groups = self.receipt.get('groups', [])
        self.groups = {g['exact_name']: g for g in groups}
        if len(self.groups) != len(groups) or set(self.groups) != {e['name'] for e in edits}:
            raise ValueError('Union receipt exact-name inventory differs from the edit bundle.')
        self.current_entries = {e['name']: e for e in edits}
        self.reference_dir, self.reference_entries = None, None
        self.reference_sha256, self.reference_audit = None, {}
        reference = spec.get('union_reference_edits')
        reference_sha = spec.get('union_reference_edits_sha256')
        if bool(reference) != bool(reference_sha):
            raise ValueError('Union reference edit directory and edits.json checksum must be supplied together.')
        if reference:
            from xta_native_filter import FORMAT
            reference_dir = Path(reference)
            if not reference_dir.is_absolute():
                reference_dir = self.specification_path.parent / reference_dir
            self.reference_dir = reference_dir.resolve()
            reference_path = self.reference_dir/'edits.json'
            if digest(reference_path) != reference_sha:
                raise ValueError('Union reference edits.json checksum mismatch.')
            baseline = load_json(reference_path)
            if (baseline.get('format') != FORMAT
                    or baseline.get('source_manifest_sha256') != digest(Path(primary_low))):
                raise ValueError('Union reference bundle does not identify the original primary preview manifest.')
            reference_entries = baseline.get('segments', [])
            self.reference_entries = {e['name']: e for e in reference_entries}
            if (len(self.reference_entries) != len(reference_entries)
                    or set(self.reference_entries) != set(self.current_entries)):
                raise ValueError('Union reference segment inventory differs from the current export.')
            for name, entry in self.reference_entries.items():
                current = self.current_entries[name]
                for field in ('source_filename', 'source_sha256'):
                    if entry.get(field) != current.get(field):
                        raise ValueError(f'Union reference original source identity mismatch: {name} ({field}).')
            self.reference_sha256 = reference_sha
        members = {}
        for group in groups:
            for member in group['members']:
                index = member['segment_index']
                if (isinstance(index, bool) or not isinstance(index, int) or index in members
                        or member['segment_name'] != group['exact_name']):
                    raise ValueError('Union receipt has duplicate/invalid member indices or names.')
                members[index] = member
        self.pairs, used = [], set()
        labels = set()
        for pair in spec.get('pairs', []):
            label = pair['label']
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError('Union source labels must be unique nonempty strings.')
            labels.add(label)
            paths, manifests, entries = {}, {}, {}
            for kind in ('low', 'full'):
                path = Path(pair[kind + '_manifest'])
                if not path.is_absolute():
                    path = self.specification_path.parent / path
                path = path.resolve()
                expected = pair[kind + '_manifest_sha256']
                if digest(path) != expected:
                    raise ValueError(f'Union {label} {kind} manifest checksum mismatch.')
                manifest = load_json(path)
                wanted = self.low_shape if kind == 'low' else self.full_shape
                if shape3(manifest['output_shape_tyx']) != wanted:
                    raise ValueError(f'Union {label} {kind} manifest shape mismatch.')
                if (kind == 'low') != (manifest.get('quality') == 'low_quality'):
                    raise ValueError(f'Union {label} {kind} manifest quality mismatch.')
                if kind == 'low' and shape3(manifest['full_quality_output_shape_tyx']) != self.full_shape:
                    raise ValueError(f'Union {label} low/full geometry mismatch.')
                by_name = {}
                for entry in manifest_entries(manifest).values():
                    name = entry['segment_name']
                    if name in by_name:
                        raise ValueError(f'Union {label} has duplicate exact segment names.')
                    by_name[name] = entry
                paths[kind], manifests[kind], entries[kind] = path, manifest, by_name
            if set(entries['low']) != set(entries['full']):
                raise ValueError(f'Union {label} low/full layer inventories differ.')
            indices = pair['receipt_member_indices']
            if (not isinstance(indices, list) or any(isinstance(i, bool) or not isinstance(i, int)
                                                   for i in indices)
                    or len(set(indices)) != len(indices) or used.intersection(indices)
                    or any(i not in members for i in indices)):
                raise ValueError('Union member assignments must form a unique receipt partition.')
            assigned_names = [members[i]['segment_name'] for i in indices]
            if len(set(assigned_names)) != len(assigned_names) or set(assigned_names) != set(entries['low']):
                raise ValueError(f'Union {label} assigned member-name inventory differs from its source manifest.')
            used.update(indices)
            globals_ = [e for e in entries['low'].values() if e.get('suffix') == 'Global_final_output']
            if len(globals_) != 1 or globals_[0]['segment_name'] != global_edit['name']:
                raise ValueError(f'Union {label} must have the matching single original Global.')
            for name, entry in entries['low'].items():
                other = entries['full'][name]
                for field in ('suffix', 'segment_name', 'layer_key', 'model_name', 'recomposition_op'):
                    if entry.get(field) != other.get(field):
                        raise ValueError(f'Union {label} low/full layer identity mismatch: {name} ({field}).')
                if name != global_edit['name'] and entry.get('recomposition_op') != 'union':
                    raise ValueError(f'Union source contains a nonadditive layer: {label}: {name}')
                Nrrd(child(paths['low'].parent, entry['filename']), self.low_shape)
                Nrrd(child(paths['full'].parent, other['filename']), self.full_shape)
            self.pairs.append(dict(label=label, paths=paths, entries=entries,
                                   global_name=global_edit['name'],
                                   manifest_sha256={k: pair[k + '_manifest_sha256'] for k in paths},
                                   receipt_member_indices=indices))
        if len(self.pairs) < 2 or used != set(members):
            raise ValueError('Union specification must partition all receipt members among at least two sources.')
        if not any(p['paths']['low'] == Path(primary_low).resolve()
                   and p['paths']['full'] == Path(primary_full).resolve() for p in self.pairs):
            raise ValueError('Primary original manifest pair is absent from union sources.')
        self.global_name = global_edit['name']
        self.primary_low = Path(primary_low).resolve()
        self.identification_basis = basis
        self._global_cache, self._hash_cache = {}, {}
        self._native_global_cache = {}
        self._layer_low_cache = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_global_cache'], state['_hash_cache'] = {}, {}
        state['_layer_low_cache'] = None
        return state

    @property
    def native_global_cache_bytes(self):
        z, y, x = self.full_shape
        return len(self.pairs) * z * y * ((x + 7) // 8)

    def prepare_native_globals(self, directory, depth):
        """Decode/validate each original Global once; share packed files on spawn."""
        from xta_native_filter import PreviewSupportCheck
        directory = Path(directory)
        for index, pair in enumerate(self.pairs):
            original = self.low_global(pair)
            check = PreviewSupportCheck(self.full_shape, original)
            path = directory / f'original-global-{index}.bits'
            count = 0
            with open(path, 'xb') as stream:
                for z, block in self.source(pair, 'full', self.global_name).slabs(depth):
                    check.observe(z, block)
                    count += int(np.count_nonzero(block))
                    stream.write(np.packbits(block, axis=-1, bitorder='little').tobytes())
            if bool(np.any(original)) != bool(count):
                raise ValueError(f'Union {pair["label"]} original native/preview Global emptiness disagrees.')
            self._native_global_cache[pair['label']] = dict(
                path=path, foreground_voxels=count, support_check=check.finish())

    def global_slabs(self, pair, depth):
        cached = self._native_global_cache.get(pair['label'])
        if cached is None:
            yield from self.source(pair, 'full', self.global_name).slabs(depth)
            return
        zsize, _, xsize = self.full_shape
        with self.global_packed(pair) as packed:
            for z in range(0, zsize, depth):
                yield z, np.unpackbits(packed[z:z+depth], axis=-1,
                                       count=xsize, bitorder='little')

    @contextmanager
    def global_packed(self, pair):
        cached = self._native_global_cache[pair['label']]
        zsize, ysize, xsize = self.full_shape
        packed = np.memmap(cached['path'], mode='r', dtype=np.uint8,
                           shape=(zsize, ysize, (xsize + 7) // 8))
        try:
            yield packed
        finally:
            # Explicit close is necessary before Windows TemporaryDirectory cleanup.
            packed._mmap.close()

    def source(self, pair, kind, name):
        from xta_native_filter import Nrrd, child
        return Nrrd(child(pair['paths'][kind].parent,
                          pair['entries'][kind][name]['filename']),
                    self.low_shape if kind == 'low' else self.full_shape)

    def low_global(self, pair):
        label = pair['label']
        if label not in self._global_cache:
            self._global_cache[label] = self.source(pair, 'low', self.global_name).array()
        return self._global_cache[label]

    def low(self, name):
        result = np.zeros(self.low_shape, np.uint8)
        originals = {}
        for pair in self.pairs:
            if name not in pair['entries']['low']:
                continue
            block = self.source(pair, 'low', name).array()
            originals[pair['label']] = block
            if name != self.global_name:
                global_ = self.low_global(pair)
                if np.any(global_):
                    block = block & global_
            np.bitwise_or(result, block, out=result)
        # Only one layer is retained; native iteration consumes it immediately.
        self._layer_low_cache = (name, originals) if name != self.global_name else None
        return result

    def take_layer_low(self, name):
        cached, self._layer_low_cache = self._layer_low_cache, None
        return cached[1] if cached is not None and cached[0] == name else {}

    def verify_edit(self, name, after, corrected_global=None):
        """Accept the exact receipt or strictly verified Update Both clipping."""
        if name == self.global_name:
            return
        baseline = after
        if self.reference_entries is not None:
            from xta_native_filter import Nrrd, child, load_edit
            entry = self.reference_entries[name]
            original = Nrrd(child(self.primary_low.parent, entry['source_filename']), self.low_shape)
            baseline = load_edit(self.reference_dir, entry, original)
        group = self.groups[name]
        extent = np.asarray(group['reference_extent_xyz'])
        if (extent.shape != (6,) or not np.issubdtype(extent.dtype, np.integer)
                or np.any(extent[1::2] < extent[::2])):
            raise ValueError('Union receipt contains an invalid output extent.')
        start, stop = extent[::2][::-1], extent[1::2][::-1] + 1
        if np.any(start < 0) or np.any(stop > self.low_shape):
            raise ValueError('Union receipt extent is outside the original preview lattice.')
        if tuple(stop - start) != tuple(group['shape_zyx']):
            raise ValueError('Union receipt extent/shape mismatch.')
        reference = np.asarray(self.receipt['reference_ijk_to_ras'], dtype=float)
        if not np.allclose(reference, np.diag([-1., -1., 1., 1.]), rtol=0, atol=1e-7):
            raise ValueError('Union receipt reference geometry differs from the original XTA lattice.')
        box = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        crop = baseline[box]
        count = int(np.count_nonzero(crop))
        h = hashlib.sha256()
        for frame in crop:
            h.update(np.ascontiguousarray(frame).tobytes())
        if (count != int(np.count_nonzero(baseline)) or count != group['foreground_voxels']
                or h.hexdigest() != group['binary_payload_sha256']):
            raise ValueError(f'Edited nonglobal mask or union reference differs from the union receipt: {name}')
        if self.reference_entries is not None:
            if corrected_global is None or corrected_global.shape != after.shape:
                raise ValueError('Union reference transfer requires the current corrected Global mask.')
            if not np.array_equal(after, baseline & corrected_global):
                raise ValueError(f'Edited nonglobal mask is not exactly union reference AND corrected Global: {name}')
            edited_sha = hashlib.sha256()
            for frame in after[box]:
                edited_sha.update(np.ascontiguousarray(frame).tobytes())
            new_count = int(np.count_nonzero(after))
            self.reference_audit[name] = dict(
                baseline_snapshot_sha256=self.reference_entries[name]['sha256'],
                edited_snapshot_sha256=self.current_entries[name]['sha256'],
                baseline_receipt_binary_payload_sha256=group['binary_payload_sha256'],
                edited_binary_payload_sha256=edited_sha.hexdigest(),
                baseline_foreground_voxels=count, edited_foreground_voxels=new_count,
                deleted_voxels=count-new_count,
                policy='current nonglobal equals receipt-verified baseline AND corrected exported Global')

    def native(self, name):
        return NativeUnion(self, name)

    def sha(self, path):
        from xta_native_filter import digest
        path = str(path)
        if path not in self._hash_cache:
            self._hash_cache[path] = digest(Path(path))
        return self._hash_cache[path]

    def report(self):
        from xta_native_filter import digest
        report = dict(receipt_sha256=digest(self.receipt_path),
                    source_specification_sha256=digest(self.specification_path),
                    identification_basis=self.identification_basis,
                    ancestry='inventory-based source identification; not cryptographic proof of edited-member ancestry',
                    operation='clip each nonempty original Global, preserve the legacy empty-Global bypass, then exact-name Boolean union before native labeling',
                    pairs=[dict(label=p['label'], low_manifest=str(p['paths']['low']),
                                full_manifest=str(p['paths']['full']),
                                manifest_sha256=p['manifest_sha256'],
                                receipt_member_indices=p['receipt_member_indices']) for p in self.pairs])
        if self.reference_dir is not None:
            report['union_reference_edits'] = str(self.reference_dir)
            report['union_reference_edits_sha256'] = self.reference_sha256
            report['reference_policy'] = 'receipt-verified baseline AND corrected exported Global, with no other nonglobal changes'
        return report


class NativeUnion:
    """One bounded slab per source plus its own Global; no full source rasters."""
    def __init__(self, sources, name):
        self.sources, self.name = sources, name
        self.shape = sources.full_shape
        self.pairs = [p for p in sources.pairs if name in p['entries']['full']]
        self.header = sources.source(self.pairs[0], 'full', name).header
        self._checks = None
        self._cached_global_checks = None

    def descriptor(self):
        rows = []
        for pair in self.pairs:
            native = self.sources.source(pair, 'full', self.name)
            row = dict(label=pair['label'], file=str(native.path), sha256=self.sources.sha(native.path))
            low = self.sources.source(pair, 'low', self.name)
            row['original_preview'] = dict(file=str(low.path), sha256=self.sources.sha(low.path))
            if self.name != self.sources.global_name:
                global_ = self.sources.source(pair, 'full', self.sources.global_name)
                row['original_global'] = dict(file=str(global_.path), sha256=self.sources.sha(global_.path))
                row['clipping_applied'] = bool(np.any(self.sources.low_global(pair)))
                row['clipping_reason'] = ('nonempty_original_global' if row['clipping_applied'] else
                                          'empty_original_global_legacy_bypass')
            else:
                row['clipping_applied'] = False
                row['clipping_reason'] = 'original_global_source_is_not_clipped'
            rows.append(row)
        return dict(operation='boolean_union_after_each_nonempty_original_global_clip', sources=rows)

    def slabs(self, depth):
        if (self.name != self.sources.global_name
                and all(p['label'] in self.sources._native_global_cache for p in self.pairs)):
            yield from self._packed_slabs(depth)
            return
        from xta_native_filter import PreviewSupportCheck
        streams, clips, checks = [], [], []
        self._checks = checks
        originals = self.sources.take_layer_low(self.name)
        cached_global = (self.name == self.sources.global_name
                         and all(p['label'] in self.sources._native_global_cache for p in self.pairs))
        if cached_global:
            self._cached_global_checks = {
                p['label']: self.sources._native_global_cache[p['label']]['support_check']
                for p in self.pairs}
        try:
            for pair in self.pairs:
                streams.append(iter(self.sources.global_slabs(pair, depth) if cached_global else
                    self.sources.source(pair, 'full', self.name).slabs(depth)))
                original = originals.pop(pair['label'], None)
                if original is None and not cached_global:
                    original = self.sources.source(pair, 'low', self.name).array()
                checks.append((pair['label'], None if cached_global else
                               PreviewSupportCheck(self.shape, original)))
                has_global = bool(np.any(self.sources.low_global(pair)))
                clips.append(iter(self.sources.global_slabs(pair, depth))
                             if self.name != self.sources.global_name and has_global else None)
            for z in range(0, self.shape[0], depth):
                out = np.zeros((min(depth, self.shape[0]-z), *self.shape[1:]), np.uint8)
                for stream, clip, (_, check) in zip(streams, clips, checks):
                    zz, block = next(stream)
                    if zz != z or block.shape != out.shape:
                        raise ValueError('Union source slabs are not synchronized.')
                    if check is not None:
                        check.observe(z, block)
                    if clip is not None:
                        cz, global_ = next(clip)
                        if cz != z or global_.shape != block.shape:
                            raise ValueError('Union original Global slabs are not synchronized.')
                        block &= global_
                    out |= block
                yield z, out
            # Exhaust readers to enforce NRRD trailing-data/CRC checks too.
            for stream in streams + [c for c in clips if c is not None]:
                if next(stream, None) is not None:
                    raise ValueError('Union source has extra slabs.')
        finally:
            for stream in streams + [c for c in clips if c is not None]:
                stream.close()

    def _packed_slabs(self, depth):
        """Clip and union in packed space; unpack once per output slab."""
        from xta_native_filter import PreviewSupportCheck
        originals = self.sources.take_layer_low(self.name)
        self._checks, self._cached_global_checks = [], None
        streams, clips = [], []
        with ExitStack() as stack:
            for pair in self.pairs:
                stream = self.sources.source(pair, 'full', self.name).slabs(depth)
                streams.append(stack.enter_context(closing(stream)))
                original = originals.pop(pair['label'], None)
                if original is None:
                    original = self.sources.source(pair, 'low', self.name).array()
                self._checks.append((pair['label'], PreviewSupportCheck(self.shape, original)))
                has_global = bool(np.any(self.sources.low_global(pair)))
                clips.append(stack.enter_context(self.sources.global_packed(pair)) if has_global else None)
            for z in range(0, self.shape[0], depth):
                shape = (min(depth, self.shape[0]-z), *self.shape[1:])
                packed = np.zeros((*shape[:-1], (shape[-1]+7)//8), np.uint8)
                for stream, clip, (_, check) in zip(streams, clips, self._checks):
                    zz, block = next(stream)
                    if zz != z or block.shape != shape:
                        raise ValueError('Union source slabs are not synchronized.')
                    # Validate the raw source before clipping. Empty blocks still
                    # pass through the reader's binary/CRC/truncation checks.
                    if not check.observe(z, block):
                        continue
                    contribution = np.packbits(block, axis=-1, bitorder='little')
                    if clip is not None:
                        contribution &= clip[z:z+len(block)]
                    packed |= contribution
                yield z, np.unpackbits(packed, axis=-1, count=shape[-1], bitorder='little')
            for stream in streams:
                if next(stream, None) is not None:
                    raise ValueError('Union source has extra slabs.')

    def observe(self, z, block):
        # Individual raw source checks are observed before their own Global clip.
        pass

    def finish(self):
        if self._checks is None:
            raise ValueError('Union native source was not consumed.')
        if self._cached_global_checks is not None:
            return dict(contract='per-source original positive-support checks before original Global clipping',
                        sources=self._cached_global_checks)
        if self.name == self.sources.global_name:
            for label, check in self._checks:
                if bool(np.any(check.original)) != bool(np.any(check.seen)):
                    raise ValueError(f'Union {label} original native/preview Global emptiness disagrees.')
        return dict(contract='per-source original positive-support checks before original Global clipping',
                    sources={label: check.finish() for label, check in self._checks})
