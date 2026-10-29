"""Mixed-confidence recovery must retain source detail and strict edit evidence."""
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from tests.test_native_filter import fixture, snapshot
import xta_native_filter as nf
from xta_native_union import UnionSources


def union_fixture(root, *, per_source_clip=False, length=10, empty_first_global=False, extra_layer=False):
    root = Path(root)
    shape = (3, 5, length)
    a, b = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    ga, gb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    if per_source_clip:
        a[1, 2, 1:9] = 1
        b[1, 2, 4:8] = 1
        ga[1, 2, 1:4] = 1
        gb[1, 2, 4:9] = 1
    else:
        a[1, 2, 1:3] = a[1, 2, length-4:length-2] = 1
        b[1, 2, 3:length-4] = 1
        ga[:] = gb[:] = 1
    if empty_first_global:
        ga.fill(0)
        gb[:] = b
    after = (a if not np.any(ga) else a & ga) | (b if not np.any(gb) else b & gb)
    masks_a, masks_b, edits = {'component': a}, {'component': b}, {'component': after}
    if extra_layer:
        masks_a['other'], masks_b['other'], edits['other'] = a.copy(), b.copy(), after.copy()
    first = fixture(root/'first', masks_a, edits,
                    global_mask=ga, low_shape=shape)
    second = fixture(root/'second', masks_b, edits,
                     global_mask=gb, low_shape=shape)
    groups = []
    bundle = nf.load_json(Path(first.edits)/'edits.json')
    segment_count = len(bundle['segments'])
    # The original union receipt may have a stale Global. The actual bundle is
    # subsequently exported after Update Both; only Global may differ here.
    for index, entry in enumerate(bundle['segments']):
        mask = after if index < segment_count-1 else np.zeros(shape, np.uint8)
        groups.append(dict(exact_name=entry['name'], is_global_name=index == segment_count-1,
                           members=[dict(segment_name=entry['name'], segment_index=index),
                                    dict(segment_name=entry['name'], segment_index=index+segment_count)],
                           reference_extent_xyz=[0, length-1, 0, 4, 0, 2], shape_zyx=list(shape),
                           foreground_voxels=int(mask.sum()),
                           binary_payload_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    receipt = root/'union-receipt.json'
    nf.save_json(receipt, dict(format='xta-preview-exact-name-union/1', groups=groups,
                              reference_ijk_to_ras=np.diag([-1., -1., 1., 1.]).tolist()))
    pairs = []
    for index, source in enumerate((first, second)):
        pairs.append(dict(label=f'confidence_{index}', low_manifest=source.low_manifest,
                          low_manifest_sha256=nf.digest(Path(source.low_manifest)),
                          full_manifest=source.full_manifest,
                          full_manifest_sha256=nf.digest(Path(source.full_manifest)),
                          receipt_member_indices=list(range(segment_count*index, segment_count*(index+1)))))
    spec = root/'source-pairs.json'
    nf.save_json(spec, dict(format='xta-native-union-sources-v1', pairs=pairs,
                           identification_basis='Explicit fixture inventories; not edited-member hash ancestry'))
    first.union_receipt, first.union_source_pairs = str(receipt), str(spec)
    first.output, first.scratch = str(root/'output'), str(root/'scratch')
    return first, after


def replace_export_masks(args, mask):
    path = Path(args.edits)/'edits.json'
    bundle = nf.load_json(path)
    for entry in bundle['segments']:
        saved = snapshot(Path(args.edits)/entry['file'], mask)
        entry.update(sha256=saved['sha256'], foreground_voxels=saved['foreground_voxels'])
    nf.save_json(path, bundle)


def bind_reference(args):
    path = Path(args.union_source_pairs)
    reference = path.parent/'union-reference-edits'
    shutil.copytree(args.edits, reference)
    spec = nf.load_json(path)
    spec.update(union_reference_edits=reference.name,
                union_reference_edits_sha256=nf.digest(reference/'edits.json'),
                union_receipt_sha256=nf.digest(Path(args.union_receipt)))
    nf.save_json(path, spec)
    return reference


class NativeUnionTests(unittest.TestCase):
    def test_packed_union_matches_uncached_clipping_for_row_padding_and_empty_slabs(self):
        for width in (10, 15, 16, 17, 31):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as folder:
                args, expected = union_fixture(folder, length=width, per_source_clip=True)
                bundle = nf.load_json(Path(args.edits)/'edits.json')
                sources = UnionSources(args.union_receipt, args.union_source_pairs, expected.shape,
                                       expected.shape, args.low_manifest, args.full_manifest,
                                       bundle['segments'], bundle['segments'][-1])
                legacy = sources.native('volume_component')
                uncached = np.concatenate([a for _, a in legacy.slabs(1)])
                legacy.finish()
                sources.prepare_native_globals(folder, 1)
                native = sources.native('volume_component')
                unpack = np.unpackbits
                pack = np.packbits
                with patch.object(sources, 'global_slabs', side_effect=AssertionError('Global was unpacked')):
                    with patch('xta_native_union.np.unpackbits', wraps=unpack) as unpacks:
                        with patch('xta_native_union.np.packbits', wraps=pack) as packs:
                            actual = np.concatenate([a for _, a in native.slabs(1)])
                native.finish()
                # Three output slabs, two occupied source slabs. The four
                # empty source slabs are still read/checked but never packed.
                self.assertEqual(unpacks.call_count, 3)
                self.assertEqual(packs.call_count, 2)
                np.testing.assert_array_equal(actual, uncached)
                np.testing.assert_array_equal(actual, expected)
                for pair in sources.pairs:
                    with sources.global_packed(pair) as packed:
                        self.assertFalse(packed.flags.writeable)
                for path in Path(folder).glob('original-global-*.bits'):
                    path.unlink()  # Every mmap is closed, including on Windows.

    def test_packed_empty_source_still_enforces_reader_crc_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder)
            bundle = nf.load_json(Path(args.edits)/'edits.json')
            sources = UnionSources(args.union_receipt, args.union_source_pairs, expected.shape,
                                   expected.shape, args.low_manifest, args.full_manifest,
                                   bundle['segments'], bundle['segments'][-1])
            sources.prepare_native_globals(folder, 1)
            source = sources.source(sources.pairs[0], 'full', 'volume_component')
            data = source.path.read_bytes()
            compressed = bytearray(gzip.compress(bytes(np.prod(expected.shape))))
            compressed[-8] ^= 1
            source.path.write_bytes(data[:source.offset] + compressed)
            with self.assertRaises((OSError, EOFError, ValueError)):
                list(sources.native('volume_component').slabs(1))
            for path in Path(folder).glob('original-global-*.bits'):
                path.unlink()

    def test_original_globals_decode_once_and_packed_cache_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder, extra_layer=True)
            original_slabs = nf.Nrrd.slabs
            decoded = {}

            def counted(source, depth):
                if source.path.parent.name == 'full' and 'Global_final_output' in source.path.name:
                    decoded[str(source.path)] = decoded.get(str(source.path), 0) + 1
                yield from original_slabs(source, depth)

            with patch.object(nf.Nrrd, 'slabs', counted):
                self.assertEqual(nf.run(args), 0)
            self.assertEqual(sorted(decoded.values()), [1, 1])
            report = nf.load_json(Path(args.output)/'report.json')
            self.assertEqual(report['native_global_cache_bytes'], 2*3*5*2)
            np.testing.assert_array_equal(nf.Nrrd(Path(args.output)/'volume.nrrd').array(), expected)
            self.assertFalse(list(Path(args.scratch).iterdir()))

    def test_cache_budget_includes_row_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = union_fixture(folder)
            # Two150-byte rasters and256MiB reserve fit; the60-byte packed
            # cache does not. Refuse before creating output or native payloads.
            available = 2*150 + 256*nf.MIB + 59
            with patch.object(nf.shutil, 'disk_usage', return_value=SimpleNamespace(free=available)):
                with self.assertRaisesRegex(ValueError, 'Scratch needs at least'):
                    nf.run(args)
            self.assertFalse(Path(args.output).exists())

    def test_cache_preserves_original_global_payload_validation_and_cleans_failure(self):
        for corruption in ('nonbinary', 'truncated', 'bad_crc', 'extra', 'missing_support', 'empty_mismatch'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                args, _ = union_fixture(folder)
                target = Path(args.full_manifest).parent/'volume_Global_final_output.seg.nrrd'
                _, offset, shape = nf.read_header(target)
                data = target.read_bytes()
                payload = bytearray(gzip.decompress(data[offset:]))
                if corruption == 'nonbinary':
                    payload[0] = 2
                elif corruption == 'truncated':
                    payload = payload[:-1]
                elif corruption == 'extra':
                    payload.append(0)
                elif corruption == 'missing_support':
                    payload[0] = 0
                elif corruption == 'empty_mismatch':
                    payload[:] = bytes(len(payload))
                if corruption == 'bad_crc':
                    corrupted = bytearray(data)
                    corrupted[-8] ^= 1
                    target.write_bytes(corrupted)
                else:
                    target.write_bytes(data[:offset] + gzip.compress(payload))
                with self.assertRaises((ValueError, OSError, EOFError)):
                    nf.run(args)
                self.assertFalse((Path(args.output)/'volume.nrrd').exists())
                self.assertFalse(list(Path(args.scratch).iterdir()))

    def test_raw_low_layer_masks_are_reused_only_for_the_current_layer(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = union_fixture(folder)
            bundle = nf.load_json(Path(args.edits)/'edits.json')
            sources = UnionSources(args.union_receipt, args.union_source_pairs, (3,5,10), (3,5,10),
                                   args.low_manifest, args.full_manifest,
                                   bundle['segments'], bundle['segments'][-1])
            original_array = nf.Nrrd.array
            reads = {}

            def counted(source):
                if source.path.parent.name == 'low' and source.path.name == 'volume_component.seg.nrrd':
                    reads[str(source.path)] = reads.get(str(source.path), 0) + 1
                return original_array(source)

            with patch.object(nf.Nrrd, 'array', counted):
                sources.low('volume_component')
                self.assertIsNotNone(sources._layer_low_cache)
                native = sources.native('volume_component')
                list(native.slabs(1))
                native.finish()
            self.assertEqual(sorted(reads.values()), [1, 1])
            self.assertIsNone(sources._layer_low_cache)

    def test_bridge_union_before_native_selection_and_corrected_global(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder)
            self.assertEqual(nf.run(args), 0)
            report = nf.load_json(Path(args.output)/'report.json')
            actual = nf.Nrrd(Path(args.output)/report['pta_label_file']).array()
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(report['final_global_voxels'], 7)
            self.assertEqual(report['union_cleanup']['input_components'], 1)
            self.assertEqual(report['source_full_sha256'], {})
            self.assertEqual(len(report['source_full_unions']), 2)
            self.assertIn('not cryptographic', report['source_union']['ancestry'])
            self.assertFalse(any(Path(args.scratch).iterdir()))

    def test_union_worker_processes_match_single_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder, extra_layer=True)
            args.workers = 2
            self.assertEqual(nf.run(args), 0)
            actual = nf.Nrrd(Path(args.output)/'volume.nrrd').array()
            np.testing.assert_array_equal(actual, expected)
            report = nf.load_json(Path(args.output)/'report.json')
            self.assertEqual(len(report['layers']), 2)
            self.assertEqual(report['native_global_cache_bytes'], 60)
            self.assertFalse(list(Path(args.scratch).iterdir()))

    def test_each_sources_global_is_applied_before_union(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder, per_source_clip=True)
            self.assertEqual(nf.run(args), 0)
            actual = nf.Nrrd(Path(args.output)/'volume.nrrd').array()
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(actual[1, 2, 8]), 0)

    def test_empty_source_global_bypass_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder, empty_first_global=True)
            self.assertEqual(nf.run(args), 0)
            report = nf.load_json(Path(args.output)/'report.json')
            actual = nf.Nrrd(Path(args.output)/'volume.nrrd').array()
            np.testing.assert_array_equal(actual, expected)
            # Source A contributes both ends outside the aggregate original
            # Global (the bridge alone). A second aggregate clip would lose them.
            self.assertEqual(int(actual.sum()), 7)
            self.assertEqual(report['initial_global_voxels'], 3)
            rows = report['source_full_unions']['volume_component.seg.nrrd']['sources']
            self.assertFalse(rows[0]['clipping_applied'])
            self.assertEqual(rows[0]['clipping_reason'], 'empty_original_global_legacy_bypass')
            self.assertTrue(rows[1]['clipping_applied'])

    def test_changed_nonglobal_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder)
            bundle_path = Path(args.edits)/'edits.json'
            bundle = nf.load_json(bundle_path)
            changed = expected.copy()
            changed[1, 2, 1] = 0
            for entry in bundle['segments']:
                new = snapshot(Path(args.edits)/entry['file'], changed)
                entry.update(sha256=new['sha256'], foreground_voxels=new['foreground_voxels'])
            nf.save_json(bundle_path, bundle)
            with self.assertRaisesRegex(ValueError, 'differs from the union receipt'):
                nf.run(args)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())

    def test_union_does_not_relax_unsupported_edit_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            args, expected = union_fixture(folder, per_source_clip=True)
            # Make the receipt and edited bundle consistently include a voxel
            # present only outside its own run's Global. Another run's Global
            # covers it, which must not make it supported.
            changed = expected.copy()
            changed[1, 2, 8] = 1
            bundle_path = Path(args.edits)/'edits.json'
            bundle = nf.load_json(bundle_path)
            for entry in bundle['segments']:
                new = snapshot(Path(args.edits)/entry['file'], changed)
                entry.update(sha256=new['sha256'], foreground_voxels=new['foreground_voxels'])
            nf.save_json(bundle_path, bundle)
            receipt = nf.load_json(Path(args.union_receipt))
            receipt['groups'][0].update(foreground_voxels=int(changed.sum()),
                                       binary_payload_sha256=hashlib.sha256(changed.tobytes()).hexdigest())
            nf.save_json(Path(args.union_receipt), receipt)
            with self.assertRaisesRegex(ValueError, 'Deletion-only transfer refused'):
                nf.run(args)

    def test_manifest_hash_and_member_inventory_are_strict(self):
        for corruption in ('hash', 'inventory', 'duplicate_index'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as folder:
                args, _ = union_fixture(folder)
                path = Path(args.union_source_pairs)
                spec = nf.load_json(path)
                if corruption == 'hash':
                    spec['pairs'][1]['full_manifest_sha256'] = '0'*64
                elif corruption == 'inventory':
                    spec['pairs'][1]['receipt_member_indices'] = [2]
                else:
                    spec['pairs'][1]['receipt_member_indices'] = [0, 3]
                nf.save_json(path, spec)
                with self.assertRaises(ValueError):
                    nf.run(args)
                self.assertFalse(Path(args.output).exists())

    def test_stale_exported_global_still_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = union_fixture(folder)
            bundle_path = Path(args.edits)/'edits.json'
            bundle = nf.load_json(bundle_path)
            entry = bundle['segments'][-1]
            new = snapshot(Path(args.edits)/entry['file'], np.zeros((3,5,10), np.uint8))
            entry.update(sha256=new['sha256'], foreground_voxels=0)
            nf.save_json(bundle_path, bundle)
            with self.assertRaisesRegex(ValueError, 'Global differs from the union'):
                nf.run(args)

    def test_validate_only_reads_no_native_payload_and_needs_no_raster_space(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _ = union_fixture(folder)
            args.validate_only = True
            with patch('xta_native_union.NativeUnion.slabs', side_effect=AssertionError('native payload read')):
                with patch.object(nf.shutil, 'disk_usage', side_effect=AssertionError('native disk allocation')):
                    self.assertEqual(nf.run(args), 0)
            report = nf.load_json(Path(args.output)/'report.json')
            self.assertEqual(report['status'], 'validated_preview_only')
            self.assertIsNone(report['pta_label_file'])
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())

    def test_reference_allows_exact_update_both_clipping_of_39_voxels(self):
        with tempfile.TemporaryDirectory() as folder:
            args, baseline = union_fixture(folder, length=50)
            reference = bind_reference(args)
            reference_hash = nf.digest(reference/'edits.json')
            corrected = baseline.copy()
            corrected[1, 2, 9:48] = 0
            self.assertEqual(int(baseline.sum()-corrected.sum()), 39)
            replace_export_masks(args, corrected)
            args.validate_only = True
            self.assertEqual(nf.run(args), 0)
            report = nf.load_json(Path(args.output)/'report.json')
            audit = report['source_union']['edit_reference_audit']['volume_component']
            self.assertEqual(audit['deleted_voxels'], 39)
            self.assertEqual(audit['baseline_foreground_voxels'], 47)
            self.assertEqual(audit['edited_foreground_voxels'], 8)
            self.assertNotEqual(audit['baseline_snapshot_sha256'], audit['edited_snapshot_sha256'])
            self.assertEqual(nf.digest(reference/'edits.json'), reference_hash)

    def test_reference_refuses_reintroduction_of_previously_deleted_voxels(self):
        with tempfile.TemporaryDirectory() as folder:
            args, original = union_fixture(folder)
            baseline = original.copy()
            baseline[1, 2, 7] = 0
            replace_export_masks(args, baseline)
            receipt_path = Path(args.union_receipt)
            receipt = nf.load_json(receipt_path)
            receipt['groups'][0].update(foreground_voxels=int(baseline.sum()),
                                       binary_payload_sha256=hashlib.sha256(baseline.tobytes()).hexdigest())
            nf.save_json(receipt_path, receipt)
            bind_reference(args)
            replace_export_masks(args, original)
            args.validate_only = True
            with self.assertRaisesRegex(ValueError, 'not exactly union reference AND corrected Global'):
                nf.run(args)

    def test_reference_and_receipt_hashes_are_bound(self):
        for field in ('union_reference_edits_sha256', 'union_receipt_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                args, _ = union_fixture(folder)
                bind_reference(args)
                path = Path(args.union_source_pairs)
                spec = nf.load_json(path)
                spec[field] = '0'*64
                nf.save_json(path, spec)
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    nf.run(args)
                self.assertFalse(Path(args.output).exists())

    def test_reference_snapshot_hash_is_verified_before_clipping(self):
        with tempfile.TemporaryDirectory() as folder:
            args, mask = union_fixture(folder)
            reference = bind_reference(args)
            snapshot(reference/'00000.npz', np.zeros_like(mask))
            args.validate_only = True
            with self.assertRaisesRegex(ValueError, 'Edited snapshot checksum mismatch'):
                nf.run(args)


if __name__ == '__main__':
    unittest.main()
