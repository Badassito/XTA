"""Continuous-field preparation, reuse, derived crops and staged-pair verification."""
from __future__ import annotations

from fractions import Fraction
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import prepare_pta_input as prep
from tools import verify_pta_staged_pair as verifier
from tools import xta_cut_edits


PREVIEW = (6, 8, 8)
# The X ratio 17/8 differs from the derived crop's 11/5, so lattice errors show.
NATIVE = (12, 16, 17)
POLICY = 'guard-cap-weighted-cardinal-B3-v1'


def edited_global():
    t, y, x = np.indices(PREVIEW)
    blob = ((t - 2.5) / 2.2) ** 2 + ((y - 3.5) / 2.6) ** 2 + ((x - 3.0) / 2.4) ** 2 <= 1
    blob[3, 6, 1:7] = True
    return blob.astype(np.uint8)


def write_bundle(directory, stem, mask):
    directory.mkdir(parents=True)
    manifest = {'quality': 'low_quality', 'output_shape_tyx': list(mask.shape),
                'full_quality_output_shape_tyx': list(NATIVE), 'exported_axes': '(X, Y, t)',
                'layers': [{'filename': 'global.seg.nrrd', 'segment_name': stem + '_Global_final_output',
                            'suffix': 'Global_final_output', 'recomposition_op': 'select'}]}
    prep.save_json(directory / 'source_manifest.json', manifest)
    npz = directory / '00000.npz'
    t, y, x = mask.shape
    np.savez_compressed(npz, mask=mask, extent_xyz=np.array([0, x - 1, 0, y - 1, 0, t - 1]),
                        ijk_to_ras=prep.REFERENCE_RAS)
    prep.save_json(directory / 'edits.json', {
        'format': 'xta-slicer-edits-v1', 'source_manifest_name': stem + '_nrrd_manifest.json',
        'source_manifest_sha256': prep.digest(directory / 'source_manifest.json'),
        'preview_shape_tyx': list(mask.shape), 'full_quality_shape_tyx': list(NATIVE),
        'segments': [{'name': stem + '_Global_final_output', 'file': npz.name,
                      'source_filename': 'global.seg.nrrd', 'source_sha256': 'a' * 64,
                      'recomposition_op': 'select', 'sha256': prep.digest(npz),
                      'foreground_voxels': int(mask.sum())}]})


def read_nrrd(path):
    header, payload = path.read_bytes().split(b'\n\n', 1)
    fields = dict(line.split(': ', 1) for line in header.decode().splitlines() if ': ' in line)
    width, height, frames = map(int, fields['sizes'].split())
    return np.frombuffer(gzip.decompress(payload), np.uint8).reshape(frames, height, width)


def direct_reconstruction(directory, preview, native):
    """Run the package without the preparation wrapper."""
    module, _ = prep.continuous_field_contract()
    module.build_field_reference(preview, native, directory / 'field')
    module.write_upscaled_nrrd(directory / 'field', directory / 'direct.nrrd', native_shape_tyx=native, block_mib=1)
    return read_nrrd(directory / 'direct.nrrd')


@unittest.skipUnless(importlib.util.find_spec('scipy') and importlib.util.find_spec('skimage'),
                     'continuous_field requires SciPy and scikit-image')
class ContinuousFieldPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stem = 'sample'
        self.mask = edited_global()
        self.bundle = self.root / 'filtered' / 'v1.2' / 'dataset'
        write_bundle(self.bundle, self.stem, self.mask)
        self.video = self.root / 'images' / 'dataset' / 'dataset.mkv'
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'original intensity bytes')
        self.output = self.root / 'labels' / 'v1.2' / 'dataset'
        self.counter = 0

    def run_cli(self, *extra, bundle=None, output=None, native=NATIVE, mode='continuous_field'):
        self.counter += 1
        log = self.root / f'log-{self.counter}'
        args = [self.stem, '--training-root', str(self.root), '--generation', 'v1.2',
                '--bundle-dir', str(bundle or self.bundle), '--source-video', str(self.video),
                '--output-dir', str(output or self.output), '--output-stem', 'dataset',
                '--intensity-mode', 'reference', '--mode', mode, '--block-mib', '1',
                '--log-dir', str(log), *extra]
        with patch.object(prep, 'probe_source', return_value={'shape_tyx': list(native), 'frame_count_method': 'synthetic'}):
            prep.main(args)
        return log

    def stage_pair(self, receipt_dir):
        stage = self.root / f'staged-{self.counter}'
        stage.mkdir()
        shutil.copyfile(self.video, stage / 'dataset.mkv')
        shutil.copyfile(receipt_dir / 'dataset.nrrd', stage / 'dataset.nrrd')
        return receipt_dir / 'preparation.json', stage / 'dataset.mkv', stage / 'dataset.nrrd'

    def test_publishes_package_payload_reuses_and_verifies_staged_pair(self):
        log = self.run_cli()
        receipt = prep.read_json(self.output / 'preparation.json')
        identity, contract = receipt['identity'], receipt['identity']['continuous_field']
        self.assertEqual(identity['mode'], 'continuous_field')
        self.assertEqual((identity['policy'], contract['policy'], contract['version']), (POLICY, POLICY, 1))
        self.assertEqual(set(contract['sources_sha256']),
                         {'__init__.py', 'policy.py', 'field_reference.py', 'radius_upscaler.py', 'nrrd_writer.py'})
        self.assertEqual(set(contract['runtime']), {'python', 'numpy', 'scipy', 'scikit_image'})
        self.assertEqual((receipt['algorithm'], receipt['label_authority']), (POLICY, 'edited_preview_sampled_support'))
        self.assertTrue(receipt['continuous_reconstruction'])
        self.assertFalse(receipt['restored_native_detail'])
        self.assertEqual(identity['intensity']['sha256'], prep.digest(self.video))
        self.assertEqual(receipt['mask']['path'], str(self.output / 'dataset.nrrd'))
        self.assertTrue(receipt['mask']['validation']['gzip_readback_to_end'])
        self.assertEqual({path.name for path in self.output.iterdir()}, {'dataset.nrrd', 'preparation.json'})
        self.assertTrue((log / 'sample.field-reference.json').is_file())
        actual = read_nrrd(self.output / 'dataset.nrrd')
        self.assertEqual(int(actual.sum()), receipt['mask']['foreground_voxels'])
        self.assertGreater(receipt['mask']['foreground_voxels'], 0)
        np.testing.assert_array_equal(actual, direct_reconstruction(self.root, self.mask, NATIVE))

        again = self.run_cli()
        self.assertEqual(prep.read_json(again / 'sample.result.json')['status'], 'validated_existing')

        receipt_path, video, mask = self.stage_pair(self.output)
        result = verifier.verify(receipt_path, video, mask, 'continuous_field', 'dataset')
        self.assertEqual((result['status'], result['policy']), ('verified', POLICY))
        with self.assertRaisesRegex(ValueError, "'continuous_field' differs from expected 'edited_shape'"):
            verifier.verify(receipt_path, video, mask, 'edited_shape', 'dataset')
        with self.assertRaisesRegex(ValueError, 'output stem'):
            verifier.verify(receipt_path, video, mask, 'continuous_field', 'other')
        output = self.root / 'verification' / 'pair.json'
        verifier.main(['--receipt', str(receipt_path), '--video', str(video), '--mask', str(mask),
                       '--expected-mode', 'continuous_field', '--output', str(output)])
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['status'], 'verified')
        with self.assertRaises(FileExistsError):
            verifier.main(['--receipt', str(receipt_path), '--video', str(video), '--mask', str(mask),
                           '--expected-mode', 'continuous_field', '--output', str(output)])
        with self.assertRaises(SystemExit):
            verifier.main(['--receipt', str(receipt_path), '--video', str(video), '--mask', str(mask)])
        mask.write_bytes(mask.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Staged mask content differs'):
            verifier.verify(receipt_path, video, mask, 'continuous_field', 'dataset')

    def test_reuse_treats_code_and_runtime_as_provenance_but_not_policy(self):
        self.run_cli()
        module, contract = prep.continuous_field_contract()
        newer = dict(contract, sources_sha256={'policy.py': '0' * 64}, runtime={'numpy': 'newer'})
        with patch.object(prep, 'continuous_field_contract', return_value=(module, newer)):
            log = self.run_cli()
        self.assertEqual(prep.read_json(log / 'sample.result.json')['status'], 'validated_existing')
        other = dict(contract, policy='another-policy')
        with patch.object(prep, 'continuous_field_contract', return_value=(module, other)):
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                self.run_cli()

    def test_changed_payload_or_missing_writer_audit_is_refused(self):
        module, contract = prep.continuous_field_contract()
        original = module.write_upscaled_nrrd

        def without_audit(*args, **kwargs):
            result = original(*args, **kwargs)
            result.pop('validation')
            return result
        with patch.object(prep, 'continuous_field_contract', return_value=(module, contract)), \
                patch.object(module, 'write_upscaled_nrrd', side_effect=without_audit):
            with self.assertRaisesRegex(ValueError, 'complete native NRRD identity'):
                self.run_cli()
        self.assertFalse(self.output.exists())
        self.run_cli()
        nrrd = self.output / 'dataset.nrrd'
        header, payload = nrrd.read_bytes().split(b'\n\n', 1)
        values = bytearray(gzip.decompress(payload))
        values[0] ^= 1
        nrrd.write_bytes(header + b'\n\n' + gzip.compress(bytes(values), mtime=0))
        with self.assertRaisesRegex(ValueError, 'validation failed'):
            self.run_cli()

    def test_legacy_continuous_b3_receipt_is_neither_reused_nor_overwritten(self):
        self.run_cli()
        path = self.output / 'preparation.json'
        receipt = prep.read_json(path)
        identity = receipt['identity']
        identity['mode'] = 'continuous_b3'
        identity['continuous_b3'] = identity.pop('continuous_field')
        identity.update(label_generation='v1.2', dataset='dataset',
                        implementation={'bundle_helper': 'pta-continuous-b3-preparation/1', 'prepare_sha256': '0' * 64})
        prep.save_json(path, receipt)
        before = {item.name: item.read_bytes() for item in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, "provenance differs \\(prepared mode 'continuous_b3', requested 'continuous_field'\\)"):
            self.run_cli()
        self.assertEqual({item.name: item.read_bytes() for item in self.output.iterdir()}, before)
        receipt_path, video, mask = self.stage_pair(self.output)
        self.assertEqual(verifier.verify(receipt_path, video, mask, 'continuous_b3', 'dataset')['status'], 'verified')
        with self.assertRaisesRegex(ValueError, 'differs from expected'):
            verifier.verify(receipt_path, video, mask, 'continuous_field', 'dataset')
        identity.pop('continuous_b3')
        prep.save_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'policy identity is missing'):
            verifier.verify(receipt_path, video, mask, 'continuous_b3', 'dataset')
        receipt['identity'] = None
        prep.save_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'provenance differs; nothing overwritten'):
            self.run_cli()

    def test_field_cache_is_reused_across_outputs_and_guarded(self):
        cache = self.root / 'field-cache'
        first = self.run_cli('--field-cache', str(cache))
        self.assertTrue((cache / 'dataset' / 'COMPLETE.json').is_file())
        second_output = self.output.with_name('dataset-again')
        second = self.run_cli('--field-cache', str(cache), output=second_output)
        built = prep.read_json(first / 'sample.field-reference.json')
        validated = prep.read_json(second / 'sample.field-reference.json')
        self.assertEqual(validated['created_utc'], built['created_utc'])
        self.assertEqual((second_output / 'dataset.nrrd').read_bytes(), (self.output / 'dataset.nrrd').read_bytes())
        unused = self.root / 'labels' / 'v1.2' / 'unused'
        invalid = (('--field-cache', str(self.bundle / 'cache')),
                   ('--field-cache', str(unused / 'cache')),
                   ('--field-cache', str(self.video.parent)),
                   ('--field-cache', str(cache), '--mode', 'edited_shape'),
                   ('--mask-source', 'components'),
                   ('--field-memory-mib', '0'))
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.run_cli(*args, output=unused)
        self.assertFalse(unused.exists())

    def test_derived_axial_crop_reconstructs_on_original_lattice(self):
        info = xta_cut_edits.inspect_edits(self.bundle)
        marker = (Fraction(-9, 2), Fraction(-3), Fraction(2))
        self.assertEqual(prep.preview_cut_columns(marker, PREVIEW[::-1], NATIVE[::-1]), (5, 11))
        cropped = self.bundle.with_name('dataset-cut')
        xta_cut_edits.write_edits(self.bundle, cropped, self.root / 'cut-preview.nrrd', info, 5, 11, marker)
        cropped_native = (*NATIVE[:2], 11)
        output = self.output.with_name('dataset-cut')
        self.run_cli(bundle=cropped, output=output, native=cropped_native)
        receipt = prep.read_json(output / 'preparation.json')
        self.assertEqual(receipt['identity']['bundle']['bundle_format'], 'xta-slicer-edits-axial-crop-v1')
        self.assertEqual(receipt['axial_crop'], receipt['mask']['axial_crop'])
        self.assertEqual(receipt['mask']['crop_audit']['retained_width'], 11)
        self.assertEqual(receipt['mask']['pre_crop_mask']['validation']['native_voxels_read'], int(np.prod(NATIVE)))
        self.assertNotIn('path', receipt['mask']['pre_crop_mask'])
        padded = self.mask.copy()
        padded[:, :, 5:] = 0
        expected = direct_reconstruction(self.root / 'original', padded, NATIVE)[:, :, :11]
        np.testing.assert_array_equal(read_nrrd(output / 'dataset.nrrd'), expected)
        rescaled = direct_reconstruction(self.root / 'rescaled', self.mask[:, :, :5], cropped_native)
        self.assertFalse(np.array_equal(expected, rescaled), 'Fixture must detect changed X sampling.')
        again = self.run_cli(bundle=cropped, output=output, native=cropped_native)
        self.assertEqual(prep.read_json(again / 'sample.result.json')['status'], 'validated_existing')
        receipt['mask']['crop_audit']['written_payload_verified'] = False
        prep.save_json(output / 'preparation.json', receipt)
        with self.assertRaisesRegex(ValueError, 'lacks its native-grid crop audit'):
            self.run_cli(bundle=cropped, output=output, native=cropped_native)

        # A manifest rewritten to the cropped grid loses the original lattice.
        manifest = prep.read_json(cropped / 'source_manifest.json')
        manifest.update(output_shape_tyx=[*PREVIEW[:2], 5], full_quality_output_shape_tyx=list(cropped_native))
        prep.save_json(cropped / 'source_manifest.json', manifest)
        edits = prep.read_json(cropped / 'edits.json')
        edits['source_manifest_sha256'] = prep.digest(cropped / 'source_manifest.json')
        prep.save_json(cropped / 'edits.json', edits)
        with self.assertRaisesRegex(ValueError, 'original geometry differs from its source manifest'):
            self.run_cli('--validate-only', bundle=cropped, output=self.output.with_name('rewritten'), native=cropped_native)


if __name__ == '__main__':
    unittest.main()
