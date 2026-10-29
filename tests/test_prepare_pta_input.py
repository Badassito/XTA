"""Preparation publishes verified pairs and retains failed native evidence."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import prepare_pta_input as prep


FAKE_NATIVE = '''from pathlib import Path
import gzip, json, sys
if '--version' in sys.argv:
    print('0.3.1'); raise SystemExit
def option(flag):return sys.argv[sys.argv.index(flag)+1]
output=Path(option('--output'));output.mkdir()
scratch=Path(option('--scratch'));(scratch/'evidence.txt').write_text('retained scratch evidence')
quality={'arguments':sys.argv[1:],'status':'complete','unresolved_components':0,'auto_resolved_components':0}
if Path(__file__).with_suffix('.unresolved').exists():
    quality.update(status='completed_with_unresolved_components',unresolved_components=2)
if Path(__file__).with_suffix('.needs_review').exists():
    quality.update(status='needs_review')
(output/'report.json').write_text(json.dumps(quality))
if Path(__file__).with_suffix('.fail').exists():
    print('explicit native failure diagnostic',flush=True);raise SystemExit(7)
stem=Path(option('--edits')).name.removesuffix('_Filtered')
header=b'NRRD0005\\ntype: unsigned char\\ndimension: 3\\nspace: left-posterior-superior\\nsizes: 2 2 2\\nspace directions: (1,0,0) (0,1,0) (0,0,1)\\nencoding: gzip\\nspace origin: (0,0,0)\\n\\n'
(output/(stem+'.nrrd')).write_bytes(header+gzip.compress(bytes([1]*8),mtime=0))
'''


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.stem='sample'
        self.bundle=self.root/'training'/'v1.1'/'sample_Filtered'
        self.bundle.mkdir(parents=True)
        self.output=self.bundle.with_name('sample_Input')
        self.video=self.root/'upload'/'sample.mkv'
        self.video.parent.mkdir();self.video.write_bytes(b'original intensity bytes')
        manifest={'quality':'low_quality','output_shape_tyx':[2,2,2],
                  'full_quality_output_shape_tyx':[2,2,2],
                  'layers':[{'filename':'global.nrrd','segment_name':'sample_Global',
                             'suffix':'Global_final_output','recomposition_op':'select'}]}
        self.low=self.bundle/'source_manifest.json';prep.save_json(self.low,manifest)
        npz=self.bundle/'global.npz'
        np.savez_compressed(npz,mask=np.ones((2,2,2),np.uint8),
                            extent_xyz=np.array([0,1,0,1,0,1]),ijk_to_ras=prep.REFERENCE_RAS)
        prep.save_json(self.bundle/'edits.json',{'format':'xta-slicer-edits-v1',
            'source_manifest_sha256':prep.digest(self.low),'preview_shape_tyx':[2,2,2],
            'full_quality_shape_tyx':[2,2,2],'segments':[{'name':'sample_Global','file':npz.name,
            'source_filename':'global.nrrd','recomposition_op':'select','sha256':prep.digest(npz),
            'foreground_voxels':8}]})
        self.full=self.root/'full.json';prep.save_json(self.full,{'output_shape_tyx':[2,2,2]})
        self.native=self.root/'native.py';self.native.write_text(FAKE_NATIVE,encoding='utf-8')
        self.union_helper=self.root/'xta_native_union.py';self.union_helper.write_text('# fixture helper\n')
        for name in ('xta_native_sparse.py','xta_native_io.py','xta_auto.py'):
            (self.root/name).write_text('# fixture native algorithm dependency\n')
        self.union=self.root/'union.json';prep.save_json(self.union,{'format':'fixture union receipt'})
        self.pairs=self.root/'pairs.json';prep.save_json(self.pairs,{'pairs':[
            {'low_manifest':str(self.low),'low_manifest_sha256':prep.digest(self.low),
             'full_manifest':str(self.full),'full_manifest_sha256':prep.digest(self.full)}]})
        self.counter=0

    def run_cli(self,*extra,native=False,union=False):
        self.counter+=1
        log=self.root/f'log-{self.counter}'
        args=[self.stem,'--training-root',str(self.root/'training'),'--generation','v1.1',
              '--source-video',str(self.video),'--intensity-mode','copy','--log-dir',str(log),
              '--scratch-root',str(self.root/'scratch')]
        if native:
            args+=['--mode','native','--native-filter',str(self.native),
                   '--native-low-template',str(self.low),'--native-full-template',str(self.full)]
        if union:args+=['--native-union-receipt',str(self.union),'--native-union-source-pairs',str(self.pairs)]
        with patch.object(prep,'probe_source',return_value={'shape_tyx':[2,2,2],'frame_count_method':'synthetic'}):
            prep.main(args+list(extra))
        return log

    def test_copy_publication_reuse_and_corruption_refusal(self):
        first=self.run_cli()
        intensity=self.output/'sample.mkv'
        self.assertFalse(intensity.is_symlink())
        self.assertEqual(intensity.read_bytes(),self.video.read_bytes())
        receipt=prep.read_json(self.output/'preparation.json')
        self.assertEqual(receipt['intensity_publication']['method'],'copy')
        self.assertEqual(receipt['source_intensity_hash'],prep.digest(self.video))
        self.assertFalse((first/'smoke-command.json').exists())
        second=self.run_cli()
        self.assertEqual(prep.read_json(second/'sample.result.json')['status'],'validated_existing')
        intensity.write_bytes(b'corrupt copied intensity')
        with self.assertRaisesRegex(ValueError,'copied intensity identity'):
            self.run_cli()
        self.assertEqual(self.video.read_bytes(),b'original intensity bytes')
        self.assertEqual(intensity.read_bytes(),b'corrupt copied intensity')

    def test_native_union_forwarding_and_reuse_identity(self):
        log=self.run_cli(native=True,union=True)
        command=prep.read_json(log/'sample.native-command.json')['command']
        self.assertEqual(command[command.index('--union-receipt')+1],str(self.union))
        self.assertEqual(command[command.index('--union-source-pairs')+1],str(self.pairs))
        contract=prep.read_json(self.output/'preparation.json')['identity']['native']
        self.assertEqual(contract['union_receipt_sha256'],prep.digest(self.union))
        self.assertEqual(contract['helper_files']['xta_native_union.py']['sha256'],prep.digest(self.union_helper))
        self.assertEqual(set(contract['helper_files']),{'xta_native_union.py','xta_native_sparse.py','xta_native_io.py','xta_auto.py'})
        self.run_cli(native=True,union=True)
        self.union.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'provenance differs'):
            self.run_cli(native=True,union=True)

    def test_native_without_union_keeps_original_contract(self):
        log=self.run_cli(native=True)
        command=prep.read_json(log/'sample.native-command.json')['command']
        self.assertNotIn('--union-receipt',command)
        contract=prep.read_json(self.output/'preparation.json')['identity']['native']
        self.assertNotIn('union_receipt',contract)
        quality=prep.read_json(self.output/'preparation.json')['native_quality']
        self.assertEqual(quality['status'],'complete')
        self.assertEqual(quality['unresolved_components'],0)
        self.run_cli(native=True)

    def test_native_zero_exit_with_unresolved_or_non_success_refuses_publication(self):
        for suffix in ('.unresolved','.needs_review'):
            marker=self.native.with_suffix(suffix);marker.touch()
            with self.subTest(suffix=suffix),self.assertRaisesRegex(ValueError,'resolved success'):
                self.run_cli(native=True)
            self.assertFalse(self.output.exists())
            log=self.root/f'log-{self.counter}'
            self.assertTrue((log/'sample.native-failed-diagnostics'/'report.json').is_file())
            marker.unlink()

    def test_native_old_receipt_without_quality_refuses_reuse(self):
        self.run_cli(native=True)
        receipt=prep.read_json(self.output/'preparation.json')
        receipt.pop('native_quality')
        prep.save_json(self.output/'preparation.json',receipt)
        with self.assertRaisesRegex(ValueError,'resolved success'):
            self.run_cli(native=True)

    def test_edited_shape_authority_copy_reuse_and_distinct_output(self):
        # A deliberately incompatible legacy Input must remain untouched.
        self.output.mkdir();(self.output/'old.txt').write_text('old labels')
        target=self.output.with_name('sample_Input_edited_shape')
        log=self.run_cli('--mode','edited_shape','--output-dir',str(target))
        self.assertEqual((self.output/'old.txt').read_text(),'old labels')
        receipt=prep.read_json(target/'preparation.json')
        self.assertFalse(receipt['native_component_recovery'])
        self.assertFalse(receipt['original_source_nrrds_required'])
        self.assertTrue(receipt['edited_shape_reconstruction'])
        self.assertFalse(receipt['restored_native_detail'])
        self.assertEqual(receipt['label_authority'],'edited_preview_global')
        self.assertIn('signed-distance',receipt['algorithm'])
        self.assertIn('helper_sha256',receipt['identity']['edited_shape'])
        self.assertEqual(receipt['mask']['reconstruction']['center_roundtrip']['false_positive_voxels'],0)
        self.assertEqual((target/'sample.mkv').read_bytes(),self.video.read_bytes())
        self.assertFalse((log/'sample.native-command.json').exists())
        again=self.run_cli('--mode','edited_shape','--output-dir',str(target))
        self.assertEqual(prep.read_json(again/'sample.result.json')['status'],'validated_existing')

    def test_edited_shape_helper_change_refuses_reuse(self):
        self.run_cli('--mode','edited_shape')
        module,contract=prep.edited_shape_contract()
        original=prep.edited_shape_contract
        def changed():
            current=dict(contract);current['helper_sha256']='changed'
            return module,current
        with patch.object(prep,'edited_shape_contract',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'provenance differs'):
                self.run_cli('--mode','edited_shape')

    def test_edited_shape_helper_changed_before_publication_refuses_output(self):
        original=prep.publish_intensity
        module,contract=prep.edited_shape_contract()
        state={'changed':False}
        def change_helper(*args):
            result=original(*args);state['changed']=True
            return result
        def get_contract():
            current=dict(contract)
            if state['changed']:current['helper_sha256']='changed'
            return module,current
        with patch.object(prep,'publish_intensity',side_effect=change_helper),patch.object(prep,'edited_shape_contract',side_effect=get_contract):
            with self.assertRaisesRegex(ValueError,'Inputs changed'):
                self.run_cli('--mode','edited_shape')
        self.assertFalse(self.output.exists())

    def test_new_edited_shape_uses_writer_audit_without_second_payload_decode(self):
        module,contract=prep.edited_shape_contract()
        with patch.object(prep,'edited_shape_contract',return_value=(module,contract)), \
                patch.object(module,'_validate_written',wraps=module._validate_written) as audit, \
                patch.object(prep,'validate_nrrd',side_effect=AssertionError('redundant decode')), \
                patch.object(prep,'verify_compressed_identity',wraps=prep.verify_compressed_identity) as identity:
            self.run_cli('--mode','edited_shape','--workers','128')
        self.assertEqual(audit.call_count,1)
        self.assertEqual(identity.call_count,1)
        receipt=prep.read_json(self.output/'preparation.json')
        self.assertEqual(receipt['mask']['reconstruction']['center_roundtrip']['sampled_voxels'],8)

    def test_edited_shape_reuse_still_validates_payload(self):
        self.run_cli('--mode','edited_shape')
        with patch.object(prep,'validate_nrrd',wraps=prep.validate_nrrd) as validate:
            self.run_cli('--mode','edited_shape')
        self.assertEqual(validate.call_count,1)
        self.assertEqual(validate.call_args.args[0],self.output/'sample.nrrd')
        nrrd=self.output/'sample.nrrd'
        content=nrrd.read_bytes();header,payload=content.split(b'\n\n',1)
        values=bytearray(gzip.decompress(payload));values[0]=0
        nrrd.write_bytes(header+b'\n\n'+gzip.compress(values,mtime=0))
        with self.assertRaisesRegex(ValueError,'validation failed'):
            self.run_cli('--mode','edited_shape')

    def test_edited_shape_changed_after_writer_audit_refuses_publication(self):
        original=prep.publish_intensity
        def change_audited_mask(source,destination,*args):
            result=original(source,destination,*args)
            nrrd=destination.with_suffix('.nrrd')
            with nrrd.open('ab') as stream:stream.write(b'changed after writer audit')
            return result
        with patch.object(prep,'publish_intensity',side_effect=change_audited_mask):
            with self.assertRaisesRegex(ValueError,'changed before publication'):
                self.run_cli('--mode','edited_shape')
        self.assertFalse(self.output.exists())

    def test_edited_shape_missing_actual_payload_audit_refuses_write_and_reuse(self):
        module,contract=prep.edited_shape_contract()
        original=module.write_edited_shape
        def missing_audit(*args,**kwargs):
            result=original(*args,**kwargs)
            result['reconstruction'].pop('separation_audit')
            return result
        with patch.object(prep,'edited_shape_contract',return_value=(module,contract)), \
                patch.object(module,'write_edited_shape',side_effect=missing_audit):
            with self.assertRaisesRegex(ValueError,'quality audit'):
                self.run_cli('--mode','edited_shape')
        self.assertFalse(self.output.exists())
        self.run_cli('--mode','edited_shape')
        receipt=prep.read_json(self.output/'preparation.json')
        receipt['mask']['reconstruction'].pop('separation_audit')
        prep.save_json(self.output/'preparation.json',receipt)
        with self.assertRaisesRegex(ValueError,'quality audit'):
            self.run_cli('--mode','edited_shape')

    def test_edited_shape_requires_global_and_explicit_target_is_disjoint(self):
        invalid=(('--mode','edited_shape','--mask-source','components'),
                 ('--output-dir',str(self.bundle)),
                 ('--output-dir',str(self.bundle/'inside')),
                 ('--output-dir',str(self.bundle.parent)),
                 ('--output-dir',str(self.video.parent)))
        for args in invalid:
            with self.subTest(args=args),self.assertRaises(SystemExit):
                self.run_cli(*args)
        with self.assertRaises(SystemExit):self.run_cli('other','--output-dir',str(self.output))

    def test_union_helper_change_rejects_reuse(self):
        self.run_cli(native=True,union=True)
        self.union_helper.write_text('# modified helper\n')
        with self.assertRaisesRegex(ValueError,'provenance differs'):
            self.run_cli(native=True,union=True)

    def test_source_pair_manifest_change_rejects_reuse(self):
        self.run_cli(native=True,union=True)
        self.full.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            self.run_cli(native=True,union=True)

    def test_native_reader_change_rejects_reuse(self):
        self.run_cli(native=True,union=True)
        (self.root/'xta_native_io.py').write_text('# modified reader\n')
        with self.assertRaisesRegex(ValueError,'provenance differs'):
            self.run_cli(native=True,union=True)

    def test_native_failure_preserves_diagnostics_and_scratch(self):
        self.native.with_suffix('.fail').touch()
        with self.assertRaises(Exception):self.run_cli(native=True,union=True)
        log=self.root/'log-1'
        self.assertFalse(self.output.exists())
        self.assertTrue((log/'sample.native-failed-diagnostics'/'report.json').is_file())
        self.assertIn('explicit native failure diagnostic',(log/'sample.native.log').read_text())
        retained=prep.read_json(log/'sample.native-failed-scratch.json')['retained_scratch']
        self.assertTrue((Path(retained)/'evidence.txt').is_file())
        self.assertFalse(list(self.bundle.parent.glob('.sample_Input.prep-*')))
        self.assertFalse((self.bundle.parent/'.sample_Input.prep.lock').exists())

    def test_inputs_changed_during_preparation_refuses_publication(self):
        original=prep.publish_intensity
        def change_source(*args):
            result=original(*args)
            before=self.video.stat()
            self.video.write_bytes(b'x'*before.st_size)
            os.utime(self.video,ns=(before.st_atime_ns,before.st_mtime_ns))
            return result
        with patch.object(prep,'publish_intensity',side_effect=change_source):
            with self.assertRaisesRegex(ValueError,'Inputs changed'):
                self.run_cli()
        self.assertFalse(self.output.exists())

    def test_default_original_path_and_symlink_mode_remain_defaults(self):
        with patch.object(prep,'prepare',return_value=self.output) as prepare:
            prep.main([self.stem,'--root',str(self.video.parent),
                       '--training-root',str(self.root/'training'),'--generation','v1.1',
                       '--log-dir',str(self.root/'defaults-log')])
        options=prepare.call_args.args[0]
        self.assertIsNone(options.source_video)
        self.assertEqual(options.intensity_mode,'symlink')
        self.assertEqual(options.mode,'upscale')

    def test_union_options_are_paired_and_native_only(self):
        for args in (['--native-union-receipt',str(self.union)],
                     ['--native-union-source-pairs',str(self.pairs)]):
            with self.subTest(args=args),self.assertRaises(SystemExit):
                self.run_cli(*args,native=True)
        with self.assertRaises(SystemExit):self.run_cli(union=True)
        with self.assertRaises(SystemExit):self.run_cli('other')


if __name__=='__main__':unittest.main()
