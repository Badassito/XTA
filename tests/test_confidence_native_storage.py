"""Versioned block storage and deferred collection avoid implicit dense views."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import geometry
from XTA.confidence_evidence import (ConfidenceEvidenceRef,ConfidenceEvidenceReader,
    configure_confidence_evidence,capture_prediction_confidence,publish_confidence_shards,
    publish_native_confidence_pieces,write_confidence_evidence,write_block_confidence_evidence)
from XTA.confidence_projection import score_projection_reader
from XTA.config import TiltedViewGroup


class ConfidenceNativeStorageTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.addCleanup(configure_confidence_evidence,None,enabled=False)
        quiet=redirect_stdout(io.StringIO());quiet.__enter__()
        self.addCleanup(quiet.__exit__,None,None,None)

    def configure(self,name='output'):
        configure_confidence_evidence(self.root/name,enabled=True,defer_projection=True)

    def test_source_schema1_and_schema2_are_exact_and_stream_small_crops(self):
        values=np.zeros((2,600,610),np.uint8)
        values[:,300:350,300:350]=137
        values[:,0,0]=241;values[:,-1,-1]=83
        old=write_confidence_evidence(self.root/'old',values.shape,lambda z:values[z],layer_key='k',model_name='m')
        new=write_block_confidence_evidence(self.root/'new',values.shape,lambda z:values[z],layer_key='k',model_name='m')
        self.assertEqual(old.metadata['schema'],'xta.confidence_evidence/1')
        self.assertEqual(new.metadata['schema'],'xta.confidence_evidence/2')
        self.assertLess((new.path/'scores.u8.zlib').stat().st_size,(old.path/'scores.u8.zlib').stat().st_size)
        for ref in (old,new):
            with ref.reader() as reader:
                self.assertIsInstance(reader,ConfidenceEvidenceReader)
                actual,known=reader(0,2)
                np.testing.assert_array_equal(actual,values)
                np.testing.assert_array_equal(known,values>0)
        with new.reader() as reader:
            crops=list(reader.iter_crops(0))
            self.assertEqual(len(crops),3)
            self.assertEqual(sum(crop.nbytes for *_,crop in crops),2502)
        with self.assertRaisesRegex(RuntimeError,'closed'):
            reader(0,1)

    def test_generic_deferred_capture_never_projects_or_allocates_a_volume(self):
        self.configure()
        view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
        scores=np.zeros((5,7,9),np.uint8);scores[:,2:5,3:7]=173
        mask=(scores>0).astype(np.uint8)
        with mock.patch('XTA.confidence_projection.score_projection_reader',side_effect=AssertionError('projected')):
            with mock.patch('numpy.memmap',side_effect=AssertionError('dense staging')):
                ref=capture_prediction_confidence(mask,scores,view=view,model_name='m',temp_dir=self.root,
                                                 output_shape=(4,6,8))
        self.assertEqual(ref.coordinate_space,'native_view_processing')
        self.assertEqual(ref.storage_shape,(5,7,9))
        self.assertEqual(ref.shape,(4,6,8))
        with self.assertRaisesRegex(ValueError,'explicit source_reader'):
            ref.reader()
        with score_projection_reader(scores,view,(4,6,8),self.root/'oracle') as read:
            expected=np.stack([read(z) for z in range(4)])
        workspace=self.root/'conversion'
        with ref.source_reader(workspace,memory_mib=4,max_staging_mib=4) as reader:
            actual,known=reader(0,4)
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(known,expected>0)
        self.assertEqual(list(workspace.iterdir()),[])
        with self.assertRaisesRegex(RuntimeError,'closed'):
            reader(0,1)

    def test_d1_collection_copies_disjoint_shards_without_dense_reassembly(self):
        self.configure()
        view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
        expected=np.zeros((5,7,9),np.uint8);expected[:2,1:4,2:5]=127;expected[2:,2:5,3:6]=211
        shards=[]
        for start,stop in ((2,5),(0,2)):
            data=expected[start:stop]
            ref=write_block_confidence_evidence(self.root/'shards'/str(start),data.shape,lambda z:data[z],
                layer_key='key',model_name='m',coordinate_space='native_view_processing',source_shape_tyx=expected.shape)
            shards.append(dict(path=str(ref.path),shape_tyx=list(data.shape),slice_start=start,slice_count=stop-start,
                view_shape_tyx=list(expected.shape),view_name=view.name,model_name='m',layer_key='key'))
        real_memmap=np.memmap
        def only_index_maps(*args,**kwargs):
            if len(kwargs.get('shape',()))==3:
                raise AssertionError('collection staged a dense view')
            return real_memmap(*args,**kwargs)
        with mock.patch('numpy.memmap',side_effect=only_index_maps),mock.patch(
                'XTA.confidence_projection.score_projection_reader',side_effect=AssertionError('projected')):
            ref=publish_confidence_shards(shards,view=view,model_name='m',temp_dir=self.root)
        self.assertEqual(ref.metadata['layout'],'uint8_zlib_blocks')
        self.assertEqual({p.name for p in ref.path.iterdir()}, {'metadata.json','index.bin','scores.u8.zlib'})
        self.assertEqual(ref.metadata['known_voxels'],int(np.count_nonzero(expected)))
        shutil.rmtree(self.root/'shards')
        with ref.native_reader() as reader:
            actual,_=reader(0,5)
            np.testing.assert_array_equal(actual,expected)
        with ref.source_reader(self.root/'convert',memory_mib=4,max_staging_mib=4) as reader:
            actual,_=reader(0,5)
            np.testing.assert_array_equal(actual,expected)
        metadata=json.loads((ref.path/'metadata.json').read_text())
        metadata['block_count'] += 1
        (ref.path/'metadata.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'index size'):
            ConfidenceEvidenceRef.open(ref.path).native_reader()

    def test_overlapping_tile_pieces_use_numeric_max_and_do_not_claim_unique_count(self):
        self.configure()
        view=geometry.get_view_infos(3,6,6,cartesian_views=('transverse',))[0]
        pieces=[]
        expected=np.zeros((3,6,6),np.uint8)
        for number,(offset,score) in enumerate(((0,128),(1,127))):
            values=np.full((3,4,4),score,np.uint8)
            ref=write_block_confidence_evidence(self.root/f'tile{number}',values.shape,lambda z:values[z],
                layer_key=f'tile{number}',model_name='m',coordinate_space='native_view_processing')
            pieces.append(dict(reference=ref,offset_tyx=(0,offset,offset)))
            target=expected[:,offset:offset+4,offset:offset+4]
            np.maximum(target,values,out=target)
        ref=publish_native_confidence_pieces(pieces,native_shape=expected.shape,view=view,model_name='m',
            source='tile',tile_config_id='s4',tile_acceptance='parent_mask',stage='pre_tile',disjoint=False)
        self.assertIsNone(ref.metadata['known_voxels'])
        self.assertEqual(ref.metadata['known_contributions'],96)
        with ref.native_reader() as reader:
            actual,_=reader(0,3)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(int(actual[0,2,2]),128)

    def test_native_conversion_budget_and_failure_cleanup_are_explicit(self):
        self.configure()
        view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
        values=np.ones((5,7,9),np.uint8)*173
        ref=capture_prediction_confidence(values>0,values,view=view,model_name='m',temp_dir=self.root)
        work=self.root/'convert'
        with self.assertRaisesRegex(MemoryError,'staging'):
            with ref.source_reader(work,memory_mib=4,max_staging_mib=.00001):
                pass
        self.assertFalse(work.exists())
        with mock.patch('XTA.confidence_projection.score_projection_reader',side_effect=OSError('injected projection failure')):
            with self.assertRaisesRegex(OSError,'injected projection'):
                with ref.source_reader(work,memory_mib=4,max_staging_mib=4):
                    pass
        self.assertEqual(list(work.iterdir()),[])

    def test_empty_native_view_needs_no_staging_but_still_validates_payload(self):
        self.configure()
        view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
        values=np.zeros((5,7,9),np.uint8)
        ref=capture_prediction_confidence(values,values,view=view,model_name='m',temp_dir=self.root)
        work=self.root/'empty_convert'
        with ref.source_reader(work,memory_mib=1,max_staging_mib=.000001) as reader:
            self.assertEqual(list(reader.iter_crops(0)),[])
        self.assertFalse(work.exists())
        (ref.path/'scores.u8.zlib').unlink()
        with self.assertRaises(FileNotFoundError):
            with ref.source_reader(work,memory_mib=1,max_staging_mib=1):
                pass

    def test_explicit_native_conversion_matches_eager_source_scores_across_families(self):
        views=geometry.get_view_infos(5,7,9,cartesian_views=('transverse','sagittal','coronal'),
            tilt_groups=(TiltedViewGroup(('transverse',),(23.,),('vertical',)),),
            azimuthal_views=('transverse','tilted_transverse'),azimuthal_azimuth_angles=(45.,45.),
            azimuthal_native_raster=0,radial_views=('transverse',),radial_min_radius=.7,radial_patch_size=5,
            spherical_views=('transverse',),spherical_min_radius=.7,spherical_patch_size=5)
        selected=[];seen=set()
        for view in views:
            key=view.name if view.family not in ('radial','spherical') else view.family
            if key not in seen:
                selected.append(view);seen.add(key)
        for index,view in enumerate(selected):
            with self.subTest(view=view.name):
                rng=np.random.default_rng(310+index)
                values=rng.choice(np.array([0,0,137,229],np.uint8),size=(view.num_slices,3,3))
                configure_confidence_evidence(self.root/f'eager{index}',enabled=True)
                eager=capture_prediction_confidence(values>0,values.copy(),view=view,model_name='m',
                    temp_dir=self.root/f'tmp{index}',output_shape=(4,6,8))
                expected,_=eager.read(0,4)
                configure_confidence_evidence(self.root/f'native{index}',enabled=True,defer_projection=True)
                native=capture_prediction_confidence(values>0,values.copy(),view=view,model_name='m',
                    temp_dir=self.root/f'tmp{index}',output_shape=(4,6,8))
                with native.source_reader(self.root/f'convert{index}',memory_mib=4,max_staging_mib=4) as reader:
                    actual,_=reader(0,4)
                    np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':
    unittest.main()
