"""Synthetic algorithm and real CLI integration tests; no biological validation."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage as ndi

import xta_native_filter as nf
from xta_auto import split_thin_neck, gather_region, project_region, mixed_bounds
from tests.test_native_filter import fixture, pool

ROOT = Path(__file__).resolve().parents[1]


def dumbbell(width=1):
    a = np.zeros((20, 20, 40), bool)
    keep = np.zeros_like(a)
    remove = np.zeros_like(a)
    keep[4:16,4:16,2:14] = True
    remove[4:16,4:16,26:38] = True
    a |= keep | remove
    lo = 10 - width//2
    a[lo:lo+width,lo:lo+width,14:26] = True
    # A native one-voxel appendage on the retained lobe must not be smoothed away.
    a[4,4,0:2] = True
    return a, keep, remove


class TestNativeAuto(unittest.TestCase):
    def test_sever_thin_bridge_preserve_lobes_and_native_detail(self):
        for width in (1,2,3):
            with self.subTest(width=width):
                a,k,r = dumbbell(width)
                out, report = split_thin_neck(a,k,r,max_radius=2)
                self.assertIsNotNone(out)
                self.assertTrue(np.all(out[k]))
                self.assertFalse(np.any(out[r]))
                self.assertTrue(np.all(out[4,4,0:2]))
                self.assertFalse(np.any(out & ~a))
                self.assertEqual(ndi.label(out,structure=nf.FACE)[1],1)
                self.assertEqual(report['decision'],'auto_split')
                self.assertLessEqual(report['separator_max_distance_native_voxels'],2)

    def test_broad_connection_preserved_not_coarse_clipped(self):
        a,k,r = dumbbell(7)
        out, report = split_thin_neck(a,k,r,max_radius=2)
        self.assertIsNone(out)
        self.assertEqual(report['decision'],'keep_unresolved')

    def test_broad_connection_can_use_explicit_larger_radius(self):
        a,k,r = dumbbell(5)
        self.assertIsNone(split_thin_neck(a,k,r,max_radius=2)[0])
        self.assertIsNotNone(split_thin_neck(a,k,r,max_radius=3)[0])

    def test_no_all_thin_structure_erosion(self):
        a=np.zeros((12,5,5),bool);a[:,2,2]=1
        k=np.zeros_like(a);k[:5]=a[:5]
        r=np.zeros_like(a);r[7:]=a[7:]
        self.assertIsNone(split_thin_neck(a,k,r)[0])

    def test_overlapping_votes_do_not_supply_false_markers(self):
        a,k,r=dumbbell()
        out,report=split_thin_neck(a,a,a)
        self.assertIsNone(out)
        self.assertEqual(report['reason'],'no_exclusive_keep_or_remove_evidence')

    def test_keep_evidence_guard_rejects_cut(self):
        a,k,r=dumbbell()
        # Protect a voxel in the middle of the eventual rejected basin; core is unchanged.
        k[10,10,24]=1
        out,report=split_thin_neck(a,k,r,max_keep_loss=0)
        self.assertIsNone(out)
        self.assertTrue(any(x['reason']=='keep_evidence_loss_guard' for x in report['attempts']))

    def test_padded_border_component_and_no_input_mutation(self):
        a,k,r=dumbbell();a=a[4:16,4:16,:];k=k[4:16,4:16,:];r=r[4:16,4:16,:]
        originals=[x.copy() for x in (a,k,r)]
        out,_=split_thin_neck(a,k,r)
        self.assertIsNotNone(out)
        for x,y in zip((a,k,r),originals):self.assertTrue(np.array_equal(x,y))

    def test_unknown_thick_core_not_automatically_rejected(self):
        a,k,r=dumbbell()
        a[0:4,8:13,4:9]=1
        out,report=split_thin_neck(a,k,r)
        self.assertIsNotNone(out)
        self.assertTrue(np.all(out[0:4,8:13,4:9]))

    def test_invalid_parameters(self):
        a,k,r=dumbbell()
        for kwargs in ({'max_radius':0},{'max_keep_loss':1},{'min_remove_fraction':0}):
            with self.assertRaises(ValueError):split_thin_neck(a,k,r,**kwargs)

    def test_region_gather_and_projection_match_full_support(self):
        rng=np.random.default_rng(345)
        for full,low in (((11,13,17),(4,5,6)),((7,9,11),(7,9,11))):
            support=nf.Support(full,low)
            decision=rng.random(low)<.2
            box=(slice(1,full[0]-1),slice(2,full[1]-1),slice(3,full[2]-1))
            self.assertTrue(np.array_equal(gather_region(support,decision,box),
                                           support.gather(decision,0,full[0])[box]))
            region=rng.random(tuple(s.stop-s.start for s in box))<.2
            a=np.zeros(full,bool);a[box]=region
            dest=np.zeros(low,bool);project_region(support,region,box,dest)
            self.assertTrue(np.array_equal(dest,pool(a,low)))

    def test_component_bounds_span_slab_boundaries(self):
        a,k,r=dumbbell()
        a[0,0,0]=1
        for depth in (1,3,20):
            labels=np.empty(a.shape,np.int32)
            roots,counts,flags,*_=nf.label_stream(nf.array_slabs(a,depth),labels,maximum=5000)
            mixed=counts>10
            boxes=mixed_bounds(labels,roots,mixed,depth)
            self.assertEqual(len(boxes),1)
            self.assertEqual([(s.start,s.stop) for s in boxes[0][1]],[(4,16),(4,16),(0,38)])


class TestAutoWorkflow(unittest.TestCase):
    def auto_fixture(self,d):
        a,k,r=dumbbell()
        low=(10,10,20)
        args=fixture(d,{'component':a.astype(np.uint8)},
                     {'component':pool(k,low)},low_shape=low)
        args.conflicts='auto'
        return args,a,k,r

    def test_real_cli_auto_output_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            args,a,k,r=self.auto_fixture(d)
            command=[sys.executable,str(ROOT/'tools'/'xta_native_filter.py'),'--auto']
            for attr in ('edits','low_manifest','full_manifest','output','scratch','block_mib'):
                command += ['--'+attr.replace('_','-'),str(getattr(args,attr))]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=nf.load_json(Path(args.output)/'report.json')
            out=nf.Nrrd(Path(args.output)/'volume.nrrd').array()!=0
            self.assertEqual(report['status'],'completed_with_heuristics')
            self.assertEqual(report['auto_resolved_components'],1)
            self.assertEqual(report['unresolved_components'],0)
            self.assertFalse(np.any(out & ~a))
            self.assertTrue(np.all(out[k]))
            self.assertFalse(np.any(out[r]))
            preview=nf.Nrrd(Path(args.output)/report['layers'][0]['auto_removed_preview_file']).array()
            self.assertEqual(preview.shape,(10,10,20))
            self.assertIn('auto_split',(Path(args.output)/'components.tsv').read_text())

    def test_roi_cap_keeps_and_reports_without_stopping(self):
        with tempfile.TemporaryDirectory() as d:
            args,a,k,r=self.auto_fixture(d);args.auto_max_roi_voxels=10
            with redirect_stdout(io.StringIO()):self.assertEqual(nf.run(args),0)
            report=nf.load_json(Path(args.output)/'report.json')
            self.assertEqual(report['status'],'completed_with_unresolved_components')
            self.assertEqual(report['layers'][0]['auto_components'][0]['reason'],'roi_voxel_limit')
            self.assertTrue(np.array_equal(nf.Nrrd(Path(args.output)/'volume.nrrd').array(),a))

    def test_broad_conflict_unattended_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            a,k,r=dumbbell(7);low=(10,10,20)
            args=fixture(d,{'component':a.astype(np.uint8)},{'component':pool(k,low)},low_shape=low)
            args.conflicts='auto'
            with redirect_stdout(io.StringIO()):self.assertEqual(nf.run(args),0)
            report=nf.load_json(Path(args.output)/'report.json')
            self.assertEqual(report['status'],'completed_with_unresolved_components')
            self.assertEqual(report['auto_removed_voxels'],0)

    def test_auto_disabled_legacy_conflict_stops(self):
        with tempfile.TemporaryDirectory() as d:
            args,a,k,r=self.auto_fixture(d);args.conflicts='error'
            with redirect_stdout(io.StringIO()):self.assertEqual(nf.run(args),2)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())

    def test_auto_same_answer_across_slab_depths(self):
        outputs=[]
        for depth in (1,7,20):
            with tempfile.TemporaryDirectory() as d:
                args,a,k,r=self.auto_fixture(d)
                original=nf.label_stream
                def relabel(slabs,labels,**kwargs):
                    blocks=list(slabs)
                    all_data=np.concatenate([b.copy() for _,b in blocks])
                    return original(nf.array_slabs(all_data,depth),labels,**kwargs)
                with patch.object(nf,'label_stream',side_effect=relabel),redirect_stdout(io.StringIO()):
                    self.assertEqual(nf.run(args),0)
                outputs.append(nf.Nrrd(Path(args.output)/'volume.nrrd').array())
        for out in outputs[1:]:self.assertTrue(np.array_equal(out,outputs[0]))

    def test_invalid_auto_parameters_no_output_created(self):
        with tempfile.TemporaryDirectory() as d:
            args,*_=self.auto_fixture(d);args.auto_max_keep_loss=float('nan')
            with self.assertRaises(ValueError):nf.run(args)
            self.assertFalse(Path(args.output).exists())

    def test_no_skimage_dependency_for_legacy_mode(self):
        with tempfile.TemporaryDirectory() as d:
            args,*_=self.auto_fixture(d);args.conflicts='keep'
            with patch.dict(sys.modules,{'skimage.segmentation':None}),redirect_stdout(io.StringIO()):
                self.assertEqual(nf.run(args),0)

    def test_fragmented_union_largest_cleanup_and_optional_count(self):
        # Different native components can occupy the same connected coarse preview.
        # The native largest-object policy should still remove small fragments.
        for keep_objects,expected_count in ((1,1),(2,2)):
            with tempfile.TemporaryDirectory() as d:
                a=np.zeros((8,8,8),np.uint8)
                a[0:3,0:3,0:3]=1
                a[3:6,3:6,3:6]=1  # corner-only touch: separate under 6-connectivity
                low=pool(a,(2,2,2))
                args=fixture(d,{'component':a},{'component':low},low_shape=(2,2,2))
                args.conflicts='auto';args.keep_objects=keep_objects
                with redirect_stdout(io.StringIO()):self.assertEqual(nf.run(args),0)
                out=nf.Nrrd(Path(args.output)/'volume.nrrd').array()
                self.assertEqual(ndi.label(out,structure=nf.FACE)[1],expected_count)
                report=nf.load_json(Path(args.output)/'report.json')
                self.assertEqual(report['union_cleanup']['kept_components'],expected_count)


if __name__=='__main__':unittest.main(verbosity=2)
