import argparse
from contextlib import redirect_stdout, redirect_stderr
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_filter as nf


def pool(a, shape):
    out = np.zeros(shape, np.uint8)
    for index in np.ndindex(shape):
        slices = tuple(slice(i*n//m, ((i+1)*n+m-1)//m) for i,n,m in zip(index,a.shape,shape))
        out[index] = np.any(a[slices])
    return out


def write(path, a, name):
    nf.write_nrrd(path, a.shape, nf.array_slabs(a, 2), name)


def snapshot(path, a, start=(0,0,0), translation=(0,0,0)):
    z,y,x = a.shape
    sx,sy,sz = start[::-1]
    matrix = np.diag([-1., -1., 1., 1.])
    matrix[:3,3] = matrix[:3,:3] @ translation
    np.savez_compressed(path, mask=a, extent_xyz=[sx,sx+x-1,sy,sy+y-1,sz,sz+z-1], ijk_to_ras=matrix)
    return {'name': 'test', 'file': path.name, 'sha256': nf.digest(path),
            'foreground_voxels': int(np.count_nonzero(a))}


def fixture(root, full_arrays, edited_arrays, global_mask=None, *, low_shape=None, roles=None):
    root = Path(root)
    full_dir, low_dir, bundle_dir = [root/n for n in ('full','low','bundle')]
    for p in (full_dir,low_dir,bundle_dir):
        p.mkdir(parents=True)
    full_shape = next(iter(full_arrays.values())).shape
    low_shape = low_shape or tuple(n//2 for n in full_shape)
    if global_mask is None:
        global_mask = np.bitwise_or.reduce(list(full_arrays.values()))
    originals = dict(full_arrays)
    originals['Global_final_output'] = global_mask
    edits = dict(edited_arrays)
    edits['Global_final_output'] = np.bitwise_or.reduce(list(edited_arrays.values()))
    entries = []
    segments = []
    for i,(suffix,a) in enumerate(originals.items()):
        name = 'volume_' + suffix
        filename = name + '.seg.nrrd'
        write(full_dir/filename, a, name)
        low_a = pool(a,low_shape)
        write(low_dir/filename, low_a, name)
        role = (roles or {}).get(suffix, 'select' if suffix=='Global_final_output' else 'union')
        entry = {'filename':filename,'segment_name':name,'suffix':suffix,'layer_key':suffix,
                 'model_name':'test','recomposition_op':role}
        entries.append(entry)
        se = snapshot(bundle_dir/f'{i:05d}.npz', edits[suffix])
        se.update(name=name, source_filename=filename, source_sha256=nf.digest(low_dir/filename))
        segments.append(se)
    full = {'layout':'one_single_layer_nrrd_per_component', 'output_shape_tyx':list(full_shape),'layers':entries}
    low = dict(full, quality='low_quality', output_shape_tyx=list(low_shape), full_quality_output_shape_tyx=list(full_shape))
    full_path=full_dir/'volume_nrrd_manifest.json'
    low_path=low_dir/'volume_nrrd_manifest.json'
    nf.save_json(full_path,full)
    nf.save_json(low_path,low)
    nf.save_json(bundle_dir/'edits.json',{'format':nf.FORMAT,'source_manifest_sha256':nf.digest(low_path),'segments':segments})
    args=nf.parser().parse_args(['--edits',str(bundle_dir),'--low-manifest',str(low_path),
                                '--full-manifest',str(full_path),'--output',str(root/'output'),
                                '--scratch',str(root/'scratch'),'--block-mib','1'])
    return args


def complex_fixture(root):
    shape=(12,12,12)
    good=np.zeros(shape,np.uint8)
    good[1:11,2:4,2:4]=1
    good[6,3,4:7]=1  # one-voxel detail that must not be block-upsampled or smoothed
    bad=np.zeros(shape,np.uint8)
    bad[3:9,8:10,8:10]=1
    bridge=np.zeros(shape,np.uint8)
    bridge[5,3:9,3]=1
    bridge[5,8,3:9]=1
    args=fixture(root,{'component':good|bad,'bridge':bridge},
                 {'component':pool(good,(6,6,6)), 'bridge':np.zeros((6,6,6),np.uint8)})
    return args,good,bad


class TestNativeIO(unittest.TestCase):
    def test_strict_external_shape(self):
        self.assertEqual(nf.shape3(['3', '4', '5']), (3, 4, 5))
        for value in ([True, 4, 5], [3.0, 4, 5], [3.5, 4, 5], [3, 4], '345', [0, 4, 5]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                nf.shape3(value)

    def test_nrrd_roundtrip_and_concatenated_gzip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.nrrd'
            a=(np.arange(7*8*9).reshape(7,8,9)%11==0).astype(np.uint8)
            write(p,a,'unicode α label')
            self.assertTrue(np.array_equal(nf.Nrrd(p).array(),a))
            h,offset,shape=nf.read_header(p)
            prefix=p.read_bytes()[:offset]
            payload=a.tobytes()
            p.write_bytes(prefix+gzip.compress(payload[:101])+gzip.compress(payload[101:]))
            self.assertTrue(np.array_equal(nf.Nrrd(p).array(),a))
            p.write_bytes(prefix.replace(b'encoding: gzip',b'encoding: raw')+payload)
            self.assertTrue(np.array_equal(nf.Nrrd(p).array(),a))

    def test_payload_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.nrrd'
            a=np.zeros((3,4,5),np.uint8);write(p,a,'a')
            _,off,_=nf.read_header(p);prefix=p.read_bytes()[:off]
            for payload in (b'\0'*59,b'\0'*61,b'\2'+b'\0'*59):
                with self.subTest(n=len(payload),v=payload[0]):
                    p.write_bytes(prefix+gzip.compress(payload))
                    with self.assertRaises(ValueError):nf.Nrrd(p).array()

    def test_whitespace_duplicate_header_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.nrrd';write(p,np.zeros((3,4,5),np.uint8),'a')
            p.write_bytes(p.read_bytes().replace(b'type: unsigned char',
                                                b'type: unsigned char\n type : unsigned char'))
            with self.assertRaisesRegex(ValueError,'Duplicate NRRD header field'):
                nf.Nrrd(p)

    def test_reject_wrong_type_and_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.nrrd';write(p,np.zeros((3,4,5),np.uint8),'a')
            original=p.read_bytes()
            for old,new in ((b'unsigned char',b'unsigned short'),(b'(1,0,0)',b'(2,0,0)'),
                            (b'dimension: 3',b'dimension: 4')):
                with self.subTest(old=old):
                    p.write_bytes(original.replace(old,new))
                    with self.assertRaises(ValueError):nf.Nrrd(p)

    def test_cropped_and_shifted_slicer_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); p=d/'a.nrrd';write(p,np.zeros((8,9,10),np.uint8),'a')
            a=np.ones((2,3,4),np.uint8)
            e=snapshot(d/'a.npz',a,start=(1,2,3),translation=(1,0,2))
            out=nf.load_edit(d,e,nf.Nrrd(p))
            expected=np.zeros((8,9,10),np.uint8);expected[3:5,2:5,4:8]=1
            self.assertTrue(np.array_equal(out,expected))

    def test_reject_resampled_or_outside_edit(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'a.nrrd';write(p,np.zeros((4,4,4),np.uint8),'a')
            a=np.ones((2,2,2),np.uint8)
            e=snapshot(d/'a.npz',a,start=(4,4,4))
            with self.assertRaises(ValueError):nf.load_edit(d,e,nf.Nrrd(p))
            e=snapshot(d/'a.npz',a,translation=(0.5,0,0))
            with self.assertRaises(ValueError):nf.load_edit(d,e,nf.Nrrd(p))
            np.savez_compressed(d/'a.npz',mask=a,extent_xyz=[0,1,0,1,0,1],ijk_to_ras=np.eye(4)*2)
            e['sha256']=nf.digest(d/'a.npz')
            with self.assertRaises(ValueError):nf.load_edit(d,e,nf.Nrrd(p))

    def test_empty_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'a.nrrd';write(p,np.zeros((4,4,4),np.uint8),'a')
            e=snapshot(d/'a.npz',np.empty((0,0,0),np.uint8))
            self.assertFalse(np.any(nf.load_edit(d,e,nf.Nrrd(p))))

    def test_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('../escape','/absolute'):
                with self.assertRaises(ValueError):nf.child(Path(d),name)


class TestComponents(unittest.TestCase):
    def test_random_slab_equivalence(self):
        rng=np.random.default_rng(1803)
        for case in range(18):
            shape=tuple(int(x) for x in rng.integers(2,15,3))
            a=(rng.random(shape)<(0.08+case*.04)).astype(np.uint8)
            reference,n=ndi.label(a,structure=nf.FACE)
            for depth in (1,2,4,100):
                with self.subTest(case=case,depth=depth):
                    labels=np.zeros(shape,np.int32)
                    roots,counts,flags,original,clipped=nf.label_stream(nf.array_slabs(a,depth),labels,maximum=100000)
                    result=roots[labels]
                    self.assertEqual(np.count_nonzero(counts),n)
                    self.assertEqual(original,np.count_nonzero(a))
                    self.assertEqual(clipped,original)
                    self.assertTrue(np.array_equal(result==0,reference==0))
                    for k in range(1,n+1):
                        unique=np.unique(result[reference==k]);self.assertEqual(len(unique),1)
                        self.assertEqual(int(counts[unique[0]]),np.count_nonzero(reference==k))

    def test_cross_boundary_evidence_and_diagonals(self):
        a=np.zeros((9,7,7),np.uint8);a[:,3,3]=1;a[0,1,1]=1;a[1,2,2]=1
        low=np.zeros_like(a);low[0,3,3]=1
        removed=np.zeros_like(a);removed[-1,3,3]=1
        for depth in (1,2,4,9):
            labels=np.zeros_like(a,dtype=np.int32)
            roots,counts,flags,_,_=nf.label_stream(nf.array_slabs(a,depth),labels,maximum=1000,
                                                keep=low,removed=removed,support=nf.Support(a.shape,a.shape))
            self.assertEqual(np.count_nonzero(counts),3)
            main=roots[labels[0,3,3]]
            self.assertEqual(flags[main],3)
            self.assertEqual(counts[main],9)

    def test_sampling_support_matches_bruteforce(self):
        full=(7,9,11);low=(3,4,5)
        s=nf.Support(full,low)
        for index in ((0,0,0),(1,2,3),(2,3,4)):
            a=np.zeros(low,np.uint8);a[index]=1
            expected=np.zeros(full,bool)
            bounds=tuple(slice(i*n//m,((i+1)*n+m-1)//m) for i,n,m in zip(index,full,low))
            expected[bounds]=1
            self.assertTrue(np.array_equal(s.gather(a,0,full[0]),expected))
            self.assertTrue(np.array_equal(s.gather(a,2,3),expected[2:5]))

    def test_component_cap(self):
        a=np.zeros((4,4,4),np.uint8);a[::2,::2,::2]=1
        with self.assertRaises(ValueError):
            nf.label_stream(nf.array_slabs(a,1),np.zeros_like(a,dtype=np.int32),maximum=2)

    def test_native_global_intersection(self):
        a=np.ones((5,4,4),np.uint8);g=np.zeros_like(a);g[:,1,1]=1
        labels=np.zeros_like(a,dtype=np.int32)
        r,c,f,before,after=nf.label_stream(nf.array_slabs(a,2),labels,maximum=100,clip=g)
        self.assertEqual(before,80);self.assertEqual(after,5);self.assertEqual(np.count_nonzero(c),1)


class TestWorkflow(unittest.TestCase):
    def quiet_run(self,args):
        with redirect_stdout(io.StringIO()):return nf.run(args)

    def test_parallel_overlap_and_unchanged_layer_match_serial(self):
        shape = (12, 12, 12)
        left = np.zeros(shape, np.uint8); left[1:11, 2:4, 2:4] = 1
        right = np.zeros(shape, np.uint8); right[1:11, 2:4, 3:5] = 1
        bad = np.zeros(shape, np.uint8); bad[3:9, 8:10, 8:10] = 1
        arrays = {'left': left | bad, 'right': right | bad, 'unchanged': left}
        edits = {'left': pool(left, (6, 6, 6)), 'right': pool(right, (6, 6, 6)),
                 'unchanged': pool(left, (6, 6, 6))}
        with tempfile.TemporaryDirectory() as d:
            for workers in (1, 4):
                with self.subTest(workers=workers):
                    args = fixture(Path(d)/str(workers), arrays, edits)
                    args.workers = workers
                    self.assertEqual(self.quiet_run(args), 0)
                    actual = nf.Nrrd(Path(args.output)/'volume.nrrd').array()
                    np.testing.assert_array_equal(actual, left | right)
                    report = nf.load_json(Path(args.output)/'report.json')
                    self.assertEqual(report['workers'], workers)
                    self.assertEqual(len(report['source_full_sha256']), 4)
                    row = next(r for r in report['layers'] if r['name'].endswith('_unchanged'))
                    self.assertEqual(row['component_analysis'], 'skipped_no_deletion_evidence')
                    self.assertIsNone(row['native_components'])
                    self.assertEqual(row['sparse_label_bytes'], 0)
                    self.assertFalse(any(Path(args.scratch).iterdir()))

    def test_parallel_failure_never_publishes_native_label(self):
        with tempfile.TemporaryDirectory() as d:
            args, _, _ = complex_fixture(d)
            args.workers = 4
            path = Path(args.full_manifest).parent/'volume_component.seg.nrrd'
            header, offset, shape = nf.read_header(path)
            prefix = path.read_bytes()[:offset]
            path.write_bytes(prefix + gzip.compress(b'\x02' + b'\0'*(np.prod(shape)-1)))
            with self.assertRaisesRegex(ValueError, 'Nonbinary'):
                self.quiet_run(args)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())
            self.assertEqual(nf.load_json(Path(args.output)/'report.json')['status'], 'failed')
            self.assertFalse(any(Path(args.scratch).iterdir()))

    def test_native_detail_and_pta_shape(self):
        with tempfile.TemporaryDirectory() as d:
            args,good,bad=complex_fixture(d)
            before={p:nf.digest(p) for p in Path(d).rglob('*') if p.is_file()}
            self.assertEqual(self.quiet_run(args),0)
            output=Path(args.output)
            a=nf.Nrrd(output/'volume.nrrd').array()
            self.assertTrue(np.array_equal(a,good))
            self.assertFalse((output/'filtered_layers').exists())
            self.assertFalse((output/'candidates').exists())
            self.assertEqual(list(output.glob('*.nrrd')), [output/'volume.nrrd'])
            coarse=pool(good,(6,6,6));upsampled=coarse.repeat(2,0).repeat(2,1).repeat(2,2)
            self.assertFalse(np.array_equal(a,upsampled))
            for p,h in before.items():self.assertEqual(nf.digest(p),h)
            r=nf.load_json(output/'report.json');self.assertEqual(r['status'],'complete')
            self.assertEqual(r['final_global_voxels'],int(np.count_nonzero(good)))
            self.assertFalse(any(Path(args.scratch).iterdir()))

    def test_conflict_refuses_pta_publication(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[1:7,2:4,2:4]=1
            low=pool(a,(4,4,4));after=low.copy();after[2:]=0
            args=fixture(d,{'component':a},{'component':after})
            self.assertEqual(self.quiet_run(args),2)
            output=Path(args.output)
            self.assertFalse((output/'volume.nrrd').exists())
            self.assertEqual(nf.load_json(output/'report.json')['status'],'needs_review')
            self.assertFalse(any(Path(args.scratch).iterdir()))

    def test_explicit_keep_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[1:7,2:4,2:4]=1
            low=pool(a,(4,4,4));after=low.copy();after[2:]=0
            args=fixture(d,{'component':a},{'component':after});args.conflicts='keep'
            self.assertEqual(self.quiet_run(args),0)
            output=Path(args.output)
            self.assertTrue(np.array_equal(nf.Nrrd(output/'volume.nrrd').array(),a))
            self.assertEqual(nf.load_json(output/'report.json')['status'],'completed_with_unresolved_components')

    def test_empty_original_global_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[:,2:4,2:4]=1
            args=fixture(d,{'component':a},{'component':pool(a,(4,4,4))},global_mask=np.zeros_like(a))
            self.assertEqual(self.quiet_run(args),0)
            self.assertTrue(np.array_equal(nf.Nrrd(Path(args.output)/'volume.nrrd').array(),a))

    def test_all_segments_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.ones((8,8,8),np.uint8)
            args=fixture(d,{'component':a},{'component':np.zeros((4,4,4),np.uint8)})
            self.assertEqual(self.quiet_run(args),0)
            self.assertFalse(np.any(nf.Nrrd(Path(args.output)/'volume.nrrd').array()))

    def test_empty_edit_does_not_restore_unobserved_native_detail(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8)
            a[0:2,0:2,0:2]=1
            a[6,6,6]=1  # legitimate native detail suppressed by lossy preview
            args=fixture(d,{'component':a},{'component':np.zeros((4,4,4),np.uint8)})
            low_path=Path(d)/'low/volume_component.seg.nrrd'
            preview=pool(a,(4,4,4));preview[3,3,3]=0
            write(low_path,preview,'volume_component')
            global_path=Path(d)/'low/volume_Global_final_output.seg.nrrd'
            write(global_path,preview,'volume_Global_final_output')
            bundle_path=Path(args.edits)/'edits.json'
            bundle=nf.load_json(bundle_path)
            for entry in bundle['segments']:
                entry['source_sha256']=nf.digest(Path(d)/'low'/entry['source_filename'])
            nf.save_json(bundle_path,bundle)
            self.assertEqual(self.quiet_run(args),0)
            self.assertFalse(np.any(nf.Nrrd(Path(args.output)/'volume.nrrd').array()))
            report=nf.load_json(Path(args.output)/'report.json')
            self.assertTrue(report['edited_global_empty'])
            self.assertEqual(report['layers'][0]['unobserved_components_preserved'],0)
            self.assertIn('remove_empty_global',(Path(args.output)/'components.tsv').read_text())

    def test_edited_addition_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[:,2:4,2:4]=1
            after=pool(a,(4,4,4));after[1,1,2]=1
            args=fixture(d,{'component':a},{'component':after})
            with self.assertRaisesRegex(ValueError,'adds foreground'):self.quiet_run(args)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())
            self.assertEqual(nf.load_json(Path(args.output)/'report.json')['status'],'failed')

    def test_source_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d)
            source=next(Path(d,'low').glob('*.seg.nrrd'))
            source.write_bytes(source.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'changed since export'):self.quiet_run(args)
            self.assertFalse(Path(args.output).exists())

    def test_manifest_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d)
            p=Path(args.full_manifest);m=nf.load_json(p);m['layers'][0]['layer_key']='OTHER';nf.save_json(p,m)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):self.quiet_run(args)

    def test_nonadditive_default_refusal_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[:,2:4,2:4]=1
            args=fixture(d,{'checkpoint':a},{'checkpoint':pool(a,(4,4,4))},roles={'checkpoint':'select'})
            with self.assertRaisesRegex(ValueError,'not additive'):self.quiet_run(args)
            args.allow_nonadditive=True
            self.assertEqual(self.quiet_run(args),0)
            self.assertTrue(nf.load_json(Path(args.output)/'report.json')['nonadditive_layers_explicitly_included'])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d);Path(args.output).mkdir();sentinel=Path(args.output)/'x';sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'NEW directory'):self.quiet_run(args)
            self.assertEqual(sentinel.read_text(),'preserve')

    def test_native_largest_recomputed_not_coarse_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            a=np.zeros((8,8,8),np.uint8);a[1:7,2:4,2]=1
            b=np.zeros_like(a);b[3:5,2:4,4]=1  # native gap x=3, but low previews touch
            args=fixture(d,{'a':a,'b':b},{'a':pool(a,(4,4,4)),'b':pool(b,(4,4,4))})
            self.assertEqual(self.quiet_run(args),0)
            self.assertTrue(np.array_equal(nf.Nrrd(Path(args.output)/'volume.nrrd').array(),a))

    def test_fault_does_not_publish_and_cleans_scratch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d)
            with patch.object(nf, 'write_nrrd', side_effect=OSError('injected write failure')):
                with self.assertRaisesRegex(OSError,'injected'):self.quiet_run(args)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())
            self.assertFalse(any(Path(args.scratch).iterdir()))
            self.assertEqual(nf.load_json(Path(args.output)/'report.json')['status'],'failed')

    def test_corrupt_snapshot_checksum_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d)
            p=Path(args.edits)/'00000.npz';p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'snapshot checksum'):self.quiet_run(args)
            self.assertFalse((Path(args.output)/'volume.nrrd').exists())

    def test_native_preview_empty_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,_,_=complex_fixture(d)
            p=Path(d)/'full/volume_Global_final_output.seg.nrrd'
            write(p,np.zeros((12,12,12),np.uint8),'volume_Global_final_output')
            with self.assertRaisesRegex(ValueError,'emptiness disagrees'):self.quiet_run(args)


if __name__=='__main__':unittest.main(verbosity=2)
