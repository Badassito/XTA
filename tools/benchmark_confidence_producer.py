"""Paired legacy/portable/native confidence producers with one current writer.

Bounded independent production-raster fixtures preserve the exact same shared
CPU budget, publication queue and artifact bytes across arms. An explicit DLL
is qualification-only; this tool never installs or builds a native library.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import numpy as np
from XTA import confidence_capture_cpu as cpu
from XTA.confidence_capture import confidence_capture_resources,plan_confidence_capture
from XTA.confidence_evidence import _MaskedNativeScoreReader,ConfidenceEvidenceRef
from XTA.confidence_native_codec import native_confidence_codec
from XTA.confidence_publication import ConfidencePublicationQueue,copy_staged_blocks
from XTA.confidence_storage import write_blocks
from XTA.native_pull_scheduling import configure_native_pull_scheduler,shutdown_native_pull_scheduler
from tools.benchmark_native_pull_projection import heatsoak,PeakRss


class LegacyCellReader(_MaskedNativeScoreReader):
    def encode_frame(self,z,block_size):
        """Frozen pre-batch producer: one copy/tobytes/zlib operation per cell."""
        if self.stop is not None and self.stop.is_set():
            raise RuntimeError('Confidence capture cancelled')
        if not self.active[z]:
            return [],dict(scan_seconds=0.,compression_seconds=0.,known_voxels=0)
        y0,y1,x0,x1=map(int,self.boxes[z])
        started=time.perf_counter()
        bounds=cpu.masked_cell_bounds(self.mask,self.scores,z,y0,y1,x0,x1,block_size)
        scan=time.perf_counter()-started
        compression=0.
        records=[]
        for item in bounds:
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError('Confidence capture cancelled')
            a,b,c,d,known=map(int,item)
            started=time.perf_counter()
            crop=cpu.copy_masked_cell(self.mask,self.scores,z,a,b,c,d)
            scan+=time.perf_counter()-started
            started=time.perf_counter()
            encoded=zlib.compress(crop.tobytes(),level=3)
            compression+=time.perf_counter()-started
            records.append((a,b,c,d,known,encoded))
        return records,dict(scan_seconds=scan,compression_seconds=compression,
            known_voxels=int(bounds[:,4].sum()) if len(bounds) else 0,
            copy_kernel_calls=len(bounds),copy_batch_peak_bytes=0,
            codec_backend='legacy_python_zlib',codec_version=zlib.ZLIB_RUNTIME_VERSION,
            codec_python_version=zlib.ZLIB_RUNTIME_VERSION,native_compression_batches=0)


SOURCES=('XTA/confidence_capture.py','XTA/confidence_capture_cpu.py',
    'XTA/confidence_native_codec.py','XTA/confidence_evidence.py','XTA/confidence_storage.py',
    'XTA/confidence_publication.py','XTA/native_pull_scheduling.py','tools/benchmark_confidence_producer.py')
FILES=('scores.u8.zlib','index.bin','metadata.json')


def fingerprints():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):result.update(data)
    return result.hexdigest()


def inputs(parents,frames,side,bbox_side,cells,seed):
    if side%128 or bbox_side%128 or not 128<=bbox_side<=side or cells<4 or cells>(bbox_side//128)**2:
        raise ValueError('Fixture requires a 128-aligned plane and 4..grid-size active cells')
    result=[]
    grid=bbox_side//128
    origin=(side-bbox_side)//2
    if origin%128:raise ValueError('Fixture ROI origin must stay on the public 128 grid')
    rows,columns=np.indices((128,128))
    support=np.ones((128,128),np.uint8)
    for parent in range(parents):
        rng=np.random.default_rng(seed+parent)
        positions={0,grid-1,(grid-1)*grid,grid*grid-1}
        choices=[position for position in range(grid*grid) if position not in positions]
        positions.update(map(int,rng.choice(choices,cells-4,replace=False)))
        # Touch every byte so independent sources own private physical pages;
        # unmodified calloc pages would be a misleading shared-zero witness.
        mask=np.empty((frames,side,side),np.uint8);mask.fill(0)
        scores=np.empty_like(mask);scores.fill(0)
        for z in range(frames):
            for cell in sorted(positions):
                y,x=divmod(cell,grid);y=origin+y*128;x=origin+x*128
                mask[z,y:y+128,x:x+128]=support
                # Few flat confidence values reflect ~126 compressed bytes per
                # occupied cell in run 160591, rather than artificial noise.
                score=17+((rows//64+2*(columns//64)+z+parent)%6)*29
                scores[z,y:y+128,x:x+128]=score.astype(np.uint8)
        active=np.ones(frames,bool)
        boxes=np.tile(np.asarray((origin,origin+bbox_side,origin,origin+bbox_side),np.int64),(frames,1))
        for array in (mask,scores,active,boxes):array.setflags(write=False)
        result.append((mask,scores,active,boxes))
    return result


def input_identity(data):
    return [{name:hashlib.sha256(memoryview(array).cast('B')).hexdigest()
             for name,array in zip(('mask','scores','active','boxes'),item)} for item in data]


def decode_exact(path,item):
    mask,scores,_,_=item
    ref=ConfidenceEvidenceRef.open(path)
    with ref.native_reader() as read:
        for z in range(mask.shape[0]):
            values,known=read(z,z+1)
            expected=np.where(mask[z]!=0,scores[z],np.uint8(0))
            np.testing.assert_array_equal(values[0],expected)
            np.testing.assert_array_equal(known[0],expected!=0)


def run_group(root,arm,data,args):
    root.mkdir(parents=True,exist_ok=True)
    # Create shared parents before parallel producers. Windows sandbox ACL
    # initialization can otherwise race recursive mkdir on the same new root.
    (root/'stages').mkdir(exist_ok=True)
    os.environ['YOLO_TTA_CONFIDENCE_NATIVE_CODEC']='auto' if arm=='native' else 'python'
    reader_type=LegacyCellReader if arm=='legacy' else _MaskedNativeScoreReader
    queue=ConfidencePublicationQueue(max_pending_bytes=args.parents*64*1024**2,
                                      stage_bytes=64*1024**2,workers=2)
    plans=[plan_confidence_capture(item[0].shape,args.cpu_budget,
        workspace_bytes=args.workspace_mib*1024**2) for item in data]
    def capture(index):
        mask,scores,active,boxes=data[index]
        metrics={}
        destination=root/f'parent-{index}'
        with confidence_capture_resources(plans[index]):
            reader=reader_type(mask,scores,active,boxes)
        def writer(stage,limit):
            write_blocks(stage,scores.shape,reader,layer_key=f'parent-{index}',model_name='producer_fixture',
                provenance=dict(benchmark='paired-confidence-producer',parent=index),
                coordinate_space='native_view_processing',source_shape=scores.shape,
                metrics=metrics,max_numeric_bytes=limit)
        future=queue.stage_and_submit(root/'stages',writer,
            lambda stage,waited:copy_staged_blocks(stage,destination,metrics=metrics),metrics=metrics)
        if future is None:raise RuntimeError('Structured fixture unexpectedly exceeded its 64 MiB stage')
        future.result()
        return dict(parent=index,metrics=metrics,destination=str(destination))
    started=time.perf_counter();cpu_started=time.process_time()
    try:
        with PeakRss() as memory,ThreadPoolExecutor(max_workers=args.parents) as pool:
            results=list(pool.map(capture,range(args.parents)))
            queue.drain()
    except BaseException:
        queue.close(raise_errors=False)
        raise
    wall=time.perf_counter()-started
    process_cpu=time.process_time()-cpu_started
    state=queue.snapshot()
    queue.close()
    return dict(arm=arm,wall_seconds=wall,process_cpu_seconds=process_cpu,
        cpu_budget=args.cpu_budget,plans=[asdict(plan) for plan in plans],
        queue=state,parents=results,**memory.record())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--parents',type=int,default=4)
    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--side',type=int,default=2048)
    parser.add_argument('--bbox-side',type=int,default=1024)
    parser.add_argument('--cells',type=int,default=25)
    parser.add_argument('--cpu-budget',type=int,default=min(16,os.cpu_count() or 1))
    parser.add_argument('--workspace-mib',type=int,default=64)
    parser.add_argument('--max-input-mib',type=int,default=1024)
    parser.add_argument('--repetitions',type=int,default=2)
    parser.add_argument('--heatsoak-seconds',type=float,default=60.)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):parser.error('Output must be in sibling Scratch')
    if (min(args.parents,args.frames,args.side,args.cells,args.cpu_budget,args.repetitions)<1
            or args.heatsoak_seconds<60 or 2*args.parents*args.frames*args.side**2>args.max_input_mib*1024**2):
        parser.error('Require positive bounded inputs and at least 60 seconds heatsoak')
    if (output/'benchmark.json').exists():parser.error('Use a fresh evidence directory')
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'temporary';temporary.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temporary),TMP=str(temporary))
    if args.library:
        library=args.library.resolve()
        if not library.is_file():parser.error('Explicit qualification library is missing')
        os.environ.update(XTA_QUALIFY_NATIVE_ZLIB='1',XTA_TEST_NATIVE_ZLIB_LIBRARY=str(library))
    os.environ['YOLO_TTA_CONFIDENCE_NATIVE_CODEC']='auto'
    before=fingerprints()
    started=time.perf_counter()
    cpu.warm_confidence_capture_kernels()
    codec,receipt=native_confidence_codec()
    if codec is None:raise RuntimeError('Native comparison requires an actual backend: '+str(receipt))
    # Match actual readonly C-layout signatures and a complete-cell bound size.
    tiny=np.ones((1,129,131),np.uint8);tiny.setflags(write=False)
    bounds=cpu.masked_cell_bounds(tiny,tiny,0,0,129,0,131,128)
    cpu.copy_masked_cells_batch(tiny,tiny,0,bounds,256*1024)
    codec.encode_batch(tiny,tiny,0,bounds,128)
    setup_seconds=time.perf_counter()-started
    import psutil
    rss_before_inputs=psutil.Process().memory_info().rss
    input_started=time.perf_counter()
    data=inputs(args.parents,args.frames,args.side,args.bbox_side,args.cells,160591)
    identity=input_identity(data)
    input_setup_seconds=time.perf_counter()-input_started
    rss_after_inputs=psutil.Process().memory_info().rss
    report=dict(scope='bounded local producer/publication sanity check; no cluster walltime claim',
        shape=[args.frames,args.side,args.side],parents=args.parents,cells_per_frame=args.cells,
        input_storage='independent fully prefaulted private mask/score arrays',
        input_logical_bytes=2*args.parents*args.frames*args.side**2,
        input_rss_before_bytes=rss_before_inputs,input_rss_after_bytes=rss_after_inputs,
        input_setup_seconds_excluded=input_setup_seconds,
        actual_scan_area_pixels_per_frame=args.bbox_side**2,
        field_160591_scan_area_pixels_per_frame=9834884256/4573/2,
        field_160591_cells_per_frame=120859/4573,
        field_160591_payload_bytes_per_cell=15210007/120859,
        native_receipt=receipt,linux_loader_executed=os.name!='nt',
        setup_seconds_excluded=setup_seconds,source_before=before,input_before=identity,
        legacy_encode_frame_sha256=hashlib.sha256(inspect.getsource(LegacyCellReader.encode_frame).encode()).hexdigest(),
        records=[],warmup_records=[],compiler_signatures=dict(
            bounds=list(map(str,cpu.masked_cell_bounds.nopython_signatures)),
            cell_copy=list(map(str,cpu.copy_masked_cell.nopython_signatures)),
            batch_copy=list(map(str,cpu.copy_masked_cells_batch.nopython_signatures)),
            native_codec=list(map(str,codec._kernel.nopython_signatures))))
    scheduler=configure_native_pull_scheduler(workers=args.cpu_budget,max_workers=args.cpu_budget)
    try:
        report['heatsoak']=heatsoak(args.heatsoak_seconds,args.cpu_budget)
        for repetition in range(-1,args.repetitions):
            order=['legacy','portable','native'] if repetition%2==0 else ['native','portable','legacy']
            identities={}
            for arm in order:
                root=output/('warmup' if repetition<0 else f'pair-{repetition}')/arm
                record=run_group(root,arm,data,args)
                record['repetition']=repetition
                blocks=sum(parent['metrics']['block_count'] for parent in record['parents'])
                record['actual_payload_bytes_per_cell']=sum(parent['metrics']['payload_bytes'] for parent in record['parents'])/max(1,blocks)
                record['actual_cells_per_frame']=blocks/(args.parents*args.frames)
                record['artifacts']=[]
                for index,item in enumerate(data):
                    path=root/f'parent-{index}'
                    decode_exact(path,item)
                    artifact={name:digest(path/name) for name in FILES}
                    record['artifacts'].append(artifact)
                identities[arm]=record['artifacts']
                if repetition>=0:report['records'].append(record)
                else:report['warmup_records'].append(record)
                print(json.dumps(dict(arm=arm,repetition=repetition,wall_seconds=record['wall_seconds'],
                    codec=[parent['metrics'].get('codec_backend') for parent in record['parents']],
                    native_batches=[parent['metrics'].get('native_compression_batches') for parent in record['parents']])),flush=True)
            assert identities['legacy']==identities['portable']==identities['native'],'Paired artifacts differ'
        report['input_after']=input_identity(data)
        assert report['input_after']==identity
        report['source_after']=fingerprints()
        report['changed_during_run']=[name for name in before if before[name]!=report['source_after'][name]]
        assert not report['changed_during_run'],'Source changed during paired measurement'
        medians={arm:statistics.median(record['wall_seconds'] for record in report['records'] if record['arm']==arm)
                 for arm in ('legacy','portable','native')}
        report['median_seconds']=medians
        report['speedup_vs_legacy']={arm:medians['legacy']/value for arm,value in medians.items()}
        report['all_artifacts_byte_identical']=True
        (output/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(median_seconds=medians,speedup_vs_legacy=report['speedup_vs_legacy'])),flush=True)
    except BaseException as exc:
        report.update(error=repr(exc),source_after=fingerprints())
        (output/'benchmark-failed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    finally:shutdown_native_pull_scheduler(scheduler)
    return 0


if __name__=='__main__':raise SystemExit(main())
