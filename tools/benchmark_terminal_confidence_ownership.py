"""CPU-only paired terminal confidence ownership qualification.

Uses the saved pre-edit orchestration as the dual-borrow witness, unchanged
actual readers/native codec, and the actual binary packer. Candidate timing
includes preparation of its frozen support-domain reader and premask. Defaults to a plan; --check-only makes no speed
claim. --execute requires a recorded quiet window and >=60s CPU heat soak.
All generated artifacts belong under sibling Scratch. No GPU is initialized.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / 'Scratch'
sys.path.insert(0, str(ROOT))
MIB = 1024**2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            value.update(block)
    return value.hexdigest()


def source_identity():
    names = list((ROOT / 'XTA').glob('confidence*.py')) + [
        ROOT / 'XTA/native_pull_scheduling.py', ROOT / 'XTA/projection_queue.py',
        ROOT / 'XTA/packed_tilted_projection.py', ROOT / 'XTA/assembly.py',
        ROOT / 'XTA/packed_dense_source.py', ROOT / 'XTA/interpolation.py',
        ROOT / 'XTA/publication_memory.py', ROOT / 'XTA/runtime.py',
        Path(__file__)]
    return {str(p.relative_to(ROOT)): digest(p) for p in names}


def baseline_orchestration(directory):
    """Execute actual pre-edit ownership code; prove reader/writer equality."""
    receipts = json.loads((directory / 'source-hashes.json').read_text())
    expected = {str(row['Path']).replace('\\','/').rsplit('/',1)[-1]: row['Hash'].lower()
                for row in receipts}
    for name in ('confidence_evidence.py', 'confidence_orchestration.py',
                 'confidence_storage.py', 'confidence_capture.py'):
        if digest(directory / name) != expected[name]:
            raise RuntimeError(f'Frozen baseline hash mismatch: {name}')
    def node(path, name):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        return ast.dump(next(n for n in tree.body if getattr(n, 'name', None) == name),
                        include_attributes=False)
    for name in ('_MaskedNativeScoreReader',):
        if node(directory / 'confidence_evidence.py', name) != node(ROOT / 'XTA/confidence_evidence.py', name):
            raise RuntimeError(f'Legacy capture node changed from baseline: {name}')
    for name in ('confidence_capture.py',):
        if digest(ROOT / 'XTA' / name) != expected[name]:
            raise RuntimeError(f'Paired capture shared dependency changed: {name}')
    spec = importlib.util.spec_from_file_location('XTA._terminal_ownership_baseline',
                                                  directory / 'confidence_orchestration.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._capture_source = directory / 'confidence_evidence.py'
    storage_spec = importlib.util.spec_from_file_location('XTA._terminal_storage_baseline',
                                                         directory / 'confidence_storage.py')
    frozen_storage = importlib.util.module_from_spec(storage_spec)
    sys.modules[storage_spec.name] = frozen_storage
    storage_spec.loader.exec_module(frozen_storage)
    from XTA import confidence_storage as storage
    # Publication fallback catches the production public exception identity.
    frozen_storage.ConfidenceStageLimit = storage.ConfidenceStageLimit
    module._storage = frozen_storage
    return module, expected


class Observation:
    """Scalar-only callbacks avoid retaining source roots under observation."""
    def __init__(self, source_bytes, parents, additional_parent_bytes):
        self.lock = threading.RLock()
        self.origin = time.perf_counter()
        self.initial_bytes = source_bytes
        self.live_bytes = source_bytes
        self.additional_parent_bytes = additional_parent_bytes
        self.events = []
        self.sources = {}
        self.parents = [{} for _ in range(parents)]
        self.samples = []
        self.monitor_errors = []
        self.stop = threading.Event()
        self.admission_capacity_seconds = None

    def event(self, label, parent=None, **values):
        with self.lock:
            row = dict(seconds=time.perf_counter()-self.origin, event=label, parent=parent, **values)
            self.events.append(row)
            return row['seconds']

    def retired(self, parent, kind, size):
        with self.lock:
            self.live_bytes -= size
            self.sources[f'{parent}:{kind}'] = self.event('source_root_retired', parent, kind=kind, bytes=size)
            if (self.admission_capacity_seconds is None
                    and self.initial_bytes-self.live_bytes >= self.additional_parent_bytes):
                self.admission_capacity_seconds = self.event('source_only_capacity_proxy', bytes=self.additional_parent_bytes)

    def monitor(self, evidence, scheduler):
        import psutil
        process = psutil.Process()
        try:
            while not self.stop.is_set():
                publication = evidence.confidence_publication_snapshot()
                with self.lock:
                    self.samples.append(dict(seconds=time.perf_counter()-self.origin,
                        rss_bytes=process.memory_info().rss, unique_source_bytes=self.live_bytes,
                        publication=publication, shared_cpu=scheduler.snapshot()))
                self.stop.wait(.02)
        except BaseException:
            self.monitor_errors.append(traceback.format_exc())


def fixture(shape, pattern, parent, seed):
    import numpy as np
    rng = np.random.default_rng(seed + parent)
    mask = np.empty(shape, np.uint8)
    mask.fill(0)  # Touch every page; do not benchmark shared lazy zero pages.
    scores = np.empty(shape, np.uint8)
    active = np.zeros(shape[0], bool)
    boxes = np.full((shape[0], 4), -1, np.int64)
    for z in range(shape[0]):
        scores[z] = rng.integers(1, 256, shape[1:], np.uint8) if pattern == 'noisy' else 173
        if z % 7 == 0:
            continue  # Deliberate nonzero background on inactive slices.
        if pattern in ('dense', 'noisy'):
            mask[z] = 255
        elif pattern == 'compact':
            # About19.4% of each active plane; retain the full2048-square source.
            y0,x0 = int(shape[1]*.2),int(shape[2]*.2)
            y1,x1 = min(shape[1],y0+max(1,int(shape[1]*.44))),min(shape[2],x0+max(1,int(shape[2]*.44)))
            mask[z,y0:y1,x0:x1] = 173
        else:
            for fy, fx in ((.02, .03), (.45, .51), (.86, .88)):
                y, x = int(shape[1]*fy), int(shape[2]*fx)
                mask[z, y:min(y+47,shape[1]), x:min(x+53,shape[2])] = 173
            mask[z, 127:min(130,shape[1]), 127:min(130,shape[2])] = 255
        mask[z, 1::31, 2::29] = 0
        scores[z, ::17, ::19] = 0  # True binary support with zero confidence.
        rows = np.flatnonzero(np.any(mask[z], axis=1))
        cols = np.flatnonzero(np.any(mask[z], axis=0))
        if rows.size:
            active[z] = True
            boxes[z] = (rows[0], rows[-1]+1, cols[0], cols[-1]+1)
    return mask, scores, active, boxes


def run_arm(args, output, arm, pattern, shape, baseline, *, warmup=False, gated=False):
    import numpy as np
    from XTA import confidence_evidence as evidence, confidence_storage as storage, geometry
    from XTA.confidence_capture import plan_confidence_capture, confidence_capture_resources
    from XTA.native_pull_scheduling import configure_native_pull_scheduler, shutdown_native_pull_scheduler
    from XTA.packed_tilted_projection import pack_terminal_native_source
    from XTA.runtime import release_memfd_owners_under
    from XTA.confidence_storage import BlockScoreReader
    from XTA.interpolation import RawBBoxMaskStore
    output.mkdir(parents=True,exist_ok=True)
    (output/'temporary'/'confidence_evidence').mkdir(parents=True,exist_ok=True)
    (output/'out'/'reconciliation_evidence').mkdir(parents=True,exist_ok=True)
    plan = plan_confidence_capture(shape, args.workers, workspace_bytes=args.workspace_mib*MIB)
    size = int(np.prod(shape))
    fixtures = [fixture(shape, pattern, p, args.seed) for p in range(args.parents)]
    input_hashes = [[hashlib.sha256(memoryview(a).cast('B')).hexdigest() for a in f] for f in fixtures]
    support = []
    for _,_,active,boxes in fixtures:
        selected = boxes[active]
        cells = int(np.sum((selected[:,1]-selected[:,0])*(selected[:,3]-selected[:,2])))
        support.append(dict(active_frames=int(active.sum()),total_frames=shape[0],
            active_bbox_cells=cells,full_grid_cells=size,
            bbox_fraction_all_frames=cells/size,
            bbox_fraction_active_frames=cells/max(1,int(active.sum())*shape[1]*shape[2])))
    del _,active,boxes,selected
    base_view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    scheduler = configure_native_pull_scheduler(workers=args.workers, max_workers=args.workers)
    current_writer = storage.write_blocks
    if arm == 'dual_borrow': storage.write_blocks = baseline._storage.write_blocks
    evidence.configure_confidence_evidence(output/'out', enabled=True,
        defer_projection=True, background_publication=True)
    queue = evidence.configure_terminal_confidence_capture(slots=args.parents,
        workspace_bytes=args.parents*plan.workspace_bytes, source_bytes=2*size*args.parents)
    if not queue:
        raise RuntimeError('Terminal capture queue not configured')
    if arm == 'dual_borrow':
        queue.close()
        queue = baseline.ConfidenceCaptureQueue(slots=args.parents,
            workspace_bytes=args.parents*plan.workspace_bytes, source_bytes=2*size*args.parents)
        evidence._CAPTURE_QUEUE = queue
    observed = Observation(2*size*args.parents, args.parents, 2*size)
    capture_metrics = []
    register = evidence._register_confidence_reference
    def record_reference(reference, *, metrics=None):
        with observed.lock:
            capture_metrics.append(dict(layer_key=reference.layer_key, **dict(metrics or {})))
        return register(reference, metrics=metrics)
    evidence._register_confidence_reference = record_reference
    scope = dict(vars(evidence))
    tree = ast.parse(baseline._capture_source.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if getattr(n,'name',None) in
             ('capture_prediction_confidence','publish_confidence_scores')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(baseline._capture_source),'exec'),scope)
    legacy_capture = scope['capture_prediction_confidence']
    gate = threading.Event()
    entered = threading.Event()
    packed_events = [threading.Event() for _ in range(args.parents)]
    gate_errors = []
    original_encode = evidence._MaskedNativeScoreReader.encode_frame
    def gated_encode(reader, *positional, **kwargs):
        entered.set()
        if not gate.wait(60): raise TimeoutError('Untimed ownership gate expired')
        return original_encode(reader, *positional, **kwargs)
    if gated: evidence._MaskedNativeScoreReader.encode_frame = gated_encode
    def check_gate():
        try:
            if not entered.wait(60):
                raise TimeoutError('Actual native reader did not enter ownership gate')
            if not all(event.wait(60) for event in packed_events):
                raise TimeoutError('Binary pack did not settle for ownership proof')
            with observed.lock:
                if any(event['event']=='capture_future_done' for event in observed.events):
                    raise AssertionError('Capture completed before the ownership gate was checked')
                retired = [f'{p}:mask' in observed.sources for p in range(args.parents)]
                if retired != [arm == 'score_only']*args.parents:
                    raise AssertionError(f'Mask lifetime differs from {arm}: {retired}')
                observed.event('gated_ownership_assertion_passed',mask_roots_retired=retired)
        except BaseException:
            gate_errors.append(traceback.format_exc())
        finally:
            gate.set()
    gate_thread = threading.Thread(target=check_gate,daemon=True) if gated else None
    finalizers = []
    for p, f in enumerate(fixtures):
        finalizers.extend(weakref.finalize(a, observed.retired, p, kind, a.nbytes)
                          for kind, a in zip(('mask', 'scores'), f[:2]))
    del f
    monitor = threading.Thread(target=observed.monitor, args=(evidence,scheduler), daemon=True)
    old_gc = gc.isenabled()
    gc.disable()
    def parent_work(p):
        mask, scores, active, boxes = fixtures[p]
        fixtures[p] = None
        row = observed.parents[p]
        kwargs = dict(view=replace(base_view, name=f'parent_{p}'), model_name='m',
            temp_dir=output/'temporary', known_slice_any=active, known_slice_bboxes=boxes)
        row['begin_seconds'] = observed.event('parent_begin', p)
        with confidence_capture_resources(plan):
            if arm == 'score_only':
                start = time.perf_counter()
                prepared = evidence.prepare_terminal_confidence_scores(mask, scores,
                    known_slice_any=active, known_slice_bboxes=boxes)
                row['premask_seconds'] = time.perf_counter()-start
                if prepared is None:
                    raise RuntimeError('Owned benchmark source unexpectedly declined score-only preparation')
                future = evidence.try_capture_terminal_confidence_scores(prepared,
                    retire=lambda _scores: observed.event('capture_retire_callback', p), **kwargs)
                prepared = None
            else:
                row['premask_seconds'] = 0.
                task = baseline.RetiringConfidenceCapture(mask, scores, plan=plan,
                    capture=legacy_capture,
                    retire=lambda _scores: observed.event('capture_retire_callback', p), kwargs=kwargs)
                future = queue.try_submit(task)
                task = None
        if future is None:
            raise RuntimeError('Funded capture unexpectedly declined; do not compare a fallback silently')
        scores = None
        row['capture_submitted_seconds'] = observed.event('capture_submitted', p)
        future.add_done_callback(lambda done: observed.event('capture_future_done', p))
        row['pack_begin_seconds'] = observed.event('pack_begin', p)
        row['pack'] = pack_terminal_native_source(mask, output/f'packed-{p}', desc=f'parent_{p}')
        row['pack_end_seconds'] = observed.event('pack_end', p)
        mask = None
        packed_events[p].set()
        future.result()
        return None
    try:
        observed.origin = time.perf_counter()
        cpu_start = time.process_time()
        monitor.start()
        if gate_thread: gate_thread.start()
        with ThreadPoolExecutor(max_workers=args.parents) as parents:
            list(parents.map(parent_work, range(args.parents)))
        evidence.drain_confidence_publication()
        wall = time.perf_counter()-observed.origin
        cpu = time.process_time()-cpu_start
        if gate_thread:
            gate_thread.join()
            if gate_errors: raise AssertionError(gate_errors)
        observed.stop.set(); monitor.join()
        if observed.monitor_errors or not observed.samples:
            raise RuntimeError(f'Resource monitor did not complete: {observed.monitor_errors}')
        if observed.live_bytes != 0 or any(f.alive for f in finalizers):
            raise AssertionError('Benchmark or capture still retains a dense source root')
        references = sorted(evidence._REGISTRY.values(), key=lambda ref: ref.layer_key)
        if len(references) != args.parents:
            raise AssertionError('Not every parent published evidence')
        hashes = [{name: digest(ref.path/name) for name in ('scores.u8.zlib','index.bin','metadata.json')}
                  for ref in references]
        packed_hashes = [{name:digest(output/f'packed-{p}'/name) for name in ('chunks.bin','index.bin')}
                         for p in range(args.parents)]
        metadata = [json.loads((ref.path/'metadata.json').read_text()) for ref in references]
        # Regenerate bounded oracles only after all measured source roots died.
        # Independent decode checks prevent two identically wrong arms passing.
        for p, reference in enumerate(references):
            expected_mask, expected_scores, _, _ = fixture(shape,pattern,p,args.seed)
            reader = BlockScoreReader(reference)
            packed = RawBBoxMaskStore.open(output/f'packed-{p}',mmap_payload=True)
            plane = np.empty(shape[1:],np.uint8)
            try:
                for z in range(shape[0]):
                    plane.fill(0)
                    for y0,y1,x0,x1,values in reader.iter_crops(z): plane[y0:y1,x0:x1]=values
                    if not np.array_equal(plane,np.where(expected_mask[z]!=0,expected_scores[z],np.uint8(0))):
                        raise AssertionError('Decoded confidence differs from independent masked-score oracle')
                    packed.fill_decoded_slice_into(z,plane)
                    if not np.array_equal(plane!=0,expected_mask[z]!=0):
                        raise AssertionError('Decoded packed mask differs from independent binary oracle')
            finally:
                reader.close(); packed.close()
            del expected_mask,expected_scores,plane
        state = scheduler.snapshot()
        if state['active_jobs'] or state['retained_operations']:
            raise AssertionError('Shared readers did not settle')
        record = dict(arm=arm, pattern=pattern, shape=shape, parents=args.parents,
            seconds=wall, process_cpu_seconds=cpu, plan=asdict(plan), input_sha256=input_hashes,
            observed_support=support,
            output_sha256=hashes, packed_sha256=packed_hashes, metadata=metadata,
            capture_metrics=capture_metrics,
            parent_timings=observed.parents, events=observed.events, samples=observed.samples,
            source_retired_seconds=observed.sources, shared_scheduler=state,
            unique_source_byte_seconds=sum(e['seconds']*e['bytes'] for e in observed.events
                                           if e['event']=='source_root_retired'),
            publication=evidence.confidence_publication_snapshot(),
            unique_source_logical_bytes=2*size*args.parents,
            caller_support_metadata_bytes=args.parents*shape[0]*33,
            capture_workspace_promise_bytes=args.parents*plan.workspace_bytes,
            premask_scratch_funded_upper_bound_bytes=min(plan.consumer_bytes,size) if arm=='score_only' else 0,
            packed_workspace_upper_bound_bytes=args.parents*(32*MIB+64*1024),
            packed_completed_resident_bytes=sum(p['pack']['resident_bytes'] for p in observed.parents),
            source_only_capacity_proxy_seconds=observed.admission_capacity_seconds,
            admission_scope='Not net admission: source-only byte proxy excludes packed retained output and independent workspace promises. Actual scheduler admission is separately qualified.',
            byte_scope='Unique source roots exact; queue promises and actual stage bytes sampled. Frame buffers bounded by funded plan; RSS sampled20ms includes allocator retention. Disk page cache is not exact resident accounting.',
            peak_sampled_rss_bytes=max((s['rss_bytes'] for s in observed.samples),default=None),
            independent_decoded_oracles_passed=True, gated_ownership_proof=gated,
            local_only=True, warmup=warmup)
        output.mkdir(parents=True, exist_ok=True)
        (output/'measurement.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    except BaseException:
        output.mkdir(parents=True,exist_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),
            events=observed.events,samples=observed.samples,parent_timings=observed.parents),indent=2)+'\n')
        raise
    finally:
        gate.set()
        observed.stop.set()
        if monitor.is_alive(): monitor.join()
        evidence.shutdown_confidence_publication(raise_errors=False)
        evidence._register_confidence_reference = register
        evidence._MaskedNativeScoreReader.encode_frame = original_encode
        storage.write_blocks = current_writer
        evidence.configure_confidence_evidence(None, enabled=False)
        shutdown_native_pull_scheduler(scheduler)
        release_memfd_owners_under(output)
        if old_gc: gc.enable()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--native-zlib-library',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--quiet-window',default='')
    parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--height',type=int,default=2048)
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--parents',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--workspace-mib',type=int,default=64)
    parser.add_argument('--repetitions',type=int,default=2)
    parser.add_argument('--heatsoak-seconds',type=float,default=60)
    parser.add_argument('--patterns',nargs='+',choices=('sparse','compact','dense','noisy'),default=['sparse','compact','dense','noisy'])
    parser.add_argument('--seed',type=int,default=160756)
    args = parser.parse_args()
    args.output_dir=Path(os.path.abspath(args.output_dir)).resolve()
    args.baseline_dir=Path(os.path.abspath(args.baseline_dir)).resolve()
    if not args.output_dir.is_relative_to(SCRATCH.resolve()) or args.output_dir.exists():
        parser.error('Use a fresh directory under sibling Scratch')
    if min(args.frames,args.height,args.width,args.parents,args.workers,args.workspace_mib,args.repetitions)<1:
        parser.error('Sizes, budgets and counts must be positive')
    if args.execute and (args.check_only or not args.quiet_window or args.heatsoak_seconds<60):
        parser.error('Execution needs quiet-window evidence, >=60s heatsoak and no --check-only')
    if args.check_only:
        args.frames,args.height,args.width,args.parents,args.repetitions=9,259,263,2,1
    args.output_dir.mkdir(parents=True)
    if not args.execute and not args.check_only:
        (args.output_dir/'plan.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
        NUMBA_CACHE_DIR=str(args.output_dir/'numba-cache'), YOLO_TTA_TELEMETRY_DIR=str(args.output_dir/'telemetry'))
    sys.dont_write_bytecode=True
    if args.native_zlib_library:
        os.environ.update(XTA_QUALIFY_NATIVE_ZLIB='1',XTA_TEST_NATIVE_ZLIB_LIBRARY=str(args.native_zlib_library.resolve()))
    import psutil
    from XTA.confidence_native_codec import native_confidence_codec
    from XTA.confidence_capture import plan_confidence_capture
    from tools.benchmark_native_pull_projection import heatsoak
    process=psutil.Process(); allowed=process.cpu_affinity(); process.cpu_affinity(allowed[:args.workers])
    codec,receipt=native_confidence_codec()
    if codec is None:
        raise RuntimeError(f'Actual native codec required; refusing silent fallback: {receipt}')
    baseline,baseline_hashes=baseline_orchestration(args.baseline_dir)
    pins=source_identity()
    shape=(args.frames,args.height,args.width)
    plan=plan_confidence_capture(shape,args.workers,workspace_bytes=args.workspace_mib*MIB)
    need=2*args.parents*args.frames*args.height*args.width+args.parents*(plan.workspace_bytes+96*MIB)+512*MIB
    if psutil.virtual_memory().available<need+512*MIB:
        raise RuntimeError('Insufficient physical headroom for bounded benchmark sources/workspaces')
    report=dict(configuration=vars(args),source_sha256=pins,baseline_sha256=baseline_hashes,
        codec=receipt,affinity=process.cpu_affinity(),conservative_memory_guard_bytes=need,
        limitations='Local CPU synthetic terminal stage only; no cluster or end-to-end throughput claim. Actual scheduler credit-return qualification is separate.',measurements=[])
    try:
        for arm in ('dual_borrow','score_only'):
            run_arm(args,args.output_dir/'warmup'/arm,arm,'sparse',(3,129,131),baseline,warmup=True)
        proof_args = argparse.Namespace(**vars(args))
        proof_args.parents = 1  # A blocked first lane must not starve a second proof lane.
        for arm in ('dual_borrow','score_only'):
            run_arm(proof_args,args.output_dir/'ownership-proof'/arm,arm,'sparse',(3,129,131),baseline,gated=True)
        if args.execute:
            report['heatsoak']=heatsoak(args.heatsoak_seconds,args.workers)
        for pattern in args.patterns:
            reference=None
            for repetition in range(args.repetitions):
                for arm in (('dual_borrow','score_only') if repetition%2==0 else ('score_only','dual_borrow')):
                    record=run_arm(args,args.output_dir/pattern/f'{arm}-{repetition}',arm,pattern,shape,baseline)
                    if (len(record['capture_metrics'])!=args.parents
                            or any(m.get('codec_backend')!=receipt['backend']
                                   or not m.get('native_compression_batches',0)
                                   for m in record['capture_metrics'])):
                        raise AssertionError('Actual captures did not exercise the qualified native codec')
                    signature=(record['input_sha256'],record['output_sha256'],record['packed_sha256'])
                    if reference is None: reference=signature
                    if signature!=reference:
                        raise AssertionError('Paired inputs, confidence bytes/metadata or packed binary bytes differ')
                    report['measurements'].append(record)
                    print(json.dumps(dict(pattern=pattern,arm=arm,repetition=repetition,
                        seconds=record['seconds'],source_retired_seconds=record['source_retired_seconds'])),flush=True)
        report['source_unchanged']=pins==source_identity()
        if not report['source_unchanged']: raise RuntimeError('Source changed during qualification')
        report['success']=True
        if args.execute:
            report['summary']=[dict(pattern=p,**{a:statistics.median(r['seconds'] for r in report['measurements']
                if r['arm']==a and r['pattern']==p) for a in ('dual_borrow','score_only')}) for p in args.patterns]
    finally:
        if sys.exc_info()[0] is not None: report['error']=traceback.format_exc()
        process.cpu_affinity(allowed)
        (args.output_dir/'benchmark.json').write_text(json.dumps(report,default=str,indent=2)+'\n')


if __name__=='__main__':
    main()
