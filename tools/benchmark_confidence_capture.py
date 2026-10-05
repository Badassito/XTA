"""Bounded, CPU-only archived/current native confidence capture comparison.

Default: write a plan without allocating score planes. --check-only verifies
bytes and inputs without throughput claims. --execute requires a recorded quiet
window authorization and at least 60 seconds of CPU heat soak. Each backend
runs in a fresh subprocess; no model, projection, CUDA, or full retained volume
is loaded. Generated files must be under sibling Scratch.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRATCH = ROOT.parent / 'Scratch'
from tools.source_archive_history import history_archive_path
TASK = SCRATCH / 'Experiments/Job150790_150798_Performance_20261002/confidence_benchmark'
ARCHIVE = history_archive_path('throughput')
ARCHIVE_SHA256 = '3cf2d7b0e759e873350c944297aa97d441fec3f4d617522f38df809a4e287e6d'
REAL_MANIFEST = SCRATCH / ('Experiments/Results/F3_10_4_30_2026_8bit_Y_150790/'
                           'reconciliation_evidence/manifest.json')
ARTIFACTS = ('scores.u8.zlib', 'index.bin', 'metadata.json')
BACKENDS = ('archived_serial', 'new_serial', 'new_parallel')
MIB = 1024**2
MAX_RETAINED_PAYLOAD_BYTES = 512*MIB
GUARD_FILES = ('tools/benchmark_confidence_capture.py', 'tools/qualify_release.py',
    'XTA/__init__.py', 'XTA/_deps.py', 'XTA/confidence_capture.py',
    'XTA/confidence_capture_cpu.py', 'XTA/confidence_evidence.py',
    'XTA/confidence_storage.py', 'XTA/json_publication.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(MIB), b''):
            digest.update(part)
    return digest.hexdigest()


def source_identity():
    return {name: digest_file(ROOT/name) for name in GUARD_FILES}


def executed_repository_modules():
    paths=set()
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name or '!/' in name:
            continue
        path=Path(name).resolve()
        if path.is_relative_to(ROOT) and path.suffix=='.py':
            paths.add(path.relative_to(ROOT).as_posix())
    return {name:digest_file(ROOT/name) for name in sorted(paths)}


def checked_output(path):
    target = Path(path).expanduser().resolve()
    if target == SCRATCH.resolve() or not target.is_relative_to(SCRATCH.resolve()):
        raise ValueError('Artifacts must live in a task directory under sibling Scratch')
    if target.is_relative_to(ROOT):
        raise ValueError('Repository output is prohibited')
    return target


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_archive(path, expected_sha256):
    path = Path(path).resolve()
    if len(expected_sha256) != 64 or digest_file(path) != expected_sha256.lower():
        raise ValueError('Archived baseline ZIP differs from its declared SHA256')
    return path


def archived_capture(path, expected_sha256=ARCHIVE_SHA256):
    """Execute unedited archived writer + exact AST-selected reader dependencies.

    A private package resolves the writer's relative evidence import without
    importing archived inference modules or shadowing the current XTA package.
    The selection changes no function/class AST and records all member hashes.
    """
    import numpy as np
    from dataclasses import asdict as dc_asdict, is_dataclass
    path = verify_archive(path, expected_sha256)
    needed = ('confidence_storage.py', 'confidence_evidence.py', 'json_publication.py')
    with zipfile.ZipFile(path) as archive:
        members = {}
        for filename in needed:
            matches = [name for name in archive.namelist()
                       if name.endswith('/XTA/' + filename) or name == 'XTA/' + filename]
            if len(matches) != 1:
                raise ValueError(f'Archive needs one unambiguous XTA/{filename}')
            members[filename] = (matches[0], archive.read(matches[0]))
    prefix = '_confidence_capture_archived_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    package = types.ModuleType(prefix)
    package.__path__ = []
    sys.modules[prefix] = package
    for filename in ('json_publication.py', 'confidence_evidence.py', 'confidence_storage.py'):
        module = types.ModuleType(prefix + '.' + filename[:-3])
        module.__package__ = prefix
        module.__file__ = str(path) + '!/' + members[filename][0]
        sys.modules[module.__name__] = module
        source = members[filename][1].decode('utf-8-sig')
        if filename == 'confidence_evidence.py':
            selected_names = {'SCORE_SEMANTICS', '_json_value', '_write_json_atomic',
                              '_MaskedNativeScoreReader'}
            tree = ast.parse(source, filename=module.__file__)
            nodes = [node for node in tree.body if getattr(node, 'name', None) in selected_names
                or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id in selected_names for target in node.targets)]
            if len(nodes) != len(selected_names):
                raise ValueError('Archive reader dependency selection is incomplete')
            module.__dict__.update(np=np, Path=Path, asdict=dc_asdict, is_dataclass=is_dataclass,
                write_json_atomic=sys.modules[prefix+'.json_publication'].write_json_atomic)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), module.__file__, 'exec'), module.__dict__)
        else:
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
    evidence = sys.modules[prefix+'.confidence_evidence']
    storage = sys.modules[prefix+'.confidence_storage']
    provenance = dict(zip_sha256=expected_sha256,
        members={name: dict(path=member, sha256=hashlib.sha256(data).hexdigest())
                 for name, (member, data) in members.items()},
        reader_loading='Unmodified AST nodes: SCORE_SEMANTICS, _json_value, _write_json_atomic, _MaskedNativeScoreReader')
    return storage.write_blocks, evidence._MaskedNativeScoreReader, provenance


def input_hashes(mask, scores, active, boxes):
    """Hash existing contiguous buffers in bounded chunks; no volume-sized copy."""
    values = {}
    for name, array in (('mask', mask), ('scores', scores), ('active', active), ('boxes', boxes)):
        if not array.flags.c_contiguous:
            raise ValueError('Benchmark fixtures require contiguous buffers')
        digest = hashlib.sha256()
        data = memoryview(array).cast('B')
        for start in range(0, len(data), MIB):
            digest.update(data[start:start+MIB])
        values[name] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=digest.hexdigest(),
                            bytes=int(array.nbytes), writeable=bool(array.flags.writeable))
    return values


def support_bounds(mask):
    import numpy as np
    active = np.zeros(mask.shape[0], bool)
    boxes = np.full((mask.shape[0], 4), -1, np.int64)
    for z in range(mask.shape[0]):
        rows = np.flatnonzero(np.any(mask[z], axis=1))
        if rows.size:
            columns = np.flatnonzero(np.any(mask[z], axis=0))
            active[z] = True
            boxes[z] = (rows[0], rows[-1]+1, columns[0], columns[-1]+1)
    return active, boxes


def synthetic_inputs(pattern, shape, seed):
    """Two owned uint8 volumes; random generation is at most one plane at a time."""
    import numpy as np
    if pattern not in ('sparse', 'dense', 'noisy'):
        raise ValueError('Unknown synthetic confidence pattern')
    mask = np.ones(shape, np.uint8) if pattern != 'sparse' else np.zeros(shape, np.uint8)
    scores = np.full(shape, 179, np.uint8)
    rng = np.random.default_rng(seed)
    for z in range(shape[0]):
        if pattern == 'noisy':
            scores[z] = rng.integers(0, 256, shape[1:], np.uint8)
        elif pattern == 'sparse' and z % 7 != 0:
            # Far-apart islands retain a wide support hull, stressing the old
            # whole-hull masked copy without requiring a full source volume.
            for row_fraction, column_fraction in ((.03,.05), (.45,.52), (.89,.93)):
                y, x = int(shape[1]*row_fraction), int(shape[2]*column_fraction)
                h, w = min(47, shape[1]-y), min(53, shape[2]-x)
                mask[z, y:y+h, x:x+w] = 1
                scores[z, y:y+h, x:x+w] = rng.integers(0, 256, (h,w), np.uint8)
                mask[z, y:y+min(h,3), x:x+min(w,5)] = 0
        scores[z, :min(2,shape[1]), :min(3,shape[2])] = 0
    return mask, scores, *support_bounds(mask)


def real_evidence_plan(manifest_path, frames, max_input_bytes):
    """Inspect a small manifest/index; never read a compressed payload here."""
    import numpy as np
    from XTA.confidence_storage import BLOCK_DTYPE, BLOCK_LAYOUT
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    candidates = []
    for layer in manifest.get('layers', ()):
        relative = Path(layer.get('directory', ''))
        directory = (manifest_path.parent/relative).resolve()
        if relative.is_absolute() or not directory.is_relative_to(manifest_path.parent):
            raise ValueError('Retained manifest directory escapes its evidence root')
        metadata_path = directory/'metadata.json'
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        shape = tuple(metadata.get('stored_shape_tyx', ()))
        if (metadata.get('layout') == BLOCK_LAYOUT and metadata.get('coordinate_space') == 'native_view_processing'
                and len(shape) == 3 and shape[1:] == (2048,2048)
                and (directory/'scores.u8.zlib').stat().st_size<=MAX_RETAINED_PAYLOAD_BYTES):
            candidates.append((str(metadata['layer_key']),directory,metadata))
    if not candidates:
        raise ValueError('No retained native 2048x2048 block layer fits the bounded compressed-payload cap')
    _, directory, metadata = sorted(candidates, key=lambda item:
        ('transverse' not in item[0], item[0]))[0]
    shape = tuple(map(int,metadata['stored_shape_tyx']))
    index_path = directory/'index.bin'
    count = int(metadata['block_count'])
    if index_path.stat().st_size != count*BLOCK_DTYPE.itemsize:
        raise ValueError('Retained evidence index differs from its metadata')
    index = np.memmap(index_path, mode='r', dtype=BLOCK_DTYPE, shape=(count,)) if count else np.empty(0,BLOCK_DTYPE)
    occupied = np.zeros(shape[0], bool)
    try:
        for first in range(0,count,65536):
            ids = index['z'][first:first+65536]
            if np.any(ids >= shape[0]):
                raise ValueError('Retained evidence frame index exceeds its native grid')
            occupied[ids] = True
    finally:
        mapping = getattr(index, '_mmap', None)
        if mapping is not None:
            mapping.close()
    available = np.flatnonzero(occupied)
    take = min(int(frames), int(available.size), max_input_bytes//(2*shape[1]*shape[2]))
    if take < 1:
        raise ValueError('No nonempty real score frame fits the declared input budget')
    positions = np.linspace(0,len(available)-1,take,dtype=np.int64)
    selected = list(map(int,available[positions]))
    return dict(name='retained150790_known_support', kind='retained', directory=str(directory),
        original_shape=list(shape), shape=[take,*shape[1:]], original_frame_ids=selected,
        frame_mapping=[dict(benchmark_frame=i, original_frame=z) for i,z in enumerate(selected)],
        layer_key=metadata['layer_key'], model_name=metadata['model_name'],
        support_note='Original raw prediction mask is not retained. Benchmark mask is decoded score>0; frames are explicitly subset and renumbered.',
        metadata_sha256=digest_file(directory/'metadata.json'), index_sha256=digest_file(index_path),
        declared_original_payload_sha256=metadata.get('payload_sha256'),
        source_payload_hard_cap_bytes=MAX_RETAINED_PAYLOAD_BYTES,
        payload_bytes=(directory/'scores.u8.zlib').stat().st_size)


def file_stats(directory):
    return {name: dict(bytes=(Path(directory)/name).stat().st_size,
                       mtime_ns=(Path(directory)/name).stat().st_mtime_ns) for name in ARTIFACTS}


def real_inputs(case, *, validation=None):
    import numpy as np
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    reference = ConfidenceEvidenceRef.open(case['directory'])
    if tuple(reference.storage_shape) != tuple(case['original_shape']):
        raise ValueError('Retained source shape changed after planning')
    if (digest_file(reference.path/'metadata.json') != case['metadata_sha256'] or
            digest_file(reference.path/'index.bin') != case['index_sha256']):
        raise ValueError('Retained source index/metadata changed after planning')
    before = file_stats(reference.path)
    if before['scores.u8.zlib']['bytes']>MAX_RETAINED_PAYLOAD_BYTES:
        raise ValueError('Retained compressed payload exceeds the bounded SHA/decoder cap')
    payload = reference.path/'scores.u8.zlib'
    declared_sha = case.get('declared_original_payload_sha256')
    if (not isinstance(declared_sha,str) or len(declared_sha)!=64 or
            any(value not in '0123456789abcdef' for value in declared_sha)
            or declared_sha != reference.metadata.get('payload_sha256')):
        raise ValueError('Retained evidence requires a valid declared payload SHA256')
    # Streaming checks never allocate/decode the original native volume and
    # remain outside all capture timings. Length/mtime cannot bind score bytes.
    payload_before_sha = digest_file(payload)
    if validation is not None:
        validation.update(declared_payload_sha256=declared_sha,
            payload_sha256_before_decode=payload_before_sha,
            payload_bytes=before['scores.u8.zlib']['bytes'],stream_chunk_bytes=MIB,
            outside_capture_timing=True)
    if payload_before_sha != declared_sha:
        raise ValueError('Retained evidence payload differs from its declared SHA256 before decoding')
    shape = tuple(case['shape'])
    mask, scores = np.empty(shape,np.uint8), np.empty(shape,np.uint8)
    with reference.native_reader() as reader:
        for target, source in enumerate(case['original_frame_ids']):
            plane, known = reader(source,source+1)
            scores[target] = plane[0]
            mask[target] = known[0]
    payload_after_sha = digest_file(payload)
    if validation is not None:
        validation['payload_sha256_after_decode']=payload_after_sha
    if payload_after_sha != declared_sha:
        raise RuntimeError('Retained evidence payload differs from its declared SHA256 after decoding')
    if file_stats(reference.path) != before:
        raise RuntimeError('Retained evidence changed while decoding selected frames')
    if validation is not None:
        validation['declared_and_actual_payload_exact']=True
    return mask,scores,*support_bounds(mask)


@contextmanager
def rss_measurement():
    import psutil
    process = psutil.Process()
    initial = process.memory_info()
    result = dict(start_rss_bytes=initial.rss, peak_sampled_rss_bytes=initial.rss,
                  sampling_interval_seconds=.01)
    stopped = threading.Event()
    def sample():
        while not stopped.wait(.01):
            result['peak_sampled_rss_bytes'] = max(result['peak_sampled_rss_bytes'],process.memory_info().rss)
    thread = threading.Thread(target=sample,name='confidence-benchmark-rss',daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stopped.set();thread.join()
        final = process.memory_info()
        result['end_rss_bytes'] = final.rss
        result['peak_sampled_rss_bytes'] = max(result['peak_sampled_rss_bytes'],final.rss)
        result['peak_sampled_increment_bytes'] = max(0,result['peak_sampled_rss_bytes']-initial.rss)
        result['process_lifetime_peak_rss_bytes'] = getattr(final,'peak_wset',None)
        result['note'] = 'RSS includes imports/JIT, inputs, metadata and capture; sampled increment is observational, not an admission guarantee.'


def expected_decoded_exact(directory, mask, scores):
    """Independent frame-sized masked-score oracle, outside capture timing."""
    import numpy as np
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    reference = ConfidenceEvidenceRef.open(directory)
    known_count = 0
    with reference.native_reader() as reader:
        for z in range(scores.shape[0]):
            actual, known = reader(z,z+1)
            expected = np.where(mask[z] != 0,scores[z],np.uint8(0))
            if not np.array_equal(actual[0],expected) or not np.array_equal(known[0],expected != 0):
                raise AssertionError(f'Decoded score/unknown evidence differs at frame {z}')
            known_count += int(np.count_nonzero(expected))
    if known_count != int(reference.metadata['known_voxels']):
        raise AssertionError('Published known-voxel count differs from independent oracle')
    return dict(decoded_frame_exact=True,known_voxels=known_count,
                max_oracle_frame_bytes=4*int(scores.shape[1]*scores.shape[2]))


def artifact_identity(directory):
    return {name: dict(bytes=(Path(directory)/name).stat().st_size,sha256=digest_file(Path(directory)/name))
            for name in ARTIFACTS}


def byte_equal(first, second):
    if Path(first).stat().st_size != Path(second).stat().st_size:
        return False
    with Path(first).open('rb') as left,Path(second).open('rb') as right:
        while True:
            a,b = left.read(MIB),right.read(MIB)
            if a != b:
                return False
            if not a:
                return True


def benchmark_child(spec):
    """Child owns one case/backend; input creation/JIT/oracles are not timed."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('CPU-only child must hide CUDA before numerical imports')
    import numpy as np
    output = checked_output(spec['output'])
    if (output/'receipt.json').exists():
        raise FileExistsError('Use a fresh child output; prior receipts and failed evidence are preserved')
    receipt = dict(schema='xta.confidence-capture-child/1',status='running',started_utc=utc_now(),
        backend=spec['backend'],case=spec['case'],source_before=source_identity(),records=[])
    if receipt['source_before'] != spec['source_guard']:
        raise RuntimeError('Source changed before child execution')
    output.mkdir(parents=True,exist_ok=True)
    path = output/'receipt.json'
    write_json(path,receipt,exclusive=True)
    try:
        case = spec['case']
        if case['kind']=='retained':
            receipt['retained_payload_validation']={}
            mask,scores,active,boxes=real_inputs(case,validation=receipt['retained_payload_validation'])
        else:
            mask,scores,active,boxes=synthetic_inputs(case['name'],tuple(case['shape']),spec['seed'])
        original = input_hashes(mask,scores,active,boxes)
        receipt['input_before'] = original
        area_max=max((int((box[1]-box[0])*(box[3]-box[2]))
                      for box in boxes[active]),default=0)
        receipt['workspace_note']=dict(
            borrowed_source_bytes=int(mask.nbytes+scores.nbytes),
            source_control_bytes=int(active.nbytes+boxes.nbytes),
            archived_max_masked_hull_result_bytes=area_max,
            archived_max_mask_predicate_bytes=area_max,
            note='Archived serial path has no explicit capture admission plan; its NumPy condition/result can span one support hull. New admission charges the complete bounded compressed frame window. RSS is observed separately and includes scientific input buffers plus runtime/JIT.')
        source_provenance = dict(benchmark='bounded_capture_only',case=case['name'],seed=spec['seed'])
        if case['kind']=='retained':
            source_provenance.update(original_frame_ids=case['original_frame_ids'],
                original_shape=case['original_shape'],support_note=case['support_note'])
        options = dict(layer_key='benchmark_pre_interpolation',model_name='confidence_fixture',
            provenance=source_provenance,coordinate_space='native_view_processing',
            source_shape=scores.shape,block_size=spec['block_size'])
        plan = None
        if spec['backend']=='archived_serial':
            writer,reader_class,receipt['archived_source'] = archived_capture(spec['archive'],spec['archive_sha256'])
        else:
            from XTA.confidence_capture import plan_confidence_capture,confidence_capture_resources
            from XTA.confidence_capture_cpu import warm_confidence_capture_kernels
            from XTA.confidence_evidence import _MaskedNativeScoreReader
            from XTA.confidence_storage import write_blocks
            requested = 1 if spec['backend']=='new_serial' else spec['workers']
            plan = plan_confidence_capture(scores.shape,requested,workspace_bytes=spec['workspace_bytes'],
                                            block_size=spec['block_size'])
            started = time.perf_counter()
            receipt['compiler_warmup'] = warm_confidence_capture_kernels()
            receipt['compiler_warmup']['elapsed_seconds'] = time.perf_counter()-started
            receipt['compiler_warmup']['excluded_from_capture_timing'] = True
            receipt['admitted_plan'] = asdict(plan)
            writer,reader_class = write_blocks,_MaskedNativeScoreReader
        receipt['executed_modules_before_capture']=executed_repository_modules()
        # The actual Numba signatures use readonly source views; warm helper
        # prepares both readonly/writeable tiny variants, without a disk cache.
        for sample in range(-spec['warmups'],spec['repetitions']):
            destination = output/('warmup_'+str(-sample) if sample<0 else 'sample_'+str(sample))
            metrics = {}
            with rss_measurement() as rss:
                cpu,start = time.process_time(),time.perf_counter()
                if plan is None:
                    reader = reader_class(mask,scores,active,boxes)
                    writer(destination,scores.shape,reader,metrics=metrics,**options)
                else:
                    with confidence_capture_resources(plan):
                        reader = reader_class(mask,scores,active,boxes)
                        writer(destination,scores.shape,reader,metrics=metrics,**options)
                wall,process_cpu = time.perf_counter()-start,time.process_time()-cpu
            identity = artifact_identity(destination)
            decoded = expected_decoded_exact(destination,mask,scores)
            after = input_hashes(mask,scores,active,boxes)
            if after != original:
                raise AssertionError('Capture changed input bytes, controls or writeability')
            row = dict(sample=sample,wall_seconds=wall,process_cpu_seconds=process_cpu,
                       rss=rss,metrics=metrics,artifact_identity=identity,oracle=decoded,
                       input_immutable=True)
            if sample>=0:
                receipt['records'].append(row)
                if sample==0:
                    receipt['retained_output'] = str(destination)
                    receipt['artifact_identity'] = identity
                elif identity != receipt['artifact_identity']:
                    raise AssertionError('Repeated capture changed publication bytes')
            write_json(path,receipt)
            # Keep the first measured output and every failed attempt. Later
            # successful repetitions do not need duplicate noisy payload files.
            if sample != 0:
                if not destination.resolve().is_relative_to(output):
                    raise RuntimeError('Repeated artifact cleanup escaped this child output')
                shutil.rmtree(destination)
        receipt.update(status='passed',input_after=input_hashes(mask,scores,active,boxes),
            numpy_version=np.__version__,input_bytes=int(mask.nbytes+scores.nbytes))
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['executed_modules_after_capture']=executed_repository_modules()
        prior=receipt.get('executed_modules_before_capture',{})
        receipt['executed_modules_unchanged']=all(
            receipt['executed_modules_after_capture'].get(name)==sha for name,sha in prior.items())
        receipt['source_after'] = source_identity()
        receipt['source_unchanged'] = receipt['source_before']==receipt['source_after']
        receipt['finished_utc'] = utc_now()
        if not receipt['source_unchanged'] or not receipt['executed_modules_unchanged']:
            receipt['status']='failed_source_guard'
        write_json(path,receipt)
    if receipt['status'] != 'passed':
        raise RuntimeError('Child source guard changed')
    return receipt


def cpu_environment(output):
    from tools.qualify_release import _qualification_environment
    environment = _qualification_environment(output,cpu_only=True)
    # Isolate controls inherited from earlier pytest/telemetry runs. This tool
    # creates no RuntimeTelemetry object and no inference/backend state.
    for name in list(environment):
        if name.startswith(('YOLO_TTA_', 'XTA_TEST_')):
            environment.pop(name,None)
    environment.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    return environment


def run_child(output, spec):
    output=checked_output(output)
    if (output/'receipt.json').exists() or (output/'spec.json').exists() or (output/'process.log').exists():
        raise FileExistsError('Use a fresh child output; prior attempts must not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    spec_path = output/'spec.json'
    write_json(spec_path,spec)
    command = [sys.executable,'-B',str(Path(__file__).resolve()),'--child-spec',str(spec_path)]
    with (output/'process.log').open('w',encoding='utf-8') as log:
        completed = subprocess.run(command,cwd=ROOT,env=cpu_environment(output),stdout=log,
                                   stderr=subprocess.STDOUT,check=False)
    if completed.returncode:
        raise RuntimeError(f'Confidence child failed ({completed.returncode}); retained log: {output/"process.log"}')
    receipt = json.loads((output/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'passed':
        raise RuntimeError('Confidence child did not pass its source guard')
    return receipt


def cpu_heatsoak(seconds, workers):
    """Bounded independent float64 transforms load CPU threads; no GPU/BLAS."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    deadline = time.perf_counter()+seconds
    def load(worker):
        values = np.full(512*1024,.3+worker*.0001,np.float64)
        iterations = 0
        while time.perf_counter()<deadline:
            np.sin(values,out=values);np.add(values,.3,out=values)
            iterations += 1
        return iterations
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterations=list(pool.map(load,range(workers)))
    return dict(elapsed_seconds=time.perf_counter()-start,workers=workers,iterations=iterations,
        array_bytes_per_worker=4*MIB,method='CPU-only NumPy sin/add over independent 4MiB arrays',
        limitation='Local CPU preparation only; does not certify thermal stability or cluster throughput')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=TASK/'plan')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--archive-sha256',default=ARCHIVE_SHA256)
    parser.add_argument('--frames',type=int,default=16)
    parser.add_argument('--height',type=int,default=2048)
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--patterns',nargs='+',choices=('sparse','dense','noisy'),default=['sparse','dense','noisy'])
    parser.add_argument('--workers',type=int,default=min(16,os.cpu_count() or 1))
    parser.add_argument('--workspace-mib',type=int,default=256)
    parser.add_argument('--max-input-mib',type=int,default=512)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--seed',type=int,default=150790)
    parser.add_argument('--real-manifest',type=Path,default=REAL_MANIFEST)
    parser.add_argument('--no-real',action='store_true')
    parser.add_argument('--heatsoak-seconds',type=float,default=60.)
    parser.add_argument('--quiet-window-authorized',help='Record root authorization and planned quiet window; required only for --execute')
    parser.add_argument('--child-spec',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.child_spec:
        return args
    if (min(args.frames,args.height,args.width,args.workers,args.workspace_mib,args.repetitions,
            args.max_input_mib,args.block_size)<1 or args.warmups<0 or args.block_size>65535):
        parser.error('Require positive shape/budgets/workers/repetitions, bounded block, nonnegative warmups')
    if args.workers>min(64,max(1,os.cpu_count() or 1)):
        parser.error('Requested workers exceed the local logical CPU count or bounded 64-worker probe cap')
    if args.max_input_mib>512 or 2*args.frames*args.height*args.width>args.max_input_mib*MIB:
        parser.error('Mask+score inputs must fit the declared hard cap (at most 512MiB)')
    if not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds<0:
        parser.error('Heat soak must be finite and nonnegative')
    if args.execute and (args.heatsoak_seconds<60 or not args.quiet_window_authorized):
        parser.error('Formal timing requires >=60s CPU heat soak and --quiet-window-authorized')
    args.output_dir=checked_output(args.output_dir)
    if (args.output_dir/'benchmark.json').exists():
        parser.error('Use a fresh output directory; failed receipts are preserved')
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.child_spec:
        spec=json.loads(checked_output(args.child_spec).read_text(encoding='utf-8'))
        if spec.get('kind')=='heatsoak':
            receipt=checked_output(spec['receipt'])
            if (not math.isfinite(spec['seconds']) or spec['seconds']<60 or
                    not 1<=spec['workers']<=min(64,max(1,os.cpu_count() or 1))):
                raise ValueError('Internal heat soak must retain the formal duration/worker bounds')
            write_json(receipt,dict(status='running',started_utc=utc_now()),exclusive=True)
            write_json(receipt,cpu_heatsoak(spec['seconds'],spec['workers']))
        else:
            benchmark_child(spec)
        return 0
    output=args.output_dir
    if (output/'benchmark.json').exists():
        raise FileExistsError('Use a fresh benchmark output; prior receipts are preserved')
    output.mkdir(parents=True,exist_ok=True)
    receipt=dict(schema='xta.confidence-capture-benchmark/1',status='planning',started_utc=utc_now(),
        scope='Native retiring mask/uint8-score capture to ordered compressed blocks; no model/projection/full-volume/cluster-wall claim',
        mode='execute' if args.execute else 'check_only' if args.check_only else 'plan_only',
        python=sys.version,platform=platform.platform(),logical_cpus=os.cpu_count(),
        source_before=source_identity(),children=[],comparisons=[])
    path=output/'benchmark.json'
    write_json(path,receipt,exclusive=True)
    try:
        archive=verify_archive(args.archive,args.archive_sha256)
        receipt['archive']=dict(path=str(archive),sha256=args.archive_sha256)
        from XTA.confidence_capture import plan_confidence_capture
        cases=[dict(name=pattern,kind='synthetic',shape=[args.frames,args.height,args.width])
               for pattern in dict.fromkeys(args.patterns)]
        if not args.no_real:
            if args.real_manifest.is_file():
                cases.append(real_evidence_plan(args.real_manifest,args.frames,args.max_input_mib*MIB))
            else:
                receipt['real_unavailable']=str(args.real_manifest)
        receipt['cases']=cases
        receipt['plans']=[dict(case=case['name'],serial=asdict(plan_confidence_capture(case['shape'],1,
            workspace_bytes=args.workspace_mib*MIB,block_size=args.block_size)),
            parallel=asdict(plan_confidence_capture(case['shape'],args.workers,
            workspace_bytes=args.workspace_mib*MIB,block_size=args.block_size))) for case in cases]
        receipt['input_hard_cap_bytes']=args.max_input_mib*MIB
        receipt['measurement_note']='Fresh process per backend/case. Backend order rotates by case; repetitions within a child are consecutive. Setup, JIT warmup, decoding, input hashing and parity checks are outside capture timing. Parallel phase metrics sum worker durations; use measured wall time for elapsed throughput. OS file cache and local machine differ from cluster.'
        if not (args.execute or args.check_only):
            receipt['status']='planned'
            return 0
        if args.execute:
            receipt['quiet_window_authorized']=args.quiet_window_authorized
            heat=output/'heatsoak';heat.mkdir()
            spec_path=heat/'spec.json'
            write_json(spec_path,dict(kind='heatsoak',seconds=args.heatsoak_seconds,
                                     workers=args.workers,receipt=str(heat/'receipt.json')))
            with (heat/'process.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child-spec',str(spec_path)],
                    env=cpu_environment(heat),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['heatsoak']=json.loads((heat/'receipt.json').read_text(encoding='utf-8'))
        for number,case in enumerate(cases):
            receipts={}
            order=BACKENDS[number%len(BACKENDS):]+BACKENDS[:number%len(BACKENDS)]
            for backend in order:
                print(f'Confidence {receipt["mode"]}: {case["name"]}/{backend}',flush=True)
                child=output/case['name']/backend
                spec=dict(output=str(child),case=case,backend=backend,seed=args.seed,
                    archive=str(archive),archive_sha256=args.archive_sha256,workers=args.workers,
                    workspace_bytes=args.workspace_mib*MIB,block_size=args.block_size,
                    repetitions=args.repetitions if args.execute else 1,
                    warmups=args.warmups if args.execute else 0,source_guard=receipt['source_before'])
                result=run_child(child,spec)
                receipts[backend]=result
                receipt['children'].append(dict(case=case['name'],backend=backend,path=str(child/'receipt.json')))
                write_json(path,receipt)
            baseline=Path(receipts['archived_serial']['retained_output'])
            comparisons=[]
            for backend in BACKENDS[1:]:
                actual=Path(receipts[backend]['retained_output'])
                equality={name:byte_equal(baseline/name,actual/name) for name in ARTIFACTS}
                same_inputs=receipts[backend]['input_before']==receipts['archived_serial']['input_before']
                comparisons.append(dict(backend=backend,exact_files=equality,same_input_bytes=same_inputs))
                if not all(equality.values()) or not same_inputs:
                    raise AssertionError(f'Archived/current capture differs: {case["name"]}/{backend}')
            receipt['comparisons'].append(dict(case=case['name'],comparisons=comparisons))
            if args.execute:
                receipt.setdefault('summary',{})[case['name']]={backend:dict(
                    median_wall_seconds=statistics.median(row['wall_seconds'] for row in item['records']),
                    median_process_cpu_seconds=statistics.median(row['process_cpu_seconds'] for row in item['records']),
                    peak_sampled_rss_bytes=max(row['rss']['peak_sampled_rss_bytes'] for row in item['records']),
                    admitted_workers=item.get('admitted_plan',{}).get('workers',1),
                    admitted_workspace_bytes=item.get('admitted_plan',{}).get('workspace_bytes'),
                    samples=len(item['records'])) for backend,item in receipts.items()}
            write_json(path,receipt)
        receipt['status']='passed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(source_after=source_identity(),finished_utc=utc_now())
        receipt['source_unchanged']=receipt['source_before']==receipt['source_after']
        if 'archive' in receipt:
            receipt['archive_sha256_after']=digest_file(receipt['archive']['path'])
            receipt['archive_unchanged']=receipt['archive_sha256_after']==receipt['archive']['sha256']
        if not receipt['source_unchanged']:
            receipt['status']='failed_source_guard'
        if receipt.get('archive_unchanged') is False:
            receipt['status']='failed_archive_guard'
        write_json(path,receipt)
        if receipt['status'] in ('failed_source_guard','failed_archive_guard'):
            raise RuntimeError('Current source or baseline archive changed during the probe')
    if receipt['status']=='failed_source_guard':
        raise RuntimeError('Source changed while confidence benchmark was executing')
    print(f'Confidence capture {receipt["status"]}: {path}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
