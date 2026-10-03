"""Bounded capture probe controls, source identity and independent byte checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
import pytest

from tools import benchmark_confidence_capture as probe


@pytest.fixture
def scratch(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'SCRATCH',tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    return tmp_path


def test_plan_default_and_input_cap_precede_numeric_allocation(scratch):
    args=probe.parse_args(['--output-dir',str(scratch/'plan'),'--no-real'])
    assert not args.execute and not args.check_only
    assert (args.frames,args.height,args.width)==(16,2048,2048)
    assert 2*args.frames*args.height*args.width==128*probe.MIB
    for extra in (['--frames','65'],['--max-input-mib','513'],
                  ['--execute'],['--execute','--quiet-window-authorized','root','--heatsoak-seconds','59']):
        with pytest.raises(SystemExit):
            probe.parse_args(['--output-dir',str(scratch/'bad'),'--no-real',*extra])


def test_outputs_cannot_escape_scratch_or_overwrite_failed_receipt(scratch):
    with pytest.raises(ValueError):
        probe.checked_output(probe.ROOT/'generated')
    with pytest.raises(ValueError):
        probe.checked_output(scratch)
    output=scratch/'failed';output.mkdir()
    (output/'benchmark.json').write_text('{"status":"failed"}')
    with pytest.raises(SystemExit):
        probe.parse_args(['--output-dir',str(output),'--no-real'])
    assert json.loads((output/'benchmark.json').read_text())['status']=='failed'


@pytest.mark.parametrize('pattern',('sparse','dense','noisy'))
def test_synthetic_inputs_deterministic_exact_bounds_and_hashes(pattern):
    inputs=probe.synthetic_inputs(pattern,(9,137,149),17)
    repeated=probe.synthetic_inputs(pattern,(9,137,149),17)
    assert probe.input_hashes(*inputs)==probe.input_hashes(*repeated)
    mask,scores,active,boxes=inputs
    for z in range(mask.shape[0]):
        y,x=np.nonzero(mask[z])
        assert active[z]==bool(y.size)
        if y.size:
            assert tuple(boxes[z])==(y.min(),y.max()+1,x.min(),x.max()+1)
        else:
            assert np.all(boxes[z]==-1)
    before=probe.input_hashes(*inputs)
    scores[1,0,0]^=np.uint8(1)
    assert probe.input_hashes(*inputs)['scores']['sha256']!=before['scores']['sha256']


def test_archive_exact_member_selection_uses_private_package_and_unchanged_ast(scratch):
    archive=scratch/'baseline.zip'
    evidence='''SCORE_SEMANTICS = "fixture"
def _json_value(value):
    return value
def _write_json_atomic(path, value):
    write_json_atomic(path, _json_value(value), sort_keys=True, trailing_newline=True)
class _MaskedNativeScoreReader:
    def __init__(self,mask,scores,active,boxes):
        self.mask,self.scores,self.active,self.boxes=mask,scores,active,boxes
    def iter_crops(self,z):
        if self.active[z]:
            y0,y1,x0,x1=map(int,self.boxes[z])
            yield y0,y1,x0,x1,np.where(self.mask[z,y0:y1,x0:x1],self.scores[z,y0:y1,x0:x1],np.uint8(0))
raise AssertionError("Unselected archive top-level code must not execute")
'''
    with zipfile.ZipFile(archive,'w') as output:
        output.writestr('version/XTA/confidence_evidence.py',evidence)
        output.writestr('version/XTA/json_publication.py',
            'def write_json_atomic(path,value,**kwargs):\n    return (str(path),value,kwargs)\n')
        output.writestr('version/XTA/confidence_storage.py',
            'def write_blocks(*args,**kwargs):\n    from .confidence_evidence import SCORE_SEMANTICS\n    return SCORE_SEMANTICS\n')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    writer,reader_type,identity=probe.archived_capture(archive,sha)
    assert writer()=='fixture'
    mask,scores,active,boxes=probe.synthetic_inputs('sparse',(3,13,15),2)
    reader=reader_type(mask,scores,active,boxes)
    for z in range(3):
        for y0,y1,x0,x1,values in reader.iter_crops(z):
            np.testing.assert_array_equal(values,np.where(mask[z,y0:y1,x0:x1],scores[z,y0:y1,x0:x1],0))
    assert identity['members']['confidence_evidence.py']['sha256']==hashlib.sha256(evidence.encode()).hexdigest()
    assert writer.__module__.startswith('_confidence_capture_archived_')
    with pytest.raises(ValueError,match='SHA256'):
        probe.archived_capture(archive,'0'*64)


def test_streamed_byte_comparison_detects_different_equal_length_payload(scratch):
    first,second=scratch/'a',scratch/'b'
    first.write_bytes(b'x'*(probe.MIB+23))
    second.write_bytes(first.read_bytes())
    assert probe.byte_equal(first,second)
    with second.open('r+b') as stream:
        stream.seek(probe.MIB+11);stream.write(b'y')
    assert not probe.byte_equal(first,second)


def test_cpu_environment_drops_test_and_telemetry_leaks(scratch,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_PATH','outside.json')
    monkeypatch.setenv('XTA_TEST_SPHERICAL_CUDA','1')
    env=probe.cpu_environment(scratch/'environment')
    assert env['CUDA_VISIBLE_DEVICES']=='-1'
    assert 'YOLO_TTA_TELEMETRY_PATH' not in env and 'XTA_TEST_SPHERICAL_CUDA' not in env
    for key in ('TEMP','TMP','TMPDIR','NUMBA_CACHE_DIR','CUPY_CACHE_DIR','CUDA_CACHE_PATH'):
        assert Path(env[key]).is_relative_to(scratch/'environment')
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'  # Parent state is not mutated.


@pytest.mark.parametrize('backend',('new_serial','new_parallel'))
def test_child_real_writer_preserves_inputs_and_independent_decoded_scores(scratch,backend):
    spec=dict(output=str(scratch/backend),source_guard=probe.source_identity(),
        case=dict(name='noisy',kind='synthetic',shape=[3,17,23]),backend=backend,
        seed=91,workers=2,workspace_bytes=16*probe.MIB,block_size=8,warmups=0,repetitions=2)
    # The helper's public architecture is a fresh --child-spec interpreter.
    # Full pytest collection imports Torch pseudo-modules and can close runtime
    # singletons; neither state belongs in this capture/source-audit process.
    before=dict(os.environ)
    receipt=probe.run_child(scratch/backend,spec)
    assert dict(os.environ)==before
    assert receipt['status']=='passed' and receipt['source_unchanged']
    assert receipt['input_before']==receipt['input_after']
    assert len(receipt['records'])==2
    assert receipt['records'][0]['artifact_identity']==receipt['records'][1]['artifact_identity']
    assert receipt['records'][0]['oracle']['decoded_frame_exact']
    assert receipt['admitted_plan']['workers']==(1 if backend=='new_serial' else 2)
    assert not (scratch/backend/'sample_1').exists()
    assert all((Path(receipt['retained_output'])/name).is_file() for name in probe.ARTIFACTS)


def test_child_failure_retains_receipt_and_failed_output(scratch):
    spec=dict(output=str(scratch/'failure'),source_guard=probe.source_identity(),
        case=dict(name='dense',kind='synthetic',shape=[2,11,19]),backend='new_serial',
        seed=91,workers=1,workspace_bytes=16*probe.MIB,block_size=8,warmups=0,repetitions=1)
    output=scratch/'failure';output.mkdir()
    spec_path=output/'spec.json'
    probe.write_json(spec_path,spec)
    # Inject only the independent oracle failure in an otherwise fresh child.
    # The real argparse --child-spec path and source guard remain unchanged.
    script='''
import sys
from tools import benchmark_confidence_capture as probe
def fail(*args):
    raise AssertionError('Injected independent decode failure')
probe.expected_decoded_exact=fail
probe.main(['--child-spec',sys.argv[1]])
'''
    before=dict(os.environ)
    completed=subprocess.run([sys.executable,'-B','-c',script,str(spec_path)],cwd=probe.ROOT,
        env=probe.cpu_environment(output),capture_output=True,text=True,timeout=120)
    (output/'process.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    assert dict(os.environ)==before
    assert completed.returncode!=0 and 'Injected independent decode failure' in completed.stderr
    receipt=json.loads((scratch/'failure/receipt.json').read_text())
    assert receipt['status']=='failed' and 'Injected' in receipt['traceback']
    assert (scratch/'failure/sample_0/metadata.json').is_file()


def test_plan_only_cannot_launch_children_or_allocate_score_inputs(scratch,monkeypatch):
    monkeypatch.setattr(probe,'verify_archive',lambda path,sha:Path(path))
    monkeypatch.setattr(probe,'digest_file',lambda path:probe.ARCHIVE_SHA256)
    def forbidden(*args,**kwargs):
        raise AssertionError('Plan-only executed numeric capture')
    monkeypatch.setattr(probe,'synthetic_inputs',forbidden)
    monkeypatch.setattr(probe,'run_child',forbidden)
    assert probe.main(['--output-dir',str(scratch/'plan'),'--no-real'])==0
    receipt=json.loads((scratch/'plan/benchmark.json').read_text())
    assert receipt['status']=='planned' and receipt['children']==[]
    assert receipt['source_unchanged']


def test_retained_subset_decodes_only_requested_original_frames_and_renumbers(scratch,monkeypatch):
    from XTA.confidence_storage import write_blocks
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    root=scratch/'retained';root.mkdir()
    shape=(5,2048,2048)
    def original(z):
        if z not in (0,2,4):
            return None
        frame=np.zeros(shape[1:],np.uint8)
        frame[117:125,231:247]=71+z
        return frame
    write_blocks(root/'layer',shape,original,layer_key='transverse__retained',model_name='fixture',
        provenance={},coordinate_space='native_view_processing',source_shape=(7,19,23))
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps({'layers':[{'directory':'layer'}]}))
    case=probe.real_evidence_plan(manifest,2,32*probe.MIB)
    assert case['original_frame_ids']==[0,4]
    assert case['frame_mapping']==[{'benchmark_frame':0,'original_frame':0},
                                   {'benchmark_frame':1,'original_frame':4}]
    real=ConfidenceEvidenceRef.native_reader
    called=[]
    class RecordingReader:
        def __init__(self,delegate):self.delegate=delegate
        def __enter__(self):self.delegate.__enter__();return self
        def __exit__(self,*exc):return self.delegate.__exit__(*exc)
        def __call__(self,z0,z1):called.append((z0,z1));return self.delegate(z0,z1)
    monkeypatch.setattr(ConfidenceEvidenceRef,'native_reader',lambda self:RecordingReader(real(self)))
    validation={}
    mask,scores,active,boxes=probe.real_inputs(case,validation=validation)
    assert called==[(0,1),(4,5)]
    np.testing.assert_array_equal(mask,scores!=0)
    assert scores.shape==(2,2048,2048) and mask.nbytes+scores.nbytes==16*probe.MIB
    assert int(scores[0,117,231])==71 and int(scores[1,117,231])==75
    assert active.all() and np.all(boxes==[117,125,231,247])
    assert 'Original raw prediction mask is not retained' in case['support_note']
    assert validation['declared_and_actual_payload_exact']
    assert validation['payload_sha256_before_decode']==validation['payload_sha256_after_decode']==case['declared_original_payload_sha256']
    assert validation['outside_capture_timing'] and validation['stream_chunk_bytes']==probe.MIB


def small_retained_case(scratch):
    from XTA.confidence_storage import write_blocks
    root=scratch/'source';root.mkdir()
    shape=(1,2048,2048)
    plane=np.zeros(shape[1:],np.uint8);plane[117:125,231:247]=71
    write_blocks(root/'layer',shape,lambda z:plane,layer_key='transverse__fixture',model_name='m',
        provenance={},coordinate_space='native_view_processing',source_shape=(3,7,9))
    manifest=root/'manifest.json'
    manifest.write_text(json.dumps({'layers':[{'directory':'layer'}]}))
    return probe.real_evidence_plan(manifest,1,8*probe.MIB)


def substitute_valid_equal_length_payload(case):
    from XTA.confidence_storage import BLOCK_DTYPE
    path=Path(case['directory'])/'scores.u8.zlib'
    original=path.read_bytes()
    before=path.stat()
    index=np.frombuffer((Path(case['directory'])/'index.bin').read_bytes(),BLOCK_DTYPE)
    row=index[0]
    length=int(row['length'])
    replacement=zlib.compress(bytes([82])*(int(row['h'])*int(row['w'])),level=3)
    assert len(replacement)==length
    changed=original[:int(row['offset'])]+replacement+original[int(row['offset'])+length:]
    assert len(changed)==len(original) and changed!=original
    path.write_bytes(changed)
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert path.stat().st_size==before.st_size and path.stat().st_mtime_ns==before.st_mtime_ns
    assert zlib.decompress(replacement)==bytes([82])*128


def test_retained_same_size_mtime_valid_zlib_substitution_rejected_before_decode(scratch,monkeypatch):
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    case=small_retained_case(scratch)
    before=probe.file_stats(case['directory'])
    substitute_valid_equal_length_payload(case)
    assert probe.file_stats(case['directory'])==before
    def forbidden(*args):
        raise AssertionError('Corrupted retained payload reached decoding')
    monkeypatch.setattr(ConfidenceEvidenceRef,'native_reader',forbidden)
    validation={}
    with pytest.raises(ValueError,match='SHA256 before decoding'):
        probe.real_inputs(case,validation=validation)
    assert validation['payload_sha256_before_decode']!=validation['declared_payload_sha256']
    assert 'declared_and_actual_payload_exact' not in validation


def test_retained_same_size_mtime_substitution_after_decode_rejected(scratch,monkeypatch):
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    case=small_retained_case(scratch)
    real=ConfidenceEvidenceRef.native_reader
    class MutatingReader:
        def __init__(self,delegate):self.delegate=delegate
        def __enter__(self):self.delegate.__enter__();return self
        def __call__(self,*args):return self.delegate(*args)
        def __exit__(self,*exc):
            result=self.delegate.__exit__(*exc)
            substitute_valid_equal_length_payload(case)
            return result
    monkeypatch.setattr(ConfidenceEvidenceRef,'native_reader',lambda self:MutatingReader(real(self)))
    validation={}
    with pytest.raises(RuntimeError,match='SHA256 after decoding'):
        probe.real_inputs(case,validation=validation)
    assert validation['payload_sha256_before_decode']==validation['declared_payload_sha256']
    assert validation['payload_sha256_after_decode']!=validation['declared_payload_sha256']


def test_retained_payload_cap_prevents_large_file_hashing_or_decoding(scratch,monkeypatch):
    case=small_retained_case(scratch)
    real_stats=probe.file_stats
    def oversized(directory):
        result=real_stats(directory)
        result['scores.u8.zlib']['bytes']=probe.MAX_RETAINED_PAYLOAD_BYTES+1
        return result
    monkeypatch.setattr(probe,'file_stats',oversized)
    real_digest=probe.digest_file
    def checked_digest(path):
        if Path(path).name=='scores.u8.zlib':
            raise AssertionError('Oversized retained payload was hashed')
        return real_digest(path)
    monkeypatch.setattr(probe,'digest_file',checked_digest)
    with pytest.raises(ValueError,match='bounded SHA/decoder cap'):
        probe.real_inputs(case)


@pytest.mark.parametrize('status',('passed','failed'))
def test_prior_parent_and_child_receipts_and_logs_cannot_be_overwritten(scratch,monkeypatch,status):
    parent=scratch/'parent';parent.mkdir()
    child=scratch/'child';child.mkdir()
    retained=json.dumps({'status':status,'evidence':'must remain unchanged'}).encode()
    (parent/'benchmark.json').write_bytes(retained)
    (child/'receipt.json').write_bytes(retained)
    (child/'spec.json').write_bytes(b'old spec')
    (child/'process.log').write_bytes(b'old failed attempt log')
    monkeypatch.setattr(probe,'parse_args',lambda argv:SimpleNamespace(child_spec=None,output_dir=parent))
    with pytest.raises(FileExistsError,match='fresh benchmark'):
        probe.main([])
    with pytest.raises(FileExistsError,match='fresh child'):
        probe.benchmark_child({'output':str(child)})
    with pytest.raises(FileExistsError,match='fresh child'):
        probe.run_child(child,{})
    assert (parent/'benchmark.json').read_bytes()==retained
    assert (child/'receipt.json').read_bytes()==retained
    assert (child/'spec.json').read_bytes()==b'old spec'
    assert (child/'process.log').read_bytes()==b'old failed attempt log'


def test_receipt_claim_is_exclusive_and_heatsoak_cannot_overwrite_prior_receipt(scratch,monkeypatch):
    output=scratch/'exclusive';output.mkdir()
    receipt=output/'receipt.json'
    probe.write_json(receipt,{'status':'failed'},exclusive=True)
    before=receipt.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_json(receipt,{'status':'running'},exclusive=True)
    spec=output/'spec.json'
    spec.write_text(json.dumps({'kind':'heatsoak','seconds':60.,'workers':1,'receipt':str(receipt)}))
    def forbidden(*args):
        raise AssertionError('Existing heatsoak receipt did not stop CPU work')
    monkeypatch.setattr(probe,'cpu_heatsoak',forbidden)
    with pytest.raises(FileExistsError):
        probe.main(['--child-spec',str(spec)])
    assert receipt.read_bytes()==before
