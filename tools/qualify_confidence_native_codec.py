"""CPU-only functional qualification of bounded native confidence compression.

--library is an explicit local qualification override, never a production
dependency. Production Linux loader selection is reported separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
SOURCES=('XTA/confidence_native_codec.py','XTA/confidence_capture_cpu.py',
    'XTA/confidence_capture.py','XTA/confidence_evidence.py','XTA/confidence_storage.py',
    'XTA/confidence_publication.py','tests/test_confidence_native_codec.py',
    'tests/test_confidence_copy_batches.py','tools/qualify_confidence_native_codec.py')


def fingerprints():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--test-path',nargs='+',default=[
        'tests/test_confidence_native_codec.py','tests/test_confidence_copy_batches.py',
        'tests/test_confidence_parallel_capture.py','tests/test_confidence_capture_adversarial.py',
        'tests/test_confidence_batched_storage.py','tests/test_confidence_shared_scheduling.py'])
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):
        parser.error('Qualification evidence must be under sibling Scratch')
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'temporary';temporary.mkdir(exist_ok=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(ROOT),
                     TEMP=str(temporary),TMP=str(temporary),PYTHONDONTWRITEBYTECODE='1')
    library=None
    if args.library:
        library=args.library.resolve()
        if not library.is_file():parser.error('Explicit qualification library does not exist')
        environment.update(XTA_QUALIFY_NATIVE_ZLIB='1',XTA_TEST_NATIVE_ZLIB_LIBRARY=str(library),
                           YOLO_TTA_CONFIDENCE_NATIVE_CODEC='auto')
    before=fingerprints()
    with (output/'qualification-tests.txt').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-x','-p','no:cacheprovider',
            '--basetemp='+str(output/'pytest'),*args.test_path],cwd=ROOT,env=environment,
            stdout=log,stderr=subprocess.STDOUT)
    after=fingerprints()
    (output/'qualification.json').write_text(json.dumps(dict(
        scope='Functional CPU qualification; no performance or Linux-loader claim',
        pytest_exit_code=result.returncode,source_before=before,source_after=after,
        changed_during_tests=[name for name in before if before[name]!=after[name]],
        explicit_qualification_library=str(library) if library else None,
        explicit_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if library else None),
        indent=2),encoding='utf-8')
    print(f'Native confidence qualification exit={result.returncode}',flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
