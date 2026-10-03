"""CPU probe admission and ownership gates; these do not benchmark throughput."""
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from tools import benchmark_native_pull_projection as probe


def limits(**changes):
    settings=dict(max_working_mib=4096,reserve_mib=2048,plan_mib=64,
        workspace_mib=256,workers=8,four_parent_probe=True)
    return SimpleNamespace(**{**settings,**changes})


def test_probe_credit_uses_physical_headroom_and_bounds_four_parent_workers():
    profile=probe.admit_resources(limits(),10*1024**3,16)
    assert profile['effective_worker_cap']==8
    assert 4*profile['four_parent_workers']<=profile['effective_worker_cap']
    assert profile['four_parent_charge_bytes']<=profile['assigned_budget_bytes']
    with pytest.raises(RuntimeError,match='Physical/cgroup'):
        probe.admit_resources(limits(),1024**3,16)
    with pytest.raises(RuntimeError,match='four-parent'):
        probe.admit_resources(limits(workers=2),10*1024**3,16)


def test_compiled_spans_preserve_all_global_addresses_with_a_small_strip(monkeypatch):
    lock=threading.Lock();calls=[];active=0
    def fake(source,plan,output,*,first_flat,scalar_max):
        nonlocal active
        with lock:active+=1
        try:
            assert len(output)<=7
            assert not np.may_share_memory(source,output)
            output[:]=np.arange(first_flat,first_flat+len(output),dtype=np.uint8)
            with lock:calls.append((first_flat,first_flat+len(output)))
            return dict(backend='compiled_test',kernel_calls=1,contribution_addresses=len(output))
        finally:
            with lock:active-=1
    monkeypatch.setitem(sys.modules,'XTA.projection_coverage_cpu',SimpleNamespace(pull_native_flat_into=fake))
    plan=SimpleNamespace(max_strip_voxels=7,backend='compiled_test',persistent_bytes=11,temporary_strip_bytes=12)
    source=np.zeros((1,1,1),np.uint8)
    output,stats=probe.compiled_flat(source,plan,19,31,True,4)
    np.testing.assert_array_equal(output,np.arange(19,50,dtype=np.uint8))
    assert sorted(calls)==[(19,26),(26,33),(33,40),(40,47),(47,50)]
    assert stats['workers']==4 and active==0


def test_production_witness_preserves_ratio_without_materializing_logical_source():
    args=SimpleNamespace(seed=150772,size=192,slab_planes=2)
    case=probe.make_case('production_slab','tilted','sagittal','vertical','full',args)
    assert case['target']==(1931,3064,3022)
    assert case['meta']['working_shape']==[2911,3064,3022]
    assert case['source'].shape==(3064,2048,2048)
    assert case['source'].nbytes>10*1024**3
    assert case['owner'].nbytes==4*1024**2
    assert case['source'].strides[0]==0 and not case['source'].flags.writeable


def test_plan_only_never_starts_projection_or_heatsoak(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'ROOT',tmp_path/'Test')
    output=tmp_path/'Scratch'/'probe'
    args=probe.parse_args(['--output-dir',str(output),'--heatsoak-seconds','0'])
    for name in ('cpu_environment','heatsoak','compare_case'):
        monkeypatch.setattr(probe,name,lambda *a,**k:pytest.fail('Plan-only executed CPU work'))
    result=probe.run(args)
    assert result['plan_only'] and not result['success'] and result['cases']==[]
    assert (output/'benchmark.json').is_file()
    with pytest.raises(SystemExit):
        probe.parse_args(['--output-dir',str(tmp_path/'Scratch'/'invalid'),
                          '--execute','--heatsoak-seconds','1'])


def test_actual_public_projection_retires_its_mapping_before_next_call(tmp_path,monkeypatch):
    from tools.qualify_release import _qualification_environment
    # Prior suite tests can disable or close the parent's telemetry singleton.
    # The public integration must also leave every parent environment value alone.
    monkeypatch.setenv('YOLO_TTA_TELEMETRY','0')
    monkeypatch.setattr(probe,'telemetry_record',lambda:{})
    before=dict(os.environ)
    output=tmp_path/'public'
    environment=_qualification_environment(output,cpu_only=True)
    script='''
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools import benchmark_native_pull_projection as probe
args=SimpleNamespace(output_dir=Path(sys.argv[1]),plan_mib=64,workspace_mib=256,
                     seed=150772,size=16,slab_planes=2)
probe.cpu_environment(args.output_dir)
case=probe.make_case('scaled','tilted','transverse','vertical','random',args,tiny=True)
reference,unused=probe.public_volume(case,'numpy',1,args,False)
assert not (args.output_dir/'temporary-volume.dat').exists()
actual,dispatch=probe.public_volume(case,'compiled',2,args,False)
np.testing.assert_array_equal(actual,reference)
assert dispatch['backend'].startswith('compiled_'),dispatch
assert not (args.output_dir/'temporary-volume.dat').exists()
print('PUBLIC_PROJECTION_RESULT='+json.dumps(dict(equal=True,retired=True,backend=dispatch['backend'])))
'''
    completed=subprocess.run([sys.executable,'-B','-c',script,str(output)],
        cwd=probe.ROOT,env=environment,capture_output=True,text=True,timeout=120)
    (tmp_path/'public_projection.stdout.txt').write_text(completed.stdout,encoding='utf-8')
    (tmp_path/'public_projection.stderr.txt').write_text(completed.stderr,encoding='utf-8')
    assert dict(os.environ)==before
    assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
    marker=next(line for line in completed.stdout.splitlines()
                if line.startswith('PUBLIC_PROJECTION_RESULT='))
    result=json.loads(marker.split('=',1)[1])
    assert result['equal'] and result['retired']
    assert result['backend'].startswith('compiled_')
