"""Portable SAM publication capture and CPU-only real-stage replay."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import assembly, component_replay, geometry
from XTA.artifact_archive import publish_directory, reference
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, write_raw_bbox_mask_store
from XTA.view_prepare import SamLayerProjectionSubmitter
from XTA.projection_queue import ComponentProjectionQueue
from tools import replay_component_projection as tool


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT','0')
    component_replay.configure_component_replay_capture(None)
    yield
    component_replay.configure_component_replay_capture(None)


def make_source(root, view):
    data=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    data[len(data)//2,view.src_h//2,view.src_w//2]=1
    path=root/'native.cvol'
    write_raw_bbox_mask_store(data,path,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    return path,data


def metadata(entry):
    return dict(sam_projection=dict(layer_kind='interpolation',entry=entry,
        model_name='detector',source='fullframe',pass_index=1,tile_config_id='',workers=1))


def captured(root, view, *, archive=False):
    source,data=make_source(root,view)
    originals={name:(source/name).read_bytes() for name in ('meta.json','index.bin','chunks.bin')}
    if archive:
        archived=Path(reference(root/'sam-artifacts.tar','scope/forward.cvol'))
        publish_directory(source,archived)
        source=archived
    component_replay.configure_component_replay_capture(root/'capture',view_names=[view.name],
        require_persistent=False)
    entry=dict(path=str(source),direction='forward',voxel_count=int(data.sum()))
    capture=component_replay.capture_component_projection(source,view=view,
        out_shape_tyx=(view.full_t,view.full_h,view.full_w),added_voxels=int(data.sum()),
        layer_metadata=metadata(entry))
    for name,value in originals.items():
        assert (capture/'input.cvol'/name).read_bytes()==value
    return capture,data


def test_archive_capture_and_real_sam_cartesian_replay_are_exact(tmp_path):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('sagittal',))[0]
    capture,data=captured(tmp_path,view,archive=True)
    result=tool.execute_backend(capture,tmp_path/'replay','sam_cpu',1)
    expected=np.ascontiguousarray(data.transpose(1,0,2))
    assert result['decoded_output']==dict(shape=list(expected.shape),
        sha256=hashlib.sha256(expected.tobytes()).hexdigest(),foreground=1)
    assert result['effective_backend']=='sam_production_cpu'
    assert not result['gpu_used'] and not result['gpu_requested']
    assert result['input_and_listed_sources_unchanged']
    assert any(row['phase']=='cartesian_transpose' for row in result['phase_timings'])
    observed=result['resource_observations']
    assert observed['workspace_complete_samples']>0
    assert observed['peak_observed_workspace_logical_bytes']>0
    assert observed['sampler_stopped'] is True


@pytest.mark.parametrize('family',('radial','spherical','azimuthal'))
def test_nonlinear_capture_replays_exact_production_sam_stage(tmp_path,family):
    from tests.test_sam_view_orientations import VIEWS
    view=next(view for view in VIEWS if view.family==family)
    capture,_data=captured(tmp_path,view,archive=True)
    result=tool.execute_backend(capture,tmp_path/'replay','sam_cpu',1)
    saved=component_replay.load_component_replay(capture)
    direct=tmp_path/'direct'
    direct.mkdir()
    old_shape=assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(saved.output_shape)
    try:
        expected=assembly._sam_directional_source_backing(saved.source_path,view=view,
            stage='sam_selected_forward',has_foreground=True,model_name='detector',
            source='fullframe',pass_index=1,tile_config_id='',workers=1,_work_root=direct)
    finally:
        assembly.set_final_source_output_shape(old_shape)
    comparison=tool.compare_projected_stores(expected['path'],result['result_store'])
    assert comparison['exact'] and comparison['changed_voxels']==0
    assert result['input_and_listed_sources_unchanged'] and not result['gpu_used']


@pytest.mark.parametrize('layer_kind',('interpolation','extrapolation'))
def test_capture_hook_is_opt_in_and_precedes_blocked_admission(tmp_path,monkeypatch,layer_kind):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    source,data=make_source(tmp_path,view)
    entry=dict(path=str(source),direction='forward',voxel_count=1)
    entered,release=threading.Event(),threading.Event()
    phases=[]
    @contextmanager
    def phase(name,scope,**kwargs):
        phases.append(('begin',name,scope,kwargs))
        entered.set()
        try:
            yield
        finally:
            phases.append(('end',name,scope,kwargs))
    monkeypatch.setattr('XTA.sam_interpolation._trace_sam_phase',phase)
    queue=ComponentProjectionQueue(workers=1,max_pending=1,max_source_bytes=1024**2,
        max_working_bytes=8*1024**3)
    queue.submit(lambda:release.wait(5),source_bytes=1,working_bytes=1)
    submitter=SamLayerProjectionSubmitter(queue,(5,7,9),lambda *a,**k:None,lambda *a,**k:None)
    kwargs=dict(layer_kind=layer_kind,view=view,model_name='detector',source='fullframe',
        pass_index=1,workers=1,sam_context=SimpleNamespace(detector_identity='detector',
            bundle_identity='sam',source_volume=SimpleNamespace(shape=(5,7,9))))
    component_replay.configure_component_replay_capture(tmp_path/'capture',view_names=[view.name],
        require_persistent=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(submitter,entry,**kwargs)
        try:
            assert entered.wait(5)
            assert not future.done() and [row[0] for row in phases]==['begin']
            captures=list((tmp_path/'capture').glob('*/manifest.json'))
            assert len(captures)==1
            saved=component_replay.load_component_replay(captures[0])
            assert saved.layer_metadata['sam_projection']['entry']==entry
            assert saved.layer_metadata['sam_projection']['pass_index']==(1 if layer_kind=='interpolation' else 0)
        finally:
            release.set()
        future.result(5).result(5)
    assert [row[0] for row in phases]==['begin','end']
    assert phases[0][1:]==('projection_admission',f'detector/{view.name}/fullframe',{'operation':layer_kind})
    # Disabled capture performs no CVOL copy, but normal queue ownership remains.
    component_replay.configure_component_replay_capture(None)
    monkeypatch.setattr(component_replay,'open_artifact',lambda *a,**k:pytest.fail('Disabled capture read payload'))
    submitter(entry,**kwargs).result(5)
    queue.shutdown()
    assert queue.snapshot()['pending']==0


def test_worker_timeout_kills_process_and_preserves_thread_dump(tmp_path):
    log=tmp_path/'blocked.log'
    script='import faulthandler,os,time; print(os.getpid(),flush=True); faulthandler.dump_traceback_later(.1); time.sleep(20)'
    with pytest.raises(subprocess.TimeoutExpired):
        tool._run_worker([sys.executable,'-B','-c',script],log,env=dict(os.environ),timeout=1)
    text=log.read_text()
    assert 'Timeout' in text and '<string>' in text
    import psutil
    assert not psutil.pid_exists(int(text.splitlines()[0]))


def test_projection_admission_phase_reports_original_submit_failure(tmp_path,monkeypatch):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    source,_data=make_source(tmp_path,view)
    events=[]
    monkeypatch.setattr('XTA.runtime.runtime_trace_event',lambda event,**fields:events.append((event,fields)))
    error=MemoryError('queue admission refused')
    class Queue:
        def submit(self,*args,**kwargs):
            raise error
    submitter=SamLayerProjectionSubmitter(Queue(),(5,7,9),lambda:None,lambda:None)
    with pytest.raises(MemoryError) as caught:
        submitter(dict(path=str(source),direction='forward',voxel_count=1),
            layer_kind='interpolation',view=view,model_name='detector',source='fullframe',
            sam_context=SimpleNamespace(detector_identity='detector',bundle_identity='sam',source_volume=None))
    assert caught.value is error
    phases=[(event,fields) for event,fields in events if fields.get('sam_phase')=='projection_admission']
    assert [event for event,_ in phases]==['sam_phase_begin','sam_phase_end']
    assert phases[-1][1]['failed'] is True and phases[-1][1]['error_type']=='MemoryError'


def test_fresh_cli_sam_replay_and_help(tmp_path):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('sagittal',))[0]
    capture,data=captured(tmp_path,view,archive=True)
    output=tmp_path/'report'
    command=[sys.executable,'-B',str(Path(tool.__file__)),str(capture),'--sam-production',
        '--output',str(output),'--scratch',str(tmp_path/'scratch'),'--workers','1',
        '--timeout-seconds','30','--stack-dump-seconds','5']
    completed=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ),timeout=45)
    assert completed.returncode==0,completed.stdout+completed.stderr
    report=json.loads((output/'replay.json').read_text())
    assert report['status']=='complete' and report['all_exact'] is None
    assert report['runs'][0]['cuda_visible_devices']=='-1'
    assert report['runs'][0]['decoded_output']['sha256']==hashlib.sha256(
        np.ascontiguousarray(data.transpose(1,0,2)).tobytes()).hexdigest()
    assert not report['workspaces_retained']
    observation_path=Path(report['resource_observation_files']['sam_cpu'])
    assert observation_path.is_file()
    observed=json.loads(observation_path.read_text())
    assert observed['workspace_complete_samples']>0
    assert observed['peak_observed_workspace_logical_bytes']>0
    assert observed['tree_complete_samples']>0
    help_result=subprocess.run([sys.executable,'-B',str(Path(tool.__file__)),'--help'],
        capture_output=True,text=True,env=dict(os.environ),timeout=30)
    assert help_result.returncode==0 and '--sam-production' in help_result.stdout


def test_resource_samples_observe_touched_mmap_before_unlink_and_skip_links(tmp_path):
    native=tmp_path/'native.u8.dat'
    outside=tmp_path.parent/(tmp_path.name+'-outside.dat')
    outside.write_bytes(b'outside must not be scanned')
    observer=tool._ReplayResourceSamples(tmp_path,None)
    size=4*1024**2
    try:
        try:
            (tmp_path/'alias').symlink_to(outside)
        except OSError:
            pass  # Windows may not grant symlink creation to this account.
        with native.open('w+b') as stream:
            stream.truncate(size)
            with mmap.mmap(stream.fileno(),size) as mapping:
                for offset in range(0,size,4096):
                    mapping[offset]=1
                with observer.phase('native_decode'):
                    observer.sample(force=True)
        native.unlink()
        observer.sample(force=True)
        measured=observer.report()
        assert measured['peak_observed_workspace_logical_bytes']==size
        assert measured['peak_observed_workspace_files']==1
        assert measured['workspace_logical_peak_context']['active_phases']==['native_decode']
        assert measured['peak_observed_tree_rss_bytes'] is None
        if hasattr(tmp_path.stat(),'st_blocks'):
            assert measured['peak_observed_workspace_allocated_bytes']>=size
        else:
            assert measured['peak_observed_workspace_allocated_bytes'] is None
        persisted=json.loads((tmp_path/'resource_observations.json').read_text())
        assert persisted['peak_observed_workspace_logical_bytes']==size
    finally:
        outside.unlink(missing_ok=True)


def test_resource_samples_tree_rss_and_unavailable_measurements_are_explicit(tmp_path,monkeypatch):
    class Process:
        def __init__(self,pid,rss):self.pid,self.rss,self.children_list=pid,rss,[]
        def children(self,recursive):return self.children_list
        def memory_info(self):return SimpleNamespace(rss=self.rss)
    parent,child=Process(1,10),Process(2,20)
    parent.children_list=[child]
    observer=tool._ReplayResourceSamples(tmp_path,parent)
    observer.sample(force=True)
    child.rss=40
    with observer.phase('source_projection'):
        observer.sample(force=True)
    measured=observer.report()
    assert measured['tree_first_observed_rss_bytes']==30
    assert measured['peak_observed_tree_rss_bytes']==50
    assert measured['peak_observed_tree_processes']==2
    assert measured['tree_peak_context']['active_phases']==['source_projection']
    def unavailable(*args,**kwargs):raise PermissionError('measurement unavailable')
    monkeypatch.setattr(parent,'children',unavailable)
    monkeypatch.setattr(tool,'_workspace_usage',unavailable)
    unknown=tool._ReplayResourceSamples(tmp_path,parent)
    unknown.sample(force=True)
    data=unknown.report()
    assert data['peak_observed_tree_rss_bytes'] is None
    assert data['peak_observed_workspace_logical_bytes'] is None
    assert data['peak_observed_workspace_allocated_bytes'] is None
    assert data['tree_incomplete_samples']==data['workspace_incomplete_samples']==1


def test_resource_samples_include_actual_child_process(tmp_path):
    import psutil
    child=subprocess.Popen([sys.executable,'-B','-c',
        'import time; payload=bytearray(4*1024**2); print("ready",flush=True); time.sleep(5)'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=dict(os.environ))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(child.stdout.readline).result(3).strip()=='ready'
        observer=tool._ReplayResourceSamples(tmp_path,psutil.Process())
        with observer.phase('source_projection'):
            observer.sample(force=True)
        result=observer.report()
        assert result['peak_observed_tree_processes']>=2
        assert result['peak_observed_tree_rss_bytes']>0
        assert result['tree_peak_context']['active_phases']==['source_projection']
    finally:
        child.kill()
        child.communicate(timeout=3)


def test_phase_boundaries_only_signal_and_workspace_scans_are_capped(tmp_path,monkeypatch):
    observer=tool._ReplayResourceSamples(tmp_path,None)
    monkeypatch.setattr(observer,'sample',lambda **kwargs:pytest.fail('Phase hook performed I/O'))
    with observer.phase('native_retirement'):
        assert observer.requested.is_set()
        assert observer.phases=={'native_retirement':1}
    assert not observer.phases
    assert observer.report()['last_boundary']['boundary']=='end'
    (tmp_path/'a').write_bytes(b'first')
    (tmp_path/'b').write_bytes(b'second')
    assert tool._workspace_usage(tmp_path,max_entries=1)['complete'] is False
    assert tool._workspace_usage(tmp_path,time_budget_seconds=0)['complete'] is False


def test_worker_timeout_survives_diagnostic_copy_failure_and_saves_before_cleanup(tmp_path,monkeypatch):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    capture,_data=captured(tmp_path,view)
    output=tmp_path/'report'
    timeout=subprocess.TimeoutExpired(['worker'],.1)
    def worker(command,*args,**kwargs):
        workspace=Path(command[command.index('--output')+1])
        workspace.mkdir()
        (workspace/'resource_observations.json').write_text('{}')
        raise timeout
    def copy(*args,**kwargs):raise PermissionError('diagnostic copy refused')
    original_cleanup=tool._remove_owned_scratch
    def cleanup(root,parent):
        assert json.loads((output/'replay.json').read_text())['status']=='timed_out'
        original_cleanup(root,parent)
    monkeypatch.setattr(tool,'_run_worker',worker)
    monkeypatch.setattr(tool,'_copy_diagnostic_file',copy)
    monkeypatch.setattr(tool,'_remove_owned_scratch',cleanup)
    monkeypatch.setattr(sys,'argv',[str(Path(tool.__file__)),str(capture),'--sam-production',
        '--output',str(output),'--scratch',str(tmp_path/'scratch')])
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        tool.main()
    assert caught.value is timeout
    report=json.loads((output/'replay.json').read_text())
    assert report['status']=='timed_out' and not report['workspaces_retained']
    assert report['persistence_errors'][0]['file']=='resource_observations'


def test_diagnostic_copy_is_byte_bounded(tmp_path):
    source,destination=tmp_path/'source.json',tmp_path/'copy.json'
    source.write_bytes(b'0123456789')
    with pytest.raises(ValueError,match='byte limit'):
        tool._copy_diagnostic_file(source,destination,max_bytes=8)
    assert not destination.exists()


def test_blocked_observer_cannot_hold_replay_shutdown(tmp_path,monkeypatch):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    capture,_data=captured(tmp_path,view)
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    def blocked(self,**kwargs):
        entered.set()
        try:
            release.wait(5)
        finally:
            finished.set()
    monkeypatch.setattr(tool._ReplayResourceSamples,'sample',blocked)
    try:
        result=tool.execute_backend(capture,tmp_path/'work','sam_cpu',1)
        assert entered.is_set() and not release.is_set()
        assert result['resource_observations']['sampler_stopped'] is False
        assert result['resource_observations']['peak_observed_workspace_logical_bytes'] is None
        assert result['decoded_output']['foreground']==1
    finally:
        release.set()
        assert finished.wait(3)


def test_measurement_access_failure_does_not_mask_projection_failure(tmp_path,monkeypatch):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    capture,_data=captured(tmp_path,view)
    import psutil
    class Unavailable:
        def memory_info(self):raise PermissionError('RSS denied')
        def children(self,recursive):raise PermissionError('tree denied')
    monkeypatch.setattr(psutil,'Process',Unavailable)
    failure=RuntimeError('original projection failure')
    def project(*args,**kwargs):raise failure
    monkeypatch.setattr(assembly,'_sam_directional_source_backing',project)
    with pytest.raises(RuntimeError) as caught:
        tool.execute_backend(capture,tmp_path/'work','sam_cpu',1)
    assert caught.value is failure


def test_failure_report_io_cannot_replace_worker_timeout(tmp_path,monkeypatch):
    view=geometry.get_view_infos(5,7,9,cartesian_views=('transverse',))[0]
    capture,_data=captured(tmp_path,view)
    timeout=subprocess.TimeoutExpired(['worker'],.1)
    worker_started=threading.Event()
    def worker(*args,**kwargs):
        worker_started.set()
        raise timeout
    original_write=Path.write_text
    def write(path,*args,**kwargs):
        if path.name=='replay.json' and worker_started.is_set():
            raise PermissionError('report write denied')
        return original_write(path,*args,**kwargs)
    monkeypatch.setattr(tool,'_run_worker',worker)
    monkeypatch.setattr(Path,'write_text',write)
    monkeypatch.setattr(sys,'argv',[str(Path(tool.__file__)),str(capture),'--sam-production',
        '--output',str(tmp_path/'report'),'--scratch',str(tmp_path/'scratch')])
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        tool.main()
    assert caught.value is timeout
    assert any('report' in note for note in timeout.__notes__)


def test_late_probe_is_discarded_after_sampling_scope_closes(tmp_path,monkeypatch):
    observer=tool._ReplayResourceSamples(tmp_path,None)
    entered,release=threading.Event(),threading.Event()
    def workspace(root):
        entered.set()
        assert release.wait(3)
        return dict(complete=True,logical_bytes=123,allocated_bytes=123,files=1)
    monkeypatch.setattr(tool,'_workspace_usage',workspace)
    worker=threading.Thread(target=lambda:observer.sample(force=True))
    try:
        worker.start()
        assert entered.wait(3)
        with observer.lock:observer.data['sampler_stopped']=False
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        assert observer.report()['peak_observed_workspace_logical_bytes'] is None
        assert observer.report()['workspace_complete_samples']==0
        assert not (tmp_path/'resource_observations.json').exists()
    finally:
        release.set()
        worker.join(3)
