"""CPU subprocess integration for confidence-family ownership and publication.

The child fixture implements the normal scalar receipt contract without model
execution. These tests exercise real child processes and parent lifecycle;
they do not claim detector/SAM or hardware equivalence.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import pytest

from XTA import confidence_sweep as sweep


_POPEN = subprocess.Popen
_CHILD = r'''
import hashlib, json, os, signal, subprocess
from pathlib import Path
import sys
import time
from XTA.unification.tta_manifest import capture_tta_artifact_identities, TTA_RUN_MANIFEST_SCHEMA
if hasattr(signal,'SIGBREAK'):
    signal.signal(signal.SIGBREAK,signal.default_int_handler)

arguments = json.loads(sys.argv[1])
def scalar(name, default=None):
    result = default
    for index, token in enumerate(arguments):
        if token == name:
            result = arguments[index + 1]
        elif token.startswith(name + '='):
            result = token.split('=', 1)[1]
    return result

cent = int(os.environ['XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'])
out = Path(scalar('--output'))
out.mkdir(parents=True, exist_ok=True)
cache = Path(os.environ['XTA_YOLO_SWEEP_CACHE_DIR'])
if float(os.environ['XTA_YOLO_SWEEP_CACHE_GIB'])>0:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ('child-' + str(cent))).write_text('forward fixture')
record = {'arguments':arguments, 'cwd':str(Path.cwd()), 'pid':os.getpid(),
          'environment':{name:os.environ.get(name) for name in (
              'XTA_YOLO_SWEEP_CACHE_DIR','XTA_YOLO_SWEEP_CACHE_GIB',
              'XTA_YOLO_SWEEP_WORK_DIR','XTA_YOLO_SWEEP_ID',
              'XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS','XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS',
              'XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS','XTA_YOLO_SWEEP_NATIVE_ROLE',
              'XTA_YOLO_SWEEP_OUTPUT_ROOT','XTA_YOLO_SWEEP_QUOTA_DATABASE',
              'XTA_YOLO_SWEEP_SAM_ARCHIVE_ROOT',
              'PYTHONPATH','PYTHONSAFEPATH')}}
(out / 'child_observed.json').write_text(json.dumps(record))
print('scalar-live-' + str(cent), flush=True)
print('scalar-stderr-' + str(cent), file=sys.stderr, flush=True)
mode = os.environ.get('SWEEP_TEST_MODE', '') if cent == 75 else ''
if mode == 'exit':
    raise SystemExit(7)
if mode == 'abandoned_sam_archive':
    from XTA.artifact_archive import StagedArtifactArchive, append_members
    public = out / 'sam-artifacts.tar'
    pending = StagedArtifactArchive.reserve(public, os.environ['XTA_YOLO_SWEEP_SAM_ARCHIVE_ROOT'])
    append_members(public, {'evidence/receipt.json': b'recoverable'})
    raise SystemExit(7)
if mode == 'missing':
    raise SystemExit(0)
if mode == 'sleep':
    time.sleep(60)
if mode in {'orphan','tree_cancel'}:
    cache.mkdir(parents=True,exist_ok=True)
    code = "import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
    code += "signal.signal(signal.SIGBREAK,signal.SIG_IGN); " if hasattr(signal,'SIGBREAK') else ''
    code += "p=open(sys.argv[1],'a'); [(p.write('active\\n'),p.flush(),time.sleep(.05)) for _ in range(1200)]"
    grandchild = subprocess.Popen([sys.executable,'-u','-c',code,str(cache/'grandchild.tick')])
    (out/'grandchild.pid').write_text(str(grandchild.pid))
    if mode == 'tree_cancel':
        time.sleep(60)
model = Path(scalar('--model').split(':',1)[1])
if mode == 'identity':
    model.write_bytes(b'changed')
inputs = capture_tta_artifact_identities(input_path=scalar('--input'),
    gpu_model_path=model, cpu_model_path=None)
inputs.update(source_shape_t_y_x=[2,3,4],processing_shape_t_y_x=[2,3,4],fps=1.)
identity = {'sweep_id':os.environ['XTA_YOLO_SWEEP_ID'], 'cutoff_hundredths':cent,
    'sam_checkpoint_identity_sha256':None, 'augmentation_policy_sha256':None,
    'reconciliation_policy_sha256':None}
role = os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE')
if role is not None:
    targets = [int(value) for value in os.environ['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'].split(',')]
    identity['upstream_fanout'] = {'role':role,'validated':True,
        'produced_views':0,'consumed_views':0,'scalar_views':1,
        'target_count':len(targets),'fallback_reasons':['fixture upstream route is unsupported']}
    fanout_mode = os.environ.get('SWEEP_TEST_FANOUT','unsupported')
    evidence = identity['upstream_fanout']
    if fanout_mode in {'native','corrupt','partial'}:
        family = Path(os.environ['XTA_YOLO_SWEEP_OUTPUT_ROOT'])
        native = Path(os.environ['XTA_YOLO_SWEEP_WORK_DIR'])/'native_fanout'
        native.mkdir(parents=True,exist_ok=True)
        views = ('transverse',) if fanout_mode=='partial' else ('transverse','sagittal')
        def event(stage,view=None,target=None):
            with (family/'fixture_work_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'stage':stage,'arm':cent,'view':view,'target':target})+'\n')
        if role == 'produce':
            for view in views:
                event('render',view)
                event('forward',view)
                for target in targets:
                    mask = [int(.9 >= target/100),int(.6 >= target/100)]
                    payload = {'cutoff_hundredths':target,'view':view,'mask':mask,
                        'sha256':hashlib.sha256(json.dumps(mask).encode()).hexdigest()}
                    if fanout_mode=='corrupt' and target==75:
                        payload['sha256'] = '0'*64
                    (native/f'{view}-{target:03d}.json').write_text(json.dumps(payload))
                    sibling = family/'confidence_cutoffs'/f'conf_{target:03d}'
                    sibling.mkdir(parents=True,exist_ok=True)
                    (sibling/'producer_precreated.json').write_text(json.dumps({'targets':targets}))
                    event('native_product',view,target)
            (family/'fanout_storage.json').write_text(json.dumps({'fixture':True,'payload_files':len(views)*len(targets)}))
        native_inputs = []
        for view in views:
            payload = json.loads((native/f'{view}-{cent:03d}.json').read_text())
            assert payload['view']==view and payload['cutoff_hundredths']==cent
            assert payload['sha256']==hashlib.sha256(json.dumps(payload['mask']).encode()).hexdigest()
            native_inputs.append(payload['mask'])
            if role=='consume':
                event('native_consume',view,cent)
        # Each arm independently enters its suffix using copied native values.
        event('independent_suffix')
        (out/'native_inputs.json').write_text(json.dumps(native_inputs))
        (out/'suffix_state.json').write_text(json.dumps({'cent':cent,'pid':os.getpid(),
            'native_inputs':native_inputs,'independent_result':[sum(mask)+cent for mask in native_inputs]}))
        evidence.update(produced_views=len(views) if role=='produce' else 0,
            consumed_views=len(views) if role=='consume' else 0,
            scalar_views=1 if fanout_mode=='partial' else 0,
            fallback_reasons=['sagittal fixture route unsupported'] if fanout_mode=='partial' else [])
    if fanout_mode=='unvalidated':
        evidence['validated'] = False
    elif fanout_mode=='negative_count':
        evidence['produced_views'] = -1
    elif fanout_mode=='boolean_count':
        evidence['scalar_views'] = True
    elif fanout_mode=='wrong_target_count':
        evidence['target_count'] += 1
    elif fanout_mode=='wrong_role':
        evidence['role'] = 'consume' if role=='produce' else 'produce'
    elif fanout_mode=='silent_fallback':
        evidence['fallback_reasons'] = []
    elif fanout_mode=='missing_receipt':
        del identity['upstream_fanout']
    elif fanout_mode=='unproduced_consumer' and role=='consume':
        evidence.update(consumed_views=1,scalar_views=0,fallback_reasons=[])
if mode == 'ownership':
    identity['sweep_id'] = 'foreign'
if mode == 'policy':
    identity['reconciliation_policy_sha256'] = 'b'*64
if mode == 'missing_identity':
    del inputs['source']['file_id']
version = os.environ['SWEEP_TEST_VERSION']
omitted_sam = mode in {'incomplete', 'strict_omission'}
receipt = {'schema':TTA_RUN_MANIFEST_SCHEMA, 'status':'incomplete' if omitted_sam else 'complete',
    'launcher':{'mode':'tta','version':version,'pipeline_version':version},
    'confidence_sweep':identity, 'resolved_configuration':{
        'conf':float(scalar('--conf')),'min_conf':float(scalar('--min_conf')),
        'input':scalar('--input'),'model':scalar('--model'),
        'output':str(out),'temp':scalar('--temp'),
        'capture_component_replay':scalar('--capture_component_replay'),
        'task':scalar('--task','segment'),'interpolation':scalar('--interpolation','sdf'),
        'interpolation_passes':2},
    'inputs':inputs,'geometry':{'physical_views':[]},
    'sam_requested_work_complete':not omitted_sam,
    'sam_planning_omissions':[{'scope':'fixture'}] if omitted_sam else [],
    'outputs':{'requested':[],'paths':{'run_manifest':str(out/'manifest.json')}}}
if mode == 'inconsistent_incomplete':
    receipt['status'] = 'incomplete'  # positive completion flag and no omissions
if mode == 'inconsistent_complete':
    receipt['sam_requested_work_complete'] = False
    receipt['sam_planning_omissions'] = [{'scope':'fixture'}]
if mode == 'unrecorded_omission':
    receipt['status'] = 'incomplete'
    receipt['sam_requested_work_complete'] = False
if mode == 'in_progress':
    receipt['status'] = 'in_progress'
if mode == 'conf':
    receipt['resolved_configuration']['conf'] = .123
if summary_section := os.environ.get('XTA_YOLO_SWEEP_SUMMARY_SECTION'):
    summary_destination = out.parent.parent / 'summary.txt'
    receipt['outputs']['paths']['summary'] = str(summary_destination)
    if mode != 'missing_summary':
        summary_section = Path(summary_section)
        summary_section.parent.mkdir(parents=True, exist_ok=True)
        summary_section.write_text(
            f'View statistics:\n  Total prediction count: {cent}\n'
            f'Final outputs:\nrun_manifest: {out / "manifest.json"}\n',
            encoding='utf-8',
        )
    if mode == 'late_summary_conflict':
        summary_destination.write_text('foreign summary', encoding='utf-8')
if media_root := os.environ.get('XTA_YOLO_SWEEP_MEDIA_ROOT'):
    stage = Path(media_root) / f'conf_{cent:03d}' / 'native.packbits'
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_bytes(bytes((1, 0, 1, 0)))
    identity['heatmap_inputs'] = {'grids': {'native': [2, 3, 4]}, 'streaming_decode': False}
if share_geometry := os.environ.get('SWEEP_TEST_SHARE_GEOMETRY'):
    from XTA.unification.tta_manifest import SWEEP_SHARED_GEOMETRY_ENV, compact_sweep_run_manifest
    receipt['forward_sampling'] = {
        'renderer': 'different' if cent == 75 and share_geometry == 'sampling' else 'reference',
    }
    if cent == 75 and share_geometry == 'geometry':
        receipt['geometry']['physical_views'].append('unexpected')
    compact_sweep_run_manifest(out / 'manifest.json', receipt, json.loads(os.environ[SWEEP_SHARED_GEOMETRY_ENV]))
(out / 'manifest.json').write_text(json.dumps(receipt))
if mode == 'strict_omission':
    raise SystemExit(1)
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    child = tmp_path / 'scalar_fixture.py'
    child.write_text(_CHILD, encoding='utf-8')
    source = tmp_path / 'source video.mkv'
    source.write_bytes(b'video fixture')
    model = tmp_path / 'model artifact.engine'
    model.write_bytes(b'model fixture')
    monkeypatch.setenv('SWEEP_TEST_VERSION', sweep.__version__)
    # These cases pin the sequential consumer contract (inherited output, order,
    # stop-after-failure). tests/test_confidence_sweep_concurrency.py covers the
    # concurrent consumer path.
    monkeypatch.setenv('XTA_YOLO_SWEEP_CONSUMER_PARALLEL', '1')
    monkeypatch.delenv('SWEEP_TEST_MODE', raising=False)
    monkeypatch.delenv('SWEEP_TEST_FANOUT', raising=False)
    monkeypatch.delenv('SWEEP_TEST_SHARE_GEOMETRY', raising=False)
    monkeypatch.delenv('XTA_YOLO_SWEEP_VIEW_MAJOR', raising=False)
    monkeypatch.delenv('XTA_YOLO_SWEEP_QUOTA_DIR', raising=False)
    monkeypatch.delenv('XTA_YOLO_SWEEP_CACHE_GIB', raising=False)
    system_temp = tmp_path/'system-temp'
    system_temp.mkdir()
    monkeypatch.setattr(sweep.tempfile,'gettempdir',lambda:str(system_temp))
    children = []

    def launch(command, **options):
        assert command[:5] == [sys.executable, '-m', 'XTA', '--mode', 'tta']
        assert 'cwd' not in options
        assert 'stdout' not in options and 'stderr' not in options
        process = _POPEN([sys.executable, '-B', str(child), json.dumps(command[5:])], **options)
        children.append(process)
        return process

    monkeypatch.setattr(sweep.subprocess, 'Popen', launch)
    publication_calls = []
    publisher = types.ModuleType('XTA.sweep_publication')

    def publish(root, rows, *, sweep_id, temp_dir):
        assert all((out/'manifest.json').is_file() for _,out in rows)
        assert temp_dir == Path(_read(root / 'manifest.json')['work_directory']).parent
        assert temp_dir.is_dir()
        publication_calls.append((root, rows, sweep_id))
        return None

    publisher.publish_sweep_review = publish
    cache_module = types.ModuleType('XTA.forward_replay')
    cache_module.replay_cache_summary = lambda root: {
        'entries':len(list(Path(root).glob('child-*'))),
        'raw_disk_bytes':sum(path.stat().st_size for path in Path(root).glob('child-*')),
        'by_backend':{'fixture':len(list(Path(root).glob('child-*')))}}
    # A staged-output test temporarily imports the real publisher. Restore its
    # package attribute as well as sys.modules when the harness is torn down.
    monkeypatch.setattr(sys.modules['XTA'], 'sweep_publication', publisher, raising=False)
    monkeypatch.setitem(sys.modules, 'XTA.sweep_publication', publisher)
    monkeypatch.setitem(sys.modules, 'XTA.forward_replay', cache_module)
    return types.SimpleNamespace(root=tmp_path, source=source, model=model,
        children=children, publications=publication_calls, publisher=publisher,
        arguments=['--input',str(source),'--model',f'gpu:{model}'])


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def test_output_mount_check_fails_before_first_cutoff(harness, monkeypatch):
    from XTA import json_publication
    output = harness.root / 'unsupported-output'

    def unsupported(stage, destination):
        assert stage.is_relative_to(output) and destination.is_relative_to(output)
        assert (stage / 'file').is_file()
        raise OSError('fixture directory rename unsupported')

    monkeypatch.setattr(json_publication, '_publish_sweep_directory', unsupported)
    with pytest.raises(RuntimeError, match='publication check failed'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75), output_root=output,
                                   save_options=('nrrd',))
    receipt = _read(output / 'manifest.json')
    assert not harness.children and receipt['branches'] == []
    assert receipt['status'] == 'failed' and receipt['work_cleanup_complete']
    assert not list(output.glob('.xta-publication-check-*'))
    assert not (output / '.confidence_sweep.lock').exists()


def test_publication_timing_identifies_completed_output_stage(tmp_path):
    from XTA.json_publication import SweepDirectoryPublicationError
    destination = tmp_path / 'review'
    staged = tmp_path / 'private' / 'review'
    manifest = {}
    with pytest.raises(SweepDirectoryPublicationError):
        with sweep._time_sweep_publication(manifest, 'review'):
            raise SweepDirectoryPublicationError(destination, staged, OSError('commit failed'))
    assert manifest['completed_publication_stages'] == {'review': 'private/review'}
    assert manifest['publication_seconds']['review'] >= 0


def test_heatmap_failure_retains_accepted_media_only_cutoffs(harness, monkeypatch):
    from XTA import sweep_heatmaps

    output = harness.root / 'media-only'

    def fail_publication(root, work, inputs, grids, cutoffs, incomplete, **options):
        assert root == output
        assert cutoffs == (50, 75)
        assert grids == {'native': [2, 3, 4]}
        assert not incomplete
        for cent in cutoffs:
            assert (work / 'heatmap-input' / f'conf_{cent:03d}' / 'native.packbits').is_file()
        raise OSError('fixture encoder failed')

    monkeypatch.setattr(sweep_heatmaps, 'publish_sweep_heatmaps', fail_publication)
    with pytest.raises(OSError, match='fixture encoder failed'):
        sweep.run_confidence_sweep(
            harness.arguments, cutoffs_hundredths=(50, 75), output_root=output,
            save_options=('high_quality',),
        )
    receipt = _read(output / 'manifest.json')
    assert receipt['status'] == 'failed'
    assert receipt['publication_seconds']['heatmaps'] >= 0
    assert receipt['heatmap_recovery']['validated_cutoffs_hundredths'] == [50, 75]
    assert Path(receipt['heatmap_recovery']['path']).is_dir()
    assert receipt['work_cleanup_complete'] is False
    assert not (output / 'heatmaps').exists()


def test_heatmaps_namespace_is_reserved_when_requested(harness):
    output = harness.root / 'media-namespace'
    directory = output / 'heatmaps'
    directory.mkdir(parents=True)
    marker = directory / 'untouched'
    marker.write_text('external', encoding='utf-8')
    with pytest.raises(FileExistsError, match='heatmaps'):
        sweep.run_confidence_sweep(
            harness.arguments, cutoffs_hundredths=(50, 75), output_root=output,
            save_options=('high_quality',),
        )
    assert not harness.children
    sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75), output_root=output)
    assert marker.read_text(encoding='utf-8') == 'external'


@pytest.mark.parametrize('mode', ('complete', 'incomplete', 'exit', 'missing_summary', 'late_summary_conflict'))
def test_sweep_summary_uses_one_public_file_and_keeps_validated_sections(harness, monkeypatch, mode):
    monkeypatch.setenv('SWEEP_TEST_MODE', mode)
    output = harness.root / 'summary-family'

    def run():
        return sweep.run_confidence_sweep([*harness.arguments, '--save', 'summary'],
            cutoffs_hundredths=(50, 100, 75), output_root=output)

    if mode == 'exit':
        with pytest.raises(subprocess.CalledProcessError):
            run()
    elif mode == 'missing_summary':
        with pytest.raises(ValueError, match='did not finish its sweep summary section'):
            run()
    elif mode == 'late_summary_conflict':
        with pytest.raises((FileExistsError, ValueError), match='appeared|exists'):
            run()
    else:
        run()

    summary = output / 'summary.txt'
    manifest = _read(output / 'manifest.json')
    if mode == 'late_summary_conflict':
        assert summary.read_text(encoding='utf-8') == 'foreign summary'
        assert 'summary' not in manifest
        assert not manifest['work_cleanup_complete']
        assert (Path(manifest['work_directory']) / 'summary' / 'conf_050.txt').is_file()
        return

    assert manifest['summary'] == 'summary.txt'
    cutoffs = (50, 100, 75) if mode in ('complete', 'incomplete') else (50, 100)
    sections = []
    for cent in cutoffs:
        branch_manifest = output / 'confidence_cutoffs' / f'conf_{cent:03d}' / 'manifest.json'
        assert Path(_read(branch_manifest)['outputs']['paths']['summary']) == summary
        sections.append(
            f'=== Confidence cutoff {cent / 100:.2f} ===\n'
            f'View statistics:\n  Total prediction count: {cent}\n'
            f'Final outputs:\nrun_manifest: {branch_manifest}\n'
        )
    assert summary.read_text(encoding='utf-8') == '\n'.join(sections)
    assert list(output.rglob('*.txt')) == [summary]
    assert manifest['work_cleanup_complete']


@pytest.mark.parametrize('different', ['none', 'sampling', 'geometry'])
def test_shared_geometry_is_written_once_and_authenticated(harness, monkeypatch, different):
    from XTA.unification.tta_manifest import load_tta_run_manifest

    monkeypatch.setenv('SWEEP_TEST_SHARE_GEOMETRY', different)
    output = harness.root / 'shared-geometry'
    arguments = dict(cutoffs_hundredths=(50, 75), output_root=output)
    if different == 'geometry':
        with pytest.raises(subprocess.CalledProcessError):
            sweep.run_confidence_sweep(harness.arguments, **arguments)
    else:
        sweep.run_confidence_sweep(harness.arguments, **arguments)
    family = _read(output / 'manifest.json')
    descriptor = family['shared_run_geometry']
    shared = output / descriptor['path']
    assert [p for p in output.rglob(shared.name)] == [shared]
    assert hashlib.sha256(shared.read_bytes()).hexdigest() == descriptor['sha256']
    receipt_path = output / 'confidence_cutoffs/conf_050/manifest.json'
    raw = _read(receipt_path)
    assert raw['shared_run_geometry'] == descriptor
    assert 'geometry' not in raw and 'forward_sampling' not in raw
    restored = load_tta_run_manifest(receipt_path)
    assert restored['geometry'] == {'physical_views': []}
    assert restored['forward_sampling'] == {'renderer': 'reference'}
    if different == 'geometry':
        assert family['status'] == 'failed'
        assert not (output / 'confidence_cutoffs/conf_075/manifest.json').exists()
        return
    sibling = output / 'confidence_cutoffs/conf_075/manifest.json'
    sibling_raw = _read(sibling)
    assert sibling_raw['shared_run_geometry'] == descriptor
    assert 'geometry' not in sibling_raw
    sibling_sampling = sibling_raw.get('forward_sampling')
    assert sibling_sampling == ({'renderer': 'different'} if different == 'sampling' else None)
    assert load_tta_run_manifest(sibling)['geometry'] == restored['geometry']
    assert load_tta_run_manifest(sibling)['forward_sampling'] == (sibling_sampling or restored['forward_sampling'])
    # The raw branch receipt continues to authenticate the shared file.
    shared.write_bytes(shared.read_bytes().replace(b'reference', b'changed!!'))
    with pytest.raises(ValueError, match='Shared sweep geometry SHA-256'):
        load_tta_run_manifest(sibling)


@pytest.mark.parametrize('child_fails', [False, True])
def test_separate_quota_storage_is_shared_and_retired(harness, monkeypatch, child_fails):
    quota_parent = harness.root / 'fast-metadata'
    quota_parent.mkdir()
    keep = quota_parent / 'unrelated'
    keep.write_text('preserve')
    monkeypatch.setenv('XTA_YOLO_SWEEP_QUOTA_DIR', str(quota_parent))
    monkeypatch.setenv('XTA_YOLO_SWEEP_QUOTA_DATABASE', str(keep))
    output = harness.root / 'family'
    if child_fails:
        monkeypatch.setenv('SWEEP_TEST_MODE', 'exit')
        with pytest.raises(subprocess.CalledProcessError) as failure:
            sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75), output_root=output)
        assert failure.value.returncode == 7
    else:
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75), output_root=output)
    manifest = _read(output / 'manifest.json')
    placement = manifest['work_storage']['native_quota']
    for cent in (50, 75):
        record = _read(output / f'confidence_cutoffs/conf_{cent:03d}/child_observed.json')
        assert Path(record['environment']['XTA_YOLO_SWEEP_QUOTA_DATABASE']) == Path(placement['directory']) / 'quota.sqlite'
    assert Path(placement['directory']).parent == quota_parent
    assert manifest['work_cleanup_complete']
    assert list(quota_parent.iterdir()) == [keep]


@pytest.mark.parametrize('failure', [None, 'branch', 'publication'])
def test_staged_nrrds_follow_group_publication_and_recovery(harness, monkeypatch, failure):
    import importlib

    import numpy as np

    # Let native dependencies use platform probes during their first import.
    with monkeypatch.context() as platform_imports:
        platform_imports.setattr(subprocess, 'Popen', _POPEN)
        importlib.import_module('XTA.runtime')
        from XTA.reconciliation_io import write_seg_nrrd

    monkeypatch.delitem(sys.modules, 'XTA.sweep_publication')
    publication = importlib.import_module('XTA.sweep_publication')
    if failure == 'branch':
        monkeypatch.setenv('SWEEP_TEST_MODE', 'exit')
    if failure == 'publication':
        def failed_group(*args, **kwargs):
            raise OSError('grouped output write failed')
        monkeypatch.setattr(publication, 'write_grouped_seg_nrrd', failed_group)
    run_child = sweep._run_child

    def with_staged_layer(command, env, *, started):
        returncode = run_child(command, env, started=started)
        if returncode == 0:
            cutoff = int(env['XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'])
            branch = Path(command[command.index('--output') + 1])
            run_path = branch / 'manifest.json'
            receipt = _read(run_path)
            receipt['confidence_sweep']['nrrd_storage'] = 'grouped'
            run_path.write_text(json.dumps(receipt), encoding='utf-8')
            stage = Path(env['XTA_YOLO_SWEEP_NRRD_STAGE_ROOT']) / f'conf_{cutoff:03d}' / 'nrrd'
            stage.mkdir(parents=True)
            shape = tuple(receipt['inputs']['source_shape_t_y_x'])
            mask = np.ones(shape, dtype=np.uint8)
            write_seg_nrrd(stage / 'volume.seg.nrrd', shape_tyx=shape,
                           read_slab=lambda z0, z1: mask[z0:z1])
            catalog = {
                'layout': 'one_single_layer_nrrd_per_component', 'output_shape_tyx': list(shape),
                'exported_axes': '(X, Y, t)',
                'layers': [{'filename': 'volume.seg.nrrd', 'segment_name': 'fixture',
                            'layer_key': 'fixture', 'model_name': 'fixture', 'source': 'global',
                            'stage': 'final_output_after_all_postprocessing'}],
            }
            if 'run_id' in receipt:
                catalog['run_id'] = receipt['run_id']
            (stage / 'volume_nrrd_manifest.json').write_text(json.dumps(catalog), encoding='utf-8')
        return returncode

    monkeypatch.setattr(sweep, '_run_child', with_staged_layer)
    output = harness.root / 'grouped'
    args = dict(cutoffs_hundredths=(50, 75), output_root=output)
    if failure == 'branch':
        with pytest.raises(subprocess.CalledProcessError):
            sweep.run_confidence_sweep(harness.arguments, **args)
    elif failure == 'publication':
        with pytest.raises(OSError, match='grouped output write failed'):
            sweep.run_confidence_sweep(harness.arguments, **args)
    else:
        sweep.run_confidence_sweep(harness.arguments, **args)
    family = _read(output / 'manifest.json')
    assert not list((output / 'confidence_cutoffs').rglob('*.seg.nrrd'))
    if failure == 'publication':
        assert family['review_manifest'] is None and not family['work_cleanup_complete']
        assert family['nrrd_recovery']['validated_cutoffs_hundredths'] == [50, 75]
        assert Path(family['nrrd_recovery']['path']).is_dir()
    else:
        assert family['work_cleanup_complete'] and 'nrrd_recovery' not in family
        path = output / family['review_manifest']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == family['review_manifest_sha256']
        review = _read(path)
        assert review['cutoffs_cent'] == ([50] if failure == 'branch' else [50, 75])
        assert review['sweep_status'] == ('failed' if failure == 'branch' else 'complete')
        assert all(grid['payload'] == 'grouped' for grid in review['grids'])


def test_scalar_processes_are_independent_and_live_output_is_inherited(harness, tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB', '0')
    original = [*harness.arguments, '--task','semantic','--interpolation','sam',
        '--conf=.10:.10:1.00','--conf','.20', '--output=ignored', '--output','ignored-too',
        '--min_conf=.9','--temp=old-root',
        '--capture_component_replay',str(tmp_path/'capture')]
    untouched = list(original)
    out = tmp_path/'family'
    path = sweep.run_confidence_sweep(original, cutoffs_hundredths=(50,75,100),
        output_root=out, temp_root=tmp_path/'work-root', min_conf=.13)
    receipt = _read(path)
    assert original == untouched
    assert receipt['status'] == 'complete'
    assert receipt['fixed_min_conf'] == .13
    assert receipt['forward_cache']['budget_gib'] == 0
    assert receipt['forward_cache']['summary']['entries'] == 0
    assert receipt['work_cleanup_complete'] is True
    assert [row['confidence'] for row in receipt['branches']] == ['0.50','0.75','1.00']
    observations = [_read(out/row['output']/'child_observed.json') for row in receipt['branches']]
    assert len({item['pid'] for item in observations}) == 3
    assert {item['cwd'] for item in observations} == {str(tmp_path)}
    assert len({item['environment']['XTA_YOLO_SWEEP_CACHE_DIR'] for item in observations}) == 1
    for cent, item in zip((50,75,100), observations):
        args = item['arguments']
        assert args.count('--conf') == args.count('--output') == args.count('--temp') == args.count('--min_conf') == 1
        assert args[args.index('--conf')+1] == f'{cent/100:.2f}'
        assert args[args.index('--task')+1] == 'semantic'
        assert args[args.index('--interpolation')+1] == 'sam'
        assert args[args.index('--min_conf')+1] == '.13' or args[args.index('--min_conf')+1] == '0.13'
        assert args[args.index('--capture_component_replay')+1].endswith(f'conf_{cent:03d}')
        env = item['environment']
        assert env['XTA_YOLO_SWEEP_ID'] == receipt['sweep_id']
        assert Path(env['XTA_YOLO_SWEEP_SAM_ARCHIVE_ROOT']) == tmp_path/'work-root'
        assert env['XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'] == str(cent)
        assert env['XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS'] == '50'
        assert env['XTA_YOLO_SWEEP_CACHE_GIB'] == '0'
        assert env['PYTHONPATH'].split(os.pathsep)[0] == str(Path(sweep.__file__).resolve().parents[1])
        assert not Path(env['XTA_YOLO_SWEEP_WORK_DIR']).exists()
    assert len(harness.publications) == 1
    assert not (out/'.confidence_sweep.lock').exists()
    assert list((tmp_path/'work-root').iterdir()) == []
    streams = capfd.readouterr()
    assert 'scalar-live-100' in streams.out
    assert 'scalar-stderr-100' in streams.err


def test_failed_branch_archive_stage_survives_family_cleanup(harness, monkeypatch):
    from XTA.artifact_archive import read_member, reference

    monkeypatch.setenv('SWEEP_TEST_MODE', 'abandoned_sam_archive')
    output = harness.root/'failed'
    temp_root = harness.root/'local-scratch'
    with pytest.raises(subprocess.CalledProcessError):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75),
                                   output_root=output, temp_root=temp_root)
    assert _read(output/'manifest.json')['work_cleanup_complete'] is True
    (stage,) = temp_root.iterdir()
    assert read_member(reference(stage/'sam-artifacts.tar', 'evidence/receipt.json')) == b'recoverable'


@pytest.mark.parametrize('mode', (
    'exit', 'missing', 'strict_omission', 'inconsistent_incomplete', 'inconsistent_complete', 'unrecorded_omission',
    'in_progress', 'ownership', 'identity', 'missing_identity', 'policy', 'conf',
))
def test_failed_or_mismatched_scalar_never_publishes_family(harness, monkeypatch, mode):
    monkeypatch.setenv('SWEEP_TEST_MODE', mode)
    out = harness.root/'failed'
    with pytest.raises((ValueError, FileNotFoundError, subprocess.CalledProcessError)):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,75,100), output_root=out)
    receipt = _read(out/'manifest.json')
    assert receipt['status'] == 'failed'
    assert receipt['branches'][0]['status'] == 'complete'
    assert receipt['branches'][-1]['status'] == 'failed'
    assert len(harness.children) == 2
    assert all(child.poll() is not None for child in harness.children)
    assert receipt['work_cleanup_complete'] is True
    assert not (out/'.confidence_sweep_work').exists()
    assert not (out/'.confidence_sweep.lock').exists()
    assert harness.publications == []


def test_degraded_sam_cutoff_preserves_later_branches_and_skips_review(harness, monkeypatch, capsys):
    monkeypatch.setenv('SWEEP_TEST_MODE', 'incomplete')
    root = harness.root / 'degraded'
    path = sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 75, 100), output_root=root)
    receipt = _read(path)
    assert receipt['status'] == 'incomplete'
    assert receipt['incomplete_cutoffs_hundredths'] == [75]
    assert [row['status'] for row in receipt['branches']] == ['complete', 'incomplete', 'complete']
    assert receipt['branches'][1]['sam_planning_omissions'] == [{'scope': 'fixture'}]
    assert all((root / row['manifest']).is_file() for row in receipt['branches'])
    assert receipt['review_manifest'] is None
    assert receipt['review_omitted_reason'] == 'incomplete_requested_sam_work'
    assert receipt['work_cleanup_complete'] is True
    assert not (root / '.confidence_sweep.lock').exists()
    assert len(harness.children) == 3 and harness.publications == []
    assert 'SAM work was incomplete at 0.75' in capsys.readouterr().err


def _running_pid(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            assert kernel.GetExitCodeProcess(handle,ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    if stat.is_file():
        return stat.read_text().rsplit(')',1)[1].split()[0] not in {'Z','X'}
    return True


@pytest.mark.parametrize('mode',('orphan','tree_cancel'))
def test_child_tree_is_settled_before_shared_work_is_deleted(harness,monkeypatch,mode):
    monkeypatch.setenv('SWEEP_TEST_MODE',mode)
    monkeypatch.setattr(sweep,'_CHILD_SETTLE_SECONDS',.1)
    monkeypatch.setattr(sweep,'_CHILD_GRACE_SECONDS',.2)
    monkeypatch.setattr(sweep,'_CHILD_KILL_SECONDS',2.)
    out = harness.root/f'tree-{mode}'
    if mode == 'tree_cancel':
        launch = sweep.subprocess.Popen
        def interrupting_launch(command,**options):
            process = launch(command,**options)
            wait = process.wait
            once = [True]
            def interrupt(*args,**kwargs):
                if once[0] and (out/'confidence_cutoffs'/'conf_075'/'grandchild.pid').is_file():
                    once[0] = False
                    raise KeyboardInterrupt('tree cancellation')
                return wait(*args,**kwargs)
            process.wait = interrupt
            return process
        monkeypatch.setattr(sweep.subprocess,'Popen',interrupting_launch)
    with pytest.raises(KeyboardInterrupt if mode == 'tree_cancel' else RuntimeError):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75,100),output_root=out)
    receipt = _read(out/'manifest.json')
    grandchild = int((out/'confidence_cutoffs'/'conf_075'/'grandchild.pid').read_text())
    assert not _running_pid(grandchild)
    assert all(child.poll() is not None for child in harness.children)
    assert receipt['status'] == ('cancelled' if mode=='tree_cancel' else 'failed')
    assert receipt['work_cleanup_complete'] is True
    assert not Path(receipt['work_directory']).exists()
    assert harness.publications == []


def test_live_unrelated_log_is_preserved(harness):
    out = harness.root/'with-live-log'
    out.mkdir()
    log = out/'slurm-preflight.out'
    log.write_text('existing live scheduler output\n')
    summary = out / 'summary.txt'
    summary.write_text('existing unrelated report', encoding='utf-8')
    with pytest.raises(FileExistsError, match='summary.txt'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,), output_root=out,
                                   save_options=('summary',))
    assert not harness.children
    sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,),output_root=out)
    assert log.read_text() == 'existing live scheduler output\n'
    assert summary.read_text(encoding='utf-8') == 'existing unrelated report'


@pytest.mark.parametrize('reserved',('review','confidence_cutoffs','manifest.json','fanout_progress.json','fanout_storage.json'))
@pytest.mark.parametrize('option',('--temp','--capture_component_replay'))
def test_work_roots_inside_owned_outputs_fail_before_children(harness,reserved,option):
    out = harness.root/f'collision-{option[2:]}-{reserved}'
    target = out/reserved/'child' if reserved!='manifest.json' else out/reserved
    arguments = list(harness.arguments)
    kwargs = {'temp_root':target} if option=='--temp' else {}
    if option=='--capture_component_replay':
        arguments.extend((option,str(target)))
    with pytest.raises(ValueError,match='reserved confidence'):
        sweep.run_confidence_sweep(arguments,cutoffs_hundredths=(50,75),output_root=out,**kwargs)
    assert harness.children == []
    assert not (out/'manifest.json').exists()


@pytest.mark.parametrize('populated', (False, True))
def test_one_cut_range_disables_shared_caches_and_preserves_system_temp(harness, monkeypatch, populated):
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB','5')
    inherited = harness.root / 'foreign-family'
    monkeypatch.setenv('XTA_YOLO_SWEEP_WORK_DIR', str(inherited))
    monkeypatch.setenv('XTA_YOLO_SWEEP_ID', 'foreign-id')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS', '12')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS', 'bad inherited cutoffs')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'consume')
    monkeypatch.setenv('XTA_YOLO_SWEEP_OUTPUT_ROOT', str(inherited))
    monkeypatch.setenv('XTA_YOLO_SWEEP_SAM_ARCHIVE_ROOT', str(inherited))
    if populated:
        import hashlib

        key = hashlib.sha256(str((harness.root / 'model artifact.engine').resolve()).encode()).hexdigest()
        foreign_record = inherited / 'startup' / f'artifact-{key}.json'
        foreign_record.parent.mkdir(parents=True)
        foreign_record.write_bytes(b'unrelated data')
    path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(100,),output_root=harness.root/'one')
    receipt = _read(path)
    observed = _read(path.parent/'confidence_cutoffs'/'conf_100'/'child_observed.json')
    assert receipt['forward_cache']['requested_budget_gib'] == 5
    assert receipt['forward_cache']['budget_gib'] == 0
    assert observed['environment']['XTA_YOLO_SWEEP_WORK_DIR'] is None
    assert observed['environment']['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'] is None
    assert observed['environment']['XTA_YOLO_SWEEP_NATIVE_ROLE'] is None
    assert observed['environment']['XTA_YOLO_SWEEP_OUTPUT_ROOT'] is None
    assert observed['environment']['XTA_YOLO_SWEEP_SAM_ARCHIVE_ROOT'] is None
    assert receipt['upstream_fanout'] == {'status':'disabled_singleton'}
    if populated:
        assert list(foreign_record.parent.iterdir()) == [foreign_record]
        assert foreign_record.read_bytes() == b'unrelated data'
    else:
        assert not inherited.exists()
    assert observed['environment']['XTA_YOLO_SWEEP_CACHE_GIB'] == '0'
    assert observed['environment']['XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS'] == '100'
    assert Path(receipt['work_directory']).parent == harness.root/'system-temp'
    assert (harness.root/'system-temp').is_dir()
    assert not Path(receipt['work_directory']).exists()


@pytest.mark.parametrize('legacy_bootstrap',(False,True))
def test_foreign_cwd_package_cannot_shadow_current_xta(harness,monkeypatch,legacy_bootstrap):
    foreign = harness.root/'foreign'
    (foreign/'XTA').mkdir(parents=True)
    for name in ('__init__.py','__main__.py'):
        (foreign/'XTA'/name).write_text("raise RuntimeError('foreign package was imported')")
    package_parent = str(Path(sweep.__file__).resolve().parents[1])
    if legacy_bootstrap:
        with mock.patch.object(sweep.sys,'version_info',(3,10)):
            command = sweep._child_command(['--help'],package_parent)
    else:
        command = sweep._child_command(['--help'],package_parent)
    env = {**os.environ,'PYTHONPATH':package_parent,'PYTHONSAFEPATH':'1'}
    with _POPEN(command,cwd=foreign,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as process:
        stdout,_ = process.communicate(timeout=20.)
    assert process.returncode == 0,stdout
    assert 'YOLO segmentation TTA' in stdout
    assert 'foreign package' not in stdout


def test_completed_root_and_foreign_owner_are_not_overwritten(harness):
    out = harness.root/'existing'
    out.mkdir()
    old = b'{"status":"complete","sweep_id":"previous"}\n'
    (out/'manifest.json').write_bytes(old)
    with pytest.raises(FileExistsError, match='fresh output'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,75), output_root=out)
    assert (out/'manifest.json').read_bytes() == old
    assert not (out/'.confidence_sweep.lock').exists()
    assert harness.children == []
    (out/'.confidence_sweep.lock').write_text('foreign owner')
    with pytest.raises(FileExistsError):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,75), output_root=out)
    assert (out/'.confidence_sweep.lock').read_text() == 'foreign owner'
    assert (out/'manifest.json').read_bytes() == old


def test_publication_failure_and_cleanup_failure_are_not_complete(harness, monkeypatch):
    out = harness.root/'publication-failure'
    harness.publisher.publish_sweep_review = mock.Mock(side_effect=OSError('publication failed'))
    with pytest.raises(OSError, match='publication failed'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,), output_root=out)
    assert _read(out/'manifest.json')['status'] == 'failed'
    harness.publisher.publish_sweep_review = mock.Mock(return_value=None)
    with mock.patch.object(sweep, '_remove_owned_work', side_effect=OSError('active work owner')):
        with pytest.raises(OSError, match='active work owner'):
            sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,),
                output_root=harness.root/'cleanup-failure')
    failed = _read(harness.root/'cleanup-failure'/'manifest.json')
    assert failed['status'] == 'failed' and failed['work_cleanup_complete'] is False
    harness.publisher.publish_sweep_review.assert_not_called()
    assert not (harness.root/'cleanup-failure'/'review').exists()


def test_review_receipt_is_relative_and_cache_is_gone_before_complete(harness, monkeypatch):
    reads = []
    read_bytes = Path.read_bytes

    def count_branch_receipts(path):
        if path.name == 'manifest.json' and path.parent.name.startswith('conf_'):
            reads.append(path.parent.name)
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', count_branch_receipts)
    def publish(root, rows, *, sweep_id, temp_dir):
        review = root/'confidence_review'/'manifest.json'
        review.parent.mkdir()
        review.write_text(json.dumps({'status':'complete','sweep_id':sweep_id}))
        assert _read(root/'manifest.json')['status'] == 'in_progress'
        assert not Path(_read(root/'manifest.json')['work_directory']).exists()
        assert len(rows) == 2
        return review
    harness.publisher.publish_sweep_review = publish
    with mock.patch.object(sweep,'assert_tta_artifacts_unchanged',wraps=sweep.assert_tta_artifacts_unchanged) as final_hash:
        path = sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,100), output_root=harness.root/'review')
    assert final_hash.call_count == 1
    assert reads == ['conf_050', 'conf_100']
    receipt = _read(path)
    assert receipt['review_manifest'] == 'confidence_review/manifest.json'
    assert receipt['status'] == 'complete'
    assert not (path.parent/'.confidence_sweep_work').exists()


def test_replaced_scalar_receipt_prevents_family_completion(harness):
    def replace_receipt(root, rows, **kwargs):
        path = rows[0][1] / 'manifest.json'
        original = path.stat()
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, path)

    harness.publisher.publish_sweep_review = replace_receipt
    root = harness.root / 'replaced-receipt'
    with pytest.raises(RuntimeError, match='Scalar cutoff receipt changed'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,), output_root=root)
    assert _read(root / 'manifest.json')['status'] == 'failed'


def test_postcommit_lock_cleanup_error_preserves_completed_receipt(harness,capsys):
    with mock.patch.object(sweep,'_release_owned_lock',side_effect=OSError('lock cleanup fixture')):
        path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,),
            output_root=harness.root/'stale-owner')
    assert _read(path)['status'] == 'complete'
    assert (path.parent/'.confidence_sweep.lock').is_file()
    assert 'could not be retired' in capsys.readouterr().err


def test_unproved_process_tree_retains_work_and_never_publishes_review(harness,monkeypatch):
    monkeypatch.setattr(sweep,'_wait_tree',lambda *args,**kwargs:False)
    out = harness.root/'unproved-tree'
    with pytest.raises(sweep._ChildTreeUnsettled):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,),output_root=out)
    receipt = _read(out/'manifest.json')
    assert receipt['status'] == 'failed'
    assert receipt['work_cleanup_complete'] is False
    assert Path(receipt['work_directory']).is_dir()
    assert harness.publications == []


def test_interrupt_joins_active_scalar_and_records_cancelled_family(harness, monkeypatch):
    real_launch = sweep.subprocess.Popen
    def interrupting_launch(command, **options):
        process = real_launch(command, **options)
        original_wait = process.wait
        once = [True]
        def wait(*args, **kwargs):
            if once[0]:
                once[0] = False
                raise KeyboardInterrupt('controlled cancellation')
            return original_wait(*args, **kwargs)
        process.wait = wait
        return process
    monkeypatch.setattr(sweep.subprocess, 'Popen', interrupting_launch)
    out = harness.root/'cancelled'
    with pytest.raises(KeyboardInterrupt):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50,75), output_root=out)
    receipt = _read(out/'manifest.json')
    assert receipt['status'] == 'cancelled'
    assert all(child.poll() is not None for child in harness.children)
    assert receipt['work_cleanup_complete'] is True
    assert harness.publications == []


@pytest.mark.parametrize('cutoffs', ((), (50,50), (-1,50), (50,101), (True,50), (.5,50)))
def test_invalid_hundredths_do_not_touch_output(harness, cutoffs):
    out = harness.root/'invalid'
    with pytest.raises(ValueError):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=cutoffs, output_root=out)
    assert not out.exists()
    assert harness.children == []


def test_producer_fans_out_each_native_view_before_independent_consumer_suffixes(harness,monkeypatch):
    monkeypatch.setenv('SWEEP_TEST_FANOUT','native')
    out = harness.root/'native-family'
    path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75,100),output_root=out)
    family = _read(path)
    upstream = family['upstream_fanout']
    assert upstream['status'] == 'shared_native_products'
    assert upstream['validated'] is True and upstream['validated_arms'] == 3
    assert upstream['produced_views'] == 2
    assert upstream['consumed_views'] == 4
    assert upstream['scalar_views'] == 0 and upstream['fallback_reasons'] == []
    assert upstream['target_cutoffs_hundredths'] == [50,75,100]
    assert upstream['producer_cutoff_hundredths'] == 50
    assert upstream['progress_file'] == 'fanout_progress.json'
    assert upstream['storage_file'] == 'fanout_storage.json'
    assert upstream['storage_measured'] is True
    assert [row['upstream_role'] for row in family['branches']] == ['produce','consume','consume']
    events = [json.loads(line) for line in (out/'fixture_work_events.jsonl').read_text().splitlines()]
    forwards = [event for event in events if event['stage']=='forward']
    renders = [event for event in events if event['stage']=='render']
    assert len(forwards) == len(renders) == 2
    assert all(event['arm']==50 for event in [*forwards,*renders])
    assert len([event for event in events if event['stage']=='native_product']) == 6
    assert len([event for event in events if event['stage']=='native_consume']) == 4
    assert [event['arm'] for event in events if event['stage']=='independent_suffix'] == [50,75,100]
    for view in ('transverse','sagittal'):
        product_positions = [index for index,event in enumerate(events)
            if event['stage']=='native_product' and event['view']==view]
        assert len(product_positions) == 3
        assert max(product_positions) < next(index for index,event in enumerate(events)
            if event['stage']=='independent_suffix')
    observations = []
    suffixes = []
    for cent,row in zip((50,75,100),family['branches']):
        branch = out/row['output']
        assert (branch/'producer_precreated.json').is_file()
        observed = _read(branch/'child_observed.json')
        observations.append(observed)
        env = observed['environment']
        assert env['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'] == '50,75,100'
        assert env['XTA_YOLO_SWEEP_OUTPUT_ROOT'] == str(out)
        assert env['XTA_YOLO_SWEEP_NATIVE_ROLE'] == row['upstream_role']
        native = _read(branch/'native_inputs.json')
        expected = [int(.9 >= cent/100),int(.6 >= cent/100)]
        assert native == [expected,expected]
        suffixes.append(_read(branch/'suffix_state.json'))
    assert len({item['pid'] for item in observations}) == 3
    assert [item['cent'] for item in suffixes] == [50,75,100]
    assert len({tuple(item['independent_result']) for item in suffixes}) == 3
    assert not Path(family['work_directory']).exists()


def test_supported_and_unsupported_upstream_views_are_reported_as_partial_fallback(harness,monkeypatch):
    monkeypatch.setenv('SWEEP_TEST_FANOUT','partial')
    path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75,100),
        output_root=harness.root/'partial-native')
    family = _read(path)
    assert family['status'] == 'complete'
    assert family['upstream_fanout']['status'] == 'partial_fallback'
    assert family['upstream_fanout']['produced_views'] == 1
    assert family['upstream_fanout']['consumed_views'] == 2
    assert family['upstream_fanout']['scalar_views'] == 3
    assert family['upstream_fanout']['fallback_reasons'] == ['sagittal fixture route unsupported']


def test_explicit_unsupported_upstream_never_claims_shared_native_work(harness):
    path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75),
        output_root=harness.root/'scalar-fallback')
    upstream = _read(path)['upstream_fanout']
    assert upstream['status'] == 'unsupported'
    assert upstream['validated'] is True
    assert upstream['produced_views'] == upstream['consumed_views'] == 0
    assert upstream['scalar_views'] == 2
    assert upstream['fallback_reasons'] == ['fixture upstream route is unsupported']
    assert upstream['storage_file'] is None and upstream['storage_measured'] is False


@pytest.mark.parametrize('mode',('unvalidated','negative_count','boolean_count','wrong_target_count',
    'wrong_role','silent_fallback','missing_receipt'))
def test_unvalidated_or_malformed_upstream_evidence_stops_before_consumer(harness,monkeypatch,mode):
    monkeypatch.setenv('SWEEP_TEST_FANOUT',mode)
    out = harness.root/f'bad-native-{mode}'
    with pytest.raises(ValueError):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75),output_root=out)
    family = _read(out/'manifest.json')
    assert family['status'] == 'failed'
    assert family['branches'][0]['status'] == 'failed'
    assert len(harness.children) == 1
    assert harness.publications == []
    assert family['work_cleanup_complete'] is True


def test_native_consumer_corruption_is_a_failure_not_scalar_fallback(harness,monkeypatch):
    monkeypatch.setenv('SWEEP_TEST_FANOUT','corrupt')
    out = harness.root/'corrupt-native'
    with pytest.raises(subprocess.CalledProcessError):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75,100),output_root=out)
    family = _read(out/'manifest.json')
    assert family['status'] == 'failed'
    assert [row['status'] for row in family['branches']] == ['complete','failed']
    assert family['upstream_fanout']['validated_arms'] == 1
    assert harness.publications == []


def test_claimed_native_consumption_without_a_producer_is_rejected(harness,monkeypatch):
    monkeypatch.setenv('SWEEP_TEST_FANOUT','unproduced_consumer')
    out = harness.root/'unproduced-native'
    with pytest.raises(ValueError,match='without a validated producer'):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75),output_root=out)
    assert _read(out/'manifest.json')['status'] == 'failed'
    assert harness.publications == []


def test_sam_omission_does_not_discard_valid_native_products_for_later_arms(harness,monkeypatch):
    monkeypatch.setenv('SWEEP_TEST_FANOUT','native')
    monkeypatch.setenv('SWEEP_TEST_MODE','incomplete')
    path = sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75,100),
        output_root=harness.root/'native-with-sam-omission')
    family = _read(path)
    assert family['status'] == 'incomplete'
    assert family['upstream_fanout']['status'] == 'shared_native_products'
    assert family['upstream_fanout']['consumed_views'] == 4
    assert [row['status'] for row in family['branches']] == ['complete','incomplete','complete']
    assert harness.publications == []


def test_preexisting_fanout_progress_is_reserved_and_preserved(harness):
    out = harness.root/'existing-progress'
    out.mkdir()
    progress = out/'fanout_progress.json'
    progress.write_bytes(b'previous progress identity')
    with pytest.raises(FileExistsError):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75),output_root=out)
    assert progress.read_bytes() == b'previous progress identity'
    assert harness.children == []


def _numeric_upstream_receipt(role='produce',**extra):
    values={'role':role,'validated':True,'produced_views':int(role=='produce'),
        'consumed_views':int(role=='consume'),'scalar_views':0,'target_count':2,
        'fallback_reasons':[],**extra}
    return {'confidence_sweep':{'upstream_fanout':values}}


def test_optional_fanout_work_and_lifetime_measurements_survive_exactly():
    work={'rendered_frames':61,'render_batches':61,'forward_calls':61,'mask_decodes':122,
        'anchors_once':2**60+1,'scalar_anchor_entries':2**60+37,'active_bins':3,
        'identical_native_crops_reused':61,'native_crop_bytes':8192}
    value=_numeric_upstream_receipt(native_frames=61,stored_native_bytes=12345,
        native_views_retired_after_projection=1,work=work,
        projection_fallbacks={'cpu scalar geometry fallback':2})
    result=sweep._fanout_evidence(value,role='produce',target_count=2)
    assert result==value['confidence_sweep']['upstream_fanout']
    assert json.loads(json.dumps(result))['work']['anchors_once']==2**60+1


def test_optional_fanout_aggregation_reports_measured_arm_coverage():
    producer=sweep._fanout_evidence(_numeric_upstream_receipt(native_frames=61,
        stored_native_bytes=12345,native_views_retired_after_projection=0,
        work={'forward_calls':61,'anchors_once':2**60+1},projection_fallbacks={'CPU geometry':2}),
        role='produce',target_count=2)
    consumer=sweep._fanout_evidence(_numeric_upstream_receipt('consume',work={'forward_calls':0},
        projection_fallbacks={'CPU geometry':1}),role='consume',target_count=2)
    manifest={'cutoffs_hundredths':[50,80],'upstream_fanout':{},
        'branches':[{'upstream_fanout':producer},{'upstream_fanout':consumer}]}
    sweep._update_fanout_summary(manifest)
    summary=manifest['upstream_fanout']
    assert summary['status']=='shared_native_products'
    assert summary['work']=={'forward_calls':61,'anchors_once':2**60+1}
    assert summary['work_reported_arms']=={'forward_calls':2,'anchors_once':1}
    assert summary['native_frames']==61 and summary['stored_native_bytes']==12345
    assert summary['optional_counter_reported_arms']=={
        'native_frames':1,'stored_native_bytes':1,'native_views_retired_after_projection':1}
    assert summary['projection_fallbacks']=={'CPU geometry':3}
    assert summary['projection_fallbacks_reported_arms']=={'CPU geometry':2}
    assert summary['producer_evidence']==producer


@pytest.mark.parametrize('extra',(
    {'native_frames':True},{'stored_native_bytes':-1},{'native_views_retired_after_projection':1.5},
    {'work':{'anchors_once':False}},{'work':{'active_bins':-1}},{'work':{'':1}},
    {'projection_fallbacks':{'unsupported':True}},{'projection_fallbacks':['unsupported']},
))
def test_optional_fanout_counters_must_be_exact_nonnegative_integers(extra):
    with pytest.raises(ValueError):
        sweep._fanout_evidence(_numeric_upstream_receipt(**extra),role='produce',target_count=2)


def test_preexisting_fanout_storage_is_reserved_and_preserved(harness):
    out=harness.root/'existing-storage'
    out.mkdir()
    storage=out/'fanout_storage.json'
    storage.write_bytes(b'previous storage identity')
    with pytest.raises(FileExistsError):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,75),output_root=out)
    assert storage.read_bytes()==b'previous storage identity'
    assert harness.children==[]


def test_reference_switch_keeps_scalar_shared_caches_but_removes_native_markers(harness,monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_VIEW_MAJOR','0')
    monkeypatch.setenv('SWEEP_TEST_FANOUT','native')
    for key,value in (
        ('XTA_YOLO_SWEEP_NATIVE_ROLE','produce'),
        ('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS','bad inherited targets'),
        ('XTA_YOLO_SWEEP_OUTPUT_ROOT','foreign family')):
        monkeypatch.setenv(key,value)
    out=harness.root/'scalar-reference'
    out.mkdir()
    for name in ('fanout_progress.json','fanout_storage.json'):
        (out/name).write_text('unmanaged existing evidence')
    path=sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,80),output_root=out)
    family=_read(path)
    assert family['status']=='complete'
    assert family['branch_policy']=='independent_scalar_full_pipeline'
    assert family['upstream_fanout']=={'status':'disabled'}
    assert family['forward_cache']['budget_gib']==64
    assert family['forward_cache']['summary']['entries']==2
    for row in family['branches']:
        assert 'upstream_role' not in row and 'upstream_fanout' not in row
        observed=_read(out/row['output']/'child_observed.json')
        env=observed['environment']
        assert env['XTA_YOLO_SWEEP_WORK_DIR']==family['work_directory']
        assert env['XTA_YOLO_SWEEP_CACHE_GIB']=='64'
        assert env['XTA_YOLO_SWEEP_FLOOR_HUNDREDTHS']=='50'
        assert env['XTA_YOLO_SWEEP_NATIVE_ROLE'] is None
        assert env['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'] is None
        assert env['XTA_YOLO_SWEEP_OUTPUT_ROOT'] is None
    for name in ('fanout_progress.json','fanout_storage.json'):
        assert (out/name).read_text()=='unmanaged existing evidence'


@pytest.mark.parametrize('value',('false','-1','01'))
def test_reference_switch_rejects_ambiguous_values_before_touching_outputs(harness,monkeypatch,value):
    monkeypatch.setenv('XTA_YOLO_SWEEP_VIEW_MAJOR',value)
    out=harness.root/'invalid-reference'
    with pytest.raises(ValueError,match='must be 0 or 1'):
        sweep.run_confidence_sweep(harness.arguments,cutoffs_hundredths=(50,80),output_root=out)
    assert not out.exists()
    assert harness.children==[]
