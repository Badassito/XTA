"""Real frozen planning/evidence/selection, live grants, and a CPU truth tracker.

The SDK is mocked deliberately. These tests qualify scientific equivalence and
the production composition boundary, not GPU throughput or SAM model quality.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from XTA import sam_extrapolation as extrapolation, sam_resources as resources
from XTA.sam_crop_retry import SamCropRetryPolicy
from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_integration import SamInterpolationContext
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_dynamic_crop_retry_adversarial import _decode_components, _result

GIB=1024**3


class _TruthTracker(SamInterpolationTracker):
    """Mock only inference; preserve live scope/request/cache admission calls."""
    def __init__(self, root, cache, observed, truth, context, *, require_overlap=False):
        super().__init__(model_path='mock-SDK',device_ids=(0,1),artifact_root=root,
            source_cache_ref=cache)
        self.observed,self.truth,self.context=observed,truth,context
        self.calls=[]
        self._calls_lock=threading.Lock()
        self.second_child=threading.Event()
        self.require_overlap=require_overlap
        self.child_threads=set()

    def _run_request(self, request, admission):
        profile=request['resource_profile']
        profile._validate_owner()
        assert self.context._resource_local.profile is profile
        x0,y0,x1,y1=request['crop_xyxy']
        admission.validate_request(request['frame_stop']-request['frame_start'],(y1-y0)*(x1-x0))
        labels=np.unique(self.observed[request['seed_frame'],y0:y1,x0:x1][request['seed_mask']])
        assert len(labels)==1 and int(labels[0]) in (1,2)
        label=int(labels[0])
        expected=self.observed[request['seed_frame'],y0:y1,x0:x1]==label
        np.testing.assert_array_equal(request['seed_mask'],expected)
        limits=resources.validate_sam_tracker_scope_admission(admission)
        capacity=admission.max_in_flight
        assert 1<=capacity<=profile.execution_slots
        session,transfer=limits['maximum_session_cpu_bytes'],limits['consumer_transfer_bytes']
        if limits['defer_refill_until_consumed']:
            assert capacity==1
            required=max(session,transfer)
        else:
            required=capacity*session+transfer
        assert transfer>=3*limits['maximum_raw_mask_bytes']
        assert required<=limits['attempt_peak_bytes']<=profile.assigned_cpu_wave_bytes
        child=profile._lease.retry_child_factory is not None
        if child:
            assert profile.scope_id!='parent'
            with self._calls_lock:
                self.child_threads.add(threading.get_ident())
            if label==2:
                self.second_child.set()
            elif self.require_overlap:
                assert self.second_child.wait(8), 'second independent group never reached the SDK mock'
        row=dict(label=label,seed_frame=request['seed_frame'],
            interval=(request['frame_start'],request['frame_stop'],request['direction']),
            crop=(y0,x0,y1,x1),seed_sha256=hashlib.sha256(request['seed_mask'].tobytes()).hexdigest(),
            child=child,thread=threading.get_ident(),run_id=request['run_id'],wave=capacity)
        with self._calls_lock:
            self.calls.append(row)
        return _result({frame:(self.truth[frame,y0:y1,x0:x1]==label)
            for frame in range(request['frame_start'],request['frame_stop'])})

    def _validate_iterator_options(self, admission, options):
        assert admission is not None
        limits=resources.validate_sam_tracker_scope_admission(admission)
        assert options.get('max_in_flight',self.worker_count)==admission.max_in_flight
        assert options.get('defer_refill_until_consumed',False)==limits['defer_refill_until_consumed']

    def _iterate(self, indexed, cache, admission):
        assert admission is not None
        admission.acquire_scope()
        try:
            admission.validate_cache(cache)
            for index,request in indexed:
                if self._cancel.is_set():
                    raise RuntimeError('truth tracker cancelled')
                yield index,self._run_request(request,admission)
        finally:
            admission.release_scope()

    def iter_results(self, requests, *, source_cache_ref=None, scope_admission=None, **options):
        self._validate_iterator_options(scope_admission,options)
        yield from self._iterate(enumerate(requests),source_cache_ref,scope_admission)

    def iter_family_results(self, families, *, source_cache_ref=None, scope_admission=None, **options):
        self._validate_iterator_options(scope_admission,options)
        factories={index:family.request_factory for family in families for index in family.input_indices}
        indexed=((index,factories[index](index)) for index in sorted(factories))
        yield from self._iterate(indexed,source_cache_ref,scope_admission)

    def release_result(self, result):
        pass  # The truth mock has no worker transfer artifacts.


@contextmanager
def _case(root, *, overlap=False, decline=False, mode='whole', failure=False, resolved=()):
    observed=np.zeros((8,384,768),np.uint8)
    observed[2:5,80:87,120:127]=1
    observed[2:5,260:267,420:427]=2
    truth=observed.copy()
    for label,y,x in ((1,83,120),(2,263,420)):
        end=min(x+461,750)
        truth[5,y,x:end]=label
        truth[6,y,end-50:end-43]=label
        truth[7,y-3:y+4,x:x+7]=label
        if label in resolved:
            for frame in (5,6,7):
                truth[frame][truth[frame]==label]=0
                truth[frame,y-3:y+4,x:x+7]=label
    source=np.zeros_like(observed)
    context=SamInterpolationContext(model_path='mock-SDK',device_ids=(0,1),
        temp_dir=root/'temp',evidence_root=root/'context-evidence',source_volume=source,
        source_identity='immutable-gray-test-source',crop_mode=mode)
    pool=type('Pool',(),{})()
    pool.capacity,pool.in_use=128*GIB,0
    pool.condition=threading.Condition(threading.RLock())
    cache=materialize_interpolation_image_cache(source,path=root/'gray-cache.bin',
        physical_view_id='transverse',source_identity='immutable-gray-test-source')
    runtime=_TruthTracker(root/'workers',cache,observed,truth,context,require_overlap=overlap)
    context._runtime=runtime
    providers=[]
    with resources.admit_sam_parent_resources(pool,10*GIB,'parent',worker_count=2,
            execution_slots=2,base_allowance_bytes=4*GIB,headroom_probe=lambda:256*GIB) as profile:
        with context.resource_scope(profile):
            prepared=extrapolation.prepare_sam_extrapolation_pass(observed,scope='sam',distance=3,
                walk_back=1,min_radius=3.,crop_mode=mode,resource_profile=profile,
                eligible_terminals=lambda terminal,direction:terminal.frame_index==4 and direction==1)
            assert len(prepared.groups)==2 and len(prepared.runs)==4
            if decline:
                pool.capacity=pool.in_use
            lock=threading.Lock()
            def provider(retry):
                current=context._resource_local.profile
                current._validate_owner()
                group=retry.groups[0]
                original_id=group.original_group_id
                with lock:
                    attempt=1+sum(row['group_id']==original_id for row in providers)
                    providers.append(dict(group_id=original_id,box=group.context_bbox_yx,
                        attempt=attempt,child=current is not profile,profile=current.scope_id,
                        wave=int(retry.cpu_wave_admission['max_in_flight'])))
                if failure and original_id==prepared.groups[1].group_id and attempt==2:
                    raise OSError('second group later retry failed')
                @contextmanager
                def entered():
                    assert context._resource_local.profile is current
                    current._validate_owner()
                    yield cache
                return entered()
            def execute(*, parallel=True, policy=None):
                return extrapolation.extrapolate_sam_view_volume_pass(observed,work_dir=root/'output',
                    scope='sam',prepared_plan=prepared,image_provider=cache,runtime=runtime,
                    distance=3,walk_back=1,min_radius=3.,crop_mode=mode,resource_profile=profile,
                    crop_retry_policy=policy or SamCropRetryPolicy(enabled=True),retry_image_provider=provider,
                    _retry_group_context=context if parallel else None)
            try:
                yield dict(run=execute,observed=observed,truth=truth,prepared=prepared,
                    providers=providers,runtime=runtime,context=context,pool=pool,profile=profile,root=root)
                assert profile._lease.scope_holds==0
                assert not any(not hold.closed for hold in getattr(context,'_retry_source_holds',()))
            finally:
                context.close()
    assert pool.in_use==0


def _science(case, components, stats):
    actual=_decode_components(components,case['observed'].shape)
    np.testing.assert_array_equal(actual,(case['truth']>0)&(case['observed']==0))
    bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
    ledger=stats['sam_crop_retry']
    calls=sorted((row['label'],row['seed_frame'],row['interval'],row['crop'],row['seed_sha256'])
        for row in case['runtime'].calls)
    boxes={gid:[(row['attempt_index'],row['crop_bbox_yx'],row.get('retry_pixel_frames',0),
        row.get('retry_tracker_frames',0),row['status'],row.get('resolution_status')) for row in rows]
        for gid,rows in ledger['attempt_history'].items()}
    original_order=[group.group_id for group in case['prepared'].groups]
    imported_originals=[group.get('original_group_id') or gid for gid,group in bundle.groups.items()]
    assert set(imported_originals)==set(original_order)
    replaced_order=[gid for gid in original_order if ledger['last_complete_attempts'][gid]['attempt_index']]
    assert [gid for gid in imported_originals if gid in replaced_order]==replaced_order
    return actual,calls,boxes,ledger['charged_pixel_frames'],ledger['charged_tracker_frames']


@pytest.mark.parametrize('mode',('whole','tiled'))
def test_independent_live_group_chains_keep_serial_science_and_funded_pairs(tmp_path,mode):
    with _case(tmp_path/'serial',mode=mode) as serial:
        _,serial_stats,serial_components=serial['run'](parallel=False)
        expected=_science(serial,serial_components,serial_stats)
        assert not any(row['child'] for row in serial['runtime'].calls)
    with _case(tmp_path/'parallel',mode=mode,overlap=True) as parallel:
        _,stats,components=parallel['run']()
        actual=_science(parallel,components,stats)
        assert stats['sam_crop_retry']['group_execution']['child_groups']==2
        assert len(parallel['runtime'].child_threads)==2
        assert all(1<=row['wave']<=2 for row in parallel['providers'])
        assert any(row['wave']==2 for row in parallel['providers'])
        assert len(parallel['providers'])>=4
        for left,right in zip(actual,expected):
            if isinstance(left,np.ndarray):
                np.testing.assert_array_equal(left,right)
            else:
                assert left==right


def test_explicit_retry_memory_cap_keeps_child_pairs_at_one_session(tmp_path,monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES','1')
    policy=SamCropRetryPolicy(enabled=True,max_retry_memory_bytes=200*1024**2)
    with _case(tmp_path,overlap=True) as case:
        _,stats,components=case['run'](policy=policy)
        _science(case,components,stats)
        assert stats['sam_crop_retry']['group_execution']['child_groups']==2
        assert case['providers'] and all(row['wave']==1 for row in case['providers'])
        assert all(row['wave']==1 for row in case['runtime'].calls if row['child'])


@pytest.mark.parametrize('cap',('pixels','frames'))
def test_either_finite_effective_global_limit_keeps_original_serial_order(tmp_path,cap):
    policy=SamCropRetryPolicy(enabled=True,**{
        'max_extra_pixel_frames' if cap=='pixels' else 'max_extra_tracker_frames':10**9})
    with _case(tmp_path) as case:
        _,stats,components=case['run'](policy=policy)
        _science(case,components,stats)
        assert stats['sam_crop_retry']['group_execution']['mode']=='serial'
        assert stats['sam_crop_retry']['group_execution']['reason']=='finite_global_retry_budget'
        assert not case['runtime'].child_threads
        assert all(not row['child'] for row in case['providers'])
        original=[group.group_id for group in case['prepared'].groups]
        assert list(dict.fromkeys(row['group_id'] for row in case['providers']))==original


def test_full_ledger_declines_children_and_replays_same_serial_authority(tmp_path):
    with _case(tmp_path,decline=True) as case:
        _,stats,components=case['run']()
        _science(case,components,stats)
        execution=stats['sam_crop_retry']['group_execution']
        assert execution['child_groups']==0 and execution['serial_fallback_groups']==2
        assert execution['resource_declines']>=2 and not case['runtime'].child_threads


def test_failed_later_group_preserves_prior_complete_attempts_without_final_publication(tmp_path):
    with _case(tmp_path,failure=True) as case:
        policy=SamCropRetryPolicy(enabled=True,max_extra_tracker_frames=10**9)
        with pytest.raises(OSError,match='second group later retry failed'):
            case['run'](policy=policy)
        ledger_path=next((tmp_path/'output').glob('sam_extrap_*/crop_retry.json'))
        ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
        first,second=(group.group_id for group in case['prepared'].groups)
        assert ledger['last_complete_attempts'][first]['attempt_index']>=2
        assert ledger['last_complete_attempts'][second]['attempt_index']==1
        assert ledger['attempt_history'][second][-1]['status']=='failed'
        assert ledger['initial_child_crop_diagnostics'].keys()=={first,second}
        assert any(row['original_group_id']==second for row in ledger['orchestration_refusals'])
        assert ledger['scope_error']['original_group_id']==second
        assert not (ledger_path.parent/'selection.json').exists()
        assert not list(ledger_path.parent.glob('sam_extrapolation_*.cvol'))


@pytest.mark.parametrize('resolved',((2,),(1,2)))
def test_no_retry_or_single_unresolved_group_never_requests_fresh_children(tmp_path,monkeypatch,resolved):
    def unexpected(*_args,**_kwargs):
        pytest.fail('a no-retry/single-chain pass requested a fresh child')
    monkeypatch.setattr(resources,'prepare_retry_child_resources',unexpected)
    with _case(tmp_path,resolved=resolved) as case:
        _,stats,components=case['run']()
        _science(case,components,stats)
        assert stats['sam_crop_retry']['group_execution']['mode']=='serial'
        assert not case['runtime'].child_threads
        assert len({row['group_id'] for row in case['providers']})==2-len(resolved)


def test_parallel_failure_records_child_mode_counts_and_last_complete_diagnostics(tmp_path,monkeypatch):
    from XTA import sam_retry_groups
    events=[]
    monkeypatch.setattr(sam_retry_groups,'_event',lambda name,**fields:events.append((name,fields)))
    with _case(tmp_path,failure=True,overlap=True) as case:
        with pytest.raises(OSError,match='second group later retry failed'):
            case['run']()
        ledger_path=next((tmp_path/'output').glob('sam_extrap_*/crop_retry.json'))
        ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
        second=case['prepared'].groups[1].group_id
        assert ledger['group_execution']['mode']=='parallel_unlimited'
        assert ledger['group_execution']['peak_pending_groups']==2
        assert ledger['last_complete_attempts'][second]['attempt_index']==1
        assert ledger['attempt_history'][second][-1]['status']=='failed'
        failed=[fields for name,fields in events if name=='sam_retry_group_end' and fields['failed']]
        assert any(row['original_group_id']==second and row['group_ordinal']==1
            and row['attempt_count']==2 for row in failed)
        assert not (ledger_path.parent/'selection.json').exists()
        assert not list(ledger_path.parent.glob('sam_extrapolation_*.cvol'))
