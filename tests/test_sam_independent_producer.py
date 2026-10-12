"""Real production sources refill funded inputs while owner evidence is blocked."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest

from XTA import sam_evidence, sam_interpolation as interpolation, sam_resources as resources
from XTA import sam_tracker_runtime as sam
from XTA.interpolation import _ByteAdmissionPool
from tests.test_lta_experimental import _Predictor, _Tracker

GIB = 1024**3


def wait_until(predicate, timeout=8.):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'SAM production boundary did not reach the expected state'


class CpuPool:
    """The real task adapter on CPU, with exact late/reversed ACK attribution."""
    def __init__(self, *, proof=True):
        self.active = {}
        self.submissions, self.completions = [], []
        self.closed, self.proof = False, proof
        self.first = True
        self.ack_gate = None

    def submit(self, task, *, execution_device_id):
        assert execution_device_id not in self.active
        self.active[execution_device_id] = task
        self.submissions.append(task)

    def wait_result(self, *, timeout):
        if not self.active:
            raise TimeoutError
        if self.completions and self.ack_gate is not None and not self.ack_gate.is_set():
            time.sleep(.001)
            raise TimeoutError
        # Reverse the first wave to prove publication follows ACK arrival.
        if self.first and len(self.active)<2:
            time.sleep(.001)
            raise TimeoutError
        self.first = False
        device = max(self.active)
        task = self.active.pop(device)
        predictor = _Predictor(_Tracker())
        def handle(request):
            if request['type']=='start_session':
                return {'session_id':'mask-test'}
            state = predictor._all_inference_states.pop('mask-test')['state']
            state.clear()
            return {'is_success':True} if self.proof else {'success':True}
        predictor.handle_request = handle
        context = SimpleNamespace(predictor=predictor, profile={}, sam_runtime={})
        previous = os.environ.get('LTA_EXECUTION_DEVICE_ID')
        os.environ['LTA_EXECUTION_DEVICE_ID'] = str(device)
        try:
            output = sam.execute_interpolation_tracker_task(context,task.kind,task.payload)
        finally:
            if previous is None:
                os.environ.pop('LTA_EXECUTION_DEVICE_ID',None)
            else:
                os.environ['LTA_EXECUTION_DEVICE_ID'] = previous
        path = Path(output['artifact_path'])
        self.completions.append(task)
        return SimpleNamespace(work_id=task.work_id,attempt_token=task.attempt_token,
            execution_device_id=device,worker_pid=os.getpid(),worker_index=0,
            artifact_path=str(path),artifact_sha256=sam._sha256(path))

    def shutdown(self, *, timeout, force):
        self.active.clear()
        self.closed = True

    def force_close(self, *, timeout):
        self.shutdown(timeout=timeout,force=True)

    @property
    def workers_settled(self):
        return self.closed


def observations():
    masks = np.zeros((5,25,85),np.uint8)
    for label in range(1,9):
        masks[0,9:15,4+9*(label-1):10+9*(label-1)] = label
        masks[4,9:15,4+9*(label-1):10+9*(label-1)] = label
    masks.setflags(write=False)
    return masks


@contextmanager
def production(tmp_path, *, proof=True, physical=True, bank_jobs=None, mode='whole'):
    masks = observations()
    ledger = _ByteAdmissionPool(20*GIB,'producer-test')
    with resources.admit_sam_parent_resources(ledger,GIB,'producer',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        prepared = interpolation.prepare_sam_interpolation_pass(masks,gap_distance=5,
            min_radius=0,interpolation_walk_back=0,resource_profile=profile,crop_mode=mode)
        assert len(prepared.runs)>8
        if not physical or bank_jobs is not None:
            wave = dict(prepared.cpu_wave_admission)
            wave['assigned_cpu_wave_bytes'] = wave['peak_cpu_wave_estimate_bytes']
            prepared = replace(prepared,cpu_wave_admission=MappingProxyType(wave))
        if not physical:
            profile._lease.headroom_probe = lambda:0
        cache = sam.materialize_interpolation_image_cache(np.zeros_like(masks),path=tmp_path/'images.bin',
            physical_view_id='transverse',source_identity='producer-test')
        if bank_jobs is not None:
            with resources.admit_sam_prepared_scope(profile,prepared,cache,max_in_flight=2) as probe:
                compact = probe._limits['per_completion_job_bytes']
            physical_bytes = ledger.in_use+bank_jobs*compact
            profile._lease.headroom_probe = lambda:physical_bytes
        pool = CpuPool(proof=proof)
        tracker = sam.SamInterpolationTracker(model_path='unused',device_ids=(0,1),
            artifact_root=tmp_path/'staging',source_cache_ref=cache)
        tracker._pool = pool
        tracker._residency_released = False
        try:
            yield masks,ledger,profile,prepared,cache,pool,tracker
        finally:
            tracker.close()
    assert ledger.in_use == 0


def source_stream(prepared,cache,profile,tracker):
    groups = {str(group.group_id):group for group in prepared.groups}
    by_id = {str(item.observation_id):item for item in prepared.plan.observations}
    permit_context = interpolation._tracker_scope_admission(tracker,prepared,cache,profile,2)
    permit = permit_context.__enter__()
    source = interpolation._tracker_request_stream(prepared,prepared.runs,groups,by_id,None,profile,permit)
    return permit_context,permit,tracker.iter_results(source,source_cache_ref=cache,max_in_flight=2,
        scope_admission=permit)


@pytest.mark.parametrize('schedule',('flat','fifo'))
def test_shared_production_caller_refills_and_runs_while_evidence_owner_is_blocked(tmp_path,monkeypatch,schedule):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE',schedule)
    entered, release = threading.Event(),threading.Event()
    allow_acks = threading.Event()
    shared,errors = {},[]
    original = sam_evidence.SamEvidenceWriter.add_run
    def block_writer(writer,*args,**kwargs):
        if not entered.is_set():
            shared['writer_thread'] = threading.get_ident()
            shared['submissions_at_hold'] = len(shared['pool'].submissions)
            entered.set()
            assert release.wait(8)
        return original(writer,*args,**kwargs)
    monkeypatch.setattr(sam_evidence.SamEvidenceWriter,'add_run',block_writer)
    def owner():
        try:
            with production(tmp_path) as (masks,ledger,profile,prepared,cache,pool,tracker):
                pool.ack_gate = allow_acks
                shared.update(pool=pool,tracker=tracker,ledger=ledger,owner=threading.get_ident())
                output,stats,_ = interpolation.interpolate_sam_view_volume_pass(masks,
                    work_dir=tmp_path/'evidence',runtime=tracker,image_provider=cache,prepared_plan=prepared,
                    gap_distance=5,min_radius=0,interpolation_walk_back=0,resource_profile=profile,
                    crop_mode='whole',return_bridge_components=True)
                shared['output'] = np.asarray(output).copy()
                if isinstance(output,np.memmap):
                    output._mmap.close()
                shared['stats'] = stats
        except BaseException as error:
            errors.append(error)
            entered.set()
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert entered.wait(10)
        assert not errors
        tracker,pool = shared['tracker'],shared['pool']
        allow_acks.set()
        wait_until(lambda:len(pool.submissions)>shared['submissions_at_hold'])
        wait_until(lambda:len(pool.completions)>2*tracker.worker_count+1)
        snapshot = tracker.snapshot()['scope_consumers'][0]
        assert snapshot['held'] and snapshot['consumer_stage']=='evidence_consumer'
        assert snapshot['independent_preparation']
        assert snapshot['preparation_thread_id'] != shared['owner'] == shared['writer_thread']
        assert snapshot['base_input_jobs']<=snapshot['base_input_capacity']
        assert snapshot['completion_bank_jobs']<=snapshot['completion_bank_capacity']
    finally:
        allow_acks.set()
        release.set()
        thread.join(15)
    assert not thread.is_alive() and not errors
    assert len(shared['pool'].submissions)==len(shared['pool'].completions)==16
    assert len({task.payload['run_id'] for task in shared['pool'].submissions})==16
    assert shared['output'][0].any() and shared['stats']['added_voxels']>0
    assert shared['ledger'].in_use==0


def test_cancelled_blocked_producer_retains_source_packets_and_actual_credit_until_join(tmp_path,monkeypatch):
    entered,release = threading.Event(),threading.Event()
    original = interpolation._tracker_requests
    calls = []
    def gated(*args,**kwargs):
        calls.append(threading.get_ident())
        if len(calls)==5:
            entered.set()
            assert release.wait(8)
        yield from original(*args,**kwargs)
    monkeypatch.setattr(interpolation,'_tracker_requests',gated)
    with production(tmp_path) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        permit_context,permit,stream = source_stream(prepared,cache,profile,tracker)
        try:
            _index,first = next(stream)
            assert entered.wait(5)
            charged = ledger.in_use
            with pytest.raises(RuntimeError,match='producer remains active'):
                stream.close()
            assert tracker.workers_settled and not tracker.cleanup_settled
            assert ledger.in_use==charged and tracker._scopes
            assert list(tracker.artifact_root.glob('run-*'))
            with pytest.raises(RuntimeError,match='active scoped iterator owner'):
                tracker.release_source_cache(cache)
            with pytest.raises(RuntimeError,match='producer is still active'):
                permit.release_scope()
            release.set()
            wait_until(lambda:all(scope.producer_settled for scope in tracker._scopes.values()))
            tracker.close()
            assert tracker.cleanup_settled and not tracker._scopes
            assert not list(tracker.artifact_root.glob('run-*'))
            assert cache.path.exists()
            assert tracker.release_source_cache(cache)['proof_basis']=='worker_processes_exited'
            assert all(identity!=threading.get_ident() for identity in calls)
        finally:
            release.set()
            stream.close()
            permit_context.__exit__(None,None,None)


def test_distinct_job_maxima_and_odd_row_padding_bound_actual_packet_without_cross_product(tmp_path):
    ledger = _ByteAdmissionPool(20*GIB,'compact-bound')
    with resources.admit_sam_parent_resources(ledger,GIB,'bound',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        max_frames,max_pixels,raw = 100,1_000_000,1_000_000
        session = max(resources.cpu_session_bytes(1,max_pixels)['estimated_peak_bytes'],
            resources.cpu_session_bytes(max_frames,9)['estimated_peak_bytes'])
        wave = resources.cpu_wave_admission(session,raw,profile.assigned_cpu_wave_bytes,2)
        with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=max_pixels,
                max_frame_count=max_frames,job_count=8) as permit:
            limits = resources.validate_sam_tracker_scope_admission(permit)
            assert limits['maximum_packed_packet_bytes']<400_000
            assert limits['completion_bank_jobs']==6
            for frames,pixels in ((1,max_pixels),(max_frames,9)):
                path = tmp_path/f'{frames}-{pixels}.npz'
                masks = np.zeros((frames,pixels),bool)
                sam._atomic_npz(path,packed_masks=np.packbits(masks,axis=1),shape=np.array([1,pixels],np.int64),
                    frame_indices=np.arange(frames,dtype=np.int64),tracker_scores=np.ones(frames),
                    removed=np.zeros(frames,bool))
                assert path.stat().st_size<=limits['maximum_packed_packet_bytes']
            assert limits['prepared_bank_bytes']==limits['completion_bank_bytes']+limits['lookahead_jobs']*limits['per_prepared_job_bytes']


def test_small_finite_inventory_reserves_no_unused_completion_or_lookahead_bank():
    ledger = _ByteAdmissionPool(20*GIB,'tiny-inventory')
    with resources.admit_sam_parent_resources(ledger,GIB,'tiny',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        session = resources.cpu_session_bytes(3,9)['estimated_peak_bytes']
        wave = resources.cpu_wave_admission(session,27,profile.assigned_cpu_wave_bytes,2)
        charged = ledger.in_use
        with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=9,
                max_frame_count=3,job_count=2) as permit:
            assert permit._limits['completion_bank_jobs']==permit.lookahead_jobs==0
            assert permit._limits['prepared_bank_bytes']==0 and ledger.in_use==charged


def test_host_probe_is_outside_ledger_lock_and_current_promises_win_before_charge():
    ledger = _ByteAdmissionPool(20*GIB,'fresh-promises')
    with resources.admit_sam_parent_resources(ledger,GIB,'fresh',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        charge = ledger.in_use
        before = profile.metadata()
        acquired,outside = threading.Event(),threading.Event()
        def promise():
            with ledger.condition:
                ledger._sam_startup_future_bytes = 2*GIB
                acquired.set()
        thread = threading.Thread(target=promise)
        def sample():
            thread.start()
            if acquired.wait(2):
                outside.set()
            thread.join(2)
            return charge+GIB
        profile._lease.headroom_probe = sample
        session = resources.cpu_session_bytes(3,9)['estimated_peak_bytes']
        wave = resources.cpu_wave_admission(session,27,profile.assigned_cpu_wave_bytes,2)
        with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=9,
                max_frame_count=3,job_count=8) as permit:
            thread.join(2)
            assert outside.is_set() and not thread.is_alive()
            assert permit._limits['prepared_bank_bytes']==0 and ledger.in_use==charge
            assert profile.metadata()==before


def test_full_ledger_lends_proven_residual_to_output_bank_without_probe_or_packing_double_spend(tmp_path):
    with production(tmp_path) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        calls = []
        profile._lease.headroom_probe = lambda:(calls.append(True),64*GIB)[1]
        with ledger.reserve(ledger.capacity-ledger.in_use,'other credited parents'):
            charged = ledger.in_use
            permit_context,permit,stream = source_stream(prepared,cache,profile,tracker)
            try:
                _index,first = next(stream)
                limits = permit._limits
                loan = limits['completion_wave_loan_bytes']
                assert loan==limits['completion_bank_bytes']>0
                assert limits['completion_fresh_bytes']==limits['prepared_bank_bytes']==0
                assert not calls and ledger.in_use==charged
                assert limits['fresh_headroom_probe']=='skipped_full_ledger'
                assert limits['fresh_headroom_sample_bytes'] is None
                assert limits['completion_bank_jobs']>limits['max_in_flight']
                residual = resources._sam_evidence_residual_bytes(limits['unused_attempt_wave_bytes'],
                    limits['evidence_fixed_bytes'],limits['evidence_contract_bytes'],limits['maximum_evidence_plane_pixels'])
                assert first.packing_admission.scratch_bytes(limits['maximum_seed_pixels'])==residual-loan
                assert loan<=GIB and loan<=residual
                wait_until(lambda:len(pool.completions)>2*tracker.worker_count+1)
                list(stream)
            finally:
                stream.close()
                permit_context.__exit__(None,None,None)
            assert ledger.in_use==charged


@pytest.mark.parametrize('contract,plane',((None,9),(0,None),(0,4_000_000)))
def test_unknown_or_full_group_evidence_bounds_cannot_lend_tile_scratch(contract,plane):
    ledger = _ByteAdmissionPool(20*GIB,'unproven-loan')
    with resources.admit_sam_parent_resources(ledger,GIB,'bound',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        session = resources.cpu_session_bytes(3,9)['estimated_peak_bytes']
        declared = resources.cpu_wave_admission(session,27,2*session+81+64*1024**2,2)
        with ledger.reserve(ledger.capacity-ledger.in_use,'other parents'):
            with resources.admit_sam_tracker_scope(profile,declared,max_seed_pixels=9,
                    max_frame_count=3,job_count=12,evidence_contract_bytes=contract,
                    max_evidence_plane_pixels=plane) as permit:
                assert permit._limits['completion_bank_bytes']==permit._limits['completion_wave_loan_bytes']==0
                assert permit._limits['prepared_bank_bytes']==0


def test_actual_extrapolation_groups_receive_funded_loan_when_the_parent_ledger_is_full(tmp_path):
    from XTA import sam_extrapolation
    from tests.test_sam_extrapolation_scope_admission import _baseline
    masks = _baseline(count=8)
    ledger = _ByteAdmissionPool(20*GIB,'tail-loan')
    with resources.admit_sam_parent_resources(ledger,GIB,'tail',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        prepared = sam_extrapolation.prepare_sam_extrapolation_pass(masks,distance=3,
            walk_back=1,min_radius=3.,crop_mode='whole',resource_profile=profile)
        assert len(prepared.runs)>2
        cache = sam.materialize_interpolation_image_cache(np.zeros(prepared.plan.virtual_shape_tyx,np.uint8),
            path=tmp_path/'tail-gray.bin',physical_view_id='tail',source_identity='tail-loan')
        with ledger.reserve(ledger.capacity-ledger.in_use,'other funded parents'):
            with resources.admit_sam_prepared_scope(profile,prepared,cache,max_in_flight=2) as permit:
                limits = permit._limits
                expected = max((group.context_bbox_yx[2]-group.context_bbox_yx[0])*
                    (group.context_bbox_yx[3]-group.context_bbox_yx[1]) for group in prepared.groups)
                assert limits['maximum_evidence_plane_pixels']==expected
                assert limits['completion_bank_bytes']==limits['completion_wave_loan_bytes']>0
                assert limits['prepared_bank_bytes']==0


@pytest.mark.parametrize('background',('larger','unknown','unknown_bbox'))
def test_all_evidence_groups_protect_loan_even_without_tracker_jobs(tmp_path,background):
    with production(tmp_path) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        exemplar = prepared.groups[0]
        if background=='unknown':
            extra = SimpleNamespace(group_id='untracked',status='skipped',
                context_bbox_yx=(0,0,25,85),crop_contract=exemplar.crop_contract)
        else:
            extra = replace(exemplar,group_id='untracked',status='skipped',
                context_bbox_yx=(0,0,25,85) if background=='larger' else (0,0,True,85))
        changed = replace(prepared,plan=replace(prepared.plan,groups=prepared.groups+(extra,)))
        with ledger.reserve(ledger.capacity-ledger.in_use,'other funded parents'):
            with resources.admit_sam_prepared_scope(profile,changed,cache,max_in_flight=2) as permit:
                if background=='larger':
                    assert permit._limits['maximum_evidence_plane_pixels']==25*85
                    assert permit._limits['completion_wave_loan_bytes']>0
                else:
                    assert permit._limits['maximum_evidence_plane_pixels'] is None
                    assert permit._limits['completion_wave_loan_bytes']==0


@pytest.mark.parametrize('fresh',(0,17*1024**2+113,GIB))
def test_mixed_fresh_and_loan_bank_has_one_cap_exact_refund_and_disjoint_packing(fresh):
    ledger = _ByteAdmissionPool(20*GIB,'mixed-bank')
    with resources.admit_sam_parent_resources(ledger,GIB,'mixed',worker_count=2,
            base_allowance_bytes=GIB,headroom_probe=lambda:64*GIB) as profile:
        session = resources.cpu_session_bytes(10,100_000)['estimated_peak_bytes']
        wave = resources.cpu_wave_admission(session,1_000_000,profile.assigned_cpu_wave_bytes,2)
        with ledger.reserve(ledger.capacity-ledger.in_use-fresh,'other funded owners'):
            before = ledger.in_use
            with resources.admit_sam_tracker_scope(profile,wave,max_seed_pixels=100_000,
                    max_frame_count=10,job_count=10_000,evidence_contract_bytes=128*1024**2,
                    evidence_persistent_bytes=64*1024**2,max_evidence_plane_pixels=2_000_000) as permit:
                limits = permit._limits
                fresh_q,loan = limits['completion_fresh_bytes'],limits['completion_wave_loan_bytes']
                assert limits['completion_bank_bytes']==fresh_q+loan
                assert fresh_q==min(fresh,limits['completion_bank_bytes'])
                assert limits['lookahead_jobs']*limits['per_prepared_job_bytes']+fresh_q+loan<=GIB
                assert ledger.in_use-before==limits['prepared_bank_bytes']==fresh_q+limits['lookahead_jobs']*limits['per_prepared_job_bytes']
                assert limits['completion_bank_jobs']+limits['lookahead_jobs']<=9_998
                permit.acquire_scope()
                packing = permit.admit_mask_packing()
                try:
                    residual = resources._sam_evidence_residual_bytes(limits['unused_attempt_wave_bytes'],
                        limits['evidence_fixed_bytes'],128*1024**2,2_000_000)
                    assert packing.scratch_bytes(100_000)==residual-loan
                finally:
                    packing.close()
                    permit.release_scope()
            assert ledger.in_use==before


def test_delegation_is_source_bound_thread_bound_and_cannot_copy_parent_permission(tmp_path):
    with production(tmp_path) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        groups = {str(group.group_id):group for group in prepared.groups}
        by_id = {str(item.observation_id):item for item in prepared.plan.observations}
        with resources.admit_sam_prepared_scope(profile,prepared,cache,max_in_flight=2) as permit:
            source = interpolation._prepared_tracker_source(prepared,prepared.runs,groups,by_id,None,profile,permit)
            permit.acquire_scope()
            delegation = permit.delegate_prepared_source(source)
            original = threading.get_ident()
            def producer():
                assert threading.get_ident()!=original
                with pytest.raises(RuntimeError,match='thread'):
                    resources.validate_live_sam_resource_profile(profile)
                with pytest.raises(RuntimeError,match='expired|thread'):
                    copy.copy(delegation).bind()
                delegation.bind()
                try:
                    request = source.request(0)
                    delegation.validate_request(request,0)
                    altered = dict(request,run_id='unapproved-job')
                    with pytest.raises(interpolation.SamInterpolationInfrastructureError,match='sealed'):
                        delegation.validate_request(altered,0)
                    with pytest.raises(RuntimeError,match='original parent profile'):
                        delegation.validate_request(dict(request,resource_profile=copy.copy(profile)),0)
                finally:
                    delegation.finish()
            with ThreadPoolExecutor(1) as worker:
                worker.submit(producer).result(5)
                with pytest.raises(RuntimeError,match='replayed'):
                    worker.submit(delegation.bind).result(5)
            permit.release_scope()


@pytest.mark.parametrize('proof',(True,False))
def test_funded_completion_bank_or_unavailable_proof_applies_exact_backpressure(tmp_path,proof):
    with production(tmp_path,proof=proof,bank_jobs=5) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        permit_context,permit,stream = source_stream(prepared,cache,profile,tracker)
        try:
            first_index,first = next(stream)
            limits = resources.validate_sam_tracker_scope_admission(permit)
            base = limits['max_in_flight']+limits['lookahead_jobs']
            q = limits['completion_bank_jobs']
            wait_until(lambda:tracker.snapshot()['scope_consumers'][0]['producer_state'] in
                {'completion_bank_full','missing_cpu_retirement','base_input_bank_full'})
            wait_until(lambda:not tracker.snapshot()['running_jobs'] and not tracker.snapshot()['ready_jobs'])
            assert len(pool.submissions)==base+(q if proof else 0)
            row = tracker.snapshot()['scope_consumers'][0]
            assert row['held']
            assert row['completion_bank_jobs']==(q if proof else 0)
            assert row['missing_cpu_retirement_jobs']==(0 if proof else base-1)
            assert row['base_input_jobs']==base
            rest = list(stream)
            actual = [first_index]+[index for index,_result in rest]
            assert actual==[task.payload['input_index'] for task in pool.completions]
            assert sorted(actual)==list(range(len(prepared.runs)))
            assert not tracker._scopes
            assert tracker.release_source_cache(cache)['completed_runs']==16
        finally:
            stream.close()
            permit_context.__exit__(None,None,None)


def test_unfunded_source_keeps_original_owner_adapter(tmp_path):
    with production(tmp_path,physical=False) as (_masks,ledger,profile,prepared,cache,pool,tracker):
        permit_context,permit,stream = source_stream(prepared,cache,profile,tracker)
        try:
            next(stream)
            row = tracker.snapshot()['scope_consumers'][0]
            assert not row['independent_preparation'] and not row['producer_alive']
            assert row['completion_bank_capacity']==0
            list(stream)
        finally:
            stream.close()
            permit_context.__exit__(None,None,None)


def test_tiled_shared_caller_uses_sealed_producer_without_advancing_cohort_assembly(tmp_path):
    with production(tmp_path,mode='tiled') as (_masks,ledger,profile,prepared,cache,pool,tracker):
        pool.first = False
        groups = {str(group.group_id):group for group in prepared.groups}
        by_id = {str(item.observation_id):item for item in prepared.plan.observations}
        order = prepared.execution_order(1)
        active_scopes,seen = [],[]
        stream = interpolation._iterate_tiled_tracker_results(tracker,prepared,1,groups,by_id,
            cache,None,profile)
        try:
            for index,result in stream:
                row = tracker.snapshot()['scope_consumers'][0]
                active_scopes.append(row['scope_token'])
                assert row['independent_preparation']
                item = prepared.tracker_jobs[order[index]]
                assert result.receipt['run_id']==item.run_id
                assert result.receipt['request_metadata']['parent_run_id']==item.original_run.run_id
                seen.append(index)
        finally:
            stream.close()
        assert sorted(seen)==list(range(len(prepared.tracker_jobs)))
        assert len(set(active_scopes))==len(prepared.execution_batches(1))
        assert not tracker._scopes and tracker.release_source_cache(cache)['completed_runs']==len(seen)
