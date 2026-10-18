"""CPU behavior oracles: real scheduler overlap and unchanged retry science."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import MappingProxyType

import numpy as np
import pytest

from XTA import geometry, sam_extrapolation as extrapolation, sam_resources as resources
from XTA.config import resolve_tilted_view_groups
from XTA.interpolation import _ByteAdmissionPool
from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_multiscope_scheduler import FourSlotPool
from tests.test_sam_retry_group_execution import _case, _science

GIB = 1024**3


@contextmanager
def selected_attempt(root, *, legacy=False, tilted=False):
    pool = _ByteAdmissionPool(4*GIB if legacy else 64*GIB, 'retry wave oracle')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'selected-attempt',
            worker_count=2, execution_slots=2, base_allowance_bytes=4*GIB,
            headroom_probe=lambda: 128*GIB) as profile:
        view = None
        if tilted:
            # The production tilted Azimuthal sampler supplies a non-square
            # canvas; crossing its half-turn seam mirrors original seed U.
            view = next(view for view in geometry.get_view_infos(128, 384, 512,
                cartesian_views=(), tilt_groups=resolve_tilted_view_groups(['transverse:20:vertical']),
                azimuthal_views=('tilted_transverse',), azimuthal_azimuth_angles=(22.5,))
                if view.family == 'azimuthal' and view.tilt_angle_deg > 0)
            assert view.azimuthal_tilted_source and view.tilt_direction == 'vertical'
            observed = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
            observed[0:3, 60:67, 30:37] = 1
            eligible = lambda terminal, direction: terminal.native_frame_index == 0 and direction == -1
        else:
            observed = np.zeros((8, 128, 128), np.uint8)
            observed[2:5, 60:67, 60:67] = 1
            eligible = lambda terminal, direction: terminal.frame_index == 4 and direction == 1
        prepared = extrapolation.prepare_sam_extrapolation_pass(observed,
            distance=3, walk_back=1, min_radius=3., crop_mode='whole', resource_profile=profile,
            view=view, wrap_axis=tilted, eligible_terminals=eligible)
        assert len(prepared.groups) == 1 and len(prepared.runs) == 2
        wave = prepared.cpu_wave_admission
        serial = resources.cpu_wave_admission(wave['maximum_session_cpu_estimate_bytes'],
            wave['maximum_raw_mask_bytes'], profile.assigned_cpu_wave_bytes, 1)
        prepared = replace(prepared, cpu_wave_admission=MappingProxyType(serial))
        yield pool, profile, prepared
    assert pool.in_use == 0


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', 'YOLO_TTA_GPU_INTERPOLATION'):
        monkeypatch.setenv(flag, '0')


def execute_selected(root, scenario, state, *, fault=None, tilted=False):
    tracker = state['tracker']
    stream = None
    try:
        with selected_attempt(root, legacy=scenario == 'legacy', tilted=tilted) as (pool, profile, prepared):
            state.update(account=pool, profile=profile, prepared=prepared, original_charge=pool.in_use)
            hard_cap = (2*prepared.cpu_wave_admission['maximum_session_cpu_estimate_bytes']-1
                        if scenario == 'hard_cap' else 4*GIB)
            wave = resources.promote_sam_retry_wave(profile, prepared,
                hard_memory_limit_bytes=hard_cap, max_in_flight=1 if scenario == 'slot_cap' else 2)
            prepared = replace(prepared, cpu_wave_admission=wave)
            state['promoted'] = prepared
            cache = materialize_interpolation_image_cache(np.zeros(prepared.plan.virtual_shape_tyx, np.uint8),
                path=root/'images.bin', physical_view_id=('tilted-azimuthal' if tilted else 'transverse'),
                source_identity='fixed-selected-source')
            tracker.source_cache_ref = cache
            state['worker_pool'].gates.update({run.run_id: threading.Event() for run in prepared.runs})
            state['ready'].set()
            records, rows = [], []
            try:
                stream = extrapolation._iterate_extrapolation_tracker_results(
                    tracker, prepared, int(wave['max_in_flight']),
                    {group.group_id: group for group in prepared.groups}, prepared.plan.by_id,
                    cache, tracker._cancel, profile, records, family_dispatch_enabled=False)
                for index, result in stream:
                    rows.append((index, result.receipt.copy(),
                                 {frame: mask.copy() for frame, mask in result.frames.items()}))
                    tracker.release_result(result)
                    if fault == 'consumer':
                        state['primary'] = OSError('controlled retry consumer failure')
                        raise state['primary']
                return rows
            finally:
                if stream is not None:
                    stream.close()
                tracker.close()
                state['after_scope_charge'] = pool.in_use
                state['after_scope_holds'] = profile._lease.scope_holds
                state['after_scope_identity'] = profile._lease.tracker_scope_identity
    finally:
        if 'account' in state:
            state['final_charge'] = state['account'].in_use


def make_state(root):
    pool = FourSlotPool()
    tracker = SamInterpolationTracker(model_path='cpu-oracle', device_ids=(0, 1),
        artifact_root=root/'tracker')
    tracker._pool = pool
    tracker.start = lambda: tracker
    return dict(worker_pool=pool, tracker=tracker, ready=threading.Event())


def await_ready(state, future):
    if not state['ready'].wait(5):
        if future.done():
            future.result()
        pytest.fail('selected retry did not reach the scheduler')


def await_pool(pool, predicate, future):
    pool.until(lambda: predicate() or future.done(), timeout=10)
    if not predicate():
        future.result()
        pytest.fail('scheduler finished before the controlled worker barrier')


def test_tilted_half_turn_retry_funds_actual_virtual_crops_and_preserves_seeds(tmp_path):
    state = make_state(tmp_path)
    pool = state['worker_pool']
    try:
        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(execute_selected, tmp_path, 'funded', state, tilted=True)
            try:
                await_ready(state, future)
                await_pool(pool, lambda: len(pool.active) == 2, future)
                original, prepared = state['prepared'], state['promoted']
                assert prepared.plan.virtual_shape_tyx != prepared.native_shape
                assert prepared.plan.native_shape_tyx == prepared.native_shape
                assert prepared.plan is original.plan and prepared.runs is original.runs
                assert prepared.frame_crop_bounds is original.frame_crop_bounds
                group = prepared.groups[0]
                y0, x0, y1, x1 = group.context_bbox_yx
                assert x0 > 0 and x1 == prepared.native_shape[2]
                assert y1-y0 != x1-x0  # Native-shape substitution cannot pass this case.
                pixels = (y1-y0)*(x1-x0)
                wave = prepared.cpu_wave_admission
                assert wave['retry_image_bytes'] == sum((b[2]-b[0])*(b[3]-b[1])
                    for b in prepared.frame_crop_bounds.values())
                assert wave['maximum_raw_mask_bytes'] == max(
                    len(run.expected_frames)*pixels for run in prepared.runs)
                assert wave['maximum_session_cpu_estimate_bytes'] == max(
                    resources.cpu_session_bytes(len(run.expected_frames), pixels)['estimated_peak_bytes']
                    for run in prepared.runs)
                assert wave['retry_persistent_bytes'] == 32*1024**2 + 16*pixels
                assert wave['retry_execution_base_peak_bytes'] <= wave['retry_execution_memory_limit_bytes']
                for run in prepared.runs:
                    seed = prepared.plan.by_id[run.seed_ids[0]]
                    assert seed.frame_index >= prepared.native_shape[0] and seed.mirror_u
                    assert seed.native_frame_index < prepared.native_shape[0]
                    assert seed.bbox_yx == (60, prepared.native_shape[2]-37,
                                           67, prepared.native_shape[2]-30)
                # Finish the later submission first while the other job stays
                # gated, proving two real production scheduler slots overlap.
                second = pool.submissions[1][1].payload['run_id']
                with pool.condition:
                    pool.gates[second].set()
                    pool.condition.notify_all()
                await_pool(pool, lambda: len(pool.completions) == 1, future)
                assert pool.completions[0][1].payload['run_id'] == second
                assert len(pool.active) == 1 and not future.done()
            finally:
                pool.unblock()
            rows = future.result(10)
        assert len(rows) == len(pool.submissions) == len(pool.completions) == 2
        for index, receipt, masks in rows:
            run = prepared.runs[index]
            assert receipt['run_id'] == run.run_id
            assert receipt['crop_xyxy'] == [x0, y0, x1, y1]
            assert receipt['seed_frame'] == run.expected_frames[0]
            assert set(masks) == set(run.expected_frames)
            seed_mask = prepared.plan.by_id[run.seed_ids[0]].mask_in_crop(group.context_bbox_yx)
            for mask in masks.values():
                np.testing.assert_array_equal(mask, seed_mask)
        assert state['after_scope_charge'] == state['original_charge']
        assert state['after_scope_holds'] == 0 and state['after_scope_identity'] is None
        assert state['final_charge'] == 0
    finally:
        pool.unblock()
        state['tracker'].close()


@pytest.mark.parametrize('scenario,expected',
    (('funded', 2), ('hard_cap', 1), ('legacy', 1), ('slot_cap', 1)))
def test_selected_attempt_reaches_real_parallel_slots_only_when_funded(tmp_path, scenario, expected):
    state = make_state(tmp_path)
    pool = state['worker_pool']
    try:
        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(execute_selected, tmp_path, scenario, state)
            try:
                await_ready(state, future)
                await_pool(pool, lambda: len(pool.active) == expected, future)
                assert not future.done() and pool.peak_active == expected
                assert state['promoted'].cpu_wave_admission['max_in_flight'] == expected
                if expected == 2:
                    assert state['profile'].has_extra_credit
                    first = pool.submissions[0][1].payload['run_id']
                    second = pool.submissions[1][1].payload['run_id']
                    assert first != second
                    with pool.condition:
                        pool.gates[second].set()
                        pool.condition.notify_all()
                    await_pool(pool, lambda: len(pool.completions) == 1, future)
                    assert pool.completions[0][1].payload['run_id'] == second
                    assert len(pool.active) == 1 and not pool.gates[first].is_set()
                else:
                    assert len(pool.submissions) == 1
            finally:
                pool.unblock()
            rows = future.result(10)
        prepared = state['promoted']
        by_run = {run.run_id: run for run in prepared.runs}
        assert len(rows) == len(pool.submissions) == len(pool.completions) == 2
        assert {receipt['run_id'] for _, receipt, _ in rows} == set(by_run)
        for _index, receipt, masks in rows:
            run = by_run[receipt['run_id']]
            group = prepared.groups[0]
            y0, x0, y1, x1 = group.context_bbox_yx
            seed = prepared.plan.by_id[run.seed_ids[0]].mask_in_crop(group.context_bbox_yx)
            assert receipt['crop_xyxy'] == [x0, y0, x1, y1]
            assert receipt['seed_frame'] == run.expected_frames[0]
            assert set(masks) == set(run.expected_frames)
            for mask in masks.values():
                np.testing.assert_array_equal(mask, seed)
        assert state['after_scope_charge'] == state['original_charge']
        assert state['after_scope_holds'] == 0 and state['after_scope_identity'] is None
        assert state['final_charge'] == 0
    finally:
        pool.unblock()
        state['tracker'].close()


@pytest.mark.parametrize('bad_owner', ('copied', 'metadata', 'foreign', 'expired'))
def test_saved_or_foreign_owner_cannot_promote_a_selected_attempt(tmp_path, bad_owner):
    with selected_attempt(tmp_path) as (pool, profile, prepared):
        before = pool.in_use
        candidate = copy.copy(profile) if bad_owner == 'copied' else profile
        if bad_owner == 'metadata':
            candidate = dict(profile.metadata(), unproved_owner_flags='metadata_transfer_unproved')
        def promote():
            return resources.promote_sam_retry_wave(candidate, prepared,
                hard_memory_limit_bytes=4*GIB, max_in_flight=2)
        if bad_owner == 'expired':
            expired, frozen = profile, prepared
        elif bad_owner == 'foreign':
            with ThreadPoolExecutor(1) as foreign:
                with pytest.raises((RuntimeError, TypeError)):
                    foreign.submit(promote).result(3)
        else:
            with pytest.raises((RuntimeError, TypeError)):
                promote()
        assert pool.in_use == before and profile._lease.scope_holds == 0
    if bad_owner == 'expired':
        with pytest.raises(RuntimeError):
            resources.promote_sam_retry_wave(expired, frozen,
                hard_memory_limit_bytes=4*GIB, max_in_flight=2)
        assert pool.in_use == 0


@pytest.mark.parametrize('fault', ('consumer', 'cancel'))
def test_promoted_scope_releases_credit_after_failure_or_cancellation(tmp_path, fault):
    state = make_state(tmp_path)
    pool = state['worker_pool']
    try:
        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(execute_selected, tmp_path, 'funded', state, fault=fault)
            try:
                await_ready(state, future)
                await_pool(pool, lambda: len(pool.active) == 2, future)
                if fault == 'cancel':
                    state['tracker'].cancel('controlled retry cancellation')
                    pool.unblock()
                else:
                    with pool.condition:
                        pool.gates[pool.submissions[-1][1].payload['run_id']].set()
                        pool.condition.notify_all()
                with pytest.raises(OSError if fault == 'consumer' else RuntimeError) as caught:
                    future.result(10)
                if fault == 'consumer':
                    assert caught.value is state['primary']
            finally:
                pool.unblock()
        assert state['after_scope_charge'] == state['original_charge']
        assert state['after_scope_holds'] == 0 and state['after_scope_identity'] is None
        assert state['final_charge'] == 0 and not state['tracker']._scopes
        state['tracker'].close()
        assert state['final_charge'] == 0
    finally:
        pool.unblock()
        state['tracker'].close()


@pytest.mark.parametrize('accepted_before_error', (False, True))
def test_partial_sdk_submission_failure_settles_all_promoted_owners(tmp_path, accepted_before_error):
    state = make_state(tmp_path)
    pool = state['worker_pool']
    submit = pool.submit
    failure = RuntimeError('controlled second promoted-job submit failure')
    def fail_second(task, *, execution_device_id):
        if pool.submissions:
            if accepted_before_error:
                submit(task, execution_device_id=execution_device_id)
            raise failure
        return submit(task, execution_device_id=execution_device_id)
    pool.submit = fail_second
    try:
        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(execute_selected, tmp_path, 'funded', state)
            try:
                with pytest.raises(RuntimeError) as caught:
                    future.result(15)
                assert caught.value is failure
            finally:
                pool.unblock()
        assert len(pool.submissions) == (2 if accepted_before_error else 1)
        assert state['after_scope_charge'] == state['original_charge']
        assert state['after_scope_holds'] == 0 and state['after_scope_identity'] is None
        assert state['final_charge'] == 0 and not state['tracker']._scopes
    finally:
        pool.unblock()
        state['tracker'].close()


def selection_prefix(stats):
    receipt = json.loads(Path(stats['sam_selection_receipt_path']).read_text())
    return (receipt['selected_run_ids'], {
        run_id: (record['planned_frames'], record['stop_frame'])
        for run_id, record in receipt['run_receipts'].items()})


@pytest.mark.parametrize('mode', ('whole', 'tiled'))
def test_retry_science_matches_serial_with_reversed_completion(tmp_path, monkeypatch, mode):
    promote = resources.promote_sam_retry_wave
    def serial(profile, prepared, *, hard_memory_limit_bytes, max_in_flight):
        return promote(profile, prepared, hard_memory_limit_bytes=hard_memory_limit_bytes,
                       max_in_flight=1)
    monkeypatch.setattr(resources, 'promote_sam_retry_wave', serial)
    with _case(tmp_path/'serial', mode=mode) as control:
        _, stats, components = control['run'](parallel=False)
        expected, prefix = _science(control, components, stats), selection_prefix(stats)
        assert control['providers'] and all(row['wave'] == 1 for row in control['providers'])
    monkeypatch.setattr(resources, 'promote_sam_retry_wave', promote)
    with _case(tmp_path/'reordered', mode=mode) as candidate:
        iterate = candidate['runtime']._iterate
        def reversed_results(indexed, cache, admission):
            yield from iterate(reversed(tuple(indexed)), cache, admission)
        monkeypatch.setattr(candidate['runtime'], '_iterate', reversed_results)
        _, stats, components = candidate['run'](parallel=False)
        actual = _science(candidate, components, stats)
        assert any(row['wave'] > 1 for row in candidate['providers'])
        assert selection_prefix(stats) == prefix
        for left, right in zip(actual, expected):
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right
