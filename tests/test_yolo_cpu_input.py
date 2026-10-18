"""Native-t input parity and bounded split/retirement ownership."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import queue
import os
import threading
import time
import unittest
from unittest import mock

import numpy as np

from XTA._deps import cv2
from XTA.config import DEFAULT_CHANNEL_FORMAT
from XTA.geometry import AffineSpec, AugJob, ViewInfo, render_fullframe_frame_for_job
from XTA.yolo_cpu_input import CpuInputPrefetch, NativeCartesianRenderer, _t_map, cpu_input_eligible
from tests.test_tta_scheduler_boundary import _scheduler, _state
from tests.test_memfd_source_reuse import _LocalDupFd, _owner


def task(axis='transverse', *, task_id=0, start=0, count=3, tilted=False, direction='vertical'):
    t, h, w = 7, 9, 10
    nh, nw, slices = {'transverse':(h,w,t), 'sagittal':(t,w,h), 'coronal':(t,h,w)}[axis]
    view = ViewInfo(name=f'{axis}-{tilted}-{direction}', physical_view_name=axis,
        num_slices=slices, src_h=nh, src_w=nw, pad_mode='pad', family='tilted' if tilted else 'orthogonal',
        full_t=t, full_h=h, full_w=w, tilt_base_view=axis, tilt_angle_deg=17 if tilted else 0,
        tilt_direction=direction if tilted else '', tilt_frame_stop=slices)
    matrix = np.asarray([[1,0,0],[0,1,0]], np.float32)
    affine = AffineSpec(view.name,0,nw,nh,12,12,12,12,0,0,matrix,matrix,matrix,matrix)
    job = AugJob('a0',0,Path('unused.json'),affine)
    return dict(task_id=task_id, view=view, job=job, job_id='a0', model_name='model',
        kind='fullframe', prediction_batch=1, channel_format=DEFAULT_CHANNEL_FORMAT,
        slice_start=start, slice_count=count, out_size=12, result_mode='file',
        gpu_eligible=True, processing_shape=(slices,nh,nw))


class ManualExecutor:
    def __init__(self):
        self.jobs=[]
        self.owner=None

    def submit(self, fn, *args):
        future=Future()
        self.jobs.append((future,fn,args))
        return future

    def run(self, index=0):
        future,fn,args=self.jobs[index]
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args))
            except BaseException as exc:
                future.set_exception(exc)
        if self.owner is not None:
            with self.owner.condition:
                if not self.owner.condition.wait_for(
                        lambda:not self.owner.pump_scheduled and not self.owner.pump_requested,timeout=3):
                    raise AssertionError('CPU input controller did not settle')


class YoloCpuInputTests(unittest.TestCase):
    def setUp(self):
        self.volume=np.random.default_rng(45).integers(0,256,(5,9,10),dtype=np.uint8)

    def test_native_t_all_cartesian_and_tilted_axes_match_cpu_cube_reference(self):
        lo,hi,alpha=_t_map(5,7)
        first=self.volume[lo].astype(np.float32)
        cube=np.rint(first+alpha[:,None,None]*(self.volume[hi].astype(np.float32)-first)).astype(np.uint8)
        for axis in ('transverse','sagittal','coronal'):
            for tilted in (False,True):
                for direction in ('vertical','horizontal'):
                    descriptor=task(axis,tilted=tilted,direction=direction)
                    render=NativeCartesianRenderer(self.volume,descriptor['view'],descriptor['job'])
                    for index in (0,2,descriptor['view'].num_slices-1):
                        with self.subTest(axis=axis,tilted=tilted,direction=direction,index=index):
                            expected=render_fullframe_frame_for_job(cube,descriptor['view'],descriptor['job'],index)
                            np.testing.assert_array_equal(render(index),expected)

    def _prefetch(self, tasks, max_bytes=100000):
        executor=ManualExecutor()
        closed=[]
        owner=CpuInputPrefetch(tasks,self.volume,executor,max_bytes=max_bytes,
            headroom=lambda:1000000, allocate=lambda shape,*_a,**_kw:np.empty(shape,np.uint8),
            backing=lambda _array:Path('/proc/fake/shared'), close_array=closed.append)
        executor.owner=owner
        self.addCleanup(owner.shutdown)
        self._idle(owner)
        return owner,executor,closed

    def _idle(self, owner):
        with owner.condition:
            self.assertTrue(owner.condition.wait_for(
                lambda:not owner.pump_scheduled and not owner.pump_requested,timeout=3))

    def test_split_children_keep_shared_bytes_until_both_consumers_complete(self):
        original=task(count=3)
        owner,executor,closed=self._prefetch([original])
        self.assertGreater(owner.bytes,3*12**2)  # Input plus funded CPU scratch.
        executor.run()
        self.assertEqual(owner.bytes,3*12**2)
        first=dict(original,slice_count=1)
        second=dict(original,task_id=7,slice_start=1,slice_count=2)
        owner.attach_ready(first); owner.attach_ready(second)
        self.assertEqual(first['cpu_input_offset'],0)
        self.assertEqual(second['cpu_input_offset'],1)
        owner.retire(first)
        self.assertEqual(owner.bytes,3*12**2)
        self.assertEqual(closed,[])
        owner.retire(second)
        self._idle(owner)
        self.assertEqual(owner.bytes,0)
        self.assertEqual(len(closed),1)
        owner.retire(second)
        self._idle(owner)
        self.assertEqual(len(closed),1)
        owner.shutdown()

    def test_early_gpu_fallback_keeps_builder_credit_until_writer_settles(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        owner.attach_ready(descriptor)
        self.assertNotIn('cpu_input_path',descriptor)
        owner.retire(descriptor)
        self.assertGreater(owner.bytes,0)
        executor.run()
        self.assertEqual(owner.bytes,0)
        self.assertEqual(len(closed),0)  # Already-consumed coverage never allocates an array.
        owner.shutdown()

    def test_unready_source_waits_in_cpu_worker_and_retains_fallback_credit(self):
        entered,ready,settled=threading.Event(),threading.Event(),threading.Event()
        closed=[]
        def wait_source(_volume):
            entered.set()
            if not ready.wait(3):
                raise RuntimeError('test source never became ready')
        def close(array):
            closed.append(array)
            settled.set()
        descriptor=task()
        with ThreadPoolExecutor(max_workers=1) as executor, mock.patch(
                'XTA.yolo_cpu_input.wait_for_volume_ready',side_effect=wait_source):
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=lambda:1000000,allocate=lambda shape,*_a,**_kw:np.empty(shape,np.uint8),
                backing=lambda _array:Path('/proc/fake/shared'),close_array=close)
            self.assertTrue(entered.wait(3))
            self.assertFalse(owner.ready(descriptor))
            self.assertEqual(closed,[])
            owner.retire(descriptor)
            self.assertGreater(owner.bytes,0)
            ready.set()
            with owner.condition:
                self.assertTrue(owner.condition.wait_for(lambda:owner.bytes==0,timeout=3))
            self.assertEqual(len(closed),0)
            owner.shutdown()

    def test_unready_cpu_builder_does_not_delay_actual_gpu_fallback_dispatch(self):
        entered,release=threading.Event(),threading.Event()
        def wait_source(_volume):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test source never released')
        with ThreadPoolExecutor(max_workers=2) as executor, mock.patch(
                'XTA.yolo_cpu_input.wait_for_volume_ready',side_effect=wait_source):
            descriptor=task()
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=lambda:1000000,allocate=lambda shape,*a,**kw:np.empty(shape,np.uint8),
                backing=lambda array:Path('/proc/fake/shared'),close_array=lambda array:None)
            try:
                self.assertTrue(entered.wait(3))
                state=_state()
                state.gpu_task_queues[0]=queue.Queue()
                state.gpu_worker_tasks_by_id[0]=descriptor
                state.gpu_worker_pending_task_ids.append(0)
                scheduler=_scheduler(Path('.'),state=state)
                scheduler.cpu_input_prefetch=owner
                scheduler.dispatch_gpu_worker_inference_window()
                actual=state.gpu_task_queues[0].get_nowait()
                self.assertNotIn('cpu_input_path',actual)
                owner.retire(descriptor)
                self.assertGreater(owner.bytes,0)
            finally:
                release.set()
                owner.shutdown()
            self.assertEqual(owner.bytes,0)

    def test_grant_shrink_after_slow_probe_cannot_submit_unfunded_builder(self):
        entered,release=threading.Event(),threading.Event()
        def headroom():
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test probe never released')
            return 1000000
        allocator=mock.Mock(side_effect=lambda shape,*a,**kw:np.empty(shape,np.uint8))
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner=CpuInputPrefetch([task()],self.volume,executor,max_bytes=100000,headroom=headroom,
                allocate=allocator,backing=lambda array:Path('/proc/fake/shared'),close_array=lambda array:None)
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(owner.set_capacity(0),0)
                release.set()
                with owner.condition:
                    self.assertTrue(owner.condition.wait_for(lambda:not owner.pump_scheduled,timeout=3))
                allocator.assert_not_called()
                self.assertEqual(owner.bytes,0)
                owner.set_capacity(100000)
                with owner.condition:
                    self.assertTrue(owner.condition.wait_for(lambda:owner.ready(task()),timeout=3))
            finally:
                release.set()
                owner.shutdown()

    def test_old_mask_lease_can_borrow_unused_prefill_grant_then_refill_it(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        executor.run()
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=30000,
            direct_union_inference_byte_limit=30000))
        scheduler.cpu_input_prefetch=owner
        scheduler.state.direct_union_postprocess_bytes[('model','old')]=10000
        scheduler.state.direct_union_backing_leases[('model','old')]=SimpleNamespace(phase='postprocess')
        candidate=dict(descriptor,result_mode='direct_union',processing_shape=(7,30,50))
        self.assertEqual(owner.bytes,432)
        self.assertEqual(owner.reserved_bytes,20000)
        self.assertTrue(scheduler.direct_union_task_admissible(candidate))
        self.assertEqual(owner.reserved_bytes,20000)  # Read-only preview.
        self.assertTrue(scheduler._commit_cpu_input_capacity(candidate))
        self.assertEqual(owner.reserved_bytes,9500)
        self.assertEqual(owner.bytes,432)
        scheduler._refresh_cpu_input_capacity()
        self.assertEqual(owner.reserved_bytes,20000)
        owner.shutdown()

    def test_grant_yield_never_refunds_actual_build_or_retiring_bytes(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        charged=owner.bytes
        self.assertEqual(owner.set_capacity(0),charged)
        self.assertEqual(owner.bytes,charged)
        self.assertEqual(owner.reserved_bytes,charged)
        owner.set_capacity(20000)
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=30000,
            direct_union_inference_byte_limit=30000))
        scheduler.cpu_input_prefetch=owner
        scheduler.state.direct_union_postprocess_bytes[('model','old')]=10000
        scheduler.state.direct_union_backing_leases[('model','old')]=SimpleNamespace(phase='postprocess')
        candidate=dict(descriptor,result_mode='direct_union',processing_shape=(7,30,50))
        self.assertFalse(owner.can_yield_capacity(9500))
        self.assertTrue(scheduler.direct_union_task_admissible(candidate))  # Pure reclaim preview.
        self.assertEqual(owner.bytes,charged)
        self.assertEqual(owner.reserved_bytes,20000)
        owner.shutdown()

    def test_impossible_mask_parent_does_not_take_usable_prefill_grant(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        executor.run()
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=30000,
            direct_union_inference_byte_limit=30000))
        scheduler.cpu_input_prefetch=owner
        scheduler.state.direct_union_postprocess_bytes[('model','old')]=10000
        scheduler.state.direct_union_backing_leases[('model','old')]=SimpleNamespace(phase='postprocess')
        candidate=dict(descriptor,result_mode='direct_union',processing_shape=(7,100,100))
        self.assertFalse(scheduler.direct_union_task_admissible(candidate))
        self.assertEqual(owner.reserved_bytes,20000)
        owner.shutdown()

    def test_cached_admission_scans_are_pure_and_chosen_parent_yields_once(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        executor.run()
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=30000,
            direct_union_inference_byte_limit=30000))
        scheduler.cpu_input_prefetch=owner
        scheduler.state.direct_union_postprocess_bytes[('model','old')]=10000
        scheduler.state.direct_union_backing_leases[('model','old')]=SimpleNamespace(phase='postprocess')
        totals=scheduler._direct_union_admission_totals()
        large=dict(descriptor,result_mode='direct_union',processing_shape=(7,30,80))
        small=dict(descriptor,result_mode='direct_union',processing_shape=(7,30,50))
        self.assertTrue(scheduler.direct_union_task_admissible(large,admission_totals=totals))
        self.assertEqual(owner.reserved_bytes,20000)
        self.assertTrue(scheduler.direct_union_task_admissible(small,admission_totals=totals))
        self.assertEqual(owner.reserved_bytes,20000)
        self.assertTrue(scheduler._commit_cpu_input_capacity(large))
        self.assertEqual(owner.reserved_bytes,3200)
        owner.shutdown()

    def test_chosen_dispatch_charge_race_requeues_without_claiming_gpu(self):
        state=_state()
        state.gpu_task_queues[0]=queue.Queue()
        descriptor=dict(task(),result_mode='direct_union',processing_shape=(7,30,50))
        state.gpu_worker_tasks_by_id[0]=descriptor
        state.gpu_worker_pending_task_ids.append(0)
        state.direct_union_postprocess_bytes[('model','old')]=10000
        state.direct_union_backing_leases[('model','old')]=SimpleNamespace(phase='postprocess')
        begin=mock.Mock(return_value=True)
        scheduler=_scheduler(Path('.'),state=state,input_overrides=dict(
            direct_union_total_dense_byte_limit=30000,direct_union_inference_byte_limit=30000),
            operation_overrides=dict(_main_process_gpu_stage_begin_inference=begin))
        entered,release=threading.Event(),threading.Event()
        calls=0
        def headroom():
            nonlocal calls
            calls+=1
            if calls==2:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test refill probe never released')
            return 1000000
        executor=ManualExecutor()
        owner=CpuInputPrefetch([descriptor,dict(descriptor,task_id=1,slice_start=3)],self.volume,
            executor,max_bytes=20000,headroom=headroom,
            allocate=lambda shape,*a,**kw:np.empty(shape,np.uint8),
            backing=lambda array:Path('/proc/fake/shared'),close_array=lambda array:None,
            on_capacity_change=scheduler.notify_gpu_stage_admission_change)
        executor.owner=owner
        self._idle(owner)
        scheduler.cpu_input_prefetch=owner
        callback=threading.Thread(target=lambda:executor.run(0))
        callback.start()
        original=scheduler.pop_gpu_worker_pending_task_id
        def select_then_charge(*args,**kwargs):
            selected=original(*args,**kwargs)
            release.set()
            with owner.condition:
                self.assertTrue(owner.condition.wait_for(lambda:owner.bytes>9500,timeout=3))
            return selected
        try:
            self.assertTrue(entered.wait(3))
            scheduler.pop_gpu_worker_pending_task_id=select_then_charge
            scheduler.dispatch_gpu_worker_inference_window()
            callback.join(3)
            begin.assert_not_called()
            self.assertEqual(list(state.gpu_worker_pending_task_ids),[0])
            self.assertEqual(state.gpu_worker_dispatched_tasks,0)
            self.assertTrue(state.gpu_task_queues[0].empty())
            scheduler.pop_gpu_worker_pending_task_id=original
            scheduler.inputs=replace(scheduler.inputs,ensure_baseline_workspaces=lambda model,view:
                state.baseline_union_paths.setdefault((model,view.name),Path('mask.dat')))
            self._idle(owner)  # Queued speculative writer is cancelled by the controller.
            self.assertTrue(executor.jobs[1][0].cancelled())
            self.assertEqual(owner.bytes,432)  # The chosen input remains ready and charged.
            self.assertTrue(state.gpu_stage_admission_dirty.is_set())
            self.assertTrue(scheduler.service_gpu_stage_admission_retry())
            self.assertEqual(begin.call_count,1)
            self.assertEqual(state.gpu_worker_dispatched_tasks,1)
            actual=state.gpu_task_queues[0].get_nowait()
            self.assertIn('cpu_input_path',actual)
        finally:
            release.set()
            callback.join(3)
            owner.shutdown()

    def test_active_parent_commit_adds_no_new_mask_charge_or_grant_yield(self):
        scheduler=_scheduler(Path('.'))
        descriptor=dict(task(),result_mode='direct_union')
        parent=('model',descriptor['view'].name)
        scheduler.state.direct_union_inference_views.add(parent)
        owner=SimpleNamespace(try_yield_capacity=mock.Mock(),reserved_bytes=20000)
        scheduler.cpu_input_prefetch=owner
        self.assertTrue(scheduler._commit_cpu_input_capacity(descriptor))
        owner.try_yield_capacity.assert_not_called()

    def test_selected_empty_parent_emergency_preserves_charged_prefill_owners(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        charged=owner.bytes
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=80000,
            direct_union_inference_byte_limit=80000))
        scheduler.cpu_input_prefetch=owner
        candidate=dict(descriptor,result_mode='direct_union',processing_shape=(7,100,100))
        self.assertLess(scheduler.direct_union_task_bytes(candidate),80000)
        self.assertGreater(charged+scheduler.direct_union_task_bytes(candidate),80000)
        self.assertTrue(scheduler.direct_union_task_admissible(candidate))
        self.assertTrue(scheduler._commit_cpu_input_capacity(candidate))
        self.assertEqual(owner.bytes,charged)
        self.assertEqual(owner.reserved_bytes,20000)
        owner.shutdown()

    def test_sparse_retirement_disabled_preserves_commit_acceptance(self):
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_sparse_retirement_active=False))
        descriptor=dict(task(),result_mode='direct_union')
        owner=SimpleNamespace(try_yield_capacity=mock.Mock(side_effect=AssertionError('unexpected grant gate')),
            reserved_bytes=20000)
        scheduler.cpu_input_prefetch=owner
        self.assertTrue(scheduler.direct_union_task_admissible(descriptor))
        self.assertTrue(scheduler._commit_cpu_input_capacity(descriptor))
        owner.try_yield_capacity.assert_not_called()

    def test_backlog_is_bounded_and_failed_preparation_falls_back(self):
        first=task(count=3)
        second=task(task_id=1,start=3,count=3)
        owner,executor,closed=self._prefetch([first,second],max_bytes=15000)
        self.assertEqual(len(executor.jobs),1)
        executor.run()
        self.assertLessEqual(owner.bytes,15000)
        owner.retire(first)
        self._idle(owner)
        self.assertEqual(len(executor.jobs),2)
        executor.jobs[1][0].set_exception(OSError('unsupported RAM'))
        self._idle(owner)
        self.assertFalse(owner.ready(second))
        self.assertEqual(owner.bytes,0)
        owner.shutdown()

    def test_preparation_admission_round_robins_views(self):
        descriptors=[task(axis,task_id=number*2+part,start=part*3,count=3)
            for number,axis in enumerate(('transverse','sagittal','coronal')) for part in range(2)]
        owner,executor,closed=self._prefetch(descriptors,max_bytes=50000)
        self.assertEqual([args[0].task['view'].physical_view_name for _,_,args in executor.jobs][:3],
            ['transverse','sagittal','coronal'])
        self.assertLessEqual(owner.bytes,50000)
        owner.shutdown()

    def test_single_executor_worker_builds_and_shuts_down_without_starving_pump(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            descriptor=task()
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=lambda:1000000,allocate=lambda shape,*_a,**_kw:np.empty(shape,np.uint8),
                backing=lambda _array:Path('/proc/fake/shared'),close_array=lambda array:None)
            with owner.condition:
                self.assertTrue(owner.condition.wait_for(lambda:owner.ready(descriptor),timeout=3))
            owner.retire(descriptor)
            owner.shutdown()
            self.assertEqual(owner.bytes,0)

    def test_executor_startup_lock_cannot_block_scheduler_control_or_gpu_dispatch(self):
        entered,release,control_submit=threading.Event(),threading.Event(),threading.Event()
        scheduler_returned,foreign_returned=threading.Event(),threading.Event()
        allocation=mock.Mock(side_effect=lambda shape,*a,**kw:np.empty(shape,np.uint8))
        descriptor=task()
        results={}
        with ThreadPoolExecutor(max_workers=1) as executor, ThreadPoolExecutor(max_workers=1) as other:
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=lambda:0,allocate=allocation,
                backing=lambda array:Path('/proc/fake/shared'),close_array=lambda array:None)
            self._idle(owner)
            def stall_startup():
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test startup lock never released')
            def foreign_submit():
                executor.submit(lambda:None)
                foreign_returned.set()
            original_submit=executor.submit
            def tracked_submit(*args,**kwargs):
                if threading.current_thread() is owner.control_thread:
                    control_submit.set()
                return original_submit(*args,**kwargs)
            def schedule():
                started=time.perf_counter()
                owner.set_capacity(0)
                owner.set_capacity(100000)
                owner.pump()
                self.assertTrue(control_submit.wait(3))
                self.assertFalse(owner.ready(descriptor))
                state=_state()
                state.gpu_task_queues[0]=queue.Queue()
                state.gpu_worker_tasks_by_id[0]=descriptor
                state.gpu_worker_pending_task_ids.append(0)
                scheduler=_scheduler(Path('.'),state=state)
                scheduler.cpu_input_prefetch=owner
                scheduler.dispatch_gpu_worker_inference_window()
                results['dispatch']=state.gpu_task_queues[0].get_nowait()
                owner.retire(descriptor)
                self.assertGreater(owner.bytes,0)  # Submission owns its credit until the writer settles.
                results['wall_seconds']=time.perf_counter()-started
                scheduler_returned.set()
            holder=foreign=main=None
            try:
                with mock.patch.object(other,'_adjust_thread_count',side_effect=stall_startup), \
                        mock.patch.object(executor,'submit',side_effect=tracked_submit):
                    holder=threading.Thread(target=lambda:other.submit(lambda:None))
                    holder.start()
                    self.assertTrue(entered.wait(3))
                    # Negative control: an ordinary submit really shares this lock.
                    foreign=threading.Thread(target=foreign_submit)
                    foreign.start()
                    self.assertFalse(foreign_returned.wait(.05))
                    owner.headroom=lambda:1000000
                    main=threading.Thread(target=schedule)
                    main.start()
                    self.assertTrue(scheduler_returned.wait(.5))
                    self.assertNotIn('cpu_input_path',results['dispatch'])
                    self.assertFalse(foreign_returned.is_set())
            finally:
                release.set()
                for thread in (holder,foreign,main):
                    if thread is not None:
                        thread.join(3)
                other._adjust_thread_count()  # Run work queued by the stalled test hook.
                owner.shutdown()
            self.assertTrue(foreign_returned.is_set())
            self.assertEqual(owner.bytes,0)
            allocation.assert_not_called()  # GPU stole coverage before the queued writer ran.

    def test_busy_metadata_lock_declines_optional_handoff_without_waiting(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        executor.run()
        entered,release=threading.Event(),threading.Event()
        def hold():
            with owner.lock:
                entered.set()
                release.wait(3)
        holder=threading.Thread(target=hold)
        holder.start()
        returned=threading.Event()
        def handoff():
            self.assertFalse(owner.ready(descriptor))
            owner.attach_ready(descriptor)
            returned.set()
        caller=threading.Thread(target=handoff)
        try:
            self.assertTrue(entered.wait(3))
            caller.start()
            self.assertTrue(returned.wait(.5))
            self.assertNotIn('cpu_input_path',descriptor)
            self.assertEqual(closed,[])
        finally:
            release.set()
            holder.join(3)
            caller.join(3)
        self.assertTrue(owner.ready(descriptor))
        owner.attach_ready(descriptor)
        self.assertEqual(descriptor['cpu_input_shape'],(3,12,12))

    def test_shutdown_collects_late_builder_from_blocked_control_submit(self):
        entered,release,stopped=threading.Event(),threading.Event(),threading.Event()
        executor=ManualExecutor()
        original_submit=executor.submit
        def submit(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test submission never released')
            return original_submit(*args)
        executor.submit=submit
        closed=[]
        owner=CpuInputPrefetch([task()],self.volume,executor,max_bytes=100000,
            headroom=lambda:1000000,allocate=lambda shape,*a,**kw:np.empty(shape,np.uint8),
            backing=lambda array:Path('/proc/fake/shared'),close_array=closed.append)
        thread=None
        try:
            self.assertTrue(entered.wait(3))
            self.assertGreater(owner.bytes,432)
            self.assertIsNone(owner.packets[0].future)
            thread=threading.Thread(target=lambda:(owner.shutdown(),stopped.set()))
            thread.start()
            with owner.condition:
                self.assertTrue(owner.condition.wait_for(lambda:owner.closed,timeout=3))
            self.assertFalse(stopped.is_set())
            self.assertGreater(owner.bytes,432)
            release.set()
            self.assertTrue(stopped.wait(3))
            self.assertTrue(executor.jobs[0][0].cancelled())
            self.assertEqual(owner.bytes,0)
            self.assertEqual(closed,[])
            self.assertFalse(owner.control_thread.is_alive())
        finally:
            release.set()
            if thread is not None:
                thread.join(3)
            owner.shutdown()

    def test_failed_submit_refunds_owned_credit_without_an_unrelated_scheduler_wake(self):
        executor=ManualExecutor()
        executor.submit=mock.Mock(side_effect=RuntimeError('executor no longer accepts work'))
        owner=CpuInputPrefetch([task()],self.volume,executor,max_bytes=100000,
            headroom=lambda:1000000,allocate=lambda shape,*a,**kw:np.empty(shape,np.uint8),
            backing=lambda array:Path('/proc/fake/shared'),close_array=lambda array:None)
        try:
            self._idle(owner)
            executor.submit.assert_called_once()
            self.assertEqual(owner.bytes,0)
            self.assertFalse(owner.packets[0].building)
            self.assertFalse(owner.packets[0].retiring)
        finally:
            owner.shutdown()

    def test_busy_metadata_lock_does_not_count_ineligible_tasks_as_cpu_fallback(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        curved=dict(descriptor,view=replace(descriptor['view'],name='radial',family='radial'))
        entered,release=threading.Event(),threading.Event()
        def hold():
            with owner.lock:
                entered.set()
                release.wait(3)
        holder=threading.Thread(target=hold)
        holder.start()
        try:
            self.assertTrue(entered.wait(3))
            with mock.patch.object(owner,'_record') as record:
                owner.attach_ready(curved)
                record.assert_not_called()
        finally:
            release.set()
            holder.join(3)

    def test_handoff_uses_adopted_descriptor_without_backing_lookup_or_writer_lock(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        executor.run()
        telemetry=SimpleNamespace(add=mock.Mock(side_effect=AssertionError('writer lock used')),
            add_scheduler_counter=mock.Mock())
        with mock.patch.object(owner,'backing',side_effect=AssertionError('main backing lookup')), \
                mock.patch('XTA.yolo_cpu_input.runtime_telemetry',return_value=telemetry):
            self.assertTrue(owner.ready(descriptor))
            owner.attach_ready(descriptor)
            telemetry.add_scheduler_counter.assert_called_once_with('yolo.cpu_input.offloaded_frames',3)
        self.assertEqual(descriptor['cpu_input_path'],str(Path('/proc/fake/shared')))

    def test_single_worker_shutdown_cancels_queued_refill_before_extra_allocations(self):
        entered,release=threading.Event(),threading.Event()
        def allocate(shape,*args,**kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test allocation never released')
            return np.empty(shape,np.uint8)
        allocator=mock.Mock(side_effect=allocate)
        with ThreadPoolExecutor(max_workers=1) as executor:
            descriptors=[task(task_id=index,start=index,count=1) for index in range(3)]
            owner=CpuInputPrefetch(descriptors,self.volume,executor,max_bytes=100000,
                headroom=lambda:1000000,allocate=allocator,
                backing=lambda _array:Path('/proc/fake/shared'),close_array=lambda array:None)
            try:
                self.assertTrue(entered.wait(3))
                owner.retire(descriptors[0])  # Queue refill behind the active builder.
                stopped=threading.Event()
                thread=threading.Thread(target=lambda:(owner.shutdown(),stopped.set()))
                thread.start()
                self.assertFalse(stopped.wait(.05))
                release.set()
                self.assertTrue(stopped.wait(3))
                thread.join(3)
                self.assertEqual(allocator.call_count,1)
                self.assertEqual(owner.bytes,0)
            finally:
                release.set()
                owner.shutdown()

    def test_slow_headroom_does_not_block_scheduler_queries_or_gpu_fallback(self):
        entered,release=threading.Event(),threading.Event()
        def headroom():
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test headroom probe never released')
            return 1000000
        with ThreadPoolExecutor(max_workers=2) as executor:
            descriptor=task()
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=headroom,allocate=lambda shape,*_a,**_kw:np.empty(shape,np.uint8),
                backing=lambda _array:Path('/proc/fake/shared'),close_array=lambda array:None)
            try:
                self.assertTrue(entered.wait(3))
                self.assertFalse(owner.ready(descriptor))
                self.assertEqual(owner.reserved_bytes,100000)
                state=_state()
                state.gpu_task_queues[0]=queue.Queue()
                state.gpu_worker_tasks_by_id[0]=descriptor
                state.gpu_worker_pending_task_ids.append(0)
                state.fullframe_remaining[('model',descriptor['view'].name)]=1
                scheduler=_scheduler(Path('.'),state=state)
                scheduler.cpu_input_prefetch=owner
                returned=threading.Event()
                def dispatch():
                    scheduler.dispatch_gpu_worker_inference_window()
                    returned.set()
                thread=threading.Thread(target=dispatch)
                thread.start()
                self.assertTrue(returned.wait(.5))
                actual=state.gpu_task_queues[0].get_nowait()
                self.assertNotIn('cpu_input_path',actual)
                owner.retire(descriptor)
                self.assertEqual(owner.reserved_bytes,0)
            finally:
                release.set()
                owner.shutdown()
                thread.join(3)

    def test_slow_retired_array_close_keeps_credit_without_blocking_queries(self):
        entered,release=threading.Event(),threading.Event()
        def close(array):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test close never released')
        with ThreadPoolExecutor(max_workers=2) as executor:
            descriptor=task()
            owner=CpuInputPrefetch([descriptor],self.volume,executor,max_bytes=100000,
                headroom=lambda:1000000,allocate=lambda shape,*_a,**_kw:np.empty(shape,np.uint8),
                backing=lambda _array:Path('/proc/fake/shared'),close_array=close)
            try:
                with owner.condition:
                    self.assertTrue(owner.condition.wait_for(lambda:owner.ready(descriptor),timeout=3))
                owner.retire(descriptor)
                self.assertTrue(entered.wait(3))
                self.assertEqual(owner.bytes,432)
                self.assertEqual(owner.reserved_bytes,100000)
                self.assertFalse(owner.ready(descriptor))
                self.assertEqual(owner.snapshot()['retiring_packets'],1)
            finally:
                release.set()
                owner.shutdown()
            self.assertEqual(owner.bytes,0)

    def test_empty_parent_emergency_admission_does_not_wait_for_prefill_consumption(self):
        scheduler=_scheduler(Path('.'),input_overrides=dict(direct_union_total_dense_byte_limit=700,
            direct_union_inference_byte_limit=700))
        scheduler.cpu_input_prefetch=SimpleNamespace(reserved_bytes=175,ready=lambda task:False)
        descriptor=dict(task(),result_mode='direct_union',processing_shape=(7,9,10))
        self.assertEqual(scheduler.direct_union_task_bytes(descriptor),630)
        self.assertGreater(scheduler.direct_union_task_bytes(descriptor)+175,700)
        self.assertTrue(scheduler.direct_union_task_admissible(descriptor))

    def test_finished_future_keeps_credit_until_callback_adopts_payload(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        future=executor.jobs[0][0]
        callbacks=future._done_callbacks[:]
        future._done_callbacks.clear()
        executor.run()
        self.assertTrue(future.done())
        owner.retire(descriptor)
        self.assertGreater(owner.bytes,0)
        self.assertEqual(closed,[])
        for callback in callbacks:
            callback(future)
        self._idle(owner)
        self.assertEqual(owner.bytes,0)
        self.assertIsNone(owner.packets[0].future)
        self.assertEqual(len(closed),1)
        owner.shutdown()

    def test_no_disk_or_unsupported_channels(self):
        self.assertTrue(cpu_input_eligible(task()))
        for change in (dict(prediction_batch=2),dict(kind='tile'),dict(sam_gray_cache_spec={"enabled":True}),
                dict(channel_format=SimpleNamespace(kind='rgb',channel_count=3))):
            self.assertFalse(cpu_input_eligible(dict(task(),**change)))

    def _selection(self, *, curved_admissible=True, singleton=False):
        state=_state()
        state.gpu_task_queues[0]=queue.Queue()
        cheap=task()
        curved=dict(task(task_id=1),view=replace(task()['view'],name='radial',family='radial'))
        descriptors=[cheap,curved] if singleton else [cheap,dict(cheap,task_id=2,slice_start=3),
            curved,dict(curved,task_id=3,slice_start=3)]
        for descriptor in descriptors:
            state.gpu_worker_tasks_by_id[descriptor['task_id']]=descriptor
            state.gpu_worker_pending_task_ids.append(descriptor['task_id'])
            parent=('model',descriptor['view'].name)
            state.fullframe_remaining[parent]=state.fullframe_remaining.get(parent,0)+1
        scheduler=_scheduler(Path('.'),state=state,
            operation_overrides=dict(gpu_worker_default_seconds_per_frame=lambda view:10 if view.family=='orthogonal' else .1))
        scheduler.cpu_input_prefetch=SimpleNamespace(bytes=0,reserved_bytes=0,ready=lambda _task:True)
        scheduler.direct_union_task_admissible=lambda descriptor,**_kwargs: curved_admissible or descriptor['view'].family=='orthogonal'
        return scheduler

    def test_curved_first_is_real_scheduler_priority_including_singleton_unlocks(self):
        for singleton in (False,True):
            scheduler=self._selection(singleton=singleton)
            selected=scheduler.pop_gpu_worker_pending_task_id()
            self.assertEqual(scheduler.state.gpu_worker_tasks_by_id[selected[0]]['view'].family,'radial')

    def test_cheap_singleton_does_not_jump_a_curved_multilease_parent(self):
        scheduler=self._selection()
        scheduler.state.gpu_worker_pending_task_ids.remove(2)
        scheduler.state.fullframe_remaining[('model',task()['view'].name)]=1
        selected=scheduler.pop_gpu_worker_pending_task_id()
        self.assertEqual(scheduler.state.gpu_worker_tasks_by_id[selected[0]]['view'].family,'radial')

    def test_curved_tiles_keep_parent_first_storage_policy(self):
        scheduler=self._selection()
        for task_id in (1,3):
            scheduler.state.gpu_worker_tasks_by_id[task_id]['kind']='tile'
        selected=scheduler.pop_gpu_worker_pending_task_id()
        self.assertEqual(scheduler.state.gpu_worker_tasks_by_id[selected[0]]['kind'],'fullframe')

    def test_blocked_curved_work_does_not_idle_gpu_with_ready_cheap_work(self):
        scheduler=self._selection(curved_admissible=False)
        selected=scheduler.pop_gpu_worker_pending_task_id()
        self.assertEqual(scheduler.state.gpu_worker_tasks_by_id[selected[0]]['view'].family,'orthogonal')

    def test_mask_admission_cannot_borrow_callback_refill_credit(self):
        descriptor=task(count=3)
        owner,executor,closed=self._prefetch([descriptor],max_bytes=20000)
        scheduler=self._selection()
        scheduler.cpu_input_prefetch=owner
        self.assertEqual(scheduler._direct_union_admission_totals()[2],20000)
        executor.run()
        self.assertEqual(owner.bytes,432)
        self.assertEqual(scheduler._direct_union_admission_totals()[2],20000)
        owner.retire(descriptor)
        self._idle(owner)
        self.assertEqual(scheduler._direct_union_admission_totals()[2],0)
        owner.shutdown()

    def test_d1_group_waits_for_curved_then_resumes_or_uses_blocked_curved_slack(self):
        for curved_admissible in (True,False):
            scheduler=self._selection(curved_admissible=curved_admissible)
            scheduler.inputs=replace(scheduler.inputs,v1613_d1_owner_active=True)
            scheduler.d1_owner_group_mode_active=lambda:True
            scheduler.d1_owner_group_size_limit=lambda:2
            scheduler.state.gpu_task_queues[1]=queue.Queue()
            parent=('model',task()['view'].name)
            scheduler.state.fullframe_task_ids_by_parent[parent]=[0,2]
            for task_id in (0,2):
                scheduler.state.gpu_worker_tasks_by_id[task_id]['result_mode']='d1_owner'
            group=object()
            scheduler.ensure_d1_parent_group=mock.Mock(return_value=group)
            if curved_admissible:
                self.assertIsNone(scheduler.plan_next_d1_parent_group())
                scheduler.ensure_d1_parent_group.assert_not_called()
                scheduler.state.gpu_worker_pending_task_ids.remove(1)
                scheduler.state.gpu_worker_pending_task_ids.remove(3)
            self.assertIs(scheduler.plan_next_d1_parent_group(),group)
            scheduler.ensure_d1_parent_group.assert_called_once()

    def test_actual_dispatch_handoff_and_compute_release_retire_prepared_owner(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        executor.run()
        state=_state()
        state.gpu_task_queues[0]=queue.Queue()
        state.gpu_worker_tasks_by_id[0]=descriptor
        state.gpu_worker_pending_task_ids.append(0)
        state.gpu_worker_total_tasks=1
        state.fullframe_remaining[('model',descriptor['view'].name)]=1
        scheduler=_scheduler(Path('.'),state=state)
        scheduler.cpu_input_prefetch=owner
        scheduler.dispatch_gpu_worker_inference_window()
        actual=state.gpu_task_queues[0].get_nowait()
        self.assertEqual(actual['cpu_input_shape'],(3,12,12))
        self.assertEqual(owner.bytes,432)
        self.assertEqual(closed,[])
        scheduler.process_one_worker_result(dict(type='compute_released',task_id=0,gpu_index=0,ok=True,
            stats=dict(worker_compute_seconds=.1,slice_count=3)),defer_refill=True)
        self._idle(owner)
        self.assertEqual(owner.bytes,0)
        self.assertEqual(len(closed),1)
        owner.shutdown()

    def test_prepared_memfd_transfer_is_transient_and_rolls_back(self):
        from XTA import runtime
        with _owner('cpu-input-test',b'prepared frame') as original, mock.patch.object(
                runtime.mp_reduction,'DupFd',_LocalDupFd,create=True):
            descriptor=dict(task_id=0,cpu_input_path='cpu-input-test')
            batch=runtime._attach_memfd_transfers_to_task(descriptor)
            duplicate=descriptor['cpu_input_fd'].original_duplicate
            transient=runtime._materialize_worker_task_memfd_paths(descriptor,{})
            self.assertEqual(transient,[duplicate])
            self.assertEqual(os.fstat(duplicate).st_size,len(b'prepared frame'))
            runtime._close_fd_list(transient)
            with self.assertRaises(OSError):
                os.fstat(duplicate)
            self.assertEqual(os.fstat(original).st_size,len(b'prepared frame'))
            failed=dict(task_id=1,cpu_input_path='cpu-input-test')
            batch=runtime._attach_memfd_transfers_to_task(failed)
            duplicate=failed['cpu_input_fd'].original_duplicate
            batch.rollback()
            with self.assertRaises(OSError):
                os.fstat(duplicate)

    def test_failed_compute_release_cannot_retire_input_or_mutate_credits(self):
        descriptor=task()
        owner,executor,closed=self._prefetch([descriptor])
        executor.run()
        state=_state()
        state.gpu_worker_tasks_by_id[0]=descriptor
        state.gpu_worker_dispatched_by_id[0]=1
        state.gpu_worker_task_predicted_seconds_by_id[0]=.3
        state.gpu_worker_predicted_load_by_id[0]=.3
        scheduler=_scheduler(Path('.'),state=state)
        scheduler.cpu_input_prefetch=owner
        with self.assertRaisesRegex(RuntimeError,'compute credit failed'):
            scheduler.process_one_worker_result(dict(type='compute_released',task_id=0,gpu_index=0,
                ok=False,error='uncertain H2D fence'),defer_refill=True)
        self.assertEqual(state.gpu_worker_compute_released_task_ids,set())
        self.assertEqual(state.gpu_worker_compute_completed_by_id,{})
        self.assertEqual(state.gpu_worker_task_predicted_seconds_by_id,{0:.3})
        self.assertEqual(state.gpu_worker_predicted_load_by_id,{0:.3})
        self.assertEqual(owner.bytes,432)
        self.assertEqual(closed,[])
        owner.shutdown()

    def test_uncertain_copy_and_final_fence_quarantine_slots_and_refuse_replay(self):
        from XTA import cuda_backend
        from XTA.backprojection import _ResidentTensorRTRingFatalError
        for phase in ('reuse_fence','copy_submit','final_fence'):
            source=cuda_backend.CpuPreparedYoloSource.__new__(cuda_backend.CpuPreparedYoloSource)
            source.prepared_frames=np.zeros((1,12,12),np.uint8)
            source._copy_failed=False
            source.copy_h2d_seconds=source.copy_wait_seconds=source.copy_submit_seconds=0.
            source.copy_bytes=0
            source.out_size=12
            slot=SimpleNamespace(_cpu_input_copy_valid=phase!='copy_submit',infer_valid=False,
                _cpu_input_copy_done=mock.Mock(),_cpu_input_copy_start=mock.Mock(),
                _cpu_input_pinned=SimpleNamespace(numpy=lambda:np.zeros((12,12),np.uint8)),
                _cpu_input_device=mock.Mock(),input=mock.Mock(),_cpu_input_read_valid=False)
            source._cpu_input_slots=[slot]
            binding=slot.input
            source._cpu_input_cuda_owners=[binding,slot._cpu_input_device,slot._cpu_input_pinned]
            source._direct_ring=None  # TRT release may already have cleared its borrowed source reference.
            source.engine=SimpleNamespace(_stream=object(),
                torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _stream:nullcontext())))
            source._cpu_input_copy_stream=mock.Mock()
            if phase=='copy_submit':
                slot._cpu_input_device.copy_.side_effect=RuntimeError('copy launch uncertain')
            else:
                slot._cpu_input_copy_done.synchronize.side_effect=RuntimeError('fence failed')
            quarantined=[]
            with mock.patch.object(cuda_backend,'_CPU_INPUT_UNCERTAIN_SOURCES',quarantined):
                with self.assertRaises(_ResidentTensorRTRingFatalError):
                    if phase=='final_fence':
                        source.finish_cpu_copies()
                    else:
                        source._copy_frame(0,slot)
                self.assertEqual(quarantined,[source])
                self.assertIs(source._cpu_input_slots[0],slot)
                slot.input=None  # Executor.close clears slot fields after an unsafe launch.
                self.assertIs(source._cpu_input_cuda_owners[0],binding)
                self.assertIsNotNone(source.prepared_frames)
                with self.assertRaises(_ResidentTensorRTRingFatalError):
                    source.close()
                self.assertEqual(quarantined,[source])

    def test_worker_finally_propagates_uncertain_cpu_input_close(self):
        from XTA import workers
        from XTA.cuda_backend import CpuPreparedYoloSource
        from XTA.backprojection import _ResidentTensorRTRingFatalError
        class FencedSource(CpuPreparedYoloSource):
            instance=None
            def __init__(self,*args,**kwargs):
                self.copy_bytes=self.copy_submit_seconds=self.copy_wait_seconds=self.copy_h2d_seconds=0
                self.calls=0
                FencedSource.instance=self
            def finish_cpu_copies(self):
                self.calls+=1
                if self.calls>1:
                    raise _ResidentTensorRTRingFatalError('uncertain final CPU input fence')
        descriptor=task()
        descriptor.update(cpu_input_path='mock-input',cpu_input_shape=(3,12,12),cpu_input_offset=0,
            result_mask_path='mock-mask',result_conf_path=None,source_volume_path='mock-source',
            source_shape=(5,9,10),M_out_to_processing=descriptor['job'].aff.M_out_to_src)
        engine=SimpleNamespace(ensure_volume=lambda *args,**kwargs:'resident')
        with mock.patch.object(workers,'CpuPreparedYoloSource',FencedSource), mock.patch.object(
                workers,'_worker_gpu_render_engine',return_value=engine), mock.patch.object(
                workers,'_canonical_raster_plan_for_task',return_value=None), mock.patch.object(
                workers,'_canonical_image_sink_for_task',return_value=None), mock.patch.object(
                workers,'open_existing_gray_memmap',return_value=np.zeros((3,12,12),np.uint8)), mock.patch.object(
                workers.np,'memmap',side_effect=lambda *a,**kw:np.zeros(kw['shape'],np.uint8)), mock.patch.object(
                workers.os,'stat',return_value=SimpleNamespace(st_size=432)), mock.patch.object(
                workers,'predict_source_and_accumulate',return_value={}):
            with self.assertRaisesRegex(_ResidentTensorRTRingFatalError,'uncertain final CPU input fence'):
                workers.run_prediction_volume_in_worker(None,SimpleNamespace(input_channels=1,batch=1,
                    quantize=None),descriptor)
        self.assertEqual(FencedSource.instance.calls,2)


if __name__=='__main__':
    unittest.main()
