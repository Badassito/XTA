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
from tests.test_memfd_source_reuse import _owner, native_dispatch


def task(axis='transverse', *, task_id=0, start=0, count=3, tilted=False, direction='vertical'):
    t, h, w = 7, 9, 10
    nh, nw, slices = {'transverse':(h,w,t), 'sagittal':(t,w,h), 'coronal':(t,h,w)}[axis]
    view = ViewInfo(name=f'{axis}-{tilted}-{direction}', physical_view_name=axis,
        num_slices=slices, src_h=nh, src_w=nw, pad_mode='pad', family='tilted' if tilted else 'orthogonal',
        full_t=t, full_h=h, full_w=w, tilt_base_view=axis, tilt_angle_deg=17 if tilted else 0,
        tilt_direction=direction if tilted else '', tilt_frame_stop=slices)
    matrix = np.asarray([[1,0,0],[0,1,0]], np.float32)
    affine = AffineSpec(view.name,0,nw,nh,12,12,12,12,0,0,matrix,matrix,matrix,matrix)
    job = AugJob('a0',0,affine)
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

    def test_native_transverse_lerp_matches_numpy_for_all_byte_pairs_and_strides(self):
        from XTA.native_cpu import kernels

        grid = np.arange(256, dtype=np.uint8)
        first = np.broadcast_to(grid[None, :], (256, 256))
        second = first.T
        source = np.stack((first, second))
        lo, hi = np.array([0], np.int64), np.array([1], np.int64)
        weights = np.array([0., .17, .5, .7, 1.], np.float32)
        weights = np.concatenate((weights, np.nextafter(weights[1:-1], 0.0), np.nextafter(weights[1:-1], 1.0)))
        for volume in (source, source[:, ::-1, ::-1], source.transpose(0, 2, 1)):
            for weight in weights:
                with self.subTest(strides=volume.strides, weight=weight):
                    a, b = volume.astype(np.float32)
                    expected = np.rint(a + weight * (b - a)).astype(np.uint8)
                    output = np.empty(volume.shape[1:], np.uint8)
                    kernels.render_cartesian_t_lerp_gray8(
                        volume, lo, hi, np.array([weight], np.float32), 0, 0, output)
                    np.testing.assert_array_equal(output, expected)

    def test_cartesian_invalid_time_index_or_aliased_output_does_not_write(self):
        from XTA.native_cpu import kernels

        volume = self.volume.copy()
        output = np.full(volume.shape[1:], 254, np.uint8)
        lo, hi = np.array([0], np.int64), np.array([len(volume)], np.int64)
        alpha = np.array([.5], np.float32)
        with self.assertRaisesRegex(ValueError, "valid slices"):
            kernels.render_cartesian_t_lerp_gray8(volume, lo, hi, alpha, 0, 0, output)
        self.assertTrue((output == 254).all())

        # A bad later row must also be detected before the bulk loop writes.
        sagittal = np.full((2, volume.shape[2]), 254, np.uint8)
        with self.assertRaisesRegex(ValueError, "valid slices"):
            kernels.render_cartesian_t_lerp_gray8(
                volume, np.array([0, 1], np.int64), np.array([1, len(volume)], np.int64),
                np.array([.5, .5], np.float32), 1, 0, sagittal)
        self.assertTrue((sagittal == 254).all())
        with self.assertRaisesRegex(ValueError, "overlap"):
            kernels.render_cartesian_t_lerp_gray8(volume, lo, lo, alpha, 0, 0, volume[0])
        np.testing.assert_array_equal(volume, self.volume)


    def test_coronal_prefetch_preserves_plane_order_across_slab_boundaries(self):
        descriptor = task('coronal', count=10)
        owner, executor, _closed = self._prefetch([descriptor])
        executor.run()
        reference = NativeCartesianRenderer(self.volume, descriptor['view'], descriptor['job'])
        np.testing.assert_array_equal(owner.packets[0].array, np.stack([reference(i) for i in range(10)]))

    def test_shutdown_leaves_unretired_mapping_charged_on_deadline(self):
        descriptor = task(count=1)
        owner, executor, _closed = self._prefetch([descriptor])
        executor.run()
        with mock.patch.object(owner, '_close_packet', return_value=False), \
                mock.patch('XTA.yolo_cpu_input._RETIREMENT_SHUTDOWN_SECONDS', 0):
            with self.assertRaisesRegex(TimeoutError, 'still hold mappings'):
                owner.shutdown()
        self.assertGreater(owner.bytes, 0)
        owner.shutdown()
        self.assertEqual(owner.bytes, 0)

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








    def test_backlog_is_bounded_and_failed_preparation_falls_back(self):
        first=task(count=3)
        second=task(task_id=1,start=3,count=3)
        # Fund one packet, native plane, affine canvas and t-map, with no room for a second.
        owner,executor,closed=self._prefetch([first,second],max_bytes=1000)
        self.assertEqual(len(executor.jobs),1)
        executor.run()
        self.assertLessEqual(owner.bytes,1000)
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









    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux detector descriptor transport')
    def test_prepared_memfd_transfer_is_transient_and_rolls_back(self):
        from XTA import runtime
        from XTA.native_detector_handles import install_detector_handles
        with _owner('cpu-input-test', b'prepared frame') as original:
            task = dict(task_id=0, cpu_input_path='cpu-input-test')
            with native_dispatch(task) as (packet, received):
                duplicate = received[0]
                transient = install_detector_handles(packet, {}, received)
            self.assertEqual(transient, [duplicate])
            self.assertEqual(os.pread(duplicate, 14, 0), b'prepared frame')
            runtime._close_fd_list(transient)
            with self.assertRaises(OSError):
                os.fstat(duplicate)
            with native_dispatch(task) as (packet, received):
                duplicate = received[0]
                packet['_detector_handle_slots']['cpu_input_path'] = -1
                with self.assertRaises(ValueError):
                    install_detector_handles(packet, {}, received)
            with self.assertRaises(OSError):
                os.fstat(duplicate)
            self.assertEqual(os.pread(original, 14, 0), b'prepared frame')


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
