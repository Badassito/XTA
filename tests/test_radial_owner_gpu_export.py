"""Owner completion selects safe GPU-packed, empty, or legacy host exports."""
from __future__ import annotations

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_d1, cylindrical_owner as owner_module
from XTA.cylindrical_bitset_compaction import RadialBitsetCompactionUnavailable
from XTA.cylindrical_cuda_projection import (
    RadialCudaProjectionUnsafeFailure, RadialEncodedBlock, RadialEncodedSlice)


def _packed_export():
    records = (RadialEncodedSlice(0, 1, 2, 1, 2, 1, 0, 1),
               RadialEncodedSlice(1, 0, 0, 0, 0, 0, 1, 0))
    block = RadialEncodedBlock(0, records, np.array([1], np.uint8), packed=True)
    return SimpleNamespace(blocks=(block,), payload_bytes=1, metadata_d2h_bytes=16)


class RadialOwnerGpuExportRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _run(self, *, nonempty=(True,), compaction='1', packed='1',
             preflight_error=None, export_error=None, mutate_on_submit=False):
        view = SimpleNamespace(name='radial_fixture__tta_a0', family='radial',
                               num_slices=2, tta_angle_deg=0.)
        output_shape = (2, 4, 5)
        export = _packed_export()
        events = []
        created = []
        RealOwner = owner_module.RadialOwner

        class FakeOwner:
            proven_empty_output = RealOwner.proven_empty_output

            def __init__(self, view, mask_shape, output_shape):
                self.view = view
                self.mask_shape = tuple(mask_shape)
                self.output_shape = tuple(output_shape)
                self.coverage = np.zeros(view.num_slices, dtype=bool)
                self.native_any = False
                self._failed = self._closed = False
                self.words = SimpleNamespace(nbytes=8)
                self.cleanup_seconds = self.projection_seconds = 0.
                self.created_at = time.perf_counter()
                self.calls = 0
                created.append(self)

            def consume(self, first, masks):
                events.append('consume')
                self.coverage[first:first + len(masks)] = True
                active = bool(nonempty[self.calls])
                self.calls += 1
                self.native_any |= active
                return int(active)

            def host_words(self):
                events.append('host_words')
                return np.array([1, 0], np.uint32)

            def host_packed_blocks(self):
                events.append('gpu_pack')
                if export_error is not None:
                    raise export_error
                return export

            def close(self):
                events.append('close')
                self._closed = True

        future = Future(); future.set_result({'ok': True})
        submitted = []

        def submit(**kwargs):
            events.append('submit')
            self.assertTrue(created[0]._closed)
            submitted.append(kwargs)
            if mutate_on_submit:
                words = kwargs.get('words')
                if words is not None:
                    words.resize((0,), refcheck=False)
                export.payload_bytes = export.metadata_d2h_bytes = 0
            return future

        tasks = []
        for index in range(len(nonempty)):
            first = 0 if len(nonempty) == 1 else index
            count = 2 if len(nonempty) == 1 else 1
            tasks.append(dict(
                projection_contract=owner_module.RADIAL_OWNER_CONTRACT,
                result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                view=view, model_name='fixture', slice_start=first, slice_count=count,
                d1_output_shape=output_shape, d1_store_dir=str(self.root / 'out.cvol')))
        text = io.StringIO()
        states = {}
        with mock.patch.dict(os.environ, {
                    'YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION': compaction,
                    'YOLO_TTA_PACKED_OWNER_PUBLICATION': packed,
                }), mock.patch.object(owner_module, 'RadialOwner', FakeOwner), \
                mock.patch.object(owner_module, '_RADIAL_OWNER_STATES', states), \
                mock.patch.object(owner_module, '_RADIAL_COMPACTION_PREFLIGHT_ERROR', preflight_error), \
                mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                mock.patch.object(cuda_d1, '_d1_submit_publication', side_effect=submit), \
                redirect_stdout(text):
            results = []
            for task in tasks:
                accumulator = SimpleNamespace(
                    union_dev=np.zeros((task['slice_count'], 2, 2), np.uint8),
                    host_written=False)
                results.append(owner_module.consume_radial_device_union(task, accumulator))
        return SimpleNamespace(results=results, owner=created[0], events=events,
                               submitted=submitted, export=export, output=text.getvalue(),
                               states=states, future=future)

    def test_nonempty_selects_gpu_export_and_closes_before_async_submit(self):
        run = self._run(mutate_on_submit=True)
        result = run.results[-1]
        self.assertEqual(run.events, ['consume', 'gpu_pack', 'close', 'submit'])
        self.assertTrue(result['d1_view_complete'])
        self.assertEqual(result['d1_bitset_export_kind'], 'gpu_packed')
        self.assertEqual(result['d1_bitset_host_transfer_bytes'], 17)
        self.assertEqual(result['d1_bitset_words'], (2 * 4 * 5 + 31) // 32)
        self.assertIs(result['_publication_future'], run.future)
        self.assertIsNone(run.submitted[0]['words'])
        self.assertEqual(run.submitted[0]['encoded_blocks'], run.export.blocks)
        self.assertIn('export_kind=gpu_packed', run.output)
        self.assertIn('host_bitset_bytes=0, packed_payload_bytes=1, host_transfer_bytes=17',
                      run.output)

    def test_completion_and_proven_emptiness_are_independent(self):
        empty = self._run(nonempty=(False, False))
        self.assertFalse(empty.results[0]['d1_view_complete'])
        self.assertTrue(empty.results[1]['d1_view_complete'])
        self.assertEqual(empty.events, ['consume', 'consume', 'close', 'submit'])
        self.assertEqual(empty.results[1]['d1_bitset_export_kind'], 'empty')
        self.assertEqual(empty.results[1]['d1_bitset_host_transfer_bytes'], 0)
        self.assertTrue(empty.submitted[0]['proven_empty'])
        self.assertIsNone(empty.submitted[0]['words'])

        retained = self._run(nonempty=(True, False))
        self.assertFalse(retained.results[0]['d1_view_complete'])
        self.assertFalse(retained.results[1]['radial_owner_empty'])
        self.assertEqual(retained.events, ['consume', 'consume', 'gpu_pack', 'close', 'submit'])
        self.assertEqual(retained.results[1]['d1_bitset_export_kind'], 'gpu_packed')

    def test_opt_out_and_raw_cvol_use_legacy_host_words(self):
        for compaction, packed in (('0', '1'), ('1', '0')):
            with self.subTest(compaction=compaction, packed=packed):
                run = self._run(compaction=compaction, packed=packed,
                                mutate_on_submit=True)
                self.assertEqual(run.events, ['consume', 'host_words', 'close', 'submit'])
                self.assertEqual(run.results[-1]['d1_bitset_export_kind'], 'host_bitset')
                self.assertEqual(run.results[-1]['d1_bitset_host_transfer_bytes'], 8)
                self.assertEqual(run.results[-1]['d1_bitset_words'], 2)
                self.assertEqual(run.submitted[0]['words'].size, 0)
                self.assertFalse(run.submitted[0]['proven_empty'])
                self.assertIn('host_bitset_bytes=8, packed_payload_bytes=0, host_transfer_bytes=8',
                              run.output)

    def test_preflight_refusal_and_safe_setup_refusal_fall_back(self):
        preflight = self._run(preflight_error='no safe compaction setup')
        self.assertEqual(preflight.events, ['consume', 'host_words', 'close', 'submit'])
        self.assertEqual(preflight.results[-1]['d1_bitset_export_kind'], 'host_bitset')
        unavailable = self._run(export_error=RadialBitsetCompactionUnavailable('not enough scratch'))
        self.assertEqual(unavailable.events,
                         ['consume', 'gpu_pack', 'host_words', 'close', 'submit'])
        self.assertEqual(unavailable.results[-1]['d1_bitset_export_kind'], 'host_bitset')
        self.assertIn('compaction unavailable', unavailable.output)

    def test_runtime_and_unsafe_export_errors_propagate_without_fallback_or_close(self):
        for error in (RuntimeError('encoding failed'),
                      RadialCudaProjectionUnsafeFailure('stream fence failed', object())):
            with self.subTest(error=type(error).__name__):
                view = SimpleNamespace(name='radial_failed__tta_a0', family='radial',
                                       num_slices=2, tta_angle_deg=0.)
                task = dict(projection_contract=owner_module.RADIAL_OWNER_CONTRACT,
                            result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                            streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                            view=view, model_name='fixture', slice_start=0, slice_count=2,
                            d1_output_shape=(2, 4, 5), d1_store_dir=str(self.root / 'failed.cvol'))
                events = []

                class FailingOwner:
                    proven_empty_output = owner_module.RadialOwner.proven_empty_output

                    def __init__(self, view, mask_shape, output_shape):
                        self.coverage = np.zeros(2, bool)
                        self.native_any = False
                        self._closed = self._failed = False
                        self.words = SimpleNamespace(nbytes=8)
                        self.cleanup_seconds = self.projection_seconds = 0.
                        self.created_at = time.perf_counter()

                    def consume(self, first, masks):
                        self.coverage[:] = True
                        self.native_any = True
                        events.append('consume')
                        return 1

                    def host_packed_blocks(self):
                        events.append('gpu_pack')
                        raise error

                    def host_words(self):
                        events.append('host_words')
                        return np.array([1, 0], np.uint32)

                    def close(self):
                        events.append('close')

                states = {}
                with mock.patch.dict(os.environ, {
                            'YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION': '1',
                            'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}), \
                        mock.patch.object(owner_module, 'RadialOwner', FailingOwner), \
                        mock.patch.object(owner_module, '_RADIAL_OWNER_STATES', states), \
                        mock.patch.object(owner_module, '_RADIAL_COMPACTION_PREFLIGHT_ERROR', None), \
                        mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                        mock.patch.object(cuda_d1, '_d1_submit_publication') as submit, \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaises(type(error)):
                        owner_module.consume_radial_device_union(
                            task, SimpleNamespace(union_dev=np.zeros((2, 2, 2), np.uint8),
                                                  host_written=False))
                self.assertEqual(events, ['consume', 'gpu_pack'])
                self.assertEqual(len(states), 1)
                submit.assert_not_called()


@unittest.skipUnless(os.environ.get('XTA_RUN_CUDA_RADIAL_OWNER') == '1',
                     'Opt-in real CUDA Radial owner qualification')
class RadialOwnerGpuExportFunctionalTests(unittest.TestCase):
    def test_real_empty_and_nonempty_packed_cvol_match_cpu_reference(self):
        output_root = os.environ.get('XTA_RADIAL_GPU_TEST_OUTPUT_DIR')
        if not output_root:
            self.skipTest('Set XTA_RADIAL_GPU_TEST_OUTPUT_DIR to task-specific Scratch directory')
        from XTA import cylindrical_projection as reference, geometry
        from XTA.interpolation import RawBBoxMaskStore
        import cupy as cp

        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        shape = (7, 9, 11)
        view = geometry.get_view_infos(*shape, cartesian_views=(),
                                       radial_views=('transverse',),
                                       radial_min_radius=.1, radial_patch_size=8)[0]
        RealOwner = owner_module.RadialOwner
        with tempfile.TemporaryDirectory(dir=root, prefix='real-gpu-export-') as temporary:
            work = Path(temporary)
            for mode in ('empty', 'nonempty'):
                native = np.zeros((view.num_slices, 8, 8), np.uint8)
                if mode == 'nonempty':
                    native[:, 1:7, 1:7] = 1
                    native[:, 2:6, 2:6] = 0
                task = dict(projection_contract=owner_module.RADIAL_OWNER_CONTRACT,
                            result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                            streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                            view=view, model_name='fixture', slice_start=0,
                            slice_count=view.num_slices, d1_output_shape=shape,
                            d1_store_dir=str(work / f'{mode}.cvol'))
                accumulator = SimpleNamespace(union_dev=cp.asarray(native), host_written=False)
                states = {}
                with mock.patch.dict(os.environ, {
                            'YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION': '1',
                            'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}), \
                        mock.patch.object(owner_module, 'RadialOwner',
                                          side_effect=lambda v, m, o: RealOwner(
                                              v, m, o, reserve_bytes=0)), \
                        mock.patch.object(owner_module, '_RADIAL_OWNER_STATES', states), \
                        mock.patch.object(owner_module, '_RADIAL_COMPACTION_PREFLIGHT_ERROR', None), \
                        mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                        mock.patch.object(RealOwner, 'host_words',
                                          side_effect=AssertionError('GPU compact/empty path downloaded words')):
                    result = owner_module.consume_radial_device_union(task, accumulator)
                    published = result['_publication_future'].result(timeout=30)
                self.assertEqual(states, {})
                self.assertEqual(result['d1_bitset_export_kind'],
                                 'empty' if mode == 'empty' else 'gpu_packed')
                self.assertEqual(result['d1_bitset_words'], (math.prod(shape) + 31) // 32)
                store = RawBBoxMaskStore.open(published['d1_layer_ref'].path)
                try:
                    actual = np.stack([store.decode_slice(z) for z in range(shape[0])])
                finally:
                    store.close()
                if mode == 'empty':
                    expected = np.zeros(shape, np.uint8)
                else:
                    cleaned = native.copy()
                    cleaned[:, 2:6, 2:6] = 1
                    radii = np.asarray(geometry.radial_global_radii(view))
                    expected = np.stack([
                        reference._pull_radial_chunk(cleaned, view, radii, shape,
                                                     z, 0, shape[1] * shape[2]).reshape(shape[1:])
                        for z in range(shape[0])])
                self.assertTrue(np.array_equal(actual, expected))
        cuda_d1._shutdown_d1_worker_pipeline()


if __name__ == '__main__':
    unittest.main()
