"""Proven-empty Radial owners publish the same CVOL without downloading a bitset."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_d1, cylindrical_owner, geometry
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore


class RadialEmptyPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (7, 9, 11)
        self.view = geometry.get_view_infos(
            *self.shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.1, radial_patch_size=8)[0]

    def _publish(self, label, *, packed, proven_empty):
        words = None if proven_empty else np.zeros((math.prod(self.shape) + 31) // 32, np.uint32)
        with mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': str(int(packed))}):
            result = cuda_d1._d1_finalize_bitset_layer(
                words=words, output_shape=self.shape, store_dir=self.root / label,
                model_name='fixture', view=self.view,
                projection_kind=cylindrical_owner.RADIAL_OWNER_CONTRACT,
                proven_empty=proven_empty)
        return result

    def test_proven_empty_matches_full_zero_bitset_in_both_formats(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                normal = self._publish(f'normal-{packed}', packed=packed, proven_empty=False)
                shortcut = self._publish(f'empty-{packed}', packed=packed, proven_empty=True)
                self.assertEqual(normal['d1_cvol_stats'], shortcut['d1_cvol_stats'])
                a, b = normal['d1_layer_ref'], shortcut['d1_layer_ref']
                for field in ('key', 'name', 'shape', 'dtype', 'storage_format',
                              'model_name', 'view_name', 'physical_view_name',
                              'segment_extent_ijk', 'segment_extent_shape_tyx',
                              'segment_extent_source'):
                    self.assertEqual(getattr(a, field), getattr(b, field), field)
                self.assertEqual(tuple(a.segment_extent_ijk), (0, -1, 0, -1, 0, -1))
                self.assertEqual(
                    json.loads((a.path / 'meta.json').read_text()),
                    json.loads((b.path / 'meta.json').read_text()))
                self.assertEqual((a.path / 'index.bin').read_bytes(),
                                 (b.path / 'index.bin').read_bytes())
                self.assertEqual((a.path / 'chunks.bin').read_bytes(),
                                 (b.path / 'chunks.bin').read_bytes())
                store = RawBBoxMaskStore.open(b.path)
                try:
                    self.assertEqual(int(np.count_nonzero(store.index['kind'])), 0)
                    for z in (0, self.shape[0] // 2, self.shape[0] - 1):
                        self.assertEqual(int(np.count_nonzero(store.decode_slice(z))), 0)
                finally:
                    store.close()

    def test_planned_payload_backing_and_spill_match_zero_bitset(self):
        # Windows test sandboxes may lack symlink privilege. An empty payload
        # hardlink exercises the same planned-backing and spill decisions.
        def link_backing(link, target, *args, **kwargs):
            os.link(target, link)

        for spill in (False, True):
            with self.subTest(spill=spill):
                outputs = []
                spill_calls = []

                def windows_empty_spill(writer):
                    # Windows cannot replace an open hardlink. For this
                    # zero-payload fixture, changing the planned-backing
                    # state tests the branch and metadata; Linux runs the
                    # actual spill implementation with a real symlink.
                    spill_calls.append(writer)
                    writer._ram_payload = False

                spill_patch = (mock.patch.object(
                    IncrementalRawBBoxMaskStoreWriter, 'spill_payload_to_disk',
                    windows_empty_spill) if spill and os.name == 'nt' else nullcontext())
                link_patch = (mock.patch.object(Path, 'symlink_to', link_backing)
                              if os.name == 'nt' else nullcontext())
                with link_patch, \
                        mock.patch.object(cuda_d1, 'publication_ram_headroom',
                                          return_value=10**12), spill_patch:
                    for proven_empty in (False, True):
                        label = f'backing-{spill}-{proven_empty}'
                        backing = self.root / f'{label}.raw'
                        backing.touch()
                        words = (None if proven_empty else
                                 np.zeros((math.prod(self.shape) + 31) // 32, np.uint32))
                        result = cuda_d1._d1_finalize_bitset_layer(
                            words=words, output_shape=self.shape,
                            store_dir=self.root / label, model_name='fixture',
                            view=self.view,
                            projection_kind=cylindrical_owner.RADIAL_OWNER_CONTRACT,
                            memory_payload_path=str(backing),
                            memory_payload_limit=0 if spill else 10**12,
                            proven_empty=proven_empty)
                        outputs.append(result)
                if spill and os.name == 'nt':
                    self.assertEqual(len(spill_calls), 2)
                a, b = (result['d1_layer_ref'].path for result in outputs)
                self.assertEqual(outputs[0]['d1_cvol_stats'], outputs[1]['d1_cvol_stats'])
                expected_backing = 'disk' if spill else 'planned_memfd'
                self.assertEqual(outputs[0]['d1_cvol_stats']['payload_backing'],
                                 expected_backing)
                for filename in ('meta.json', 'index.bin', 'chunks.bin'):
                    self.assertEqual((a / filename).read_bytes(),
                                     (b / filename).read_bytes(), filename)

    def test_submit_keeps_future_contract_and_rejects_mismatched_inputs(self):
        state = SimpleNamespace(output_shape=self.shape, store_dir=self.root / 'future',
                                key=('fixture', self.view.name), view=self.view,
                                projection_kind=cylindrical_owner.RADIAL_OWNER_CONTRACT)
        with self.assertRaises(ValueError):
            cuda_d1._d1_submit_publication(words=None, state=state)
        with self.assertRaises(ValueError):
            cuda_d1._d1_submit_publication(words=np.zeros(1, np.uint32), state=state,
                                            proven_empty=True)
        with ThreadPoolExecutor(max_workers=1) as pool, \
                mock.patch.object(cuda_d1, '_d1_publication_executor', return_value=pool), \
                mock.patch.object(cuda_d1, '_D1_PUBLICATION_SEMAPHORE', threading.BoundedSemaphore(1)):
            future = cuda_d1._d1_submit_publication(words=None, state=state,
                                                      proven_empty=True)
            self.assertIsInstance(future, Future)
            result = future.result(timeout=5)
        self.assertEqual(result['d1_cvol_stats']['nonempty_slices'], 0)
        self.assertGreaterEqual(result['d1_publication_seconds'], 0)

    def test_nonempty_bitset_still_publishes_selected_bit_and_ref(self):
        z, y, x = 2, 3, 4
        bit = (z * self.shape[1] + y) * self.shape[2] + x
        for packed in (False, True):
            with self.subTest(packed=packed):
                words = np.zeros((math.prod(self.shape) + 31) // 32, np.uint32)
                words[bit // 32] = np.uint32(1 << (bit % 32))
                with mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': str(int(packed))}):
                    result = cuda_d1._d1_finalize_bitset_layer(
                        words=words, output_shape=self.shape,
                        store_dir=self.root / f'nonempty-{packed}',
                        model_name='fixture', view=self.view,
                        projection_kind=cylindrical_owner.RADIAL_OWNER_CONTRACT)
                self.assertEqual(result['d1_cvol_stats']['nonempty_slices'], 1)
                self.assertEqual(result['d1_cvol_stats']['foreground_voxels'], 1)
                ref = result['d1_layer_ref']
                self.assertEqual(tuple(ref.segment_extent_ijk), (x, x, y, y, z, z))
                store = RawBBoxMaskStore.open(ref.path)
                try:
                    decoded = store.decode_slice(z)
                    self.assertEqual(int(np.count_nonzero(decoded)), 1)
                    self.assertEqual(int(decoded[y, x]), 1)
                finally:
                    store.close()

    def test_empty_proof_requires_complete_healthy_open_owner(self):
        owner = cylindrical_owner.RadialOwner.__new__(cylindrical_owner.RadialOwner)
        owner.coverage = np.array([True, False], dtype=bool)
        owner.native_any = False
        owner._failed = owner._closed = False
        self.assertFalse(owner.proven_empty_output())
        owner.coverage[:] = True
        owner._failed = True
        self.assertFalse(owner.proven_empty_output())
        owner._failed = False
        owner._closed = True
        self.assertFalse(owner.proven_empty_output())
        owner._closed = False
        owner.native_any = True
        self.assertFalse(owner.proven_empty_output())
        owner.native_any = False
        self.assertTrue(owner.proven_empty_output())

    def test_final_empty_owner_skips_download_and_preserves_logical_word_count(self):
        view = SimpleNamespace(name='radial_fixture__tta_a0', family='radial',
                               num_slices=2, tta_angle_deg=0.)
        output_shape = (3, 4, 5)
        created = []

        class FakeOwner:
            proven_empty_output = cylindrical_owner.RadialOwner.proven_empty_output

            def __init__(self, view, mask_shape, output_shape):
                self.coverage = np.zeros(view.num_slices, dtype=bool)
                self.native_any = False
                self._closed = self._failed = False
                self.words = SimpleNamespace(nbytes=8)
                self.cleanup_seconds = self.projection_seconds = 0.
                self.created_at = time.perf_counter()
                self.downloads = 0
                created.append(self)

            def consume(self, first, masks):
                self.coverage[first:first + len(masks)] = True
                return 0

            def host_words(self):
                self.downloads += 1
                raise AssertionError('Empty owner must not download host words')

            def close(self):
                self._closed = True

        task = dict(result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                    streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                    view=view, model_name='fixture', slice_start=0, slice_count=2,
                    d1_output_shape=output_shape, d1_store_dir=str(self.root / 'fake'))
        accumulator = SimpleNamespace(union_dev=np.zeros((2, 2, 2), np.uint8), host_written=False)
        future = Future(); future.set_result({'ok': True})
        text = io.StringIO()
        with mock.patch.object(cylindrical_owner, 'is_radial_owner_task', return_value=True), \
                mock.patch.object(cylindrical_owner, 'RadialOwner', FakeOwner), \
                mock.patch.object(cylindrical_owner, '_RADIAL_OWNER_STATES', {}), \
                mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                mock.patch.object(cuda_d1, '_d1_submit_publication', return_value=future) as submit, \
                redirect_stdout(text):
            result = cylindrical_owner.consume_radial_device_union(task, accumulator)
        self.assertTrue(result['d1_view_complete'])
        self.assertTrue(result['radial_owner_empty'])
        self.assertEqual(result['d1_bitset_words'], (math.prod(output_shape) + 31) // 32)
        self.assertIs(result['_publication_future'], future)
        self.assertEqual(created[0].downloads, 0)
        submit.assert_called_once_with(words=None, state=created[0], proven_empty=True)
        self.assertIn('setup_s=', text.getvalue())
        self.assertIn('host_bitset_bytes=0, packed_payload_bytes=0, '
                      'host_transfer_bytes=0, empty_shortcut=1', text.getvalue())

    def test_final_nonempty_owner_keeps_download_and_submission(self):
        view = SimpleNamespace(name='radial_fixture__tta_a0', family='radial',
                               num_slices=2, tta_angle_deg=0.)
        output_shape = (3, 4, 5)
        created = []

        class FakeOwner:
            proven_empty_output = cylindrical_owner.RadialOwner.proven_empty_output

            def __init__(self, view, mask_shape, output_shape):
                self.coverage = np.zeros(view.num_slices, dtype=bool)
                self.native_any = False
                self._closed = self._failed = False
                self.words = SimpleNamespace(nbytes=8)
                self.cleanup_seconds = self.projection_seconds = 0.
                self.created_at = time.perf_counter()
                self.downloads = 0
                created.append(self)

            def consume(self, first, masks):
                self.coverage[first:first + len(masks)] = True
                self.native_any = True
                return 1

            def host_words(self):
                self.downloads += 1
                return np.array([1, 0], np.uint32)

            def close(self):
                self._closed = True

        task = dict(result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                    streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                    view=view, model_name='fixture', slice_start=0, slice_count=2,
                    d1_output_shape=output_shape, d1_store_dir=str(self.root / 'fake-nonempty'))
        accumulator = SimpleNamespace(union_dev=np.zeros((2, 2, 2), np.uint8), host_written=False)
        future = Future(); future.set_result({'ok': True})
        text = io.StringIO()
        with mock.patch.dict(os.environ, {'YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION': '0'}), \
                mock.patch.object(cylindrical_owner, 'is_radial_owner_task', return_value=True), \
                mock.patch.object(cylindrical_owner, 'RadialOwner', FakeOwner), \
                mock.patch.object(cylindrical_owner, '_RADIAL_OWNER_STATES', {}), \
                mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                mock.patch.object(cuda_d1, '_d1_submit_publication', return_value=future) as submit, \
                redirect_stdout(text):
            result = cylindrical_owner.consume_radial_device_union(task, accumulator)
        self.assertFalse(result['radial_owner_empty'])
        self.assertEqual(created[0].downloads, 1)
        self.assertEqual(result['d1_bitset_words'], 2)
        self.assertIs(result['_publication_future'], future)
        submit.assert_called_once()
        kwargs = submit.call_args.kwargs
        self.assertTrue(np.array_equal(kwargs['words'], np.array([1, 0], np.uint32)))
        self.assertIs(kwargs['state'], created[0])
        self.assertFalse(kwargs['proven_empty'])
        self.assertIn('host_bitset_bytes=8, packed_payload_bytes=0, '
                      'host_transfer_bytes=8, empty_shortcut=0', text.getvalue())

    def test_partial_or_failed_owner_never_reaches_empty_publication(self):
        view = SimpleNamespace(name='radial_partial__tta_a0', family='radial',
                               num_slices=2, tta_angle_deg=0.)
        task = dict(result_mode='d1_owner', kind='fullframe', prediction_batch=1,
                    streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.,
                    view=view, model_name='fixture', slice_start=0, slice_count=1,
                    d1_output_shape=(3, 4, 5), d1_store_dir=str(self.root / 'partial'))
        for fail in (False, True):
            with self.subTest(fail=fail):
                owner = mock.Mock()
                owner.coverage = np.array([True, False], dtype=bool)
                owner.native_any = False
                owner.words.nbytes = 8
                owner.consume.return_value = 0
                if fail:
                    owner.consume.side_effect = RuntimeError('simulated failed consume')
                accumulator = SimpleNamespace(
                    union_dev=np.zeros((1, 2, 2), np.uint8), host_written=False)
                with mock.patch.object(cylindrical_owner, 'is_radial_owner_task', return_value=True), \
                        mock.patch.object(cylindrical_owner, 'RadialOwner', return_value=owner), \
                        mock.patch.object(cylindrical_owner, '_RADIAL_OWNER_STATES', {}), \
                        mock.patch.object(cuda_d1, '_D1_WORKER_VIEW_STATES', {}), \
                        mock.patch.object(cuda_d1, '_d1_submit_publication') as submit, \
                        redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'simulated failed consume'):
                            cylindrical_owner.consume_radial_device_union(task, accumulator)
                    else:
                        result = cylindrical_owner.consume_radial_device_union(task, accumulator)
                        self.assertFalse(result['d1_view_complete'])
                owner.proven_empty_output.assert_not_called()
                owner.host_words.assert_not_called()
                submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
