"""Detached encoded Radial blocks retain the established D1 CVOL contract."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import traceback
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from XTA import cuda_d1, geometry
from XTA.cylindrical_cuda_projection import RadialEncodedBlock, RadialEncodedSlice
from XTA.interpolation import RawBBoxMaskStore


def _words(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.reshape(-1), bitorder='little')
    padded = np.pad(packed, (0, -len(packed) % 4))
    return np.ascontiguousarray(padded.view(np.uint32))


def _blocks(mask: np.ndarray, *, packed: bool, depth: int = 6) -> tuple[RadialEncodedBlock, ...]:
    blocks = []
    for first in range(0, mask.shape[0], depth):
        records = []
        pieces = []
        offset = 0
        for z in range(first, min(first + depth, mask.shape[0])):
            plane = mask[z]
            rows = np.flatnonzero(np.any(plane, axis=1))
            cols = np.flatnonzero(np.any(plane, axis=0))
            if len(rows):
                y0, y1, x0, x1 = int(rows[0]), int(rows[-1])+1, int(cols[0]), int(cols[-1])+1
                crop = np.ascontiguousarray(plane[y0:y1, x0:x1])
                payload = (np.packbits(crop, axis=1, bitorder='little')
                           if packed else crop).reshape(-1).copy()
                foreground = int(np.count_nonzero(crop))
            else:
                y0 = y1 = x0 = x1 = foreground = 0
                payload = np.empty(0, np.uint8)
            size = int(payload.size)
            records.append(RadialEncodedSlice(z, y0, y1, x0, x1,
                                              foreground, offset, size))
            pieces.append(payload)
            offset += size
        owned = np.concatenate(pieces).astype(np.uint8, copy=True)
        owned.flags.writeable = False
        blocks.append(RadialEncodedBlock(first, tuple(records), owned, packed))
    return tuple(blocks)


class D1EncodedPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.view = geometry.get_view_infos(
            7, 9, 11, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.1, radial_patch_size=8,
        )[0]

    def _finalize(self, name, shape, *, words=None, blocks=None, packed=True):
        with mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': str(int(packed))}), \
                mock.patch.object(cuda_d1, 'd1_unpack_target_mib', return_value=1):
            return cuda_d1._d1_finalize_bitset_layer(
                words=words, encoded_blocks=blocks, output_shape=shape,
                store_dir=self.root / name, model_name='fixture', view=self.view,
                projection_kind='radial_native_pull_v1',
            )

    def test_packed_and_raw_rechunk_match_full_words_across_original_z_bands(self):
        # 1 MiB / (201*193) yields the original 27-slice publication band.
        # Six-slice input blocks cross its z=27 boundary, including empty spans.
        shape = (35, 201, 193)
        mask = np.zeros(shape, np.uint8)
        for z, y, x in ((0, 1, 2), (1, 198, 191), (24, 2, 32),
                        (25, 17, 33), (26, 100, 192), (27, 0, 0),
                        (29, 200, 190), (30, 12, 8), (34, 199, 1)):
            mask[z, y, x] = 1
        mask[24, 4:9, 14:19] = 1  # odd packed row width and cross-word addresses
        for packed in (False, True):
            with self.subTest(packed=packed):
                encoded = _blocks(mask, packed=packed)
                original = self._finalize(f'words-{packed}', shape,
                                          words=_words(mask), packed=packed)
                candidate = self._finalize(f'encoded-{packed}', shape,
                                           blocks=encoded, packed=packed)
                self.assertEqual(original['d1_cvol_stats'], candidate['d1_cvol_stats'])
                for field in ('key', 'name', 'shape', 'dtype', 'storage_format',
                              'segment_extent_ijk', 'segment_extent_shape_tyx'):
                    self.assertEqual(getattr(original['d1_layer_ref'], field),
                                     getattr(candidate['d1_layer_ref'], field), field)
                for name in ('index.bin', 'chunks.bin', 'meta.json'):
                    a = (original['d1_layer_ref'].path / name).read_bytes()
                    b = (candidate['d1_layer_ref'].path / name).read_bytes()
                    self.assertEqual(a, b, name)
                store = RawBBoxMaskStore.open(candidate['d1_layer_ref'].path)
                try:
                    for z in (0, 12, 24, 26, 27, 29, 30, 34):
                        np.testing.assert_array_equal(store.decode_slice(z), mask[z])
                finally:
                    store.close()

    def test_rejects_lazy_mutable_borrowed_out_of_order_and_wrong_mode(self):
        shape = (3, 3, 11)
        mask = np.zeros(shape, np.uint8)
        mask[0, 1, 4] = 1
        good = _blocks(mask, packed=True, depth=2)
        bads = [
            (),
            list(good),
            good[::-1],
            good[:-1],
            (replace(good[0], packed=False), good[1]),
            (replace(good[0], payload=np.array(good[0].payload, copy=True)), good[1]),
            (replace(good[0], payload=np.frombuffer(good[0].payload.tobytes(), np.uint8)), good[1]),
            (replace(good[0], records=(replace(good[0].records[0], offset=1),
                                       *good[0].records[1:])), good[1]),
        ]
        state = SimpleNamespace(output_shape=shape)
        for blocks in bads:
            with self.subTest(reason=type(blocks).__name__), \
                    mock.patch.object(cuda_d1, '_d1_publication_executor') as executor, \
                    mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}):
                with self.assertRaises((TypeError, ValueError)):
                    cuda_d1._d1_submit_publication(words=None, state=state,
                                                    encoded_blocks=blocks)
                executor.assert_not_called()
        with self.assertRaises(ValueError):
            cuda_d1._d1_submit_publication(words=_words(mask), state=state,
                                            encoded_blocks=good)
        with self.assertRaises(ValueError):
            cuda_d1._d1_submit_publication(words=None, state=state,
                                            proven_empty=True, encoded_blocks=good)

    def test_rechunk_rebases_offsets_at_original_band_and_preserves_spill(self):
        shape = (35, 201, 193)
        mask = np.zeros(shape, np.uint8)
        for z, y, x in ((0, 1, 2), (24, 4, 14), (27, 8, 17), (29, 9, 18), (34, 2, 190)):
            mask[z, y, x] = 1
        blocks = _blocks(mask, packed=True)
        created = []

        class SpyWriter:
            def __init__(self, **_kwargs):
                self._ram_payload = True
                self._next_offset = 0
                self.calls = []
                created.append(self)

            def spill_payload_to_disk(self):
                self.calls.append(('spill',))
                self._ram_payload = False

            def consume_empty_range(self, z0, count):
                self.calls.append(('empty', z0, count))

            def consume_encoded_block(self, z0, records, payload, *, packed):
                self.calls.append(('encoded', z0, len(records), tuple(records),
                                   payload.tobytes(), packed))
                self._next_offset += len(payload)

            def finalize(self):
                return {'segment_extent_ijk': [0, -1, 0, -1, 0, -1],
                        'payload_backing': 'disk' if not self._ram_payload else 'planned_memfd'}

            def abort(self, _exc):
                raise AssertionError('Unexpected writer abort')

            def discard(self):
                raise AssertionError('Unexpected writer discard')

        with mock.patch.object(cuda_d1, 'IncrementalRawBBoxMaskStoreWriter', SpyWriter), \
                mock.patch.object(cuda_d1, 'd1_unpack_target_mib', return_value=1), \
                mock.patch.object(cuda_d1, 'publication_ram_headroom', side_effect=[10**9, 0]), \
                mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}):
            result = cuda_d1._d1_finalize_bitset_layer(
                words=None, encoded_blocks=blocks, output_shape=shape,
                store_dir=self.root / 'spy', model_name='fixture', view=self.view,
                projection_kind='radial_native_pull_v1',
                memory_payload_path='unused-spy-backing', memory_payload_limit=10**9,
            )
        calls = created[0].calls
        self.assertEqual(sum(call[0] == 'spill' for call in calls), 1)
        self.assertEqual(result['d1_cvol_stats']['payload_backing'], 'disk')
        encoded = [call for call in calls if call[0] == 'encoded']
        self.assertTrue(any(call[1] == 24 and call[2] == 3 for call in encoded))
        self.assertTrue(any(call[1] == 27 and call[2] == 3 for call in encoded))
        self.assertEqual(b''.join(call[4] for call in encoded),
                         b''.join(block.payload.tobytes() for block in blocks))
        for _kind, z0, count, records, payload, packed in encoded:
            self.assertTrue(packed)
            self.assertEqual(records[0].z, z0)
            self.assertEqual(records[0].offset, 0)
            self.assertEqual(records[-1].offset + records[-1].size, len(payload))

    def test_async_future_freezes_packed_mode_and_releases_semaphore(self):
        shape = (3, 3, 11)
        mask = np.zeros(shape, np.uint8)
        mask[0, 1, 4] = 1
        blocks = _blocks(mask, packed=True, depth=2)
        state = SimpleNamespace(output_shape=shape, store_dir=self.root / 'future',
                                key=('fixture', self.view.name), view=self.view,
                                projection_kind='radial_native_pull_v1')
        gate = threading.Event()
        semaphore = threading.BoundedSemaphore(1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocker = pool.submit(gate.wait)
            with mock.patch.object(cuda_d1, '_d1_publication_executor', return_value=pool), \
                    mock.patch.object(cuda_d1, '_D1_PUBLICATION_SEMAPHORE', semaphore), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}):
                future = cuda_d1._d1_submit_publication(words=None, state=state,
                                                          encoded_blocks=blocks)
            self.assertIsInstance(future, Future)
            gate.set()
            blocker.result(timeout=5)
            result = future.result(timeout=10)
        self.assertEqual(result['d1_cvol_stats']['nonempty_slices'], 1)
        self.assertEqual(result['d1_layer_ref'].storage_format,
                         cuda_d1.INTERNAL_PACKED_CVOL_FORMAT)
        self.assertTrue(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_writer_error_aborts_encoded_publication(self):
        shape = (3, 3, 11)
        mask = np.zeros(shape, np.uint8)
        mask[0, 1, 4] = 1
        blocks = list(_blocks(mask, packed=True, depth=2))
        payload = blocks[0].payload.copy()
        payload[-1] |= np.uint8(0x80)  # invalid nonzero padding above width one
        payload.flags.writeable = False
        blocks[0] = replace(blocks[0], payload=payload)
        with self.assertRaises(ValueError):
            self._finalize('malformed', shape, blocks=tuple(blocks), packed=True)
        self.assertFalse((self.root / 'malformed' / 'meta.json').exists())

    def test_failed_async_publication_releases_packed_payload_before_credit(self):
        shape = (3, 3, 11)
        mask = np.zeros(shape, np.uint8)
        mask[0, 1, 4] = 1
        state = SimpleNamespace(output_shape=shape, store_dir=self.root / 'failed-future',
                                key=('fixture', self.view.name), view=self.view,
                                projection_kind='radial_native_pull_v1')
        calls = []

        class FailingWriter:
            def __init__(self, **_kwargs):
                self._ram_payload = False
                self._next_offset = 0

            def consume_encoded_block(self, *_args, **_kwargs):
                calls.append('consume')
                raise RuntimeError('injected encoded write failure')

            def abort(self, _error):
                calls.append('abort')

            def discard(self):
                calls.append('discard')

        def submit():
            blocks = _blocks(mask, packed=True, depth=2)
            payload_ref = weakref.ref(blocks[0].payload)
            future = cuda_d1._d1_submit_publication(
                words=None, state=state, encoded_blocks=blocks,
            )
            return future, payload_ref

        semaphore = threading.BoundedSemaphore(1)
        with ThreadPoolExecutor(max_workers=1) as pool, \
                mock.patch.object(cuda_d1, '_d1_publication_executor', return_value=pool), \
                mock.patch.object(cuda_d1, '_D1_PUBLICATION_SEMAPHORE', semaphore), \
                mock.patch.object(cuda_d1, 'IncrementalRawBBoxMaskStoreWriter', FailingWriter), \
                mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1'}):
            future, payload_ref = submit()
            try:
                future.result(timeout=10)
            except RuntimeError as error:
                self.assertEqual(str(error), 'injected encoded write failure')
                diagnostic = ''.join(traceback.format_exception(error))
            else:
                self.fail('Failed writer unexpectedly published encoded blocks')
            gc.collect()
            self.assertIsNone(payload_ref())
            self.assertEqual(calls, ['consume', 'abort', 'discard'])
            self.assertTrue(semaphore.acquire(blocking=False))
            semaphore.release()
            self.assertIn('consume_encoded_block', diagnostic)
            self.assertIn('injected encoded write failure', diagnostic)


if __name__ == '__main__':
    unittest.main()
