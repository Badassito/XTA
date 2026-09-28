"""Dense preparation failures release mappings without retiring publication inputs."""
from __future__ import annotations

import contextlib
from concurrent.futures import Future
import gc
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from XTA import assembly
from XTA.config import GIB
from XTA.geometry import ViewInfo
from XTA.interpolation import CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.interpolation import _DirectUnionBackingLease
from XTA.runtime import close_memmap_array_without_flush
from XTA.view_prepare import (
    AdmittedViewPrepare, ComponentProjectionSubmitter, ViewPrepareLeaseState,
)


class ViewPrepareFailureOwnershipTests(unittest.TestCase):
    shape = (4, 5, 6)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(gc.collect)
        self.root = Path(temporary.name)
        self.view = ViewInfo(name='transverse__tta_0', num_slices=4, src_h=5, src_w=6,
                             pad_mode='clamp', physical_view_name='transverse', tta_aug_id='0')

    def mapping(self, name):
        value = np.memmap(self.root / name, mode='w+', dtype=np.uint8, shape=self.shape)
        value[:] = 0
        value[0, 1, 1] = 1
        self.addCleanup(close_memmap_array_without_flush, value, unlink_path=Path(value.filename))
        return value

    def arguments(self, original):
        return dict(model_name='model', view=self.view, union_mm=original, confmap_mm=None,
                    union_path=self.root / 'original.dat', confmap_path=None, temp_dir=self.root,
                    dense_tiling_active=False, min_conf=0, min_radius=0, interpolate=3,
                    interpolation_walk_back=1, interpolation_candidates=2, interpolate_passes=1,
                    interpolate_min_radius=0, interpolation_search_angle=0, keep_temp=False,
                    slice_workers=1, interpolation_task_workers=1, nrrd_layers_enabled=True,
                    precleaned_slice_cleanup=True, hole_fill_done_on_device=True,
                    preinterpolation_layer_already_published=True)

    def component_stats(self, component_dir):
        entries = []
        data = np.zeros(self.shape, dtype=np.uint8)
        data[1, 2:4, 2:5] = 1
        for candidate in (1, 2):
            path = component_dir / f'candidate{candidate}.cvol'
            write_raw_bbox_mask_store(data, path, format_name=CVOL_FORMAT,
                                      desc='failure ownership fixture', workers=1)
            entries.append(dict(path=str(path), walk_back_index=1, candidate_index=candidate,
                                added_voxels=int(data.sum())))
        return dict(added_voxels=int(data.sum()), bridge_component_deltas=entries), data

    def test_rebound_canvas_remains_readable_with_failure_traceback(self):
        for failure in ('coverage', 'submit', 'publication', 'later_pass'):
            with self.subTest(failure=failure):
                original = self.mapping(f'{failure}-original.dat')
                rebound = self.mapping(f'{failure}-rebound.dat')
                completed_inputs, submitted = [], []
                calls = 0

                def interpolate(**kwargs):
                    nonlocal calls
                    calls += 1
                    if failure == 'later_pass' and calls == 2:
                        raise RuntimeError('later pass failed')
                    stats, data = self.component_stats(kwargs['bridge_component_dir'])
                    completed_inputs.extend((Path(entry['path']), data.copy())
                                            for entry in stats['bridge_component_deltas'])
                    if failure == 'coverage':
                        stats['bridge_component_deltas'].pop()
                    return rebound, stats

                def submit(path, **kwargs):
                    if failure == 'submit' and submitted:
                        raise RuntimeError('another projection failed')
                    future = Future()
                    # The detached consumer keeps an independently opened sparse input.
                    reader = RawBBoxMaskStore.open(path, mmap_payload=True)
                    self.addCleanup(reader.close)
                    submitted.append((future, reader))
                    return future

                args = self.arguments(original)
                args['temp_dir'] = self.root / failure
                args['union_path'] = Path(original.filename)
                if failure != 'publication':
                    args['submit_component_projection'] = submit
                if failure == 'later_pass':
                    args['interpolate_passes'] = 2
                caught = []
                with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace'), \
                     mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process', side_effect=interpolate), \
                     mock.patch.object(assembly, 'materialize_interpolation_component_nrrd_view_layer',
                                       side_effect=RuntimeError('publication failed')), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        assembly.prepare_view_volume_after_fullframe(**args)
                    except RuntimeError as exc:
                        caught.append(exc)  # Keep its frames alive while checking the mappings.
                self.assertEqual(len(caught), 1)
                self.assertIsNotNone(caught[0].__traceback__)
                self.assertFalse(rebound._mmap.closed)
                self.assertEqual(int(rebound.sum()), 1)
                # The admission wrapper, not assembly, owns this original mapping.
                self.assertFalse(original._mmap.closed)
                for path, data in completed_inputs:
                    self.assertTrue(path.is_dir())
                    with contextlib.closing(RawBBoxMaskStore.open(path, mmap_payload=True)) as reader:
                        plane = np.empty(self.shape[1:], dtype=np.uint8)
                        reader.fill_decoded_slice_into(1, plane)
                        np.testing.assert_array_equal(plane, data[1])
                for future, reader in submitted:
                    self.assertFalse(future.cancelled())
                    self.assertFalse(future.done())
                    plane = np.empty(self.shape[1:], dtype=np.uint8)
                    reader.fill_decoded_slice_into(1, plane)
                    self.assertEqual(int(plane.sum()), 6)

    def test_failure_leaves_published_tile_support_open(self):
        original = self.mapping('tile-original.dat')
        supports = []
        args = self.arguments(original)
        args.update(dense_tiling_active=True,
                    parent_mask_ready_callback=lambda model, view, support: supports.append(support))
        with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace'), \
             mock.patch.object(assembly, 'allocate_workspace_array',
                               side_effect=lambda **kw: np.zeros(kw['shape'], dtype=kw['dtype'])), \
             mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process',
                               side_effect=RuntimeError('tile parent pass failed')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'tile parent pass failed'):
                assembly.prepare_view_volume_after_fullframe(**args)
        self.assertEqual(len(supports), 1)
        self.addCleanup(supports[0].close)
        # Simulate the outer admission owner's failure cleanup while a gate owns P.
        close_memmap_array_without_flush(original)
        plane = np.empty(self.shape[1:], dtype=np.uint8)
        supports[0].fill_decoded_slice_into(0, plane)
        self.assertEqual(int(plane.sum()), 1)

    def run_admitted(self, *, materialized, fail):
        local = self.mapping(f'admitted-{materialized}-{fail}.dat')
        exit_closed = []

        @contextlib.contextmanager
        def reserve(*args):
            try:
                yield
            finally:
                exit_closed.append(local._mmap.closed)

        result = SimpleNamespace(live_array=local)
        prepare = mock.Mock(return_value=result)
        if fail:
            prepare.side_effect = RuntimeError('component queue refused submission')
        materialize_workspace = mock.Mock(return_value=local)
        task = AdmittedViewPrepare(
            admission=SimpleNamespace(reserve=reserve), transient_bytes=local.nbytes,
            model_name='model', view=self.view, union_mm=None if materialized else local,
            confmap_mm=None, d1_shadow_path=self.root / 'input.cvol',
            union_path=Path(local.filename), confmap_path=None, temp_dir=self.root,
            dense_tiling_active=False, min_conf=0, min_radius=0,
            interpolation_distance=3, interpolation_walk_back=1,
            interpolation_candidates=2, interpolation_passes=1,
            interpolation_min_radius=0, interpolation_search_angle=0,
            keep_temp_artifacts=True, slice_workers=1, interpolation_task_workers=1,
            component_layers_needed=True, precleaned_slice_cleanup=True,
            hole_fill_done_on_device=True, slice_meta=None,
            fuse_azimuthal_component_layers=lambda: False,
            component_ref_dense_retirement_active=True,
            preinterpolation_layer_already_published=True,
            parent_mask_ready_callback=None,
            submit_component_projection=mock.Mock(),
            materialize_workspace=materialize_workspace, prepare=prepare,
        )
        if fail:
            caught = None
            try:
                task()
            except RuntimeError as exc:
                caught = exc
            self.assertIsNotNone(caught)
            self.assertIsNotNone(caught.__traceback__)
            self.assertFalse(local._mmap.closed)
            self.assertEqual(int(local.sum()), 1)
        else:
            self.assertIs(task(), result)
            self.assertFalse(local._mmap.closed)
            self.assertEqual(int(result.live_array.sum()), 1)
        self.assertEqual(exit_closed, [False])
        self.assertEqual(materialize_workspace.call_count, int(materialized))
        self.assertTrue(prepare.call_args.kwargs['nrrd_layers_enabled'])

    def test_admission_keeps_failed_canvas_safe_while_traceback_retains_it(self):
        for materialized in (False, True):
            with self.subTest(materialized=materialized):
                self.run_admitted(materialized=materialized, fail=True)

    def test_success_keeps_existing_or_materialized_canvas_owned_by_result(self):
        for materialized in (False, True):
            with self.subTest(materialized=materialized):
                self.run_admitted(materialized=materialized, fail=False)


class ProjectionReservationTests(unittest.TestCase):
    def test_azimuthal_reservation_includes_target_maps_and_matches_fullframe_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'meta.json').write_text(json.dumps({'shape': [6, 3, 4]}), encoding='utf-8')
            queue = mock.Mock()
            submit = ComponentProjectionSubmitter(
                queue=queue, source_shape=(30, 40, 50), numba_available=True,
                materialize=mock.Mock(),
            )
            view = SimpleNamespace(family='azimuthal', full_t=3, full_h=4, full_w=5,
                                   num_slices=6, tta_angle_deg=15, physical_view_name='azimuthal_transverse')
            submit(path, view=view, added_voxels=3, source='fullframe')
            expected = 30 * 40 * ((50 + 7) // 8) + 16 * (40 * 50) + GIB
            self.assertEqual(queue.submit.call_args.kwargs['working_bytes'], expected)
            submit(path, view=view, added_voxels=3, source='tile')
            expected = 2 * (6 * 3 * 4) + 2 * (30 * 40 * 50) + 4 * GIB
            self.assertEqual(queue.submit.call_args.kwargs['working_bytes'], expected)


class ViewPrepareLeaseStateTests(unittest.TestCase):
    def test_handoff_rollback_and_completion_preserve_single_owner(self):
        key = ('model', 'view')
        lease = _DirectUnionBackingLease(key=key, nbytes=120, phase='inference', owner_count=1)
        state = ViewPrepareLeaseState(
            leases={key: lease}, inference_views={key}, inference_bytes={key: 120},
            postprocess_views=set(), postprocess_bytes={},
        )
        self.assertTrue(state.handoff(key))
        self.assertEqual(lease.phase, 'postprocess')
        self.assertEqual(state.postprocess_bytes, {key: 120})
        state.rollback_handoff(key)
        self.assertEqual(lease.phase, 'inference')
        self.assertEqual(state.inference_bytes, {key: 120})
        self.assertTrue(state.handoff(key))
        self.assertFalse(state.complete(key, retain_for_dense_retirement=True))
        self.assertIn(key, state.leases)
        self.assertTrue(state.complete(key, retain_for_dense_retirement=False))
        self.assertFalse(state.leases)
        self.assertFalse(state.postprocess_views)


if __name__ == '__main__':
    unittest.main()
