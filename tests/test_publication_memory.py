from __future__ import annotations
import unittest
from unittest import mock
import os
import sys
import json
import gzip
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from XTA import publication_memory as memory
from XTA import interpolation
from XTA.config import GIB
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, INTERNAL_PACKED_CVOL_FORMAT
from XTA.runtime import release_memfd_owners_under


def _cache_export_volumes():
    a = np.zeros((5,17,43), np.uint8)
    pattern = (np.indices((5,13,33)).sum(axis=0) % 2).astype(np.uint8)
    a[:,2:15,4:37] = pattern
    b = np.zeros_like(a); b[:,2:15,4:37] = 1-pattern
    c = np.zeros_like(a); c[:,1:16,2:41] = 1
    return a,b,c


def _verify_concurrent_cached_exports(root, volumes):
    """Hold multiple cached layers through the real native/mirror NRRD writers."""
    from XTA import outputs
    refs = [interpolation.NrrdLayerRef(key=f'layer{i}', name=f'layer{i}', path=root/f'store{i}',
                shape=a.shape, storage_format=INTERNAL_PACKED_CVOL_FORMAT)
            for i,a in enumerate(volumes)]
    def export(i, tag):
        native = root/f'{tag}-{i}.nrrd'; mirror = root/f'{tag}-{i}-mirror.nrrd'
        outputs.write_layer_nrrd_with_low_quality_mirrors(refs[i], volumes[i].shape,
                native, [((2,7,11), mirror)], segment_name=f'layer{i}')
        def decode(path):
            _header, payload = path.read_bytes().split(b'\n\n', 1)
            return gzip.decompress(payload)
        result = decode(native), decode(mirror)
        np.testing.assert_array_equal(np.frombuffer(result[0], np.uint8).reshape(volumes[i].shape), volumes[i])
        return result
    def software_writer(fh, **kwargs):
        return outputs._MemberParallelGzipPayloadWriter(fh, codec_spec=('zlib',1,gzip.compress))
    with mock.patch.object(outputs, '_open_nrrd_payload_writer', side_effect=software_writer):
        # Independent uncached reads establish both native and mirror oracles.
        with mock.patch.object(outputs, '_open_nrrd_layer_ref',
                side_effect=lambda ref: RawBBoxMaskStore.open(ref.path, mmap_payload=True)):
            expected = [export(i, 'uncached') for i in range(len(refs))]
        with ExitStack() as stack:
            for ref in refs:
                reader = RawBBoxMaskStore.open(ref.path, cache_payload_in_ram=True)
                stack.callback(interpolation._release_raw_store_chunks_ram_cache, reader.chunks_path)
                stack.callback(reader.close)
            with ThreadPoolExecutor(max_workers=len(refs)) as pool:
                futures = [pool.submit(export, i, 'cached') for i in range(len(refs))]
                actual = [future.result() for future in futures]
            if actual != expected:
                raise AssertionError('Cached layer or low-quality mirror contains another layer payload')


def _verify_cached_exports_in_subprocess(root, *, emulate_memfd_resolution):
    # Aggregate discovery substitutes dependency stubs. Exercise the actual
    # OpenCV mirror path in a fresh interpreter, as the other numerical fixtures do.
    import subprocess
    program = '''
import sys
from pathlib import Path
from unittest import mock
from tests.test_publication_memory import _cache_export_volumes,_verify_concurrent_cached_exports
resolve=Path.resolve
def display(path,*args,**kwargs):
    if sys.argv[2]=='1' and path.name=='chunks.bin':
        return Path('/memfd:xta-packed-publication (deleted)')
    return resolve(path,*args,**kwargs)
with mock.patch.object(Path,'resolve',display):
    _verify_concurrent_cached_exports(Path(sys.argv[1]),_cache_export_volumes())
'''
    result = subprocess.run([sys.executable, '-c', program, str(root),
                             '1' if emulate_memfd_resolution else '0'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise AssertionError(result.stdout)


class PublicationMemoryPlanTests(unittest.TestCase):
    def setUp(self):
        def bbox(a):
            ys, xs = np.nonzero(a)
            return ((int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
                    if len(xs) else (0, 0, 0, 0))
        patcher = mock.patch('XTA.interpolation.cv2.boundingRect', side_effect=bbox)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def policy_tasks(groups=1, ratio=4, shape=(3072, 3072, 3072), leases=1, mode='file'):
        """Production-size metadata only; never allocate the canvases."""
        tasks = []
        for group in range(groups):
            passes = [dict(kind='fullframe', model_name='model',
                view=SimpleNamespace(name=f'view{group}__policy{index}'),
                processing_shape=shape, result_mode=mode, bounded_parent_admission=True)
                for index in range(ratio)]
            passes[0]['augmentation_pass_tasks'] = passes[1:]
            tasks.extend(dict(passes[0], slice_start=index) for index in range(leases))
        return tasks

    @staticmethod
    def cluster_policy_tasks():
        """Reproduce the 144736 geometry using descriptors, never mask canvases."""
        from XTA.config import (AzimuthalViewRequest, RadialViewRequest,
                                SphericalViewRequest, TiltedViewGroup)
        from XTA.geometry import (expand_views_into_policy_variants,
                                  expand_views_into_tta_variants, view_processing_volume_shape)
        from XTA.unification.runtime import compile_physical_views
        axes = ('transverse', 'sagittal', 'coronal')
        targets = axes + tuple('tilted_' + axis for axis in axes)
        views = compile_physical_views(t_dim=2911, height=3064, width=3022,
            cartesian_views=axes,
            azimuthal_requests=tuple(AzimuthalViewRequest(target) for target in targets),
            tilted_groups=(TiltedViewGroup(axes, (30.,), ('vertical', 'horizontal')),),
            azimuthal_native_raster=3072,
            radial_requests=tuple(RadialViewRequest(target) for target in targets),
            radial_patch_size=3072,
            spherical_requests=(SphericalViewRequest('transverse'), SphericalViewRequest('tilted_transverse')),
            spherical_patch_size=3072, sampling_policy='coverage').views
        tasks = []
        for view in expand_views_into_tta_variants(views, [0]):
            passes = [dict(kind='fullframe', model_name='model', view=copy,
                processing_shape=view_processing_volume_shape(copy, 3072),
                slice_start=0, slice_count=copy.num_slices, out_size=3072,
                result_mode='direct_union', bounded_parent_admission=True)
                for copy in expand_views_into_policy_variants([view], 4)]
            passes[0]['augmentation_pass_tasks'] = passes[1:]
            tasks.append(passes[0])
        return tasks

    @staticmethod
    def cluster_policy_memory_plan(tasks, available_gib):
        source = (1931, 3064, 3022)
        available = int(available_gib * GIB)
        sink = SimpleNamespace(max_workers=12, output_shape=source,
            low_quality_specs=[SimpleNamespace(output_shape_t_y_x=(388,612,604))])
        largest_transient = int(np.prod(source)) + 4*GIB
        transient = max(largest_transient, min(available//8, 4*largest_transient))
        workers = 4*max(256*1024**2, 3072**2*80) + 4*2*(2*16*3072**2 + 16)
        return memory.policy_parent_memory_plan(tasks, requested_dense_limit=384*GIB,
            available_ram_bytes=available, source_shape=source,
            output_reserve_bytes=memory.publication_output_reserve(sink, 512*1024**2, 16*1024**2),
            parent_transient_reserve_bytes=transient, worker_buffer_reserve_bytes=workers,
            batch_size=1)

    def test_cluster_384_gib_window_fits_actual_groups_and_remains_ram_guarded(self):
        tasks = self.cluster_policy_tasks()
        self.assertEqual(len(tasks), 110)
        self.assertEqual(sum(task['slice_count'] for task in tasks), 160593)
        largest = max(4*int(np.prod(task['processing_shape'])) for task in tasks)
        self.assertAlmostEqual(largest/GIB, 155.56647767871618)
        self.assertLess(2*largest, 384*GIB)
        high = self.cluster_policy_memory_plan(tasks, 976.78)
        self.assertEqual(high['dense_limit_bytes'], 384*GIB)
        self.assertEqual(high['file_result_reserve_bytes'], 0)
        self.assertAlmostEqual(high['total_reserve_bytes']/GIB, 707.002436, places=4)
        self.assertLess(high['total_reserve_bytes'], high['available_ram_bytes'])
        self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
            total_dense_limit=high['dense_limit_bytes']), 384*GIB)
        low = self.cluster_policy_memory_plan(tasks, 300)
        self.assertLess(low['dense_limit_bytes'], largest)
        self.assertLessEqual(low['total_reserve_bytes'], low['available_ram_bytes'])
        with self.assertRaisesRegex(RuntimeError, 'dense admission limit'):
            memory.native_fullframe_dense_reserve(tasks, total_dense_limit=low['dense_limit_bytes'])

    def test_cluster_384_gib_allows_postprocess_overlap_without_two_large_inference_groups(self):
        from tests.test_tta_scheduler_boundary import _scheduler, _state
        tasks = sorted(self.cluster_policy_tasks(),
                       key=lambda task: int(np.prod(task['processing_shape'])), reverse=True)
        first, second = tasks[:2]
        state = _state()
        scheduler = _scheduler(Path('metadata-only'), state=state, input_overrides={
            'direct_union_inference_view_limit': 4,
            'direct_union_inference_byte_limit': 128*GIB,
            'direct_union_total_dense_byte_limit': 384*GIB})
        self.assertTrue(scheduler.direct_union_task_admissible(first))
        for parent in (first, *first['augmentation_pass_tasks']):
            key = (parent['model_name'], parent['view'].name)
            size = int(np.prod(parent['processing_shape']))
            state.direct_union_inference_views.add(key)
            state.direct_union_inference_bytes[key] = size
            state.direct_union_backing_leases[key] = interpolation._DirectUnionBackingLease(key, size)
            state.direct_union_admission_group_by_parent[key] = (first['model_name'], first['view'].name)
        self.assertFalse(scheduler.direct_union_task_admissible(second))
        for key in tuple(state.direct_union_inference_views):
            state.direct_union_backing_leases[key].transition('inference', 'postprocess')
            state.direct_union_postprocess_views.add(key)
            state.direct_union_postprocess_bytes[key] = state.direct_union_inference_bytes.pop(key)
        state.direct_union_inference_views.clear()
        self.assertTrue(scheduler.direct_union_task_admissible(second))
        old_scheduler = _scheduler(Path('metadata-only'), state=state, input_overrides={
            'direct_union_inference_view_limit': 4,
            'direct_union_inference_byte_limit': 128*GIB,
            'direct_union_total_dense_byte_limit': 256*GIB})
        self.assertFalse(old_scheduler.direct_union_task_admissible(second))

    def test_detection_confidence_and_policy_profile_do_not_resize_parent_canvases(self):
        tasks = self.policy_tasks(shape=(4747,2911,3022), mode='direct_union')
        sizes = []
        for confidence, profile in ((0.5, 'superheavy'), (0.8, 'baseline')):
            for task in (tasks[0], *tasks[0]['augmentation_pass_tasks']):
                task['conf'] = confidence
                task['augmentation_settings'] = SimpleNamespace(coverage='packed', profile=profile)
            sizes.append(memory.native_fullframe_dense_reserve(tasks,
                total_dense_limit=384*GIB, min_conf=0))
        self.assertEqual(sizes[0], sizes[1])
        # --min_conf is different: positive values retain a confidence canvas.
        self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
            total_dense_limit=384*GIB, min_conf=0.1), 2*sizes[0])

    def test_policy_job_reserves_live_window_instead_of_all_future_parents(self):
        # Fifty-four 4-pass cube groups have 5,832 GiB of logical canvases;
        # their bounded scheduler window remains 256 GiB on both disk and tmpfs.
        tasks = self.policy_tasks(groups=54, leases=4)
        for memory_backed in (False, True):
            with self.subTest(memory_backed=memory_backed), mock.patch.object(
                    memory, 'scratch_dir_is_memory_backed', return_value=memory_backed):
                self.assertEqual(memory.native_fullframe_dense_reserve(
                    tasks, total_dense_limit=256*GIB), 256*GIB)

    def test_policy_parent_accounting_deduplicates_leases_and_includes_confidence_tiles(self):
        tasks = self.policy_tasks(shape=(2, 3, 7), leases=7)
        for confidence, tiles, layers, canvases in ((0, False, False, 1),
                (0.2, False, False, 2), (0, True, False, 2),
                (0, True, True, 4), (0.2, True, True, 5)):
            with self.subTest(confidence=confidence, tiles=tiles, layers=layers):
                self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
                    total_dense_limit=10000, min_conf=confidence,
                    dense_tiling=tiles, nrrd_layers=layers), 2*3*7*4*canvases)

    def test_policy_group_must_fit_the_simultaneous_dense_window(self):
        for mode in ('file', 'direct_union'):
            with self.subTest(mode=mode):
                tasks = self.policy_tasks(mode=mode)
                self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
                    total_dense_limit=108*GIB), 108*GIB)
                with self.assertRaisesRegex(RuntimeError, 'group requires 108.0 GiB.*107.0 GiB'):
                    memory.native_fullframe_dense_reserve(tasks, total_dense_limit=107*GIB)

    def test_policy_marker_cannot_claim_a_bound_without_retirement(self):
        with self.assertRaisesRegex(RuntimeError, 'File-mode.*5832.0 GiB'):
            memory.native_fullframe_dense_reserve(self.policy_tasks(groups=54),
                total_dense_limit=256*GIB, bounded_retirement=False)

    def test_partial_policy_group_admission_is_rejected(self):
        tasks = self.policy_tasks(shape=(2, 3, 7))
        tasks[0]['augmentation_pass_tasks'][0].pop('bounded_parent_admission')
        with self.assertRaisesRegex(ValueError, 'admission for every sibling parent'):
            memory.native_fullframe_dense_reserve(tasks, total_dense_limit=10000)

    def test_shared_policy_groups_preserve_window_with_disk_or_ram_parent_backings(self):
        tasks = self.policy_tasks(groups=54, leases=4, mode='direct_union')
        for memfd in (False, True):
            with self.subTest(memfd=memfd), mock.patch.object(memory,
                    'memfd_workspace_enabled', return_value=memfd):
                self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
                    total_dense_limit=256*GIB), 256*GIB)

    def test_policy_group_must_have_one_storage_contract_and_complete_admission(self):
        for fault in ('mixed', 'partial', 'unmarked_shared', 'tile_sibling', 'unsupported'):
            tasks = self.policy_tasks(shape=(2,3,7), mode='direct_union')
            base, copy = tasks[0], tasks[0]['augmentation_pass_tasks'][0]
            if fault == 'mixed':
                copy['result_mode'] = 'file'
            elif fault == 'partial':
                copy.pop('bounded_parent_admission')
            elif fault == 'unmarked_shared':
                for parent in (base, *base['augmentation_pass_tasks']):
                    parent.pop('bounded_parent_admission')
            elif fault == 'tile_sibling':
                copy['kind'] = 'tile'
            else:
                copy['result_mode'] = 'd1_owner'
            with self.subTest(fault=fault):
                with self.assertRaises(ValueError):
                    memory.policy_parent_memory_plan(tasks, requested_dense_limit=10000,
                        available_ram_bytes=8*GIB, source_shape=(2,3,7))
                # Unmarked ordinary shared tasks retain their existing accounting
                # contract, but never enter the policy-specific physical planner.
                if fault != 'unmarked_shared':
                    with self.assertRaises(ValueError):
                        memory.native_fullframe_dense_reserve(tasks, total_dense_limit=10000)

    def test_policy_group_cannot_repeat_a_sibling_canvas(self):
        tasks = self.policy_tasks(mode='direct_union')
        tasks[0]['augmentation_pass_tasks'].append(tasks[0]['augmentation_pass_tasks'][0])
        with self.assertRaisesRegex(ValueError, 'repeats a sibling parent'):
            memory.native_fullframe_dense_reserve(tasks, total_dense_limit=256*GIB)

    def test_policy_and_shared_parents_reserve_one_common_window(self):
        tasks = self.policy_tasks(groups=2)
        tasks.append(dict(kind='fullframe', model_name='model',
            view=SimpleNamespace(name='shared'), processing_shape=(3072,3072,3072),
            result_mode='direct_union'))
        self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
            total_dense_limit=200*GIB), 200*GIB)

    def test_policy_parent_cannot_be_charged_under_two_different_groups(self):
        tasks = self.policy_tasks(groups=2, shape=(2, 3, 7))
        tasks[1]['augmentation_pass_tasks'][0] = tasks[0]['augmentation_pass_tasks'][0]
        with self.assertRaisesRegex(ValueError, 'inconsistent admission groups'):
            memory.native_fullframe_dense_reserve(tasks, total_dense_limit=10000)

    def test_policy_leases_cannot_disagree_about_parent_geometry(self):
        tasks = self.policy_tasks(shape=(2, 3, 7), leases=2)
        tasks[1]['processing_shape'] = (3, 3, 7)
        with self.assertRaisesRegex(ValueError, 'inconsistent tasks'):
            memory.native_fullframe_dense_reserve(tasks, total_dense_limit=10000)

    def test_physical_policy_plan_reserves_results_support_and_final_work(self):
        plan = memory.policy_parent_memory_plan(self.policy_tasks(groups=54),
            requested_dense_limit=256*GIB, available_ram_bytes=1024*GIB,
            source_shape=(3072,3072,3072), output_reserve_bytes=32*GIB,
            parent_transient_reserve_bytes=192*GIB, worker_buffer_reserve_bytes=4*GIB)
        self.assertEqual(plan['dense_limit_bytes'], 256*GIB)
        self.assertEqual(plan['file_result_reserve_bytes'], 256*GIB)
        self.assertEqual(plan['source_topology_reserve_bytes'], 135*GIB)
        self.assertGreater(plan['coverage_reserve_bytes'], 48*GIB)
        self.assertLessEqual(plan['total_reserve_bytes'], 1024*GIB)

    def test_shared_policy_outputs_do_not_reserve_a_second_parent_canvas(self):
        kwargs = dict(requested_dense_limit=256*GIB, available_ram_bytes=1024*GIB,
            source_shape=(3072,3072,3072), output_reserve_bytes=32*GIB,
            parent_transient_reserve_bytes=192*GIB, worker_buffer_reserve_bytes=4*GIB)
        fallback = memory.policy_parent_memory_plan(self.policy_tasks(groups=54), **kwargs)
        shared = memory.policy_parent_memory_plan(
            self.policy_tasks(groups=54, mode='direct_union'), **kwargs)
        self.assertEqual(fallback['dense_limit_bytes'], shared['dense_limit_bytes'])
        self.assertEqual(fallback['file_result_reserve_bytes'], 256*GIB)
        self.assertEqual(shared['file_result_reserve_bytes'], 0)
        self.assertEqual(shared['file_result_ratio_numerator'], 0)
        self.assertEqual(fallback['coverage_reserve_bytes'], shared['coverage_reserve_bytes'])
        self.assertEqual(fallback['fixed_reserve_bytes'], shared['fixed_reserve_bytes'])
        self.assertEqual(fallback['total_reserve_bytes'] - shared['total_reserve_bytes'], 256*GIB)
        self.assertLessEqual(shared['total_reserve_bytes'], kwargs['available_ram_bytes'])

    def test_low_ram_shrinks_policy_window_before_any_parent_allocation(self):
        tasks = self.policy_tasks(groups=54)
        plan = memory.policy_parent_memory_plan(tasks, requested_dense_limit=256*GIB,
            available_ram_bytes=300*GIB, source_shape=(3072,3072,3072),
            output_reserve_bytes=32*GIB, parent_transient_reserve_bytes=64*GIB,
            worker_buffer_reserve_bytes=4*GIB)
        self.assertLess(plan['dense_limit_bytes'], 108*GIB)
        self.assertLessEqual(plan['total_reserve_bytes'], 300*GIB)
        with self.assertRaisesRegex(RuntimeError, 'group requires 108.0 GiB'):
            memory.native_fullframe_dense_reserve(tasks,
                total_dense_limit=plan['dense_limit_bytes'])

    def test_tiny_policy_smoke_does_not_inherit_a_32_gib_fixed_margin(self):
        tasks = self.policy_tasks(shape=(8,8,8))
        plan = memory.policy_parent_memory_plan(tasks, requested_dense_limit=256*GIB,
            available_ram_bytes=8*GIB, source_shape=(8,8,8),
            parent_transient_reserve_bytes=2*GIB)
        self.assertEqual(plan['safety_reserve_bytes'], GIB//4)
        self.assertLessEqual(plan['total_reserve_bytes'], 8*GIB)
        self.assertEqual(memory.native_fullframe_dense_reserve(tasks,
            total_dense_limit=plan['dense_limit_bytes']), 4*8**3)

    def test_policy_support_budget_uses_raster_dimensions_instead_of_parent_ratio(self):
        tasks = self.policy_tasks(shape=(32,8,8))
        kwargs = dict(requested_dense_limit=256*GIB, available_ram_bytes=16*GIB,
                      source_shape=(32,8,8))
        normal = memory.policy_parent_memory_plan(tasks, **kwargs)
        for task in (tasks[0], *tasks[0]['augmentation_pass_tasks']):
            task['out_size'] = 64
        large_raster = memory.policy_parent_memory_plan(tasks, **kwargs)
        self.assertLess(large_raster['dense_limit_bytes'], normal['dense_limit_bytes']//4)
        self.assertGreater(large_raster['coverage_reserve_bytes'],
                           12*large_raster['dense_limit_bytes'])
        self.assertLessEqual(large_raster['total_reserve_bytes'], 16*GIB)
        for task in (tasks[0], *tasks[0]['augmentation_pass_tasks']):
            task['augmentation_settings'] = SimpleNamespace(coverage='none')
        disabled = memory.policy_parent_memory_plan(tasks, **kwargs)
        self.assertEqual(disabled['coverage_reserve_bytes'], 0)
        self.assertGreater(disabled['dense_limit_bytes'], normal['dense_limit_bytes'])

    def test_unknown_or_insufficient_ram_cannot_admit_a_policy_parent(self):
        tasks = self.policy_tasks(shape=(8,8,8))
        for available in (0, 1024):
            with self.subTest(available=available):
                plan = memory.policy_parent_memory_plan(tasks, requested_dense_limit=256*GIB,
                    available_ram_bytes=available, source_shape=(8,8,8))
                self.assertEqual(plan['dense_limit_bytes'], 0)
                with self.assertRaisesRegex(RuntimeError, 'dense admission limit'):
                    memory.native_fullframe_dense_reserve(tasks, total_dense_limit=0)

    def test_azimuthal_tail_seam_buffers_are_included_in_physical_plan(self):
        from XTA.geometry import ViewInfo
        tasks = self.policy_tasks(shape=(5,8,8))
        base = tasks[0]
        for index, task in enumerate((base, *base['augmentation_pass_tasks'])):
            task['view'] = ViewInfo(name=f'azimuthal_transverse__policy{index}',
                family='azimuthal', num_slices=5, src_h=8, src_w=8, pad_mode='clamp')
            task['slice_start'] = 4
            task['slice_count'] = 1
        plan = memory.policy_parent_memory_plan(tasks, requested_dense_limit=GIB,
            available_ram_bytes=16*GIB, source_shape=(5,8,8), batch_size=4)
        self.assertGreater(plan['file_result_reserve_bytes'], plan['dense_limit_bytes'])
        self.assertGreater(plan['coverage_reserve_bytes'], plan['dense_limit_bytes']//2)
        self.assertLessEqual(plan['total_reserve_bytes'], 16*GIB)
        for task in (base, *base['augmentation_pass_tasks']):
            task['result_mode'] = 'direct_union'
        shared = memory.policy_parent_memory_plan(tasks, requested_dense_limit=GIB,
            available_ram_bytes=16*GIB, source_shape=(5,8,8), batch_size=4)
        self.assertEqual(shared['dense_limit_bytes'], GIB)
        self.assertEqual(shared['file_result_reserve_bytes'], 3*GIB)
        self.assertEqual(plan['file_result_reserve_bytes'], 4*GIB)
        self.assertEqual(shared['coverage_reserve_bytes'], plan['coverage_reserve_bytes'])
        self.assertLessEqual(shared['total_reserve_bytes'], 16*GIB)

    def test_same_named_memfds_cannot_share_cached_layer_bytes(self):
        # Linux /proc/<pid>/fd/N resolves different, equally named memfds to the
        # same /memfd:name (deleted) display string. Emulate only that resolution;
        # use real files and mmap acquisition/refcounts on every platform.
        resolve = Path.resolve
        def memfd_display(path, *args, **kwargs):
            if path.name == 'chunks.bin':
                return Path('/memfd:xta-packed-publication (deleted)')
            return resolve(path, *args, **kwargs)
        for payloads in ((b'abcdefgh', b'ABCDEFGH'), (b'abcdefgh', b'0123456789abcdef')):
            with self.subTest(lengths=tuple(map(len, payloads))), tempfile.TemporaryDirectory() as td:
                paths = [Path(td)/str(i)/'chunks.bin' for i in range(2)]
                for path, data in zip(paths, payloads):
                    path.parent.mkdir(); path.write_bytes(data)
                acquired = []
                with mock.patch.object(Path, 'resolve', memfd_display):
                    try:
                        a, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[0])
                        acquired.append(paths[0]); self.assertFalse(reused)
                        b, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[1])
                        acquired.append(paths[1]); self.assertFalse(reused)
                        self.assertIsNot(a, b)
                        self.assertEqual(bytes(a), payloads[0])
                        self.assertEqual(bytes(b), payloads[1])
                        again, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[0])
                        acquired.append(paths[0]); self.assertTrue(reused)
                        self.assertIs(again, a)
                        # Rewriting one logical layer must not evict another layer.
                        interpolation._invalidate_raw_store_chunks_ram_cache(paths[0])
                        self.assertEqual(bytes(b), payloads[1])
                    finally:
                        for path in reversed(acquired):
                            interpolation._release_raw_store_chunks_ram_cache(path)

    def test_cached_payload_can_retire_after_proc_target_disappears(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'chunks.bin'; path.write_bytes(b'owned bytes')
            with mock.patch.object(Path, 'resolve', return_value=Path('/memfd:shared (deleted)')):
                payload, _reused = interpolation._acquire_raw_store_chunks_ram_cache(path)
            try:
                with mock.patch.object(Path, 'resolve', side_effect=FileNotFoundError('owner retired')):
                    interpolation._release_raw_store_chunks_ram_cache(path)
                self.assertTrue(payload.closed)
            finally:
                # Also clean up the intentionally broken implementation in the
                # red regression, whose stale key otherwise keeps Windows mmap open.
                with mock.patch.object(Path, 'resolve', return_value=Path('/memfd:shared (deleted)')):
                    interpolation._invalidate_raw_store_chunks_ram_cache(path)

    def test_concurrent_cached_nrrds_with_same_memfd_display_name(self):
        volumes = _cache_export_volumes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i,a in enumerate(volumes):
                writer = IncrementalRawBBoxMaskStoreWriter(shape=a.shape, store_dir=root/f'store{i}',
                    format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='cached-export fixture')
                try:
                    writer.consume(0,a); writer.finalize()
                except BaseException:
                    writer.discard(); raise
            # First two payloads have identical length but different pixel bytes.
            self.assertEqual((root/'store0/chunks.bin').stat().st_size, (root/'store1/chunks.bin').stat().st_size)
            _verify_cached_exports_in_subprocess(root, emulate_memfd_resolution=True)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_multiple_parent_memfds_survive_producer_exit_and_cached_exports(self):
        import subprocess
        volumes = _cache_export_volumes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks = [dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                          d1_store_dir=str(root/f'store{i}'), d1_output_shape=list(a.shape),
                          model_name='m', view=SimpleNamespace(name=f'radial-{i}'))
                     for i,a in enumerate(volumes)]
            with mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                    mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                    mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES':'1','YOLO_TTA_PUBLICATION_RAM':'1',
                                                'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1','YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
                memory.plan_native_publication_memory(tasks, keep_temp=False, worker_count=1,
                                                      publication_pending=1, unpack_bytes=1024)
            program = '''
import json,sys
from pathlib import Path
from tests.test_publication_memory import _cache_export_volumes
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter,INTERNAL_PACKED_CVOL_FORMAT
for task,a in zip(json.loads(sys.argv[1]), _cache_export_volumes()):
    w=IncrementalRawBBoxMaskStoreWriter(shape=a.shape,store_dir=Path(task['d1_store_dir']),format_name=INTERNAL_PACKED_CVOL_FORMAT,desc='child multi-layer',payload_backing=task['d1_memory_payload_path'])
    try:
        w.consume(0,a); assert w.finalize()['payload_backing']=='planned_memfd'
    except BaseException:
        w.discard(); raise
'''
            try:
                subprocess.run([sys.executable, '-c', program,
                    json.dumps([{k:v for k,v in t.items() if k!='view'} for t in tasks])], check=True)
                _verify_cached_exports_in_subprocess(root, emulate_memfd_resolution=False)
            finally:
                self.assertEqual(release_memfd_owners_under(root), len(volumes))

    def test_large_codec_windows_and_mirror_canvases_reduce_retention_budget(self):
        sink = SimpleNamespace(max_workers=12, output_shape=(1931,3064,3022),
                low_quality_specs=[SimpleNamespace(output_shape_t_y_x=(388,612,604))])
        small = memory.publication_output_reserve(sink, 512*1024**2, 16*1024**2)
        huge = memory.publication_output_reserve(sink, 16*GIB, 16*1024**2)
        self.assertGreater(huge, small)
        normal = memory.retained_payload_plan([sink.output_shape]*50, 950*GIB, 4, 12, 256*1024**2,
                                               output_reserve_bytes=small)
        limited = memory.retained_payload_plan([sink.output_shape]*50, 950*GIB, 4, 12, 256*1024**2,
                                                output_reserve_bytes=huge)
        self.assertGreater(normal['budget_bytes'], limited['budget_bytes'])

    def test_small_layers_cannot_exhaust_parent_file_descriptors(self):
        tasks = [dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                      d1_store_dir=f'not-created-store-{i}', d1_output_shape=[2,4,16],
                      model_name='m', view=SimpleNamespace(name=f'radial-{i}')) for i in range(10)]
        resource = SimpleNamespace(RLIMIT_NOFILE=7, getrlimit=lambda _which: (80,80))
        with mock.patch.dict(sys.modules, {'resource':resource}), \
                mock.patch.object(memory, 'memfd_workspace_enabled', return_value=True), \
                mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                mock.patch.object(memory.os, 'listdir', return_value=[]), \
                mock.patch.object(memory.os, 'memfd_create', create=True, side_effect=range(10,20)) as create, \
                mock.patch.object(memory, '_register_memfd_owner'), \
                mock.patch.dict(os.environ, {'YOLO_TTA_PUBLICATION_RAM':'1',
                                            'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1','YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
            plan = memory.plan_native_publication_memory(tasks, keep_temp=False, worker_count=1,
                                                        publication_pending=1, unpack_bytes=1024)
        self.assertEqual(create.call_count, 8)
        self.assertEqual(sum('d1_memory_payload_path' in t for t in tasks), 8)
        self.assertEqual(plan['grants'][-2:], [0,0])
        self.assertEqual(plan['reserved_bytes'], sum(plan['grants']))

    def test_production_plan_charges_every_future_layer_and_reserves_tail(self):
        shape = (1931, 3064, 3022)
        plan = memory.retained_payload_plan([shape]*50, 950*GIB, 4, 12, 256*1024**2)
        self.assertEqual(len([v for v in plan['grants'] if v]), 50)
        self.assertEqual(sum(plan['grants']), plan['reserved_bytes'])
        self.assertLessEqual(plan['reserved_bytes'], plan['budget_bytes'])
        self.assertGreater(plan['reserve_bytes'], 200*GIB)
        self.assertLessEqual(plan['reserved_bytes'] + plan['reserve_bytes'], 950*GIB)

    def test_small_job_limit_and_explicit_cap_spill_instead_of_overcommitting(self):
        shapes = [(1931, 3064, 3022)]*500
        low = memory.retained_payload_plan(shapes, 100*GIB, 4, 12, 256*1024**2)
        self.assertEqual(sum(low['grants']), 0)
        bounded = memory.retained_payload_plan(shapes, 950*GIB, 4, 12, 256*1024**2, cap=8*GIB)
        self.assertLessEqual(sum(bounded['grants']), 8*GIB)
        self.assertTrue(any(bounded['grants']))
        self.assertTrue(any(v == 0 for v in bounded['grants']))

    def test_swap_is_never_counted_as_ram_headroom(self):
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={'MemAvailable': 10, 'SwapFree': 1000}), \
                mock.patch.object(memory, 'available_anon_work_bytes', return_value=1010):
            self.assertEqual(memory.publication_ram_headroom(), 10)
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={'MemAvailable': 1000}), \
                mock.patch.object(memory, 'available_anon_work_bytes', return_value=20):
            self.assertEqual(memory.publication_ram_headroom(), 20)

    def test_windows_headroom_uses_physical_available_without_swap(self):
        psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1234),
                                 swap_memory=lambda: SimpleNamespace(free=987654321))
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={}), \
                mock.patch.object(memory.os, 'name', 'nt'), \
                mock.patch.dict(sys.modules, {'psutil': psutil}), \
                mock.patch.object(memory, 'available_anon_work_bytes', return_value=99999999):
            self.assertEqual(memory.publication_ram_headroom(), 1234)
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={}), \
                mock.patch.object(memory.os, 'name', 'nt'), \
                mock.patch.dict(sys.modules, {'psutil': None}):
            self.assertEqual(memory.publication_ram_headroom(), 0)

    def test_retained_scratch_never_uses_ephemeral_parent_descriptors(self):
        with mock.patch.object(memory.os, 'memfd_create', create=True) as create:
            self.assertIsNone(memory.plan_native_publication_memory([], keep_temp=True,
                              worker_count=4, publication_pending=12, unpack_bytes=256*1024**2))
            create.assert_not_called()

    def test_private_payload_spill_preserves_bytes_and_releases_original_backing(self):
        volume = np.random.default_rng(13).integers(0, 2, (5, 17, 43), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backing = root/'parent-backed-payload'
            backing.touch()
            # A hardlink stands in for the Linux /proc symlink on Windows. Both
            # descriptors refer to the same payload and replacement preserves it.
            def link(path, target, **kwargs):
                os.link(target, path)
            original_open = os.open
            def share_delete_open(path, flags, mode=0o666):
                if os.name != 'nt':
                    return original_open(path, flags, mode)
                # Linux permits replacing open files. Give the Windows fixture
                # that same sharing behavior; production RAM backings are Linux.
                import _winapi
                import msvcrt
                disposition = 1 if flags & os.O_EXCL else (2 if flags & os.O_TRUNC else (4 if flags & os.O_CREAT else 3))
                handle = _winapi.CreateFile(str(path), 0xc0000000, 7, 0, disposition, 0, 0)
                return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(os, 'open', share_delete_open))
                stack.enter_context(mock.patch.object(Path, 'symlink_to', link))
                writer = IncrementalRawBBoxMaskStoreWriter(shape=volume.shape, store_dir=root/'store',
                    format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='spill-test', payload_backing=str(backing))
                stack.callback(writer.discard)
                writer.consume(0, volume[:2])
                with mock.patch.object(writer, '_pwrite_all', side_effect=OSError('disk full')):
                    with self.assertRaisesRegex(OSError, 'disk full'):
                        writer.spill_payload_to_disk()
                self.assertTrue(writer._ram_payload)
                self.assertGreater(backing.stat().st_size, 0)
                writer.spill_payload_to_disk()
                self.assertEqual(backing.stat().st_size, 0)
                writer.consume(2, volume[2:])
                self.assertEqual(writer.finalize()['payload_backing'], 'disk')
                reader = RawBBoxMaskStore.open(writer.store_dir, mmap_payload=True)
                try:
                    np.testing.assert_array_equal(np.stack([reader.decode_slice(z) for z in range(5)]), volume)
                finally:
                    reader.close()

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_parent_descriptor_survives_spawned_producer_and_retires(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view = SimpleNamespace(name='radial-test')
            task = dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                        d1_store_dir=str(root/'store'), d1_output_shape=[3, 5, 43], model_name='m', view=view)
            with mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                    mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                    mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES':'1', 'YOLO_TTA_PUBLICATION_RAM':'1',
                                                'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1', 'YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
                memory.plan_native_publication_memory([task], keep_temp=False, worker_count=1,
                                                      publication_pending=1, unpack_bytes=1024)
            program = '''
import sys,json,numpy as np
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter,INTERNAL_PACKED_CVOL_FORMAT
from pathlib import Path
t=json.loads(sys.argv[1]); a=np.arange(3*5*43).reshape(3,5,43)%2
w=IncrementalRawBBoxMaskStoreWriter(shape=a.shape,store_dir=Path(t['d1_store_dir']),format_name=INTERNAL_PACKED_CVOL_FORMAT,desc='child',payload_backing=t['d1_memory_payload_path'])
w.consume(0,a); assert w.finalize()['payload_backing']=='planned_memfd'
'''
            try:
                subprocess.run([sys.executable, '-c', program, json.dumps({k: v for k, v in task.items() if k != 'view'})], check=True)
                reader = RawBBoxMaskStore.open(root/'store', mmap_payload=True)
                try:
                    np.testing.assert_array_equal(np.stack([reader.decode_slice(z) for z in range(3)]),
                                                  np.arange(3*5*43).reshape(3,5,43)%2)
                finally:
                    reader.close()
            finally:
                self.assertEqual(release_memfd_owners_under(root), 1)


if __name__ == '__main__':
    unittest.main()
