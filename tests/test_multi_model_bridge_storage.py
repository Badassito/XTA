"""Concurrent models must retain their own bridge artifacts until publication."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import gc
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from XTA import assembly, finalization
from XTA.geometry import ViewInfo
from XTA.interpolation import CVOL_FORMAT, NrrdLayerRef, RawBBoxMaskStore, write_raw_bbox_mask_store


class MultiModelBridgeStorageTests(unittest.TestCase):
    def _run_concurrent_models(self, *, tile=False, delta_only=False):
        shape = (3, 4, 5)
        view = ViewInfo(name='transverse', physical_view_name='transverse',
                        num_slices=shape[0], src_h=shape[1], src_w=shape[2], pad_mode='clamp')
        first_written = threading.Event()
        both_written = threading.Barrier(2)
        deltas = {}
        requests = {}
        for index, model in enumerate(('first', 'second')):
            delta = np.zeros(shape, dtype=np.uint8)
            delta[1, 1, index + 2] = 1
            deltas[model] = delta

        def interpolate(**kwargs):
            model = next(name for name in deltas if name in kwargs['work_dir'].parts)
            requests[model] = kwargs
            delta = deltas[model]
            kwargs['mask_mm'] |= delta
            if model == 'second' and not first_written.wait(timeout=10):
                raise RuntimeError('first model did not publish its bridge fixture')
            stats = dict(added_voxels=int(delta.sum()), bridge_component_deltas=[])
            if kwargs.get('bridge_delta_path') is not None:
                path = kwargs['bridge_delta_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(delta.tobytes())
                stats['bridge_delta_path'] = str(path)
            if kwargs['bridge_component_dir'] is not None:
                path = kwargs['bridge_component_dir'] / 'walkback01_candidate01.cvol'
                write_raw_bbox_mask_store(delta, path, format_name=CVOL_FORMAT,
                                          desc='concurrent model bridge fixture', workers=1)
                stats['bridge_component_deltas'].append(dict(
                    path=str(path), walk_back_index=1, candidate_index=1,
                    added_voxels=int(delta.sum())))
            if model == 'first':
                first_written.set()
            # Both writers finish before either publication reads its artifact.
            # A shared path deterministically exposes the second model's data.
            both_written.wait(timeout=10)
            return kwargs['mask_mm'], stats

        def materialize_component(path, **kwargs):
            store = RawBBoxMaskStore.open(path)
            try:
                data = np.stack([store.decode_slice(index) for index in range(shape[0])])
            finally:
                store.close()
            return NrrdLayerRef(key=kwargs['model_name'], name='bridge', path=path,
                                shape=shape, model_name=kwargs['model_name'],
                                view_name=view.name, live_array=data)

        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(assembly, 'allocate_workspace_array',
                                  new=lambda **kwargs: np.zeros(kwargs['shape'], dtype=kwargs['dtype'])), \
                mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace',
                                  new=lambda *_args, **_kwargs: None), \
                mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process', new=interpolate), \
                mock.patch.object(assembly, 'materialize_nrrd_view_layer', new=lambda *_args, **_kwargs: None), \
                mock.patch.object(assembly, 'materialize_interpolation_component_nrrd_view_layer',
                                  new=materialize_component), \
                mock.patch.object(finalization, 'union_volume_into_volume',
                                  new=lambda dst, src, **_kwargs: np.bitwise_or(dst, src, out=dst)):
            root = Path(directory)

            def run_model(model):
                volume = np.zeros(shape, dtype=np.uint8)
                volume[0, 0, 0] = 1
                shared = dict(model_name=model, view=view, temp_dir=root,
                              interpolate=2, interpolation_walk_back=1,
                              interpolation_candidates=1, interpolate_passes=1,
                              interpolate_min_radius=0, interpolation_search_angle=0,
                              keep_temp=True, slice_workers=1, interpolation_task_workers=1,
                              nrrd_layers_enabled=not delta_only)
                if tile:
                    result = assembly.finalize_consolidated_tile_volume_for_parent(
                        tile_accumulator_mm=volume, destination_mm=np.zeros_like(volume),
                        destination_lock=threading.Lock(), config_id='tiles', **shared)
                else:
                    result = assembly.prepare_view_volume_after_fullframe(
                        union_mm=volume, confmap_mm=None, union_path=root / model / 'union.dat',
                        confmap_path=None, dense_tiling_active=False, min_conf=0, min_radius=0,
                        precleaned_slice_cleanup=True, hole_fill_done_on_device=True,
                        preinterpolation_layer_already_published=True, **shared)
                data = (np.array(result.final_view_volume_mm, copy=True)
                        if delta_only else result.nrrd_layers[0].live_array)
                if not tile:
                    assembly.close_memmap_array(result.final_view_volume_mm)
                return data

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {name: executor.submit(run_model, name) for name in deltas}
                actual = {name: future.result(timeout=20) for name, future in futures.items()}
            for model, data in actual.items():
                np.testing.assert_array_equal(data, deltas[model])
            path_key = 'bridge_delta_path' if delta_only else 'bridge_component_dir'
            self.assertNotEqual(requests['first'][path_key], requests['second'][path_key])
            del actual, requests, futures
            gc.collect()

    def test_concurrent_fullframe_component_publication_keeps_model_voxels(self):
        self._run_concurrent_models()

    def test_concurrent_d1_delta_continuations_keep_model_voxels(self):
        self._run_concurrent_models(delta_only=True)

    def test_concurrent_tile_component_publication_keeps_model_voxels(self):
        self._run_concurrent_models(tile=True)


if __name__ == '__main__':
    unittest.main()
