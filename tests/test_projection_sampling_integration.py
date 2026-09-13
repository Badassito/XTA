"""Coverage policy admission, planning identities, routing, and replay contracts."""
from __future__ import annotations

import ast
import contextlib
from dataclasses import asdict, replace
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import component_replay, config, geometry as g, pipeline
from XTA.azimuthal_coverage import requires_native_pull
from XTA.interpolation import write_raw_bbox_mask_store
from XTA.unification.context import activate_unified_launch
from XTA.unification.runtime import compile_physical_views
from XTA.unification.sampling import build_forward_raster_plan
from XTA.unification.tta_manifest import (
    build_tta_run_manifest,
    projection_sampling_record,
    radial_view_plan_metadata,
    spherical_view_plan_metadata,
)


def compiled(shape=(17, 19, 21), *, size=32, azimuthal=('transverse',),
             radial=(), spherical=(), tilted=(), **kwargs):
    return compile_physical_views(
        t_dim=shape[0], height=shape[1], width=shape[2],
        cartesian_views=(), tilted_groups=tilted,
        azimuthal_requests=config.resolve_azimuthal_view_requests(azimuthal),
        azimuthal_native_raster=size,
        radial_requests=config.resolve_radial_view_requests(radial),
        radial_patch_size=size,
        spherical_requests=config.resolve_spherical_view_requests(spherical),
        spherical_patch_size=size,
        **kwargs,
    )


def _assigned(node, name, value=None):
    return (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and (value is None or isinstance(node.value, ast.Constant) and node.value.value == value))


class ProjectionSamplingIntegrationTests(unittest.TestCase):
    def setUp(self):
        source_mode = mock.patch.dict(os.environ, {'YOLO_TTA_AZIMUTHAL_SOURCE_MODE': 'texture_linear'})
        source_mode.start()
        self.addCleanup(source_mode.stop)

    def test_shared_compiler_defaults_remain_identical_to_explicit_dense(self):
        arguments = dict(radial=('transverse',), spherical=('transverse',))
        default = compiled(**arguments)
        dense = compiled(**arguments, sampling_policy='dense')
        self.assertEqual(default, dense)
        for view in default.views:
            self.assertEqual(view.sampling_policy, 'dense')
            self.assertEqual(projection_sampling_record(view), {})
            if view.family == 'radial':
                self.assertEqual(view.radial_global_count, 0)

    def test_3072_counts_and_compiled_azimuth_metadata_match_actual_geometry(self):
        arguments = dict(shape=(3072,) * 3, size=3072,
                         radial=('transverse',), spherical=('transverse',))
        dense = compiled(**arguments, sampling_policy='dense')
        coverage = compiled(**arguments, sampling_policy='coverage')
        expected = {'spherical': (31032, 6402), 'radial': (2969, 1726), 'azimuthal': (4826, 2805)}
        for family, counts in expected.items():
            with self.subTest(family=family):
                self.assertEqual(tuple(sum(view.num_slices for view in result.views if view.family == family)
                                       for result in (dense, coverage)), counts)
                self.assertTrue(all(view.sampling_certificate for view in coverage.views if view.family == family))
        azimuthal = next(view for view in coverage.views if view.family == 'azimuthal')
        self.assertEqual(coverage.azimuthal_azimuth_angles,
                         (azimuthal.azimuths_deg[1] - azimuthal.azimuths_deg[0],))
        self.assertEqual(tuple(g.build_azimuthal_azimuths(coverage.azimuthal_azimuth_angles[0])),
                         azimuthal.azimuths_deg)

    def test_single_frame_azimuth_metadata_reconstructs_its_actual_vector(self):
        result = compiled(shape=(1, 1, 1), size=1, sampling_policy='coverage')
        self.assertEqual(result.views[0].azimuths_deg, (0.0,))
        self.assertEqual(tuple(g.build_azimuthal_azimuths(result.azimuthal_azimuth_angles[0])),
                         result.views[0].azimuths_deg)

    def test_explicit_azimuth_spacing_is_preserved_even_if_numerically_equal_to_auto(self):
        diameter = 19
        auto_step = 360.0 / (np.pi * diameter)
        for step in (0.25, 60.0, auto_step):
            request = (f'transverse:{step!r}',)
            dense = compiled(azimuthal=request, sampling_policy='dense')
            coverage = compiled(azimuthal=request, sampling_policy='coverage')
            with self.subTest(step=step):
                self.assertEqual(dense, coverage)
                self.assertEqual(coverage.azimuthal_azimuth_angles, (step,))
                self.assertFalse(requires_native_pull(coverage.views[0]))

    def test_auto_fallback_retains_geometry_for_compact_tilted_and_large_sources(self):
        cases = (
            dict(shape=(65, 67, 69), size=32),
            dict(shape=(4097, 19, 21), size=8192),
            dict(azimuthal=('tilted_transverse',),
                 tilted=(config.TiltedViewGroup(('transverse',), (30.0,), ('vertical',)),)),
        )
        for arguments in cases:
            dense = compiled(**arguments, sampling_policy='dense')
            coverage = compiled(**arguments, sampling_policy='coverage')
            with self.subTest(arguments=arguments):
                self.assertEqual(dense.azimuthal_azimuth_angles, coverage.azimuthal_azimuth_angles)
                for before, after in zip(dense.views, coverage.views):
                    original, actual = asdict(before), asdict(after)
                    actual['sampling_reason'] = original['sampling_reason']
                    self.assertEqual(original, actual)
                    if after.family == 'azimuthal':
                        self.assertTrue(after.sampling_reason)
                        self.assertFalse(requires_native_pull(after))

    def test_tta_default_and_actual_zero_angle_admission(self):
        parser = config.build_argparser()
        arguments = ['--input', 'source.mkv', '--model', 'cpu:model.xml']
        self.assertEqual(parser.parse_args(arguments).projection_sampling, 'coverage')
        self.assertEqual(parser.parse_args(arguments + ['--projection_sampling', 'dense']).projection_sampling, 'dense')
        tree = ast.parse(inspect.getsource(pipeline._main_impl))
        assignment = next(node for node in ast.walk(tree) if _assigned(node, 'projection_sampling'))
        admission = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                         and {'projection_sampling', 'angles'} <= {
                             child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)})
        program = compile(ast.fix_missing_locations(ast.Module(body=[assignment, admission], type_ignores=[])),
                          '<projection-sampling-admission>', 'exec')
        for requested in ('dense', 'coverage'):
            for angles in ((0.0,), (360.0,), (-360.0,), (45.0,), (0.0, 45.0)):
                env = {'args': SimpleNamespace(projection_sampling=requested), 'angles': angles}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(program, env)
                expected = requested if any(angle % 360.0 == 0 for angle in angles) else 'dense'
                self.assertEqual(env['projection_sampling'], expected)

    def test_d1_and_hybrid_production_routes_bypass_only_coarsened_azimuthal(self):
        tree = ast.parse(inspect.getsource(pipeline._main_impl))
        hybrid = next(node for node in ast.walk(tree) if _assigned(node, 'hybrid_deferred'))
        routing = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                       and node.test.id == 'radial_owner'
                       and any(_assigned(statement, 'result_mode', 'd1_owner') for statement in node.body))
        program = compile(ast.fix_missing_locations(ast.Module(body=[hybrid, routing], type_ignores=[])),
                          '<coverage-result-routing>', 'exec')
        dense = compiled().views[0]
        coverage = compiled(sampling_policy='coverage').views[0]
        fallback = compiled(shape=(65, 67, 69), size=32, sampling_policy='coverage').views[0]
        for view, bypass in ((dense, False), (coverage, True), (fallback, False),
                             (replace(coverage, family='orthogonal'), False)):
            self.assertEqual(requires_native_pull(view), bypass)
            for cpu in (False, True):
                env = dict(view=view, kind='fullframe', v1613_d1_owner_active=True,
                           legacy_d1_model_eligible=True, worker_direct_union_active=cpu,
                           cpu_eligible=cpu, gpu_eligible=True, radial_owner=False,
                           azimuthal_parent_requires_seam_union=False, prefix='probe', chunk_idx=0,
                           gpu_worker_result_dir=Path('unused'), args=SimpleNamespace(min_conf=0.0),
                           requires_native_pull=requires_native_pull,
                           HYBRID_DEFERRED_RESULT_MODE=pipeline.HYBRID_DEFERRED_RESULT_MODE)
                exec(program, env)
                expected = ('direct_union' if cpu else 'file') if bypass else (
                    pipeline.HYBRID_DEFERRED_RESULT_MODE if cpu else 'd1_owner')
                with self.subTest(view=view.name, bypass=bypass, cpu=cpu):
                    self.assertEqual(env['result_mode'], expected)

    def test_plan_metadata_is_conditional_and_coverage_changes_fullframe_and_tile_identity(self):
        dense = compiled().views[0]
        coverage = compiled(sampling_policy='coverage').views[0]
        affine = g.build_affine(dense.name, dense.src_w, dense.src_h, 32, 0.0, dense.pad_mode)
        job = g.AugJob('a0', 0.0, Path('unused.json'), affine)
        dense_plan = g.build_fullframe_raster_plan(dense, job)
        coverage_plan = g.build_fullframe_raster_plan(coverage, job)
        old_metadata = {'runtime_view_id': dense.name, 'runtime_job_id': 'a0', 'runtime_kind': 'fullframe',
                        **radial_view_plan_metadata(dense), **spherical_view_plan_metadata(dense)}
        old_plan = build_forward_raster_plan(
            mode='tta', physical_view_id=g.physical_view_name(dense), angle_deg=0.0,
            channel_token='gray', channel_kind='gray', channel_count=1, channel_stride=1,
            channel_offsets=(0,), channel_direction='ascending', output_shape=(32, 32), metadata=old_metadata)
        self.assertEqual(dense_plan.digest, old_plan.digest)
        self.assertNotEqual(dense_plan.digest, coverage_plan.digest)
        self.assertNotIn('projection_sampling', dict(dense_plan.metadata))
        self.assertIn('projection_sampling', dict(coverage_plan.metadata))
        tile = SimpleNamespace(out_size=32, tile_size=16, tile_stride=8,
                               tile_id='tile0', config_id='tiles16_8', tile_x=0, tile_y=0)
        dense_tile = g.build_dense_tile_raster_plan(dense, tile)
        coverage_tile = g.build_dense_tile_raster_plan(coverage, tile)
        self.assertNotEqual(dense_tile.digest, coverage_tile.digest)
        self.assertNotIn('projection_sampling', dict(dense_tile.metadata))
        self.assertIn('projection_sampling', dict(coverage_tile.metadata))

    def test_manifest_records_resolved_certificates_and_dense_fallback_reason(self):
        requests = config.resolve_azimuthal_view_requests(('transverse',))
        results = compiled(sampling_policy='coverage')
        view = results.views[0]
        with activate_unified_launch(version='22.0.0', launcher='xta', mode='tta', mode_arguments=()) as launch:
            manifest = build_tta_run_manifest(
                launch_context=launch, pipeline_version='22.0.0', resolved_config={}, artifact_identities={},
                source_shape_tyx=(17, 19, 21), processing_shape_tyx=(17, 19, 21), fps=1.0,
                physical_views=results.views, inference_views=results.views, angles=(0.0,),
                channel_format=config.DEFAULT_CHANNEL_FORMAT, tile_configs=(),
                azimuthal_requests=requests, azimuthal_diameters=results.azimuthal_diameters,
                azimuthal_azimuth_angles=results.azimuthal_azimuth_angles,
                backend={}, forward_sampling={}, prediction_processing={}, requested_outputs=(), output_paths={})
        record = manifest['geometry']['physical_views'][0]['projection_sampling']
        self.assertEqual(record['certificate'], view.sampling_certificate)
        self.assertEqual(record['dense_reference_frames'], view.sampling_reference_frames)
        self.assertEqual(record['reference_scope'], 'view_trajectory')
        self.assertIn('positive native intensity interpolation support', record['coverage_basis'])
        group = manifest['geometry']['azimuthal_groups'][0]
        self.assertEqual(group['resolved_azimuth_angle_deg'], results.azimuthal_azimuth_angles[0])
        self.assertEqual(group['concrete_azimuth_vectors'][0]['azimuths_deg'], list(view.azimuths_deg))
        fallback = compiled(shape=(65, 67, 69), size=32, sampling_policy='coverage').views[0]
        fallback_record = projection_sampling_record(fallback)['projection_sampling']
        self.assertEqual(fallback_record['policy'], 'dense')
        self.assertFalse(fallback_record['certificate'])
        self.assertTrue(fallback_record['fallback_reason'])

    def test_zero_angle_padded_and_upscaled_model_grids_keep_azimuthal_support(self):
        # Reconstruct the direct renderer's composed identity affine and source
        # coordinates; these include half-pixel padding and upscale phases.
        for shape, size in (((16, 31, 33), 31), ((17, 32, 33), 48),
                            ((32, 17, 19), 32), ((33, 18, 19), 48)):
            view = compiled(shape=shape, size=size, sampling_policy='coverage').views[0]
            self.assertTrue(view.sampling_certificate)
            variant = g.expand_views_into_tta_variants((view,), (0.0,))[0]
            affine = g.build_aug_job_for_variant(variant, size, Path('unused')).aff
            oy, ox = np.indices((size, size), dtype=np.float64)
            ax = affine.M_out_to_src[0, 0] * ox + affine.M_out_to_src[0, 1] * oy + affine.M_out_to_src[0, 2]
            ay = affine.M_out_to_src[1, 0] * ox + affine.M_out_to_src[1, 1] * oy + affine.M_out_to_src[1, 2]
            valid = (ax > -1.0) & (ax < view.src_w) & (ay > -1.0) & (ay < view.src_h)
            if view.src_h > view.src_w:
                # The direct CUDA renderer retains these positive-weight fringe
                # samples and clamps them to the diameter endpoints before
                # mapping radius. Hard [0,D-1] clipping would lose both ends.
                self.assertTrue(np.any((ax[valid] < 0.0)))
                self.assertTrue(np.any((ax[valid] > view.src_w - 1)))
            line = -view.roi_radius + (2 * view.roi_radius) * np.clip(ax[valid], 0, view.src_w - 1) / (view.src_w - 1)
            self.assertEqual(float(line.min()), -view.roi_radius)
            self.assertEqual(float(line.max()), view.roi_radius)
            stack = np.clip(ay[valid], 0, view.src_h - 1)
            observed = np.zeros(shape, dtype=bool)
            for angle in view.azimuths_deg:
                theta = np.deg2rad(angle)
                px = view.center_x + line * np.cos(theta)
                py = view.center_y + line * np.sin(theta)
                lower = [np.floor(coordinate).astype(int) for coordinate in (stack, py, px)]
                for zs in (0, 1):
                    z = lower[0] + zs
                    for ys in (0, 1):
                        y = lower[1] + ys
                        for xs in (0, 1):
                            x = lower[2] + xs
                            positive = ((np.abs(stack - z) < 1) & (np.abs(py - y) < 1) & (np.abs(px - x) < 1)
                                        & (z >= 0) & (z < shape[0]) & (y >= 0) & (y < shape[1])
                                        & (x >= 0) & (x < shape[2]))
                            observed[z[positive], y[positive], x[positive]] = True
            y, x = np.ogrid[:shape[1], :shape[2]]
            roi = (x - view.center_x)**2 + (y - view.center_y)**2 <= (view.roi_radius + 0.5)**2
            self.assertTrue(np.all(observed[:, roi]))


class ProjectionSamplingReplayTests(unittest.TestCase):
    def setUp(self):
        source_mode = mock.patch.dict(os.environ, {'YOLO_TTA_AZIMUTHAL_SOURCE_MODE': 'texture_linear'})
        source_mode.start()
        self.addCleanup(source_mode.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(component_replay.configure_component_replay_capture, None)

    def capture(self, view):
        data = np.zeros((view.num_slices, view.src_h, view.src_w), dtype=np.uint8)
        data[0, 0, 0] = 1
        source = self.root / 'source.cvol'
        with contextlib.redirect_stdout(io.StringIO()):
            write_raw_bbox_mask_store(data, source, desc='sampling replay fixture', workers=1)
            component_replay.configure_component_replay_capture(self.root / 'captures', require_persistent=False)
            return component_replay.capture_component_projection(
                source, view=view, out_shape_tyx=(view.full_t, view.full_h, view.full_w),
                added_voxels=1, layer_metadata={})

    def rewrite(self, capture, remove_fields):
        path = capture / 'manifest.json'
        descriptor = json.loads(path.read_text(encoding='utf-8'))
        descriptor['view'] = {key: value for key, value in descriptor['view'].items() if not remove_fields(key)}
        descriptor['descriptor_sha256'] = component_replay._descriptor_digest(descriptor)
        path.write_text(json.dumps(descriptor), encoding='utf-8')

    def test_legacy_dense_replay_defaults_additive_sampling_and_augmentation_fields(self):
        view = compiled(shape=(5, 7, 9), size=9).views[0]
        captured = self.capture(view)
        self.rewrite(captured, lambda key: key.startswith(('sampling_', 'augmentation_')) or key == 'radial_global_count')
        self.assertEqual(component_replay.load_component_replay(captured).view, view)

    def test_certified_replay_preserves_metadata_and_rejects_partially_missing_policy(self):
        view = compiled(shape=(5, 7, 9), size=9, sampling_policy='coverage').views[0]
        captured = self.capture(view)
        self.assertEqual(component_replay.load_component_replay(captured).view, view)
        self.rewrite(captured, lambda key: key == 'sampling_policy')
        with self.assertRaisesRegex(ValueError, 'ViewInfo fields'):
            component_replay.load_component_replay(captured)


if __name__ == '__main__':
    unittest.main()
