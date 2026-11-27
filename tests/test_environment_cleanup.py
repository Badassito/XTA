from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


from XTA import geometry, inference, runtime, workspace


class EnvironmentCleanupTests(unittest.TestCase):
    def tearDown(self) -> None:
        workspace.configure_pipeline_modes(
            fast_bundle_active=False,
            d1_pipeline_active=False,
        )

    def test_default_scratch_is_output_temp_and_explicit_root_is_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            output = root / 'output'
            explicit_root = root / 'explicit'

            with mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_SCRATCH_PREFER_SHM': 'force',
                    'YOLO_TTA_SCRATCH_SHM_DIR': str(explicit_root),
                    'YOLO_TTA_SCRATCH_SHM_MIN_FREE_GIB': '0',
                },
                clear=False,
            ):
                default_scratch = runtime.choose_scratch_dir(None, output, 'scan')
                self.assertEqual(default_scratch, output / 'temp')

                explicit_scratch = runtime.choose_scratch_dir(
                    str(explicit_root), output, 'scan',
                )
                self.assertEqual(
                    explicit_scratch,
                    explicit_root / f'scan_{os.getpid()}_temp',
                )

    def test_fast_bundle_defaults_follow_process_local_resolved_state(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_V1613_BUNDLE_ACTIVE': '1',
                'YOLO_TTA_V1613_D1_PIPELINE_ACTIVE': '0',
            },
            clear=False,
        ):
            for name in (
                'YOLO_TTA_PROTO_HOLE_TREATMENT',
                'YOLO_TTA_PROTO_HOLE_RADIUS',
                'YOLO_TTA_GPU_UNION_RETIREMENT_LANES',
            ):
                os.environ.pop(name, None)
            workspace.configure_pipeline_modes(
                fast_bundle_active=False,
                d1_pipeline_active=True,
            )
            self.assertFalse(workspace.v1613_fast_bundle_active())
            self.assertFalse(workspace.v1613_d1_pipeline_active())
            self.assertEqual(workspace.proto_hole_treatment_mode(), 'off')
            self.assertEqual(workspace.proto_hole_treatment_radius(), 0)
            self.assertEqual(inference.gpu_union_retirement_lane_count(), 2)

            workspace.configure_pipeline_modes(
                fast_bundle_active=True,
                d1_pipeline_active=True,
                enable_yolo_hole_fill=True,
            )
            self.assertTrue(workspace.v1613_fast_bundle_active())
            self.assertTrue(workspace.v1613_d1_pipeline_active())
            self.assertEqual(workspace.proto_hole_treatment_mode(), 'close')
            self.assertEqual(workspace.proto_hole_treatment_radius(), 2)
            self.assertEqual(inference.gpu_union_retirement_lane_count(), 3)

    def test_consolidated_streaming_defaults_preserve_unset_behavior(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('YOLO_TTA_STREAMING_SOURCE_PREFETCH_FRAMES', None)
            self.assertEqual(geometry.streaming_prediction_source_prefetch_frames(64), 2048)

    def test_zero_staging_batches_disables_eager_cuda_staging(self) -> None:
        cfg = SimpleNamespace(device='cuda:0')
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with (
            mock.patch.object(inference, 'canonical_single_device', return_value='cuda:0'),
            mock.patch.dict(sys.modules, {'torch': fake_torch}),
            mock.patch.dict(os.environ, {'YOLO_TTA_GPU_INPUT_STAGING_BATCHES': '0'}),
        ):
            self.assertFalse(geometry.gpu_input_staging_enabled(cfg))
            os.environ['YOLO_TTA_GPU_INPUT_STAGING_BATCHES'] = '2'
            self.assertTrue(geometry.gpu_input_staging_enabled(cfg))

    def test_retired_runtime_toggles_preserve_default_behavior(self) -> None:
        with mock.patch.object(
            workspace, '_memory_sample',
            return_value=({'MemAvailable': 120, 'SwapFree': 30}, 80, None),
        ):
            self.assertEqual(workspace.available_anon_work_bytes(), 80)

if __name__ == '__main__':
    unittest.main()
