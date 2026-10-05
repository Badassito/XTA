from __future__ import annotations

import contextlib
import io
import unittest

from XTA.config import (
    SAVE_OPTION_TOKENS,
    build_argparser,
    resolve_backend_batches,
    resolve_backend_devices,
    resolve_backend_models,
    resolve_backend_precisions,
    resolve_channel_format,
    resolve_interpolation_settings,
    resolve_sam_devices,
    resolve_save_request,
    resolve_tta_angles,
)


class ConfigTests(unittest.TestCase):
    REQUIRED = ["--input", "input.mkv", "--model", "gpu:model.engine"]

    def test_semantic_task_and_output_are_independent(self) -> None:
        parser = build_argparser()
        self.assertEqual(parser.parse_args(self.REQUIRED).task, "segment")
        for task in ("segment", "semantic"):
            args = parser.parse_args([*self.REQUIRED, "--task", task, "--save", "semantic,binary"])
            self.assertEqual(args.task, task)
            self.assertEqual(resolve_save_request(args.save).options, ("semantic", "binary"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*self.REQUIRED, "--task", "detect"])

    def test_numeric_options_reject_nonfinite_values(self) -> None:
        parser = build_argparser()
        for option in (
            "--conf", "--min_conf", "--min_radius", "--interpolation_min_radius",
            "--interpolation_search_angle", "--centerline_radius_factor",
            "--centerline_timeout",
        ):
            for value in ("nan", "inf", "-inf", "1e309", "-1e309"):
                with self.subTest(option=option, value=value):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                        parser.parse_args([*self.REQUIRED, f"{option}={value}"])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(option, stderr.getvalue())
                    self.assertIn("must be finite", stderr.getvalue())

    def test_numeric_options_reject_out_of_range_values(self) -> None:
        parser = build_argparser()
        cases = {
            "--conf": ("-0.01", "1.01", "2"),
            "--min_conf": ("-0.01", "1.01"),
            "--imgsz": ("-1", "0", "1.5"),
            "--min_radius": ("-0.01",),
            "--centerline_filter_passes": ("-1",),
            "--centerline_radius_factor": ("0", "1"),
            "--centerline_temporal_context": ("-1",),
            "--centerline_surface_max_dim": ("0", "63"),
            "--centerline_surface_points": ("0", "999"),
            "--centerline_timeout": ("0", "-0.01"),
            "--interpolation_distance": ("-1",),
            "--sam_feature_cache_mib": ("-1", "1.5", "nan"),
            "--interpolation_walk_back": ("-1",),
            "--interpolation_candidates": ("-1", "0"),
            "--interpolation_passes": ("-1", "0"),
            "--interpolation_min_radius": ("-0.01",),
            "--interpolation_search_angle": ("-90", "90", "91"),
            "--capture_component_limit": ("-1", "0"),
        }
        for option, values in cases.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                        parser.parse_args([*self.REQUIRED, f"{option}={value}"])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(option, stderr.getvalue())

    def test_numeric_boundaries_preserve_disabled_and_minimum_settings(self) -> None:
        parser = build_argparser()
        values = {
            "imgsz": "1", "min_radius": "0", "min_conf": "0",
            "centerline_filter_passes": "0", "centerline_radius_factor": "1.01",
            "centerline_temporal_context": "0", "centerline_surface_max_dim": "64",
            "centerline_surface_points": "1000", "centerline_timeout": "0.01",
            "interpolation_distance": "0", "interpolation_walk_back": "0",
            "interpolation_candidates": "1", "interpolation_passes": "1",
            "interpolation_min_radius": "0", "capture_component_limit": "1",
        }
        for confidence in ("0", "1"):
            for angle in ("-89.99", "0", "89.99"):
                with self.subTest(confidence=confidence, angle=angle):
                    args = parser.parse_args([
                        *self.REQUIRED, f"--conf={confidence}",
                        f"--interpolation_search_angle={angle}",
                        *(f"--{name}={value}" for name, value in values.items()),
                    ])
                    for name, value in values.items():
                        self.assertEqual(getattr(args, name), float(value))
                    self.assertEqual(args.conf, float(confidence))
                    self.assertEqual(args.interpolation_search_angle, float(angle))
        self.assertEqual(parser.parse_args([*self.REQUIRED, "--min_conf=1"]).min_conf, 1)

    def test_v17_1_inference_and_interpolation_defaults(self) -> None:
        args = build_argparser().parse_args([
            '--input', 'input.mkv', '--model', 'gpu:model.engine',
        ])
        self.assertEqual(args.imgsz, 3072)
        self.assertEqual(args.interpolation_walk_back, 1)
        self.assertEqual(args.interpolation_candidates, 1)

    def test_temp_help_requires_explicit_memory_backed_scratch(self) -> None:
        help_text = build_argparser().format_help()
        self.assertIn('/dev/shm', help_text)
        self.assertNotIn('YOLO_TTA_SCRATCH', help_text)

    def test_unavailable_processor_and_crop_options_are_rejected(self) -> None:
        parser = build_argparser()
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        unavailable_options = [
            "--" + "_".join(("retina", "mask", "processor")),
            "--" + "_".join(("save", "nrrd", "tight", "crop")),
        ]
        help_text = parser.format_help()
        required = ["--input", "input.mkv", "--model", "gpu:model.engine"]
        for underscored in unavailable_options:
            for option in (underscored, underscored.replace("_", "-")):
                with self.subTest(option=option):
                    self.assertNotIn(option, option_strings)
                    self.assertNotIn(option, help_text)
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parser.parse_args([*required, option])

        crop_value = "_".join(("nrrd", "tight", "crop"))
        for value in (crop_value, crop_value.replace("_", "-")):
            with self.subTest(save_value=value):
                self.assertNotIn(value, SAVE_OPTION_TOKENS)
                with self.assertRaises(ValueError):
                    resolve_save_request(value)

    def test_retired_centerline_backend_and_tilt_alias_are_not_registered(self) -> None:
        parser = build_argparser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn('--centerline_filter_backend', option_strings)
        self.assertNotIn('--enable_tilt', option_strings)
        self.assertIn('--enable_tilted', option_strings)
        required = ['--input', 'input.mkv', '--model', 'gpu:model.engine']
        for retired_option, value in (
            ('--centerline_filter_backend', 'off'),
            ('--enable_tilt', 'transverse'),
        ):
            with self.subTest(retired_option=retired_option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([*required, retired_option, value])

    def test_angles_are_normalized_and_unique(self) -> None:
        self.assertEqual(resolve_tta_angles("-120,0,120"), [240.0, 0.0, 120.0])
        with self.assertRaises(ValueError):
            resolve_tta_angles("0,360")

    def test_channel_layouts(self) -> None:
        self.assertEqual(resolve_channel_format("grey").offsets, (0,))
        self.assertEqual(resolve_channel_format("RGB").offsets, (0, 0, 0))
        custom = resolve_channel_format("c5s2")
        self.assertEqual(custom.token, "C5S2")
        self.assertEqual(custom.offsets, (-4, -2, 0, 2, 4))
        with self.assertRaises(ValueError):
            resolve_channel_format("C4S1")

    def test_current_cpu_cuda_cli_contract_is_preserved(self) -> None:
        devices = resolve_backend_devices(["0,2:cpu"])
        self.assertEqual(devices.gpu_devices, ("cuda:0", "cuda:2"))
        self.assertTrue(devices.cpu)
        models = resolve_backend_models(["cpu:/models/openvino", "gpu:/models/model.engine"])
        self.assertEqual(models.cpu, "/models/openvino")
        self.assertEqual(models.gpu, "/models/model.engine")
        precisions = resolve_backend_precisions(["gpu:fp16", "cpu:bf16"], devices)
        self.assertEqual(precisions.gpu, 16)
        self.assertEqual(precisions.cpu, "bf16")
        batches = resolve_backend_batches(["gpu:4", "cpu:2"], devices)
        self.assertEqual((batches.gpu, batches.cpu), (4, 2))

    def test_model_roles_preserve_spaced_windows_paths_and_colons(self) -> None:
        models = resolve_backend_models([
            r'sam:C:\Model Bundles\sam:3.1',
            r'cpu:C:\Detector Models\openvino',
            r'gpu:C:\Detector Models\model.engine',
        ])
        self.assertEqual(models.sam, r'C:\Model Bundles\sam:3.1')
        self.assertEqual(models.cpu, r'C:\Detector Models\openvino')
        self.assertEqual(models.gpu, r'C:\Detector Models\model.engine')
        self.assertEqual(
            resolve_backend_models('gpu:"/models/detector with spaces.engine" sam:"/models/sam bundle"').sam,
            '/models/sam bundle',
        )

    def test_malformed_duplicate_and_sam_only_model_roles_fail(self) -> None:
        for entries in (
            ['gpu:model.engine', 'sam:'],
            ['gpu:model.engine', 'sam:a', 'sam:b'],
            ['gpu:model.engine', 'tracker:a'],
            ['gpu:model.engine', 'sam'],
            ['sam:bundle'],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                resolve_backend_models(entries)

    def test_interpolation_backend_rejects_both_before_resolution(self) -> None:
        parser = build_argparser()
        self.assertEqual(parser.parse_args(self.REQUIRED).interpolation_backend, 'sdf')
        for backend in ('both', 'auto', 'SAM'):
            with self.subTest(backend=backend):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args([*self.REQUIRED, '--interpolation_backend', backend])

    def test_cpu_detector_can_select_gpu_sam_without_gpu_detector_artifact(self) -> None:
        args = build_argparser().parse_args([
            '--input', 'input.mkv', '--model', 'cpu:openvino', 'sam:bundle',
            '--device', 'cpu', '--sam_device', '2,0', '--interpolation_backend', 'sam',
        ])
        models = resolve_backend_models(args.model)
        devices = resolve_backend_devices(args.device)
        settings = resolve_interpolation_settings(args, models, devices)
        self.assertEqual(settings.backend, 'sam')
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.sam_model, 'bundle')
        self.assertEqual(settings.sam_devices, ('cuda:2', 'cuda:0'))
        self.assertIsNone(models.gpu)
        self.assertFalse(devices.gpu_devices)
        self.assertTrue(devices.cpu)

    def test_active_sam_requires_bundle_and_cuda_pool(self) -> None:
        parser = build_argparser()
        args = parser.parse_args([
            *self.REQUIRED, '--interpolation_backend', 'sam', '--device', '0',
        ])
        with self.assertRaisesRegex(ValueError, 'requires a sam:PATH'):
            resolve_interpolation_settings(
                args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
            )
        args = parser.parse_args([
            '--input', 'input.mkv', '--model', 'cpu:openvino', 'sam:bundle',
            '--device', 'cpu', '--interpolation_backend', 'sam',
        ])
        with self.assertRaisesRegex(ValueError, 'CPU-only detector requires --sam_device'):
            resolve_interpolation_settings(
                args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
            )

    def test_sam_default_inherits_detector_cuda_and_override_stays_separate(self) -> None:
        parser = build_argparser()
        common = [
            *self.REQUIRED, 'sam:bundle', '--device', '3,1', '--interpolation_backend', 'sam',
        ]
        for extra, expected in (([], ('cuda:3', 'cuda:1')), (['--sam_device', '0'], ('cuda:0',))):
            args = parser.parse_args([*common, *extra])
            devices = resolve_backend_devices(args.device)
            settings = resolve_interpolation_settings(args, resolve_backend_models(args.model), devices)
            self.assertEqual(settings.sam_devices, expected)
            self.assertEqual(devices.gpu_devices, ('cuda:3', 'cuda:1'))

    def test_disabled_sam_and_sdf_expose_no_sam_resources(self) -> None:
        parser = build_argparser()
        for extra, backend, enabled in (
            (['--interpolation_backend', 'sam', '--interpolation_distance', '0'], 'sam', False),
            ([], 'sdf', True),
        ):
            args = parser.parse_args([
                '--input', 'input.mkv', '--model', 'cpu:openvino', '--device', 'cpu', *extra,
            ])
            settings = resolve_interpolation_settings(
                args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
            )
            self.assertEqual(settings.backend, backend)
            self.assertEqual(settings.enabled, enabled)
            self.assertIsNone(settings.sam_model)
            self.assertFalse(settings.sam_devices)

        args = parser.parse_args([
            *self.REQUIRED, 'sam:nonexistent-unused-bundle', '--device', '0',
            '--interpolation_backend', 'sam', '--interpolation_distance', '0',
        ])
        settings = resolve_interpolation_settings(
            args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
        )
        self.assertIsNone(settings.sam_model)

    def test_sam_devices_reject_cpu_and_malformed_indexes(self) -> None:
        self.assertEqual(resolve_sam_devices(['0,2', 'cuda:2', 'gpu:1']), ('cuda:0', 'cuda:2', 'cuda:1'))
        for values in (None, [], ['cpu'], ['0:cpu'], ['-1'], ['1.5'], ['cuda:']):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, '--sam_device'):
                resolve_sam_devices(values)

    def test_sam_feature_cache_limit_defaults_and_overrides_are_operational_only(self) -> None:
        parser = build_argparser()
        default_args = parser.parse_args(self.REQUIRED)
        self.assertEqual(default_args.sam_feature_cache_mib, 1024)
        for legacy_namespace in (False, True):
            with self.subTest(legacy_namespace=legacy_namespace):
                args = parser.parse_args([*self.REQUIRED, '--device', '0'])
                if legacy_namespace:
                    del args.sam_feature_cache_mib
                settings = resolve_interpolation_settings(
                    args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
                )
                self.assertEqual(settings.sam_feature_cache_mib, 1024)
        for budget in (0, 192, 512, 1024, 2048):
            args = parser.parse_args([
                *self.REQUIRED, 'sam:bundle', '--device', '0', '--interpolation_backend', 'sam',
                '--sam_feature_cache_mib', str(budget),
            ])
            settings = resolve_interpolation_settings(
                args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
            )
            self.assertEqual(settings.sam_feature_cache_mib, budget)
            self.assertEqual((settings.backend, settings.enabled, settings.sam_model, settings.sam_devices),
                             ('sam', True, 'bundle', ('cuda:0',)))
            self.assertEqual((args.interpolation_distance, args.interpolation_candidates,
                              args.interpolation_walk_back, args.interpolation_passes,
                              args.interpolation_min_radius, args.interpolation_search_angle),
                             (15, 1, 1, 1, 3, 15.0))

    def test_inactive_sam_retains_cache_configuration_without_resources(self) -> None:
        args = build_argparser().parse_args([
            '--input', 'input.mkv', '--model', 'cpu:detector', '--device', 'cpu',
            '--interpolation_backend', 'sam', '--interpolation_distance', '0',
            '--sam_feature_cache_mib', '2048',
        ])
        settings = resolve_interpolation_settings(
            args, resolve_backend_models(args.model), resolve_backend_devices(args.device),
        )
        self.assertEqual(settings.sam_feature_cache_mib, 2048)
        self.assertFalse(settings.enabled)
        self.assertIsNone(settings.sam_model)
        self.assertEqual(settings.sam_devices, ())

    def test_unselected_backend_settings_fail_instead_of_being_discarded(self) -> None:
        cpu_only = resolve_backend_devices(["cpu"])
        with self.assertRaisesRegex(ValueError, "selected no GPU backend"):
            resolve_backend_precisions(["gpu:fp16"], cpu_only)
        with self.assertRaisesRegex(ValueError, "selected no GPU backend"):
            resolve_backend_batches(["gpu:8"], cpu_only)

        gpu_only = resolve_backend_devices(["0"])
        with self.assertRaisesRegex(ValueError, "selected no CPU backend"):
            resolve_backend_precisions(["cpu:bf16"], gpu_only)
        with self.assertRaisesRegex(ValueError, "selected no CPU backend"):
            resolve_backend_batches(["cpu:2"], gpu_only)


if __name__ == "__main__":
    unittest.main()
