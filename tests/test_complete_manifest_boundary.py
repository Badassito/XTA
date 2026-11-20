from __future__ import annotations

import tempfile
import unittest
import json
import ast
import contextlib
import io
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


from XTA import pipeline
from XTA import pta
from XTA.pta_runtime import build_runtime_options
from XTA.pta_config import parse_pta_args


class CompleteManifestBoundaryTests(unittest.TestCase):
    def test_incomplete_manifest_keeps_cleanup_and_adjacent_identity_check(self) -> None:
        events = []
        manifest = {'status': 'incomplete', 'sam_requested_work_complete': False,
                    'sam_planning_omissions': [{'operation': 'interpolation'}]}
        with (mock.patch.object(pipeline, '_cleanup_tta_selected_run_scratch',
                                side_effect=lambda **kwargs: events.append('cleanup')),
              mock.patch.object(pipeline, 'assert_tta_artifacts_unchanged',
                                side_effect=lambda value: events.append('revalidate')),
              mock.patch.object(pipeline, 'write_json_manifest',
                                side_effect=lambda path, value: events.append('write') or path)):
            pipeline._finalize_tta_selected_run_and_publish(temp_dir=Path('temp'), out_dir=Path('out'),
                manifest_path=Path('manifest.json'),
                manifest=manifest, artifact_identities={})
        self.assertEqual(events, ['cleanup', 'revalidate', 'write'])

    def test_incomplete_manifest_cannot_bypass_cleanup_or_identity_failure(self) -> None:
        manifest = {'status': 'incomplete', 'sam_requested_work_complete': False,
                    'sam_planning_omissions': [{'operation': 'interpolation'}]}
        for failure_at in ('cleanup', 'identity'):
            with (self.subTest(failure_at=failure_at),
                mock.patch.object(pipeline, '_cleanup_tta_selected_run_scratch',
                    side_effect=RuntimeError('cleanup failed') if failure_at == 'cleanup' else None),
                mock.patch.object(pipeline, 'assert_tta_artifacts_unchanged',
                    side_effect=RuntimeError('identity failed') if failure_at == 'identity' else None) as check,
                mock.patch.object(pipeline, 'write_json_manifest') as write):
                with self.assertRaisesRegex(RuntimeError, failure_at + ' failed'):
                    pipeline._finalize_tta_selected_run_and_publish(temp_dir=Path('temp'), out_dir=Path('out'),
                        manifest_path=Path('manifest.json'),
                        manifest=manifest, artifact_identities={})
                write.assert_not_called()
                if failure_at == 'cleanup':
                    check.assert_not_called()

    def test_manifest_publishers_reject_false_completion_or_unsubstantiated_incomplete(self) -> None:
        with mock.patch.object(pipeline, 'assert_tta_artifacts_unchanged') as check:
            with self.assertRaisesRegex(ValueError, 'omitted SAM work'):
                pipeline._publish_complete_tta_manifest(path=Path('manifest.json'), artifact_identities={},
                    manifest={'status': 'complete', 'sam_requested_work_complete': False})
            for manifest in ({'status': 'complete'}, {'status': 'incomplete'}):
                with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, 'requires omitted SAM work'):
                    pipeline._publish_incomplete_tta_manifest(path=Path('manifest.json'),
                        artifact_identities={}, manifest=manifest)
            check.assert_not_called()

    def test_actual_final_tail_retires_outputs_before_omission_failure_without_optional_reports(self) -> None:
        tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_main_impl')
        start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.With)
            and "tta.final.output_retirement_and_publication" in ast.unparse(node))
        code = compile(ast.Module(body=main.body[start:], type_ignores=[]), pipeline.__file__, 'exec')
        omission = dict(operation='interpolation', scope_id='model/azimuthal/fullframe', view='azimuthal',
                        pass_index=1, status='unresolved', reasons=['observation_inventory_memory_limit'],
                        unresolved_plan=True, unresolved_group_count=0, group_refusal_reasons={})
        cases = (([omission], None), ([omission], '1'), ([], None), ([], '1'))
        for omissions, strict in cases:
            with self.subTest(incomplete=bool(omissions), strict=strict):
                events = []
                telemetry = mock.MagicMock()
                telemetry.gauge.side_effect = lambda name, value: events.append((name, value))
                def retire(*args, **kwargs):
                    events.append('retired')
                    kwargs['after_close']()
                namespace = dict(_close_tta_output_artifacts=retire,
                    _finalize_selected_run_after_output_close=lambda: events.append('finalized'),
                    _report_sam_requested_work_status=pipeline._report_sam_requested_work_status,
                    runtime_telemetry=lambda: telemetry, output_artifacts=None,
                    sam_planning_omissions=omissions,
                    time=SimpleNamespace(perf_counter=lambda: 1.), post_inference_tail_started=0.,
                    pipeline_wall_started=0., out_dir=Path('out'), temp_dir=Path('temp'),
                    final_paths={}, summary_path=None, run_manifest_path=None)
                text = io.StringIO()
                env = {} if strict is None else {'YOLO_TTA_SAM_REQUIRE_COMPLETE': strict}
                with (mock.patch.object(pipeline, 'runtime_telemetry', return_value=telemetry),
                      mock.patch.dict(pipeline.os.environ, env), contextlib.redirect_stdout(text)):
                    if strict is None:
                        pipeline.os.environ.pop('YOLO_TTA_SAM_REQUIRE_COMPLETE', None)
                    if omissions and strict:
                        with self.assertRaisesRegex(RuntimeError, 'valid outputs were preserved'):
                            exec(code, namespace)
                    else:
                        exec(code, namespace)
                self.assertEqual(events[:2], ['retired', 'finalized'])
                phases = [event[1] for event in events if isinstance(event, tuple) and event[0] == 'pipeline.phase']
                self.assertEqual(phases, ['incomplete' if omissions else 'complete'])
                failed = bool(omissions and strict)
                self.assertEqual('Done.' in text.getvalue(), not failed)
                self.assertEqual('WARNING: Requested SAM work incomplete' in text.getvalue(),
                                 bool(omissions and not strict))

    def test_require_complete_fails_only_for_explicit_true_spellings(self) -> None:
        # The run fails late, so an unrecognized value keeps the recorded warning path.
        omission = dict(unresolved_plan=True, unresolved_group_count=0,
                        reasons=['observation_inventory_memory_limit'])
        cases = (('1', True), ('true', True), (' ON ', True), ('yes', True),
                 ('0', False), ('ture', False), ('disabled', False), ('warn', False), ('2', False))
        for value, fails in cases:
            with self.subTest(value=value), \
                    mock.patch.object(pipeline, 'runtime_telemetry', return_value=mock.MagicMock()), \
                    mock.patch.dict(pipeline.os.environ, {'YOLO_TTA_SAM_REQUIRE_COMPLETE': value}), \
                    contextlib.redirect_stdout(io.StringIO()) as text:
                if fails:
                    with self.assertRaisesRegex(RuntimeError, 'Requested SAM work incomplete'):
                        pipeline._report_sam_requested_work_status([omission])
                else:
                    pipeline._report_sam_requested_work_status([omission])
                    self.assertIn('WARNING: Requested SAM work incomplete', text.getvalue())

    def test_tta_finalize_facade_preserves_private_callback_patch_seams(self) -> None:
        manifest_path = Path("manifest.json")
        manifest = {"status": "complete"}
        identities = {"input": {}}
        with (
            mock.patch.object(pipeline, "_cleanup_tta_selected_run_scratch") as cleanup,
            mock.patch.object(
                pipeline,
                "_publish_complete_tta_manifest",
                return_value=manifest_path,
            ) as publish,
        ):
            result = pipeline._finalize_tta_selected_run_and_publish(
                temp_dir=Path("temp"),
                out_dir=Path("output"),
                manifest_path=manifest_path,
                manifest=manifest,
                artifact_identities=identities,
            )

        self.assertEqual(result, manifest_path)
        cleanup.assert_called_once_with(
            temp_dir=Path("temp"), out_dir=Path("output")
        )
        publish.assert_called_once_with(
            path=manifest_path,
            manifest=manifest,
            artifact_identities=identities,
        )

    def test_tta_cleanup_failure_prevents_revalidation_and_complete_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            scratch_dir = output_dir / "temp"
            (scratch_dir / "locked").mkdir(parents=True)
            manifest_path = output_dir / "manifest.json"
            identities = {"input": {"path": "source.mkv"}}

            with (
                mock.patch.object(
                    pipeline.shutil,
                    "rmtree",
                    side_effect=OSError("injected locked scratch directory"),
                ),
                mock.patch.object(
                    pipeline, "assert_tta_artifacts_unchanged"
                ) as revalidate,
                mock.patch.object(pipeline, "write_json_manifest") as atomic_write,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cleanup failed; refusing a complete manifest"
                ):
                    pipeline._finalize_tta_selected_run_and_publish(
                        temp_dir=scratch_dir,
                        out_dir=output_dir,
                        manifest_path=manifest_path,
                        manifest={"status": "complete"},
                        artifact_identities=identities,
                    )

            revalidate.assert_not_called()
            atomic_write.assert_not_called()
            self.assertFalse(manifest_path.exists())

    def test_tta_revalidates_immediately_before_atomic_complete_write(self) -> None:
        events: list[str] = []
        identities = {"input": {"path": "source.mkv"}}
        manifest = {"status": "complete", "mode": "tta"}
        manifest_path = Path("manifest.json")

        def revalidate(value: object) -> None:
            self.assertIs(value, identities)
            events.append("revalidate")

        def atomic_write(path: Path, value: object) -> Path:
            self.assertEqual(path, manifest_path)
            self.assertIs(value, manifest)
            events.append("atomic_write")
            return path

        with (
            mock.patch.object(pipeline, "_cleanup_tta_selected_run_scratch"),
            mock.patch.object(
                pipeline,
                "assert_tta_artifacts_unchanged",
                side_effect=revalidate,
            ),
            mock.patch.object(
                pipeline,
                "write_json_manifest",
                side_effect=atomic_write,
            ),
        ):
            result = pipeline._finalize_tta_selected_run_and_publish(
                temp_dir=Path("unused"),
                out_dir=Path("unused-output"),
                manifest_path=manifest_path,
                manifest=manifest,
                artifact_identities=identities,
            )

        self.assertEqual(result, manifest_path)
        self.assertEqual(events, ["revalidate", "atomic_write"])

    def test_tta_late_identity_failure_prevents_atomic_complete_write(self) -> None:
        with (
            mock.patch.object(pipeline, "_cleanup_tta_selected_run_scratch"),
            mock.patch.object(
                pipeline,
                "assert_tta_artifacts_unchanged",
                side_effect=RuntimeError("injected late artifact mutation"),
            ),
            mock.patch.object(pipeline, "write_json_manifest") as atomic_write,
        ):
            with self.assertRaisesRegex(RuntimeError, "late artifact mutation"):
                pipeline._finalize_tta_selected_run_and_publish(
                    temp_dir=Path("unused"),
                    out_dir=Path("unused-output"),
                    manifest_path=Path("manifest.json"),
                    manifest={"status": "complete"},
                    artifact_identities={"input": {}},
                )

        atomic_write.assert_not_called()

    def test_pta_work_cleanup_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / ".v18_work"
            work_dir.mkdir()
            (work_dir / "pending.bin").write_bytes(b"pending")
            with mock.patch.object(
                pta.shutil,
                "rmtree",
                side_effect=OSError("injected locked PTA work directory"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cleanup failed; refusing a complete manifest"
                ):
                    pta._cleanup_v18_pta_selected_run_work(work_dir)

            self.assertTrue(work_dir.exists())

    def test_pta_main_cleanup_failure_prevents_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            arguments = [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--save",
                "summary",
                "voxel_volume",
            ]
            config = parse_pta_args(arguments)
            runtime_options = build_runtime_options(config)
            topology = mock.Mock(
                cuda_device_ids=(),
                worker_cpu_order=(),
                allowed_cpus=(0,),
                summary="test topology",
            )

            with (
                mock.patch.object(pta, "discover_topology", return_value=topology),
                mock.patch.object(pta, "discover_volume_specs", return_value=[]),
                mock.patch.object(
                    pta,
                    "write_pta_summary",
                    return_value=output_dir / "summary.txt",
                ),
                mock.patch.object(
                    pta,
                    "write_v18_voxel_volume_report",
                    return_value=output_dir / "voxel_volume.json",
                ),
                mock.patch.object(
                    pta,
                    "_cleanup_v18_pta_selected_run_work",
                    side_effect=RuntimeError("injected selected-run cleanup failure"),
                ),
                mock.patch.object(
                    pta, "assert_v18_pta_inputs_unchanged"
                ) as revalidate,
                mock.patch.object(pta, "write_v18_pta_manifest") as complete_write,
                mock.patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                    pta.main(args=runtime_options, argv=arguments)

            revalidate.assert_not_called()
            complete_write.assert_not_called()
            manifest_path = output_dir / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["status"],
                "in_progress",
            )


if __name__ == "__main__":
    unittest.main()
