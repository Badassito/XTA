from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import XTA
from XTA import cli, config


ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "22.3.2"
CURRENT_LAUNCHER = "GPT-6-Astra-Ultra_v22.3.2_SLURM.py"
PREVIOUS_LAUNCHER = "GPT-6-Astra-Ultra_v22.3.1_SLURM.py"
SCRATCH_REPORTS = (
    "TTA_EXTERNAL_AUGMENTATION.md",
    "TTA_TEST_CLI_AUDIT.md",
    "PROJECTION_SAMPLING.md",
)


def _toml_section(source: str, name: str) -> str:
    marker = f"[{name}]"
    start = source.index(marker) + len(marker)
    remainder = source[start:]
    next_section = remainder.find("\n[")
    return remainder if next_section < 0 else remainder[:next_section]


class PackageMetadataTests(unittest.TestCase):
    def test_runtime_version_constants_are_aligned(self) -> None:
        self.assertEqual(XTA.__version__, CURRENT_VERSION)
        self.assertEqual(config.SCRIPT_VERSION, CURRENT_VERSION)
        self.assertEqual(config.SCRIPT_VERSION_COMPACT, "2232")
        self.assertEqual(config.SCRIPT_BASENAME, CURRENT_LAUNCHER)
        self.assertEqual(cli.SCRIPT_VERSION, CURRENT_VERSION)
        self.assertEqual(cli.SCRIPT_BASENAME, CURRENT_LAUNCHER)

    def test_project_metadata_uses_mode_dispatcher(self) -> None:
        source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project = _toml_section(source, "project")
        scripts = _toml_section(source, "project.scripts")
        package_data = _toml_section(source, "tool.setuptools.package-data")
        data_files = _toml_section(source, "tool.setuptools.data-files")

        self.assertIn('name = "xta"', project)
        self.assertIn(f'version = "{CURRENT_VERSION}"', project)
        self.assertIn('xta = "XTA.cli:run"', scripts)
        self.assertIn(
            '"XTA.examples.external_augmentations" = ["README.md"]',
            package_data,
        )
        self.assertIn('"XTA.examples.external_reconciliation" = ["README.md"]', package_data)
        self.assertIn(f'"{CURRENT_LAUNCHER}"', data_files)
        self.assertIn('"ARCHITECTURE.md"', data_files)
        for report in SCRATCH_REPORTS:
            self.assertNotIn(report, data_files)
        self.assertIn('"tools/hgx_selftest.py"', data_files)
        self.assertIn('"tools/compare_reconciliation.py"', data_files)
        self.assertIn('"tools/export_reconciliation_evidence.py"', data_files)
        self.assertIn('"tools/qualify_tta_reconciliation.py"', data_files)
        self.assertIn('"tools/qualify_confidence_consolidation.py"', data_files)
        self.assertIn('"tools/replay_component_projection.py"', data_files)
        self.assertIn('"tools/certify_qsc_lipschitz.py"', data_files)
        self.assertIn('"tools/plan_projection_sampling.py"', data_files)
        self.assertIn('"tools/qualify_pta_gpu_publication.py"', data_files)
        self.assertIn('"tools/qualify_tta_cpu_augmentation.py"', data_files)
        self.assertIn('"tools/qualify_tta_hybrid_augmentation.py"', data_files)
        self.assertIn('"tools/benchmark_tilted_azimuthal_projection.py"', data_files)
        self.assertIn('"tools/d1_ipc_selftest.py"', data_files)
        self.assertIn('"tools/tta_augmentation_smoke.py"', data_files)
        self.assertIn('"tools/check_tta_augmentation_run.py"', data_files)
        self.assertIn('"tools/lta_gpu_smoke.py"', data_files)
        self.assertIn('"tools/lta_point_smoke.py"', data_files)
        self.assertIn('"tools/lta_tile_smoke.py"', data_files)
        self.assertIn('"tools/lta_mask_seed_smoke.py"', data_files)
        self.assertIn('"tools/lta_full_volume.py"', data_files)
        self.assertIn('"tools/lta_tracklet_pair.py"', data_files)
        self.assertIn('"tools/lta_production_smoke.py"', data_files)
        self.assertIn('"tools/lta_worker_profile.py"', data_files)
        self.assertIn('"tools/lta_tracker_feature_smoke.py"', data_files)
        for tool in ("lta_trace_summary", "lta_host_io_profile", "lta_host_pipeline_smoke", "lta_window_gpu_smoke"):
            self.assertIn(f'"tools/{tool}.py"', data_files)
        self.assertNotIn('GPT-5.6-Sol-Ultra_v18.0.3_SLURM.py', data_files)
        self.assertNotIn('GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py', data_files)
        self.assertNotIn(PREVIOUS_LAUNCHER, data_files)

    def test_source_distribution_has_one_versioned_launcher(self) -> None:
        manifest_lines = {
            line.strip()
            for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn(f"include {CURRENT_LAUNCHER}", manifest_lines)
        self.assertNotIn("include GPT-5.6-Sol-Ultra_v18.0.3_SLURM.py", manifest_lines)
        self.assertNotIn("include GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py", manifest_lines)
        self.assertNotIn(f"include {PREVIOUS_LAUNCHER}", manifest_lines)
        self.assertIn("include XTA/_package_inventory.json", manifest_lines)
        self.assertIn("include ARCHITECTURE.md", manifest_lines)
        for report in SCRATCH_REPORTS:
            self.assertNotIn(f"include {report}", manifest_lines)
            self.assertFalse((ROOT / report).exists())
        self.assertIn("recursive-include XTA/examples *.py *.md", manifest_lines)
        self.assertIn("recursive-include tools *.py", manifest_lines)
        self.assertTrue((ROOT / "tools" / "hgx_selftest.py").is_file())
        self.assertTrue((ROOT / "tools" / "compare_reconciliation.py").is_file())
        self.assertTrue((ROOT / "tools" / "qualify_tta_reconciliation.py").is_file())
        self.assertTrue((ROOT / "tools" / "replay_component_projection.py").is_file())
        self.assertTrue((ROOT / "tools" / "certify_qsc_lipschitz.py").is_file())
        self.assertTrue((ROOT / "tools" / "plan_projection_sampling.py").is_file())
        self.assertTrue((ROOT / "tools" / "d1_ipc_selftest.py").is_file())
        self.assertTrue((ROOT / "tools" / "tta_augmentation_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "check_tta_augmentation_run.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_gpu_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_point_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_tile_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_mask_seed_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_full_volume.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_tracklet_pair.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_production_smoke.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_worker_profile.py").is_file())
        self.assertTrue((ROOT / "tools" / "lta_tracker_feature_smoke.py").is_file())
        for tool in ("lta_trace_summary", "lta_host_io_profile", "lta_host_pipeline_smoke", "lta_window_gpu_smoke"):
            self.assertTrue((ROOT / "tools" / f"{tool}.py").is_file())
        self.assertTrue((ROOT / CURRENT_LAUNCHER).is_file())
        self.assertEqual(
            sorted(path.name for path in ROOT.glob('*_SLURM.py')),
            [CURRENT_LAUNCHER],
        )
        self.assertFalse((ROOT / PREVIOUS_LAUNCHER).exists())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.3_SLURM.py").exists())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.0_SLURM.py").exists())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.1_SLURM.py").exists())
        self.assertFalse((ROOT / "GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py").exists())

    def test_complete_source_bundle_keeps_architecture_and_excludes_scratch_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xta-release-metadata-") as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "build_source_release.py"),
                 "--output-dir", directory],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            archive = Path(directory) / f"XTA_v{CURRENT_VERSION}_complete_source.zip"
            prefix = f"XTA_v{CURRENT_VERSION}/"
            with zipfile.ZipFile(archive) as source:
                manifest = json.loads(source.read(prefix + "RELEASE_MANIFEST.json"))
                self.assertEqual(manifest["version"], CURRENT_VERSION)
                self.assertEqual(manifest["launcher"], CURRENT_LAUNCHER)
                for name in (CURRENT_LAUNCHER, "ARCHITECTURE.md", ".gitattributes",
                             "native/README.md", "native/README_QAT.md", "native/README_QPL.md",
                             "XTA/examples/external_augmentations/README.md",
                             "XTA/examples/external_reconciliation/README.md",
                             "tools/compare_reconciliation.py", "tools/qualify_tta_reconciliation.py",
                             "tools/export_reconciliation_evidence.py", "tools/qualify_d1_confidence_bounds.py"):
                    with self.subTest(member=name):
                        expected = (ROOT / name).read_bytes()
                        self.assertEqual(source.read(prefix + name), expected)
                        self.assertEqual(manifest["files"][name], hashlib.sha256(expected).hexdigest())
                for report in SCRATCH_REPORTS:
                    self.assertNotIn(prefix + report, source.namelist())
                    self.assertNotIn(report, manifest["files"])
                self.assertEqual(
                    [name for name in source.namelist() if name.endswith("_SLURM.py")],
                    [prefix + CURRENT_LAUNCHER],
                )

    def test_tools_and_tests_do_not_use_release_or_sample_filenames(self) -> None:
        import re

        forbidden = re.compile(r"(?:^|[_-])(?:v\d+|m1)(?:[_-]|$)", re.IGNORECASE)
        offenders = [
            str(path.relative_to(ROOT))
            for folder in (ROOT / "tools", ROOT / "tests")
            for path in folder.glob("*.py")
            if forbidden.search(path.stem)
        ]
        self.assertEqual(offenders, [])

    def test_lta_and_experimental_sources_have_no_retired_identifiers(self) -> None:
        paths = (
            *sorted((ROOT / "tools").glob("*.py")),
            *sorted((ROOT / "tests").glob("test_lta*.py")),
            *sorted((ROOT / "XTA").glob("lta*.py")),
            ROOT / "XTA" / "experimental_features.py",
        )
        forbidden = ("v19_lta_", "v1803", "YOLO_TTA_V1803", "M1_", '"m1"', "'m1'")
        offenders = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    offenders.append((str(path.relative_to(ROOT)), token))
        self.assertEqual(offenders, [])

    def test_sample_moniker_is_absent_from_package_and_tool_sources(self) -> None:
        import re

        sample_name = re.compile(r"(?<![A-Za-z0-9])M1(?:_|\b)")
        offenders = [
            str(path.relative_to(ROOT))
            for folder in (ROOT / "XTA", ROOT / "tools")
            for path in folder.glob("*.py")
            if sample_name.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_legacy_distribution_identity_is_absent_from_text_sources(self) -> None:
        forbidden = (
            "volume" + "-tta",
            "volume" + "_tta",
            "VOLUME" + "_TTA",
            "Volume" + " TTA",
        )
        text_suffixes = {".c", ".h", ".in", ".json", ".md", ".py", ".toml"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(path=path.relative_to(ROOT), token=token):
                    self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
