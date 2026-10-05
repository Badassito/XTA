from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import warnings

from XTA import run_transport as transport
from tools.analyze_pipeline_trace import read_events
from tools.analyze_tta_scheduling import _parent_final_sample
from tools.lta_trace_summary import summarize
from tools.run_transport import main as transport_main


class RunTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "manifest.json").write_text(json.dumps({"status": "failed", "launcher":
            {"command": ["GPT-6-Astra-Ultra_v25.1.0_SLURM.py"]}}), encoding="utf-8")

    def write(self, relative, data):
        path = self.run.joinpath(*relative.split("/"))
        os.makedirs(transport._api_path(path.parent), exist_ok=True)
        with open(transport._api_path(path), "wb") as handle:
            handle.write(data)
        return path

    def test_lossless_all_roundtrip_preserves_scientific_schema_and_originals(self):
        originals = {
            "sam_interpolation/model/view/fullframe/sam_abc/evidence/manifest.json": b'{"schema":"xta.sam_evidence/1"}\n',
            "sam_interpolation/model/view/fullframe/sam_abc/evidence/index.json": b'{"masks":{}}',
            "sam_interpolation/model/view/fullframe/sam_abc/evidence/masks.bin": bytes(range(256)) * 19,
            "nrrd/image.seg.nrrd": b"NRRD0005\n\npayload",
            "telemetry/telemetry-job-1.jsonl": b'{"events":[]}\n',
        }
        for name, data in originals.items():
            self.write(name, data)
        archive = self.root / "run.zip"
        packed = transport.pack_run(self.run, archive, scope="all")
        with zipfile.ZipFile(archive) as stored:
            self.assertTrue(all(info.extract_version >= 45 for info in stored.infolist() if info.filename.startswith("DATA/")))
        self.assertEqual(packed["source_run"], {"status": "failed", "origin": "source_manifest", "partial": True})
        self.assertEqual(packed["source_changes"], [])
        self.assertIn("unknown", packed["source_inventory_completeness"])
        output = self.root / "output"
        unpacked = transport.unpack_run(archive, output)
        self.assertEqual(packed, unpacked)
        for name, data in originals.items():
            self.assertEqual((self.run / name).read_bytes(), data)
            self.assertEqual((output / name).read_bytes(), data)
        kinds = {row["path"]: row["category"] for row in packed["files"]}
        self.assertEqual(kinds["sam_interpolation/model/view/fullframe/sam_abc/evidence/masks.bin"], "scientific_evidence")

    def test_default_diagnostics_includes_evidence_and_lists_external_nrrds(self):
        self.write("telemetry/telemetry-1.jsonl", b"{}\n")
        self.write("sam_extrapolation/a/selection.json", b"{}")
        self.write("sam_extrapolation/a/evidence/masks.bin", b"mask")
        self.write("sam_extrapolation/a/evidence/index.json", b"{}")
        self.write("sam_extrapolation/a/evidence/manifest.json", b'{"complete":true}')
        self.write("sam_interpolation/a/component.cvol/chunks.bin", b"component")
        self.write("sam_interpolation/a/component.cvol/index.bin", b"index")
        self.write("sam_interpolation/a/component.cvol/manifest.json", b"{}")
        self.write("reconciliation_evidence/a/index.bin", b"index")
        self.write("reconciliation_evidence/a/payloads.zlib", b"payload")
        self.write("reconciliation_evidence/a/metadata.json", b"{}")
        self.write("nrrd/mask.nrrd", b"mask")
        packed = transport.pack_run(self.run, self.root / "diagnostics.zip", run_status="in_progress")
        self.assertEqual(packed["scope"], "diagnostics")
        self.assertEqual({row["path"] for row in packed["files"]},
                         {"manifest.json", "telemetry/telemetry-1.jsonl", "sam_extrapolation/a/selection.json",
                          "sam_extrapolation/a/evidence/masks.bin", "sam_extrapolation/a/evidence/index.json",
                          "sam_extrapolation/a/evidence/manifest.json", "sam_interpolation/a/component.cvol/chunks.bin",
                          "sam_interpolation/a/component.cvol/index.bin", "sam_interpolation/a/component.cvol/manifest.json",
                          "reconciliation_evidence/a/index.bin", "reconciliation_evidence/a/payloads.zlib",
                          "reconciliation_evidence/a/metadata.json"})
        self.assertEqual(len(packed["omitted"]), 1)
        self.assertEqual(packed["omitted"][0]["kind"], "external_nrrd")
        self.assertEqual(packed["external_outputs"][0]["path"], "nrrd/mask.nrrd")
        self.assertEqual(packed["external_outputs"][0]["sha256"], hashlib.sha256(b"mask").hexdigest())
        destination = self.root / "diag-restored"
        transport.unpack_run(self.root / "diagnostics.zip", destination)
        self.assertEqual((destination / "sam_extrapolation/a/evidence/masks.bin").read_bytes(), b"mask")
        self.assertFalse((destination / "nrrd/mask.nrrd").exists())
        self.assertEqual(packed["source_run"]["origin"], "caller")

    def test_external_nrrd_verification_separates_transfer_status_from_run_status(self):
        output = self.write("nrrd/mask.nrrd", b"mask")
        archive = self.root / "external.zip"
        packed = transport.pack_run(self.run, archive, run_status="failed")
        valid = transport.verify_external_outputs(packed, self.run)
        self.assertEqual(valid["status"], "verified")
        self.assertEqual(valid["source_run"]["status"], "failed")
        output.unlink()
        missing = transport.verify_external_outputs(packed, self.run)
        self.assertEqual(missing["missing"], 1)
        self.assertEqual(missing["source_run"], valid["source_run"])
        output.write_bytes(b"MASK")
        corrupt = transport.verify_external_outputs(packed, self.run)
        self.assertEqual(corrupt["corrupt"], 1)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stream:
            self.assertEqual(transport_main(["verify", str(archive), "--outputs-root", str(self.run)]), 3)
            self.assertEqual(json.loads(stream.getvalue())["status"], "archive_verified_external_outputs_incomplete")

    def test_unknown_run_state_is_not_promoted_to_complete(self):
        (self.run / "manifest.json").write_text("{}")
        packed = transport.pack_run(self.run, self.root / "unknown.zip")
        self.assertEqual(packed["source_run"]["status"], "unknown")
        self.assertIsNone(packed["source_run"]["partial"])

    def test_owned_runtime_temp_skipped_but_arbitrary_temp_named_data_retained(self):
        self.write("temp/checkpoint.dat", b"runtime")
        self.write("interesting.tmp", b"retain me")
        self.write("other/temp/checkpoint.dat", b"retain nested data")
        result = transport.pack_run(self.run, self.root / "temps.zip")
        self.assertIn("other/temp/checkpoint.dat", {row["path"] for row in result["files"]})
        self.assertIn("interesting.tmp", {row["path"] for row in result["files"]})
        self.assertEqual(result["omitted"][0]["path"], "temp")

    def test_fresh_outside_destinations_and_overwrites_refused(self):
        with self.assertRaises(transport.TransportError):
            transport.pack_run(self.run, self.run / "self.zip")
        archive = self.root / "run.zip"
        transport.pack_run(self.run, archive)
        before = archive.read_bytes()
        with self.assertRaises(transport.TransportError):
            transport.pack_run(self.run, archive)
        self.assertEqual(archive.read_bytes(), before)
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(transport.TransportError):
            transport.unpack_run(archive, output)

    def test_source_mutation_or_new_file_aborts_atomic_pack(self):
        data = self.write("data.bin", b"source")
        original = transport._copy_hash
        def mutate(source, destination):
            value = original(source, destination)
            if getattr(source, "name", "").endswith("data.bin"):
                data.write_bytes(b"changed source")
            return value
        with mock.patch.object(transport, "_copy_hash", side_effect=mutate):
            with self.assertRaisesRegex(transport.TransportError, "Source changed"):
                transport.pack_run(self.run, self.root / "changed.zip")
        self.assertFalse((self.root / "changed.zip").exists())
        self.assertFalse(list(self.root.glob(".xta-pack-*")))
        added = False
        def add_file(source, destination):
            nonlocal added
            value = original(source, destination)
            if not added:
                added = True
                self.write("new.bin", b"new")
            return value
        with mock.patch.object(transport, "_copy_hash", side_effect=add_file):
            with self.assertRaisesRegex(transport.TransportError, "inventory changed"):
                transport.pack_run(self.run, self.root / "added.zip")
        self.assertFalse((self.root / "added.zip").exists())

    def test_source_links_are_never_followed(self):
        target = self.root / "outside"
        target.mkdir()
        (target / "large-checkpoint").write_bytes(b"outside")
        link = self.run / "external"
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable")
        with self.assertRaisesRegex(transport.TransportError, "Link/reparse"):
            transport.pack_run(self.run, self.root / "links.zip")

    def test_windows_reparse_flag_is_rejected_before_descending(self):
        junction = self.run / "junction"
        junction.mkdir()
        original = transport._lstat
        def reparse(path):
            info = original(path)
            if Path(path) == junction:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=transport._REPARSE)
            return info
        with mock.patch.object(transport, "_lstat", side_effect=reparse):
            with self.assertRaisesRegex(transport.TransportError, "Link/reparse"):
                transport.pack_run(self.run, self.root / "junction.zip")
        self.assertFalse((self.root / "junction.zip").exists())

    def _malicious(self, filename, members, *, mode=None):
        path = self.root / filename
        entries = [{"path": name.removeprefix("DATA/"), "category": transport._category(Path(name.removeprefix("DATA/"))),
                    "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                   for name, data in members]
        manifest = {"schema": transport.SCHEMA, "scope": "all", "source_run": {"status": "unknown"}, "files": entries}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(transport.INDEX, json.dumps(manifest))
            for name, data in members:
                if mode is None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Duplicate name:.*", category=UserWarning)
                        archive.writestr(name, data)
                else:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = mode << 16
                    archive.writestr(info, data)
        return path

    def test_unsafe_paths_duplicates_links_and_case_collisions_refused(self):
        for i, name in enumerate(("DATA/../escaped", "DATA//absolute", "DATA/C:/drive", "DATA/a\\b", "DATA/CON.txt")):
            with self.subTest(name=name):
                path = self._malicious(f"unsafe{i}.zip", [(name, b"bad")])
                with self.assertRaises(transport.TransportError):
                    transport.unpack_run(path, self.root / f"unsafe{i}")
                self.assertFalse((self.root / f"unsafe{i}").exists())
        for label, members, mode in (("duplicates", [("DATA/a", b"1"), ("DATA/a", b"2")], None),
                                     ("case", [("DATA/a", b"1"), ("DATA/A", b"2")], None),
                                     ("prefix", [("DATA/a", b"1"), ("DATA/a/b", b"2")], None),
                                     ("link", [("DATA/a", b"target")], stat.S_IFLNK | 0o777)):
            with self.subTest(label=label):
                path = self._malicious(label + ".zip", members, mode=mode)
                with self.assertRaises(transport.TransportError):
                    transport.unpack_run(path, self.root / label)

    def test_crc_and_sha_corruption_never_publish_output(self):
        self.write("payload.bin", b"0123456789")
        path = self.root / "valid.zip"
        transport.pack_run(self.run, path)
        with zipfile.ZipFile(path) as archive:
            index = json.loads(archive.read(transport.INDEX))
            payloads = [(info.filename, archive.read(info)) for info in archive.infolist() if info.filename != transport.INDEX]
        next(row for row in index["files"] if row["path"] == "payload.bin")["sha256"] = "0" * 64
        sha = self.root / "sha.zip"
        with zipfile.ZipFile(sha, "w") as archive:
            archive.writestr(transport.INDEX, json.dumps(index))
            for name, data in payloads:
                archive.writestr(name, data)
        with self.assertRaisesRegex(transport.TransportError, "checksum"):
            transport.unpack_run(sha, self.root / "badsha")
        self.assertFalse((self.root / "badsha").exists())
        corrupted = bytearray(path.read_bytes())
        offset = corrupted.index(b"0123456789")
        corrupted[offset] = ord("X")
        crc = self.root / "crc.zip"
        crc.write_bytes(corrupted)
        with self.assertRaises(zipfile.BadZipFile):
            transport.unpack_run(crc, self.root / "badcrc")
        self.assertFalse((self.root / "badcrc").exists())
        self.assertFalse(list(self.root.glob(".xta-unpack-*")))

    def test_typed_inventory_cannot_relabel_scientific_payload_as_telemetry(self):
        self.write("sam_interpolation/evidence/masks.bin", b"raw scientific evidence")
        path = self.root / "typed.zip"
        transport.pack_run(self.run, path)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(transport.INDEX))
            contents = [(item.filename, archive.read(item)) for item in archive.infolist() if item.filename != transport.INDEX]
        next(row for row in manifest["files"] if row["path"].endswith("masks.bin"))["category"] = "telemetry"
        bad = self.root / "badtype.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr(transport.INDEX, json.dumps(manifest))
            for name, data in contents:
                archive.writestr(name, data)
        with self.assertRaisesRegex(transport.TransportError, "category mismatch"):
            read_events([bad])

    def test_cli_errors_are_explicit_and_truncated_archives_never_publish(self):
        archive = self.root / "cli.zip"
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(transport_main(["pack", str(self.run), str(archive)]), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "verified")
        with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
            self.assertEqual(transport_main(["pack", str(self.run), str(archive)]), 2)
            self.assertEqual(json.loads(error.getvalue())["status"], "failed")
        truncated = self.root / "truncated.zip"
        truncated.write_bytes(archive.read_bytes()[:-20])
        with self.assertRaises(zipfile.BadZipFile):
            transport.unpack_run(truncated, self.root / "truncated")
        self.assertFalse((self.root / "truncated").exists())

    def test_real_cli_relative_dot_segments_verify_unpack_and_outputs_root(self):
        self.write("nrrd/mask.nrrd", b"separate NRRD")
        archive = self.root / "relative.zip"
        transport.pack_run(self.run, archive)
        working = self.root / "working" / "nested"
        working.mkdir(parents=True)
        script = Path(__file__).resolve().parents[1] / "tools" / "run_transport.py"
        relative_archive = os.path.relpath(archive, working)
        relative_outputs = os.path.relpath(self.run, working)
        destination = self.root / "relative_restored"
        relative_destination = os.path.relpath(destination, working)
        self.assertIn("..", Path(relative_archive).parts)
        verification = subprocess.run([sys.executable, "-B", str(script), "verify", relative_archive,
                                        "--outputs-root", relative_outputs], cwd=working, capture_output=True, text=True)
        self.assertEqual(verification.returncode, 0, verification.stderr)
        self.assertEqual(json.loads(verification.stdout)["external_verification"]["verified"], 1)
        unpacking = subprocess.run([sys.executable, "-B", str(script), "unpack", relative_archive, relative_destination],
                                  cwd=working, capture_output=True, text=True)
        self.assertEqual(unpacking.returncode, 0, unpacking.stderr)
        self.assertEqual((destination / "manifest.json").read_bytes(), (self.run / "manifest.json").read_bytes())

    @unittest.skipUnless(os.name == "nt", "Windows extended prefix semantics")
    def test_existing_extended_windows_prefix_also_normalizes_dot_segments(self):
        target = self.root / "relative.zip"
        prefixed = "\\\\?\\" + str(self.root / "working" / ".." / target.name)
        self.assertEqual(transport._api_path(Path(prefixed)), transport._api_path(target))

    def test_long_paths_roundtrip_with_windows_extended_api(self):
        relative = "/".join(["sam_extrapolation", "v" * 70, "retry" * 15, "s" * 70, "selection.json"])
        original = self.write(relative, b"long path evidence")
        self.assertGreater(len(str(original)), 260)
        path = self.root / "long.zip"
        transport.pack_run(self.run, path)
        destination = self.root / "short"
        transport.unpack_run(path, destination)
        with open(transport._api_path(destination / relative), "rb") as handle:
            self.assertEqual(handle.read(), b"long path evidence")

    def test_archive_telemetry_reader_preserves_events_samples_and_partial_tails(self):
        event = {"trace_session": "session", "sequence": 1, "event": "worker_compute_start", "monotonic_ns": 1}
        record = {"schema": "gpt-6-astra-ultra-v25.1.0.telemetry.v1", "events": [event],
                  "counters": {"scheduler.operation.finished": 3}}
        self.write("telemetry/telemetry-job-1.jsonl", json.dumps(record).encode() + b'\n{"torn":')
        lta = {"schema": "lta.host-phase/1", "event": "process_start", "monotonic_ns": 1}
        self.write("lta_diagnostics/coordinator-host-1.jsonl", json.dumps(lta).encode() + b"\n")
        archive = self.root / "reader.zip"
        transport.pack_run(self.run, archive)
        loose_events, loose_warnings = read_events([self.run / "telemetry"])
        zip_events, zip_warnings = read_events([archive])
        self.assertEqual(loose_events, zip_events)
        self.assertEqual(len(loose_warnings), len(zip_warnings))
        self.assertIn("incomplete or invalid JSON line", zip_warnings[0])
        self.assertEqual(_parent_final_sample([archive]), _parent_final_sample(list((self.run / "telemetry").glob("*.jsonl"))))
        self.assertEqual(summarize(archive)["streams"][0]["events"], summarize(self.run / "lta_diagnostics")["streams"][0]["events"])


if __name__ == "__main__":
    unittest.main()
