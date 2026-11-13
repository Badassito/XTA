"""Confidence cutoffs share discovery without rereading unchanged model files."""

import hashlib
from unittest import mock

import pytest

from XTA import startup_io
from XTA.unification import tta_manifest


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.delenv("XTA_YOLO_SWEEP_WORK_DIR", raising=False)
    monkeypatch.setenv("XTA_YOLO_SWEEP_ID", "fixture-family")
    monkeypatch.setenv("XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS", "50")
    monkeypatch.delenv("YOLO_TTA_FRAME_COUNT_FULL_SCAN", raising=False)
    tta_manifest._ARTIFACT_DIGESTS.clear()
    source = tmp_path / "input.mkv"
    model = tmp_path / "model.engine"
    source.write_bytes(b"video bytes")
    model.write_bytes(b"model bytes")
    return source, model


def capture(source, model):
    return tta_manifest.capture_tta_artifact_identities(
        input_path=source, gpu_model_path=model, cpu_model_path=None)


def test_standalone_startup_ignores_inherited_workspace(files, tmp_path, monkeypatch):
    foreign = tmp_path / "foreign-family"
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(foreign))
    monkeypatch.delenv("XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS")
    capture(*files)
    assert not foreign.exists()


def test_scalar_completion_reuses_digest_but_new_version_is_accepted(files):
    source, model = files
    with mock.patch.object(tta_manifest, "_sha256_file", wraps=tta_manifest._sha256_file) as digest:
        initial = capture(source, model)
        for _ in range(3):
            tta_manifest.assert_tta_artifacts_unchanged(initial)
        assert digest.call_count == 1
        model.write_bytes(b"legitimate next scalar model")
        with pytest.raises(RuntimeError, match="artifact changed during execution"):
            tta_manifest.assert_tta_artifacts_unchanged(initial)
        new = capture(source, model)
        tta_manifest.assert_tta_artifacts_unchanged(new)
        assert digest.call_count == 2
    assert new["models"]["gpu"]["artifacts"][0]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()


def test_sweep_cache_survives_new_process_local_state(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    with mock.patch.object(tta_manifest, "_sha256_file", wraps=tta_manifest._sha256_file) as digest:
        first = capture(source, model)
        # A new scalar subprocess has a fresh module cache and the same private directory.
        tta_manifest._ARTIFACT_DIGESTS.clear()
        second = capture(source, model)
        for _ in range(3):
            tta_manifest.assert_tta_artifacts_unchanged(second)
        assert first == second
        assert digest.call_count == 1
        assert not list((tmp_path / "private_work/startup").glob("*.tmp"))


def test_model_mutation_stops_later_cutoff_without_rereading(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    capture(source, model)
    model.write_bytes(b"changed model")
    with mock.patch.object(tta_manifest, "_sha256_file") as digest:
        with pytest.raises(RuntimeError, match="model artifact changed within confidence sweep"):
            capture(source, model)
        digest.assert_not_called()


def test_openvino_artifact_inventory_change_stops_later_cutoff(files, tmp_path, monkeypatch):
    source, _ = files
    export = tmp_path / "cpu_export"
    export.mkdir()
    (export / "cpu_export.xml").write_text("<net/>")
    (export / "cpu_export.bin").write_bytes(b"weights")
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    arguments = dict(input_path=source, gpu_model_path=None, cpu_model_path=export)
    tta_manifest.capture_tta_artifact_identities(**arguments)
    (export / "metadata.json").write_text('{"stride":32}')
    with pytest.raises(RuntimeError, match="artifact set changed within confidence sweep"):
        tta_manifest.capture_tta_artifact_identities(**arguments)


def test_source_probe_reused_at_next_cutoff(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    probe = mock.Mock(return_value=dict(width=128, height=96, num_frames=31, fps=29.97))
    invalidate = mock.Mock()
    arguments = dict(input_path=source, capture_identities=lambda: capture(source, model),
                     probe=probe, invalidate_manifest=invalidate)
    first = startup_io.capture_and_probe_tta_startup(**arguments)
    monkeypatch.setenv("XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS", "80")
    second = startup_io.capture_and_probe_tta_startup(**arguments)
    assert first == second
    assert probe.call_count == 1
    assert invalidate.call_count == 2


def test_source_change_stops_cached_probe_before_new_decode(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    probe = mock.Mock(return_value=dict(width=8, height=8, num_frames=8, fps=1.0))
    arguments = dict(input_path=source, capture_identities=lambda: capture(source, model),
                     probe=probe, invalidate_manifest=lambda: None)
    startup_io.capture_and_probe_tta_startup(**arguments)
    source.write_bytes(b"different input video")
    with pytest.raises(RuntimeError, match="changed within confidence sweep"):
        startup_io.capture_and_probe_tta_startup(**arguments)
    assert probe.call_count == 1


def test_frame_count_mode_has_its_own_probe_result(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    probe = mock.Mock(side_effect=[dict(num_frames=7), dict(num_frames=8)])
    arguments = dict(input_path=source, capture_identities=lambda: capture(source, model),
                     probe=probe, invalidate_manifest=lambda: None)
    assert startup_io.capture_and_probe_tta_startup(**arguments)[1] == dict(num_frames=7)
    monkeypatch.setenv("YOLO_TTA_FRAME_COUNT_FULL_SCAN", "1")
    assert startup_io.capture_and_probe_tta_startup(**arguments)[1] == dict(num_frames=8)
    assert probe.call_count == 2


def test_failed_probe_is_retried_without_cached_partial_metadata(files, tmp_path, monkeypatch):
    source, model = files
    monkeypatch.setenv("XTA_YOLO_SWEEP_WORK_DIR", str(tmp_path / "private_work"))
    probe = mock.Mock(side_effect=[OSError("failed discovery"), dict(num_frames=8)])
    arguments = dict(input_path=source, capture_identities=lambda: capture(source, model),
                     probe=probe, invalidate_manifest=lambda: None)
    with pytest.raises(OSError, match="failed discovery"):
        startup_io.capture_and_probe_tta_startup(**arguments)
    assert startup_io.capture_and_probe_tta_startup(**arguments)[1] == dict(num_frames=8)
    assert probe.call_count == 2
