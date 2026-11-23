"""Per-input pta_input_options.json overrides for fraction and enable_cpr."""
import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pta, pta_cpr
from XTA.pta_config import parse_pta_args
from XTA.pta_input_options import (
    PTA_INPUT_OPTIONS_FILENAME, read_pta_input_options, resolve_volume_options,
)
from XTA.pta_runtime import build_runtime_options


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    topology = SimpleNamespace(cuda_device_ids=(), gpu_cpu_sets=(),
        worker_cpu_order=(), allowed_cpus=(0, 1, 2, 3), summary="CPU input-options test")
    monkeypatch.setattr(pta, "discover_topology", lambda **kwargs: topology)
    monkeypatch.setattr(pta, "available_memory_budget_bytes", lambda: 16 * pta.GIB)


def sequence(root, stem, *, frames=10):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, frames + 1):
        image = np.full((6, 6), 20 * index % 256, np.uint8)
        image[1:5, 2:4] = 255 - index
        assert pta.cv2.imwrite(str(root / f"{stem}_{index:04d}.png"), image)
        polygon = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n" if index % 3 else ""
        (root / f"{stem}_{index:04d}.txt").write_text(polygon, encoding="utf-8")
    return root


def write_options(root, payload):
    path = root / PTA_INPUT_OPTIONS_FILENAME
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def run(roots, output, *, depth=2, extra=()):
    arguments = ["--input", *(str(root) for root in roots), "--output", str(output),
        "--imgsz", "6", "--enable_cartesian", "transverse", "sagittal", "coronal",
        "--save", "images", "labels", "summary", "--worker_backend", "thread",
        "--pipeline_depth", str(depth), "--workers", "3", "--frame_workers", "2",
        "--jpeg_decode_backend", "opencv", "--no-topology_aware", *extra]
    pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def artifacts(output):
    return {str(path.relative_to(output)): path.read_bytes()
        for folder in ("images", "labels") for path in (output / folder).rglob("*") if path.is_file()}


@pytest.mark.parametrize("payload,expected", [
    ({}, {}),
    ({"fraction": 0.0475}, {"fraction": 0.0475}),
    ({"fraction": 1}, {"fraction": 1.0}),
    ({"enable_cpr": False}, {"enable_cpr": False}),
    ({"fraction": 5e-324, "enable_cpr": True}, {"fraction": 5e-324, "enable_cpr": True}),
])
def test_valid_options_parse_exactly(tmp_path, payload, expected):
    path = write_options(tmp_path, payload)
    options = read_pta_input_options(tmp_path)
    assert options.overrides() == expected
    assert options.path == path.resolve()
    # The parsed fraction is the same double the CLI parser produces.
    if "fraction" in payload:
        assert options.fraction == float(repr(payload["fraction"]))
        assert isinstance(options.fraction, float)


def test_missing_options_file_is_not_an_override(tmp_path):
    assert read_pta_input_options(tmp_path) is None


@pytest.mark.parametrize("payload,match", [
    ({"fraction": 0.5, "ratio": 2}, "unknown key"),
    ({"Fraction": 0.5}, "unknown key"),
    ({"fraction": True}, "fraction must be"),
    ({"fraction": "0.5"}, "fraction must be"),
    ({"fraction": None}, "fraction must be"),
    ({"fraction": 0}, "fraction must be"),
    ({"fraction": -0.25}, "fraction must be"),
    ({"fraction": 1.0000001}, "fraction must be"),
    ({"fraction": 10 ** 400}, "fraction must be"),
    ('{"fraction": NaN}', "fraction must be"),
    ('{"fraction": Infinity}', "fraction must be"),
    ('{"fraction": 1e400}', "fraction must be"),
    ({"enable_cpr": 1}, "enable_cpr must be"),
    ({"enable_cpr": "true"}, "enable_cpr must be"),
    ({"enable_cpr": None}, "enable_cpr must be"),
    ([{"fraction": 0.5}], "JSON object"),
    ("0.5", "JSON object"),
    ('{"fraction": 0.5,}', "Invalid PTA input options"),
    ('{"fraction": 0.5, "fraction": 0.25}', "duplicate key"),
    ("", "Invalid PTA input options"),
])
def test_invalid_options_fail(tmp_path, payload, match):
    write_options(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        read_pta_input_options(tmp_path)


def test_unreadable_options_entry_fails(tmp_path):
    (tmp_path / PTA_INPUT_OPTIONS_FILENAME).mkdir()
    with pytest.raises(ValueError, match="Invalid PTA input options"):
        read_pta_input_options(tmp_path)


def test_resolve_volume_options_defaults_and_overrides():
    args = argparse.Namespace(enable_cpr=True, fraction=0.25, imgsz=8)
    plain = SimpleNamespace()
    assert resolve_volume_options(plain, args, fraction=0.25) == (0.25, True, args)
    options = SimpleNamespace(input_options=SimpleNamespace(fraction=None, enable_cpr=True))
    assert resolve_volume_options(options, args, fraction=0.25) == (0.25, True, args)
    options.input_options = SimpleNamespace(fraction=0.5, enable_cpr=False)
    fraction, cpr, volume_args = resolve_volume_options(options, args, fraction=0.25)
    assert (fraction, cpr, volume_args.enable_cpr, volume_args.imgsz) == (0.5, False, False, 8)
    assert volume_args is not args and args.enable_cpr is True


def test_discovery_ignores_options_file_as_data_and_tags_only_its_directory(tmp_path, monkeypatch):
    first, second = sequence(tmp_path / "first", "a"), sequence(tmp_path / "second", "b")
    before = pta.discover_volume_specs([first, second], force=False, warnings=pta.WarningLog())
    write_options(first, {"fraction": 0.5})
    after = pta.discover_volume_specs([first, second], force=False, warnings=pta.WarningLog())
    assert [spec.input_options.overrides() if spec.input_options else None for spec in after] == [
        {"fraction": 0.5}, None]
    assert [pta.replace(spec, input_options=None) for spec in after] == before
    records = pta.capture_v18_pta_input_identities(after)
    assert records[0]["input_options"]["overrides"] == {"fraction": 0.5}
    assert records[0]["input_options"]["path"] == str((first / PTA_INPUT_OPTIONS_FILENAME).resolve())
    assert records[1]["input_options"] is None
    pta.assert_v18_pta_inputs_unchanged(records)
    options_path = first / PTA_INPUT_OPTIONS_FILENAME
    initial_stat = options_path.stat()
    records[0]["input_options"]["change_or_creation_time_ns"] -= 1
    # Option content verification streams the file through the shared hash reader.
    def require_streaming(*args, **kwargs):
        pytest.fail("PTA provenance verification should hash in bounded reads")

    monkeypatch.setattr(Path, "read_bytes", require_streaming)
    pta.assert_v18_pta_inputs_unchanged(records)
    write_options(first, {"fraction": 0.6})
    os.utime(options_path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="changed during execution"):
        pta.assert_v18_pta_inputs_unchanged(records)


def test_invalid_options_fail_before_any_directory_is_probed(tmp_path, monkeypatch):
    first, second = sequence(tmp_path / "first", "a"), sequence(tmp_path / "second", "b")
    write_options(second, {"fraction": 2})
    probed = []
    real = pta._discover_directory_volume_specs
    monkeypatch.setattr(pta, "_discover_directory_volume_specs",
        lambda root, **kwargs: probed.append(root) or real(root, **kwargs))
    with pytest.raises(ValueError, match="fraction must be"):
        pta.discover_volume_specs([first, second], force=False, warnings=pta.WarningLog())
    assert probed == []
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="fraction must be"):
        run([first, second], output)
    assert not output.exists()


@pytest.mark.parametrize("first_options,second_options,cli", [
    # The first volume's override shifts the second volume's global order only.
    ({"fraction": 0.5, "enable_cpr": False}, None, ["--fraction", "0.75", "--enable_cpr"]),
    (None, {"fraction": 1}, ["--fraction", "0.5"]),
    ({"enable_cpr": True}, {"enable_cpr": True, "fraction": 0.3}, []),
])
def test_grouped_depth_two_overrides_match_separate_cli_runs(tmp_path, monkeypatch, first_options, second_options, cli):
    cpr_calls = []
    monkeypatch.setattr(pta_cpr, "extract_cpr_geometries",
        lambda mask, **kwargs: cpr_calls.append(mask.shape) or [])
    grouped_roots = [sequence(tmp_path / "grouped" / "first", "a"),
                     sequence(tmp_path / "grouped" / "second", "b")]
    effective = []
    for root, options in zip(grouped_roots, (first_options, second_options)):
        if options is not None:
            write_options(root, options)
        values = dict(options or {})
        fraction = float(values.get("fraction", cli[cli.index("--fraction") + 1] if "--fraction" in cli else 1.0))
        effective.append((fraction, bool(values.get("enable_cpr", "--enable_cpr" in cli))))

    expected = {}
    for index, (root, (fraction, cpr)) in enumerate(zip(grouped_roots, effective)):
        single = tmp_path / "single" / root.name
        shutil.copytree(root, single)
        if (single / PTA_INPUT_OPTIONS_FILENAME).exists():
            os.remove(single / PTA_INPUT_OPTIONS_FILENAME)
        cpr_calls.clear()
        manifest = run([single], tmp_path / f"single_out_{index}", depth=1,
            extra=["--fraction", repr(fraction), *(["--enable_cpr"] if cpr else [])])
        assert len(cpr_calls) == int(cpr)
        assert manifest["volumes"][0]["input_option_overrides"] == {}
        expected.update(artifacts(tmp_path / f"single_out_{index}"))

    cpr_calls.clear()
    output = tmp_path / "grouped_out"
    manifest = run(grouped_roots, output, extra=cli)
    assert artifacts(output) == expected
    assert len(expected) > 4
    assert len(cpr_calls) == sum(int(cpr) for _fraction, cpr in effective)
    volumes = manifest["volumes"]
    assert [(record["fraction"], record["enable_cpr"]) for record in volumes] == effective
    assert [record["input_option_overrides"] for record in volumes] == [
        {key: (float(value) if key == "fraction" else value) for key, value in (options or {}).items()}
        for options in (first_options, second_options)]
    configuration = manifest["resolved_configuration"]
    assert configuration["fraction_sampling"]["fraction"] == (
        float(cli[cli.index("--fraction") + 1]) if "--fraction" in cli else 1.0)
    assert configuration["enable_cpr"] is ("--enable_cpr" in cli)
    assert [bool(source["input_options"]) for source in manifest["inputs"]["artifacts"]] == [
        first_options is not None, second_options is not None]
    summary = (output / "summary.txt").read_text(encoding="utf-8")
    assert summary.count("    enable_cpr: ") == 2
    assert summary.count("    input_option_overrides: ") == sum(
        options is not None for options in (first_options, second_options))
