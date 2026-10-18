"""Each PTA directory keeps its annotation rules within one shared volume pipeline."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pta
from XTA.pta_config import build_pta_argparser, parse_pta_args
from XTA.pta_runtime import build_runtime_options


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    topology = SimpleNamespace(cuda_device_ids=(), gpu_cpu_sets=(),
        worker_cpu_order=(), allowed_cpus=(0, 1, 2, 3), summary="CPU multi-input test")
    monkeypatch.setattr(pta, "discover_topology", lambda **kwargs: topology)
    monkeypatch.setattr(pta, "available_memory_budget_bytes", lambda: 16 * pta.GIB)


def sequence(root, stem="sample", *, labels=None, indices=(1, 2, 3, 4)):
    root.mkdir(parents=True, exist_ok=True)
    for index in indices:
        assert pta.cv2.imwrite(str(root / f"{stem}_{index:04d}.png"),
            np.full((4, 4), 30 * index, np.uint8))
        if labels is not None and index in labels:
            polygon = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n" if index == 1 else ""
            (root / f"{stem}_{index:04d}.txt").write_text(polygon, encoding="utf-8")
    return root


def discover(roots, *, force=False):
    return pta.discover_volume_specs(roots, force=force, warnings=pta.WarningLog())


def run(roots, output, *, depth=2, task="segment", extra=()):
    save = ["images", "semantic" if task == "semantic" else "labels", "summary"]
    arguments = ["--input", *(str(root) for root in roots), "--output", str(output),
        "--imgsz", "4", "--enable_cartesian", "transverse", "--task", task,
        "--save", *save, "--worker_backend", "thread", "--pipeline_depth", str(depth),
        "--workers", "3", "--frame_workers", "1", "--jpeg_decode_backend", "opencv",
        "--no-topology_aware", *extra]
    pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def artifacts(output, folder):
    return {str(path.relative_to(output / folder)): path.read_bytes()
        for path in (output / folder).rglob("*") if path.is_file()}


def test_input_parser_forwards_roots_and_preserves_single_input():
    assert parse_pta_args(["--input", "one"]).args.input == "one"
    config = parse_pta_args(["--input", "one", "two", "--pipeline_depth", "2"])
    assert build_runtime_options(config).input == ["one", "two"]
    assert parse_pta_args(["--input", "one", "--input", "two"]).args.input == "two"
    parser = build_pta_argparser()
    assert parser.parse_args(["--input", "one"]).input == ["one"]
    assert parser.parse_args(["--input", "two"]).input == ["two"]


def test_root_order_multiple_volumes_and_collision_identity(tmp_path):
    first = sequence(tmp_path / "first", "sample")
    sequence(first, "alpha")
    second = sequence(tmp_path / "second", "sample")
    specs = discover([first, second])
    assert [spec.input_dir for spec in specs] == [first, first, second]
    assert specs[0].stem == "alpha"
    assert discover(first)[1].stem == "sample"
    assert specs[1].stem != specs[2].stem
    assert all(spec.source_stem == "sample" for spec in specs[1:])
    assert {spec.input_dir: spec.stem for spec in specs[1:]} == {
        spec.input_dir: spec.stem for spec in discover([second, first]) if spec.source_stem}
    assert specs == discover([first, second])
    candidates = [pta.OutputCandidate(order=i, volume_name=spec.stem,
        parent_view_tag="transverse", output_tag="transverse", item_key="full",
        frame_idx=0, is_tile=False, label_enabled=False) for i, spec in enumerate(specs[1:])]
    identities = [pta.candidate_source_identity(candidate) for candidate in candidates]
    assert identities[0] != identities[1]
    policy = SimpleNamespace(content_sha256="same-policy")
    seeds = [pta.augmentation_seed_for_identity(policy, identity, 1) for identity in identities]
    assert seeds[0] != seeds[1]
    assert len(pta.assign_volume_split_by_stem(specs, train_split=0.5)) == 3
    records = pta.capture_v18_pta_input_identities(specs)
    assert records[1]["source_stem"] == "sample"
    assert records[1]["input_directory"] == str(first)


def test_collision_suffix_cannot_overwrite_an_existing_volume(tmp_path):
    first, second = sequence(tmp_path / "first"), sequence(tmp_path / "second")
    collision_name = discover([first, second])[0].stem
    sequence(first, collision_name)
    specs = discover([first, second])
    assert len({spec.stem.casefold() for spec in specs}) == 3
    assert next(spec for spec in specs if spec.source_stem is None).stem == collision_name


def test_collision_provenance_preserves_an_empty_source_prefix(tmp_path):
    roots = [sequence(tmp_path / "first", ""), sequence(tmp_path / "second", "")]
    specs = discover(roots)
    assert len({spec.stem for spec in specs}) == 2
    assert all(record["source_stem"] == "" for record in pta.capture_v18_pta_input_identities(specs))


def test_annotation_uniformity_is_checked_per_directory(tmp_path):
    full = sequence(tmp_path / "full", labels={1, 2, 3, 4})
    partial = sequence(tmp_path / "partial", labels={1, 3})
    unlabeled = sequence(tmp_path / "unlabeled")
    assert [spec.volume_class for spec in discover([full, partial, unlabeled])] == [
        "fully_labeled", "partially_labeled", "unlabeled"]
    sequence(full, "partial", labels={1, 3})
    with pytest.raises(ValueError, match="same volume type"):
        discover([full, unlabeled])
    assert [spec.volume_class for spec in discover(full, force=True)] == [
        "partially_labeled", "fully_labeled"]


@pytest.mark.parametrize("case", ["duplicate", "alias", "ancestor", "missing", "file", "empty"])
def test_invalid_input_roots_fail_before_discovery(tmp_path, case):
    root = sequence(tmp_path / "input")
    child = root / "child"
    child.mkdir()
    paths = {
        "duplicate": [root, root], "alias": [root, root / ".." / "input"],
        "ancestor": [root, child], "missing": [root, tmp_path / "missing"],
        "file": [root / "sample_0001.png"], "empty": [],
    }
    with pytest.raises((ValueError, FileNotFoundError)):
        discover(paths[case])


def test_output_safety_covers_every_root_and_precedes_cleanup(tmp_path, monkeypatch):
    first, second = sequence(tmp_path / "first"), sequence(tmp_path / "second")
    specs = discover([first, second])
    for output in (second, second / "published", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            pta.validate_fresh_output_safety(output, input_dir=first, specs=specs)
    pta.validate_fresh_output_safety(tmp_path / "published", input_dir=[first, second], specs=specs)
    deleted = []
    monkeypatch.setattr(pta, "clean_generated_output_dirs", lambda *a, **kw: deleted.append(a))
    with pytest.raises(ValueError, match="disjoint"):
        run([first, second], second / "published")
    assert deleted == []
    assert not (second / "published").exists()


@pytest.mark.parametrize("separate_roots", [True, False])
def test_volumes_enter_the_same_depth_two_pipeline(tmp_path, monkeypatch, separate_roots):
    first = sequence(tmp_path / "first", "a")
    second = sequence(tmp_path / "second" if separate_roots else first, "b")
    roots = [first, second] if separate_roots else [first]
    events, admissions = [], []
    real_prepare, real_drain = pta.prepare_loaded_source, pta.drain_render_results
    real_depth = pta.resolve_pta_pipeline_depth
    def prepare(source, **kwargs):
        events.append(("prepare", source.input_dir))
        return real_prepare(source, **kwargs)
    def drain(*args, **kwargs):
        events.append(("drain", None))
        return real_drain(*args, **kwargs)
    def depth(**kwargs):
        admissions.append(kwargs)
        return real_depth(**kwargs)
    monkeypatch.setattr(pta, "prepare_loaded_source", prepare)
    monkeypatch.setattr(pta, "drain_render_results", drain)
    monkeypatch.setattr(pta, "resolve_pta_pipeline_depth", depth)
    manifest = run(roots, tmp_path / "output")
    assert admissions[0]["volume_count"] == 2
    assert len(admissions[0]["resident_estimates"]) == 2
    assert events[:3] == [("prepare", first), ("prepare", second), ("drain", None)]
    assert [record["stem"] for record in manifest["volumes"]] == ["a", "b"]
    assert manifest["resolved_configuration"]["input_directories"] == [str(root) for root in roots]


@pytest.mark.parametrize("task", ["segment", "semantic"])
@pytest.mark.parametrize("unlabeled_first", [True, False])
def test_mixed_roots_preserve_individual_publication_and_background_rules(tmp_path, task, unlabeled_first):
    roots = [sequence(tmp_path / "full", "full", labels={1, 2, 3, 4}),
        sequence(tmp_path / "partial", "partial", labels={1, 3}),
        sequence(tmp_path / "unlabeled", "unlabeled")]
    extra = ["--background_percent", "0", "--train_split", "1", "--split_method", "slice"]
    expected = {folder: {} for folder in ("images", "masks" if task == "semantic" else "labels")}
    for index, root in enumerate(roots):
        output = tmp_path / f"baseline_{index}"
        run([root], output, depth=1, task=task, extra=extra)
        for folder in expected:
            expected[folder].update(artifacts(output, folder))
    combined = tmp_path / "combined"
    if unlabeled_first:
        roots.reverse()
    manifest = run(roots, combined, task=task, extra=extra)
    for folder in expected:
        assert artifacts(combined, folder) == expected[folder]
    assert len(expected["images"]) == 6
    assert len(expected["masks" if task == "semantic" else "labels"]) == 2
    expected_classes = ["fully_labeled", "partially_labeled", "unlabeled"]
    assert [record["effective_volume_class"] for record in manifest["volumes"]] == (
        list(reversed(expected_classes)) if unlabeled_first else expected_classes)
    assert "nc: 1" in (combined / "dataset.yaml").read_text()
    assert "Background filtering:\n  active: True\n" in (combined / "summary.txt").read_text()
    assert not (combined / ".v18_work").exists()


def test_collision_publication_is_deterministic_on_fresh_restart(tmp_path):
    first = sequence(tmp_path / "first", labels={1, 2, 3, 4})
    second = sequence(tmp_path / "second", labels={1, 2, 3, 4})
    output = tmp_path / "output"
    first_manifest = run([first, second], output)
    original = {folder: artifacts(output, folder) for folder in ("images", "labels")}
    second_manifest = run([first, second], output)
    assert len(original["images"]) == 8
    assert all(artifacts(output, folder) == files for folder, files in original.items())
    assert first_manifest["inputs"] == second_manifest["inputs"]
    assert [record["stem"] for record in first_manifest["volumes"]] == [
        record["stem"] for record in second_manifest["volumes"]]
