from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_reconciliation import compare, discover_geometry_context, _load_policies
from XTA.reconciliation import EvidenceLayer, reconcile
from XTA.reconciliation_io import write_seg_nrrd


POLICIES = Path(__file__).resolve().parents[1] / "XTA" / "examples" / "external_reconciliation"


def saved_run(tmp_path):
    root = tmp_path / "input_run"
    directory = root / "low_quality" / "small" / "nrrd"
    directory.mkdir(parents=True)
    shape = (4, 5, 6)
    a = np.zeros(shape, np.uint8)
    b = np.zeros(shape, np.uint8)
    a[1:3, 1:4, 1:4] = 1
    b[1:3, 2:4, 2:5] = 1
    entries = []
    for name, array in (("transverse", a), ("coronal", b), ("final", np.ones(shape, np.uint8))):
        path = directory / f"{name}.seg.nrrd"
        write_seg_nrrd(path, shape_tyx=shape, read_slab=lambda lo, hi, current=array: current[lo:hi])
        entries.append({"filename": path.name, "physical_view_name": name, "view_name": name,
                        "view_family": "orthogonal" if name != "final" else "global",
                        "source": "fullframe" if name != "final" else "global",
                        "mask_kind": "yolo" if name != "final" else "union",
                        "layer_role": "additive_component" if name != "final" else "checkpoint",
                        "recomposition_op": "union" if name != "final" else "select",
                        "output_shape_tyx": list(shape), "stored_shape_tyx": list(shape), "empty_segment": False})
    manifest = directory / "sample_nrrd_manifest.json"
    manifest.write_text(json.dumps({"layers": entries, "layer_count": len(entries), "output_shape_tyx": shape, "exported_axes": "(X, Y, t)"}))
    run = {"inputs": {"source_shape_t_y_x": (8, 10, 12), "processing_shape_t_y_x": (12, 10, 12)},
           "geometry": {"physical_views": [{"name": "transverse", "family": "orthogonal"}, {"name": "coronal", "family": "orthogonal"}],
                        "inference_view_variants": [{"name": "transverse__tta_a0", "physical_view_name": "transverse", "family": "orthogonal"}]},
           "resolved_configuration": {"conf": .65}}
    (root / "manifest.json").write_text(json.dumps(run))
    return manifest, a, b, root


def decoded(path):
    header, compressed = Path(path).read_bytes().split(b"\n\n", 1)
    sizes = next(line.split(b":", 1)[1].strip() for line in header.splitlines() if line.startswith(b"sizes:"))
    shape = tuple(int(v) for v in sizes.split())[::-1]
    return np.frombuffer(gzip.decompress(compressed), np.uint8).reshape(shape)


def saved_confidence(manifest_path, root, a, b, *, different_shape=False, different_model=False):
    from XTA.confidence_evidence import SCHEMA, configure_confidence_evidence, write_confidence_evidence
    data = json.loads(manifest_path.read_text())
    evidence_root = root / "reconciliation_evidence"
    entries = []
    for index, mask in enumerate((a, b)):
        layer = data["layers"][index]
        layer.update(layer_key=f"{layer['view_name']}__fullframe__yolo__pre_interpolation", model_name="model")
        score_shape = (mask.shape[0]+1, *mask.shape[1:]) if different_shape else mask.shape
        scores = np.zeros(score_shape, np.uint8)
        scores[:mask.shape[0]] = mask * (210 if index == 0 else 230)
        model_name = "other_model" if different_model else "model"
        directory = evidence_root / f"scores_{index}"
        reference = write_confidence_evidence(directory, score_shape, lambda z, current=scores: current[z],
                                             layer_key=layer["layer_key"], model_name=model_name)
        entries.append({"layer_key": reference.layer_key, "model_name": reference.model_name,
                        "directory": directory.name, "output_shape_tyx": list(reference.shape)})
    manifest_path.write_text(json.dumps(data))
    (evidence_root / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "layers": entries}))
    configure_confidence_evidence(enabled=False)
    return evidence_root


def test_discovers_source_run_geometry_and_preserves_different_t_scales(tmp_path):
    manifest, _, _, run = saved_run(tmp_path)
    context, path, data = discover_geometry_context(manifest)
    assert path == run / "manifest.json"
    assert context["source_shape_tyx"] == [8, 10, 12]
    assert context["processing_shape_tyx"] == [12, 10, 12]
    assert set(context["views_by_name"]) == {"transverse", "coronal", "transverse__tta_a0"}
    assert data["resolved_configuration"]["conf"] == .65


def test_real_core_comparison_excludes_checkpoint_and_preserves_inputs(tmp_path):
    manifest, a, b, root = saved_run(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "comparison"
    report = compare([manifest], [POLICIES / "union.py", POLICIES / "quorum3.py", POLICIES / "largest_island.py"],
                     output, memory_mib=8, previews=False, progress=lambda _: None)
    dataset = report["datasets"][0]
    methods = {method["policy_name"]: method for method in dataset["methods"]}
    np.testing.assert_array_equal(decoded(methods["union"]["output"]), a | b)
    np.testing.assert_array_equal(decoded(methods["quorum3"]["output"]), np.zeros_like(a))
    np.testing.assert_array_equal(decoded(methods["largest_island"]["output"]), a & b)
    assert methods["union"]["candidate_voxels"] == int(np.count_nonzero(a | b))
    assert dataset["additive_layer_count"] == 2
    assert dataset["inputs_unchanged"] is True
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in before.items())
    assert "before all global postprocessing" in report["comparison_stage"]
    assert not list(output.rglob("*.uint8"))
    assert not list(output.rglob(".working"))
    assert (output / "comparison.csv").is_file()
    assert json.loads((output / "comparison.json").read_text())["datasets"][0]["inputs_unchanged"] is True


def test_confidence_unavailable_is_explicit_and_union_reference_is_added(tmp_path):
    manifest, a, b, _ = saved_run(tmp_path)
    report = compare([manifest], [POLICIES / "confidence_anchored.py"], tmp_path / "comparison", memory_mib=8,
                     previews=False, progress=lambda _: None)
    methods = report["datasets"][0]["methods"]
    assert len(methods) == 2 and methods[0]["is_union_reference"]
    assert methods[1]["status"] == "unavailable"
    assert "threshold is not a confidence value" in methods[1]["reason"]
    assert report["confidence_available"] is False
    np.testing.assert_array_equal(decoded(methods[0]["output"]), a | b)


def test_comparison_creates_readable_slice_and_roi_previews(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image
    manifest, _, _, _ = saved_run(tmp_path)
    report = compare([manifest], [POLICIES / "quorum3.py"], tmp_path / "comparison", memory_mib=8, progress=lambda _: None)
    previews = report["datasets"][0]["previews"]
    assert len(previews["paths"]) == 2
    assert previews["slices"][0]["z"] == 2
    assert previews["slices"][1]["z"] == 1
    for path in previews["paths"]:
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 500


def test_fresh_output_and_duplicate_inputs_are_required(tmp_path):
    manifest, _, _, _ = saved_run(tmp_path)
    with pytest.raises(FileExistsError, match="fresh directory"):
        compare([manifest], [POLICIES / "union.py"], tmp_path, memory_mib=8, previews=False)
    with pytest.raises(ValueError, match="only once"):
        compare([manifest, manifest], [POLICIES / "union.py"], tmp_path / "comparison", memory_mib=8, previews=False)
    with pytest.raises(ValueError, match="outside the input"):
        compare([manifest], [POLICIES / "union.py"], manifest.parent / "comparison", memory_mib=8, previews=False)


def test_largest_island_two_average_vote_threshold_distinguishes_weak_and_strong_pairs():
    shape = (6, 6, 6)
    weak = np.zeros(shape, np.uint8)
    strong = np.zeros(shape, np.uint8)
    weak[0, 0, :4] = 1
    strong[3:5, 1:5, 1:5] = 1
    layers = []
    for name, mask in (("transverse", weak), ("coronal", weak),
                       ("tilted_transverse_vertical_p30", strong), ("tilted_coronal_vertical_p30", strong)):
        layers.append(EvidenceLayer(name, shape, {"physical_view_name": name, "view_family": "orthogonal", "source": "fullframe", "mask_kind": "yolo"},
                                    lambda lo,hi,array=mask: array[lo:hi]))
    policy = next(policy for _,policy in _load_policies([POLICIES / "largest_island.py"], 8) if policy["name"] == "largest_island")
    assert policy["threshold"] == 2.0
    result = np.zeros(shape, np.uint8)
    report = reconcile(layers, shape_tyx=shape, policy=policy, write_slab=lambda lo,hi,slab: result.__setitem__(slice(lo,hi),slab), memory_mib=8)
    assert report["components"]["transverse"]["weight"] < 1
    assert report["components"]["tilted_transverse_vertical_p30"]["weight"] > 1
    np.testing.assert_array_equal(result, strong)


def test_persisted_confidence_replays_after_registry_reset(tmp_path):
    from XTA.confidence_evidence import _REGISTRY
    manifest, a, b, root = saved_run(tmp_path)
    evidence_root = saved_confidence(manifest, root, a, b)
    assert not _REGISTRY
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_root.rglob("*") if p.is_file()}
    report = compare([manifest], [POLICIES / "confidence_anchored.py"], tmp_path / "comparison", memory_mib=8,
                     previews=False, progress=lambda _: None)
    method = next(m for m in report["datasets"][0]["methods"] if m["policy_name"] == "confidence_anchored")
    assert method["status"] == "complete"
    assert report["confidence_available"] is True
    assert report["datasets"][0]["confidence_evidence"]["matched_prediction_layers"] == 2
    np.testing.assert_array_equal(decoded(method["output"]), a & b)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path,digest in before.items())


def test_confidence_grid_mismatch_is_reported_without_resizing(tmp_path):
    manifest, a, b, root = saved_run(tmp_path)
    saved_confidence(manifest, root, a, b, different_shape=True)
    report = compare([manifest], [POLICIES / "confidence_anchored.py"], tmp_path / "comparison", memory_mib=8,
                     previews=False, progress=lambda _: None)
    dataset = report["datasets"][0]
    method = next(m for m in dataset["methods"] if m["policy_name"] == "confidence_anchored")
    assert method["status"] == "unsupported_confidence_grid"
    assert "does not resize scores" in method["reason"]
    assert len(dataset["confidence_evidence"]["unsupported_grids"]) == 2
    np.testing.assert_array_equal(decoded(dataset["methods"][0]["output"]), a | b)


def test_confidence_requires_exact_model_and_layer_identity(tmp_path):
    manifest, a, b, root = saved_run(tmp_path)
    saved_confidence(manifest, root, a, b, different_model=True)
    report = compare([manifest], [POLICIES / "confidence_anchored.py"], tmp_path / "comparison", memory_mib=8,
                     previews=False, progress=lambda _: None)
    dataset = report["datasets"][0]
    method = next(m for m in dataset["methods"] if m["policy_name"] == "confidence_anchored")
    assert method["status"] == "unavailable"
    assert dataset["confidence_evidence"]["matched_prediction_layers"] == 0
    assert len(dataset["confidence_evidence"]["missing_prediction_layers"]) == 2


def test_raw_output_is_retired_before_next_policy_and_previews_use_only_planes(tmp_path, monkeypatch):
    import tools.compare_reconciliation as tool
    manifest, a, b, _ = saved_run(tmp_path)
    output = tmp_path / "comparison"
    original_reconcile = tool.reconcile
    original_previews = tool.write_previews
    starts = []
    def checked_reconcile(*args, **kwargs):
        raw = list(output.rglob("*.uint8"))
        assert len(raw) == 1
        starts.append(raw[0].name)
        return original_reconcile(*args, **kwargs)
    def checked_previews(*args, **kwargs):
        assert not list(output.rglob("*.uint8"))
        planes = kwargs["preview_planes"]
        assert len(planes) == 3
        for selected in planes.values():
            assert set(selected) == {1, 2}
            for plane in selected.values():
                assert plane.shape == a.shape[1:] and type(plane) is np.ndarray and plane.base is None
        return original_previews(*args, **kwargs)
    monkeypatch.setattr(tool, "reconcile", checked_reconcile)
    monkeypatch.setattr(tool, "write_previews", checked_previews)
    report = tool.compare([manifest], [POLICIES / "quorum3.py", POLICIES / "largest_island.py"],
                          output, memory_mib=8, progress=lambda _: None)
    assert len(starts) == 3
    assert report["datasets"][0]["preview_retained_bytes"] == 3 * 2 * a[0].nbytes
    assert "_raw_path" not in (output / "comparison.json").read_text()


def test_policy_failure_removes_its_raw_output_map(tmp_path, monkeypatch):
    import tools.compare_reconciliation as tool
    manifest, _, _, _ = saved_run(tmp_path)
    output = tmp_path / "comparison"
    def fail(*args, **kwargs):
        raise RuntimeError("injected policy failure")
    monkeypatch.setattr(tool, "reconcile", fail)
    with pytest.raises(RuntimeError, match="injected policy failure"):
        tool.compare([manifest], [POLICIES / "union.py"], output, memory_mib=8, previews=False)
    assert not list(output.rglob("*.uint8"))


def test_deferred_native_confidence_is_reported_without_implicit_projection(tmp_path, monkeypatch):
    from XTA.confidence_evidence import ConfidenceEvidenceRef, write_block_confidence_evidence
    manifest,a,b,root = saved_run(tmp_path)
    data=json.loads(manifest.read_text())
    entries=[]
    for i,mask in enumerate((a,b)):
        metadata=data["layers"][i]
        metadata.update(model_name="model",layer_key=f"native_{i}")
        directory=root/"reconciliation_evidence"/f"blocks_{i}"
        ref=write_block_confidence_evidence(directory,mask.shape,lambda z,m=mask:m[z]*200,
            model_name="model",layer_key=metadata["layer_key"],coordinate_space="native_view_processing",
            source_shape_tyx=(8,10,12))
        entries.append(dict(directory=directory.name,model_name=ref.model_name,layer_key=ref.layer_key,
                            output_shape_tyx=list(ref.shape)))
    manifest.write_text(json.dumps(data))
    (root/"reconciliation_evidence/manifest.json").write_text(json.dumps(dict(schema="xta.confidence_evidence/1",layers=entries)))
    def fail(*args,**kwargs):raise AssertionError("Comparison must not project native confidence implicitly")
    monkeypatch.setattr(ConfidenceEvidenceRef,"source_reader",fail)
    report=compare([manifest],[POLICIES/"confidence_anchored.py"],tmp_path/"comparison",memory_mib=8,
                   previews=False,progress=lambda _:None)
    methods=report["datasets"][0]["methods"]
    assert methods[0]["status"]=="complete"
    assert methods[1]["status"]=="native_confidence_requires_export"
    assert len(report["datasets"][0]["confidence_evidence"]["deferred_native_layers"])==2
