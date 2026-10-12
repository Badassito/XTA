import json
from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import qualify_sam_precision as probe


def test_mask_comparison_measures_small_object_loss_and_added_foreground():
    a = np.zeros((4, 5), dtype=bool)
    a[1, 1:3] = True
    b = np.zeros_like(a)
    b[1, 1] = b[3, 4] = True
    result = probe.mask_difference(a, b)
    assert result["iou"] == pytest.approx(1/3)
    assert result["added_pixels"] == result["removed_pixels"] == 1
    assert result["changed_pixels"] == 2
    assert not result["empty_flip"]
    assert probe.mask_difference(a, np.zeros_like(a))["empty_flip"]
    with pytest.raises(ValueError, match="binary"):
        probe.mask_difference(a, np.full_like(a, 2, dtype=np.uint8))


def arm(path, *, changed=False):
    path.mkdir()
    a = np.zeros((4, 5), dtype=bool)
    a[1, 1:3] = True
    b = a.copy()
    if changed:
        b[1, 2] = False
    np.savez_compressed(path / "job.npz", **{"0": a, "1": b})
    request = dict(ordered_frames=[0, 1], seed_shape=[4, 5], run_id="r")
    row = dict(index=0, payload="job.npz", payload_sha256=probe.sha(path / "job.npz"),
               scores={"0": .9, "1": .8}, observation_status={"0": "present", "1": "present"},
               runtime_receipt=dict(sam_runtime=dict(git_commit="sdk", package_tree_sha256="tree", distribution_version="0.1.0"),
                                    sam_model=dict(checkpoint_sha256="model", model_version="sam3.1"),
                                    profile=dict(name="egpu", weight_storage="bfloat16", compile=False,
                                        use_fa3=changed,
                                        attention_backend="meta_fa3_fp8_e4m3fn" if changed else "torch_sdpa_flash_efficient_math",
                                        attention_preflight=dict(fallback=False, kernel_checks=[{}, {}]),
                                        attention_routing=dict(image_encoder=list(range(32)), memory_attention=list(range(8))))))
    report = dict(schema=probe.SCHEMA, status="completed", cache_mib=512,
                  source_before={"module": "sha"}, source_after={"module": "sha"},
                  hardware=dict(device="GPU", torch="test"), profile="auto", device=0,
                  attention_backend="fa3_fp8" if changed else "sdpa",
                  input_identity=dict(images_sha256="images", checkpoint_sha256="model",
                                      ordered_requests=[request]),
                  trials=[dict(measured=True, inference_service_seconds=1 if changed else 2,
                               rows=[row])])
    probe.write_json(path / "results.json", report)
    return report


def test_comparison_accepts_numerical_differences_without_claiming_quality(tmp_path):
    arm(tmp_path / "baseline")
    arm(tmp_path / "candidate", changed=True)
    report = probe.compare(tmp_path / "baseline", tmp_path / "candidate", tmp_path / "comparison.json")
    assert report["same_complete_work"]
    assert report["service_speedup"] == 2
    assert not report["accuracy_claim"]
    assert report["aggregate"]["foreground_iou"] == pytest.approx(.75)
    assert report["aggregate"]["removed_pixels"] == 1


@pytest.mark.parametrize("alteration", ["source", "frame", "duplicate", "failed", "score", "payload", "code", "hardware", "runtime", "backend"])
def test_comparison_rejects_invalid_or_unlike_arms(tmp_path, alteration):
    arm(tmp_path / "baseline")
    candidate = arm(tmp_path / "candidate", changed=True)
    row = candidate["trials"][0]["rows"][0]
    if alteration == "source":
        candidate["input_identity"]["images_sha256"] = "changed"
    elif alteration == "frame":
        row["scores"].pop("1")
    elif alteration == "duplicate":
        candidate["trials"][0]["rows"].append(row)
    elif alteration == "failed":
        candidate["status"] = "failed"
    elif alteration == "score":
        row["scores"]["1"] = "nan"
    elif alteration == "payload":
        (tmp_path / "candidate" / "job.npz").write_bytes(b"changed")
    elif alteration == "code":
        candidate["source_before"] = candidate["source_after"] = {"module": "changed"}
    elif alteration == "hardware":
        candidate["hardware"]["device"] = "another GPU"
    elif alteration == "runtime":
        row["runtime_receipt"]["sam_runtime"]["package_tree_sha256"] = "another SDK"
    elif alteration == "backend":
        row["runtime_receipt"]["profile"]["use_fa3"] = False
    probe.write_json(tmp_path / "candidate" / "results.json", candidate)
    with pytest.raises(ValueError):
        probe.compare(tmp_path / "baseline", tmp_path / "candidate", tmp_path / "comparison.json")


def test_reconstructs_complete_reverse_job_and_preserves_frame_cap(tmp_path, monkeypatch):
    from XTA.sam_evidence import SamEvidenceBundle
    np.zeros((3, 4, 5), dtype=np.uint8).tofile(tmp_path / "images.dat")
    (tmp_path / "case").mkdir()
    probe.write_json(tmp_path / "cases.json", dict(image_path="images.dat", image_shape=[3, 4, 5],
                                                 cases=[dict(id="case")]))
    probe.write_json(tmp_path / "case" / "sam_stats.json", dict(sam_evidence_path="evidence"))
    seed = np.zeros((2, 3), dtype=bool)
    seed[0, 1] = True
    bundle = SimpleNamespace(scope={}, groups={"g": dict(context_bbox_yx=[1, 1, 3, 4])},
                             runs={"r": dict(group_id="g", run_id="r", expected_frames=[0, 1, 2],
                                             direction=-1, seed_ids=["s"])},
                             group_mask=lambda *_: seed)
    monkeypatch.setattr(SamEvidenceBundle, "open", lambda *_: bundle)
    image, shape, requests, identities = probe.load_requests(tmp_path, 1, 3)
    assert image == tmp_path / "images.dat" and shape == (3, 4, 5)
    assert requests[0]["seed_frame"] == 2
    assert requests[0]["direction"] == "backward"
    assert identities[0]["ordered_frames"] == [2, 1, 0]
    assert np.array_equal(requests[0]["seed_mask"], seed)
    with pytest.raises(ValueError, match="Whole saved tracks"):
        probe.load_requests(tmp_path, 1, 2)
    bundle.runs["r"]["expected_frames"] = [0, 2]
    with pytest.raises(ValueError, match="every frame"):
        probe.load_requests(tmp_path, 1, 3)


def test_result_validation_rejects_missing_or_nonfinite_outputs(tmp_path):
    identity = dict(ordered_frames=[0, 1], seed_shape=[2, 3])
    masks = {0: np.zeros((2, 3), dtype=bool), 1: np.ones((2, 3), dtype=bool)}
    result = SimpleNamespace(frames=masks, receipt=dict(coverage_complete=True),
                             tracker_scores={0: .9, 1: .8}, observation_status={0: "present", 1: "present"})
    saved = probe.save_result(result, identity, tmp_path / "valid.npz")
    assert saved["scores"] == {"0": .9, "1": .8}
    result.tracker_scores[1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        probe.save_result(result, identity, tmp_path / "invalid.npz")
    result.frames.pop(1)
    with pytest.raises(ValueError, match="frames"):
        probe.save_result(result, identity, tmp_path / "missing.npz")


def test_native_fixture_preserves_real_planes_and_full_seed(tmp_path):
    images = np.arange(3*4*5, dtype=np.uint8).reshape(3, 4, 5)
    images.tofile(tmp_path / "images.dat")
    seed = np.zeros((4, 5), dtype=bool)
    seed[1, 2:4] = True
    np.save(tmp_path / "seed.npy", seed)
    fixture = dict(image_path="images.dat", image_sha256=probe.sha(tmp_path / "images.dat"),
                   shape_tyx=[3, 4, 5], original_seed_file="seed.npy", crop_xyxy=[1, 0, 5, 3])
    probe.write_json(tmp_path / "fixture.json", fixture)
    _, shape, requests, identities = probe.load_fixture(tmp_path / "fixture.json", 3)
    assert shape == (3, 4, 5)
    assert identities[0]["ordered_frames"] == [0, 1, 2]
    assert requests[0]["seed_mask"].sum() == seed.sum()
    fixture["crop_xyxy"] = [1, 0, 3, 3]
    probe.write_json(tmp_path / "fixture.json", fixture)
    with pytest.raises(ValueError, match="clipped"):
        probe.load_fixture(tmp_path / "fixture.json", 3)


def test_preflight_preserves_allocated_device_binding_and_failed_receipt(tmp_path, monkeypatch):
    import XTA.lta_sam as sam
    import tools.diagnose_sam_interpolation as diagnostic
    monkeypatch.setattr(diagnostic, "gpu_lock", lambda *_: nullcontext())
    monkeypatch.setattr(probe, "source_identity", lambda: {"source": "stable"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="13.0"),
                            cuda=SimpleNamespace(get_device_name=lambda _: "H100"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sam, "resolve_pinned_sam_runtime_provenance", lambda: {"git_commit": "pinned"})
    selected = []
    monkeypatch.setattr(sam, "preflight_sam_fa3", lambda _, device: selected.append(device) or {"fallback": False})
    args = SimpleNamespace(output=tmp_path / "valid", gpu_lock=tmp_path / "lock", device=1)
    probe.preflight(args)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    assert selected == [0]
    assert json.loads((args.output / "results.json").read_text())["status"] == "completed"
    args.output, args.device = tmp_path / "invalid", 2
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        probe.preflight(args)
    failed = json.loads((args.output / "results.json").read_text())
    assert failed["status"] == "failed" and failed["error_type"] == "ValueError"
