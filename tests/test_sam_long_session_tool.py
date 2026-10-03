from __future__ import annotations

import contextlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import qualify_sam_interpolation_long_session as tool


def arguments(tmp_path, *, extra=()):
    images = np.arange(3*9*13, dtype=np.uint16).reshape(3,9,13).astype(np.uint8)
    image_path = tmp_path/"retained.uint8.dat"
    image_path.write_bytes(images.tobytes())
    seed = np.zeros((9,13), bool)
    seed[2:7,3:10] = True
    seed_path = tmp_path/"original.npy"
    np.save(seed_path, seed)
    argv = ["--images",str(image_path),"--shape","3","9","13",
            "--seed-mask",str(seed_path),"--seed-source-frame","1",
            "--frames","33","--crop-margin","1","--output",str(tmp_path/"output"),
            "--gpu-lock",str(tmp_path/"diagnostic_GPU_LOCK"),*extra]
    args = tool.parser().parse_args(argv)
    return args, argv, images, seed


def test_prepare_only_preserves_native_seed_pixels_and_never_imports_torch(tmp_path):
    args, argv, real_images, original = arguments(tmp_path, extra=("--prepare-only",))
    result = subprocess.run([sys.executable,"-c",
        "import sys; from tools import qualify_sam_interpolation_long_session as t; "
        "result=t.main(sys.argv[1:]); assert 'torch' not in sys.modules; raise SystemExit(result)",
        *argv], cwd=tool.REPOSITORY, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    prepared = json.loads((args.output/"prepared_fixture.json").read_text())
    repeated = np.memmap(prepared["image_path"],dtype=np.uint8,mode="r",shape=tuple(prepared["shape_tyx"]))
    x0,y0,x1,y1 = prepared["crop_xyxy_in_source"]
    np.testing.assert_array_equal(repeated, np.broadcast_to(real_images[1,y0:y1,x0:x1],repeated.shape))
    np.testing.assert_array_equal(np.load(prepared["seed_path"]),original[y0:y1,x0:x1])
    repeated._mmap.close()
    assert prepared["original_component_foreground"] == int(original.sum())
    assert prepared["synthetic_temporal_fixture"] and not prepared["accuracy_claim"]
    assert prepared["session_contract"]["ordinary_lta_partitions"] == [[0,30],[30,33]]
    assert not args.gpu_lock.exists()
    assert not (args.output/"tracker_staging").exists()
    assert not (args.output/"qualification.json").exists()


def test_unprescribed_union_fails_before_model_lock_or_fixture_writes(tmp_path):
    args, argv, _, seed = arguments(tmp_path, extra=("--prepare-only",))
    seed[0,0] = True
    np.save(args.seed_mask, seed)
    before = tool.file_sha(args.seed_mask)
    assert tool.main(argv) == 1
    report = json.loads((args.output/"qualification.json").read_text())
    assert report["status"] == "failed" and not report["model_loaded"]
    assert "prescribe --component-id" in report["error"]
    assert tool.file_sha(args.seed_mask) == before
    assert not args.gpu_lock.exists()
    assert not (args.output/"repeated_real_frame.uint8.dat").exists()
    assert not list(args.output.glob("*.tmp"))
    args.output = tmp_path/"explicit_component_output"
    args.component_id = 2
    prepared = tool.prepare(args)
    assert prepared["original_component_count"] == 2
    assert prepared["original_component_foreground"] == int(seed.sum())-1


class FakeRuntime:
    def __init__(self, **kwargs):
        self.closed = False
        self.released_results = 0
        self.residency_released = False
        self.missing_terminal = False
        self.fail_close = False

    def start(self):
        return self

    def run(self, **request):
        count = request["frame_stop"]
        indices = range(count-1 if self.missing_terminal else count)
        return SimpleNamespace(frames={index:request["seed_mask"].copy() for index in indices},
            tracker_scores={index:.9 for index in indices},
            observation_status={index:"observed" for index in indices},
            receipt=dict(run_id=request["run_id"],direction=request["direction"],
                frame_range=[0,count],seed_frame=request["seed_frame"],expected_frames=list(range(count)),
                status="complete",coverage_complete=True,prediction_valid=True,
                adapter_receipt=dict(raw_observation_complete=True,raw_observation_callback_count=count,
                    seed_roundtrip_passed=True,seed_roundtrip_exact=True),sam_model=dict(model_version="sam3.1")))

    def release_result(self, result):
        self.released_results += 1

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("Controlled worker exit remains unproven")
        self.residency_released = True


def execute_fake(tmp_path, *, missing_terminal=False, fail_close=False):
    args, _, _, _ = arguments(tmp_path)
    args.model = tmp_path/"fake_checkpoint.pt"
    args.model.write_bytes(b"CPU protocol fixture, never a model")
    prepared = tool.prepare(args)
    holder = {}
    def factory(**kwargs):
        runtime = FakeRuntime(**kwargs)
        runtime.missing_terminal = missing_terminal
        runtime.fail_close = fail_close
        holder["runtime"] = runtime
        return runtime
    return args, prepared, factory, holder


def test_missing_terminal_fails_coverage_and_settles_runtime_and_lock(tmp_path):
    args, prepared, factory, holder = execute_fake(tmp_path, missing_terminal=True)
    with pytest.raises(AssertionError,match="exact full requested"):
        tool.execute(args,prepared,runtime_factory=factory,monitor_factory=lambda *_:contextlib.nullcontext())
    report = json.loads((args.output/"qualification.json").read_text())
    assert report["status"] == "failed" and report["runs"] == []
    assert report["worker_residency_released"] and report["gpu_lock_released"]
    assert holder["runtime"].closed and holder["runtime"].released_results == 1
    assert not args.gpu_lock.exists()
    assert not list(args.output.glob("*.tmp"))


def test_unproven_exit_retains_owned_lock_and_never_reports_passed(tmp_path):
    args, prepared, factory, holder = execute_fake(tmp_path, fail_close=True)
    with pytest.raises(RuntimeError,match="unproven"):
        tool.execute(args,prepared,runtime_factory=factory,monitor_factory=lambda *_:contextlib.nullcontext())
    report = json.loads((args.output/"qualification.json").read_text())
    assert report["status"] == "failed"
    assert not report["worker_residency_released"] and not report["gpu_lock_released"]
    assert "unproven" in report["gpu_lock_retained_reason"]
    assert args.gpu_lock.exists() and holder["runtime"].closed
    assert len(report["runs"]) == 2 and holder["runtime"].released_results == 2
    for direction in ("forward","backward"):
        with np.load(args.output/(direction+"_raw_masks.npz"),allow_pickle=False) as data:
            np.testing.assert_array_equal(data["frame_indices"],np.arange(33))
            assert data["observation_status"].tolist() == ["observed"]*33
