"""Actual CUDA rank output through native packing and numeric CVOL publication."""
from contextlib import closing
import hashlib
from inspect import isbuiltin
import json
import os
from pathlib import Path

import numpy as np
import pytest

from XTA import backprojection, sweep_backprojection
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore
from tests.native_stage_test_support import native_coordinator
from tests.reference_backends.native_pull import iter_destination_samples
from tests.test_sweep_gpu_sibling_projection import _native_views
from tests.test_sweep_shell_projection import _binary_oracle, _views

pytestmark = pytest.mark.skipif(
    os.environ.get("XTA_TEST_SWEEP_PROJECTION_CUDA") != "1",
    reason="requires coordinated actual CUDA qualification",
)


def reference_grouped_rank_mask(source, view, shape):
    if view.family in ("radial", "spherical"):
        return _binary_oracle(source, view, shape)
    expected = np.zeros(shape, np.uint8)
    flat = expected.reshape(-1)
    for dest, frame, row, column in iter_destination_samples(view, source.shape, shape):
        flat[dest[source[frame, row, column] != 0]] = 1
    return expected


def project_grouped_rank_view(view, planes, shape, work_dir, metrics):
    return sweep_backprojection.project_sweep_view(
        view=view, model_name="cleanup", cutoffs_hundredths=tuple(planes),
        native_shape=next(iter(planes.values())).shape, output_shape=shape,
        work_dir=work_dir, read_mask_plane=lambda cent, z: planes[cent][z],
        workers=4, memory_bytes=32 * 1024**2, metrics=metrics,
    )


@pytest.mark.parametrize("family", ("radial", "spherical", "tilted"))
def test_actual_grouped_rank_publication_uses_fused_pack(tmp_path, monkeypatch, request, family):
    import torch
    assert torch.cuda.is_available()
    assert os.environ.get("XTA_TEST_SWEEP_PROJECTION_CUDA") == "1"
    owners = json.loads(Path(os.environ["XTA_GPU_LOCK_PATH"]).read_text())
    assert owners["pid"] == os.getpid() and owners["token"] == os.environ["XTA_GPU_LOCK_TOKEN"]
    coordinator = native_coordinator(request, monkeypatch)
    coordinator.configure_workers([0])
    coordinator.set_pending_inference_backlog(False)
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection, "_MAIN_PROCESS_GPU_STAGE_COORDINATOR", coordinator)
    monkeypatch.setenv("YOLO_TTA_GPU_PROJECTION_OVERLAP", "0")
    monkeypatch.setenv("YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB", "1")
    monkeypatch.setenv("XTA_YOLO_SWEEP_RANK_GPU", "1")
    monkeypatch.setenv("YOLO_TTA_NATIVE_PULL_CUDA", "1")

    shell_views = _views()
    view = _native_views()[0] if family == "tilted" else next(v for v in shell_views if v.family == family)
    ranks = np.zeros((view.num_slices, view.src_h, view.src_w), dtype=np.uint8)
    ranks[:, 1:4, 1:4] = np.random.default_rng(891).integers(
        1, 5, (view.num_slices, 3, 3), dtype=np.uint8,
    )
    ranks[1:2] = 0
    planes = {cent: (ranks >= level).astype(np.uint8) for level, cent in enumerate((41, 53, 65, 77), 1)}
    shape = (9, 12, 15)
    input_digest = hashlib.sha256(b"".join(mask.tobytes() for mask in planes.values())).hexdigest()
    with monkeypatch.context() as config:
        config.setenv("CUDA_VISIBLE_DEVICES", "-1")
        config.setenv("YOLO_TTA_GPU_BACKPROJECT", "0")
        cpu_metrics = {}
        cpu_refs = project_grouped_rank_view(view, planes, shape, tmp_path / "cpu", cpu_metrics)
    assert all(entry["backend"].startswith("cpu_") for entry in cpu_metrics["projectors"])
    monkeypatch.setenv("YOLO_TTA_GPU_BACKPROJECT", "1")

    pack = sweep_backprojection._cpu.pack_rank_levels
    assert isbuiltin(pack)
    write = IncrementalRawBBoxMaskStoreWriter.consume_encoded_fields
    packs, writes = [], []

    def count_pack(*args, **kwargs):
        fields, offsets, payload = result = pack(*args, **kwargs)
        packs.append((len(fields), len(payload)))
        return result

    def count_write(writer, z0, fields, payload, *, packed):
        assert packed
        result = write(writer, z0, fields, payload, packed=packed)
        writes.append((z0, len(fields), len(payload)))
        return result

    monkeypatch.setattr(sweep_backprojection._cpu, "pack_rank_levels", count_pack, raising=False)
    monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter, "consume_encoded_fields", count_write)
    from XTA.cylindrical_cuda_projection import RadialCudaProjector
    from XTA.native_pull_cuda import NativePullCudaSession
    from XTA.spherical_projection_cuda import SphericalCudaProjector

    projector = {"radial": RadialCudaProjector, "spherical": SphericalCudaProjector,
                 "tilted": NativePullCudaSession}[family]
    project_crops, close = projector.project_rank_crops, projector.close
    projected, closed = set(), set()

    def count_projection(session, *args, **kwargs):
        result = project_crops(session, *args, **kwargs)
        projected.add(id(session))
        return result

    def count_close(session):
        result = close(session)
        closed.add(id(session))
        return result

    monkeypatch.setattr(projector, "project_rank_crops", count_projection)
    monkeypatch.setattr(projector, "close", count_close)
    gpu_metrics = {}
    gpu_refs = project_grouped_rank_view(view, planes, shape, tmp_path / "gpu", gpu_metrics)
    expected_backend = ("cuda_native_uint8_max_rank_crops" if family == "tilted"
                        else f"cuda_{family}_uint8_max_rank_crops")
    receipt, = gpu_metrics["projectors"]
    assert receipt["backend"] == expected_backend, receipt
    lane, = receipt["lane_diagnostics"]
    assert lane["shell_blocks"] > 0 and lane["payload_d2h_bytes"] > 0
    assert projected and projected <= closed
    assert packs and sum(n for n, _ in packs) > 0
    assert sum(n for n, _ in packs) == sum(n for _, n, _ in writes)
    assert sum(size for _, size in packs) == sum(size for _, _, size in writes)
    assert not any(coordinator.snapshot()[key] for key in ("stage_leases", "overlap_leases", "resident_owners"))
    assert hashlib.sha256(b"".join(mask.tobytes() for mask in planes.values())).hexdigest() == input_digest

    compared, foreground = 0, 0
    for cent, mask in planes.items():
        expected = reference_grouped_rank_mask(mask, view, shape)
        cpu_ref, gpu_ref = cpu_refs[cent], gpu_refs[cent]
        if gpu_ref is None:
            assert cpu_ref is None and not expected.any()
            continue
        assert cpu_ref is not None
        for name in ("index.bin", "chunks.bin"):
            assert (gpu_ref.path / name).read_bytes() == (cpu_ref.path / name).read_bytes()
        with closing(RawBBoxMaskStore.open(gpu_ref.path)) as store:
            actual = np.stack([store.decode_slice(z) for z in range(shape[0])])
        np.testing.assert_array_equal(actual, expected)
        compared += 1
        foreground += np.count_nonzero(actual)
    assert compared >= 2 and foreground > 0
    report = dict(
        family=family, backend=expected_backend, gpu_blocks=lane["shell_blocks"],
        native_pack_calls=len(packs), numeric_writer_rows=sum(n for _, n, _ in writes),
        packed_bytes=sum(size for _, _, size in writes), cutoffs_compared=compared,
        foreground=int(foreground), exact=True, device_sessions_closed=len(projected), lease_retired=True,
    )
    report_path = Path(os.environ["XTA_GPU_PARITY_ROWS"]).with_name("grouped-rank-publication.jsonl")
    with report_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(report) + "\n")
