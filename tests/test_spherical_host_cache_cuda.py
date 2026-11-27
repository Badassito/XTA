"""Actual CUDA qualification for a TTA worker's spherical host-cache lease."""

import hashlib
import os
import weakref

import numpy as np
import pytest

from XTA import cuda_backend, inference, spherical_cuda, workers
from XTA.spherical_host_cache import close_spherical_host_cache, install_spherical_host_cache
from tests.test_spherical_cuda import rotation_xyz, spherical_view


@pytest.mark.skipif(os.environ.get("XTA_TEST_SPHERICAL_CUDA") != "1",
                    reason="explicit available-GPU qualification only")
def test_host_cache_survives_device_eviction_and_retires_after_worker_fence(monkeypatch):
    import torch
    from XTA.spherical_geometry import render_shell_frame

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setenv("YOLO_TTA_GPU_RENDER_RESIDENT", "1")
    monkeypatch.setenv("YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL", "1")
    monkeypatch.setenv("YOLO_TTA_GPU_SPHERICAL_FP32", "0")
    volume = np.random.default_rng(281).integers(0, 256, (11, 13, 15), dtype=np.uint8)
    view = spherical_view(volume.shape, face=4, size=70, intervals=64,
                          origin=(-2, -3), rotation=rotation_xyz(), radii=(.63, 5.))
    engine = cuda_backend._GpuWorkerRenderEngine("cuda:0")
    key = spherical_cuda.spherical_direction_key(view, engine.device)
    table_bytes = view.src_h * view.src_w * (3 * np.dtype(np.float64).itemsize + 1)

    def pixels(index):
        with torch.cuda.stream(engine._stream):
            result = spherical_cuda.render_spherical_native_resident(engine, view, index)
        assert result.is_cuda
        engine._stream.synchronize()
        return result.cpu().numpy()

    try:
        assert engine.ensure_volume_array(volume) == "resident"
        baseline = pixels(0)
        np.testing.assert_array_equal(baseline, render_shell_frame(volume, view, 0))
        spherical_cuda.clear_spherical_render_cache(engine)
        assert spherical_cuda.cached_spherical_directions(engine, key) is None

        # Supply explicit credit so this low-RAM test device exercises the funded path.
        cache = install_spherical_host_cache(engine, budget_bytes=4 * 1024**2)
        assert cache is not None
        before = spherical_cuda.spherical_direction_cache_stats(engine)
        np.testing.assert_array_equal(pixels(0), baseline)
        first = cache.snapshot()
        assert (first["builds"], first["hits"], first["live_bytes"]) == (1, 0, table_bytes)
        assert first["borrowed_bytes"] == first["scratch_bytes"] == 0

        # The comparison above fenced the renderer. Evict only the device directions.
        spherical_cuda.clear_spherical_render_cache(engine)
        assert spherical_cuda.cached_spherical_directions(engine, key) is None
        np.testing.assert_array_equal(pixels(0), baseline)
        replay = cache.snapshot()
        after = spherical_cuda.spherical_direction_cache_stats(engine)
        assert (replay["builds"], replay["hits"], replay["live_bytes"]) == (1, 1, table_bytes)
        assert after["cache_misses"] - before["cache_misses"] == 2
        assert after["host_build_blocks"] - before["host_build_blocks"] == 2
        assert after["h2d_copies"] - before["h2d_copies"] == 4
        assert after["h2d_bytes"] - before["h2d_bytes"] == 2 * table_bytes
        assert engine._spherical_native_kernel_announced
        assert not getattr(engine, "_spherical_native_kernel_disabled", False)

        monkeypatch.setattr(cuda_backend, "_WORKER_GPU_RENDER_ENGINE", engine)
        for name in ("_GPU_UNION_RETIREMENT_WORKER_CONFIG", "_GPU_UNION_RETIREMENT_WORKER_ATTEMPTED",
                     "_GPU_UNION_RETIREMENT_WORKER_RELEASED", "_AFFINE_GRID_CACHE_MIN_ENTRIES"):
            monkeypatch.setattr(inference, name, getattr(inference, name))
        assets = workers._GpuWorkerInferenceAssets(model=None)
        fence_event = torch.cuda.Event()
        close = cache.close
        close_observations = []

        def observe_close():
            assert fence_event.query()
            close_observations.append(cache.snapshot()["borrowed_bytes"])
            close()

        def preserve_live_credit_before_fence():
            snapshot = cache.snapshot()
            assert snapshot["enabled"]
            assert snapshot["live_bytes"] == snapshot["borrowed_bytes"] == table_bytes

        def unexpected_build(*_args):
            pytest.fail("The exact QSC table should already be cached")

        with cache.borrow(key, np.float64, unexpected_build) as borrowed:
            assert borrowed is not None
            references = tuple(weakref.ref(array) for array in borrowed)
            with torch.cuda.stream(engine._stream):
                pending_render = spherical_cuda.render_spherical_native_resident(engine, view, 1)
                fence_event.record(engine._stream)
            with monkeypatch.context() as scoped:
                scoped.setattr(cache, "close", observe_close)
                receipt = workers._release_gpu_worker_inference_assets(
                    assets, inference_drained=True,
                    wait_for_publications=preserve_live_credit_before_fence)
            assert receipt["released"] and receipt["renderer"]["source_bytes"] == volume.nbytes
            assert close_observations == [table_bytes]
            np.testing.assert_array_equal(pending_render.cpu().numpy(), render_shell_frame(volume, view, 1))
            retired = cache.snapshot()
            assert not retired["enabled"]
            assert retired["entries"] == retired["cache_bytes"] == 0
            assert retired["live_bytes"] == retired["borrowed_bytes"] == table_bytes
        del borrowed
        assert all(reference() is None for reference in references)
        assert cache.snapshot()["live_bytes"] == cache.snapshot()["borrowed_bytes"] == 0
        print(f"Exact CUDA spherical host-cache parity: pixels={baseline.size}, "
              f"sha256={hashlib.sha256(baseline.tobytes()).hexdigest()}, "
              f"builds={replay['builds']}, hits={replay['hits']}, "
              f"h2d_copies={after['h2d_copies'] - before['h2d_copies']}, "
              f"h2d_bytes={2 * table_bytes}, pinned_at_fenced_close={close_observations[0]}, "
              f"retired_live_bytes={cache.snapshot()['live_bytes']}")
    finally:
        engine._stream.synchronize()
        engine.release_inference_assets()
        close_spherical_host_cache(engine)
