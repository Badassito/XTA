"""Ordinary test imports must leave the real numerical runtime intact.

Dependency-free import smoke checks use child processes; ordinary tests use the
installed runtime even when their individual cases mock numerical operations.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "categorical_forward_geometry", "complete_manifest_boundary",
    "cross_mode_geometry_matrix", "cuda_d1_groups", "cuda_finalization",
    "cuda_interpolation", "dsa_regressions", "environment_cleanup",
    "finalization_streaming_union", "gpu_render_optimizations",
    "interpolation_components", "interpolation_scheduler", "lta_execution",
    "lta_postprocessing", "lta_rendering", "lta_runtime", "lta_worker_adapter",
    "media_categorical_cube_resize", "mount_detection",
    "openvino_output_concurrency", "pipeline_streaming_finalization",
    "process_regressions", "pta_augmentation_boundary", "pta_dataset_boundary",
    "pta_geometry_integration", "pta_spawn_scheduler", "pta_worker_boundary",
    "radial_batch_padding", "radial_strided_sampling", "render_batch_image_tee",
    "shared_gaussian", "tile_config_regressions", "tta_output_ownership",
    "tta_pipeline_refactor", "tta_scheduler_boundary",
    "worker_raster_plan_provenance", "spherical_pressure", "publication_memory",
    "radial_retirement_admission", "runtime_task_trace", "spherical_host_admission",
    "spherical_locality", "tta_gpu_asset_release", "tta_policy_shared_union",
    "tta_policy_parent_admission",
)


@pytest.mark.parametrize("numerical_first", (False, True))
def test_test_imports_preserve_numerical_dependencies(numerical_first):
    program = textwrap.dedent("""
        import importlib
        import json
        import sys
        import numpy as np

        names = json.loads(sys.argv[1])
        numerical_first = sys.argv[2] == 'True'

        def numerical_check():
            import cv2
            from scipy import ndimage
            from XTA import _deps
            from XTA.reconciliation_components import component_statistics
            assert _deps.cv2 is cv2
            assert _deps.ndi is ndimage
            image = np.zeros((9, 9), np.uint8)
            image[1:3, 1:3] = 1
            image[6, 6] = 1
            assert cv2.connectedComponents(image, connectivity=4)[0] == 3
            assert ndimage.label(image)[1] == 2
            volume = np.stack((image, image, image))
            stats = component_statistics(lambda a, b: volume[a:b], volume.shape, memory_mib=2)
            assert stats['foreground_voxels'] == 15
            assert stats['largest_component_voxels'] == 12
            assert stats['component_count'] == 2
            return tuple(importlib.import_module(name) for name in
                         ('cv2', 'scipy', 'scipy.ndimage', 'tifffile', 'tqdm'))

        before = numerical_check() if numerical_first else None
        for name in reversed(names) if numerical_first else names:
            importlib.import_module('tests.test_' + name)
        after = numerical_check()
        if before is not None:
            assert all(first is last for first, last in zip(before, after))
        for module in tuple(sys.modules.values()):
            assert type(module).__name__ != '_StubModule', repr(module)
        print('real SciPy/OpenCV operations and dependency identities preserved')
    """)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, json.dumps(MODULES), str(numerical_first)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "real SciPy/OpenCV operations and dependency identities preserved" in completed.stdout


def test_mock_gpu_finalization_preserves_lazy_numba_registrations():
    pytest.importorskip("numba", reason="Numba-specific lazy import registration check")
    program = textwrap.dedent("""
        import sys
        import unittest
        import numpy as np
        import numba
        from numba.core import pythonapi, types
        from tests.test_cuda_finalization import CudaFinalizationContractTests

        case = CudaFinalizationContractTests(
            'test_blockwise_3d_ccl_matches_independent_reference_across_layouts')
        result = unittest.TextTestRunner().run(unittest.TestSuite((case,)))
        assert result.wasSuccessful()
        if types.PolynomialType in pythonapi._unboxers.functions:
            assert 'numba.np.polynomial.polynomial_core' in sys.modules

        @numba.njit
        def add_one(values):
            return values + 1

        np.testing.assert_array_equal(add_one(np.arange(5)), np.arange(1, 6))
        print('Numba compilation succeeds after mocked GPU finalization')
    """)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "Numba compilation succeeds after mocked GPU finalization" in completed.stdout
