"""Lightweight entry point for spawned TTA detector actors."""

from __future__ import annotations

import os
import pickle
import sys
import time


def loaded_runtime_versions():
    """Describe already imported numerical libraries without loading anything."""
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "numba", "llvmlite", "torch", "cupy", "tensorrt", "openvino"):
        module = sys.modules.get(name)
        version = vars(module).get("__version__") if module is not None else None
        if isinstance(version, str):
            versions[name] = version[:128]
    return versions


def start_detector_worker(kind, worker_id, model_path, init_payload, task_queue, result_queue):
    """Measure worker imports before reconstructing the scientific configuration."""
    prefix = f"[startup] {kind} worker {worker_id} (pid {os.getpid()})"
    bootstrap_wall_ns = time.time_ns()
    bootstrap_mono_ns = time.monotonic_ns()
    print(f"{prefix} bootstrap entered at {bootstrap_wall_ns} Unix ns; importing worker runtime.",
          flush=True)
    import_started_ns = time.monotonic_ns()
    from . import workers

    import_done_ns = time.monotonic_ns()
    init_dict = pickle.loads(init_payload)
    init_done_ns = time.monotonic_ns()
    receipt = dict(
        bootstrap_wall_time_ns=bootstrap_wall_ns,
        bootstrap_monotonic_ns=bootstrap_mono_ns,
        import_seconds=(import_done_ns - import_started_ns) / 1e9,
        init_decode_seconds=(init_done_ns - import_done_ns) / 1e9,
    )
    print(f"{prefix} worker runtime imported in {receipt['import_seconds']:.3f}s; "
          f"configuration decoded in {receipt['init_decode_seconds']:.3f}s.", flush=True)
    entry = {"gpu": workers._gpu_inference_worker_main,
             "cpu": workers._cpu_inference_worker_main}[kind]
    entry(worker_id, model_path, init_dict, task_queue, result_queue, bootstrap_receipt=receipt)
