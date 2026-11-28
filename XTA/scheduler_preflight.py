"""Capability checks shared by the scalar and sweep TTA launchers."""

import os
import time


_native_preflight_reported = False


def preflight_tta_scheduler(*, gpu_device_count, cpu_enabled, tile_configs, views=(),
                            d1_owner_groups=None):
    from .experimental_features import d1_owner_groups_requested
    from .native_scheduler import require_native_scheduler

    if gpu_device_count and cpu_enabled:
        raise ValueError('Native detector scheduler does not yet support hybrid OpenVINO/CUDA dispatch')
    if d1_owner_groups is None:
        d1_owner_groups = d1_owner_groups_requested()
    if gpu_device_count > 1 and d1_owner_groups:
        raise ValueError('Native detector scheduler does not yet support multi-worker D1 owner groups')
    if cpu_enabled and views:
        from .runtime import cpu_inference_supports_view
        unsupported = [v.name for v in views if not cpu_inference_supports_view(v)]
        if unsupported:
            raise ValueError(f'OpenVINO cannot execute requested detector views {unsupported}')
    from .native_cpu import load_native_cpu

    load_native_cpu()
    scheduler_options = {'tiles': True} if tile_configs else {}
    global _native_preflight_reported
    if _native_preflight_reported:
        require_native_scheduler(**scheduler_options)
        return
    started = time.perf_counter()
    print(f'[startup] Checking native scheduler in pid {os.getpid()} at {time.time_ns()} Unix ns.',
          flush=True)
    failed = True
    try:
        require_native_scheduler(**scheduler_options)
        failed = False
    finally:
        status = 'failed' if failed else 'finished'
        print(f'[startup] Native scheduler check in pid {os.getpid()} {status} in '
              f'{time.perf_counter() - started:.3f}s.', flush=True)
    _native_preflight_reported = True
