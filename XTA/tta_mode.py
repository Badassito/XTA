"""Production TTA runner used by the unified CLI."""

from __future__ import annotations

import time


def run() -> None:
    """Execute one already-validated TTA invocation from ``sys.argv``."""

    # Keep discovery dependency-light: validation and help happen in cli.py
    # before this production dependency surface is imported.
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    print(f"[startup] Importing TTA runtime at {started_wall_ns} Unix ns.", flush=True)
    from .media import abort_streaming_producers
    from .outputs import nrrd_layer_sink, set_nrrd_layer_sink
    from .pipeline import main
    from .runtime import runtime_trace_event

    elapsed = time.perf_counter() - started
    print(f"[startup] TTA runtime imported in {elapsed:.3f}s.", flush=True)
    runtime_trace_event('startup_tta_import_done', started_wall_time_ns=started_wall_ns,
                        wall_seconds=elapsed)

    try:
        main()
    except BaseException:
        abort_streaming_producers("fatal error in main()")
        fatal_layer_sink = nrrd_layer_sink()
        if fatal_layer_sink is not None:
            try:
                fatal_layer_sink.shutdown()
            except Exception:
                pass
            set_nrrd_layer_sink(None)
        raise


__all__ = ["run"]
