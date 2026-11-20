"""Bounded discovery overlap before any TTA processing begins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .env_flags import env_flag
from .unification.tta_manifest import (
    _file_identity,
    _sweep_startup_cache_path,
    _write_sweep_startup_cache,
)


def _probe_source(
    input_path: Path,
    source_identity: Mapping[str, Any],
    probe: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    cache_path = _sweep_startup_cache_path("source", Path(source_identity["path"]))
    if cache_path is None:
        return probe(input_path)
    cached = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached is not None and cached.get("source") != dict(source_identity):
        raise RuntimeError(f"TTA source artifact changed within confidence sweep: {input_path}")
    # This setting selects tail-timestamp discovery ("0") versus a complete packet scan ("1").
    mode = "1" if env_flag("YOLO_TTA_FRAME_COUNT_FULL_SCAN", False) else "0"
    probes = {} if cached is None else cached["probes"]
    if mode in probes:
        return dict(probes[mode])
    info = probe(input_path)
    if _file_identity(input_path) != dict(source_identity):
        raise RuntimeError(f"TTA source artifact changed during startup discovery: {input_path}")
    probes[mode] = info
    _write_sweep_startup_cache(cache_path, {"source": dict(source_identity), "probes": probes})
    return info


def capture_and_probe_tta_startup(
    *,
    input_path: Path,
    capture_identities: Callable[[], dict[str, Any]],
    probe: Callable[[Path], dict[str, Any]],
    invalidate_manifest: Callable[[], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join model identity and exact source discovery before returning.

    Invalidate the old successful association first, including when discovery
    fails. One auxiliary thread reads model digests while the caller probes the
    source. Executor shutdown joins that reader on *every* exit path. The source
    snapshots bracket both readers so their results cannot describe different
    ordinary filesystem versions of the input.
    """
    invalidate_manifest()
    source_before = _file_identity(input_path)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='tta-startup-identity') as executor:
        identity_future = executor.submit(capture_identities)
        source_info = _probe_source(input_path, source_before, probe)
        identities = identity_future.result()
    source_after = _file_identity(input_path)
    captured_source: Mapping[str, Any] = identities['source']
    if source_before != source_after or dict(captured_source) != source_before:
        raise RuntimeError(
            'TTA source artifact changed during startup discovery; refusing mixed '
            f'probe/provenance results: path={input_path}'
        )
    return identities, source_info
