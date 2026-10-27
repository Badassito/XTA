"""Bounded discovery overlap before any TTA processing begins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from .unification.tta_manifest import _file_identity


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
        source_info = probe(input_path)
        identities = identity_future.result()
    source_after = _file_identity(input_path)
    captured_source: Mapping[str, Any] = identities['source']
    if source_before != source_after or dict(captured_source) != source_before:
        raise RuntimeError(
            'TTA source artifact changed during startup discovery; refusing mixed '
            f'probe/provenance results: path={input_path}'
        )
    return identities, source_info
