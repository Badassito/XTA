"""Reuse native and eagerly resampled gray8 inputs across scalar TTA children."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from .json_publication import write_json_atomic
from .unification.tta_manifest import _file_identity


def _open_or_build(directory, name, input_path, shape, streaming, build):
    """Return a shared mapping or defer to branch-local preparation when space is limited.

    The sweep parent owns a fresh directory and starts the next child only after
    the previous one exits. Only a successful producer publishes the ready
    receipt. The first cutoff can still overlap streaming work with inference.
    """
    destination = directory / name
    receipt = directory / (name + '.json')
    identity = _file_identity(input_path)
    metadata = {'source': identity, 'shape_tyx': list(shape), 'dtype': 'uint8', 'streaming': streaming}
    nbytes = math.prod(shape)
    if receipt.is_file():
        if json.loads(receipt.read_text(encoding='utf-8')) != metadata:
            raise RuntimeError('Input video or dimensions changed between confidence cutoffs')
        if destination.stat().st_size != nbytes:
            raise RuntimeError('Shared confidence-sweep input volume is incomplete')
        print(f'Reusing {name} for this confidence cutoff: {destination}', flush=True)
        return np.memmap(destination, dtype=np.uint8, mode='r', shape=shape)

    if os.environ.get('XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT') == '1':
        # Later cutoffs running concurrently must never build or overwrite a
        # shared mapping another sibling may be reading. Only the producer,
        # which runs alone, publishes shared inputs.
        print(f'No shared {name} from the producer; concurrent cutoff uses branch-local preparation.', flush=True)
        return None
    if shutil.disk_usage(directory).free < nbytes:
        print(f'Insufficient scratch for shared {name} ({nbytes} bytes); using branch-local preparation.', flush=True)
        return None

    def publish_ready():
        if _file_identity(input_path) != identity:
            raise RuntimeError('Input video changed during confidence-sweep preprocessing')
        write_json_atomic(receipt, metadata)

    result = build(destination, {'on_verified': publish_ready} if streaming else {})
    if not streaming:
        publish_ready()
    return result


def open_or_decode_sweep_input(
    directory: Path, input_path: Path, shape: tuple[int, int, int], *,
    streaming: bool, decoder,
) -> np.ndarray | None:
    def build(destination, options):
        return decoder(
            input_video=input_path, out_dat=destination, num_frames=shape[0],
            height=shape[1], width=shape[2], overwrite=True,
            prefer_memory=False, prefer_memfd=False, **options,
        )

    return _open_or_build(directory, 'source.gray8', input_path, shape, streaming, build)


def open_or_resize_sweep_cube(directory, input_path, source, shape, *, streaming, resize, workers):
    # Streaming and eager scalar resize paths can round differently. Keep the selected one.
    def build(destination, options):
        return resize(source, shape, destination, prefer_memory=False, workers=workers, **options)

    return _open_or_build(directory, 'cube.gray8', input_path, shape, streaming, build)
