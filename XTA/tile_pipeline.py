"""Own closed tile observations through independent configuration science and final OR."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile


class IndependentTileSets:
    """The scheduler thread seals a configuration after every declared observation arrives."""

    def __init__(self, expected_ids):
        self.expected = {key: tuple(ids) for key, ids in expected_ids.items()}
        for key, ids in self.expected.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f'Invalid tile observation inventory for {key}')
        self.pending = {key: {} for key in self.expected}

    def accept(self, key, tile_id, result):
        if key not in self.expected or tile_id not in self.expected[key]:
            raise RuntimeError(f'Unexpected tile observation {key}/{tile_id}')
        received = self.pending.get(key)
        if received is None or tile_id in received:
            raise RuntimeError(f'Duplicate tile observation {key}/{tile_id}')
        if result is not None and (result.model_name, result.view_name, result.config_id, result.tile_id) != (*key, tile_id):
            raise ValueError(f'Tile observation identity differs from its receipt: {key}/{tile_id}')
        received[tile_id] = result
        if len(received) != len(self.expected[key]):
            return None
        del self.pending[key]
        return tuple(received[tile] for tile in self.expected[key] if received[tile] is not None)

    def incomplete(self):
        return {key: {'completed': len(items), 'expected': len(self.expected[key])}
                for key, items in self.pending.items()}


class TileStaging:
    """Private scratch for closed observations and contributions delayed by another owner."""

    def __init__(self, temp_dir, output_dir):
        from .runtime import path_is_memory_backed
        parent = Path(temp_dir)
        if path_is_memory_backed(parent):
            if path_is_memory_backed(Path(output_dir)):
                raise RuntimeError('Independent tiles need disk-backed --temp or --output for retained '
                                   'tile observations, confidence and config handoffs; both are memory-backed')
            parent = Path(output_dir)
        self._directory = tempfile.TemporaryDirectory(prefix='.xta-tile-', dir=parent)
        self.path = Path(self._directory.name)

    def close(self):
        from .runtime import release_memfd_owners_under
        release_memfd_owners_under(self.path)
        self._directory.cleanup()


def tile_result_headroom(physical_bytes, *, shared_reserve, transient_reserve,
                         policy_non_tile_reserve, retained_publication, anon_cap=0):
    """Independent raw tiles may use only RAM left after simultaneous owners."""
    remaining = max(0, int(physical_bytes) - max(int(shared_reserve) + int(transient_reserve),
                                                  int(policy_non_tile_reserve)) - int(retained_publication))
    return min(remaining, int(anon_cap)) if anon_cap > 0 else remaining


def consume_tile_files(mask_path, confidence_path, shape, consume):
    """Return a closed sparse receipt only after the task's own input mappings retire."""
    from .cuda_backend import open_existing_gray_memmap
    from .runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks
    paths = [Path(mask_path)]
    if confidence_path is not None:
        paths.append(Path(confidence_path))
    mappings = []
    try:
        for path in paths:
            mappings.append(open_existing_gray_memmap(path, shape, 'uint8', mode='r+'))
            # Register the file identity before the consumer can unlink it. The
            # finalizer runs only when every view of the mapping is gone.
            close_memmap_array_without_flush(mappings[-1], unlink_path=path)
        result = consume(mappings[0], mappings[1] if len(mappings) == 2 else None)
    except BaseException as error:
        import traceback
        traceback.clear_frames(error.__traceback__)
        mappings.clear()
        for path in paths:
            try:
                wait_for_retired_memmap_unlinks(path=path)
            except Exception as cleanup:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Tile file retirement for {path} also failed: {cleanup}')
        raise
    finally:
        mappings.clear()
    for path in paths:
        wait_for_retired_memmap_unlinks(path=path)
    return result


def analyze_tile_config(observations, *, shape, staging_dir, settings, admission,
                        needs_sam=False, destination=None, destination_lock=None):
    """Assemble and interpolate one sealed tile config under its own workspace admission.

    An available final destination receives the completed config directly. Otherwise
    a packed native-grid handoff releases the dense science workspace before the join.
    """
    from .assembly import _delete_tile_result_storage, finalize_independent_tile_config
    from .confidence_tiles import discard_unpublished_tile_confidence, retire_tile_confidence
    from .config import GIB
    from .cuda_d1 import retire_temporary_binary_volume_storage, tile_intermediate_accumulator_reserve_bytes
    from .finalization import union_volume_into_volume
    from .interpolation import INTERNAL_PACKED_CVOL_FORMAT, write_raw_bbox_mask_store
    from .runtime import allocate_workspace_array
    from .tile_observations import accumulate_independent_tile_observation

    model, view, config = settings['model_name'], settings['view'], settings['config_id']
    workers = settings['slice_workers']
    root = Path(staging_dir) / 'configs' / str(model) / str(view.name) / str(config)
    output = root / 'contribution.cvol'
    unconsumed = iter(observations)

    def analyze():
        root.mkdir(parents=True, exist_ok=False)
        canvas = final = None
        try:
            if observations:
                canvas = allocate_workspace_array(
                    shape=shape, dtype='uint8', path=root / 'observations.u8.dat',
                    desc=f'Independent tile observations {model}/{view.name}/{config}',
                    prefer_memory=False, reserve_bytes=tile_intermediate_accumulator_reserve_bytes())
                for observation in unconsumed:
                    accumulate_independent_tile_observation(observation, tile_accumulator_mm=canvas)
            result = finalize_independent_tile_config(tile_accumulator_mm=canvas, **settings)
            final = result.final_accumulator_mm
            result = replace(result, final_accumulator_mm=None)
            handoff = None
            if final is not None:
                if destination is None:
                    write_raw_bbox_mask_store(final, output, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                                              workers=workers, desc=f'Independent tile handoff {config}')
                    handoff = output
                else:
                    with destination_lock:
                        union_volume_into_volume(destination, final, workers=workers,
                                                 desc=f'Join independent tiles {model}/{view.name}/{config}')
            return result, handoff
        finally:
            if final is not canvas:
                retire_temporary_binary_volume_storage(final)
            retire_temporary_binary_volume_storage(canvas)

    try:
        if not observations:
            return analyze()
        nbytes = math.prod(shape)
        if needs_sam:
            from .sam_resources import run_admitted_sam_call
            return run_admitted_sam_call(admission, 2*nbytes + 4*GIB, f'{model}/{view.name}/tile/{config}',
                                         settings['sam_context'], analyze, {})
        with admission.reserve(2*nbytes, f'Independent tile configuration {model}/{view.name}/{config}'):
            return analyze()
    except BaseException:
        for observation in unconsumed:
            _delete_tile_result_storage(observation)
            discard_unpublished_tile_confidence(observation)
        retire_tile_confidence(model, view.name, config)
        raise


def join_tile_config(path, *, destination, lock, workers):
    """Consume only a completed configuration, on its declared view grid."""
    from .tile_observations import merge_tile_config_store
    merge_tile_config_store(path, destination_mm=destination, destination_lock=lock, workers=workers)
    shutil.rmtree(path)
