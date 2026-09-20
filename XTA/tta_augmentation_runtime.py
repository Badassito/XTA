"""Bounded render-once / infer-many GPU execution for external TTA policies.

A model batch is the image ownership boundary. Every pass consumes the same
rendered tensor before it is released. Only masks/support move to the host.
Completed mask copies retire on the existing event-fenced background lanes.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    GpuPrefetchedYoloBatch, InMemoryYoloVolumeSource,
    azimuthal_batch_padding_frame_specs, mirrored_azimuthal_parent_crop,
    prediction_result_frame_spec,
)
from .tta_augmentation import SpatialReplay, worker_policy
from .tta_augmentation_config import policy_seed
from .tta_augmentation_retirement import (
    CoverageTransfers, SliceMetadataAccumulator, reserve_policy_retirement,
)


class PolicyTensorBatchSource(InMemoryYoloVolumeSource):
    """One already-staged batch; subclassing retains the registered YOLO source contract."""
    _tta_already_gpu_staged = True

    def __init__(self, *, batch: Any, paths: Any, info: Any, specs: Any,
                 real_count: int, name: str, replay: list[SpatialReplay] | None) -> None:
        self.batch, self.paths, self.info, self.specs = batch, paths, info, specs
        self.nf = int(real_count)
        self.bs = len(paths)
        self.yield_nf = self.bs
        self.channel_count = int(batch._tta_gpu_tensor.shape[1])
        self._tta_channel_count = self.channel_count
        self.out_size = int(batch._tta_gpu_tensor.shape[-1])
        self.name, self.mode = name, 'image'
        self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)
        self.azimuthal_padding_count = sum(int(s is not None and s.is_azimuthal_padding) for s in specs)
        self.count = 0
        self.replay = replay
        self._replay_ready = None
        if replay is not None:
            import torch
            self._replay_ready = torch.cuda.Event()
            self._replay_ready.record(torch.cuda.current_stream(batch._tta_gpu_tensor.device))

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> 'PolicyTensorBatchSource':
        self.count = 0
        return self

    def __next__(self) -> Any:
        if self.count:
            raise StopIteration
        self.count = self.bs
        return self.paths, self.batch, self.info

    def result_frame_spec(self, result_index: int) -> Any:
        return self.specs[result_index] if 0 <= result_index < len(self.specs) else None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def restore_prediction(self, spec: Any, payload: Any) -> Any:
        if self.replay is None or payload is None:
            return payload
        from .inference import GpuFlattenedRetinaPayload
        import torch
        if not isinstance(payload, GpuFlattenedRetinaPayload):
            raise RuntimeError('External TTA requires GPU flattened masks; CPU mask fallback cannot bypass inverse mapping')
        if payload.union_gpu is None:
            return payload
        replay = self.replay[int(spec.result_index)]
        stream = torch.cuda.current_stream(payload.union_gpu.device)
        if payload.ready_event is not None:
            stream.wait_event(payload.ready_event)
        if self._replay_ready is not None:
            stream.wait_event(self._replay_ready)
        old_planes = [payload.union_gpu] + ([payload.conf_gpu] if payload.conf_gpu is not None else [])
        for value in [*old_planes, replay.inverse_grid, replay.valid]:
            value.record_stream(stream)
        with torch.inference_mode():
            restored = replay.restore_planes(old_planes)
        payload.union_gpu = restored[0]
        payload.conf_gpu = restored[1] if len(restored) > 1 else None
        payload.device_refs = tuple(payload.device_refs or ()) + tuple(old_planes) + (replay.inverse_grid, replay.valid)
        payload.ready_event = torch.cuda.Event()
        payload.ready_event.record(stream)
        return payload


class _CoverageWriter:
    """A task-sized packed mmap, compressed atomically without a task-sized RAM copy."""
    def __init__(self, root: Path, task: dict[str, Any], out_size: int, total_slots: int,
                 *, backend: str = '', policy_sha256: str = '') -> None:
        self.task = task
        self.backend, self.policy_sha256 = backend, policy_sha256
        token = hashlib.sha256(f'{task["view"].name}/{task["job_id"]}/{task["task_id"]}'.encode()).hexdigest()[:20]
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f'task{int(task["task_id"]):06d}_p{int(task["view"].augmentation_pass):03d}_{token}.npz'
        self.raw_path = self.path.with_suffix('.validity.partial.npy')
        self.array = np.lib.format.open_memmap(self.raw_path, mode='w+', dtype=np.uint8,
                                               shape=(total_slots, out_size, (out_size + 7) // 8))
        self.seeds = np.zeros(total_slots, dtype=np.uint64)
        self.destinations = np.full(total_slots, -1, dtype=np.int64)
        self.mirrors = np.zeros(total_slots, dtype=np.bool_)
        self.written = np.zeros(total_slots, dtype=np.bool_)
        self.out_size = out_size

    def put(self, slot: int, spec: Any, seed: int, replay: SpatialReplay) -> None:
        self.put_array(slot, spec, seed, replay.packed_validity())

    def put_array(self, slot: int, spec: Any, seed: int, packed: np.ndarray) -> None:
        if self.written[slot]:
            raise RuntimeError('duplicate coverage slot')
        self.array[slot] = packed
        self.seeds[slot] = seed
        self.destinations[slot] = int(spec.global_destination_index)
        self.mirrors[slot] = bool(spec.mirror_azimuthal_u)
        self.written[slot] = True

    def finish(self) -> dict[str, Any]:
        if not bool(np.all(self.written)):
            raise RuntimeError(f'incomplete augmentation coverage: {self.path}')
        metadata = {
            'schema': 'xta.tta.inverse-support/1', 'view': self.task['view'].name,
            'job_id': self.task['job_id'], 'task_id': int(self.task['task_id']),
            'pass_index': int(self.task['view'].augmentation_pass),
            'slice_start': int(self.task.get('slice_start', 0)),
            'logical_slices': int(self.task['slice_count']),
            'raster_shape': [self.out_size, self.out_size], 'bitorder': 'big',
            'M_out_to_processing': np.asarray(self.task['M_out_to_processing']).tolist(),
            'parent_crop': self.task.get('parent_crop'),
            'meaning': '1=converged locally invertible in-frame inverse; 0=unknown, not a negative vote',
            'coordinates': 'unaugmented model raster; compose with recorded processing affine and view projection',
            'source_acquisition_coverage_included': False,
        }
        if self.backend:
            metadata.update(backend=self.backend, policy_sha256=self.policy_sha256)
        pending = self.path.with_suffix('.npz.partial')
        try:
            import zipfile

            with pending.open('wb') as stream:
                # Support is lossless auxiliary output; lower deflate effort keeps
                # policy retirement from waiting on CPU compression. Stream each
                # NPY member in NumPy's bounded chunks instead of copying the map.
                arrays = dict(validity_bits=self.array, seeds=self.seeds,
                              global_destinations=self.destinations, mirror_azimuthal_u=self.mirrors,
                              metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                     compresslevel=1) as archive:
                    for name, value in arrays.items():
                        with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                            np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
            pending.replace(self.path)
            return {**metadata, 'path': str(self.path)}
        finally:
            self.close()
            pending.unlink(missing_ok=True)

    def close(self) -> None:
        array, self.array = self.array, None
        if array is not None:
            array._mmap.close()
        self.raw_path.unlink(missing_ok=True)


def _padding_metadata(task: dict[str, Any], mask_path: Path | None, conf_path: Path | None,
                      shape: Any, batch_size: int) -> dict[str, Any]:
    if mask_path is None:
        return {}
    specs = azimuthal_batch_padding_frame_specs(task['view'], int(task['slice_count']), int(batch_size),
                                               slice_offset=int(task.get('slice_start', 0)))
    frames = []
    for spec in specs:
        frame = {'ordinal': int(spec.azimuthal_padding_ordinal or 0),
                 'destination': int(spec.global_destination_index),
                 'mirror_azimuthal_u': bool(spec.mirror_azimuthal_u)}
        if task['kind'] == 'tile':
            crop = tuple(int(v) for v in task['parent_crop'])
            width = int(task['threshold_plane_shape'][1])
            frame['parent_crop'] = mirrored_azimuthal_parent_crop(crop, width) if spec.mirror_azimuthal_u else crop
        frames.append(frame)
    return {'azimuthal_padding_count': len(specs), 'azimuthal_padding_mask_path': str(mask_path),
            'azimuthal_padding_conf_path': str(conf_path) if conf_path else '',
            'azimuthal_padding_destinations': tuple(s.global_destination_index for s in specs),
            'azimuthal_padding_frames': tuple(frames), 'azimuthal_padding_shape': tuple(shape)}


def _validate_policy_parent_group(pass_tasks: list[dict[str, Any]], shape: tuple[int, ...]) -> bool:
    """Shared policy parents require separate canvases and one disjoint slice lease."""
    modes = {str(task.get('result_mode', 'file')) for task in pass_tasks}
    if modes == {'file'}:
        return False
    if modes != {'direct_union'}:
        raise RuntimeError('Policy passes must use a uniform file or shared-parent result contract')
    names, paths = set(), set()
    first_start = int(pass_tasks[0].get('slice_start', 0))
    for task in pass_tasks:
        if task.get('kind') != 'fullframe' or not task.get('bounded_parent_admission', False):
            raise RuntimeError('Shared policy outputs require admitted full-frame parents')
        parent_shape = tuple(int(v) for v in task.get('processing_shape', ()))
        start, count = int(task.get('slice_start', 0)), int(task['slice_count'])
        if (len(parent_shape) != 3 or min(parent_shape) <= 0
                or len(shape) != 3 or tuple(shape) != (count, *parent_shape[1:])
                or int(task.get('union_num_slices', parent_shape[0])) != parent_shape[0]
                or start != first_start or start < 0 or start + count > parent_shape[0]):
            raise RuntimeError('Shared policy output has an invalid or mismatched parent slice window')
        key = (str(task.get('model_name', '')), str(task['view'].name))
        if key in names:
            raise RuntimeError(f'Policy passes target the same shared parent {key}')
        names.add(key)
        for name in ('result_mask_path', 'result_conf_path'):
            raw = task.get(name)
            if raw is None and name == 'result_conf_path':
                continue
            if not raw:
                raise RuntimeError(f'Shared policy output is missing {name}')
            # Do not resolve /proc links: different equally named memfds can
            # resolve to the same display name despite owning separate storage.
            path = str(Path(str(raw)).absolute())
            if path in paths:
                raise RuntimeError('Policy outputs would overlap the same shared backing path')
            paths.add(path)
    return True


def _open_policy_sibling_outputs(task: dict[str, Any], *, shape: tuple[int, ...],
                                 padding_count: int, shared_parent: bool,
                                 owned: list[Any]) -> tuple[tuple[Any, ...], tuple[Path | None, Path | None]]:
    """Open one sibling lease without touching other shared-parent slices."""
    paths = (Path(str(task['result_mask_path'])),
             Path(str(task['result_conf_path'])) if task.get('result_conf_path') else None)
    buffers: list[Any] = []
    padding_paths: list[Path | None] = [None, None]
    owned_start = len(owned)
    try:
        for path in paths:
            if path is None:
                buffers.append(None)
            elif shared_parent:
                parent_shape = tuple(int(v) for v in task['processing_shape'])
                required = int(np.prod(parent_shape, dtype=np.int64))
                if path.stat().st_size < required:
                    # numpy.memmap(r+) can grow a short file. A shared lease
                    # must never resize or truncate the parent's live backing.
                    raise RuntimeError(f'Shared policy parent backing is shorter than its declared shape: {path}')
                root = np.memmap(path, mode='r+', dtype=np.uint8, shape=parent_shape)
                owned.append(root)
                start = int(task.get('slice_start', 0))
                buffers.append(root[start:start + int(task['slice_count'])])
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                root = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
                # w+ truncates the backing; newly extended bytes are already
                # zero. Touching the whole lease here needlessly dirties pages.
                owned.append(root)
                buffers.append(root)
        for index, path in enumerate(paths):
            if path is None or padding_count <= 0:
                buffers.append(None)
                continue
            directory = task.get('azimuthal_padding_dir')
            if shared_parent and not directory:
                raise RuntimeError('Shared policy seam outputs require azimuthal_padding_dir')
            if directory:
                root_dir = Path(str(directory))
                root_dir.mkdir(parents=True, exist_ok=True)
                identity = '/'.join(str(task.get(name, '')) for name in
                                    ('model_name', 'job_id', 'task_id', 'slice_start'))
                token = hashlib.sha256(f'{task["view"].name}/{identity}'.encode()).hexdigest()[:20]
                plane = 'mask' if index == 0 else 'conf'
                padding_paths[index] = root_dir / (
                    f'policy_task{int(task["task_id"]):06d}_p{int(task["view"].augmentation_pass):03d}_'
                    f'{token}.{plane}.seam.u8.dat')
            else:
                padding_paths[index] = path.with_name(path.name + '.seam')
            root = np.memmap(padding_paths[index], mode='w+', dtype=np.uint8,
                             shape=(padding_count, shape[1], shape[2]))
            owned.append(root)
            buffers.append(root)
        return tuple(buffers), (padding_paths[0], padding_paths[1])
    except BaseException:
        for root in owned[owned_start:]:
            root._mmap.close()
        del owned[owned_start:]
        raise


def predict_policy_source(model: Any, source: Any, *, task: dict[str, Any], cfg: Any,
                          predict_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Infer base + augmented copies per GPU batch; never re-render for policy copies."""
    import torch
    from .inference import predict_source_and_accumulate, canonical_single_device
    from .config import quantize_uses_fp16
    settings = task['augmentation_settings'].for_backend('gpu')
    settings.assert_unchanged()
    adapter = worker_policy(settings, device=canonical_single_device(str(cfg.device)), batch_size=int(cfg.batch))
    pass_tasks = [task, *task['augmentation_pass_tasks']]
    if len(pass_tasks) != settings.ratio:
        raise RuntimeError('augmentation task fan-out does not match total pass count')
    shape = tuple(int(v) for v in predict_kwargs['view_union_mm'].shape)
    shared_parent = _validate_policy_parent_group(pass_tasks, shape)
    total = int(task['slice_count'])
    padding_count = int(getattr(source, 'azimuthal_padding_count', 0))
    targets = [(predict_kwargs['view_union_mm'], predict_kwargs['view_confmap_mm'],
                predict_kwargs.get('azimuthal_padding_union_mm'), predict_kwargs.get('azimuthal_padding_confmap_mm'))]
    owned: list[Any] = []
    writers: list[_CoverageWriter | None] = [None]
    stats = [dict(prediction_count=0, frames_with_predictions=0, device_hole_filled_frames=0,
                  azimuthal_padding_processed=0, slice_meta=None) for _ in pass_tasks]
    metadata = [SliceMetadataAccumulator(total) for _ in pass_tasks]
    pending: deque[tuple[int, int, int, Future]] = deque()
    transfers = CoverageTransfers()
    reservation = None
    ownership_transferred = False
    rendered_batches = 0
    def merge(pass_index: int, start: int, count: int, result: dict[str, Any]) -> None:
        for key in ('prediction_count', 'frames_with_predictions', 'device_hole_filled_frames', 'azimuthal_padding_processed'):
            stats[pass_index][key] += int(result.get(key, 0))
        metadata[pass_index].merge(result.get('slice_meta'), start, count)
    def settle_one() -> None:
        p, start, count, future = pending.popleft()
        merge(p, start, count, future.result())
    def close_resources() -> None:
        import sys
        active_error = sys.exc_info()[1]
        error = None
        while pending:
            try:
                settle_one()
            except BaseException as exc:
                if error is None:
                    error = exc
        try:
            transfers.close()
        except BaseException as exc:
            if error is None:
                error = exc
        for writer in writers:
            if writer is not None:
                try:
                    writer.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
        for mm in owned:
            try:
                mm._mmap.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        owned.clear()
        if active_error is None and error is not None:
            raise error
    def finish_task() -> dict[str, Any]:
        try:
            while pending:
                settle_one()
            transfers.close()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='tta-support') as pool:
                futures = [pool.submit(w.finish) for w in writers if w is not None]
                records = [future.result() for future in futures]
            settings.assert_unchanged()
            for index, accumulator in enumerate(metadata):
                stats[index]['slice_meta'] = accumulator.finish()
            result = dict(stats[0])
            result['augmentation_results'] = stats[1:]
            result['augmentation_records'] = records
            result['augmentation_execution'] = {
                'backend': 'gpu', 'policy_sha256': settings.content_sha256,
                'rendered_batches': rendered_batches, 'model_batches': rendered_batches * settings.ratio,
                'pass_count': settings.ratio, 'source_render_replays': 0,
            }
            return result
        finally:
            close_resources()
    try:
        from .inference import gpu_union_flush_overlap_enabled
        if predict_kwargs.get('defer_device_union_flush', False) and gpu_union_flush_overlap_enabled():
            # Reserve before allocating task maps so CPU publication cannot queue unbounded tasks.
            reservation = reserve_policy_retirement()
        for i, sibling in enumerate(pass_tasks[1:], 1):
            buffers, pad_paths = _open_policy_sibling_outputs(
                sibling, shape=shape, padding_count=padding_count,
                shared_parent=shared_parent, owned=owned)
            pad_shape = (padding_count, shape[1], shape[2])
            targets.append(buffers)
            stats[i].update(_padding_metadata(sibling, *pad_paths, pad_shape, int(cfg.batch)))
            writers.append(_CoverageWriter(Path(task['augmentation_support_dir']), sibling,
                                           int(predict_kwargs['out_size']), total + padding_count,
                                           backend='gpu', policy_sha256=settings.content_sha256)
                           if settings.coverage == 'packed' else None)
        batch_start = 0
        for paths, original_batch, info in iter(source):
            if hasattr(original_batch, 'wait_ready'):
                tensor = original_batch.wait_ready()
            else:
                # Existing CPU-render fallback may upload once. There is NEVER a GPU->CPU
                # image transfer for augmentation; no CPU policy is constructed.
                values = np.stack([np.asarray(im)[:, :, None] if np.asarray(im).ndim == 2 else np.asarray(im)
                                   for im in original_batch])
                tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2))).to(
                    canonical_single_device(str(cfg.device)), non_blocking=False)
                tensor = tensor.to(torch.float16 if quantize_uses_fp16(cfg.quantize) else torch.float32) / 255.
            if not tensor.is_cuda:
                raise RuntimeError('external TTA prediction received a non-CUDA rendered tensor')
            real_count = max(0, min(len(paths), total - batch_start))
            if real_count <= 0:
                raise RuntimeError('unexpected standalone synthetic batch')
            original_specs = [prediction_result_frame_spec(source, batch_start + j, num_frames=total)
                              for j in range(len(paths))]
            local_specs = [None if spec is None else replace(spec, result_index=j,
                            task_index=spec.task_index - batch_start if not spec.is_azimuthal_padding else spec.task_index)
                           for j, spec in enumerate(original_specs)]
            for i, sibling in enumerate(pass_tasks):
                # Retain at most two event-fenced mask retirements, independently of N.
                if len(pending) >= 2:
                    settle_one()
                replay = None
                active_tensor = tensor
                if i:
                    trajectory = f'{task["view"].name}/{task["kind"]}/{task["job_id"]}'
                    seeds = [policy_seed(settings, trajectory=trajectory, pass_index=i,
                                         slice_index=(int(spec.global_destination_index) if spec else
                                                      int(task.get('slice_start', 0)) + total - 1),
                                         lease_start=int(task.get('slice_start', 0)))
                             for spec in original_specs]
                    active_tensor, replay = adapter.apply(tensor, seeds)
                    writer = writers[i]
                    if writer is not None:
                        transfers.submit(writer, [
                            (batch_start + j, spec, seeds[j], replay[j])
                            for j, spec in enumerate(original_specs) if spec is not None
                        ])
                batch = GpuPrefetchedYoloBatch(original_batch, gpu_tensor=active_tensor, source_label=str(sibling['view'].name))
                local_source = PolicyTensorBatchSource(batch=batch, paths=paths, info=info, specs=local_specs,
                                                       real_count=real_count, name=str(sibling['view'].name), replay=replay)
                local_source._tta_announce_prediction = batch_start == 0
                mask, conf, pad_mask, pad_conf = targets[i]
                kwargs = {**predict_kwargs, 'source_label': f'{sibling["view"].name}/{sibling["job_id"]}',
                          'num_frames': real_count, 'cfg': cfg,
                          'view_union_mm': mask[batch_start:batch_start + real_count],
                          'view_confmap_mm': conf[batch_start:batch_start + real_count] if conf is not None else None,
                          'azimuthal_padding_union_mm': pad_mask, 'azimuthal_padding_confmap_mm': pad_conf,
                          'device_union_consumer': None, 'require_device_union': False,
                          'owned_disjoint_output': True,
                          'require_proto_hole_treatment': False, 'defer_device_union_flush': True,
                          'device_hole_fill': False}
                result = predict_source_and_accumulate(model, local_source, **kwargs)
                future = result.get('_device_union_flush_future')
                if isinstance(future, Future):
                    pending.append((i, batch_start, real_count, future))
                else:
                    merge(i, batch_start, real_count, result)
            batch_start += len(paths)
            rendered_batches += 1
        if batch_start < total:
            raise RuntimeError(f'augmentation source exhausted at {batch_start}/{total} frames')
        if reservation is not None:
            result = dict(stats[0])
            # The worker attaches its known seam paths before deferred stats arrive.
            result['azimuthal_padding_processed'] = padding_count
            result['_device_union_flush_future'] = reservation.submit(finish_task)
            ownership_transferred = True
            return result
        ownership_transferred = True
        return finish_task()
    finally:
        try:
            if not ownership_transferred:
                close_resources()
        finally:
            if reservation is not None:
                reservation.close()
