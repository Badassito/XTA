"""Render-once CPU TTA passes through the persistent OpenVINO inference queue."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .geometry import prediction_result_frame_spec
from .tta_augmentation_config import policy_seed
from .tta_augmentation_cpu import worker_cpu_policy
from .tta_augmentation_retirement import SliceMetadataAccumulator
from .tta_augmentation_runtime import (
    _CoverageWriter, _open_policy_sibling_outputs, _padding_metadata, _validate_policy_parent_group,
)


class CpuPolicyBatchSource:
    """One batch owns the replay until all its async inference callbacks settle."""
    def __init__(self, paths, images, info, specs, replay):
        self.paths, self.images, self.info = paths, images, info
        self.specs, self.replay = specs, replay

    def __iter__(self):
        yield self.paths, self.images, self.info

    def result_frame_spec(self, index):
        return self.specs[index] if 0 <= index < len(self.specs) else None

    def restore_prediction_planes(self, spec, planes):
        if self.replay is None:
            return planes
        return self.replay[int(spec.result_index)].restore_planes(planes)


def predict_cpu_policy_source(runner: Any, source: Any, *, task: dict[str, Any], cfg: Any,
                              predict_kwargs: dict[str, Any]) -> dict[str, Any]:
    settings = task['augmentation_settings'].for_backend('cpu')
    settings.assert_unchanged()
    adapter = worker_cpu_policy(settings)
    pass_tasks = [task, *task['augmentation_pass_tasks']]
    if len(pass_tasks) != settings.ratio:
        raise RuntimeError('CPU augmentation task fan-out does not match total pass count')
    shape = tuple(int(v) for v in predict_kwargs['view_union_mm'].shape)
    shared_parent = _validate_policy_parent_group(pass_tasks, shape)
    total = int(task['slice_count'])
    padding_count = int(getattr(source, 'azimuthal_padding_count', 0))
    targets = [(predict_kwargs['view_union_mm'], predict_kwargs['view_confmap_mm'],
                predict_kwargs.get('azimuthal_padding_union_mm'), predict_kwargs.get('azimuthal_padding_confmap_mm'))]
    owned, writers = [], [None]
    stats = [dict(prediction_count=0, frames_with_predictions=0, device_hole_filled_frames=0,
                  azimuthal_padding_processed=0, slice_meta=None) for _ in pass_tasks]
    metadata = [SliceMetadataAccumulator(total) for _ in pass_tasks]
    rendered_batches = 0
    try:
        for i, sibling in enumerate(pass_tasks[1:], 1):
            buffers, pad_paths = _open_policy_sibling_outputs(
                sibling, shape=shape, padding_count=padding_count,
                shared_parent=shared_parent, owned=owned)
            targets.append(buffers)
            stats[i].update(_padding_metadata(sibling, *pad_paths,
                                             (padding_count, shape[1], shape[2]), int(cfg.batch)))
            writers.append(_CoverageWriter(Path(task['augmentation_support_dir']), sibling,
                                           int(predict_kwargs['out_size']), total + padding_count,
                                           backend='cpu', policy_sha256=settings.content_sha256)
                           if settings.coverage == 'packed' else None)
        batch_start = 0
        for paths, original, info in iter(source):
            real_count = max(0, min(len(paths), total - batch_start))
            if real_count <= 0:
                raise RuntimeError('unexpected standalone CPU synthetic batch')
            specs = [prediction_result_frame_spec(source, batch_start + j, num_frames=total)
                     for j in range(len(paths))]
            local_specs = [None if spec is None else replace(spec, result_index=j,
                           task_index=spec.task_index - batch_start if not spec.is_azimuthal_padding else spec.task_index)
                           for j, spec in enumerate(specs)]
            for i, sibling in enumerate(pass_tasks):
                active, replay = original, None
                if i:
                    trajectory = f'{task["view"].name}/{task["kind"]}/{task["job_id"]}'
                    seeds = [policy_seed(settings, trajectory=trajectory, pass_index=i,
                                         slice_index=(int(spec.global_destination_index) if spec else
                                                      int(task.get('slice_start', 0)) + total - 1),
                                         lease_start=int(task.get('slice_start', 0))) for spec in specs]
                    active, replay = adapter.apply(original, seeds)
                    if writers[i] is not None:
                        for j, spec in enumerate(specs):
                            if spec is not None:
                                writers[i].put(batch_start + j, spec, seeds[j], replay[j])
                mask, conf, pad_mask, pad_conf = targets[i]
                local_source = CpuPolicyBatchSource(paths, active, info, local_specs, replay)
                result = runner.infer_source_to_union(local_source, **{
                    **predict_kwargs, 'num_frames': real_count,
                    'view_union_mm': mask[batch_start:batch_start + real_count],
                    'view_confmap_mm': conf[batch_start:batch_start + real_count] if conf is not None else None,
                    'azimuthal_padding_union_mm': pad_mask,
                    'azimuthal_padding_confmap_mm': pad_conf,
                })
                for key in ('prediction_count', 'frames_with_predictions', 'device_hole_filled_frames',
                            'azimuthal_padding_processed'):
                    stats[i][key] += int(result.get(key, 0))
                for key, value in result.items():
                    if key.startswith('openvino_'):
                        stats[i][key] = value
                metadata[i].merge(result.get('slice_meta'), batch_start, real_count)
            batch_start += len(paths)
            rendered_batches += 1
        if batch_start < total:
            raise RuntimeError(f'CPU augmentation source exhausted at {batch_start}/{total} frames')
        records = [dict(writer.finish(), backend='cpu', policy_sha256=settings.content_sha256)
                   for writer in writers if writer is not None]
        settings.assert_unchanged()
        for i, accumulator in enumerate(metadata):
            stats[i]['slice_meta'] = accumulator.finish()
        return {**stats[0], 'augmentation_results': stats[1:], 'augmentation_records': records,
                'augmentation_execution': {
                    'backend': 'cpu', 'policy_sha256': settings.content_sha256,
                    'rendered_batches': rendered_batches, 'model_batches': rendered_batches * settings.ratio,
                    'pass_count': settings.ratio, 'source_render_replays': 0,
                }}
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
        for mm in owned:
            mm._mmap.close()
