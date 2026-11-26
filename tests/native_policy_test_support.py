"""Actual small file-backed scientific workspaces for native policy-group tests."""
import math
from pathlib import Path

import numpy as np

from XTA.interpolation import _DirectUnionBackingLease
from tests.native_detector_test_support import detector_view, detector_state, detector_inputs
from tests.native_scheduler_cases import detector_task


def policy_task(group, *, ratio=4, size=100, task_id=0, slice_start=0, mode='file', root=None):
    members = []
    output_root = Path('.') if root is None else Path(root)
    for index in range(ratio):
        view = detector_view(name=f'spherical_{group}__policy_{index}', family='spherical',
                             num_slices=4, src_h=5, src_w=5, augmentation_pass=index,
                             physical_view_name=f'spherical_{group}')
        members.append(detector_task(task_id, view=view, start=slice_start, mode=mode,
                                    job_id='a0',
                                    processing_shape=(1, 1, size), bounded_parent_admission=True,
                                    result_mask_path=str(output_root / f'chunk-{task_id}-{index}.mask'),
                                    result_conf_path=str(output_root / f'chunk-{task_id}-{index}.conf')))
    members[0]['augmentation_pass_tasks'] = members[1:]
    return members[0]


class PolicyWorkspaces:
    def __init__(self, root, *, confidence=False, capacity=4096, views=2):
        self.root = root
        self.state = detector_state()
        self.confidence = confidence
        self.arrays = {}
        self.file_results = set()
        self.consumed_file_bytes = 0
        self.requested = []
        self.inputs = detector_inputs(root, min_conf=0.1 if confidence else 0,
            direct_union_inference_byte_limit=capacity,
            direct_union_total_dense_byte_limit=capacity,
            direct_union_inference_view_limit=views,
            request_baseline_workspaces=self.request)

    def request(self, task):
        members = (task, *task.get('augmentation_pass_tasks', ()))
        self.requested.append(tuple(member['view'].name for member in members))
        for member in members:
            key = ('model', member['view'].name)
            if key in self.state.direct_union_backing_leases:
                continue
            shape = member['processing_shape']
            paths = []
            arrays = []
            for suffix in ('mask', 'conf') if self.confidence else ('mask',):
                path = self.root / f'{member["view"].name}.{suffix}'
                array = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
                array[:] = 0
                arrays.append(array)
                paths.append(path)
            self.arrays[key] = arrays
            self.state.baseline_union_paths[key] = paths[0]
            self.state.baseline_confmap_paths[key] = paths[1] if self.confidence else None
            size = math.prod(shape) * (2 if self.confidence else 1)
            self.state.direct_union_backing_leases[key] = _DirectUnionBackingLease(key, size)
            self.state.direct_union_inference_views.add(key)
            self.state.direct_union_inference_bytes[key] = size
        return True

    def write_file_results(self, task):
        """Actor output is a separate allocation from the admitted parent maps."""
        total = 0
        for index, member in enumerate((task, *task.get('augmentation_pass_tasks', ()))):
            if member['result_mode'] != 'file':
                raise ValueError('File result fixture requires an actual file-result task')
            shape = (member['slice_count'], *member['processing_shape'][-2:])
            for field, value in (('result_mask_path', 1), ('result_conf_path', 200 + index)):
                if not member.get(field):
                    continue
                path = Path(member[field])
                if path.resolve().parent != self.root.resolve():
                    raise ValueError('Test result path is outside its owned temporary directory')
                array = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
                array[:] = value
                array.flush()
                array._mmap.close()
                self.file_results.add(path)
                total += path.stat().st_size
        return total

    def consume_file_results(self, task):
        """Read/merge/close/delete actual actor files before the consumer returns."""
        consumed = 0
        for index, member in enumerate((task, *task.get('augmentation_pass_tasks', ()))):
            key = ('model', member['view'].name)
            shape = (member['slice_count'], *member['processing_shape'][-2:])
            start = member['slice_start']
            for plane, field, value in ((0, 'result_mask_path', 1), (1, 'result_conf_path', 200 + index)):
                if not member.get(field):
                    continue
                path = Path(member[field])
                if path not in self.file_results:
                    raise ValueError('Consumer did not receive a produced file')
                source = np.memmap(path, mode='r', dtype=np.uint8, shape=shape)
                try:
                    if not np.all(source == value):
                        raise AssertionError('Policy pass file contents changed during handoff')
                    if plane < len(self.arrays[key]):
                        target = self.arrays[key][plane][start:start + member['slice_count']]
                        np.maximum(target, source, out=target)
                    consumed += source.nbytes
                finally:
                    source._mmap.close()
                path.unlink()
                self.file_results.remove(path)
        self.consumed_file_bytes += consumed
        return consumed

    def retire(self, key):
        for array in self.arrays.pop(key):
            array._mmap.close()
        lease = self.state.direct_union_backing_leases.pop(key)
        lease.release(lease.phase)
        self.state.direct_union_inference_views.discard(key)
        self.state.direct_union_inference_bytes.pop(key, None)
        self.state.direct_union_postprocess_views.discard(key)
        self.state.direct_union_postprocess_bytes.pop(key, None)

    def close(self):
        for key in list(self.arrays):
            self.retire(key)
        for path in list(self.file_results):
            path.unlink(missing_ok=True)
            self.file_results.remove(path)
