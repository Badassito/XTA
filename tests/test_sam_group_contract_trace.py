"""One balanced contract phase exposes large-family consumer work."""
from types import SimpleNamespace

import pytest

from XTA import sam_interpolation as core
import numpy as np


def test_group_contract_trace_records_geometry_and_balanced_failure(monkeypatch):
    events = []
    monkeypatch.setattr('XTA.runtime.runtime_trace_event', lambda event, **fields: events.append((event, fields)))
    group = SimpleNamespace(group_id='g', context_bbox_yx=(2, 3, 12, 23),
        frame_indices=tuple(range(5)), endpoint_ids=('a', 'b'), observation_ids=('a', 'b', 'c'), edges=('e',))
    with pytest.raises(RuntimeError, match='controlled contract failure'):
        with core._trace_sam_phase('group_contract', 'scope', operation='interpolation', group=group):
            raise RuntimeError('controlled contract failure')
    assert [event for event, _fields in events] == ['sam_phase_begin', 'sam_phase_end']
    assert events[0][1]['sam_phase_id'] == events[1][1]['sam_phase_id']
    assert events[0][1]['crop_pixels'] == 200 and events[0][1]['frame_count'] == 5
    assert events[0][1]['endpoint_count'] == 2 and events[0][1]['edge_count'] == 1
    assert events[1][1]['failed']
    assert events[0][1]['thread_native_id'] == events[1][1]['thread_native_id']
    assert events[1][1]['phase_host_seconds'] >= 0
    assert events[1][1]['phase_thread_cpu_seconds'] >= 0


def test_phase_trace_separates_owner_cpu_from_elapsed_time(monkeypatch):
    events = []
    wall, cpu = iter((10., 18.)), iter((2., 2.75))
    monkeypatch.setattr('XTA.runtime.runtime_trace_event', lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(core.time, 'monotonic', lambda: next(wall))
    monkeypatch.setattr(core.time, 'thread_time', lambda: next(cpu))
    monkeypatch.setattr(core.threading, 'get_native_id', lambda: 1234)
    with core._trace_sam_phase('group_contract', 'scope'):
        pass
    begin, end = events
    assert begin[1]['thread_native_id'] == end[1]['thread_native_id'] == 1234
    assert 'phase_host_seconds' not in begin[1]
    assert end[1]['phase_host_seconds'] == 8.
    assert end[1]['phase_thread_cpu_seconds'] == .75
    assert not end[1]['failed']


def test_publication_reuses_existing_group_and_run_maps_once(tmp_path, monkeypatch):
    from tests.test_sam_interpolation import _generate, _observations, RepeatedSeedTracker
    calls = []
    original = core._records_by_id
    def records(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(core, '_records_by_id', records)
    merged, stats, components = _generate(tmp_path, RepeatedSeedTracker())
    try:
        assert calls == ['group_id', 'run_id']
        assert stats['added_voxels'] == 108 and len(components) == 2
        expected = np.repeat(_observations()[0:1], 5, axis=0)
        np.testing.assert_array_equal(merged, expected)
    finally:
        if isinstance(merged, np.memmap):
            merged._mmap.close()
