"""Closing a retained render proxy releases source pins before Windows cleanup."""
import gc
import weakref

import numpy as np

from XTA.media import LazyProcessingCube
from XTA.runtime import close_memmap_array, wait_for_retired_memmap_unlinks


def test_completed_lazy_cube_drops_source_pin_without_invalidating_live_views(tmp_path):
    source_path = tmp_path / 'decoded.dat'
    source = np.memmap(source_path, dtype=np.uint8, mode='w+', shape=(3, 5, 5))
    source[:] = 17
    consumer = np.asarray(source[1])
    mapping = weakref.ref(source._mmap)
    proxy = LazyProcessingCube(source, source.shape, tmp_path / 'cube.dat', workers=1,
        request_path=tmp_path / 'request', ready_path=tmp_path / 'ready',
        failed_path=tmp_path / 'failed')
    proxy._started = True
    proxy._ready.set()
    close_memmap_array(source, unlink_path=source_path)
    proxy.close()
    assert proxy.source is None
    del source
    gc.collect()
    assert mapping() is not None
    assert int(consumer.max()) == 17
    del consumer
    gc.collect()
    wait_for_retired_memmap_unlinks(path=source_path)
    assert mapping() is None
    assert not source_path.exists()
    # Keep proxy live: its diagnostic/config references no longer pin the file.
    assert proxy.shape == (3, 5, 5)


def test_unstarted_lazy_cube_close_drops_source_pin(tmp_path):
    source = np.zeros((3, 5, 5), dtype=np.uint8)
    proxy = LazyProcessingCube(source, source.shape, tmp_path / 'cube.dat', workers=1,
        request_path=tmp_path / 'request', ready_path=tmp_path / 'ready',
        failed_path=tmp_path / 'failed')
    proxy.close()
    assert proxy.source is None

