"""Immutable policy identities and owner-scoped azimuthal CUDA lookup reuse."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, pta_cuda_azimuthal as azimuthal
from XTA.unification import contracts
from XTA.unification.sampling import forward_sampling_policy


def test_immutable_policy_digest_is_computed_once_and_keeps_canonical_bytes():
    policy = replace(forward_sampling_policy())
    expected = contracts._digest_record(policy.canonical_record())
    with mock.patch.object(contracts, '_digest_record', wraps=contracts._digest_record) as digest:
        assert {policy.digest for _ in range(100)} == {expected}
        digest.assert_called_once()
    with pytest.raises(FrozenInstanceError):
        policy.digest = 'changed'


def test_policy_replace_and_old_pickle_without_cached_identity_are_independent():
    policy = replace(forward_sampling_policy())
    first = policy.digest
    changed = replace(policy, policy_version=policy.policy_version + 1)
    assert changed.digest != first
    assert pickle.loads(pickle.dumps(policy)).digest == first
    policy.__dict__.pop('digest')  # Previous pickle schema had no memoized property.
    restored = pickle.loads(pickle.dumps(policy))
    assert 'digest' not in restored.__dict__
    assert restored.digest == first
    assert set(restored.canonical_record()) == set(changed.canonical_record())


class Tensor:
    def __init__(self, size):
        self.data = np.arange(size, dtype=np.float32)
        self.streams = []
    def numel(self): return self.data.size
    def element_size(self): return self.data.itemsize
    def record_stream(self, stream): self.streams.append(stream)


class Stream:
    def __init__(self): self.waits = []
    def wait_event(self, event): self.waits.append(event)


class Event:
    def record(self, stream): self.stream = stream


def test_table_cache_reuses_uploads_and_orders_cross_stream_reads():
    cache = azimuthal.AzimuthalTableCache()
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
    a, b = Stream(), Stream()
    build = mock.Mock(return_value=Tensor(4))
    first = cache.table(('rows', 'cuda:0'), build, torch=torch, stream=a)
    assert cache.table(('rows', 'cuda:0'), build, torch=torch, stream=b) is first
    build.assert_called_once()
    assert len(b.waits) == 1 and b.waits[0].stream is a
    assert first.streams == [a, b]
    assert cache.snapshot() == dict(hits=1, misses=1, uploads=1, evictions=0, entries=1, cached_bytes=16)


def test_table_cache_is_byte_and_entry_bounded_without_invalidating_returned_owners():
    cache = azimuthal.AzimuthalTableCache(max_bytes=20, max_entries=2)
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
    stream = Stream()
    first = cache.table('a', lambda: Tensor(4), torch=torch, stream=stream)
    cache.table('b', lambda: Tensor(4), torch=torch, stream=stream)
    assert cache.bytes == 16 and len(cache.tables) == 1 and cache.evictions == 1
    np.testing.assert_array_equal(first.data, np.arange(4))
    oversized = cache.table('large', lambda: Tensor(9), torch=torch, stream=stream)
    assert oversized.numel() == 9 and 'large' not in cache.tables
    cache.clear()
    assert cache.bytes == 0 and not cache.tables


def view_fixture():
    return next(v for v in geometry.get_view_infos(9, 11, 13, cartesian_views=(),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(37.,), azimuthal_native_raster=17)
        if geometry.is_azimuthal_view(v))


def test_cache_keys_cover_table_geometry_and_device_but_not_output_tile_affine():
    view = view_fixture()
    key = azimuthal._nearest_table_key(view, 0, 'cuda:0')
    for field, value in [('center_x', view.center_x + .25), ('center_y', view.center_y + .25),
                         ('roi_radius', view.roi_radius + .25), ('src_w', view.src_w + 1),
                         ('diameter', view.diameter + 1), ('full_h', view.full_h + 1)]:
        assert azimuthal._nearest_table_key(replace(view, **{field: value}), 0, 'cuda:0') != key
    assert azimuthal._nearest_table_key(view, 1, 'cuda:0') != key
    assert azimuthal._nearest_table_key(view, 0, 'cuda:1') != key
    row = azimuthal._row_table_key(view, False, 'cuda:0')
    assert azimuthal._row_table_key(view, True, 'cuda:0') != row
    assert azimuthal._row_table_key(replace(view, src_h=view.src_h + 1), False, 'cuda:0') != row
    assert azimuthal._row_table_key(replace(view, full_t=view.full_t + 1), False, 'cuda:0') != row


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_CACHE_GPU') != '1',
                    reason='requires explicit GPU qualification and an owned GPU lock')
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_real_cuda_cached_tables_match_uncached_across_tiles_and_streams(base):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    from XTA.config import TiltedViewGroup
    assert torch.cuda.is_available()
    source = torch.as_tensor((np.random.default_rng(49).random((9, 11, 13)) < .3).astype(np.uint8), device='cuda')
    views = geometry.get_view_infos(9, 11, 13, cartesian_views=(),
        azimuthal_views=(base, f'tilted_{base}'), azimuthal_azimuth_angles=(37., 37.),
        tilt_groups=(TiltedViewGroup(views=(base,), tilt_angles=(30.,),
                                     tilt_directions=('vertical', 'horizontal')),), azimuthal_native_raster=17)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cache = azimuthal.AzimuthalTableCache(max_bytes=512, max_entries=4)
    evicting = azimuthal.AzimuthalTableCache(max_bytes=512, max_entries=1)
    comparisons = 0
    for view in [v for v in views if geometry.is_azimuthal_view(v)]:
        index = min(1, len(view.azimuths_deg) - 1)
        for tile in range(3):
            matrix = np.asarray(((.75, 0, tile - 1.25), (0, .83, -.3)), np.float32)
            expected = azimuthal.render_azimuthal_categorical_pair(source, None, view, index,
                M_grid_to_src=matrix, out_h=15, out_w=19, stream=streams[0])[0]
            actual = azimuthal.render_azimuthal_categorical_pair(source, None, view, index,
                M_grid_to_src=matrix, out_h=15, out_w=19, stream=streams[tile % 2], table_cache=cache)[0]
            evicted = azimuthal.render_azimuthal_categorical_pair(source, None, view, index,
                M_grid_to_src=matrix, out_h=15, out_w=19, stream=streams[tile % 2], table_cache=evicting)[0]
            streams[0].synchronize(); streams[1].synchronize()
            np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())
            np.testing.assert_array_equal(evicted.cpu().numpy(), expected.cpu().numpy())
            comparisons += 1
    assert comparisons >= 9 and cache.hits > cache.misses
    assert evicting.evictions > 0
