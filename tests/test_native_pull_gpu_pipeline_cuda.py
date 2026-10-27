"""Actual CUDA lifetime and thread ownership of the bounded native pipeline."""
import threading

import numpy as np
import pytest

from XTA import packed_tilted_projection as packed, runtime
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter
from XTA.native_pull_cuda import NativePullCudaSession
from tests.test_packed_tilted_cuda_integration import (
    actual_cuda, _views, _oracle, _decode, _hashes, pytestmark,
)


@pytest.mark.parametrize('writer_failure', [False, True], ids=['complete', 'writer-failure'])
def test_actual_gpu_producer_joins_before_source_and_device_retirement(
        tmp_path, monkeypatch, actual_cuda, writer_failure, request):
    view = _views()[1]
    source = np.random.default_rng(160624).integers(
        0, 2, (view.num_slices, view.src_h, view.src_w), dtype=np.uint8)
    target = (27, 97, 111)
    expected = _oracle(source, view, target)
    source_path, output_path = tmp_path / 'native.cvol', tmp_path / 'output.cvol'
    packed.pack_terminal_native_source(source, source_path, desc='Actual GPU pipeline ownership')
    request.addfinalizer(lambda: runtime.release_memfd_owners_under(source_path))
    before = _hashes(source_path)
    caller = threading.get_ident()
    producer_threads, writer_threads = [], []
    original_pull = NativePullCudaSession.pull_flat_into
    original_block = NativePullCudaSession.project_encoded
    original_encoded = IncrementalRawBBoxMaskStoreWriter.consume_encoded_block

    def pull(session, *args, **kwargs):
        producer_threads.append(threading.get_ident())
        return original_pull(session, *args, **kwargs)

    def block(session, *args, **kwargs):
        producer_threads.append(threading.get_ident())
        result = original_block(session, *args, **kwargs)
        assert result.stats['dense_d2h_bytes'] == 0
        return result

    def consume(original, writer, *args, **kwargs):
        writer_threads.append(threading.get_ident())
        if writer_failure:
            raise OSError('injected actual CUDA pipeline writer failure')
        return original(writer, *args, **kwargs)

    monkeypatch.setattr(NativePullCudaSession, 'pull_flat_into', pull)
    monkeypatch.setattr(NativePullCudaSession, 'project_encoded', block)
    def dense_consume(writer, *args, **kwargs):
        raise AssertionError('Packed GPU publication used the dense CPU writer')
    monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter, 'consume',
        dense_consume)
    monkeypatch.setattr(IncrementalRawBBoxMaskStoreWriter, 'consume_encoded_block',
        lambda writer, *args, **kwargs: consume(original_encoded, writer, *args, **kwargs))
    if writer_failure:
        with pytest.raises(OSError, match='injected actual CUDA pipeline writer failure'):
            packed.project_tilted_sparse_store(source_path, view, output_path,
                out_shape_tyx=target, workers=3)
        assert not output_path.exists()
    else:
        result = packed.project_tilted_sparse_store(source_path, view, output_path,
            out_shape_tyx=target, workers=3)
        np.testing.assert_array_equal(_decode(output_path), expected)
        assert result['gpu_planes'] == target[0]
        assert result['timings_seconds']['cpu_warmup_calls'] == 0
    assert producer_threads and writer_threads
    assert all(thread != caller for thread in producer_threads)
    assert set(writer_threads) == {caller}
    assert _hashes(source_path) == before
    # The actual_cuda fixture also checks empty private pools and returned leases.
    assert all(session._closed for session in actual_cuda)
    assert all(session._staging.pool.n_free_blocks() == 0 for session in actual_cuda)
    assert not any(thread.name.startswith('native-pull-gpu') for thread in threading.enumerate())
