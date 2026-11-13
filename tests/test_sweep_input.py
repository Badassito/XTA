"""Once-per-sweep native gray8 decoding and failed producer handling."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA.sweep_input import open_or_decode_sweep_input, open_or_resize_sweep_cube


@pytest.mark.parametrize('streaming', [False, True])
def test_successful_source_is_reopened_without_another_decode(tmp_path, streaming):
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'video A')
    shape = (4, 6, 8)
    expected = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    callbacks = []

    def decode_once(**options):
        assert options['overwrite']
        assert not (options['prefer_memory'] or options['prefer_memfd'])
        mapping = np.memmap(options['out_dat'], mode='w+', dtype=np.uint8, shape=shape)
        mapping[...] = expected
        if streaming:
            callbacks.append(options['on_verified'])
        return mapping

    decoder = Mock(side_effect=decode_once)
    first = open_or_decode_sweep_input(tmp_path, source, shape, streaming=streaming, decoder=decoder)
    if streaming:
        assert not (tmp_path / 'source.gray8.json').exists()
        callbacks.pop()()
    second = open_or_decode_sweep_input(tmp_path, source, shape, streaming=streaming, decoder=decoder)
    np.testing.assert_array_equal(first, second)
    assert not second.flags.writeable
    assert decoder.call_count == 1
    source.write_bytes(b'video B with a different size')
    with pytest.raises(RuntimeError, match='changed between'):
        open_or_decode_sweep_input(tmp_path, source, shape, streaming=streaming, decoder=decoder)


def test_failed_decode_is_never_reused(tmp_path):
    source = tmp_path / 'source'
    source.touch()
    (tmp_path / 'source.gray8').write_bytes(b'partial!')
    decoder = Mock(side_effect=RuntimeError('invalid stream'))
    with pytest.raises(RuntimeError, match='invalid stream'):
        open_or_decode_sweep_input(tmp_path, source, (2,2,2), streaming=False, decoder=decoder)
    assert decoder.call_args.kwargs['overwrite']
    assert not (tmp_path / 'source.gray8.json').exists()


def test_low_space_falls_back_to_normal_scalar_decode(tmp_path, monkeypatch):
    source = tmp_path / 'video'
    source.touch()
    monkeypatch.setattr('XTA.sweep_input.shutil.disk_usage', lambda _: SimpleNamespace(free=0))
    decoder = Mock()
    result = open_or_decode_sweep_input(tmp_path, source, (2,2,2), streaming=True, decoder=decoder)
    assert result is None
    decoder.assert_not_called()


@pytest.mark.parametrize('streaming', [False, True])
def test_resized_input_reuses_exact_scalar_cube(tmp_path, streaming):
    from XTA import media

    source_path = tmp_path / 'video'
    source_path.touch()
    source = np.arange(5 * 9 * 10, dtype=np.uint8).reshape(5, 9, 10)
    target = (8, 9, 10)
    resize = (media.resize_volume_to_processing_cube_gray8_streaming if streaming else
              media.resize_volume_to_processing_cube_gray8)
    reference = resize(source, target, tmp_path / 'scalar_cube', workers=1, prefer_memory=False)
    build = Mock(wraps=resize)
    first = open_or_resize_sweep_cube(tmp_path, source_path, source, target,
                                      streaming=streaming, resize=build, workers=1)
    assert media.wait_for_streaming_producers(timeout=10)
    media.wait_for_volume_ready(reference)
    media.wait_for_volume_ready(first)
    second = open_or_resize_sweep_cube(tmp_path, source_path, source, target,
                                       streaming=streaming, resize=build, workers=1)
    np.testing.assert_array_equal(first, reference)
    np.testing.assert_array_equal(second, reference)
    assert build.call_count == 1


def test_concurrent_cutoff_never_builds_a_shared_mapping(tmp_path, monkeypatch):
    """Siblings running at once may read the producer's file but never (re)build it."""
    source = tmp_path / 'video'
    source.touch()
    monkeypatch.setenv('XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT', '1')
    build = Mock()
    assert open_or_resize_sweep_cube(tmp_path, source, None, (2, 2, 2), streaming=False,
                                     resize=build, workers=1) is None
    assert open_or_decode_sweep_input(tmp_path, source, (2, 2, 2), streaming=False, decoder=build) is None
    build.assert_not_called()
    assert not (tmp_path / 'cube.gray8').exists() and not (tmp_path / 'source.gray8').exists()
