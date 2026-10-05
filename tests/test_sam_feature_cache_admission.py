"""A larger retention request still respects live CUDA memory and headroom."""
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import sam_tracker_runtime


GIB = 1024 ** 3


def _build(tmp_path, *, requested, free, total, reserved=0, allocated=0, headroom=None):
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fake checkpoint; no inference')
    bundle = SimpleNamespace(
        checkpoint_path=checkpoint, checkpoint_identity_sha256='checkpoint-identity',
        model_version='sam3.1', bpe_path=None,
    )
    cuda = SimpleNamespace(
        mem_get_info=mock.Mock(return_value=(free, total)),
        memory_reserved=mock.Mock(return_value=reserved),
        memory_allocated=mock.Mock(return_value=allocated),
        synchronize=mock.Mock(),
        empty_cache=mock.Mock(),
    )
    context = SimpleNamespace(
        predictor=SimpleNamespace(model=SimpleNamespace(tracker=SimpleNamespace(fill_hole_area=0)),
                                  shutdown=mock.Mock()),
        sam_runtime={}, profile={}, torch_module=SimpleNamespace(cuda=cuda),
        restore_sdpa=mock.Mock(),
    )
    config = dict(model_path='unused', feature_cache_bytes=requested)
    if headroom is not None:
        config['feature_cache_headroom_bytes'] = headroom
    with mock.patch('XTA.lta_worker_adapter.build_worker_predictor', return_value=context), \
            mock.patch('XTA.lta_sam.resolve_local_sam_bundle', return_value=bundle):
        result = sam_tracker_runtime.build_interpolation_predictor(config)
    cuda.synchronize.assert_called_once_with(0)
    assert result.sam_runtime['startup_cuda_quiescence'] == {
        'synchronized': True, 'worker_local_device': 0,
    }
    return result, cuda


def _close(context):
    predictor = context.predictor
    sam_tracker_runtime.close_interpolation_predictor(context)
    context.restore_sdpa.assert_called_once_with()
    predictor.shutdown.assert_called_once_with()
    context.torch_module.cuda.empty_cache.assert_called_once_with()
    assert context.feature_cache is None
    assert context.predictor is None


@pytest.mark.parametrize(('free', 'total', 'expected'), [
    (60 * GIB, 80 * GIB, GIB),
    (12 * GIB + GIB // 4, 80 * GIB, GIB // 4),
    (12 * GIB - 1, 80 * GIB, 0),
    (2 * GIB + GIB // 4, 8 * GIB, GIB // 4),
])
def test_one_gib_request_is_capped_by_mandatory_device_headroom(tmp_path, free, total, expected):
    context, _ = _build(tmp_path, requested=GIB, free=free, total=total)
    try:
        snapshot = context.feature_cache.snapshot()
        assert snapshot['max_bytes'] == expected
        assert snapshot['headroom_bytes'] == max(2 * GIB, int(total * .15))
        assert snapshot['live_bytes'] == 0
    finally:
        _close(context)


def test_only_unused_allocator_reserve_contributes_to_usable_memory(tmp_path):
    context, _ = _build(tmp_path, requested=GIB, free=12 * GIB,
                        total=80 * GIB, reserved=3 * GIB, allocated=11 * GIB // 4)
    try:
        assert context.feature_cache.snapshot()['max_bytes'] == GIB // 4
    finally:
        _close(context)


def test_explicit_headroom_can_raise_but_cannot_lower_the_reserve(tmp_path):
    for headroom, expected in ((0, GIB), (13 * GIB, 0)):
        context, _ = _build(tmp_path, requested=GIB, free=13 * GIB,
                            total=80 * GIB, headroom=headroom)
        try:
            assert context.feature_cache.snapshot()['max_bytes'] == expected
            assert context.feature_cache.snapshot()['headroom_bytes'] >= 12 * GIB
        finally:
            _close(context)


def test_explicit_zero_disables_retention_without_a_cuda_memory_probe(tmp_path):
    context, cuda = _build(tmp_path, requested=0, free=80 * GIB, total=80 * GIB)
    try:
        assert context.feature_cache is None
        cuda.mem_get_info.assert_not_called()
        cuda.memory_reserved.assert_not_called()
        cuda.memory_allocated.assert_not_called()
    finally:
        _close(context)
