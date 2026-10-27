"""Explicit-seed checks for synchronization-free standalone GPU preset setup."""
import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from tests.test_tta_external_augmentation import cpu_policy


@pytest.fixture(params=['light', 'baseline', 'heavy', 'superheavy'])
def profile(request):
    return request.param


def legacy_inverse(matrices, device):
    return torch.linalg.inv(torch.tensor(matrices, device=device, dtype=torch.float32))


def legacy_grid(expression, *operands, **kwargs):
    if expression == 'nij,hwj->nhwi':
        inverse, pixels = operands
        return torch.stack([_EINSUM('ij,hwj->hwi', matrix, pixels) for matrix in inverse])
    return _EINSUM(expression, *operands, **kwargs)


_EINSUM = torch.einsum
_GRID_SAMPLE = torch.nn.functional.grid_sample


def materialized_grid_sample(source, grid, *args, **kwargs):
    return _GRID_SAMPLE(source.contiguous(), grid, *args, **kwargs)


def sample_inputs(batch_size, shape=(32, 48), channels=1):
    rng = np.random.default_rng(20261020)
    count = batch_size // 5
    images = [rng.integers(0, 256, (*shape, channels), np.uint8) for _ in range(count)]
    if channels == 1:
        images = [image[..., 0] for image in images]
    masks = []
    for index in range(count):
        mask = np.zeros(shape, np.uint8)
        mask[shape[0] // 4:shape[0] * 3 // 4, shape[1] // 4:shape[1] * 3 // 4] = 1
        mask[0, index] = mask[-1, -1 - index] = 1
        masks.append(mask)
    seeds = [[None, *range(11 + 20 * index, 15 + 20 * index)] for index in range(count)]
    return dict(images=images, masks=masks, seeds=seeds, output_size=shape)


@pytest.mark.parametrize('batch_size', [5, 10, 15])
@pytest.mark.parametrize('channels', [1, 3])
def test_explicit_seed_outputs_and_original_order_match_legacy(profile, batch_size, channels):
    policy, module = cpu_policy(profile, batch_size=batch_size)
    kwargs = sample_inputs(batch_size, channels=channels)
    with (mock.patch.object(torch.Tensor, 'pin_memory', lambda self, *a, **kw: self),
          mock.patch.object(module, '_inverse_affine_batch', side_effect=legacy_inverse),
          mock.patch.object(torch, 'einsum', side_effect=legacy_grid),
          mock.patch.object(module.F, 'grid_sample', side_effect=materialized_grid_sample)):
        expected_images, expected_masks = policy.apply_batch_many(**kwargs)
    with mock.patch.object(torch.Tensor, 'pin_memory', lambda self, *a, **kw: self):
        images, masks = policy.apply_batch_many(**kwargs)
    assert torch.equal(masks, expected_masks)
    assert torch.equal(images, expected_images)
    for source_index, source in enumerate(kwargs['images']):
        expected = source[None] if source.ndim == 2 else source.transpose(2, 0, 1)
        np.testing.assert_array_equal(images[source_index * 5].numpy(), expected)
        np.testing.assert_array_equal(masks[source_index * 5].numpy(), kwargs['masks'][source_index])


@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('originals_only', [False, True])
def test_single_source_expansion_preserves_inputs_and_independent_outputs(profile, channels, originals_only):
    policy, module = cpu_policy(profile, batch_size=5)
    kwargs = sample_inputs(5, shape=(16, 24), channels=channels)
    if originals_only:
        kwargs['seeds'] = [[None] * 5]
    before_image, before_mask = kwargs['images'][0].copy(), kwargs['masks'][0].copy()
    sources = []

    def record(source, grid, **options):
        sources.append(source)
        assert source.stride(0) == 0
        assert source.untyped_storage().nbytes() == source[0].numel() * source.element_size()
        return _GRID_SAMPLE(source, grid, **options)

    with (mock.patch.object(torch.Tensor, 'pin_memory', lambda self, *a, **kw: self),
          mock.patch.object(module.F, 'grid_sample', side_effect=record)):
        images, masks = policy.apply_batch_many(**kwargs)
    assert len(sources) == (0 if originals_only else 2)
    expected_image, expected_mask = images[1:].clone(), masks[1:].clone()
    with torch.inference_mode():
        images[0].zero_()
        masks[0].zero_()
    assert torch.equal(images[1:], expected_image)
    assert torch.equal(masks[1:], expected_mask)
    np.testing.assert_array_equal(kwargs['images'][0], before_image)
    np.testing.assert_array_equal(kwargs['masks'][0], before_mask)


def test_safe_cuda_inverse_uses_actual_float32_coefficients_without_checked_readback(profile):
    _policy, module = cpu_policy(profile)
    matrices = [[[1.00000003, .2, 15.], [-.3, -.9, -3.], [0., 0., 1.]]]
    expected = legacy_inverse(matrices, torch.device('cpu'))
    captured = []

    def inverse(batch, *, check_errors):
        captured.append(batch)
        assert check_errors is False
        return SimpleNamespace(inverse=expected)

    with (mock.patch.object(torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock, return_value=True),
          mock.patch.object(torch.linalg, 'inv_ex', side_effect=inverse),
          mock.patch.object(torch.linalg, 'inv', side_effect=AssertionError('synchronized checked inverse'))):
        actual = module._inverse_affine_batch(matrices, torch.device('cpu'))
    assert actual is expected
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0].numpy(), np.asarray(matrices, np.float32))


@pytest.mark.parametrize('matrix', [
    [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 1., 0.], [1., 1.0000001, 0.], [0., 0., 1.]],
    [[1., 0., 0.], [0., 1., 0.], [.01, 0., 1.]],
    [[float('nan'), 0., 0.], [0., 1., 0.], [0., 0., 1.]],
])
def test_exceptional_affines_retain_checked_inverse_and_singular_error(profile, matrix):
    _policy, module = cpu_policy(profile)
    inverse = torch.linalg.inv
    with (mock.patch.object(torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock, return_value=True),
          mock.patch.object(torch.linalg, 'inv', wraps=inverse) as checked,
          mock.patch.object(torch.linalg, 'inv_ex', side_effect=AssertionError('unchecked exceptional matrix'))):
        if matrix[0][0] == 0:
            with pytest.raises(RuntimeError, match='singular'):
                module._inverse_affine_batch([matrix], torch.device('cpu'))
        else:
            actual = module._inverse_affine_batch([matrix], torch.device('cpu'))
            torch.testing.assert_close(actual, inverse(torch.tensor([matrix], dtype=torch.float32)),
                                       rtol=0, atol=0, equal_nan=True)
    assert checked.call_count >= 1


@pytest.mark.skipif(os.environ.get('YOLO_TTA_RUN_GPU_TESTS') != '1', reason='explicit CUDA qualification')
@pytest.mark.parametrize('stage', ['inverse', 'grid', 'combined'])
@pytest.mark.parametrize('batch_size', [5, 10, 15])
@pytest.mark.parametrize('shape', [(32, 48), (2048, 2048)])
def test_cuda_explicit_seed_geometry_matches_legacy(profile, monkeypatch, stage, batch_size, shape):
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '0')
    _cpu, module = cpu_policy(profile, batch_size=batch_size)
    policy = module.GPUAugmentation(device='cuda:0', batch_size=batch_size)
    kwargs = sample_inputs(batch_size, shape=shape)
    kwargs['images'] = [torch.as_tensor(image, device='cuda:0') for image in kwargs['images']]
    kwargs['masks'] = [torch.as_tensor(mask, device='cuda:0') for mask in kwargs['masks']]
    with (mock.patch.object(module, '_inverse_affine_batch', side_effect=legacy_inverse),
          mock.patch.object(torch, 'einsum', side_effect=legacy_grid),
          mock.patch.object(module.F, 'grid_sample', side_effect=materialized_grid_sample)):
        expected_images, expected_masks = policy.apply_batch_many(**kwargs)
    with ExitStack() as stack:
        if stage == 'grid':
            stack.enter_context(mock.patch.object(module, '_inverse_affine_batch', side_effect=legacy_inverse))
        elif stage == 'inverse':
            stack.enter_context(mock.patch.object(torch, 'einsum', side_effect=legacy_grid))
        images, masks = policy.apply_batch_many(**kwargs)
    assert torch.equal(masks, expected_masks), f'categorical mask changed ({stage}, {shape}, {batch_size})'
    difference = (images.to(torch.int16) - expected_images.to(torch.int16)).abs()
    assert int(difference.max().item()) <= 1, f'intensity change exceeded one gray level ({stage})'


@pytest.mark.skipif(os.environ.get('YOLO_TTA_RUN_GPU_TESTS') != '1', reason='explicit CUDA qualification')
@pytest.mark.parametrize('shape', [(32, 48), (2048, 2048)])
@pytest.mark.parametrize('channels', [1, 3])
def test_cuda_single_source_expansion_preserves_inputs_and_outputs(profile, monkeypatch, shape, channels):
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '0')
    _cpu, module = cpu_policy(profile, batch_size=5)
    policy = module.GPUAugmentation(device='cuda:0', batch_size=5)
    kwargs = sample_inputs(5, shape=shape, channels=channels)
    kwargs['images'] = [torch.as_tensor(value, device='cuda:0') for value in kwargs['images']]
    kwargs['masks'] = [torch.as_tensor(value, device='cuda:0') for value in kwargs['masks']]
    before_image, before_mask = kwargs['images'][0].clone(), kwargs['masks'][0].clone()
    with mock.patch.object(module.F, 'grid_sample', side_effect=materialized_grid_sample):
        expected_images, expected_masks = policy.apply_batch_many(**kwargs)
    images, masks = policy.apply_batch_many(**kwargs)
    assert torch.equal(images, expected_images)
    assert torch.equal(masks, expected_masks)
    assert torch.equal(kwargs['images'][0], before_image)
    assert torch.equal(kwargs['masks'][0], before_mask)
    other_images, other_masks = images[1:].clone(), masks[1:].clone()
    with torch.inference_mode():
        images[0].zero_()
        masks[0].zero_()
    assert torch.equal(images[1:], other_images)
    assert torch.equal(masks[1:], other_masks)


def test_compile_fallback_warns_once_without_exposing_exception_paths(profile):
    policy, module = cpu_policy(profile, batch_size=5)
    failed = mock.Mock(side_effect=RuntimeError('private/path/that/must/not/leak'))
    policy._pointwise_kernel, policy._pointwise_compiled = failed, True
    kwargs = sample_inputs(5, shape=(16, 24))
    with mock.patch.object(torch.Tensor, 'pin_memory', lambda self, *a, **kw: self):
        with pytest.warns(RuntimeWarning, match='compilation failed.*RuntimeError.*eager Torch') as captured:
            first_images, first_masks = policy.apply_batch_many(**kwargs)
        assert 'private/path' not in str(captured[0].message)
        with __import__('warnings').catch_warnings(record=True) as repeated:
            second_images, second_masks = policy.apply_batch_many(**kwargs)
        assert not repeated
    failed.assert_called_once()
    assert policy._pointwise_compiled is False
    assert torch.equal(first_images, second_images) and torch.equal(first_masks, second_masks)


def test_compile_setup_preserves_fusion_without_cuda_graph_capture(profile, monkeypatch):
    _policy, module = cpu_policy(profile)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '1')
    initialized = torch.cuda.is_initialized()
    with mock.patch.object(torch, 'compile', return_value=module._fused_pointwise) as compiler:
        policy = module.GPUAugmentation(device='cuda:0')
    compiler.assert_called_once_with(
        module._fused_pointwise, fullgraph=True, dynamic=False,
        options={'triton.cudagraphs': False},
    )
    assert policy._pointwise_compiled
    assert torch.cuda.is_initialized() == initialized


def test_installed_compiler_disables_capture_even_when_ambient_config_enables_it(profile, monkeypatch):
    _policy, module = cpu_policy(profile)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '1')
    if not callable(getattr(torch, 'compile', None)):
        pytest.skip('installed Torch has no optional compiler')
    from torch._inductor import config
    initialized = torch.cuda.is_initialized()
    compiler_type = torch._TorchCompileInductorWrapper
    compilers = []

    def observe_compiler(*args, **kwargs):
        compiler = compiler_type(*args, **kwargs)
        compilers.append(compiler)
        return compiler

    with config.patch({'triton.cudagraphs': True}):
        with mock.patch.object(torch, '_TorchCompileInductorWrapper', side_effect=observe_compiler):
            policy = module.GPUAugmentation(device='cuda:0')
        assert policy._pointwise_compiled
        inspect_config = getattr(policy._pointwise_kernel, 'get_compiler_config', None)
        compiled_config = inspect_config() if callable(inspect_config) else None
        # Some fullgraph Torch versions omit public config metadata; inspect the actual backend instead.
        if compiled_config is None:
            assert len(compilers) == 1
            inspect_backend = getattr(compilers[0], 'get_compiler_config', None)
            compiled_config = inspect_backend() if callable(inspect_backend) else compilers[0].config
        assert compiled_config['triton.cudagraphs'] is False
        assert config.triton.cudagraphs is True
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize('disabled', ['0', 'false', 'no'])
def test_explicit_compile_disable_retains_eager_kernel(profile, monkeypatch, disabled):
    _policy, module = cpu_policy(profile)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', disabled)
    with mock.patch.object(torch, 'compile', side_effect=AssertionError('compilation disabled')):
        policy = module.GPUAugmentation(device='cuda:0')
    assert policy._pointwise_kernel is module._fused_pointwise
    assert not policy._pointwise_compiled


def test_missing_optional_compiler_retains_eager_kernel(profile, monkeypatch):
    _policy, module = cpu_policy(profile)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '1')
    with mock.patch.object(torch, 'compile', None):
        policy = module.GPUAugmentation(device='cuda:0')
    assert policy._pointwise_kernel is module._fused_pointwise
    assert not policy._pointwise_compiled


@pytest.mark.parametrize('failure', [RuntimeError, TypeError])
def test_compile_setup_failure_warns_without_initializing_cuda(profile, monkeypatch, failure):
    _policy, module = cpu_policy(profile)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '1')
    initialized = torch.cuda.is_initialized()
    with mock.patch.object(torch, 'compile', side_effect=failure('private/compiler/path')):
        with pytest.warns(RuntimeWarning, match=f'compilation failed.*{failure.__name__}.*eager Torch') as captured:
            policy = module.GPUAugmentation(device='cuda:0')
    assert 'private/compiler' not in str(captured[0].message)
    assert not policy._pointwise_compiled
    assert torch.cuda.is_initialized() == initialized
