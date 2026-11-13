import os

import pytest

from XTA import sweep_proto


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_candidate_bins_match_float32_native_gate_boundary(dtype):
    torch = pytest.importorskip("torch")
    head = torch.zeros(37, 10, dtype=getattr(torch, dtype))
    head[0] = torch.arange(10)
    head[4] = torch.tensor([.9, .5, .5, .6, 0, 1, .7001, .499, float("nan"), .59])
    cuts = (50, 60, 70, 80, 90, 100)
    packed, offsets = sweep_proto._candidate_bins(head, cuts)
    for j, cutoff in enumerate(cuts):
        expected = head[0, head[4].float() >= cutoff / 100].tolist()
        assert sorted(packed[0, offsets[j]:].tolist()) == sorted(expected)
    assert offsets[-1] == packed.shape[1]


def test_entirely_empty_frame_has_no_packed_head():
    torch = pytest.importorskip("torch")
    head, offsets = sweep_proto._candidate_bins(torch.zeros(37, 21), (69, 70, 100))
    assert head is None and offsets == [0, 0, 0, 0]


@pytest.mark.skipif(os.getenv("XTA_TEST_NATIVE_FANOUT_CUDA") != "1", reason="explicit CUDA qualification")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cuda_disjoint_maxima_match_scalar_tiled_compaction(dtype, monkeypatch):
    from XTA import inference
    torch = pytest.importorskip("torch")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(inference, "gpu_flatten_conf_tracking_enabled", lambda: True)
    monkeypatch.setattr(inference, "angle_variant_gpu_fastpath", lambda: None)
    generator = torch.Generator().manual_seed(168338)
    proto = torch.randn((32, 12, 16), generator=generator).to("cuda", getattr(torch, dtype))
    head = torch.randn((37, 16), generator=generator)
    head[:2] = head[:2] * 15 + 28
    head[2:4] = head[2:4].abs() * 20
    head[4] = torch.tensor([0, .3, .5, .5, .502, .55, .67, .7, .71, .705, .79, .8, .81, .82, .83, .9])
    head = head.to("cuda", getattr(torch, dtype))
    image = torch.empty((1, 1, 48, 64), device="cuda", dtype=proto.dtype)

    class Check:
        cutoffs = (50, 51, 60, 69, 70, 80, 81, 82, 90, 100)

        def record_detector_work(self, **_counts):
            pass

        def record_proto_work(self, once, scalar, occupied):
            assert 0 < once < scalar and 1 < occupied < len(self.cutoffs)

        def consume_result(self, index, cent, product):
            reference = inference._build_direct_device_compacted_payload(head, proto, image, cent / 100)
            assert reference is not None
            assert index == 0 and product.instance_count == int(reference.instance_count_device.item())
            if product.union_gpu is None:
                assert not bool(reference.union_gpu.any()) and not bool(reference.conf_gpu.any())
            else:
                assert torch.equal(product.union_gpu, reference.union_gpu)
                assert torch.equal(product.conf_gpu, reference.conf_gpu)
            seen.append(cent)

    seen = []
    assert sweep_proto.consume_tiled_siblings(Check(), 0, head, proto, image, allow_tiled=True)
    assert seen == list(reversed(Check.cutoffs))
