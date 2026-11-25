"""Independent CPU retina geometry and the device fallback's score plane."""
import os
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip('torch')
import torch.nn.functional as F  # noqa: E402

from XTA import inference  # noqa: E402
from XTA.reconciliation import EvidenceLayer, reconcile  # noqa: E402

IMG = 64
PROTO = 16  # stride 4


@pytest.fixture(autouse=True)
def generic_proto_path():
    with mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=True), \
            mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None):
        yield


def _inputs(boxes, confs, fields):
    fields = np.asarray(fields, np.float32)
    n = len(boxes)
    pred = torch.zeros((n, 6 + n), dtype=torch.float32)
    pred[:, :4] = torch.tensor(np.asarray(boxes, np.float32))
    pred[:, 4] = torch.tensor(np.asarray(confs, np.float32))
    pred[:, 6:] = torch.eye(n)
    return pred, torch.zeros((1, 3, IMG, IMG)), torch.from_numpy(fields)


def proto_path(boxes, confs, fields):
    payload = inference._build_gpu_flattened_payload_from_proto(*_inputs(boxes, confs, fields))
    assert payload is not None
    return payload


def proto_union(boxes, confs, fields):
    payload = proto_path(boxes, confs, fields)
    return np.zeros((IMG, IMG), np.uint8) if payload.union_gpu is None else payload.union_gpu.numpy().astype(np.uint8)


def cpu_retina(boxes, confs, fields):
    pred, _, proto = _inputs(boxes, confs, fields)
    payload = inference.CpuRetinaMaskPayload(
        proto=proto.numpy(), coeffs=pred[:, 6:].numpy(), boxes_xyxy=pred[:, :4].numpy(),
        confs=pred[:, 4].numpy(), orig_shape=(IMG, IMG), img_shape=(IMG, IMG), frame_path='x')
    union, confidence, _ = inference._accumulate_cpu_retina_payload_to_prediction_frame(payload, IMG)
    return union, confidence


def extent(mask, axis):
    present = np.flatnonzero(np.asarray(mask).any(axis=axis))
    return int(present[0]), int(present[-1]) + 1


def quantized(payload):
    union = payload.union_gpu.numpy() != 0
    return union, np.where(union, np.round(np.clip(payload.conf_gpu.numpy(), 0, 1) * 255), 0).astype(np.uint8)


FILLED = np.full((1, PROTO, PROTO), 5.0, np.float32)


def test_cpu_retina_mask_only_skips_score_quantization():
    pred, _, proto = _inputs([[4.4, 7.6, 30.0, 40.0]], [.9], FILLED)
    payload = inference.CpuRetinaMaskPayload(
        proto=proto.numpy(), coeffs=pred[:, 6:].numpy(), boxes_xyxy=pred[:, :4].numpy(),
        confs=pred[:, 4].numpy(), orig_shape=(IMG, IMG), img_shape=(IMG, IMG), frame_path='x')
    affine = np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32)
    scored = np.zeros((1, IMG, IMG), dtype=np.uint8)
    mask_only = np.zeros_like(scored)
    counts = inference._process_cpu_retina_prediction_frame(
        0, payload, IMG, scored, np.zeros_like(scored), affine, IMG, IMG)
    with mock.patch.object(inference, 'quantize_conf_to_u8', side_effect=AssertionError('unrequested scores')):
        assert inference._process_cpu_retina_prediction_frame(
            0, payload, IMG, mask_only, None, affine, IMG, IMG) == counts == (1, 1)
    np.testing.assert_array_equal(mask_only, scored)


def test_cpu_retina_dot_uses_fused_rounding_at_ambiguous_zero():
    # The middle product is exactly halfway between adjacent float32 values.
    # The tiny accumulated value still changes which way the fused step rounds.
    midpoint = np.float32(1 + 2 ** -12)
    fields = np.broadcast_to(
        np.array([1., midpoint, 1.], np.float32)[:, None, None], (3, 2, 2)).copy()
    coeffs = np.array([[2 ** -60, midpoint, -(1 + 2 ** -11)]], np.float32)
    payload = inference.CpuRetinaMaskPayload(
        fields, coeffs, np.array([[0, 0, 8, 8]], np.float32), np.array([.3], np.float32),
        (8, 8), (8, 8),
    )
    mask, scores, count = inference._accumulate_cpu_retina_payload_to_prediction_frame(payload, 8)
    assert mask.all() and np.all(scores == 77) and count == 1


@pytest.mark.parametrize("shape", [(0, 2, 2), (1, 0, 2), (1, 2, 0)])
def test_cpu_retina_accepts_empty_prototype_geometry(shape):
    payload = inference.CpuRetinaMaskPayload(
        np.zeros(shape, np.float32), np.zeros((1, shape[0]), np.float32),
        np.array([[0, 0, 8, 8]], np.float32), np.array([.3], np.float32),
        (8, 8), (8, 8),
    )
    mask, score, count = inference._accumulate_cpu_retina_payload_to_prediction_frame(payload, 8)
    assert not mask.any() and not score.any() and count == 0


@pytest.mark.parametrize('x1,x2', [(4.4, 7.6), (5.0, 7.0), (9.2, 11.8), (6.4, 7.6), (8.2, 9.8)])
def test_boxes_narrower_than_one_proto_cell_keep_their_detection(x1, x2):
    # The last two hold no proto cell centre; only the box-centre cell keeps them.
    union = proto_union([[x1, 20.0, x2, 40.0]], [.9], FILLED)
    assert union.any()
    left, right = extent(union, 0)
    assert (left, right) == (int(np.ceil(x1)), int(np.ceil(x2)))
    np.testing.assert_array_equal(union, cpu_retina([[x1, 20.0, x2, 40.0]], [.9], FILLED)[0])


def test_wide_box_edges_track_the_per_instance_retina_reference():
    left, right = [], []
    for x1 in 8.0 + np.arange(0.0, 8.0, 0.37):
        for width in (12.3, 17.9, 25.1):
            box = [[x1, 16.0, x1 + width, 48.0]]
            proto_extent = extent(proto_union(box, [.9], FILLED), 0)
            retina_extent = extent(cpu_retina(box, [.9], FILLED)[0], 0)
            left.append(proto_extent[0] - retina_extent[0])
            right.append(proto_extent[1] - retina_extent[1])
    for difference in (np.asarray(left), np.asarray(right)):
        assert abs(difference.mean()) <= 1.0
        assert np.abs(difference).max() <= 2


def test_tight_box_disks_have_no_directional_edge_shift():
    shifts_x, shifts_y = [], []
    centres = proto_centres = np.arange(PROTO) * 4 + 2.0
    yy, xx = np.meshgrid(proto_centres, proto_centres, indexing='ij')
    for cx in np.arange(22.0, 42.0, 2.7):
        for cy in np.arange(21.0, 43.0, 3.1):
            for radius in (5.0, 7.5, 10.5):
                field = (1.5 * (radius - np.hypot(xx - cx, yy - cy)) / 4.0).astype(np.float32)[None]
                box = [[cx - radius, cy - radius, cx + radius, cy + radius]]
                retina = cpu_retina(box, [.9], field)[0]
                union = proto_union(box, [.9], field)
                assert retina.any() and union.any()
                (pl, pr), (rl, rr) = extent(union, 0), extent(retina, 0)
                (pt, pb), (rt, rb) = extent(union, 1), extent(retina, 1)
                shifts_x.append(((pl - rl) + (pr - rr)) / 2.0)
                shifts_y.append(((pt - rt) + (pb - rb)) / 2.0)
    assert centres.size == PROTO
    assert abs(np.mean(shifts_x)) <= 0.5
    assert abs(np.mean(shifts_y)) <= 0.5


def test_reversed_zero_size_and_nan_boxes_contribute_no_pixels():
    boxes = [[30.0, 20.0, 10.0, 40.0], [10.0, 30.0, 30.0, 30.0], [float('nan'), 10.0, 20.0, 30.0]]
    payload = proto_path(boxes, [.9, .9, .9], np.repeat(FILLED, 3, axis=0))
    assert payload.instance_count == 3
    assert not bool(payload.union_gpu.any())
    assert not bool(payload.conf_gpu.any())


def _worked_example():
    field = np.full((1, PROTO, PROTO), -6.0, np.float32)
    field[0, 8:13, 6:11] = 6.0
    field[0, 8:13, 11] = -1.0
    return [[0.0, 0.0, 64.0, 64.0]], [.9], field


def test_bilinear_boundary_pixels_carry_only_the_positive_instance_score():
    boxes, confs, field = _worked_example()
    payload = proto_path(boxes, confs, field)
    union, confidence = quantized(payload)
    conf_plane = torch.from_numpy(np.where(field[0] > 0, np.float32(.9), np.float32(0)))
    nearest = F.interpolate(conf_plane[None, None], size=(IMG, IMG), mode='nearest')[0, 0].numpy()
    assert np.count_nonzero(union & (nearest == 0)) > 0, 'fixture must exercise the fallback'
    assert np.count_nonzero(union & (confidence == 0)) == 0
    assert np.count_nonzero(confidence[~union]) == 0
    expected_union, expected_score = cpu_retina(boxes, confs, field)
    np.testing.assert_array_equal(union, expected_union)
    np.testing.assert_array_equal(confidence, expected_score)


def test_contested_parallel_rois_do_not_merge_coarse_logits():
    fields = np.full((2, PROTO, PROTO), -7., np.float32)
    fields[0, :, :8] = 1.
    fields[1, :, 8:] = 1.
    boxes = [[0., 1., 48., 64.], [16., 1., 64., 64.]]
    union, scores = quantized(proto_path(boxes, [.3, .8], fields))
    expected_mask, expected_score = cpu_retina(boxes, [.3, .8], fields)
    assert union[20, 29] and union[20, 34]
    assert not union[20, 31] and not union[20, 32]
    assert scores[20, 29] == 77
    np.testing.assert_array_equal(union, expected_mask)
    np.testing.assert_array_equal(scores, expected_score)


@pytest.mark.skipif(os.getenv("XTA_TEST_NATIVE_FANOUT_CUDA") != "1", reason="explicit CUDA qualification")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_exact_retina_matches_cpu_across_contested_cells_and_thresholds(dtype):
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    size = 8
    # Different corner owners would exclude the only interior winner.
    fields = np.array([[[8, 8], [-32, -32]], [[7, -32], [7, -32]],
                       [[-32, 8], [-32, 8]], [[-32, -32], [7, 7]], [[1, 1], [1, 1]]], np.float32)
    confs = torch.tensor([.82, .83, .80, .81, .9], dtype=dtype)
    boxes = np.tile(np.array([0, 0, size, size], np.float32), (len(fields), 1))
    head = torch.zeros((5 + len(fields), len(fields)), device="cuda", dtype=dtype)
    head[:2] = size / 2
    head[2:4] = size
    head[4] = confs
    head[5:] = torch.eye(len(fields), device="cuda", dtype=dtype)
    proto = torch.from_numpy(fields).to("cuda", dtype)
    image = torch.empty((1, 1, size, size), device="cuda")
    previous = None
    for cutoff in (.5, .8, .85, .9, 1.):
        payload = inference._build_direct_device_compacted_payload(head, proto, image, cutoff)
        assert payload is not None and payload.compaction_kernel == 'exact_retina'
        payload.ready_event.synchronize()
        actual = payload.union_gpu.cpu().numpy()
        actual_scores = np.rint(payload.conf_gpu.cpu().numpy() * 255).astype(np.uint8)
        mask = confs.float().numpy() >= np.float32(cutoff)
        cpu = inference.CpuRetinaMaskPayload(
            fields, np.eye(len(fields), dtype=np.float32)[mask], boxes[mask],
            confs.float().numpy()[mask], (size, size), (size, size),
        )
        expected, expected_scores, _ = inference._accumulate_cpu_retina_payload_to_prediction_frame(cpu, size)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        if previous is not None:
            assert np.all(actual <= previous)
        previous = actual


def test_identity_scale_returns_the_nearest_plane_and_positive_scores_are_kept():
    generator = torch.Generator().manual_seed(11)
    plane = torch.rand((7, 9), generator=generator)
    plane[plane < .4] = 0
    union = torch.ones((7, 9))
    assert torch.equal(inference._upsample_proto_confidence(plane, union, (7, 9)), plane)
    sparse = torch.zeros((4, 4))
    sparse[2, 2], sparse[2, 3] = .3, .9
    upsampled = inference._upsample_proto_confidence(sparse, torch.ones((16, 16)), (16, 16))
    # Pixels whose nearest cell is (2,2) keep .3, even beside the .9 tap.
    assert torch.all(upsampled[8:12, 8:12] == sparse[2, 2])
    assert torch.all(upsampled[8:12, 12:16] == sparse[2, 3])
    background = inference._upsample_proto_confidence(sparse, torch.zeros((16, 16)), (16, 16))
    assert torch.equal(background, F.interpolate(sparse[None, None], size=(16, 16), mode='nearest')[0, 0])


@pytest.mark.parametrize('n_in,n_out', [(16, 64), (160, 640), (15, 64), (7, 20), (5, 5)])
def test_bilinear_taps_name_exactly_the_cells_with_positive_weight(n_in, n_out):
    i0, i1 = inference._proto_bilinear_taps(torch, n_in, n_out, torch.device('cpu'))
    for cell in range(n_in):
        one_hot = torch.zeros((1, 1, 1, n_in))
        one_hot[..., cell] = 1.0
        weight = F.interpolate(one_hot, size=(1, n_out), mode='bilinear', align_corners=False)[0, 0, 0]
        np.testing.assert_array_equal((weight > 0).numpy(), ((i0 == cell) | (i1 == cell)).numpy())


def _smooth_field(seed):
    generator = torch.Generator().manual_seed(seed)
    low = torch.randn((1, 1, 4, 4), generator=generator) * 6.0
    return F.interpolate(low, size=(PROTO, PROTO), mode='bicubic', align_corners=False)[0].numpy()


def _reconciled(union, confidence):
    volume, scores = union[None].astype(np.uint8), confidence[None].astype(np.uint8)
    layers = [EvidenceLayer(f'view_{name}', (1, IMG, IMG), {'mask_kind': 'yolo', 'view_name': name},
                            lambda z0, z1: volume[z0:z1],
                            lambda z0, z1: (scores[z0:z1], np.ones_like(volume[z0:z1], dtype=bool)))
              for name in ('a', 'b')]
    written = {}
    policy = dict(name='proto_confidence', mode='confidence', grouping='views', threshold=1.1,
                  min_sources=2, min_prediction_sources=2, anchor_confidence=.8, anchor_bonus=0.)
    receipt = reconcile(layers, shape_tyx=(1, IMG, IMG), policy=policy,
                        write_slab=lambda z0, z1, value: written.update(mask=value), memory_mib=64)
    return written['mask'][0], receipt['counts']


def test_confidence_reconciliation_keeps_every_proto_union_pixel_like_cpu_retina():
    box = [[0.0, 0.0, 64.0, 64.0]]
    for seed in range(6):
        field = _smooth_field(seed)
        union, confidence = quantized(proto_path(box, [.8], field))
        if not union.any():
            continue
        kept, counts = _reconciled(union, confidence)
        np.testing.assert_array_equal(kept != 0, union)
        assert counts['confidence_unknown_prediction_voxels'] == 0
        retina_union, retina_confidence = cpu_retina(box, [.8], field)
        retina_kept, retina_counts = _reconciled(retina_union != 0, retina_confidence)
        np.testing.assert_array_equal(retina_kept, retina_union)
        assert retina_counts['confidence_unknown_prediction_voxels'] == 0
