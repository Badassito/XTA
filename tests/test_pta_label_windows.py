"""Cropped polygon export and its GPU windows reproduce full-frame labels exactly."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import os
import random
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

from XTA import polygon_keyholes, pta_workers as workers
from XTA.polygon_keyholes import holes_by_parent, splice_holes
from XTA.pta_dataset import OutputCandidate
from XTA.pta_publication import (
    _yolo_coordinate_text, candidate_output_paths, mask_to_yolo_lines, yolo_label_window,
)
from tests.test_pta_gpu_publication import FakeTensor, FakeTorch


def reference_lines(mask01, *, warnings, context):
    """The full-frame, per-vertex exporter before label windows (v25.0.3), with keyhole holes."""
    m = np.asarray(mask01, dtype=np.uint8)
    if not m.size:
        return []
    h, w = m.shape
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    children = holes_by_parent(hierarchy)
    discarded = False
    lines = []
    hierarchy_arr = [None] * len(contours) if hierarchy is None else list(hierarchy[0])
    for index, (cnt, hrow) in enumerate(zip(contours, hierarchy_arr)):
        if hrow is not None and int(hrow[3]) >= 0:
            continue
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is None or len(approx) < 3:
            warnings.add("degenerate_export_polygon", context)
            continue
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            warnings.add("degenerate_export_polygon", context)
            continue
        holes = [cv2.approxPolyDP(contours[child], 1.0, True).reshape(-1, 2) for child in children.get(index, ())]
        discarded |= any(len(hole) < 3 for hole in holes)
        pts = splice_holes(pts, [hole for hole in holes if len(hole) >= 3])
        coords = []
        for x, y in pts:
            xn = min(1.0, max(0.0, float(x) / float(max(1, w))))
            yn = min(1.0, max(0.0, float(y) / float(max(1, h))))
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
        lines.append("0 " + " ".join(coords))
    if discarded:
        warnings.add("holes_discarded_during_polygon_export", context)
    return lines


def export(function, mask, **kwargs):
    events = []
    lines = function(mask, warnings=SimpleNamespace(add=lambda *args: events.append(args)),
                     context='case', **kwargs)
    return lines, events


def bounds(mask):
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return [mask.shape[0], -1, mask.shape[1], -1]
    return [int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])]


def windowed(mask):
    window = yolo_label_window(bounds(mask), mask.shape)
    if window is None:
        return export(mask_to_yolo_lines, np.zeros((0, 0), np.uint8))
    y0, y1, x0, x1 = window
    return export(mask_to_yolo_lines, np.ascontiguousarray(mask[y0:y1, x0:x1]),
                  frame_shape=mask.shape, origin=(x0, y0))


def adversarial(rng, h, w, kinds=tuple(range(9))):
    m = np.zeros((h, w), np.uint8)
    kind = rng.choice(kinds)
    if kind == 0:  # Blobs, often clipped by the frame.
        for _ in range(rng.randrange(1, 8)):
            cv2.ellipse(m, (rng.randrange(-w // 4, w + w // 4 + 1), rng.randrange(-h // 4, h + h // 4 + 1)),
                        (rng.randrange(1, max(2, w // 2)), rng.randrange(1, max(2, h // 2))),
                        rng.uniform(0, 180), 0, 360, 1, -1)
    elif kind == 1:  # Holes with nested islands.
        for _ in range(rng.randrange(1, 4)):
            cy, cx, r = rng.randrange(h), rng.randrange(w), rng.randrange(3, max(4, min(h, w) // 2))
            cv2.circle(m, (cx, cy), r, 1, -1)
            cv2.circle(m, (cx, cy), max(1, r // 2), 0, -1)
            cv2.circle(m, (cx, cy), max(1, r // 4), 1, -1)
    elif kind == 2:  # Single pixels and diagonal-only pairs, including corners.
        for _ in range(rng.randrange(1, 40)):
            y, x = rng.choice([0, h - 1, rng.randrange(h)]), rng.choice([0, w - 1, rng.randrange(w)])
            m[y, x] = 1
            if rng.random() < 0.6:
                m[min(h - 1, y + 1), min(w - 1, max(0, x + rng.choice([-1, 1])))] = 1
    elif kind == 3:  # Strips along frame edges.
        t = rng.randrange(1, 4)
        for side in rng.sample(range(4), rng.randrange(1, 5)):
            a, b = sorted(rng.sample(range(max(h, w) + 1), 2))
            if side == 0:
                m[:t, a % w:b % w + 1] = 1
            elif side == 1:
                m[-t:, a % w:b % w + 1] = 1
            elif side == 2:
                m[a % h:b % h + 1, :t] = 1
            else:
                m[a % h:b % h + 1, -t:] = 1
    elif kind == 4:  # Full frame with holes, some open to the frame.
        m[:] = 1
        for _ in range(rng.randrange(0, 4)):
            cv2.circle(m, (rng.randrange(w), rng.randrange(h)), rng.randrange(1, max(2, min(h, w) // 3)), 0, -1)
    elif kind == 5:  # Checkerboard: diagonal connectivity everywhere.
        s = rng.randrange(1, 3)
        yy, xx = np.mgrid[0:h, 0:w]
        m[:] = ((yy // s + xx // s) % 2).astype(np.uint8)
        m[: rng.randrange(h + 1)] = 0
    elif kind == 6:  # Sparse noise.
        m[:] = np.random.default_rng(rng.randrange(1 << 30)).random((h, w)) > rng.uniform(0.7, 0.99)
    elif kind == 7:  # Foreground one pixel inside an edge, so the margin reaches it.
        y0 = rng.choice([1, max(1, h - 2)]) if h > 2 else 0
        x0 = rng.choice([1, max(1, w - 2)]) if w > 2 else 0
        m[y0:y0 + rng.randrange(1, 3), x0:x0 + rng.randrange(1, 3)] = 1
    # kind 8 stays empty.
    return m * rng.choice([1, 1, 7, 255])


def adversarial_masks(count, seed=20261107):
    rng = random.Random(seed)
    sizes = [(1, 1), (1, 5), (5, 1), (2, 3), (3, 3), (6, 9), (17, 23), (64, 64), (97, 131), (160, 120)]
    return [adversarial(rng, *rng.choice(sizes)) for _ in range(count)]


def test_label_window_margin_stops_at_frame_edges():
    assert yolo_label_window([5, -1, 7, -1], (5, 7)) is None
    assert yolo_label_window([2, 3, 4, 4], (10, 10)) == (1, 5, 3, 6)
    assert yolo_label_window([0, 9, 0, 9], (10, 10)) == (0, 10, 0, 10)
    assert yolo_label_window([0, 0, 9, 9], (1, 10)) == (0, 1, 8, 10)


def test_crop_outside_frame_is_rejected():
    with pytest.raises(ValueError, match='outside frame'):
        export(mask_to_yolo_lines, np.ones((3, 3), np.uint8), frame_shape=(4, 4), origin=(2, 0))
    with pytest.raises(ValueError, match='outside frame'):
        export(mask_to_yolo_lines, np.ones((3, 3), np.uint8), frame_shape=(4, 4), origin=(-1, 0))


def test_full_frame_and_windowed_exports_match_reference_on_adversarial_masks():
    edge = 0
    for index, mask in enumerate(adversarial_masks(1500)):
        expected = export(reference_lines, mask)
        assert export(mask_to_yolo_lines, mask) == expected, index
        assert windowed(mask) == expected, index
        # Strided host views take the same full-frame path.
        assert export(mask_to_yolo_lines, mask.T) == export(reference_lines, mask.T), index
        edge += bool(mask.size and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))
    assert edge > 500


def test_windowed_hole_splices_match_full_frame_on_the_kd_tree_branch(monkeypatch):
    monkeypatch.setattr(polygon_keyholes, '_BRUTE_FORCE_PAIRS', 0)
    grid = np.zeros((1024, 1024), np.uint8)
    grid[3:1021, 5:1019] = 1
    for y in range(9, 1012, 16):
        for x in range(9, 1012, 16):
            grid[y:y + 4, x:x + 4] = 0
    for index, mask in enumerate(adversarial_masks(300) + [grid]):
        expected = export(reference_lines, mask)
        assert export(mask_to_yolo_lines, mask) == expected, index
        assert windowed(mask) == expected, index


@pytest.mark.parametrize('shape', [(3072, 3072), (2048, 3072), (3061, 17)])
def test_large_frames_match_reference(shape):
    rng = random.Random(sum(shape))
    for _ in range(3):
        mask = adversarial(rng, *shape, kinds=(0, 1, 2, 3, 4, 7, 8))  # Not millions of specks.
        expected = export(reference_lines, mask)
        assert export(mask_to_yolo_lines, mask) == expected
        assert windowed(mask) == expected


def test_coordinate_text_matches_per_vertex_format_for_every_pixel():
    for size in (1, 2, 3, 7, 640, 3072, 3064):
        text = _yolo_coordinate_text(size)
        assert not text.flags.writeable
        assert text.tolist() == [f"{min(1.0, max(0.0, float(v) / float(max(1, size)))):.6f}"
                                 for v in range(size)]


def device_masks():
    masks = [mask.astype(np.uint8) for mask in adversarial_masks(64, seed=7) if mask.shape == (17, 23)]
    masks += [np.zeros((17, 23), np.uint8), np.ones((17, 23), np.uint8)]
    for y, x in ((0, 0), (16, 22), (0, 22), (16, 0), (8, 11)):
        masks.append(np.zeros((17, 23), np.uint8))
        masks[-1][y, x] = 9
    return np.stack(masks)


@pytest.mark.parametrize('layout', ['contiguous', 'channel_view'])
def test_device_bounds_and_windows_match_host_masks(layout):
    torch = pytest.importorskip('torch')
    host = device_masks()
    tensor = torch.as_tensor(host)
    if layout == 'channel_view':
        tensor = tensor[:, None].repeat(1, 2, 1, 1)[:, 1]
        assert not tensor.is_contiguous()
    got = workers._gpu_mask_bounds(torch, tensor).tolist()
    assert got == [bounds(mask) for mask in host]
    coverage = torch.as_tensor(np.random.default_rng(3).random(host.shape) > 0.5)
    assert workers._gpu_mask_bounds(torch, tensor * coverage).tolist() == [
        bounds(mask) for mask in host * coverage.numpy()]
    windows = [(index, *yolo_label_window(b, host.shape[1:]))
               for index, b in enumerate(got) if b[1] >= 0][::-1]
    crops = workers._download_gpu_mask_windows(torch, tensor, windows)
    for (index, y0, y1, x0, x1), crop in zip(windows, crops):
        assert crop.flags.c_contiguous and crop.dtype == np.uint8
        np.testing.assert_array_equal(crop, host[index, y0:y1, x0:x1])


def label_candidates(count):
    return [OutputCandidate(order=index, volume_name='sample', parent_view_tag='Transverse',
                            output_tag='Transverse', item_key='full', frame_idx=index, is_tile=False,
                            label_enabled=True) for index in range(count)]


@contextmanager
def recorded_spans():
    spans = []
    telemetry = SimpleNamespace(span=lambda name, **fields: spans.append((name, fields)) or nullcontext())
    with mock.patch.object(workers, 'runtime_telemetry', lambda: telemetry):
        yield spans


def label_bytes(lines):
    return ('\n'.join(lines) + '\n' if lines else '').replace('\n', os.linesep).encode()


@pytest.mark.parametrize('save_binary', [False, True])
def test_publisher_label_files_and_warnings_match_full_frame_reference(tmp_path, save_binary):
    torch = FakeTorch()
    masks = [mask for mask in adversarial_masks(400, seed=11) if mask.shape == (17, 23)]
    masks[1] = np.zeros((17, 23), np.uint8)
    masks[1][2:9, 3:12] = 1
    masks[2] = np.zeros((17, 23), np.uint8)
    masks = np.stack(masks)
    candidates = label_candidates(len(masks))
    for index in (1, 2):  # Augmented foreground: kept with a polygon, dropped without.
        candidates[index] = replace(candidates[index], augmentation_index=1, augmentation_seed=3,
                                    augmentation_tag='AbCd1234EfGh5678', foreground=True)
    downloads = []
    original_to = FakeTensor.to

    def record_to(tensor, device):
        downloads.append(tuple(tensor.shape))
        return original_to(tensor, device)

    with mock.patch.dict(workers._WORKER_STATIC, dict(
        out_dir=tmp_path, split_active=False, image_format='png', save_images=False,
        save_labels=True, save_binary=save_binary, save_semantic=False,
    ), clear=True), mock.patch.object(FakeTensor, 'to', record_to), recorded_spans() as spans, \
            mock.patch.object(workers, 'publish_binary_mask_payloads'):
        warnings = workers.WarningLog()
        result = workers._publish_gpu_policy_batch(
            runtime={'torch': torch, 'device_id': 0},
            batch_images=FakeTensor(torch, np.zeros((len(masks), 1, 17, 23), np.uint8)),
            batch_masks=FakeTensor(torch, masks), candidates=candidates, output_size=(17, 23),
            channel_kind='gray', local_warnings=warnings,
            source_mask_nonempty=tuple(bool(mask.any()) or index % 3 == 0 for index, mask in enumerate(masks)))
    expected = workers.WarningLog()
    for index, (mask, cand) in enumerate(zip(masks, candidates)):
        lines = reference_lines(mask, warnings=expected, context=f'sample Transverse frame {index + 1:04d}')
        path = candidate_output_paths(tmp_path, cand, split_active=False, image_format='png')[1]
        assert path.exists() == (index != 2)
        assert index == 2 or path.read_bytes() == label_bytes(lines), index
    expected.add('augmented_foreground_flip_dropped', 'sample/Transverse/frame=0003/tag=AbCd1234EfGh5678')
    assert result == (len(masks) - 1, {'all': 1})
    assert dict(warnings.counts) == dict(expected.counts) and dict(warnings.examples) == dict(expected.examples)
    assert downloads[0] == (len(masks), 4)
    if save_binary:
        assert downloads[1:] == [masks.shape]
        assert [name for name, _ in spans] == ['pta.publish.mask_bounds']
    else:
        nonempty = [b for b in map(bounds, masks) if b[1] >= 0]
        window_pixels = sum((w[1] - w[0]) * (w[3] - w[2])
                            for w in (yolo_label_window(b, (17, 23)) for b in nonempty))
        assert downloads[1:] == [(window_pixels,)]
        assert spans == [('pta.publish.mask_bounds', {}), ('pta.publish.label_window_download', dict(
            windows=len(nonempty), window_pixels=window_pixels, frame_pixels=len(nonempty) * 17 * 23))]


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_GPU_PUBLICATION') != '1',
                    reason='opt-in CUDA; requires exclusive Scratch/Temp/GPU_LOCK')
def test_cuda_pipeline_label_windows_match_full_frame_reference(tmp_path):
    import torch
    from XTA.pta_gpu_publication import GpuPublicationResources
    rng = random.Random(5)
    host = np.stack([adversarial(rng, 300, 200) for _ in range(12)] + [np.zeros((300, 200), np.uint8)])
    runtime = {'torch': torch, 'device_id': 0}
    resources = GpuPublicationResources(runtime, cpu_threads=2)
    results = []
    task = resources.task(workers._publish_gpu_policy_batch, results.append)
    candidates = label_candidates(len(host))
    try:
        with mock.patch.dict(workers._WORKER_STATIC, dict(
            out_dir=tmp_path, split_active=False, image_format='png', save_images=False,
            save_labels=True, save_binary=False, save_semantic=False,
        ), clear=True):
            images = torch.zeros((len(host), 1, 300, 200), dtype=torch.uint8, device='cuda:0')
            masks = torch.as_tensor(host, device='cuda:0')[:, None].repeat(1, 2, 1, 1)[:, 1]
            with task.reserve(2 * (images.numel() + masks.numel())) as slot:
                task.submit(slot, dict(runtime=runtime, batch_images=images, batch_masks=masks,
                    candidates=candidates, output_size=(300, 200), channel_kind='gray',
                    local_warnings=workers.WarningLog(),
                    source_mask_nonempty=(True,) * len(host)))
            masks.zero_()
            task.close()
        assert results == [(len(host), {})]
        for mask, cand in zip(host, candidates):
            lines = reference_lines(mask, warnings=mock.Mock(), context='x')
            path = candidate_output_paths(tmp_path, cand, split_active=False, image_format='png')[1]
            assert path.read_bytes() == label_bytes(lines)
    finally:
        try:
            task.close()
        finally:
            resources.finalizer.cancel()
            resources.close()
