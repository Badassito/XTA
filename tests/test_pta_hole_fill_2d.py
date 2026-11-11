"""PTA orthogonal closure seals 3D tunnels while keeping exterior slice gaps."""

from pathlib import Path
from unittest import mock
import weakref

import numpy as np
import pytest
from scipy import ndimage

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_preprocessing import fill_orthogonal_mask_holes_2d_inplace
from XTA import pta_preprocessing
from XTA.pta_runtime import build_runtime_options


def _source(mask, *, volume_class="fully_labeled", label_source="nrrd"):
    depth = mask.shape[0]
    return pta.SourceVolume(
        input_dir=Path("dataset"), stem="sample", kind="sequence",
        image_paths=[], video_path=None, labels_by_frame={},
        segmentation_nrrd_path=Path("sample.nrrd"), mask_volume=mask,
        volume_class=volume_class, label_source=label_source,
        input_start_index=0,
        encoded_indices=tuple(range(depth)),
        volume=np.arange(mask.size, dtype=np.uint8).reshape(mask.shape), fps=1.0,
    )


def _prepare(source, output, *options):
    config = parse_pta_args(["--input", "dataset", "--save", "voxel_volume", *options])
    args = build_runtime_options(config)
    warnings = pta.WarningLog()
    prepared = pta.prepare_loaded_source(
        source, args=args, warnings=warnings, workers=1, out_dir=output,
        tile_configs=(), channel_variants=pta.expand_channel_variants(
            pta.resolve_channel_formats(args.channel_format)), write_side_effects=False,
    )
    return prepared, warnings


@pytest.mark.parametrize("axis", range(3))
def test_seals_boundary_connected_tunnel_on_each_orthogonal_axis(axis):
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[1:8, 1:8, 1:8] = 1
    tunnel = [4, 4, 4]
    tunnel[axis] = slice(None)
    mask[tuple(tunnel)] = 0
    before = mask.copy()
    assert not ndimage.binary_fill_holes(mask)[4, 4, 4]
    stats = fill_orthogonal_mask_holes_2d_inplace(mask)
    assert mask[4, 4, 4] == 1
    assert np.all(mask[before != 0] == before[before != 0])
    assert stats["added_voxels"] == 7
    assert stats["execution_order"] == ["xy", "xz", "yz"]


def test_plane_streaming_matches_four_connected_scipy_reference():
    generator = np.random.default_rng(971)
    for shape in ((1, 9, 11), (7, 1, 9), (7, 8, 1), (7, 8, 9)):
        for _ in range(10):
            source = (generator.random(shape) > .35).astype(np.uint8)
            # Preserve actual nonzero values; newly closed pixels are binary 1.
            source[source != 0] = 7
            expected = source.copy()
            for axis in range(3):
                for plane in np.moveaxis(expected, axis, 0):
                    filled = ndimage.binary_fill_holes(plane != 0)
                    plane[filled & (plane == 0)] = 1
            actual = source.copy()
            stats = fill_orthogonal_mask_holes_2d_inplace(actual)
            np.testing.assert_array_equal(actual, expected)
            assert stats["added_voxels"] == np.count_nonzero(actual) - np.count_nonzero(source)


def test_background_slit_and_foreground_at_corner_preserve_exterior():
    mask = np.ones((1, 9, 9), dtype=np.uint8)
    mask[:, 4, :5] = 0
    before = mask.copy()
    assert fill_orthogonal_mask_holes_2d_inplace(mask)["added_voxels"] == 0
    np.testing.assert_array_equal(mask, before)


def _scipy_ordered_fill(source):
    expected = source.copy()
    counts = []
    for axis in range(3):
        added = changed = 0
        for plane in np.moveaxis(expected, axis, 0):
            holes = ndimage.binary_fill_holes(plane != 0) & (plane == 0)
            count = int(np.count_nonzero(holes))
            plane[holes] = 1
            added += count
            changed += bool(count)
        counts.append((added, changed))
    return expected, counts


def test_shifted_plane_bounds_reuse_scratch_and_preserve_open_exterior():
    planes = np.zeros((4, 13, 17), np.uint8)
    planes[0, 1:10, 2:13] = 7
    planes[0, 3:8, 4:11] = 0
    planes[1, :7, 10:] = 255
    planes[1, 2:5, 12:15] = 0
    planes[1, 3, 10:13] = 0
    planes[2, 4:11, 3:5] = 7
    planes[3, 8:13, :8] = 255
    expected = planes.copy()
    for source, target in zip(planes, expected):
        holes = ndimage.binary_fill_holes(source != 0) & (source == 0)
        target[holes] = 1

    exterior = np.full((15, 19), 42, np.uint8)
    added, changed = pta_preprocessing._fill_mask_hole_planes(planes, exterior)

    np.testing.assert_array_equal(planes, expected)
    assert (added, changed) == (35, 1)


def test_ordered_sweep_retains_single_pass_behavior_when_later_fill_seals_exit():
    mask = np.zeros((3, 3, 4), np.uint8)
    mask[1, 0, 1:3] = 1
    mask[1, 2, 1:3] = 1
    mask[0, 1, 1] = 1
    mask[2, 1, 1] = 1
    mask[1, 1, 3] = 1

    first = fill_orthogonal_mask_holes_2d_inplace(mask)
    assert [row["added_voxels"] for row in first["axes"]] == [0, 0, 1]
    assert mask[1, 1, 2] == 0
    second = fill_orthogonal_mask_holes_2d_inplace(mask)
    assert [row["added_voxels"] for row in second["axes"]] == [1, 0, 0]
    assert mask[1, 1, 2] == 1
    third = fill_orthogonal_mask_holes_2d_inplace(mask)
    assert third["added_voxels"] == 0


@pytest.mark.parametrize("strided", [False, True])
def test_blocked_yz_matches_ordered_reference_across_block_boundaries(strided):
    source = np.random.default_rng(471).choice(
        np.array([0, 7, 255], np.uint8), size=(9, 11, 131), p=[.4, .4, .2])
    expected, counts = _scipy_ordered_fill(source)
    if strided:
        backing = np.zeros((9, 11, 262), np.uint8)
        actual = backing[:, :, ::2]
        actual[:] = source
    else:
        actual = source.copy()
    stats = fill_orthogonal_mask_holes_2d_inplace(actual)
    np.testing.assert_array_equal(actual, expected)
    assert [(row['added_voxels'], row['changed_planes']) for row in stats['axes']] == counts
    if strided:
        assert not np.any(backing[:, :, 1::2])


@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('block_width', [2, 64])
def test_yz_staging_bounds_live_blocks_and_preserves_open_tunnel(monkeypatch, block_width, workers):
    source = np.zeros((9, 11, 131), np.uint8)
    source[1:8, 1:10] = 7
    source[3:6, 4:7] = 0  # Open along X; only YZ closure fills it.
    expected, counts = _scipy_ordered_fill(source)
    original = np.ascontiguousarray
    previous = []
    block_bytes = source.shape[0] * source.shape[1] * block_width
    monkeypatch.setattr(pta_preprocessing, '_ORTHOGONAL_HOLE_FILL_BLOCK_BYTES', block_bytes)
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: 1 << 40)

    def stage(array):
        # Each task retires its block on return, so at most one block per
        # worker is live; a serial fill never holds two at once.
        assert sum(ref() is not None for ref in list(previous)) < workers
        staged = original(array)
        assert staged.nbytes <= block_bytes
        previous.append(weakref.ref(staged))
        return staged

    monkeypatch.setattr(pta_preprocessing.np, 'ascontiguousarray', stage)
    stats = fill_orthogonal_mask_holes_2d_inplace(source, workers=workers)
    assert len(previous) == (source.shape[2] + block_width - 1) // block_width
    assert all(ref() is None for ref in previous)
    np.testing.assert_array_equal(source, expected)
    assert counts[:2] == [(0, 0), (0, 0)]
    assert [(row['added_voxels'], row['changed_planes']) for row in stats['axes']] == counts
    assert stats['axes'][2]['workers'] == workers


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('successful_blocks', [0, 1])
def test_yz_staging_allocation_failure_keeps_plane_fallback(monkeypatch, successful_blocks, workers):
    source = np.zeros((9, 11, 131), np.uint8)
    source[1:8, 1:10] = 255
    source[3:6, 4:7] = 0
    expected, counts = _scipy_ordered_fill(source)

    original = np.ascontiguousarray
    calls = []

    def exhausted(array):
        calls.append(None)
        if len(calls) <= successful_blocks:
            return original(array)
        raise MemoryError('no optional staging headroom')

    monkeypatch.setattr(pta_preprocessing.np, 'ascontiguousarray', exhausted)
    stats = fill_orthogonal_mask_holes_2d_inplace(source, workers=workers)
    np.testing.assert_array_equal(source, expected)
    assert [(row['added_voxels'], row['changed_planes']) for row in stats['axes']] == counts
    assert all(row['examined_planes'] == row['planes'] for row in stats['axes'])


def _random_masks(generator, count):
    """Random, structured, edge-touching, empty, full and multi-valued masks."""
    for index in range(count):
        shape = tuple(int(value) for value in generator.integers(1, 26, size=3))
        kind = index % 6
        if kind == 0:
            mask = (generator.random(shape) < generator.uniform(.3, .95)).astype(np.uint8)
        elif kind == 1:
            noise = ndimage.gaussian_filter(generator.random(shape), sigma=float(generator.uniform(.5, 2.5)))
            mask = (noise > np.quantile(noise, float(generator.uniform(.1, .8)))).astype(np.uint8)
        elif kind == 2:
            # Nested hollow shells, some opened by a tunnel along one axis.
            mask = np.zeros(shape, np.uint8)
            for _ in range(int(generator.integers(1, 5))):
                low = [int(generator.integers(0, max(1, size - 2))) for size in shape]
                high = [int(generator.integers(lo + 1, size + 1)) for lo, size in zip(low, shape)]
                mask[tuple(slice(lo, hi) for lo, hi in zip(low, high))] = 1
                if all(hi - lo > 2 for lo, hi in zip(low, high)):
                    mask[tuple(slice(lo + 1, hi - 1) for lo, hi in zip(low, high))] = 0
                    if generator.random() < .5:
                        axis = int(generator.integers(0, 3))
                        tunnel = [(lo + hi) // 2 for lo, hi in zip(low, high)]
                        tunnel[axis] = slice(None)
                        mask[tuple(tunnel)] = 0
        elif kind == 3:
            mask = (generator.integers(0, 4, size=shape) * (generator.random(shape) < .7)).astype(np.uint8)
        elif kind == 4:
            mask = np.full(shape, int(generator.integers(0, 2)) * 255, np.uint8)
        else:
            # Foreground on every face around random cavities, plus a face-open pit.
            mask = np.ones(shape, np.uint8)
            if min(shape) > 2:
                mask[1:-1, 1:-1, 1:-1] = generator.random(tuple(size - 2 for size in shape)) < .4
            mask[0, shape[1] // 2, shape[2] // 2] = 0
        yield mask


@pytest.mark.parametrize('strided', [False, True])
def test_threaded_fill_matches_serial_bytes_and_axis_stats(monkeypatch, strided):
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: 1 << 40)
    generator = np.random.default_rng(1531)
    fields = ('family', 'axis', 'planes', 'examined_planes', 'changed_planes', 'added_voxels')
    for index, source in enumerate(_random_masks(generator, 120)):
        # Small staging budgets exercise many concurrently staged YZ blocks.
        block_bytes = source.shape[0] * source.shape[1] * int(generator.integers(2, 6))
        monkeypatch.setattr(pta_preprocessing, '_ORTHOGONAL_HOLE_FILL_BLOCK_BYTES', block_bytes)
        serial = source.copy()
        serial_stats = fill_orthogonal_mask_holes_2d_inplace(serial, workers=1)
        for workers in (2, 3, 8):
            if strided:
                backing = np.full(source.shape[:2] + (source.shape[2] * 2,), 9, np.uint8)
                actual = backing[:, :, ::2]
                actual[:] = source
            else:
                actual = source.copy()
            stats = fill_orthogonal_mask_holes_2d_inplace(actual, workers=workers)
            np.testing.assert_array_equal(actual, serial, err_msg=f'case {index} workers {workers}')
            if strided:
                assert np.all(backing[:, :, 1::2] == 9)
            for row, expected in zip(stats['axes'], serial_stats['axes']):
                assert [row[key] for key in fields] == [expected[key] for key in fields]
                assert 1 <= row['workers'] <= workers
            assert stats['added_voxels'] == serial_stats['added_voxels']
            assert stats['execution_order'] == ['xy', 'xz', 'yz']


def test_threaded_staging_admits_workers_against_host_headroom(monkeypatch):
    source = np.zeros((9, 11, 131), np.uint8)
    source[1:8, 1:10] = 7
    source[3:6, 4:7] = 0
    expected, _ = _scipy_ordered_fill(source)
    block_bytes = source.shape[0] * source.shape[1] * 4
    monkeypatch.setattr(pta_preprocessing, '_ORTHOGONAL_HOLE_FILL_BLOCK_BYTES', block_bytes)
    yz_task = (source.shape[0] + 2) * (source.shape[1] + 2) + block_bytes
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: (256 << 20) + 2 * yz_task)
    stats = fill_orthogonal_mask_holes_2d_inplace(source, workers=16)
    np.testing.assert_array_equal(source, expected)
    assert stats['axes'][2]['workers'] == 2
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: 1)
    assert fill_orthogonal_mask_holes_2d_inplace(source, workers=16)['axes'][2]['workers'] == 1


def test_threaded_fill_propagates_task_failure(monkeypatch):
    original = pta_preprocessing._fill_mask_hole_planes
    calls = []

    def failing(planes, exterior):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('plane task failed')
        return original(planes, exterior)

    monkeypatch.setattr(pta_preprocessing, '_fill_mask_hole_planes', failing)
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: 1 << 40)
    with pytest.raises(RuntimeError, match='plane task failed'):
        fill_orthogonal_mask_holes_2d_inplace(np.ones((16, 5, 5), np.uint8), workers=4)


@pytest.mark.parametrize("mask", [
    np.zeros((3, 3), np.uint8), np.zeros((0, 3, 3), np.uint8),
    np.zeros((3, 3, 3), np.float32),
])
def test_invalid_input_fails_before_mutation(mask):
    with pytest.raises(ValueError, match="writable nonempty 3-D uint8"):
        fill_orthogonal_mask_holes_2d_inplace(mask)


def test_readonly_input_fails():
    mask = np.zeros((3, 3, 3), np.uint8)
    mask.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        fill_orthogonal_mask_holes_2d_inplace(mask)


def test_flag_implies_void_fill_and_complete_closure_skips_proven_noop_void_fill(tmp_path):
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    mask[:, 3, 3] = 0
    source = _source(mask.copy())
    intensity = source.volume.copy()
    phases = []

    def resize(volume, current, **kwargs):
        assert current[3, 3, 3] == 1
        phases.append("resize")
        return volume, current, current.shape

    config = parse_pta_args(["--input", "dataset", "--enable_2D_hole_fill"])
    assert config.preprocessing.enable_2d_hole_fill
    assert config.preprocessing.enable_3d_void_fill
    args = build_runtime_options(config)
    with mock.patch.object(pta, "fill_3d_voids_inplace_streaming", side_effect=AssertionError("void fill ran")), \
            mock.patch("XTA.pta_preprocessing.run_gpu_preprocessing", side_effect=AssertionError("GPU void fill ran")), \
            mock.patch.object(pta, "_VOID_FILL_GPU_MIN_VOXELS", 0), \
            mock.patch.object(np, "count_nonzero", wraps=np.count_nonzero) as counts, \
            mock.patch.object(pta, "resize_to_approximately_cube", side_effect=resize):
        prepared, _ = _prepare(source, tmp_path, "--enable_2D_hole_fill")
        counts.reset_mock()
        # Without voxel accounting the skip adds no foreground scan.
        bare = pta.prepare_loaded_source(
            _source(mask.copy()), args=args, warnings=pta.WarningLog(), workers=2, out_dir=tmp_path,
            preprocessing_device_ids=(0,), tile_configs=(), channel_variants=pta.expand_channel_variants(
                pta.resolve_channel_formats(args.channel_format)), write_side_effects=False,
        )
        assert not any(call.args and getattr(call.args[0], "shape", None) == mask.shape for call in counts.call_args_list)
    assert phases == ["resize", "resize"]
    assert prepared.hole_fill_2d_stats["added_voxels"] == 5
    assert prepared.hole_fill_2d_stats["axes"][-1]["examined_planes"] == 7
    assert prepared.void_fill_stats == {
        "background_connectivity": 6, "added_voxels": 0, "backend": "skipped",
        "reason": "skipped_proven_noop_after_2d_hole_fill",
        "foreground_before": 125, "foreground_after": 125,
    }
    assert prepared.voxel_final == 125
    assert bare.void_fill_stats == {
        "background_connectivity": 6, "added_voxels": 0, "backend": "skipped",
        "reason": "skipped_proven_noop_after_2d_hole_fill",
    }
    np.testing.assert_array_equal(bare.mask_for_render, prepared.mask_for_render)
    np.testing.assert_array_equal(source.volume, intensity)


def _break_last_axis(field):
    real = pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace

    def fill(mask, **kwargs):
        stats = real(mask, **kwargs)
        if field == "order":
            stats["execution_order"] = ["xy", "yz", "xz"]
        elif field == "connectivity":
            stats["background_connectivity"] = 8
        elif field == "family":
            stats["axes"][-1]["family"] = "xz"
        else:
            stats["axes"][-1][field] -= 1
        return stats
    return fill


@pytest.mark.parametrize("field", ["order", "connectivity", "family", "examined_planes", "planes"])
def test_unproven_closure_keeps_void_fill_before_resize(tmp_path, field):
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    mask[:, 3, 3] = 0
    phases = []

    def void_fill(current, *args, **kwargs):
        assert current[3, 3, 3] == 1
        phases.append("void_fill")

    def resize(volume, current, **kwargs):
        phases.append("resize")
        return volume, current, current.shape

    with mock.patch("XTA.pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace", side_effect=_break_last_axis(field)), \
            mock.patch.object(pta, "fill_3d_voids_inplace_streaming", side_effect=void_fill), \
            mock.patch.object(pta, "resize_to_approximately_cube", side_effect=resize):
        prepared, _ = _prepare(_source(mask), tmp_path, "--enable_2D_hole_fill")
    assert phases == ["void_fill", "resize"]
    assert "reason" not in prepared.void_fill_stats
    assert prepared.void_fill_stats["foreground_before"] == 125


@pytest.mark.parametrize("text,expected", [("", 6), ("18", 18), (" 26 ", 26), ("six", 6), ("4", None)])
def test_skipped_void_fill_keeps_connectivity_parsing(tmp_path, monkeypatch, text, expected):
    monkeypatch.setenv("YOLO_TTA_VOIDFILL_CONNECTIVITY", text)
    mask = np.ones((5, 5, 5), np.uint8)
    mask[2, 2, 2] = 0
    if expected is None:
        for options in (["--enable_2D_hole_fill"], ["--preprocessing", "3d_void_fill"]):
            with pytest.raises(ValueError, match="one of 6, 18, or 26"):
                _prepare(_source(mask.copy()), tmp_path, *options)
        return
    prepared, _ = _prepare(_source(mask), tmp_path, "--enable_2D_hole_fill")
    assert prepared.void_fill_stats["background_connectivity"] == expected
    assert prepared.void_fill_stats["reason"] == "skipped_proven_noop_after_2d_hole_fill"


def _enclosed_background(mask, connectivity):
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, count = ndimage.label(mask == 0, structure=structure)
    exterior = np.zeros(count + 1, bool)
    exterior[0] = True
    for axis in range(3):
        for index in (0, -1):
            exterior[np.take(labels, index, axis=axis)] = True
    return int(np.count_nonzero(~exterior[labels]))


def test_void_fill_after_complete_2d_closure_adds_no_voxels(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_TTA_VOIDFILL_CONNECTIVITY", raising=False)
    monkeypatch.setattr(pta_preprocessing, 'available_anon_work_bytes', lambda: 1 << 40)
    generator = np.random.default_rng(20261107)
    enclosed_before = dict.fromkeys((6, 18, 26), 0)
    for index, mask in enumerate(_random_masks(generator, 150)):
        for connectivity in enclosed_before:
            enclosed = _enclosed_background(mask, connectivity)
            enclosed_before[connectivity] += bool(enclosed)
            if enclosed:
                # Non-vacuous: the production fill closes them before 2D closure.
                filled = mask.copy()
                pta.fill_3d_voids_inplace_streaming(filled, tmp_path / f"before_{index}", connectivity=connectivity)
                assert int(np.count_nonzero(filled != mask)) == enclosed
        # Force the staged YZ path on half of the volumes.
        monkeypatch.setattr(pta_preprocessing, '_ORTHOGONAL_HOLE_FILL_BLOCK_BYTES',
                            mask.shape[0] * mask.shape[1] * 3 if index % 2 else 128 << 20)
        stats = fill_orthogonal_mask_holes_2d_inplace(mask, workers=1 + index % 4)
        assert pta._void_fill_proven_noop(stats, mask)
        for connectivity in enclosed_before:
            assert _enclosed_background(mask, connectivity) == 0
            filled = mask.copy()
            pta.fill_3d_voids_inplace_streaming(filled, tmp_path / f"void_{index}", connectivity=connectivity)
            np.testing.assert_array_equal(filled, mask, err_msg=f"case {index} connectivity {connectivity}")
    # The generated volumes include enclosed cavities before the 2D closure.
    assert all(count >= 5 for count in enclosed_before.values()), enclosed_before


def test_flag_off_preserves_mask_and_preprocessing_path(tmp_path):
    mask = np.ones((5, 5, 5), np.uint8)
    mask[:, 2, 2] = 0
    with mock.patch("XTA.pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace",
                    side_effect=AssertionError("unrequested hole filling")):
        prepared, _ = _prepare(_source(mask.copy()), tmp_path)
    np.testing.assert_array_equal(prepared.mask_for_render, mask)
    assert prepared.hole_fill_2d_stats is None
    assert prepared.void_fill_stats is None


def test_real_preparation_closes_tunnel_that_standalone_void_fill_retains(tmp_path):
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[1:8, 1:8, 1:8] = 1
    mask[:, 4, 4] = 0
    standalone, _ = _prepare(_source(mask.copy()), tmp_path / "standalone",
                              "--preprocessing", "3d_void_fill")
    combined, _ = _prepare(_source(mask.copy()), tmp_path / "combined",
                            "--enable_2D_hole_fill")
    assert standalone.mask_for_render[4, 4, 4] == 0
    assert combined.mask_for_render[4, 4, 4] == 1
    assert standalone.voxel_final == 336
    assert combined.voxel_final == 343
    assert combined.hole_fill_2d_stats["added_voxels"] == 7
    assert combined.void_fill_stats["foreground_before"] == 343
    np.testing.assert_array_equal(standalone.volume_for_render, combined.volume_for_render)


@pytest.mark.parametrize("reason", ["partial_volume", "unlabeled_volume", "encoded_gaps"])
def test_ineligible_volume_skips_2d_and_3d_transforms(tmp_path, reason):
    source = _source(np.ones((5, 5, 5), np.uint8))
    if reason == "partial_volume":
        source.volume_class = "partially_labeled"
    elif reason == "unlabeled_volume":
        source.volume_class = "unlabeled"
        source.label_source = "none"
    else:
        source.encoded_indices = (0, 1, 3, 4, 5)
    with mock.patch("XTA.pta_preprocessing.fill_orthogonal_mask_holes_2d_inplace",
                    side_effect=AssertionError("ineligible hole filling")):
        prepared, warnings = _prepare(source, tmp_path, "--enable_2D_hole_fill")
    assert prepared.hole_fill_2d_stats is None
    assert prepared.void_fill_stats is None
    assert f"2d_hole_fill_disabled_for_{reason}" in warnings.counts


def test_forced_partial_keeps_semantic_unknown_coverage(tmp_path):
    mask = np.ones((5, 5, 5), np.uint8)
    mask[:, 2, 2] = 0
    source = _source(mask, volume_class="partially_labeled", label_source="yolo")
    states = (pta.ANNOTATION_FOREGROUND, pta.ANNOTATION_FOREGROUND,
              pta.ANNOTATION_UNANNOTATED, pta.ANNOTATION_FOREGROUND,
              pta.ANNOTATION_FOREGROUND)
    with mock.patch.object(pta, "rasterize_yolo_labels", return_value=(mask, None)), \
            mock.patch.object(pta, "derive_annotation_states", return_value=states), \
            mock.patch.object(pta, "collect_foreground_slice_anchors", return_value=((), 0)):
        prepared, _ = _prepare(source, tmp_path, "--enable_2D_hole_fill", "--force",
                               "--task", "semantic", "--save", "semantic")
    assert prepared.hole_fill_2d_stats["added_voxels"] == 5
    assert prepared.mask_for_render[2, 2, 2] == 1
    assert not np.any(prepared.semantic_coverage_for_render[2])
    assert np.all(prepared.semantic_coverage_for_render[1])


def test_pipeline_outputs_match_forced_void_fill(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import json
    import re

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.delenv('YOLO_TTA_VOIDFILL_CONNECTIVITY', raising=False)
    monkeypatch.setattr(pta, 'discover_topology', lambda **kwargs: SimpleNamespace(
        cuda_device_ids=(), gpu_cpu_sets=(), worker_cpu_order=(),
        allowed_cpus=(0, 1, 2, 3), summary='CPU hole fill'))
    source_dir = tmp_path / 'input'
    source_dir.mkdir()
    shape = (9, 12, 10)
    generator = np.random.default_rng(5)
    for index in range(shape[0]):
        assert pta.cv2.imwrite(str(source_dir / f'sample_{index:04d}.png'),
                               generator.integers(0, 255, shape[1:], dtype=np.uint8))
    mask = np.zeros(shape, np.uint8)
    mask[1:8, 2:10, 2:9] = 1
    mask[3:6, 4:8, 4:7] = 0  # Enclosed cavity.
    mask[:, 6, 3] = 0  # Tunnel through every transverse plane.
    mask[0, 0, :] = 1  # Foreground on a volume edge.
    pta.write_nrrd(mask, source_dir / 'sample.nrrd')

    def run(name):
        output = tmp_path / name
        arguments = ['--input', str(source_dir), '--output', str(output), '--imgsz', '8',
                     '--enable_cartesian', 'transverse', 'coronal', 'sagittal', '--enable_2D_hole_fill',
                     '--background_percent', '0.5', '--save', 'images', 'labels', 'nrrd', 'summary',
                     '--worker_backend', 'thread', '--workers', '3', '--frame_workers', '1',
                     '--jpeg_decode_backend', 'opencv', '--no-topology_aware']
        pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
        # pynrrd stamps a wall-clock comment line; every other byte must match.
        files = {str(path.relative_to(output)): re.sub(rb'\n# on [^\n]*', b'', path.read_bytes(), count=1)
                 for folder in ('images', 'labels', 'nrrds')
                 for path in (output / folder).rglob('*') if path.is_file()}
        volume = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['volumes'][0]
        return files, volume

    classify = pta.classify_original_foregrounds_for_volume
    classified = []

    def checked_classify(prep, **kwargs):
        # Anchor support replaces the full render-mask scan with the same result.
        support = prep.foreground_preservation_stats.pop('guaranteed_output_foreground_transverse_slices')
        try:
            expected = classify(prep, **kwargs)
        finally:
            prep.foreground_preservation_stats['guaranteed_output_foreground_transverse_slices'] = support
        count_nonzero = np.count_nonzero

        def guarded(value, *args, **kw):
            assert value is not prep.mask_for_render, 'render mask scanned'
            return count_nonzero(value, *args, **kw)

        with mock.patch.object(pta.np, 'count_nonzero', side_effect=guarded):
            actual = classify(prep, **kwargs)
        assert support > 0 and actual == expected
        classified.append(actual)
        return actual

    monkeypatch.setattr(pta, 'classify_original_foregrounds_for_volume', checked_classify)
    skipped, skipped_volume = run('skipped')
    assert len(classified) == 1 and any(classified[0].values()) and not all(classified[0].values())
    assert all(row['workers'] > 1 for row in skipped_volume['2d_hole_fill_preprocessing']['axes'])
    monkeypatch.setattr(pta, 'classify_original_foregrounds_for_volume', classify)
    monkeypatch.setattr(pta, '_void_fill_proven_noop', lambda *args: False)
    filled, filled_volume = run('filled')
    assert skipped_volume['3d_void_fill_preprocessing']['reason'] == 'skipped_proven_noop_after_2d_hole_fill'
    assert filled_volume['3d_void_fill_preprocessing']['added_voxels'] == 0
    assert 'sample_VoidFill.nrrd' in {Path(name).name for name in skipped}
    assert any(name.startswith('labels') for name in skipped)
    assert skipped == filled
    for key in ('render_plans', 'candidates_total', 'candidates_retained', 'foreground_anchor_repair'):
        assert skipped_volume[key] == filled_volume[key]
