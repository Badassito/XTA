"""CPU-only exact science and ownership checks for independently scheduled tiles."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from threading import Event, Lock
import weakref

import numpy as np
import pytest

from XTA import _prediction_cleanup, assembly, confidence_tiles, geometry, runtime
from XTA.confidence_evidence import configure_confidence_evidence
from XTA.interpolation import DeferredTilePostprocessResult, RawBBoxMaskStore, TilePostprocessTask


@pytest.fixture
def confidence_work(tmp_path, monkeypatch):
    monkeypatch.setattr(_prediction_cleanup, "yolo_hole_fill_enabled", lambda: False)
    configure_confidence_evidence(tmp_path / "out", enabled=True, defer_projection=True)
    try:
        yield tmp_path
    finally:
        configure_confidence_evidence(None, enabled=False)


def _concrete_views(shape):
    view, = geometry.get_view_infos(*shape, cartesian_views=("transverse",))
    return geometry.expand_views_into_policy_variants(
        geometry.expand_views_into_tta_variants([view], [0]), 2,
    )


def _read_ctile(result):
    assert isinstance(result, DeferredTilePostprocessResult)
    with closing(RawBBoxMaskStore.open(result.tile_mask_path, mmap_payload=True)) as store:
        assert store.shape == result.tile_shape
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])



def _read_native_scores(reference):
    shape = reference.storage_shape
    scores = np.zeros(shape, np.uint8)
    with reference.native_reader() as reader:
        for z in range(shape[0]):
            for y0, y1, x0, x1, crop in reader.iter_crops(z):
                scores[z, y0:y1, x0:x1] = crop
    return scores

def _retire_main(work, view, config, tile, crop, mask, scores):
    mask_path, scores_path = (work / f"{view.name}-{tile}-{mode}.dat" for mode in ("mask", "scores"))
    mask_path.write_bytes(mask.tobytes())
    scores_path.write_bytes(scores.tobytes())
    raw_mask = np.memmap(mask_path, mode="r+", dtype=np.uint8, shape=mask.shape)
    raw_scores = np.memmap(scores_path, mode="r+", dtype=np.uint8, shape=scores.shape)
    owners = weakref.ref(raw_mask._mmap), weakref.ref(raw_scores._mmap)
    task = TilePostprocessTask(
        model_name="detector", view_name=view.name, aug_id=view.tta_aug_id,
        angle_deg=view.tta_angle_deg, config_id=config, tile_id=tile, parent_crop=crop,
        tile_mask_mm=raw_mask, tile_confmap_mm=raw_scores,
        tile_mask_path=mask_path, tile_confmap_path=scores_path,
        threshold_plane_shape=(view.src_h, view.src_w),
    )
    result = assembly.postprocess_tile_volume_after_inference(
        task, view=view, min_conf=0, min_radius=0, slice_workers=1,
        sparse_retire_dir=work,
    )
    return result, owners, (mask_path, scores_path)


def test_empty_main_and_policy_crop_retire_to_separate_owners(confidence_work):
    work = confidence_work
    base, policy = _concrete_views((4, 8, 11))
    crop = (2, 6, 3, 8)
    empty = np.zeros((4, 4, 5), np.uint8)
    mask = empty.copy()
    mask[2, 1:3, 1:4] = 1
    scores = np.full(mask.shape, 199, np.uint8)
    scores[2, 1, 1] = 0  # An observed mask may include an unknown score.

    base_result, base_owners, base_paths = _retire_main(
        work, base, "one", "same-tile-coordinate", crop, empty, empty,
    )
    result, policy_owners, policy_paths = _retire_main(
        work, policy, "one", "same-tile-coordinate", crop, mask, scores,
    )
    assert base_result is None
    for owner, path in zip((*base_owners, *policy_owners), (*base_paths, *policy_paths), strict=True):
        assert owner() is None
        runtime.wait_for_retired_memmap_unlinks(path=path)
        assert not path.exists()
    assert result.view_name == policy.name and result.aug_id == policy.tta_aug_id
    assert result.parent_crop == crop
    np.testing.assert_array_equal(_read_ctile(result), mask)

    retained = confidence_tiles._RAW[("detector", policy.name, "one", "same-tile-coordinate")]
    raw_scores = _read_native_scores(retained.reference)
    expected = np.where(mask, scores, 0)
    np.testing.assert_array_equal(raw_scores, expected)
    assembly._delete_tile_result_storage(result)
    confidence_tiles.retire_tile_confidence("detector", base.name, "one")
    confidence_tiles.retire_tile_confidence("detector", policy.name, "one")
    assert not confidence_tiles._RAW
    assert not result.tile_mask_path.exists()



def test_empty_main_and_two_nonempty_seam_groups_seal_without_parent(confidence_work):
    from XTA.tile_pipeline import IndependentTileSets

    work = confidence_work
    shape = (6, 8, 13)
    view = geometry.ViewInfo("azimuthal__tta_r000p000", *shape, "clamp", family="azimuthal")
    key = ("detector", view.name, "one")
    main_id = "tile"
    ordinary_id = geometry.azimuthal_batch_padding_tile_id(main_id, mirror_azimuthal_u=False)
    reflected_id = geometry.azimuthal_batch_padding_tile_id(main_id, mirror_azimuthal_u=True)
    crop = (1, 3, 2, 5)
    mirrored = geometry.mirrored_azimuthal_parent_crop(crop, shape[2])
    slots = np.zeros((3, 2, 3), np.uint8)
    slots[0, 0, :2] = 1
    slots[1, 0, 1:] = 1
    slots[2, 1, 0] = 1
    scores = np.zeros_like(slots)
    scores[0, 0, :2] = (120, 141)
    scores[1, 0, 1:] = (130, 167)
    scores[2, 1, 0] = 211

    def retire(tile_id, resolved_crop, placements):
        return assembly.retire_azimuthal_seam_tile(
            slots, scores, placements, model_name=key[0], view=view, config_id=key[2],
            tile_id=tile_id, parent_crop=resolved_crop, threshold_plane_shape=shape[1:],
            precleaned=False, min_conf=0, min_radius=0, workers=1, work_dir=work,
        )

    ordinary = retire(ordinary_id, crop, ((1, 5), (0, 5)))
    reflected = retire(reflected_id, mirrored, ((2, 1),))
    ledger = IndependentTileSets({key: (main_id, ordinary_id, reflected_id)})
    assert ledger.accept(key, reflected_id, reflected) is None
    assert ledger.accept(key, main_id, None) is None
    observations = ledger.accept(key, ordinary_id, ordinary)
    assert len(observations) == 2
    assert {item.tile_id for item in observations} == {ordinary_id, reflected_id}
    assert ordinary.parent_crop == crop and reflected.parent_crop == mirrored

    expected = np.zeros((6, 2, 3), np.uint8)
    expected[5, 0, :] = 1
    np.testing.assert_array_equal(_read_ctile(ordinary), expected)
    expected.fill(0)
    expected[1, 1, 0] = 1
    np.testing.assert_array_equal(_read_ctile(reflected), expected)
    for result, z, y, row in ((ordinary, 5, 0, (120, 141, 167)), (reflected, 1, 1, (211, 0, 0))):
        retained = confidence_tiles._RAW[(*key, result.tile_id)]
        saved_scores = _read_native_scores(retained.reference)
        expected_scores = np.zeros((6, 2, 3), np.uint8)
        expected_scores[z, y] = row
        np.testing.assert_array_equal(saved_scores, expected_scores)
        assembly._delete_tile_result_storage(result)
    confidence_tiles.retire_tile_confidence(*key)
    assert not confidence_tiles._RAW


def test_tile_set_rejects_duplicate_without_completing_a_missing_key():
    from XTA.tile_pipeline import IndependentTileSets

    primary = ("detector", "view", "first")
    other = ("detector", "view__policy_001", "first")
    with pytest.raises(ValueError, match="Invalid tile observation inventory"):
        IndependentTileSets({primary: ("same", "same")})
    ledger = IndependentTileSets({primary: ("same", "second"), other: ("same",)})
    with pytest.raises(RuntimeError, match="Unexpected tile observation"):
        ledger.accept(primary, "foreign", None)
    assert ledger.accept(primary, "same", None) is None
    with pytest.raises(RuntimeError, match="Duplicate tile observation"):
        ledger.accept(primary, "same", None)
    assert ledger.accept(other, "same", None) == ()
    assert ledger.incomplete() == {primary: {"completed": 1, "expected": 2}}
    assert ledger.accept(primary, "second", None) == ()
    assert not ledger.incomplete()



def _config_settings(work, view, config):
    return dict(
        model_name="detector", view=view, config_id=config, temp_dir=work, interpolate=0,
        interpolation_walk_back=1, interpolation_candidates=1, interpolate_passes=0,
        interpolate_min_radius=0, interpolation_search_angle=30, slice_workers=1,
        interpolation_task_workers=1,
    )


class _RecordingAdmission:
    def __init__(self):
        self.active = 0
        self.reservations = []

    @contextmanager
    def reserve(self, size, description):
        assert not self.active
        self.reservations.append(size)
        self.active = size
        try:
            yield
        finally:
            self.active = 0


@pytest.mark.parametrize("parent_first", [False, True], ids=["parent-last", "config-last"])
def test_independent_configs_join_exactly_in_either_completion_order(confidence_work, parent_first):
    from XTA.tile_pipeline import IndependentTileSets, analyze_tile_config, join_tile_config

    work = confidence_work
    shape = (4, 8, 11)
    base, _ = _concrete_views(shape)
    parent = np.zeros(shape, np.uint8)
    parent[0, 0, 0] = 1
    local1, local2 = np.zeros((4, 3, 4), np.uint8), np.zeros((4, 2, 3), np.uint8)
    local1[1, 1, 1:3] = 1
    local2[3, 0, 1:] = 1
    entries = (("first", (1, 4, 2, 6), local1), ("second", (5, 7, 7, 10), local2))
    expected = parent.copy()
    staging = work / "staging"
    admission = _RecordingAdmission()
    destination = parent.copy() if parent_first else None
    paths = []
    lock = Lock()
    expected_ids = {("detector", base.name, config): (f"{config}-tile", "empty") for config, _, _ in entries}
    expected_ids[("detector", base.name, "all-empty")] = ("only",)
    ledger = IndependentTileSets(expected_ids)

    empty = ledger.accept(("detector", base.name, "all-empty"), "only", None)
    empty_result, empty_path = analyze_tile_config(
        empty, shape=shape, staging_dir=staging,
        settings=_config_settings(work, base, "all-empty"), admission=admission)
    assert empty_path is None and empty_result.final_accumulator_mm is None
    assert not admission.reservations

    for config, crop, local in entries:
        key = ("detector", base.name, config)
        original, _, _ = _retire_main(work, base, config, f"{config}-tile", crop, local, local * np.uint8(181))
        expected[:, crop[0]:crop[1], crop[2]:crop[3]] |= local
        assert ledger.accept(key, original.tile_id, original) is None
        observations = ledger.accept(key, "empty", None)
        assert tuple(item.tile_id for item in observations) == (original.tile_id,)
        result, handoff = analyze_tile_config(
            observations, shape=shape, staging_dir=staging, settings=_config_settings(work, base, config),
            admission=admission, destination=destination, destination_lock=lock)
        assert result.final_accumulator_mm is None and not admission.active
        assert not original.tile_mask_path.exists()
        if parent_first:
            assert handoff is None
        else:
            assert handoff.is_dir()
            paths.append(handoff)

    if not parent_first:
        destination = parent.copy()
        assert len(paths) == len(entries)
        wrong_grid = np.zeros((4, 8, 10), np.uint8)
        with pytest.raises(ValueError, match="destination processing grid"):
            join_tile_config(paths[0], destination=wrong_grid, lock=lock, workers=1)
        assert paths[0].is_dir() and not wrong_grid.any()
        for handoff in reversed(paths):
            join_tile_config(handoff, destination=destination, lock=lock, workers=1)
            assert not handoff.exists()

    np.testing.assert_array_equal(destination, expected)
    assert parent[0, 0, 0] == 1 and int(parent.sum()) == 1
    assert int(destination.sum()) == int(parent.sum()) + int(local1.sum()) + int(local2.sum())
    assert admission.reservations == [2 * parent.size] * len(entries)
    runtime.wait_for_retired_memmap_unlinks(under=staging)
    assert not list(staging.rglob("observations.u8.dat"))
    assert not confidence_tiles._RAW


def test_analysis_failure_retires_unconsumed_stores_and_owned_credit(confidence_work):
    from XTA.tile_pipeline import IndependentTileSets, analyze_tile_config

    work = confidence_work
    shape, crop = (4, 8, 11), (1, 4, 2, 6)
    base, _ = _concrete_views(shape)
    config = "test"
    key = ("detector", base.name, config)
    local = np.zeros((4, 3, 4), np.uint8)
    local[1, 1, 1:3] = 1
    observations = [_retire_main(work, base, config, str(i), crop, local, local * np.uint8(137))[0]
                    for i in range(3)]
    ledger = IndependentTileSets({key: ("0", "1", "2")})
    with pytest.raises(ValueError, match="identity differs"):
        ledger.accept(key, "0", replace(observations[0], view_name="foreign-view"))
    assert ledger.incomplete() == {key: {"completed": 0, "expected": 3}}
    assert ledger.accept(key, "2", observations[2]) is None
    assert ledger.accept(key, "0", observations[0]) is None
    sealed = ledger.accept(key, "1", replace(observations[1], parent_crop=(0, 3, 10, 14)))
    assert [item.tile_id for item in sealed] == ["0", "1", "2"]
    assert not ledger.incomplete()
    with pytest.raises(RuntimeError, match="Duplicate"):
        ledger.accept(key, "1", None)

    admission = _RecordingAdmission()
    staging = work / "staging"
    parent = np.zeros(shape, np.uint8)
    with pytest.raises(ValueError, match="crop or shape differs"):
        analyze_tile_config(
            sealed, shape=shape, staging_dir=staging, settings=_config_settings(work, base, config),
            admission=admission, destination=parent, destination_lock=Lock())
    assert not admission.active and admission.reservations == [2 * parent.size]
    assert not parent.any()
    assert not any(observation.tile_mask_path.exists() for observation in observations)
    assert not confidence_tiles._RAW
    runtime.wait_for_retired_memmap_unlinks(under=staging)
    assert not list(staging.rglob("observations.u8.dat"))



def test_raw_result_retirement_waits_for_exported_owners(confidence_work, monkeypatch):
    from XTA import tile_pipeline

    work = confidence_work
    base, _ = _concrete_views((4, 8, 11))
    crop, shape = (1, 4, 2, 6), (4, 3, 4)
    expected = np.zeros(shape, np.uint8)
    expected[2, 1:, 1:3] = 1
    mask_path, score_path = work / "raw-mask.dat", work / "raw-scores.dat"
    mask_path.write_bytes(expected.tobytes())
    score_path.write_bytes((expected * np.uint8(137)).tobytes())
    aliases, backing_refs, live_at_credit = [], [], []
    waiting, credit_ready = Event(), Event()
    actual_wait = runtime.wait_for_retired_memmap_unlinks

    def note_wait(*args, **kwargs):
        waiting.set()
        return actual_wait(*args, **kwargs)

    def consume(mask, scores):
        aliases.extend((np.asarray(mask)[:, :, :], np.asarray(scores)[:, :, :]))
        backing_refs.extend(weakref.ref(array._mmap) for array in (mask, scores))
        task = TilePostprocessTask(
            model_name="detector", view_name=base.name, aug_id=base.tta_aug_id,
            angle_deg=base.tta_angle_deg, config_id="config", tile_id="tile", parent_crop=crop,
            tile_mask_mm=mask, tile_confmap_mm=scores, tile_mask_path=mask_path, tile_confmap_path=score_path,
            threshold_plane_shape=(base.src_h, base.src_w),
        )
        return assembly.postprocess_tile_volume_after_inference(
            task, view=base, min_conf=0, min_radius=0, slice_workers=1, sparse_retire_dir=work,
        )

    def cleanup_receipt(_future):
        live_at_credit.extend(ref() is not None for ref in backing_refs)
        credit_ready.set()

    monkeypatch.setattr(runtime, "wait_for_retired_memmap_unlinks", note_wait)
    monkeypatch.setattr(tile_pipeline, "wait_for_retired_memmap_unlinks", note_wait, raising=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tile_pipeline.consume_tile_files, mask_path, score_path, shape, consume)
        future.add_done_callback(cleanup_receipt)
        try:
            assert waiting.wait(3), (future.exception() if future.done() else "cleanup did not start")
            assert not future.done() and not credit_ready.is_set()
            assert all(owner() is not None for owner in backing_refs)
            np.testing.assert_array_equal(aliases[0], expected)
        finally:
            aliases.clear()
        result = future.result(timeout=5)
        assert credit_ready.wait(3)
    assert live_at_credit == [False, False]
    assert not mask_path.exists() and not score_path.exists()
    np.testing.assert_array_equal(_read_ctile(result), expected)
    assembly._delete_tile_result_storage(result)
    confidence_tiles.retire_tile_confidence("detector", base.name, "config")


def test_raw_result_callback_failure_keeps_cause_and_retires_files(tmp_path):
    from XTA.tile_pipeline import consume_tile_files

    shape = (2, 3, 4)
    mask_path, score_path = (tmp_path / f"{lane}.dat" for lane in ("mask", "score"))
    mask_path.write_bytes(bytes(np.prod(shape)))
    score_path.write_bytes(bytes(np.prod(shape)))
    failure = OSError("independent tile cleanup failed")
    owners = []

    def fail(mask, scores):
        owners.extend(weakref.ref(array._mmap) for array in (mask, scores))
        raise failure

    with pytest.raises(OSError, match="independent tile cleanup failed") as caught:
        consume_tile_files(mask_path, score_path, shape, fail)
    assert caught.value is failure
    assert len(owners) == 2 and all(owner() is None for owner in owners)
    assert not mask_path.exists() and not score_path.exists()
