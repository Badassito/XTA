"""Verify the exact gzip-copy path through the managed scalar sink and review."""
from dataclasses import replace
import json
import os

import numpy as np
import pytest

from XTA import grouped_seg_nrrd, managed_review_source, outputs
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, NrrdLayerRef, write_raw_bbox_mask_store
from XTA.sweep_publication import publish_sweep_review


def _managed_layer(tmp_path, storage_format, *, check_link=False, expect_receipt=True, second_layer=False):
    stage = tmp_path / "stage"
    branch = tmp_path / "confidence_cutoffs" / "conf_090"
    branch.mkdir(parents=True)
    (branch / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "confidence_sweep": {"sweep_id": "fixture", "cutoff_hundredths": 90, "nrrd_storage": "grouped"},
    }))
    shape = (9, 15, 17)
    data = np.zeros(shape, dtype=np.uint8)
    data[2, 4:9, 3:12] = 1
    data[4, 3:10, 4:15] = 1
    data[4, 5:8, 6:13] = 0
    source = tmp_path / "source.cvol"
    stats = write_raw_bbox_mask_store(data, source, format_name=storage_format, workers=1)
    ref = NrrdLayerRef(
        key="mask", name="mask", path=source, shape=shape, storage_format=storage_format,
        source="fullframe", mask_kind="yolo", stage="pre_interpolation",
        segment_extent_ijk=tuple(stats["segment_extent_ijk"]), segment_extent_shape_tyx=shape,
        segment_extent_source="raw_bbox_cvol_index",
    )
    sink = outputs.NrrdLayerSink(
        nrrd_dir=stage / branch.name / "nrrd", stem="case", output_shape_tyx=shape, max_workers=1,
        managed_review_root=stage,
    )
    try:
        scalar = sink.submit_layer(ref, "layer")
        if second_layer:
            other_data = np.zeros_like(data)
            other_data[1, 8:14, 1:4] = 1
            other_data[7, 6:11, 10:14] = 1
            other_source = tmp_path / "other.cvol"
            other_stats = write_raw_bbox_mask_store(other_data, other_source, format_name=storage_format, workers=1)
            other_ref = replace(ref, key="other", name="other", path=other_source,
                                segment_extent_ijk=tuple(other_stats["segment_extent_ijk"]))
            sink.submit_layer(other_ref, "other")
        sink.wait()
        sink.write_manifest()
    finally:
        sink.shutdown()
    receipt = managed_review_source.read_native_review_source(scalar)
    assert (receipt is not None) == expect_receipt
    if receipt is not None:
        assert (receipt.directory / "chunks.bin").read_bytes() == (source / "chunks.bin").read_bytes()
        if check_link:
            assert os.path.samefile(receipt.directory / "chunks.bin", source / "chunks.bin")
    for name in ("chunks.bin", "index.bin", "meta.json"):
        (source / name).unlink()
    source.rmdir()
    return branch, stage, scalar, receipt


@pytest.mark.parametrize("storage_format", [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
def test_wider_group_publishes_exact_managed_crops(tmp_path, monkeypatch, storage_format):
    branch, stage, _, _ = _managed_layer(tmp_path, storage_format, second_layer=True)
    with monkeypatch.context() as scalar:
        scalar.setattr(managed_review_source, "read_native_review_source", lambda _path: None)
        old = publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", review_name="baseline",
                                   memory_mib=2, nrrd_stage_root=stage)

    def no_inflation(*_args, **_kwargs):
        pytest.fail("Both native sources have verified crops")

    monkeypatch.setattr(grouped_seg_nrrd, "_scan_segment", no_inflation)
    monkeypatch.setattr(managed_review_source.ManagedReviewSource, "copy_verified", no_inflation)
    new = publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", memory_mib=2, nrrd_stage_root=stage)
    group, = json.loads(new.read_bytes())["grids"][0]["groups"]
    assert len(group["segments"]) == 2
    assert {p.relative_to(new.parent): p.read_bytes() for p in new.parent.rglob("*") if p.is_file()} == {
        p.relative_to(old.parent): p.read_bytes() for p in old.parent.rglob("*") if p.is_file()}


def test_wider_group_closes_source_on_coverage_failure(tmp_path, monkeypatch):
    from XTA.sweep_coverage import DirectCoverage

    branch, stage, _, _ = _managed_layer(tmp_path, INTERNAL_PACKED_CVOL_FORMAT, second_layer=True)
    original = managed_review_source._source_planes
    closed = []

    def planes(*args):
        try:
            yield from original(*args)
        finally:
            closed.append(True)

    def fail(*_args):
        raise RuntimeError("coverage failure during repack")

    monkeypatch.setattr(managed_review_source, "_source_planes", planes)
    monkeypatch.setattr(DirectCoverage, "observe_plane", fail)
    with pytest.raises(RuntimeError, match="coverage failure during repack") as failure:
        publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", memory_mib=2, nrrd_stage_root=stage)
    assert failure.traceback and closed == [True]
    assert not (tmp_path / "review").exists()


def test_capture_counts_every_bit_in_largest_packed_plane(monkeypatch, tmp_path):
    crop = np.full((4096, 4096), 255, dtype=np.uint8)
    width = 8 * crop.shape[1]
    monkeypatch.setattr(managed_review_source, "_source_planes", lambda *_: iter([(0, crop, 0, 0, width)]))
    count, _ = managed_review_source._inspect_source(
        tmp_path, (1, 4096, width), (0, width - 1, 0, 4095, 0, 0), True)
    assert count == crop.size * 8


@pytest.mark.parametrize("storage_format", [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
def test_managed_aligned_review_matches_inflated_review(tmp_path, monkeypatch, storage_format):
    branch, stage, _, _ = _managed_layer(tmp_path, storage_format)
    with monkeypatch.context() as baseline:
        baseline.setattr(managed_review_source, "read_native_review_source", lambda _path: None)
        old = publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", review_name="baseline",
                                   memory_mib=2, nrrd_stage_root=stage)

    def unexpected_scan(*_args, **_kwargs):
        pytest.fail("Managed native reuse fell back to scalar inflation")

    monkeypatch.setattr(grouped_seg_nrrd, "_scan_segment", unexpected_scan)
    new = publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture",
                               memory_mib=2, nrrd_stage_root=stage)
    observed = {p.relative_to(new.parent): p.read_bytes() for p in new.parent.rglob("*") if p.is_file()}
    expected = {p.relative_to(old.parent): p.read_bytes() for p in old.parent.rglob("*") if p.is_file()}
    assert observed == expected


def test_native_writer_hardlinks_the_review_source_when_supported(tmp_path):
    probe = tmp_path / "probe"
    probe.write_bytes(b"link capability")
    try:
        os.link(probe, tmp_path / "probe-link")
    except OSError as error:
        pytest.skip(f"Filesystem hardlink unavailable: {error}")
    _managed_layer(tmp_path, INTERNAL_PACKED_CVOL_FORMAT, check_link=True)


def test_review_source_quota_uses_validated_scalar_path(tmp_path, monkeypatch):
    # A positive limit below one receipt exercises the shared quota admission.
    monkeypatch.setenv("XTA_YOLO_SWEEP_REVIEW_SOURCE_GIB", "0.000001")
    branch, stage, _, _ = _managed_layer(tmp_path, INTERNAL_PACKED_CVOL_FORMAT, expect_receipt=False)
    manifest = publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", memory_mib=2,
                                    nrrd_stage_root=stage)
    grid = json.loads(manifest.read_bytes())["grids"][0]
    assert grid["groups"][0]["segments"][0]["foreground_voxels"] == 101
    assert grid["group_compression"]["reused_spans"] == 1


@pytest.mark.parametrize("corruption", ["scalar", "crops", "index"])
@pytest.mark.parametrize("second_layer", [False, True])
def test_damaged_managed_source_prevents_atomic_publication(tmp_path, corruption, second_layer):
    branch, stage, scalar, receipt = _managed_layer(tmp_path, INTERNAL_PACKED_CVOL_FORMAT, second_layer=second_layer)
    path = {"scalar": scalar, "crops": receipt.directory / "chunks.bin",
            "index": receipt.directory / "index.bin"}[corruption]
    with path.open("r+b") as file:
        offset = file.seek(-1, 2)
        byte = file.read(1)[0]
        file.seek(offset)
        file.write(bytes([byte ^ 1]))
    with pytest.raises(ValueError, match="Managed review .* digest mismatch"):
        publish_sweep_review(tmp_path, [(90, branch)], sweep_id="fixture", memory_mib=2, nrrd_stage_root=stage)
    assert not (tmp_path / "review").exists()
    assert not list(tmp_path.glob(".review.atomic-result-*"))
