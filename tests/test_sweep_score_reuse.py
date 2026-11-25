from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import confidence_evidence as evidence
from XTA import confidence_projection
from XTA.geometry import get_view_infos
from XTA.sweep_score_reuse import SweepScoreProjection
from XTA import sweep_score_reuse


@pytest.fixture
def publication(tmp_path, monkeypatch):
    # The existing score projector takes the maximum across contributing
    # native positions. Count its invocation while keeping its real geometry
    # and source evidence writer on this downsampled transverse fixture.
    original = confidence_projection.score_projection_reader
    calls = []

    @contextmanager
    def observe(source, view, shape, work):
        calls.append(True)
        with original(source, view, shape, work) as reader:
            yield reader

    monkeypatch.setattr(confidence_projection, "score_projection_reader", observe)
    view = get_view_infos(4, 260, 262, cartesian_views=("transverse",))[0]
    work = tmp_path / "work"
    work.mkdir()
    output_shape = (2, 130, 131)

    def publish(values, cutoff, *, reusable=True, streamed=False, budget=8 * 1024**2):
        cache = (SweepScoreProjection(work, sweep_id="fixture", cutoff=cutoff,
                                      floor=0, budget_bytes=budget) if reusable else None)
        run = len(list(tmp_path.glob("out-*")))
        evidence.configure_confidence_evidence(
            tmp_path / f"out-{run}", enabled=True, run_id=f"branch-{cutoff}",
            prediction_task="segment", confidence_floor=cutoff / 100,
            sweep_score_projection=cache,
        )
        try:
            source = SimpleNamespace(shape=values.shape) if streamed else values
            return evidence.publish_confidence_scores(
                source, view=view, model_name="model", temp_dir=tmp_path / f"scratch-{run}",
                output_shape=output_shape, native_reader=values.__getitem__ if streamed else None,
            )
        finally:
            evidence.configure_confidence_evidence(None)

    return work, output_shape, publish, calls


@pytest.mark.parametrize("steps", [26, 51])
def test_one_projection_and_identical_evidence_for_certified_family(publication, steps):
    _, shape, publish, calls = publication
    rng = np.random.default_rng(361)
    native = rng.integers(0, 256, (4, 260, 262), np.uint8)
    native[rng.random(native.shape) < 0.97] = 0
    for offset, cutoff in enumerate(range(0, 101, 100 // (steps - 1))):
        scores = np.where(native >= offset * 4, native, np.uint8(0))
        if offset:
            expected = publish(scores, cutoff, reusable=False)
            count = len(calls)
        result = publish(scores, cutoff)
        if offset:
            assert len(calls) == count
            for name in ("index.bin", "scores.u8.zlib"):
                assert (expected.path / name).read_bytes() == (result.path / name).read_bytes()
            assert expected.metadata == result.metadata
            np.testing.assert_array_equal(expected.read(0, shape[0])[0], result.read(0, shape[0])[0])
    # One real projection for the family and one per baseline sibling.
    assert len(calls) == steps


@pytest.mark.parametrize("fault", ["score_changed", "ordering", "split_bin", "new_support", "missing", "modified",
                                  "zero_credit", "streamed", "unchanged"])
def test_unsafe_or_unavailable_cache_keeps_scalar_scores(publication, fault, monkeypatch):
    work, shape, publish, calls = publication
    floor = np.zeros((4, 260, 262), np.uint8)
    floor[0, 3, 3] = 60
    floor[0, 3, 4] = 200
    floor[0, 4, 4] = 200
    strict = floor.copy()
    if fault != "unchanged":
        strict[0, 3, 3] = 0
    if fault == "score_changed":
        strict[0, 3, 4] = 240
    elif fault == "ordering":
        strict[0, 3, 4] = 0
        strict[0, 4, 4] = 0
        strict[0, 3, 3] = 60
    elif fault == "split_bin":
        strict[0, 3, 4] = 0
    elif fault == "new_support":
        strict[-1, -1, -1] = 251
    streamed = fault == "streamed"
    publish(floor, 0, reusable=fault != "missing", streamed=streamed,
            budget=0 if fault == "zero_credit" else 8 * 1024**2)
    if fault == "modified":
        proof = next(work.glob("confidence-projection-floor/*/native.scores"))
        proof.write_bytes(proof.read_bytes() + b"\x00")
    expected = publish(strict, 50, reusable=False)
    count = len(calls)
    if streamed:
        def unexpected(*args, **kwargs):
            pytest.fail("An exact streamed proof should bypass the output resize")

        monkeypatch.setattr(confidence_projection, "read_score_slice_in_output_shape", unexpected)
    result = publish(strict, 50, streamed=streamed)
    assert len(calls) == count + int(fault not in ("streamed", "unchanged"))
    for name in ("index.bin", "scores.u8.zlib"):
        assert (expected.path / name).read_bytes() == (result.path / name).read_bytes()
    np.testing.assert_array_equal(expected.read(0, shape[0])[0], result.read(0, shape[0])[0])
    assert result.metadata == expected.metadata


def test_native_proof_reads_bounded_chunks_and_skips_empty_planes(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_score_reuse, "_NATIVE_CHUNK_BYTES", 23)
    floor = np.zeros((3, 7, 19), dtype=np.uint8)[..., ::-1]
    floor[0, 3, 17] = 255
    floor[1, 1, 7] = 80
    floor[2, 6, 18] = 160
    strict = floor.copy()
    strict[1] = 0
    path = tmp_path / "proof.bin"
    assert sweep_score_reuse._write_native_proof(path, floor.shape, floor.__getitem__, 8192)
    assert sweep_score_reuse._threshold_for_native_scores(
        path, floor.shape, lambda z: None if z == 1 else strict[z]) == 81
    assert sweep_score_reuse._threshold_for_native_scores(path, floor.shape, lambda _: None) == 256


@pytest.mark.parametrize("view_name", ["transverse", "sagittal", "coronal"])
def test_exact_orthogonal_grid_borrows_planes_without_building_a_cache(tmp_path, view_name):
    native = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    view = get_view_infos(*native.shape, cartesian_views=(view_name,))[0]
    axes = {"transverse": (0, 1, 2), "sagittal": (1, 0, 2), "coronal": (1, 2, 0)}[view_name]
    projected = native.transpose(axes)
    source = native
    cache = SweepScoreProjection(tmp_path, sweep_id="fixture", cutoff=0, floor=0, budget_bytes=1024**2)
    evidence.configure_confidence_evidence(tmp_path / "out", enabled=True, sweep_score_projection=cache)
    try:
        result = evidence.publish_confidence_scores(source, view=view, model_name="model", temp_dir=tmp_path,
                                                    output_shape=projected.shape)
        np.testing.assert_array_equal(projected, result.read(0, projected.shape[0])[0])
        assert not cache.root.exists()
    finally:
        evidence.configure_confidence_evidence(None)
