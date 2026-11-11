"""Deferred CPR extraction: exact plans and candidates; a thread only beside GPU classification."""
from __future__ import annotations

import ast
import json
import os
import pickle
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_cpr, pta_workers
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options
from tests.test_pta_cpr import _tube_mask
from tests.test_pta_deferred_classification import parent_planner
from tests.test_pta_multi_input import artifacts


SHAPE = (26, 30, 28)
TILES = (pta.TileConfig(8, 5, "t8s5"), pta.TileConfig(12, 12, "t12s12"))
VARIANTS = tuple(pta.expand_channel_variants(pta.resolve_channel_formats(["gray", "C3S1"])))


def tube_mask(kind="tubes"):
    if kind == "empty":
        return np.zeros(SHAPE, np.uint8)
    if kind == "speck":  # below the coarse extraction resolution: no CPR views
        mask = np.zeros(SHAPE, np.uint8)
        mask[13, 15, 14] = 1
        return mask
    # A straight tube, a bent tube with a side branch, and a short separate stub.
    return _tube_mask(SHAPE, [
        ((2, 6, 6), (23, 6, 6), 2.5, 1),
        ((3, 20, 8), (12, 22, 18), 2.0, 1),
        ((12, 22, 18), (23, 14, 24), 2.0, 1),
        ((12, 22, 18), (20, 27, 12), 1.5, 1),
        ((17, 11, 19), (21, 12, 22), 1.5, 1),
    ])


def source(tmp_path, mask, stem="tube"):
    volume = np.random.default_rng(11).integers(0, 256, mask.shape, dtype=np.uint8)
    return pta.SourceVolume(
        input_dir=tmp_path, stem=stem, kind="sequence", image_paths=[], video_path=None,
        labels_by_frame={}, segmentation_nrrd_path=tmp_path / f"{stem}.nrrd", mask_volume=mask.copy(),
        volume_class="fully_labeled", label_source="nrrd", input_start_index=0,
        encoded_indices=tuple(range(mask.shape[0])), volume=volume, fps=1.0,
    )


def runtime_args(tmp_path, *extra):
    return build_runtime_options(parse_pta_args([
        "--input", str(tmp_path), "--enable_cpr", "--imgsz", "32",
        "--enable_cartesian", "transverse,sagittal,coronal", *extra,
    ]))


def fresh_dir(tmp_path):
    """Each preparation keeps its disk-backed processing cubes open in its own directory."""
    index = len(list(tmp_path.glob("out*")))
    return tmp_path / f"out{index}"


def prepare(tmp_path, *, defer, kind="tubes", args=None, warnings=None, workers=2):
    return pta.prepare_loaded_source(
        source(tmp_path, tube_mask(kind)), args=args or runtime_args(tmp_path),
        warnings=pta.WarningLog() if warnings is None else warnings, workers=workers,
        out_dir=fresh_dir(tmp_path), tile_configs=TILES, channel_variants=VARIANTS,
        write_side_effects=False, defer_cpr=defer,
    )


def plan_record(plan):
    return (plan.tag, plan.view.name, plan.view.family, plan.eligible_frame_indices, dict(plan.stats),
            [(tile.tile_tag, tile.x, tile.y, tile.out_w, tile.out_h) for tile in plan.tile_layout],
            plan.aff.M_src_to_out.tobytes(), plan.aff.M_canvas_to_src.tobytes())


def is_cpr(plan):
    return plan.view.family == "cpr"


class TruthPool:
    """Stands in for the CUDA owners with exact CPU decisions and chosen declines."""

    def __init__(self, truth, declined=()):
        self.truth, self.declined, self.calls = truth, set(declined), []

    def classify_foregrounds(self, prep, *, gen, jobs, selected_items=None):
        tags = [plan.tag for plan in prep.plans]
        named = [(tags[plan_idx], frame_idx) for plan_idx, frame_idx in jobs]
        self.calls.append((gen, named, len(prep.plans)))
        decided = {}
        for (plan_idx, _), (tag, frame_idx) in zip(jobs, named):
            if (tag, frame_idx) in self.declined:
                continue
            keys = ("full", *(tile.tile_tag for tile in prep.plans[plan_idx].tile_layout))
            if selected_items is not None:
                keys = [key for key in keys if key in selected_items[(tag, frame_idx)]]
            decided.update({(tag, frame_idx, key, 0): self.truth[(tag, frame_idx, key, 0)] for key in keys})
        remaining = [job for job, name in zip(jobs, named) if name in self.declined]
        return decided, remaining, {"backend": "truth", "gpu_jobs": len(jobs) - len(remaining)}


@pytest.mark.parametrize("kind", ["tubes", "speck", "empty"])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("start", [False, True])
def test_deferred_extraction_appends_exactly_the_inline_views_and_plans(tmp_path, kind, workers, start):
    inline_warnings, deferred_warnings = pta.WarningLog(), pta.WarningLog()
    inline = prepare(tmp_path, defer=False, kind=kind, warnings=inline_warnings, workers=workers)
    prepared = prepare(tmp_path, defer=True, kind=kind, warnings=deferred_warnings, workers=workers)
    assert inline.cpr_pending is None and prepared.cpr_pending is not None
    assert not any(is_cpr(plan) for plan in prepared.plans)
    early = list(prepared.plans)
    if start:
        prepared.cpr_pending.start()
        assert prepared.cpr_pending.future is not None
    cpr_plans = pta.complete_cpr_views(prepared)
    assert prepared.cpr_pending is None and pta.complete_cpr_views(prepared) == []
    assert all(a is b for a, b in zip(prepared.plans, early)) and prepared.plans[len(early):] == cpr_plans
    assert all(is_cpr(plan) for plan in cpr_plans) and (len(cpr_plans) > 0) == (kind == "tubes")
    assert [pta._v18_view_manifest_record(view) for view in prepared.views] == [
        pta._v18_view_manifest_record(view) for view in inline.views]
    assert [plan_record(plan) for plan in prepared.plans] == [plan_record(plan) for plan in inline.plans]
    assert prepared.mask_for_render.tobytes() == inline.mask_for_render.tobytes()
    assert prepared.foreground_preservation_stats == inline.foreground_preservation_stats
    assert deferred_warnings.summary_lines() == inline_warnings.summary_lines()


@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.13])
def test_merged_enumerations_equal_the_full_enumeration_at_every_split(tmp_path, fraction):
    prepared = prepare(tmp_path, defer=False)
    plans = prepared.plans
    full = pta.enumerate_candidates_for_volume(prepared, fraction=fraction)
    assert any(c.is_tile for c in full) and any(not c.is_tile for c in full)
    assert any(c.parent_view_tag.startswith("CPR") for c in full)
    for split in range(len(plans) + 1):
        first = pta.enumerate_candidates_for_volume(prepared, fraction=fraction, plans=plans[:split])
        second = pta.enumerate_candidates_for_volume(prepared, fraction=fraction, plans=plans[split:])
        assert pta.merge_candidate_enumerations(first, second) == full
    # Tile-less plans on either side.
    bare = [replace(plan, tile_layout=()) if index % 3 == 0 else plan for index, plan in enumerate(plans)]
    full = pta.enumerate_candidates_for_volume(prepared, fraction=fraction, plans=bare)
    for split in (1, len(bare) // 2, len(bare) - 1):
        assert pta.merge_candidate_enumerations(
            pta.enumerate_candidates_for_volume(prepared, fraction=fraction, plans=bare[:split]),
            pta.enumerate_candidates_for_volume(prepared, fraction=fraction, plans=bare[split:])) == full
    assert pta.merge_candidate_enumerations([], []) == []


def single_call_jobs(inline, truth, selected):
    probe = TruthPool(truth)
    pta.classify_original_foregrounds_for_volume(inline, workers=2, selected_candidates=selected, gpu_pool=probe)
    (call,) = probe.calls
    return call[1]


@pytest.mark.parametrize("fraction", [1.0, 0.4])
@pytest.mark.parametrize("decline", [False, True])
def test_prefetched_gpu_decisions_match_one_gpu_call_and_cpu_truth(tmp_path, fraction, decline):
    inline = prepare(tmp_path, defer=False)
    truth = pta.classify_original_foregrounds_for_volume(inline, workers=2)
    full_cands = pta.enumerate_candidates_for_volume(inline, fraction=fraction)
    selected = full_cands if fraction < 1.0 else None
    expected = pta.classify_original_foregrounds_for_volume(inline, workers=2, selected_candidates=selected)
    families = {plan.tag: plan.view.family for plan in inline.plans}
    jobs = single_call_jobs(inline, truth, selected)
    other_jobs = [job for job in jobs if families[job[0]] != "cpr"]
    cpr_jobs = [job for job in jobs if families[job[0]] == "cpr"]
    assert other_jobs and cpr_jobs and any(truth.values()) and not all(truth.values())
    declined = {other_jobs[len(other_jobs) // 2], cpr_jobs[-1]} if decline else set()
    single = TruthPool(truth, declined)
    assert pta.classify_original_foregrounds_for_volume(
        inline, workers=2, selected_candidates=selected, gpu_pool=single, generation=5) == expected

    prepared = prepare(tmp_path, defer=True)
    early = pta.enumerate_candidates_for_volume(prepared, fraction=fraction)
    pool = TruthPool(truth, declined)
    with mock.patch.object(pta, "parallel_for_indices", wraps=pta.parallel_for_indices) as cpu_jobs:
        prepared.cpr_pending.start()
        prefetch = pta.prefetch_original_foregrounds_on_gpu(
            prepared, gpu_pool=pool, generation=5, selected_candidates=early if fraction < 1.0 else None)
        assert prepared.cpr_pending is not None and len(pool.calls) == 1
        cpr_plans = pta.complete_cpr_views(prepared)
        cands = pta.merge_candidate_enumerations(early, pta.enumerate_candidates_for_volume(
            prepared, fraction=fraction, plans=cpr_plans))
        assert cands == full_cands
        actual = pta.classify_original_foregrounds_for_volume(
            prepared, workers=2, selected_candidates=cands if fraction < 1.0 else None,
            gpu_pool=pool, generation=5, gpu_prefetch=prefetch)
    assert actual == expected
    (gen0, first, plan_count), (gen1, second, _count) = pool.calls
    assert gen0 == gen1 == 5 and plan_count == len(prepared.plans) - len(cpr_plans)
    assert first == other_jobs and second == cpr_jobs
    # Only declined jobs replay on the CPU.
    assert [call.args[0] for call in cpu_jobs.call_args_list] == [len(declined)]


def test_prefetch_needs_anchor_proof_and_no_coverage_scan(tmp_path):
    prepared = prepare(tmp_path, defer=True)
    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=AssertionError("unexpected GPU call")))
    try:
        stats = prepared.foreground_preservation_stats
        known = stats.pop("guaranteed_output_foreground_transverse_slices")
        assert known > 0
        assert pta.prefetch_original_foregrounds_on_gpu(prepared, gpu_pool=pool) is None
        stats["guaranteed_output_foreground_transverse_slices"] = 0
        assert pta.prefetch_original_foregrounds_on_gpu(prepared, gpu_pool=pool) is None
        stats["guaranteed_output_foreground_transverse_slices"] = known
        prepared.semantic_foreground = True
        prepared.semantic_coverage_for_render = np.ones_like(prepared.mask_for_render)
        assert pta.prefetch_original_foregrounds_on_gpu(prepared, gpu_pool=pool) is None
        prepared.semantic_foreground, prepared.semantic_coverage_for_render = False, None
        assert pta.prefetch_original_foregrounds_on_gpu(prepared, gpu_pool=pool, selected_candidates=[]) is None
        prepared.label_enabled = False
        assert pta.prefetch_original_foregrounds_on_gpu(prepared, gpu_pool=pool) is None
    finally:
        pta.complete_cpr_views(prepared)


def test_mismatched_or_invalid_prefetch_fails(tmp_path):
    inline = prepare(tmp_path, defer=False)
    truth = pta.classify_original_foregrounds_for_volume(inline, workers=2)
    pool = TruthPool(truth)
    prefetch = pta.prefetch_original_foregrounds_on_gpu(inline, gpu_pool=pool)
    # A prefetch that covers every dense job needs no further GPU call.
    assert pta.classify_original_foregrounds_for_volume(
        inline, workers=2, gpu_pool=pool, gpu_prefetch=prefetch) == truth
    assert len(pool.calls) == 1
    equal_plans = prepare(tmp_path, defer=False)
    with pytest.raises(RuntimeError, match="prefetch does not match"):
        pta.classify_original_foregrounds_for_volume(equal_plans, workers=2, gpu_pool=pool, gpu_prefetch=prefetch)
    with pytest.raises(RuntimeError, match="prefetch does not match"):
        pta.classify_original_foregrounds_for_volume(inline, workers=2, gpu_pool=pool, gpu_prefetch=replace(
            prefetch, jobs=[*prefetch.jobs, (len(inline.plans), 0)]))
    with pytest.raises(RuntimeError, match="invalid remaining"):
        pta.classify_original_foregrounds_for_volume(inline, workers=2, gpu_pool=pool, gpu_prefetch=replace(
            prefetch, remaining=[prefetch.jobs[0], prefetch.jobs[0]]))
    with pytest.raises(RuntimeError, match="did not return complete"):
        pta.classify_original_foregrounds_for_volume(inline, workers=2, gpu_pool=pool, gpu_prefetch=replace(
            prefetch, foregrounds=dict(list(prefetch.foregrounds.items())[1:])))
    # Decisions made for a different fraction selection are incomplete for this one.
    selected = pta.enumerate_candidates_for_volume(inline, fraction=0.5)
    with pytest.raises(RuntimeError, match="did not return complete"):
        pta.classify_original_foregrounds_for_volume(
            inline, workers=2, gpu_pool=pool, gpu_prefetch=prefetch, selected_candidates=selected)


def planner_namespace(tmp_path, args, src, *, defer, gpu_pool):
    real_prepare = pta.prepare_loaded_source

    def prepare_volume(loaded, **kwargs):
        assert kwargs.pop("defer_cpr") is True
        return real_prepare(loaded, defer_cpr=defer, **kwargs)

    namespace = dict(vars(pta))
    namespace.update(
        pending_loads={0: (SimpleNamespace(result=lambda: src), pta.WarningLog())},
        _submit_load=lambda index: None, pending_specs=[SimpleNamespace(stem=src.stem, input_options=None)],
        states={}, args=args, out_dir=fresh_dir(tmp_path), tile_configs=list(TILES), channel_variants=VARIANTS,
        volume_allocator=None, topology=SimpleNamespace(cuda_device_ids=()), effective_pipeline_depth=1,
        dataset_publication_selected=True, fraction=float(args.fraction),
        background_filter_requested=float(args.background_percent) < 1.0,
        augmentation=pta.load_offline_augmentation_definition(args.augmentation),
        augmentation_definition=pta.inspect_augmentation_definition(args.augmentation),
        gpu_offline_active=gpu_pool is not None, render_pool=gpu_pool, split_active=False,
        split_method=str(args.split_method), train_split=None, volume_split_by_stem=None,
        used_augmentation_tags=set(), prepare_loaded_source=prepare_volume,
    )
    return namespace


def identity_policy(tmp_path):
    policy = tmp_path / "identity.py"
    policy.write_text("import albumentations as A\ndef build_augmentation():\n    return A.Compose([A.NoOp(p=1)])\n")
    return policy


@pytest.mark.parametrize("route", ["classify", "deferred"])
@pytest.mark.parametrize("fraction", [1.0, 0.4])
def test_parent_planner_states_match_inline_extraction_on_cpu_and_gpu_routes(tmp_path, monkeypatch, route, fraction):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    ratio, background = ("3.7", "0.6") if route == "classify" else ("4", "1")
    args = runtime_args(tmp_path, "--fraction", repr(fraction), "--augmentation", f"cpu:{identity_policy(tmp_path)}",
                        "--augmentation_execution", "offline", "--augmentation_ratio", ratio,
                        "--background_percent", background)
    truth = pta.classify_original_foregrounds_for_volume(prepare(tmp_path, defer=False, args=args), workers=2)

    starts = []
    real_start = pta.PendingCprViews.start
    monkeypatch.setattr(pta.PendingCprViews, "start", lambda self: starts.append(1) or real_start(self))

    def plan(defer, pool):
        namespace = planner_namespace(tmp_path, args, source(tmp_path, tube_mask()), defer=defer, gpu_pool=pool)
        starts.clear()
        state = parent_planner(namespace)(0, 2)
        assert state["prep"].cpr_pending is None
        # A thread only while the CUDA owners classify; inline extraction otherwise.
        assert len(starts) == int(defer and pool is not None and route == "classify")
        return state

    reference = plan(False, None)  # v25.0.3 order: inline extraction, CPU classification
    assert any(c.parent_view_tag.startswith("CPR") for c in reference["cands"])
    for defer, pool in ((True, None), (True, TruthPool(truth)), (False, TruthPool(truth))):
        state = plan(defer, pool)
        for key in ("cands", "physical", "candidates_before_fraction", "background_stats",
                    "augmentation_stats", "split_stats", "classification_deferred", "fraction"):
            assert state[key] == reference[key], key
        assert state["warnings"].summary_lines() == reference["warnings"].summary_lines()
        assert [plan_record(p) for p in state["prep"].plans] == [plan_record(p) for p in reference["prep"].plans]
        if pool is not None:
            # Deferred classification never classifies; otherwise deferred CPR prefetches.
            expected_calls = 0 if route == "deferred" else (2 if defer else 1)
            assert len(pool.calls) == expected_calls
            if expected_calls == 2:
                assert pool.calls[0][2] < len(state["prep"].plans)


def release_render_state():
    tree = ast.parse(Path(pta.__file__).read_text(encoding="utf-8"))
    selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                and node.name in {"_release_source_volume", "_release_render_state"}]
    namespace = dict(np=np, SourceVolume=pta.SourceVolume, PreparedVolume=pta.PreparedVolume,
                     VolumeRenderProgress=pta_workers.VolumeRenderProgress,
                     RenderPhaseHandle=pta_workers.RenderPhaseHandle, progress_by_gen={})
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual render state cleanup>", "exec"), namespace)
    return namespace["_release_render_state"]


def blocking_extraction(monkeypatch, events):
    """Extraction that runs until cancelled (or 20 s), recording each step."""

    def extract(mask, *, cancel=None, **kwargs):
        events.append("started")
        try:
            if cancel is not None and cancel.wait(20):
                events.append("cancelled")
                raise pta_cpr.CprExtractionCancelled("cancelled")
            return []
        finally:
            events.append("finished")

    monkeypatch.setattr(pta_cpr, "extract_cpr_geometries", extract)


def test_classification_error_stops_and_joins_extraction_before_blocks_are_released(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    events = []
    blocking_extraction(monkeypatch, events)
    args = runtime_args(tmp_path, "--augmentation", f"cpu:{identity_policy(tmp_path)}",
                        "--augmentation_execution", "offline", "--augmentation_ratio", "3.7")
    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=RuntimeError("CUDA kernel failed")))
    namespace = planner_namespace(tmp_path, args, source(tmp_path, tube_mask()), defer=True, gpu_pool=pool)
    with pytest.raises(RuntimeError, match="CUDA kernel failed"):
        parent_planner(namespace)(0, 2)
    state = namespace["states"][0]
    pending = state["prep"].cpr_pending
    assert pending is not None and pending.future is not None
    started = threading.Event()
    for _ in range(200):  # the thread starts just before the failing prefetch
        if events:
            started.set()
            break
        started.wait(0.05)
    assert events == ["started"]
    block = mock.Mock()
    block.release.side_effect = lambda: events.append("released")
    state["prep"].shm_blocks.append(block)
    release_render_state()(state)
    assert events == ["started", "cancelled", "finished", "released"]
    assert isinstance(pending.future.exception(), pta_cpr.CprExtractionCancelled)
    assert state["prep"] is None


def test_deferred_extraction_starts_once_and_abandon_joins_only_a_started_thread(tmp_path, monkeypatch):
    events = []
    blocking_extraction(monkeypatch, events)
    idle = prepare(tmp_path, defer=True)
    idle.cpr_pending.abandon()
    assert events == [] and idle.cpr_pending.future is None
    pending = prepare(tmp_path, defer=True).cpr_pending
    pending.start()
    future = pending.future
    pending.start()
    assert pending.future is future
    pending.abandon()
    assert events == ["started", "cancelled", "finished"]
    assert isinstance(future.exception(), pta_cpr.CprExtractionCancelled)


@pytest.mark.parametrize("defer,gpu", [(False, False), (True, False), (True, True)])
def test_cpr_failure_still_fails_the_volume(tmp_path, monkeypatch, defer, gpu):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    args = runtime_args(tmp_path, "--augmentation", f"cpu:{identity_policy(tmp_path)}",
                        "--augmentation_execution", "offline", "--augmentation_ratio", "3.7")
    pool = TruthPool(pta.classify_original_foregrounds_for_volume(
        prepare(tmp_path, defer=False, args=args), workers=2)) if gpu else None
    monkeypatch.setattr(pta_cpr, "extract_cpr_geometries", mock.Mock(side_effect=MemoryError("synthetic CPR failure")))
    namespace = planner_namespace(tmp_path, args, source(tmp_path, tube_mask()), defer=defer, gpu_pool=pool)
    with pytest.raises(MemoryError, match="synthetic CPR failure"):
        parent_planner(namespace)(0, 2)
    assert pool is None or len(pool.calls) == 1  # the prefetch ran beside the failing extraction
    if defer:
        state = namespace["states"][0]
        release_render_state()(state)
        assert state["prep"] is None


def test_a_failed_extraction_reports_before_a_later_gpu_prefetch_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    args = runtime_args(tmp_path, "--augmentation", f"cpu:{identity_policy(tmp_path)}",
                        "--augmentation_execution", "offline", "--augmentation_ratio", "3.7")
    monkeypatch.setattr(pta_cpr, "extract_cpr_geometries", mock.Mock(side_effect=MemoryError("synthetic CPR failure")))
    started = []
    real_start = pta.PendingCprViews.start

    def start(self):
        real_start(self)
        started.append(self)

    monkeypatch.setattr(pta.PendingCprViews, "start", start)

    def classify_after_the_extraction_failed(*args, **kwargs):
        assert isinstance(started[0].future.exception(timeout=20), MemoryError)
        raise RuntimeError("CUDA kernel failed")

    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=classify_after_the_extraction_failed))
    namespace = planner_namespace(tmp_path, args, source(tmp_path, tube_mask()), defer=True, gpu_pool=pool)
    with pytest.raises(MemoryError, match="synthetic CPR failure") as raised:
        parent_planner(namespace)(0, 2)
    assert isinstance(raised.value.__context__, RuntimeError)  # the GPU error stays attached
    state = namespace["states"][0]
    release_render_state()(state)
    assert state["prep"] is None


@pytest.mark.parametrize("workers", [1, 4])
def test_cancel_stops_extraction_at_the_next_block_without_a_component_warning(workers):
    cancel = threading.Event()
    cancel.set()
    warnings = []
    with pytest.raises(pta_cpr.CprExtractionCancelled):
        pta_cpr.extract_cpr_geometries(tube_mask(), warn=warnings.append, workers=workers, cancel=cancel)
    assert warnings == []


@pytest.mark.parametrize("workers", [1, 4])
def test_cancel_during_refinement_stops_before_later_branches(monkeypatch, workers):
    mask = tube_mask()
    real = pta_cpr._refine_source_centerline
    calls = []

    def refine(*args):
        calls.append(len(calls))
        if cancel is not None:
            cancel.set()
        return real(*args)

    monkeypatch.setattr(pta_cpr, "_refine_source_centerline", refine)
    cancel = None
    complete = pta_cpr.extract_cpr_geometries(mask, workers=workers, cancel=threading.Event())
    total, calls[:] = len(calls), []
    assert len(complete) >= 2 and total >= 2
    cancel = threading.Event()
    warnings = []
    with pytest.raises(pta_cpr.CprExtractionCancelled):
        pta_cpr.extract_cpr_geometries(mask, warn=warnings.append, workers=workers, cancel=cancel)
    assert warnings == [] and 0 < len(calls) < total


@pytest.fixture
def cpu_topology(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    topology = SimpleNamespace(cuda_device_ids=(), gpu_cpu_sets=(), worker_cpu_order=(),
                               allowed_cpus=(0, 1, 2, 3), summary="CPU CPR overlap test")
    monkeypatch.setattr(pta, "discover_topology", lambda **kwargs: topology)
    monkeypatch.setattr(pta, "available_memory_budget_bytes", lambda: 16 * pta.GIB)


def write_input(root):
    root.mkdir(parents=True)
    mask = tube_mask()
    volume = np.random.default_rng(5).integers(0, 256, mask.shape, dtype=np.uint8)
    for index, plane in enumerate(volume):
        assert pta.cv2.imwrite(str(root / f"tube_{index:04d}.png"), plane)
    pta.write_nrrd(mask, root / "tube.nrrd")
    return root


@pytest.mark.parametrize("fraction", ["1", "0.45"])
def test_published_outputs_match_inline_extraction(tmp_path, monkeypatch, cpu_topology, fraction):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    root = write_input(tmp_path / "input")
    output = tmp_path / "output"
    arguments = ["--input", str(root), "--output", str(output), "--imgsz", "32", "--enable_cpr",
                 "--enable_cartesian", "transverse,sagittal,coronal", "--enable_tile", "16:12",
                 "--fraction", fraction, "--background_percent", "0.6",
                 "--augmentation", f"cpu:{identity_policy(tmp_path)}", "--augmentation_execution", "offline",
                 "--augmentation_ratio", "1.5", "--save", "images", "labels", "summary",
                 "--worker_backend", "thread", "--workers", "3", "--frame_workers", "2",
                 "--jpeg_decode_backend", "opencv", "--no-topology_aware"]
    real_prepare = pta.prepare_loaded_source
    results = {}
    for defer in (False, True):
        def prepare_volume(loaded, **kwargs):
            kwargs["defer_cpr"] = defer
            return real_prepare(loaded, **kwargs)

        monkeypatch.setattr(pta, "prepare_loaded_source", prepare_volume)
        pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        results[defer] = (artifacts(output, "images"), artifacts(output, "labels"),
                                manifest["volumes"], (output / "summary.txt").read_text(encoding="utf-8"))
        shutil.rmtree(output)
    inline, overlapped = results[False], results[True]
    assert inline[0] and any(name.startswith("tube_CPR") or "CPR" in name for name in inline[0])
    assert overlapped[0] == inline[0] and overlapped[1] == inline[1]
    assert overlapped[2] == inline[2]
    strip = lambda text: [line for line in text.splitlines() if "second" not in line and "time" not in line]
    assert strip(overlapped[3]) == strip(inline[3])


class InProcessOwnerPool:
    """One real CUDA owner in this process; every call unpickles its own plan payload."""

    def __init__(self, runtime):
        self.runtime, self.calls = runtime, []

    def classify_foregrounds(self, prep, *, gen, jobs, selected_items=None):
        from XTA.pta_gpu_classification import build_classification_tasks, classification_stats, classify_task

        plans = pickle.loads(pickle.dumps(list(prep.plans), protocol=pickle.HIGHEST_PROTOCOL))
        self.calls.append([(plans[plan_idx].tag, frame_idx) for plan_idx, frame_idx in jobs])
        truth, declined, stats = {}, set(), classification_stats()
        for task in build_classification_tasks(plans, jobs, selected_items=selected_items,
                                               semantic=bool(prep.semantic_foreground)):
            result, remaining, measured = classify_task(self.runtime, prep.mask_for_render, plans, task,
                                                        identity=f"gen{gen}")
            truth.update(result)
            declined.update(tuple(job) for job in remaining)
            for name in stats:
                if name != "backend":
                    stats[name] += int(measured[name])
        return truth, [job for job in jobs if tuple(job) in declined], stats


@pytest.mark.skipif(os.environ.get("XTA_TEST_PTA_CLASSIFICATION_GPU") != "1",
                    reason="requires explicit GPU qualification and an owned GPU lock")
@pytest.mark.parametrize("fraction", [1.0, 0.4])
def test_cuda_owner_prefetch_then_cpr_jobs_match_one_call(tmp_path, monkeypatch, fraction):
    import torch
    from XTA import pta_cuda_masks

    owner = json.loads(Path(os.environ["XTA_GPU_LOCK_PATH"]).read_text())
    assert owner["pid"] == os.getpid() and owner["token"] == os.environ["XTA_GPU_LOCK_TOKEN"]
    assert torch.cuda.is_available()
    runtime = dict(torch=torch, device_id=0, azimuthal_renderer=SimpleNamespace(_stream=torch.cuda.Stream(device=0)))
    try:
        inline = prepare(tmp_path, defer=False)
        selected = pta.enumerate_candidates_for_volume(inline, fraction=fraction) if fraction < 1.0 else None
        cpu = pta.classify_original_foregrounds_for_volume(inline, workers=2, selected_candidates=selected)
        single = InProcessOwnerPool(runtime)
        one_call = pta.classify_original_foregrounds_for_volume(
            inline, workers=2, selected_candidates=selected, gpu_pool=single, generation=0)
        pta_cuda_masks.retire_gpu_categorical_volume(runtime)

        prepared = prepare(tmp_path, defer=True)
        early = pta.enumerate_candidates_for_volume(prepared, fraction=fraction)
        pool = InProcessOwnerPool(runtime)
        prepared.cpr_pending.start()
        prefetch = pta.prefetch_original_foregrounds_on_gpu(
            prepared, gpu_pool=pool, generation=1, selected_candidates=early if fraction < 1.0 else None)
        cpr_plans = pta.complete_cpr_views(prepared)
        cands = pta.merge_candidate_enumerations(early, pta.enumerate_candidates_for_volume(
            prepared, fraction=fraction, plans=cpr_plans))
        overlapped = pta.classify_original_foregrounds_for_volume(
            prepared, workers=2, selected_candidates=cands if fraction < 1.0 else None,
            gpu_pool=pool, generation=1, gpu_prefetch=prefetch)
    finally:
        pta_cuda_masks.retire_gpu_categorical_volume(runtime)
    assert prefetch.remaining == [] and overlapped == one_call == cpu
    (first, second), (whole,) = pool.calls, single.calls
    assert sorted(first + second) == sorted(whole) and second
    assert all(tag.startswith("CPR") for tag, _frame in second)
    assert not any(tag.startswith("CPR") for tag, _frame in first)
