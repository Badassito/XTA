"""Failure handling of the cross-cutoff review store section of tools/slicer/xta_slicerrc.py.

Store operations are all or nothing: a click, sync, update, undo or redo that raises part-way leaves the layers,
boxes, maps, counts, painted set, stale flag, undo stacks and edit log as they were, and a rollback that fails too
marks the store diverged. A store made before the file was executed again is refused until adopted, and adopting it
keeps the identity of its journal entries. The edit log: damage before the tail is refused without touching the
file, a zero-filled tail is cut, a tail is cut only after the replay, one session at a time writes a log, the undo
caps it recorded replay exactly, a log that cannot be written still replays read-only, a failed start leaves no
file, the summary checks every CRC, and a store dropped without close() is freed by reference counting. Bit planes
are set in bounded slabs, within the builder's memory estimate.
"""
import gc
import os
import stat
import tracemalloc
import weakref

import numpy as np
import pytest

from tests import slicer_cutoff_store_support as sup

ENGINE = sup.default_engine()
ACCEPT_ALL = dict(confirm_huge_displayed=lambda info: "remove", confirm_huge_looser=lambda info: "remove_all")


@pytest.fixture(scope="module")
def S():
    module = sup.load_store_module(sup.MODULE_NAME + "_robustness")
    yield module
    sup.unload_store_module(module)


@pytest.fixture(scope="module")
def kit(S):
    return sup.StoreKit(S)


def identity(store):
    return dict(sweep_id="sweep-scene", grid_id="low_quality/scene", review_manifest_sha256="0" * 64,
                cutoffs_cent=list(range(store.n_cutoffs)), shape_tyx=list(store.shape))


def body_click(S, store):
    """Remove Islands on the hand scene's body at k0 = 2: seven layer writes (A, B, D in step 1; A, B in step 2;
    two Global follow writes, after the maps were updated)."""
    return S._xta_store_remove_islands(store, 2, np.array([[8, 14, 10]]), engine=ENGINE, **ACCEPT_ALL)


def island_click(S, store):
    return S._xta_store_remove_islands(store, 2, np.array([[8, 14, 30]]), engine=ENGINE, **ACCEPT_ALL)


class FailAt:
    """The store's nth call of method `name` raises MemoryError (an instance attribute shadows the method)."""

    def __init__(self, store, name, n):
        self.store, self.name, self.n, self.calls = store, name, n, 0
        real = getattr(store, name)

        def failing(*args, **kwargs):
            self.calls += 1
            if self.calls == self.n:
                raise MemoryError(f"simulated failure in {name}")
            return real(*args, **kwargs)
        setattr(store, name, failing)

    def remove(self):
        delattr(self.store, self.name)


def stacks(store):
    return ([(e.kind, e.k0) for e in store.journal], [(e.kind, e.k0) for e in store.redo_stack], store.dropped_entries)


def unchanged(kit, store, snap, depth, log=None, records=None):
    assert kit.same_state(store, snap) and kit.maps_consistent(store) and kit.counts_consistent(store)
    assert stacks(store) == depth and store.diverged is None
    if log is not None:
        assert log.records == records and log.error is None and store.edit_log is log


# --------------------------------------------------------------------------------------------
# All or nothing


@pytest.mark.parametrize("fail_at", (1, 4, 7))
def test_failed_click_changes_nothing(S, kit, tmp_path, fail_at):
    """A click that fails at its first, a middle or its last layer write (the Global follow, after the maps
    changed) leaves everything as it was; the same click then goes through, and the log replays to it."""
    store = sup.g_scene(kit)
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity(store))
    sup.random_syncs(kit, store, np.random.default_rng(3), 4)            # a painted set and an undo history
    reference = kit.copy(store)
    snap, depth, records = kit.snapshot(store), stacks(store), log.records
    fail = FailAt(store, "_write_masked", fail_at)
    with pytest.raises(MemoryError):
        body_click(S, store)
    fail.remove()
    assert fail.calls == fail_at
    unchanged(kit, store, snap, depth, log, records)
    assert body_click(S, store)["status"] == body_click(S, reference)["status"] == "applied"
    assert kit.same_state(store, kit.snapshot(reference)) and log.records == records + 1
    log.close()
    fresh = sup.g_scene(kit)
    S.XTACutoffEditLog.resume(tmp_path / "edits.xtalog", fresh, identity(fresh)).close()
    assert kit.same_state(fresh, kit.snapshot(store)) and stacks(fresh) == stacks(store)


@pytest.mark.parametrize("op", ("click", "paint_growth", "update_global", "update_nonglobal"))
def test_operation_failing_before_its_push_changes_nothing(S, kit, op):
    """Every write of an operation is undone when it raises after writing everything but before its entry is
    pushed: layer values, a grown layer box, a replaced (Update Global) layer, the maps, counts and stale flag."""
    store = sup.g_scene(kit)
    if op.startswith("update"):
        island_click(S, store)                                          # a stale Global with work to do
        if op == "update_nonglobal":
            S._xta_store_update_global(store, tie_breaker="raster_first")
    snap, depth = kit.snapshot(store), stacks(store)
    run = dict(click=lambda: body_click(S, store),
               paint_growth=lambda: S._xta_store_sync_paint(store, "B", 1, (np.zeros(3, np.int64),
                                                                           np.ones((2, 2, 2), bool))),
               update_global=lambda: S._xta_store_update_global(store, tie_breaker="raster_first"),
               update_nonglobal=lambda: S._xta_store_update_nonglobal(store))[op]
    fail = FailAt(store, "_push", 1)
    with pytest.raises(MemoryError):
        run()
    fail.remove()
    unchanged(kit, store, snap, depth)
    if op == "paint_growth":
        assert tuple(store.layers["B"].origin) == (0, 4, 1)
    out = run()                                                          # and then it goes through
    assert len(store.journal) == len(depth[0]) + 1, out


def test_failed_undo_and_redo_change_nothing(S, kit, tmp_path):
    store = sup.g_scene(kit, global_bits=True)
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity(store))
    before_click = kit.snapshot(store)
    body_click(S, store)
    after_click = kit.snapshot(store)
    for name, step, target in (("_revert", store.undo, before_click), ("_reapply", store.redo, after_click)):
        snap, depth, records = kit.snapshot(store), stacks(store), log.records
        fail = FailAt(store, name, 3)                                    # the third of the click's 7 changes
        with pytest.raises(MemoryError):
            step()
        fail.remove()
        unchanged(kit, store, snap, depth, log, records)
        assert step() and kit.same_state(store, target) and log.records == records + 1
    assert store.undo()
    spilled = store.redo_stack[-1]
    store._spill(spilled)
    os.remove(spilled.spill_path)                                        # the spill file is gone: redo refuses
    snap, depth, records = kit.snapshot(store), stacks(store), log.records
    with pytest.raises(FileNotFoundError):
        store.redo()
    unchanged(kit, store, snap, depth, log, records)
    assert store.redo_stack[-1] is spilled                               # still there, not lost
    log.close()


def test_failed_rollback_marks_the_store_diverged(S, kit, tmp_path):
    store = sup.g_scene(kit)
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity(store))
    errors = []
    log.on_error = errors.append
    body_fail, rollback_fail = FailAt(store, "_write_masked", 4), FailAt(store, "_revert", 1)
    with pytest.warns(S.XTACutoffEditLogWarning, match="could not be rolled back"), pytest.raises(MemoryError):
        body_click(S, store)
    body_fail.remove()
    rollback_fail.remove()
    assert store.diverged and "rebuild it from the review" in store.diverged
    assert log.error == store.diverged and errors == [store.diverged] and log.closed and store.edit_log is None
    for op in (store.undo, store.redo, lambda: island_click(S, store), lambda: store.begin_entry("custom"),
               lambda: S._xta_store_sync_erase(store, "A", 2, np.array([[8, 14, 10]])),
               lambda: S._xta_store_update_global(store, tie_breaker="raster_first"), lambda: store.set_undo_bytes(1)):
        with pytest.raises(RuntimeError, match="could not be rolled back"):
            op()
    store.close()


def test_entry_builder_write_that_raises_changes_nothing(S, kit):
    """A write with one value too few, outside the layer box: neither the values nor the box change."""
    store = sup.g_scene(kit)
    snap = kit.snapshot(store)
    assert store.layers["A"].origin[1] > 0                              # (0, 0, 0) lies outside A's box
    builder = store.begin_entry("custom")
    with pytest.raises(ValueError, match="values of shape"):
        builder.write("A", (0, 0, 0), np.ones((1, 1, 3), bool), np.array([1, 1], np.uint8))
    assert kit.same_state(store, snap) and tuple(store.layers["A"].origin) == (0, 3, 1)
    with pytest.raises(ValueError, match="above the number of cutoffs"):
        builder.replace_layer("A", np.full((2, 2, 2), 99, np.uint8), (0, 0, 0))
    assert kit.same_state(store, snap) and kit.counts_consistent(store)
    assert builder.commit() is None and not store.journal


# --------------------------------------------------------------------------------------------
# A second execution of the file


def test_reexec_without_adopt_is_refused_and_records_survive_adopt(tmp_path):
    S = sup.load_store_module(sup.MODULE_NAME + "_reexec_guard")
    try:
        k1 = sup.StoreKit(S)
        s = sup.g_scene(k1)
        log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", s, identity(s))
        S._xta_store_sync_paint(s, "B", 1, (np.array([2, 2, 30]), np.ones((3, 3, 3), bool)))
        island_click(S, s)
        rec = S._xta_store_update_global(s, tie_breaker="raster_first")
        assert rec.entry is not None and s.undo() and s.redo() and s.journal[-1] is rec.entry
        snap, depth, records = k1.snapshot(s), stacks(s), log.records
        sup.reexec_section(S)
        k2 = sup.StoreKit(S)
        for op in (s.undo, s.redo, lambda: s.begin_entry("custom"), lambda: body_click(S, s),
                   lambda: S._xta_store_sync_erase(s, "A", 2, np.array([[8, 14, 10]])),
                   lambda: S._xta_store_update_global(s, tie_breaker="raster_first"),
                   lambda: S._xta_store_update_nonglobal(s), lambda: S._xta_store_undo_update(s, rec),
                   lambda: s.set_undo_bytes(1)):
            with pytest.raises(RuntimeError, match="executed again"):
                op()
        assert k2.same_state(s, snap) and stacks(s) == depth and log.records == records
        assert S._xta_store_adopt(s) is s and type(s.edit_log) is S.XTACutoffEditLog
        assert s.journal[-1] is rec.entry and type(rec.entry) is S.XTACutoffJournalEntry
        S._xta_store_undo_update(s, rec)                 # a record taken before the re-exec still names its entry
        assert log.records == records + 1 and s.redo_stack[-1] is rec.entry
        assert k2.maps_consistent(s) and k2.counts_consistent(s)
        s.close()
    finally:
        sup.unload_store_module(S)


# --------------------------------------------------------------------------------------------
# The edit log


def log_with_steps(S, kit, path, steps=8, seed=5, **store_kw):
    """A log of `steps` records written on the hand scene (syncs, clicks, undo, redo); returns (live store, file
    size after each record, the header included)."""
    store = kit.build_store(*hand_specs(kit), **store_kw)
    log = S.XTACutoffEditLog.start(path, store, identity(store))
    rng = np.random.default_rng(seed)
    sizes = [path.stat().st_size]
    while log.records < steps:
        n, u = log.records, rng.random()
        if u < 0.15 and store.journal:
            store.undo()
        elif u < 0.25 and store.redo_stack:
            store.redo()
        elif u < 0.4:
            island_click(S, store) if rng.random() < 0.5 else body_click(S, store)
        else:
            sup.random_syncs(kit, store, rng, 1)
        if log.records > n:
            sizes.append(path.stat().st_size)
    log.close()
    return store, sizes


def hand_specs(kit):
    store = sup.g_scene(kit)
    specs = [dict(lid=lid, role=l.role, additive=l.additive if l.role == "derived" else None, visible=l.visible,
                  data=kit.full_data(store, lid)) for lid, l in store.layers.items()]
    return store.shape, store.n_cutoffs, specs


def test_log_damage_before_the_tail_is_refused_and_a_zero_tail_is_cut(S, kit, tmp_path):
    path = tmp_path / "edits.xtalog"
    live, sizes = log_with_steps(S, kit, path)
    full = path.read_bytes()
    frame = S._XTA_STORE_LOG_FRAME

    def fresh():
        return kit.build_store(*hand_specs(kit))
    # a bit flip in a middle record's length field: refused, nothing cut (the frame has its own CRC)
    damaged = bytearray(full)
    damaged[sizes[3] + 5] ^= 0x01
    path.write_bytes(bytes(damaged))
    store = fresh()
    with pytest.raises(S.XTACutoffEditLogCorrupt, match="frame at byte"):
        S.XTACutoffEditLog.resume(path, store, identity(store))
    assert path.read_bytes() == bytes(damaged) and not store.journal and store.edit_log is None
    summary = S._xta_store_edit_log_summary(path)
    assert "damaged" in summary["error"] and summary["records"] == 3 and summary["torn_bytes"] == 0
    # zeros where an append was cut by a power loss: cut off, every record before them replayed
    for tail in (full + b"\0" * 4096,                                          # zeros after the last record
                 full[:sizes[-2] + frame] + b"\0" * (len(full) - sizes[-2] - frame),   # a zeroed last body
                 full[:sizes[-2]] + b"\0" * (len(full) - sizes[-2])):          # a zeroed last frame and body
        path.write_bytes(tail)
        store = fresh()
        records = len(sizes) - 1 if len(tail) > len(full) else len(sizes) - 2
        assert S._xta_store_edit_log_summary(path)["records"] == records
        with pytest.warns(S.XTACutoffEditLogWarning, match="cut off"):
            log = S.XTACutoffEditLog.resume(path, store, identity(store))
        assert log.replayed == records and path.read_bytes() == full[:sizes[records]]
        log.close()
    # the torn tail is cut only after the replay: a cancelled replay leaves the file as it was
    path.write_bytes(full[:-3])
    with pytest.warns(S.XTACutoffEditLogWarning), pytest.raises(S.XTACutoffBuildCancelled):
        S.XTACutoffEditLog.resume(path, fresh(), identity(live), cancel=lambda: True)
    assert path.read_bytes() == full[:-3]
    # the summary checks every CRC: a damaged last record is a torn tail, not an edit
    last = bytearray(full)
    last[-2] ^= 0x5A
    path.write_bytes(bytes(last))
    summary = S._xta_store_edit_log_summary(path)
    assert summary["records"] == len(sizes) - 2 and summary["torn_bytes"] == len(full) - sizes[-2]
    assert summary["error"] is None


def test_one_session_writes_a_log(S, kit, tmp_path):
    path = tmp_path / "edits.xtalog"
    live, _ = log_with_steps(S, kit, path, steps=5)

    def fresh():
        return kit.build_store(*hand_specs(kit))
    first = fresh()
    log = S.XTACutoffEditLog.resume(path, first, identity(first))
    before = path.read_bytes()
    second = fresh()
    with pytest.raises(S.XTACutoffEditLogInUse):
        S.XTACutoffEditLog.resume(path, second, identity(second))
    assert not second.journal and second.edit_log is None
    with pytest.raises(S.XTACutoffEditLogInUse):
        S.XTACutoffEditLog.start(path, fresh(), identity(live), overwrite=True)
    with pytest.raises(FileExistsError):
        S.XTACutoffEditLog.start(path, fresh(), identity(live))
    assert path.read_bytes() == before
    view = fresh()
    viewer = S.XTACutoffEditLog.resume(path, view, identity(view), read_only=True)   # viewing is allowed
    assert viewer.closed and view.edit_log is None and kit.same_state(view, kit.snapshot(first))
    sup.random_syncs(kit, first, np.random.default_rng(8), 3)
    assert log.records > 5
    log.close()
    third = fresh()
    S.XTACutoffEditLog.resume(path, third, identity(third)).close()                  # free again after close()
    assert kit.same_state(third, kit.snapshot(first)) and stacks(third) == stacks(first)


def test_log_replays_its_undo_caps(S, kit, tmp_path):
    """The replay drops the undo entries the session dropped, under the caps the log recorded (header and
    set_undo_bytes); the resumed session's own cap then applies, logged."""
    path = tmp_path / "edits.xtalog"
    live, _ = log_with_steps(S, kit, path, steps=10, undo_bytes=1500)
    assert live.dropped_entries > 0 and live.edit_log is None
    second = kit.build_store(*hand_specs(kit))
    log = S.XTACutoffEditLog.resume(path, second, identity(second))      # default cap: 2 GiB
    assert stacks(second) == stacks(live) and kit.same_state(second, kit.snapshot(live))
    assert second.undo_bytes == 2 * S._XTA_STORE_GIB and log.records == 11
    second.set_undo_bytes(600)                                           # drops more now, logged
    second.set_undo_bytes(600)                                           # unchanged: nothing logged
    assert log.records == 12 and second.dropped_entries >= live.dropped_entries
    assert second.journal_nbytes() <= 600 or len(second.journal) == 1
    sup.random_syncs(kit, second, np.random.default_rng(4), 3)
    log.close()
    summary = S._xta_store_edit_log_summary(path)
    assert summary["undo_caps"] == 2 and summary["records"] == log.records and summary["error"] is None
    third = kit.build_store(*hand_specs(kit), undo_bytes=600)
    S.XTACutoffEditLog.resume(path, third, identity(third)).close()
    assert stacks(third) == stacks(second) and kit.same_state(third, kit.snapshot(second))
    assert S._xta_store_edit_log_summary(path)["undo_caps"] == 2      # the same cap: no new record


def test_log_that_cannot_be_written(S, kit, tmp_path):
    path = tmp_path / "edits.xtalog"
    live, _ = log_with_steps(S, kit, path, steps=4)
    if os.name != "nt" and os.geteuid() == 0:
        pytest.skip("root writes read-only files")
    os.chmod(path, stat.S_IREAD)
    try:
        store = kit.build_store(*hand_specs(kit))
        with pytest.raises(S.XTACutoffEditLogUnavailable):
            S.XTACutoffEditLog.resume(path, store, identity(store))
        assert not store.journal
        view = S.XTACutoffEditLog.resume(path, store, identity(store), read_only=True)
        assert view.closed and kit.same_state(store, kit.snapshot(live)) and stacks(store) == stacks(live)
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    with pytest.raises(S.XTACutoffEditLogUnavailable):
        S.XTACutoffEditLog.start(blocker / "sub" / "edits.xtalog", sup.g_scene(kit), identity(live))


def test_failed_start_leaves_no_file(S, kit, tmp_path, monkeypatch):
    path = tmp_path / "log" / "edits.xtalog"
    store = sup.g_scene(kit)
    with pytest.raises(TypeError):
        S.XTACutoffEditLog.start(path, store, dict(identity(store), cutoffs_cent={1, 2}))   # a set is no JSON
    assert not path.exists()

    def broken(self, kind, pieces):
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(S.XTACutoffEditLog, "_append", broken)
        with pytest.raises(OSError, match="disk full"):
            S.XTACutoffEditLog.start(path, store, identity(store))
    assert not path.exists() and store.edit_log is None
    S.XTACutoffEditLog.start(path, store, identity(store)).close()


def test_store_dropped_without_close_is_freed(S, kit, tmp_path):
    spill = tmp_path / "spill"
    spill.mkdir()
    store = kit.build_store(*hand_specs(kit), spill_dir=str(spill))
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity(store))
    body_click(S, store)
    island_click(S, store)
    for e in store.journal:
        store._spill(e)
    assert len(os.listdir(spill)) == 2 and log.store is store
    ref = weakref.ref(store)
    gc.disable()
    try:
        del store
        assert ref() is None and not os.listdir(spill) and log.store is None   # reference counting alone
    finally:
        gc.enable()
    log.close()


def test_failures_are_reported(S, kit, tmp_path, monkeypatch):
    store = sup.g_scene(kit)
    log = S.XTACutoffEditLog.start(tmp_path / "edits.xtalog", store, identity(store))

    def bad_callback(message):
        raise ZeroDivisionError("callback")
    log.on_error = bad_callback

    def broken(self, kind, pieces):
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(S.XTACutoffEditLog, "_append", broken)
        with pytest.warns(S.XTACutoffEditLogWarning) as caught:
            st = S._xta_store_sync_paint(store, "B", 1, np.array([[2, 2, 30]]))
    messages = [str(w.message) for w in caught]
    assert st["status"] == "applied" and any("OSError: disk full" in t for t in messages)
    assert any("on_error callback failed (ZeroDivisionError" in t for t in messages)
    spill = tmp_path / "spill"
    spill.mkdir()
    other = kit.build_store(*hand_specs(kit), spill_dir=str(spill))
    island_click(S, other)
    other._spill(other.journal[-1])

    def refuse(path):
        raise PermissionError("held by a scanner")
    with monkeypatch.context() as m:
        m.setattr(S._xta_store_os, "remove", refuse)
        with pytest.warns(RuntimeWarning, match="Could not delete the undo spill file"):
            other.close()
    for name in os.listdir(spill):
        os.remove(spill / name)


# --------------------------------------------------------------------------------------------
# Bit planes in bounded slabs


def test_set_bit_in_slabs_equals_one_pass(S):
    rng = np.random.default_rng(11)
    for K in (11, 30, 64, 101):
        dtype, words = S._xta_store_bits_layout(K)
        for shape in ((40, 130, 210), (1, 1500, 1000), (3_000_001,), ()):  # slabs, one big row, a column, a scalar
            bits = rng.integers(0, 2**15, shape + ((2,) if words == 2 else ()), dtype=np.uint64).astype(dtype)
            want = bits.copy()
            for k in (0, K // 2, K - 1):
                mask = rng.random(shape) < 0.4
                if words == 2:
                    want[..., k // 64] |= mask.astype(np.uint64) << np.uint64(k % 64)
                else:
                    want |= mask.astype(dtype) << dtype.type(k)
                S._xta_store_set_bit(bits, k, mask, words)
            assert np.array_equal(bits, want), (K, shape)


def test_bit_plane_reencode_memory_within_the_estimate(S):
    """A layer whose masks stop nesting is re-encoded while the store is built: its transient memory above the
    level map stays within the estimate's per-worker term (box x (bit-plane bytes + 1) + two pieces of
    _XTA_STORE_SET_BIT_CHUNK values; the one-pass shift took twice the bit planes more). A thin box (one z plane)
    is split as well."""
    K, shape = 11, (1, 2000, 1000)
    n = int(np.prod(shape))
    layer = S._XTAStoreBuildLayer("x", (np.zeros(3, np.int64), np.array(shape, np.int64)), K)
    layer.level = (np.arange(n, dtype=np.int64) % (K + 1)).astype(np.uint8).reshape(shape)
    layer.level[0, 0, 0] = 0
    sub = np.zeros((1, 2, 1000), bool)
    sub[0, 0, 0] = True                                                  # set at cutoff 3 but not at 2: no nesting
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        layer.add(3, (0, 0, 0), sub)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    size = S._xta_store_bits_itemsize(K)
    assert layer.bits is not None and layer.bits_from_cutoff == 3
    assert peak - base <= n * (size + 1) + 2 * S._XTA_STORE_SET_BIT_CHUNK * size, (peak - base) / n
