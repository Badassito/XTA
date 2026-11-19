"""The cross-cutoff store built from a published review, its edit log and its export (tools/slicer/xta_slicerrc.py).

Builder: on the published-review fixture (tests/test_slicer_review_folder.publish, tests/test_sweep_publication.scalar)
every (logical layer, cutoff) mask is reproduced; nested layers become level maps and non-nested ones bit planes;
legacy groups that mix cutoffs, empty and absent rows; damaged gzip and headers that disagree with the manifest are
refused; cancel and progress; the manifest-only memory pre-check against the built store.
Edit log: write and replay equal the live store and its undo / redo stacks after random edit / undo / redo / sync /
update sequences; a torn tail is cut off; CRC damage and a header that does not match are refused; archive; summary;
one record per push, undo and redo; a failed append never breaks the edit; re-exec of the file keeps the log.
Export: the store's bundles equal today's ExportXTASweepEdits (arrays, and JSON without created_utc / edits_sha256).
"""
import copy
import gzip
import json
import os
import shutil
import warnings

import numpy as np
import pytest

from tests import slicer_cutoff_store_support as sup
from tests.test_slicer_review_folder import LOW, publish
from tests.test_sweep_publication import scalar
from XTA.grouped_seg_nrrd import GroupedSegment, write_grouped_seg_nrrd
from XTA.reconciliation_io import ReferenceGeometry
from XTA.sweep_publication import publish_sweep_review

ENGINE = sup.default_engine()
ACCEPT_ALL = dict(confirm_huge_displayed=lambda info: "remove", confirm_huge_looser=lambda info: "remove_all")
CALLBACK_SETS = (ACCEPT_ALL, {},
                 dict(confirm_huge_displayed=lambda i: "remove", confirm_huge_looser=lambda i: "displayed_stricter"))
KEYS = ("d", "t", "b", "a", "g")


@pytest.fixture(scope="module")
def S():
    module = sup.load_store_module(sup.MODULE_NAME + "_review")
    yield module
    sup.unload_store_module(module)


@pytest.fixture(scope="module")
def kit(S):
    return sup.StoreKit(S)


@pytest.fixture(scope="module")
def expected(tmp_path_factory):
    """{cent: {(grid, key): mask}} of the scalar fixture the published review is made from."""
    root = tmp_path_factory.mktemp("arrays")
    return {cent: scalar(root, cent)[1] for cent in (50, 80)}


@pytest.fixture()
def review(tmp_path):
    return publish(tmp_path)


def logical_keys(plan):
    return {plan.layers[lid]["layer_key"]: lid for lid in plan.order}


def check_masks(kit, store, plan, masks):
    """masks: {(cent, key): full-grid mask}; absent pairs must be empty."""
    for key, lid in logical_keys(plan).items():
        for k, cent in enumerate(plan.cutoffs):
            want = masks.get((cent, key))
            want = np.zeros(store.shape, bool) if want is None else np.asarray(want, bool)
            np.testing.assert_array_equal(kit.full_mask(store, lid, k), want, err_msg=f"{key} at {cent}")
    assert kit.maps_consistent(store) and kit.counts_consistent(store)


def published_masks(expected):
    return {(cent, key): expected[cent][LOW, key] for cent in (50, 80) for key in KEYS}


# --------------------------------------------------------------------------------------------
# Builder


def test_builder_reproduces_every_published_mask(S, kit, review, expected, tmp_path):
    plan = S._xta_store_review_plan(review, LOW)
    assert plan.cutoffs == [50, 80] and plan.shape == (6, 8, 9) and not plan.detached
    assert plan.order == sorted(plan.order) and plan.global_lid == logical_keys(plan)["g"]
    store, rep = S._xta_store_build_from_review(plan)
    check_masks(kit, store, plan, published_masks(expected))
    keys = logical_keys(plan)
    storage = {key: store.layers[lid].storage for key, lid in keys.items()}
    assert storage == dict(d="level", t="level", a="level", b="bits", g="bits")      # b and g do not nest
    assert rep.bit_plane_layers == {keys["b"]: 80, keys["g"]: 80}
    assert store.layers[keys["b"]].data.dtype == np.uint16 and store.layers[keys["d"]].data.dtype == np.uint8
    roles = {key: (store.layers[lid].role, store.layers[lid].additive) for key, lid in keys.items()}
    assert roles == dict(d=("direct", True), t=("direct", True), b=("generated", True), a=("derived", False),
                         g=("global", False))
    for key, lid in keys.items():          # trusted counts are the rows' foreground_voxels
        rows = plan.layers[lid]["rows"]
        assert store.layers[lid].counts.tolist() == [rows[k]["foreground_voxels"] for k in (0, 1)]
    assert rep.rows == 10 and rep.group_reads == 2 and rep.store_bytes == sum(
        l.data.nbytes for l in store.layers.values())
    assert not store.journal and not store.global_stale
    # one worker gives the same store; so do the grid folder's copy and a detached copy of the folder
    folder = review.parent / LOW
    detached = tmp_path / "desk" / "half"
    shutil.copytree(folder, detached)
    digest = S._xta_store_state_digest(store)
    for source, workers in ((review, 1), (folder / "manifest.json", 4), (detached / "manifest.json", 2)):
        other, _ = S._xta_store_build_from_review(source, workers=workers)
        assert S._xta_store_state_digest(other) == digest
        assert all(np.array_equal(other.layers[l].data, store.layers[l].data) for l in store.layers)
    assert S._xta_store_review_plan(detached / "manifest.json").detached
    with pytest.raises(S.XTACutoffReviewError, match="holds only"):
        S._xta_store_review_plan(detached / "manifest.json", "low_quality/other")


def test_direct_counts_equal_published_coverage(S, review):
    _text, tree = sup.parse_source()
    ns = {}
    nodes = sup.function_nodes(tree, ("_xta_display_coverage_entries", "_xta_display_read_coverage"))
    exec(compile(sup.ast.Module(body=nodes, type_ignores=[]), str(sup.SOURCE), "exec"), ns)
    plan = S._xta_store_review_plan(review, LOW)
    store, _ = S._xta_store_build_from_review(plan)
    entries = ns["_xta_display_coverage_entries"](plan.grid)
    direct = [lid for lid in plan.order if plan.layers[lid]["role"] == "direct"]
    for k, cent in enumerate(plan.cutoffs):
        published = np.zeros(plan.shape, np.uint8)
        ns["_xta_display_read_coverage"](review.parent / entries[cent]["path"], plan.grid, published,
                                         sweep_id=plan.sweep_id, grid_id=LOW, cutoff_cent=cent)
        counts = np.zeros(plan.shape, np.uint8)
        for lid in direct:
            counts += sup.StoreKit(S).full_mask(store, lid, k)
        np.testing.assert_array_equal(counts, published)


def test_precheck_estimate_from_the_manifest_alone(S, review):
    plan = S._xta_store_review_plan(review, LOW)
    for g in plan.groups:                        # the pre-check opens no group file
        os.rename(g["file"], str(g["file"]) + ".away")
    est = S._xta_store_review_estimate(plan, undo_bytes=1 << 20, workers=2)
    for g in plan.groups:
        os.rename(str(g["file"]) + ".away", g["file"])
    store, rep = S._xta_store_build_from_review(plan)
    built = sum(l.data.nbytes for l in store.layers.values())
    assert abs(est.store_bytes - built) <= 0.05 * built
    assert set(est.bit_plane_layers) == set(rep.bit_plane_layers)
    assert est.store_bytes <= est.store_bytes_max
    voxels = int(np.prod(plan.shape))
    assert est.voxels == voxels and est.maps_bytes == 2 * voxels == store.U_vis.nbytes + store.U_all.nbytes
    assert est.working_labelmaps_bytes == sum(int(np.prod(l.shape3)) for l in store.layers.values())
    assert est.undo_bytes == 1 << 20
    assert est.total_bytes == (est.store_bytes + est.maps_bytes + est.working_labelmaps_bytes + est.display_bytes
                               + est.undo_bytes + est.decode_bytes)
    assert est.total_bytes_max - est.total_bytes == est.store_bytes_max - est.store_bytes
    assert S._xta_store_review_estimate(review).store_bytes == est.store_bytes       # a path works too


def read_review(review):
    return json.loads(review.read_text(encoding="utf-8"))


def save_review(review, doc):
    """Write the review manifest and the grid folder's byte-identical copy."""
    data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    review.write_bytes(data)
    (review.parent / LOW / "manifest.json").write_bytes(data)


def regroup(review, masks, layout):
    """Rewrite LOW's group files: layout lists groups of (cent, key); masks[(cent, key)] is a full-grid mask (a pair
    in no group is an absent row). Groups may mix cutoffs (legacy)."""
    doc = read_review(review)
    grid = next(g for g in doc["grids"] if g["grid_id"] == LOW)
    template = {(r["cutoff_cent"], r["layer_key"]): r for g in grid["groups"] for r in g["segments"]}
    for g in grid["groups"]:
        (review.parent / g["path"]).unlink()
    geometry = ReferenceGeometry(tuple(grid["shape_tyx"]), space=grid["space"],
                                 directions_xyz=tuple(map(tuple, grid["directions_xyz"])),
                                 origin_xyz=tuple(grid["origin_xyz"]))
    groups = []
    for i, members in enumerate(layout):
        segments = []
        for cent, key in members:
            row = template[(cent, key)]
            tags = {"XTA.SweepID": doc["sweep_id"], "XTA.GridID": LOW, "XTA.SegmentID": row["segment_id"],
                    "XTA.LogicalID": row["logical_id"], "XTA.CutoffCent": str(cent), "XTA.Role": row["role"]}
            mask = np.asarray(masks[(cent, key)], np.uint8)
            segments.append(GroupedSegment(row["segment_id"], row["display_name"], tuple(row["color_rgb"]),
                                           lambda z0, z1, m=mask: m[z0:z1], tags))
        rel = f"{LOW}/legacy_{i:03d}.seg.nrrd"
        stats = write_grouped_seg_nrrd(review.parent / rel, shape_tyx=tuple(grid["shape_tyx"]), segments=segments,
                                       geometry=geometry, memory_mib=2)
        rows = []
        for (cent, key), written in zip(members, stats["segments"]):
            rows.append({**template[(cent, key)], "empty_segment": written["empty_segment"],
                         "extent_xyz": written["extent_xyz"], "layer": written["layer"],
                         "foreground_voxels": written["foreground_voxels"]})
        groups.append({"path": rel, "cutoff_cent": min(c for c, _ in members), "segments": rows,
                       "bytes": stats["bytes"], "stored_shape_tyx": stats["stored_shape_tyx"],
                       "offset_xyz": stats["offset_xyz"]})
    grid["groups"] = groups
    save_review(review, doc)


def test_mixed_cutoff_groups_empty_and_absent_rows(S, kit, review, expected):
    masks = published_masks(expected)
    masks[(80, "d")] = np.zeros_like(masks[(80, "d")])          # an empty row at 80
    del masks[(80, "t")]                                         # t is absent at 80
    del masks[(50, "a")]                                         # a appears only at 80: not nested
    layout = [[(50, "d"), (80, "b"), (50, "t")], [(80, "d"), (80, "g"), (50, "b")], [(50, "g"), (80, "a")]]
    regroup(review, masks, layout)
    plan = S._xta_store_review_plan(review, LOW)
    assert all(len(g["cutoffs"]) == 2 for g in plan.groups)
    keys = logical_keys(plan)
    d_rows = plan.layers[keys["d"]]["rows"]
    assert d_rows[1]["empty_segment"] and d_rows[1]["extent_xyz"] == [0, -1, 0, -1, 0, -1]
    assert sorted(plan.layers[keys["t"]]["rows"]) == [0] and sorted(plan.layers[keys["a"]]["rows"]) == [1]
    # the box of d is its non-empty row's extent only
    np.testing.assert_array_equal(plan.layers[keys["d"]]["box"][0], [2, 2, 2])
    np.testing.assert_array_equal(plan.layers[keys["d"]]["box"][1], [4, 5, 5])
    est = S._xta_store_review_estimate(plan)
    store, rep = S._xta_store_build_from_review(plan, workers=3)
    check_masks(kit, store, plan, masks)
    assert rep.group_reads == 6 and rep.rows == 8              # every legacy group is decoded once per cutoff
    storage = {key: store.layers[lid].storage for key, lid in keys.items()}
    assert storage == dict(d="level", t="level", a="bits", b="bits", g="bits")
    assert rep.bit_plane_layers == {keys["a"]: 80, keys["b"]: 80, keys["g"]: 80}
    assert set(est.bit_plane_layers) == set(rep.bit_plane_layers)
    assert est.store_bytes == sum(l.data.nbytes for l in store.layers.values())
    assert store.layers[keys["d"]].counts.tolist() == [int(masks[(50, "d")].sum()), 0]


def first_group(review):
    doc = read_review(review)
    grid = next(g for g in doc["grids"] if g["grid_id"] == LOW)
    return doc, grid, review.parent / grid["groups"][0]["path"]


def edit_header(path, old, new):
    raw = path.read_bytes()
    end = raw.index(b"\n\n")
    head = raw[:end]
    assert head.count(old) >= 1, old
    path.write_bytes(head.replace(old, new, 1) + raw[end:])


def damage(review, problem):
    doc, grid, path = first_group(review)
    row = grid["groups"][0]["segments"][0]
    if problem == "gzip_data":
        raw = bytearray(path.read_bytes())
        start = raw.index(b"\n\n") + 2
        i = start + 40
        raw[i] ^= 0xFF
        path.write_bytes(bytes(raw))
    elif problem == "truncated":
        path.write_bytes(path.read_bytes()[:-9])
    elif problem == "longer":
        path.write_bytes(path.read_bytes() + gzip.compress(b"\x01" * 50))
    elif problem == "raw_encoding_claimed":
        edit_header(path, b"encoding: gzip", b"encoding: raw")
    elif problem == "sizes":
        edit_header(path, b"sizes: ", b"sizes: 1")
    elif problem == "label_value":
        edit_header(path, b"Segment0_LabelValue:=1", b"Segment0_LabelValue:=2")
    elif problem == "grid_tag":
        edit_header(path, f"XTA.GridID:{LOW}".encode(), b"XTA.GridID:low_quality/other")
    elif problem == "segment_id":
        edit_header(path, f"Segment0_ID:={row['segment_id']}".encode(), b"Segment0_ID:=xta_other")
    elif problem == "origin":
        edit_header(path, b"space origin: (", b"space origin: (1")
    elif problem == "not_nrrd":
        path.write_bytes(b"PNG" + path.read_bytes()[4:])
    elif problem == "foreground":
        r = next(r for r in grid["groups"][0]["segments"] if r["foreground_voxels"] > 1)
        r["foreground_voxels"] -= 1
        save_review(review, doc)
    elif problem == "not_tight":                                  # one more x column than the mask fills
        r = next(r for r in grid["groups"][0]["segments"] if r["layer_key"] == "d")
        ox, oy, oz = grid["groups"][0]["offset_xyz"]

        def local(e):
            return " ".join(str(v - o) for v, o in zip(e, (ox, ox, oy, oy, oz, oz)))
        old = f"Segment{r['layer']}_Extent:={local(r['extent_xyz'])}"
        r["extent_xyz"][1] += 1
        edit_header(path, old.encode(), f"Segment{r['layer']}_Extent:={local(r['extent_xyz'])}".encode())
        save_review(review, doc)
    else:
        raise AssertionError(problem)


@pytest.mark.parametrize("problem,message", [
    ("gzip_data", "damaged gzip payload"), ("truncated", "truncated"), ("longer", "longer than its header"),
    ("raw_encoding_claimed", "values other than its label values"), ("sizes", "stored box lies outside|stored shape"),
    ("label_value", "values other than its label values"), ("grid_tag", r"XTA tags \['GridID'\]"),
    ("segment_id", "segments differ from the manifest"), ("origin", "space origin"), ("not_nrrd", "not an NRRD"),
    ("foreground", "has 3 voxels, the manifest 2"), ("not_tight", "not tight")])
def test_damaged_or_mismatched_group_files_are_refused(S, review, problem, message):
    damage(review, problem)
    with pytest.raises(S.XTACutoffReviewError, match=message):
        S._xta_store_build_from_review(review, LOW, workers=2)


def test_raw_encoded_group_files_build(S, kit, review, expected):
    doc, grid, _ = first_group(review)
    for g in grid["groups"]:
        path = review.parent / g["path"]
        raw = path.read_bytes()
        end = raw.index(b"\n\n") + 2
        path.write_bytes(raw[:end].replace(b"encoding: gzip", b"encoding: raw") + gzip.decompress(raw[end:]))
    plan = S._xta_store_review_plan(review, LOW)
    store, _ = S._xta_store_build_from_review(plan)
    check_masks(kit, store, plan, published_masks(expected))


def test_manifest_refusals(S, review):
    doc = read_review(review)
    grid = next(g for g in doc["grids"] if g["grid_id"] == LOW)
    rows = [r for g in grid["groups"] for r in g["segments"]]
    cases = []
    bad = copy.deepcopy(doc)
    next(r for g in next(x for x in bad["grids"] if x["grid_id"] == LOW)["groups"] for r in g["segments"]
         if r["cutoff_cent"] == 80 and r["layer_key"] == "d")["segment_name"] = "renamed"
    cases.append((bad, "changes its segment_name"))
    bad = copy.deepcopy(doc)
    for g in next(x for x in bad["grids"] if x["grid_id"] == LOW)["groups"]:
        for r in g["segments"]:
            if r["layer_key"] == "g":
                r["source_filename"] = "volume_g.seg.nrrd"
    cases.append((bad, "exactly one Global"))
    bad = copy.deepcopy(doc)
    bad["cutoffs_cent"] = [50, 70, 80]
    cases.append((bad, "cutoffs_cent"))
    bad = copy.deepcopy(doc)
    for g in next(x for x in bad["grids"] if x["grid_id"] == LOW)["groups"]:
        for r in g["segments"]:
            if r["layer_key"] == "b":
                r["layer_role"] = "diagnostic_only"
    cases.append((bad, "would not tag it additive"))
    assert {r["layer_key"] for r in rows} == set(KEYS)
    original = review.read_bytes()
    for bad_doc, message in cases:
        save_review(review, bad_doc)
        with pytest.raises(S.XTACutoffReviewError, match=message):
            S._xta_store_review_plan(review, LOW)
    review.write_bytes(original)
    with pytest.raises(S.XTACutoffReviewError):
        S._xta_store_review_plan(review, "native")                # reference-only grid


def test_build_cancel_and_progress(S, review):
    plan = S._xta_store_review_plan(review, LOW)
    seen = []
    store, _ = S._xta_store_build_from_review(plan, progress=seen.append)
    done = [p["done_bytes"] for p in seen]
    assert done == sorted(done) and seen[-1]["done_bytes"] == seen[-1]["total_bytes"] > 0
    assert seen[-1]["cutoffs_done"] == 2
    with pytest.raises(S.XTACutoffBuildCancelled):
        S._xta_store_build_from_review(plan, cancel=lambda: True)
    calls = []

    def later():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(S.XTACutoffBuildCancelled):
        S._xta_store_build_from_review(plan, cancel=later)
    # a worker stops at its next slab once the build is stopped
    stop = S._xta_store_threading.Event()
    stop.set()
    states = {lid: S._XTAStoreBuildLayer(lid, plan.layers[lid]["box"], 2) for lid in plan.order}
    with pytest.raises(S.XTACutoffBuildCancelled):
        S._xta_store_decode_group(plan, plan.groups[0], 0, states, stop, [0], 0)


# --------------------------------------------------------------------------------------------
# Edit log


def identity(store, **change):
    out = dict(sweep_id="sweep-scene", grid_id="low_quality/scene", review_manifest_sha256="0" * 64,
               cutoffs_cent=list(range(store.n_cutoffs)), shape_tyx=list(store.shape))
    out.update(change)
    return out


def entry_key(S, store, e):
    full = store._load(e, keep_file=True) if e.spill_path else e
    changes = [(c.lid, tuple(int(v) for v in c.origin), tuple(c.shape), c.dtype, c.words, c.mask, c.prior, c.changed,
                c.after, c.box_before, c.box_after, c.replace, c.prior_meta, c.after_meta) for c in full.changes]

    def painted(d):
        return {lid: S._xta_store_as_lin(v).tolist() for lid, v in d.items()}
    return (e.kind, int(e.k0), e.global_stale_before, e.global_stale_after, changes, painted(full.painted_added),
            painted(full.painted_removed), e.report, e.stats, e.boxes, e.nbytes, e.spill_path is None)


def assert_same_store(S, kit, live, replayed):
    """Layers, maps, counts, painted set, stale flag and the undo / redo stacks entry by entry."""
    replayed.set_visible_many({lid: l.visible for lid, l in live.layers.items()})
    assert list(live.layers) == list(replayed.layers)
    for lid, a in live.layers.items():
        b = replayed.layers[lid]
        assert (a.storage, a.data.dtype, a.visible) == (b.storage, b.data.dtype, b.visible)
        assert np.array_equal(a.origin, b.origin) and np.array_equal(a.data, b.data), lid
        assert np.array_equal(a.counts, b.counts), lid
    assert np.array_equal(live.U_vis, replayed.U_vis) and np.array_equal(live.U_all, replayed.U_all)
    assert live.global_stale == replayed.global_stale
    assert set(live.painted) == set(replayed.painted)
    assert all(np.array_equal(live.painted[k], replayed.painted[k]) for k in live.painted)
    assert (len(live.journal), len(live.redo_stack), live.dropped_entries) == (
        len(replayed.journal), len(replayed.redo_stack), replayed.dropped_entries)
    for a, b in zip(live.journal + live.redo_stack, replayed.journal + replayed.redo_stack):
        assert entry_key(S, live, a) == entry_key(S, replayed, b)
    assert live.last_changed_boxes == replayed.last_changed_boxes
    assert kit.maps_consistent(replayed) and kit.counts_consistent(replayed)


class Ops:
    """Random store operations on a store with a log; counts the records each one must append."""

    def __init__(self, S, kit, scene, store, rng):
        self.S, self.kit, self.scene, self.s, self.rng = S, kit, scene, store, rng
        self.expected = 0
        self.pushes = 0
        original = store._push

        def counting(entry):
            self.pushes += 1
            return original(entry)
        store._push = counting

    def step(self, i):
        S, s, rng = self.S, self.s, self.rng
        before = self.pushes
        u = rng.random()
        T = S.XTACutoffThresholds(huge_displayed=1500, huge_looser=600, highlight_share=0.25, highlight_voxels=300)
        undone = 0
        if u < 0.14 and s.journal:
            undone += bool(s.undo())
        elif u < 0.24 and s.redo_stack:
            undone += bool(s.redo())
        elif u < 0.29:
            lid = list(s.layers)[int(rng.integers(0, len(s.layers)))]
            s.set_visible_many({lid: not s.layers[lid].visible})
        elif u < 0.5:
            sup.random_syncs(self.kit, s, rng, 1)
        elif u < 0.62 and s.global_layer() is not None:
            K = s.n_cutoffs
            cutoffs = None if rng.random() < 0.6 else sorted(set(rng.integers(0, K, 2).tolist()))
            v = rng.random()
            if v < 0.3:
                S._xta_store_update_global(s, cutoffs, tie_breaker="raster_first")
            elif v < 0.5:
                S._xta_store_update_nonglobal(s, cutoffs)
            elif v < 0.8:
                S._xta_store_update_both(s, cutoffs, tie_breaker="raster_first")
            elif v < 0.9:
                S._xta_store_update_global(s, cutoffs, tie_breaker="raster_first", dry_run=True)
            else:                                    # an export's failed update, rolled back (UNDO + DROP_REDO)
                rec = S._xta_store_update_global(s, None, tie_breaker="raster_first")
                if rec.entry is not None:
                    S._xta_store_rollback_update(s, rec)
                    undone += 2
        else:
            k0, markers, _ = sup.make_case(self.kit, self.scene, 300 + i, store=s)
            fn = S._xta_store_remove_slices if rng.random() < 0.3 else S._xta_store_remove_islands
            fn(s, k0, markers, engine=ENGINE, scope="stricter" if rng.random() < 0.2 else "all", thresholds=T,
               **CALLBACK_SETS[int(rng.integers(0, len(CALLBACK_SETS)))])
        self.expected += (self.pushes - before) + undone


def scene_store(S, kit, seed, K, **store_kw):
    scene = sup.make_scene(kit, seed, K)
    store = scene["store"]
    if store_kw:
        rebuilt = S.XTACutoffStore(store.shape, store.n_cutoffs, **store_kw)
        for lid, l in store.layers.items():
            rebuilt.add_layer(S.XTACutoffLayer(lid, l.role, l.data, l.origin, visible=l.visible,
                                               additive=l.additive if l.role == "derived" else None))
        scene["store"] = store = rebuilt
    return scene, store


@pytest.mark.parametrize("seed,K,cap", [(71001, 5, None), (71002, 17, None), (71003, 65, None), (71004, 9, 400)])
def test_log_replay_equals_live_store_and_stacks(S, kit, tmp_path, seed, K, cap):
    """Session 1 edits and logs; session 2 rebuilds, replays (equal to session 1) and keeps editing into the same
    log; a third replay equals session 2. A small undo cap makes the replays drop and spill as the sessions did."""
    kw = dict(spill_dir=str(tmp_path / "spill"))
    os.makedirs(kw["spill_dir"])
    rng = np.random.default_rng([seed, 3])

    def fresh():
        scene, store = scene_store(S, kit, seed, K, **kw)
        if cap:
            store.undo_bytes = cap
        return scene, store
    scene, live = fresh()
    path = tmp_path / "log" / "edits.xtalog"
    log = S.XTACutoffEditLog.start(path, live, identity(live))
    assert live.edit_log is log and log.records == 0
    ops = Ops(S, kit, scene, live, rng)
    for i in range(18):
        ops.step(i)
        assert log.error is None and log.records == ops.expected, i
    log.close()
    assert live.edit_log is None
    scene2, second = fresh()
    log = S.XTACutoffEditLog.resume(path, second, identity(second))
    assert log.replayed == log.records == ops.expected and log.truncated_bytes == 0
    assert_same_store(S, kit, live, second)            # also gives the second session the first one's eyes
    live.close()
    ops2 = Ops(S, kit, scene2, second, rng)
    ops2.expected = ops.expected
    for i in range(18, 36):
        ops2.step(i)
        assert log.error is None and log.records == ops2.expected, i
    assert ops.pushes + ops2.pushes > 5 and second.journal
    log.close()
    _, third = fresh()
    seen = []
    log3 = S.XTACutoffEditLog.resume(path, third, identity(third), progress=seen.append)
    assert log3.replayed == ops2.expected == seen[-1]["total"] == seen[-1]["done"]
    assert_same_store(S, kit, second, third)
    if cap:
        assert third.dropped_entries > 0 and any(e.spill_path for e in third.journal + third.redo_stack)
    while second.journal:                                # both keep working the same way
        assert second.undo() and third.undo()
    while second.redo_stack:
        assert second.redo() and third.redo()
    assert_same_store(S, kit, second, third)
    summary = S._xta_store_edit_log_summary(path)
    assert summary["records"] == log3.records and summary["error"] is None and summary["torn_bytes"] == 0
    assert summary["edits"] + summary["undos"] + summary["redos"] + summary["dropped_redos"] == summary["records"]
    second.close()
    third.close()
    assert log3.closed and not os.listdir(kw["spill_dir"])


def simple_step(S, kit, scene, store, rng, i):
    """One operation that appends at most one record: a click, a sync, an undo or a redo."""
    u = rng.random()
    if u < 0.15 and store.journal:
        store.undo()
    elif u < 0.25 and store.redo_stack:
        store.redo()
    elif u < 0.6:
        sup.random_syncs(kit, store, rng, 1)
    else:
        k0, markers, _ = sup.make_case(kit, scene, 500 + i, store=store)
        S._xta_store_remove_islands(store, k0, markers, engine=ENGINE, **ACCEPT_ALL)


def log_with_steps(S, kit, tmp_path, seed=72001, K=6, steps=10):
    """(path, fresh store factory, live store, snapshot after each record, file size after each record)."""
    def fresh():
        return scene_store(S, kit, seed, K)[1]
    scene, live = scene_store(S, kit, seed, K)
    path = tmp_path / "edits.xtalog"
    log = S.XTACutoffEditLog.start(path, live, identity(live))
    rng = np.random.default_rng([seed, 9])
    snaps, sizes = [kit.snapshot(live)], [path.stat().st_size]
    i = 0
    while log.records < steps:
        n = log.records
        simple_step(S, kit, scene, live, rng, i)
        i += 1
        assert log.records - n in (0, 1)
        if log.records > n:
            snaps.append(kit.snapshot(live))
            sizes.append(path.stat().st_size)
    log.close()
    return path, fresh, live, snaps, sizes


def test_log_torn_tail_is_cut_off(S, kit, tmp_path):
    path, fresh, live, snaps, sizes = log_with_steps(S, kit, tmp_path)
    full = path.read_bytes()
    for cut in (sizes[-1] - 1, sizes[-2] + 5, sizes[-2] + 21 + 3):     # torn body, partial frame, torn body
        path.write_bytes(full[:cut])
        store = fresh()
        with pytest.warns(S.XTACutoffEditLogWarning, match="cut off"):
            log = S.XTACutoffEditLog.resume(path, store, identity(store))
        assert log.records == len(snaps) - 2 and path.stat().st_size == sizes[-2]
        assert kit.same_state(store, snaps[-2], maps=False) and kit.maps_consistent(store)
        st = S._xta_store_sync_paint(store, list(store.layers)[0], store.n_cutoffs - 1, np.array([[0, 0, 0]]))
        assert log.records == len(snaps) - 2 + (st["status"] == "applied")     # appends again
        log.close()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            again = fresh()
            S.XTACutoffEditLog.resume(path, again, identity(again)).close()
        assert kit.same_state(again, kit.snapshot(store))


def test_log_crc_damage(S, kit, tmp_path):
    path, fresh, live, snaps, sizes = log_with_steps(S, kit, tmp_path)
    full = path.read_bytes()
    middle = bytearray(full)
    middle[sizes[3] + 30] ^= 0x5A                                 # a byte of the fourth record
    path.write_bytes(bytes(middle))
    store = fresh()
    digest = S._xta_store_state_digest(store)
    with pytest.raises(S.XTACutoffEditLogCorrupt, match="damaged"):
        S.XTACutoffEditLog.resume(path, store, identity(store))
    assert not store.journal and S._xta_store_state_digest(store) == digest and store.edit_log is None
    assert path.read_bytes() == bytes(middle)                       # refused: nothing changed
    last = bytearray(full)
    last[-2] ^= 0x5A                                               # the last record: a torn tail
    path.write_bytes(bytes(last))
    store = fresh()
    with pytest.warns(S.XTACutoffEditLogWarning):
        S.XTACutoffEditLog.resume(path, store, identity(store)).close()
    assert kit.same_state(store, snaps[-2], maps=False)
    path.write_bytes(b"not a log" + full)
    with pytest.raises(S.XTACutoffEditLogCorrupt):
        S.XTACutoffEditLog.resume(path, fresh(), identity(store))


def test_log_header_mismatch_archive_and_summary(S, kit, tmp_path):
    path, fresh, live, snaps, sizes = log_with_steps(S, kit, tmp_path, steps=6)
    summary = S._xta_store_edit_log_summary(path)
    assert summary["exists"] and summary["records"] == 6 and summary["last_utc"] >= summary["first_utc"]
    assert summary["header"]["identity"] == identity(live) and summary["error"] is None
    assert summary["journal_depth"] == len(live.journal) and summary["redo_depth"] == len(live.redo_stack)
    before = path.read_bytes()
    store = fresh()
    with pytest.raises(S.XTACutoffBuildCancelled):
        S.XTACutoffEditLog.resume(path, store, identity(store), cancel=lambda: True)
    for other in (dict(grid_id="low_quality/other"), dict(review_manifest_sha256="1" * 64)):
        store = fresh()
        with pytest.raises(S.XTACutoffEditLogMismatch):
            S.XTACutoffEditLog.resume(path, store, identity(store, **other))
        assert not store.journal and store.edit_log is None
    store = fresh()                                                   # the review's masks changed
    lid = next(l for l, layer in store.layers.items() if layer.storage == "level")
    changed = S.XTACutoffStore(store.shape, store.n_cutoffs)
    for l, layer in store.layers.items():
        data = layer.data.copy()
        if l == lid:
            data.reshape(-1)[0] = 1 if data.reshape(-1)[0] == 0 else 0
        changed.add_layer(S.XTACutoffLayer(l, layer.role, data, layer.origin, visible=layer.visible,
                                           additive=layer.additive if layer.role == "derived" else None))
    with pytest.raises(S.XTACutoffEditLogMismatch, match="store digest"):
        S.XTACutoffEditLog.resume(path, changed, identity(changed))
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        S.XTACutoffEditLog.start(path, fresh(), identity(live))
    archived = S._xta_store_edit_log_archive(path)
    assert archived.parent == path.parent and archived.name.startswith("edits.archived-")
    assert archived.suffix == ".xtalog" and not path.exists() and archived.read_bytes() == before
    store = fresh()
    S.XTACutoffEditLog.start(path, store, identity(store)).close()
    assert S._xta_store_edit_log_summary(path)["records"] == 0
    assert S._xta_store_edit_log_summary(archived)["records"] == 6
    again = S._xta_store_edit_log_archive(path)
    assert again != archived and S._xta_store_edit_log_archive(path) is None
    assert not S._xta_store_edit_log_summary(path)["exists"]
    with pytest.raises(S.XTACutoffEditLogError):
        S.XTACutoffEditLog.start(tmp_path / "x.xtalog", live, identity(live))   # not a fresh store


def test_log_hook_noops_and_write_failure(S, kit, tmp_path):
    store = sup.g_scene(kit)
    log = S.XTACutoffEditLog.start(tmp_path / "e.xtalog", store, identity(store))
    S._xta_store_remove_islands(store, 2, np.array([[0, 0, 0]]), engine=ENGINE)             # no hit
    S._xta_store_sync_erase(store, "A", 2, np.array([[0, 0, 39]]))                          # nothing to erase
    S._xta_store_update_global(store, tie_breaker="raster_first", dry_run=True)
    assert not store.undo() and not store.redo()
    assert log.records == 0
    S._xta_store_remove_islands(store, 2, np.array([[8, 14, 30]]), engine=ENGINE, **ACCEPT_ALL)
    assert log.records == 1 and store.undo() and store.redo() and log.records == 3
    errors = []
    log.on_error = errors.append

    class Broken:
        def write(self, data):
            raise OSError("disk full")

        def close(self):
            pass
    log._f = Broken()
    with pytest.warns(S.XTACutoffEditLogWarning, match="disk full"):
        st = S._xta_store_sync_paint(store, "B", 1, np.array([[2, 2, 30]]))
    assert st["status"] == "applied" and len(store.journal) == 2         # the edit itself went through
    assert log.error and errors == [log.error] and store.edit_log is None and log.closed
    assert store.undo()                                                  # nothing logged any more


def test_log_path_rule(S, review, tmp_path):
    token = S._xta_store_hashlib.sha256(LOW.encode()).hexdigest()[:12]
    sweep = review.parent.parent
    want = sweep.parent / f"{sweep.name}.cutoff_edits" / token / "edits.xtalog"
    assert S._xta_store_edit_log_path(review, LOW) == want
    assert S._xta_store_edit_log_path(review.parent / LOW / "manifest.json", LOW) == want
    detached = tmp_path / "desk" / "half"
    shutil.copytree(review.parent / LOW, detached)
    assert S._xta_store_edit_log_path(detached / "manifest.json", LOW) == (
        tmp_path / "desk" / "half.cutoff_edits" / token / "edits.xtalog")


def test_built_review_store_log_and_reexec(review, tmp_path):
    """A store built from the review, logged at the review's log path; a re-exec of the file keeps the log."""
    S = sup.load_store_module(sup.MODULE_NAME + "_review_reexec")
    try:
        kit = sup.StoreKit(S)
        plan = S._xta_store_review_plan(review, LOW)
        path = S._xta_store_edit_log_path(plan.manifest_path, plan.grid_id)
        live, _ = S._xta_store_build_from_review(plan)
        S.XTACutoffEditLog.start(path, live, plan.identity())
        keys = logical_keys(plan)
        S._xta_store_remove_islands(live, 0, np.array([[3, 3, 3]]), engine=ENGINE, **ACCEPT_ALL)
        S._xta_store_sync_paint(live, keys["b"], 1, np.array([[5, 7, 8]]))
        S._xta_store_update_both(live, tie_breaker="raster_first")
        sup.reexec_section(S)
        assert type(live.edit_log) is not S.XTACutoffEditLog
        assert S._xta_store_adopt(live) is live and type(live.edit_log) is S.XTACutoffEditLog
        n = live.edit_log.records
        assert n >= 3 and live.undo() and live.edit_log.records == n + 1
        st = S._xta_store_sync_paint(live, keys["d"], 1, np.array([[0, 0, 0]]))
        assert st["status"] == "applied" and live.edit_log.records == n + 2 and live.edit_log.error is None
        live.edit_log.close()
        assert live.edit_log is None
        replayed, _ = S._xta_store_build_from_review(S._xta_store_review_plan(review, LOW))
        S.XTACutoffEditLog.resume(path, replayed, plan.identity())
        assert_same_store(S, kit, live, replayed)
        live.close()
        replayed.close()                                 # closing the store closes its log too
        assert replayed.edit_log is None and not replayed.journal
    finally:
        sup.unload_store_module(S)


# --------------------------------------------------------------------------------------------
# Export against today's ExportXTASweepEdits


class TightSegmentationNode:
    """Wraps test_sweep_edits' fake node: each labelmap comes back cropped to its foreground with its extent, as
    Slicer holds an Importer-loaded segment (today's export writes the labelmap's own extent)."""

    def __init__(self, node):
        self.node = node

    def __getattr__(self, name):
        return getattr(self.node, name)

    def GetSegmentation(self):
        return self

    def GetBinaryLabelmapRepresentation(self, sid, image):
        full = np.asarray(self.node.arrays[self.node.rows[sid][1]], np.uint8)
        nz = np.argwhere(full)
        if not len(nz):
            image.mask, image.extent = np.zeros((1, 1, 1), np.uint8), [0, -1, 0, -1, 0, -1]
        else:
            lo, hi = nz.min(axis=0), nz.max(axis=0)
            image.mask = full[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].copy()
            image.extent = [int(lo[2]), int(hi[2]), int(lo[1]), int(hi[1]), int(lo[0]), int(hi[0])]
        image.matrix = self.node.matrix.copy()
        return True


def compare_exports(today, store_out):
    a = json.loads((today / "sweep_edits.json").read_text(encoding="utf-8"))
    b = json.loads((store_out / "sweep_edits.json").read_text(encoding="utf-8"))
    strip = lambda doc: {**doc, "cuts": [{k: v for k, v in c.items() if k != "edits_sha256"} for c in doc["cuts"]]}  # noqa: E731
    assert strip(a) == strip(b) and a["cuts"]
    for cut in a["cuts"]:
        x, y = today / cut["directory"], store_out / cut["directory"]
        assert sorted(p.name for p in x.iterdir()) == sorted(p.name for p in y.iterdir())
        for name in ("source_manifest.json", "source_native_manifest.json", "source_run_manifest.json"):
            assert (x / name).read_bytes() == (y / name).read_bytes()
        ex = json.loads((x / "edits.json").read_text(encoding="utf-8"))
        ey = json.loads((y / "edits.json").read_text(encoding="utf-8"))
        for doc in (ex, ey):
            doc.pop("created_utc")
        sx, sy = ex.pop("segments"), ey.pop("segments")
        assert ex == ey and len(sx) == len(sy) >= 2
        for rx_, ry_ in zip(sx, sy):
            assert ry_["segment_id"] == ry_["sweep_segment_id"]           # the store writes the review id
            assert rx_["sweep_segment_id"] == ry_["sweep_segment_id"]
            scene_fields = ("sha256", "node_name", "segment_id")         # npz bytes carry zip times
            assert {k: v for k, v in rx_.items() if k not in scene_fields} == {
                k: v for k, v in ry_.items() if k not in scene_fields}
            with np.load(x / rx_["file"]) as na, np.load(y / ry_["file"]) as nb:
                assert sorted(na.files) == sorted(nb.files) == ["extent_xyz", "ijk_to_ras", "mask"]
                for member in na.files:
                    assert na[member].dtype == nb[member].dtype
                    np.testing.assert_array_equal(na[member], nb[member])
    return a


def test_store_export_equals_today_export(S, kit, tmp_path):
    from tests import test_sweep_edits as tse
    from tools import prepare_pta_input as prep
    from tools import xta_sweep_edits as edits
    review, arrays = tse.sweep_fixture(tmp_path)
    namespace, node, _ = tse.export_api(review, arrays)
    tight = TightSegmentationNode(node)
    namespace["slicer"].util.getNodesByClass = lambda kind: [tight]
    today = tmp_path / "today"
    namespace["ExportXTASweepEdits"](review, today)
    grid = "low_quality/0p50"
    rx = S.XTACutoffReviewExport.open(review, grid)
    assert rx.cutoffs == [25, 75] and rx.geometry_error is None
    masks = [arrays[grid, cent] for cent in rx.cutoffs]
    store = S.XTACutoffStore(masks[0].shape, len(masks))
    kit.add_layer_from_masks(store, "component", "direct", masks)
    kit.add_layer_from_masks(store, "global", "global", masks)
    out = tmp_path / "store"
    report = S._xta_store_export_cutoffs(store, rx, None, out)
    assert report.cuts and not report.stale_cutoffs
    compare_exports(today, out)
    for cent, mask in zip(rx.cutoffs, masks):           # the PTA prepper reads the store's bundle unchanged
        child = edits.resolve_cut_bundle(out, cent)
        low, _full, selected, _ = prep.load_bundle(child, "sample")
        target = np.zeros(low, np.uint8)
        prep.insert_snapshot(target, child / selected[0]["file"], selected[0], 1024**2)
        np.testing.assert_array_equal(target, mask)


def test_built_store_export_equals_today_export(S, tmp_path):
    """Builder then export on a unit-LPS published review (transfer_fixture), against today's export of the same
    published masks loaded as tagged per-cutoff segments."""
    from tests import test_sweep_edits as tse
    root = tmp_path / "sweep"
    found = {}
    branches = []
    for cent in (50, 80):
        branch, expected = scalar(root, cent, transfer_fixture=True)
        branches.append(branch)
        found[cent] = expected
    review = publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2, workers=1)
    store, _ = S._xta_store_build_from_review(review, LOW)
    namespace, node, _ = tse.export_api(review, {})
    data = namespace["_xta_sweep_review"](review)
    rows, arrays = {}, {}
    for row in data["records"].values():
        if row["grid_id"] != LOW:
            continue
        tags = {"XTA.SweepID": data["sweep_id"], "XTA.SegmentID": row["segment_id"],
                "XTA.LogicalID": row["logical_id"], "XTA.CutoffCent": str(row["cutoff_cent"]),
                "XTA.GridID": LOW, "XTA.Role": row["role"], "XTA.Model": row["model_name"],
                "XTA.LayerKey": row["layer_key"]}
        key = (row["logical_id"], row["cutoff_cent"])
        rows[f"local-{len(rows)}"] = (tse.Segment(tags), key)
        arrays[key] = found[row["cutoff_cent"]][LOW, row["layer_key"]]
    node.rows, node.arrays = rows, arrays
    tight = TightSegmentationNode(node)
    namespace["slicer"].util.getNodesByClass = lambda kind: [tight]
    today = tmp_path / "today"
    namespace["ExportXTASweepEdits"](review, today)
    rx = S.XTACutoffReviewExport.open(review, LOW)
    out = tmp_path / "store"
    S._xta_store_export_cutoffs(store, rx, None, out)
    index = compare_exports(today, out)
    assert [c["cutoff_cent"] for c in index["cuts"]] == [50, 80]
    assert all(c["segment_count"] == 4 for c in index["cuts"])     # Global + d, t, b (a is not a union layer)
