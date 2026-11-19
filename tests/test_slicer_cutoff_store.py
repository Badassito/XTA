"""The cross-cutoff review store section of tools/slicer/xta_slicerrc.py (SPEC v3.1).

The section is compiled from the file between its sentinels (tests/slicer_cutoff_store_support.py). Checks:
the whole file compiles and the section is self-contained, prefixed and collision-free; the SPEC guarantees
on hand-made scenes (G2-G5, the Global follow rule, prompts, scopes, fragments, painted seeds, the report);
G1, every visible layer at the displayed cutoff equals today's Remove Islands / Remove Slices job compiled
from the same file; G6, undo / redo exact with the byte cap, plain-data spill files and a re-exec of the
file; G7, the windowed edits equal the full-grid rules on random scenes (level maps and bit planes, K in
{5, 17, 65}); G8 (store part), a store reproduces the masks it was built from; Update Global / Nonglobal
equal today's update numerics. Without SimpleITK the scikit-image reference flood engine is used.
"""
import ast
import gc
import os

import numpy as np
import pytest
from scipy import ndimage

from tests import slicer_cutoff_store_support as sup

ENGINE = sup.default_engine()
ACCEPT_ALL = dict(confirm_huge_displayed=lambda info: "remove", confirm_huge_looser=lambda info: "remove_all")
CALLBACK_SETS = (ACCEPT_ALL, {},
                 dict(confirm_huge_displayed=lambda i: "remove", confirm_huge_looser=lambda i: "displayed_stricter"),
                 dict(confirm_huge_displayed=lambda i: "remove", confirm_huge_looser=lambda i: "cancel"))


@pytest.fixture(scope="module")
def S():
    module = sup.load_store_module()
    yield module
    sup.unload_store_module(module)


@pytest.fixture(scope="module")
def kit(S):
    return sup.StoreKit(S)


def small_thresholds(S):
    return S.XTACutoffThresholds(huge_displayed=1500, huge_looser=600, highlight_share=0.25, highlight_voxels=300)


def islands(S, store, k0, markers, **kw):
    return S._xta_store_remove_islands(store, k0, markers, engine=ENGINE, **kw)


def slices(S, store, k0, markers, **kw):
    return S._xta_store_remove_slices(store, k0, markers, engine=ENGINE, **kw)


# --------------------------------------------------------------------------------------------
# The file and the section


def test_whole_file_compiles():
    """exec(open(...).read()) works: ast.parse accepts a mid-file 'from __future__' that compile rejects."""
    text = sup.SOURCE.read_text(encoding="utf-8")
    compile(text, str(sup.SOURCE), "exec")


def test_section_is_self_contained_prefixed_and_collision_free():
    _text, tree = sup.parse_source()
    first, last = sup.section_range(tree)
    body = tree.body[first:last + 1]
    names, imported = set(), set()
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets)
        elif isinstance(n, ast.Import):
            imported.update(a.name.split(".")[0] for a in n.names)
            names.update(a.asname for a in n.names)
            assert all(a.asname for a in n.names), "section imports are aliased"
        elif isinstance(n, ast.ImportFrom):
            assert n.module != "__future__"
            imported.add(n.module.split(".")[0])
            names.update(a.asname for a in n.names)
            assert all(a.asname for a in n.names), "section imports are aliased"
        else:
            assert isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant), ast.dump(n)[:80]
    assert imported <= {"concurrent", "dataclasses", "datetime", "functools", "hashlib", "importlib", "itertools",
                        "json", "marshal", "os", "pathlib", "shutil", "tempfile", "threading", "time", "warnings",
                        "weakref", "zlib"}, imported
    for name in names:
        assert name.startswith(("_xta_store_", "_XTA_STORE_", "XTACutoff", "_XTAStore", "_XTA_CUTOFF_STORE")), name
    # lazy imports inside functions: SciPy through the lazy modules, SimpleITK, vtk / vtkITK for Update Global
    # ties, and the platform's file lock for the edit log; never qt or slicer, and no pickle anywhere
    lazy = {(a.name if isinstance(n, ast.Import) else n.module).split(".")[0]
            for node in body for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names} - imported
    assert lazy <= {"SimpleITK", "vtk", "vtkITK", "msvcrt", "fcntl"}, lazy
    assert not any(isinstance(x, ast.Name) and x.id == "pickle" for node in body for x in ast.walk(node))
    # nothing else in the file defines a section name; today's helpers keep their single definitions
    outside = tree.body[:first] + tree.body[last + 1:]
    file_names = set()
    for n in outside:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            file_names.add(n.name)
        for x in ast.walk(n) if isinstance(n, (ast.Assign, ast.For, ast.With)) else ():
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                file_names.add(x.id)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            file_names.update((a.asname or a.name).split(".")[0] for a in n.names)
    assert not (names & file_names), sorted(names & file_names)
    defs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    for helper in ("_ras_points_to_continuous_ijk", "_voxel_indices_intersected_by_line",
                   "_clip_line_segment_to_box", "_remove_consecutive_duplicate_points", "_scanline_polygon_u_values",
                   "_closed_curve_interior_candidates", "_one_seed_per_axis_run"):
        assert defs.count(helper) == 1, helper
    # placed before the restore observers that run at exec, and before the toolbar
    restore = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and getattr(n.value.func, "id", None) == "_xta_restore_review")
    assert last < restore


def test_section_reuses_today_helpers(S):
    _text, tree = sup.parse_source()
    first, last = sup.section_range(tree)
    lo, hi = tree.body[first].lineno, tree.body[last].end_lineno
    for helper in ("_voxel_indices_intersected_by_line", "_closed_curve_interior_candidates",
                   "_remove_consecutive_duplicate_points", "_one_seed_per_axis_run", "_xta_sweep_binding"):
        code = getattr(S, helper).__code__
        assert code.co_filename == str(sup.SOURCE) and not lo <= code.co_firstlineno <= hi, helper
    # the matrix form of today's RAS -> IJK has its own name; today's takes an image
    m = np.diag([-1.0, -1.0, 1.0, 1.0])
    m[:3, 3] = (4.0, -2.0, 7.0)
    ijk = S._xta_store_ras_to_ijk(m, [(0.0, 0.0, 0.0), (-1.0, 3.0, 9.0)])
    np.testing.assert_allclose(ijk, [[4.0, -2.0, -7.0], [5.0, -5.0, 2.0]])


def test_lazy_scipy_and_struct6(S):
    assert np.array_equal(S._XTA_STORE_STRUCT6, ndimage.generate_binary_structure(3, 1))
    assert S._XTA_STORE_STRUCT6.dtype == bool
    assert type(S._xta_store_ndimage).__name__ == "_XTAStoreLazyModule"
    assert S._xta_store_ndimage.label is ndimage.label
    from scipy.sparse import csgraph
    assert S._xta_store_csgraph.connected_components is csgraph.connected_components
    with pytest.raises(AttributeError):
        S._xta_store_sparse._private_name


def test_preload_imports_the_flood_engine(S):
    pytest.importorskip("SimpleITK")
    S._xta_store_preload()


def test_defaults_are_spec_v31(S):
    T = S.XTACutoffThresholds()
    assert (T.huge_displayed, T.huge_looser, T.highlight_share, T.highlight_voxels, T.painted_kept,
            T.highlight_click_share, T.highlight_looser_share) == (500_000, 500_000, 0.25, 100_000, True, 0.002, 0.25)
    assert S._XTA_STORE_WORKERS == 4 and S._XTA_STORE_MAX_CUTOFFS == 101
    assert S._XTA_STORE_SCOPES == ("all", "stricter")


# --------------------------------------------------------------------------------------------
# Bit planes and the store build (G8, store part)


def test_bit_layouts(S, kit):
    rng = np.random.default_rng(5)
    for K, dt, words in ((5, np.uint16, 1), (16, np.uint16, 1), (17, np.uint32, 1), (33, np.uint64, 1),
                         (64, np.uint64, 1), (65, np.uint64, 2), (101, np.uint64, 2)):
        store = S.XTACutoffStore((6, 7, 8), K)
        masks = [rng.random((6, 7, 8)) < 0.3 for _ in range(K)]
        layer = kit.add_layer_from_masks(store, "B", "direct", masks)
        assert layer.storage == "bits" and layer.data.dtype == dt and layer.words == words, (K, layer.data.dtype)
        assert all(np.array_equal(layer.mask(k), masks[k]) for k in range(K))
        env = np.zeros((6, 7, 8), np.uint8)
        for k in range(K):
            env[masks[k]] = k + 1
        assert np.array_equal(layer.env(), env)
        assert np.array_equal(layer.counts, [m.sum() for m in masks])
        t = rng.integers(0, K + 1, (6, 7, 8)).astype(np.uint8)
        kept = S._xta_store_keep_below_array(layer.data, t, words)
        k0 = int(rng.integers(0, K))
        q = S._xta_store_rule_q(layer.data, k0, "bits", words)
        s = S._xta_store_rule1(layer.data, "stricter", k0, "bits", words)
        p = S._xta_store_or_const(layer.data, S._xta_store_low(k0 + 1), words)
        level = env.copy()
        b = S._xta_store_bits_from_level(level, K, K)
        for k in range(K):
            assert np.array_equal(S._xta_store_bit_test(kept, k, words), masks[k] & (k < t))
            assert np.array_equal(S._xta_store_bit_test(q, k, words), masks[k] & (k > k0))
            assert np.array_equal(S._xta_store_bit_test(s, k, words), masks[k] & (k < k0))
            assert np.array_equal(S._xta_store_bit_test(p, k, words), masks[k] | (k <= k0))
            assert np.array_equal(S._xta_store_bit_test(b, k, words), level > k)
        nested = [env > k for k in range(K)]                    # nested masks become a level map
        assert kit.add_layer_from_masks(store, "N", "direct", nested).storage == "level"
    with pytest.raises(ValueError):
        S.XTACutoffStore((2, 2, 2), 102)


@pytest.mark.parametrize("K", (5, 17, 65))
def test_store_reproduces_its_masks(S, kit, K):
    """G8 (store part): every mask a store was built from comes back; trusted counts equal scanned counts."""
    for seed in range(3):
        scene = sup.make_scene(kit, 9000 + seed + K, K)
        store = scene["store"]
        rebuilt = S.XTACutoffStore(store.shape, K)
        trusted = S.XTACutoffStore(store.shape, K)
        for lid, l in store.layers.items():
            masks = [l.mask(k) for k in range(K)]
            r = kit.add_layer_from_masks(rebuilt, lid, l.role, masks, l.origin, visible=l.visible, additive=l.additive)
            assert all(np.array_equal(r.mask(k), masks[k]) for k in range(K))
            nested = all(not (masks[k + 1] & ~masks[k]).any() for k in range(K - 1))
            assert r.storage == ("level" if nested else "bits")
            t = trusted.add_layer(S.XTACutoffLayer(lid, l.role, l.data.copy(), l.origin, visible=l.visible,
                                                   additive=l.additive), counts=[int(m.sum()) for m in masks])
            assert np.array_equal(t.counts, r.counts)
        assert kit.maps_consistent(rebuilt) and kit.counts_consistent(rebuilt) and kit.maps_consistent(trusted)
    with pytest.raises(ValueError):
        trusted.add_layer(S.XTACutoffLayer("X", "direct", np.zeros((1, 1, 1), np.uint8)), counts=[0])


# --------------------------------------------------------------------------------------------
# Hand-made scenes: G2-G5, the Global follow rule, prompts, scopes, fragments, painted seeds, the report


def follow_expected(kit, before, G_lid, all_before, all_after):
    """Where U_all fell, the Global's masks are cleared at cutoffs k >= U_all after (both storages)."""
    out = []
    fell = all_after < all_before
    for k in range(before.n_cutoffs):
        m = kit.full_mask(before, G_lid, k).copy()
        m[fell & (all_after <= k)] = False
        out.append(m)
    return out


@pytest.mark.parametrize("global_bits", (False, True))
def test_guarantees_on_hand_scene(S, kit, global_bits):
    base = sup.g_scene(kit, global_bits)
    K = base.n_cutoffs
    M = np.array([[8, 14, 30]])
    applied = 0
    for k0 in range(1, K - 1):
        s = kit.copy(base)
        before = {lid: kit.full_masks(base, lid) for lid in base.layers}
        st = islands(S, s, k0, M, **ACCEPT_ALL)
        if st["status"] != "applied":
            continue
        applied += 1
        for lid in ("A", "B", "D"):                      # G2: level maps lose exactly their k0-removed voxels
            rem = before[lid][k0] & ~kit.full_mask(s, lid, k0)
            for k in range(k0 + 1, K):
                assert np.array_equal(kit.full_mask(s, lid, k), before[lid][k] & ~rem), (lid, k)
        assert np.array_equal(s.layers["H"].data, base.layers["H"].data)          # G3: hidden untouched
        remD = before["D"][k0] & ~kit.full_mask(s, "D", k0)
        for k in range(K):                                # non-additive derived: its C_l only
            assert np.array_equal(kit.full_mask(s, "D", k), before["D"][k] & ~remD)
        exp = follow_expected(kit, base, "G", base.U_all, s.U_all)
        assert all(np.array_equal(kit.full_mask(s, "G", k), exp[k]) for k in range(K)), "Global follow"
        assert s.global_stale == bool((s.U_all < base.U_all).any())
        assert np.all(kit.full_mask(s, "G", 0)[0:2, 0:3, 36:40]), "Global-only voxels untouched"
        for lid, l in s.layers.items():                   # G4: level maps stay nested
            if l.storage == "level":
                m = kit.full_masks(s, lid)
                assert all(not (m[k + 1] & ~m[k]).any() for k in range(K - 1))
        snap, n = kit.snapshot(s), len(s.journal)        # G5: a second identical click changes nothing
        st2 = islands(S, s, k0, M, **ACCEPT_ALL)
        assert st2["status"] == "no_hit" and kit.same_state(s, snap) and len(s.journal) == n
        assert kit.maps_consistent(s) and kit.counts_consistent(s)
        for lid, row in st["report"]["layers"].items():  # the report comes from the counts
            d = [int(before[lid][k].sum() - kit.full_mask(s, lid, k).sum()) for k in range(K)]
            assert row["removed_by_cutoff"] == d and row["removed_displayed"] == d[k0]
            assert row["removed_looser"] == sum(d[:k0]) and row["removed_stricter"] == sum(d[k0 + 1:])
            assert abs(row["share"] - d[k0] / before[lid][k0].sum()) < 1e-12
    assert applied >= 2


@pytest.mark.parametrize("bits", (False, True))
def test_halo_removed_with_one_visible_layer(S, kit, bits):
    """The union basin removes an island's looser halo at every cutoff; a stricter-only bit stays."""
    K, k0 = 6, 3
    shape = (12, 20, 20)
    grid = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij")
    score = np.maximum(sup.gauss(grid, (6, 10, 10), (2, 2, 2), 0.8), sup.gauss(grid, (6, 10, 10), (5, 6, 6), 0.3))
    lev = np.searchsorted(np.linspace(0.1, 0.9, K), score, side="right").astype(np.uint8)
    data, v = lev, (6, 10, 16)
    if bits:
        masks = [lev > k for k in range(K)]
        assert 1 <= lev[v] <= k0
        masks[k0 + 2][v] = True                          # bits {0, ..., k0 + 2}: env > k0 but not shown
        data = S._xta_store_pack_bit_planes(masks, K)
    s = kit.build_store(shape, K, [dict(lid="A", role="direct", data=data)])
    ref = kit.copy(s)
    st = islands(S, s, k0, np.array([[6, 10, 10]]))
    kit.remove_islands_full_grid(ref, k0, np.array([[6, 10, 10]]), engine=ENGINE)
    assert st["status"] == "applied" and kit.same_content(s, ref)
    for k in range(k0 + 1):
        assert not kit.full_mask(s, "A", k).any(), k
    if bits:
        assert kit.full_mask(s, "A", k0 + 2)[v]


def test_marker_messages(S, kit):
    s = sup.g_scene(kit)
    snap = kit.snapshot(s)
    st = islands(S, s, 2, np.array([[0, 0, 38]]))         # Global-only voxel
    assert st["status"] == "no_hit" and st["hidden_or_global_layers"] == ["G"] and "Global" in st["message"]
    hid = np.argwhere(kit.full_mask(s, "H", 2) & ~kit.shown(s, 2))
    if len(hid):
        st = islands(S, s, 2, hid[:1])
        assert st["status"] == "no_hit" and "H" in st["hidden_or_global_layers"]
    st = islands(S, s, 2, np.array([[15, 27, 0]]))
    assert st["status"] == "no_hit" and st["message"] == S._XTA_STORE_NO_HIT_MESSAGE
    st = islands(S, s, 2, np.array([[-1, 3, 3], [3, 3, 99]]))
    assert st["status"] == "no_hit" and st["markers_off_grid"] == 2
    assert slices(S, s, 2, np.array([[15, 27, 0]]))["status"] == "no_hit"
    assert kit.same_state(s, snap) and not s.journal
    with pytest.raises(ValueError):
        islands(S, s, 2, np.array([[8, 14, 30]]), scope="displayed")
    with pytest.raises(ValueError):
        islands(S, s, 6, np.array([[8, 14, 30]]))


def test_huge_prompts(S, kit):
    base = sup.g_scene(kit)
    M = np.array([[8, 14, 10]])                                  # the body
    T = S.XTACutoffThresholds(huge_displayed=200, huge_looser=50)
    s = kit.copy(base)
    st = islands(S, s, 2, M, thresholds=T)                      # default answer: Cancel
    assert st["status"] == "cancelled_huge_displayed" and kit.same_state(s, kit.snapshot(base)) and not s.journal
    info = st["prompts"][0]
    assert info["kind"] == "huge_displayed" and info["voxels_found"] > 200 and info["layers"] and "estimate_s" in info
    st = islands(S, s, 2, M, thresholds=T, confirm_huge_displayed=lambda i: "remove",
                 confirm_huge_looser=lambda i: "cancel")
    assert st["answers"]["huge_displayed"] == "remove"
    assert st["status"] == "applied" if "huge_looser" not in st["answers"] else not s.journal
    early = islands(S, kit.copy(base), 2, M, thresholds=S.XTACutoffThresholds(huge_displayed=50), margin=2)
    full = islands(S, kit.copy(base), 2, M, margin=2, **ACCEPT_ALL)
    assert early["status"] == "cancelled_huge_displayed"
    assert early["prompts"][0]["voxels_found"] <= full["displayed_total"]
    # huge looser claim: the default answer equals the Displayed + stricter scope
    isl = np.array([[8, 14, 30]])
    T2 = S.XTACutoffThresholds(huge_looser=5)
    a = kit.copy(base)
    sa = islands(S, a, 1, isl, thresholds=T2)
    assert sa["answers"].get("huge_looser") == "displayed_stricter"
    b = kit.copy(base)
    islands(S, b, 1, isl, scope="stricter")
    assert kit.same_content(a, b)
    d, e = kit.copy(base), kit.copy(base)
    islands(S, d, 1, isl, thresholds=T2, confirm_huge_looser=lambda i: "remove_all")
    islands(S, e, 1, isl)
    assert kit.same_content(d, e)
    for x, cb in ((a, {}), (d, dict(confirm_huge_looser=lambda i: "remove_all"))):
        ref = kit.copy(base)
        kit.remove_islands_full_grid(ref, 1, isl, engine=ENGINE, thresholds=T2, **cb)
        assert kit.same_content(x, ref)


@pytest.mark.parametrize("global_bits", (False, True))
def test_displayed_stricter_scope(S, kit, global_bits):
    base = sup.g_scene(kit, global_bits)
    K = base.n_cutoffs
    for k0 in range(1, K - 1):
        s = kit.copy(base)
        before = {lid: kit.full_masks(base, lid) for lid in ("A", "B", "D")}
        st = islands(S, s, k0, np.array([[8, 14, 30]]), scope="stricter")
        if st["status"] != "applied":
            continue
        assert st["Q"] == 0 and st["Q_minus_F"] == 0
        for lid in ("A", "B", "D"):
            rem = before[lid][k0] & ~kit.full_mask(s, lid, k0)
            lev_b, lev_a = kit.full_env(base, lid), kit.full_env(s, lid)
            assert np.array_equal(lev_a, np.where(rem, np.minimum(lev_b, k0), lev_b)), (lid, k0)
        ref = kit.copy(base)
        kit.remove_islands_full_grid(ref, k0, np.array([[8, 14, 30]]), scope="stricter", engine=ENGINE)
        assert kit.same_content(s, ref)


def test_bit_fragments(S, kit):
    """G2 (bit planes): a bit-plane layer also loses the stricter-only fragments next to its island, only them."""
    K = 8
    shape = (10, 16, 24)
    lev = np.zeros(shape, np.uint8)
    lev[4:7, 6:10, 4:8] = 5                                   # island (shown at k0 = 2)
    lev[4:7, 6:10, 16:20] = 5                                 # a second island
    masks = [lev > k for k in range(K)]
    for k in (4, 5, 6):                                       # stricter-only fragment touching island 1
        masks[k][5, 7:9, 8:11] = True
    for k in (5, 6):                                          # stricter-only voxels touching nothing
        masks[k][5, 8, 13] = True
    masks[1][5, 7, 5] = False                                 # a looser hole
    data = S._xta_store_pack_bit_planes(masks, K)
    s = kit.build_store(shape, K, [dict(lid="B", role="direct", data=data)])
    ref = kit.copy(s)
    st = islands(S, s, 2, np.array([[5, 7, 5]]))
    kit.remove_islands_full_grid(ref, 2, np.array([[5, 7, 5]]), engine=ENGINE)
    assert kit.same_content(s, ref) and st["fragments"] == {"B": 6}, st["fragments"]
    for k in range(K):
        m = kit.full_mask(s, "B", k)
        assert not m[4:7, 6:10, 4:8].any() and not m[5, 7:9, 8:11].any()
        assert np.array_equal(m[4:7, 6:10, 16:20], masks[k][4:7, 6:10, 16:20])
        assert m[5, 8, 13] == masks[k][5, 8, 13]
    s2 = kit.build_store(shape, K, [dict(lid="B", role="direct", data=data)])
    islands(S, s2, 2, np.array([[5, 7, 5]]), scope="stricter")
    for k in range(K):
        exp = masks[k].copy()
        if k >= 2:
            exp[4:7, 6:10, 4:8] = False
            exp[5, 7:9, 8:11] = False
        assert np.array_equal(kit.full_mask(s2, "B", k), exp), k


def test_painted_kept_seed_and_sync(S, kit):
    K = 6
    shape = (10, 12, 30)
    lev = np.zeros(shape, np.uint8)
    lev[4:7, 4:8, 2:8] = 5                                   # island A (marker)
    lev[4:7, 4:8, 8:22] = 1                                  # a looser corridor
    lev[4:7, 4:8, 22:28] = 5                                 # island B (kept)
    s = kit.build_store(shape, K, [dict(lid="A", role="direct", data=lev),
                                   dict(lid="P", role="direct", data=np.zeros(shape, np.uint8))])
    S._xta_store_sync_paint(s, "P", 0, (np.array([5, 5, 12]), np.ones((1, 1, 1), bool)))  # painted at k0 = 0
    assert np.array_equal(s.painted["P"], [np.ravel_multi_index((5, 5, 12), shape)])
    assert kit.maps_consistent(s) and kit.counts_consistent(s)
    a, b = kit.copy(s), kit.copy(s)
    sa = islands(S, a, 2, np.array([[5, 5, 4]]))
    sb = islands(S, b, 2, np.array([[5, 5, 4]]), thresholds=S.XTACutoffThresholds(painted_kept=False))
    ref = kit.copy(s)
    kit.remove_islands_full_grid(ref, 2, np.array([[5, 5, 4]]), engine=ENGINE)
    assert kit.same_content(a, ref)
    assert sa["painted_kept"] == 1 and sa["Q"] < sb["Q"]
    assert kit.full_mask(a, "P", 0)[5, 5, 12] and kit.full_mask(a, "A", 0)[5, 5, 13]  # the painted seed's side
    S._xta_store_sync_erase(a, "P", 0, np.array([[5, 5, 12]]))      # erase leaves the painted set
    assert "P" not in a.painted and kit.maps_consistent(a) and kit.counts_consistent(a)


def test_sync_box_growth_erase_follow_and_undo(S, kit):
    base = sup.g_scene(kit)
    s = kit.copy(base)
    snap0 = kit.snapshot(s)
    o0, h0 = s.layers["B"].origin.copy(), s.layers["B"].hi.copy()
    st = S._xta_store_sync_paint(s, "B", 2, np.array([[0, 0, 0], [15, 27, 39]]))
    assert st["status"] == "applied" and st["painted"] == 2
    assert np.all(s.layers["B"].origin == 0) and np.all(s.layers["B"].hi == s.shape)    # the box grew
    assert all(kit.full_mask(s, "B", k)[0, 0, 0] == (k <= 2) for k in range(s.n_cutoffs))  # paint: 0..k0
    assert kit.maps_consistent(s) and kit.counts_consistent(s) and not s.global_stale
    snap1 = kit.snapshot(s)
    before = kit.copy(s)
    st = S._xta_store_sync_erase(s, "A", 2, (np.array([4, 8, 4]), np.ones((8, 12, 12), bool)))
    assert st["status"] == "applied" and kit.maps_consistent(s) and kit.counts_consistent(s)
    assert not kit.full_mask(s, "A", 0)[4:12, 8:20, 4:16].any(), "erase clears every cutoff"
    exp = follow_expected(kit, before, "G", before.U_all, s.U_all)
    assert all(np.array_equal(kit.full_mask(s, "G", k), exp[k]) for k in range(s.n_cutoffs)), "erase: Global follow"
    assert s.global_stale == bool((s.U_all < before.U_all).any())
    snap2 = kit.snapshot(s)
    assert s.undo() and kit.same_state(s, snap1)
    assert s.undo() and kit.same_state(s, snap0)
    assert np.array_equal(s.layers["B"].origin, o0) and np.array_equal(s.layers["B"].hi, h0)   # box shrinks back
    assert s.redo() and kit.same_state(s, snap1)
    assert s.redo() and kit.same_state(s, snap2)
    assert not s.redo()
    nothing = S._xta_store_sync_edit(s, "A", 2)
    assert nothing["status"] == "no_change" and len(s.journal) == 2


def test_report_lists_changed_layers(S):
    """A sync whose erase and paint cancel in the counts is still reported."""
    s = S.XTACutoffStore((1, 1, 2), 4)
    s.add_layer(S.XTACutoffLayer("A", "direct", np.array([[[3, 0]]], np.uint8)))
    st = S._xta_store_sync_edit(s, "A", 2, erase=np.array([[0, 0, 0]]), paint=np.array([[0, 0, 1]]))
    assert st["status"] == "applied" and s.layers["A"].data.ravel().tolist() == [0, 3]
    row = st["report"]["layers"]["A"]
    assert row["removed_by_cutoff"] == [0, 0, 0, 0] and st["report"]["highlight"] == []


def test_slices_nonadditive_footprint(S, kit):
    """A visible non-additive derived layer takes no part in Remove Slices step 2."""
    K, k0 = 4, 1
    a = np.array([[[3, 3, 3, 3, 0, 0, 0], [3, 3, 0, 3, 1, 1, 0], [0, 0, 0, 0, 0, 3, 3]]], np.uint8)
    d = np.array([[[0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], np.uint8)
    for with_dn in (True, False):
        s = S.XTACutoffStore(a.shape, K)
        s.add_layer(S.XTACutoffLayer("A", "direct", a.copy()))
        if with_dn:
            s.add_layer(S.XTACutoffLayer("Dn", "derived", d.copy(), additive=False))
        ref = kit.copy(s)
        slices(S, s, k0, np.array([[0, 1, 0]]))
        kit.remove_slices_full_grid(ref, k0, np.array([[0, 1, 0]]), engine=ENGINE)
        assert kit.full_data(s, "A")[0, 1].tolist() == [0, 0, 0, 0, 0, 0, 0]
        assert kit.same_content(s, ref)
        if with_dn:
            assert not kit.full_mask(s, "Dn", 0)[0, 1, :4].any()          # Dn's own cross section is written


def test_report_click_and_looser_flags(S, kit):
    base = sup.g_scene(kit)
    M = np.array([[8, 14, 30]])
    st_default = islands(S, kit.copy(base), 1, M)
    st = islands(S, kit.copy(base), 1, M,
                 thresholds=S.XTACutoffThresholds(highlight_click_share=None, highlight_looser_share=None))
    click = st["report"]["click"]
    assert st_default["report"]["click"]["share"] == click["share"]
    assert st_default["report"]["click"]["flagged"] == (click["removed_displayed"] > 0 and click["share"] >= 0.002)
    assert set(st_default["report"]["highlight_looser"]) == {
        lid for lid, row in st["report"]["layers"].items() if row["looser_share"] >= 0.25}
    V = base.visible_lids()
    assert click["displayed_before"] == sum(int(base.layers[l].counts[1]) for l in V)
    assert click["removed_displayed"] == sum(st["report"]["layers"][l]["removed_displayed"]
                                             for l in V if l in st["report"]["layers"])
    assert not click["flagged"] and st["report"]["highlight_looser"] == []
    T = S.XTACutoffThresholds(highlight_click_share=click["share"] * 0.99, highlight_looser_share=0.0)
    st2 = islands(S, kit.copy(base), 1, M, thresholds=T)
    assert st2["report"]["click"]["flagged"] and set(st2["report"]["highlight_looser"]) == set(st2["report"]["layers"])
    T3 = S.XTACutoffThresholds(highlight_click_share=min(1.0, click["share"] * 1.5 + 1e-9))
    assert not islands(S, kit.copy(base), 1, M, thresholds=T3)["report"]["click"]["flagged"]
    body = islands(S, kit.copy(base), 2, np.array([[8, 14, 10]]), **ACCEPT_ALL)
    assert body["report"]["click"]["flagged"] and body["report"]["highlight"]


def test_changed_boxes(S, kit):
    """last_changed_boxes / entry.boxes hold every changed voxel of each layer (edit, undo, redo)."""
    base = sup.g_scene(kit, global_bits=True)
    s = kit.copy(base)
    islands(S, s, 1, np.array([[8, 14, 30]]))
    boxes = dict(s.last_changed_boxes)
    assert boxes and boxes == s.journal[-1].boxes and "G" in boxes
    for lid in s.layers:
        diff = S._xta_store_neq(kit.full_data(base, lid), kit.full_data(s, lid), s.layers[lid].words)
        if diff.any():
            lo, hi = boxes[lid]
            inside = np.zeros(s.shape, bool)
            inside[S._xta_store_sl(lo, hi)] = True
            assert not (diff & ~inside).any(), lid
        else:
            assert lid not in boxes, lid
    assert s.undo() and s.last_changed_boxes == boxes
    assert s.redo() and s.last_changed_boxes == boxes


def test_entry_builder_replace(S, kit):
    """Another store operation (a Global replacement with bit planes, a layer write) shares the undo stack."""
    s = kit.copy(sup.g_scene(kit))
    islands(S, s, 2, np.array([[8, 14, 30]]))
    snap1 = kit.snapshot(s)
    K = s.n_cutoffs
    G = s.layers["G"]
    masks = [kit.full_mask(s, "G", k) for k in range(K)]
    masks[3][8, 14, 10] = False                              # stops nesting: bit planes
    eb = s.begin_entry("update_global", 3)
    eb.replace_layer("G", S._xta_store_pack_bit_planes(masks, K), (0, 0, 0))
    eb.set_stale(False)
    eb.commit()
    assert s.layers["G"].storage == "bits" and not s.global_stale and kit.counts_consistent(s)
    snap2 = kit.snapshot(s)
    eb = s.begin_entry("update_nonglobal", 3)
    m = np.zeros((1, 2, 2), bool)
    m[0, 0, 0] = m[0, 1, 1] = True
    eb.write("A", (8, 14, 10), m, np.zeros(2, np.uint8))
    eb.commit()
    assert kit.maps_consistent(s) and kit.counts_consistent(s)
    snap3 = kit.snapshot(s)
    assert s.undo() and kit.same_state(s, snap2)
    assert s.undo() and kit.same_state(s, snap1) and G is s.layers["G"] and G.storage == "level"
    assert s.redo() and kit.same_state(s, snap2)
    assert s.redo() and kit.same_state(s, snap3)
    assert s.begin_entry("nothing").commit() is None and len(s.journal) == 3


def test_set_visible_many(S, kit):
    base = sup.g_scene(kit)
    a, b = kit.copy(base), kit.copy(base)
    flags = {"A": False, "H": True, "D": False}
    for lid, flag in flags.items():
        a.set_visible(lid, flag)
    assert b.set_visible_many(flags) is True
    assert np.array_equal(a.U_vis, b.U_vis) and kit.maps_consistent(b) and b.visible_lids() == a.visible_lids()
    assert b.set_visible_many({"D": True}) is False          # non-additive: U_vis unchanged
    with pytest.raises(KeyError):
        b.set_visible_many({"A": True, "missing": False})
    assert not b.layers["A"].visible                          # unknown ids raise before any change


# --------------------------------------------------------------------------------------------
# Undo: byte cap, plain-data spill files, a re-exec of the file


def test_undo_cap_and_spill(S, kit):
    s = kit.copy(sup.g_scene(kit))
    snaps = [kit.snapshot(s)]
    islands(S, s, 2, np.array([[8, 14, 30]]), **ACCEPT_ALL)
    snaps.append(kit.snapshot(s))
    S._xta_store_sync_paint(s, "B", 1, (np.array([2, 2, 30]), np.ones((3, 3, 3), bool)))
    snaps.append(kit.snapshot(s))
    islands(S, s, 1, np.array([[8, 14, 10]]), **ACCEPT_ALL)
    snaps.append(kit.snapshot(s))
    sizes = [e.nbytes for e in s.journal]
    assert len(sizes) == 3 and all(v > 0 for v in sizes)
    s.undo_bytes = sizes[1] + sizes[2]                     # the oldest entry drops first
    s._enforce_cap()
    assert s.dropped_entries == 1 and len(s.journal) == 2 and s.journal_nbytes() <= s.undo_bytes
    s.undo_bytes = sizes[2] - 1                            # the newest alone is over the cap: spilled
    s._enforce_cap()
    assert s.dropped_entries == 2 and len(s.journal) == 1 and s.journal[0].spill_path is not None
    path = s.journal[0].spill_path
    assert s.journal_nbytes() == 0 and os.path.exists(path)
    assert s.undo() and kit.same_state(s, snaps[2]) and not os.path.exists(path)
    assert not s.undo()                                     # the dropped entries are gone
    s.undo_bytes = 2 * S._XTA_STORE_GIB
    assert s.redo() and kit.same_state(s, snaps[3])
    s.close()


def test_undo_near_cap_keeps_steps(S, kit):
    """An undo near the cap spills instead of dropping an older entry."""
    def build(cap):
        s = S.XTACutoffStore((6, 6, 6), 3, undo_bytes=cap)
        s.add_layer(S.XTACutoffLayer("A", "direct", np.zeros((6, 6, 6), np.uint8)))
        rng = np.random.default_rng(0)
        snaps = [kit.snapshot(s)]
        S._xta_store_sync_paint(s, "A", 1, np.argwhere(rng.random((6, 6, 6)) < 0.5))
        snaps.append(kit.snapshot(s))
        S._xta_store_sync_paint(s, "A", 2, np.argwhere(rng.random((6, 6, 6)) < 0.5))
        snaps.append(kit.snapshot(s))
        S._xta_store_sync_erase(s, "A", 0, np.argwhere(rng.random((6, 6, 6)) < 0.5))
        snaps.append(kit.snapshot(s))
        return s, snaps
    probe, _ = build(1 << 30)
    s, snaps = build(probe.journal_nbytes())
    assert len(s.journal) == 3 and s.dropped_entries == 0
    for i in (2, 1, 0):
        assert s.undo() and kit.same_state(s, snaps[i]) and s.dropped_entries == 0, i
        assert s.journal_nbytes() <= s.undo_bytes
    assert not s.undo()
    for i in (1, 2, 3):
        assert s.redo() and kit.same_state(s, snaps[i]), i
    s.close()


def test_spill_files_hold_plain_data_and_go_with_the_store(S, kit, tmp_path):
    import marshal

    def make():
        s = S.XTACutoffStore((4, 4, 4), 3, undo_bytes=1, spill_dir=str(tmp_path))
        s.add_layer(S.XTACutoffLayer("A", "direct", np.zeros((4, 4, 4), np.uint8)))
        S._xta_store_sync_paint(s, "A", 1, np.argwhere(np.ones((2, 2, 2), bool)))
        S._xta_store_sync_paint(s, "A", 2, np.argwhere(np.ones((3, 3, 3), bool)))
        s.undo()
        return s
    s = make()
    files = os.listdir(tmp_path)
    assert len(files) == len(s._spill_files) >= 1
    for name in files:
        with open(tmp_path / name, "rb") as f:              # no pickle: plain tuples, bytes, str and int
            head = marshal.load(f)
            changes = [marshal.load(f) for _ in range(head[1])]
            assert f.read() == b""
        assert head[0] == S._XTA_STORE_SPILL_FORMAT and changes and all(type(c) is tuple for c in changes)
    c = kit.copy(s)                                         # the copy holds its entries in memory
    s.close()
    assert not os.listdir(tmp_path) and not s.journal and not s.redo_stack
    assert c.redo() and c.undo() and c.redo()
    c.close()
    assert not os.listdir(tmp_path)
    s = make()
    assert os.listdir(tmp_path)
    del s
    gc.collect()
    assert not os.listdir(tmp_path)


def test_reexec_keeps_the_store(kit, tmp_path):
    """A re-exec of the file redefines the classes; adopt() rebinds a live store and its undo history."""
    S = sup.load_store_module(sup.MODULE_NAME + "_reexec")
    try:
        k1 = sup.StoreKit(S)
        s = k1.build_store(*hand_scene_specs(k1), spill_dir=str(tmp_path))
        snaps = [k1.snapshot(s)]
        islands(S, s, 2, np.array([[8, 14, 30]]), **ACCEPT_ALL)
        snaps.append(k1.snapshot(s))
        S._xta_store_sync_paint(s, "B", 1, (np.array([2, 2, 30]), np.ones((3, 3, 3), bool)))
        snaps.append(k1.snapshot(s))
        for e in s.journal:                                          # spilled to plain-data files
            s._spill(e)
        S._xta_store_sync_paint(s, "A", 2, np.array([[0, 0, 0]]))   # in memory, with a packed painted set
        snaps.append(k1.snapshot(s))
        assert any(e.spill_path for e in s.journal) and any(not e.spill_path for e in s.journal)
        old_class = type(s)
        sup.reexec_section(S)
        assert type(s) is not S.XTACutoffStore and old_class is not S.XTACutoffStore
        assert S._xta_store_adopt(s) is s and type(s) is S.XTACutoffStore
        assert all(type(l) is S.XTACutoffLayer for l in s.layers.values())
        assert all(type(e) is S.XTACutoffJournalEntry for e in s.journal)
        k2 = sup.StoreKit(S)
        for i in (2, 1, 0):
            assert s.undo() and k2.same_state(s, snaps[i]), i
        for i in (1, 2, 3):
            assert s.redo() and k2.same_state(s, snaps[i]), i
        islands(S, s, 1, np.array([[8, 14, 10]]), **ACCEPT_ALL)     # the adopted store keeps working
        assert k2.maps_consistent(s) and k2.counts_consistent(s)
        s.close()
        assert not os.listdir(tmp_path)
        foreign = old_class.__new__(old_class)
        foreign.store_format = S._XTA_STORE_FORMAT + 1
        assert S._xta_store_adopt(foreign) is None
    finally:
        sup.unload_store_module(S)


def hand_scene_specs(kit):
    """(shape, K, specs) of the hand scene, for build_store with other store options."""
    store = sup.g_scene(kit)
    specs = [dict(lid=lid, role=l.role, additive=l.additive if l.role == "derived" else None,
                  visible=l.visible, data=kit.full_data(store, lid)) for lid, l in store.layers.items()]
    return store.shape, store.n_cutoffs, specs


# --------------------------------------------------------------------------------------------
# G1: today at the displayed cutoff


def oriented_matrix(rng):
    a = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ rot @ np.diag(rng.uniform(0.5, 2.0, 3))
    m[:3, 3] = rng.uniform(-50, 50, 3)
    return m


def today_markers(kit, rng, store, k0, m):
    """Random points, lines and closed curves (segmentation RAS through m) near shown voxels."""
    shape = np.array(store.shape)
    U = kit.shown(store, k0)
    to_ras = lambda zyx: (m @ np.append(np.asarray(zyx, np.float64)[::-1], 1.0))[:3]   # noqa: E731
    pts, lines, curves = [], [], []
    for _ in range(int(rng.integers(0, 3))):
        p = sup.pick(rng, U)
        if p is not None:
            pts.append(to_ras(p + rng.uniform(-0.45, 0.45, 3)))
    if rng.random() < 0.6:
        a = sup.pick(rng, U)
        if a is not None:
            lines.append((to_ras(a + rng.uniform(-0.5, 0.5, 3)), to_ras(rng.uniform(-3, shape + 3))))
    if rng.random() < 0.6:
        a = sup.pick(rng, U)
        if a is not None:
            r = rng.uniform(1.0, 7.0)
            ax = int(rng.integers(0, 3))
            o = [i for i in range(3) if i != ax]
            c = []
            for t in np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 9)))):
                p = a.astype(np.float64) + rng.uniform(-0.4, 0.4, 3)
                p[o[0]] += r * np.cos(t)
                p[o[1]] += r * np.sin(t)
                p[ax] += rng.uniform(-1.0, 1.0)
                c.append(to_ras(p))
            curves.append(c)
    if not (pts or lines or curves):
        p = sup.pick(rng, U)
        pts.append(to_ras(p if p is not None else shape // 2))
    return pts, lines, curves


@pytest.fixture(scope="module")
def today_ns():
    return sup.today_job_namespace()


@pytest.mark.parametrize("seed", range(12))
def test_today_at_the_displayed_cutoff(S, kit, today_ns, seed):
    """G1: with every prompt answered Remove, each visible layer's mask at k0 equals today's Remove Islands and
    today's Remove Slices (points, lines and closed curves; identity and oriented geometry; some synced edits)."""
    K = (5, 17, 65)[seed % 3]
    scene = sup.make_scene(kit, 20000 + seed, K)
    base = scene["store"]
    rng = np.random.default_rng([seed, 99])
    if seed % 3 == 0:
        sup.random_syncs(kit, base, rng, 3)
    usable = [k for k in range(K) if kit.shown(base, k).any()]
    assert usable
    z, y, x = base.shape
    extent = (0, x - 1, 0, y - 1, 0, z - 1)
    for case in range(2):
        k0 = int(rng.choice(usable))
        m = np.eye(4) if (seed + case) % 2 else oriented_matrix(rng)
        pts, lines, curves = today_markers(kit, rng, base, k0, m)
        markers = S._xta_store_rasterise_markers(base.shape, points_ras=pts, lines_ras=lines, curves_ras=curves,
                                                 image_to_world=m)
        Vp = base.visible_prime_lids()
        for remove_slices in (False, True):
            s = kit.copy(base)
            (slices if remove_slices else islands)(S, s, k0, markers, thresholds=small_thresholds(S), **ACCEPT_ALL)
            for lid in base.layers:
                if lid in Vp:
                    expected = sup.today_job(today_ns, kit.full_mask(base, lid, k0), extent, m, pts, lines, curves,
                                             remove_slices=remove_slices)
                    after = kit.full_mask(s, lid, k0)
                    assert np.array_equal(after, expected), (lid, k0, remove_slices, int((after ^ expected).sum()))
                elif base.layers[lid].role != "global":
                    assert np.array_equal(s.layers[lid].data, base.layers[lid].data), lid
            assert kit.maps_consistent(s) and kit.counts_consistent(s)


# --------------------------------------------------------------------------------------------
# G7: windowed edits == the full-grid rules; G5 and G6 on random scenes


WINDOW_VARIANTS = (("default", {}, {}), ("margin 2", dict(margin=2), {}),
                   ("threaded", dict(workers=4), dict(_XTA_STORE_PARALLEL_VOXELS=-1, _XTA_STORE_STEP1_SWITCH_WORK=-1)),
                   ("region stop", dict(margin=24, cap=1, slab=5), {}),
                   ("slab labelled", dict(margin=2, slab=3), dict(_XTA_STORE_DIRECT_LABEL_LIMIT=0)))


@pytest.mark.parametrize("K,seed", [(K, seed) for K in (5, 17, 65) for seed in range(8)])
def test_windowed_equals_full_grid(S, kit, monkeypatch, K, seed):
    scene = sup.make_scene(kit, 1000 * K + seed, K)
    base = scene["store"]
    srng = np.random.default_rng([seed, 17])
    if seed % 2 == 0:
        sup.random_syncs(kit, base, srng, int(srng.integers(1, 4)))
    assert kit.maps_consistent(base) and kit.counts_consistent(base)
    for index in range(2):
        k0, markers, kind = sup.make_case(kit, scene, index)
        cb = CALLBACK_SETS[(seed + index) % len(CALLBACK_SETS)]
        T = small_thresholds(S) if (seed + index) % 2 else None
        scope = "stricter" if (seed + index) % 5 == 4 else "all"
        for remove_slices in (False, True):
            full = kit.remove_slices_full_grid if remove_slices else kit.remove_islands_full_grid
            ref = kit.copy(base)
            key_ref = sup.stats_key(full(ref, k0, markers, scope=scope, engine=ENGINE, thresholds=T, **cb))
            for v, (label, kw, tunables) in enumerate(WINDOW_VARIANTS):
                if (v + seed + index) % 2 and v:              # every variant runs, not on every click
                    continue
                if remove_slices and "cap" in kw:
                    continue
                w = kit.copy(base)
                snap0, n0 = kit.snapshot(w), len(w.journal)
                with monkeypatch.context() as mp:
                    for name, value in tunables.items():
                        mp.setattr(S, name, value)
                    st = (slices if remove_slices else islands)(S, w, k0, markers, scope=scope, thresholds=T,
                                                                **cb, **kw)
                assert sup.stats_key(st) == key_ref, (label, kind, remove_slices)
                assert kit.same_content(w, ref), (label, kind, remove_slices)
                assert kit.maps_consistent(w) and kit.counts_consistent(w), label
                if v == 0:
                    snap1, n1 = kit.snapshot(w), len(w.journal)
                    st2 = (slices if remove_slices else islands)(S, w, k0, markers, scope=scope, thresholds=T, **cb)
                    if st["status"] == "applied":            # G5: a second identical click changes nothing
                        assert kit.same_state(w, snap1) and len(w.journal) == n1, st2["status"]
                    while len(w.journal) > n0:                # G6
                        w.undo()
                    assert kit.same_state(w, snap0)
                    while w.redo():
                        pass
                    assert kit.same_content(w, ref)


@pytest.mark.parametrize("k0", (1, 3))
def test_window_growth_and_region_stop_equal_full_grid(S, kit, monkeypatch, k0):
    """The step-2 window grows face by face, passes the pessimistic re-flood check, or falls back to the exact
    region stop; step 1 also labels in z-slabs. Every path gives the full-grid result."""
    base = sup.g_scene(kit)
    M = np.array([[8, 14, 30]])
    ref = kit.copy(base)
    key = sup.stats_key(kit.remove_islands_full_grid(ref, k0, M, engine=ENGINE, **ACCEPT_ALL))
    seen = set()
    for kw, tunables in ((dict(margin=1), {}), (dict(margin=24, cap=1, slab=5), {}),
                         (dict(margin=1, slab=3), dict(_XTA_STORE_DIRECT_LABEL_LIMIT=0))):
        w = kit.copy(base)
        with monkeypatch.context() as mp:
            for name, value in tunables.items():
                mp.setattr(S, name, value)
            st = islands(S, w, k0, M, **ACCEPT_ALL, **kw)
        assert sup.stats_key(st) == key and kit.same_content(w, ref), kw
        win = st["step2_window"]
        seen.update(name for name, hit in (("grown", win["rounds"] > 1), ("re-flood", win["face_checks"] > 0),
                                           ("region stop", win["fallback"])) if hit)
    assert seen == {"grown", "re-flood", "region stop"}


def test_engines_agree(S):
    engines = ["python", "skimage"]
    try:
        import SimpleITK  # noqa: F401
        engines.append("sitk")
    except ImportError:
        pass
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(40):
        shape = (int(rng.integers(3, 8)), int(rng.integers(3, 10)), int(rng.integers(3, 10)))
        D = rng.random(shape) < 0.8
        Ub = (rng.integers(0, 6, shape) * D).astype(np.uint8)
        D &= Ub > 0
        F = D & (rng.random(shape) < 0.08)
        Ua = Ub.copy()
        Ua[F] = 0
        kept = D & ~F & (rng.random(shape) < 0.1)
        if not F.any():
            continue
        Dsel, _, _ = S._xta_store_select_components(Ub > 0, np.argwhere(F))
        p, mk, _ = S._xta_store_priorities(Ub, Ua, F, kept, Dsel)
        out = [S._xta_store_claim(p, mk, Dsel, kept, e) for e in engines]
        assert all(np.array_equal(out[0], o) for o in out[1:])
        compared += 1
    assert compared > 20
    with pytest.raises(ValueError):
        S._xta_store_flood(p, mk, Dsel, engine="unknown")


@pytest.mark.parametrize("seed", range(6))
def test_random_chains_undo_redo(S, kit, seed):
    """G6: chains of clicks, syncs, eye changes, undo and redo; every undo / redo against snapshots."""
    scene = sup.make_scene(kit, 50000 + seed, (5, 17, 65, 11, 101, 33)[seed])
    s = scene["store"]
    rng = np.random.default_rng([seed, 5])
    stack, redo = [kit.snapshot(s)], []
    for step in range(12):
        u = rng.random()
        if u < 0.18 and s.journal:
            assert s.undo()
            redo.append(stack.pop())
            assert kit.same_state(s, stack[-1], maps=False) and kit.maps_consistent(s) and kit.counts_consistent(s)
            continue
        if u < 0.3 and s.redo_stack:
            assert s.redo()
            stack.append(redo.pop())
            assert kit.same_state(s, stack[-1], maps=False) and kit.maps_consistent(s) and kit.counts_consistent(s)
            continue
        if u < 0.36:
            lid = list(s.layers)[int(rng.integers(0, len(s.layers)))]
            if s.layers[lid].role != "global":
                s.set_visible(lid, not s.layers[lid].visible)
            continue
        n_before = len(s.journal)
        if u < 0.6:
            sup.random_syncs(kit, s, rng, 1)
        else:
            k0, markers, _ = sup.make_case(kit, scene, 100 + step, store=s)
            fn = slices if rng.random() < 0.3 else islands
            fn(S, s, k0, markers, scope="stricter" if rng.random() < 0.2 else "all", thresholds=small_thresholds(S),
               **CALLBACK_SETS[int(rng.integers(0, len(CALLBACK_SETS)))])
        assert kit.maps_consistent(s) and kit.counts_consistent(s)
        if len(s.journal) > n_before:
            stack.append(kit.snapshot(s))
            redo = []
            assert not s.redo_stack, "a new operation clears redo"
    while s.journal:
        s.undo()
        stack.pop()
        assert kit.same_state(s, stack[-1], maps=False) and kit.maps_consistent(s)


# --------------------------------------------------------------------------------------------
# Update Global / Update Nonglobal against today's update numerics


def smooth(rng, shape, sigma):
    f = ndimage.gaussian_filter(rng.random(shape), sigma)
    return (f - f.min()) / (np.ptp(f) + 1e-9)


def ball(shape, centre, r):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2 <= r * r


def update_scene(rng, kind, K):
    """(layer masks {lid: [mask per k]}, roles {lid: (role, additive)}, visible, shape)."""
    shape = tuple(int(v) for v in rng.integers(10, 18, 3))
    layers, roles = {}, {}
    base = smooth(rng, shape, 1.6)
    for i in range(int(rng.integers(1, 5))):
        f = 0.6 * base + 0.4 * smooth(rng, shape, float(rng.uniform(0.8, 2.0)))
        t = np.sort(rng.uniform(0.35, 0.8, K))
        layers[f"D{i}"] = [f > t[k] for k in range(K)]
        roles[f"D{i}"] = ("direct", True)
    if rng.random() < 0.7:          # a generated layer that does not nest: bit planes
        layers["GEN"] = [smooth(rng, shape, 1.2) > rng.uniform(0.55, 0.8) for _ in range(K)]
        roles["GEN"] = ("generated", True)
    if rng.random() < 0.5:          # additive derived
        f = smooth(rng, shape, 1.4)
        layers["DERA"] = [f > 0.6 + 0.3 * k / K for k in range(K)]
        roles["DERA"] = ("derived", True)
    if rng.random() < 0.5:          # non-additive derived: never a source, never clipped
        layers["DERN"] = [smooth(rng, shape, 1.4) > 0.6 for _ in range(K)]
        roles["DERN"] = ("derived", False)
    if kind == "switch":            # island A wins loose cutoffs, island B strict ones: a non-nested Global
        A0, B0 = np.array([s // 4 + 2 for s in shape]), np.array([3 * s // 4 - 1 for s in shape])
        layers["SW"] = [ball(shape, A0, max(0.0, 4.5 - 1.6 * k * 6 / K)) | ball(shape, B0, 2.2) for k in range(K)]
        roles["SW"] = ("direct", True)
        far = ~ndimage.binary_dilation(layers["SW"][0], iterations=2)
        for lid in [l for l, (r, a) in roles.items() if a and l != "SW"]:
            layers[lid] = [m & far & (smooth(rng, shape, 1) > 0.75) for m in layers[lid]]
    if kind == "tie":               # two identical islands at every cutoff
        layers = {lid: ms for lid, ms in layers.items() if not roles[lid][1]}
        roles = {lid: r for lid, r in roles.items() if lid in layers}
        ms = []
        for k in range(K):
            m = np.zeros(shape, bool)
            w = max(1, 3 - k // 2)
            m[1:1 + w, 1:1 + w, 1:1 + w] = True
            m[shape[0] - 5:shape[0] - 5 + w, shape[1] - 5:shape[1] - 5 + w, shape[2] - 5:shape[2] - 5 + w] = True
            ms.append(m)
        layers["TIE"] = ms
        roles["TIE"] = ("direct", True)
    if kind == "empty":
        for lid, (r, a) in roles.items():
            if a:
                layers[lid] = [np.zeros(shape, bool) for _ in range(K)]
    if kind == "strict_empty":      # the union is empty at the strictest cutoffs
        cut = int(rng.integers(1, K))
        for lid, (r, a) in roles.items():
            if a:
                layers[lid] = [m if k < cut else np.zeros(shape, bool) for k, m in enumerate(layers[lid])]
    union = [np.logical_or.reduce([layers[l][k] for l, (r, a) in roles.items() if a] or [np.zeros(shape, bool)])
             for k in range(K)]
    G = []
    for k in range(K):
        lab, n = ndimage.label(union[k])
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            G.append(lab == int(np.argmax(sizes)))
        else:
            G.append(np.zeros(shape, bool))
    mode = int(rng.integers(0, 4))
    if mode == 1:                   # Global-only voxels and holes
        G = [g ^ (smooth(rng, shape, 1) > 0.85) for g in G]
    elif mode == 2:                 # empty at some cutoffs (Update Nonglobal skips them)
        G = [g if rng.random() < 0.6 else np.zeros(shape, bool) for g in G]
    elif mode == 3:                 # a non-nested Global
        G = [smooth(rng, shape, 1.3) > 0.62 for _ in range(K)]
    layers["GLOBAL"] = G
    roles["GLOBAL"] = ("global", False)
    visible = {lid: bool(rng.random() < 0.75) for lid in layers}
    return layers, roles, visible, shape


def store_from_masks(S, kit, shape, K, layers, roles, visible):
    store = S.XTACutoffStore(shape, K)
    for lid, ms in layers.items():
        role, additive = roles[lid]
        any_mask = np.logical_or.reduce(ms)
        tb = S._xta_store_tight_box(any_mask)
        lo, hi = (np.zeros(3, np.int64), np.ones(3, np.int64)) if tb is None else tb
        sl = S._xta_store_sl(lo, hi)
        kw = dict(visible=visible[lid])
        if role == "derived":
            kw["additive"] = additive
        kit.add_layer_from_masks(store, lid, role, [m[sl] for m in ms], lo, **kw)
    return store


def store_masks(kit, store):
    return {lid: kit.full_masks(store, lid) for lid in store.layers}


def mask_diff(a, b):
    return [(lid, k) for lid in a for k, (x, y) in enumerate(zip(a[lid], b[lid])) if not np.array_equal(x, y)]


def storage_ok(kit, store):
    """Level map iff the layer's masks nest."""
    for lid, l in store.layers.items():
        ms = kit.full_masks(store, lid)
        nested = all(not (ms[k + 1] & ~ms[k]).any() for k in range(len(ms) - 1))
        if (l.storage == "level") != nested:
            return False
    return True


@pytest.fixture(scope="module")
def today_update_api():
    return sup.today_update_namespace()


@pytest.mark.parametrize("kind,K,seed", [(kind, K, seed) for seed, (kind, K) in enumerate(
    [(k, K) for K in (4, 9, 65) for k in ("plain", "switch", "tie", "empty", "strict_empty")])])
def test_updates_match_today(S, kit, today_update_api, kind, K, seed):
    rng = np.random.default_rng(11 + seed)
    layers, roles, visible, shape = update_scene(rng, kind, K)
    store = store_from_masks(S, kit, shape, K, layers, roles, visible)
    api = today_update_api
    M0 = store_masks(kit, store)
    # Update Global (a dry run first: same changed cutoffs, nothing written)
    s = kit.copy(store)
    s.global_stale = True
    dry = S._xta_store_update_global(s, tie_breaker="raster_first", dry_run=True)
    assert not mask_diff(store_masks(kit, s), M0) and s.global_stale and not s.journal
    rec = S._xta_store_update_global(s, tie_breaker="raster_first")
    assert dry.changed_cutoffs == rec.changed_cutoffs
    ref, _ = sup.today_update(api, M0, roles, K, do_nonglobal=False)
    assert not mask_diff(store_masks(kit, s), ref)
    assert s.global_stale is False and kit.maps_consistent(s) and kit.counts_consistent(s) and storage_ok(kit, s)
    S._xta_store_undo_update(s, rec)
    assert not mask_diff(store_masks(kit, s), M0) and s.global_stale is True
    # Update Nonglobal (on the stored Global: possibly empty somewhere, or bit planes)
    s = kit.copy(store)
    rec = S._xta_store_update_nonglobal(s)
    ref, info = sup.today_update(api, M0, roles, K, do_global=False)
    got = store_masks(kit, s)
    assert not mask_diff(got, ref), mask_diff(got, ref)[:4]
    assert sorted(rec.skipped_empty_global) == sorted(r["k"] for r in info if r.get("skipped"))
    assert kit.maps_consistent(s) and kit.counts_consistent(s) and storage_ok(kit, s)
    assert rec.removed_by_cutoff == [sum(int(np.count_nonzero(M0[l][k] & ~got[l][k])) for l in M0) for k in range(K)]
    S._xta_store_undo_update(s, rec)
    assert not mask_diff(store_masks(kit, s), M0) and kit.maps_consistent(s) and kit.counts_consistent(s)
    # Update Both, then a second run changes nothing; undo all four
    s = kit.copy(store)
    g_rec, n_rec = S._xta_store_update_both(s, tie_breaker="raster_first")
    ref, _ = sup.today_update(api, M0, roles, K, both=True)
    assert not mask_diff(store_masks(kit, s), ref)
    g2, n2 = S._xta_store_update_both(s, tie_breaker="raster_first")
    assert not g2.changed and not n2.changed and g2.entry is None and n2.entry is None
    for r in (n2, g2, n_rec, g_rec):
        S._xta_store_undo_update(s, r)
    assert not mask_diff(store_masks(kit, s), M0) and kit.maps_consistent(s) and kit.counts_consistent(s)
    # one cutoff at a time: the stale flag stays
    k = int(rng.integers(0, K))
    s = kit.copy(store)
    s.global_stale = True
    rec = S._xta_store_update_global(s, [k], tie_breaker="raster_first")
    ref, _ = sup.today_update(api, M0, roles, K, do_nonglobal=False, cutoffs=[k])
    assert not mask_diff(store_masks(kit, s), ref) and s.global_stale is True and storage_ok(kit, s)
    rec2 = S._xta_store_update_nonglobal(s, [k])
    ref, _ = sup.today_update(api, ref, roles, K, do_global=False, cutoffs=[k])
    assert not mask_diff(store_masks(kit, s), ref) and kit.maps_consistent(s) and storage_ok(kit, s)
    if rec.entry is not None and rec2.entry is not None:
        with pytest.raises(RuntimeError):
            S._xta_store_undo_update(s, rec)                # not the newest entry
    S._xta_store_undo_update(s, rec2)
    S._xta_store_undo_update(s, rec)
    assert not mask_diff(store_masks(kit, s), M0)


def test_stale_global_cutoffs_after_a_click(S, kit):
    s = kit.copy(sup.g_scene(kit))
    islands(S, s, 1, np.array([[8, 14, 30]]), **ACCEPT_ALL)
    assert s.global_stale
    stale = S._xta_store_stale_global_cutoffs(s, tie_breaker="raster_first")
    depth = len(s.journal)
    rec = S._xta_store_update_global(s, tie_breaker="raster_first")
    assert rec.changed_cutoffs == stale and len(s.journal) == depth + (1 if rec.entry is not None else 0)
    assert not s.global_stale and S._xta_store_stale_global_cutoffs(s, tie_breaker="raster_first") == []


def test_export_lattice_refuses_what_today_refuses(S):
    lattice = S._xta_store_export_lattice
    ref = np.diag([-1.0, -1.0, 1.0, 1.0])
    assert lattice(ref)[0].tolist() == [0, 0, 0] and lattice(ref)[1] is None
    shifted = ref.copy()
    shifted[:3, 3] = (-3.0, -2.0, 1.0)                       # RAS of the review lattice's IJK 0 voxel
    assert lattice(shifted)[0].tolist() == [1, 2, 3]
    rotated = ref.copy()
    rotated[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, 1]]
    assert lattice(rotated) == (None, "Sweep edit grid was resampled or rotated.")
    off = ref.copy()
    off[0, 3] = 0.5
    assert lattice(off) == (None, "Sweep edit origin is not on its original preview lattice.")
