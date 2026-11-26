import ast
import builtins
from contextlib import nullcontext
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from tests import slicer_cutoff_store_support as support
from tools.build_slicer_review_bundle import build
from XTA.slicer_reconciliation import (
    ReconciliationCancelled,
    apply_reconciliation,
    load_preview_policy,
    prepare_reconciliation,
)


@pytest.fixture
def review():
    editor = support.load_store_module(engines=False)
    kit = support.StoreKit(editor)
    shape = (3, 5, 7)
    store = editor.XTACutoffStore(shape, 2)
    masks = {name: [np.zeros(shape, bool), np.zeros(shape, bool)]
             for name in ("radial", "radial-duplicate", "transverse", "global")}
    masks["radial"][0][1, 2, 3:5] = True
    masks["radial-duplicate"][0][1, 2, 3:5] = True
    masks["transverse"][0][1, 2, 4] = True
    masks["transverse"][1][2, 3, 5] = True
    masks["global"][0][0, 0, 0] = True
    for name, values in masks.items():
        role = "global" if name == "global" else "direct"
        kit.add_layer_from_masks(store, name, role, values)
    store.global_stale = True
    plan = SimpleNamespace(
        shape=shape, cutoffs=(80, 90), grid_id="low_quality/test",
        layers={name: {"rows": {0: {"mask_kind": "yolo",
                                   "physical_view_name": name.removesuffix("-duplicate")}}}
                for name in masks if name != "global"},
    )
    try:
        yield editor, store, plan, masks
    finally:
        store.close()
        support.unload_store_module(editor)


def _global(editor, store, cutoff):
    return editor._xta_store_layer_mask_in_box(store.global_layer(), cutoff, (0, 0, 0), store.shape)


def test_exact_reconciliation_stages_and_undoes_all_cutoffs(review):
    editor, store, plan, masks = review
    policy, digest = load_preview_policy(None, 16)
    before = tuple(_global(editor, store, cutoff) for cutoff in range(2))
    staged = prepare_reconciliation(store, plan, policy, memory_mib=16, policy_sha256=digest)
    assert [(r.added, r.removed) for r in staged.cutoffs] == [(2, 1), (1, 0)]
    assert store.global_stale and not store.journal
    for cutoff, mask in enumerate(before):
        np.testing.assert_array_equal(_global(editor, store, cutoff), mask)

    entry = apply_reconciliation(store, staged, editor._XTAStoreFamily)
    assert entry.kind == "reconcile_global"
    assert len(store.journal) == 1 and not store.global_stale
    for cutoff in range(2):
        expected = masks["radial"][cutoff] | masks["transverse"][cutoff]
        np.testing.assert_array_equal(_global(editor, store, cutoff), expected)
        for name in ("radial", "radial-duplicate", "transverse"):
            actual = editor._xta_store_layer_mask_in_box(store.layers[name], cutoff, (0, 0, 0), store.shape)
            np.testing.assert_array_equal(actual, masks[name][cutoff])
    assert store.undo() and store.global_stale
    for cutoff, mask in enumerate(before):
        np.testing.assert_array_equal(_global(editor, store, cutoff), mask)
    assert store.redo() and not store.global_stale
    with pytest.raises(RuntimeError, match="changed"):
        apply_reconciliation(store, staged, editor._XTAStoreFamily)


def test_custom_policy_counts_independent_view_groups(review, tmp_path):
    editor, store, plan, _ = review
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        "def build_reconciliation():\n"
        "    return dict(name='two views', mode='weighted', grouping='views', "
        "threshold=2, min_sources=2)\n",
        encoding="utf-8",
    )
    policy, digest = load_preview_policy(policy_path, 16)
    staged = prepare_reconciliation(store, plan, policy, memory_mib=16, policy_sha256=digest)
    entry = apply_reconciliation(store, staged, editor._XTAStoreFamily)
    assert entry.report["policy_sha256"] == digest
    assert np.argwhere(_global(editor, store, 0)).tolist() == [[1, 2, 4]]
    assert not _global(editor, store, 1).any()


@pytest.mark.parametrize("policy, message", [
    ({"mode": "confidence"}, "observed confidence"),
    ({"mode": "weighted", "grouping": "sections"}, "section geometry"),
    ({"mode": "union", "sam_bridge_policy": "permissive"}, "SAM proposals"),
])
def test_unsupported_policy_refuses_without_edits(review, policy, message):
    _, store, plan, _ = review
    with pytest.raises(ValueError, match=message):
        prepare_reconciliation(store, plan, policy)
    assert not store.journal and store.global_stale


def test_policy_loader_rejects_sam_before_importing_optional_runtime(tmp_path, monkeypatch):
    path = tmp_path / "sam.py"
    path.write_text("def build_reconciliation():\n"
                    "    return dict(mode='union', sam_bridge_policy='strict')\n", encoding="utf-8")
    original = builtins.__import__

    def no_sam(name, *args, **kwargs):
        if name in ("sam_policy", "XTA.sam_policy"):
            pytest.fail("Loading an unsupported policy should not import the SAM validator")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sam)
    with pytest.raises(ValueError, match="SAM proposals"):
        load_preview_policy(path, 16)


def test_cancel_discards_prepared_cutoff_without_creating_an_entry(review):
    editor, store, plan, masks = review

    def progress(cutoff, _phase, _done, _total):
        if cutoff == 90:
            raise ReconciliationCancelled

    with pytest.raises(ReconciliationCancelled):
        prepare_reconciliation(store, plan, {"mode": "union"}, memory_mib=16, progress=progress)
    assert not store.journal and store.global_stale
    np.testing.assert_array_equal(_global(editor, store, 0), masks["global"][0])


def test_snapshot_budget_refuses_before_previews(review):
    _, store, plan, _ = review
    with pytest.raises(ValueError, match="reconciliation budget"):
        prepare_reconciliation(store, plan, {"mode": "union"}, memory_mib=1,
                               progress=lambda *_: pytest.fail("Snapshot allocation should be refused"))
    assert not store.journal


def test_staging_reads_frozen_masks_and_cannot_commit_after_new_edit(review):
    editor, store, plan, _ = review
    changed = False

    def progress(_cutoff, phase, _done, _total):
        nonlocal changed
        if phase == "setup" and not changed:
            changed = True
            # Simulate a competing accepted edit after the snapshots were copied.
            builder = store.begin_entry("test_edit")
            builder.replace_layer("radial", np.zeros((1, 1, 1), np.uint8), (0, 0, 0))
            builder.replace_layer("radial-duplicate", np.zeros((1, 1, 1), np.uint8), (0, 0, 0))
            builder.commit()

    staged = prepare_reconciliation(store, plan, {"mode": "union"}, memory_mib=16, progress=progress)
    assert staged.cutoffs[0].added == 2
    with pytest.raises(RuntimeError, match="changed"):
        apply_reconciliation(store, staged, editor._XTAStoreFamily)
    assert len(store.journal) == 1 and _global(editor, store, 0)[0, 0, 0]


@pytest.mark.parametrize("cancel_progress, choose_apply, exact", [
    (True, True, None), (False, False, None), (False, True, None), (False, True, 0.333),
])
def test_toolbar_action_uses_existing_store_and_confirms(review, cancel_progress, choose_apply, exact, request):
    editor, store, plan, masks = review
    if exact is not None:
        store = editor.XTACutoffStore(store.shape, 1)
        request.addfinalizer(store.close)
        kit = support.StoreKit(editor)
        for name, values in masks.items():
            kit.add_layer_from_masks(store, name, "global" if name == "global" else "direct", values[:1])
        store.global_stale = True
        plan.cutoffs = (0,)
        plan.manifest = {"cutoffs_cent": (0,), "confidence_threshold_exact": exact}
    refreshes = []
    current_review = SimpleNamespace(
        store=store, plan=plan, closed=False, sync=lambda: None, require=lambda _: store,
        apply_store_changes=refreshes.append,
    )

    class Progress:
        def __getattr__(self, name):
            return lambda *args: None

        wasCanceled = cancel_progress

    def choose(_title, prompt, choices):
        if choices[0] == "Union":
            return "Union"
        assert "low_quality/test" in prompt and "+2 / −1" in prompt
        if exact is not None:
            assert "0.333: +2" in prompt and "0.00:" not in prompt
        return "Apply reconciliation" if choose_apply else "Keep current Global"

    tools = vars(editor).copy()
    tools.update(
        _xta_review_for_update=lambda _: current_review,
        _xta_review_exclusive=lambda _: nullcontext(),
        _xta_review_store_call=lambda _review, _action, operation: operation(store),
        _xta_native_review_support=lambda: None,
        slicer=SimpleNamespace(
            util=SimpleNamespace(mainWindow=object, showStatusMessage=lambda *_: None),
            app=SimpleNamespace(processEvents=lambda: None),
        ),
        qt=SimpleNamespace(QProgressDialog=lambda *_: Progress(), Qt=SimpleNamespace(ApplicationModal=1)),
        _choose_item=choose,
    )
    _, tree = support.parse_source()
    nodes = support.function_nodes(tree, ("ReconcileGlobal", "_xta_cutoff_label"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(support.SOURCE), "exec"), tools)
    entry = tools["ReconcileGlobal"](memory_mib=16)
    committed = not cancel_progress and choose_apply
    assert (entry is not None) == committed
    assert bool(refreshes) == committed
    assert bool(store.journal) == committed
    assert store.global_stale != committed


def test_bundled_reconciler_imports_without_inference(tmp_path):
    path = build(tmp_path / "SlicerToolbar.zip")
    with ZipFile(path) as bundle:
        assert {"XTA/slicer_reconciliation.py", "XTA/reconciliation.py",
                "XTA/reconciliation_policy.py", "XTA/slicer_preview_review.py"} <= set(bundle.namelist())
    subprocess.run([
        sys.executable, "-I", "-B", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); import XTA.slicer_reconciliation;"
        "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules",
        str(path),
    ], check=True)
