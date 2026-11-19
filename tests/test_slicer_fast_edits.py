"""Faster Remove Islands / Remove Slices / Update Both produce exactly the previous masks.

The Slicer profile now reads internal labelmaps in place: a marker probe skips
masks no marker can reach, Update Global unions sources over their extents,
Update Nonglobal writes back only sources that actually change, and the 2D
coverage display updates incrementally. These tests compile the real profile
functions and compare them with direct reference computations of the
previous semantics on random masks. Native Slicer runs live in Scratch.
"""
import ast
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

SOURCE = Path(__file__).resolve().parents[1] / "tools/slicer/xta_slicerrc.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def compile_names(names, constants=(), classes=(), **namespace):
    body = [n for n in TREE.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.ClassDef) and n.name in classes)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in constants for t in n.targets))
            or (classes and isinstance(n, ast.Import)
                and any((a.asname or "").startswith("_xta_display_") for a in n.names))]
    found = {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert found == set(names) | set(classes)
    namespace.setdefault("np", np)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


# ── Update Global / Nonglobal numerics ────────────────────────────────

UPDATE = ("_xta_extent_view", "_xta_crop_mask", "_xta_same_mask", "_xta_run_threads", "_xta_union_sources",
          "_xta_clip_source", "_xta_removed_voxels", "_extent_intersection", "_extent_is_valid")


def random_sources(rng, grid=(18, 22, 26), count=7, shared=True):
    """(array, extent, label, None) sources; some share a layer with another label value."""
    sources, masks = [], []
    for index in range(count):
        lo = [int(rng.integers(0, n // 2)) for n in grid[::-1]]
        hi = [int(rng.integers(l + 1, n)) for l, n in zip(lo, grid[::-1])]
        extent = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        shape = (hi[2] - lo[2] + 1, hi[1] - lo[1] + 1, hi[0] - lo[0] + 1)
        fg = ndimage.binary_opening(rng.random(shape) < 0.55) | (rng.random(shape) < 0.02)
        label = int(rng.integers(1, 4)) if shared else 1
        array = np.zeros(shape, np.uint8)
        array[fg] = label
        if shared:  # another segment's voxels in the same layer
            array[(~fg) & (rng.random(shape) < 0.2)] = label + 1
        if index == count - 1:
            array[:] = 0  # an allocated but empty labelmap
        full = np.zeros(grid, bool)
        full[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = array == label
        sources.append((array, extent, label, None))
        masks.append(full)
    return sources, masks


def tight(full):
    """Reference: (mask, extent) cropped to the foreground of a full-grid mask, as the VTK crop did."""
    if not full.any():
        return None
    z, y, x = (np.flatnonzero(full.any(axis=a)) for a in ((1, 2), (0, 2), (0, 1)))
    return (full[z[0]:z[-1] + 1, y[0]:y[-1] + 1, x[0]:x[-1] + 1],
            (int(x[0]), int(x[-1]), int(y[0]), int(y[-1]), int(z[0]), int(z[-1])))


def same(api, a, b):
    return api["_xta_same_mask"](a, b)


@pytest.mark.parametrize("seed", range(12))
def test_union_and_clip_match_the_reference_semantics(seed):
    api = compile_names(UPDATE, concurrent=concurrent)
    rng = np.random.default_rng(seed)
    sources, masks = random_sources(rng, shared=seed % 2 == 0)
    union_reference = np.logical_or.reduce(masks)
    for workers in (1, 3, 8):
        assert same(api, api["_xta_union_sources"](sources, workers), tight(union_reference))
    union = api["_xta_union_sources"](sources, 4)
    labels, _ = ndimage.label(union_reference)  # six-connected, as vtkITKIslandMath(FullyConnected=False)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    global_full = labels == int(np.argmax(sizes)) if sizes.any() else np.zeros_like(union_reference)
    if seed % 3 == 0:  # a Global edited by hand: not the largest island of the union
        global_full = global_full & (rng.random(global_full.shape) < 0.9)
    global_mask = tight(global_full)
    if global_mask is None:
        pytest.skip("empty union")
    removed = api["_xta_removed_voxels"](union, global_mask)
    assert same(api, removed, tight(union_reference & ~global_full))
    for source, full in zip(sources, masks):
        expected = tight(full & global_full)
        unchanged = same(api, expected, tight(full))
        for hint in (None, removed):
            changed, clipped = api["_xta_clip_source"](source, global_mask, hint)
            assert changed == (not unchanged)
            if changed:
                assert same(api, clipped, expected)


def test_identical_and_empty_masks_compare_by_voxels_not_storage():
    api = compile_names(UPDATE)
    a = np.zeros((2, 3, 4), np.uint8)
    a[1, 1, 1:3] = 1
    assert same(api, (a, (0, 3, 0, 2, 0, 1)), (a.astype(bool), (0, 3, 0, 2, 0, 1)))
    assert not same(api, (a, (0, 3, 0, 2, 0, 1)), (a, (1, 4, 0, 2, 0, 1)))
    assert same(api, None, None) and not same(api, None, (a, (0, 3, 0, 2, 0, 1)))
    assert api["_xta_crop_mask"](np.zeros((2, 2, 2), np.uint8), (0, 1, 0, 1, 0, 1)) is None
    mask, extent = api["_xta_crop_mask"](a, (10, 13, 20, 22, 30, 31))
    assert extent == (11, 12, 21, 21, 31, 31) and mask.shape == (1, 1, 2)


# ── Marker probe: never misses a mask the exact per-mask search would seed ──

PROBE = ("_marker_probe_voxels", "_probe_hits_foreground", "_ras_points_to_continuous_ijk",
         "_ras_line_segments_to_continuous_ijk", "_voxel_indices_intersected_by_line", "_clip_line_segment_to_box",
         "_remove_consecutive_duplicate_points", "_closed_curve_interior_candidates",
         "_closed_curve_interior_seed_indices", "_closed_curve_boundary_seed_indices", "_scanline_polygon_u_values",
         "_seed_indices_from_points", "_seed_indices_from_lines", "_seed_indices_from_closed_curves",
         "_foreground_mask_for_indices", "_unique_foreground_seed_indices", "_combine_seed_indices",
         "_one_seed_per_axis_run")


class Matrix:
    def __init__(self, array):
        self.array = np.asarray(array, float)

    def GetElement(self, row, column):
        return float(self.array[row, column])


class LatticeImage:
    """Only the geometry the seed functions read."""
    def __init__(self, matrix):
        self.matrix = matrix


def probe_api():
    return compile_names(PROBE, constants=("_NEIGHBORHOOD_26",),
                         _image_to_world_matrix=lambda image: Matrix(image.matrix))


def exact_seed_count(api, image, layer, extent, label, markers):
    """The per-mask seeds the job computes on its effective-cropped binary copy."""
    full = layer == label
    if not full.any():
        return 0
    z, y, x = (np.flatnonzero(full.any(axis=a)) for a in ((1, 2), (0, 2), (0, 1)))
    binary = full[z[0]:z[-1] + 1, y[0]:y[-1] + 1, x[0]:x[-1] + 1].astype(np.uint8)
    effective = np.array([extent[0] + x[0], extent[0] + x[-1], extent[2] + y[0], extent[2] + y[-1],
                          extent[4] + z[0], extent[4] + z[-1]], np.int64)
    context = {"extent": effective, "extent_origin": effective[::2].copy(),
               "dimensions": tuple(int(v) for v in binary.shape[::-1]), "array": binary}
    points, lines, curves = markers
    try:
        found = (api["_seed_indices_from_points"](image, context, points),
                 api["_seed_indices_from_lines"](image, context, lines),
                 api["_seed_indices_from_closed_curves"](image, context, curves))
    except RuntimeError:  # the existing walk can fail on exact voxel corners; the action then fails
        return None
    return sum(seeds.shape[0] for seeds, _hits in found)


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("oriented", [False, True])
def test_marker_probe_is_a_superset_of_exact_seeds_and_exact_for_points(seed, oriented):
    api = probe_api()
    rng = np.random.default_rng(100 + seed)
    matrix = np.eye(4)
    if oriented:
        matrix[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ np.diag([0.8, 1.0, 1.3])
        matrix[:3, 3] = [5.0, -3.0, 2.0]
    image = LatticeImage(matrix)
    to_ras = lambda ijk: tuple((matrix @ np.append(np.asarray(ijk, float), 1.0))[:3])
    grid = (14, 18, 20)
    sources, _ = random_sources(rng, grid=grid, count=8)
    extent = [min(s[1][0] for s in sources), max(s[1][1] for s in sources), min(s[1][2] for s in sources),
              max(s[1][3] for s in sources), min(s[1][4] for s in sources), max(s[1][5] for s in sources)]
    random_point = lambda: rng.uniform(-1.0, np.array(grid[::-1]) + 0.5)
    points = [to_ras(random_point()) for _ in range(3)]
    # Include walks through exact voxel corners and edges (tie-breaking cases).
    corner = np.floor(random_point()) + 0.5
    lines = [(to_ras(random_point()), to_ras(random_point())),
             (to_ras(corner), to_ras(corner + np.array([4.0, 4.0, 4.0]))),
             (to_ras(corner + [0, 0.5, 0]), to_ras(corner + [6.0, 0.5, 6.0]))]
    plane_z = float(rng.integers(2, grid[0] - 2))
    curves = [[to_ras(np.array([x, y, plane_z + rng.uniform(-0.3, 0.3)])) for x, y in
               ((2, 2), (12, 3), (14, 12), (4, 15))]]
    for markers, exact in (((points, [], []), True), (([], lines, []), False), (([], [], curves), False),
                           ((points, lines, curves), False)):
        probe = api["_marker_probe_voxels"](image, *markers, extent)
        for array, source_extent, label, _ in sources:
            count = exact_seed_count(api, image, array, source_extent, label, markers)
            if count is None:
                continue
            seeded = count > 0
            hit = api["_probe_hits_foreground"]((array, source_extent, label, image), probe)
            assert hit or not seeded
            if exact:
                assert hit == seeded


def test_closed_curve_interior_candidates_feed_the_unchanged_seed_reduction():
    api = probe_api()
    curve = np.array([[2.0, 2.0, 5.2], [11.0, 3.0, 5.1], [12.0, 10.0, 4.9], [3.0, 11.0, 5.0]])
    binary = np.zeros((10, 14, 15), np.uint8)
    binary[5, 3:9, 4:10] = 1
    binary[4, 6, 6] = 1
    context = {"extent": np.array([0, 14, 0, 13, 0, 9]), "extent_origin": np.zeros(3, np.int64),
               "dimensions": (15, 14, 10), "array": binary}
    seeds = api["_closed_curve_interior_seed_indices"](context, curve)
    # One seed per face-connected run along the scan axis, each a foreground voxel.
    assert seeds.shape[0] == 6 and np.all(binary[seeds[:, 2], seeds[:, 1], seeds[:, 0]] == 1)
    candidates = np.concatenate([c for c, _ in api["_closed_curve_interior_candidates"](curve, context["extent"])])
    wide = np.concatenate([c for c, _ in api["_closed_curve_interior_candidates"](curve, np.array([-5, 30, -5, 30, -5, 30]))])
    as_set = lambda rows: {tuple(r) for r in rows}
    assert as_set(seeds) <= as_set(candidates) <= as_set(wide)


# ── Coverage display: incremental refresh from remembered contributions ──

DISPLAY = tuple(n.name for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_xta_display_"))


class Scalars:
    def __init__(self, array):
        self.array = array

    def GetNumberOfComponents(self):
        return 1


class Layer:
    def __init__(self, array, extent, matrix):
        self.array, self.extent, self.matrix = array, tuple(extent), matrix

    def IsA(self, name):
        return name == "vtkOrientedImageData"

    def GetPointData(self):
        return SimpleNamespace(GetScalars=lambda: Scalars(self.array) if self.array is not None else None)

    def GetExtent(self):
        return self.extent

    def GetImageToWorldMatrix(self, out):
        out.array = self.matrix.copy()


class DisplayMatrix(Matrix):
    def __init__(self):
        super().__init__(np.eye(4))


class SegmentationNode:
    """One node holding separate layers; GetBinaryLabelmapRepresentation is Slicer's copy path."""
    def __init__(self, affine):
        self.affine, self.segments = affine, {}

    def set(self, sid, array, extent, label=1):
        self.segments[sid] = SimpleNamespace(
            layer=Layer(array, extent, self.affine), label=label,
            GetRepresentation=lambda name, sid=sid: self.segments[sid].layer,
            GetLabelValue=lambda sid=sid: self.segments[sid].label)

    def GetID(self):
        return "node"

    def GetParentTransformNode(self):
        return None

    def GetSegmentation(self):
        return self

    def GetSegment(self, sid):
        return self.segments.get(sid)

    def GetDisplayNode(self):
        return SimpleNamespace(GetVisibility=lambda: True, GetSegmentVisibility=lambda sid: True)

    def GetBinaryLabelmapRepresentation(self, sid, image):
        segment = self.segments[sid]
        image.layer = Layer((segment.layer.array == segment.label).astype(np.uint8), segment.layer.extent,
                            self.affine)
        return True


class CopiedImage:
    def __getattr__(self, name):
        return getattr(self.layer, name)


def controller(api, node, rows, shape, affine, budget):
    display = object.__new__(api["XTASweepDisplay"])
    np_support = SimpleNamespace(vtk_to_numpy=lambda scalars: scalars.array.ravel())
    slicer = SimpleNamespace(vtkOrientedImageData=CopiedImage, vtkSegmentationConverter=SimpleNamespace(
        GetSegmentationBinaryLabelmapRepresentationName=lambda: "Binary labelmap"))
    capacity = sum(row["role"] == "direct" for row in rows.values())
    display.__dict__.update(
        _np=np, _vtk=SimpleNamespace(vtkMatrix4x4=DisplayMatrix), _numpy_support=np_support, _slicer=slicer,
        affine=affine, shape=shape, scene=SimpleNamespace(IsBatchProcessing=lambda: False),
        _enabled=True, _closed=False, _depth=0, _updating=False, auto_refresh=False, _binding_dirty=False,
        selected_cutoff=50, _loaded_cutoffs=frozenset({50}), _published_cutoff=None,
        _cut_bindings={50: {sid: (node, sid, row) for sid, row in rows.items()}},
        _metadata=rows, _logical_branches={}, _node_sid_ids={"node": {sid: sid for sid in rows}},
        _full_dirty=True, _dirty=set(), _source_bounds={}, _cache={}, _cache_bytes=0, _cache_budget=budget,
        _previous={}, _direct_capacities={50: capacity}, capacity=capacity,
        _direct=np.zeros(shape, np.uint8), _generated=np.zeros(shape, np.uint8), _global=np.zeros(shape, np.uint8),
        _codes=np.zeros(shape, np.uint8), _field_dtypes={}, _code_dtype=np.dtype("uint8"),
        stats={"refresh_count": 0, "snapshots_read": 0, "full_refreshes": 0, "incremental_refreshes": 0},
        _set_palette=lambda: None, _write_overlay_full=lambda: None, _wire_views=lambda: None,
        _persist=lambda: None, last_error=None, overlay_node=SimpleNamespace(GetImageData=lambda: None))
    return display


@pytest.mark.parametrize("budget", [0, 1 << 30])
@pytest.mark.parametrize("edit", ["inside", "shrink", "clear"])
def test_remembered_contributions_update_coverage_exactly_without_recomposing(budget, edit):
    api = compile_names(DISPLAY, classes=("XTASweepDisplay",))
    rng = np.random.default_rng(7)
    shape = (12, 16, 18)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    affine[:3, 3] = [3.0, 4.0, -2.0]
    rows = {f"s{i}": {"segment_id": f"s{i}", "logical_id": f"s{i}", "cutoff_cent": 50,
                      "role": role, "layer_role": "additive_component", "recomposition_op": "union"}
            for i, role in enumerate(("direct", "direct", "direct", "generated", "global"))}

    def scene(edit=None):
        node = SegmentationNode(affine)
        local = np.random.default_rng(7)
        for sid in rows:
            array = (local.random((8, 10, 12)) < 0.4).astype(np.uint8) * 2  # label value 2
            if edit and sid == "s1":
                if edit == "inside":  # same bounds: the packed-difference block
                    array[2:6, 3:5] = 0
                elif edit == "shrink":  # smaller bounds: the union-bounds difference
                    array[:3] = 0
                else:
                    array[:] = 0
            node.set(sid, array, (3, 14, 4, 13, 2, 9), label=2)
        return node

    node = scene()
    display = controller(api, node, rows, shape, affine, budget)
    assert display.refresh()
    edited = scene(edit=edit)
    for remember in (True, False):
        display = controller(api, node, rows, shape, affine, budget)
        display.refresh()
        if remember:
            display.remember_previous(node, ["s1"])
        node.segments["s1"] = edited.segments["s1"]
        display.mark_dirty(["s1"])
        assert display.refresh()
        reference = controller(api, node, rows, shape, affine, budget)
        reference.refresh()
        np.testing.assert_array_equal(display._codes, reference._codes)
        np.testing.assert_array_equal(display._direct, reference._direct)
        assert display.stats["roi_recomposed"] == (not remember and budget == 0)
        node = scene()

    # The in-place snapshot equals Slicer's detached binary copy, voxel for voxel.
    display = controller(api, node, rows, shape, affine, budget)
    for sid in rows:
        direct = display._snapshot(node, sid)
        display._internal_view = lambda *args: None
        copied = display._snapshot(node, sid)
        del display._internal_view
        assert direct[0] == copied[0] and np.array_equal(direct[1], copied[1])


def test_largest_island_uses_scipy_only_for_a_unique_largest_island():
    calls = []
    api = compile_names(("_xta_largest_island", "_xta_crop_mask"),
                        _xta_binary_image=lambda mask, extent, geometry: ("image", np.array(mask, bool), extent),
                        _keep_largest_island=lambda image: calls.append(image) or "vtkITKIslandMath")
    union = np.zeros((4, 5, 9), np.uint8)
    union[1:3, 1:3, 1:3] = 1          # 8 voxels
    union[0:4, 0:2, 6:8] = 1          # 16 voxels: unique largest
    union[3, 4, 0] = 1                # touches the first island only diagonally: separate
    kind, mask, extent = api["_xta_largest_island"]((union, (10, 18, 20, 24, 30, 33)), "geometry")
    labels, _ = ndimage.label(union)
    expected = tight(labels == labels[0, 0, 6])
    assert kind == "image" and extent == (16, 17, 20, 21, 30, 33) == tuple(
        e + o for e, o in zip(expected[1], (10, 10, 20, 20, 30, 30)))
    np.testing.assert_array_equal(mask, expected[0])
    assert not calls
    tie = np.zeros((2, 2, 7), np.uint8)
    tie[:, :, :2] = tie[:, :, 4:6] = 1  # two equal islands: keep vtkITKIslandMath's choice
    assert api["_xta_largest_island"]((tie, (0, 6, 0, 1, 0, 1)), "geometry") == "vtkITKIslandMath"
    assert calls and calls[0][2] == (0, 6, 0, 1, 0, 1)


# ── Remove Islands / Remove Slices job internals (NumPy paths) ─────────

class Volume:
    """A single-component image: KJI array over an inclusive XYZ extent."""
    def __init__(self, array, extent):
        self.array, self.extent = array, tuple(extent)

    def GetExtent(self):
        return self.extent

    def GetDimensions(self):
        return tuple(self.array.shape[::-1])

    def GetPointData(self):
        return SimpleNamespace(GetScalars=lambda: SimpleNamespace(
            array=self.array, GetNumberOfComponents=lambda: 1))


JOB = ("_binary_effective_copy", "_remove_seeded_slices", "_face_connected_labels", "_erase_seeded_island_slices",
       "_xta_crop_mask", "_unique_foreground_seed_indices", "_foreground_mask_for_indices", "_scipy_face_labels",
       "_remove_seeded_islands")


def job_api():
    return compile_names(
        JOB, _image_has_scalars=lambda image: image is not None,
        numpy_support=SimpleNamespace(vtk_to_numpy=lambda scalars: scalars.array.ravel()),
        _xta_binary_image=lambda mask, extent, geometry: Volume(np.array(mask != 0, np.uint8), extent))


@pytest.mark.parametrize("seed", range(6))
def test_binary_copy_and_slice_removal_match_the_reference(seed):
    api = job_api()
    rng = np.random.default_rng(300 + seed)
    values = np.zeros((9, 11, 13), np.int16)
    values[2:7, 1:9, 3:11] = (rng.random((5, 8, 8)) < 0.6) * rng.integers(1, 4, (5, 8, 8))
    values[3, 3, 4] = -2  # not foreground: the threshold keeps values > 0 only
    image = Volume(values, (20, 32, 40, 50, 60, 68))
    binary = api["_binary_effective_copy"](image)
    expected = tight(values > 0)
    assert binary.extent == tuple(v + o for v, o in zip(expected[1], (20, 20, 40, 40, 60, 60)))
    np.testing.assert_array_equal(binary.array, expected[0].astype(np.uint8))
    assert api["_binary_effective_copy"](Volume(np.zeros((2, 2, 2), np.uint8), (0, 1, 0, 1, 0, 1))) is None

    # Remove Slices: three planes of each seeded island, on its own labels.
    origin = np.array(binary.extent[::2], np.int64)
    context = {"extent": np.array(binary.extent), "extent_origin": origin,
               "dimensions": binary.GetDimensions(), "array": binary.array}
    foreground = np.argwhere(binary.array)
    seeds = foreground[rng.choice(len(foreground), 3, replace=False)][:, ::-1] + origin
    result, islands = api["_remove_seeded_slices"](binary, context, seeds)
    labels, _ = ndimage.label(binary.array)
    reference = binary.array.copy()
    z, y, x = np.indices(reference.shape)
    for sx, sy, sz in seeds - origin:
        same = labels == labels[sz, sy, sx]
        reference[same & (x == sx)] = 0
        reference[same & (y == sy)] = 0
        reference[same & (z == sz)] = 0
    assert islands == len({labels[sz, sy, sx] for sx, sy, sz in seeds - origin})

    # Remove Islands: every face-connected region holding a seed, nothing else.
    removed, count = api["_remove_seeded_islands"](binary, context, seeds)
    seeded = {labels[sz, sy, sx] for sx, sy, sz in seeds - origin}
    kept = tight((binary.array != 0) & ~np.isin(labels, list(seeded)))
    assert count == len(seeded)
    if kept is None:
        assert removed is None
    else:
        assert removed.extent == tuple(v + o for v, o in zip(kept[1], np.repeat(origin, 2)))
        np.testing.assert_array_equal(removed.array, kept[0].astype(np.uint8))
    expected = tight(reference != 0)
    if expected is None:
        assert result is None
    else:
        assert result.extent == tuple(v + o for v, o in zip(expected[1], np.repeat(origin, 2)))
        np.testing.assert_array_equal(result.array, expected[0].astype(np.uint8))


# ── Commit: closed surfaces of emptied segments ──────────────────────

CLOSED, BINARY = "Closed surface", "Binary labelmap"


class SurfaceSegment:
    def __init__(self, segmentation, sid, mask):
        self.segmentation, self.sid = segmentation, sid
        self.tags = {"SegmentID": sid}
        self.representations = {BINARY: mask}

    def RemoveRepresentation(self, name):
        self.segmentation.log.append(("remove", self.sid, name, self.segmentation.owner_depth()))
        self.representations.pop(name, None)

    def GetRepresentation(self, name):
        return self.representations.get(name)


class SurfaceSegmentation:
    """Slicer 5.12.4 as measured natively: a replaced labelmap regenerates that
    segment's derived closed surface; ClearSegment on a segment alone in its
    layer keeps the old one; CreateRepresentation converts segments lacking it."""

    def __init__(self, masks, surfaces, owner_depth=lambda: 0):
        self.log, self.owner_depth = [], owner_depth
        self.segments = {sid: SurfaceSegment(self, sid, mask) for sid, mask in masks.items()}
        if surfaces:
            for segment in self.segments.values():
                segment.representations[CLOSED] = self.convert(segment)

    @staticmethod
    def convert(segment):
        mask = segment.representations[BINARY]
        return ("surface", None if mask is None else int(np.count_nonzero(mask)))

    def ContainsRepresentation(self, name):
        return all(name in s.representations for s in self.segments.values())

    def CreateRepresentation(self, name):
        self.log.append(("create", name, self.owner_depth()))
        for segment in self.segments.values():
            if name not in segment.representations:
                segment.representations[name] = self.convert(segment)
        return True

    def GetSegment(self, sid):
        return self.segments.get(sid)


class SurfaceNode:
    def __init__(self, node_id, segmentation, managed):
        self.node_id, self.segmentation, self.managed = node_id, segmentation, managed

    def GetID(self):
        return self.node_id

    GetName = GetID

    def GetSegmentation(self):
        return self.segmentation

    def GetAttribute(self, key):
        return "review/manifest.json" if self.managed and key == "XTA.ReviewManifest" else None


def commit_api(nodes, display):
    from contextlib import nullcontext

    def set_labelmap(binary, node, sid, mode):
        segment = node.GetSegmentation().GetSegment(sid)
        segment.representations[BINARY] = binary
        if CLOSED in segment.representations:
            segment.representations[CLOSED] = SurfaceSegmentation.convert(segment)
        return True

    def clear(node, sid):
        node.GetSegmentation().GetSegment(sid).representations[BINARY] = None  # surface left in place
        return True

    logic = SimpleNamespace(SetBinaryLabelmapToSegment=set_labelmap, ClearSegment=clear, MODE_REPLACE=0)
    slicer = SimpleNamespace(
        mrmlScene=SimpleNamespace(GetNodeByID=lambda node_id: nodes.get(node_id)),
        util=SimpleNamespace(RenderBlocker=nullcontext, NodeModify=lambda node: nullcontext()),
        vtkSegmentationConverter=SimpleNamespace(GetSegmentationClosedSurfaceRepresentationName=lambda: CLOSED),
        vtkSlicerSegmentationsModuleLogic=logic)
    return compile_names(("_xta_commit_binary_updates", "_xta_refresh_cleared_surfaces",
                          "_set_segment_binary_labelmap"),
                         slicer=slicer, nullcontext=nullcontext, _xta_active_review=display,
                         _xta_tag=lambda segment, key: segment.tags.get(key))


class CommitDisplay:
    def __init__(self):
        from contextlib import contextmanager
        self.depth, self.dirty, self.remembered = 0, set(), []

        @contextmanager
        def own_edits():
            self.depth += 1
            try:
                yield self
            finally:
                self.depth -= 1

        @contextmanager
        def transaction():
            yield self

        self.own_edits, self.transaction = own_edits, transaction

    def remember_previous(self, node, sids):
        self.remembered.append(list(sids))

    def mark_dirty(self, ids):
        self.dirty.update(ids)


@pytest.mark.parametrize("managed", [True, False])
def test_commit_reconverts_only_emptied_segments_with_stale_surfaces(managed):
    display = CommitDisplay()
    masks = {sid: np.ones((2, 2, 2), np.uint8) for sid in ("kept", "shrunk", "emptied", "also_emptied")}
    review = SurfaceSegmentation(masks, surfaces=True, owner_depth=lambda: display.depth)
    plain = SurfaceSegmentation(dict(masks), surfaces=False)
    nodes = {"review": SurfaceNode("review", review, managed), "plain": SurfaceNode("plain", plain, managed)}
    untouched = review.GetSegment("kept").representations[CLOSED]
    smaller = np.zeros((2, 2, 2), np.uint8)
    smaller[0, 0, 0] = 1
    api = commit_api(nodes, display)
    api["_xta_commit_binary_updates"]({
        "review": [("shrunk", smaller), ("emptied", None), ("also_emptied", None)],
        "plain": [("emptied", None)]})

    surfaces = {sid: s.representations.get(CLOSED) for sid, s in review.segments.items()}
    assert surfaces == {"kept": ("surface", 8), "shrunk": ("surface", 1),
                        "emptied": ("surface", None), "also_emptied": ("surface", None)}
    assert surfaces["kept"] is untouched  # neighbours are not reconverted
    # One conversion per node, for the emptied segments only, inside the display's own-edit guard.
    depth = 1 if managed else 0
    assert review.log == [("remove", "emptied", CLOSED, depth), ("remove", "also_emptied", CLOSED, depth),
                          ("create", CLOSED, depth)]
    # A node without closed surfaces gets none.
    assert plain.log == [] and all(CLOSED not in s.representations for s in plain.segments.values())
    if managed:
        assert display.dirty == {"shrunk", "emptied", "also_emptied"}
