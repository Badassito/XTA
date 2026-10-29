# Topology options and centerline views

These options were introduced in v25.0.0, following the v24.2.2 patch checkpoint
(`1df0c28`). This guide describes the current checkout's behavior.

## Optional 2D hole filling

The spelling is `--enable_2D_hole_fill`. Its arguments depend on the mode:

| Mode | Syntax | Default without the flag |
| --- | --- | --- |
| TTA | `--enable_2D_hole_fill yolo`, `sam`, or `yolo sam` | Preserve holes in both stages |
| PTA | `--enable_2D_hole_fill` | No additional 2D filling |
| LTA | `--enable_2D_hole_fill` | Preserve SAM mask holes |

TTA also accepts `--enable_2D_hole_fill yolo,sam`. PTA and LTA accept no value
after this flag. A bare flag is invalid in TTA because it needs a stage.

**TTA migration:** YOLO previously filled 2D holes by default. Add
`--enable_2D_hole_fill yolo` to retain that behavior. Leaving it off skips hole
filling while retaining the existing accelerated inference paths. SAM continues
to preserve holes by default; selecting `sam` fills predicted masks before
encoding the memory used for the next prediction. Filling therefore affects
propagation, rather than only the final exported mask.
The SAM opt-in uses the shared exact 2D implementation, including a CPU mask
transfer from CUDA, because the SDK's size-limited fill occurs after memory
encoding. With the flag off, no filling hook or transfer is installed. SDF
interpolation and explicitly selected 3D postprocessing retain their own
topology behavior; this flag controls the YOLO and SAM stages.
YOLO polygon labels cannot encode enclosed hole contours; preserving holes in
the working binary mask does not add hole contours to that output format.

**LTA migration:** LTA previously filled propagation masks unconditionally.
v25 preserves their holes by default. Add the bare flag to restore filling
before subsequent SAM predictions.

**PTA:** The flag enables this complete label-processing sequence:

```text
native XY/XZ/YZ slices -> 2D hole filling -> native-mask closure
    -> 3D void filling -> ordinary PTA resizing, views and slicing
```

It automatically enables 3D void filling; adding
`--preprocessing 3d_void_fill` separately is unnecessary. Orthogonal slicing
and backprojection are exact array views, so the implementation writes filled
cross sections directly into the native mask. A deterministic XY, XZ, YZ sweep
can seal tunnels that otherwise connect a cavity to the volume exterior,
allowing the subsequent 3D fill to close it. These operations add foreground;
they do not modify intensity data. Existing Gaussian preprocessing, if selected,
runs after these fills. Use a complete labeled volume with adjacent source
slices. Existing partial-volume restrictions apply: `--force` can treat partial
labels as complete, while encoded slice gaps still disable these operations.

See the mode parsers in [TTA config](../XTA/config.py),
[PTA config](../XTA/pta_config.py), and [LTA config](../XTA/lta_config.py), plus
[PTA preprocessing](../XTA/pta_preprocessing.py) and
[SAM propagation filling](../XTA/sam_hole_fill.py).

## PTA centerline cross sections

Add `--enable_cpr` to generate stacks of cross sections perpendicular to each
extracted centerline's local tangent. A tubular object appears as approximately
circular foreground in these images. Each parent image puts its centerline
sample at the geometric center, between the central pixels for an even-sized
patch. Ordinary PTA tiles may place that sample anywhere
within the tile, or omit it.

The parent raster samples approximately one post-cubic-resize voxel per pixel
in both plane directions. `--imgsz 2048` directly selects a centered 2048-square
patch at that spacing. Successive cross sections are spaced approximately one
voxel along the centerline. Patches may extend outside the source volume;
those samples are zero for both intensity and labels. Intensity uses trilinear
sampling and labels use nearest-neighbor sampling. Explicit tile-output resizing
can change tile pixel spacing; parent patches retain the native sampling pitch.

CPR reuses TTA's embedded medial-ridge centerline extractor and generates views
per usable branch. It requires a complete labeled volume with adjacent slices;
the existing `--force` treatment of partial labels applies. Failed or unsuitable
centerline extraction produces a warning and no CPR views for that component.
The manifest records centerline identity, orientation, source raster dimensions,
and processing-grid spacing. See the [CPR user guide](pta_cpr.md),
[CPR geometry](../XTA/pta_cpr.py),
[PTA integration](../XTA/pta.py), and [view rendering](../XTA/pta_rendering.py).

## Slicer profile

The My Tools panel now has two rows. Show All precedes Hide All. Remove Slices
sits between Remove Islands and Remove Markers. It removes the selected
island's XY, YZ and XZ cross sections through each marker seed, preserving
unrelated islands and off-plane voxels. It supports the existing F points,
L lines and C/CC closed curves and respects segment visibility and transforms.
Large line/curve selections can cover enough planes to remove an entire small
island; a single point selects just three planes. See the
[Slicer workflow](slicer_pta_cleanup.md#load-the-toolbar) and
[profile source](../tools/slicer/xta_slicerrc.py).

## Can SAM predictions grow beyond their starting patch?

**LTA:** Yes, the production tracker can relay predictions into adjacent
detector tiles. Its default handoff requires at least 16 overlap pixels and
tracker probability of at least 0.5, so thin or low-confidence crossings can
fail. The optional dynamic backend also replans bounded crops between sealed
windows; its escape patches do not recurse. See
[tile tracking](../XTA/lta_tile_tracking.py),
[production adapter](../XTA/lta_worker_adapter.py), and
[dynamic crops](lta_dynamic_crops.md).

**TTA SAM interpolation:** It can cross old detector-tile seams inside its
planned family rectangle. Whole-crop SAM resizes that native rectangle to the
model input, so a 1008-pixel model input is not a 1008-native-pixel growth limit.
The rectangle stays fixed by default. Set `YOLO_TTA_SAM_ADAPTIVE_CROP=1` to
replay tracking from the original seeds in an enlarged crop, within the same
view canvas and available resources. Optional SAM child tiles remain independent:
neighbor predictions do not seed them, and outer adaptive replay does not solve
clipping at their internal boundaries. See
[SAM interpolation](sam_interpolation.md),
[adaptive crop replay](../XTA/sam_crop_retry.py), and
[child tiling](../XTA/sam_crop_tiling.py).

**Spherical and Radial TTA:** There is no SAM tracker handoff across separate
Spherical geometry patches, Radial arcs, or separate dynamic radius bands.
Each has its own view canvas. Existing projection can combine overlapping
reconstructions in source space, but that does not propagate tracker state
between these views. See the [SAM geometry contract](sam_interpolation.md) and
[view integration](../XTA/sam_integration.py).
