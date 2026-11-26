# Native-window cleanup in 3D Slicer

An archive produced with `--save low_quality:0.20:xtareview` opens in the
full-field cutoff editor through **Importer**, **Add Data**, or
`OpenXTAReview`. It retains independent preview masks and an edit log beside
the archive. **Export XTA Edits** writes a `preview.xtareview` snapshot of the
edited Global for PTA preparation. Native archives use the window editor
described below.

Use this workflow to clean a managed TTA sweep at its original voxel resolution.
An `.xtareview` file holds the exact masks by view, cutoff and spatial block.
Slicer opens a navigation overview and presents the neighborhood around the
current position as ordinary editable native labelmaps. The mask exported for
PTA or final delivery is assembled on the original grid.

## Prepare a study

Run these commands in the XTA checkout with the completed sweep and its original video.

```shell
python tools/prepare_native_review.py STUDY/review/manifest.json sample.xtareview --intensity sample.mkv --prepare-update-both
python tools/build_slicer_review_bundle.py xta-native-slicer.zip
```

Preparation reads the indexed native groups, checks their source hashes and
decoded voxel data, and writes one spatially indexed archive. Use a new output
filename and allow working space beside it. This is a one-time conversion of
existing TTA output. With `--prepare-update-both`, preparation also computes
the initial connected-component update and writes
`sample.initial-update.sqlite` beside the archive. This pays the computation
once ahead of editing. Copy the archive, prepared companion, and portable
toolbar ZIP to the workstation. The original group folders can stay in the
study.

To prepare an archive you already have, run
`python tools/prepare_native_review.py sample.xtareview --prepare-update-both`.
If you saved the result of **Update Both** as the very first and only action
in a previous journal, pass `--from-existing-journal <path>` to reuse that
trusted result. This validates the journal's source binding, mask changes,
and cutoff totals before publishing a new companion. Both the archive and
the supplied journal stay intact. Use a journal whose provenance you know.

`--intensity` also accepts an embedded raw or gzip scalar uint8 NRRD on the
native grid. Video preparation checks frame dimensions and count. When the
TTA receipt contains a source digest, it checks that digest. For a relocated
video with file metadata only, it checks the filename, size and modification
time and records the supplied video's SHA-256. With intensity attached, both
the grayscale neighborhood and the working masks load on demand. A mask-only
archive supplies its original Global as a coarse navigation reference. You
can also supply an already aligned Slicer intensity volume. Its memory use
is additional to the window.

## Open and edit

The portable toolbar was tested in 3D Slicer 5.12.4, using its Python, NumPy,
SciPy, SimpleITK and VTK. In a fresh Slicer Python console, set the ZIP path
and install the toolbar.

```python
import sys
sys.path.insert(0, r"/path/to/xta-native-slicer.zip")
from xta_toolbar_startup import install
install()
```

With the portable toolbar installed, you can drag an `.xtareview` into Slicer
or choose it in **Add Data**. To open an archive directly in the Python console,
use `OpenXTANativeReview(r"/path/to/sample.xtareview")`. All three routes open
the native working window and its edit journal.

For a shared checkout, install from its toolbar directory.

```python
import sys
sys.path.insert(0, r"/path/to/XTA/tools/slicer")
from xta_toolbar_startup import install
install()
```

1. In a fresh scene, click **Importer** and pick the `.xtareview` file.
   Slicer may already show an empty **Segmentation** from Segment Editor.
   If the scene contains ordinary segmentation data, **Importer** merges those
   segments. Use **Add Data** to open an archive in that case. Use **2D Cutoff** to
   choose the displayed cutoff or switch between direct coverage and view
   colors. The coverage overlay counts visible direct contributors. Generated
   and Global-only foreground have separate colors.
   On opening, the 3D view shows the original Global as a translucent grey
   overview. Its surface uses the fraction of native foreground in each
   navigation cell. The finest branches can still merge or disappear at this
   scale. Choose **Native View → Both 3D Surfaces** to see the edited Global
   in cyan. **3D Edited Window** shows the edited surface alone. At close zoom
   the cyan mesh follows the native binary mask, so its edges can show voxel
   steps. Use **Center window at last marker** to inspect a fork before removing it.
2. **Update Both** finds the largest six-connected additive island at each
   cutoff, including hidden contributors, then clips each additive layer to
   its Global. This is one undo step across the study. XTA uses the prepared
   companion when the review is at its original masks. Opening the archive
   still shows the original masks until you click the action. Repeating it
   on an already settled state adds no undo step. After a local edit, XTA
   compares the old and new six-connected paths across the boundary of the
   edited area. When those paths prove which component wins, it updates only
   the affected blocks. A possible split, an unresolved tie, or a large edit
   can require a whole-grid calculation. The progress dialog shows the
   current phase. Cancel discards the pending update across all cutoffs.
3. Use the existing F points or L lines and **Remove Islands** or
   **Remove Slices**. The marker is evaluated on native voxels in the visible
   layers. Component searches can follow connections outside the displayed
   neighborhood. Slice removal acts on cross-sections through the marked
   component. **Scope** selects **All Cutoffs** or **Displayed+Stricter**.
   When separating looser-cutoff growth cannot be resolved exactly within
   the working budget, the action stops atomically. You can inspect the
   region and explicitly choose a different scope.
4. Pan or zoom in a slice or the 3D view. The editor sizes the native window
   from the active view's field of view and leaves a margin for scrolling.
   In 3D it uses the visible review surface under the center of the view
   when available. At wide zoom the native edit area can reach its memory
   limit. An archive with grayscale shows the intensity overview around the
   exact native patch. When the 3D field exceeds the loaded area, the edited
   Global can use a lower-resolution surface for context. Zoom in to inspect
   the native boundary. Slicer saves pending brush strokes in native
   coordinates before changing the window. **Undo** and **Redo** also work
   after you move or reopen the archive.
5. Use **Export XTA Edits** to save a cleaned native Global NRRD. In native
   mode the export is the volume itself. If Global needs an update, Slicer
   offers to rebuild it. To use the result in PTA, give the exported NRRD the
   same stem as the original intensity and place them together in the PTA
   input directory.

Keep the prepared companion beside the immutable archive with the same stem.
Slicer verifies it before applying its changes to your writable journal.
By default the journal is `sample.edits.sqlite` beside `sample.xtareview`. It holds the
native changes and undo/redo history. Reopen the archive to resume. Close the
review before copying the journal. On a read-only or high-latency share, use
a local journal directory with room for edits and whole-study scratch work.

New archives can include the finer 3D overview. When opening an older archive,
Slicer builds it from the original Global and caches it for later sessions
in `sample.xtareview.overview.sqlite` beside the archive if writable. The
overview cache can be regenerated.

The Python console can also open a review and jump to a native location.

```python
review = OpenXTANativeReview(r"/data/sample.xtareview", cutoff_cent=90,
                             max_working_mib=128,
                             journal_path=r"/local-work/sample.edits.sqlite")
SetXTANativeReviewWindow((ras_x, ras_y, ras_z))
ExportXTANativeReview(r"/output/sample.nrrd", if_stale="update")
CloseXTANativeReview()
```

Omit `window_shape` for zoom tracking. For a fixed-size request, pass
`window_shape=(64, 128, 128)` in native Z, Y, X voxels. The jump takes Slicer
world RAS coordinates. The display may shrink the requested window if many
layers occupy it. The working budget governs native mask windows and edit buffers.
Slicer's application, rendering and any separately loaded data also use RAM.
The overview supplies navigation. Exported voxels come from the native
archive and native journal. At a junction, an exact native marker can select
a different island from a marker evaluated on a downsampled preview.
