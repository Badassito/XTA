# Slicer cleanup to native-resolution PTA labels

The cleanup tool transfers **deletion evidence** from edited low-quality (LQ)
segments onto their original full-quality (FQ) components. The original native
voxels supply the detail. It publishes one cleaned native Global NRRD for PTA;
it does not upscale the edited Global or regenerate the 300+ native layers.

Validate the first matched real dataset before accepting automatic decisions as
training ground truth.

## Keep from the TTA run

Preserve both original `nrrd/` and `low_quality/<bin>/nrrd/` directories,
including their `*_nrrd_manifest.json` files, and the original intensity source.
Use FQ and LQ files from the same run. Keep the originals separate from the
edited Slicer scene. The manifests alone cannot recover discarded FQ voxels.

You can leave the FQ files on the cluster. Slicer needs only the LQ files and,
for visual inspection, the full-resolution intensity. The edit bundle is the
small handoff back to the machine holding FQ data.

## Load the toolbar

In Slicer's Python console, execute the maintained script, adjusting its path:

```python
exec(open(r"C:\path\to\XTA\tools\slicer\xta_slicerrc.py", encoding="utf-8").read())
```

This replaces the My Tools toolbar. It does not run Importer or change voxel
arrays on load. The script is self-contained in Slicer. Keep a copy of an
existing startup script before choosing to replace it permanently.

### Multi-cutoff review

Load the matching intensity in Slicer. In a fresh scene click **Importer** and
select the sweep's `review/manifest.json`. Each grouped grid folder, such as
`review/low_quality/<bin>/`, also holds a byte-identical copy of that
`manifest.json` beside its group files, coverage labelmaps and color table.
Choosing that copy in Importer opens the same grid. Dropping the whole folder
into Slicer instead loads its group files as
[per-cutoff segments](#per-cutoff-segments); then click **Importer**. If Add
Data offers `manifest.json` as some other data type, leave it unchecked:
Importer reads it. While the folder is still inside its review, the copy
resolves to `review/manifest.json`, so the sweep receipt check, other grids and
export are unchanged. A folder copied elsewhere on its own still loads,
displays and edits, but **Export XTA Edits** then stops with a message: export
binds each cutoff to its scalar branch manifests in the complete sweep output.
Every grid numbers its group files from `group_0001`, so with several
low-quality grids a folder copied on its own is identified by the grid ID in
its group file headers, even when it was renamed. Reviews published before this
copy existed work from `review/manifest.json`; copying that file into the grid
folder by hand has the same effect.

Importer offers grouped review grids only, smallest low-quality grid first.
When the sweep has a low-quality tier, its native grid is reference-only: it
binds exported edits to the native branch NRRDs and is never loaded. Use the
matching LQ intensity or confirm **Align Native Intensity** when pairing a
full-size intensity stack with an LQ grid.

**Store-backed review.** By default Importer opens the grid store-backed: every
cutoff is held in one in-memory store and stays editable. After the grid,
Importer asks which cutoff to display, listing the review default first, then
the other cutoffs, then **Per-cutoff segments (legacy)…**. If the grid already
has an edit log, it then asks **Resume N edits (last …)** (the default) or
**Start fresh (archive the log)**; see
[Edit log and reopening](#edit-log-and-reopening). Then it aligns the
intensity if needed, reads every group file of the grid into the store (a
progress dialog names the cutoff being read; **Cancel** stops), writes the
working segments at the displayed cutoff, replays the log and installs the
slice overlay. A cancel or failure at any step removes what the
import created and reopens the previous review display. Running **Importer**
again on an open store-backed review does nothing; on a view-only one it
rebuilds the store.

Two working segmentations hold the displayed cutoff. **XTA Review Layers** has
one segment per logical layer except the Global, named without its cutoff
suffix and kept in one color, its lowest cutoff's (the published colors can
differ per cutoff). **XTA Review Global** holds the Global. **2D Cutoff**
rewrites both in place at the chosen cutoff. The 3D view shows only the
displayed cutoff's Global, and the 3D button reads **3D: Global Only**. The
layers stay 2D: surfaces that Segment Editor's **Show 3D** adds to XTA Review
Layers are removed at the review's next write.

**Fallback to per-cutoff segments.** Importer keeps the earlier per-cutoff
loading when you choose **Per-cutoff segments (legacy)…**. It also uses it on
its own for a review packaged from independent scalar runs (their cutoff masks
need not nest), a native-resolution review grid, a manifest the store cannot
use, and a store that would not fit in memory; the status bar says which. A
damaged review does not fall back: when a group file is missing or unreadable,
or its voxels disagree with the manifest, the store-backed import stops with an
error naming the file and removes what it created. To open such a review
anyway, choose **Per-cutoff segments (legacy)…**; that path loads only the
cutoffs you pick and does not check voxel counts against the manifest.
Before reading any mask, Importer estimates the store-backed review's memory
from the manifest: the layer arrays (one byte per voxel of each layer's box,
two or more bytes for a layer whose cutoff masks do not nest), two bytes per
grid voxel for the store's maps, the working labelmaps, the 2D display fields
and cache, decode buffers and up to 2 GiB of undo history. When that exceeds the
physical memory available, it opens per-cutoff segments instead. The status
message after a store-backed import gives the store size and the estimate. A
scene with per-cutoff segments loaded keeps that path, and one scene cannot
hold both kinds.

Programmatic entry supports `ImportXTASegments(manifest_path,
grid_id="low_quality/half", volume_node=intensity_node, cutoff_cents=70)` with
an actual grid ID from the package: it opens the grid store-backed and displays
0.70. A list such as `cutoff_cents=[50, 70]`, `cutoff_cents="all"` or
`review_mode="legacy"` loads per-cutoff segments instead; `review_mode="store"`
raises instead of falling back. `edit_log="resume"` or `edit_log="fresh"`
answers the edit-log question. Without a main window Importer displays the
review default and resumes an existing log.

**2D Cutoff** selects the confidence cutoff shown over the intensity, or
restores the original segment display. Its colors encode how many direct
prediction layers at that cutoff cover the pixel. The count starts at one,
including a single weakly supported location. Foreground supplied only by an
accepted interpolation or extrapolation, and remaining Global-only foreground,
have separate colors. Global and generated masks do not add direct votes.
Switching the displayed cutoff does not change segment eyes.

The working label values remain the actual direct-layer counts. Colors group
counts into 1, 2–3, 4–7, 8–15 and 16+. The display refreshes from source-mask
edits, while ordinary slice navigation reads the existing label volume. Toolbar
edits update only the voxels of the segments they changed; the previous
contribution of each changed segment is kept even when it is not in the cache.
`XTA_SWEEP_DISPLAY_CACHE_MIB` controls the optional packed contribution cache
(256 MiB by default). Other source changes missing from the cache recompute the
affected region from current sources. The array sizes depend on the selected
grid and maximum possible layer counts. An explicit `ImportXTASegments(...,
max_working_mib=...)` or `XTA_SWEEP_DISPLAY_MAX_WORKING_MIB` sets a working
memory limit for that workstation.

### Editing a store-backed review

Place normal **F**, **L** or **C/CC** markers over the structure and use
**Remove Islands** or **Remove Slices** once. A marker must touch a visible
layer at the displayed cutoff: a marker on voxels that only hidden layers or
the Global show there changes nothing, and the status message names those
layers. With **Scope: All Cutoffs** (the default) one click edits every cutoff
of the visible layers:

- At the displayed cutoff each visible layer loses the 6-connected islands the
  marker touches, as on per-cutoff segments.
- At stricter cutoffs it loses the same voxels.
- At looser cutoffs, where the marked island usually joins larger regions, the
  click removes only the looser growth of the removed island. A priority flood
  over the union of the visible layers, ordered by the strictest cutoff that
  still shows each voxel, splits that region between the removed island and
  the foreground kept at the displayed cutoff (the union basin). Kept and
  painted voxels win ties. This removes less than a click at that cutoff would,
  which takes the whole island holding the marker there; display that cutoff
  and click again to remove more.

**Scope: Displayed+Stricter** removes the displayed islands at the displayed
and stricter cutoffs only. A store-backed review has no **Displayed** scope
(per-cutoff segments keep it): a scene saved with it opens as
Displayed+Stricter, and `SelectEditScope("displayed")` refuses and changes
nothing (see Per-cutoff segments below for how the two kinds keep their
scopes). **Remove Slices** follows the same
rules on the XY, YZ and XZ planes through each seed; at looser cutoffs its
flood stays in each plane. Hidden layers never change. A click does not edit
the Global directly: where the union of the additive layers loses voxels, the
Global loses them too and is marked stale until **Update Global**.

Each click writes one status-bar line: the voxels removed at the displayed
cutoff, with their share of the visible layers there; the voxels removed at
looser and at stricter cutoffs; the number of layers changed; and whether the
Global is stale. The Python console log adds the counts per layer and cutoff.
When a flag fires, a non-modal **click report** window lists the flagged
layers. A layer is flagged when it lost a quarter or more of its voxels, or
more than 100,000 voxels, at the displayed cutoff, or a quarter or more at a
looser cutoff. The click is flagged when it removed 0.2% or more of the
visible layers' voxels at the displayed cutoff. The window blocks nothing;
**Undo** reverts the click.

Large clicks ask first. When the islands found at the displayed cutoff,
counted once across layers, pass 500,000 voxels, the click asks **Cancel
(change nothing)**, the default, or **Remove them**. When the looser growth
would remove more than 500,000 voxels beyond those, it asks **Displayed and
stricter cutoffs only** (the default; the Displayed+Stricter result), **Remove
at every cutoff** or **Cancel (change nothing)**. Esc cancels. Without a main
window the defaults apply.

Segment Editor paint and erase on the working segmentations are synced into
the store about 250 ms after each stroke, and before every store action. Each
sync is one undo entry for every layer it changed. Erase removes voxels at
every cutoff; paint adds them at the displayed cutoff and every looser one.
Painting a layer does not add to the Global until **Update Global**; an erase
follows the click rule above. The Global itself is painted and erased like a
layer. While the review is open, Segment Editor's **Modify other segments** is
set to **Allow overlap**, and a change is turned back while the editor shows a
working segmentation: overwriting would erase the other layers at every
cutoff. The editor's own undo is off while it shows a working
segmentation. Both settings return when the review closes.

A working labelmap that was resampled or moved off the review lattice cannot
be synced: it is rewritten from the store with a warning, and the stroke is
lost. Voxels painted outside the review grid are dropped. A segment added to a
working segmentation is not part of the store, so it is not synced, undone or
exported. A deleted working segment comes back at the review's next write.
Painting a layer at a cutoff where the review has no row for it is allowed,
but exporting that cutoff then stops until those voxels are erased or the paint
is undone. When a sync fails, the status bar says why, and a prompt offers
**Keep them** (the default; sync again at the next action) or **Discard the
unsynced edits** of the layers concerned.

**Undo** and **Redo**, on the toolbar after Remove Markers, and Ctrl+Z /
Ctrl+Y undo and redo the newest store edit at every cutoff it changed: a
click, a Segment Editor sync or an update. A pending stroke is synced first, so
Undo right after a stroke reverts it. A new edit clears Redo. The keys belong
to the review while an editable store-backed review is open; while the Segment
Editor shows another segmentation they are its own again. The history keeps up
to 2 GiB of compressed changes; past that a new edit drops the oldest steps.
Undo and Redo apply only to a store-backed review.

On a store-backed review **Update Global**, **Update Nonglobal** and **Update
Both** run at every cutoff, each as one undo entry (Update Both included), with
the per-cutoff rules. Update Global keeps the largest 6-connected island of the
union of all additive layers at each cutoff, hidden ones included. Update
Nonglobal clips each additive layer to its cutoff's Global, skipping cutoffs
whose Global is empty. Other segmentations in the scene are updated as before
in the same click.

### Edit log and reopening

Every store edit, undo and redo is appended to an edit log and written to disk
at once, so a crash loses nothing. The log lives beside the sweep folder that
holds `review/`: for `sample_sweep/review/manifest.json` it is
`sample_sweep.cutoff_edits/<token>/edits.xtalog`, where `<token>` is the first
12 hexadecimal digits of the SHA-256 of the grid ID. A grid folder opened on
its own uses `<folder>.cutoff_edits/` beside it. The review tree is never
written. Move the `.cutoff_edits` folder with the sweep folder; a rebuild
refuses when the review moved and its log did not.

**Resume** rebuilds the store from the review and replays the log, undo and
redo history included. **Start fresh** renames the log to
`edits.archived-<UTC time>.xtalog` beside it and opens the review as
published. A damaged last record is cut off with a warning. A log that is
damaged or does not match the review cannot be replayed; Importer offers to
archive it and start fresh. A log held by another Slicer session offers **Open
view-only**, which replays its edits but cannot edit or export. When the log
cannot be written, Importer offers to edit without it. When writing stops
later, the review keeps working and its status messages say **EDIT LOG
STOPPED**: later edits are not saved.

Reopening a saved scene restores the working segmentations view-only. A prompt
then offers **Rebuild the store and replay N logged edits** or **View only for
now**; **Importer** also rebuilds it later. Segment Editor edits on a view-only
review are not saved in the store or the log, and the status bar warns once.
When the saved working segments differ from the rebuilt store, the rebuild asks
whether to keep those edits as one Segment Editor edit (the default; Undo
discards them), discard them, or stay view-only. Closing the scene or deleting
a working segmentation frees the store; the log stays. Run **Importer** to
reopen the review; it first offers to remove working segmentations left
without their display. Executing the script again keeps an open store-backed
review. Importer ignores a copy of a working segmentation, and Remove Islands
and Remove Slices refuse while a copy is visible.

### Export from a cutoff review

For native transfer work on an LQ grid, run **Export XTA Edits**. Run
**Update Both** first only when each cutoff's Global should be rebuilt from its
layers with the largest-island rule (see **Stale Global** below). The output
includes a sweep index and independently bound per-cutoff
child edit bundles in the `xta-slicer-edits-v1` format. Use the chosen child
with the corresponding original LQ and FQ manifests. Keep the original branch
data and sweep review manifest available for export and transfer.

On a store-backed review the toolbar button exports the cutoff the working
segments hold, straight from the store, also with **Original segment display**
selected. `ExportXTAEdits(review_manifest)` in the console exports every
cutoff, and `cutoff_cents=[...]` a subset. Before writing anything, export
checks that the grid is unit LPS and that no exported layer holds voxels at a
cutoff where the review has no row for it. On per-cutoff segments the toolbar
button exports the displayed cutoff, which must be loaded, and
`ExportXTAEdits(review_manifest)` exports every loaded cutoff.

**Stale Global.** A click or an erase that removes voxels from the additive
layers' union leaves the Global stale. Update Global keeps only the largest
6-connected island of that union, while the published Global can keep every
island, so updating can remove a large share of it. When the Global is stale
and Update Global would change it at an exported cutoff, export asks **Update
Global first?** It lists, for each such cutoff, the voxels Update Global would
remove and add there and their share of that cutoff's current Global, from a
dry run on the store. **Export as is (the stale Global)** is the default.
**Update Global first, then export** runs Update Global, not Update Nonglobal,
at every cutoff as one undo entry, then exports. Esc exports nothing. Without
a main window a stale export stops; pass `if_stale="update"` or
`if_stale="warn"` (export as is) to `ExportXTAEdits`.

### Per-cutoff segments

Importer uses this earlier path in the cases listed above. It asks which
cutoffs to load as editable masks, with the review default first, or all of
them. Click **Importer** again later to add cutoffs; loaded cutoffs stay.
Alternatively, load grouped `.seg.nrrd` files through **Add Data** and click
**Importer**; each group normally holds one cutoff (groups in some early sweep
reviews mix cutoffs, and Importer handles both). Importer recognizes the published
segment IDs, completes each loaded cutoff, creates the normal smoothed surfaces
from complete branch masks, and installs one slice overlay. Source colors and
logical IDs are preserved. A cutoff that is not loaded is shown from its
published direct-count labelmap, read-only and without the generated-only and
Global-only accents. By default the 3D view shows only the displayed cutoff's
surfaces; **3D: Show All Cutoffs** shows every loaded cutoff, and the button
then offers **3D: Show Displayed Only**.

Markers work as above, but each visible source segment in the edit scope
receives the operation on its own complete mask and geometry, a visible Global
included. The scope button next to **2D Cutoff** names the current scope:
**Scope: Displayed** (the default) edits only masks at the displayed cutoff,
**Scope: Displayed+Stricter** also edits higher cutoffs, and **Scope: All
Cutoffs** edits every visible loaded cutoff, each with its own connectivity.
Nested masks connect more widely at lower cutoffs, so the island holding a
marker can be far larger there. Ordinary segments ignore the scope. The status
message reports the scope and how many visible masks it left unchanged. Only
masks that a marker actually reaches are
copied and searched, and only segments whose voxels changed are written back:
Slicer regenerates the closed surfaces of those segments (a segment emptied
by the edit gets an empty surface), and the 2D display updates their voxels.
**Update Global** rebuilds
each loaded cutoff from that cutoff's additive layers, preserving the existing
largest-island rule. **Update Nonglobal** clips those same layers to their own
nonempty Global. **Update Both** runs the two operations in that order. As in
a normal scene, hidden contributors participate in recomposition, so a
Global-only deletion can be restored from retained source masks. Both read the
loaded labelmaps in place and write back only a Global or layer whose voxels
change; the status message reports how many changed. The resulting masks are
the same as before.

Per-cutoff segments and store-backed reviews remember separate scopes for the
Slicer session, and each review saves its scope with the scene. A choice never
changes the other kind's scope, so per-cutoff segments keep Displayed, or your
last per-cutoff choice, after a store-backed review used All Cutoffs. With no
review open, the button shows and sets the per-cutoff scope only. A new
store-backed review opens at the last scope chosen for a store-backed review
(All Cutoffs in a new session); change it with the button once the review is
open.

### Other toolbar actions

The My Tools panel uses two rows of nine buttons. The first runs from
**Importer** to **Remove Slices**; the second starts with **Remove Markers**,
**Undo** and **Redo** and ends with **2D Cutoff**, the scope button and the 3D
button. One **Show All / Hide All** button toggles every segmentation node. Its
label names the next action. When the scene is entirely shown or entirely
hidden, the button follows the scene instead.

**Highlight** draws a red silhouette outline of a surface model in every 3D
view. With one model, it uses that model; with several, it asks which one;
with none, it reports an error. Click Highlight on an outlined model to remove
its outline. The outline follows the model's visibility and transform.

**Build Video** exports a cumulative reveal: Transverse, Sagittal and Coronal,
then the other view segments in natural name order. It detects the dataset name
prefix from the segment names, asking when several datasets are loaded, and
asks for a parent folder. Each export writes a new `SlicerBuild_<timestamp>`
folder with one PNG per state, `timeline.json` and a lossless 3840 x 2160,
30 fps FFV1/MKV. FFmpeg must be on `PATH` or set in Screen Capture > Advanced.
It renders privately from the first 3D view's camera; scene visibility and the
camera are not changed. On a store-backed review it renders the cutoff the
working segments hold; it adds surfaces to XTA Review Layers only while it
prepares the video. For other settings, call it from the console, for
example `BuildVideo(make_mp4=True, fps=60)`.

**Remove
Slices**, between Remove Islands and Remove Markers, provides a more precise
deletion tool: each marker removes the XY, YZ and XZ cross sections of its
6-connected foreground island through the marker's voxel position. These are
planes of the segmentation voxel grid, including when the segmentation is
transformed in the scene. Other islands on those planes and voxels outside
the selected planes are preserved. It uses the same F points, L lines and
C/CC closed curves as Remove Islands, and processes only visible segments.
Lines and curves select planes through every foreground voxel they intersect,
so extensive markers can still remove an entire small island. Use a single
point to make the narrowest three-plane selection. Marker nodes remain until
Remove Markers is clicked.

## Align full-resolution intensity with the LQ segments

Load the full-resolution scalar intensity and the matching LQ segmentations.
Run Importer as usual, then **Align Native Intensity**, selecting the original
LQ manifest and the full-resolution intensity node.

XTA's FQ and LQ NRRDs each use unit spacing and zero origin. Their different
array sizes therefore occupy different physical extents in Slicer. The button
uses the manifest dimensions to apply a reversible linear transform to the
intensity node. Segmentation arrays, intensity pixels and markers are preserved.
Only display coordinates change, so fine intensity features remain available
without expanding the segmentation collection.

The mapping is based on voxel-cell boundaries, per axis:

```text
preview_index = (native_index - native_extent_start + 0.5) * LQ_size / FQ_size - 0.5
```

It supports noninteger and unequal axis ratios and accounts for the intensity's
IJK-to-RAS matrix. It requires the same source, frame range and stored X/Y/stack
order as the XTA output. Matching dimensions cannot establish those facts or
detect a flip. Check landmarks in all three planes on the first dataset.
Pooled LQ foreground represents a region of native support, not a unique native
voxel; an LQ boundary need not track the fine intensity boundary exactly.

Keep this transform **unhardened** and save the Slicer scene to retain it.
**Reset Intensity Alignment** detaches it. Scene coordinates remain in the LQ
index frame; displayed distances are not calibrated physical measurements.
Avoid Segment Editor's geometry-resampling controls if you want to retain the
small segmentation grids.

For a separate, simple upscaling comparison, use the original full-resolution
source volume in Segment Editor, then run:

```python
exec(open(r"C:\path\to\XTA\tools\slicer\xta_upscale_preview.py", encoding="utf-8").read())
```

On the Ubuntu Slicer workstation, copy just `xta_upscale_preview.py` there and
use its local path instead; no XTA installation is needed:

```python
exec(open("/home/YOUR_USERNAME/xta_upscale_preview.py", encoding="utf-8").read())
```

The script finds Global automatically and leaves all nonglobal layers alone.
If Align Native Intensity already recorded the LQ/FQ grid sizes, no manifest
prompt is needed. Otherwise choose the original LQ manifest: the cropped Global
bounding box alone cannot establish the original field of view. If several
datasets are loaded, choose the intended segmentation node in Segment Editor.
This creates a new full-resolution segment using
3D cubic interpolation and threshold 0.5, with the volume's geometry and parent
transform. It processes interpolation in small slabs, but the resulting uint8
mask and Slicer's copy may need about two bytes per native voxel in additional
RAM. It preserves the original LQ segments. It cannot recover native detail and
may change thin structures. Hide the original for visual comparison, then remove
the comparison segmentation before Export XTA Edits, which expects only the
original manifest's working segments. `xta_upscale_preview.py` has not been
checked in a live Slicer session.

## The extended workflow

1. Run TTA and retain the matching FQ/LQ NRRDs and manifests.
2. Load LQ NRRDs in Slicer, run Importer, then Update NonGlobal.
3. Align the full-resolution intensity if needed. Use markers and Remove Islands
   for the manual cleanup.
4. Run Update Both and save the scene.
5. Click **Export XTA Edits**, select the original LQ manifest, and choose a new
   output directory. It snapshots current LQ labelmaps and source identities.
   Hidden working segments are included. Preserve original segment names and
   use one source/bin per scene.
6. Transfer that edit directory to the FQ machine and run native filtering.
7. Review `report.json` and the first real result. Pair the one output mask with
   the original intensity source and run PTA with the desired 3D views.

The original LQ Global and the nonglobal originals let the processor distinguish
the initial Update NonGlobal clipping from subsequent manual deletions. Export
after Update Both: the exported Global must equal the nonglobal union. Additions,
painted boundary reconstruction, transformed segmentation grids and renamed
segments are outside this deletion-transfer contract.

## Run native filtering

The processor uses Python 3.10+, NumPy and SciPy. Auto mode also uses
scikit-image. In an existing XTA environment, `pip install scikit-image` supplies
that dependency; a new XTA installation can use the `cleanup` extra. For a
standalone machine, install `numpy scipy scikit-image` and keep
`tools/xta_native_filter.py`, `tools/xta_native_sparse.py`, and `tools/xta_auto.py`
together.

Example cluster command, from the XTA source directory (replace example paths):

```bash
python tools/xta_native_filter.py --auto \
  --edits /work/M1_edits \
  --low-manifest /work/TTA/low_quality/bin/nrrd/M1_nrrd_manifest.json \
  --full-manifest /work/TTA/nrrd/M1_nrrd_manifest.json \
  --output /work/M1_clean_native \
  --scratch /local/scratch/xta-cleanup \
  --workers 16 \
  --block-mib 128
```

Use a new output directory. The input originals are read-only. The output is
`M1.nrrd`, `report.json`, `components.tsv`, and small review previews when needed.
No FQ candidate or filtered-layer collection is written.

Without `--auto`, mixed keep/delete evidence stops processing with exit code 2
and `needs_review`, without publishing a PTA label. With auto mode:

- A native component with only deletion evidence is removed whole; one with
  only retention evidence keeps its original detail.
- Mixed components are tested for separable native distance-transform cores.
  Watershed proposes a cut within the original foreground. Evidence checks
  reject cuts that lose too much retained evidence or remove too little rejected
  evidence. No foreground is added.
- If any mixed component stays unresolved (for example no separable core, a cut
  that fails the evidence checks, or a padded box above
  `--auto-max-roi-voxels`), processing stops. `report.json` and
  `components.tsv` record it, review previews remain for inspection, no PTA
  label is published, and the exit code is 2 with `needs_review`. A
  deliberately empty edited Global produces an empty native label.
- The filtered layers are unioned at native resolution. The largest six-connected
  union component is retained by default, matching the toolbar's single-object
  workflow. `--keep-objects N` can retain multiple native fragments of that
  connected preview object; the edited LQ Global must still be connected (or empty).

Defaults are a maximum core-distance radius of 2 native index voxels, at most
1% loss of exclusive keep evidence, at least 90% removal of exclusive delete
evidence, and an 8,000,000-voxel padded bounding-box limit per auto case. These
are heuristic settings, not validated anatomical error bounds. Set
`--auto-max-keep-loss 0` for strict preservation of exclusive keep evidence.

Cuts operate per layer: overlap from another retained layer can restore a
connection in the final union. There is no second global cut optimization or
intensity-guided adjudication. Tiny structures merged by downsampling can remain
ambiguous; inspect the report instead of treating successful execution as proof
that every manual deletion was reproduced.

`complete` and `completed_with_heuristics` are the completion states of the
default and auto policies; only `complete` needs no heuristic cuts.
`--conflicts keep` instead preserves unresolved mixed components and publishes a
conservative candidate with status `completed_with_unresolved_components`. All
three exit 0; see [native union recovery](native_union_recovery.md).
`review_previews/` contains diagnostic overlays, which should not be added to
the working segmentation or its Global union.

Current original inputs must be XTA inline raw/gzip, full-raster 3D uint8 binary
NRRDs with unchanged unit-LPS geometry. Cropped **edited** snapshots are supported.
Nonadditive checkpoint or audit layers are rejected by default; load the intended
working component layers and Global. `--allow-nonadditive` is an explicit escape
hatch for reproducing the toolbar's literal union of all loaded layers, not a
normal XTA recomposition policy.

## PTA handoff

Copy the cleaned `M1.nrrd` beside `M1.mkv`, or beside the original supported image
sequence whose parsed volume stem is `M1`, in a dedicated PTA input directory.
Use the original full-resolution image source with the same frame order and
dimensions. PTA currently takes an image sequence or video as intensity input;
a Slicer intensity NRRD is not a replacement for that input interface.

Do not put the diagnostics or additional NRRDs in that input directory. PTA
matches one NRRD by volume stem, and the native shape must match the images.
For example:

```bash
python -m XTA --mode pta --input /work/M1_pta_input \
  --output /work/M1_training \
  --enable_cartesian transverse,sagittal,coronal \
  --enable_azimuthal transverse:30 \
  --save images labels summary
```

Add the desired tilted/spherical views or augmentation policy using the existing
PTA flags. PTA preprocessing stays off unless explicitly requested. To fill
enclosed native 3D cavities, add `--preprocessing 3d_void_fill`. It fills only
background disconnected from the volume boundary (6-connected, the dual of the
26-connected foreground), before cube resizing or physical views. It can be
combined with `gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]`; void
filling always runs first, regardless of token order. Each operation may appear
once. These 3D operations require a fully labeled contiguous volume; `--force`
permits partial volumes, while encoded frame gaps and unlabeled volumes remain
excluded. Forced partial semantic exports retain unknown pixels as class 255.

YOLO polygon labels keep enclosed 2D holes after rendering and augmentation
through zero-width bridges to the outer ring. A bridge that crosses another hole
or a concave notch fills a 1-px line; holes below the 1-px polygon tolerance
stay filled and are counted as `holes_discarded_during_polygon_export`.
Binary and semantic outputs retain the actual rendered mask,
including holes that remain after the selected native preprocessing. Void fill
statistics and voxel counts are recorded in the PTA manifest and requested
summary. Its workspace uses about four bytes per native voxel plus component
tables; it falls back to scratch storage when RAM admission requires it.

PTA now uses the selected CUDA devices for large-mask void filling and Gaussian
smoothing. Preprocessing runs in a separate spawned process so the parent stays
safe for its existing render-worker lifecycle. Void filling labels bounded slabs
on multiple GPUs, reconciles their boundaries, then applies the result on the CPU.
Gaussian smoothing uses bounded 3D tiles with full filter halos.
Replicate-edge (nearest) boundaries, per-pass thresholding at 0.5, and
preprocessing order are shared with TTA. Structures crossing a volume face are
not eroded there; a feature lying only in a face slice is extended outward and
may survive a pass that would remove it in the interior.
Floating-point rounding can affect Gaussian voxels very near the threshold.

No new CLI flags are needed. The initial volume, or each volume with
`--pipeline_depth 1`, can use GPU preprocessing; overlapping preparation of later
volumes stays on CPU to avoid competing with render workers. Small masks stay on
CPU to avoid CUDA-process startup: Gaussian below 64 Mi voxels, void fill below
1 Gi voxels with multiple devices or 2 Gi voxels with one device. These are
conservative selection thresholds, not measured throughput guarantees.
`YOLO_TTA_GAUSSIAN_SMOOTHING_GPU=0` keeps Gaussian on CPU. The existing
`YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB` controls the PTA tile cap (256 MiB by default),
further limited by free VRAM.

The isolated helper needs one additional uint8 mask of host/shared memory for
void fill, or two for Gaussian, beyond the original input. Admission checks
available host memory and POSIX shared-memory capacity. The original mask is
committed only after the helper succeeds; failures retain it for CPU fallback.
Manifest preprocessing statistics record GPU backend/devices/chunks and helper
timing when that path is selected. Telemetry now separately measures void fill,
annotation-state scans, anchor collection, and render-array sharing.

Image-sequence loading uses the allocated CPU budget, including CPU PNG decoding.
Every volume can use the full worker budget for loading and preparation, including
alongside rendering with `--pipeline_depth 2`. GPU owners keep their NUMA-local
affinity and bounded thread counts. The startup log reports the shared budgets. More decode threads can
increase per-image scratch memory and contention with concurrent classification;
`--workers` still limits the overall budget.
At depth 2, the prepared next volume enters the worker queue before the current
volume's last tasks finish. Each free worker can advance independently; the old
volume's arrays remain owned until all of its tasks have drained. Existing task
tracing records `pta.worker_task_start` and `pta.worker_task_end` with the volume
stem and generation, alongside parent planning/submission events.
For unchanged transverse YOLO inputs, foreground classification reuses prepared
foreground-support statistics instead of first scanning the entire dense mask.
Transformed masks, semantic coverage, and other views retain the dense check.

Offline augmentation with a ratio above 1 requires an original foreground/background
decision for each selected full image and tile, even with `--background_percent 1`.
That decision keeps genuine background copies while dropping foreground copies
whose objects disappear during augmentation. `--fraction` selects originals before
this classification; a fraction of 1 retains all eligible queries.

For complete volumes with unrestricted backgrounds and an integer offline ratio,
the decision can reuse the canonical source mask already rendered for publication.
Eligible runs skip the separate classification pass, resolve all copies before
policy mutation, and report the measured classes from worker results. Fractional
ratios, background caps, partial inputs and unchanged YOLO polygon/ROI shortcuts
retain upfront classification. Tiny-mask support and polygon class remain distinct.

With a GPU policy, dense classification uses the process pool's CUDA categorical
renderers and retains the resident mask for subsequent rendering.
GPU pixel counts reproduce the existing segmentation classifier: zero is background,
16 or more pixels are provisionally foreground, and only 1-15-pixel masks need the
CPU polygon check. Semantic classification intersects separately rendered foreground
and known-coverage masks. This pass does not render intensity images or apply
augmentations. Unsupported layouts and prelaunch memory rejection retain the CPU
path; errors after CUDA work launches are reported rather than silently retried.
With `--pipeline_depth 2`, the next volume's classification tasks queue behind the
rendering volume's tasks in the pool's FIFO queue. Each owner switches volumes only
after its last task of the previous volume, at the usual drained handoff, so no
owner holds two volumes and the next volume is not sent to the CPU just because
another volume is rendering; owner-declined jobs keep the CPU path above.
`pta.classify.gpu.first_result` includes that wait. The log and telemetry report
GPU, CPU, and polygon-only job counts.

A negative polygon classification does not prove that the binary mask is empty:
tiny masks can contain pixels without an exportable contour. Publication checks
actual source-mask support before augmentation, preserving such originals and
legitimate growth of their pixels while still rejecting foreground synthesized
from a truly empty source.

Azimuthal coordinate tables and immutable sampling-policy identities are reused
across queries. Process workers receive individual bounded tasks; the shared
phase payload holds plans and volume handles rather than every planned output.
The parent still constructs the complete version list. Telemetry separates
augmentation planning, render-task construction, phase installation/submission,
and first worker metadata decoding, and records phase-payload sizes.

Resizing writes directly into the final shared arrays. Canonical uint8 NRRDs use
bounded raw/gzip reads; other layouts retain the general reader. Unforced partial
PNG sequences decode annotated centers and required channel context, retaining
encoded positions and validating unused PNG dimensions. Force and overlay outputs
retain full decoding.

GPU workers reuse native uint8 planes across tiles and share exact FP64 spherical
direction plans between intensity and categorical sampling. Augmentation batches
fill available slots across source groups while retaining bounded shape tails.
nvJPEG reuses bounded output buffers after copying validated compressed bytes;
CUDA fences and native error handling remain in place. Updated external policy
files have new content hashes, so their generated tags/seeds also change even
when fixed-seed comparisons establish equivalent augmentation behavior.

PTA FP64 CUDA spherical views also reuse exact QSC geometry in a bounded host
memo. `YOLO_TTA_PTA_SPHERICAL_HOST_CACHE_MIB` caps host tables per GPU owner
(default `3072` MiB; `0` disables the memo). Admission further limits that cap
using physical/cgroup RAM headroom and the configured GPU-owner count; swap is
excluded. Unknown or insufficient headroom keeps the existing uncached builder.
At 2048², each direction/validity table is 100 MiB. Construction scratch and
tables still held by borrowers remain charged beside cached entries.

The host memo shares geometry between categorical classification and intensity
rendering while retaining both 256 MiB GPU direction-cache budgets and the
existing FP64 QSC math. CPU reference and FP32 sampling retain their existing
paths. Classification-warmed tables survive first intensity source admission,
then retire at the successful fenced source-generation handoff.
`pta.cuda.spherical_host_cache` telemetry reports limits, hits/builds/waits, and
current/peak owned host bytes. Existing Foundation environment names are unchanged.

With `YOLO_TTA_PTA_GPU_NATIVE_T=1`, depth-only enlargement with unchanged X/Y
dimensions lets CUDA intensity workers retain the smaller native-depth image and reconstruct logical gray8 voxels
using the existing TTA kernels. Logical masks, physical-view plans and the host
intensity cube remain available unchanged for CPU fallback. The worker requires
a shared native uint8 source; other layouts retain the logical-cube upload.
It is opt-in because a logical-cube hardware texture can render faster when it
fits. Native-depth Azimuthal rendering uses the reference-linear pointer kernel
and skips the hardware texture, preserving intermediate gray8 voxel rounding.
`YOLO_TTA_PTA_GPU_NATIVE_T=0` retains logical-cube residency. The actual
`pta.render.source_residency` telemetry records admission, native/logical shapes,
upload bytes and saved GPU bytes. The geometry request record's
`native_t_fusion: false` means intermediate virtual-gray8 rounding is preserved;
it does not disable native-depth residency.

Each GPU owner builds its CUDA runtime (policy, render engine and
nvImageCodec encoder) in the pool initializer, while the parent loads and
prepares the volume; the parent starts the owners before any CUDA use, and this
assumes the Default GPU compute mode. Owners are forked where the host supports
fork; elsewhere (e.g. Windows) they are spawned and reload the policy file after
checking its content hash. A failed warm-up is
logged and the owner's first task then behaves as the lazy build did: it builds
the runtime itself and raises any failure. With native-depth
residency, an owner starts its first render task's intensity upload in the
background once classification has admitted the categorical source, using the
same admission and arguments; the first render task joins it. A declined or
failed early upload leaves the decision to that task. Set
`YOLO_TTA_PTA_GPU_OWNER_PREWARM=0` or `YOLO_TTA_PTA_GPU_INTENSITY_PRESTAGE=0`
to restore first-task construction or upload. Telemetry spans:
`pta.worker.gpu_runtime_prewarm`, `pta.worker.intensity_prestage` (also a gauge)
and `pta.worker.intensity_prestage_wait`.

## Resources and validation boundary

Native processing shares two disk-backed uint8 Global/union arrays: 2 bytes per
native voxel, plus sparse foreground-run labels, component tables, outputs and
working memory. The F3 `1931 x 3064 x 3022` example needs 33.3 GiB for those two
arrays, replacing the earlier 99.9 GiB dense workspace. Sparse label storage
depends on the number of foreground X-runs; dense or fragmented inputs can still
need substantial space. Fast local scratch helps.

`--workers` controls concurrent layers and defaults to 1. Multiple workers use
separate spawned processes with one BLAS thread each, sharing file-backed native
rasters. Sixteen is a conservative starting point; the supplied 600-GiB Foundation
job uses its allocated CPU count (64). Foreground-only writes store the same byte
value in overlapping regions, so there is no whole-layer union lock. Each worker
has its own LQ evidence,
component tables and bounded auto ROI. All original payloads are still decoded
and checked. Layers with no deletion evidence skip component analysis and copy
their clipped native foreground directly; their component-count fields are null.
The report records each layer's elapsed time and sparse-label bytes. Completion
messages identify finished layers, and TSV rows follow completion order.
With enough RAM, the operating system caches both shared arrays in memory.
Stage the compressed FQ/LQ NRRDs and edits on node-local storage when the source
filesystem is slow; do not copy the unrelated SAM artifact archives. Foundationv3
does this staging before native transfer. In tool v0.3.1, `merge_seconds` measures
merge work without the old whole-layer lock queue.

`--block-mib` controls per-worker slab work, not total RSS. Auto mode loads one
bounded component region and distance/watershed temporaries; its voxel limit is
not a RAM limit. No GPU is needed by the transfer tool. Copy the updated source
and restart in a new output directory; a running job cannot adopt these changes.

Synthetic tests cover native component decisions, conflict handling, geometry,
and the native writer-to-PTA reader handoff. Actual Slicer interaction and
anatomical validity require the retained matched FQ/LQ run and manual result.
Save that scene and edit bundle along with the originals for that validation.
