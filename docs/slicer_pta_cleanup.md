# Slicer cleanup to native-resolution PTA labels

The cleanup tool transfers **deletion evidence** from edited low-quality (LQ)
segments onto their original full-quality (FQ) components. The original native
voxels supply the detail. It publishes one cleaned native Global NRRD for PTA;
it does not upscale the edited Global or regenerate the 300+ native layers.

This is a development integration of the PTA_ENHANCEMENT draft. Validate the
first matched real dataset before accepting automatic decisions as training
ground truth. It is included in the v24.2.0 development source.

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
exec(open(r"C:\Users\Bry\Documents\ChatGPT\Test\tools\slicer\xta_slicerrc.py", encoding="utf-8").read())
```

This replaces the My Tools toolbar. It does not run Importer or change voxel
arrays on load. The script is self-contained in Slicer. Keep a copy of an
existing startup script before choosing to replace it permanently.

The My Tools panel uses two rows, with Show All before Hide All. **Remove
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
exec(open(r"C:\Users\Bry\Documents\ChatGPT\Test\tools\slicer\xta_upscale_preview.py", encoding="utf-8").read())
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
original manifest's working segments. Live Slicer execution still needs checking.

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
- Unresolved or oversized components are preserved and reported. A deliberately
  empty edited Global produces an empty native label.
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

`complete`, `completed_with_heuristics`, and
`completed_with_unresolved_components` are distinct completion states. All return
exit code 0; only the first needs no heuristic cuts and has no mixed components
under these rules. `review_previews/` contains diagnostic overlays, which should
not be added to the working segmentation or its Global union.

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
background disconnected from the volume boundary (6-connected by default),
before cube resizing or physical views. It can be combined with
`gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]`; void filling always
runs first, regardless of token order. Each operation may appear once. These
3D operations require a fully labeled contiguous volume; `--force` permits
partial volumes, while encoded frame gaps and unlabeled volumes remain excluded.
Forced partial semantic exports retain unknown pixels as class 255.

YOLO polygon labels already omit enclosed 2D hole contours after rendering and
augmentation. Binary and semantic outputs retain the actual rendered mask,
including holes that remain after the selected native preprocessing. Void fill
statistics and voxel counts are recorded in the PTA manifest and requested
summary. Its workspace uses about four bytes per native voxel plus component
tables; it falls back to scratch storage when RAM admission requires it.

PTA now uses the selected CUDA devices for large-mask void filling and Gaussian
smoothing. Preprocessing runs in a separate spawned process so the parent stays
safe for its existing render-worker lifecycle. Void filling labels bounded slabs
on multiple GPUs, reconciles their boundaries, then applies the result on the CPU.
Gaussian smoothing uses bounded 3D tiles with full filter halos. Constant-zero
boundaries, per-pass thresholding at 0.5, and preprocessing order are unchanged.
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

When the existing GPU-policy process pool is idle, dense classification uses its
CUDA categorical renderers and retains the resident mask for subsequent rendering.
GPU pixel counts reproduce the existing segmentation classifier: zero is background,
16 or more pixels are provisionally foreground, and only 1-15-pixel masks need the
CPU polygon check. Semantic classification intersects separately rendered foreground
and known-coverage masks. This pass does not render intensity images or apply
augmentations. Unsupported layouts and prelaunch memory rejection retain the CPU
path; errors after CUDA work launches are reported rather than silently retried.
Overlapping planning while another volume renders stays on CPU to avoid replacing
its resident mask. The log and telemetry report GPU, CPU, and polygon-only job counts.

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
