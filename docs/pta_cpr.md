# PTA curved planar reformation

`--enable_cpr` adds straightened cross-section views from a fully labeled PTA
volume. Each output frame is perpendicular to the local centerline tangent: a
tube produces a circular foreground cross-section. The centerline passes through
the geometric center of each square parent patch: the central pixel for odd
sizes, or between the four central pixels for even sizes. Ordinary PTA tiles
crop that parent and do not recenter it.

```text
python -m XTA --mode pta --input DATASET --enable_cpr --imgsz 2048 --save images labels
```

CPR runs after native-mask preprocessing and cubic resizing. The centerline is
resampled at approximately one processing voxel along the curve, and the parent
plane samples one processing voxel per pixel in both directions. For CPR,
`--imgsz 2048` directly samples a 2048-by-2048 patch centered on the centerline;
there is no oversized full-volume parent to downsample. Samples outside the
processing volume are zero. The patch is allowed to omit distant parts of the
volume. With `--imgsz 0`, its side defaults to the largest processing-volume
dimension, still at one voxel per pixel. Explicit tile output resizing keeps
the usual PTA behavior and can change tile pixel spacing. Intensities use
trilinear sampling and categorical labels use nearest sampling; oblique planes
therefore interpolate between voxel centers while preserving the sampling pitch.

The implementation reuses TTA's current embedded exact-EDT medial-ridge
centerline extraction, separately for connected objects and their returned
branches. It does not require VMTK. Extraction uses a block-maximum raster bounded
to 256 voxels per axis, preserving foreground that simple striding would miss.
Parallel transport keeps the perpendicular plane axes continuous along curves.
Coarse centerline seeds are refined against bounded perpendicular planes from
the original processing mask so block pooling does not shift a thin tube away
from the parent image center. Temporary geometry planes fill enclosed holes for
centering; the original mask and rendered categorical holes are preserved.
One plane is rendered at a time, without allocating a dense straightened volume.

Centerline extraction remains a CPU preparation step after requested 3D void
filling, Gaussian preprocessing and cubic resizing. All centerlines are available
before candidate planning. For GPU offline execution, CPR intensity, categorical
masks and annotation coverage use a resident CUDA sampler when the source volumes
are admitted. CPR reuses the ordinary view renderer's source allocation and the
categorical volume owner; it does not upload another cube for each branch or frame.
Outputs stay on the GPU through policy and publication. Native parents and affine
address maps are bounded caches, retired with their source volume.

The CUDA sampler uses the CPU coordinate and zero-border rules. Categorical
samples preserve nearest-neighbor labels. Native-T intensity follows the existing
GPU renderer's virtual gray8 processing grid: integer processing voxels are
interpolated and quantized before CPR's trilinear sampling. This can introduce
small intensity rounding differences from an OpenCV-materialized CPU cube; masks
use the actual processing-grid raster. Nonidentity categorical affine transforms
use cached OpenCV-derived address maps to preserve backend-specific tie behavior.
Native 2048-pixel parents use the identity path and need no affine-map setup.

`YOLO_TTA_PTA_GPU_CPR_RENDER=0` disables CPR CUDA rendering. Unavailable CUDA or
insufficient resident-source capacity retains CPU reslicing with a diagnostic.
CPR tasks remain distributed among ordinary view tasks, so CPU fallback can
overlap other GPU work. Each selected GPU has its own rendering owner; GPUs do
not pool VRAM. Runtime logs and telemetry report actual CPR CUDA use or fallback.
Frame ownership, candidate identities and within-family ordering are preserved.
Centerline extraction is not overlapped with the same volume's rendering.

When `--enable_2D_hole_fill` is requested, the native orthogonal-plane closure
runs before 3D void filling and CPR extraction. Foreground-anchor preservation
only restores a mapped slice that has become entirely empty; it no longer
expands a square around an already surviving boundary pixel.

`--fraction` applies to CPR through the same deterministic frame/tile candidate
selection as other views, before classification and rendering. It does not reduce
input loading, preprocessing, or centerline extraction. A small fraction can
select no full parent from a short branch. A selected tile still uses its parent
frame's render, shared with other selected items from that frame.

Unlabeled volumes, source sequences with numeric gaps, and partial volumes
without `--force` cannot establish a complete labeled centerline and skip CPR
with a diagnostic. Empty masks produce no CPR views. Components without a usable
tubular centerline are reported and omitted. Very thin structures below the
coarse extraction resolution and broad sheets may not yield usable branches.
Per-view manifest records include the extraction backend, centerline geometry
hash, sample spacing, centering, and sampling policy.
