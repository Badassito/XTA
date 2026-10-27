# PTA curved planar reformation

`--enable_cpr` adds straightened cross-section views from a fully labeled PTA
volume. Each output frame is perpendicular to the local centerline tangent: a
tube produces a circular foreground cross-section. The centerline passes through
the exact central pixel of each odd-sized square parent frame. Ordinary PTA tiles
crop that parent and do not recenter it.

```text
python -m XTA --mode pta --input DATASET --enable_cpr --save images labels
```

CPR runs after native-mask preprocessing and cubic resizing. The centerline is
resampled at approximately one processing voxel along the curve, and the parent
plane samples one processing voxel per pixel in both directions. Explicit
`--imgsz` scaling and tile output sizing retain their usual PTA behavior and can
change the emitted pixel spacing. The parent extends symmetrically around the
centerline to cover the source volume; samples beyond the input are zero.
Intensities use trilinear sampling and categorical labels use nearest sampling.

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

Unlabeled volumes, source sequences with numeric gaps, and partial volumes
without `--force` cannot establish a complete labeled centerline and skip CPR
with a diagnostic. Empty masks produce no CPR views. Components without a usable
tubular centerline are reported and omitted. Very thin structures below the
coarse extraction resolution and broad sheets may not yield usable branches.
Per-view manifest records include the extraction backend, centerline geometry
hash, sample spacing, centering, and sampling policy.
