# Paired-volume registration

`xta register` maps two reconstructed grayscale slice stacks by image content.
The fixed stack defines the reference grid. The moving stack can be resliced onto
that grid so its intensities line up with the fixed stack's original annotation
planes. The same commands work through `python -m XTA register`.

## Fit and inspect a mapping

Install the project's `registration` extra and make `ffmpeg` and `ffprobe`
available on PATH. The movie commands accept native 8-bit grayscale frames in
presentation order, with one frame per Z slice.

```sh
python -m pip install ".[registration]"
xta register fit --fixed earlier.mkv --fixed-header earlier_Header.txt --moving later.mkv --moving-header later_Header.txt --pair sample01 --output sample01_mapping.zip --work-dir scratch
xta register inspect sample01_mapping.zip
```

Reconstruction headers use the `[Image Info]` fields `Images Taken`, `Image
Height`, `Image Width` and `Pixel Size`. The scalar size is applied in all
three axes. Alternatively, give `--fixed-spacing X Y Z` and `--moving-spacing X Y Z`
in the same length unit. Without either scan's spacing, the fit assumes unit
voxels on both grids. Playback FPS is recorded separately.

For a scan with a black acquisition exterior and gray interior, add
`--black-exterior`. XTA measures a native XY observation mask for each scan,
including interior dark structures. The default treats the rectangular images
as observed. To exclude early artifacts from fitting, use `--exclude-first 400`
or give separate counts for the fixed and moving acquisitions. The registered
movie still includes every reference frame.

The estimator combines candidate 3D orientations, intensity-based refinement
and local patch correspondences. It compares a rigid motion in physical space
with a fit that adds uniform scale. Patch errors are reported in moving native
voxels. On downsampled fitting grids, the default also checks native-resolution
patches. The packet records the level of validation actually obtained.
`--broad-search` adds a CPU orientation and translation search for acquisitions
that may have settled very differently. `--step` overrides the power-of-two
pyramid step. `--pyramid-only` skips the additional native refinement.

The working directory needs room for temporary decoded arrays. For a grayscale
stack, native byte size is width × height × frame count. Allow for both stacks
when fitting with native refinement. Temporary files are removed on exit.
Insufficient registration evidence produces an `unavailable` packet and exit
status 3. Its validation record can be inspected. Transfer commands require a
validated mapping.

## Make a full registered movie

```sh
xta register video --packet sample01_mapping.zip --input later.mkv --output later_on_earlier.mkv --work-dir scratch
```

This writes a lossless grayscale FFV1 MKV with the fixed dimensions and frame
count, including slices with no annotations. For each fixed voxel center, XTA
samples the moving volume using trilinear interpolation. The input identity is
checked against the packet. A `.registration.json` sidecar records the output
identity, sampling matrix, target geometry and decoded frame checks. For the
opposite output grid, supply the original fixed movie and
`--direction fixed-to-moving`.

To inspect every reference slice in a lighter playback file, compare the fixed
movie with the moving-on-fixed export:

```sh
xta register review --packet sample01_mapping.zip --fixed earlier.mkv --registered-video later_on_earlier.mkv --output sample01_review.mp4
```

The MP4 shows the reference, registered intensity and red/cyan overlay at
reduced XY display resolution. It contains one frame per reference Z slice.

## Hand registered image/label pairs to PTA

```sh
xta register pairs --packet sample01_mapping.zip --registered-video later_on_earlier.mkv --labels earlier_labels --output pta_pairs/sample01
```

This workflow keeps the labels on their original slice planes. The output
directory contains same-stem lossless PNG and YOLO TXT files side by side.
Pass this directory directly to PTA's `--input` option. PTA handles image
views, augmentation, dataset construction and splits. An explicitly
empty input TXT denotes an empty annotation on the observed area. Missing
source slices remain unknown. Polygon areas falling beyond either scan's
observed field are clipped. In the paired PNGs, pixels outside the shared
observed field are set to zero so that tissue visible only in the later scan
does not enter as unlabeled background. Full registered movies retain that
content. Pair provenance records clipping and omitted planes. Review
propagated contours where local registration is uncertain.

The default source filename convention is one-based (`_0001.txt` is frame
zero). Pass `--index-base 0` for a zero-based source. Output suffixes are
one-based. For labels on the moving scan and a fixed movie resliced onto its
grid, use `--direction moving-to-fixed` on the `pairs` command.

To transfer the labels onto the **original moving slice grid** instead:

```sh
xta register yolo --packet sample01_mapping.zip --labels earlier_labels --output later_partial_labels
```

Each supplied annotation is rasterized as a one-voxel-thick native slab.
Nearest-neighbor 3D reslicing can distribute one slab across several target
planes. The result includes partial YOLO polygons, exact cropped class masks
and annotation coverage. Holes and thin components can be lost in the polygon
representation. The transfer manifest quantifies that loss. Use the
resliced-movie workflow above for ordinary whole-slice supervision.

## Transfer a scalar NRRD labelmap

```sh
xta register nrrd --packet sample01_mapping.zip --input earlier_labels.nrrd --source-reference earlier_scan.nrrd --target-reference later_scan.nrrd --output later_labels.nrrd
```

The reference headers connect native movie indices to world coordinates.
Source labelmaps can be cropped or reoriented when this geometry is available.
For a full-sized label array already in the source movie's exact XYZ index
order, use `--source-grid-aligned` in place of `--source-reference`. Without
a target reference, output coordinates are scan-local, with packet spacing and
origin at target index zero. XTA uses nearest-neighbor sampling for integer
classes and writes a companion coverage NRRD and provenance JSON. Supported
inputs are embedded, scalar, three-dimensional integer NRRDs with raw or gzip
encoding. Slicer segmentation files that carry segment-specific metadata
require a scalar labelmap export.

## Packet conventions

The ZIP contains `mapping.json`, a short README, and any recorded acquisition
masks. It can be used without XTA. Each scan record gives its filename,
SHA-256, native shape in ZYX order, and voxel spacing in XYZ order. An
observed-field array is boolean YX at native resolution and repeats along Z.

Multiply a **column vector** `[x, y, z, 1]` by `fixed_to_moving_xyz` to locate
that fixed-grid point in the moving acquisition. Here X is column, Y is row,
and Z is the zero-based frame index. This same matrix supplies the source
coordinates when pulling moving intensities onto the fixed grid. Use
`moving_to_fixed_xyz` for moving-grid points. The physical matrix uses each
scan's local voxel axes and origin. Scanner-world orientation and origin come
from reference image geometry when provided.
