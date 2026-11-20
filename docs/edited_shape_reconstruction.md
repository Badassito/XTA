# Authoritative edited-shape reconstruction

Use `tools/prepare_pta_input.py --mode edited_shape --mask-source global` when the
corrected edited Global defines the desired shape. This reconstructs a smooth
mask at the original input dimensions. Original full-resolution predictions
are not consulted or unioned back into the result, so deleted native components
cannot return through an ambiguous component match. For a smoother label
without the edited-center and background-cell guarantees below, see
[continuous-field reconstruction](continuous_field_reconstruction.md).

The implementation computes a signed Euclidean distance field on the edited
preview, with axis distances measured in native voxel units. It linearly samples
that field at native voxel centers using `(j + 0.5) * L / H - 0.5`. Foreground
requires both a positive field and membership in an edited foreground cell.
This second condition limits smoothing to the inside of the edited shape:
no foreground may enter an edited background cell. Each edited foreground cell
also keeps the native voxel nearest its center, the voxel the audit samples.
Linear interpolation alone can erase it at convex corners and isolated cells on
low-ratio grids, and at every outward corner at exactly 2x.
A one-cell background border handles volume boundaries. No binary dilation,
native-component watershed, or largest-object selection is applied.

The written NRRD is read back before publication. Validation checks its binary
payload, count and hashes, every edited voxel center, exclusion of edited
background cells, and equality of the edited/native six-connected foreground
component counts. A failed check prevents publication. The receipt key
`reconstruction.retained_center_voxels` counts voxels kept only by the
center rule above (0 means interpolation reproduced every center). The remaining
numerical refusals come from thin connections whose component count changes;
payload integrity checks still apply. The background-cell
constraint prevents new foreground paths across edited background; together
with retained centers and equal component counts, it preserves correspondence
between foreground components without merging, splitting or dropping them.
It does not establish every possible subvoxel tunnel/cavity invariant, nor
recover biological detail that is missing or incorrect in the edited preview.
Existing notches and holes remain part of the authoritative input shape.

The distance field and EDT workspaces are preview-sized. Native output is
streamed through bounded row buffers, and connected-component validation uses
current/previous planes rather than a native-volume integer label array.
`--workers` controls bounded parallel interpolation. `--block-mib` is an aggregate
native interpolation-buffer budget across those workers; large allocations need
a larger explicit budget to avoid tiny tasks. For example, the F2 job with 128
workers and 4096 MiB schedules 5,790 row blocks, compared with about 349,000 at
64 MiB. Empty edited-background blocks skip interpolation, and trailing background
is cropped without changing the coordinate origin or numerical policy.

Output is a binary gzip NRRD with the existing unit-LPS TYX/XYZ convention. ISA-L
compression/decompression is used when installed, with a standard gzip fallback.
Compressed bytes can differ by codec while the binary payload and audits remain
identical. The writer still rereads and audits the complete output. The preparation
wrapper validates that audit and the compressed file identity instead of decoding
the just-written output a second time; reuse of an existing input still performs
a full payload validation.

`--output-dir` selects a separate destination for one volume. It allows an edited
reconstruction to be inspected without replacing an existing native-recovery
input. `--bundle-dir` selects an explicit filtered bundle for validation or
preparation; otherwise the legacy generation/stem layout remains available.
Input and output identities, helper hashes, written-mask audits, and the explicit
`edited_preview_global` authority are recorded in `preparation.json`.
`--intensity-mode copy` retains the verified independent video-copy workflow.

For separate training directories, such as `images/DATASET/VIDEO.mkv`,
`filtered/GENERATION/DATASET/` and `labels/GENERATION/DATASET/`, pass each path
explicitly:

```text
python tools/prepare_pta_input.py SOURCE_STEM --mode edited_shape \
    --mask-source global --bundle-dir BUNDLE_DIR --source-video SOURCE_VIDEO \
    --output-dir OUTPUT_DIR --output-stem OUTPUT_STEM --intensity-mode reference \
    --log-dir LOG_DIR
```

`SOURCE_STEM` is the source identity stem the bundle was exported under (for
example `DATASET_S1S0_8bit_Y`), and `--output-stem` names the label.
`--intensity-mode reference` leaves the video where it is and publishes only the
NRRD and `preparation.json` in `OUTPUT_DIR`; the receipt records and validates
the original video reference. Add `--validate-only` first to check the bundle
and video dimensions without writing a label. `--log-dir` must name a new
directory on every run, for example `LOG_DIR/validate` for the validation and
`LOG_DIR/prepare` for the preparation. With `--smoke --smoke-script
/path/to/pta_launcher.py`, the video and prepared NRRD are copied with matching
basenames into a new `pta-smoke-*` directory under `--scratch-root` (default
`$SLURM_TMPDIR`, else the system temporary directory), and PTA uses that
directory as input. Each run gets its own directory; it is removed after a
passing smoke and kept for diagnosis when the smoke fails.

`--root` and `--training-root` have no defaults and are needed only for the
legacy layout. Without `--source-video` the video is `ROOT/STEM.mkv`; without
`--bundle-dir` or `--output-dir` the bundle and output are
`TRAINING_ROOT/GENERATION/STEM_Filtered` and `STEM_Input`.

Keep `prepare_pta_input.py` and `xta_edited_shape.py` together when copying them
to a cluster, plus the `continuous_field_reconstruction/` folder for
`--mode continuous_field`. Cluster job scripts are site-specific and stay
outside this repository; they call the command above with their own paths.
NumPy, SciPy, and ffprobe are required; ISA-L is optional.

This reconstruction is separate from PTA preprocessing. Passing
`--enable_2D_hole_fill` to PTA deliberately closes enclosed native-plane holes
before 3D void filling and centerline extraction. That requested operation may
change holes in the reconstructed input; it does not repair an open notch or
infer a different biological ground truth.
