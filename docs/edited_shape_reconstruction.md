# Authoritative edited-shape reconstruction

Use `tools/prepare_pta_input.py --mode edited_shape --mask-source global` when the
corrected edited Global defines the desired shape. This reconstructs a smooth
mask at the original input dimensions. Original full-resolution predictions
are not consulted or unioned back into the result, so deleted native components
cannot return through an ambiguous component match.

The implementation computes a signed Euclidean distance field on the edited
preview, with axis distances measured in native voxel units. It linearly samples
that field at native voxel centers using `(j + 0.5) * L / H - 0.5`. Foreground
requires both a positive field and membership in an edited foreground cell.
This second condition limits smoothing to the inside of the edited shape:
no foreground may enter an edited background cell. A one-cell background border
handles volume boundaries. No binary dilation, native-component watershed, or
largest-object selection is applied.

The written NRRD is read back before publication. Validation checks its binary
payload, count and hashes, every edited voxel center, exclusion of edited
background cells, and equality of the edited/native six-connected foreground
component counts. A failed check prevents publication. The background-cell
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

For the separate training directories, `tools/prepper.sh` uses
`images/<dataset>/<video>.mkv`, `filtered/<generation>/<dataset>/`, and
`labels/<generation>/<dataset>/`. It defaults to generation `v1.1`, dataset
`F3_20_2_3_2026`, and video `F3_20_2_3_2026_S1S0.mkv`. The source identity stem
remains `F3_20_2_3_2026_S1S0_8bit_Y`, while `--output-stem F3_20_2_3_2026`
names the label. `--intensity-mode reference` leaves the video in `images/`
and publishes only the NRRD and `preparation.json` under `labels/`; the receipt
records and validates the original video reference. With `--smoke --smoke-script
/path/to/pta_launcher.py`, the video and prepared NRRD are copied into
`/tmp/pta-smoke` with matching basenames, and PTA uses that directory as input.
The copies remain there between runs; no job ID is added to the directory name.

Keep `prepper.sh`, `prepare_pta_input.py`, and `xta_edited_shape.py` together when
copying them to the cluster. Submit `sbatch prepper.sh --validate-only` to check
the bundle and video dimensions, then `sbatch prepper.sh` to prepare the label.
The cluster root defaults to `/project/community/sanghyus`; `ROOT`,
`TRAINING_ROOT`, and the path variables documented in the script can override it.
NumPy, SciPy, and ffprobe are required; ISA-L is optional.

This reconstruction is separate from PTA preprocessing. Passing
`--enable_2D_hole_fill` to PTA deliberately closes enclosed native-plane holes
before 3D void filling and centerline extraction. That requested operation may
change holes in the reconstructed input; it does not repair an open notch or
infer a different biological ground truth.
