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
input. The filtered source still comes from the requested training generation.
Input and output identities, helper hashes, written-mask audits, and the explicit
`edited_preview_global` authority are recorded in `preparation.json`.
`--intensity-mode copy` retains the verified independent video-copy workflow.

This reconstruction is separate from PTA preprocessing. Passing
`--enable_2D_hole_fill` to PTA deliberately closes enclosed native-plane holes
before 3D void filling and centerline extraction. That requested operation may
change holes in the reconstructed input; it does not repair an open notch or
infer a different biological ground truth.
