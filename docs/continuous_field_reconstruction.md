# Continuous-field reconstruction

Use `tools/prepare_pta_input.py --mode continuous_field --mask-source global` to
rebuild a smooth native-resolution PTA label from the edited preview Global
alone. Native predictions are never read, so their high-frequency label noise
and ambiguous component matches cannot return. Compared with
[edited-shape reconstruction](edited_shape_reconstruction.md), the boundary is
smoother, but there is no edited-center, edited-background-cell or topology
guarantee. It is not the continuous-field view and float-polygon publisher from
the 2026-11-06 upscaling-mechanism experiment: it writes a native binary NRRD,
and PTA generates its views from that file as usual.

## Method

The `tools/continuous_field_reconstruction/` package builds two preview-sized
fields over the complete foreground bounding box plus three zero cells:

- `phi`: the signed Euclidean distance to the edited boundary, in preview units
  (positive inside).
- `cap`: a local radius cap, `min(0.5, 0.2 * max(r - 0.5, 0))`. Here `r` is the
  inside distance at the nearest Lee skeleton voxel. The cap is 0 when that
  voxel lies in another six-connected component. Outside the mask, each voxel
  takes the cap of its nearest foreground voxel.

Each native voxel center is mapped to `(j + 0.5) * L / H - 0.5` on the preview
grid. There `L = linear(phi) - linear(cap)`, `C = B3(phi) - B3(cap)` and
`W = B3(cap) / 0.5`, where B3 is SciPy's cubic B-spline (`order=3`,
`prefilter=False`). Foreground is `(1 - W) L + W C >= 0`. Thick regions use the
smoother cubic field, while thin regions stay close to the linear field. The
fixed policy is `guard-cap-weighted-cardinal-B3-v1`, version 1.

Only footprints that mix foreground and background are interpolated. Uniform
footprints and voxels outside the compact field use exact sign shortcuts, so
the result does not depend on `--workers` or `--block-mib`. The writer streams
the native NRRD, then reads it back to the end before publishing. That readback
checks the header, binary values, payload hash and voxel count.
`prepare_pta_input.py` requires this audit in the writer's receipt and checks
the compressed file hash again just before publication.

## Running it

```text
python tools/prepare_pta_input.py SOURCE_STEM --mode continuous_field \
    --mask-source global --bundle-dir BUNDLE_DIR --source-video SOURCE_VIDEO \
    --output-dir OUTPUT_DIR --output-stem OUTPUT_STEM --intensity-mode reference \
    --workers 32 --block-mib 1024 --field-memory-mib 2048 --log-dir LOG_DIR
```

`SOURCE_STEM`, `OUTPUT_STEM` and the paths are as in
[edited-shape reconstruction](edited_shape_reconstruction.md). Keep the data,
labels and logs outside the repository checkout.

- `--log-dir` must name a new directory on every run, for example
  `LOG_DIR/validate` for a `--validate-only` pass and `LOG_DIR/prepare` for the
  preparation.
- `--field-memory-mib` (default 2048) budgets the exact global field build. It
  is separate from the aggregate native buffer budget, `--block-mib`.
- `--field-cache DIR` keeps the field under `DIR/` plus the output stem, so
  later runs can validate and reuse it. A cached field built from other edits
  is refused, not replaced. Without this option, the field is written beside
  the run logs.

Dependencies: NumPy, SciPy, scikit-image (the `cleanup` extra) and ffprobe. Keep
`prepare_pta_input.py`, `verify_pta_staged_pair.py` and the
`continuous_field_reconstruction/` folder together when copying them to a
cluster.

## Receipt and reuse

`preparation.json` records the mode as `continuous_field`. It also includes
`identity.continuous_field`, which holds the policy, version, package source
hashes and Python/NumPy/SciPy/scikit-image versions. Further fields are
`continuous_reconstruction: true`, `label_authority:
edited_preview_sampled_support`, and the writer's readback audit under `mask`.
The source video is always hashed.

An existing label is reused only when the data, geometry and fixed policy still
match. The data include the source video's path, size and modification time, so
a moved, re-copied or touched video also stops reuse. Package hashes and library
versions are recorded as provenance, not as reuse locks. Any other difference
stops preparation, and nothing is overwritten.

`tools/verify_pta_staged_pair.py` checks that a staged video and mask copy
match the receipt by content hash. It also checks the recorded mode and output
stem, and that the mode's reconstruction contract names the receipt's policy:

```text
python tools/verify_pta_staged_pair.py --receipt OUTPUT_DIR/preparation.json \
    --video STAGE_DIR/OUTPUT_STEM.mkv --mask STAGE_DIR/OUTPUT_STEM.nrrd \
    --expected-mode continuous_field --expected-output-stem OUTPUT_STEM
```

## Derived axial crops

A bundle from `tools/xta_cut_axial_pair.py` (`xta-slicer-edits-axial-crop-v1`)
is reconstructed on the original preview and native lattices, with zero beyond
the cut. The removed native columns are then dropped, so the retained voxels
match an uncropped run. `mask.pre_crop_mask` holds the readback audit taken
before the crop.

The cutter keeps the original `source_manifest.json` byte for byte. If a
manifest's shapes were rewritten to the cropped grid, the original sampling
lattice is lost. Such a bundle is refused: cut it again from the original
export.

## Labels from the standalone bundle

The 2026-11-07 standalone bundle wrote the same algorithm under the mode name
`continuous_b3`. Those `preparation.json` files are not reused. Preparing into
the same directory stops with `prepared mode 'continuous_b3'` and leaves it
untouched. Their NRRDs remain valid as they are, and their staged pairs still
verify with `--expected-mode continuous_b3`.

To prepare the label with this mode instead, move the old label directory
aside or choose a new output directory. On the standard F2_30_5_3_2026 bundle,
the new mode reproduced the standalone bundle's NRRD byte for byte. Crop
bundles are the exception. The cluster copy that accepted them sampled on the
cropped grid, so a crop label prepared with this mode differs.
