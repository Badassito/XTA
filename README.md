# XTA

XTA processes large image volumes in three modes:

- **TTA** combines model predictions from transformed views of a volume.
- **PTA** generates pretraining images and labels from labeled volumes.
- **LTA** propagates authoritative annotations with a local SAM model.

Use the mode-specific command help for the current interface:

```text
python -m XTA --mode tta --help
python -m XTA --mode pta --help
python -m XTA --mode lta --help
```

The versioned launcher and the installed `xta` command use the same dispatcher.
Runtime dependencies and optional backends are declared in [pyproject.toml](pyproject.toml).

## Paired-volume registration

Run `python -m XTA register --help` to fit a mapping, reslice every frame of a
grayscale stack, or transfer annotations. Mapping packets carry the source
identities, both coordinate directions and acquisition coverage. The
[registration guide](docs/registration.md) shows how to produce image/label
pairs for PTA and transfer scalar NRRD labelmaps.

## TTA confidence sweeps

Use `--conf 0.50:0.10:1.00` for six operating points or
`--conf 0.50:0.01:1.00` for 51. On eligible full-frame runs, the worker
renders each batch once and evaluates the cutoffs from one YOLO forward. With
interpolation off, compatible view masks are backprojected together and their
NRRDs can appear across cutoff directories as each view finishes. With SDF or
SAM, each cutoff runs its own interpolation from its own masks.
With `--save nrrd low_quality:0.20`, the sweep also publishes a Slicer review:
complete editable segments for smoothed 3D surfaces and one coverage-colored
overlay over intensity in the slice views. One marker action applies to the
usual visible masks across cutoffs. See the [confidence sweep guide](docs/confidence_sweeps.md).

TTA layer `.seg.nrrd` files store only each layer's segment bounding box, placed
in the full output grid with Slicer's `Segmentation_ReferenceImageExtentOffset`
and `Segmentation_ReferenceImageExtent`. Slicer loads and edits them exactly as
full-grid files, and the XTA readers (`reconciliation_io.read_layer_manifest`,
`tools/xta_native_filter.py`, `tools/nrrd_reference_grid.py`, the cut tools and
PTA mask import) reconstruct the full grid. Manifest rows record
`stored_offset_tyx`, `stored_shape_tyx` and `stored_layout`; an empty layer is
one zero voxel.

## PTA label geometry

`--force` allows NRRD labels to differ from the image by up to three voxels
per resized axis, inclusive. PTA uses nearest-neighbor sampling onto the image
grid, assuming the same field of view along each resized axis.

- Contiguous 3D volumes resize the label in XYZ when each dimension differs by
  at most three voxels.
- 2D inputs, including source sequences with numeric gaps, resize matching label
  planes in XY when each dimension differs by at most three pixels. Each label
  retains its original slice correspondence. With one NRRD plane per image,
  plane order follows the sorted image indices. A gapped sequence with a different
  number of label planes requires an explicit mapping.
- A depth difference greater than three requires a known slice mapping for 2D
  processing. PTA uses matched slices and leaves unmatched images unannotated.
  A dense MKV/NRRD pair with an unknown starting-slice offset fails validation.

Use `--force --nrrd_z_start_index INDEX` to supply that mapping. `INDEX` is the
encoded image index, or zero-based video frame, corresponding to NRRD plane 0.
Each subsequent label plane advances one image slice, with the same spacing and
direction. Signed indices allow a label volume to begin before image frame 0.
This assertion applies to every paired NRRD in the run and selects transverse
2D processing, including when the dimensions match. For example, if NRRD plane 0
corresponds to video frame 5, use
`python -m XTA --mode pta --input INPUT_DIR --force --nrrd_z_start_index 5 --enable_cartesian transverse --save images labels`.

If explicit slice matching leaves image planes unannotated, PTA skips the dense
binary `--save nrrd` snapshot because it would encode those planes as background.

Normalized YOLO polygons are rasterized at the image's XY dimensions. Explicit
slice indices determine their image correspondence. The run diagnostics record
any forced label resizing.

## Guides

- [v25.0.0 options and migration](docs/topology_and_cpr.md)
- [SAM interpolation and extrapolation](docs/sam_interpolation.md)
- [Dynamic LTA crops](docs/lta_dynamic_crops.md)
- [PTA centerline cross-sections](docs/pta_cpr.md)
- [Slicer cleanup workflow](docs/slicer_pta_cleanup.md)
- [Projection coverage contracts](docs/projection_coverage.md)
- [Moving completed runs](docs/run_transport.md)

## Development

Start with the [maintenance workflow](docs/maintenance.md) for source inventory,
packaging, validation, generated artifacts, and release checks. The
[tools index](tools/README.md) distinguishes user utilities from source-checkout
qualification tools. [ARCHITECTURE.md](ARCHITECTURE.md) describes runtime ownership
and numerical contracts; [AGENTS.md](AGENTS.md) governs repository-specific work.
