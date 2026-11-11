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
