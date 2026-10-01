# Dynamic crops in LTA

LTA keeps `--lta_crop_backend tiled` as its default. Select
`--lta_crop_backend dynamic` (alias `--crop_backend dynamic`) to track bounded
object-following native crops. Omit `--enable_tile` in dynamic mode. The initial
scope is native Transverse at angle zero with both source frame dimensions at
least 1008 pixels. LTA's SAM model grammar remains untagged.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lta_crop_margin` | `96` | Native-pixel context around the seed bounds |
| `--lta_crop_guard` | `24` | Model-pixel interior-edge distance that requests a bounded patch |
| `--lta_crop_max_scale` | `3` | Maximum native crop side divided by 1008; permitted range 1 to 3 |

A crop stays fixed within a tracker window. At a sealed predecessor boundary,
the next crop is planned from complete native masks and deterministic object
batches. An 84-pixel origin snap is constrained by full seed containment. The
maximum native side is the smaller of the configured scale times 1008, frame
height, and frame width. A seed too large for the bound raises a planning error.
Native crops are explicitly transformed to the tracker's 1008-square input,
and predictions are restored to native coordinates.

Crop windows follow support across the earlier fixed tile grid. An interior
guard or a spatial split event can request one patch from the earliest event
to that window's boundary. Simultaneous split and guard events both apply.
Patches do not recursively create more patches. Split depth is bounded at
three, with at most 16 objects per crop batch and 16 split children. Smaller or
excess islands remain with the largest daughter so partitioning preserves
foreground. Event and guard-budget receipts make exhausted support
explicit.

Independent forward and backward anchor branches retain their own lineage.
Batch membership depends on sealed predecessors, so task arrival order cannot
change the crop composition. The normal authoritative-anchor preservation and
final publication contracts still apply.

This mode changes model context and may change predicted support. It does not
add motion extrapolation, area-collapse recovery, or discovery of unseeded
objects. SAM interpolation in TTA uses its own fixed bounded family context;
it does not require this LTA crop backend. Real-model evaluation and its
limitations are recorded separately in task Scratch.

`tools/lta_dynamic_crop_diagnostic.py` prepares a lossless 1-to-30-frame clip,
one original exemplar polygon, and production commands for native and scaled
dynamic contexts. Run `--help` for its source and output arguments. The tool
does not run inference; the caller must reserve `Scratch/Temp/GPU_LOCK` before
running the generated GPU commands. Its preparation receipt is a lifecycle
and transform fixture, not independent segmentation-quality evaluation.
