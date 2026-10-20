# PTA candidate sampling with --fraction

`--fraction` selects original full-frame and tile candidates **before foreground
classification, output rendering, offline augmentation, or publication**. It is
a PTA option. It does not reduce TTA prediction coverage.

```bash
python -m XTA --mode pta --input /data/clean --output /data/training \
  --imgsz 2048 --enable_cartesian transverse,sagittal,coronal \
  --enable_tile 1024:512 --fraction 0.1 --background_percent 0.2 \
  --save images labels summary
```

The default is `1.0`, preserving all eligible candidates before existing filters.
Values must satisfy `0 < fraction <= 1`. A fraction of `0.1` means approximately
10%, rather than an exact quota. A small population may yield no selections.

Selection uses a fixed hash of each original's volume, physical view, channel
variant, frame index and full-frame/tile identity. Repeating the same inputs and
geometry gives the same selection; worker scheduling and candidate iteration
order do not change it. Each tile has an independent decision: a rejected full
frame can have accepted tiles, and vice versa. Increasing the fraction preserves
the smaller selected pool before subsequent background/split/augmentation rules.

The metadata iterator discards rejected candidates immediately rather than
materializing the entire candidate population. It still visits candidate
identities, but does not create millions of rejected image files or first retain
millions of rejected candidate records.

The order is:

1. Enumerate eligible original frame/tile metadata and apply the fraction.
2. Classify only selected candidates when background or augmentation planning
   needs foreground/background information.
3. Assign existing train/validation splits and apply background filtering.
4. Plan offline copies from retained originals and render/publish the result.

The fraction therefore affects both train and validation output pools. Offline
augmentation copies are descendants of selected originals, not separate sampling
draws. Deferred augmentation generates copies in the training loader and does not
multiply the files written by PTA.

`--background_percent` works within the selected pool. No discarded candidates
are rendered to refill a quota. Background retention and rare foreground-to-
background augmentation changes keep their existing handling; there is no claim
that the final file count is exactly `fraction * original_count`.

The source volume still needs decoding/loading and requested preprocessing to
support arbitrary 3D views and channel context. A selected legacy tile can require
rendering a shared source canvas even when its full-frame output was rejected.
Canonical direct tile paths avoid generating the unused full output. Requested
full-volume diagnostics such as NRRD snapshots and overlay videos remain complete
and are outside dataset sampling; omit them when minimizing work/disk usage.

The summary and manifest record the requested fraction, the candidate population
before sampling and the selected count before background filtering. Full
foreground-slice retention is intentionally not guaranteed when sampling is on;
the underlying preprocessing/anchor behavior remains unchanged.

## Example workload

For the F2 source `1930 x 3064 x 3023`, PTA uses processing shape
`2911 x 3064 x 3023`. All view families, all signed 30-degree tilt variants in
both directions, one 0-degree in-plane angle, gray channels and no tiles yield:

| Family | 2048 raster | 3072 raster |
| --- | ---: | ---: |
| Cartesian | 8,998 | 8,998 |
| Tilted Cartesian | 35,992 | 35,992 |
| Upright azimuthal | 10,127 | 8,077 |
| Tilted azimuthal | 55,580 | 55,580 |
| Radial | 69,020 | 23,820 |
| Spherical | 27,150 | 20,520 |
| Total originals | 206,867 | 152,987 |

These are planner counts before fraction/background filtering. Radial/Spherical
require an explicit positive raster size. Larger patches need fewer radial and
spherical outputs, but each image contains more pixels.

At 2048, offline ratio 5 plans 1,034,335 versions if every original is eligible
for copies and retained. Fraction 0.1 reduces that to approximately 103,434 before
background filtering or augmentation drops. Tiles enlarge the candidate pool
according to their size/stride, and each tile is sampled independently.
