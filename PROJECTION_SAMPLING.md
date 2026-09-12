# Certified projection sampling

TTA defaults to `--projection_sampling coverage`. This reduces native model
frames for Spherical, Radial and automatic Azimuthal views while preserving
their declared native intensity-coverage domains. Use
`--projection_sampling dense` to retain the previous schedules. Shared geometry
APIs, PTA and LTA continue to default to dense sampling unless an API caller
explicitly requests coverage mode.

An unrotated TTA base pass must be selected. If `--angle` contains no 0-degree
pass (modulo 360), TTA retains dense schedules and explains the fallback.
Explicit `--enable_azimuthal view:degrees` spacing is never changed.

## What coverage means

Every voxel center in the declared annulus has a positive interpolation-basis
weight in at least one native intensity sample. The minimum/maximum radii and
source domains remain unchanged; excluded central cores remain excluded.
This does not guarantee a foreground prediction, nonzero output after gray8
rounding, nearest-neighbor categorical sampling, or coverage after arbitrary
cropping/downsampling. The unrotated base pass supplies the native coverage;
external augmented copies may crop it and keep their separate support sidecars.

Fewer radius frames change the physical distance between neighboring context
channels and interpolation steps. `C5S1` still selects five neighboring radius
frames, but they can be farther apart. Predictions need not equal dense-mode
predictions. This change retains geometric coverage, not model recall or
identical output masks. Use dense mode for model-quality comparisons.

Each remaining trajectory/angle/policy pass keeps its own NRRD. The planner does
not combine predictions early to obtain the reduction.

## Spherical

The [QSC mapping used by PROJ](https://proj.org/en/stable/operations/projections/qsc.html)
maps each cube face to an equal-area region of the sphere. XTA's inverse-map
formula is unchanged. A new bound proves its Euclidean Lipschitz constant is at
most **1**, replacing the previous conservative 5/3 bound.

`tools/certify_qsc_lipschitz.py` reproduces the certificate using exact rational
arithmetic. It encloses complete parameter intervals, not just sampled points.
The canonical Jacobian has determinant pi/6. Convexity reduces its trace bound
to two radius endpoints; 64 closed intervals in the remaining parameter prove
trace < 5/4. Together with determinant squared > 1/4 and < 1, this bounds both
squared singular values below 1. Sector continuity covers boundaries and the
center. Numerical Jacobian tests separately check that the derivation agrees
with the implemented inverse map.

For outer radius R, endpoint-inclusive shell gap g, and n intervals per QSC
face axis, the native sample-distance certificate is

```text
distance_squared <= g*g/4 + 2*(R/n)^2 <= 281/324 < 1.
```

The planner preserves the previous **281/324** distance budget. It jointly
chooses the fixed face lattice and a uniform endpoint-inclusive radius grid.
For k patches per face axis, it uses the largest even n whose n+1 nodes fit
within k model patches. This spends otherwise redundant patch area on angular
sampling and permits fewer shells. A finite lower-bound search minimizes
frame count within these uniform-shell/fixed-lattice choices. It does not claim
a global optimum over every possible atlas or adaptive radius schedule.

All radii retain the same face grid and patch origins; channels and interpolation
stay within a trajectory. CPU/CUDA projection uses the explicit radius grid.
Projection admission recomputes the certificate from the realized maximum gap
and rejects invalid or tampered coverage metadata.

## Radial

Arc-length and height pixels retain unit spacing. Let d be half the maximum
shell gap and r_min the unchanged minimum radius. The conservative planar bound
is

```text
planar_error_squared <= d*d + 1/4 + d/(4*r_min) <= 0.99.
```

This gives strict positive separable input support. It also covers supported
sampled-shear tilts up to 45 degrees: stack error is at most
`max(1/2, abs(tan(tilt))*planar_error)`, hence remains below one.

Arc/height origins, periodic occurrences and the modeled annulus remain fixed.
An explicit global radius count preserves nearest-shell ownership when a patch
starts partway through the radius grid. The projector validates the actual
maximum gap, certificate, count and stored bound. The old grid is retained
when the certificate offers no reduction; unsupported manual tilt metadata
also falls back to dense sampling.

## Azimuthal

Coverage reduction applies only to automatic upright views with full native
diameter/height rasters, the default hardware-linear source sampler, and source
axes within the qualified 1..4096 range. Compact native rasters, tilted views,
larger axes and optional nearest-XY pointer samplers retain their schedules
with an explicit reason. Explicit user angle spacing remains authoritative.

The pull ROI reaches radius D/2; sampled diameter endpoints reach (D-1)/2.
Nearest diameter-sample parallel error is at most 1/2, including that outer
half-voxel rim. A maximum angular gap delta gives perpendicular error at most
`(D/2)*delta/2`. The planner selects delta so their squared sum is at most
0.99, retaining margin for the native interpolation basis. Model-canvas padding
does not discard the endpoint samples: the renderer retains a positive
zero-padded interpolation fringe and clamps its sampling coordinate to the
endpoint. This also covers half-pixel padding and upscaling phases.

Coarsened Azimuthal views use the existing native pull projector. They cannot
enter either ordinary or hybrid D1 nearest-scatter routing: an input-support
proof is not a nearest-scatter coverage proof. Other eligible views can still
use D1. This routing can change the balance between inference and publication
cost, so frame reduction alone is not a whole-pipeline speed claim.

## Planning and qualification

For a 3072³ working volume, `--imgsz 3072`, default minimum radii, one upright
Spherical cube and one Transverse Radial/Azimuthal axis:

| Family | Dense native frames | Coverage native frames | Reduction |
|---|---:|---:|---:|
| Spherical | 31,032 | 6,402 | 79.37% |
| Radial | 2,969 | 1,726 | 41.87% |
| Azimuthal | 4,826 | 2,805 | 41.88% |

Counts precede in-plane angles, external policy copies, tile passes and final
fixed-batch padding. Spherical trajectories reduce from 24 to 6; its radius
count reduces from 1,293 to 1,067. Radial retains four trajectories and reduces
its global radius count to 752.

Inspect another shape without loading a model:

```bash
python tools/plan_projection_sampling.py --shape 3072 3072 3072 --imgsz 3072 \
  --output /scratch/sampling/plan.json
python tools/certify_qsc_lipschitz.py --output /scratch/sampling/proof.json
```

The pipeline logs dense/selected frame counts and fallback reasons. Run/raster
manifests include conditional `projection_sampling` certificate records;
Spherical reference counts explicitly describe a whole cube group, while
Radial/Azimuthal counts describe their trajectory. Legacy dense plan identities
remain unchanged. Old dense replay captures receive additive defaults; partial
coverage descriptors are rejected instead of guessing their sampling contract.

Run the small synthetic end-to-end qualification with a compatible local model:

```bash
python tools/qualify_tta_augmentation.py --model /models/best.pt \
  --output /scratch/sampling/run --channel_format gray --imgsz 64 --batch 4 \
  --augmentation_ratio 2 --no-tiles --projection_sampling coverage \
  --enable_spherical transverse --enable_radial transverse \
  --enable_azimuthal transverse:auto
```

It checks planned pass groups, independent NRRDs and exact component-union
recomposition. Use a fresh directory; the synthetic fixture tests plumbing,
not segmentation quality. Geometry tests additionally enumerate positive
source taps, compare terminal all-one domains, exercise odd/even/noncubic and
tilted shapes, and compare native/rescaled CPU/CUDA projection results.
