# Projection coverage

Projection prevents source-grid holes caused by treating reduced prediction
canvases as isolated points. A positive categorical sample represents support
under its declared view transform and sampling rule. Projection must evaluate
that support on the destination grid, including the valid space between sample
locations. It must also retain genuine background and invalid geometry.

Geometric coverage is not blanket dilation. It does not expand a mask by an
arbitrary radius, fill an entire bounding box, paint outside the view's field of
view, or repair a missing model prediction. A detector's lost feature cannot be
recovered simply by changing projection.

## Routes

The routes and device admission boundaries are below. Actual numerical checks,
device execution and complete source qualification belong in separate receipts.

| View family | Projection route | Device admission |
| --- | --- | --- |
| Cartesian: Transverse, Sagittal, Coronal | Canonical categorical restoration; matching bounded D1 native-coverage pull for eligible layouts | CUDA requires its own eligible layout and device checks |
| Tilted Cartesian | Shared view union followed by canonical destination pull | CPU; projection CUDA disabled |
| Upright Azimuthal, dense | Shared view union and canonical destination pull | Eligible Transverse CUDA; Sagittal/Coronal CPU |
| Upright Azimuthal, coverage | Declared reconstruction-plan angular/column pull | Eligible Transverse CUDA; Sagittal/Coronal CPU |
| Tilted Azimuthal | Shared view union and bounded canonical CPU pull; unsafe CUDA plan refused | CPU; projection CUDA unsupported |
| Radial and tilted Radial | Bounded native shell pull | Eligible parent CUDA or separate native owner |
| Spherical and rotated Spherical | Bounded destination-native QSC pull | CUDA requires its own eligible layout and device checks |

Immediate D1 point scatter is no longer used for Tilted Cartesian, dense
Azimuthal, or Tilted Azimuthal, including hybrid detector runs. Their masks pass
through a bounded shared view union and the canonical projector. Cartesian D1
native coverage and the separate Radial native owner remain available.
The legacy D1 worker also rejects stale non-Cartesian requests before allocating
CUDA state, so a later routing mistake cannot silently reactivate point scatter.

This route change can add host shared-union and storage overhead to workloads
that previously used non-Cartesian D1. It does not disable all GPU projection:
eligible Cartesian, upright Azimuthal, Radial, and Spherical projectors
retain their separate device paths. Upright Azimuthal CUDA is Transverse-only;
Sagittal and Coronal use the orientation-aware CPU route. SAM and detector model
inference on GPU is unchanged. Full-cluster wall-time improvement remains
unmeasured; local CPU projection trials have a separate scope.
Ordinary Tilted projection uses canonical CPU processing, including engine
views whose legacy D1 fast projection previously used GPU. The old Tilted
Azimuthal CUDA plan encodes the unsafe scatter geometry, so dispatch refuses
it before allocation/publication and automatically uses bounded CPU pull.
Both ordinary Tilted and Tilted Azimuthal engine projection can be slower than
the former scatter route. CUDA projection remains unavailable for those routes;
a qualified inverse GPU plan is a separate optimization. Local CPU timings do
not establish full-cluster wall time.

## Bounded CPU execution

The CPU Tilted and Azimuthal routes use prepared geometry and compiled
destination-owned OR/MAX loops in `projection_coverage_cpu.py`. Numba releases
the GIL without fastmath or a disk compilation cache. Angular ownership retains
the exact prepared NumPy address rules. Initialization and angle records are
checked against their conservative build allowance before table allocation.
Plans retain local read-only numeric tables, with packed uint32 or uint64 owners
as capacity requires; there is no global Python plan LRU. The kernel does not
copy its source mask or create a global thread pool.

Backprojection honors the caller's per-view worker count with a bounded thread
pool. Each worker owns a distinct destination plane, and callbacks receive
planes in source order. The admitted count is capped by the requested count,
destination depth, and transient workspace. Cancellation/error paths join all
readers before releasing their source. Sink-only publication retains bounded
pending planes rather than another full-volume accumulator.

| Environment control | Default | Scope |
| --- | --- | --- |
| `YOLO_TTA_NATIVE_PULL_BACKEND` | `compiled` | `compiled` or the exact bounded `numpy` reference route |
| `YOLO_TTA_NATIVE_PULL_PLAN_MIB` | `64` | Per-plan geometry/build allowance; packed Azimuthal lookup is read-only and reused across planes; larger geometry uses bounded strips |
| `YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB` | `256` | Per-projection pending-plane and per-worker strip allowance, including a consumer-held plane for sink-only publication |
| `YOLO_TTA_NATIVE_PULL_CHUNK` | `65536` | NumPy reference chunk points, clamped to 1 through 262,144 |

The plan and workspace caps are separate. Source/dense-destination credits,
other concurrent views, and runtime/JIT compiler memory remain separate from
these explicit geometry and buffer bounds. A workspace too small for one worker
and the consumer plane is refused. Production uint8 masks/scores use the
compiled route; other historical array dtypes retain the exact NumPy route
without a whole-source dtype copy.

Confidence readers retain one scalar-max plan per reader. Their admitted
workspace includes output/control bytes and plan construction/runtime strips;
small budgets use the exact bounded sampler when a compiled plan cannot fit.
An explicit budget first reserves a returned uint8 plane and 256 KiB control;
the remaining credit bounds the plan/build/strip peak. The implicit reader allows
64 MiB for that plan plus output/control. It creates no reader thread pool and
caches neither source nor returned output. Only declared plan-admission or
unsupported refusals select the reference sampler; unexpected allocation and
kernel failures propagate. Zero-score unknown semantics and binary/scalar
address correspondence are unchanged.

Confidence capture occurs before native layer materialization and is distinct
from the source-aligned score reader above. Parent preparation passes its existing
allocated slice-worker count through a typed task-local capture scope. The fused
capture keeps the original masking, maximum and payload/index semantics, with
effective workers no greater than the caller allocation. Its exact plan workspace
is added to the existing parent transient request, capped at 256 MiB per parent;
the dense and transient admission limits are unchanged. Planning occurs before
the owner is popped, and execution stays inside the live reservation with failure
cleanup. Disabled confidence, absent score maps, and already-published D1 parents
skip the plan and charge.

The direct capture helper defaults to 64 MiB; the production pipeline explicitly
admits up to 256 MiB. That allowance includes initializer/control, snapshot,
worker-window and encoded-consumer bytes. A scope binds the actual shape and
cell grid (128×128 pixels by default) and cannot upgrade its credit.
Boolean/uint8 ndarray masks
with uint8 scores use read-only borrowed views, fused scanning and bounded cell
copies when trusted activity/bbox hints are present; generic readers and missing
hints retain the original serial path. Deferred tile/native pieces and merged
source confidence remain on that compatible reader path. Scores are preserved where
the mask is nonzero, quantized zero remains unknown, frame/cell order is stable,
and each crop retains zlib level 3. Output payload, index and metadata semantics
are unchanged. Compiler/code-cache initialization is separate process overhead;
the helper changes neither global Numba thread settings nor disk caching.

Fused serial capture is not uniformly faster: direct typed calls, requested-one
workers, single-frame inputs or small admitted credits can expose the measured
dense/noisy regressions. The main full-frame pipeline uses its admitted caller
worker budget; local parallel gains do not establish a full-cluster speedup.

Per-view native-pull telemetry records the backend, admitted workers,
completed/total planes, elapsed time, and geometry/worker/consumer bytes. These
fields expose CPU projection progress while GPUs are idle. This execution change
is paired with `scheduler.wait_activity`, which separates running/queued parent
preparation, child-publication waits, pending inference, and active native
projection. A running parent has entered its executor and may still be awaiting
admission; it is not proof of current numerical work. The diagnostics do not
classify deadlock or change resource caps. The execution change does not
establish a full-cluster speedup. Matched local trials and source qualification
are recorded separately in task output directories and the workspace History.

## Geometry and evidence boundaries

Projection uses recorded canvas size, affine/axis transform, slice or angle
addresses, physical spacing, and shell/patch identity. A reduced prediction
canvas and the final native output volume are distinct grids. Source-volume
restoration must receive the intended final output shape and orientation.

The corrected Tilted and Azimuthal address plans use half-open native cell
support `[-0.5, N-0.5)`. Azimuthal retains the declared cylindrical ROI limit
`radius + 0.5`. Upright Azimuthal lookup follows the actual reconstruction-plan
angles, including nonzero origins, nonuniform spacing, and the closing seam.
The existing coarse-sweep plan still densifies requested angles into virtual
reconstruction planes assigned to the nearest completed source frame, with
half-turn mirroring where required. This patch preserves those reconstruction
rules; angular prediction interpolation is unchanged.
Azimuthal physical circle membership uses stable centered float64 coordinates
and an eight-float64-epsilon roundoff allowance at the closed boundary, while
angular and diameter raster quantization remains float32. Dense, sparse, and
confidence routes share that plane-address implementation; reduced prediction
canvases cannot change the physical circle or its boundary membership.
Reduction combines categorical foreground with OR and scores with maximum;
confidence uses the same address
plan. These rules do not extend the physical field of view.

When an Azimuthal radius must be inferred, it is `max(0, (diameter-1)/2)`.
A single-column plane therefore uses radius zero and column zero; its native
restoration retains the valid disk instead of admitting excluded corner pixels.
Positive explicitly declared radii are unchanged.

For shell views, valid support belongs to the declared shell/annulus and patch
field of view. Spherical coverage is not a promise to fill the entire cube.
Seams, rotations, padding, and chunk/ROI bounds must obey the same sampling
contract as the full output.

Spherical restoration uses stable centered native-grid coordinates and a shared
relative boundary comparison of eight float64 epsilons. This handles roundoff
at the declared closed annulus boundary; it does not add a voxel-scale margin
or change the radii. Face seams are closed and nearest-shell midpoint ties
resolve inward.

Radial pull admits valid height cells in `[-0.5, N-0.5)` and clamps their
nearest sample addresses. Tilted Radial applies that rule after inverse shear.
Its closed annulus uses the same roundoff-only relative tolerance, scaled
separately at each radius limit so the outer radius cannot enlarge the inner
hole. Source/model raster sampling, arc periodicity, and native-owner eligibility
remain unchanged; arc wrapping does not wrap radius or frame addresses.

Foreground, geometric validity, and observed confidence remain separate.
Projection must not invent detector confidence values, turn
unknown coverage into known background, or let scores and masks sample different
source locations. Source reconciliation, saved layers, and final publication
must consume the same settled projection.

Score projection uses numeric maximum over the same destination/source cell
relations as binary OR; zero score remains unknown evidence. Reduced Tilted
component layers project directly to final native shape rather than scatter to
an intermediate grid. Incremental and dense-fallback component stores record
`projection_geometry_contract="xta.native_destination_pull/1"` and the actual
output shape. Cartesian components retain axis permutation followed by one
source restoration.

Sparse upright Azimuthal components use a separable destination-plane pull with
the same reconstruction-plan angles and native stack-row OR. Sparse Tilted
Azimuthal
components share the dense/score destination addresses while reading encoded
CVOL bits directly. Raw and packed inputs preserve selected-component bounds
and physical shear; neither route requires a decoded native volume.

Sparse Tilted Azimuthal projection uses the prepared angular plan
and a fused compiled encoded-mask reader for these sparse Tilted Azimuthal
components. Bounded CPU workers own separate output slices, so packed output
bits cannot race. Geometry preparation is shared across the workers; source
payloads remain immutable. Memory admission includes output planes and packing
workspace, and live projection progress exposes work that previously appeared
only in completion counters.
If the prepared plan or one output worker cannot fit the configured fast-path
bounds, the existing bounded reference path remains available with an explicit
fallback reason. Invalid geometry and corrupt payloads still fail normally.
Successful publication retires the owned packed mapping and waits for its
scratch-file deletion before removing the temporary directory. Failed workers
or encoders preserve their original exception; traceback-held mappings remain
valid until their consumers retire, while partial owned output staging is
cleaned separately. Borrowed input stores are never part of this cleanup.

Projection correctness is checked with independent synthetic geometry and the
native destination-pull oracle. Timing traces measure pipeline behavior; they
cannot serve as a numerical correctness reference.

## Validation and reproduction

Routing checks establish worker admission and preserve separate
Cartesian/Radial/Spherical routes. Numerical checks must independently cover
physical ROI and field of view, reduced-to-native and anisotropic grids,
positive and all-black inputs, sharp background bands and shell gaps,
seams/padding, chunk/ROI paths, and mask/confidence alignment.

Spherical checks cover all faces, corners, poles, active-shell gaps, enlarged
cubic and noncubic grids, rotations, signed tilted groups, Cartesian aliases,
positive-score support and rational closed-radius boundaries. Radial checks
cover all base axes, enlarged and contracted noncubic grids, height caps, signed
tilted directions, shell gaps, radius boundaries and actual model-grid
restoration with intrinsic ring-tile seams. Sparse-component checks additionally
exercise encoded raw/packed masks, reduced-to-native support and single-column
disk projection. Confidence/publication checks cover inverse-shear values,
offset/nonuniform angular ownership, contraction maximum with unknown zeros,
positive-score versus binary support, and final native shape/metadata.

Receipts must retain source pins, workload identity, executed and skipped checks,
and device scope. Overlapping focused suites are not additive counts of unique
tests. CPU evidence does not establish CUDA or native-owner execution. Compiler
or routing success alone is not release qualification; the final source freeze
and complete qualification receipt remain separate.

Use a fresh task-specific output directory for reproducible synthetic CPU work:

```text
python -B tools/benchmark_sparse_azimuthal_projection.py --output ../Scratch/Experiments/REVIEW_NAME/synthetic-projection --shape 384 512 640 --target 256 480 608 --workers 4 --heatsoak-seconds 60 --repetitions 3
```

Replace `REVIEW_NAME` with the task's output directory name. The tool compares the
bounded NumPy path with compiled single/multiple-worker execution, checks an
independent scalar physical row-band oracle, and records exact native mask and
source hashes. `--source-root EXTRACTED_SOURCE_ROOT --backends numpy` can exercise
an extracted predecessor with the same synthetic fixture. Local CPU timings
remain sanity checks rather than target-cluster throughput claims. Preserve
run-specific measurements and qualification history outside the repository in
`Scratch/Data/XTA/History`.
