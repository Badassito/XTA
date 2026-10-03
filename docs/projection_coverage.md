# Projection coverage in v25.0.1

This patch addresses source-grid holes caused by projecting reduced prediction
canvases as isolated points. A positive categorical sample represents support
under its declared view transform and sampling rule. Projection must evaluate
that support on the destination grid, including the valid space between sample
locations. It must also retain genuine background and invalid geometry.

Geometric coverage is not blanket dilation. It does not expand a mask by an
arbitrary radius, fill an entire bounding box, paint outside the view's field of
view, or repair a missing model prediction. A detector's lost feature cannot be
recovered simply by changing projection.

## Routes

The CPU routes and qualifications are below. Actual device execution and the
complete release qualification are recorded in separate receipts.

| View family | Projection route | Qualification |
| --- | --- | --- |
| Cartesian: Transverse, Sagittal, Coronal | Canonical categorical restoration; matching bounded D1 native-coverage pull for eligible layouts | CPU route/coverage checks passed; device receipt separate |
| Tilted Cartesian | Shared view union followed by canonical destination pull | CPU passed; projection CUDA disabled |
| Upright Azimuthal, dense | Shared view union and canonical destination pull; eligible Transverse CUDA, Sagittal/Coronal CPU | CPU passed; device receipt separate |
| Upright Azimuthal, coverage | Declared reconstruction-plan angular/column pull; eligible Transverse CUDA, Sagittal/Coronal CPU | CPU passed; device receipt separate |
| Tilted Azimuthal | Shared view union and bounded canonical CPU pull; unsafe CUDA plan refused | CPU passed; projection CUDA unsupported |
| Radial and tilted Radial | Bounded native shell pull; eligible parent CUDA or separate native owner | CPU passed; device receipt separate |
| Spherical and rotated Spherical | Bounded destination-native QSC pull | CPU passed; device receipt separate |

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
Azimuthal CUDA plan encodes the unsafe scatter geometry, so this patch refuses
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
establish a full-cluster speedup; matched local trials and source
qualification are recorded separately under
`Scratch/Experiments/Job150772_Performance_20261002`.

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

## CPU qualification and release receipts

The focused receipts below describe the original v25.0.1 coverage patch.
The Job150772 CPU execution refinement has separate qualification receipts;
its local measurements do not qualify a full-cluster inference run.

Routing tests currently cover removal of the unsafe non-Cartesian D1 dispatch.
The frozen D1/compatibility CPU suite reports 89 passed, 315 subtests, and eight
skipped CUDA checks. Its assertions cover worker admission and preserve the
separate Cartesian/Radial/Spherical routes.
They do not alone establish numerical correctness for every view. Separate
device and full-suite receipts must record independent physical ROI/field-of-view
oracles, reduced-to-native and anisotropic grids, positive and all-black cases,
sharp background bands and
shell gaps, seams/padding, chunk/ROI paths, and mask/confidence alignment.
Unavailable or unexecuted CUDA checks must remain explicit.

Spherical CPU evidence now reports 47 independent native-coverage tests and
61 existing tests with 1,177 subtests passed; two CUDA checks were skipped.
The independent cases cover six faces, seams/padding/corners/poles, active-shell
gaps and ROI limits, enlarged cubic and noncubic output grids, four rotations,
signed tilted groups, Cartesian aliases, and positive-score support. The
rational boundary audit records zero missing and zero excess voxels after the
fix. Actual CUDA execution is recorded separately.

Radial CPU evidence includes 60 new tests in a broader receipt of 96 passed,
642 subtests passed, and 38 skipped checks. It covers all three base axes,
enlarged and contracted noncubic grids, height caps, signed tilted directions,
shell gaps, positive-score support, rational radius boundaries, and actual
model-grid restoration/augmentation with intrinsic ring-tile seams. The CPU
receipt does not establish execution of the parent CUDA or native-owner paths.

Canonical Tilted/Azimuthal final-v2 CPU evidence reports 360 tests and 335 subtests
passed with all nine audited source pins unchanged before and after the run.
Sparse-component final-v2 evidence reports 130 CPU tests and 235 subtests passed,
including encoded-mask, reduced-to-native, and single-column disk projection
paths. Its audited
source pins also remained unchanged. These receipts overlap other focused
suites and must not be added together as a count of unique tests.

The confidence/publication CPU audit reports 323 tests and 77 subtests passed,
with 13 GPU-specific tests skipped. Independent checks cover inverse-shear
numeric values, offset/nonuniform angular ownership, contraction maximum with
unknown zeros, exact positive-score versus binary support, and final native
shape/metadata publication. These are correctness receipts, not benchmarks.

Receipts, plots, and the short mobile report belong under
`Scratch/Experiments/Projection_Coverage_v25_0_1_20261001`. Compilation or routing
success alone is not release qualification; the final source freeze and full
qualification receipt remain separate.
