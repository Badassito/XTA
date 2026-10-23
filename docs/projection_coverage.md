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
| Tilted Cartesian | Shared view union followed by canonical destination pull | Shared CPU pool; exact CUDA gather after device admission |
| Upright Azimuthal, full-mask output | Shared view union and canonical destination pull | Shared CPU pool or admitted exact native CUDA gather |
| Upright Azimuthal, coverage | Declared reconstruction-plan angular/column pull | Existing eligible Transverse CUDA; native destination gather also supports all bases |
| Tilted Azimuthal | Shared view union and bounded canonical destination pull; unsafe scatter refused | Shared CPU pool; exact CUDA gather after device admission |
| Radial and tilted Radial | Bounded native shell pull | Eligible parent CUDA or separate native owner |
| Spherical and rotated Spherical | Bounded destination-native QSC pull | CUDA requires its own eligible layout and device checks |

Immediate D1 point scatter is no longer used for Tilted Cartesian, full-mask
Azimuthal, or Tilted Azimuthal, including hybrid detector runs. Their masks pass
through a bounded shared view union and the canonical projector. Cartesian D1
native coverage and the separate Radial native owner remain available.
The legacy D1 worker also rejects stale non-Cartesian requests before allocating
CUDA state, so a later routing mistake cannot silently reactivate point scatter.

This route change can add host shared-union and storage overhead to workloads
that previously used non-Cartesian D1. It does not disable all GPU projection:
eligible Cartesian, upright Azimuthal, Radial, and Spherical projectors
retain their separate device paths. The new native gather also handles upright
Sagittal and Coronal plans. SAM and detector model
inference on GPU is unchanged. Full-cluster wall-time improvement remains
unmeasured; local CPU projection trials have a separate scope.
The old Tilted Azimuthal CUDA plan encodes unsafe scatter geometry and remains
disabled. The separate native CUDA gather evaluates the destination cells with
the same prepared geometry and contributor rules as CPU projection. It does not
reactivate D1 scatter. Inference retains device priority; an existing CPU
projection can hand off its unfinished suffix when a device becomes available.
Native gather uses the same bounded source-retirement admission as Spherical
projection. An idle device can be borrowed when no inference task is admissible;
under a live backlog, pressure or starvation age grants a bounded turn and then
returns the device to inference. The global inference-active flag alone does not
block a native gather. Admission reasons and current device leases are recorded.
Local numerical and performance checks do not establish full-cluster wall time.

## Compact upright Azimuthal sampling

Automatic upright Azimuthal requests can use fewer angles at the existing
compact raster size without creating patches. For diameter `D`, `N` endpoint
samples and angular gap `delta` radians, let
`p = max(1/2, (D-1)/(2*(N-1)))`. The selected gap satisfies
`p*p + (D*delta/4)^2 <= 0.99`. The independent stack-axis bound is
`max(1/2, stack_length/(2*rows))^2 <= 0.99`; it is not added to the planar
quadrature because it is a separate source coordinate. Both bounds retain a
positive native interpolation tap with numerical margin.

The direct sampler's positive-weight fringe clamps preserve diameter/stack
endpoints through admitted model padding. Existing guards still require a
qualifying unrotated pass, source axes within 4096, linear source sampling,
matching endpoint radius and supported raster pitch. Tilted, nearest-source,
excessively coarse and explicit user spacing requests keep their established
behavior. Coarsened views retain native pull projection, never D1 point scatter.
This certifies the interpolation basis, not a nonzero gray8 result or nearest
categorical-label coverage.

For F3 working shape 2911/3064/3022 and imgsz2048, upright Transverse angles
fall from 4,747 to 3,557; Sagittal and Coronal each fall from 4,573 to 3,284.
Total native frames decrease 13,893 to 10,125 (27.12%). Radial patching and its
current certified radius spacing are unchanged. These are planning counts;
GPU/model throughput and quality require separate qualification.

## Bounded CPU execution

The CPU Tilted and Azimuthal routes use prepared geometry and compiled
destination-owned OR/MAX loops in `projection_coverage_cpu.py`. Numba releases
the GIL without fastmath or a disk compilation cache. Angular ownership retains
the exact prepared NumPy address rules. Initialization and angle records are
checked against their conservative build allowance before table allocation.
Plans retain local read-only numeric tables, with packed uint32 or uint64 owners
as capacity requires; there is no global Python plan LRU. The kernel does not
copy its source mask or create a global thread pool.

Independent array callers retain their requested per-view worker count. During
a pipeline run, native pulls share one CPU pool rather than dividing their
parallelism by the configured number of parent preparations. The pool grows
at inference drain, including for projections already running. It fairly shares
credits across live projections and bounds both running operations and completed
planes awaiting their consumer. The per-projection workspace allowance remains
an additional cap. A slow consumer therefore cannot accumulate an unbounded
queue of planes or multiply the CPU budget by the number of views.

Each worker owns a distinct destination plane, and callbacks receive planes in
source order. Cancellation/error paths join all readers before releasing their
source. Tilted component layers now use the existing incremental sparse-store
writer, avoiding a full native destination allocation and its later encoding
scan. The existing dense retry remains available after a safely discarded sink
failure, so its conservative parent memory reservation is retained.

CPU kernels traverse contiguous row spans and reuse coordinate bounds within
each row. The contributor order, nearest rounding, numeric maximum and binary
OR semantics are unchanged, including partial flat spans and strided sources.

| Environment control | Default | Scope |
| --- | --- | --- |
| `YOLO_TTA_NATIVE_PULL_BACKEND` | `compiled` | `compiled` or the exact bounded `numpy` reference route |
| `YOLO_TTA_NATIVE_PULL_PLAN_MIB` | `64` | Per-plan geometry/build allowance; packed Azimuthal lookup is read-only and reused across planes; larger geometry uses bounded strips |
| `YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB` | `256` | Per-projection pending-plane and per-worker strip allowance, including a consumer-held plane for sink-only publication |
| `YOLO_TTA_NATIVE_PULL_CHUNK` | `65536` | NumPy reference chunk points, clamped to 1 through 262,144 |
| `YOLO_TTA_NATIVE_PULL_CUDA` | `1` | Allow exact native gather when the existing GPU stage coordinator grants a device; `YOLO_TTA_GPU_BACKPROJECT=0` still disables it |
| `YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS` | `67108864` | Minimum remaining output work for automatic CUDA setup; avoids source upload for short suffixes and small projections |
| `YOLO_TTA_NATIVE_PULL_GPU_RESERVE_GIB` | `1` | Device memory kept free beyond the admitted source, plan and output scratch |
| `YOLO_TTA_NATIVE_PULL_GPU_UPLOAD_MIB` | `8` | Maximum input chunk for packing/upload; the whole host volume is never copied to a second dense staging array |

The plan and workspace caps are separate. Source/dense-destination credits,
other concurrent views, and runtime/JIT compiler memory remain separate from
these explicit geometry and buffer bounds. A workspace too small for one worker
and the consumer plane is refused. Production uint8 masks/scores use the
compiled route; other historical array dtypes retain the exact NumPy route
without a whole-source dtype copy.

The CUDA session packs categorical nonzero values to one bit per voxel; numeric
MAX retains uint8 values. Angular keys are prepared by the same NumPy geometry
plan used on CPU. Device arithmetic uses float64 without fused multiply-add or
fastmath. Output scratch is bounded to a plane/strip and copied to the caller's
owned destination. A private allocator and stream retire before the device lease
is released. Declared admission failures retain CPU execution; execution or
unsettled cleanup failures abort publication instead of retrying with live GPU
ownership. The explicit `numpy` backend remains on CPU.

Handoff preserves the already-published prefix. The CPU stream cancels queued
work and joins running readers before CUDA writes the remaining planes, so no
callback is duplicated and no borrowed source is retired while still in use.
Small synthetic probes can favor CPU because CUDA setup and transfer costs
dominate; production throughput must be checked with the actual workload.

## Packed parent retention

Eligible terminal Tilted and Azimuthal parents seal their cleaned binary masks
into the existing cropped, row-packed CVOL representation before projection.
This uses one bit per stored sample and omits empty crops. On Linux the encoded
payload can grow in a memfd while physical RAM headroom permits; it spills to
disk between bounded writes if headroom becomes insufficient. It requires no
ZRAM, swap device or system configuration change. CPU and CUDA project directly
from the encoded crop/index, without recreating the dense native mask.

The dense input remains charged while packing. Credit returns only when its
last actual owner dies, and the compressed resident bytes stay charged until
publication settles. Delayed mask and confidence aliases are fenced separately.
Completed GPU publication callbacks detach their input mappings immediately
after their copy Future settles; callback reference cycles no longer keep those
large mappings alive until inference-wide garbage collection.

`YOLO_TTA_PACKED_PARENT=0` disables this continuation for comparisons.
`YOLO_TTA_PACKED_PARENT_MIN_MIB` defaults to 64 to avoid encoding tiny inputs.
`YOLO_TTA_PACKED_PARENT_RAM_RESERVE_MIB` defaults to 1024 and guards memfd growth.
Interpolation/extrapolation, tiled parents, retained debug workspaces, an already
published layer, and explicit NumPy projection retain their established paths.
Packing changes storage, not view resolution or the destination coverage rules.

Terminal projected layers also use private packed CVOLs. Their retained RAM
grants share the existing Radial publication budget, with the large parent
outputs allocated first. Grants include future working/output reserves and
descriptor limits. Planned memfd payloads survive until union/NRRD readers
finish and normal scratch cleanup releases them. A memory decline or spill
keeps the disk payload packed. Public NRRD pixels and source geometry are unchanged.

Tilted inference inputs use the existing resident GPU renderer by default.
Speculative CPU Tilted preparation requires `YOLO_TTA_CPU_INPUT_TILTED=1`;
Cartesian prefetch retains its existing control. Eligible disjoint Spherical
GPU task leases finish exact per-slice hole filling before their D2H publication,
and report coverage so the parent can skip the completed-view CPU pass.

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
from the source-aligned score reader above. During a run its ordered frame jobs
share the native-projection CPU pool. The charged memory window is planned before
parent admission; actual CPU credits are divided among captures that have obtained
a numeric staging slot. Queued parent threads do not reserve CPU lanes. Standalone
callers retain their requested worker count. The original masking, maximum and
payload/index semantics remain unchanged. Exact plan workspace is added to the
existing parent transient request, capped at 256 MiB per parent;
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

Confidence payload/index writes are batched within the charged consumer window.
Logical byte counters replace per-cell file-position queries. Numeric stages
prefer two owned Linux memfds within the existing 64 MiB per-stage and 256 MiB
combined staging limit. Small metadata files and the stage directory retain the
established artifact layout. Unsupported platforms or insufficient physical
headroom report a disk fallback. Numeric handles stay owned through the final
copy and close before their slot credit returns.

Where a compatible zlib C API is available, a probed native loop performs masked
copying and independent level-3 compression without Python calls inside a batch.
Python coordinates ownership and publication. `YOLO_TTA_CONFIDENCE_NATIVE_CODEC=python`
selects the portable encoder for comparison. Default `auto` probes Python's zlib
extension and the system library on Linux; backend, codec version and fallback
reason are recorded. Every path preserves the zlib-block format and decoded uint8
values. Different library versions are identified explicitly. Workspace and
source/output parity checks belong in the qualification receipt.

Fused serial capture is not uniformly faster: direct typed calls, requested-one
workers, single-frame inputs or small admitted credits can expose the measured
dense/noisy regressions. The full-frame pipeline shares its admitted CPU budget
across active jobs; local parallel gains do not establish a full-cluster speedup.

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

TTA, PTA and LTA use the same coverage-only physical-view planner. There is no
selectable dense sampling mode. PTA's built-in physical render uses angle zero;
TTA and LTA preserve their requested angles. Spherical and Radial render their
native patch before the in-plane affine, so their native sampling certificates
do not depend on including angle zero. Azimuthal's fused TTA model-grid sampler
retains its separate coarsening admission check: without a qualifying unrotated
pass, it keeps conservative angular spacing and records the reason. Explicit
Azimuthal spacing and unsupported geometry retain their existing semantics.

Spherical plans vary both the angular grid and radial spacing by band. Each
band keeps one endpoint-inclusive QSC lattice, fixed patch origins and a uniform
radius trajectory. Patches sample the volume directly at native coordinates;
the planner never combines four rendered patches into one resized input.

The approved certificate means that every voxel center in the declared annulus
has positive weight in at least one direct trilinear input sample. For a band's
outer radius `R`, face interval count `n` and largest realized shell gap `g`, it
checks an outward-rounded bound:

```text
g*g/4 + 2*(41/50)*(R/n)*(R/n) <= (255/256)^2 < 1.
```

The inverse QSC map's squared Lipschitz bound `41/50` is certified offline with
exact rational interval arithmetic, a short Machin-series enclosure of pi, and
1,024 closed intervals. That certificate is not evaluated during production
planning. The remaining 1/256-voxel distance reserve exceeds the tested coordinate
rounding envelope for the existing qualified FP32 path (axes at most 4096).
FP64 coordinate tests include large anisotropic centers. These numerical tests
are not a formal proof of every platform's math library.

The guarantee is positive native input contribution, not selection of every
voxel by the nearest categorical sampler or survival through arbitrary later
augmentation/downsampling. The separately evaluated nearest-voxel proposal was
rejected. Intensity and categorical rendering retain their established samplers.
Historical v1/v2 captures retain their original bounds and provenance; their
validation does not expose an alternative production sampling mode.

Adjacent bands share their exact endpoint radius. Both lattices render that
shell, and the closed projection annuli meet without a gap. Channels and SAM
interpolation stay within each band because its frame spacing and ownership are
part of the trajectory, even when neighboring bands share the same angular grid.

The planner minimizes actual native patch images among a small practical set:
a fixed outer grid, the adaptive patch-class partition of a uniform shell grid,
and two, four or eight bands placed uniformly in radius or in an arcsine transform
of radius. Each candidate band is independently certified after constructing its
actual uniform radii. Duplicate endpoints are charged in the cost; ties prefer
fewer bands. This is a bounded candidate comparison, not a global-optimum claim.
Shell count may increase or decrease when doing so reduces total patch images.

For `imgsz=2048`, automatic minimum radius and one upright cube/angle:

| Working shape T/Y/X | Previous dynamic TTA patches | Previous dynamic PTA patches | Current shared patches | Current unique shells | Current band trajectories |
| --- | --- | --- | --- | --- | --- |
| F3: 2911/3064/3022 | 13,326 | 24,504 | 5,430 | 898 | 48 |
| Synthetic: 3072/3072/3072 | 14,430 | 28,872 | 6,390 | 1058 | 48 |

The F3 result uses eight bands of 108–122 radii each. Its 905 shell occurrences
include seven duplicated boundaries; unique radii are counted once. All eight
bands fit one native patch per face. Four arcsine-spaced bands use 5,778 images
and 24 trajectories, while eight use 5,430 and 48. The lower patch count wins the
stated objective, but additional trajectory setup and downstream SAM work remain
runtime costs. Planning counts are not measured cluster speedups or model-quality
claims. [Candidate comparison](../../Scratch/Experiments/spherical-coverage-20261014/joint-planning/candidate_counts.json)
and [implemented counts](../../Scratch/Experiments/spherical-coverage-20261014/joint-planning/implemented_counts.json)
retain the exact inputs and band layouts.

Planning scans a small set of scalar candidates and shell radii, never every
source voxel. One band plan is shared across its faces and patches. GPU direction
keys exclude radius/band identity and retain the existing bounded cache; CPU cone
bounds share at most 128 immutable face/rotation entries. Projection's necessary
voxel work remains in compiled CPU or CUDA kernels. The CPU reference renderer
still evaluates vectorized QSC coordinates in bounded row strips; it does not
allocate an uncharged persistent full-patch direction bank.

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

Spherical checks cover actual native input taps, changing band lattices,
shared radius endpoints, all faces, corners, poles, active-shell gaps, enlarged
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
