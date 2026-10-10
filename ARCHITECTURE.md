# XTA architecture

XTA has three production modes: test-time augmentation (TTA), pretraining
augmentation (PTA), and label-time augmentation (LTA). This document describes
their data flow, resource ownership, and failure boundaries. Measurements,
experiments, and release history live in the workspace's
`Scratch/Data/XTA/History`, outside installed packages and source bundles.
Historical evidence uses the
[release identity map](../Scratch/Data/XTA/History/release/version_migrations.json) and
[interpretation rules](../Scratch/Data/XTA/History/release/README.md#historical-release-identities).

## Entry points and shared contracts

The versioned SLURM launcher, the installed `xta` command, and
`python -m XTA` enter `XTA.cli.run()`. The CLI selects exactly one mode and
validates that mode's grammar before importing its heavy runtime. `tta_mode`
enters `pipeline.main`, `pta_mode` resolves `PtaConfig` before `pta_runtime`
enters `pta.main`, and `lta_mode` resolves `LtaConfig` before `lta_runtime`
plans and executes propagation. Dependency-heavy runtimes initialize only
after their mode has been selected.

| Shared boundary | Owner |
| --- | --- |
| Immutable geometry and sampling identity | [sampling](XTA/unification/sampling.py), [physical recipes](XTA/unification/geometry_identity.py), [contracts](XTA/unification/contracts.py), [views](XTA/unification/views.py), [tiles](XTA/unification/tiles.py), [geometry](XTA/geometry.py), [render batches](XTA/render_batch.py) |
| Source and worker resources | [workspace](XTA/workspace.py), [media](XTA/media.py), [runtime](XTA/runtime.py), [TTA workers](XTA/workers.py) |
| Input/output identity and manifests | [launch context](XTA/unification/context.py), [manifest helpers](XTA/unification/manifest.py), [TTA manifest](XTA/unification/tta_manifest.py) |
| Image-space and source-space projection | [backprojection](XTA/backprojection.py), [sparse projection](XTA/sparse_projection.py), the `cylindrical_*`, `spherical_*`, and `tilted_azimuthal_projection*` modules |
| Common filtering and output | [Gaussian](XTA/gaussian.py), [finalization](XTA/finalization.py), [outputs](XTA/outputs.py), [packed publication](XTA/packed_publication.py), [memory grants](XTA/publication_memory.py) |

A `RasterPlan` fixes a concrete view, output raster, optional tile, and sampling
policy. Its digest identifies the request and is recorded with output provenance.
`ForwardSamplingPolicy` specifies the order of coordinate transforms, sampling
kernels, and boundary rules. A runtime binds a data role and implementation
through `require_forward_sampling()`; missing bindings fail rather than silently
selecting another kernel. Image intensity and categorical labels use distinct
sampling rules. Categorical rendering uses nearest spatial taps. Tilted
Cartesian and Azimuthal label planes blend adjacent stack planes and threshold
binary foreground and coverage at 0.5; shell views use nearest voxel sampling.

TTA, PTA, and LTA raster plans bind versioned physical geometry and the effective
affine transform at their construction boundary. Source dimensions, orientation,
trajectories, and sampling matrices contribute to identity. Exact trajectory
digests keep metadata compact; source-memory addresses and display labels do not
establish physical equivalence. Shared tilted render-plan keys bind their physical recipe
and exact effective affine; dense Azimuthal map keys retain exact angular ownership.
The Coronal pixel-block cache additionally binds source owner, layout,
dtype, and captured block width without retaining a source volume indefinitely.
Stored recipe snapshots are immutable; callers must keep the executing geometry
unchanged after binding. Snapshotting does not freeze every producer NumPy array.

Angle identifiers preserve conventional spelling when it round-trips exactly and
otherwise retain full float precision. Distinct requested views cannot disappear
through rounded-name deduplication. Generic raster records without complete
physical recipes remain readable with their original hashes, but cannot establish
equivalence to a task's complete recipe. Parent and worker code must use matching
plan semantics.

`RenderRequestBatch` describes logical frame addresses and plan identities;
`render_batch.RenderBatch` owns the actual rendered frames consumed by a model
or image sink. Full frames and tiles share geometry. Cartesian, tilted
Cartesian, Azimuthal, Radial cylindrical, and Spherical QSC views use their
own physical coordinates. Radial and Spherical patches retain their shell,
radius, and patch origin so backprojection can reconstruct source support.
A view is projected to source coordinates only after its own frame and variant
ownership has settled. Source-volume restoration to native shape is distinct
from the working cube.

## Projection coverage

Reduced categorical canvases must restore the geometrically valid source support
represented by their samples, rather than publish isolated scatter points on a
larger destination grid. Canonical destination pulls follow each view's axis,
affine, angular, or shell/patch contract. Real background, invalid field of view,
and shell gaps remain excluded. Coverage reconstruction is not blind dilation or
an attempt to recover a feature absent from the prediction mask.

Dispatch excludes immediate D1 point scatter for Tilted Cartesian,
dense Azimuthal, and Tilted Azimuthal, including hybrid inference. Those routes
settle a bounded shared view union before canonical source projection. Eligible
Cartesian D1 native coverage and the separate Radial native owner publish through
their own routes.
Legacy D1 admission rejects stale non-Cartesian tasks before CUDA allocation.
Native Cartesian, Transverse upright Azimuthal, Radial, and Spherical projection
GPU paths have independent admission. Upright Sagittal/Coronal Azimuthal uses
orientation-aware CPU projection; model inference can execute on the GPU.
Ordinary Tilted projection uses canonical CPU processing, including engine views.
Tilted Azimuthal uses a bounded canonical CPU pull and refuses its legacy CUDA
scatter plan before allocation or publication. Both tilted routes require shared
view-union storage and CPU projection capacity.

CPU Tilted/Azimuthal projection prepares exact geometry once and runs compiled
nogil OR/MAX kernels without fastmath. Preallocation checks admit local numeric
geometry before allocation; plans contain no global Python record LRU.
A bounded per-view thread pool honors the
caller worker budget, owns disjoint native planes, and delivers callbacks in
order. Read-only packed angular lookup has a separate 64 MiB plan/build default;
the 256 MiB transient default accounts for pending planes, worker strips, and a
consumer-held plane. Larger lookup geometry uses bounded strips. These are
per-projection explicit-buffer limits, separate from dense credits and JIT
runtime memory. Confidence retains a scalar-max plan within its own admission,
with the exact bounded sampler for small budgets and no reader thread pool.
Projection telemetry exposes progress and admitted workers; scheduler wait
records distinguish executor activity, queued preparation, pending inference
and child-publication waits.

Mask, validity, and confidence sampling must agree with final native shape and
geometry, and publication waits for complete producer coverage. Shell validity
is the declared annulus/patch domain, rather than the entire cube. Radial height
cells include valid half-cell caps; stable centered coordinates and roundoff-only
closed-radius comparisons preserve exact Radial/Spherical boundaries without
changing radii or filling shell gaps. See
[projection route and qualification matrix](docs/projection_coverage.md) for
per-family evidence and limits. Routing checks and numerical qualification are
reported separately.

External policies own transforms *after* shared rendering. TTA and PTA accept
CPU and GPU policy paths through `--augmentation`. Policy identity and selected
backend are recorded in manifests. A policy must preserve its paired image and
label contract; unsupported combinations fail during validation. PTA executes the
same source bytes whose SHA-256 it verifies, rather than reloading a policy or
accepting cached bytecode after the identity check. Ultralytics adapters check
the private APIs they replace and fail worker startup if a required patch cannot
be installed.

## Environment controls

The supported defaults select ROI-only CPU mask resizing, GPU
proto union, retina flattening and eligible GPU warping, tiled proto composition, bilinear
Tilted intensity sampling, slice-local interpolation labels, bounded bridge
merges, and run-based topology adjacency. Categorical sampling, generic compact
label IDs, and automatic eligibility and failure fallbacks have separate contracts.

Output publication uses cropped CVOL stores, eligible compact Spherical CPU
publication, crop-row NRRD streaming, extent skipping, and live immutable global
layers. Final fusion uses grouped restore and the eligible native sparse CPU
path. Grouping unions before resampling preserves subpixel support that separate
integer AREA restores can round away.

Worker GPUs retain inference-first ownership, full-frame workers write bounded
direct unions, and result queues wake the scheduler through the result pump.
The scheduler reuses identical parent-admission decisions within each
read-only backlog scan. It also returns the confidence portion of a parent's
dense credit after the backing mapping and its aliases have actually retired,
so slow source projection does not keep charging confidence storage it no longer
owns. Parent-union credit remains held until its own retirement.
Split-view hole filling runs after view completion; single-lease eligibility
permits device filling. Resource controls set concurrency and peak storage.
`YOLO_TTA_GPU_INPUT_STAGING_BATCHES=0` disables input staging and its eager source
warmup. `YOLO_TTA_HYBRID_GPU_STEALBACK_MAX_FRACTION=0` disables hybrid GPU assistance.

## Required compiled CPU backends

Numerical runtimes require Numba alongside NumPy, OpenCV and SciPy. Missing or
broken Numba imports raise a diagnostic instead of selecting interpreted bulk
work. Compiled topology, interpolation, Spherical/Radial CPU projection, sparse
scatter and packed publication propagate kernel failures. Configuration and CLI
help/version discovery keep their lightweight import boundary.

Independent slow implementations live in `tests/reference_backends` for
qualification. Interpolation grows its compiled workspace for valid fragmented
windows; Radial uses a bounded compiled direct pull when a factored ownership
plan cannot fit. Native-array operations that preserve aliasing/conversion
semantics and explicit OpenVINO CPU inference remain supported. GPU eligibility
and safe fallback to compiled CPU remain separate from compiler availability.

H100 is the primary deployment target and A100 is the fallback target; Volta/V100
is outside the supported deployment scope. See [production backend policy](docs/PRODUCTION_BACKENDS.md)
for the test-reference boundary and development benchmark commands.

## TTA: inference to source union

The TTA pipeline is coordinated by `pipeline`, `tta_scheduler`, `tta_prediction`,
`tta_lifecycle`, `tta_terminal`, and `tta_outputs`. `view_prepare` owns admitted
parent preparation, component-projection reservations, and inference-to-postprocess
lease handoffs through explicit state objects. The scheduler owns mutable
task identities, backend claims, worker liveness, parent memory credits, and
result transport. An execution target is a local CUDA GPU or a socket-local
OpenVINO process. Worker processes receive picklable contracts; bulk source and
result data use shared descriptors or artifact paths. CUDA, TensorRT, and
OpenVINO are initialized inside their respective runtime owners.
Immutable task descriptors keep the scheduler's shape and byte accounting
stable across refill passes; mutable owner, credit, and readiness decisions
are evaluated at selection time.

Each physical view renders frames and optional tiles/augmentations, performs
model inference, assembles the accepted masks, and settles its image-space
result. Completed physical views can project to source space while other views
continue inference. `assembly` and `tta_terminal` reduce these independent
layers into a single source union. `interpolation` can add bridges between
eligible frames; those bridges retain their own provenance. `finalization`
applies requested global postprocessing after the full union exists.

`--task segment` (default) composes YOLO instance masks from detections and
prototypes in `inference`. `--task semantic` decodes raw one- or two-channel
logits in `semantic_inference`: sigmoid for one foreground channel, or softmax
for background/foreground channels. `--conf` thresholds foreground probability
before binary cleanup, including thresholds below 0.5. `--min_conf` retains a
connected component only if its maximum encoded confidence reaches the
threshold. Models with unsupported class counts or baked argmax output fail
rather than inventing missing probabilities. `tools/export_semantic_logits.py`
exports a raw-logit ONNX/OpenVINO graph when the ordinary export path bakes an
argmax. `semantic_cuda` and `semantic_trt` provide eligible CUDA cleanup and a
private two-context TensorRT ring; the instance pipeline retains its separate
inference path. GPU failures after a partially consumed task abort it rather
than replaying writes.

GPU instance-mask composition uses a deliberate approximation: it reduces
instance logits with a maximum on the prototype grid before interpolation.
Interpolation and that maximum do not commute. The result can add foreground
between neighboring instances even away from box boundaries; native-grid
restoration can enlarge the difference. Coverage qualification proves the
projection of the accepted canvas, not equivalence to per-instance retina
masks. Quality changes to this reduction need a separate mask-level comparison
and end-to-end timing. Disabling direct prediction does not select an exact
GPU alternative.

Generic prediction consumers require every logical frame and declared angular
padding frame before reporting completion. Missing or duplicate logical results
abort the task, and asynchronous failure drains pending destination writes.
If both GPU accumulation and CPU mask recovery fail, the frame is an error
rather than a successful empty prediction.

TTA augmentation runs each policy pass independently and records the policy
snapshot. Each pass owns its result and inverse-validity support; passes
contribute to the terminal union while interpolation uses the base pass.
Supported CUDA paths map accepted masks directly to owned parent windows.
Parent admission charges all passes in a policy group against the dense
memory window; file-backed results and retained outputs remain separate
storage consumers. Inference, completed canvases, source projection, and
publication have separate credits so one backlog cannot grow without bound.

Cartesian D1 publication uses `d1_orthogonal_coverage` to reproduce canonical
source restoration: transpose the view axes first, apply the temporal
union rule, and cover the exact nearest-neighbor XY cells. A smaller detector
canvas must not become isolated points on a larger source grid. Small lookup
tables and a bbox-bounded CUDA pull kernel provide that coverage; source bits
remain an idempotent union across detector chunks. Spatial area-contraction
cases use the established bounded shared-union/canonical projection route.
Confidence uses the matching source sampling taps. This route covers
Transverse, Sagittal and Coronal; oblique and Azimuthal kernels are separate.
Source publication begins only after the complete producer coverage has settled.

Sparse Tilted Azimuthal bridge publication uses the same prepared angular
ownership and native destination equations as dense projection, with encoded
CVOL bit reads fused into the compiled CPU loop. Workers own distinct output
slices under a bounded plane-and-packing budget. This avoids serial construction
of destination address vectors while completed-parent masks hold admission
credit. Live projection gauges include this sparse work. Projection correctness
requires independent geometry oracles; pipeline timing has a separate scope.

`--reconciliation POLICY.py` evaluates completed source-space component layers
before global postprocessing. The additive layers remain available; the policy
changes the derived final mask. `reconciliation_policy` resolves external
policy identity; `reconciliation_components`, `reconciliation_geometry`,
`reconciliation_io`, and `reconciliation_runtime` consume bounded source slabs
and write its manifest. A plain union can reuse the assembled union. Weighted
and confidence policies read independent evidence. `confidence_*` modules
capture native pieces or source-aligned uint8 maxima of detector scores
(segment) or foreground probabilities (semantic); zero means unknown.
Interpolation does not fabricate confidence. Native evidence conversion is
explicit and bounded. `confidence_export` can match numeric evidence to
selected compact masks without loading the NRRD masks.

### Interpolation and SAM extrapolation

TTA selects `--interpolation_backend sdf|sam`, default `sdf`. The selector
applies to both full-frame and consolidated tile interpolation. Distance zero
disables interpolation. Active SAM uses a separate `sam:` bundle and CUDA pool
resolved by `--sam_device`; CPU detector execution does not require a GPU
detector artifact. SAM never falls back to SDF. SAM implements TTA's orthogonal,
tilted, Azimuthal, Radial, and Spherical view families; unknown
geometry fails preflight. Detector augmentation is inverted before SAM consumes
the canonical angle-zero accumulation canvas, with the original angle retained
as provenance. LTA remains Transverse-only under its separate execution contract.
See [SAM propagation controls and replay](docs/sam_interpolation.md).
Native addresses are on the detector/processing-volume canvas;
canonical preparation may resample stack depth, so they do not imply raw source
slice identity. SAM intensity rendering preserves that detector canvas and
records its transform back to source coordinates.
Only Azimuthal frame order wraps, using half-turn mirrored-column aliases.
`xta.sam_cyclic_view_frames/1` preserves bounded unfolded evidence addresses;
publication folds selected support into native frame/column coordinates before
categorical source projection. Radial and Spherical radius frames clamp without
cross-patch fusion. Native proposal checks are measured,
while final transformed-source connection survival may be explicitly
`not_assessed`. Focused CPU routing checks are separate from full model/cluster
qualification of each orientation.
Generic SAM views share at most one materialization of the processing
memmap, then use exact cropped TTA sampling grids. Immutable image caches reuse
identical demands; eligible CPU routes also reuse covered subsets and overlaps.
Tracker state stays independent.
Runtime telemetry separates source materialization, rendered pixels, and cache
reuse from predictor startup and GPU dispatch.

`YOLO_TTA_SAM_CROP_MODE=whole|tiled` is an experimental SAM tracking selector,
default `whole`, validated and pinned for one launch. It preserves the
working canvas and family context. Whole rectangles use SDK resizing; tiled
contexts use independent original-seeded working-canvas footprints with a
1008-side cap, 128 halo, and fixed midpoint ownership. Full raw halos retain
their parent/tile lineage and frame coverage. No cross-tile state or predicted
seed handoff occurs. This setting selects generation geometry and is
separate from detector tiling and quality policy.
Runs without active SAM interpolation or extrapolation ignore this unused
setting and record a null mode. The retained canvas contract distinguishes working and native shapes;
disabling delayed native mask expansion does not bypass processing-cube
resampling.

SAM-only extrapolation is independently enabled by `--extrapolation_distance`
(default zero), with `--extrapolation_walk_back` (default one) and
`--extrapolation_min_radius` (default three). `sam_extrapolation_planning` freezes
the local post-interpolation baseline and seeds remaining outward terminals.
The radius gate skips terminals at or below the threshold; predicted thin tails
remain untouched. Independent walk-back runs retain their original seeds and
count distance beyond the terminal. All plans precede tail merging.
`sam_extrapolation_policy` retains a prefix until raw SAM emptiness or the
available distance horizon, continuing through unrelated objects without using
their masks as prompts. Publication subtracts the frozen native baseline and
uses a separate extrapolation provenance role. The shared persistent SAM context
avoids model reloads. Local scope completion governs readiness, while device and
memory ownership govern concurrency. Planning/tagging and packed mask evidence
currently use CPU memory; inference uses the GPU.

`YOLO_TTA_SAM_ADAPTIVE_CROP=1` opts into repeated outer-context enlargement and
replay from the same frozen seeds and full intervals. Each complete attempt is
rescanned, and contact with an internal group-crop edge requires further admitted
enlargement. Geometry grows strictly within the declared canvas. Work is recorded
without default per-scope work quotas; live resource and backend bounds
apply. A needed enlargement that cannot run, or a failed attempt, stops the SAM
scope with an explicit error and retained evidence. Internal child-tile clipping
is separately diagnosed under the owned-core/halo rules; resolving
outer context neither certifies full object extent nor permits reseeding a
neighbor. Unchanged
evidence imports copy authenticated encoded packets without decoding/recompressing
the whole inventory. Interpolation applies one global selection pass to the
chosen attempts before extrapolation can consume that local result.

The default policies are whole v6 and tiled v7. `sam_branch_selection`
certifies each requested connection independently using original seed-connected
SAM support and immutable detector anchors. It can retain a successful daughter
when another fails, or join independently seeded prefixes that meet inside the
gap. Certification checks actual gap connectivity and original seed attachment.
Output uses the fixed tracker context, subtracts all
original detections, and preserves radius filtering and unrelated-contact checks.
Context-border contact is recorded as censored object extent rather than erasing
an otherwise certified connection; no pixels beyond the crop are invented.
Packed per-owner masks and observed attachment proofs are shared by policy,
replay, publication and final-survival readers. Final cleanup cannot restore a
removed observed attachment merely to certify survival.

`YOLO_TTA_SAM_TIGHT_CROP_GUARD=0|1` controls the inherited acceptance-corridor
veto. An unset value inherits the resolved policy: off in v6/v7, on in explicit
legacy v2-v5. Launch pinning distinguishes an unset request from an explicit
choice. Saved complete policies do not depend on ambient environment settings.
Whole v2/tiled v3 use strict family selection; whole v4/tiled v5 use a separate
guarded-rescue stage. Replay preserves the saved policy version and run-ID
callback semantics. See [SAM policy controls](docs/sam_interpolation.md).
Tiled selected additions use native owned-core assembly, while
retained full halo unions remain quality evidence. Complete raw
halo, seed, frame, score, and availability records live in the indexed
bundle. Unknown unseeded cores are not successful empty predictions. Fixed
replay reads the saved mode and cannot turn whole generation into tiled evidence.

`sam_bridge_planning` retains original detector slice-component identities and
plans bounded observed families. Each plan fixes context, acceptance, and legacy
write geometry. Context can cross detector tile seams. The selected policy
chooses the legacy write mask or certified support within that fixed context;
both protect original observations. `sam_tracker_runtime` reuses isolated persistent
LTA tracker workers with independent endpoint-seeded sessions. Raw observations
precede tracker confidence filtering and publication cleanup. `sam_evidence`
keeps overlapping masks attributable in an indexed compressed scope bundle.

`sam_policy` resolves proposal quality and selects contributors before their
forward/backward directional unions are published by `sam_interpolation`.
Public directional references carry actually projected source-oriented backing;
native transform metadata alone does not perform that projection. Native
disjoint additions can overlap detector support after source restoration or
low-quality downsampling merges distinct samples.
Source-slab reconciliation runs later and its union-reuse optimization consumes
already selected support. Tracker scores remain separate from detector
confidence. A selected bridge and its eventual survival after source voting
and global filtering are distinct receipt facts.

`sam_integration` plans before image rendering, model construction, and GPU
admission. Empty/exhausted passes create no image cache or predictor. Exact
canonical backing can be aliased; other scopes render only their demanded
frame/rectangle inventory into compact immutable caches. Lazy processing cubes
serve targeted Transverse slabs without full materialization; other view families
reuse one shared processing memmap when their sampler requires it.
Image-cache and rendering budgets remain separate from model memory. Ready
materialized Cartesian sources supply only the native region needed by a crop;
the canonical CPU remap preserves its global phase without a full-volume GPU
upload. Qualified Radial requests upload a conservative source-time slab covering
every demanded radius and native patch tap. Their sampling retains the original
native/logical time dimensions; the slab origin changes source addresses only.
Unsupported bounds keep full-source residency.
Ready materialized uint8 Radial and Spherical sources use bounded canonical CPU
preparation for admitted ephemeral cohorts and retries. Each native tap rectangle
comes from the canonical affine crop with its two-pixel guard. Both synchronous
preparation and parallel jobs render only that rectangle in global native
coordinates. The grayscale ROI contract permits at most one gray level of
rounding difference from the full-plane reference; geometry, frame, crop and
categorical sampling rules stay fixed. Unready,
unsupported or oversized demands follow the GPU admission and CPU fallback
rules; initial interpolation and single-cohort retained caches keep their
provider lifetimes.
Eligible native shell ROIs prepare as independent CPU frame jobs within the
same render-scratch credit and unchanged 256 MiB default. ROI-sized accounting
bounds up to eight workers by CPU count, frame count and scratch after reserving
owner remap space and a transfer-plane margin.
Results arrive in original order; one owner remaps, copies and publishes crops.
Cancellation and joins settle jobs before source ownership returns. Tight
budgets use serial CPU preparation.
On an admitted idle GPU, `sam_gpu_rendering` reuses TTA's projectors
and uint8 affine path for demanded SAM crops. It copies back only requested
pixels. A completed builder can hand its source and the same compute lease to
an already-ready, compatible image request, including a smaller demand covered
by its resident slab. Actual resident bytes stay charged; planned sampling
identity remains fixed. At most two builds share a source
while SDK work is ready; further handoffs require an atomic idle-SDK check.
Each handoff fences and clears view scratch and rechecks headroom. Ready SDK
work receives a turn after the admitted source burst. A clean, known renderer
can skip broad garbage collection and allocator trimming when fresh driver-free
bytes cover the recipient's full
additional workspace and reserve. Low free space or unproven ownership keeps
conservative trimming; ordinary retirement still releases source and scratch.
No source allocation remains after the compute lease returns. Admission occurs
after CPU image credit and duplicate-demand checks; no GPU is held while those
checks wait. Source/grid scratch retires before the compute lease returns, and
predictor-worker shutdown cannot release a parent renderer's borrowed lease.
Ready materialized processing cubes are reused directly. An unfinished streaming
cube retains CPU preparation: its endpoint-aligned temporal grid cannot be
substituted by TTA's center-aligned native-T route. Nonstreaming native-T inputs
use center-aligned geometry.
Unavailable devices or capacity select the CPU route.
`YOLO_TTA_SAM_GPU_IMAGES=0` explicitly selects CPU image construction.
The registered TTA GPU intensity semantics may differ slightly from canonical
OpenCV pixels. Sampler-specific cache and feature identities prevent mixing them
as identical data. GPU identities also include the exact crop demand because
translated GPU grids may round differently at shared crop pixels; partial GPU
donors from another demand cannot supply them. CPU shell image and feature
identities bind the native ROI recipe and demand, so different ROI contexts
cannot claim identical bytes. Identical demands reuse their cache; Cartesian
and other CPU routes retain canonical overlap reuse.
Live matching source/grid/frame-address proofs permit a
sampler change between cohorts or retries; portable per-input sampler records
retain the actual identities while evidence import keeps its pinned scope ID.
The CPU route's eligible lazy Transverse requests batch OpenCV temporal resize
over bounded spatial slabs and scatter only requested frames. Native planes,
outputs and exact remap intermediates share admitted workspace. Materialized,
cached and unsupported routes use their own registered sampler.

Independent image-cache builds run as private transactions on their admitting
parent threads. Registry locks cover lookup, ownership and publication rather
than pixel rendering or worker-retirement waits. Distinct live parent profiles
fund their own concurrent image and render reservations,
while uncredited callers share the aggregate allowance. Identical
demand joins one builder. Donor pins and per-invocation cohort claims prevent
retirement races; cancellation and close settle creators, waiters and consumers
before releasing sources. Per-descriptor limits and spatial coverage remain
enforced; the chosen intensity sampler is recorded explicitly.

Extrapolation plans complete original groups before splitting image demand into
bounded cohorts. The full frozen baseline, one writer, predictor pool and global
retry budgets remain shared. Immutable frame and observation-ID indexes avoid
repeated scans of unrelated observations. Detached RGB input and completed worker/result barriers
precede retirement of a cohort's owned gray backing; borrowed/shared sources are
protected. A planned group that cannot fit alone fails admission explicitly.
Native extrapolation publication authenticates one immutable owner index and
visits each active native frame once for both directions and their union count.
Receipt/evidence checks bracket the transaction; standalone plane helpers
validate independently. Index admission and a bounded fallback retain
valid work when credit is tight. Failed paired publication retracts both stores.

Fullframe selected SAM source projection/publication uses the bounded
component queue. Frozen metadata and immutable CVOL paths cross that boundary;
live model contexts, source arrays and thread-owned profiles do not. Terminal
output waits for every required child reference. Non-tiled parents without
retained debug arrays can return redundant dense credit before those children
finish, but only after actual original/result allocation or mmap owners die.
Scheduler-owned retirement receipts fence aliases and outstanding confidence
credit. Tiles retain their synchronous publication path.

Shared detector/SAM devices require global YOLO task drain and each device's
authenticated detector asset-retirement acknowledgement. Progressive startup
applies only to multiple SAM GPUs all shared with YOLO; an acknowledged device
can then start its cohort while others finish retirement or startup. Mixed
shared/dedicated and single-device fleets use legacy whole-pool startup.
Shared devices wait for the YOLO handoff; fully dedicated SAM fleets and CPU
detector routes can acquire their devices independently.
`sam_parent_staging` prevents a shared-device wait cycle by checkpointing completed
parent inputs before they reserve preparation transients or wait for SAM. Dense
credit returns only after the original mappings close; deferred parents retain
no bridge-ready or terminal milestone. The scheduler resumes them after detector
retirement using its resolved dense limit. During budgeted background drains,
both stager pump calls share the existing time/action allowance, including
prospective memory probes. Dense/startup denials rotate deferred candidates to
the queue tail. Budget checks occur between atomic probes and ownership
transitions; a running probe is not preempted. With physical RAM and dense
credit, multiple eligible parents can own byte-bounded RAM backing and publish
their first disk checkpoint through packed-mask and lossless
confidence-block formats. A completed immutable RAM parent sits outside the
active detector admission window; its original backing lease and RAM charge
remain until the checkpoint writer closes the source. Lease identity is checked
before submission and whenever that backlog is excluded from detector admission.
Original numeric mask values and every confidence byte remain exact. Already
written regular disk owners are reused; unsupported inputs or busy codec credit
use bounded raw streaming. Restore obtains fresh dense credit and falls back to
raw disk backing if RAM is unavailable. Ordinary disk-backed parent retirement
uses the preparation executor so it cannot queue behind RAM codecs.
RAM is claimed at allocation and returned only after owner close;
RAM admission excludes proven disk-only backing from its active commitments.
RAM, mixed, unknown and pending allocations retain their complete promise;
preparation restores that promise before it can replace an input with RAM output.
Logical dense leases remain charged through retirement.
Bank/backlog gauges include the last RAM refusal's inputs and reason. Codec time is part
of checkpoint walltime. Checkpoint workers use the parent concurrency
allocation; each must acquire bounded codec scratch without waiting. New parents
fall back to disk at the RAM threshold and become RAM-eligible again after drain.
The physical guard discounts already-resident owned memfd pages only with fresh
Linux no-swap proof and unchanged allocation identities. Unfaulted promises and
all logical caps remain; missing proof grants no discount. Raw uint8 confidence
checkpoints reuse the compiled ordered-frame encoder within their paid
codec workspace, preserving all scores independently of the detector mask.
Compression starts after the final detector writer, without optional mutating
SAM preparation in front of it.
The shared lazy processing cube can materialize independently once decoded input
and physical headroom permit. SAM planning runs under fresh admission and reuses
matching regular-disk image caches.
Cleanup runs once; confidence capture, tile support and publication retain their
normal order. Dense tiling and bounded policy groups use ordinary preparation.
`sam_resources` reserves additional SAM CPU workspace atomically with the
parent's base reservation, using actual physical/cgroup/SLURM headroom and the
resolved pool limit. Temporary pool or promised-RAM contention waits without
holding partial credit when total capacity and isolated physical headroom can
fund the minimum extra or identified base CPU allowance. Genuine isolated
resource shortages retain the lower declared bounds. Cancellation wakes
admission waits. Lazy family contracts occupy one bounded resident lease that
is reused across families.
Larger contract/topology allowances require a live profile, not serialized
metadata or a pool's emergency oversize lane.
Image builders with distinct live parent credits can run concurrently through
those producer lanes. A parent's image allowance cannot be spent twice;
uncredited callers retain the aggregate cache-plus-render byte limit. There is
no separate two-builder ceiling over independently funded parents.
The SAM parent executor derives its default from the allocated main-process CPU
budget (one parent per sixteen allocated workers), with a minimum feeding target
of two producers per requested tracker slot. View count, CPU budget and the
per-view physical-memory estimate bound that target; live weighted admission
decides which parents can start. Slice workers divide the same CPU budget,
and the explicit parent-worker override remains available. Whole-parent
selection/publication can therefore leave other lanes preparing independent
views. On large SAM hosts, automatic dense and transient defaults each use
40% of one physical/cgroup/SLURM headroom snapshot after a 64 GiB reserve.
The extra transient allowance fits the room below an 80% combined budget after
resolved dense credit; existing floors and explicit limits remain authoritative.
Non-SAM defaults have 256/192 GiB ceilings. Small-host floors, explicit limits and
policy memory clamps apply; policy planning reserves the full SAM
transient pool, including grants beyond each parent's base working set.
Production SAM requests two isolated predictor processes per physical device
(`YOLO_TTA_SAM_SESSIONS_PER_GPU=1` selects one). Each process preserves separate
tracker state, precision and feature ownership. Dual startup enforces per-process
Torch allocator quotas from measured physical free VRAM and checks host headroom
including live parent promises. In progressive startup, device cohorts initialize
concurrently and join one shared tracker without changing their configured
device or worker-slot identities. With fleet startup credit owned, the first admitted cohort permits
parent work; later cohort startup
and failures remain tracked. A cohort resource refusal falls back only after
its attempted workers settle. Valid empty detector metadata keeps startup lazy;
nonempty tile callbacks warm before taking their own parent grant. Live SAM
residency continues to forbid YOLO reuse on the same GPU.
Unfunded model forecasts remain planning metadata and occupy no owned pool debt.
They do not fence dense admission. Direct progressive preparation waits for the
first retirement proof before funding; its host deadline starts after that proof.
The first grant validates the complete forecast. Before progressive cohort
threads start, one host admission funds the full parent transient
envelope, image bound, actual pending checkpoint/source RAM, all model startup
peaks and reserve. Remaining model peaks are charged to the parent ledger;
device claims do not charge them again. Later cohort checks reuse the funded
parent envelope and charge only capacity/oversize growth beyond it, while still
checking current image, pending RAM, remaining models and reserve. Credit stays
owned across retirement/GPU waits and returns only after proven settlement.
Checkpoint debt uses lease RAM commitments minus authenticated resident memfd
pages sampled before fresh physical headroom; unused dense capacity is excluded.
While funded model startup remains owed, RAM births and preparation proof resets
check the full parent capacity plus prospective RAM under the shared condition.
Refusals defer
preparation proof resets for retry without waiving the RAM obligation. Backing
creation can select disk instead; decoding runs outside
the condition. `sam.startup_progress` reports device, stage and budget. Host/GPU
admission waits have bounded per-attempt deadlines; retirement ACK waits remain
separate.
Physical budget eligibility and per-session limits apply; authenticated
logical execution slots fill only the CPU wave already funded. TTA's interpolation
session type has no fixed 30-frame ceiling and retains full requested history.
Resident ownership fences detector/auxiliary reuse separately from one compute
lease per active physical device. Every worker slot must authenticate its own
context's completion before the final ACK allows eligible projection borrowing through
capability and actual free-VRAM checks. Startup, shutdown and failure
quarantine retain ownership until cleanup is proved; retained model/feature
memory is never advertised as free capacity.
On scheduler failure, the original traceback is emitted before executor joins.
Queued preparation is cancelled; dense credit returns only after its actual
input owners retire. Planning failures persist a preparation receipt even before
a plan exists. Wait telemetry exposes parent-pool usage, GPU resident owners and
quarantine reasons, asset retirement, and inference backlog.
LTA keeps its separate 30-frame session contract. A persistent SAM pool scheduler
owns worker submission and completion reception across admitted scopes. It serves
ready scopes round-robin, fills idle physical GPUs before second slots, preserves
each scope's FIFO submission order and treats slot-local crop affinity as a
preference. CPU producers prepare immutable seed/job envelopes
before the scheduler acquires compute credit. A separately funded, bounded bank
allows ready jobs to run while the producer consumes an earlier result.
Factories, live-profile validation and raw decoding stay on the original thread.
Worker input preparation happens after dispatch, under compute ownership.
For the pinned SDK, it fills one admitted CPU float16 clip, using an exact
Pillow/LUT path or per-frame CUDA resizing with uint8 rounding between axes.
The full clip remains CPU-offloaded. A scoped loader adapter preserves native
SDK session initialization; feature identities include the input loader policy.
Queued requests are not additional GPU-resident tracking sessions.
Each scope retains its original SDK wave and decoded-consumer allowance; extra
prepared/packed slots require a nonblocking reservation from the pool.
Without extra headroom the bank is disabled, while independently admitted scopes
can overlap. Uncredited legacy iterators remain exclusive. Pool shutdown
follows scheduler join; outstanding producers, packets and cache users retain
their ownership until settlement. Internal task IDs include a scope token, while
scientific run IDs, lineage, selection, and directional support remain stable;
packed evidence offsets and hashes can reflect execution completion order.
Group evidence encoding can interleave with tracking when its contract, packing
and tiled-assembly scratch fits alongside the SDK CPU wave in already
owned credit. Otherwise generation retains the evidence-before-tracking barrier.
For each decoded result, a live funded tracker permit can lend unused attempt
bytes to bounded mask packing and compression. Helpers receive owned immutable
plane snapshots; lazy masks, descriptor creation and ordered stream publication
stay on the original thread. The shared packing pool uses at most 32 workers
within the coordinator CPU budget, with at most eight jobs per result. The full
SDK/raw-transfer allowance and one held decoded result remain reserved.
Insufficient scratch uses serial encoding;
failure or cancellation joins packing work before releasing its permit.
Consumer queue and hold diagnostics describe ownership and waiting, rather than
CPU cores or active processing time.
After validating frozen extrapolation cohorts, eligible CPU builders start the
current and next image cohort before evidence writing. Their separate image
grants allow preparation to overlap that writing without advancing SDK tracking
past its evidence barrier; cancellation joins both builders before returning credit.
Every group and skipped tile enters the final evidence inventory before
commit, including groups without generated runs.
`lta_feature_cache` reuses exact immutable frame features across sessions under
model/source/transform/precision identity and headroom admission. It accounts
shared tensor storage and never reuses tracker state. `sam_mask_reader` retains
bounded immutable mask/measurement products within one verified evidence
transaction. Owners close on success, cancellation, or infrastructure failure.
Intrinsic readers share packed effective-mask products with the outer reader
inside its cache allowance. Ordinary products borrow unused compact
capacity; pressure causes recomputation without changing filtering or topology.
Family execution provenance comes from an iterator-owned immutable receipt,
not another scope's shared runtime counters. Cache retirement checks the actual
descriptor's active scopes and mapping proofs, allowing unrelated caches to retire
without waiting for the entire tracker pool to become idle.
Multi-cohort extrapolation uses the same bounded producer wrapper for first,
current and next frozen image cohorts; interpolation and extrapolation retry
callbacks use it for eligible ephemeral crops. A separate aggregate 16 GiB
image-staging allowance funds cache payload and render scratch. Physical,
cgroup and SLURM checks reserve the full future parent-pool capacity before
admitting staging work. Parent defaults remain fixed, and emergency parent
reservations are exclusive with image staging. Each funded parent can retain
two wrappers; unavailable staging credit uses synchronous preparation.
Image credit cannot authorize an SDK wave. Cache and worker ownership retain
staging debt through consumption and uncertain retirement; credit returns only
after settlement. A healthy consumed prefetch joins its producer in short slices
without the cancellation deadline. Cancellation, consumer failure and unused
abandonment use a 30-second join budget; expiry retains image/profile ownership.
Telemetry separates image-staging admissions, declines and
credit return from parent grants, including retained debt, and records CPU
image admissions and host preparation time. Shell diagnostics separate sampled
native pixels from full-plane counterparts, native wall/thread CPU time, remap,
future waits and cache flushes, with actual worker and workspace gauges.
Extrapolation groups exact image/crop requests within bounded batches,
subject to the frame-work balance guard and explicit flat backout. Whole-crop
retry contacts use authenticated packed-bit edge counts and full foreground
validation; overlapping tiled halo unions retain dense decoding. Host SAM phase
boundaries use optional task telemetry, including interrupted work.
Ordered branch selection shares reader-owned immutable validated prefixes,
validates each incoming chunk and charges its shallow indexes to live topology
and measurement credit. The full-merge path remains the bounded fallback.
Selection can enlarge its mask cache from 32 MiB to at most 2 GiB using
live phase credit left after actual topology and spool needs. Parallel lanes
retain their 32 MiB ceiling; outer cache storage is deducted before their and
the prefix indexes' admission. Cache capacity does not alter group acceptance or
filtering. Final portable recipes are flattened, fingerprinted and fully validated.
Ordered cross-family contact and topology rules apply to selected support.
The main component gate uses a direct-parent and residual-parent-bridge
whole-component OR.
It receives completed selected parent support. Admitted tile observations are
then consolidated in parent coordinates before SAM plans the tiled scope.
Gate-support fingerprints preserve dependency lineage for later replay.

Fixed-proposal replay requires no inference, but an upstream parent selection
change may invalidate downstream tile admission and planning. Changed inputs
require explicit regeneration or a labelled frozen-evidence comparison. The
current-source inventory detects file-set and byte drift; the enclosing Git
commit or complete-source archive binds that record. Source identity is separate
from workload qualification and measured quality.

The paired crop-strategy tools under `tools/` compare whole-native-crop resizing
with independently seeded overlapping tiles. They preserve matched source and
observation inputs and disable cross-tile propagation in the tile experiment.
Geometry and seed-survival diagnostics precede held-out scoring. These research
tools have separate inventory and targeted checks; their results do not select
production crop backends or defaults.

The swept-context planner bounds complete transported original endpoint
silhouettes, uses the legacy raster origin, and records context/canvas clamping
under `xta.sam_fixed_family_swept_context/2`. The context is fixed before tracking;
it does not grow in response to a predicted track. A complete swept envelope
can exceed configured group or resident-contract budgets. Refusals stay explicit
rather than re-clipping contract space or reporting admitted-only coverage.

Single-worker tiled cohorts drain identical crop queues consecutively to reuse
immutable visual features; endpoint tracker states remain independent. Benchmark
receipts distinguish startup, tracking and bounded assembly work from decoding
and detection. Research context and acceptance-only variants declare their own
geometry and evidence scope; their results do not automatically change production
defaults. Source audits and workload qualification have separate scopes.

TTA can save the final source-volume semantic mask as grayscale PNGs with
`0` background and `1` foreground. This differs from `--save images`, which
saves rendered model inputs. NRRD layers and their manifests preserve view,
model, geometry, and policy identities. `tta_outputs` publishes a complete
run manifest only after model workers, projection, finalization, and all output
futures settle.

### TTA resource and failure boundaries

A parent canvas holds its dense memory credit through assembly and projection
until its immutable component backing is published. File-mode results and
support sidecars have separate accounting. The scheduler can lend an idle
worker GPU to eligible Radial or Spherical projection after queued inference
drains; CPU projection can proceed while admission waits. A projection may
switch at the first unpublished source slice. A failed preflight leaves CPU
progress intact. A failure after CUDA publication starts aborts that layer
rather than replaying partial output.
The exact compiled Spherical CPU pull is required for CPU projection and is
independent of approximate GPU geometry controls. The Radial CUDA owner packs
eligible bitsets on the device before transferring their bounded slice payloads;
it skips the full-bitset download for a proven-empty output. Both Radial packing
flags, `YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION` and
`YOLO_TTA_PACKED_OWNER_PUBLICATION`, allow the legacy host export path for
diagnosis. D1 confidence retirement can mask a uint8 score crop on the device
before one host transfer when the crop has at least 16,384 pixels. Set
`YOLO_TTA_D1_GPU_MASK_MIN_PIXELS=0` to request that path for every CUDA crop,
or `YOLO_TTA_D1_GPU_MASK_CONFIDENCE=0` to select the two-crop host path.

Completed view layers are reduced by a single source-union writer. A dense
handoff credit bounds waiting source-sized volumes. Final sink joins precede
the complete manifest. TTA first writes an `in_progress` manifest and replaces
it only after successful completion. Reconciliation cannot consume unfinished
confidence publication.

## PTA: deterministic dataset construction

`pta` owns candidate membership, augmentation versions, split assignment, and
output identities before asynchronous rendering begins. `pta_classification`
tests foreground occupancy for background sampling, using native categorical
planes and cached full/tile index maps where possible. If semantic output keeps
every candidate (`--background_percent 1`) and has no offline copies, it skips
the costly all-view classification while retaining missing-label eligibility
and a foreground-preservation check. The summary marks unavailable class totals
as unmeasured. `pta_scheduler` packs retained candidates into source-frame work
so one decoded or resident volume serves many neighboring outputs.

`--task segment|semantic` selects YOLO task semantics for PTA. `--save labels`
continues the polygon-backed path. `--save semantic` writes class-index PNGs
under split-local `masks/` directories, paired by stem with images:
`0` background, `1` foreground, `255` ignore. The generated dataset YAML uses
`nc: 1`, `names: ['0']`, and `masks_dir: masks`. A missing YOLO label file is
unknown coverage in forced partial-label mode; an existing empty file is known
background. Foreground and coverage travel as separate categorical volumes
through geometry and paired augmentation; only publication combines them into
the 0/1/255 plane. A custom GPU policy used for partial semantic labels must
declare `mask_independent_geometry = True` before the same sampled transform
can be replayed for coverage. The bundled GPU policies declare this contract.
`--save binary` writes retained exact binary masks and grouped lossless videos
through `pta_binary`.

CPU rendering uses persistent workers and shared volume memory. Offline GPU
augmentation uses one persistent process per selected GPU. `pta_rendering`
prepares view plans; `pta_workers` renders and applies policies. GPU owners keep
foreground and coverage volumes resident when VRAM admission succeeds.
`pta_cuda_masks` coordinates their lifetime; `pta_cuda_cartesian`,
`pta_cuda_azimuthal`, and `pta_cuda_shells` project categorical values directly
to the final full or tile raster. Intensity and categorical sampling remain
separate. GPU projection events fence policy reads and source retirement.
Unsupported geometry or insufficient prelaunch VRAM selects the CPU categorical
reference path; a launched CUDA failure propagates. The manifest records which
backend actually rendered categorical items.

`pta_batch_pipeline` and `pta_gpu_publication` overlap bounded GPU policy
batches with encoding and host publication. Output tensors are snapshotted
before a policy can reuse buffers. `pta_publication` writes and verifies
image and label files; the nvJPEG batch uses staged files and atomic renames,
while ordinary image, text-label, and semantic PNG writers write their final
paths directly. `nvjpeg` and `nvtiff` select explicit GPU image encoders;
CPU formats use their requested encoders. When semantic output is requested
without polygon or binary output, foreground and coverage combine on GPU
before one class-index download. With those other outputs, publication also
downloads the separate planes and combines semantic IDs on the host. PNG
compression and filesystem writes use bounded host workers. A batch holds its
queue reservations until all image, semantic-mask, and label writes settle;
failures join admitted work before the worker reports failure.

PTA validates output ownership before cleaning an existing generated dataset.
It establishes ownership and atomically replaces any prior completion with an
`in_progress` manifest before deleting generated artifacts. Cleanup preserves
that marker; failed invalidation prevents deletion. An interrupted restart must
not leave a stale complete manifest referring to removed images.
Requested and effective image formats are recorded independently. Its complete
manifest is written last, after every file and worker closes. Unforced partial
volumes and sequences with encoded-frame gaps retain native depth and restrict
labels to Transverse full frames or tiles. Forced partial volumes without
encoded-frame gaps can use 3-D views, including supported shells, with unknown
semantic coverage preserved through projection. Fully labeled contiguous data
can use all supported views.

## LTA: authoritative seeds and bounded propagation

LTA takes a target volume, aligned exemplar masks, and a verified local SAM
bundle. Production uses native Transverse, angle-zero overlapping tiles.
`lta_inputs` discovers and validates the target and exemplar identities.
`lta_runtime` plans views and tile grids. `lta_scheduler` assigns one physical
view owner for its render cache and backprojection; idle devices may help with
unopened SAM sessions. `lta_workers` owns persistent GPU processes and each
process's model/tracker state. A live session stays on one worker. Results
commit in plan order even when execution completes out of order.

`lta_propagation` and `lta_windows` advance authoritative anchors through
bounded temporal windows, with independent backward and forward branches.
Sessions admit at most 30 frames and 128 objects. Authoritative foreground
is never replaced by a tracker prediction. `lta_tiles` and
`lta_tile_tracking` plan overlaps and spatial relays; `lta_relay_episodes`,
`lta_frontier`, and `lta_frontier_execution` settle new support over bounded
temporal waves. Relay identity includes lineage, tile, prompt frame, and
direction. Complete same-event masks merge before coverage checks. A subset
already visited in the same lineage and direction can hand off; new support
remains eligible. The finite growth guard fails publication if relays have not
settled.

Workers return sparse cropped, row-packed foreground and coverage artifacts.
The coordinator verifies each packet, ORs its indexed crops into the private
view, and releases it before admitting more work. `lta_coverage` stores the
ephemeral visited-transition ledger; `lta_union_artifacts` owns the sparse
view union. Once a physical view is complete, it receives final two-dimensional
hole fill and one backprojection into the native union. `lta_postprocessing`
writes requested filter checkpoints; exact authoritative foreground is
restored in the final output after destructive filters. `lta_outputs` publishes
the mandatory final NRRD and complete manifest after checkpoint and identity
checks. A checkpoint can remain after a later failure; only a complete manifest
marks run success.

`lta_sam` resolves and audits the local SAM 3.1 assets; `lta_worker_adapter`
owns model construction and tracker calls. The production feature path in
`lta_tracker_features` skips unused grounding detection while retaining the
tracker's expected visual features. GPU workers divide the visible CPU
allocation and record their process/device identities. Increasing
`--lta_workers_per_gpu` creates independent model contexts, so model memory and
CUDA scheduling scale with that count. LTA does not use YOLO's `--task` flag.
Cross-anchor identity matching remains a diagnostic facility in
`lta_tracklets`; production merges independent anchor chains by recall union.

LTA can opt into `--lta_crop_backend dynamic` for native Transverse at angle
zero. `lta_dynamic_crops` plans full-seed native rectangles and explicit 1008
model transforms. `lta_dynamic_execution` advances sealed predecessor batches
through fixed per-window contexts, with bounded split/interior-edge patches.
Crops follow objects across tile boundaries, while deterministic batch
membership preserves arrival-order independence. The default tiled path uses
spatial relays. Dynamic crops require native dimensions of at least 1008,
reject an oversized seed instead of clipping it, and record exhausted patch
and split bounds. See [dynamic LTA controls](docs/lta_dynamic_crops.md).

SAM import and construction run in the isolated model worker. XTA restores the
caller's CUDA-matmul and cuDNN TF32 settings after those operations, including
failure paths. SAM's model-process BF16 autocast behavior remains confined to
that worker; a predictor is not constructed inside a shared TTA/PTA process.

## Publication, diagnostics, and validation

Large source and mask arrays use explicit ownership: shared mappings, worker
descriptors, bounded RAM credits, or path-backed artifacts. Native shell
payloads can use parent-owned Linux memfds with spill to disk under pressure.
Immutable component stores may outlive a dense canvas, but each retained layer
consumes filesystem capacity. Scratch placement can therefore affect
RAM usage when its filesystem is memory backed.

For YOLO result masks/confidence, a worker can open parent memfds directly after
its source ACK proves procfs access and exact storage identity. Capability
is bound to the current parent/worker PIDs and invalidated on worker replacement.
Every output open checks device/inode/size before prediction; those descriptors
remain task-local and close after publication. Unsupported workers use descriptor
transfer. Worker telemetry separates direct opens from descriptor detaches.

Retiring a shared NumPy mapping does not invalidate live views or independent
arrays backed by the same mmap. The mapping's last reference governs unmapping
and memfd-owner release. Named scratch deletion follows that lifetime and checks
file identity before unlinking and skips an observed replacement. Callers retain
ownership of their unique scratch names until retirement completes; borrowed input
paths and keep-temp artifacts are not deletion targets. Windows
deletion may briefly wait for the mapping destructor. Owners must release their
references explicitly; the retirement wait helper can verify cleanup and reports
consumers that still retain the mapping. Forked workers reset inherited
retirement queues and cannot delete their parent's pending scratch files.
When successful view preparation replaces its input union allocation, the
original scratch backing enters this same deferred retirement path. Returned
aliases retain it; independent sealed publication stores do not. The input
pathname's identity is captured before preparation and checked again at handoff.

`outputs` and `nrrd_spans` stream native NRRD rows and crops; publication
records durability and releases source references after completion. Optional
native codecs and hardware copy/compression helpers live in `native/`,
`intel_compression`, and `intel_dsa`. Encoder/backend admission is explicit:
a requested unavailable accelerator fails or uses the documented supported
fallback before an output starts. Partial artifacts and worker failures never
produce a complete run manifest.

TTA writes retained SAM interpolation and extrapolation artifacts into
`OUTPUT/sam-artifacts.tar` while the run executes. `artifact_archive` serializes
writers through `sam-artifacts.tar.lock` and commits sealed evidence bundles,
CVOL stores, and JSON receipts as independent TAR/PAX transactions. Long scope
names remain logical members under `sam_interpolation/` and `sam_extrapolation/`;
published references use `sam-artifacts.tar#/MEMBER/PATH`. Active builders use
short temporary stages and retire successful stages after commit. Public NRRDs,
telemetry and ordinary run manifests use their separate filesystem paths.

SAM evidence and CVOL readers accept both archive references and legacy
directories. Private seekable streams bound reads to one committed member;
scientific schemas, mask ownership and payload checksums remain intact. An
interrupted final append does not hide earlier committed transactions, and a
subsequent writer discards the uncommitted tail before appending. A readable
container does not certify scope or run completion; required selection receipts
and complete manifests retain their own checks.

Run, LTA, NRRD-sidecar and confidence JSON writers share `json_publication`.
They reject non-finite values. Filesystem destinations sync a unique sibling
stage, replace the destination, and sync its directory where supported. SAM
archive destinations append a committed member transaction. Failures are not
acknowledged as successful publication. Same-destination writes serialize;
independent filesystem destinations can sync concurrently. Forked workers reset
inherited publication locks.

Runtime diagnostics separate queue waits, rendering, inference, projection,
encoding, publication, and worker retirement. These are overlapping stage
timings, so summing them does not recover wall time. Validation tools under
`tools/` check geometry, model backends, semantic decoding, PTA categorical
projection, and LTA propagation. Tests live under `tests/`. Benchmark receipts
and full workload evidence belong outside the repository in
`Scratch/Data/XTA/History` and task-specific Scratch directories.

`run_transport` optionally packages a stopped run's diagnostics and scientific
evidence into one verified ZIP64 envelope, with public NRRDs separate by default. It preserves
the original bytes and schemas, distinguishes transport integrity from run
completion, and inventories the external NRRDs for transfer checks. Telemetry
readers can stream ZIP members. The live SAM container is independently readable
by scientific APIs without extraction; a container inside an outer transport ZIP
is first restored as a file.
Extraction uses bounded paths, refuses overwrites and unsafe members, and handles
long Windows paths. Layer NRRD basenames are capped at 120 characters while full
labels, colors and provenance remain in headers/manifests. See
[run transfer](docs/run_transport.md) for packing, verification and recovery.

`prepare_reconciliation_release.py` writes the current-source inventory from the
reviewed source tree. The inventory records all maintained files.
NUL-free UTF-8 text hashes normalize CRLF to LF for checkout portability, while
raw before/after checks detect byte changes during verification. Binary
bytes stay exact. The inventory excludes its own digest; Git and the complete
source manifest cover that file too.

Release qualification checks numbering before GPU work, then runs the full suite
from the repository root before inventory verification and source-bundle construction:

```powershell
python -B tools/prepare_reconciliation_release.py --output-dir ../Scratch/Releases/REVIEW_NAME/inventory --write
python -B tools/qualify_release.py --output-dir ../Scratch/Releases/REVIEW_NAME/qualification
```

The default requires a clean Git checkout and bundles committed Git bytes.
`--snapshot` explicitly validates an uncommitted development tree and labels its
bundle accordingly. Each step must pass without changing the source tree;
the receipt and logs stay in the requested Scratch directory. Tests establish
Ultralytics settings outside the checkout and check for leaked dependency stubs
and TF32 settings. CUDA tests share the workspace's atomic `Scratch/Temp/GPU_LOCK`.
The gate requires an available CUDA device and successful execution of the
CUDA geometry/policy cases that check TF32 isolation, plus independent
Cartesian, upright Azimuthal, Radial and Spherical native-coverage oracles.
Required groups are listed in `tools/qualify_release.py`; missing, skipped or
failed cases cannot qualify, and receipts retain their executed case names.
Guarded tilted routes use their CPU checks. `--snapshot --cpu-only`
records explicitly reduced coverage and does not qualify a release.
