# SAM interpolation and extrapolation in TTA

`--interpolation_backend sdf|sam` selects interpolation, with `sdf` as the default.
Choose one interpolation backend for a run. `sam` uses the local mask-conditioned LTA tracker
to propose image-guided additions between detector observations. A rejected SAM
proposal receives no SDF fallback. `both` is rejected during argument parsing.

SAM extrapolation is independently enabled by `--extrapolation_distance` and
can follow either interpolation backend. Source review and snapshot qualification
are separate from a tagged release; see [the development audit workflow](../../Scratch/Data/XTA/History/release/README.md).

The current development SAM path supports all existing TTA view families:
Transverse, Sagittal and Coronal, their tilted variants, Azimuthal views,
Radial shells, and Spherical patches. This applies to full-frame and consolidated
tile observations. Detector angle augmentation is inverted before accumulation,
so SAM uses the canonical angle-zero canvas and retains the original detector
angle as provenance. LTA remains Transverse-only; sharing its tracker does not
extend LTA's supported views. SDF retains its existing iterative passes and layer
decomposition. Interpolation is disabled by default (`--interpolation_distance 0`);
set a positive distance to enable it. SAM assets
are initialized only when SAM interpolation or extrapolation is active.

Here, a native view means the active detector/working-view canvas. It
includes TTA's existing processing-volume transform, including any stack
resampling performed during canonical volume preparation. Its frame index need
not equal a raw input-video slice index. SAM renders and tracks that exact
detector canvas, as SDF bridges do; scope metadata preserves native processing
frame addresses and the `native_transform` back to the source grid.

Only Azimuthal frame order wraps. At the half-turn seam, aliased frames reflect
the column coordinate. The compact `xta.sam_cyclic_view_frames/1` recipe retains
unfolded addresses for generation and local selection; directional publication
folds them back to native frame addresses before the existing TTA categorical
source projector runs. Cartesian and tilted-Cartesian stacks, and Radial or
Spherical radius-frame order, remain clamped. Spherical dynamic radius bands
retain separate patch trajectories, so channels and interpolation cannot cross
into another band's radius spacing or ownership, even if its angular grid matches. There is no new fusion across radial arcs
or spherical patches. These routes have focused CPU geometry tests;
they do not constitute full GPU or cluster qualification for every orientation.

## SAM attention precision

The automatic H100 profile selects the pinned Meta SAM 3.1 FA3 route. Attention
Q/K/V use FP8 E4M3 and attention outputs return to BF16 runtime compute. Stored
checkpoint weights remain FP32. This is attention precision selection, not an
FP8 checkpoint or replacement model. The profile retains eager execution,
complex-valued RoPE, the original input grid and complete tracking intervals.

`YOLO_TTA_SAM_ATTENTION_BACKEND` accepts `auto`, `sdpa`, or `fa3_fp8`. `auto`
uses FA3 on the H100 profile and SDPA on the supported 4090 profile. `sdpa`
provides the baseline for matched experiments. Explicit FP8 FA3 requires
Hopper or newer. The worker configuration's `attention_backend` field takes
precedence over the environment when supplied.

Each FA3 worker checks the official `flash-attn-3==3.0.0` runtime and executes
head-64 self-attention and head-32 cross-attention probes before model loading.
The checks cover the imported interface, CUDA execution, output shape, dtype and
finite known-value results. After construction, the worker audits the image
encoder and multiplex tracker's attention routing. Its profile records the
backend, runtime version, imported implementation hashes and probe results.
An import alone does not establish compatibility. A failed FA3 check is an
initialization error; it does not silently substitute another backend.

Runtime probes establish execution compatibility. They do not establish mask
quality or H100 throughput. A matched complete-track replay compares the
original seeds, crops, frames and outputs under SDPA and FP8 FA3. Differences
in masks and scores are measured rather than rejected solely for being
nonidentical. Quality assessment also needs representative annotated cases and
the downstream branch-selection results.

## SAM extrapolation

Extrapolation propagates a single initial mask outward from a remaining terminal
after local interpolation. It has no opposite endpoint and no backend selector.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--extrapolation_distance` | `0` | Maximum additional native frames beyond the terminal; zero disables extrapolation |
| `--extrapolation_walk_back` | `1` | Additional inward observations used as independent original-seeded runs; zero uses only the terminal |
| `--extrapolation_min_radius` | `3` | Skip a terminal whose maximum inscribed radius is at or below this value; never filter predicted tails |

Radius is measured in the current working canvas, with background padding around
the terminal mask. Thin walk-back seeds remain eligible when their terminal passes.
Walk-back history does not consume the outward distance. Each independent run
receives exactly one original seed; subsequent observations are not injected.

The pipeline freezes the cleaned detector masks plus accepted interpolation
support for the local native scope before finding terminals. A terminal is a
component with no continuation in the outward adjacent slice, including an early
ending daughter. Already connected interpolation endpoints do not create tails.
All seeds and plans are frozen before any tail is merged, so extrapolated masks
cannot generate further extrapolation in the same pass.

Propagation ends at the first **raw SAM-empty mask** or the effective distance
horizon. It continues through unrelated observed objects. Their detector masks,
bridges, or combined masks do not replace the SAM prediction, become new seeds,
or terminate the run. Existing baseline pixels are subtracted only when publishing
new support. A fully overlapped but nonempty prediction can therefore lead to new
support in later slices. Scores, object-removal bookkeeping, radius, crop contact,
area change, and overlap between consecutive masks do not add stopping rules.
Missing or corrupt tracker evidence remains an infrastructure error.

This subtraction and the added-voxel count refer to the native working canvas.
Source restoration or low-resolution export can map a tail and a baseline pixel
to the same source voxel, as with interpolation. A separate exported tail layer
can therefore overlap source predictions even though its native additions were
disjoint; the final union still combines those supports normally.

Noncyclic views stop at the available native stack boundary. Azimuthal tails use
the same mirrored half-turn addressing as interpolation and emit at most one
unique native period excluding the terminal (`N - 1` frames). Receipts record
requested and period-limited horizons; each run's planned output frames also
record clipping at the available stack boundary. Walk-back frames remain separate from this
emitted-tail limit, including when their native address is revisited later at a
different unfolded phase.

The current tracker generates the declared interval, then selection retains its
empty-limited prefix. `generation_early_stop=False` makes this explicit: an early
empty prediction limits output but does not yet save inference on later planned
frames. Tiled runs test the retained raw halo union for emptiness and publish only
attributable owned-core pixels. An unseeded tile is unknown coverage, not an empty
SAM prediction.

Evidence is stored under `OUTPUT/sam_extrapolation` with purpose
`sam_extrapolation` and source stage `post_interpolation`. Directional output
layers use the separate `extrapolation` role and record the terminal, seed,
distance, radius gate, and selection identity. These are one-seed tails rather
than certified two-endpoint bridges. Built-in union includes them. Custom weighted
reconciliation exposes separate extrapolation support, defaults its weight to
`0.35`, and inherits the configured bridge weight for legacy three-role mappings;
an explicit extrapolation weight of zero opts out.

For example, add SAM extrapolation after SDF interpolation:

```text
python -m XTA --mode tta --input /data/volume.mkv --output /data/tails \
  --device 0 --model gpu:/models/detector.engine sam:/models/sam_bundle \
  --enable_cartesian transverse --interpolation_backend sdf --interpolation_distance 15 \
  --extrapolation_distance 5 --extrapolation_walk_back 1 --extrapolation_min_radius 3
```

The native processing order is projection, detector prediction, filtering,
interpolation, extrapolation, then backprojection. Readiness is local: a completed
full-frame view/angle or consolidated tile configuration may extrapolate while
interpolation continues in an independent scope. All interpolation able to change
that scope's baseline, including crop retries, must finish first. Device ownership
and memory admission still constrain overlap.

For parents using asynchronous layer projection, admission to the bounded output
queue happens after SAM extrapolation returns. At most two immutable interpolation
layer descriptors wait locally; they are submitted before the extrapolation layers
to preserve output order. A full projection queue therefore cannot prevent that
parent from starting extrapolation. Publication after tracking remains bounded.
The `projection_admission` SAM phase records time waiting for queue admission;
it ends when a future is returned, not when the projection finishes.

The component projection queue allows up to four active projections and eight
pending jobs, independent of `YOLO_TTA_PARENT_POSTPROCESS_WORKERS`. Each bounded
pending job can seek admission without taking an active projection slot. The
oldest job starts first when it fits; at most two younger fitting jobs may pass
a job waiting for memory before it receives priority as active work drains.
Its working-byte ceiling is
recomputed at admission from one eighth of current physical/cgroup/SLURM memory
headroom, with the existing 4 GiB minimum; swap grants no credit. This replaces
the fixed ceiling and lets waiting work notice memory released by other stages.
Falling headroom restricts new overlap without revoking active reservations.
An oversized job still runs alone until it finishes, even if the budget grows.
The allowance is accounting credit, not an immediate allocation or an RSS
measurement. Startup prints the initial allowance. The existing telemetry sampler
records live `projection.component_queue` occupancy, working/source credit,
admission waiters and backfill counts, including in partial captures. No
environment-variable override is needed.

Selected upright Spherical SAM masks use the established CUDA-capable projection
route automatically when CUDA is available. This restores a temporary native
uint8 input and retains the existing GPU admission, compact output, preflight,
and CPU-to-CUDA promotion behavior. An unset GPU flag no longer pins these
publications to the CPU after tracker work releases the GPUs.
When CUDA is disabled or unavailable, masks whose stored bounding boxes leave
unused native canvas use an encoded CPU input adapter. It reads
raw or packed mask bits through the existing geometry kernels and streams bounded output
planes, without restoring a full native uint8 volume. Existing geometric bounds
skip proven-empty planes where supported. All Radial orientations, including
tilted variants, use the established CPU/GPU projection route: production-sized
checks found the encoded implementation failed to scale across workers. Tilted
Spherical views also retain the established route pending equivalent performance
qualification. Encoded Radial remains available for direct qualification only.
Full bounding-box coverage or an explicit `YOLO_TTA_GPU_SPHERICAL_BACKPROJECT=1`
also selects the established Spherical route, preserving the explicit flag's
previous behavior. `YOLO_TTA_GPU_SPHERICAL_BACKPROJECT=0` selects CPU processing.
Existing family GPU flags still
control GPU admission on the restored routes. The ordinary per-job
reservation remains conservative enough for fallback. Only CPU capability or
workspace refusal before publication can fall back; malformed stores and
runtime failures remain errors. The
`projection.sam_encoded_shell.native_decode_bytes_avoided` counter records the
logical native restore omitted, not measured RAM savings.

SAM interpolation and extrapolation share the persistent predictor and bounded
feature cache; reuse requires identical image/crop geometry. The current terminal
labeling and planning, image preparation, and packed mask evidence use CPU memory.
Tracker inference uses the GPU. The entire chain is therefore not yet resident on
device, even when detector inference and backprojection use accelerated paths.

When both stages run for a parent, they also share a bounded inventory of slice
components. Unchanged binary or single-label slices reuse their immutable component
crops, including observations without an interpolation candidate. Changed slices
and slices with multiple labels are discovered again. Each stage still computes
its own topology, terminals, seed identities and acceptance decisions from the
current mask. This preserves the existing plans; it does not copy tracker output
or retain recurrent SAM state. The inventory is cleared after extrapolation and
on failure. Returned `sam_observation_inventory` statistics count discovery and
reuse without changing scientific identities.

Component crops of at least 1,024 pixels retain lossless packed bits; smaller
crops retain immutable Boolean arrays. The inventory charges the stored bytes
plus its existing per-observation metadata allowance against the unchanged
512 MiB ceiling. Consumers decode a crop when needed. Pixel values, component
identities, cyclic mirrors and original seeds are unchanged; no resizing or
component removal is involved. Area and centroid reuse immutable scalar results.

## Experimental SAM tracking crop mode

`YOLO_TTA_SAM_CROP_MODE` selects `whole` (default) or `tiled` for active SAM
tracking. It is validated before heavy startup and pinned for that launch.
Values are trimmed and case-insensitive; other active values fail clearly.
Runs with neither SAM interpolation nor extrapolation active do not read this
unused environment setting; their recorded mode is null and no SAM crop resources start. This is an
experimental environment control with no CLI counterpart or 1260-pixel mode.

For example, enable the tiled experiment in a PowerShell session before using
the SAM invocation shown below:

```powershell
$env:YOLO_TTA_SAM_CROP_MODE = 'tiled'
```

Both modes retain the existing family planner, context rectangle, and detector
working canvas. `whole` sends that native working rectangle to the SDK, whose
image preprocessing stretches it to 1008 by 1008. `tiled` partitions the same
rectangle into overlapping working-canvas footprints capped at 1008 per side,
with a 128-pixel halo and fixed midpoint ownership. It seeds each independent
tile session from original endpoint intersections; predictions are not handed
to neighboring tiles or used as new seeds.
A footprint with no original seed is not tracked and remains unavailable
spatial coverage. Missing tile coverage is not a successful empty prediction.
Contexts that fit one footprint use a single tile; a reduced working canvas may
therefore exercise no oversized tiling even when the original images are large.
The tiled inventory is bounded to 256 tiles per original run and 20,000
scope jobs/footprints, with a 32 GiB logical assembly cap. Cohorts drain at most
16 partial parent assemblies, spilling bounded native support to owned temporary
files that retire on completion or cancellation. These bounds do not permit
longer tracker sessions or a looser proposal policy.

Tile size is measured on the current working canvas. Existing input/cube
resampling and detector canvas reduction still apply, so `tiled` does not imply
one original-camera pixel per model pixel. Model mask conditioning uses its
separate 1152/288 grids. Complete raw tile halos, object scores, frame status,
and parent/tile identities remain attributable evidence; ownership governs
stitching and additions, rather than erasing raw halo observations.
The assembled parent run contains fixed owned-core support. Its aggregate
tracker probability is undefined; retain and inspect the individual child
scores instead of inventing one probability or detector confidence.

The interpolation defaults are whole-policy v6 and tiled-policy v7, which qualify and
publish connected branches independently. Explicit whole v2 and tiled v3 retain
strict legacy behavior; whole v4 and tiled v5 add the historical guarded-rescue
stage described below. Endpoint, family, and topology decisions use filtered owned-core
support. A separately filtered full union of that original seed's tile halos
remains quality evidence, never output support; discarded halos cannot hide
spill from either stage. Whole-policy versions cannot select tiled evidence.

Changing this mode changes the generation attempt. Fixed-evidence policy replay
does not generate the other mode's predictions. The mode adds no different
interpolation backend, quality threshold, detector tile-admission rule, or
outer-crop planner. A comparison of the modes must use their recorded geometry
and input snapshots.

The existing `YOLO_TTA_DELAY_NATIVE_EXPANSION=0` explicitly selects a full
native-view mask canvas instead of delayed mask expansion, which can increase
memory and runtime cost. It does not disable processing-cube resampling or
change the family's outer context planner. Scope receipts record the crop mode,
1008 tile side, 128 halo, working/native shapes and canvas kind, and the delayed
expansion setting at launch.
Run totals distinguish `sam_oversized_group_count`,
`sam_multi_tile_group_count`, `sam_tiled_child_job_count`, and
`sam_tiled_skipped_empty_seed_tile_count`; inspect these to verify which geometry
the selected working canvas actually exercised.

## Models and devices

TTA model tokens have a role tag followed by a path: `cpu:PATH`, `gpu:PATH`, or
`sam:PATH`. CPU and GPU entries remain detector artifacts. A SAM entry names a
local SAM bundle; it neither creates detector observations nor selects a GPU.
At least one usable detector source is required. Tags must be unique. Only the
leading tag is split, so quote a path containing spaces and preserve its drive
colon or any later colons. LTA keeps its existing untagged model grammar.

`--sam_device` selects logical CUDA indexes independently of detector devices.
It accepts comma-separated or whitespace-separated indexes, including `cuda:N`
and `gpu:N`. For example, `--sam_device 0,2` and `--sam_device 0 2` select the
same ordered device pool. Active SAM defaults to the detector CUDA pool when
the flag is absent. A CPU-only detector must specify a SAM CUDA device.

```text
python -m XTA --mode tta --input /data/volume.mkv --output /data/sam-run \
  --device 0 --model gpu:/models/detector.engine sam:/models/sam_bundle \
  --enable_cartesian transverse --angle 0 --interpolation_backend sam \
  --interpolation_distance 15 --save nrrd summary

python -m XTA --mode tta --input /data/volume.mkv --output /data/cpu-sam-run \
  --device cpu --sam_device 0 \
  --model cpu:/models/openvino sam:/models/sam_bundle \
  --enable_cartesian transverse --angle 0 --interpolation_backend sam \
  --interpolation_distance 15 --save nrrd summary
```

These are argument examples; substitute verified detector and SAM assets and
the appropriate detector channel format for the actual workload. CPU detector
selection does not require a `gpu:` detector artifact to run SAM on CUDA.

SAM plans the observed-anchor jobs before rendering images, acquiring GPUs, or
loading a predictor. Empty or exhausted plans perform none of those operations.
When SAM and detector devices overlap, admission requires global YOLO task drain
and each physical device's authenticated asset-retirement acknowledgement.
Progressive startup applies only to multiple SAM GPUs all shared with YOLO. In
that mode, an acknowledged device can begin its cohort before other devices
complete retirement or startup. Mixed shared/dedicated fleets retain legacy
whole-pool startup after the YOLO handoff. Fully dedicated and single-device
fleets also use legacy startup; dedicated SAM pools and CPU detector routes can
acquire devices independently. Failed retirement or
unsettled model work must not advertise a shared device as available.

After startup, a resident-owner guard keeps predictors and feature caches
loaded while fencing detector and auxiliary inference reuse. One physical GPU
compute lease covers every active worker slot on that device. Each worker
acknowledges completion of its own CUDA context after session cleanup; only the
last authenticated completion can return the physical compute lease. Eligible
Spherical, Radial and Tilted-Azimuthal projection stages may borrow an idle
resident device through their existing capability and live-VRAM checks. This
does not enable a projection backend whose geometry contract refuses CUDA.
Startup, shutdown and cancellation retain or quarantine ownership until cleanup
is proved; an idle model is never treated as free memory.

A persistent pool scheduler owns worker submission and result reception across
admitted scopes. It verifies each exact completion manifest and CUDA proof before
returning compute credit. A paused CPU consumer or a one-job retry does not exclude
another ready, admitted scope from idle workers. Internal task IDs are namespaced
by scope; original scientific run IDs, seeds, crops, frames and result indices are
preserved and validated on return.

The original producer thread prepares requests and decodes raw masks. Ready job
envelopes can be staged ahead in a bounded bank, letting the scheduler refill while
that thread consumes a result. Each scope keeps its existing SDK-wave limit and
decoded-consumer margin. Additional seed/packed-result/metadata slots require a
separate nonblocking reservation from the same parent pool and available physical
headroom. If no extra bank can be funded, the original bounded window remains
usable across independently admitted scopes. Saved metadata cannot grant credit;
legacy callers without live scope admission retain exclusive execution.

Shared-device parent preparation uses a bounded mixed wave rather than an
unbounded executor backlog. Curved parents start first, while up to two lanes
and matching dense/transient capacity remain available for fitting Cartesian
and Tilted Cartesian parents. One selected parent inventory governs all of these
reservations. A parent that cannot obtain its dense backing cannot block curved
transient admission with a protection token. Late parents with existing dense
ownership join this wave without checkpointing or being charged again. No
all-curved-finished barrier prevents ready work from using an available GPU.
Within the expensive-parent class, one size-priority admission alternates with
three admissions in original arrival order. The size proxy uses recorded native
input bytes. Fit checks, protected cheap lanes and the configured worker cap
remain authoritative. Failed probes preserve scan progress across short pump
budgets. See [parent preparation order](sam_parent_prepare_order.md) for the
fairness rules and bounded admission diagnostics.

Multi-cohort extrapolation admits eligible first/current image cohorts through
the same producer wrapper as next-cohort prefetch. Interpolation and extrapolation
retry callbacks use that wrapper for ephemeral crops. At most two wrappers per
original funded parent are retained, including a current and next builder; one
slow image build does not impose a global prefetch limit on other parents.
Once frozen extrapolation cohorts pass validation, the current and next eligible
image builders start before evidence writing. Their image-only grants allow
this overlap even when group evidence must finish before SDK tracking. Later
cohorts retain one-cohort lookahead; failure or cancellation closes both pending
builders before releasing their ownership.

Production SAM parents declare a mandatory image phase of twice
`YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES` plus `YOLO_TTA_SAM_RENDER_MAX_BYTES`:
the current payload, one inline retry payload and rendering scratch. Before
preparation, parent admission atomically reserves the shortfall beyond its
identified image allowance. Future-parent estimates and pending-parent protection
include that charge once. The charge is separate from SDK, observation and
persistent-metadata partitions; aggregate memory limits remain unchanged.
Contention waits without partial credit. An intrinsically unaffordable mandatory
phase raises `SamMandatoryImageAdmissionError` before preparation rather than
using the emergency clamp. Direct callers without a live parent profile retain
their existing declared limits.

The separate aggregate 16 GiB image-staging allowance funds optional current/next
cohort overlap and eligible retry producers. Fresh physical/cgroup/SLURM headroom
must cover the full future parent-pool capacity plus outstanding staging debt
and the requested grant. Emergency parent reservations and image staging remain
mutually exclusive. If optional credit is unavailable, synchronous mandatory
work uses the image budget already owned by its parent. Image credit cannot
authorize an SDK wave. Initial interpolation and single-cohort extrapolation
use lexical image leases. Cache funding survives cancellation, failed startup
and unproven worker, mmap or file retirement until settlement. Different
mandatory parents can reuse donor pixels through bounded pins, but cannot extend
the donor cache's funded lifetime with a borrowed lease. Independently funded
caches retain shared borrowing; borrowed input backing retains its original
owner. Retry geometry still comes from observed boundary contact.
Each actual owned cache or unpublished private file retains its own funding
hold. Producer wrappers do not add a second hold or a separate retained-credit
list. Shutdown must retire the published file before returning its credit for
mandatory and optional grants alike. One unproven cache does not prevent an
unrelated, successfully retired cache from returning its grant.
Healthy consumed prefetch exits join the producer in short slices without a
normal-retirement deadline. Cancellation, consumer failure or unused abandonment
uses a 30-second join budget. Expiry reports unsettled work and retains its
image/profile ownership; it does not force a refund.

GPU compute credit is acquired after request/seed preparation, immediately before
submission. The worker still renders RGB frames and initializes SDK inputs under
that lease. Scheduler join precedes pool shutdown, and outstanding producer,
worker, packet and cache owners retain their resources through cancellation or
uncertain cleanup. A cache's retirement depends on its own scope/mapping proofs,
rather than requiring unrelated SAM scopes to finish.

The existing system sampler reports aggregate `sam.scheduler.live` counts for
ready, preparing, running, completed-waiting and consumer-held work. Running
includes worker preparation and packing until its completion acknowledgement;
it is not a GPU-kernel utilization measurement. Consumer counts and hold spans
describe queue/ownership state, not CPU-core use or continuous computation.

For shared detector/SAM devices, completed full-frame parents can be checkpointed
before preparation waits on that retirement signal. The scheduler closes their
original dense mappings and returns their backing credit, allowing the remaining
detector views to finish. It then resumes preparation under the resolved dense
limit and the existing transient admission rules. Tile detector cleanup can
publish compact results while its parent support is deferred. A checkpoint is
not a completed parent, a published bridge, or permission to start SAM early.
During budgeted background drains, both stager pump calls share the existing
`BackgroundDrainBudget` time/action allowance. Prospective dense/startup memory
probes consume that allowance even when denied, and denied deferred candidates
rotate to the queue tail. Checks between atomic probes and ownership transitions
bound the sweep; one blocking probe can still exceed the time budget.
Resume admission tries the shared parent condition without blocking. Lock
contention rotates that candidate without acquiring dense credit, and
`resume_lock_denials` records those refusals. Settled model startup requires no
new startup-specific resident/headroom probe. Background budgeting remains
active while SAM is live, including after YOLO inference finishes.

Multiple parents can own RAM-first backing, claimed before workspace allocation
and bounded by retained bytes, physical/cgroup headroom and room for the next
detector group. Completed immutable RAM parents leave the active detector
admission window while retaining their original backing lease and RAM charge
until checkpoint retirement. At the RAM threshold, new parents use reusable
disk backing; freed RAM becomes eligible again after drain. Disk
flush/fsync/retirement runs independently of the RAM codecs. Completed
uint8 masks with a single nonzero value
use the existing packed CVOL format; uint8 confidence uses existing lossless
blocks, preserving all scores independently of mask support. The first disk
write for a RAM checkpoint is compact. Codec workspace is separately bounded; unavailable checkpoint
codec credit takes the raw streaming route without waiting behind SAM.
Unsupported numeric inputs remain exact raw arrays. Already written owned disk
maps are reused, and existing compact D1 shadows stay compact.

Short-lived detector RAM birth is qualified separately from long-lived backlog
retention. A full backlog quota can still permit RAM backing when verified
physical headroom covers the birth and outstanding promises. When it cannot,
new mask/confidence files use an authenticated ordinary-filesystem directory
under the checkpoint root, including when `--temp` names tmpfs. Both files and
sibling births share the disk-space preflight; allocated extents or outstanding
sparse-file promises prevent spending the same capacity twice. Existing files
map with `r+` and are reused as checkpoints. Failed births retain credit until
mapped aliases retire, and root/file identity checks protect rollback and final
cleanup. Unknown backing or insufficient disk space fails explicitly rather
than returning to tmpfs. The admitted-RAM path retains its existing behavior.

Original owners close before dense credit returns. Both raw backing reopen and
compact decoding run on the preparation executor. Compact restores use bounded
codec credit and RAM when admitted; raw reopening needs neither codec scratch
nor a RAM-promotion probe. Disk remains the safe backing fallback. Checkpoints
require ordinary disk scratch or output backing. Existing
`sam_interpolation.parent_staging` counters include compact logical/stored bytes,
raw fallbacks and `checkpoint_wall_seconds` (including CPU encoding and fsync).
They measure application work, not physical device traffic or peak occupancy.
Each `sam_parent_restore` event separates codec-credit waiting, backing allocation,
mask/confidence restoration and decoded bytes, including failure or cancellation.
Restoration includes reading, validation and copying; it is not isolated codec CPU
time. These timings exclude the earlier dense/executor queue and later SAM work.
The existing `sam.scheduler.live` sample also reports cumulative packet-decode
and consumer-hold seconds, including current holds. Consumer holds cover refill,
decoding, evidence consumption and retirement; overlapping scopes do not represent
serial walltime.
`sam_parent_memory_requested`, `admitted`, `owner_expired` and `credit_returned`
events record admission waiting and reservation lifetime. The latter three names
use the same `sam_parent_memory_` prefix. Caller expiry and actual credit return
are separate: unsettled tracker owners retain their reservation. These are promised
bytes, not measured resident allocations, and the admission policy is unchanged.
RAM birth admission counts actual RAM commitments rather than full logical
disk-backed arrays. Only proven disk-only mappings are excluded; mixed backing,
unknown filesystems, lazy RAM and future canvases stay fully charged. Preparation
restores the full promise before it can create RAM output. Logical dense leases
and ownership limits remain unchanged. The existing staging gauge records the
latest refusal's requested bytes, active RAM commitments, excluded disk bytes,
detector reserve, limits and sampled physical headroom.
For registered memfds on Linux, fresh `SwapTotal == 0` permits a resident-byte
discount in the physical check. Occupied blocks are sampled before free headroom;
allocation identity and lease basis are revalidated afterwards without counting
later growth. Stale, changed, swapped or unproven backing receives zero discount.
Unfaulted promises, logical dense limits, the RAM-bank cap and reserves remain
unchanged. Refusal telemetry records the proof status and resident/future bytes.
On scheduler failure, the first traceback is flushed before teardown waits.
Queued preparation is cancelled before restore or admission; its dense credit
returns only after actual input owners retire. Running work still settles, and
GPU quarantine remains in force until cleanup is proved.

Checkpoint workers use the existing parent concurrency allocation and start
encoding immediately after the last detector writer finishes. Optional SAM
cleanup/planning no longer delays that drain. A separate request can build
the existing lazy host processing cube after decode readiness, while reserving
headroom for unused dense/transient allowances. Neither path loads SAM on a GPU.
Authoritative planning runs under fresh admission and reuses matching pixels;
the ordinary preparation stage performs detector cleanup exactly once.
Confidence checkpointing uses the existing compiled ordered-frame encoder,
bounded by the task's slice workers and already-reserved codec workspace. Scores
serve as their own nonzero mask with full-frame bounds, preserving every uint8
value, including scores outside detector support. If its frame window cannot
fit, the existing serial codec is selected before writing. Cancellation joins
all encoding work before retiring the source; checkpoint formats are unchanged.

SAM parent preparation derives its default from the allocated main-process CPU
budget: one parent per sixteen allocated workers, or two parents per requested
tracker slot, whichever is larger. Four GPUs with two sessions therefore target
at least sixteen parents; larger CPU allocations can raise the target. View,
CPU and physical-memory caps still apply, and live memory admission determines
which parents start. Slice workers divide the existing CPU budget across them.
Image builds with distinct live parent credits can use those producer lanes
without a separate two-builder ceiling. Same-credit builds cannot overlap, and
uncredited callers share the existing aggregate cache/render allowance.
`YOLO_TTA_PARENT_POSTPROCESS_WORKERS` retains its explicit override. These tasks
include planning, rendering, selection and publication, so they need spare lanes
while other parents wait or consume results; they are not extra GPU sessions.
On large hosts, automatic SAM dense and transient defaults each use 40% of
physical/cgroup/SLURM headroom after a 64 GiB reserve. The extra transient
allowance is limited by the room below an 80% combined dense/transient budget
after resolved dense credit. The inference cap, small-host floors, explicit
byte limits and policy clamps remain authoritative; non-SAM defaults keep their
existing ceilings. Policy admission includes the full SAM transient pool
when reserving space for dense parents and output work.

Before dispatch, eligible GPU-detector-only SAM runs transfer spare default dense
capacity to the transient pool without increasing their combined ceiling. The
dense floor retains the inference allowance, the original CPU-input cache budget
and two allocations of the largest full parent in the actual task inventory, and is
never below twice the inference allowance. Individual memory charges are unchanged.
Policy groups, tiles, CPU/hybrid detectors, retained temporary output, explicit
dense/transient overrides and existing resource owners keep the previous split.
`sam.parent_memory_partition` records the resolved split and its inputs. This can
admit more parent preparation; it does not guarantee that every configured parent
fits concurrently or that dense capacity cannot become the next constraint.

After detector and input preparation have fully retired, the coordinator checks
the split again as work completes. It retains the exact bytes of live dense
leases plus two allocations of the largest deferred restore, and transfers the
remaining dense capacity to SAM. Pending builders, checkpoint owners, input
packets, startup promises or uncertain ownership postpone this adjustment.
The combined ceiling stays fixed; active parent grants keep their original size
and planning limits. Waiting parents are notified when capacity grows.
`sam.parent_memory_dynamic_partition` records each transfer and the retained
ownership and future-restore bounds. This uses the allocation ledger rather than
discounting a live reservation because its pages are currently absent from RSS.

Production requests two isolated predictor processes per physical GPU. Set
`YOLO_TTA_SAM_SESSIONS_PER_GPU=1` for the serial path, or 3 or 4 to evaluate more
independent tracks. Two remains the default; compare total workload completion
and peak memory before adopting a higher count.
For four workers, put this before the Python launcher in the SLURM script:

```bash
export YOLO_TTA_SAM_SESSIONS_PER_GPU=4
```

To isolate the worker-count effect, keep `YOLO_TTA_PARENT_POSTPROCESS_WORKERS`
fixed across trials; its automatic value also scales with the SAM slot count.
For example, `export YOLO_TTA_PARENT_POSTPROCESS_WORKERS=16` keeps sixteen parent
workers in each trial. These settings apply to newly started jobs.
Test parent concurrency separately from the SAM process count. Raising the
parent override does not enlarge the dense or transient memory pools, and it
divides the default per-parent slice-worker allocation. When live grants fill
those pools, additional parent threads wait rather than prepare more work.
Each slot has independent model/session state, precision context and feature
cache. This uses more model memory and permits CPU preparation/consumption in one
worker while another uses the GPU. Separate CUDA contexts do not imply simultaneous
kernels without MPS; no MPS service or policy is changed automatically.
[NVIDIA documents that scheduling distinction](https://docs.nvidia.com/deploy/mps/595/architecture.html).
When image preparation needs a device, unused slots may join the current compute
turn once; used slots then drain before image preparation acquires the device.

In progressive startup, each physical device initializes a fixed cohort. Cohorts
can initialize concurrently and register with one shared tracker scheduler after
admission; they keep their configured device and exact worker-slot identities.
Once a detector device has retired and the full fleet's host credit is funded,
parent restoration, CPU cleanup and bounded observation discovery may overlap
model startup. The existing observation inventory avoids repeating discovery
in normal planning, which still rebuilds scope identities and topology. This
prefix retains its original parent reservation and releases its cached data
before that reservation ends. Projection and GPU image/tracking work still wait
for the first ready admitted cohort; later cohorts may continue starting.
Already-ready parents do not perform an extra discovery pass. SDF interpolation
before SAM extrapolation skips early discovery of masks that SDF will change.
Later startup failures remain explicit and retain cleanup ownership; a partial
ready pool does not silently become a successful smaller fleet. SAM residency
continues to forbid YOLO reuse on the same GPU.

Multi-worker startup divides measured free VRAM, after mandatory headroom, equally into enforced
per-process PyTorch allocator quotas before loading that device's models.
Host checks include live memory promises, and admission rechecks actual free
memory after its workers are ready. A resource refusal retires the attempted
cohort before falling back to one worker on that device. Valid empty detector
metadata avoids unnecessary model loading. Nonempty tile callbacks warm before
reserving their own SAM workspace.
Unfunded model forecasts remain planning metadata and publish zero owned pool
debt; unused/no-tracking paths do not reserve that model credit or fence dense
admission. Direct progressive preparation waits for the first retirement proof
with no owned fleet credit; the host-funding deadline starts after that proof.
The first grant validates the complete forecast. Before starting progressive
cohort threads or admitting parents through the first ready device, one host
check funds the full parent transient envelope, image bound,
actual pending checkpoint/source RAM, all configured model startup peaks and
reserve. Every remaining model peak is charged to the shared parent ledger.
Device claims consume this owned credit without charging it again, including
devices still waiting for retirement or GPU ownership.

Later cohort checks reuse the funded parent envelope: only capacity or oversize
growth beyond it adds new parent debt. Current image bounds, actual pending RAM,
remaining model peaks and reserve still require headroom. Proven readiness or
fallback cleanup reduces the owned credit; uncertain constructors retain it.
The checkpoint portion uses actual lease RAM commitments minus authenticated
resident memfd pages sampled before fresh physical headroom. Unused dense
capacity is not a checkpoint allocation.

While funded model startup is owed, new RAM backing and preparation proof resets
must fund the full parent-capacity snapshot plus prospective RAM under the
shared parent condition. A refusal defers preparation without waiving its RAM
obligation when a proof reset cannot be funded; it retries as startup settles.
Backing creation can select disk on RAM refusal,
and decoding runs outside the critical section.
`sam.startup_admission` telemetry records per-device attempts, admitted devices,
requested/effective session counts and fallback reasons, including partial
startup in an interrupted run.
`sam.startup_progress` reports per-device stages, including retirement waits;
host checks include budget snapshots. Host funding and GPU-ownership waits have
300-second per-attempt limits; retirement acknowledgements do not acquire that deadline.
Successful startup does not prove that every future full history fits a quota;
later OOM fails explicitly without shortening histories or publishing partial
support. Serial mode retains the previous allocator behavior.

Resource-grant eligibility and per-session acceptance keep their physical-device
policy. Separate authenticated `execution_slots` can increase concurrency only
within the same owned CPU-wave bytes; uncredited legacy calls retain their old
physical-device cap. When both view classes are ready, the scheduler gives curved
scopes three submissions per cheap-view submission, with round robin inside each
class. If only one class can run it proceeds immediately; unclassified callers
retain ordinary round robin. Actual geometry family comes from the producer
context, without changing scientific request metadata or run IDs. Dispatch fills idle
physical GPUs before second slots, then uses advisory slot-local crop affinity.
Slot index, PID and attempt are checked independently of scientific model identity.
Evidence commits retain
stable run IDs, lineage, selection, and directional pixels despite out-of-order
worker completion. Packed bundle offsets, checksums, and timings can differ
between fresh generation attempts.

Group evidence is written as results arrive when its estimated contract/packing
and tiled-assembly scratch fits beside the unchanged CPU inference wave in owned
credit. Insufficient credit retains the original pre-inference barrier. All
groups and skipped tiles are still included before evidence commit. Generation
stats report `sam_group_evidence_schedule` and
`sam_group_evidence_overlap_bytes`; interleaving changes scheduling, not masks,
acceptance rules or frame intervals.

Within each decoded result, a live tracker-scope permit can fund concurrent mask
packing and compression from unused attempt-wave bytes. The complete SDK wave,
raw-transfer histories, persistent assembly and serial writer allowance remain
reserved, as does the single held decoded result. The process-wide packing pool
is sized by the coordinator CPU budget, up to 32 workers. At most eight mask jobs
per result fit the available scratch and coordinator CPU reserve; insufficient
credit uses serial encoding. Helpers receive owned immutable boolean planes.
Lazy contract reads, descriptor creation, payload offsets and ordered publication
stay on the original producer thread, preserving mask identity and selection.
Permitted-support owner intersections operate on the exact nonzero rectangle
of each already-generated Boolean contribution. Empty contributions perform no
integer owner comparisons. Corridor geometry, owner dtype and the existing
resource charge stay unchanged; no full coordinate list or mask stack is added.
Cancellation and failure join the pending jobs before refunding the permit;
unproven submission or completion retains ownership.
`sam.consumer.parallel_mask_submissions`, `mask_encode_host_seconds`,
`mask_encode_thread_cpu_seconds` and `mask_publish_host_seconds` use the same
`sam.consumer.` prefix to distinguish this work from consumer queue holds.
Long-group evidence validation indexes branch-mask names once, then preserves
the existing frame order and subset checks. It does not scan all mask names for
each frame or materialize additional dense geometry masks.
Private evidence reads reuse one unbuffered staging handle. Every read still
flushes pending writes, verifies file identity and authenticates the complete
mask record. Commit and abort close the handle before publishing or removing
staging; no decoded-mask cache is added.

With `YOLO_TTA_TASK_TRACE=1`, `sam_phase_begin` and `sam_phase_end` events identify
planning, image rendering, retry, selection and publication in the existing
telemetry streams. Scope, operation and phase identities pair concurrent calls.
An unmatched begin in an interrupted capture identifies unfinished host work;
these spans may overlap and are not GPU kernel durations.
Both events include `thread_native_id`; phase-end records also include
`phase_host_seconds` and `phase_thread_cpu_seconds`. Calling-thread CPU time
excludes helper threads and waits, so their difference does not establish
global GIL contention.
Failed phase-end events include the exception class and a bounded message.
`context_preparation_failure.json` also records planning errors raised before a
prepared plan exists; failure to write diagnostics does not replace the original
exception. Scheduler wait telemetry includes transient-pool usage, GPU resident
owners and quarantine reasons, asset retirement, and inference backlog.
Per-job `timings.sdk_session_init_seconds` separates SDK input loading/state
initialization from the remaining work inside inclusive `tracker_seconds`.
After authenticated worker completion is accepted, `sam.worker.`
counters accumulate `completed_jobs`, `render_seconds`,
`input_preparation_seconds`, `tracker_seconds`, `sdk_session_init_seconds`,
`pack_seconds`, `worker_seconds` and `queue_seconds` from existing receipts.
`completed_output_frames` counts a receipt's complete directed frame inventory
only when it matches the admitted request. It includes seed and policy-generated
outputs and repeated frames across tracks; it is not unique video frames or
model inference calls. Missing or malformed optional timing/frame metadata is
omitted. These host timings overlap across workers, and SDK initialization is
already inside tracker time; they must not be summed as elapsed time or kernel
durations. The final worker CUDA fence follows `tracker_seconds` and is covered
by the broader `worker_seconds`; manifest publication and final owner retirement
occur after that timer is captured.

Automatic whole-crop generation prefers family FIFO when the runtime supports
its family API, subject to the balance guard below. One family stays on one device through independently seeded runs,
allowing exact feature reuse; a ready worker takes the next family immediately.
The final family tail can leave other workers idle. Capacity follows the actual
admitted devices and CPU wave, with no artificial four-device cap. Each endpoint
still starts its own tracker session; cache/model/quality settings are unchanged.

Automatic interpolation selection compares flat-job and FIFO-family frame-work spans using a
greedy minimum-heap assignment to the actual admitted slots. It falls back to
flat when `family_span * 4 > flat_span * 5`; this is a load-balance proxy, not a
runtime prediction. Explicit selectors bypass this guard. A small job count
alone is not an automatic fallback criterion.

`YOLO_TTA_SAM_FAMILY_SCHEDULE=flat` restores the prior crop-queue order. Values
`flat` and `fifo` are stripped and lowercased; empty/unknown values fail. An unset
selector automatically falls back to flat for older/custom runtimes lacking
callable family dispatch, recording `runtime_without_family_dispatch`. Explicit
FIFO for whole-crop tracker work requires that API and fails clearly if absent.
Tiled interpolation remains flat, recording `tiled_generation_uses_flat_dispatch`.
Flat dispatch continues crop affinity and lending idle workers between crops.

Extrapolation groups independent requests with identical image and crop identity
on one worker, within each existing image cohort or bounded tiled batch. Other
workers take other ready crop families. The same 25% frame-work imbalance guard
keeps a large single family from unnecessarily serializing the worker pool.
The explicit `flat` environment value also disables this grouping; extrapolation
retains its imbalance guard even for explicit `fifo`. These choices affect
dispatch and exact feature reuse, while original seeds, intervals and request
identities remain fixed.

Receipts retain original input indices, requested/effective/explicit scheduling,
fallback reason, device/worker identities, effective in-flight count and execution/
family completion order. Changed completion order does not change run IDs or
selection. Completed masks are packed and released promptly. No-op or reject-all
scopes retain the original volume and empty directional slots without allocating
a dense merged workspace. There is no new command-line flag.

The image provider pins the detector canvas geometry. It first checks compatible
captured canonical gray views before GPU rendering admission. It aliases verified
canonical Transverse backing when available; remaining requests render the
frame/crop rectangles into an immutable compact cache. Generic CPU views can
require one shared processing-volume memmap
materialization, reused across their demands. The decoded-slice shortcut remains
limited to Transverse. Identical demands reuse saved pixels; eligible CPU routes
also reuse matching covered subsets and overlaps. Empty plans cause no
materialization. Reduced square canvases retain their existing affine and
stack-resampling semantics.

Canonical YOLO gray capture is automatic when SAM uses GPU detector workers and
ordinary disk-backed temp storage. `YOLO_TTA_SAM_GRAY_CACHE=0` disables capture
and its host-buffer reservation; SAM retains its ordinary source-rendering path.
This switch controls the YOLO-produced greyscale cache, independently of
`YOLO_TTA_SAM_GPU_IMAGES`. Capture uses the first selected YOLO model's
angle-zero base full frames for compatible Tilted, Azimuthal, Radial and Spherical
views. Ordinary Cartesian and dense optional tile inputs are not captured.
Generic CUDA inference and the direct prediction ring use the actual normalized
center-channel tensor: float32 clamp to [0,1], multiply by 255 and round to uint8.
CUDA events fence input-buffer reuse after the last source read and separately
prove copy completion before compute credit returns. This is a
recorded detector-input sampling contract, separate from the renderer's pixels.

Each worker has bounded capture buffers for two depth-eight packets per
configured encoder, with a minimum of 32 raw frames, and up to sixteen CPU
encoders within its supplied CPU allowance. The first four encoder threads keep
their inherited priority. Extra Linux threads lower their own scheduling
priority, allowing idle-share work to yield to ordinary CPU work. This is
best-effort OS scheduling within the existing affinity mask, not a global core
reservation broker. Unsupported or denied priority changes are recorded.
At 2048 pixels, sixteen encoders fund up to 256 raw frames and reserve
4.0625 GiB per worker, or 16.25 GiB for four workers. The reservation includes
host copies, concatenation and encoding workspace and stays in ordinary RAM
admission accounting. CPU compression overlaps YOLO execution; the remaining
compression/publication tail is deferred after compute credit returns and keeps
its host buffers and publication future owned until settlement.
One SQLite database per physical view stores lossless 8×256×256 chunks. Optional
`imagecodecs` supplies Zstd level 1. If unavailable, automatic selection tries
`deflate>=0.9.0`, then ISA-L, then standard-library zlib, all producing the same
zlib wire format at level 1. Known old, unknown and prerelease libdeflate bindings
are not selected as the parallel fallback. Chunks that do not shrink stay raw.
Encoder resolution is cached per process, including unavailable optional imports;
a bounded CPU roundtrip proves the selected encoder before use. Explicit `zlib`
in the Python API retains the standard-library baseline. The root is fresh per run
and retires with run-scratch cleanup under the existing keep-temp semantics.
Opening an initialized store validates the header and identity without taking
the schema-creation write lock. Actual packet publication remains transactional;
parallel encoders do not hold that lock while compressing.
An incomplete view checks its frame count without taking the sealing write lock.
Potentially complete views repeat identity and coverage checks and validate the
content digest inside the final transaction.

Startup prints the encoder implementation and version before GPU workers start.
`sam.gray_cache.encoder.auto` records the actual process-local selection and
fallback reasons. From the updated repository, the following CPU-only check
reports the active environment's selection and exercises its known-answer test:

```bash
CUDA_VISIBLE_DEVICES='' python -c "from XTA.gray_cache import gray_encoder_info; print(gray_encoder_info())"
```

One fast implementation is sufficient; installing all codec packages is not
required. The existing `acceleration` extra supplies `deflate>=0.9.0` and ISA-L.
The standalone deflate wheel can also be installed without changing model
dependencies: `python -m pip install --no-deps --only-binary=:all: deflate==0.9.0`.

SAM reads a sealed complete canonical full view only when source, processing
shape, canvas, physical recipe and capture identity match. It copies demanded
crops before taking a GPU rendering lease. A decoded-chunk LRU of at most 64 MiB
per cohort is charged alongside crop/decode workspace. Enlarged contexts inside
the canvas and cyclic mirrored addresses use the existing geometry and coverage
rules. Missing, incomplete or source-incompatible stores keep the ordinary renderer;
CPU inference steals can leave capture incomplete. `sam.gray_cache.*` diagnostics
record admissions, rejections and read pixels.
`sam.gray_cache.zstd_chunks` and its raw/zlib counterparts identify actual codecs;
matching `<codec>_raw_bytes` and `<codec>_encoded_bytes` counters record their
bytes. `encode_thread_cpu_seconds` separates codec CPU time from its wall
time. `sam.gray_capture.pool` reports active/queued encoders, priority policy,
live buffer credit and producer wait time; capture counters separate device-copy
waits from CPU encoding/publication. A full bounded backlog can still block
capture if sustained encoding or writing cannot keep up.
The cache uses ordinary disk scratch. Lossless compression savings depend on
the input data; a 16 GiB free-space floor declines capture when it cannot be
preserved, retaining the source renderer fallback. Cache availability and codec
timings do not establish target-system throughput.

GPU image construction is enabled by default (`YOLO_TTA_SAM_GPU_IMAGES=0` selects
CPU only). Ready materialized Cartesian sources use bounded native CPU crops
and the canonical global affine remap, avoiding full-volume GPU uploads.
Live admitted Cartesian/Tilted Cartesian cohorts select CPU preparation when the
materialized source and demanded workspace are ready. Tilted native ROIs use
bounded 32-row two-tap sampling and the canonical affine remap, without building
full-native-plane plans. Its sampler identity and demand prevent incompatible
feature reuse. Unready or unsupported cheap inputs retain the GPU route.
Other curved requests use the existing TTA GPU renderer first, with the same
registered intensity semantics and geometry-quality selectors as YOLO. Tilted
curved views retain their Radial/Azimuthal/Spherical classification. Complete
reusable image caches take precedence. GPU capacity checks, FIFO admission, SDK
turns and cancellation apply to background producers and synchronous builders.
Eligible Radial and Spherical image preparation uses the bounded CPU sampler
even when the SDK queue is empty, avoiding a cold full-volume GPU upload and
retirement before tracking can resume. This requires a ready, complete uint8
processing source, live image credit, and native ROI, shell strips and remap
workspace that fit the render cap. The same rule covers background prefetch,
synchronous first cohorts and retry fallback; synchronous CPU rendering stays
serial. Unready, unsupported or unfunded requests retain ordinary GPU admission.
The canonical affine
crop determines each frame's native tap rectangle with the existing two-pixel
guard. Both synchronous fallback and parallel futures sample only that rectangle,
using global native coordinates. `xta.sam_native_shell_demand_crop/1` permits at
most one gray level of rounding difference from full-plane grayscale sampling;
geometry, frame order, crop coverage and categorical sampling rules stay fixed.
Background prefetch of ready native shell ROIs uses independent CPU frame jobs within the same
render-scratch credit and unchanged 256 MiB default. Admission reserves the
owner's remap workspace and a transfer-plane margin, then uses ROI-sized job
accounting to bound up to eight workers by CPU count, frame count and remaining
scratch. Results yield in original order. One owner remaps, copies
and publishes the crops; cancellation joins outstanding jobs before releasing
source ownership. Tight budgets use the serial CPU path.
SAM's uint8 Radial and Spherical ROI preparation uses compiled, GIL-releasing
CPU kernels with float64 coordinates and ordered float32 trilinear accumulation.
Radial arc coordinates are calculated once per ROI column, including tilted
views, rather than once per row stripe. The kernels use the existing prefetch
workers and scratch grants, with no additional native thread pool. The NumPy
renderers remain the references and continue to serve non-SAM and categorical
requests. SAM image identities include the native
sampler source, compiler and CPU target before cache keys are formed; failed
native execution cannot silently publish reference pixels under that identity.
Eligible GPU requests reuse TTA's projectors and uint8 affine sampling for supported
Cartesian, Tilted, Azimuthal, Radial and Spherical views. Tilted crops use TTA's
existing fused crop entry point and its fallback. Upright Azimuthal SAM ROIs
may fuse four-tap construction into one CUDA kernel while retaining the original
full-width Torch coordinates, FP32 operation order, row folding, gray8 rounding
and affine warp. Optional CuPy compilation uses explicit no-FMA, no-FTZ and
precise-division options without changing global compiler settings. Missing
capabilities or unqualified geometry retain the Torch reference; a launched
kernel failure follows ordinary renderer cleanup and is not a silent retry.
`YOLO_TTA_SAM_FUSED_AZIMUTHAL_TAPS=0` selects the original tap builder.

Tilted-Azimuthal crops restrict the existing Torch source gathers to the native
rectangle needed by the affine. Full-width taps and full-plane row coordinates
are formed before slicing, preserving the original interpolation arithmetic.
The sampled rectangle is quantized into the existing full-size zero canvas and
uses the same affine warp. This retains coordinate normalization and gray8
rounding while avoiding full-plane gathers; it needs no additional source copy.

`sam.gpu_images.azimuthal_taps.cuda:N` records the selected tap backend, engine,
device, stable reason and monotonic selection time, with bounded capability
error details. It updates only when that engine's outcome changes, not for
every crop. `enqueued` means the fused launch was submitted; it does not claim
completion or count rendered crops. A launch failure replaces that engine's
last successful selection state with `failed`. These are runtime diagnostics,
separate from the image sampling identity and existing crop completion phases.

A complete existing CPU
cache takes precedence. Otherwise a producer wins its host-memory credit and
cache-build ticket before requesting a GPU. Concrete GPU-ready requests use FIFO
admission. The head drains one GPU by preventing new SDK refills there, including
reuse of an existing dual-worker fence. Accepted jobs finish and acknowledge
normally; other GPUs continue. Ready SDK work receives a turn between source
bursts. The 30-second admission watchdog starts when a request becomes head,
not while it waits behind other builders. Cancellation wakes the wait; no GPU
lease or source allocation is held by waiters. A proven VRAM refusal rotates
the head to another candidate; insufficient capacity retains CPU fallback.
The transaction admits source
residency and projection/grid scratch, returns only demanded crops to the host,
and releases GPU storage before returning its compute lease. A builder may first
hand its source and exact compute lease to one already-ready same-source request,
after fencing and clearing view scratch and rechecking recipient headroom. At
most two builds share that upload while SDK work is ready; additional compatible
handoffs require a scheduler-locked check that no SDK work is runnable. The next
turn belongs to ready SDK work, and no source
is retained idle outside the compute lease or predictor quotas. Missing capacity,
unfinished source decode or unsupported source geometry keeps the CPU path.
Within a large image build, a completed crop/batch boundary can yield to
dispatchable tracker work after 30 seconds of render service. Source upload
time is excluded from that service turn. The renderer fully retires CUDA source
and scratch before returning compute; the private host cache and its image
credit remain owned. The next render reacquires through the existing FIFO and
SDK-turn rules. A busy voluntary resume remains cancellable beyond the initial
30-second admission timeout without changing the selected GPU backend; hard
device/headroom failures keep their existing handling. A nonpreemptible crop
can extend the turn, and reacquisition requires another source upload. No yield
is introduced while tracker work is unavailable.
A clean source handoff from a known renderer can omit process-wide garbage
collection and allocator trimming after its stream and scratch fences, provided
fresh driver-free bytes cover the recipient's full additional workspace and
reserve. Low free space or missing ownership/free-byte proof keeps conservative
trimming. Ordinary renderer retirement still releases the source, fences work,
collects garbage and releases allocator caches; failure, cancellation and
unproven retirement keep the full cleanup path.
Ready materialized processing cubes, including streaming-preprocessing outputs,
are used directly. Unmaterialized streaming cubes keep CPU preparation to retain
their endpoint-aligned temporal grid; the GPU virtual native-T path is used only
for the matching nonstreaming center-aligned recipe.

GPU-selected Radial cohorts upload only the conservative native-time slab needed
by their frozen radius inventory and full native patch. Bounds include every
angular column, both height endpoints and temporal sampling guards. Unsupported
geometry retains the full source. The CUDA and Torch samplers preserve global
native/logical time dimensions and gray8 reconstruction; the slab origin only
rebases source addresses. A larger resident slab can serve a contained demand
without reuploading; handoff verifies the same original source and retains its
actual origin and memory charge. Planned sampling identity remains fixed, and
saved-upload counters count the recipient's planned bytes. This leaves predictor memory quotas,
tracking intervals and proposal-quality policy unchanged.

This route accepts the existing TTA fast-path intensity differences; it does not
change frame selection, crop coverage or source coordinates. CPU/GPU pixel and
feature-cache identities stay separate. GPU identities include the exact crop
demand, preventing a feature hit or partial donor reuse across GPU crop-origin
rounding phases. Cartesian and other CPU routes retain canonical cross-demand reuse.
CPU shell image and feature identities include both the native ROI recipe and
the demand hash. Different ROI contexts cannot claim byte-identical pixels.
Authenticated matching source, geometry,
logical shape and cyclic frame-address proofs permit a retry/cohort to use the
other sampler. Saved metadata alone cannot authorize that switch. Portable
`image_sampling_sources` and per-run/cohort identities record the actual inputs;
the evidence scope identity remains pinned for import validation.
`sam.gpu_images.*` counters report admissions, rendered pixels, source uploads,
admission wait time/timeouts, source handoffs and fallback reasons. Separate host
timings cover allocation/upload, render submission, copy waits and retirement;
these are not CUDA kernel durations. Window savings and actual upload bytes are
separate from handoff savings. These are route diagnostics, not a throughput guarantee.
`sam.gpu_images.phase.cuda:{device}` keeps one latest operation record per GPU,
including its start time, image owner, scope, frame and reporting thread. It marks
entry into source upload, native rendering, quantization, warping, device-to-host
copy, cache publication and retirement. Interpret it alongside current image-lease
ownership: a final release record can remain after the device changes purpose.
These bounded gauges add no CUDA queries, synchronization or per-frame trace events.
`sam.cpu_images.prefetch_admissions` and `sam.cpu_images.prefetch_host_seconds` report admitted
CPU image construction and its host time. `radial_prefetch_admissions` and
`spherical_prefetch_admissions` under the same prefix count each shell family's
CPU prefetch route; `radial_synchronous_admissions` and
`spherical_synchronous_admissions` count the corresponding synchronous route.
`sam_image_memory_admitted`,
`sam_image_memory_declined` and `sam_image_memory_credit_returned` separately
report staging grants and settlement. Their `image_staging_in_use_bytes` and
`image_staging_capacity_bytes` fields retain unsettled debt after a producer
wrapper exits; they do not measure resident memory or alter parent grants.
`sam.cpu_images.native_sampled_pixels` counts actual native ROI pixels, while
`native_full_frame_pixels` records their full-plane counterparts. Separate
`native_sampling_host_seconds`, `native_sampling_thread_cpu_seconds`,
`canonical_remap_host_seconds`, `native_prefetch_wait_seconds` and
`cache_write_host_seconds` and `cache_flush_host_seconds` counters use the same
`sam.cpu_images.` prefix.
The `last_native_prefetch` gauge records actual workers, ROI job/plane bounds,
owner remap allowance and admitted workspace. Worker timings can overlap and
do not establish pipeline walltime or target-system speedup.

The pinned SAM SDK's RGB loader also has a prepared-input route. The worker
allocates one contiguous CPU float16 clip within its existing CPU-wave allowance.
The exact CPU path uses Pillow resizing and a 256-value normalization lookup;
the CUDA path resizes one frame at a time with separable antialiased bicubic
sampling and uint8 rounding/clamping between axes, then copies normalized pixels
to the CPU clip. It retains native frame count, geometry and CPU offloading.
Insufficient CUDA workspace keeps the exact CPU route. Unsupported SDKs retain
their original loader. The scoped adapter preserves the SDK's complete session
initialization and restores the original loader on success or error. Per-run
receipts and feature-cache identities record the actual input policy; the CUDA
path accepts small intensity differences. `YOLO_TTA_SAM_GPU_IMAGES=0` also selects
the exact CPU input-preparation path.

When a lazy Transverse source has unchanged XY geometry and only its time axis
needs resizing, bounded batches reuse the exact OpenCV temporal resize for
multiple requested frames. Native planes, crop outputs, resize slabs and remap
workspace share the existing render cap and live resource allowance. A source
already materialized by another view, cached partial pixels, unsupported geometry
or insufficient capacity retains the ordinary rendering path. The
`sam.transverse_cache.*` counters identify which route ran.

Extrapolation freezes its complete plan and baseline, then groups whole original
terminal groups into image cohorts that fit `YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES`.
The union of requested rectangles on each frame determines each cohort's actual
cache demand. Groups and their complete tracking intervals are not split. Every
planned group is checked before rendering or SDK work; a single oversized group
causes an explicit resource error rather than silently disappearing from output.
Preexisting planner refusals remain separately reported.

All cohorts share one evidence writer, predictor pool, final selection and
publication. Adaptive retry budgets apply to the complete pass. Frozen
per-frame observation indexes avoid rescanning all observations for each crop,
and cohort transitions do not repeat whole-volume snapshot hashing.
In a multi-cohort pass, owned gray caches retire only after detached RGB inputs,
completed worker streams and raw-result packing/release have been verified.
Borrowed source backing and protected shared caches retain their owners.
Predictors and encoded feature caches survive cohort transitions. A scope that
already fits one cache retains its ordinary cache-reuse behavior.

## Bounded reuse and resource controls

These caches reuse identical inputs and measurements. They do not share tracker
state between independent endpoint sessions or relax proposal quality.

| Control | Default | Scope |
| --- | --- | --- |
| `YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES` | 1 GiB | Each immutable image-demand cache; not a total process budget. Aliased canonical backing adds no cache storage |
| `YOLO_TTA_SAM_RENDER_MAX_BYTES` | 256 MiB | Bounded native sampling and remap workspace |
| `--sam_feature_cache_mib` | `1024` MiB | Requested retained feature budget per admitted predictor; zero disables retention |
| `YOLO_TTA_SAM_FAMILY_SCHEDULE` | Automatic whole-crop scheduling | Balanced supported FIFO, otherwise recorded flat fallback; explicit `flat` backout or `fifo`; tiled stays flat |
| Proposal reader `max_cache_bytes` Python argument | Automatic with live credit; otherwise 32 MiB | One evidence-reader transaction; explicit budgets remain explicit, zero disables retention |

Production selection may enlarge the existing reader cache to the smaller of
2 GiB, one eighth of live phase credit, or the credit remaining after the
largest actual group topology workspace and bounded spool. The full outer cache
is charged before intrinsic parallel and branch-prefix admission; each intrinsic
lane still has at most 32 MiB. Insufficient prefix slack retains the existing
equivalent full-merge path. Saved metadata cannot authorize this allocation,
and group acceptance bounds and selected-mask semantics are unchanged.

Feature-cache admission reserves CUDA headroom: by default the larger of 2 GiB
or 15% of device memory. The retained budget is limited by current usable free
memory after that headroom. Active sessions and the model remain separate memory
consumers. The worker reuses exact frame features only when source cache,
physical view, crop, native frame, loader/preprocessing, model, layout, device,
and precision identities match. Positional tensor storage is accounted once,
including shared storage. Existing LTA execution keeps its cache disabled by
default. `--sam_feature_cache_mib` is the production control; there is no
environment override. The lower-level tracker accepts `feature_cache_bytes`
and an optional `feature_cache_headroom_bytes` Python argument.
The CLI value must be a nonnegative integer. Zero disables cross-session LRU
retention while preserving ordinary within-session tracker handling and quality
settings. Its configured value remains recorded when interpolation is inactive.
Shared positional reuse additionally requires the verified pinned SDK tree and
inspected position encoder. Missing, custom, or compiled encoder identities
retain ordinary per-frame positions; exact same-frame feature reuse remains
available.

The 1 GiB request reduces eviction during longer traversals and nearby walk-back
sessions. Actual capacity depends on the retained tensor layout and shared
positional storage. The budget remains a fixed upper limit, capped by usable
CUDA memory and the headroom above; explicit CLI budgets,
including zero, remain authoritative. Reuse changes visual-feature retention,
not the independent prompt or tracker history. Larger budgets cannot eliminate
first-use misses for different immutable images or crop footprints.

The CPU proposal reader retains immutable raw, filtered, and candidate products
within its transaction. Memory-limited replay and final audits cap retention at
the smaller of 32 MiB or one eighth of their workspace budget. Entry/exit
integrity checks preserve source/payload identity. Cache statistics distinguish
hits, decoding, filtering, evictions, peak bytes, and completed transactions;
retention is released when the owner closes.

Stock branch selection keeps a reader-owned immutable prefix of validated
records. Each incoming chunk is fully validated, while earlier packed support
records are shared instead of repeatedly deep-copied. Ordered cross-family
contact and topology checks remain in place. Simultaneous prefix/candidate
indexes are charged to admitted topology slack and effective measurement
credit; insufficient space uses the original full-merge path. The complete
portable receipt is flattened, fingerprinted and fully validated before return.
Prefixes cannot move to unrelated or closed readers; authorized borrowed lanes
retain their parent transaction's lifetime checks.

The existing interpolation worker hint also reaches stock proposal measurement.
Parallel intrinsic measurements require authenticated live extra parent credit;
without that credit, or with a custom proposal hook, evaluation remains serial.
Within the current family, each lane owns its cache/cursor over shared immutable
metadata and one outer integrity transaction. Conservative charges include plane,
diagnostic, cache and control buffers, and pending work is bounded by the worker
hint. Completed lanes return their credit immediately and allow new independent
measurements to start without waiting for an older slow lane. Results are restored
to their original run order; an oversized serial measurement waits for all active
lanes to retire. All lanes join and retire their caches before the unchanged ordered stock,
topology, joint, rescue or custom-hook decisions. Operational receipts under
`selection_resources.intrinsic_measurements` record admitted credit, pending work,
serial reasons, task/reader counters and measurement time. This concurrency does
not change quality thresholds or deterministic selection order.

Whole-crop containment uses one compiled, GIL-releasing scan to count raw and
effective foreground, acceptance leakage and boundary contacts. It reproduces
the finite 3x3 erosion with zero-valued exterior pixels without constructing
full-plane erosion and intersection temporaries. Tiled halo diagnostics retain
their existing full boundary mask. Evidence authentication and selection
thresholds are unchanged.

Empty supporting regions skip mask reads. Group dilation evaluates foreground
plus a one-voxel halo, and edge connected-component work crops only known zero
margins before translating coordinates back. Component IDs remain identical for
the supported 6/18/26 connectivity choices. These shortcuts preserve the original
full-shape admission and caps; a smaller numerical crop does not admit an
otherwise refused family or change its declared geometry.

Performance receipts measure these paths separately. Local machine results are
sanity checks for the exercised workload, not target-system throughput claims.
For a cluster check, inspect runtime telemetry under `sam_interpolation.runtime`:
`image_cache_hits`, `image_cache_superset_hits`, and `image_cache_reused_pixels`
distinguish demand/pixel reuse; `source_materializations` and
`source_materialization_seconds` record shared processing-volume construction.
Read these alongside `rendered_frames`, `rendered_pixels`,
`image_cache_payload_bytes`, `image_render_seconds`, predictor startup, and
detector-retirement wait time. `rendered_frames` counts missing-rectangle sampler
calls, so an expanded frame with several uncovered strips can count more than
once; `rendered_pixels` counts newly rendered pixels and
`image_cache_reused_pixels` counts copied cache intersections. Counters absent
from older or injected contexts are null, not measured zero.
`native_sampling_calls` and `native_sampling_pixels` expose native-plane
derivation separately from returned crop pixels. Missing strips share the
transaction's native plane. `canonical_sampling_pixels` includes the bounded
alignment columns required to reproduce the installed OpenCV backend's global
interpolation phase. Crops use the original canonical grid, rather than a
rebased float32 affine, so cache contents do not depend on demand order.
The provider runs a small synthetic phase check once per rendering helper and
active OpenCV backend before a transformed cache is published or SAM is admitted.
Its largest synthetic canvas is 1535 by 73 pixels; it does not use real images
or render a full real canonical view. Unsupported numerical behavior fails
closed with a compatibility error. Exact source-backing reuse needs no check.
The receipt is exposed as `sam_canonical_phase_self_check` in pass statistics
and `canonical_phase_self_check` in runtime telemetry. OpenCV 4 and other builds
still need their own cluster validation; local OpenCV 5 tests do not establish
their behavior. A quick CPU precheck on each cluster environment is:

```sh
python -c "from XTA.sam_canvas_rendering import ensure_canonical_phase_supported; print(ensure_canonical_phase_supported()['status'])"
```

Model feature-cache/dispatch counters are separate
from image rendering reuse.

## Existing interpolation controls

The flag names retain their purpose across the two generators. Frame distances
and directions refer to the selected working view's native frame index. If
preprocessing changes the stack depth, `--interpolation_distance` counts those
processing frames. Bridge radius is measured in the selected view canvas's
pixels; it is not an input-video pixel measurement after a changed canvas scale.

| Flag | Default | SDF | SAM |
| --- | --- | --- | --- |
| `--interpolation_distance` | `0` | Maximum endpoint search frame distance; zero disables interpolation | Same search bound and disable behavior |
| `--interpolation_walk_back` | `1` | Additional source slices before an endpoint; existing SDF layer layout | Additional original detector-observed source slices; zero still keeps endpoint bridges |
| `--interpolation_candidates` | `1` | Consider up to the Nth nearest connection for an endpoint projection | Same candidate search budget; relevant observed family continuations also inform the bounded group |
| `--interpolation_passes` | `1` | Later passes may use the previous interpolated result | Maximum distinct observed-anchor planning rounds; stop when exhausted and record skipped rounds |
| `--interpolation_min_radius` | `3` | Reject a proposed bridge with radius at or below the threshold | Remove undersized 2D components from each full raw prediction before quality checks and publication; do not reject a whole run solely for those removed components; zero disables filtering |
| `--interpolation_search_angle` | `15` | Projection growth/search angle, strictly between -90 and 90 degrees | Same geometric candidate search cone |

SAM never uses a generated bridge as a fresh observation or walk-back seed.
An exhaustive first pass may consume all useful hypotheses. More requested
passes then do not create repeated evidence or additional prediction votes.

## Crops, proposals, and tile propagation

Planning starts from immutable detector-produced slice components and retains
their detector/view/augmentation/tile lineage. When original instance identity
is unavailable, endpoint identity is explicitly based on connected components.
A bounded family group can contain multiple observed daughter endpoints and
relevant continuations, including siblings that need no repair.

The planner bounds each group to 1,024 endpoints, 1,024 observations, 256 edges,
and 4,194,304 crop pixels. Explicit lower caller limits remain authoritative.
Production planning retains immutable family recipes. The native evidence writer
can generate exact contract planes with a bounded last-frame cache instead of
retaining every edge's complete frame stack. It retains the dense route where
that route has the lower proven peak. Family membership, crops, acceptance,
write domains and independent original-seeded sessions remain unchanged.
The resident-memory limit
therefore controls simultaneous storage, rather than permanently excluding
later families after earlier families consume a cumulative allowance. Explicit
eager/research calls retain their declared memory limits; baseline limits are
256 MiB per group and 512 MiB for retained contracts. A genuine per-family or
structural refusal remains explicit instead of silently omitting a sibling.
Physical spacing participates in geometric search; crop coordinates remain
view-native pixels.

The usual identified 4 GiB parent allowance separates 512 MiB for observations
and 512 MiB for persistent evidence metadata from image and other work. Smaller
identified allowances scale these partitions down; uncredited/direct calls do
not gain an owned allowance from a serialized receipt. Metadata preflight counts
all planned and refused families, run-frame records and tiled child packets.
Mandatory production image demand beyond the identified base image allowance is
reserved with the parent before preparation. Optional overlap may acquire a
separate image-only grant. Both remain charged through their actual cache and
worker-mapping lifetimes; neither authorizes additional SDK work. Image-only
prefetch grants reserve their whole allocation for images.

Expanded production capacities preserve compatible existing group/run identities
and their conflict priority. Newly admitted families follow existing admissible
families; existing stock-before-rescue policy ordering remains unchanged. A
storage or resource change must not silently reorder conflicting old proposals.
The dense research materialization API retains its borrowed-array lifetime
checks. Streamed production scratch stays charged until the native writer drains
its independently funded packing queue, including on failure.

TTA interpolation has no fixed 30-frame session ceiling or default 128-frame
family ceiling. A distance of 30 can require 31 endpoint-inclusive frames, and
walk-back can require more. `SamInterpolationSessionPlan` preserves every
requested frame and the original-seeded history; it does not split, reseed or
silently truncate a long session. LTA's separate `SamSessionPlan` still admits
at most 30 frames. An explicitly supplied planner frame bound remains enforced.
Known CPU input/output buffers are checked before staging and again in the
worker against its admitted byte budget. GPU model and tracker history memory
are separate; a backend memory failure remains an explicit failure.

Production can admit larger families through a live memory profile. Additional
SAM credit, up to 16 GiB, is reserved atomically with the parent's existing
transient work and constrained by physical/cgroup/SLURM headroom and the current
pool capacity. Pool capacity or swap alone cannot authorize the increase.
If total pool capacity and isolated physical headroom can fund the minimum extra,
admission waits when incumbent promises consume either pool credit or physical
allowance. It also waits when incumbents alone clamp the identified base CPU
allowance. Waiting takes no partial reservation, observes cancellation, and
resamples both limits. Genuine isolated low headroom, insufficient total capacity,
and the emergency oversize lane retain the lower declared bounds.
Contract construction
and policy topology use the credit in separate phases, and the resolved limits
are recorded with the plan and selection. Serialized profile metadata does not
authorize a replay allocation: replay must obtain its own live credit.
The host input/output allowance also limits simultaneous tracker jobs. Dispatch
uses the planned session sizes and transfer overhead; it can reduce concurrency
or wait for the previous result to be consumed before refilling a worker. It
does not shorten the requested session. Summaries report planned/refused family
counts and reasons, endpoint-inclusive session lengths, assigned and effective
planning budgets, and the resulting in-flight limit.

Each group fixes a context rectangle for its whole tracking interval, a tighter
acceptance region for measurements, and a branch-specific write region for
missing support. A context rectangle can cross an old detector tile seam. The
tracker does not grow its declared acceptance region in response to leakage.
An observed branch reaching its endpoint does not freeze the entire slice:
another missing branch may still receive an addition there.

Endpoint-seeded sessions run independently, so daughters converging on a shared
parent may overlap. Only the chosen starting observation is injected; opposite
endpoints are held out for measurement. Raw masks, empty observations, frame
coverage, containment violations, and tracker scores are retained before
publication cleanup. Structural failures and incomplete evidence cannot be
converted into a successful empty bridge by relaxing a quality policy.

The main pipeline's component-gated OR is used for both backends. Direct parent
detector support admits whole original tile components first. Residual tile
components are then evaluated against the completed selected parent bridge
union. Accepted detections are consolidated into parent-view coordinates per
tile configuration before running that same backend's tiled interpolation.
Rejected SAM runs never provide tile rescue. A tile observation rescued by a
selected parent SAM bridge remains a detector observation, with its admission
dependency recorded; consolidated tile bridges cannot rescue their own seeds.

## Selected output and retained evidence

SAM reserves two binary additive bridge slots per completed pass and provenance
scope: forward and backward. They contain policy-selected additions, and may
overlap. Direction means increasing or decreasing view-native frame index.
Seed detections, rejected runs, and raw context masks do not belong in these
selected layers. Empty slots follow the existing empty-layer convention.
The scope includes detector/model identity, SAM bundle identity, view, base
augmentation, full-frame or tile configuration, pass, and policy identity.

A compact indexed proposal bundle retains each run's raw masks and exact mask
ownership separately from the two directional unions. Two overlapping runs can
share voxels; rejecting one rebuilds the union from selected contributors.
Subtracting a rejected run from an already collapsed union cannot preserve
that ownership. Original endpoint masks, native crop transforms, input/gate
snapshot identities, completeness, diagnostics, and checksums remain in the
bundle so temporary generation workspaces can be retired.
The `xta.sam_proposals/1` bundle uses `manifest.json`, `index.json`, and
`masks.bin`; binary crop masks are bit-packed and independently compressed.
Tiled runs add the optional `xta.sam_run_tiles/1` evidence extension in those
same three files. Each tile retains full raw masks, crop/owner geometry,
original seed IDs, frame/injection status, and scores. Per-frame availability
marks unseeded owner cores as unknown; failed/incomplete tile attempts remain
explicitly incomplete evidence.
Selection still chooses original parent run IDs. Tile children do not create
extra independent prediction votes or public NRRD layers; the two directional
selected slots per pass/scope retain their existing publication role.

Integrated runs retain SAM artifacts in `OUTPUT/sam-artifacts.tar`, with
`sam-artifacts.tar.lock` serializing concurrent writers. Scope names are logical
members under `sam_interpolation/MODEL/VIEW/fullframe` or `tile_CONFIG`;
extrapolation uses the `sam_extrapolation/` namespace in the same container.
Stats, layer references and receipts identify them as
`OUTPUT/sam-artifacts.tar#/sam_interpolation/.../evidence` and sibling member
paths. Long names do not become loose output directories or per-frame files.

Active evidence and directional-store builders use short temporary stages.
Each sealed bundle or CVOL store is appended as a committed TAR/PAX transaction,
and its successful stage retires. JSON receipts commit independently as they
become available. This publication happens during the run; it does not wait for
full-run completion. Readers expose only committed transactions, so a torn final
append leaves earlier evidence readable. The next writer discards only the
uncommitted tail. Scope completeness and required selection-receipt checks still
apply; an archive with recoverable members is not a completed-run receipt.

Readers of already committed members can use an authenticated prefix while a
writer appends. A reader that arrives before that prefix is published wakes and
checks it again, avoiding a wait for the entire append. New members and complete
inventories still wait for the writer; inode, payload bounds and checksum checks
remain required. The wait condition is recreated after a process fork.

`SamEvidenceBundle`, bounded mask readers, CVOL readers, export and replay tools
read these references directly without unpacking. They also accept legacy bundle
directories. Seekable member streams enforce payload bounds and retain raw-byte
checksums, mask identities and ownership. Public NRRDs, telemetry and ordinary
run manifests remain filesystem files.

Evidence indexes retain canonical JSON and bounded authenticated reads. The
index ceiling is 128 MiB; manifests and other receipt limits remain unchanged.
Index writing streams already-normalized metadata, and reader freezing takes
ownership of its freshly parsed tree. Closed writer metadata is handed off before
reopening, avoiding overlapping full copies. The writer can also reuse one small
encoding for a repeatedly supplied, proven-immutable empty plane; every record
and payload packet is still published in its original order.

The run manifest records the configured backend, active state, SAM model, and SAM device pool.
Layer and gate receipts retain directional identities, selected policy hashes,
and parent-support fingerprints, including successfully empty selected support.

Proposal selection occurs before directional publication and before any
source-level plain-union optimization. Detector confidence, tracker score,
endpoint agreement, and family agreement are separate measurements. Later
source reconciliation and global cleanup may remove selected bridge support;
selected-bridge connectivity and final connection survival are separate facts.
Support survival retains separate before/after-global-postprocessing receipts.
If the actual `RawBBoxMaskStore` backing matches the final native grid, counts
use decoded bounding-box crops and the matching final-output windows. Each
covered final Z plane receives full binary validation once per invocation,
including empty interior planes and pixels outside occupied crops. An original
proposal's `native_transform` does not remap backing already projected to source
space. Dense backings and grids requiring restoration keep the shared dense
reader. Crop support counts do not expand connection-topology qualification.
The final connection audit currently certifies only a verified identity
Transverse mapping. Resampled and other transformed scopes report final
connectivity as `not_assessed` while
still measuring retained/removed bridge voxels; voxel survival alone does not
establish a connected final repair.

When plain-union reuse publishes the same final mask and all writers have
settled, the final audit can reuse the source audit's receipts. This requires
identity restoration, no configured global mask mutators, the same in-memory
output and selected references, and matching SHA-256 content for the mask,
selected backing, evidence, indices and selection metadata. The final audit's
256 MiB workspace limit and source-integrity assertions still apply; topology
receipts additionally require compatible audit budgets. Any failed guard runs
the ordinary final audit. Reused receipts are copied and marked with their
origin while retaining separate before/after stages and existing
`not_assessed` scopes. Runtime telemetry separates the reuse checks from the
audit itself: reading and hashing the mask and evidence has a real cost, so
reuse does not imply a measured throughput improvement.

The publication path projects each selected directional mask into
orthogonal/source coordinates before constructing its public layer reference.
Earlier SAM references carried `native_transform` as metadata without applying
that transform. This misplaced decomposed non-Transverse SAM layers and affected
consumers of those references, including custom source reconciliation. The
default plain union used its separately projected assembled volume. Native
proposal masks remain available for exact replay and tile admission.

Additive support is defined in the detector's working canvas. It need not remain
disjoint from detector support after source restoration: several processing
frames can contribute to one source frame, and smaller spatial output cells can
combine neighboring support. Low-quality NRRDs make that collapse more visible.
Inspect full-resolution additions and native selection receipts before treating
a low-quality layer's overlap as evidence that SAM painted detector pixels.

## Proposal quality policy

Existing external reconciliation policies remain usable. They inherit the
stock SAM bridge policy unless `build_reconciliation()` returns
`sam_bridge_policy` settings or a versioned proposal callback. Source-slab
`decide(block)` continues to operate after bridge selection.

Shipped online presets are `sam_conservative.py`, `sam_strict.py`, and
`sam_raw_candidates.py` under `XTA/examples/external_reconciliation/`. Select
one with `--reconciliation`, for example:

```text
--reconciliation XTA/examples/external_reconciliation/sam_strict.py
```

All three use source union. They differ in conservative SAM quality,
strict independent family agreement, or the explicit permissive raw-candidate
ablation. Omitting `--reconciliation` still uses stock conservative SAM quality.

The ordinary conservative stage first applies `--interpolation_min_radius` to each full
crop-space prediction. It labels 8-connected 2D foreground components and
removes an entire component when its maximum inscribed radius is at or below
the threshold. It does not erode a healthy component's outline, remove its thin
extensions, fill holes, or select only the largest component. A value of zero
disables this filter.

Radius computation preserves the same 8-connected component IDs and full-canvas
EDT maxima while restricting work to exact component bounding regions and using
exact rectangle maxima where applicable. If the summed component bounding areas
would exceed the full plane, it uses one full-plane EDT with a compiled linear
maximum reduction. Original canvas boundaries and foreground values remain part
of the radius contract; the optimization does not insert artificial background,
change the threshold comparison, or alter the filtered mask.

Filtering precedes acceptance-region and write-region clipping. A removed dot
outside either region remains visible in the raw diagnostics but cannot reject
the run or appear in selected output. A larger component that survives the
radius filter still records a containment violation if it touches or crosses the
declared acceptance boundary. When strict containment is enabled, clipping it
to a write region cannot hide that violation.
Original detector observations and stored raw tracker masks remain immutable.

### Tight-crop guard switch

`YOLO_TTA_SAM_TIGHT_CROP_GUARD=0` disables an inherited conservative
policy's acceptance-containment veto; `1` enables it. An unset value inherits the
policy: off for the new v6/v7 defaults, on for explicit legacy v2-v5. Accepted values
are `1`/`0`, `true`/`false`, and `on`/`off`, with case and surrounding whitespace
ignored. Active SAM interpolation validates and pins the setting at launch;
SDF interpolation and extrapolation alone ignore it.

```powershell
$env:YOLO_TTA_SAM_TIGHT_CROP_GUARD = '0'
```

This guard checks the fixed acceptance corridor inferred from the observations.
A prediction that grows beyond that corridor and later shrinks can fail the
entire run. Turning the guard off disables that quality veto for both whole
predictions and retained tiled halos. It also disables guarded rescue, whose
purpose is to reconsider containment failures. Endpoint agreement, topology,
unintended-contact checks, component filtering, and evidence completeness still
apply.

The switch does not enlarge the tracker context or choose a write-domain policy.
Legacy policies remain clipped to their old branch write masks. The v6/v7
defaults use the broader fixed context with the connected-path restrictions
below. Both exclude original detector observations. Disabling containment on
a legacy policy alone cannot recover growth beyond its old write masks.
An explicitly configured `sam_bridge_policy.strict_containment` takes
precedence. The effective policy and its hash record the choice; replaying a
complete saved policy does not consult the ambient switch.

### Connected branch selection (whole v6 / tiled v7)

Quality versions 6 and 7 certify each requested edge independently. A failed
daughter or a bad contact does not erase another qualified connection in the
same family. Selected receipts retain explicit successful and failed edge IDs,
original run owners, and an authenticated packed mask for each owner's selected
support. Every online, replay, export, and survival reader uses that same mask.
Radius filtering still precedes the connectivity analysis; no SDF pixels are
inserted into SAM output.

The defaults select `branch_write_domain="fixed_context"`,
`min_endpoint_recall=0`, `branch_crop_boundary_policy="retain_censored"`, and
disable strict containment and guarded rescue. Local connectivity, original-seed
identity, component-radius filtering, endpoint excess, unrelated-contact checks,
and attributable frame/pixel coverage remain enforced. Explicit v2-v5 policies
retain their recorded semantics. Existing proposal callbacks that return run IDs
inherit legacy selection semantics rather than silently acquiring branch behavior.

The branch path must connect its original observed endpoints through actual
seed-connected tracker support. Independent directional prefixes may meet in
the open gap. They cannot claim agreement merely by touching different parts
of a broad endpoint reference. Walk-back sessions retain their actual original
seed lineage. Only contributing owners appear in selected output.

`branch_write_domain="fixed_context"` permits growth beyond the old interpolated
silhouette corridor, within the tracker context fixed before inference. It
subtracts every original detector pixel, keeps only connected selected paths,
and rejects unrelated observed contacts. An observed same-family section
outside the old corridor can attach a path only where actual seed-connected SAM
support also saw it. Remote detector-only routes cannot certify a bridge.

With `min_endpoint_recall=0`, original detector endpoints serve as trusted
anchors: a small daughter need not reproduce half of a much larger merged
parent to establish a real connection. Unknown tracker frames remain invalid,
and a missing interior gap plane cannot be supplied by the endpoint reference.
For tiled tracking, selected pixels must have attributable owned-core coverage;
unknown regions and discarded halos never become published support.

`branch_crop_boundary_policy="retain_censored"` retains a certified connection
that touches an internal context border, with `extent_censored` and border
counts in the edge receipt. This certifies the observed gap connection, not
complete object extent outside the crop. `"reject"` retains the stricter
requirement for a larger crop. Neither setting predicts pixels outside the
fixed context.

The final-survival audit also uses the tracked-observation attachment proof.
An attachment removed by final reconciliation or cleanup cannot be silently
restored to certify survival. Nonidentity source transforms retain their
separate `not_assessed` connectivity limitation.

Branch admission reserves a conservative numeric workspace of 32 bytes per
group voxel plus 2 MiB; legacy topology keeps its existing estimate. A bounded
compressed owner spool uses only remaining current workspace credit and falls
back to recomputation when full. Label membership uses a bounded lookup instead
of NumPy's larger temporary vectors. Reader caches and packed receipt metadata
retain their separate explicit caps. A saved receipt's budget never authorizes
a new replay allocation, and publishing saved support decodes one bounded plane
at a time.

Within one integrity-checked evidence transaction, half the reader cache allowance
can retain packed component-filter results and their original diagnostics. Intrinsic
measurement lanes share these immutable products with the parent so connected
branch qualification and publication can reuse the same filtering decisions.
Other mask and topology products can borrow unused compact capacity. Both portions stay
inside the existing total allowance, expire with the owning outer transaction,
and preserve its complete payload checks before and after use. Cache pressure
causes recomputation; it cannot change a mask or relax a selection gate.

### Adaptive crop enlargement and replay

`YOLO_TTA_SAM_ADAPTIVE_CROP=1` enables adaptive replay for interpolation and
extrapolation. It defaults to `0` and is validated and pinned only when SAM work
is active. Each attempt uses a fixed crop throughout its complete tracker
interval; the crop does not move between individual frames.

After each complete attempt, raw mask contact with an internal group-crop edge
requires another admitted larger attempt for that original group. Contacts are checked across
the sequence, so growth that touches an edge and later shrinks is still detected.
For extrapolation, only the prefix reached before raw-empty termination can
trigger a retry. Contact with the declared canvas boundary cannot request pixels
outside that canvas; the canvas boundary is not necessarily a physical image
boundary after view transforms.

Whole-crop contact checks authenticate the complete compressed and packed mask,
its shape and its foreground count, then inspect boundary bits without expanding
the full binary raster. The caller's original crop and canvas geometry must
match the evidence. Tiled checks retain the dense overlapping halo union.
Each enlargement is based on freshly authenticated contacts from the latest
complete attempt, including an object that touches an edge and later shrinks.

This retry enlarges the group's outer context. In tiled mode it may change the
footprint layout, but it does not enlarge an individual child's fixed 1008-pixel
footprint or seed previously unseeded neighbors from predictions. Contact with an
internal child-crop edge remains separately diagnosed censoring. Neither group
enlargement nor a nonempty halo proves coverage beyond an owned tile core.

The retry enlarges the contacted sides and rerenders the needed image rectangles.
It restarts fresh tracker sessions using the exact same frozen original seeds and
complete intervals. A clipped prediction never becomes a new prompt. Attempt
identity and evidence remain separate, and the selected attempt undergoes the
operation's usual selection rules. Only the final complete attempt with resolved
outer-context contact replaces the original group; older attempts are not unioned
or spliced into it. Interpolation still checks contacts and topology across the
combined scope. Extrapolation retains its raw-empty/horizon rule.

The contacted sides initially request at least 64 pixels or 25% of their current
dimension, with at most a twofold area increase per step. If that proposal cannot
be admitted, a bounded sequence of smaller enlargements is evaluated independently;
tiled resource costs are not assumed to be monotone. Only the admitted candidate
is charged. Crops grow strictly within the declared canvas, so the chain is finite.
Full original-seed histories are replayed
at every step; a nearer predicted frame is never substituted as a seed.
Each attempt uses an owned image-cache context. The cache retires after its worker
mapping proof and iterator lifetime settle, while compact attempt evidence and
the resident predictor/feature cache remain available.

Each attempt still requires live resource admission. Session/CPU-wave allowances,
immutable image-cache limits, planner/decoder bounds and GPU admission remain
operative. The controller has no default single-attempt, extra-frame, pixel-work,
absolute crop-area or separate 2 GiB retry ceiling. Explicit caller-supplied caps
remain enforceable and are reported as such; free system RAM does not authorize
allocations outside the current lease.
Standalone callers without a live resource profile keep the existing conservative
2 GiB declared allowance. Reported available/admitted bytes describe that operative
budget, not a measurement of all free system memory.

Interpolation and extrapolation retain the original serial estimate when choosing
retry crops and charging scientific work. After the exact frozen attempt is
chosen, a live owner using the sealed tracker-scope API may fund more of its
independent SDK jobs concurrently. The execution wave is bounded by the actual
job inventory, configured execution slots, that owner's assigned CPU credit and the
explicit retry-memory cap. Retained images, production contracts, tiled assembly
and the serial evidence writer remain protected. Eager or unknown contracts,
uncredited callers, exclusive-transfer waves and insufficient credit retain the
previous execution wave. No future adaptive enlargement is speculated.

The serial eligibility peak is reported separately from the live execution base
peak (SDK, images and protected evidence) and hard limit. A larger SDK peak reduces
the residual available to completion loans
and mask packing; fresh completion/lookahead banks also fit inside the same hard
retry limit. The original parent debit and exact scope/source retirement proofs
remain in force. Diagnostic bounds with unproved ownership fields never grant
allocation permission.

Production extrapolation can overlap independent enlargement chains when its
`SamInterpolationContext` owns the live `SamInterpolationTracker`, the original
profile has live extra credit, more than one started worker slot and unresolved
group exist, and both effective global retry work caps (pixel-frames and tracker
frames) are unlimited. Groups needing no enlargement follow their existing
serial path. Finite global caps, standalone callers and unsupported resource or
runtime boundaries retain serial execution.

A dedicated pool is shared by every parent using the same tracker runtime.
Active child controllers are bounded by that runtime's actually started SDK
slots; the executor's configured-slot ceiling permits progressive startup to
add slots without replacing a live pool. Each pass also respects its requested
pending limit. Started-slot growth refills new capacity while older groups run;
a reduction prevents new admission without revoking active work. The pool owns
no runtime reference and shuts down when its settled runtime is collected.
Controller admission is nonblocking: a saturated pool uses the original owner's funded fallback
instead of parking that owner behind unrelated retry chains. Before accessing the
frozen source, each child acquires fresh physical and parent-pool credit with
exactly matching effective budgets and binds its authenticated profile on that
thread. Each admitted child applies the same funded execution-wave rule;
successive enlargements within its group remain serial. Compatible crop features
remain a soft worker preference, with fallback to other eligible workers. Declined child
requests retry after credit returns. After harvesting completed children, the
original parent may execute one already-deferred group alongside pending
children when its tracker wave is retired and the configured per-parent capacity
has a free slot. It uses its existing profile, without acquiring another grant
or advancing untouched inventory. Unsettled or possibly submitted requests cannot
be duplicated. A child failure signals cancellation promptly and preserves the
original exception even if the inline parent reports cancellation first. If no
bounded request can be admitted, the original owner still advances one group
with its existing authenticated profile. Private results
merge in original group order before final evidence import and selection.
Cancellation and failure retain mapped source ownership until accepted children,
their inputs and nested SDK owners settle. The retry ledger and trace report
execution mode, admitted or declined children, serial fallbacks and group spans.
`parent_inline_overlap_groups` separately counts parent work overlapping children.
Each retry reuses its already-generated selection receipt after checking the
producer-authored byte count, selection and policy identities, and immutable
evidence binding. This avoids selecting the same masks a second time.

Frame accounting includes all independent seeds, walk-back runs, and tiled child
jobs with their halos. Pixel-frame accounting charges the full retry, not just
the newly exposed area. Cumulative work and every attempt's geometry, contacts,
resource estimate and outcome remain in the retry ledger. Failed work is not
refunded. Runtime can increase when an object needs several enlargements.

If required enlargement cannot fit an operative bound, makes no progress, or
fails, the affected SAM scope raises a clear error and retains its diagnostic
evidence. It does not publish the unresolved clipped attempt as a successful
adaptive result. Original detector observations remain intact. Resolution here
means no internal **outer-group** contact; it does not certify extent beyond the
canvas or change the separate child-tile rules above. With adaptive cropping
disabled, growth and shrinkage remain limited to the original fixed context.

### Largest-island and one-direction continuation experiments

`tools/study_sam_largest_island.py` compares retained raw masks, the existing
component-radius filter, largest 8-connected island per frame, and both filters.
It filters each original seed run independently after assembling tile owners,
then unions independent runs. It preserves the raw evidence and reports both
all-endpoint recall and the experimental any-endpoint alternative. It does not
change production filtering or feed filtered masks back into tracker memory.

Largest area is not a lineage rule. Connected daughters remain one island, and
the largest disconnected island can change between frames. In particular, a
parent run following only one daughter fails the legacy all-held-out endpoint
test when another requested daughter is missed. The v6/v7 connected-edge policy
addresses that acceptance problem without replacing radius filtering with
largest-area selection.

`tools/evaluate_sam_extrapolation.py` provides `replay-holdout` and `generate`
subcommands. Replay measures single-seed directional prefixes while declaring
that the original crops were planned with future endpoints. Fresh generation
uses the original seed's bounding rectangle plus fixed padding and a bounded
outward horizon, without an opposite endpoint. Both modes compare raw,
score-filtered, largest-island, and cautious contiguous prefixes. An unavailable,
empty, or rejected frame stops a prefix; later recovery cannot restart it.
Growth and shrinkage are measured without a monotonic-area requirement.

These tools emit research evidence, not production-accepted bridge layers.
Fresh terminal-free tails require their own annotated quality validation;
tracker confidence and agreement with the detector are not ground truth.
These research tools retain their own prefix variants. Production extrapolation,
described below, uses raw-mask emptiness and the distance horizon alone.

### Legacy quality versions 2 through 5

These policies evaluate containment, held-out endpoint agreement, family agreement,
local topology and unintended contact using the filtered support. They require
held-out endpoint recall of at least 0.5, permits endpoint excess of at most 0.5
in its evaluation region, and requires the requested local connections without
unintended observed contacts. Filtering alone never supplies a whole-run rejection
reason, but a remaining leak, missed endpoint or broken required connection can
still reject a run or family. Both raw violations and effective violations after
filtering are retained, together with removed-component/pixel counts. It preserves
missing observations as unknown coverage and records 26-neighbor local topology
as its default connectivity. Strict independent family agreement
is off by default; when enabled, default family IoU and minimum compatible
slice IoU thresholds are 0.5 and 0.25. These are measured quality settings,
not changes to the main tile-admission rule.

For example, a source union policy can request stricter SAM agreement:

```python
def build_reconciliation():
    return {
        "name": "sam_strict_family",
        "mode": "union",
        "sam_bridge_policy": {"strict_family_agreement": True},
    }
```

A custom `select_proposals(context)` hook requires
`proposal_api_version=1`. The bounded read-only context exposes `scope`,
`group`, `runs`, `measurements`, `resolved_policy`, and lazy mask accessors.
`raw_mask` and `candidate_mask` preserve original evidence;
`effective_raw_mask` and `effective_candidate_mask` reflect the declared
read-only `mask_filter`. `group_mask` exposes fixed geometry. Return stable run ID strings or
`{"selected_run_ids": [...], "reasons": {...}, "name": "policy"}` (the last
two fields are optional). Selected IDs must belong to
that group and have complete structurally valid evidence. Custom policies may
change quality decisions, but cannot waive shape, transfer, coverage, identity,
or write-domain invariants.

`sam_bridge_policy="permissive"` is an explicit raw-candidate ablation. It
relaxes stock quality checks and disables component-radius filtering while
retaining structural validity. Label such
results as permissive candidates; they do not become stock-selected repairs.
The unchanged parent bridge tile gate still applies to their resulting support.
This legacy preset retains the narrow write domain, so it is not a pixel
superset of the new fixed-context v6/v7 output. Research comparisons distinguish
unqualified full raw support from these clipped candidate masks.

Quality policy versions 2 through 7 record the exact component filter in each selection
receipt. Online output, directional replay, previews and connection-survival
checks reconstruct the same filtered contributors. Older selection receipts
without a `mask_filter` retain their original unfiltered interpretation;
reselecting their immutable raw evidence creates a new versioned receipt.
The proposal callback interface remains `proposal_api_version=1`. Explicit
version-1 quality settings are rejected rather than silently reinterpreted.

For an explicit fixed-evidence quality experiment, a policy may set
`component_min_radius` to a nonnegative value. Its default, `None`, inherits
the recorded `--interpolation_min_radius` for each group. This is a policy
override, recorded in the selection/filter hashes; it does not modify the raw
bundle or add a second command-line radius control. Setting
`enforce_interpolation_min_radius=False` disables filtering, as used by the
raw-candidate ablation.

### Historical guarded rescue (explicit whole v4 / tiled v5)

These explicitly requested versions resolve to
`sam_conservative_guarded_rescue_v4` and
`sam_conservative_tiled_guarded_rescue_v5`. They were defaults before v24.0.2.
Use `sam_bridge_policy={"version": 4}` (or `5` for tiled evidence) to compare
their historical behavior. They are separate from the new connected-branch policy.

This is a second **selection** stage over existing complete raw evidence, not
another interpolation pass or a new model run. Ordinary stock selections are
kept first and cannot be displaced. Only previously unselected, complete
families with containment-only candidate rejection can enter rescue. Missing
observations, unavailable required tile cores, invalid transfers, incomplete
family inventories, and other safety rejections remain ineligible.

This initial rescue implementation supports **exactly one requested edge per
group**. Multi-edge groups are excluded with
`rescue_multibranch_attribution_not_supported`; the exclusion cannot be
overridden by policy thresholds. Ordinary stock decisions for multi-edge groups
remain unchanged. Rescue does not salvage individual branches or partition
branch ownership.

Both directions of the single requested edge need independently original-seeded
support over every interior slice of its declared write domain. Empty or unknown
coverage is not agreement. The stronger rescue gates are:

| Rescue measurement | Default requirement |
| --- | --- |
| Held-out endpoint recall | At least 0.90 |
| Endpoint excess in its evaluation region | At most 0.10 |
| Endpoint excess over the full context against original known family support | At most 0.10; also checked on full halos for tiled runs |
| Independent forward/backward agreement for the single edge | Aggregate IoU at least 0.95 and every compatible interior-slice IoU at least 0.90 |
| Independent connection support | Each direction must separately connect the original endpoints through the fixed local contract and attachments; agreement of large masks alone is insufficient |
| Each anchored component's outside/inside acceptance ratio | At most 0.05; detached components fail except for the bounded nonwriting allowance below |
| Maximum distance outside acceptance | At most 64 working-canvas pixels |
| Relevant local acceptance-boundary occupancy | At most 0.25 of the predeclared relevant acceptance-component boundary within the owned branch/original-family acceptance-margin neighborhood; the prediction's bounding box does not set this denominator |
| Allowed outer-context contact | Observed-family long-axis sides only, with original endpoint, evaluation, and write domains clear of every crop edge by 16 pixels |

Short-axis or ambiguous-axis crop contact fails. A working-canvas edge does not
prove a physical camera boundary. Crop-edge allowance never widens acceptance
or the write domain. Rescue still requires the requested local connection,
rejects unintended observed attachments, and rejects conflicts with previously
selected groups. Tiled runs check both owned cores and the separately filtered
full raw-halo union. Their raw halos are never published as additions.

A wholly outside, unanchored component can receive a bounded **nonwriting**
allowance. Each plane permits at most eight such fragments, each at most
512 pixels, with a combined area at most 1024 pixels **and** 0.2% of the
anchored foreground inside acceptance. Every fragment must stay within
64 working-canvas pixels of acceptance, avoid all crop edges, and have no
one-pixel contact with the write domain, endpoint evaluation regions,
or original observed references on the same frame. A larger, distant, crop-censored, protected,
or partly inside-acceptance detached component still fails. These fragments
cannot contribute published additions because they lie outside the protected
write domain. Their raw masks and full halos remain inspectable; this quality
allowance does not lower or replace the component-radius filter.

The original component-radius filter, write region, coverage rules and topology
checks are unchanged. Without additional live resource credit, each rescue
plane retains its 128 MiB workspace bound. An admitted production profile can
provide a larger workspace, recorded separately from the quality thresholds.
Planes are conservatively charged at 64 bytes per plane pixel within that
budget, with at most 256 support
or acceptance components; further scan limits can refuse evaluation. A group
refused before generation has no complete raw evidence to rescue. The patch
therefore does not promise to recover memory-refused groups or qualify a full
cluster workload.

For a strict comparator using the historical policy identity, an external source
policy can disable only the additional stage:

```python
def build_reconciliation():
    return {
        "name": "stock_without_guarded_rescue",
        "mode": "union",
        "sam_bridge_policy": {"version": 4, "guarded_rescue": False},
    }
```

Explicit `sam_bridge_policy={"version": 2}` selects legacy strict whole policy;
`{"version": 3}` selects legacy strict tiled policy. Automatic rescue does not
run for a custom `select_proposals` hook or a permissive raw-candidate policy.

Inspect `selection.json` under the scope output. Its `guarded_rescue` entry uses
`xta.sam_guarded_rescue/1` and records `quality_version`, `enabled`,
`attempted_group_count`, `rescued_group_ids`, `rescued_run_ids`,
`rejected_group_count`, `stock_selected_group_ids`, and
`stock_selected_run_ids`. Per-group and per-run entries preserve `stock_status`
and `stock_reasons`; a successful addition uses `guarded_rescue_selected`.
Endpoint, edge-agreement, spill, protected-domain, topology, and conflict reasons
remain inspectable. `generation.json` records `sam_policy_name`,
`sam_policy_version`, and the complete `sam_guarded_rescue` summary. The SAM
console line exposes `rescue_enabled`, `rescued_groups`, `rescued_runs`, and
`policy_version`, allowing the active cluster policy to be checked directly.
Spill-plane receipts include `nonwriting_satellites`, their counts/area and
anchored-inside denominator. Allowed fragments use
`bounded_nonwriting_satellite_allowed`; aggregate budget failure reports
`rescue_nonwriting_satellite_plane_budget`.
`independent_direction_topology` records the separate directional connection
checks; a failure reports `rescue_independent_direction_local_connection`.

Previously saved selection receipts keep their recorded versions and selected
contributors. Applying current stock selection to their immutable raw bundle
creates a new v6/v7 receipt when the saved edge contracts support it; it does
not rewrite the historical decision. A
changed parent selection can change ordinary tile admission, so dependent
consolidated evidence still needs its upstream fingerprint checked or regenerated.

## Replay and validation boundaries

### Concurrent pressure checks and partial-run diagnostics

Exercise the producer, evidence consumer and resource handoff together before a
long cluster run. The pressure tests deliberately block cache finalization and
mask packing while tracker work is ready; they check progress, bounded ownership,
unchanged pixels and cleanup on success, cancellation and failure.

```text
python -B -m pytest -q tests/test_sam_pipeline_pressure.py tests/test_sam_image_finalization.py tests/test_sam_endpoint_competition.py tests/test_sam_azimuthal_tap_gather.py --basetemp ../Scratch/Experiments/sam-pressure-check/pytest -o cache_dir=../Scratch/Experiments/sam-pressure-check/pytest-cache
python -B tools/benchmark_sam_azimuthal_submission.py --output-dir ../Scratch/Experiments/sam-pressure-check/render
```

The GPU submission benchmark reserves `Scratch/Temp/GPU_LOCK`, warms the local
CPU/GPU, and compares the explicit 32-row reference with the bounded default at
2048-sized geometry, both alone and alongside evidence consumers. It records
pixel checks, host submission time, GPU timings and memory. These measurements
expose launch starvation that an isolated small-image benchmark can miss; they
do not predict target-cluster throughput or establish segmentation accuracy.

The health checker can inspect a run before it finishes:

```text
python -B tools/analyze_pipeline_health.py RUN_TELEMETRY_DIRECTORY --output ../Scratch/Experiments/sam-pressure-check/health.json
```

One invocation analyzes the available raw-log prefix and exits. Rerun it to
inspect later progress; `--until-seconds 600` replays the first ten minutes from
retained logs. It distinguishes fresh GPU samples from repeated trace flushes,
reports sustained idle-device and completed-result backpressure observations,
and keeps missing clocks, physical-device mappings and open phases explicit.
Repeated attention observations are not separate failures. A final telemetry
flag alone does not establish scientific completion, and retained VRAM alone
does not establish a leak. New runs publish timestamped `gpu.stage.live` and
trusted `gpu.device_identity` records for ownership attribution.

### Short component-projection loop

To retain a real SAM projection case during an ordinary run, add these arguments
with a persistent capture directory. Specify an actual view name or glob; the
historical default selectors target tilted Azimuthal views.

```bash
--capture_component_replay "$DEST_DIR/component-replay" \
--capture_component_views 'radial_coronal_patch_u4_h1*' \
--capture_component_limit 1
```

Capture is opt-in, limited to one nonempty component per selected view and the
existing 4 GiB total quota. It retains the checksummed native CVOL, full view
recipe, output shape and SAM publication metadata before queue admission.
Archive-backed components copy only their required members. A completed capture
can be replayed immediately without waiting for the full run to finish:

```bash
python -B tools/replay_component_projection.py "$CAPTURE" \
  --sam-production --output "$FRESH_REPLAY_DIR" --scratch /tmp --workers 6
```

`CAPTURE` is the emitted case directory or its `manifest.json`; output must be
fresh and separate from the capture. This runs the production SAM source-projection
path on CPU for Cartesian, Radial, Spherical or Azimuthal inputs, without video,
YOLO or SAM inference. The new route defaults to a 300-second worker deadline
and thread-stack dumps every 30 seconds; `--timeout-seconds` and
`--stack-dump-seconds` override them. Logs and metrics survive timeout; temporary
workspaces are removed unless `--keep-work` is set. Existing replay routes retain
their previous defaults.

The report records input identity, stage wall/thread CPU time and output SHA256.
SAM replay also records `resource_observations`: sampled process-tree RSS and
owned workspace file lengths/allocated blocks, with active-phase context and
incomplete-sample counts. Sampling is bounded and asynchronous; unsupported
values remain null. Latest observations are preserved on failure or timeout.
File lengths are not RAM, shared mappings can appear in multiple RSS totals,
and brief allocations or unlinked mapped files can be missed. These are observed
peaks, not proven reservation bounds; phase context does not assign ownership.
Supply `--reference-store` for an exact voxel comparison against an existing
projected CVOL. Without it, `all_exact` is null: a completed replay alone does not
establish output equivalence. This isolates source projection, not queue waits,
NRRD compression or fleet throughput. The saturated-queue assembly tests cover
the ordering bug separately. Use warmed, quiet target hardware for performance
comparisons and full runs for final makespan validation.

Diagnostic `.ready` snapshots contain logs and telemetry, not these replay inputs.
Already retained evidence can drive the selection tools below; an exact projection
replay additionally needs the captured view recipe and native component.

### Evidence selection and scope replay

Fixed-evidence replay changes proposal selection without loading the detector
or SAM model. It requires the same complete, structurally valid proposals,
planning inputs, and gate snapshot. Changes to crop geometry, model,
preprocessing, group inventory, missing frame coverage, or admitted detector
observations require generation again.
Replay resolves crop mode from saved evidence, rather than the current
environment. Legacy bundles without tiled evidence keep whole generation
geometry. Historical v2/v3 selection receipts keep their recorded decisions;
an explicit new selection uses its resolved v6/v7 or requested legacy policy
identity. Changing crop mode requires generation.

Changing parent bridge selection can change tile admission and therefore the
downstream proposal inventory. A changed upstream gate-support fingerprint
invalidates affected downstream evidence or requires regeneration. A comparison
on the original frozen proposal set must identify that snapshot; it is not a
fresh end-to-end run under a different admission result.

Proposal comparison accepts a retained bundle directory or a live archive
reference directly, without loading detector or SAM runtimes. Portable export
optionally copies that bundle to a fresh directory or archive prefix:

```text
python tools/export_reconciliation_evidence.py --sam_bundle BUNDLE --output PORTABLE_BUNDLE

python tools/compare_reconciliation.py --sam_bundle PORTABLE_BUNDLE \
  --output FRESH_COMPARISON --memory_mib 256

python tools/compare_reconciliation.py --sam_bundle PORTABLE_BUNDLE \
  --sam_policy stock strict permissive --output FRESH_ABLATION
```

`BUNDLE` can be a quoted reference such as
`"C:/XTA/run/sam-artifacts.tar#/sam_interpolation/MODEL/VIEW/fullframe/sam_SCOPE/evidence"`.
The saved stats provide the exact member path. After transferring the container,
the reference uses its local physical path and the same logical member path.

The SAM comparison defaults to stock and strict-family policies. Permissive
raw-candidate comparison is opt-in. Alternatively, use `--policy POLICY.py`
for external policies; `--policy` and `--sam_policy` are mutually exclusive.
Multiple bundles may follow `--sam_bundle`. Every destination must be fresh.
Portable export leaves the immutable three-file raw bundle byte-identical.
When an online selection receipt is available, it also retains
`online_selection.json` and `export.json` with verified bundle fingerprint,
selected IDs, and checksum. This keeps publication measurements and the online
policy decision reviewable without increasing file count per proposal.

Replay writes two selected binary NRRD slots per pass and scope, with a
`xta.sam_directional_replay/1` manifest. These are view-native canvases in
generic right-handed coordinates with recorded spacing, direction, and original
scope transforms. The manifest declares `source_grid_projected=False`.
A spatially or temporally resampled canvas needs a separate verified projection
before comparing source-grid layers. This scope replay does not rerun source
voting, global cleanup, detector generation, or the upstream tile gate.

Use `--gate_snapshot FINGERPRINTS.json` with a current input/upstream fingerprint
map when validating downstream dependencies. A changed gate snapshot returns
`regeneration_required` and exit code 2. `--frozen_evidence` explicitly permits
a diagnostic comparison of the original proposal set and records that status;
it does not certify a complete run under changed upstream inputs.

`tools/diagnose_sam_interpolation.py` provides a bounded real-data comparison
entrypoint. Run `--help` for its staged commands. It limits case search to five
tries and preserves detector inputs, case identities, timing, raw/selected
support, and reviewable overlays in a persistent output directory. Scientific
comparisons and qualification receipts belong in task Scratch, not in the
installed package. Quality claims require measured evidence: detector endpoint
agreement alone is not independent ground truth.

The additional improvement tools keep research separate from production:

| Tool | Purpose |
| --- | --- |
| `prepare_sam_holdout.py` | Register detector-only cases and input identities before opening held-out annotations |
| `study_sam_fusion.py` | Study frozen fusion variants over retained selected tracks without additional inference; prior diagnostic cases are development data |
| `qualify_sdf_alignment.py` | Compare predeclared SDF anchor, transport, and area experiments while leaving production SDF unchanged |
| `heatsoak_sam_benchmark.py` | Reserve the local GPU and heatsoak CPU/GPU before scheduling sanity measurements; preserve thermal receipts |
| `qualify_sam_feature_cache.py` | Compare real-model cache/dispatch masks and scores against fixed baseline jobs, recording warm local timings and source identity |
| `qualify_sam_precision.py` | Preflight FA3 kernels before a full job, replay complete real tracks under SDPA/FP8, and report paired masks, scores, backend provenance and service time without asserting biological quality |
| `benchmark_sam_encoder_batches.py` | Measure encoder/feature-projection batches separately from recurrent tracking, with explicit device profile and attention backend |
| `analyze_sam_grouping_inventory.py` | Count metadata-only grouping candidates and repeated encoder inputs by worker; candidate counts are not speed or quality qualification |
| `evaluate_sam_holdout.py` | Freeze pre-registered fusion outputs before reading held-out annotations, then score the declared source-canvas comparisons |
| `audit_sam_context.py` | Audit original observation masks without labels to identify artificial diagnostic ROI-boundary censoring |
| `qualify_sam_tiled_integration.py` | Exercise the actual assembly/runtime seam on a bounded native-source fixture; its canvas is explicitly distinct from ordinary CLI processing-cube preparation |

Run each tool's `--help` for its input/output contract. Nearest-anchor SAM fusion
and subpixel SDF transport remain research variants after their separate
development and held-out comparisons. They do not change production fusion,
the backend selector, stock defaults, or tile gate. Their measured tradeoffs
and local timing receipts belong in the improvement report.
An artificial diagnostic ROI can clip additional observed structures before
planning. Such a case is context-censored and does not establish full-view
family completeness. Preserve its original preregistration and report the
label-free context audit; do not use it to relax production containment policy.

## Paired crop-strategy follow-up

The research follow-up compares two strategies on the same observation anchors
and source images: resize the whole native context crop to the model canvas,
or run independent overlapping native tiles and stitch their predictions back
into that context. The tile comparison uses no cross-tile propagation. A
prediction from one tile does not become a seed in another tile.
Raw strategy labels are `whole_crop` and `independent_tiles`;
`identical_crop_control` identifies a family that fits one tile.
Both strategies use the same full native family crop. Tiles add no surrounding
pixels outside it. The frozen tile geometry caps each side at 1008 pixels,
uses a stride of at most 752, and assigns overlap ownership at fixed midpoints
with a minimum 128-pixel interior halo. A family fitting one tile is an exact
crop control. A tile with no original seed has unavailable support, rather than
a successful empty prediction.

| Research tool | Role |
| --- | --- |
| `compare_sam_crop_strategies.py` | Run the paired SAM experiments on matched cases |
| `sam_crop_strategy_geometry.py` | Define native/model transforms and inverse tile geometry |
| `sam_crop_seed_diagnostics.py` | Distinguish native seed coverage from survival after model-canvas resizing |
| `report_sam_crop_strategies.py` | Measure matched outputs and publish reviewable comparisons |
| `analyze_sam_crop_strategies.py` | Analyze frozen paired outputs offline with their declared geometry |
| `sam_crop_quality.py` | Compute research diagnostic eligibility; this is separate from versioned stock production acceptance |

Keep crop planning and seed diagnostics independent of withheld annotations.
Native coverage and model-space seed survival are separate measurements:
resizing can change a thin seed even when the native crop contains it. Account
for overlap, edges, padding, and inverse transforms when comparing stitched
tile predictions. Report raw and selected support and failures as well as
accepted repairs. Results are paired experimental evidence; these tools do
not add a production interpolation backend or change production crop defaults.
Raw tracker-strategy outputs must be labelled as such. They do not establish
production proposal acceptance, tile admission, or a new independent accuracy
holdout when scoring reuses an annotation from an earlier experiment.
The quality helper reports its own frozen diagnostic contract, including a
default 16-pixel margin and 512 MiB topology allowance. These differ from the
baseline planner contract and its 256 MiB group budget; production live resource
profiles are accounted separately. Diagnostic eligibility is
not a production policy receipt.
The seed diagnostic uses a CPU raster proxy for the pinned mask-conditioning
resampling chain. It can report changed pixels and components without asserting
GPU operator equivalence, neural information loss, or subsequent tracker
accuracy. The model's echoed native anchor is not a measurement of its internal
conditioning mask.

The experiment runner has separate endpoint and SAM stages:

```text
python tools/compare_sam_crop_strategies.py endpoints --images SOURCE.u8.dat \
  --shape T H W --frame-start FIRST_FRAME --frames ANCHOR_A ANCHOR_B \
  --detector DETECTOR.pt --imgsz 3072 --conf 0.15 --output ENDPOINT_DIRECTORY

python tools/compare_sam_crop_strategies.py run --plan STRATEGY_PLAN.json \
  --model SAM_BUNDLE --repeats 2 --cache-mib 512 --output EXPERIMENT_DIRECTORY
```

The runner validates the frozen plans and seed geometry before model work and
uses a fresh predictor/cache for each strategy. Its indexed raw output retains
halo masks, object probabilities, timing, and cache counters separately from
the stitched comparison. These CLI commands are research entrypoints.
The whole-crop run passes the native rectangle and original seed to the existing
SDK; SDK preprocessing performs model resizing. The CPU seed proxy is measured
separately and never replaces a tracking seed.

The geometry planner reads saved full native endpoints and prior review regions:

```text
python tools/sam_crop_strategy_geometry.py --endpoints ENDPOINTS.json \
  --prior-reviews PRIOR_REVIEW_DIRECTORY --output FRESH_PLAN_DIRECTORY
```

Prior review regions select original full components; they do not trim their
silhouettes to the earlier artificial ROI. The default boundary diagnostic
uses a fixed two-native-pixel tolerance. Source coordinates, stitched support,
and unavailable tile ownership must remain explicit in the paired report.

The report tool runs a matched legacy SDF reference and then builds the paired
report from frozen SAM, SDF, analysis, and quality artifacts:

```text
python tools/report_sam_crop_strategies.py sdf-reference --experiment EXPERIMENT_DIRECTORY
python tools/report_sam_crop_strategies.py report --experiment EXPERIMENT_DIRECTORY \
  --output EXPERIMENT_DIRECTORY/REPORT.html --workers 1
```

Keep the original annotation identity, native frame/canvas, plan hash, and
repeat selection with every result. Report full-crop boundary diagnostics and
review-region metrics separately; a trend on one domain does not imply the
same trend on the other.
Use the task's persistent Scratch directory for every generated destination.

The narrow two-tile follow-up reuses the same 2065-by-659 whole crop and original
seeds. Two 1260-by-659 native tiles overlap by 455 pixels and use the existing
1008-wide model input, giving a horizontal scale of 0.8. This is a research
plan variant with no cross-tile propagation. It retains prior whole-crop and
three-tile evidence instead of repeating their experiments.
`prepare_sam_crop_two_tile.py --prior-plan PLAN --output NEW_EXPERIMENT_DIRECTORY`
checks the frozen family/geometry before preparing the variant. The runner's
research-only `--family FAMILY_ID --strategy independent_tiles` selectors limit
new inference to the requested family and strategy; they add no production flag.
`compare_sam_crop_zoom.py --baseline PRIOR_EXPERIMENT_DIRECTORY --variant
NEW_EXPERIMENT_DIRECTORY --label ANNOTATION.txt --repeat 1` compares the three
retained strategies offline. It rebuilds unions from retained contributors
when replacing a family's result, preserving support shared with other families.
The reporter's `two-tile-report --experiment NEW_EXPERIMENT_DIRECTORY` stage
writes a separate three-way report in that new directory and preserves the
earlier paired-study reports.

Further crop/frontier planning and extracted tile-gating policies remain
separate changes to the declared rollout.

LTA's opt-in [dynamic crop backend](lta_dynamic_crops.md) has its own execution
contract. Each TTA attempt shares a fixed family context across independently
seeded sessions and does not inherit LTA's between-window crop updates.

## Bounded selected-mask publication

Internal native-plane publication validates each selected owner's ordered frame
tuple once, then uses binary lookup and the saved cyclic aliases for each output
frame. The lookup keeps O(selected owners) descriptors, borrowing existing frame
tuples and address maps. It does not create a full frame-to-owner table or skip
candidate-mask reads, filtering, corruption checks, or their order.

The descriptors are charged within the existing 32 MiB reader allowance, with an
8 MiB descriptor ceiling. Authentication binds the active reader, original
bundle metadata and records, original validated geometry cache, private ordered
selection identifiers, filter snapshot and output shape. Replaced, mutable,
unbound or noncanonical inputs use the existing traversal. Missing or rebound
geometry caches cannot reuse a stale lookup.

This publication creates the native interpolation result used by subsequent
extrapolation. It is separate from asynchronous source-grid backprojection.

Admitted full-frame parents record their immutable SAM layer requests locally
until native preparation, merging and cleanup finish. They leave the ordinary
SAM resource scope and attempt nonblocking submission to the bounded projection
queue. A completed parent result owns any unadmitted requests, freeing its
preparation thread. The existing scheduler drain admits them incrementally;
unresolved futures and a finite heartbeat keep admission wakeable. Unfinished
parents rotate through the bounded drain so an older blocked output cannot
hide another parent's dense retirement or failure.

The staged inventory is bounded by the configured directional outputs of the
finite planned parent set. It retains sealed persistent artifact paths and
detached metadata, not open decoders, dense arrays or SAM resource profiles.
Persistent artifacts still consume filesystem/RAM capacity. Accepted count,
source-byte and working-byte limits are unchanged; queue admission is atomic,
and each dependency completes only when its actual projection completes.
Immutable path proofs allow the existing dense ledger to retire independent
backing before admission, still respecting all outside readers and tile/debug
exclusions. Outstanding image/tracker holds still delay SAM credit return.

Abort cancels unadmitted requests, including those returned by parents finishing
during teardown; accepted readers retain their ownership until they settle.
Output ordering and terminal readiness are unchanged. Direct/custom submitters
keep their blocking behavior. Deferred admission emits one
`sam_projection_deferred_admitted` receipt with `deferred_seconds`, source bytes
and working bytes; rejected probes do not create repeated phase traces. The
legacy `projection_admission` phase still measures blocking direct submissions.

## Bounded Azimuthal GPU image batches

Fresh upright Azimuthal image-cache transactions can render up to four adjacent
native requests together when their physical crop and inverse transform match.
Donor reuse, cyclic aliases, fragmented holes and incompatible requests retain
the scalar path. The chooser tries four, three, then two images using the actual
device's retained cache storage after source admission. A candidate must fit the
existing host/GPU allowances and must not require more native gather-loop
iterations than its matching scalar calls. This guard prevents a smaller memory-
limited row block from making a large batch issue more gather work.

The batch uses the existing tap arithmetic, native gray8 quantization and affine
grid. It borrows the existing source and whole-GPU lease, writes ordered results
directly into the admitted private mmap, and advances only after exactly-once
completion. It adds no owner, executor, source copy, credit or publication point.
Failure still retires the renderer and rolls back the complete private cache;
normal renderer retirement still precedes disk flush and descriptor commit.
`YOLO_TTA_SAM_AZIMUTHAL_BATCH=0` selects the established scalar loop for controlled
comparison. The helper's static implementation hash is recorded in image
sampling identity and execution provenance; runtime batch choices do not alter
pixel identity.

`sam.gpu_images.frames` and `.pixels` count outputs across both routes.
`.batch_frames`, `.batches` and `.batches_size_N` identify completed batch work.
`render_enqueue_host_seconds` includes both routes, while
`batch_render_enqueue_host_seconds` is its batch subset. The separate
`batch_copy_and_sink_host_seconds` includes D2H completion and private mmap
copying; `crop_copy_wait_host_seconds` remains the scalar D2H measurement.
The coalesced `sam.gpu_images.batch.cuda:N` receipt records completed batch size,
row block, gather-loop comparison and retained/planned bytes. A completed batch
receipt does not mean the entire cache descriptor has been committed. Image
phases include `batch_render`, `batch_host_cache_copy` and `batch_complete`;
interpret retained phase gauges alongside current GPU stage ownership.

## Context geometry and crop scheduling

The safe planner correction bounds the complete endpoint silhouettes swept
between their original observed anchors. Bounding only the observed endpoint
boxes could clip the intermediate translated shapes. The corrected context is
fixed before tracking, includes observed continuations and the existing margins,
and remains limited by the actual working canvas and resource caps. Raster
rounding stays tied to the legacy origin, preserving existing contract pixels
where they were already covered. The planning receipt records
`xta.sam_fixed_family_swept_context/2`, observed/swept/unclipped/clamped boxes,
canvas-clamped sides, margins, and charged memory. This changes generation
geometry; fixed-evidence selection cannot reproduce an omitted context.
Resource caps can refuse a larger corrected family. Re-clipping the corrected
sweep would hide intended acceptance/write space. Refusals must remain in
evaluation denominators; admitted-only scores do not represent full inventory
coverage. Run-specific admission comparisons and measurements belong in the
Scratch XTA History directory.

Tiled execution on one worker now finishes each exact crop queue within the
existing bounded parent cohort before switching crops. Independent endpoint
sessions reuse immutable visual features, not tracker state. Multi-worker crop
waves, job inventory, seeds, ownership, and quality thresholds retain their
existing contracts. Explicit diagnostics distinguish current working canvas,
native-view dimensions, context clamping, and actual tile/crop counts.
`xta.sam_crop_contacts/1` counts raw and filtered contacts with crop and declared
working-canvas edges separately, deduplicating shared corner pixels. Inconsistent
canvas metadata stays unknown. A working-canvas edge is not proof of a physical
source-image boundary, and no contact category waives containment or changes
selection.

The maintained development tools separate protocol, inference, scoring, and
reporting:

| Tool | Development role |
| --- | --- |
| `sam_outer_crop_geometry.py` | Build declared context/acceptance research variants by integer embedding of fixed contracts |
| `prepare_sam_outer_crop_experiment.py` | Extract exact native input windows without reading annotations |
| `prepare_sam_outer_crop_protocol.py` | Lock data-only recipes and seal geometry before inference/label scoring |
| `run_sam_outer_crop_experiment.py` | Run sealed tagged/revised/context variants with attributable raw evidence |
| `run_sam_outer_crop_stress.py` | Run separate single-family context stress controls and reuse retained controls |
| `analyze_sam_outer_crop.py` | Score raw, radius-filtered, write-limited, and selected support separately, keeping refusal/unknown outcomes explicit |
| `derive_sam_acceptance_evidence.py` | Derive an acceptance-only fixed-raw diagnostic with preserved source masks/scores and explicit changed-geometry attribution |
| `qualify_sam_tiled_schedule.py` | Check exact-mask/score ABBA scheduling equivalence and separate startup/work timings |
| `report_sam_outer_crop.py` | Present already scored artifacts and provenance without selecting masks or computing new accuracy |
| `generate_sam_outer_crop_sdf.py` | Generate unchanged CPU SDF references from sealed observations, without reading images/labels or replacing prior references |

The wider-context and acceptance-only variants remain experiments. The latter
remeasures retained complete raw support under a changed declared acceptance
region; it is not a fresh pipeline-equivalent replay. Preserve the tagged
baseline, original recipe seals, source/frame/seed identities, and raw reuse
attribution. Those research geometry variants do not alter central component
filtering or the ordinary tile gate. Their historical guarded-rescue successor
is described above; the defaults use the connected-branch policy.
