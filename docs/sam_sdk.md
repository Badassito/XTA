# SAM runtime and SDK contracts

This companion to [SAM interpolation and extrapolation](sam_interpolation.md)
describes current model execution, scheduling, image preparation and resource
ownership. Model role/device syntax and examples remain in the
[models and devices section](sam_interpolation.md#models-and-devices).
See [attention precision](sam_interpolation.md#sam-attention-precision) for
SDPA/FA3 selection, [bounded reuse](sam_interpolation.md#bounded-reuse-and-resource-controls)
for cache controls, and [bounded Azimuthal batches](sam_interpolation.md#bounded-azimuthal-gpu-image-batches)
for image-batch publication. Research protocols and legacy policy details
are in the [experiment reference](sam_history.md).

## Device handoff and predictor ownership

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

## Parent admission and image funding

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

## Detector backing, checkpoints and restoration

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

## Parent concurrency and memory partitions

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

## Predictor sessions and progressive startup

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

## Tracker dispatch and evidence consumption

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

## Runtime trace interpretation

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

## Family dispatch and balance guard

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

## Image identity and canonical gray capture

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

## CPU and GPU image construction

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

## GPU image turns and source residency

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

## Pinned SDK prepared input

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

## Extrapolation image cohorts

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

## Concurrent pressure checks and partial-run diagnostics

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
