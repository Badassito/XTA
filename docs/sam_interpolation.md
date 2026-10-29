# SAM interpolation and extrapolation in TTA

This guide describes current TTA behavior and its retained-evidence contracts.
Use the [runtime and SDK guide](sam_sdk.md) for startup, scheduling, image
preparation and resource ownership; [parent preparation order](sam_parent_prepare_order.md)
for admission fairness; and the [experiment and historical policy reference](sam_history.md)
for research protocols and explicitly selected legacy policies. Quality-policy
numbers such as v6/v7 are policy identities, separate from XTA release versions.

`--interpolation_backend sdf|sam` selects interpolation, with `sdf` as the default.
Choose one interpolation backend for a run. `sam` uses the local mask-conditioned LTA tracker
to propose image-guided additions between detector observations. A rejected SAM
proposal receives no SDF fallback. `both` is rejected during argument parsing.

In v25, `--enable_2D_hole_fill sam` optionally fills SAM masks before recurrent
memory encoding and publication. Without that stage selection, holes remain in
SAM masks. The SDK's area-limited output-only filling stays disabled; the opt-in
uses the shared exact 2D fill at the memory boundary, including a CPU transfer
when SAM runs on CUDA. `--enable_2D_hole_fill yolo` independently selects detector
mask filling. These flags do not change SDF interpolation or explicitly selected
3D postprocessing. See [v25 migration](topology_and_cpr.md).

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

The [SAM runtime and SDK guide](sam_sdk.md) describes device handoff,
predictor startup, parent admission, image preparation, scheduling and their
ownership and telemetry contracts. The model/device syntax above applies
to those execution paths.

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

The [experiment reference](sam_history.md#largest-island-and-one-direction-continuation-experiments)
retains these research filters and prefix variants. Production extrapolation
uses the [raw-empty/horizon rule](#sam-extrapolation).

### Legacy quality versions 2 through 5

The [historical policy reference](sam_history.md#legacy-quality-versions-2-through-5)
retains their thresholds and stricter-family example. Explicit legacy policies
keep those semantics; current defaults use [connected branch selection](#connected-branch-selection-whole-v6--tiled-v7).

### Custom proposal callbacks and filter receipts

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

The [historical policy reference](sam_history.md#historical-guarded-rescue-explicit-whole-v4--tiled-v5)
retains the explicit v4/v5 selection contracts, thresholds, resource bounds
and receipt fields. Current connected-branch defaults disable guarded rescue.
Reselecting old raw evidence creates a new receipt and must still validate
dependent tile-admission fingerprints.

## Replay and validation boundaries

### Concurrent pressure checks and partial-run diagnostics

See the [runtime validation procedure](sam_sdk.md#concurrent-pressure-checks-and-partial-run-diagnostics)
for pressure tests, warmed GPU submission measurements and partial-run health
inspection. These checks keep runtime progress separate from scientific quality.

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

The [fixed-evidence improvement studies](sam_history.md#fixed-evidence-improvement-studies)
index research tools and their holdout, context-censoring and validation boundaries.

## Paired crop-strategy follow-up

The [experiment reference](sam_history.md#paired-crop-strategy-follow-up)
retains the paired whole-crop/tile protocols, frozen geometry, two-tile
follow-up and attribution requirements. These studies do not change the
[active crop-mode contract](#experimental-sam-tracking-crop-mode).

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

The [outer-context research tools](sam_history.md#outer-context-and-acceptance-experiments)
separate protocol, inference, scoring and reporting. Their changed-geometry
and acceptance-only diagnostics remain distinct from this planner contract.
