# SAM interpolation in TTA

Version 25.0.0 adds `--interpolation_backend sdf|sam`. The default is `sdf`.
Choose one backend for a run. `sam` uses the local mask-conditioned LTA tracker
to propose image-guided additions between detector observations. A rejected SAM
proposal receives no SDF fallback. `both` is rejected during argument parsing.

The initial SAM scope is native Transverse at angle zero, for full-frame and
consolidated tile observations. Unsupported requested views, tilts, or wrapping
are errors. SDF retains its existing view support, iterative passes, and layer
decomposition. `--interpolation_distance 0` disables either configured backend
and avoids initializing unused SAM assets.

Here, native Transverse means the active detector/working-view canvas. It
includes TTA's existing processing-volume transform, including any stack
resampling performed during canonical volume preparation. Its frame index need
not equal a raw input-video slice index. SAM renders and tracks that exact
detector canvas, as SDF bridges do; scope metadata preserves native processing
frame addresses and the `native_transform` back to the source grid.

## Experimental SAM tracking crop mode

`YOLO_TTA_SAM_CROP_MODE` selects `whole` (default) or `tiled` for active SAM
tracking. It is validated before heavy startup and pinned for that launch.
Values are trimmed and case-insensitive; other active values fail clearly.
SDF and `--interpolation_distance 0` do not read this unused environment setting;
their recorded mode is null and no SAM crop resources start. This is an
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

Whole tracking and historical bundles without a mode use conservative v2.
Tiled tracking uses the explicit conservative v3 contract with the same numeric
thresholds and component-radius filter. Endpoint, family, and topology decisions
use filtered owned-core support. A separately filtered full native union of that
original seed's tile halos supplies an additional containment veto. That halo
union is quality evidence and never output support, so a discarded halo cannot
hide a leak. Explicit v2 settings cannot select tiled evidence.

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
  --enable_cartesian transverse --angle 0 --interpolation_backend sam --save nrrd summary

python -m XTA --mode tta --input /data/volume.mkv --output /data/cpu-sam-run \
  --device cpu --sam_device 0 \
  --model cpu:/models/openvino sam:/models/sam_bundle \
  --enable_cartesian transverse --angle 0 --interpolation_backend sam --save nrrd summary
```

These are argument examples; substitute verified detector and SAM assets and
the appropriate detector channel format for the actual workload. CPU detector
selection does not require a `gpu:` detector artifact to run SAM on CUDA.

SAM plans the observed-anchor jobs before rendering images, acquiring GPUs, or
loading a predictor. Empty or exhausted plans perform none of those operations.
When SAM and detector devices overlap, admission requires a successful detector
asset-retirement acknowledgement. A CPU detector route or an entirely separate
SAM device pool can acquire its devices independently. Failed retirement or
unreleased model residency must not advertise a shared device as available.

Each admitted SAM device owns one isolated persistent predictor and at most one
endpoint job at a time. A free worker takes the next bounded job. GPU/result
consumption for different scopes is serialized by its owner, while their CPU
planning, rendering, and reconciliation can overlap. Evidence commits retain
stable run IDs, lineage, selection, and directional pixels despite out-of-order
worker completion. Packed bundle offsets, checksums, and timings can differ
between fresh generation attempts.

Multiple devices interleave distinct crop queues and prefer a crop's previous
worker while lending idle workers to other crops. Completed masks are packed
and released promptly. Single-device requests retain grouped order. These
scheduler limits add no CLI or environment switch. No-op or reject-all scopes
retain the original volume and empty directional slots without allocating a
dense merged workspace.

The image provider pins the exact detector canvas. It aliases verified canonical
native backing when available; otherwise it materializes only the requested
frame/crop rectangles in a compact immutable cache. An unused lazy processing
cube is not built merely to supply SAM frames. Reduced square canvases retain
their existing affine and stack-resampling semantics.

## Bounded reuse and resource controls

These caches reuse identical inputs and measurements. They do not share tracker
state between independent endpoint sessions or relax proposal quality.

| Control | Default | Scope |
| --- | --- | --- |
| `YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES` | 1 GiB | Each immutable image-demand cache; not a total process budget. Aliased canonical backing adds no cache storage |
| `YOLO_TTA_SAM_RENDER_MAX_BYTES` | 256 MiB | Exact native rendering intermediates |
| `--sam_feature_cache_mib` | `512` MiB | Requested retained feature budget per admitted predictor; zero disables retention |
| Proposal reader `max_cache_bytes` Python argument | 32 MiB | One evidence-reader transaction; zero disables retention |

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

The CPU proposal reader retains immutable raw, filtered, and candidate products
within its transaction. Memory-limited replay and final audits cap retention at
the smaller of 32 MiB or one eighth of their workspace budget. Entry/exit
integrity checks preserve source/payload identity. Cache statistics distinguish
hits, decoding, filtering, evictions, peak bytes, and completed transactions;
retention is released when the owner closes.

Performance receipts measure these paths separately. Local machine results are
sanity checks for the exercised workload, not target-system throughput claims.

## Existing interpolation controls

The flag names retain their purpose across the two generators. Frame distances
and directions refer to the selected working view's native frame index. If
preprocessing changes the stack depth, `--interpolation_distance` counts those
processing frames. Bridge radius is measured in the selected view canvas's
pixels; it is not an input-video pixel measurement after a changed canvas scale.

| Flag | Default | SDF | SAM |
| --- | --- | --- | --- |
| `--interpolation_distance` | `15` | Maximum endpoint search frame distance; zero disables interpolation | Same search bound and disable behavior |
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

The planner bounds each group to 64 endpoints, 512 observations, 128 edges,
128 frames, and 4,194,304 crop pixels. Declared mask contracts are limited to
256 MiB per group and 512 MiB across a plan. Reaching a limit records an
incomplete or unresolved family instead of silently omitting a sibling.
Physical spacing participates in geometric search; crop coordinates remain
view-native pixels.
The planner's group frame cap is not a tracker-session allowance. Each actual
SAM run must fit the runtime's 30-frame maximum, including its seed and terminal
observations. Structural validation enforces that limit before expensive work;
a larger planning inventory does not authorize longer tracking.

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

Integrated scope bundles remain under
`OUTPUT/sam_interpolation/MODEL/VIEW/fullframe` or `tile_CONFIG`. The run manifest
records the configured backend, active state, SAM model, and SAM device pool.
Layer and gate receipts retain directional identities, selected policy hashes,
and parent-support fingerprints, including successfully empty selected support.

Proposal selection occurs before directional publication and before any
source-level plain-union optimization. Detector confidence, tracker score,
endpoint agreement, and family agreement are separate measurements. Later
source reconciliation and global cleanup may remove selected bridge support;
selected-bridge connectivity and final connection survival are separate facts.
The final connection audit currently certifies only a verified identity
Transverse mapping. Resampled scopes report connectivity as unassessed while
still measuring retained/removed bridge voxels; voxel survival alone does not
establish a connected final repair.

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

For whole mode, the stock `sam_conservative_v2` policy first applies `--interpolation_min_radius` to each full
crop-space prediction. It labels 8-connected 2D foreground components and
removes an entire component when its maximum inscribed radius is at or below
the threshold. It does not erode a healthy component's outline, remove its thin
extensions, fill holes, or select only the largest component. A value of zero
disables this filter.

Filtering precedes acceptance-region and write-region clipping. A removed dot
outside either region remains visible in the raw diagnostics but cannot reject
the run or appear in selected output. A larger component that survives the
radius filter still violates containment if it touches or crosses the declared
acceptance boundary; clipping it to the write region cannot hide that violation.
Original detector observations and stored raw tracker masks remain immutable.

The policy evaluates containment, held-out endpoint agreement, family agreement,
local topology and unintended contact using the filtered support. It requires
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

Quality policy version 2 records the exact component filter in each selection
receipt. Online output, directional replay, previews and connection-survival
checks reconstruct the same filtered contributors. Older selection receipts
without a `mask_filter` retain their original unfiltered interpretation;
reselecting their immutable raw evidence with version 2 creates a new receipt.
The proposal callback interface remains `proposal_api_version=1`. Explicit
version-1 quality settings are rejected rather than silently reinterpreted.

For an explicit fixed-evidence quality experiment, a policy may set
`component_min_radius` to a nonnegative value. Its default, `None`, inherits
the recorded `--interpolation_min_radius` for each group. This is a policy
override, recorded in the selection/filter hashes; it does not modify the raw
bundle or add a second command-line radius control. Setting
`enforce_interpolation_min_radius=False` disables filtering, as used by the
raw-candidate ablation.

## Replay and validation boundaries

Fixed-evidence replay changes proposal selection without loading the detector
or SAM model. It requires the same complete, structurally valid proposals,
planning inputs, and gate snapshot. Changes to crop geometry, model,
preprocessing, group inventory, missing frame coverage, or admitted detector
observations require generation again.
Replay resolves crop mode from saved evidence, rather than the current
environment. Legacy bundles without tiled evidence keep whole/v2 interpretation.
Historical whole receipts are not silently reinterpreted as tiled/v3 receipts;
changing crop mode requires generation.

Changing parent bridge selection can change tile admission and therefore the
downstream proposal inventory. A changed upstream gate-support fingerprint
invalidates affected downstream evidence or requires regeneration. A comparison
on the original frozen proposal set must identify that snapshot; it is not a
fresh end-to-end run under a different admission result.

Copy a retained bundle to a fresh portable destination, then compare proposal
policies without loading detector or SAM runtimes:

```text
python tools/export_reconciliation_evidence.py --sam_bundle BUNDLE --output PORTABLE_BUNDLE

python tools/compare_reconciliation.py --sam_bundle PORTABLE_BUNDLE \
  --output FRESH_COMPARISON --memory_mib 256

python tools/compare_reconciliation.py --sam_bundle PORTABLE_BUNDLE \
  --sam_policy stock strict permissive --output FRESH_ABLATION
```

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
| `sam_crop_quality.py` | Compute research diagnostic eligibility; this is separate from stock production v2 acceptance |

Keep crop planning and seed diagnostics independent of withheld annotations.
Native coverage and model-space seed survival are separate measurements:
resizing can change a thin seed even when the native crop contains it. Account
for overlap, edges, padding, and inverse transforms when comparing stitched
tile predictions. Report raw and selected support and failures as well as
accepted repairs. Results are paired experimental evidence; these tools do
not add a production interpolation backend or change production crop defaults.
Raw tracker-strategy outputs must be labelled as such. They do not establish
production v2 proposal acceptance, tile admission, or a new independent accuracy
holdout when scoring reuses an annotation from an earlier experiment.
The quality helper reports its own frozen diagnostic contract, including a
default 16-pixel margin and 512 MiB topology allowance. These differ from the
production contract and its 256 MiB group budget; diagnostic eligibility is
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

Future work includes other SAM view geometries, more general crop/frontier
planning, extracted tile-gating policies, and mixed-backend execution. These
are separate changes to the declared rollout.

LTA's opt-in [dynamic crop backend](lta_dynamic_crops.md) has its own execution
contract. The TTA generator shares a fixed family context across independently
seeded sessions and does not inherit LTA's between-window crop updates.
