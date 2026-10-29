# SAM experiments and historical policy reference

This reference retains research protocols, comparison tools and explicitly
selected legacy policies previously interleaved with the current SAM guide.
Research entrypoints remain useful development tools; their results do not
change production defaults. Use the [current guide](sam_interpolation.md) for
active interpolation, extrapolation, connected-branch selection and evidence
contracts, or the [runtime and SDK guide](sam_sdk.md) for execution ownership.

Preserve original labels, artifact paths, bytes and hashes when interpreting
retained studies. Read the [release identity map](../../Scratch/Data/XTA/History/release/version_migrations.json)
and [interpretation rules](../../Scratch/Data/XTA/History/release/README.md#historical-release-identities)
before resolving old release names. A version label alone does not identify
source, and a dirty/development run is not equivalent to its base release.
Quality-policy versions v2-v7 below are independent of XTA release numbering.

The guarded-rescue cutoff below uses the renamed release label v24.0.2
(original release label v25.0.2). Its source annotation belongs to commit
`cdc702d3eabea7de372cfc17d85802acd0f6e0ee`, as resolved through the identity
map; this does not rename raw study artifacts or establish development-run
equivalence.

## Largest-island and one-direction continuation experiments

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
described in the [current guide](sam_interpolation.md#sam-extrapolation), uses
raw-mask emptiness and the distance horizon alone.

## Legacy quality versions 2 through 5

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

## Historical guarded rescue (explicit whole v4 / tiled v5)

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

## Fixed-evidence improvement studies

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

## Outer-context and acceptance experiments

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
is described in [the historical policy section](#historical-guarded-rescue-explicit-whole-v4--tiled-v5);
the defaults use the [connected-branch policy](sam_interpolation.md#connected-branch-selection-whole-v6--tiled-v7).
