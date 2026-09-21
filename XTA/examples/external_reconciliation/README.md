# External reconciliation policies

Reconciliation combines independently editable source-space component layers
before TTA's global postprocessing. It changes the derived final result, while
retaining the prediction, bridge, and gated-tile layers used to construct it.

```text
python -m XTA --mode tta ... --reconciliation XTA/examples/external_reconciliation/confidence_core_rescue.py
```

Runs with reconciliation also retain prediction-confidence evidence under
`reconciliation_evidence/`. The `union.py` reference is useful when collecting
this evidence while preserving the additive union decision. Omitting
`--reconciliation` retains ordinary TTA behavior.

In TTA, a plain union policy reuses the already assembled source union. It
counts and validates that buffer without reopening component masks or allocating
a second source-sized output. Its manifest marks unmeasured per-layer and
confidence counts as null. Custom decision callbacks still run normally.

`reconciliation/manifest.json` records the policy and source hash, evidence
groups, per-layer contributions, component statistics and weights, retained and
rejected voxel counts, and working-memory plan. `reconciliation/policy.py` is
the executed policy snapshot. These are distinct from final postprocessing.

## Included policies

| File | Decision |
|---|---|
| `union.py` | Exact OR of additive components; reference for checking recomposition |
| `confidence_core_rescue.py` | Preferred confidence preset: coherent cores and bounded rescue through weaker observed support |
| `quorum3.py` | Simple control requiring three independently capped section groups |
| `largest_island.py` | Add bounded island weighting, using two average votes as its threshold |
| `confidence_anchored.py` | Give a bounded local bonus when a strong prediction corroborates another prediction |
| `hybrid_with_fill.py` | Combine component agreement, selective quorum, bounded rescue and local candidate-only fill without confidence scores |

Core rescue is the selected starting point for confidence-based decisions. The
other examples retain distinct approaches for different evidence and tradeoffs;
none is presented as universally accuracy-optimal. Use `union.py` for confidence
collection while retaining the additive union. Confidence-based decisions still
require source-grid score projection, which can be expensive on large views.

The spatial core/rescue and hybrid rules operate independently within each
source XY plane. Their votes include all projected views; they do not apply
cross-slice morphology. Core rescue preserves a strong observed anchor with
corroboration or sufficiently strong independent consensus, extracts coherent
cores, and grows only a bounded distance through observed candidate support.
The hybrid applies a component-agreement gate, stricter support for small
islands, and limited candidate-only local additions. These spatial decisions
preserve the raw candidate boundary and do not fabricate confidence scores.

Files beginning with `_` are packaged implementation helpers. Presets import
them through `XTA`, so a saved `reconciliation/policy.py` snapshot can be loaded
from another directory with the matching package installed.

## Policy contract

Each file defines `build_reconciliation()` and returns a dictionary. The
frontend-independent engine is `XTA.reconciliation`; it accepts immutable TYX
slab readers and an output writer. It imports no inference backend or Slicer
runtime. The same numerical contract can later be used by a Slicer frontend.

Supported fields:

| Field | Meaning |
|---|---|
| `name` | Human-readable policy identity |
| `mode` | `union`, `weighted`, or `confidence` |
| `grouping` | `views` or `sections` |
| `threshold` | Minimum positive support score |
| `min_sources` | Minimum independently capped evidence groups |
| `min_prediction_sources` | Minimum groups containing direct prediction evidence |
| `provenance_weights` | Weights for `prediction`, `bridge`, and `mixed` layers |
| `island_weighting` | Enable exact grouped six-connected component statistics |
| `island_weight_min`, `island_weight_max` | Bounds on the island multiplier |
| `angular_tolerance_deg` | Orientation-bin width used by section grouping |
| `anchor_confidence` | Minimum actual score for strong prediction support |
| `anchor_bonus` | Bounded bonus where at least two independent predictions meet |
| `decide` | Optional callable taking a read-only vote-block dictionary and returning a boolean mask |

A vote block exposes `candidate`, `score`, `support`, `prediction_support`,
`anchored`, and `z0`/`z1`. Decisions must preserve shape and may not introduce
foreground outside the additive candidate union. Input layers are never
modified. Source bytes are compiled directly after hash verification, avoiding
stale bytecode when policy files are edited.

## Confidence and provenance

Retained scores are uint8 maxima of surviving detector instance confidence,
projected with the categorical support addresses and max reduction. They are
not calibrated per-voxel probabilities. Score zero means unknown, including
background, unobserved pixels, and values quantized to zero. Existing cleanup
can create foreground without an observed score; such pixels remain unknown.
The engine does not substitute the run's `--conf` threshold for missing scores.

Confidence collection is independent of `--min_conf`. CPU, GPU, external
augmentation, resident D1, and accepted tile paths keep their existing binary
mask processing and morphology. Policies whose mode is not `confidence` retain
native-view evidence without immediately expanding it into source space.
Compressed numeric blocks and native-piece manifests explicitly distinguish
their storage coordinates from the source grid. D1 retires bounded score shards
before device-buffer release; collection preserves those shards without a
decompress/merge/project/recompress cycle. Accepted tile pieces retain their
parent offsets and max-overlap semantics.

Confidence-mode policies still project scores for their source-grid decision.
New source and native payloads use schema-two blocks; schema-one source sidecars
remain readable. Explicit `reference.source_reader(workspace, ...)` conversion
stages at most one native layer, checks buffer and temporary-disk budgets, and
removes staging afterward. Reading native storage never silently starts that
conversion. Conversion can remain expensive on large views; collection no
longer pays that cost when confidence is not part of the decision.

Prediction, bridge, and mixed-layer weights remain separate. A bridge has no
detector confidence of its own; its configured weight is a provenance prior.
Tile predictions retain the distinction between parent-prediction acceptance
and parent-bridge acceptance. Layers belonging to one evidence group contribute
their maximum at a voxel, so more tiles, TTA copies, or repeated files cannot
create additional votes for that group.
Reconciliation operates on accepted components. Proposals already rejected by
the existing tile gates are not restored by later voting.

The anchored example is deliberately **voxel-local**. It rewards an independent
strong/weak intersection without promoting an entire object because of a
single touching voxel. Whole-slice or whole-object propagation is a different
policy and needs separate validation.

## Section and island grouping

Plane grouping derives orientations from the actual view geometry, including
the source-to-processing temporal scale. Azimuthal orientations use the nearest
recorded azimuth at representative output-voxel centers. Tilted Azimuthal
stack shears retain the same plane family. Overlapping patches and rotations of
the spherical surface share one group; radial patches share their base cylinder.
TTA repetitions and tiles do not independently multiply geometric evidence.

These are orientation/surface correlation caps, not a claim to reconstruct
every voxel's exact source-frame identity. Compact NRRDs may pool several
source samples. Even-sized images place some central planes between integer
slices, and orientation bins have boundaries. Ordinary voxel overlap or a
global Dice score is not used as proof that two views are duplicates.

Island statistics are computed on the union of members within a logical
group and provenance role, preventing duplicate files from changing the
normalization. Log largest-component sizes are normalized around the cohort
median using its interquartile range; multipliers are clipped to the policy's
bounds, normally 0.75–1.25. This limits the influence of a large connected
false-positive mass. The reported largest fraction and component count are
descriptive metrics, not ground-truth quality scores.

## Memory and saved-layer comparison

`--reconciliation_memory_mib` defaults to 4096. Voting uses bounded slabs;
component statistics retain only components touching the active slab frontier.
No full-volume label array or four-dimensional layer stack is built. A budget
too small for an XY plane fails with a required-memory estimate. A conservative
planned-view check runs before model inference, so an insufficient voting
budget does not waste a completed inference run. The ordinary
union reference uses the existing TTA union, while saved-layer recomposition
uses a separate small-memory OR path. The budget controls algorithm workspace,
not all process memory or OS-managed mapped-file cache.

Saved-layer experiments can use:

```text
python tools/compare_reconciliation.py --input PATH_TO_NRRD_MANIFEST --output FRESH_OUTPUT_DIRECTORY --policy XTA/examples/external_reconciliation/quorum3.py XTA/examples/external_reconciliation/hybrid_with_fill.py
```

The reader validates a common reference grid, explicit crops and offsets,
binary values, payload sizes and checksums. It supports concatenated gzip
members and advances each layer's decoder through ordered slabs. Global
checkpoints and audit layers are excluded from evidence. Missing confidence is
reported explicitly. Comparing retained/rejected voxels measures a policy's
effect; it does not measure false-positive accuracy without reference labels.

The comparison tool keeps only the selected preview planes after each output
is published. It removes that policy's temporary raw volume before starting
the next policy, so temporary disk does not grow by one full volume per policy.

To prepare matched compact masks and confidence from a completed run:

```text
python tools/export_reconciliation_evidence.py --run_manifest RUN/manifest.json --compact_manifest RUN/low_quality/BIN/nrrd/LAYER_MANIFEST.json --output FRESH_BUNDLE
```

This reads source-aligned confidence crops and the existing compact masks;
native NRRD masks are not opened. It retains original geometry/provenance,
uses numeric maximum over the source-to-compact footprints, and intersects
scores with the compact mask. Zero remains unknown. The result includes portable
run/layer manifests and can be passed to `tools/compare_reconciliation.py`.

For deferred native evidence, add `--allow_native_projection`. That explicitly
converts one view at a time using a checked staging limit; `--max_staging_mib`
defaults to 32768 and includes additional projection maps. Some native tilted
views require more than that. This opt-in conversion is separate from the
collection run and from legacy source-aligned compact export.

Confidence publication identifies native retention or source projection in the
log. Long block writes and the final pending-evidence drain report progress every
30 seconds, so remaining confidence work is visible after mask inference ends.

The first integration is TTA. The supplied Slicer startup script is unchanged;
its existing surgical-edit operations remain available while this policy
interface is validated for later frontend integration.
