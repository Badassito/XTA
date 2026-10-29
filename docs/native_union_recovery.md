# Native recovery from a mixed-source preview union

`tools/xta_native_filter.py` transfers deletion evidence to original native
components. It does not upscale the edited preview into a native segmentation.
The ordinary single-source arguments remain available.

This mode preserves original native prediction geometry. It is not an exact
reconstruction of the edited preview's shape. When the edit is authoritative
for boundaries and separation, use
[edited-shape reconstruction](edited_shape_reconstruction.md) instead.

An exact-name union from several confidence runs requires explicit additional
provenance. The manifest copied into a Slicer edit bundle can identify only its
primary run; it does not identify every contributor to a later union. Use:

```text
python -B tools/xta_native_filter.py --auto --validate-only
  --edits EDITED_BUNDLE --low-manifest PRIMARY_LOW_MANIFEST
  --full-manifest PRIMARY_FULL_MANIFEST --union-receipt UNION_MANIFEST
  --union-source-pairs SOURCE_SPEC --output NEW_PREFLIGHT_DIRECTORY
```

The multiline example lists arguments; use the continuation syntax of the
selected shell. Remove `--validate-only` and provide `--scratch` for native
processing. Preflight validates preview evidence and source headers without
allocating native working rasters. Full processing additionally validates native
sampling support and records the actual source-file hashes.

The source specification has format `xta-native-union-sources-v1`, an explicit
`identification_basis`, optional expected `union_receipt_sha256`, and at least
two `pairs`. Each pair declares a unique label, original low/full manifest paths,
their expected SHA256 values, and `receipt_member_indices`. These integers are
the receipt members' `segment_index` values, not their positions after flattening
the groups. They must uniquely partition the receipt; each assigned exact-name
inventory must equal the corresponding low/full source inventories. Receipts
with duplicate indices across nodes are rejected rather than guessed.

The tool does not infer confidence from Slicer IDs or array order. An
inventory-based reconstruction is recorded as such, not as cryptographic ancestry
of unedited source masks. Relative paths resolve against the specification file.

For each exact-name component, original Global clipping is applied separately
within each source run, then the sources are unioned before native component
selection and automatic cutting. The legacy empty-original-Global exception is
preserved and disclosed per source. Clipping the combined sources to a combined
Global would produce different results and is not used. The source streams are
bounded by slabs, while native selection retains its sparse foreground-run
storage and final largest-component policy.

Native processing decodes and validates each source's original Global once,
then stores it as a separate bit-packed scratch file. Workers map these files
read-only, clip and union component contributions in packed form, then unpack
the resulting slab once. Empty source slabs still undergo integrity validation
but skip packed clipping and union work. The combined initial Global uses
the same caches; each component still uses its own source's Global before the
union. Cache storage is `sources * T * Y * ceil(X / 8)` bytes in addition to the
existing working rasters, and is included in the scratch-space preflight. The
caches are temporary and removed when processing exits. Preview-only validation
does not build them. Raw source support, binary-value and gzip integrity checks
remain active, including for empty Globals.

The reader uses ISA-L when the optional `isal` package is installed (also included
in the project's `acceleration` extra). Install it into the preparation job's
Python environment with `python -m pip install "isal>=1.8,<2"`. The report and
startup log identify the selected gzip backend. `XTA_NATIVE_GZIP_BACKEND=isal`
requires the accelerated reader and fails clearly if it is missing;
`XTA_NATIVE_GZIP_BACKEND=stdlib` selects the reference reader. The default is
`auto`. A corrupt stream is never retried using another decoder. Each component
NRRD still requires a complete source read; caching Globals does not change
native recovery into preview upscaling.

Every exported nonglobal mask must match its union-receipt payload, including
coordinates and counts. To apply Update Both after the union receipt was made,
archive the original union-reference edit bundle and specify:

```json
{
  "union_reference_edits": "union-reference-edits",
  "union_reference_edits_sha256": "EXPECTED_SHA256_OF_ITS_EDITS_JSON"
}
```

These fields extend the source specification. The baseline snapshots must match
the receipt exactly, and new nonglobal masks must equal **baseline AND corrected
Global**. The corrected Global must equal the nonglobal union and satisfy the
existing face-connectivity requirement. Hash checks, raw source support checks,
and deletion-only restrictions remain active. This does not authorize arbitrary
edits or restoring previously removed foreground.

`tools/prepare_pta_input.py` can publish the resulting native mask/intensity pair
using `--mode native`, paired `--native-union-receipt` and
`--native-union-source-pairs`, and the ordinary primary manifest arguments.
`--source-video` selects an explicit original video; `--intensity-mode copy`
publishes a SHA256-verified copy instead of a symlink. Existing inputs are reused
only when their complete preparation identity and output checks match. They are
never overwritten. Failed native diagnostics are retained for review.

The native auto policy is a geometry/evidence heuristic. Unresolved keep/remove
conflicts now stop `--auto` with `needs_review`, exit code 2, and no published PTA
mask. Diagnostics and accepted-cut counts remain available. Only explicit
`--conflicts keep` permits the conservative result with unresolved components;
that output is not a faithful transfer of edited shape. The preparation wrapper
also rejects unresolved or unsuccessful native reports even if a legacy helper
exits zero. Inspect successful heuristic cuts before accepting a native result
as training ground truth. Preview-only success is not full native validation.
